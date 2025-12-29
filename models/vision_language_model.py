import json
import os
import tempfile
from dataclasses import asdict
from typing import Optional


from models.utils import top_k_top_p_filtering
from models.vision_transformer import ViT
from models.modality_projector import ModalityProjector
from models.config import VLMConfig

from data.processors import get_tokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model, save_model
from transformers import AutoModelForCausalLM, AutoConfig

class VisionLanguageModel(nn.Module):
    def __init__(self, cfg: VLMConfig, load_backbone=True):
        super().__init__()
        self.cfg = cfg
        if self.cfg.lm_tokenizer is None:
            self.cfg.lm_tokenizer = self.cfg.lm_model_type
        self.tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template)
        if self.cfg.lm_chat_template is None and getattr(self.tokenizer, "chat_template", None):
            self.cfg.lm_chat_template = self.tokenizer.chat_template
        if load_backbone:
            print("Loading from backbone weights")
            self.vision_encoder = ViT.from_pretrained(cfg, pretrained=self.cfg.vit_pretrained)
            self.lm = AutoModelForCausalLM.from_pretrained(cfg.lm_model_type)
        else:
            self.vision_encoder = ViT.from_pretrained(cfg, pretrained=False)
            lm_config = AutoConfig.from_pretrained(cfg.lm_model_type)
            self.lm = AutoModelForCausalLM.from_config(lm_config)
        if getattr(self.cfg, "max_img_size", None) in (None, 2048, -1):
            self.cfg.max_img_size = self.cfg.vit_img_size
        cfg.lm_hidden_dim = self.lm.config.hidden_size
        # Update vision hidden dim from backbone config when available (important for adapters)
        if hasattr(self.vision_encoder, "backbone") and hasattr(self.vision_encoder.backbone, "config"):
            cfg.vit_hidden_dim = getattr(self.vision_encoder.backbone.config, "hidden_size", cfg.vit_hidden_dim)
        # Resize LM embeddings to include added tokens
        self.lm.resize_token_embeddings(len(self.tokenizer))
        if self.lm.config.is_encoder_decoder:
            raise ValueError("Encoder-decoder models are not supported; please choose a decoder-only causal LM.")
        self.MP = ModalityProjector(cfg)
        self.load_backbone = load_backbone

    def _replace_img_tokens_with_embd(self, input_ids, token_embd, image_embd):
        """
        Replace every image-token placeholder in `input_ids` with the corresponding slice
        from `image_embd`. Supports an arbitrary number of image-token placeholders per sample.
        The first example in the batch might have 2 images and the second none.
        """
        # Clone the original embeddings to avoid in-place issues
        updated_token_embd = token_embd.clone()

        # Build a mask of all image-token positions: shape [B, T_seq]
        mask = (input_ids == self.tokenizer.image_token_id)
        if mask.sum() == 0:
            return updated_token_embd

        flat_image_tokens = image_embd.view(-1, image_embd.size(-1)).to(updated_token_embd.dtype)
        if flat_image_tokens.size(0) != mask.sum():
            # If counts don't line up (e.g., random ids in tests), skip replacement for safety
            return updated_token_embd
        updated_token_embd[mask] = image_embd.view(-1, image_embd.size(-1)).to(updated_token_embd.dtype) # torch flattens before assigning

        return updated_token_embd

    def _process_images(self, images, device):
        if isinstance(images, list):
            if images and isinstance(images[0], list):
                images = [img for sublist in images for img in sublist]

            if not images:  # Handle cases with no images
                return None
            else:
                return torch.cat(images, dim=0).to(device)
        return images # Already a tensor

    def forward(self, input_ids, images, attention_mask=None, targets=None):
        images_tensor = self._process_images(images, input_ids.device)
        token_embd = self.lm.get_input_embeddings()(input_ids) # [B, T_sequence, D_lm]

        if images_tensor is not None:
            image_embd = self.vision_encoder(images_tensor)
            if image_embd.size(-1) != self.cfg.vit_hidden_dim:
                raise ValueError(f"Vision encoder hidden dim {image_embd.size(-1)} does not match cfg.vit_hidden_dim {self.cfg.vit_hidden_dim}.")
            image_embd = self.MP(image_embd)  # [num_images, mp_image_token_length, D_lm]
            token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)

        outputs = self.lm(
            inputs_embeds=token_embd,
            attention_mask=attention_mask,
            labels=targets
        )

        logits = outputs.logits
        loss = outputs.loss if targets is not None else None

        return logits, loss

    @torch.inference_mode()
    def generate(self, input_ids, images, attention_mask=None, max_new_tokens=5, top_k=50, top_p=0.9, temperature=0.5, greedy=False, use_kv_cache=True):
        images_tensor = self._process_images(images, input_ids.device)
        token_embd = self.lm.get_input_embeddings()(input_ids) # [B, T_prompt_text, D_lm]
        full_embeds = token_embd

        if images_tensor is not None:
            # 1. Process image if present
            image_embd = self.vision_encoder(images_tensor) # [B, T_img_feat, D_model]
            if image_embd.size(-1) != self.cfg.vit_hidden_dim:
                raise ValueError(f"Vision encoder hidden dim {image_embd.size(-1)} does not match cfg.vit_hidden_dim {self.cfg.vit_hidden_dim}.")
            image_embd = self.MP(image_embd)      # [B, mp_image_token_length, D_lm]
            # 2. Combine image and text embeddings
            token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)
            full_embeds = token_embd

        current_total_seq_len = full_embeds.size(1)
        batch_size = input_ids.size(0)
        
        # --- Multimodal Prefill Phase ---
        kv_cache_list = None
        prefill_output = self.lm(
            inputs_embeds=full_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=None
        )
        
        current_logits = prefill_output.logits[:, -1, :]
        kv_cache_list = prefill_output.past_key_values

        # Store newly generated token IDs
        newly_generated_ids_list = []

        # --- Decode Phase by sampling tokens autoregressively using the kv-cache ---
        for _ in range(max_new_tokens):
            if greedy:
                next_token_id = torch.argmax(current_logits, dim=-1, keepdim=True)
            else:
                filtered_logits = top_k_top_p_filtering(current_logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(filtered_logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            
            newly_generated_ids_list.append(next_token_id)
            
            # Embed the newly generated token
            next_token_embed = self.lm.get_input_embeddings()(next_token_id) # [B, 1, D_lm]
            
            # update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat((attention_mask, torch.ones((batch_size, 1), device=attention_mask.device, dtype=attention_mask.dtype)), dim=1)

            if use_kv_cache:
                current_total_seq_len += 1

                decode_step_output = self.lm(
                    inputs_embeds=next_token_embed,
                    attention_mask=attention_mask,
                    past_key_values=kv_cache_list,
                    use_cache=True
                )
                kv_cache_list = decode_step_output.past_key_values
                last_token_output = decode_step_output.logits[:, -1, :] 
            else:
                full_embeds = torch.cat((full_embeds, next_token_embed), dim=1)
                decode_step_output = self.lm(
                    inputs_embeds=full_embeds,
                    attention_mask=attention_mask,
                    use_cache=False
                )
                last_token_output = decode_step_output.logits[:, -1, :]
                current_total_seq_len = full_embeds.size(1)

            current_logits = last_token_output
        
        if not newly_generated_ids_list: # Handle case where max_new_tokens might be 0
            return torch.empty((batch_size,0), dtype=torch.long, device=input_ids.device)

        generated_ids = torch.cat(newly_generated_ids_list, dim=1)

        # Post-process to handle EOS token.
        if self.tokenizer.eos_token_id is not None and generated_ids.numel() > 0: # Ensure generated_ids is not empty
            seq_len = generated_ids.size(1)
            device = generated_ids.device

            eos_mask = (generated_ids == self.tokenizer.eos_token_id) # Create a boolean mask for EOS tokens

            col_indices_for_min = torch.arange(seq_len, device=device) # Create column indices [0, 1, ..., seq_len-1]
            
            # In eos_mask, mark positions with actual col_idx, others with a large number
            masked_col_indices = torch.where(eos_mask, col_indices_for_min.unsqueeze(0).expand_as(generated_ids), seq_len + 1) 

            first_eos_indices_values = torch.min(masked_col_indices, dim=1).values
            
            # Clamp values to seq_len (if no EOS found, min will be seq_len + 1, clamp brings it to seq_len0. This means if no EOS, or EOS is the last token, no replacement will happen for that sample.
            actual_first_eos_indices = torch.clamp(first_eos_indices_values, max=seq_len)

            # Create column indices for comparison, shape [batch_size, seq_len]
            col_indices_for_comparison = torch.arange(seq_len, device=device).unsqueeze(0).expand_as(generated_ids)
            
            # Tokens are replaced if their column index is greater than the index of the first EOS token
            replace_mask = col_indices_for_comparison > actual_first_eos_indices.unsqueeze(1)
            
            generated_ids[replace_mask] = self.tokenizer.eos_token_id
        
        return generated_ids

    @classmethod
    def from_pretrained(
        cls, repo_id_or_path: str, *, revision: Optional[str] = None
    ) -> "VisionLanguageModel":
        """
        Load a VisionLanguageModel from a local directory or a repo on the Hugging Face Hub.

        Args:
            repo_id_or_path (str): The path to the local directory or the Hugging Face Hub repo ID.

        Returns:
            VisionLanguageModel: The loaded model.
        """
        # If local folder exists => load from there
        if os.path.exists(repo_id_or_path):
            config_path = os.path.join(repo_id_or_path, "config.json")
            weights_path = os.path.join(repo_id_or_path, "model.pt")
            if not os.path.exists(config_path):
                raise ValueError(
                    f"Config file not found at {config_path}. Please provide a valid path."
                )
            # Weights optional for adapter-only loads; warn if missing
            if not os.path.exists(weights_path):
                print(f"Warning: weights file not found at {weights_path}, loading config only.")
                weights_path = None
        # Otherwise, assume it's a Hugging Face Hub repo
        else:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="config.json", revision=revision
            )
            try:
                weights_path = hf_hub_download(
                    repo_id=repo_id_or_path, filename="model.pt", revision=revision
                )
            except Exception:
                print("Warning: model.pt not found on Hub; loading config only.")
                weights_path = None

        # Load config
        with open(config_path, "r") as f:
            cfg = VLMConfig(**json.load(f))

        # Initialize model without loading the backbone
        model = cls(cfg, load_backbone=False)

        # Load weights
        if weights_path:
            state = torch.load(weights_path, map_location="cpu")
            if "vision_encoder" in state and hasattr(model.vision_encoder, "load_state_dict"):
                model.vision_encoder.load_state_dict(state["vision_encoder"], strict=False)
            if "MP" in state:
                model.MP.load_state_dict(state["MP"], strict=False)
            if "lm" in state:
                model.lm.load_state_dict(state["lm"], strict=False)

        # Done!
        return model

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the model and configuration to a directory.

        Args:
            save_directory (str): The directory to save the model and config.
        """
        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)

        # Save config
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            f.write(json.dumps(asdict(self.cfg), indent=4))

        # Save weights: vision encoder (ours or adapter), MP, and LM
        torch.save({
            "vision_encoder": self.vision_encoder.state_dict(),
            "MP": self.MP.state_dict(),
            "lm": self.lm.state_dict()
        }, os.path.join(save_directory, "model.pt"))

    def push_to_hub(self, repo_id: str, private: bool = False) -> None:
        """
        Push the model and configuration to the Hugging Face Hub.

        Args:
            repo_id (str): The repo ID on the Hugging Face Hub.
        """
        from huggingface_hub import create_repo, upload_folder

        # Create repo
        repo_url = create_repo(repo_id=repo_id, private=private, exist_ok=True)
        repo_id = repo_url.repo_id
        print("Created repo: ", repo_url)

        with tempfile.TemporaryDirectory() as save_path:
            # Save to tmp directory
            self.save_pretrained(save_path)

            # Save model card
            with open(os.path.join(save_path, "README.md"), "w") as f:
                f.write(MODEL_CARD_TEMPLATE.format(repo_id=repo_id))

            # Upload
            return upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=save_path,
                commit_message="Upload nanoVLM using push_to_hub",
            )


MODEL_CARD_TEMPLATE = """
---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
library_name: nanovlm
license: mit
pipeline_tag: image-text-to-text
tags:
  - vision-language
  - multimodal
  - research
---

**nanoVLM** is a minimal and lightweight Vision-Language Model (VLM) designed for efficient training and experimentation. Built using pure PyTorch, the entire model architecture and training logic fits within ~750 lines of code. It combines a ViT-based image encoder (SigLIP-B/16-224-85M) with a lightweight causal language model (SmolLM2-135M), resulting in a compact 222M parameter model.

For more information, check out the base model on https://huggingface.co/lusxvr/nanoVLM-222M.

**Usage:**

Clone the nanoVLM repository: https://github.com/huggingface/nanoVLM.
Follow the install instructions and run the following code:

```python
from models.vision_language_model import VisionLanguageModel

model = VisionLanguageModel.from_pretrained("{repo_id}")
```
"""
