import torch
import unittest
from models.vision_language_model import VisionLanguageModel
from models.config import VLMConfig # Assuming VLMConfig is in models.config
from types import SimpleNamespace

class TestVisionLanguageModel(unittest.TestCase):
    def setUp(self):
        # Minimal config for testing VLM
        self.cfg = VLMConfig(
            vit_model_type="hf-internal-testing/tiny-random-clip",
            vit_patch_size=2,
            vit_img_size=30,
            max_img_size=30,
            lm_model_type="hf-internal-testing/tiny-random-gpt2",
            mp_pixel_shuffle_factor=1,
        )

        self.model = VisionLanguageModel(self.cfg, load_backbone=True) # Load tiny pretrained backbones
        self.model.eval() # Set model to evaluation mode

    def test_generate_kv_caching_consistency(self):
        batch_size = 16
        prompt_seq_len = 32
        max_new_tokens = 16 # Generate a few tokens

        # Dummy image (Batch, Channels, Height, Width)
        image_input = torch.randn(batch_size, 3, self.cfg.vit_img_size, self.cfg.vit_img_size)
        # Dummy prompt input_ids
        vocab_size = len(self.model.tokenizer)
        prompt_ids = torch.randint(0, vocab_size, (batch_size, prompt_seq_len))

        # Generation with KV caching (default)
        generated_ids_with_cache = self.model.generate(
            prompt_ids,
            image_input,
            max_new_tokens=max_new_tokens,
            use_kv_cache=True,
            greedy=True # Use greedy for deterministic output
        )

        # Generation without KV caching
        generated_ids_without_cache = self.model.generate(
            prompt_ids,
            image_input,
            max_new_tokens=max_new_tokens,
            use_kv_cache=False,
            greedy=True # Use greedy for deterministic output
        )
        
        self.assertTrue(
            torch.equal(generated_ids_with_cache, generated_ids_without_cache),
            f"Generated token IDs with and without KV caching do not match.\nWith cache: {generated_ids_with_cache}\nWithout cache: {generated_ids_without_cache}"
        )

if __name__ == '__main__':
    unittest.main() 
