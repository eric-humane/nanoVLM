# Models

- `vision_transformer.py`: Timm-only vision loader (`timm.create_model`, including `hf-hub:` repos). Patch size, image size, hidden dim, and depth auto-derive from the backbone; `hydrate_vision_cfg_from_timm` syncs config before dataloaders. Drops CLS if needed to keep patch grids square.
- `modality_projector.py`: Pixel-shuffle + linear projection from vision tokens to the LM hidden size. Set `mp_pixel_shuffle_factor=1` to bypass pixel shuffle for irregular/non-square grids.
- `vision_language_model.py`: Uses HF `AutoModelForCausalLM` end-to-end; tokenizer/chat template derive from the LM. Vision encoder + modality projector feed image tokens directly into the LM embeddings; supports `save_pretrained`/`from_pretrained` round-trips with a single `model.pt`.
