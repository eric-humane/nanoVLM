# Models

- `vision_transformer.py`: Loads SigLIP or CLIP/ViT weights into the nano ViT when available; otherwise wraps HF `AutoModel`/`CLIPVisionModel` as a vision encoder. Pulls patch/image size and hidden dim from the HF vision config.
- `modality_projector.py`: Pixel-shuffle + linear projection from vision features to the LM hidden size. Set `mp_pixel_shuffle_factor=1` to bypass pixel shuffle for irregular grids.
- `vision_language_model.py`: HF `AutoModelForCausalLM` for text, the vision encoder above, and the modality projector to bridge them. Embeddings are resized to include extra tokens. Decoder-only LMs only.
