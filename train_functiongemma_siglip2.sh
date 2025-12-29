#!/bin/bash
set -euo pipefail

# Example single-node launch for functiongemma + SigLIP2-SO400M.
# Adjust batch size / accumulation to fit your GPU memory.
python train.py \
  --lm_model_type google/functiongemma-270m-it \
  --vit_model_type google/siglip2-so400m-patch16-384 \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --lr_language_backbone 5e-5 \
  --lr_vision_backbone 5e-5 \
  --max_training_steps 20000
