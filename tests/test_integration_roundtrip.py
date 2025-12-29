import os
import shutil
import tempfile
import torch
import unittest

from models.config import VLMConfig
from models.vision_language_model import VisionLanguageModel


class TestIntegrationRoundtrip(unittest.TestCase):
    def test_forward_backward_save_load(self):
        cfg = VLMConfig(
            vit_model_type="vit_tiny_patch16_224",
            vit_pretrained=False,
            mp_pixel_shuffle_factor=1,
            lm_model_type="hf-internal-testing/tiny-random-gpt2",
        )

        model = VisionLanguageModel(cfg, load_backbone=True)
        model.train()
        vision_size = model.cfg.vit_img_size

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        batch_size = 2
        seq_len = 8
        vocab_size = len(model.tokenizer)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        attention_mask = torch.ones_like(input_ids, device=device)
        # Dummy image batch matching vision config
        images = torch.randn(batch_size, 3, vision_size, vision_size, device=device)

        logits, loss = model(input_ids, images, attention_mask=attention_mask, targets=input_ids)
        self.assertIsNotNone(loss)
        loss.backward()

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            reloaded = VisionLanguageModel.from_pretrained(tmpdir)
            reloaded.to(device)
            reloaded.eval()
            with torch.no_grad():
                logits2, _ = reloaded(input_ids, images, attention_mask=attention_mask)
            self.assertEqual(logits.shape, logits2.shape)


if __name__ == "__main__":
    unittest.main()
