import os
import torch
import unittest

from models.config import VLMConfig
from models.vision_language_model import VisionLanguageModel


@unittest.skipUnless(os.getenv("RUN_HEAVY_INTEGRATION") == "1", "Set RUN_HEAVY_INTEGRATION=1 to run heavy backbone test.")
class TestFunctionGemmaSigLIP2(unittest.TestCase):
    def test_forward_functiongemma_siglip2(self):
        cfg = VLMConfig(
            vit_model_type="vit_so400m_patch16_siglip_384",
            vit_pretrained=False,
            lm_model_type="google/functiongemma-270m-it",
        )

        model = VisionLanguageModel(cfg, load_backbone=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.train()

        batch_size = 1
        seq_len = 8
        vocab_size = len(model.tokenizer)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        attention_mask = torch.ones_like(input_ids, device=device)
        images = torch.randn(batch_size, 3, model.cfg.vit_img_size, model.cfg.vit_img_size, device=device)

        logits, loss = model(input_ids, images, attention_mask=attention_mask, targets=input_ids)
        self.assertIsNotNone(loss)
        loss.backward()
        self.assertEqual(logits.shape[:2], (batch_size, seq_len))


if __name__ == "__main__":
    unittest.main()
