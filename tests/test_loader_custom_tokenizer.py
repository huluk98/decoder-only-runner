from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch

from decoder_only.custom_model import DecoderConfig, DecoderOnlyTransformer
from decoder_only.loader import load_model_and_tokenizer


class CustomTokenizerLoaderTest(unittest.TestCase):
    def test_custom_decoder_preferred_with_fast_tokenizer_files(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            config = DecoderConfig(
                vocab_size=16,
                block_size=8,
                n_layer=1,
                n_head=2,
                n_embd=8,
            )
            config_payload = asdict(config)
            config_payload.update(
                {
                    "model_type": "llama",
                    "architectures": ["LlamaForCausalLM"],
                }
            )
            (tmpdir / "config.json").write_text(
                json.dumps(config_payload, indent=2),
                encoding="utf-8",
            )
            model = DecoderOnlyTransformer(config)
            torch.save(
                {"model_state_dict": model.state_dict(), "config": asdict(config)},
                tmpdir / "pytorch_model.bin",
            )
            self._write_fast_tokenizer(tmpdir)

            with patch.dict(os.environ, {"DECODER_ONLY_MODEL_KIND": "auto"}):
                kind, loaded_model, tokenizer, _device = load_model_and_tokenizer(tmpdir, device="cpu")

            self.assertEqual(kind, "custom")
            self.assertIsInstance(loaded_model, DecoderOnlyTransformer)
            self.assertTrue(hasattr(tokenizer, "save_pretrained"))
            self.assertEqual(tokenizer.eos_token, "<|eos|>")
            self.assertEqual(tokenizer.pad_token, "<|pad|>")

    @staticmethod
    def _write_fast_tokenizer(path: Path) -> None:
        try:
            from tokenizers import Tokenizer
            from tokenizers.models import WordLevel
            from tokenizers.pre_tokenizers import Whitespace
        except ImportError as exc:
            raise unittest.SkipTest("tokenizers is not installed") from exc

        vocab = {
            "<|pad|>": 0,
            "<|unk|>": 1,
            "<|bos|>": 2,
            "<|eos|>": 3,
            "<|user|>": 4,
            "<|assistant|>": 5,
            "<|system|>": 6,
            "打开": 7,
            "关闭": 8,
            "客厅": 9,
            "灯": 10,
            "。": 11,
            "好的": 12,
            "已": 13,
            "空调": 14,
            "电视": 15,
        }
        tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<|unk|>"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.save(str(path / "tokenizer.json"))
        tokenizer_config = {
            "pad_token": "<|pad|>",
            "unk_token": "<|unk|>",
            "bos_token": "<|bos|>",
            "eos_token": "<|eos|>",
            "additional_special_tokens": ["<|user|>", "<|assistant|>", "<|system|>"],
            "model_max_length": 2048,
        }
        (path / "tokenizer_config.json").write_text(
            json.dumps(tokenizer_config, indent=2),
            encoding="utf-8",
        )
        (path / "special_tokens_map.json").write_text(
            json.dumps(tokenizer_config, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
