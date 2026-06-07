from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from decoder_only.loader import load_model_and_tokenizer


class HfLocalLoaderTest(unittest.TestCase):
    def test_hf_loader_uses_local_fast_tokenizer_without_remote_code(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            (tmpdir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "llama",
                        "architectures": ["LlamaForCausalLM"],
                        "vocab_size": 16,
                    }
                ),
                encoding="utf-8",
            )
            (tmpdir / "tokenizer.json").write_text("{}", encoding="utf-8")
            (tmpdir / "tokenizer_config.json").write_text(
                json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"}),
                encoding="utf-8",
            )
            (tmpdir / "model.safetensors").write_bytes(b"placeholder")

            tokenizer = MagicMock()
            tokenizer.pad_token_id = 0
            model = torch.nn.Linear(1, 1)
            model.eval = MagicMock(return_value=model)

            with (
                patch.dict(os.environ, {"DECODER_ONLY_MODEL_KIND": "hf"}, clear=False),
                patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer) as load_tok,
                patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model) as load_model,
            ):
                kind, loaded_model, loaded_tokenizer, _device = load_model_and_tokenizer(
                    tmpdir,
                    device="cpu",
                )

            self.assertEqual(kind, "hf")
            self.assertIs(loaded_model, model)
            self.assertIs(loaded_tokenizer, tokenizer)
            load_tok.assert_called_once()
            self.assertEqual(load_tok.call_args.kwargs["local_files_only"], True)
            self.assertEqual(load_tok.call_args.kwargs["trust_remote_code"], False)
            self.assertEqual(load_tok.call_args.kwargs["use_fast"], True)
            load_model.assert_called_once()
            self.assertEqual(load_model.call_args.kwargs["local_files_only"], True)
            self.assertEqual(load_model.call_args.kwargs["trust_remote_code"], False)


if __name__ == "__main__":
    unittest.main()
