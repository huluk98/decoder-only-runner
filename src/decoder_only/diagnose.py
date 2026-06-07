from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from decoder_only.loader import _looks_like_custom_model, _looks_like_hf_model, load_model_and_tokenizer


KEY_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "model.pt",
    "checkpoint.pt",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose a local decoder checkpoint layout.")
    parser.add_argument("model_path")
    parser.add_argument("--model-kind", choices=("auto", "hf", "custom"), default=None)
    parser.add_argument("--local-only", choices=("0", "1"), default="1")
    parser.add_argument("--load-model", action="store_true", help="Also load model weights on CPU.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_path = Path(args.model_path).expanduser().resolve()
    model_kind = args.model_kind or os.environ.get("DECODER_ONLY_MODEL_KIND", "auto")
    os.environ["DECODER_ONLY_MODEL_KIND"] = model_kind
    if args.local_only == "1":
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print(f"checkpoint: {model_path}")
    print(f"requested_model_kind: {model_kind}")
    print(f"local_only: {args.local_only}")

    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {model_path}")
    if not model_path.is_dir():
        raise NotADirectoryError(f"Checkpoint path must be a directory: {model_path}")

    print("\nkey files:")
    for name in KEY_FILES:
        path = model_path / name
        print(f"  {name}: {'yes' if path.exists() else 'no'}")

    config = read_json(model_path / "config.json")
    if config:
        print("\nconfig:")
        for key in (
            "model_type",
            "architectures",
            "vocab_size",
            "max_position_embeddings",
            "block_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
        ):
            if key in config:
                print(f"  {key}: {config[key]}")

    looks_hf = _looks_like_hf_model(model_path)
    looks_custom = _looks_like_custom_model(model_path)
    print("\ndetection:")
    print(f"  looks_like_hf: {looks_hf}")
    print(f"  looks_like_custom: {looks_custom}")

    if model_kind == "hf" or (model_kind == "auto" and looks_hf and not looks_custom):
        diagnose_hf_tokenizer(model_path)
    elif model_kind == "custom" or looks_custom:
        print("\ncustom checkpoint layout selected.")
    else:
        raise RuntimeError(
            "Could not identify this checkpoint. For Decoder-Chinese-SLM, the folder should contain "
            "config.json, tokenizer.json, tokenizer_config.json, special_tokens_map.json, and "
            "model.safetensors or pytorch_model.bin."
        )

    if args.load_model:
        print("\nloading model on CPU...")
        kind, model, tokenizer, _device = load_model_and_tokenizer(model_path, device="cpu")
        print(f"  loaded_kind: {kind}")
        print(f"  model_class: {model.__class__.__name__}")
        print(f"  tokenizer_class: {tokenizer.__class__.__name__}")
        print("  status: ok")
    else:
        print("\nmodel weight load: skipped; use --load-model to test CPU load")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected object JSON in: {path}")
    return value


def diagnose_hf_tokenizer(model_path: Path) -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for Hugging Face checkpoint loading.") from exc

    print("\nhuggingface tokenizer:")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    print(f"  class: {tokenizer.__class__.__name__}")
    print(f"  vocab_size: {len(tokenizer)}")
    print(f"  eos: {tokenizer.eos_token!r} id={tokenizer.eos_token_id}")
    print(f"  pad: {tokenizer.pad_token!r} id={tokenizer.pad_token_id}")
    if tokenizer.pad_token_id is None:
        raise RuntimeError("Tokenizer has no pad token. Re-run the PreTrainedTokenizerFast repair snippet.")


if __name__ == "__main__":
    main()
