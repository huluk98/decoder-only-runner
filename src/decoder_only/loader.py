from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from decoder_only.custom_model import DecoderConfig, DecoderOnlyTransformer
from decoder_only.tokenizers import load_custom_tokenizer


def resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _looks_like_hf_model(model_path: Path) -> bool:
    tokenizer_files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "sentencepiece.bpe.model",
    )
    weight_files = (
        "model.safetensors",
        "pytorch_model.bin",
        "tf_model.h5",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    return (
        (model_path / "config.json").exists()
        and any((model_path / name).exists() for name in tokenizer_files)
        and any((model_path / name).exists() for name in weight_files)
    )


def _load_hf_model(model_path: Path, device: torch.device) -> tuple[str, Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for Hugging Face checkpoints. "
            "Install the environment from environment.yml or requirements.txt."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    model.to(device)
    model.eval()
    return "hf", model, tokenizer


def _find_custom_checkpoint(model_path: Path) -> Path:
    for name in ("model.pt", "checkpoint.pt", "model.pth", "checkpoint.pth", "pytorch_model.bin"):
        candidate = model_path / name
        if candidate.exists():
            return candidate

    candidates = sorted(model_path.glob("*.pt")) + sorted(model_path.glob("*.pth"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"No custom checkpoint found in {model_path}. Expected model.pt, checkpoint.pt, or *.pth."
    )


def _load_config(model_path: Path, checkpoint_data: dict[str, Any] | None) -> DecoderConfig:
    config_data: dict[str, Any] = {}
    for filename in ("config.json", "decoder_config.json"):
        config_path = model_path / filename
        if config_path.exists():
            config_data.update(json.loads(config_path.read_text()))
            break

    if checkpoint_data:
        raw_checkpoint_config = checkpoint_data.get("config") or checkpoint_data.get("model_config")
        if isinstance(raw_checkpoint_config, dict):
            config_data.update(raw_checkpoint_config)

    aliases = {
        "max_seq_len": "block_size",
        "context_length": "block_size",
        "num_layers": "n_layer",
        "num_heads": "n_head",
        "hidden_size": "n_embd",
        "embedding_dim": "n_embd",
    }
    for source, target in aliases.items():
        if source in config_data and target not in config_data:
            config_data[target] = config_data[source]

    allowed = {field.name for field in fields(DecoderConfig)}
    filtered = {key: value for key, value in config_data.items() if key in allowed}
    missing = [
        key
        for key in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd")
        if key not in filtered
    ]
    if missing:
        raise ValueError(
            "Missing required custom config fields: "
            + ", ".join(missing)
            + ". Add them to config.json next to the checkpoint."
        )

    return DecoderConfig(**filtered)


def _extract_state_dict(checkpoint_data: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint_data, dict):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            value = checkpoint_data.get(key)
            if isinstance(value, dict):
                return value
        if all(torch.is_tensor(value) for value in checkpoint_data.values()):
            return checkpoint_data
    raise ValueError("Could not find a PyTorch state_dict in the checkpoint.")


def _normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ("_orig_mod.", "module.", "model.")
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        clean_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix) :]
                    changed = True
        normalized[clean_key] = value
    return normalized


def _load_checkpoint(checkpoint_path: Path) -> Any:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def _load_custom_model(model_path: Path, device: torch.device) -> tuple[str, Any, Any]:
    checkpoint_path = _find_custom_checkpoint(model_path)
    checkpoint_data = _load_checkpoint(checkpoint_path)
    checkpoint_meta = checkpoint_data if isinstance(checkpoint_data, dict) else None
    config = _load_config(model_path, checkpoint_meta)
    model = DecoderOnlyTransformer(config)
    state_dict = _normalize_state_dict_keys(_extract_state_dict(checkpoint_data))
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint architecture does not match the custom decoder config. "
            "Check vocab_size, block_size, n_layer, n_head, n_embd, and state_dict key names."
        ) from exc
    model.to(device)
    model.eval()
    tokenizer = load_custom_tokenizer(model_path, checkpoint_meta, config.vocab_size)
    return "custom", model, tokenizer


def load_model_and_tokenizer(
    model_path: str | Path,
    device: str | None = None,
) -> tuple[str, Any, Any, torch.device]:
    resolved_path = Path(model_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {resolved_path}")

    torch_device = resolve_device(device)
    if _looks_like_hf_model(resolved_path):
        kind, model, tokenizer = _load_hf_model(resolved_path, torch_device)
    else:
        kind, model, tokenizer = _load_custom_model(resolved_path, torch_device)
    return kind, model, tokenizer, torch_device
