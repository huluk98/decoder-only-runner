from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(int(i) for i in ids).decode("utf-8", errors="replace")


class VocabTokenizer:
    def __init__(self, token_to_id: dict[str, int]) -> None:
        self.token_to_id = token_to_id
        self.id_to_token = {int(idx): token for token, idx in token_to_id.items()}
        self.vocab_size = len(token_to_id)
        self.unk_token = "<unk>" if "<unk>" in token_to_id else None

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for char in text:
            if char in self.token_to_id:
                ids.append(self.token_to_id[char])
            elif self.unk_token is not None:
                ids.append(self.token_to_id[self.unk_token])
            else:
                raise ValueError(
                    f"Character {char!r} is not in vocab.json and no <unk> token exists."
                )
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_token.get(int(idx), "") for idx in ids)


def _token_to_id_from_json(value: Any) -> dict[str, int] | None:
    if isinstance(value, dict):
        if all(isinstance(k, str) and isinstance(v, int) for k, v in value.items()):
            return value
        if "token_to_id" in value:
            return _token_to_id_from_json(value["token_to_id"])
        if "stoi" in value:
            return _token_to_id_from_json(value["stoi"])
    if isinstance(value, list) and all(isinstance(token, str) for token in value):
        return {token: idx for idx, token in enumerate(value)}
    return None


def load_custom_tokenizer(
    model_path: Path,
    checkpoint_data: dict[str, Any] | None,
    vocab_size: int,
):
    tokenizer_json = model_path / "tokenizer.json"
    if tokenizer_json.exists():
        try:
            from transformers import PreTrainedTokenizerFast
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required to load tokenizer.json as a local fast tokenizer."
            ) from exc

        kwargs: dict[str, Any] = {"tokenizer_file": str(tokenizer_json)}
        for filename in ("tokenizer_config.json", "special_tokens_map.json"):
            path = model_path / filename
            if path.exists():
                value = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    for key in (
                        "pad_token",
                        "unk_token",
                        "bos_token",
                        "eos_token",
                        "sep_token",
                        "cls_token",
                        "mask_token",
                        "additional_special_tokens",
                        "model_max_length",
                    ):
                        if key in value:
                            kwargs[key] = value[key]
        tokenizer = PreTrainedTokenizerFast(**kwargs)
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    vocab_path = model_path / "vocab.json"
    if vocab_path.exists():
        token_to_id = _token_to_id_from_json(json.loads(vocab_path.read_text()))
        if token_to_id is None:
            raise ValueError(f"Could not parse token-to-id mapping from {vocab_path}")
        return VocabTokenizer(token_to_id)

    if checkpoint_data:
        for key in ("token_to_id", "stoi", "vocab", "itos"):
            token_to_id = _token_to_id_from_json(checkpoint_data.get(key))
            if token_to_id is not None:
                return VocabTokenizer(token_to_id)

    if vocab_size == ByteTokenizer.vocab_size:
        return ByteTokenizer()

    raise FileNotFoundError(
        "No tokenizer found. Add vocab.json next to the checkpoint, include tokenizer "
        "data in the checkpoint, or use vocab_size=256 for byte-level tokenization."
    )
