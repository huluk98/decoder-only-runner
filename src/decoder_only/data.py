from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset


TEXT_SUFFIXES = {".txt", ".text", ".md"}
JSONL_SUFFIXES = {".jsonl", ".ndjson"}
JSON_SUFFIXES = {".json"}
PROMPT_FIELDS = ("prompt", "instruction", "question", "input", "anchor", "x")
RESPONSE_FIELDS = ("response", "output", "answer", "completion", "target", "y")
POSITIVE_FIELDS = ("positive", "pos", "x_positive", "x_plus", "chosen")
NEGATIVE_FIELDS = ("negative", "neg", "x_negative", "x_minus", "rejected")


def iter_data_files(paths: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            for suffix in sorted(TEXT_SUFFIXES | JSONL_SUFFIXES | JSON_SUFFIXES):
                files.extend(sorted(path.rglob(f"*{suffix}")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"Training data path does not exist: {path}")
    if not files:
        raise FileNotFoundError("No supported training data files were found.")
    return files


def load_text_records(
    paths: Iterable[str | Path],
    text_field: str = "text",
    prompt_field: str = "prompt",
    completion_field: str = "completion",
) -> list[str]:
    records: list[str] = []
    for path in iter_data_files(paths):
        suffix = path.suffix.lower()
        if suffix in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8")
            if text.strip():
                records.append(text)
            continue

        if suffix in JSONL_SUFFIXES:
            records.extend(
                _load_jsonl_records(
                    path,
                    text_field=text_field,
                    prompt_field=prompt_field,
                    completion_field=completion_field,
                )
            )
            continue

        if suffix in JSON_SUFFIXES:
            records.extend(
                _load_json_records(
                    path,
                    text_field=text_field,
                    prompt_field=prompt_field,
                    completion_field=completion_field,
                )
            )
            continue

        raise ValueError(f"Unsupported data file type: {path}")

    if not records:
        raise ValueError("No non-empty training records were loaded.")
    return records


def load_contrastive_records(
    paths: Iterable[str | Path],
    negative_field: str = "negative",
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for path in iter_data_files(paths):
        suffix = path.suffix.lower()
        if suffix not in JSONL_SUFFIXES | JSON_SUFFIXES:
            raise ValueError(f"Contrastive data must be JSON or JSONL, got: {path}")
        for index, payload in enumerate(_iter_json_payloads(path)):
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{index} must contain JSON objects.")
            anchor = _clean_text(payload.get("anchor")) or _first_text(payload, PROMPT_FIELDS)
            positive = _clean_text(payload.get("positive")) or _first_text(payload, POSITIVE_FIELDS)
            negative = _clean_text(payload.get(negative_field)) or _first_text(payload, NEGATIVE_FIELDS)
            response = _first_text(payload, RESPONSE_FIELDS)
            if not anchor or not positive or not negative or not response:
                raise ValueError(
                    f"{path}:{index} needs anchor, positive, {negative_field}, and response fields."
                )
            records.append(
                {
                    "anchor": anchor,
                    "positive": positive,
                    "negative": negative,
                    "response": response,
                }
            )
    if not records:
        raise ValueError("No contrastive records were loaded.")
    return records


def _load_json_records(
    path: Path,
    text_field: str,
    prompt_field: str,
    completion_field: str,
) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("data", "rows", "records", "examples"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        payload = [payload]
    return [
        text
        for item in payload
        if (text := _record_to_text(item, text_field, prompt_field, completion_field)).strip()
    ]


def _iter_json_payloads(path: Path) -> list[Any]:
    suffix = path.suffix.lower()
    if suffix in JSONL_SUFFIXES:
        payloads: list[Any] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payloads.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
        return payloads

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("data", "rows", "records", "examples", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return [payload]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"{path} must contain a JSON object, JSON list, or JSONL records.")


def _load_jsonl_records(
    path: Path,
    text_field: str,
    prompt_field: str,
    completion_field: str,
) -> list[str]:
    records: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
            text = _record_to_text(payload, text_field, prompt_field, completion_field)
            if text.strip():
                records.append(text)
    return records


def _record_to_text(
    payload: Any,
    text_field: str,
    prompt_field: str,
    completion_field: str,
) -> str:
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("JSONL records must be objects or strings.")
    if text_field in payload:
        return str(payload[text_field])
    if "prompt" in payload and "response" in payload:
        return str(payload.get("prompt", "")) + str(payload.get("response", ""))
    if prompt_field in payload or completion_field in payload:
        return str(payload.get(prompt_field, "")) + str(payload.get(completion_field, ""))
    if "anchor" in payload and "response" in payload:
        text = str(payload.get("anchor", "")) + str(payload.get("response", ""))
        if payload.get("positive"):
            text += "\n" + str(payload.get("positive", "")) + str(payload.get("response", ""))
        return text
    raise ValueError(
        f"JSON record needs '{text_field}', prompt/response, or "
        f"'{prompt_field}'/'{completion_field}' fields."
    )


def _clean_text(value: Any) -> str:
    return "" if value is None else str(value).replace("\ufeff", "").strip()


def _first_text(payload: dict[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = _clean_text(payload.get(field))
        if value:
            return value
    return ""


def encode_records(
    records: Iterable[str],
    tokenizer: Any,
    append_eos: bool = True,
) -> list[int]:
    token_ids: list[int] = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    for record in records:
        ids = _encode_text(tokenizer, record)
        if not ids:
            continue
        token_ids.extend(ids)
        if append_eos and eos_token_id is not None:
            token_ids.append(int(eos_token_id))
    if not token_ids:
        raise ValueError("Tokenizer produced no tokens from the loaded records.")
    return token_ids


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


class TokenBlockDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        token_ids: list[int],
        block_size: int,
        stride: int | None = None,
    ) -> None:
        if block_size < 2:
            raise ValueError("block_size must be at least 2.")
        self.token_ids = token_ids
        self.block_size = block_size
        self.stride = stride or block_size
        usable = len(token_ids) - block_size
        if usable <= 0:
            raise ValueError(
                f"Need at least {block_size + 1} tokens, but only found {len(token_ids)}."
            )
        self.length = 1 + ((usable - 1) // self.stride)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = index * self.stride
        end = start + self.block_size + 1
        chunk = self.token_ids[start:end]
        if len(chunk) < self.block_size + 1:
            pad_value = chunk[-1]
            chunk = chunk + [pad_value] * (self.block_size + 1 - len(chunk))
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return {"input_ids": x, "labels": y}
