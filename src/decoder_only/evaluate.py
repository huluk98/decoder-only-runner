from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from decoder_only.loader import load_model_and_tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate decoder outputs with EM1/EM5.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--predictions-output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-rows", type=int, default=0, help="0 means all rows.")
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = load_eval_records(Path(args.data))
    if args.max_rows > 0:
        records = records[: args.max_rows]
    kind, model, tokenizer, device = load_model_and_tokenizer(args.model_path, args.device)
    predictions = []
    for record in records:
        candidates = generate_candidates(kind, model, tokenizer, device, record["prompt"], args.max_new_tokens)
        expected = normalize(record["response"])
        em_flags = [normalize(candidate).startswith(expected) for candidate in candidates]
        predictions.append(
            {
                **record,
                "candidates": candidates,
                "em1": bool(em_flags[:1] and em_flags[0]),
                "em5": bool(any(em_flags[:5])),
            }
        )
    write_outputs(args, records, predictions)


def load_eval_records(path: Path) -> list[dict[str, Any]]:
    payloads: list[Any] = []
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as handle:
            payloads = [json.loads(line) for line in handle if line.strip()]
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        payloads = data if isinstance(data, list) else [data]
    records = []
    for idx, payload in enumerate(payloads):
        if not isinstance(payload, dict):
            continue
        prompt = payload.get("prompt") or payload.get("anchor") or payload.get("instruction")
        response = payload.get("response") or payload.get("completion") or payload.get("answer")
        if prompt is None or response is None:
            continue
        records.append(
            {
                "id": payload.get("id") or payload.get("source_id") or str(idx),
                "difficulty": payload.get("difficulty"),
                "prompt": str(prompt),
                "response": str(response),
            }
        )
    if not records:
        raise ValueError(f"No prompt/response evaluation records found in {path}")
    return records


def generate_candidates(
    kind: str,
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
) -> list[str]:
    if kind == "hf":
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=5,
                num_return_sequences=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        decoded = [tokenizer.decode(item, skip_special_tokens=True) for item in output]
        return [remove_prompt(text, prompt) for text in decoded]

    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [0]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_k=None,
            do_sample=False,
        )
    text = tokenizer.decode(output[0].tolist())
    return [remove_prompt(text, prompt)]


def remove_prompt(text: str, prompt: str) -> str:
    if text.startswith(prompt):
        return text[len(prompt) :]
    return text


def normalize(text: str) -> str:
    return "".join(str(text).split())


def write_outputs(args: argparse.Namespace, records: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> None:
    summary_path = Path(args.summary_output)
    predictions_path = Path(args.predictions_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    scored = len(predictions)
    em1 = sum(1 for row in predictions if row["em1"]) / scored if scored else None
    em5 = sum(1 for row in predictions if row["em5"]) / scored if scored else None
    difficulty = {}
    for label in ("easy", "medium", "hard"):
        subset = [row for row in predictions if row.get("difficulty") == label]
        count = len(subset)
        difficulty[label] = {
            "rows": count,
            "scored_rows": count,
            "em1": sum(1 for row in subset if row["em1"]) / count if count else None,
            "em5": sum(1 for row in subset if row["em5"]) / count if count else None,
        }
    summary = {
        "json": args.data,
        "rows": len(records),
        "scored_rows": scored,
        "exact_match_accuracy": em1,
        "top5_accuracy": em5,
        "difficulty": difficulty,
        "summary_output": str(summary_path),
        "predictions_output": str(predictions_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
