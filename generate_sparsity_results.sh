#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./generate_sparsity_results.sh MODEL_PATH [OUTPUT_JSON]

Arguments:
  MODEL_PATH    Path to a local decoder-only model directory or checkpoint file.
  OUTPUT_JSON   Optional output file. Defaults to ./all_sparsity_results.json.

Optional environment labels:
  MODEL_TRAINING   Row label for training type. Default: decoder_slm
  RUN_LABEL        Row label. Default: basename of MODEL_PATH
  TARGET_SPARSITY  Desired sparsity value. Default: measured linear sparsity
  PRUNING_MODE     Pruning mode label. Default: measured
  PRUNING_METHOD   Pruning method label. Default: unknown
  METHOD_LABEL     Method display label. Default: PRUNING_METHOD
  SEED             Seed value to include in JSON. Default: null
  INCLUDE_LM_HEAD  Count lm_head/output heads as linear weights when set to 1.
  PYTHON_BIN       Python executable. Default: python3

Examples:
  conda activate decoder-only-runner
  ./generate_sparsity_results.sh checkpoints/my-decoder-slm results.json

  MODEL_TRAINING=regular_sft RUN_LABEL=magnitude_0p5 TARGET_SPARSITY=0.5 \
    ./generate_sparsity_results.sh /path/to/checkpoint /tmp/all_sparsity_results.json
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

MODEL_PATH="$1"
OUTPUT_JSON="${2:-all_sparsity_results.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" - "$MODEL_PATH" "$OUTPUT_JSON" <<'PY'
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "Could not import torch. Activate the project environment first, for example:\n"
        "  conda activate decoder-only-runner"
    ) from exc


MODEL_PATH = Path(sys.argv[1]).expanduser().resolve()
OUTPUT_JSON = Path(sys.argv[2]).expanduser().resolve()

if not MODEL_PATH.exists():
    raise SystemExit(f"Model path does not exist: {MODEL_PATH}")


def env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def parse_float_or_none(value: str | None) -> float | None:
    if value is None:
        return None
    return float(value)


def parse_seed(value: str | None) -> int | None:
    if value is None:
        return None
    return int(value)


def load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise SystemExit(
            "safetensors is required for .safetensors checkpoints. "
            "Install the project requirements first."
        ) from exc
    return load_file(path, device="cpu")


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(payload: Any, source: Path) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model", "net", "module"):
            value = payload.get(key)
            if isinstance(value, dict) and any(torch.is_tensor(v) for v in value.values()):
                return {str(k): v for k, v in value.items() if torch.is_tensor(v)}
        if any(torch.is_tensor(v) for v in payload.values()):
            return {str(k): v for k, v in payload.items() if torch.is_tensor(v)}
    raise ValueError(f"Could not find tensor state_dict in {source}")


def index_shards(index_path: Path) -> list[Path]:
    data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"Missing weight_map in {index_path}")
    return sorted({index_path.parent / str(name) for name in weight_map.values()})


def discover_weight_files(model_path: Path) -> list[Path]:
    if model_path.is_file():
        return [model_path]

    index_names = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
    for name in index_names:
        index_path = model_path / name
        if index_path.exists():
            return index_shards(index_path)

    preferred = (
        "model.safetensors",
        "pytorch_model.bin",
        "model.pt",
        "checkpoint.pt",
        "model.pth",
        "checkpoint.pth",
    )
    files = [model_path / name for name in preferred if (model_path / name).exists()]
    if files:
        return files

    patterns = ("*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt")
    discovered: list[Path] = []
    for pattern in patterns:
        discovered.extend(sorted(model_path.glob(pattern)))
    if discovered:
        return discovered

    raise FileNotFoundError(f"No checkpoint weight files found in {model_path}")


def normalize_name(name: str) -> str:
    prefixes = ("_orig_mod.", "module.", "model.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                changed = True
    return name


def is_float_tensor(tensor: torch.Tensor) -> bool:
    return torch.is_floating_point(tensor)


def is_linear_weight(name: str, tensor: torch.Tensor) -> bool:
    if not is_float_tensor(tensor):
        return False
    if tensor.ndim != 2:
        return False

    lname = name.lower()
    excluded = (
        "embed",
        "embedding",
        "embeddings",
        "wte",
        "wpe",
        "position",
        "rotary",
        "rope",
        "token_embedding",
        "tok_emb",
    )
    if any(part in lname for part in excluded):
        return False
    if env("INCLUDE_LM_HEAD", "0") != "1" and (
        "lm_head" in lname or "output_head" in lname or "classifier" in lname
    ):
        return False
    return lname.endswith("weight") or ".weight" in lname or "weight" in lname


def tensor_zero_stats(tensor: torch.Tensor) -> tuple[int, int]:
    numel = int(tensor.numel())
    if numel == 0:
        return 0, 0
    zeros = int(numel - torch.count_nonzero(tensor).item())
    return zeros, numel


def add_stats(total: dict[str, int], zeros: int, numel: int) -> None:
    total["zero"] += zeros
    total["total"] += numel


def ratio(stats: dict[str, int]) -> float | None:
    if stats["total"] == 0:
        return None
    return stats["zero"] / stats["total"]


def load_state_dicts(weight_files: list[Path]):
    for path in weight_files:
        suffix = path.suffix.lower()
        if suffix == ".safetensors":
            state_dict = load_safetensors(path)
        else:
            state_dict = extract_state_dict(torch_load(path), path)
        yield path, state_dict


def null_metrics(summary_output: str | None = None) -> dict[str, Any]:
    return {
        "json": None,
        "rows": None,
        "scored_rows": None,
        "em1": None,
        "em5": None,
        "difficulty": {
            "easy": {"rows": None, "scored_rows": None, "em1": None, "em5": None},
            "medium": {"rows": None, "scored_rows": None, "em1": None, "em5": None},
            "hard": {"rows": None, "scored_rows": None, "em1": None, "em5": None},
        },
        "summary_output": summary_output,
        "predictions_output": None,
    }


def make_report() -> dict[str, Any]:
    weight_files = discover_weight_files(MODEL_PATH)
    whole = {"zero": 0, "total": 0}
    linear = {"zero": 0, "total": 0}
    tensor_counts = {
        "weight_files": len(weight_files),
        "float_tensors": 0,
        "linear_weight_tensors": 0,
    }
    skipped: list[dict[str, Any]] = []

    for source, state_dict in load_state_dicts(weight_files):
        for raw_name, tensor in state_dict.items():
            if not torch.is_tensor(tensor) or not is_float_tensor(tensor):
                continue
            name = normalize_name(raw_name)
            zeros, numel = tensor_zero_stats(tensor.detach().cpu())
            add_stats(whole, zeros, numel)
            tensor_counts["float_tensors"] += 1
            if is_linear_weight(name, tensor):
                add_stats(linear, zeros, numel)
                tensor_counts["linear_weight_tensors"] += 1
        del state_dict

    linear_sparsity = ratio(linear)
    whole_sparsity = ratio(whole)
    if linear_sparsity is None:
        skipped.append(
            {
                "checkpoint_path": str(MODEL_PATH),
                "reason": "No 2D linear weight tensors found by the default heuristic",
            }
        )

    target_sparsity = parse_float_or_none(env("TARGET_SPARSITY"))
    if target_sparsity is None:
        target_sparsity = linear_sparsity

    run_label = env("RUN_LABEL", MODEL_PATH.name)
    pruning_method = env("PRUNING_METHOD", "unknown")
    method_label = env("METHOD_LABEL", pruning_method)
    output_summary = str(OUTPUT_JSON)

    row = {
        "result_block": env("RESULT_BLOCK", "decoder_model_sparsity_scan"),
        "model_training": env("MODEL_TRAINING", "decoder_slm"),
        "run_label": run_label,
        "model_family": "decoder_only",
        "pruning_mode": env("PRUNING_MODE", "measured"),
        "pruning_method": pruning_method,
        "method_label": method_label,
        "target_sparsity": target_sparsity,
        "targeted_linear_sparsity_actual": linear_sparsity,
        "whole_model_sparsity_actual": whole_sparsity,
        "seed": parse_seed(env("SEED")),
        "checkpoint_path": str(MODEL_PATH),
        "mask_path": env("MASK_PATH"),
        "training_metrics": null_metrics(summary_output=output_summary),
        "benchmark_metrics": null_metrics(summary_output=output_summary),
        "training_em1_overall": None,
        "training_em5_overall": None,
        "training_count_total": None,
        "training_scored_rows": None,
        "training_em1_easy": None,
        "training_em5_easy": None,
        "training_count_easy": None,
        "training_em1_medium": None,
        "training_em5_medium": None,
        "training_count_medium": None,
        "training_em1_hard": None,
        "training_em5_hard": None,
        "training_count_hard": None,
        "benchmark_em1_overall": None,
        "benchmark_em5_overall": None,
        "benchmark_count_total": None,
        "benchmark_scored_rows": None,
        "benchmark_em1_easy": None,
        "benchmark_em5_easy": None,
        "benchmark_count_easy": None,
        "benchmark_em1_medium": None,
        "benchmark_em5_medium": None,
        "benchmark_count_medium": None,
        "benchmark_em1_hard": None,
        "benchmark_em5_hard": None,
        "benchmark_count_hard": None,
        "training_config": None,
        "pruning_config": {
            "prune_scope": "linear_weights",
            "linear_weight_zero_params": linear["zero"],
            "linear_weight_total_params": linear["total"],
            "whole_model_zero_params": whole["zero"],
            "whole_model_total_float_params": whole["total"],
            "include_lm_head": env("INCLUDE_LM_HEAD", "0") == "1",
            **tensor_counts,
        },
        "notes": env(
            "NOTES",
            "Generated by generate_sparsity_results.sh; EM metrics are null because this script only measures checkpoint sparsity.",
        ),
    }

    return {
        "report_type": env("REPORT_TYPE", "decoder_sparsity_results"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "base_model": str(MODEL_PATH),
            "model_family": "decoder_only",
            "expected_rows_total": 1,
            "prune_scope": "linear_weights",
            "include_lm_head": env("INCLUDE_LM_HEAD", "0") == "1",
        },
        "source_files": {
            "model_path": str(MODEL_PATH),
            "weight_files": [str(path) for path in weight_files],
        },
        "actual_rows_total": 1,
        "rows": [row],
        "skipped": skipped,
    }


report = make_report()
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {OUTPUT_JSON}")
PY
