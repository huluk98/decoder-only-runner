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
  MODEL_TRAINING        Row label for training type. Default: decoder_slm
  RUN_LABEL             Row label. Default: basename of MODEL_PATH
  RESULT_BLOCK          Result block label. Default: decoder_model_sparsity_scan
  TARGET_SPARSITY       Desired sparsity value. Default: measured linear sparsity
  PRUNING_MODE          Pruning mode label. Default: measured
  PRUNING_METHOD        Pruning method label. Default: magnitude
  METHOD_LABEL          Method display label. Default: PRUNING_METHOD
  SEED                  Seed value to include in JSON. Default: null
  PRUNE_OUTPUT_HEADS    Include lm_head/classifier/output heads when set to 1.
  REPORT_TYPE           Top-level report type. Default follows encoder-only JSON.
  PYTHON_BIN            Python executable. Default: python

Examples:
  conda activate decoder-only-runner
  ./generate_sparsity_results.sh checkpoints/my-decoder-slm all_sparsity_results.json

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
if [[ -z "${PYTHON_BIN:-}" && -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

"$PYTHON_BIN" - "$MODEL_PATH" "$OUTPUT_JSON" <<'PY'
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise SystemExit(
        "Could not import torch. Activate the project environment first, for example:\n"
        "  conda activate decoder-only-runner"
    ) from exc


MODEL_PATH = Path(sys.argv[1]).expanduser().resolve()
OUTPUT_JSON = Path(sys.argv[2]).expanduser().resolve()

if not MODEL_PATH.exists():
    raise SystemExit(f"Model path does not exist: {MODEL_PATH}")


HEAD_PREFIXES = ("classifier", "lm_head", "qa_outputs", "score", "cls")
HEAD_SUBSTRINGS = (
    "response_classifier",
    "response_projection",
    "final_response_projection",
    "final_projection",
    "output_head",
    "prediction_head",
)
EMBEDDING_NAME_PARTS = (
    "embed",
    "embedding",
    "embeddings",
    "wte",
    "wpe",
    "position",
    "rotary",
    "rope",
    "tok_emb",
    "token_embedding",
)


def env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def env_bool(name: str, default: bool = False) -> bool:
    value = env(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def canonical_module_name(name: str) -> str:
    for prefix in ("_orig_mod.", "module.", "model."):
        while name.startswith(prefix):
            name = name[len(prefix) :]
    return name


def is_output_head_module(name: str) -> bool:
    normalized = canonical_module_name(name).lower()
    parts = tuple(part for part in normalized.split(".") if part)
    if parts and parts[0] in HEAD_PREFIXES:
        return True
    if parts and parts[-1] in HEAD_PREFIXES:
        return True
    return any(fragment in normalized for fragment in HEAD_SUBSTRINGS)


def count_zeros(tensor: torch.Tensor) -> int:
    return int((tensor.detach() == 0).sum().item())


def parameter_sparsity(parameters: list[tuple[str, torch.Tensor]]) -> dict[str, int | float]:
    numel = 0
    zeros = 0
    for _name, parameter in parameters:
        tensor = parameter.detach()
        numel += int(tensor.numel())
        zeros += count_zeros(tensor)
    return {
        "numel": numel,
        "zeros": zeros,
        "sparsity": zeros / numel if numel else 0.0,
    }


def collect_prunable_linear_modules(
    model: nn.Module,
    prune_output_heads: bool,
) -> list[tuple[str, nn.Linear]]:
    modules: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not prune_output_heads and is_output_head_module(name):
            continue
        modules.append((canonical_module_name(name), module))
    return modules


def whole_model_sparsity(model: nn.Module) -> dict[str, int | float]:
    return parameter_sparsity([(name, parameter) for name, parameter in model.named_parameters()])


def current_linear_sparsity_summary_from_model(
    model: nn.Module,
    target_sparsity: float,
    method: str,
    prune_output_heads: bool,
) -> dict[str, Any]:
    modules = collect_prunable_linear_modules(model, prune_output_heads=prune_output_heads)
    target_stats = parameter_sparsity([(f"{name}.weight", module.weight) for name, module in modules])
    whole_stats = whole_model_sparsity(model)
    return {
        "prune_scope": "linear_weights",
        "prune_method": method,
        "target_sparsity": float(target_sparsity),
        "prune_output_heads": bool(prune_output_heads),
        "global_pruning": False,
        "regrowth": False,
        "targeted_linear_parameters": int(target_stats["numel"]),
        "targeted_linear_zeros": int(target_stats["zeros"]),
        "targeted_linear_sparsity_actual": float(target_stats["sparsity"]),
        "whole_model_parameters": int(whole_stats["numel"]),
        "whole_model_zeros": int(whole_stats["zeros"]),
        "whole_model_sparsity_actual": float(whole_stats["sparsity"]),
        "selected_linear_tensors": [
            {
                "name": name,
                "shape": list(module.weight.shape),
                "numel": int(module.weight.numel()),
                "zeros": count_zeros(module.weight),
                "sparsity": count_zeros(module.weight) / int(module.weight.numel())
                if int(module.weight.numel())
                else 0.0,
            }
            for name, module in modules
        ],
    }


def load_model_for_measurement() -> nn.Module:
    try:
        from decoder_only.loader import load_model_and_tokenizer
    except ImportError as exc:
        raise RuntimeError("Could not import decoder_only.loader. Run `pip install -e .` first.") from exc

    # A file checkpoint is measured through its parent directory when possible, because the
    # decoder loader expects config/tokenizer files beside the weight file.
    load_path = MODEL_PATH.parent if MODEL_PATH.is_file() else MODEL_PATH
    _kind, model, _tokenizer, _device = load_model_and_tokenizer(load_path, device="cpu")
    model.eval()
    return model


def load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required for .safetensors checkpoints.") from exc
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

    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
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

    discovered: list[Path] = []
    for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt"):
        discovered.extend(sorted(model_path.glob(pattern)))
    if discovered:
        return discovered
    raise FileNotFoundError(f"No checkpoint weight files found in {model_path}")


def is_probably_linear_weight(name: str, tensor: torch.Tensor, prune_output_heads: bool) -> bool:
    if not torch.is_floating_point(tensor) or tensor.ndim != 2:
        return False
    if not name.endswith(".weight"):
        return False
    module_name = canonical_module_name(name[: -len(".weight")])
    lowered = module_name.lower()
    if any(part in lowered for part in EMBEDDING_NAME_PARTS):
        return False
    if not prune_output_heads and is_output_head_module(module_name):
        return False
    return True


def load_state_dicts(weight_files: list[Path]):
    for path in weight_files:
        if path.suffix.lower() == ".safetensors":
            yield path, load_safetensors(path)
        else:
            yield path, extract_state_dict(torch_load(path), path)


def current_linear_sparsity_summary_from_state_dict(
    target_sparsity: float,
    method: str,
    prune_output_heads: bool,
) -> tuple[dict[str, Any], list[Path]]:
    load_path = MODEL_PATH if MODEL_PATH.is_dir() else MODEL_PATH.parent
    weight_files = discover_weight_files(MODEL_PATH if MODEL_PATH.is_file() else load_path)
    whole_parameters: list[tuple[str, torch.Tensor]] = []
    linear_parameters: list[tuple[str, torch.Tensor]] = []
    selected_linear_tensors: list[dict[str, Any]] = []

    for _source, state_dict in load_state_dicts(weight_files):
        for raw_name, tensor in state_dict.items():
            if not torch.is_tensor(tensor) or not torch.is_floating_point(tensor):
                continue
            name = canonical_module_name(raw_name)
            # State dict fallback cannot distinguish parameters from buffers perfectly, so ignore
            # common non-parameter masks/bias buffers by only counting named weights/biases.
            if name.endswith(".weight") or name.endswith(".bias"):
                whole_parameters.append((name, tensor))
            if is_probably_linear_weight(name, tensor, prune_output_heads=prune_output_heads):
                linear_parameters.append((name, tensor))
                zeros = count_zeros(tensor)
                numel = int(tensor.numel())
                selected_linear_tensors.append(
                    {
                        "name": name[: -len(".weight")],
                        "shape": list(tensor.shape),
                        "numel": numel,
                        "zeros": zeros,
                        "sparsity": zeros / numel if numel else 0.0,
                    }
                )
        del state_dict

    target_stats = parameter_sparsity(linear_parameters)
    whole_stats = parameter_sparsity(whole_parameters)
    return (
        {
            "prune_scope": "linear_weights",
            "prune_method": method,
            "target_sparsity": float(target_sparsity),
            "prune_output_heads": bool(prune_output_heads),
            "global_pruning": False,
            "regrowth": False,
            "targeted_linear_parameters": int(target_stats["numel"]),
            "targeted_linear_zeros": int(target_stats["zeros"]),
            "targeted_linear_sparsity_actual": float(target_stats["sparsity"]),
            "whole_model_parameters": int(whole_stats["numel"]),
            "whole_model_zeros": int(whole_stats["zeros"]),
            "whole_model_sparsity_actual": float(whole_stats["sparsity"]),
            "selected_linear_tensors": selected_linear_tensors,
        },
        weight_files,
    )


def metric_block(summary_output: str | None = None) -> dict[str, Any]:
    return {
        "json": None,
        "rows": None,
        "scored_rows": None,
        "em1": None,
        "em5": None,
        "difficulty": {
            label: {"rows": None, "scored_rows": None, "em1": None, "em5": None}
            for label in ("easy", "medium", "hard")
        },
        "summary_output": summary_output,
        "predictions_output": None,
    }


def flat_metric_fields(prefix: str, metrics: dict[str, Any] | None) -> dict[str, Any]:
    metrics = metrics or {}
    difficulty = metrics.get("difficulty") or {}
    fields = {
        f"{prefix}_em1_overall": metrics.get("em1"),
        f"{prefix}_em5_overall": metrics.get("em5"),
        f"{prefix}_count_total": metrics.get("rows"),
        f"{prefix}_scored_rows": metrics.get("scored_rows"),
    }
    for label in ("easy", "medium", "hard"):
        block = difficulty.get(label) or {}
        fields[f"{prefix}_em1_{label}"] = block.get("em1")
        fields[f"{prefix}_em5_{label}"] = block.get("em5")
        fields[f"{prefix}_count_{label}"] = block.get("rows")
    return fields


def make_payload() -> dict[str, Any]:
    pruning_method = env("PRUNING_METHOD", "magnitude") or "magnitude"
    method_label = env("METHOD_LABEL", pruning_method)
    prune_output_heads = env_bool("PRUNE_OUTPUT_HEADS", False)
    target_sparsity_env = as_float(env("TARGET_SPARSITY"))
    skipped: list[dict[str, Any]] = []
    try:
        weight_files = discover_weight_files(MODEL_PATH if MODEL_PATH.is_file() else MODEL_PATH)
    except Exception:
        weight_files = []

    try:
        model = load_model_for_measurement()
        provisional_target = target_sparsity_env if target_sparsity_env is not None else 0.0
        pruning_config = current_linear_sparsity_summary_from_model(
            model,
            target_sparsity=provisional_target,
            method=pruning_method,
            prune_output_heads=prune_output_heads,
        )
        measurement_source = "model_modules"
    except Exception as exc:
        provisional_target = target_sparsity_env if target_sparsity_env is not None else 0.0
        pruning_config, fallback_weight_files = current_linear_sparsity_summary_from_state_dict(
            target_sparsity=provisional_target,
            method=pruning_method,
            prune_output_heads=prune_output_heads,
        )
        weight_files = fallback_weight_files
        measurement_source = "state_dict_fallback"
        skipped.append(
            {
                "checkpoint_path": str(MODEL_PATH),
                "reason": f"Model module loading failed; used state_dict fallback: {exc}",
            }
        )

    if target_sparsity_env is None:
        pruning_config["target_sparsity"] = pruning_config["targeted_linear_sparsity_actual"]

    if int(pruning_config["targeted_linear_parameters"]) == 0:
        skipped.append(
            {
                "checkpoint_path": str(MODEL_PATH),
                "reason": "No prunable Linear weights found",
            }
        )

    training_metrics = metric_block(summary_output=str(OUTPUT_JSON))
    benchmark_metrics = metric_block(summary_output=str(OUTPUT_JSON))
    model_training = env("MODEL_TRAINING", "decoder_slm")
    row = {
        "result_block": env("RESULT_BLOCK", "decoder_model_sparsity_scan"),
        "model_training": model_training,
        "run_label": env("RUN_LABEL", MODEL_PATH.name),
        "model_family": "decoder_only",
        "pruning_mode": env("PRUNING_MODE", "measured"),
        "pruning_method": pruning_method,
        "method_label": method_label,
        "target_sparsity": as_float(pruning_config["target_sparsity"]),
        "targeted_linear_sparsity_actual": as_float(
            pruning_config["targeted_linear_sparsity_actual"]
        ),
        "whole_model_sparsity_actual": as_float(pruning_config["whole_model_sparsity_actual"]),
        "seed": as_int(env("SEED")),
        "checkpoint_path": str(MODEL_PATH),
        "mask_path": env("MASK_PATH"),
        "training_metrics": training_metrics,
        "benchmark_metrics": benchmark_metrics,
        **flat_metric_fields("training", training_metrics),
        **flat_metric_fields("benchmark", benchmark_metrics),
        "em1_retention_overall": None,
        "em5_retention_overall": None,
        "training_config": None,
        "pruning_config": {
            **pruning_config,
            "measurement_source": measurement_source,
        },
        "notes": env(
            "NOTES",
            "Generated by generate_sparsity_results.sh following the encoder-only sparsity report schema; EM metrics are null because this script only measures checkpoint sparsity.",
        ),
    }

    return {
        "report_type": env("REPORT_TYPE", "scenic_sparsity_revision_combined_results"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "base_model": str(MODEL_PATH),
            "sft_epochs": as_int(env("SFT_EPOCHS")),
            "model_trainings": [model_training],
            "original_one_shot_expected_rows_per_training": 1,
            "original_one_shot_expected_rows_total": 1,
            "dense_baseline_expected_rows_total": 0,
            "progressive_expected_rows_per_training": 0,
            "progressive_expected_rows_total": 0,
            "expected_rows_total": 1,
            "gradual_prune_method": pruning_method,
            "recovery_epochs_per_stage": as_int(env("RECOVERY_EPOCHS_PER_STAGE")),
            "final_recovery_epochs": as_int(env("FINAL_RECOVERY_EPOCHS")),
            "gradient_calibration_batch_size": as_int(env("GRADIENT_CALIBRATION_BATCH_SIZE")),
            "gradient_calibration_batches": as_int(env("GRADIENT_CALIBRATION_BATCHES")),
        },
        "source_files": {
            "original_one_shot_summary": None,
            "linear_sparsity_retune_summaries": [],
            "decoder_model_path": str(MODEL_PATH),
            "weight_files": [str(path) for path in weight_files],
        },
        "actual_rows_total": 1,
        "rows": [row],
        "skipped": skipped,
    }


payload = make_payload()
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {OUTPUT_JSON}")
PY
