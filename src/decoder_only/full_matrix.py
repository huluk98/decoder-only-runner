from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from decoder_only.loader import load_model_and_tokenizer
from decoder_only.sparsity import LinearSparsityConfig, current_linear_sparsity_summary


TRAINING_DATASET = "data/scenic/SCENIC_full_training_dataset.json"
CONTRASTIVE_TRAINING_DATASET = "data/scenic/SCENIC_full_anchor_positive_negative.json"
BENCHMARK_DATASET = "data/benchmarks/iot_instruction_benchmark_200.json"
FAMILIES = ("regular_sft", "contrastive_sft")
ONE_SHOT_SPECS = (
    ("magnitude", 0.30),
    ("wanda", 0.30),
    ("gradient", 0.30),
    ("magnitude", 0.50),
    ("wanda", 0.50),
    ("gradient", 0.50),
    ("nvidia24", 0.50),
)
PROGRESSIVE_TARGETS = (0.30, 0.50)
REQUIRED_TOP_LEVEL_KEYS = (
    "report_type",
    "generated_at",
    "experiment",
    "source_files",
    "actual_rows_total",
    "rows",
    "skipped",
)
METRIC_KEYS = (
    "training_em1",
    "training_em5",
    "benchmark_em1",
    "benchmark_em5",
    "benchmark_easy_em1",
    "benchmark_easy_em5",
    "benchmark_medium_em1",
    "benchmark_medium_em5",
    "benchmark_hard_em1",
    "benchmark_hard_em5",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full decoder pruning matrix.")
    parser.add_argument("input_checkpoint_path")
    parser.add_argument("--output-root", default="outputs/decoder_pruning_full_matrix")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--training-dataset", default=TRAINING_DATASET)
    parser.add_argument("--contrastive-training-dataset", default=CONTRASTIVE_TRAINING_DATASET)
    parser.add_argument("--benchmark", default=BENCHMARK_DATASET)
    parser.add_argument("--nproc-per-node", default=os.environ.get("NPROC_PER_NODE", "8"))
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7"))
    parser.add_argument("--sparsity-gpu-ids", default=os.environ.get("SPARSITY_GPU_IDS", "0,1,2,3,4,5,6,7"))
    parser.add_argument("--regular-sft-epochs", type=int, default=5)
    parser.add_argument("--contrastive-sft-epochs", type=int, default=5)
    parser.add_argument("--recovery-epochs-per-stage", type=int, default=1)
    parser.add_argument("--final-recovery-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--mixed-precision", default="bf16")
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--contrastive-loss-weight", type=float, default=0.1)
    parser.add_argument("--contrastive-margin", type=float, default=0.5)
    parser.add_argument("--negative-field", default="negative")
    parser.add_argument("--dry-run", action="store_true", help="Write the 20-row plan without running training/pruning.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else output_root / "all_sparsity_results.json"
    output_root.mkdir(parents=True, exist_ok=True)

    report = build_initial_report(args, output_root)
    write_report(report, output_json)
    if not args.dry_run:
        run_pipeline(args, output_root, output_json, report)
    else:
        for row in report["rows"]:
            row["error"] = "dry_run_not_executed"
            row["notes"] = append_note(row.get("notes"), "Dry run wrote the expected 20-row plan only.")
        write_report(report, output_json)
    print(f"Wrote {output_json}")


def build_initial_report(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    rows = plan_rows(output_root, args.training_dataset, args.sparsity_gpu_ids)
    return {
        "report_type": "decoder_pruning_full_matrix",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "input_checkpoint_path": args.input_checkpoint_path,
            "regular_sft_checkpoint_path": dense_checkpoint_path(output_root, "regular_sft"),
            "contrastive_sft_checkpoint_path": dense_checkpoint_path(output_root, "contrastive_sft"),
            "training_epochs_regular_sft": args.regular_sft_epochs,
            "training_epochs_contrastive_sft": args.contrastive_sft_epochs,
            "dense_baseline_count": 2,
            "one_shot_pruning_row_count": 14,
            "progressive_pruning_row_count": 4,
            "total_expected_rows": 20,
            "cuda_visible_devices": args.cuda_visible_devices,
            "nproc_per_node": str(args.nproc_per_node),
            "sparsity_gpu_ids": args.sparsity_gpu_ids,
            "recovery_epochs_per_stage": args.recovery_epochs_per_stage,
            "final_recovery_epochs": args.final_recovery_epochs,
            "contrastive_loss_weight": args.contrastive_loss_weight,
            "contrastive_margin": args.contrastive_margin,
            "contrastive_negative_field": args.negative_field,
            "datasets": {
                "training_dataset": args.training_dataset,
                "contrastive_training_dataset": args.contrastive_training_dataset,
                "benchmark": args.benchmark,
            },
            "methods": {
                "one_shot": [method for method, _target in ONE_SHOT_SPECS],
                "progressive": ["progressive_magnitude"],
            },
            "target_sparsities": [0.30, 0.50],
            "notes": "Rows are created before execution so failures are recorded instead of skipped.",
        },
        "source_files": {
            "input_checkpoint_path": args.input_checkpoint_path,
            "training_dataset": args.training_dataset,
            "contrastive_training_dataset": args.contrastive_training_dataset,
            "benchmark": args.benchmark,
            "generated_config_paths": [],
            "output_checkpoint_paths": [],
        },
        "actual_rows_total": len(rows),
        "rows": rows,
        "skipped": [],
    }


def plan_rows(output_root: Path, recovery_train_path: str = TRAINING_DATASET, sparsity_gpu_ids: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    row_id = 1
    for family in FAMILIES:
        rows.append(
            base_row(
                row_id=row_id,
                family=family,
                checkpoint_path=dense_checkpoint_path(output_root, family),
                dense_baseline=True,
                method="dense",
                pruning_schedule="none",
                pruning_base_method=None,
                pruning_target_sparsity=None,
                pruning_stage_sparsities=None,
                recovery_epochs_per_stage=None,
                final_recovery_epochs=None,
                recovery_train_path=None,
                assigned_gpu_id=None,
                execution_backend="torchrun_8gpu",
            )
        )
        row_id += 1

    for family in FAMILIES:
        for method, target in ONE_SHOT_SPECS:
            rows.append(
                base_row(
                    row_id=row_id,
                    family=family,
                    checkpoint_path=one_shot_checkpoint_path(output_root, family, method, target),
                    dense_baseline=False,
                    method=method,
                    pruning_schedule="one_shot",
                    pruning_base_method=method,
                    pruning_target_sparsity=target,
                    pruning_stage_sparsities=None,
                    recovery_epochs_per_stage=None,
                    final_recovery_epochs=None,
                    recovery_train_path=None,
                    assigned_gpu_id=None,
                    execution_backend="single_process_prune",
                )
            )
            row_id += 1

    gpu_ids = default_progressive_gpu_ids(sparsity_gpu_ids)
    for idx, family in enumerate(FAMILIES):
        for target_idx, target in enumerate(PROGRESSIVE_TARGETS):
            assigned_gpu = gpu_ids[idx * len(PROGRESSIVE_TARGETS) + target_idx] if idx * len(PROGRESSIVE_TARGETS) + target_idx < len(gpu_ids) else None
            rows.append(
                base_row(
                    row_id=row_id,
                    family=family,
                    checkpoint_path=progressive_checkpoint_path(output_root, family, target),
                    dense_baseline=False,
                    method="progressive_magnitude",
                    pruning_schedule="progressive",
                    pruning_base_method="magnitude",
                    pruning_target_sparsity=target,
                    pruning_stage_sparsities=progressive_stages(target),
                    recovery_epochs_per_stage=1,
                    final_recovery_epochs=1,
                    recovery_train_path=recovery_train_path,
                    assigned_gpu_id=assigned_gpu,
                    execution_backend="single_gpu_progressive",
                )
            )
            rows[-1]["progressive_parallelism_strategy"] = "split_across_sparsity_gpu_ids"
            row_id += 1
    return rows


def base_row(
    *,
    row_id: int,
    family: str,
    checkpoint_path: str,
    dense_baseline: bool,
    method: str,
    pruning_schedule: str,
    pruning_base_method: str | None,
    pruning_target_sparsity: float | None,
    pruning_stage_sparsities: list[float] | None,
    recovery_epochs_per_stage: int | None,
    final_recovery_epochs: int | None,
    recovery_train_path: str | None,
    assigned_gpu_id: str | None,
    execution_backend: str,
) -> dict[str, Any]:
    row = {
        "row_id": row_id,
        "model_family": "decoder_only",
        "family": family,
        "checkpoint_path": checkpoint_path,
        "dense_baseline": dense_baseline,
        "method": method,
        "pruning_schedule": pruning_schedule,
        "pruning_base_method": pruning_base_method,
        "pruning_target_sparsity": pruning_target_sparsity,
        "pruning_applied": False,
        "measurement_only": False,
        "pruning_stage_sparsities": pruning_stage_sparsities,
        "recovery_epochs_per_stage": recovery_epochs_per_stage,
        "final_recovery_epochs": final_recovery_epochs,
        "recovery_train_path": recovery_train_path,
        "assigned_gpu_id": assigned_gpu_id,
        "execution_backend": execution_backend,
        "targeted_linear_parameters": None,
        "targeted_linear_zeros": None,
        "targeted_linear_sparsity_actual": None,
        "whole_model_parameters": None,
        "whole_model_zeros": None,
        "whole_model_sparsity_actual": None,
        "target_linear_module_count": None,
        "skipped_linear_module_count": None,
        "output_checkpoint_path": checkpoint_path,
        "notes": None,
        "error": None,
    }
    for key in METRIC_KEYS:
        row[key] = None
    return row


def run_pipeline(args: argparse.Namespace, output_root: Path, output_json: Path, report: dict[str, Any]) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    dense_paths = {
        "regular_sft": Path(dense_checkpoint_path(output_root, "regular_sft")),
        "contrastive_sft": Path(dense_checkpoint_path(output_root, "contrastive_sft")),
    }

    run_dense_training(args, output_root, report, env)
    for row in report["rows"]:
        if row["dense_baseline"]:
            finalize_row(args, row, Path(row["checkpoint_path"]), report)
    write_report(report, output_json)

    for row in [item for item in report["rows"] if item["pruning_schedule"] == "one_shot"]:
        base = dense_paths[row["family"]]
        run_one_shot_row(args, row, base, env)
        finalize_row(args, row, Path(row["checkpoint_path"]), report)
        write_report(report, output_json)

    progressive_rows = [item for item in report["rows"] if item["pruning_schedule"] == "progressive"]
    run_progressive_rows(args, progressive_rows, dense_paths, report)
    for row in progressive_rows:
        finalize_row(args, row, Path(row["checkpoint_path"]), report)
    write_report(report, output_json)


def run_dense_training(args: argparse.Namespace, output_root: Path, report: dict[str, Any], env: dict[str, str]) -> None:
    regular_output = output_root / "dense" / "regular_sft"
    contrastive_output = output_root / "dense" / "contrastive_sft"
    commands = [
        (
            "regular_sft",
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={args.nproc_per_node}",
                "-m",
                "decoder_only.train",
                "--model-path",
                args.input_checkpoint_path,
                "--training-mode",
                "sft",
                "--train-data",
                args.training_dataset,
                "--output-dir",
                str(regular_output),
                "--epochs",
                str(args.regular_sft_epochs),
                "--batch-size",
                str(args.batch_size),
                "--gradient-accumulation-steps",
                str(args.gradient_accumulation_steps),
                "--learning-rate",
                str(args.learning_rate),
                "--mixed-precision",
                args.mixed_precision,
                "--gradient-checkpointing",
            ],
        ),
        (
            "contrastive_sft",
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={args.nproc_per_node}",
                "-m",
                "decoder_only.train",
                "--model-path",
                args.input_checkpoint_path,
                "--training-mode",
                "contrastive",
                "--train-data",
                args.contrastive_training_dataset,
                "--output-dir",
                str(contrastive_output),
                "--epochs",
                str(args.contrastive_sft_epochs),
                "--batch-size",
                str(args.batch_size),
                "--gradient-accumulation-steps",
                str(args.gradient_accumulation_steps),
                "--learning-rate",
                str(args.learning_rate),
                "--mixed-precision",
                args.mixed_precision,
                "--max-source-length",
                str(args.max_source_length),
                "--contrastive-loss-weight",
                str(args.contrastive_loss_weight),
                "--contrastive-margin",
                str(args.contrastive_margin),
                "--negative-field",
                args.negative_field,
                "--gradient-checkpointing",
            ],
        ),
    ]
    for family, command in commands:
        row = find_row(report, family=family, method="dense")
        row["execution_command"] = command
        log_path = command_log_path(output_root, f"dense_{family}")
        row["execution_log"] = str(log_path)
        result = run_command(command, env=env, log_path=log_path)
        if result:
            row["error"] = result
            row["notes"] = append_note(row.get("notes"), "Dense SFT failed; row preserved with null metrics.")


def run_one_shot_row(args: argparse.Namespace, row: dict[str, Any], base_checkpoint: Path, env: dict[str, str]) -> None:
    command = [
        sys.executable,
        "-m",
        "decoder_only.prune",
        "--model-path",
        str(base_checkpoint),
        "--output-dir",
        row["checkpoint_path"],
        "--method",
        row["method"],
        "--target-sparsity",
        str(row["pruning_target_sparsity"]),
        "--calibration-data",
        args.training_dataset,
        "--batch-size",
        str(args.batch_size),
    ]
    row["execution_command"] = command
    log_path = command_log_path(
        Path(args.output_root),
        f"row_{row['row_id']:02d}_{row['family']}_{row['method']}_{slug_sparsity(float(row['pruning_target_sparsity']))}",
    )
    row["execution_log"] = str(log_path)
    result = run_command(command, env=env, log_path=log_path)
    if result:
        row["error"] = result
        row["notes"] = append_note(row.get("notes"), "One-shot pruning failed; row preserved with null metrics.")
    else:
        row["pruning_applied"] = True


def run_progressive_rows(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    dense_paths: dict[str, Path],
    report: dict[str, Any],
) -> None:
    processes: list[tuple[dict[str, Any], subprocess.Popen[str]]] = []
    for row in rows:
        env = os.environ.copy()
        if row.get("assigned_gpu_id") is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(row["assigned_gpu_id"])
        base = dense_paths[row["family"]]
        command = [
            sys.executable,
            "-m",
            "decoder_only.prune",
            "--model-path",
            str(base),
            "--output-dir",
            row["checkpoint_path"],
            "--method",
            "magnitude",
            "--target-sparsity",
            str(row["pruning_target_sparsity"]),
            "--progressive-stages",
            ",".join(str(value) for value in row["pruning_stage_sparsities"]),
            "--recovery-train-data",
            args.training_dataset,
            "--recovery-epochs-per-stage",
            str(args.recovery_epochs_per_stage),
            "--final-recovery-epochs",
            str(args.final_recovery_epochs),
            "--batch-size",
            str(args.batch_size),
            "--assigned-gpu-id",
            str(row.get("assigned_gpu_id")),
        ]
        row["execution_command"] = command
        log_path = command_log_path(
            Path(args.output_root),
            f"row_{row['row_id']:02d}_{row['family']}_progressive_{slug_sparsity(float(row['pruning_target_sparsity']))}",
        )
        row["execution_log"] = str(log_path)
        try:
            processes.append((row, subprocess.Popen(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)))
        except Exception as exc:
            row["error"] = str(exc)
            row["notes"] = append_note(row.get("notes"), "Progressive pruning failed to launch.")

    for row, process in processes:
        stdout, stderr = process.communicate()
        if row.get("execution_log"):
            write_command_log(Path(row["execution_log"]), row["execution_command"], process.returncode, stdout, stderr)
        row["stdout_tail"] = tail(stdout)
        row["stderr_tail"] = tail(stderr)
        if process.returncode != 0:
            print_command_failure(row["execution_command"], process.returncode, stdout, stderr, Path(row["execution_log"]))
            row["error"] = f"command_failed_returncode_{process.returncode}: {tail(stderr) or tail(stdout)}"
            row["notes"] = append_note(row.get("notes"), "Progressive pruning failed; row preserved with null metrics.")
        else:
            row["pruning_applied"] = True


def finalize_row(args: argparse.Namespace, row: dict[str, Any], checkpoint_path: Path, report: dict[str, Any]) -> None:
    if not checkpoint_path.exists():
        row["error"] = row.get("error") or f"checkpoint_missing: {checkpoint_path}"
        return
    try:
        kind, model, _tokenizer, _device = load_model_and_tokenizer(checkpoint_path, device="cpu")
        del kind
        summary = current_linear_sparsity_summary(
            model,
            config=LinearSparsityConfig(prune_output_heads=False),
            target_sparsity=row["pruning_target_sparsity"],
            method=row["method"],
        )
        row["targeted_linear_parameters"] = summary["targeted_linear_parameters"]
        row["targeted_linear_zeros"] = summary["targeted_linear_zeros"]
        row["targeted_linear_sparsity_actual"] = summary["targeted_linear_sparsity_actual"]
        row["whole_model_parameters"] = summary["whole_model_parameters"]
        row["whole_model_zeros"] = summary["whole_model_zeros"]
        row["whole_model_sparsity_actual"] = summary["whole_model_sparsity_actual"]
        row["target_linear_module_count"] = summary["target_linear_module_count"]
        row["skipped_linear_module_count"] = summary["skipped_linear_module_count"]
        report["skipped"].extend(summary["skipped_linear_modules"])
        append_source_checkpoint(report, checkpoint_path)
        run_evaluations(args, row, checkpoint_path)
    except Exception as exc:
        row["error"] = row.get("error") or f"sparsity_measurement_failed: {exc}"


def run_evaluations(args: argparse.Namespace, row: dict[str, Any], checkpoint_path: Path) -> None:
    eval_root = Path(row["checkpoint_path"]) / "eval"
    datasets = (
        ("training", args.training_dataset),
        ("benchmark", args.benchmark),
    )
    for prefix, dataset_path in datasets:
        summary_path = eval_root / prefix / "summary.json"
        predictions_path = eval_root / prefix / "predictions.jsonl"
        command = [
            sys.executable,
            "-m",
            "decoder_only.evaluate",
            "--model-path",
            str(checkpoint_path),
            "--data",
            dataset_path,
            "--summary-output",
            str(summary_path),
            "--predictions-output",
            str(predictions_path),
        ]
        log_path = Path(row["checkpoint_path"]) / "eval" / prefix / "evaluate.log"
        result = run_command(command, env=os.environ.copy(), log_path=log_path)
        row[f"{prefix}_evaluation_command"] = command
        row[f"{prefix}_evaluation_log"] = str(log_path)
        if result:
            row["notes"] = append_note(row.get("notes"), f"{prefix} evaluation failed; metrics left null.")
            row["error"] = row.get("error") or result
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            apply_metric_summary(row, prefix, summary)
        except Exception as exc:
            row["notes"] = append_note(row.get("notes"), f"{prefix} evaluation summary parse failed.")
            row["error"] = row.get("error") or str(exc)


def apply_metric_summary(row: dict[str, Any], prefix: str, summary: dict[str, Any]) -> None:
    if prefix == "training":
        row["training_em1"] = summary.get("exact_match_accuracy")
        row["training_em5"] = summary.get("top5_accuracy")
        return
    row["benchmark_em1"] = summary.get("exact_match_accuracy")
    row["benchmark_em5"] = summary.get("top5_accuracy")
    difficulty = summary.get("difficulty") or {}
    for label in ("easy", "medium", "hard"):
        block = difficulty.get(label) or {}
        row[f"benchmark_{label}_em1"] = block.get("em1")
        row[f"benchmark_{label}_em5"] = block.get("em5")


def run_command(command: list[str], env: dict[str, str], log_path: Path | None = None) -> str | None:
    verbose = os.environ.get("DECODER_ONLY_VERBOSE_COMMANDS", "1") not in {"0", "false", "False"}
    if verbose:
        print(f"Running: {shlex.join(command)}", flush=True)
        if log_path is not None:
            print(f"Log: {log_path}", flush=True)
    try:
        completed = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
    except Exception as exc:
        return str(exc)
    if log_path is not None:
        write_command_log(log_path, command, completed.returncode, completed.stdout, completed.stderr)
    if completed.returncode != 0:
        print_command_failure(command, completed.returncode, completed.stdout, completed.stderr, log_path)
        return f"command_failed_returncode_{completed.returncode}: {tail(completed.stderr) or tail(completed.stdout)}"
    return None


def command_log_path(output_root: Path, label: str) -> Path:
    root = Path(os.environ.get("DECODER_ONLY_LOG_DIR", str(output_root / "logs"))).expanduser()
    return root / f"{label}.log"


def write_command_log(
    log_path: Path,
    command: list[str],
    returncode: int,
    stdout: str | None,
    stderr: str | None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        f"$ {shlex.join(command)}",
        f"returncode={returncode}",
        "",
        "===== STDOUT =====",
        stdout or "",
        "",
        "===== STDERR =====",
        stderr or "",
    ]
    log_path.write_text("\n".join(payload), encoding="utf-8")


def print_command_failure(
    command: list[str],
    returncode: int,
    stdout: str | None,
    stderr: str | None,
    log_path: Path | None,
) -> None:
    print(f"Command failed with return code {returncode}: {shlex.join(command)}", file=sys.stderr, flush=True)
    if log_path is not None:
        print(f"Full log: {log_path}", file=sys.stderr, flush=True)
    print("Last command output:", file=sys.stderr, flush=True)
    print(tail(stderr, chars=4000) or tail(stdout, chars=4000) or "(no output)", file=sys.stderr, flush=True)


def write_report(report: dict[str, Any], output_json: Path) -> None:
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["actual_rows_total"] = len(report["rows"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def dense_checkpoint_path(output_root: Path, family: str) -> str:
    return str(output_root / "dense" / family / "checkpoint-final")


def one_shot_checkpoint_path(output_root: Path, family: str, method: str, target: float) -> str:
    return str(output_root / "one_shot" / family / f"{method}_{slug_sparsity(target)}")


def progressive_checkpoint_path(output_root: Path, family: str, target: float) -> str:
    return str(output_root / "progressive" / family / f"progressive_magnitude_{slug_sparsity(target)}")


def slug_sparsity(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def progressive_stages(target: float) -> list[float]:
    if abs(target - 0.30) < 1e-9:
        return [0.10, 0.20, 0.30]
    if abs(target - 0.50) < 1e-9:
        return [0.10, 0.20, 0.30, 0.40, 0.50]
    raise ValueError(f"Unsupported progressive target: {target}")


def default_progressive_gpu_ids(raw: str | None = None) -> list[str]:
    value = raw or os.environ.get("SPARSITY_GPU_IDS", "0,1,2,3,4,5,6,7")
    return [item.strip() for item in value.split(",") if item.strip()]


def find_row(report: dict[str, Any], family: str, method: str) -> dict[str, Any]:
    for row in report["rows"]:
        if row["family"] == family and row["method"] == method:
            return row
    raise KeyError(f"Missing row family={family} method={method}")


def append_source_checkpoint(report: dict[str, Any], checkpoint_path: Path) -> None:
    paths = report["source_files"].setdefault("output_checkpoint_paths", [])
    value = str(checkpoint_path)
    if value not in paths:
        paths.append(value)


def append_note(current: str | None, note: str) -> str:
    if not current:
        return note
    return f"{current} {note}"


def tail(text: str | None, chars: int = 4000) -> str:
    if not text:
        return ""
    return text[-chars:]


if __name__ == "__main__":
    main()
