from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from decoder_only.custom_model import DecoderOnlyTransformer
from decoder_only.data import TokenBlockDataset, encode_records, load_text_records
from decoder_only.loader import load_model_and_tokenizer, resolve_device
from decoder_only.sparsity import (
    LinearSparsityConfig,
    apply_masks,
    collect_linear_module_report,
    current_linear_sparsity_summary,
    global_unstructured_masks,
    gradient_scores,
    magnitude_scores,
    nvidia24_masks,
)
from decoder_only.train import build_optimizer, build_scheduler, forward_loss, infer_block_size
from decoder_only.tokenizers import VocabTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prune a decoder-only checkpoint.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", choices=("magnitude", "wanda", "gradient", "nvidia24"), required=True)
    parser.add_argument("--target-sparsity", type=float, required=True)
    parser.add_argument("--calibration-data", nargs="*", default=None)
    parser.add_argument("--recovery-train-data", nargs="*", default=None)
    parser.add_argument("--progressive-stages", default=None, help="Comma-separated stages, e.g. 0.1,0.2,0.3")
    parser.add_argument("--recovery-epochs-per-stage", type=int, default=0)
    parser.add_argument("--final-recovery-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-calibration-batches", type=int, default=8)
    parser.add_argument("--prune-output-heads", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--assigned-gpu-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    if args.method == "nvidia24" and abs(args.target_sparsity - 0.5) > 1e-9:
        raise ValueError("nvidia24 is exact 2:4 sparsity and only supports target_sparsity=0.5")

    kind, model, tokenizer, _ = load_model_and_tokenizer(args.model_path, device="cpu")
    device = resolve_device(args.device)
    model.to(device)
    model.eval()

    block_size = args.block_size or infer_block_size(kind, model, tokenizer)
    stages = parse_stages(args.progressive_stages, args.target_sparsity)
    masks: dict[str, torch.Tensor] = {}

    if args.progressive_stages:
        for stage in stages:
            masks = make_masks(args, kind, model, tokenizer, block_size, stage)
            apply_masks(model, masks)
            if args.recovery_epochs_per_stage > 0:
                run_recovery(args, kind, model, tokenizer, block_size, masks, args.recovery_epochs_per_stage)
        if args.final_recovery_epochs > 0:
            run_recovery(args, kind, model, tokenizer, block_size, masks, args.final_recovery_epochs)
    else:
        masks = make_masks(args, kind, model, tokenizer, block_size, args.target_sparsity)
        apply_masks(model, masks)

    save_pruned_checkpoint(args, kind, model, tokenizer, masks)


def parse_stages(raw: str | None, target: float) -> list[float]:
    if not raw:
        return [target]
    stages = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not stages or abs(stages[-1] - target) > 1e-9:
        raise ValueError("progressive stages must end at target_sparsity")
    return stages


def make_masks(
    args: argparse.Namespace,
    kind: str,
    model: torch.nn.Module,
    tokenizer: Any,
    block_size: int,
    target_sparsity: float,
) -> dict[str, torch.Tensor]:
    modules, _skipped = collect_linear_module_report(model, prune_output_heads=args.prune_output_heads)
    if args.method == "nvidia24":
        return nvidia24_masks(modules)
    if args.method == "magnitude":
        return global_unstructured_masks(modules, magnitude_scores(modules), target_sparsity)
    if args.method == "gradient":
        calibrate_gradients(args, kind, model, tokenizer, block_size)
        return global_unstructured_masks(modules, gradient_scores(modules), target_sparsity)
    if args.method == "wanda":
        scores = wanda_scores(args, kind, model, tokenizer, block_size, modules)
        return global_unstructured_masks(modules, scores, target_sparsity)
    raise ValueError(f"Unsupported method: {args.method}")


def build_token_loader(
    paths: list[str] | None,
    tokenizer: Any,
    block_size: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    if not paths:
        raise ValueError("Calibration/recovery data is required for this pruning method.")
    records = load_text_records(paths)
    token_ids = encode_records(records, tokenizer)
    dataset = TokenBlockDataset(token_ids, block_size=block_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def calibrate_gradients(
    args: argparse.Namespace,
    kind: str,
    model: torch.nn.Module,
    tokenizer: Any,
    block_size: int,
) -> None:
    loader = build_token_loader(
        args.calibration_data or args.recovery_train_data,
        tokenizer,
        block_size,
        args.batch_size,
        shuffle=False,
    )
    model.train()
    model.zero_grad(set_to_none=True)
    for idx, batch in enumerate(loader):
        batch = {key: value.to(next(model.parameters()).device) for key, value in batch.items()}
        loss = forward_loss(kind, model, batch)
        loss.backward()
        if idx + 1 >= args.max_calibration_batches:
            break
    model.eval()


def wanda_scores(
    args: argparse.Namespace,
    kind: str,
    model: torch.nn.Module,
    tokenizer: Any,
    block_size: int,
    modules: list[tuple[str, torch.nn.Linear]],
) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    loader = build_token_loader(
        args.calibration_data or args.recovery_train_data,
        tokenizer,
        block_size,
        args.batch_size,
        shuffle=False,
    )
    activation_sums: dict[str, torch.Tensor] = {}
    activation_counts: dict[str, int] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, inputs, _output):
            x = inputs[0].detach().float()
            x = x.reshape(-1, x.shape[-1])
            activation_sums[name] = activation_sums.get(name, torch.zeros(x.shape[-1], device=x.device)) + (
                x.pow(2).sum(dim=0)
            )
            activation_counts[name] = activation_counts.get(name, 0) + x.shape[0]

        return hook

    for name, module in modules:
        handles.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            if kind == "hf":
                model(input_ids=input_ids)
            else:
                model(input_ids)
            if idx + 1 >= args.max_calibration_batches:
                break

    for handle in handles:
        handle.remove()

    scores: dict[str, torch.Tensor] = {}
    for name, module in modules:
        if name not in activation_sums:
            scores[name] = module.weight.detach().abs().cpu()
            continue
        rms = (activation_sums[name] / max(1, activation_counts[name])).sqrt().to(module.weight.device)
        scores[name] = (module.weight.detach().abs() * rms.reshape(1, -1)).cpu()
    return scores


def run_recovery(
    args: argparse.Namespace,
    kind: str,
    model: torch.nn.Module,
    tokenizer: Any,
    block_size: int,
    masks: dict[str, torch.Tensor],
    epochs: int,
) -> None:
    loader = build_token_loader(args.recovery_train_data, tokenizer, block_size, args.batch_size, shuffle=True)
    optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
    scheduler = build_scheduler(optimizer, max(1, len(loader) * epochs), args.warmup_steps)
    device = next(model.parameters()).device
    model.train()
    step = 0
    for _epoch in range(epochs):
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = forward_loss(kind, model, batch)
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            apply_masks(model, masks)
            step += 1
    model.eval()


def save_pruned_checkpoint(
    args: argparse.Namespace,
    kind: str,
    model: torch.nn.Module,
    tokenizer: Any,
    masks: dict[str, torch.Tensor],
) -> None:
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = current_linear_sparsity_summary(
        model,
        config=LinearSparsityConfig(prune_output_heads=args.prune_output_heads),
        target_sparsity=args.target_sparsity,
        method=args.method,
    )
    summary.update(
        {
            "method": args.method,
            "checkpoint_path": str(output_dir),
            "assigned_gpu_id": args.assigned_gpu_id,
            "progressive_stages": parse_stages(args.progressive_stages, args.target_sparsity)
            if args.progressive_stages
            else None,
            "recovery_epochs_per_stage": args.recovery_epochs_per_stage,
            "final_recovery_epochs": args.final_recovery_epochs,
        }
    )
    if kind == "hf":
        model.save_pretrained(output_dir)
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(output_dir)
    else:
        if not isinstance(model, DecoderOnlyTransformer):
            raise TypeError("Expected DecoderOnlyTransformer for custom checkpoint saving.")
        config = asdict(model.config)
        (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        torch.save({"model_state_dict": model.state_dict(), "config": config}, output_dir / "model.pt")
        if isinstance(tokenizer, VocabTokenizer):
            (output_dir / "vocab.json").write_text(
                json.dumps(tokenizer.token_to_id, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
    torch.save({"masks": {name: mask.to(torch.uint8).cpu() for name, mask in masks.items()}}, output_dir / "masks.pt")
    (output_dir / "pruning_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
