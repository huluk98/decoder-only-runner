from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from decoder_only.custom_model import DecoderConfig, DecoderOnlyTransformer
from decoder_only.data import TokenBlockDataset, encode_records, load_text_records
from decoder_only.loader import load_model_and_tokenizer
from decoder_only.tokenizers import VocabTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Continue training a decoder-only model.")
    parser.add_argument("--model-path", required=True, help="Existing trained model/checkpoint dir.")
    parser.add_argument("--train-data", nargs="+", required=True, help="Text/JSONL files or dirs.")
    parser.add_argument("--validation-data", nargs="*", default=None)
    parser.add_argument("--output-dir", default="outputs/further-trained")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--completion-field", default="completion")
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="no")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile when available.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    set_seed(args.seed)

    kind, model, tokenizer, _ = load_model_and_tokenizer(args.model_path, device="cpu")
    model.train()
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if args.compile:
        model = torch.compile(model)

    block_size = args.block_size or infer_block_size(kind, model, tokenizer)
    train_dataset = build_dataset(args.train_data, tokenizer, block_size, args)
    val_dataset = (
        build_dataset(args.validation_data, tokenizer, block_size, args)
        if args.validation_data
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        if val_dataset is not None
        else None
    )

    optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
    total_steps = infer_total_steps(args, train_loader)
    scheduler = build_scheduler(optimizer, total_steps, args.warmup_steps)

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)

    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_training_args(output_dir, args, kind, block_size, len(train_dataset))
    accelerator.wait_for_everyone()

    train_loop(
        args=args,
        accelerator=accelerator,
        kind=kind,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        total_steps=total_steps,
    )


def build_dataset(
    paths: list[str] | None,
    tokenizer: Any,
    block_size: int,
    args: argparse.Namespace,
) -> TokenBlockDataset:
    if not paths:
        raise ValueError("Dataset paths cannot be empty.")
    records = load_text_records(
        paths,
        text_field=args.text_field,
        prompt_field=args.prompt_field,
        completion_field=args.completion_field,
    )
    tokens = encode_records(records, tokenizer)
    return TokenBlockDataset(tokens, block_size=block_size, stride=args.stride)


def infer_block_size(kind: str, model: Any, tokenizer: Any) -> int:
    if kind == "custom":
        return int(model.config.block_size)
    config = getattr(model, "config", None)
    for name in ("max_position_embeddings", "n_positions", "n_ctx"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return min(value, 4096)
    model_max_length = getattr(tokenizer, "model_max_length", None)
    if isinstance(model_max_length, int) and 0 < model_max_length < 1_000_000:
        return min(model_max_length, 4096)
    return 1024


def infer_total_steps(args: argparse.Namespace, train_loader: DataLoader) -> int:
    if args.max_steps > 0:
        return args.max_steps
    update_steps_per_epoch = max(1, math.ceil(len(train_loader) / args.gradient_accumulation_steps))
    return max(1, math.ceil(update_steps_per_epoch * args.epochs))


def build_optimizer(model: torch.nn.Module, learning_rate: float, weight_decay: float):
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or "ln_" in name or "layernorm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


def build_scheduler(optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_loop(
    args: argparse.Namespace,
    accelerator: Accelerator,
    kind: str,
    model: torch.nn.Module,
    tokenizer: Any,
    optimizer,
    scheduler,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    output_dir: Path,
    total_steps: int,
) -> None:
    global_step = 0
    running_loss = 0.0
    running_count = 0
    model.train()

    while global_step < total_steps:
        for batch in train_loader:
            with accelerator.accumulate(model):
                loss = forward_loss(kind, model, batch)
                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += accelerator.gather_for_metrics(loss.detach()).mean().item()
            running_count += 1
            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process and global_step % args.log_every == 0:
                    print(
                        f"step={global_step} "
                        f"train_loss={running_loss / max(1, running_count):.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.6g}",
                        flush=True,
                    )
                    running_loss = 0.0
                    running_count = 0

                if val_loader is not None and global_step % args.eval_every == 0:
                    val_loss = evaluate(kind, model, val_loader, accelerator)
                    if accelerator.is_main_process:
                        print(f"step={global_step} val_loss={val_loss:.4f}", flush=True)

                if args.save_every > 0 and global_step % args.save_every == 0:
                    save_checkpoint(
                        kind, model, tokenizer, output_dir / f"checkpoint-{global_step}", accelerator
                    )

                if global_step >= total_steps:
                    break

    save_checkpoint(kind, model, tokenizer, output_dir / "checkpoint-final", accelerator)


def forward_loss(kind: str, model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    if kind == "hf":
        return model(input_ids=input_ids, labels=input_ids).loss
    _, loss = model(input_ids, labels)
    if loss is None:
        raise RuntimeError("Custom model did not return a loss.")
    return loss


@torch.no_grad()
def evaluate(
    kind: str,
    model: torch.nn.Module,
    val_loader: DataLoader,
    accelerator: Accelerator,
) -> float:
    model.eval()
    losses: list[torch.Tensor] = []
    for batch in val_loader:
        loss = forward_loss(kind, model, batch)
        losses.append(accelerator.gather_for_metrics(loss.detach()).mean())
    model.train()
    if not losses:
        return float("nan")
    return torch.stack(losses).mean().item()


def save_checkpoint(
    kind: str,
    model: torch.nn.Module,
    tokenizer: Any,
    checkpoint_dir: Path,
    accelerator: Accelerator,
) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    if hasattr(unwrapped, "_orig_mod"):
        unwrapped = unwrapped._orig_mod
    if kind == "hf":
        unwrapped.save_pretrained(
            checkpoint_dir,
            is_main_process=True,
            save_function=accelerator.save,
        )
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(checkpoint_dir)
        return

    if not isinstance(unwrapped, DecoderOnlyTransformer):
        raise TypeError("Expected DecoderOnlyTransformer for custom checkpoint saving.")
    config = asdict(unwrapped.config)
    (checkpoint_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    accelerator.save(
        {"model_state_dict": unwrapped.state_dict(), "config": config},
        checkpoint_dir / "model.pt",
    )
    if isinstance(tokenizer, VocabTokenizer):
        (checkpoint_dir / "vocab.json").write_text(
            json.dumps(tokenizer.token_to_id, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def write_training_args(
    output_dir: Path,
    args: argparse.Namespace,
    kind: str,
    block_size: int,
    train_blocks: int,
) -> None:
    payload = vars(args).copy()
    payload.update({"model_kind": kind, "block_size": block_size, "train_blocks": train_blocks})
    (output_dir / "training_args.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
