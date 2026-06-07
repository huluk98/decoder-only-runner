from __future__ import annotations

import argparse

import torch

from decoder_only.loader import load_model_and_tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate text from a decoder-only model.")
    parser.add_argument("--model-path", required=True, help="Path to checkpoint/model directory.")
    parser.add_argument("--prompt", default="", help="Prompt text to continue.")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default=None, help="cpu, mps, cuda, or leave unset for auto.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of sampling.")
    return parser


def _generate_hf(args: argparse.Namespace, model, tokenizer, device: torch.device) -> str:
    inputs = tokenizer(args.prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=not args.greedy,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            pad_token_id=pad_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def _generate_custom(args: argparse.Namespace, model, tokenizer, device: torch.device) -> str:
    token_ids = tokenizer.encode(args.prompt)
    if not token_ids:
        token_ids = [0]
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)
    output = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        do_sample=not args.greedy,
    )
    return tokenizer.decode(output[0].tolist())


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    kind, model, tokenizer, device = load_model_and_tokenizer(args.model_path, args.device)
    if kind == "hf":
        text = _generate_hf(args, model, tokenizer, device)
    else:
        text = _generate_custom(args, model, tokenizer, device)
    print(text)


if __name__ == "__main__":
    main()
