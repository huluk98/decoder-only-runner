from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


HEAD_PREFIXES = ("classifier", "lm_head", "qa_outputs", "score", "cls")
HEAD_EXACT_PARTS = (
    "lm_head",
    "classifier",
    "cls",
    "score",
    "output_head",
    "output_heads",
    "prediction_head",
    "predictions",
    "language_model_head",
)
HEAD_SUBSTRINGS = (
    "response_classifier",
    "response_projection",
    "final_response_projection",
    "final_projection",
    "prediction_head",
    "language_model_head",
)


@dataclass(frozen=True)
class LinearSparsityConfig:
    prune_output_heads: bool = False
    global_pruning: bool = True
    regrowth: bool = False


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
    if parts and parts[-1] in HEAD_EXACT_PARTS:
        return True
    return any(fragment in normalized for fragment in HEAD_SUBSTRINGS)


def count_zeros(tensor: torch.Tensor) -> int:
    return int((tensor.detach() == 0).sum().item())


def parameter_sparsity(parameters: list[tuple[str, torch.Tensor | nn.Parameter]]) -> dict[str, int | float]:
    numel = 0
    zeros = 0
    for _name, parameter in parameters:
        tensor = parameter.detach().cpu()
        numel += int(tensor.numel())
        zeros += count_zeros(tensor)
    return {"numel": numel, "zeros": zeros, "sparsity": zeros / numel if numel else 0.0}


def collect_linear_module_report(
    model: nn.Module,
    prune_output_heads: bool = False,
) -> tuple[list[tuple[str, nn.Linear]], list[dict[str, Any]]]:
    modules: list[tuple[str, nn.Linear]] = []
    skipped: list[dict[str, Any]] = []
    for raw_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        name = canonical_module_name(raw_name)
        zeros = count_zeros(module.weight)
        numel = int(module.weight.numel())
        item = {
            "module_name": name,
            "parameter_count": numel,
            "zero_count": zeros,
            "sparsity_actual": zeros / numel if numel else 0.0,
        }
        if not prune_output_heads and is_output_head_module(name):
            skipped.append({**item, "reason": "output_head_excluded"})
            continue
        modules.append((name, module))
    return modules, skipped


def selected_linear_tensors(modules: list[tuple[str, nn.Linear]]) -> list[dict[str, Any]]:
    rows = []
    for name, module in modules:
        zeros = count_zeros(module.weight)
        numel = int(module.weight.numel())
        rows.append(
            {
                "name": name,
                "shape": list(module.weight.shape),
                "numel": numel,
                "zeros": zeros,
                "sparsity": zeros / numel if numel else 0.0,
            }
        )
    return rows


def current_linear_sparsity_summary(
    model: nn.Module,
    config: LinearSparsityConfig | None = None,
    target_sparsity: float | None = None,
    method: str | None = None,
) -> dict[str, Any]:
    config = config or LinearSparsityConfig()
    modules, skipped = collect_linear_module_report(model, config.prune_output_heads)
    target_stats = parameter_sparsity([(f"{name}.weight", module.weight) for name, module in modules])
    whole_stats = parameter_sparsity(list(model.named_parameters()))
    return {
        "prune_scope": "linear_weights",
        "prune_method": method,
        "target_sparsity": target_sparsity,
        "prune_output_heads": bool(config.prune_output_heads),
        "global_pruning": bool(config.global_pruning),
        "regrowth": bool(config.regrowth),
        "targeted_linear_parameters": int(target_stats["numel"]),
        "targeted_linear_zeros": int(target_stats["zeros"]),
        "targeted_linear_sparsity_actual": float(target_stats["sparsity"]),
        "whole_model_parameters": int(whole_stats["numel"]),
        "whole_model_zeros": int(whole_stats["zeros"]),
        "whole_model_sparsity_actual": float(whole_stats["sparsity"]),
        "target_linear_module_count": len(modules),
        "skipped_linear_module_count": len(skipped),
        "selected_linear_tensors": selected_linear_tensors(modules),
        "skipped_linear_modules": skipped,
    }


def _keep_count(numel: int, sparsity: float) -> int:
    if not 0.0 <= float(sparsity) <= 1.0:
        raise ValueError("sparsity must be between 0.0 and 1.0")
    return max(0, min(numel, int(round(numel * (1.0 - float(sparsity))))))


def global_unstructured_masks(
    modules: list[tuple[str, nn.Linear]],
    scores_by_name: dict[str, torch.Tensor],
    target_sparsity: float,
) -> dict[str, torch.Tensor]:
    names: list[str] = []
    scores: list[torch.Tensor] = []
    shapes: dict[str, torch.Size] = {}
    for name, module in modules:
        score = scores_by_name[name].detach().float().reshape(-1)
        names.append(name)
        scores.append(score)
        shapes[name] = module.weight.shape

    if not scores:
        return {}

    all_scores = torch.cat(scores)
    keep = _keep_count(int(all_scores.numel()), target_sparsity)
    flat_mask = torch.zeros_like(all_scores, dtype=torch.bool)
    if keep > 0:
        keep_indices = torch.topk(all_scores, k=keep, largest=True, sorted=False).indices
        flat_mask[keep_indices] = True

    masks: dict[str, torch.Tensor] = {}
    offset = 0
    for name, score in zip(names, scores):
        size = int(score.numel())
        masks[name] = flat_mask[offset : offset + size].reshape(shapes[name])
        offset += size
    return masks


def magnitude_scores(modules: list[tuple[str, nn.Linear]]) -> dict[str, torch.Tensor]:
    return {name: module.weight.detach().abs().cpu() for name, module in modules}


def gradient_scores(modules: list[tuple[str, nn.Linear]]) -> dict[str, torch.Tensor]:
    scores = {}
    for name, module in modules:
        grad = module.weight.grad
        if grad is None:
            scores[name] = module.weight.detach().abs().cpu()
        else:
            scores[name] = (module.weight.detach() * grad.detach()).abs().cpu()
    return scores


def nvidia24_masks(modules: list[tuple[str, nn.Linear]]) -> dict[str, torch.Tensor]:
    masks: dict[str, torch.Tensor] = {}
    for name, module in modules:
        weight = module.weight.detach().abs().cpu()
        original_shape = weight.shape
        if weight.ndim != 2 or weight.shape[-1] % 4 != 0:
            masks[name] = torch.ones_like(weight, dtype=torch.bool)
            continue
        grouped = weight.reshape(-1, 4)
        keep = torch.ones_like(grouped, dtype=torch.bool)
        prune_idx = torch.topk(grouped, k=2, largest=False, sorted=False).indices
        keep.scatter_(1, prune_idx, False)
        masks[name] = keep.reshape(original_shape)
    return masks


@torch.no_grad()
def apply_masks(model: nn.Module, masks: dict[str, torch.Tensor]) -> None:
    module_lookup = {canonical_module_name(name): module for name, module in model.named_modules()}
    for name, mask in masks.items():
        module = module_lookup[name]
        module.weight.mul_(mask.to(device=module.weight.device, dtype=module.weight.dtype))
