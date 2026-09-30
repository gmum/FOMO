from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
from peft import LoraConfig


DEFAULT_HUNYUAN_ERASE_TARGET_MODULES = ["attn.add_q_proj", "attn.add_k_proj"]


def freeze_model(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad_(False)


def find_matching_module_names(model: torch.nn.Module, target_modules: Iterable[str]) -> List[str]:
    targets = tuple(target_modules)
    matches = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(name.endswith(target) for target in targets):
            matches.append(name)
    return matches


def add_hunyuan_erase_lora(
    transformer: torch.nn.Module,
    rank: int = 8,
    alpha: int | None = None,
    dropout: float = 0.0,
    target_modules: Iterable[str] | None = None,
    adapter_name: str = "erase_lora",
) -> Tuple[torch.nn.Module, List[str]]:
    target_modules = list(target_modules or DEFAULT_HUNYUAN_ERASE_TARGET_MODULES)
    matches = find_matching_module_names(transformer, target_modules)
    if not matches:
        raise ValueError(
            "No Hunyuan LoRA target modules were found. "
            f"Requested suffixes: {target_modules}"
        )

    freeze_model(transformer)

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha if alpha is not None else rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
    )
    transformer.add_adapter(lora_config, adapter_name=adapter_name)
    return transformer, matches


def trainable_parameter_summary(model: torch.nn.Module) -> dict:
    trainable = 0
    total = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    pct = 100.0 * trainable / total if total else 0.0
    return {"trainable": trainable, "total": total, "percent": pct}


def iter_trainable_parameters(model: torch.nn.Module):
    return (param for param in model.parameters() if param.requires_grad)
