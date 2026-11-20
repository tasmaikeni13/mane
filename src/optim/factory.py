"""Optimizer + LR-schedule construction for the Lion-MANE study.

Three arms, one benchmark (WikiText-103, ~39M decoder LM):

    adamw      — the target to beat (tuned baseline)
    lion       — vanilla Lion (the unstable animal)
    lion_mane  — Lion with the MANE gate (ours)

Defaults are starting points confirmed by the pilot lr sweep; the headline
comparison always uses the pilot-selected lr per arm. Lion-family lrs are
~3–6× smaller than AdamW's with weight decay scaled up to conserve lr·wd,
following Chen et al. 2023.
"""

import math

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from optim.lion import Lion
from optim.mane import MANE

DEFAULTS = {
    "adamw": dict(lr=6e-4, weight_decay=0.1, betas=(0.9, 0.95), warmup=2000),
    "lion":  dict(lr=1e-4, weight_decay=0.6, betas=(0.95, 0.98), warmup=2000),
    # MANE: Micro-batch Agreement Noise Equilibration.
    # Spends a noise budget equal to vanilla Lion (budget_ratio = 1.0) proportionally
    # to coordinate reliability (z-score), with trust capped at 3.0.
    "mane":  dict(lr=1.4e-4, weight_decay=0.43, betas=(0.95, 0.98), warmup=2000,
                  beta_v=0.99, budget_ratio=1.0, trust_cap=3.0),
}

FOLD_BASED = {"mane"}  # optimizers that consume two fold gradients per step


def default_hparams(optimizer: str) -> dict:
    return dict(DEFAULTS[optimizer])


def make_optimizer(name: str, params, lr: float, weight_decay: float,
                   betas=(0.9, 0.999), beta_v: float = 0.99,
                   budget_ratio: float = 1.0, trust_cap: float = 3.0):
    name = name.lower()
    if name == "adamw":
        return optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
    if name == "lion":
        return Lion(params, lr=lr, betas=betas, weight_decay=weight_decay)
    if name == "mane":
        return MANE(params, lr=lr, betas=betas, beta_v=beta_v,
                    weight_decay=weight_decay, budget_ratio=budget_ratio,
                    trust_cap=trust_cap)
    raise ValueError(f"Unknown optimizer: {name!r} (expected adamw|lion|mane)")


def make_scheduler(optimizer, total_steps: int, warmup_steps: int = 0):
    """Linear warmup for ``warmup_steps`` then cosine decay to zero."""
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)
