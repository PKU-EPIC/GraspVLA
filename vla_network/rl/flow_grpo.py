"""Small, framework-independent Flow-GRPO utilities.

The rollout collector is deliberately separate from the Hugging Face Trainer: a flow
transition is a Gaussian policy action and its log probability is part of the data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class FlowTransition:
    x_t: torch.Tensor
    next_x: torch.Tensor
    time: torch.Tensor
    old_log_prob: torch.Tensor
    reward: float
    group_id: int


def group_relative_advantages(rewards: torch.Tensor, group_ids: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute standardized within-task advantages; uniform groups get zero signal."""
    advantages = torch.zeros_like(rewards, dtype=torch.float32)
    for group in torch.unique(group_ids):
        mask = group_ids == group
        values = rewards[mask].float()
        if values.numel() < 2:
            continue
        std = values.std(unbiased=False)
        if std <= eps:
            continue
        advantages[mask] = (values - values.mean()) / (std + eps)
    return advantages


def clipped_surrogate_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """PPO clipped loss for flow transitions, with detached rollout statistics."""
    ratio = torch.exp(new_log_prob - old_log_prob.detach())
    adv = advantages.detach().to(ratio.dtype)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv
    valid = adv.abs() > 0
    if valid.any():
        loss = -torch.minimum(unclipped[valid], clipped[valid]).mean()
    else:
        loss = new_log_prob.sum() * 0.0
    with torch.no_grad():
        clip_fraction = ((ratio - 1.0).abs() > clip_range)[valid].float().mean() if valid.any() else ratio.new_zeros(())
    return loss, {
        "loss": float(loss.detach()),
        "clip_fraction": float(clip_fraction),
        "valid_fraction": float(valid.float().mean()),
        "mean_ratio": float(ratio.detach().mean()),
    }


def flatten_transition_log_probs(log_probs: Sequence[torch.Tensor]) -> torch.Tensor:
    """Sum flow-step log probabilities into one log probability per episode/chunk."""
    return torch.stack(list(log_probs), dim=0).sum(dim=0)
