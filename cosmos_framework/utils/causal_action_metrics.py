# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Detached, sample-weighted causal action metrics (independent of backward)."""

import torch

MODE_NAMES = ("policy", "inverse_dynamics", "forward_dynamics")
MODE_TAGS = ("policy", "id", "fd")


@torch.no_grad()
def collect_mode_metrics(modes, modalities, device):
    """Return fixed [mode, (loss sum, valid sample count)] statistics.

    Each modality supplies original batch indices, existing per-sample
    losses, condition masks, and its training loss scale. Auxiliary losses are
    excluded because they have no per-sample attribution.
    """
    mode_ids = torch.tensor([MODE_NAMES.index(mode) for mode in modes], device=device)
    values = torch.zeros(len(modes), device=device, dtype=torch.float32)
    active = torch.zeros(len(modes), device=device, dtype=torch.bool)
    for indices, losses, masks, scale in modalities:
        if indices is None or not (len(indices) == len(losses) == len(masks)):
            raise ValueError("Mode metrics require aligned sample indices, losses and masks")
        if len(set(indices)) != len(indices) or any(i < 0 or i >= len(modes) for i in indices):
            raise ValueError("Invalid or duplicate mode metric sample indices")
        for index, value, mask in zip(indices, losses, masks, strict=True):
            valid = (mask < 1).any()
            values[index] += torch.where(valid, value.detach().float() * scale, 0.0)
            active[index] |= valid
    # 没有预测目标的样本不计入分母；缺失模式保留 count=0，日志层不补零。
    stats = torch.zeros((len(MODE_NAMES), 2), device=device, dtype=torch.float32)
    stats[:, 0].scatter_add_(0, mode_ids, torch.where(active, values, 0.0))
    stats[:, 1].scatter_add_(0, mode_ids, active.float())
    return stats


def mode_metric_means(stats):
    """Convert rank-local window statistics to sparse scalar metrics."""
    return {
        f"train_mode/{tag}_loss": total / count
        for tag, (total, count) in zip(MODE_TAGS, stats.detach().cpu().tolist(), strict=True)
        if count > 0
    }
