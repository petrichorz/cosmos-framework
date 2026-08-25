# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tensor and geometry helpers for GenKVCache causal video inference."""

import torch


def causal_total_latent_frames(num_blocks: int, block_size: int) -> int:
    """Return singleton C0 plus the requested generated blocks."""

    if num_blocks < 1:
        raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    return 1 + num_blocks * block_size


def causal_generated_block_span(block_id: int, block_size: int) -> tuple[int, int]:
    """Return the absolute latent-frame span for a generated block."""

    if block_id < 1:
        raise ValueError(f"generated block_id must be >= 1, got {block_id}")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    start = 1 + (block_id - 1) * block_size
    return start, start + block_size


def require_single_vision_5d(vision_tokens: list[torch.Tensor] | None) -> torch.Tensor:
    if vision_tokens is None or len(vision_tokens) != 1:
        raise ValueError("causal inference requires exactly one vision latent item")
    latent = vision_tokens[0]
    if latent.ndim != 5 or latent.shape[0] != 1:
        raise ValueError(f"causal inference requires vision latent shape [1,C,T,H,W], got {tuple(latent.shape)}")
    return latent


def flatten_vision_5d(latent: torch.Tensor) -> torch.Tensor:
    if latent.ndim != 5 or latent.shape[0] != 1:
        raise ValueError(f"expected [1,C,T,H,W], got {tuple(latent.shape)}")
    return latent.reshape(-1)


def unflatten_vision_5d(flat: torch.Tensor, shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
    if len(shape) != 5 or shape[0] != 1:
        raise ValueError(f"expected target shape [1,C,T,H,W], got {tuple(shape)}")
    expected = 1
    for dimension in shape:
        expected *= int(dimension)
    if flat.numel() != expected:
        raise ValueError(f"flat latent contains {flat.numel()} values, expected {expected}")
    return flat.reshape(shape)


def slice_vision_time(latent: torch.Tensor, start: int, end: int) -> torch.Tensor:
    if latent.ndim != 5 or latent.shape[0] != 1:
        raise ValueError(f"expected [1,C,T,H,W], got {tuple(latent.shape)}")
    if start < 0 or end <= start or end > latent.shape[2]:
        raise ValueError(f"invalid temporal slice [{start}:{end}] for T={latent.shape[2]}")
    return latent[:, :, start:end, :, :]


def concat_vision_time(parts: list[torch.Tensor]) -> torch.Tensor:
    if not parts:
        raise ValueError("cannot concatenate an empty vision history")
    reference = parts[0]
    if reference.ndim != 5 or reference.shape[0] != 1:
        raise ValueError(f"expected [1,C,T,H,W], got {tuple(reference.shape)}")
    for part in parts[1:]:
        if part.ndim != 5 or part.shape[0] != 1:
            raise ValueError(f"expected [1,C,T,H,W], got {tuple(part.shape)}")
        if part.shape[1] != reference.shape[1] or part.shape[3:] != reference.shape[3:]:
            raise ValueError("all causal vision parts must share channel and spatial dimensions")
    return torch.cat(parts, dim=2)
