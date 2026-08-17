# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Pure configuration and packing helpers for Scheme-B causal training."""

from typing import Protocol

import torch

from cosmos_framework.data.generator.sequence_packing import (
    PackedSequence,
    expand_packed_sequence_for_teacher_forcing,
    sample_teacher_forcing_parameters,
)


class _ParallelismConfig(Protocol):
    context_parallel_shard_degree: int


class TeacherForcingConfig(Protocol):
    causal_training_strategy: str
    vision_gen: bool
    action_gen: bool
    sound_gen: bool
    video_temporal_causal: bool
    teacher_forcing_block_size_min: int
    teacher_forcing_block_size_max: int
    teacher_forcing_history_blocks_min: int
    teacher_forcing_history_blocks_max: int
    teacher_forcing_max_sequence_length: int | None
    teacher_forcing_dense_mode: str
    parallelism: _ParallelismConfig


def validate_teacher_forcing_config(config: TeacherForcingConfig) -> None:
    """Validate the supported first vertical slice before model construction."""

    if config.causal_training_strategy != "teacher_forcing":
        raise ValueError(
            "OmniMoTCausalModel requires causal_training_strategy='teacher_forcing', "
            f"got {config.causal_training_strategy!r}"
        )
    if not config.vision_gen:
        raise ValueError("OmniMoTCausalModel requires vision_gen=True")
    if config.action_gen:
        raise ValueError("OmniMoTCausalModel teacher forcing does not support action_gen=True")
    if config.sound_gen:
        raise ValueError("OmniMoTCausalModel teacher forcing does not support sound_gen=True")
    if config.video_temporal_causal:
        raise ValueError("teacher-forcing block causality requires video_temporal_causal=False")
    if config.parallelism.context_parallel_shard_degree != 1:
        raise ValueError("teacher-forcing Dense attention does not support context parallel")
    if (
        config.teacher_forcing_block_size_min < 1
        or config.teacher_forcing_block_size_min > config.teacher_forcing_block_size_max
    ):
        raise ValueError(
            "teacher-forcing block_size range must satisfy 1 <= min <= max, "
            f"got {config.teacher_forcing_block_size_min}..{config.teacher_forcing_block_size_max}"
        )
    if (
        config.teacher_forcing_history_blocks_min < 1
        or config.teacher_forcing_history_blocks_min > config.teacher_forcing_history_blocks_max
    ):
        raise ValueError(
            "teacher-forcing history_blocks range must satisfy 1 <= min <= max, "
            f"got {config.teacher_forcing_history_blocks_min}..{config.teacher_forcing_history_blocks_max}"
        )
    if config.teacher_forcing_max_sequence_length is None or config.teacher_forcing_max_sequence_length < 1:
        raise ValueError("teacher_forcing_max_sequence_length must be explicitly configured to a positive integer")
    if config.teacher_forcing_dense_mode not in {"global", "per_sample"}:
        raise ValueError(
            "teacher_forcing_dense_mode must be 'global' or 'per_sample', "
            f"got {config.teacher_forcing_dense_mode!r}"
        )


def expand_teacher_forcing_training_sequence(
    packed_sequence: PackedSequence,
    *,
    clean_vision_tokens: list[torch.Tensor],
    config: TeacherForcingConfig,
    generator: torch.Generator | None = None,
) -> PackedSequence:
    """Sample batch-shared S/K and expand an already-noised video sequence."""

    validate_teacher_forcing_config(config)
    if packed_sequence.is_image_batch:
        raise ValueError("teacher-forcing causal training currently requires a video batch")

    block_size, history_blocks = sample_teacher_forcing_parameters(
        block_size_min=config.teacher_forcing_block_size_min,
        block_size_max=config.teacher_forcing_block_size_max,
        history_blocks_min=config.teacher_forcing_history_blocks_min,
        history_blocks_max=config.teacher_forcing_history_blocks_max,
        generator=generator,
    )
    return expand_packed_sequence_for_teacher_forcing(
        packed_sequence,
        clean_vision_tokens=clean_vision_tokens,
        block_size=block_size,
        history_blocks=history_blocks,
    )
