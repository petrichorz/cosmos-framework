# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Layout metadata helpers for teacher-forcing causal video training."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

import torch


class TeacherForcingStream(IntEnum):
    """Token stream identifiers used by teacher-forcing attention."""

    UND = -1
    CLEAN = 0
    NOISY = 1


@dataclass(frozen=True)
class TeacherForcingLayout:
    """Immutable geometry shared by packing, attention, and output recovery."""

    block_size: int
    history_blocks: int
    original_sample_lens: tuple[int, ...]
    sample_lens: tuple[int, ...]
    split_lens: tuple[int, ...]
    attn_modes: tuple[str, ...]
    source_sequence_indexes: torch.LongTensor
    sample_ids: torch.LongTensor
    stream_ids: torch.LongTensor
    block_ids: torch.LongTensor
    gen_query_indexes: torch.LongTensor
    clean_token_indexes: torch.LongTensor
    noisy_output_indexes: torch.LongTensor


def _validate_inclusive_range(name: str, minimum: int, maximum: int) -> None:
    if minimum < 1:
        raise ValueError(f"{name}_min must be >= 1, got {minimum}")
    if maximum < 1:
        raise ValueError(f"{name}_max must be >= 1, got {maximum}")
    if minimum > maximum:
        raise ValueError(f"{name} range must satisfy min <= max, got min={minimum}, max={maximum}")


def sample_teacher_forcing_parameters(
    *,
    block_size_min: int = 1,
    block_size_max: int = 4,
    history_blocks_min: int = 1,
    history_blocks_max: int = 32,
    generator: torch.Generator | None = None,
) -> tuple[int, int]:
    """Sample one block size and history window shared by the whole forward."""

    _validate_inclusive_range("block_size", block_size_min, block_size_max)
    _validate_inclusive_range("history_blocks", history_blocks_min, history_blocks_max)

    block_size = int(
        torch.randint(block_size_min, block_size_max + 1, (1,), generator=generator, device="cpu").item()
    )
    history_blocks = int(
        torch.randint(history_blocks_min, history_blocks_max + 1, (1,), generator=generator, device="cpu").item()
    )
    return block_size, history_blocks


def build_teacher_forcing_layout(
    *,
    und_token_counts: Sequence[int],
    vision_token_shapes: Sequence[tuple[int, int, int]],
    block_size: int,
    history_blocks: int,
) -> TeacherForcingLayout:
    """Build batch metadata for ``[UND | clean vision | noisy vision]`` samples."""

    if len(und_token_counts) != len(vision_token_shapes):
        raise ValueError(
            "und_token_counts and vision_token_shapes must contain the same number of samples, "
            f"got {len(und_token_counts)} and {len(vision_token_shapes)}"
        )
    if not und_token_counts:
        raise ValueError("teacher-forcing layout cannot be built from an empty batch")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    if history_blocks < 1:
        raise ValueError(f"history_blocks must be >= 1, got {history_blocks}")

    original_sample_lens: list[int] = []
    sample_lens: list[int] = []
    split_lens: list[int] = []
    attn_modes: list[str] = []
    source_sequence_indexes: list[int] = []
    sample_ids: list[int] = []
    stream_ids: list[int] = []
    block_ids: list[int] = []
    gen_query_indexes: list[int] = []
    clean_token_indexes: list[int] = []
    noisy_output_indexes: list[int] = []

    original_offset = 0
    new_offset = 0
    for sample_id, (und_count, vision_shape) in enumerate(zip(und_token_counts, vision_token_shapes)):
        if und_count < 1:
            raise ValueError(f"und_token_counts[{sample_id}] must be >= 1, got {und_count}")
        if len(vision_shape) != 3:
            raise ValueError(
                f"vision_token_shapes[{sample_id}] must contain exactly (T, H, W), got {vision_shape}"
            )
        num_frames, height, width = vision_shape
        if num_frames < 1 or height < 1 or width < 1:
            raise ValueError(f"vision_token_shapes[{sample_id}] must be positive, got {vision_shape}")

        spatial_tokens = height * width
        vision_count = num_frames * spatial_tokens
        original_sample_len = und_count + vision_count
        new_sample_len = und_count + 2 * vision_count

        und_source = list(range(original_offset, original_offset + und_count))
        vision_source = list(range(original_offset + und_count, original_offset + original_sample_len))
        vision_frame_ids = torch.arange(num_frames, dtype=torch.long).repeat_interleave(spatial_tokens)
        vision_block_ids = torch.div(vision_frame_ids, block_size, rounding_mode="floor").tolist()

        clean_start = new_offset + und_count
        noisy_start = clean_start + vision_count
        new_sample_end = new_offset + new_sample_len

        original_sample_lens.append(original_sample_len)
        sample_lens.append(new_sample_len)
        split_lens.extend((und_count, 2 * vision_count))
        attn_modes.extend(("causal", "full"))
        source_sequence_indexes.extend(und_source + vision_source + vision_source)
        sample_ids.extend([sample_id] * new_sample_len)
        stream_ids.extend(
            [int(TeacherForcingStream.UND)] * und_count
            + [int(TeacherForcingStream.CLEAN)] * vision_count
            + [int(TeacherForcingStream.NOISY)] * vision_count
        )
        block_ids.extend([-1] * und_count + vision_block_ids + vision_block_ids)
        gen_query_indexes.extend(range(clean_start, new_sample_end))
        clean_token_indexes.extend(range(clean_start, noisy_start))
        noisy_output_indexes.extend(range(noisy_start, new_sample_end))

        original_offset += original_sample_len
        new_offset = new_sample_end

    return TeacherForcingLayout(
        block_size=block_size,
        history_blocks=history_blocks,
        original_sample_lens=tuple(original_sample_lens),
        sample_lens=tuple(sample_lens),
        split_lens=tuple(split_lens),
        attn_modes=tuple(attn_modes),
        source_sequence_indexes=torch.tensor(source_sequence_indexes, dtype=torch.long),
        sample_ids=torch.tensor(sample_ids, dtype=torch.long),
        stream_ids=torch.tensor(stream_ids, dtype=torch.long),
        block_ids=torch.tensor(block_ids, dtype=torch.long),
        gen_query_indexes=torch.tensor(gen_query_indexes, dtype=torch.long),
        clean_token_indexes=torch.tensor(clean_token_indexes, dtype=torch.long),
        noisy_output_indexes=torch.tensor(noisy_output_indexes, dtype=torch.long),
    )
