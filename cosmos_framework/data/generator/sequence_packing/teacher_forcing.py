# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Layout metadata helpers for teacher-forcing causal video training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence


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

    def to(self, device: torch.device | str) -> TeacherForcingLayout:
        """Return a copy with all tensor metadata moved to ``device``."""

        return replace(
            self,
            source_sequence_indexes=self.source_sequence_indexes.to(device=device),
            sample_ids=self.sample_ids.to(device=device),
            stream_ids=self.stream_ids.to(device=device),
            block_ids=self.block_ids.to(device=device),
            gen_query_indexes=self.gen_query_indexes.to(device=device),
            clean_token_indexes=self.clean_token_indexes.to(device=device),
            noisy_output_indexes=self.noisy_output_indexes.to(device=device),
        )


@dataclass
class TeacherForcingData:
    """Runtime data attached to a dual-stream ``PackedSequence``."""

    layout: TeacherForcingLayout
    clean_vision_tokens: list[torch.Tensor]

    def to_cuda(self) -> None:
        """Move clean payloads and layout tensors to CUDA/NPU in-place."""

        self.layout = self.layout.to("cuda")
        self.clean_vision_tokens = [token.cuda() for token in self.clean_vision_tokens]


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

    block_size = int(torch.randint(block_size_min, block_size_max + 1, (1,), generator=generator, device="cpu").item())
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
            raise ValueError(f"vision_token_shapes[{sample_id}] must contain exactly (T, H, W), got {vision_shape}")
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


def build_dense_teacher_forcing_gen_mask(
    layout: TeacherForcingLayout,
    *,
    max_mask_elements: int,
) -> torch.BoolTensor:
    """Build the reference GEN-query mask over all dual-stream KV tokens."""

    if max_mask_elements < 1:
        raise ValueError(f"max_mask_elements must be >= 1, got {max_mask_elements}")

    num_queries = layout.gen_query_indexes.numel()
    num_keys = layout.source_sequence_indexes.numel()
    num_elements = num_queries * num_keys
    if num_elements > max_mask_elements:
        raise ValueError(
            f"Dense teacher-forcing mask needs {num_elements} elements, exceeding max_mask_elements={max_mask_elements}"
        )

    query_indexes = layout.gen_query_indexes[:, None]
    query_sample_ids = layout.sample_ids[query_indexes]
    query_stream_ids = layout.stream_ids[query_indexes]
    query_block_ids = layout.block_ids[query_indexes]

    is_gen_query = (query_stream_ids == int(TeacherForcingStream.CLEAN)) | (
        query_stream_ids == int(TeacherForcingStream.NOISY)
    )
    if not bool(is_gen_query.all()):
        raise ValueError("gen_query_indexes must contain only CLEAN or NOISY GEN queries")

    key_sample_ids = layout.sample_ids[None, :]
    key_stream_ids = layout.stream_ids[None, :]
    key_block_ids = layout.block_ids[None, :]

    same_sample = query_sample_ids == key_sample_ids
    key_is_und = key_stream_ids == int(TeacherForcingStream.UND)
    key_is_clean = key_stream_ids == int(TeacherForcingStream.CLEAN)
    key_is_noisy = key_stream_ids == int(TeacherForcingStream.NOISY)

    inside_history = key_block_ids >= query_block_ids - layout.history_blocks
    clean_query_visible = key_is_clean & inside_history & (key_block_ids <= query_block_ids)
    noisy_query_visible = (key_is_clean & inside_history & (key_block_ids < query_block_ids)) | (
        key_is_noisy & (key_block_ids == query_block_ids)
    )

    allowed_by_stream = torch.where(
        query_stream_ids == int(TeacherForcingStream.CLEAN),
        clean_query_visible,
        noisy_query_visible,
    )
    return same_sample & (key_is_und | allowed_by_stream)


def _validate_teacher_forcing_packed_sequence(
    packed_sequence: PackedSequence,
) -> tuple[list[int], list[tuple[int, int, int]]]:
    """Validate the currently supported vision-only source packing contract."""

    if packed_sequence.teacher_forcing is not None:
        raise ValueError("packed_sequence already contains teacher-forcing data")
    if packed_sequence.vision is None:
        raise ValueError("teacher-forcing expansion requires vision data")
    if packed_sequence.action is not None:
        raise ValueError("teacher-forcing expansion does not support action data")
    if packed_sequence.sound is not None:
        raise ValueError("teacher-forcing expansion does not support sound data")
    if packed_sequence.is_image_batch:
        raise ValueError("teacher-forcing expansion requires a video batch")
    if packed_sequence.vision_item_split_lens or packed_sequence.control_weights is not None:
        raise ValueError("teacher-forcing expansion does not support multi-item vision packing")
    if packed_sequence.sequence_length != sum(packed_sequence.sample_lens):
        raise ValueError(
            "packed_sequence.sequence_length must equal sum(sample_lens), "
            f"got {packed_sequence.sequence_length} and {sum(packed_sequence.sample_lens)}"
        )
    if (
        packed_sequence.position_ids.ndim != 2
        or packed_sequence.position_ids.shape[-1] != packed_sequence.sequence_length
    ):
        raise ValueError(
            "packed_sequence.position_ids must have shape [axes, sequence_length], "
            f"got {tuple(packed_sequence.position_ids.shape)}"
        )

    vision = packed_sequence.vision
    num_samples = len(packed_sequence.sample_lens)
    if len(vision.token_shapes) != num_samples or len(vision.tokens) != num_samples:
        raise ValueError(
            "teacher-forcing expansion requires exactly one vision item per packed sample, "
            f"got {len(vision.token_shapes)} shapes, {len(vision.tokens)} payloads, and {num_samples} samples"
        )

    vision_token_shapes: list[tuple[int, int, int]] = []
    und_token_counts: list[int] = []
    expected_text_indexes: list[int] = []
    expected_vision_indexes: list[int] = []
    expected_split_lens: list[int] = []
    expected_attn_modes: list[str] = []
    sample_offset = 0
    for sample_id, (sample_len, token_shape) in enumerate(zip(packed_sequence.sample_lens, vision.token_shapes)):
        if len(token_shape) != 3:
            raise ValueError(f"vision.token_shapes[{sample_id}] must contain (T, H, W), got {token_shape}")
        num_frames, height, width = token_shape
        vision_count = num_frames * height * width
        und_count = sample_len - vision_count
        if und_count < 1:
            raise ValueError(
                f"sample {sample_id} must contain at least one UND token before its vision tokens, got {und_count}"
            )
        expected_text_indexes.extend(range(sample_offset, sample_offset + und_count))
        expected_vision_indexes.extend(range(sample_offset + und_count, sample_offset + sample_len))
        expected_split_lens.extend((und_count, vision_count))
        expected_attn_modes.extend(("causal", "full"))
        und_token_counts.append(und_count)
        vision_token_shapes.append((num_frames, height, width))
        sample_offset += sample_len

    if packed_sequence.text_indexes.tolist() != expected_text_indexes:
        raise ValueError("text and vision sequence indexes must form the expected per-sample UND/GEN partition")
    if vision.sequence_indexes.tolist() != expected_vision_indexes:
        raise ValueError("text and vision sequence indexes must form the expected per-sample UND/GEN partition")
    if packed_sequence.split_lens != expected_split_lens or packed_sequence.attn_modes != expected_attn_modes:
        raise ValueError(
            "source attention splits must alternate one causal UND split and one full vision split per sample"
        )
    if packed_sequence.text_ids.numel() != len(expected_text_indexes):
        raise ValueError(
            "text_ids must contain one payload per UND sequence index, "
            f"got {packed_sequence.text_ids.numel()} and {len(expected_text_indexes)}"
        )

    return und_token_counts, vision_token_shapes


def _build_source_to_stream_index(
    layout: TeacherForcingLayout,
    stream: TeacherForcingStream,
) -> dict[int, int]:
    stream_indexes = torch.nonzero(layout.stream_ids == int(stream), as_tuple=True)[0]
    return {int(layout.source_sequence_indexes[new_index]): int(new_index) for new_index in stream_indexes}


def _remap_indexes(indexes: torch.Tensor | None, source_to_new: dict[int, int], name: str) -> torch.Tensor | None:
    if indexes is None:
        return None
    try:
        remapped = [source_to_new[int(index)] for index in indexes]
    except KeyError as error:
        raise ValueError(
            f"{name} contains an index outside the supported source stream: {int(error.args[0])}"
        ) from error
    return torch.tensor(remapped, dtype=torch.long)


def expand_packed_sequence_for_teacher_forcing(
    packed_sequence: PackedSequence,
    *,
    clean_vision_tokens: Sequence[torch.Tensor],
    block_size: int,
    history_blocks: int,
) -> PackedSequence:
    """Return a vision-only packed sequence expanded into clean/noisy GEN streams."""

    from cosmos_framework.data.generator.sequence_packing.modality import ModalitySpan

    und_token_counts, vision_token_shapes = _validate_teacher_forcing_packed_sequence(packed_sequence)
    assert packed_sequence.vision is not None
    vision = packed_sequence.vision

    if len(clean_vision_tokens) != len(vision.tokens):
        raise ValueError(
            "clean_vision_tokens must contain one payload per noisy vision payload, "
            f"got {len(clean_vision_tokens)} and {len(vision.tokens)}"
        )
    for payload_id, (clean_token, noisy_token) in enumerate(zip(clean_vision_tokens, vision.tokens)):
        if clean_token.shape != noisy_token.shape:
            raise ValueError(
                f"clean/noisy vision payload {payload_id} must have the same shape, "
                f"got {tuple(clean_token.shape)} and {tuple(noisy_token.shape)}"
            )
        if clean_token.dtype != noisy_token.dtype:
            raise ValueError(
                f"clean/noisy vision payload {payload_id} must have the same dtype, "
                f"got {clean_token.dtype} and {noisy_token.dtype}"
            )
        if clean_token.device != noisy_token.device:
            raise ValueError(
                f"clean/noisy vision payload {payload_id} must be on the same device, "
                f"got {clean_token.device} and {noisy_token.device}"
            )

    layout = build_teacher_forcing_layout(
        und_token_counts=und_token_counts,
        vision_token_shapes=vision_token_shapes,
        block_size=block_size,
        history_blocks=history_blocks,
    )
    source_to_und = _build_source_to_stream_index(layout, TeacherForcingStream.UND)
    source_to_noisy = _build_source_to_stream_index(layout, TeacherForcingStream.NOISY)

    remapped_text_indexes = _remap_indexes(packed_sequence.text_indexes, source_to_und, "text_indexes")
    remapped_ce_loss_indexes = _remap_indexes(
        packed_sequence.ce_loss_indexes,
        source_to_und,
        "ce_loss_indexes",
    )
    remapped_vision_indexes = _remap_indexes(vision.sequence_indexes, source_to_noisy, "vision.sequence_indexes")
    remapped_mse_loss_indexes = _remap_indexes(
        vision.mse_loss_indexes,
        source_to_noisy,
        "vision.mse_loss_indexes",
    )
    assert remapped_text_indexes is not None
    assert remapped_vision_indexes is not None
    assert remapped_mse_loss_indexes is not None

    remapped_spans: list[ModalitySpan] = []
    for span in vision.spans:
        span_indexes = [
            source_to_noisy[index] for index in range(span.sequence_start, span.sequence_start + span.sequence_len)
        ]
        if span_indexes != list(range(span_indexes[0], span_indexes[0] + span.sequence_len)):
            raise ValueError(f"vision span at source index {span.sequence_start} is not contiguous after remapping")
        remapped_spans.append(replace(span, sequence_start=span_indexes[0]))

    expanded_vision = replace(
        vision,
        sequence_indexes=remapped_vision_indexes,
        mse_loss_indexes=remapped_mse_loss_indexes,
        spans=remapped_spans,
        tokens=list(vision.tokens),
        token_shapes=list(vision.token_shapes),
        condition_mask=list(vision.condition_mask),
        noisy_frame_indexes=list(vision.noisy_frame_indexes),
    )
    teacher_forcing = TeacherForcingData(
        layout=layout,
        clean_vision_tokens=list(clean_vision_tokens),
    )
    return replace(
        packed_sequence,
        sample_lens=list(layout.sample_lens),
        split_lens=list(layout.split_lens),
        attn_modes=list(layout.attn_modes),
        uses_single_timestep=False,
        sequence_length=sum(layout.sample_lens),
        text_indexes=remapped_text_indexes,
        position_ids=packed_sequence.position_ids[:, layout.source_sequence_indexes],
        ce_loss_indexes=remapped_ce_loss_indexes,
        vision=expanded_vision,
        teacher_forcing=teacher_forcing,
    )


def select_teacher_forcing_noisy_outputs(
    packed_output: torch.Tensor,
    layout: TeacherForcingLayout,
) -> torch.Tensor:
    """Select the complete noisy vision stream in original packed order."""

    expected_length = layout.source_sequence_indexes.numel()
    if packed_output.ndim < 1 or packed_output.shape[0] != expected_length:
        raise ValueError(
            "packed_output first dimension must equal the expanded sequence length, "
            f"got shape {tuple(packed_output.shape)} and expected {expected_length}"
        )
    return packed_output.index_select(0, layout.noisy_output_indexes.to(device=packed_output.device))
