# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Layout metadata helpers for teacher-forcing causal video training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import IntEnum
from pathlib import Path
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
    spatial_token_counts: tuple[int, ...]
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
    spatial_token_counts: list[int] = []
    sample_lens: list[int] = []
    split_lens: list[int] = []
    attn_modes: list[str] = []
    source_sequence_index_chunks: list[torch.Tensor] = []
    sample_id_chunks: list[torch.Tensor] = []
    stream_id_chunks: list[torch.Tensor] = []
    block_id_chunks: list[torch.Tensor] = []
    gen_query_index_chunks: list[torch.Tensor] = []
    clean_token_index_chunks: list[torch.Tensor] = []
    noisy_output_index_chunks: list[torch.Tensor] = []

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
        spatial_token_counts.append(spatial_tokens)
        vision_count = num_frames * spatial_tokens
        original_sample_len = und_count + vision_count
        new_sample_len = und_count + 2 * vision_count

        und_source = torch.arange(original_offset, original_offset + und_count, dtype=torch.long)
        vision_source = torch.arange(
            original_offset + und_count,
            original_offset + original_sample_len,
            dtype=torch.long,
        )
        vision_block_ids = torch.div(
            torch.arange(num_frames, dtype=torch.long),
            block_size,
            rounding_mode="floor",
        ).repeat_interleave(spatial_tokens)

        clean_start = new_offset + und_count
        noisy_start = clean_start + vision_count
        new_sample_end = new_offset + new_sample_len

        original_sample_lens.append(original_sample_len)
        sample_lens.append(new_sample_len)
        split_lens.extend((und_count, 2 * vision_count))
        attn_modes.extend(("causal", "full"))
        source_sequence_index_chunks.append(torch.cat((und_source, vision_source, vision_source)))
        sample_id_chunks.append(torch.full((new_sample_len,), sample_id, dtype=torch.long))
        stream_id_chunks.append(
            torch.cat(
                (
                    torch.full((und_count,), int(TeacherForcingStream.UND), dtype=torch.long),
                    torch.full((vision_count,), int(TeacherForcingStream.CLEAN), dtype=torch.long),
                    torch.full((vision_count,), int(TeacherForcingStream.NOISY), dtype=torch.long),
                )
            )
        )
        block_id_chunks.append(
            torch.cat(
                (
                    torch.full((und_count,), -1, dtype=torch.long),
                    vision_block_ids,
                    vision_block_ids,
                )
            )
        )
        gen_query_index_chunks.append(torch.arange(clean_start, new_sample_end, dtype=torch.long))
        clean_token_index_chunks.append(torch.arange(clean_start, noisy_start, dtype=torch.long))
        noisy_output_index_chunks.append(torch.arange(noisy_start, new_sample_end, dtype=torch.long))

        original_offset += original_sample_len
        new_offset = new_sample_end

    return TeacherForcingLayout(
        block_size=block_size,
        history_blocks=history_blocks,
        spatial_token_counts=tuple(spatial_token_counts),
        original_sample_lens=tuple(original_sample_lens),
        sample_lens=tuple(sample_lens),
        split_lens=tuple(split_lens),
        attn_modes=tuple(attn_modes),
        source_sequence_indexes=torch.cat(source_sequence_index_chunks),
        sample_ids=torch.cat(sample_id_chunks),
        stream_ids=torch.cat(stream_id_chunks),
        block_ids=torch.cat(block_id_chunks),
        gen_query_indexes=torch.cat(gen_query_index_chunks),
        clean_token_indexes=torch.cat(clean_token_index_chunks),
        noisy_output_indexes=torch.cat(noisy_output_index_chunks),
    )


def build_dense_teacher_forcing_gen_mask(
    layout: TeacherForcingLayout,
    *,
    max_sequence_length: int,
) -> torch.BoolTensor:
    """Build the reference GEN-query mask over all dual-stream KV tokens."""

    if max_sequence_length < 1:
        raise ValueError(f"max_sequence_length must be >= 1, got {max_sequence_length}")

    num_queries = layout.gen_query_indexes.numel()
    num_keys = layout.source_sequence_indexes.numel()
    if num_keys > max_sequence_length:
        raise ValueError(
            f"Teacher-forcing sequence has {num_keys} tokens, exceeding max_sequence_length={max_sequence_length}"
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


def build_per_sample_teacher_forcing_gen_masks(
    layout: TeacherForcingLayout,
    *,
    max_sequence_length: int,
) -> tuple[torch.BoolTensor, ...]:
    """Build one GEN-query dense mask per packed sample without a global 2D allocation."""

    if max_sequence_length < 1:
        raise ValueError(f"max_sequence_length must be >= 1, got {max_sequence_length}")

    num_keys = layout.source_sequence_indexes.numel()
    if num_keys > max_sequence_length:
        raise ValueError(
            f"Teacher-forcing sequence has {num_keys} tokens, exceeding max_sequence_length={max_sequence_length}"
        )

    masks: list[torch.BoolTensor] = []
    sample_offset = 0
    for sample_len in layout.sample_lens:
        sample_slice = slice(sample_offset, sample_offset + sample_len)
        sample_stream_ids = layout.stream_ids[sample_slice]
        sample_block_ids = layout.block_ids[sample_slice]
        query_rows = (sample_stream_ids == int(TeacherForcingStream.CLEAN)) | (
            sample_stream_ids == int(TeacherForcingStream.NOISY)
        )
        query_stream_ids = sample_stream_ids[query_rows, None]
        query_block_ids = sample_block_ids[query_rows, None]
        key_stream_ids = sample_stream_ids[None, :]
        key_block_ids = sample_block_ids[None, :]

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
        mask = key_is_und | allowed_by_stream
        masks.append(mask)
        sample_offset += sample_len

    if sample_offset != num_keys:
        raise ValueError("teacher-forcing sample lengths do not cover the complete packed sequence")
    return tuple(masks)


def visualize_dense_teacher_forcing_gen_mask(
    dense_gen_mask: torch.BoolTensor,
    layout: TeacherForcingLayout,
    output_path: str | Path,
    *,
    und_block_size: int = 256,
) -> Path:
    """Save a compact view of the complete teacher-forcing attention pattern.

    UND tokens are grouped into blocks of ``und_block_size`` for a compact
    causal lower-triangular overview. CLEAN/NOISY tokens are grouped by their
    causal blocks because all tokens in such a block have identical
    teacher-forcing visibility. The GEN rows come from ``dense_gen_mask``;
    the UND rows summarize the separate causal self-attention call.
    """

    if und_block_size < 1:
        raise ValueError(f"und_block_size must be >= 1, got {und_block_size}")
    if dense_gen_mask.dtype != torch.bool:
        raise TypeError(f"dense_gen_mask must be bool, got {dense_gen_mask.dtype}")
    expected_shape = (layout.gen_query_indexes.numel(), layout.source_sequence_indexes.numel())
    if tuple(dense_gen_mask.shape) != expected_shape:
        raise ValueError(f"dense_gen_mask shape must be {expected_shape}, got {tuple(dense_gen_mask.shape)}")

    def _group_starts(sample_ids: torch.Tensor, stream_ids: torch.Tensor, block_ids: torch.Tensor) -> torch.Tensor:
        starts = torch.ones(sample_ids.numel(), dtype=torch.bool, device=sample_ids.device)
        if sample_ids.numel() > 1:
            starts[1:] = (
                (sample_ids[1:] != sample_ids[:-1])
                | (stream_ids[1:] != stream_ids[:-1])
                | (block_ids[1:] != block_ids[:-1])
            )
        sample_offset = 0
        for sample_len, und_count in zip(layout.sample_lens, layout.split_lens[::2]):
            for block_start in range(und_block_size, und_count, und_block_size):
                starts[sample_offset + block_start] = True
            sample_offset += sample_len
        return torch.nonzero(starts, as_tuple=True)[0]

    gen_indexes = layout.gen_query_indexes
    representatives = _group_starts(layout.sample_ids, layout.stream_ids, layout.block_ids)
    representative_sample_ids = layout.sample_ids.index_select(0, representatives)
    representative_stream_ids = layout.stream_ids.index_select(0, representatives)
    representative_block_ids = layout.block_ids.index_select(0, representatives)

    num_groups = representatives.numel()
    grouped_mask = torch.zeros((num_groups, num_groups), dtype=torch.bool, device=dense_gen_mask.device)
    und_rows = representative_stream_ids == int(TeacherForcingStream.UND)
    grouped_mask[und_rows] = (
        (representative_sample_ids[und_rows, None] == representative_sample_ids[None, :])
        & (representative_stream_ids[None, :] == int(TeacherForcingStream.UND))
        & (representatives[None, :] <= representatives[und_rows, None])
    )

    gen_rows = ~und_rows
    gen_representatives = representatives[gen_rows]
    gen_row_by_source = torch.full(
        (layout.source_sequence_indexes.numel(),),
        -1,
        dtype=torch.long,
        device=gen_indexes.device,
    )
    gen_row_by_source[gen_indexes] = torch.arange(gen_indexes.numel(), device=gen_indexes.device)
    gen_mask_rows = gen_row_by_source.index_select(0, gen_representatives)
    if bool((gen_mask_rows < 0).any()):
        raise ValueError("CLEAN/NOISY representatives must be present in gen_query_indexes")
    grouped_mask[gen_rows] = dense_gen_mask.index_select(0, gen_mask_rows).index_select(1, representatives)
    grouped_mask = grouped_mask.detach().to(device="cpu")

    query_sample_ids = representative_sample_ids.detach().cpu()
    query_stream_ids = representative_stream_ids.detach().cpu()
    query_block_ids = representative_block_ids.detach().cpu()
    key_representatives = representatives
    key_sample_ids = layout.sample_ids.index_select(0, key_representatives).detach().cpu()
    key_stream_ids = layout.stream_ids.index_select(0, key_representatives).detach().cpu()
    key_block_ids = layout.block_ids.index_select(0, key_representatives).detach().cpu()

    # Pillow is intentionally imported only when the opt-in debug switch is on.
    from PIL import Image, ImageDraw

    num_rows, num_cols = grouped_mask.shape
    cell_size = max(1, min(18, 1400 // max(num_rows, num_cols, 1)))
    left_margin = 125
    top_margin = 135
    mask_width = max(num_cols * cell_size, 1)
    mask_height = max(num_rows * cell_size, 1)

    false_color = torch.tensor((24, 27, 35), dtype=torch.uint8)
    visible_colors = {
        int(TeacherForcingStream.UND): torch.tensor((245, 166, 35), dtype=torch.uint8),
        int(TeacherForcingStream.CLEAN): torch.tensor((68, 190, 120), dtype=torch.uint8),
        int(TeacherForcingStream.NOISY): torch.tensor((79, 145, 245), dtype=torch.uint8),
    }
    pixels = false_color.expand(num_rows, num_cols, 3).clone()
    for column, stream_id in enumerate(key_stream_ids.tolist()):
        pixels[grouped_mask[:, column], column] = visible_colors[int(stream_id)]

    mask_image = Image.fromarray(pixels.numpy())
    if cell_size != 1:
        mask_image = mask_image.resize((mask_width, mask_height), resample=Image.Resampling.NEAREST)
    image = Image.new("RGB", (left_margin + mask_width + 15, top_margin + mask_height + 15), "white")
    image.paste(mask_image, (left_margin, top_margin))
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), "Teacher-forcing attention visibility (True = visible)", fill="black")
    draw.text((8, 25), "rows/columns: [UND blocks | CLEAN blocks | NOISY blocks]", fill="black")
    draw.text(
        (8, 44),
        f"und_block={und_block_size}, vision_block={layout.block_size}, history={layout.history_blocks}",
        fill="black",
    )
    legend_x = 8
    for label, color in (
        ("UND", tuple(visible_colors[-1].tolist())),
        ("CLEAN", tuple(visible_colors[0].tolist())),
        ("NOISY", tuple(visible_colors[1].tolist())),
        ("MASKED", tuple(false_color.tolist())),
    ):
        draw.rectangle((legend_x, 66, legend_x + 10, 76), fill=color)
        draw.text((legend_x + 14, 65), label, fill="black")
        legend_x += 69

    stream_names = {-1: "U", 0: "C", 1: "N"}

    und_block_ids: list[int] = []
    next_und_block_id: dict[int, int] = {}
    for sample_id, stream_id in zip(query_sample_ids.tolist(), query_stream_ids.tolist()):
        if stream_id == int(TeacherForcingStream.UND):
            und_block_ids.append(next_und_block_id.get(sample_id, 0))
            next_und_block_id[sample_id] = und_block_ids[-1] + 1
        else:
            und_block_ids.append(-1)

    def _label(sample_id: int, stream_id: int, block_id: int, und_block_id: int) -> str:
        suffix = str(und_block_id) if stream_id == int(TeacherForcingStream.UND) else str(block_id)
        return f"s{sample_id}:{stream_names[stream_id]}{suffix}"

    # Draw all labels when cells are readable; otherwise retain sample boundary
    # labels and the color legend so large packed batches remain interpretable.
    label_stride = 1 if cell_size >= 8 else max(1, 48 // cell_size)
    for row in range(0, num_rows, label_stride):
        label = _label(
            int(query_sample_ids[row]),
            int(query_stream_ids[row]),
            int(query_block_ids[row]),
            und_block_ids[row],
        )
        draw.text((4, top_margin + row * cell_size), label, fill="black")
    for column in range(0, num_cols, label_stride):
        label = _label(
            int(key_sample_ids[column]),
            int(key_stream_ids[column]),
            int(key_block_ids[column]),
            und_block_ids[column],
        )
        label_image = Image.new("RGBA", (60, 12), (255, 255, 255, 0))
        ImageDraw.Draw(label_image).text((0, 0), label, fill="black")
        label_image = label_image.rotate(90, expand=True)
        image.paste(
            label_image,
            (left_margin + column * cell_size, top_margin - label_image.height - 2),
            label_image,
        )

    if cell_size >= 6:
        for row in range(1, num_rows):
            y = top_margin + row * cell_size
            draw.line((left_margin, y, left_margin + mask_width, y), fill=(90, 95, 105))
        for column in range(1, num_cols):
            x = left_margin + column * cell_size
            draw.line((x, top_margin, x, top_margin + mask_height), fill=(90, 95, 105))

    def _draw_boundaries(sample_ids: torch.Tensor, *, rows: bool) -> None:
        for index in range(1, sample_ids.numel()):
            if int(sample_ids[index]) == int(sample_ids[index - 1]):
                continue
            if rows:
                y = top_margin + index * cell_size
                draw.line((left_margin, y, left_margin + mask_width, y), fill=(220, 45, 45), width=2)
            else:
                x = left_margin + index * cell_size
                draw.line((x, top_margin, x, top_margin + mask_height), fill=(220, 45, 45), width=2)

    _draw_boundaries(query_sample_ids, rows=True)
    _draw_boundaries(key_sample_ids, rows=False)

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


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
    expected_text_index_chunks: list[torch.Tensor] = []
    expected_vision_index_chunks: list[torch.Tensor] = []
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
        expected_text_index_chunks.append(
            torch.arange(
                sample_offset,
                sample_offset + und_count,
                dtype=packed_sequence.text_indexes.dtype,
                device=packed_sequence.text_indexes.device,
            )
        )
        expected_vision_index_chunks.append(
            torch.arange(
                sample_offset + und_count,
                sample_offset + sample_len,
                dtype=vision.sequence_indexes.dtype,
                device=vision.sequence_indexes.device,
            )
        )
        expected_split_lens.extend((und_count, vision_count))
        expected_attn_modes.extend(("causal", "full"))
        und_token_counts.append(und_count)
        vision_token_shapes.append((num_frames, height, width))
        sample_offset += sample_len

    expected_text_indexes = torch.cat(expected_text_index_chunks)
    expected_vision_indexes = torch.cat(expected_vision_index_chunks)
    if not torch.equal(packed_sequence.text_indexes, expected_text_indexes):
        raise ValueError("text and vision sequence indexes must form the expected per-sample UND/GEN partition")
    if not torch.equal(vision.sequence_indexes, expected_vision_indexes):
        raise ValueError("text and vision sequence indexes must form the expected per-sample UND/GEN partition")
    if packed_sequence.split_lens != expected_split_lens or packed_sequence.attn_modes != expected_attn_modes:
        raise ValueError(
            "source attention splits must alternate one causal UND split and one full vision split per sample"
        )
    if packed_sequence.text_ids.numel() != expected_text_indexes.numel():
        raise ValueError(
            "text_ids must contain one payload per UND sequence index, "
            f"got {packed_sequence.text_ids.numel()} and {expected_text_indexes.numel()}"
        )

    return und_token_counts, vision_token_shapes


def _build_source_to_stream_index(
    layout: TeacherForcingLayout,
    stream: TeacherForcingStream,
) -> torch.LongTensor:
    """Build a dense source-index lookup without extracting Python scalars."""

    stream_indexes = torch.nonzero(layout.stream_ids == int(stream), as_tuple=True)[0]
    source_indexes = layout.source_sequence_indexes.index_select(0, stream_indexes)
    source_to_new = torch.full(
        (sum(layout.original_sample_lens),),
        -1,
        dtype=torch.long,
        device=layout.source_sequence_indexes.device,
    )
    source_to_new[source_indexes] = stream_indexes
    return source_to_new


def _remap_indexes(indexes: torch.Tensor | None, source_to_new: torch.Tensor, name: str) -> torch.Tensor | None:
    if indexes is None:
        return None
    if indexes.device != source_to_new.device:
        raise ValueError(f"{name} and source index lookup must be on the same device")
    indexes = indexes.to(dtype=torch.long)
    if bool(((indexes < 0) | (indexes >= source_to_new.numel())).any()):
        raise ValueError(f"{name} contains an index outside the packed source sequence")
    remapped = source_to_new.index_select(0, indexes)
    if bool((remapped < 0).any()):
        raise ValueError(f"{name} contains an index outside the supported source stream")
    return remapped


def expand_packed_sequence_for_teacher_forcing(
    packed_sequence: PackedSequence,
    *,
    clean_vision_tokens: Sequence[torch.Tensor],
    block_size: int,
    history_blocks: int,
) -> PackedSequence:
    """Return a vision-only packed sequence expanded into clean/noisy GEN streams."""

    from cosmos_framework.data.generator.sequence_packing.modality import ModalitySpan

    with torch.autograd.profiler.record_function("teacher_forcing/validate_layout"):
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

    with torch.autograd.profiler.record_function("teacher_forcing/build_layout"):
        layout = build_teacher_forcing_layout(
            und_token_counts=und_token_counts,
            vision_token_shapes=vision_token_shapes,
            block_size=block_size,
            history_blocks=history_blocks,
        )
    with torch.autograd.profiler.record_function("teacher_forcing/remap_indexes"):
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
        span_source_indexes = torch.arange(
            span.sequence_start,
            span.sequence_start + span.sequence_len,
            dtype=torch.long,
            device=source_to_noisy.device,
        )
        span_indexes = _remap_indexes(span_source_indexes, source_to_noisy, "vision span")
        assert span_indexes is not None
        if span_indexes.numel() > 1 and not bool((span_indexes[1:] == span_indexes[:-1] + 1).all()):
            raise ValueError(f"vision span at source index {span.sequence_start} is not contiguous after remapping")
        remapped_spans.append(replace(span, sequence_start=int(span_indexes[0])))

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
