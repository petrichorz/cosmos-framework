# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Pure tensor metadata used by teacher-forcing block attention.

This module does not call the NPU block-attention kernel.  Phase 1 only moves
each sample's KV tokens from ``[UND, clean, noisy]`` to
``[clean, noisy, UND]`` while preserving the order inside every stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cosmos_framework.data.generator.sequence_packing.teacher_forcing import (
    TeacherForcingLayout,
    TeacherForcingStream,
)


@dataclass(frozen=True)
class TeacherForcingKVPermutation:
    """Global packed KV permutation and its inverse.

    ``forward_indices[new_position]`` gives the corresponding position in the
    reference ``[UND, clean, noisy]`` pack.  ``inverse_indices`` restores a
    tensor whose first dimension is in the reordered layout.
    """

    forward_indices: torch.LongTensor
    inverse_indices: torch.LongTensor
    sample_lens: tuple[int, ...]

    def to(self, device: torch.device | str) -> TeacherForcingKVPermutation:
        return TeacherForcingKVPermutation(
            forward_indices=self.forward_indices.to(device=device),
            inverse_indices=self.inverse_indices.to(device=device),
            sample_lens=self.sample_lens,
        )


@dataclass(frozen=True)
class TeacherForcingBlockMetadata:
    """Per-sample homogeneous tile metadata for the reordered Q/KV streams."""

    block_shape: tuple[int, int]
    history_blocks: int
    q_actual_lengths: tuple[int, ...]
    kv_actual_lengths: tuple[int, ...]
    q_tile_stream_ids: tuple[torch.LongTensor, ...]
    q_tile_block_ids: tuple[torch.LongTensor, ...]
    kv_tile_stream_ids: tuple[torch.LongTensor, ...]
    kv_tile_block_ids: tuple[torch.LongTensor, ...]


def build_teacher_forcing_kv_permutation(layout: TeacherForcingLayout) -> TeacherForcingKVPermutation:
    """Build a sample-local ``UND,clean,noisy -> clean,noisy,UND`` permutation."""

    num_tokens = int(layout.source_sequence_indexes.numel())
    if sum(layout.sample_lens) != num_tokens:
        raise ValueError("teacher-forcing sample lengths must cover the complete packed KV sequence")
    if layout.stream_ids.numel() != num_tokens or layout.sample_ids.numel() != num_tokens:
        raise ValueError("teacher-forcing stream/sample metadata must match the packed KV sequence")

    chunks: list[torch.Tensor] = []
    sample_offset = 0
    for sample_id, sample_len in enumerate(layout.sample_lens):
        sample_end = sample_offset + sample_len
        sample_slice = slice(sample_offset, sample_end)
        sample_stream_ids = layout.stream_ids[sample_slice]
        sample_ids = layout.sample_ids[sample_slice]
        if not bool((sample_ids == sample_id).all()):
            raise ValueError(f"teacher-forcing sample {sample_id} metadata is not contiguous")

        local_order = torch.cat(
            [
                torch.nonzero(sample_stream_ids == int(stream), as_tuple=True)[0]
                for stream in (
                    TeacherForcingStream.CLEAN,
                    TeacherForcingStream.NOISY,
                    TeacherForcingStream.UND,
                )
            ]
        )
        if local_order.numel() != sample_len:
            raise ValueError(f"teacher-forcing sample {sample_id} contains an unknown KV stream")
        chunks.append(local_order + sample_offset)
        sample_offset = sample_end

    forward_indices = torch.cat(chunks)
    inverse_indices = torch.empty_like(forward_indices)
    inverse_indices[forward_indices] = torch.arange(num_tokens, device=forward_indices.device)
    return TeacherForcingKVPermutation(
        forward_indices=forward_indices,
        inverse_indices=inverse_indices,
        sample_lens=layout.sample_lens,
    )


def reorder_teacher_forcing_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    permutation: TeacherForcingKVPermutation,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the same sample-local permutation to K and V."""

    num_tokens = permutation.forward_indices.numel()
    if key.shape[0] != num_tokens or value.shape[0] != num_tokens:
        raise ValueError(
            "teacher-forcing K/V token counts must match the permutation, "
            f"got K={key.shape[0]}, V={value.shape[0]}, permutation={num_tokens}"
        )
    if key.device != value.device or key.device != permutation.forward_indices.device:
        raise ValueError("teacher-forcing K, V, and permutation must be on the same device")
    return (
        key.index_select(0, permutation.forward_indices),
        value.index_select(0, permutation.forward_indices),
    )


def restore_teacher_forcing_kv_order(
    tensor: torch.Tensor,
    permutation: TeacherForcingKVPermutation,
) -> torch.Tensor:
    """Restore the reference ``[UND, clean, noisy]`` order along dimension 0."""

    if tensor.shape[0] != permutation.inverse_indices.numel():
        raise ValueError("teacher-forcing tensor token count must match the inverse permutation")
    if tensor.device != permutation.inverse_indices.device:
        raise ValueError("teacher-forcing tensor and inverse permutation must be on the same device")
    return tensor.index_select(0, permutation.inverse_indices)


def reorder_teacher_forcing_allowed_mask(
    allowed_mask: torch.Tensor,
    permutation: TeacherForcingKVPermutation,
) -> torch.Tensor:
    """Permute the KV axis of a global reference dense mask."""

    if allowed_mask.ndim != 2 or allowed_mask.shape[1] != permutation.forward_indices.numel():
        raise ValueError("teacher-forcing allowed mask KV axis must match the permutation")
    if allowed_mask.device != permutation.forward_indices.device:
        raise ValueError("teacher-forcing allowed mask and permutation must be on the same device")
    return allowed_mask.index_select(1, permutation.forward_indices)


def reorder_teacher_forcing_sample_masks(
    allowed_masks: tuple[torch.Tensor, ...],
    permutation: TeacherForcingKVPermutation,
) -> tuple[torch.Tensor, ...]:
    """Permute each per-sample reference mask without crossing sample boundaries."""

    if len(allowed_masks) != len(permutation.sample_lens):
        raise ValueError("teacher-forcing masks and permutation must contain the same number of samples")

    reordered: list[torch.Tensor] = []
    sample_offset = 0
    for sample_id, (allowed_mask, sample_len) in enumerate(zip(allowed_masks, permutation.sample_lens, strict=True)):
        if allowed_mask.ndim != 2 or allowed_mask.shape[1] != sample_len:
            raise ValueError(f"teacher-forcing sample {sample_id} mask KV axis must equal {sample_len}")
        sample_indices = permutation.forward_indices[sample_offset : sample_offset + sample_len] - sample_offset
        if allowed_mask.device != sample_indices.device:
            raise ValueError(f"teacher-forcing sample {sample_id} mask and permutation must be on the same device")
        reordered.append(allowed_mask.index_select(1, sample_indices))
        sample_offset += sample_len
    return tuple(reordered)


def _extract_homogeneous_tiles(
    stream_ids: torch.Tensor,
    block_ids: torch.Tensor,
    *,
    tile_size: int,
    allow_partial_tail: bool,
    sequence_name: str,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    """Return one stream/block ID per tile, rejecting mixed semantic tiles."""

    num_tokens = stream_ids.numel()
    if num_tokens == 0 or block_ids.numel() != num_tokens:
        raise ValueError(f"{sequence_name} stream/block metadata must be non-empty and have equal lengths")
    if not allow_partial_tail and num_tokens % tile_size != 0:
        raise ValueError(f"{sequence_name} token count must be divisible by block size {tile_size}, got {num_tokens}")

    starts = torch.arange(0, num_tokens, tile_size, device=stream_ids.device)
    tile_stream_ids = stream_ids.index_select(0, starts)
    tile_block_ids = block_ids.index_select(0, starts)
    token_tile_ids = torch.div(torch.arange(num_tokens, device=stream_ids.device), tile_size, rounding_mode="floor")
    expected_stream_ids = tile_stream_ids.index_select(0, token_tile_ids)
    expected_block_ids = tile_block_ids.index_select(0, token_tile_ids)
    if not bool(((stream_ids == expected_stream_ids) & (block_ids == expected_block_ids)).all()):
        raise ValueError(f"{sequence_name} contains a {tile_size}-token tile with mixed stream or logical block IDs")
    return tile_stream_ids, tile_block_ids


def build_teacher_forcing_block_metadata(
    layout: TeacherForcingLayout,
    permutation: TeacherForcingKVPermutation,
    *,
    block_shape: tuple[int, int] = (128, 128),
) -> TeacherForcingBlockMetadata:
    """Build exact tile metadata without constructing an O(Q*KV) dense mask."""

    if block_shape != (128, 128):
        raise ValueError(f"teacher-forcing Phase 2 requires block_shape=(128, 128), got {block_shape}")
    if permutation.sample_lens != layout.sample_lens:
        raise ValueError("teacher-forcing permutation and layout sample lengths must match")

    q_stream_ids = layout.stream_ids.index_select(0, layout.gen_query_indexes)
    q_block_ids = layout.block_ids.index_select(0, layout.gen_query_indexes)
    kv_stream_ids = layout.stream_ids.index_select(0, permutation.forward_indices)
    kv_block_ids = layout.block_ids.index_select(0, permutation.forward_indices)
    kv_sample_ids = layout.sample_ids.index_select(0, permutation.forward_indices)

    q_actual_lengths = tuple(layout.split_lens[1::2])
    kv_actual_lengths = layout.sample_lens
    q_tile_stream_ids: list[torch.LongTensor] = []
    q_tile_block_ids: list[torch.LongTensor] = []
    kv_tile_stream_ids: list[torch.LongTensor] = []
    kv_tile_block_ids: list[torch.LongTensor] = []
    q_offset = 0
    kv_offset = 0
    for sample_id, (q_len, kv_len) in enumerate(zip(q_actual_lengths, kv_actual_lengths, strict=True)):
        q_slice = slice(q_offset, q_offset + q_len)
        kv_slice = slice(kv_offset, kv_offset + kv_len)
        if not bool((kv_sample_ids[kv_slice] == sample_id).all()):
            raise ValueError(f"teacher-forcing KV permutation crosses sample {sample_id} boundary")

        q_stream, q_blocks = _extract_homogeneous_tiles(
            q_stream_ids[q_slice],
            q_block_ids[q_slice],
            tile_size=block_shape[0],
            allow_partial_tail=False,
            sequence_name=f"sample {sample_id} Q",
        )
        kv_stream, kv_blocks = _extract_homogeneous_tiles(
            kv_stream_ids[kv_slice],
            kv_block_ids[kv_slice],
            tile_size=block_shape[1],
            allow_partial_tail=True,
            sequence_name=f"sample {sample_id} KV",
        )
        if kv_stream[-1] != int(TeacherForcingStream.UND):
            raise ValueError(f"teacher-forcing sample {sample_id} KV tail must contain UND tokens")
        q_tile_stream_ids.append(q_stream)
        q_tile_block_ids.append(q_blocks)
        kv_tile_stream_ids.append(kv_stream)
        kv_tile_block_ids.append(kv_blocks)
        q_offset += q_len
        kv_offset += kv_len

    return TeacherForcingBlockMetadata(
        block_shape=block_shape,
        history_blocks=layout.history_blocks,
        q_actual_lengths=q_actual_lengths,
        kv_actual_lengths=kv_actual_lengths,
        q_tile_stream_ids=tuple(q_tile_stream_ids),
        q_tile_block_ids=tuple(q_tile_block_ids),
        kv_tile_stream_ids=tuple(kv_tile_stream_ids),
        kv_tile_block_ids=tuple(kv_tile_block_ids),
    )


def build_teacher_forcing_block_sparse_mask(
    metadata: TeacherForcingBlockMetadata,
    *,
    num_q_heads: int,
) -> torch.Tensor:
    """Build the int8 block mask from compact semantic tile metadata."""

    if num_q_heads < 1:
        raise ValueError(f"num_q_heads must be positive, got {num_q_heads}")
    num_samples = len(metadata.q_actual_lengths)
    if num_samples == 0:
        raise ValueError("teacher-forcing block metadata cannot be empty")
    max_q_tiles = max(ids.numel() for ids in metadata.q_tile_stream_ids)
    max_kv_tiles = max(ids.numel() for ids in metadata.kv_tile_stream_ids)
    device = metadata.q_tile_stream_ids[0].device
    mask = torch.zeros(
        num_samples,
        num_q_heads,
        max_q_tiles,
        max_kv_tiles,
        dtype=torch.uint8,
        device=device,
    )

    for sample_id in range(num_samples):
        q_stream = metadata.q_tile_stream_ids[sample_id][:, None]
        q_block = metadata.q_tile_block_ids[sample_id][:, None]
        kv_stream = metadata.kv_tile_stream_ids[sample_id][None, :]
        kv_block = metadata.kv_tile_block_ids[sample_id][None, :]
        key_is_und = kv_stream == int(TeacherForcingStream.UND)
        key_is_clean = kv_stream == int(TeacherForcingStream.CLEAN)
        key_is_noisy = kv_stream == int(TeacherForcingStream.NOISY)
        inside_history = kv_block >= q_block - metadata.history_blocks
        clean_visible = key_is_clean & inside_history & (kv_block <= q_block)
        noisy_visible = (key_is_clean & inside_history & (kv_block < q_block)) | (key_is_noisy & (kv_block == q_block))
        allowed = key_is_und | torch.where(
            q_stream == int(TeacherForcingStream.CLEAN),
            clean_visible,
            noisy_visible,
        )
        q_tiles, kv_tiles = allowed.shape
        mask[sample_id, :, :q_tiles, :kv_tiles] = allowed.to(torch.uint8).unsqueeze(0)
    return mask


def expand_teacher_forcing_block_sparse_mask(
    block_sparse_mask: torch.Tensor,
    metadata: TeacherForcingBlockMetadata,
) -> tuple[torch.BoolTensor, ...]:
    """Expand head zero into per-sample token masks for correctness tests."""

    if block_sparse_mask.ndim != 4 or block_sparse_mask.shape[0] != len(metadata.q_actual_lengths):
        raise ValueError("teacher-forcing block mask shape does not match metadata")
    block_q, block_kv = metadata.block_shape
    expanded: list[torch.BoolTensor] = []
    for sample_id, (q_len, kv_len) in enumerate(
        zip(metadata.q_actual_lengths, metadata.kv_actual_lengths, strict=True)
    ):
        q_tiles = metadata.q_tile_stream_ids[sample_id].numel()
        kv_tiles = metadata.kv_tile_stream_ids[sample_id].numel()
        token_mask = block_sparse_mask[sample_id, 0, :q_tiles, :kv_tiles].bool()
        token_mask = token_mask.repeat_interleave(block_q, dim=0).repeat_interleave(block_kv, dim=1)
        expanded.append(token_mask[:q_len, :kv_len])
    return tuple(expanded)
