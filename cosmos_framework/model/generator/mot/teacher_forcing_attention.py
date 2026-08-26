# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Teacher-forcing adapter for the registered explicit-mask attention backend."""

import torch

from cosmos_framework.model.attention import attention
from cosmos_framework.model.generator.mot.teacher_forcing_block_attention import (
    TeacherForcingBlockMetadata,
    TeacherForcingKVPermutation,
    reorder_teacher_forcing_kv,
)


def teacher_forcing_dense_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    allowed_mask: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Attend GEN queries once over unified UND/clean/noisy keys.

    ``allowed_mask[q, k] == True`` means key ``k`` is visible to query ``q``.
    The implementation uses one SDPA softmax and never exposes or merges LSE.
    """

    output = attention(
        query.unsqueeze(0),
        key.unsqueeze(0),
        value.unsqueeze(0),
        backend="masked_sdpa",
        backend_kwargs={"allowed_mask": allowed_mask},
        scale=scale,
    )
    return output.squeeze(0)


def teacher_forcing_per_sample_dense_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    allowed_masks: tuple[torch.Tensor, ...],
    *,
    sample_lens: tuple[int, ...],
    gen_sample_lens: tuple[int, ...],
    scale: float | None = None,
) -> torch.Tensor:
    """Run Scheme-B dense attention independently for each packed sample."""

    num_samples = len(sample_lens)
    if num_samples == 0 or len(gen_sample_lens) != num_samples or len(allowed_masks) != num_samples:
        raise ValueError("per-sample teacher-forcing metadata must contain one entry per packed sample")
    if sum(sample_lens) != key.shape[0] or value.shape[0] != key.shape[0]:
        raise ValueError("per-sample KV lengths must cover the complete packed key/value sequence")
    if sum(gen_sample_lens) != query.shape[0]:
        raise ValueError("per-sample GEN lengths must cover the complete packed query sequence")

    outputs: list[torch.Tensor] = []
    query_offset = 0
    kv_offset = 0
    for sample_len, gen_len, allowed_mask in zip(sample_lens, gen_sample_lens, allowed_masks, strict=True):
        query_end = query_offset + gen_len
        kv_end = kv_offset + sample_len
        outputs.append(
            teacher_forcing_dense_attention(
                query[query_offset:query_end],
                key[kv_offset:kv_end],
                value[kv_offset:kv_end],
                allowed_mask,
                scale=scale,
            )
        )
        query_offset = query_end
        kv_offset = kv_end
    return torch.cat(outputs, dim=0)


def teacher_forcing_block_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    permutation: TeacherForcingKVPermutation,
    metadata: TeacherForcingBlockMetadata,
    block_sparse_mask: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Run one packed GEN attention call with reordered K/V and a block mask."""

    reordered_key, reordered_value = reorder_teacher_forcing_kv(key, value, permutation)

    def cumulative_offsets(lengths: tuple[int, ...]) -> torch.Tensor:
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        return torch.tensor(offsets, dtype=torch.int32, device=query.device)

    output = attention(
        query.unsqueeze(0),
        reordered_key.unsqueeze(0),
        reordered_value.unsqueeze(0),
        cumulative_seqlen_Q=cumulative_offsets(metadata.q_actual_lengths),
        cumulative_seqlen_KV=cumulative_offsets(metadata.kv_actual_lengths),
        max_seqlen_Q=max(metadata.q_actual_lengths),
        max_seqlen_KV=max(metadata.kv_actual_lengths),
        backend="block_attention",
        backend_kwargs={
            "block_sparse_mask": block_sparse_mask,
            "block_shape": list(metadata.block_shape),
            "inner_precise": 0 if query.dtype == torch.bfloat16 else 1,
        },
        scale=scale,
    )
    return output.squeeze(0)
