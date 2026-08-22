# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Teacher-forcing adapter for the registered explicit-mask attention backend."""

import torch

from cosmos_framework.model.attention import attention


def teacher_forcing_dense_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    allowed_mask: torch.Tensor,
    *,
    scale: float | None = None,
    mask_is_prevalidated: bool = False,
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
        backend_kwargs={
            "allowed_mask": allowed_mask,
            "validate_allowed_mask": not mask_is_prevalidated,
        },
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
    masks_are_prevalidated: bool = False,
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
    for sample_len, gen_len, allowed_mask in zip(
        sample_lens, gen_sample_lens, allowed_masks, strict=True
    ):
        query_end = query_offset + gen_len
        kv_end = kv_offset + sample_len
        outputs.append(
            teacher_forcing_dense_attention(
                query[query_offset:query_end],
                key[kv_offset:kv_end],
                value[kv_offset:kv_end],
                allowed_mask,
                scale=scale,
                mask_is_prevalidated=masks_are_prevalidated,
            )
        )
        query_offset = query_end
        kv_offset = kv_end
    return torch.cat(outputs, dim=0)
