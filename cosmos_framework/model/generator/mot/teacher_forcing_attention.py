# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Teacher-forcing adapters for dense SDPA and Ascend fused attention."""

import os

import torch

from cosmos_framework.model.attention import attention

_ASCEND_FUSED_ATTENTION_ENV = "COSMOS_ASCEND_FUSED_TEACHER_FORCING_ATTENTION"


def use_ascend_teacher_forcing_fused_attention(device: torch.device) -> bool:
    """Return whether the opt-in Ascend teacher-forcing FA experiment is active."""

    value = os.environ.get(_ASCEND_FUSED_ATTENTION_ENV, "0").strip().lower()
    if value not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
        raise ValueError(f"{_ASCEND_FUSED_ATTENTION_ENV} must be a boolean value, got {value!r}")
    return device.type == "npu" and value in {"1", "true", "yes", "on"}


def _ascend_teacher_forcing_fused_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    blocked_mask: torch.Tensor,
    *,
    scale: float | None,
) -> torch.Tensor:
    """Run arbitrary-mask training FA with native GQA in BSND layout.

    Ascend's mask convention is the inverse of PyTorch SDPA: ``True`` means
    blocked. ``sparse_mode=1`` consumes the complete arbitrary mask and does
    not impose causal/band structure on the teacher-forcing topology.
    """

    import torch_npu

    # Cosmos already stores these tensors as contiguous [S, N, D]. BSND adds
    # only a view batch dimension; BNSD would introduce non-contiguous transposes
    # and can force three large format-conversion copies before every layer.
    q = query.unsqueeze(0)
    k = key.unsqueeze(0)
    v = value.unsqueeze(0)
    output = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        head_num=query.shape[1],
        input_layout="BSND",
        atten_mask=blocked_mask,
        scale=query.shape[-1] ** -0.5 if scale is None else scale,
        keep_prob=1.0,
        sparse_mode=1,
    )[0]
    expected_shape = (1, query.shape[0], query.shape[1], value.shape[2])
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected teacher-forcing npu_fusion_attention output shape: "
            f"expected {expected_shape}, got {tuple(output.shape)}"
        )
    return output.squeeze(0)


def teacher_forcing_dense_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    allowed_mask: torch.Tensor | None,
    *,
    blocked_mask: torch.Tensor | None = None,
    scale: float | None = None,
    mask_is_prevalidated: bool = False,
) -> torch.Tensor:
    """Attend GEN queries once over unified UND/clean/noisy keys.

    ``allowed_mask[q, k] == True`` means key ``k`` is visible to query ``q``.
    The implementation uses one SDPA softmax and never exposes or merges LSE.
    """

    if allowed_mask is None and blocked_mask is None:
        raise ValueError("teacher-forcing attention requires an allowed_mask or blocked_mask")
    if allowed_mask is not None and blocked_mask is not None:
        raise ValueError("teacher-forcing attention accepts only one mask representation")

    mask = allowed_mask if allowed_mask is not None else blocked_mask
    assert mask is not None
    if mask.dtype != torch.bool:
        raise TypeError(f"teacher-forcing mask must use bool dtype, got {mask.dtype}")
    expected_mask_shape = (query.shape[0], key.shape[0])
    if tuple(mask.shape) != expected_mask_shape:
        raise ValueError(f"teacher-forcing mask must have shape {expected_mask_shape}, got {tuple(mask.shape)}")
    if mask.device != query.device:
        raise ValueError(
            f"teacher-forcing mask must be on the same device as query, got {mask.device} and {query.device}"
        )

    if use_ascend_teacher_forcing_fused_attention(query.device):
        if query.shape[1] % key.shape[1] != 0:
            raise ValueError("query heads must evenly divide into key/value heads")
        if query.shape[-1] != key.shape[-1] or key.shape != value.shape:
            raise ValueError("teacher-forcing fused attention requires matching key/value shapes and head dims")
        if blocked_mask is None:
            # Compatibility fallback for direct callers. The full training path
            # creates this representation once in build_packed_sequence instead.
            blocked_mask = torch.logical_not(allowed_mask)
        return _ascend_teacher_forcing_fused_attention(
            query,
            key,
            value,
            blocked_mask,
            scale=scale,
        )

    if allowed_mask is None:
        # This only matters if the environment changes after metadata creation.
        allowed_mask = torch.logical_not(blocked_mask)

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
    allowed_masks: tuple[torch.Tensor, ...] | None,
    *,
    blocked_masks: tuple[torch.Tensor, ...] | None = None,
    sample_lens: tuple[int, ...],
    gen_sample_lens: tuple[int, ...],
    scale: float | None = None,
    masks_are_prevalidated: bool = False,
) -> torch.Tensor:
    """Run Scheme-B dense attention independently for each packed sample."""

    num_samples = len(sample_lens)
    masks = allowed_masks if allowed_masks is not None else blocked_masks
    if masks is None or (allowed_masks is not None and blocked_masks is not None):
        raise ValueError("per-sample teacher-forcing attention requires exactly one mask representation")
    if num_samples == 0 or len(gen_sample_lens) != num_samples or len(masks) != num_samples:
        raise ValueError("per-sample teacher-forcing metadata must contain one entry per packed sample")
    if sum(sample_lens) != key.shape[0] or value.shape[0] != key.shape[0]:
        raise ValueError("per-sample KV lengths must cover the complete packed key/value sequence")
    if sum(gen_sample_lens) != query.shape[0]:
        raise ValueError("per-sample GEN lengths must cover the complete packed query sequence")

    outputs: list[torch.Tensor] = []
    query_offset = 0
    kv_offset = 0
    for sample_len, gen_len, sample_mask in zip(sample_lens, gen_sample_lens, masks, strict=True):
        query_end = query_offset + gen_len
        kv_end = kv_offset + sample_len
        outputs.append(
            teacher_forcing_dense_attention(
                query[query_offset:query_end],
                key[kv_offset:kv_end],
                value[kv_offset:kv_end],
                sample_mask if allowed_masks is not None else None,
                blocked_mask=sample_mask if blocked_masks is not None else None,
                scale=scale,
                mask_is_prevalidated=masks_are_prevalidated,
            )
        )
        query_offset = query_end
        kv_offset = kv_end
    return torch.cat(outputs, dim=0)
