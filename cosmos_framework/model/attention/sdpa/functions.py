# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

SDPA Backend: intermediate APIs.

Implements the same backend contract as ``flash2_attention`` /
``natten_attention`` (heads-last ``[B, S, H, D]`` layout, varlen via
``cumulative_seqlen_{Q,KV}``, optional ``return_lse``) on top of
``torch.nn.functional.scaled_dot_product_attention``.

Because ``scaled_dot_product_attention`` has no native varlen mode and does not
expose logsumexp portably, this backend:

  * Loops over the varlen segments defined by ``cumulative_seqlen_{Q,KV}`` and
    runs one SDPA call per segment. Each segment is a single sample's token
    count, so per-segment attention stays memory-feasible even when the packed
    sequence is long (e.g. 74k tokens for the action-policy recipe).
  * Computes ``lse`` (when requested) via an explicit ``logsumexp`` on the
    scores. This materialises the per-segment score matrix, so it is only used
    on the ``return_lse=True`` path; the common ``return_lse=False`` path stays
    on the fused SDPA kernel.

Device-agnostic: runs on CUDA, Ascend NPU (via torch_npu's registered SDPA
backend), and CPU. Selected automatically on non-CUDA devices - see
``backends.get_backend_list``.
"""

import torch
from torch import Tensor

from cosmos_framework.model.attention.checks import assert_universal_tensor_checks
from cosmos_framework.model.attention.masks import CausalType
from cosmos_framework.model.attention.sdpa.checks import sdpa_attention_check


def _expand_gqa(k: Tensor, v: Tensor, num_q_heads: int) -> tuple[Tensor, Tensor]:
    """Expand GQA/MQA k, v to match the query head count.

    Args:
        k: ``[H_kv, S, D]`` (segment) or ``[B, H_kv, S, D]`` (batched) key tensor.
        v: same layout as ``k``, value tensor.
        num_q_heads: query head count ``H``.

    Returns:
        ``(k, v)`` with the head dim expanded to ``num_q_heads`` (by repeating
        each kv head ``H // H_kv`` times). No-op when ``H_kv == num_q_heads``.
    """
    h_kv = k.shape[-3 if k.dim() == 4 else 0]
    if h_kv == num_q_heads:
        return k, v
    rep = num_q_heads // h_kv
    return k.repeat_interleave(rep, dim=-3 if k.dim() == 4 else 0), v.repeat_interleave(
        rep, dim=-3 if v.dim() == 4 else 0
    )


def _sdpa_segment(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool,
    scale: float,
    return_lse: bool,
) -> tuple[Tensor, Tensor | None]:
    """Run SDPA on one varlen segment.

    Args:
        q: ``[H, S_q, D]`` heads-first query.
        k: ``[H_kv, S_kv, D]`` heads-first key.
        v: ``[H_kv, S_kv, D_v]`` heads-first value.
        is_causal: TopLeft causal flag (SDPA ``is_causal=True`` semantics).
        scale: dot-product scale.
        return_lse: if True, also return logsumexp ``[H, S_q]``.

    Returns:
        output ``[H, S_q, D_v]`` and lse ``[H, S_q] | None``.
    """
    k, v = _expand_gqa(k, v, q.shape[0])
    # SDPA expects [B, H, S, D]; use a singleton batch.
    q4 = q.unsqueeze(0)
    k4 = k.unsqueeze(0)
    v4 = v.unsqueeze(0)
    if not return_lse:
        out = torch.nn.functional.scaled_dot_product_attention(
            q4, k4, v4, is_causal=is_causal, scale=scale
        )
        return out.squeeze(0), None
    # Math path to also recover logsumexp (SDPA does not expose lse portably).
    scores = torch.matmul(q4, k4.transpose(-2, -1)) * scale  # [1, H, S_q, S_kv]
    if is_causal:
        # TopLeft causal (matches SDPA is_causal=True): mask keys j > query i.
        s_q, s_kv = q4.shape[-2], k4.shape[-2]
        causal_mask = torch.triu(
            torch.ones(s_q, s_kv, dtype=torch.bool, device=q4.device), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)  # [1, H, S_q]
    attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(v4.dtype)
    out = torch.matmul(attn, v4)  # [1, H, S_q, D_v]
    return out.squeeze(0), lse.squeeze(0)


def sdpa_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool = False,
    causal_type: CausalType | None = None,
    scale: float | None = None,
    cumulative_seqlen_Q: Tensor | None = None,
    cumulative_seqlen_KV: Tensor | None = None,
    max_seqlen_Q: int | None = None,
    max_seqlen_KV: int | None = None,
    return_lse: bool = False,
    backend_kwargs: dict | None = None,
    deterministic: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """
    Runs ``scaled_dot_product_attention`` on given operands (Q, K, V) with the
    heads-last contiguous layout (`[batch, seqlen, heads, head_dim]`).

    Device-agnostic fallback backend. Mirrors the ``flash2_attention`` contract
    (same arguments, same return shapes) so it is a drop-in for
    ``BACKEND_MAP``.

    Varlen (sequence-packed, batch size 1) is handled by looping over the
    segments defined by ``cumulative_seqlen_{Q,KV}`` and running one SDPA call
    per segment, then concatenating. ``return_lse`` is computed via an explicit
    ``logsumexp`` on the scores (per segment).

    Parameters:
        query (Tensor): 4-D query tensor (`[batch, seqlen_q, heads, head_dim]`).

        key (Tensor): 4-D key tensor (`[batch, seqlen_kv, heads_kv, head_dim]`).

        value (Tensor): 4-D value tensor (`[batch, seqlen_kv, heads_kv, head_dim_v]`).

        is_causal (bool): whether or not causal masking is enabled. Default False.

        causal_type (CausalType): causal masking mode. SDPA's ``is_causal=True``
            is TopLeft; ``DontCare`` (seqlen_q == seqlen_kv) is also supported.
            ``BottomRight`` is rejected by the check.

        scale (float | None): Dot product scale. Defaults to ``head_dim ** -0.5``.

        cumulative_seqlen_Q (Tensor | None): (varlen) `[batch + 1]` cumulative
            query token counts with a leading 0.

        cumulative_seqlen_KV (Tensor | None): (varlen) `[batch + 1]` cumulative
            key/value token counts with a leading 0.

        max_seqlen_Q (int | None): (varlen) max query segment length.

        max_seqlen_KV (int | None): (varlen) max key/value segment length.

    Other Parameters:
        return_lse (bool): Whether to return logsumexp. Default False.

        backend_kwargs (dict | None): Ignored (accepted for contract parity).

        deterministic (bool): Accepted for contract parity; not enforced.

    Returns:
        output (Tensor): 4-D `[batch, seqlen_q, heads, head_dim_v]`.

        logsumexp (Tensor): 3-D `[batch, seqlen_q, heads]`. Only returned when
            ``return_lse`` is True. Shape matches the flash2 backend's final lse
            (after its ``permute(0, 2, 1)``) so ``merge_attentions`` can consume
            it. NOTE: unlike flash2, this lse is computed by an explicit
            ``logsumexp`` op (not a fused attention autograd op), so it does not
            share a data pointer with the output - sufficient for the
            ``return_lse=False`` MoT paths; the lse-merging path additionally
            requires NATTEN, which is not available on NPU.
    """
    is_varlen = cumulative_seqlen_Q is not None
    assert_universal_tensor_checks(query, key, value)

    backend_kwargs = backend_kwargs.copy() if backend_kwargs is not None else {}
    # Determinism in backend_kwargs supersedes primary flag, if set to True
    if "deterministic" in backend_kwargs:
        deterministic = deterministic or backend_kwargs["deterministic"]
        del backend_kwargs["deterministic"]

    assert sdpa_attention_check(
        query_shape=query.shape,
        key_shape=key.shape,
        value_shape=value.shape,
        dtype=query.dtype,
        device=query.device,
        requires_grad=query.requires_grad or key.requires_grad or value.requires_grad,
        is_causal=is_causal,
        causal_type=causal_type,
        is_varlen=is_varlen,
        deterministic=deterministic,
        raise_error=True,
    )

    scale = scale if scale is not None else query.shape[-1] ** -0.5

    # SDPA's is_causal=True is TopLeft (lower-triangular). Valid for TopLeft and
    # DontCare (the latter only when seqlen_q == seqlen_kv, already validated).
    sdpa_is_causal = bool(is_causal and causal_type in (CausalType.TopLeft, CausalType.DontCare, None))

    if is_varlen:
        assert query.shape[0] == key.shape[0] == value.shape[0] == 1
        q = query.squeeze(0)  # [total_q, H, D]
        k = key.squeeze(0)  # [total_kv, H_kv, D]
        v = value.squeeze(0)  # [total_kv, H_kv, D_v]
        num_q_heads, head_dim_v = q.shape[1], v.shape[2]

        # `.tolist()` syncs a small int tensor to host; fine for a length-B+1 list.
        cu_q = cumulative_seqlen_Q.tolist()
        cu_kv = cumulative_seqlen_KV.tolist()

        out_chunks: list[Tensor] = []
        lse_chunks: list[Tensor] = []
        for i in range(len(cu_q) - 1):
            q_start, q_end = cu_q[i], cu_q[i + 1]
            kv_start, kv_end = cu_kv[i], cu_kv[i + 1]
            if q_end <= q_start:
                # Empty query segment - skip SDPA, emit an empty placeholder so
                # the concatenated output preserves segment alignment.
                out_chunks.append(q.new_empty((num_q_heads, 0, head_dim_v)))
                if return_lse:
                    lse_chunks.append(q.new_empty((num_q_heads, 0)))
                continue
            qs = q[q_start:q_end].transpose(0, 1)  # [H, S_q, D]
            ks = k[kv_start:kv_end].transpose(0, 1)  # [H_kv, S_kv, D]
            vs = v[kv_start:kv_end].transpose(0, 1)  # [H_kv, S_kv, D_v]
            out_i, lse_i = _sdpa_segment(
                qs, ks, vs, is_causal=sdpa_is_causal, scale=scale, return_lse=return_lse
            )
            out_chunks.append(out_i)
            if return_lse:
                lse_chunks.append(lse_i)

        output = (
            torch.cat(out_chunks, dim=1) if out_chunks else q.new_empty((num_q_heads, 0, head_dim_v))
        )  # [H, total_q, D_v]
        output = output.transpose(0, 1).unsqueeze(0)  # [1, total_q, H, D_v]

        if return_lse:
            lse = (
                torch.cat(lse_chunks, dim=1) if lse_chunks else q.new_empty((num_q_heads, 0))
            )  # [H, total_q]
            lse = lse.transpose(0, 1).unsqueeze(0)  # [1, total_q, H]
        else:
            lse = None

    else:
        # Non-varlen: [B, S, H, D] -> [B, H, S, D]
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        k, v = _expand_gqa(k, v, q.shape[-3])
        if not return_lse:
            output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=sdpa_is_causal, scale=scale
            )  # [B, H, S_q, D_v]
            lse = None
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, S_q, S_kv]
            if sdpa_is_causal:
                s_q, s_kv = q.shape[-2], k.shape[-2]
                causal_mask = torch.triu(
                    torch.ones(s_q, s_kv, dtype=torch.bool, device=q.device), diagonal=1
                )
                scores = scores.masked_fill(causal_mask, float("-inf"))
            lse = torch.logsumexp(scores, dim=-1)  # [B, H, S_q]
            attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(v.dtype)
            output = torch.matmul(attn, v)  # [B, H, S_q, D_v]
            lse = lse.permute(0, 2, 1)  # [B, S_q, H]
        output = output.transpose(1, 2)  # [B, S_q, H, D_v]

    assert isinstance(output, Tensor)
    assert output.dim() == 4  # [B, S_q, H, D_v]

    if return_lse:
        assert lse is not None and lse.dim() == 3  # [B, S_q, H]
        return output, lse

    return output
