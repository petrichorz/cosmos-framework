# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Exact teacher-forcing visibility as independent maskless attention groups."""

import warnings
from dataclasses import dataclass

import torch

from cosmos_framework.data.generator.sequence_packing.teacher_forcing import TeacherForcingLayout
from cosmos_framework.model.attention.npu_fusion_attention.functions import (
    NPU_FUSION_ATTENTION_TND_MAX_SEQUENCES,
    NPU_FUSION_ATTENTION_TND_MAX_TOKENS,
)


@dataclass(frozen=True)
class TNDChunk:
    query_indexes: torch.Tensor
    kv_indexes: torch.Tensor
    query_ends: tuple[int, ...]
    kv_ends: tuple[int, ...]


@dataclass(frozen=True)
class TNDPlan:
    chunks: tuple[TNDChunk, ...]
    num_queries: int
    num_keys: int


def build_tnd_plan(
    layout: TeacherForcingLayout,
    *,
    device: torch.device | str,
    max_kv_tokens: int = 131072,
) -> TNDPlan:
    """Build linear indices once per batch, without any token-pair allocation.

    Groups preserve GEN query order. KV repetitions are deliberate and their
    gradients must accumulate into the original tokens. Chunking limits each
    operator's TND length, but does not itself bound autograd's saved tensors.
    """
    if not 1 <= max_kv_tokens <= NPU_FUSION_ATTENTION_TND_MAX_TOKENS:
        raise ValueError(f"max_kv_tokens must be in [1, {NPU_FUSION_ATTENTION_TND_MAX_TOKENS}]")
    blocks = layout.block_ids.cpu().tolist()
    chunks = []
    qs, ks, qe, ke = [], [], [], []
    qoffset = koffset = 0

    def flush():
        if qe:
            chunks.append(
                TNDChunk(
                    torch.tensor(qs, dtype=torch.long, device=device),
                    torch.tensor(ks, dtype=torch.long, device=device),
                    tuple(qe),
                    tuple(ke),
                )
            )
            qs.clear()
            ks.clear()
            qe.clear()
            ke.clear()

    for sample, (length, history) in enumerate(zip(layout.sample_lens, layout.geometry.history_blocks, strict=True)):
        und = layout.split_lens[2 * sample]  # split lens [s1_und_len, s1_gen_len, s2_und_len, s2_gen_len,...]
        vision = layout.split_lens[2 * sample + 1] // 2
        clean_start = koffset + und
        sample_blocks = blocks[clean_start : clean_start + vision]
        starts = [0] + [i for i in range(1, vision) if sample_blocks[i] != sample_blocks[i - 1]] + [vision] # 获取每个block的起始位置
        for stream in (0, 1):
            for block, (start, end) in enumerate(zip(starts[:-1], starts[1:], strict=True)):
                history_start = starts[max(0, block - history)]
                group_k = list(range(koffset, clean_start)) # input und k
                group_k.extend(range(clean_start + history_start, clean_start + start)) # history clean
                group_k.extend(range(clean_start + stream * vision + start, clean_start + stream * vision + end)) # 当前帧，用stream区分clean/noise
                if len(group_k) > NPU_FUSION_ATTENTION_TND_MAX_TOKENS:
                    raise ValueError("one teacher-forcing KV group exceeds the TND token limit")
                if qe and (
                    len(ks) + len(group_k) > max_kv_tokens
                    or len(qe) == NPU_FUSION_ATTENTION_TND_MAX_SEQUENCES
                ):
                    flush()
                qs.extend(range(qoffset + stream * vision + start, qoffset + stream * vision + end))
                ks.extend(group_k)
                qe.append(len(qs))
                ke.append(len(ks))
        qoffset += 2 * vision
        koffset += length
    flush()
    return TNDPlan(tuple(chunks), qoffset, koffset)


class _GatherGroups(torch.autograd.Function):
    """Accumulate shared KV gradients in FP32 before casting once to BF16."""

    @staticmethod
    def forward(ctx, tensor, *indexes):
        ctx.save_for_backward(*indexes)
        ctx.shape = tensor.shape
        ctx.dtype = tensor.dtype
        return tuple(tensor.index_select(0, index) for index in indexes)

    @staticmethod
    def backward(ctx, *gradients):
        dtype = torch.float32 if ctx.dtype in (torch.bfloat16, torch.float16) else ctx.dtype
        result = torch.zeros(ctx.shape, dtype=dtype, device=gradients[0].device)
        for index, gradient in zip(ctx.saved_tensors, gradients, strict=True):
            result.index_add_(0, index, gradient.to(dtype))
        return (result.to(ctx.dtype),) + (None,) * len(gradients)


class _NPUTeacherForcing(torch.autograd.Function):
    """Save original KV, rematerializing only one gathered chunk in backward.

    This training operator has no dropout and supports first-order gradients.
    Native FA statistics are passed back unchanged; no LSE merge is involved.
    """

    @staticmethod
    def forward(ctx, query, key, value, plan, scale):
        import torch_npu

        outputs, statistics, rng = [], [], []
        for chunk in plan.chunks:
            q = query.index_select(0, chunk.query_indexes)
            k = key.index_select(0, chunk.kv_indexes)
            v = value.index_select(0, chunk.kv_indexes)
            result = torch_npu.npu_fusion_attention(
                q,
                k,
                v,
                head_num=q.shape[1],
                input_layout="TND",
                atten_mask=None,
                scale=scale,
                keep_prob=1.0,
                sparse_mode=0,
                actual_seq_qlen=chunk.query_ends,
                actual_seq_kvlen=chunk.kv_ends,
            )
            outputs.append(result[0])
            statistics.extend(result[1:3])
            rng.append(result[4:7])
            del q, k, v, result
        output = torch.cat(outputs)
        ctx.save_for_backward(query, key, value, output, *statistics)
        ctx.plan, ctx.scale, ctx.rng = plan, scale, rng
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        query, key, value, output, *statistics = ctx.saved_tensors
        dq = torch.empty_like(query)
        dk = torch.zeros_like(key, dtype=torch.float32)
        dv = torch.zeros_like(value, dtype=torch.float32)
        offset = 0
        for i, chunk in enumerate(ctx.plan.chunks):
            end = offset + chunk.query_ends[-1]
            q = query.index_select(0, chunk.query_indexes)
            k = key.index_select(0, chunk.kv_indexes)
            v = value.index_select(0, chunk.kv_indexes)
            seed, rng_offset, numels = ctx.rng[i]
            gradients = torch.ops.npu.npu_fusion_attention_grad.default(
                q,
                k,
                v,
                grad_output[offset:end].contiguous(),
                q.shape[1],
                "TND",
                atten_mask=None,
                softmax_max=statistics[2 * i],
                softmax_sum=statistics[2 * i + 1],
                attention_in=output[offset:end].contiguous(),
                scale_value=ctx.scale,
                keep_prob=1.0,
                sparse_mode=0,
                seed=seed,
                offset=rng_offset,
                numels=numels,
                actual_seq_qlen=chunk.query_ends,
                actual_seq_kvlen=chunk.kv_ends,
            )
            dq[offset:end] = gradients[0]
            dk.index_add_(0, chunk.kv_indexes, gradients[1].float())
            dv.index_add_(0, chunk.kv_indexes, gradients[2].float())
            offset = end
            del q, k, v, gradients
        return dq, dk.to(key.dtype), dv.to(value.dtype), None, None


def teacher_forcing_tnd_attention(query, key, value, plan: TNDPlan, *, scale=None):
    """One softmax over each group's exact KV set; clean KV stays differentiable."""
    if query.shape[0] != plan.num_queries or key.shape[0] != plan.num_keys or value.shape != key.shape:
        raise ValueError("Q/K/V do not match the teacher-forcing TND plan")
    outputs = []
    scale = query.shape[-1] ** -0.5 if scale is None else scale
    if query.device.type == "npu":
        return _NPUTeacherForcing.apply(query, key, value, plan, scale)
    warnings.warn(
        f"grouped_tnd fusion is only supported on NPU; falling back to PyTorch SDPA on {query.device.type}",
        RuntimeWarning,
        stacklevel=2,
    )
    indexes = tuple(chunk.kv_indexes for chunk in plan.chunks)
    keys = _GatherGroups.apply(key, *indexes)
    values = _GatherGroups.apply(value, *indexes)
    for chunk, k, v in zip(plan.chunks, keys, values, strict=True):
        q = query.index_select(0, chunk.query_indexes)
        parts = []
        qstart = kstart = 0
        for qend, kend in zip(chunk.query_ends, chunk.kv_ends, strict=True):
            repeats = q.shape[1] // k.shape[1]
            parts.append(
                torch.nn.functional.scaled_dot_product_attention(
                    q[qstart:qend].transpose(0, 1),
                    k[kstart:kend].repeat_interleave(repeats, dim=1).transpose(0, 1),
                    v[kstart:kend].repeat_interleave(repeats, dim=1).transpose(0, 1),
                    dropout_p=0.0,
                    scale=scale,
                ).transpose(0, 1)
            )
            qstart, kstart = qend, kend
        out = torch.cat(parts)
        outputs.append(out)
    return torch.cat(outputs)
