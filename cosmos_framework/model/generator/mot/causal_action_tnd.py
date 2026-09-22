# SPDX-License-Identifier: OpenMDW-1.1
"""Linear-memory TND planner for multimodal causal teacher forcing."""

import torch

from cosmos_framework.model.attention.npu_fusion_attention.functions import (
    NPU_FUSION_ATTENTION_TND_MAX_SEQUENCES as MAX_SEQUENCES,
)
from cosmos_framework.model.attention.npu_fusion_attention.functions import (
    NPU_FUSION_ATTENTION_TND_MAX_TOKENS as MAX_TOKENS,
)

from .teacher_forcing_tnd import TNDChunk, TNDPlan


def build_action_tnd_plan(layout, *, device, max_kv_tokens=131072):
    if not 1 <= max_kv_tokens <= MAX_TOKENS:
        raise ValueError(f"max_kv_tokens must be in [1, {MAX_TOKENS}]")
    roles, blocks, samples = (x.cpu().tolist() for x in (layout.stream_ids, layout.block_ids, layout.sample_ids))
    gen = layout.gen_query_indexes.cpu().tolist()
    buckets = {}
    for k, (s, r, b) in enumerate(zip(samples, roles, blocks, strict=True)):
        buckets.setdefault((s, r, b), []).append(k)
    chunks, qs, ks, qe, ke = [], [], [], [], []

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

    start = 0
    while start < len(gen):
        k = gen[start]
        s, r, b = samples[k], roles[k], blocks[k]
        end = start + 1
        while end < len(gen) and (samples[gen[end]], roles[gen[end]], blocks[gen[end]]) == (s, r, b):
            end += 1
        keys = list(buckets.get((s, -1, -1), []))
        if r == 3:
            keys += buckets.get((s, 3, b), [])
        elif b == -1:
            keys += buckets.get((s, r, -1), [])
        else:
            for past in range(max(-1, b - layout.geometry.history_blocks[s]), b):
                keys += buckets.get((s, 0, past), [])
            if r != 0:
                keys += buckets.get((s, 3, b), [])
            for current_role in (0,) if r == 0 else (2,) if r == 2 else (1, 2):
                keys += buckets.get((s, current_role, b), [])
        keys.sort()
        if len(keys) > MAX_TOKENS or end - start > MAX_TOKENS:
            raise ValueError("One causal action attention group exceeds the hardware TND limit")
        if qe and (
            len(ks) + len(keys) > max_kv_tokens or len(qs) + end - start > MAX_TOKENS or len(qe) == MAX_SEQUENCES
        ):
            flush()
        qs.extend(range(start, end))
        ks.extend(keys)
        qe.append(len(qs))
        ke.append(len(ks))
        start = end
    flush()
    return TNDPlan(tuple(chunks), len(gen), len(roles))
