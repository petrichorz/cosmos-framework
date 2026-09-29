# SPDX-License-Identifier: OpenMDW-1.1
"""Request-local action KV/attention cache, isolated between CFG branches.

Caches immutable history/condition attention within a diffusion block. Projection
and FFN work is still replayed; this is a correctness-first cache, not the video
rolling-cache optimization. A new block invalidates all entries.
"""

from contextlib import contextmanager

import torch

from cosmos_framework.data.generator.sequence_packing.runtime import (
    from_mode_splits,
    get_all_seq,
    get_causal_seq,
    get_full_only_seq,
)

from .teacher_forcing_tnd import TNDChunk, TNDPlan, teacher_forcing_tnd_attention


def target_only_plan(layout, plan):
    target = torch.where(layout.stream_ids[layout.gen_query_indexes] == 1)[0]
    mapping = torch.full((plan.num_queries,), -1, dtype=torch.long, device=target.device)
    mapping[target] = torch.arange(target.numel(), device=target.device)
    chunks = []
    for c in plan.chunks:
        qs = []
        ks = []
        qe = []
        ke = []
        q0 = k0 = 0
        for q1, k1 in zip(c.query_ends, c.kv_ends, strict=True):
            q = c.query_indexes[q0:q1]
            if int(mapping[q[0]]) >= 0:
                qs.append(mapping[q])
                ks.append(c.kv_indexes[k0:k1])
                qe.append(sum(x.numel() for x in qs))
                ke.append(sum(x.numel() for x in ks))
            q0, k0 = q1, k1
        if qs:
            chunks.append(TNDChunk(torch.cat(qs), torch.cat(ks), tuple(qe), tuple(ke)))
    return target, TNDPlan(tuple(chunks), target.numel(), plan.num_keys)


@contextmanager
def action_attention_cache(net, enabled):
    """Yield a controller; caller sets key=(block, cfg_branch) before each forward."""
    controller = {"key": None, "entries": {}}
    if not enabled:
        yield controller
        return
    previous = []

    def wrapper(original, layer):
        def run(
            packed_query_states,
            packed_key_states,
            packed_value_states,
            attention_mask,
            natten_metadata=None,
            memory_value=None,
            packed_key_states_normalized=None,
        ):
            if torch.is_grad_enabled():
                raise RuntimeError("Action inference cache cannot run with gradients enabled")
            cachekey = (controller["key"], layer)
            entries = controller["entries"]
            if cachekey not in entries:
                result, store = original(
                    packed_query_states,
                    packed_key_states,
                    packed_value_states,
                    attention_mask,
                    natten_metadata=natten_metadata,
                    memory_value=memory_value,
                    packed_key_states_normalized=packed_key_states_normalized,
                )
                layout = attention_mask.layout
                target, plan = target_only_plan(layout, attention_mask.tnd_plan)
                fixed = torch.where(layout.stream_ids != 1)[0]
                key = get_all_seq(
                    packed_key_states_normalized if packed_key_states_normalized is not None else packed_key_states
                )
                value = get_all_seq(packed_value_states)
                entries[cachekey] = (
                    target,
                    plan,
                    fixed,
                    key[fixed].clone(),
                    value[fixed].clone(),
                    get_causal_seq(result)[0].clone(),
                    get_full_only_seq(result)[0].clone(),
                )
                return result, store
            target, plan, fixed, kcache, vcache, und, gen = entries[cachekey]
            key = get_all_seq(
                packed_key_states_normalized if packed_key_states_normalized is not None else packed_key_states
            ).clone()
            value = get_all_seq(packed_value_states).clone()
            key[fixed] = kcache
            value[fixed] = vcache
            q = get_full_only_seq(packed_query_states)[0][target]
            output = gen.clone()
            if target.numel():
                output[target] = teacher_forcing_tnd_attention(
                    q, key[: plan.num_keys], value[: plan.num_keys], plan
                ).flatten(-2, -1)
            return from_mode_splits(und, output, packed_query_states), None

        return run

    try:
        for i, layer in enumerate(net.language_model.model.layers):
            attn = layer.self_attn
            previous.append((attn, attn.dispatch_attention_fn))
            attn.dispatch_attention_fn = wrapper(attn.dispatch_attention_fn, i)
        yield controller
    finally:
        for attn, fn in previous:
            attn.dispatch_attention_fn = fn
        controller["entries"].clear()
