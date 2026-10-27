# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Episode-local observed-video KV, independent of predicted video and state.

Use compact training-compatible action packs: prefill the independent L0 or a
complete clean training block jointly. Prediction slots are never committed.
No historical pixels, actions or states are replayed through the transformer.
Text and the compact pack still run projections/FFNs on each forward.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

from cosmos_framework.data.generator.sequence_packing.runtime import (
    from_mode_splits,
    get_all_seq,
    get_causal_seq,
    get_full_only_seq,
)
from cosmos_framework.model.attention import attention
from cosmos_framework.model.attention.masks import CausalType


def attend(q, k, v, *, causal=False):
    return attention(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        is_causal=causal,
        causal_type=CausalType.TopLeft if causal else None,
    ).squeeze(0)


@dataclass
class LayerKV:
    text_k: torch.Tensor
    text_v: torch.Tensor
    text_out: torch.Tensor
    # One entry per observed latent frame, with its absolute episode index.
    video: list[tuple[int, torch.Tensor, torch.Tensor]] = field(default_factory=list)


class ObservedVideoCache:
    def __init__(self, num_layers, max_frames):
        if num_layers < 1 or max_frames < 1:
            raise ValueError("Layer count and history capacity must be positive")
        self.num_layers, self.max_frames = num_layers, max_frames
        self.layers = {}

    def clear(self):
        self.layers.clear()

    @contextmanager
    def forward(self, net, *, append_frames=None, clean_block=-1):
        """Stage an observed block atomically, or read history during denoising.

        The state query attends only to text and itself. Target queries attend
        to text, observed video, current state and current video/action targets.
        Unknown clean-video slots and duplicate condition slots are never keys.
        """
        if append_frames is not None:
            append_frames = tuple(append_frames)
            if not append_frames or append_frames != tuple(range(append_frames[0], append_frames[-1] + 1)):
                raise ValueError("Observed latent frame IDs must be nonempty and consecutive")
            if clean_block == -1:
                if append_frames != (0,):
                    raise ValueError("Only the initial observation may prefill the independent prefix")
            elif clean_block != 0 or append_frames[0] < 1 or (append_frames[0] - 1) % len(append_frames):
                raise ValueError("Observed frames must form one aligned training block")
            if len(append_frames) > self.max_frames:
                raise ValueError("Observation block exceeds history capacity")
        if append_frames is None and len(self.layers) != self.num_layers:
            raise RuntimeError("Denoising requires a completed observation prefill")
        staged, previous = {}, []

        def wrapper(layer_id):
            def dispatch(
                packed_query_states,
                packed_key_states,
                packed_value_states,
                attention_mask,
                natten_metadata=None,
                memory_value=None,
                packed_key_states_normalized=None,
            ):
                if torch.is_grad_enabled() or memory_value is not None:
                    raise RuntimeError("Observed video cache requires no-grad, single-rank inference without memory")
                layout = attention_mask.layout
                roles = layout.stream_ids
                q = get_full_only_seq(packed_query_states)[0]
                k = get_all_seq(
                    packed_key_states if packed_key_states_normalized is None else packed_key_states_normalized
                )
                v = get_all_seq(packed_value_states)
                entry = self.layers.get(layer_id)
                if entry is None:
                    uq = get_causal_seq(packed_query_states)[0]
                    uk = get_causal_seq(packed_key_states)[0]
                    uv = get_causal_seq(packed_value_states)[0]
                    entry = LayerKV(k[roles == -1].clone(), v[roles == -1].clone(), attend(uq, uk, uv, causal=True))
                gen_roles = roles[layout.gen_query_indexes]
                out = q.new_zeros(q.shape)
                # A clean block reads the full old history; evict only after computing its KV.
                past = entry.video
                history_k = [item[1] for item in past]
                history_v = [item[2] for item in past]
                if append_frames is not None:
                    if clean_block == 0 and len(append_frames) != layout.geometry.block_sizes[0]:
                        raise ValueError("Observation prefill requires one complete training block")
                    expected = entry.video[-1][0] + 1 if entry.video else 0
                    if append_frames[0] != expected:
                        raise ValueError("Observed latent frames must be appended once, in order")
                    clean = (roles == 0) & (layout.block_ids == clean_block)
                    query = (gen_roles == 0) & (layout.block_ids[layout.gen_query_indexes] == clean_block)
                    ck, cv = k[clean], v[clean]
                    if not len(ck) or len(ck) % len(append_frames):
                        raise ValueError("Clean block tokens must split evenly across observed frames")
                    out[query] = attend(
                        q[query], torch.cat([entry.text_k, *history_k, ck]), torch.cat([entry.text_v, *history_v, cv])
                    )
                    frames = [
                        (frame, fk.detach().clone(), fv.detach().clone())
                        for frame, fk, fv in zip(
                            append_frames, ck.chunk(len(append_frames)), cv.chunk(len(append_frames)), strict=True
                        )
                    ]
                    staged[layer_id] = LayerKV(
                        entry.text_k,
                        entry.text_v,
                        entry.text_out,
                        [*past, *frames][-self.max_frames :],
                    )
                else:
                    state, target = roles == 3, roles == 1
                    out[gen_roles == 3] = attend(
                        q[gen_roles == 3], torch.cat([entry.text_k, k[state]]), torch.cat([entry.text_v, v[state]])
                    )
                    out[gen_roles == 1] = attend(
                        q[gen_roles == 1],
                        torch.cat([entry.text_k, *history_k, k[state], k[target]]),
                        torch.cat([entry.text_v, *history_v, v[state], v[target]]),
                    )
                return from_mode_splits(entry.text_out.flatten(-2), out.flatten(-2), packed_query_states), None

            return dispatch

        try:
            layers = net.language_model.model.layers
            if len(layers) != self.num_layers:
                raise ValueError("Network/cache layer counts differ")
            for i, layer in enumerate(layers):
                attn = layer.self_attn
                previous.append((attn, attn.dispatch_attention_fn))
                attn.dispatch_attention_fn = wrapper(i)
            yield
            if append_frames is not None:
                if len(staged) != self.num_layers:
                    raise RuntimeError("Observation prefill did not complete every transformer layer")
                self.layers = staged
        finally:
            for attn, original in previous:
                attn.dispatch_attention_fn = original
