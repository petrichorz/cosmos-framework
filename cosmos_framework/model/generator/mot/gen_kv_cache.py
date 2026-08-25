# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Generation K/V cache used only by causal Text+Image-to-Video inference.

The cache keeps the static understanding-token K/V and a rolling window of
finalized clean vision blocks.  Denoising uses ``READONLY`` memory; a separate
timestep-zero prefill uses ``APPEND`` memory and is committed only after every
transformer layer has completed successfully.

``DISABLED`` is intentionally retained as an enum value for API stability, but
the causal cache state machine does not enter it.  Ordinary non-causal forwards
continue to run without a ``GenKVCacheMemoryState``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from cosmos_framework.data.generator.sequence_packing.runtime import SequencePack, from_und_gen_splits, get_gen_seq
from cosmos_framework.model.attention import attention
from cosmos_framework.model.generator.mot.attention import SplitInfo, dispatch_attention
from cosmos_framework.model.generator.utils.memory import KVToStore, MemoryState, MemoryValue


class GenKVCacheMode(str, Enum):
    DISABLED = "disabled"
    READONLY = "readonly"
    APPEND = "append"


@dataclass(frozen=True)
class CachedCleanBlock:
    block_id: int
    start_latent_frame: int
    num_latent_frames: int
    token_count: int


@dataclass
class _LayerCache:
    und_k: torch.Tensor | None = None
    und_v: torch.Tensor | None = None
    clean_k: torch.Tensor | None = None
    clean_v: torch.Tensor | None = None
    clean_len: int = 0


@dataclass
class GenKVCacheMemoryValue(MemoryValue):
    """Immutable-per-forward view consumed inside a compiled decoder layer."""

    mode: GenKVCacheMode
    und_k_cached: torch.Tensor | None
    und_v_cached: torch.Tensor | None
    clean_k_cached: torch.Tensor | None
    clean_v_cached: torch.Tensor | None
    frame_idx: int
    gen_len: int
    for_cuda_graphs: bool = False

    @property
    def supports_context_parallel_attention(self) -> bool:
        return False


class GenKVCache:
    """Per-CFG-branch, request-local rolling K/V storage."""

    def __init__(self, num_layers: int, history_blocks: int, max_clean_tokens: int) -> None:
        if num_layers < 1:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if not 1 <= history_blocks <= 16:
            raise ValueError(f"history_blocks must be in [1, 16], got {history_blocks}")
        if max_clean_tokens < 1:
            raise ValueError(f"max_clean_tokens must be positive, got {max_clean_tokens}")
        self.layers = [_LayerCache() for _ in range(num_layers)]
        self.history_blocks = history_blocks
        self.max_clean_tokens = max_clean_tokens
        self.blocks: list[CachedCleanBlock] = []

    @property
    def is_initialized(self) -> bool:
        states = [layer.und_k is not None for layer in self.layers]
        if any(states) and not all(states):
            raise RuntimeError("GenKVCache is partially initialized")
        return all(states)

    def new_state(self, mode: GenKVCacheMode, block: CachedCleanBlock | None = None) -> "GenKVCacheMemoryState":
        if mode is GenKVCacheMode.DISABLED:
            raise ValueError("DISABLED is not a causal GenKVCache state; omit memory for an ordinary forward")
        if mode is GenKVCacheMode.APPEND and block is None:
            raise ValueError("APPEND requires clean-block metadata")
        if mode is GenKVCacheMode.READONLY and not self.is_initialized:
            raise RuntimeError("READONLY requires a completed causal prefill")
        return GenKVCacheMemoryState(self, mode, block)


class GenKVCacheMemoryState(MemoryState):
    """One forward's staged view of a :class:`GenKVCache`."""

    def __init__(self, cache: GenKVCache, mode: GenKVCacheMode, block: CachedCleanBlock | None) -> None:
        self.cache = cache
        self.mode = mode
        self.block = block
        self._gen_len = 0
        self._staged: dict[int, KVToStore] = {}

    def init(self, hidden_states: dict, device: torch.device) -> None:
        del device
        self._gen_len = int(hidden_states["_num_full_tokens"])

    def read_for_layer(self, layer_idx: int) -> GenKVCacheMemoryValue:
        layer = self.cache.layers[layer_idx]
        return GenKVCacheMemoryValue(
            mode=self.mode,
            und_k_cached=layer.und_k,
            und_v_cached=layer.und_v,
            clean_k_cached=None if layer.clean_k is None else layer.clean_k[:, : layer.clean_len],
            clean_v_cached=None if layer.clean_v is None else layer.clean_v[:, : layer.clean_len],
            frame_idx=1 if self.cache.is_initialized else 0,
            gen_len=self._gen_len,
        )

    def write_for_layer(self, layer_idx: int, kv_to_store: KVToStore) -> None:
        if self.mode is GenKVCacheMode.READONLY:
            return
        if layer_idx in self._staged:
            raise RuntimeError(f"layer {layer_idx} was staged twice")
        self._staged[layer_idx] = tuple(t.detach().clone() for t in kv_to_store)  # type: ignore[assignment]

    def is_gen_only(self) -> bool:
        return self.cache.is_initialized

    def requires_natten_metadata(self) -> bool:
        return False

    def validate_append(self) -> None:
        if self.mode is not GenKVCacheMode.APPEND or self.block is None:
            raise RuntimeError("only an APPEND state can be committed")
        if len(self._staged) != len(self.cache.layers):
            raise RuntimeError(f"expected {len(self.cache.layers)} staged layers, got {len(self._staged)}")
        for layer_idx, (gen_k, gen_v, und_k, und_v) in self._staged.items():
            if gen_k.shape != gen_v.shape or gen_k.ndim != 4 or gen_k.shape[0] != 1:
                raise ValueError(f"invalid generated K/V at layer {layer_idx}: {gen_k.shape}, {gen_v.shape}")
            if gen_k.shape[1] != self.block.token_count:
                raise ValueError(
                    f"block {self.block.block_id} declares {self.block.token_count} tokens, "
                    f"but layer {layer_idx} staged {gen_k.shape[1]}"
                )
            layer = self.cache.layers[layer_idx]
            if layer.und_k is None:
                if und_k.shape != und_v.shape or und_k.ndim != 4 or und_k.shape[0] != 1:
                    raise ValueError(f"invalid understanding K/V at layer {layer_idx}")
            elif und_k.shape[1] != 0 or und_v.shape[1] != 0:
                raise ValueError("gen-only APPEND unexpectedly produced understanding tokens")

    def commit_append(self) -> None:
        self.validate_append()
        assert self.block is not None
        evicted_tokens = 0
        if len(self.cache.blocks) == self.cache.history_blocks:
            evicted_tokens = self.cache.blocks.pop(0).token_count

        for layer_idx, (gen_k, gen_v, und_k, und_v) in self._staged.items():
            layer = self.cache.layers[layer_idx]
            if layer.und_k is None:
                layer.und_k, layer.und_v = und_k, und_v
            assert layer.und_v is not None

            if layer.clean_k is None:
                shape = (1, self.cache.max_clean_tokens, gen_k.shape[2], gen_k.shape[3])
                layer.clean_k = torch.empty(shape, dtype=gen_k.dtype, device=gen_k.device)
                layer.clean_v = torch.empty(shape, dtype=gen_v.dtype, device=gen_v.device)
            assert layer.clean_v is not None
            if evicted_tokens:
                remaining = layer.clean_len - evicted_tokens
                layer.clean_k[:, :remaining].copy_(layer.clean_k[:, evicted_tokens : layer.clean_len].clone())
                layer.clean_v[:, :remaining].copy_(layer.clean_v[:, evicted_tokens : layer.clean_len].clone())
                layer.clean_len = remaining
            new_len = layer.clean_len + gen_k.shape[1]
            if new_len > self.cache.max_clean_tokens:
                raise RuntimeError(f"clean K/V capacity exceeded: {new_len} > {self.cache.max_clean_tokens}")
            layer.clean_k[:, layer.clean_len : new_len].copy_(gen_k)
            layer.clean_v[:, layer.clean_len : new_len].copy_(gen_v)
            layer.clean_len = new_len

        self.cache.blocks.append(self.block)
        self._staged.clear()

    def rollback(self) -> None:
        self._staged.clear()


def commit_cfg_append_pair(cond: GenKVCacheMemoryState, uncond: GenKVCacheMemoryState | None) -> None:
    """Validate both CFG branches before mutating either cache."""

    cond.validate_append()
    if uncond is not None:
        uncond.validate_append()
        if cond.block != uncond.block:
            raise ValueError("conditional and unconditional cache commits describe different blocks")
    cond.commit_append()
    if uncond is not None:
        uncond.commit_append()


def _attention_gen_with_cache(
    query: SequencePack,
    key: SequencePack,
    value: SequencePack,
    memory: GenKVCacheMemoryValue,
) -> tuple[SequencePack, KVToStore | None]:
    q_gen = get_gen_seq(query)
    k_current = get_gen_seq(key)[: memory.gen_len].unsqueeze(0)
    v_current = get_gen_seq(value)[: memory.gen_len].unsqueeze(0)
    k_parts = [memory.und_k_cached, memory.clean_k_cached, k_current]
    v_parts = [memory.und_v_cached, memory.clean_v_cached, v_current]
    k_full = torch.cat([part for part in k_parts if part is not None], dim=1)
    v_full = torch.cat([part for part in v_parts if part is not None], dim=1)
    result = attention(q_gen.unsqueeze(0), k_full, v_full, is_causal=False, return_lse=False)
    assert isinstance(result, torch.Tensor)
    gen_out = result.squeeze(0).flatten(-2, -1)
    return from_und_gen_splits(gen_out.new_empty(0, gen_out.shape[-1]), gen_out, query), None


def dispatch_attention_with_gen_kv_cache(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    attention_mask: object | SplitInfo,
    natten_metadata: dict | None = None,
    memory_value: MemoryValue | None = None,
    packed_key_states_normalized: SequencePack | None = None,
) -> tuple[SequencePack, KVToStore | None]:
    if isinstance(memory_value, GenKVCacheMemoryValue) and memory_value.frame_idx > 0:
        return _attention_gen_with_cache(packed_query_states, packed_key_states, packed_value_states, memory_value)
    return dispatch_attention(
        packed_query_states,
        packed_key_states,
        packed_value_states,
        attention_mask,
        natten_metadata=natten_metadata,
        memory_value=None,
        packed_key_states_normalized=packed_key_states_normalized,
    )


def install_gen_kv_cache_attention_dispatch(net: torch.nn.Module) -> list[tuple[torch.nn.Module, object]]:
    previous: list[tuple[torch.nn.Module, object]] = []
    try:
        for layer in net.language_model.model.layers:
            attn = layer.self_attn
            current = attn.dispatch_attention_fn
            if current is not dispatch_attention and current is not dispatch_attention_with_gen_kv_cache:
                raise RuntimeError(
                    "GenKVCache requires the default attention dispatcher; context parallel and other "
                    "custom dispatchers are unsupported"
                )
            previous.append((attn, current))
            attn.dispatch_attention_fn = dispatch_attention_with_gen_kv_cache
    except Exception:
        restore_gen_kv_cache_attention_dispatch(previous)
        raise
    return previous


def restore_gen_kv_cache_attention_dispatch(previous: list[tuple[torch.nn.Module, object]]) -> None:
    for attn, previous_fn in previous:
        attn.dispatch_attention_fn = previous_fn
