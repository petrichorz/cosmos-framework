# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing.runtime import from_und_gen_splits, init_sequence_pack
from cosmos_framework.model.generator.mot import gen_kv_cache as gen_kv_cache_module
from cosmos_framework.model.generator.mot.gen_kv_cache import (
    CachedCleanBlock,
    GenKVCache,
    GenKVCacheMemoryValue,
    GenKVCacheMode,
    _attention_gen_with_cache,
    commit_cfg_append_pair,
)


def _kv(gen_value: float, gen_tokens: int, und_tokens: int = 0):
    gen = torch.full((1, gen_tokens, 2, 3), gen_value)
    und = torch.full((1, und_tokens, 2, 3), -gen_value)
    return gen, gen.clone(), und, und.clone()


def _append(cache: GenKVCache, block_id: int, token_count: int, value: float, *, first: bool = False):
    block = CachedCleanBlock(block_id, block_id, 1, token_count)
    state = cache.new_state(GenKVCacheMode.APPEND, block)
    state.write_for_layer(0, _kv(value, token_count, und_tokens=2 if first else 0))
    state.commit_append()
    return state


def test_append_is_staged_until_commit_and_readonly_never_mutates():
    cache = GenKVCache(num_layers=1, history_blocks=2, max_clean_tokens=4)
    state = cache.new_state(GenKVCacheMode.APPEND, CachedCleanBlock(0, 0, 1, 1))
    state.write_for_layer(0, _kv(1, 1, und_tokens=2))
    assert not cache.is_initialized
    state.commit_append()
    assert cache.is_initialized

    readonly = cache.new_state(GenKVCacheMode.READONLY)
    readonly.write_for_layer(0, _kv(9, 2))
    assert cache.layers[0].clean_len == 1


def test_rolling_window_evicts_c0_like_an_ordinary_clean_block():
    cache = GenKVCache(num_layers=1, history_blocks=2, max_clean_tokens=4)
    _append(cache, 0, 1, 10, first=True)
    _append(cache, 1, 2, 20)
    _append(cache, 2, 2, 30)
    assert [block.block_id for block in cache.blocks] == [1, 2]
    clean = cache.layers[0].clean_k[:, : cache.layers[0].clean_len]
    assert clean.shape[1] == 4
    assert torch.equal(clean[:, :2], torch.full_like(clean[:, :2], 20))
    assert torch.equal(clean[:, 2:], torch.full_like(clean[:, 2:], 30))


def test_cfg_pair_validates_both_before_committing():
    cond = GenKVCache(1, 2, 4)
    uncond = GenKVCache(1, 2, 4)
    block = CachedCleanBlock(0, 0, 1, 1)
    cond_state = cond.new_state(GenKVCacheMode.APPEND, block)
    uncond_state = uncond.new_state(GenKVCacheMode.APPEND, block)
    cond_state.write_for_layer(0, _kv(1, 1, und_tokens=2))
    with pytest.raises(RuntimeError, match="staged layers"):
        commit_cfg_append_pair(cond_state, uncond_state)
    assert not cond.is_initialized


def test_disabled_is_not_entered_by_causal_state_machine():
    cache = GenKVCache(1, 1, 1)
    with pytest.raises(ValueError, match="not a causal"):
        cache.new_state(GenKVCacheMode.DISABLED)


def test_attention_kv_order_matches_reference_oracle(monkeypatch: pytest.MonkeyPatch):
    metadata = init_sequence_pack([2], [2], ["full"], torch.device("cpu"))
    metadata["is_sharded"] = False
    empty = torch.empty((0, 1, 2))
    query = from_und_gen_splits(empty, torch.ones((2, 1, 2)), metadata)
    current_k = torch.full((2, 1, 2), 3.0)
    current_v = torch.full((2, 1, 2), 30.0)
    key = from_und_gen_splits(empty, current_k, metadata)
    value = from_und_gen_splits(empty, current_v, metadata)
    memory = GenKVCacheMemoryValue(
        mode=GenKVCacheMode.READONLY,
        und_k_cached=torch.full((1, 1, 1, 2), 1.0),
        und_v_cached=torch.full((1, 1, 1, 2), 10.0),
        clean_k_cached=torch.full((1, 2, 1, 2), 2.0),
        clean_v_cached=torch.full((1, 2, 1, 2), 20.0),
        frame_idx=1,
        gen_len=2,
    )
    captured: dict[str, torch.Tensor] = {}

    def reference_attention(query, key, value, **kwargs):
        del kwargs
        captured.update(query=query, key=key, value=value)
        return torch.zeros_like(query)

    monkeypatch.setattr(gen_kv_cache_module, "attention", reference_attention)
    output, kv_to_store = _attention_gen_with_cache(query, key, value, memory)

    assert kv_to_store is None
    assert output["full_only_seq"].shape == (2, 2)
    assert torch.equal(captured["key"][:, :1], torch.full((1, 1, 1, 2), 1.0))
    assert torch.equal(captured["key"][:, 1:3], torch.full((1, 2, 1, 2), 2.0))
    assert torch.equal(captured["key"][:, 3:], torch.full((1, 2, 1, 2), 3.0))
    assert torch.equal(captured["value"][:, 3:], torch.full((1, 2, 1, 2), 30.0))
