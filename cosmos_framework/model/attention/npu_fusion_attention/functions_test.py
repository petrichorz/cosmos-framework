# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from cosmos_framework.data.generator.sequence_packing.runtime import init_sequence_pack
from cosmos_framework.model.attention.npu_fusion_attention import functions


class _FakeOffsets:
    def __init__(self, values: list[int]):
        self.values = values
        self._version = 0
        self.tolist_calls = 0

    def tolist(self) -> list[int]:
        self.tolist_calls += 1
        return list(self.values)


def test_actual_seq_lengths_reuses_unchanged_tensor_metadata(monkeypatch):
    monkeypatch.delenv("COSMOS_ASCEND_SEQUENCE_PACKING_TOLIST_OPT", raising=False)
    functions._actual_seq_lengths_cache.clear()
    offsets = _FakeOffsets([0, 4, 9])

    assert functions._ascend_actual_seq_lengths(offsets) == [4, 9]
    assert functions._ascend_actual_seq_lengths(offsets) == [4, 9]
    assert offsets.tolist_calls == 1


def test_actual_seq_lengths_invalidates_cache_after_mutation(monkeypatch):
    monkeypatch.delenv("COSMOS_ASCEND_SEQUENCE_PACKING_TOLIST_OPT", raising=False)
    functions._actual_seq_lengths_cache.clear()
    offsets = _FakeOffsets([0, 4, 9])
    assert functions._ascend_actual_seq_lengths(offsets) == [4, 9]

    offsets.values = [0, 5, 10]
    offsets._version += 1

    assert functions._ascend_actual_seq_lengths(offsets) == [5, 10]
    assert offsets.tolist_calls == 2


def test_sequence_pack_offsets_keep_host_lengths_for_npu_backend(monkeypatch):
    monkeypatch.delenv("COSMOS_ASCEND_SEQUENCE_PACKING_TOLIST_OPT", raising=False)
    pack = init_sequence_pack(
        sample_lens=[7, 11],
        split_lens=[2, 5, 3, 8],
        attn_modes=["causal", "full", "causal", "full"],
        device="cpu",
    )

    assert functions._ascend_actual_seq_lengths(pack["sample_offsets"]) == [7, 18]
    assert functions._ascend_actual_seq_lengths(pack["_causal_seq_offsets"]) == [2, 5]
    assert functions._ascend_actual_seq_lengths(pack["_full_only_seq_offsets"]) == [5, 13]


def test_disabled_optimization_repeats_host_conversion_and_skips_metadata(monkeypatch):
    monkeypatch.setenv("COSMOS_ASCEND_SEQUENCE_PACKING_TOLIST_OPT", "0")
    functions._actual_seq_lengths_cache.clear()
    offsets = _FakeOffsets([0, 4, 9])

    assert functions._ascend_actual_seq_lengths(offsets) == [4, 9]
    assert functions._ascend_actual_seq_lengths(offsets) == [4, 9]
    assert offsets.tolist_calls == 2
    assert not functions._actual_seq_lengths_cache

    pack = init_sequence_pack(
        sample_lens=[7, 11],
        split_lens=[2, 5, 3, 8],
        attn_modes=["causal", "full", "causal", "full"],
        device="cpu",
    )
    assert not hasattr(pack["sample_offsets"], "_cosmos_actual_seq_lengths")
    assert not hasattr(pack["_causal_seq_offsets"], "_cosmos_actual_seq_lengths")
    assert not hasattr(pack["_full_only_seq_offsets"], "_cosmos_actual_seq_lengths")
