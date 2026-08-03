# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import importlib.util
import sys
from types import ModuleType

import pytest
import torch

if importlib.util.find_spec("loguru") is None:
    log_stub = ModuleType("cosmos_framework.utils.log")
    for log_method in ("critical", "debug", "exception", "info", "success", "trace", "warning"):
        setattr(log_stub, log_method, lambda *args, **kwargs: None)
    sys.modules["cosmos_framework.utils.log"] = log_stub

from cosmos_framework.data.generator.sequence_packing.runtime import (
    from_all_seq,
    get_all_seq,
    get_gen_seq,
)
from cosmos_framework.data.generator.sequence_packing.teacher_forcing import (
    build_dense_teacher_forcing_gen_mask,
    build_teacher_forcing_layout,
)
from cosmos_framework.model.generator.mot.attention import (
    TeacherForcingAttentionInfo,
    build_packed_sequence,
    dispatch_attention,
)
from cosmos_framework.model.generator.mot.teacher_forcing_attention import teacher_forcing_dense_attention


def _explicit_attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    allowed_mask: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    repeats = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(repeats, dim=1)
    value = value.repeat_interleave(repeats, dim=1)
    scale = query.shape[-1] ** -0.5 if scale is None else scale
    scores = torch.einsum("qhd,khd->hqk", query, key) * scale
    scores = scores.masked_fill(~allowed_mask.unsqueeze(0), float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,khd->qhd", probabilities, value)


def _make_inputs(dtype: torch.dtype = torch.float64):
    generator = torch.Generator().manual_seed(123)
    query = torch.randn(5, 4, 3, dtype=dtype, generator=generator)
    key = torch.randn(7, 2, 3, dtype=dtype, generator=generator)
    value = torch.randn(7, 2, 3, dtype=dtype, generator=generator)
    allowed_mask = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 0, 1, 1, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0],
            [1, 1, 0, 0, 0, 1, 0],
        ],
        dtype=torch.bool,
    )
    return query, key, value, allowed_mask


def test_teacher_forcing_dense_attention_matches_independent_gqa_reference():
    query, key, value, allowed_mask = _make_inputs()

    actual = teacher_forcing_dense_attention(query, key, value, allowed_mask)
    expected = _explicit_attention_reference(query, key, value, allowed_mask)

    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_teacher_forcing_dense_attention_gradients_match_independent_reference():
    query, key, value, allowed_mask = _make_inputs()
    actual_inputs = [tensor.detach().clone().requires_grad_() for tensor in (query, key, value)]
    expected_inputs = [tensor.detach().clone().requires_grad_() for tensor in (query, key, value)]
    output_weight = torch.linspace(0.1, 1.0, query.numel(), dtype=query.dtype).reshape_as(query)

    actual = teacher_forcing_dense_attention(*actual_inputs, allowed_mask, scale=0.7)
    expected = _explicit_attention_reference(*expected_inputs, allowed_mask, scale=0.7)
    (actual * output_weight).sum().backward()
    (expected * output_weight).sum().backward()

    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    for actual_input, expected_input in zip(actual_inputs, expected_inputs):
        torch.testing.assert_close(actual_input.grad, expected_input.grad, atol=1e-11, rtol=1e-11)


def test_teacher_forcing_dense_attention_ignores_masked_values():
    query, key, value, allowed_mask = _make_inputs(torch.float32)
    baseline = teacher_forcing_dense_attention(query, key, value, allowed_mask)
    modified_value = value.clone()
    modified_value[-1] += 100_000

    unchanged = teacher_forcing_dense_attention(query, key, modified_value, allowed_mask)

    torch.testing.assert_close(unchanged, baseline)


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda q, k, v, m: (q, k, v, m.float()), "bool"),
        (lambda q, k, v, m: (q, k, v, m[:-1]), "shape"),
        (lambda q, k, v, m: (q, k, v, m.index_fill(1, torch.arange(m.shape[1]), False)), "visible key"),
        (lambda q, k, v, m: (q[:, :3], k, v, m), "divisible"),
        (lambda q, k, v, m: (q, k[:, :, :2], v, m), "head dimension"),
    ],
)
def test_teacher_forcing_dense_attention_rejects_invalid_inputs(mutate, error: str):
    query, key, value, allowed_mask = _make_inputs(torch.float32)
    query, key, value, allowed_mask = mutate(query, key, value, allowed_mask)

    with pytest.raises((TypeError, ValueError), match=error):
        teacher_forcing_dense_attention(query, key, value, allowed_mask)


def _make_teacher_forcing_packs():
    layout = build_teacher_forcing_layout(
        und_token_counts=[1, 2],
        vision_token_shapes=[(2, 1, 1), (1, 1, 2)],
        block_size=1,
        history_blocks=1,
    )
    generator = torch.Generator().manual_seed(456)
    query = torch.randn(sum(layout.sample_lens), 4, 3, generator=generator)
    key = torch.randn(sum(layout.sample_lens), 2, 3, generator=generator)
    value = torch.randn(sum(layout.sample_lens), 2, 3, generator=generator)
    common_kwargs = dict(
        joint_attn_implementation="three_way",
        attn_modes=list(layout.attn_modes),
        split_lens=list(layout.split_lens),
        sample_lens=list(layout.sample_lens),
        packed_und_token_indexes=torch.nonzero(layout.stream_ids == -1, as_tuple=True)[0],
        packed_gen_token_indexes=layout.gen_query_indexes,
        num_heads=4,
        head_dim=3,
        num_layers=1,
        teacher_forcing_layout=layout,
        teacher_forcing_max_sequence_length=layout.source_sequence_indexes.numel(),
    )
    query_pack, attention_meta, natten_metadata = build_packed_sequence(packed_sequence=query, **common_kwargs)
    key_pack, _, _ = build_packed_sequence(packed_sequence=key, **common_kwargs)
    value_pack, _, _ = build_packed_sequence(packed_sequence=value, **common_kwargs)
    return layout, query, key, value, query_pack, key_pack, value_pack, attention_meta, natten_metadata


def test_build_packed_sequence_constructs_teacher_forcing_attention_info_without_natten():
    layout, _, _, _, _, _, _, attention_meta, natten_metadata = _make_teacher_forcing_packs()

    assert isinstance(attention_meta, TeacherForcingAttentionInfo)
    assert attention_meta.layout is layout
    torch.testing.assert_close(
        attention_meta.dense_gen_mask,
        build_dense_teacher_forcing_gen_mask(
            layout,
            max_sequence_length=layout.source_sequence_indexes.numel(),
        ),
    )
    assert natten_metadata is None


def test_dispatch_teacher_forcing_attention_matches_unified_dense_gen_attention():
    layout, _, _, _, query_pack, key_pack, value_pack, attention_meta, _ = _make_teacher_forcing_packs()

    output_pack, kv_to_store = dispatch_attention(
        query_pack,
        key_pack,
        value_pack,
        attention_meta,
    )

    expected_gen = teacher_forcing_dense_attention(
        get_all_seq(query_pack)[layout.gen_query_indexes],
        get_all_seq(key_pack),
        get_all_seq(value_pack),
        attention_meta.dense_gen_mask,
    ).flatten(-2, -1)
    torch.testing.assert_close(get_gen_seq(output_pack)[: expected_gen.shape[0]], expected_gen)
    assert kv_to_store is None


def test_dispatch_teacher_forcing_attention_uses_normalized_und_keys_for_gen():
    layout, _, _, _, query_pack, key_pack, value_pack, attention_meta, _ = _make_teacher_forcing_packs()
    normalized_all_keys = get_all_seq(key_pack).clone()
    und_indexes = torch.nonzero(layout.stream_ids == -1, as_tuple=True)[0]
    normalized_all_keys[und_indexes] += 10
    normalized_key_pack = from_all_seq(normalized_all_keys, key_pack)

    output_pack, _ = dispatch_attention(
        query_pack,
        key_pack,
        value_pack,
        attention_meta,
        packed_key_states_normalized=normalized_key_pack,
    )

    expected_gen = teacher_forcing_dense_attention(
        get_all_seq(query_pack)[layout.gen_query_indexes],
        normalized_all_keys,
        get_all_seq(value_pack),
        attention_meta.dense_gen_mask,
    ).flatten(-2, -1)
    torch.testing.assert_close(get_gen_seq(output_pack)[: expected_gen.shape[0]], expected_gen)


def test_build_packed_sequence_applies_teacher_forcing_sequence_length_guard():
    layout = build_teacher_forcing_layout(
        und_token_counts=[1],
        vision_token_shapes=[(2, 1, 1)],
        block_size=1,
        history_blocks=1,
    )
    packed_sequence = torch.randn(sum(layout.sample_lens), 2, 4)

    with pytest.raises(ValueError, match="max_sequence_length"):
        build_packed_sequence(
            "three_way",
            packed_sequence=packed_sequence,
            attn_modes=list(layout.attn_modes),
            split_lens=list(layout.split_lens),
            sample_lens=list(layout.sample_lens),
            packed_und_token_indexes=torch.tensor([0]),
            packed_gen_token_indexes=layout.gen_query_indexes,
            num_heads=2,
            head_dim=4,
            num_layers=1,
            teacher_forcing_layout=layout,
            teacher_forcing_max_sequence_length=1,
        )


def test_build_packed_sequence_rejects_teacher_forcing_layout_geometry_mismatch():
    layout = build_teacher_forcing_layout(
        und_token_counts=[1],
        vision_token_shapes=[(2, 1, 1)],
        block_size=1,
        history_blocks=1,
    )
    packed_sequence = torch.randn(sum(layout.sample_lens), 2, 4)

    with pytest.raises(ValueError, match="layout geometry"):
        build_packed_sequence(
            "three_way",
            packed_sequence=packed_sequence,
            attn_modes=["causal", "full"],
            split_lens=[2, 3],
            sample_lens=[5],
            packed_und_token_indexes=torch.tensor([0]),
            packed_gen_token_indexes=layout.gen_query_indexes,
            num_heads=2,
            head_dim=4,
            num_layers=1,
            teacher_forcing_layout=layout,
            teacher_forcing_max_sequence_length=20,
        )
