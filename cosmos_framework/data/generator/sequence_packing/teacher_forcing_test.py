# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing.teacher_forcing import (
    TeacherForcingLayout,
    TeacherForcingStream,
    build_dense_teacher_forcing_gen_mask,
    build_teacher_forcing_layout,
    sample_teacher_forcing_parameters,
)


def test_teacher_forcing_stream_values_are_stable():
    assert int(TeacherForcingStream.UND) == -1
    assert int(TeacherForcingStream.CLEAN) == 0
    assert int(TeacherForcingStream.NOISY) == 1


def test_teacher_forcing_layout_is_frozen():
    empty = torch.empty(0, dtype=torch.long)
    layout = TeacherForcingLayout(
        block_size=1,
        history_blocks=1,
        original_sample_lens=(),
        sample_lens=(),
        split_lens=(),
        attn_modes=(),
        source_sequence_indexes=empty,
        sample_ids=empty,
        stream_ids=empty,
        block_ids=empty,
        gen_query_indexes=empty,
        clean_token_indexes=empty,
        noisy_output_indexes=empty,
    )

    with pytest.raises(FrozenInstanceError):
        layout.block_size = 2


def test_sample_teacher_forcing_parameters_is_reproducible():
    generator_a = torch.Generator().manual_seed(1234)
    generator_b = torch.Generator().manual_seed(1234)

    draws_a = [sample_teacher_forcing_parameters(generator=generator_a) for _ in range(32)]
    draws_b = [sample_teacher_forcing_parameters(generator=generator_b) for _ in range(32)]

    assert draws_a == draws_b
    assert all(1 <= block_size <= 4 for block_size, _ in draws_a)
    assert all(1 <= history_blocks <= 32 for _, history_blocks in draws_a)


@pytest.mark.parametrize(
    ("kwargs", "invalid_field"),
    [
        ({"block_size_min": 0}, "block_size_min"),
        ({"block_size_min": 4, "block_size_max": 3}, "block_size"),
        ({"history_blocks_min": 0}, "history_blocks_min"),
        ({"history_blocks_min": 32, "history_blocks_max": 31}, "history_blocks"),
    ],
)
def test_sample_teacher_forcing_parameters_rejects_invalid_ranges(
    kwargs: dict[str, int],
    invalid_field: str,
):
    with pytest.raises(ValueError, match=invalid_field):
        sample_teacher_forcing_parameters(**kwargs)


def test_build_teacher_forcing_layout_maps_both_streams_to_the_original_tokens():
    layout = build_teacher_forcing_layout(
        und_token_counts=[2],
        vision_token_shapes=[(5, 1, 1)],
        block_size=2,
        history_blocks=1,
    )

    assert layout.original_sample_lens == (7,)
    assert layout.sample_lens == (12,)
    assert layout.split_lens == (2, 10)
    assert layout.attn_modes == ("causal", "full")
    assert layout.source_sequence_indexes.tolist() == [0, 1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6]
    assert layout.sample_ids.tolist() == [0] * 12
    assert layout.stream_ids.tolist() == [-1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    assert layout.block_ids.tolist() == [-1, -1, 0, 0, 1, 1, 2, 0, 0, 1, 1, 2]
    assert layout.gen_query_indexes.tolist() == list(range(2, 12))
    assert layout.clean_token_indexes.tolist() == list(range(2, 7))
    assert layout.noisy_output_indexes.tolist() == list(range(7, 12))


def test_build_teacher_forcing_layout_expands_spatial_tokens_and_isolates_sample_offsets():
    layout = build_teacher_forcing_layout(
        und_token_counts=[1, 2],
        vision_token_shapes=[(3, 1, 2), (2, 2, 1)],
        block_size=2,
        history_blocks=3,
    )

    assert layout.original_sample_lens == (7, 6)
    assert layout.sample_lens == (13, 10)
    assert layout.split_lens == (1, 12, 2, 8)
    assert layout.attn_modes == ("causal", "full", "causal", "full")
    assert layout.source_sequence_indexes.tolist() == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        9,
        10,
        11,
        12,
    ]
    assert layout.block_ids.tolist() == [
        -1,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        -1,
        -1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    assert layout.sample_ids.tolist() == [0] * 13 + [1] * 10


@pytest.mark.parametrize(
    ("kwargs", "invalid_field"),
    [
        (
            {
                "und_token_counts": [1],
                "vision_token_shapes": [(2, 1, 1), (2, 1, 1)],
                "block_size": 1,
                "history_blocks": 1,
            },
            "same number",
        ),
        (
            {
                "und_token_counts": [],
                "vision_token_shapes": [],
                "block_size": 1,
                "history_blocks": 1,
            },
            "empty",
        ),
        (
            {
                "und_token_counts": [0],
                "vision_token_shapes": [(2, 1, 1)],
                "block_size": 1,
                "history_blocks": 1,
            },
            "und_token_counts",
        ),
        (
            {
                "und_token_counts": [1],
                "vision_token_shapes": [(2, 0, 1)],
                "block_size": 1,
                "history_blocks": 1,
            },
            "vision_token_shapes",
        ),
        (
            {
                "und_token_counts": [1],
                "vision_token_shapes": [(2, 1, 1)],
                "block_size": 0,
                "history_blocks": 1,
            },
            "block_size",
        ),
        (
            {
                "und_token_counts": [1],
                "vision_token_shapes": [(2, 1, 1)],
                "block_size": 1,
                "history_blocks": 0,
            },
            "history_blocks",
        ),
    ],
)
def test_build_teacher_forcing_layout_rejects_invalid_geometry(
    kwargs: dict[str, object],
    invalid_field: str,
):
    with pytest.raises(ValueError, match=invalid_field):
        build_teacher_forcing_layout(**kwargs)


def test_dense_mask_matches_s1_k1_block_causal_matrix():
    layout = build_teacher_forcing_layout(
        und_token_counts=[1],
        vision_token_shapes=[(3, 1, 1)],
        block_size=1,
        history_blocks=1,
    )

    mask = build_dense_teacher_forcing_gen_mask(layout, max_mask_elements=42)

    expected = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0, 0],  # C0 -> U,C0
            [1, 1, 1, 0, 0, 0, 0],  # C1 -> U,C0,C1
            [1, 0, 1, 1, 0, 0, 0],  # C2 -> U,C1,C2
            [1, 0, 0, 0, 1, 0, 0],  # N0 -> U,N0
            [1, 1, 0, 0, 0, 1, 0],  # N1 -> U,C0,N1
            [1, 0, 1, 0, 0, 0, 1],  # N2 -> U,C1,N2
        ],
        dtype=torch.bool,
    )
    torch.testing.assert_close(mask, expected)
    assert not mask[3, 1]
    assert not mask[4, 2]
    assert not mask[5, 3]


def test_dense_mask_keeps_blocks_full_and_limits_clean_history_to_k_blocks():
    layout = build_teacher_forcing_layout(
        und_token_counts=[1],
        vision_token_shapes=[(5, 1, 2)],
        block_size=2,
        history_blocks=1,
    )

    mask = build_dense_teacher_forcing_gen_mask(layout, max_mask_elements=420)

    assert mask[4].nonzero(as_tuple=True)[0].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert mask[8].nonzero(as_tuple=True)[0].tolist() == [0, 5, 6, 7, 8, 9, 10]
    assert mask[18].nonzero(as_tuple=True)[0].tolist() == [0, 5, 6, 7, 8, 19, 20]


def test_dense_mask_isolates_packed_samples():
    layout = build_teacher_forcing_layout(
        und_token_counts=[1, 2],
        vision_token_shapes=[(2, 1, 1), (2, 1, 1)],
        block_size=1,
        history_blocks=2,
    )

    mask = build_dense_teacher_forcing_gen_mask(layout, max_mask_elements=layout.gen_query_indexes.numel() * 11)
    query_sample_ids = layout.sample_ids[layout.gen_query_indexes]

    assert not mask[query_sample_ids == 0][:, layout.sample_ids == 1].any()
    assert not mask[query_sample_ids == 1][:, layout.sample_ids == 0].any()


def test_dense_mask_rejects_allocations_over_the_configured_limit():
    layout = build_teacher_forcing_layout(
        und_token_counts=[1],
        vision_token_shapes=[(3, 1, 1)],
        block_size=1,
        history_blocks=1,
    )
    num_elements = layout.gen_query_indexes.numel() * layout.source_sequence_indexes.numel()

    with pytest.raises(ValueError, match="max_mask_elements"):
        build_dense_teacher_forcing_gen_mask(layout, max_mask_elements=num_elements - 1)
    with pytest.raises(ValueError, match="max_mask_elements"):
        build_dense_teacher_forcing_gen_mask(layout, max_mask_elements=0)


def test_dense_mask_rejects_und_queries_in_gen_query_indexes():
    layout = build_teacher_forcing_layout(
        und_token_counts=[1],
        vision_token_shapes=[(1, 1, 1)],
        block_size=1,
        history_blocks=1,
    )
    corrupted = replace(layout, gen_query_indexes=torch.tensor([0], dtype=torch.long))

    with pytest.raises(ValueError, match="GEN queries"):
        build_dense_teacher_forcing_gen_mask(corrupted, max_mask_elements=3)
