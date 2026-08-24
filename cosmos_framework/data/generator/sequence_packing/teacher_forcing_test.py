# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing.modality import ModalityData, ModalitySpan
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence
from cosmos_framework.data.generator.sequence_packing.teacher_forcing import (
    TeacherForcingData,
    TeacherForcingGeometry,
    TeacherForcingLayout,
    TeacherForcingStream,
    build_dense_teacher_forcing_gen_mask,
    build_teacher_forcing_frame_block_ids,
    build_teacher_forcing_layout,
    expand_packed_sequence_for_teacher_forcing,
    sample_teacher_forcing_parameters,
    select_teacher_forcing_noisy_outputs,
)


def _geometry(block_size: int, history_blocks: int, num_samples: int = 1) -> TeacherForcingGeometry:
    return TeacherForcingGeometry(
        block_sizes=(block_size,) * num_samples,
        history_blocks=(history_blocks,) * num_samples,
    )


def test_teacher_forcing_stream_values_are_stable():
    assert int(TeacherForcingStream.UND) == -1
    assert int(TeacherForcingStream.CLEAN) == 0
    assert int(TeacherForcingStream.NOISY) == 1


def test_teacher_forcing_layout_is_frozen():
    empty = torch.empty(0, dtype=torch.long)
    layout = TeacherForcingLayout(
        geometry=TeacherForcingGeometry(block_sizes=(1,), history_blocks=(1,)),
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
        layout.geometry = TeacherForcingGeometry(block_sizes=(2,), history_blocks=(1,))


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"block_sizes": (), "history_blocks": ()}, "empty"),
        ({"block_sizes": (1, 2), "history_blocks": (1,)}, "one block_size"),
        ({"block_sizes": (0,), "history_blocks": (1,)}, "block_size"),
        ({"block_sizes": (1,), "history_blocks": (0,)}, "history_blocks"),
    ],
)
def test_teacher_forcing_geometry_rejects_invalid_values(kwargs: dict[str, tuple[int, ...]], error: str):
    with pytest.raises(ValueError, match=error):
        TeacherForcingGeometry(**kwargs)


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
        geometry=_geometry(2, 1),
    )

    assert layout.original_sample_lens == (7,)
    assert layout.sample_lens == (12,)
    assert layout.split_lens == (2, 10)
    assert layout.attn_modes == ("causal", "full")
    assert layout.source_sequence_indexes.tolist() == [0, 1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6]
    assert layout.sample_ids.tolist() == [0] * 12
    assert layout.stream_ids.tolist() == [-1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    assert layout.block_ids.tolist() == [-1, -1, 0, 1, 1, 2, 2, 0, 1, 1, 2, 2]
    assert layout.gen_query_indexes.tolist() == list(range(2, 12))
    assert layout.clean_token_indexes.tolist() == list(range(2, 7))
    assert layout.noisy_output_indexes.tolist() == list(range(7, 12))


def test_build_teacher_forcing_layout_expands_spatial_tokens_and_isolates_sample_offsets():
    layout = build_teacher_forcing_layout(
        und_token_counts=[1, 2],
        vision_token_shapes=[(3, 1, 2), (2, 2, 1)],
        geometry=_geometry(2, 3, num_samples=2),
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
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        1,
        -1,
        -1,
        0,
        0,
        1,
        1,
        0,
        0,
        1,
        1,
    ]
    assert layout.sample_ids.tolist() == [0] * 13 + [1] * 10


@pytest.mark.parametrize(
    ("kwargs", "invalid_field"),
    [
        (
            {
                "und_token_counts": [1],
                "vision_token_shapes": [(2, 1, 1), (2, 1, 1)],
                "geometry": _geometry(1, 1),
            },
            "same number",
        ),
        (
            {
                "und_token_counts": [],
                "vision_token_shapes": [],
                "geometry": _geometry(1, 1),
            },
            "empty",
        ),
        (
            {
                "und_token_counts": [0],
                "vision_token_shapes": [(2, 1, 1)],
                "geometry": _geometry(1, 1),
            },
            "und_token_counts",
        ),
        (
            {
                "und_token_counts": [1],
                "vision_token_shapes": [(2, 0, 1)],
                "geometry": _geometry(1, 1),
            },
            "vision_token_shapes",
        ),
        (
            {
                "und_token_counts": [1],
                "vision_token_shapes": [(2, 1, 1)],
                "geometry": _geometry(1, 1, num_samples=2),
            },
            "one entry",
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
        geometry=_geometry(1, 1),
    )

    mask = build_dense_teacher_forcing_gen_mask(
        layout, max_sequence_length=layout.source_sequence_indexes.numel()
    )

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


def test_teacher_forcing_frame_block_ids_keep_first_latent_in_a_singleton_block():
    assert build_teacher_forcing_frame_block_ids(1, 4).tolist() == [0]
    assert build_teacher_forcing_frame_block_ids(8, 3).tolist() == [0, 1, 1, 1, 2, 2, 2, 3]


def test_dense_mask_keeps_blocks_full_and_limits_clean_history_to_k_blocks():
    layout = build_teacher_forcing_layout(
        und_token_counts=[1],
        vision_token_shapes=[(5, 1, 2)],
        geometry=_geometry(2, 1),
    )

    mask = build_dense_teacher_forcing_gen_mask(
        layout, max_sequence_length=layout.source_sequence_indexes.numel()
    )

    clean_columns = layout.clean_token_indexes
    noisy_columns = layout.noisy_output_indexes
    frame_block_ids = build_teacher_forcing_frame_block_ids(5, 2).repeat_interleave(2)

    # The first latent is a singleton block; the following two latent pairs form ordinary blocks.
    assert frame_block_ids.tolist() == [0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    for token_id, block_id in enumerate(frame_block_ids.tolist()):
        clean_row = token_id
        noisy_row = frame_block_ids.numel() + token_id
        expected_clean_for_clean = (frame_block_ids >= block_id - 1) & (frame_block_ids <= block_id)
        expected_clean_for_noisy = (frame_block_ids >= block_id - 1) & (frame_block_ids < block_id)
        expected_noisy_for_noisy = frame_block_ids == block_id
        assert mask[clean_row, clean_columns].tolist() == expected_clean_for_clean.tolist()
        assert mask[noisy_row, clean_columns].tolist() == expected_clean_for_noisy.tolist()
        assert mask[noisy_row, noisy_columns].tolist() == expected_noisy_for_noisy.tolist()


@pytest.mark.parametrize("block_size", [1, 2, 3, 4])
@pytest.mark.parametrize("history_blocks", [1, 32])
def test_dense_mask_matches_singleton_first_latent_boundaries(block_size: int, history_blocks: int):
    num_frames = 7
    layout = build_teacher_forcing_layout(
        und_token_counts=[1],
        vision_token_shapes=[(num_frames, 1, 1)],
        geometry=_geometry(block_size, history_blocks),
    )
    mask = build_dense_teacher_forcing_gen_mask(
        layout,
        max_sequence_length=layout.source_sequence_indexes.numel(),
    )

    clean_columns = layout.clean_token_indexes
    noisy_columns = layout.noisy_output_indexes
    for frame_id in range(num_frames):
        frame_block_ids = build_teacher_forcing_frame_block_ids(num_frames, block_size)
        block_id = int(frame_block_ids[frame_id])
        oldest_visible_block = max(0, block_id - history_blocks)
        expected_clean_for_clean = [
            oldest_visible_block <= int(frame_block_ids[candidate]) <= block_id for candidate in range(num_frames)
        ]
        expected_clean_for_noisy = [
            oldest_visible_block <= int(frame_block_ids[candidate]) < block_id for candidate in range(num_frames)
        ]
        expected_noisy_for_noisy = [int(frame_block_ids[candidate]) == block_id for candidate in range(num_frames)]

        clean_row = frame_id
        noisy_row = num_frames + frame_id
        assert mask[clean_row, clean_columns].tolist() == expected_clean_for_clean
        assert not mask[clean_row, noisy_columns].any()
        assert mask[noisy_row, clean_columns].tolist() == expected_clean_for_noisy
        assert mask[noisy_row, noisy_columns].tolist() == expected_noisy_for_noisy


def test_dense_mask_isolates_packed_samples():
    layout = build_teacher_forcing_layout(
        und_token_counts=[1, 2],
        vision_token_shapes=[(2, 1, 1), (2, 1, 1)],
        geometry=_geometry(1, 2, num_samples=2),
    )

    mask = build_dense_teacher_forcing_gen_mask(
        layout, max_sequence_length=layout.source_sequence_indexes.numel()
    )
    query_sample_ids = layout.sample_ids[layout.gen_query_indexes]

    assert not mask[query_sample_ids == 0][:, layout.sample_ids == 1].any()
    assert not mask[query_sample_ids == 1][:, layout.sample_ids == 0].any()


def test_dense_mask_rejects_sequences_over_the_configured_limit():
    layout = build_teacher_forcing_layout(
        und_token_counts=[1],
        vision_token_shapes=[(3, 1, 1)],
        geometry=_geometry(1, 1),
    )
    sequence_length = layout.source_sequence_indexes.numel()

    with pytest.raises(ValueError, match="max_sequence_length"):
        build_dense_teacher_forcing_gen_mask(layout, max_sequence_length=sequence_length - 1)
    with pytest.raises(ValueError, match="max_sequence_length"):
        build_dense_teacher_forcing_gen_mask(layout, max_sequence_length=0)


def test_dense_mask_rejects_und_queries_in_gen_query_indexes():
    layout = build_teacher_forcing_layout(
        und_token_counts=[1],
        vision_token_shapes=[(1, 1, 1)],
        geometry=_geometry(1, 1),
    )
    corrupted = replace(layout, gen_query_indexes=torch.tensor([0], dtype=torch.long))

    with pytest.raises(ValueError, match="GEN queries"):
        build_dense_teacher_forcing_gen_mask(corrupted, max_sequence_length=3)


def test_teacher_forcing_api_is_exported():
    from cosmos_framework.data.generator import sequence_packing

    assert sequence_packing.TeacherForcingData is TeacherForcingData
    assert sequence_packing.TeacherForcingLayout is TeacherForcingLayout
    assert sequence_packing.build_teacher_forcing_frame_block_ids is build_teacher_forcing_frame_block_ids
    assert sequence_packing.build_teacher_forcing_layout is build_teacher_forcing_layout
    assert sequence_packing.build_dense_teacher_forcing_gen_mask is build_dense_teacher_forcing_gen_mask
    assert sequence_packing.expand_packed_sequence_for_teacher_forcing is expand_packed_sequence_for_teacher_forcing
    assert sequence_packing.sample_teacher_forcing_parameters is sample_teacher_forcing_parameters
    assert sequence_packing.select_teacher_forcing_noisy_outputs is select_teacher_forcing_noisy_outputs


def _make_packed_video_sequence() -> PackedSequence:
    noisy_0 = torch.tensor([[[[[10.0]], [[11.0]], [[12.0]]]]])
    noisy_1 = torch.tensor([[[[[20.0, 21.0]], [[22.0, 23.0]]]]])
    vision = ModalityData(
        sequence_indexes=torch.tensor([2, 3, 4, 6, 7, 8, 9]),
        timesteps=torch.tensor([0.2, 0.4, 0.4, 0.7, 0.7, 0.7, 0.7]),
        mse_loss_indexes=torch.tensor([2, 3, 4, 6, 7, 8, 9]),
        spans=[
            ModalitySpan(2, 1, 0, 0, 1, (1, 1, 1)),
            ModalitySpan(3, 1, 0, 1, 1, (1, 1, 1)),
            ModalitySpan(4, 1, 0, 2, 1, (1, 1, 1)),
            ModalitySpan(6, 2, 1, 0, 2, (1, 1, 2)),
            ModalitySpan(8, 2, 1, 2, 2, (1, 1, 2)),
        ],
        token_shapes=[(3, 1, 1), (2, 1, 2)],
        tokens=[noisy_0, noisy_1],
        condition_mask=[torch.zeros(3, 1, 1), torch.zeros(2, 1, 1)],
        noisy_frame_indexes=[torch.tensor([0, 1, 2]), torch.tensor([0, 1])],
    )
    return PackedSequence(
        sample_lens=[5, 5],
        split_lens=[2, 3, 1, 4],
        attn_modes=["causal", "full", "causal", "full"],
        is_image_batch=False,
        uses_single_timestep=True,
        sequence_length=10,
        text_ids=torch.tensor([101, 102, 103]),
        text_indexes=torch.tensor([0, 1, 5]),
        position_ids=torch.arange(30).reshape(3, 10),
        label_ids=torch.tensor([102, 103]),
        ce_loss_indexes=torch.tensor([0, 5]),
        ce_loss_weights=torch.tensor([1.0, 0.5]),
        vision=vision,
    )


def test_packed_sequence_has_no_teacher_forcing_data_by_default():
    assert PackedSequence().teacher_forcing is None


def test_expand_packed_sequence_preserves_noisy_contract_and_duplicates_rope():
    packed = _make_packed_video_sequence()
    clean_tokens = [torch.full_like(packed.vision.tokens[0], 1.0), torch.full_like(packed.vision.tokens[1], 2.0)]

    expanded = expand_packed_sequence_for_teacher_forcing(
        packed,
        clean_vision_tokens=clean_tokens,
        geometry=_geometry(2, 3, num_samples=2),
    )

    assert expanded is not packed
    assert expanded.vision is not packed.vision
    assert expanded.teacher_forcing is not None
    layout = expanded.teacher_forcing.layout
    assert isinstance(expanded.teacher_forcing, TeacherForcingData)
    assert expanded.sample_lens == [8, 9]
    assert expanded.split_lens == [2, 6, 1, 8]
    assert expanded.attn_modes == ["causal", "full", "causal", "full"]
    assert expanded.sequence_length == 17
    assert expanded.uses_single_timestep is False
    assert expanded.text_ids is packed.text_ids
    assert expanded.text_indexes.tolist() == [0, 1, 8]
    assert expanded.ce_loss_indexes.tolist() == [0, 8]
    assert expanded.position_ids.tolist() == packed.position_ids[:, layout.source_sequence_indexes].tolist()
    assert torch.equal(
        expanded.position_ids[:, layout.clean_token_indexes],
        expanded.position_ids[:, layout.noisy_output_indexes],
    )

    assert layout.clean_token_indexes.tolist() == [2, 3, 4, 9, 10, 11, 12]
    assert layout.noisy_output_indexes.tolist() == [5, 6, 7, 13, 14, 15, 16]
    assert expanded.vision.sequence_indexes.tolist() == [5, 6, 7, 13, 14, 15, 16]
    assert expanded.vision.mse_loss_indexes.tolist() == [5, 6, 7, 13, 14, 15, 16]
    assert expanded.vision.timesteps is packed.vision.timesteps
    assert expanded.vision.tokens == packed.vision.tokens
    assert expanded.vision.condition_mask == packed.vision.condition_mask
    assert expanded.vision.noisy_frame_indexes == packed.vision.noisy_frame_indexes
    assert [span.sequence_start for span in expanded.vision.spans] == [5, 6, 7, 13, 15]
    assert expanded.teacher_forcing.clean_vision_tokens == clean_tokens

    assert packed.sample_lens == [5, 5]
    assert packed.vision.sequence_indexes.tolist() == [2, 3, 4, 6, 7, 8, 9]
    assert packed.teacher_forcing is None


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda packed: setattr(packed, "vision", None), "vision"),
        (
            lambda packed: setattr(
                packed,
                "action",
                ModalityData(sequence_indexes=torch.tensor([0])),
            ),
            "action",
        ),
        (lambda packed: setattr(packed, "is_image_batch", True), "video"),
        (lambda packed: packed.text_indexes.__setitem__(0, 9), "partition"),
        (lambda packed: packed.sample_lens.__setitem__(0, 6), "sequence_length"),
        (lambda packed: packed.split_lens.__setitem__(0, 1), "attention splits"),
        (lambda packed: packed.attn_modes.__setitem__(1, "causal"), "attention splits"),
    ],
)
def test_expand_packed_sequence_rejects_unsupported_layouts(mutate, error: str):
    packed = _make_packed_video_sequence()
    mutate(packed)
    clean_tokens = [torch.zeros(1, 1, 3, 1, 1), torch.zeros(1, 1, 2, 1, 2)]

    with pytest.raises((TypeError, ValueError), match=error):
        expand_packed_sequence_for_teacher_forcing(
            packed,
            clean_vision_tokens=clean_tokens,
            geometry=_geometry(1, 1, num_samples=2),
        )


def test_expand_packed_sequence_rejects_clean_payload_shape_mismatch():
    packed = _make_packed_video_sequence()
    clean_tokens = [torch.zeros(1, 1, 2, 1, 1), torch.zeros(1, 1, 2, 1, 2)]

    with pytest.raises(ValueError, match="shape"):
        expand_packed_sequence_for_teacher_forcing(
            packed,
            clean_vision_tokens=clean_tokens,
            geometry=_geometry(1, 1, num_samples=2),
        )


def test_expand_packed_sequence_rejects_clean_payload_dtype_mismatch():
    packed = _make_packed_video_sequence()
    assert packed.vision is not None
    packed.vision.tokens = [token.to(torch.bfloat16) for token in packed.vision.tokens]
    clean_tokens = [torch.ones_like(token, dtype=torch.float32) for token in packed.vision.tokens]

    with pytest.raises(ValueError, match="dtype"):
        expand_packed_sequence_for_teacher_forcing(
            packed,
            clean_vision_tokens=clean_tokens,
            geometry=_geometry(1, 1, num_samples=2),
        )


@pytest.mark.parametrize(
    "condition_mask",
    [
        torch.tensor([[[1.0]], [[0.0]], [[0.0]]]),
        torch.tensor([[[1.0]], [[1.0]], [[0.0]]]),
        torch.tensor([[[0.0]], [[1.0]], [[0.0]]]),
    ],
)
def test_expand_packed_sequence_rejects_non_t2v_conditioning(condition_mask: torch.Tensor):
    packed = _make_packed_video_sequence()
    assert packed.vision is not None
    packed.vision.condition_mask[0] = condition_mask
    clean_tokens = [torch.zeros_like(token) for token in packed.vision.tokens]

    with pytest.raises(ValueError, match="only T2V.*I2V"):
        expand_packed_sequence_for_teacher_forcing(
            packed,
            clean_vision_tokens=clean_tokens,
            geometry=_geometry(1, 1, num_samples=2),
        )


def test_select_teacher_forcing_noisy_outputs_preserves_order_and_gradient():
    packed = _make_packed_video_sequence()
    clean_tokens = [torch.zeros_like(token) for token in packed.vision.tokens]
    expanded = expand_packed_sequence_for_teacher_forcing(
        packed,
        clean_vision_tokens=clean_tokens,
        geometry=_geometry(1, 1, num_samples=2),
    )
    assert expanded.teacher_forcing is not None
    output = torch.arange(expanded.sequence_length * 2, dtype=torch.float32).reshape(expanded.sequence_length, 2)
    output.requires_grad_()

    noisy = select_teacher_forcing_noisy_outputs(output, expanded.teacher_forcing.layout)

    torch.testing.assert_close(noisy, output[expanded.teacher_forcing.layout.noisy_output_indexes])
    noisy.sum().backward()
    assert output.grad is not None
    selected = torch.zeros(expanded.sequence_length, dtype=torch.bool)
    selected[expanded.teacher_forcing.layout.noisy_output_indexes] = True
    assert torch.equal(output.grad[selected], torch.ones_like(output.grad[selected]))
    assert torch.equal(output.grad[~selected], torch.zeros_like(output.grad[~selected]))
