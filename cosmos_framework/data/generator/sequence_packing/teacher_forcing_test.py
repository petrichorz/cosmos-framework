# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from dataclasses import FrozenInstanceError

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing.teacher_forcing import (
    TeacherForcingLayout,
    TeacherForcingStream,
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
