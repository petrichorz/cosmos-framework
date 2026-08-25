# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from cosmos_framework.model.generator.causal_inference import (
    causal_generated_block_span,
    causal_total_latent_frames,
    concat_vision_time,
    flatten_vision_5d,
    slice_vision_time,
    unflatten_vision_5d,
)


@pytest.mark.parametrize(
    ("num_blocks", "block_size", "expected"),
    [(1, 1, 2), (16, 1, 17), (3, 4, 13)],
)
def test_causal_total_latent_frames(num_blocks: int, block_size: int, expected: int):
    assert causal_total_latent_frames(num_blocks, block_size) == expected


@pytest.mark.parametrize(
    ("block_id", "block_size", "expected"),
    [(1, 1, (1, 2)), (2, 1, (2, 3)), (1, 4, (1, 5)), (3, 4, (9, 13))],
)
def test_causal_generated_block_span(block_id: int, block_size: int, expected: tuple[int, int]):
    assert causal_generated_block_span(block_id, block_size) == expected


@pytest.mark.parametrize(("num_blocks", "block_size"), [(0, 1), (1, 0)])
def test_causal_total_latent_frames_rejects_invalid_geometry(num_blocks: int, block_size: int):
    with pytest.raises(ValueError):
        causal_total_latent_frames(num_blocks, block_size)


def test_vision_5d_helpers_only_slice_temporal_dimension():
    latent = torch.arange(1 * 2 * 5 * 3 * 4).reshape(1, 2, 5, 3, 4)
    first = slice_vision_time(latent, 0, 2)
    second = slice_vision_time(latent, 2, 5)
    assert first.shape == (1, 2, 2, 3, 4)
    assert second.shape == (1, 2, 3, 3, 4)
    torch.testing.assert_close(concat_vision_time([first, second]), latent)
    torch.testing.assert_close(unflatten_vision_5d(flatten_vision_5d(latent), latent.shape), latent)
