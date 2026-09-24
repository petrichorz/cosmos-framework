# SPDX-License-Identifier: OpenMDW-1.1
"""Pure temporal geometry checks; no torch, VAE or dataset loading required."""

import pytest

from cosmos_framework.data.generator.action.causal_block_geometry import CausalBlockGeometry


def test_current_recipe_and_independent_first_frame():
    geometry = CausalBlockGeometry(temporal_compression_factor=4)
    assert geometry.actions_per_latent_frame == 16
    assert geometry.latent_frames_per_block == 2
    assert geometry.min_observation_frames == 33
    assert geometry.num_blocks(33) == 1
    assert geometry.num_video_frames(33) == 9
    assert geometry.num_latent_frames(33) == 3
    assert geometry.latent_block_index(0) == -1
    assert geometry.block_action_span(0) == (0, 32)
    assert geometry.block_latent_span(0) == (1, 3)


def test_multiple_blocks_cover_actions_and_latents_without_prefix_actions():
    geometry = CausalBlockGeometry(temporal_compression_factor=4)
    assert geometry.num_blocks(97) == 3
    assert geometry.num_video_frames(97) == 25
    assert geometry.num_latent_frames(97) == 7
    for block in range(3):
        action_start, action_stop = geometry.block_action_span(block)
        latent_start, latent_stop = geometry.block_latent_span(block)
        assert action_start == block * 32
        assert set(geometry.action_latent_index(i) for i in range(action_start, action_stop)) == set(
            range(latent_start, latent_stop)
        )
        assert {geometry.latent_block_index(i) for i in range(latent_start, latent_stop)} == {block}


@pytest.mark.parametrize("available,expected", [(0, 0), (1, 0), (32, 0), (33, 33), (65, 65), (78, 65), (97, 97)])
def test_max_complete_length(available, expected):
    geometry = CausalBlockGeometry(temporal_compression_factor=4)
    assert geometry.max_complete_observation_frames(available) == expected
    if expected:
        geometry.validate_observation_frames(expected)


@pytest.mark.parametrize("frames", [0, 1, 32, 34, 78, True, 33.0])
def test_reject_invalid_segment_length(frames):
    geometry = CausalBlockGeometry(temporal_compression_factor=4)
    with pytest.raises(ValueError):
        geometry.validate_observation_frames(frames)


@pytest.mark.parametrize(
    "options",
    [
        {"actions_per_block": 0},
        {"actions_per_block": 31},
        {"actions_per_block": True},
        {"actions_per_block": 32.0},
        {"video_stride": 0},
        {"video_stride": 3},
        {"temporal_compression_factor": 0},
        {"temporal_compression_factor": 3},
        {"temporal_compression_factor": False},
    ],
)
def test_reject_incompatible_configuration(options):
    with pytest.raises(ValueError):
        CausalBlockGeometry(**({"temporal_compression_factor": 4} | options))


def test_compression_must_be_supplied_by_caller():
    with pytest.raises(TypeError):
        CausalBlockGeometry()


def test_other_geometry_has_no_hidden_32_or_compression_4():
    geometry = CausalBlockGeometry(actions_per_block=12, video_stride=2, temporal_compression_factor=3)
    assert geometry.min_observation_frames == 13
    assert geometry.latent_frames_per_block == 2
    assert geometry.max_complete_observation_frames(30) == 25
    assert geometry.num_blocks(25) == 2
    assert geometry.num_video_frames(25) == 13
    assert geometry.num_latent_frames(25) == 5
    assert geometry.block_action_span(1) == (12, 24)
    assert geometry.block_latent_span(1) == (3, 5)
    assert geometry.action_latent_index(12) == 3


@pytest.mark.parametrize("invalid", [-1, True, 1.5])
def test_reject_invalid_indexes_and_available_lengths(invalid):
    geometry = CausalBlockGeometry(temporal_compression_factor=4)
    for method in (
        geometry.block_action_span,
        geometry.block_latent_span,
        geometry.action_latent_index,
        geometry.latent_block_index,
        geometry.max_complete_observation_frames,
    ):
        with pytest.raises(ValueError):
            method(invalid)


@pytest.mark.parametrize("aligned_fps", [30, 29.97, 10.0])
@pytest.mark.parametrize("stride,compression", [(4, 4), (1, 4), (8, 2)])
def test_subsampling_preserves_duration_and_cross_modal_time(aligned_fps, stride, compression):
    geometry = CausalBlockGeometry(video_stride=stride, temporal_compression_factor=compression)
    video_fps = geometry.video_fps(aligned_fps)
    action_fps = geometry.action_fps(aligned_fps)
    assert video_fps == pytest.approx(aligned_fps / stride)
    assert action_fps == aligned_fps
    # 排除独立首帧后，action、视频和 latent 表示相同的物理时长。
    duration = 96 / action_fps
    assert (geometry.num_video_frames(97) - 1) / video_fps == pytest.approx(duration)
    assert (geometry.num_latent_frames(97) - 1) * compression / video_fps == pytest.approx(duration)
    # 一个 block 终点同时对应 action 的终点与其最后一个 latent 的时间。
    _, action_stop = geometry.block_action_span(0)
    _, latent_stop = geometry.block_latent_span(0)
    assert action_stop / action_fps == pytest.approx((latent_stop - 1) * compression / video_fps)


@pytest.mark.parametrize("invalid", [0, -30, float("nan"), float("inf"), float("-inf"), True, "30", None])
def test_reject_invalid_fps(invalid):
    geometry = CausalBlockGeometry(temporal_compression_factor=4)
    for method in (geometry.video_fps, geometry.action_fps):
        with pytest.raises(ValueError, match="finite positive"):
            method(invalid)
