# SPDX-License-Identifier: OpenMDW-1.1
"""Index-only checks; no parquet, video, torch or model loading."""

import warnings

import pytest

from cosmos_framework.data.generator.action.causal_block_geometry import CausalBlockGeometry
from cosmos_framework.data.generator.action.segment_planner import SegmentPlanner

GEOMETRY = CausalBlockGeometry(temporal_compression_factor=4)


def planner(length=96, overlap=20, geometry=GEOMETRY):
    return SegmentPlanner(max_action_steps=length, overlap_action_steps=overlap, geometry=geometry)


def test_long_episode_right_aligns_tail_without_block_aligned_start():
    p = planner()
    assert p.plan(211) == [(0, 96), (76, 96), (114, 96)]
    assert p.discarded_action_steps == p.skipped_ranges == 0


@pytest.mark.parametrize("frames,starts", [(97, [0]), (173, [0, 76]), (249, [0, 76, 152]), (98, [0, 1])])
def test_exact_ends_are_not_duplicated_and_small_tail_is_retained(frames, starts):
    segments = planner().plan(frames)
    assert [start for start, _ in segments] == starts
    assert segments[-1][0] + segments[-1][1] + 1 == frames


@pytest.mark.parametrize("frames,kept,discarded", [(78, 65, 13), (65, 65, 0), (33, 33, 0), (96, 65, 31)])
def test_short_episode_keeps_beginning_and_truncates_end(frames, kept, discarded):
    p = planner()
    assert p.plan(frames, observation_start=13) == [(13, kept - 1)]
    assert p.discarded_action_steps == discarded
    assert p.skipped_ranges == 0


@pytest.mark.parametrize("block_actions", [16, 32])
def test_preserve_tail_covers_all_actions_without_duplicate_windows(block_actions):
    geometry = CausalBlockGeometry(actions_per_block=block_actions, temporal_compression_factor=4)
    for frames in range(block_actions + 1, 260):
        p = planner(length=block_actions * 3, overlap=7, geometry=geometry)
        segments = p.plan(frames, observation_start=11, preserve_tail=True)
        covered = set()
        for start, actions in segments:
            assert actions % block_actions == 0 and actions <= p.max_action_steps
            assert 11 <= start and start + actions < 11 + frames
            covered.update(range(start, start + actions))
        assert covered == set(range(11, 11 + frames - 1))
        assert len(segments) == len(set(segments))
        assert p.discarded_action_steps == p.skipped_ranges == 0


@pytest.mark.parametrize("frames", [0, 1, 2, 32])
def test_too_short_warns_with_source_and_range_then_skips(frames):
    p = planner()
    with pytest.warns(UserWarning, match="source='agibot/task', episode=7, segment='instruction-1'") as captured:
        assert (
            p.plan(frames, observation_start=9, source_id="agibot/task", episode_id=7, segment_id="instruction-1") == []
        )
    assert len(captured) == 1
    assert f"{frames} aligned frames available; at least 33 required" in str(captured[0].message)
    assert p.skipped_ranges == 1
    assert p.discarded_action_steps == max(0, frames - 1)


def test_counts_accumulate_while_ranges_remain_independent():
    p = planner()
    first = p.plan(211, observation_start=10)
    assert first[-1][0] + first[-1][1] + 1 == 221
    assert p.plan(78, observation_start=230) == [(230, 64)]
    with pytest.warns(UserWarning):
        assert p.plan(32, observation_start=500) == []
    assert p.skipped_ranges == 1
    assert p.discarded_action_steps == 13 + 31
    assert p.plan(97) == [(0, 96)]
    assert p.discarded_action_steps == 44  # overlap 不能增加丢弃计数。


@pytest.mark.parametrize("length,overlap", [(0, 0), (95, 20), (96, 96), (96, 97), (96, -1), (True, 0), (96, 1.5)])
def test_invalid_configuration_is_rejected(length, overlap):
    with pytest.raises(ValueError):
        planner(length, overlap)


@pytest.mark.parametrize("frames,start", [(-1, 0), (33, -1), (33, True), (33.0, 0)])
def test_invalid_range_is_rejected(frames, start):
    with pytest.raises(ValueError):
        planner().plan(frames, observation_start=start)


def test_alternative_geometry_controls_length_and_warning_threshold():
    geometry = CausalBlockGeometry(actions_per_block=12, video_stride=2, temporal_compression_factor=3)
    p = planner(24, 5, geometry)
    assert p.plan(50, observation_start=5) == [(5, 24), (24, 24), (30, 24)]
    assert p.plan(20) == [(0, 12)]
    with pytest.warns(UserWarning, match="at least 13 required"):
        assert p.plan(12) == []


def test_coverage_and_discard_counts_over_many_lengths_and_overlaps():
    # 验证区间并集，不复写规划器的起点公式。
    for overlap in (0, 1, 20, 95):
        for frames in range(260):
            p = planner(overlap=overlap)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                segments = p.plan(frames, observation_start=11)
            covered = set()
            for start, actions in segments:
                GEOMETRY.validate_observation_frames(actions + 1)
                assert 11 <= start < start + actions + 1 <= 11 + frames
                covered.update(range(start, start + actions))
            starts = [start for start, _ in segments]
            assert starts == sorted(set(starts))
            assert len(covered) + p.discarded_action_steps == max(0, frames - 1)
            if frames >= 97:
                assert covered == set(range(11, 11 + frames - 1))
            elif frames >= 33:
                assert covered == set(range(11, 11 + len(covered)))
