# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.local_datasets.sft_dataset_lerobot3 import (
    _build_balanced_video_windows,
    _limit_temporal_interval_by_fps,
    _select_resolution_tier,
    _validate_resolution_tiers,
)
from lerobot.datasets import video_utils as _vu


def test_balanced_video_windows_cover_source_with_exact_overlap():
    windows = _build_balanced_video_windows(
        start_frame=0,
        end_frame=2729,
        fps=30.0,
        max_video_duration_s=61.0,
        video_window_overlap_s=5.0,
    )

    assert windows == [(0, 1439), (1290, 2729)]
    assert windows[0][1] - windows[1][0] + 1 == 150


def test_balanced_video_windows_validate_overlap():
    with pytest.raises(ValueError, match="0 <= overlap < max_video_duration_s"):
        _build_balanced_video_windows(0, 100, 30.0, 5.0, 5.0)


@pytest.mark.parametrize(
    ("original_fps", "temporal_interval", "max_video_fps", "expected"),
    [
        (20.0, 1, 30.0, 1),
        (30.0, 1, 30.0, 1),
        (31.0, 1, 30.0, 2),
        (50.0, 1, 30.0, 2),
        (60.0, 1, 30.0, 2),
        (61.0, 1, 30.0, 3),
        (60.0, 4, 30.0, 4),
        (60.0, 1, 0.0, 1),
    ],
)
def test_limit_temporal_interval_by_fps(original_fps, temporal_interval, max_video_fps, expected):
    assert _limit_temporal_interval_by_fps(original_fps, temporal_interval, max_video_fps) == expected


def test_torchcodec_requests_expected_frame_indices(monkeypatch):
    calls = []

    class FakeDecoder:
        metadata = SimpleNamespace(average_fps=30.0)

        def get_frames_at(self, *, indices):
            calls.append(indices)
            return SimpleNamespace(
                data=torch.zeros((4, 3, 2, 2), dtype=torch.uint8),
                pts_seconds=torch.tensor([0.1, 0.2, 0.3, 0.4]),
            )

    monkeypatch.setattr(
        _vu,
        "_default_decoder_cache",
        SimpleNamespace(get_decoder=lambda video_path: FakeDecoder()),
    )

    frames = _vu.decode_video_frames(
        "video.mp4",
        timestamps=[0.1, 0.2, 0.3, 0.4],
        tolerance_s=1e-4,
        backend="torchcodec",
    )

    assert calls == [[3, 6, 9, 12]]
    assert frames.shape[0] == 4


def test_single_resolution_tier_is_fixed_even_for_smaller_input() -> None:
    assert _select_resolution_tier(("480",), video_min_edge=256) == "480"


def test_multiple_resolution_tiers_select_from_eligible_tiers() -> None:
    assert _select_resolution_tier(("256", "480"), video_min_edge=300) == "256"


def test_multiple_resolution_tiers_fall_back_to_smallest_configured_tier() -> None:
    assert _select_resolution_tier(("480", "720"), video_min_edge=256) == "480"


@pytest.mark.parametrize("tiers", [(), ("480", "480"), ("unknown",)])
def test_invalid_resolution_tiers_raise(tiers: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="resolution_tiers"):
        _validate_resolution_tiers(tiers)
