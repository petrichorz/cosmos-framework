# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.local_datasets.sft_dataset_lerobot3 import (
    LeRobotSFTDataset,
    _build_balanced_video_windows,
    _limit_temporal_interval_by_fps,
)


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


def test_torchcodec_passes_temporal_interval_as_decoder_step():
    calls = []

    class FakeDecoder:
        def get_frames_in_range(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(data=torch.zeros((4, 3, 2, 2), dtype=torch.uint8))

    dataset = object.__new__(LeRobotSFTDataset)
    dataset._decoder_cache = SimpleNamespace(get_decoder=lambda video_path, resize_hw: FakeDecoder())

    frames = dataset._decode_video_frames_torchcodec(
        "video.mp4",
        start_frame=3,
        end_frame=12,
        temporal_interval=3,
    )

    assert calls == [{"start": 3, "stop": 13, "step": 3}]
    assert frames.shape[0] == 4
