# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import pytest
import torch

from cosmos_framework.data.generator.local_datasets.sft_dataset_lerobot3 import LeRobotSFTDataset


def test_rejects_unknown_video_backend():
    dataset = object.__new__(LeRobotSFTDataset)
    with pytest.raises(ValueError, match="Unsupported video_backend='opencv'"):
        dataset.__init__(video_backend="opencv")


def test_pyav_backend_uses_lerobot_timestamp_decoder(monkeypatch):
    calls = []

    def fake_decode(video_path, timestamps, tolerance_s, backend):
        calls.append((video_path, timestamps, tolerance_s, backend))
        return torch.tensor([[[[0.0]], [[0.5]], [[1.0]]]]).repeat(len(timestamps), 1, 1, 1)

    monkeypatch.setattr("lerobot.datasets.video_utils.decode_video_frames", fake_decode)
    dataset = object.__new__(LeRobotSFTDataset)
    dataset.video_tolerance_s = 1e-4

    frames = dataset._decode_video_frames_pyav(
        "video.mp4",
        start_frame=10,
        end_frame=14,
        temporal_interval=2,
        original_fps=10.0,
    )

    assert calls == [("video.mp4", [1.0, 1.2, 1.4], 1e-4, "pyav")]
    assert frames.dtype == torch.uint8
    assert frames[:, :, 0, 0].tolist() == [[0, 128, 255]] * 3


def test_decode_dispatches_to_configured_backend_and_resizes(monkeypatch):
    dataset = object.__new__(LeRobotSFTDataset)
    dataset.video_backend = "pyav"
    source = torch.arange(3 * 2 * 4 * 6, dtype=torch.uint8).reshape(2, 3, 4, 6)
    calls = []

    def fake_pyav(video_path, start_frame, end_frame, temporal_interval, original_fps):
        calls.append((video_path, start_frame, end_frame, temporal_interval, original_fps))
        return source

    monkeypatch.setattr(dataset, "_decode_video_frames_pyav", fake_pyav)
    frames = dataset._decode_video_frames(
        "video.mp4",
        start_frame=3,
        end_frame=5,
        temporal_interval=2,
        original_fps=30.0,
        resize_h=8,
        resize_w=10,
    )

    assert calls == [("video.mp4", 3, 5, 2, 30.0)]
    assert len(frames) == 2
    assert frames[0].shape == (8, 10, 3)
    assert frames[0].dtype.name == "uint8"


def test_decode_dispatches_to_torchcodec_backend(monkeypatch):
    dataset = object.__new__(LeRobotSFTDataset)
    dataset.video_backend = "torchcodec"
    source = torch.zeros((2, 3, 4, 6), dtype=torch.uint8)
    calls = []

    def fake_torchcodec(video_path, start_frame, end_frame, temporal_interval):
        calls.append((video_path, start_frame, end_frame, temporal_interval))
        return source

    monkeypatch.setattr(dataset, "_decode_video_frames_torchcodec", fake_torchcodec)
    frames = dataset._decode_video_frames(
        "video.mp4",
        start_frame=3,
        end_frame=5,
        temporal_interval=2,
        original_fps=30.0,
        resize_h=4,
        resize_w=6,
    )

    assert calls == [("video.mp4", 3, 5, 2)]
    assert len(frames) == 2
    assert frames[0].shape == (4, 6, 3)
