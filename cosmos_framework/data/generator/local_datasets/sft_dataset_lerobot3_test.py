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


def test_rejects_unknown_video_resize_mode():
    dataset = object.__new__(LeRobotSFTDataset)
    with pytest.raises(ValueError, match="Unsupported video_resize_mode='decoder'"):
        dataset.__init__(video_resize_mode="decoder")


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
        resize_hw=None,
    )

    assert calls == [("video.mp4", [1.0, 1.2, 1.4], 1e-4, "pyav")]
    assert frames.dtype == torch.uint8
    assert frames[:, :, 0, 0].tolist() == [[0, 128, 255]] * 3


def test_decode_dispatches_to_configured_backend_and_resizes(monkeypatch):
    dataset = object.__new__(LeRobotSFTDataset)
    dataset.video_backend = "pyav"
    dataset.video_resize_mode = "post_decode"
    source = torch.arange(3 * 2 * 4 * 6, dtype=torch.uint8).reshape(2, 3, 4, 6)
    calls = []

    def fake_pyav(video_path, start_frame, end_frame, temporal_interval, original_fps, resize_hw):
        calls.append((video_path, start_frame, end_frame, temporal_interval, original_fps, resize_hw))
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

    assert calls == [("video.mp4", 3, 5, 2, 30.0, None)]
    assert len(frames) == 2
    assert frames[0].shape == (8, 10, 3)
    assert frames[0].dtype.name == "uint8"


def test_decode_dispatches_to_torchcodec_backend(monkeypatch):
    dataset = object.__new__(LeRobotSFTDataset)
    dataset.video_backend = "torchcodec"
    dataset.video_resize_mode = "post_decode"
    source = torch.zeros((2, 3, 4, 6), dtype=torch.uint8)
    calls = []

    def fake_torchcodec(video_path, start_frame, end_frame, temporal_interval, resize_hw):
        calls.append((video_path, start_frame, end_frame, temporal_interval, resize_hw))
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

    assert calls == [("video.mp4", 3, 5, 2, None)]
    assert len(frames) == 2
    assert frames[0].shape == (4, 6, 3)


@pytest.mark.parametrize("backend", ["pyav", "torchcodec"])
def test_decode_transform_passes_target_size_to_backend_without_post_resize(monkeypatch, backend):
    dataset = object.__new__(LeRobotSFTDataset)
    dataset.video_backend = backend
    dataset.video_resize_mode = "decode_transform"
    source = torch.zeros((2, 3, 8, 10), dtype=torch.uint8)
    calls = []

    if backend == "torchcodec":

        def fake_decode(video_path, start_frame, end_frame, temporal_interval, resize_hw):
            calls.append(resize_hw)
            return source

        monkeypatch.setattr(dataset, "_decode_video_frames_torchcodec", fake_decode)
    else:

        def fake_decode(video_path, start_frame, end_frame, temporal_interval, original_fps, resize_hw):
            calls.append(resize_hw)
            return source

        monkeypatch.setattr(dataset, "_decode_video_frames_pyav", fake_decode)

    frames = dataset._decode_video_frames(
        "video.mp4",
        start_frame=3,
        end_frame=5,
        temporal_interval=2,
        original_fps=30.0,
        resize_h=8,
        resize_w=10,
    )

    assert calls == [(8, 10)]
    assert len(frames) == 2
    assert frames[0].shape == (8, 10, 3)
