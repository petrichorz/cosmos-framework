# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from cosmos_framework.data.generator.local_datasets.sft_dataset_lerobot3 import (
    LeRobotSFTDataset,
    _LeRobotVideoDecoderCache,
    _limit_temporal_interval_by_fps,
    _load_single_lerobot_metadata,
)


def test_rejects_unknown_video_backend():
    dataset = object.__new__(LeRobotSFTDataset)
    with pytest.raises(ValueError, match="Unsupported video_backend='opencv'"):
        dataset.__init__(video_backend="opencv")


def test_rejects_unknown_video_resize_mode():
    dataset = object.__new__(LeRobotSFTDataset)
    with pytest.raises(ValueError, match="Unsupported video_resize_mode='decoder'"):
        dataset.__init__(video_resize_mode="decoder")


def test_lerobot_metadata_duration_and_frame_filters_are_configurable(tmp_path, monkeypatch):
    video_key = "observation.images.top"
    info = {
        "fps": 10,
        "features": {video_key: {"dtype": "video", "shape": [8, 8, 3]}},
        "video_path": "videos/{video_key}/chunk-{chunk_index}/file-{file_index}.mp4",
    }
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text(json.dumps(info))
    episodes_dir = tmp_path / "meta" / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True)
    (episodes_dir / "file-000.parquet").touch()

    rows = [
        {
            "episode_index": 0,
            f"videos/{video_key}/from_timestamp": 0.0,
            f"videos/{video_key}/to_timestamp": 6.1,
            "tasks": ["exactly 61 frames"],
        },
        {
            "episode_index": 1,
            f"videos/{video_key}/from_timestamp": 6.1,
            f"videos/{video_key}/to_timestamp": 12.0,
            "tasks": ["fewer than 61 frames"],
        },
        {
            "episode_index": 2,
            f"videos/{video_key}/from_timestamp": 12.0,
            f"videos/{video_key}/to_timestamp": 74.0,
            "tasks": ["longer than 61 seconds"],
        },
    ]
    monkeypatch.setattr(pd, "read_parquet", lambda _: pd.DataFrame(rows))

    default_filtered = _load_single_lerobot_metadata(
        str(tmp_path),
        min_frames=61,
        max_video_duration_s=61.0,
        min_short_edge=0,
        video_feature_key=video_key,
        caption_key="caption",
    )
    assert [item["uuid"] for item in default_filtered] == [f"{tmp_path.name}_chunk_0_file_0_episode_0"]

    custom_filtered = _load_single_lerobot_metadata(
        str(tmp_path),
        min_frames=50,
        max_video_duration_s=0,
        min_short_edge=0,
        video_feature_key=video_key,
        caption_key="caption",
    )
    assert len(custom_filtered) == 3


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


def _native_fps_dataset() -> LeRobotSFTDataset:
    dataset = object.__new__(LeRobotSFTDataset)
    dataset.output_sizes = {"1,1": (8, 8)}
    dataset.num_video_frames = -1
    dataset.max_video_fps = 30.0
    dataset.temporal_compression_factor = 4
    dataset.conditioning_fps = -1
    dataset.conditioning_fps_noise_std = 0.0
    dataset.caption_suffix = ""
    dataset.cfg_dropout_keep_metadata = False
    dataset.cfg_dropout_rate = 0.0
    dataset.append_duration_fps_timestamps = False
    dataset.append_resolution_info = False
    dataset.conditioning_config = None
    dataset._decoder_cache = None
    dataset._tokenize_caption = lambda caption: ([1], caption)
    return dataset


def test_native_chunk_applies_max_video_fps(monkeypatch):
    dataset = _native_fps_dataset()
    calls = []

    monkeypatch.setattr(
        "cosmos_framework.data.generator.local_datasets.sft_dataset_lerobot3.get_video_metadata",
        lambda path: {"fps": 60.0, "total_frames": 61},
    )

    def fake_decode(**kwargs):
        calls.append(kwargs)
        return [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(31)]

    dataset._decode_video_frames = fake_decode
    metadata = {
        "uuid": "episode-1",
        "vision_path": "video.mp4",
        "width": 8,
        "height": 8,
        "aspect_ratio": "1,1",
        "t2w_windows": [{"start_frame": 0, "end_frame": 60, "temporal_interval": 1, "caption": "task"}],
    }

    sample = dataset.process_one_sample(metadata)

    assert calls[0]["temporal_interval"] == 2
    assert sample["conditioning_fps"] == 30.0
    assert sample["num_multiplier"] == 2
    assert sample["num_frames"] == 29


def test_fixed_frame_request_is_skipped_when_fps_cap_makes_window_too_short(monkeypatch):
    dataset = _native_fps_dataset()
    dataset.num_video_frames = 40
    dataset.temporal_interval_mode = "force_one"

    monkeypatch.setattr(
        "cosmos_framework.data.generator.local_datasets.sft_dataset_lerobot3.get_video_metadata",
        lambda path: {"fps": 60.0, "total_frames": 61},
    )
    dataset._decode_video_frames = lambda **kwargs: pytest.fail("decode should not be called")

    metadata = _minimal_metadata() | {
        "vision_path": "video.mp4",
        "t2w_windows": [{"start_frame": 0, "end_frame": 60, "temporal_interval": 1, "caption": "task"}],
    }
    assert dataset.process_one_sample(metadata) is None


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


def _minimal_metadata() -> dict:
    return {
        "uuid": "episode-1",
        "vision_path": "broken.mp4",
        "width": 16,
        "height": 16,
        "aspect_ratio": "1,1",
        "t2w_windows": [{"start_frame": 0, "end_frame": 4, "temporal_interval": 1}],
    }


def test_video_metadata_error_is_logged_and_sample_is_skipped(monkeypatch):
    dataset = object.__new__(LeRobotSFTDataset)
    dataset.output_sizes = {"1,1": (8, 8)}
    logged = []

    def fail_metadata(video_path):
        raise OSError("ffprobe failed")

    monkeypatch.setattr(
        "cosmos_framework.data.generator.local_datasets.sft_dataset_lerobot3.get_video_metadata",
        fail_metadata,
    )
    monkeypatch.setattr(
        "cosmos_framework.data.generator.local_datasets.sft_dataset_lerobot3.log.exception",
        lambda message, rank0_only: logged.append((message, rank0_only)),
    )

    assert dataset.process_one_sample(_minimal_metadata()) is None
    assert "ffprobe failed" in logged[0][0]
    assert "advancing to the next video" in logged[0][0]
    assert logged[0][1] is False


def test_video_decode_error_discards_cache_and_skips_sample(monkeypatch):
    dataset = object.__new__(LeRobotSFTDataset)
    dataset.output_sizes = {"1,1": (8, 8)}
    dataset.num_video_frames = -1
    dataset.max_video_fps = 30.0
    dataset.video_backend = "torchcodec"
    dataset.video_resize_mode = "decode_transform"
    discarded = []
    logged = []
    dataset._decoder_cache = type("FakeCache", (), {"discard": lambda self, path: discarded.append(path)})()

    monkeypatch.setattr(
        "cosmos_framework.data.generator.local_datasets.sft_dataset_lerobot3.get_video_metadata",
        lambda path: {"fps": 20.0, "total_frames": 100},
    )
    monkeypatch.setattr(
        dataset,
        "_decode_video_frames",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("decoder failed")),
    )
    monkeypatch.setattr(
        "cosmos_framework.data.generator.local_datasets.sft_dataset_lerobot3.log.exception",
        lambda message, rank0_only: logged.append((message, rank0_only)),
    )

    assert dataset.process_one_sample(_minimal_metadata()) is None
    assert discarded == ["broken.mp4"]
    assert "decoder failed" in logged[0][0]
    assert "backend=torchcodec" in logged[0][0]
    assert logged[0][1] is False


def test_decoder_cache_discard_closes_all_resize_variants():
    class FakeHandle:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    cache = _LeRobotVideoDecoderCache(max_size=4)
    matching_handles = [FakeHandle(), FakeHandle()]
    retained_handle = FakeHandle()
    cache._cache[("broken.mp4", None)] = (object(), matching_handles[0])
    cache._cache[("broken.mp4", (8, 8))] = (object(), matching_handles[1])
    cache._cache[("healthy.mp4", None)] = (object(), retained_handle)

    cache.discard("broken.mp4")

    assert all(handle.closed for handle in matching_handles)
    assert not retained_handle.closed
    assert list(cache._cache) == [("healthy.mp4", None)]
