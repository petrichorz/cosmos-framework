# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from fractions import Fraction

import av
import numpy as np
import torch

from cosmos_framework.data.generator.local_datasets.sft_dataset_lerobot3 import LeRobotSFTDataset


class _FakeFrame:
    def __init__(self, frame_idx: int):
        self.pts = frame_idx
        self._frame_idx = frame_idx

    def reformat(self, *, width, height, format, interpolation):
        assert format == "rgb24"
        assert interpolation is av.video.reformatter.Interpolation.BICUBIC
        self._array = np.full((height, width, 3), self._frame_idx, dtype=np.uint8)
        return self

    def to_ndarray(self):
        return self._array


class _FakeStream:
    time_base = Fraction(1, 20)
    start_time = 0
    thread_count = None


class _FakeContainer:
    def __init__(self):
        self.stream = _FakeStream()
        self.streams = type("_Streams", (), {"video": [self.stream]})()
        self.seek_args = None
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True

    def seek(self, offset, *, stream, backward, any_frame):
        self.seek_args = (offset, stream, backward, any_frame)

    def decode(self, stream):
        assert stream is self.stream
        yield from (_FakeFrame(frame_idx) for frame_idx in range(4, 20))


def test_pyav_streaming_decode_selects_resizes_and_closes(monkeypatch):
    container = _FakeContainer()
    monkeypatch.setattr(av, "open", lambda *_args, **_kwargs: container)

    dataset = object.__new__(LeRobotSFTDataset)
    dataset.temporal_compression_factor = 4
    video = dataset._decode_video_frames(
        video_path="unused.mp4",
        start_frame=5,
        end_frame=15,
        temporal_interval=2,
        resize_h=8,
        resize_w=10,
        target_h=4,
        target_w=6,
        crop_y=2,
        crop_x=2,
        original_fps=20.0,
    )

    assert video is not None
    assert video.shape == (3, 5, 4, 6)
    assert video.dtype is torch.uint8
    assert video.is_contiguous()
    assert video[:, :, 0, 0].tolist() == [[5, 7, 9, 11, 13]] * 3
    assert container.seek_args == (5, container.stream, True, False)
    assert container.stream.thread_count == 1
    assert container.closed
