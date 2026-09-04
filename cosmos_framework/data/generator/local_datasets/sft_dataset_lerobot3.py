# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# LeRobot 3.x SFT dataset loader —— 动态加载 LeRobot 数据集（episodes 表带 caption 列）。
#
# 本文件是 sft_dataset.py 的「LeRobot 扩展」，通过子类 + 新增函数承载新增逻辑，
# 原 sft_dataset.py（JSONL / S3 流程）保持一行不改。
#
# 与父模块的关系：
#   - 复用 SFTDataset（继承）、_select_caption / _flatten_metadata_by_window /
#     _DURATION_TEMPLATE / _RESOLUTION_TEMPLATE / _MAX_CAPTION_TOKENS（import）
#   - LeRobotSFTDataset override process_one_sample，只替换「中段视频加载」：
#     父类 = download 到临时文件 + 全量 ffmpeg decode + 过滤
#     本类 = 本地 mp4 直接 get_video_metadata + 可配置后端按帧区间读取
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from cosmos_framework.data.generator.local_datasets.helper import (
    get_aspect_ratio,
    get_video_metadata,
)
from cosmos_framework.data.generator.local_datasets.sft_dataset import (
    _DURATION_TEMPLATE,
    _MAX_CAPTION_TOKENS,
    _RESOLUTION_TEMPLATE,
    SFTDataset,
    _flatten_metadata_by_window,
    _select_caption,
)
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.data.generator.utils import VIDEO_RES_SIZE_INFO
from cosmos_framework.utils import log
from cosmos_framework.utils.flags import INTERNAL

# ============================================================================
# 1. video 字段选择 + metadata 加载
# ============================================================================


def _select_lerobot_video_key(
    info: dict,
    video_feature_key: str | None = None,
    video_feature_keywords: list[str] | None = None,
) -> str:
    """从 info.json 的 features 里选定要用的 video 字段名。

    优先级：
    1. 显式传入的 ``video_feature_key``（精确匹配）
    2. 关键字匹配：``video_feature_keywords`` 里任一关键字是 key 名的子串
       （如 ["top", "head"] 命中 "observation.images.top"），取第一个命中字段
    3. 第一个 ``dtype == "video"`` 的字段（兜底）
    """
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    if not video_keys:
        raise ValueError("info.json 的 features 里没有 dtype=video 的字段")

    if video_feature_key:
        if video_feature_key not in video_keys:
            raise ValueError(f"video_feature_key={video_feature_key!r} 不在 features 里")
        return video_feature_key

    if video_feature_keywords:
        for k in video_keys:
            if any(kw in k for kw in video_feature_keywords):
                return k

    return video_keys[0]


def _get_lerobot_video_width_height(info: dict, video_key: str) -> tuple[int, int]:
    """从 video 字段的 shape 抓 (width, height)。

    shape 是 [H, W, C]（或 names 里有 width/height），所以 width=shape[1], height=shape[0]。
    """
    feat = info["features"][video_key]
    shape = feat["shape"]  # [H, W, C]
    names = feat.get("names")  # 通常 ["height", "width", "channels"]
    if names and "width" in names and "height" in names:
        w = shape[names.index("width")]
        h = shape[names.index("height")]
    else:
        h, w = shape[0], shape[1]
    return w, h


def _discover_lerobot_roots(lerobot_root: str) -> list[str]:
    """发现给定路径下的所有 LeRobot 数据集根目录。

    - 若 ``lerobot_root`` 本身直接含 ``meta/info.json``，则它就是单个数据集，返回 ``[lerobot_root]``。
    - 否则递归遍历所有子目录，把每个含 ``meta/info.json`` 的子目录当作一个数据集根。

    不写死层数：任意深度下含 ``meta/info.json`` 的目录都会被当作一个数据集。
    """
    root = Path(lerobot_root)
    if (root / "meta" / "info.json").is_file():
        return [str(root)]

    roots = sorted(str(p) for p in root.rglob("meta/info.json"))
    # rglob 找到的是 .../meta/info.json，取其上一级目录（去掉 /meta/info.json）
    dataset_roots = [str(Path(p).parent.parent) for p in roots]
    if not dataset_roots:
        raise ValueError(f"在 {lerobot_root} 下没找到任何含 meta/info.json 的 LeRobot 数据集目录")
    return dataset_roots


def _load_single_lerobot_metadata(
    lerobot_root: str,
    min_frames: int,
    max_video_duration_s: float,
    min_short_edge: int,
    video_feature_key: str | None,
    caption_key: str,
    video_feature_keywords: list[str] | None = None,
) -> list[dict]:
    """读【单个】LeRobot 数据集，产出 metadata list。

    被 ``_load_lerobot_metadata`` 调用（后者负责发现多个数据集根并逐个加载合并）。
    输出的 metadata dict 结构与 ``sft_dataset._load_sft_metadata_from_s3`` 完全一致：
    {uuid, vision_path, width, height, nb_frames, framerate, aspect_ratio, t2w_windows}。
    """
    import pandas as pd

    root = Path(lerobot_root)
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = float(info["fps"])

    video_key = _select_lerobot_video_key(info, video_feature_key, video_feature_keywords)
    width, height = _get_lerobot_video_width_height(info, video_key)

    # 读 episodes 表（可能跨多个 chunk/file parquet），每行一个 episode
    episodes_files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not episodes_files:
        raise ValueError(f"在 {root / 'meta' / 'episodes'} 下没找到 episodes parquet")
    episodes_df = pd.concat([pd.read_parquet(p) for p in episodes_files], ignore_index=True)

    metadata_list: list[dict] = []
    for row in episodes_df.to_dict("records"):
        episode_index = int(row["episode_index"])

        # 定位该 episode 的数据文件（跨文件唯一）
        data_chunk = int(row.get("data/chunk_index", 0))
        data_file = int(row.get("data/file_index", 0))

        # 该 episode 视频在选定 camera 的 mp4 里的时间区间
        from_ts = float(row.get(f"videos/{video_key}/from_timestamp", 0.0))
        to_ts = float(row.get(f"videos/{video_key}/to_timestamp", 0.0))

        # 时间区间 → 帧编号（to 是开区间，end 要 -1）
        start_frame = round(from_ts * fps)
        end_frame = round(to_ts * fps) - 1

        # 视频 chunk/file 定位（用于拼 vision_path）
        video_chunk = int(row.get(f"videos/{video_key}/chunk_index", 0))
        video_file = int(row.get(f"videos/{video_key}/file_index", 0))

        # Prefer the configured caption column.  Official LeRobot v3 datasets
        # (including nvidia/LIBERO_LeRobot_v3) store the natural-language task
        # as a one-element ``tasks`` array instead of a scalar ``caption``.
        # Accept that representation so the stock dataset does not silently
        # skip every episode.
        caption = row.get(caption_key)
        if caption is None and caption_key == "caption":
            caption = row.get("tasks")
        if isinstance(caption, (list, tuple, np.ndarray)):
            caption = next((str(item).strip() for item in caption if str(item).strip()), None)

        length = int(row.get("length", end_frame - start_frame + 1))
        duration = to_ts - from_ts

        # 过滤（默认值对齐 sft_dataset._load_sft_metadata_from_s3；LeRobot 路径可配置）
        if max_video_duration_s > 0 and duration > max_video_duration_s:
            continue
        if min_short_edge > 0 and min(width, height) < min_short_edge:
            continue
        frames_in_window = end_frame - start_frame + 1
        if frames_in_window < min_frames:
            continue

        uuid = f"{root.name}_chunk_{data_chunk}_file_{data_file}_episode_{episode_index}"

        # 用 info.json 的 video_path 模板拼本地 mp4 路径
        vision_path = str(
            root
            / info["video_path"].format(
                video_key=video_key,
                chunk_index=video_chunk,
                file_index=video_file,
                episode_chunk=video_chunk,
                episode_file=video_file,
            )
        )

        window = {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "temporal_interval": 1,
        }
        # caption 为空时不写 caption key，让下游 _select_caption 找不到 key → 返回 None → 优雅跳过该样本
        if caption:
            window["caption"] = caption

        metadata_list.append(
            {
                "uuid": uuid,
                "vision_path": vision_path,
                "width": width,
                "height": height,
                "nb_frames": length,
                "framerate": fps,
                "aspect_ratio": get_aspect_ratio(width, height),
                "t2w_windows": [window],
            }
        )

    return metadata_list


def _load_lerobot_metadata(
    lerobot_root: str,
    min_frames: int,
    max_video_duration_s: float,
    min_short_edge: int = 0,
    video_feature_key: str | None = None,
    caption_key: str = "caption",
    video_feature_keywords: list[str] | None = None,
) -> list[dict]:
    """读 LeRobot 数据集（单个根或父目录），产出 metadata list。

    支持两种 ``lerobot_root``：
    1. 单个数据集根目录（含 meta/info.json）
    2. 父目录（不含 meta/info.json，其任意深度子目录下含多个 meta/info.json）

    父目录场景会自动递归发现所有含 ``meta/info.json`` 的子目录，逐个加载并合并。
    """
    roots = _discover_lerobot_roots(lerobot_root)
    log.info(f"LeRobot 数据加载：发现 {len(roots)} 个数据集目录")

    metadata_list: list[dict] = []
    for root in roots:
        metadata_list.extend(
            _load_single_lerobot_metadata(
                root,
                min_frames=min_frames,
                max_video_duration_s=max_video_duration_s,
                min_short_edge=min_short_edge,
                video_feature_key=video_feature_key,
                caption_key=caption_key,
                video_feature_keywords=video_feature_keywords,
            )
        )
    return metadata_list


# ============================================================================
# 2. 可配置视频后端 + LeRobotSFTDataset 子类
# ============================================================================


_SUPPORTED_VIDEO_BACKENDS = {"pyav", "torchcodec"}
_SUPPORTED_VIDEO_RESIZE_MODES = {"decode_transform", "post_decode"}


def _limit_temporal_interval_by_fps(
    original_fps: float,
    temporal_interval: int,
    max_video_fps: float,
) -> int:
    """Return an integer frame stride whose effective FPS does not exceed the cap."""
    if original_fps <= 0:
        raise ValueError(f"original_fps must be positive, got {original_fps}")
    if temporal_interval < 1:
        raise ValueError(f"temporal_interval must be at least 1, got {temporal_interval}")
    if max_video_fps < 0:
        raise ValueError(f"max_video_fps must be non-negative, got {max_video_fps}")
    if max_video_fps == 0:
        return temporal_interval
    return max(temporal_interval, math.ceil(original_fps / max_video_fps))


class _LeRobotVideoDecoderCache:
    """按视频路径缓存 torchcodec VideoDecoder 的 LRU 缓存。

    仿照 action 侧 ``cosmos3_action_lerobot._LRUVideoDecoderCache``，但：
    - ``seek_mode="exact"``（精确帧定位，vision SFT 需切准 episode 帧边界）
    - torchcodec/fsspec 惰性 import，不污染主模块 import 路径
    """

    def __init__(self, max_size: int = 64):
        from collections import OrderedDict

        self._max_size = max_size
        self._cache: "OrderedDict[tuple[str, tuple[int, int] | None], tuple]" = OrderedDict()

    def get_decoder(self, video_path: str, resize_hw: tuple[int, int] | None = None):
        import fsspec
        from torchcodec.decoders import VideoDecoder
        from torchcodec.transforms import Resize

        cache_key = (video_path, resize_hw)
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key][0]

        file_handle = fsspec.open(video_path).__enter__()
        decoder_kwargs = {"seek_mode": "exact"}
        if resize_hw is not None:
            decoder_kwargs["transforms"] = [Resize(resize_hw)]
        try:
            decoder = VideoDecoder(file_handle, **decoder_kwargs)
        except Exception:
            file_handle.close()
            raise
        self._cache[cache_key] = (decoder, file_handle)

        while len(self._cache) > self._max_size:
            _, (_, old_fh) = self._cache.popitem(last=False)
            try:
                old_fh.close()
            except Exception:
                pass
        return decoder

    def discard(self, video_path: str) -> None:
        """Close and remove every cached decoder variant for one video path."""
        matching_keys = [key for key in self._cache if key[0] == video_path]
        for key in matching_keys:
            _, file_handle = self._cache.pop(key)
            try:
                file_handle.close()
            except Exception:
                pass


class LeRobotSFTDataset(SFTDataset):
    """SFTDataset 子类，override process_one_sample 为「本地 mp4 + 可配置后端读取」。

    与父类的唯一差异：``process_one_sample`` 里视频加载那一段。
    """

    def __init__(
        self,
        *args,
        video_backend: str = "torchcodec",
        video_resize_mode: str = "post_decode",
        video_tolerance_s: float = 1e-4,
        max_video_fps: float = 30.0,
        decoder_cache_max_size: int = 64,
        **kwargs,
    ):
        if video_backend not in _SUPPORTED_VIDEO_BACKENDS:
            raise ValueError(
                f"Unsupported video_backend={video_backend!r}; expected one of {sorted(_SUPPORTED_VIDEO_BACKENDS)}"
            )
        if video_resize_mode not in _SUPPORTED_VIDEO_RESIZE_MODES:
            raise ValueError(
                f"Unsupported video_resize_mode={video_resize_mode!r}; "
                f"expected one of {sorted(_SUPPORTED_VIDEO_RESIZE_MODES)}"
            )
        if video_tolerance_s <= 0:
            raise ValueError(f"video_tolerance_s must be positive, got {video_tolerance_s}")
        if max_video_fps < 0:
            raise ValueError(f"max_video_fps must be non-negative, got {max_video_fps}")
        if video_backend == "torchcodec" and decoder_cache_max_size < 1:
            raise ValueError(f"decoder_cache_max_size must be at least 1, got {decoder_cache_max_size}")
        super().__init__(*args, **kwargs)
        self.video_backend = video_backend
        self.video_resize_mode = video_resize_mode
        self.video_tolerance_s = video_tolerance_s
        self.max_video_fps = float(max_video_fps)
        self._decoder_cache = (
            _LeRobotVideoDecoderCache(max_size=decoder_cache_max_size) if video_backend == "torchcodec" else None
        )

    def _decode_video_frames_torchcodec(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        temporal_interval: int,
        resize_hw: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Decode an exact frame range with the existing cached TorchCodec path."""
        assert self._decoder_cache is not None
        decoder = self._decoder_cache.get_decoder(video_path, resize_hw=resize_hw)

        # torchcodec uses a half-open [start, stop) range.
        frame_batch = decoder.get_frames_in_range(
            start=start_frame,
            stop=end_frame + 1,
            step=temporal_interval,
        )
        data = frame_batch.data  # [N, C, H, W] uint8
        return data

    def _decode_video_frames_pyav(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        temporal_interval: int,
        original_fps: float,
        resize_hw: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Decode requested timestamps through PyAV, optionally resizing before tensor materialization."""

        frame_indices = range(start_frame, end_frame + 1, temporal_interval)
        timestamps = [frame_index / original_fps for frame_index in frame_indices]
        if resize_hw is None:
            from lerobot.datasets.video_utils import decode_video_frames

            data = decode_video_frames(
                video_path,
                timestamps,
                tolerance_s=self.video_tolerance_s,
                backend="pyav",
            )
            # LeRobot returns float32 TCHW in [0, 1]; normalize to the uint8
            # contract shared with the TorchCodec path.
            return data.mul(255).round().clamp(0, 255).to(torch.uint8)
        return self._decode_video_frames_pyav_resized(video_path, timestamps, resize_hw)

    def _decode_video_frames_pyav_resized(
        self,
        video_path: str,
        timestamps: list[float],
        resize_hw: tuple[int, int],
    ) -> torch.Tensor:
        """Mirror LeRobot's PyAV timestamp selection, resizing each AVFrame before stacking."""
        import av
        from av.video.reformatter import Interpolation
        from lerobot.datasets.video_utils import FrameTimestampError

        first_ts = min(timestamps)
        last_ts = max(timestamps)
        resize_h, resize_w = resize_hw
        loaded_frames = []
        loaded_ts = []

        container = av.open(video_path, metadata_errors="ignore")
        try:
            stream = container.streams.video[0]
            offset = int(round(max(first_ts, 0) / stream.time_base))
            container.seek(offset, backward=True, any_frame=False, stream=stream)
            for frame in container.decode(video=0):
                current_ts = float(frame.pts * frame.time_base)
                resized = frame.reformat(
                    width=resize_w,
                    height=resize_h,
                    format="rgb24",
                    interpolation=Interpolation.BICUBIC,
                )
                loaded_frames.append(torch.as_tensor(resized.to_ndarray()).permute(2, 0, 1))
                loaded_ts.append(current_ts)
                if current_ts >= last_ts:
                    break
        finally:
            container.close()

        query_ts = torch.tensor(timestamps)
        decoded_ts = torch.tensor(loaded_ts)
        if not loaded_frames:
            raise FrameTimestampError(f"No frames decoded from video: {video_path}")
        distances = torch.cdist(query_ts[:, None], decoded_ts[:, None], p=1)
        minimum, closest_indices = distances.min(1)
        within_tolerance = minimum < self.video_tolerance_s
        if not within_tolerance.all():
            raise FrameTimestampError(
                "One or several query timestamps unexpectedly violate the tolerance "
                f"({minimum[~within_tolerance]} > tolerance_s={self.video_tolerance_s})."
                f"\nqueried timestamps: {query_ts}"
                f"\nloaded timestamps: {decoded_ts}"
                f"\nvideo: {video_path}"
                "\nbackend: pyav"
            )

        closest_frames = torch.stack([loaded_frames[index] for index in closest_indices])
        return closest_frames

    def _decode_video_frames(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        temporal_interval: int,
        original_fps: float,
        resize_h: int,
        resize_w: int,
    ) -> list[np.ndarray]:
        """按帧编号区间 seek，只解码目标 episode 的帧。

        与父类对齐的返回格式：list[np.ndarray]（每帧 HWC uint8），供下游 np.stack。
        """
        decode_resize_hw = (resize_h, resize_w) if self.video_resize_mode == "decode_transform" else None

        if self.video_backend == "torchcodec":
            data = self._decode_video_frames_torchcodec(
                video_path,
                start_frame,
                end_frame,
                temporal_interval,
                resize_hw=decode_resize_hw,
            )
        else:
            data = self._decode_video_frames_pyav(
                video_path,
                start_frame,
                end_frame,
                temporal_interval,
                original_fps,
                resize_hw=decode_resize_hw,
            )

        if self.video_resize_mode == "post_decode":
            import torch.nn.functional as F

            data = data.float()
            data = F.interpolate(data, size=(resize_h, resize_w), mode="bicubic", align_corners=False)
            data = data.round().clamp(0, 255).to(torch.uint8)

        # [N, C, H, W] (uint8) -> list of [H, W, C] (uint8)，对齐父类返回格式
        data_nhwc = data.permute(0, 2, 3, 1).cpu().numpy()  # [N, H, W, C] uint8
        return [data_nhwc[i] for i in range(data_nhwc.shape[0])]

    def process_one_sample(self, metadata: dict) -> dict | None:
        """Process a single LeRobot SFT sample.

        ⚠️ 与 sft_dataset.py:SFTDataset.process_one_sample 保持同步，唯一差异是中段视频加载：
        父类 = download 到临时文件 + 全量 ffmpeg decode + 过滤；
        本类 = 本地 mp4 直接 get_video_metadata + 可配置后端按帧区间读取。
        """
        windows = metadata["t2w_windows"]
        win_idx = random.randrange(len(windows))
        t2w_window = windows[win_idx]
        window_start = t2w_window["start_frame"]
        window_end = t2w_window["end_frame"]

        # Compute output resolution
        input_w, input_h = metadata["width"], metadata["height"]
        target_w, target_h = self.output_sizes[metadata["aspect_ratio"]]
        resize_ratio = max(target_w / input_w, target_h / input_h)
        resize_h, resize_w = (round(input_h * resize_ratio), round(input_w * resize_ratio))
        crop_y, crop_x = (round((resize_h - target_h) / 2), round((resize_w - target_w) / 2))

        # 【LeRobot 差异】本地 mp4 直接读 metadata，跳过 download + 临时文件
        input_video_path = metadata["vision_path"]
        try:
            video_info = get_video_metadata(input_video_path)
        except Exception as error:
            log.exception(
                "Failed to read video metadata; skipping sample and advancing to the next video. "
                f"uuid={metadata['uuid']}, path={input_video_path}, "
                f"error={type(error).__name__}: {error}",
                rank0_only=False,
            )
            return None
        original_fps = video_info["fps"]
        total_frames = video_info["total_frames"]

        # Constrain to the t2w window
        actual_end = min(window_end, total_frames - 1)
        frames_in_window = actual_end - window_start + 1

        if self.num_video_frames == -1:
            # Native chunk mode: use start/end/interval directly from the window
            temporal_interval = t2w_window["temporal_interval"]
            start_frame = window_start
            end_frame = actual_end
        else:
            if frames_in_window < self.num_video_frames:
                log.warning(
                    f"Not enough frames in window: {metadata['uuid']}, "
                    f"frames_in_window: {frames_in_window}, required: {self.num_video_frames}"
                )
                return None

            # Compute temporal interval
            if self.temporal_interval_mode == "force_one":
                temporal_interval = 1
            elif self.temporal_interval_mode == "max_30fps":
                temporal_interval = max(1, math.ceil(original_fps / 30.0))
            elif self.temporal_interval_mode == "entire_chunk":
                temporal_interval = frames_in_window // self.num_video_frames
                temporal_interval = max(1, temporal_interval)
            else:
                raise ValueError(f"Unknown temporal_interval_mode: {self.temporal_interval_mode}")

        temporal_interval = _limit_temporal_interval_by_fps(
            original_fps,
            temporal_interval,
            self.max_video_fps,
        )

        if self.num_video_frames != -1:
            num_frames_before_downsample = (self.num_video_frames - 1) * temporal_interval + 1
            if num_frames_before_downsample > frames_in_window:
                log.warning(
                    f"FPS cap leaves too few frames in window: {metadata['uuid']}, "
                    f"original_fps={original_fps}, max_video_fps={self.max_video_fps}, "
                    f"temporal_interval={temporal_interval}, frames_in_window={frames_in_window}, "
                    f"required_span={num_frames_before_downsample}, requested_frames={self.num_video_frames}"
                )
                return None
            if self.frame_selection_mode == "first":
                start_frame = window_start
            elif self.frame_selection_mode == "center":
                start_frame = window_start + (frames_in_window - num_frames_before_downsample) // 2
            elif self.frame_selection_mode == "random":
                max_offset = frames_in_window - num_frames_before_downsample
                start_frame = window_start + random.randint(0, max(0, max_offset))
            else:
                raise ValueError(f"Unknown frame_selection_mode: {self.frame_selection_mode}")
            end_frame = start_frame + num_frames_before_downsample - 1

        fps = original_fps / temporal_interval

        # 【LeRobot 差异】可配置后端按 episode 帧区间读取，替代父类全量 ffmpeg decode
        try:
            video_chunk = self._decode_video_frames(
                video_path=input_video_path,
                start_frame=start_frame,
                end_frame=end_frame,
                temporal_interval=temporal_interval,
                original_fps=original_fps,
                resize_h=resize_h,
                resize_w=resize_w,
            )
        except Exception as error:
            if self._decoder_cache is not None:
                self._decoder_cache.discard(input_video_path)
            log.exception(
                "Failed to decode video; skipping sample and advancing to the next video. "
                f"uuid={metadata['uuid']}, path={input_video_path}, "
                f"backend={self.video_backend}, resize_mode={self.video_resize_mode}, "
                f"frame_range=[{start_frame}, {end_frame}], temporal_interval={temporal_interval}, "
                f"error={type(error).__name__}: {error}",
                rank0_only=False,
            )
            return None

        if not video_chunk:
            log.warning(
                f"No frames decoded for sample: {metadata['uuid']} "
                f"(start={start_frame}, end={end_frame}, path={metadata['vision_path']})"
            )
            return None

        video_chunk = np.stack(video_chunk, axis=0)  # [T,H,W,3]

        # Truncate temporally to temporal_compression_factor * N + 1
        target_t = (video_chunk.shape[0] - 1) // self.temporal_compression_factor * self.temporal_compression_factor + 1

        # Apply spatial center crop and temporal truncation
        video_chunk = video_chunk[:target_t, crop_y : crop_y + target_h, crop_x : crop_x + target_w]  # [T,H,W,3]

        # THWC -> CTHW
        video_chunk = np.transpose(video_chunk, (3, 0, 1, 2))  # [3,T,H,W]
        video = torch.from_numpy(np.ascontiguousarray(video_chunk)).to(torch.uint8)  # [3,T,H,W]
        padding_mask = torch.zeros((1, target_h, target_w), dtype=torch.float32)
        # image_size: [target_h, target_w, orig_h, orig_w] in pixel space, for the model to crop the video
        image_size = torch.tensor([target_h, target_w, target_h, target_w], dtype=torch.float32)

        selected = _select_caption(t2w_window)
        if selected is None:
            log.warning(
                f"No known caption key found in t2w_window for sample {metadata['uuid']}. "
                f"Keys: {list(t2w_window)}. Skipping sample."
            )
            return None
        caption_key, caption, used_structured_json = selected

        num_decoded_frames = video.shape[1]
        cond_fps = fps if self.conditioning_fps < 0 else self.conditioning_fps
        if self.conditioning_fps_noise_std > 0:
            noise_factor = np.exp(np.random.randn() * self.conditioning_fps_noise_std)
            cond_fps = cond_fps * noise_factor

        if self.caption_suffix and not used_structured_json:
            caption = (caption + " " + self.caption_suffix).strip()

        # CFG dropout: when cfg_dropout_keep_metadata is True, dropout fires
        # before appending resolution/duration/FPS so that metadata text is
        # preserved even under unconditional guidance.
        if self.cfg_dropout_keep_metadata and self.cfg_dropout_rate > 0:
            if random.random() < self.cfg_dropout_rate:
                caption = ""

        # Structured-JSON captions already carry duration/fps/resolution inside the
        # JSON, so skip the natural-language metadata suffixes for them. This also
        # makes the training prompt byte-match the inference prompt.
        if self.append_duration_fps_timestamps and not used_structured_json:
            duration = num_decoded_frames / cond_fps
            suffix = _DURATION_TEMPLATE.format(duration=duration, fps=cond_fps)
            caption = caption + " " + suffix
        if self.append_resolution_info and not used_structured_json:
            suffix = _RESOLUTION_TEMPLATE.format(height=target_h, width=target_w)
            caption = caption + " " + suffix
        caption = caption.strip()

        if not self.cfg_dropout_keep_metadata and self.cfg_dropout_rate > 0:
            if random.random() < self.cfg_dropout_rate:
                caption = ""
        text_ids, caption = self._tokenize_caption(caption)

        ret = dict(
            __key__=f"{metadata['uuid']}_w{win_idx}",
            __url__=metadata["vision_path"],
            fps=original_fps,
            n_orig_video_frames=total_frames,
            chunk_index=win_idx,
            frame_start=start_frame,
            frame_end=end_frame,
            num_frames=video.shape[1],
            video=video,
            num_multiplier=temporal_interval,
            conditioning_fps=cond_fps,
            padding_mask=padding_mask,
            image_size=image_size,
            ai_caption=caption,
            sampled_caption_style=caption_key,
            text_token_ids=torch.tensor(text_ids),
        )

        if self.conditioning_config is not None:
            num_frames_pixel = video.shape[1]
            t_latent = 1 + (num_frames_pixel - 1) // self.temporal_compression_factor
            frames_options = list(self.conditioning_config.keys())
            weights = list(self.conditioning_config.values())
            num_cond = random.choices(frames_options, weights=weights, k=1)[0]
            num_cond = min(num_cond, t_latent - 1)
            ret["sequence_plan"] = SequencePlan(
                has_text=True,
                has_vision=True,
                condition_frame_indexes_vision=list(range(num_cond)),
            )

        return ret


# ============================================================================
# 3. 入口函数
# ============================================================================


def get_sft_dataset_from_lerobot(
    lerobot_root: str,
    resolution: str = "720",
    num_video_frames: int = -1,  # LeRobot 场景默认 -1（native chunk mode，直接用 t2w_windows 里的帧区间）
    min_video_frames: int = 61,
    max_video_duration_s: float = 61.0,
    temporal_interval_mode: str = "entire_chunk",
    frame_selection_mode: str = "center",
    tokenizer_config: Optional[Any] = None,
    cfg_dropout_rate: float = 0.1,
    use_system_prompt: bool = False,
    max_caption_tokens: int = _MAX_CAPTION_TOKENS,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    cfg_dropout_keep_metadata: bool = False,
    sample_by_window: bool = False,
    min_short_edge: int = 0,
    caption_suffix: str = "",
    conditioning_fps: float = 24,
    conditioning_fps_noise_std: float = 0.0,
    conditioning_config: dict[int, float] | None = None,
    temporal_compression_factor: int = 4,
    video_feature_key: str | None = None,
    caption_key: str = "caption",
    video_feature_keywords: list[str] | None = None,
    video_backend: str = "torchcodec",
    video_resize_mode: str = "post_decode",
    video_tolerance_s: float = 1e-4,
    max_video_fps: float = 30.0,
    decoder_cache_max_size: int = 64,
    **kwargs,
) -> LeRobotSFTDataset:
    """LeRobot 版 get_sft_dataset，动态加载 LeRobot 数据集。

    与 ``sft_dataset.get_sft_dataset`` 的差异仅 3 处：
    1. 签名：``lerobot_root`` + ``video_feature_key``/``video_feature_keywords`` 替代 ``jsonl_paths``
    2. metadata 来源：``_load_lerobot_metadata`` 替代 ``_load_sft_metadata_from_s3``
    3. 构造类：``LeRobotSFTDataset`` 替代 ``SFTDataset``（后者 override 视频加载为按帧编号 seek）

    其余参数、flatten/shuffle、构造参数列表均与 ``get_sft_dataset`` 一致。
    """
    log.info(f"Unknown kwargs for get_sft_dataset_from_lerobot: {kwargs}")
    assert resolution in VIDEO_RES_SIZE_INFO.keys(), "The provided resolution cannot be found in VIDEO_RES_SIZE_INFO."
    if min_video_frames < 1:
        raise ValueError(f"min_video_frames must be at least 1, got {min_video_frames}")
    if max_video_duration_s < 0:
        raise ValueError(f"max_video_duration_s must be non-negative, got {max_video_duration_s}")
    if max_video_fps < 0:
        raise ValueError(f"max_video_fps must be non-negative, got {max_video_fps}")

    # LeRobot 是本地加载，不需要 S3 下载凭证（SFTDataset 构造仍要求 s3_credentials 参数）
    if INTERNAL:
        with open("credentials/gcs.secret", "r") as f:
            credentials = json.load(f)
    else:
        credentials = {}

    metadata_list = _load_lerobot_metadata(
        lerobot_root,
        min_frames=min_video_frames,
        max_video_duration_s=max_video_duration_s,
        min_short_edge=min_short_edge,
        video_feature_key=video_feature_key,
        caption_key=caption_key,
        video_feature_keywords=video_feature_keywords,
    )

    total_windows = sum(len(m["t2w_windows"]) for m in metadata_list)
    log.info(
        f"Finished loading LeRobot metadata from {lerobot_root}. "
        f"Total episodes: {len(metadata_list)}, total windows: {total_windows}"
    )

    if sample_by_window:
        metadata_list = _flatten_metadata_by_window(metadata_list)
        log.info(f"sample_by_window=True: flattened to {len(metadata_list)} samples (one per window)")

    # Deterministic shuffle based on the sha256 hash of uuid（与 get_sft_dataset 一致）
    metadata_list.sort(key=lambda x: hashlib.sha256(x["uuid"].encode("utf-8")).hexdigest())

    dataset = LeRobotSFTDataset(
        metadata=metadata_list,
        num_video_frames=num_video_frames,
        resolution=resolution,
        s3_credentials=credentials,
        temporal_interval_mode=temporal_interval_mode,
        frame_selection_mode=frame_selection_mode,
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        use_system_prompt=use_system_prompt,
        max_caption_tokens=max_caption_tokens,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        cfg_dropout_keep_metadata=cfg_dropout_keep_metadata,
        caption_suffix=caption_suffix,
        conditioning_fps=conditioning_fps,
        conditioning_fps_noise_std=conditioning_fps_noise_std,
        conditioning_config=conditioning_config,
        temporal_compression_factor=temporal_compression_factor,
        video_backend=video_backend,
        video_resize_mode=video_resize_mode,
        video_tolerance_s=video_tolerance_s,
        max_video_fps=max_video_fps,
        decoder_cache_max_size=decoder_cache_max_size,
    )
    return dataset
