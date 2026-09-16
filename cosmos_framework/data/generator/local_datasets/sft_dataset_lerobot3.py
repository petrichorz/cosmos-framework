# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# LeRobot 3.x SFT dataset loader —— 动态加载 LeRobot 数据集（episodes 表带 caption 列）。
#
# 本文件是 sft_dataset.py 的「LeRobot 独立实现」，通过独立 IterableDataset + 新增函数
# 承载 LeRobot 数据加载逻辑，原 sft_dataset.py（JSONL / S3 流程）保持一行不改。
#
# 与 sft_dataset.py 的关系（只 import 纯函数，不继承 SFTDataset）：
#   - 复用 _select_caption / _CAUSAL_DURATION_TEMPLATE / _RESOLUTION_TEMPLATE /
#     _MAX_CAPTION_TOKENS（import）
#   - metadata 与视频加载都基于 lerobot 官方 package（参考 action 侧 cosmos3_action_lerobot）：
#     LeRobotDatasetMetadata 读 info/episodes（自动 drop stats 列、保留 caption），
#     decode_video_frames 按时间戳解码视频（torchcodec + LRU decoder cache）。
#   - LeRobotSFTDataset 是独立 IterableDataset，自实现 __init__/__len__/__iter__/
#     _tokenize_caption/process_one_sample，保留多分辨率/多 fps 的扩展。
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import huggingface_hub.constants as _hf_const
import numpy as np
import torch
from lerobot.datasets import video_utils as _vu
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.video_utils import FrameTimestampError

from cosmos_framework.data.generator.local_datasets.helper import (
    get_aspect_ratio,
)
from cosmos_framework.data.generator.local_datasets.sft_dataset import (
    _MAX_CAPTION_TOKENS,
    _RESOLUTION_TEMPLATE,
    _select_caption,
)
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.data.generator.sequence_packing.modalities import add_special_tokens
from cosmos_framework.data.generator.utils import VIDEO_RES_SIZE_INFO
from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import tokenize_caption
from cosmos_framework.utils import log
from cosmos_framework.utils.lazy_config import instantiate as lazy_instantiate

# 多分辨率训练：候选档位（短边像素），只选 <= 视频短边的档位（不上采样）。
# _MULTI_RESOLUTION_TIERS = ("256", "480", "720")
_MULTI_RESOLUTION_TIERS = ("256", "480")
# 多 fps 训练：候选 temporal_interval（保留 1/2、1/3、1/4）。
_MULTI_FPS_INTERVALS = (2, 3, 4)

# causal 训练：caption 只追加 FPS，不再追加时长（时长信息由帧数/时序隐式表达）。
_CAUSAL_DURATION_TEMPLATE = "The video is of {fps:.0f} FPS."

# lerobot 解码器 LRU 缓存：见 lerobot.datasets.video_utils.LRUVideoDecoderCache。
# ⚠️ 仅 torchcodec 后端生效：lerobot 的 pyav 路径（decode_video_frames_torchvision）每次新建
# reader、用完即关，不查 _default_decoder_cache。所以当前 video_backend="pyav" 下本缓存是 no-op。

# long_video_policy 支持的策略：drop（丢弃超长 episode）/ split（切成均衡、连续的窗口）。
_SUPPORTED_LONG_VIDEO_POLICIES = {"drop", "split"}
# 可配置视频解码后端。
_SUPPORTED_VIDEO_BACKENDS = {"pyav", "torchcodec"}

_hf_offline_applied = False
_decoder_cache_patched = False


def _ensure_hf_hub_offline() -> None:
    """强制 HF Hub 离线，仅加载本地数据集（repo_id="local"）。

    幂等，每个进程只生效一次。参考 action 侧 ``cosmos3_action_lerobot._ensure_hf_hub_offline``。
    """
    global _hf_offline_applied
    if _hf_offline_applied:
        return
    if "HF_HUB_OFFLINE" not in os.environ:
        os.environ["HF_HUB_OFFLINE"] = "1"
    if not _hf_const.HF_HUB_OFFLINE:
        _hf_const.HF_HUB_OFFLINE = True
    _hf_offline_applied = True


def _patch_decoder_cache(max_size: int = None) -> None:
    """把 lerobot 模块级 ``_default_decoder_cache`` 替换为 LRU 版，防止无界内存增长。

    幂等，每个进程只 patch 一次。参考 action 侧 ``cosmos3_action_lerobot._patch_decoder_cache``。

    ⚠️ 仅 torchcodec 后端生效：pyav 路径不查 ``_default_decoder_cache``，因此在
    ``video_backend="pyav"`` 下本函数是 no-op（替换了也无人使用）。保留它是为了将来
    切 torchcodec 时能防 lerobot 无界缓存的 worker 内存膨胀。
    """
    global _decoder_cache_patched
    if _decoder_cache_patched:
        return
    if max_size is None:
        max_size = _vu.LRU_VIDEO_CACHE_MAX_SIZE
    _vu._default_decoder_cache = _vu.LRUVideoDecoderCache(max_size=max_size)
    _decoder_cache_patched = True


# ============================================================================
# 1. video 字段选择 + metadata 加载
# ============================================================================


@dataclass
class _LerobotSource:
    """一个 LeRobot 数据集的惰性加载描述符。

    数据集级常量（width/height/fps/aspect_ratio/total_frames/video_key 等）
    只在此存一份；episode 级字段（from/to_timestamp、start/end_frame、caption、
    uuid、vision_path 等）在采样期由 ``process_one_sample`` 从
    ``meta.episodes[ep_idx]`` 现算，不再在加载期物化成 dict。
    """

    root: Path
    meta: LeRobotDatasetMetadata
    video_key: str
    width: int
    height: int
    fps: float
    aspect_ratio: str
    total_frames: int
    caption_key: str
    root_hash: str
    name: str
    # 每个 episode 的 clip 帧范围列表（与 meta.episodes 对齐）：未切分时每项为 [(start, end)]，
    # long_video_policy="split" 时每项为多个连续窗口。
    episode_clips: list[list[tuple[int, int]]]


def _select_lerobot_video_key(
    meta,
    video_feature_key: str | None = None,
    video_feature_keywords: list[str] | None = None,
) -> str:
    """从 LeRobot 数据集 metadata 里选定要用的 video 字段名。

    优先级：
    1. 显式传入的 ``video_feature_key``（精确匹配）
    2. 关键字匹配：``video_feature_keywords`` 里任一关键字是 key 名的子串
       （如 ["top", "head"] 命中 "observation.images.top"），取第一个命中字段
    3. 第一个 video 字段（兜底）

    用官方 ``LeRobotDatasetMetadata.video_keys``（dtype=="video"）做候选集合，
    不再手写 ``info["features"]`` 的 dtype 过滤。
    """
    video_keys = meta.video_keys
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


def _discover_lerobot_roots(lerobot_root: str) -> list[str]:
    """发现给定路径下的所有 LeRobot 数据集根目录。

    - 若 ``lerobot_root`` 本身直接含 ``meta/info.json``，则它就是单个数据集，返回 ``[lerobot_root]``。
    - 否则递归遍历所有子目录，把每个含 ``meta/info.json`` 的子目录当作一个数据集根。

    不写死层数：任意深度下含 ``meta/info.json`` 的目录都会被当作一个数据集。
    """
    root = Path(lerobot_root)
    if (root / "meta" / "info.json").is_file():
        return [str(root)]

    roots = sorted(
        str(p)
        for p in root.rglob("meta/info.json")
    )
    # rglob 找到的是 .../meta/info.json，取其上一级目录（去掉 /meta/info.json）
    dataset_roots = [str(Path(p).parent.parent) for p in roots]
    if not dataset_roots:
        raise ValueError(
            f"在 {lerobot_root} 下没找到任何含 meta/info.json 的 LeRobot 数据集目录"
        )
    return dataset_roots


def _build_balanced_video_windows(
    start_frame: int,
    end_frame: int,
    fps: float,
    max_video_duration_s: float,
    video_window_overlap_s: float,
) -> list[tuple[int, int]]:
    """Split an inclusive frame range into balanced, continuous overlapping windows."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if end_frame < start_frame:
        raise ValueError(f"end_frame must be >= start_frame, got [{start_frame}, {end_frame}]")
    if max_video_duration_s <= 0:
        return [(start_frame, end_frame)]
    if video_window_overlap_s < 0 or video_window_overlap_s >= max_video_duration_s:
        raise ValueError(
            "video_window_overlap_s must satisfy 0 <= overlap < max_video_duration_s, "
            f"got overlap={video_window_overlap_s}, max={max_video_duration_s}"
        )

    max_frames = math.floor(max_video_duration_s * fps)
    overlap_frames = math.floor(video_window_overlap_s * fps)
    if max_frames < 1:
        raise ValueError(f"max_video_duration_s={max_video_duration_s} is shorter than one frame at fps={fps}")
    if overlap_frames >= max_frames:
        raise ValueError(f"Rounded overlap_frames={overlap_frames} must be smaller than max_frames={max_frames}")

    source_frames = end_frame - start_frame + 1
    if source_frames <= max_frames:
        return [(start_frame, end_frame)]

    stride_capacity = max_frames - overlap_frames
    num_windows = math.ceil((source_frames - overlap_frames) / stride_capacity)
    materialized_frames = source_frames + (num_windows - 1) * overlap_frames
    base_size, extra = divmod(materialized_frames, num_windows)

    windows: list[tuple[int, int]] = []
    cursor = start_frame
    for clip_index in range(num_windows):
        clip_size = base_size + int(clip_index < extra)
        clip_end = cursor + clip_size - 1
        windows.append((cursor, clip_end))
        cursor = clip_end + 1 - overlap_frames

    assert windows[-1][1] == end_frame
    assert all(window_end - window_start + 1 <= max_frames for window_start, window_end in windows)
    return windows


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


def _build_lerobot_source(
    lerobot_root: str,
    min_frames: int,
    max_duration_s: float,
    min_short_edge: int,
    video_feature_key: str | None,
    caption_key: str,
    video_feature_keywords: list[str] | None = None,
    long_video_policy: str = "drop",
    video_window_overlap_s: float = 0.0,
) -> tuple[_LerobotSource, list[tuple[int, int]]]:
    """读【单个】LeRobot 数据集，产出 (source 描述符, 有效 clip 索引列表)。

    惰性化加载：这里只解析数据集级常量（宽高/fps/aspect_ratio/total_frames 等）
    并跑一遍过滤，收集「有效 clip 的索引」``(ep_idx, clip_idx)``，**不物化任何
    episode dict**。episode 级字段在采样期由 ``process_one_sample`` 现算。

    long_video_policy="split" 时，超长 episode 切成多个 clip，每个 clip 独立成为
    一条训练样本；``source.episode_clips[ep_idx]`` 存该 episode 的全部 clip 帧范围。
    """
    if long_video_policy not in _SUPPORTED_LONG_VIDEO_POLICIES:
        raise ValueError(
            f"Unsupported long_video_policy={long_video_policy!r}; "
            f"expected one of {sorted(_SUPPORTED_LONG_VIDEO_POLICIES)}"
        )

    root = Path(lerobot_root)
    # repo_id="local" + revision="local"：本地数据集，避开 HF Hub 联网（"local" 不是合法 version）
    meta = LeRobotDatasetMetadata(repo_id="local", root=str(root), revision="local")
    fps = float(meta.fps)

    video_key = _select_lerobot_video_key(meta, video_feature_key, video_feature_keywords)
    # 直接读该 video feature 的 shape/names（用 .get 兜底），避免用 meta.names / meta.shapes
    # 这两个官方 property——它们用 ft["names"] / ft["shape"] 方括号遍历【所有】feature，
    # 任一标量列（如 frame_index/timestamp）缺 names 字段就会整体 KeyError。
    ft = meta.features[video_key]
    shape = ft["shape"]  # [H, W, C]（或 [H, W]）
    names = ft.get("names")  # 通常是 ["height", "width", "channels"]，可能为 None
    if names and "width" in names and "height" in names:
        width = shape[names.index("width")]
        height = shape[names.index("height")]
    else:
        height, width = shape[0], shape[1]

    # 数据集级常量只存一份（原来每个 episode dict 都重复存一份）
    source = _LerobotSource(
        root=root,
        meta=meta,
        video_key=video_key,
        width=width,
        height=height,
        fps=fps,
        aspect_ratio=get_aspect_ratio(width, height),
        total_frames=int(meta.total_frames),
        caption_key=caption_key,
        root_hash=hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:8],
        name=root.name,
        episode_clips=[],
    )

    # 数据集级过滤：短边不满足时整个数据集所有 episode 都无效
    if min_short_edge > 0 and min(width, height) < min_short_edge:
        return source, []

    valid_clips: list[tuple[int, int]] = []
    # meta.episodes 是 HF Dataset（pyarrow 内存映射，load_episodes 已自动 drop 掉 stats/ 列）
    for ep_pos, ep in enumerate(meta.episodes):
        from_ts = float(ep.get(f"videos/{video_key}/from_timestamp", 0.0))
        to_ts = float(ep.get(f"videos/{video_key}/to_timestamp", 0.0))
        duration = to_ts - from_ts
        start_frame = round(from_ts * fps)
        end_frame = round(to_ts * fps) - 1

        # 过滤：max_duration_s=0 关闭时长上限；>0 且超长时，drop 丢弃 / split 切分
        if max_duration_s > 0 and duration > max_duration_s and long_video_policy == "drop":
            source.episode_clips.append([])
            continue
        frames_in_window = end_frame - start_frame + 1
        if frames_in_window < min_frames:
            source.episode_clips.append([])
            continue

        if long_video_policy == "split":
            clip_ranges = _build_balanced_video_windows(
                start_frame, end_frame, fps, max_duration_s, video_window_overlap_s
            )
        else:
            clip_ranges = [(start_frame, end_frame)]

        source.episode_clips.append(clip_ranges)
        for clip_idx in range(len(clip_ranges)):
            valid_clips.append((ep_pos, clip_idx))

    return source, valid_clips


def _load_lerobot_metadata(
    lerobot_root: str,
    min_frames: int = 61,
    max_duration_s: float = 61.0,
    min_short_edge: int = 0,
    video_feature_key: str | None = None,
    caption_key: str = "caption",
    video_feature_keywords: list[str] | None = None,
    long_video_policy: str = "drop",
    video_window_overlap_s: float = 0.0,
) -> tuple[list[_LerobotSource], list[tuple[int, int, int]]]:
    """读 LeRobot 数据集（单个根或父目录），产出 (sources, episode_index)。

    支持两种 ``lerobot_root``：
    1. 单个数据集根目录（含 meta/info.json）
    2. 父目录（不含 meta/info.json，其任意深度子目录下含多个 meta/info.json）

    父目录场景会自动递归发现所有含 ``meta/info.json`` 的子目录，逐个加载并合并。

    ``episode_index`` 是扁平索引，每项为 ``(ds_idx, ep_idx, clip_idx)``，``ds_idx``
    是 ``sources`` 里的下标（本地）。
    """
    roots = _discover_lerobot_roots(lerobot_root)
    log.info(f"LeRobot 数据加载：发现 {len(roots)} 个数据集目录")

    sources: list[_LerobotSource] = []
    episode_index: list[tuple[int, int, int]] = []
    for root in roots:
        source, valid_clips = _build_lerobot_source(
            root,
            min_frames=min_frames,
            max_duration_s=max_duration_s,
            min_short_edge=min_short_edge,
            video_feature_key=video_feature_key,
            caption_key=caption_key,
            video_feature_keywords=video_feature_keywords,
            long_video_policy=long_video_policy,
            video_window_overlap_s=video_window_overlap_s,
        )
        ds_idx = len(sources)
        sources.append(source)
        episode_index.extend((ds_idx, ep, clip) for ep, clip in valid_clips)

    return sources, episode_index


def _load_lerobot_metadata_from_manifest(
    manifest_path: str,
    min_frames: int = 61,
    max_duration_s: float = 61.0,
    min_short_edge: int = 0,
    video_feature_key: str | None = None,
    caption_key: str = "caption",
    video_feature_keywords: list[str] | None = None,
    long_video_policy: str = "drop",
    video_window_overlap_s: float = 0.0,
    manifest_max_workers: int | None = None,
) -> tuple[list[_LerobotSource], list[tuple[int, int, int]]]:
    """读 manifest 文件（JSONL，每行一个 dict），并行加载所有数据集并合并。

    manifest 每行支持的 key（其余 key 静默忽略）：
    - ``path``（必需）：数据集路径（单数据集根 or 父目录）
    - ``video_feature_key``（可选）：显式指定 feature 名
    - ``video_feature_keywords``（可选）：关键字 list
    - ``caption_key``（可选）：caption 列名

    三个参数可**逐行覆盖**；某行没写时回退到函数参数（config 传入的全局值）。

    并行策略：每个 path 的加载用 ``ThreadPoolExecutor`` 并行（``pd.read_parquet``
    是 I/O + C++ 密集、会释放 GIL，多线程即可并行，无需多进程的 pickle 开销）。
    ``manifest_max_workers`` 默认 ``min(len(tasks), 8)``。
    """
    # 第 1 步：解析 manifest → 任务列表（纯 json 解析，串行很快）
    tasks: list[tuple[str, str | None, list[str] | None, str]] = []
    with open(manifest_path, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            path = entry.get("path")
            if not path:
                log.warning(f"manifest 第 {line_no} 行缺少 'path' key，跳过")
                continue
            # 三个参数逐行覆盖，缺省回退 config 全局值
            row_feature_key = entry.get("video_feature_key", video_feature_key)                    # 显式 feature 名
            row_feature_keywords = entry.get("video_feature_keywords", video_feature_keywords)      # 关键字 list
            row_caption_key = entry.get("caption_key", caption_key)                                 # caption 列名
            tasks.append((path, row_feature_key, row_feature_keywords, row_caption_key))

    if not tasks:
        return [], []

    def _load_one(task):
        path, fk, fkw, ck = task
        return _load_lerobot_metadata(
            path,
            min_frames=min_frames,
            max_duration_s=max_duration_s,
            min_short_edge=min_short_edge,
            video_feature_key=fk,
            caption_key=ck,
            video_feature_keywords=fkw,
            long_video_policy=long_video_policy,
            video_window_overlap_s=video_window_overlap_s,
        )

    if manifest_max_workers is None:
        manifest_max_workers = min(len(tasks), 8)

    def _merge(results):
        sources: list[_LerobotSource] = []
        episode_index: list[tuple[int, int, int]] = []
        for srcs, clips in results:
            offset = len(sources)
            sources.extend(srcs)
            episode_index.extend((ds + offset, ep, clip) for ds, ep, clip in clips)
        return sources, episode_index

    if manifest_max_workers <= 1 or len(tasks) == 1:
        # 单线程：保持原有顺序，无并发开销
        results = [_load_one(task) for task in tasks]
    else:
        from concurrent.futures import ThreadPoolExecutor

        log.info(f"[manifest] 并行加载 {len(tasks)} 个数据集，max_workers={manifest_max_workers}")
        with ThreadPoolExecutor(max_workers=manifest_max_workers) as ex:
            # ex.map 保持输入顺序返回，结果顺序与 manifest 行顺序一致
            results = list(ex.map(_load_one, tasks))

    return _merge(results)


# ============================================================================
# 2. LeRobotSFTDataset（独立 IterableDataset，不继承 SFTDataset）
# ============================================================================


class LeRobotSFTDataset(torch.utils.data.IterableDataset):
    """LeRobot 3.x 版 vision SFT 数据集（独立实现，不继承 ``SFTDataset``）。

    与 ``sft_dataset.SFTDataset`` 的关系：
      - 数据来源不同：本地 LeRobot 目录（官方 ``LeRobotDatasetMetadata`` +
        ``decode_video_frames``），而非 S3 JSONL + ffmpeg。
      - 接口对齐：保持 IterableDataset 约定（``__iter__`` + ``shard_*`` 属性），
        供 ``RankPartitionedDataLoader`` 消费；返回 dict 结构与 ``SFTDataset`` 一致。
      - 复用纯函数：``_select_caption`` / ``_CAUSAL_DURATION_TEMPLATE`` /
        ``_RESOLUTION_TEMPLATE`` / ``_MAX_CAPTION_TOKENS``。
    """

    def __init__(
        self,
        sources: list[_LerobotSource],
        episode_index: list[tuple[int, int, int]],
        resolution: str,
        tokenizer_config: Optional[Any] = None,
        cfg_dropout_rate: float = 0.0,
        use_system_prompt: bool = False,
        max_caption_tokens: int = _MAX_CAPTION_TOKENS,
        append_duration_fps_timestamps: bool = True,
        append_resolution_info: bool = True,
        cfg_dropout_keep_metadata: bool = False,
        caption_suffix: str = "",
        conditioning_fps: float = 24,
        conditioning_fps_noise_std: float = 0.0,
        conditioning_config: dict[int, float] | None = None,
        temporal_compression_factor: int = 4,
        use_multi_resolution: bool = False,
        use_multi_fps: bool = False,
        video_backend: str | None = None,
        video_tolerance_s: float = 0.034,
        max_video_fps: float = 30.0,
        decoder_cache_max_size: int = _vu.LRU_VIDEO_CACHE_MAX_SIZE,
    ):
        assert temporal_compression_factor >= 1, "temporal_compression_factor must be >= 1"
        if video_backend is not None and video_backend not in _SUPPORTED_VIDEO_BACKENDS:
            raise ValueError(
                f"Unsupported video_backend={video_backend!r}; expected one of {sorted(_SUPPORTED_VIDEO_BACKENDS)}"
            )
        if video_tolerance_s <= 0:
            raise ValueError(f"video_tolerance_s must be positive, got {video_tolerance_s}")
        if max_video_fps < 0:
            raise ValueError(f"max_video_fps must be non-negative, got {max_video_fps}")

        _ensure_hf_hub_offline()
        # 仅 torchcodec 后端生效；pyav 下是 no-op（见 _patch_decoder_cache docstring）。
        _patch_decoder_cache(max_size=decoder_cache_max_size)

        self.sources = sources
        self.episode_index = episode_index
        self.resolution = resolution
        self.tokenizer_config = tokenizer_config
        self.cfg_dropout_rate = cfg_dropout_rate
        self.use_system_prompt = use_system_prompt
        self.max_caption_tokens = max_caption_tokens
        self.append_duration_fps_timestamps = append_duration_fps_timestamps
        self.append_resolution_info = append_resolution_info
        self.cfg_dropout_keep_metadata = cfg_dropout_keep_metadata
        self.caption_suffix = caption_suffix.strip()
        self.conditioning_fps = conditioning_fps
        self.conditioning_fps_noise_std = conditioning_fps_noise_std
        self.temporal_compression_factor = temporal_compression_factor

        self.conditioning_config: dict[int, float] | None = None
        if conditioning_config is not None:
            total_prob = sum(conditioning_config.values())
            assert total_prob > 0, "conditioning_config probabilities must sum to a positive number"
            self.conditioning_config = {k: v / total_prob for k, v in conditioning_config.items()}
            log.info(f"Conditioning config: {self.conditioning_config}")

        # LeRobot 扩展参数
        self.use_multi_resolution = use_multi_resolution
        self.use_multi_fps = use_multi_fps
        # 视频后端：默认走 get_safe_default_codec()（torchcodec 可用则用 torchcodec，否则 pyav）。
        self.video_backend = video_backend if video_backend else _vu.get_safe_default_codec()
        self.video_tolerance_s = video_tolerance_s
        self.max_video_fps = float(max_video_fps)

        # They will be set by the RankPartitionedDataLoader
        self.shard_world_size = None
        self.shard_rank = None
        self.shard_id = 0
        self.is_initialized = False
        self.output_sizes = VIDEO_RES_SIZE_INFO[resolution]

        _vlm_proc = lazy_instantiate(self.tokenizer_config)
        self.vlm_tokenizer = _vlm_proc.tokenizer
        self.vlm_tokenizer, _ = add_special_tokens(self.vlm_tokenizer)

    def __len__(self):
        return len(self.episode_index)

    def _tokenize_caption(self, caption: str) -> tuple[list[int], str]:
        text_ids = tokenize_caption(
            caption,
            self.vlm_tokenizer,
            is_video=True,
            use_system_prompt=self.use_system_prompt,
        )
        if len(text_ids) > self.max_caption_tokens:
            log.warning(f"Text ids are too long, truncating: {len(text_ids)} > {self.max_caption_tokens}")
        text_ids = text_ids[: self.max_caption_tokens]
        return text_ids, caption

    def process_one_sample(self, ds_idx: int, ep_idx: int, clip_idx: int) -> dict | None:
        """Process a single LeRobot SFT sample.

        惰性化采样：传入扁平索引 ``(ds_idx, ep_idx, clip_idx)``，从 ``self.sources[ds_idx]``
        取数据集级常量，从 ``meta.episodes[ep_idx]`` 现算 episode 级字段（caption、uuid、
        vision_path），clip 帧区间取自 ``source.episode_clips[ep_idx][clip_idx]``。随后依次
        做：分辨率选择 → 抽帧 → 按 backend/resize_mode 解码 → 空间/时间裁剪 → caption
        生成 → tokenize → 组装返回 dict。
        """
        source = self.sources[ds_idx]
        meta = source.meta
        ep = meta.episodes[ep_idx]

        # ---- episode 级字段现算（原加载期物化的 dict 字段） ----
        video_key = source.video_key
        episode_index = int(ep["episode_index"])
        data_chunk = int(ep.get("data/chunk_index", 0))
        data_file = int(ep.get("data/file_index", 0))

        # clip 帧区间直接取自加载期算好的 episode_clips（long_video_policy="split" 时一个
        # episode 可能对应多个 clip，clip_idx 定位到具体的连续窗口）
        window_start, window_end = source.episode_clips[ep_idx][clip_idx]

        # caption：优先读 episodes 表新增的 caption_key 列；取不到回退官方 tasks 列（任务名）。
        caption = ep.get(source.caption_key)
        if not caption:
            tasks = ep.get("tasks")
            # tasks 列是官方 episodes 表原生列，实际类型为 numpy.ndarray（非 list），
            # 用 hasattr(x, "__len__") 判断，取第一个任务名作为 caption。
            if tasks is not None and hasattr(tasks, "__len__") and len(tasks) > 0:
                caption = str(tasks[0])

        num_clips = len(source.episode_clips[ep_idx])
        uuid = f"{source.name}_{source.root_hash}_chunk_{data_chunk}_file_{data_file}_episode_{episode_index}"
        if num_clips > 1:
            uuid = f"{uuid}_clip_{clip_idx:03d}_of_{num_clips:03d}"
        input_video_path = str(source.root / meta.get_video_file_path(ep_idx, video_key))

        # window 为单元素（一个 clip = 一段帧区间 + 一个 caption）
        t2w_window = {"start_frame": window_start, "end_frame": window_end, "temporal_interval": 1}
        # caption 为空时不写 caption key，让下游 _select_caption 找不到 key → 返回 None → 优雅跳过该样本
        if caption:
            t2w_window["caption"] = caption

        # ---- 数据集级常量（来自 source，只存一份） ----
        input_w, input_h = source.width, source.height
        original_fps = source.fps
        total_frames = source.total_frames

        # Compute output resolution
        if self.use_multi_resolution:
            # 多分辨率：候选档位 = 所有 <= 视频短边 的档位（不上采样），随机选一个。
            # 视频太小时 fallback 到最小档 "256"。
            video_min_edge = min(input_w, input_h)
            candidates = [r for r in _MULTI_RESOLUTION_TIERS if int(r) <= video_min_edge]
            if not candidates:
                candidates = ["256"]
            output_sizes = VIDEO_RES_SIZE_INFO[random.choice(candidates)]
        else:
            output_sizes = self.output_sizes
        target_w, target_h = output_sizes[source.aspect_ratio]
        resize_ratio = max(target_w / input_w, target_h / input_h)
        resize_h, resize_w = (round(input_h * resize_ratio), round(input_w * resize_ratio))
        crop_y, crop_x = (round((resize_h - target_h) / 2), round((resize_w - target_w) / 2))

        # Native chunk mode：直接用 window 的帧区间抽帧。
        # 抽帧步长：use_multi_fps 时随机 2/3/4（保留 1/2、1/3、1/4），否则取 window 自带 interval（=1）；
        # 再叠加 max_video_fps 上限（整数 stride，保证有效 fps 不超过 cap，0 表示关闭）。
        actual_end = min(window_end, total_frames - 1)
        if self.use_multi_fps:
            temporal_interval = random.choice(_MULTI_FPS_INTERVALS)
        else:
            temporal_interval = t2w_window["temporal_interval"]
        temporal_interval = _limit_temporal_interval_by_fps(
            original_fps,
            temporal_interval,
            self.max_video_fps,
        )
        start_frame = window_start
        end_frame = actual_end

        fps = original_fps / temporal_interval

        # 【lerobot 加载】帧号 → 绝对时间戳 → decode_video_frames 按 backend 解码
        # （pyav 在解码时 resize；torchcodec post-decode 单独 resize，两者都返回 uint8 [T,3,resize_h,resize_w]）
        frame_indices = list(range(start_frame, end_frame + 1, temporal_interval))
        timestamps = [idx / original_fps for idx in frame_indices]
        try:
            video_frames = _vu.decode_video_frames(
                input_video_path,
                timestamps,
                self.video_tolerance_s,
                self.video_backend,
                resize_h=resize_h,
                resize_w=resize_w,
            )
        except FrameTimestampError as e:
            # 时间戳与视频 pts 偏差超过 video_tolerance_s 时抛 FrameTimestampError。
            # 打印其详细提示（哪些时间戳违反 tolerance、视频路径等），并跳过该样本，避免中断训练。
            log.warning(
                f"FrameTimestampError decoding video for sample {uuid} "
                f"(start={start_frame}, end={end_frame}, path={input_video_path}): {e}"
            )
            return None
        except Exception as e:
            # 其它解码失败（坏文件、解码器异常等），同样跳过该样本。
            log.warning(
                f"Failed to decode video for sample {uuid} "
                f"(start={start_frame}, end={end_frame}, path={input_video_path}): "
                f"{type(e).__name__}: {e}"
            )
            return None

        if video_frames.shape[0] == 0:
            log.warning(
                f"No frames decoded for sample: {uuid} "
                f"(start={start_frame}, end={end_frame}, path={input_video_path})"
            )
            return None

        # _decode_video_frames 已保证输出 (resize_h, resize_w)，直接转 [T,H,W,3] uint8
        video_chunk = video_frames.permute(0, 2, 3, 1).cpu().numpy()  # [T,H,W,3] uint8

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
                f"No known caption key found in t2w_window for sample {uuid}. "
                f"Keys: {list(t2w_window)}. Skipping sample."
            )
            return None
        caption_key, caption, used_structured_json = selected

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
            suffix = _CAUSAL_DURATION_TEMPLATE.format(fps=cond_fps)
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
            __key__=f"{uuid}_w0",
            __url__=input_video_path,
            fps=original_fps,
            n_orig_video_frames=total_frames,
            chunk_index=0,
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

    def __iter__(self):
        assert not self.is_initialized, "Dataset can only be initialized once."
        assert len(self.episode_index) > 0, "Did not find any data."

        # Ranks of the same pp/tp/cp group will have the same dp rank and thus share the same group id.
        # zhao: Cosmos3 does not support TP/SP/CP
        if self.shard_world_size is not None:
            train_world_size = self.shard_world_size
            train_rank = self.shard_rank
            log.info(f"Using shard_world_size: {train_world_size} and shard_rank: {train_rank}", rank0_only=False)
        else:
            train_world_size = torch.distributed.get_world_size()
            train_rank = torch.distributed.get_rank()
        train_dp_rank = train_rank
        train_num_dp_groups = train_world_size
        train_dp_group_size = 1

        # Get data worker rank. Each trainer have multiple dataloaders
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_rank = worker_info.id
            total_data_ranks = worker_info.num_workers * train_num_dp_groups
            data_rank = worker_rank + train_dp_rank * worker_info.num_workers
            seed = worker_info.seed
        else:
            log.warning("No data worker info found. Using default worker rank and number of workers.", rank0_only=False)
            total_data_ranks = train_num_dp_groups
            data_rank = train_dp_rank
            seed = 42

        log.info(
            f"train_world_size: {train_world_size}; "
            f"train_rank: {train_rank}; "
            f"train_dp_rank: {train_dp_rank}; "
            f"train_num_dp_groups: {train_num_dp_groups}; "
            f"train_dp_group_size: {train_dp_group_size}; "
            f"worker_info: {worker_info}; "
            f"total_data_ranks: {total_data_ranks}; "
            f"data_rank: {data_rank}; "
            f"seed: {seed}"
            f"shard_id: {self.shard_id}; "
            f"shard_world_size: {self.shard_world_size}; "
            f"shard_rank: {self.shard_rank}",
            rank0_only=False,
        )

        # Make sure len(self.episode_index) is divisible by self.num_groups
        multiplier = max(1, total_data_ranks * 50 // len(self.episode_index))
        log.info(f"Dataset multiplier: {multiplier}", rank0_only=False)
        self.episode_index = self.episode_index * multiplier  # reduce bias caused by sharding
        num_pad = total_data_ranks - len(self.episode_index) % total_data_ranks
        self.episode_index = self.episode_index + self.episode_index[:num_pad]
        # Deterministic shuffle + split list to keep only the data for this rank
        random.Random(self.shard_id).shuffle(self.episode_index)
        log.info(f"Shuffled episode index for shard {self.shard_id}", rank0_only=False)
        self.episode_index = self.episode_index[data_rank::total_data_ranks]
        log.info(
            f"DRank {data_rank} has {len(self.episode_index)} episodes.",
            rank0_only=False,
        )

        self.is_initialized = True

        # Make sure the data within a DRank is identical
        rng = random.Random(data_rank + self.shard_id * 12345)
        while True:
            rng.shuffle(self.episode_index)
            for ds_idx, ep_idx, clip_idx in self.episode_index:
                sample = self.process_one_sample(ds_idx, ep_idx, clip_idx)
                if sample is None:
                    log.warning(f"Failed to process sample (ds={ds_idx}, ep={ep_idx}, clip={clip_idx}), skipping...")
                    continue
                yield sample


# ============================================================================
# 3. 入口函数
# ============================================================================


def get_sft_dataset_from_lerobot(
    dataset_path: str,
    resolution: str = "720",
    use_multi_resolution: bool = False,  # 多分辨率训练开关：True 时在 256/480 随机（不上采样）
    use_multi_fps: bool = False,  # 多 fps 训练开关：True 时 temporal_interval 在 [2,3,4] 随机
    tokenizer_config: Optional[Any] = None,
    cfg_dropout_rate: float = 0.1,
    use_system_prompt: bool = False,
    max_caption_tokens: int = _MAX_CAPTION_TOKENS,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    cfg_dropout_keep_metadata: bool = False,
    min_frames: int = 61,
    max_duration_s: float = 61.0,
    long_video_policy: str = "drop",
    video_window_overlap_s: float = 0.0,
    min_short_edge: int = 0,
    caption_suffix: str = "",
    conditioning_fps: float = 24,
    conditioning_fps_noise_std: float = 0.0,
    conditioning_config: dict[int, float] | None = None,
    temporal_compression_factor: int = 4,
    video_feature_key: str | None = None,
    caption_key: str = "caption",
    video_feature_keywords: list[str] | None = None,
    video_backend: str = "pyav",
    video_tolerance_s: float = 0.034,
    max_video_fps: float = 30.0,
    decoder_cache_max_size: int = _vu.LRU_VIDEO_CACHE_MAX_SIZE,
    **kwargs,
) -> LeRobotSFTDataset:
    """LeRobot 版 get_sft_dataset，动态加载 LeRobot 数据集。

    ``dataset_path`` 是统一入口，按类型自动分流：
    - ``.jsonl`` 文件 → manifest 模式（每行一个 ``{"path": ...}``，加载所有 path 的数据）
    - 目录 → 单数据集根 / 父目录（递归发现其下所有 meta/info.json）

    与 ``sft_dataset.get_sft_dataset`` 的差异：
    1. 签名：``dataset_path`` + ``video_feature_key``/``video_feature_keywords`` 替代 ``jsonl_paths``
    2. metadata 来源：``_load_lerobot_metadata(_from_manifest)`` 替代 ``_load_sft_metadata_from_s3``
    3. 构造类：独立 ``LeRobotSFTDataset``（不继承 SFTDataset），本地解码、无需 S3 凭证
    4. 惰性化加载：加载期只产出 (sources, episode_index) 扁平索引，episode 字段采样期现算
    """
    log.info(f"Unknown kwargs for get_sft_dataset_from_lerobot: {kwargs}")
    assert resolution in VIDEO_RES_SIZE_INFO.keys(), "The provided resolution cannot be found in VIDEO_RES_SIZE_INFO."

    # 加载 metadata 前就确保 HF 离线，避免 LeRobotDatasetMetadata 触发 HF Hub 联网
    _ensure_hf_hub_offline()

    if dataset_path.endswith(".jsonl"):
        sources, episode_index = _load_lerobot_metadata_from_manifest(
            dataset_path,
            min_frames=min_frames,
            max_duration_s=max_duration_s,
            min_short_edge=min_short_edge,
            video_feature_key=video_feature_key,
            caption_key=caption_key,
            video_feature_keywords=video_feature_keywords,
            long_video_policy=long_video_policy,
            video_window_overlap_s=video_window_overlap_s,
        )
        source = f"manifest {dataset_path}"
    else:
        sources, episode_index = _load_lerobot_metadata(
            dataset_path,
            min_frames=min_frames,
            max_duration_s=max_duration_s,
            min_short_edge=min_short_edge,
            video_feature_key=video_feature_key,
            caption_key=caption_key,
            video_feature_keywords=video_feature_keywords,
            long_video_policy=long_video_policy,
            video_window_overlap_s=video_window_overlap_s,
        )
        source = dataset_path

    log.info(
        f"Finished loading LeRobot metadata from {source}. "
        f"Total datasets: {len(sources)}, total clips: {len(episode_index)}"
    )

    dataset = LeRobotSFTDataset(
        sources=sources,
        episode_index=episode_index,
        resolution=resolution,
        use_multi_resolution=use_multi_resolution,
        use_multi_fps=use_multi_fps,
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
        video_tolerance_s=video_tolerance_s,
        max_video_fps=max_video_fps,
        decoder_cache_max_size=decoder_cache_max_size,
    )
    return dataset
