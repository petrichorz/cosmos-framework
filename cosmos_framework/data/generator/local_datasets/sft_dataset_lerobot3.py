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
#   - metadata 与视频加载都基于 lerobot 官方 package（参考 action 侧 cosmos3_action_lerobot）：
#     LeRobotDatasetMetadata 读 info/episodes（自动 drop stats 列、保留 caption），
#     decode_video_frames 按时间戳解码视频（torchcodec + LRU decoder cache）。
#   - LeRobotSFTDataset override process_one_sample，仅替换「中段视频加载」，
#     保留多分辨率/多 fps 的扩展。
import hashlib
import json
import os
import random
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import huggingface_hub.constants as _hf_const
import numpy as np
import torch
from lerobot.datasets import video_utils as _vu
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from cosmos_framework.data.generator.local_datasets.helper import (
    get_aspect_ratio,
)
from cosmos_framework.data.generator.local_datasets.sft_dataset import (
    SFTDataset,
    _DURATION_TEMPLATE,
    _MAX_CAPTION_TOKENS,
    _RESOLUTION_TEMPLATE,
    _flatten_metadata_by_window,
    _select_caption,
)
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.data.generator.utils import VIDEO_RES_SIZE_INFO
from cosmos_framework.utils import log
from cosmos_framework.utils.flags import INTERNAL

# 多分辨率训练：候选档位（短边像素），只选 <= 视频短边的档位（不上采样）。
# _MULTI_RESOLUTION_TIERS = ("256", "480", "720")
_MULTI_RESOLUTION_TIERS = ("256", "480")
# 多 fps 训练：候选 temporal_interval（保留 1/2、1/3、1/4）。
_MULTI_FPS_INTERVALS = (2, 3, 4)

# lerobot 解码器 LRU 缓存容量（替换 lerobot 自带的「无界 dict」缓存，防止 worker 内存无界增长）。
_LRU_VIDEO_CACHE_MAX_SIZE = 64

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


class _LRUVideoDecoderCache:
    """LRU 版 torchcodec decoder 缓存，替换 lerobot 的 ``video_utils.VideoDecoderCache``。

    与 action 侧 ``cosmos3_action_lerobot._LRUVideoDecoderCache`` 的唯一差异：
    ``seek_mode="exact"``（vision SFT 需精确切 episode 帧边界，action 用 approximate）。

    lerobot 自带的 ``VideoDecoderCache`` 是「无界 dict、只加不删」，多 worker 场景下
    decoder 索引 + FFmpeg 上下文会持续累积导致内存上涨；这里用 LRU 封顶。
    """

    def __init__(self, max_size: int = _LRU_VIDEO_CACHE_MAX_SIZE) -> None:
        self._max_size = max_size
        self._cache: "OrderedDict[str, tuple[Any, Any]]" = OrderedDict()
        self._lock = Lock()

    def get_decoder(self, video_path: str) -> Any:
        import importlib.util

        if importlib.util.find_spec("torchcodec"):
            from torchcodec.decoders import VideoDecoder
        else:
            raise ImportError("torchcodec is required but not available.")

        import fsspec

        video_path = str(video_path)
        with self._lock:
            if video_path in self._cache:
                self._cache.move_to_end(video_path)
                return self._cache[video_path][0]

            file_handle = fsspec.open(video_path).__enter__()
            try:
                decoder = VideoDecoder(file_handle, seek_mode="exact")
            except Exception:
                # 构造失败时，已打开的文件句柄必须显式关闭，否则坏文件会累积 fd。
                try:
                    file_handle.close()
                except Exception:
                    pass
                raise
            self._cache[video_path] = (decoder, file_handle)

            while len(self._cache) > self._max_size:
                _, (old_decoder, old_fh) = self._cache.popitem(last=False)
                # torchcodec VideoDecoder 无 close API，靠 del 触发 C++ 引用计数释放。
                del old_decoder
                try:
                    old_fh.close()
                except Exception:
                    pass
            return decoder

    def clear(self) -> None:
        with self._lock:
            for _, file_handle in self._cache.values():
                try:
                    file_handle.close()
                except Exception:
                    pass
            self._cache.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._cache)


def _patch_decoder_cache(max_size: int = _LRU_VIDEO_CACHE_MAX_SIZE) -> None:
    """把 lerobot 模块级 ``_default_decoder_cache`` 替换为 LRU 版，防止无界内存增长。

    幂等，每个进程只 patch 一次。参考 action 侧 ``cosmos3_action_lerobot._patch_decoder_cache``。
    """
    global _decoder_cache_patched
    if _decoder_cache_patched:
        return
    _vu._default_decoder_cache = _LRUVideoDecoderCache(max_size=max_size)
    _decoder_cache_patched = True


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


def _load_single_lerobot_metadata(
    lerobot_root: str,
    min_frames: int,
    min_short_edge: int,
    video_feature_key: str | None,
    caption_key: str,
    video_feature_keywords: list[str] | None = None,
) -> list[dict]:
    """读【单个】LeRobot 数据集，产出 metadata list（基于 lerobot 官方 package）。

    被 ``_load_lerobot_metadata`` 调用（后者负责发现多个数据集根并逐个加载合并）。
    输出的 metadata dict 结构对齐 ``sft_dataset._load_sft_metadata_from_s3``：
    {uuid, vision_path, width, height, nb_frames, framerate, aspect_ratio, t2w_windows}，
    额外带上视频解码所需的定位字段 {lerobot_root, episode_index, video_key, from_timestamp}。
    """
    root = Path(lerobot_root)
    # repo_id="local" + revision="local"：本地数据集，避开 HF Hub 联网（"local" 不是合法 version）
    meta = LeRobotDatasetMetadata(repo_id="local", root=str(root), revision="local")
    fps = float(meta.fps)

    video_key = _select_lerobot_video_key(meta.info, video_feature_key, video_feature_keywords)
    width, height = _get_lerobot_video_width_height(meta.info, video_key)

    root_hash = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:8]

    metadata_list: list[dict] = []
    # meta.episodes 是 HF Dataset（pyarrow 内存映射，load_episodes 已自动 drop 掉 stats/ 列）
    for ep_pos, ep in enumerate(meta.episodes):
        episode_index = int(ep["episode_index"])

        # 定位该 episode 的数据文件（跨文件唯一）
        data_chunk = int(ep.get("data/chunk_index", 0))
        data_file = int(ep.get("data/file_index", 0))

        # 该 episode 视频在选定 camera 的 mp4 里的时间区间
        from_ts = float(ep.get(f"videos/{video_key}/from_timestamp", 0.0))
        to_ts = float(ep.get(f"videos/{video_key}/to_timestamp", 0.0))

        # 时间区间 → 帧编号（to 是开区间，end 要 -1）
        start_frame = round(from_ts * fps)
        end_frame = round(to_ts * fps) - 1

        # caption：只读 episodes 表新增的 caption 列（列名由 caption_key 指定），取不到为 None。
        caption = ep.get(caption_key)

        length = int(ep.get("length", end_frame - start_frame + 1))
        duration = to_ts - from_ts

        # 过滤（对齐 sft_dataset._load_sft_metadata_from_s3）
        if duration > 61.0:
            continue
        if min_short_edge > 0 and min(width, height) < min_short_edge:
            continue
        frames_in_window = end_frame - start_frame + 1
        if frames_in_window < min_frames:
            continue

        # uuid 前缀用「目录名 + 完整路径短 hash」，保证不同父目录下的同名数据集（如两个 so100_battery）不冲突
        uuid = f"{root.name}_{root_hash}_chunk_{data_chunk}_file_{data_file}_episode_{episode_index}"

        # 用 lerobot 原生接口拼本地 mp4 路径
        vision_path = str(root / meta.get_video_file_path(ep_pos, video_key))

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
                "total_frames": int(meta.total_frames),
                "t2w_windows": [window],
                # 视频解码定位字段（lerobot decode_video_frames 需要绝对时间戳）
                "lerobot_root": str(root),
                "episode_index": episode_index,
                "video_key": video_key,
                "from_timestamp": from_ts,
            }
        )

    return metadata_list


def _load_lerobot_metadata(
    lerobot_root: str,
    min_frames: int = 61,
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
                min_short_edge=min_short_edge,
                video_feature_key=video_feature_key,
                caption_key=caption_key,
                video_feature_keywords=video_feature_keywords,
            )
        )
    return metadata_list


def _load_lerobot_metadata_from_manifest(
    manifest_path: str,
    min_frames: int = 61,
    min_short_edge: int = 0,
    video_feature_key: str | None = None,
    caption_key: str = "caption",
    video_feature_keywords: list[str] | None = None,
    manifest_max_workers: int | None = None,
) -> list[dict]:
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
        return []

    def _load_one(task):
        path, fk, fkw, ck = task
        return _load_lerobot_metadata(
            path,
            min_frames=min_frames,
            min_short_edge=min_short_edge,
            video_feature_key=fk,
            caption_key=ck,
            video_feature_keywords=fkw,
        )

    if manifest_max_workers is None:
        manifest_max_workers = min(len(tasks), 8)

    metadata_list: list[dict] = []
    if manifest_max_workers <= 1 or len(tasks) == 1:
        # 单线程：保持原有顺序，无并发开销
        for task in tasks:
            metadata_list.extend(_load_one(task))
    else:
        from concurrent.futures import ThreadPoolExecutor

        log.info(f"[manifest] 并行加载 {len(tasks)} 个数据集，max_workers={manifest_max_workers}")
        with ThreadPoolExecutor(max_workers=manifest_max_workers) as ex:
            # ex.map 保持输入顺序返回，结果顺序与 manifest 行顺序一致
            for result in ex.map(_load_one, tasks):
                metadata_list.extend(result)

    return metadata_list


# ============================================================================
# 2. LeRobotSFTDataset 子类
# ============================================================================


class LeRobotSFTDataset(SFTDataset):
    """SFTDataset 子类，override process_one_sample 为「本地 mp4 + 按帧编号区间 seek」。

    与父类的唯一差异：``process_one_sample`` 里视频加载那一段。
    """

    def __init__(
        self,
        *args,
        use_multi_resolution: bool = False,
        use_multi_fps: bool = False,
        video_backend: str | None = None,
        tolerance_s: float = 0.034,
        decoder_cache_max_size: int = _LRU_VIDEO_CACHE_MAX_SIZE,
        **kwargs,
    ):
        _ensure_hf_hub_offline()
        _patch_decoder_cache(max_size=decoder_cache_max_size)
        super().__init__(*args, **kwargs)
        self.use_multi_resolution = use_multi_resolution
        self.use_multi_fps = use_multi_fps
        # lerobot 默认 codec：torchcodec 可用则用 torchcodec，否则 pyav
        self.video_backend = video_backend if video_backend else _vu.get_safe_default_codec()
        self.tolerance_s = tolerance_s

    def process_one_sample(self, metadata: dict) -> dict | None:
        """Process a single LeRobot SFT sample.

        ⚠️ 与 sft_dataset.py:SFTDataset.process_one_sample 保持同步。视频加载与父类完全一致
        （read_bytes → 临时文件 → ffmpeg 全量解码 + 按帧区间过滤）；仅在此基础上扩展了
        多分辨率（use_multi_resolution）与多 fps（use_multi_fps）。
        """
        windows = metadata["t2w_windows"]
        win_idx = random.randrange(len(windows))
        t2w_window = windows[win_idx]
        window_start = t2w_window["start_frame"]
        window_end = t2w_window["end_frame"]

        # Compute output resolution
        input_w, input_h = metadata["width"], metadata["height"]
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
        target_w, target_h = output_sizes[metadata["aspect_ratio"]]
        resize_ratio = max(target_w / input_w, target_h / input_h)
        resize_h, resize_w = (round(input_h * resize_ratio), round(input_w * resize_ratio))
        crop_y, crop_x = (round((resize_h - target_h) / 2), round((resize_w - target_w) / 2))

        # 【lerobot 加载】本地 mp4，用 lerobot decode_video_frames 按时间戳解码（复用 LRU decoder cache）
        input_video_path = metadata["vision_path"]
        original_fps = metadata["framerate"]
        total_frames = metadata["total_frames"]

        # Constrain to the t2w window
        actual_end = min(window_end, total_frames - 1)
        frames_in_window = actual_end - window_start + 1

        if self.num_video_frames == -1:
            # Native chunk mode: use start/end/interval directly from the window
            if self.use_multi_fps:
                # 多 fps：temporal_interval 在 [2,3,4] 随机（保留 1/2、1/3、1/4）
                temporal_interval = random.choice(_MULTI_FPS_INTERVALS)
            else:
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
                temporal_interval = max(1, int(original_fps / 30.0))
            elif self.temporal_interval_mode == "entire_chunk":
                temporal_interval = frames_in_window // self.num_video_frames
                temporal_interval = max(1, temporal_interval)
            else:
                raise ValueError(f"Unknown temporal_interval_mode: {self.temporal_interval_mode}")

            num_frames_before_downsample = (self.num_video_frames - 1) * temporal_interval + 1

            # 【防御】抽帧跨度超过窗口时，end_frame 会越界（读到相邻 episode 的帧）。
            # force_one/max_30fps（或将来加的 max_15fps）在 interval>1 时可能触发；
            # entire_chunk 因 interval 由 frames_in_window//N 推导，几乎不会触发，此检查对其无害。
            # 注意：JSONL 版 sft_dataset.py 的 process_one_sample 目前【未加】此防御，仍可能有同样问题。
            if num_frames_before_downsample > frames_in_window:
                log.warning(
                    f"Window too short for interval={temporal_interval}: {metadata['uuid']}, "
                    f"frames_in_window={frames_in_window}, required span={num_frames_before_downsample}"
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

        # 【lerobot 加载】帧号 → 绝对时间戳 → decode_video_frames（返回 [T,C,H,W] float ∈ [0,1]）
        frame_indices = list(range(start_frame, end_frame + 1, temporal_interval))
        timestamps = [idx / original_fps for idx in frame_indices]
        try:
            video_frames = _vu.decode_video_frames(
                input_video_path,
                timestamps,
                tolerance_s=self.tolerance_s,
                backend=self.video_backend,
            )  # [T, C, H, W] float32 ∈ [0,1]
        except AssertionError as e:
            # 时间戳与视频 pts 偏差超过 tolerance_s 时，lerobot 内部 assert 会抛 AssertionError。
            # 打印其详细提示（哪些时间戳违反 tolerance、视频路径等），并跳过该样本，避免中断训练。
            log.warning(
                f"AssertionError decoding video for sample {metadata['uuid']} "
                f"(start={start_frame}, end={end_frame}, path={metadata['vision_path']}): {e}"
            )
            return None
        except Exception as e:
            # 其它解码失败（坏文件、解码器异常等），同样跳过该样本。
            log.warning(
                f"Failed to decode video for sample {metadata['uuid']} "
                f"(start={start_frame}, end={end_frame}, path={metadata['vision_path']}): "
                f"{type(e).__name__}: {e}"
            )
            return None

        if video_frames.shape[0] == 0:
            log.warning(
                f"No frames decoded for sample: {metadata['uuid']} "
                f"(start={start_frame}, end={end_frame}, path={metadata['vision_path']})"
            )
            return None

        # resize 到 (resize_h, resize_w)（对齐原 ffmpeg 的 scale + bicubic），再转 [T,H,W,C] uint8
        import torch.nn.functional as F

        video_frames = video_frames.float()
        video_frames = F.interpolate(video_frames, size=(resize_h, resize_w), mode="bicubic", align_corners=False)
        video_frames = video_frames.round().clamp(0, 255).to(torch.uint8)
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
    dataset_path: str,
    resolution: str = "720",
    use_multi_resolution: bool = False,  # 多分辨率训练开关：True 时在 256/480/720 随机（不上采样）
    num_video_frames: int = -1,  # LeRobot 场景默认 -1（native chunk mode，直接用 t2w_windows 里的帧区间）
    use_multi_fps: bool = False,  # 多 fps 训练开关：True 时 temporal_interval 在 [2,3,4] 随机
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
    video_backend: str | None = None,
    tolerance_s: float = 0.034,
    decoder_cache_max_size: int = _LRU_VIDEO_CACHE_MAX_SIZE,
    **kwargs,
) -> LeRobotSFTDataset:
    """LeRobot 版 get_sft_dataset，动态加载 LeRobot 数据集。

    ``dataset_path`` 是统一入口，按类型自动分流：
    - ``.jsonl`` 文件 → manifest 模式（每行一个 ``{"path": ...}``，加载所有 path 的数据）
    - 目录 → 单数据集根 / 父目录（递归发现其下所有 meta/info.json）

    与 ``sft_dataset.get_sft_dataset`` 的差异仅 3 处：
    1. 签名：``dataset_path`` + ``video_feature_key``/``video_feature_keywords`` 替代 ``jsonl_paths``
    2. metadata 来源：``_load_lerobot_metadata(_from_manifest)`` 替代 ``_load_sft_metadata_from_s3``
    3. 构造类：``LeRobotSFTDataset`` 替代 ``SFTDataset``（后者 override 视频加载为按帧编号 seek）

    其余参数、flatten/shuffle、构造参数列表均与 ``get_sft_dataset`` 一致。
    """
    log.info(f"Unknown kwargs for get_sft_dataset_from_lerobot: {kwargs}")
    assert resolution in VIDEO_RES_SIZE_INFO.keys(), "The provided resolution cannot be found in VIDEO_RES_SIZE_INFO."

    # 加载 metadata 前就确保 HF 离线，避免 LeRobotDatasetMetadata 触发 HF Hub 联网
    _ensure_hf_hub_offline()

    # LeRobot 是本地加载，不需要 S3 下载凭证（SFTDataset 构造仍要求 s3_credentials 参数）
    if INTERNAL:
        with open("credentials/gcs.secret", "r") as f:
            credentials = json.load(f)
    else:
        credentials = {}

    if dataset_path.endswith(".jsonl"):
        metadata_list = _load_lerobot_metadata_from_manifest(
            dataset_path,
            min_frames=61,
            min_short_edge=min_short_edge,
            video_feature_key=video_feature_key,
            caption_key=caption_key,
            video_feature_keywords=video_feature_keywords,
        )
        source = f"manifest {dataset_path}"
    else:
        metadata_list = _load_lerobot_metadata(
            dataset_path,
            min_frames=61,
            min_short_edge=min_short_edge,
            video_feature_key=video_feature_key,
            caption_key=caption_key,
            video_feature_keywords=video_feature_keywords,
        )
        source = dataset_path

    total_windows = sum(len(m["t2w_windows"]) for m in metadata_list)
    log.info(
        f"Finished loading LeRobot metadata from {source}. "
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
        use_multi_resolution=use_multi_resolution,
        use_multi_fps=use_multi_fps,
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
        tolerance_s=tolerance_s,
        decoder_cache_max_size=decoder_cache_max_size,
    )
    return dataset
