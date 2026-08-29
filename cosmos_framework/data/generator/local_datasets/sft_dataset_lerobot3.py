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
#     本类 = 本地 mp4 直接 get_video_metadata + torchcodec 按帧编号区间 seek
import hashlib
import json
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

        # caption：只读 episodes 表新增的 caption 列（列名由 caption_key 指定），取不到为 None。
        caption = row.get(caption_key)

        length = int(row.get("length", end_frame - start_frame + 1))
        duration = to_ts - from_ts

        # 过滤（对齐 sft_dataset._load_sft_metadata_from_s3）
        if duration > 61.0:
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


# ============================================================================
# 2. torchcodec decoder LRU 缓存 + LeRobotSFTDataset 子类
# ============================================================================


class _LeRobotVideoDecoderCache:
    """按视频路径缓存 torchcodec VideoDecoder 的 LRU 缓存。

    仿照 action 侧 ``cosmos3_action_lerobot._LRUVideoDecoderCache``，但：
    - ``seek_mode="exact"``（精确帧定位，vision SFT 需切准 episode 帧边界）
    - torchcodec/fsspec 惰性 import，不污染主模块 import 路径
    """

    def __init__(self, max_size: int = 64):
        from collections import OrderedDict

        self._max_size = max_size
        self._cache: "OrderedDict[str, tuple]" = OrderedDict()

    def get_decoder(self, video_path: str):
        from torchcodec.decoders import VideoDecoder
        import fsspec

        if video_path in self._cache:
            self._cache.move_to_end(video_path)
            return self._cache[video_path][0]

        file_handle = fsspec.open(video_path).__enter__()
        decoder = VideoDecoder(file_handle, seek_mode="exact")
        self._cache[video_path] = (decoder, file_handle)

        while len(self._cache) > self._max_size:
            _, (_, old_fh) = self._cache.popitem(last=False)
            try:
                old_fh.close()
            except Exception:
                pass
        return decoder


class LeRobotSFTDataset(SFTDataset):
    """SFTDataset 子类，override process_one_sample 为「本地 mp4 + 按帧编号区间 seek」。

    与父类的唯一差异：``process_one_sample`` 里视频加载那一段。
    """

    def __init__(self, *args, decoder_cache_max_size: int = 64, **kwargs):
        super().__init__(*args, **kwargs)
        self._decoder_cache = _LeRobotVideoDecoderCache(max_size=decoder_cache_max_size)

    def _decode_video_frames(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        temporal_interval: int,
        resize_h: int,
        resize_w: int,
    ) -> list[np.ndarray]:
        """按帧编号区间 seek，只解码目标 episode 的帧。

        与父类对齐的返回格式：list[np.ndarray]（每帧 HWC uint8），供下游 np.stack。
        """
        import torch.nn.functional as F

        decoder = self._decoder_cache.get_decoder(video_path)

        # torchcodec：开区间 [start, stop)，所以 stop = end_frame + 1
        frame_batch = decoder.get_frames_in_range(start=start_frame, stop=end_frame + 1)
        # frame_batch.data: [N, C, H, W] uint8（dimension_order="NCHW" 默认）

        data = frame_batch.data  # [N, C, H, W] uint8

        # 抽帧：原版语义为「相对 start_frame 取余 interval == 0」，等价于 step=interval
        if temporal_interval > 1:
            data = data[0::temporal_interval]

        # resize：原版 ffmpeg 用 -vf scale + bicubic 在解码时 resize 到 (resize_h, resize_w)。
        # 这里用 torch interpolate(bicubic) 对齐，保证后续 center crop 尺寸正确。
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
        本类 = 本地 mp4 直接 get_video_metadata + torchcodec 按帧编号区间 seek。
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
        video_info = get_video_metadata(input_video_path)
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
                temporal_interval = max(1, int(original_fps / 30.0))
            elif self.temporal_interval_mode == "entire_chunk":
                temporal_interval = frames_in_window // self.num_video_frames
                temporal_interval = max(1, temporal_interval)
            else:
                raise ValueError(f"Unknown temporal_interval_mode: {self.temporal_interval_mode}")

            num_frames_before_downsample = (self.num_video_frames - 1) * temporal_interval + 1
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

        # 【LeRobot 差异】torchcodec 区间 seek 替代父类全量 ffmpeg decode
        video_chunk = self._decode_video_frames(
            video_path=input_video_path,
            start_frame=start_frame,
            end_frame=end_frame,
            temporal_interval=temporal_interval,
            resize_h=resize_h,
            resize_w=resize_w,
        )

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

    # LeRobot 是本地加载，不需要 S3 下载凭证（SFTDataset 构造仍要求 s3_credentials 参数）
    if INTERNAL:
        with open("credentials/gcs.secret", "r") as f:
            credentials = json.load(f)
    else:
        credentials = {}

    metadata_list = _load_lerobot_metadata(
        lerobot_root,
        min_frames=61,
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
        decoder_cache_max_size=decoder_cache_max_size,
    )
    return dataset
