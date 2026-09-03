# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""LeRobot v3 full-episode adapter for causal vision SFT.

The adapter builds a lightweight episode catalog up front and decodes one
complete episode per sample. Action columns are deliberately not loaded: this
is a vision-only dataset that preserves the regular :class:`SFTDataset` output
contract.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import torch

from cosmos_framework.data.generator.local_datasets.helper import get_aspect_ratio
from cosmos_framework.data.generator.local_datasets.sft_dataset import _MAX_CAPTION_TOKENS, SFTDataset
from cosmos_framework.data.generator.utils import VIDEO_RES_SIZE_INFO
from cosmos_framework.utils import log


def _is_lerobot_root(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file()


def _discover_lerobot_roots(roots: str | os.PathLike[str] | list[str | os.PathLike[str]]) -> list[Path]:
    """Resolve dataset roots with root/direct-child/recursive-fallback discovery."""
    candidates = [roots] if isinstance(roots, (str, os.PathLike)) else roots
    if not candidates:
        raise ValueError("At least one LeRobot root or parent directory is required.")

    discovered: list[Path] = []
    for candidate in candidates:
        parent = Path(candidate).expanduser().resolve()
        if not parent.is_dir():
            raise FileNotFoundError(f"LeRobot input directory does not exist: {parent}")
        if _is_lerobot_root(parent):
            discovered.append(parent)
            continue

        direct = [path for path in sorted(parent.iterdir()) if path.is_dir() and _is_lerobot_root(path)]
        if direct:
            discovered.extend(direct)
            continue

        discovered.extend(path.parent.parent for path in sorted(parent.rglob("meta/info.json")))

    unique = sorted(set(discovered), key=lambda path: path.as_posix())
    if not unique:
        raise FileNotFoundError(f"No LeRobot dataset containing meta/info.json was found under {candidates!r}.")
    return unique


def _select_video_key(
    features: dict[str, dict[str, Any]],
    video_view: str,
    video_view_aliases: list[str] | None,
) -> str:
    video_keys = sorted(key for key, feature in features.items() if feature.get("dtype") == "video")
    if not video_keys:
        raise ValueError("LeRobot dataset has no dtype=video feature.")

    requested = video_view.casefold()
    for key in video_keys:
        if key.casefold() == requested:
            return key
    for key in video_keys:
        if key.rsplit(".", 1)[-1].casefold() == requested:
            return key

    aliases = [video_view, *(video_view_aliases or [])]
    for alias in aliases:
        alias = alias.strip().casefold()
        if not alias:
            continue
        matches = [key for key in video_keys if alias in key.casefold()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Video view alias {alias!r} is ambiguous; matching fields: {matches}")

    raise ValueError(f"Cannot resolve video_view={video_view!r}; available video fields: {video_keys}")


def _video_width_height(feature: dict[str, Any]) -> tuple[int, int]:
    shape = list(feature.get("shape") or [])
    if len(shape) < 2:
        raise ValueError(f"Video feature has invalid shape: {shape!r}")
    names = feature.get("names")
    if names and "width" in names and "height" in names:
        return int(shape[names.index("width")]), int(shape[names.index("height")])
    return int(shape[1]), int(shape[0])


def _video_path(root: Path, info: dict[str, Any], episode: dict[str, Any], video_key: str) -> Path:
    """Resolve a LeRobot v2/v3 video path.

    Kept locally by design. This compatibility logic is copied from
    ``action/datasets/base_dataset.py::_video_path`` rather than coupling the
    vision-only adapter to the action package.
    """
    chunk_idx = int(
        episode.get(
            f"videos/{video_key}/chunk_index",
            episode.get(f"videos/{video_key}/episode_chunk", episode.get("data/chunk_index", 0)),
        )
    )
    file_idx = int(
        episode.get(
            f"videos/{video_key}/file_index",
            episode.get(f"videos/{video_key}/episode_file", episode.get("data/file_index", 0)),
        )
    )
    relative_path = info["video_path"].format(
        video_key=video_key,
        chunk_index=chunk_idx,
        file_index=file_idx,
        episode_chunk=chunk_idx,
        episode_file=file_idx,
    )
    return root / relative_path


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text or None


def _episode_captions(
    episode: dict[str, Any], caption_key: str | None, task_by_index: dict[int, str]
) -> list[tuple[str, str]]:
    """Return caption candidates, preferring a configured episode field."""
    if caption_key:
        caption = _clean_text(episode.get(caption_key))
        if caption is not None:
            return [(caption_key, caption)]

    tasks = episode.get("tasks")
    if isinstance(tasks, str):
        tasks = [tasks]
    if tasks is not None:
        captions = [text for value in tasks if (text := _clean_text(value)) is not None]
        if captions:
            return [("task", text) for text in dict.fromkeys(captions)]

    task_indices = episode.get("task_indices", episode.get("task_index"))
    if task_indices is not None:
        if not isinstance(task_indices, (list, tuple)):
            task_indices = [task_indices]
        captions = [task_by_index[int(index)] for index in task_indices if int(index) in task_by_index]
        if captions:
            return [("task", text) for text in dict.fromkeys(captions)]

    for key in ("task", "subtask"):
        caption = _clean_text(episode.get(key))
        if caption is not None:
            return [(key, caption)]
    return []


def _load_lerobot_metadata(root: Path):
    # Import lazily so importing the regular JSONL SFT dataset does not require
    # the optional training dependency. No frame/action dataset is constructed.
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    required = [root / "meta" / "info.json", root / "meta" / "tasks.parquet", root / "meta" / "episodes"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Invalid LeRobot root {root}: missing {missing}")
    return LeRobotDatasetMetadata(repo_id="local", root=root, revision="local")


def _dataset_labels(roots: list[Path]) -> dict[Path, str]:
    if len(roots) == 1:
        return {roots[0]: roots[0].name}
    common = Path(os.path.commonpath([str(root) for root in roots]))
    return {root: root.relative_to(common).as_posix() for root in roots}


def _build_episode_catalog(
    roots: str | os.PathLike[str] | list[str | os.PathLike[str]],
    *,
    metadata_load_workers: int,
    video_view: str,
    video_view_aliases: list[str] | None,
    caption_key: str | None,
    min_short_edge: int,
) -> list[dict[str, Any]]:
    dataset_roots = _discover_lerobot_roots(roots)
    labels = _dataset_labels(dataset_roots)
    workers = max(1, min(metadata_load_workers, len(dataset_roots)))
    if metadata_load_workers < 1:
        raise ValueError(f"metadata_load_workers must be >= 1, got {metadata_load_workers}")

    def load(root: Path) -> tuple[Path, Any | None, Exception | None]:
        try:
            return root, _load_lerobot_metadata(root), None
        except Exception as error:  # aggregate initialization failures across roots
            return root, None, error

    if workers == 1:
        loaded = [load(root) for root in dataset_roots]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            loaded = list(executor.map(load, dataset_roots))

    failures = [(root, error) for root, _, error in loaded if error is not None]
    if failures:
        detail = "\n".join(f"  - {root}: {error}" for root, error in failures)
        raise RuntimeError(f"Failed to load {len(failures)} LeRobot dataset(s):\n{detail}")

    catalog: list[dict[str, Any]] = []
    schema_errors: list[str] = []
    missing_videos: set[Path] = set()
    no_caption_count = 0
    for root, metadata, _ in loaded:
        assert metadata is not None
        try:
            video_key = _select_video_key(metadata.features, video_view, video_view_aliases)
            width, height = _video_width_height(metadata.features[video_key])
        except Exception as error:
            schema_errors.append(f"{root}: {error}")
            continue
        if min_short_edge > 0 and min(width, height) < min_short_edge:
            log.warning(f"Skipping LeRobot dataset below min_short_edge={min_short_edge}: {root} ({width}x{height})")
            continue

        fps = float(metadata.fps)
        if fps <= 0:
            schema_errors.append(f"{root}: invalid fps={fps}")
            continue
        task_by_index = {
            int(row["task_index"]): str(row["task"] if "task" in row else task)
            for task, row in metadata.tasks.iterrows()
            if "task_index" in row
        }
        for row_index in range(len(metadata.episodes)):
            episode = dict(metadata.episodes[row_index])
            episode_index = int(episode.get("episode_index", row_index))
            captions = _episode_captions(episode, caption_key, task_by_index)
            if not captions:
                no_caption_count += 1
                continue

            path = _video_path(root, metadata.info, episode, video_key)
            if not path.is_file():
                missing_videos.add(path)
                continue

            length = int(episode.get("length", 0))
            from_timestamp = float(episode.get(f"videos/{video_key}/from_timestamp", 0.0))
            to_timestamp_value = episode.get(f"videos/{video_key}/to_timestamp")
            start_frame = round(from_timestamp * fps)
            if to_timestamp_value is not None:
                stop_frame = round(float(to_timestamp_value) * fps)
            elif length > 0:
                stop_frame = start_frame + length
            else:
                schema_errors.append(f"{root}: episode {episode_index} has neither length nor video timestamps")
                continue
            if stop_frame <= start_frame:
                schema_errors.append(
                    f"{root}: episode {episode_index} has invalid frame range [{start_frame}, {stop_frame})"
                )
                continue

            windows = [
                {
                    "start_frame": start_frame,
                    "end_frame": stop_frame - 1,
                    "temporal_interval": 1,
                    # Normalize LeRobot task/custom text to the canonical SFT
                    # caption key consumed by _select_caption().
                    "caption": caption,
                    "lerobot_caption_source": caption_type,
                }
                for caption_type, caption in captions
            ]
            dataset_label = labels[root]
            uuid_source = f"{dataset_label}/episode-{episode_index:06d}"
            catalog.append(
                {
                    "uuid": uuid_source,
                    "vision_path": str(path),
                    "width": width,
                    "height": height,
                    "nb_frames": length or stop_frame - start_frame,
                    "framerate": fps,
                    "aspect_ratio": get_aspect_ratio(width, height),
                    "t2w_windows": windows,
                    "lerobot_dataset": dataset_label,
                    "lerobot_episode_index": episode_index,
                }
            )

    if schema_errors or missing_videos:
        details = [*schema_errors[:20], *(f"missing video: {path}" for path in sorted(missing_videos)[:20])]
        omitted = len(schema_errors) + len(missing_videos) - len(details)
        suffix = f"\n  ... and {omitted} more" if omitted > 0 else ""
        raise ValueError("LeRobot catalog validation failed:\n  " + "\n  ".join(details) + suffix)
    if not catalog:
        raise ValueError("LeRobot catalog contains no trainable episodes with captions.")
    if no_caption_count:
        log.warning(f"Skipped {no_caption_count} LeRobot episode(s) without a caption/task.")

    # Match the deterministic ordering used by the canonical SFT factory.
    catalog.sort(key=lambda item: hashlib.sha256(item["uuid"].encode("utf-8")).hexdigest())
    log.info(
        f"Loaded {len(catalog)} full episodes from {len(dataset_roots)} LeRobot dataset(s) "
        f"using video_view={video_view!r}."
    )
    return catalog


class LeRobotEpisodeSFTDataset(SFTDataset):
    """Vision SFT dataset that decodes exactly one complete LeRobot episode."""

    def __init__(self, *args, max_video_fps: float | None = 15.0, num_ffmpeg_threads: int = 1, **kwargs):
        if max_video_fps is not None and max_video_fps <= 0:
            raise ValueError(f"max_video_fps must be positive or None, got {max_video_fps}")
        if num_ffmpeg_threads < 0:
            raise ValueError(f"num_ffmpeg_threads must be >= 0, got {num_ffmpeg_threads}")
        super().__init__(*args, **kwargs)
        self.max_video_fps = max_video_fps
        self.num_ffmpeg_threads = num_ffmpeg_threads
        self.num_samples_per_epoch = len(self.metadata)

    def _initialize_worker_resources(self) -> None:
        # Local TorchCodec decoding needs no S3 client or persistent decoder.
        return None

    @staticmethod
    def _resize_geometry(input_w: int, input_h: int, target_w: int, target_h: int) -> tuple[int, int]:
        resize_ratio = max(target_w / input_w, target_h / input_h)
        return round(input_h * resize_ratio), round(input_w * resize_ratio)

    def _decode_episode(
        self,
        *,
        video_path: str,
        start_frame: int,
        stop_frame: int,
        temporal_interval: int,
        resize_h: int,
        resize_w: int,
        target_h: int,
        target_w: int,
    ) -> tuple[torch.Tensor, int, int]:
        from torchcodec.decoders import VideoDecoder
        from torchcodec.transforms import CenterCrop, Resize

        decoder = VideoDecoder(
            video_path,
            seek_mode="exact",
            device="cpu",
            dimension_order="NCHW",
            num_ffmpeg_threads=self.num_ffmpeg_threads,
            transforms=[Resize((resize_h, resize_w)), CenterCrop((target_h, target_w))],
        )
        total_frames = len(decoder)
        actual_stop = min(stop_frame, total_frames)
        if start_frame >= actual_stop:
            del decoder
            return torch.empty((0, 3, target_h, target_w), dtype=torch.uint8), total_frames, actual_stop
        frame_batch = decoder.get_frames_in_range(start=start_frame, stop=actual_stop, step=temporal_interval)
        frames = frame_batch.data
        del frame_batch
        del decoder
        return frames, total_frames, actual_stop

    def process_one_sample(self, metadata: dict) -> dict | None:
        win_idx = random.randrange(len(metadata["t2w_windows"]))
        t2w_window = metadata["t2w_windows"][win_idx]
        start_frame = int(t2w_window["start_frame"])
        stop_frame = int(t2w_window["end_frame"]) + 1
        original_fps = float(metadata["framerate"])
        temporal_interval = int(t2w_window.get("temporal_interval", 1))
        if self.max_video_fps is not None:
            temporal_interval = max(temporal_interval, math.ceil(original_fps / self.max_video_fps))

        input_w, input_h = int(metadata["width"]), int(metadata["height"])
        target_w, target_h = self.output_sizes[metadata["aspect_ratio"]]
        resize_h, resize_w = self._resize_geometry(input_w, input_h, target_w, target_h)
        frames, total_frames, actual_stop = self._decode_episode(
            video_path=metadata["vision_path"],
            start_frame=start_frame,
            stop_frame=stop_frame,
            temporal_interval=temporal_interval,
            resize_h=resize_h,
            resize_w=resize_w,
            target_h=target_h,
            target_w=target_w,
        )
        if frames.shape[0] == 0:
            log.warning(
                f"No frames decoded for LeRobot episode {metadata['uuid']} "
                f"([{start_frame}, {stop_frame}), path={metadata['vision_path']})."
            )
            return None

        target_t = (frames.shape[0] - 1) // self.temporal_compression_factor * self.temporal_compression_factor + 1
        video = frames[:target_t].permute(1, 0, 2, 3).contiguous().to(torch.uint8)
        sampled_end = start_frame + (target_t - 1) * temporal_interval
        return self._build_processed_sample(
            metadata=metadata,
            t2w_window=t2w_window,
            win_idx=win_idx,
            video=video,
            original_fps=original_fps,
            total_frames=total_frames,
            start_frame=start_frame,
            end_frame=min(sampled_end, actual_stop - 1),
            temporal_interval=temporal_interval,
            target_h=target_h,
            target_w=target_w,
        )


def get_lerobot_episode_sft_dataset(
    roots: str | os.PathLike[str] | list[str | os.PathLike[str]],
    resolution: str | int = "480",
    tokenizer_config: Optional[Any] = None,
    metadata_load_workers: int = 8,
    video_view: str = "head",
    video_view_aliases: list[str] | None = None,
    max_video_fps: float | None = 15.0,
    caption_key: str | None = None,
    min_short_edge: int = 0,
    temporal_compression_factor: int = 4,
    num_ffmpeg_threads: int = 1,
    cfg_dropout_rate: float = 0.1,
    use_system_prompt: bool = False,
    max_caption_tokens: int = _MAX_CAPTION_TOKENS,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    cfg_dropout_keep_metadata: bool = False,
    caption_suffix: str = "",
    conditioning_fps: float = -1,
    conditioning_fps_noise_std: float = 0.0,
    conditioning_config: dict[int, float] | None = None,
    **kwargs,
) -> LeRobotEpisodeSFTDataset:
    """Create a standard causal vision-SFT dataset from local LeRobot v3 roots."""
    # Hydra parses numeric-looking TOML strings such as ``"256"`` as integers
    # when emitting dotted overrides. VIDEO_RES_SIZE_INFO uses string keys.
    resolution = str(resolution)
    if kwargs:
        log.info(f"Unknown kwargs for get_lerobot_episode_sft_dataset: {kwargs}")
    if resolution not in VIDEO_RES_SIZE_INFO:
        raise ValueError(f"Unknown resolution {resolution!r}; choices: {sorted(VIDEO_RES_SIZE_INFO)}")

    catalog = _build_episode_catalog(
        roots,
        metadata_load_workers=metadata_load_workers,
        video_view=video_view,
        video_view_aliases=video_view_aliases,
        caption_key=caption_key,
        min_short_edge=min_short_edge,
    )
    return LeRobotEpisodeSFTDataset(
        metadata=catalog,
        num_video_frames=-1,
        resolution=resolution,
        s3_credentials={},
        temporal_interval_mode="force_one",
        frame_selection_mode="first",
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
        max_video_fps=max_video_fps,
        num_ffmpeg_threads=num_ffmpeg_threads,
    )
