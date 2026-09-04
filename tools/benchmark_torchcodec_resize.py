# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Benchmark two TorchCodec resize placements on the LeRobot v3 read path.

The two measured paths are:

1. ``post_decode``: TorchCodec returns source-resolution frames, then the frames
   selected by ``temporal_interval`` are resized.
2. ``decode_transform``: the same resize is passed to
   ``VideoDecoder(transforms=[...])`` and is therefore applied while frames are
   returned by TorchCodec.

Everything after decoding (temporal sampling, temporal alignment, centre crop,
THWC -> CTHW conversion) follows ``LeRobotSFTDataset.process_one_sample``. No
tokenizer, model, accelerator, or network operation is involved.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing as mp
import os
import queue
import random
import statistics
import threading
import time
import traceback
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

# This is a CPU-only data benchmark. Avoid importing an installed accelerator
# backend when torch is imported in spawned workers.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

Method = Literal["post_decode", "decode_transform"]


@dataclass
class RunResult:
    method: Method
    repeat: int
    samples: int
    frames: int
    output_bytes: int
    metadata_seconds: float
    read_seconds: float
    samples_per_second: float
    frames_per_second: float
    rss_before_metadata_mib: float
    rss_before_read_mib: float
    peak_read_rss_mib: float
    peak_read_rss_delta_mib: float
    output_shapes: list[list[int]]


class _RSSMonitor:
    """Sample current process RSS without adding a psutil dependency."""

    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def current_bytes() -> int:
        # This benchmark targets the Linux training environment used by the repo.
        with open("/proc/self/statm") as statm:
            resident_pages = int(statm.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")

    def start(self) -> None:
        self.peak_bytes = self.current_bytes()
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.peak_bytes = max(self.peak_bytes, self.current_bytes())

    def checkpoint(self) -> None:
        self.peak_bytes = max(self.peak_bytes, self.current_bytes())

    def stop(self) -> int:
        self.checkpoint()
        self._stop.set()
        assert self._thread is not None
        self._thread.join()
        return self.peak_bytes


class _DecoderCache:
    """LRU cache matching the LeRobot reader, including open file handles."""

    def __init__(self, max_size: int, num_ffmpeg_threads: int | None) -> None:
        self.max_size = max_size
        self.num_ffmpeg_threads = num_ffmpeg_threads
        self.cache: OrderedDict[tuple[str, tuple[int, int] | None], tuple[Any, Any]] = OrderedDict()

    def get(self, path: str, target_hw: tuple[int, int] | None) -> Any:
        from torchcodec.decoders import VideoDecoder
        from torchvision.transforms.v2 import Resize

        key = (path, target_hw)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key][0]

        file_handle = open(path, "rb")
        kwargs: dict[str, Any] = {"seek_mode": "exact"}
        if self.num_ffmpeg_threads is not None:
            kwargs["num_ffmpeg_threads"] = self.num_ffmpeg_threads
        if target_hw is not None:
            kwargs["transforms"] = [Resize(target_hw)]
        try:
            decoder = VideoDecoder(file_handle, **kwargs)
        except Exception:
            file_handle.close()
            raise
        self.cache[key] = (decoder, file_handle)
        while len(self.cache) > self.max_size:
            _, (_, old_handle) = self.cache.popitem(last=False)
            old_handle.close()
        return decoder

    def close(self) -> None:
        while self.cache:
            _, (_, file_handle) = self.cache.popitem(last=False)
            file_handle.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True, help="LeRobot 根目录，或包含多个 LeRobot 根目录的父目录")
    parser.add_argument("--resolution", default="256", help="VIDEO_RES_SIZE_INFO 中的分辨率档位（默认：256）")
    parser.add_argument("--samples", type=int, default=32, help="每轮正式测量的 episode 数（默认：32）")
    parser.add_argument("--warmup-samples", type=int, default=1, help="每个子进程正式计时前的预热 episode 数（默认：1）")
    parser.add_argument("--repeats", type=int, default=3, help="每种方法用独立子进程重复的次数（默认：3）")
    parser.add_argument("--num-video-frames", type=int, default=-1, help="-1 表示读取完整 episode；正数与训练数据集语义相同")
    parser.add_argument(
        "--temporal-interval-mode",
        choices=("force_one", "max_30fps", "entire_chunk"),
        default="max_30fps",
    )
    parser.add_argument("--frame-selection-mode", choices=("first", "center", "random"), default="first")
    parser.add_argument("--temporal-compression-factor", type=int, default=4)
    parser.add_argument("--video-feature-key", default=None, help="精确指定 LeRobot video feature")
    parser.add_argument(
        "--video-feature-keywords",
        nargs="*",
        default=["top", "head"],
        help="按顺序匹配 video feature；传空列表时回退到首个 video feature",
    )
    parser.add_argument("--caption-key", default="caption")
    parser.add_argument("--min-frames", type=int, default=61, help="metadata 过滤阈值，与当前训练配置默认值一致")
    parser.add_argument("--decoder-cache-size", type=int, default=64)
    parser.add_argument(
        "--num-ffmpeg-threads",
        type=int,
        default=None,
        help="默认不传该参数，与当前 LeRobot reader 一致；0 表示让 FFmpeg 自动选择",
    )
    parser.add_argument("--memory-sample-ms", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None, help="可选：保存逐轮结果和汇总")
    parser.add_argument("--_worker-method", choices=("post_decode", "decode_transform"), help=argparse.SUPPRESS)
    parser.add_argument("--_worker-repeat", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    for name in ("samples", "repeats", "decoder_cache_size", "temporal_compression_factor", "min_frames"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} 必须大于 0")
    if args.warmup_samples < 0:
        parser.error("--warmup-samples 不能小于 0")
    if args.memory_sample_ms <= 0:
        parser.error("--memory-sample-ms 必须大于 0")
    return args


def _choose_metadata(args: argparse.Namespace) -> list[dict[str, Any]]:
    from cosmos_framework.data.generator.local_datasets.sft_dataset_lerobot3 import _load_lerobot_metadata

    metadata = _load_lerobot_metadata(
        str(args.data_root),
        min_frames=args.min_frames,
        min_short_edge=0,
        video_feature_key=args.video_feature_key,
        caption_key=args.caption_key,
        video_feature_keywords=args.video_feature_keywords or None,
    )
    # Match get_sft_dataset_from_lerobot's stable ordering.
    metadata.sort(key=lambda item: hashlib.sha256(item["uuid"].encode()).hexdigest())
    required = args.samples + args.warmup_samples
    if len(metadata) < required:
        raise ValueError(f"仅发现 {len(metadata)} 个有效 episode，但预热和测量共需要 {required} 个")
    selected = metadata[:required]
    missing = sorted({item["vision_path"] for item in selected if not Path(item["vision_path"]).is_file()})
    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(f"以下视频不存在（最多展示 5 个）：\n{preview}")
    return selected


def _frame_plan(
    metadata: dict[str, Any], decoder_length: int, args: argparse.Namespace, rng: random.Random
) -> tuple[int, int, int]:
    window = metadata["t2w_windows"][0]
    window_start = int(window["start_frame"])
    actual_end = min(int(window["end_frame"]), decoder_length - 1)
    frames_in_window = actual_end - window_start + 1
    if frames_in_window < 1:
        raise ValueError(f"episode {metadata['uuid']} 没有可读取帧")

    if args.num_video_frames == -1:
        return window_start, actual_end, int(window["temporal_interval"])
    if frames_in_window < args.num_video_frames:
        raise ValueError(
            f"episode {metadata['uuid']} 只有 {frames_in_window} 帧，少于 --num-video-frames={args.num_video_frames}"
        )
    if args.temporal_interval_mode == "force_one":
        interval = 1
    elif args.temporal_interval_mode == "max_30fps":
        interval = max(1, int(float(metadata["framerate"]) / 30.0))
    else:
        interval = max(1, frames_in_window // args.num_video_frames)

    span = (args.num_video_frames - 1) * interval + 1
    if args.frame_selection_mode == "first":
        start = window_start
    elif args.frame_selection_mode == "center":
        start = window_start + (frames_in_window - span) // 2
    else:
        start = window_start + rng.randint(0, max(0, frames_in_window - span))
    return start, start + span - 1, interval


def _read_one(
    metadata: dict[str, Any], method: Method, cache: _DecoderCache, args: argparse.Namespace, rng: random.Random
) -> Any:
    import numpy as np
    import torch
    from torchvision.transforms.v2 import Resize

    from cosmos_framework.data.generator.utils import VIDEO_RES_SIZE_INFO

    input_w, input_h = int(metadata["width"]), int(metadata["height"])
    target_w, target_h = VIDEO_RES_SIZE_INFO[args.resolution][metadata["aspect_ratio"]]
    resize_ratio = max(target_w / input_w, target_h / input_h)
    resize_hw = (round(input_h * resize_ratio), round(input_w * resize_ratio))
    crop_y = round((resize_hw[0] - target_h) / 2)
    crop_x = round((resize_hw[1] - target_w) / 2)

    decoder = cache.get(metadata["vision_path"], resize_hw if method == "decode_transform" else None)
    start, end, interval = _frame_plan(metadata, len(decoder), args, rng)
    # Deliberately retain get_frames_in_range + post-read striding: this is the
    # current LeRobotSFTDataset behavior, not a synthetic get_frames_at microbench.
    frames = decoder.get_frames_in_range(start=start, stop=end + 1).data
    if interval > 1:
        frames = frames[0::interval]
    if method == "post_decode":
        frames = Resize(resize_hw)(frames)

    # Keep the same copies/layout transitions as LeRobotSFTDataset so measured
    # memory is representative of the complete data-read output construction.
    frames_nhwc = frames.permute(0, 2, 3, 1).cpu().numpy()
    video_chunk = np.stack([frames_nhwc[index] for index in range(frames_nhwc.shape[0])], axis=0)
    target_t = (
        (video_chunk.shape[0] - 1) // args.temporal_compression_factor * args.temporal_compression_factor + 1
    )
    video_chunk = video_chunk[:target_t, crop_y : crop_y + target_h, crop_x : crop_x + target_w]
    video_chunk = np.transpose(video_chunk, (3, 0, 1, 2))
    return torch.from_numpy(np.ascontiguousarray(video_chunk)).to(torch.uint8)


def _worker(args: argparse.Namespace, result_queue: Any) -> None:
    try:
        method = cast(Method, args._worker_method)
        rss_before_metadata = _RSSMonitor.current_bytes()
        metadata_started = time.perf_counter()
        metadata = _choose_metadata(args)
        metadata_seconds = time.perf_counter() - metadata_started

        # Warm imports/libraries and page cache, then discard all decoder/output state.
        if args.warmup_samples:
            warm_cache = _DecoderCache(args.decoder_cache_size, args.num_ffmpeg_threads)
            warm_rng = random.Random(args.seed)
            for item in metadata[: args.warmup_samples]:
                _read_one(item, method, warm_cache, args, warm_rng)
            warm_cache.close()
        gc.collect()

        rss_before_read = _RSSMonitor.current_bytes()
        monitor = _RSSMonitor(args.memory_sample_ms / 1000.0)
        monitor.start()
        cache = _DecoderCache(args.decoder_cache_size, args.num_ffmpeg_threads)
        rng = random.Random(args.seed)
        output_shapes: list[list[int]] = []
        frames = 0
        output_bytes = 0
        started = time.perf_counter()
        try:
            for item in metadata[args.warmup_samples :]:
                output = _read_one(item, method, cache, args, rng)
                output_shapes.append(list(output.shape))
                frames += int(output.shape[1])
                output_bytes += output.numel() * output.element_size()
                monitor.checkpoint()
                del output
        finally:
            cache.close()
        read_seconds = time.perf_counter() - started
        peak_read_rss = monitor.stop()

        result = RunResult(
            method=method,
            repeat=args._worker_repeat,
            samples=len(output_shapes),
            frames=frames,
            output_bytes=output_bytes,
            metadata_seconds=metadata_seconds,
            read_seconds=read_seconds,
            samples_per_second=len(output_shapes) / read_seconds,
            frames_per_second=frames / read_seconds,
            rss_before_metadata_mib=rss_before_metadata / 2**20,
            rss_before_read_mib=rss_before_read / 2**20,
            peak_read_rss_mib=peak_read_rss / 2**20,
            peak_read_rss_delta_mib=max(0, peak_read_rss - rss_before_read) / 2**20,
            output_shapes=output_shapes,
        )
        result_queue.put({"ok": True, "result": asdict(result)})
    except BaseException:
        result_queue.put({"ok": False, "traceback": traceback.format_exc()})


def _run_child(args: argparse.Namespace, method: Method, repeat: int) -> dict[str, Any]:
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    worker_args = argparse.Namespace(**vars(args))
    worker_args._worker_method = method
    worker_args._worker_repeat = repeat
    process = context.Process(target=_worker, args=(worker_args, result_queue))
    process.start()
    process.join()
    try:
        message = result_queue.get(timeout=1)
    except queue.Empty as error:
        raise RuntimeError(f"{method} 子进程退出且未返回结果，exitcode={process.exitcode}") from error
    finally:
        result_queue.close()
    if not message["ok"]:
        raise RuntimeError(f"{method} 子进程失败：\n{message['traceback']}")
    if process.exitcode != 0:
        raise RuntimeError(f"{method} 子进程异常退出，exitcode={process.exitcode}")
    return message["result"]


def _median(results: list[dict[str, Any]], key: str) -> float:
    return float(statistics.median(result[key] for result in results))


def _summarize(all_results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary = {}
    for method in ("post_decode", "decode_transform"):
        method_results = [result for result in all_results if result["method"] == method]
        summary[method] = {
            "median_read_seconds": _median(method_results, "read_seconds"),
            "median_samples_per_second": _median(method_results, "samples_per_second"),
            "median_frames_per_second": _median(method_results, "frames_per_second"),
            "median_peak_read_rss_mib": _median(method_results, "peak_read_rss_mib"),
            "median_peak_read_rss_delta_mib": _median(method_results, "peak_read_rss_delta_mib"),
        }
    baseline = summary["post_decode"]
    fused = summary["decode_transform"]
    summary["decode_transform_vs_post_decode"] = {
        "time_change_percent": (fused["median_read_seconds"] / baseline["median_read_seconds"] - 1) * 100,
        "throughput_change_percent": (
            fused["median_frames_per_second"] / baseline["median_frames_per_second"] - 1
        )
        * 100,
        "peak_rss_delta_change_mib": (
            fused["median_peak_read_rss_delta_mib"] - baseline["median_peak_read_rss_delta_mib"]
        ),
    }
    return summary


def _print_summary(summary: dict[str, dict[str, float]]) -> None:
    print("\n中位数结果（每轮均为独立子进程）")
    print(f"{'method':<20} {'time(s)':>10} {'sample/s':>12} {'frame/s':>12} {'peak RSS':>12} {'RSS delta':>12}")
    for method in ("post_decode", "decode_transform"):
        item = summary[method]
        print(
            f"{method:<20} {item['median_read_seconds']:>10.3f} "
            f"{item['median_samples_per_second']:>12.3f} {item['median_frames_per_second']:>12.3f} "
            f"{item['median_peak_read_rss_mib']:>9.1f} MiB {item['median_peak_read_rss_delta_mib']:>9.1f} MiB"
        )
    comparison = summary["decode_transform_vs_post_decode"]
    print(
        "\ndecode_transform 相对 post_decode："
        f"耗时 {comparison['time_change_percent']:+.2f}%，"
        f"帧吞吐 {comparison['throughput_change_percent']:+.2f}%，"
        f"读取阶段峰值 RSS 增量 {comparison['peak_rss_delta_change_mib']:+.1f} MiB"
    )


def main() -> None:
    args = _parse_args()
    if args._worker_method is not None:
        raise RuntimeError("内部 worker 参数不能直接使用")

    all_results = []
    # Alternate order by repeat to reduce systematic page-cache/order bias.
    for repeat in range(args.repeats):
        methods: tuple[Method, Method] = (
            ("post_decode", "decode_transform") if repeat % 2 == 0 else ("decode_transform", "post_decode")
        )
        for method in methods:
            print(f"[{repeat + 1}/{args.repeats}] running {method} ...", flush=True)
            result = _run_child(args, method, repeat)
            all_results.append(result)
            print(
                f"  {result['read_seconds']:.3f}s, {result['frames_per_second']:.2f} frame/s, "
                f"peak RSS delta {result['peak_read_rss_delta_mib']:.1f} MiB",
                flush=True,
            )

    # Verify both methods produced exactly the same tensor shapes for each repeat.
    for repeat in range(args.repeats):
        pair = [result for result in all_results if result["repeat"] == repeat]
        if len(pair) != 2 or pair[0]["output_shapes"] != pair[1]["output_shapes"]:
            raise RuntimeError(f"第 {repeat + 1} 轮两种方法的输出 shape 不一致")

    summary = _summarize(all_results)
    _print_summary(summary)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
            },
            "runs": all_results,
            "summary": summary,
        }
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        print(f"逐轮结果已写入 {args.output_json}")


if __name__ == "__main__":
    main()
