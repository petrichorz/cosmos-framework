#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run repeatable Cosmos performance experiments on Ascend NPUs."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAUNCH_SCRIPT = (
    REPO_ROOT.parent / "cosmos/cookbooks/cosmos3/generator/audiovisual/finetune/launch_sft_vision_edge_profile_local.sh"
)
DEFAULT_DATASET = Path(os.environ.get("DATASET_DIR", "/mnt/sfs_turbo/public/datasets/Cosmos3-DROID/success"))
DEFAULT_OUTPUT = Path("/mnt/sfs_turbo/zheng/cosmos-ascend-profile/cosmos-profile-logs")


def _split_evenly(items: list[dict[str, Any]], count: int) -> list[list[dict[str, Any]]]:
    chunk_size = max(1, (len(items) + count - 1) // count)
    return [items[start : start + chunk_size] for start in range(0, len(items), chunk_size)]


def _run_data_chunk(
    metadata: list[dict[str, Any]],
    output_dir: str,
    backend: str,
    resize_mode: str,
    resolution: str,
    cache_size: int,
    max_video_fps: float,
) -> dict[str, int]:
    os.environ["COSMOS_PERF_OUTPUT_DIR"] = output_dir
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

    from cosmos_framework.data.generator.local_datasets.helper import get_video_metadata
    from cosmos_framework.data.generator.local_datasets.sft_dataset_lerobot3 import (
        LeRobotSFTDataset,
        _LeRobotVideoDecoderCache,
        _limit_temporal_interval_by_fps,
    )
    from cosmos_framework.data.generator.utils import VIDEO_RES_SIZE_INFO
    from cosmos_framework.utils.performance import performance_scope, record_performance_event

    dataset = object.__new__(LeRobotSFTDataset)
    dataset.video_backend = backend
    dataset.video_resize_mode = resize_mode
    dataset.video_tolerance_s = 0.033
    dataset._decoder_cache = _LeRobotVideoDecoderCache(cache_size) if backend == "torchcodec" else None

    successes = 0
    failures = 0
    frames = 0
    for item in metadata:
        video_path = item["vision_path"]
        window = item["t2w_windows"][0]
        try:
            with performance_scope("data_benchmark_sample", video_path=video_path, uuid=item["uuid"]):
                with performance_scope("video_metadata", video_path=video_path):
                    video_info = get_video_metadata(video_path)
                original_fps = float(video_info["fps"])
                end_frame = min(int(window["end_frame"]), int(video_info["total_frames"]) - 1)
                step = _limit_temporal_interval_by_fps(
                    original_fps,
                    int(window.get("temporal_interval", 1)),
                    max_video_fps,
                )
                target_w, target_h = VIDEO_RES_SIZE_INFO[resolution][item["aspect_ratio"]]
                scale = max(target_w / item["width"], target_h / item["height"])
                resize_h = round(item["height"] * scale)
                resize_w = round(item["width"] * scale)
                decoded = dataset._decode_video_frames(
                    video_path=video_path,
                    start_frame=int(window["start_frame"]),
                    end_frame=end_frame,
                    temporal_interval=step,
                    original_fps=original_fps,
                    resize_h=resize_h,
                    resize_w=resize_w,
                )
                frames += len(decoded)
            successes += 1
        except Exception as error:
            failures += 1
            record_performance_event(
                "data_benchmark_error",
                video_path=video_path,
                error_type=type(error).__name__,
                error_message=str(error),
            )
            if dataset._decoder_cache is not None:
                dataset._decoder_cache.discard(video_path)
    return {"successes": successes, "failures": failures, "frames": frames}


def run_data_benchmark(args: argparse.Namespace, run_dir: Path) -> None:
    from cosmos_framework.data.generator.local_datasets.sft_dataset_lerobot3 import _load_lerobot_metadata
    from cosmos_framework.utils.performance import summarize_performance_events

    metadata = _load_lerobot_metadata(
        str(args.dataset),
        min_frames=1,
        max_video_duration_s=0,
        video_feature_keywords=["top", "head"],
        caption_key="caption",
    )
    if args.data_order == "video_grouped":
        metadata.sort(key=lambda item: (item["vision_path"], item["uuid"]))
    else:
        random.Random(args.seed).shuffle(metadata)
    metadata = metadata[: args.data_samples]
    if not metadata:
        raise RuntimeError(f"No usable LeRobot samples were found under {args.dataset}")

    events_dir = run_dir / "events"
    chunks = _split_evenly(metadata, min(args.num_workers, len(metadata)))
    worker_args = (
        str(events_dir),
        args.video_backend,
        args.video_resize_mode,
        args.resolution,
        args.decoder_cache_size,
        args.max_video_fps,
    )
    start_time = time.perf_counter()
    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [executor.submit(_run_data_chunk, chunk, *worker_args) for chunk in chunks]
        results = [future.result() for future in futures]

    totals = {key: sum(result[key] for result in results) for key in ("successes", "failures", "frames")}
    elapsed_s = time.perf_counter() - start_time
    totals.update(
        {
            "elapsed_s": elapsed_s,
            "samples_per_s": totals["successes"] / elapsed_s,
            "frames_per_s": totals["frames"] / elapsed_s,
        }
    )
    (run_dir / "data_result.json").write_text(json.dumps(totals, indent=2), encoding="utf-8")
    summarize_performance_events(events_dir, run_dir / "summary")
    print(f"Data benchmark: {totals}")
    if totals["successes"] == 0:
        raise RuntimeError(f"All {totals['failures']} data benchmark samples failed; inspect events JSONL")


def _training_overrides(args: argparse.Namespace, mode: str) -> list[str]:
    overrides = [
        f"trainer.max_iter={args.max_steps}",
        f"checkpoint.save_iter={args.max_steps + 1000}",
        f"dataloader_train.dataloader.num_workers={args.num_workers}",
        f"dataloader_train.dataloader.datasets.video.dataset.video_backend={args.video_backend}",
        f"dataloader_train.dataloader.datasets.video.dataset.video_resize_mode={args.video_resize_mode}",
        f"dataloader_train.dataloader.datasets.video.dataset.decoder_cache_max_size={args.decoder_cache_size}",
        f"dataloader_train.dataloader.datasets.video.dataset.max_video_fps={args.max_video_fps}",
    ]
    if mode in {"npu", "distributed"}:
        target_ranks = [0] if mode == "npu" else list(range(args.nproc_per_node))
        overrides.extend(
            [
                "trainer.profiling.enable_profiling=true",
                f"trainer.profiling.profile_freq={args.profile_step}",
                f"trainer.profiling.profile_warmup={args.profile_warmup}",
                f"trainer.profiling.target_ranks={target_ranks}",
                f"trainer.profiling.record_shape={str(args.record_shapes).lower()}",
                f"trainer.profiling.profile_memory={str(args.profile_memory).lower()}",
                f"trainer.profiling.with_stack={str(args.with_stack).lower()}",
                f"trainer.profiling.with_modules={str(args.with_modules).lower()}",
            ]
        )
    return overrides + args.extra_override


def run_training(args: argparse.Namespace, mode: str, run_dir: Path) -> None:
    from cosmos_framework.utils.performance import summarize_ascend_profiler_outputs, summarize_performance_events

    events_dir = run_dir / "events"
    env = os.environ.copy()
    env.update(
        {
            "COSMOS_PERF_OUTPUT_DIR": str(events_dir),
            "COSMOS_PERF_PROFILE_MODE": mode,
            "COSMOS_NPU_PROFILER_LEVEL": args.profiler_level,
            "COSMOS_NPU_AIC_METRICS": args.aic_metrics,
            "COSMOS_NPU_EXPORT_DB": "0" if args.no_db else "1",
            "COSMOS_NPU_ASYNC_ANALYSIS": "0" if args.sync_analysis else "1",
            "COSMOS_NPU_PROFILE_ACTIVE_STEPS": str(args.profile_active_steps),
            "COSMOS_NPU_MSTX": "1" if args.mstx_forward and mode in {"npu", "distributed"} else "0",
            "COSMOS_PERF_SKIP_FINAL_CHECKPOINT": "1",
            "COSMOS_PERF_RECORD_MEMORY": "1",
            "DATASET_DIR": str(args.dataset),
            "OUTPUT_ROOT": str(run_dir / "training_output"),
            "NPROC_PER_NODE": str(args.nproc_per_node),
        }
    )
    command = ["bash", str(args.launch_script), *_training_overrides(args, mode)]
    print("Running:", " ".join(command))
    with (run_dir / "training.log").open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        returncode = process.wait()
    summarize_performance_events(events_dir, run_dir / "summary")
    if mode in {"npu", "distributed"}:
        ascend_summary = summarize_ascend_profiler_outputs(run_dir / "training_output", run_dir / "summary")
        if not ascend_summary["trace_view_files"]:
            print("Warning: no trace_view.json was found; inspect the training log and profiler output directory")
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "data", "npu", "distributed", "all"), default="baseline")
    parser.add_argument("--launch-script", type=Path, default=DEFAULT_LAUNCH_SCRIPT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument(
        "--profile-step",
        type=int,
        default=8,
        help="final step of the first profiler capture window",
    )
    parser.add_argument("--profile-warmup", type=int, default=2)
    parser.add_argument(
        "--profile-active-steps",
        type=int,
        default=1,
        help="number of consecutive active steps to retain in the profiler capture window",
    )
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--video-backend", choices=("pyav", "torchcodec"), default="pyav")
    parser.add_argument("--video-resize-mode", choices=("decode_transform", "post_decode"), default="decode_transform")
    parser.add_argument("--profiler-level", choices=("level0", "level1", "level2"), default="level0")
    parser.add_argument("--aic-metrics", choices=("none", "pipe", "arithmetic", "memory", "l2cache"), default="none")
    parser.add_argument("--record-shapes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--sync-analysis",
        action="store_true",
        help="wait for Ascend trace analysis before summarizing outputs",
    )
    parser.add_argument("--with-stack", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--with-modules", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--mstx-forward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="emit nested Cosmos forward ranges into the Ascend trace",
    )
    parser.add_argument("--no-db", action="store_true")
    parser.add_argument("--data-samples", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-order", choices=("random", "video_grouped"), default="random")
    parser.add_argument("--decoder-cache-size", type=int, default=64)
    parser.add_argument("--resolution", choices=("256", "480", "704", "720", "768", "1080"), default="256")
    parser.add_argument("--max-video-fps", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extra-override", action="append", default=[])
    args = parser.parse_args()
    if args.profile_warmup < 0:
        parser.error("--profile-warmup must be non-negative")
    if args.profile_active_steps < 1:
        parser.error("--profile-active-steps must be positive")
    if args.profile_step < args.profile_warmup + args.profile_active_steps:
        parser.error("--profile-step must be at least --profile-warmup + --profile-active-steps")
    if args.num_workers < 1 or args.data_samples < 1:
        parser.error("--num-workers and --data-samples must be positive")
    if args.nproc_per_node < 1 or args.max_steps < 1 or args.decoder_cache_size < 1:
        parser.error("--nproc-per-node, --max-steps, and --decoder-cache-size must be positive")
    if args.profiler_level == "level0" and args.aic_metrics != "none":
        parser.error("AI Core metrics require --profiler-level level1 or level2")
    if args.mode in {"npu", "distributed", "all"} and args.max_steps < args.profile_step:
        parser.error("--max-steps must be greater than or equal to --profile-step for profiler modes")
    return args


def main() -> None:
    args = parse_args()
    python_bin = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = python_bin + os.pathsep + os.environ.get("PATH", "")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    root = args.output_root / f"ascend_profile_{timestamp}"
    root.mkdir(parents=True, exist_ok=False)
    modes = ["data", "baseline", "npu", "distributed"] if args.mode == "all" else [args.mode]
    for mode in modes:
        mode_dir = root / mode
        mode_dir.mkdir(parents=True)
        if mode == "data":
            run_data_benchmark(args, mode_dir)
        else:
            run_training(args, mode, mode_dir)
    print(f"Profiling output: {root}")


if __name__ == "__main__":
    main()
