# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Probe host RSS while iterating the LeRobot vision-SFT dataloader (sft_dataset_lerobot3).

Builds the same ``PackingDataLoader(RankPartitionedDataLoader(get_sft_dataset_from_lerobot))``
stack as ``vision_sft_edge_lerobot3``, then pins ``num_workers=1``, ``prefetch_factor=1``,
``max_samples_per_batch=1``. Does not load the model, VAE, or NPU.

Each step takes one packed batch, records parent/worker RSS, drops the batch, and
``gc.collect()``s. After warmup, a linear fit on total RSS vs step decides whether
memory is still climbing.

Example::

    HF_HUB_OFFLINE=1 PYTHONPATH=. \\
      python -m cosmos_framework.scripts.probe_sft_lerobot3_rss \\
        --video-backend pyav --num-batches 40 --warmup 8 --no-tokenizer
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Any


def _reexec_with_conda_lib() -> None:
    """Put ``$CONDA_PREFIX/lib`` first on ``LD_LIBRARY_PATH`` so native video
    backends can load conda ``libstdc++``. ``conda activate`` does this; a bare
    ``$ENV/bin/python -m ...`` does not."""
    if os.environ.get("_SFT_LEROBOT3_RSS_PROBE_REEXEC") == "1":
        return
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        exe = Path(sys.executable)
        if exe.parent.name == "bin":
            prefix = str(exe.parent.parent)
    if not prefix:
        return
    lib = str(Path(prefix) / "lib")
    bindir = str(Path(prefix) / "bin")
    current = os.environ.get("LD_LIBRARY_PATH", "")
    lib_parts = [p for p in current.split(":") if p]
    path_parts = [p for p in os.environ.get("PATH", "").split(":") if p]
    already = lib_parts[:1] == [lib] and path_parts[:1] == [bindir]
    if already:
        return
    os.environ["LD_LIBRARY_PATH"] = lib + ((":" + current) if current else "")
    os.environ["PATH"] = bindir + ((":" + os.environ["PATH"]) if os.environ.get("PATH") else "")
    os.environ["_SFT_LEROBOT3_RSS_PROBE_REEXEC"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


_reexec_with_conda_lib()

import psutil
import torch
import torch.distributed as dist

from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, RankPartitionedDataLoader
from cosmos_framework.data.generator.local_datasets.sft_dataset_lerobot3 import get_sft_dataset_from_lerobot
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import instantiate

_DEFAULT_DATASET = "/data5T/Embodied-AI/datasets/Cosmos3-DROID/success"
_DEFAULT_PROCESSOR = "/data5T/Embodied-AI/ckpts/Cosmos/Cosmos3-Edge"
_DEFAULT_MASTER_PORT = "29751"
_MIB = 1024 * 1024


class _DummyTokenizer:
    special_tokens_map: dict = {}

    def add_tokens(self, tokens):
        return 0

    def convert_tokens_to_ids(self, token):
        return 0

    def apply_chat_template(self, *args, **kwargs):
        return [1]


class _DummyProcessor:
    tokenizer = _DummyTokenizer()


def _absolute(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _mib(n_bytes: int | float) -> float:
    return float(n_bytes) / _MIB


def _tensor_nbytes(obj: Any) -> int:
    if torch.is_tensor(obj):
        return int(obj.numel() * obj.element_size())
    if isinstance(obj, dict):
        return sum(_tensor_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_nbytes(v) for v in obj)
    return 0


def _first_video_tensor(obj: Any) -> torch.Tensor | None:
    if torch.is_tensor(obj) and obj.ndim >= 4:
        return obj
    if isinstance(obj, dict):
        if "video" in obj:
            found = _first_video_tensor(obj["video"])
            if found is not None:
                return found
        for v in obj.values():
            found = _first_video_tensor(v)
            if found is not None:
                return found
        return None
    if isinstance(obj, (list, tuple)):
        for v in obj:
            found = _first_video_tensor(v)
            if found is not None:
                return found
    return None


def _video_t(batch: dict) -> int | None:
    video = _first_video_tensor(batch)
    if video is None:
        return None
    # Vision SFT samples are uint8 [C, T, H, W] (or a list of those).
    return int(video.shape[-3])


def _rss_parent_and_children(proc: psutil.Process) -> tuple[int, int, list[int]]:
    parent = int(proc.memory_info().rss)
    children = 0
    child_pids: list[int] = []
    for child in proc.children(recursive=True):
        try:
            children += int(child.memory_info().rss)
            child_pids.append(int(child.pid))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return parent, children, child_pids


def _linear_slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    sx = sum(xs)
    sy = sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sxx = sum(x * x for x in xs)
    den = n * sxx - sx * sx
    if den == 0:
        return 0.0
    return (n * sxy - sx * sy) / den


def _init_dist() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", os.environ.get("MASTER_PORT", _DEFAULT_MASTER_PORT))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)


def _tokenizer_config(processor_path: str) -> Any:
    from cosmos_framework.data.generator.processors import build_processor_lazy

    return L(build_processor_lazy)(tokenizer_type=processor_path)


def build_sft_lerobot3_dataloader(
    *,
    lerobot_root: str,
    num_workers: int,
    prefetch_factor: int,
    max_samples_per_batch: int,
    pin_memory: bool,
    persistent_workers: bool,
    tokenizer_config: Any,
    video_backend: str,
    video_resize_mode: str,
    video_tolerance_s: float,
    resolution: str,
    min_video_frames: int,
    max_video_duration_s: float,
    max_video_fps: float,
    video_feature_key: str | None,
    video_feature_keywords: list[str] | None,
    caption_key: str,
    decoder_cache_max_size: int,
) -> PackingDataLoader:
    """Mirror ``vision_sft_edge_lerobot3.dataloader_train`` with probe overrides."""
    return instantiate(
        L(PackingDataLoader)(
            audio_sample_rate=48000,
            dataset_name="video",
            max_samples_per_batch=max_samples_per_batch,
            max_sequence_length=None,
            lookahead_limit=1,
            patch_spatial=2,
            sound_latent_fps=0,
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            dataloader=L(RankPartitionedDataLoader)(
                batch_size=1,
                num_workers=num_workers,
                persistent_workers=bool(persistent_workers and num_workers > 0),
                pin_memory=pin_memory,
                **({"prefetch_factor": prefetch_factor} if num_workers > 0 else {}),
                sampler=None,
                datasets=dict(
                    video=dict(
                        ratio=1,
                        dataset=L(get_sft_dataset_from_lerobot)(
                            append_duration_fps_timestamps=True,
                            append_resolution_info=True,
                            max_caption_tokens=2048,
                            caption_suffix="",
                            cfg_dropout_keep_metadata=False,
                            cfg_dropout_rate=0.0,
                            conditioning_config={0: 0.7, 1: 0.2, 2: 0.1},
                            conditioning_fps=-1,
                            conditioning_fps_noise_std=0.0,
                            frame_selection_mode="first",
                            lerobot_root=lerobot_root,
                            video_feature_key=video_feature_key,
                            video_feature_keywords=video_feature_keywords,
                            video_backend=video_backend,
                            video_resize_mode=video_resize_mode,
                            video_tolerance_s=video_tolerance_s,
                            max_video_fps=max_video_fps,
                            caption_key=caption_key,
                            min_short_edge=0,
                            num_video_frames=-1,
                            min_video_frames=min_video_frames,
                            max_video_duration_s=max_video_duration_s,
                            resolution=resolution,
                            sample_by_window=False,
                            temporal_compression_factor=4,
                            temporal_interval_mode="max_30fps",
                            use_system_prompt=False,
                            tokenizer_config=tokenizer_config,
                            decoder_cache_max_size=decoder_cache_max_size,
                        ),
                    ),
                ),
            ),
        )
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--lerobot-root",
        type=Path,
        default=Path(os.environ.get("DATASET_PATH", os.environ.get("DATASET_DIR", _DEFAULT_DATASET))),
        help="LeRobot root (contains meta/info.json) or a parent of several.",
    )
    parser.add_argument("--num-batches", type=int, default=40, help="Packed batches to iterate.")
    parser.add_argument("--warmup", type=int, default=8, help="Steps skipped by the leak slope fit.")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--max-samples-per-batch", type=int, default=1)
    parser.add_argument("--video-backend", type=str, default="pyav", choices=["pyav", "torchcodec"])
    parser.add_argument(
        "--video-resize-mode",
        type=str,
        default="post_decode",
        choices=["post_decode", "decode_transform"],
    )
    parser.add_argument(
        "--video-tolerance-s",
        type=float,
        default=1e-4,
        help="PyAV nearest-frame tolerance. Training default is 1e-4.",
    )
    parser.add_argument("--resolution", type=str, default="256")
    parser.add_argument("--min-video-frames", type=int, default=61)
    parser.add_argument("--max-video-duration-s", type=float, default=61.0)
    parser.add_argument("--max-video-fps", type=float, default=30.0)
    parser.add_argument(
        "--video-feature-key",
        type=str,
        default=None,
        help="Exact LeRobot video feature name. Default: keyword match then first video.",
    )
    parser.add_argument(
        "--video-feature-keywords",
        type=str,
        nargs="*",
        default=["top", "head"],
        help="Substring match against video feature names (training default).",
    )
    parser.add_argument("--caption-key", type=str, default="caption")
    parser.add_argument(
        "--decoder-cache-max-size",
        type=int,
        default=64,
        help="TorchCodec VideoDecoder LRU. Ignored for pyav.",
    )
    parser.add_argument("--pin-memory", action="store_true", help="Match training pin_memory=True (hides RSS).")
    parser.add_argument(
        "--no-tokenizer",
        action="store_true",
        help="Skip the Edge processor; use a dummy tokenizer so video RSS is isolated.",
    )
    parser.add_argument("--processor-path", type=Path, default=Path(_DEFAULT_PROCESSOR))
    parser.add_argument("--tsv-out", type=Path, default=None, help="Optional TSV path for the per-step table.")
    parser.add_argument("--fail-on-leak", action="store_true", help="Exit 1 when the leak heuristic fires.")
    parser.add_argument("--fail-slope-mib", type=float, default=32.0, help="MiB/step slope threshold after warmup.")
    parser.add_argument(
        "--fail-delta-mib",
        type=float,
        default=512.0,
        help="MiB growth from first post-warmup step to last.",
    )
    return parser.parse_args(argv)


def _require_paths(lerobot_root: Path) -> None:
    if not lerobot_root.is_dir():
        raise SystemExit(f"ERROR: lerobot root not found: {lerobot_root}")
    direct = lerobot_root / "meta" / "info.json"
    if direct.is_file():
        return
    found = next(lerobot_root.rglob("meta/info.json"), None)
    if found is None:
        raise SystemExit(f"ERROR: no meta/info.json under {lerobot_root}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.num_batches < 1:
        raise SystemExit("ERROR: --num-batches must be >= 1")
    if args.warmup < 0 or args.warmup >= args.num_batches:
        raise SystemExit("ERROR: --warmup must be in [0, num_batches)")
    if args.video_tolerance_s <= 0:
        raise SystemExit("ERROR: --video-tolerance-s must be > 0")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    lerobot_root = _absolute(args.lerobot_root)
    _require_paths(lerobot_root)

    if args.no_tokenizer:
        tokenizer_config = _DummyProcessor()
    else:
        processor_path = _absolute(args.processor_path)
        if not processor_path.is_dir():
            raise SystemExit(f"ERROR: missing Edge processor dir at {processor_path}")
        tokenizer_config = _tokenizer_config(str(processor_path))

    keywords = list(args.video_feature_keywords) if args.video_feature_keywords else None

    _init_dist()
    proc = psutil.Process(os.getpid())
    print(
        "probe config: "
        f"lerobot_root={lerobot_root} video_backend={args.video_backend} "
        f"video_resize_mode={args.video_resize_mode} video_tolerance_s={args.video_tolerance_s} "
        f"resolution={args.resolution} min_video_frames={args.min_video_frames} "
        f"max_video_duration_s={args.max_video_duration_s} max_video_fps={args.max_video_fps} "
        f"video_feature_key={args.video_feature_key} video_feature_keywords={keywords} "
        f"num_workers={args.num_workers} prefetch_factor={args.prefetch_factor} "
        f"max_samples_per_batch={args.max_samples_per_batch} "
        f"decoder_cache_max_size={args.decoder_cache_max_size} "
        f"pin_memory={args.pin_memory} tokenizer={not args.no_tokenizer} "
        f"num_batches={args.num_batches} warmup={args.warmup}",
        file=sys.stderr,
        flush=True,
    )

    loader = build_sft_lerobot3_dataloader(
        lerobot_root=str(lerobot_root),
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        max_samples_per_batch=args.max_samples_per_batch,
        pin_memory=args.pin_memory,
        persistent_workers=True,
        tokenizer_config=tokenizer_config,
        video_backend=args.video_backend,
        video_resize_mode=args.video_resize_mode,
        video_tolerance_s=args.video_tolerance_s,
        resolution=args.resolution,
        min_video_frames=args.min_video_frames,
        max_video_duration_s=args.max_video_duration_s,
        max_video_fps=args.max_video_fps,
        video_feature_key=args.video_feature_key,
        video_feature_keywords=keywords,
        caption_key=args.caption_key,
        decoder_cache_max_size=args.decoder_cache_max_size,
    )

    header = "step\tparent_mib\tworker_mib\ttotal_mib\td_total_mib\tbatch_mib\tvideo_T\tsec\tworker_pids"
    print(header, flush=True)
    tsv_fh = None
    if args.tsv_out is not None:
        args.tsv_out.parent.mkdir(parents=True, exist_ok=True)
        tsv_fh = args.tsv_out.open("w")
        tsv_fh.write(header + "\n")

    rows: list[dict[str, float | int | None]] = []
    prev_total = None
    loader_iter = iter(loader)
    try:
        for step in range(1, args.num_batches + 1):
            t0 = time.perf_counter()
            batch = next(loader_iter)
            batch_bytes = _tensor_nbytes(batch)
            video_t = _video_t(batch)
            parent, worker, worker_pids = _rss_parent_and_children(proc)
            total = parent + worker
            dt = time.perf_counter() - t0
            delta = 0.0 if prev_total is None else _mib(total - prev_total)
            prev_total = total
            pid_s = ",".join(str(p) for p in worker_pids)
            line = (
                f"{step}\t{_mib(parent):.1f}\t{_mib(worker):.1f}\t{_mib(total):.1f}\t"
                f"{delta:.1f}\t{_mib(batch_bytes):.1f}\t{video_t if video_t is not None else ''}\t"
                f"{dt:.2f}\t{pid_s}"
            )
            print(line, flush=True)
            if tsv_fh is not None:
                tsv_fh.write(line + "\n")
                tsv_fh.flush()
            rows.append(
                {
                    "step": step,
                    "parent_mib": _mib(parent),
                    "worker_mib": _mib(worker),
                    "total_mib": _mib(total),
                    "batch_mib": _mib(batch_bytes),
                    "video_T": video_t,
                    "sec": dt,
                }
            )
            del batch
            gc.collect()
    finally:
        if tsv_fh is not None:
            tsv_fh.close()
        if dist.is_initialized():
            dist.destroy_process_group()

    post = [r for r in rows if int(r["step"]) > args.warmup]
    if len(post) < 2:
        print("verdict: not enough post-warmup steps to fit a slope", file=sys.stderr)
        return 0

    xs = [float(r["step"]) for r in post]
    totals = [float(r["total_mib"]) for r in post]
    parents = [float(r["parent_mib"]) for r in post]
    workers = [float(r["worker_mib"]) for r in post]
    slope = _linear_slope(xs, totals)
    parent_slope = _linear_slope(xs, parents)
    worker_slope = _linear_slope(xs, workers)
    delta_total = totals[-1] - totals[0]
    leak = slope > args.fail_slope_mib and delta_total > args.fail_delta_mib

    print(
        f"summary: n={len(rows)} warmup={args.warmup} "
        f"total_slope={slope:.2f} MiB/step parent_slope={parent_slope:.2f} "
        f"worker_slope={worker_slope:.2f} post_warmup_delta={delta_total:.1f} MiB "
        f"last_total={totals[-1]:.1f} MiB last_worker={workers[-1]:.1f} MiB "
        f"last_parent={parents[-1]:.1f} MiB",
        file=sys.stderr,
    )
    if abs(worker_slope) >= abs(parent_slope) and worker_slope > 1.0:
        loc = "worker-side cache / decoder / shard LRU"
    elif parent_slope > 1.0 and abs(parent_slope) > abs(worker_slope):
        loc = "main-process queue / packing buffer / unreleased batch refs"
    else:
        loc = "balanced parent+worker"
    if leak:
        print(
            f"verdict: LEAK ({loc}); slope {slope:.2f} MiB/step and "
            f"delta {delta_total:.1f} MiB exceed thresholds "
            f"{args.fail_slope_mib}/{args.fail_delta_mib}",
            file=sys.stderr,
        )
        return 1 if args.fail_on_leak else 0

    print(
        f"verdict: no leak ({loc}); post-warmup slope {slope:.2f} MiB/step "
        f"(video_backend={args.video_backend})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
