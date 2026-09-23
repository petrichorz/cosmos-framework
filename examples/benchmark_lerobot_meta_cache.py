# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""评测 LeRobot metadata 索引缓存的加载加速效果。

跑两遍 ``_load_lerobot_metadata``，对比「冷启动（重建缓存）」与「热启动（命中缓存）」
的耗时，并给出每遍各阶段的时间占比（discover / fingerprint / 读 parquet / 提取 /
写缓存 / 读缓存 / 过滤）。

用法（需在 conda 环境、repo 根目录下运行）::

    # 目录 / 父目录数据集
    python examples/benchmark_lerobot_meta_cache.py \
        /mnt/xxx/egosuite_demo_v1 \
        --cache-dir /data5T/liujin/lerobot_meta_cache \
        --long-video-policy split

    # jsonl manifest 数据集
    python examples/benchmark_lerobot_meta_cache.py \
        /mnt/xxx/manifest.jsonl \
        --cache-dir /data5T/liujin/lerobot_meta_cache \
        --long-video-policy split

说明：
- 支持目录 / 父目录，也支持 ``.jsonl`` manifest（按后缀自动分流）。
- 默认第一遍前清空该数据集的缓存，保证冷启动；用 ``--keep-cache`` 可跳过清空。
- 阶段耗时通过给内部函数加计时 wrapper 得到（不修改主代码），过滤耗时用「总耗时 - 已知阶段」反推。
"""

import argparse
import gc
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path


# 阶段显示顺序（第一遍含 meta_load/extract/write_cache，第二遍含 read_cache）
_STAGE_ORDER = [
    "discover",
    "fingerprint",
    "meta_load",
    "extract",
    "write_cache",
    "read_cache",
    "filter",
]
_STAGE_LABELS = {
    "discover": "发现数据集目录",
    "fingerprint": "计算指纹(stat parquet)",
    "meta_load": "读 parquet(构造 metadata)",
    "extract": "提取 episode 字段",
    "write_cache": "写缓存 index.json",
    "read_cache": "读缓存 index.json",
    "filter": "过滤 + split",
}


def _patch_timers(mod, timings):
    """给模块内函数加计时 wrapper，累计到 timings dict。返回恢复函数。"""
    orig = {}
    targets = {
        "discover": "_discover_lerobot_roots",
        "fingerprint": "compute_fingerprint",
        "read_cache": "read_cache",
        "write_cache": "write_cache",
        "meta_load": "LeRobotDatasetMetadata",
        "extract": "_extract_raw_episodes",
    }

    def _make(key, fn):
        def _wrap(*a, **k):
            t = time.perf_counter()
            r = fn(*a, **k)
            timings[key] += time.perf_counter() - t
            return r

        return _wrap

    for key, name in targets.items():
        orig[name] = getattr(mod, name)
        setattr(mod, name, _make(key, orig[name]))

    def _restore():
        for name, fn in orig.items():
            setattr(mod, name, fn)

    return _restore


def _parse_manifest_paths(manifest_path: str) -> list[str]:
    """解析 jsonl manifest，返回每行的 ``path``（跳过空行 / 缺 path 的行）。"""
    paths = []
    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("path"):
                paths.append(entry["path"])
    return paths


def _run_once(mod, args, timings):
    """跑一次 metadata 加载（目录或 jsonl manifest），返回 (总耗时, 数据集数, 有效clip数, 阶段耗时dict)。"""
    timings.clear()
    gc.collect()
    is_manifest = args.dataset_path.endswith(".jsonl")
    t0 = time.perf_counter()
    if is_manifest:
        sources, idx = mod._load_lerobot_metadata_from_manifest(
            args.dataset_path,
            min_video_frames=args.min_video_frames,
            max_video_duration_s=args.max_video_duration_s,
            min_short_edge=args.min_short_edge,
            video_feature_key=args.video_feature_key,
            caption_key=args.caption_key,
            video_feature_keywords=args.video_feature_keywords,
            long_video_policy=args.long_video_policy,
            video_window_overlap_s=args.video_window_overlap_s,
        )
    else:
        sources, idx = mod._load_lerobot_metadata(
            args.dataset_path,
            min_video_frames=args.min_video_frames,
            max_video_duration_s=args.max_video_duration_s,
            min_short_edge=args.min_short_edge,
            video_feature_key=args.video_feature_key,
            caption_key=args.caption_key,
            video_feature_keywords=args.video_feature_keywords,
            long_video_policy=args.long_video_policy,
            video_window_overlap_s=args.video_window_overlap_s,
        )
    total = time.perf_counter() - t0

    known = (
        timings["discover"]
        + timings["fingerprint"]
        + timings["meta_load"]
        + timings["extract"]
        + timings["write_cache"]
        + timings["read_cache"]
    )
    timings["filter"] = max(total - known, 0.0)
    return total, len(sources), len(idx), dict(timings)


def _print_stage_report(label, total, timings):
    print(f"\n--- {label} ---")
    print(f"{'阶段':<22} {'耗时(s)':>12} {'占比':>8}")
    print("-" * 44)
    for key in _STAGE_ORDER:
        v = timings.get(key, 0.0)
        if v <= 0 and key not in ("discover", "fingerprint", "filter"):
            continue  # 本遍未发生的阶段不打印
        pct = (v / total * 100) if total > 0 else 0.0
        print(f"{_STAGE_LABELS[key]:<22} {v:>12.3f} {pct:>7.1f}%")
    print("-" * 44)
    print(f"{'总计':<22} {total:>12.3f} {'100.0%':>8}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_path", help="LeRobot 数据集根目录（单数据集或父目录）")
    parser.add_argument("--cache-dir", default=None, help="缓存根目录；默认读 LEROBOT_META_CACHE_DIR，再缺省 ~/.cache/...")
    parser.add_argument("--min-video-frames", type=int, default=61)
    parser.add_argument("--max-video-duration-s", type=float, default=61.0)
    parser.add_argument("--long-video-policy", default="drop", choices=["drop", "split"])
    parser.add_argument("--video-window-overlap-s", type=float, default=0.0)
    parser.add_argument("--min-short-edge", type=int, default=0)
    parser.add_argument("--video-feature-key", default=None)
    parser.add_argument("--caption-key", default="caption")
    parser.add_argument("--video-feature-keywords", nargs="*", default=None)
    parser.add_argument("--keep-cache", action="store_true", help="第一遍前不清空缓存（默认清空，保证冷启动）")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if args.cache_dir:
        os.environ["LEROBOT_META_CACHE_DIR"] = args.cache_dir

    # 在设置好环境变量后再 import，保证本地 lerobot 断言与缓存根读取正确
    import cosmos_framework.data.generator.local_datasets.sft_dataset_lerobot3 as mod
    from cosmos_framework.data.generator.local_datasets.lerobot_meta_cache import get_cache_root

    cache_root = get_cache_root()
    is_manifest = args.dataset_path.endswith(".jsonl")

    # 计算需要清空的缓存目录：目录模式是「缓存根/<数据集名>」；manifest 模式是
    # 每个 path 各自的「缓存根/<path 的目录名>」。
    if is_manifest:
        manifest_paths = _parse_manifest_paths(args.dataset_path)
        top_cache_dirs = [Path(cache_root) / Path(p).name for p in manifest_paths]
    else:
        top_cache_dirs = [Path(cache_root) / Path(args.dataset_path).name]

    print("=" * 60)
    print(f"数据集路径: {args.dataset_path}" + ("  (jsonl manifest)" if is_manifest else ""))
    print(f"缓存根目录: {cache_root}")
    if is_manifest:
        print(f"manifest 含 {len(manifest_paths)} 个数据集 path")
    print(f"本数据集缓存目录: {top_cache_dirs[0] if len(top_cache_dirs) == 1 else [str(d) for d in top_cache_dirs]}")
    print(f"过滤参数: min_frames={args.min_video_frames}, max_dur={args.max_video_duration_s}s, "
          f"policy={args.long_video_policy}, overlap={args.video_window_overlap_s}s")
    print("=" * 60)

    if not args.keep_cache:
        for d in top_cache_dirs:
            shutil.rmtree(d, ignore_errors=True)
        print(f"[清空] 已清空缓存目录，保证冷启动: {[str(d) for d in top_cache_dirs]}")

    timings = defaultdict(float)
    restore = _patch_timers(mod, timings)
    try:
        t1, nsrc1, nidx1, tm1 = _run_once(mod, args, timings)
        t2, nsrc2, nidx2, tm2 = _run_once(mod, args, timings)
    finally:
        restore()

    _print_stage_report(f"第 1 遍（冷启动：重建缓存）", t1, tm1)
    _print_stage_report(f"第 2 遍（热启动：命中缓存）", t2, tm2)

    print("\n" + "=" * 60)
    print("对比汇总")
    print("=" * 60)
    print(f"数据集数       : {nsrc1}（两遍一致: {nsrc1 == nsrc2}）")
    print(f"有效 clip 数   : {nidx1}（两遍一致: {nidx1 == nidx2}）")
    print(f"第 1 遍总耗时  : {t1:.3f} s")
    print(f"第 2 遍总耗时  : {t2:.3f} s")
    if t2 > 0:
        print(f"加速比         : {t1 / t2:.1f}x")
        print(f"节省时间       : {t1 - t2:.3f} s")


if __name__ == "__main__":
    main()
