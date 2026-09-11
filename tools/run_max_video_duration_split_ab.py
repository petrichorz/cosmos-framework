#!/usr/bin/env python3
"""Run a reproducible 4-rank full-episode versus overlapping-split A/B test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_DRIVER = REPO_ROOT / "tools/profile_cosmos_ascend.py"
DATASET = Path("/mnt/sfs_turbo/public/datasets/egosuite_demo_v1")
OUTPUT_ROOT = Path("/mnt/sfs_turbo/zheng/cosmos-ascend-profile/cosmos-profile-logs/max_video_duration_split_ab")
LAUNCH_SCRIPT = Path(
    "/mnt/sfs_turbo/zheng/cosmos-ascend-profile/cosmos/"
    "cookbooks/cosmos3/generator/audiovisual/finetune/launch_sft_vision_edge_profile_local.sh"
)

VARIANTS = {
    "A_full_episode": (0.0, "drop", 0.0),
    "B_split_61s_overlap_5s": (61.0, "split", 5.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("timing", "profile"), required=True)
    parser.add_argument("--variant", action="append", choices=tuple(VARIANTS), dest="variants")
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--master-port-base", type=int, default=50420)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = args.variants or list(VARIANTS)
    session = args.output_root / time.strftime("%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=False)
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    git_diff = subprocess.check_output(["git", "diff", "--binary"], cwd=REPO_ROOT)
    git_status = subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True).splitlines()
    results = []

    for index, name in enumerate(variants):
        max_duration, policy, overlap = VARIANTS[name]
        run_root = session / name / args.phase
        run_root.mkdir(parents=True)
        env = os.environ.copy()
        env["MASTER_PORT"] = str(args.master_port_base + index)
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        command = [
            sys.executable,
            str(PROFILE_DRIVER),
            "--mode",
            "baseline" if args.phase == "timing" else "distributed",
            "--launch-script",
            str(LAUNCH_SCRIPT),
            "--dataset",
            str(DATASET),
            "--output-root",
            str(run_root),
            "--nproc-per-node",
            str(args.nproc_per_node),
            "--num-workers",
            "8",
            "--video-backend",
            "pyav",
            "--video-resize-mode",
            "decode_transform",
            "--max-video-fps",
            "15",
            "--seed",
            "42",
            "--extra-override",
            "dataloader_train.max_sequence_length=45056",
            "--extra-override",
            f"dataloader_train.dataloader.datasets.video.dataset.max_video_duration_s={max_duration}",
            "--extra-override",
            f"dataloader_train.dataloader.datasets.video.dataset.long_video_policy={policy}",
            "--extra-override",
            f"dataloader_train.dataloader.datasets.video.dataset.video_window_overlap_s={overlap}",
        ]
        if args.phase == "timing":
            command.extend(["--max-steps", "7"])
        else:
            command.extend(
                [
                    "--max-steps",
                    "5",
                    "--profile-step",
                    "4",
                    "--profile-warmup",
                    "2",
                    "--profile-active-steps",
                    "2",
                    "--profiler-level",
                    "level1",
                    "--aic-metrics",
                    "pipe",
                    "--profile-memory",
                    "--sync-analysis",
                    "--no-with-stack",
                    "--no-with-modules",
                ]
            )

        manifest = {
            "variant": name,
            "phase": args.phase,
            "max_video_duration_s": max_duration,
            "long_video_policy": policy,
            "video_window_overlap_s": overlap,
            "command": command,
            "python": sys.executable,
            "git_head": git_head,
            "git_diff_sha256": hashlib.sha256(git_diff).hexdigest(),
            "git_status": git_status,
            "dataset": str(DATASET),
            "nproc_per_node": args.nproc_per_node,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\n===== {name} / {args.phase} =====", flush=True)
        completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
        manifest["returncode"] = completed.returncode
        manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        results.append({"variant": name, "returncode": completed.returncode, "path": str(run_root)})

    (session / f"{args.phase}_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Session: {session}")
    if any(result["returncode"] != 0 for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
