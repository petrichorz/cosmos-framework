#!/usr/bin/env python3
"""Run the fixed 8-rank EgoSuite training fusion A/B matrix sequentially."""

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
OUTPUT_ROOT = Path("/mnt/sfs_turbo/zheng/cosmos-ascend-profile/cosmos-profile-logs/egosuite_fusion_ab")
LAUNCH_SCRIPT = Path(
    "/mnt/sfs_turbo/zheng/cosmos-ascend-profile/cosmos/"
    "cookbooks/cosmos3/generator/audiovisual/finetune/launch_sft_vision_edge_profile_local.sh"
)

FLAGS = (
    "COSMOS_ASCEND_SEQUENCE_PACKING_TOLIST_OPT",
    "COSMOS_ASCEND_FUSED_TEACHER_FORCING_ATTENTION",
    "COSMOS_ASCEND_FUSED_ROPE",
    "COSMOS_ASCEND_FUSED_RMSNORM",
)

VARIANTS = {
    "E0_original": (0, 0, 0, 0),
    "E1_tolist": (1, 0, 0, 0),
    "E2_attention": (1, 1, 0, 0),
    "E3_rope": (1, 0, 1, 0),
    "E4_rmsnorm": (1, 0, 0, 1),
    # Best single-candidate set after the fixed 8-rank timing gate. Explicit
    # attention is excluded: it saves allocated memory but regresses step time.
    "E5_combined": (1, 0, 1, 1),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("timing", "profile"), required=True)
    parser.add_argument("--variant", action="append", choices=tuple(VARIANTS), dest="variants")
    parser.add_argument("--nproc-per-node", type=int, choices=(4, 8), default=8)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--master-port-base", type=int, default=50320)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = args.variants or list(VARIANTS)
    session = args.output_root / time.strftime("%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=False)
    results = []
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    git_diff = subprocess.check_output(["git", "diff", "--binary"], cwd=REPO_ROOT)
    git_status = subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True).splitlines()

    for index, name in enumerate(variants):
        values = VARIANTS[name]
        run_root = session / name / args.phase
        run_root.mkdir(parents=True)
        env = os.environ.copy()
        env.update({flag: str(value) for flag, value in zip(FLAGS, values, strict=True)})
        env["MASTER_PORT"] = str(args.master_port_base + index)
        # The py312 environment also contains a wheel snapshot. Force the experiment
        # to import this worktree so every A/B switch and measurement hook is active.
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
            "--decoder-cache-size",
            "64",
            "--video-backend",
            "pyav",
            "--video-resize-mode",
            "decode_transform",
            "--seed",
            "42",
            "--extra-override",
            "dataloader_train.max_sequence_length=45056",
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
                    "--record-shapes",
                    "--profile-memory",
                    "--sync-analysis",
                    "--no-with-stack",
                    "--no-with-modules",
                ]
            )

        manifest = {
            "variant": name,
            "phase": args.phase,
            "flags": dict(zip(FLAGS, values, strict=True)),
            "command": command,
            "python": sys.executable,
            "pythonpath": env["PYTHONPATH"],
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
