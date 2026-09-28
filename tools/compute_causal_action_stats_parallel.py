# SPDX-License-Identifier: OpenMDW-1.1
"""Run one existing statistics CLI process per dataset, then aggregate results.

python -m tools.compute_causal_action_stats_parallel --dataset-root /data/parent \
    --profile agibot --dataset-processes 4 --num-workers 4 --output outputs/stats.json
Each child writes meta/causal_action_stats.json; meta/stats.json is untouched.
Existing child results are recomputed, never silently reused after data changes.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from cosmos_framework.data.generator.action.lerobot_discovery import discover_dataset_roots
from tools.aggregate_causal_action_stats import STATS_RELATIVE_PATH, aggregate_statistics
from tools.compute_causal_action_stats import parse_args


def compute_child(root, args):
    """独立 Python 进程运行现有多线程实现；成功后原子替换本地统计文件。"""
    output = root / STATS_RELATIVE_PATH
    fd, temporary = tempfile.mkstemp(prefix=".causal_stats_", suffix=".json", dir=output.parent)
    os.close(fd)
    command = [
        sys.executable,
        "-m",
        "tools.compute_causal_action_stats",
        "--dataset-root",
        str(root),
        "--output",
        temporary,
    ]
    excluded = {"dataset_root", "output", "dataset_processes"}
    for key, value in vars(args).items():
        if key in excluded or value is None or value is False:
            continue
        command.append("--" + key.replace("_", "-"))
        if value is not True:
            command.append(str(Path(value).resolve()) if key == "source_contract" else str(value))
    try:
        logging.info("Starting %s", root)
        subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1])
        # 新流程只保存 q01/q99；不改变原始单进程脚本的输出兼容性。
        data = json.loads(Path(temporary).read_text())
        for kind in ("state", "action"):
            for key in ("q10", "q50", "q90"):
                data[kind].pop(key, None)
        data["provenance"]["quantiles"] = [0.01, 0.99]
        Path(temporary).write_text(json.dumps(data, indent=2) + "\n")
        os.replace(temporary, output)
        return output
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    args = parse_args(parallel=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    roots = discover_dataset_roots(args.dataset_root)
    paths = [root / STATS_RELATIVE_PATH for root in roots]
    if args.output.resolve() in paths:
        raise ValueError("Output must not overwrite per-dataset statistics")
    # 调度线程只等待子进程；每个子进程有独立累积器及 num_workers 个编码线程。
    with (
        logging_redirect_tqdm(),
        tqdm(total=len(roots), desc="Dataset statistics", unit="dataset", dynamic_ncols=True) as progress,
        ThreadPoolExecutor(max_workers=args.dataset_processes) as pool,
    ):
        futures = {pool.submit(compute_child, root, args): root for root in roots}
        try:
            for future in as_completed(futures):
                future.result()
                # 仅成功完成并保存子集统计后计数，失败的任务不计入完成数。
                root = futures[future]
                progress.set_postfix_str(f"done={root.parent.name}/{root.name}", refresh=False)
                progress.update(1)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    result = aggregate_statistics(paths, bounds=args.bounds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    logging.info("Wrote aggregate %s; partial=%s", args.output, result.provenance["partial"])


if __name__ == "__main__":
    main()
