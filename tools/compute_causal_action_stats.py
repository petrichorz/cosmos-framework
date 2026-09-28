# SPDX-License-Identifier: OpenMDW-1.1
"""Fit merged state/action statistics over valid 32-action blocks (start stride 1).

Run from the repository root, in the training Python environment:
    python -m tools.compute_causal_action_stats \
        --dataset-root /path/to/canonical_55d --profile agibot \
        --output /path/to/block_stats.json

The default reservoir method scans ALL training blocks but estimates quantiles
with a uniform sample. --method exact retains all encoded values in RAM and is
intended for small datasets/reference checks. No videos or models are loaded.
A root may be a LeRobot v3 dataset or a parent containing multiple datasets.
All discovered datasets feed shared accumulators and produce ONE output JSON.
Mean/std/min/max use all valid values; only quantiles may be sampled.
--bounds selects q01_q99 (default) or min_max for ordinary-channel low/high.
Quaternion low/high are fixed at -1/+1; measured statistics remain unchanged.
The CLI defaults to --batch-size 512 --num-workers 4. Use 1/1 for serial reads.
Windows keep their original anchors; neither option changes the sampling stride.
"""

import argparse
import json
import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.data.generator.action.action_state_template import TemplateSourceContract, resolve_action_template
from cosmos_framework.data.generator.action.block_state import BlockStatistics, Quantiles, build_block_sample
from cosmos_framework.data.generator.action.causal_block_geometry import CausalBlockGeometry
from cosmos_framework.data.generator.action.lerobot_discovery import discover_dataset_roots
from cosmos_framework.data.generator.action.sample_contract import ActionReadOptions
from cosmos_framework.data.generator.action.segment_planner import SegmentPlanner


class QuantileAccumulator:
    """合并所有子集的有效值；逐通道计算全量矩统计和均匀蓄水池分位数。"""

    def __init__(self, *, method, capacity, seed):
        if method not in ("exact", "reservoir") or capacity < 1 or seed < 0:
            raise ValueError("Expected exact/reservoir method, positive capacity and nonnegative seed")
        self.method, self.capacity, self.seed = method, capacity, seed
        self.count = 0
        self.counts = None

    def update(self, values, masks):
        values = values.detach().cpu().float().numpy()
        masks = masks.detach().cpu().bool().numpy()
        if values.ndim != 2 or not len(values) or masks.shape != values.shape:
            raise ValueError("Expected nonempty [N, D] values and masks")
        if not np.isfinite(values[masks]).all():
            raise ValueError("Nonfinite value in a valid statistics channel")
        width = values.shape[1]
        if self.counts is None:
            self.counts = np.zeros(width, dtype=np.int64)
            self.mean = np.zeros(width, dtype=np.float64)
            self.m2 = np.zeros(width, dtype=np.float64)
            self.minimum = np.full(width, np.inf)
            self.maximum = np.full(width, -np.inf)
            self.rows = np.empty((self.capacity, width), dtype=np.float32) if self.method == "reservoir" else None
            self.chunks = [[] for _ in range(width)]
            self.rng = [np.random.default_rng(np.random.SeedSequence([self.seed, d])) for d in range(width)]
        elif width != len(self.counts):
            raise ValueError("Cannot merge different template widths")
        self.count += len(values)
        for d in range(width):
            # 不同子集的 mask 可以不同；无效补零不进入该通道的计数、矩统计或蓄水池。
            x = values[masks[:, d], d].astype(np.float64)
            if not len(x):
                continue
            old, n = int(self.counts[d]), len(x)
            total = old + n
            batch_mean = x.mean()
            delta = batch_mean - self.mean[d]
            # 合并每批原始有效值的均值与中心二阶矩，包含批间均值差；std 使用总体分母 N。
            self.m2[d] += np.square(x - batch_mean).sum() + delta * delta * old * n / total
            self.mean[d] += delta * n / total
            self.minimum[d] = min(self.minimum[d], x.min())
            self.maximum[d] = max(self.maximum[d], x.max())
            if self.method == "exact":
                self.chunks[d].append(x.astype(np.float32))
            else:
                fill = min(n, max(0, self.capacity - old))
                self.rows[old : old + fill, d] = x[:fill]
                remaining = x[fill:]
                if len(remaining):
                    # Algorithm R 对该通道跨所有子集的有效值连续采样，不在子集边界重置。
                    slots = self.rng[d].integers(0, np.arange(old + fill + 1, total + 1))
                    selected = np.flatnonzero(slots < self.capacity)
                    # 批内同一槽位可多次命中，仅保留最后一次写入，等价于逐条更新。
                    _, last = np.unique(slots[selected[::-1]], return_index=True)
                    selected = selected[::-1][last]
                    self.rows[slots[selected], d] = remaining[selected]
            self.counts[d] = total

    def finalize(self, rotation_groups=(), *, bounds="q01_q99"):
        if bounds not in ("q01_q99", "min_max"):
            raise ValueError("bounds must be q01_q99 or min_max")
        if not self.count:
            raise ValueError("Statistics population is empty")
        valid = self.counts > 0
        qs = np.zeros((5, len(valid)), dtype=np.float64)
        for d in np.flatnonzero(valid):
            values = (
                np.concatenate(self.chunks[d])
                if self.method == "exact"
                else self.rows[: min(self.counts[d], self.capacity), d]
            )
            qs[:, d] = np.quantile(values, [0.01, 0.10, 0.50, 0.90, 0.99], method="linear")
        # 仅选择归一化边界，所有实测统计仍完整保存；min/max 来自全量有效值。
        if bounds == "min_max":
            low = np.where(valid, self.minimum, 0).astype(np.float32)
            high = np.where(valid, self.maximum, 0).astype(np.float32)
        else:
            low, high = qs[0].astype(np.float32), qs[4].astype(np.float32)
        # q01/q99 保留实测值；只有有效四元数通道的归一化边界固定为理论范围。
        for group in rotation_groups:
            if len(group) != 4:
                raise ValueError("Fixed quaternion bounds require four-component rotation groups")
            indexes = np.asarray(group)
            if valid[indexes].any() and not valid[indexes].all():
                raise ValueError("Incomplete quaternion statistics coverage")
            if valid[indexes].all():
                low[indexes], high[indexes] = -1, 1
        metrics = {
            "min": np.where(valid, self.minimum, 0).tolist(),
            "max": np.where(valid, self.maximum, 0).tolist(),
            "mean": self.mean.tolist(),
            "std": np.sqrt(np.maximum(0, self.m2 / np.maximum(self.counts, 1))).tolist(),
            "count": [self.count],
            "valid_counts": self.counts.tolist(),
            **{k: v.tolist() for k, v in zip(("q01", "q10", "q50", "q90", "q99"), qs)},
        }
        return Quantiles(torch.from_numpy(low), torch.from_numpy(high), torch.from_numpy(valid), metrics=metrics)

    def summary(self):
        sampled = self.counts if self.method == "exact" else np.minimum(self.counts, self.capacity)
        return {
            "rows": self.count,
            "valid_counts": self.counts.tolist(),
            "sampled_counts": sampled.tolist(),
            "approximate": bool((sampled < self.counts).any()),
        }


def _iter_encoded_batches(dataset, stop, *, batch_size, num_workers):
    """批量读取相邻滑窗，并按原顺序返回编码结果；线程不修改统计累积器。"""
    from cosmos_framework.data.generator.action.datasets.segment_lerobot_dataset import SegmentLeRobotDataset

    batched_reader = isinstance(dataset, SegmentLeRobotDataset) and dataset.video_view is None

    def ranges():
        # 不跨 episode 合并；limit 仍按原始窗口计数，最后一批可不足 batch_size。
        if batched_reader:
            for start, count in dataset.get_shuffle_blocks():
                end = min(start + count, stop)
                for index in range(start, end, batch_size):
                    yield index, min(batch_size, end - index)
                if end == stop:
                    break
        else:
            for index in range(stop):
                yield index, 1

    def encode(item):
        index, count = item
        if batched_reader:
            records = dataset._segments[index : index + count]
            ds, episode, first, width = records[0]
            # 每个任务使用独立索引视图，共享只读 parquet；不改原 Reader 的索引和 LRU。
            view = copy(dataset)
            view._loaded_lru = dataset._loaded_lru.copy()
            view._segments = [(ds, episode, first, records[-1][2] - first + width)]
            raw = view[0]
            offsets = torch.tensor([record[2] - first for record in records])
            indexes = offsets[:, None] + torch.arange(width)[None, :]
            # 重叠窗口分别展开为 block，保留各自 anchor；最后一帧只满足 T+1 契约。
            state = raw["state_trajectory"]
            raw = dict(
                state_trajectory=torch.cat((state[indexes].flatten(0, 1), state[-1:])),
                action_target=raw["action_target"][indexes].flatten(0, 1),
                state_mask=raw["state_mask"],
                action_mask=raw["action_mask"],
                source_contract=raw["source_contract"],
                conditioning_fps=raw["conditioning_fps"],
            )
        else:
            raw = dataset[index]
        if raw["source_contract"].split != "train":
            raise ValueError("Statistics must be fitted on the training split")
        data, metadata = build_block_sample(raw, template=dataset.template, planner=dataset.planner, history_blocks=1)
        return count, metadata.states, metadata.state_mask, data["action"], metadata.action_mask

    items = iter(ranges())
    if num_workers == 1:
        for item in items:
            yield encode(item)
        return
    # 最多保留 num_workers 个任务，避免把全数据集的 Future/编码张量积存在内存。
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        pending = deque()
        try:
            for _ in range(num_workers):
                item = next(items, None)
                if item is not None:
                    pending.append(pool.submit(encode, item))
            while pending:
                yield pending.popleft().result()
                item = next(items, None)
                if item is not None:
                    pending.append(pool.submit(encode, item))
        finally:
            for future in pending:
                future.cancel()


def compute_statistics(
    datasets,
    *,
    method="reservoir",
    reservoir_size=50_000,
    seed=42,
    limit=None,
    log_every=1000,
    bounds="q01_q99",
    batch_size=1,
    num_workers=1,
):
    """每个子集遍历一次，共用两套累积器；limit 是每个子集的调试 block 上限。"""
    if bounds not in ("q01_q99", "min_max"):
        raise ValueError("bounds must be q01_q99 or min_max")
    if hasattr(datasets, "template"):
        datasets = (datasets,)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if batch_size < 1 or num_workers < 1:
        raise ValueError("batch_size and num_workers must be positive")
    state = QuantileAccumulator(method=method, capacity=reservoir_size, seed=seed)
    action = QuantileAccumulator(method=method, capacity=reservoir_size, seed=seed + 1)
    sources = []
    template = geometry = None
    for dataset in datasets:
        g = dataset.planner.geometry
        if (
            dataset.planner.max_action_steps != g.actions_per_block
            or dataset.planner.overlap_action_steps != g.actions_per_block - 1
        ):
            raise ValueError("Expected single-block windows with start stride 1")
        if template is None:
            template, geometry = dataset.template, g
        elif (
            dataset.template.template_id != template.template_id
            or dataset.template.width != template.width
            or g != geometry
        ):
            raise ValueError("Merged datasets must share a template and block geometry")
        stop = len(dataset) if limit is None else min(limit, len(dataset))
        source = {
            "blocks": stop,
            "available_blocks": len(dataset),
            "skipped_ranges": dataset.planner.skipped_ranges,
            "discarded_action_steps": dataset.planner.discarded_action_steps,
        }
        if hasattr(dataset, "source_contract"):
            source["source_contract"] = asdict(dataset.source_contract)
        if hasattr(dataset, "read_options"):
            source["read_options"] = asdict(dataset.read_options)
        source_started = time.monotonic()
        logging.info("Scanning source %d: %d/%d blocks", len(sources) + 1, stop, len(dataset))
        if stop:
            # 在线程启动前完成惰性 parquet / mask 初始化，避免并发初始化同一个 Reader。
            raw = dataset[0]
            source["state_mask"] = raw["state_mask"].tolist()
            source["action_mask"] = raw["action_mask"].tolist()
            del raw
        processed = 0
        for count, states, state_mask, actions, action_mask in _iter_encoded_batches(
            dataset, stop, batch_size=batch_size, num_workers=num_workers
        ):
            # 累积器和随机抽样只在主线程按窗口顺序更新；exact/reservoir 共用此路径。
            state.update(states, state_mask)
            action.update(actions, action_mask)
            previous, processed = processed, processed + count
            if log_every and (processed // log_every > previous // log_every or processed == stop):
                elapsed = time.monotonic() - source_started
                rate = processed / max(elapsed, 1e-9)
                # ETA 仅估计当前子集的扫描时间，不包含最后的分位数计算和写盘。
                logging.info(
                    "Source %d: %d/%d blocks (%.1f%%); %.1f blocks/s; elapsed %s; scan ETA %s; "
                    "merged state rows %d; action rows %d",
                    len(sources) + 1,
                    processed,
                    stop,
                    100 * processed / stop,
                    rate,
                    timedelta(seconds=int(elapsed)),
                    timedelta(seconds=int((stop - processed) / rate)),
                    state.count,
                    action.count,
                )
        sources.append(source)
    if template is None:
        raise ValueError("No datasets supplied")
    logging.info("Scan complete; finalizing quantiles and normalization bounds")
    rotations = getattr(template, "rotation_groups", ())
    statistics = BlockStatistics(state.finalize(rotations, bounds=bounds), action.finalize(rotations, bounds=bounds))
    blocks = sum(s["blocks"] for s in sources)
    available = sum(s["available_blocks"] for s in sources)
    statistics.provenance = {
        "kind": "fitted",
        "population": "all_valid_single_block_starts",
        "start_stride": 1,
        "split": "train",
        "template_id": template.template_id,
        "geometry": asdict(geometry),
        "max_action_steps": geometry.actions_per_block,
        "overlap_action_steps": geometry.actions_per_block - 1,
        "blocks": blocks,
        "available_blocks": available,
        "partial": blocks < available,
        "skipped_ranges": sum(s["skipped_ranges"] for s in sources),
        "discarded_action_steps": sum(s["discarded_action_steps"] for s in sources),
        "sources": sources,
        "method": method,
        "quantiles": [0.01, 0.10, 0.50, 0.90, 0.99],
        "interpolation": "linear",
        "std_ddof": 0,
        "seed": seed,
        "reservoir_size": reservoir_size if method == "reservoir" else None,
        "sampling": "per_channel_uniform_valid_values",
        "bounds": bounds,
        "quaternion_bounds": [-1, 1],
        "rotation_groups": [list(g) for g in rotations],
        "batch_size": batch_size,
        "num_workers": num_workers,
        "state": state.summary(),
        "action": action.summary(),
    }
    return statistics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        nargs="+",
        required=True,
        help="LeRobot v3 roots or parent directories; all children are merged",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--profile", choices=("agibot", "egosuite"))
    source.add_argument("--source-contract", type=Path, help="Explicit TemplateSourceContract JSON")
    parser.add_argument("--template", help="Fully qualified ActionStateTemplate class (default: unified55)")
    parser.add_argument("--state-key", default="state_unified")
    parser.add_argument("--action-key", default="action_unified")
    parser.add_argument("--state-mask-key", default="mask_state")
    parser.add_argument("--action-mask-key", default="mask_action")
    parser.add_argument("--action-from-state", action="store_true", help="Use state as target; offset remains explicit")
    parser.add_argument("--action-time-offset-steps", type=int, default=0)
    parser.add_argument("--split-val-ratio", type=float, default=0.0, help="Match training split; 0 uses all episodes")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--method", choices=("exact", "reservoir"), default="reservoir")
    parser.add_argument(
        "--bounds",
        choices=("q01_q99", "min_max"),
        default="q01_q99",
        help="Statistics used for state/action low/high; quaternion bounds remain -1/+1",
    )
    parser.add_argument("--reservoir-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="Debug only: first N blocks PER dataset; output marked partial")
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512, help="Adjacent windows per read, within one episode")
    parser.add_argument("--num-workers", type=int, default=4, help="Read/encode threads; accumulation stays ordered")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.split_val_ratio < 1:
        parser.error("--split-val-ratio must be in [0, 1)")
    if args.reservoir_size < 1 or args.seed < 0 or args.log_every < 0:
        parser.error("reservoir size must be positive; seed and log interval must be nonnegative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.batch_size < 1 or args.num_workers < 1:
        parser.error("--batch-size and --num-workers must be positive")
    return args


def main():
    started = time.monotonic()
    args = parse_args()
    # 外层线程并行，避免每个任务再启动大量 PyTorch 算子线程。
    torch.set_num_threads(1)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from cosmos_framework.data.generator.action.datasets.segment_lerobot_dataset import SegmentLeRobotDataset

    roots = discover_dataset_roots(args.dataset_root)
    template = resolve_action_template(args.template)
    # 默认直接读 action_unified[t]。若改用 state[t+1]，必须同时指定
    # --action-from-state 和 --action-time-offset-steps 1；字段切换不会自动增加偏移。
    options = ActionReadOptions(
        state_key=args.state_key,
        action_key=args.action_key,
        state_mask_key=args.state_mask_key,
        action_mask_key=args.action_mask_key,
        action_from_state=args.action_from_state,
        action_time_offset_steps=args.action_time_offset_steps,
    )

    def readers():
        # 顺序构造子集，避免同时保留全部来源的 parquet 缓存与片段索引。
        for source_index, root in enumerate(roots, 1):
            logging.info("Preparing source %d/%d: %s", source_index, len(roots), root)
            if args.source_contract:
                contract = replace(
                    TemplateSourceContract(**json.loads(args.source_contract.read_text())), source=str(root)
                )
            else:
                contract = template.source_contract(
                    args.profile,
                    source=str(root),
                    info=json.loads((root / "meta/info.json").read_text()),
                    target_semantics=f"absolute target from {options.target_key}, offset={options.action_time_offset_steps} steps",
                )
            geometry = CausalBlockGeometry(temporal_compression_factor=4)
            # 32-action block、overlap=31，对应 stride=1；仅读数值，不解码视频。
            yield SegmentLeRobotDataset(
                root=root,
                template=template,
                source_contract=contract,
                planner=SegmentPlanner(
                    max_action_steps=geometry.actions_per_block,
                    overlap_action_steps=geometry.actions_per_block - 1,
                    geometry=geometry,
                ),
                read_options=options,
                split="train",
                split_seed=args.split_seed,
                split_val_ratio=args.split_val_ratio,
                video_view=None,
            )

    logging.info("Merging %d LeRobot v3 datasets", len(roots))
    statistics = compute_statistics(
        readers(),
        method=args.method,
        reservoir_size=args.reservoir_size,
        seed=args.seed,
        limit=args.limit,
        log_every=args.log_every,
        bounds=args.bounds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    statistics.provenance.update(
        dataset_roots=[str(p) for p in roots],
        read_options=asdict(options),
        split_seed=args.split_seed,
        split_val_ratio=args.split_val_ratio,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 写一个独立 JSON；不修改任何子集的 meta/stats.json，也不改训练配置。
    statistics.save(args.output)
    logging.info(
        "Wrote %s; blocks=%d; partial=%s; total elapsed %s",
        args.output,
        statistics.provenance["blocks"],
        statistics.provenance["partial"],
        timedelta(seconds=int(time.monotonic() - started)),
    )


if __name__ == "__main__":
    main()
