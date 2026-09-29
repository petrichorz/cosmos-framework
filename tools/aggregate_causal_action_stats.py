# SPDX-License-Identifier: OpenMDW-1.1
"""Aggregate per-dataset statistics without reading trajectories.

python -m tools.aggregate_causal_action_stats --dataset-root /data/parent --output outputs/stats.json
Global quantiles cannot be recovered from local quantiles. q01/q99 store the minimum source q01 and maximum source q99, respectively.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.data.generator.action.block_state import BlockStatistics, Quantiles
from cosmos_framework.data.generator.action.lerobot_discovery import discover_dataset_roots

STATS_RELATIVE_PATH = Path("meta/causal_action_stats.json")


def aggregate_statistics(paths, *, bounds="q01_q99"):
    """合并全量矩；分位数只保存子集外包范围，不冒充总体分位数。"""
    if bounds not in ("q01_q99", "min_max"):
        raise ValueError("Unknown bounds")
    paths = [Path(p).resolve() for p in paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Expected nonempty, distinct statistics paths")
    records = [json.loads(p.read_text()) for p in paths]
    reference = records[0]["provenance"]
    contract_keys = (
        "template_id",
        "geometry",
        "population",
        "start_stride",
        "split",
        "std_ddof",
        "rotation_groups",
        "read_options",
        "split_seed",
        "split_val_ratio",
    )
    roots = []
    for record in records:
        provenance = record["provenance"]
        if provenance.get("method") not in ("exact", "reservoir") or provenance.get("std_ddof") != 0:
            raise ValueError("Expected directly fitted population statistics")
        for key in contract_keys:
            if key not in provenance or provenance[key] != reference[key]:
                raise ValueError(f"Incompatible statistics: {key}")
        roots.extend(provenance["dataset_roots"])
    if len(set(roots)) != len(roots):
        raise ValueError("Overlapping source datasets")

    def merge(kind):
        width = len(records[0][kind]["valid_counts"])
        counts = np.zeros(width, dtype=np.int64)
        mean, m2 = np.zeros(width), np.zeros(width)
        minimum, maximum = np.full(width, np.inf), np.full(width, -np.inf)
        qs = ("q01", "q99")
        qmin = {q: np.full(width, np.inf) for q in qs}
        qmax = {q: np.full(width, -np.inf) for q in qs}
        rows = 0
        for record in records:
            data = record[kind]
            n = np.asarray(data["valid_counts"])
            row_counts = data["count"]
            if (
                not isinstance(row_counts, list)
                or len(row_counts) != 1
                or type(row_counts[0]) is not int
                or row_counts[0] < 0
            ):
                raise ValueError("Invalid total row count")
            rows_i = row_counts[0]
            if n.shape != (width,) or not np.issubdtype(n.dtype, np.integer) or (n < 0).any() or (n > rows_i).any():
                raise ValueError("Invalid per-channel counts")
            values = {key: np.asarray(data[key], dtype=np.float64) for key in ("mean", "std", "min", "max", *qs)}
            if any(v.shape != (width,) or not np.isfinite(v).all() for v in values.values()):
                raise ValueError("Invalid statistics shape or values")
            active = n > 0
            if not np.array_equal(active, data["valid"]) or (values["std"] < 0).any():
                raise ValueError("Invalid validity mask or standard deviation")
            if any(
                (values[left][active] > values[right][active]).any()
                for left, right in (("min", "q01"), ("q01", "q99"), ("q99", "max"))
            ):
                raise ValueError("Invalid quantile or extrema ordering")
            # 每个子集都必须具有完整四元数组，不能让互补的错误 mask 在聚合后被掩盖。
            for group in reference["rotation_groups"]:
                if (
                    len(group) != 4
                    or len(set(group)) != 4
                    or any(type(i) is not int or not 0 <= i < width for i in group)
                    or (active[group].any() and not active[group].all())
                ):
                    raise ValueError("Invalid quaternion coverage")
            total = counts + n
            delta = values["mean"] - mean
            # std 是总体标准差；恢复 M2 后加入组间均值差，不能直接平均 std。
            m2 += values["std"] ** 2 * n + delta**2 * counts * n / np.maximum(total, 1)
            mean += delta * n / np.maximum(total, 1)
            counts = total
            rows += rows_i
            minimum = np.minimum(minimum, np.where(active, values["min"], np.inf))
            maximum = np.maximum(maximum, np.where(active, values["max"], -np.inf))
            for q in qs:
                qmin[q] = np.minimum(qmin[q], np.where(active, values[q], np.inf))
                qmax[q] = np.maximum(qmax[q], np.where(active, values[q], -np.inf))
        valid = counts > 0

        def clean(x):
            return np.where(valid, x, 0)

        metrics = {
            "min": clean(minimum).tolist(),
            "max": clean(maximum).tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(np.maximum(m2, 0) / np.maximum(counts, 1)).tolist(),
            "count": [rows],
            "valid_counts": counts.tolist(),
            "q01": clean(qmin["q01"]).tolist(),
            "q99": clean(qmax["q99"]).tolist(),
        }
        low, high = (minimum, maximum) if bounds == "min_max" else (qmin["q01"], qmax["q99"])
        low, high = clean(low).astype(np.float32), clean(high).astype(np.float32)
        for group in reference["rotation_groups"]:
            indexes = np.asarray(group)
            if len(group) != 4 or (valid[indexes].any() and not valid[indexes].all()):
                raise ValueError("Invalid quaternion coverage")
            if valid[indexes].all():
                low[indexes], high[indexes] = -1, 1
        return Quantiles(torch.from_numpy(low), torch.from_numpy(high), torch.from_numpy(valid), metrics)

    provenance = {key: reference[key] for key in contract_keys}
    provenance.update(
        kind="fitted",
        method="source_quantile_envelope",
        bounds=bounds,
        quantiles=[0.01, 0.99],
        quantile_semantics="q01=min(source_q01); q99=max(source_q99); not global quantiles",
        quaternion_bounds=[-1, 1],
        dataset_roots=roots,
        input_statistics=[str(p) for p in paths],
        partial=any(r["provenance"]["partial"] for r in records),
        sources=[s for r in records for s in r["provenance"]["sources"]],
    )
    for key in ("blocks", "available_blocks", "skipped_ranges", "discarded_action_steps"):
        provenance[key] = sum(r["provenance"][key] for r in records)
    return BlockStatistics(merge("state"), merge("action"), provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", nargs="+", type=Path, required=True)
    parser.add_argument("--bounds", choices=("q01_q99", "min_max"), default="q01_q99")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [root / STATS_RELATIVE_PATH for root in discover_dataset_roots(args.dataset_root)]
    if args.output.resolve() in [p.resolve() for p in paths]:
        parser.error("Output must not overwrite per-dataset statistics")
    result = aggregate_statistics(paths, bounds=args.bounds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    print(f"Aggregated {len(paths)} datasets into {args.output}; partial={result.provenance['partial']}")


if __name__ == "__main__":
    main()
