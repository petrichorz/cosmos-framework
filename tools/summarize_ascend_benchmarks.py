"""Aggregate completed rank JSONL measurements, refusing incomplete comparisons."""

import argparse
import json
import statistics
from pathlib import Path


def summarize(path):
    files = sorted(path.glob("benchmark_rank*.jsonl"))
    ranks = {int(f.stem.split("rank")[1]): [json.loads(l) for l in f.read_text().splitlines()] for f in files}
    valid = {r: {x["iteration"]: x for x in rows if x["recorded"]} for r, rows in ranks.items()}
    steps = sorted(set.intersection(*(set(rows) for rows in valid.values()))) if valid else []
    result = {"run": str(path), "ranks": len(ranks), "measured_steps": steps, "complete": len(steps) == 10}
    if not steps:
        return result
    slowest = [max(rows[s]["seconds"] for rows in valid.values()) for s in steps]
    result.update(
        {
            "mean_rank_max_iter_seconds": statistics.mean(slowest),
            "rank_max_iter_seconds": slowest,
            "per_rank_mean_seconds": {
                r: statistics.mean(rows[s]["seconds"] for s in steps) for r, rows in valid.items()
            },
            "peak_allocated_gib": max(rows[s]["peak_allocated_bytes"] for rows in valid.values() for s in steps)
            / 2**30,
            "peak_reserved_gib": max(rows[s]["peak_reserved_bytes"] for rows in valid.values() for s in steps) / 2**30,
        }
    )
    packing = {
        int(f.stem.split("rank")[1]): [json.loads(line) for line in f.read_text().splitlines()]
        for f in path.glob("packing_rank*.jsonl")
    }
    if set(packing) == set(valid) and all(len(packing[r]) >= max(steps) for r in valid):
        tokens = [sum(sum(packing[r][step - 1]["split_lens"][1::2]) // 2 for r in valid) for step in steps]
        result["noisy_vision_tokens"] = sum(tokens)
        result["noisy_vision_tokens_per_second"] = sum(tokens) / sum(slowest)
    return result


def input_equal(a, b):
    fa, fb = sorted(a.glob("benchmark_rank*.jsonl")), sorted(b.glob("benchmark_rank*.jsonl"))
    if len(fa) != len(fb):
        return False
    keys = ("__key__", "frame_start", "frame_end", "num_frames", "image_size", "text_token_ids", "conditioning_fps")
    for x, y in zip(fa, fb):
        left = [json.loads(l) for l in x.read_text().splitlines()]
        right = [json.loads(l) for l in y.read_text().splitlines()]
        if len(left) != len(right):
            return False
        for u, v in zip(left, right):
            if any(u["input"].get(k) != v["input"].get(k) for k in keys):
                return False
    pa, pb = sorted(a.glob("packing_rank*.jsonl")), sorted(b.glob("packing_rank*.jsonl"))
    if len(pa) != len(fa) or len(pb) != len(fb) or not pa:
        return False
    if any(x.read_text() != y.read_text() for x, y in zip(pa, pb)):
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    results = [summarize(p) for p in args.runs]
    pairs = []
    if len(results) == 2 and all(r["complete"] for r in results):
        equal = input_equal(*args.runs)
        base, test = results
        pairs.append(
            {
                "input_equal": equal,
                "iter_time_reduction_percent": (
                    1 - test["mean_rank_max_iter_seconds"] / base["mean_rank_max_iter_seconds"]
                )
                * 100
                if equal
                else None,
                "allocated_reduction_gib": base["peak_allocated_gib"] - test["peak_allocated_gib"] if equal else None,
            }
        )
    print(json.dumps({"runs": results, "comparison": pairs}, indent=2))
