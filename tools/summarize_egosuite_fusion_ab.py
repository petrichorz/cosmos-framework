#!/usr/bin/env python3
"""Summarize fixed-condition Cosmos distributed A/B runs from performance JSONL."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[round((len(ordered) - 1) * fraction)]


def summarize_run(path: Path, discard_iterations: int) -> dict[str, Any]:
    events_root = path / "events" if (path / "events").is_dir() else path
    stage_by_iteration: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    memory_allocated: list[int] = []
    memory_reserved: list[int] = []
    ranks: set[int] = set()

    for event_path in sorted(events_root.rglob("events_rank*_worker-1_pid*.jsonl")):
        for line in event_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            rank = int(event.get("rank", 0))
            ranks.add(rank)
            iteration = event.get("iteration")
            duration = event.get("duration_ms")
            name = str(event.get("name", "unknown"))
            # Stage scopes are emitted before the trainer increments its counter;
            # iteration_core is emitted immediately afterwards.
            logical_iteration = (
                iteration if name == "iteration_core" else iteration + 1 if isinstance(iteration, int) else None
            )
            if (
                isinstance(logical_iteration, int)
                and logical_iteration > discard_iterations
                and isinstance(duration, int | float)
            ):
                stage_by_iteration[name][logical_iteration].append(float(duration))
            if event.get("name") == "iteration_core" and isinstance(iteration, int) and iteration > discard_iterations:
                allocated = event.get("max_memory_allocated_bytes")
                reserved = event.get("max_memory_reserved_bytes")
                if isinstance(allocated, int):
                    memory_allocated.append(allocated)
                if isinstance(reserved, int):
                    memory_reserved.append(reserved)

    stages: dict[str, Any] = {}
    for name, iteration_values in stage_by_iteration.items():
        rank_maxima = [max(values) for _, values in sorted(iteration_values.items()) if values]
        all_rank_values = [value for values in iteration_values.values() for value in values]
        stages[name] = {
            "iterations": len(rank_maxima),
            "rank_samples": len(all_rank_values),
            "critical_mean_ms": statistics.fmean(rank_maxima) if rank_maxima else 0.0,
            "critical_p50_ms": _percentile(rank_maxima, 0.5),
            "critical_p90_ms": _percentile(rank_maxima, 0.9),
            "all_rank_mean_ms": statistics.fmean(all_rank_values) if all_rank_values else 0.0,
        }

    gib = 1024**3
    return {
        "path": str(path.resolve()),
        "rank_count": len(ranks),
        "discard_iterations": discard_iterations,
        "stages": stages,
        "memory": {
            "max_allocated_gib": max(memory_allocated, default=0) / gib,
            "mean_rank_step_allocated_gib": statistics.fmean(memory_allocated) / gib if memory_allocated else 0.0,
            "max_reserved_gib": max(memory_reserved, default=0) / gib,
            "mean_rank_step_reserved_gib": statistics.fmean(memory_reserved) / gib if memory_reserved else 0.0,
            "samples": len(memory_allocated),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--discard-iterations", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries: dict[str, Any] = {}
    for spec in args.run:
        if "=" not in spec:
            raise ValueError(f"Expected NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        summaries[name] = summarize_run(Path(path), args.discard_iterations)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    markdown_path = args.output.with_suffix(".md")
    lines = [
        "# EgoSuite fusion A/B summary",
        "",
        "| Run | Ranks | Stable steps | Iteration critical mean (ms) | Forward critical mean (ms) | "
        "Backward critical mean (ms) | Peak allocated (GiB) | Peak reserved (GiB) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, summary in summaries.items():
        stages = summary["stages"]
        iteration = stages.get("iteration_core", {})
        forward = stages.get("forward", {})
        backward = stages.get("backward", {})
        memory = summary["memory"]
        lines.append(
            f"| {name} | {summary['rank_count']} | {iteration.get('iterations', 0)} | "
            f"{iteration.get('critical_mean_ms', 0.0):.3f} | {forward.get('critical_mean_ms', 0.0):.3f} | "
            f"{backward.get('critical_mean_ms', 0.0):.3f} | {memory['max_allocated_gib']:.3f} | "
            f"{memory['max_reserved_gib']:.3f} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(markdown_path)


if __name__ == "__main__":
    main()
