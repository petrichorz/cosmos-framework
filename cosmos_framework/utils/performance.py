# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Low-overhead performance markers for trainers and data-loader workers.

Set ``COSMOS_PERF_OUTPUT_DIR`` to enable JSONL timing output. Each process writes
to its own file, so DataLoader workers never contend on a shared descriptor. The
same scopes are emitted as PyTorch ``record_function`` ranges and are therefore
visible in Ascend PyTorch Profiler traces.
"""

from __future__ import annotations

import atexit
import contextlib
import csv
import json
import os
import statistics
import time
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO

import torch

_FILE_HANDLES: dict[int, TextIO] = {}
_MSTX_DOMAIN = "cosmos_forward"
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _worker_id() -> int:
    worker_info = torch.utils.data.get_worker_info()
    return worker_info.id if worker_info is not None else -1


def _rank() -> int:
    try:
        return int(os.environ.get("RANK", "0"))
    except ValueError:
        return 0


def performance_output_dir() -> Path | None:
    output = os.environ.get("COSMOS_PERF_OUTPUT_DIR")
    return Path(output) if output else None


def performance_enabled() -> bool:
    return performance_output_dir() is not None


def npu_mstx_enabled() -> bool:
    """Return whether low-overhead Ascend MSTX forward ranges are enabled."""
    return os.environ.get("COSMOS_NPU_MSTX", "0").lower() in _TRUE_VALUES


@contextlib.contextmanager
def npu_mstx_scope(name: str) -> Iterator[None]:
    """Annotate host work and its current NPU stream without synchronizing it."""
    if not npu_mstx_enabled():
        yield
        return

    import torch_npu

    range_id = torch_npu.npu.mstx.range_start(
        f"COSMOS::{name.upper()}",
        torch_npu.npu.current_stream(),
        domain=_MSTX_DOMAIN,
    )
    try:
        yield
    finally:
        torch_npu.npu.mstx.range_end(range_id, domain=_MSTX_DOMAIN)


def _event_file() -> TextIO | None:
    output_dir = performance_output_dir()
    if output_dir is None:
        return None
    pid = os.getpid()
    if pid not in _FILE_HANDLES:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"events_rank{_rank()}_worker{_worker_id()}_pid{pid}.jsonl"
        _FILE_HANDLES[pid] = path.open("a", encoding="utf-8", buffering=1)
    return _FILE_HANDLES[pid]


def _close_files() -> None:
    for file_handle in _FILE_HANDLES.values():
        with contextlib.suppress(Exception):
            file_handle.close()
    _FILE_HANDLES.clear()


atexit.register(_close_files)


def record_performance_event(name: str, *, duration_ms: float | None = None, **metadata: Any) -> None:
    """Append one event to the current process' JSONL file when enabled."""
    file_handle = _event_file()
    if file_handle is None:
        return
    event = {
        "name": name,
        "timestamp_ns": time.time_ns(),
        "duration_ms": duration_ms,
        "pid": os.getpid(),
        "rank": _rank(),
        "worker_id": _worker_id(),
        **metadata,
    }
    file_handle.write(json.dumps(event, default=str, separators=(",", ":")) + "\n")


@contextlib.contextmanager
def performance_scope(name: str, **metadata: Any) -> Iterator[None]:
    """Emit a wall-clock JSONL duration and a profiler-visible named range."""
    if not performance_enabled():
        yield
        return

    start_ns = time.perf_counter_ns()
    error_name = None
    try:
        with torch.profiler.record_function(f"COSMOS::{name.upper()}"):
            yield
    except Exception as error:
        error_name = type(error).__name__
        raise
    finally:
        record_performance_event(
            name,
            duration_ms=(time.perf_counter_ns() - start_ns) / 1_000_000,
            error=error_name,
            **metadata,
        )


def summarize_performance_events(input_dir: str | Path, output_dir: str | Path) -> list[dict[str, Any]]:
    """Aggregate process-local JSONL events into JSON, CSV, and Markdown summaries."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    durations: dict[str, list[float]] = defaultdict(list)
    event_counts: dict[str, int] = defaultdict(int)

    for event_path in sorted(input_path.rglob("events_*.jsonl")):
        with event_path.open(encoding="utf-8") as file_handle:
            for line in file_handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                name = str(event.get("name", "unknown"))
                event_counts[name] += 1
                duration = event.get("duration_ms")
                if isinstance(duration, int | float):
                    durations[name].append(float(duration))

    def percentile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
        return ordered[index]

    rows = []
    for name in sorted(event_counts):
        values = durations[name]
        rows.append(
            {
                "name": name,
                "count": event_counts[name],
                "timed_count": len(values),
                "total_ms": sum(values),
                "mean_ms": statistics.fmean(values) if values else 0.0,
                "p50_ms": percentile(values, 0.50),
                "p90_ms": percentile(values, 0.90),
                "max_ms": max(values, default=0.0),
            }
        )
    rows.sort(key=lambda row: row["total_ms"], reverse=True)

    (output_path / "performance_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (output_path / "performance_summary.csv").open("w", newline="", encoding="utf-8") as file_handle:
        fieldnames = ["name", "count", "timed_count", "total_ms", "mean_ms", "p50_ms", "p90_ms", "max_ms"]
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "# Cosmos performance summary",
        "",
        "| Stage | Count | Mean ms | P50 ms | P90 ms | Max ms | Total ms |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    markdown.extend(
        f"| {row['name']} | {row['count']} | {row['mean_ms']:.3f} | {row['p50_ms']:.3f} | "
        f"{row['p90_ms']:.3f} | {row['max_ms']:.3f} | {row['total_ms']:.3f} |"
        for row in rows
    )
    (output_path / "performance_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return rows


def _float_from_csv(value: Any) -> float:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def _aggregate_ascend_csv(
    paths: list[Path], duration_fields: tuple[str, ...], extra_fields: tuple[str, ...] = ()
) -> list[dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as file_handle:
            for row in csv.DictReader(file_handle):
                name = row.get("Name") or row.get("Op Name") or "unknown"
                populated_duration_fields = [
                    field
                    for field in duration_fields
                    if row.get(field) is not None and str(row[field]).strip() != ""
                ]
                duration_field = next(
                    (field for field in populated_duration_fields if _float_from_csv(row[field]) != 0.0),
                    populated_duration_fields[0] if populated_duration_fields else None,
                )
                duration_us = _float_from_csv(row.get(duration_field)) if duration_field else 0.0
                item = aggregated.setdefault(
                    name,
                    {
                        "name": name,
                        "calls": 0,
                        "total_us": 0.0,
                        "mean_us": 0.0,
                        "max_us": 0.0,
                        "duration_field": duration_field or "unavailable",
                        **{field: "" for field in extra_fields},
                    },
                )
                item["calls"] += 1
                item["total_us"] += duration_us
                item["max_us"] = max(item["max_us"], duration_us)
                for field in extra_fields:
                    if not item[field] and row.get(field):
                        item[field] = row[field]
    for item in aggregated.values():
        item["mean_us"] = item["total_us"] / item["calls"]
    return sorted(aggregated.values(), key=lambda item: item["total_us"], reverse=True)


def summarize_ascend_profiler_outputs(input_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Create compact hot-spot tables and a trace index from Ascend profiler exports."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    operator_paths = sorted(input_path.rglob("operator_details.csv"))
    kernel_paths = sorted(input_path.rglob("kernel_details.csv"))
    trace_paths = sorted(input_path.rglob("trace_view.json"))
    operators = _aggregate_ascend_csv(
        operator_paths,
        (
            "Device Self Duration With AICore(us)",
            "Device Self Duration(us)",
            "Device Total Duration(us)",
            "Host Self Duration(us)",
        ),
        ("Input Shapes", "Call Stack"),
    )
    kernels = _aggregate_ascend_csv(kernel_paths, ("Duration(us)", "Task Duration(us)"))

    def write_table(name: str, rows: list[dict[str, Any]]) -> None:
        (output_path / f"{name}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        if not rows:
            return
        with (output_path / f"{name}.csv").open("w", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_table("ascend_operator_hotspots", operators)
    write_table("ascend_kernel_hotspots", kernels)
    traces = [str(path.resolve()) for path in trace_paths]
    result = {
        "operator_detail_files": [str(path.resolve()) for path in operator_paths],
        "kernel_detail_files": [str(path.resolve()) for path in kernel_paths],
        "trace_view_files": traces,
    }
    (output_path / "ascend_artifacts.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    markdown = [
        "# Ascend profiler hot spots",
        "",
        f"Operator tables: {len(operator_paths)}; kernel tables: {len(kernel_paths)}; traces: {len(traces)}.",
        "",
        "## Top operators",
        "",
        "| Operator | Calls | Total us | Mean us | Max us |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    markdown.extend(
        f"| {row['name']} | {row['calls']} | {row['total_us']:.3f} | {row['mean_us']:.3f} | "
        f"{row['max_us']:.3f} |"
        for row in operators[:30]
    )
    markdown.extend(
        [
            "",
            "## Top kernels",
            "",
            "| Kernel | Calls | Total us | Mean us | Max us |",
            "| :--- | ---: | ---: | ---: | ---: |",
        ]
    )
    markdown.extend(
        f"| {row['name']} | {row['calls']} | {row['total_us']:.3f} | {row['mean_us']:.3f} | "
        f"{row['max_us']:.3f} |"
        for row in kernels[:30]
    )
    (output_path / "ascend_hotspots.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return {"operators": operators, "kernels": kernels, **result}
