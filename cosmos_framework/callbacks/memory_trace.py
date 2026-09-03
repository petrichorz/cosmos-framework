# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Low-overhead, phase-level memory tracing for leak investigations."""

from __future__ import annotations

import ctypes
import gc
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import psutil
import torch

from cosmos_framework.utils import distributed, log
from cosmos_framework.utils.callback import Callback

_PROC_STATUS_KEYS = {
    "VmLck": "process_locked_bytes",
    "VmPin": "process_pinned_bytes",
    "RssAnon": "process_rss_anon_bytes",
    "RssFile": "process_rss_file_bytes",
    "RssShmem": "process_rss_shmem_bytes",
    "VmSwap": "process_swap_bytes",
}


def _read_proc_status_bytes(pid: int) -> dict[str, int]:
    """Read Linux physical-memory categories from /proc/<pid>/status."""
    result: dict[str, int] = {}
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as status_file:
            for line in status_file:
                key, separator, value = line.partition(":")
                if separator and key in _PROC_STATUS_KEYS:
                    # Linux reports these fields in KiB.
                    result[_PROC_STATUS_KEYS[key]] = int(value.split()[0]) * 1024
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
    return result


def _summarize_tensors(value: Any) -> dict[str, Any]:
    """Return tensor sizes without retaining references to tensors or storages."""
    by_device: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "logical_bytes": 0})
    seen: set[int] = set()
    stack = [value]
    containers = 0
    while stack:
        item = stack.pop()
        item_id = id(item)
        if item_id in seen:
            continue
        seen.add(item_id)
        if isinstance(item, torch.Tensor):
            key = str(item.device)
            by_device[key]["count"] += 1
            by_device[key]["logical_bytes"] += item.numel() * item.element_size()
        elif isinstance(item, dict):
            containers += 1
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            containers += 1
            stack.extend(item)
    return {"by_device": dict(by_device), "container_count": containers}


class MemoryTrace(Callback):
    """Write process, worker and accelerator memory at training phase boundaries.

    One JSONL file is written per distributed rank under
    ``<job.path_local>/<output_subdir>``. The callback deliberately stores only
    scalar summaries, never a batch, output, loss, tensor, or process object.
    """

    def __init__(
        self,
        every_n: int = 1,
        output_subdir: str = "memory_trace",
        include_batch_summary: bool = True,
        include_worker_details: bool = True,
        include_worker_full_info: bool = True,
        track_python_objects: bool = False,
        trim_cpu_every_n: int = 0,
    ) -> None:
        self.every_n = every_n
        self.output_subdir = output_subdir
        self.include_batch_summary = include_batch_summary
        self.include_worker_details = include_worker_details
        self.include_worker_full_info = include_worker_full_info
        self.track_python_objects = track_python_objects
        self.trim_cpu_every_n = trim_cpu_every_n
        self._started_at = time.monotonic()
        self._file = None
        self._last: dict[str, int] = {}
        self._previous_python_type_counts: Counter[str] | None = None
        self._warned = False

    def _should_record(self, iteration: int) -> bool:
        return self.every_n > 0 and iteration % self.every_n == 0

    @staticmethod
    def _accelerator_memory() -> dict[str, int | str]:
        backend = getattr(torch, "npu", None)
        if backend is None or not backend.is_available():
            backend = torch.cuda
        result: dict[str, int | str] = {
            "accelerator_backend": "npu" if backend is getattr(torch, "npu", None) else "cuda"
        }
        try:
            result.update(
                accelerator_allocated_bytes=backend.memory_allocated(),
                accelerator_reserved_bytes=backend.memory_reserved(),
                accelerator_max_allocated_bytes=backend.max_memory_allocated(),
                accelerator_max_reserved_bytes=backend.max_memory_reserved(),
            )
            stats = backend.memory_stats()
            for source, target in (
                ("active_bytes.all.current", "accelerator_active_bytes"),
                ("inactive_split_bytes.all.current", "accelerator_inactive_split_bytes"),
                ("requested_bytes.all.current", "accelerator_requested_bytes"),
                ("segment.all.current", "accelerator_segment_count"),
                ("num_alloc_retries", "accelerator_alloc_retries"),
                ("num_ooms", "accelerator_ooms"),
            ):
                if source in stats:
                    result[target] = stats[source]
        except Exception as error:  # Monitoring must never terminate training.
            result["accelerator_error"] = str(error)
        return result

    def _process_memory(self) -> dict[str, Any]:
        process = psutil.Process(os.getpid())
        info = process.memory_info()
        result: dict[str, Any] = {
            "process_rss_bytes": info.rss,
            "process_vms_bytes": info.vms,
            "process_threads": process.num_threads(),
        }
        result.update(_read_proc_status_bytes(process.pid))
        if hasattr(process, "num_fds"):
            result["process_fds"] = process.num_fds()
        try:
            full_info = process.memory_full_info()
            result["process_uss_bytes"] = full_info.uss
            result["process_pss_bytes"] = full_info.pss
        except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
            pass

        workers = []
        for child in process.children(recursive=True):
            try:
                worker = {"pid": child.pid, "rss_bytes": child.memory_info().rss, "name": child.name()}
                if hasattr(child, "num_fds"):
                    worker["num_fds"] = child.num_fds()
                if self.include_worker_full_info:
                    child_full_info = child.memory_full_info()
                    worker["uss_bytes"] = child_full_info.uss
                    worker["pss_bytes"] = child_full_info.pss
                worker.update(_read_proc_status_bytes(child.pid))
                workers.append(worker)
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
        result["children_count"] = len(workers)
        result["children_rss_bytes"] = sum(worker["rss_bytes"] for worker in workers)
        result["children_uss_bytes"] = sum(worker.get("uss_bytes", 0) for worker in workers)
        result["children_pss_bytes"] = sum(worker.get("pss_bytes", 0) for worker in workers)
        result["children_pinned_bytes"] = sum(worker.get("process_pinned_bytes", 0) for worker in workers)
        result["children_fds"] = sum(worker.get("num_fds", 0) for worker in workers)
        if self.include_worker_details:
            result["children"] = sorted(workers, key=lambda worker: worker["pid"])
        return result

    def _record(self, phase: str, iteration: int, tensors: Any = None) -> None:
        if self._file is None or not self._should_record(iteration):
            return
        try:
            record: dict[str, Any] = {
                "timestamp_ns": time.time_ns(),
                "elapsed_seconds": round(time.monotonic() - self._started_at, 6),
                "rank": distributed.get_rank(),
                "pid": os.getpid(),
                "iteration": iteration,
                "phase": phase,
                "gc_count": list(gc.get_count()),
                "gc_enabled": gc.isenabled(),
            }
            virtual_memory = psutil.virtual_memory()
            record["system_memory_available_bytes"] = virtual_memory.available
            record["system_memory_used_bytes"] = virtual_memory.used
            try:
                shm_usage = psutil.disk_usage("/dev/shm")
                record["system_shm_used_bytes"] = shm_usage.used
            except FileNotFoundError:
                pass
            record.update(self._process_memory())
            record.update(self._accelerator_memory())
            if self.track_python_objects and phase in {"iteration_end", "iteration_end_after_trim"}:
                tracked_objects = gc.get_objects()
                type_counts = Counter(f"{type(item).__module__}.{type(item).__qualname__}" for item in tracked_objects)
                record["python_gc_tracked_objects"] = len(tracked_objects)
                record["python_gc_top_types"] = type_counts.most_common(20)
                if self._previous_python_type_counts is not None:
                    deltas = type_counts.copy()
                    deltas.subtract(self._previous_python_type_counts)
                    record["python_gc_top_growth"] = sorted(
                        ((type_name, count) for type_name, count in deltas.items() if count > 0),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:20]
                self._previous_python_type_counts = type_counts
                del tracked_objects
            for key in (
                "process_rss_bytes",
                "process_uss_bytes",
                "process_pss_bytes",
                "children_rss_bytes",
                "children_uss_bytes",
                "children_pss_bytes",
                "process_pinned_bytes",
                "children_pinned_bytes",
                "children_fds",
                "accelerator_allocated_bytes",
                "accelerator_reserved_bytes",
            ):
                value = record.get(key)
                if isinstance(value, int) and key in self._last:
                    record[f"delta_{key}"] = value - self._last[key]
                if isinstance(value, int):
                    self._last[key] = value
            if tensors is not None and self.include_batch_summary:
                record["tensors"] = _summarize_tensors(tensors)
            self._file.write(json.dumps(record, sort_keys=True) + "\n")
            self._file.flush()
        except Exception as error:  # Monitoring must never terminate training.
            if not self._warned:
                log.warning(f"MemoryTrace disabled after monitoring error: {error}")
                self._warned = True
            self._close()

    def _close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def on_train_start(self, model, iteration: int = 0) -> None:
        del model
        rank = distributed.get_rank()
        output_dir = Path(self.config.job.path_local) / self.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        self._file = (output_dir / f"rank_{rank:03d}.jsonl").open("a", encoding="utf-8", buffering=1)
        self._record("train_start", iteration)
        log.info(f"MemoryTrace rank {rank}: writing phase snapshots to {output_dir}")

    def on_before_dataloading(self, iteration: int = 0) -> None:
        self._record("before_dataloading", iteration)

    def on_after_dataloading(self, iteration: int = 0) -> None:
        self._record("after_dataloading", iteration)

    def on_training_step_batch_start(self, model, data, iteration: int = 0) -> None:
        del model
        self._record("batch_on_device", iteration, data)

    def on_before_forward(self, iteration: int = 0) -> None:
        self._record("before_forward", iteration)

    def on_after_forward(self, iteration: int = 0) -> None:
        self._record("after_forward", iteration)

    def on_before_backward(self, model, loss, iteration: int = 0) -> None:
        del model, loss
        self._record("before_backward", iteration)

    def on_after_backward(self, model, iteration: int = 0) -> None:
        del model
        self._record("after_backward", iteration)

    def on_before_optimizer_step(self, model, optimizer, scheduler, grad_scaler, iteration: int = 0) -> None:
        del model, optimizer, scheduler, grad_scaler
        self._record("before_optimizer", iteration)

    def on_before_zero_grad(self, model, optimizer, scheduler, iteration: int = 0) -> None:
        del model, optimizer, scheduler
        self._record("before_zero_grad", iteration)

    def on_training_step_batch_end(self, model, data_batch, output_batch, loss, iteration: int = 0) -> None:
        del model, data_batch, loss
        self._record("batch_end", iteration, output_batch)

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration: int = 0) -> None:
        del model, data_batch, output_batch, loss
        self._record("iteration_end", iteration)
        if self.trim_cpu_every_n > 0 and iteration % self.trim_cpu_every_n == 0:
            gc.collect()
            try:
                ctypes.CDLL(None).malloc_trim(0)
            except (AttributeError, OSError):
                pass
            self._record("iteration_end_after_trim", iteration)

    def on_train_end(self, model, iteration: int = 0) -> None:
        del model
        self._record("train_end", iteration)
        self._close()

    def on_app_end(self) -> None:
        self._close()
