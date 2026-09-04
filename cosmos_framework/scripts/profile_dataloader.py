# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run the configured training dataloader without model computation.

This entry point composes the same SFT TOML and instantiates the exact
``dataloader_train`` object used by training. It deliberately does not create a
trainer, model, optimizer, scheduler, checkpointer, loss, or accelerator batch.

Example: ``python -m cosmos_framework.scripts.profile_dataloader --sft-toml
examples/toml/sft_config/vision_sft_edge.toml --iterations 500``.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import random
import time
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sft-toml", required=True, help="与训练命令相同的 structured SFT TOML")
    parser.add_argument("--iterations", type=int, default=500, help="读取 batch 数（默认：500）")
    parser.add_argument("--log-every", type=int, default=1, help="每隔多少个 batch 写一次内存快照")
    parser.add_argument(
        "--full-info-every",
        type=int,
        default=10,
        help="每隔多少个 batch 读取一次较慢的 USS/PSS；0 表示关闭",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="默认写到当前 job 目录下")
    parser.add_argument(
        "--gloo-interface",
        default=None,
        help="Gloo 使用的网卡名；单机 torchrun 默认自动使用 lo，多机需指定各节点共有的数据网卡",
    )
    parser.add_argument("--gc-every", type=int, default=0, help="诊断选项：定期 gc.collect()；0 表示关闭")
    parser.add_argument(
        "--malloc-trim-every",
        type=int,
        default=0,
        help="诊断选项：定期 malloc_trim(0)；0 表示关闭",
    )
    parser.add_argument(
        "--disable-accelerator-autoload",
        action="store_true",
        help="纯 CPU 环境使用；NPU/CUDA 训练机上不要设置，否则 pin_memory 行为可能不同",
    )
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="可选 Hydra overrides；建议在选项前使用 -- 分隔",
    )
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations 必须大于 0")
    if args.log_every < 1:
        parser.error("--log-every 必须大于 0")
    if args.full_info_every < 0 or args.gc_every < 0 or args.malloc_trim_every < 0:
        parser.error("--full-info-every/--gc-every/--malloc-trim-every 不能小于 0")
    return args


def _configure_gloo_interface(requested_interface: str | None) -> str | None:
    """Choose an explicit Gloo interface without resolving the host name."""
    if requested_interface:
        interface = requested_interface
    elif os.environ.get("GLOO_SOCKET_IFNAME"):
        return os.environ["GLOO_SOCKET_IFNAME"]
    else:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_world_size = os.environ.get("LOCAL_WORLD_SIZE")
        is_single_node = world_size == 1 or (local_world_size is not None and int(local_world_size) == world_size)
        if not is_single_node:
            return None
        interface = "lo"

    network_interfaces = Path("/sys/class/net")
    if network_interfaces.is_dir() and not (network_interfaces / interface).exists():
        available = sorted(path.name for path in network_interfaces.iterdir())
        raise ValueError(f"Gloo interface {interface!r} does not exist; available interfaces: {available}")
    os.environ["GLOO_SOCKET_IFNAME"] = interface
    return interface


def _ensure_distributed_initialized(gloo_interface: str | None = None) -> bool:
    """Initialize a CPU process group required by RankPartitionedDataLoader."""
    import torch.distributed as dist

    if dist.is_initialized():
        return False
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    _configure_gloo_interface(gloo_interface)
    dist.init_process_group(backend="gloo", init_method="env://")
    return True


def _tensor_summary(value: Any) -> dict[str, Any]:
    """Summarize tensors without retaining references to the batch."""
    import torch

    by_device: dict[str, dict[str, int]] = {}
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
            device = str(item.device)
            stats = by_device.setdefault(device, {"count": 0, "logical_bytes": 0})
            stats["count"] += 1
            stats["logical_bytes"] += item.numel() * item.element_size()
        elif isinstance(item, dict):
            containers += 1
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            containers += 1
            stack.extend(item)
    return {"by_device": by_device, "container_count": containers}


class _MemoryRecorder:
    def __init__(self, path: Path, rank: int) -> None:
        import psutil

        self.path = path
        self.rank = rank
        self.process = psutil.Process(os.getpid())
        self.started_at = time.monotonic()
        self.last: dict[str, int] = {}
        self.file = path.open("w", encoding="utf-8", buffering=1)

    @staticmethod
    def _proc_status(pid: int) -> dict[str, int]:
        names = {
            "VmLck": "locked_bytes",
            "VmPin": "pinned_bytes",
            "RssAnon": "rss_anon_bytes",
            "RssFile": "rss_file_bytes",
            "RssShmem": "rss_shmem_bytes",
            "VmSwap": "swap_bytes",
        }
        result = {}
        try:
            with open(f"/proc/{pid}/status", encoding="utf-8") as status_file:
                for line in status_file:
                    key, separator, value = line.partition(":")
                    if separator and key in names:
                        result[names[key]] = int(value.split()[0]) * 1024
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
        return result

    @staticmethod
    def _num_fds(process: Any) -> int | None:
        try:
            return process.num_fds()
        except (AttributeError, OSError):
            return None

    def _one_process(self, process: Any, include_full_info: bool) -> dict[str, Any]:
        import psutil

        info = process.memory_info()
        result: dict[str, Any] = {
            "pid": process.pid,
            "name": process.name(),
            "rss_bytes": info.rss,
            "vms_bytes": info.vms,
            "threads": process.num_threads(),
        }
        num_fds = self._num_fds(process)
        if num_fds is not None:
            result["num_fds"] = num_fds
        result.update(self._proc_status(process.pid))
        if include_full_info:
            try:
                full_info = process.memory_full_info()
                result["uss_bytes"] = full_info.uss
                result["pss_bytes"] = full_info.pss
            except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
                pass
        return result

    def record(
        self,
        phase: str,
        iteration: int,
        *,
        batch: Any = None,
        include_full_info: bool = False,
        fetch_seconds: float | None = None,
    ) -> dict[str, Any]:
        import psutil

        parent = self._one_process(self.process, include_full_info)
        children = []
        for child in self.process.children(recursive=True):
            try:
                children.append(self._one_process(child, include_full_info))
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
        children.sort(key=lambda item: item["pid"])

        record: dict[str, Any] = {
            "timestamp_ns": time.time_ns(),
            "elapsed_seconds": round(time.monotonic() - self.started_at, 6),
            "rank": self.rank,
            "iteration": iteration,
            "phase": phase,
            "parent": parent,
            "children": children,
            "children_count": len(children),
            "children_rss_bytes": sum(item["rss_bytes"] for item in children),
            "children_pinned_bytes": sum(item.get("pinned_bytes", 0) for item in children),
            "children_fds": sum(item.get("num_fds", 0) for item in children),
            "system_memory_available_bytes": psutil.virtual_memory().available,
        }
        record["process_tree_rss_bytes"] = parent["rss_bytes"] + record["children_rss_bytes"]
        if include_full_info:
            record["children_uss_bytes"] = sum(item.get("uss_bytes", 0) for item in children)
            record["children_pss_bytes"] = sum(item.get("pss_bytes", 0) for item in children)
            record["process_tree_uss_bytes"] = parent.get("uss_bytes", 0) + record["children_uss_bytes"]
            record["process_tree_pss_bytes"] = parent.get("pss_bytes", 0) + record["children_pss_bytes"]
        try:
            record["system_shm_used_bytes"] = psutil.disk_usage("/dev/shm").used
        except FileNotFoundError:
            pass
        if fetch_seconds is not None:
            record["fetch_seconds"] = fetch_seconds
        if batch is not None:
            record["batch_tensors"] = _tensor_summary(batch)

        current = {
            "parent_rss_bytes": parent["rss_bytes"],
            "parent_rss_anon_bytes": parent.get("rss_anon_bytes", 0),
            "parent_pinned_bytes": parent.get("pinned_bytes", 0),
            "children_rss_bytes": record["children_rss_bytes"],
            "children_pinned_bytes": record["children_pinned_bytes"],
            "children_fds": record["children_fds"],
        }
        for key in ("children_uss_bytes", "children_pss_bytes", "process_tree_uss_bytes", "process_tree_pss_bytes"):
            if key in record:
                current[key] = record[key]
        for key, value in current.items():
            if key in self.last:
                record[f"delta_{key}"] = value - self.last[key]
            self.last[key] = value
        self.file.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def close(self) -> None:
        self.file.close()


def _trim_allocator() -> None:
    gc.collect()
    try:
        ctypes.CDLL(None).malloc_trim(0)
    except (AttributeError, OSError):
        pass


def _run(args: argparse.Namespace) -> Path:
    import numpy as np
    import torch
    import torch.distributed as dist

    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos_framework.utils.lazy_config import LazyConfig, instantiate

    initialized_here = _ensure_distributed_initialized(args.gloo_interface)
    rank = dist.get_rank()
    config = load_experiment_from_toml(args.sft_toml, extra_overrides=args.opts)
    seed = int(config.trainer.seed) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    output_dir = args.output_dir or Path(config.job.path_local) / "dataloader_memory_trace"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"rank_{rank:03d}.jsonl"
    if rank == 0:
        LazyConfig.save_yaml(config, str(output_dir / "effective_config.yaml"))
    recorder = _MemoryRecorder(output_path, rank)

    dataloader = None
    data_iterator = None
    try:
        recorder.record("dataloader_init_before", -1, include_full_info=True)
        init_started = time.perf_counter()
        dataloader = instantiate(config.dataloader_train)
        init_seconds = time.perf_counter() - init_started
        recorder.record("dataloader_init_after", -1, include_full_info=True, fetch_seconds=init_seconds)
        data_iterator = iter(dataloader)
        recorder.record("iterator_ready", -1, include_full_info=True)

        for iteration in range(args.iterations):
            should_log = iteration % args.log_every == 0
            include_full = args.full_info_every > 0 and iteration % args.full_info_every == 0
            if should_log:
                recorder.record("before_next", iteration, include_full_info=include_full)
            fetch_started = time.perf_counter()
            batch = next(data_iterator)
            fetch_seconds = time.perf_counter() - fetch_started
            if should_log:
                snapshot = recorder.record(
                    "after_next",
                    iteration,
                    batch=batch,
                    include_full_info=include_full,
                    fetch_seconds=fetch_seconds,
                )
                if rank == 0:
                    parent_mib = snapshot["parent"]["rss_bytes"] / 2**20
                    children_mib = snapshot["children_rss_bytes"] / 2**20
                    batch_mib = (
                        sum(item["logical_bytes"] for item in snapshot["batch_tensors"]["by_device"].values()) / 2**20
                    )
                    print(
                        f"iteration={iteration:06d} fetch={fetch_seconds:.3f}s "
                        f"batch={batch_mib:.1f}MiB parent_rss={parent_mib:.1f}MiB "
                        f"children_rss={children_mib:.1f}MiB",
                        flush=True,
                    )
            del batch

            trimmed = False
            if args.gc_every > 0 and (iteration + 1) % args.gc_every == 0:
                gc.collect()
            if args.malloc_trim_every > 0 and (iteration + 1) % args.malloc_trim_every == 0:
                _trim_allocator()
                trimmed = True
            if should_log:
                recorder.record(
                    "after_release_trimmed" if trimmed else "after_release",
                    iteration,
                    include_full_info=include_full,
                )
        recorder.record("run_end", args.iterations, include_full_info=True)
    finally:
        del data_iterator
        del dataloader
        gc.collect()
        recorder.record("workers_shutdown", args.iterations, include_full_info=True)
        recorder.close()
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()
    return output_path


def main() -> None:
    args = _parse_args()
    if args.disable_accelerator_autoload:
        os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    output_path = _run(args)
    print(f"Dataloader-only memory trace written to: {output_path}")


if __name__ == "__main__":
    main()
