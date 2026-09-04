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
import time
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sft-toml", required=True, help="与训练命令相同的 structured SFT TOML")
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="模拟的 optimizer iteration 数；默认从 --start-iteration 跑到 trainer.max_iter",
    )
    parser.add_argument(
        "--start-iteration",
        type=int,
        default=0,
        help="模拟 checkpoint 恢复后的起始 optimizer iteration（默认：0）",
    )
    parser.add_argument("--log-every", type=int, default=1, help="每隔多少个 batch 写一次内存快照")
    parser.add_argument(
        "--full-info-every",
        type=int,
        default=10,
        help="每隔多少个 batch 读取一次较慢的 USS/PSS；0 表示关闭",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="默认写到当前 job 目录下")
    parser.add_argument(
        "--wandb-mode",
        choices=("disabled", "offline", "online"),
        default="disabled",
        help="每次内存快照实时写入 W&B；默认关闭，offline 仅写本地目录",
    )
    parser.add_argument(
        "--wandb-ranks",
        choices=("rank0", "all"),
        default="rank0",
        help="上传 rank 0，或让每个 rank 创建独立 W&B run（默认：rank0）",
    )
    parser.add_argument("--wandb-project", default=None, help="默认沿用 TOML 的 job.project")
    parser.add_argument("--wandb-group", default=None, help="默认沿用 TOML 的 job.group")
    parser.add_argument("--wandb-name", default=None, help="默认使用 <job.name>-dataloader-profile-rankNNN")
    parser.add_argument("--gc-every", type=int, default=0, help="诊断选项：定期 gc.collect()；0 表示关闭")
    parser.add_argument(
        "--malloc-trim-every",
        type=int,
        default=0,
        help="诊断选项：定期 malloc_trim(0)；0 表示关闭",
    )
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="可选 Hydra overrides；建议在选项前使用 -- 分隔",
    )
    args = parser.parse_args()
    if args.iterations is not None and args.iterations < 1:
        parser.error("--iterations 必须大于 0")
    if args.start_iteration < 0:
        parser.error("--start-iteration 不能小于 0")
    if args.log_every < 1:
        parser.error("--log-every 必须大于 0")
    if args.full_info_every < 0 or args.gc_every < 0 or args.malloc_trim_every < 0:
        parser.error("--full-info-every/--gc-every/--malloc-trim-every 不能小于 0")
    return args


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
    def __init__(self, path: Path, rank: int, wandb_run: Any = None) -> None:
        import psutil

        self.path = path
        self.rank = rank
        self.wandb_run = wandb_run
        self.trace_step = 0
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
        batch_index: int | None = None,
        grad_accum_index: int | None = None,
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
        if batch_index is not None:
            record["batch_index"] = batch_index
        if grad_accum_index is not None:
            record["grad_accum_index"] = grad_accum_index
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
        if self.wandb_run is not None:
            self.wandb_run.log(
                _record_to_wandb_metrics(record, self.trace_step),
                step=self.trace_step,
            )
        self.trace_step += 1
        return record

    def close(self) -> None:
        self.file.close()


def _trim_allocator() -> None:
    gc.collect()
    try:
        ctypes.CDLL(None).malloc_trim(0)
    except (AttributeError, OSError):
        pass


def _advance_training_position(iteration: int, grad_accum_index: int, grad_accum_steps: int) -> tuple[int, int]:
    """Mirror the counter update performed after ``training_step``."""
    grad_accum_index += 1
    if grad_accum_index == grad_accum_steps:
        return iteration + 1, 0
    return iteration, grad_accum_index


def _wandb_metric_name(name: str) -> str:
    if name.endswith("_bytes"):
        return f"{name[:-6]}_mib"
    return name


def _wandb_metric_value(name: str, value: int | float) -> int | float:
    if name.endswith("_bytes"):
        return value / 2**20
    return value


def _record_to_wandb_metrics(record: dict[str, Any], trace_step: int) -> dict[str, Any]:
    """Select scalar trace fields for W&B while the JSONL remains the lossless record."""
    phase = str(record["phase"])
    prefix = f"phase/{phase}"
    metrics: dict[str, Any] = {
        "trace/step": trace_step,
        "trace/phase": phase,
        "trace/elapsed_seconds": record["elapsed_seconds"],
        "trace/iteration": record["iteration"],
        "trace/rank": record["rank"],
    }
    for key in ("batch_index", "grad_accum_index"):
        if key in record:
            metrics[f"trace/{key}"] = record[key]

    parent = record.get("parent", {})
    for key, value in parent.items():
        if key == "pid" or not isinstance(value, (int, float)):
            continue
        metrics[f"{prefix}/parent/{_wandb_metric_name(key)}"] = _wandb_metric_value(key, value)

    top_level_scalars = (
        "children_count",
        "children_rss_bytes",
        "children_pinned_bytes",
        "children_fds",
        "system_memory_available_bytes",
        "process_tree_rss_bytes",
        "children_uss_bytes",
        "children_pss_bytes",
        "process_tree_uss_bytes",
        "process_tree_pss_bytes",
        "system_shm_used_bytes",
        "fetch_seconds",
    )
    for key in top_level_scalars:
        value = record.get(key)
        if isinstance(value, (int, float)):
            metrics[f"{prefix}/{_wandb_metric_name(key)}"] = _wandb_metric_value(key, value)

    for key, value in record.items():
        if key.startswith("delta_") and isinstance(value, (int, float)):
            metrics[f"{prefix}/{_wandb_metric_name(key)}"] = _wandb_metric_value(key, value)

    children = record.get("children", [])
    if children:
        worker_rss = [child["rss_bytes"] for child in children if "rss_bytes" in child]
        worker_fds = [child["num_fds"] for child in children if "num_fds" in child]
        if worker_rss:
            metrics[f"{prefix}/worker_max_rss_mib"] = max(worker_rss) / 2**20
        if worker_fds:
            metrics[f"{prefix}/worker_max_num_fds"] = max(worker_fds)

    for device, stats in record.get("batch_tensors", {}).get("by_device", {}).items():
        safe_device = str(device).replace(":", "_")
        metrics[f"{prefix}/batch/{safe_device}/tensor_count"] = stats["count"]
        metrics[f"{prefix}/batch/{safe_device}/logical_mib"] = stats["logical_bytes"] / 2**20
    return metrics


def _init_wandb_run(
    output_path: Path,
    *,
    mode: str,
    project: str,
    group: str | None,
    name: str,
    config: dict[str, Any],
) -> tuple[Any, Path]:
    """Initialize the real-time W&B sink inside this rank process."""
    rank_dir = output_path.parent / "wandb" / output_path.stem
    rank_dir.mkdir(parents=True, exist_ok=True)
    temporary_env = {
        "WANDB_CACHE_DIR": str(rank_dir / "cache"),
        "WANDB_DATA_DIR": str(rank_dir / "data"),
    }
    added_env = []
    try:
        for key, value in temporary_env.items():
            if key not in os.environ:
                os.environ[key] = value
                added_env.append(key)
        import wandb

        run = wandb.init(
            project=project,
            group=group or None,
            name=name,
            mode=mode,
            dir=str(rank_dir),
            config=config,
            tags=["dataloader-profile"],
        )
    except ImportError as error:
        raise RuntimeError("--wandb-mode requires the wandb package") from error
    finally:
        for key in added_env:
            os.environ.pop(key, None)
    return run, rank_dir


def _run(args: argparse.Namespace) -> Path:
    # Match the NPU compatibility bootstrap in cosmos_framework.scripts.train
    # before importing the backend selector used by distributed.init().
    try:
        import torch_npu  # noqa: F401
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
    except (ImportError, OSError):
        pass

    import torch.distributed as dist

    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos_framework.utils import distributed, misc
    from cosmos_framework.utils.context_managers import data_loader_init, distributed_init
    from cosmos_framework.utils.lazy_config import LazyConfig, instantiate

    config = load_experiment_from_toml(args.sft_toml, extra_overrides=args.opts)

    # Match cosmos_framework.scripts.train.launch: initialize the accelerator
    # process group before config validation. DIST_BACKEND resolves to HCCL on
    # NPU, NCCL on CUDA, and Gloo only for a genuine CPU run.
    initialized_here = not dist.is_initialized()
    with distributed_init():
        distributed.init()
    config.validate()
    config.freeze()

    rank = dist.get_rank()
    # ImaginaireTrainer normally applies this immediately before model and
    # dataloader construction. The model is intentionally skipped here.
    misc.set_random_seed(seed=config.trainer.seed, by_rank=True)
    context_parallel_degree = int(config.model.config.parallelism.context_parallel_shard_degree)
    if context_parallel_degree != 1:
        raise ValueError(
            "Dataloader-only profiling cannot faithfully reproduce context-parallel data broadcast without a model; "
            f"got context_parallel_shard_degree={context_parallel_degree}. Use a recipe with degree 1."
        )
    grad_accum_steps = int(config.trainer.grad_accum_iter)
    if grad_accum_steps < 1:
        raise ValueError(f"trainer.grad_accum_iter must be at least 1, got {grad_accum_steps}")
    start_iteration = args.start_iteration
    max_iteration = start_iteration + args.iterations if args.iterations is not None else int(config.trainer.max_iter)
    if max_iteration < start_iteration:
        raise ValueError(
            f"trainer.max_iter ({max_iteration}) must not be smaller than --start-iteration ({start_iteration})"
        )
    output_dir = args.output_dir or Path(config.job.path_local) / "dataloader_memory_trace"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"rank_{rank:03d}.jsonl"
    if rank == 0:
        LazyConfig.save_yaml(config, str(output_dir / "effective_config.yaml"))
    wandb_run = None
    wandb_dir = None
    if args.wandb_mode != "disabled" and (args.wandb_ranks == "all" or rank == 0):
        project = args.wandb_project or str(config.job.project) or "cosmos-dataloader-profile"
        group = args.wandb_group or str(config.job.group) or None
        base_name = args.wandb_name or f"{config.job.name}-dataloader-profile"
        run_name = f"{base_name}-rank{rank:03d}"
        wandb_run, wandb_dir = _init_wandb_run(
            output_path,
            mode=args.wandb_mode,
            project=project,
            group=group,
            name=run_name,
            config={
                "sft_toml": args.sft_toml,
                "rank": rank,
                "iterations": args.iterations,
                "start_iteration": args.start_iteration,
                "log_every": args.log_every,
                "full_info_every": args.full_info_every,
                "hydra_overrides": list(args.opts),
                "effective_config_path": str(output_dir / "effective_config.yaml"),
                "logging_semantics": "real_time",
            },
        )
        print(f"W&B {args.wandb_mode} real-time run started; local files: {wandb_dir}", flush=True)
    recorder = _MemoryRecorder(output_path, rank, wandb_run=wandb_run)

    dataloader = None
    data_iterator = None
    iteration = start_iteration
    grad_accum_index = 0
    batch_index = 0
    try:
        recorder.record("dataloader_init_before", -1, include_full_info=True)
        init_started = time.perf_counter()
        with data_loader_init():
            dataloader = instantiate(config.dataloader_train)
        init_seconds = time.perf_counter() - init_started
        recorder.record("dataloader_init_after", -1, include_full_info=True, fetch_seconds=init_seconds)
        if hasattr(dataloader, "set_start_iteration"):
            dataloader.set_start_iteration(start_iteration * grad_accum_steps)

        # Mirror ImaginaireTrainer.train's two nested loops. ``iteration`` is
        # an optimizer-step index and advances only after grad_accum_steps
        # successful fetches. StopIteration rebuilds the iterator. As in the
        # real loop, the max-iteration check happens after the next batch has
        # already been fetched.
        end_profiling = False
        while True:
            data_iterator = iter(dataloader)
            recorder.record(
                "iterator_ready",
                iteration,
                include_full_info=True,
                batch_index=batch_index,
                grad_accum_index=grad_accum_index,
            )
            while True:
                should_log = batch_index % args.log_every == 0
                include_full = args.full_info_every > 0 and batch_index % args.full_info_every == 0
                if should_log:
                    recorder.record(
                        "before_next",
                        iteration,
                        include_full_info=include_full,
                        batch_index=batch_index,
                        grad_accum_index=grad_accum_index,
                    )
                fetch_started = time.perf_counter()
                try:
                    batch = next(data_iterator)
                except StopIteration:
                    if should_log:
                        recorder.record(
                            "iterator_exhausted",
                            iteration,
                            include_full_info=include_full,
                            batch_index=batch_index,
                            grad_accum_index=grad_accum_index,
                        )
                    break
                fetch_seconds = time.perf_counter() - fetch_started
                if should_log:
                    snapshot = recorder.record(
                        "after_next",
                        iteration,
                        batch=batch,
                        include_full_info=include_full,
                        fetch_seconds=fetch_seconds,
                        batch_index=batch_index,
                        grad_accum_index=grad_accum_index,
                    )
                    if rank == 0:
                        parent_mib = snapshot["parent"]["rss_bytes"] / 2**20
                        children_mib = snapshot["children_rss_bytes"] / 2**20
                        batch_mib = (
                            sum(item["logical_bytes"] for item in snapshot["batch_tensors"]["by_device"].values())
                            / 2**20
                        )
                        print(
                            f"iteration={iteration:06d} microbatch={grad_accum_index:02d} "
                            f"batch_index={batch_index:06d} fetch={fetch_seconds:.3f}s "
                            f"batch={batch_mib:.1f}MiB parent_rss={parent_mib:.1f}MiB "
                            f"children_rss={children_mib:.1f}MiB",
                            flush=True,
                        )

                # This is the intentional cutoff: the real loop checks
                # max_iter here, then moves the batch to the accelerator and
                # enters training_step. This profiler releases it instead.
                terminal_fetch = iteration >= max_iteration
                del batch

                trimmed = False
                if args.gc_every > 0 and (batch_index + 1) % args.gc_every == 0:
                    gc.collect()
                if args.malloc_trim_every > 0 and (batch_index + 1) % args.malloc_trim_every == 0:
                    _trim_allocator()
                    trimmed = True
                if should_log:
                    if terminal_fetch:
                        release_phase = "after_release_terminal"
                    elif trimmed:
                        release_phase = "after_release_trimmed"
                    else:
                        release_phase = "after_release"
                    recorder.record(
                        release_phase,
                        iteration,
                        include_full_info=include_full,
                        batch_index=batch_index,
                        grad_accum_index=grad_accum_index,
                    )
                batch_index += 1

                if terminal_fetch:
                    end_profiling = True
                    break

                # Simulate only training_step's control-flow result. No model,
                # loss, backward, optimizer, scheduler, or callbacks run.
                iteration, grad_accum_index = _advance_training_position(
                    iteration,
                    grad_accum_index,
                    grad_accum_steps,
                )
            if end_profiling:
                break
        recorder.record(
            "run_end",
            iteration,
            include_full_info=True,
            batch_index=batch_index,
            grad_accum_index=grad_accum_index,
        )
    finally:
        del data_iterator
        del dataloader
        gc.collect()
        try:
            recorder.record(
                "workers_shutdown",
                iteration,
                include_full_info=True,
                batch_index=batch_index,
                grad_accum_index=grad_accum_index,
            )
        finally:
            recorder.close()
        # Let every rank finish recording before the selected W&B runs flush.
        if args.wandb_mode != "disabled" and dist.is_initialized():
            dist.barrier()
        if wandb_run is not None:
            wandb_run.summary["trace_path"] = str(output_path)
            wandb_run.summary["trace_records"] = recorder.trace_step
            wandb_run.finish()
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()
    return output_path


def main() -> None:
    args = _parse_args()
    output_path = _run(args)
    print(f"Dataloader-only memory trace written to: {output_path}")


if __name__ == "__main__":
    main()
