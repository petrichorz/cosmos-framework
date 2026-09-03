# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import contextlib
import os
import time

import torch

from cosmos_framework.utils import distributed, log
from cosmos_framework.utils.easy_io import easy_io

# (qsh 2024-11-23)  credits
# https://github.com/pytorch/torchtitan/blob/main/torchtitan/profiling.py

# how much memory allocation/free ops to record in memory snapshots
MEMORY_SNAPSHOT_MAX_ENTRIES = 100000


@contextlib.contextmanager
def maybe_enable_npu_profiling(config, *, global_step: int = 0):
    """Collect an Ascend trace on selected ranks when NPU profiling is enabled."""

    profiling = config.trainer.profiling
    if not profiling.enable_profiling:
        yield None
        return

    _validate_npu_profiling_config(profiling)
    rank = distributed.get_rank()
    if rank not in profiling.target_ranks:
        yield None
        return

    try:
        import torch_npu
    except ImportError as error:
        raise RuntimeError("NPU profiling requires torch_npu to be installed") from error
    if not torch_npu.npu.is_available():
        raise RuntimeError("NPU profiling is enabled, but no Ascend NPU is available")

    trace_dir = os.path.join(config.job.path_local, "npu_trace")
    rank_trace_dir = os.path.join(trace_dir, f"rank{rank}")
    os.makedirs(rank_trace_dir, exist_ok=True)
    trace_handler = torch_npu.profiler.tensorboard_trace_handler(
        rank_trace_dir,
        worker_name=f"rank{rank}",
        analyse_flag=False,
    )

    def on_trace_ready(prof):
        absolute_step = global_step + prof.step_num
        log.info(f"Dumping NPU profiler trace for rank {rank} at step {absolute_step}")
        begin = time.monotonic()
        trace_handler(prof)
        log.info(f"Finished dumping NPU profiler trace in {time.monotonic() - begin:.2f} seconds")

    log.info(f"NPU profiling active on rank {rank}. Traces will be saved at {rank_trace_dir}")

    npu_profiler = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=profiling.profile_wait,
            warmup=profiling.profile_warmup,
            active=profiling.profile_active,
            repeat=profiling.profile_repeat,
            skip_first=profiling.profile_skip_first,
        ),
        on_trace_ready=on_trace_ready,
        record_shapes=profiling.record_shape,
        profile_memory=profiling.profile_memory,
        with_stack=profiling.with_stack,
        with_modules=profiling.with_modules,
        with_flops=profiling.with_flops,
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        ),
    )
    # Keep the schedule relative to this invocation while labeling trace steps
    # with the resumed training iteration.
    npu_profiler._step_num_offset = global_step
    with npu_profiler:
        yield npu_profiler


def _validate_npu_profiling_config(profiling) -> None:
    """Validate schedule and rank settings before constructing torch_npu profiler."""

    non_negative = {
        "profile_wait": profiling.profile_wait,
        "profile_warmup": profiling.profile_warmup,
        "profile_skip_first": profiling.profile_skip_first,
    }
    for name, value in non_negative.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"trainer.profiling.{name} must be a non-negative integer, got {value!r}")
    positive = {
        "profile_active": profiling.profile_active,
        "profile_repeat": profiling.profile_repeat,
    }
    for name, value in positive.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"trainer.profiling.{name} must be a positive integer, got {value!r}")
    target_ranks = profiling.target_ranks
    if not isinstance(target_ranks, list) or not target_ranks:
        raise ValueError("trainer.profiling.target_ranks must be a non-empty list")
    if any(not isinstance(rank, int) or isinstance(rank, bool) or rank < 0 for rank in target_ranks):
        raise ValueError(f"trainer.profiling.target_ranks must contain non-negative integers, got {target_ranks!r}")
    if len(set(target_ranks)) != len(target_ranks):
        raise ValueError(f"trainer.profiling.target_ranks must not contain duplicates, got {target_ranks!r}")


@contextlib.contextmanager
def maybe_enable_memory_snapshot(config, *, global_step: int = 0):
    enable_snapshot = config.trainer.profiling.enable_memory_snapshot
    if enable_snapshot:
        if config.trainer.profiling.save_s3:
            snapshot_dir = "s3://rundir"
        else:
            snapshot_dir = os.path.join(config.job.path_local, "memory_snapshot")
            if distributed.get_rank() == 0:
                os.makedirs(snapshot_dir, exist_ok=True)

        rank = torch.distributed.get_rank()

        class MemoryProfiler:
            def __init__(self, step_num: int, freq: int):
                torch.cuda.memory._record_memory_history(max_entries=MEMORY_SNAPSHOT_MAX_ENTRIES)
                # when resume training, we start from the last step
                self.step_num = step_num
                self.freq = freq

            def step(self, exit_ctx: bool = False):
                self.step_num += 1
                if not exit_ctx and self.step_num % self.freq != 0:
                    return
                if not exit_ctx:
                    curr_step = self.step_num
                    dir_name = f"iteration_{curr_step}"
                else:
                    # dump as iteration_0_exit if OOM at iter 1
                    curr_step = self.step_num - 1
                    dir_name = f"iteration_{curr_step}_exit"
                curr_snapshot_dir = os.path.join(snapshot_dir, dir_name)
                if not config.trainer.profiling.save_s3 and not os.path.exists(curr_snapshot_dir):
                    os.makedirs(curr_snapshot_dir, exist_ok=True)
                log.info(f"Dumping memory snapshot at step {curr_step}")
                begin = time.monotonic()

                if rank in config.trainer.profiling.target_ranks:
                    easy_io.dump(
                        torch.cuda.memory._snapshot(),
                        f"{curr_snapshot_dir}/rank{rank}_memory_snapshot.pickle",
                    )
                log.info(f"Finished dumping memory snapshot in {time.monotonic() - begin:.2f} seconds")

        log.info(f"Memory profiler active. Snapshot will be saved at {snapshot_dir}")
        profiler = MemoryProfiler(global_step, config.trainer.profiling.profile_freq)
        try:
            yield profiler
        except torch.cuda.OutOfMemoryError as e:
            profiler.step(exit_ctx=True)
    else:
        yield None


@contextlib.contextmanager
def maybe_enable_nsys_profiling(config, *, global_step: int = 0):
    """Context manager for Nsight Systems profiling via cudaProfilerStart/Stop.

    Usage: launch training with
        nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop python ...
    and set trainer.profiling.enable_nsys=true, profile_freq=<iter>.

    Reuses the torch-profile flags (profile_freq, target_ranks, profile_warmup).
    The profiler is started `profile_warmup` iterations before the target and
    stopped right after it.
    """
    enable_nsys = config.trainer.profiling.enable_nsys
    if not enable_nsys:
        yield None
        return

    rank = distributed.get_rank()
    target_ranks = config.trainer.profiling.target_ranks
    freq = config.trainer.profiling.profile_freq
    warmup = config.trainer.profiling.profile_warmup

    active_iter = freq - 1  # profile_freq=5001 profiles iter 5000
    start_iter = max(0, active_iter - warmup)

    class NsysProfiler:
        def __init__(self, step_num: int):
            self.step_num = step_num
            self._profiling = False

        def step(self):
            self.step_num += 1
            if rank not in target_ranks:
                return
            if self.step_num == start_iter and not self._profiling:
                log.info(f"[Nsys] Starting CUDA profiler at iter {self.step_num} (active iter: {active_iter})")
                torch.cuda.cudart().cudaProfilerStart()
                self._profiling = True
            if self.step_num == active_iter + 1 and self._profiling:
                torch.cuda.cudart().cudaProfilerStop()
                self._profiling = False
                log.info(f"[Nsys] Stopped CUDA profiler at iter {self.step_num}")

    log.info(f"[Nsys] Profiling enabled. Will capture iter {start_iter}-{active_iter} on ranks {target_ranks}")
    profiler = NsysProfiler(global_step)
    try:
        yield profiler
    finally:
        if profiler._profiling:
            torch.cuda.cudart().cudaProfilerStop()
