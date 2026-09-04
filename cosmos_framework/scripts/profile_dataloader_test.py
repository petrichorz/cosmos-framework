# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import sys
from types import SimpleNamespace

import pytest

from cosmos_framework.scripts.profile_dataloader import (
    _advance_training_position,
    _init_wandb_run,
    _MemoryRecorder,
    _record_to_wandb_metrics,
)


def test_training_iteration_advances_after_gradient_accumulation_window():
    iteration = 7
    grad_accum_index = 0

    iteration, grad_accum_index = _advance_training_position(iteration, grad_accum_index, 2)
    assert (iteration, grad_accum_index) == (7, 1)

    iteration, grad_accum_index = _advance_training_position(iteration, grad_accum_index, 2)
    assert (iteration, grad_accum_index) == (8, 0)


def test_training_loop_fetches_once_after_reaching_max_iteration():
    iteration = 0
    grad_accum_index = 0
    fetch_count = 0
    max_iteration = 3

    while True:
        fetch_count += 1
        if iteration >= max_iteration:
            break
        iteration, grad_accum_index = _advance_training_position(iteration, grad_accum_index, 2)

    assert iteration == max_iteration
    assert fetch_count == max_iteration * 2 + 1


def test_wandb_metrics_use_phase_namespaces_and_mib():
    record = {
        "phase": "after_next",
        "elapsed_seconds": 3.5,
        "iteration": 7,
        "rank": 0,
        "batch_index": 15,
        "grad_accum_index": 1,
        "parent": {"pid": 123, "rss_bytes": 2**20, "num_fds": 12},
        "children": [{"rss_bytes": 3 * 2**20, "num_fds": 70}],
        "children_rss_bytes": 3 * 2**20,
        "children_fds": 70,
        "process_tree_rss_bytes": 4 * 2**20,
        "fetch_seconds": 2.25,
        "batch_tensors": {"by_device": {"cpu": {"count": 2, "logical_bytes": 5 * 2**20}}},
    }

    metrics = _record_to_wandb_metrics(record, trace_step=9)

    assert metrics["trace/step"] == 9
    assert metrics["trace/phase"] == "after_next"
    assert metrics["phase/after_next/parent/rss_mib"] == 1
    assert metrics["phase/after_next/children_rss_mib"] == 3
    assert metrics["phase/after_next/worker_max_num_fds"] == 70
    assert metrics["phase/after_next/batch/cpu/logical_mib"] == 5
    assert "phase/after_next/parent/pid" not in metrics


def test_wandb_run_writes_to_rank_folder(tmp_path, monkeypatch: pytest.MonkeyPatch):
    class FakeRun:
        def __init__(self):
            self.logged = []
            self.summary = {}
            self.finished = False

        def log(self, metrics, step):
            self.logged.append((metrics, step))

        def finish(self):
            self.finished = True

    fake_run = FakeRun()
    init_kwargs = {}

    def fake_init(**kwargs):
        init_kwargs.update(kwargs)
        return fake_run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fake_init))

    fake_result, rank_dir = _init_wandb_run(
        tmp_path / "rank_000.jsonl",
        mode="offline",
        project="profile-test",
        group="comparison",
        name="torchcodec-rank000",
        config={"rank": 0},
    )

    assert rank_dir == tmp_path / "wandb" / "rank_000"
    assert rank_dir.is_dir()
    assert init_kwargs["dir"] == str(rank_dir)
    assert init_kwargs["mode"] == "offline"
    assert fake_result is fake_run
    assert not fake_run.logged
    assert not fake_run.finished


def test_memory_recorder_logs_each_snapshot_to_wandb_immediately(tmp_path):
    class FakeRun:
        def __init__(self):
            self.logged = []

        def log(self, metrics, step):
            self.logged.append((metrics, step))

    fake_run = FakeRun()
    trace_path = tmp_path / "rank_000.jsonl"
    recorder = _MemoryRecorder(trace_path, rank=0, wandb_run=fake_run)
    try:
        recorder.record("before_next", iteration=3, batch_index=6, grad_accum_index=0)
        assert len(fake_run.logged) == 1
        metrics, step = fake_run.logged[0]
        assert step == 0
        assert metrics["trace/phase"] == "before_next"
        assert metrics["trace/iteration"] == 3
    finally:
        recorder.close()

    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["phase"] == "before_next"
