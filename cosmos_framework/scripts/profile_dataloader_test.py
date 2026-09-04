# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import sys
from types import SimpleNamespace

import pytest

from cosmos_framework.scripts.profile_dataloader import (
    _advance_training_position,
    _record_to_wandb_metrics,
    _upload_trace_to_wandb,
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


def test_wandb_upload_replays_completed_trace_into_rank_folder(tmp_path, monkeypatch: pytest.MonkeyPatch):
    trace_path = tmp_path / "rank_000.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "phase": "run_end",
                "elapsed_seconds": 1.0,
                "iteration": 2,
                "rank": 0,
                "parent": {"pid": 123, "rss_bytes": 2**20},
                "children": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

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

    rank_dir = _upload_trace_to_wandb(
        trace_path,
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
    assert fake_run.logged[0][1] == 0
    assert fake_run.summary["trace_records"] == 1
    assert fake_run.finished
