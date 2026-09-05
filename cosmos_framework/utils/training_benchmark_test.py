# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from types import SimpleNamespace

import torch

from cosmos_framework.utils.training_benchmark import TrainingBenchmark, _summarize


class _SizedLoader:
    def __len__(self) -> int:
        return 3


def test_summarize_percentiles():
    summary = _summarize([1.0, 2.0, 3.0])
    assert summary["count"] == 3
    assert summary["mean"] == 2.0
    assert summary["p50"] == 2.0
    assert summary["max"] == 3.0


def test_training_benchmark_stops_on_sample_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMARK_DATASET_NAME", "tiny")
    monkeypatch.setattr("cosmos_framework.utils.training_benchmark.distributed.get_rank", lambda: 0)
    config = SimpleNamespace(
        enabled=True,
        num_epochs=1,
        warmup_iterations=2,
        output_subdir="benchmark",
    )
    benchmark = TrainingBenchmark(config, str(tmp_path), _SizedLoader())
    benchmark.begin_iteration()
    with benchmark.phase("forward"):
        pass
    benchmark.record_batch(
        {
            "video": [torch.zeros(3, 5, 8, 8)],
            "num_frames": [torch.tensor([5]), torch.tensor([7]), torch.tensor([9])],
            "_packing_num_samples": 3,
            "_packing_num_tokens": 30,
            "_packing_efficiency": 0.5,
        }
    )
    benchmark.record_batch(
        {
            "video": [torch.zeros(3, 2, 8, 8)],
            "num_frames": [torch.tensor([2])],
            "_packing_num_samples": 1,
            "_packing_num_tokens": 10,
            "_packing_efficiency": 0.25,
        }
    )
    assert benchmark.finish_iteration(1)
    benchmark.close()
    benchmark.write_global_summary()

    rank_summary = json.loads((tmp_path / "benchmark" / "rank_000_summary.json").read_text())
    global_summary = json.loads((tmp_path / "benchmark" / "summary.json").read_text())
    assert rank_summary["dataset_name"] == "tiny"
    assert rank_summary["sample_overshoot"] == 1
    assert global_summary["consumed_global_samples"] == 4
    assert global_summary["metrics"]["global_frames"]["mean"] == 23
    assert global_summary["metrics"]["global_tokens"]["mean"] == 40
    assert (tmp_path / "benchmark" / "iterations.csv").is_file()
