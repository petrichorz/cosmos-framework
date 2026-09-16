# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import csv
import json
import sys
from types import SimpleNamespace

from cosmos_framework.utils import performance


def test_npu_mstx_scope_disabled(monkeypatch):
    monkeypatch.delenv("COSMOS_NPU_MSTX", raising=False)
    with performance.npu_mstx_scope("forward/test"):
        pass


def test_npu_mstx_scope_uses_current_stream(monkeypatch):
    calls = []
    fake_mstx = SimpleNamespace(
        range_start=lambda message, stream, domain: calls.append(("start", message, stream, domain)) or 17,
        range_end=lambda range_id, domain: calls.append(("end", range_id, domain)),
    )
    fake_torch_npu = SimpleNamespace(
        npu=SimpleNamespace(current_stream=lambda: "stream0", mstx=fake_mstx),
    )
    monkeypatch.setenv("COSMOS_NPU_MSTX", "1")
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)

    with performance.npu_mstx_scope("forward/denoise"):
        pass

    assert calls == [
        ("start", "COSMOS::FORWARD/DENOISE", "stream0", "cosmos_forward"),
        ("end", 17, "cosmos_forward"),
    ]


def test_scope_records_jsonl_and_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("COSMOS_PERF_OUTPUT_DIR", str(tmp_path / "events"))
    performance._close_files()

    with performance.performance_scope("decode", backend="torchcodec"):
        pass
    performance.record_performance_event("cache_hit", video="a.mp4")
    performance._close_files()

    event_files = list((tmp_path / "events").glob("events_*.jsonl"))
    assert len(event_files) == 1
    events = [json.loads(line) for line in event_files[0].read_text().splitlines()]
    assert [event["name"] for event in events] == ["decode", "cache_hit"]
    assert events[0]["duration_ms"] >= 0

    rows = performance.summarize_performance_events(tmp_path / "events", tmp_path / "summary")
    assert {row["name"] for row in rows} == {"decode", "cache_hit"}
    assert (tmp_path / "summary" / "performance_summary.csv").is_file()
    assert (tmp_path / "summary" / "performance_summary.md").is_file()


def test_summarize_ascend_profiler_outputs(tmp_path):
    profiler_dir = tmp_path / "rank0" / "ASCEND_PROFILER_OUTPUT"
    profiler_dir.mkdir(parents=True)
    with (profiler_dir / "operator_details.csv").open("w", newline="") as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=[
                "Name",
                "Device Self Duration With AICore(us)",
                "Device Self Duration(us)",
                "Input Shapes",
                "Call Stack",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "Name": "MatMul",
                    "Device Self Duration With AICore(us)": "0",
                    "Device Self Duration(us)": "10",
                    "Call Stack": "model.py:7",
                },
                {"Name": "MatMul", "Device Self Duration(us)": "20", "Call Stack": "model.py:7"},
                {"Name": "Cast", "Device Self Duration(us)": "2"},
            ]
        )
    with (profiler_dir / "kernel_details.csv").open("w", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=["Name", "Duration(us)"])
        writer.writeheader()
        writer.writerow({"Name": "matmul_kernel", "Duration(us)": "12.5"})
    (profiler_dir / "trace_view.json").write_text("{}")

    result = performance.summarize_ascend_profiler_outputs(tmp_path, tmp_path / "summary")

    assert result["operators"][0]["name"] == "MatMul"
    assert result["operators"][0]["total_us"] == 30
    assert result["kernels"][0]["name"] == "matmul_kernel"
    assert len(result["trace_view_files"]) == 1
    assert (tmp_path / "summary" / "ascend_hotspots.md").is_file()
