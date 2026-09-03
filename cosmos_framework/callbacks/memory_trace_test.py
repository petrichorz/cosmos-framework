# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import os

import torch

from cosmos_framework.callbacks.memory_trace import MemoryTrace, _read_proc_status_bytes, _summarize_tensors


def test_summarize_tensors_handles_nested_and_duplicate_values() -> None:
    tensor = torch.zeros(2, 3, dtype=torch.float32)
    result = _summarize_tensors({"a": tensor, "nested": [tensor, torch.ones(4, dtype=torch.int16)]})

    assert result["by_device"]["cpu"] == {"count": 2, "logical_bytes": 32}
    assert result["container_count"] == 2


def test_should_record_respects_interval_and_disabled_value() -> None:
    assert MemoryTrace(every_n=2)._should_record(4)
    assert not MemoryTrace(every_n=2)._should_record(3)
    assert not MemoryTrace(every_n=0)._should_record(0)


def test_read_proc_status_reports_physical_memory_categories() -> None:
    status = _read_proc_status_bytes(os.getpid())

    assert status["process_rss_anon_bytes"] >= 0
    assert status["process_rss_file_bytes"] >= 0
