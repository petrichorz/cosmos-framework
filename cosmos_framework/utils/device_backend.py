# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Device-agnostic backend layer.

Abstracts pynvml (CUDA) / torch.npu (Ascend NPU) / CPU so the rest of the
framework never imports pynvml at module top level. Selection is automatic:
torch_npu available -> NPU; elif torch.cuda available -> CUDA; else CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401  -- importing registers torch.npu
        return bool(torch.npu.is_available())
    except Exception:
        return False


def _cuda_available() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def select_backend_kind(npu_available: Optional[bool] = None, cuda_available: Optional[bool] = None) -> str:
    """Return 'npu' | 'cuda' | 'cpu'. Params injectable for tests."""
    if npu_available is None:
        npu_available = _npu_available()
    if cuda_available is None:
        cuda_available = _cuda_available()
    if npu_available:
        return "npu"
    if cuda_available:
        return "cuda"
    return "cpu"


_KIND = select_backend_kind()
IS_NPU: bool = _KIND == "npu"
IS_CUDA: bool = _KIND == "cuda"
DIST_BACKEND: str = "hccl" if IS_NPU else ("nccl" if IS_CUDA else "gloo")
DEVICE_TYPE: str = "npu" if IS_NPU else ("cuda" if IS_CUDA else "cpu")
"""Runtime accelerator device type for DeviceMesh / DTensor placement.

Use this (not ``flags.DEVICE``) as the ``device_type`` for
``init_device_mesh`` / ``build_meshes`` so the FSDP/HSDP mesh is built on
``"npu"`` under Ascend (and ``"cuda"`` under NVIDIA). ``flags.DEVICE`` stays
``"cuda"`` and is still used for ``torch.device(...)`` - ``transfer_to_npu``
redirects those - but DeviceMesh dispatch needs the real backend string.
"""


@dataclass
class MemoryInfo:
    """Device-agnostic memory snapshot (bytes)."""

    total: int
    used: int
    free: int


class DeviceBackend:
    """Abstract device backend."""

    def init(self) -> None:
        raise NotImplementedError

    def get_handle(self, idx: int) -> Any:
        raise NotImplementedError

    def get_memory_info(self, handle: Any) -> Optional[MemoryInfo]:
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError


class CpuBackend(DeviceBackend):
    """No-op backend for CPU runs (e.g. checkpoint conversion, smoke tests)."""

    def init(self) -> None:
        pass

    def get_handle(self, idx: int) -> Any:
        return None

    def get_memory_info(self, handle: Any) -> Optional[MemoryInfo]:
        return None

    def shutdown(self) -> None:
        pass


class CudaBackend(DeviceBackend):
    """pynvml-backed CUDA backend. pynvml is imported lazily inside methods."""

    def init(self) -> None:
        import pynvml

        pynvml.nvmlInit()

    def get_handle(self, idx: int) -> Any:
        import pynvml

        return pynvml.nvmlDeviceGetHandleByIndex(idx)

    def get_memory_info(self, handle: Any) -> Optional[MemoryInfo]:
        import pynvml

        try:
            try:
                mi = pynvml.nvmlDeviceGetMemoryInfo_v2(handle)
            except AttributeError:
                mi = pynvml.nvmlDeviceGetMemoryInfo(handle)
            except pynvml.NVMLError_NotSupported:
                mi = pynvml.nvmlDeviceGetMemoryInfo(handle)
        except Exception:
            return None
        return MemoryInfo(total=int(mi.total), used=int(mi.used), free=int(mi.free))

    def shutdown(self) -> None:
        import pynvml

        pynvml.nvmlShutdown()


class NpuBackend(DeviceBackend):
    """torch.npu-backed Ascend backend. No NVML equivalent; uses torch.npu APIs."""

    def init(self) -> None:
        pass

    def get_handle(self, idx: int) -> Any:
        return idx  # torch.npu queries by device index

    def get_memory_info(self, handle: Any) -> Optional[MemoryInfo]:
        try:
            # torch.npu.mem_get_info mirrors torch.cuda.mem_get_info -> (free, total)
            free, total = torch.npu.mem_get_info(handle)
            return MemoryInfo(total=int(total), used=int(total) - int(free), free=int(free))
        except Exception:
            return None

    def shutdown(self) -> None:
        pass


def _select_backend() -> DeviceBackend:
    if IS_NPU:
        return NpuBackend()
    if IS_CUDA:
        return CudaBackend()
    return CpuBackend()


BACKEND: DeviceBackend = _select_backend()
