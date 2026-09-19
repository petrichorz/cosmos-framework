# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Diagnostic train entrypoint: record conditions and pair real-model cache paths.

Accepts the same CLI arguments as cosmos_framework.scripts.train. This wrapper
is for short experiments only: each online sample is generated twice.
"""

import copy
import json
import os
import runpy
from dataclasses import replace
from pathlib import Path

import torch
from torch_npu.contrib import transfer_to_npu  # noqa: F401

from cosmos_framework.callbacks.iter_speed import IterSpeed
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel


def record(kind, **values):
    root = Path(os.environ["IMAGINAIRE_OUTPUT_ROOT"])
    root.mkdir(parents=True, exist_ok=True)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    with (root / f"audit_rank{rank}.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(kind=kind, rank=rank, **values)) + "\n")


original_generate = OmniMoTModel.generate_samples_from_batch
original_step_end = IterSpeed.on_training_step_end


@torch.no_grad()
def paired_generate(self, batch, **kwargs):
    original_count = len(batch["sequence_plan"][0].condition_frame_indexes_vision)
    counts = [1, 2] if original_count == 2 else [original_count]
    for count in counts:
        request = copy.deepcopy(batch)
        request["sequence_plan"][0] = replace(
            request["sequence_plan"][0], condition_frame_indexes_vision=list(range(count))
        )
        outputs, traces = [], []
        original_denoise = self.denoise
        for cache in [True, False]:
            trace = []

            def traced_denoise(*args, **call):
                result = original_denoise(*args, **call)
                packed = call["data_batch_packed"]
                ts = packed.vision.timesteps
                if ts.numel() and ts.abs().max().item() > 0:
                    width = kwargs["causal_block_size"]
                    trace.append(
                        (
                            packed.vision.tokens[0][..., -width:, :, :].float().cpu().clone(),
                            result["preds_vision"][0][..., -width:, :, :].float().cpu().clone(),
                        )
                    )
                return result

            self.denoise = traced_denoise
            try:
                outputs.append(
                    original_generate(self, copy.deepcopy(request), **{**kwargs, "causal_use_kv_cache": cache})
                )
            finally:
                self.denoise = original_denoise
            traces.append(trace)
        step_errors = []
        for (x, a), (y, b) in zip(*traces, strict=True):
            step_errors.append(
                dict(
                    input_max=(x - y).abs().max().item(),
                    velocity_max=(a - b).abs().max().item(),
                    velocity_l2=((a - b).norm() / b.norm().clamp_min(1e-12)).item(),
                )
            )
        a, b = [output["vision"][0].float() for output in outputs]
        relative = ((a - b).norm() / b.norm().clamp_min(1e-12)).item()
        record(
            "cache_pair",
            conditions=count,
            guidance=kwargs["guidance"],
            shape=list(a.shape),
            step_errors=step_errors,
            max_abs=(a - b).abs().max().item(),
            relative_l2=relative,
            allocated_gib=torch.npu.max_memory_allocated() / 1024**3,
        )
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        tolerance = os.environ.get("CACHE_PAIR_RTOL")
        if tolerance:
            assert relative < float(tolerance), f"cache comparison relative L2={relative}"
    return outputs[0 if kwargs.get("causal_use_kv_cache", True) else 1]


def step_end(self, model, data_batch, output_batch, loss, iteration=0):
    record(
        "train_step",
        iteration=iteration,
        loss=loss.detach().item(),
        conditions=[int(mask.sum().item()) for mask in output_batch["condition_mask_vision"]],
        pixel_shapes=output_batch["vae_pixel_shapes"],
        allocated_gib=torch.npu.max_memory_allocated() / 1024**3,
        reserved_gib=torch.npu.max_memory_reserved() / 1024**3,
    )
    return original_step_end(self, model, data_batch, output_batch, loss, iteration)


if __name__ == "__main__":
    OmniMoTModel.generate_samples_from_batch = paired_generate
    IterSpeed.on_training_step_end = step_end
    runpy.run_module("cosmos_framework.scripts.train", run_name="__main__")
