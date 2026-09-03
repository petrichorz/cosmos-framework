# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""NPU microbenchmark for causal teacher-forcing masked and block attention."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable

import torch

from cosmos_framework.data.generator.sequence_packing.teacher_forcing import (
    build_per_sample_teacher_forcing_gen_masks,
    build_teacher_forcing_layout,
)
from cosmos_framework.model.generator.mot.teacher_forcing_attention import (
    teacher_forcing_block_attention,
    teacher_forcing_per_sample_dense_attention,
)
from cosmos_framework.model.generator.mot.teacher_forcing_block_attention import (
    build_teacher_forcing_block_metadata,
    build_teacher_forcing_block_sparse_mask,
    build_teacher_forcing_kv_permutation,
    reorder_teacher_forcing_kv,
)


def percentile_nearest_rank(values: list[float], percentile: float) -> float:
    """Return a deterministic nearest-rank percentile for benchmark reporting."""

    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= percentile <= 1.0:
        raise ValueError(f"percentile must be in [0, 1], got {percentile}")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * percentile + 0.999999) - 1))
    return ordered[index]


def _measure(
    operation: Callable[[], None],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float | int]:
    torch.npu.synchronize()
    first_start = time.perf_counter()
    operation()
    torch.npu.synchronize()
    first_ms = (time.perf_counter() - first_start) * 1000.0

    for _ in range(warmup):
        operation()
    torch.npu.synchronize()

    latencies_ms: list[float] = []
    for _ in range(iterations):
        torch.npu.synchronize()
        start = time.perf_counter()
        operation()
        torch.npu.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    return {
        "first_ms": first_ms,
        "median_ms": statistics.median(latencies_ms),
        "p90_ms": percentile_nearest_rank(latencies_ms, 0.90),
        "min_ms": min(latencies_ms),
        "iterations": iterations,
        "warmup": warmup,
    }


def _parse_shape(value: str) -> tuple[int, int, int]:
    parts = tuple(int(part) for part in value.split("x"))
    if len(parts) != 3 or any(part < 1 for part in parts):
        raise argparse.ArgumentTypeError("vision shape must be TxHxW with positive integers")
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--und-tokens", type=int, default=129)
    parser.add_argument("--vision-shape", type=_parse_shape, default=(4, 8, 16))
    parser.add_argument("--logical-block-size", type=int, default=2)
    parser.add_argument("--history-blocks", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    if args.und_tokens < 1 or args.warmup < 0 or args.iterations < 1:
        parser.error("und-tokens and iterations must be positive; warmup must be non-negative")
    if not torch.npu.is_available():
        parser.error("an Ascend NPU is required")

    device = torch.device("npu")
    layout = build_teacher_forcing_layout(
        und_token_counts=[args.und_tokens],
        vision_token_shapes=[args.vision_shape],
        block_size=args.logical_block_size,
        history_blocks=args.history_blocks,
    ).to(device)

    metadata_start = time.perf_counter()
    permutation = build_teacher_forcing_kv_permutation(layout)
    metadata = build_teacher_forcing_block_metadata(layout, permutation)
    block_mask = build_teacher_forcing_block_sparse_mask(metadata, num_q_heads=16)
    torch.npu.synchronize()
    metadata_ms = (time.perf_counter() - metadata_start) * 1000.0
    dense_masks = tuple(
        mask.to(device)
        for mask in build_per_sample_teacher_forcing_gen_masks(
            layout,
            max_sequence_length=sum(layout.sample_lens),
        )
    )

    generator = torch.Generator().manual_seed(args.seed)
    num_queries = sum(metadata.q_actual_lengths)
    num_keys = sum(metadata.kv_actual_lengths)
    query = torch.randn(num_queries, 16, 128, dtype=torch.bfloat16, generator=generator).to(device)
    key = torch.randn(num_keys, 8, 128, dtype=torch.bfloat16, generator=generator).to(device)
    value = torch.randn(num_keys, 8, 128, dtype=torch.bfloat16, generator=generator).to(device)

    def masked_forward() -> None:
        teacher_forcing_per_sample_dense_attention(
            query,
            key,
            value,
            dense_masks,
            sample_lens=metadata.kv_actual_lengths,
            gen_sample_lens=metadata.q_actual_lengths,
        )

    def block_forward() -> None:
        teacher_forcing_block_attention(
            query,
            key,
            value,
            permutation=permutation,
            metadata=metadata,
            block_sparse_mask=block_mask,
        )

    def forward_backward(operation: Callable[[], torch.Tensor]) -> Callable[[], None]:
        inputs = [tensor.detach().requires_grad_(True) for tensor in (query, key, value)]

        def measured() -> None:
            for tensor in inputs:
                tensor.grad = None
            operation_with_inputs(operation, inputs).float().square().mean().backward()

        return measured

    def operation_with_inputs(operation: Callable[[], torch.Tensor], inputs: list[torch.Tensor]) -> torch.Tensor:
        if operation is masked_forward:
            return teacher_forcing_per_sample_dense_attention(
                *inputs,
                dense_masks,
                sample_lens=metadata.kv_actual_lengths,
                gen_sample_lens=metadata.q_actual_lengths,
            )
        return teacher_forcing_block_attention(
            *inputs,
            permutation=permutation,
            metadata=metadata,
            block_sparse_mask=block_mask,
        )

    def permutation_only() -> None:
        reorder_teacher_forcing_kv(key, value, permutation)

    if hasattr(torch.npu, "reset_peak_memory_stats"):
        torch.npu.reset_peak_memory_stats(device)

    results = {
        "shape": {
            "und_tokens": args.und_tokens,
            "vision_shape": args.vision_shape,
            "q_tokens": num_queries,
            "kv_tokens": num_keys,
            "q_heads": 16,
            "kv_heads": 8,
            "head_dim": 128,
            "dtype": "bfloat16",
        },
        "block_density": float(torch.count_nonzero(block_mask).item() / block_mask.numel()),
        "metadata_build_ms": metadata_ms,
        "masked_forward": _measure(masked_forward, warmup=args.warmup, iterations=args.iterations),
        "block_forward": _measure(block_forward, warmup=args.warmup, iterations=args.iterations),
        "masked_forward_backward": _measure(
            forward_backward(masked_forward), warmup=args.warmup, iterations=args.iterations
        ),
        "block_forward_backward": _measure(
            forward_backward(block_forward), warmup=args.warmup, iterations=args.iterations
        ),
        "kv_permutation": _measure(permutation_only, warmup=args.warmup, iterations=args.iterations),
    }
    if hasattr(torch.npu, "max_memory_allocated"):
        results["peak_memory_bytes"] = int(torch.npu.max_memory_allocated(device))

    masked_ms = float(results["masked_forward_backward"]["median_ms"])
    block_ms = float(results["block_forward_backward"]["median_ms"])
    results["forward_backward_speedup"] = masked_ms / block_ms
    rendered = json.dumps(results, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(rendered + "\n")


if __name__ == "__main__":
    main()
