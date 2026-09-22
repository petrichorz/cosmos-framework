# SPDX-License-Identifier: OpenMDW-1.1
"""Collect separate measured-state and block-relative action training quantiles.

CLI DROID reader is a convenience; ``collect_statistics`` accepts any adapter
implementing the shared source contract. No model weights or absolute-action
normalizers are used.
"""

import argparse
import random

import torch

from cosmos_framework.data.generator.action.block_state import BlockStatistics, Quantiles, build_block_sample


def collect_statistics(dataset, *, indices, block_sizes, video_stride, seed=123):
    rng = random.Random(seed)
    states, smasks, actions, amasks = [], [], [], []
    key = None
    for i in indices:
        raw = dataset[i]
        current = raw["source_contract"].statistics_key(block_sizes, video_stride, len(raw["action_target"]))
        if raw["source_contract"].split != "train":
            raise ValueError("Quantiles must be collected on the training split")
        if key is not None and key != current:
            raise ValueError("Cannot pool different source contracts implicitly")
        key = current
        if raw.get("video") is None:
            raw["video"] = torch.empty(3, len(raw["action_target"]) + 1, 1, 1, dtype=torch.uint8)
        data, metadata = build_block_sample(
            raw, block_size=rng.choice(block_sizes), history_blocks=1, video_stride=video_stride
        )
        states.append(metadata.states)
        smasks.append(metadata.state_mask)
        actions.append(data["action"])
        amasks.append(metadata.action_mask)
    if key is None:
        raise ValueError("Statistics population is empty")
    return BlockStatistics(key, Quantiles.fit(states, smasks), Quantiles.fit(actions, amasks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-length", type=int, default=32)
    parser.add_argument("--video-stride", type=int, default=1)
    parser.add_argument("--block-size-min", type=int, default=1)
    parser.add_argument("--block-size-max", type=int, default=4)
    parser.add_argument("--windows", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset

    dataset = DROIDLeRobotDataset(
        root=args.root,
        chunk_length=args.chunk_length,
        action_space="causal_eef",
        use_state=True,
        action_normalization=None,
        use_success_only=True,
        dataset_version="droid_plus_lerobot_640x360_20260412",
    )
    dataset._skip_video_loading = True
    indices = random.Random(args.seed).sample(range(len(dataset)), min(args.windows, len(dataset)))
    statistics = collect_statistics(
        dataset,
        indices=indices,
        block_sizes=tuple(range(args.block_size_min, args.block_size_max + 1)),
        video_stride=args.video_stride,
        seed=args.seed,
    )
    statistics.save(args.output)
    import json

    with open(args.output) as f:
        record = json.load(f)
    record["population"] = dict(
        root=args.root,
        split="train",
        windows=len(indices),
        seed=args.seed,
        chunk_length=args.chunk_length,
        block_size_min=args.block_size_min,
        block_size_max=args.block_size_max,
        video_stride=args.video_stride,
        sampling="uniform_windows_uniform_block_sizes",
        indices=indices,
    )
    with open(args.output, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Saved {len(indices)} training windows to {args.output}; contract={statistics.key}")


if __name__ == "__main__":
    main()
