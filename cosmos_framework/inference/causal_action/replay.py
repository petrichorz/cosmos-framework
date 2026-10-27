# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Replay processed RoboTwin observations against a live server, without simulation.

Dataset frames/state advance by execute_steps independently of predicted actions. This checks
serving integration, not closed-loop policy success or dynamics accuracy.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from cosmos_framework.inference.causal_action.client import CausalPolicyClient, save_structured_prompt


def load_observations(args):
    from cosmos_framework.data.generator.action.action_state_template import ActionStateTemplate55
    from cosmos_framework.data.generator.action.causal_block_geometry import CausalBlockGeometry
    from cosmos_framework.data.generator.action.datasets.robotwin_segment_lerobot_dataset import (
        RoboTwinSegmentLeRobotDataset,
    )
    from cosmos_framework.data.generator.action.lerobot_discovery import discover_dataset_roots
    from cosmos_framework.data.generator.action.segment_planner import SegmentPlanner

    roots = discover_dataset_roots([args.dataset_root])
    geometry = CausalBlockGeometry(temporal_compression_factor=4)
    reader = RoboTwinSegmentLeRobotDataset(
        root=roots[args.dataset_index],
        template=ActionStateTemplate55(),
        planner=SegmentPlanner(
            max_action_steps=args.max_action_steps,
            overlap_action_steps=args.overlap_action_steps,
            geometry=geometry,
        ),
        video_backend="pyav_resize",
        table_backend="parquet",
        split="full",
        split_seed=args.split_seed,
    )
    return read_observations(reader, args.index)


def read_observations(reader, index, *, required=None):
    """Read native cameras/state once for both WebSocket and offline policy replay."""
    from cosmos_framework.inference.causal_action.robotwin import RobotwinAdapter
    from lerobot.datasets.video_utils import decode_video_frames

    _, episode_id, start, actions = reader._resolve_index(index)
    state, target, sm, am, times, task = reader._read_parquet_segment(episode_id, start, actions)
    required = len(state) if required is None else required
    if len(state) < required:
        raise ValueError(f"Selected segment has {len(state)} observations; {required} required")
    adapter = RobotwinAdapter(reader.template, state_mask=sm, action_mask=am)
    meta = reader._direct_meta
    cameras = {}
    for role, key in reader.video_view.cameras.items():
        feature = meta.features[key]
        names, shape = feature["names"], feature["shape"]
        height, width = shape[names.index("height")], shape[names.index("width")]
        offset = meta.episodes[episode_id][f"videos/{key}/from_timestamp"]
        video = decode_video_frames(
            meta.root / meta.get_video_file_path(episode_id, key),
            [offset + timestamp for timestamp in times[:required].tolist()],
            reader._tolerance_s,
            backend="pyav_resize",
            resize_h=height,
            resize_w=width,
        )
        short_key = {"head": "cam_high", "left": "cam_left_wrist", "right": "cam_right_wrist"}[role]
        cameras[short_key] = video.permute(0, 2, 3, 1).numpy()
    return dict(
        cameras=cameras,
        state=[adapter.state_for_replay(row) for row in state[:required].numpy()],
        prompt=reader._caption_for_index(index) or str(meta.tasks.iloc[int(task)].name),
        fps=reader._read_source_fps(episode_id),
        root=str(reader.root),
        episode=episode_id,
        start=start,
        split_seed=reader._split_seed,
        max_action_steps=reader.planner.max_action_steps,
        overlap_action_steps=reader.planner.overlap_action_steps,
        state_unified=state[:required],
        action_target=target[: required - 1],
        state_mask=sm,
        action_mask=am,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--uri", default="ws://localhost:8000")
    p.add_argument(
        "--dataset-root", default=os.environ.get("ROBOTWIN_ROOT"), required="ROBOTWIN_ROOT" not in os.environ
    )
    p.add_argument("--dataset-index", type=int, default=0)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=42, help="Dataset ordering seed; separate from model sampling")
    p.add_argument("--max-action-steps", type=int, default=96, help="Match the offline segment planner")
    p.add_argument("--overlap-action-steps", type=int, default=16, help="Match the offline segment planner")
    p.add_argument("--control-mode", choices=["joint", "eef"], default="joint")
    p.add_argument(
        "--output", default="outputs", help="Output root; results are saved under seed{server sampling seed}"
    )
    args = p.parse_args()
    data = load_observations(args)
    total = len(data["action_target"])
    metrics = []
    with CausalPolicyClient(args.uri, control_mode=args.control_mode) as client:
        count = client.metadata["execute_steps"]
        predicted = client.metadata["actions_per_block"]
        if type(count) is not int or not 0 < count <= predicted:
            raise ValueError("Server execution prefix must be positive and no greater than its prediction block")
        if total % count:
            raise ValueError(f"Selected segment must contain complete {count}-step execution prefixes")
        rounds = total // count
        print(f"Replay selected segment: {total} action steps, {rounds} prediction rounds", flush=True)
        if not np.isclose(client.metadata["fps"], data["fps"]):
            raise ValueError(f"Server FPS {client.metadata['fps']} differs from dataset aligned FPS {data['fps']}")
        if client.metadata.get("video_decode_mode") != "whole_sequence":
            raise ValueError("Replay requires a server supporting whole-sequence prediction video decoding")
        seed = client.metadata["seed"]
        output = Path(args.output) / f"seed{seed}"
        output.mkdir(parents=True, exist_ok=True)
        print(f"Replay output: {output}", flush=True)
        client.reset(prompt=data["prompt"], record_video=True)
        for revision, step in enumerate(range(0, total, count)):
            begin = 0 if revision == 0 else step - count + 1
            request = dict(
                obs={key: frames[begin : step + 1] for key, frames in data["cameras"].items()},
                state=data["state"][step],
                observation_step=step,
            )
            if revision:
                request["executed_steps"] = count
            result = client.infer(request)
            save_structured_prompt(result, output / "structured_prompt.json")
            action = client.actions(result)
            width = (
                client.metadata["left_arm_dof"] + client.metadata["right_arm_dof"] + 2
                if args.control_mode == "joint"
                else 16
            )
            if action.shape != (predicted, width) or not np.isfinite(action).all():
                raise RuntimeError("Invalid server action output")
            for mode, values in result["action"].items():
                if values is not None:
                    np.save(output / f"step{step:06d}_action_{mode}.npy", values)
            if "action_unified" in result:
                np.save(output / f"step{step:06d}_action_unified.npy", result["action_unified"])
            if "video_latents" in result:
                np.save(output / f"step{step:06d}_video_latents.npy", result["video_latents"])
            row = dict(step=step, revision=result["revision"], **result["server_timing"])
            metrics.append(row)
            print(json.dumps(row), flush=True)
        prediction = client.save_prediction_video(output / "prediction.mp4")
    (output / "metrics.json").write_text(
        json.dumps(
            dict(
                source=data["root"],
                episode=data["episode"],
                start=data["start"],
                split_seed=data["split_seed"],
                max_action_steps=data["max_action_steps"],
                overlap_action_steps=data["overlap_action_steps"],
                fps=data["fps"],
                seed=seed,
                action_steps=total,
                rounds=rounds,
                actions_per_block=predicted,
                execute_steps=count,
                semantics="dataset observation replay; predicted actions do not drive subsequent observations",
                requests=metrics,
                video_decode_mode="whole_sequence",
                **prediction,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
