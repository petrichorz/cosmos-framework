# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Offline dataset replay through the server's receding-horizon policy engine."""

import json
import time
from bisect import bisect_right
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.data.generator.action.block_state import build_block_sample
from cosmos_framework.data.generator.action.causal_block_geometry import CausalBlockGeometry
from cosmos_framework.data.generator.action.datasets.robotwin_segment_lerobot_dataset import (
    RoboTwinSegmentLeRobotDataset,
)
from cosmos_framework.data.generator.action.segment_planner import SegmentPlanner
from cosmos_framework.data.generator.action.transforms import build_sequence_plan_from_mode
from cosmos_framework.inference.causal_action.online import CausalPolicyEngine
from cosmos_framework.inference.causal_action.outputs import normalized_action_metrics
from cosmos_framework.inference.causal_action.replay import read_observations
from cosmos_framework.inference.causal_action.robotwin import RobotwinAdapter
from cosmos_framework.inference.causal_action.video import save_latent_video


@torch.no_grad()
def run_receding_policy(model, dataset, args):
    if not 0 <= args.index < len(dataset):
        raise ValueError("Sample index is outside the selected source group")
    child = bisect_right(dataset.cumulative_sizes, args.index)
    offset = dataset.cumulative_sizes[child - 1] if child else 0
    source = dataset.datasets[child]
    reader = source._dataset
    if not isinstance(reader, RoboTwinSegmentLeRobotDataset):
        raise ValueError("--execute-steps currently supports RoboTwin policy sources")
    data = read_observations(reader, args.index - offset)
    adapter = RobotwinAdapter.from_statistics(source.template, source.statistics)
    if (adapter.state_mask & ~data["state_mask"]).any() or (adapter.action_mask & ~data["action_mask"]).any():
        raise ValueError("Dataset is missing dimensions required by the server robot configuration")
    engine = CausalPolicyEngine(
        model,
        statistics=source.statistics,
        template=source.template,
        actions_per_block=args.actions_per_block,
        execute_steps=args.execute_steps,
        video_stride=args.video_stride,
        history_blocks=args.history,
        steps=args.steps,
        guidance=args.guidance,
        resolution=args.resolution,
        fps=data["fps"],
        seed=args.seed,
    )
    count = args.execute_steps
    total = len(data["action_target"])
    if total % count:
        raise ValueError("Selected segment must contain complete execution prefixes")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "prompt.json").write_text(json.dumps([data["prompt"]], ensure_ascii=False, indent=2))
    print("REPLAY PROMPT:", data["prompt"], flush=True)
    actions, normalized, latents, prefixes, requests = [], [], [], [], []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for revision, step in enumerate(range(0, total, count)):
        begin = 0 if step == 0 else step - count + 1
        cameras = {key: frames[begin : step + 1] for key, frames in data["cameras"].items()}
        state = source.template.sanitize(
            torch.from_numpy(adapter.assemble_state(data["state"][step])), adapter.state_mask
        )
        call_started = time.perf_counter()
        engine.observe(
            cameras,
            state=state,
            state_mask=adapter.state_mask,
            action_mask=adapter.action_mask,
            prompt=data["prompt"],
        )
        result = engine.predict(
            state=state, state_mask=adapter.state_mask, action_mask=adapter.action_mask, revision=revision
        )
        if (
            result["action"].shape != (args.actions_per_block, source.template.width)
            or not np.isfinite(result["action"]).all()
        ):
            raise RuntimeError("Model returned invalid actions")
        actions.append(torch.from_numpy(result["action"][:count].copy()))
        normalized.append(torch.from_numpy(result["action_normalized"][:count].copy()))
        keep_latents = count // engine.geometry.actions_per_latent_frame
        latents.append(torch.from_numpy(result["video_latents"][:, :, :keep_latents].copy()))
        prefixes.append(engine.latest.detach().cpu())
        row = dict(
            observation_step=step,
            predicted_steps=args.actions_per_block,
            saved_steps=count,
            infer_ms=(time.perf_counter() - call_started) * 1000,
        )
        requests.append(row)
        print(json.dumps(row), flush=True)
    action = torch.cat(actions)
    prediction = torch.cat(normalized)
    video_latents = torch.cat([prefixes[0], *latents], dim=2)
    video_metrics = dict(video_path=None, video_frames=0, video_fps=16)
    if args.decode_video:
        video_metrics = save_latent_video(model, video_latents, output / "video.mp4")
    payload = dict(
        action=[action],
        action_normalized=[prediction],
        vision_latents=[video_latents],
        observed_prefix_latents=prefixes,
        prediction_starts=torch.arange(0, total, count),
        execute_steps=count,
        video_decode_mode="Decode initial observed latent plus all selected prediction latents once",
    )
    metrics = dict(
        source=data["root"],
        episode=data["episode"],
        start=data["start"],
        split_seed=data["split_seed"],
        max_action_steps=data["max_action_steps"],
        overlap_action_steps=data["overlap_action_steps"],
        checkpoint=args.checkpoint,
        seed=args.seed,
        steps=args.steps,
        guidance=args.guidance,
        actions_per_block=args.actions_per_block,
        execute_steps=count,
        seconds=time.perf_counter() - started,
        peak_gib=torch.cuda.max_memory_allocated() / 2**30,
        mode="policy",
        semantics="Dataset observation replay using the server engine; save execution prefixes only",
        kv_cache=True,
        action_shape=list(action.shape),
        requests=requests,
        **video_metrics,
    )
    if args.evaluate_actions:
        geometry = CausalBlockGeometry(
            actions_per_block=count, video_stride=args.video_stride, temporal_compression_factor=4
        )
        raw, metadata = build_block_sample(
            dict(
                state_trajectory=data["state_unified"],
                action_target=data["action_target"],
                state_mask=adapter.state_mask,
                action_mask=adapter.action_mask,
                conditioning_fps=data["fps"],
                source=data["root"],
            ),
            template=source.template,
            planner=SegmentPlanner(max_action_steps=total, overlap_action_steps=0, geometry=geometry),
            history_blocks=args.history,
            statistics=source.statistics,
        )
        plan = replace(
            build_sequence_plan_from_mode(
                "policy", video_length=1 + total // args.video_stride, action_length=total, video_temporal_downsample=4
            ),
            causal_action_metadata=metadata,
        )
        evaluation = dict(action=[raw["action"]], sequence_plan=[plan])
        action_metrics = normalized_action_metrics(dict(action=[prediction]), evaluation)
        action_metrics["reanchor_steps"] = count
        (output / "action_loss.json").write_text(json.dumps(action_metrics, ensure_ascii=False, indent=2))
        payload.update(gt_action_normalized=[raw["action"]], action_valid_mask=[metadata.action_mask])
        metrics.update(normalized_action_mse=action_metrics["mse"], action_loss_file="action_loss.json")
    torch.save(payload, output / "sample.pt")
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(metrics, flush=True)
