# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Single-simulator causal policy WebSocket server, launched with torchrun."""

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

from cosmos_framework.inference.causal_action.wire import packb, unpackb

logger = logging.getLogger(__name__)


async def serve(policy, *, host="0.0.0.0", port=8000):
    from websockets.asyncio.server import serve as websocket_serve
    from websockets.exceptions import ConnectionClosed

    connected = False

    async def handler(connection):
        nonlocal connected
        if connected:
            await connection.close(code=1013, reason="A simulator is already connected")
            return
        connected = True
        policy.reset()
        try:
            await connection.send(
                packb(
                    dict(
                        protocol="cosmos-causal-action-v2",
                        mode="policy",
                        action_dim=policy.engine.template.width,
                        actions_per_block=policy.engine.geometry.actions_per_block,
                        execute_steps=policy.engine.execute_steps,
                        steps=policy.engine.steps,
                        seed=policy.engine.seed,
                        fps=policy.engine.fps,
                        video_stride=policy.engine.geometry.video_stride,
                        video_fps=policy.engine.geometry.video_fps(policy.engine.fps),
                        video_decode_mode="whole_sequence",
                        prediction_video_fps=16,
                        camera_keys=["cam_high", "cam_left_wrist", "cam_right_wrist"],
                        **policy.adapter.metadata,
                    )
                )
            )
            async for message in connection:
                start = time.perf_counter()
                try:
                    result = policy.infer(unpackb(message))
                    result["server_timing"] = {"infer_ms": (time.perf_counter() - start) * 1000}
                except Exception as error:
                    logger.exception("Causal policy request failed")
                    result = dict(error=str(error), reset_required=policy.needs_reset)
                await connection.send(packb(result))
        except ConnectionClosed:
            pass
        finally:
            policy.reset()
            connected = False

    async with websocket_serve(
        handler,
        host,
        port,
        compression=None,
        max_size=128 * 1024 * 1024,
        ping_interval=None,
        ping_timeout=None,
    ) as server:
        logger.info("Causal policy listening on ws://%s:%s", host, port)
        await server.serve_forever()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--processor", required=True)
    p.add_argument("--vae", required=True)
    p.add_argument("--statistics", help="Training state/delta quantiles; otherwise read from --sources-file")
    p.add_argument("--sources-file")
    p.add_argument("--source-index", type=int, default=0)
    p.add_argument(
        "--robot-config", help="JSON overriding source masks and arm DOFs; EEF poses must already match training data"
    )
    p.add_argument(
        "--template", default="cosmos_framework.data.generator.action.action_state_template.ActionStateTemplate55"
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--actions-per-block", type=int, default=32)
    p.add_argument("--execute-steps", type=int, default=16)
    p.add_argument("--video-stride", type=int, default=4)
    p.add_argument(
        "--history", type=int, default=32, help="Capacity in training blocks (two latent frames each by default)"
    )
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--fps", type=float, default=30.0, help="Aligned action/control FPS, before video subsampling")
    p.add_argument("--resolution", default="480")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--return-video-latents", action="store_true")
    p.add_argument("--return-unified-action", action="store_true", help="Include internal [T,55] actions for debugging")
    p.add_argument("--override", action="append", default=[])
    return p


def main():
    args = parser().parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This server supports one model rank and one simulator connection")
    if not args.statistics:
        if not args.sources_file:
            raise ValueError("Pass --statistics or --sources-file with training statistics_path")
        source = json.loads(os.path.expandvars(Path(args.sources_file).read_text()))["sources"][args.source_index]
        if source["reader"] != "robotwin" or source["viewpoint"] != "concat_view":
            raise ValueError("This server expects a RoboTwin concat_view source")
        args.statistics = source["statistics_path"]
    os.environ.setdefault("COSMOS_DEVICE", "npu")
    os.environ["WAN_VAE_PATH"] = args.vae
    if os.environ["COSMOS_DEVICE"] == "npu":
        from torch_npu.contrib import transfer_to_npu  # noqa: F401

    import torch

    from cosmos_framework.data.generator.action.action_state_template import resolve_action_template
    from cosmos_framework.data.generator.action.block_state import BlockStatistics
    from cosmos_framework.inference.causal_action.online import CausalPolicyEngine
    from cosmos_framework.inference.causal_action.policy import CausalPolicy
    from cosmos_framework.inference.causal_action.robotwin import RobotwinAdapter
    from cosmos_framework.utils import distributed
    from cosmos_framework.utils.generator.model_loader import load_model_from_checkpoint

    template = resolve_action_template(args.template)
    statistics = BlockStatistics.load(args.statistics)
    adapter = RobotwinAdapter.from_statistics(template, statistics, args.robot_config)
    distributed.init()
    try:
        model, _ = load_model_from_checkpoint(
            "action_causal_midtrain_edge",
            checkpoint_path=args.checkpoint,
            parallelism_config=dict(
                data_parallel_shard_degree=1,
                data_parallel_replicate_degree=1,
                context_parallel_shard_degree=1,
                cfg_parallel_shard_degree=1,
                enable_inference_mode=True,
            ),
            compile_config=dict(enabled=False),
            experiment_opts=[
                f"data_setting.action.template={args.template}",
                f"data_setting.action.actions_per_block={args.actions_per_block}",
                f"data_setting.action.video_stride={args.video_stride}",
                "model.config.vlm_config.tokenizer.repository=null",
                "model.config.vlm_config.tokenizer.revision=null",
                f"+model.config.vlm_config.tokenizer.tokenizer_type={args.processor}",
                f"model.config.tokenizer.vae_path={args.vae}",
                *args.override,
            ],
            keys_to_skip_loading=["net_ema."],
            seed=args.seed,
        )
        model.net.eval()
        engine = CausalPolicyEngine(
            model,
            statistics=statistics,
            template=template,
            actions_per_block=args.actions_per_block,
            execute_steps=args.execute_steps,
            video_stride=args.video_stride,
            history_blocks=args.history,
            steps=args.steps,
            guidance=args.guidance,
            shift=args.shift,
            resolution=args.resolution,
            fps=args.fps,
            seed=args.seed,
        )
        policy = CausalPolicy(
            engine,
            adapter=adapter,
            return_video_latents=args.return_video_latents,
            return_unified_action=args.return_unified_action,
        )
        asyncio.run(serve(policy, host=args.host, port=args.port))
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
