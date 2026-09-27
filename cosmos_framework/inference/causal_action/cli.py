# SPDX-License-Identifier: OpenMDW-1.1
"""Offline FD/ID/Policy inference using the same template source manifest as training."""

import argparse
import json
import os
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sources-file", required=True)
    p.add_argument("--source-index", type=int, default=0)
    p.add_argument(
        "--template", default="cosmos_framework.data.generator.action.action_state_template.ActionStateTemplate55"
    )
    p.add_argument("--processor", required=True)
    p.add_argument("--vae", required=True)
    p.add_argument("--mode", choices=["policy", "forward_dynamics", "inverse_dynamics"], default="policy")
    p.add_argument("--video-stride", type=int, default=4)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--resolution", default="480")
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--actions-per-block", type=int, default=32)
    p.add_argument("--max-action-steps", type=int, default=96)
    p.add_argument("--overlap-action-steps", type=int, default=16)
    p.add_argument("--history", type=int, default=8)
    p.add_argument("--current-block", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--kv-cache", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--allow-mock-statistics", action="store_true")
    p.add_argument("--mock-state-stats-path", default="")
    p.add_argument("--mock-delta-stats-path", default="")
    p.add_argument(
        "--preview-ground-truth-states", action="store_true", help="Explicit offline truth-state conditioned preview"
    )
    p.add_argument("--decode-video", action="store_true")
    p.add_argument("--override", action="append", default=[])
    args = p.parse_args()
    os.environ.setdefault("COSMOS_DEVICE", "npu")
    os.environ["WAN_VAE_PATH"] = args.vae
    import torch
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

    from cosmos_framework.data.generator.action.datasets.causal_action_factory import (
        get_causal_action_dataset,
        latent_block_size,
    )
    from cosmos_framework.data.generator.joint_dataloader import custom_collate_fn
    from cosmos_framework.inference.causal_action.outputs import real_actions
    from cosmos_framework.utils import distributed
    from cosmos_framework.utils.generator.model_loader import load_model_from_checkpoint

    # Initialize the same device compatibility layer as the standard entry point.
    distributed.init()
    block_size = latent_block_size(args.actions_per_block, args.video_stride)
    overrides = [
        f"data_setting.action.template={args.template}",
        f"data_setting.action.actions_per_block={args.actions_per_block}",
        f"data_setting.action.video_stride={args.video_stride}",
        f"data_setting.action.max_action_steps={args.max_action_steps}",
        "model.config.vlm_config.tokenizer.repository=null",
        "model.config.vlm_config.tokenizer.revision=null",
        f"+model.config.vlm_config.tokenizer.tokenizer_type={args.processor}",
        f"model.config.tokenizer.vae_path={args.vae}",
        *args.override,
    ]
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
        experiment_opts=overrides,
        keys_to_skip_loading=["net_ema."],
        seed=args.seed,
    )
    model.net.eval()
    mixture = get_causal_action_dataset(
        sources_file=args.sources_file,
        template=args.template,
        actions_per_block=args.actions_per_block,
        video_stride=args.video_stride,
        max_action_steps=args.max_action_steps,
        overlap_action_steps=args.overlap_action_steps,
        mode=args.mode,
        resolution=args.resolution,
        cfg_dropout_rate=0,
        allow_mock_statistics=args.allow_mock_statistics,
        mock_state_stats_path=args.mock_state_stats_path,
        mock_delta_stats_path=args.mock_delta_stats_path,
        history_blocks_min=args.history,
        history_blocks_max=args.history,
        tokenizer_config=model.config.vlm_config.tokenizer,
    )
    dataset = mixture.datasets[args.source_index]
    # 离线任务由命令行显式选择，不沿用训练来源清单中的 joint 模式覆盖。
    dataset.mode = args.mode
    batch = custom_collate_fn([dataset[args.index]])
    batch["causal_action_preview"] = args.preview_ground_truth_states
    batch["causal_action_current_block"] = args.current_block if args.current_block is not None else 0
    if not args.preview_ground_truth_states:
        plan = batch["sequence_plan"][0]
        # Offline input supplies real observations before the requested block.
        plan.condition_frame_indexes_vision = sorted(
            set(plan.condition_frame_indexes_vision) | set(range(1 + batch["causal_action_current_block"] * block_size))
        )
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "prompt.json").write_text(json.dumps(batch[model.input_caption_key], ensure_ascii=False, indent=2))
    print("MODEL PROMPT:", batch[model.input_caption_key], flush=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad():
        result = model.generate_samples_from_batch(
            batch,
            num_steps=args.steps,
            guidance=args.guidance,
            seed=[args.seed],
            causal_block_size=block_size,
            causal_history_blocks=args.history,
            causal_use_kv_cache=args.kv_cache,
        )
        actions = real_actions(result, batch)
        torch.save(
            dict(vision_latents=[v.cpu() for v in result["vision"]], action=[a.cpu() for a in actions]),
            outdir / "sample.pt",
        )
        if args.decode_video:
            import imageio.v3 as iio

            frames = model.decode(result["vision"][0]).float().cpu().squeeze(0)
            frames = ((frames.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 3, 0).numpy()
            iio.imwrite(outdir / "video.mp4", frames, fps=float(batch["conditioning_fps"][0]))
    torch.cuda.synchronize()
    metrics = dict(
        seconds=time.perf_counter() - started,
        peak_gib=torch.cuda.max_memory_allocated() / 2**30,
        mode=args.mode,
        semantics=result["preview_semantics"],
        kv_cache=args.kv_cache,
        action_shape=list(actions[0].shape),
    )
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(metrics)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
