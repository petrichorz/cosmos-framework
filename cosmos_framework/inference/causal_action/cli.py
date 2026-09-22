# SPDX-License-Identifier: OpenMDW-1.1
"""Offline FD/ID/Policy inference on a DROID window, with strict training semantics."""

import argparse
import json
import os
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--processor", required=True)
    p.add_argument("--vae", required=True)
    p.add_argument("--mode", choices=["policy", "forward_dynamics", "inverse_dynamics"], default="policy")
    p.add_argument("--video-stride", type=int, default=1)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--resolution", default="480")
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--block", type=int, default=1)
    p.add_argument("--history", type=int, default=8)
    p.add_argument("--current-block", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--kv-cache", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--statistics", required=True, help="Separate state/delta training statistics for this block geometry"
    )
    p.add_argument(
        "--preview-ground-truth-states", action="store_true", help="Explicit offline truth-state conditioned preview"
    )
    p.add_argument("--decode-video", action="store_true")
    p.add_argument("--override", action="append", default=[])
    args = p.parse_args()
    os.environ.setdefault("COSMOS_DEVICE", "npu")
    os.environ["DROID_ROOT"] = args.dataset_root
    os.environ["WAN_VAE_PATH"] = args.vae
    os.environ["ACTION_STATISTICS_PATH"] = args.statistics
    import torch
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

    from cosmos_framework.data.generator.action.datasets.causal_action_sft_dataset import (
        get_causal_action_droid_sft_dataset,
    )
    from cosmos_framework.data.generator.joint_dataloader import custom_collate_fn
    from cosmos_framework.inference.causal_action.outputs import real_actions
    from cosmos_framework.utils import distributed
    from cosmos_framework.utils.generator.model_loader import load_model_from_checkpoint

    # Initialize the same device compatibility layer as the standard entry point.
    distributed.init()
    overrides = [
        "model.config.vlm_config.tokenizer.repository=null",
        "model.config.vlm_config.tokenizer.revision=null",
        f"+model.config.vlm_config.tokenizer.tokenizer_type={args.processor}",
        f"model.config.tokenizer.vae_path={args.vae}",
        *args.override,
    ]
    model, _ = load_model_from_checkpoint(
        "action_causal_droid_edge",
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
    dataset = get_causal_action_droid_sft_dataset(
        root=args.dataset_root,
        video_stride=args.video_stride,
        mode=args.mode,
        iterable_shuffle=False,
        resolution=args.resolution,
        cfg_dropout_rate=0,
        use_state=True,
        statistics_path=args.statistics,
        block_size_min=args.block,
        block_size_max=args.block,
        history_blocks_min=args.history,
        history_blocks_max=args.history,
        tokenizer_config=model.config.vlm_config.tokenizer,
        format_prompt_as_json=True,
    )
    batch = custom_collate_fn([dataset[args.index]])
    batch["causal_action_preview"] = args.preview_ground_truth_states
    batch["causal_action_current_block"] = (
        args.current_block if args.current_block is not None else 0
    )
    if not args.preview_ground_truth_states:
        plan = batch["sequence_plan"][0]
        # Offline input supplies real observations before the requested block.
        plan.condition_frame_indexes_vision = sorted(
            set(plan.condition_frame_indexes_vision)
            | set(range(1 + batch["causal_action_current_block"] * args.block))
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
            causal_block_size=args.block,
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
            iio.imwrite(outdir / "video.mp4", frames, fps=15 / args.video_stride)
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
