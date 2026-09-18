# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Single-image causal TI2V entry point for examples/run_causal_ti2v.sh."""

import json
import os
import sys
import time
from pathlib import Path


def main():
    env = os.environ
    output = Path(env["OUTPUT_ROOT"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    prompt = Path(env["PROMPT_FILE"]).read_text() if env.get("PROMPT_FILE") else env["PROMPT"]
    sample = {
        "name": "ti2v",
        "model_mode": "image2video",
        "vision_path": str(Path(env["IMAGE_PATH"]).resolve()),
        "prompt": prompt,
        "resolution": env["RESOLUTION"],
        "aspect_ratio": env["ASPECT_RATIO"],
        "fps": int(env["FPS"]),
        "causal_num_blocks": int(env["NUM_BLOCKS"]),
        "causal_block_size": int(env["BLOCK_SIZE"]),
        "causal_history_blocks": int(env["HISTORY_BLOCKS"]),
        "num_steps": int(env["NUM_STEPS"]),
        "guidance": float(env["GUIDANCE"]),
    }
    input_file = output / "input.json"
    input_file.write_text(json.dumps(sample, ensure_ascii=False, indent=2))
    sys.argv = [
        "causal_ti2v",
        "--checkpoint-path",
        env["CHECKPOINT_ROOT"],
        "--config-file",
        env["CONFIG_FILE"],
        "--parallelism-preset",
        "throughput",
        "--dp-shard-size",
        "1",
        "--dp-replicate-size",
        "1",
        "--cp-size",
        "1",
        "--cfgp-size",
        "1",
        "--no-use-torch-compile",
        "--no-guardrails",
        "--seed",
        env["SEED"],
        "-i",
        str(input_file),
        "-o",
        str(output),
    ]
    (output / "launch_args.json").write_text(json.dumps(sys.argv, ensure_ascii=False, indent=2))

    import torch

    from cosmos_framework.scripts.inference import main as inference_main

    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    start = time.perf_counter()
    inference_main()
    torch.npu.synchronize()
    metrics = {
        "total_seconds_including_model_load_and_save": time.perf_counter() - start,
        "peak_allocated_gib": torch.npu.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.npu.max_memory_reserved() / 1024**3,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Output: {output / 'ti2v'}")


if __name__ == "__main__":
    main()
