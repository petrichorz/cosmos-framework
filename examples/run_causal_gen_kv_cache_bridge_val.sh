#!/usr/bin/env bash

# Run causal Text+Image-to-Video inference with the request-local GenKVCache.
# The source validation JSON is converted to the causal length contract:
# users configure B/S/K and must not provide num_frames.

set -euo pipefail

for required_command in conda jq; do
  if ! command -v "$required_command" >/dev/null 2>&1; then
    echo "Required command is not available: $required_command" >&2
    exit 1
  fi
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-cosmos-framework-causal}"
SAMPLE_ID_RAW="${SAMPLE_ID:-002345}"
NUM_BLOCKS="${NUM_BLOCKS:-16}"
BLOCK_SIZE="${BLOCK_SIZE:-2}"
HISTORY_BLOCKS="${HISTORY_BLOCKS:-16}"
NUM_STEPS="${NUM_STEPS:-35}"
GUIDANCE="${GUIDANCE:-1.0}"
SEED="${SEED:-1}"
DEVICE_ID="${DEVICE_ID:-0}"
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-0}"

if [[ ! "$SAMPLE_ID_RAW" =~ ^[0-9]+$ ]]; then
  echo "SAMPLE_ID must contain digits only: $SAMPLE_ID_RAW" >&2
  exit 1
fi
printf -v SAMPLE_ID "%06d" "$((10#$SAMPLE_ID_RAW))"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/mi/data2T/Embodied-AI/codes/cosmos_ascend/outputs/causal-edge-training/cosmos3/causal_sft/vision_causal_edge/checkpoints/iter_000000400}"
TRAIN_RUN_ROOT="${TRAIN_RUN_ROOT:-$(dirname "$(dirname "$CHECKPOINT_ROOT")")}"
CONFIG_FILE="${CONFIG_FILE:-$TRAIN_RUN_ROOT/config.yaml}"
VAL_ROOT="${VAL_ROOT:-/mi/data2T/Embodied-AI/datasets/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge/val}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/causal_gen_kv_cache/iter_000000400_bridge_val_${SAMPLE_ID}_b${NUM_BLOCKS}_s${BLOCK_SIZE}}"

SOURCE_JSON="$VAL_ROOT/inference_prompt_i2v/episode_${SAMPLE_ID}_clip000.json"
SOURCE_IMAGE="$VAL_ROOT/images/episode_${SAMPLE_ID}_clip000.jpg"
INPUT_DIR="$OUTPUT_ROOT/input"
INPUT_JSON="$INPUT_DIR/episode_${SAMPLE_ID}_clip000.json"

for required_path in "$CHECKPOINT_ROOT/model/.metadata" "$CONFIG_FILE" "$SOURCE_JSON" "$SOURCE_IMAGE"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path does not exist: $required_path" >&2
    exit 1
  fi
done

mkdir -p "$INPUT_DIR"
jq \
  --arg name "causal_i2v/episode_${SAMPLE_ID}_clip000" \
  --arg vision_path "$SOURCE_IMAGE" \
  --argjson causal_num_blocks "$NUM_BLOCKS" \
  --argjson causal_block_size "$BLOCK_SIZE" \
  --argjson causal_history_blocks "$HISTORY_BLOCKS" \
  --argjson num_steps "$NUM_STEPS" \
  --argjson guidance "$GUIDANCE" \
  '
    del(.num_frames)
    | .name = $name
    | .model_mode = "image2video"
    | .vision_path = $vision_path
    | .causal_num_blocks = $causal_num_blocks
    | .causal_block_size = $causal_block_size
    | .causal_history_blocks = $causal_history_blocks
    | .num_steps = $num_steps
    | .guidance = $guidance
  ' "$SOURCE_JSON" > "$INPUT_JSON"

cd "$REPO_ROOT"
compile_args=(--no-use-torch-compile)
if [[ "$USE_TORCH_COMPILE" == "1" ]]; then
  compile_args=(--use-torch-compile)
fi

ASCEND_RT_VISIBLE_DEVICES="$DEVICE_ID" \
COSMOS_DEVICE=npu \
conda run --no-capture-output -n "$ENV_NAME" \
  python -m cosmos_framework.scripts.inference \
  --checkpoint-path "$CHECKPOINT_ROOT" \
  --config-file "$CONFIG_FILE" \
  --parallelism-preset throughput \
  --dp-shard-size 1 \
  --dp-replicate-size 1 \
  --cp-size 1 \
  --cfgp-size 1 \
  "${compile_args[@]}" \
  --no-guardrails \
  --seed "$SEED" \
  -i "$INPUT_JSON" \
  -o "$OUTPUT_ROOT"
