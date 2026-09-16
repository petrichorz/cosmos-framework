#!/usr/bin/env bash
# TND counterpart of the cookbook launch_causal_edge_profiling.sh.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

: "${DATASET_DIR:=$SCRIPT_DIR/data/Cosmos3-DROID/success}"
: "${CHECKPOINT_DIR:=$SCRIPT_DIR/checkpoints/Cosmos3-Edge-DCP}"
: "${COSMOS3_EDGE_PROCESSOR_PATH:=$SCRIPT_DIR/checkpoints/Cosmos3-Edge}"
: "${VAE_PATH:=$SCRIPT_DIR/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
: "${OUTPUT_ROOT:=$REPO_ROOT/outputs/vision_edge_tnd_profile}"
TOML_PATH="${TOML_PATH:-$SCRIPT_DIR/toml/sft_config/vision_causal_edge_tnd_profile.toml}"
# 调用者传入的相对路径在切换工作目录前解析，路径含空格也保持为一个参数。
for path_var in DATASET_DIR CHECKPOINT_DIR COSMOS3_EDGE_PROCESSOR_PATH VAE_PATH OUTPUT_ROOT TOML_PATH; do
    [[ "${!path_var}" = /* ]] || printf -v "$path_var" '%s/%s' "$PWD" "${!path_var}"
done
export BASE_CHECKPOINT_PATH="$CHECKPOINT_DIR" WAN_VAE_PATH="$VAE_PATH" DATASET_PATH="$DATASET_DIR"
export COSMOS3_EDGE_PROCESSOR_PATH IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export COSMOS_DEVICE=npu COSMOS_GEN_BSA64=0
export COSMOS_BSA64_BUCKETS="${COSMOS_BSA64_BUCKETS:-0}"

TORCHRUN_ARGS=(--nproc_per_node="${NPROC_PER_NODE:-8}" --master_port="${MASTER_PORT:-50012}")
[[ -n "${NNODES:-}" ]] && TORCHRUN_ARGS+=(--nnodes="$NNODES")
[[ -n "${NODE_RANK:-}" ]] && TORCHRUN_ARGS+=(--node_rank="$NODE_RANK")
[[ -n "${MASTER_ADDR:-}" ]] && TORCHRUN_ARGS+=(--master_addr="$MASTER_ADDR")
TRAIN_ARGS=(-m cosmos_framework.scripts.train --sft-toml="$TOML_PATH")
[[ "${DRY_RUN:-0}" == 1 ]] && TRAIN_ARGS+=(--dryrun)
CMD=(torchrun "${TORCHRUN_ARGS[@]}" "${TRAIN_ARGS[@]}" --
    model=mot_causal_fsdp
    model.config.vlm_config.tokenizer.repository=null
    model.config.vlm_config.tokenizer.revision=null
    "+model.config.vlm_config.tokenizer.tokenizer_type='${COSMOS3_EDGE_PROCESSOR_PATH}'"
    '~dataloader_train.dataloader.datasets.video.dataset.conditioning_config={0:0.7,1:0.2,2:0.1}'
    '+dataloader_train.dataloader.datasets.video.dataset.conditioning_config={0:1.0}'
    model.config.teacher_forcing_dense_mode=grouped_tnd
    "$@")

printf 'Framework: %s\nOutput: %s\n' "$REPO_ROOT" "$OUTPUT_ROOT"
printf 'Command: '; printf '%q ' "${CMD[@]}"; printf '\n'
[[ "${PRINT_ONLY:-0}" == 1 ]] && exit 0
for path_var in DATASET_DIR CHECKPOINT_DIR COSMOS3_EDGE_PROCESSOR_PATH; do
    [[ -d "${!path_var}" ]] || { echo "Missing directory: $path_var=${!path_var}" >&2; exit 1; }
done
for path_var in VAE_PATH TOML_PATH; do
    [[ -f "${!path_var}" ]] || { echo "Missing file: $path_var=${!path_var}" >&2; exit 1; }
done
cd "$REPO_ROOT"
mkdir -p "$OUTPUT_ROOT"
"${CMD[@]}" 2>&1 | tee -a "$OUTPUT_ROOT/launcher_rank${NODE_RANK:-0}.log"
