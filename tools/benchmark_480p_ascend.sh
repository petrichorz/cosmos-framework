#!/usr/bin/env bash
set -eo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/tools/ascend_experiment_env.sh"
set -u
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
NPROC="${NPROC:-4}"
AC_MODE="${AC_MODE:-full}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
export DATASET_PATH=/mnt/sfs_turbo/public/datasets/egosuite_demo_v1
export BASE_CHECKPOINT_PATH=/mnt/sfs_turbo/public/ckpts/Cosmos/Cosmos3-Edge-DCP
export COSMOS3_EDGE_PROCESSOR_PATH=/mnt/sfs_turbo/public/ckpts/Cosmos/Cosmos3-Edge
export WAN_VAE_PATH=/mnt/sfs_turbo/public/ckpts/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
export COSMOS_PERF_SKIP_FINAL_CHECKPOINT=1 COSMOS_ASCEND_BENCHMARK=1
export IMAGINAIRE_OUTPUT_ROOT="${BENCHMARK_RUN_DIR:?Specify a fresh BENCHMARK_RUN_DIR}"
mkdir -p "$IMAGINAIRE_OUTPUT_ROOT"
exec > >(tee -a "$IMAGINAIRE_OUTPUT_ROOT/launcher.log") 2>&1
cd "$REPO_ROOT"
python -c 'import sys,torch,torch_npu,cosmos_framework; print(sys.version,torch.__version__,torch_npu.__version__,cosmos_framework.__file__)'
torchrun --nproc_per_node="$NPROC" --master_port="${MASTER_PORT:-50410}" \
  -m cosmos_framework.scripts.train \
  --sft-toml="$REPO_ROOT/examples/toml/sft_config/vision_edge_egosuite_ascend_profile.toml" -- \
  model=mot_causal_fsdp \
  model.config.vlm_config.tokenizer.repository=null \
  model.config.vlm_config.tokenizer.revision=null \
  +model.config.vlm_config.tokenizer.tokenizer_type="$COSMOS3_EDGE_PROCESSOR_PATH" \
  '~dataloader_train.dataloader.datasets.video.dataset.conditioning_config={0:0.7,1:0.2,2:0.1}' \
  '+dataloader_train.dataloader.datasets.video.dataset.conditioning_config={0:1.0}' \
  trainer.seed=42 trainer.max_iter=15 trainer.profiling.enable_profiling=false \
  trainer.run_validation=false \
  model.config.activation_checkpointing.mode="$AC_MODE" \
  'model.config.activation_checkpointing.save_ops_regex=["fusion_attention"]' \
  dataloader_train.dataloader.num_workers=8 \
  dataloader_train.dataloader.datasets.video.dataset.max_video_duration_s=30.0 \
  'dataloader_train.dataloader.datasets.video.dataset.resolution="480"' \
  "$@"
