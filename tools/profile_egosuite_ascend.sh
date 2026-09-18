#!/usr/bin/env bash
# Four-rank Ascend profiling with Python stacks; run from any directory.
set -eo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /mnt/sfs_turbo/public/apps/miniforge3/etc/profile.d/conda.sh
conda activate cosmos-framework-py312
set -u
# The cloned environment's activation hook still points at Python 3.13 Torch.
stale_torch_lib="${CONDA_PREFIX}/lib/python3.13/site-packages/torch/lib"
LD_LIBRARY_PATH=":${LD_LIBRARY_PATH:-}:"
LD_LIBRARY_PATH="${LD_LIBRARY_PATH//:${stale_torch_lib}:/:}"
LD_LIBRARY_PATH="${LD_LIBRARY_PATH#:}"
LD_LIBRARY_PATH="${LD_LIBRARY_PATH%:}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export COSMOS_DEVICE=npu
export HF_HUB_OFFLINE=1
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
export DATASET_PATH="${DATASET_PATH:-/mnt/sfs_turbo/public/datasets/egosuite_demo_v1}"
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-/mnt/sfs_turbo/public/ckpts/Cosmos/Cosmos3-Edge-DCP}"
export COSMOS3_EDGE_PROCESSOR_PATH="${COSMOS3_EDGE_PROCESSOR_PATH:-/mnt/sfs_turbo/public/ckpts/Cosmos/Cosmos3-Edge}"
export WAN_VAE_PATH="${WAN_VAE_PATH:-/mnt/sfs_turbo/public/ckpts/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
export COSMOS_NPU_PROFILE_ACTIVE_STEPS=2
export IMAGINAIRE_OUTPUT_ROOT="${PROFILE_RUN_DIR:-/mnt/sfs_turbo/zheng/cosmos-ascend-profile/cosmos-profile-logs/egosuite_4npu_stack_$(date -u +%Y%m%d_%H%M%S)}"
mkdir -p "$IMAGINAIRE_OUTPUT_ROOT"
exec > >(tee -a "$IMAGINAIRE_OUTPUT_ROOT/launcher.log") 2>&1
cd "$REPO_ROOT"
printf 'Repository: %s\nProfile output: %s\n' "$REPO_ROOT" "$IMAGINAIRE_OUTPUT_ROOT"
python -c 'import sys, torch, torch_npu, cosmos_framework; print("Runtime:", sys.version, torch.__version__, torch_npu.__version__, cosmos_framework.__file__)'
torchrun --nproc_per_node=4 --master_port="${MASTER_PORT:-50229}" \
  -m cosmos_framework.scripts.train \
  --sft-toml="$REPO_ROOT/examples/toml/sft_config/vision_edge_egosuite_ascend_profile.toml" -- \
  model=mot_causal_fsdp \
  model.config.vlm_config.tokenizer.repository=null \
  model.config.vlm_config.tokenizer.revision=null \
  +model.config.vlm_config.tokenizer.tokenizer_type="$COSMOS3_EDGE_PROCESSOR_PATH" \
  '~dataloader_train.dataloader.datasets.video.dataset.conditioning_config={0:0.7,1:0.2,2:0.1}' \
  '+dataloader_train.dataloader.datasets.video.dataset.conditioning_config={0:1.0}' \
  trainer.max_iter=9 trainer.profiling.enable_profiling=true \
  trainer.profiling.profile_freq=8 trainer.profiling.profile_warmup=2 \
  'trainer.profiling.target_ranks=[0,1,2,3]' \
  trainer.profiling.record_shape=true trainer.profiling.profile_memory=true \
  trainer.profiling.with_stack=true trainer.profiling.with_modules=true \
  dataloader_train.dataloader.num_workers=8 \
  'dataloader_train.dataloader.datasets.video.dataset.resolution="256"' \
  "$@"
