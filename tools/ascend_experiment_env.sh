#!/usr/bin/env bash
# Source this file; it does not alter the shared environment installation.
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /mnt/sfs_turbo/public/apps/miniforge3/etc/profile.d/conda.sh
conda activate cosmos-framework-py312
stale_torch_lib="${CONDA_PREFIX}/lib/python3.13/site-packages/torch/lib"
LD_LIBRARY_PATH=":${LD_LIBRARY_PATH:-}:"
LD_LIBRARY_PATH="${LD_LIBRARY_PATH//:${stale_torch_lib}:/:}"
LD_LIBRARY_PATH="${LD_LIBRARY_PATH#:}"
LD_LIBRARY_PATH="${LD_LIBRARY_PATH%:}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH}"
export COSMOS_DEVICE=npu HF_HUB_OFFLINE=1
