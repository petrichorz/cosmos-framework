#!/usr/bin/env bash
set -eo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source tools/ascend_experiment_env.sh
set -u
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-4,5,6,7}"
export NPROC=4 AC_MODE=full COSMOS_BSA64_BUCKETS=1 COSMOS_GEN_BSA64=0
export COSMOS_ASCEND_BENCHMARK_WARMUP=3
export MASTER_PORT="${MASTER_PORT:-50551}" HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-64000}"
RUN_PREFIX="${RUN_PREFIX:?Specify a fresh output prefix}"
python - <<'PY'
import os, re, subprocess
info = subprocess.check_output(['npu-smi', 'info'], text=True)
busy = {int(m[0]) for m in re.findall(r'\|\s*(\d+)\s+\d+\s*\|\s*(\d+)\s*\|', info)}
cards = set(map(int, os.environ['ASCEND_RT_VISIBLE_DEVICES'].split(',')))
assert len(cards) == 4
assert not busy & cards, f'Occupied cards: {sorted(busy & cards)}'
PY
for history in 1 64; do
    test ! -e "${RUN_PREFIX}_h${history}"
done
for history in 1 64; do
    BENCHMARK_RUN_DIR="${RUN_PREFIX}_h${history}" bash tools/benchmark_480p_ascend.sh \
        trainer.max_iter=8 model.config.teacher_forcing_dense_mode=grouped_tnd \
        model.config.teacher_forcing_block_size_min=1 model.config.teacher_forcing_block_size_max=1 \
        model.config.teacher_forcing_history_blocks_min="$history" \
        model.config.teacher_forcing_history_blocks_max="$history"
done
