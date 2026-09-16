#!/usr/bin/env bash
# Run on idle physical cards. Both variants use identical original 480p buckets.
set -eo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/tools/ascend_experiment_env.sh"
set -u
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-4,5,6,7}"
export NPROC=4
export MASTER_PORT="${MASTER_PORT:-50531}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-64000}"
PAIR_ROOT="${PAIR_ROOT:?Specify a fresh output prefix}"
for mode in per_sample grouped_tnd; do
    if [[ -e "${PAIR_ROOT}_${mode}" ]]; then
        echo "Output already exists: ${PAIR_ROOT}_${mode}" >&2
        exit 1
    fi
done
python - <<'PY'
import os,re,subprocess
info=subprocess.check_output(['npu-smi','info'],text=True)
busy={int(m[0]) for m in re.findall(r'\|\s*(\d+)\s+\d+\s+\|\s*(\d+)\s*\|',info)}
cards=set(map(int,os.environ['ASCEND_RT_VISIBLE_DEVICES'].split(',')))
if busy & cards:
    raise SystemExit(f'Refusing occupied cards: {sorted(busy & cards)}')
PY
mkdir -p "${PAIR_ROOT}_validation"
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES%%,*}" python tools/check_teacher_forcing_tnd.py --device npu > "${PAIR_ROOT}_validation/small.log" 2>&1
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES%%,*}" python tools/check_teacher_forcing_tnd.py --device npu --long > "${PAIR_ROOT}_validation/long.log" 2>&1
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES%%,*}" python tools/benchmark_teacher_forcing_tnd_kernel.py --output "${PAIR_ROOT}_validation/kernel.json" > "${PAIR_ROOT}_validation/kernel.log" 2>&1
for mode in per_sample grouped_tnd; do
    BENCHMARK_RUN_DIR="${PAIR_ROOT}_${mode}" bash tools/benchmark_480p_ascend.sh \
        model.config.teacher_forcing_dense_mode="$mode"
done
python tools/summarize_ascend_benchmarks.py "${PAIR_ROOT}_per_sample" "${PAIR_ROOT}_grouped_tnd" > "${PAIR_ROOT}_summary.json"
