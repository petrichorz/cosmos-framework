#!/usr/bin/env bash
set -eo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/tools/ascend_experiment_env.sh"
set -u
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/../bsa64-isolated-kernel${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-4,5,6,7}"
export NPROC=4 COSMOS_BSA64_BUCKETS=1 AC_MODE=full COSMOS_ASCEND_BENCHMARK_WARMUP=5
export MASTER_PORT="${MASTER_PORT:-50541}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-64000}"
PAIR_ROOT="${PAIR_ROOT:?Specify a fresh output prefix}"
for mode in bsa64 tnd; do
    [[ ! -e "${PAIR_ROOT}_${mode}" ]] || { echo 'Refusing existing output directory' >&2; exit 1; }
done
python - <<'PY'
import os,re,subprocess
info=subprocess.check_output(['npu-smi','info'],text=True)
busy={int(m[0]) for m in re.findall(r'\|\s*(\d+)\s+\d+\s*\|\s*(\d+)\s*\|',info)}
cards=set(map(int,os.environ['ASCEND_RT_VISIBLE_DEVICES'].split(',')))
assert len(cards)==4
if busy & cards: raise SystemExit(f'Occupied cards: {sorted(busy & cards)}')
PY
for mode in bsa64 tnd; do
    if [[ "$mode" == bsa64 ]]; then
        export COSMOS_GEN_BSA64=1
        attention_mode=per_sample
    else
        export COSMOS_GEN_BSA64=0
        attention_mode=grouped_tnd
    fi
    BENCHMARK_RUN_DIR="${PAIR_ROOT}_${mode}" bash tools/benchmark_480p_ascend.sh \
        model.config.teacher_forcing_dense_mode="$attention_mode"
done
python tools/summarize_ascend_benchmarks.py "${PAIR_ROOT}_bsa64" "${PAIR_ROOT}_tnd" > "${PAIR_ROOT}_summary.json"
python - "${PAIR_ROOT}_summary.json" <<'PY_CHECK'
import json, sys
with open(sys.argv[1]) as handle:
    result = json.load(handle)
assert all(r["ranks"] == 4 and r["measured_steps"] == list(range(6, 16)) for r in result["runs"])
assert len(result["comparison"]) == 1 and result["comparison"][0]["input_equal"]
PY_CHECK
