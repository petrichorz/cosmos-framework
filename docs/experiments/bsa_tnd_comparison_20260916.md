# BSA64 versus grouped TND: matched-input comparison

## Protocol

- Comparison branch: `exp/tnd-bsa-comparison-20260916`, based on TND commit `94db9f2`.
- Same four physical 910B3 devices (4–7), sequential BSA64 then TND runs, Python environment `cosmos-framework-py312`, dataset `egosuite_demo_v1`, Cosmos3-Edge BF16.
- Both use original BSA experiment bucket overrides (`COSMOS_BSA64_BUCKETS=1`), 480p, 30-second maximum split windows, 2-second overlap, seed 42, full activation checkpointing, 8 workers/rank and no profiler.
- Each run has 15 updates, with updates 6–15 measured. Compare source sample IDs/frame ranges/shapes and all packing/geometry metadata.
- Iteration timing includes data loading, VAE, forward/backward and optimizer update, synchronized at the boundaries; aggregate the slowest rank per update. Memory is the maximum reported per-device peak across measured updates and ranks.
- Only GEN attention backend differs. The BSA64 adapter is copied from the existing experiment; its library comes from the isolated kernel directory without installing into the shared environment.
- Kernel source commit: `dfb00fed829e652cc1abf92a0ab03eca39e8eed0`. Library SHA256: `8f9fc5d8420a751f0fc50a5b3537e1a3aabc94f8ecd7797b465b84fc2bd7dff3`.
- Historical TND 35.34 s/step and BSA64 36.09 s/step used different buckets and cannot establish a backend speedup. Fresh matched runs below replace that comparison.

## Precision and isolated attention

- Existing BSA64 small checks: 24 cases passed, including ragged UND tails and MHA/GQA, against the dense oracle.
- Direct BSA64/TND output and Q/K/V gradient checks: four long geometries passed `atol=rtol=0.008`.
- All kernel comparisons use the same tensors: visual shape `(96,16,24)`, UND=67, Hq/Hkv=24/8, D=128, BF16. Three warmup and ten measured calls per backend; order alternates between geometries.
- Kernel timing includes reorder/gather, native backward and gradient accumulation. Metadata preparation is separate. Kernel tests ran on physical card 0 and finished during whole-model BSA warmup, before measured updates.

| Block frames | History blocks | BSA64 F+B s | TND F+B s | TND speedup | BSA allocated GiB | TND allocated GiB |
| ------------ | -------------- | ----------- | --------- | ----------- | ----------------- | ----------------- |
| 1            | 1              | 0.119041    | 0.039898  | 2.9836x     | 5.3748            | 6.0679            |
| 1            | 15             | 0.303951    | 0.253416  | 1.1994x     | 5.3748            | 4.9026            |
| 1            | 64             | 0.732753    | 0.707997  | 1.0350x     | 5.3748            | 4.8780            |
| 4            | 22             | 0.837753    | 0.507511  | 1.6507x     | 5.3748            | 5.1054            |

For block=1/history=64, measured forward is slower with TND (0.1884 versus 0.1545 s), while backward is faster (0.5196 versus 0.5783 s). With full checkpointing, the simple `2F+B` attention estimate is 0.8964 s for TND and 0.8872 s for BSA64, effectively close. This estimate is not an independently measured checkpoint iteration.

TND metadata preparation increases with duplicated history indices (0.7997 s for history=64 in this test versus BSA64 0.0314 s). Metadata is prepared once per batch, so this cost must not be multiplied by the number of transformer layers.

Numerical checks cover attention outputs and first-order gradients; these short training runs do not establish long-run convergence equivalence.

These measurements do not imply universal memory superiority: for history=1, the current TND chunk plan has a higher isolated allocated peak than BSA64. Whole-model memory is measured independently below.

## Whole-model results

Both launchers completed successfully. All four ranks have updates 6–15, and all 15 updates have identical source sample/frame metadata and packing metadata. Both measured intervals process 1,570,944 noisy vision tokens.

| Metric                    | BSA64         | Grouped TND   | TND change       |
| ------------------------- | ------------- | ------------- | ---------------- |
| Mean rank-max iteration   | 36.103076 s   | 32.103487 s   | 11.08% less time |
| Noisy vision tokens/s     | 4351.27       | 4893.37       | +12.46%          |
| Peak allocated per device | 38.447774 GiB | 38.541619 GiB | +0.0938 GiB      |
| Peak reserved per device  | 56.675781 GiB | 57.884766 GiB | +1.2090 GiB      |

The observed whole-model speedup is 1.1246x. TND is faster in 8/10 paired updates, but slower in updates 7 and 11. This is one sequential pair of runs, not a repeated-run confidence interval. Per-step differences also include data-loading/runtime variation and cannot be attributed solely to attention geometry.

| Update | BSA64 seconds | TND seconds |
| ------ | ------------- | ----------- |
| 6      | 38.472230     | 31.668104   |
| 7      | 33.926160     | 34.982394   |
| 8      | 33.088905     | 26.979687   |
| 9      | 37.606807     | 33.501971   |
| 10     | 37.300727     | 34.156766   |
| 11     | 33.543798     | 36.144160   |
| 12     | 35.900312     | 29.611590   |
| 13     | 38.646411     | 31.403575   |
| 14     | 38.052092     | 34.947398   |
| 15     | 34.493314     | 27.639226   |

TND does not reduce whole-model memory versus BSA64 in this comparison: allocated increases by about 0.24%, and reserved by about 1.21 GiB. Its benefit here is throughput and support for non-64-aligned layouts without a custom BSA library. The earlier dense-to-TND memory saving must not be reused as a BSA-to-TND claim.

For this workload, grouped TND is the preferred general backend based on measured throughput and broader shape support. Retain BSA64 as an optional comparison backend; any history-dependent hybrid dispatch needs broader measurements before implementation.

Structured results, including all isolated timings and precision errors: [bsa_tnd_comparison_20260916.json](./bsa_tnd_comparison_20260916.json). Raw rank logs live under `../cosmos-profile-logs/20260916_compare_bsa_tnd_4npu_{bsa64,tnd}` relative to this worktree.

## Reproduction

```bash
PAIR_ROOT=/mnt/sfs_turbo/zheng/cosmos-ascend-profile/cosmos-profile-logs/bsa_tnd_NEW \
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 bash tools/compare_bsa_tnd_480p.sh
```

Run from this comparison worktree. BSA64 requires exact post-patch token boundaries aligned to 64; the adapter rejects unsupported layouts. This dataset/run exercises compatible buckets and does not validate every advertised resolution/aspect ratio.

The isolated attention benchmark can be reproduced on an idle card with the same environment:

```bash
source tools/ascend_experiment_env.sh
export PYTHONPATH="$PWD:$PWD/../bsa64-isolated-kernel${PYTHONPATH:+:$PYTHONPATH}"
ASCEND_RT_VISIBLE_DEVICES=0 python tools/benchmark_bsa_tnd_kernels.py --output /tmp/bsa_tnd_kernels.json
```
