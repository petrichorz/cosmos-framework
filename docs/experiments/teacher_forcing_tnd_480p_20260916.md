# Teacher-forcing maskless TND experiment

## Scope and reproducibility

- Branch: `exp/teacher-forcing-tnd-20260916`, baseline commit `4af8f9d`.
- Separate worktree: `cosmos-framework-exp-tnd`; original worktree changes are preserved.
- User confirmed environment `cosmos-framework-py312` and dataset `/mnt/sfs_turbo/public/datasets/egosuite_demo_v1`.
- Runtime: Python 3.12.10, torch 2.10.0+cpu with torch_npu 2.10.0.post6.
- Cosmos3-Edge DCP, BF16, original 480p buckets, 30-second maximum windows, split policy with 2-second overlap, original packing limit 54000, seed 42.
- Four 910B3 devices, physical 4–7. Other users' jobs occupy 0–3; no processes from those jobs were stopped. Shared CPU/storage contention is a limitation.
- Full activation checkpointing, profiler off, 15 updates, first 5 excluded. Step time includes data loading, VAE, packing, forward, backward, optimizer and callbacks, with device synchronization.
- Report mean of the slowest rank per measured step, maximum allocated/reserved memory across measured steps and ranks, and aggregate noisy vision tokens per second. Reserved memory is allocator retention, not live allocation.
- Compare sample identity, frame ranges, shapes, original/expanded token counts, block size and history for every step. Do not compare speed if inputs differ.

## Implementation

`teacher_forcing_dense_mode="grouped_tnd"` keeps the joint per-layer clean/noisy network. Each `(sample, stream, temporal block)` becomes an independent full-attention group with exactly its legal keys:

- Clean block i: UND + clean blocks max(0, i-H) through i.
- Noisy block i: UND + clean blocks max(0, i-H) through i-1 + noisy block i.

`teacher_forcing_tnd.py` constructs one-dimensional indices once per batch, preserving GEN query order. It neither allocates token-pair masks nor changes spatial buckets. Existing UND causal attention is unchanged.

The NPU custom autograd function gathers at most one chunk's KV at a time. It saves original Q/K/V, the output and native FA softmax statistics. Backward gathers the same KV again and invokes the native `npu_fusion_attention_grad` without recomputing FA forward. KV gradients accumulate in FP32 and cast to the input dtype once. Clean KV is never detached. Only first-order differentiation is supported; dropout remains disabled as in the dense baseline.

The default chunk budget is 131072 expanded KV tokens, or approximately 0.5 GiB K+V for BF16, 8 KV heads and head dimension 128. An indivisible group may exceed this budget, up to the native TND limit. Runtime memory also includes original tensors, outputs, softmax statistics, gradient buffers and operator workspace; 0.5 GiB is not a peak-memory claim.

Native API constraints were inspected from installed `_op_plugin_docs.py` and `torch.ops.npu.npu_fusion_attention_grad.default._schema`. Forward uses TND, sparse mode 0, no mask, cumulative query/KV ends and keep_prob 1.0. Backward receives unchanged native softmax max/sum and attention output. Group count is capped at 1024 and individual KV groups at 1048576 tokens.

## Validation and initial failure

- CPU float64: 24 combinations of mixed sample lengths, singleton first blocks, partial last blocks, history 1/2/64, MHA/GQA, chunked/unchunked execution and activation checkpointing passed output and Q/K/V gradient checks.
- Existing packing/teacher-forcing tests: 97 passed.
- Added CPU integration tests: 2 passed, including network parameter and input gradients, and a guard forbidding dense-mask construction on the grouped path.
- Configuration tests: 34 passed.
- Initial naive NPU gather implementation failed the preselected `atol=rtol=0.008` check for 2 of 159744 elements in one tensor, max offending absolute difference 0.0234375. This failure is retained; thresholds were not widened. Shared KV gradient accumulation was changed to FP32 and gathered buffers were removed from the saved autograd state. The revised NPU path passed all 24 cases with the original tolerances. Maximum output absolute difference was 0.001953125 and maximum gradient absolute difference 0.03125 (the latter passes the relative tolerance at its magnitude). The small hardware check ended during baseline warmup step 4; it does not overlap measured steps 6–15. The actual long shape (96, 17, 23), UND 67, block size 1, history 15 also passed both ordinary and checkpoint execution: max output error 0.001953125, max Q/K/V gradient error 0.0078125. This shape has 75072 GEN queries and 75139 KV tokens.

## Results

Both four-rank runs exited with code 0 and completed all 15 updates. Source inputs and packing metadata matched for every rank and step. The 10 measured steps processed exactly 1578858 noisy vision tokens in each run.

| Metric                            | Per-sample dense | Grouped TND | Improvement                         |
| --------------------------------- | ---------------- | ----------- | ----------------------------------- |
| Mean step time, seconds           | 82.387611        | 35.336976   | 57.1089% less time; 2.3315x speedup |
| Noisy vision tokens/s, four ranks | 1916.377936      | 4468.005373 | 133.1484% higher                    |
| Peak allocated, GiB               | 48.196772        | 38.693160   | 9.503613 GiB / 19.7184% lower       |
| Peak reserved, GiB                | 59.904297        | 58.693359   | 1.210938 GiB / 2.0215% lower        |

Allocated memory measures live tensors/workspace. Reserved memory decreases less because the allocator retains cached blocks; the allocated reduction must not be interpreted as an equally sized drop in driver-visible occupancy.

This is one paired short training run on a shared server, not a long-term convergence test. Other jobs used physical cards 0–3 during the experiment, and had finished by the final device check; CPU/storage contention was not controlled. All experiment processes have exited and cards 4–7 are released.

The isolated long-shape attention benchmark uses the same geometry as the long precision check, 3 warmup iterations and 10 measured iterations, and includes KV gather/rematerialization, native backward and FP32 gradient accumulation. It does not include model projections, MLP, FSDP, VAE or optimizer. Metadata preparation is timed separately and is not included in the forward/backward totals.

| Variant     | Preparation s | Forward s | Backward s | Total s  | Peak allocated GiB | Peak reserved GiB |
| ----------- | ------------- | --------- | ---------- | -------- | ------------------ | ----------------- |
| Dense       | 3.403239      | 0.728310  | 1.318079   | 2.046389 | 9.237178           | 9.361328          |
| Grouped TND | 0.275273      | 0.080409  | 0.212341   | 0.292750 | 4.991440           | 6.003906          |

Measured attention speedup: 6.9902x, total time reduction 85.6943%. Peak allocated reduction: 4.245738 GiB. These are isolated attention results, not whole-model gains.

Across the 40 measured rank-step layouts, exact legal token pairs are 20.6719% of the dense rectangles. This is a theoretical pair-count statistic, not a measured FLOP or MFU metric.

Raw artifacts are under `../cosmos-profile-logs/20260916_tnd_validation/` and the paired `20260916_tnd_dense_4npu` / `20260916_tnd_grouped_4npu` directories.

## Reproduction

Run from the experiment worktree on four idle cards; the script refuses occupied devices or existing output directories. It sources the confirmed conda environment without modifying its installation.

```bash
PAIR_ROOT=/mnt/sfs_turbo/zheng/cosmos-ascend-profile/cosmos-profile-logs/tnd_reproduce_NEW \
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 bash tools/run_teacher_forcing_tnd_pair.sh
```

To select the implementation in a recipe, set `[model] teacher_forcing_dense_mode = "grouped_tnd"`. The default remains `global`, and `per_sample` remains the dense comparison path. The launcher explicitly overrides the recipe's duration/resolution/iteration settings to 30 seconds, 480p and 15 updates.
