# Cosmos3 DROID Ascend four-stage performance analysis

Dataset: `/data5T/Embodied-AI/datasets/Cosmos3-DROID/success`

## Result locations

- Stage 1 data pipeline: `/data5T/zheng/cosmos-ascend-profile/cosmos-profile-logs/ascend_profile_20260907_162640/data`
- Stage 2 baseline: `/data5T/zheng/cosmos-ascend-profile/cosmos-profile-logs/ascend_profile_20260908_000428/baseline`
- Stage 3 rank-0 operator profile: `/data5T/zheng/cosmos-ascend-profile/cosmos-profile-logs/ascend_profile_20260908_013341/npu`
- Stage 4 all-rank distributed profile: `/data5T/zheng/cosmos-ascend-profile/cosmos-profile-logs/ascend_profile_20260908_014250/distributed`

## Stage 1: data pipeline

- 64/64 samples succeeded; 21,906 frames were decoded.
- Wall throughput was 2.083 samples/s and 712.979 frames/s with 8 workers.
- Per-sample video decode mean/P50/P90 was 2751/2522/4829 ms.
- Decoder initialization mean was 295 ms. There were 52 misses and 12 hits in this short cold-cache run.

Video decode is expensive in worker CPU time, but the baseline training run shows that worker prefetch hides it from the accelerator critical path.

## Stage 2: non-profiler baseline

Across ranks and after excluding the first two iterations:

| Scope | Mean ms | Approx. share of training step |
| :--- | ---: | ---: |
| Forward | 16,973 | 46.8% |
| Backward | 18,024 | 49.7% |
| Optimizer | 1,266 | 3.5% |
| Training step | 36,273 | 100% |
| Data wait | 6.0 | 0.02% |
| Host-to-device | 6.5 | 0.02% |

The steady iteration mean was 36.319 s (P50 36.038 s, range 33.271--40.632 s). The data pipeline is not the current throughput limiter: mean data wait is only about 6 ms despite mean worker-side TorchCodec decode time of about 1.98 s. The baseline decoder cache recorded 906 hits and 672 misses, a 57.4% hit rate.

## Stage 3: rank-0 operator profile

The successful retry used Ascend Level 0, shape recording, and disabled Python stack/module capture. The active step was about 45.4 s, approximately 25% slower than the non-profiled steady mean. Profiler timings should therefore be used for composition and attribution, not as the production step-time baseline.

Largest rank-0 kernels in the captured step:

| Kernel family | Total ms |
| :--- | ---: |
| FlashAttention forward | 9,054 |
| FlashAttention backward | 7,032 |
| MatMul | 4,220 |
| Conv3D | 3,886 |
| Mul | 1,982 |
| AllGather (largest instance) | 1,976 |
| Add | 1,537 |

The first attempt with `with_stack=true` and `with_modules=true` produced truncated raw data. Disabling those two high-overhead options fixed the failure.

## Stage 4: all-rank distributed profile

All 8 ranks produced `analyse.done`, `trace_view.json`, `operator_details.csv`, `kernel_details.csv`, and `step_trace_time.csv`. Captured stage times span only 45.536--45.568 s across ranks because collectives synchronize the step.

Mean Ascend step decomposition:

| Component | Mean ms | Stage share |
| :--- | ---: | ---: |
| Computing | 31,539 | 69.23% |
| Communication, total | 24,977 | 54.83% |
| Communication overlapped with compute | 16,562 | 36.36% |
| Communication not overlapped | 8,415 | 18.47% |
| Free | 5,599 | 12.29% |
| Preparing | 62 | 0.14% |

About 66.3% of communication is hidden by compute, but the remaining 8.4 s mean non-overlapped communication is a material bottleneck.

The workload is imbalanced even though wall times look balanced. Rank 0 spends 35.44 s computing and only 1.76 s in non-overlapped communication; rank 5 spends 29.27 s computing and 12.44 s in non-overlapped communication. FlashAttention kernel time ranges from 8.97 s on rank 5 to 16.18 s on rank 0. MatMul and Conv3D times are comparatively stable. This points to variable packed-sequence shapes/attention FLOPs, with lighter ranks waiting for the heaviest attention workload at FSDP collectives.

Across all ranks, FlashAttention forward/backward averages 11.60 s/rank (36.8% of mean compute time), MatMul 4.32 s/rank (13.7%), and Conv3D 4.03 s/rank (12.8%). Together these three families are about 63.3% of mean compute time. The trace also attributes large waits to FSDP `all_gather` and backward `reduce_scatter` operations.

## Optimization priority

1. Balance batches by estimated attention FLOPs (sequence-length-squared contribution), not just sample count or raw token count. Bucket similar video/token lengths and distribute heavy packs across ranks.
2. Reduce exposed FSDP communication by inspecting layer-level prefetch/wrap policy while preserving overlap. Track maximum per-rank compute and non-overlapped communication, not only averages.
3. Optimize the attention workload before low-level pointwise kernels. Since attention is already fused, sequence length, packing, activation-checkpoint recomputation, and attention layout are the higher-leverage knobs.
4. Investigate Conv3D and TransData layout conversions in the video VAE path after attention and communication.
5. Keep 8 data workers for now. More workers are unlikely to improve throughput while data wait is about 6 ms. A larger decoder cache can be A/B tested for hit rate and RSS/file-descriptor cost, but it is not on the current critical path.

## Interpretation notes

- Worker-side decode durations overlap across processes and must not be summed against iteration wall time.
- Operator totals are nested and may include asynchronous device work. Use `step_trace_time.csv` for stage decomposition and Stage 2 events for production wall time.
- Profiler active-step overhead is about 25%, so do not use the 45.5 s profiled step as normal training throughput.
