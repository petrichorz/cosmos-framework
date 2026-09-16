# TND teacher-forcing hyperparameters

## Geometry and semantics

Let T be the number of latent frames, S the number of spatial tokens per latent frame after patchification, U the UND token count, B the block size in latent frames, and H the number of preceding clean blocks visible to a query. The first latent frame is a singleton block; remaining frames are partitioned into blocks of at most B frames.

- Temporal block count: G = 1 + ceil((T - 1) / B).
- Total GEN queries: 2TS, covering clean and noisy streams. Total original KV: U + 2TS. Changing B/H does not change these lengths for fixed input.
- For a block containing s_i tokens, query length is s_i and its allowed KV length is k_i = U + s_i + sum(s_j for j in the preceding H blocks).
- Clean queries see UND, preceding clean blocks and their current clean block. Noisy queries see UND, preceding clean blocks and their current noisy block. Each group performs a single softmax across its whole legal KV set.
- Expanded KV token count across both streams is 2 *sum(k_i); query-key pair count is 2* sum(s_i * k_i). These distinguish gather volume from attention arithmetic.
- Away from sequence boundaries, the historical span is approximately BH latent frames. H is not seconds; conversion requires temporal compression and the actual sampled FPS.

B/H change attention semantics. The TND chunk budget changes execution grouping only. Do not treat these as interchangeable performance knobs.

## Measured history and block-size effects

Same tensors, T=96, S=16*24=384, U=67, BF16, 24 query heads, 8 KV heads, head dimension 128, chunk budget 131072 expanded KV tokens. Each geometry has a fresh process, two warmup and five measured iterations. Peaks include inputs, output gradient, plan, outputs and returned gradients. These are isolated attention forward/backward numbers, not whole-model allocation.

| H   | B=1 peak GiB | B=1 F+B seconds | B=4 peak GiB | B=4 F+B seconds |
| --- | ------------ | --------------- | ------------ | --------------- |
| 1   | 6.0679       | 0.0405          | 6.2531       | 0.0884          |
| 2   | 5.6533       | 0.0595          | 5.7587       | 0.1211          |
| 4   | 5.2615       | 0.0884          | 5.3237       | 0.1902          |
| 8   | 5.0587       | 0.1453          | 5.1876       | 0.3057          |
| 16  | 4.8696       | 0.2714          | 5.1057       | 0.4609          |
| 32  | 4.8645       | 0.4685          | 5.1067       | 0.5120          |
| 64  | 4.8785       | 0.7098          | 5.1067       | 0.5117          |
| 96  | 4.8817       | 0.7862          | 5.1067       | 0.5125          |

At equal H below saturation, B=4 spans roughly four times as many historical frames as B=1, so it is not an equal-context speed comparison. Larger blocks also permit more within-block attention. Fewer groups alone does not guarantee lower runtime.

For approximately equal historical spans, B=1/H=64 versus B=4/H=16 takes 0.7098 versus 0.4609 seconds (about 1.54x attention speedup), with allocated peaks 4.8785 versus 5.1057 GiB. This is a useful engineering comparison, but their attention visibility is not identical: block boundaries and within-block visibility differ. No training-quality equivalence has been established.

For T=96/B=4 there are 25 temporal blocks, so H>=24 already exposes all preceding clean blocks. H=32/64/96 therefore generate identical plans. If H is sampled uniformly from 1 through 64 at this fixed geometry, 41/64 (64.1%) of draws already cover the entire available history. The threshold changes with T; H=32 is not globally sufficient for every 30-second clip.

## Memory and computation

For the tested fixed lengths, allocation initially falls and then plateaus as H increases. The implementation targets at most 131072 expanded KV tokens per chunk, so longer histories place fewer query groups in each chunk. This reduces query-side temporaries. Backward regathers KV one chunk at a time instead of saving all expanded KV activations. Original Q/K/V, output and FP32 shared-gradient accumulators still occupy memory; index storage grows with expanded history.

The chunk target is soft for an indivisible group: a single group can exceed it. Groups cannot be split arbitrarily across softmax without changing the implementation. The current code allows at most 1048576 KV tokens in one group and 1024 groups per chunk.

Whole-model confirmation used the actual dataset, four cards, 480p, 30-second maximum splits, fixed B=1 and full checkpointing. H=1 versus H=64 measured 37.440 versus 37.497 GiB peak allocated per device (+59 MiB, 0.15%), while update time increased from 14.49 to 35.34 seconds (2.44x). Whole-model intermediate histories and fixed B=4 were not measured. Do not transfer isolated attention speedups or memory differences directly to the whole model.

## Other parameters

| Parameter                               | Meaning and expected impact                                                                                                                                                                                                                         | Evidence status                                                                    |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| max_kv_tokens                           | Expanded KV budget per TND call; smaller budgets usually reduce individual temporary tensors but increase calls, while larger budgets may improve batching and raise workspace. Actual peaks/speed may be nonmonotonic. Does not change visibility. | Fixed at 131072; no budget sweep yet.                                              |
| Latent length T                         | Adds original Q/K/V, output, indices and work. Limited fixed H gives approximately linear pair-count growth away from boundaries; full-history attention grows approximately quadratically in T.                                                    | Fixed T in controlled sweep; whole-model samples have varying T.                   |
| Spatial tokens S                        | Set by resolution, compression and patchification. Original activations scale with S; pair count scales roughly with S squared at fixed temporal geometry.                                                                                          | Fixed S=384 in this sweep.                                                         |
| UND count U                             | Included in every group's KV, increasing gather traffic, indices and attention work.                                                                                                                                                                | Fixed U=67.                                                                        |
| Query heads / KV heads / head dimension | Query heads and dimension affect attention compute; KV heads and dimension affect gathered KV storage. Original activations and gradient buffers also change.                                                                                       | Fixed 24/8/128; no sweep.                                                          |
| Packing limit / samples per device      | Controls input token volume and number of independently grouped samples. Per-sample isolation is retained. Raw packing tokens, doubled teacher-forcing tokens and expanded chunk KV are different quantities.                                       | Dataset packer max_sequence_length=54000; not a TND chunk limit.                   |
| Activation checkpointing                | Trades saved activations for recomputation. The attention-only wrapper can have a higher peak in isolation, which does not predict whole-model checkpoint savings.                                                                                  | Tested attention with/without checkpoint; whole-model runs use full checkpointing. |
| BF16 and FP32 accumulation              | Q/K/V use BF16; shared KV gradient accumulation uses FP32 to maintain numerical agreement. Removing FP32 accumulation is not a validated memory optimization.                                                                                       | Current implementation and previous precision tests.                               |

## Configuration and interpretation

To fix B=4/H=16, set both endpoints of each sampling range:

```text
model.config.teacher_forcing_dense_mode=grouped_tnd
model.config.teacher_forcing_block_size_min=4
model.config.teacher_forcing_block_size_max=4
model.config.teacher_forcing_history_blocks_min=16
model.config.teacher_forcing_history_blocks_max=16
```

Setting only a maximum retains randomized geometry. The existing 1–4 block / 1–64 history ranges independently sample per sample, so they mix short-history, long-history and fully saturated cases. Their aggregate performance cannot be labeled a fixed B=4 or H=64 result.

Choose B/H according to required temporal visibility and training/inference behavior first. If the intended history is about 64 latent frames with B=4, H=16 is the corresponding candidate; H=64 requests about 256 latent frames, clipped by available context. This is a semantic mapping, not a claim that either setting preserves model quality. Keep the currently tested chunk target as the baseline; a chunk-budget sweep is the next implementation-only tuning experiment because it preserves attention visibility.

## Sources

- [History experiment, raw data and curves](./tnd_history_memory_20260916.md)
- [TND plan and rematerialized backward](../../cosmos_framework/model/generator/mot/teacher_forcing_tnd.py)
- [Geometry sampling and singleton first block](../../cosmos_framework/data/generator/sequence_packing/teacher_forcing.py)
- [Dataset packing configuration](../../examples/toml/sft_config/vision_edge_egosuite_ascend_profile.toml)
