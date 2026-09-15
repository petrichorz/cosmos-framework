# Cosmos3 Vision Edge Ascend 性能实验报告

本文汇总 Vision Edge 在 Ascend NPU 上已经完成的性能优化。每项实验只回答四个问题：发现了什么问题、问题为什么发生、代码如何修改、端到端收益如何。

## 1. 实验口径

- 数据集：`/mnt/sfs_turbo/public/datasets/egosuite_demo_v1`。
- 融合算子矩阵使用 8 张 Ascend 910B3；GQA、视频时长与切片实验使用 4 张 NPU。
- A/B 固定随机种子、packing 上限、并行拓扑、视频后端、worker 数和统计区间。
- 吞吐结论来自无 Profiler 稳态 step；Level1 + PipeUtilization Profile 只用于算子与通信归因。
- 多卡 step 取各 rank 的最大 `iteration_core`，避免平均值掩盖慢 rank。

## 2. 结论

| 优化点                    | 吞吐结果                          | 显存结果                 | 结论                            |
| :------------------------ | :-------------------------------- | :----------------------- | :------------------------------ |
| 消除 Host 同步            | 四卡验证最快下降 20.6%            | 无明显变化               | 保留                            |
| NPU RoPE                  | 相对同步优化基线提升 1.62%        | 无稳定收益               | 保留                            |
| NPU RMSNorm               | 相对同步优化基线提升 3.64%        | allocated 减少 3.909 GiB | 保留                            |
| RoPE + RMSNorm            | 相对原始路径提升 6.57%            | allocated 减少 3.399 GiB | 当前吞吐最优组合                |
| Teacher-forcing 原生 GQA  | 四卡复测回退 2.97%                | allocated 减少 3.227 GiB | 当前分支以吞吐换显存            |
| 超长 Episode 均衡重叠切片 | 四卡 Training step P50 下降 26.1% | 降低长序列峰值压力       | 使用 `split + 61s + 5s overlap` |

## 3. 消除热路径中的 Host 同步

### 问题

Profile 中 `aten::to`、`POST_NOISE_PACK`、timestep embedding 和 FSDP AllGather 出现秒级停顿；同时存在 26 万次以上的 `aten::item`。

### 原因

NPU 算子默认异步执行。packing 中的 `int(Tensor)`、Attention 每层的 `.tolist()`、以及先在 CPU 创建再搬到 NPU 的 timestep frequency，会迫使 Host 等待设备。Profiler 将前序积累的等待记到后续 `aten::to` 或 collective 上，因此看起来像拷贝或 AllGather 本身异常慢。

### 修改

- 将 teacher-forcing packing 的 Python dict/list remap 改为 Tensor lookup/indexing。
- 构造 packing offsets 时同步保存 Host 侧 actual sequence lengths，Attention 直接复用，不再逐层从 NPU Tensor 调用 `.tolist()`。
- timestep frequency 直接在目标设备创建，去掉 CPU→NPU 搬运。

### 收益

| 指标                     |     修改前 |    修改后 |     变化 |
| :----------------------- | ---------: | --------: | -------: |
| CPU packing 微基准       |  661.538ms |   7.322ms |    90.3× |
| 四卡 step wall time      |    45.272s |   35.944s |   -20.6% |
| `POST_NOISE_PACK` rank 1 | 9793.775ms | 290.666ms |   -97.0% |
| root AllGather 最大事件  | 6744.667ms | 562.567ms |   -91.7% |
| `EMBED_CLEAN_TIMESTEPS`  | 6322.790ms |   1.272ms | 近乎消除 |

`aten::item` 从 266,130 次降至 2,630 次，`aten::select` 从 212,912 次降至 2,634 次。另一组 EgoSuite 八卡矩阵中，同步优化单独带来 1.24% 提升，说明收益会随 batch geometry 和同步等待程度变化。

## 4. 融合 Nemotron RoPE

### 问题

RoPE 在每层分别执行切片、`rotate_half`、`cat`、乘法和加法，产生较多小算子及中间 Tensor。

### 原因

原实现是通用 PyTorch 表达式，NPU 无法始终把旋转、乘法和加法合并为一次执行；层数增加后，算子启动和中间结果开销会累积。

### 修改

NPU Tensor 改用 `torch_npu.npu_rotary_mul`，3D 输入临时扩展为算子支持的 4D 布局后恢复；非 NPU 设备继续使用原 PyTorch 路径。

### 收益

在八卡 EgoSuite A/B 中，加入 RoPE 后 step 从 55.195s 降至 54.299s，相对同步优化基线提升 1.62%。与 RMSNorm 同时启用时效果最好。

## 5. 融合 Nemotron RMSNorm

### 问题

每层 RMSNorm 都会执行 FP32 类型转换、平方、均值、`rsqrt`、缩放和结果 dtype 恢复，算子数量多且会保留中间 Tensor。

### 原因

通用实现用多个 PyTorch 算子表达 RMSNorm；深层 Transformer 会重复支付 kernel 启动、类型转换和临时显存成本。

### 修改

NPU Tensor 使用 `torch_npu.npu_rms_norm`，weight 转成输入 dtype 后由单个融合算子完成归一化；CPU/CUDA 保留原实现。

### 收益

- 单独启用 RMSNorm：step 从 55.195s 降至 53.187s，提升 3.64%；Peak allocated 从 46.240 GiB 降至 42.331 GiB，减少 3.909 GiB/卡。
- RoPE 与 RMSNorm 同时启用：step 从原始路径的 55.888s 降至 52.217s，提升 6.57%；Peak allocated 减少 3.399 GiB/卡。

## 6. Teacher-forcing Attention 使用 NPU 原生 GQA

### 问题

原 `masked_sdpa` 为匹配 Q heads，会通过 `repeat_interleave` 将 K/V heads 物化展开。Teacher-forcing 长序列下，这些临时 K/V Tensor 占用数 GiB 显存。

### 原因

通用 PyTorch SDPA 路径没有直接消费较少的 K/V heads，而 Ascend `npu_fusion_attention` 原生支持 GQA。如果先扩展 K/V，就失去了 GQA 的显存优势。

### 修改

NPU 路径将 blocked mask 直接传给 `torch_npu.npu_fusion_attention`，保持 Q/K/V 原始 head 数；CPU/CUDA 路径继续扩展 K/V 并调用 PyTorch SDPA。

### 收益

| 四卡复测       | `masked_sdpa` 基线 | NPU 原生 GQA |        变化 |
| :------------- | -----------------: | -----------: | ----------: |
| Step 均值      |            32.862s |      33.837s |      +2.97% |
| Step P50       |            32.898s |      33.891s |      +3.02% |
| Peak allocated |         42.425 GiB |   39.197 GiB |  -3.227 GiB |
| Peak reserved  |         58.916 GiB |   58.553 GiB |  -0.363 GiB |

10 个配对 step 中 GQA 均更慢，但 allocated 显存稳定下降。reserved 降幅小，是因为缓存分配器继续持有已申请的 segment。该改动是显存优化，不是吞吐优化；当前分支选择保留，代价是约 2.97% 的实测吞吐回退。

## 7. 超长 Episode 均衡重叠切片

### 问题

`max_video_duration_s` 原本只负责丢弃超过阈值的整条 episode。阈值设小会损失长视频，设大则允许长样本进入独立 Attention window，显著拖慢训练。

### 原因

packing token 总量接近不代表计算量接近。每个样本的 Attention 成本近似为：

```text
sum_i(s_und_i^2 + s_gen_i * (s_und_i + s_gen_i))
```

成本随单样本序列长度近似二次增长。61s 上限提高到 91s 后，总 FLOPs 增加 40.95%，step time 同步增加 44.27%；Data wait 基本不变，证明根因在模型计算而非读取。

| 时长上限实验         |          61s |          91s |       变化 |
| :------------------- | -----------: | -----------: | ---------: |
| 保留 episode         |           78 |          122 |        +44 |
| 平均 episode 时长    |      26.652s |      42.904s |    +60.97% |
| 临界 step 均值       |      30.607s |      44.156s |    +44.27% |
| Peak allocated       |   41.095 GiB |   45.287 GiB | +4.193 GiB |
| 总 FLOPs/step        | 3.009 PFLOPs | 4.242 PFLOPs |    +40.95% |

### 修改

在 parquet metadata 展开阶段把超长 episode 切成均衡、连续、独立的窗口，不生成新视频文件。窗口采用帧级边界，首尾完整覆盖源区间，相邻窗口精确 overlap，单窗口不超过上限。FPS 下采样共享源 episode 的相位，保证 overlap 中相同源帧不会因窗口边界错位。

```toml
max_video_duration_s   = 61.0
long_video_policy      = "split"
video_window_overlap_s = 5.0
```

### 收益

- 126 个源 episode 全部保留并展开为 175 个样本；物化帧从 169,921 增至 177,271，overlap 成本为 4.33%。
- 无 Profiler 四卡 A/B：最慢 rank 稳态 P50 从 47.98s 降至 35.23s，下降 26.6%。
- Level1 Profile：Training step P50 从 48.875s 降至 36.123s，下降 26.1%。
- FlashAttention 正向/反向总时间下降约 33.7%/34.7%，ReduceScatter/AllGather 聚合时间下降约 46.9%/54.6%。
- 数据集测试 26 项通过，真实 PyAV `decode_transform` 冒烟 4/4 成功。

收益来自缩短独立 Attention window，而不是丢弃长 episode。代价是 overlap 让长 episode 产生多个独立样本，增加 4.33% 物化帧，并提高长 episode 的采样权重。

## 8. 最终决策与边界

1. 保留 Host 同步、NPU RoPE 和 NPU RMSNorm 优化。
2. Vision Edge LeRobot 使用 `split + 61s + 5s overlap`；通用 Python 默认保持 `drop + overlap=0`，避免改变其他任务的数据分布。
3. 当前分支保留 NPU 原生 GQA，以吞吐换显存；如果训练不受显存约束，应回退或增加开关。
4. 下一步应按 Attention FLOPs 而不是样本数或 token 总数做跨 rank packing 均衡。
5. structured JSON caption 在切片后仍复制源 caption 并告警；使用包含 duration、FPS 或 timestamp 的 caption 前，需要实现字段重写。
6. 各实验的卡数和冻结条件不同，只能比较各自 A/B，不能横向比较绝对 step time。
7. 时长与切片 A/B 使用 15 FPS，而当前示例 TOML 是 30 FPS；最终训练配置需要复测收益幅度。
