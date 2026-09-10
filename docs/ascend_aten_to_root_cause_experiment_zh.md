# Ascend `aten::to` 超长耗时根因与优化实验报告

日期：2026-09-09  
分支：`perf/vision-edge-ascend-profile`

## 1. 实验目标

分析并修复 Cosmos3 Vision Edge 四卡训练中两类异常 `aten::to`：

1. `51s~57s` 附近、持续约 6.32 秒的单次 `aten::to`；
2. Transformer 每层规律出现的 `int32[9] aten::to`，每对中第一次约 220ms、第二次约 0.06ms。

实验要求区分真正的数据搬运、NPU stream 同步以及 FSDP/HCCL 跨 rank 等待，不能仅依据
PyTorch API 名称归因。

## 2. 实验环境与基线

- 基线结果：`/data5T/zheng/cosmos-ascend-profile/cosmos-profile-logs/fsdp_cast_verification/ascend_profile_20260909_133451`
- rank 0 PyTorch 线程：PID/thread `2896028`
- 设备：4 × Ascend 910B3
- FSDP mesh：`dp_replicate=1, dp_shard=4`
- 计算精度：BFloat16；FSDP master dtype：Float32
- Profile step：`ProfilerStep#7`
- 数据集：Cosmos3-DROID success
- Attention backend：`npu_fusion_attention`

## 3. 问题一：6.32 秒的 `aten::to`

### 3.1 现象

目标事件位于：

```text
COSMOS::FORWARD/12_DENOISE
└── COSMOS::DENOISE/02_ENCODE_VISION
    └── COSMOS::ENCODE_VISION/07_EMBED_CLEAN_TIMESTEPS
        └── aten::to                         6321.334 ms
            └── aten::_to_copy              6321.328 ms
                └── aten::copy_             6321.293 ms
```

输入只是一个 `float32[128]` Tensor。对应代码原为：

```python
freqs = torch.exp(
    -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
).to(device=t.device)
```

### 3.2 `aten::to` 内部实证

| 内部 CANN API | 耗时 |
|---|---:|
| `aclrtSynchronizeStream` | 6321.201 ms |
| `aclrtMemcpy` | 0.052 ms |
| event query 合计 | 约 0.068 ms |

实际复制 128 个 float 只用了 0.052ms，`99.9979%` 的时间是 stream 同步。因此
`aten::to` 只是同步等待的承载者，不是大数据传输。

### 3.3 与 FSDP AllGather 的时间对齐

该 `aten::to` 内部同步与 `hcom_allGather__612_93_1` 几乎完全重叠：

| 事件 | 开始时间 ns | 结束时间 ns | 耗时 |
|---|---:|---:|---:|
| rank0 `aten::to` | 1788961249404302030 | 1788961255725636390 | 6321.334 ms |
| rank0 HCCL AllGather | 1788961249399482020 | 1788961255721165500 | 6321.683 ms |

AllGather 元素数为 `135,596,592 BFP16`，约 258.6MiB。它由
`parallelize_vfm_network.py` 对根模型调用 `fully_shard(model, ...)` 后，FSDP2
pre-forward 自动执行参数 unshard 引起。

四个 rank 的同一 collective：

| rank | 相对最早开始 | HCCL 事件耗时 | 结束关系 |
|---:|---:|---:|---|
| 3 | 0ms | 6744.667 ms | 与其他 rank 基本同时 |
| 0 | +422.785ms | 6321.683 ms | 与其他 rank 基本同时 |
| 2 | +1105.817ms | 5638.865 ms | 与其他 rank 基本同时 |
| 1 | +6726.060ms | **18.620 ms** | 与其他 rank 基本同时 |

rank1 最晚进入，但只需要 18.62ms 就完成；其他 rank 的长 HCCL 区间主要是在等
rank1，而不是传输 258.6MiB 本身需要 6 秒。

### 3.4 rank1 晚到的上游根因

AllGather 前的 `COSMOS::FORWARD/09_POST_NOISE_PACK`：

| rank | 耗时 |
|---:|---:|
| 0 | 3848.316 ms |
| 1 | **9793.775 ms** |
| 2 | 3903.969 ms |
| 3 | 3264.797 ms |

rank1 比其他 rank 慢约 6 秒，且后续四卡重新对齐。rank1 在此范围内有：

| PyTorch API | 调用次数 | 累计嵌套时间 |
|---|---:|---:|
| `aten::select` | 212,912 | 4227.370 ms |
| `aten::unbind` | 18 | 3670.702 ms |
| `aten::item` | 266,130 | 3014.410 ms |
| `aten::_local_scalar_dense` | 266,130 | 1619.145 ms |

数据规模在四卡间接近（约 51,780~52,560 个索引），排除了 rank1 输入显著更大的解释。
热点来自 `teacher_forcing.py` 中对 Tensor 的 Python 逐元素遍历和 `int(tensor)`：

```python
{int(source_indexes[new_index]): int(new_index) for new_index in stream_indexes}
[source_to_new[int(index)] for index in indexes]
```

此实现生成几十万次 `unbind/select/item`。rank1 在 Host CPU 调度/资源争用下单次小操作
更慢，最终形成 rank 到达 collective 的偏斜。

### 3.5 根因结论

```text
POST_NOISE_PACK Python 标量循环
    -> rank1 Host 侧晚到约 6.7s
    -> FSDP root AllGather 中其他 rank 等待
    -> CPU 创建的 freqs.to(NPU) 首次阻塞式同步
    -> 6.32s 被归到 aten::to
```

### 3.6 解决方案

1. 将 source index 字典构造和 remap 改为 dense Tensor lookup 和高级索引；
2. `torch.arange` 直接在 `t.device` 创建，移除 CPU -> NPU 的小 Tensor 阻塞复制；
3. 若向量化后仍有 rank Host 抖动，再进行 rank/DataLoader worker 的 CPU/NUMA 绑核。

第二项只移除错误归因的同步点；真正消除 6 秒等待依赖第一项消除 rank 晚到。

## 4. 问题二：Transformer 每层规律性 `aten::to`

### 4.1 现象与计数实证

在一个 28 层 Transformer forward 中，筛选 `dtype=int32, shape=[9]`：

- `aten::to` 共 56 次，严格对应 `28 层 × 2 次`；
- 奇数次通常为 `217~228ms`；
- 紧随其后的偶数次仅 `0.056~0.101ms`；
- 56 次总计 `6132.789ms`。

典型一对：

```text
aten::to  217.218 ms
aten::to    0.066 ms
npu::npu_fusion_attention
```

长事件内部 `aclrtSynchronizeStream=217.118ms`，实际 `aclrtMemcpy=0.040ms`。

### 4.2 代码根因

`npu_fusion_attention/functions.py` 每次调用都执行：

```python
actual_seq_qlen = cumulative_seqlen_Q.tolist()
actual_seq_kvlen = cumulative_seqlen_KV.tolist()
```

Ascend fused-attention 接口需要 Python `list[int]`，但 cumulative offsets 已在 NPU。
NPU Tensor `.tolist()` 必须同步 stream 并 D2H。第一次 `.tolist()` 排空前序 QKV/RoPE
计算，第二次因 stream 刚同步而很短。该逻辑在 28 层重复，形成规律性空洞。

### 4.3 解决方案

1. sequence packing 创建 offsets 时，Python `split_lens/sample_lens` 本来就存在，直接
   将对应 host tuple 绑定到 offset Tensor；
2. NPU backend 优先使用预计算 host tuple，不再从 NPU Tensor `.tolist()`；
3. 对非 Cosmos sequence-pack 调用保留一个按 Tensor identity、mutation version 和弱引用
   校验的 64 项 LRU fallback，使相同 metadata 至多同步一次，并避免陈旧缓存。

这比 `non_blocking=True` 更有效，因为 `.tolist()` 的调用者立即需要 Python 数据，本身无法
异步；必须避免从设备读取。

## 5. 修改内容

| 文件 | 修改 |
|---|---|
| `data/generator/sequence_packing/teacher_forcing.py` | dict/list scalar loop 改为 Tensor lookup/indexing |
| `data/generator/sequence_packing/runtime.py` | 构造 offsets 时保留 host actual sequence lengths |
| `model/attention/npu_fusion_attention/functions.py` | 使用 host metadata；增加安全 fallback cache |
| `model/generator/mot/modeling_utils.py` | frequency `arange` 直接创建在目标 device |
| `model/attention/npu_fusion_attention/functions_test.py` | 缓存、失效及 host metadata 测试 |

## 6. 修改后验证

### 6.1 正确性测试

```text
77 passed, 17 warnings
ruff: All checks passed
```

覆盖 teacher-forcing packing、causal hook、SequencePack metadata、GenKV cache 以及
NPU actual-sequence-length cache 命中/失效。

### 6.2 等价 CPU 微基准

使用与 trace 同量级的 52,560 个 source indexes，比较旧字典/标量 remap 和新 Tensor remap，
各运行 5 次取中位数：

| 实现 | 中位耗时 | 输出 |
|---|---:|---|
| 旧 Python dict/list 标量循环 | 661.538 ms | reference |
| 新 Tensor lookup/indexing | 7.322 ms | 与 reference 完全相等 |
| 加速 | **90.3×** | `torch.equal=True` |

### 6.3 四卡端到端 Profile

优化后结果：

```text
/data5T/zheng/cosmos-ascend-profile/cosmos-profile-logs/aten_to_fix_validation/
ascend_profile_20260909_190247
```

使用与基线相同的数据集、随机种子、4 rank、4 workers 和 `ProfilerStep#7`。训练正常完成，
四个 rank 的 step 墙钟时间均为 35.944s；基线为 45.272s，减少 **9.328s / 20.6%**。

#### 6.3.1 `POST_NOISE_PACK`

| rank | 基线 | 优化后 | 降幅 |
|---:|---:|---:|---:|
| 0 | 3848.316 ms | 342.850 ms | 91.1% |
| 1 | 9793.775 ms | 290.666 ms | 97.0% |
| 2 | 3903.969 ms | 277.936 ms | 92.9% |
| 3 | 3264.797 ms | 284.540 ms | 91.3% |

rank1 的 `aten::item` 从 266,130 次降至 2,630 次，`aten::select` 从 212,912 次降至
2,634 次，分别减少约 99.0% 和 98.8%。四卡该范围收敛到 278~343ms，不再存在 rank1
特有的 9.79 秒 Host 长尾。这验证了 Python Tensor 逐元素遍历是 rank 晚到的根因。

#### 6.3.2 root FSDP AllGather

同一个 `135,596,592 BFP16` AllGather：

| rank | 基线 HCCL 事件 | 优化后 HCCL 事件 |
|---:|---:|---:|
| 0 | 6321.683 ms | 562.567 ms |
| 1 | 18.620 ms | 317.631 ms |
| 2 | 5638.865 ms | **17.649 ms** |
| 3 | 6744.667 ms | 473.708 ms |

优化后 rank2 最晚进入 collective，实际通信约 17.65ms。最早和最晚 rank 的到达差由约
6.73s 降至约 0.54s，最大 HCCL 事件减少 91.7%。剩余约 0.54s 偏斜来自更上游的
`GENERATION_TOKENIZE`：四卡耗时为 7.47s、7.77s、8.07s、7.61s，而不是
`POST_NOISE_PACK`。

#### 6.3.3 timestep embedding 的长 `aten::to`

rank0 的 `COSMOS::ENCODE_VISION/07_EMBED_CLEAN_TIMESTEPS`：

```text
基线：6322.790 ms
优化后：1.272 ms
```

优化后该范围不再存在 `float32[128]` CPU -> NPU `aten::to`。范围内仅剩 FSDP mixed
precision 对 Linear 参数的 BFloat16 转换，每次 0.040~0.062ms。原 6.32 秒事件消失，
验证了 CPU 创建 frequency tensor 的 `.to(device)` 是暴露前序 AllGather 等待的同步点。

#### 6.3.4 Transformer sequence-length `.tolist()`

优化后的四个 rank 在 `COSMOS::DENOISE/07_TRANSFORMER` 内均未发现累计序列长度小 Tensor
的 `aten::to`；基线 rank0 为 `int32[9]` 56 次、总计 6132.789ms。

Transformer host range 四卡平均值：

```text
基线：约 6753.6 ms
优化后：约 5339.1 ms
下降：约 20.9%
```

这验证了每层两次长短相间的 `aten::to` 确实由 NPU Tensor `.tolist()` 引起；保留创建
offset 时已有的 host lengths 后，不再发生相应 D2H 和 stream synchronization。

### 6.4 验收结论

五项验收指标全部通过：

1. `POST_NOISE_PACK` 的数十万次标量操作减少约 99%；
2. root AllGather 的跨 rank 到达偏斜由约 6.73s 降到约 0.54s；
3. 最后到达 rank 的实际 AllGather 为 17.65ms，证明原多秒耗时主要是等待；
4. `EMBED_CLEAN_TIMESTEPS` 的 `float32[128] aten::to` 消失；
5. Transformer 每层 cumulative-offset `.tolist()` 对应的 `aten::to` 消失。

## 7. 复现与数据库核验方法

在每个 rank 的 `ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_<rank>.db` 中：

```sql
-- 找通信量相同的 root FSDP AllGather
SELECT startNs, endNs, (endNs-startNs)/1e6 AS ms, count, dataType
FROM COMMUNICATION_OP
WHERE count = 135596592;

-- 检查目标 aten::to 的同步和实际 memcpy
SELECT s.value, (c.endNs-c.startNs)/1e6 AS ms
FROM CANN_API c JOIN STRING_IDS s ON s.id = c.name
WHERE c.startNs >= :aten_to_start AND c.endNs <= :aten_to_end;

-- 检查 Transformer 的 int32[9] aten::to 次数和总耗时
SELECT COUNT(*), SUM(CAST(endNs AS INTEGER)-CAST(startNs AS INTEGER))/1e6
FROM PYTORCH_API
WHERE name = (SELECT id FROM STRING_IDS WHERE value='aten::to')
  AND inputDtypes = (SELECT id FROM STRING_IDS WHERE value LIKE 'int;%')
  AND inputShapes = (SELECT id FROM STRING_IDS WHERE value LIKE '9;%');
```

分析 operator duration 时必须继续展开 CANN API 和 HCCL 时间线。PyTorch API 的墙钟耗时
可能包含前序异步工作，不能直接等价为该 API 自身计算或搬运成本。
