# OmniMoT Causal Teacher-Forcing 改造说明与验证指南

本文说明 `feat/omni-mot-causal-training` 分支做了什么、为什么这样做，以及如何在
Ascend NPU 上进行第一轮功能验证。阅读本文不要求了解扩散模型或注意力算法。

## 1. 一句话说明

原模型一次只接收一条“被加噪的视频”。改造后，同一次 forward 会同时接收：

```text
完整 clean 视频（答案/历史） + 完整 noisy 视频（需要预测的内容）
```

注意力 mask 严格控制模型只能读取允许的历史 clean block，不能偷看当前或未来的
clean 答案。这就是本文采用的 Scheme B teacher forcing。

## 2. 用外行能理解的例子解释

可以把一段视频理解成一本按页排列的答案册：

- `clean stream` 是未被破坏的答案册；
- `noisy stream` 是被涂花、需要模型修复的答案册；
- 一个 `block` 是连续若干页；
- attention mask 是监考规则。

模型修复第 `i` 个 noisy block 时：

1. 可以看文字题目；
2. 可以看有限数量的、更早的 clean blocks；
3. 可以看当前 noisy block 内的所有位置；
4. 不能看当前 clean block；
5. 不能看未来 clean/noisy blocks。

因此模型能学习“根据历史生成当前内容”，但不能直接抄当前答案。

## 3. frame、latent frame、chunk 和 block

这些词容易混淆：

- 原始视频 frame：普通视频画面；
- VAE latent frame：视频经过 Wan VAE 时间压缩后的表示；
- 基础 chunk：本项目中一个 VAE latent frame，即约 4 个原始视频帧；
- causal block：随机包含 `S=1..4` 个基础 chunks。

例如有 7 个 VAE latent frames，随机得到 `S=3`：

```text
latent frame:  0 1 2 | 3 4 5 | 6
causal block:  0 0 0 | 1 1 1 | 2
```

对应实现位置：
[`teacher_forcing.py::build_teacher_forcing_layout()`](../cosmos_framework/data/generator/sequence_packing/teacher_forcing.py#L103)。

最后一个 block 可以不完整，不要求视频长度能被 `S` 整除。block 始终从 latent
frame 0 开始划分，不存在条件前缀边界或首帧特殊重切。

## 4. 最终确定的训练语义

本实现沿用 Lingbot-VA 的整视频 teacher-forcing 定义，不新增 TI2V/首帧条件训练：

```text
N0 -> N0
N1 -> 历史 clean blocks + N1
N2 -> 历史 clean blocks + N2
...
```

对应实现位置：
[`teacher_forcing.py::build_dense_teacher_forcing_gen_mask()`](../cosmos_framework/data/generator/sequence_packing/teacher_forcing.py#L197)。

第一个 noisy block `N0` 没有更早的 clean block，因此只能读取文字和当前 noisy
block。Lingbot-VA 推理阶段固定第一个 latent frame 的行为，不属于本次
training-only 网络改造。

每次 forward 为整个 packed batch 共同采样：

- `S ∈ [1,4]`：一个 causal block 含多少个 VAE latent chunks；
- `K ∈ [1,32]`：最多读取多少个历史 clean blocks。

batch 内共享 `S/K` 可以让 mask 几何保持一致，同时每次 forward 的随机变化又能增强
模型对不同生成粒度和历史长度的适应能力。

## 5. Attention mask 的准确规则

设 query 位于 block `i`，历史窗口左边界为：

```text
lo(i) = max(0, i - K)
```

对应实现位置：
[`teacher_forcing.py::build_dense_teacher_forcing_gen_mask()`](../cosmos_framework/data/generator/sequence_packing/teacher_forcing.py#L197)。

| Query | 可以读取的视觉 KV | 禁止读取 |
| --- | --- | --- |
| clean `Ci` | `Clo(i)..Ci` | 所有 noisy、窗口外 clean、未来 clean |
| noisy `Ni` | `Clo(i)..C(i-1)` 和当前 `Ni` | 当前 clean `Ci`、未来、其他 noisy blocks |

所有视觉 query 都可以读取同一条样本的 UND/text token。不同 packed samples 之间完全
隔离。

这里最重要的是严格小于：

```text
noisy block i -> clean block j 仅当 j < i
```

如果写成 `j <= i`，模型就能读取当前 clean 答案，训练 loss 看似很快下降，但推理时
没有这份答案，模型会失效。

## 6. 代码数据流

一次训练 step 的数据流如下：

```text
视频 -> Wan VAE -> clean x0
                    |
                    +-> 普通 flow-matching 加噪 -> noisy xt
                    |
                    +-> post-noise hook 构造 [UND | clean | noisy]
                                      |
                                      +-> 复制相同 position IDs
                                      +-> clean timestep = 0
                                      +-> noisy timestep = 原采样 t
                                      +-> 构造随机 S/K Dense mask
                                                    |
                                                    v
                                         单次网络 forward
                                                    |
                                                    v
                                      只取 noisy stream 输出
                                                    |
                                                    v
                                         原有 flow-matching loss
```

这张流程图各节点对应的代码位置见本节 6.1～6.5，以及 6.6 的总索引表。

### 6.1 Causal 模型入口

`OmniMoTCausalModel` 继承原来的 `OmniMoTModel`。它不复制整套训练流程，只覆盖
`post_noise_packing_hook`：普通路径完成加噪后，再插入 clean stream。这样原有 VAE、
噪声调度、decoder 和 loss 都继续复用。

对应代码位置：

- [`omni_mot_causal_model.py::OmniMoTCausalModel.post_noise_packing_hook()`](../cosmos_framework/model/generator/omni_mot_causal_model.py#L23)；
- [`causal_teacher_forcing.py::validate_teacher_forcing_config()`](../cosmos_framework/model/generator/causal_teacher_forcing.py#L35)；
- [`causal_teacher_forcing.py::expand_teacher_forcing_training_sequence()`](../cosmos_framework/model/generator/causal_teacher_forcing.py#L73)。

### 6.2 双流 packing

原始单样本布局：

```text
[UND | vision]
```

扩展后：

```text
[UND | clean vision | noisy vision]
```

clean/noisy 对应同一个物理视频位置，因此复制相同 position IDs，而不是把 noisy
stream 当作更晚的一段视频。`PackedSequence.vision` 仍指向 noisy 目标，现有 decoder
和 MSE loss 不需要理解双流结构；额外的 clean payload 和布局元数据放在
`TeacherForcingData` 中。

对应代码位置：

- [`teacher_forcing.py::TeacherForcingLayout`](../cosmos_framework/data/generator/sequence_packing/teacher_forcing.py#L28)；
- [`teacher_forcing.py::TeacherForcingData`](../cosmos_framework/data/generator/sequence_packing/teacher_forcing.py#L61)；
- [`teacher_forcing.py::expand_packed_sequence_for_teacher_forcing()`](../cosmos_framework/data/generator/sequence_packing/teacher_forcing.py#L349)。

### 6.3 时间步编码

- clean stream 使用真实 `x0`，timestep 固定为 0；
- noisy stream 使用 `xt`，保留原 flow-matching timestep；
- clean hidden states 不直接计算生成 loss，但允许梯度通过合法 attention 路径回传。

对应代码位置：
[`cosmos3_vfm_network.py::Cosmos3VFMNetwork._encode_vision()`](../cosmos_framework/model/generator/mot/cosmos3_vfm_network.py#L574)
中的 `packed_seq.teacher_forcing` 分支。

### 6.4 Dense attention

首版使用一张 boolean Dense Mask，然后只调用一次 PyTorch SDPA：

```text
Q_GEN x K_[UND|clean|noisy] -> one masked softmax
```

没有把注意力拆成多次调用，也没有使用外部 LSE 合并。这样选择的原因是：

1. 语义直接，容易和显式矩阵 oracle 对照；
2. forward/backward 实现简单；
3. 不依赖 Ascend `npu_fusion_attention` 是否正确返回 LSE；
4. 当前阶段允许牺牲一部分吞吐，优先确认网络逻辑正确。

代价是 Dense Mask 的显存随 query×key 二次增长。Mask 尺寸直接由当前 packed layout
展开后的实际长度决定；原始 packed sequence 仍可由 dataloader 的 token budget 控制，
但 teacher forcing 不再提供第二个独立的展开后长度配置。真实尺寸性能优化属于后续工作，
不在当前功能闭环中。

对应代码位置：

- mask 构造：
  [`teacher_forcing.py::build_dense_teacher_forcing_gen_mask()`](../cosmos_framework/data/generator/sequence_packing/teacher_forcing.py#L270)；
- attention metadata 构造：
  [`attention.py::build_packed_sequence()`](../cosmos_framework/model/generator/mot/attention.py#L685)；
- UND/GEN attention 分发：
  [`attention.py::teacher_forcing_attention()`](../cosmos_framework/model/generator/mot/attention.py#L278)；
- 单次 SDPA：
  [`teacher_forcing_attention.py::teacher_forcing_dense_attention()`](../cosmos_framework/model/generator/mot/teacher_forcing_attention.py#L11)。

### 6.5 输出和 loss

网络会产生 clean/noisy 两部分 hidden output，但 decoder 只恢复 noisy stream，并沿用
原有 flow-matching target、condition mask 和 loss 计算。clean stream 不单独增加 loss。

对应代码位置：

- noisy hidden state 解码：
  [`cosmos3_vfm_network.py::Cosmos3VFMNetwork._decode_vision()`](../cosmos_framework/model/generator/mot/cosmos3_vfm_network.py#L662)；
- noisy 索引辅助函数：
  [`teacher_forcing.py::select_teacher_forcing_noisy_outputs()`](../cosmos_framework/data/generator/sequence_packing/teacher_forcing.py#L449)；
- 原 flow-matching loss：
  [`omni_mot_model.py::OmniMoTModel._compute_flow_matching_loss()`](../cosmos_framework/model/generator/omni_mot_model.py#L1097)。

### 6.6 代码位置总索引

这里使用 `文件路径::类/函数` 标记实现位置，不写固定行号，避免后续代码调整后行号失效。

| 功能 | 对应代码位置 |
| --- | --- |
| causal 模型入口 | [`OmniMoTCausalModel`](../cosmos_framework/model/generator/omni_mot_causal_model.py#L16) |
| 参数合法性与 S/K 采样入口 | [`validate_teacher_forcing_config()`](../cosmos_framework/model/generator/causal_teacher_forcing.py#L35)、[`expand_teacher_forcing_training_sequence()`](../cosmos_framework/model/generator/causal_teacher_forcing.py#L73) |
| S/K 随机采样 | [`sample_teacher_forcing_parameters()`](../cosmos_framework/data/generator/sequence_packing/teacher_forcing.py#L83) |
| block/sample/stream 布局 | [`build_teacher_forcing_layout()`](../cosmos_framework/data/generator/sequence_packing/teacher_forcing.py#L103) |
| clean/noisy 双流展开 | [`expand_packed_sequence_for_teacher_forcing()`](../cosmos_framework/data/generator/sequence_packing/teacher_forcing.py#L349) |
| Dense mask 与一维长度保护 | [`build_dense_teacher_forcing_gen_mask()`](../cosmos_framework/data/generator/sequence_packing/teacher_forcing.py#L197) |
| clean timestep=0 | [`Cosmos3VFMNetwork._encode_vision()`](../cosmos_framework/model/generator/mot/cosmos3_vfm_network.py#L574) |
| attention metadata | [`build_packed_sequence()`](../cosmos_framework/model/generator/mot/attention.py#L636) |
| teacher-forcing attention 分发 | [`teacher_forcing_attention()`](../cosmos_framework/model/generator/mot/attention.py#L246) |
| 单 softmax Dense SDPA | [`teacher_forcing_dense_attention()`](../cosmos_framework/model/generator/mot/teacher_forcing_attention.py#L52) |
| noisy stream 输出与 loss | [`Cosmos3VFMNetwork._decode_vision()`](../cosmos_framework/model/generator/mot/cosmos3_vfm_network.py#L662)、[`OmniMoTModel._compute_flow_matching_loss()`](../cosmos_framework/model/generator/omni_mot_model.py#L1097) |
| Hydra causal 模型组 | [`MOT_CAUSAL_DDP_CONFIG`](../cosmos_framework/configs/base/defaults/model.py#L37)、[`MOT_CAUSAL_FSDP_CONFIG`](../cosmos_framework/configs/base/defaults/model.py#L50) |
| TOML causal 字段定义 | [`ModelConfig`](../cosmos_framework/configs/toml_config/sft_config.py#L300) |

## 7. 为什么采用这种方案

当前已经实现的是 Scheme B：把 clean/noisy 放进同一次网络 forward，并用单张 Dense
Mask 表达完整可见性。历史讨论中的 Scheme A 则是另一种**模型级执行图**：先计算 clean
路径并保存逐层可微 KV，再让 noisy 路径读取对应层的历史 clean KV。它不能和“多次
attention 后用 LSE 合并”的 attention 级实验混为一谈。

当前选择 Scheme B，原因不是它最终性能最好，而是它最适合作为正确性基线：

- Scheme B 的 clean/noisy 可见性可以在一张 mask 中完整表达；
- 不依赖尚未验证的 NPU LSE 输出；
- 出错时可以逐元素检查 mask；
- CPU 显式 attention 能提供独立数值参考；
- 等语义稳定后，可以用更快的 Ascend/block-sparse 实现替换 backend，而不改变上层
  packing 和训练定义。

### 7.1 历史 Scheme A 的核心思想

Scheme A 不再构造一个包含所有合法和非法 QK 位置的全局矩阵。它在加噪前先运行 clean
路径，并在每个 Transformer layer 保存 UND/clean 的 K/V。随后 noisy block `Ni` 只把
真正允许读取的 token 作为该 block 的 KV：

```text
clean x0
   |
   +-- clean causal forward
           |
           +-- layer 0 UND/clean K/V
           +-- layer 1 UND/clean K/V
           +-- ...

noisy block Ni at layer l
   |
   +-- Q = Ni
   +-- KV = same-sample UND
           + clean[max(0, i-K) .. i-1]
           + current noisy block Ni
   |
   +-- attention output -> residual/MLP -> next noisy layer
```

clean block 自身的规则与 noisy 不同：

```text
clean block Ci KV = UND + clean[max(0, i-K) .. i]
noisy block Ni KV = UND + clean[max(0, i-K) .. i-1] + Ni
```

因此 clean query 可以读取当前 clean block，noisy query 不能读取当前 clean 答案，但可以
在当前 noisy block 内做 full attention。该规则必须和
[`build_dense_teacher_forcing_gen_mask()`](../cosmos_framework/data/generator/sequence_packing/teacher_forcing.py#L197)
保持一致；Dense Mask 应继续作为 Scheme A 的语义 oracle。

现有基础模型已经预留了这类执行图的生命周期入口：

- [`memory_init_training()`](../cosmos_framework/model/generator/omni_mot_model.py#L742)：训练 memory/cache 初始化和 segment 生命周期；
- [`build_memory_state()`](../cosmos_framework/model/generator/omni_mot_model.py#L769)：构造 `ARMemoryState` 或 `KVCacheTrainMemoryState` 的注入点；
- [`pre_noise_memory_hook()`](../cosmos_framework/model/generator/omni_mot_model.py#L790)：加噪前运行 clean forward 的注入点。

这些 hook 证明基础代码允许表达 Scheme A，但当前 `OmniMoTCausalModel` 并未实现该
memory/cache 数据流；下面内容都是候选优化设计，不是已经验证的行为。

### 7.2 Scheme A 如何支持多样本 TND packing

Scheme A 支持 dataloader 已经产出的多样本 TND packing，但不能简单地把“一条完整样本”
作为一个 full-attention TND segment。因为同一条样本内，不同 block 的可见 KV 不同。

进入 attention 前，需要把 TND 的逻辑 segment 进一步定义成一个
`(sample_id, block_id, stream)` group。以两个样本、历史窗口 `K=2` 为例，noisy 阶段为：

```text
Q groups:
[N_A0 | N_A1 | N_A2 | N_B0 | N_B1]

KV groups:
[UND_A + N_A0
 | UND_A + C_A0 + N_A1
 | UND_A + C_A0 + C_A1 + N_A2
 | UND_B + N_B0
 | UND_B + C_B0 + N_B1]
```

然后构造各 group 的累计 Q/KV 边界：

```text
cu_q  = cumulative([len(N_A0), len(N_A1), len(N_A2), len(N_B0), len(N_B1)])
cu_kv = cumulative([len(KV_A0), len(KV_A1), len(KV_A2), len(KV_B0), len(KV_B1)])
```

第 `j` 个 Q segment 只对应第 `j` 个 KV segment。因此跨样本隔离、stream 规则和历史
窗口由“哪些 token 被 gather 进 group”决定，而不是依赖全局 Dense Mask。所有样本、所有
block 仍可拼成一次 TND fused-attention 调用，不要求 Python 按样本或 block 循环。

当前 `TeacherForcingLayout` 已保存 `sample_ids`、`stream_ids`、`block_ids` 和输出索引，
[`build_teacher_forcing_layout()`](../cosmos_framework/data/generator/sequence_packing/teacher_forcing.py#L103)
逐样本构造了这些信息；`PackedSequence` 同时保留 `sample_lens` 和 `split_lens`。现有元数据
足以生成 group 和 scatter 索引，无需取消多样本 packing。

需要注意：普通 TND `actual_seq_lengths` 只能描述连续 segment，不能让多个 Q group
零拷贝共享同一段 UND/clean KV。因此同一份 UND 和重叠 clean history 通常需要 gather
成多份临时 KV。重复的是 KV 数据和访存；每个合法 QK pair 本身并没有因此多计算一次。

### 7.3 Scheme A 与 Scheme B 的计算量

设：

- `U`：UND token 数；
- `V`：单条样本原始视觉 token 数，clean/noisy 各有 `V`；
- `P`：一个 causal block 的视觉 token 数；
- `K`：历史窗口包含的最大 clean block 数。

普通 Dense backend 不会因为 boolean mask 为 false 就跳过对应矩阵乘法，因此 Scheme B
的 attention 主计算量近似为：

```text
C_B ~= 2V * (U + 2V)
```

Scheme A 中，每个 query 只读取 UND、有限 clean history 和当前 block，近似为：

```text
C_A ~= 2V * [U + (K + 1)P]
```

二者比例近似：

```text
C_A / C_B ~= [U + (K + 1)P] / (U + 2V)
```

当 `K` 和 `P` 固定、视频长度持续增长时，Scheme B 的 attention 随 `V` 二次增长，
Scheme A 更接近随 `V` 线性增长。对于多样本 packing，Scheme A 还不会形成不同样本之间
的 QK pair。

这里减少的是 `QK^T`、softmax 和 `P*V` 部分。只要实现没有重复执行 clean prefix，QKV
projection、output projection、Norm 和 MLP 的 token 数仍可以接近 Scheme B 的
`U+2V`。以下朴素实现必须避免：

```text
for each noisy block i:
    重新运行 clean[0..i] 的完整模型 forward
```

它会反复计算 clean prefix，并可能把原本线性的 token 计算变成随 block 数二次增长。
正确实现应让 clean token 每层只计算一次，随后由所有合法 noisy group 复用其 K/V。

### 7.4 Scheme A 是否一定更快

不一定。它的 attention FLOPs 在长序列、有限历史窗口下更小，但新增了：

- UND 和重叠 clean history 的 KV gather；
- TND 临时 KV buffer 写入；
- attention 输出 scatter；
- clean/noisy 两阶段依赖和同步；
- group 数超出算子上限后的多次调度；
- 为节省训练显存而启用 checkpoint 时的反向重算。

因此总时间应理解为：

```text
T_A = 有效 attention 计算
    + KV gather/scatter
    + kernel/阶段调度
    + 可选 checkpoint 重算
```

长视频、block 较大且 `K << num_blocks` 时，减少的 Dense QK 计算更可能覆盖这些开销；
短视频、block 很小、历史接近全长时，Scheme A 可能没有净加速。不能只用理论 FLOPs
决定是否切换，必须同时测量 step time、有效 vision tokens/s、NPU kernel 时间、HBM
带宽和 peak memory。

### 7.5 Scheme A 与 Dense Mask 的显存关系

Scheme B 的单张 bool mask 约占：

```text
M_B_mask ~= 2V * (U + 2V) bytes
```

这张 mask 很大，但可以跨 Transformer layers 复用。Scheme A 不需要这张全局 mask，
其主要新增显存是重复 gather 后的 K/V。所有 noisy groups 的 KV token 副本数近似为：

```text
E_KV ~= (V/P) * U + V * (K + 1)
```

若每个 K 或 V token 的总 head 宽度为 `d_kv`，BF16 下单阶段 gather buffer 近似为：

```text
M_A_gather ~= E_KV * d_kv * 2（K 和 V 两份）* 2 bytes
```

所以 Scheme A 的 buffer 对 `V` 近似线性，但常数可能很大。尤其在训练中，attention
backward 通常需要 K/V；若所有层的展开 buffer 都被 autograd 保留，朴素 Scheme A 的
峰值显存可能高于只保存一张 Dense Mask 的 Scheme B。

仅仅把 group 分批调用并不能保证节省训练显存，因为各 chunk 的 K/V 仍可能被 backward
保存。要让 Scheme A 稳定省显存，通常还需要：

- 按显存预算对 TND groups 做 chunking；
- 对 gather + attention 使用 activation checkpointing，在 backward 重建临时 KV；
- clean/noisy 阶段不同时持有全部展开 buffer；
- 避免保存所有 layers 的完整展开 KV 副本。

因此不能把“Scheme A 天然更省显存”作为设计前提。它的上限更容易控制，但朴素实现反而
可能更占显存。

### 7.6 精度和反向传播风险

Scheme A 数学上可以与 Scheme B 等价，但必须把它视为训练计算图重构，而不是普通推理
KV cache。关键约束如下：

1. **clean KV 不得 detach。** 当前 noisy loss 会经过 clean K/V 回传到 clean hidden
   states 和模型参数。若 clean forward 使用 `torch.no_grad()` 或 cache 被 `detach()`，
   训练目标会改变。
2. **重复 gather 的梯度必须 scatter-add。** 同一个 clean token 被多个 noisy groups
   使用时，所有使用者的梯度必须累加回原 token。PyTorch 可微索引通常能表达该语义，
   自定义 NPU copy/gather 则需要单独验证 autograd。
3. **逐层 cache 必须对齐。** noisy layer `l` 必须读取 clean layer `l` 投影得到的 K/V，
   不能只保存 clean 最后一层 hidden state 给所有 noisy layers 使用。
4. **group 边界必须精确复现 Dense oracle。** 特别检查 noisy 不读取当前 clean、当前
   noisy block 内 full attention、尾部 partial block、mixed-length packing 和 history
   左边界。
5. **checkpoint 重算必须确定。** block size、history、group 索引和 RNG 状态在 backward
   重算时必须与原 forward 一致，不能重新采样布局。
6. **不要求 bitwise 一致。** Dense masked SDPA 和 TND fused attention 的 tiling、
   softmax reduction 与 BF16 累加顺序不同，允许合理浮点误差，但 forward 和 backward
   都必须对照验证。

最大的精度风险是 clean cache 被无意 detach；最大的实现风险是为了性能使用自定义
gather/buffer 后破坏 autograd。两者都会出现“forward 看起来正确，但参数梯度错误”的情况。

### 7.7 推荐的实现与验证顺序

Scheme A 不应直接从 Dense Mask 一步替换为 Ascend 高性能版本。建议分阶段实现：

1. 使用小序列、FP32 和普通 SDPA，实现可微的 exact-KV group reference；
2. 用当前 Dense Mask oracle 比较 output、Q/K/V gradient 和参数 gradient；
3. 覆盖多个 packed samples、mixed lengths、首/尾 block、`K=1/32` 和 partial block；
4. 换成 BF16，确定可接受的 forward/backward 误差阈值；
5. 再接 Ascend TND fused attention，并重复全部梯度比较；
6. 最后加入 group chunking、checkpoint 和 compile，逐项测量正确性与性能。

在通过上述验证前，Scheme B 仍是训练语义的权威基线，不应删除 Dense Mask oracle。

### 7.8 “多次 attention + LSE 合并”是另一条路线

曾经还讨论过把一个 query 的合法 KV 拆成多个集合，例如 UND、clean history、current
noisy，分别调用 attention 后用 log-sum-exp 合并，试图恢复一次统一 softmax。这是
attention 级分解，不等同于上述 clean pre-forward + 可微 KV memory 的 Scheme A。

仅做 LSE 拆分只会取消统一大 mask；如果各分支仍计算全局 dense 矩阵，并不会自动减少
内部空洞。它还依赖后端提供可用于精确合并且反向正确的 LSE。当前没有完成该能力的 NPU
forward/backward 验证，因此不作为近期实现基础。

### 7.9 Scheme B 的逐 packed sample Dense 模式

在不改变 Scheme B 训练语义的前提下，可以利用不同 packed samples 原本就完全隔离这一
性质，把一张全局 block-diagonal Dense Mask 拆成每条样本一张局部 mask：

```text
global:
Q=[GEN_A|GEN_B] x KV=[ALL_A|ALL_B] -> one masked SDPA

per_sample:
GEN_A x ALL_A -> masked SDPA A
GEN_B x ALL_B -> masked SDPA B
outputs = concat([A, B])
```

该模式由以下配置切换：

```toml
teacher_forcing_dense_mode = "global"      # 正确性基线，默认值
teacher_forcing_dense_mode = "per_sample"  # 跳过跨样本 QK 区域
```

`per_sample` 只循环 GEN attention；Norm、QKV projection、UND causal attention、output
projection 和 MLP 仍对完整 packing 一次执行。它不会引入 Scheme A 的 clean pre-forward
或 KV cache，也不会消除单条样本内部由 clean/noisy 与历史窗口形成的空洞。

设 packing 中各样本分别有 `Ui` 个 UND token 和 `Vi` 个原始视觉 token，attention 主
计算量从：

```text
global:     2 * sum(Vi) * [sum(Ui) + 2*sum(Vi)]
per_sample: sum(2*Vi * [Ui + 2*Vi])
```

下降部分正是跨样本、最终被 `same_sample` mask 排除的 QK 区域。代价是每层 GEN
attention 的 kernel 调用数从 1 增加为 packed sample 数，可能降低单 kernel MFU。因此
性能判断应以 step time 和有效 tokens/s 为主，而不是只观察硬件 MFU。

实现保留全局 Dense Mask 构造函数作为 oracle。局部 mask 必须逐样本直接构造，不能先
分配全局 mask 再切片，否则无法获得 mask 峰值显存收益。对应实现位置：

- [`build_per_sample_teacher_forcing_gen_masks()`](../cosmos_framework/data/generator/sequence_packing/teacher_forcing.py)；
- [`teacher_forcing_per_sample_dense_attention()`](../cosmos_framework/model/generator/mot/teacher_forcing_attention.py)；
- [`teacher_forcing_attention()`](../cosmos_framework/model/generator/mot/attention.py)。

## 8. 已完成的代码模块

1. `TeacherForcingLayout`：描述 sample、stream、block 和输出索引。
2. 随机参数采样：batch-shared `S=1..4`、`K=1..32`。
3. Dense mask oracle：固定可见性和防泄漏规则。
4. PackedSequence 双流扩展：支持不同视频长度的 packed batch。
5. clean/noisy 相同 RoPE position IDs。
6. clean timestep=0、noisy timestep=t。
7. global 模式使用一次 Dense SDPA；per-sample 模式每条样本独立 softmax；均支持显式 GQA KV head 扩展。
8. 只恢复 noisy output，复用原 decoder/loss。
9. 独立 `mot_causal_ddp` / `mot_causal_fsdp` 模型组和 TOML 字段。
10. 小模型 CPU forward/backward 与梯度闭环测试。

## 9. 已验证内容

当前定向测试覆盖：

- `S=1..4`；
- `K=1` 和 `K=32`；
- 第一个 noisy block 没有 clean history；
- block 内 full attention；
- 当前 clean block 对 noisy query 永远不可见；
- 尾部 partial block；
- mixed-length packed batch 隔离；
- clean/noisy position IDs 相同；
- Dense attention forward 和 Q/K/V gradient；
- clean/noisy 两条路径均能获得有限、非零梯度。

最新定向 CPU 回归结果为 `101 passed`。这证明网络语义闭环，不代表真实 Ascend 大尺寸训练
已经验证。

## 10. 尚未完成或有意推迟的内容

- NPU BF16 真实模型 forward/loss/backward；
- 多卡 DDP/FSDP；
- activation checkpointing；
- `torch.compile`；
- 真实尺寸 Dense Mask 显存和吞吐；
- 更高性能的无 LSE attention backend；
- causal inference/KV cache；
- TI2V 首帧条件训练。

## 11. 单卡 Ascend smoke 配置

新增文件：

- `examples/toml/sft_config/vision_causal_smoke_edge.toml`
- `examples/launch_sft_vision_causal_smoke_edge.sh`

它的目标不是得到有用模型，而是用最少变量回答以下问题：

1. causal subclass 能否正确构造；
2. clean/noisy 双流能否进入真实 Edge 网络；
3. NPU SDPA 是否接受 boolean Dense Mask；
4. BF16 forward、loss、backward 和 optimizer step 是否完成；
5. loss 和梯度是否有限。

关键配置：

| 配置 | 值 | 原因 |
| --- | --- | --- |
| 设备数 | 1 | 先排除多卡通信变量 |
| 模型组 | `mot_causal_ddp` | 实例化 `OmniMoTCausalModel` |
| 精度 | BF16 | 目标 Ascend 训练精度 |
| 数据 | T2V-only、17 RGB frames | 不混入条件帧并控制序列长度 |
| batch | 最多 1 sample | 控制 Dense Mask 和激活显存 |
| 迭代 | 3 | 足够覆盖初始化、forward、backward、step |
| compile | 关闭 | 首轮不混入图编译问题 |
| EMA | 关闭 | 减少额外参数副本 |
| activation checkpointing | 关闭 | 先验证原始 backward |
| optimizer | 非 fused AdamW | 避免 CUDA-only fused optimizer |

17 个 RGB frames 经 Wan VAE 后约得到 5 个 latent frames。随机 `S=1..4` 时既能产生
多个 causal blocks，又能覆盖尾部 partial block，同时比正式长视频显著省显存。

### 11.1 Teacher forcing 展开与 Dense Mask 规模

原 Edge 配置中的 `max_num_tokens_after_packing=45056` 和
`dataloader_train.max_sequence_length=45056` 是**一维 packed sequence 的 token 数上限**。
方案 B 会再把每个视觉 token 展开为 clean/noisy 两条流；展开后的长度不再由另一个配置
截断或拒绝，而是直接根据当前 batch 的实际 layout 构造 attention mask。

设原始样本包含 `U` 个 UND token 和 `V` 个 GEN 视觉 token。方案 B 展开为
`[UND | clean GEN | noisy GEN]` 后：

```text
query 数 = 2V
key 数   = U + 2V
mask 元素数 = 2V × (U + 2V)
```

因此，如果把接近 45,056 个视觉 token 的正式 packed batch 直接送进当前 Dense 实现，
极端情况下 mask 会接近 `90,112 × 90,112 ≈ 81.2 亿` 个 boolean 元素，单是 mask
就约 7.6 GiB，尚未计算 attention 中间张量。也就是说，45,056 对一维 packing 并不小；
恰恰是 Dense Mask 无法直接承接正式长度的原因。

当前 smoke 使用 256×256、17 RGB frames：Wan VAE 约得到 5 个 latent frames，每帧
patch 后约 8×8=64 个视觉 token，所以 `V≈320`。若文字侧按约 512 token 估算，展开后
一维长度约为 `512+2×320=1,152`；对应的实际 mask 约为
`640×1,152=737,280` 个元素。Dense Mask 的实际大小由 `2V×(U+2V)` 自动确定。

另外，`PackingDataLoader` 要求 `max_sequence_length` 与 `max_samples_per_batch` 二选一。
smoke 采用“每 batch 最多 1 个样本”，因此 launcher 显式覆盖
`dataloader_train.max_sequence_length=null`，依靠短视频和单样本自然控制展开后的规模。

## 12. 正式训练配置与启动位置

正式训练入口放在 sibling `cosmos` 仓库的 cookbook 中，而不是
`cosmos-framework/examples`：

| 内容 | 对应位置 |
| --- | --- |
| 正式 TOML | [`vision_causal_edge.toml`](../../cosmos/cookbooks/cosmos3/generator/audiovisual/finetune/toml/sft_config/vision_causal_edge.toml) |
| causal 参数 | 上述 TOML 的 `[model]` block |
| FSDP/500 iterations | 上述 TOML 的 `[trainer]` block |
| packing 45,056 | 上述 TOML 的 `[dataloader_train]` block |
| 完整下载、转换和训练流程 | [`launch_sft_vision_causal_edge.sh`](../../cosmos/cookbooks/cosmos3/generator/audiovisual/finetune/launch_sft_vision_causal_edge.sh) |
| causal 模型选择 | 上述脚本 torchrun block 中的 `model=mot_causal_fsdp` |

正式启动命令：

```bash
cd ../cosmos/cookbooks/cosmos3/generator/audiovisual/finetune
bash launch_sft_vision_causal_edge.sh
```

该命令对应的执行代码块位于
`launch_sft_vision_causal_edge.sh` 的 `# 4. Train with the causal FSDP model group.` 段落。
脚本读取同目录下的 `toml/sft_config/vision_causal_edge.toml`，并通过
`model=mot_causal_fsdp` 实例化 `OmniMoTCausalModel`。

## 13. 验证步骤

### 13.1 准备路径

```bash
cd /mi/data2T/Embodied-AI/codes/cosmos_ascend/cosmos-framework

export DATASET_PATH=/path/to/sft_dataset_bridge
export BASE_CHECKPOINT_PATH=/path/to/Cosmos3-Edge-DCP
export WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth
export OUTPUT_ROOT=/path/to/causal_smoke_output
```

`DATASET_PATH` 下必须存在：

```text
train/video_dataset_file.jsonl
```

### 13.2 只验证配置，不训练

```bash
PYTHONPATH=. python -m cosmos_framework.scripts.train \
  --dryrun \
  --sft-toml=examples/toml/sft_config/vision_causal_smoke_edge.toml \
  -- \
  model=mot_causal_ddp \
  dataloader_train.max_sequence_length=null \
  '~dataloader_train.dataloader.datasets.video.dataset.conditioning_config={0:0.7,1:0.2,2:0.1}' \
  '+dataloader_train.dataloader.datasets.video.dataset.conditioning_config={0:1.0,1:0.0,2:0.0}' \
  dataloader_train.dataloader.datasets.video.dataset.num_video_frames=17
```

dryrun 应确认最终 `_target_` 是 `OmniMoTCausalModel`，并显示：

```text
causal_training_strategy: teacher_forcing
teacher_forcing_block_size_min/max: 1/4
teacher_forcing_history_blocks_min/max: 1/32
context_parallel_shard_degree: 1
```

注意：当前 Ascend 适配环境在导入 `torch_npu` 时可能初始化 NPU runtime，因此这里的
`--dryrun` 虽然不执行训练 step，仍可能要求机器能正常访问 NPU 驱动。若出现
`aclInit`/`drvGetDevNum` 错误，应先检查设备挂载和驱动环境，而不是修改 causal 配置。

### 13.3 运行三步单卡 smoke

```bash
NPROC_PER_NODE=1 bash examples/launch_sft_vision_causal_smoke_edge.sh
```

### 13.4 成功标准

满足以下条件才算功能验证通过：

1. 三个 iteration 全部完成；
2. 日志包含有限的 `flow_matching_loss_vision`；
3. 没有 NaN/Inf；
4. backward 和 optimizer step 完成；
5. 没有 Dense mask 超限、shape、dtype 或 NPU SDPA 报错；
6. NPU 显存没有持续逐 iteration 增长。

日志默认写入：

```text
$OUTPUT_ROOT/logs/vision_causal_smoke_edge_sft.log
```

## 14. 首轮失败时如何判断问题位置

| 错误 | 含义 | 下一步 |
| --- | --- | --- |
| SDPA mask/dtype/shape 错误 | NPU SDPA 分支不兼容 Dense bool mask | 记录 Q/K/V/mask shape 和 dtype，单独修 NPU backend |
| OOM | Dense attention 或模型激活过大 | 先确认 mask 元素数，再考虑开启 AC；不要同时开 compile |
| clean/noisy shape mismatch | VAE/patch packing 两条流不一致 | 检查 token shape 和 position index，不绕过校验 |
| NaN/Inf loss | BF16/backend/optimizer 数值问题 | 固定一个 batch，对照 FP32/CPU 小尺寸 reference |
| checkpoint key mismatch | 基础 DCP 与 Edge recipe 不匹配 | 核对转换来源和 `BASE_CHECKPOINT_PATH` |

## 15. 验证通过后的顺序

建议逐项增加变量，每一步都重复相同 smoke：

1. 开启 activation checkpointing；
2. 增加视频长度；
3. 尝试多卡 DDP；
4. 尝试 FSDP；
5. 最后评估 compile；
6. 测量 Dense Mask 的显存/吞吐，决定高性能无 LSE backend。

不要一次同时打开多卡、AC 和 compile，否则失败时无法判断是哪一层造成的。
