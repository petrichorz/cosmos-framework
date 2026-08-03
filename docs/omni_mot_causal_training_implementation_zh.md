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

### 6.1 Causal 模型入口

`OmniMoTCausalModel` 继承原来的 `OmniMoTModel`。它不复制整套训练流程，只覆盖
`post_noise_packing_hook`：普通路径完成加噪后，再插入 clean stream。这样原有 VAE、
噪声调度、decoder 和 loss 都继续复用。

代码：

- `cosmos_framework/model/generator/omni_mot_causal_model.py`
- `cosmos_framework/model/generator/causal_teacher_forcing.py`

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

代码：`cosmos_framework/data/generator/sequence_packing/teacher_forcing.py`。

### 6.3 时间步编码

- clean stream 使用真实 `x0`，timestep 固定为 0；
- noisy stream 使用 `xt`，保留原 flow-matching timestep；
- clean hidden states 不直接计算生成 loss，但允许梯度通过合法 attention 路径回传。

代码：`cosmos_framework/model/generator/mot/cosmos3_vfm_network.py`。

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

代价是 Dense Mask 的显存随 query×key 二次增长。因此配置必须显式设置
`teacher_forcing_max_mask_elements`，超过上限时提前报错。真实尺寸性能优化属于后续
工作，不在当前功能闭环中。

代码：

- `cosmos_framework/model/generator/mot/teacher_forcing_attention.py`
- `cosmos_framework/model/generator/mot/attention.py`

### 6.5 输出和 loss

网络会产生 clean/noisy 两部分 hidden output，但 decoder 只恢复 noisy stream，并沿用
原有 flow-matching target、condition mask 和 loss 计算。clean stream 不单独增加 loss。

代码：

- `select_teacher_forcing_noisy_outputs()`；
- `cosmos_framework/model/generator/omni_mot_model.py` 的原 loss 路径。

## 7. 为什么采用这种方案

我们比较过两类方向：多次 attention 后用 LSE 合并，以及单次 Dense Mask attention。
当前选择后者，原因不是它最终性能最好，而是它最适合作为正确性基线：

- Scheme B 的 clean/noisy 可见性可以在一张 mask 中完整表达；
- 不依赖尚未验证的 NPU LSE 输出；
- 出错时可以逐元素检查 mask；
- CPU 显式 attention 能提供独立数值参考；
- 等语义稳定后，可以用更快的 Ascend/block-sparse 实现替换 backend，而不改变上层
  packing 和训练定义。

## 8. 已完成的代码模块

1. `TeacherForcingLayout`：描述 sample、stream、block 和输出索引。
2. 随机参数采样：batch-shared `S=1..4`、`K=1..32`。
3. Dense mask oracle：固定可见性和防泄漏规则。
4. PackedSequence 双流扩展：支持不同视频长度的 packed batch。
5. clean/noisy 相同 RoPE position IDs。
6. clean timestep=0、noisy timestep=t。
7. 单 softmax Dense SDPA，支持显式 GQA KV head 扩展。
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

定向 CPU 回归结果为 `70 passed`。这证明网络语义闭环，不代表真实 Ascend 大尺寸训练
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
| Dense mask 上限 | 4,194,304 elements | 超过约 4 MiB bool mask 时提前失败 |

17 个 RGB frames 经 Wan VAE 后约得到 5 个 latent frames。随机 `S=1..4` 时既能产生
多个 causal blocks，又能覆盖尾部 partial block，同时比正式长视频显著省显存。

## 12. 验证步骤

### 12.1 准备路径

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

### 12.2 只验证配置，不训练

```bash
PYTHONPATH=. python -m cosmos_framework.scripts.train \
  --dryrun \
  --sft-toml=examples/toml/sft_config/vision_causal_smoke_edge.toml \
  -- \
  model=mot_causal_ddp \
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

### 12.3 运行三步单卡 smoke

```bash
NPROC_PER_NODE=1 bash examples/launch_sft_vision_causal_smoke_edge.sh
```

### 12.4 成功标准

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

## 13. 首轮失败时如何判断问题位置

| 错误 | 含义 | 下一步 |
| --- | --- | --- |
| `teacher_forcing_max_mask_elements` | 样本仍太长 | 先减视频帧/文字长度，不直接无限增大上限 |
| SDPA mask/dtype/shape 错误 | NPU SDPA 分支不兼容 Dense bool mask | 记录 Q/K/V/mask shape 和 dtype，单独修 NPU backend |
| OOM | Dense attention 或模型激活过大 | 先确认 mask 元素数，再考虑开启 AC；不要同时开 compile |
| clean/noisy shape mismatch | VAE/patch packing 两条流不一致 | 检查 token shape 和 position index，不绕过校验 |
| NaN/Inf loss | BF16/backend/optimizer 数值问题 | 固定一个 batch，对照 FP32/CPU 小尺寸 reference |
| checkpoint key mismatch | 基础 DCP 与 Edge recipe 不匹配 | 核对转换来源和 `BASE_CHECKPOINT_PATH` |

## 14. 验证通过后的顺序

建议逐项增加变量，每一步都重复相同 smoke：

1. 开启 activation checkpointing；
2. 增加视频长度；
3. 尝试多卡 DDP；
4. 尝试 FSDP；
5. 最后评估 compile；
6. 测量 Dense Mask 的显存/吞吐，决定高性能无 LSE backend。

不要一次同时打开多卡、AC 和 compile，否则失败时无法判断是哪一层造成的。
