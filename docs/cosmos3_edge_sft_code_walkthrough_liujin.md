# Cosmos3-Edge SFT 代码串讲

> 本文档基于 `launch_sft_vision_edge_yundao.sh`（已验证可正常拉起 Edge SFT 训练），
> 对 Cosmos3 的 SFT（Supervised Fine-Tuning）代码流程进行完整串讲。


---

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [调用链路](#2-调用链路)
3. [第一层：启动脚本](#3-第一层启动脚本)
   - [3.1 结构划分](#31-结构划分)
   - [3.2 关键语法说明](#32-关键语法说明)
4. [第二层：共用启动逻辑](#4-第二层共用启动逻辑)
   - [4.1 核心逻辑](#41-核心逻辑)
   - [4.2 最终拼出的训练命令](#42-最终拼出的训练命令以-yundao-脚本为例)
5. [第三层：TOML 配置](#5-第三层toml-配置)
   - [5.1 TOML 结构](#51-toml-结构)
   - [5.2 关键配置解读](#52-关键配置解读)
   - [5.3 TOML → Hydra Config 的转换链](#53-toml--hydra-config-的转换链)
6. [第四层：训练入口 train.py](#6-第四层训练入口-trainpy)
   - [6.1 NPU 适配](#61-npu-适配第-30-33-行)
   - [6.2 主流程 launch()](#62-主流程-launch第-186-229-行)
   - [6.3 配置加载](#63-配置加载第-286-行)
7. [第五层：模型定义](#7-第五层模型定义)
   - [7.1 实验层 vision_sft_edge.py](#71-实验层-vision_sft_edgepy)
   - [7.2 模型骨干 edge_model_config.py](#72-模型骨干-edge_model_configpy)
   - [7.3 OmniMoT 模型 omni_mot_model.py](#73-omnimot-模型-omni_mot_modelpy)
   - [7.4 MoT 网络架构 cosmos3_vfm_network.py](#74-mot-网络架构-cosmos3_vfm_networkpy)
8. [第六层：数据加载](#8-第六层数据加载)
   - [8.1 数据格式（JSONL）](#81-数据格式jsonl)
   - [8.2 数据加载器 sft_dataset.py](#82-数据加载器-sft_datasetpy)
   - [8.3 一句话总结](#83-一句话总结)
9. [第七层：训练循环](#9-第七层训练循环)
   - [9.1 训练步](#91-训练步)
   - [9.2 关键回调](#92-关键回调vision_sft_edgepy-中注册的)
   - [9.3 可训练参数](#93-可训练参数)
   - [9.4 损失计算全景](#94-损失计算全景)
   - [9.5 流匹配损失](#95-流匹配损失-compute_flow_matching_loss)
   - [9.6 速度目标与插值 RectifiedFlow](#96-速度目标与插值-rectifiedflow)
   - [9.7 时间加权 TrainTimeWeight](#97-时间加权-traintimeweight)
   - [9.8 MoE 负载均衡损失](#98-moe-负载均衡损失-compute_load_balancing_loss)
10. [第八层：NPU 适配](#10-第八层npu-适配)
   - [10.1 设备后端自动检测](#101-设备后端自动检测)
   - [10.2 CUDA → NPU 重定向](#102-cuda--npu-重定向)
   - [10.3 注意力后端](#103-注意力后端)
   - [10.4 FusedAdam on NPU](#104-fusedadam-on-npu)
   - [10.5 全局设备标志](#105-全局设备标志)
11. [如何修改脚本跑起来](#11-如何修改脚本跑起来)
   - [11.1 必需的环境变量](#111-必需的环境变量)
   - [11.2 必需的特殊环境变量](#112-必需的特殊环境变量)
   - [11.3 分布式拓扑（单卡示例）](#113-分布式拓扑单卡示例)
   - [11.4 TOML 需要改的项](#114-toml-需要改的项)
   - [11.5 硬件要求](#115-硬件要求)
   - [11.6 运行步骤](#116-运行步骤)
   - [11.7 常见错误](#117-常见错误)
12. [训练产物](#12-训练产物)
   - [DCP → safetensors 导出](#dcp--safetensors-导出)

---

## 1. 整体架构概览

Cosmos3-Edge 的 SFT 训练是一个 **8 层调用链**：

```
launch_sft_vision_edge_yundao.sh        ← 用户入口（设置路径、环境变量）
    │
    ├── source _sft_launcher_common.sh  ← 共用启动逻辑
    │       │
    │       └── torchrun -m cosmos_framework.scripts.train  ← Python 入口
    │               │
    │               ├── vision_sft_edge.toml                 ← 配置文件
    │               ├── sft_config.py                        ← TOML → Hydra 转换
    │               ├── vision_sft_edge.py                   ← 实验定义（模型/数据加载器）
    │               ├── edge_model_config.py                 ← 模型骨干定义
    │               └── train.py                             ← 训练循环
    │
    └── 环境准备（conda、CANN、torchcodec、HuggingFace 缓存）
```

**核心思想**：所有训练超参收敛到一份 TOML 文件中，Python 入口只有 `--sft-toml` 一个参数，杜绝"改错配置"的问题。

---

## 2. 调用链路

下面是一个完整的训练启动过程中，各文件的执行顺序和数据流向：

```
bash launch_sft_vision_edge_yundao.sh
│
├─ [1] 设置环境变量 (HF_HUB_OFFLINE, COSMOS_DEVICE, DATASET_PATH, ...)
├─ [2] 激活 conda 环境 (cosmos-framework)
├─ [3] pip install -e . (安装 cosmos_framework 包)
├─ [4] 设置 torchcodec 编译环境
├─ [5] 指定 TOML_FILE + TAIL_OVERRIDES
│
├─ [6] source _sft_launcher_common.sh
│   ├─ 路径解析（相对→绝对，REPO_ROOT 锚定）
│   ├─ 输入校验（TOML、数据集、DCP、VAE 是否存在）
│   ├─ 拼接 torchrun 命令：
│   │   torchrun --nproc_per_node=1 --master_port=50012 \
│   │       -m cosmos_framework.scripts.train \
│   │       --sft-toml=examples/toml/sft_config/vision_sft_edge.toml \
│   │       -- \
│   │       model.config.vlm_config.tokenizer.repository=null \
│   │       model.config.vlm_config.tokenizer.revision=null \
│   │       +model.config.vlm_config.tokenizer.tokenizer_type=/path/to/Cosmos3-Edge
│   └─ 执行
│
└─ [7] cosmos_framework/scripts/train.py
    ├─ import torch_npu (Ascend 适配)
    ├─ load_experiment_from_toml(TOML) → Hydra Config
    │   ├─ TOML → pydantic 校验 (SFTExperimentConfig)
    │   ├─ build_hydra_overrides() → Hydra 风格 key=value 列表
    │   ├─ compose() 合并 experiment=vision_sft_edge 的 LazyDict
    │   └─ 返回合并后的 Config
    ├─ config.validate() + config.freeze()
    ├─ instantiate(trainer)     → ImaginaireTrainer (FSDP)
    ├─ instantiate(model)       → OmniMoTModel (Nemotron-2B-Dense-VL)
    ├─ instantiate(dataloader)  → PackingDataLoader (BridgeData2)
    └─ trainer.train(model, dataloader_train, dataloader_val)
```

---

## 3. 第一层：启动脚本

**文件**：`examples/launch_sft_vision_edge_yundao.sh`（88 行）

### 3.1 结构划分

| 行号 | 功能 | 说明 |
|------|------|------|
| 1-2 | Shebang + 安全选项 | `set -euo pipefail`：任一步失败立即退出 |
| 5-6 | 运行环境 | `HF_HUB_OFFLINE=1` 禁用 HuggingFace 联网；`COSMOS_DEVICE=npu` 强制用 NPU |
| 9-13 | 路径变量 | 数据、权重、processor、VAE、输出目录 |
| 16-19 | HF 缓存 | 软链接 `/mi/.../huggingface` → `~/.cache/huggingface` |
| 22-27 | 分布式拓扑 | 单机单卡：`NPROC_PER_NODE=1` |
| 37-39 | conda 环境 | 激活 `cosmos-framework` |
| 42-43 | 安装包 | `pip install -e .` 在 develop 模式安装框架 |
| 48-53 | torchcodec | 视频编解码库的编译环境 |
| 76-77 | TOML 路径 + 默认值 | `: "${VAR:=default}"` 语法：如果环境变量已经 export 了就用环境变量，否则用默认值 |
| 80-85 | TAIL_OVERRIDES | 追加的 Hydra 覆盖参数（processor 本地路径） |
| 87 | 启动 | source 共用脚本 |

### 3.2 关键语法说明

```bash
: "${DATASET_PATH:=examples/data/...}"   # := 表示"如果未设或用空，则赋默认值"
```

这意味着：
- 如果你已经 `export DATASET_PATH=/my/path`，脚本用你的路径
- 如果没设，脚本用默认的相对路径

```bash
TAIL_OVERRIDES=(
    "model.config.vlm_config.tokenizer.repository=null"
    "model.config.vlm_config.tokenizer.revision=null"
    "+model.config.vlm_config.tokenizer.tokenizer_type=$COSMOS3_EDGE_PROCESSOR_PATH"
)
```

这是 Hydra 风格覆盖，`--` 后面传到 `train.py` 的 `opts` 参数。加号 `+` 表示追加新的配置字段（schema 里不存在的键，不加 `+` 会报错）。

---

## 4. 第二层：共用启动逻辑

**文件**：`examples/_sft_launcher_common.sh`（107 行）

### 4.1 核心逻辑

```bash
# 1) 确定仓库根目录
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[1]}")/.." && pwd)"

# 2) 相对路径 → 绝对路径
[[ "$TOML_FILE" = /* ]] || TOML_FILE="$WORKDIR/$TOML_FILE"

# 3) 输入校验
[[ -f "$TOML_FILE" ]] || exit 1
[[ -d "$DATASET_PATH" ]] || exit 1
[[ -d "$BASE_CHECKPOINT_PATH" ]] || exit 1
[[ -f "$WAN_VAE_PATH" ]] || exit 1

# 4) 拼 TAIL_OVERRIDES（非空时才加 -- 分隔符）
if (( ${#TAIL_OVERRIDES[@]} > 0 )); then
    TRAILING_ARGS=(-- "${TAIL_OVERRIDES[@]}")
fi

# 5) torchrun 拓扑配置
TORCHRUN_ARGS=(--nproc_per_node="${NPROC_PER_NODE:-8}" --master_port="${MASTER_PORT:-50012}")

# 6) 启动
IMAGINAIRE_OUTPUT_ROOT="$IMAGINAIRE_OUTPUT_ROOT" PYTHONPATH=. \
    torchrun "${TORCHRUN_ARGS[@]}" -m cosmos_framework.scripts.train \
    --sft-toml="$TOML_FILE" \
    "${TRAILING_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"
```

### 4.2 最终拼出的训练命令（以 yundao 脚本为例）

```bash
IMAGINAIRE_OUTPUT_ROOT=/mi/.../cosmos_trainging_logs PYTHONPATH=. \
torchrun --nproc_per_node=1 --master_port=50012 \
    -m cosmos_framework.scripts.train \
    --sft-toml=/mi/.../examples/toml/sft_config/vision_sft_edge.toml \
    -- \
    model.config.vlm_config.tokenizer.repository=null \
    model.config.vlm_config.tokenizer.revision=null \
    +model.config.vlm_config.tokenizer.tokenizer_type=/mi/.../Cosmos3-Edge
```

---

## 5. 第三层：TOML 配置

**文件**：`examples/toml/sft_config/vision_sft_edge.toml`（93 行）

### 5.1 TOML 结构

TOML 被 `sft_config.py` 的 pydantic 模型严格校验（`extra="forbid"`），写错任何一个 key 名训练都不会启动。

```toml
[job]         # 任务元信息
[model]       # 模型超参
[optimizer]   # 优化器
[scheduler]   # 学习率调度
[trainer]     # 训练器
[checkpoint]  # 检查点
[dataloader_train]  # 数据加载
```

### 5.2 关键配置解读

| 配置路径 | 值 | 含义 |
|----------|-----|------|
| `[job].task` | `"vfm"` | Video Foundation Model（生成器）模式 |
| `[job].experiment` | `"vision_sft_edge"` | 实验名，对应 `vision_sft_edge.py` |
| `[model].precision` | `"bfloat16"` | BF16 混合精度训练 |
| `[model].compile.enabled` | `false` | 关 torch.compile（NPU 不兼容） |
| `[model].parallelism.data_parallel_shard_degree` | `-1` | FSDP 自动适配（单卡退化为 no-op） |
| `[model].activation_checkpointing.mode` | `"full"` | 重计算换显存 |
| `[model].ema` | `enabled=true` | 指数滑动平均（EMA），推理时用 EMA 权重 |
| `[optimizer].fused` | `true` | 使用 FusedAdam（TE 版，NPU 兼容） |
| `[optimizer].keys_to_select` | 5 个子串 | 只训练这 5 类参数，其余冻结 |
| `[optimizer].lr` | `1e-4` | 学习率 |
| `[scheduler].warm_up_steps` | `[50]` | 前 50 步线性 warmup |
| `[trainer].max_iter` | `500` | 总训练步数 |
| `[trainer].grad_accum_iter` | `2` | 梯度累积 2 步 |
| `[checkpoint].save_iter` | `100` | 每 100 步存检查点 |
| `[checkpoint].load_path` | `${oc.env:BASE_CHECKPOINT_PATH}` | 从环境变量读入 |
| `[dataloader_train].max_sequence_length` | `45056` | 序列 token 上限 |

### 5.3 TOML → Hydra Config 的转换链

```
vision_sft_edge.toml
    │
    ▼
SFTExperimentConfig.model_validate(raw)
    │  pydantic 校验：多一个字段就报错
    ▼
build_hydra_overrides(raw)
    │  把 TOML 的嵌套结构展平为 key=value 列表
    │  例如 [model].precision="bfloat16" → model.precision=bfloat16
    ▼
load_config(base_config_path, overrides)
    │  compose() 合并 experiment Python + 默认值 + overrides
    ▼
Config (Hydra OmegaConf)
    │  config.validate() + config.freeze()
    ▼
instantiate(model/trainer/dataloader)
```

---

## 6. 第四层：训练入口 train.py

**文件**：`cosmos_framework/scripts/train.py`（304 行）

### 6.1 NPU 适配（第 30-33 行）

```python
import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except:
    print("Use GPU or CPU Platform")
```

`transfer_to_npu` 会把所有 `torch.cuda.*` 调用自动重定向到 `torch.npu.*`，实现 CUDA→NPU 的无感切换。

### 6.2 主流程 `launch()`（第 186-229 行）

```python
def launch(config, args):
    with distributed_init():
        distributed.init()            # 分布式初始化（hccl for NPU）

    config.validate()
    config.freeze()                   # 冻结，防止后续意外修改

    trainer = config.trainer.type(config)       # 实例化 ImaginaireTrainer
    model   = instantiate(config.model)         # 实例化模型
    dataloader = instantiate(config.dataloader_train)

    trainer.train(model, dataloader_train, dataloader_val)
```

### 6.3 配置加载（第 286 行）

```python
config = load_experiment_from_toml(args.sft_toml, extra_overrides=args.opts)
```

`args.opts` 就是 `--` 后面 TAIL_OVERRIDES 的内容，优先级高于 TOML。

---

## 7. 第五层：模型定义

### 7.1 实验层 `vision_sft_edge.py`

**文件**：`cosmos_framework/configs/base/experiment/sft/vision_sft_edge.py`（298 行）

定义了 `vision_sft_edge` 的完整 LazyDict：模型、优化器、调度器、数据加载器、回调函数。

关键数据加载器定义（第 229-286 行）：

```python
dataloader_train=L(PackingDataLoader)(
    dataloader=L(RankPartitionedDataLoader)(
        batch_size=1,
        num_workers=4,
        datasets=dict(
            video=dict(
                ratio=1,
                dataset=L(get_sft_dataset)(
                    # 70% T2V, 20% I2V (first frame), 10% V2V (first 5 frames)
                    conditioning_config={0: 0.7, 1: 0.2, 2: 0.1},
                    jsonl_paths=["${oc.env:DATASET_PATH}/train/video_dataset_file.jsonl"],
                    resolution="256",
                    ...
                ),
            ),
        ),
    ),
)
```

**重要**：同一个 JSONL 数据集，训练时按 70/20/10 概率随机投喂 T2V、I2V、V2V 三模式。

---

### 7.2 模型骨干：三类模型的配置差异

**文件**：`cosmos_framework/configs/base/experiment/sft/models/` 下的三个配置

| 模型 | 文件 | 行数 | 语言骨干 |
|------|------|------|---------|
| Edge | `edge_model_config.py` | 176 | Nemotron-2B-Dense-VL |
| Nano | `nano_model_config.py` | 147 | Qwen3-VL-8B |
| Super | `super_model_config.py` | 164 | Qwen3-VL-32B |

三者都实例化同一个 `OmniMoTModel`（7.3 节）和 `Cosmos3VFMNetwork`（7.4 节），
差异全部体现在各自 `*_MODEL_CONFIG` 字典里的字段值上。

> **关键背景**：Edge 的注释（文件头第 12-23 行）明确说明，`EDGE_MODEL_CONFIG` 是**从 `NANO_MODEL_CONFIG` 派生**的——
> 除了若干 "Edge delta" 外，其余字段与 Nano 完全相同。而 Super 则是 Nano 的 32B 升级版 + LoRA 微调。
> 所以下面以「与 Nano 的差异」为主线展开。

#### 7.2.1 语言骨干（`vlm_config`）——最根本的差异

| 字段 | Edge | Nano | Super |
|------|------|------|-------|
| `model_instance` | `Nemotron3DenseVLTextForCausalLM` | `Qwen3VLTextForCausalLM` | `Qwen3VLTextForCausalLM` |
| 配置类 | `Nemotron3DenseVLMoTConfig` | `Qwen3VLMoTConfig` | `Qwen3VLMoTConfig` |
| 架构 JSON | `Nemotron-2B-Dense-VL.json` | `Qwen3-VL-8B-Instruct.json` | `Qwen3-VL-32B-Instruct.json` |
| `model_name` | `nvidia/Cosmos3-Edge-Reasoner` | `Qwen/Qwen3-VL-8B-Instruct` | `Qwen/Qwen3-VL-32B-Instruct` |
| 参数量 | 2B | 8B | 32B |
| `layer_module`（顶层） | `None` | `"Qwen2MoTDecoderLayer"` | `"Qwen2MoTDecoderLayer"` |
| `qk_norm_for_text` | `False` | `True` | `True` |
| `use_und_k_norm_for_gen` | `True` | 未设（默认 False） | 未设（默认 False） |
| `include_visual` | `None`（无视觉塔） | 默认（有视觉塔） | 默认（有视觉塔） |
| tokenizer 构造 | `build_processor_lazy(repository="nvidia/Cosmos3-Edge")` | `create_qwen2_tokenizer_with_download(config_variant="hf", ...)` | 同 Nano（32B） |

**两个值得注意的差异**：

1. **QK norm 的差别**（Edge 独有）：Nano/Super 用 `qk_norm_for_text=True`（理解塔也做 QK 归一化），
   而 Edge 用 `qk_norm_for_text=False` + `use_und_k_norm_for_gen=True`。
   这对应 7.4.3 里 `PackedAttentionMoT` 的 `k_norm_und_for_gen` 分支——
   因为 Nemotron 的生成塔有 QK norm 但理解塔没有，需要在「gen→und 交叉注意力」路径上
   额外对 und 的 K 做一次 RMSNorm（`k_norm_und_for_gen`），否则 K 的尺度会失控、主导注意力。
   这也正是 SFT 可训练参数 `keys_to_select` 里那个 `"k_norm_und_for_gen"` 的来源。

2. **`include_visual=None`**：Edge 没有独立的视觉塔（vision tower），图像/视频直接走 VAE 编码；
   Nano/Super（Qwen3-VL）则带 VL 视觉编码器。这就是之前导出权重时 `--no-vit` 对 Edge 生效的原因。

#### 7.2.2 分辨率与 loss scale

| 字段 | Edge | Nano | Super |
|------|------|------|-------|
| `resolution` | `"480"` | `"720"` | `"720"` |
| `loss_scale` | `10.0` | `1.0` | `1.0` |
| `image_loss_scale` | `None` | `1.0` | `1.0` |
| `sound_loss_scale` | `2.0`（Edge 独有键） | 无 | 无 |

- Edge 原生推理分辨率是 480p，所以 `resolution="480"`；Nano/Super 是 720p。
- **`loss_scale=10.0`** 是 Edge 独有的「delta」（文件头注释明确标注 `1.0 -> 10.0`），
  这就是 7.3.10 里视觉损失 `×10.0` 的来源。
- `sound_loss_scale=2.0` 是 Edge 独有的配置键，但因为 `sound_gen=False` 实际不生效。

#### 7.2.3 动作生成（action）配置

| 字段 | Edge | Nano | Super |
|------|------|------|-------|
| `action_gen` | `True` | `True` | `False` |
| `max_action_dim` | `64` | `64` | `32` |

- Edge/Nano 的 baseline 保持 `action_gen=True`（对应发布 checkpoint 里带动作头权重），
  但**视觉 SFT 实验**（`vision_sft_edge.py`/`vision_sft_nano.py`）会再覆盖为 `False`（不训动作数据）。
- Super 直接 `action_gen=False`，且 `max_action_dim=32`（动作维度更小）。

#### 7.2.4 微调方式：LoRA vs 全量

| 字段 | Edge | Nano | Super |
|------|------|------|-------|
| `lora_enabled` | 未设（False） | 未设（False） | `True` |
| `lora_rank` | — | — | `16` |
| `lora_alpha` | — | — | `32` |
| `lora_target_modules` | — | — | `"q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen"` |

- **Edge / Nano：全量微调**（`keys_to_select` 那 5 类参数全部真实更新）。
- **Super：LoRA 微调**，只对生成塔注意力的 4 个投影（`*_moe_gen`）加低秩适配器，
  底模 32B 冻结。这是 32B 模型显存不够全量微调时的标准做法。

#### 7.2.5 并行拓扑与编译

| 字段 | Edge | Nano | Super |
|------|------|------|-------|
| `data_parallel_shard_degree` | `8` | `8` | `4` |
| `context_parallel_shard_degree` | `1` | `1` | `2` |
| `compile.enabled` | `True` | `True` | `False` |
| `ema.enabled` | `True` | `True` | `False` |

- **并行拓扑**：Edge/Nano 用纯数据并行（DP=8）；Super 用 DP=4 + CP=2（上下文并行），
  因为 32B 单卡放不下，需要切分序列长度。
- **`compile`**：Edge/Nano 默认开 `torch.compile`（但 NPU 上你已在 TOML 里关掉）；Super 直接关。
- **EMA**：Edge/Nano 开 EMA（指数滑动平均，推理用 EMA 权重）；Super 关（LoRA 微调不追 EMA）。

#### 7.2.6 VAE 路径

| 字段 | Edge | Nano | Super |
|------|------|------|-------|
| `vae_path` | `/mi/data2T/Embodied-AI/ckpts/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth` | `pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth` | 同 Nano |

Edge 用的是**本地绝对路径**（你的机器上）；Nano/Super 用相对路径（`pretrained/...`，部署环境里解析）。
三者都是同一个 **Wan2.2 VAE**，只是路径写法不同。

#### 7.2.7 差异总表

| 维度 | Edge (2B) | Nano (8B) | Super (32B) |
|------|-----------|-----------|-------------|
| 语言骨干 | Nemotron-2B-Dense-VL | Qwen3-VL-8B | Qwen3-VL-32B |
| 分辨率 | 480p | 720p | 720p |
| 视觉塔 | 无（`include_visual=None`） | 有 | 有 |
| QK norm（文本塔） | False + und-K norm | True | True |
| loss_scale | 10.0 | 1.0 | 1.0 |
| 动作生成 | True（SFT 覆盖为 False） | True（SFT 覆盖为 False） | False |
| max_action_dim | 64 | 64 | 32 |
| 微调方式 | 全量 | 全量 | LoRA（rank=16） |
| 并行拓扑 | DP=8 | DP=8 | DP=4 + CP=2 |
| torch.compile | 开（NPU 关） | 开（NPU 关） | 关 |
| EMA | 开 | 开 | 关 |

**一句话**：三类模型「骨架相同、骨干不同、规模不同、微调策略不同」——
Edge 是 Nemotron 2B 全量微调（480p、loss×10），Nano 是 Qwen 8B 全量微调（720p），
Super 是 Qwen 32B 的 LoRA 微调（720p、DP+CP 混合并行、关 EMA/compile）。

---

### 7.3 OmniMoT 模型 `omni_mot_model.py`

**文件**：`cosmos_framework/model/generator/omni_mot_model.py`（共 5199 行）

这是整个训练的核心模型对象。Edge / Nano / Super 三类模型**共用同一个类** `OmniMoTModel`，
差异只在内部的语言骨干（见 7.2）。

#### 7.3.1 类定义（第 82 行）

```python
class OmniMoTModel(ImaginaireModel):
    """
    Mixture of Transformers (MoT) model to be trained with the flow matching objective
    for visual / sound / action generation.
    """
```

要点：
- 继承 `ImaginaireModel`（框架基类，提供 `state_dict`/`load_state_dict`/EMA 等通用能力）。
- 名字里的 **MoT = Mixture of Transformers**：语言理解塔 + 扩散生成塔由同一套 Transformer 骨干共享，
  其中生成塔是 MoE（Mixture-of-Experts，稀疏专家）结构。
- 训练目标是 **Flow Matching（流匹配，即 Rectified Flow）**，同时支持视觉 / 声音 / 动作三种生成模态。

#### 7.3.2 初始化五步（`__init__`，第 88 行）

```python
def __init__(self, config: OmniMoTModelConfig):
    super().__init__()
    self.config = config
    self.set_precision()              # 0. 精度
    self.set_up_data_key()            # 1. 数据 key
    self.set_up_tokenizers()          # 2. 三个 tokenizer
    self.set_up_parallelism()         # 3. FSDP 并行（必须先于建模型）
    self.set_up_model()               # 4. 构建去噪网络
    self.set_up_scheduler_and_sampler()  # 5. 训练 scheduler + 推理 sampler
```

顺序很重要：**`set_up_parallelism()` 必须在 `set_up_model()` 之前**（源码注释明确强调），
因为建网络时要根据并行拓扑决定 FSDP 分片方式。

| 步骤 | 方法 | 做什么 |
|------|------|--------|
| 0 | `set_precision()` | `self.precision = getattr(torch, "bfloat16")`，准备 `tensor_kwargs`，关掉 TF32 |
| 1 | `set_up_data_key()` | 记录 `input_video_key`/`input_image_key`/`input_caption_key` 字段名 |
| 2 | `set_up_tokenizers()` | 实例化文本/视觉/声音三个 tokenizer |
| 3 | `set_up_parallelism()` | 构建 `ParallelDims`（FSDP/CFG/CP 三维并行网格） |
| 4 | `set_up_model()` | 调 `build_net()` 建网络 + 建 EMA 副本 |
| 5 | `set_up_scheduler_and_sampler()` | 建 RectifiedFlow（训练用）+ UniPC/EDM 采样器（推理用） |

#### 7.3.3 三个 tokenizer（`set_up_tokenizers`，第 128 行）

```python
# 1. 文本 tokenizer（语言骨干自带的 processor，Nemotron/Qwen）
self.vlm_processor = lazy_instantiate(self.vlm_config.tokenizer)
vlm_tokenizer, special_tokens = add_special_tokens(vlm_tokenizer)

# 2. 视觉 tokenizer（Wan2.2 VAE，像素 ↔ latent）
self.tokenizer_vision_gen = lazy_instantiate(self.config.tokenizer)

# 3. 声音 tokenizer（可选，Edge 没有 sound_gen，此处为 None）
if self.config.sound_gen:
    self.tokenizer_sound_gen = lazy_instantiate(self.config.sound_tokenizer)
```

- **文本**：来自 `vlm_config.tokenizer`，即 7.2 里说的本地 `Cosmos3-Edge` processor；
  `add_special_tokens` 会追加 Cosmos3 的 `<|vision_start|>`/`<|vision_end|>` 等特殊 token。
- **视觉**：`config.tokenizer` 就是 `vae_path` 指向的 **Wan2.2 VAE**，负责把视频帧压缩成 latent token，
  这是「像素 → latent」的编码器，是扩散模型工作的空间。
- **声音**：Edge 的 `sound_gen=False`，所以 `tokenizer_sound_gen=None`，跳过。

#### 7.3.4 构建网络 `build_net()`（第 192 行）

这是把「语言骨干 + 扩散专家」拼装成完整 MoT 网络的地方，核心流程：

```python
with torch.device("meta"):                       # 先在 meta 设备上搭骨架（不占显存）
    language_model = lazy_instantiate(self.vlm_config.model_instance)  # Nemotron / Qwen
    network_config = Cosmos3VFMNetworkConfig(vlm_config=language_model.config, ...)
    net = Cosmos3VFMNetwork(language_model=language_model, config=network_config)

    if lora_enabled:                              # Super 用 LoRA，Edge/Nano 全量
        net = self.add_lora(net, ...)

net = parallelize_vfm_network(net, parallel_dims=self.parallel_dims, ...)  # FSDP/TP/CP 切分

net = net.to(dtype=dtype)                         # bf16
net.to_empty(device=DEVICE)                       # meta → 真实设备
net.init_weights(buffer_device=DEVICE)            # 初始化权重
```

要点：
- **meta 设备**：先在 `torch.device("meta")` 上建结构（不分配显存），拿到完整网络后一次性 `to_empty` 搬到 NPU，
  避免大模型在显存里反复 copy。
- `language_model` 就是 7.2 的 `Nemotron3DenseVLTextForCausalLM`（Edge）或 `Qwen3VLTextForCausalLM`（Nano/Super）。
- `Cosmos3VFMNetwork` 是真正的 MoT 网络（语言塔 + 生成塔 + 桥接层），`omni_mot_model.py` 只是它的「外壳」，
  负责 tokenizer、噪声调度、loss 计算，网络前向本身在 `mot/cosmos3_vfm_network.py` 里。
- `parallelize_vfm_network` 根据 `parallel_dims` 做 FSDP 分片 + 激活重计算（`ac_config`）+ 注意力 IO 布局。

#### 7.3.5 并行拓扑 `set_up_parallelism()`（第 413 行）

```python
self.parallel_dims = ParallelDims(
    world_size=dist.get_world_size(),                    # 单卡 = 1
    dp_shard=config.parallelism.data_parallel_shard_degree,   # -1 = 自动
    cfgp=config.parallelism.cfg_parallel_shard_degree,        # CFG 并行
    cp=config.parallelism.context_parallel_shard_degree,      # 上下文并行
)
self.parallel_dims.build_meshes(device_type=DEVICE_TYPE)     # 建通信 mesh（NPU 用 hccl）
```

单卡场景下 `world_size=1`，`dp_shard=-1` 自动退化为 **no-op**（不做 FSDP 分片），
这就是为什么 Edge 单卡能跑起来——并行层在单卡下全部退化为直通。

#### 7.3.6 训练/推理的流匹配 `set_up_scheduler_and_sampler()`（第 424 行）

这一步建立流匹配（Rectified Flow）的核心对象：

```python
# 1) 取 shift：int 直接用；dict 则按分辨率查表（Edge: {256:3, 480:5, 720:10}）
shift = shift_dict[resolution] if isinstance(shift_config, dict) else shift_config

# 2) 训练用 RectifiedFlow（按模态各建一个，velocity_field 都指向 self.net）
self.rectified_flow_image = RectifiedFlow(velocity_field=self.net, shift=shift, ...)
self.rectified_flow_video = RectifiedFlow(velocity_field=self.net, shift=shift, ...)
self.rectified_flow_action = RectifiedFlow(...)   # action_gen=True 时
self.rectified_flow_sound  = RectifiedFlow(...)   # sound_gen=True 时

# 3) 推理用采样器（求解器）
self.sampler = UniPCSampler(cfg=...)  # 或 EDMSampler
self.fixed_step_sampler = FixedStepSampler(...)  # 蒸馏模型专用，基座模型为 None
```

- **RectifiedFlow**：训练时负责「采样噪声时间 t」和「计算插值 xt、速度目标 vt」（详见 7.3.8）。
- **shift**：之前讲过的流匹配时间步重映射参数，Edge 按分辨率查表（480p → shift=5）。
- **sampler**：推理时把噪声逐步去噪成 latent 的 ODE 求解器，训练阶段不参与，但提前建好。

#### 7.3.7 训练主流程 `training_step()`（第 802 行）

这是训练时**每一步**执行的完整流水线，是整个文件最重要的方法。流程如下：

```
① 加载文本 → tokenize                _load_and_tokenize_text_data
② 构建 sequence plans（含条件信息）    build_sequence_plans_from_data_batch
③ 编码视觉/动作/声音 → latent         get_data_and_condition
④ 采样噪声时间 t（含 shift）           _get_train_noise_level_vision
⑤ 打包成一条序列                     _pack_input_sequence
⑥ 加噪：x0 → xt，算速度目标 vt        _add_noise_to_input
⑦ 网络前向，预测速度 v                denoise
⑧ 算 loss                            _compute_losses
```

对应源码（精简）：

```python
def training_step(self, data_batch, iteration):
    input_text_indexes = self._load_and_tokenize_text_data(data_batch, iteration)
    sequence_plans = build_sequence_plans_from_data_batch(data_batch, ...)

    # ③ 像素 → latent token（调 Wan2.2 VAE 编码）
    gen_data_clean = self.get_data_and_condition(data_batch, iteration=iteration)

    # ④ 采样 sigma / timestep（按分辨率查 shift）
    timesteps_vision, sigmas_vision = self._get_train_noise_level_vision(...)

    # ⑤ 文本 + latent token 打包成一条序列
    packed_sequence = self._pack_input_sequence(sequence_plans, input_text_indexes, gen_data_clean, ...)

    # ⑥ 加噪，得到 xt 和速度目标 vt = ε - x0
    gen_data_noised = self._add_noise_to_input(gen_data_clean, packed_sequence, sigmas_vision, ...)
    self._replace_clean_with_noised(packed_sequence, gen_data_noised)
    packed_sequence.to_cuda()

    # ⑦ 网络前向，预测速度场
    out_net = self.denoise(data_batch_packed=packed_sequence, memory=memory)

    # ⑧ Flow Matching loss + 负载均衡辅助 loss
    loss, losses_dict = self._compute_losses(out_net=out_net, ...)

    return output_batch, loss
```

> 注意：这个类里的 `forward()`（第 3159 行）是**空的**（`pass`，带 `@torch.no_grad` 装饰器）。
> 训练真正的 forward 是 `training_step()`，推理真正的 forward 是 `generate_samples_from_batch()`。
> 这是 Imaginaire 框架的风格：训练/推理各有专门入口，而不是复用同一个 `forward()`。

#### 7.3.8 采样噪声时间（含 shift）`_get_train_noise_level_vision()`（第 1299 行）

```python
rectified_flow = self.rectified_flow_image if is_image_batch else self.rectified_flow_video

# shift 三种取值方式：
#   int  → 所有样本用同一个 shift
#   dict → 按分辨率查表（Edge: 256→3, 480→5, 720→10）
#   dynamic → 按 token 数动态算 sqrt(num_tokens / base_tokens)
sigmas = rectified_flow.sample_train_time(batch_size, iteration, shifts=shifts)  # [B,1]
timesteps = sigmas * max_timestep   # 归一化 sigma → 时间步
```

- `sample_train_time` 从配置的训练时间分布（如 logit-normal）里采样一个 **sigma（噪声强度）**。
- `timesteps = sigmas × num_train_timesteps`：把 sigma 换算到网络的时间步尺度，供时间嵌入使用。
- **shift 的作用**：shift 越大，采样到的 sigma 越偏向「高噪声区」，
  让模型把更多训练算力花在难去噪的大噪声阶段（与之前讲的 shift 一致）。

#### 7.3.9 加噪与速度目标 `_add_noise_to_input()`（第 1477 行）

流匹配的核心公式在这里落地：

```python
epsilon = torch.randn(x0.size())                    # 采样标准高斯噪声 ε

# 条件帧的 sigma 置 0（condition_mask=1 的地方不参与去噪学习）
noisy_mask = 1.0 - condition_mask
sigmas = sigmas * noisy_mask

# 关键一步：RectifiedFlow 插值
xt, vt = rectified_flow.get_interpolation(epsilon, x0, sigmas)
# xt = (1-σ)·x0 + σ·ε          （带噪 latent）
# vt = ε - x0                    （速度目标，即流匹配要学的东西）
```

- **xt**：干净 latent `x0` 与噪声 `ε` 按 sigma 线性插值得到的「中间态」，喂给网络。
- **vt**：速度目标 = `ε - x0`。网络预测的速度 `v` 越接近 `vt`，去噪路径越准。
- **condition_mask**：I2V 的第一帧 / V2V 的前 5 帧是「条件」，它们的 sigma 被置 0（保持干净），
  不参与加噪，也就不会计入 loss——这正是三种模式（T2V/I2V/V2V）共用一个前向的原因。
- 动作（action）和声音（sound）也走同样的 `get_interpolation` 流程，只是张量形状不同。

#### 7.3.10 损失计算 `_compute_losses()`（第 1130 行）

```python
total_loss = 0.0

# 1) 视觉流匹配损失（Edge 的主损失）
fm_loss_vision = compute_flow_matching_loss(pred, target=vt, condition_mask, timesteps, ...)
total_loss += fm_loss_vision * loss_scale           # loss_scale 默认 10.0

# 2) 动作损失（action_gen=True）
total_loss += fm_loss_action * action_loss_weight

# 3) 声音损失（Edge 无）

# 4) MoE 负载均衡辅助损失（und/gen 两路）
for t in ["und", "gen"]:
    total_loss += compute_load_balancing_loss(lbl_metadata, coeff, method, ...)
```

要点：
- **主损失**是流匹配损失 `compute_flow_matching_loss`（在 `algorithm/loss/flow_matching.py`），
  度量网络预测速度 `v` 与目标 `vt` 的误差，并按时间权重加权。
- **`loss_scale=10.0`**：Edge 把视觉损失放大 10 倍（对应 7.2 的 `diffusion loss scale = 10.0`）。
- **负载均衡损失**：MoT 生成塔是 MoE，为了防止「所有 token 都挤进同一个专家」，
  加了一个辅助损失（aux-loss）鼓励专家负载均衡，系数在 `[lbl]` 配置里。
- 没有 action/sound 数据的 batch 会用 `0.0 * sum(preds)` 造一个 dummy loss，
  保证 FSDP 的梯度 reduce 不会因为某些 rank 没有对应模态而挂死。

#### 7.3.11 推理 `generate_samples_from_batch()`（第 2438 行）

推理入口，结构与 `training_step` 平行，但把「加噪一步到位」换成「采样循环逐步去噪」：

```
① 建 sequence plans
② get_data_and_condition（编码条件视觉）
③ 初始化纯噪声（条件帧保持干净）
④ 采样循环：重复 num_steps 次
      denoise() 预测速度 → sampler 沿 ODE 前进一步 → 更新 latent
⑤ 返回去噪后的 latent（再经 VAE decode 回像素）
```

关键参数：`guidance=1.5`（classifier-free guidance）、`num_steps=35`、`shift=5.0`、`sigma_max=80.0`。
若传了 `upsample_task`（`"t2i"`/`"t2v"`/`"i2v"`），会先用 reasoner 塔把简短 prompt 扩写成详细描述再生成。

#### 7.3.12 检查点 `state_dict()` / `load_state_dict()`（第 3784 / 3831 行）

- 权重以 **扁平的 `net.*` 键** 保存（如 `net.xxx.weight`），EMA 副本存为 `net_ema.*`。
- 可选 `exclude_reasoner_weights_from_checkpoint`：只存生成侧权重、不存 reasoner 塔
  （SFT 时 reasoner 塔被冻结不训练，导出时可以省掉）。
- `load_state_dict` 要求 `strict=False`，把缺失/多余键返回给上层的 DCP 加载器处理
  （这就是「断电续训」时能容忍配置略有差异的原因）。

#### 7.3.13 小结：`omni_mot_model.py` 的职责边界

| 做什么 | 在这个文件 | 具体位置 |
|--------|-----------|---------|
| tokenizer 管理（文本/视觉/声音） | ✅ | `set_up_tokenizers` |
| 网络搭建（MoT 骨架 + FSDP 切分） | ✅ | `build_net` / `parallelize_vfm_network` |
| 流匹配噪声调度（采样 t、加噪、算 vt） | ✅ | `_get_train_noise_level_*` / `_add_noise_to_input` |
| 训练 loss（流匹配 + 负载均衡） | ✅ | `_compute_losses` |
| 推理采样循环 | ✅ | `generate_samples_from_batch` |
| 网络前向（MoT 注意力、专家路由） | ❌ 在 `mot/cosmos3_vfm_network.py` | `denoise()` 只调用 `net(...)` |
| 流匹配数学公式 | ❌ 在 `diffusion/rectified_flow.py` | `get_interpolation` |
| 损失数学公式 | ❌ 在 `algorithm/loss/flow_matching.py` | `compute_flow_matching_loss` |

一句话：**`OmniMoTModel` 是「导演」**，负责把 tokenizer、网络、噪声调度、损失、采样器串成完整流程；
真正干活的「演员」（MoT 网络前向、流匹配公式）在别的文件里。

---

### 7.4 MoT 网络架构 `cosmos3_vfm_network.py`

**文件**：`cosmos_framework/model/generator/mot/cosmos3_vfm_network.py`（共 1178 行）

上一节说到 `OmniMoTModel` 是「导演」，那么 **`Cosmos3VFMNetwork` 就是「主角演员」**——真正的 MoT 网络架构在这里。

#### 7.4.1 一句话定位

```
OmniMoTModel.denoise()          （导演喊"开拍"）
        │
        └── self.net( ... )      （演员开始演）
                │
                └── Cosmos3VFMNetwork.forward()   ← 真正的网络前向，就在这个文件
```

这个文件定义了两个类：

| 类 | 作用 | 行号 |
|----|------|------|
| `Cosmos3VFMNetworkConfig` | 网络结构超参（通道数、patch 大小、模态开关） | 24 |
| `Cosmos3VFMNetwork` | 真正的 MoT 网络（桥接层 + 语言骨干 + 前向） | 104 |

#### 7.4.2 网络结构 `__init__()`（第 108 行）

`Cosmos3VFMNetwork.__init__` 的关键逻辑——它**不新建 Transformer 层**，而是：

```python
class Cosmos3VFMNetwork(PreTrainedModel):
    def __init__(self, language_model, config):
        super().__init__(config)
        self.language_model = language_model          # ← 语言骨干（Nemotron/Qwen）整个塞进来

        text_config = config.vlm_config.text_config
        self.hidden_size = text_config.hidden_size     # 隐层维度（Edge: 2560）
        self.num_heads = text_config.num_attention_heads
        self.num_hidden_layers = text_config.num_hidden_layers

        if config.vision_gen:                          # Edge 有 vision_gen=True
            self.latent_patch_size = config.latent_patch_size          # patch 大小 = 2
            self.latent_downsample = config.latent_downsample_factor * config.latent_patch_size  # 8×2=16
            self.patch_latent_dim = self.latent_patch_size**2 * self.latent_channel  # 2²×16=64

            self.time_embedder = TimestepEmbedder(self.hidden_size, bias=...)  # 时间步嵌入
            self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)  # VAE latent → LLM 空间
            self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)  # LLM 空间 → VAE latent

        if config.action_gen:                          # Edge 有 action_gen=True
            self.action2llm = DomainAwareLinear(self.action_dim, self.hidden_size, self.num_embodiment_domains)
            self.llm2action = DomainAwareLinear(self.hidden_size, self.action_dim, self.num_embodiment_domains)
            self.action_modality_embed = nn.Parameter(...)
```

**关键理解**：MoT 的「Transformer 层」**不在这个文件**，而在 `language_model`（Nemotron-2B 骨干）里。
这个文件只负责两件事：

1. **桥接层**（把不同模态的 token 投影进/出 LLM 的统一隐层空间）：`vae2llm`/`llm2vae`/`action2llm`/`llm2action`/`sound2llm`/`llm2sound`
2. **时间嵌入**：`time_embedder`（把扩散时间步 t 变成向量，注入带噪 token）

这些就是 SFT 的 `keys_to_select` 里那 5 类可训练参数（`moe_gen` 在语言骨干里，`time_embedder`/`vae2llm`/`llm2vae` 在这里）。

#### 7.4.3 双通路注意力（MoT 的核心，在 `unified_mot.py`）

真正的 MoT 结构在语言骨干里，由 `MoTDecoderLayer` + `PackedAttentionMoT` 实现。
虽然不在 `cosmos3_vfm_network.py` 文件里，但这是理解整个网络架构的关键，必须一起讲。

**MoT = 理解通路（und）+ 生成通路（gen）双份投影**：

```python
class PackedAttentionMoT(nn.Module):
    def __init__(self, config, ...):
        # —— 理解通路（Reasoner，处理文本）——
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size)
        self.q_norm = RMSNorm(head_dim)   # QK norm（文本侧）

        # —— 生成通路（Generator，处理 latent token）——
        self.q_proj_moe_gen = nn.Linear(hidden_size, num_heads * head_dim)
        self.k_proj_moe_gen = nn.Linear(hidden_size, num_kv_heads * head_dim)
        self.v_proj_moe_gen = nn.Linear(hidden_size, num_kv_heads * head_dim)
        self.o_proj_moe_gen = nn.Linear(num_heads * head_dim, hidden_size)
        self.q_norm_moe_gen = RMSNorm(head_dim)  # QK norm（生成侧）
```

即：**每一层 Transformer 都有两套独立的 Q/K/V/O 投影**，一套给文本（理解塔），一套给 latent token（生成塔）。
这正是「Mixture of Transformers」名字的由来——不是混合多个模型，而是**在同一个 Transformer 层里混合理解与生成两套参数**。

对应的 `MoTDecoderLayer`（第 1013 行）里，MLP 也分两套：

```python
class MoTDecoderLayer(nn.Module):
    def __init__(self, config, ...):
        self.self_attn = PackedAttentionMoT(...)   # 双通路注意力

        if (layer_idx not in config.mlp_only_layers) and (config.num_experts > 0 and (layer_idx+1) % decoder_sparse_step == 0):
            # 稀疏层：理解塔 + 生成塔各一个 MoE 块
            self.mlp = Qwen3VLMoeTextSparseMoeBlock(config)          # 理解塔的 MoE
            self.mlp_moe_gen = Qwen3VLMoeTextSparseMoeBlock(config)  # 生成塔的 MoE（可训练部分）
        else:
            # 稠密层：普通 MLP
            self.mlp = layer_types.mlp(config)
            self.mlp_moe_gen = layer_types.mlp(config)
```

这就是 SFT 里 `keys_to_select` 包含 `"moe_gen"` 的原因——生成塔的 MoE 专家（`mlp_moe_gen`）是要微调的目标，
而理解塔（`q_proj`/`k_proj`/`mlp` 等）被冻结。

#### 7.4.4 双通路注意力怎么跑：`two_way_attention`（`attention.py` 第 113 行）

MoT 的一条序列里混着「文本 token（理解）+ latent token（生成）」，注意力分两路算：

```
文本 token (und, N_und 个)  ──causal self-attention──▶ 文本只看自己（因果，从左到右）
latent token (gen, N_gen 个) ──full attention────────▶ latent 看全部（文本 + 所有 latent，双向）
```

对应代码（`attention.py`）：

```python
def two_way_attention(...):
    # 理解塔：因果自注意力（und → und）
    causal_res = attention(causal_q, causal_k, causal_v, is_causal=True, ...)

    # 生成塔：全注意力（gen → 全部 und + gen）
    full_res = attention(full_q, get_all_seq(k), get_all_seq(v), ...)

    out_all = from_mode_splits(causal_out, full_out, packed_query_states)
```

即 **two-way = 两条注意力路径**：
- **und 路径**：文本 token 因果自注意力（像普通 LLM）
- **gen 路径**：latent token 对「全部 token」做全注意力（像扩散模型的 transformer，能双向看）

（Edge 用的就是 `joint_attn_implementation="two_way"`。另有 `three_way` 把生成塔的 self-attention 和 cross-attention 再拆开，支持 NATTEN 稀疏注意力，Nano/Super 用。）

#### 7.4.5 patchify：latent 切成 patch（第 294 行）

VAE 输出的 latent 是 `[C, T, H, W]` 四维，网络要的是「1D token 序列」，所以先 patchify：

```python
def patchify_and_pack_latents(self, tokens_vision, token_shapes_vision):
    p = self.latent_patch_size   # = 2
    for latent, (t, h, w) in zip(tokens_vision, token_shapes_vision):
        # [C,T,H,W] → reshape → [T, h/p, w/p, p, p, C] → einsum → [T*h_patches*w_patches, p*p*C]
        latent = latent.reshape(C, t, h_patches, p, w_patches, p)
        latent = torch.einsum("cthpwq->thwpqc", latent).reshape(-1, p*p*C)
        packed_latent.append(latent)
    return torch.cat(packed_latent, dim=0)   # [total_patches, patch_latent_dim=64]
```

- 每个 `2×2` 的空间块打包成 1 个 token（patch token），维度 = `2×2×16=64`。
- 所有样本的 patch 沿第 0 维拼成 1 条长序列，喂给 `vae2llm` 投影到 `hidden_size=2560`。
- `unpatchify_and_unpack_latents`（第 341 行）是逆过程，把网络输出的 patch 预测还原成 `[C,T,H,W]`。

#### 7.4.6 完整前向 `forward()`（第 894 行）

这是网络的「一镜到底」，流程与 `OmniMoTModel.training_step` 里的 `denoise()` 对接：

```python
def forward(self, packed_seq, memory=None, ...):
    # ① 编码各模态 → 统一隐层空间
    packed_sequence, target_dtype = self._encode_text(packed_seq)      # 文本 → embedding
    original_latent_shapes = self._encode_vision(packed_seq, ...)       # latent → patch → vae2llm → +时间嵌入
    self._encode_action(packed_seq, ...)                                # 动作 → action2llm
    self._encode_sound(packed_seq, ...)                                 # 声音（Edge 无）

    # ② 构建注意力元数据（SplitInfo：哪段 causal、哪段 full）
    input_pack, attention_meta, natten_metadata = build_packed_sequence(
        joint_attn_implementation, packed_sequence, attn_modes, split_lens, ...)

    # ③ 语言骨干前向（MoT 的 N 层 Transformer）
    packed_outputs, lbl_metadata = self.language_model(
        input_pack, attention_mask=attention_meta, position_ids=..., ...)

    # ④ 解码各模态 → 预测速度
    self._decode_vision(packed_seq, last_hidden_state, output_dict, ...)  # llm2vae → unpatchify
    self._decode_action(packed_seq, last_hidden_state, output_dict)       # llm2action
    self._decode_sound(packed_seq, last_hidden_state, output_dict)        # llm2sound

    return output_dict   # preds_vision / preds_action / preds_sound / lbl_metadata_*
```

四个 `_encode_*` / `_decode_*` 是对称的：

| 阶段 | 视觉 | 动作 | 声音 |
|------|------|------|------|
| 编码 | `patchify` → `vae2llm` → +时间嵌入 | `action2llm` → +模态嵌入 | `sound2llm` → +模态嵌入 |
| 解码 | `llm2vae` → `unpatchify` | `llm2action` | `llm2sound` |

#### 7.4.7 时间步怎么注入：`TimestepEmbedder`（`modeling_utils.py` 第 28 行）

扩散模型必须知道「当前在第几步」，所以每个带噪 token 要加时间嵌入：

```python
class TimestepEmbedder(nn.Module):
    def forward(self, t):
        t_freq = self.timestep_embedding(t, 256)   # 正弦位置编码（DiT 风格）
        t_emb = self.mlp(t_freq)                   # Linear → SiLU → Linear
        return t_emb                                # [N, hidden_size]
```

注入点：`_encode_vision` 里（第 606 行）：

```python
timesteps_vision = vision.timesteps * self.timestep_scale      # 先乘 timestep_scale
packed_timestep_embeds_vision = self._embed_packed_timesteps(...)  # 时间嵌入
packed_tokens_vision = _apply_timestep_embeds_to_noisy_tokens(...)  # 只加到「带噪」token 上
```

关键：**时间嵌入只加到带噪 token 上，条件 token（如 I2V 第一帧）不加**——
`_apply_timestep_embeds_to_noisy_tokens`（第 1111 行）用 `noisy_frame_indexes` 精确定位，
通过 `scatter_add` 把时间嵌入累加到对应 patch 的 embedding 上。

#### 7.4.8 FSDP 并行 `parallelize_vfm_network`（`parallelize_vfm_network.py` 第 45 行）

建好网络后，`OmniMoTModel.build_net()` 调这个函数做并行优化：

```python
def parallelize_vfm_network(model, parallel_dims, compile_config, ac_config, ...):
    model.language_model = parallelize_unified_mot(...)   # 语言骨干内部做 FSDP/AC/compile
    if compile_config.enabled and compiled_region == "all":
        model = apply_compile(model, ...)                  # torch.compile 编码/解码头
    if parallel_dims.dp_enabled:
        model = fully_shard(model, mesh=dp_mesh, ...)      # FSDP2 分片
    return model
```

单卡场景（`dp_enabled=False`）下这些全是 no-op，直接返回原网络。

#### 7.4.9 小结：MoT 架构分层全景

```
Cosmos3VFMNetwork（cosmos3_vfm_network.py）      ← 外壳：桥接层 + 时间嵌入 + 前向编排
    │
    ├── time_embedder / vae2llm / llm2vae        ← 视觉桥接（可训练）
    ├── action2llm / llm2action                  ← 动作桥接（可训练，DomainAware）
    ├── sound2llm / llm2sound                    ← 声音桥接（Edge 无）
    │
    └── language_model（Nemotron-2B-Dense-VL）   ← 骨干：MoT 双通路 Transformer
            │
            ├── embed_tokens / norm / lm_head    ← 文本嵌入与输出（冻结）
            └── layers × N
                └── MoTDecoderLayer              ← 每层双通路
                    ├── PackedAttentionMoT       ← 双套 QKV（und + gen）
                    ├── mlp（理解塔 MoE/MLP）      ← 冻结
                    └── mlp_moe_gen（生成塔 MoE/MLP）← 可训练（keys_to_select 的 moe_gen）
```

一句话：**MoT 架构 = 同一个 Transformer 层里装两套投影（理解塔 + 生成塔）**，
`cosmos3_vfm_network.py` 负责「把 latent token 桥接进 LLM 空间、注入时间嵌入、再把预测桥接回 latent 空间」，
真正的注意力与 MoE 计算在 `language_model`（`unified_mot.py` 的 `MoTDecoderLayer`）里。

---

## 8. 第六层：数据加载

### 8.1 数据格式（JSONL）

数据集是 BridgeData2 机器人操作视频，JSONL 格式：

```json
{
    "uuid": "episode_000015_clip000",
    "duration": 17.4,
    "width": 256, "height": 256,
    "vision_path": "videos/episode_000015_clip000.mp4",
    "t2w_windows": [{
        "start_frame": 0, "end_frame": 86,
        "caption_json": {
            "subjects": [...],
            "actions": [...],
            "temporal_caption": "A black robotic arm...",
            ...
        },
        "caption": "plain text backup..."
    }]
}
```

### 8.2 数据加载器 `sft_dataset.py`

**文件**：`cosmos_framework/data/generator/local_datasets/sft_dataset.py`

核心逻辑：

1. 读取 JSONL 中的 `vision_path`，用 torchcodec 加载视频
2. 根据 `conditioning_config={0:0.7, 1:0.2, 2:0.1}` 随机选模式：
   - 0 (T2V)：不提供帧条件，纯文本→视频
   - 1 (I2V)：取第一帧作为条件
   - 2 (V2V)：取前 5 帧作为条件
3. 读取 `caption_json`（结构化 JSON 描述，优先）或 `caption`（纯文本，备选）
4. 帧采样、分辨率处理、序列打包（PackingDataLoader）

### 8.3 一句话总结

> 同一个 JSONL 文件，同一个视频，加载器随机决定"这次用于 T2V 还是 I2V 还是 V2V 训练"，所以只用一份数据集就能训三种模式。

---

## 9. 第七层：训练循环

**文件**：`cosmos_framework/trainer/__init__.py`

### 9.1 训练步

```python
def train(self, model, dataloader_train, dataloader_val):
    for iteration in range(max_iter):
        data_batch = next(dataloader_train)
        loss = model(data_batch)               # 扩散去噪前向
        loss.backward()                         # 反向传播
        optimizer.step()                        # FusedAdam 更新
        scheduler.step()                        # LR 衰减
        # 回调：loss 日志、梯度裁剪、norm 监控、检查点保存...
```

### 9.2 关键回调（`vision_sft_edge.py` 中注册的）

| 回调 | 功能 |
|------|------|
| `iter_speed` | 每步打印 loss + 耗时 |
| `grad_clip` | 梯度裁剪（L2 norm ≤ 0.1） |
| `norm_monitor` | 监控各层梯度/激活范数 |
| `device_monitor` | 监控显存/功耗/温度 |
| `wandb_2x` | wandb 日志 |
| `skip_nan_step` | NaN loss 自动跳过 |

### 9.3 可训练参数

```toml
keys_to_select = [
    "moe_gen",      # MoE 生成专家
    "time_embedder", # 时间嵌入
    "vae2llm",      # VAE→LLM 桥接
    "llm2vae",      # LLM→VAE 桥接
    "k_norm_und_for_gen",  # und-K norm
]
```

只有名称包含这些子串的参数才会被优化器更新，其余全部冻结。

### 9.4 损失计算全景

训练循环里 `model(data_batch)` 实际调用的是 `OmniMoTModel.training_step()`（见 7.3.7），
其第 8 步 `_compute_losses()` 汇总所有损失。整个 loss 体系分三层：

| 层次 | 文件 | 负责什么 |
|------|------|---------|
| 编排层 | `omni_mot_model.py` → `_compute_losses()` | 决定「算哪几个 loss、各乘多少系数、怎么加总」 |
| 公式层 | `algorithm/loss/flow_matching.py` | 流匹配损失的数学公式 |
| 公式层 | `algorithm/loss/load_balancing.py` | MoE 负载均衡损失的数学公式 |
| 支撑层 | `diffusion/rectified_flow.py` | 采样噪声时间 t、算插值 xt 与速度目标 vt、时间加权 |

Edge SFT 的最终 loss 组成（对应 7.3.10）：

```
total_loss = flow_matching_loss_vision × loss_scale(10.0)
           + flow_matching_loss_action × action_loss_weight
           + aux_loss_und   （理解塔 MoE 负载均衡，若启用）
           + aux_loss_gen   （生成塔 MoE 负载均衡，若启用）
```

### 9.5 流匹配损失 `compute_flow_matching_loss()`

**文件**：`cosmos_framework/model/generator/algorithm/loss/flow_matching.py`（共 91 行）

这是 Edge SFT 的**主损失**。核心数学（第 66-90 行）：

```python
def compute_flow_matching_loss(pred, target, condition_mask, timesteps, ...):
    for i in range(len(pred)):
        sqerr_i = (pred[i] - target[i]) ** 2          # ① 逐元素平方误差
        noisy_mask_i = 1.0 - condition_mask[i]         # ② 只保留「带噪」token

        # ③ 时间加权：按 sigma 查权重 w(σ_t)
        tw_i = rectified_flow.train_time_weight(timesteps[i, :T_i], ...)
        tw_i = tw_i.reshape(-1, *([1]*(condition_mask[i].ndim - 1)))  # 广播到 [T,1,1]

        # ④ 加权 + 掩码 + 求平均
        per_instance_weighted_losses.append((sqerr_i * tw_i * noisy_mask_i).mean())

    return per_instance_weighted_loss.mean(), per_instance_loss
```

逐步拆解：

| 步骤 | 数学 | 说明 |
|------|------|------|
| ① 平方误差 | `(pred − target)²` | `pred` 是网络预测的速度 v，`target` 是真实速度 vt = ε − x0 |
| ② 掩码 | `×(1 − condition_mask)` | 条件 token（I2V 第一帧等）mask=1，被置 0，不参与 loss |
| ③ 时间加权 | `×w(σ_t)` | 按当前噪声强度 sigma 加权（见 9.7） |
| ④ 平均 | `.mean()` | 先每个样本内平均，再跨样本平均 |

**两个返回值的区别**：
- `per_instance_weighted_loss.mean()` → 加权后的**标量 loss**，用于反向传播
- `per_instance_loss` → 每个样本**未加权的 loss**，用于日志记录（wandb 里看到的 `flow_matching_loss_vision_per_instance`）

**没有 valid token 时的 dummy loss**（第 56 行）：

```python
if not has_valid_tokens:
    dummy_loss = 0.0 * sum(p.sum() for p in pred)   # 保持梯度图一致，防止 FSDP 挂死
    return dummy_loss, dummy_loss.unsqueeze(0)
```

### 9.6 速度目标与插值 `RectifiedFlow`

**文件**：`cosmos_framework/model/generator/diffusion/rectified_flow.py`（共 211 行）

这是流匹配（Rectified Flow）的核心数学。训练时做两件事：**采样噪声时间** + **算插值和速度目标**。

#### 9.6.1 采样噪声时间 `sample_train_time()`（第 134 行）

```python
def sample_train_time(self, batch_size, iteration=None, shifts=None):
    time = self.train_time_sampler(batch_size, device=..., shifts=shifts)
    return time   # sigma ∈ [0,1]，形状 [B]
```

`TrainTimeSampler`（第 13 行）支持多种分布，最常用的两种：

```python
if self.distribution == "uniform":
    t = torch.rand((batch_size,))                      # 均匀采样 t ∈ [0,1]
elif self.distribution == "logitnormal":
    t = torch.sigmoid(torch.randn((batch_size,)))      # logit-normal 采样

# 关键：shift 的 warping（非 ltx2 分布）
if shifts is not None:
    t = shifts * t / (1 + (shifts - 1) * t)            # ← shift 重映射公式
```

**这就是 `shift` 参数的数学本质**：`sigma = shift·t / (1 + (shift−1)·t)`。
- shift=1 时退化为恒等映射 `sigma=t`（均匀分布）
- shift 越大，`sigma` 越向 1（高噪声端）压缩，模型把更多训练算力花在「大噪声难去噪」阶段
- Edge 用 `shift={256:3, 480:5, 720:10}`，480p 训练时 shift=5

#### 9.6.2 插值与速度目标 `get_interpolation()`（第 176 行）

这是流匹配**最核心的一行代码**，在 `_add_noise_to_input()`（7.3.9）里被调用：

```python
def get_interpolation(self, x_0, x_1, t):
    # 注意命名：x_0 是噪声 ε，x_1 是干净数据 x0（与扩散社区命名相反）
    for i in range(len(x_0)):
        x_t.append(x_0[i] * t[i] + x_1[i] * (1 - t[i]))   # xt = σ·ε + (1−σ)·x0
        dot_x_t.append(x_0[i] - x_1[i])                    # vt = ε − x0
    return x_t, dot_x_t
```

对应公式：

```
xt = (1−σ)·x0 + σ·ε      （带噪 latent，喂给网络）
vt = ε − x0              （速度目标，网络要学的东西）
```

> **为什么叫「速度」？** 流匹配把去噪过程看成一条从噪声 ε（t=1）流向干净数据 x0（t=0）的轨迹，
> `vt = ε − x0` 是这条轨迹在 t 处的**切线方向（速度场）**。网络学会预测这个速度场后，
> 推理时就能沿它逐步「流动」回干净数据。

### 9.7 时间加权 `TrainTimeWeight`

时间加权决定「不同噪声强度下的误差在 loss 里占多大比重」，`train_time_weight` 在
`RectifiedFlow.__init__`（第 129 行）里由 `TrainTimeWeight(noise_scheduler, method)` 构建。
Edge 默认 `train_time_weight="uniform"`，即所有 sigma 一视同仁（权重=1）。

配置里常见的几种 `train_time_weight_method` 对应不同的加权函数 `w(σ)`，
它们会在 9.5 的第 ③ 步把每个时间步的误差按 `w(σ_t)` 缩放，实现对难易样本的重新分配。

### 9.8 MoE 负载均衡损失 `compute_load_balancing_loss()`

**文件**：`cosmos_framework/model/generator/algorithm/loss/load_balancing.py`（共 73 行）

MoT 的生成塔/理解塔是 MoE（多个专家），如果没有约束，路由器可能把所有 token 都路由到少数几个专家，
导致其余专家「饿死」。负载均衡损失就是防止这个的辅助损失。

```python
def compute_load_balancing_loss(lbl_metadata, coeff, method, device_mesh):
    # lbl_metadata 包含：
    #   num_tokens_per_expert      [num_layers, num_experts]  每个专家收到的 token 数
    #   mean_router_prob_per_expert[num_layers, num_experts]  路由到每个专家的平均概率
    #   top_k                      [num_layers, 1]            每个 token 选的专家数

    # 归一化：每层每个专家实际占比 f_i（除以 top_k 使 ∑f_i = 1）
    mean_tokens_per_expert = num_tokens_per_expert / (num_tokens * top_k)

    # 负载均衡损失 = mean(∑ f_i · p_i) × num_experts
    lbl = torch.mean(torch.sum(mean_tokens_per_expert * mean_router_prob_per_expert, dim=-1) * num_experts)
    return lbl * coeff
```

直觉理解：
- 如果路由「均衡」，每个专家的实际占比 `f_i` 都接近 `1/num_experts`，且 `p_i` 均匀，loss 小
- 如果路由「失衡」，某些专家 `f_i` 很大而 `p_i` 也集中在少数专家，`∑ f_i·p_i` 偏离均衡值，loss 大
- `method="global"` 时先跨 rank 汇总（DTensor 的 `full_tensor()`），`"local"` 只看本 rank

Edge 的系数在 `[lbl]` 配置里（`coeff_und`/`coeff_gen`），若为 None 则不启用该路辅助损失。

---

## 10. 第八层：NPU 适配

### 10.1 设备后端自动检测

**文件**：`cosmos_framework/utils/device_backend.py`（165 行）

```python
_KIND = select_backend_kind()            # "npu" | "cuda" | "cpu"
IS_NPU = _KIND == "npu"
DIST_BACKEND = "hccl" if IS_NPU else "nccl"
DEVICE_TYPE = "npu" if IS_NPU else "cuda"
```

自动检测逻辑：先看 `torch.npu.is_available()`，再看 `torch.cuda.is_available()`。

### 10.2 CUDA → NPU 重定向

`train.py` 中的 `transfer_to_npu` 把所有 `torch.cuda.*` 调用映射到 `torch.npu.*`，模型代码无需修改就能在 NPU 上跑。

### 10.3 注意力后端

SDPA 注意力已适配 NPU（`model/attention/sdpa/__init__.py`），检测到 NPU 后自动选用 NPU 注册的融合 kernel。

### 10.4 FusedAdam on NPU

`cosmos_framework/utils/generator/fused_adam.py` 基于 TransformerEngine，TE 的底层 kernel 已被替换为 CANN 实现，`fused=true` 可直接使用。

### 10.5 全局设备标志

```bash
# 环境变量
export COSMOS_DEVICE=npu        # 强制 NPU
export ASCEND_RT_VISIBLE_DEVICES="8"  # 选哪颗芯片

# Python 侧
# flags.py: DEVICE = Device(os.environ.get("COSMOS_DEVICE", "cuda"))
```

---

## 11. 如何修改脚本跑起来

以下以 `launch_sft_vision_edge_yundao.sh` 为例，说明需要修改哪些内容才能将脚本跑起来。

### 11.1 必需的环境变量

| 变量 | 含义 | 示例值 |
|------|------|--------|
| `DATASET_PATH` | JSONL 数据集路径 | `/mi/.../sft_dataset_bridge` |
| `BASE_CHECKPOINT_PATH` | DCP 权重路径 | `/mi/.../Cosmos3-Edge-DCP` |
| `COSMOS3_EDGE_PROCESSOR_PATH` | HF processor 路径（本地，用于离线加载） | `/mi/.../Cosmos3-Edge` |
| `WAN_VAE_PATH` | Wan2.2 VAE 权重文件 | `/mi/.../Wan2.2_VAE.pth` |
| `OUTPUT_ROOT` | 训练产物输出根目录 | `/mi/.../cosmos_trainging_logs` |

### 11.2 必需的特殊环境变量

| 变量 | 含义 |
|------|------|
| `HF_HUB_OFFLINE=1` | 禁止 HuggingFace 联网 |
| `COSMOS_DEVICE=npu` | 强制使用 NPU |
| `ASCEND_RT_VISIBLE_DEVICES="8"` | 指定 NPU 芯片（逻辑编号，对应物理 NPU 4, Chip 0） |

### 11.3 分布式拓扑（单卡示例）

```bash
NPROC_PER_NODE=1          # 每节点 1 进程 = 1 芯片
NNODES=1                  # 单节点
NODE_RANK=0               # 当前节点编号
MASTER_ADDR="127.0.0.1"   # 主节点地址（本机）
MASTER_PORT=50012         # 通信端口
```

### 11.4 TOML 需要改的项

| 配置路径 | 原始默认值 | NPU 应改为 | 原因 |
|----------|-----------|-----------|------|
| `[model].compile.enabled` | `true` | `false` | NPU 不支持 torch.compile |
| `[optimizer].fused` | `true` | `true`（不动） | Ascend 版 FusedAdam 支持，不需要改 |
| `[trainer].grad_accum_iter` | `2` | `2`（不动） | 单卡时按需可调大（如 4）补偿 batch size |
| `[job].wandb_mode` | `"disabled"` | `"offline"` | 存本地日志方便回顾 |

### 11.5 硬件要求

| 项 | 要求 |
|----|------|
| GPU/NPU | 1 × Ascend 910，显存 ≥ 50 GB |
| Edge 模型 SFT 单卡显存 | ~48 GB |
| 数据集 | BridgeData2-Subset-Synthetic-Captions（~5 GB） |
| DCP 权重 | Cosmos3-Edge 导出的 DCP（~20 GB） |
| VAE | Wan2.2_VAE.pth（~1 GB） |

### 11.6 运行步骤

```bash
# 1. 确认 conda 环境
conda activate cosmos-framework
python -c "import torch; import torch_npu; print(torch.npu.is_available())"  # 必须 True

# 2. 确认所有路径存在
ls $DATASET_PATH/train/video_dataset_file.jsonl
ls $BASE_CHECKPOINT_PATH/model/
ls $WAN_VAE_PATH
ls $COSMOS3_EDGE_PROCESSOR_PATH/tokenizer_config.json

# 3. 启动
bash /mi/data2T/liujin/code/cosmos_ascend/cosmos-framework/examples/launch_sft_vision_edge_yundao.sh
```

### 11.7 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| `ValueError: Optimizers with fused=False are not supported` | `fused` 设成了 `false` | 改回 `true` |
| `ModuleNotFoundError: No module named 'loguru'` | conda 环境没激活 | `conda activate cosmos-framework` |
| `Checkpoint directory does not exist` | DCP 路径不对 | 确认 `BASE_CHECKPOINT_PATH` |
| `missing video_dataset_file.jsonl` | 数据集路径不对 | 确认 `DATASET_PATH` |
| `HCCL connect timeout` | 单卡 torchrun 的 hccl init 问题 | 确认 `NPROC_PER_NODE=1` |

---

## 12. 训练产物

训练输出在 `$IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<name>/`：

```
cosmos_trainging_logs/cosmos3/sft/vision_sft_edge/
├── config.yaml                   # 完整生效的训练配置
├── config.pkl                    # Python pickle 版本
├── launch_info.yaml              # 启动元信息
├── job_env.yaml                  # 环境变量快照
├── wandb_id.txt                  # wandb run ID（用于续写/恢复）
├── checkpoints/
│   ├── iter_000000100/           # DCP 检查点
│   │   ├── model/                # 模型权重 (.distcp)
│   │   ├── optim/                # 优化器状态
│   │   ├── scheduler/            # LR 调度器状态
│   │   └── trainer/              # 训练步数 + RNG 状态
│   ├── iter_000000200/
│   ├── ...                       # 每 100 步一个
│   └── latest_checkpoint.txt     # 最新检查点名称
├── DeviceMonitor/                # 设备监控日志（显存/功耗/温度）
├── norm_monitor/                 # 梯度/激活范数日志
├── EveryNDrawSample/             # 采样可视化
└── wandb/
    └── offline-run-*/            # wandb 离线日志
        ├── files/
        │   ├── requirements.txt
        │   ├── wandb-summary.json    # 最终汇总
        │   └── wandb-history.jsonl   # 每步指标
        └── run-*.wandb               # 二进制数据
```

### DCP → safetensors 导出

首先设定指定的conda环境，建议跟训练时的保持一致
```bash
CONDA_HOME="/mi/sfs_turbo/lilin_v1/anaconda3"
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate cosmos-framework
```

不导出vision tower，模型只用于生成

```bash
python -m cosmos_framework.scripts.export_model \
    --checkpoint-path /path/to/checkpoints/iter_000000500 \
    --config-file /path/to/config.yaml \
    --no-vit
    -o /path/to/output_safetensors
```

导出后的 safetensors 目录可直接用于推理（`--checkpoint-path output_safetensors/`）。

---

## 附录：关键文件索引

| 文件 | 作用 |
|------|------|
| `examples/launch_sft_vision_edge_yundao.sh` | 用户启动脚本（路径 + 环境变量） |
| `examples/_sft_launcher_common.sh` | 共用 torchrun 启动逻辑 |
| `examples/toml/sft_config/vision_sft_edge.toml` | 所有训练超参 |
| `cosmos_framework/scripts/train.py` | Python 训练入口 |
| `cosmos_framework/configs/toml_config/sft_config.py` | TOML → Hydra 转换 + pydantic 校验 |
| `cosmos_framework/configs/base/experiment/sft/vision_sft_edge.py` | Edge 实验定义（模型/数据加载器/回调） |
| `cosmos_framework/configs/base/experiment/sft/models/edge_model_config.py` | Nemotron-2B-Dense-VL 模型骨干 |
| `cosmos_framework/model/generator/omni_mot_model.py` | OmniMoT 模型实现 |
| `cosmos_framework/trainer/__init__.py` | 训练循环 |
| `cosmos_framework/data/generator/local_datasets/sft_dataset.py` | JSONL 数据加载 |
| `cosmos_framework/utils/device_backend.py` | NPU/CUDA/CPU 自动检测 |
| `cosmos_framework/utils/generator/fused_adam.py` | FusedAdam 优化器 |
| `cosmos_framework/utils/generator/optimizer.py` | 优化器构建 |
| `cosmos_framework/scripts/export_model.py` | DCP → safetensors 导出 |
| `cosmos_framework/scripts/inference.py` | 推理入口 |
