# Cosmos3 Edge SFT 代码导读：从启动脚本到 loss 与参数更新

本文以当前工作区中的
[launch_sft_vision_edge_local.sh](../../cosmos/cookbooks/cosmos3/generator/audiovisual/finetune/launch_sft_vision_edge_local.sh#L1,1)
为唯一入口，追踪普通 Cosmos3-Edge Vision SFT 的真实执行路径。

> 当前 local 脚本第 28 行调用普通 `launch_sft_vision_edge.sh`，第 29 行的 causal launcher 已被注释。因此本文主角是 `OmniMoTModel`，不是 `OmniMoTCausalModel`；训练策略是 `causal_training_strategy="none"`，不是 teacher forcing。

## 1. 先建立一张全局地图

```text
launch_sft_vision_edge_local.sh
  └─ launch_sft_vision_edge.sh
       ├─ 设置 DATASET_PATH / BASE_CHECKPOINT_PATH / WAN_VAE_PATH
       └─ torchrun -m cosmos_framework.scripts.train
            ├─ load_experiment_from_toml(...)
            │    ├─ 选择 vfm 基础配置
            │    ├─ 加载 experiment=vision_sft_edge
            │    ├─ 用 vision_sft_edge.toml 覆盖实验配置
            │    └─ 用命令行 tokenizer 路径再次覆盖
            ├─ instantiate(config.model)       -> OmniMoTModel
            ├─ instantiate(config.dataloader)  -> PackingDataLoader
            └─ Trainer.train(...)
                 ├─ 读取并打包数据
                 ├─ OmniMoTModel.training_step
                 │    ├─ 文本 tokenize + 视频 VAE encode
                 │    ├─ 构造 T2V / I2V / V2V condition mask
                 │    ├─ 添加 RF 噪声并 pack 多模态 token
                 │    ├─ Cosmos3VFMNetwork.forward
                 │    │    └─ Nemotron Dense MoT 多层 forward
                 │    └─ flow-matching loss
                 ├─ backward
                 └─ AdamW step + scheduler step
```

建议第一次阅读严格按上图从上到下走。不要一开始钻进 attention kernel；先搞清楚一条样本是什么、配置如何落到对象上、网络预测什么以及 loss 如何算，再深入 attention 才有上下文。

## 2. 第一站：本机启动脚本

本机脚本主要做三件事：选择 NPU、写入本机路径、配置 `torchrun`。对应代码在
[本机环境和路径](../../cosmos/cookbooks/cosmos3/generator/audiovisual/finetune/launch_sft_vision_edge_local.sh#L6,1)、
[分布式参数](../../cosmos/cookbooks/cosmos3/generator/audiovisual/finetune/launch_sft_vision_edge_local.sh#L18,1)
和[实际 launcher 选择](../../cosmos/cookbooks/cosmos3/generator/audiovisual/finetune/launch_sft_vision_edge_local.sh#L28,1)。

当前值得注意的有效设置是：

- `ASCEND_RT_VISIBLE_DEVICES="13,14"` 暴露两张卡，但 `NPROC_PER_NODE=1` 只启动一个训练进程，因此当前是单进程训练，不会同时使用两个 rank。
- `COSMOS_DEVICE=npu` 选择工程的 Ascend 设备适配路径。
- `HF_HUB_OFFLINE=1` 禁止运行时访问 Hugging Face；processor、DCP 和 VAE 都必须在本机路径存在。
- 普通 Edge launcher 被执行，causal launcher 只是一行注释。

第二层脚本的职责是补默认路径、在资源不存在时下载或转换、导出 TOML 使用的环境变量，并启动 Python。可直接看
[资源准备](../../cosmos/cookbooks/cosmos3/generator/audiovisual/finetune/launch_sft_vision_edge.sh#L16,1)、
[环境变量到 TOML 的桥接](../../cosmos/cookbooks/cosmos3/generator/audiovisual/finetune/launch_sft_vision_edge.sh#L38,1)
和[最终 torchrun 命令](../../cosmos/cookbooks/cosmos3/generator/audiovisual/finetune/launch_sft_vision_edge.sh#L47,1)。

真正执行的命令可简化为：

```bash
torchrun --nproc_per_node=1 \
  -m cosmos_framework.scripts.train \
  --attach_vscode_debugger \
  --sft-toml=/absolute/path/to/vision_sft_edge.toml \
  -- \
  model.config.vlm_config.tokenizer.repository=null \
  model.config.vlm_config.tokenizer.revision=null \
  +model.config.vlm_config.tokenizer.tokenizer_type=/local/Cosmos3-Edge
```

这里最后三项是 Hydra 命令行覆盖，优先级高于 TOML 和 Python 实验配置。`--attach_vscode_debugger` 会让 rank 0 在实例化模型之前监听 3002 并阻塞等待 VS Code，入口见
[train.py:207](../cosmos_framework/scripts/train.py#L207,1)。

## 3. 第二站：理解四层配置，而不是只读 TOML

这套训练的最终配置不是任何一个文件单独决定的，而是四层叠加：

| 优先级 | 来源                        | 主要作用                                                     |
| ------ | --------------------------- | ------------------------------------------------------------ |
| 低     | VFM 基础配置与 config group | 建立完整对象图，例如 trainer、optimizer、model 的 `_target_` |
| 中     | `vision_sft_edge.py`        | 选择 Edge 模型、数据管线、AdamW、FSDP 等实验默认值           |
| 高     | `vision_sft_edge.toml`      | 覆盖本次训练希望暴露的超参数                                 |
| 最高   | launcher 末尾的 Hydra 参数  | 把在线 tokenizer 配置替换为本地 processor 路径               |

### 3.1 TOML 先负责“校验”，再变成 Hydra override

当前 TOML 在
[vision_sft_edge.toml](../../cosmos/cookbooks/cosmos3/generator/audiovisual/finetune/toml/sft_config/vision_sft_edge.toml#L16,1)。
它声明 `task="vfm"`、`experiment="vision_sft_edge"`，并给出本次训练的 bf16、FSDP、AdamW 参数、训练步数和数据 token 上限。

TOML 的 Pydantic 总入口是
[SFTExperimentConfig](../cosmos_framework/configs/toml_config/sft_config.py#L714,1)，
未知字段会因为 `extra="forbid"` 直接报错，见
[校验策略](../cosmos_framework/configs/toml_config/sft_config.py#L24,1)。
常用字段的 schema 默认值分别在
[ModelConfig](../cosmos_framework/configs/toml_config/sft_config.py#L300,1)
和[OptimizerConfig](../cosmos_framework/configs/toml_config/sft_config.py#L426,1)。

一个重要规则是：一般情况下，TOML **没有写出的字段不会被 schema 默认值强行覆盖到 Hydra 树**。loader 遍历原始 TOML，而不是把完整 Pydantic 对象全部输出。实现见
[load_experiment_from_toml](../cosmos_framework/configs/toml_config/sft_config.py#L742,1)
和[最终 load_config](../cosmos_framework/configs/toml_config/sft_config.py#L798,1)。

例如 TOML 的：

```toml
[model]
precision = "bfloat16"
```

会被 VFM remap 规则改写成：

```text
model.config.precision=bfloat16
```

`model.* -> model.config.*` 和 caption token 上限的特殊映射在
[VFM PATH_REMAPS](../cosmos_framework/configs/toml_config/toml_config_helper.py#L44,1)，
递归生成 override 的位置在
[build_hydra_overrides](../cosmos_framework/configs/toml_config/toml_config_helper.py#L135,1)。

### 3.2 Python 实验配置决定完整对象图

`experiment="vision_sft_edge"` 对应
[vision_sft_edge.py](../cosmos_framework/configs/base/experiment/sft/vision_sft_edge.py#L62,1)。
它最关键的选择是：

```python
defaults = [
    {"override /model": "mot_fsdp"},
    {"override /optimizer": "adamw"},
    {"override /scheduler": "lambdacosine"},
    # ...
]
```

原代码见[实验 defaults](../cosmos_framework/configs/base/experiment/sft/vision_sft_edge.py#L64,1)。
`mot_fsdp` 最终把 model 的类设成 `OmniMoTModel`，对应
[MOT_FSDP_CONFIG](../cosmos_framework/configs/base/defaults/model.py#L23,1)
和[config group 注册](../cosmos_framework/configs/base/defaults/model.py#L66,1)。

这也解释了为什么仅在 TOML 中写 `causal_training_strategy="teacher_forcing"` 并不足以切换 causal 模型：模型 config group 也必须从 `mot_fsdp` 换为 `mot_causal_fsdp`。当前主线没有做这个切换。

### 3.3 “默认类”其实有两类

容易混淆的两类默认配置是：

- `sft_config.py` 中的 Pydantic 类：定义 TOML 可以写什么、如何校验、schema 默认是什么。
- `model_config.py` 中的 attrs 类：是实际传给 `OmniMoTModel` 的运行时配置对象。

运行时模型默认类是
[OmniMoTModelConfig](../cosmos_framework/configs/base/defaults/model_config.py#L127,1)，
其中 precision、VLM/diffusion 配置和输入 key 分别在
[precision](../cosmos_framework/configs/base/defaults/model_config.py#L136,1)、
[模型子配置](../cosmos_framework/configs/base/defaults/model_config.py#L160,1)
和[数据 key](../cosmos_framework/configs/base/defaults/model_config.py#L170,1)。

Edge 的具体模型值来自
[EDGE_MODEL_CONFIG](../cosmos_framework/configs/base/experiment/sft/models/edge_model_config.py#L34,1)：

- `vision_gen=True`，`sound_gen=False`。
- Vision SFT 再将 `action_gen` 改为 `False`，见[实验覆盖](../cosmos_framework/configs/base/experiment/sft/vision_sft_edge.py#L54,1)。
- `causal_training_strategy="none"`。
- `joint_attn_implementation="two_way"`。
- Wan VAE 空间压缩 16、时间压缩 4，见[tokenizer 配置](../cosmos_framework/configs/base/experiment/sft/models/edge_model_config.py#L127,1)。
- backbone 是 `Nemotron3DenseVLTextForCausalLM`，构造位置见[Edge VLM model_instance](../cosmos_framework/configs/base/experiment/sft/models/edge_model_config.py#L139,1)。

### 3.4 当前关键有效值

| 配置                  |               当前有效值 | 最终来源                    |
| --------------------- | -----------------------: | --------------------------- |
| model class           |           `OmniMoTModel` | `mot_fsdp` group            |
| backbone              | Nemotron 2B Dense VL MoT | `EDGE_MODEL_CONFIG`         |
| causal strategy       |                   `none` | Edge model config           |
| model precision       |               `bfloat16` | TOML                        |
| distributed mode      |                   `fsdp` | experiment/TOML             |
| torch compile         |                  `false` | 当前 TOML                   |
| activation checkpoint |                   `full` | TOML                        |
| optimizer class       |      `torch.optim.AdamW` | experiment 的 `adamw` group |
| optimizer fused 参数  |                   `true` | 当前 TOML                   |
| learning rate         |                   `1e-4` | 当前 TOML 覆盖实验的 `5e-4` |
| grad accumulation     |                        2 | TOML                        |
| optimizer steps       |                      500 | TOML                        |
| checkpoint interval   |                100 steps | TOML                        |
| packed token budget   |                    45056 | TOML/experiment             |
| caption token cap     |                     2048 | TOML/experiment             |

`adamw` group 明确选择 `optimizer_type="AdamW"`，见
[AdamW config group](../cosmos_framework/configs/base/defaults/optimizer.py#L108,1)，
最后分发到 `torch.optim.AdamW`，见
[_optimizer_cls](../cosmos_framework/utils/generator/optimizer.py#L42,1)。

## 4. 第三站：Python 训练入口怎样实例化一切

命令行入口在
[train.py 主函数](../cosmos_framework/scripts/train.py#L232,1)。
它先在第 286 行加载 TOML 合并配置，然后进入 `launch()`：

1. 初始化 distributed。
2. 校验并冻结最终配置。
3. 构造 trainer。
4. 在 `model_init()` 中执行 `instantiate(config.model)`。
5. 实例化 train/val dataloader。
6. 调用 `trainer.train()`。

对应核心代码是
[launch](../cosmos_framework/scripts/train.py#L184,1)
和[对象实例化及训练启动](../cosmos_framework/scripts/train.py#L203,1)。

如果只想确认最终配置，不加载模型，可从 `cosmos-framework` 根目录运行：

```bash
conda activate cosmos-framework
export DATASET_PATH=/path/to/sft_dataset_bridge
export BASE_CHECKPOINT_PATH=/path/to/Cosmos3-Edge-DCP
export WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth

python -m cosmos_framework.scripts.train \
  --sft-toml=/mi/data2T/Embodied-AI/codes/cosmos_ascend/cosmos/cookbooks/cosmos3/generator/audiovisual/finetune/toml/sft_config/vision_sft_edge.toml \
  --dryrun
```

`--dryrun` 打印最终配置并保存 `config.yaml`，实现见
[dryrun 分支](../cosmos_framework/scripts/train.py#L292,1)。

## 5. 第四站：数据从 JSONL 到训练 batch

### 5.1 数据管线在哪里定义

完整 dataloader 对象图不在 TOML，而在实验文件的
[dataloader_train 定义](../cosmos_framework/configs/base/experiment/sft/vision_sft_edge.py#L229,1)：

```python
PackingDataLoader(
    max_sequence_length=45056,
    dataloader=RankPartitionedDataLoader(
        batch_size=1,
        datasets={
            "video": {
                "ratio": 1,
                "dataset": get_sft_dataset(...),
            }
        },
    ),
)
```

这里的 `batch_size=1` 是内层 PyTorch DataLoader 每次取一个样本，不代表外层训练 batch 永远只有一条视频。外层 `PackingDataLoader` 会按 45056 token 的预算贪心拼入多条样本，直到下一条放不下。实现见
[PackingDataLoader 定义](../cosmos_framework/data/generator/joint_dataloader.py#L852,1)
和[packing 循环](../cosmos_framework/data/generator/joint_dataloader.py#L923,1)。

`RankPartitionedDataLoader` 的作用是按数据集 ratio 把不同 rank 分给不同数据集，然后给被选中的 dataset 写入 shard 信息。当前只有 `video: ratio=1`，所以所有 rank 都读 video dataset。入口见
[RankPartitionedDataLoader](../cosmos_framework/data/generator/joint_dataloader.py#L724,1)。

### 5.2 JSONL 如何变成一个样本

数据集工厂是
[get_sft_dataset](../cosmos_framework/data/generator/local_datasets/sft_dataset.py#L577,1)。
它逐行读取 JSONL，保留视频路径、宽高以及符合要求的 `t2w_windows`，见
[JSONL metadata 解析](../cosmos_framework/data/generator/local_datasets/sft_dataset.py#L478,1)。

可以用下面这个缩略结构理解一行 JSONL；字段内容以实际数据为准：

```json
{
  "uuid": "video-id",
  "vision_path": "relative/or/absolute/video.mp4",
  "width": 640,
  "height": 480,
  "duration": 8.0,
  "t2w_windows": [
    {
      "start_frame": 0,
      "end_frame": 92,
      "temporal_interval": 1,
      "caption": "..."
    }
  ]
}
```

每次迭代并不是固定取视频首段。`process_one_sample()` 会从该视频的 `t2w_windows` 中随机选一个窗口，见
[随机窗口选择](../cosmos_framework/data/generator/local_datasets/sft_dataset.py#L179,1)。

当前实验设置：

```python
num_video_frames = -1
frame_selection_mode = "first"
sample_by_window = False
temporal_interval_mode = "max_30fps"
```

但在 `num_video_frames=-1` 的 native chunk 模式下，会直接采用窗口自己的 `start_frame/end_frame/temporal_interval`，因此 `frame_selection_mode="first"` 和重新计算 interval 的分支并不会参与这条路径。代码见
[native chunk 分支](../cosmos_framework/data/generator/local_datasets/sft_dataset.py#L211,1)。

解码完成后，帧数会向下截断为：

```python
target_t = ((decoded_t - 1) // 4) * 4 + 1
```

因此送入 Wan VAE 的像素帧数满足 `1 + 4N`。实现见
[时间对齐](../cosmos_framework/data/generator/local_datasets/sft_dataset.py#L272,1)。

### 5.3 T2V、I2V、V2V 是怎样混合的

实验配置使用：

```python
conditioning_config = {0: 0.7, 1: 0.2, 2: 0.1}
```

定义位置在
[conditioning_config](../cosmos_framework/configs/base/experiment/sft/vision_sft_edge.py#L249,1)。
dataset 会按这个分布随机选 `num_cond`，并把最前面的 latent frame index 写入 `SequencePlan.condition_frame_indexes_vision`，见
[SequencePlan 构造](../cosmos_framework/data/generator/local_datasets/sft_dataset.py#L348,1)。

| num_cond | 训练形式 | 条件内容                               |
| -------: | -------- | -------------------------------------- |
|        0 | T2V，70% | 没有 clean vision latent，仅文本条件   |
|        1 | I2V，20% | 第 1 个 VAE latent frame 为 clean 条件 |
|        2 | V2V，10% | 前 2 个 VAE latent frame 为 clean 条件 |

Wan VAE 时间压缩因子是 4。这里的 2 个 latent frame 对应从像素时间轴开头覆盖 5 帧的因果编码范围，即 `1 + (5-1)/4 = 2`，因此实验注释写的是 “first 5 frames / 2 latent frames”。

最终 dataset 样本包含 `video`、`ai_caption`、`text_token_ids`、`padding_mask`、`image_size`、`sequence_plan` 等字段，构造位置在
[样本返回字典](../cosmos_framework/data/generator/local_datasets/sft_dataset.py#L329,1)。

## 6. 第五站：模型定义和对象层次

最外层训练模型是
[OmniMoTModel](../cosmos_framework/model/generator/omni_mot_model.py#L82,1)。
它不是单纯的一层 Transformer，而是训练编排器，负责：

- 建立文本 processor 和 Wan VAE tokenizer。
- 建立 FSDP/并行配置。
- 建立真正的 denoiser network。
- 把像素、文本和条件计划转成 packed sequence。
- 采样噪声、调用 network、计算 loss。

初始化顺序可看
[OmniMoTModel.__init__](../cosmos_framework/model/generator/omni_mot_model.py#L88,1)，
tokenizer 构造在
[set_up_tokenizers](../cosmos_framework/model/generator/omni_mot_model.py#L127,1)。

真正的网络对象在 `build_net()` 中建立：

```python
language_model = lazy_instantiate(self.vlm_config.model_instance)
network_config = Cosmos3VFMNetworkConfig(...)
net = Cosmos3VFMNetwork(language_model=language_model, config=network_config)
net = parallelize_vfm_network(net, ...)
```

对应
[build_net](../cosmos_framework/model/generator/omni_mot_model.py#L191,1)。
因此对象关系是：

```text
OmniMoTModel                         训练流程、VAE、噪声与 loss
└─ Cosmos3VFMNetwork                多模态 encode/pack/decode
   └─ Nemotron3DenseVLTextForCausalLM
      └─ Nemotron3DenseVLTextModel
         └─ N × MoTDecoderLayer
            ├─ understanding/reasoner 参数
            └─ generation 参数（名字通常带 _moe_gen）
```

注意：类名里的 `ForCausalLM` 表示它继承的语言模型接口和 reasoner 文本能力，不等价于当前视频训练启用了 causal teacher forcing。视频训练是否 causal 由 model group、`causal_training_strategy` 和 packing/attention 路径共同决定。

## 7. 第六站：一条训练 batch 的完整 forward

真正应首先逐行阅读的是
[OmniMoTModel.training_step](../cosmos_framework/model/generator/omni_mot_model.py#L816,1)，而不是寻找一个传统的 `OmniMoTModel.forward()`。

### 7.1 文本、clean latent 和 sequence plan

`training_step()` 先读取/tokenize caption，取得或补全 `SequencePlan`，再通过 Wan VAE 把 `[C,T,H,W]` uint8 视频变成 clean latent `x0`。对应
[文本与 clean data](../cosmos_framework/model/generator/omni_mot_model.py#L843,1)。

在普通 Edge SFT 中，`SequencePlan` 只决定哪些开头 latent frame 是 clean 条件；它不是 Scheme-B 的 clean/noisy 双序列 teacher forcing。

### 7.2 采样 sigma、pack 并添加噪声

模型按样本分辨率采样 RF timestep/sigma，见
[vision noise level](../cosmos_framework/model/generator/omni_mot_model.py#L880,1)，
然后把文本和 generation token 打包，见
[pack input sequence](../cosmos_framework/model/generator/omni_mot_model.py#L961,1)。

对每个 vision latent，clean 条件位置的有效 sigma 会乘上 `1-condition_mask`，所以条件帧 sigma 为 0；其余帧按采样 sigma 加噪。实现见
[_add_noise_to_input](../cosmos_framework/model/generator/omni_mot_model.py#L1492,1)
和[vision noising](../cosmos_framework/model/generator/omni_mot_model.py#L1539,1)。

Rectified Flow 的数学形式是：

```python
xt = epsilon * sigma + x0 * (1 - sigma)
target_velocity = epsilon - x0
```

原始实现见
[RectifiedFlow.get_interpolation](../cosmos_framework/model/generator/diffusion/rectified_flow.py#L176,1)。
其中 `sigma=0` 得到 clean `x0`，`sigma=1` 得到纯噪声 `epsilon`。

### 7.3 从 denoise 到最底层 Transformer 循环

外层 network 调用位于
[training_step 的 denoise 调用](../cosmos_framework/model/generator/omni_mot_model.py#L1032,1)。
`denoise()` 本身只是把 `PackedSequence` 交给 `self.net` 并整理返回字典，见
[OmniMoTModel.denoise](../cosmos_framework/model/generator/omni_mot_model.py#L4271,1)。

`Cosmos3VFMNetwork.forward()` 是真正的多模态 forward：

1. 把文本 token embedding 写入统一 hidden sequence。
2. 把 vision latent patchify/project 后写到 generation 位置。
3. 构造 two-way attention metadata 和 position ids。
4. 调用 Nemotron language model。
5. 将 generation hidden state 投影、unpack 回 vision velocity tensor。

入口见
[Cosmos3VFMNetwork.forward](../cosmos_framework/model/generator/mot/cosmos3_vfm_network.py#L928,1)，
attention pack 在
[build_packed_sequence](../cosmos_framework/model/generator/mot/cosmos3_vfm_network.py#L1040,1)，
backbone 调用和 vision decode 在
[language_model forward](../cosmos_framework/model/generator/mot/cosmos3_vfm_network.py#L1122,1)。

Edge wrapper 的 `forward()` 继续转给 `Nemotron3DenseVLTextModel`，见
[Nemotron3DenseVLTextForCausalLM.forward](../cosmos_framework/model/generator/mot/unified_mot.py#L2406,1)。
TextModel 是一层薄包装，见
[Nemotron3DenseVLTextModel.forward](../cosmos_framework/model/generator/mot/unified_mot.py#L1337,1)。

最底层值得下断点的 Transformer 主循环是
[_impl_forward](../cosmos_framework/model/generator/mot/unified_mot.py#L887,1)：

```python
for i, decoder_layer in enumerate(self.layers):
    hidden_states, lbl_metadata, kv_to_store = decoder_layer(
        hidden_states,
        attention_mask,
        position_embeddings,
        ...,
    )
```

每层的双 pathway layer 是
[MoTDecoderLayer.forward](../cosmos_framework/model/generator/mot/unified_mot.py#L1068,1)，
真正生成 understanding/generation 两套 Q/K/V 的 attention 层在
[PackedAttentionMoT.forward](../cosmos_framework/model/generator/mot/unified_mot.py#L584,1)。

## 8. 第七站：loss 到底在哪里设置、怎么算

### 8.1 数学 loss

独立的 flow-matching loss 函数是
[compute_flow_matching_loss](../cosmos_framework/model/generator/algorithm/loss/flow_matching.py#L18,1)。
它先计算：

```python
sqerr = (predicted_velocity - (epsilon - x0)) ** 2
noisy_mask = 1 - condition_mask
loss_i = (sqerr * noisy_mask).mean()
loss = mean(time_weight * loss_i over batch)
```

关键语义是 `condition_mask=1` 表示 clean 条件位置，这些位置被 `1-condition_mask` 从 loss 中屏蔽；`condition_mask=0` 才是需要预测的 noisy 位置。源码见
[平方误差与 noisy mask](../cosmos_framework/model/generator/algorithm/loss/flow_matching.py#L61,1)。

当前 `normalize_loss_by_active=False`，所以使用整个 tensor 的 `.mean()`：条件帧越多，有效 noisy 元素占比越低，该样本的 loss 量级也会相应变小。这个开关的运行时默认与解释在
[RectifiedFlowTrainingConfig](../cosmos_framework/configs/base/defaults/model_config.py#L59,1)。

### 8.2 当前 Edge 总 loss

各模态 loss 的组合在
[_compute_losses](../cosmos_framework/model/generator/omni_mot_model.py#L1145,1)。
当前 Vision SFT 中：

- `vision_gen=True`，所以计算 vision flow-matching loss。
- `action_gen=False`，action loss 为记录用的 0。
- `sound_gen=False`，sound loss 为记录用的 0。
- Edge 的 `loss_scale=10.0`、`image_loss_scale=None`，见
  [Edge RF training config](../cosmos_framework/configs/base/experiment/sft/models/edge_model_config.py#L102,1)。
- Dense Nemotron 通常没有 MoE load-balancing loss；代码仍保留通用的 auxiliary loss 接口。

所以当前主损失可以近似写为：

```text
total_loss = 10.0 * vision_flow_matching_loss
```

精确的 vision scale 选择和累加见
[vision loss scaling](../cosmos_framework/model/generator/omni_mot_model.py#L1172,1)，
可选 auxiliary loss 的累加见
[load-balancing loss](../cosmos_framework/model/generator/omni_mot_model.py#L1261,1)。

## 9. 第八站：backward、梯度累积与 AdamW 更新

通用 Trainer 在
[Trainer.train](../cosmos_framework/trainer/__init__.py#L195,1)
中建立 optimizer、scheduler 和 GradScaler，加载 DCP，然后反复取 batch。数据搬到设备和 training step 的位置见
[主训练循环](../cosmos_framework/trainer/__init__.py#L254,1)。

一次 micro-batch 的核心过程是：

```python
output_batch, loss = model_ddp.training_step(data, iteration)
loss_scaled = grad_scaler.scale(loss / grad_accum_iter)
loss_scaled.backward()

# 累积满 2 个 micro-batch 后
grad_scaler.step(optimizer)
grad_scaler.update()
scheduler.step()
optimizer.zero_grad(set_to_none=True)
```

真实代码见
[Trainer.training_step](../cosmos_framework/trainer/__init__.py#L338,1)
和[optimizer step](../cosmos_framework/trainer/__init__.py#L400,1)。

当前 `grad_accum_iter=2`，所以 TOML 的 `max_iter=500` 指 500 次 optimizer update，而不是只读取 500 个 micro-batch。正常情况下会执行约 1000 个 micro-batch；packing 后每个 micro-batch 又可能包含多条视频，因此它也不等于视频样本数。

## 10. 推荐的实际导读与断点顺序

第一次调试建议按以下顺序，只观察一个 batch：

1. [train.py:286](../cosmos_framework/scripts/train.py#L286,1)：检查合并后的 `config`，确认 model target 是 `OmniMoTModel`。
2. [train.py:216](../cosmos_framework/scripts/train.py#L216,1)：观察 model 实例化。
3. [sft_dataset.py:179](../cosmos_framework/data/generator/local_datasets/sft_dataset.py#L179,1)：看随机选中的 `t2w_window`。
4. [sft_dataset.py:329](../cosmos_framework/data/generator/local_datasets/sft_dataset.py#L329,1)：记录 `video.shape`、caption、`sequence_plan`。
5. [joint_dataloader.py:935](../cosmos_framework/data/generator/joint_dataloader.py#L935,1)：看单样本 token 数以及一个 packed batch 放入几条视频。
6. [omni_mot_model.py:816](../cosmos_framework/model/generator/omni_mot_model.py#L816,1)：进入模型训练编排。
7. [omni_mot_model.py:1020](../cosmos_framework/model/generator/omni_mot_model.py#L1020,1)：比较 `x0`、`sigma`、condition mask、`xt` 和 target velocity。
8. [cosmos3_vfm_network.py:928](../cosmos_framework/model/generator/mot/cosmos3_vfm_network.py#L928,1)：检查统一序列中的 text/vision token 数。
9. [unified_mot.py:941](../cosmos_framework/model/generator/mot/unified_mot.py#L941,1)：只进入第 0 个 decoder layer，理解双 pathway 后先跳出循环。
10. [flow_matching.py:66](../cosmos_framework/model/generator/algorithm/loss/flow_matching.py#L66,1)：确认 clean 条件位置的 loss 被屏蔽。
11. [trainer/__init__.py:379](../cosmos_framework/trainer/__init__.py#L379,1)：观察 loss 除以 2 后 backward，并确认第二个 micro-batch 才 optimizer step。

在断点处可用下面的示范表达式快速检查数据；不要把它永久写进训练代码：

```python
# dataset 返回后
print(data["video"].shape)
print(data["num_frames"])
print(data.get("sequence_plan"))

# model.training_step 中
print([x.shape for x in gen_data_clean.x0_tokens_vision])
print([m.flatten().tolist() for m in packed_sequence.vision.condition_mask])
print(timesteps_vision.shape, sigmas_vision.shape)

# network.forward 中
print(packed_seq.sequence_length)
print(packed_seq.split_lens, packed_seq.sample_lens)

# loss 中
print(float(fm_loss_vision), float(total_loss))
```

## 11. 阅读时最常见的误区

- **把 SFT 写成 STF。** 代码、脚本和 TOML 都使用 SFT，即 supervised fine-tuning。
- **认为 `ForCausalLM` 就是视频 causal 训练。** 它是 backbone 接口名；当前视频策略仍是 `none`。
- **认为 TOML 是完整配置。** TOML 只是对 Python 实验对象图的一组已校验覆盖。
- **认为内层 `batch_size=1` 就只能训练一条视频。** 外层 PackingDataLoader 仍可把多条视频打进一个 batch。
- **认为当前每条视频固定帧数。** 当前随机选 `t2w_window` 并使用它的原生跨度，最后只保证 `1+4N` 对齐。
- **把像素条件帧数和 latent 条件帧数混为一谈。** `SequencePlan` 中的 1/2 是 Wan VAE latent frame 数。
- **直接从 attention kernel 开始读。** 在没弄懂 packed sequence 的 text/gen index 和 condition mask 之前，attention metadata 很难读懂。

## 12. 一条最短但有价值的学习路线

如果时间有限，只读下面八处即可形成闭环：

1. [本机 launcher](../../cosmos/cookbooks/cosmos3/generator/audiovisual/finetune/launch_sft_vision_edge_local.sh#L1,1)
2. [正式 launcher 的 torchrun](../../cosmos/cookbooks/cosmos3/generator/audiovisual/finetune/launch_sft_vision_edge.sh#L38,1)
3. [Edge TOML](../../cosmos/cookbooks/cosmos3/generator/audiovisual/finetune/toml/sft_config/vision_sft_edge.toml#L16,1)
4. [vision_sft_edge 实验对象图](../cosmos_framework/configs/base/experiment/sft/vision_sft_edge.py#L62,1)
5. [dataset 单样本处理](../cosmos_framework/data/generator/local_datasets/sft_dataset.py#L179,1)
6. [OmniMoTModel.training_step](../cosmos_framework/model/generator/omni_mot_model.py#L816,1)
7. [Cosmos3VFMNetwork.forward](../cosmos_framework/model/generator/mot/cosmos3_vfm_network.py#L928,1)
8. [flow-matching loss](../cosmos_framework/model/generator/algorithm/loss/flow_matching.py#L18,1)

读完这条最短路线后，再进入 `unified_mot.py` 的 decoder-layer 与 attention 实现，便能把每个 tensor 与上游的文本、clean latent、noisy latent 和 loss mask 对应起来。
