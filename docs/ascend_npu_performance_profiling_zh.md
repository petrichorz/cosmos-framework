# Cosmos3 Ascend NPU 性能采集与实验报告

本文是 Cosmos3 视觉 SFT 在 Ascend NPU 上的统一性能文档，既说明如何使用
`tools/profile_cosmos_ascend.py` 完成端到端分析，也集中记录已经完成的优化实验。内容覆盖
数据读取、正常训练基线、单 rank 算子、多 rank 通信，以及 EgoSuite 融合算子、
`max_video_duration_s` 和超长 Episode 均衡重叠切片的 A/B 结果。

分析应始终按以下顺序进行：

```text
数据读取是否健康
    -> 数据是否真的阻塞训练
    -> Forward/Backward 中哪些算子耗时
    -> 多卡负载是否均衡、通信是否暴露
```

不要直接从几百 MB 的 `trace_view.json` 开始分析。没有训练基线时，Profiler 中的耗时无法判断是否影响正常训练吞吐。

## 1. 工具提供的四个阶段

| 阶段 | `--mode` | 回答的问题 | 主要产物 |
| :--- | :--- | :--- | :--- |
| 1 | `data` | 视频能否稳定读取，解码吞吐和缓存命中率如何 | `data_result.json`、事件摘要 |
| 2 | `baseline` | 正常训练中数据、Forward、Backward、Optimizer 各占多久 | JSONL 事件和基线摘要 |
| 3 | `npu` | 单个 rank 上哪些 PyTorch 算子和 Ascend kernel 最耗时 | rank 0 的 Ascend trace、算子表、kernel 表 |
| 4 | `distributed` | 8 个 rank 是否负载不均，FSDP 通信有多少未被计算掩盖 | 所有 rank 的 trace 和 `step_trace_time.csv` |

也可以使用 `--mode all` 顺序运行四个阶段，但第一次在新机器上分析时建议分开运行：每完成一个阶段先检查结果，再决定是否继续产生体积较大的算子 trace。

## 2. 环境准备

### 2.1 获取性能分析分支

```bash
git clone https://github.com/petrichorz/cosmos-framework.git
cd cosmos-framework
git switch perf/vision-edge-ascend-profile
git pull --ff-only
```

如果仓库已经存在：

```bash
git fetch origin
git switch perf/vision-edge-ascend-profile
git pull --ff-only
```

### 2.2 激活 CANN 和 Conda 环境

根据机器上的实际安装路径调整：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /data5T/apps/miniconda3/etc/profile.d/conda.sh
conda activate cosmos-framework
pip install -e .
```

检查关键依赖：

```bash
python -c "import torch, torch_npu, torchcodec; print(torch.__version__); print(torch_npu.__version__)"
npu-smi info
```

需要确保：

- CANN、PyTorch 和 `torch_npu` 版本兼容。
- TorchCodec 已正确安装；本次验证使用了 `v0.10.0`。
- 数据集和 Cosmos3-Edge checkpoint 在本机可访问。
- 训练前 8 张 NPU 没有被其他进程占用。
- `MASTER_PORT` 没有被其他分布式任务占用。
- 输出磁盘有足够空间。单 rank Profile 约需 1 GiB，全 8 rank Profile 可能需要 8 GiB 或更多。

### 2.3 确认启动脚本

工具默认使用项目相邻目录中的：

```text
cosmos/cookbooks/cosmos3/generator/audiovisual/finetune/launch_sft_vision_edge_profile_local.sh
```

不同机器的目录结构可能不同，推荐显式指定：

```bash
--launch-script /实际路径/launch_sft_vision_edge_profile_local.sh
```

查看全部参数：

```bash
python tools/profile_cosmos_ascend.py --help
```

## 3. 推荐的公共参数

下面的例子使用：

```text
数据集：/data5T/Embodied-AI/datasets/Cosmos3-DROID/success
视频后端：TorchCodec
缩放方式：解码时缩放
最大视频 FPS：15
DataLoader workers：8
每个 worker 的 decoder LRU cache：64
输出根目录：/data5T/profile-results
```

如果需要向训练配置追加 Hydra override，可重复使用：

```bash
--extra-override key=value
```

## 4. 第一阶段：数据读取基准

运行：

```bash
python tools/profile_cosmos_ascend.py \
  --mode data \
  --dataset /data5T/Embodied-AI/datasets/Cosmos3-DROID/success \
  --video-backend torchcodec \
  --video-resize-mode decode_transform \
  --data-samples 64 \
  --num-workers 8 \
  --decoder-cache-size 64 \
  --max-video-fps 15 \
  --output-root /data5T/profile-results
```

输出目录类似：

```text
ascend_profile_YYYYMMDD_HHMMSS/data/
├── data_result.json
├── events/
│   └── events_*.jsonl
└── summary/
    ├── performance_summary.csv
    ├── performance_summary.json
    └── performance_summary.md
```

### 4.1 阅读顺序

1. `data_result.json`
2. `summary/performance_summary.md`
3. 出现失败或长尾时再看 `events/*.jsonl`

### 4.2 重点指标

| 指标 | 含义 |
| :--- | :--- |
| `successes` / `failures` | 成功和失败样本数 |
| `samples_per_s` | 多 worker 的整体样本吞吐 |
| `frames_per_s` | 多 worker 的整体帧吞吐 |
| `video_decode_torchcodec` | TorchCodec 取帧耗时 |
| `decoder_init_exact` | 新建精确 seek decoder 的耗时 |
| `decoder_cache_hit/miss/eviction` | 每个 worker 的 decoder LRU 行为 |
| `video_metadata` | MP4 metadata 读取耗时 |
| `video_tensor_prepare` | NumPy/Torch tensor 整理和裁剪耗时 |

这一阶段只能说明解码本身是否昂贵，不能单独证明数据是训练瓶颈。多个 worker 的解码时间互相重叠，不能将所有 `video_decode_torchcodec` 耗时相加后与训练 step wall time 比较。

## 5. 第二阶段：无专业 Profiler 的训练基线

运行 12 步，以便排除首步初始化并观察稳态：

```bash
MASTER_PORT=50124 python tools/profile_cosmos_ascend.py \
  --mode baseline \
  --dataset /data5T/Embodied-AI/datasets/Cosmos3-DROID/success \
  --video-backend torchcodec \
  --video-resize-mode decode_transform \
  --max-steps 12 \
  --num-workers 8 \
  --decoder-cache-size 64 \
  --max-video-fps 15 \
  --output-root /data5T/profile-results
```

性能专用运行会设置 `COSMOS_PERF_SKIP_FINAL_CHECKPOINT=1`，避免短基准结束时写出大型 checkpoint。正常训练不受该设置影响。

### 5.1 阅读顺序

1. `baseline/training.log`：确认训练正常结束，观察每个 iteration 的 wall time。
2. `baseline/summary/performance_summary.md`：查看所有 rank 汇总。
3. `baseline/events/events_rank*_worker-1_*.jsonl`：需要逐 rank、逐 iteration 分析时使用。
4. worker ID 非 `-1` 的 JSONL：分析具体数据 worker 的解码和缓存行为。

### 5.2 重点 scope

| Scope | 含义 |
| :--- | :--- |
| `iteration_core` | 从开始取数据到完成一次有效 optimizer update |
| `training_step` | 模型训练主体 |
| `forward` | 前向计算 |
| `backward` | 反向计算，可能包含 FSDP 通信等待 |
| `optimizer_step` | 梯度处理、参数更新和清梯度 |
| `data_wait` | 主训练进程等待 DataLoader 的时间 |
| `host_to_device` | batch 从 host 搬到 NPU 的时间 |

### 5.3 判断数据是否为瓶颈

- `video_decode_torchcodec` 很高，但 `data_wait` 只有毫秒级：解码被多 worker 预取隐藏，不在关键路径上。
- `data_wait` 达到秒级或频繁出现长尾：数据供给不足，需要检查磁盘、解码、worker 数量、缓存和坏视频重试。
- `host_to_device` 高：检查 batch 体积、pin memory、重复拷贝和 tensor 整理。

第一步常包含初始化、内存分配或图编译开销。正常吞吐建议至少排除前两步再计算均值和 P50。

## 6. 第三阶段：rank 0 Ascend 算子 Profile

先只采集一个 rank，以较低磁盘和采集开销定位主要算子：

```bash
MASTER_PORT=50125 python tools/profile_cosmos_ascend.py \
  --mode npu \
  --dataset /data5T/Embodied-AI/datasets/Cosmos3-DROID/success \
  --video-backend torchcodec \
  --video-resize-mode decode_transform \
  --max-steps 8 \
  --profile-step 8 \
  --profile-warmup 2 \
  --profile-active-steps 1 \
  --profiler-level level0 \
  --record-shapes \
  --no-with-stack \
  --no-with-modules \
  --num-workers 8 \
  --decoder-cache-size 64 \
  --output-root /data5T/profile-results
```

调度为 `wait + warmup + active`，并且只执行一个采集窗口（`repeat=1`）。上述设置中：

```text
wait   = profile_step - profile_warmup - profile_active_steps
       = 8 - 2 - 1
       = 5

step 1～5：wait，不记录
step 6～7：warmup，采集但不导出
step 8：active，正式保存
```

`--profile-step` 表示第一个采集窗口的结束 step，不是开始 step。

### 6.1 为什么默认关闭 stack 和 modules

`with_stack=true` 和 `with_modules=true` 会显著增加 Python 事件量、内存和落盘开销。在长序列视觉训练中可能造成原始 Profile 截断。默认使用：

```text
record_shapes=true
with_stack=false
with_modules=false
```

只有当算子无法通过名称和模块范围反推代码，而且已缩小采集范围时，才单独尝试 stack capture。

### 6.2 阅读顺序

1. `npu/summary/performance_summary.md`
2. `npu/summary/ascend_hotspots.md`
3. `npu/summary/ascend_kernel_hotspots.csv`
4. `npu/summary/ascend_operator_hotspots.csv`
5. 原始 `kernel_details.csv`
6. 原始 `operator_details.csv`
7. 最后打开 `trace_view.json`

查找产物：

```bash
find /data5T/profile-results -name analyse.done
find /data5T/profile-results -name trace_view.json
find /data5T/profile-results -name operator_details.csv
find /data5T/profile-results -name kernel_details.csv
find /data5T/profile-results -name step_trace_time.csv
```

出现 `analyse.done` 才表示 CANN 数据解析完整结束。训练日志显示 `Done with training`，但异步解析进程还在运行时，不要移动或删除 raw profiler 目录。

### 6.3 Profiler level 和 AI Core metric

- `level0`：适合第一次定位 timeline、算子和通信热点。
- `level1` / `level2`：用于进一步查看 AI Core 指标，采集成本更高。
- `--aic-metrics` 可选 `pipe`、`arithmetic`、`memory` 或 `l2cache`。
- Level 0 只能配合 `--aic-metrics none`。

例如在已经确认某个 step 后采集 Memory 指标：

```bash
MASTER_PORT=50127 python tools/profile_cosmos_ascend.py \
  --mode npu \
  --dataset /data5T/Embodied-AI/datasets/Cosmos3-DROID/success \
  --max-steps 8 \
  --profile-step 8 \
  --profile-warmup 2 \
  --profile-active-steps 1 \
  --profiler-level level1 \
  --aic-metrics memory \
  --no-with-stack \
  --no-with-modules \
  --output-root /data5T/profile-results
```

一次只采集一种 AI Core metric，避免不必要的开销和结果混淆。

### 6.4 连续采集多个 active step

周期性或间歇性问题可能无法通过单个 step 捕获。使用 `--profile-active-steps` 可以在一个窗口中保留连续多个 step。

例如连续采集第 8～10 步：

```bash
MASTER_PORT=50128 python tools/profile_cosmos_ascend.py \
  --mode distributed \
  --dataset /data5T/Embodied-AI/datasets/Cosmos3-DROID/success \
  --video-backend torchcodec \
  --video-resize-mode decode_transform \
  --max-steps 10 \
  --profile-step 10 \
  --profile-warmup 2 \
  --profile-active-steps 3 \
  --profiler-level level0 \
  --record-shapes \
  --no-with-stack \
  --no-with-modules \
  --num-workers 8 \
  --decoder-cache-size 64 \
  --output-root /data5T/profile-results
```

对应调度为：

```text
wait   = 10 - 2 - 3 = 5

step 1～5：wait
step 6～7：warmup
step 8～10：active，三个 step 均保留
```

如果希望采集窗口仍结束在第 8 步，则使用：

```bash
--max-steps 8 \
--profile-step 8 \
--profile-warmup 2 \
--profile-active-steps 3
```

此时采集第 6～8 步。`--max-steps` 必须不小于 `--profile-step`，同时 `--profile-step` 必须不小于 `--profile-warmup + --profile-active-steps`。

全 8 rank 连续采集 3 步可能产生 20～25 GiB 或更多数据。建议先使用 Level 0 连续采集定位异常 step，再用 Level 1 对单个代表性 step 采集通信矩阵和详细通信数据。

## 7. 第四阶段：全 rank 分布式 Profile

单 rank 热点明确后，再采集 8 个 rank：

```bash
MASTER_PORT=50126 python tools/profile_cosmos_ascend.py \
  --mode distributed \
  --dataset /data5T/Embodied-AI/datasets/Cosmos3-DROID/success \
  --video-backend torchcodec \
  --video-resize-mode decode_transform \
  --max-steps 8 \
  --profile-step 8 \
  --profile-warmup 2 \
  --profile-active-steps 1 \
  --profiler-level level0 \
  --record-shapes \
  --no-with-stack \
  --no-with-modules \
  --num-workers 8 \
  --decoder-cache-size 64 \
  --output-root /data5T/profile-results
```

确认所有 rank 都完成解析：

```bash
find /data5T/profile-results \
  -path '*distributed*' \
  -name analyse.done | wc -l
```

8 卡任务应输出 `8`。同时检查：

```bash
find /data5T/profile-results \
  -path '*distributed*' \
  -name trace_view.json | wc -l
```

### 7.1 首先看 `step_trace_time.csv`

| 列 | 含义 |
| :--- | :--- |
| `Stage` | 被采集 step 的整体时间 |
| `Computing` | NPU 计算时间 |
| `Communication` | 通信总时间，可能与计算重叠 |
| `Communication(Not Overlapped)` | 未被计算掩盖、真正延长 step 的通信时间 |
| `Overlapped` | 与计算重叠的通信时间 |
| `Free` | 无计算和通信活动的空闲时间 |
| `Preparing` | 准备阶段时间 |

近似关系为：

```text
Stage ~= Computing + Communication(Not Overlapped) + Free + Preparing
```

`Communication` 可以与 `Computing` 重叠，因此两者百分比之和可能超过 100%，这是正常现象。

通信重叠率可计算为：

```text
Overlapped / Communication
```

### 7.2 不要只比较各 rank 的 Stage

collective 会同步所有 rank，因此所有 rank 的 `Stage` 很接近并不代表负载均衡。应该逐 rank 比较：

- `Computing`
- `Communication(Not Overlapped)`
- `Free`
- FlashAttention kernel 总时间和输入 shape
- AllGather/ReduceScatter 时间

典型的计算不均衡表现：

```text
重 rank：Computing 较长，collective 等待较少
轻 rank：Computing 较短，未重叠通信或 Free 较长
```

如果 FlashAttention 跨 rank 差异明显，而 MatMul/Conv3D 相对稳定，通常说明 packed sequence 的长度和 Attention FLOPs 不均衡。Attention 复杂度与序列长度平方相关，因此只平衡样本数或 token 总数可能不够，应按估算 FLOPs 分桶和分配。

## 8. 使用浏览器查看 `trace_view.json`

Profiler 模式默认启用 `--mstx-forward`。它不依赖 `--with-stack` 或
`--with-modules`，会在 `cosmos_forward` domain 中记录两层低开销范围：

- `COSMOS::FORWARD/01_TEXT_TOKENIZE` 到 `COSMOS::FORWARD/13_LOSS`：训练
  forward 的数据准备、generation tokenizer/VAE、packing、Host→NPU、denoise 和 loss。
- `COSMOS::DENOISE/01_ENCODE_TEXT` 到 `COSMOS::DENOISE/11_DECODE_SOUND`：
  denoise 内部的模态编码、attention metadata、CP 输入/输出、Transformer 和模态解码。

当 `aten::to` 集中在 vision encoding 或 Transformer 时，可继续展开以下范围：

| 范围 | 对应 Python 位置 |
| :--- | :--- |
| `COSMOS::ENCODE_VISION/02_NOISY_LATENTS_TO_TARGET_DTYPE` | `Cosmos3VFMNetwork._encode_vision`: noisy latent `.to(target_dtype)` |
| `COSMOS::ENCODE_VISION/05_CLEAN_LATENTS_TO_TARGET_DTYPE` | `Cosmos3VFMNetwork._encode_vision`: teacher-forcing clean latent `.to(target_dtype)` |
| `COSMOS::ENCODE_VISION/08_CLEAN_TIMESTEP_EMBED_TO_TARGET_DTYPE` | clean timestep embedding dtype conversion |
| `COSMOS::ENCODE_VISION/10_NOISY_TIMESTEPS_TO_FP32` | `vision.timesteps.to(torch.float32)` |
| `COSMOS::ENCODE_VISION/13_NOISY_TIMESTEP_EMBED_TO_TARGET_DTYPE` | noisy timestep embedding dtype conversion |
| `COSMOS::TRANSFORMER/LAYER_XX/01_PRE_ATTENTION_NORM` | decoder layer input RMSNorm |
| `COSMOS::TRANSFORMER/LAYER_XX/02_SELF_ATTENTION` | decoder layer attention implementation |
| `COSMOS::TRANSFORMER/LAYER_XX/04_PRE_MLP_NORM` | decoder layer post-attention RMSNorm |
| `COSMOS::TRANSFORMER/LAYER_XX/05_MLP_UND` / `06_MLP_GEN` | understanding/generation MLP |
| `COSMOS::RMSNORM/01_HIDDEN_STATES_TO_FP32` | `hidden_states.to(torch.float32)` |
| `COSMOS::RMSNORM/03_WEIGHT_TO_FP32` | `self.weight.to(torch.float32)` |
| `COSMOS::RMSNORM/04_OUTPUT_TO_INPUT_DTYPE` | normalized result `.to(input_dtype)` |
| `COSMOS::ROTARY/01...05` | Nemotron RoPE 中各个 FP32/input-dtype conversion |

`RMSNORM/*` 和 `ROTARY/*` 是行级范围，并嵌套在对应的
`TRANSFORMER/LAYER_XX/*` 范围下，因此可以同时判断转换发生在哪一层、哪条
Python dtype conversion 语句。标签使用函数和语句语义而不是硬编码行号，避免
源码增删后行号漂移。

因此，即使当前 torch_npu 组合开启调用栈会崩溃，也可以保持：

```bash
--mstx-forward --no-with-stack --no-with-modules
```

如果要关闭这些范围，传入 `--no-mstx-forward`。MSTX 打点不会执行 NPU
同步；它把当前 stream 与范围关联起来，适合判断某个 `aten::to`、通信或
NPU kernel 属于哪个 forward 阶段。

推荐使用 Perfetto：

1. 在浏览器访问 `https://ui.perfetto.dev`。
2. 点击 `Open trace file`。
3. 选择某个 rank 的 `ASCEND_PROFILER_OUTPUT/trace_view.json`。
4. 搜索 `COSMOS::FORWARD/`、`COSMOS::DENOISE/`、`COSMOS::BACKWARD`、`FSDP`、`allGather` 或 `reduceScatter`。

建议先打开：

1. 计算时间最长的 rank。
2. 未重叠通信时间最长的 rank。
3. 二者并排对比相同时间段。

trace 可能超过 400 MB，浏览器需要足够内存。不要一次打开 8 份完整 trace。

## 9. 从算子反推代码

建议按以下路径定位：

```text
step_trace_time.csv 确认计算/通信类别
    -> ascend_kernel_hotspots.csv 找 kernel family
    -> operator_details.csv 找对应 PyTorch/NPU operator
    -> trace_view.json 看它位于 Forward、Backward 或 FSDP 范围中的位置
    -> 使用 rg 在源码中搜索 operator/backend 名称
```

常见对应关系：

| Profile 名称 | 代码方向 |
| :--- | :--- |
| `npu_fusion_attention_v3` / `FlashAttentionScore` | `cosmos_framework/model/attention/npu_fusion_attention/` |
| `MatMulV3` | Attention QKV/输出投影、MLP 或其他线性层 |
| `Conv3DV2` | 视频 VAE/Tokenizer 中的 `Conv3d` 或 `CausalConv3d` |
| `TransData` | tensor layout 或 format 转换 |
| `FSDP::all_gather` | FSDP 参数解分片和前向预取 |
| `reduceScatter` | FSDP backward 梯度规约与重新分片 |

搜索示例：

```bash
rg -n "npu_fusion_attention|scaled_dot_product_attention" cosmos_framework
rg -n "Conv3d|conv3d" cosmos_framework/model
rg -n "fully_shard|FSDP|all_gather|reduce_scatter" cosmos_framework
```

Operator duration 可能包含嵌套范围或异步设备工作，不能把所有 operator total 简单相加。计算关键路径优先使用 `step_trace_time.csv` 和 timeline，operator/kernel 表用于热点归因。

## 10. 后台运行和查看进度

长任务可以后台执行：

```bash
nohup bash -lc '
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /data5T/apps/miniconda3/etc/profile.d/conda.sh
conda activate cosmos-framework
MASTER_PORT=50126 python tools/profile_cosmos_ascend.py \
  --mode distributed \
  --dataset /data5T/Embodied-AI/datasets/Cosmos3-DROID/success \
  --video-backend torchcodec \
  --video-resize-mode decode_transform \
  --max-steps 8 \
  --profile-step 8 \
  --profile-warmup 2 \
  --profile-active-steps 1 \
  --profiler-level level0 \
  --record-shapes \
  --no-with-stack \
  --no-with-modules \
  --num-workers 8 \
  --decoder-cache-size 64 \
  --output-root /data5T/profile-results
' > /data5T/profile-results/distributed_profile_launcher.log 2>&1 &
```

查看进度：

```bash
tail -f /data5T/profile-results/distributed_profile_launcher.log
pgrep -af "profile_cosmos_ascend|torchrun|torch.distributed.run"
npu-smi info
```

工具还会在每次训练的 mode 目录生成 `training.log`。训练结束后，继续等待日志出现：

```text
All profiling data parsed
Profiling output: ...
```

## 11. 常见问题

### 11.1 Profile 在 active step 崩溃或文件截断

首先关闭高开销选项：

```bash
--record-shapes --no-with-stack --no-with-modules
```

然后检查：

```bash
rg -n "Traceback|ERROR|ChildFailed|Signal|timeout" /结果目录/training.log
find /结果目录 -name analyse.done
```

缺少 `analyse.done`、`trace_view.json` 极小或 raw 文件停在固定小尺寸，通常表示采集或解析没有完整结束。

### 11.2 分布式任务无法启动

检查旧进程和端口：

```bash
pgrep -af "torchrun|torch.distributed.run|launch_sft"
ss -ltnp | rg ':50124|:50125|:50126'
npu-smi info
```

不要在未确认 PID 所属任务前直接终止进程。

### 11.3 解码很慢但 NPU 没有断粮

这是预取正常工作的表现。只要 `data_wait` 稳定在毫秒级，就不应优先增加 worker。更多 worker 可能增加 CPU、内存、文件句柄和 decoder cache 占用。

### 11.4 decoder cache 命中率低

cache 是每个 DataLoader worker 独立持有的。当单个 worker 分到的视频数远大于 cache size 时会发生 LRU 淘汰。可以 A/B 测试 `64/128/256`，但必须同时观察：

- `decoder_cache_hit/miss/eviction`
- `data_wait`
- worker RSS
- 打开文件数
- 整体 step time

如果 `data_wait` 已经很小，仅提高命中率未必能改善训练吞吐。

### 11.5 Profiler step 比基线慢

这是正常的采集开销。算子 Profile 用于分析组成和归因，正常训练吞吐必须使用第二阶段的无 Profiler 基线。

## 12. 优化实验的对比方法

每次只修改一个主要变量，并保持以下条件相同：

- 数据集和随机种子
- 最大 FPS、分辨率和视频采样策略
- global/local batch 与 packing 参数
- NPU 数量和并行拓扑
- warmup/统计 step 范围
- checkpoint 和模型版本

建议记录：

| 类别 | 指标 |
| :--- | :--- |
| 正常吞吐 | 稳态 `iteration_core` mean/P50/P90 |
| 数据关键路径 | `data_wait` P50/P90/max |
| 模型阶段 | Forward、Backward、Optimizer |
| 通信 | 未重叠通信时间、通信重叠率 |
| 负载均衡 | 最大/最小 rank Computing、FlashAttention 时间 |
| 数据缓存 | hit rate、eviction、worker RSS |

只有当第二阶段稳态 step time 改善时，才能确认优化真正提升了训练吞吐。第三和第四阶段用于解释改善来自哪里。

## 13. 分析报告模板

```markdown
# Ascend 性能分析

## 环境
- 分支/commit：
- NPU 型号和数量：
- CANN / torch / torch_npu：
- 数据集与样本数：
- 训练和 packing 配置：

## 数据阶段
- 成功/失败样本：
- samples/s、frames/s：
- decode P50/P90：
- cache hit rate：

## 正常训练基线
- 稳态 step mean/P50/P90：
- forward/backward/optimizer：
- data_wait/host_to_device：

## 单 rank 算子热点
- Top kernel families：
- Profile active-step 开销：
- 对应代码路径：

## 多 rank 分析
- Computing 最大/最小 rank：
- 未重叠通信最大/平均：
- 通信重叠率：
- Attention FLOPs 是否均衡：

## 结论和优化优先级
1.
2.
3.
```

## 14. 已完成实验总览

本节是 Ascend 性能实验的唯一汇总入口。原始 JSON、CSV、数据库和 trace 仍保存在各实验目录；
实验背景、实现决策、关键数据和结论统一维护在本文，不再为每个实验单独维护 Markdown。

| 案例 | 卡数 | 核心问题 | 结果 |
| :--- | ---: | :--- | :--- |
| `aten::to` 根因修复 | 4 | Host 标量循环和隐式同步 | step time `-20.6%` |
| EgoSuite 融合算子矩阵 | 8 | `to_list`、Attention、RoPE、RMSNorm | 最优组合 `+6.57%`，allocated `-3.399 GiB/卡` |
| 当前分片设计 GQA 复测 | 4 | `masked_sdpa` 对比原生 GQA FA | allocated `-3.227 GiB`，但 step time `+2.97%` |
| `max_video_duration_s` 影响 | 4 | 长 episode 的二次 attention 成本 | 91s 相对 61s 慢 `44.27%` |
| 均衡重叠切片 | 4 | 保留全部 episode，同时限制单窗口长度 | step time P50 `-26.1%` |

## 15. 案例一：`aten::to` 超长耗时根因与修复

### 15.1 根因

trace 中多秒级 `aten::to` 不是小 Tensor 搬运本身需要数秒，而是同步点替前序异步 NPU 工作
结算时间。根因包括：

1. teacher-forcing packing 使用 Python dict/list 和 `int(Tensor)` 逐元素 remap；
2. FA 每层对 NPU cumulative lengths 调用 `.tolist()`；
3. timestep embedding 先在 CPU 创建 frequency Tensor，再 `.to(device)`；
4. rank 到达 collective 的时间不同，使 FSDP AllGather 表面耗时被放大。

修复后保留 host actual sequence lengths，并使用 Tensor lookup/indexing；frequency `arange`
直接创建在目标设备。CPU 微基准从 `661.538 ms` 降至 `7.322 ms`，加速 `90.3×`，输出完全一致。

### 15.2 四卡验证

| 指标 | 基线 | 修复后 | 变化 |
| :--- | ---: | ---: | ---: |
| step wall time | 45.272s | 35.944s | `-20.6%` |
| `POST_NOISE_PACK` rank1 | 9793.775ms | 290.666ms | `-97.0%` |
| root AllGather 最大事件 | 6744.667ms | 562.567ms | `-91.7%` |
| `EMBED_CLEAN_TIMESTEPS` | 6322.790ms | 1.272ms | 近乎消除 |
| Transformer host range | 约 6753.6ms | 约 5339.1ms | `-20.9%` |

`aten::item` 从 266,130 次降至 2,630 次，`aten::select` 从 212,912 次降至 2,634 次。
验收测试为 `77 passed`，静态检查通过。原始结果位于：

```text
/mnt/sfs_turbo/zheng/cosmos-ascend-profile/cosmos-profile-logs/aten_to_fix_validation/
ascend_profile_20260909_190247
```

数据库核验时必须展开 `PYTORCH_API`、`CANN_API` 和 `COMMUNICATION_OP` 时间线；不能把
PyTorch API 的墙钟时间直接等价为 API 自身的计算或搬运时间。

## 16. 案例二：EgoSuite 融合算子 A/B

### 16.1 冻结条件

- 数据集：`/mnt/sfs_turbo/public/datasets/egosuite_demo_v1`；
- 8 张 Ascend 910B3，`max_sequence_length=45056`；
- 每组丢弃前 2 步，统计 5 个稳定步；
- 主指标为每步跨 rank 最大 `iteration_core`；
- profiler 使用 Level1 + PipeUtilization、2 个 active step、record shapes 和 profile memory。

### 16.2 计时结果

| 实验 | 配置 | 迭代均值 | 相对 E0 | Peak allocated |
| :--- | :--- | ---: | ---: | ---: |
| E0 | 原始路径 | 55.888s | 基线 | 46.240 GiB |
| E1 | host lengths + Tensor remap | 55.195s | `+1.24%` | 46.240 GiB |
| E2 | E1 + 显式 Attention | 57.201s | `-2.35%` | 40.627 GiB |
| E3 | E1 + RoPE | 54.299s | `+2.84%` | 46.751 GiB |
| E4 | E1 + RMSNorm | 53.187s | `+4.83%` | 42.331 GiB |
| E5 | E1 + RoPE + RMSNorm | 52.217s | `+6.57%` | 42.841 GiB |

关键结论：

- `to_list` 路径移除了 98.06% 的 `aten::item`，但异步工作会移动到后续同步点，端到端净收益
  为 1.24%；
- 显式 Attention 通过原生 GQA 每卡节省 5.613 GiB allocated，但当前 dense allMask FA
  端到端回退 3.63%，只建议在显存受限时显式启用；
- RoPE 命中融合算子并减少 cat/add/mul，较 E1 提升 1.62%；
- RMSNorm 移除 FP32 pow/mean/rsqrt 链，较 E1 提升 3.64%，每卡节省 3.909 GiB；
- 最快组合为 E1 + RoPE + RMSNorm，显式 Attention 保持关闭。

推荐开关：

```bash
export COSMOS_ASCEND_SEQUENCE_PACKING_TOLIST_OPT=1
export COSMOS_ASCEND_FUSED_ROPE=1
export COSMOS_ASCEND_FUSED_RMSNORM=1
export COSMOS_ASCEND_FUSED_TEACHER_FORCING_ATTENTION=0
```

原始计时和 profiler 结果分别位于：

```text
/mnt/sfs_turbo/zheng/cosmos-ascend-profile/cosmos-profile-logs/egosuite_fusion_ab/20260910_174026
/mnt/sfs_turbo/zheng/cosmos-ascend-profile/cosmos-profile-logs/egosuite_fusion_ab/20260910_184004
```

### 16.3 当前分片设计下的四卡 GQA 复测

在均衡重叠分片实现完成后，重新进行了严格单变量 A/B。两组均使用 EgoSuite、4 张 NPU、
同一 DCP checkpoint 和随机种子，`max_video_duration_s=61`、`long_video_policy=split`、
`video_window_overlap_s=5`、`max_sequence_length=45056`、PyAV `decode_transform`，每个 rank
使用 8 个 DataLoader worker。每组运行 15 轮，丢弃前 5 轮，统计第 6--15 轮；两组均启用
`to_list`、融合 RoPE 和融合 RMSNorm，唯一变量为是否启用
`COSMOS_ASCEND_FUSED_TEACHER_FORCING_ATTENTION`。

| 后端 | 稳定轮数 | iteration 均值 | iteration 中位数 | Peak allocated | Peak reserved |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `masked_sdpa` | 10 | 32.862s | 32.898s | 42.425 GiB | 58.916 GiB |
| `npu_fusion_attention` 原生 GQA | 10 | 33.837s | 33.891s | 39.197 GiB | 58.553 GiB |
| GQA 变化 | - | `+0.975s`（`+2.97%`） | `+0.993s`（`+3.02%`） | `-3.227 GiB`（`-7.61%`） | `-0.363 GiB`（`-0.62%`） |

GQA 在全部 10 个配对迭代中都更慢，单步回退为 0.503--1.691s，有效吞吐下降 2.88%。
因此当前 61 秒任意 blocked mask 分片负载再次验证了此前结论：原生 GQA 有稳定显存收益，
但没有训练吞吐收益；追求速度时保持关闭，仅在显存约束成为主要矛盾时启用。

#### Peak allocated 与 Peak reserved 分别表示什么

- `max_memory_allocated` 是活跃 Tensor 实际占用显存的历史峰值，用于判断模型和算子是否
  真正减少了张量显存。
- `max_memory_reserved` 是 PyTorch NPU 缓存分配器向设备申请并持有的显存历史峰值，既包含
  allocated，也包含已经空闲但仍留在分配器中等待复用的缓存块，因此通常有
  `reserved >= allocated`。

| 后端 | Peak allocated | Peak reserved | `reserved - allocated` 缓存余量 |
| :--- | ---: | ---: | ---: |
| `masked_sdpa` | 42.425 GiB | 58.916 GiB | 16.491 GiB |
| 原生 GQA | 39.197 GiB | 58.553 GiB | 19.355 GiB |

`masked_sdpa` 会用 `repeat_interleave` 将 K/V heads 物化展开到与 Q heads 相同；原生 GQA
直接消费较少的 K/V heads，因此避免大型临时 K/V Tensor，使 allocated 峰值下降
3.227 GiB。该下降在 4 个 rank 上均可复现，每个 rank 约减少 2.7--3.2 GiB，不是单卡
采样异常。

reserved 没有同比下降，是因为模型初始化、VAE、FSDP AllGather、优化器和其他临时计算
已经使缓存分配器申请了较大的内存 segment。临时 Tensor 释放后，分配器会保留这些块供
后续迭代复用，而不是立即归还 NPU 驱动；减少部分活跃占用也不一定能释放一个完整 segment。
所以 GQA 节省的显存主要转化为 allocator 内部可复用余量，`npu-smi` 显示的进程占用可能
不会明显下降。

判断算子显存收益应优先看 allocated；判断进程向设备保留的总量和其他进程可用空间，则更
关注 reserved。`empty_cache()` 只能归还未被活跃 Tensor 使用的缓存块，不能降低仍在使用的
allocated；训练热路径频繁调用还会引入重复申请开销，不作为常规优化手段。

本次未启用重型算子 profiler，以免污染端到端计时；显存来自每个 rank 的 NPU allocator
高水位记录。原始事件、逐轮数据及机器可读汇总位于：

```text
/mnt/sfs_turbo/zheng/cosmos-ascend-profile/cosmos-profile-logs/gqa_split_ab/20260911_064836
```

## 17. 案例三：`max_video_duration_s` 的性能影响

### 17.1 原行为与问题

原 LeRobot loader 把 `max_video_duration_s` 当作整条 episode 的过滤阈值，而不是裁剪长度。
61s 配置只保留 78 条 episode，91s 配置保留 122 条；后者平均 episode 时长从 26.652s
增加到 42.904s。

在 packed token 总量接近时，attention 仍按每个独立样本计算近似二次成本：

```text
cost = sum_i(s_und_i^2 + s_gen_i * (s_und_i + s_gen_i))
```

因此少量长窗口会显著提高 FLOPs，不能只用 packed token 总数预测 step time。

### 17.2 四卡结果

固定 PyAV、`decode_transform`、8 workers/rank、15 FPS、45,056 token 和 seed 42：

| 指标 | D61 | D91 | D91 相对 D61 |
| :--- | ---: | ---: | ---: |
| 临界 `iteration_core` 均值 | 30.607s | 44.156s | `+44.27%` |
| forward 临界均值 | 13.773s | 18.576s | `+34.87%` |
| backward + optimizer | 17.130s | 26.046s | `+52.05%` |
| data wait | 5.437ms | 5.738ms | 基本不变 |
| Peak allocated | 41.095 GiB | 45.287 GiB | `+4.193 GiB` |
| 总 FLOPs/step | 3.009 PFLOPs | 4.242 PFLOPs | `+40.95%` |
| 非 VAE FLOPs/step | 2.390 PFLOPs | 3.623 PFLOPs | `+51.57%` |

总 FLOPs 增长 40.95% 可以解释大部分 44.27% 的 step 增长；VAE FLOPs、稳定期数据等待和
rank 不均衡均不是一阶根因。原始结果位于：

```text
/mnt/sfs_turbo/zheng/cosmos-ascend-profile/cosmos-profile-logs/max_video_duration_ab/20260911_0256
```

## 18. 案例四：超长 Episode 均衡重叠切片

### 18.1 配置与行为

新增配置：

```toml
max_video_duration_s   = 61.0
long_video_policy      = "split"
video_window_overlap_s = 5.0
```

默认仍为 `drop + overlap=0`，兼容旧行为；目标 Vision Edge recipe 显式启用
`split + overlap=5`。约束仅为 `0 <= overlap < max_video_duration_s`，不设置四分之一上限。
`max_video_duration_s=0` 继续表示关闭时长上限。

切片发生在 parquet metadata 展开阶段，不创建新视频文件。每个 clip 是独立、等概率样本，
因此长 episode 会因产生多个窗口获得更高采样权重，且不依赖 `sample_by_window`。

### 18.2 帧级均衡算法

设源区间为半开区间 `[source_start, source_stop)`，源帧数为 `N`，最大帧数为 `M`，重叠帧数
为 `O`：

```text
n = ceil((N - O) / (M - O))
materialized = N + (n - 1) * O
base, remainder = divmod(materialized, n)
```

前 `remainder` 个窗口长度为 `base+1`，其余为 `base`；相邻窗口精确重叠 `O` 帧，首尾覆盖
源区间且每个窗口不超过上限。例如 91s、30 FPS、61s 上限和 5s overlap 生成两个 48s 窗口：

```text
[0, 1440) 和 [1290, 2730)，重叠 150 帧
```

metadata 记录 `source_episode_uuid`、源帧边界、clip index/count 和 overlap frames，并生成稳定
clip UUID。FPS 下采样以源 episode 起点作为共享相位，保证 overlap 中同一个源帧不会因窗口
边界改变而在两个 clip 中错位。

structured JSON caption 第一版原样复制并输出 warning，因为其中嵌入的 duration、FPS 或
timestamp 可能仍描述源 episode；普通 EgoSuite `tasks` caption 不受影响。

### 18.3 正确性与数据覆盖

- 数据集单测：`26 passed`；
- 静态检查：Ruff 通过；
- 真实 PyAV `decode_transform` 冒烟：4/4 个长视频切片成功；
- EgoSuite 126 个源 episode 全部覆盖，展开为 175 个独立样本；
- 完整源数据为 169,921 帧，5s overlap 后为 177,271 帧，仅增加 4.33%。

### 18.4 四卡无 profiler 计时

两桶保持数据集、15 FPS、45,056 token、PyAV、workers 和 seed 相同：

| 桶 | 策略 | 稳定步最慢 rank P50 |
| :--- | :--- | ---: |
| A | 完整 episode，不设时长上限 | 47.98s |
| B | 61s 均衡切片 + 5s overlap | 35.23s |

B 桶 step time 下降 `26.6%`。事件汇总中 training step、forward、backward 和 optimizer 的
均值分别下降约 22.3%、18.9%、23.7% 和 25.7%。切片桶处理更多预取样本且保留全部源数据，
因此收益不是通过丢弃长 episode 获得。

计时结果：

```text
/mnt/sfs_turbo/zheng/cosmos-ascend-profile/cosmos-profile-logs/max_video_duration_split_ab/
20260911_053437
```

### 18.5 四卡 Level1 + PipeUtilization

A/B 均采集 4 个 rank、2 个 active step，开启 record shapes 和 profile memory；每桶均生成
4 份 `kernel_details.csv` 和 4 份 `trace_view.json`。

| 指标 P50 | A 完整 episode | B 61s + 5s | 变化 |
| :--- | ---: | ---: | ---: |
| training step | 48.875s | 36.123s | `-26.1%` |
| forward | 19.925s | 14.077s | `-29.4%` |
| backward | 16.429s | 12.510s | `-23.9%` |
| optimizer | 11.703s | 8.790s | `-24.9%` |

算子聚合进一步显示：FlashAttention 正向 kernel 总时间下降约 33.7%，反向下降约 34.7%；
HCCL ReduceScatter 和 AllGather 聚合时间分别下降约 46.9% 和 54.6%。虽然 B 桶因 packing
包含更多短样本而产生更多 FA 调用，但单次 attention 的长度和二次成本显著降低。

A/B profile 结果：

```text
# A 桶
/mnt/sfs_turbo/zheng/cosmos-ascend-profile/cosmos-profile-logs/max_video_duration_split_ab/
20260911_055014/A_full_episode/profile/ascend_profile_20260911_055015

# B 桶
/mnt/sfs_turbo/zheng/cosmos-ascend-profile/cosmos-profile-logs/max_video_duration_split_ab/
20260911_060844/B_split_61s_overlap_5s/profile/ascend_profile_20260911_060846
```

### 18.6 验收结论

均衡重叠切片已同时通过覆盖率、解码、训练和 profiler 验证。它在保留全部长视频语义覆盖、
支付 4.33% overlap 数据增量的前提下，将 4 卡训练 step time 降低约 26%。收益来自缩短独立
attention window，并同步降低 FA 和 FSDP 通信等待，而不是 DataLoader 偶然波动。

## 19. 当前建议与后续工作

1. Vision Edge LeRobot 训练采用 `split + 61s + 5s overlap` 作为目标 recipe；通用默认仍保持
   `drop + overlap=0`，避免无意改变其他任务的数据权重。
2. 性能选择必须同时报告 step time、源视频覆盖、物化视频帧/秒、有效 token、FLOPs 和峰值
   allocated memory；不能只比较每步耗时。
3. 继续探索 attention-cost-aware packing，使 sampler 显式平衡平方成本，而不只限制 packed
   token 总数。
4. structured JSON 数据集在正式使用切片前，应决定是否按 clip 重写 duration/FPS/timestamps；
   当前 warning 策略只保证不静默产生错误假设。
5. 若启用融合优化，优先使用 `to_list + RoPE + RMSNorm`；显式 Attention 仅在显存压力优先于
   吞吐时使用，并在目标数据集重新 A/B。
