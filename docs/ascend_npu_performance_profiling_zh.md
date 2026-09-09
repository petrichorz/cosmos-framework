# Cosmos3 Ascend NPU 四阶段性能采集与分析教程

本文说明如何使用 `tools/profile_cosmos_ascend.py` 对 Cosmos3 视觉 SFT 任务进行端到端性能分析，覆盖数据读取、正常训练基线、单 rank 算子分析和多 rank 通信分析。

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

## 14. 已完成案例

仓库根目录的 `PERFORMANCE_ANALYSIS_20260908.md` 是一份使用 Cosmos3-DROID `success` 数据集完成的实际四阶段报告，可作为阅读顺序、指标计算和结论表述的参考。
