# LeRobot 数据读取内存分析

`profile_dataloader` 用于复现训练路径中的数据读取行为，同时排除网络计算对内存的影响。它加载与训练命令相同的 SFT TOML，只实例化 `dataloader_train` 并持续获取、统计、释放 batch。

脚本不会实例化 Trainer、模型、优化器、学习率调度器或 checkpoint，也不会执行 batch 设备拷贝、forward、loss、backward 和参数更新。它会与训练入口一样初始化加速器进程组，并在配置校验时短暂创建一个很小的设备张量；这些是原训练进入 dataloader 前已有的行为。因此，它适合判断持续增长发生在视频解码、worker 预取、pin memory、packing 或 Python/原生内存分配的哪一侧。

## 准备环境

在仓库根目录进入训练所用的 `cosmos-framework` 环境，并沿用训练任务的动态库设置。`ffprobe` 和 TorchCodec 应来自同一个 Conda 环境：

```bash
conda activate cosmos-framework
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export DATASET_PATH=/public/Embodied-AI/datasets/nvidia/LIBERO_LeRobot_v3/libero_10
export BASE_CHECKPOINT_PATH=/public/Embodied-AI/ckpts/Cosmos3-Edge-DCP
export COSMOS3_EDGE_PROCESSOR_PATH=/public/Embodied-AI/ckpts/Cosmos3-Edge
export WAN_VAE_PATH=/public/Embodied-AI/ckpts/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
export IMAGINAIRE_OUTPUT_ROOT=/tmp/cosmos_dataloader_profile
```

虽然本工具不会读取模型权重，`BASE_CHECKPOINT_PATH` 和 `WAN_VAE_PATH` 仍需设置，以解析现有 TOML 中的环境变量插值。处理器路径用于构建数据 tokenizer。

## LIBERO 单进程基线

以下命令完整保留 `vision_sft_edge_lerobot3` 的数据集、TorchCodec 解码和 packing 路径：

```bash
python -m cosmos_framework.scripts.profile_dataloader \
  --sft-toml examples/toml/sft_config/vision_sft_edge.toml \
  --iterations 500 \
  --log-every 1 \
  --full-info-every 10 \
  --output-dir /tmp/libero_dataloader_baseline \
  -- \
  model.config.vlm_config.tokenizer.repository=null \
  model.config.vlm_config.tokenizer.revision=null \
  +model.config.vlm_config.tokenizer.tokenizer_type="$COSMOS3_EDGE_PROCESSOR_PATH"
```

`--iterations` 的单位与训练日志中的 iteration 相同，即 optimizer iteration，不是 batch 数。脚本按照 `trainer.grad_accum_iter` 模拟控制流；当前配置为 2，所以 `--iterations 500` 会读取 1000 个有效 microbatch，并与真实训练一致，在达到终点后再 fetch 一个 batch、检查终止条件并释放它。如果省略 `--iterations`，脚本会运行到 TOML 中的 `trainer.max_iter`。

模拟 checkpoint 恢复时，使用 `--start-iteration N`。脚本会像 Trainer 一样执行：

```text
dataloader.set_start_iteration(N * trainer.grad_accum_iter)
```

此时 `--iterations` 表示从 N 开始继续模拟多少个 optimizer iteration。

基线配置使用 `batch_size=1`、`num_workers=4`、`persistent_workers=true`、`pin_memory=true`、`prefetch_factor=4`，每个 worker 的 LeRobot 视频 decoder cache 上限为 64。数据仍会读取完整 episode，并经过训练使用的 `PackingDataLoader`；batch 的逻辑张量大小达到数百 MiB 是可能的。

### 切换视频读取后端

配置默认使用 `torchcodec`。可以直接修改 TOML：

```toml
[dataloader_train]
video_backend = "pyav"
video_tolerance_s = 1.0e-4
```

临时切换到 LeRobot 注册的 PyAV 后端时，也可以在命令末尾增加：

```text
dataloader_train.dataloader.datasets.video.dataset.video_backend=pyav
```

切回 TorchCodec：

```text
dataloader_train.dataloader.datasets.video.dataset.video_backend=torchcodec
```

PyAV 路径复用 LeRobot 的时间戳读取与最近帧匹配逻辑，默认容差为 `1e-4` 秒；必要时可通过下面的 override 调整：

```text
dataloader_train.dataloader.datasets.video.dataset.video_tolerance_s=0.0001
```

TorchCodec 路径继续使用精确 frame range seek 和每个 worker 独立的 decoder LRU cache；`decoder_cache_max_size` 只影响 TorchCodec。比较两个后端时，除 `video_backend` 外应保持 worker、预取、pin memory、数据顺序和迭代数一致，并分别使用独立输出目录。

每个输出目录包含：

- `effective_config.yaml`：所有 TOML 和命令行 override 合并后的有效配置。
- `rank_000.jsonl`：每行一个内存快照；多 rank 时每个 rank 各写一个文件。
- `wandb/rank_000/`：启用 W&B 后，该 rank 的本地 W&B run 文件。

不要让两个同时运行的任务共用输出目录，同 rank 文件会在启动时覆盖。

### 写入 Weights & Biases

W&B 默认关闭。在线上传并将 W&B 本地文件保存在 profiler 输出目录时，增加：

```text
--wandb-mode online \
--wandb-project cosmos-dataloader-profile \
--wandb-group torchcodec-vs-pyav \
--wandb-name torchcodec-cache64
```

只生成本地 W&B run、不访问网络时，使用：

```text
--wandb-mode offline
```

默认只为 rank 0 创建 W&B run。需要观察每个 rank 时增加：

```text
--wandb-ranks all
```

此时每个 rank 创建独立 run，run 名自动追加 `-rankNNN`，本地文件分别位于 `wandb/rank_NNN/`。W&B 中的指标按快照 phase 分组，例如：

```text
phase/after_next/process_tree_rss_mib
phase/after_release/process_tree_pss_mib
phase/after_next/children_fds
phase/after_next/worker_max_num_fds
phase/after_next/fetch_seconds
```

原始字节数和每个 worker 的完整明细仍以 JSONL 为准；W&B 将内存字段转换成 MiB，并上传常用聚合值。脚本先完成全部 profiling、关闭 worker 并写完 JSONL，然后才初始化 W&B 并回放记录。因此 W&B 后台进程、网络线程及其文件描述符不会进入内存快照，也不会影响 `fetch_seconds`。

## 多 rank 复现

如果训练本身使用多个 rank，应保持相同的每节点进程数：

```bash
torchrun --nnodes=1 --node-rank=0 --nproc-per-node=8 \
  --master-addr=127.0.0.1 --master-port=29501 \
  -m cosmos_framework.scripts.profile_dataloader \
  --sft-toml examples/toml/sft_config/vision_sft_edge.toml \
  --iterations 500 \
  --log-every 1 \
  --full-info-every 10 \
  --output-dir /tmp/libero_dataloader_8rank \
  -- \
  model.config.vlm_config.tokenizer.repository=null \
  model.config.vlm_config.tokenizer.revision=null \
  +model.config.vlm_config.tokenizer.tokenizer_type="$COSMOS3_EDGE_PROCESSOR_PATH"
```

脚本复用训练入口的 `distributed.init()`：NPU 使用 HCCL、CUDA 使用 NCCL，只有纯 CPU 运行才使用 Gloo。随后按训练顺序执行配置校验、冻结、按 rank 设置随机种子，并在 `data_loader_init()` 上下文中实例化 dataloader。`num_workers=4` 是每个 rank 的 worker 数；8 rank 会产生约 32 个读取 worker，运行前需确认主机内存足够。

这里不再创建额外的 Gloo 进程组。单机多卡仍建议把 `--master-addr` 显式设为 `127.0.0.1`；多机多卡则应使用 rank 0 节点可达的 IPv4 地址，并保持与原训练命令完全一致的 HCCL/NCCL 网络环境。

## 快照阶段与字段

一次迭代会依次记录 `before_next`、`after_next` 和 `after_release`。如果启用了 allocator trim，则最后一个阶段名为 `after_release_trimmed`。

| 阶段                     | 含义                                                       |
| ------------------------ | ---------------------------------------------------------- |
| `dataloader_init_before` | 实例化 dataloader 前                                       |
| `dataloader_init_after`  | dataloader 已构建，但迭代器尚未创建                        |
| `iterator_ready`         | worker 已启动，预取可能已开始                              |
| `before_next`            | 调用 `next()` 前                                           |
| `after_next`             | batch 已返回；包含读取耗时与 batch 张量逻辑大小            |
| `after_release`          | 删除主进程中的 batch 引用后                                |
| `after_release_terminal` | 达到目标 iteration 后额外 fetch 的 batch 已释放            |
| `iterator_exhausted`     | 捕获 `StopIteration`；随后按训练逻辑重新创建 iterator      |
| `run_end`                | 完成指定迭代次数                                           |
| `workers_shutdown`       | 释放 dataloader/迭代器并执行 GC 后；正常情况下 worker 为 0 |

常用字段如下：

| 字段                                      | 含义                                                       |
| ----------------------------------------- | ---------------------------------------------------------- |
| `parent.rss_bytes`                        | 当前 rank 主进程 RSS                                       |
| `children_rss_bytes`                      | 当前 rank 所有递归子进程 RSS 之和                          |
| `process_tree_rss_bytes`                  | 主进程与 worker RSS 之和                                   |
| `process_tree_uss_bytes`                  | 进程树独占内存；仅在 full-info 快照中存在                  |
| `process_tree_pss_bytes`                  | 按共享比例折算后的内存；仅在 full-info 快照中存在          |
| `parent.rss_anon_bytes`                   | 主进程匿名 RSS，通常比总 RSS 更适合观察堆内存增长          |
| `children_pinned_bytes`                   | `/proc` 报告的 worker pinned/locked memory                 |
| `children_fds`                            | worker 文件描述符总数，可用于发现 decoder/文件句柄持续累积 |
| `system_shm_used_bytes`                   | `/dev/shm` 已用空间                                        |
| `system_memory_available_bytes`           | 主机可用内存                                               |
| `fetch_seconds`                           | 当前 `next()` 或初始化耗时                                 |
| `iteration`                               | 与训练一致的虚拟 optimizer iteration                       |
| `batch_index`                             | 从本次 profiler 启动开始累计的成功 fetch 次数              |
| `grad_accum_index`                        | 当前 optimizer iteration 内的 microbatch 下标              |
| `batch_tensors.by_device.*.logical_bytes` | batch 中张量的逻辑字节数，不等同于进程实际新增内存         |

Linux RSS 会把共享页计入多个进程，因此不能把多个 worker 的 RSS 简单相加后当成物理占用。跨实验优先比较 PSS；定位不能共享的泄漏时优先看 USS。`--full-info-every` 的采集开销较高，长跑建议设为 10 或更大。

下面的命令打印每个 full-info 快照的进程树内存：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("/tmp/libero_dataloader_baseline/rank_000.jsonl")
for line in path.open():
    row = json.loads(line)
    if "process_tree_pss_bytes" not in row:
        continue
    mib = 2**20
    print(
        row["iteration"],
        row["phase"],
        f"PSS={row['process_tree_pss_bytes'] / mib:.1f} MiB",
        f"USS={row['process_tree_uss_bytes'] / mib:.1f} MiB",
        f"FDs={row['children_fds']}",
    )
PY
```

## 如何判断持续增长

先覆盖至少 500 个 batch；当前 `grad_accum_iter=2` 时可使用 `--iterations 250`。如果数据集 episode 较长或 decoder cache 尚未填满，建议增加到 `--iterations 500`，即 1000 个有效 batch。重点比较相同 phase 的点，不要把 `before_next` 和 `after_next` 直接连成增长趋势。

- 如果前几十个 batch 阶梯式增加，之后 PSS/USS 稳定，通常是 worker 预取队列、decoder cache 或 allocator 高水位的预热。
- 如果 `after_next` 上升而 `after_release` 回落，内存主要由当前 batch 持有。
- 如果 `after_release` 的 USS/PSS 仍随迭代近似线性上升，说明释放 batch 后仍有对象、decoder 或原生缓冲区被保留。
- 如果 `children_fds` 持续增加，优先检查视频 decoder 和文件句柄生命周期。
- 如果只有主进程上涨，优先检查 packing 和 pin-memory 搬运；如果主要是 children 上涨，优先检查 dataset、worker 预取和 TorchCodec decoder cache。
- `workers_shutdown` 后 children 应为 0；此时主进程仍较高但 `malloc_trim` 后下降，通常是 allocator 保留而不是活跃对象泄漏。

## 建议的逐项消融

每项使用独立输出目录，并且一次只改一个变量。

### 无 worker

`num_workers=0` 时，PyTorch 同时要求关闭 persistent worker 和 prefetch：

```bash
python -m cosmos_framework.scripts.profile_dataloader \
  --sft-toml examples/toml/sft_config/vision_sft_edge.toml \
  --iterations 500 --output-dir /tmp/libero_workers0 -- \
  dataloader_train.dataloader.num_workers=0 \
  dataloader_train.dataloader.prefetch_factor=null \
  dataloader_train.dataloader.persistent_workers=false \
  model.config.vlm_config.tokenizer.repository=null \
  model.config.vlm_config.tokenizer.revision=null \
  +model.config.vlm_config.tokenizer.tokenizer_type="$COSMOS3_EDGE_PROCESSOR_PATH"
```

### 降低预取深度

```text
dataloader_train.dataloader.prefetch_factor=1
```

### 关闭 persistent worker

```text
dataloader_train.dataloader.persistent_workers=false
```

### 关闭 pin memory

```text
dataloader_train.dataloader.pin_memory=false
```

### 缩小 TorchCodec decoder cache

```text
dataloader_train.dataloader.datasets.video.dataset.decoder_cache_max_size=1
```

如果 cache 为 1 时曲线稳定、64 时持续上升到更高平台，增长更可能来自预期的 decoder 缓存。如果两者都不封顶，再检查 decoder 的逐 batch 生命周期。

### 区分活跃对象与 allocator 高水位

先保持基线运行，然后单独执行以下诊断版本：

```text
--gc-every 10 --malloc-trim-every 10
```

这两个选项会改变运行时行为，不应作为最终性能数据；它们只用于判断 Python GC 或 glibc allocator 是否解释了观察到的 RSS。若 `after_release_trimmed` 明显下降而 USS 中没有等量活跃内存，通常意味着 allocator 保留。
