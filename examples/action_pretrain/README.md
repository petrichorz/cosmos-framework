# AgiBot / EgoSuite：从组统计到训练

本文面向其他 Ascend 机器，所有命令从 `cosmos-framework` 仓库根目录执行。先替换 `/path/to/xxx`，再在同一个 Bash 会话中按顺序运行。要求已安装本仓库、`cosmos-framework-py312` 环境和匹配的 CANN/torch_npu。

这里的“一组数据集”是一个父目录下的多个 LeRobot v3 子数据集。程序自动递归发现子集，无需逐个填写路径。数据必须满足当前 action 模板及 `state_unified`、`action_unified`、`mask_state`、`mask_action` 字段约定。

- **路线 A：只训练 AgiBot。** 计算一份 AgiBot 组统计，然后启动单组训练。
- **路线 B：AgiBot + EgoSuite 混合训练。** 分别计算两份组统计，按训练片段数设置组权重，再启动 mix。

## 1. 公共准备：环境与路径

```bash
export CONDA_ROOT=/path/to/miniforge3
export CONDA_ENV=cosmos-framework-py312
export ASCEND_ENV=/path/to/Ascend/ascend-toolkit/set_env.sh
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# 父目录下可以有任意多个子数据集；只训练 AgiBot 时不需要设置 EGOSUITE_ROOT。
export AGIBOT_ROOT=/path/to/agibot_processed
export EGOSUITE_ROOT=/path/to/egosuite_processed

export BASE_CHECKPOINT_PATH=/path/to/Cosmos3-Edge-DCP
export COSMOS3_EDGE_PROCESSOR_PATH=/path/to/Cosmos3-Edge
export WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth

# 每次新实验换一个目录；沿用训练输出目录可能自动恢复旧 checkpoint。
export RUN_DIR="$PWD/outputs/my_action_run"
export HF_DATASETS_CACHE="$RUN_DIR/hf_cache"
export TMPDIR=/path/to/local_tmp
mkdir -p "$RUN_DIR/stats" "$RUN_DIR/config" "$HF_DATASETS_CACHE" "$TMPDIR"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1

# 文件名由你决定；来源 JSON 会通过这两个环境变量引用统计文件。
export AGIBOT_GROUP_STATS_PATH="$RUN_DIR/stats/agibot_group_stats.json"
export EGOSUITE_GROUP_STATS_PATH="$RUN_DIR/stats/egosuite_group_stats.json"

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4

set -o pipefail
```

`TMPDIR` 使用本机可写的短路径目录，避免共享文件系统上的 multiprocessing 临时文件清理问题。首次准备机器时，要提前下载模型与 tokenizer；这里默认离线运行。

## 2. 路线 A：AgiBot 单组训练

### A1. 计算全组统计

命令前的两个环境变量仅作用于本次 CPU 统计，用于关闭 NPU 自动加载、避免动态库冲突，不影响后续训练。

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 LD_LIBRARY_PATH='' \
  python -m tools.compute_causal_action_stats \
  --dataset-root "$AGIBOT_ROOT" --profile agibot \
  --action-from-state --action-time-offset-steps 1 \
  --split-val-ratio 0 --split-seed 42 \
  --method reservoir --reservoir-size 50000 --seed 42 \
  --num-workers 4 --batch-size 512 \
  --output "$AGIBOT_GROUP_STATS_PATH" \
  2>&1 | tee "$RUN_DIR/stats/agibot.log"
```

AgiBot 的绝对目标取 `state[t+1]`，随后按模板计算 block delta；夹爪、灵巧手使用绝对值。父目录中的所有子集共同生成一份统计，不是平均各子集的 q01/q99。

**命令成功退出后再继续。正式统计不要加 `--limit`。** 输出 JSON 的 `provenance.partial` 应为 `false`。默认扫描全部合法 block 起点，均值、方差、极值来自全量数据，q01/q99 通过每通道最多 50000 个 reservoir 样本估计。

`--num-workers 4` 是统计读取/编码线程数；`--batch-size 512` 是每批最多处理 512 个窗口。每个窗口仍为 32 action、33 state，并保留自己的 anchor。它们不改变统计窗口大小，也不决定训练的 DataLoader worker 数。

### A2. 将统计文件接入训练

```bash
export ACTION_SOURCES_FILE="$PWD/examples/action_pretrain/sources/agibot.json"
export OUTPUT_ROOT="$RUN_DIR/train_agibot"
```

该来源 JSON 已包含下面的绑定，无需修改数据集的 `meta/stats.json`：

```json
"root": "${AGIBOT_ROOT}",
"statistics_path": "${AGIBOT_GROUP_STATS_PATH}"
```

训练首次读取各子集时，会加载同一份组统计，对 state 和模板编码后的 action 分别归一化。以后替换统计文件，只需更新 `AGIBOT_GROUP_STATS_PATH`，然后重新启动训练。

### A3. 启动训练

```bash
# 可先只打印命令，检查路径；这一步不会读取数据或启动训练。
PRINT_ONLY=1 bash examples/action_pretrain/launch_midtrain_template.sh

# 正式启动，默认 FD / ID / Policy 联训。
bash examples/action_pretrain/launch_midtrain_template.sh
```

默认 4 卡、每卡 4 个 DataLoader worker。需要先短测时，使用独立输出目录：

```bash
OUTPUT_ROOT="$RUN_DIR/smoke_agibot" \
  bash examples/action_pretrain/launch_midtrain_template.sh \
  trainer.max_iter=10 checkpoint.save_iter=10
```

## 3. 路线 B：AgiBot + EgoSuite 混合训练

### B1. 分别计算两组统计

先执行 **A1 的 AgiBot 统计命令**。如果同一组数据、读取规则和模板的统计已经计算完成，可以复用。然后计算 EgoSuite：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 LD_LIBRARY_PATH='' \
  python -m tools.compute_causal_action_stats \
  --dataset-root "$EGOSUITE_ROOT" --profile egosuite \
  --action-time-offset-steps 0 \
  --split-val-ratio 0 --split-seed 42 \
  --method reservoir --reservoir-size 50000 --seed 42 \
  --num-workers 4 --batch-size 512 \
  --output "$EGOSUITE_GROUP_STATS_PATH" \
  2>&1 | tee "$RUN_DIR/stats/egosuite.log"
```

EgoSuite 直接使用已处理的 `action_unified[t]`，**不要加 `--action-from-state`，也不要再次移位**。两份统计分别服务各自的数据组，不将 AgiBot 与 EgoSuite 合成一份统计。

### B2. 接入统计，并更新混合权重

```bash
export ACTION_SOURCES_FILE="$RUN_DIR/config/mixed.json"
cp examples/action_pretrain/sources/mixed.json "$ACTION_SOURCES_FILE"
export OUTPUT_ROOT="$RUN_DIR/train_mixed"
```

混合清单已经通过 `AGIBOT_GROUP_STATS_PATH`、`EGOSUITE_GROUP_STATS_PATH` 分别绑定两份统计。但仓库中的 `4976:4` 是原实验数据的权重，**换机器上的数据后不要直接照用**。

希望每个训练片段大致等机会采样时，组权重应取各组的有效训练片段数。下面读取 TOML 中当前的切片配置，使用正式数据集构造逻辑计数并更新刚复制的清单；只构造索引，不解码视频或启动训练：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 LD_LIBRARY_PATH='' python - <<'PY'
import json
import os
import tomllib
from pathlib import Path

from cosmos_framework.data.generator.action.datasets.causal_action_factory import get_causal_action_dataset

config = tomllib.loads(Path("examples/action_pretrain/action_midtrain_edge_causal_tnd.toml").read_text())
action = config["action"]
manifest = Path(os.environ["ACTION_SOURCES_FILE"])
data = json.loads(manifest.read_text())
for source in data["sources"]:
    stats_path = Path(os.path.expandvars(source["statistics_path"]))
    stats = json.loads(stats_path.read_text())
    assert not stats["provenance"]["partial"], f"Incomplete statistics: {stats_path}"

mixture = get_causal_action_dataset(
    sources_file=str(manifest),
    **{key: action[key] for key in (
        "template", "actions_per_block", "video_stride", "max_action_steps",
        "overlap_action_steps", "mode", "seed", "resolution",
    )},
    history_blocks_min=config["model"]["teacher_forcing_history_blocks_min"],
    history_blocks_max=config["model"]["teacher_forcing_history_blocks_max"],
    allow_mock_statistics=False,
)
for source, group in zip(data["sources"], mixture.datasets):
    source["weight"] = len(group)
    print(source["reader"], "subdatasets=", len(group.datasets), "segments=", len(group))
manifest.write_text(json.dumps(data, indent=2) + "\n")
print("Updated:", manifest)
PY
```

组内按片段索引遍历，组间按上述权重采样；无需将权重手动归一化到和为 1。不要使用统计 JSON 的 `blocks` 作为训练权重：统计滑窗与训练长片段是两套索引。增减子集或修改切片长度、overlap 后，应重新执行此步骤；修改模板、block 长度或目标读取规则后，还需要重新计算匹配统计。

### B3. 启动 mix 训练

```bash
PRINT_ONLY=1 bash examples/action_pretrain/launch_midtrain_template.sh
bash examples/action_pretrain/launch_midtrain_template.sh
```

与单组训练共用相同 TOML。混合的是数据来源，任务仍为 `mode=joint`，默认 Policy / ID / FD 按 1:1:1 采样，每个样本选择一种任务。

## 4. 常用训练参数在哪里改

文件：`examples/action_pretrain/action_midtrain_edge_causal_tnd.toml`。

| 配置位置                                     | 当前值  | 含义                                  |
| -------------------------------------------- | ------- | ------------------------------------- |
| `[action] max_action_steps`                  | 896     | 每片段最多 896 action、897 个原始观测 |
| `[action] overlap_action_steps`              | 161     | 相邻片段重叠的 action 步数            |
| `[action] mode`                              | `joint` | FD / ID / Policy 联训                 |
| `[model] teacher_forcing_history_blocks_max` | 32      | causal 历史 block 上限                |
| `[dataloader_train] max_sequence_length`     | 54000   | 每个 rank 的 packing token 上限       |
| `[dataloader_train] num_workers`             | 4       | 每个 rank 的数据加载进程数            |
| `[trainer] max_iter`                         | 10000   | 总训练步数                            |
| `[checkpoint] save_iter`                     | 5000    | 保存 checkpoint 的间隔                |

固定 block 当前为 32 action；视频每 4 帧采样一次，最长片段进入 VAE 前有 225 帧。`lookahead_limit` 在启动脚本中默认 1，可通过 `LOOKAHEAD_LIMIT` 覆盖。数据加载 worker、统计线程和 OpenMP 线程是不同设置；脚本不主动设置 `OMP_NUM_THREADS`。

临时修改训练 worker 数：

```bash
bash examples/action_pretrain/launch_midtrain_template.sh \
  dataloader_train.dataloader.num_workers=1
```

**小数据组注意分片数量。** 每组的有效片段数至少应达到 `卡数 × 每卡 worker 数`。例如原实验 EgoSuite 只有 4 个片段，4 卡单组或 mix 调试时需用每卡 1 worker；更多 worker 会产生空分片。换成 180 个 AgiBot 子集或其他 EgoSuite 数据后，应以 B2 的实际计数为准。

本手册使用正式 Reader。昨晚约 14.8 秒/iter 的结果来自实验版 PyAV resize/uint8 读取优化，尚未接入正式路径，仅增加 worker 数不能保证达到该速度。

## 5. 文件分工与结果位置

- `tools/compute_causal_action_stats.py`：CPU 组统计，支持批量读取和线程并行。
- `sources/agibot.json`、`sources/egosuite.json`、`sources/mixed.json`：来源、读取规则、组统计路径和权重。
- `action_midtrain_edge_causal_tnd.toml`：公共训练参数。
- `launch_midtrain_template.sh`：跨机器环境模板，路径为 `/path/to/xxx`；调用通用训练启动脚本。
- `launch_midtrain_local.sh`：原开发机器的独立副本，已经填入本机路径；其他机器使用 template。
- `launch_midtrain_action_causal.sh`：构造并执行 torchrun 命令，末尾传入的命令行参数优先。

统计文件在 `$RUN_DIR/stats`；生成的 mix 清单在 `$RUN_DIR/config/mixed.json`；启动日志在 `$OUTPUT_ROOT/launcher_rank0.log`，训练 checkpoint 位于该输出目录的任务子目录中。恢复训练复用输出目录，新实验使用新目录。
