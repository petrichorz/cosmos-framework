# Cosmos3 SFT 数据加载重构方案（LeRobot 动态加载）

> 本文档记录「让 vision SFT 直接动态加载 LeRobot 3.x 数据集（episodes 表带 caption 字段）」的最终实现方案。
>
> 背景：私有数据为 LeRobot 3.x 格式，episodes 表的 parquet 里**侵入式新增 caption 字段**（已验证：官方 lerobot 库加载 episodes 表加列不报错；data 表加列会报错，禁止）。
>
> **代码组织原则**：原 `sft_dataset.py` / `vision_sft_edge.py` **一行不改**，所有新增逻辑放到独立的新文件里。

---

## 已实现特性总览

| 特性 | 章节 | 简述 |
|------|------|------|
| LeRobot 3.x 动态加载 | 3、4 | 不转 JSONL，用官方 `LeRobotDatasetMetadata` 读 info.json + episodes parquet |
| **惰性化加载** | 3、4 | 加载期只产出 `(sources, episode_index)` 扁平索引，episode 字段采样期现算，不物化 dict |
| **独立 IterableDataset** | 2.3 | `LeRobotSFTDataset` 不继承 `SFTDataset`，自实现 `__init__`/`__len__`/`__iter__`/`process_one_sample` |
| 统一数据入口 `dataset_path` | 4.7 | `.jsonl`→manifest 模式；目录→单数据集根/父目录递归 |
| manifest 多数据集加载 | 4.8 | 每行一个 path，逐行覆盖 `video_feature_key`/`keywords`/`caption_key` |
| manifest 并行加载 | 4.8 | ThreadPoolExecutor 并行读多个数据集 |
| 多分辨率训练 | 4.10 | `use_multi_resolution`：256/480 随机，不上采样 |
| 多 fps 训练 | 4.10 | `use_multi_fps`：temporal_interval 随机 2/3/4 |
| 视频解码 | 5 | 复用官方 lerobot `decode_video_frames` 按绝对时间戳解码 |
| decoder LRU 缓存 | 5.5 | 替换 lerobot 无界缓存为 LRU（**仅 torchcodec 生效，pyav 下 no-op**） |
| caption 回退 | 4.6 | `caption_key` 列优先，回退官方 `tasks` 列 |

---

## 目录

- [1. 目标与约束](#1-目标与约束)
- [2. 最终文件组织](#2-最终文件组织)
- [3. 数据流与惰性化加载回顾](#3-数据流与惰性化加载回顾)
- [4. 字段映射方案（LeRobot → 样本）](#4-字段映射方案lerobot--样本)
- [5. 视频解码（复用官方 lerobot `decode_video_frames`）](#5-视频解码复用官方-lerobot-decode_video_frames)
- [6. 已确认的风险点与注意事项](#6-已确认的风险点与注意事项)
- [7. 关键结论速查](#7-关键结论速查)
- [8. 训练接入改动](#8-训练接入改动)
- [9. 验证与实测记录](#9-验证与实测记录)
- [附录：相关文件索引](#附录相关文件索引)

---

## 1. 目标与约束

### 目标

让现有 vision SFT 训练流程**动态加载 LeRobot 数据集**（不预先转 JSONL），并复用现有 `SFTDataset` 的 caption tokenize、sequence_plan、CFG dropout 等**纯函数逻辑**（但不继承 `SFTDataset` 类）。

### 已拍板的决策

| # | 决策 | 说明 |
|---|------|------|
| 1 | 路线 | 路线 B：动态加载，不转 JSONL |
| 2 | 相机 | 关键字匹配选视角（key 名含 `top`/`head` 等关键字即选中；匹配不到回退第一个 video），用官方 `meta.video_keys` |
| 3 | caption 粒度 | episode 级（一个任务一个描述）；`caption_key` 列优先，回退官方 `tasks` 列 |
| 4 | 字段语义 | 帧区间存**帧编号**（非 timestamp）；episode 级字段在采样期现算 |
| 5 | 代码组织 | **原 `sft_dataset.py` / `vision_sft_edge.py` 一行不改**；新建 `sft_dataset_lerobot3.py` + `vision_sft_edge_lerobot3.py`，且 `LeRobotSFTDataset` **不继承 `SFTDataset`** |
| 6 | 效率 | 惰性化加载（`sources` + `episode_index` 扁平索引）；视频解码复用官方 lerobot `decode_video_frames` |
| 7 | **数据入口** | 统一 `DATASET_PATH`：`.jsonl` 文件 → manifest 模式（每行一个数据集 path）；目录 → 单数据集根/父目录递归 |
| 8 | 解码后端 | `video_backend="pyav"`（torchcodec 当前 NPU 环境不可用）；decoder LRU 缓存仅 torchcodec 生效 |

---

## 2. 最终文件组织

新增逻辑全部放在两个新文件里，原文件保持不动：

```
cosmos_framework/data/generator/local_datasets/
  ├── sft_dataset.py            # 原文件（JSONL/S3 流程），一行未改
  └── sft_dataset_lerobot3.py   # ★ 新增：LeRobot 动态加载（924 行）

cosmos_framework/configs/base/experiment/sft/
  ├── vision_sft_edge.py        # 原文件（JSONL 流程），一行未改
  └── vision_sft_edge_lerobot3.py  # ★ 新增：LeRobot experiment（256 行）

cosmos_framework/configs/base/config.py     # 加 1 行 import（注册新 experiment）
cosmos_framework/configs/toml_config/sft_config.py  # DataloaderTrainConfig 新增 use_multi_resolution/use_multi_fps 两字段
cosmos_framework/configs/toml_config/toml_config_helper.py  # PATH_REMAPS 新增两条 remap，把 toml 开关路由到 dataset 节点
cosmos_framework/configs/base/experiment/sft/models/edge_model_config.py  # vae_path 改绝对路径（环境相关，非本特性）
examples/toml/sft_config/vision_sft_edge.toml  # experiment 指向新名字 + 多分辨率/fps 开关 + token 预算放大
examples/launch_sft_vision_edge_yundao_lerobot.sh  # 启动脚本（DATASET_PATH 统一入口）
examples/_sft_launcher_common.sh            # 公共启动脚本，**未改动**（已回滚）
```

### 2.1 `sft_dataset_lerobot3.py` 内容

| 符号 | 行号 | 作用 |
|------|------|------|
| `_MULTI_RESOLUTION_TIERS` / `_MULTI_FPS_INTERVALS` | 51 / 53 | 多分辨率档位（256/480）/ 多 fps 间隔（2/3/4） |
| `_LRU_VIDEO_CACHE_MAX_SIZE` | 58 | decoder LRU 缓存容量（仅 torchcodec 生效） |
| `_ensure_hf_hub_offline` | 64 | 强制 HF Hub 离线（幂等，只加载本地数据集） |
| `_LRUVideoDecoderCache` | 79 | LRU 版 torchcodec decoder 缓存（`seek_mode="exact"`，pyav 下 no-op） |
| `_patch_decoder_cache` | 151 | 把 lerobot 模块级无界缓存替换为 LRU 版（pyav 下 no-op） |
| `_LerobotSource` | 172 | dataclass：数据集级常量 + 官方 meta 对象，每数据集一份 |
| `_select_lerobot_video_key` | 195 | 选定 video 字段（用官方 `meta.video_keys`；显式 → 关键字 → 第一个） |
| `_discover_lerobot_roots` | 228 | 单数据集根 or 父目录多数据集发现 |
| `_build_lerobot_source` | 253 | 读单个数据集 → `(source, valid_eps)`，不物化 episode dict |
| `_load_lerobot_metadata` | 322 | 目录入口：发现多个数据集 → 产出 `(sources, episode_index)` |
| `_load_lerobot_metadata_from_manifest` | 362 | **manifest 入口**：读 JSONL，每行一个 path，并行加载合并 |
| `LeRobotSFTDataset(IterableDataset)` | 448 | **独立** IterableDataset，不继承 `SFTDataset` |
| `get_sft_dataset_from_lerobot` | 832 | LeRobot 版入口，按 `dataset_path` 后缀分流，构造 `LeRobotSFTDataset` |

### 2.2 复用父模块符号（不重复实现）

新文件从 `sft_dataset.py` **只 import 纯函数**（不继承 `SFTDataset`）：

```python
from cosmos_framework.data.generator.local_datasets.sft_dataset import (
    _DURATION_TEMPLATE,          # caption 时长/FPS 模板
    _MAX_CAPTION_TOKENS,         # caption token 上限
    _RESOLUTION_TEMPLATE,        # caption 分辨率模板
    _select_caption,             # t2w_window 里选 caption key
)
```

其它 import（同样来自既有模块，非重复实现）：

```python
from cosmos_framework.data.generator.local_datasets.helper import get_aspect_ratio
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.data.generator.sequence_packing.modalities import add_special_tokens
from cosmos_framework.data.generator.utils import VIDEO_RES_SIZE_INFO
from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import tokenize_caption
from lerobot.datasets import video_utils as _vu                 # decode_video_frames + get_safe_default_codec
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # 读 info/episodes
```

### 2.3 `LeRobotSFTDataset` 是独立实现（不继承 `SFTDataset`）

早期版本通过继承 `SFTDataset` 并 override `process_one_sample` 复用逻辑，但父类是单体方法（无钩子），导致要逐字复制整个方法、且 S3 相关属性（`boto3`、`s3_client`）被白带进来。最终改为**独立 `IterableDataset`**：

| 项 | 处理方式 |
|----|----------|
| 类声明 | `class LeRobotSFTDataset(torch.utils.data.IterableDataset)` |
| `__init__` | 自实现（含 tokenizer 实例化、shard 占位属性），**无 S3 依赖** |
| `__len__` | `len(self.episode_index)` |
| `_tokenize_caption` | 自封装 `tokenize_caption` + 截断 |
| `__iter__` | 复制父类 shard 脚手架（分区/shuffle/无限循环），删掉 boto3 |
| `process_one_sample(ds_idx, ep_idx)` | 惰性化采样：现算 episode 字段，走官方 `decode_video_frames` |
| 复用纯函数 | `_select_caption` / `_DURATION_TEMPLATE` / `_RESOLUTION_TEMPLATE` / `_MAX_CAPTION_TOKENS` |

这样保留了 `RankPartitionedDataLoader` 依赖的 `shard_world_size`/`shard_rank`/`shard_id` + `__iter__` 约定，同时彻底去掉 S3 流程的耦合。

---

## 3. 数据流与惰性化加载回顾

### 3.1 惰性化加载的核心思路

LeRobot 版**不再在加载期物化 episode dict**，而是把样本信息拆成两层：

| 层 | 存什么 | 何时产生 |
|----|--------|---------|
| **数据集级常量**（`_LerobotSource`） | root / meta / video_key / width / height / fps / aspect_ratio / total_frames / caption_key / root_hash / name | 加载期，每数据集一份 |
| **扁平 episode 索引** | `episode_index: list[tuple[int, int]]`，每项 `(ds_idx, ep_idx)` | 加载期，每 episode 一个 2 元组 |
| **episode 级字段** | uuid / vision_path / start_frame / end_frame / caption | **采样期**由 `process_one_sample(ds_idx, ep_idx)` 从 `meta.episodes[ep_idx]` 现算 |

数据流：

```
加载期：
  _build_lerobot_source(root) → (_LerobotSource, valid_eps)   # 只解析常量 + 跑过滤，不造 dict
  _load_lerobot_metadata(...)  → (sources, episode_index)      # 合并成扁平索引
  _load_lerobot_metadata_from_manifest(...) → 同上（ThreadPoolExecutor 并行）

采样期（__iter__ → process_one_sample(ds_idx, ep_idx)）：
  source = sources[ds_idx]          # 取数据集级常量
  ep = source.meta.episodes[ep_idx] # mmap 按行取
  → 现算 uuid / vision_path / start/end_frame / caption
  → decode_video_frames → resize → crop → tokenize → ret
```

### 3.2 与早期「加载期物化」的对比

| 阶段 | 早期（物化） | 现在（惰性化） |
|------|-------------|---------------|
| 加载期产出 | `list[dict]`，每 episode 一个完整 dict（~15 字段 + 嵌套 window） | `sources` + `episode_index`（每 episode 一个 `(int,int)` 元组） |
| 数据集级字段 | 每 episode 重复存一份（7 字段 × N） | 每数据集存一份 |
| episode 级字段 | 提前算好存 dict | 采样期从 `meta.episodes[ep_idx]` 现算 |
| 内存 | 大数据集下几十~上百 MB Python 对象 | 主要是 mmap + 轻量元组列表 |

**收益**：省内存 + 省加载期耗时；**代价**：采样期多一次 mmap 行取 + 字段现算（被视频解码开销覆盖，可忽略）。shard/shuffle 算法不变，只是被 shuffle 的对象从重 dict 列表换成轻索引。

---

## 4. 字段映射方案（LeRobot → 样本）

> 字段映射分两层：**数据集级**（width/height/fps/aspect_ratio 等）在 `_build_lerobot_source` 解析进 `_LerobotSource`；**episode 级**（uuid/vision_path/start/end_frame/caption）在 `process_one_sample` 采样期现算。

### 4.1 uuid（采样期现算）

**格式**：`{dataset_name}_{root_hash}_chunk_{chunk_index}_file_{file_index}_episode_{episode_index}`

```python
# source.root_hash = hashlib.sha256(str(root)).hexdigest()[:8]  # 加载期算好存进 source
uuid = f"{source.name}_{source.root_hash}_chunk_{data_chunk}_file_{data_file}_episode_{episode_index}"
```

- 来源：数据集目录名（`source.name`）+ **完整路径的 sha256 前 8 位**（`source.root_hash`）+ episodes 表的三列
- 目的：跨**数据集**、跨 chunk、跨 file、跨 episode 唯一

> episode_index 单数据集内唯一，但**多数据集合并时**：
> 1. 不同数据集可能有相同编号 → 加数据集目录名。
> 2. **不同父目录下可能有同名数据集目录** → 光有目录名不够，再加**完整路径的短 hash** 保证唯一。

### 4.2 vision_path（采样期现算）

用官方 lerobot 的 `meta.get_video_file_path(ep_idx, video_key)` 拼出本地 mp4 路径：

```python
input_video_path = str(source.root / meta.get_video_file_path(ep_idx, source.video_key))
# 例：.../videos/observation.images.top/chunk-000/file-000.mp4
```

`ep_idx` 是 episode 在 `meta.episodes` 里的位置，`source.video_key` 是选定的相机字段名。

### 4.3 width / height（加载期解析进 source）

在 `_build_lerobot_source` 里用官方 `meta.shapes` / `meta.names` 解析（替代早期手写 `info["features"]` 解析）：

```python
shape = meta.shapes[video_key]        # [H, W, C] 或 [H, W]
names = meta.names.get(video_key)     # 通常 ["height","width","channels"]
if names and "width" in names and "height" in names:
    width = shape[names.index("width")]
    height = shape[names.index("height")]
else:
    height, width = shape[0], shape[1]   # 兜底：默认 [H, W, C]
```

- toy 数据：`shape=[480, 640, 3]` → `width=640, height=480`
- 同一数据集各相机分辨率一致，所以取选定相机的 shape 即可，存进 `source.width`/`source.height`（只一份）。

### 4.4 aspect_ratio

复用 `get_aspect_ratio(width, height)`（`helper.py`），按宽高比分桶成 `"1,1"/"4,3"/"3,4"/"16,9"/"9,16"`。

### 4.5 帧区间（采样期现算，存帧编号）

`start_frame`/`end_frame` 存的是**帧编号**（0-based，闭区间），在 `process_one_sample` 里现算：

```python
fps = source.fps   # 30

# 时间区间来自 episodes 表里选定相机对应的列（不是裸 from/to_timestamp）：
from_ts = float(ep.get(f"videos/{video_key}/from_timestamp", 0.0))
to_ts   = float(ep.get(f"videos/{video_key}/to_timestamp", 0.0))

start_frame = round(from_ts * fps)     # 含
end_frame   = round(to_ts * fps) - 1   # to 是开区间，-1 变闭区间
```

验证（toy 数据）：

| episode | from | to | start | end | length | 吻合 |
|---------|------|-----|-------|-----|--------|------|
| 0 | 0.0 | 18.466667 | 0 | 553 | 554 | ✅ |
| 1 | 18.466667 | 36.133333 | 554 | 1083 | 530 | ✅ |

> `to_timestamp` 是**开区间**，该时刻的帧属于下一个 episode，所以 `end_frame = round(to×fps) - 1`。

构造出的 window（恒单元素，一个 episode = 一个 window）：

```python
t2w_window = {"start_frame": start_frame, "end_frame": end_frame, "temporal_interval": 1}
if caption:                     # caption 为空时不写 key，下游优雅跳过
    t2w_window["caption"] = caption
```

### 4.6 caption 处理逻辑（`caption_key` 优先 + `tasks` 回退）

```python
caption = ep.get(source.caption_key)   # 优先读 episodes 表里 caption_key 指定的列

if not caption:
    tasks = ep.get("tasks")            # 回退官方 tasks 列（episode 级多任务 list）
    if tasks is not None and hasattr(tasks, "__len__") and len(tasks) > 0:
        caption = str(tasks[0])        # 取第一个任务名
```

> 关键边界行为：caption 为空时，`t2w_window` 里**没有** `caption` key，下游 `_select_caption` 找不到已知 key → `return None` → `process_one_sample` 跳过该样本。**不写 `caption: None`**，否则下游 `raw.strip()` 会 `AttributeError` 崩溃。

> `tasks` 列是官方 episodes 表原生列，实际类型为 `numpy.ndarray`（非 list），用 `hasattr(x, "__len__")` 判断。

### 4.7 统一数据入口（`dataset_path` 分流）

`get_sft_dataset_from_lerobot` 的入口参数统一为 `dataset_path`，按后缀自动分流：

```python
if dataset_path.endswith(".jsonl"):
    sources, episode_index = _load_lerobot_metadata_from_manifest(dataset_path, ...)  # manifest 模式
else:
    sources, episode_index = _load_lerobot_metadata(dataset_path, ...)                # 目录模式
```

| 传入值 | 类型 | 走哪条逻辑 | 行为 |
|--------|------|-----------|------|
| `xxx.jsonl` | 文件 | `_load_lerobot_metadata_from_manifest` | 逐行读 path，加载所有数据集 |
| 单数据集根（含 `meta/info.json`） | 目录 | `_discover_lerobot_roots` 第 1 分支 | 返回 `[root]` |
| 父目录（含多个数据集） | 目录 | `_discover_lerobot_roots` 第 2 分支 | `rglob("meta/info.json")` 递归发现 |

### 4.8 manifest 文件格式（JSONL）

manifest 是 JSONL 文件，**每行一个 dict**，每行支持以下 key（其余 key 静默忽略）：

| key | 必需 | 含义 | 缺省回退 |
|-----|------|------|---------|
| `path` | ✅ 必需 | 数据集路径（单数据集根 or 父目录） | 无（缺则 warning 跳过） |
| `video_feature_key` | 可选 | 显式指定 feature 名 | config 全局值（`None`） |
| `video_feature_keywords` | 可选 | 关键字 list | config 全局值（`["top","head"]`） |
| `caption_key` | 可选 | caption 列名 | config 全局值（`"caption"`） |

```jsonl
{"path": "/data/dataset_a", "video_feature_keywords": ["side"]}
{"path": "/data/dataset_b", "video_feature_key": "observation.images.wrist", "caption_key": "my_caption"}
{"path": "/data/dataset_c"}
```

`_load_lerobot_metadata_from_manifest` 逻辑：

```python
with open(manifest_path, "r") as f:
    for line_no, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        path = entry.get("path")
        if not path:
            log.warning(f"manifest 第 {line_no} 行缺少 'path' key，跳过")
            continue
        # 三个参数逐行覆盖，缺省回退 config 全局值
        row_feature_key = entry.get("video_feature_key", video_feature_key)               # 显式 feature 名
        row_feature_keywords = entry.get("video_feature_keywords", video_feature_keywords) # 关键字 list
        row_caption_key = entry.get("caption_key", caption_key)                           # caption 列名
        tasks.append((path, row_feature_key, row_feature_keywords, row_caption_key))
```

关键点：

- **逐行覆盖**：每个数据集可以用不同的视角选择方式、不同的 caption 列名，不用为一个全局值反复改 config。
- **缺省回退**：某行没写某个参数时，回退到 `get_sft_dataset_from_lerobot` 传入的全局值（来自 config）。
- 每个 `path` 又可以是单数据集根 or 父目录（复用目录模式的能力）。
- 缺 `path` 的行会 warning 并跳过，不会中断整个加载。
- 不认识的其他 key（如 `name`/`description`）静默忽略。

#### 并行加载（已落地）

manifest 多个数据集的加载用 `ThreadPoolExecutor` 并行（新增 `manifest_max_workers` 参数，默认 `min(len(tasks), 8)`）：

```python
# 第 1 步：解析 manifest → 任务列表（纯 json 解析，串行很快）
tasks = [(path, fk, fkw, ck), ...]

# 第 2 步：并行加载每个 path
results = ex.map(_load_one, tasks)   # 每个 _load_one 返回 (sources, episode_index)
# 第 3 步：合并，做全局 ds_idx 偏移
sources, episode_index = _merge(results)
```

**为什么选多线程而非多进程**：

| 环节 | 性质 | 多线程能否并行 |
|------|------|--------------|
| `LeRobotDatasetMetadata`（info.json + episodes parquet I/O + C++ 解压） | 会释放 GIL | ✅ 能并行 |
| 字段映射（Python 纯代码） | 受 GIL 限制 | ❌ 不能 |

因为耗时大头是 `LeRobotDatasetMetadata` 加载（I/O 密集、释放 GIL），`ThreadPoolExecutor` 就能吃到大部分收益，且免去多进程的 pickle 开销、冷启动和日志乱序。

**关键设计**：
- `ex.map` 保持输入顺序，结果顺序与 manifest 行顺序一致
- `workers<=1` 或只有 1 个数据集时，走原串行路径（零并发开销）
- 只进入并行分支时打一条 `[manifest] 并行加载 N 个数据集` 日志

> 若未来实测「线程加速不明显」（说明瓶颈在字段映射的 Python 代码），再升级为 `ProcessPoolExecutor`。

### 4.9 三个参数的默认值来源（两层）

`video_feature_key` / `video_feature_keywords` / `caption_key` 有**两层默认值**：

| 层 | 位置 | 值 |
|----|------|-----|
| 函数签名默认值 | `sft_dataset_lerobot3.py:850-852` | `None` / `None` / `"caption"` |
| **experiment 显式传入**（实际生效） | `vision_sft_edge_lerobot3.py:231-233` | `None` / `["top","head"]` / `"task"` |

experiment 显式传了这三个参数，所以**真正生效的是 experiment 的值**（其中 `video_feature_keywords` 用 `["top","head"]` 覆盖了签名默认 `None`；`caption_key` 用 `"task"` 覆盖了签名默认 `"caption"`——真实数据集的 caption 列名叫 `task`）。这三个 experiment 值就是 manifest 逐行缺省时的回退值。

### 4.10 多分辨率 + 多 fps 训练（已落地）

通过两个 bool flag 控制，toml `[dataloader_train]` 里开关，默认关闭（向后兼容单值模式）。

```toml
[dataloader_train]
use_multi_resolution = true   # 多分辨率：256/480 随机
use_multi_fps = true          # 多 fps：temporal_interval 随机 2/3/4
```

#### 候选常量（`sft_dataset_lerobot3.py:51-53`）

```python
_MULTI_RESOLUTION_TIERS = ("256", "480")   # 分辨率档位（短边），720 已移除
_MULTI_FPS_INTERVALS = (2, 3, 4)           # 抽帧间隔（保留 1/2、1/3、1/4）
```

#### 分辨率随机选择（`process_one_sample`）

```python
if self.use_multi_resolution:
    video_min_edge = min(input_w, input_h)
    candidates = [r for r in _MULTI_RESOLUTION_TIERS if int(r) <= video_min_edge]  # 不上采样
    if not candidates:
        candidates = ["256"]                       # 视频太小 fallback
    output_sizes = VIDEO_RES_SIZE_INFO[random.choice(candidates)]
else:
    output_sizes = self.output_sizes               # 单值模式
```

**不上采样**：候选 = 所有 ≤ 视频短边的档位。

| 视频 | 候选档位 |
|------|---------|
| 1080p / 720p / 480p | 256 / 480 |
| 360p | 256 |
| 超小（短边<256） | fallback 256 |

#### fps 随机选择（`process_one_sample`）

```python
if self.use_multi_fps:
    temporal_interval = random.choice(_MULTI_FPS_INTERVALS)  # 随机 2/3/4
else:
    temporal_interval = t2w_window["temporal_interval"]       # = 1
```

抽帧通过「帧号列表步长」实现（`list(range(start_frame, end_frame+1, temporal_interval))`），在解码前就把帧号按 `temporal_interval` 采样，再交给 lerobot 解码（见 5.3）。

> 注意：已删除 `num_video_frames` 参数（固定抽 N 帧的 JSONL 兼容能力），LeRobot 恒用 native chunk 模式。

#### shift 自适应（无需改动）

`edge_model_config.py` 已配置 `shift={"256":3, "480":5, "720":10}`，模型侧按 resize 后尺寸经 `get_vision_data_resolution` 反推档位、自动查对应 shift 值，与原始视频分辨率无关。当前候选档位只用了 256/480 两档（720 已从 `_MULTI_RESOLUTION_TIERS` 移除），仍落在 shift 的 key 里，无 KeyError 风险。

#### 传递链路（flag → dataset）

```
toml [dataloader_train].use_multi_resolution
  → PATH_REMAPS["vfm"] 路由
  → dataloader_train.dataloader.datasets.video.dataset.use_multi_resolution
  → get_sft_dataset_from_lerobot(use_multi_resolution=...) → LeRobotSFTDataset
```

#### 边界

- `use_multi_fps` 直接作用于 native chunk 抽帧步长（已无 `num_video_frames>0` 分支）。
- `resolution` 参数**保留**，作为单分辨率模式的 fallback（`use_multi_resolution=False` 时生效）。

### 4.11 episode 过滤阈值（`min_frames` + `max_duration_s`）

加载期在 `_build_lerobot_source` 里对每个 episode 做两层过滤，阈值已提取为可配置参数并暴露到 toml：

| 参数 | 单位 | 方向 | 默认 | 作用 | 等价（30fps） |
|------|------|------|------|------|--------------|
| `min_frames` | 帧 | 下界 | 61 | 丢弃帧数 < 61 的短 episode | ~2 秒 |
| `max_duration_s` | 秒 | 上界 | 61.0 | 丢弃时长 > 61 秒的长 episode | ~1830 帧 |

> 注意：两者数值巧合都是 61，但**单位与过滤方向完全不同**——`min_frames` 管「太短」（帧），`max_duration_s` 管「太长」（秒）。

过滤逻辑（`sft_dataset_lerobot3.py` 的 `_build_lerobot_source`）：

```python
duration = to_ts - from_ts                    # 秒
if duration > max_duration_s:                 # 上界：时长超限
    continue
frames_in_window = end_frame - start_frame + 1  # 帧
if frames_in_window < min_frames:             # 下界：帧数不足
    continue
```

传递链路（与多分辨率/多 fps 同构）：

```
toml [dataloader_train].min_frames / .max_duration_s
  → PATH_REMAPS["vfm"] 路由
  → dataset.min_frames / dataset.max_duration_s
  → get_sft_dataset_from_lerobot(min_frames=..., max_duration_s=...)
  → _load_lerobot_metadata(_from_manifest) → _build_lerobot_source
```

#### ⚠️ max_sequence_length 必须配套放大

开启多分辨率后，480 档位单样本 token 数会超过默认的 `max_sequence_length=45056`，导致样本被 `PackingDataLoader` 丢弃（日志报 `Discarding oversized sample`）。

**token 预算表**（demo 640×480 数据，4:3，T=1341 帧）：

| 分辨率档 | patch 数 | interval=2 时单样本 token | 是否超 45056 |
|---------|---------|--------------------------|-------------|
| 256 | 80 | ~13440 | ✅ 不超 |
| 480 | 391 | ~65688 | ❌ 超 |

**必须同步改 toml 两处**（`vision_sft_edge.toml`）：

```toml
[model]
max_num_tokens_after_packing = 65760    # 从 45056 放大

[dataloader_train]
max_sequence_length = 65760             # 从 45056 放大（真正生效的 budget）
```

- `max_sequence_length` 是 **PackingDataLoader 真正生效**的 token 预算，必须放大。
- `max_num_tokens_after_packing` 在 vision SFT 里 dataloader 不直接引用，但为语义一致也同步放大。
- 放大到 65760 正好覆盖 **480 档**（~65688 token）；720 档已从候选移除，无需 ~196608+ 的预算。

> 注意：序列长度放大后 attention 显存按序列长度平方增长，需确认显存够用。

---

## 5. 视频解码（复用官方 lerobot `decode_video_frames`）

> ⚠️ **本章已重写**。早期方案（旧版第 5 章）是「自建 torchcodec decoder + `get_frames_in_range` 按帧编号 seek + 自写 resize」。最终落地**放弃该方案**，改为直接复用官方 lerobot 的 `decode_video_frames` 按绝对时间戳解码。下面是实际实现。

### 5.1 问题

父类 `process_one_sample` 的视频解码是「全量 decode + 按帧编号过滤」：

```python
for idx, frame in enumerate(ffmpeg_decode_video(input_video_path, ...)):
    if idx < start_frame: continue
    elif idx <= end_frame: ...
```

toy 数据一个 mp4 含 50 个 episode，每个 episode 都全量解码同一文件 → 50 次全量解码，极低效。

### 5.2 方案：复用官方 lerobot（非自建 seek）

最终落地放弃「自建 decoder + 帧编号 seek + 自写 resize」，改为直接复用官方 lerobot：

- 按**绝对时间戳**（非帧编号）解码，由 lerobot 内部用 torchcodec/pyav + 自带 decoder cache 实现。
- 帧号在 `process_one_sample` 里先转成 timestamp（`idx / original_fps`）再交给 lerobot。
- 解码、resize、dtype 归一化（返回 `[T,C,H,W]` float ∈ [0,1]）都交给官方实现，避免自建缓存与 resize 的维护成本。

### 5.3 帧号 → 时间戳 → 解码（核心流程）

`process_one_sample` 里（`sft_dataset_lerobot3.py:621-631`）：

```python
# 帧号 → 绝对时间戳
frame_indices = list(range(start_frame, end_frame + 1, temporal_interval))
timestamps = [idx / original_fps for idx in frame_indices]

# 交给 lerobot 按时间戳解码
video_frames = _vu.decode_video_frames(
    input_video_path,
    timestamps,
    tolerance_s=self.tolerance_s,
    backend=self.video_backend,
)  # [T, C, H, W] float32 ∈ [0,1]
```

关键点：

- `start_frame`/`end_frame` 仍存帧编号（见 4.5），解码前才转成绝对时间戳。
- **抽帧**通过「帧号列表步长」实现（`range(..., temporal_interval)`），而非解码层跳帧。
- `backend`：`video_backend` 参数控制，experiment 配 `"pyav"`（`vision_sft_edge_lerobot3.py:241`）；默认 `_vu.get_safe_default_codec()`（torchcodec 可用则 torchcodec，否则 pyav）。

### 5.4 解码容错

lerobot 内部对「时间戳与视频 pts 偏差超过 `tolerance_s`」会 `assert` 抛 `AssertionError`；其它坏文件/解码器异常抛通用 `Exception`。`process_one_sample` 里两者都 catch，打印 warning 并跳过该样本，避免中断训练（`sft_dataset_lerobot3.py:624-653`）：

```python
try:
    video_frames = _vu.decode_video_frames(...)
except AssertionError as e:
    log.warning(...); return None   # 时间戳超出 tolerance
except Exception as e:
    log.warning(...); return None   # 坏文件 / 解码器异常

if video_frames.shape[0] == 0:      # 空解码结果也跳过
    log.warning(...); return None
```

### 5.5 decoder 缓存：LRU 替换 lerobot 无界缓存（⚠️ 仅 torchcodec 生效）

lerobot 模块级 `_vu._default_decoder_cache` 是**无界 dict、只加不删**，多 worker 场景下 decoder 索引 + FFmpeg 上下文持续累积导致内存上涨。`_patch_decoder_cache`（`sft_dataset_lerobot3.py:151`）把它替换成本地 LRU 版 `_LRUVideoDecoderCache`（`max_size=64`，`seek_mode="exact"`）：

```python
_vu._default_decoder_cache = _LRUVideoDecoderCache(max_size=max_size)
```

`_LRUVideoDecoderCache`（`sft_dataset_lerobot3.py:79`）要点：

1. **LRU 用 `OrderedDict`**：`move_to_end`（命中标记最近）+ `popitem(last=False)`（淘汰最久未用）。
2. **缓存 `(decoder, file_handle)` 二元组**：构造失败时显式 `close()` 防坏文件累积 fd；淘汰时 `del old_decoder` + `close()`。
3. **`max_size=64`**：限制同时打开的 decoder 数量；带 `Lock` 保证多线程安全。

> 与 action 侧 `_LRUVideoDecoderCache` 唯一差异：`seek_mode="exact"`（vision SFT 要精确切 episode 帧边界；action 对精确帧不敏感，用 `approximate`）。

> ⚠️ **重要：本缓存仅在 torchcodec 后端生效。** lerobot 的 pyav 路径（`decode_video_frames_torchvision`）每次调用都新建 `VideoReader` 并 close，**完全不查 `_default_decoder_cache`**。当前 experiment 配 `video_backend="pyav"`，所以 `_patch_decoder_cache` / `_LRUVideoDecoderCache` / `decoder_cache_max_size` 全是 **no-op**。保留它们是给将来切 torchcodec 时防无界内存膨胀（方案 B）。

### 5.6 后处理（resize + 转 uint8）

lerobot 返回 `[T,C,H,W]` float ∈ [0,1]，仍需 resize 到目标尺寸并转 uint8。这段代码在 `process_one_sample` 内（`sft_dataset_lerobot3.py:655-661`）：

```python
import torch.nn.functional as F

video_frames = video_frames.float()
video_frames = F.interpolate(video_frames, size=(resize_h, resize_w), mode="bicubic", align_corners=False)
video_frames = video_frames.round().clamp(0, 255).to(torch.uint8)
video_chunk = video_frames.permute(0, 2, 3, 1).cpu().numpy()  # [T,H,W,3] uint8
```

`F.interpolate(mode="bicubic")` 对齐原版 ffmpeg 的 `-vf scale + bicubic`。

### 5.7 内存账（为什么不能全量 decode 缓存）

toy 一个 mp4：24263 帧 × 480 × 640 × 3 = **约 22.4 GB**（解压后 RGB）。全量缓存会 OOM。lerobot 的 `decode_video_frames` 也是按需解码，不缓存全量帧；缓存的只是 **decoder**（且仅 torchcodec），不是解码后的帧数据，所以内存安全。

---

## 6. 已确认的风险点与注意事项

> 本节集中记录**通过源码/数据验证**得出的易踩坑点，供后续版本迭代参考。

### 6.1 数据格式层（LeRobot）

| # | 风险点 | 结论 | 依据 |
|---|--------|------|------|
| 1 | v3 命名单数还是复数 | **复数 `observation.images.*`**（不是单数 `image`） | toy 数据实测 |
| 2 | 一个 mp4 含几个 episode | **多个 episode 共享一个 mp4**，靠 `from/to_timestamp` 切分 | toy：50 episode 共享 file-000.mp4 |
| 3 | `to_timestamp` 开闭区间 | **开区间**：该时刻帧属于下一 episode | toy：`to×fps` 精确等于下一集起始帧 |
| 4 | `length` / 行跨度 / 时间跨度 | 精确相等：`length = round((to-from)×fps)` | toy 数据验证 |
| 5 | `frame_index` 全局还是 episode 内 | **episode 内从 0 重新开始** | toy：episode 1 首帧 frame_index=0 |

### 6.2 字段映射层

| # | 风险点 | 结论 |
|---|--------|------|
| 1 | `shape` 是 H×W 还是 W×H | **`[H, W, C]`**，`width=shape[1]`, `height=shape[0]` |
| 2 | 视频字段 key 能否硬编码 | **不能**（`observation.images.top` 是数据集特定的），用关键字匹配/兜底动态选 |
| 3 | `end_frame` 语义 | 存**帧编号**（文件内），非 timestamp；`end = round(to×fps) - 1` |
| 4 | `aspect_ratio` | `get_aspect_ratio(width,height)` 派生，JSONL/LeRobot 里没有现成字段 |

### 6.3 加 caption 的侵入位置

| # | 风险点 | 结论 |
|---|--------|------|
| 1 | 加到 episodes 表 | ✅ **安全**，官方 lerobot 加载不报错（已实测） |
| 2 | 加到 data 表 | ❌ **禁止**，破坏 `info.json` schema 校验，报 `CastError`（已实测） |
| 3 | `info.json` 的 `features` 约束范围 | **只严格约束 data 表**，不约束 episodes 表 |
| 4 | **pandas 3.0.5 破坏嵌套 list 列** | ❌ 不能用 pandas 3.0.5 读写 episodes parquet（`stats/*/*` 列损坏）。加 caption 用 **pyarrow**（append_column）或 **pandas 2.x** |
| 5 | `tasks` 列实际类型 | **`numpy.ndarray`**（不是 list/tuple），判断用 `hasattr(x, "__len__")` |
| 6 | **toy 数据集 stats 列 parquet 编码损坏** | 磁盘 def/rep level 损坏，重写整表必然失败。**解法：drop 掉 `stats/*/*` 列**，只保留有效列 + caption |

### 6.4 视频解码层（lerobot `decode_video_frames`）

| # | 风险点 | 结论 |
|---|--------|------|
| 1 | 解码 API | 官方 `lerobot.datasets.video_utils.decode_video_frames`，按**绝对时间戳**解码（非帧编号 seek） |
| 2 | 返回形状/dtype | `[T, C, H, W]` float32 ∈ [0,1]（**不是 uint8 也不是 `[N,H,W,C]`**），后处理 resize + 转 uint8（见 5.6） |
| 3 | seek_mode | `_LRUVideoDecoderCache` 用 **`exact`**（精确帧定位），不是 action 侧的 `approximate`（**但仅 torchcodec 生效**） |
| 4 | decoder 缓存 | lerobot 模块级无界缓存 → 替换为 LRU 版 `_LRUVideoDecoderCache`（防内存膨胀）；**pyav 下 no-op**（见 5.5） |
| 5 | 时间戳偏差 | 偏差超过 `tolerance_s` 会抛 `AssertionError`；已 catch 并跳过坏样本（见 5.4） |
| 6 | 不能全量 decode 缓存 | 一个 mp4 解压后 ~22GB（toy），会 OOM；缓存的只是 decoder，不是解码帧 |
| 7 | 抽帧越界 | 已删除固定抽 N 帧（`num_video_frames>0`）能力，native chunk 下 `end_frame` 由 `to_timestamp` 精确切分，不再有抽帧跨度越界问题 |

### 6.5 尚未验证、版本迭代时需重点关注

| # | 风险点 | 说明 |
|---|--------|------|
| 1 | 多文件场景的 `from_timestamp` 语义 | toy 是单 mp4，无法验证；真实多文件数据需确认是全局时间还是文件内相对时间 |
| 2 | 抽帧语义 | lerobot 按时间戳解码，抽帧通过「帧号列表步长」（`range(start, end+1, interval)`）实现 |
| 3 | 惰性化的 `meta` 对象 pickle | `_LerobotSource.meta` 含 HF Dataset mmap + pandas DataFrame；fork 下 worker 直接继承没问题，若切 spawn 需确认可 pickle |

---

## 7. 关键结论速查

| 问题 | 结论 |
|------|------|
| 类形态 | `LeRobotSFTDataset` 是**独立 `IterableDataset`**，不继承 `SFTDataset`；自实现 `__init__`/`__len__`/`__iter__`/`process_one_sample` |
| 惰性化加载 | 加载期只产出 `(sources, episode_index)` 扁平索引，episode 字段采样期从 `meta.episodes[ep_idx]` 现算 |
| uuid 格式 | `{dataset_name}_{root_hash}_chunk_{chunk_idx}_file_{file_idx}_episode_{ep_idx}`（目录名 + 完整路径短 hash，跨数据集/同名目录唯一） |
| 统一数据入口 | `dataset_path`：`.jsonl` → manifest 模式；目录 → 单数据集根/父目录递归 |
| 多数据集加载 | 目录支持单数据集根 or 父目录，父目录自动 `rglob("meta/info.json")` 递归；manifest 每行一个 path，并行加载合并 |
| 选哪路视频 | `_select_lerobot_video_key`：用官方 `meta.video_keys`；显式 key > 关键字匹配 > 第一个 video 字段 |
| width/height 来源 | 官方 `meta.shapes`/`meta.names`，`width=shape[names.index("width")]`，存进 `_LerobotSource`（每数据集一份） |
| caption 来源 | `caption_key` 列优先，回退官方 `tasks` 列第一个任务名 |
| 视频解码怎么做 | 复用官方 lerobot `decode_video_frames` 按**绝对时间戳**解码；帧号先转 timestamp |
| decoder 缓存 | LRU 版 `_LRUVideoDecoderCache`（max_size=64，`seek_mode="exact"`），**仅 torchcodec 生效，pyav 下 no-op** |
| 加 caption 到哪 | episodes 表加列（安全）；**禁止加到 data 表** |
| 多分辨率训练 | `use_multi_resolution=True` 时在 256/480 随机（只选 ≤ 视频短边，不上采样；720 已移除） |
| 多 fps 训练 | `use_multi_fps=True` 时 `temporal_interval` 随机 2/3/4（保留 1/2、1/3、1/4），通过帧号列表步长抽帧 |

---

## 8. 训练接入改动

### 8.1 新增 experiment：`vision_sft_edge_lerobot3.py`

基于原 `vision_sft_edge.py` 派生（新文件 256 行），核心只改 3 处：

| 位置 | 改动 |
|------|------|
| import（32 行） | `from ...sft_dataset_lerobot3 import get_sft_dataset_from_lerobot` |
| `dataset=`（220 行） | `L(get_sft_dataset_from_lerobot)(...)`，参数用统一入口 `dataset_path` + `video_feature_key` + `video_feature_keywords` + `caption_key` |
| `job.name`（70 行） | `"vision_sft_edge_lerobot3"` |

关键参数（第 220-241 行）：

```python
dataset=L(get_sft_dataset_from_lerobot)(
    ...
    dataset_path="${oc.env:DATASET_PATH}",       # 统一入口：.jsonl→manifest；目录→单数据集根/父目录
    video_feature_key=None,                      # 显式指定 feature 名（精确匹配）；None 则不显式指定
    video_feature_keywords=["top", "head"],      # 关键字 list：key 名含任一关键字即选中；匹配不到回退第一个 video
    caption_key="task",                          # episodes 表里的 caption 列名（真实数据集用 task 列）
    min_frames=61,                               # episode 过滤下界（帧）
    max_duration_s=61.0,                         # episode 过滤上界（秒）
    resolution="256",                            # 单分辨率 fallback（use_multi_resolution=False 时生效）
    use_multi_resolution=False,                  # 多分辨率开关（experiment 默认 False，被 toml 覆盖为 true）
    use_multi_fps=False,                         # 多 fps 开关（experiment 默认 False，被 toml 覆盖为 true）
    conditioning_fps=-1,                         # caption 时长/FPS 用实际有效帧率（original_fps/interval）
    video_backend="pyav",                        # 解码后端（torchcodec 当前 NPU 环境不可用）
    ...
)
```

> 说明：
> - `use_multi_resolution`/`use_multi_fps` 在 experiment 里写 `False`，但 `vision_sft_edge.toml` 里是 `true`，经 `PATH_REMAPS` 覆盖到 dataset 节点后**实际生效为 true**（见 8.3）。
> - 已删除 `num_video_frames` / `temporal_interval_mode` / `frame_selection_mode` / `sample_by_window`（JSONL 兼容参数，LeRobot 恒 native chunk）。

### 8.2 注册 experiment：`config.py` 加一行 import

hydra 的 experiment **不是自动发现的**，必须在 `make_config()` 里显式 import（触发模块末尾的 `cs.store()` 注册）：

```python
# cosmos_framework/configs/base/config.py  make_config() 内
import cosmos_framework.configs.base.experiment.sft.vision_sft_edge  # noqa: F401
import cosmos_framework.configs.base.experiment.sft.vision_sft_edge_lerobot3  # noqa: F401  ← 新增
```

### 8.3 toml 指向新 experiment + 多分辨率/fps 开关

`examples/toml/sft_config/vision_sft_edge.toml`：

```toml
[job]
experiment   = "vision_sft_edge_lerobot3"   # ← 从 vision_sft_edge 改来
name         = "vision_sft_edge_test_lerobot3"
wandb_mode   = "offline"

[model]
max_num_tokens_after_packing = 65760       # 从 45056 放大（配合 480 档）

[dataloader_train]
max_sequence_length = 65760                # 从 45056 放大（真正生效的 budget，见 4.10）
use_multi_resolution = true                # 多分辨率开关（256/480 随机）
use_multi_fps = true                       # 多 fps 开关（temporal_interval 随机 2/3/4）
min_frames = 61                            # episode 过滤下界（帧）
max_duration_s = 61.0                      # episode 过滤上界（秒）
```

两个开关和两个过滤阈值都经 `toml_config_helper.py` 的 `PATH_REMAPS["vfm"]` 路由到 `dataloader_train.dataloader.datasets.video.dataset.*` 节点，覆盖 experiment 里的默认值。

### 8.4 启动脚本

`examples/launch_sft_vision_edge_yundao_lerobot.sh`：

| 位置 | 改动 |
|------|------|
| `DATASET_PATH`（第 10 行） | 统一入口，`.jsonl` 文件 or 目录均可 |
| 绕过 `-d` 检查（第 66-71 行） | 保存原值 → 若是文件则临时指向父目录 |
| `EXTRA_DATASET_CHECK`（第 74 行） | 校验原路径存在 + 恢复 `DATASET_PATH` |

**绕过 common 脚本 `-d` 硬检查的技巧**：`_sft_launcher_common.sh` 对 `DATASET_PATH` 做 `-d`（仅目录）检查，`.jsonl` 文件过不了。启动脚本用「临时指向父目录 + EXTRA 恢复」绕过，**不改 common 脚本**：

```bash
_DATASET_ORIGINAL="$DATASET_PATH"
if [[ -f "$DATASET_PATH" && ! -d "$DATASET_PATH" ]]; then
    DATASET_PATH="$(dirname "$DATASET_PATH")"   # jsonl → 父目录，通过 -d 检查
fi

# EXTRA 里：校验原始路径存在 + 恢复 DATASET_PATH（双引号提前展开，路径固化进字符串）
EXTRA_DATASET_CHECK="[[ -e \"$_DATASET_ORIGINAL\" ]] || { echo \"ERROR: dataset not found: $_DATASET_ORIGINAL\" >&2; exit 1; }; export DATASET_PATH=\"$_DATASET_ORIGINAL\";"
```

- 目录模式：`-f` 判断不命中，`DATASET_PATH` 保持目录，`EXTRA` 里 `export` 回去无变化。
- jsonl 模式：临时指向父目录通过 `-d`，`EXTRA` 里恢复成原始 jsonl 路径，config 读 `${oc.env:DATASET_PATH}` 拿到 jsonl → manifest 模式。

### 8.5 启动调用链

```
启动脚本 → TOML_FILE=.../vision_sft_edge.toml
   ↓
_sft_launcher_common.sh → torchrun -m cosmos_framework.scripts.train --sft-toml=...
   ↓
load_experiment_from_toml() → build_hydra_overrides()
   ↓  生成 "experiment=vision_sft_edge_lerobot3"
load_config(config.py, overrides)
   ↓  make_config() import 所有 experiment 模块 → cs.store() 注册
Hydra compose 按 "experiment=..." 查 ConfigStore → 命中 vision_sft_edge_lerobot3
   ↓
最终 Config = experiment 基础配置 + toml 覆盖 → 训练
```

---

## 9. 验证与实测记录

### 9.1 数据构造（drop stats 列 + 加 caption）

**问题**：toy 数据集的 episodes 表有 45 个 `stats/*/*` 嵌套 list 列，磁盘编码损坏，重写整个表必然失败（见 6.3）。

**解法**：drop 掉 `stats/*/*` 列，只保留 17 个有效列 + 加 caption 列，用 pyarrow 25 正常读写。

**结果**：✅ 成功构造 `/mi/data2T/liujin/dataset/toy_lerobot3_with_caption/`（episodes 表 18 列：17 有效 + caption）。

**caption 值**：`"Grasp a battery and put it in the bin."`（toy 数据 `tasks.parquet` 自带任务指令，50 个 episode 全相同，只够冒烟测试）。

### 9.2 字段验证

> 早期在 `_load_single_lerobot_metadata` 里验证；现在改为惰性化后字段在 `process_one_sample` 现算，但取值语义不变。

| 字段 | 验证值 | 结果 |
|------|--------|------|
| uuid | `chunk_0_file_0_episode_0` | ✅ |
| width/height | 640×480（shape `[480,640,3]` 正确取位） | ✅ |
| aspect_ratio | `4,3` | ✅ |
| 帧编号 | `[0,553]`、`[554,1083]`（to 开区间 -1 生效） | ✅ |
| vision_path | 指向 `observation.images.top`（关键字 `top` 命中） | ✅ |
| caption 读取 | 正确读到 caption 列 | ✅ |

### 9.3 训练全链路测试结果

**结果**：✅ **全链路跑通**（metadata → caption → 视频解码 → 训练前向 loss 计算成功）。

日志关键证据：
```
Total number of parameters: 1414924992（模型加载成功）
PackedSequence(sample_lens=[11184, 10464, 11184, 10544], ...)（4 个 sample 打包）
loss = 2.0362（前向成功）
```

> ⚠️ 说明：9.3 的测试是在**惰性化重构之前**做的（旧版 `LeRobotSFTDataset(SFTDataset)` 子类），重构后尚未重新跑全链路，需重新验证。

### 9.4 性能基线测试（demo 数据集 `toy_lerobot3_multi_with_caption`）

**测试数据集**：3 个副本，每个 40 episode，side 4 mp4 + wrist 2 mp4，共 120 episode。

| 指标 | 数值 |
|------|------|
| 平均单 episode | 2.823s |
| read_bytes 整个 mp4 | 0 次（直接用本地路径） |
| 写临时文件 | 0 次 |

**诊断结论**：

- 真正的瓶颈是 **av1 软解 1341 帧的固定成本（~2.45s/个）**，占 ~87%，数据加载代码无法消除。
- ⚠️ 旧版结论「同一 mp4 decoder 新建 1 次（LRU 缓存命中）」**在 pyav 下不成立**：lerobot 的 pyav 路径每次调用都新建 `VideoReader` 并 close，不共享 decoder。`_LRUVideoDecoderCache` 仅 torchcodec 生效（见 5.5）。
- 解码加速手段（按收益排序）：离线转码（av1→h264 可快 2~5 倍，且顺带降分辨率省掉 resize）> 修好 torchcodec（精确按索引取帧）> 调 `num_workers`/`prefetch_factor` > 抽帧（pyav 下只省 resize 不省解码）。

---

## 附录：相关文件索引

| 文件 | 作用 |
|------|------|
| `cosmos_framework/data/generator/local_datasets/sft_dataset.py` | 原 vision SFT 数据加载（JSONL/S3 流程），**未改动**，提供 `_select_caption`/`_DURATION_TEMPLATE`/`_RESOLUTION_TEMPLATE`/`_MAX_CAPTION_TOKENS` 等纯函数供复用（不再继承其 `SFTDataset`） |
| `cosmos_framework/data/generator/local_datasets/sft_dataset_lerobot3.py` | ★ 新增：LeRobot 动态加载（惰性化 `sources`+`episode_index` + manifest 加载 + 独立 `LeRobotSFTDataset` + `get_sft_dataset_from_lerobot`） |
| `cosmos_framework/data/generator/local_datasets/sft_dataset_260907.py` | ⚠️ 备份文件：是 JSONL 父类 `sft_dataset.py` 的副本（非 LeRobot 版），勿与 `sft_dataset_lerobot3.py` 混淆 |
| `cosmos_framework/data/generator/local_datasets/helper.py` | `ffmpeg_decode_video`、`get_aspect_ratio`、`get_video_metadata`、`download_from_s3`（未改动） |
| `cosmos_framework/data/generator/action/datasets/cosmos3_action_lerobot.py` | action 侧 LeRobot 加载 + `_LRUVideoDecoderCache`（可借鉴） |
| `cosmos_framework/configs/base/experiment/sft/vision_sft_edge.py` | 原 vision SFT 实验配置（JSONL 流程），**未改动** |
| `cosmos_framework/configs/base/experiment/sft/vision_sft_edge_lerobot3.py` | ★ 新增：LeRobot experiment（`get_sft_dataset_from_lerobot` + 统一 `dataset_path` 接入；`caption_key="task"`、`video_backend="pyav"`） |
| `cosmos_framework/configs/base/config.py` | 加 1 行 import 注册新 experiment |
| `cosmos_framework/configs/toml_config/sft_config.py` | `DataloaderTrainConfig` 新增 `use_multi_resolution`/`use_multi_fps` 两个 bool 字段 |
| `cosmos_framework/configs/toml_config/toml_config_helper.py` | `PATH_REMAPS` 新增两条 remap，把 toml 开关路由到 dataset 节点 |
| `cosmos_framework/configs/base/experiment/sft/models/edge_model_config.py` | `vae_path` 改绝对路径（环境相关，非本特性，慎提交） |
| `examples/toml/sft_config/vision_sft_edge.toml` | `experiment` 字段指向 `vision_sft_edge_lerobot3` + 多分辨率/fps 开关 + token 预算放大到 65760 |
| `examples/launch_sft_vision_edge_yundao_lerobot.sh` | 启动脚本（`DATASET_PATH` 统一入口 + 绕过 `-d` 检查技巧） |
| `examples/_sft_launcher_common.sh` | 公共启动脚本，**未改动**（已回滚） |
