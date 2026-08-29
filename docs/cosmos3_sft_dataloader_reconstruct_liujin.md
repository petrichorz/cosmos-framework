# Cosmos3 SFT 数据加载重构方案（LeRobot 动态加载）

> 本文档记录「让 vision SFT 直接动态加载 LeRobot 3.x 数据集（episodes 表带 caption 字段）」的最终实现方案。
>
> 背景：私有数据为 LeRobot 3.x 格式，episodes 表的 parquet 里**侵入式新增 caption 字段**（已验证：官方 lerobot 库加载 episodes 表加列不报错；data 表加列会报错，禁止）。
>
> **代码组织原则**：原 `sft_dataset.py` / `vision_sft_edge.py` **一行不改**，所有新增逻辑放到独立的新文件里。

---

## 目录

- [1. 目标与约束](#1-目标与约束)
- [2. 最终文件组织](#2-最终文件组织)
- [3. 现有 SFTDataset 数据流回顾](#3-现有-sftdataset-数据流回顾)
- [4. 字段映射方案（LeRobot → metadata）](#4-字段映射方案lerobot--metadata)
- [5. 视频解码效率改造（共享 decoder + 帧编号 seek）](#5-视频解码效率改造共享-decoder--帧编号-seek)
- [6. 已确认的风险点与注意事项](#6-已确认的风险点与注意事项)
- [7. 关键结论速查](#7-关键结论速查)
- [8. 训练接入改动](#8-训练接入改动)
- [9. 验证与实测记录](#9-验证与实测记录)
- [附录：相关文件索引](#附录相关文件索引)

---

## 1. 目标与约束

### 目标

让现有 vision SFT 训练流程**动态加载 LeRobot 数据集**（不预先转 JSONL），并复用现有 `SFTDataset` 的视频解码、resize、caption tokenize、sequence_plan、CFG dropout 等逻辑。

### 已拍板的决策

| #   | 决策         | 说明                                                                                            |
| --- | ------------ | ----------------------------------------------------------------------------------------------- |
| 1   | 路线         | 路线 B：动态加载，不转 JSONL                                                                    |
| 2   | 相机         | 关键字匹配选视角（key 名含 `top`/`head` 等关键字即选中；匹配不到回退第一个 video）              |
| 3   | caption 粒度 | episode 级（一个任务一个描述）                                                                  |
| 4   | 字段语义     | `t2w_windows` 存**帧编号**（非 timestamp），对齐原版                                            |
| 5   | 代码组织     | **原文件一行不改**；新建 `sft_dataset_lerobot3.py` + `vision_sft_edge_lerobot3.py` 承载新增逻辑 |
| 6   | 效率         | 共享 decoder + LRU 缓存，同一 mp4 只打开一次                                                    |

---

## 2. 最终文件组织

新增逻辑全部放在两个新文件里，原文件保持不动：

```
cosmos_framework/data/generator/local_datasets/
  ├── sft_dataset.py            # 原文件（JSONL/S3 流程），一行未改
  └── sft_dataset_lerobot3.py   # ★ 新增：LeRobot 动态加载（612 行）

cosmos_framework/configs/base/experiment/sft/
  ├── vision_sft_edge.py        # 原文件（JSONL 流程），一行未改
  └── vision_sft_edge_lerobot3.py  # ★ 新增：LeRobot experiment（257 行）

cosmos_framework/configs/base/config.py     # 加 1 行 import（注册新 experiment）
examples/toml/sft_config/vision_sft_edge.toml  # experiment 字段指向新名字
examples/launch_sft_vision_edge_yundao_lerobot.sh  # 启动脚本（DATASET_PATH 指向 LeRobot）
```

### 2.1 `sft_dataset_lerobot3.py` 内容

| 符号                              | 行号 | 作用                                                    |
| --------------------------------- | ---- | ------------------------------------------------------- |
| `_select_lerobot_video_key`       | 47   | 选定 video 字段（显式指定 → 关键字匹配 → 第一个 video） |
| `_get_lerobot_video_width_height` | 77   | 从 video 字段 shape 抓 (width, height)                  |
| `_discover_lerobot_roots`         | 93   | 单数据集根 or 父目录多数据集发现                        |
| `_load_single_lerobot_metadata`   | 118  | 读单个数据集 → metadata list                            |
| `_load_lerobot_metadata`          | 221  | 统一入口：发现多个数据集 → 逐个加载 → 合并              |
| `_LeRobotVideoDecoderCache`       | 260  | torchcodec decoder LRU 缓存（`seek_mode="exact"`）      |
| `LeRobotSFTDataset(SFTDataset)`   | 295  | 子类，override `process_one_sample`                     |
| `get_sft_dataset_from_lerobot`    | 524  | LeRobot 版入口，构造 `LeRobotSFTDataset`                |

### 2.2 复用父模块符号（不重复实现）

新文件从 `sft_dataset.py` import 这些符号，避免重复：

```python
from cosmos_framework.data.generator.local_datasets.sft_dataset import (
    SFTDataset,                  # 继承
    _DURATION_TEMPLATE,          # caption 时长/FPS 模板
    _MAX_CAPTION_TOKENS,         # caption token 上限
    _RESOLUTION_TEMPLATE,        # caption 分辨率模板
    _flatten_metadata_by_window, # sample_by_window 展开
    _select_caption,             # t2w_window 里选 caption key
)
```

### 2.3 `LeRobotSFTDataset.process_one_sample` 与父类的唯一差异

父类 `SFTDataset.process_one_sample` 是一个单体方法（无钩子），所以子类 override 整个方法，但**只替换中段视频加载**，前段（选 window + 算分辨率）和后段（crop + caption + tokenize + ret）逐字对齐：

| 段                                    | 父类                                            | `LeRobotSFTDataset`                                               |
| ------------------------------------- | ----------------------------------------------- | ----------------------------------------------------------------- |
| 前段：选 window + 算分辨率            | 原样                                            | 逐字复制                                                          |
| **中段：视频加载**                    | download 到临时文件 + 全量 ffmpeg decode + 过滤 | **本地 mp4 直接 `get_video_metadata` + torchcodec 按帧区间 seek** |
| 后段：stack/crop/caption/tokenize/ret | 原样                                            | 逐字复制                                                          |

子类新增 `_decode_video_frames`（torchcodec 区间解码 + LRU 缓存），供 override 后的 `process_one_sample` 调用。

---

## 3. 现有 SFTDataset 数据流回顾

`SFTDataset`（`sft_dataset.py`，未改动）是 `IterableDataset`，依赖 `metadata: list[dict]` 输入，每个 dict 字段：

```python
{
    "uuid": str,              # 样本唯一标识
    "vision_path": str,       # 视频本地绝对路径 / s3://
    "width": int,             # 原始视频宽
    "height": int,            # 原始视频高
    "nb_frames": int|None,    # 可选
    "framerate": float|None,  # 可选
    "aspect_ratio": str,      # 派生：get_aspect_ratio(width,height)
    "t2w_windows": [          # 过滤后的 window 列表
        {"start_frame": int, "end_frame": int, "temporal_interval": int, "caption": str, ...}
    ],
}
```

这些字段原由 `_load_sft_metadata_from_s3`（JSONL 版）解析。**LeRobot 版在 `sft_dataset_lerobot3.py` 里用 `_load_lerobot_metadata` 产出同样结构的 dict 列表**，交给 `LeRobotSFTDataset`（继承 `SFTDataset`）消费。

---

## 4. 字段映射方案（LeRobot → metadata）

### 4.1 uuid

**格式**：`{dataset_name}_chunk_{chunk_index}_file_{file_index}_episode_{episode_index}`

```python
uuid = f"{root.name}_chunk_{data_chunk}_file_{data_file}_episode_{episode_index}"
```

- 来源：数据集目录名（`root.name`）+ episodes 表的三列
- 目的：跨**数据集**、跨 chunk、跨 file、跨 episode 唯一

> episode_index 单数据集内唯一，但**多数据集合并时**不同数据集可能有相同编号，所以 uuid 前加数据集目录名保证唯一。

### 4.2 vision_path

用 `info.json["video_path"]` 模板 + episodes 表的 chunk/file 拼出本地 mp4 路径：

```python
# info.json 示例
# "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"

vision_path = str(
    root / info["video_path"].format(
        video_key=video_key,          # 选定的相机字段名，如 observation.images.top
        chunk_index=video_chunk,
        file_index=video_file,
        episode_chunk=video_chunk,
        episode_file=video_file,
    )
)
```

### 4.3 width / height

从选定 video 字段的 `shape` 抓，**注意 shape 是 `[H, W, C]`，要转成 `(width, height)`**：

```python
feat = info["features"][video_key]
shape = feat["shape"]          # [H, W, C]，如 [480, 640, 3]
names = feat.get("names")      # 通常 ["height","width","channels"]
if names and "width" in names and "height" in names:
    w = shape[names.index("width")]
    h = shape[names.index("height")]
else:
    h, w = shape[0], shape[1]   # 兜底：默认 [H, W, C]
```

- toy 数据：`shape=[480, 640, 3]` → `width=640, height=480`
- 同一数据集各相机分辨率一致，所以取选定相机的 shape 即可

### 4.4 aspect_ratio

复用 `get_aspect_ratio(width, height)`（`helper.py`），按宽高比分桶成 `"1,1"/"4,3"/"3,4"/"16,9"/"9,16"`。

### 4.5 t2w_windows（存帧编号，重点）

原版 `start_frame`/`end_frame` 存的是**帧编号**（0-based，闭区间），LeRobot 版也对齐此语义。

换算公式（toy 数据实测精确成立）：

```python
fps = info["fps"]   # 30

start_frame = round(from_timestamp * fps)     # 含
end_frame   = round(to_timestamp * fps) - 1   # to 是开区间，-1 变闭区间
```

验证（toy 数据）：

| episode | from      | to        | start | end  | length | 吻合 |
| ------- | --------- | --------- | ----- | ---- | ------ | ---- |
| 0       | 0.0       | 18.466667 | 0     | 553  | 554    | ✅   |
| 1       | 18.466667 | 36.133333 | 554   | 1083 | 530    | ✅   |

> `to_timestamp` 是**开区间**，该时刻的帧属于下一个 episode，所以 `end_frame = round(to×fps) - 1`。

构造出的 window：

```python
window = {
    "start_frame": start_frame,
    "end_frame": end_frame,
    "temporal_interval": 1,
}
if caption:                     # caption 为空时不写 key，下游优雅跳过
    window["caption"] = caption
```

### 4.6 caption 处理逻辑

```python
caption = row.get(caption_key)   # 只读 caption_key 指定的列，取不到为 None（不做 tasks 兜底）

if caption:                      # 为空时不写 caption key
    window["caption"] = caption
```

> 关键边界行为：缺 caption 列（或值为空）时，`window` 里**没有** `caption` key，下游 `_select_caption` 找不到已知 key → `return None` → `process_one_sample` 跳过该样本。**不写 `caption: None`**，否则下游 `raw.strip()` 会 `AttributeError` 崩溃。

---

## 5. 视频解码效率改造（共享 decoder + 帧编号 seek）

### 5.1 问题

父类 `process_one_sample` 的视频解码是「全量 decode + 按帧编号过滤」：

```python
for idx, frame in enumerate(ffmpeg_decode_video(input_video_path, ...)):
    if idx < start_frame: continue
    elif idx <= end_frame: ...
```

toy 数据一个 mp4 含 50 个 episode，每个 episode 都全量解码同一文件 → 50 次全量解码，极低效。

### 5.2 内存账（为什么不能「全量 decode 缓存」）

toy 一个 mp4：24263 帧 × 480 × 640 × 3 = **约 22.4 GB**（解压后 RGB）。全量缓存会 OOM。所以正确做法是：**每个 mp4 只打开一次、建一个 decoder，之后按帧编号 seek，只解码目标区间**。

### 5.3 目标架构

```
按 vision_path（mp4 文件）缓存 decoder（LRU）
   │
   ▼
episode 需要帧时：
   decoder.get_frames_in_range(start_frame, end_frame+1)
   只 decode [start_frame, end_frame] 这一段（~500 帧）
```

### 5.4 torchcodec seek API

| 方法                  | 签名                                  | 作用                                  |
| --------------------- | ------------------------------------- | ------------------------------------- |
| `get_frame_at`        | `(index: int) -> Frame`               | 取单个帧                              |
| `get_frames_at`       | `(indices: list[int]) -> FrameBatch`  | 取指定索引列表                        |
| `get_frames_in_range` | `(start, stop, step=1) -> FrameBatch` | 取 `[start, stop)` 开区间（**推荐**） |

**本方案用 `get_frames_in_range`**：

```python
frames = decoder.get_frames_in_range(start=start_frame, stop=end_frame + 1)
# 开区间 [start, stop)，所以 stop = end_frame + 1
# 返回 FrameBatch（批量，比逐帧 get_frame_at 快）
```

### 5.5 seek_mode 精度

| seek_mode       | 精度                           | 代价                   | 适用       |
| --------------- | ------------------------------ | ---------------------- | ---------- |
| `exact`（默认） | 请求第 i 帧**一定**返回第 i 帧 | 初始扫描整个文件建索引 | 要求帧精确 |
| `approximate`   | 快，不扫描                     | 可能不准               | 允许近似   |

> **本方案用 `seek_mode="exact"`**。vision SFT 要精确切 episode 帧区间（切错一帧就把下一集的帧混进来），`exact` 的扫描代价靠「共享 decoder」摊薄——同一 mp4 的多个 episode 共享一个 decoder，只扫描一次。action 侧用 `approximate` 是因为它对精确帧不敏感。

### 5.6 `_LeRobotVideoDecoderCache`（LRU 缓存）

`VideoDecoder` 的 `exact` 模式要扫描整个文件，`__init__` 代价高，所以按路径缓存 decoder 复用。核心结构（`OrderedDict` 实现 LRU）：

```python
class _LeRobotVideoDecoderCache:
    def __init__(self, max_size: int = 64):
        from collections import OrderedDict
        self._max_size = max_size
        self._cache: "OrderedDict[str, tuple]" = OrderedDict()

    def get_decoder(self, video_path: str):
        from torchcodec.decoders import VideoDecoder
        import fsspec
        if video_path in self._cache:
            self._cache.move_to_end(video_path)   # 命中 → 标记最近使用
            return self._cache[video_path][0]
        file_handle = fsspec.open(video_path).__enter__()
        decoder = VideoDecoder(file_handle, seek_mode="exact")
        self._cache[video_path] = (decoder, file_handle)
        while len(self._cache) > self._max_size:
            _, (_, old_fh) = self._cache.popitem(last=False)  # 淘汰最久未用
            old_fh.close()
        return decoder
```

三个关键点：

1. **LRU 用 `OrderedDict`**：`move_to_end`（命中标记最近）+ `popitem(last=False)`（淘汰最久未用）。
2. **缓存 `(decoder, file_handle)` 二元组**：decoder 依赖底层文件句柄，淘汰时 `close()` 释放。
3. **`max_size=64`**：限制同时打开的 decoder 数量，避免耗尽文件描述符/内存。

> 相比 action 侧 `_LRUVideoDecoderCache`，这里把 `seek_mode` 从 `approximate` 改成 `exact`；且**本地定义**（不从 `cosmos3_action_lerobot` import，避免连带触发 lerobot 库 import），torchcodec/fsspec 惰性 import。

### 5.7 子类解码实现

```python
def _decode_video_frames(self, video_path, start_frame, end_frame, temporal_interval, resize_h, resize_w):
    import torch.nn.functional as F
    decoder = self._decoder_cache.get_decoder(video_path)
    frame_batch = decoder.get_frames_in_range(start=start_frame, stop=end_frame + 1)
    data = frame_batch.data  # [N, C, H, W] uint8
    if temporal_interval > 1:
        data = data[0::temporal_interval]           # 对齐原版「相对 start 取余 interval==0」
    data = data.float()
    data = F.interpolate(data, size=(resize_h, resize_w), mode="bicubic", align_corners=False)  # 对齐 ffmpeg -vf scale+bicubic
    data = data.round().clamp(0, 255).to(torch.uint8)
    data_nhwc = data.permute(0, 2, 3, 1).cpu().numpy()  # [N,H,W,C]
    return [data_nhwc[i] for i in range(data_nhwc.shape[0])]
```

两个落地补全点：

1. **resize**：用 `F.interpolate(mode="bicubic")` 对齐原版 ffmpeg 的 `-vf scale + bicubic`。
2. **抽帧**：`data[0::temporal_interval]` 等价于原版「相对 start 取余 interval == 0」（`get_frames_in_range` 返回连续帧，第 0 个元素即全局 start_frame）。

---

## 6. 已确认的风险点与注意事项

> 本节集中记录**通过源码/数据验证**得出的易踩坑点，供后续版本迭代参考。

### 6.1 数据格式层（LeRobot）

| #   | 风险点                            | 结论                                                       | 依据                               |
| --- | --------------------------------- | ---------------------------------------------------------- | ---------------------------------- |
| 1   | v3 命名单数还是复数               | **复数 `observation.images.*`**（不是单数 `image`）        | toy 数据实测                       |
| 2   | 一个 mp4 含几个 episode           | **多个 episode 共享一个 mp4**，靠 `from/to_timestamp` 切分 | toy：50 episode 共享 file-000.mp4  |
| 3   | `to_timestamp` 开闭区间           | **开区间**：该时刻帧属于下一 episode                       | toy：`to×fps` 精确等于下一集起始帧 |
| 4   | `length` / 行跨度 / 时间跨度      | 精确相等：`length = round((to-from)×fps)`                  | toy 数据验证                       |
| 5   | `frame_index` 全局还是 episode 内 | **episode 内从 0 重新开始**                                | toy：episode 1 首帧 frame_index=0  |

### 6.2 字段映射层

| #   | 风险点                  | 结论                                                                         |
| --- | ----------------------- | ---------------------------------------------------------------------------- |
| 1   | `shape` 是 H×W 还是 W×H | **`[H, W, C]`**，`width=shape[1]`, `height=shape[0]`                         |
| 2   | 视频字段 key 能否硬编码 | **不能**（`observation.images.top` 是数据集特定的），用关键字匹配/兜底动态选 |
| 3   | `end_frame` 语义        | 存**帧编号**（文件内），非 timestamp；`end = round(to×fps) - 1`              |
| 4   | `aspect_ratio`          | `get_aspect_ratio(width,height)` 派生，JSONL/LeRobot 里没有现成字段          |

### 6.3 加 caption 的侵入位置

| #   | 风险点                                   | 结论                                                                                                                            |
| --- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| 1   | 加到 episodes 表                         | ✅ **安全**，官方 lerobot 加载不报错（已实测）                                                                                  |
| 2   | 加到 data 表                             | ❌ **禁止**，破坏 `info.json` schema 校验，报 `CastError`（已实测）                                                             |
| 3   | `info.json` 的 `features` 约束范围       | **只严格约束 data 表**，不约束 episodes 表                                                                                      |
| 4   | **pandas 3.0.5 破坏嵌套 list 列**        | ❌ 不能用 pandas 3.0.5 读写 episodes parquet（`stats/*/*` 列损坏）。加 caption 用 **pyarrow**（append_column）或 **pandas 2.x** |
| 5   | `tasks` 列实际类型                       | **`numpy.ndarray`**（不是 list/tuple），判断用 `hasattr(x, "__len__")`                                                          |
| 6   | **toy 数据集 stats 列 parquet 编码损坏** | 磁盘 def/rep level 损坏，重写整表必然失败。**解法：drop 掉 `stats/*/*` 列**，只保留有效列 + caption                             |

### 6.4 视频解码层（torchcodec）

| #   | 风险点                         | 结论                                                       |
| --- | ------------------------------ | ---------------------------------------------------------- |
| 1   | `get_frames_in_range` 区间语义 | **开区间 `[start, stop)`**，`stop = end_frame + 1`         |
| 2   | `FrameBatch.data` 形状         | **4D `[N, C, H, W]`**，不是 3D                             |
| 3   | `FrameBatch.data` dtype        | **uint8**，进 VAE 前转 float + normalize                   |
| 4   | seek_mode 选哪个               | **`exact`**（精确帧定位），不是 action 侧的 `approximate`  |
| 5   | `exact` 扫描代价               | 每个 mp4 建 decoder 扫描一次，靠「共享 decoder + LRU」摊薄 |
| 6   | 不能全量 decode 缓存           | 一个 mp4 解压后 ~22GB（toy），会 OOM，必须按帧区间 seek    |

### 6.5 尚未验证、版本迭代时需重点关注

| #   | 风险点                             | 说明                                                                                                                |
| --- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| 1   | 多文件场景的 `from_timestamp` 语义 | toy 是单 mp4，无法验证；真实多文件数据需确认是全局时间还是文件内相对时间                                            |
| 2   | 抽帧语义                           | `get_frames_in_range(step=interval)` 的全局步进 vs 原版「相对 start 取余」的差异（当前用 `data[0::interval]` 规避） |

---

## 7. 关键结论速查

| 问题               | 结论                                                                                                |
| ------------------ | --------------------------------------------------------------------------------------------------- |
| uuid 格式          | `{dataset_name}_chunk_{chunk_idx}_file_{file_idx}_episode_{ep_idx}`（含数据集目录名，跨数据集唯一） |
| 多数据集加载       | `lerobot_root` 支持单数据集根 or 父目录；父目录自动 `rglob("meta/info.json")` 递归发现              |
| 选哪路视频         | `_select_lerobot_video_key`：显式 key > 关键字匹配（`video_feature_keywords`）> 第一个 video 字段   |
| width/height 来源  | 选定 video 字段的 shape 前两位，`width=shape[1]`, `height=shape[0]`                                 |
| t2w_windows 存什么 | **帧编号**（非 timestamp）：`start=round(from×fps)`, `end=round(to×fps)-1`                          |
| 视频只读一次怎么做 | 共享 decoder + LRU 缓存 + `get_frames_in_range` 区间 seek（非全量缓存）                             |
| 加 caption 到哪    | episodes 表加列（安全）；**禁止加到 data 表**                                                       |

---

## 8. 训练接入改动

### 8.1 新增 experiment：`vision_sft_edge_lerobot3.py`

完整复制原 `vision_sft_edge.py`（297 行），只改 3 处：

| 位置                 | 改动                                                                                                                           |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| import（32 行）      | `from ...sft_dataset_lerobot3 import get_sft_dataset_from_lerobot`                                                             |
| `dataset=`（220 行） | `L(get_sft_dataset_from_lerobot)(...)`，参数换 `lerobot_root` + `video_feature_key` + `video_feature_keywords` + `caption_key` |
| `job.name`（70 行）  | `"vision_sft_edge_lerobot3"`                                                                                                   |

关键参数（第 220-234 行）：

```python
dataset=L(get_sft_dataset_from_lerobot)(
    ...
    lerobot_root="${oc.env:DATASET_PATH}",      # LeRobot 数据集根目录（含 meta/info.json）
    video_feature_key=None,                      # 显式指定 feature 名（精确匹配）；None 则不显式指定
    video_feature_keywords=["top", "head"],      # 关键字 list：key 名含任一关键字即选中；匹配不到回退第一个 video
    caption_key="caption",                       # episodes 表里的 caption 列名
    num_video_frames=-1,                         # native chunk mode，直接按 t2w_windows 帧区间
    ...
)
```

### 8.2 注册 experiment：`config.py` 加一行 import

hydra 的 experiment **不是自动发现的**，必须在 `make_config()` 里显式 import（触发模块末尾的 `cs.store()` 注册）：

```python
# cosmos_framework/configs/base/config.py  make_config() 内
import cosmos_framework.configs.base.experiment.sft.vision_sft_edge  # noqa: F401
import cosmos_framework.configs.base.experiment.sft.vision_sft_edge_lerobot3  # noqa: F401  ← 新增
```

### 8.3 toml 指向新 experiment

`examples/toml/sft_config/vision_sft_edge.toml`：

```toml
[job]
experiment   = "vision_sft_edge_lerobot3"   # ← 从 vision_sft_edge 改来
```

### 8.4 启动脚本

`examples/launch_sft_vision_edge_yundao_lerobot.sh`：

| 位置                              | 改动                                                   |
| --------------------------------- | ------------------------------------------------------ |
| `DATASET_PATH`（第 10 行）        | LeRobot 数据集根目录 `toy_lerobot3_multi_with_caption` |
| `EXTRA_DATASET_CHECK`（第 66 行） | 校验 `$DATASET_PATH` 下任意深度存在 `meta/info.json`   |

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

### 9.2 `_load_lerobot_metadata` 字段验证

| 字段         | 验证值                                             | 结果 |
| ------------ | -------------------------------------------------- | ---- |
| uuid         | `chunk_0_file_0_episode_0`                         | ✅   |
| width/height | 640×480（shape `[480,640,3]` 正确取位）            | ✅   |
| aspect_ratio | `4,3`                                              | ✅   |
| 帧编号       | `[0,553]`、`[554,1083]`（to 开区间 -1 生效）       | ✅   |
| vision_path  | 指向 `observation.images.top`（关键字 `top` 命中） | ✅   |
| caption 读取 | 正确读到 caption 列                                | ✅   |

### 9.3 训练全链路测试结果

**结果**：✅ **全链路跑通**（metadata → caption → 视频解码 → 训练前向 loss 计算成功）。

日志关键证据：

```
Total number of parameters: 1414924992（模型加载成功）
PackedSequence(sample_lens=[11184, 10464, 11184, 10544], ...)（4 个 sample 打包）
loss = 2.0362（前向成功）
```

### 9.4 性能基线测试（demo 数据集 `toy_lerobot3_multi_with_caption`）

**测试数据集**：3 个副本，每个 40 episode，side 4 mp4 + wrist 2 mp4，共 120 episode。

| 指标                                  | 数值                   |
| ------------------------------------- | ---------------------- |
| 平均单 episode（共享 decoder 命中后） | 2.823s                 |
| read_bytes 整个 mp4                   | 0 次（直接用本地路径） |
| 写临时文件                            | 0 次                   |
| 同一 mp4 decoder 新建                 | 1 次（LRU 缓存命中）   |

**诊断结论**：

- 共享 decoder + LRU 缓存生效，消除了「每个 episode 全量解码」和「反复 read_bytes + 临时文件」的浪费。
- 真正的瓶颈是 **av1 软解 1341 帧的固定成本（~2.45s/个）**，数据加载代码无法消除。
- 更大的优化空间在**配置层**：`num_video_frames=-1`（全帧）→ `93`（抽帧），seek 时间可降一个数量级。

---

## 附录：相关文件索引

| 文件                                                                        | 作用                                                                                                                            |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `cosmos_framework/data/generator/local_datasets/sft_dataset.py`             | 原 vision SFT 数据加载（JSONL/S3 流程），**未改动**，提供 `SFTDataset`/`_select_caption`/`_flatten_metadata_by_window` 等供复用 |
| `cosmos_framework/data/generator/local_datasets/sft_dataset_lerobot3.py`    | ★ 新增：LeRobot 动态加载（metadata 构造 + `LeRobotSFTDataset` + `get_sft_dataset_from_lerobot`）                                |
| `cosmos_framework/data/generator/local_datasets/helper.py`                  | `ffmpeg_decode_video`、`get_aspect_ratio`、`get_video_metadata`、`download_from_s3`（未改动）                                   |
| `cosmos_framework/data/generator/action/datasets/cosmos3_action_lerobot.py` | action 侧 LeRobot 加载 + `_LRUVideoDecoderCache`（可借鉴）                                                                      |
| `cosmos_framework/configs/base/experiment/sft/vision_sft_edge.py`           | 原 vision SFT 实验配置（JSONL 流程），**未改动**                                                                                |
| `cosmos_framework/configs/base/experiment/sft/vision_sft_edge_lerobot3.py`  | ★ 新增：LeRobot experiment（`get_sft_dataset_from_lerobot` 接入）                                                               |
| `cosmos_framework/configs/base/config.py`                                   | 加 1 行 import 注册新 experiment                                                                                                |
| `examples/toml/sft_config/vision_sft_edge.toml`                             | `experiment` 字段指向 `vision_sft_edge_lerobot3`                                                                                |
| `examples/launch_sft_vision_edge_yundao_lerobot.sh`                         | 启动脚本（`DATASET_PATH` 指向 LeRobot）                                                                                         |
