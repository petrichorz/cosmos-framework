# Cosmos3 SFT 数据加载重构方案（LeRobot 动态加载）

> 本文档记录「让 vision SFT 直接动态加载 LeRobot 3.x 数据集（episodes 表带 caption 字段）」的整体设计思路与流程，供后续实现参考。
>
> 背景：私有数据为 LeRobot 3.x 格式，后续会在 episodes 表的 parquet 里**侵入式新增 caption 字段**（已验证：官方 lerobot 库加载 episodes 表加列不会报错；data 表加列会报错，禁止）。

---

## 目录

- [1. 目标与约束](#1-目标与约束)
- [2. 现有 SFTDataset 数据流回顾](#2-现有-sftdataset-数据流回顾)
- [3. 字段映射方案（LeRobot → metadata）](#3-字段映射方案lerobot--metadata)
- [4. 视频解码效率改造（共享 decoder + 帧编号 seek）](#4-视频解码效率改造共享-decoder--帧编号-seek)
- [5. 待落地时确认的实现细节](#5-待落地时确认的实现细节)
- [6. 已确认的风险点与注意事项（版本迭代参考）](#6-已确认的风险点与注意事项版本迭代参考)
- [7. 实施流程（建议顺序）](#7-实施流程建议顺序)
- [8. 关键结论速查](#8-关键结论速查)
- [附录：相关文件索引](#附录相关文件索引)

---

## 1. 目标与约束

### 目标

让现有 vision SFT 训练流程**动态加载 LeRobot 数据集**（不预先转 JSONL），并复用现有 `SFTDataset` 的视频解码、resize、caption tokenize、sequence_plan、CFG dropout 等逻辑。

### 已拍板的决策（本次会话确认）

| # | 决策 | 说明 |
|---|------|------|
| 1 | 路线 | 路线 B：动态加载，不转 JSONL |
| 2 | 相机 | 只用 top 路（toy 数据里 `usable=true`） |
| 3 | caption 粒度 | episode 级（一个任务一个描述） |
| 4 | 字段语义 | `t2w_windows` 存**帧编号**（非 timestamp），对齐原版 |
| 5 | 视频切片 | 侵入改 `SFTDataset`，支持共享 mp4 按帧编号 seek |
| 6 | 效率 | 共享 decoder + LRU 缓存，同一 mp4 只打开一次 |

---

## 2. 现有 SFTDataset 数据流回顾

`SFTDataset`（`cosmos_framework/data/generator/local_datasets/sft_dataset.py`）是 `IterableDataset`，依赖一个 `metadata: list[dict]` 输入，每个 dict 字段：

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

这些字段目前由 `_load_sft_metadata_from_s3`（`sft_dataset.py:497-574`）从 JSONL 解析。**改造点是：新增一个 LeRobot 版本的 metadata 构造器，产出同样结构的 dict 列表。**

`process_one_sample`（`sft_dataset.py:179-361`）消费这些字段，其中视频解码在 `:202-263`，当前是「全量 decode + 按帧编号过滤」（低效，需改）。

---

## 3. 字段映射方案（LeRobot → metadata）

### 3.1 uuid

**格式**：`chunk_{chunk_index}_file_{file_index}_episode_{episode_index}`

- 来源：episodes 表的三列
- 目的：跨 chunk、跨 file、跨 episode 唯一，且能从名字看出各 index 含义

```python
uuid = f"chunk_{chunk_idx}_file_{file_idx}_episode_{ep_idx}"
```

> 说明：episode_index 在单数据集内虽唯一，但拼接多分片会重复；chunk_index + file_index + episode_index 三者才能唯一定位到「哪个文件里的哪个 episode」。

### 3.2 vision_path

用 `info.json["video_path"]` 模板 + episodes 表的 chunk/file 拼出本地 mp4 路径。

```python
# info.json 示例
# "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"

rel = info["video_path"].format(
    video_key=video_key,          # 选定的相机字段名，如 observation.images.top
    chunk_index=chunk_idx,
    file_index=file_idx,
)
vision_path = str(root / rel)
```

> 现有 `base_dataset.py:160-180` 的 `_video_path` 就是这个逻辑，可参考复用。

### 3.3 width / height

**动态抓取**：从 `info.json["features"]` 里取**第一个 `dtype == "video"`** 的字段，读其 `shape` 的前两位。**注意 shape 是 `[H, W, C]`，要转成 `(width, height)`。**

```python
def get_video_width_height(info: dict) -> tuple[int, int]:
    for key, feat in info["features"].items():
        if feat.get("dtype") != "video":
            continue
        shape = feat["shape"]          # [H, W, C]，如 [480, 640, 3]
        names = feat.get("names")       # 通常 ["height","width","channels"]
        if names and "width" in names and "height" in names:
            w = shape[names.index("width")]
            h = shape[names.index("height")]
        else:
            h, w = shape[0], shape[1]   # 兜底：默认 [H, W, C]
        return w, h
    raise ValueError("info.json features 里没有 dtype=video 的字段")
```

- toy 数据：`shape=[480, 640, 3]` → `width=640, height=480`
- 同一数据集各相机分辨率一致，所以「第一个 video 字段」的 shape 即可，与选哪路相机无关

### 3.4 aspect_ratio

复用 `get_aspect_ratio(width, height)`（`helper.py:173`），按宽高比分桶成 `"1,1"/"4,3"/"3,4"/"16,9"/"9,16"`。

### 3.5 t2w_windows（存帧编号，重点）

**结论**：原版 `start_frame`/`end_frame` 存的是**帧编号**（0-based，闭区间），不是 timestamp。新版本也存帧编号，字段语义对齐，复用现有按帧编号过滤的解码逻辑。

换算公式（toy 数据实测验证精确成立）：

```python
fps = info["fps"]   # 30

start_frame = round(from_timestamp * fps)     # 含
end_frame   = round(to_timestamp * fps) - 1   # to 是开区间，-1 变闭区间
```

验证（toy 数据）：

| episode | from | to | start | end | length | 吻合 |
|---------|------|-----|-------|-----|--------|------|
| 0 | 0.0 | 18.466667 | 0 | 553 | 554 | ✅ |
| 1 | 18.466667 | 36.133333 | 554 | 1083 | 530 | ✅ |

**关键**：`to_timestamp` 是开区间，该时刻的帧属于下一个 episode，所以 `end_frame = round(to×fps) - 1`。

构造出的 window：

```python
window = {
    "start_frame": start_frame,
    "end_frame": end_frame,
    "temporal_interval": 1,
    "caption": caption,        # 来自 episodes 表的 caption 列
}
```

> 注意：这里的 start_frame/end_frame 是**该 mp4 文件内**的帧编号（0-based）。当 episodes 表用 `from/to_timestamp` 描述时，需换算；若未来 episodes 表直接有帧编号列，则直接读。

---

## 4. 视频解码效率改造（共享 decoder + 帧编号 seek）

### 4.1 问题

现有 `process_one_sample` 里（`sft_dataset.py:202-263`）：

```python
for idx, frame in enumerate(ffmpeg_decode_video(input_video_path, ...)):
    if idx < start_frame: continue
    elif idx <= end_frame: ...
```

这是**全量解码整个 mp4**，再用帧编号过滤。toy 数据一个 mp4 含 50 个 episode，每个 episode 都要全量解码一次 → 50 次全量解码同一文件，极低效。

### 4.2 内存账（为什么不能「全量 decode 缓存」）

toy 一个 mp4：24263 帧 × 480 × 640 × 3 = **约 22.4 GB**（解压后 RGB）。全量缓存会 OOM。所以「只读一次」的正确含义是：**每个 mp4 只打开一次、建一个 decoder，之后按帧编号 seek，只解码目标区间**，而非缓存所有帧。

### 4.3 目标架构

```
按 vision_path（mp4 文件）分组 episode
   │
   ▼
每个 mp4 维护一个 decoder（LRU 缓存）
   │
   ▼
episode 需要帧时：
   decoder.seek(start_frame)
   只 decode [start_frame, end_frame] 这一段（~500 帧）
   │
   ▼
返回该段帧（不再全量解码 24263 帧）
```

### 4.4 可借鉴的现有实现

action 侧已有成熟的 decoder 缓存模式（`cosmos3_action_lerobot.py:120-173`）：

```python
class _LRUVideoDecoderCache:
    def get_decoder(self, video_path: str) -> Any:
        # 按路径缓存 torchcodec VideoDecoder，LRU 淘汰
        decoder = VideoDecoder(file_handle, seek_mode="approximate")
```

以及 torchcodec 的 seek 能力（`cosmos3_action_lerobot.py:152`）：`seek_mode="approximate"`。

### 4.5 torchcodec 的 seek API（已确认，源码 `torchcodec/decoders/_video_decoder.py`）

| 方法 | 签名 | 作用 |
|------|------|------|
| `get_frame_at` | `(index: int) -> Frame` | 取单个帧 |
| `get_frames_at` | `(indices: list[int]) -> FrameBatch` | 取指定索引列表的多个帧 |
| `get_frames_in_range` | `(start, stop, step=1) -> FrameBatch` | 取 `[start, stop)` 开区间的帧（**推荐用这个**） |

**对本方案最合适的是 `get_frames_in_range`**：

```python
frames = decoder.get_frames_in_range(start=start_frame, stop=end_frame + 1)
# 开区间 [start, stop)，所以 stop = end_frame + 1
# 返回 FrameBatch（批量，比逐帧 get_frame_at 快）
```

### 4.6 seek_mode 精度（`_video_decoder.py:117-122`）

| seek_mode | 精度 | 代价 | 适用 |
|-----------|------|------|------|
| `exact`（默认） | 请求第 i 帧**一定**返回第 i 帧 | 初始要**扫描整个文件**建索引 | 要求帧精确 |
| `approximate` | 快，不扫描 | 用文件 metadata 估算帧位置，**可能不准** | 允许近似 |

> 关键结论：**本方案用 `seek_mode="exact"`（默认）**。因为 vision SFT 要精确切 episode 帧区间（切错一帧就把下一集的帧混进来）。`exact` 的扫描代价靠「共享 decoder」摊薄——50 个 episode 共享一个 decoder，只扫描一次。action 侧用 `approximate` 是因为它对精确帧不敏感、数据量大。
>
> 更优选项 `custom_frame_mappings`（第 130-133 行）：用 ffprobe 预生成帧→metadata 映射 JSON，既能精确又免扫描，适合反复训练同一批数据的场景。**建议先用 `exact` 跑通，再考虑优化。**

### 4.7 `_LRUVideoDecoderCache` 的功能与作用

**文件**：`cosmos3_action_lerobot.py:120-173`。它解决「反复打开/关闭 decoder」的低效——`VideoDecoder` 的 `exact` 模式要扫描整个文件，`__init__` 代价高，所以按路径缓存 decoder 复用。

核心结构（`OrderedDict` 实现 LRU）：

```python
class _LRUVideoDecoderCache:
    def __init__(self, max_size: int = 64):
        self._cache: OrderedDict[str, tuple[decoder, file_handle]] = {}

    def get_decoder(self, video_path: str) -> Any:
        # ① 命中缓存：move_to_end 标记最近使用，直接返回
        if video_path in self._cache:
            self._cache.move_to_end(video_path)
            return self._cache[video_path][0]

        # ② 未命中：打开文件建 decoder
        file_handle = fsspec.open(video_path).__enter__()
        decoder = VideoDecoder(file_handle, seek_mode="approximate")
        self._cache[video_path] = (decoder, file_handle)

        # ③ 满员：popitem(last=False) 淘汰最久未用，关闭文件句柄
        while len(self._cache) > self._max_size:
            _, (_, old_fh) = self._cache.popitem(last=False)
            old_fh.close()
        return decoder
```

三个关键点：

1. **LRU 用 `OrderedDict`**：`move_to_end`（命中标记最近）+ `popitem(last=False)`（淘汰头部 = 最久未用）。
2. **缓存 `(decoder, file_handle)` 二元组**：decoder 依赖底层文件句柄，淘汰时要 `close()` 释放。
3. **`max_size=64`**：限制同时打开的 decoder 数量，避免耗尽文件描述符/内存。

> 对本方案：**照搬 `_LRUVideoDecoderCache`，把 `seek_mode` 从 `approximate` 改成 `exact`** 即可，得到"共享 decoder + 按帧编号精确 seek"的核心组件。

### 4.8 改造后 process_one_sample 视频解码伪代码

改造前（`sft_dataset.py:202-263`）是「全量 decode + 按帧编号过滤」，改造后换成「共享 decoder + 按帧区间 seek」：

```python
# 改造后的视频解码核心逻辑（伪代码）
class SFTDataset(...):
    def __init__(self, ...):
        ...
        self._decoder_cache = _LRUVideoDecoderCache(max_size=...)  # 复用 action 侧缓存
        # 注意：seek_mode 用 exact（精确帧定位）

    def _decode_episode_frames(self, vision_path, start_frame, end_frame):
        """只解码 [start_frame, end_frame] 这一段的帧。"""
        decoder = self._decoder_cache.get_decoder(vision_path)  # 共享，同一 mp4 只建一次

        # torchcodec：开区间 [start, stop)，所以 stop = end_frame + 1
        frame_batch = decoder.get_frames_in_range(start=start_frame, stop=end_frame + 1)
        # frame_batch.data: [N, C, H, W]，dtype=uint8
        #   N = 帧数（= end_frame - start_frame + 1）
        #   dimension_order="NCHW"（默认）：N=batch, C=channel, H=height, W=width

        return frame_batch.data  # [N, C, H, W], uint8

    def process_one_sample(self, metadata):
        # ... 前面 uuid/window 解析、caption tokenize 等不变 ...

        # 改造点：视频解码从「全量 decode + 过滤」换成「区间 seek」
        video = self._decode_episode_frames(
            vision_path=metadata["vision_path"],
            start_frame=window["start_frame"],
            end_frame=window["end_frame"],
        )

        # 后续 resize、temporal_interval 抽帧、VAE 编码等逻辑不变
        # ...
```

**关键对应关系**：

| 原版（全量 decode） | 改造后（区间 seek） |
|--------------------|--------------------|
| `ffmpeg_decode_video(path)` 全量解码 | `decoder.get_frames_in_range(start, end+1)` 区间解码 |
| `if idx < start: continue` | `start=start_frame`（seek 起点） |
| `elif idx <= end: 取` | `stop=end_frame+1`（开区间） |
| `(idx-start) % interval == 0` 抽帧 | 解码后同样做抽帧（或后续统一处理） |
| 每个 episode 全量解码一次 | 同一 mp4 只解码目标区间 |

**三个待定实现点**（落到代码时确认）：

1. **dtype 转换**（已确认）：`frame_batch.data` 是 **uint8**（`_frame.py:73` 明确写 `torch.Tensor of uint8`），后续 VAE 编码前要转 float + normalize（`/255` 之类）。原版 `ffmpeg_decode_video` 返回的也是 uint8，所以这条其实是"沿用原版后处理"，不算新问题。

2. **resize**：原版 `ffmpeg_decode_video` 用 `-vf scale` 在解码时 resize。torchcodec 可在 `VideoDecoder(transforms=[...])` 里挂 resize transform，或解码后再手动 resize——选哪种看现有 resize 工具链。

3. **temporal_interval 抽帧**：原版在 decode 循环里按 `(idx-start) % interval` 抽帧。改造后 `get_frames_in_range` 支持 `step` 参数（`get_frames_in_range(start, stop, step=interval)`），可把抽帧下沉到 decode 层；但注意 step 是全局帧步进，和原版"相对 start 取余"的语义略有差异，需确认是否一致。

### 4.9 改动点

1. **按文件分组**：metadata 加载后，把同属一个 mp4 的 episode 聚在一起。
2. **decoder 缓存层**：复用/仿照 `_LRUVideoDecoderCache`，按 `vision_path` 缓存 decoder（seek_mode 用 exact）。
3. **seek + 区间解码**：替换「全量 decode + 过滤」为 `get_frames_in_range(start, end+1)` 按帧区间解码。
4. **帧编号定位**：因为存帧编号，seek 用帧索引，避开 timestamp 的浮点误差/开区间问题。

---

## 5. 待落地时确认的实现细节

| # | 待确认项 | 说明 |
|---|---------|------|
| 1 | ~~torchcodec seek 方法~~ | ✅ 已确认：`get_frames_in_range(start, stop)` 开区间，`stop=end_frame+1` |
| 2 | ~~seek_mode 精度~~ | ✅ 已确认：本方案用 `exact`（精确，扫描代价靠共享 decoder 摊薄） |
| 3 | 多文件场景的 from_timestamp 语义 | toy 是单 mp4；真实多文件数据需验证 `from_timestamp` 是全局时间还是文件内相对时间 |
| 4 | resize 时机 | 现有 `ffmpeg_decode_video` 支持 `scale_hw`；torchcodec 用 `transforms` 参数或解码后手动 resize |
| 5 | LRU 缓存大小 | 显存允许同时开几个 decoder |

---

## 6. 已确认的风险点与注意事项（版本迭代参考）

> 本节集中记录本次会话中**通过源码/数据验证**得出的易踩坑点，供后续版本迭代时参考，防止问题复发。

### 6.1 数据格式层（LeRobot）

| # | 风险点 | 结论 | 依据 |
|---|--------|------|------|
| 1 | v3 命名是单数还是复数 | **复数 `observation.images.*`**（不是单数 `image`） | toy 数据实际验证 |
| 2 | 一个 mp4 含几个 episode | **多个 episode 共享一个 mp4**，靠 `from/to_timestamp` 切分，不是一 episode 一文件 | toy 数据：50 episode 共享一个 file-000.mp4 |
| 3 | `to_timestamp` 是开区间还是闭区间 | **开区间**：该时刻的帧属于下一个 episode | toy 数据：`to×fps` 精确等于下一集起始帧 |
| 4 | `length` / 行跨度 / 时间跨度三者关系 | 精确相等：`length = to_index - from_index = round((to-from)×fps)` | toy 数据验证 |
| 5 | `frame_index` 是全局还是 episode 内 | **episode 内从 0 重新开始** | toy 数据：episode 1 首帧 frame_index=0 |

### 6.2 字段映射层

| # | 风险点 | 结论 |
|---|--------|------|
| 1 | `shape` 是 H×W 还是 W×H | **`[H, W, C]`**，所以 `width=shape[1]`, `height=shape[0]`，别搞反 |
| 2 | 视频字段 key 能否硬编码 | **不能**（`observation.images.top` 是数据集特定的），要动态抓第一个 `dtype=video` 字段 |
| 3 | `end_frame` 语义 | 存**帧编号**（文件内），非 timestamp；`end = round(to×fps) - 1`（-1 因开区间） |
| 4 | `aspect_ratio` | 是 `get_aspect_ratio(width,height)` 派生的，JSONL/LeRobot 里没有现成字段 |

### 6.3 加 caption 的侵入位置

| # | 风险点 | 结论 |
|---|--------|------|
| 1 | 加到 episodes 表 | ✅ **安全**，官方 lerobot 加载不报错（已实测），cosmos-framework 的 `self._episodes[idx]` 还能白捡该字段 |
| 2 | 加到 data 表 | ❌ **禁止**，会破坏 `info.json` 的 schema 校验，加载报 `CastError: column names don't match`（已实测） |
| 3 | `info.json` 的 `features` 约束范围 | **只严格约束 data 表**，不约束 episodes 表 |

### 6.4 视频解码层（torchcodec）

| # | 风险点 | 结论 |
|---|--------|------|
| 1 | `get_frames_in_range` 区间语义 | **开区间 `[start, stop)`**，所以 `stop = end_frame + 1` |
| 2 | `FrameBatch.data` 形状 | **4D `[N, C, H, W]`**（N=帧数），不是 3D |
| 3 | `FrameBatch.data` dtype | **uint8**，进 VAE 前要转 float + normalize |
| 4 | seek_mode 选哪个 | **`exact`**（精确帧定位），不是 action 侧的 `approximate`（可能定位到关键帧、切错帧） |
| 5 | `exact` 的扫描代价 | 每个 mp4 建 decoder 时扫描一次，靠「共享 decoder + LRU」摊薄到多个 episode |
| 6 | 不能全量 decode 缓存 | 一个 mp4 解压后 ~22GB（toy 数据），会 OOM，必须按帧区间 seek |

### 6.5 尚未验证、版本迭代时需重点关注

| # | 风险点 | 说明 |
|---|--------|------|
| 1 | 多文件场景的 `from_timestamp` 语义 | toy 是单 mp4，无法验证；真实多文件数据需确认是全局时间还是文件内相对时间 |
| 2 | resize 工具链 | torchcodec 的 `transforms` 参数 vs 解码后手动 resize |
| 3 | 抽帧语义 | `get_frames_in_range(step=interval)` 的全局步进 vs 原版「相对 start 取余」的差异 |

---

## 7. 实施流程（建议顺序）

```
步骤 1：写 LeRobot metadata 构造器
        └ 读 info.json + episodes parquet（含 caption 列）
        └ 产出 SFTDataset 认识的 metadata list（第 3 节映射）

步骤 2：验证字段映射正确性
        └ uuid、width/height、aspect_ratio、vision_path、t2w_windows 帧编号

步骤 3：改造视频解码（共享 decoder + 帧编号 seek）
        └ 先跑通功能（正确切出每个 episode 的帧）
        └ 再验证效率（同一 mp4 只打开一次）

步骤 4：config 接入
        └ 在 vision_sft_edge.py 的 dataloader 定义处接入新数据源

步骤 5：小规模冒烟训练
        └ 确认 caption、视频帧、sequence_plan、loss 全链路正确
```

---

## 8. 关键结论速查

| 问题 | 结论 |
|------|------|
| uuid 格式 | `chunk_{chunk_idx}_file_{file_idx}_episode_{ep_idx}` |
| width/height 来源 | 第一个 `dtype=video` 字段的 shape 前两位，`width=shape[1]`, `height=shape[0]` |
| t2w_windows 存什么 | **帧编号**（非 timestamp）：`start=round(from×fps)`, `end=round(to×fps)-1` |
| end_frame 是最后一帧吗 | 是「该 episode 的最后一帧」（文件内帧编号），不是「整个 mp4 最后一帧」 |
| 视频只读一次怎么做 | 共享 decoder + LRU 缓存 + 按帧编号 seek 只解码目标区间（非全量缓存） |
| 加 caption 到哪 | episodes 表加列（已验证安全）；**禁止加到 data 表**（会破坏 info.json schema 校验） |

---

## 附录：相关文件索引

| 文件 | 作用 |
|------|------|
| `cosmos_framework/data/generator/local_datasets/sft_dataset.py` | vision SFT 数据加载（`SFTDataset` + `get_sft_dataset` + metadata 解析） |
| `cosmos_framework/data/generator/local_datasets/helper.py` | `ffmpeg_decode_video`、`get_aspect_ratio`、`download_from_s3` |
| `cosmos_framework/data/generator/action/datasets/cosmos3_action_lerobot.py` | action 侧 LeRobot 加载 + `_LRUVideoDecoderCache`（可借鉴） |
| `cosmos_framework/data/generator/action/datasets/base_dataset.py` | `_video_path`（video_path 模板拼接，可参考） |
| `cosmos_framework/configs/base/experiment/sft/vision_sft_edge.py` | vision SFT 实验配置（dataloader 接入点） |
