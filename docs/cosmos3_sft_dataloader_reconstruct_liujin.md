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
- [9. 具体文件修改方案（做法 2：抽方法 + 子类 override）](#9-具体文件修改方案做法-2抽方法--子类-override)
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

**格式**：`{dataset_name}_chunk_{chunk_index}_file_{file_index}_episode_{episode_index}`

- 来源：数据集目录名（`root.name`）+ episodes 表的三列
- 目的：跨**数据集**、跨 chunk、跨 file、跨 episode 唯一，且能从名字看出各 index 含义

```python
uuid = f"{root.name}_chunk_{chunk_idx}_file_{file_idx}_episode_{ep_idx}"
```

> 说明：episode_index 在单数据集内虽唯一，但**多个数据集合并时**，不同数据集可能有相同的 chunk/file/episode 编号，会冲突。所以 uuid 前面加数据集目录名（`root.name`），保证跨数据集唯一。

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
| 4 | **pandas 3.0.5 破坏嵌套 list 列** | ❌ **不能用 pandas 3.0.5 读写 episodes parquet**（`stats/action/min` 等嵌套 list 列会损坏，报 `expected length 50 but got length 230` / `Malformed levels`）。加 caption 列用 **pyarrow**（append_column 保留 schema）或 **pandas 2.x**（py311 环境 2.3.1 实测 OK） |
| 5 | `tasks` 列的实际类型 | **`numpy.ndarray`**（不是 list/tuple）。代码里判断要用 `hasattr(x, "__len__")` 而非 `isinstance(x, (list, tuple))`（已修复） |
| 6 | **toy 数据集 stats 列 parquet 编码损坏** | ⚠️ **重大发现**：toy 数据集的 `stats/*/*` 嵌套 list 列，磁盘上的 def/rep level 编码损坏（`Malformed levels`）。读容忍，但**重写整个表（如加 caption 列）必然失败**（pyarrow 25 自己读→写→读都报错）。pyarrow 21（py311）容忍度更高能勉强写。真实私有数据需单独验证是否有此问题 |
| 7 | **stats 列损坏的最终解法** | ✅ **drop 掉 `stats/*/*` 列**（45 个损坏列，训练用不到），只保留 17 个有效列 + 加 caption 列，即可用 pyarrow 25 正常读写。已用此法成功构造 `toy_lerobot3_with_caption` 数据集 |

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
| uuid 格式 | `{dataset_name}_chunk_{chunk_idx}_file_{file_idx}_episode_{ep_idx}`（含数据集目录名，跨数据集唯一） |
| 多数据集加载 | `lerobot_root` 支持单数据集根 or 父目录；父目录自动 `rglob("meta/info.json")` 递归发现所有数据集 |
| width/height 来源 | 第一个 `dtype=video` 字段的 shape 前两位，`width=shape[1]`, `height=shape[0]` |
| t2w_windows 存什么 | **帧编号**（非 timestamp）：`start=round(from×fps)`, `end=round(to×fps)-1` |
| end_frame 是最后一帧吗 | 是「该 episode 的最后一帧」（文件内帧编号），不是「整个 mp4 最后一帧」 |
| 视频只读一次怎么做 | 共享 decoder + LRU 缓存 + 按帧编号 seek 只解码目标区间（非全量缓存） |
| 加 caption 到哪 | episodes 表加列（已验证安全）；**禁止加到 data 表**（会破坏 info.json schema 校验） |

---

## 9. 具体文件修改方案（做法 2：抽方法 + 子类 override）

> 采用**做法 2**（抽 `_decode_video_frames` 方法 + `LeRobotSFTDataset` 子类 override）。
> **唯一修改的文件**：`cosmos_framework/data/generator/local_datasets/sft_dataset.py`（修改后 **1084 行**）。
> 原 JSONL 流程（`get_sft_dataset`、`_load_sft_metadata_from_s3`、`SFTDataset` 原有行为）**完全不动**。
> ✅ **本方案已实际落地**，以下行号均为文件修改后的真实行号。

### 9.0 行号锚点（已落地后的真实行号）

| 锚点 | 行号 | 备注 |
|------|------|------|
| `SFTDataset` 类定义 | 98 | 原有，未动 |
| `_tokenize_caption` 方法 | 167-177 | 原有，未动 |
| **`_decode_video_frames`（父类默认）** | **179-207** | ✅ 新增（改动 1） |
| `process_one_sample` 方法 | 210 | 原有，decode 循环被替换（改动 2） |
| `_LeRobotVideoDecoderCache` | **495-528** | ✅ 新增（改动 3） |
| `LeRobotSFTDataset` | **530-577** | ✅ 新增（改动 3） |
| `_flatten_metadata_by_window` | 580 | 原有，偏移后 |
| `_load_sft_metadata_from_s3` | 600-696 | 原有，偏移后 |
| `_select_lerobot_video_key` | **704-729** | ✅ 新增（改动 4） |
| `_get_lerobot_video_width_height` | **731-745** | ✅ 新增（改动 4） |
| `_load_lerobot_metadata` | **747-853** | ✅ 新增（改动 4） |
| `get_sft_dataset` | 856-992 | 原有，偏移后 |
| `get_sft_dataset_from_lerobot` | **999-1085** | ✅ 新增（改动 5，文件末尾） |

---

### 9.1 改动 1：新增 `_decode_video_frames` 方法

**位置**：✅ 已落地在 **179-207 行**（`_tokenize_caption` 之后、`process_one_sample` 之前）。

**类型**：新增方法（原 `SFTDataset` 内）

**目的**：把 `process_one_sample` 里的视频解码循环抽成可 override 的钩子。

**改动前**（`process_one_sample` 内 253-263 行）：

```python
            video_chunk = []
            for idx, frame in enumerate(
                ffmpeg_decode_video(input_video_path, scale_hw=(resize_h, resize_w), num_threads=2)
            ):
                if idx < start_frame:
                    continue
                elif idx <= end_frame:
                    if (idx - start_frame) % temporal_interval == 0:
                        video_chunk.append(frame)
                else:
                    break
```

**改动后**（新增方法，放 178 行）：

```python
    def _decode_video_frames(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        temporal_interval: int,
        resize_h: int,
        resize_w: int,
    ) -> list[np.ndarray]:
        """Decode frames in [start_frame, end_frame] from a local video path.

        原版实现：全量 ffmpeg decode + 按帧编号过滤。
        返回 list[np.ndarray]（每帧 HWC uint8），供 process_one_sample 后续 np.stack。
        """
        video_chunk = []
        for idx, frame in enumerate(
            ffmpeg_decode_video(video_path, scale_hw=(resize_h, resize_w), num_threads=2)
        ):
            if idx < start_frame:
                continue
            elif idx <= end_frame:
                if (idx - start_frame) % temporal_interval == 0:
                    video_chunk.append(frame)
            else:
                break
        return video_chunk
```

**关键**：方法签名覆盖 decode 循环依赖的全部变量（video_path / start_frame / end_frame / temporal_interval / resize_h / resize_w），返回 `video_chunk`（list of HWC uint8），这样 `process_one_sample` 里 265 行之后的 `np.stack` 等逻辑**不用改**。

---

### 9.2 改动 2：修改 `process_one_sample` 的 decode 循环

**位置**：✅ 已落地在 `process_one_sample` 内（原 253-263 行，现偏移到 ~280 行附近，已替换为方法调用）。

**类型**：替换（10 行 decode 循环 → 1 行方法调用）

**改动后**（已落地）：

```python
            video_chunk = self._decode_video_frames(
                video_path=input_video_path,
                start_frame=start_frame,
                end_frame=end_frame,
                temporal_interval=temporal_interval,
                resize_h=resize_h,
                resize_w=resize_w,
            )
```

**注意**：
- `input_video_path`、`start_frame`、`end_frame`、`temporal_interval`、`resize_h`、`resize_w` 都是 `process_one_sample` 里已经存在的局部变量，无需改动。
- 之后的 `if not video_chunk` / `np.stack` / truncate / crop / transpose 逻辑**完全不动**。
- 这一改动对原 `SFTDataset` 的行为**零影响**（方法体就是原来那 10 行，抽到了 `_decode_video_frames`）。

---

### 9.3 改动 3：新增 `_LeRobotVideoDecoderCache` + `LeRobotSFTDataset(SFTDataset)` 子类

**位置**：✅ 已落地在 **487-577 行**（`SFTDataset` 类结束之后、`_flatten_metadata_by_window` 之前）。

**类型**：新增类（2 个：decoder 缓存 + 子类）

**目的**：override `_decode_video_frames`，用 torchcodec 共享 decoder + 按帧编号 seek，替换全量 decode。

> 说明：原本方案里计划「从 `cosmos3_action_lerobot` 引入 `_LRUVideoDecoderCache`」，但落地时发现从那里 import 会**连带触发 lerobot 库 import**（那个模块顶部 `from lerobot.datasets.lerobot_dataset import LeRobotDataset`）。为避免污染 vision SFT 的 import 路径，改为**本地定义** `_LeRobotVideoDecoderCache`，且 torchcodec/fsspec 惰性 import。

**已落地代码**（`_LeRobotVideoDecoderCache` 在 495-528 行，`LeRobotSFTDataset` 在 530-577 行）：

```python
class _LeRobotVideoDecoderCache:
    """按视频路径缓存 torchcodec VideoDecoder 的 LRU 缓存（seek_mode="exact"）。"""
    def __init__(self, max_size: int = 64):
        from collections import OrderedDict
        self._max_size = max_size
        self._cache: "OrderedDict[str, tuple]" = OrderedDict()

    def get_decoder(self, video_path: str):
        from torchcodec.decoders import VideoDecoder
        import fsspec
        if video_path in self._cache:
            self._cache.move_to_end(video_path)
            return self._cache[video_path][0]
        file_handle = fsspec.open(video_path).__enter__()
        decoder = VideoDecoder(file_handle, seek_mode="exact")
        self._cache[video_path] = (decoder, file_handle)
        while len(self._cache) > self._max_size:
            _, (_, old_fh) = self._cache.popitem(last=False)
            old_fh.close()
        return decoder


class LeRobotSFTDataset(SFTDataset):
    """override _decode_video_frames：共享 decoder + get_frames_in_range 区间 seek。"""
    def __init__(self, *args, decoder_cache_max_size: int = 64, **kwargs):
        super().__init__(*args, **kwargs)
        self._decoder_cache = _LeRobotVideoDecoderCache(max_size=decoder_cache_max_size)

    def _decode_video_frames(self, video_path, start_frame, end_frame, temporal_interval, resize_h, resize_w):
        import torch.nn.functional as F
        decoder = self._decoder_cache.get_decoder(video_path)
        frame_batch = decoder.get_frames_in_range(start=start_frame, stop=end_frame + 1)
        data = frame_batch.data  # [N, C, H, W] uint8
        if temporal_interval > 1:
            data = data[0::temporal_interval]
        data = data.float()
        data = F.interpolate(data, size=(resize_h, resize_w), mode="bicubic", align_corners=False)
        data = data.round().clamp(0, 255).to(torch.uint8)
        data_nhwc = data.permute(0, 2, 3, 1).cpu().numpy()
        return [data_nhwc[i] for i in range(data_nhwc.shape[0])]
```

**落地时补全的两个"待定项"**（原第 5 章标记）：

1. **resize**：用 `torch.nn.functional.interpolate(mode="bicubic")` 对齐原版 ffmpeg 的 `-vf scale + bicubic`。
2. **抽帧**：`data[0::temporal_interval]` 等价于原版「相对 start 取余 interval == 0」（因为 get_frames_in_range 返回的是连续帧，第 0 个元素就是全局 start_frame）。

**注意**：`LeRobotSFTDataset.__init__` 用 `*args, **kwargs` 透传，兼容父类 `SFTDataset.__init__`（101 行）签名，额外加 `decoder_cache_max_size`。

---

### 9.4 改动 4：新增 5 个 LeRobot metadata 函数

**位置**：✅ 已落地（`_load_sft_metadata_from_s3` 结束之后、`get_sft_dataset` 之前）。

**类型**：新增模块函数（5 个）

| 函数 | 作用 |
|------|------|
| `_select_lerobot_video_key` | 选定 video 字段（显式指定 → 第一个 usable → 第一个 video） |
| `_get_lerobot_video_width_height` | 从 video 字段 shape 抓 (width, height) |
| `_discover_lerobot_roots` | **多数据集发现**（新增） |
| `_load_single_lerobot_metadata` | 加载**单个**数据集（原 `_load_lerobot_metadata` 拆分而来） |
| `_load_lerobot_metadata` | 统一入口：发现多个数据集 → 逐个加载 → 合并 |

**目的**：读 LeRobot（支持**单个数据集根 or 父目录**）→ 产出和 `_load_sft_metadata_from_s3` **结构完全一致**的 metadata list。

### 9.4.1 多数据集发现（`_discover_lerobot_roots`）

支持两种 `lerobot_root`：

1. **单个数据集根**（直接含 `meta/info.json`）→ 返回 `[lerobot_root]`
2. **父目录**（不含 `meta/info.json`，其任意深度子目录含多个数据集）→ 递归 `rglob("meta/info.json")` 发现所有

```python
def _discover_lerobot_roots(lerobot_root: str) -> list[str]:
    root = Path(lerobot_root)
    if (root / "meta" / "info.json").is_file():   # 单数据集根
        return [str(root)]
    # 父目录：递归找所有 meta/info.json，取其上级目录
    dataset_roots = [str(Path(p).parent.parent) for p in root.rglob("meta/info.json")]
    if not dataset_roots:
        raise ValueError(f"在 {lerobot_root} 下没找到任何含 meta/info.json 的 LeRobot 数据集目录")
    return sorted(dataset_roots)
```

**关键**：不写死层数，`rglob` 递归查找任意深度。

### 9.4.2 字段映射

```python
def _load_lerobot_metadata(
    lerobot_root: str,
    min_frames: int = 61,
    min_short_edge: int = 0,
    video_feature_key: str | None = None,
    caption_key: str = "caption",        # episodes 表里 caption 列的列名
) -> list[dict]:
    roots = _discover_lerobot_roots(lerobot_root)   # 发现多个数据集
    metadata_list = []
    for root in roots:
        metadata_list.extend(_load_single_lerobot_metadata(root, ...))
    return metadata_list
```

单数据集内部字段映射（`_load_single_lerobot_metadata`）：

| 字段 | 来源 |
|------|------|
| `uuid` | `{root.name}_chunk_{c}_file_{f}_episode_{e}`（**含数据集目录名，保证跨数据集唯一**） |
| `vision_path` | info.json["video_path"] 模板填充 |
| `width`/`height` | video 字段 shape 前两位（shape=[H,W,C]） |
| `aspect_ratio` | `get_aspect_ratio(width, height)` |
| `t2w_windows` | 每 episode 一个 window，start/end 存**帧编号**：`start=round(from×fps)`, `end=round(to×fps)-1` |

### 9.4.3 caption 处理逻辑

```python
caption = row.get(caption_key)   # 只读 caption_key 指定的列，取不到为 None（不做 tasks 兜底）

# caption 为空时不写 caption key，让下游 _select_caption 找不到 key → 返回 None → 优雅跳过
window = {"start_frame": ..., "end_frame": ..., "temporal_interval": 1}
if caption:
    window["caption"] = caption
```

> 关键边界行为：缺 caption 列（或值为空）时，`window` 里**没有** `caption` key，下游 `_select_caption` 找不到已知 key → `return None` → `process_one_sample` 跳过该样本（对齐 JSONL 版本对缺 caption 的处理）。**不写 `caption: None`**，否则下游 `raw.strip()` 会 `AttributeError` 崩溃。

**关键**：返回的 metadata dict 字段必须和 `_load_sft_metadata_from_s3` **完全一致**：`uuid` / `vision_path` / `width` / `height` / `nb_frames` / `framerate` / `aspect_ratio` / `t2w_windows`。

---

### 9.5 改动 5：新增 `get_sft_dataset_from_lerobot` 函数

**位置**：✅ 已落地在 **1000-1084 行**（`get_sft_dataset` 结束之后，文件末尾）。

**类型**：新增模块函数

**目的**：LeRobot 版入口，构造 `LeRobotSFTDataset`（而非 `SFTDataset`）。

```python
def get_sft_dataset_from_lerobot(
    lerobot_root: str,
    resolution: str = "720",
    num_video_frames: int = -1,          # LeRobot 场景建议 -1（native chunk mode）
    video_feature_key: str | None = None,
    caption_key: str = "caption",        # ← 新增：传给 _load_lerobot_metadata
    decoder_cache_max_size: int = 64,
    # ... 其余参数与 get_sft_dataset 完全一致 ...
    **kwargs,
) -> LeRobotSFTDataset:
    """LeRobot 版 get_sft_dataset，产出 metadata 后构造 LeRobotSFTDataset。"""
    metadata_list = _load_lerobot_metadata(
        lerobot_root,
        video_feature_key=video_feature_key,
        caption_key=caption_key,          # ← 唯一来源差异 + 列名透传
    )

    # ↓ 以下和 get_sft_dataset 后半段完全一样 ↓
    if sample_by_window:
        metadata_list = _flatten_metadata_by_window(metadata_list)
    metadata_list.sort(key=lambda x: hashlib.sha256(x["uuid"].encode("utf-8")).hexdigest())

    return LeRobotSFTDataset(
        metadata=metadata_list,
        num_video_frames=num_video_frames,
        resolution=resolution,
        decoder_cache_max_size=decoder_cache_max_size,
        # ... 其余构造参数与 get_sft_dataset 一致 ...
    )
```

---

### 9.6 两入口函数差异清单（准确版）

| # | 差异点 | `get_sft_dataset` | `get_sft_dataset_from_lerobot` |
|---|--------|------------------|-------------------------------|
| 1 | 签名 | `jsonl_paths: str\|list[str]` | `lerobot_root: str` + `video_feature_key: str\|None` |
| 2 | metadata 来源 | `_load_sft_metadata_from_s3(...)` | `_load_lerobot_metadata(...)`（747-853 行） |
| 3 | 构造的类 | `SFTDataset(...)` | `LeRobotSFTDataset(...)`（530-577 行） |

**复用的部分**（两函数完全相同）：`_flatten_metadata_by_window`（580 行）+ `sha256` 排序 + 构造参数列表（因为子类 `__init__` 签名兼容父类）。

> 之前文档里"只差 metadata 来源一行"的说法**不准确**，实际是 3 处差异（签名、来源、类名），见 9.6。

---

### 9.7 完全不动（复用）的清单

| 项 | 说明 |
|----|------|
| `SFTDataset.__init__`（101 行） | 纯 metadata 驱动，不关心来源 |
| `SFTDataset.__len__`（164 行） | 不动 |
| `SFTDataset._tokenize_caption`（167-177 行） | 不动 |
| `process_one_sample` 的 download + window 计算部分 | 不动（LeRobot 走 native chunk mode 分支） |
| `process_one_sample` 的 stack/crop/caption tokenize/ret 构造部分 | 不动 |
| `__iter__`（391 行） | 不动 |
| `_flatten_metadata_by_window`（580 行） | 不动（被 get_sft_dataset_from_lerobot 复用） |
| `_load_sft_metadata_from_s3`（600 行） | 不动（JSONL 入口保留） |
| `get_sft_dataset`（856 行） | 不动（JSONL 入口保留） |

---

## 10. 训练接入改动（已落地，含留档注释）

> 本方案的**训练接入点**已实际改好：`vision_sft_edge.py`（dataloader 定义）+ 启动脚本。原 JSONL 版均**注释保留，未删除**。

### 10.1 `cosmos_framework/configs/base/experiment/sft/vision_sft_edge.py`

| 位置 | 改动 |
|------|------|
| 顶部 import（48 行后） | 新增 `from ... import get_sft_dataset_from_lerobot`（带 `# LeRobot 3.x 适配` 注释） |
| dataloader 的 `dataset=`（249 行起） | `L(get_sft_dataset)` → `L(get_sft_dataset_from_lerobot)`，参数从 `jsonl_paths` 换成 `lerobot_root` + `video_feature_key` + `caption_key`；原 JSONL 版整体注释保留 |

新参数：

```python
dataset=L(get_sft_dataset_from_lerobot)(
    ...
    lerobot_root="${oc.env:DATASET_PATH}",               # LeRobot 数据集根目录
    video_feature_key="observation.images.top",          # 指定 top 相机；不填则自动选第一个 usable
    caption_key="caption",                               # episodes 表的 caption 列名
    ...
)
```

### 10.2 `examples/launch_sft_vision_edge_yundao.sh`

| 位置 | 改动 |
|------|------|
| `DATASET_PATH`（第 9 行） | 改为 LeRobot 数据集根目录 `toy_lerobot3_with_caption`；原 JSONL 路径注释保留 |
| `EXTRA_DATASET_CHECK`（第 80 行） | 校验 `meta/info.json` 存在（原来是 `train/video_dataset_file.jsonl`）；原校验注释保留 |

---

## 11. 验证与实测记录

### 11.1 数据构造（drop stats 列 + 加 caption）

**问题**：toy 数据集的 episodes 表有 45 个 `stats/*/*` 嵌套 list 列，磁盘编码损坏，重写整个表必然失败（见 6.3）。

**解法**：drop 掉 `stats/*/*` 列（训练用不到），只保留 17 个有效列 + 加 caption 列，用 pyarrow 25 即可正常读写。

**结果**：✅ 成功构造 `/mi/data2T/liujin/dataset/toy_lerobot3_with_caption/00ri/so100_battery/`（episodes 表 18 列：17 有效 + caption）。

**caption 值**：`"Grasp a battery and put it in the bin."`（toy 数据 `tasks.parquet` 里自带的**任务指令**，50 个 episode 全部相同，非高质量视频描述，只够冒烟测试）。

### 11.2 `_load_lerobot_metadata` 字段验证

| 字段 | 验证值 | 结果 |
|------|--------|------|
| uuid | `chunk_0_file_0_episode_0` | ✅ |
| width/height | 640×480（shape `[480,640,3]` 正确取位） | ✅ |
| aspect_ratio | `4,3` | ✅ |
| 帧编号 | `[0,553]`、`[554,1083]`（to 开区间 -1 生效） | ✅ |
| vision_path | 指向 `observation.images.top`（usable=true） | ✅ |
| caption 读取 | 正确读到 caption 列 | ✅ |

### 11.3 训练全链路测试结果

**结果**：✅ **全链路跑通**（metadata → caption → 视频解码 → 训练前向 loss 计算成功）。

日志关键证据：
```
Total number of parameters: 1414924992（模型加载成功）
PackedSequence(sample_lens=[11184, 10464, 11184, 10544], ...)（4 个 sample 打包）
loss = 2.0362（前向成功）
```

---

## 附录：相关文件索引

| 文件 | 作用 |
|------|------|
| `cosmos_framework/data/generator/local_datasets/sft_dataset.py` | vision SFT 数据加载（`SFTDataset` + `get_sft_dataset` + `get_sft_dataset_from_lerobot` + metadata 解析） |
| `cosmos_framework/data/generator/local_datasets/helper.py` | `ffmpeg_decode_video`、`get_aspect_ratio`、`download_from_s3` |
| `cosmos_framework/data/generator/action/datasets/cosmos3_action_lerobot.py` | action 侧 LeRobot 加载 + `_LRUVideoDecoderCache`（可借鉴） |
| `cosmos_framework/data/generator/action/datasets/base_dataset.py` | `_video_path`（video_path 模板拼接，可参考） |
| `cosmos_framework/configs/base/experiment/sft/vision_sft_edge.py` | vision SFT 实验配置（dataloader 接入点，已改为 LeRobot） |
| `examples/launch_sft_vision_edge_yundao.sh` | 启动脚本（DATASET_PATH 已改为 LeRobot） |
