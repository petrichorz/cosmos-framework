# Cosmos3-Edge SFT 数据预处理代码走读

> 本文档走读两个数据预处理脚本，它们一起把「原始视频」变成「SFT 训练用的 JSONL」：
>
> 1. `caption_from_video.py` —— 用 VLM 给视频生成 caption（结构化 JSON + 稠密叙事）
> 2. `captions_to_sft_jsonl.py` —— 把 caption + 视频组装成 SFT JSONL
>
> 两者是**上下游**关系：`caption_from_video.py` 产出 caption 目录，`captions_to_sft_jsonl.py` 消费它。

---

## 目录

- [1. 整体流程图](#1-整体流程图)
- [2. caption_from_video.py 走读](#2-caption_from_videopy-走读)
  - [2.1 脚本定位与输入输出](#21-脚本定位与输入输出)
  - [2.2 CLI 参数](#22-cli-参数)
  - [2.3 主流程](#23-主流程)
  - [2.4 两阶段 VLM caption（Phase 1 + Phase 2）](#24-两阶段-vlm-captionphase-1--phase-2)
  - [2.5 关键函数清单](#25-关键函数清单)
- [3. captions_to_sft_jsonl.py 走读](#3-captions_to_sft_jsonlpy-走读)
  - [3.1 脚本定位](#31-脚本定位)
  - [3.2 输入格式](#32-输入格式)
  - [3.3 CLI 参数](#33-cli-参数)
  - [3.4 工作流程](#34-工作流程)
  - [3.5 输出格式](#35-输出格式)
- [4. 两个脚本的关系](#4-两个脚本的关系)
- [5. 私有数据接入指引](#5-私有数据接入指引)
- [附录：关键函数/文件索引](#附录关键函数文件索引)

---

## 1. 整体流程图

```
原始视频
   │
   ▼
caption_from_video.py          （VLM 打 caption）
   │  ┌─ 输入：视频文件 / manifest
   │  └─ 输出：output_dir/<clip>/caption.json + caption.txt
   │
   ▼
captions_to_sft_jsonl.py       （组装训练 JSONL）
   │  ┌─ 输入：captions_dir + videos_dir
   │  └─ 输出：train.jsonl + train.jsonl.summary.json
   │
   ▼
SFT 训练（sft_dataset.py 消费 JSONL）
```

---

## 2. caption_from_video.py 走读

**文件**：`cosmos_framework/scripts/caption_from_video.py`（339 行）

### 2.1 脚本定位与输入输出

用 **VLM（视觉语言模型）** 给视频自动生成两种 caption：

| 产物 | 内容 | 位置 |
|------|------|------|
| `caption.json` | 结构化 JSON caption（含 `temporal_caption` + 媒体字段） | 首选训练目标 |
| `caption.txt` | 稠密叙事（一段自然语言描述） | 备选 / 人读 |

VLM server 要求：**OpenAI chat-completions 兼容 + 支持视觉**（vLLM 跑 Qwen2-VL/Qwen3-VL、LLaVA-Next-Video 等），且本地视频需用 `--allowed-local-media-path` 指向视频根目录（见 docstring 第 14-17 行）。

### 2.2 CLI 参数

**文件**：`caption_from_video.py:61-92`（`Args` 类）

| 参数 | 别名 | 含义 | 默认 |
|------|------|------|------|
| `input_files` | `-i` | 输入 manifest（JSON/JSONL，含 `vision_path`） | None |
| `video` | `-v` | 单个视频文件 或 视频目录 | None |
| `output_dir` | `-o` | 输出目录 | 必填 |
| `server` | — | VLM API 地址 | `http://localhost:8000/v1` |
| `model` | — | 模型名（不填则用 server 的第一个模型） | None |
| `max_workers` | — | 并发请求数 | 16 |
| `max_retries` | — | 每个视频重试次数 | 5 |
| `timeout` | — | 单请求超时（秒） | 600 |
| `prompt_template_path` | — | 自定义 prompt 模板 | 内置 `video_captioner.txt` |
| `debug` | — | 保存原始 API 响应 | False |

> `-i` 和 `-v` 二选一，不能同时给（`caption_from_video.py:290-291`）。

#### `-i` manifest 输入格式（`_read_manifest_entries`，第 235-262 行）

`-i` 支持 JSON 或 JSONL 文件，每条 entry 需要 `vision_path`（本地路径或 http(s)/data URL），可选 `name` 和 `media`：

```jsonl
{"name": "clip001", "vision_path": "/path/clip001.mp4", "media": {"resolution": {"H":256,"W":256}, "fps": 5, "duration": "17s", "aspect_ratio": "1,1"}}
{"vision_path": "https://example.com/clip002.mp4"}
```

| 字段 | 必填 | 含义 |
|------|------|------|
| `vision_path` | ✅ | 视频本地路径或远程 URL |
| `name` | ❌ | clip 名，缺省用 `Path(vision_path).stem` |
| `media` | ❌ | 媒体字段 override（远程 URL 无法 ffprobe 时用） |

> `media` 的作用：远程 URL 视频 ffprobe 读不到，靠 manifest 里的 `media` dict 直接提供 `resolution/aspect_ratio/duration/fps`（见 `_process_single` 第 178-179 行的 `media_override` 分支）。

### 2.3 主流程

`caption_from_video()`（`caption_from_video.py:289-329`）：

```
① 加载 prompt 模板（内置 video_captioner.txt 或自定义）
② _collect_video_items() 收集 (name, video_ref, media_override) 列表
③ 建 AsyncOpenAI client（api_key="EMPTY"，base_url=server）
④ 不指定 model 时，client.models.list() 取第一个
⑤ 建 asyncio.Semaphore(max_workers) 限并发
⑥ 对每个视频并发跑 _process_single()
⑦ 统计成功/失败
```

**并发模型**：`asyncio + openai.AsyncOpenAI + Semaphore`，`max_workers=16` 控制并发上限。

### 2.4 两阶段 VLM caption（Phase 1 + Phase 2）

这是脚本的核心。prompt 模板（`inference/defaults/video_captioner.txt`）让 VLM **一次调用分两步输出**，返回带 XML 标签的文本。

#### 输入消息构建（`_build_vlm_messages`，第 113-123 行）

```python
messages = [{
    "role": "user",
    "content": [
        {"type": "video_url", "video_url": {"url": _video_url(video_ref)}},
        {"type": "text", "text": prompt_template},
    ],
}]
```

视频通过 `video_url` content part 直接传给 VLM。本地路径转 `file://` URL（`_video_url`，第 100-110 行），远程 URL 原样透传。

#### VLM 调用（`_process_single`，第 141-148 行）

```python
response = await client.chat.completions.create(
    model=args.model,
    messages=messages,
    max_tokens=2048,
    temperature=0.7,
    top_p=0.8,
    extra_body={"top_k": 20, "min_p": 0.0},
)
```

#### 拆出 Phase 1 和 Phase 2（第 165-174 行）

```python
text = choice.message.content.strip()

final_prompt = extract_xml_tag(text, "final_prompt")   # Phase 2 产物
scene_draft  = parse_structured_caption(text)           # Phase 1 产物
```

VLM 返回的文本结构：

```xml
<scene_draft>
{ "subjects": [...], "lighting": {...}, "actions": [...], ... }
</scene_draft>

<final_prompt>
Two robotic arms sit at a wooden table, the right arm reaches...
</final_prompt>
```

| 阶段 | VLM 产物 | 提取函数 | 最终去向 |
|------|---------|---------|---------|
| Phase 1 | `<scene_draft>` 结构化 JSON | `parse_structured_caption`（`structured_caption.py:184`） | `caption.json` 主体 |
| Phase 2 | `<final_prompt>` 稠密叙事 | `extract_xml_tag(text, "final_prompt")`（`structured_caption.py:136`） | `caption.txt` + `temporal_caption` 字段 |

**Phase 1（结构化 JSON）**：VLM 按模板里的 JSON schema 填 `subjects`/`lighting`/`cinematography`/`actions`/`segments` 等槽位（`video_captioner.txt:11-83`）。模板明确要求「不要填 `resolution`/`aspect_ratio`/`duration`/`fps`，这些从视频文件自动填」。

**Phase 2（稠密叙事）**：VLM 把 Phase 1 的场景分析改写成一整段连贯的自然语言（`video_captioner.txt:86-97`），规则包括：一个段落、不引用"the video/the scene"、用主体自身视角描述方位、只描述可见内容不幻觉。

**prompt 模板的完整内容**（`inference/defaults/video_captioner.txt`，97 行）分三块：

Phase 1 的 JSON schema 顶层字段（`video_captioner.txt:12-83`）：

| 字段 | 类型 | 含义 |
|------|------|------|
| `subjects[]` | 对象数组 | 主体（含 description/appearance_details/pose/action/… 等 ~20 个子字段） |
| `background_setting` | 字符串 | 环境与场景 |
| `lighting{}` | 对象 | 光照（conditions/direction/shadows/illumination_effect） |
| `aesthetics{}` | 对象 | 美学（composition/color_scheme/mood_atmosphere/patterns） |
| `cinematography{}` | 对象 | 镜头（camera_motion/framing/camera_angle/depth_of_field/focus/lens_focal_length） |
| `style_medium` / `artistic_style` / `context` | 字符串 | 媒介/风格/场景语境 |
| `actions[]` | 对象数组 | 时间分段动作（time/description） |
| `text_and_signage_elements[]` | 对象数组 | 画面内文字（text/category/appearance/…） |
| `segments[]` | 对象数组 | 分段（segment_index/time_range/description/key_changes/camera） |
| `transitions[]` | 数组 | 段间转场 |
| `temporal_caption` | 字符串 | 逐拍叙事 |
| `audio_description` | 字符串 | 音频描述 |

Phase 2 的完整写作规则（`video_captioner.txt:89-96`）：

1. **Length & Format**：写**一个**连贯段落（多句完整句），不用 bullet。
2. **Content**：覆盖主体、场景、时序动作、背景元素、光照、镜头、时间结构。
3. **Taboo Words**：不引用视频本身（禁 "the video/scene/clip/frame"）。
4. **Perspective**：从主体自身视角描述左右（"her right hand" = 主体的右手）。
5. **Spatial Phrasing**：避免镜头中心词（"enters the frame"），改用空间关系（"approaches from the left"）。
6. **Tone**：中性、客观，无观点、无揣测动机。
7. **Faithfulness**：只描述实际可见内容，不幻觉。

> 关键：模板明确要求 VLM **不要填** `resolution`/`aspect_ratio`/`duration`/`fps`（第 7 行）——这些由脚本从视频 ffprobe 自动填，避免 VLM 猜错。

#### Phase 1 提取的三级容错（`parse_structured_caption`）

Phase 2 的 `<final_prompt>` 用简单正则 `extract_xml_tag` 提取（`structured_caption.py:136`），但 Phase 1 的 `<scene_draft>` 是 **JSON**，VLM 输出格式可能不稳定，所以用 `parse_structured_caption`（`structured_caption.py:184-213`）做**三级容错**：

```python
def parse_structured_caption(text: str) -> dict | None:
    candidates = []
    tagged = extract_xml_tag(text, "scene_draft")   # ① 先找 <scene_draft> 标签
    if tagged is not None:
        candidates.append(tagged)
    candidates.append(text)                          # ② 把整个响应也作为候选

    for candidate in candidates:
        cleaned = _strip_code_fences(candidate)      # 去 ```json 围栏
        for blob in (cleaned, _first_json_object(cleaned)):   # ③ 先整体，再抠第一个 {...}
            if not blob:
                continue
            try:
                parsed = json.loads(blob)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None
```

三级容错，从强到弱：

| 级 | 尝试 | 应对的 VLM 情况 |
|----|------|---------------|
| ① | `<scene_draft>` 标签内容 | 乖乖按模板输出 |
| ② | 整个响应直接当 JSON | 不包标签，直接吐 JSON |
| ③ | `_first_json_object` 抠第一个 `{...}` | JSON 外裹了废话/围栏 |

两个辅助函数：

- **`_strip_code_fences`（第 142-148 行）**：去掉 Markdown 代码围栏（` ```json ` / ` ``` `）。
- **`_first_json_object`（第 151-181 行）**：花括号计数，抠出第一个平衡的 `{...}` 块，且**手动跟踪字符串状态**（`in_str`/`escaped`），忽略 JSON 字符串内部（如 `"{"`）的花括号，避免计数被干扰。

> 为什么要这么"折腾"：因为 VLM 输出格式不可靠——可能按模板输出、可能不包标签、可能在 JSON 外裹解释文字。`parse_structured_caption` 的目标是「能解析就解析，解析不了换下一种」，最大程度从 VLM 的不稳定输出里抠出可用的 JSON（对应其 docstring 的 "permissive"——做结构验证，而非拒绝）。

#### 组装最终 caption_json（第 189-190 行）

```python
caption_json = assemble_caption_json(scene_draft, final_prompt, media)
```

`assemble_caption_json`（`structured_caption.py:241`）做三件事：

1. `data["temporal_caption"] = final_prompt`（Phase 2 稠密叙事内嵌进 `temporal_caption`）
2. `data.update(media)`（分辨率/帧率/时长等媒体字段）
3. `model_dump(exclude_none=True)`（清洗空字段，pydantic 校验）

**`media` 字段从哪来**（第 178-187 行）：三选一，优先级从高到低：

```python
if media_override is not None:
    media = media_override                     # ① manifest 提供的 override（远程 URL 用）
elif not _is_remote_ref(video_ref):
    media = media_fields_from_metadata(probe_video_metadata(video_ref))  # ② 本地视频 ffprobe
else:
    media = {}                                  # ③ 远程 URL 且无 override，留空
```

`probe_video_metadata`（`scripts/video_metadata.py:18`）用 ffprobe 返回 `{fps, duration, width, height, total_frames}`；`media_fields_from_metadata`（`structured_caption.py:226`）把它转成 caption 的 media 字段：

```python
def media_fields_from_metadata(meta: dict) -> dict:
    width, height = int(meta["width"]), int(meta["height"])
    return {
        "resolution": {"H": height, "W": width},   # 宽高
        "aspect_ratio": aspect_ratio_str(width, height),  # 宽高比（如 "1,1"、"16,9"）
        "duration": f"{round(float(meta['duration']))}s",  # 如 "17s"
        "fps": int(round(float(meta["fps"]))),             # 取整帧率
    }
```

> 注意：这就是为什么 prompt 模板里要求 VLM **不要填** `resolution`/`aspect_ratio`/`duration`/`fps`——这些媒体字段是脚本从视频文件 ffprobe 出来、自动填进 caption_json 的，不靠 VLM 猜（VLM 猜时长/帧率会不准）。

#### 写文件（第 195-205 行）

```python
output_dir = args.output_dir / name
(output_dir / "sample_args.json").write_text(...)    # 推理用采样配置
(output_dir / "caption.txt").write_text(final_prompt)   # 稠密叙事
(output_dir / "caption.json").write_text(json.dumps(caption_json, indent=2, ...))  # 结构化 JSON
```

三个产物里，`caption.json` 和 `caption.txt` 是给训练的；**`sample_args.json` 是给推理的**——它由 `OmniSampleOverrides`（第 197-202 行，`inference/args.py`）生成，记录这个 clip 的推理采样配置（`name`/`prompt`/`vision_path`/`output_dir` 等），**训练阶段完全不用**，`captions_to_sft_jsonl.py` 后续也会忽略它（只读 `caption.json` + `caption.txt`）。

#### 重试机制

`_process_single` 里 `for i_retry in range(max_retries)` 循环（第 139 行），每次失败（API 错误、finish_reason 非 stop、提取 XML 失败、JSON 校验失败）都 sleep 1 秒重试，最多 5 次。

#### 长度告警（第 207-215 行）

```python
approx_tokens = len(json.dumps(caption_json)) // 4   # 粗略估计 token 数
if approx_tokens > 1024:
    log.warning("structured caption is ~N tokens ... ensure max_caption_tokens covers it")
```

因为 SFT loader 会按 `max_caption_tokens` 截断超长 prompt（我们走读文档讲过），这里提前告警。

### 2.5 关键函数清单

| 函数 | 行号 | 作用 |
|------|------|------|
| `_is_remote_ref` | 95 | 判断是否远程 URL/data URI |
| `_video_url` | 100 | 本地路径 → `file://` URL |
| `_build_vlm_messages` | 113 | 组装视频 + prompt 消息 |
| `_process_single` | 126 | 处理单个视频（含重试） |
| `_process_with_semaphore` | 222 | 加并发限流 |
| `_read_manifest_entries` | 235 | 解析 `-i` manifest |
| `_collect_video_items` | 265 | 从 CLI 收集视频列表 |
| `caption_from_video` | 289 | 主流程 |

---

## 3. captions_to_sft_jsonl.py 走读

**文件**：`cosmos_framework/scripts/captions_to_sft_jsonl.py`（229 行）

### 3.1 脚本定位

`caption_from_video.py` 的**下游**——把「caption 目录 + 视频目录」转成 SFT 训练 JSONL。它**不生成 caption、不切分视频**，只做「配对 + 组装 + 过滤」。

### 3.2 输入格式

两个目录，靠**目录名 = clip 名**配对：

```
captions_dir/                    videos_dir/
├── clip001/                     ├── clip001.mp4
│   ├── caption.txt              ├── clip002.mp4
│   └── caption.json   (可选)     └── clip003.mp4
├── clip002/
│   └── caption.txt
└── clip003/
    ├── caption.txt
    └── caption.json
```

| 文件 | 内容 | 作用 |
|------|------|------|
| `caption.json` | 结构化 JSON caption（Phase 1 产物） | **首选**（`caption_json` 字段） |
| `caption.txt` | 稠密叙事（Phase 2 产物） | **备选**（`caption` 字段） |

配对规则（`_find_video`，第 68-73 行）：caption 子目录名 `name` → `videos_dir/name.{mp4,mov,avi,mkv,webm}`。

### 3.3 CLI 参数

**文件**：`captions_to_sft_jsonl.py:88-106`

| 参数 | 别名 | 含义 | 默认 |
|------|------|------|------|
| `captions_dir` | — | caption 子目录的父目录 | 必填 |
| `videos_dir` | — | 视频文件目录 | 必填 |
| `output` | `-o` | 输出 JSONL 路径 | 必填 |
| `caption_key` | — | 稠密 caption 的字段名 | `caption` |
| `num_video_frames` | — | 训练每窗口帧数（决定帧数过滤下限） | -1（只用 61 帧下限） |
| `min_short_edge` | — | 短边最小像素，0 关闭 | 0 |

### 3.4 工作流程

对每个 caption 子目录，按顺序做 5 步（第 123-191 行）：

```
① 读 caption.txt（稠密）+ caption.json（结构化，可选）
   ├─ caption.json 解析失败 → 只用稠密，告警
   └─ 两者都空 → SKIP（empty_caption）

② 找视频 → 找不到 → SKIP（missing_video）

③ probe_video_metadata 读元信息（ffprobe）→ 失败 → SKIP（ffprobe_error）

④ 过滤（3 个条件）：
   ├─ duration > 61s → SKIP（duration_too_long）
   ├─ 帧数 < max(61, num_video_frames) → SKIP（too_few_frames）
   └─ 短边 < min_short_edge → SKIP（short_edge_too_small）

⑤ 组装一个 window + 一条 record
```

**过滤逻辑与训练对齐**（docstring 第 19-25 行）：过滤条件刻意 mirror `sft_dataset.py` 训练时的硬限制，保证"JSONL 样本数"和"训练实际能吃到的样本数"一致。

三个过滤常量（第 63-65 行）：

```python
_MAX_DURATION = 61.0   # 匹配 sft_dataset.py 的硬上限
_MIN_FRAMES = 61       # 匹配 get_sft_dataset() 的 min_frames=61
_VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
```

### 3.5 输出格式

每行一条 record：

```json
{
    "uuid": "clip001",
    "duration": 17.4,
    "width": 256,
    "height": 256,
    "vision_path": "videos/clip001.mp4",
    "t2w_windows": [{
        "start_frame": 0,
        "end_frame": 417,
        "temporal_interval": 1,
        "caption_json": {
            "subjects": [...],
            "background_setting": "...",
            "lighting": {...},
            "cinematography": {...},
            "actions": [...],
            "segments": [...],
            "temporal_caption": "A robotic arm ...",
            "audio_description": "...",
            "resolution": {"H": 256, "W": 256},
            "aspect_ratio": "1,1",
            "duration": "17s",
            "fps": 24
        },
        "caption": "A robotic arm ..."
    }]
}
```

> 说明：`caption_json` 就是 `caption_from_video.py` 生成的 `caption.json` 原样写入——Phase 1 结构化字段（subjects/lighting/cinematography/actions/segments…）+ Phase 2 稠密叙事（`temporal_caption`）+ 媒体字段（`resolution`/`aspect_ratio`/`duration`/`fps`）。
> 其中 `caption`（稠密）与 `caption_json.temporal_caption` 是**同一段话**（`captions_to_sft_jsonl.py` 里 `caption.txt` 的内容就是 final_prompt，又被存进 `temporal_caption`）。
> 自洽性：`end_frame=417`（418 帧）+ `fps=24` → `418/24 ≈ 17.4s`，与顶层 `duration` 一致。

关键点：

1. **每条 record 一个 window**，`start_frame=0, end_frame=总帧数-1`（整个视频是一个 window，不切分）。
2. **`caption_json` 和 `caption` 都写**（如果都有）——`caption_json` 首选，`caption` 备选。
3. **`vision_path` 相对化**（`_relativize_vision_path`，第 76-85 行）：改成相对 JSONL 所在目录的路径，整个数据集可迁移；`s3://` 等 URI 原样保留。
4. **`temporal_interval=1`** 固定（不抽帧）。

#### summary 文件

写一个 `output.summary.json` 文件（第 198-214 行）：

```json
{
    "records_kept": 123,
    "records_with_caption_json": 100,
    "records_dropped": 10,
    "drops_by_reason": {"too_few_frames": 5, "empty_caption": 5},
    "filters": {...}
}
```

---

## 4. 两个脚本的关系

```
caption_from_video.py                captions_to_sft_jsonl.py
        │                                   │
   生成 caption 目录                      消费 caption 目录
        │                                   │
   output_dir/                          captions_dir/（同一个目录）
   ├── clip001/                         ├── clip001/
   │   ├── caption.json                 │   ├── caption.json
   │   ├── caption.txt                  │   ├── caption.txt
   │   └── sample_args.json             │   └── sample_args.json（忽略）
   └── clip002/...                      └── clip002/...
        │                                   │
        └─────────────┬─────────────────────┘
                      ▼
              captions_to_sft_jsonl 还要 videos_dir
                      ▼
              train.jsonl（SFT 训练输入）
```

**关键衔接**：`caption_from_video.py` 的 `output_dir` 就是 `captions_to_sft_jsonl.py` 的 `captions_dir`（前者给每个 clip 生成子目录，后者按子目录名去 `videos_dir` 找视频配对）。

## 5. 私有数据接入指引

以 LeRobot 3.1 格式的私有数据为例，接入 vision SFT 的完整链路：

```
LeRobot 3.1 数据（meta/ + videos/ + data/parquet）
   │
   ▼
[前置脚本]：从 LeRobot 提取视频 + 映射 caption
   │   产出扁平的 videos_dir/（clip名.mp4）
   │   若有现成语言标注 → 生成 caption.txt
   ▼
caption_from_video.py           （若无 caption，用 VLM 从视频生成 caption.json + caption.txt）
   │
   ▼
captions_to_sft_jsonl.py        （组装成 train.jsonl）
   │
   ▼
SFT 训练
```

**两个关键缺口**：

1. **LeRobot 视频位置**：在 `videos/observation.images.<cam>/chunk-xxx/file-xxx.mp4`，不是扁平 `clip名.mp4`，需前置脚本按 episode 提取。
2. **caption 缺失**：LeRobot 原生没有文本 caption，需用 `caption_from_video.py`（VLM 生成）或复用私有数据的语言标注。

#### LeRobot 3.1 目录结构

LeRobot 3.1 是 HuggingFace datasets 结构：

```
your_dataset/
├── meta/
│   ├── info.json                # 数据集元信息（codebase_version 等）
│   ├── tasks.jsonl
│   ├── episodes/                # 每个 episode 的元数据（起始/结束帧、任务描述）
│   └── stats/
├── videos/
│   └── observation.images.<cam>/   # 每路相机的观测视频
│       ├── chunk-000/file-000.mp4
│       └── ...
└── data/
    └── chunk-000/*.parquet      # observations/actions/states 表
```

#### 前置脚本骨架（LeRobot → 扁平视频目录）

核心思路：遍历 episode，提取每个 episode 的观测视频，排成 `captions_to_sft_jsonl.py` 要的输入。**以下是概念性伪代码，需按你的 LeRobot 版本和相机配置调整**：

```python
# 概念骨架：LeRobot episode → 扁平 videos_dir/ + 可选 captions_dir/
from lerobot.datasets import LeRobotDataset

ds = LeRobotDataset(repo_id="your/dataset")   # 或本地路径

for episode_index in ds.meta.episodes:
    # ① 提取该 episode 的观测视频（选定一路相机）
    #    注意：LeRobot 视频按 chunk 分片，一个 episode 可能跨多个 chunk
    video_frames = extract_episode_video(ds, episode_index, camera="observation.images.<cam>")

    # ② 存成扁平视频文件，命名 = clip 名
    name = f"episode_{episode_index:06d}"
    save_mp4(video_frames, f"videos_dir/{name}.mp4")

    # ③ 若有现成语言标注（如 tasks.jsonl 里的任务描述），写 caption
    if has_language_annotation(episode_index):
        write_text(f"captions_dir/{name}/caption.txt", annotation)
```

落地时的关键点：

| 注意点 | 说明 |
|--------|------|
| chunk 分片 | LeRobot 视频按 `chunk-xxx/file-xxx.mp4` 分片，一个 episode 可能跨多个 chunk 文件，需按 episode 的起止帧拼接 |
| 多相机 | `observation.images.<cam>` 是每路相机一个目录，需选定一路（或合并多路）作为训练视角 |
| caption 来源 | 若 `tasks.jsonl`/episode 元数据里有任务描述，可直接当稠密 caption；否则交给 `caption_from_video.py` 用 VLM 生成 |
| 视频格式 | LeRobot 观测视频帧率/分辨率可能不统一，后续训练时由 `sft_dataset.py` 统一 resize/降帧率处理 |

> 产出 `videos_dir/`（扁平 `episode_xxx.mp4`）后，走 `caption_from_video.py`（生成 caption）→ `captions_to_sft_jsonl.py`（组装 JSONL），就接上了本仓库的标准 SFT 数据流。

---

## 附录：关键函数/文件索引

| 文件/函数 | 作用 |
|-----------|------|
| `cosmos_framework/scripts/caption_from_video.py` | VLM 给视频生成结构化 + 稠密 caption |
| `cosmos_framework/scripts/captions_to_sft_jsonl.py` | caption + 视频 → SFT JSONL |
| `cosmos_framework/inference/defaults/video_captioner.txt` | 两阶段 caption 的 prompt 模板 |
| `cosmos_framework/inference/structured_caption.py` | 结构化 JSON 的 schema 与解析/组装工具 |
| `cosmos_framework/scripts/video_metadata.py` | ffprobe 读视频元信息 |
