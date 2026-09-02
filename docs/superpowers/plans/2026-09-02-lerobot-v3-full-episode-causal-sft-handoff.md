# LeRobot v3 完整 Episode Causal Vision SFT 数据适配 Handoff

日期：2026-09-02
状态：第一版已实现，等待真实训练验证与剩余训练语义确认

第一版实现位置：

- `cosmos_framework/data/generator/local_datasets/lerobot_episode_sft_dataset.py`
- `cosmos_framework/configs/base/experiment/sft/vision_sft_edge_lerobot3.py`

实现同时在 `SFTDataset` 中抽取了公共的 worker 初始化与已解码样本组装入口，
LeRobot 数据集复用原有 caption、CFG dropout、conditioning 和 `SequencePlan`
逻辑，不再复制整段 `process_one_sample()`。

当前第一版读取 episode-level `tasks` 或用户指定的 caption 字段。由于 subtask
如何进入训练文本仍未确定，暂不读取 frame-level timestamp/subtask 列，也不会把
未消费的大型标注对象塞入 DataLoader batch；确定语义后应在 reader 层补齐并与
视频 stride 使用相同的采样索引。

## 1. 目标

实现一个与具体启动脚本和 experiment 解耦的通用数据适配层，将一个或多个本地 LeRobot v3 数据集转换为现有 Cosmos causal vision SFT 所消费的样本格式。

核心需求如下：

1. 输入路径既可以是单个 LeRobot v3 数据集根目录，也可以是包含多个 LeRobot v3 子数据集的父目录。
2. 每个训练样本对应一个完整 episode，不使用固定窗口或随机片段。
3. 同步读取该 episode 的视频、task/caption、subtask 或其他文本标注、timestamp 等数据；本次不读取 action。
4. 训练侧继续消费既有 vision SFT 字段，不要求模型知道数据来自 LeRobot。
5. 视角可配置，默认读取 head 视角。
6. 视频使用 Cosmos 既有 `VIDEO_RES_SIZE_INFO` 分桶，并遵循 causal vision SFT 原有的等比例 cover resize 和 center crop 语义。
7. 尽量复用现有 LeRobot 和 Cosmos 数据处理函数，避免再次手写和维护完整的 LeRobot v3 schema。

## 2. 本规划不包含的内容

- 不绑定 `launch_sft_vision_edge_local.sh` 或其他具体启动脚本。
- 当前阶段不决定使用哪个具体 experiment；数据集工厂应能被任意合适的 causal vision recipe 引用。
- 本次不读取或训练 action，不启用 action head、action tokenizer 或 action loss。
- 第一版只规划单视角输入，不规划多视角拼接。
- 当前文档只落盘设计和决策点，不进行代码修改。

## 2.1 已确认的实现决策

1. 数据适配与具体启动脚本和 experiment 解耦。
2. LeRobot 视频路径解析不抽取公共函数，直接复制现有 `_video_path()` 的兼容逻辑并注明来源。
3. 视频使用 TorchCodec，不使用命令行 ffmpeg。
4. 不使用 decoder LRU；每个 episode 独立创建并释放 decoder。
5. TorchCodec 在 CPU 上运行，通过 decoder transforms 完成 resize 和 center crop。
6. 不允许先返回完整 1080p episode Tensor，再在 Python/PyTorch 层缩放到 480p/256p。
7. LeRobot 根目录发现保持单线程，采用“输入路径自身、一级子目录、递归兜底”的分级扫描。
8. 不同数据集的 `LeRobotDatasetMetadata` 使用受控线程池并行加载，默认最多 8 个线程，并在 DataLoader worker 创建前完成。
9. 本次完全不读取 action，不构造 action Tensor，不放入 CPU/NPU batch，也不参与 packing、`SequencePlan` 或 loss；只保留代码注释说明未来的 action 扩展位置。
10. 最大输出 FPS 使用整数 stride：`ceil(original_fps / max_output_fps)`；参数语义是 FPS 上限，不要求精确重采样到目标 FPS。

## 3. 已确认的现有能力

### 3.1 LeRobot 元数据与多数据源管理

现有 action LeRobot 适配已经使用 `LeRobotDatasetMetadata` 读取 `info.json`、episodes 和 tasks，并支持 metadata-only 注册、延迟构造 `LeRobotDataset` 以及多数据源管理，参见：

- `cosmos_framework/data/generator/action/datasets/cosmos3_action_lerobot.py:443`
- `cosmos_framework/data/generator/action/datasets/cosmos3_action_lerobot.py:628`

它还提供了确定性的 train/val episode 切分以及基于 `dataset_from_index`、`dataset_to_index`、`length` 的 episode 范围计算：

- `cosmos_framework/data/generator/action/datasets/cosmos3_action_lerobot.py:230`
- `cosmos_framework/data/generator/action/datasets/cosmos3_action_lerobot.py:243`

完整 episode 数据集不直接复用 `build_episode_spans()` 的窗口计算结果，但应复用同一组 LeRobot episode 索引字段。

### 3.2 LeRobot v3 task 兼容

现有基础 action dataset 已兼容 LeRobot v2 的 `task` 列以及 LeRobot v3 将 task 文本存储在 DataFrame index 中的形式：

- `cosmos_framework/data/generator/action/datasets/base_dataset.py:66`
- `cosmos_framework/data/generator/action/datasets/base_dataset.py:73`

新的 episode adapter 不应只依赖自定义的 `episodes.parquet::caption` 列；应优先使用官方 metadata/task 映射，并将自定义 caption 作为可配置覆盖项。

### 3.3 视频路径解析

现有 `_video_path()` 已兼容 `chunk_index/file_index`、`episode_chunk/episode_file` 等路径字段：

- `cosmos_framework/data/generator/action/datasets/base_dataset.py:160`

本次不抽取公共函数。直接将 `_video_path()` 中这段已验证的路径兼容逻辑复制到新的 LeRobot vision dataset 内，并注明来源。这样可以避免修改 action 数据集模块或引入 vision-to-action 的跨模块依赖，同时保持对上述 LeRobot v2/v3 路径字段的兼容。

### 3.4 TorchCodec 视频解码（已决定不使用 LRU）

现有 LeRobot action 路径实现了带容量上限的 LRU decoder cache，并修补 LeRobot 的默认 decoder cache：

- `cosmos_framework/data/generator/action/datasets/cosmos3_action_lerobot.py:120`
- `cosmos_framework/data/generator/action/datasets/cosmos3_action_lerobot.py:189`

本任务已决定**不复用 LRU**。完整 episode 一次连续解码，本身不需要在单个样本内部复用 decoder；在视频数量很多、episode 全局打乱且一个 episode 对应一个视频的情况下，跨样本缓存的命中率预计较低，而每个 worker 长期保留多个 decoder 会放大文件描述符和 CPU 内存占用。

新实现采用以下生命周期：

```text
开始读取 episode
  -> 使用本地视频路径创建 CPU TorchCodec VideoDecoder
  -> 一次读取 episode 所需的连续帧区间
  -> 返回已经完成抽帧、resize 和 center crop 的 uint8 Tensor
  -> 立即释放 decoder 和底层文件资源
```

不能通过把现有 LRU 容量设为零来模拟无缓存，因为该实现可能在返回 decoder 前就淘汰并关闭自身文件句柄。应实现明确的 per-episode decoder 生命周期。

选择 TorchCodec 而不是命令行 ffmpeg 的目的主要是：

- 避免每个样本创建 ffmpeg 子进程；
- 避免通过 pipe 将 RGB 字节重组为 NumPy frame 后再转为 Torch Tensor；
- 通过 `get_frames_in_range(..., step=...)` 只返回目标 FPS 需要的帧；
- 使用 `VideoDecoder(transforms=[Resize(...), CenterCrop(...)])` 在 decoder 内完成空间变换；
- 避免先向 Python 返回完整 1080p episode Tensor，再通过 `torch.nn.functional.interpolate` 缩放到 480p/256p。

TorchCodec 和命令行 ffmpeg 底层都使用 FFmpeg，因此不假定仅替换 API 就会获得数倍提升。主要收益预计来自更少的进程、pipe、对象转换和高分辨率中间 Tensor 开销；最终性能应通过实际数据 benchmark 验证。

Ascend 训练环境中 decoder 使用 `device="cpu"`。不在 DataLoader worker 内将完整视频搬到 NPU 做 resize。

### 3.5 Cosmos 分辨率桶

Cosmos 的分辨率桶定义在：

- `cosmos_framework/data/generator/utils.py:42`

原始 vision SFT 使用 `get_aspect_ratio()` 选择桶，并采用 cover resize 后 center crop：

- `cosmos_framework/data/generator/local_datasets/helper.py:173`
- `cosmos_framework/data/generator/local_datasets/sft_dataset.py:193`
- `cosmos_framework/data/generator/local_datasets/sft_dataset.py:274`

Action 路径的 `find_closest_target_size()` 可以复用其“选择最近桶”的思想，但 `ActionTransformPipeline` 默认采用 fit resize 加 reflection padding，与 vision SFT 的像素语义不同：

- `cosmos_framework/data/generator/action/transforms.py:43`
- `cosmos_framework/data/generator/action/transforms.py:428`

因此不应直接将完整 `ActionTransformPipeline` 用于本任务的视频空间预处理。

## 4. 对已有 feature 分支实现的判断

`feat/causal-pretrain-lerobot-v3` 中的 `sft_dataset_lerobot3.py` 可作为参考，它已经具备：

- 递归发现多个 LeRobot 根目录；
- 一个 episode 对应一个 metadata entry；
- TorchCodec 连续区间解码；
- 复用原 vision SFT 输出字段；
- 按关键字选择视频视角。

但不建议原样合入，原因包括：

1. 手工读取并解释 LeRobot metadata，没有充分复用 `LeRobotDatasetMetadata`。
2. caption 只读取 episodes 表中的自定义列，没有完整覆盖标准 task/task_index 路径。
3. 未覆盖标准 task/subtask 等 frame-level 标注；action 不属于本次实现范围。
4. 找不到目标视角时静默回退第一路视频，可能读错 camera。
5. 保留了 61 秒时长过滤，与完整 episode 需求冲突。
6. 复制了大段 `SFTDataset.process_one_sample()`，后续容易与主实现漂移。
7. resize 在完整高分辨率 episode 解码完成后通过 `interpolate` 执行，CPU 内存与拷贝成本偏高。

## 5. 推荐架构

建议拆成三层，而不是再实现一个单体 dataset。

```text
LeRobot root / parent directory
              |
              v
LeRobotEpisodeCatalog
  - discover roots
  - LeRobotDatasetMetadata
  - episode/task/video schema validation
  - one record per episode
              |
              v
LeRobotEpisodeReader
  - decode one complete episode video
  - read required episode parquet columns
  - align task/subtask/timestamps
              |
              v
VisionSFTSampleProcessor
  - FPS sampling
  - bucket resize + center crop
  - VAE temporal alignment
  - caption/tokenization
  - SequencePlan
  - standard SFT output
```

### 5.1 `LeRobotEpisodeCatalog`

职责：

- 判断输入路径自身是否为 LeRobot 数据集。
- 输入路径自身不是数据集时，优先检查一级子目录；只有一级子目录没有发现数据集时才递归查找所有 `meta/info.json`。
- 对解析后的绝对路径去重并按相对路径稳定排序。
- 为每个根构造 `LeRobotDatasetMetadata`。
- 枚举 episode，并记录 dataset root、episode id、数据行范围、视频范围、FPS、可用视角和 task 信息。
- UUID 必须包含数据集相对路径与 episode id，避免不同子数据集的 episode id 冲突。
- 初始化时汇总 schema 错误；不应让坏数据随机延迟到训练中才暴露。

旧分支的 `_discover_lerobot_roots()` 使用单线程 `Path.rglob("meta/info.json")`，支持任意深度的多个数据集，但它会从输入路径开始完整递归，并且后续通过普通 `for` 循环逐个加载每个数据集的 metadata。本实现保留以 `meta/info.json` 判断 LeRobot 根目录的思想，但将发现和 metadata 加载拆成两个阶段。

#### 根目录发现阶段

目录发现保持单线程，执行顺序为：

```text
root/meta/info.json 存在
  -> 将 root 作为唯一数据集

否则检查 root 的一级子目录
  -> 收集 child/meta/info.json 存在的所有 child

一级子目录没有找到数据集
  -> 使用 rglob("meta/info.json") 递归兜底

最后
  -> Path.resolve 去重
  -> 按相对路径稳定排序
```

目录遍历主要是文件系统 metadata I/O。对本地磁盘而言，多线程 `stat/readdir` 通常收益有限；在共享存储上大量并发扫描还可能加重 metadata server 压力。因此不对递归发现本身使用线程池。

配置入口同时允许显式提供多个 dataset roots。显式 roots 存在时直接进行校验和排序，完全跳过父目录扫描。

#### Metadata 加载阶段

根目录确定后，不同数据集的 metadata 加载彼此独立，适合线程池并行。复用现有 `_parallel_map()` 的行为，通过 `ThreadPoolExecutor.map()` 并行构造：

```python
LeRobotDatasetMetadata(
    repo_id="local",
    root=root,
    revision="local",
)
```

现有实现位置：

- `cosmos_framework/data/generator/action/datasets/cosmos3_action_lerobot.py:204`
- `cosmos_framework/data/generator/action/datasets/cosmos3_action_lerobot.py:603`

`executor.map()` 保持输入顺序，因此线程完成先后不会改变最终数据集和 episode 顺序。建议配置：

```python
metadata_load_workers: int = 8
```

实际线程数为：

```python
workers = max(1, min(metadata_load_workers, len(dataset_roots)))
```

不建议默认使用 64 个线程。Dataset 会在每个训练 rank 内构造，线程数还会乘以 rank 数；共享存储上的过高并发可能让启动更慢。默认最多 8 个线程，并允许配置为 1 以关闭并行加载。

完整 episode catalog 必须在 DataLoader workers 创建前构造完成。worker 只继承只读 catalog，不得再次执行目录发现、`LeRobotDatasetMetadata` 构造或全量 episodes/tasks metadata 扫描。第一版允许不同训练 rank 各自初始化一次 catalog；如果未来数千个子数据集导致启动成为瓶颈，再单独设计持久化 manifest 或 rank 间共享，不纳入当前实现。

### 5.2 `LeRobotEpisodeReader`

职责：

- 通过 episode 的 `dataset_from_index` 和 `dataset_to_index` 读取完整的结构化数据行。
- 只读取 timestamp、task_index 以及可选 subtask 等本次需要的 parquet 列，不读取 action 列。
- 根据所选视角解析该 episode 的视频文件和起止时间。
- 对完整 episode 做一次连续区间解码，避免逐帧 `LeRobotDataset.__getitem__()` 带来的 Python 调用和重复 seek。
- 每个 episode 独立创建 CPU TorchCodec decoder，完成读取后立即释放，不使用 LRU。
- 通过 TorchCodec decoder transforms 完成 resize 和 center crop，不生成完整 1080p Python/Torch 中间 Tensor。

结构化 parquet 应按 episode 所在文件和行范围读取；不能在每个 worker 中将所有数据转换成 Python dict。现有基础 dataset 明确指出，全量构造数千万帧的 Python dict 会消耗十几分钟及数十 GB 内存：

- `cosmos_framework/data/generator/action/datasets/base_dataset.py:78`

### 5.3 `VisionSFTSampleProcessor`

职责：

- 产生与当前 vision SFT 相同的数据类型和 tensor layout。
- 复用 caption dropout、metadata suffix、tokenization 和 `SequencePlan` 逻辑。
- 使用 Cosmos 既有分辨率桶。
- 保证视频与 timestamp、task/subtask 等时序字段使用同一组采样索引。

为了避免复制完整的 `process_one_sample()`，建议小幅重构 `SFTDataset`，提取可覆盖的 I/O hook：

```python
class SFTDataset:
    def _decode_video(self, metadata, frame_indices, target_size):
        ...  # 原 ffmpeg 路径

    def _load_sample_extras(self, metadata, frame_indices):
        return {}
```

LeRobot 子类只替换视频读取和 episode extras，分桶、caption、tokenization、`SequencePlan` 和标准返回字段仍由公共路径维护。

这里的 hook 只是普通的可覆盖方法，不是 PyTorch forward hook。

## 6. 多数据集行为

一个父目录下的所有合法数据集合并为一个逻辑 episode 列表：

```text
parent/
  dataset_a/meta/info.json  -> episodes A0 ... An
  dataset_b/meta/info.json  -> episodes B0 ... Bm
  dataset_c/meta/info.json  -> episodes C0 ... Ck
```

不同数据集允许拥有不同的：

- 原始分辨率；
- FPS；
- 视频文件模板；
- episode 长度；
- task 数量。

选择的视角必须都能解析。各数据集可以包含不同 action schema，但本实现不读取 action 列，因此 action 维度不参与数据集合并与校验。

## 7. 视角选择

第一版只读取一个视角，建议配置接口为：

```toml
video_view = "head"
video_view_aliases = ["head", "head_camera", "top"]
```

匹配顺序：

1. 完整 feature key 精确匹配；
2. feature key 最后一级名称匹配；
3. alias 大小写不敏感子串匹配；
4. 找不到则报错，并列出该数据集所有 `dtype=video` 字段。

不允许静默回退第一路视频。若不同子数据集使用完全不同的视角命名，后续可以支持按 dataset root 配置 alias，但第一版先采用全局 alias 列表。

## 8. Caption、task 与 subtask

建议区分“提供给模型的主 caption”和“保留时间结构的标注”。

主 caption 来源优先级：

1. 用户配置的 episode caption 字段存在且非空；
2. frame data 的 `task_index` 映射 `meta/tasks.parquet`；
3. 没有主 task 时，使用按时间去重后的 subtask 文本。

标准输出仍写入：

- `ai_caption`
- `sampled_caption_style`
- `text_token_ids`

如果一个 episode 内有多个 subtask，应保留为带时间或帧区间的结构化 metadata，而不是只取第一条。是否将多个 subtask 拼成主 caption 是尚待确认的训练语义。

## 9. FPS 与跨模态时间对齐

若配置最大 FPS，例如：

```toml
max_video_fps = 15.0
```

采用固定整数 stride：

```python
temporal_interval = max(
    existing_temporal_interval,
    math.ceil(original_fps / max_video_fps),
)
effective_fps = original_fps / temporal_interval
```

该配置的语义是“输出 FPS 不超过上限”，不是“精确重采样到指定 FPS”。例如：

```text
30 FPS -> stride 2 -> 15 FPS
25 FPS -> stride 2 -> 12.5 FPS
20 FPS -> stride 2 -> 10 FPS
15 FPS -> stride 1 -> 15 FPS
低于 15 FPS -> stride 1 -> 保持原生 FPS
```

这是预期行为。固定 stride 保持相邻采样帧的原始帧间距一致，不使用非均匀索引去逼近 15 FPS，也不重复帧进行上采样。

同一组整数 stride 采样选择必须作用于：

- 视频帧；
- task/subtask index；
- timestamp；
- 其他 frame-level annotations。

本次不读取 action。timestamp 和 task/subtask 等 frame-level 标注如果被读取，必须按照视频相同的 frame index/stride 选择；timestamp 用于验证实际时间顺序和记录有效 FPS，不用于构造非均匀重采样索引。

## 10. 分辨率分桶

目标语义保持当前 causal vision SFT：

```text
input H/W
  -> get_aspect_ratio / VIDEO_RES_SIZE_INFO
  -> target bucket
  -> aspect-preserving cover resize
  -> center crop to exact target W/H
```

使用 TorchCodec decoder transforms 将 `Resize((resize_h, resize_w))` 和 `CenterCrop((target_h, target_w))` 放入 decoder。禁止先解码并返回完整 1080p episode Tensor 后再统一 `F.interpolate`。桶选择、resize 尺寸和 crop 参数仍使用 causal vision SFT 的既有计算规则，保证输出语义一致。

## 11. 完整 Episode 与模型约束

“完整 episode”定义为：不做随机窗口和固定长度切片，读取 episode 的完整时间跨度。

仍存在两个必须明确的边界。

### 11.1 VAE 时间几何

当前视频 tokenizer 要求输入帧数满足：

```text
T = temporal_compression_factor * N + 1
```

当 compression factor 为 4 时，需要 `T = 4N + 1`。推荐统一裁掉尾部最多 3 个采样帧，并同步裁剪 timestamp 和 frame annotations。输出应记录：

- 原始 episode 帧数；
- FPS 采样后帧数；
- VAE 对齐后的实际模型输入帧数。

这意味着“完整”是完整读取 episode，但模型输入可能因 VAE 几何丢弃尾部最多 3 帧。

### 11.2 单个 Episode 超过 packing 上限

如果一个 episode 自身的估算 token 数超过 `max_sequence_length`，它无法作为完整样本进入 packing。

推荐策略是启动时或首次建立索引时标记为 oversized，训练时跳过并统计，而不是静默截断。是否允许为超长 episode 提供显式的 fallback 切分模式仍待用户决定。

## 12. 输出接口与 Action 边界

标准 vision SFT 字段保持不变，包括：

```python
{
    "__key__": ...,
    "__url__": ...,
    "fps": ...,
    "n_orig_video_frames": ...,
    "chunk_index": ...,
    "frame_start": ...,
    "frame_end": ...,
    "num_frames": ...,
    "video": ...,  # uint8 [3, T, H, W]
    "num_multiplier": ...,
    "conditioning_fps": ...,
    "padding_mask": ...,
    "image_size": ...,
    "ai_caption": ...,
    "sampled_caption_style": ...,
    "text_token_ids": ...,
    "sequence_plan": ...,
}
```

本次不读取 LeRobot action 列，也不在任何阶段构造或传递 action Tensor。最终样本不得包含 `action`、`action_raw`、`lerobot_action_raw` 等字段，因此：

- action 不占用 DataLoader worker 的 CPU 内存；
- action 不进入 CPU batch 或 NPU；
- action 不参与 packing token 估算；
- `SequencePlan.has_action` 保持 `False`；
- 不启用 action tokenizer、head 或 loss。

这也避免触发 `PackingDataLoader` 对标准 `action` 字段的 token 计数：

- `cosmos_framework/data/generator/joint_dataloader.py:455`

实现中只保留简短注释，标明如果未来扩展为 video/action 联合训练，需要从 episode 的相同行范围读取 action，并与视频采样索引对齐。未来扩展还必须另外设计：

- dataset-specific action spec；
- action normalization；
- action dimension padding；
- `SequencePlan.has_action`；
- action tokenizer/head/loss；
- video `T` 与 action `T` 或 `T-1` 的严格定义；
- 多数据集 action schema 的统一方式。

这属于 video/action 联合训练，不再只是 vision SFT 数据源适配。

## 13. 建议配置面

数据集工厂建议提供以下核心参数，具体 TOML schema 在实现阶段再根据最终 experiment 接入：

```python
get_lerobot_episode_sft_dataset(
    roots: str | list[str],
    metadata_load_workers: int = 8,
    video_view: str = "head",
    video_view_aliases: list[str] | None = None,
    resolution: str = "480",
    max_video_fps: float | None = 15.0,
    caption_key: str | None = None,
    temporal_compression_factor: int = 4,
)
```

`roots` 同时接受一个父目录或显式目录列表。若只保留一个配置字段，也可以规定字符串路径自动进行单根/父目录发现。

## 14. 推荐实施顺序

1. 新增分级 LeRobot root 发现和视频 key 选择 helper；视频路径解析直接复制现有 `_video_path()` 的已验证逻辑，不做公共抽象。
2. 实现 `LeRobotEpisodeCatalog`：根发现保持单线程，各 root 的 `LeRobotDatasetMetadata` 使用最多 8 个线程并行加载，只建立 metadata 和 episode 索引，不解码视频。
3. 实现 episode parquet 必要列的范围读取，验证 task/subtask/timestamp 对齐；明确排除 action 列，仅保留未来扩展注释。
4. 实现无 LRU 的 per-episode CPU TorchCodec 连续解码，并在 decoder transforms 内完成 resize 和 center crop。
5. 对 `SFTDataset` 做最小 hook 重构，保持原有 JSONL/S3 路径行为不变。
6. 实现 LeRobot episode 子类，仅覆盖 I/O 与 extras。
7. 接入 FPS 采样、VAE 时间对齐和 Cosmos 分桶。
8. 新增独立 dataset factory；之后再由选定的 experiment/TOML 引用。
9. 增加启动前 dataset scan/validation，输出数据集数、episode 数、视角、FPS、长度和 oversized 统计。
10. 使用小型多根 LeRobot fixture 验证完整 episode、多视角选择、task 映射、时间对齐和原 SFT 接口兼容性。

## 15. 尚待讨论的决策

1. **Subtask 语义**：主 caption 使用 episode task，还是将 subtask 按时间拼接进 caption？
2. **FPS 默认值**：整数 stride 算法已确定；默认上限使用 15 FPS，还是默认保持原生 FPS、由配置显式设置上限？
3. **VAE 尾帧**：是否接受为了 `4N+1` 丢弃尾部最多 3 帧？
4. **超长 episode**：推荐跳过并记录；是否需要显式允许切分的 fallback？
5. **缺少 head 视角**：推荐整数据集报错；是否允许通过 alias 匹配 top 等近似视角？
6. **数据集混合权重**：多个子数据集按 episode 等概率混合，还是允许配置 dataset-level sampling weight？

本次不得接入 action pipeline；其余未决项确认后再完成最终数据与配置接入，避免把 vision-only 数据适配错误扩大为联合 action 训练改造。
