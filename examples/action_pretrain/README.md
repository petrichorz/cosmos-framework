# Causal action mid-training（方案 C：视觉历史）

本分支从 main 的独立首帧布局接入 action mid-training，保留方案 C：历史只读取视觉，
state 仅供所属当前预测块使用。训练入口为
`action_causal_midtrain_edge`，不是单一数据集微调接口。默认用 DROID 调试；
`CausalActionMixture` 在每个 rank 上按样本混合多个来源，然后按 `max_tokens` 打包。
纯视频仍使用独立配方。

## 表示与来源

- 输入/输出动作统一为 OpenWAM 80D；每块一个独立 state token。state 不进入 action encoder、decoder 或预测 loss。
- `state2llm`、`action2llm`、`llm2action` 分别共享于所有机器人。`domain_id` 只保留作元数据。
- 位置、夹爪、手指关节相对整块起点；旋转使用 `R_b.T @ R_target`，rot6d 为矩阵前两列依次拼接。
- State 与 delta action 分别使用训练集 q01/q99，所有有效 rot6d 通道也参与统计。无效通道不参与统计或监督；L0 只有视觉，不构造 action padding 或 state token。
- `B` 在数据侧均匀采样，随后选择 state、计算 delta、归一化、计数和 packing。模型复用该 geometry。
- `history=H` 表示最多读取前 H 个历史视觉块（含窗口内的独立首帧 L0），要求 H>=1；H=1 时只读取前一个视觉块。历史不包含 action/state，干净视觉也不吸收它们。
- state Q 只读文本和自身，仅所属块的条件/目标可读取它。FD 保留当前动作条件，ID 保留当前视频条件。

DROID 适配器读取 `observation.state.cartesian_position` 的绝对 XYZ/Euler-XYZ，端点沿用源 reader
声明的 Panda link8、参考系为 robot base。Pose 目标来自下一时刻实测轨迹，夹爪目标来自
`action.gripper_position`；实测夹爪来自 `observation.state.gripper_position`。夹爪统一为
`1 - raw`（0 闭、1 开），单臂放左侧 `0:10`。这不是 FK 推算的命令 EEF 目标；不能把该训练目标解释为控制器命令。

## 生成训练统计

在 py312 环境、仓库根目录运行；所有路径可替换。统计文件不可沿用绝对 action 的旧统计。

```bash
mkdir -p outputs/action_stats
python -m cosmos_framework.data.generator.action.block_statistics \
  --root /mnt/sfs_turbo/public/datasets/Cosmos3-DROID/success \
  --output outputs/action_stats/droid_b1_4_s1.json \
  --block-size-min 1 --block-size-max 4 --video-stride 1 \
  --chunk-length 32 --windows 2048 --seed 123
```

统计绑定来源、版本、train split、划分 seed/比例、坐标系、端点、映射版本、delta 定义、
窗口长度、B 分布、stride 和分块布局版本。变更这些参数应重新采集。JSON 的 `population` 保存实际窗口索引，
默认均匀采样窗口和 B；`--windows` 指明估计统计的训练窗口数。固定 B=2 实验使用
`--block-size-min 2 --block-size-max 2` 单独生成统计。生产训练应评估足够覆盖来源的统计样本量。

## 启动

```bash
export ACTION_STATISTICS_PATH="$PWD/outputs/action_stats/droid_b1_4_s1.json"
export OUTPUT_ROOT="$PWD/outputs/action_midtrain"
bash examples/action_pretrain/launch_midtrain_action_causal.sh
```

上述 launcher 假定当前环境已经安装 torch_npu/CANN。需要同时配置运行环境时，复制并修改
`launch_midtrain_template.sh` 中的环境路径。默认 TOML 已指向通用 mid-training 入口：

```bash
examples/action_pretrain/action_midtrain_edge_causal_tnd.toml
```

关键 overrides：

```text
dataloader_train.max_sequence_length=16384
model.config.teacher_forcing_block_size_min=1
model.config.teacher_forcing_block_size_max=4
model.config.teacher_forcing_history_blocks_min=1
model.config.teacher_forcing_history_blocks_max=8
model.config.activation_checkpointing.mode=selective
```

默认 joint 比例为 FD:ID:Policy=1:1:1，通过每个来源的 `joint_mode_weights` 调整。
`MODE=policy/forward_dynamics/inverse_dynamics/joint` 修改默认调试来源；增加来源后，分别配置各来源的 mode/比例。
训练视频的固定抽帧步长由 TOML 的 `[action].video_stride` 配置，可取 1、2 或 4；它不复用
video pre-training 的随机 multi-FPS 配置。统计文件必须使用相同的 `video_stride` 生成。

固定窗口验收还需设置来源 `debug_fixed_index=0`、`cfg_dropout_rate=0`、固定 B，以及
`model.config.causal_action_debug_noise_seed=123`。此调试 seed 固定 sigma 和噪声，并恢复外部 RNG；
正式随机窗口训练必须保持其为 `null`。不要把固定窗口和随机 joint 曲线合并解释。

Edge-DCP 加载仍走原 DCP 流程，只在 warm start 初始化新的 state/action 接口，打印跳过的 key。
自动恢复本次训练 checkpoint 时，原 DCP 逻辑会恢复这些权重；不提供旧 domain action checkpoint 语义兼容。

## 增加数据源

适配器返回以下原始字段：

- `video`: `[3,N+1,H,W]`，`conditioning_fps`: 同步网格帧率。
- `state_trajectory`、`state_mask`: `[S,80]` 实测绝对状态与独立布尔 mask。
- `action_target`、`action_mask`: `[N,80]` 绝对目标与独立布尔 mask。
- `action_state_indexes`: `[N]`，动作开始前对应的实测状态下标；不允许未来 state。
- `state_timestamps`、`action_timestamps`: 秒，和上述下标相匹配；动作须落在同步视频区间网格上。
- `source_contract`: `SourceContract`，声明机器人、参考系、物理端点、左右映射、单位、夹爪方向和目标来源语义。
- 现有 transform 所需的 caption、viewpoint、domain 等元数据。

用 `scatter_openwam_fields` 把不同原始维数映射到物理字段；机械臂关节必须先做可靠 FK，
不能直接放入灵巧手槽位。速度-only 底盘显式拒绝，需要先决定独立的物理契约。

构建 `ActionSFTDataset(raw_reader, ActionTransformPipeline(max_action_dim=80, ...), resolution)`，
再用 `CausalActionSFTDataset(base, statistics_path=..., mode="joint", ...)` 包装。
通过通用 `collect_statistics(dataset, indices=..., block_sizes=..., video_stride=...)` 为每个契约生成独立统计。
加入 mid-training 配方的 `robots.dataset.datasets`，设置相应 `weights`。
各来源必须是 map-style（`iterable_shuffle=False`），mixture 统一负责分布式/worker episode 分片。
每个来源的 episode 数至少覆盖 rank × worker 数，避免空分片挂起。

默认配方只注册已核对契约的 DROID。现有 `agibot_processed/canonical_55d` 未附与 80D 对应的完整
转换契约，且原统计含 `actions.robot.velocity`；目前不自动接入，不丢弃或猜测其控制字段。
不同原始维数、左右槽位和手指 mask 的共享路径另有合成多源测试。

## 推理和在线预览

```bash
export CHECKPOINT_ROOT=/path/to/new/action/checkpoint
export ACTION_STATISTICS_PATH=/path/to/droid_b2_s1.json
BLOCK_SIZE=2 bash examples/action_pretrain/inference_causal_action.sh
```

默认只生成当前预测块；缺下一块实测 state 时停在边界。所有 B 均默认 `current-block=0`，
即 L0 后第一组 B 个 latent 的真实动作。B=1/2、stride=1 时分别输出 4/8 个动作。
`--current-block` 使用不含视觉前缀的预测块编号。
历史视频必须显式标为已观测；历史 action/state 可使用零占位，不需要提供历史测量或预测动作。
离线 CLI 的 `--current-block` 会把该块之前的数据集视频标为已观测条件。

离线多块预览必须显式传 `--preview-ground-truth-states`，产物标记为“真值 state 条件预览”。
默认 mid-training 配方每 5000 iter 也使用这个预览语义，保留样本真实 state 和历史；不称为自由闭环 rollout。

执行接口使用 `CausalActionSession.generate_current_block` 与 `commit_execution_feedback`。
`CausalActionSession.for_model` 可连接 Cosmos 和机器人侧 `batch_builder`；后者负责图像/实际历史的
时间对齐与输入打包，并保留当前实测 state、mask、原块 anchor 和训练统计。
调用 `commit_execution_feedback(observation=观测视频块, executed_steps=完整块动作数)` 确认执行。
第一版只允许完整块提交，部分执行会报错；历史只保存观测视频，不再要求回传 action 或旧 state。
下一次规划必须传入新的当前 state 和图像。
每次调用创建独立、CFG 隔离的块内缓存，不承诺持续滚动 KV 加速。

`real_actions` 仅导出本次生成的真实动作：先反归一化 delta，再用原所属块 state 还原绝对目标。
执行适配器通过显式槽位 mask gather，返回机器人控制接口所需格式。
