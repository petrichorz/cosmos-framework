# 多数据源 Causal Action：设计与实现

更新：2026-09-28。本文整理已实现的设计、关键演变和验证边界，保留原文件名作为统一入口。实际启动命令见 [action_pretrain 使用说明](../examples/action_pretrain/README.md)，实验数据和性能结果见文末链接。

## 1. 目标与整体分工

在同一 causal 模型中接入处理后的 AgiBot、EgoSuite：复用视频生成主干，以实测 state 作为条件，联合学习视频和 action。各来源可以有不同的有效通道、相机和目标来源，但必须遵守共同的模板与 block 时序。

```text
LeRobot v3 子数据集
  → SegmentLeRobotDataset：按片段读取绝对 state/target、mask、视频
  → ActionStateTemplate：按字段编码 block 目标
  → BlockStatistics：分别归一化 state/action
  → CausalActionSFTDataset：选择 FD / ID / Policy
  → DatasetGroup / Mixture：组内索引、组间采样
  → packing：位置编码、样本隔离、causal attention
  → 模型：视频与 action 的 flow matching
```

职责保持分离：

- **模板**定义通道布局、有效性、编码与解码，不选择数据时间点或切片起点。
- **读取选项**定义 state/action 字段、目标来源和时间偏移，不决定相对编码方式。
- **geometry**定义 action、视频与 latent 的对应关系；**planner**负责切片、overlap 和尾段。
- **来源契约**声明坐标系、单位、末端、目标语义；共同的 55D 排布不意味着物理语义自动一致。
- Dataset 和模型接口不硬编码 55D 槽位；具体槽位只由模板解释。DROID 不在本轮新路径的接入范围，原有非 causal Reader 保留。

## 2. state 与 action 分别是什么

### 2.1 三层数值表示

| 名称               | 含义                                                    | 形状        |
| ------------------ | ------------------------------------------------------- | ----------- |
| 原始 state 轨迹    | 对齐时间网格上的实测绝对状态                            | `[T+1, D]`  |
| 原始 action target | 按来源配置读取的绝对目标，尚未做 block 编码             | `[T, D]`    |
| 模型 state         | 每个 block 起点的实测 state，经规范化、归一化后作为条件 | `[T/32, D]` |
| 模型 action        | 原始 target 按字段编码，再归一化后的训练目标            | `[T, D]`    |

模型 state 不是待预测的下一状态，也不是整个片段唯一的初始状态。每进入一个新 block，就使用该 block 的实测起点作为新 anchor。

模型 action 也不是统一的“相邻帧差分”：相对字段在同一 block 内都相对于同一个 anchor；夹爪和灵巧手则始终保留绝对目标。

### 2.2 当前两类来源

| 来源     | state 字段         | 绝对 target                  | target mask   |
| -------- | ------------------ | ---------------------------- | ------------- |
| AgiBot   | `state_unified[t]` | `state_unified[t+1]`         | `mask_state`  |
| EgoSuite | `state_unified[t]` | 已处理的 `action_unified[t]` | `mask_action` |

AgiBot 配置为 `action_from_state=true, action_time_offset_steps=1`；EgoSuite 为 `false, 0`，不能再次移位。切换为 state 来源不会自动增加偏移，二者必须分别配置。`--profile` 只声明来源语义，不自动选择 target。

因此，AgiBot 当前监督的是下一实测状态轨迹，并不能直接等同于机器人的底层控制指令。部署时还需要执行适配层。

代码：[读取选项](../cosmos_framework/data/generator/action/sample_contract.py)、[AgiBot 配置](../examples/action_pretrain/sources/agibot.json)、[EgoSuite 配置](../examples/action_pretrain/sources/egosuite.json)。

### 2.3 当前 55D 模板布局

模板版本：`unified55-xyzw-absolute-gripper-hand-v4`。下表使用 Python 左闭右开切片；state 和 target 共用布局，但可使用不同 mask。

| 切片              | 字段                    | action 编码                               |
| ----------------- | ----------------------- | ----------------------------------------- |
| `0:7` / `7:14`    | 左 / 右臂关节           | target − anchor                           |
| `14:17` / `21:24` | 左 / 右末端位置         | target − anchor                           |
| `17:21` / `24:28` | 左 / 右末端四元数，xyzw | 相对旋转                                  |
| `28:29` / `29:30` | 左 / 右夹爪开合度       | 绝对 target                               |
| `30:36` / `36:42` | 左 / 右灵巧手           | 绝对 target                               |
| `42:46`           | waist                   | target − anchor，分量含义由来源声明       |
| `46:48`           | head                    | target − anchor，分量含义由来源声明       |
| `48:51`           | mobile                  | target − anchor，必须声明为可作差的绝对量 |
| `51:55`           | reserved                | 必须无效                                  |

不能把速度指令直接当作绝对位姿相减；当前模板也不隐式做角度 wrap、单位换算、FK 或坐标系转换。

对 block 起点 `b`、块内偏移 `k=0…31`：

```text
state 条件       = state[b]
相对标量 action  = target[b+k] − state[b]
夹爪/灵巧手      = target[b+k]
相对旋转         = inverse(q_state[b]) × q_target[b+k]
```

解码时，相对标量加回同一 anchor，旋转使用 `q_anchor × q_delta`；夹爪和灵巧手不加 anchor。有效四元数先单位化并统一符号：通常令 w 非负，w 为零时使用确定性的分量符号规则，消除 q 与 −q 的歧义。

每个子数据集缓存两份 `[D]` mask，广播到时间维。无效值清零、不进入统计；有效值必须有限。四元数组必须整组有效或无效。相对 action 必须有对应有效 state，绝对夹爪/灵巧手不要求同维 state 有效。

代码：[ActionStateTemplate55](../cosmos_framework/data/generator/action/action_state_template.py)。

## 3. 片段与 block 时序

当前 `actions_per_block=32`、`video_stride=4`、VAE 时间压缩倍率为 4。因此一个预测 block 对应 **32 action → 8 个后续视频帧 → 2 个后续 latent**。

首个 latent `L0` 是独立视觉条件，不为它补虚拟 action。含 K 个 block 的片段具有：

```text
原始 observation：32K+1
原始 action：     32K
送入 VAE 的视频： 8K+1
视频 latent：     2K+1
state token：     K
```

| 预测 block | anchor      | action 区间 | 预测 latent | state 绑定的视觉边界 |
| ---------- | ----------- | ----------- | ----------- | -------------------- |
| 0          | `state[0]`  | `[0,32)`    | L1、L2      | L0                   |
| 1          | `state[32]` | `[32,64)`   | L3、L4      | L2                   |
| 2          | `state[64]` | `[64,96)`   | L5、L6      | L4                   |

每个 action 时间步对应一个 D 维 action token，不是每维单独一个 token；每个 block 的 D 维 state 投影为一个 state token。每个视觉 latent 则包含空间网格上的多个 token。

**片段比 block 更长。** 当前训练示例最多 896 action、overlap=161，即最多 28 个 block、897 个原始 observation、225 个 VAE 输入帧；这些是配方参数，不是模板定义。

切片规则：

- 起点间隔为 `L−O`；L 必须兼容完整 block，O 和起点无需对齐到 32。
- 长 episode 的最后一段右对齐到有效末端，保持长度 L，避免重复生成已有范围。
- 短 episode 保留开头的最大完整 block 数；不足 33 个 observation 时 warning 并跳过，不 padding。
- 先扣除目标偏移导致的不可读边界，再切片；不跨 episode。
- 每个片段独立抽视频帧、划 block、选 anchor、重置历史。overlap 正常参与监督，不额外去重。

例如 L=96、O=20、有效 action 数为 210，起点为 `0、76、114`。同一绝对目标出现在不同片段时，可以因 anchor 不同而得到不同的相对 action。

代码：[geometry](../cosmos_framework/data/generator/action/causal_block_geometry.py)、[planner](../cosmos_framework/data/generator/action/segment_planner.py)、[block 编码](../cosmos_framework/data/generator/action/block_state.py)。

## 4. mRoPE：对齐时间，不混淆序列位置

### 4.1 三轴与 action 时间

mRoPE 使用 `(time, height, width)` 三个坐标。文本三轴同步递增；视频使用时空网格；action 使用 `1×1` 空间网格。token 在 packed 序列中的先后位置，不等于物理时间先后。

`action_frame_ids[j] = 1 + floor(j/16)` 表示 action 属于哪个后续 latent，用于 block 分组和相关调度；它不是逐 action 的最终 mRoPE 时间。前 16 个 action 属于 L1，后 16 个属于 L2，但各时间步仍有不同的时间坐标。

关闭 FPS modulation 时，忽略共同前缀偏移 `t0`：

```text
视频 L0,L1,L2,L3,L4,…：0, 1, 2, 3, 4,…
action j：             (j+1)/16
各 block 的 state：    0, 2, 4,…
```

action[0] 位于 `1/16`，action[31] 位于 2。这里按真实 action 区间的后续位置对齐；Reader 的 action timestamp 仍标记区间起点，不应据此把目标再移一帧。

### 4.2 state 直接绑定视觉边界

state 的时间坐标直接复制本 block 前一个视频边界 latent 的最终坐标：首块复制 L0，第二块复制 L2，依次类推。空间坐标沿用 action 的位置约定，不把 state 当作图像 patch。

早期实现经过“state timestamp → 原始步数 → action 间隔 → mRoPE”的间接换算。现在由 geometry 给出边界索引，packing 直接复制视频时间坐标，同时保留 FPS 缩放和文本前缀偏移。这避免重复换算和不同模态漂移。

### 4.3 存储 FPS 与训练 FPS 分开

- `meta.fps` 和存储 timestamp 用于 parquet/video 读取与连续性校验。
- episode 声明 `source_fps` 时用它表达训练时间尺度；缺失则回退 `meta.fps`。
- 视频抽帧后传入 `video_fps=source_fps/4`；action 仍为 `action_fps=source_fps`。
- 不再提前除 VAE 压缩倍率；mRoPE 内部负责 latent 时间换算。前提是每行对应一个均匀原始时间步，不能无条件套用到补帧或不规则重采样数据。

开启 FPS modulation，设训练 FPS 为 f、参考 `base_fps=24`、参考压缩倍率为 4，则参考时间速率为 6。当前标准 geometry 下：

```text
time(video latent l) = t0 + (16l / f) × 6
time(action j)       = t0 + ((j+1) / f) × 6
time(state block b)  = time(video latent 2b)
```

例如 f=30 时，视频时间依次为 `t0+0、3.2、6.4…`，action 间隔为 0.2；state 依旧精确落在对应视觉边界。base_fps 是模型参考尺度，不随每个来源修改。

代码：[mRoPE](../cosmos_framework/data/generator/sequence_packing/mrope.py)、[action 时间修正](../cosmos_framework/model/generator/omni_mot_causal_action_model.py)、[state 位置绑定](../cosmos_framework/data/generator/sequence_packing/causal_action.py)、[位置与隔离测试](../cosmos_framework/data/generator/sequence_packing/causal_action_position_test.py)。

## 5. 训练任务与注意力边界

| 任务                   | 当前 block 的视觉 | 当前 block 的 action |
| ---------------------- | ----------------- | -------------------- |
| Forward Dynamics（FD） | 预测              | 干净条件             |
| Inverse Dynamics（ID） | 干净条件          | 预测                 |
| Policy                 | 预测              | 预测                 |

三种任务都使用文本、窗口内干净视觉历史和当前实测 state。`joint` 按任务权重选择任务，默认等权；任务权重与 AgiBot/EgoSuite 来源权重独立。Policy 在这里是视频与 action 联合预测，不是仅输出 action。

为防止 teacher forcing 信息泄漏，序列区分以下角色：

| 角色             | 可见内容                                                               |
| ---------------- | ---------------------------------------------------------------------- |
| 干净视觉历史分支 | 本样本文本、窗口内更早的干净视觉、同 block 干净视觉；不读 state/action |
| 当前条件 token   | 文本、窗口内视觉历史、当前 state、当前条件；不读当前预测目标           |
| 当前预测 token   | 文本、窗口内视觉历史、当前 state、当前 block 条件与预测 token          |
| 当前 state token | 文本与自身；不读视觉、action 或其他 block 的 state                     |

历史只保留视觉，不将历史 action/state 作为后续 block 的条件。state 的隐藏表示可随网络层变换，但不能从其他模态吸收信息，因此不能成为泄漏未来信息的桥梁。

当前 block 内允许联合注意力，跨 block 按窗口限制；不同 packed 样本始终隔离。生产路径使用 grouped TND 表达可见关系，dense mask 作为小规模正确性参考，避免构造完整的大型稠密 mask。

state 通过独立 `state2llm` 和模态 embedding 接入；action 使用共享的 `action2llm/llm2action`，各来源不单独分配投影头。state 和干净条件使用零扩散时间；state 不作为独立的生成输出头。

代码：[注意力布局](../cosmos_framework/data/generator/sequence_packing/causal_action.py)、[TND 分组](../cosmos_framework/model/generator/mot/causal_action_tnd.py)、[模型接口](../cosmos_framework/model/generator/mot/causal_action_network.py)。

## 6. 统计、归一化与 clipping

### 6.1 统计总体

state 统计 block 起点状态，action 统计模板编码后的目标；二者分别保存。统计窗口固定为 32 action、起点 stride=1，遍历 episode 内全部合法起点；每窗口贡献 1 个 state 和 32 个 action。

这覆盖训练 block 的编码方式，但与长片段及 overlap 的训练采样频率不完全相同。不能用统计文件的 block 数替代训练片段数或直接推导训练来源权重。

当前推荐流程：每个 LeRobot v3 子集用独立进程计算，进程内多线程读数值和编码；使用 `--method exact`，不解码视频。结果写入子集 `meta/causal_action_stats.json`，然后生成指定位置的组统计。exact 保留全部有效编码值，内存随数据量和并行进程数增长；CLI 默认仍为 reservoir，因此 exact 必须显式传入。

聚合按通道有效 mask 进行：

- min/max、count/valid_counts、mean/std 合并为全量样本统计；std 使用总体分母 N，并包含子集间均值差。
- q01 取各有效子集的最小 q01，q99 取最大 q99。它们是覆盖子集的边界汇总，**不是合并样本的总体分位数**，也不是分位数加权平均。
- 新并行流程的分位数字段只保存 q01/q99。旧单进程总体统计入口仍保留其他分位数，以兼容既有用途。
- `--limit` 是每个子集的调试窗口上限；正式统计不设置它，被截断的结果标记 `partial=true`。

### 6.2 边界与归一化

`low/high` 与实测统计字段分开：普通通道可选 q01/q99 或 min/max；有效四元数固定为 `[-1,1]`，不会用这个理论范围覆盖实测 min/max/q01/q99。

```text
scaled = 2 × (value − low) / (high − low) − 1
模型输入/目标 = clamp(scaled, −1.5, 1.5)
```

low/high 仍映射到 ±1。放宽的是截断阈值，不是重新缩放整个分布。无效通道和 low==high 的通道输出 0。无需因截断范围变化重算统计，但训练与推理应使用一致的归一化代码。

`clipping.state/action` 分别记录每个通道 `|scaled|>1.5` 的有效值比例；不是数值本身，也不是梯度裁剪。state 的分母是 anchor 数，action 的分母是时间步数。0 可能表示未越界、通道无效或退化边界；不能单凭 0 判断通道是否有效。历史使用 ±1 阈值的日志不能直接与当前裁剪率比较。

代码：[统计计算](../tools/compute_causal_action_stats.py)、[并行入口](../tools/compute_causal_action_stats_parallel.py)、[聚合](../tools/aggregate_causal_action_stats.py)、[归一化](../cosmos_framework/data/generator/action/block_state.py)。

## 7. 多源训练与推理边界

每个来源组递归发现子数据集，组内按片段建立索引，共享一份组统计；组间按来源清单的 weight 抽样。希望每个训练片段近似等机会时，使用真实训练 planner 计算各组片段数作为权重。mask 允许不同，模板版本与宽度必须相同。

当前实现按 rank×worker 分配片段；每个来源至少需要足够的片段覆盖全部 shard，过小来源会报错。不能直接将单源增加 worker 的收益推广到小来源混合训练。

推理遵循闭环：输入当前实测 state/image → 生成当前 block → 执行完整 block → 提交视觉反馈 → 使用新实测 state 规划下一块。session 不保存历史 action/state，只保留有限视觉历史。每次新请求使旧请求缓存失效；预测视频不能直接代替真实执行反馈。

接口和离线入口已接入，但完整模型推理、有限窗口移动后的坐标与 KV cache 一致性仍需端到端验证。训练使用各 block 实测 state，不代表已支持无反馈的自由 rollout。

代码：[组与来源工厂](../cosmos_framework/data/generator/action/datasets/causal_action_factory.py)、[混合采样](../cosmos_framework/data/generator/action/datasets/causal_action_mixture.py)、[推理 session](../cosmos_framework/inference/causal_action/session.py)。

## 8. 关键设计演变

| 阶段                 | 决定及原因                                                                                                   |
| -------------------- | ------------------------------------------------------------------------------------------------------------ |
| 模板与来源解耦       | 将槽位、绝对/相对字段、旋转编码放入模板；来源仅处理读取、坐标声明和视频差异，避免在 Dataset 中散布机器人判断 |
| 明确目标读取         | 不做全局 state 重标；AgiBot 显式取 next state，EgoSuite 保留已处理 action，防止重复移位                      |
| 片段独立、block 固定 | 用统一 geometry 协调切片和模型时序；保留真实尾段、不 padding，每个 block 使用自身 anchor                     |
| state 绑定视觉边界   | 移除 timestamp 到位置坐标的间接换算，直接复用视频最终 mRoPE 时间，保留 FPS 调制与前缀偏移                    |
| 四元数规范化         | state/target 共用单位化和确定性符号，模板升至 v4；归一化边界使用理论范围，实测统计保留                       |
| 从单数据集到来源组   | 父目录递归发现多个 v3 子集，组内共享统计、组间独立配置读取规则与采样权重                                     |
| 统计并行与边界汇总   | 初期直接合并样本统计；为适应大量子集，新增子集多进程与独立聚合，明确 q01/q99 改为子集边界汇总                |
| 放宽截断             | 从 ±1 改为 ±1.5，保留更多尾部幅度，clipping 阈值同步修改；不据此宣称训练质量提升                             |

原 C01～C08b 的基础接口、Reader、模板编码、模型接入及来源组功能已落地。提交编号和当时的“待推送”状态不再作为本文主线，当前代码与 Git 历史为准。

## 9. 验证结论与未完成项

已覆盖模板编码/解码、切片边界、mask、mRoPE state 对齐、样本隔离、统计聚合及推理 session 的针对性检查。AgiBot 正确的 next-state 路径已做串行/多线程 exact 对照，896-action 片段的 28 个 block 与 stride-one 统计编码一致。已有单源、混合短程训练和续训记录，但不能将历史 mock 或 reservoir 实验当作新 exact 聚合配置的完整验收。

仍需保留的限制：

1. 训练统计加载校验目前主要覆盖维度、有效 mask 和 mock 标记，尚未完整核对来源读取偏移、geometry、模板版本与 partial；使用者必须确保统计与配置匹配。
2. low==high 的通道被置零且不计 clipping，可能掩盖稀少事件；更换分位边界不能自动解决这一问题。
3. 全模型闭环推理、滑窗 cache 行为和机器人执行质量仍未完成统一验收；短程 loss 有限不代表策略收敛。
4. 数据读取性能优化是独立后续任务，不混入本文的表示与统计语义。

独立记录：

- [C08 多源短训报告](c08_multisource_training_report_zh.md)：早期模拟统计下的训练与续训证据。
- [长片段组统计与训练实验](group_long_training_experiment_20260927_zh.md)：后续真实统计、时间戳修复和混合训练记录。
- [训练 review](action_training_review_20260927_zh.md)：历史问题与设计疑点，具体条目需结合当前代码判断。
- [性能分析](action_training_profiling_20260927_zh.md)、[数据读取优化计划](action_dataloader_optimization_plan_20260928_zh.md)：实测瓶颈与后续实施范围。

## 附录 A：原 C01～C09 执行步骤与阶段记录

以下保留原实施计划的执行顺序、范围、依赖、验收要求及阶段 review 记录，包括 C01b、C07a、C08b 等补充步骤。它们用于追溯设计与落地过程；其中的“当前”“待提交”“尚未验证”均指当时阶段，不代表最新分支状态。现行设计与已知限制以正文为准。

P0 为语义和索引基础，优先完成；P1 为单源端到端数据路径；P2 为混合、模型和配方接入。编号是建议落地顺序，不代表已经创建 commit。

每个 commit 应包含本功能必要的校验、针对性测试和直接相关文档。不要先提交破坏调用方的接口，再在后续 commit 修复；涉及共同签名的最小调用方调整应放在同一个 commit。未接通的新增能力暂不注册到默认训练入口。

### C01 / P0：模板规则与公共样本契约

建议标题：`feat(action): define template-driven sample contracts`

- 范围：完善 ActionStateTemplate 的接口和具体模板规则；补齐灵巧手绝对值处理；定义读取选项和公共原始样本契约。
- 边界：不读真实视频、不引入切片算法、不改变旧训练配方。字段语义只在模板中表达。
- 依赖：无；具体旋转规则在此项完成前明确。
- 验收：编码/解码保持夹爪和灵巧手绝对值；有效字段 round-trip 正确；模板维度和 mask 不匹配明确报错；更换一个简单模板时公共接口无需修改。

已 review 的 C01 实现（13 项 CPU 检查通过）：

- `action_state_template.py`：夹爪与灵巧手均保留绝对目标，模板版本升级，防止旧语义被误认为兼容；用户确认保留 xyzw 相对四元数，编码为 inverse(anchor) × target，解码为 anchor × delta。
- `sample_contract.py`：ActionReadOptions 定义来源读取选项，默认 action 列、零偏移；原 RawActionSample 包装已在 C03/C04 review 中删除，改为训练字典；后续 review 又删除逐样本统一校验函数，校验保留在各自负责约束的环节，宽度从注入的模板取得。
- state/action mask 分别为每个 LeRobot v3 子数据集共享的 `[D]` 向量；ActionReadOptions 提供 state_mask_key/action_mask_key，默认 mask_state/mask_action。每个子数据集加载、校验并缓存一次，不逐帧读取或检查一致性；若存储于 parquet 行中，取一条有效记录作为声明即可。真实读取与按子数据集缓存已在 C03 实现。使用 state 构造目标时，目标 mask 来自 state，不能沿用原始 action mask。
- 实际样本为 `[T+1,D]` state 和 `[T,D]` 绝对目标，视频可选。时间戳表示对齐网格，action 时间戳为区间起点，不等同于物理目标时刻。读取来源和目标偏移只声明，不在 C01 执行。
- C01 不检查 32 步 block、不读取数据、不修改训练配方。来源契约辅助方法不再默认声称目标来自下一状态，调用时必须显式提供 target_semantics。既有来源辅助和模拟统计方法暂保留，后续对应 commit 再处理。
- `sample_contract_test.py` 使用 CPU 合成数据检查绝对字段、相对旋转、无效 mask/模板版本、读取选项、时间对齐以及另一种通道宽度的模板；不依赖真实视频和模型权重。

### C01b / P0：前置公共 causal 时序契约

建议标题：`feat(action): define shared causal block geometry`

- 范围：从原 C06 中前置抽取 block 时序定义与校验，作为切片、读取、编码、统计及模型配置共同使用的唯一配置来源。它独立于 ActionStateTemplate，后者只负责通道和字段语义。
- 当前配置：`actions_per_block=32`、`video_stride=4`、VAE 时间压缩倍率为 4；推导每个预测 block 为 2 个 latent frame。geometry 根据这三个参数推导长度。最新决定：实际 VAE 时间压缩倍率固定为 4，C07 不增加模型与 geometry 的倍率同步或额外校验。
- 提供：合法 observation 长度规则 `A*x+1`、最低帧数 `A+1`、短范围最大合法长度的计算，以及 block 到 action/latent 的映射规则，其中 A 为 actions_per_block；提供视频抽帧后的 FPS 换算与有效性校验。
- 边界：只定义几何及合法性，不读取数据，不决定 overlap/尾段策略，不计算 action delta，也不加载统计量；不需要把整个 C06 提前。
- 依赖：无数据来源依赖，可与 C01 分别审阅，但必须先于 C02 和 C06 落地。
- 验收：当前配置推导为 32 action / 2 latent；不兼容的参数明确报错；所有长度计算均来自同一契约。使用其他合法 A 验证切片通用逻辑不暗藏常数，不意味着本轮新增可变 block 的训练配方。

已 review 并提交的实现：

- `causal_block_geometry.py` 新增不可变的 CausalBlockGeometry，仅做 Python 数值计算，不依赖 torch、Reader 或 action 模板。
- temporal_compression_factor 必须由调用方从选定 VAE/配置显式传入，无默认值；actions_per_block 默认 32，video_stride 默认 4。latent_frames_per_block 由三者推导，不作为第二份独立配置。
- validate_observation_frames 校验完整片段；max_complete_observation_frames 只返回最大合法长度，不足一个 block 返回 0，不负责 warning 或选择裁剪方向。
- num_blocks / num_video_frames / num_latent_frames 给出长度关系；block_action_span / block_latent_span 返回左闭右开区间；action_latent_index / latent_block_index 完成局部索引映射。L0 的预测 block 索引为 -1，不分配 action。
- 当前配置下 33 observation → 9 video frames → 3 latent，其中 1 个是独立首帧、2 个组成预测 block。97 observation 对应 3 个预测 block。
- video_fps(aligned_fps) 返回 aligned_fps/video_stride，action_fps(aligned_fps) 返回原 aligned_fps；输入必须为有限正数，拒绝 bool 和字符串，输出保留小数。通用几何接口中的 aligned_fps 是调用方传入的 action/observation 时间网格频率；接口本身不读取 metadata，也不执行重采样。当前 SegmentLeRobotDataset 连续读取源数据，不做整体降采样；读取使用 meta.fps，输出 conditioning_fps 使用 episode source_fps（缺失时回退 meta.fps）。后续 C06 以该训练 FPS 调用几何接口，仅对视频执行 video_stride 抽帧，视频 FPS 变为训练 FPS/video_stride，action FPS 保持训练 FPS。不在几何对象中存储来源 FPS，以便不同来源复用同一几何契约。
- 47 项纯 CPU 检查通过，包含多 block 覆盖、合法长度、异常参数、FPS 校验、不同 stride 下的跨模态时长一致性，以及压缩倍率 3、action block 长度 12 的替代几何。替代参数仅验证通用计算，不代表现有模型支持相应配方；mRoPE 实际 position_ids 验证留在 C07/C08。
- 本次未修改现有 block_state 或训练调用方。C02/C03/C06/C07/C08 后续逐步消费此对象；现有模型仍有旧的压缩倍率与时序校验，尚未替换。两个规划文档只保留本地，不加入代码提交。

### C02 / P0：消费时序契约的纯索引切片规划器

建议标题：`feat(data): plan overlapping episode segments`

- 范围：片段记录和确定性切分算法、调用公共时序契约完成长度校验、尾段右对齐、短范围末尾截断、warning 与跳过计数。
- 边界：输入为有效时间网格范围；不处理 parquet、视频或 action 数值，不为满足长度修改用户的配置。
- 依赖：C01b 的时序契约及公共配置约定；不依赖 C06 的数值编码和统计实现，不单独硬编码 32/33。
- 接口方向：输入有效 episode/segment 范围、L、O 和时序契约；用 A 校验 `L % A == 0`，短范围保留 `floor((N-1)/A)*A+1` 帧，不足 `A+1` 帧 warning 并跳过。具体切片和尾段策略仍由本层负责。
- 验收：覆盖非整 block overlap、刚好覆盖末尾、不足最大片段、最低长度和重复尾段；长范围无间隙、无越界，短范围只截末尾。

已 review 并提交的实现：

- `action/segment_planner.py` 简化为一个 SegmentPlanner 类：构造时提供最大 action 步数、overlap 和 CausalBlockGeometry；plan() 接收有效 observation 数与起点，直接返回 `(起点, action 数量)` 列表。不再使用 EpisodeRange、ActionSegment、SegmentPlan 包装记录。
- 这些索引不是 parquet 行号。来源身份由调用方保留；plan 的来源/episode/segment 参数只用于 warning，不校验标识格式。目标偏移导致的可读边界由 Reader 确定，不在 C02 推断；Reader 连续读取源数据，不提供整体时间降采样。
- `(start, actions)` 对应 `observation[start:start+actions+1]`。例如 210 个 action 的示例返回 `[(0,96), (76,96), (114,96)]`，其 observation 范围为 `[0,97)`、`[76,173)`、`[114,211)`。
- skipped_ranges 和 discarded_action_steps 保存在规划器实例中，累计各次 plan 调用结果。建立新数据集索引时新建实例；丢弃计数只含没有被任何片段覆盖的有效 action 间隔，包括被跳过范围的所有间隔，不把 overlap 重复计算为丢弃。
- 只保留切片正确性所需的校验：geometry 长度兼容性、合法 overlap、非负整数起点/长度；不增加来源字符串或 episode 标识的防御性检查。
- 过短范围发出带来源、episode/segment、实际帧数与最低要求的 UserWarning。调用方在建立索引时规划，不应在每次 **getitem** 时重复规划和告警。
- C02 新增 27 项检查，与 C01b 的 47 项检查联合共 74 项通过。覆盖多种长度和 overlap 下的区间并集、边界、重复尾段、截断/跳过计数及另一套 geometry，验证没有硬编码 32/33。
- C02 本身不修改旧 build_episode_spans；C03 已在新 Reader 中实现 _append_index_records 和 _resolve_index，接入片段记录与实际读取。旧训练行为保持不变。

建议固定以下边界用例，observation 索引包含两端：

```text
L=96, O=20, 有效 action 数=210：
  action [0,96)、[76,172)、[114,210)
  对应 observation [0,96]、[76,172]、[114,210]

短范围 78 帧 observation：保留前 65 帧，截掉末尾 13 帧
短范围 65 帧 observation：完整保留
短范围 33 帧 observation：生成一个 block
短范围 32 帧 observation：warning 并跳过
```

### C03 / P1：公共 LeRobot 片段读取层

建议标题：`feat(data): read template samples by segment index`

- 范围：在 BaseActionLeRobotDataset 的读取能力上增加或抽取公共片段读取实现，接入 C01/C02；支持来源读取选项和实际变长片段。
- 边界：不复刻整份旧基类，不顺带重写已有 Reader；只抽取新路径必需的公共能力，机械搬移与行为变化避免混在一起。
- 依赖：C01、C02。
- 验收：默认读取 action 列且不移位；显式 state 目标偏移只执行一次；有效范围计算考虑目标偏移，连续读取源数据时间步；无边界复制；按片段返回正确长度、时间映射与 mask。
- 特别检查：LeRobot 的固定 delta_timestamps 如何服务变长请求，需给出按实际范围读取的实现，不能依赖其边界补齐行为；shuffle 分组索引须指向新片段索引空间。

已 review 并本地提交的实现及决策（`37dfe6e`）：

- `datasets/segment_lerobot_dataset.py`：SegmentLeRobotDataset 继承 BaseActionLeRobotDataset，一个实例对应一个 LeRobot 根目录；复用注册、懒加载和缓存，片段索引调用 C02 planner。shuffle 分组指向片段索引空间。
- 移除自行增加的 observation_stride。Reader 连续读取源表，读取 FPS 为 meta.fps，训练输出 FPS 采用 source_fps 并支持缺失回退；action_time_offset_steps=1 表示源数据下一行。仅后续 C06 按 video_stride 对视频抽帧。
- 不使用 mode="raw"。Reader 显式传 mode=None，FD/ID/Policy 由后续训练适配层选择；未分配模型 domain 时 domain_id=None，不伪造旧 embodiment。
- 删除 RawActionSample dataclass；通过继承的 `_build_result` 返回字典。该 BaseActionLeRobotDataset 方法不执行归一化，不能与旧 ActionBaseDataset 的同名方法混淆。
- 字典保留 action、video、ai_caption、conditioning_fps、mode、domain_id、viewpoint；通过 extras 携带 state_trajectory、action_target、state_mask、action_mask、时间戳、action_state_indexes、source_contract、read_options 和视角描述。
- Reader 当前的 action 与 action_target 指向同一绝对目标；C06 必须从 action_target 编码并将结果写入 action，不能把当前 action 直接当作 block delta。视频使用父类的 `[C,T+1,H,W] uint8` 格式，保留浮点 FPS。
- 字段存在性、metadata 声明维度、FPS、相机声明在初始化时检查；mask 与来源契约首次读取时检查并缓存；片段长度和目标边界建索引时确定。
- 后续 review 删除逐样本统一校验函数及 Reader 调用：不再重复检查由读取索引、arange 和切片直接构造的长度与映射，时间间隔检查已单独前移到读取 timestamp 后，使用 meta.fps，避免与训练 FPS 混用。初始化、mask 首次加载、片段规划、视频解码各自保留所需检查，读取对齐通过测试验证；实际数值、维度与旋转合法性由后续模板编码检查。读取通过不代表已完成数值语义验收。
- 数值按实际行索引查询，不修改共享 delta_timestamps、不使用边界复制；时间戳直接保留原始精度，避免默认 float32 转换损失长轨迹精度。state 下一步目标的 T 条监督只需要 T+1 个 state。

### C04 / P1：视频适配与首个 AgiBot 来源

建议标题：`feat(data): add agibot source adapter and view configuration`

- 范围：公共视角操作和来源相机配置，接入处理后的 AgiBot，建立第一个可审阅的完整 Reader 样例。
- 边界：不沿用原始 AgiBot FK/action 拼接处理已转换的数据；不改变旧 AgiBot Reader 的含义。
- 依赖：C03；先确认该来源字段、mask 和视频映射。
- 验收：一个和多个 block 的片段均能读取；视频与数值对齐；相机缺失按显式策略处理；无重复坐标转换或归一化。

已 review 并本地提交的实现及验证（`37dfe6e`）：

- `datasets/agibot_segment_lerobot_dataset.py`：AgiBotSegmentLeRobotDataset 为薄适配；默认 state_unified/action_unified、mask_state/mask_action，默认原始 action 不移位。template、source_contract、planner 外部传入，不做 FK、维度重排或归一化。
- 相机 key 在文件顶部定义常量；ego 使用 head，默认 concat 为上方 head、下方 hand_left/hand_right。`video_view.py::VideoViewConfig` 提供公共布局及读取；所选相机或文件缺失明确报错，相机配置不再保存 viewpoint，布局由 Dataset 独立指定。
- 布局 review 后移除 rows 隐式行列规则。VideoViewConfig.cameras 只声明 head/left/right 到来源视频字段的映射；取样时先令 `viewpoint = self._viewpoint`，读取调用显式传入 `self._read_video(..., viewpoint=viewpoint)`，参数继续传递到 read 和 compose；相机选择和 compose 按传入的 viewpoint 显式分支，分别调用 _compose_ego_view、_compose_concat_view。ego 保留 head，concat 明确缩放左右腕部并横拼，再与 head 纵拼。
- VideoViewConfig 只保存 cameras，不保存 viewpoint 或 description。初始化的 validate_features、读取的 read/compose、输出的 describe 都显式接收布局参数。默认 viewpoint 保存在 Dataset，本次样本局部 viewpoint 同时决定视频、返回标签和描述；未来布局增强可在取样时选择，不修改共享相机配置。本轮不实现增强策略；启用时需在初始化阶段校验所有候选布局所需相机。
- 新布局需要明确声明相机角色并新增对应拼接函数，不按相机数量推断缩放比例；two_view 尚未定义具体排布，当前不实现，未知布局报错。
- VideoViewConfig.read 复用 LeRobotDataset._query_videos：按源数据行的 timestamp 查询，加各相机的 episode 文件内时间偏移后解码。Reader 不做视频时间抽帧。
- 当前与旧 AgiBot Reader 的差异：默认直接拼三路视频，不会优先探测预合成 concat 文件；metadata 时间偏移要求显式存在，不使用旧逻辑的默认零回退。此为当前实现范围，不代表旧 Reader 已被等价替代；后续若需要预合成视频选择策略，应显式配置。
- 使用 cosmos-framework-py312 和 `/mnt/sfs_turbo/public/datasets/agibot_processed/v0/task_352/canonical_55d`，验证非零片段起点：32 action 的 ego 返回视频 `[3,33,480,640]`、目标 `[32,55]`；64 action 的 concat 返回视频 `[3,65,720,640]`、目标 `[64,55]`。这里 55 仅是该真实数据与所选模板的宽度，不是 Reader 约束。
- 字典返回修改后已复验原始 action、state 下一步目标、uint8 视频格式；sample_contract、片段 Reader、视频配置相关 36 项测试通过。未进行训练或 NPU 验证，不能据此宣称 packing 已接通。

### C05 / P1：EgoSuite 来源适配

建议标题：`feat(data): add egosuite template source adapter`

- 范围：EgoSuite 的薄适配、配置和针对性验证；若仅参数不同，不强造重复子类。
- 依赖：C04 的公共能力。
- 验收：真实字段、视频布局、时间含义和有效 mask，而不是只检查 shape。

已完成 review、提交并推送（`d28fdfe`）：

- 新增 `datasets/egosuite_segment_lerobot_dataset.py::EgoSuiteSegmentLeRobotDataset`，继承公共 SegmentLeRobotDataset；默认 `ego_view`，head 角色只映射 `observation.images.head_left`，不解码右相机。相机键在文件开头定义常量。
- 默认列 `state_unified/action_unified`，不移位、不重新重标；这批导出的 action 已是下一行 state。template/source_contract/planner 由调用方传入，不硬编码维度或物理槽位，不执行归一化。
- 使用 cosmos-framework-py312，分别读取两个真实来源的开头/中间/尾部 32/64/96 action 片段，得到 33/65/97 帧左视角视频。parquet 数值、mask、时间戳及独立左视频解码均一致；PTS 误差与编码前像素差均为 0。尾部不读取末行复制的 action。
- 视频和逐帧检查记录保存在 `outputs/egosuite_reader_review/`，入口 `index.html`，报告 `report.md`；共六个 MP4，保留供人工 review，不加入提交。公共 Reader/视频布局 33 项测试通过。
- C05 当时只验证读取适配；后续 C06/C07/C08 已完成模板 block 编码、packing 与多源训练接入，EgoSuite 单源及混合 NPU 验证见 C08。

### C06 / P1：固定 block 编码与统计契约

建议标题：`feat(action): encode fixed action blocks through templates`

- 范围：CausalActionSFTDataset、block_state 和相关统计接口支持模板；消费 C01b 的时序契约，按当前配置的 32 步 block 选择片段局部 anchor；夹爪/灵巧手绝对监督；统计与训练复用编码函数。此项不再首次定义 block 大小或片段合法长度。
- 边界：这是一个数据数值语义 commit，包含编码和匹配的统计读取改动，避免编码已变而仍加载旧统计；暂不启用新模型配方。数据编码统一使用新模板，不保留旧固定维度编码和旧统计兼容分支；examples/action_pretrain、旧模型配方及注册、推理 CLI 保持原样，待 C07 模型接入完成后在 C08 统一适配，不在 C06 提前删除或改写。
- 依赖：C01、C01b、C03；C04 提供首个真实数据验收来源，C05 不是本项必要前置。统计来源记录时序配置，使用者显式选择统计文件，不按配置哈希强制匹配。
- 验收：三 block 样本依次使用局部 state[0]、state[32]、state[64]；非对齐片段起点也正确；首帧不分配 action；先编码后归一化；统计维度或有效维度覆盖不满足时报错；模拟统计显式标记。
- FPS 接入：Reader 已将 conditioning_fps 设置为 source_fps（缺失时回退 meta.fps）；在实际执行视频抽帧的同一处调用 `geometry.video_fps(aligned_fps)` 和 `geometry.action_fps(aligned_fps)`（具体调用示意见下节），同时设置 conditioning_fps 与 conditioning_fps_action。action 不抽帧，不得在 Reader、transform、packing 多处重复除以 stride。
- state 时间位置的前置数据：由同一个 geometry 确定 anchor 的 action/state 索引及其视频 latent 边界索引。对第 b 个 block，anchor 索引来自 `geometry.block_action_span(b)[0]`，对应的 latent 索引为 `geometry.block_latent_span(b)[0] - 1`。当前配置分别得到 `0,32,64,...` 和 `0,2,4,...`；不得在 Reader 或 packing 中重新硬编码这些数值。
- 将边界 latent 索引随 block metadata 传给 packing（字段建议为 `state_latent_indexes`），它表示片段内索引，不是秒数或最终 mRoPE 坐标。C06 只提供映射，不计算最终位置、不改变扩散噪声 timestep。
- 映射验收：不同合法 geometry、不同片段起点及多个 block 均正确，独立首帧对应首个 state。为保证提交之间可运行，旧消费者仍依赖的 `state_frame_times` 暂不提前删除；实际替换与清理在 C07a 的独立提交完成。

#### C06 实施记录（已 review，已提交 `1cac896`）

以下记录 C06 提交时的边界；其中暂留的旧入口已在 C08 迁移，state 时间位置已在 C07a 简化，不再代表当前训练入口状态。

- `build_block_sample` 仅接受模板 Reader 的输出，通过 planner.geometry 统一计算 anchor、action 所属 latent 与 state 边界 latent。先调用模板编码，再独立归一化 state/action；不会再次从 state 重标目标。
- video 在同一函数按 video_stride 抽帧，保留独立首帧，同时更新 video/action FPS。默认 97 个 observation 形成 96 个 action、25 个视频输入帧、3 个 block state，anchor 索引为 `[0,32,64]`，state latent 索引为 `[0,2,4]`。
- 所有维度与旋转/绝对字段规则由模板提供。state_mask 与 action_mask 独立，仅相对字段要求 anchor 的对应 state 有效；夹爪与灵巧手不要求同维 state 有效。
- Review 后删除 statistics_key、payload 哈希及保存/比较逻辑；统计文件由调用方显式选择，geometry 仅作为可读来源信息保存。加载时检查统计数值有限、上下界合法；训练适配层首次使用时检查模板维度、有效维度覆盖及 mock 显式开关，不逐样本重复检查。验证集复用训练统计，`collect_statistics` 禁止在验证集拟合。
- `collect_statistics` 与训练复用编码；当前为指定有限样本的内存内精确 q01/q99 拟合接口，不是大规模流式统计任务。正式统计仍待人工提供。
- `mock_template_statistics` 显式接收 AgiBot state/delta 统计文件，由模板定义映射与占位范围；文件保留来源、字段映射和 mock 标记，训练使用必须设置 allow_mock_statistics=True。没有隐式回退到模拟统计。
- `CausalActionSFTDataset` 使用新 Reader 的 template/planner，支持 FD/ID/Policy；普通 transform 不再对 action 重复归一化，输出维度必须等于模板宽度。旧 DROID causal factory 已移除；旧注册配方、examples/action_pretrain 下的配置/脚本/README 及离线推理 CLI 已恢复原样，待模型接入完成后统一迁移。它们仍引用旧接口，当前不作为可运行的新模板训练入口；普通 DROID 与非 causal 配方不变。
- 推理解码辅助函数改为调用同一模板，避免删除旧编码后留下失效导入；模型与在线推理 session 的维度接入仍归 C07，不代表当前可以直接训练或推理。
- 保留 state_frame_times 的现有消费接口，并新增 state_latent_indexes；C07a 独立替换 position_ids 的时间计算，不在本次修改 mRoPE 或 diffusion timestep。
- 真实 AgiBot 原始 action / state 下一步目标、两个 EgoSuite 来源均完成 96-action 多 block 数值读取验证。模拟统计存在明显裁剪，EgoSuite 夹爪均被裁到同一边界，不能据此判断正式训练质量；明细保存在本地 `outputs/c06_review/`。

### C07 / P2：模型与 packing 的模板维度接入

建议标题：`feat(model): support template action dimensions in causal training`

- 范围：模型 action/state 投影与输出头、transform 维度校验、causal 元信息、必要的推理解码接口和 checkpoint 加载校验。
- 边界：不改变已确认的 attention 或任务定义；不同维度权重的加载策略显式处理，不静默 reshape。旧模型配方、注册和启动入口在 C06 保持原样；完成模型接入后，在 C08 将保留的配置和脚本适配新模板。
- 依赖：C06；模型维度/旧 checkpoint 处理策略在此项完成前明确。
- 验收：同一模板贯穿输入、loss mask、输出及解码；packing 中片段隔离，历史不串联；不同合法片段长度可正确形成 SequencePlan；新模板路径相关检查通过。
- 时间位置验收：追踪 conditioning_fps/conditioning_fps_action 到模型的 fps_vision/fps_action，再到 packing/mRoPE；检查视频、action 和 block state 的时间原点、间隔及 block 边界一致。覆盖小数 FPS、非整 block 的片段起点、多个 block 和混合 FPS 来源。

#### C07 第一部分实施记录（已 review，已提交 `8c350a0`）

- `CausalActionNetwork` 的 action 输入、state 输入和 action 输出层改为使用 `self.action_dim`，移除固定维度要求。已有模型构造路径将 `config.max_action_dim` 传入网络；C08 由模板宽度同时配置 transform 与模型，不在模型内定义字段语义。
- VAE 时间压缩倍率固定为 4，按用户决定删除模型里的重复倍率判断；不新增 geometry/VAE 同步逻辑，也不修改 VAE。
- 现有 packing、mask、噪声和 loss 路径已使用实际张量维度，沿用原实现；FD/ID/Policy、block state 和有限历史可见性保持不变。
- `CausalActionSession` 显式接收 template，state/action 宽度由模板提供。anchor 有效性使用模板规则，绝对夹爪/灵巧手无需同维 state 有效；无效 state 清零，action_mask 显式传给 planner/batch_builder。
- 权重加载复用现有机制，不自动裁剪、补零或改变语义。相同维度可以按既有流程加载；跨维度初始化需显式跳过不兼容接口，采用现有 `keys_to_skip_loading`。同维度并不保证语义兼容，正式初始化配置仍需结合用户提供的 checkpoint 决定；本次不修改配方。
- CPU 验证使用实际 action/state 投影、packing、解码和 loss，配合小规模 dense attention 验证梯度；覆盖替代维度、一个/三个 block 的不同长度片段、三任务、片段隔离、有效维度 mask，以及 session 历史重置和模板解码。没有加载完整预训练主干，也未进行 NPU 或正式训练验证。
- 本次未修改 `state_frame_times`、最终位置编码、训练脚本、旧配方或注册；C07a 单独处理时间位置，C08 再接入入口。

#### C07a：按 geometry 绑定 state 时间位置（独立提交）

建议标题：`refactor(action): align state positions with video block boundaries`

- 在 C06 的映射可用后实施，与 C07 的模型维度/权重改造分开提交。这里修改的是 `position_ids` 的时间分量，不是 diffusion timestep、噪声调度或 attention 可见范围。
- 修改 `sequence_packing/causal_action.py::expand_action_sequence`：根据样本的 `state_latent_indexes`，找到原始 packed video 中对应 latent 的 token，直接取其最终时间坐标写入该 state token。一个 latent 内的视频 token 共享时间坐标，取其中一个即可；正确处理空间 token 数、batch 中各样本偏移及打包后的 state token 索引。
- 只替换 state 的时间分量，保留现有其他位置分量。关闭 FPS modulation 时位置随 latent 索引走；开启时复用视频已经完成 FPS 缩放的坐标。文本前缀偏移也随视频位置保留，不重新乘 FPS、不从 action 位置间隔反推。
- 同一提交删除 `build_block_sample` 中生成 `state_frame_times` 的旧公式、metadata 对应字段及 packing 的旧消费逻辑，并更新必要的训练/推理调用方。检查残留引用，保证没有只有生产端或消费端完成迁移的中间状态。
- 新 Reader 已检查存储时间间隔，新模板 block 路径不再依赖 `action_timestamps/state_timestamps` 生成位置；移除无其他用途的训练层透传。旧 Reader 尚需的兼容校验独立保留，不据此删除视频查询使用的存储 timestamp；`storage_fps` 的保留范围按剩余消费者确定。
- 验收最终 `position_ids`：独立首帧和首个 state 同位置；每 block 两个预测 latent 时 state 绑定 `0,2,4,...`；覆盖 FPS modulation 开关、小数/混合 FPS、存储 FPS 与 source_fps 不同、文本前缀、多样本不同长度与空间网格。严格对齐输入下，应与旧换算的预期位置在浮点误差范围内一致。
- 确认 action 坐标、anchor 数值、模板编码、loss/attention mask 和噪声 timestep 不因本项改变。复用该 packing 的有限窗口推理同步验证；不将完整长时滑窗 batch_builder 和 KV cache 重定位扩展进本提交。

#### C07a 实施记录（已 review，已提交 `d2ede42`，未推送）

- `expand_action_sequence` 使用 `state_latent_indexes × H × W` 定位每个边界 latent 的首个空间 token，再通过该样本在 video.sequence_indexes 中的偏移取全局 token 索引。state 只复制该 token 的最终时间坐标，其他位置分量沿用原实现。
- `BlockStateMetadata.state_latent_indexes` 改为必填；删除 state_frame_times 字段及 block 编码中的旧时间差公式。代码中已无旧字段引用；build_block_sample 不再要求输入 state/action timestamps 或 storage_fps。
- Reader 的存储时间戳查询、时间对齐检查和 FPS 输出保持原样。SFT 层继续去掉 state/action timestamps，并去掉已无训练用途的 storage_fps，不再向 transform/模型透传。
- 验证使用实际 pack_input_sequence，覆盖 FPS modulation 开关、25/29.97 混合 source FPS 与 30 存储 FPS、非 block 对齐起点、不同文本长度、非零窗口起点、不同视频空间 token 数、两种片段长度及三任务。另覆盖当前 block 前已有观测历史的推理 packing。
- 确认相邻 block 边界处前一 action 与下一 state 时间坐标相同，但 attention 不可读取下一 state；不同样本互相不可见，超出历史窗口的视频不可见。action/视频位置、非时间分量、action 数值、anchor、噪声 timestep 保持原数据路径；不修改 attention 或 loss 实现。
- 更新 C07 模型接口测试，继续检查不同维度下的前向/反向与有效维度 loss。没有启动完整模型训练、NPU 验证或修改训练配方；长时滑窗 batch_builder 与 KV cache 重定位不在本次范围内。

### C08 / P2：多源混合及正式配置入口

建议标题：`feat(train): configure multisource causal action training`

- 范围：复用 CausalActionMixture，补模板一致性与来源参数校验；配置 schema、factory、示例配方、任务/来源权重、VAE 合法时长和统计配置一起接通。
- 边界：以新增显式入口启用本方案，不把默认旧入口偷偷切换为新读取行为；不引入新的混合框架。
- 依赖：C05、C06、C07。
- 验收：AgiBot 与 EgoSuite 两来源配置可构造混合样本；rank/worker 分片不漏发或误重复片段索引；overlap 导致的内容重叠属于预期；小数据源分片不足时有清晰错误；独立检查数据源权重和任务权重。
- 配置约束：允许的 VAE 输入长度由合法片段长度导出，避免仍只支持旧的固定 encode duration。
- mRoPE 配置：新入口必须明确 enable_fps_modulation 的取值。使用真实 FPS 时间缩放时启用它并沿用一致的 base_fps；关闭时需验证现有 causal action 的 latent 相对时间修正分支，不能把关闭解释为已经按真实秒数编码。不得仅修改 FPS metadata 就宣称完成 mRoPE 接入。

### C08 实现记录（已 review，已提交，未推送）

- 训练代码与回归测试：`c106b6c`；配置 schema、recipe、来源 JSON 与通用训练脚本：`4c7c999`。提交前复查 31 项数据测试和 42 项配置测试通过。
- 离线推理 CLI 与脚本：`0dc90fe`，已迁移公共来源清单、模板和 geometry；完整推理尚未验证。
- 训练监控 callback：`8d0f1e6`，增加任务类型、action 步数、state block 数，loss 转日志标量前 detach；静态检查通过。
- README、本机环境入口 `launch_midtrain_template.sh` 尚未提交；临时验证脚本 `validate_training.sh` 和日志汇总脚本 `summarize_training.py` 已按要求删除。
- `[action]` 配置统一提供可替换模板、actions_per_block、video_stride、max_action_steps 和 overlap；factory 由 `sources/*.json` 构造来源，transform 与模型宽度均取模板。
- 来源清单独立配置路径、视角、读取选项、统计路径、来源权重和可选任务权重。示例 AgiBot 使用下一步 state，EgoSuite 读取已有 action，不重复移位。
- VAE 合法输入帧数由 geometry 导出，默认 9/17/25；模型固定 block 大小为 2 latent。VAE 压缩倍率仍固定 4。
- 明确开启 FPS modulation，base_fps 默认 24；state 复用 C07a 的最终 video 坐标。
- mixture 改为片段分片，使仅一条 episode 的 EgoSuite 子集可以跨 rank/worker 训练；同一来源 epoch 内索引不重复，保留 episode 内的顺序读取。
- 统计按来源指定；未提供时必须显式允许 AgiBot 模拟统计。默认不自动收集、不自动将旧统计视为正式 block 统计。
- `examples/action_pretrain` 的通用训练脚本、TOML、来源清单及离线推理入口已提交；使用说明和本机环境入口目前仅保留本地，三组顺序验证脚本与日志汇总工具已删除。
- 接入测试发现并修复 read_options 配置对象进入 collate 的问题；该对象在 SFT 适配后不再进入训练 batch。
- 已分别完成 AgiBot、EgoSuite、混合的 4 卡×10 步训练，三组 checkpoint 均完整保存；混合 batch 已实际覆盖不同来源。193 项相关 CPU 检查通过。具体范围与模拟统计限制见 C08 实验报告。

### C08b / P2：自动发现子数据集与来源组混合

状态：已完成编码及读取验证，已本地提交（`ec17886`、`2bcb6f8`），未推送。训练 factory 已支持单个数据集或父目录；以下保留已确认方案，实施结果见本节末尾。

目标：AgiBot、EgoSuite 各用一条 JSON 配置声明来源组，自动使用父目录下的 LeRobot v3 子数据集，避免逐目录维护清单。

#### 配置与目录发现

- 保留现有 `sources` 结构和 `root` 字段。`root` 本身是数据集时直接读取；否则递归发现其下的数据集，兼容已有单数据集配置。
- 依据 `meta/info.json`、LeRobot v3 版本及 `data` 目录识别数据集。找到根目录后停止向其内部递归，避免扫描 parquet 和视频文件。
- 发现结果按规范化路径去重、排序，保证各 rank 的来源顺序一致；启动时记录展开后的实际目录清单，便于复现和检查。找不到数据集时报错，不静默训练空来源组。
- 提取统计工具 `tools/compute_causal_action_stats.py` 中的目录发现逻辑到公共模块，由训练 factory 和统计工具共同调用，训练代码不反向依赖 tools。
- 同组共享 `reader`、`viewpoint`、`target_semantics`、`read_options`、任务配置和 `statistics_path`。第一版每组显式指定一份共享统计文件；模拟统计仍受现有显式开关控制，不自动计算统计。
- 每个子目录仍创建自己的 Reader，独立读取 metadata、FPS、mask、episode 和视频，保留真实来源路径。不同读取规则的数据划入不同配置条目，不根据目录名猜测语义。

示例配置如下，路径替换为实际部署位置：

```json
{
  "sources": [
    {
      "reader": "agibot",
      "root": "${AGIBOT_ROOT}",
      "viewpoint": "concat_view",
      "target_semantics": "next_state",
      "read_options": {
        "action_from_state": true,
        "action_time_offset_steps": 1
      },
      "statistics_path": "/path/to/agibot_stats.json",
      "weight": 1.0
    },
    {
      "reader": "egosuite",
      "root": "${EGOSUITE_ROOT}",
      "viewpoint": "ego_view",
      "target_semantics": "preprocessed_next_state",
      "statistics_path": "/path/to/egosuite_stats.json",
      "weight": 1.0
    }
  ]
}
```

#### 组内索引与采样权重

- 各子数据集分别经过现有模板编码和 SFT 适配，再由轻量 map-style 组合 Dataset 合并片段索引。组合层只做全局索引到子集局部索引的映射，并提供加上子集偏移后的 episode 索引范围。
- 每个来源组作为一个 dataset 交给现有 `CausalActionMixture`，随后复用现有 packing。不得跨子目录拼接 episode 或片段，也不改变样本中的真实来源信息。
- JSON 的 `weight` 表示整组权重。示例中 AgiBot/EgoSuite 各约占 50%，不随各组子目录数量改变。
- 组内按有效片段数分配采样机会，不给每个子目录复制整组权重；小子集不会仅因单独成目录而被过度采样。
- rank/worker 在组内统一对片段索引分片，最低片段数要求由每个子集改为每个组至少覆盖 `rank × worker`。同一组 epoch 内片段索引不漏发、不误重复；已有 overlap 带来的数据内容重复仍属预期。
- 来源权重和 FD/ID/Policy 任务权重保持独立。模板语义、block anchor、片段开头的 causal 历史重置、attention 和 loss 不变。

#### 提交边界与验收

1. **公共目录发现**，建议标题 `refactor(data): share LeRobot v3 dataset discovery`：提取公共发现函数，统计工具迁移调用；检查单根目录、多层父目录、去重、稳定排序、空目录和遇到数据集后停止递归。此提交不修改训练混合行为。
2. **来源组训练接入**，建议标题 `feat(action): train from discovered dataset groups`：增加组合 Dataset，在 factory 中展开并构建来源组，同步更新来源 JSON 示例；验证全局索引映射、episode 范围偏移、来源 metadata 保留、组级分片和组间/组内采样比例。相关回归测试随功能提交。

- 依赖：C08 的公共 factory、来源配置及片段分片路径；单数据集 JSON 保持兼容。
- 集成验收：使用 AgiBot/EgoSuite 父目录构建数据，核对发现清单、各子集及整组片段数，读取不同子集的样本；包含单个子集片段少于 worker 数但整组足够的场景。
- 推理入口复用同一 factory，需同步检查组合 Dataset 的按索引读取及任务选择；`source-index` 对应 JSON 来源组，`index` 对应该组的全局片段索引。
- 代价与边界：初始化需要发现并建立更多子集索引。此项只简化配置和组合方式，不承诺减少现有 Reader 的 parquet 缓存或索引内存，也不在此提交扩展为缓存管理优化。
- 实施前仍按逐项授权推进；规划文档继续仅保留本地，不随代码提交。

#### C08b 实施记录（2026-09-27，已提交，未推送）

- 提交拆分：公共目录发现 `ec17886`；来源组训练接入 `2bcb6f8`。本次新增的三个 `_test.py` 文件已按用户要求删除，既有测试文件保留；以下测试结果记录删除前的验证，不表示新测试已进入提交。
- 新增公共 `action/lerobot_discovery.py`，将统计工具的目录发现实现原样提取并复用；不引入训练对 tools 的依赖。
- 新增 `CausalActionDatasetGroup`，继承 PyTorch `ConcatDataset` 复用累计长度和索引定位，只补 template、episode 范围偏移和显式 `set_mode`。没有另写分片算法，现有 mixture 按组合后的组索引工作。
- factory 对每条配置发现目录并创建子 Reader/SFT，组成一个组；日志记录实际路径与片段数。组内读取选项及正式统计路径共享，模拟统计仍按各子集 mask 显式构建。零有效片段子集沿用原 factory 的明确报错，不静默跳过。
- EgoSuite 单源与混合 JSON 改为父目录条目；AgiBot 原本已使用 `${AGIBOT_ROOT}`，只需将该环境变量指向父目录，也继续支持单个数据集根目录。组权重不随子目录数量增加；EgoSuite 从旧的两子集等权改为组内按片段数采样。
- CLI 改为调用组的 `set_mode`，将显式推理任务传到所有子集；`source-index` 对应组，`index` 对应组内全局片段。
- 20 项目录发现、组合层、factory 和 mixture 测试通过，覆盖单根目录兼容、父目录展开、配置继承、组级分片、索引边界及采样比例；统计工具迁移后的 27 项回归测试通过。Ruff、格式检查与 CLI 帮助检查通过。
- 使用 `cosmos-framework-py312`，从 `/mnt/sfs_turbo/public/datasets/agibot_processed/v0` 发现 8 个子集，共 39,311 个片段、2,487 个 episode 索引范围；从 EgoSuite 父目录发现 2 个子集，共 29 个片段、2 个 episode 索引范围。两组权重为 1:1。
- 实际通过 factory 构造上述两组，逐子集读取首尾各一个片段，共 20 个样本；检查来源路径、Policy 任务覆盖、action 数值有限及真实 collate。当前样本均为 96 个 action、25 个视频帧，视频 FPS 为 7.5；AgiBot 视频尺寸为 256×256，EgoSuite 为 256×320。
- 验证使用旧 AgiBot 模拟统计，无 tokenizer、无模型前后向，不代表正式统计或扩展后的 NPU 联训已验收。仅读取样本，不导出视频或 checkpoint。
- 清理记录：C08b 临时读取脚本、日志、Hugging Face 缓存及临时目录已删除；正式功能保留；本次新增测试后续已按用户要求删除，既有回归测试保留。
- 环境记录：默认 Hugging Face 缓存不可写，且 `/tmp` 所在盘已满；验证通过环境变量将缓存及临时目录放到工作盘完成，未修改公共环境或数据集。

### FPS 与 mRoPE 的接入边界（后续实现必查）

C01b 已补纯换算接口，C06 负责实际抽帧和 metadata 更新，C07/C08 负责模型与位置编码链路验收：

```python
# 后续 C06 的调用示意；本次未修改训练调用方。
aligned_fps = float(raw["conditioning_fps"])
video = raw["video"][:, ::geometry.video_stride]
video_fps = geometry.video_fps(aligned_fps)
action_fps = geometry.action_fps(aligned_fps)
# video 与 conditioning_fps=video_fps 一起写回新样本；
# conditioning_fps_action=action_fps，action/state 不随视频降采样。
```

例如 aligned_fps=30、video_stride=4 时，视频 FPS 为 7.5，action FPS 仍为 30。传给视频 mRoPE 的 FPS 是 7.5，不是再除 VAE 压缩倍率后的数值；mRoPE 内部会根据实际 VAE 压缩倍率换算 latent 时间。保持 base_fps 为模型的参考尺度，不跟随每个来源的 stride 改动。

block state 的最终时间位置按 C06/C07a 的 geometry 映射绑定视频边界，不再通过测量时间与 action 位置间隔反推。片段独立首帧及首个 state 位于同一局部时间零；action 的区间起点语义与现有 action_start_frame_offset 约定须在接入时一起检查，避免多移或少移一个时间步。

当前代码已有以下机制，C06/C07a/C08 已完成对应接入，后续扩展继续复用：

- `action/block_state.py` 的 build_block_sample 同时执行视频抽帧和两个 FPS 的赋值。
- `model/generator/omni_mot_model.py` 分别读取视频/action FPS；未提供 conditioning_fps_action 时会回退到视频 FPS，新路径应始终显式提供。
- `sequence_packing/modalities.py` 与 `mrope.py` 在 FPS modulation 开启时用各模态 FPS 生成时间位置。
- `model/generator/omni_mot_causal_action_model.py` 在 modulation 关闭时仍有基于 action_frame_ids 的相对 latent 时间修正。
- `sequence_packing/causal_action.py` 的 expand_action_sequence 已在 C07a 改为使用 state_latent_indexes 复制对应 video token 的最终坐标，不再消费 state_frame_times。

C01b 的纯数值检查只证明抽帧前后物理时长一致，不证明现有训练配置或最终 mRoPE position_ids 已正确使用这些值。

### C09 / P2：集成验证与使用说明

状态：已完成 C08 范围内的三组短程 NPU 训练及续训验证，实验报告保留本地；尚未完成本项全部验收。C08b 已完成真实数据读取集成验证，扩展后 NPU 训练、完整模型推理验证及使用说明整理仍待进行，正式统计与长训质量不在已有模拟统计验证结论之内。

建议标题：`docs(action): document multisource training validation`

- 范围：经授权实施后记录真实数据读取、混合 batch、模型前后向及 NPU 验证结果和使用说明；必要集成检查随相关功能 commit 落地，不把全部测试拖到此项。
- 边界：不提交临时探查脚本、大型输出、数据副本或 checkpoint，不把发现的功能修复隐藏在文档 commit 中；修复应单独说明或归入尚未提交的所属功能。
- 依赖：C08。
- 验收：报告明确配置、数据范围、模板和统计版本、警告/跳过数量及验证局限；模拟统计的跑通不能当作正式统计或模型质量结论。

## 附录 B：原 review 里程碑与依赖顺序

1. **基础规则 review（C01、C01b、C02）**：只看模板、读取契约、公共 causal 时序约定和切片输入输出，不需要加载训练模型。
2. **首个来源 review（C03–C04）**：从片段记录追踪到真实视频、state 和绝对目标，确认没有重复变换。
3. **数值语义 review（C06）**：展示多 block anchor 与编码前后数值，确认统计口径。可先于 C05 完成。
4. **完整训练接入 review（C05、C07–C08）**：验证多源、变长片段、mask、packing 隔离和模型维度共同成立。
5. **验证报告 review（C09）**：记录实测证据，不以 shape 正确或 loss 有限替代语义验证。

建议主线顺序为 C01 → C01b → C02 → C03 → C04 → C06 → C07 → C05 → C08 → C08b → C09，先完成单源闭环，再扩展混合。当前 C01～C08 已提交，C08b 已实现、验证并提交。编号用于功能追踪，C05 提前完成也不影响依赖关系。

核心依赖为 `公共 causal 时序契约 → 切片规划` 和 `公共 causal 时序契约 → block 编码/统计`。C02 负责“如何取片段”，C06 负责“如何编码片段内的目标”，两者共同遵守前置的模型时序要求，不相互重复定义规则。
