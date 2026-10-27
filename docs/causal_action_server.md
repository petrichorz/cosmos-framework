# Causal action policy 推理服务

服务入口：`python -m cosmos_framework.inference.causal_action.server`。使用与
`examples/action_pretrain/inference_robotwin_template.sh` 相同的 causal action checkpoint、
Wan2.2 VAE、unified55 模板和训练 state/delta 统计。网络协议兼容 LingBot/OpenPI 的
WebSocket + msgpack/NumPy 编码，单模型 rank、单仿真器连接。
当前请求协议是 `cosmos-causal-action-v2`：client 发送 joint/endpose 字典，
server 组装 unified55 并返回拆分后的动作。EEF 输入和输出均为训练数据坐标系的
`xyz+xyzw`；server 不做坐标系或四元数顺序转换，client 与 server 需要同步更新。
首次连接先接收 metadata 字典，其中包含支持的执行模式、左右臂 DOF 和四元数顺序。

## 模板启动与数据回放

`examples/action_pretrain/*_template.sh` 是跨机器模板。使用前将 `/path/to/...`
替换为实际路径，或通过环境变量设置 `ASCEND_ENV`、`CONDA_ROOT`、`CONDA_ENV`；
模型服务还需配置 `CHECKPOINT_ROOT`、`ROBOTWIN_GROUP_STATS_PATH`、
`COSMOS3_EDGE_PROCESSOR_PATH` 和 `WAN_VAE_PATH`，数据回放还需配置 `ROBOTWIN_ROOT`。
checkpoint、统计文件、processor 和 VAE 应与训练配方对应。

终端一启动服务，等待日志出现 `Causal policy listening`：

```bash
bash examples/action_pretrain/robotwin_server_template.sh
```

默认使用 NPU 0，10 次 UniPC denoise，CFG guidance=1，
预测并执行完整 32 个 action。可通过环境变量 `CHECKPOINT_ROOT`、
`ROBOTWIN_GROUP_STATS_PATH`、`COSMOS3_EDGE_PROCESSOR_PATH`、`WAN_VAE_PATH`、
`ASCEND_RT_VISIBLE_DEVICES`、`NUM_STEPS`、`GUIDANCE`、`HISTORY_BLOCKS`、`FPS`、
`HOST`、`PORT`、`SEED` 覆盖；其他 Python 参数可以直接追加到 shell 后面。
`FPS` 是训练对齐后的原始 action/control FPS（默认 30），视频抽帧后 FPS=FPS/4。

`--execute-steps` 必须等于 `--actions-per-block`，未指定时自动取该值。
首次观测独立 prefill；后续反馈必须覆盖完整训练 block，并联合 prefill。
默认 32 action / 2 latent，执行 16 步的半块反馈会报错。例如：

```bash
bash examples/action_pretrain/robotwin_server_template.sh --execute-steps 32
```

客户端从 metadata/响应读取执行步数；首次仍发送 1 帧，随后每轮发送 32 帧新观测和
末帧 state。server 抽取第 4、8、……、32 帧，按两个 4 帧 chunk 依次更新流式 VAE，
得到两个真实 latent 后做一次联合 prefill，再使用最新 state 预测下一轮。
预测的 2 个 latent 不作为真实反馈。以下未特别说明的时序示例均使用默认 32 步。

终端二使用训练 Robotwin 数据假装仿真器观测，无需运行 Robotwin 仿真：

```bash
bash examples/action_pretrain/robotwin_replay_template.sh
```

每次运行固定完整回放所选片段一次，开始前 reset server 并清空历史缓存。
结果直接保存到 `outputs/seed{server 采样 seed}/`，例如默认模型采样 seed 为 42 时保存到
`outputs/seed42/`。seed 从 server 连接 metadata 读取，与 `DATA_SPLIT_SEED` 分开。
`OUTPUT_ROOT`（Python 参数 `--output`）可修改输出根目录，seed 文件夹会在该目录下创建。

回放从 `ROBOTWIN_ROOT` 下第一个数据集的第一个训练片段读取三视角原始图像、
state、mask、任务描述和实际 FPS；`--dataset-index`、`--index` 可选择其他数据。
本机和模板脚本的离线/replay 默认统一为 `DATA_SPLIT_SEED=42`、`MAX_ACTION_STEPS=896`、
`OVERLAP_ACTION_STEPS=161`，并共用 `SAMPLE_INDEX`；replay 的 `DATASET_INDEX` 默认是 0。
对应 Python 参数为 `--split-seed`、`--max-action-steps`、`--overlap-action-steps` 和 `--index`。
`DATA_SPLIT_SEED` 只控制数据排序，与模型采样 seed 分开。
replay 与离线均根据所选片段的实际 action 数自动计算预测轮数：`轮数 = 实际 action 数 / execute_steps`，
自动处理完整片段，无需配置轮数。
回放会将训练 state 转成原生 joint/endpose 字典，只发送数据里实际有效的字段，
不会从关节目标伪造 EEF。EEF 直接使用数据集的 `xyz+xyzw`，不转换坐标系或四元数顺序。
mask 只用于本地字段选择，不发送给 server。
请求观测时刻依次为 0、32、64、……，直到处理完整片段，每次返回动作保存至
`outputs/seed42/stepXXXXXX_action_joint.npy`（以 server 采样 seed 为 42 为例），
如果模型支持 EEF，则同时保存 `_action_eef.npy`。
预测期间只生成 latent，不逐轮解码或保存 `stepXXXXXX_prediction.mp4`。
每轮保留完整 32 步对应的两个预测 latent，片段结束后将首个真实观测 latent 与
全部保留的预测 latent 拼接，调用一次 `model.decode()`，保存为 `prediction.mp4`。
视频保存复用训练 `EveryNDrawSample` 的 `save_img_or_video()`：MP4、默认 `quality=5`、
16 FPS，包含首帧重建；完整 896-action 片段对应 225 帧。
这是展示用的整段解码；每轮推理仍接收真实观测，预测 latent 不写入 observed KV。
`metrics.json` 的 `prediction_video_path/prediction_video_frames/prediction_video_fps` 记录完整视频信息。
`metrics.json` 保存模型采样 `seed`、数据来源及原始数据集 `episode` 编号、片段实际动作数
`action_steps`、自动预测轮数 `rounds` 及服务耗时。回放数据的下一帧不受预测动作影响，
这只能验证接口和推理链路，不衡量真实闭环成功率。

## 离线使用相同的完整 block 预测与保存流程

`inference_robotwin_template.sh` 和 `inference_robotwin_local.sh` 默认使用 10 步采样、
guidance=1、预测 32 步，并通过 `--execute-steps 32` 每轮保留完整 32 步动作及两个预测 latent。
随后从数据集读取新的 32 组真实观测和对应 state，复用 server 的 `CausalPolicyEngine`
更新流式 VAE / observed KV，再预测下一轮；预测视频不写入历史。
`NUM_STEPS`、`GUIDANCE`、`EXECUTE_STEPS` 可覆盖这些默认值。
执行步数与 server 保持相同约束，必须等于预测 block 的 action 步数；
当前预测 32 步时使用 `EXECUTE_STEPS=32`，不支持半块反馈。

```bash
NUM_STEPS=10 GUIDANCE=1 EXECUTE_STEPS=32 \
  bash examples/action_pretrain/inference_robotwin_template.sh
```

`action.npy` 和 `sample.pt` 的 action 累积每轮完整的 32 步。
`sample.pt` 的 `vision_latents` 包含首个真实观测 latent 和全部保留的预测 latent，
可直接作为一个完整序列传给 `model.decode()`。`video.mp4` 在全部预测结束后一次解码并保存，
包含首帧重建，沿用 `EveryNDrawSample` 的 MP4 保存函数、默认质量和 16 FPS。
`observed_prefix_latents` 仍保存各轮真实观测 latent，供检查推理条件使用；
整段解码只使用其中第一个，其余真实 latent 不替换保存的预测 latent。
`metrics.json` 保存实际数据来源、episode、起点、采样参数、各轮保留步数和视频帧数/FPS；
传入 `--evaluate-actions` 时，`action_loss.json` 评估每 32 步重新锚定当前 state 后的归一化 action 误差。

对比离线与 WebSocket replay 时，必须匹配 checkpoint、统计文件、processor、VAE、prompt、
模型采样 seed、数据 episode 和起点。对同一个具体数据集、相同切片参数、排序 seed，
当前 RoboTwin 配方的 `split_val_ratio=0` 下，两端相同的 `SAMPLE_INDEX` 对应相同片段。
父目录包含多个子数据集时，离线 `SAMPLE_INDEX` 是来源组内的全局索引，
replay `--index` 是 `DATASET_INDEX` 指定子集内的局部索引；对照时建议将 `ROBOTWIN_ROOT`
直接指向同一个具体数据集，例如 `/path/to/RoboTwin__adjust_bottle/clean`。
两端 `metrics.json` 都记录 `split_seed/max_action_steps/overlap_action_steps`，
还应核对 `source/episode/start`。两端自动处理完整片段，相同片段的视频长度一致。

例如，将两端 checkpoint、统计和数据根目录设为一致后，使用相同样本索引对照：

```bash
export DATA_SPLIT_SEED=42 MAX_ACTION_STEPS=64 OVERLAP_ACTION_STEPS=0 SAMPLE_INDEX=0

bash examples/action_pretrain/inference_robotwin_template.sh
bash examples/action_pretrain/robotwin_replay_template.sh
```

离线入口未传 `--execute-steps` 时保留原有 FD/ID/Policy 采样接口；
`--preview-ground-truth-states` 是原有整块离线预览，与新的执行前缀模式不能同时使用。

## 联合 RoboTwin 运行真实闭环仿真

已保存可直接启动的脚本
[`examples/action_pretrain/robotwin_client.py`](../examples/action_pretrain/robotwin_client.py)，
具体实现位于
[`robotwin_client.py`](../cosmos_framework/inference/causal_action/robotwin_client.py)。
它创建 RoboTwin 环境并调用 `CausalPolicyClient`，完成 reset、执行动作、收集反馈和结束清理。
不需要修改 RoboTwin 的 `get_obs()` 或 `take_action()`。
本环境没有运行 RoboTwin 仿真；脚本通过模拟环境测试验证交互流程，真实仿真效果待实际部署验证。

### 两端环境与启动顺序

server 使用当前模型推理环境，client 使用能够独立运行 RoboTwin 的 Python/CUDA/SAPIEN 环境。
两端可以在不同机器，不要求共享 Python 环境或 GPU。client 不加载 checkpoint、Torch 或 VAE，
只需能够读取本仓库和 RoboTwin 仓库，且其 Python 环境安装 NumPy、msgpack、
OpenCV（用于双腕视角 resize）、支持 `libx264` 编码器的 FFmpeg 命令行程序，以及
支持 `websockets.sync.client.connect` 的 websockets（本仓库使用的接口需要 15.x 或更新版本）。
无需在 RoboTwin 环境安装整套 Cosmos 训练依赖，示例入口会自动添加本仓库至 Python 搜索路径。

1. 在 server 机器的仓库根目录启动模型服务，选择空闲设备，等待 `Causal policy listening`：

   ```bash
   cd /path/to/cosmos-framework
   ASCEND_RT_VISIBLE_DEVICES=0 PORT=8000 NUM_STEPS=10 GUIDANCE=1 SEED=42 \
     bash examples/action_pretrain/robotwin_server_template.sh
   ```

2. 在具备 RoboTwin 仿真能力的机器激活其原有环境，再运行下面的 client。
   将 `/path/to/...` 和 `SERVER_IP` 换成该机器可访问的实际位置；同机连接可以使用 `127.0.0.1`。
   运行时必须显式传入 `--robotwin-root`，不使用默认路径或 `ROBOTWIN_ROOT` 环境变量。

   ```bash
   python /path/to/cosmos-framework/examples/action_pretrain/robotwin_client.py \
     --robotwin-root /path/to/RoboTwin \
     --server-uri ws://SERVER_IP:8000 \
     --task-name adjust_bottle \
     --task-config demo_clean \
     --control-mode joint \
     --state-source command \
     --seed 100000 \
     --episodes 1 \
     --max-steps 32 \
     --output /path/to/outputs/causal_action/robotwin_sim
   ```

   `--max-steps 32` 适合先检查一轮动作执行；删除这个参数后使用 RoboTwin 任务本身的步数上限。
   默认由 RoboTwin 根据任务和当前 seed 的场景生成 instruction，无需手写文本。
   如需固定文本调试，可以额外传入 `--instruction "..."` 覆盖。

3. 查看输出目录的 `metrics.json`，确认请求、执行步数和 episode 是否成功。
   如需连续测试多个 episode，设置 `--episodes 2`；每次新 episode 都会先清空 server 缓存。

脚本会读取所指定 RoboTwin checkout 的 `scripts/eval_policy_xpolicylab.py`，复用其中的
`load_task_args()`、`class_decorator()`、`safe_close_env()`、`resolve_eval_instruction()` 和
`build_instruction()`，不进入其原有 RPC 推理循环。
因此该文件必须存在；换用其他 RoboTwin 版本时，需要对应适配这些配置/环境/文本辅助接口。
载入时切换工作目录到 RoboTwin 根目录，以解析机器人资产等相对路径；输出路径会提前转换成绝对路径。

### 参数

| 参数                            | 默认值                               | 含义                                                                                     |
| ------------------------------- | ------------------------------------ | ---------------------------------------------------------------------------------------- |
| `--robotwin-root`               | 无，必须显式指定                     | 具有仿真依赖和资产的 RoboTwin 仓库                                                       |
| `--server-uri`                  | `ws://127.0.0.1:8000`                | 模型服务地址                                                                             |
| `--task-name` / `--task-config` | `adjust_bottle` / `demo_clean`       | 环境类名及 `env_cfg/task_config/<配置>.yml`                                              |
| `--instruction`                 | 不指定，由 RoboTwin 生成             | 可选文本覆盖；指定后所有 episode 共用该文本，跳过专家预运行                              |
| `--instruction-type`            | task config 的 `eval_instruction`    | 自动生成文本使用 `seen` 或 `unseen` 模板；`demo_clean` 默认为 `seen`                     |
| `--control-mode`                | `joint`                              | 选择 server 返回的 `joint` 或 `eef` 动作；当前 checkpoint 只支持 `joint`                 |
| `--state-source`                | `command`                            | `command` 使用 get_obs 的 drive target；`measured` 改用实测关节 qpos                     |
| `--seed` / `--episodes`         | `100000` / `1`                       | 起始仿真 seed 及完成的 episode 数量；每次尝试后 seed 加 1，不稳定 seed 不占 episode 数量 |
| `--max-steps`                   | 不额外限制                           | 实际上限取该值与 RoboTwin 任务上限的较小者，单位为 take_action 调用次数                  |
| `--render-freq`                 | `0`                                  | 透传 RoboTwin 的界面渲染频率；0 不开启 viewer，仍采集相机图像                            |
| `--video-fps`                   | `30`                                 | 保存的观测视频播放 FPS；不改变仿真推进或模型请求频率                                     |
| `--output`                      | `outputs/causal_action/robotwin_sim` | 相对启动目录解析的日志、预测动作和观测视频保存目录                                       |

server 的 `SEED` 控制模型采样，client 的 `--seed` 控制仿真场景，两者分别配置。
`--help` 只解析参数，不导入 RoboTwin/SAPIEN，也不会连接 server。

`--task-name`、`--task-config`、`--control-mode` 都支持用逗号分隔多个值，
按 task × config × mode 的所有组合依次执行，顺序与参数中的值一致。
逗号两侧的空格会被去除；空项、重复项和 `joint/eef` 以外的控制模式会在参数解析时报错。
例如下面会执行两个任务、两种场景配置和两种控制模式的 8 个组合，每个组合完成 3 个 episode：

```bash
python examples/action_pretrain/robotwin_client.py \
  --robotwin-root /path/to/RoboTwin \
  --task-name adjust_bottle,beat_block_hammer \
  --task-config demo_clean,demo_randomized \
  --control-mode joint,eef \
  --episodes 3 \
  --seed 100000 \
  --output outputs/causal_action/robotwin_sim
```

各组合分别建立连接，单组合和多组合统一保存到 `<output>/<task>/<config>/<mode>/`，
其中包含本组合的 `metrics.json` 和 `seed{seed}/` 结果。
每个组合都从 `--seed` 开始，不稳定 seed 的重试规则不变。
所有模式都必须由当前 server 支持；不支持的模式会报错并停止后续组合。
客户端在每轮结束时打印结果摘要，包括成功状态、步数、耗时、视频及请求记录的保存路径，
并打印 `metrics.json` 路径；`requests` 内容只写入独立的 `requests.json`，不打印到终端。

每个组合的 `metrics.json` 还记录 `completed_episodes`（正常完成轮数）、
`successful_episodes`（其中任务成功的轮数）和 `success_rate`（成功轮数 / 正常完成轮数，取值 0..1）。
每次尝试结束后更新统计；不稳定 seed 和异常退出轮次不计入分母。
尚无正常完成轮次时，两个计数均为 0，`success_rate` 为 `null`。

### 单个 episode 的具体执行顺序

1. client 建立 WebSocket 连接，读取 server metadata，校验所选 `control_mode`。
   选择不支持的 EEF 模式会在创建仿真环境之前失败。
2. 根据 task config 加载双臂 embodiment、相机尺寸和机器人资产配置，创建任务环境，
   调用 `setup_demo(now_ep_num=episode, seed=seed, is_test=True, ...)`。
   强制启用 `eval_mode`、头部/双腕相机和 `rgb/qpos/endpose`；由 client 保存视频，关闭原有评测录像。
   校验仿真左右臂关节数量与 server 的 DOF 相符。
   默认先在关闭 viewer 的情况下执行一次 `play_once()` 获取 `episode_info["info"]`，
   按任务名选择 `description/task_instruction/<task_name>.json`，结合本轮物体/手臂等占位字段、
   `seen/unseen` 类型和 seed 调用 RoboTwin 的 `build_instruction()`，生成本轮任务文本。
   随后关闭专家预运行环境，使用**相同 seed** 重新 `setup_demo()` 恢复初始场景，
   再设置 policy 的步数上限和 instruction。专家观测不发送给 server，不计入 policy 的执行步数。
   指定 `--instruction` 时直接设置该文本，跳过专家预运行和文本生成。
3. 调用 `client.reset(prompt=env.get_instruction(), record_video=True)`，server 清空上一 episode 的缓存并记录预测 latent。
   初次 `get_obs()` 读取当前三视角图像和 joint/endpose，client 将 EEF 从世界系
   `xyz+wxyz` 转为训练坐标系 `xyz+xyzw`，发送 `observation_step=0`。
   server 根据自身 mask 组装 unified55、归一化、更新首帧 KV，再进行 10 次 denoise。
4. 接收完整 32 步预测，检查 native action 的维度、有限值及 `execute_steps`，
   将动作块保存为 `.npy`。server 每轮为整段视频保留完整 32 步对应的两个预测 latent，
   客户端按顺序执行完整动作块。
   `joint` 映射到 `env.take_action(action, action_type="qpos")`；
   `eef` 先由 client 转为世界系 `xyz+wxyz`，再传给 `action_type="ee"`。
5. **每次 take_action 返回后立即 get_obs**，记录执行后的三视角图像。
   对 RGB 数组做独立拷贝，防止下一次渲染复用缓冲区覆盖旧帧。
   初始观测和每次执行后的观测都追加到同一 `video.mp4`，覆盖整个 policy episode，包含终止帧。
   不重复发送上一轮末帧；正常一轮获得恰好 32 组新图像。
6. 将这 32 组图像和最后一组对应的 joint/endpose 发给 server，同时发送
   `executed_steps=32` 和累计 `observation_step=32/64/96/...`。
   server 更新真实观测 KV，以最新 state 重新预测 32 步，继续执行完整 32 步。
   client 负责 EEF 坐标系和四元数换序；unified55 组装、mask、归一化和绝对目标解码由 server 处理。
7. 每次动作后检查 `env.eval_success` 和 `env.take_action_cnt >= env.step_lim`。
   成功或达到上限立即停止，即使当前 32 步尚未执行完；不足 32 帧不补齐、不提交下一次 infer。
   保存 episode 结果，通过 `finally` 关闭环境；下一 episode 使用新 seed、重新 reset。
   遇到 RoboTwin 的 `UnStableError`（物体不稳定）时关闭当前环境，记录跳过的 seed，
   将 seed 加 1 后重新执行同一 episode 的初始化、文本生成和 policy rollout，
   直到可正常模拟；下一次 rollout 先 reset server 缓存。跳过的 seed 不占 `--episodes` 数量。
   退出时关闭 WebSocket，server 清理该连接的缓存。其他初始化或推理错误仍记录后报错退出。

默认 `--state-source command` 保留现有训练数据的 get_obs state 语义。
若选择 `measured`，脚本用 `robot.get_left_arm_real_jointState()` 和
`get_right_arm_real_jointState()` 替换左右关节、夹爪及 `joint_action.vector`，
EEF 仍取本次 get_obs 的 `endpose`；这不自动解决实测 state 与训练 state 的分布差异。
Robotwin 的一次 take_action 内部含路径规划和多次物理推进，因此这里的“一帧”是一个动作目标
执行完成后的观测，并不保证相邻两帧对应固定 1/30 秒；真实闭环时序需要与训练采样方式对齐。

### 输出与测试范围

输出结构示例：

```text
robotwin_sim/
  adjust_bottle/
    demo_clean/
      joint/
        metrics.json
        seed100000/
          video.mp4
          prediction.mp4
          step000000_action_joint.npy
          step000032_action_joint.npy
        seed100001/
          ...
```

每个 `.npy` 保存 server 返回的完整 32 步动作；EEF 文件保留训练坐标系的 `xyz+xyzw`，
在 client 转为世界系执行之前保存。正常执行循环使用完整 32 步，
终止发生在块内时实际执行更少。`metrics.json` 保存任务、文本、server 地址、控制模式、
state 来源和各 episode 的 seed、实际 instruction、文本来源/类型、状态、success、执行步数、
policy 闭环耗时及各次请求的 server 耗时（闭环耗时不包含专家预运行及场景初始化）。
正常结束时，client 请求 server 将首个真实观测 latent 与全部保留的预测 latent 一次解码，
使用与 `EveryNDrawSample` 相同的默认质量和 16 FPS 保存为 `prediction.mp4`。
episode metrics 记录完整预测视频的路径、帧数及 FPS；不再保存逐请求预测 MP4。
最后一轮即使提前终止，预测视频仍保留该轮完整 32 步的预测范围；它与逐步记录的真实观测视频分别保存。
`--video-fps` 只影响真实观测的 `video.mp4`。
`episodes` 保存正常结束或因其他错误退出的 episode；`skipped_seeds` 单独保存
`UnStableError` 尝试的 episode 编号、seed、`skipped_unstable` 状态和错误原因。
动作文件直接按仿真 seed 分目录，保留重试前已保存的动作，避免不同 seed 的结果相互覆盖。
每个组合目录下创建 `seedXXXX/`，在其中保存完整 policy 观测视频 `video.mp4`，
布局与训练 `concat_view` 对齐：头部相机保持原尺寸 `H × W` 放在上方，
左右腕部相机先分别双线性 resize 为 `(H//2) × (W//2)` 和
`(H//2) × (W-W//2)`，再左右拼接到底部。拼接尺寸为 `(H+H//2) × W`；
仅在编码要求偶数尺寸时，在最右侧或最下侧补一列或一行黑像素。
视频包含初始帧及每次 `take_action()` 返回后的观测，正常结束时帧数为实际执行步数加 1，
包括最后不足 32 步的动作块；不包含专家预运行。
默认播放 FPS 为 30，可用 `--video-fps` 调整；这不是逐物理步录像，也不表示仿真实际耗时。
`--render-freq 0` 时同样保存视频。视频边执行边写入，正常结束或异常退出时关闭编码器；
若执行途中发生 `UnStableError`，保留该 seed 已获取的画面，重试写入新 seed 的目录。
初始化阶段尚未获取 policy 观测就失败的 seed 不生成视频。
正常结束的 episode 在 `metrics.json` 中记录 `video_path`、`video_frames` 和 `video_fps`。
真实观测视频使用 FFmpeg `libx264` 编码，MP4 容器标记为
`avc1`，像素格式为 `yuv420p`，启用 `faststart` 以便浏览器播放。
client 环境的 `PATH` 必须能找到支持 `libx264` 的 `ffmpeg`；编码器不可用会报错。
真实观测视频逐帧写入，不在内存中累积整个 episode。
预测视频由 server 在片段结束后整段解码，沿用训练回调的默认 MP4 质量；client 只接收和保存编码后的文件。

自动生成文本通常需要专家预运行，因为多数任务在 `play_once()` 中填充文本占位参数，
仅有 task-name 或初始 `get_instruction()` 不足以获得完整描述；例如 `adjust_bottle` 包括瓶子和手臂。
这会增加每个 episode 的启动时间。脚本沿用 RoboTwin 的模板生成与 seed 选择逻辑，
但不按专家成功率筛选/跳过 seed，也不替代官方 benchmark 的评估流程。
若 RoboTwin 生成失败并退回原始 task-name，脚本报错退出，避免把任务类名当作模型指令；
可以检查任务描述/物体描述文件，或通过 `--instruction` 显式覆盖。
初始化、专家预运行、重建场景或 policy 执行遇到 `UnStableError` 都会记录并换下一个 seed 重试，
不设重试次数上限，也不将该尝试计为模型失败；其他异常仍记录后退出。
当前检查覆盖模拟环境中的动作分块、图像时间顺序、提前成功、实测状态替换、自动文本生成与重建场景、
手动覆盖、连续不稳定 seed 重试、MP4 编解码、视频帧顺序/颜色、提前结束和异常时录像收尾；
未在本机运行真实 RoboTwin 仿真。

## RoboTwin EEF 数据集回放

[`robotwin_eef_replay.py`](../examples/action_pretrain/robotwin_eef_replay.py) 使用训练数据中单个 episode 的 EEF 目标驱动 RoboTwin，并记录仿真观测。这是一个独立入口，直接读取数据集，无需 policy server 或模型 checkpoint；环境加载、相机和视频处理复用现有 RoboTwin client 的辅助函数。

### 数据与环境要求

将 `--dataset-root` 指向包含 `meta/info.json` 和 `data/` 的单个 LeRobot 数据集，例如 `RoboTwin__adjust_bottle/clean`。通过 `--episode-index` 选择 episode，其 `frame_index` 必须从零开始且连续。脚本直接读取物理量表示的绝对 `action_unified` 值，不做归一化、额外时间偏移或按视频步长采样。

数据集必须包含 `episode_index`、`frame_index`、`action_unified` 和 `mask_action`。动作和 mask 数组的宽度必须为 55，且每一行都必须启用双臂 EEF 和夹爪对应的维度：

| 统一维度切片 | 含义           | 要求的表示形式         |
| ------------ | -------------- | ---------------------- |
| `14:21`      | 左臂 EEF 位姿  | `x,y,z,qx,qy,qz,qw`    |
| `21:28`      | 右臂 EEF 位姿  | `x,y,z,qx,qy,qz,qw`    |
| `28:30`      | 左右夹爪目标值 | `[0,1]` 范围内的开度值 |

默认的 `--pose-frame base` 将 EEF 位姿解释为机器人共用基座坐标系中的位姿。脚本从 `meta/info.json` 读取 `custom.eef_enrichment.T_world_base`，使用其中的 `position` 和 `quaternion_wxyz` 字段，将位置和朝向一并转换到世界坐标系。可传入 `--world-from-base x y z qw qx qy qz` 覆盖该变换。如果统一动作中的 EEF 维度已存储世界坐标系位姿，则使用 `--pose-frame world`，并省略 `--world-from-base`。两种模式都按 `xyzw` 解释输入四元数，并为 RoboTwin 导出 `wxyz`。

执行仿真需要可正常运行的 RoboTwin Python 环境，并安装 NumPy、PyArrow、OpenCV 和 msgpack；`PATH` 中还需提供支持 `libx264` 编码器的 FFmpeg。RoboTwin 代码目录必须包含 `scripts/eval_policy_xpolicylab.py`，以提供任务和配置辅助函数。脚本自动将 `--episode-index` 用作仿真 seed，没有单独的 `--seed` 选项。这只是回放约定，并非恢复出的原始采集 seed。当前数据集元数据未保留原始 seed 映射，因此这一约定不能保证还原原始场景布局；精确复现场景还需要匹配任务配置、机器人和资产。

### 不启动仿真器，仅导出动作

在 Cosmos 仓库根目录运行，并替换示例路径。以下示例选择 episode 1，使用仿真 seed 1，结果写入 `episode001/`。默认 episode 索引为 0，使用 seed 0，输出目录为 `episode000/`。

```bash
cd /path/to/cosmos-framework
export DATASET_ROOT=/path/to/robotwin_processed/RoboTwin__adjust_bottle/clean

python examples/action_pretrain/robotwin_eef_replay.py \
  --dataset-root "$DATASET_ROOT" \
  --episode-index 1 \
  --prepare-only \
  --output outputs/robotwin_eef_replay
```

`--prepare-only` 需要 NumPy、PyArrow、msgpack、OpenCV 和 FFmpeg，但不会导入 RoboTwin/SAPIEN。该模式校验并导出动作和数据集参考视频，在 `outputs/robotwin_eef_replay/episode001/` 下写入 `action.npy`、`action_dataset.npy`、`video_data.mp4` 和 `metrics.json`。这只能检查数据准备和视频提取，不能验证仿真执行或任务成功情况。

### 在 RoboTwin 环境中运行回放

沿用上述数据集变量，在已正确安装 RoboTwin 的机器上运行：

```bash
python examples/action_pretrain/robotwin_eef_replay.py \
  --robotwin-root /path/to/RoboTwin \
  --dataset-root "$DATASET_ROOT" \
  --episode-index 1 \
  --task-name adjust_bottle \
  --task-config demo_clean \
  --output outputs/robotwin_eef_replay
```

对于对应的随机化数据，使用 `--task-config demo_randomized`。若 `meta/info.json` 提供了 `custom.source.task`，可省略 `--task-name`。添加 `--max-steps 32` 可将回放限制为前 32 行动作；RoboTwin 可能因任务成功或达到任务步数上限而提前停止。`--video-fps` 设置视频播放帧率，默认为 30。

若希望任务成功后仍执行完所有选中的动作，可添加 `--execute-all-actions`。该选项还会临时扩展 RoboTwin 的任务步数上限，使其覆盖选中的动作数量。`--max-steps` 仍用于决定选取动作序列的前多少步；省略该参数即可回放整个 episode。例如：

```bash
python examples/action_pretrain/robotwin_eef_replay.py \
  --robotwin-root /path/to/RoboTwin \
  --dataset-root "$DATASET_ROOT" \
  --episode-index 1 \
  --execute-all-actions \
  --output outputs/robotwin_eef_replay
```

该选项仅作用于此独立回放脚本。回放期间，脚本临时包装环境实例的 `check_success()`，在记录成功状态的同时避免触发 `take_action()` 提前返回，并防止 `eval_success` 阻止后续动作。退出时（包括异常退出）会恢复原有的成功检查函数和任务步数上限，不修改 RoboTwin 源文件。`metrics.json` 记录 `execute_all_actions`、首次成功的动作序号 `first_success_step`（0 表示执行任何动作前已成功，null 表示未成功），以及是否观测到成功的 `eval_success`。`observations.npz` 中的成功标记在首次观测到成功后持续为 true。所有选中动作执行完毕后，即使任务此前已成功，`stop_reason` 仍为 `actions_exhausted`。发生错误时仍会终止运行并正常记录错误。

每个导出动作包含 16 个值，排列为 `[左臂 xyz, 左臂 qw,qx,qy,qz, 左夹爪, 右臂 xyz, 右臂 qw,qx,qy,qz, 右夹爪]`。脚本调用 `env.take_action(action, action_type="ee")`，并在每个动作执行完成后调用 `env.get_obs()`。首个动作前还会保存初始观测，因此正常记录 N 个已执行动作时，会得到 N+1 帧观测。

### 输出文件

所有文件直接写入 `--output/episode{episode_index:03d}/`，例如 `outputs/robotwin_eef_replay/episode001/`。目录由选中的数据集 episode 索引决定。在同一输出根目录下重新运行同一 episode 会覆盖同名文件。

| 文件                 | 内容                                                                                                           |
| -------------------- | -------------------------------------------------------------------------------------------------------------- |
| `action.npy`         | 世界坐标系中的 EEF/夹爪控制目标，形状为 `[T,16]`；T 为应用 `--max-steps` 后选中的动作数量。                    |
| `action_dataset.npy` | 数据集原始 EEF/夹爪目标，形状为 `[T,16]`，夹爪值按相同顺序交错排列，位姿使用数据集坐标系中的 `xyz+xyzw` 表示。 |
| `observations.npz`   | 初始及后续 EEF 位姿、实测机械臂 qpos、关节/夹爪指令、成功标记、已执行动作和源帧索引。                          |
| `video.mp4`          | 仿真 RGB 观测的 H.264 视频：上方为头部相机，下方并排放置缩放后的左右腕部相机画面。                             |
| `video_data.mp4`     | 从所选数据集 episode 提取的 H.264 参考视频，采用相同的三视角布局和数据集帧率。                                 |
| `metrics.json`       | 数据集/场景设置、运行状态、执行数量、停止原因、任务成功状态，以及位置和旋转跟踪误差的均值与最大值。            |

在 `observations.npz` 中，`eef_world_wxyz` 的形状为 `[N+1,14]`（左臂位姿在前，右臂位姿在后），`measured_arm_qpos` 保存实际机械臂关节位置，`joint_command` 保存包含夹爪指令值的驱动目标。`eval_success` 与 N+1 帧观测对齐。`executed_action` 和 `source_frame_index` 各有 N 行，其第 i 行对应第 i+1 行观测。不单独写出 RGB 图片文件。

每个动作执行后，脚本将 `get_obs()["endpose"]` 从世界坐标系的 `xyz+wxyz` 转回数据集坐标系和 `xyz+xyzw` 表示。在默认的 `--pose-frame base` 模式下，使用构造执行目标时所用 `T_world_base` 的逆变换。随后将转换后的观测与原始数据集动作行比较，不再施加额外时间偏移。`--pose-frame world` 则直接在世界坐标系中比较。`metrics.json` 记录 `tracking_error_frame`、`tracking_error_quaternion_order` 和 `tracking_error_reference`；`tracking_error.left/right` 分别报告位置距离（米）和旋转角度（弧度）的均值与最大值，旋转角度在四元数归一化后按 `2*acos(abs(dot(q_target, q_observed)))` 计算。

NPZ 还保存 `eef_dataset_xyzw [N+1,14]`、`target_eef_dataset_xyzw [N,14]`、`eef_position_error_m [N,2]` 和 `eef_rotation_error_rad [N,2]`。两个误差列分别对应左臂和右臂；第 i 行误差比较第 i 行目标与数据集坐标系中的第 i+1 行观测。初始观测会保留，但不计入跟踪误差统计。

提取参考视频时，脚本结合 `meta/episodes/chunk-*/file-*.parquet` 中各相机的 chunk/file 索引、episode 时间戳，以及 `meta/info.json` 中的 `video_path`，从打包存储的源视频中选取当前 episode。数据集的三个相机视频都必须可用，且帧率必须与数据集一致。视频帧以流式方式写入，采用与 `video.mp4` 相同的头部/腕部相机布局，并保留源头部相机的分辨率。`--video-fps` 仅控制仿真视频的播放帧率；参考视频保留数据集帧率。

参考视频包含所选动作前缀对应的观测，并在可用时额外包含下一帧观测，总帧数不超过该 episode 的已记录帧数。例如，`--max-steps 32` 最多导出 33 帧参考视频。对于包含 L 行动作和 L 帧已记录视频的完整 episode，参考视频有 L 帧；仿真可记录 L+1 帧，因为还会观测最后一个目标执行后的状态。仿真提前成功或达到任务步数上限不会缩短已导出的参考视频。`metrics.json` 记录 `video_data_path`、`video_data_frames`、`video_data_fps` 和 `seed_source="episode_index"`。

补充了 EEF 信息的 RoboTwin 数据以记录序列中的下一帧位姿构造 EEF 目标。RoboTwin 会为每个目标重新规划轨迹，因此此回放不能保证还原原始物理时序或轨迹。视频帧率仅控制播放，不决定仿真或控制频率。回放完成不一定意味着任务成功，应检查 `metrics.json` 中的 `eval_success`、`stop_reason` 和实测跟踪误差。

## Server 的机器人配置

server 负责 `state_mask` 和 `action_mask`。默认从统计文件的
`provenance.sources` 加载两套 mask，要求所有来源完全一致；不会直接把组统计的
`valid` 并集当成当前机器人的 mask。当前 Robotwin 统计满足这一条件。

若统计中没有来源 mask，或者包含不同机器人，可传 `--robot-config /path/to/robot.json`。
JSON 可覆盖 `state_mask`、`action_mask`（各 55 个 bool 或精确的 0/1），以及
`left_arm_dof`、`right_arm_dof`（默认均为 6，取值 1..7）。
覆盖 mask 时必须同时提供两套；两套均省略时，仍从统计的来源记录加载。
server 会校验模板、统计覆盖、关节槽位以及相对动作所需的 state anchor。
reset 和连接断开只清空 episode 数据，保留启动时加载的配置。

只启用左右六关节和夹爪的旧统计对应 `available_control_modes=["joint"]`。
EEF 返回 `None`，不能作为动作执行；只有 action mask 同时覆盖双臂 EEF 和夹爪时，
才会开放 `eef`。输入同时提供 joint 和 EEF 不会改变模型训练过的有效维度。

### EEF 坐标系配置

server 直接接收训练数据坐标系的 `xyz+xyzw`，返回的 EEF action 也使用相同表示。
server 仍执行 mask、校验、归一化、反归一化和绝对目标解码，不读取数据集坐标元信息，
也不要求能够访问统计文件中记录的数据路径。`--robot-config` 只用于 mask 和 arm DOF；
旧配置中的 `eef_pose_frame/T_world_base` 应移到仿真 client 的 `--eef-config`。

数据集 WebSocket replay 和离线执行前缀推理直接使用数据集 EEF 值。
仿真 client 从 RoboTwin 读取世界系 `xyz+wxyz`，发送前转换为训练数据坐标系 `xyz+xyzw`；
收到预测 EEF 后，再转换回世界系 `xyz+wxyz` 给 `env.take_action(..., action_type="ee")`。
关节和夹爪值不参与 pose 转换。其他数据集也应由自己的 client 完成相应转换。

RoboTwin client 可用以下互斥参数之一配置：

- `--eef-dataset-root /path/to/processed/task/clean`：读取该数据集 `meta/info.json` 中的
  `custom.eef_enrichment`，要求 `unified_quaternion="xyzw"`、明确的 `unified_pose_frame`，
  基座系还需 `T_world_base`。
- `--eef-config /path/to/eef.json`：从独立 JSON 加载 `eef_pose_frame`（`base` 或 `world`），
  `base` 必须同时提供 `T_world_base`。示例：

```json
{
  "eef_pose_frame": "base",
  "T_world_base": {
    "position": [0, -0.65, 0],
    "quaternion_wxyz": [0.707, 0, 0, 0.707]
  }
}
```

实际使用时应复制目标数据的变换，并确保仿真基座放置与其一致；变换四元数会归一化。
省略这两个参数时，client 默认训练 pose 位于世界系，只做 `wxyz/xyzw` 换序。
基座系训练数据需要显式传入上述任一参数，例如追加到现有 client 命令：

```bash
python examples/action_pretrain/robotwin_client.py \
  --robotwin-root /path/to/RoboTwin \
  --eef-dataset-root /path/to/processed/task/clean \
  --control-mode eef
```

连接 metadata 声明 `quaternion_order="xyzw"`、`eef_pose_frame="dataset"`。
更新后的 client 会检查这两个字段，避免连接到仍转换世界系 `wxyz` 的旧 server。
实际训练坐标系由数据集及其 client 约定，server 不推断或变换。
独立 `robotwin_eef_replay.py` 不连接 server，仍自行将数据集目标转换成仿真世界系执行。

## 仿真器客户端协议

轻量客户端 `cosmos_framework/inference/causal_action/client.py` 只依赖 NumPy、
msgpack 和 websockets，不加载 Torch、VAE 或模型。也可以直接沿用 LingBot
客户端的 msgpack/NumPy 编解码。

### 客户端入口与使用方式

模型 server 和仿真器客户端分别运行在各自的 Python 环境中，通过 WebSocket 二进制
MessagePack 消息交换 NumPy 图像、状态和动作。server 负责模型推理，客户端负责环境初始化、
采集观测、执行动作和判断任务结束；两端可以部署在不同机器。

- 使用现有 RoboTwin：运行
  [`examples/action_pretrain/robotwin_client.py`](../examples/action_pretrain/robotwin_client.py)，
  启动命令、环境依赖和参数见[联合 RoboTwin 运行真实闭环仿真](#联合-robotwin-运行真实闭环仿真)。
  通过 `--server-uri ws://<server-host>:8000` 指定模型服务地址。
- 接入其他仿真器或真机：复用
  [`CausalPolicyClient`](../cosmos_framework/inference/causal_action/client.py)，
  自行实现设备观测读取、状态/动作转换和控制执行，见下文[接入其他仿真器或真机](#接入其他仿真器或真机)。

`CausalPolicyClient` 的使用顺序为：建立连接并校验 metadata → `reset()` 开始推理会话 →
`infer()` 提交观测 → `actions()` 选择执行模式对应的动作 → 执行并反馈新观测。
需要预测视频时，在 reset 开启 `record_video=True`，结束后调用 `save_prediction_video()`；
退出 `with` 块会关闭连接。

### 设备接口与最小闭环示例

客户端分为通信层和设备适配层，仿真器与真机都通过相同的设备接口接入：

- `CausalPolicyClient` 负责连接 server、发送 reset/观测请求、接收动作和保存预测视频。
- 设备适配层负责调用仿真器或真机 SDK，读取图像与状态、转换坐标、执行动作和判断结束。
- server 负责状态校验与归一化、模型推理和绝对动作目标解码，不直接调用设备 SDK。

下面的 `Device` 是文档中约定的接口，用于说明职责边界，仓库未提供该接口的公共实现。
调用闭环函数前，由调用方初始化场景或真机、设置任务和起始状态；结束后由调用方关闭环境或释放设备。
`client.reset()` 只重置 server 推理会话。

```python
from typing import Protocol


class Device(Protocol):
    def observe(self) -> tuple[dict, dict]:
        """返回同一采样时刻的 (三相机 RGB 字典, 协议 state 字典)。"""
        ...

    def execute(self, action, *, control_mode: str) -> None:
        """将一个协议绝对目标转换为设备命令，按约定控制周期执行后返回。"""
        ...

    def is_done(self) -> bool:
        """判断任务成功、达到执行上限或设备要求停止。"""
        ...
```

`observe()` 返回的图像使用 `cam_high`、`cam_left_wrist`、`cam_right_wrist` 三个键，
各为 `[H,W,3]` RGB `uint8`，并复制可能被后续采集覆盖的缓冲区。
state 使用下文的 `joint_action` / `endpose` 结构；EEF 已转换为训练数据坐标系的 `xyz+xyzw`。
`execute()` 接收的动作也是协议表示，设备坐标转换和控制器插值在该方法内部完成。
对真机而言，返回表示已到下一个动作采样时刻，不要求每个目标都达到静态收敛，
但不能仅把异步命令入队后立即返回并采集下一帧。

以下闭环函数只使用上述接口；替换设备实现时可复用同一套 client/server 交互：

```python
from cosmos_framework.inference.causal_action.client import CausalPolicyClient


def run_episode(
    device: Device,
    *,
    prompt: str,
    server_uri="ws://127.0.0.1:8000",
    control_mode="joint",  # 或 "eef"，连接时校验 server 是否支持
    prediction_path="outputs/causal_action/custom_client/prediction.mp4",
):
    if device.is_done():
        return
    with CausalPolicyClient(server_uri, control_mode=control_mode) as client:
        # client ↔ server：开始推理会话，设置任务文本并开启预测视频记录。
        client.reset(prompt=prompt, record_video=True)

        # 设备：采集初始图像和同一时刻的状态。
        images, state = device.observe()
        response = client.infer(dict(obs=images, state=state, observation_step=0))
        step = 0
        done = False
        while not done:
            # client：从 server 响应选择 joint/eef 绝对目标，默认预测 32 步。
            actions = client.actions(response)
            observations = []
            for action in actions[:response["execute_steps"]]:  # 默认执行完整 32 步
                # 设备：转换并执行一个目标，再采集执行后的新图像和状态。
                device.execute(action, control_mode=control_mode)
                images, state = device.observe()
                observations.append(images)
                step += 1
                done = device.is_done()
                if done:
                    break
            if not done:
                # client ↔ server：只反馈本轮新图像，state 对应最后一张图。
                response = client.infer(dict(
                    obs=observations,
                    state=state,
                    executed_steps=len(observations),
                    observation_step=step,
                ))
        # client ↔ server：请求整段预测视频；client 将返回的 MP4 写入文件。
        client.save_prediction_video(prediction_path)
```

每轮执行 `execute_steps` 个目标，覆盖完整训练 block，随后根据新观测重新规划。
`executed_steps` 表示上一轮执行数量，`observation_step` 表示 episode 内累计执行数量。
任务成功或设备要求停止时直接结束，不提交不足一轮的部分反馈，也不补帧。
`prediction.mp4` 是模型预测视频，设备观测录像由设备适配层或调用方另外记录。

按初始观测为第 0 帧，默认反馈时序为：

```text
第 0 帧三相机图像 + state_0 → 预测 32 步 → 执行 32 步
第 1～32 帧三相机图像 + state_32 → 预测 32 步 → 执行 32 步
第 33～64 帧三相机图像 + state_64 → 继续预测与执行
```

### RoboTwin 设备适配示例

以下适配器实现上述 `Device` 接口，将 RoboTwin API 和坐标转换集中在设备层。
`TASK_ENV` 是调用方已经创建并完成场景初始化的任务环境；
`RobotwinPoseConverter.from_dataset()` 只读取 EEF 坐标元信息，实际观测来自运行中的环境。

```python
from cosmos_framework.inference.causal_action.robotwin_client import (
    RobotwinPoseConverter,
    camera_images,
    episode_finished,
    native_state,
)


class RoboTwinDevice:
    def __init__(self, env, poses, *, state_source="command"):
        self.env = env
        self.poses = poses
        self.state_source = state_source

    def observe(self):
        observation = self.env.get_obs()
        return camera_images(observation), native_state(
            observation, self.env, state_source=self.state_source, pose_converter=self.poses,
        )

    def execute(self, action, *, control_mode):
        if control_mode == "eef":
            action = self.poses.actions_to_world(action)
        action_type = {"joint": "qpos", "eef": "ee"}[control_mode]
        self.env.take_action(action, action_type=action_type)

    def is_done(self):
        return episode_finished(self.env)


poses = RobotwinPoseConverter.from_dataset("/path/to/processed/task/clean")
device = RoboTwinDevice(TASK_ENV, poses)
run_episode(device, prompt=TASK_ENV.get_instruction(), control_mode="joint")
```

RoboTwin 的 `observe()` 将世界系 `xyz+wxyz` 转为训练数据坐标系的 `xyz+xyzw`；
`execute()` 在 EEF 模式下完成反向转换，再调用 `take_action(..., action_type="ee")`。
关节模式直接使用 `action_type="qpos"`。真机接入时实现另一个满足 `Device` 接口的类，
通过真机 SDK 完成这三个方法，再将其实例传给 `run_episode()`。
设备初始化、任务文本来源和资源释放也由调用方相应替换。

### 请求、响应与视频

`state.joint_action` 使用 `left_arm`、`right_arm`、`left_gripper`、`right_gripper`；
关节向量长度等于 server 配置的左右臂 DOF，夹爪可为标量或长度 1 的数组。
`state.endpose` 使用 `left_endpose`、`right_endpose`，各为
`[x,y,z,qx,qy,qz,qw]`，坐标系与训练数据一致；夹爪也可放在 endpose 下作为 joint_action 缺失时的补充。
若两处都提供夹爪，则以 joint_action 中的值为准。RoboTwin get_obs() 的 endpose 需要先由 client 转换。
server 只读取 state_mask 启用的字段；有效字段缺失、形状错误或非法值会报错。
无效字段忽略并清零。EEF 直接按输入值组装为 unified55，再进行校验和归一化。

RoboTwin 的 `get_obs().joint_action` 关节值来自 drive target，并非实际 qpos。
如需实测关节状态，client 应明确调用 `get_*_arm_real_jointState()` 后替换对应字段，
并确认与训练 state 语义一致。EEF 的参考坐标系和 EE/TCP 定义也应与训练一致。
现有 take_action() 包含路径规划和多次物理步推进，不保证一次调用对应固定 1/30 秒。

`reset=True` 可以单独发送，也可以与首次 `obs`、`state` 合并发送。
prompt 可在 reset 或首次观测请求中提供，随后沿用。client 不发送 mask 或 55D state。
可选 `control_mode` 在 reset 时传 `joint` 或 `eef`，只用于支持性校验，不改变模型 mask；
episode 内修改任务或执行模式需要 reset。`obs` 同时支持三个相机的 `[T,H,W,3]`
数组字典，以及长度为 T 的相机字典列表。第一轮 T=1，之后严格 T=`execute_steps`（默认 32）。
可选 `observation_step` 用于拒绝重复或乱序反馈，建议真实 client 始终发送。

响应包含 `action`、`available_control_modes`、`execute_steps`（默认 32）、`observation_step`、
`revision` 和 `server_timing.infer_ms`，不逐轮返回解码的 RGB 视频。
metadata 中 `video_stride`（默认 4）和 `video_fps`（默认 7.5）描述模型采样时间网格；
`video_decode_mode="whole_sequence"` 表示支持整段解码，`prediction_video_fps=16` 是保存的播放 FPS，
与训练回调一致；因此保存视频的播放时长可能与 action 的控制时长不同。
需要视频时，在 reset 中传 `record_video=True`，结束时发送 `finish_video=True`。
server 将首个真实观测 latent 和每轮前 `execute_steps` 步对应的预测 latent 一次解码，并返回
`video_mp4`（已编码的 MP4 bytes）、`video_frames` 和 `video_fps`。
`client.save_prediction_video(path)` 封装该请求并将 bytes 原样写入文件，client 不需安装 Torch 或 VAE。
完成保存后清空记录；下一 episode 必须重新 reset 并开启记录。
`action` 是含以下两个键的字典：

- `joint`：`float32 [32,left_dof+right_dof+2]`，每行顺序为左关节、左夹爪、右关节、右夹爪。
- `eef`：训练数据坐标系 `float32 [32,16]`，每行顺序为左 xyz+xyzw、左夹爪、右 xyz+xyzw、右夹爪。

未训练的执行模式值为 `None`；两种模式都有效时同时返回，由 client 选择。
动作已经反归一化并恢复成绝对目标，client 不再加初始位姿，也不需要知道 unified55 槽位。
启动时加 `--return-unified-action` 可额外返回内部 `action_unified [32,55]` 用于调试，
该数组保持训练坐标系的 `xyzw` 表示，不做世界系转换。
启动时加 `--return-video-latents` 可返回本轮预测的 `video_latents`
（`[1,C,2,H_latent,W_latent]`，不包含观测前缀）；默认仍联合预测 video/action，
该选项只控制是否额外传输本轮 latent，不影响 server 为整段视频记录所选 latent。
视频保存统一使用 `model.decode()` 和训练回调的 `save_img_or_video()`，不逐轮重置 decoder 后拼接像素帧。
预测视频不进入真实观测的 encoder cache 或 Transformer KV cache。

请求错误返回 `error` 和 `reset_required`。输入校验失败可以修正后重发；
VAE/prefill/denoise/视频解码失败后必须 reset，避免继续使用部分更新的缓存。
连接断开会自动清空 episode；第二个同时连接的客户端会被拒绝。

### 接入其他仿真器或真机

自定义客户端可以只导入 `CausalPolicyClient`，无需调用 RoboTwin 的环境加载和状态辅助函数。
保留上述 `run_episode()`，实现目标设备的 `Device` 接口：

| 设备接口                            | 自定义客户端需要实现的内容                                                                         |
| ----------------------------------- | -------------------------------------------------------------------------------------------------- |
| `observe()`                         | 读取并对齐三相机 RGB 和状态，返回协议字典；将 EEF 转为训练数据坐标系，复制可能被覆盖的图像缓冲区。 |
| `execute(action, control_mode=...)` | 将协议绝对目标转换为设备命令，匹配坐标、关节顺序、单位、零位、正方向和夹爪开度，并管理控制周期。   |
| `is_done()`                         | 查询任务成功、执行次数上限及设备停止条件。                                                         |

创建和关闭仿真环境、真机连接，以及设置任务与起始状态，由调用方在 `run_episode()` 外完成。

例如，自定义状态组装函数可以返回以下结构；变量表示从设备读取并转换后的值：

```python
state = {
    "joint_action": {
        "left_arm": left_joint_values,
        "right_arm": right_joint_values,
        "left_gripper": left_gripper_value,
        "right_gripper": right_gripper_value,
    },
    "endpose": {
        "left_endpose": left_pose_dataset_xyzw,
        "right_endpose": right_pose_dataset_xyzw,
    },
}
```

关节数组长度由 server 配置的左右臂 DOF 决定；EEF 各含 7 个值，夹爪为标量或长度 1 的数组。
只需提供 `state_mask` 启用的字段，初始请求通过
`client.infer(dict(obs=images, state=state, observation_step=0))` 发送，
其中 `images` 为三相机数组字典。后续可以使用长度 32 的相机字典列表，
并提供 `executed_steps=32` 和累计的 `observation_step`，state 始终对应最后一个图像时刻。
状态应明确采用实测位置还是控制指令目标，并与训练语义匹配。

EEF 状态发送前转换为训练数据坐标系的 `xyz+xyzw`；收到 EEF 动作后转换为设备控制器所需的表示。
server 不执行设备坐标转换。动作已完成反归一化和绝对目标解码，客户端不再重复处理；
若设备控制器只接受增量命令，则由客户端根据设备状态和返回的绝对目标完成转换。

当前服务端仍使用
[`RobotwinAdapter`](../cosmos_framework/inference/causal_action/robotwin.py)，
支持左右臂各 1～7 个关节、双臂 EEF 和夹爪，所选执行模式的 action mask 必须覆盖双臂及夹爪。
结构兼容时可沿用现有协议，通过 `--robot-config` 配置 DOF 和 mask，并提供匹配的训练统计文件。
单臂、灵巧手或移动底盘等结构需要扩展 server adapter 和对应协议；相机数量或布局不同时，
还需适配 `policy.py` 的相机字段校验与 `online.py` 的视觉预处理，并与训练输入保持一致。
客户端连通不代表已有 checkpoint 适用于新设备；修改 mask 或 DOF 不能替代对应的训练数据和模型能力。

真机客户端还需要自行管理控制周期及相机/状态时间对齐。`executed_steps` 统计模型动作目标数量，
不统计控制器内部插值次数或仿真物理步数。默认每 32 个动作反馈一次，期间必须采集对应的新图像；
仅下发异步控制命令不代表这些动作已经按采样时序执行完毕。
反馈必须覆盖一个完整训练 block，不能只修改客户端循环就改为半块或任意步数。
上述示例同步等待每次推理结果；固定频率控制、推理等待期间的行为和设备停止处理需要客户端实现。
server 的 `--fps` 描述模型时间网格，不会替客户端调度控制周期。

## 缓存与时间对齐

1. reset 清空流式 VAE 特征、所有层与 CFG 分支的 text/video KV、时间坐标及 state。
2. 首次三视角按训练 `concat_view` 布局拼接：head 在上，左右 wrist 在下。
   使用同一 resize/reflection-padding 和归一化，单帧 VAE 编码后，以零 timestep
   prefill 各层 observed video KV。
3. 以当前收到的 state 和 observed KV 为条件，10 次 UniPC 联合预测未来
   32 个 action 和 2 个 video latent。相对 action 使用当前 state 为 anchor 解码；
   gripper/hand 字段保持模板定义的绝对目标语义。
4. 执行完整 32 个 action 后，取新观测的第 4、8、12、16、20、24、28、32 张（1-based）。
   保留 Wan2.2 因果卷积特征，按两个 4 帧 chunk 顺序编码出 L1、L2；随后联合 prefill，
   两帧在每层互相可见，所有层完成后一次提交。后续依次处理 [L3,L4]、[L5,L6]。
   首帧 L0 始终独立 prefill；不会将预测 latent 写入历史。每帧保留独立的绝对 mRoPE
   时间坐标，预测从最新观测边界重新锚定 state 和 action 时间位置。
5. 保留最近 `history * latent_frames_per_training_block` 个 observed latent KV，
   默认 64 帧。prefill 先读取完整旧历史与当前 block，提交后再淘汰过期 KV，
   不切断保留的训练 block，并保留绝对 mRoPE 时间坐标。state 只作用于当前预测，
   不进入 visual history。预测的 action/video 都不写入 observed KV。

训练块是 32 action / 2 latent；在线每次反馈完整 block，clean 视频块内双向可见、
块间因果，与训练的 clean attention 规则一致。当前实现拒绝 `execute_steps < actions_per_block`，
避免提前提交半块 KV。真实闭环效果需要后续仿真评估。

实现使用 compact action teacher-forcing pack；观测 prefill 只提交独立首帧或完整 clean block，
预测处理最新观测和未来两个 latent 的槽位，历史各层 KV 跨请求保留。text 及 compact pack 仍会执行投影/FFN，
尚未做固定 state 的跨 denoise 步投影缓存或 CUDA graph 优化。
当前服务仅支持 Wan2.2 VAE、UniPC、单 rank；训练基础代码和离线完整块 session 保持原有接口。
