# 统一因果分块与条件训练实验

日期：2026-09-18 至 2026-09-19。分支：`feat/causal-uniform-conditioning`。
结论：统一分块、条件训练、KV/无 KV 推理及在线采样已实现；2026-09-18 的真实数据 60 步训练 loss 正常下降，最终代码于 2026-09-19 完成回归验证，未重新进行实机训练。未评估视频质量。

## 语义与范围

| 项目     | 本次实现                                                                                      |
| -------- | --------------------------------------------------------------------------------------------- |
| 分块     | 固定 uniform，第 0 帧进入第一块；尾块允许不足；无布局选择参数                                 |
| 初始化   | 沿用现有加载逻辑从 Edge-DCP 初始化，不转换 checkpoint                                         |
| 条件比例 | 主实验 `{0:0.1,1:0.9,2:0.0}`，T2V 10%、TI2V 90%；TV2V 支持但暂不采样                          |
| 条件单位 | 配置键是 VAE latent 帧数；Wan 时间压缩率为 4 时，1/2 个 latent 条件分别对应 1/5 个 RGB 前缀帧 |
| 时间步   | 每块一个随机时间步；条件帧使用 clean 输入和 t=0 的时间嵌入                                    |
| 注意力   | 保留 UND、clean/noisy 双流与最近 H 个 clean 历史块；noisy 当前块不能读取当前 clean 块         |
| Loss     | 仅监督 noisy 流的非条件位置；uniform 按有效目标元素数归一化，条件位置仍可通过 KV 影响其他预测 |
| 验收     | 功能与数值正确、真实数据 loss 下降、无 NaN/Inf；不以视频质量作为验收条件                      |

B=3 的唯一分块方式：`[0,1,2][3,4,5][6]`。
History 仍以 block 为单位。训练仍读取真实 clean 历史，不属于自生成历史训练。

## 推理与在线采样

| 场景             | 处理方式                                                                                         |
| ---------------- | ------------------------------------------------------------------------------------------------ |
| T2V              | 第一块直接从噪声生成，没有虚构首帧                                                               |
| 首块部分为条件   | 条件和噪声共同参加块内注意力；每次模型评估与最终输出固定条件 latent                              |
| 完整条件块       | 直接 clean prefill，不执行扩散；覆盖 B=1 时两帧条件跨块                                          |
| KV cache         | 完成整个块后，以 t=0 重新前向并提交各层 KV；窗口按逻辑块淘汰                                     |
| Without KV cache | 完整重算已生成前缀，每层保持相同块因果及 H 窗口，作为正确性参考                                  |
| CFG              | 正负文本各自维护缓存，两条分支保留相同图像条件                                                   |
| 长度             | 总 latent 长度为 N×B；RGB 长度由 tokenizer 换算                                                  |
| CLI              | `MODEL_MODE=text2video/image2video/video2video`；`USE_KV_CACHE=1/0`；图片或视频使用 `IMAGE_PATH` |
| 5k 在线采样      | 沿用原 5000 iter 回调；支持 C=0/1/2 和缓存开关，固定样本/seed，结束后恢复训练状态和 RNG          |

无缓存参考不能只截取最近 H 个原始 latent：缓存的高层历史表征已吸收更早的信息，直接截断重算会改变结果。
推理保留已有 H 最大为 16、单样本、连续前缀条件的约束；不支持稀疏未来帧条件。KV 与无 KV 路径使用同一分块规则。

## 真实训练结果

环境 `cosmos-framework-py312`，4×Ascend 910B3；数据 `egosuite_demo_v1`，从 Cosmos3-Edge-DCP 初始化。
480p 档位，抽查实际桶 **宽 736×高 544**；固定 15 FPS，最长 30 秒分片，每卡单样本。
抽查片段 333/349/393 帧，并非每个片段都满 30 秒。B 随机 1–4，H 随机 1–64，TND budget=131072。
BF16 AMP、FP32 主参数、full activation checkpoint；学习率 1e-4，warmup 50 步，主实验关闭 EMA。

| 步数  | 全局平均 loss 的区间均值 | 区间最小–最大  |
| ----- | ------------------------ | -------------- |
| 1–10  | 9.2786                   | 3.8950–16.3499 |
| 11–20 | 2.7645                   | 2.2746–3.5726  |
| 21–30 | 2.3005                   | 1.9232–2.8120  |
| 31–40 | 2.2491                   | 1.8937–2.5448  |
| 41–50 | 2.1451                   | 1.8766–2.4684  |
| 51–60 | 2.1961                   | 1.9137–2.5049  |

首步 **15.9619**，末步 **2.1816**。采用 W&B 中跨卡归约后的 loss，避免把 rank 0 的局部 loss 当全局值。
训练正常退出，已保存 `iter_000000060`；已记录梯度范数有限，未出现 NaN/Inf 或训练中断。
5–59 步可用计时 54 条，中位数 30.18 秒/步；第 51 步计时器切换未输出，第 60 步含保存 checkpoint。
运行期间有同机调试负载，无旧版本配对基线，**不据此声称性能收益**；主实验未采集可靠的峰值显存。

## 数值与功能验证

| 实验             | 设置                                                                                     | 结果                                                                                     |
| ---------------- | ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| CPU 回归         | 分块、时间步、条件、loss、配置兼容、在线采样等                                           | 272 项通过；1 项下载外部 checkpoint 的测试因离线环境未运行                               |
| KV/无 KV         | 两层真实 MoT；B=1/2/3/4，H=1/2，C=0/1/2，CFG=1/3；FixedStep/UniPC 多步，含窗口淘汰及尾块 | 48 组通过，FP32 atol/rtol=2e-5；含与实际模型一致的 UND K-normalization 配置              |
| TND/Dense 正反向 | uniform，变长双样本 packing，4 组 B/H ×3 个 budget                                       | FP64 CPU 输出及 Q/K/V 梯度均通过 1e-11 容差                                              |
| NPU TND/Dense    | BF16 TND 对比 FP32 Dense；B/H=1/1、2/2、4/16                                             | 输出相对 L2 0.187%–0.199%；梯度 0.270%–0.332%                                            |
| 条件与 loss      | C=0/1/2，t=0 条件嵌入、有效目标归一化                                                    | 条件位置不贡献直接去噪 loss，非条件位置梯度正常                                          |
| 在线采样实机     | 4 卡，256 档位，B=2/H=1/N=3，UniPC 2 步；reg TV2V 无 KV、EMA T2V 有 KV                   | 连续 3 个训练 step 触发回调并恢复训练；峰值 allocated 16.68 GiB、reserved 18.73 GiB      |
| 独立 TI2V        | 第 60 步 checkpoint，真实数据首帧，单卡，256 档位，B=2/H=1/N=3，2 步                     | 输出 320×256、21 帧视频及最终 prompt；加载/生成/保存共 89.79 秒，峰值 allocated 8.58 GiB |

在线采样测试将原 5000 步触发临时改为每步，未实际训练到 5000 步。
独立推理使用两步仅验证流程，不用于判断视频质量或正式生成速度。

### 真实大模型 KV 对照

同一输入/seed、B=2、H=1、3 块、UniPC 2 步，比较最终 latent 的相对 L2。
实际 Cosmos3-Edge 全模型，4 卡；reg 用刚执行第 1 个训练 step 的权重，T2V 用该步 EMA 权重。
FP32 与 BF16 各自成对比较，不把两种精度间的权重更新差异当 cache 误差。

| 条件          | CFG | FP32 KV/无 KV 相对 L2 | BF16 KV/无 KV 相对 L2 |
| ------------- | --- | --------------------- | --------------------- |
| TI2V：C=1     | 1   | 1.94e-6               | 2.73%                 |
| TV2V：C=2     | 1   | 1.57e-6               | 1.54%                 |
| TI2V：C=1     | 3   | 1.48e-5               | 10.30%                |
| TV2V：C=2     | 3   | 4.96e-6               | 5.14%                 |
| T2V：C=0，EMA | 1   | 1.36e-6               | 1.31%                 |

**FP32 对照支持语义一致，BF16 不保证相同轨迹。** 首次模型评估输入完全相同；FP32 单次 velocity 相对误差约 1e-6，BF16 的 TI2V/TV2V 已为 1.51%/1.09%。
不同序列形状下的低精度计算及后续采样会累积差异，CFG 可放大；尚未定位到具体哪一个算子的舍入贡献。
尝试统一 cached attention 为 TND 后，BF16 上述误差不变，已撤回该无效修改。未通过放宽阈值宣称 BF16 等价。

## 调试记录

| 现象                                  | 处理                                                                    | 结果                             |
| ------------------------------------- | ----------------------------------------------------------------------- | -------------------------------- |
| 100k token packing 反向 OOM           | 保持分辨率/时长，改每卡单样本，启用 expandable segments                 | 60 步完成                        |
| HCCL 监听端口冲突                     | 设置独立 `HCCL_NPU_SOCKET_PORT_RANGE`（默认 16666 冲突）                | 训练、在线采样均完成             |
| CPU 测试 UND 路径调用 NPU             | 测试中切换 SDPA                                                         | 生产 attention 路径不变          |
| conda 重激活误引 Python 3.13 Torch 库 | 本机先修正环境变量；启动脚本增加可选 `PYTHON_BIN`，直接使用已激活解释器 | TI2V bash 验证通过，未改共享环境 |

## 复现与产物

在仓库根目录激活 `cosmos-framework-py312` 和 CANN，设置 `DATASET_PATH`、`BASE_CHECKPOINT_PATH`、`COSMOS3_EDGE_PROCESSOR_PATH`、`WAN_VAE_PATH`。
本机数据路径为 `/mnt/sfs_turbo/public/datasets/egosuite_demo_v1`，初始权重为 `/mnt/sfs_turbo/public/ckpts/Cosmos/Cosmos3-Edge-DCP`。

```bash
# 480p / 30s / 15FPS，4 卡 60 步；OUTPUT_ROOT 可指定输出位置
bash tools/run_uniform_conditioning_experiment.sh
# 真实模型 FP32 KV/无 KV 对照，同时验证 reg/EMA 在线回调
bash tools/run_uniform_conditioning_audit.sh
# 小形状 NPU TND 正反向与 cache 数值验证
python -m tools.validate_uniform_conditioning
```

BF16 对照可用 `CACHE_PAIR_RTOL=''` 关闭 FP32 专用阈值，并传入
`model.config.precision=bfloat16 model.config.parallelism.fsdp_mixed_precision_enabled=true`；仍检查有限性并记录误差。
启动训练模板为 `examples/launch_pretrain_vision_causal.sh`；新模式配方为 `examples/toml/sft_config/vision_pretrain_edge_causal_tnd.toml`。
推理使用 `examples/run_causal_ti2v.sh`，填写 `CHECKPOINT_ROOT` 和对应运行的 `CONFIG_FILE`，权重加载接口不变。

| 本机产物（相对仓库根目录）                                                                    | 内容                                                  |
| --------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| `outputs/uniform_conditioning/train_history.json`                                             | 全部 60 步全局 loss 与已记录梯度                      |
| `outputs/uniform_conditioning/train_console_15fps.log`                                        | 480p 主训练完整日志                                   |
| `outputs/uniform_conditioning/train/cosmos3/mot_causal_fsdp/vision_causal_edge_tnd_pretrain/` | config.yaml、W&B 离线记录、checkpoints/iter_000000060 |
| `outputs/uniform_conditioning/online_smoke.log`                                               | 3 步训练中在线采样及恢复训练                          |
| `outputs/uniform_conditioning/online_audit_fp32_final/audit_rank*.jsonl`                      | 全模型 FP32 成对及逐步误差                            |
| `outputs/uniform_conditioning/online_audit_bf16_final/audit_rank*.jsonl`                      | 全模型 BF16 成对及逐步误差                            |
| `outputs/uniform_conditioning/npu_numerical.json`                                             | NPU TND 正反向误差及小模型 cache 对照                 |
| `outputs/uniform_conditioning/cli_ti2v/ti2v/`                                                 | vision.mp4、model_input_prompt.txt、sample_args.json  |

主实验更改了 loss 分母，不能直接与旧训练 loss 比大小。本次证明可训练、条件语义及推理链路可用；长训练稳定性和生成质量留待后续实验。

## 最终实现与验证边界

训练、KV/无 KV 推理与在线采样均从第 0 帧按 B 分块，无布局选择字段。
Teacher forcing 的条件帧固定使用 t=0 嵌入；其他 attention 模式保持既有行为。
原有 Edge-DCP 加载路径、参数选择和条件比例不变，无 checkpoint 转换或迁移逻辑。

2026-09-19 的主回归 259 项通过；更新旧分块编号断言后，相关 86 项测试全部通过（含新增的非 teacher-forcing 条件嵌入保护测试）。
1 项外部 checkpoint 下载测试在离线环境下未运行。48 组 KV/无 KV 多步对照包含在通过的回归中。
原启动脚本 dry-run 成功生成配置，确认 Edge-DCP load_path、480p/30s、15 FPS、条件比例保持不变。
回归日志与生成的配置位于 `outputs/uniform_only_20260919/`。
当时 8 张 NPU 均被其他训练占用，显存占用约 59–64 GiB，因此未对最终代码重新启动实机训练。
上面的 60 步训练及实机数值结果属于 2026-09-18 的实验，不代表对最终代码的新增实机测量。
