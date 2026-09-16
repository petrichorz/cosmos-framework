# TND profile_local 启动脚本

参考 cookbook 的 `launch_sft_vision_edge_profile_local.sh`，保留“本机配置 wrapper → 训练 launcher → TOML”结构。入口位于当前 framework 仓库，不修改另一个 cosmos checkout。

- [本机配置入口](../../examples/launch_sft_vision_edge_tnd_profile_local.sh)
- [训练 launcher](../../examples/launch_causal_edge_tnd_profiling.sh)
- [TND 配方](../../examples/toml/sft_config/vision_causal_edge_tnd_profile.toml)

配方从参考脚本使用的 `vision_causal_edge_profile.toml` 复制，仅改变 attention mode 为 `grouped_tnd` 和实验名称。保留 EMA 开启、2000 步、500 步保存周期、full checkpoint、61 秒最大视频时长、block=1–4/history=1–64 等设置。分辨率继承该实验配置，不强制覆盖为 480；脚本名称带 profile 也不表示默认开启 profiler。

## 使用

先在目标机器激活 Python/torch_npu 环境并加载 CANN，再在 framework 仓库中运行：

```bash
export DATASET_DIR=/data/Cosmos3-DROID/success
export CHECKPOINT_DIR=/models/Cosmos3-Edge-DCP
export COSMOS3_EDGE_PROCESSOR_PATH=/models/Cosmos3-Edge
export VAE_PATH=/models/Wan2.2_VAE.pth
export OUTPUT_ROOT=/outputs/tnd_profile_run1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4

# 仅打印命令，无需此时已准备好数据或权重。
PRINT_ONLY=1 bash examples/launch_sft_vision_edge_tnd_profile_local.sh

# 加载配置并写出 config.yaml，不训练。
DRY_RUN=1 NPROC_PER_NODE=1 bash examples/launch_sft_vision_edge_tnd_profile_local.sh

# 开始训练。
bash examples/launch_sft_vision_edge_tnd_profile_local.sh
```

保持参考脚本的环境变量接口。默认设备及进程数都是 8；使用更少设备时同时修改 `ASCEND_RT_VISIBLE_DEVICES` 和 `NPROC_PER_NODE`。多机使用相同的 `NNODES`、`MASTER_ADDR`、`MASTER_PORT`，各节点设置不同的 `NODE_RANK`。

若需要之前的 480p、30 秒分片，显式追加配置：

```bash
bash examples/launch_sft_vision_edge_tnd_profile_local.sh \
  'dataloader_train.dataloader.datasets.video.dataset.resolution="480"' \
  dataloader_train.dataloader.datasets.video.dataset.max_video_duration_s=30.0 \
  dataloader_train.dataloader.datasets.video.dataset.long_video_policy=split \
  dataloader_train.dataloader.datasets.video.dataset.video_window_overlap_s=2.0
```

其他 Hydra 参数同样追加在末尾并优先于默认值。若改训练总步数，也按需要同步修改 `scheduler.cycle_lengths`。配置 dry-run 只验证配置加载，不验证模型权重加载、显存容量或训练收敛。

## 跨机器调整

- 不调用本机 conda 路径，不修补其他机器的 `LD_LIBRARY_PATH`；环境由调用方准备。
- 默认数据和权重路径位于 checkout 的 `examples/data` 和 `examples/checkpoints`，均可通过环境变量覆盖；用户传入相对路径以调用目录为准。
- 使用当前脚本所属 framework checkout，避免误用环境中另一份代码。
- 强制关闭 `COSMOS_GEN_BSA64`，无需编译 BSA。默认 `COSMOS_BSA64_BUCKETS=0` 使用原始桶，设为 1 仅调整尺寸。
- 不复制 optimized_local 的融合开关，也不强制修改训练步数、分片或 profiler 设置。
- 路径缺失时明确报错，不自动下载/转换权重；日志保存为 `OUTPUT_ROOT/launcher_rank<NODE_RANK>.log`。
- 不主动开启实验计时或跳过最终 checkpoint；若当前 shell 曾设置 `COSMOS_ASCEND_BENCHMARK` 或 `COSMOS_PERF_SKIP_FINAL_CHECKPOINT`，正式训练前将其取消或设为 0。

验证：Bash 语法检查；使用真实环境完成配置 dry-run；模拟 launcher 检查从任意目录启动、含空格路径、多机参数、尾部覆盖、禁用 BSA 以及训练失败状态传递。未新增多机或完整训练性能验证。
