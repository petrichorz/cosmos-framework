# Causal GenKVCache 推理使用说明

本文说明如何使用 causal checkpoint 对 BridgeData2 验证集中的 Text+Image 样例做 block-autoregressive 视频生成。推理只使用带 KV cache 的 causal 路径；用户配置生成 block 数量 `B`、每个 block 的 VAE latent frame 数 `S`，不配置 `num_frames`。

## 已验证配置

- 环境：`cosmos-framework-causal`
- checkpoint：`/mi/data2T/Embodied-AI/codes/cosmos_ascend/outputs/causal-edge-training/cosmos3/causal_sft/vision_causal_edge/checkpoints/iter_000000400`
- 训练配置：从 checkpoint 所属 run 的 `config.yaml` 加载，不使用 checkpoint 目录内的配置
- 验证样例：`episode_002345_clip000`
- 参数：`B=16`、`S=2`、`K=16`、`num_steps=35`、`guidance=1.0`
- CFG：关闭。`guidance=1.0` 时只创建 conditional GenKVCache
- 输出：129 帧、256×256、5 FPS 的 MP4

长度换算如下：

```text
latent frame 数 = 1 + B * S
pixel frame 数  = 1 + 4 * B * S
```

首个 `1` 是独立的 first-frame condition block。Wan2.2 causal VAE 的时间压缩率为 4，因此本次 `B=16, S=2` 得到 33 个 latent frame，并解码为 129 个像素帧。

## 直接运行

在仓库根目录执行：

```bash
examples/run_causal_gen_kv_cache_bridge_val.sh
```

脚本内部通过 `conda run -n cosmos-framework-causal` 启动，所以无需预先 `conda activate`。如果希望手动激活环境，也可以：

```bash
conda activate cosmos-framework-causal
examples/run_causal_gen_kv_cache_bridge_val.sh
```

默认输出视频位于：

```text
outputs/causal_gen_kv_cache/iter_000000400_bridge_val_002345_b16_s2/
  causal_i2v/episode_002345_clip000/vision.mp4
```

同一目录还会保存：

- `sample_args.json`：框架最终解析后的完整推理参数
- `sample_outputs.json`：输出状态和产物记录
- `console.log`、`debug.log`：运行日志
- `input/episode_002345_clip000.json`：由验证集 JSON 转换出的 causal 输入

## 可配置参数

所有参数都通过环境变量覆盖。例如，用样例 2345 在 NPU 1 上生成 8 个 block：

```bash
SAMPLE_ID=002345 \
NUM_BLOCKS=8 \
BLOCK_SIZE=2 \
HISTORY_BLOCKS=8 \
DEVICE_ID=1 \
OUTPUT_ROOT=/tmp/causal_b8 \
examples/run_causal_gen_kv_cache_bridge_val.sh
```

主要变量：

| 变量 | 默认值 | 含义 |
|---|---:|---|
| `NUM_BLOCKS` | `16` | 顺序生成的 block 数 `B` |
| `BLOCK_SIZE` | `2` | 脚本默认每个 block 联合去噪的 VAE latent frame 数 `S`；模型参数自身默认仍为 1 |
| `HISTORY_BLOCKS` | `16` | GenKVCache 最多保留的 clean block 数 `K` |
| `NUM_STEPS` | `35` | 每个 block 的 diffusion steps |
| `GUIDANCE` | `1.0` | CFG scale；`1.0` 表示关闭 CFG |
| `SEED` | `1` | 整次推理的随机种子 |
| `DEVICE_ID` | `0` | 对脚本可见的 Ascend NPU 编号 |
| `USE_TORCH_COMPILE` | `0` | `1` 开启 torch compile；Ascend 首次调试建议保持关闭 |
| `SAMPLE_ID` | `002345` | BridgeData2 验证集 episode 编号，不带 `episode_` 前缀；`2345` 和 `002345` 均可 |
| `CHECKPOINT_ROOT` | iter 400 路径 | checkpoint 根目录；其下必须存在 `model/.metadata` |
| `CONFIG_FILE` | run 的 `config.yaml` | 模型配置文件 |
| `VAL_ROOT` | BridgeData2 val 路径 | 验证集根目录 |
| `OUTPUT_ROOT` | 仓库 `outputs/causal_gen_kv_cache/...` | 输出根目录 |

`BLOCK_SIZE` 必须落在 checkpoint 训练时的 block-size 范围内；当前 checkpoint 的训练范围是 1 到 4。`HISTORY_BLOCKS` 当前最大为 16。

## 输入转换

脚本读取：

```text
$VAL_ROOT/inference_prompt_i2v/episode_${SAMPLE_ID}_clip000.json
$VAL_ROOT/images/episode_${SAMPLE_ID}_clip000.jpg
```

它保留验证集中的结构化正、负文本 prompt 及分辨率/FPS信息，删除源 JSON 的 `num_frames`，并写入：

```json
{
  "model_mode": "image2video",
  "causal_num_blocks": 16,
  "causal_block_size": 2,
  "causal_history_blocks": 16,
  "num_steps": 35,
  "guidance": 1.0
}
```

不要同时提供 `num_frames`。causal 推理由 `1+B*S` 计算目标 latent 长度，统一推导 VAE 解码后的像素帧数。

## Ascend 注意事项

脚本显式设置 `COSMOS_DEVICE=npu`。这一项不能省略：Wan VAE 构造时会读取 Cosmos 的设备标志；仅依赖 `transfer_to_npu` 无法可靠转换通过变量传入的 `device="cuda"`，会导致 `PyTorch is not linked with support for cuda devices`。

脚本默认添加 `--no-guardrails`，因为当前推理环境没有安装 guardrail 的 `better_profanity` 可选依赖。该选项只关闭输入/输出安全过滤，不改变 causal 生成或 KV cache 逻辑。

如需快速排查端到端环境，可先运行：

```bash
NUM_BLOCKS=2 NUM_STEPS=2 OUTPUT_ROOT=/tmp/causal_smoke \
examples/run_causal_gen_kv_cache_bridge_val.sh
```

两 block smoke test 会同时覆盖 first-frame prefill、首个生成 block 的独立 prefill/commit，以及下一 block 的 KV cache 读取。
