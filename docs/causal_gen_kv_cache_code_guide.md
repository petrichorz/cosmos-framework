# Causal GenKVCache 改动总结与代码导读

## 1. 这版代码解决什么问题

这版推理实现的是唯一一种输入形式：**Text + Image → Video**。

输入图片先成为独立的 condition block 0，模型随后按 block 自回归生成视频：

```text
图片 C0 → 生成 block 1 → 生成 block 2 → ... → 生成 block B
```

用户不提供 `num_frames`，只提供：

- `B = causal_num_blocks`：循环生成多少次；
- `S = causal_block_size`：每次联合生成多少个 VAE latent frame；
- `K = causal_history_blocks`：KV cache 最多保留多少个已经完成的 block。

目标 latent frame 数为：

```text
T_latent = 1 + B * S
```

其中 `1` 是独立的 first-frame block 0。以 Wan2.2 causal VAE 的时间压缩率 4 为例，像素帧数为：

```text
T_pixel = 1 + 4 * B * S
```

例如 `B=16, S=1`：模型循环生成 16 次，每次生成 1 个 latent frame，最终得到 17 个 latent frame和 65 个像素帧。

## 2. 当前版本改了哪些地方

### 2.1 推理参数和长度契约

文件：`cosmos_framework/inference/args.py`

新增三个参数：

| 参数                    | 默认/约束                | 作用                              |
| ----------------------- | ------------------------ | --------------------------------- |
| `causal_num_blocks`     | causal 推理必须显式提供  | 循环生成的 block 数 `B`           |
| `causal_block_size`     | 默认 `1`                 | 每个 block 的 latent frame 数 `S` |
| `causal_history_blocks` | 默认 `16`，范围 `[1,16]` | cache 历史窗口 `K`                |

当 checkpoint 配置的 `causal_training_strategy == "teacher_forcing"` 时：

1. 拒绝用户显式提供 `num_frames`；
2. 用 `1+B*S` 计算 latent 长度；
3. 再根据 VAE 时间压缩率计算内部需要准备的像素帧数。

这样长度只有一个来源，不会出现 `num_frames` 与 `B/S` 互相矛盾。

文件：`cosmos_framework/inference/inference.py`

标准 `OmniInference` 入口把上述三个参数原样传给 `generate_samples_from_batch()`。没有增加第二套 inference CLI。

文件：`cosmos_framework/inference/common/public_model_config.py`

增加 `OmniMoTCausalModel` 的旧路径 alias，使训练保存的 YAML 配置能在当前代码目录结构下正确实例化 causal 模型。

### 2.2 在原版推理工作流中增加一个干净的 causal hook

文件：`cosmos_framework/model/generator/omni_mot_model.py`

基础类新增 `_generate_causal_inference_from_prepared()` hook。基础实现返回 `None`，因此普通模型仍继续走原来的整段视频采样流程。

调用位置位于原版 `_prepare_inference_data()` 之后。也就是说，下面这些成熟逻辑全部直接复用：

- prompt 与 negative prompt tokenization；
- Text+Image sequence plan；
- 图片的 VAE 编码与 condition mask；
- 按请求 seed 创建整段初始 noise；
- sampler、VAE decode 和最终文件保存。

只有实际 diffusion 循环由 `OmniMoTCausalModel` 接管。训练调用链没有进入这个 hook，原有 teacher-forcing 训练分支不受影响。

### 2.3 causal block 推理循环

文件：`cosmos_framework/model/generator/omni_mot_causal_model.py`

新增的推理主体负责：

1. 校验请求确实是 batch size 1 的 Text+Image→Video；
2. 创建 conditional cache；开启 CFG 时再创建独立的 unconditional cache；
3. 对 first-frame block 0 做一次 timestep=0 prefill；
4. 顺序生成 `B` 个 block；
5. 每个 block 去噪完成后，再做 timestep=0 prefill 并提交 CLEAN K/V；
6. 拼接 block 0 和所有生成 block，交回原版 VAE decode/保存流程。

文件：`cosmos_framework/model/generator/causal_inference.py`

集中保存长度、时间切片、flatten/unflatten 和拼接工具，避免在主循环中反复手写 tensor shape 逻辑。

### 2.4 GenKVCache

文件：`cosmos_framework/model/generator/mot/gen_kv_cache.py`

新增 request-local、per-layer、per-CFG-branch 的 KV cache：

- request-local：每个推理请求新建，不跨请求共享；
- per-layer：transformer 每层各有一份 K/V；
- per-CFG-branch：conditional 与 unconditional 绝不共用缓存；
- rolling window：只保留最近 `K` 个 finalized block。

它复用了 Cosmos 已有的 `MemoryState` / `MemoryValue` 接口，以及 decoder layer 已经支持的 `(gen_k, gen_v, und_k, und_v)` 回传机制，没有侵入式改写 transformer layer。

### 2.5 测试与可运行样例

- `cosmos_framework/model/generator/causal_inference_test.py`：验证 `1+B*S`、block 时间区间和 tensor reshape/slice；
- `cosmos_framework/model/generator/mot/gen_kv_cache_test.py`：验证 staging、READONLY、窗口淘汰、CFG 双分支提交和 K/V 拼接顺序；
- `examples/run_causal_gen_kv_cache_bridge_val.sh`：从 BridgeData2 验证集构造 causal JSON 并运行 checkpoint；
- `docs/causal_gen_kv_cache_inference.md`：实际运行参数和环境说明。

真实 NPU 端到端验证已覆盖 `B=16, S=2, K=16, num_steps=35, guidance=1.0`：生成 33 个 latent frame，并由 Wan2.2 causal VAE 正确解码为 129 帧、256×256 的 MP4。这验证了 `S>1` 的 block 联合去噪、CLEAN prefill、KV append/read 和视频解码链路。

## 3. 最重要的问题：到底缓存哪些 K/V

可以把一次生成想象成写一份多页报告：

- **UND 文本**是报告的任务说明，整个过程中不变；
- **CLEAN block**是已经定稿的页面，后面的页面可以参考；
- **NOISE block**是正在反复修改的草稿，每个 diffusion step 都不同。

因此 cache 内容是：

| 内容                   | 是否缓存                | 原因                                             |
| ---------------------- | ----------------------- | ------------------------------------------------ |
| UND 文本 K/V           | 是，只写一次            | prompt 在整个请求中不变                          |
| 已完成 CLEAN block K/V | 是，最多最近 K 个 block | 已经定稿，后续 block 需要参考                    |
| 当前 NOISE block K/V   | 否                      | 每个 diffusion step 都会变化，缓存后会污染下一步 |
| 未来 block K/V         | 否                      | 还没有生成，不存在真实 CLEAN latent              |

这也是为什么不能从不存在的“完整 CLEAN video”复制 K/V。推理开始时只有真实图片 block 0；后续 CLEAN history 必须等每个 block 去噪结束后逐步产生。

每一层的实际 cache 结构是：

```text
LayerCache
├── und_k / und_v             # 固定文本上下文
├── clean_k / clean_v         # rolling clean history 的预分配存储
└── clean_len                 # 当前有效 token 数
```

`CachedCleanBlock` 额外记录 block id、起始 latent frame、frame 数和 token 数，用来按完整 block 淘汰历史，而不是随意截断 token。

## 4. 两种 cache 状态为什么必要

### READONLY：去噪时只读

一个 block 通常要运行 35 个 diffusion step。每一步执行：

```text
Q = 当前 NOISE block
K/V = cached UND + cached CLEAN history + 当前 NOISE block
```

attention 输出只用于预测当前 step 的 velocity，cache 完全不修改。否则第 1 个 step 的噪声 K/V 会残留，并在第 2、3、4 个 step 中不断污染历史。

### APPEND：完成后独立 prefill

当 sampler 完成后，当前 block 才成为 CLEAN。代码用 timestep 0 单独 forward 一次：

```text
final CLEAN latent
    ↓ timestep=0 prefill
每层产出 CLEAN K/V
    ↓ staging
所有层成功后统一 commit
```

这次 forward 不负责继续去噪，只负责把已经定稿的 block 转成可以安全复用的 K/V。

`DISABLED` 枚举被保留用于 API 语义完整性，但 causal 状态机不会创建 DISABLED state。普通非 causal forward 的“关闭 cache”仍然是直接传 `memory=None`。

## 5. 为什么先 staging，再 commit

transformer 有很多层。如果运行到一半发生 OOM 或算子错误，前几层可能已经算出 K/V，而后几层没有。

如果每层算完就立刻写正式 cache，会留下“半个 block”：

```text
layer 0..10：有新 block
layer 11..N：没有新 block
```

后续 forward 将无法保持每层历史一致。因此 `GenKVCacheMemoryState.write_for_layer()` 先把每层结果放进 `_staged`；只有全部层都存在且 shape 正确，`commit_append()` 才一次性更新正式 cache。

CFG 时还会先同时验证 conditional 和 unconditional staging，再提交两个 cache，避免一个分支完整、另一个分支缺层。

## 6. attention 到底怎样使用 cache

Cosmos transformer 本来就支持外部 `MemoryState`：

```text
每个 decoder layer 前：memory.read_for_layer(layer_id)
decoder layer forward
每个 decoder layer 后：memory.write_for_layer(layer_id, kv_to_store)
```

GenKVCache 只临时替换每层 `self_attn.dispatch_attention_fn`。当 cache 尚未初始化时，仍调用原版 `dispatch_attention()`，完成同时包含 UND 和 block 0 的第一次 prefill。

cache 初始化后，模型进入 gen-only 路径，不再重复计算 UND token。自定义 attention 按以下顺序拼接：

```text
K_full = [UND K, CLEAN history K, current block K]
V_full = [UND V, CLEAN history V, current block V]

output = attention(current block Q, K_full, V_full, is_causal=False)
```

这里没有未来 block，`current block` 内的所有 token 属于同一次联合去噪，因此使用 dense、non-causal attention。

临时 dispatcher 安装在 `try` 内，并在 `finally` 恢复，确保一次 causal 请求结束后不会永久改变模型，也不会影响普通推理。

## 7. 从命令到视频的完整调用链

下面按照真实执行顺序阅读代码。

### 第一步：脚本构造输入

入口：`examples/run_causal_gen_kv_cache_bridge_val.sh`

脚本读取验证集的结构化 JSON 和首帧图片，删除 `num_frames`，补入 `B/S/K`、steps 和 guidance，然后调用：

```bash
python -m cosmos_framework.scripts.inference
```

### 第二步：解析参数并推导长度

入口：`cosmos_framework/inference/args.py::OmniSampleOverrides.build_sample`

检测 causal checkpoint 后，根据 `1+B*S` 推导 latent/pixel 长度，生成最终 `OmniSampleArgs`。

### 第三步：复用原版输入准备

入口：`cosmos_framework/model/generator/omni_mot_model.py::generate_samples_from_batch`

`_prepare_inference_data()` 一次性准备：

- `sequence_plans`；
- 编码后的完整目标形状；
- cond/uncond 文本 tokens；
- 根据请求 seed 生成的整段 `initial_noise`；
- 只在 frame 0 有效的 `condition_reference` 和 `condition_mask`。

这里没有 `block_seed`。整段初始 noise 只按请求 seed 生成一次，后续 block 从对应时间区间切片。调用 block sampler 时传 `seed=[None]`，避免 sampler 再次生成或覆盖这段 noise。

### 第四步：进入 causal hook

入口：`cosmos_framework/model/generator/omni_mot_causal_model.py::_generate_causal_inference_from_prepared`

代码先校验输入模式和几何尺寸，再创建一个覆盖整个目标时间轴的 position template。实际 forward 虽然只 pack 当前 block，但会从 full template 复制当前绝对时间位置的 mRoPE position ids。

这一步很重要：block 2 不能假装自己仍处于 frame 1；否则所有 block 都会重复相同时间位置。

### 第五步：prefill first-frame block 0

入口：`_run_causal_prefill()`

从 `condition_reference` 取真实首帧 latent，以 timestep 0 执行 APPEND forward：

- 第一次同时计算并 staging UND K/V 与 C0 CLEAN K/V；
- 所有 transformer 层成功后 commit；
- 从此 cache 标记为 initialized，后续 forward 进入 gen-only。

### 第六步：循环生成一个 block

入口：`for block_id in range(1, causal_num_blocks + 1)`

对每个 block：

1. 用 `causal_generated_block_span()` 得到它在完整 latent 中的 `[start:end]`；
2. 从整段 initial noise 取出这段作为 sampler 初值；
3. 构建只含当前 block 的 compact packed template；
4. sampler 重复调用 `velocity_fn()`；
5. 每次 conditional/unconditional forward 都使用各自 cache 的 READONLY state；
6. sampler 结束后得到 `block_clean`。

如果 `guidance == 1.0`，只运行 conditional branch。如果 `guidance != 1.0`，运行两个独立 cache：

```text
cond_velocity   = cond cache forward
uncond_velocity = uncond cache forward
guided = uncond + guidance * (cond - uncond)
```

`guidance_interval` 和 `normalize_cfg` 仍沿用原版语义。

### 第七步：将完成的 block 加入历史

除最后一个 block 外，每个 `block_clean` 都通过 `_run_causal_prefill()` 做 timestep 0 APPEND。

最后一个 block 不 prefill，因为本次请求中已经没有下一个 block 会读取它；省掉这次 forward 不影响最终输出。

当 cache 已有 `K` 个 block 时，提交新 block 前按 FIFO 淘汰最老的完整 block及其 K/V。block 0 和普通 clean block 采用相同窗口规则，所以在长序列中也可能被淘汰。

### 第八步：拼接、VAE 解码、保存

`concat_vision_time([C0, block1, ..., blockB])` 得到完整 latent。函数检查时间长度确实等于 `1+B*S`，然后返回基础 inference pipeline。

`OmniInference` 继续调用原版 `decode_vision()` 和输出保存逻辑，最终产生 `vision.mp4`、`sample_args.json` 和 `sample_outputs.json`。

## 8. CFG 下的两套 cache

CFG 的 conditional 与 unconditional 文本不同，因此 UND K/V 不同；同一个 clean vision block经过两个文本上下文的 transformer forward 后，生成侧 K/V 也可能不同。

因此不能只缓存一套，也不能只给其中一边做 prefill。当前实现是：

```text
conditional branch   → cond GenKVCache
unconditional branch → uncond GenKVCache
```

两边使用相同的 block geometry，但 staging、UND K/V 和 CLEAN K/V 都彼此独立。`guidance=1.0` 时不会创建 unconditional cache，也不会产生额外 forward。

## 9. 与普通推理、训练代码的边界

### 普通非 causal 推理

基础类 hook 返回 `None` 后，继续执行原版整段 diffusion。原版已有的 text-only inference KV reuse 仍在后面的普通路径中生效。causal hook 提前返回，因此不会和 text-only cache 叠加安装 dispatcher。

### causal 训练

训练仍走：

- `prepare_teacher_forcing_geometry()`；
- `post_noise_packing_hook()`；
- `expand_teacher_forcing_training_sequence()`。

GenKVCache 只在 `_generate_causal_inference_from_prepared()` 中创建，没有被训练 step 引用。这样推理 cache 状态机不会污染 Scheme-B teacher-forcing 训练逻辑。

## 10. 当前限制和阅读时容易误解的点

1. 当前只支持 batch size 1、Text+Image→Video、condition frame index `[0]`。
2. 不支持 action、sound、velocity postprocess。
3. 暂不支持 CP、CFGP 或 FSDP sharding；脚本中的 size 1 表示这些并行方式没有真正启用。
4. `causal_block_size` 必须落在 checkpoint 训练范围内；当前训练配置是 `[1,4]`，并非只支持 `S=1`。
5. `causal_history_blocks` 最大 16；它控制保留多少个 block，不是 frame 数或 token 数。
6. cache 中没有完整 CLEAN video，也没有从 NOISE 复制 CLEAN；历史只能在 block 完成后逐个建立。
7. PageAttention 主要解决大量请求下的 KV 内存分页和调度问题，不会减少首次 prefill 计算；当前单请求、固定上限、预分配 rolling buffer 暂不需要它。
8. 当前 cache 是请求内内存，不会保存到磁盘，也不会让 server 保存每一步推理图片。

## 11. 建议的阅读顺序

第一次阅读时按以下顺序最容易理解：

1. `cosmos_framework/model/generator/causal_inference.py`：先理解 `1+B*S` 和 block slice；
2. `cosmos_framework/model/generator/mot/gen_kv_cache.py`：理解 READONLY / APPEND 和缓存内容；
3. `cosmos_framework/model/generator/mot/unified_mot.py`：查看每层怎样 read/write `MemoryState`；
4. `cosmos_framework/model/generator/omni_mot_causal_model.py::_run_causal_prefill`：理解 CLEAN 如何进入 cache；
5. 同文件 `_generate_causal_inference_from_prepared`：阅读完整 block 循环；
6. `cosmos_framework/model/generator/omni_mot_model.py::generate_samples_from_batch`：确认 causal hook 在原版工作流中的位置；
7. `cosmos_framework/inference/args.py` 与 `inference/inference.py`：最后看参数如何从 JSON 一路传入模型。
