# Cosmos3-Edge I2V / V2V 推理代码导读

本文基于 `cosmos-framework`，从 Edge 的 I2V / V2V 推理入口开始，沿实际调用链追踪到 UniPC 去噪循环和底层 MoT Transformer `forward`。重点回答：

1. I2V 和 V2V 的输入条件分别如何构造；
2. reasoner 在推理时如何处理文本，文本是否 causal；
3. reasoner 的 K/V 如何被 generator 使用；
4. 独立 AR reasoner 的 KV cache 与扩散期 text KV cache 有什么区别；
5. 哪个函数才是真正执行网络 forward，如何设置断点跟踪。

本文讨论的是标准 Edge I2V / V2V 推理。当前 Edge 配置中的 `video_temporal_causal=false`，不要将它和 causal teacher-forcing 训练混为一谈。

## 1. 先记住三个结论

### 1.1 默认不会先生成 reasoner 文本

标准 I2V / V2V 推理默认不会先让 reasoner 自回归生成一段新文本，再把这段 AR KV cache 交给 generator。

默认流程是：

```text
原始 prompt
  → chat template + tokenize
  → 作为 und/reasoner tokens 与视频 gen tokens 一起打包
  → 在每一层联合 MoT attention 中为 generator 提供 K/V
```

只有显式启用 native prompt upsampling，或者单独运行 `model_mode=reasoner`，才会进入 reasoner 的自回归 token 生成循环。

### 1.2 文本 causal，不代表视频 causal

Edge 默认使用 `two_way` attention：

```text
文本 Query：只看当前及历史文本 token，属于 causal self-attention
视频 Query：看全部文本 K/V 和全部当前视频 K/V，属于 full attention
```

因此，“文本是 causal 的”和“视频不是 temporal causal 的”可以同时成立。

### 1.3 存在两种完全不同的 KV cache

| KV cache | 使用阶段 | 作用 | 是否直接交给扩散 generator |
| --- | --- | --- | --- |
| `ReasonerKVCache` | 自回归文本生成 | 缓存 prompt 和已生成文本 token 的 K/V | 否 |
| `InferenceTextKVMemoryState` | 多步扩散去噪 | 缓存静态 prompt 的 und/text K/V，供后续去噪步复用 | 是 |

## 2. 推理启动方式

### 2.1 命令行

I2V：

```bash
conda activate cosmos-framework
cd /mi/data2T/Embodied-AI/codes/cosmos_ascend/cosmos-framework

python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  --no-guardrails \
  -i inputs/omni/i2v.json \
  -o outputs/edge_i2v \
  --checkpoint-path Cosmos3-Edge \
  --seed=0
```

V2V：

```bash
python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  --no-guardrails \
  -i inputs/omni/v2v.json \
  -o outputs/edge_v2v \
  --checkpoint-path Cosmos3-Edge \
  --seed=0
```

示例输入：

- [I2V JSON](../inputs/omni/i2v.json)
- [V2V JSON](../inputs/omni/v2v.json)

### 2.2 VS Code Debug

工作区 `.vscode/launch.json` 中已经加入：

- `Python: Cosmos3 Edge I2V 推理`
- `Python: Cosmos3 Edge V2V 推理`

两项配置都通过以下模块启动：

```text
cosmos_framework.scripts.inference
```

并将 `cwd` 固定为 `cosmos-framework`，以便正确解析 `inputs/omni/*.json` 等相对路径。

调试配置还显式设置了 `--no-guardrails`。Guardrail 默认开启，但属于可选依赖，使用 `nltk`、`better-profanity`、RetinaFace 等额外包，并会加载额外安全模型。关闭它可以让代码导读只关注 I2V/V2V 主推理链。代价是调试输出不会经过文本安全过滤、视频安全分类和人脸模糊，正式对外提供生成服务时不应直接沿用该选项。

## 3. 总体调用链

先从整体上理解一次推理：

```text
cosmos_framework.scripts.inference.main
  └─ inference(args)
      ├─ 解析 JSON、默认参数和 checkpoint
      ├─ OmniInference.create(setup_args)
      │   └─ 加载 Cosmos3OmniModel / OmniMoTModel
      └─ Inference.generate(sample_args_list)
          └─ OmniInference.create_batches(...)
              └─ SampleDataset.__getitem__
                  └─ get_sample_data(...)
                      ├─ I2V：load_conditioning_image
                      └─ V2V：load_conditioning_video
          └─ OmniInference.generate_batch(...)
              └─ OmniMoTModel.generate_samples_from_batch(...)
                  ├─ _prepare_inference_data(...)
                  │   ├─ VAE encode
                  │   ├─ tokenize prompt
                  │   ├─ 构造 condition mask
                  │   └─ 初始化 clean/noise 混合 latent
                  └─ UniPCSampler.forward(...)
                      └─ for timestep in timesteps
                          └─ velocity_fn(...)
                              └─ _get_velocity(...)
                                  └─ denoise(...)
                                      └─ Cosmos3VFMNetwork.forward(...)
                                          └─ language_model.forward(...)
                                              └─ MoT decoder layers
```

## 4. 顶层入口与参数解析

入口文件是 [`cosmos_framework/scripts/inference.py`](../cosmos_framework/scripts/inference.py#L21)。

`InferenceArgs` 定义两个主要参数：

```python
class InferenceArgs(pydantic.BaseModel):
    input_files: list[Path]
    setup: SetupOverrides
```

真正的入口逻辑位于 [`inference()`](../cosmos_framework/scripts/inference.py#L34)：

```python
setup_args = args.setup.build_setup()

sample_overrides_list = setup_args.get_sample_overrides_cls().from_files(
    args.input_files,
    overrides=setup_args.sample_overrides,
)

pipe = setup_args.get_inference_cls().create(setup_args)
pipe.generate(sample_args_list)
```

它依次完成：

1. 读取输入 JSON；
2. 合并 modality 默认参数、JSON 参数和 CLI override；
3. 根据 `--checkpoint-path Cosmos3-Edge` 选择 Edge 配置；
4. 创建 `OmniInference`；
5. 加载模型并执行批次推理。

`OmniSetupArgs.get_inference_cls()` 明确返回 `OmniInference`，见 [`args.py`](../cosmos_framework/inference/args.py#L1293)。

### 4.1 参数默认值的来源

I2V / V2V 的通用默认值分别来自：

- [`defaults/image2video/sample_args.json`](../cosmos_framework/inference/defaults/image2video/sample_args.json)
- [`defaults/video2video/sample_args.json`](../cosmos_framework/inference/defaults/video2video/sample_args.json)

主要默认值包括：

```json
{
  "num_steps": 35,
  "guidance": 6.0,
  "num_frames": 189,
  "fps": 24,
  "autoregressive": false
}
```

实际解析还会叠加模型专属规则。Edge 的普通视频默认帧数会从通用的 189 改成 121，见 [`OmniSampleOverrides._NUM_FRAMES_DEFAULTS`](../cosmos_framework/inference/args.py#L1050)。

## 5. Edge 模型加载

模型加载发生在 [`OmniInference._create()`](../cosmos_framework/inference/inference.py#L1171)。对于 HF 或导出 checkpoint，核心调用是：

```python
model = Cosmos3OmniModel.from_pretrained_dcp(
    checkpoint_path,
    config=config,
    parallelism_config=parallelism_config,
    compile_config=compile_config,
    quantization_config=quantization_config,
).model
```

包装类 [`Cosmos3OmniModel`](../cosmos_framework/inference/model.py#L489) 根据配置实例化真正的 `OmniMoTModel`：

```python
self.model = hydra.utils.instantiate(model_dict)
```

Edge 配置位于 [`Cosmos3-Edge.yaml`](../cosmos_framework/inference/configs/model/Cosmos3-Edge.yaml#L10)。与本次导读最相关的字段是：

```yaml
joint_attn_implementation: two_way
precision: bfloat16
video_temporal_causal: false

tokenizer:
  spatial_compression_factor: 16
  temporal_compression_factor: 4

vlm_config:
  model_name: nvidia/Cosmos3-Edge-Reasoner
  use_system_prompt: false
```

含义如下：

- 文本和视频使用 `two_way` 联合注意力；
- 模型主计算精度是 BF16；
- 当前标准推理没有启用视频时间因果注意力；
- Wan VAE 空间压缩 16 倍、时间压缩 4 倍；
- 文本侧使用 Edge Reasoner 权重。

## 6. I2V / V2V 输入在哪里分叉

批次准备由 [`OmniInference.create_batches()`](../cosmos_framework/inference/inference.py#L1391) 和 [`SampleDataset.__getitem__()`](../cosmos_framework/inference/inference.py#L1120) 驱动，单样本最终进入 [`get_sample_data()`](../cosmos_framework/inference/inference.py#L638)。

代码根据 `vision_path` 的扩展名判断条件媒体类型：

```python
match sample_args.condition_vision_mode:
    case "image":
        conditioning_frames = load_conditioning_image(...)
    case "video":
        conditioning_frames = load_conditioning_video(...)
```

### 6.1 I2V

图片经过 [`load_conditioning_image()`](../cosmos_framework/inference/vision.py#L72)：

1. 读取 RGB 图片；
2. 等比例 resize；
3. center crop 到目标分辨率；
4. 归一化到 `[-1, 1]`；
5. 增加时间维，得到 `[3, 1, H, W]`。

图片条件默认对应 latent frame index：

```python
[0]
```

默认值定义在 [`DEFAULT_CONDITION_FRAME_INDEXES_VISION`](../cosmos_framework/inference/args.py#L304)。因此，I2V 真正固定的是第 0 个 VAE latent frame。

### 6.2 V2V

视频条件默认对应 latent frame indexes：

```python
[0, 1]
```

读取视频之前，代码先根据最大条件 latent index 计算需要多少像素帧，见 [`get_sample_data()`](../cosmos_framework/inference/inference.py#L690)：

```python
num_condition_latent_frames = max(condition_frame_indexes_vision) + 1
max_frames = tokenizer.get_pixel_num_frames(num_condition_latent_frames)
```

Wan VAE 的换算规则见 [`get_pixel_num_frames()`](../cosmos_framework/model/generator/tokenizers/wan2pt2_vae_4x16x16.py#L1783)：

```python
pixel_frames = (latent_frames - 1) * 4 + 1
```

因此默认 V2V 条件为：

```text
2 个 latent frames
→ (2 - 1) × 4 + 1
→ 5 个像素帧
```

[`load_conditioning_video()`](../cosmos_framework/inference/vision.py#L79) 默认取输入视频开头的 5 帧，而不是把整条输入视频作为固定控制条件：

```python
frames = frames[:max_frames]  # condition_video_keep="first"
```

所以这里的默认 V2V 更准确地说是“视频前缀条件生成”。可以在输入 JSON 中调整：

```json
{
  "condition_frame_indexes_vision": [0, 1, 2, 3],
  "condition_video_keep": "first"
}
```

4 个条件 latent frames 对应 13 个输入像素帧。

## 7. 输出帧数和 `1+4N` 对齐

Edge 默认输出 121 帧：

```text
121 = 1 + 4 × 30
```

在 24 FPS 下约为：

```text
121 / 24 ≈ 5.04 秒
```

如果用户显式设置其他 `num_frames`，构建参数时仍会向上对齐到 `1+4N`，见 [`VisionDataOverrides._build_vision_data()`](../cosmos_framework/inference/args.py#L532)：

```python
self.num_frames = (
    ceil((self.num_frames - 1) / temporal_compression_factor)
    * temporal_compression_factor
    + 1
)
```

这与 Wan VAE 的首帧加后续 4 帧一组的时间压缩结构一致。

## 8. 条件帧和噪声如何组成初始 latent

推理准备的核心函数是 [`_prepare_inference_data()`](../cosmos_framework/model/generator/omni_mot_model.py#L1749)。它完成：

1. 构建 `SequencePlan`；
2. VAE 编码输入图像或视频；
3. tokenize 正向和负向文本；
4. 构建 packed sequence 以获得 condition mask；
5. 创建随机噪声；
6. 将 clean condition 和 noise 合并成采样初态。

关键公式见 [`omni_mot_model.py`](../cosmos_framework/model/generator/omni_mot_model.py#L1856)：

```python
noise_i = cond_mask * x0_token + (1.0 - cond_mask) * pure_noise_i
```

其中：

```text
cond_mask = 1：保留 VAE 编码后的 clean condition
cond_mask = 0：使用随机噪声，等待扩散模型生成
```

I2V 默认只固定 latent frame 0；V2V 默认固定 latent frames 0 和 1。

在每一步网络预测完成后，条件位置的 velocity 也会被清零，见 [`_get_velocity()`](../cosmos_framework/model/generator/omni_mot_model.py#L2255)：

```python
noisy_mask = 1.0 - condition_mask
velocity = prediction * noisy_mask
```

因此采样器不会更新已经固定的条件 latent。

## 9. reasoner 如何处理文本

### 9.1 默认 I2V / V2V：直接 tokenize

默认 prompt 在 [`_get_inference_text_tokens()`](../cosmos_framework/model/generator/omni_mot_model.py#L1713) 中处理：

```python
cond_tokens = self._tokenize_captions(
    data_batch[self.input_caption_key],
    use_system_prompt=use_system_prompt,
    system_prompt=system_prompt,
)
```

同时还会构建 CFG 的 unconditional tokens：

```python
uncond_tokens = self._tokenize_captions(uncond_captions, ...)
```

这些 token 随后通过 [`_pack_input_sequence()`](../cosmos_framework/model/generator/omni_mot_model.py#L578) 与视频 latent 一起进入 `PackedSequence`。

概念上，一条样本可以表示为：

```text
[text tokens / und pathway]
[vision latent tokens / gen pathway]
```

实际存储还带有 `text_indexes`、`gen indexes`、位置编码、condition mask 和 timestep 等元数据，并不依赖简单的物理连续区间来区分模态。

### 9.2 文本输入是否 causal

是。Edge 使用的 [`two_way_attention()`](../cosmos_framework/model/generator/mot/attention.py#L143) 对文本/und 路径执行：

```python
causal_res = attention(
    causal_q,
    causal_k,
    causal_v,
    is_causal=True,
)
```

所以文本 token `i` 只能看到：

```text
text[0 ... i]
```

文本 Query 不会看到视频 latent。

但 generator/video 路径使用 full attention：

```python
full_res = attention(
    full_q,
    get_all_seq(packed_key_normalized),
    get_all_seq(packed_value_states),
)
```

generator Query 可以看到：

```text
所有文本 K/V + 所有当前视频 K/V
```

因此其可见关系为：

```text
text Query  ──causal──► text K/V

video Query ──full────► text K/V
            └─────────► video K/V
```

### 9.3 与视频 temporal causal 的区别

当前 Edge YAML 中：

```yaml
video_temporal_causal: false
```

这表示当前视频 latent 之间不是按时间单向可见。它不影响文本路径始终使用 causal self-attention。

## 10. reasoner K/V 如何进入 generator

这里不是“先运行一个完整 reasoner 模型，再运行一个独立 generator 模型”。reasoner/und 和 generator/gen 是同一个 Transformer layer 中的两套投影路径。

每个 MoT attention layer 分别计算，见 [`unified_mot.py`](../cosmos_framework/model/generator/mot/unified_mot.py#L612)：

```python
q_und = q_proj(und_hidden)
k_und = k_proj(und_hidden)
v_und = v_proj(und_hidden)

q_gen = q_proj_moe_gen(gen_hidden)
k_gen = k_proj_moe_gen(gen_hidden)
v_gen = v_proj_moe_gen(gen_hidden)
```

generator attention 的逻辑等价于：

```text
Q = q_gen
K = concat(k_und, k_gen)
V = concat(v_und, v_gen)
```

因此文本条件不是只在网络输入层注入一次，而是在每一层通过 `Q_gen → K/V_und` attention 持续注入。

### 10.1 Edge 对 generator 看到的文本 K 做额外归一化

Edge 配置设置：

```yaml
use_und_k_norm_for_gen: true
```

对应实现见 [`unified_mot.py`](../cosmos_framework/model/generator/mot/unified_mot.py#L657)：

```python
k_und_normalized = self.k_norm_und_for_gen(k_und)
```

reasoner 自己的 causal attention 仍使用原始文本 K；generator 读取文本条件时使用额外归一化后的 K，以缓解文本路径和 diffusion 路径 QK normalization 策略不同造成的尺度不匹配。

## 11. 扩散期 text KV cache

prompt 在 35 个扩散步骤中保持不变。如果每一步都重新计算所有文本 K/V，会产生重复工作。

单卡、单样本且满足条件时，代码通过 [`InferenceTextKVMemoryState`](../cosmos_framework/model/generator/mot/inference_text_kv_memory.py#L54) 缓存每一层的 und/text K/V。

第一次去噪 forward：

```text
计算 text K/V
计算当前 video K/V
保存每层 text K/V
执行联合 attention
```

后续去噪 forward：

```text
跳过 text pathway 的重复计算
读取 cached text K/V
计算当前 video K/V
拼接 [cached text KV | current video KV]
执行 generator attention
```

拼接实现位于 [`inference_text_kv_memory.py`](../cosmos_framework/model/generator/mot/inference_text_kv_memory.py#L100)：

```python
kv_parts_k = [k_curr]
kv_parts_v = [v_curr]

kv_parts_k.insert(0, memory_value.und_k_cached)
kv_parts_v.insert(0, memory_value.und_v_cached)

k_full = torch.cat(kv_parts_k, dim=1)
v_full = torch.cat(kv_parts_v, dim=1)
```

启用条件由 [`_can_reuse_inference_text_kv()`](../cosmos_framework/model/generator/omni_mot_model.py#L2026) 控制，主要要求：

- 单样本；
- `two_way` attention；
- 没有 CP、CFGP 或 FSDP shard；
- 没有 temporal causal video；
- 没有 sound；
- 没有 transfer velocity postprocess。

CFG 的 conditional 和 unconditional prompt 内容不同，因此使用两套独立缓存：

```text
cond_text_kv_cache
uncond_text_kv_cache
```

## 12. 独立 AR reasoner 路径

当运行 `model_mode=reasoner` 或启用 native prompt upsampling 时，才会进入 [`OmniMoTModel.generate_reasoner_text()`](../cosmos_framework/model/generator/omni_mot_model.py#L4511)。

该路径执行：

```text
prompt / image / video
  → multimodal chat template
  → reasoner prefill
  → token-by-token AR decode
  → 输出字符串
```

真正的 token 循环在 [`_impl_generate_reasoner_text()`](../cosmos_framework/model/generator/mot/unified_mot.py#L1941)：

```python
for _ in range(max_new_tokens - 1):
    hidden = model.reasoner_forward(
        step_input,
        cache=cache,
        position_ids=position_ids,
    )
    logits = causal_lm.lm_head(hidden[:, -1, :])
    next_token = _sample_next_token(logits, ...)
```

其 [`ReasonerKVCache`](../cosmos_framework/model/generator/mot/unified_mot.py#L1393) 沿序列维追加已生成 token 的 K/V：

```python
cached_k = torch.cat([cached_k, k], dim=1)
cached_v = torch.cat([cached_v, v], dim=1)
```

reasoner prefill 使用 causal attention；单 token decode 时，当前 token 位于序列最右端，可以直接读取全部历史 cache，无需再构造 causal mask，见 [`reasoner_forward()`](../cosmos_framework/model/generator/mot/unified_mot.py#L742)。

### 12.1 AR KV 不直接注入 diffusion generator

如果使用 prompt upsampling，正确的数据流是：

```text
原始 prompt / 条件图片
  → AR reasoner
  → 生成增强后的文本字符串
  → 重新 tokenize
  → 作为 und/text tokens 进入标准 diffusion forward
```

`ReasonerKVCache` 在 AR 文本生成结束后不再使用。真正跨扩散步骤复用、并参与 generator attention 的是 `InferenceTextKVMemoryState`。

## 13. 真正的去噪循环

普通 I2V / V2V 从 [`generate_samples_from_batch()`](../cosmos_framework/model/generator/omni_mot_model.py#L2453) 进入采样器。

Edge 默认 scheduler 是 UniPC，真正的时间步循环位于 [`UniPCSampler.forward()`](../cosmos_framework/model/generator/diffusion/samplers/unipc.py#L31)：

```python
for timestep in timesteps:
    velocity_pred = velocity_fn(latent, timestep.reshape(1, 1))
    latent = sample_scheduler.step(
        model_output=velocity_pred,
        timestep=timestep,
        sample=latent.unsqueeze(0),
    )[0].squeeze(0)
```

默认：

```text
num_steps = 35
guidance = 6.0
```

`velocity_fn` 定义在 [`generate_samples_from_batch()`](../cosmos_framework/model/generator/omni_mot_model.py#L2730)。当 guidance 不等于 1 时，通常每个 timestep 包含：

```text
conditional forward
+ unconditional forward
+ CFG 合成
```

CFG 公式为：

```python
v_pred = uncond_v + guidance * (cond_v - uncond_v)
```

所以 35 个 UniPC timestep 通常会触发约 70 次网络 forward，而不是 35 次。

## 14. 单个 timestep 的 forward

`velocity_fn` 最终调用 [`_get_velocity()`](../cosmos_framework/model/generator/omni_mot_model.py#L2091)。它完成：

1. 将扁平采样状态恢复成 vision/action/sound tensor；
2. 更新当前 timestep 和 noisy latent；
3. 将文本与多模态 token 打包；
4. 调用 `denoise()`；
5. 对条件位置应用 velocity mask；
6. 再次展平并返回给 UniPC scheduler。

网络调用位置见 [`omni_mot_model.py`](../cosmos_framework/model/generator/omni_mot_model.py#L2248)：

```python
out = self.denoise(
    net=net,
    data_batch_packed=packed_sequence,
    memory=memory,
)
```

随后 [`denoise()`](../cosmos_framework/model/generator/omni_mot_model.py#L4271) 执行：

```python
out_net = net(
    packed_seq=data_batch_packed,
    memory=memory,
    video_temporal_causal=video_temporal_causal,
)
```

这里的 `net` 是 `Cosmos3VFMNetwork`。

### 14.1 不要在空的 `OmniMoTModel.forward()` 上停留

[`OmniMoTModel.forward()`](../cosmos_framework/model/generator/omni_mot_model.py#L3173) 当前是空实现：

```python
@torch.no_grad()
def forward(self, xt, t):
    pass
```

标准推理不会通过这个函数。真正的网络入口是：

```text
OmniMoTModel.denoise
  → Cosmos3VFMNetwork.forward
```

## 15. `Cosmos3VFMNetwork.forward`

真正的多模态网络 forward 位于 [`cosmos3_vfm_network.py`](../cosmos_framework/model/generator/mot/cosmos3_vfm_network.py#L928)。主要步骤是：

```text
_encode_text
_encode_vision
_encode_action（普通 I2V/V2V 没有 action 输入）
_encode_sound（Edge 默认 sound_gen=false）
build_packed_sequence
language_model(...)
_decode_vision
```

### 15.1 文本 embedding

[`_encode_text()`](../cosmos_framework/model/generator/mot/cosmos3_vfm_network.py#L540) 先查 embedding：

```python
packed_text_embedding = self.language_model.model.embed_tokens(packed_seq.text_ids)
```

然后根据 `text_indexes` scatter 到统一 packed sequence：

```python
packed_sequence[packed_seq.text_indexes] = packed_text_embedding
```

### 15.2 进入统一 language model

底层模型调用位于 [`cosmos3_vfm_network.py`](../cosmos_framework/model/generator/mot/cosmos3_vfm_network.py#L1122)：

```python
packed_outputs, lbl_metadata = self.language_model(
    input_pack,
    attention_mask=attention_meta,
    position_ids=packed_position_ids,
    natten_metadata_list=natten_metadata_list,
    memory=memory,
)
```

在这里之后，执行每一个 MoT decoder layer，分别计算 und/gen attention 和 MLP，再将 gen hidden states 解码为 vision velocity。

## 16. 输出解码

35 步采样完成后，`generate_samples_from_batch()` 将扁平 latent 拆回 vision latent，见 [`omni_mot_model.py`](../cosmos_framework/model/generator/omni_mot_model.py#L2955)。

`OmniInference.generate_batch()` 随后调用 VAE decode，并通过 `save_img_or_video()` 保存：

```text
<output_dir>/<sample_name>/vision.mp4
```

保存逻辑位于 [`inference.py`](../cosmos_framework/inference/inference.py#L1694) 和 [`inference.py`](../cosmos_framework/inference/inference.py#L1740)。

## 17. 推荐断点顺序

第一次导读不建议直接钻进所有 attention kernel。按以下顺序设置断点，更容易建立完整心智模型：

| 顺序 | 断点 | 观察内容 |
| --- | --- | --- |
| 1 | [`scripts/inference.py:34`](../cosmos_framework/scripts/inference.py#L34) | CLI 和输入文件如何解析 |
| 2 | [`inference.py:638`](../cosmos_framework/inference/inference.py#L638) | I2V/V2V 条件媒体分支 |
| 3 | [`inference.py:1483`](../cosmos_framework/inference/inference.py#L1483) | batch 参数、CFG、采样器选择 |
| 4 | [`omni_mot_model.py:1749`](../cosmos_framework/model/generator/omni_mot_model.py#L1749) | VAE latent、condition mask、初始噪声 |
| 5 | [`omni_mot_model.py:2453`](../cosmos_framework/model/generator/omni_mot_model.py#L2453) | 整体采样入口 |
| 6 | [`unipc.py:83`](../cosmos_framework/model/generator/diffusion/samplers/unipc.py#L83) | 真正的 35 步循环 |
| 7 | [`omni_mot_model.py:2730`](../cosmos_framework/model/generator/omni_mot_model.py#L2730) | 单步 `velocity_fn` 和 CFG |
| 8 | [`omni_mot_model.py:2248`](../cosmos_framework/model/generator/omni_mot_model.py#L2248) | 单次网络调用前的 packed data |
| 9 | [`cosmos3_vfm_network.py:928`](../cosmos_framework/model/generator/mot/cosmos3_vfm_network.py#L928) | 真正的多模态 forward |
| 10 | [`attention.py:143`](../cosmos_framework/model/generator/mot/attention.py#L143) | 文本 causal 与 generator full attention |

建议重点观察这些变量：

```text
sample_args.condition_frame_indexes_vision
sequence_plans
gen_data_clean.x0_tokens_vision
packed_sequence.vision.condition_mask
cond_tokens / uncond_tokens
noise_x
timestep
packed_seq.text_indexes
vision_sequence_indexes
memory
```

## 18. 常见误区

### 误区一：V2V 默认使用完整输入视频

不是。默认只条件化前两个 VAE latent frames，即输入视频开头 5 个像素帧。

### 误区二：reasoner 先生成文本，再把它的 AR KV 给 generator

默认不是。原始 prompt 直接 tokenize 后进入联合 MoT；开启 prompt upsampling 时，也只有增强后的字符串会重新 tokenize，AR cache 本身不会交给 generator。

### 误区三：文本 causal，所以生成视频也是 temporal causal

不是。文本 causal mask 和视频 temporal causal 是两个独立维度。当前 Edge 标准配置中前者开启，后者关闭。

### 误区四：`OmniMoTModel.forward()` 是推理主入口

不是。它是空函数。实际路径是 `generate_samples_from_batch → _get_velocity → denoise → Cosmos3VFMNetwork.forward`。

### 误区五：35 个采样步等于 35 次网络 forward

默认 guidance 为 6.0，需要 conditional/unconditional 两个分支，因此通常约为 70 次 forward。

## 19. 一句话总结

Cosmos3-Edge I2V/V2V 推理本质上是：将 causal 文本 tokens 和当前 noisy/conditioned 视频 latent tokens 打包进同一个 MoT Transformer，在每层中让 generator Query 读取 reasoner 文本 K/V，经过约 35 个 UniPC 去噪 timestep（默认 CFG 下约 70 次网络 forward），最后由 Wan VAE 解码为视频。
