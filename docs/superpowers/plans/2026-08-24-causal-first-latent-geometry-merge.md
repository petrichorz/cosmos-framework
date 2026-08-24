# Causal First-Latent Geometry Bug Fix 合并说明

日期：2026-08-24

目标分支：`feat/data-augmentation`

来源分支：`fix/causal-first-latent-geometry`

详细设计与实现 handoff：
[`docs/superpowers/plans/2026-08-22-causal-first-latent-geometry-handoff.md`](./2026-08-22-causal-first-latent-geometry-handoff.md)

## 1. 合并目的

本次合并修复 causal teacher-forcing 训练中的两个问题：

1. VAE 首 latent 会按照普通 `block_size` 与后续 latent 合并，不能表达 T2V 首 latent 独立生成、后续 latent 再分 block 的时序几何。
2. teacher-forcing masked SDPA 会在每一层、每个 packed sample 中重复扫描同一张 bool mask，并把设备归约结果转换成 Python `bool`，造成 NPU 到 CPU 的同步。

## 2. 已合并决策

### 2.1 T2V-only teacher forcing

- causal teacher-forcing 训练始终使用 T2V 数据；所有 latent 均加噪并参与 flow-matching loss。
- 首 latent 独立作为 singleton block 0。
- latent 1 之后按每条 packed sample 独立采样的 `block_size=S` 分块。
- frame-to-block 映射统一为：

  ```text
  frame:     0 | 1 ... S | S+1 ... 2S | ...
  block_id:  0 |    1    |      2      | ...
  ```

- attention geometry 与 sigma 展开复用同一个 frame-to-block helper，避免两处语义漂移。
- CLEAN 与 NOISY stream 保持完全等长；NOISY block 0 不读取 CLEAN block 0，后续 noisy block 可以读取 history window 内严格更早的 clean blocks。
- teacher-forcing 训练只接受 `condition_frame_indexes_vision=[]`。I2V 的外部 clean 首帧初始化保留为推理能力，不作为该训练路径的数据模式；V2V 同样被显式拒绝。
- 首块不永久可见，仍按照每条 sample 独立采样的 `history_blocks` 正常淘汰。

对应提交：

- `8e08f44 fix(teacher-forcing): isolate the first latent causal block`

### 2.2 移除 masked SDPA 热路径同步

- 通用 `masked_sdpa` 默认仍检查每个 query 至少有一个 visible key，保持直接调用者的安全契约。
- teacher-forcing layout builder 生成的 mask 通过显式 prevalidated 标志进入 attention。
- global 和 per-sample teacher-forcing 调度不再在每层重复执行：

  ```python
  bool(allowed_mask.any(dim=-1).all())
  ```

- dtype、shape、device、数值结果和反向传播行为保持不变。

对应提交：

- `5f2c06c perf(masked-sdpa): avoid repeated device mask synchronization`

## 3. 数据增强分支兼容性

- packing 中的每条 sample 仍然独立采样 `block_size` 和 `history_blocks`。
- block 内共享 sigma 的数据增强保留，但共享范围改为 `[1, S, S, ...]`：首 latent 使用独立 sigma，后续普通 block 内共享 sigma。
- 不增加新的 geometry mode 配置项，也不改变普通 `two_way`、`three_way`、diffusion-forcing 或非 causal 工作流的默认路由。
- 若数据集仍产生 I2V/V2V condition，causal teacher-forcing 会在加噪前给出明确错误；配套 `cosmos` launcher 需要使用 100% T2V conditioning 配置。

## 4. 验证结果

使用 `conda activate cosmos-framework` 验证：

- sequence-packing 与 causal model teacher-forcing：`72 passed`；
- attention、network 与 vertical-slice：`30 passed`；
- 相关 Python 文件 `ruff check` 通过；
- `git diff --check` 通过。

测试覆盖包括：

- `block_size=1` 和 `block_size>1` 的 singleton 首 latent geometry；
- packed samples 隔离与独立 geometry；
- clean/noisy mask 可见性和 history window 淘汰；
- `[1,S,S,...]` sigma 展开；
- 全部训练 latent 的 loss 索引保留；
- I2V/V2V condition 拒绝；
- masked SDPA 默认校验契约与 teacher-forcing prevalidated 热路径；
- attention 数值一致性和 clean/noisy 两条 stream 的反向传播。

## 5. 合并后注意事项

- 正式训练应关闭 `teacher_forcing_visualize_sdpa_mask`；per-sample 模式下开启该调试项仍会为了绘图额外构造完整 global dense mask。
- 当前 teacher-forcing topology 是训练专用；没有 teacher-forcing layout 的 validation/inference 会回退到普通 `two_way`。block-causal T2V/I2V 推理需要按照独立 inference handoff 继续实现。
- 本次合并不包含工作区中的其他文档或配置修改。
