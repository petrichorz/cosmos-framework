# Causal Teacher-Forcing First-Latent Geometry Handoff

日期：2026-08-22

基线分支：`feat/data-augmentation`

实现分支：`fix/causal-first-latent-geometry`

## 1. 问题

当前 teacher-forcing layout 使用 `frame_id // block_size` 给所有 vision latent 分块。该规则会在
`block_size > 1` 时把 VAE 的首 latent 与后续 latent 放进同一个 block。

这会破坏 T2V teacher forcing 与 I2V 推理共享的首块语义：

- T2V 的首 latent 应当独立生成并参与 loss；
- I2V 推理可以跳过首块生成，直接用外部图片的 clean latent 初始化 block 0；
- 两条推理路径从 latent 1 开始应当使用完全相同的普通 block 划分。

当前 noisy query 只允许读取严格早于自身 block 的 clean block，因此把 I2V 的 clean latent 0 与
noisy latent 1 放在同一个 block，还会导致第一个生成 block看不到条件首帧。

## 2. 已确认决策

1. 首 latent 永远是独立的 singleton block 0，不受采样得到的普通 `block_size=S` 影响。
2. latent 1 之后按大小 `S` 分块：

   ```text
   frame:     0 | 1 ... S | S+1 ... 2S | ...
   block_id:  0 |    1    |      2      | ...
   ```

3. causal teacher-forcing 训练始终使用 T2V：所有 latent 都进入 NOISY stream、加噪并参与 loss。
4. I2V 不作为 teacher-forcing 训练模式；它只在推理时用外部 clean `C0` 初始化已经完成的 singleton block 0，然后复用 T2V 学到的后续续写能力：

   ```text
   T2V train/infer: B0=[N0], B1=[N1..NS], ...
   I2V inference:   B0=[C0], B1=[N1..NS], ...
   ```

5. 首块不长期可见。它与其他 finalized clean block 一样受 `history_blocks` 限制并正常淘汰。
6. teacher-forcing 训练路径只允许 `condition_frame_indexes_vision=[]`；`[0]`、多个条件 latent、非首帧 condition 或其他组合均显式报错。
7. 该限制只属于 causal teacher-forcing 训练；不删除普通模型的数据增强能力，也不关闭 causal checkpoint 的 I2V 推理能力。
8. packing 中每条 sample 都必须是 T2V，但仍独立采样自己的 block size 与 history window。

## 3. 实现范围

### 3.1 Geometry 与 attention

统一使用：

```python
block_id(0) = 0
block_id(t) = 1 + (t - 1) // block_size  # t >= 1
```

clean/noisy stream 必须共享完全相同的 block ID。现有规则保持不变：

- clean query 看 history window 内不晚于自己的 clean blocks；
- noisy query 看 history window 内严格早于自己的 clean blocks；
- noisy query 看自己的 noisy block；
- singleton block 0 不获得永久可见特权。

### 3.2 Sigma

每条 sample 的 sigma block 数改为：

```python
1 + ceil((num_frames - 1) / block_size)
```

采样结果按 `[1, S, S, ...]` 展开到 latent frames。训练始终是 T2V，因此首 latent 使用
singleton block 自己采样的 sigma，不再通过 condition mask 将它置零。

### 3.3 Loss

沿用现有 `condition_mask` / `mse_loss_indexes` 语义：

- `condition_frame_indexes_vision=[]`，包括首 latent 在内的全部帧都在 loss 中；
- teacher-forcing 训练不再接受 I2V condition mask；
- 不额外增加 task-specific loss 分支。

## 4. 验收条件

1. `S=1` 与 `S>1` 均产生 `[1,S,S,...]` block geometry。
2. T2V noisy block 0 看 UND 与 noisy block 0，不读取 clean block 0。
3. T2V 的第二个 block 能读取 teacher-forcing CLEAN stream 中的 singleton block 0；I2V 推理从外部 clean `C0` 初始化相同状态。
4. 当 query block 超过 `history_blocks` 窗口后，首块自然不可见。
5. packed samples 全部为 T2V，sample 之间保持隔离。
6. 任意非空 vision condition 在 teacher-forcing 训练中明确失败。
7. sigma 在 singleton 首块独立采样，后续普通 block 内共享。

## 5. 实现状态

已完成：

- 新增统一的 `[1, S, S, ...]` frame-to-block 映射，并同时用于 packing layout 与 sigma 展开；
- T2V 的 singleton block 0 与 I2V 推理的 clean 初始化状态共享相同 geometry，后续 latent 从 1 开始分块；
- teacher-forcing 模型在 timestep 采样前校验每条 packed sample 的 condition，只接受 `[]`；
- packing expansion 再次校验实际 `condition_mask`，防止绕过上游检查进入 I2V/V2V 训练；
- 所有训练 latent 均加噪并参与 loss，首 latent 使用独立 block sigma；
- causal Edge 启动脚本将数据增强分布调整为 100% T2V；
- 更新既有 geometry、mask、packing、sigma 与 T2V-only 校验测试。

验证结果：

- sequence-packing + causal model teacher-forcing tests：72 passed；
- attention/network/vertical-slice tests：29 passed；
- 相关 Python 文件 `ruff check` 通过；
- `git diff --check` 与 causal launcher `bash -n` 通过。
