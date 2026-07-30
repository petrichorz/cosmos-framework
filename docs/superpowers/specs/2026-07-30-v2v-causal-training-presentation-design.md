# V2V 与 Causal Training 技术说明 PPT 设计

## 目标

基于 `docs/v2v_causal_training_analysis_zh.md` 和当前代码仓，为算法研发人员制作一份中文技术说明 PPT。演示应帮助听众建立 V2V 条件帧、temporal causal attention 与 causal training strategy 之间的清晰边界，并能从关键代码路径验证结论。

## 受众与场景

- 受众：算法研发人员
- 时长：约 25–35 分钟
- 比例：16:9
- 形式：机制推导为主，代码证据和可用性边界为辅
- 基线：本地提交 `ae9a530`，分析日期 2026-07-30

## 核心信息

1. V2V/I2V 的 clean frame 以 VAE latent 进入 Generator，不是当前 Vision SFT 中 Reasoner 的图片输入。
2. `condition_frame_indexes_vision` 决定 clean 条件帧；其余帧被加噪并参与 flow-matching loss。
3. `video_temporal_causal` 管注意力可见性，`causal_training_strategy` 管噪声或 memory 策略，二者不能混为一谈。
4. 当前基础类的 `teacher_forcing` 没有 clean-history memory，不能视为完整逐帧 teacher forcing。
5. `diffusion_forcing` 实现逐 latent frame 的独立 sigma，但不自动提供 clean history。
6. Ascend 当前稳妥路线是普通 V2V/I2V SFT；严格 causal training 仍需补 temporal-causal NPU attention 和 clean-history teacher forcing。

## 页面结构

共 15 页：

1. 封面：Cosmos3 Edge V2V 与 Causal Training
2. 一页结论：当前代码能做什么、不能做什么
3. 概念分层：条件帧、temporal causal attention、causal training strategy
4. V2V/I2V 的条件帧表达：`condition_frame_indexes_vision`
5. 端到端训练数据流：Dataset → latent → packing → denoise → loss
6. Reasoner 与 Generator 的职责边界
7. 标准 V2V token 排布与 patch 展平顺序
8. clean/noisy token、mask、timestep 与 loss 的关系
9. temporal causal 的 supertoken 排布
10. `(T,S)` 注意力可见性矩阵与 `three_way` 约束
11. `none / teacher_forcing / diffusion_forcing` 对比总表
12. 理想 teacher forcing 与当前实现的差距
13. 组合分析：attention 可见性 × 噪声策略
14. 当前可落地配置与两阶段补齐路线
15. 总结、代码阅读路径与讨论问题

不单独设置 diffusion forcing 详解页或 Ascend 阻塞链路页。必要信息分别合并进第 11、13、14 页。

## 视觉系统

- 背景：深蓝黑技术风
- clean/conditioning：青色
- noisy/generation：橙色
- text/reasoner：紫色
- 限制、未实现或风险：红色
- 中性元数据与辅助线：冷灰色
- 字体：优先使用系统可用的中文无衬线字体

图示使用一致的视觉语义：

- 时间序列：`V₀ … Vₜ`
- temporal causal：`T × S` supertoken 方块
- 注意力可见性：下三角矩阵
- clean/noisy：同一时间轴上的青色/橙色块
- 设计意图与当前实现：左右对照图

## 页面密度与讲解原则

- 每页只承载一个主结论。
- 每页以一张主图为中心，辅以 2–4 条短解释。
- 不贴大段源码；代码片段只保留决定行为的条件或返回值。
- 每个关键结论在页脚给出 `file:line` 形式的代码或文档索引。
- 明确标注“设计意图”“当前基础类行为”“建议实现”，避免从配置名推断不存在的能力。
- 演讲者备注补充代码位置、常见误区和可能的追问。

## 内容证据

主要证据来自：

- `docs/v2v_causal_training_analysis_zh.md`
- `cosmos_framework/data/generator/local_datasets/sft_dataset.py`
- `cosmos_framework/data/generator/augmentors/sequence_plan.py`
- `cosmos_framework/data/generator/sequence_packing/{packers,sequence,temporal_causal,natten}.py`
- `cosmos_framework/model/generator/omni_mot_model.py`
- `cosmos_framework/model/generator/mot/{cosmos3_vfm_network,attention}.py`
- `cosmos_framework/model/attention/{frontend,natten/__init__}.py`
- `cosmos_framework/configs/base/defaults/model_config.py`

制作时重新提取精确行号，以当前工作树为准。

## 交付与验证

交付：

- 一份可编辑的 `.pptx`
- 必要时保留生成脚本，便于后续更新代码行号或内容

验证：

- PowerPoint 文件可被解析和重新打开
- 页数为 15，页面比例为 16:9
- 中文字体不溢出，图形和表格不越界
- 关键结论均有代码索引
- 无 `TBD`、`TODO`、占位文本或与当前代码矛盾的能力声明
- 将各页渲染为图片并进行视觉巡检
