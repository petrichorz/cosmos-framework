"""Structured Chinese content for the Cosmos3 V2V causal-training deck."""

from __future__ import annotations

import re
from typing import Any


SLIDES: list[dict[str, Any]] = [
    {
        "number": 1,
        "title": "Cosmos3 Edge\nV2V 与 Causal Training",
        "takeaway": "从条件帧、序列打包到时序因果训练：以当前代码行为为准",
        "bullets": ["算法研发技术说明", "代码基线 ae9a530 · 2026-07-30"],
        "references": [
            "cosmos_framework/model/generator/omni_mot_model.py:802",
            "cosmos_framework/configs/base/defaults/model_config.py:224",
        ],
        "notes": "开场先声明：本报告区分配置接口、设计意图与当前基础类的真实行为；结论以本地提交 ae9a530 为准。",
        "visual": "cover",
    },
    {
        "number": 2,
        "title": "结论先行｜当前代码的能力边界",
        "takeaway": "普通 V2V/I2V SFT 已形成闭环；严格逐帧 causal training 尚未闭环",
        "bullets": [
            "能做：前缀 clean latent 条件 + 其余 latent 联合去噪",
            "能做：CUDA/NATTEN 路径上的 temporal-causal 可见性",
            "未闭环：基础类 teacher_forcing 没有 clean-history memory",
            "Ascend 当前建议：two_way + non-causal + strategy=none",
        ],
        "references": [
            "cosmos_framework/data/generator/local_datasets/sft_dataset.py:348-359",
            "cosmos_framework/model/generator/omni_mot_model.py:768-800",
            "cosmos_framework/model/attention/natten/__init__.py:58-69",
        ],
        "notes": "算法上最重要的判断：V2V 条件机制、时序 attention mask、训练噪声策略是三件事。不要把配置名 teacher_forcing 等同于完整 clean-history teacher forcing。",
        "visual": "capability",
    },
    {
        "number": 3,
        "title": "先拆开三个概念",
        "takeaway": "条件帧决定 x₀ 是否保持 clean；causal attention 决定能看谁；strategy 决定如何加噪或建 memory",
        "bullets": [
            "Conditioning：哪些 latent frame 的 σ_eff = 0",
            "Temporal causal：query frame 是否能读取未来 supertoken",
            "Training strategy：共享 σ、逐帧 σ，或预留的 memory 逻辑",
        ],
        "references": [
            "cosmos_framework/configs/base/defaults/model_config.py:224-238",
            "cosmos_framework/data/generator/sequence_packing/sequence.py:282-297",
        ],
        "notes": "用三层图建立全篇词汇表。之后每个功能都问三个问题：帧是否 clean？能看到哪些帧？历史帧处于什么噪声状态？",
        "visual": "three_layers",
    },
    {
        "number": 4,
        "title": "V2V / I2V：条件帧只是前缀索引",
        "takeaway": "I2V 与 V2V 复用同一机制：差别只在 clean latent 前缀长度 k",
        "bullets": [
            "[]：T2V，全部 latent frame 参与生成",
            "[0]：I2V，第 0 个 latent frame 保持 clean",
            "[0…k−1]：V2V，以前 k 个 latent frame 作为视频前缀",
        ],
        "references": [
            "cosmos_framework/data/generator/local_datasets/sft_dataset.py:348-359",
            "cosmos_framework/data/generator/sequence_packing/types.py:317-336",
        ],
        "notes": "强调索引作用在 latent frame，不是原始 pixel frame。Dataset 会把采样到的 num_cond 裁到 t_latent-1，保证至少保留一个待生成帧。",
        "visual": "condition_timeline",
    },
    {
        "number": 5,
        "title": "训练主链路｜从 Dataset 到 flow-matching loss",
        "takeaway": "condition index 贯穿 clean latent、packing metadata、加噪和 loss 索引",
        "bullets": [
            "Dataset 产出视频、文本 token 与 SequencePlan",
            "VAE 得到 clean x₀；packer 写入 mask / timestep / loss index",
            "只替换生成位置为 xₜ，随后 denoise 并监督 vₜ",
        ],
        "references": [
            "cosmos_framework/model/generator/omni_mot_model.py:802-814",
            "cosmos_framework/model/generator/omni_mot_model.py:1004-1035",
            "cosmos_framework/model/generator/omni_mot_model.py:1299-1325",
        ],
        "notes": "讲解顺序沿 training_step：text → plan → VAE clean latent → sigma → pack → pre-noise hook → add noise → memory → denoise → loss。",
        "visual": "pipeline",
    },
    {
        "number": 6,
        "title": "Reasoner 与 Generator：条件信息单向流动",
        "takeaway": "当前 Vision SFT 中，Reasoner 读文本；clean/noisy 视觉 latent 位于 Generator split",
        "bullets": [
            "Understanding split：文本 causal self-attention",
            "Generator split：视觉 latent 的生成与 flow matching",
            "Generator 可读取文本条件；Reasoner 不反向读取视觉生成 token",
        ],
        "references": [
            "cosmos_framework/model/generator/omni_mot_model.py:1288-1297",
            "cosmos_framework/data/generator/sequence_packing/packers.py:203-222",
            "cosmos_framework/model/generator/mot/attention.py:574-613",
        ],
        "notes": "避免把推理侧视觉语言能力推广到这条训练链路。这里的 clean frame 不是 Reasoner 图像输入，而是 Generator 中的 VAE latent 条件。",
        "visual": "two_tower",
    },
    {
        "number": 7,
        "title": "标准 V2V token 排布｜时间先于空间",
        "takeaway": "视觉 token 物理顺序始终沿 T → H_patch → W_patch 展平",
        "bullets": [
            "文本位于 causal / understanding split",
            "视觉位于 generator / full-attention split",
            "clean 与 noisy frame 不拆 block，仍留在原始时间位置",
        ],
        "references": [
            "cosmos_framework/data/generator/sequence_packing/sequence.py:276-304",
            "cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:963-1009",
        ],
        "notes": "可用 V0 的四个 patch、V1 的四个 patch说明展平顺序。标准 non-causal 路径虽按时间排序，但 full attention 允许读取未来。",
        "visual": "token_layout",
    },
    {
        "number": 8,
        "title": "同一条时间轴上的 clean、noisy 与 loss",
        "takeaway": "condition_mask 决定是否加噪；noisy_frame_indexes 与 mse_loss_indexes 决定监督范围",
        "bullets": [
            "condition_mask=1 → σ_eff=0 → 保留 clean x₀",
            "condition_mask=0 → xₜ=interpolate(x₀, ε, σ)",
            "loss 只覆盖待生成 patch；条件 patch 不进入 MSE",
        ],
        "references": [
            "cosmos_framework/data/generator/sequence_packing/sequence.py:282-299",
            "cosmos_framework/model/generator/omni_mot_model.py:1006-1015",
            "cosmos_framework/model/generator/algorithm/loss/flow_matching.py:30-45",
        ],
        "notes": "这一页说明 V2V 的训练本质：clean 与 noisy 共存于同一 packed sequence，差异由 metadata 和替换逻辑表达。",
        "visual": "mask_mapping",
    },
    {
        "number": 9,
        "title": "Temporal causal：把每帧组织成 supertoken",
        "takeaway": "纯视觉是 [Vₜ]；有 action 时是 [Aₜ, Vₜ]，时序单元仍按 t 递增",
        "bullets": [
            "无 action：S = H_patch × W_patch",
            "有 action：S = temporal_compression_factor + H_patch × W_patch",
            "whole-clip 的首个 action 槽可为 null，真实 action 从后续帧开始",
        ],
        "references": [
            "cosmos_framework/data/generator/sequence_packing/temporal_causal.py:14-37",
            "cosmos_framework/data/generator/sequence_packing/temporal_causal.py:91-130",
            "cosmos_framework/data/generator/sequence_packing/temporal_causal.py:169-170",
        ],
        "notes": "物理 token 顺序的核心变化主要出现在 action 与 vision 的逐帧交错；纯视觉场景仍近似 V0→V1→…，真正关键是 attention metadata。",
        "visual": "supertoken",
    },
    {
        "number": 10,
        "title": "Temporal causal mask｜T 因果，S 内全连接",
        "takeaway": "第 t 个 supertoken 只能读取 0…t；同一 supertoken 内 action/patch 全可见",
        "bullets": [
            "metadata 将 generator tokens 解释为二维布局 (T, S)",
            "is_causal=(True, False)：T 维 causal，S 维 full",
            "配置强约束：video_temporal_causal=true 需要 three_way",
        ],
        "references": [
            "cosmos_framework/data/generator/sequence_packing/natten.py:448-495",
            "cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:130-134",
        ],
        "notes": "下三角矩阵只说明可见性，不说明过去 frame 是否 clean。单帧 T=1 会退化为 dense；混合 T=1/T>1 batch 会被拒绝。",
        "visual": "attention_matrix",
    },
    {
        "number": 11,
        "title": "三种训练策略｜名字相近，行为不同",
        "takeaway": "none 与当前 teacher_forcing 都是共享 σ；只有 diffusion_forcing 进入逐帧 σ 分支",
        "bullets": [
            "none：共享 σ，无额外 clean-history memory",
            "teacher_forcing：共享 σ；基础类 memory hook 为空",
            "diffusion_forcing：每个 latent frame 独立 σ；历史不保证 clean",
        ],
        "references": [
            "cosmos_framework/configs/base/defaults/model_config.py:232-238",
            "cosmos_framework/model/generator/omni_mot_model.py:1316-1325",
            "cosmos_framework/model/generator/omni_mot_model.py:1377-1394",
        ],
        "notes": "对比表把四个维度放在一起：sigma shape、显式条件帧、clean-history memory、与 temporal causal 的关系。teacher_forcing_dcm 仅作为配置接受值，不扩展为本页主策略。",
        "visual": "strategy_table",
    },
    {
        "number": 12,
        "title": "Teacher forcing｜设计意图 ≠ 当前基础类行为",
        "takeaway": "当前 V₂ 读取的是 clean V₀ + noisy V₁，而不是 clean V₀ + clean V₁",
        "bullets": [
            "理想：预测 Vₜ 时读取 clean V₀…Vₜ₋₁",
            "当前：只有显式 condition frame 保持 clean",
            "原因：pre_noise_memory_hook 原样返回；build_memory_state 返回 None",
        ],
        "references": [
            "cosmos_framework/model/generator/omni_mot_model.py:768-800",
            "cosmos_framework/model/generator/omni_mot_model.py:1004-1025",
        ],
        "notes": "这一页是核心纠偏。代码注释提到子类覆盖，但当前基础训练路径不能仅凭 strategy=teacher_forcing 推断已有 clean KV cache。",
        "visual": "teacher_forcing_gap",
    },
    {
        "number": 13,
        "title": "组合分析｜可见性 × 历史噪声状态",
        "takeaway": "因果 mask 只消除“看未来”；它不会自动把“看过去”变成 clean history",
        "bullets": [
            "non-causal + shared σ：标准联合去噪，可读取未来",
            "causal + shared σ：只读过去，但过去通常仍 noisy",
            "causal + per-frame σ：只读过去，且历史噪声等级各异",
        ],
        "references": [
            "cosmos_framework/model/generator/omni_mot_model.py:1377-1394",
            "cosmos_framework/data/generator/sequence_packing/natten.py:448-495",
        ],
        "notes": "用 2×2 决策矩阵回答常见误区：diffusion forcing 在 non-causal attention 下不是完整因果训练；temporal causal + teacher_forcing 也不等于 clean-history teacher forcing。",
        "visual": "combination_matrix",
    },
    {
        "number": 14,
        "title": "Ascend 落地｜先跑通 V2V，再补齐严格 causal",
        "takeaway": "当前稳妥配置保持 two_way / non-causal / none；严格 causal 需要 attention 与 memory 两阶段补齐",
        "bullets": [
            "现在：conditioning_config 控制 clean 前缀，完成普通 V2V/I2V SFT",
            "阶段 A：为 (T,S) temporal mask 实现 NPU attention backend",
            "阶段 B：加入逐帧 clean-history forward / KV memory",
        ],
        "references": [
            "cosmos_framework/model/attention/natten/__init__.py:58-69",
            "cosmos_framework/data/generator/sequence_packing/natten.py:488-495",
            "cosmos_framework/configs/base/experiment/sft/models/edge_model_config.py:38-57",
        ],
        "notes": "此页合并 Ascend 阻塞原因和路线，不单列阻塞链路页。指出延迟导入 Triton 或普通 SDPA fallback 不能替代当前 temporal-causal 变长多维 NATTEN backend。",
        "visual": "roadmap",
    },
    {
        "number": 15,
        "title": "Takeaways｜判断一个“causal”配置的三问",
        "takeaway": "帧是否 clean？能看哪些帧？历史帧是什么噪声状态？",
        "bullets": [
            "V2V 条件由 condition_frame_indexes_vision 表达",
            "Temporal causal 只管可见性；strategy 只管噪声或 memory",
            "当前 teacher_forcing 未形成 clean-history 闭环",
            "代码阅读顺序：Dataset → packing → training_step → attention",
        ],
        "references": [
            "cosmos_framework/data/generator/local_datasets/sft_dataset.py:348-359",
            "cosmos_framework/model/generator/omni_mot_model.py:768-802",
            "cosmos_framework/data/generator/sequence_packing/natten.py:448-495",
        ],
        "notes": "收束到三问框架，并邀请讨论：目标是联合去噪式 V2V，还是严格逐帧 AR/causal generation？两者的工程成本和训练语义不同。",
        "visual": "summary",
    },
]


_REFERENCE_RE = re.compile(r"^[\w./{}_-]+\.py:\d+(?:[–-]\d+)?$")
_PLACEHOLDER_RE = re.compile(r"\b(?:TBD|TODO)\b|待补充|占位文本", re.IGNORECASE)


def validate_content(slides: list[dict[str, Any]]) -> list[str]:
    """Return all presentation-content contract violations."""

    errors: list[str] = []
    if len(slides) != 15:
        errors.append(f"expected 15 slides, got {len(slides)}")
    if [slide.get("number") for slide in slides] != list(range(1, 16)):
        errors.append("slide numbers must be consecutive from 1 to 15")

    required = {"number", "title", "takeaway", "bullets", "references", "notes", "visual"}
    for index, slide in enumerate(slides, start=1):
        missing = required - slide.keys()
        if missing:
            errors.append(f"slide {index} missing fields: {sorted(missing)}")
            continue
        if not slide["title"] or not slide["takeaway"]:
            errors.append(f"slide {index} has an empty title or takeaway")
        if not 2 <= len(slide["bullets"]) <= 4:
            errors.append(f"slide {index} must contain 2–4 bullets")
        if not slide["references"]:
            errors.append(f"slide {index} has no code references")
        for reference in slide["references"]:
            if not _REFERENCE_RE.match(reference):
                errors.append(f"slide {index} has invalid reference: {reference}")
        serialized = " ".join(
            [str(slide["title"]), str(slide["takeaway"]), *slide["bullets"], str(slide["notes"])]
        )
        if _PLACEHOLDER_RE.search(serialized):
            errors.append(f"slide {index} contains placeholder text")

    titles = {slide.get("title") for slide in slides}
    if "Diffusion forcing：逐帧 σ" in titles:
        errors.append("removed diffusion-forcing standalone slide was restored")
    if "Ascend NPU 阻塞链路" in titles:
        errors.append("removed Ascend blocking-chain standalone slide was restored")
    return errors
