# V2V 与 Causal Training 技术说明 PPT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成并验证一份面向算法研发、15 页、中文 16:9 的 Cosmos3 Edge V2V 与 Causal Training 技术说明 PPT。

**Architecture:** 使用一个独立 Python 生成器，以结构化的 slide 数据和可复用绘图组件构建可编辑的 PowerPoint 图形、文本和表格。生成器从已核对的代码路径写入精确引用，并同时执行结构校验；另用 PowerPoint 渲染链路生成逐页预览和联系表进行视觉巡检。

**Tech Stack:** Python 3、python-pptx、Pillow、LibreOffice headless（如环境可用）、Poppler `pdftoppm`、PowerPoint Open XML。

## Global Constraints

- 受众为算法研发人员，演示时长约 25–35 分钟。
- 页面比例必须为 16:9，成稿必须为 15 页。
- 使用深蓝黑背景；青色表示 clean/conditioning，橙色表示 noisy/generation，紫色表示 text/reasoner，红色表示限制或未实现能力。
- 每页只承载一个主结论，以一张主图和 2–4 条解释为主。
- 关键结论必须带 `file:line` 代码索引。
- 必须区分设计意图、当前基础类行为和建议实现。
- 不单独设置 diffusion forcing 详解页或 Ascend 阻塞链路页。
- 不修改 Cosmos3 训练实现或用户已有工作树改动。

---

### Task 1: 固化代码证据与演示数据

**Files:**
- Create: `tools/presentation/v2v_causal_training_content.py`
- Test: `tools/presentation/test_v2v_causal_training_content.py`

**Interfaces:**
- Produces: `SLIDES: list[dict[str, object]]`，包含 15 页标题、主结论、要点、代码引用和演讲者备注。
- Produces: `validate_content(slides: list[dict[str, object]]) -> list[str]`，返回所有内容错误。

- [ ] **Step 1: 写内容结构测试**

```python
def test_slide_contract():
    assert len(SLIDES) == 15
    assert [slide["number"] for slide in SLIDES] == list(range(1, 16))
    assert all(slide["title"] for slide in SLIDES)
    assert all(slide["takeaway"] for slide in SLIDES)
    assert validate_content(SLIDES) == []
```

- [ ] **Step 2: 运行测试并确认内容模块尚未存在**

Run: `python -m unittest tools.presentation.test_v2v_causal_training_content -v`
Expected: FAIL with import error for `v2v_causal_training_content`.

- [ ] **Step 3: 写入 15 页结构化内容**

实现 `SLIDES`，页序与已批准规格一致；引用至少覆盖 `sft_dataset.py:358`、`omni_mot_model.py:768`、`omni_mot_model.py:789`、`omni_mot_model.py:802`、`omni_mot_model.py:1378`、`cosmos3_vfm_network.py:132`、`natten.py:495` 和 `natten/__init__.py:67`。

- [ ] **Step 4: 实现内容校验**

`validate_content()` 检查页数、连续页码、必填字段、禁止占位词、引用格式 `path.py:line`，以及第 13 页和第 15 页标题没有被误恢复为已删除的独立主题。

- [ ] **Step 5: 运行测试**

Run: `python -m unittest tools.presentation.test_v2v_causal_training_content -v`
Expected: PASS.

### Task 2: 实现可复用 PPT 视觉组件和 15 页生成器

**Files:**
- Create: `tools/presentation/generate_v2v_causal_training_ppt.py`
- Create: `docs/presentations/v2v_causal_training_zh.pptx`
- Test: `tools/presentation/test_generate_v2v_causal_training_ppt.py`

**Interfaces:**
- Consumes: `SLIDES` from `v2v_causal_training_content.py`.
- Produces: `build_presentation(output_path: Path) -> Path`.
- Produces: editable PowerPoint shapes, tables, text boxes, speaker notes, and diagrams.

- [ ] **Step 1: 写结构校验测试**

```python
def test_generated_deck(tmp_path):
    path = build_presentation(tmp_path / "deck.pptx")
    prs = Presentation(path)
    assert len(prs.slides) == 15
    assert prs.slide_width / prs.slide_height == pytest.approx(16 / 9, rel=0.01)
```

测试还需确认每页至少有标题和页脚引用、主题色被使用、所有 shape 坐标均在页面边界内。

- [ ] **Step 2: 运行测试并确认生成器尚未存在**

Run: `python -m unittest tools.presentation.test_generate_v2v_causal_training_ppt -v`
Expected: FAIL with import error for generator.

- [ ] **Step 3: 实现基础组件**

实现 `add_title()`、`add_takeaway()`、`add_footer()`、`add_chip()`、`add_token_row()`、`add_attention_matrix()`、`add_comparison_table()`、`add_flow_arrow()` 和 `add_speaker_notes()`；字体优先使用 `Droid Sans Fallback`。

- [ ] **Step 4: 实现页面绘制**

按页实现封面、结论卡片、三层概念图、条件帧时间轴、训练数据流、Reasoner/Generator 双塔、token 排布、mask 元数据关系图、supertoken、注意力矩阵、策略对比表、teacher forcing 对照、二维组合矩阵、两阶段路线和总结。

- [ ] **Step 5: 写出正式 PPT**

Run: `python tools/presentation/generate_v2v_causal_training_ppt.py --output docs/presentations/v2v_causal_training_zh.pptx`
Expected: 输出 15 页 PPT，并打印内容校验和 shape 越界校验均通过。

- [ ] **Step 6: 运行生成器测试**

Run: `python -m unittest tools.presentation.test_generate_v2v_causal_training_ppt -v`
Expected: PASS.

### Task 3: 渲染、视觉巡检与最终验证

**Files:**
- Create: `docs/presentations/v2v_causal_training_zh_preview/`
- Create: `docs/presentations/v2v_causal_training_zh_contact_sheet.png`
- Modify: `tools/presentation/generate_v2v_causal_training_ppt.py` only if visual defects are found.

**Interfaces:**
- Consumes: `docs/presentations/v2v_causal_training_zh.pptx`.
- Produces: 15 张逐页 PNG 和一张联系表。

- [ ] **Step 1: 安装或确认渲染依赖**

确认 `python-pptx` 可导入；优先使用 `libreoffice --headless` 转 PDF 和 `pdftoppm` 转 PNG。若 LibreOffice 不可用，使用 PPTX 结构检查和生成器的同布局预览输出。

- [ ] **Step 2: 渲染逐页预览**

Run: `libreoffice --headless --convert-to pdf --outdir docs/presentations docs/presentations/v2v_causal_training_zh.pptx`

Run: `pdftoppm -png -r 120 docs/presentations/v2v_causal_training_zh.pdf docs/presentations/v2v_causal_training_zh_preview/slide`

Expected: 15 张 PNG。

- [ ] **Step 3: 生成联系表并视觉巡检**

使用 Pillow 将 15 张预览缩放后排成 3×5 联系表；检查文本截断、重叠、越界、对比度、色义一致性和信息密度。

- [ ] **Step 4: 修复视觉问题并重新生成**

如发现问题，只调整相关页面的布局参数，重新运行 Task 2 Step 5 和本任务 Step 2–3，直到联系表无明显缺陷。

- [ ] **Step 5: 执行最终结构与内容验证**

Run: `python -m unittest discover -s tools/presentation -p 'test_*.py' -v`

Run: `unzip -t docs/presentations/v2v_causal_training_zh.pptx`

Run: `python tools/presentation/generate_v2v_causal_training_ppt.py --check docs/presentations/v2v_causal_training_zh.pptx`

Expected: 所有测试通过；ZIP 结构完整；15 页、16:9、无越界、无占位词、关键页面含引用。

- [ ] **Step 6: 提交演示文稿相关文件**

只暂存本计划创建的 `tools/presentation/`、`docs/presentations/` 和计划文档，避免包含用户已有的 Edge YAML 改动与分析文档状态。
