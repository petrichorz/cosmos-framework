# SPDX-License-Identifier: OpenMDW-1.1
"""Construct template readers and the shared causal training mixture from source declarations."""

import json
import os
from dataclasses import replace
from pathlib import Path

from cosmos_framework.data.generator.action.action_state_template import resolve_action_template
from cosmos_framework.data.generator.action.block_statistics import mock_template_statistics
from cosmos_framework.data.generator.action.causal_block_geometry import CausalBlockGeometry
from cosmos_framework.data.generator.action.segment_planner import SegmentPlanner
from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline

from .action_sft_dataset import ActionSFTDataset
from .agibot_segment_lerobot_dataset import AgiBotSegmentLeRobotDataset
from .causal_action_mixture import CausalActionMixture
from .causal_action_sft_dataset import CausalActionSFTDataset
from .egosuite_segment_lerobot_dataset import EgoSuiteSegmentLeRobotDataset

READERS = {"agibot": AgiBotSegmentLeRobotDataset, "egosuite": EgoSuiteSegmentLeRobotDataset}


def template_width(template):
    """配置解析时从模板获取模型和 transform 共用的宽度。"""
    return resolve_action_template(template).width


def latent_block_size(actions_per_block, video_stride):
    """将固定 action block 换算为模型使用的 latent 帧数。"""
    return CausalBlockGeometry(
        actions_per_block=actions_per_block, video_stride=video_stride, temporal_compression_factor=4
    ).latent_frames_per_block


def video_durations(max_action_steps, actions_per_block, video_stride):
    """列出所有合法片段抽帧后的 VAE 输入长度，包括独立首帧。"""
    geometry = CausalBlockGeometry(
        actions_per_block=actions_per_block, video_stride=video_stride, temporal_compression_factor=4
    )
    geometry.validate_observation_frames(max_action_steps + 1)
    return [1 + n // video_stride for n in range(actions_per_block, max_action_steps + 1, actions_per_block)]


def get_causal_action_dataset(
    *,
    sources_file,
    template,
    actions_per_block=32,
    video_stride=4,
    max_action_steps=96,
    overlap_action_steps=16,
    mode="joint",
    history_blocks_min=1,
    history_blocks_max=8,
    seed=42,
    resolution="480",
    tokenizer_config=None,
    cfg_dropout_rate=0.1,
    allow_mock_statistics=False,
    mock_state_stats_path="",
    mock_delta_stats_path="",
):
    """来源文件只声明路径/读取规则；模板、geometry 和 transform 共用任务级配置。"""
    template = resolve_action_template(template)
    geometry = CausalBlockGeometry(
        actions_per_block=actions_per_block, video_stride=video_stride, temporal_compression_factor=4
    )
    planner = SegmentPlanner(
        max_action_steps=max_action_steps, overlap_action_steps=overlap_action_steps, geometry=geometry
    )
    manifest = Path(sources_file)
    # 环境变量仅用于部署路径；来源语义和读取选项保留在版本化 JSON 中。
    sources = json.loads(os.path.expandvars(manifest.read_text()))["sources"]
    datasets, weights = [], []
    transform = ActionTransformPipeline(
        max_action_dim=template.width,
        video_temporal_downsample=4,
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        format_prompt_as_json=True,
    )
    for source in sources:
        root = Path(source["root"])
        profile = source["reader"]
        info = json.loads((root / "meta/info.json").read_text())
        options = replace(READERS[profile].default_read_options, **source.get("read_options", {}))
        reader = READERS[profile](
            root=root,
            template=template,
            planner=planner,
            source_contract=template.source_contract(
                profile, source=str(root), info=info, target_semantics=source["target_semantics"]
            ),
            read_options=options,
            viewpoint=source["viewpoint"],
            split="train",
            split_seed=seed,
            split_val_ratio=source.get("split_val_ratio", 0.0),
            tolerance_s=source.get("tolerance_s", 1e-4),
            video_backend=source.get("video_backend", "pyav"),
        )
        if not len(reader):
            raise ValueError(f"No complete training segments: {root}")
        statistics_path = source.get("statistics_path")
        statistics = None
        if not statistics_path:
            if not allow_mock_statistics:
                raise ValueError(f"{root}: specify statistics_path or explicitly enable mock statistics")
            statistics = mock_template_statistics(
                reader[0],
                template=template,
                planner=reader.planner,
                state_stats_path=mock_state_stats_path,
                delta_stats_path=mock_delta_stats_path,
            )
        dataset = CausalActionSFTDataset(
            ActionSFTDataset(reader, transform, resolution),
            mode=source.get("mode", mode),
            statistics_path=statistics_path,
            statistics=statistics,
            allow_mock_statistics=allow_mock_statistics,
            history_blocks_min=history_blocks_min,
            history_blocks_max=history_blocks_max,
            joint_mode_weights=source.get("joint_mode_weights"),
        )
        datasets.append(dataset)
        weights.append(source.get("weight", 1.0))
    return CausalActionMixture(datasets, weights, seed=seed)
