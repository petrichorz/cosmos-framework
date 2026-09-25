# SPDX-License-Identifier: OpenMDW-1.1
"""Template block statistics: shared encoding and explicitly marked mock ranges."""

import json
from dataclasses import asdict

from cosmos_framework.data.generator.action.block_state import BlockStatistics, Quantiles, build_block_sample


def collect_statistics(dataset, *, indices):
    """仅从训练 split 的片段拟合统计；与训练使用相同 anchor 和模板编码。"""
    states, smasks, actions, amasks = [], [], [], []
    for i in indices:
        raw = dataset[i]
        if raw["source_contract"].split != "train":
            raise ValueError("Quantiles must be collected on the training split")
        data, metadata = build_block_sample(
            raw,
            template=dataset.template,
            planner=dataset.planner,
            history_blocks=1,
        )
        states.append(metadata.states)
        smasks.append(metadata.state_mask)
        actions.append(data["action"])
        amasks.append(metadata.action_mask)
    if not states:
        raise ValueError("Statistics population is empty")
    return BlockStatistics(
        Quantiles.fit(states, smasks),
        Quantiles.fit(actions, amasks),
        provenance={
            "kind": "fitted",
            "split": "train",
            "segments": len(states),
            "geometry": asdict(dataset.planner.geometry),
        },
    )


def mock_template_statistics(raw, *, template, planner, state_stats_path, delta_stats_path):
    """显式读取 AgiBot 模拟统计；字段映射/占位范围只能由模板定义。"""
    with open(state_stats_path) as f:
        state_stats = json.load(f)
    with open(delta_stats_path) as f:
        delta_stats = json.load(f)
    sm = template.validate_valid_mask(raw["state_mask"])
    am = template.validate_valid_mask(raw["action_mask"])
    state_bounds, _, state_fields = template.mock_quantiles(state_stats, delta_stats, sm)
    _, action_bounds, action_fields = template.mock_quantiles(state_stats, delta_stats, am)
    return BlockStatistics(
        Quantiles(*state_bounds, sm.clone()),
        Quantiles(*action_bounds, am.clone()),
        provenance={
            "kind": "mock",
            "geometry": asdict(planner.geometry),
            "state_stats_path": str(state_stats_path),
            "delta_stats_path": str(delta_stats_path),
            "state_fields": state_fields,
            "action_fields": action_fields,
        },
    )
