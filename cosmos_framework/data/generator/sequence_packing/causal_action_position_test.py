# SPDX-License-Identifier: OpenMDW-1.1
"""State positions follow packed video boundaries, independently of storage time."""

import pytest
import torch

from cosmos_framework.data.generator.action.action_state_template import ActionStateTemplate55
from cosmos_framework.data.generator.action.block_state import build_block_sample
from cosmos_framework.data.generator.action.causal_block_geometry import CausalBlockGeometry
from cosmos_framework.data.generator.action.segment_planner import SegmentPlanner
from cosmos_framework.data.generator.sequence_packing import SequencePlan, pack_input_sequence
from cosmos_framework.data.generator.sequence_packing.causal_action import (
    CausalActionGeometry,
    dense_action_mask,
    expand_action_sequence,
)
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean


@pytest.mark.parametrize("fps_modulation", [False, True])
@pytest.mark.parametrize("mode", ["policy", "forward_dynamics", "inverse_dynamics", "policy_with_history"])
@pytest.mark.parametrize("start", [0, 17])
def test_state_positions_match_video_and_preserve_block_isolation(fps_modulation, mode, start):
    template = ActionStateTemplate55()
    geometry = CausalBlockGeometry(temporal_compression_factor=4)
    planner = SegmentPlanner(max_action_steps=96, overlap_action_steps=16, geometry=geometry)
    plans, samples, metadata, videos = [], [], [], []
    # 混合长度、空间网格和 source FPS；存储时间统一为 30 FPS。
    for n, h, w, source_fps in [(64, 2, 3, 25.0), (96, 2, 2, 29.97)]:
        state = torch.zeros(n + 1, template.width)
        state[:, 0] = torch.arange(start, start + n + 1)
        state[:, 28] = 0.5
        mask = torch.zeros(template.width, dtype=torch.bool)
        mask[[0, 28]] = True
        raw = dict(
            state_trajectory=state,
            action_target=state[1:].clone(),
            state_mask=mask,
            action_mask=mask,
            source_contract=template.source_contract("agibot", source="test", info={}, target_semantics="next state"),
            conditioning_fps=source_fps,
            storage_fps=30.0,
            state_timestamps=torch.arange(start, start + n + 1).double() / 30,
            action_timestamps=torch.arange(start, start + n).double() / 30,
        )
        sample, m = build_block_sample(raw, template=template, planner=planner, history_blocks=1)
        # block 编码不再需要任何存储时间字段。
        untimed = {k: v for k, v in raw.items() if k not in ("state_timestamps", "action_timestamps", "storage_fps")}
        other, om = build_block_sample(untimed, template=template, planner=planner, history_blocks=1)
        torch.testing.assert_close(other["action"], sample["action"])
        torch.testing.assert_close(om.anchors, m.anchors)
        t = geometry.num_latent_frames(n + 1)
        videos.append(torch.zeros(1, 2, t, h, w))
        samples.append(sample)
        metadata.append(m)
        # 在线推理当前 block=1 时，L0、L1、L2 已作为观测历史输入。
        observed = list(range(t)) if mode == "inverse_dynamics" else [0, 1, 2] if mode == "policy_with_history" else [0]
        plans.append(
            SequencePlan(
                has_text=True,
                has_vision=True,
                has_action=True,
                condition_frame_indexes_vision=observed,
                condition_frame_indexes_action=list(range(n)) if mode == "forward_dynamics" else [],
                action_start_frame_offset=1,
                causal_action_metadata=m,
            )
        )
    clean = GenerationDataClean(
        batch_size=2,
        is_image_batch=False,
        x0_tokens_vision=videos,
        x0_tokens_action=[s["action"] for s in samples],
        fps_vision=torch.stack([s["conditioning_fps"] for s in samples]),
        fps_action=torch.stack([s["conditioning_fps_action"] for s in samples]),
    )
    packed = pack_input_sequence(
        plans,
        [[11, 12], [21, 22, 23, 24, 25]],
        clean,
        torch.tensor([0.3, 0.7]),
        dict(bos_token_id=1, eos_token_id=2, start_of_generation=3, end_of_generation=4),
        enable_fps_modulation=fps_modulation,
        action_dim=template.width,
        initial_mrope_temporal_offset=7,
    )
    packed.causal_action_metadata = metadata
    if not fps_modulation:
        # 复现现有模型对 action 的 latent 相对时间修正，C07a 不改这一分支。
        packed.position_ids = packed.position_ids.float()
        vo = ao = 0
        for m, (t, h, w), s in zip(metadata, packed.vision.token_shapes, samples, strict=True):
            n = len(s["action"])
            origin = packed.position_ids[0, packed.vision.sequence_indexes[vo]]
            packed.position_ids[0, packed.action.sequence_indexes[ao : ao + n]] = origin + torch.arange(1, n + 1) / 16
            vo += t * h * w
            ao += n
    original = packed.position_ids.clone()
    expanded = expand_action_sequence(packed, videos, CausalActionGeometry((2, 2), (1, 1)))
    layout = expanded.teacher_forcing.layout
    expected_all = original[:, layout.source_sequence_indexes].clone()
    vo = so = ao = 0
    for sample_id, (m, (t, h, w), sample) in enumerate(zip(metadata, packed.vision.token_shapes, samples, strict=True)):
        states = layout.state_indexes[so : so + len(m.states)]
        vi = packed.vision.sequence_indexes[vo + m.state_latent_indexes * h * w]
        expected_all[0, states] = original[0, vi]
        torch.testing.assert_close(expanded.position_ids[0, states], original[0, vi])
        torch.testing.assert_close(
            expanded.position_ids[1:, states], original[1:, layout.source_sequence_indexes[states]]
        )
        # 前一 block 的末 action 与后一 state 同时间，但 attention 不能读取后一 state。
        last_actions = expanded.action.sequence_indexes[ao + m.state_action_indexes[1:] - 1]
        torch.testing.assert_close(expanded.position_ids[0, last_actions], expanded.position_ids[0, states[1:]])
        vo += t * h * w
        so += len(m.states)
        ao += len(sample["action"])
    torch.testing.assert_close(expanded.position_ids, expected_all)
    torch.testing.assert_close(packed.position_ids, original)
    torch.testing.assert_close(expanded.action.timesteps, packed.action.timesteps)
    torch.testing.assert_close(expanded.vision.timesteps, packed.vision.timesteps)
    for actual, expected in zip(expanded.action.tokens, clean.x0_tokens_action, strict=True):
        torch.testing.assert_close(actual, expected)
    visible = dense_action_mask(layout)
    queries = layout.gen_query_indexes
    cross_sample = layout.sample_ids[queries, None] != layout.sample_ids[None, :]
    assert not visible[cross_sample].any()
    # 任一 block 都不能看到下一 block 的 state；当前 state 仅对当前非 history 流可见。
    future_state = (layout.stream_ids[None, :] == 3) & (layout.block_ids[None, :] > layout.block_ids[queries, None])
    assert not visible[future_state].any()
    expired_history = (layout.stream_ids[None, :] == 0) & (
        layout.block_ids[None, :] < layout.block_ids[queries, None] - 1
    )
    assert not visible[expired_history].any()
