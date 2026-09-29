# SPDX-License-Identifier: OpenMDW-1.1
"""Template-based execution state and absolute target recovery."""

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.action.action_state_template import ActionStateTemplate55
from cosmos_framework.data.generator.action.block_state import BlockStatistics, Quantiles, build_block_sample
from cosmos_framework.data.generator.action.causal_block_geometry import CausalBlockGeometry
from cosmos_framework.data.generator.action.segment_planner import SegmentPlanner
from cosmos_framework.inference.causal_action.session import CausalActionSession


def inputs():
    template = ActionStateTemplate55()
    state = torch.zeros(template.width)
    state[0] = 10
    sm = torch.zeros(template.width, dtype=torch.bool)
    sm[0] = True
    am = sm.clone()
    am[template.fields["left_gripper"]] = True
    # 无效的夹爪 state 不应阻止绝对夹爪目标，也不应传出 NaN。
    state[template.fields["left_gripper"]] = float("nan")
    return template, dict(state=state, state_mask=sm, action_mask=am, image=torch.zeros(3, 4, 4))


def test_session_absolute_field_and_history_reset():
    template, data = inputs()
    calls = []

    def planner(**kwargs):
        calls.append(kwargs)
        return torch.ones(32, template.width)

    session = CausalActionSession(None, planner, history_blocks=1, template=template)
    out = session.generate_current_block(**data)
    assert torch.isfinite(calls[0]["state"]).all()
    torch.testing.assert_close(calls[0]["action_mask"], data["action_mask"])
    assert out[:, 28].eq(1).all()
    assert not out[:, ~data["action_mask"]].any()
    with pytest.raises(RuntimeError, match="feedback"):
        session.generate_current_block(**data)
    for _ in range(2):
        session.commit_execution_feedback(observation=data["image"], executed_steps=32)
        session.generate_current_block(**data)
        assert len(calls[-1]["history"]) == 1
    assert calls[-1]["revision"] == 2
    fresh = CausalActionSession(None, planner, template=template)
    fresh.generate_current_block(**data)
    assert calls[-1]["history"] == []
    data["action_mask"][1] = True
    with pytest.raises(ValueError, match="anchor"):
        fresh.commit_execution_feedback(observation=data["image"], executed_steps=32)
        fresh.generate_current_block(**data)


def test_model_session_decodes_with_current_template():
    template, data = inputs()
    contract = template.source_contract("agibot", source="test", info={}, target_semantics="absolute")
    planner = SegmentPlanner(
        max_action_steps=32, overlap_action_steps=0, geometry=CausalBlockGeometry(temporal_compression_factor=4)
    )
    state = template.sanitize(data["state"], data["state_mask"])
    target = torch.zeros(32, template.width)
    target[:, 0] = state[0] + torch.arange(1, 33)
    target[:, 28] = 0.5
    raw = dict(
        state_trajectory=state.expand(33, -1),
        action_target=target,
        state_mask=data["state_mask"],
        action_mask=data["action_mask"],
        source_contract=contract,
        conditioning_fps=30.0,
    )
    low, high = torch.full((template.width,), -100.0), torch.full((template.width,), 100.0)
    stats = BlockStatistics(Quantiles(low, high, data["state_mask"]), Quantiles(low, high, data["action_mask"]))
    sample, metadata = build_block_sample(raw, template=template, planner=planner, history_blocks=1, statistics=stats)
    batch = dict(sequence_plan=[SimpleNamespace(causal_action_metadata=metadata)])

    def batch_builder(**kwargs):
        torch.testing.assert_close(kwargs["state"], state)
        torch.testing.assert_close(kwargs["action_mask"], data["action_mask"])
        return batch

    class Model:
        def generate_samples_from_batch(self, batch, **kwargs):
            assert kwargs["causal_block_size"] == 2
            return dict(action=[sample["action"]])

    session = CausalActionSession.for_model(contract, Model(), batch_builder, template=template)
    decoded = session.generate_current_block(**data)
    torch.testing.assert_close(decoded, target, atol=1e-5, rtol=1e-5)
