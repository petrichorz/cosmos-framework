# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json

import numpy as np
import pytest
import torch

from cosmos_framework.data.generator.action.action_state_template import ActionStateTemplate55
from cosmos_framework.data.generator.action.block_state import BlockStatistics, Quantiles
from cosmos_framework.inference.causal_action.robotwin import RobotwinAdapter


def make_adapter(*, eef=True, joint=True):
    mask = np.zeros(55, dtype=bool)
    if joint:
        mask[:6] = mask[7:13] = True
    if eef:
        mask[14:28] = True
    mask[28:30] = True
    return RobotwinAdapter(ActionStateTemplate55(), state_mask=mask, action_mask=mask)


def native_state():
    return dict(
        joint_action=dict(left_arm=np.arange(6), right_arm=np.arange(6) + 10, left_gripper=0.25, right_gripper=0.75),
        endpose=dict(left_endpose=[1, 2, 3, 0.5, 0.5, 0.5, 0.5], right_endpose=[4, 5, 6, 0, 0, 0, 1]),
    )


def test_native_state_preserves_dataset_xyz_xyzw_and_packs_joints_grippers():
    adapter = make_adapter()
    state = native_state()
    packed = adapter.assemble_state(state)
    np.testing.assert_array_equal(packed[:7], [0, 1, 2, 3, 4, 5, 0])
    np.testing.assert_array_equal(packed[7:14], [10, 11, 12, 13, 14, 15, 0])
    np.testing.assert_array_equal(packed[21:28], [4, 5, 6, 0, 0, 0, 1])
    np.testing.assert_array_equal(packed[28:30], [0.25, 0.75])
    np.testing.assert_array_equal(packed[30:], 0)
    sanitized = adapter.template.sanitize(torch.from_numpy(packed), adapter.state_mask)
    np.testing.assert_array_equal(sanitized.numpy(), packed)
    # Round-trip to an offline native replay does not invent invalid slots.
    replay = adapter.state_for_replay(packed)
    for key in state["endpose"]:
        np.testing.assert_array_equal(replay["endpose"][key], state["endpose"][key])


def test_native_actions_use_robotwin_interleaved_gripper_order_and_quaternion_order():
    adapter = make_adapter()
    action = np.tile(adapter.assemble_state(native_state()), (32, 1))
    result = adapter.split_actions(action)
    assert result["joint"].shape == (32, 14)
    assert result["eef"].shape == (32, 16)
    np.testing.assert_array_equal(result["joint"][0], [0, 1, 2, 3, 4, 5, 0.25, 10, 11, 12, 13, 14, 15, 0.75])
    np.testing.assert_array_equal(result["eef"][0], [1, 2, 3, 0.5, 0.5, 0.5, 0.5, 0.25, 4, 5, 6, 0, 0, 0, 1, 0.75])


def test_joint_model_ignores_untrained_eef_state_and_does_not_export_zeros_as_eef_actions():
    adapter = make_adapter(eef=False)
    state = native_state()
    state["endpose"]["left_endpose"] = [float("nan")] * 7
    packed = adapter.assemble_state(state)
    assert np.isfinite(packed).all()
    assert not packed[14:28].any()
    assert adapter.split_actions(packed[None])["eef"] is None
    assert adapter.available_control_modes == ["joint"]
    del state["joint_action"]["left_arm"]
    with pytest.raises(ValueError, match="left_arm"):
        adapter.assemble_state(state)


def test_eef_model_does_not_require_joint_fields():
    adapter = make_adapter(joint=False)
    state = native_state()
    state["endpose"].update(left_gripper=0.25, right_gripper=0.75)
    del state["joint_action"]
    packed = adapter.assemble_state(state)
    assert adapter.available_control_modes == ["eef"]
    assert adapter.split_actions(packed[None])["joint"] is None
    del state["endpose"]["right_endpose"]
    with pytest.raises(ValueError, match="right_endpose"):
        adapter.assemble_state(state)


def test_statistics_source_masks_must_be_unambiguous_and_config_must_be_covered(tmp_path):
    adapter = make_adapter()
    q = Quantiles(torch.zeros(55), torch.ones(55), adapter.state_mask)
    mask = adapter.state_mask.tolist()
    source = dict(state_mask=mask, action_mask=mask, source="/unavailable/dataset")
    statistics = BlockStatistics(q, q, dict(sources=[source, source]))
    loaded = RobotwinAdapter.from_statistics(adapter.template, statistics)
    assert loaded.available_control_modes == ["joint", "eef"]
    other = make_adapter(eef=False).state_mask.tolist()
    statistics.provenance["sources"][1] = dict(state_mask=other, action_mask=other)
    with pytest.raises(ValueError, match="robot-config"):
        RobotwinAdapter.from_statistics(adapter.template, statistics)
    config = tmp_path / "robot.json"
    config.write_text(json.dumps(dict(state_mask=other, action_mask=other)))
    assert RobotwinAdapter.from_statistics(adapter.template, statistics, config).available_control_modes == ["joint"]
    statistics.state.valid = torch.tensor(other)
    config.write_text(json.dumps(dict(state_mask=mask, action_mask=mask)))
    with pytest.raises(ValueError, match="Statistics do not cover"):
        RobotwinAdapter.from_statistics(adapter.template, statistics, config)


def test_profile_rejects_extra_joint_slots_and_bad_native_shapes():
    adapter = make_adapter()
    sm = adapter.state_mask.clone()
    sm[6] = True
    with pytest.raises(ValueError, match="outside"):
        RobotwinAdapter(adapter.template, state_mask=sm, action_mask=adapter.action_mask)
    state = native_state()
    state["joint_action"]["left_arm"] = [0] * 7
    with pytest.raises(ValueError, match="6 values"):
        adapter.assemble_state(state)
    with pytest.raises(ValueError, match="native"):
        adapter.assemble_state(np.zeros(55))
