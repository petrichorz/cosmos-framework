# SPDX-License-Identifier: OpenMDW-1.1
"""CPU checks of C01 contracts; no datasets, video decoders or model weights."""

from dataclasses import replace

import pytest
import torch

from cosmos_framework.data.generator.action.action_state_template import (
    ActionStateTemplate55,
    TemplateSourceContract,
)
from cosmos_framework.data.generator.action.sample_contract import (
    ActionReadOptions,
)


def make_contract(template):
    names = template.fields.keys()
    return TemplateSourceContract(
        template_id=template.template_id,
        source="synthetic",
        frame="robot_base",
        endpoints={name: "test_endpoint" for name in names if name.endswith("_eef")},
        units={name: "metres,unit_quaternion" if name.endswith("_eef") else "source_native" for name in names},
        scalar_semantics=dict.fromkeys(names, "absolute"),
        target_semantics="existing absolute action command at each interval start",
    )


def template_values():
    template = ActionStateTemplate55()
    mask = torch.ones(template.width, dtype=torch.bool)
    mask[template.fields["reserved"]] = False
    anchor = torch.full((template.width,), 0.25)
    target = torch.full((3, template.width), 0.75)
    for group in template.rotation_groups:
        anchor[list(group)] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        target[:, list(group)] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    anchor[~mask] = float("nan")
    target[:, ~mask] = float("nan")
    return template, mask, anchor, target


def test_absolute_hands_grippers_and_relative_fields_round_trip():
    template, mask, anchor, target = template_values()
    contract = make_contract(template)
    delta = template.encode_action_delta(target, anchor, mask, source_contract=contract)
    for name in ("left_gripper", "right_gripper", "left_hand", "right_hand"):
        torch.testing.assert_close(delta[:, template.fields[name]], target[:, template.fields[name]])
    torch.testing.assert_close(delta[:, template.fields["left_arm_joint"]], torch.full((3, 7), 0.5))
    assert (delta[:, ~mask] == 0).all()
    restored = template.decode_action_delta(delta, anchor, mask, source_contract=contract)
    torch.testing.assert_close(restored, template.sanitize(target, mask))
    assert torch.isnan(target[:, ~mask]).all()  # Validation/conversion does not mutate input.


def test_quaternion_composition_order_and_sign():
    template, mask, anchor, target = template_values()
    # Noncommuting 90-degree rotations expose an incorrect multiplication order.
    half = 2**-0.5
    for group in template.rotation_groups:
        anchor[list(group)] = torch.tensor([half, 0.0, 0.0, half])
        target[:, list(group)] = torch.tensor([0.0, -half, 0.0, -half])
    contract = make_contract(template)
    delta = template.encode_action_delta(target, anchor, mask, source_contract=contract)
    decoded = template.decode_action_delta(delta, anchor, mask, source_contract=contract)
    for group in template.rotation_groups:
        torch.testing.assert_close(delta[:, list(group)], torch.tensor([[-0.5, 0.5, -0.5, 0.5]]).expand(3, -1))
        torch.testing.assert_close(decoded[:, list(group)], torch.tensor([[0.0, half, 0.0, half]]).expand(3, -1))


def test_template_rejects_invalid_representation():
    template, mask, anchor, target = template_values()
    contract = make_contract(template)
    with pytest.raises(ValueError, match="version"):
        template.validate_source_contract(replace(contract, template_id="old-template"), mask)
    with pytest.raises(ValueError, match="vector"):
        template.validate_valid_mask(mask[:-1])
    partial = mask.clone()
    partial[template.rotation_groups[0][0]] = False
    with pytest.raises(ValueError, match="entirely"):
        template.validate_valid_mask(partial)
    target[:, template.rotation_groups[0]] = 0
    with pytest.raises(ValueError, match="nonzero"):
        template.encode_action_delta(target, anchor, mask, source_contract=contract)


def test_read_options_are_explicit_and_do_not_relabel():
    options = ActionReadOptions()
    assert options.action_from_state is False
    assert options.target_key == "action" and options.action_time_offset_steps == 0
    assert options.state_mask_key == "mask_state" and options.target_mask_key == "mask_action"
    next_state = ActionReadOptions(state_key="robot.state", action_from_state=True, action_time_offset_steps=1)
    assert next_state.target_key == "robot.state" and next_state.action_time_offset_steps == 1
    assert next_state.target_mask_key == "mask_state"
    custom = ActionReadOptions(state_mask_key="robot.valid_state", action_mask_key="robot.valid_action")
    assert custom.target_mask_key == "robot.valid_action"
    assert replace(custom, action_from_state=True).target_mask_key == "robot.valid_state"
    for kwargs in (
        {"action_time_offset_steps": True},
        {"action_time_offset_steps": 0.5},
        {"action_from_state": "false"},
        {"state_key": ""},
        {"state_mask_key": " "},
        {"action_mask_key": None},
    ):
        with pytest.raises(ValueError):
            ActionReadOptions(**kwargs)
