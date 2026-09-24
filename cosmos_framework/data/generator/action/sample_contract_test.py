# SPDX-License-Identifier: OpenMDW-1.1
"""CPU checks of C01 contracts; no datasets, video decoders or model weights."""

from dataclasses import replace

import pytest
import torch

from cosmos_framework.data.generator.action.action_state_template import (
    ActionStateTemplate,
    ActionStateTemplate55,
    TemplateSourceContract,
    resolve_action_template,
)
from cosmos_framework.data.generator.action.sample_contract import (
    ActionReadOptions,
    RawActionSample,
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


class ScalarTemplate(ActionStateTemplate):
    """An unrelated width/layout verifies that the raw contract has no slot assumptions."""

    template_id = "two-scalars-v1"
    width = 2
    fields = {"coordinates": slice(0, 2)}
    rotation_groups = ()

    def validate_valid_mask(self, valid_mask, *, template_id=None):
        if valid_mask.shape != (self.width,) or valid_mask.dtype != torch.bool:
            raise ValueError("Invalid scalar mask")
        return valid_mask

    def validate_source_contract(self, contract, valid_mask):
        self.validate_valid_mask(valid_mask)
        if contract.template_id != self.template_id:
            raise ValueError("Template mismatch")

    def sanitize(self, values, valid_mask):
        return values.masked_fill(~self.validate_valid_mask(valid_mask), 0)

    def validate(self, state, action, valid_mask, source_contract):
        self.validate_source_contract(source_contract, valid_mask)
        torch.broadcast_shapes(state.shape, action.shape)

    def encode_action_delta(self, absolute_action, anchor_state, valid_mask, *, source_contract):
        return self.sanitize(absolute_action - anchor_state, valid_mask)

    def decode_action_delta(self, action_delta, anchor_state, valid_mask, *, source_contract):
        return self.sanitize(action_delta + anchor_state, valid_mask)


def make_sample(template):
    return RawActionSample(
        state_trajectory=torch.zeros(4, template.width),
        action_target=torch.ones(3, template.width),
        state_mask=torch.ones(template.width, dtype=torch.bool),
        action_mask=torch.ones(template.width, dtype=torch.bool),
        state_timestamps=torch.arange(4, dtype=torch.float64) / 10,
        action_timestamps=torch.arange(3, dtype=torch.float64) / 10,
        action_state_indexes=torch.arange(3),
        source_contract=make_contract(template),
        conditioning_fps=10,
    )


def test_state_derived_target_requires_state_mask():
    template = ScalarTemplate()
    sample = make_sample(template)
    sample.read_options = ActionReadOptions(action_from_state=True, action_time_offset_steps=1)
    sample.validate(template)
    sample.action_mask[1] = False
    with pytest.raises(ValueError, match="state mask"):
        sample.validate(template)


def test_raw_contract_accepts_another_template_and_independent_masks():
    template = resolve_action_template(ScalarTemplate)
    sample = make_sample(template)
    sample.action_mask[1] = False
    sample.action_target[:, 1] = float("nan")
    sample.validate(template)  # No block geometry or minimum of 33 belongs in C01.
    assert resolve_action_template(template) is template


@pytest.mark.parametrize(
    "field,value",
    [
        ("action_target", torch.ones(3, 3)),
        ("state_trajectory", torch.zeros(3, 2)),
        ("action_state_indexes", torch.tensor([1, 2, 3])),
        ("action_timestamps", torch.tensor([0.0, 0.1, 0.1])),
        ("action_timestamps", torch.tensor([0.1, 0.2, 0.3])),
        ("conditioning_fps", 0),
        ("video", torch.zeros(3, 3, 8, 8)),
    ],
)
def test_raw_contract_rejects_misalignment(field, value):
    template = ScalarTemplate()
    with pytest.raises(ValueError):
        replace(make_sample(template), **{field: value}).validate(template)
