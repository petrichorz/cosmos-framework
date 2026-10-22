# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Pack state/actions into unified55 without dataset-specific pose conversions."""

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np


class RobotwinAdapter:
    def __init__(
        self,
        template,
        *,
        state_mask,
        action_mask,
        left_arm_dof=6,
        right_arm_dof=6,
    ):
        if template.template_id != "unified55-xyzw-absolute-gripper-hand-v4":
            raise ValueError("RoboTwin native serving requires ActionStateTemplate55")
        for dof in (left_arm_dof, right_arm_dof):
            if type(dof) is not int or not 1 <= dof <= 7:
                raise ValueError("RoboTwin arm DOF must be an integer in [1,7]")
        self.template = template
        self.state_mask = template.validate_valid_mask(state_mask).cpu().clone()
        self.action_mask = template.validate_valid_mask(action_mask).cpu().clone()
        if not self.state_mask.any() or not self.action_mask.any():
            raise ValueError("RoboTwin state/action masks must be nonempty")
        if (template.required_anchor_mask(self.action_mask) & ~self.state_mask).any():
            raise ValueError("Relative action channels require valid state anchors")
        self.dofs = {"left": left_arm_dof, "right": right_arm_dof}
        self.joint_slots = {}
        for side, dof in self.dofs.items():
            slots = template.fields[f"{side}_arm_joint"]
            self.joint_slots[side] = np.arange(slots.start, slots.start + dof)
        self.gripper_slots = np.array([template.fields[f"{side}_gripper"].start for side in self.dofs])
        self.eef_slots = np.concatenate([np.arange(*template.fields[f"{side}_eef"].indices(55)) for side in self.dofs])
        joint_slots = np.concatenate(list(self.joint_slots.values()))
        supported = np.zeros(55, dtype=bool)
        supported[np.r_[joint_slots, self.eef_slots, self.gripper_slots]] = True
        if ((self.state_mask.numpy() | self.action_mask.numpy()) & ~supported).any():
            raise ValueError("Masks enable slots outside this robot's joints, EEFs and grippers")
        am = self.action_mask.numpy()
        self.available_control_modes = [
            mode
            for mode, slots in (("joint", joint_slots), ("eef", self.eef_slots))
            if am[np.r_[slots, self.gripper_slots]].all()
        ]
        if not self.available_control_modes:
            raise ValueError("Action mask must cover both arms and grippers for at least one execution mode")

    @classmethod
    def from_statistics(cls, template, statistics, config_path=None):
        """Use exact source masks, never the union of group-statistics valid slots."""
        config = json.loads(Path(config_path).read_text()) if config_path else {}
        if "eef_pose_frame" in config or "T_world_base" in config:
            raise ValueError("Configure EEF transforms on robotwin_client with --eef-config, not on the server")
        sources = statistics.provenance.get("sources", [])
        if ("state_mask" in config) != ("action_mask" in config):
            raise ValueError("--robot-config must provide both state_mask and action_mask when overriding masks")
        if "state_mask" not in config and "action_mask" not in config:
            pairs = {(tuple(source["state_mask"]), tuple(source["action_mask"])) for source in sources}
            if len(pairs) != 1:
                raise ValueError("Statistics must contain identical source masks; otherwise pass --robot-config")
            sm, am = pairs.pop()
            config.update(state_mask=list(sm), action_mask=list(am))
        adapter = cls(template, **config)
        statistics.validate(template.width, adapter.state_mask, adapter.action_mask)
        return adapter

    @property
    def metadata(self):
        return dict(
            state_format="robotwin-native",
            action_format="robotwin-native",
            template_id=self.template.template_id,
            quaternion_order="xyzw",
            eef_pose_frame="dataset",
            left_arm_dof=self.dofs["left"],
            right_arm_dof=self.dofs["right"],
            available_control_modes=self.available_control_modes,
        )

    def assemble_state(self, state):
        """Read dataset-frame xyz+xyzw poses; apply server mask before normalization."""
        if not isinstance(state, Mapping):
            raise ValueError("state must contain RoboTwin native joint_action and endpose dictionaries")
        joint, eef = state.get("joint_action", {}), state.get("endpose", {})
        if not isinstance(joint, Mapping) or not isinstance(eef, Mapping):
            raise ValueError("joint_action and endpose must be dictionaries")
        result = np.zeros(55, dtype=np.float32)
        mask = self.state_mask.numpy()

        def read(values, key, width):
            if key not in values:
                raise ValueError(f"Missing state field: {key}")
            array = np.asarray(values[key], dtype=np.float32)
            expected = () if width == 1 else (width,)
            if array.shape not in (expected, (width,)):
                raise ValueError(f"{key} must contain {width} values, got shape {array.shape}")
            return array.reshape(width)

        for side, dof in self.dofs.items():
            slots = self.joint_slots[side]
            if mask[slots].any():
                result[slots] = read(joint, f"{side}_arm", dof)
            slots = self.template.fields[f"{side}_eef"]
            if mask[slots].any():
                result[slots] = read(eef, f"{side}_endpose", 7)
            slot = self.template.fields[f"{side}_gripper"].start
            if mask[slot]:
                key = f"{side}_gripper"
                result[slot] = read(joint if key in joint else eef, key, 1)[0]
        result[~mask] = 0
        # The template performs finite checks and quaternion sanitization in policy.infer.
        return result

    def split_actions(self, action):
        """Pack absolute joints or dataset-frame xyz+xyzw EEFs with interleaved grippers."""
        action = np.asarray(action, dtype=np.float32)
        if action.ndim != 2 or action.shape[1] != 55 or not np.isfinite(action).all():
            raise ValueError("Generated unified action must be finite [T,55]")
        result = dict(joint=None, eef=None)
        for mode in self.available_control_modes:
            parts = []
            for side in self.dofs:
                if mode == "joint":
                    arm = action[:, self.joint_slots[side]]
                else:
                    arm = action[:, self.template.fields[f"{side}_eef"]]
                parts.extend([arm, action[:, self.template.fields[f"{side}_gripper"]]])
            result[mode] = np.concatenate(parts, axis=1)
        return result

    def state_for_replay(self, unified):
        """Pack a training observation for replay, preserving dataset EEF values."""
        unified = np.asarray(unified, dtype=np.float32)
        if unified.shape != (55,):
            raise ValueError("Replay state must have shape [55]")
        joint, eef = {}, {}
        for side in self.dofs:
            joint[f"{side}_arm"] = unified[self.joint_slots[side]]
            joint[f"{side}_gripper"] = unified[self.template.fields[f"{side}_gripper"]][0]
            if self.state_mask[self.template.fields[f"{side}_eef"]].any():
                eef[f"{side}_endpose"] = unified[self.template.fields[f"{side}_eef"]]
        return dict(joint_action=joint, endpose=eef)
