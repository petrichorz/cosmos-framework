# SPDX-License-Identifier: OpenMDW-1.1
"""Template-driven causal training adapter for segmented readers."""

import math
import random
from dataclasses import replace

from .action_sft_dataset import ActionSFTDataset

MODES = ("forward_dynamics", "inverse_dynamics", "policy")


def validate_joint_weights(weights=None):
    weights = dict.fromkeys(MODES, 1.0) if weights is None else dict(weights)
    if set(weights) != set(MODES):
        raise ValueError(f"joint_mode_weights requires exactly {MODES}")
    values = tuple(float(weights[m]) for m in MODES)
    if any(not math.isfinite(x) or x < 0 for x in values) or sum(values) <= 0:
        raise ValueError("joint_mode_weights must be finite, nonnegative, and have positive sum")
    return values


class CausalActionSFTDataset(ActionSFTDataset):
    """Any source implementing the measured-state contract can share this path."""

    def __init__(
        self,
        base,
        *,
        mode,
        statistics_path=None,
        statistics=None,
        allow_mock_statistics=False,
        history_blocks_min=1,
        history_blocks_max=8,
        joint_mode_weights=None,
        use_state=True,
    ):
        super().__init__(base._dataset, base._transform, base._resolution)
        if mode not in (*MODES, "joint"):
            raise ValueError(f"Unsupported causal action mode: {mode}")
        if not use_state or (statistics_path is None and statistics is None):
            raise ValueError("Measured block states and separate state/delta statistics are required")
        if not 1 <= history_blocks_min <= history_blocks_max:
            raise ValueError("Invalid history sampling range")
        if statistics_path is not None and statistics is not None:
            raise ValueError("Provide statistics or statistics_path, not both")
        self.template = self._dataset.template
        self.planner = self._dataset.planner
        self.mode = mode
        self.history_range = (history_blocks_min, history_blocks_max)
        self.statistics_path = statistics_path
        self.statistics = statistics
        self._statistics_validated = False
        self.allow_mock_statistics = allow_mock_statistics
        self.weights = validate_joint_weights(joint_mode_weights)
        self.debug_fixed_index = None

    def __getitem__(self, idx):
        from cosmos_framework.data.generator.action.block_state import BlockStatistics, build_block_sample

        idx = idx if self.debug_fixed_index is None else self.debug_fixed_index
        rng = random.getstate()
        try:
            if self.debug_fixed_index is not None:
                random.seed(42)
            raw = self._dataset[idx]
        finally:
            if self.debug_fixed_index is not None:
                random.setstate(rng)
        if self.statistics is None:
            self.statistics = BlockStatistics.load(self.statistics_path)
        if not self._statistics_validated:
            self.statistics.validate(
                self.template.width, raw["state_mask"], raw["action_mask"], allow_mock=self.allow_mock_statistics
            )
            self._statistics_validated = True
        mode = random.choices(MODES, weights=self.weights, k=1)[0] if self.mode == "joint" else self.mode
        raw, metadata = build_block_sample(
            raw,
            template=self.template,
            planner=self.planner,
            history_blocks=random.randint(*self.history_range),
            statistics=self.statistics,
        )
        for key in (
            "state_trajectory",
            "action_target",
            "state_mask",
            "action_mask",
            "action_state_indexes",
            "state_timestamps",
            "action_timestamps",
            "storage_fps",
            "source_contract",
            "read_options",
        ):
            raw.pop(key, None)
        raw["mode"] = mode
        result = self._transform(raw, self._resolution, action_normalizer=None)
        length = result["action"].shape[0]
        if result["action"].shape[-1] != self.template.width:
            raise ValueError("The shared action transform must use template.width without extra padding")
        result["sequence_plan"] = replace(
            result["sequence_plan"],
            condition_frame_indexes_action=list(range(length if mode == "forward_dynamics" else 0)),
            action_start_frame_offset=1,
            causal_action_metadata=metadata,
        )
        result["raw_action_dim"] = None
        result["causal_action_mode"] = mode
        return result
