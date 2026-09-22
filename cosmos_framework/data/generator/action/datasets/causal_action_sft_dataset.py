# SPDX-License-Identifier: OpenMDW-1.1
"""Multi-source causal mid-training adapter; DROID is one debugging source."""

import math
import random
from dataclasses import replace

from .action_sft_dataset import ActionIterableShuffleDataset, ActionSFTDataset, get_action_droid_sft_dataset

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
        statistics_path,
        block_size_min=1,
        block_size_max=4,
        history_blocks_min=1,
        history_blocks_max=8,
        joint_mode_weights=None,
        video_stride=1,
        use_state=True,
    ):
        super().__init__(base._dataset, base._transform, base._resolution)
        if mode not in (*MODES, "joint"):
            raise ValueError(f"Unsupported causal action mode: {mode}")
        if not use_state or not statistics_path:
            raise ValueError("Measured block states and separate state/delta statistics are required")
        if not 1 <= block_size_min <= block_size_max or not 1 <= history_blocks_min <= history_blocks_max:
            raise ValueError("Invalid geometry sampling range")
        if isinstance(video_stride, bool) or not isinstance(video_stride, int) or video_stride < 1:
            raise ValueError("video_stride must be a positive integer")
        self.video_stride = video_stride
        self.mode = mode
        self.block_sizes = tuple(range(block_size_min, block_size_max + 1))
        self.history_range = (history_blocks_min, history_blocks_max)
        self.statistics_path = statistics_path
        self.statistics = None
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
        key = raw["source_contract"].statistics_key(self.block_sizes, self.video_stride, len(raw["action_target"]))
        if self.statistics is None:
            self.statistics = BlockStatistics.load(self.statistics_path, key)
        if self.statistics.key != key:
            raise ValueError("Each source/representation contract requires its own statistics")
        mode = random.choices(MODES, weights=self.weights, k=1)[0] if self.mode == "joint" else self.mode
        raw, metadata = build_block_sample(
            raw,
            block_size=random.choice(self.block_sizes),
            history_blocks=random.randint(*self.history_range),
            video_stride=self.video_stride,
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
            "source_contract",
        ):
            raw.pop(key)
        raw["mode"] = mode
        result = self._transform(raw, self._resolution, action_normalizer=None)
        length = result["action"].shape[0]
        if result["action"].shape[-1] != 80:
            raise ValueError("The shared action transform must use max_action_dim=80")
        result["sequence_plan"] = replace(
            result["sequence_plan"],
            condition_frame_indexes_action=list(range(length if mode == "forward_dynamics" else 0)),
            action_start_frame_offset=1,
            causal_action_metadata=metadata,
        )
        result["raw_action_dim"] = None
        result["causal_action_mode"] = mode
        return result


def get_causal_action_droid_sft_dataset(
    *,
    mode="joint",
    joint_mode_weights=None,
    use_state=True,
    iterable_shuffle=True,
    episode_shuffle_seed=42,
    debug_fixed_index=None,
    video_stride=1,
    statistics_path=None,
    block_size_min=1,
    block_size_max=4,
    history_blocks_min=1,
    history_blocks_max=8,
    **kwargs,
):
    if kwargs.get("action_space", "causal_eef") != "causal_eef":
        raise ValueError("Causal mid-training uses measured EEF poses, not joint vectors")
    if video_stride not in (1, 2, 4):
        raise ValueError("video_stride must be 1, 2 or 4")
    if kwargs.get("chunk_length", 32) < 4 * video_stride or kwargs.get("chunk_length", 32) % (4 * video_stride):
        raise ValueError("chunk_length must be a positive multiple of 4 * video_stride")
    kwargs["action_space"] = "causal_eef"
    kwargs["max_action_dim"] = 80
    kwargs["dataset_version"] = kwargs.get("dataset_version") or "droid_plus_lerobot_640x360_20260412"
    base = get_action_droid_sft_dataset(mode="policy", use_state=True, iterable_shuffle=False, **kwargs)
    result = CausalActionSFTDataset(
        base,
        mode=mode,
        use_state=use_state,
        joint_mode_weights=joint_mode_weights,
        video_stride=video_stride,
        statistics_path=statistics_path,
        block_size_min=block_size_min,
        block_size_max=block_size_max,
        history_blocks_min=history_blocks_min,
        history_blocks_max=history_blocks_max,
    )
    result.debug_fixed_index = debug_fixed_index
    return ActionIterableShuffleDataset(result, seed=episode_shuffle_seed) if iterable_shuffle else result
