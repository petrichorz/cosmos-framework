# SPDX-License-Identifier: OpenMDW-1.1
"""Isolated causal DROID adapter; ordinary video/action recipes are unchanged."""

import math
import random
from dataclasses import replace

import torch

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


def pad_causal_actions(data, *, use_state, compression=4, video_stride=1):
    """Subsample synchronized video only; retain every source action.

    The source reader supplies one action per video interval. Padding occupies
    one latent group (compression * video_stride) BEFORE normalization and is
    fixed conditioning (ACTION-001), not a supervised stationary command.
    """
    if isinstance(video_stride, bool) or not isinstance(video_stride, int) or video_stride < 1:
        raise ValueError("video_stride must be a positive integer")
    data = dict(data)
    action = data["action"]
    frames = data["video"].shape[1]
    state = int(use_state)
    if action.shape[0] != frames - 1 + state or frames <= 1 or (frames - 1) % (compression * video_stride):
        raise ValueError("Expected one action per source-frame interval and (frames - 1) divisible by 4 * video_stride")
    padding = compression * video_stride
    data["video"] = data["video"][:, ::video_stride].contiguous() # 按步长做下采样 TODO 这里可以和动态fps结合；
    if "conditioning_fps" in data:
        source_fps = torch.as_tensor(data["conditioning_fps"]).float()
        if "conditioning_fps_action" in data and not torch.allclose(
            torch.as_tensor(data["conditioning_fps_action"]).float(), source_fps
        ):
            raise ValueError("video_stride requires synchronized, same-rate source video and actions")
        data["conditioning_fps_action"] = source_fps
        data["conditioning_fps"] = source_fps / video_stride
    data["action"] = torch.cat([action[:state], action.new_zeros(padding, action.shape[-1]), action[state:]]) # 填充第一个block
    return data


class CausalActionSFTDataset(ActionSFTDataset):
    def __init__(self, base, *, mode, use_state, joint_mode_weights=None, video_stride=1):
        super().__init__(base._dataset, base._transform, base._resolution)
        if mode not in (*MODES, "joint"):
            raise ValueError(f"Unsupported causal action mode: {mode}")
        if isinstance(video_stride, bool) or not isinstance(video_stride, int) or video_stride < 1:
            raise ValueError("video_stride must be a positive integer")
        self.video_stride = video_stride
        self.mode = mode
        self.use_state = use_state
        self.weights = validate_joint_weights(joint_mode_weights)
        self.debug_fixed_index = None

    def __getitem__(self, idx):
        idx = idx if self.debug_fixed_index is None else self.debug_fixed_index
        rng = random.getstate()
        try:
            if self.debug_fixed_index is not None:
                random.seed(42)
            raw = pad_causal_actions(self._dataset[idx], use_state=self.use_state, video_stride=self.video_stride)
        finally:
            if self.debug_fixed_index is not None:
                random.setstate(rng)
        mode = random.choices(MODES, weights=self.weights, k=1)[0] if self.mode == "joint" else self.mode
        raw["mode"] = mode
        result = self._transform(raw, self._resolution, action_normalizer=self._dataset.get_action_normalizer())
        length = result["action"].shape[0]
        result["sequence_plan"] = replace(
            result["sequence_plan"],
            condition_frame_indexes_action=list(
                range(length if mode == "forward_dynamics" else 4 * self.video_stride + int(self.use_state))
            ),
            action_start_frame_offset=1 - 4 * self.video_stride - int(self.use_state),
        )
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
    **kwargs,
):
    if kwargs.get("action_space", "joint_pos") != "joint_pos":
        raise ValueError("First causal action version supports joint_pos only (ACTION-002)")
    if isinstance(video_stride, bool) or not isinstance(video_stride, int) or video_stride < 1:
        raise ValueError("video_stride must be a positive integer")
    if kwargs.get("chunk_length", 32) < 4 * video_stride or kwargs.get("chunk_length", 32) % (4 * video_stride):
        raise ValueError("chunk_length must be a positive multiple of 4 * video_stride")
    # The raw reader uses fixed policy; sampling occurs once here after reading.
    kwargs["dataset_version"] = kwargs.get("dataset_version") or "droid_plus_lerobot_640x360_20260412"
    base = get_action_droid_sft_dataset(mode="policy", use_state=use_state, iterable_shuffle=False, **kwargs)
    result = CausalActionSFTDataset(
        base, mode=mode, use_state=use_state, joint_mode_weights=joint_mode_weights, video_stride=video_stride
    )
    result.debug_fixed_index = debug_fixed_index
    return ActionIterableShuffleDataset(result, seed=episode_shuffle_seed) if iterable_shuffle else result
