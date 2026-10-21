# SPDX-License-Identifier: OpenMDW-1.1
"""CobotMagic measured-state trajectories with explicit terminal padding."""

from pathlib import Path

import torch

from cosmos_framework.data.generator.action.sample_contract import ActionReadOptions
from cosmos_framework.data.generator.action.video_view import VideoViewConfig

from .segment_lerobot_dataset import ActionTrainingRange, SegmentLeRobotDataset

CAMERAS = {
    "head": "observation.images.cam_high",
    "left": "observation.images.cam_left_wrist",
    "right": "observation.images.cam_right_wrist",
}
READ_OPTIONS = ActionReadOptions(
    state_key="state_unified",
    action_key="action_unified",
    state_mask_key="mask_state",
    action_mask_key="mask_action",
    action_from_state=True,
    action_time_offset_steps=1,
)


class CobotMagicSegmentLeRobotDataset(SegmentLeRobotDataset):
    default_read_options = READ_OPTIONS

    def __init__(
        self,
        *,
        viewpoint="concat_view",
        read_options=READ_OPTIONS,
        full_episode=True,
        load_video=True,
        **kwargs,
    ):
        self.full_episode = full_episode
        if kwargs.get("table_backend", "parquet") != "parquet":
            raise ValueError("CobotMagic requires the direct parquet backend")
        if read_options != READ_OPTIONS:
            raise ValueError("CobotMagic targets must be the next measured state")
        # Resolve per-subset holdouts without splitting one mixture group into tiny sources.
        ratios = kwargs.get("split_val_ratio")
        if isinstance(ratios, dict):
            subset = Path(kwargs["root"]).name
            if subset not in ratios:
                raise ValueError(f"Missing CobotMagic validation ratio for subset: {subset}")
            kwargs["split_val_ratio"] = ratios[subset]
        super().__init__(
            viewpoint=viewpoint,
            read_options=read_options,
            video_view=VideoViewConfig(cameras=CAMERAS.copy()) if load_video else None,
            **kwargs,
        )

    def _iter_training_ranges(self, episode_id, episode_length):
        episode = self._direct_meta.episodes[episode_id]
        original = int(episode["original_length"])
        if not 2 <= original <= episode_length:
            raise ValueError("Invalid CobotMagic original episode length")
        if self.full_episode:
            self.planner.geometry.validate_observation_frames(episode_length)
            if episode_length - 1 > self.planner.max_action_steps:
                raise ValueError(f"Episode {episode_id} exceeds the full-episode action limit")
        yield ActionTrainingRange(0, episode_length if self.full_episode else original)

    def _read_video(self, meta, episode_id, timestamps, *, viewpoint):
        last = (int(meta.episodes[episode_id]["original_length"]) - 1) / self.fps
        return super()._read_video(meta, episode_id, [min(t, last) for t in timestamps], viewpoint=viewpoint)

    def __getitem__(self, idx):
        _, episode_id, start, actions = self._resolve_index(idx)
        episode = self._direct_meta.episodes[episode_id]
        local_start = start - int(episode["dataset_from_index"])
        valid_actions = min(actions, int(episode["original_length"]) - 1 - local_start)
        sample = super().__getitem__(idx)
        sample["cobotmagic_target"] = sample["action_target"].clone()
        sample["action_time_mask"] = torch.arange(actions) < valid_actions
        sample["valid_action_steps"] = valid_actions
        sample["episode_index"] = episode_id
        return sample
