# SPDX-License-Identifier: OpenMDW-1.1
"""Source-group selection regressions for dataset observation replay."""

import json
from types import SimpleNamespace

import pytest

from cosmos_framework.inference.causal_action import replay


@pytest.fixture
def replay_group(monkeypatch, tmp_path):
    from cosmos_framework.data.generator.action.datasets import (
        cobotmagic_segment_lerobot_dataset as cobotmagic,
    )
    from cosmos_framework.data.generator.action.datasets import (
        robotwin_segment_lerobot_dataset as robotwin,
    )

    class Reader:
        default_read_options = robotwin.RoboTwinSegmentLeRobotDataset.default_read_options

        def __init__(self, *, root, **options):
            self.root, self.options = root, options

        def __len__(self):
            return ({"a": 0, "b": 1} if self.options["split"] == "val" else {"a": 2, "b": 3})[self.root.name]

    monkeypatch.setattr(cobotmagic, "CobotMagicSegmentLeRobotDataset", Reader)
    monkeypatch.setattr(robotwin, "RoboTwinSegmentLeRobotDataset", Reader)
    monkeypatch.setattr(
        "cosmos_framework.data.generator.action.lerobot_discovery.discover_dataset_roots",
        lambda paths: [tmp_path / "a", tmp_path / "b"],
    )
    monkeypatch.setattr(
        replay,
        "read_observations",
        lambda reader, index: dict(root=reader.root, local_index=index, options=reader.options),
    )
    manifest = tmp_path / "sources.json"
    manifest.write_text(
        json.dumps(
            dict(
                sources=[
                    dict(
                        reader="cobotmagic",
                        root=str(tmp_path),
                        viewpoint="concat_view",
                        read_options=dict(
                            state_key="action_unified",
                            state_mask_key="mask_action",
                            action_from_state=True,
                            action_time_offset_steps=1,
                        ),
                        split_val_ratio=0.05,
                    )
                ]
            )
        )
    )
    return SimpleNamespace(
        sources_file=str(manifest),
        source_index=0,
        dataset_root=str(tmp_path),
        dataset_index=None,
        index=0,
        split="train",
        split_seed=42,
        max_action_steps=1792,
        overlap_action_steps=321,
    )


@pytest.mark.parametrize(("index", "child", "local_index"), [(0, "a", 0), (2, "b", 0), (4, "b", 2)])
def test_manifest_indexes_across_children_and_preserves_state_contract(replay_group, index, child, local_index):
    replay_group.index = index
    data = replay.load_observations(replay_group)
    assert (data["root"].name, data["local_index"]) == (child, local_index)
    options = data["options"]["read_options"]
    assert options.state_key == "action_unified" and options.state_mask_key == "mask_action"
    assert options.action_from_state and options.action_time_offset_steps == 1
    assert data["split"] == "train" and data["source_index"] == 0


def test_explicit_child_does_not_spill_into_next_dataset(replay_group):
    replay_group.dataset_index, replay_group.index = 0, 2
    with pytest.raises(IndexError, match="outside"):
        replay.load_observations(replay_group)


def test_validation_skips_empty_children(replay_group):
    replay_group.split = "val"
    data = replay.load_observations(replay_group)
    assert data["root"].name == "b" and data["local_index"] == 0
    assert data["split"] == "val"


def test_legacy_root_preserves_explicit_dataset_selection(replay_group):
    replay_group.sources_file, replay_group.split = None, "full"
    replay_group.dataset_index, replay_group.index = 1, 1
    data = replay.load_observations(replay_group)
    assert data["root"].name == "b" and data["local_index"] == 1
    assert "read_options" not in data["options"]
    assert data["source_index"] is None


@pytest.mark.parametrize("mask_padding", [False, True])
@pytest.mark.parametrize("required", [None, 33])
def test_observation_replay_preserves_reader_padding_and_clamps_video(monkeypatch, tmp_path, mask_padding, required):
    import torch

    from lerobot.datasets import video_utils

    channels = torch.zeros(55, dtype=torch.bool)
    channels[:6] = channels[7:13] = channels[28:30] = True
    time_mask = torch.arange(64) < 50 if mask_padding else torch.ones(64, dtype=torch.bool)
    views = {"head": "high", "left": "left", "right": "right"}

    class Reader:
        root = tmp_path
        video_view = SimpleNamespace(cameras=views)
        _split_seed = 42
        _tolerance_s = 1e-4
        planner = SimpleNamespace(max_action_steps=1792, overlap_action_steps=321)

        def _resolve_index(self, index):
            return 0, 0, 0, 50

        def _read_source_fps(self, episode_id):
            return 30

        def __getitem__(self, index):
            assert self.video_view is None
            state = torch.arange(65).clamp(max=50).float()[:, None].expand(-1, 55).clone()
            return dict(
                state_trajectory=state,
                action_target=state[1:],
                state_mask=channels,
                action_mask=channels,
                state_timestamps=torch.arange(65, dtype=torch.float64) / 30,
                action_time_mask=time_mask,
                valid_action_steps=50,
                ai_caption="Pick up the box.",
            )

    from cosmos_framework.data.generator.action.action_state_template import ActionStateTemplate55

    reader = Reader()
    reader.template = ActionStateTemplate55()
    reader._direct_meta = SimpleNamespace(
        root=tmp_path,
        features={key: dict(names=["height", "width", "channels"], shape=[4, 6, 3]) for key in views.values()},
        episodes=[{f"videos/{key}/from_timestamp": 0 for key in views.values()}],
        get_video_file_path=lambda episode_id, key: f"{key}.mp4",
    )
    requested = []

    def decode(path, timestamps, tolerance, **kwargs):
        requested.append(timestamps)
        return torch.zeros(len(timestamps), 3, 4, 6, dtype=torch.uint8)

    monkeypatch.setattr(video_utils, "decode_video_frames", decode)
    data = replay.read_observations(reader, 0, required=required)
    frames = required or 65
    assert reader.video_view is not None
    assert data["state_unified"].shape == (frames, 55)
    assert data["action_target"].shape == (frames - 1, 55)
    torch.testing.assert_close(data["action_time_mask"], time_mask[: frames - 1])
    assert data["valid_action_steps"] == min(50, frames - 1)
    assert all(v.shape == (frames, 4, 6, 3) for v in data["cameras"].values())
    assert all(times[-1] == pytest.approx(min(frames - 1, 50) / 30) for times in requested)
    assert data["state"][-1]["joint_action"]["left_arm"][0] == min(frames - 1, 50)


def test_cobotmagic_mask_keeps_valid_steps_after_parent_padding(monkeypatch):
    import torch

    from cosmos_framework.data.generator.action.datasets.cobotmagic_segment_lerobot_dataset import (
        CobotMagicSegmentLeRobotDataset,
    )
    from cosmos_framework.data.generator.action.datasets.segment_lerobot_dataset import SegmentLeRobotDataset

    reader = object.__new__(CobotMagicSegmentLeRobotDataset)
    reader._direct_meta = SimpleNamespace(episodes=[dict(length=51, dataset_from_index=0)])
    reader._resolve_index = lambda index: (0, 0, 0, 50)
    monkeypatch.setattr(
        SegmentLeRobotDataset, "__getitem__", lambda self, index: dict(action_target=torch.zeros(64, 55))
    )
    sample = reader[0]
    assert len(sample["action_time_mask"]) == len(sample["action_target"]) == 64
    assert sample["valid_action_steps"] == sample["action_time_mask"].sum() == 50
    assert not sample["action_time_mask"][50:].any()
