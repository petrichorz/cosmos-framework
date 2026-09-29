# SPDX-License-Identifier: OpenMDW-1.1
"""Reader boundary and alignment checks without video/model dependencies."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from datasets import Dataset

from cosmos_framework.data.generator.action.action_state_template import ActionStateTemplate55
from cosmos_framework.data.generator.action.causal_block_geometry import CausalBlockGeometry
from cosmos_framework.data.generator.action.datasets import segment_lerobot_dataset as module
from cosmos_framework.data.generator.action.sample_contract import ActionReadOptions
from cosmos_framework.data.generator.action.segment_planner import SegmentPlanner


def make_reader(monkeypatch, *, lengths=(100, 50), offset=0, from_state=False, root="synthetic"):
    template = ActionStateTemplate55()
    size = sum(lengths)
    state = torch.arange(size, dtype=torch.float32)[:, None].expand(-1, template.width).clone()
    mask = torch.zeros(template.width, dtype=torch.bool)
    mask[template.fields["left_arm_joint"]] = True
    columns = {
        "observation.state": state,
        "action": state + 1000,
        "mask_state": mask.expand(size, -1),
        "mask_action": mask.expand(size, -1),
        "timestamp": torch.cat([torch.arange(n, dtype=torch.float64) / 30 for n in lengths]),
        "task_index": torch.zeros(size, dtype=torch.long),
    }
    episodes = []
    start = 0
    for n in lengths:
        episodes.append({"dataset_from_index": start, "dataset_to_index": start + n})
        start += n
    meta = SimpleNamespace(
        fps=30,
        features={key: {"shape": list(value.shape[1:])} for key, value in columns.items()},
        episodes=episodes,
        total_episodes=len(lengths),
        root=root,
        tasks=pd.DataFrame({"task_index": [0]}, index=["test task"]),
    )
    monkeypatch.setattr(module, "LeRobotDatasetMetadata", lambda **kwargs: meta)
    queries = []

    def query(indices):
        queries.append(indices)
        return {key: columns[key][rows] for key, rows in indices.items()}

    ds = SimpleNamespace(
        meta=meta,
        _query_hf_dataset=query,
        _ensure_hf_dataset_loaded=lambda: None,
        hf_dataset=Dataset.from_dict({"timestamp": columns["timestamp"].tolist()}),
    )
    monkeypatch.setattr(module.SegmentLeRobotDataset, "_get_dataset", lambda self, index: ds)
    reader = module.SegmentLeRobotDataset(
        root=root,
        table_backend="hf",
        template=template,
        source_contract=template.source_contract("agibot", source="test", info={}, target_semantics="absolute"),
        planner=SegmentPlanner(
            max_action_steps=64, overlap_action_steps=16, geometry=CausalBlockGeometry(temporal_compression_factor=4)
        ),
        read_options=ActionReadOptions(action_from_state=from_state, action_time_offset_steps=offset),
    )
    return reader, queries


@pytest.mark.parametrize("offset", [-2, -1, 0, 1, 2, 3])
@pytest.mark.parametrize("from_state", [False, True])
def test_targets_use_one_offset_and_stay_in_episode(monkeypatch, offset, from_state):
    reader, queries = make_reader(monkeypatch, lengths=(210, 170), offset=offset, from_state=from_state)
    for idx in range(len(reader)):
        _, ep, start, count = reader._segments[idx]
        sample = reader[idx]
        expected = torch.arange(start, start + count) + offset
        expected = expected + (0 if from_state else 1000)
        torch.testing.assert_close(sample["action_target"][:, 0], expected.float())
        assert sample["state_trajectory"].shape[0] == count + 1
        assert sample["conditioning_fps"] == 30
        begin, end = ((0, 210), (210, 380))[ep]
        assert start >= begin and start + count < end
        assert min(expected).item() - (0 if from_state else 1000) >= begin
        assert max(expected).item() - (0 if from_state else 1000) < end
    # 多次取样依然只查询一次数据集级 mask。
    assert sum("mask_state" in query for query in queries) == 1


def test_variable_lengths_and_shuffle_indexes(monkeypatch):
    reader, _ = make_reader(monkeypatch)
    assert sorted(reader[i]["action_target"].shape[0] for i in range(len(reader))) == [32, 64, 64]
    covered = [i for start, count in reader.get_shuffle_blocks() for i in range(start, start + count)]
    assert covered == list(range(len(reader)))
    assert reader[-1]["ai_caption"] == "test task"
    with pytest.raises(IndexError):
        reader[len(reader)]


def test_next_state_needs_only_endpoint_observation(monkeypatch):
    reader, _ = make_reader(monkeypatch, lengths=(33,), from_state=True, offset=1)
    assert len(reader) == 1
    assert reader[0]["action_target"][-1, 0] == 32


def test_short_episode_warns_and_is_skipped(monkeypatch):
    with pytest.warns(UserWarning, match="at least 33 required"):
        reader, _ = make_reader(monkeypatch, lengths=(32,))
    assert len(reader) == 0
    assert reader.planner.skipped_ranges == 1


def test_timestamp_precision_and_video_hook(monkeypatch):
    reader, _ = make_reader(monkeypatch, lengths=(65,))
    ds = reader._get_dataset(0)
    # 1000 秒附近 float32 相邻帧差分已不足以满足采样率校验。
    timestamps = (1000 + torch.arange(65, dtype=torch.float64) / 30).tolist()
    ds.hf_dataset = Dataset.from_dict({"timestamp": timestamps})
    received = []
    monkeypatch.setattr(reader, "_read_video", lambda ds, episode, times, *, viewpoint: received.append(times))
    sample = reader[0]
    assert sample["state_timestamps"].tolist() == timestamps
    assert received == [timestamps]


def test_dataset_contract_is_checked_only_when_loading_masks(monkeypatch):
    reader, _ = make_reader(monkeypatch)
    calls = []
    validate = reader.template.validate_source_contract

    def record(contract, mask):
        calls.append(mask)
        validate(contract, mask)

    monkeypatch.setattr(reader.template, "validate_source_contract", record)
    reader[0]
    assert len(calls) == 2
    reader[1]
    assert len(calls) == 2


def test_result_uses_parent_video_format_and_keeps_absolute_targets(monkeypatch):
    reader, _ = make_reader(monkeypatch, lengths=(65,), from_state=True, offset=1)
    frames = torch.full((3, 65, 2, 2), 0.5)
    monkeypatch.setattr(reader, "_read_video", lambda ds, episode, times, *, viewpoint: frames)
    sample = reader[0]
    assert isinstance(sample, dict)
    assert sample["action"] is sample["action_target"]
    assert sample["video"].dtype == torch.uint8
    assert sample["video"].shape == frames.shape
    assert (sample["video"] == 127).all()
    torch.testing.assert_close(sample["action_target"], sample["state_trajectory"][1:])
    torch.testing.assert_close(sample["action_mask"], sample["state_mask"])
    assert sample["mode"] is None and sample["domain_id"] is None


def test_source_fps_uses_dense_array_with_missing_field_fallback(monkeypatch, tmp_path):
    path = tmp_path / "meta" / "episodes.jsonl"
    path.parent.mkdir()
    # 故意打乱 JSONL 顺序，必须按 episode_index 而非行号关联。
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"episode_index": 2},
                {"episode_index": 1, "source_fps": 29.97},
                {"episode_index": 0, "source_fps": 25},
            ]
        )
    )
    reader, _ = make_reader(monkeypatch, root=tmp_path, lengths=(100, 100, 100))
    assert reader._episode_fps.dtype == np.float32
    assert reader._episode_fps.nbytes == 3 * 4
    assert np.isnan(reader._episode_fps[2])
    assert not reader._episode_fps.flags.writeable
    path.unlink()  # getitem 只查数组，不再打开 JSONL。
    for index, (_, episode, _, _) in enumerate(reader._segments):
        sample = reader[index]
        assert sample["source_fps"].item() == pytest.approx({0: 25, 1: 29.97, 2: 30}[episode])
        assert sample["conditioning_fps"].item() == sample["source_fps"].item()
        assert sample["storage_fps"].item() == 30
    other, _ = make_reader(monkeypatch)  # 不同根目录独立读取。
    assert other[0]["source_fps"].item() == 30


def test_reader_rejects_nonuniform_storage_timestamps(monkeypatch):
    reader, _ = make_reader(monkeypatch, lengths=(65,))
    ds = reader._get_dataset(0)
    times = [i / 30 for i in range(65)]
    times[12] += 0.01
    ds.hf_dataset = Dataset.from_dict({"timestamp": times})
    with pytest.raises(ValueError, match="timestamp intervals must match meta.fps"):
        reader[0]


@pytest.mark.parametrize("fps", [0, -1, None, True, float("nan"), float("inf")])
def test_invalid_source_fps_does_not_fall_back(monkeypatch, tmp_path, fps):
    path = tmp_path / "meta" / "episodes.jsonl"
    path.parent.mkdir()
    path.write_text(json.dumps({"episode_index": 0, "source_fps": fps}))
    with pytest.raises(ValueError, match="Invalid source_fps"):
        make_reader(monkeypatch, root=tmp_path, lengths=(65,))


def test_resized_video_conversion_is_opt_in(monkeypatch):
    reader, _ = make_reader(monkeypatch, lengths=(33,))
    frames = torch.full((33, 3, 2, 2), 128, dtype=torch.uint8)
    with pytest.raises(TypeError, match="floating-point"):
        reader._convert_video(frames)
    reader._pyav_resize = True
    assert torch.equal(reader._convert_video(frames), frames.permute(1, 0, 2, 3))
    with pytest.raises(TypeError, match="uint8"):
        reader._convert_video(frames.float())
    with pytest.raises(ValueError, match="shape"):
        reader._convert_video(frames[0])
    assert reader._convert_video(None) is None
    reader._skip_video_loading = True
    assert reader._convert_video(frames) is None
