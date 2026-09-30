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


def make_reader(monkeypatch, *, lengths=(100, 50), offset=0, from_state=False, root="synthetic", use_subtask=False):
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
        use_subtask=use_subtask,
    )
    return reader, queries


@pytest.mark.parametrize("offset", [-2, 0, 1, 3])
@pytest.mark.parametrize("from_state", [False, True])
def test_subtask_extension_covers_boundaries_and_keeps_targets_aligned(monkeypatch, tmp_path, offset, from_state):
    annotations = [[(0, 33), (40, 177), (177, 181)], [(0, 65)]]
    meta = tmp_path / "meta"
    meta.mkdir()
    episodes = [
        {
            "episode_index": ep,
            "action_config": [
                {"start_frame": lo, "end_frame": hi, "action_text": f"{ep}/{i}"} for i, (lo, hi) in enumerate(ranges)
            ],
        }
        for ep, ranges in enumerate(annotations)
    ]
    (meta / "episodes.jsonl").write_text("\n".join(json.dumps(ep) for ep in episodes))
    reader, _ = make_reader(
        monkeypatch, lengths=(181, 65), offset=offset, from_state=from_state, root=tmp_path, use_subtask=True
    )
    for (begin_idx, size), (lo, hi, first, stop) in zip(reader.get_shuffle_blocks(), reader._range_bounds):
        covered = set()
        for idx in range(begin_idx, begin_idx + size):
            _, ep, start, actions = reader._segments[idx]
            begin, end = ((0, 181), (181, 246))[ep]
            covered.update(range(start - begin, start - begin + actions))
            assert begin <= start and start + actions < end
            assert begin <= start + offset and start + actions - 1 + offset < end
            sample = reader[idx]
            expected = torch.arange(start + offset, start + offset + actions).float()
            torch.testing.assert_close(sample["action_target"][:, 0], expected + (0 if from_state else 1000))
            torch.testing.assert_close(
                sample["state_trajectory"][:, 0], torch.arange(start, start + actions + 1).float()
            )
            assert sample["ai_caption"] == reader._caption_for_index(begin_idx)
        legal_end = min(end - begin - 1, end - begin - offset)
        assert set(range(max(lo, -offset, 0), min(hi, legal_end))) <= covered
        assert covered == set(range(first, stop - 1))
    assert reader.planner.discarded_action_steps == reader.planner.skipped_ranges == 0


@pytest.mark.parametrize(
    "bounds,length,expected",
    [
        ((0, 181), 602, (0, 193)),
        ((181, 421), 602, (181, 438)),
        ((421, 602), 602, (409, 602)),
        ((0, 33), 100, (0, 65)),  # 已整除也不能漏掉动作起点 32。
        ((90, 99), 100, (67, 100)),
        ((10, 11), 100, (10, 43)),
        ((0, 181), 181, (0, 181)),  # 整个 episode 不够向上补齐，交给 planner 保尾。
    ],
)
def test_subtask_borrows_forward_before_backward(bounds, length, expected):
    reader = object.__new__(module.SegmentLeRobotDataset)
    reader.read_options = ActionReadOptions(action_time_offset_steps=1)
    reader.planner = SegmentPlanner(
        max_action_steps=896, overlap_action_steps=16, geometry=CausalBlockGeometry(temporal_compression_factor=4)
    )
    assert reader._subtask_read_bounds(module.ActionTrainingRange(*bounds), length) == expected


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


@pytest.mark.parametrize(
    "bad_config",
    [
        None,
        [],
        {},
        "bad",
        [None],
        [{"start_frame": 0, "end_frame": 32}],
        [{"start_frame": True, "end_frame": 32, "action_text": "bad"}],
        [{"start_frame": -1, "end_frame": 32, "action_text": "bad"}],
        [{"start_frame": 20, "end_frame": 20, "action_text": "bad"}],
        [{"start_frame": 0, "end_frame": 101, "action_text": "bad"}],
        [{"start_frame": 0, "end_frame": 32, "action_text": " "}],
        [
            {"start_frame": 0, "end_frame": 40, "action_text": "valid first"},
            {"start_frame": 39, "end_frame": 65, "action_text": "overlap"},
        ],
    ],
)
@pytest.mark.parametrize("offset", [-2, 0, 1, 3])
def test_invalid_subtask_falls_back_only_affected_episode(monkeypatch, tmp_path, bad_config, offset):
    meta = tmp_path / "meta"
    meta.mkdir()
    first = {"episode_index": 0}
    if bad_config is not None:
        first["action_config"] = bad_config
    second = {
        "episode_index": 1,
        "action_config": [{"start_frame": 10, "end_frame": 42, "action_text": "valid subtask"}],
    }
    (meta / "episodes.jsonl").write_text("\n".join(json.dumps(x) for x in [first, second]))
    baseline, _ = make_reader(monkeypatch, lengths=(100, 100), root=tmp_path, offset=offset)
    with pytest.warns(UserWarning, match="Episode 0.*falling back to episode mode"):
        reader, _ = make_reader(monkeypatch, lengths=(100, 100), root=tmp_path, offset=offset, use_subtask=True)
    assert [s for s in reader._segments if s[1] == 0] == [s for s in baseline._segments if s[1] == 0]
    for index, segment in enumerate(reader._segments):
        sample = reader[index]
        if segment[1] == 0:
            assert sample["ai_caption"] == "test task"
            expected = baseline[index]
            for key in ["action_target", "state_trajectory", "state_timestamps", "action_timestamps"]:
                torch.testing.assert_close(sample[key], expected[key])
        else:
            assert sample["ai_caption"] == "valid subtask"
    assert len(reader._range_bounds) == 1


@pytest.mark.parametrize("metadata", [None, "", '{"episode_index": 0, "action_config": null}\n'])
def test_missing_subtasks_match_episode_mode(monkeypatch, tmp_path, metadata):
    if metadata is not None:
        (tmp_path / "meta").mkdir()
        (tmp_path / "meta/episodes.jsonl").write_text(metadata)
    baseline, _ = make_reader(monkeypatch, lengths=(50, 50), root=tmp_path)
    with pytest.warns(UserWarning, match="falling back to episode mode") as warnings:
        reader, _ = make_reader(monkeypatch, lengths=(50, 50), root=tmp_path, use_subtask=True)
    assert len(warnings) == 2
    assert reader._segments == baseline._segments
    assert reader._range_captions == baseline._range_captions
    assert reader.planner.discarded_action_steps == baseline.planner.discarded_action_steps
