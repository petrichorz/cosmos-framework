# SPDX-License-Identifier: OpenMDW-1.1
"""Numerical and population checks for the standalone stride-one statistics tool."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from cosmos_framework.data.generator.action.action_state_template import ActionStateTemplate55
from cosmos_framework.data.generator.action.block_state import BlockStatistics
from cosmos_framework.data.generator.action.block_statistics import collect_statistics
from cosmos_framework.data.generator.action.causal_block_geometry import CausalBlockGeometry
from cosmos_framework.data.generator.action.segment_planner import SegmentPlanner
from tools.compute_causal_action_stats import (
    QuantileAccumulator,
    compute_statistics,
    discover_dataset_roots,
    parse_args,
)


class SyntheticBlocks:
    def __init__(self):
        self.template = ActionStateTemplate55()
        self.planner = SegmentPlanner(
            max_action_steps=32, overlap_action_steps=31, geometry=CausalBlockGeometry(temporal_compression_factor=4)
        )
        self.contract = self.template.source_contract("agibot", source="test", info={}, target_semantics="next state")
        self.episodes = []
        self.index = []
        for episode, length in enumerate((35, 33, 32)):
            state = torch.zeros(length, 55)
            state[:, 0] = torch.arange(length).float().square() + episode * 10000
            state[:, 28] = torch.arange(length).float() / 100
            # Relative rotations around Z, with alternating equivalent signs.
            angle = torch.arange(length).float() / 30
            state[:, 19] = torch.sin(angle / 2)
            state[:, 20] = torch.cos(angle / 2)
            state[::2, 17:21] *= -1
            state[:, 54] = float("nan")  # Invalid input must not pollute statistics.
            self.episodes.append(state)
            self.index.extend((episode, start, n) for start, n in self.planner.plan(length))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        episode, start, n = self.index[index]
        state = self.episodes[episode][start : start + n + 1]
        sm = torch.zeros(55, dtype=torch.bool)
        sm[0] = True
        sm[14:21] = True
        am = sm.clone()
        am[28] = True  # Absolute action can be valid without same-channel state.
        return dict(
            state_trajectory=state,
            action_target=state[1:],
            state_mask=sm,
            action_mask=am,
            source_contract=self.contract,
            conditioning_fps=30.0,
        )


def test_stride_one_population_and_training_encoding(tmp_path):
    with pytest.warns(UserWarning, match="Skipping"):
        dataset = SyntheticBlocks()
    assert dataset.index == [(0, 0, 32), (0, 1, 32), (0, 2, 32), (1, 0, 32)]
    actual = compute_statistics(dataset, method="exact")
    reference = collect_statistics(dataset, indices=range(len(dataset)))
    for kind in ("state", "action"):
        got, ref = getattr(actual, kind), getattr(reference, kind)
        torch.testing.assert_close(torch.tensor(got.metrics["q01"]), ref.low)
        torch.testing.assert_close(torch.tensor(got.metrics["q99"]), ref.high)
        torch.testing.assert_close(got.valid, ref.valid)
        assert got.low[17:21].tolist() == [-1.0] * 4
        assert got.high[17:21].tolist() == [1.0] * 4
        assert got.metrics["q01"][20] != got.low[20]
    anchors = torch.tensor([0.0, 1.0, 4.0, 10000.0])
    torch.testing.assert_close(actual.state.low[0], torch.quantile(anchors, 0.01))
    assert not actual.state.valid[28] and actual.action.valid[28]
    assert actual.action.low[28] > 0  # Absolute gripper; not target minus anchor.
    assert actual.provenance["action"]["rows"] == 128
    assert actual.provenance["state"]["rows"] == 4
    assert actual.provenance["skipped_ranges"] == 1
    path = tmp_path / "stats.json"
    actual.save(path)
    loaded = BlockStatistics.load(path)
    raw = dataset[0]
    loaded.validate(55, raw["state_mask"], raw["action_mask"])
    assert loaded.provenance == actual.provenance
    assert loaded.state.metrics == actual.state.metrics
    assert loaded.action.metrics == actual.action.metrics
    fitted = compute_statistics(dataset, reservoir_size=1000)
    torch.testing.assert_close(fitted.action.low, actual.action.low)
    assert not fitted.provenance["action"]["approximate"]


def test_partial_and_validation_split():
    with pytest.warns(UserWarning):
        dataset = SyntheticBlocks()
    result = compute_statistics(dataset, method="exact", limit=2)
    assert result.provenance["partial"] and result.provenance["blocks"] == 2
    dataset.contract = replace(dataset.contract, split="val")
    with pytest.raises(ValueError, match="training split"):
        compute_statistics(dataset)


@pytest.mark.parametrize("capacity", [1, 7, 32, 100, 120])
@pytest.mark.parametrize("batches", [1, 9, 100])
def test_batched_reservoir_matches_sequential_algorithm_r(capacity, batches):
    values = np.arange(300, dtype=np.float32).reshape(100, 3)
    reference = values[:capacity].copy()
    for d in range(3):
        rng = np.random.default_rng(np.random.SeedSequence([123, d]))
        for index in range(capacity, len(values)):
            slot = rng.integers(0, index + 1)
            if slot < capacity:
                reference[slot, d] = values[index, d]
    accumulator = QuantileAccumulator(method="reservoir", capacity=capacity, seed=123)
    for chunk in np.array_split(values, batches):
        accumulator.update(torch.from_numpy(chunk), torch.ones(chunk.shape, dtype=torch.bool))
    np.testing.assert_array_equal(accumulator.rows[: min(capacity, len(values))], reference)
    np.testing.assert_allclose(accumulator.finalize().low, np.quantile(reference, 0.01, axis=0))
    assert accumulator.summary()["approximate"] == (capacity < len(values))


def test_mask_and_empty_errors():
    accumulator = QuantileAccumulator(method="exact", capacity=5, seed=0)
    with pytest.raises(ValueError, match="empty"):
        accumulator.finalize()
    accumulator.update(torch.tensor([[1.0, float("nan")]]), torch.tensor([[True, False]]))
    assert accumulator.finalize().valid.tolist() == [True, False]
    accumulator.update(torch.ones(1, 2), torch.ones(1, 2, dtype=torch.bool))
    assert accumulator.finalize().metrics["valid_counts"] == [2, 1]
    with pytest.raises(ValueError, match="Nonfinite"):
        accumulator.update(torch.tensor([[float("nan"), 0.0]]), torch.tensor([[True, False]]))


@pytest.mark.parametrize("method", ["exact", "reservoir"])
def test_merged_statistics_use_valid_population_not_average_of_quantiles(method):
    acc = QuantileAccumulator(method=method, capacity=20, seed=7)
    # 子集大小与有效通道不同；第二通道第一批的 0 是无效占位。
    values = torch.tensor([[0.0, 0.0, 99.0], [100.0, 0.0, 99.0], [101.0, 5.0, 99.0], [102.0, 7.0, 99.0]])
    masks = torch.tensor([[1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0]], dtype=torch.bool)
    acc.update(values[:2], masks[:2])
    acc.update(values[2:], masks[2:])
    result = acc.finalize()
    assert result.metrics["count"] == [4]
    assert result.metrics["valid_counts"] == [4, 2, 0]
    for d in [0, 1]:
        x = values[masks[:, d], d].numpy().astype(np.float64)
        for key, ref in [("min", x.min()), ("max", x.max()), ("mean", x.mean()), ("std", x.std(ddof=0))]:
            assert result.metrics[key][d] == pytest.approx(ref)
        for key, q in [("q01", 0.01), ("q10", 0.10), ("q50", 0.50), ("q90", 0.90), ("q99", 0.99)]:
            assert result.metrics[key][d] == pytest.approx(np.quantile(x, q))
    assert result.metrics["q50"][0] == 100.5
    assert result.metrics["q50"][0] != (50 + 101.5) / 2
    assert result.valid.tolist() == [True, True, False]
    assert all(result.metrics[k][2] == 0 for k in ["min", "max", "mean", "std", "q01", "q99"])


def test_reservoir_moments_are_full_population_and_ignore_invalid_rows():
    acc = QuantileAccumulator(method="reservoir", capacity=7, seed=1)
    x = torch.arange(200, dtype=torch.float32).reshape(100, 2)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[::2, 1] = False
    for a, b in zip(x.split(13), mask.split(13)):
        acc.update(a, b)
    q = acc.finalize()
    for d in range(2):
        valid = x[mask[:, d], d].numpy().astype(np.float64)
        assert q.metrics["mean"][d] == pytest.approx(valid.mean())
        assert q.metrics["std"][d] == pytest.approx(valid.std())
        assert q.metrics["min"][d] == valid.min()
        assert q.metrics["max"][d] == valid.max()
        single = QuantileAccumulator(method="reservoir", capacity=7, seed=1)
        # 保持通道编号，删除无效行后有效值流应保持相同采样。
        vv = x[mask[:, d]]
        single.update(vv, torch.ones_like(vv, dtype=torch.bool))
        np.testing.assert_array_equal(acc.rows[:, d], single.rows[:, d])
    assert acc.summary()["sampled_counts"] == [7, 7]


def test_discover_parent_roots_and_deduplicate(tmp_path):
    import json

    for name in ["b/nested", "a"]:
        root = tmp_path / name
        (root / "meta").mkdir(parents=True)
        (root / "data").mkdir()
        (root / "meta/info.json").write_text(json.dumps({"codebase_version": "v3.0"}))
    expected = [tmp_path / "a", tmp_path / "b/nested"]
    assert discover_dataset_roots([tmp_path, tmp_path / "a"]) == expected
    with pytest.raises(ValueError, match="No LeRobot"):
        discover_dataset_roots([tmp_path / "a/data"])
    (tmp_path / "a/meta/info.json").write_text(json.dumps({"codebase_version": "v2.1"}))
    with pytest.raises(ValueError, match="v3"):
        discover_dataset_roots([tmp_path])


def test_merge_multiple_datasets_and_quaternion_bounds():
    with pytest.warns(UserWarning):
        a, b = SyntheticBlocks(), SyntheticBlocks()
    # state 原始四元数故意缩放，编码前应与训练一起单位化且统一符号。
    for values in b.episodes:
        values[:, 17:21] *= -3
        values[:, 0] += 500
    one = compute_statistics(a, method="exact")
    both = compute_statistics([a, b], method="exact")
    assert both.provenance["blocks"] == 8
    assert len(both.provenance["sources"]) == 2
    assert both.state.metrics["count"] == [8]
    assert both.action.metrics["count"] == [256]
    assert both.state.metrics["mean"][0] == pytest.approx(one.state.metrics["mean"][0] + 250)
    np.testing.assert_allclose(both.state.metrics["mean"][17:21], one.state.metrics["mean"][17:21], atol=1e-7)
    assert both.state.low[17:21].tolist() == [-1.0] * 4
    assert both.action.high[17:21].tolist() == [1.0] * 4


def test_bounds_selection_preserves_measured_statistics_and_quaternions():
    with pytest.warns(UserWarning):
        dataset = SyntheticBlocks()
    default = compute_statistics(dataset, reservoir_size=7)
    extrema = compute_statistics(dataset, reservoir_size=7, bounds="min_max")
    assert default.provenance["bounds"] == "q01_q99"
    assert extrema.provenance["bounds"] == "min_max"
    for kind in ("state", "action"):
        q, m = getattr(default, kind), getattr(extrema, kind)
        assert q.metrics == m.metrics
        assert m.low[0].item() == pytest.approx(m.metrics["min"][0])
        assert m.high[0].item() == pytest.approx(m.metrics["max"][0])
        assert q.low[0].item() == pytest.approx(q.metrics["q01"][0])
        assert q.high[0].item() == pytest.approx(q.metrics["q99"][0])
        assert (m.low[~m.valid] == 0).all() and (m.high[~m.valid] == 0).all()
        assert m.low[17:21].tolist() == [-1.0] * 4
        assert m.high[17:21].tolist() == [1.0] * 4
    assert extrema.state.high[0] > default.state.high[0]


@pytest.mark.parametrize("value", [None, "q01_q99", "min_max"])
def test_bounds_cli(monkeypatch, value):
    import sys

    argv = ["stats", "--dataset-root", "/unused", "--profile", "egosuite", "--output", "/unused.json"]
    if value is not None:
        argv += ["--bounds", value]
    monkeypatch.setattr(sys, "argv", argv)
    assert parse_args().bounds == (value or "q01_q99")


@pytest.mark.parametrize("method", ["exact", "reservoir"])
@pytest.mark.parametrize("offset,from_state", [(0, False), (1, True), (-1, False)])
def test_parallel_batches_preserve_episode_anchors_and_limit(monkeypatch, method, offset, from_state):
    from cosmos_framework.data.generator.action.datasets.segment_lerobot_dataset_test import make_reader

    reader, queries = make_reader(monkeypatch, lengths=(39, 38), offset=offset, from_state=from_state)
    reader.source_contract = replace(reader.source_contract, split="train")
    reader.planner = SegmentPlanner(max_action_steps=32, overlap_action_steps=31, geometry=reader.planner.geometry)
    reader._segments.clear()
    reader._episode_cum_ends.clear()
    reader._append_index_records(meta=reader._get_dataset(0).meta, ds_idx=0)
    original_segments = list(reader._segments)
    kwargs = dict(method=method, reservoir_size=7, seed=42, limit=10)
    reference = compute_statistics(reader, **kwargs)
    queries.clear()
    actual = compute_statistics(reader, batch_size=4, num_workers=3, **kwargs)
    for kind in ("state", "action"):
        got, expected = getattr(actual, kind), getattr(reference, kind)
        for key in expected.metrics:
            np.testing.assert_allclose(got.metrics[key], expected.metrics[key], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(got.low, expected.low)
        torch.testing.assert_close(got.high, expected.high)
    for key in ("sources", "blocks", "available_blocks", "partial", "state", "action"):
        assert actual.provenance[key] == reference.provenance[key]
    assert actual.provenance["blocks"] == 10
    assert actual.provenance["partial"]
    assert reader._segments == original_segments
    state_reads = [q["observation.state"] for q in queries if "observation.state" in q]
    assert len(state_reads) < 10
    assert all(rows[-1] < 39 or rows[0] >= 39 for rows in state_reads)


def test_parallel_encoding_propagates_reader_error():
    class BrokenBlocks(SyntheticBlocks):
        def __getitem__(self, index):
            if index == 2:
                raise RuntimeError("broken input window")
            return super().__getitem__(index)

    with pytest.warns(UserWarning):
        dataset = BrokenBlocks()
    with pytest.raises(RuntimeError, match="broken input window"):
        compute_statistics(dataset, num_workers=3)
