# SPDX-License-Identifier: OpenMDW-1.1
import json

import numpy as np
import pytest
import torch

from cosmos_framework.data.generator.action.block_state import BlockStatistics
from tools.aggregate_causal_action_stats import aggregate_statistics
from tools.compute_causal_action_stats import QuantileAccumulator


def write_source(path, values, masks, **overrides):
    acc = QuantileAccumulator(method="exact", capacity=10, seed=42)
    acc.update(torch.tensor(values).float(), torch.tensor(masks).bool())
    q = acc.finalize()
    provenance = dict(
        method="exact",
        template_id="test",
        geometry={},
        population="all_valid_single_block_starts",
        start_stride=1,
        split="train",
        std_ddof=0,
        rotation_groups=[],
        read_options={},
        split_seed=0,
        split_val_ratio=0,
        dataset_roots=[str(path)],
        partial=False,
        sources=[],
        blocks=len(values),
        available_blocks=len(values),
        skipped_ranges=0,
        discarded_action_steps=0,
    )
    provenance.update(overrides)
    BlockStatistics(q, q, provenance).save(path)
    return path


@pytest.mark.parametrize("bounds", ["q01_q99", "min_max"])
def test_population_moments_masks_and_envelope(tmp_path, bounds):
    paths = [
        write_source(tmp_path / "a.json", [[1, 0], [3, 0]], [[1, 0], [1, 0]]),
        write_source(tmp_path / "b.json", [[100, 5], [200, 7], [300, 9]], [[1, 1]] * 3, partial=True),
    ]
    result = aggregate_statistics(paths, bounds=bounds)
    for kind in (result.state, result.action):
        stats = kind.metrics
        for key, fn in (("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)):
            np.testing.assert_allclose(stats[key], [fn([1, 3, 100, 200, 300]), fn([5, 7, 9])])
        assert stats["valid_counts"] == [5, 3]
        assert stats["count"] == [5]
        assert "q50" not in stats
        assert stats["q01"] == pytest.approx([1.02, 5.04])
        assert stats["q99"] == pytest.approx([298, 8.96])
        np.testing.assert_allclose(kind.low, [1, 5] if bounds == "min_max" else [1.02, 5.04])
    assert result.provenance["partial"]
    result.save(tmp_path / "merged.json")
    assert BlockStatistics.load(tmp_path / "merged.json").state.metrics == result.state.metrics


@pytest.mark.parametrize("bounds", ["q01_q99", "min_max"])
def test_quaternion_bounds(tmp_path, bounds):
    path = write_source(tmp_path / "a.json", [[0, 0, 0, 1]], [[1] * 4], rotation_groups=[[0, 1, 2, 3]])
    result = aggregate_statistics([path], bounds=bounds)
    np.testing.assert_array_equal(result.state.low, [-1] * 4)
    np.testing.assert_array_equal(result.state.high, [1] * 4)
    assert result.state.metrics["min"] == [0, 0, 0, 1]


def test_reject_incompatible_or_duplicate_inputs(tmp_path):
    a = write_source(tmp_path / "a.json", [[1]], [[1]])
    b = write_source(tmp_path / "b.json", [[2]], [[1]], template_id="different")
    with pytest.raises(ValueError, match="template_id"):
        aggregate_statistics([a, b])
    with pytest.raises(ValueError, match="distinct"):
        aggregate_statistics([a, a])
    with pytest.raises(FileNotFoundError):
        aggregate_statistics([tmp_path / "missing.json"])
    data = json.loads(a.read_text())
    data["state"]["valid_counts"] = [-1]
    a.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="counts"):
        aggregate_statistics([a])


def test_child_failure_preserves_existing_statistics(tmp_path, monkeypatch):
    import argparse
    import subprocess

    from tools.compute_causal_action_stats_parallel import compute_child

    (tmp_path / "meta").mkdir()
    output = tmp_path / "meta/causal_action_stats.json"
    output.write_text("previous result")

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        compute_child(tmp_path, argparse.Namespace(profile="agibot", num_workers=2))
    assert output.read_text() == "previous result"
    assert list((tmp_path / "meta").iterdir()) == [output]


def test_200_sources_match_concatenated_population(tmp_path):
    rng = np.random.default_rng(72)
    paths, populations = [], [[] for _ in range(5)]
    for i in range(200):
        values = (rng.normal(size=(int(rng.integers(1, 40)), 5)) + i * 10).astype(np.float32)
        masks = rng.random(values.shape) > 0.4
        masks[:, -1] = False  # 所有子集均无效的通道应保持零值。
        paths.append(write_source(tmp_path / f"{i}.json", values, masks))
        for d in range(5):
            populations[d].extend(values[masks[:, d], d].astype(np.float64))
    result = aggregate_statistics(paths)
    reverse = aggregate_statistics(paths[::-1])
    for key, fn in (("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)):
        expected = [fn(v) if v else 0 for v in populations]
        np.testing.assert_allclose(result.state.metrics[key], expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(reverse.state.metrics[key], expected, rtol=1e-12, atol=1e-12)
    assert result.state.metrics["valid_counts"] == [len(v) for v in populations]
    assert not result.state.valid[-1]
    result.validate(5, result.state.valid, result.action.valid)
    normalized, _ = result.state.normalize(torch.zeros(2, 5), result.state.valid.expand(2, -1))
    assert torch.isfinite(normalized).all()
    assert (normalized[:, -1] == 0).all()


@pytest.mark.parametrize("key,value", [("count", [1.5]), ("count", [-1]), ("q01", [3]), ("q99", [-1])])
def test_reject_malformed_statistics(tmp_path, key, value):
    path = write_source(tmp_path / "a.json", [[1], [2]], [[1], [1]])
    data = json.loads(path.read_text())
    data["state"][key] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        aggregate_statistics([path])


def test_reject_complementary_invalid_quaternion_masks(tmp_path):
    a = write_source(tmp_path / "a.json", [[0, 0, 0, 1]], [[1, 1, 0, 0]], rotation_groups=[[0, 1, 2, 3]])
    b = write_source(tmp_path / "b.json", [[0, 0, 0, 1]], [[0, 0, 1, 1]], rotation_groups=[[0, 1, 2, 3]])
    with pytest.raises(ValueError, match="quaternion"):
        aggregate_statistics([a, b])


def test_child_cli_forwarding_and_saved_quantiles(tmp_path, monkeypatch):
    import argparse
    import subprocess

    from tools.compute_causal_action_stats_parallel import compute_child

    (tmp_path / "meta").mkdir()
    source = write_source(tmp_path / "source.json", [[1], [2]], [[1], [1]])

    def run(command, **kwargs):
        assert command[command.index("--num-workers") + 1] == "3"
        assert command[command.index("--action-time-offset-steps") + 1] == "0"
        assert "--action-from-state" in command
        assert "--dataset-processes" not in command
        destination = command[command.index("--output") + 1]
        from pathlib import Path

        Path(destination).write_text(source.read_text())

    monkeypatch.setattr(subprocess, "run", run)
    path = compute_child(
        tmp_path,
        argparse.Namespace(num_workers=3, action_from_state=True, action_time_offset_steps=0, dataset_processes=2),
    )
    data = json.loads(path.read_text())
    assert data["provenance"]["quantiles"] == [0.01, 0.99]
    assert set(k for k in data["state"] if k.startswith("q")) == {"q01", "q99"}
    assert len(list((tmp_path / "meta").iterdir())) == 1
