# SPDX-License-Identifier: OpenMDW-1.1
"""Regression checks for template geometry, independent masks and statistics."""

import json
from dataclasses import replace

import pytest
import torch

from cosmos_framework.data.generator.action.action_state_template import ActionStateTemplate55
from cosmos_framework.data.generator.action.block_state import BlockStatistics, Quantiles, build_block_sample
from cosmos_framework.data.generator.action.block_statistics import collect_statistics, mock_template_statistics
from cosmos_framework.data.generator.action.causal_block_geometry import CausalBlockGeometry
from cosmos_framework.data.generator.action.sample_contract import ActionReadOptions
from cosmos_framework.data.generator.action.segment_planner import SegmentPlanner


def sample(n=96, *, offset=0, steps=32, stride=4):
    template = ActionStateTemplate55()
    geometry = CausalBlockGeometry(actions_per_block=steps, video_stride=stride, temporal_compression_factor=4)
    planner = SegmentPlanner(max_action_steps=96, overlap_action_steps=16, geometry=geometry)
    state = torch.zeros(n + 1, template.width)
    state[:, 0] = torch.arange(offset, offset + n + 1)
    state[:, 28] = torch.arange(n + 1) / (n + 1)
    state[:, 30] = 0.25
    state[:, 20] = 1
    mask = torch.zeros(template.width, dtype=torch.bool)
    mask[[0, 17, 18, 19, 20, 28, 30]] = True
    raw = dict(
        state_trajectory=state,
        action_target=state[1:].clone(),
        state_mask=mask,
        action_mask=mask.clone(),
        source_contract=replace(
            template.source_contract("agibot", source="test", info={}, target_semantics="next state"), split="train"
        ),
        read_options=ActionReadOptions(action_from_state=True, action_time_offset_steps=1),
        state_timestamps=torch.arange(offset, offset + n + 1, dtype=torch.float64) / 30,
        action_timestamps=torch.arange(offset, offset + n, dtype=torch.float64) / 30,
        storage_fps=torch.tensor(30.0),
        conditioning_fps=torch.tensor(25.0),
        video=torch.arange(n + 1).view(1, n + 1, 1, 1),
    )
    return raw, template, planner


def build(raw, template, planner, **kwargs):
    return build_block_sample(raw, template=template, planner=planner, history_blocks=2, **kwargs)


def statistics(raw, planner, bound=100.0):
    d = raw["state_mask"].numel()
    return BlockStatistics(
        Quantiles(torch.full((d,), -bound), torch.full((d,), bound), raw["state_mask"]),
        Quantiles(torch.full((d,), -bound), torch.full((d,), bound), raw["action_mask"]),
    )


@pytest.mark.parametrize("offset", [0, 7, 101])
@pytest.mark.parametrize("steps,stride", [(32, 4), (16, 2), (48, 4)])
def test_geometry_anchors_and_absolute_fields(offset, steps, stride):
    raw, template, planner = sample(offset=offset, steps=steps, stride=stride)
    result, m = build(raw, template, planner)
    torch.testing.assert_close(result["action"][:, 0], torch.arange(1, steps + 1).float().repeat(96 // steps))
    torch.testing.assert_close(result["action"][:, [28, 30]], raw["action_target"][:, [28, 30]])
    torch.testing.assert_close(m.anchors[:, 0], torch.arange(offset, offset + 96, steps).float())
    assert m.state_latent_indexes.tolist() == list(range(0, 96 // (4 * stride), steps // (4 * stride)))
    assert m.action_frame_ids.min() == 1
    assert result["video"].shape[1] == 96 // stride + 1
    assert result["conditioning_fps"].item() == 25 / stride
    assert result["conditioning_fps_action"].item() == 25
    assert raw["conditioning_fps"].item() == 25
    decoded = template.decode_action_delta(
        result["action"],
        m.anchors.repeat_interleave(steps, 0),
        raw["action_mask"],
        source_contract=raw["source_contract"],
    )
    torch.testing.assert_close(decoded, raw["action_target"])


def test_rotation_is_relative_and_not_component_subtraction():
    raw, template, planner = sample()
    # anchor 为绕 z 的 90 度，目标为 180 度，delta 应为 90 度。
    q = torch.tensor([0.0, 0.0, 2**-0.5, 2**-0.5])
    raw["state_trajectory"][:, 17:21] = q
    raw["action_target"][:, 17:21] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    result, _ = build(raw, template, planner)
    torch.testing.assert_close(result["action"][:, 17:21], q.expand(96, -1))


def test_independent_masks_and_invalid_anchor():
    raw, template, planner = sample()
    raw["state_mask"][28] = False
    raw["state_trajectory"][:, 28] = float("nan")
    result, m = build(raw, template, planner)
    assert not m.state_mask[:, 28].any()
    assert m.action_mask[:, 28].all()
    torch.testing.assert_close(result["action"][:, 28], raw["action_target"][:, 28])
    raw["state_mask"][0] = False
    with pytest.raises(ValueError, match="Relative action"):
        build(raw, template, planner)


def test_statistics_contract_normalization_and_roundtrip(tmp_path):
    raw, template, planner = sample()
    stats = statistics(raw, planner)
    encoded, _ = build(raw, template, planner)
    normalized, m = build(raw, template, planner, statistics=stats)
    torch.testing.assert_close(
        stats.action.denormalize(normalized["action"], m.action_mask), encoded["action"], atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(stats.state.denormalize(m.states, m.state_mask), m.anchors, atol=1e-5, rtol=1e-5)
    path = tmp_path / "stats.json"
    stats.save(path)
    loaded = BlockStatistics.load(path)
    build(raw, template, planner, statistics=loaded)
    assert "key" not in json.loads(path.read_text())
    stats.validate(template.width, raw["state_mask"], raw["action_mask"])
    with pytest.raises(ValueError, match="width"):
        stats.validate(template.width + 1, raw["state_mask"], raw["action_mask"])
    missing = raw["action_mask"].clone()
    missing[1] = True
    with pytest.raises(ValueError, match="cover"):
        stats.validate(template.width, raw["state_mask"], missing)
    # overlap 等采样配置变化不再禁止使用人工指定的统计。
    different = SegmentPlanner(max_action_steps=96, overlap_action_steps=8, geometry=planner.geometry)
    build(raw, template, different, statistics=stats)
    short, _, _ = sample(n=32)
    build(short, template, planner, statistics=stats)  # 尾部长度不同，不需要另一份统计。
    validation = dict(raw, source_contract=replace(raw["source_contract"], split="val"))
    build(validation, template, planner, statistics=stats)


def test_mock_requires_explicit_opt_in_and_preserves_provenance(tmp_path):
    raw, template, planner = sample()
    paths = [tmp_path / "state.json", tmp_path / "delta.json"]
    for p in paths:
        p.write_text(json.dumps({}))
    mock = mock_template_statistics(
        raw, template=template, planner=planner, state_stats_path=paths[0], delta_stats_path=paths[1]
    )
    with pytest.raises(ValueError, match="allow_mock_statistics"):
        mock.validate(template.width, raw["state_mask"], raw["action_mask"])
    mock.validate(template.width, raw["state_mask"], raw["action_mask"], allow_mock=True)
    build(raw, template, planner, statistics=mock)
    path = tmp_path / "mock.json"
    mock.save(path)
    loaded = BlockStatistics.load(path)
    assert loaded.provenance == mock.provenance
    with pytest.raises(ValueError, match="allow_mock_statistics"):
        loaded.validate(template.width, raw["state_mask"], raw["action_mask"])


def test_collect_uses_same_block_encoding_and_training_split():
    raw, template, planner = sample()

    class Dataset:
        def __getitem__(self, idx):
            return raw

    ds = Dataset()
    ds.template = template
    ds.planner = planner
    stats = collect_statistics(ds, indices=[0])
    expected = torch.quantile(torch.arange(1, 33).float().repeat(3), torch.tensor([0.01, 0.99]))
    torch.testing.assert_close(torch.stack([stats.action.low[0], stats.action.high[0]]), expected)
    raw["source_contract"] = replace(raw["source_contract"], split="val")
    with pytest.raises(ValueError, match="training split"):
        collect_statistics(ds, indices=[0])


def test_rejects_incomplete_block():
    raw, template, planner = sample(n=40)
    with pytest.raises(ValueError, match="num_frames"):
        build(raw, template, planner)


def test_replaceable_template_and_nonstandard_compression():
    from cosmos_framework.data.generator.action.action_state_template import ActionStateTemplate, TemplateSourceContract

    class ScalarTemplate(ActionStateTemplate):
        template_id = "scalar-test-v1"
        width = 3

        def validate_valid_mask(self, mask, *, template_id=None):
            mask = torch.as_tensor(mask).bool()
            assert mask.shape == (self.width,)
            return mask

        def validate_source_contract(self, contract, mask):
            assert contract.template_id == self.template_id

        def sanitize(self, values, mask):
            assert values.shape[-1] == self.width
            return values.masked_fill(~mask, 0)

        def validate(self, state, action, valid_mask, source_contract):
            self.validate_source_contract(source_contract, valid_mask)

        def encode_action_delta(self, absolute_action, anchor_state, valid_mask, *, source_contract):
            return self.sanitize(absolute_action - anchor_state, valid_mask)

        def decode_action_delta(self, action_delta, anchor_state, valid_mask, *, source_contract):
            return self.sanitize(action_delta + anchor_state, valid_mask)

    raw, _, _ = sample()
    template = ScalarTemplate()
    for key in ("state_trajectory", "action_target"):
        raw[key] = raw[key][:, :3]
    for key in ("state_mask", "action_mask"):
        raw[key] = raw[key][:3]
    raw["source_contract"] = TemplateSourceContract(template.template_id, "test", "world", {}, {}, {}, "next state")
    planner = SegmentPlanner(
        max_action_steps=96,
        overlap_action_steps=16,
        geometry=CausalBlockGeometry(actions_per_block=24, video_stride=3, temporal_compression_factor=2),
    )
    stats = statistics(raw, planner)
    result, m = build(raw, template, planner, statistics=stats)
    assert result["action"].shape == (96, 3)
    assert m.state_latent_indexes.tolist() == [0, 4, 8, 12]
    assert m.block_size == 4


@pytest.mark.parametrize("mode", ["policy", "forward_dynamics", "inverse_dynamics"])
def test_sft_adapter_uses_fixed_geometry_and_template(mode):
    from cosmos_framework.data.generator.action.datasets.action_sft_dataset import ActionSFTDataset
    from cosmos_framework.data.generator.action.datasets.causal_action_sft_dataset import CausalActionSFTDataset
    from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline

    raw, template, planner = sample()
    raw["video"] = torch.zeros(3, 97, 16, 16, dtype=torch.uint8)

    class Reader:
        def __getitem__(self, idx):
            return dict(raw)

        def __len__(self):
            return 1

    reader = Reader()
    reader.template = template
    reader.planner = planner

    transform = ActionTransformPipeline(
        max_action_dim=template.width,
        video_temporal_downsample=planner.geometry.temporal_compression_factor,
        append_viewpoint_info=False,
        append_duration_fps_timestamps=False,
        append_resolution_info=False,
    )

    dataset = CausalActionSFTDataset(
        ActionSFTDataset(reader, transform, "256"), mode=mode, statistics=statistics(raw, planner)
    )
    result = dataset[0]
    plan = result["sequence_plan"]
    assert plan.causal_action_metadata.block_size == 2
    assert len(plan.condition_frame_indexes_action) == (96 if mode == "forward_dynamics" else 0)
    assert result["action"].shape[-1] == template.width
    assert not {"state_timestamps", "action_timestamps", "storage_fps"} & result.keys()
