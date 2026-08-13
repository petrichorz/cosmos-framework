# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Loader tests for the free-form ``[custom]`` escape-hatch section of the SFT TOML."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from cosmos_framework.configs.toml_config.sft_config import SFTExperimentConfig
from cosmos_framework.configs.toml_config.toml_config_helper import build_hydra_overrides

# Representative payload: scalars, a nested sub-table, and an array-of-tables.
_CUSTOM_PAYLOAD = {
    "scalar_int": 5,
    "scalar_str": "hello",
    "flag": True,
    "ratio": 0.3,
    "sampling": {"bug_ratio": 0.3, "nested": {"deep": 1}},
    "items": [
        {"path": "/data/a", "weight": 1.0},
        {"path": "/data/b", "weight": 2.0},
    ],
}


# --------------------------------------------------------------------------- #
# 1. pydantic schema validation                                               #
# --------------------------------------------------------------------------- #
class TestSchemaValidation:
    def test_teacher_forcing_model_fields_validate(self) -> None:
        raw = {
            "job": {"task": "vfm", "experiment": "vision_sft_edge"},
            "model": {
                "causal_training_strategy": "teacher_forcing",
                "teacher_forcing_block_size_min": 1,
                "teacher_forcing_block_size_max": 4,
                "teacher_forcing_history_blocks_min": 1,
                "teacher_forcing_history_blocks_max": 32,
                "teacher_forcing_max_sequence_length": 123456,
                "teacher_forcing_dense_mode": "per_sample",
                "teacher_forcing_visualize_sdpa_mask": True,
            },
        }

        cfg = SFTExperimentConfig.model_validate(raw)

        assert cfg.model.causal_training_strategy == "teacher_forcing"
        assert cfg.model.teacher_forcing_max_sequence_length == 123456
        assert cfg.model.teacher_forcing_dense_mode == "per_sample"
        assert cfg.model.teacher_forcing_visualize_sdpa_mask is True

    def test_custom_section_validates_arbitrary_nested_content(self) -> None:
        """Arbitrary nested [custom] content passes through untouched."""
        raw = {
            "job": {"task": "vfm", "experiment": "vision_sft_nano"},
            "custom": _CUSTOM_PAYLOAD,
        }
        cfg = SFTExperimentConfig.model_validate(raw)
        # The framework stores it verbatim — no coercion, no inner validation.
        assert cfg.custom == _CUSTOM_PAYLOAD

    def test_no_custom_section_defaults_empty(self) -> None:
        cfg = SFTExperimentConfig.model_validate({"job": {"task": "vfm", "experiment": "vision_sft_nano"}})
        assert cfg.custom == {}

    def test_unknown_top_level_key_raises(self) -> None:
        """Any unknown top-level section that is NOT `custom` still raises."""
        with pytest.raises(ValidationError):
            SFTExperimentConfig.model_validate(
                {
                    "job": {"task": "vfm", "experiment": "vision_sft_nano"},
                    "bogus_section": {"x": 1},
                }
            )

    def test_unknown_key_inside_optimizer_raises(self) -> None:
        """A typo inside a KNOWN section is still a hard error (extra='forbid')."""
        with pytest.raises(ValidationError):
            SFTExperimentConfig.model_validate(
                {
                    "job": {"task": "vfm", "experiment": "vision_sft_nano"},
                    "optimizer": {"lr": 1.0e-4, "not_a_real_key": 1},
                }
            )

    def test_custom_does_not_loosen_sibling_validation(self) -> None:
        """Presence of [custom] must not relax extra='forbid' elsewhere."""
        with pytest.raises(ValidationError):
            SFTExperimentConfig.model_validate(
                {
                    "job": {"task": "vfm", "experiment": "vision_sft_nano"},
                    "custom": _CUSTOM_PAYLOAD,
                    "trainer": {"max_iter": 10, "typo_here": True},
                }
            )


# --------------------------------------------------------------------------- #
# 2. build_hydra_overrides must NOT emit [custom] as per-leaf overrides        #
# --------------------------------------------------------------------------- #
class TestBuildHydraOverrides:
    def test_teacher_forcing_model_fields_route_to_vfm_model_config(self) -> None:
        raw = {
            "job": {"task": "vfm", "experiment": "vision_sft_edge"},
            "model": {
                "causal_training_strategy": "teacher_forcing",
                "teacher_forcing_block_size_min": 1,
                "teacher_forcing_block_size_max": 4,
                "teacher_forcing_history_blocks_min": 1,
                "teacher_forcing_history_blocks_max": 32,
                "teacher_forcing_max_sequence_length": 123456,
                "teacher_forcing_dense_mode": "per_sample",
                "teacher_forcing_visualize_sdpa_mask": True,
            },
        }

        overrides = build_hydra_overrides(raw)

        assert "model.config.causal_training_strategy=teacher_forcing" in overrides
        assert "model.config.teacher_forcing_block_size_min=1" in overrides
        assert "model.config.teacher_forcing_block_size_max=4" in overrides
        assert "model.config.teacher_forcing_history_blocks_min=1" in overrides
        assert "model.config.teacher_forcing_history_blocks_max=32" in overrides
        assert "model.config.teacher_forcing_max_sequence_length=123456" in overrides
        assert "model.config.teacher_forcing_dense_mode=per_sample" in overrides
        assert "model.config.teacher_forcing_visualize_sdpa_mask=true" in overrides

    def test_teacher_forcing_model_fields_are_skipped_for_vlm(self) -> None:
        raw = {
            "job": {"task": "vlm", "experiment": "llava_ov"},
            "model": {
                "causal_training_strategy": "teacher_forcing",
                "teacher_forcing_block_size_min": 1,
                "teacher_forcing_block_size_max": 4,
                "teacher_forcing_history_blocks_min": 1,
                "teacher_forcing_history_blocks_max": 32,
                "teacher_forcing_max_sequence_length": 123456,
                "teacher_forcing_visualize_sdpa_mask": True,
            },
        }

        overrides = build_hydra_overrides(raw)

        assert all("teacher_forcing" not in override for override in overrides)
        assert all("causal_training_strategy" not in override for override in overrides)

    def test_custom_not_emitted_as_overrides(self) -> None:
        raw = {
            "job": {"task": "vfm", "experiment": "vision_sft_nano"},
            "optimizer": {"lr": 1.0e-5},
            "custom": _CUSTOM_PAYLOAD,
        }
        overrides = build_hydra_overrides(raw)
        # Nothing under custom (verbatim or remapped) should appear.
        assert all("custom" not in o for o in overrides), overrides

    def test_other_keys_still_emitted(self) -> None:
        raw = {
            "job": {"task": "vfm", "experiment": "vision_sft_nano"},
            "optimizer": {"lr": 1.0e-5},
            "custom": {"a": 1},
        }
        overrides = build_hydra_overrides(raw)
        assert "experiment=vision_sft_nano" in overrides
        assert any(o.startswith("optimizer.lr=") for o in overrides), overrides


# --------------------------------------------------------------------------- #
# 3. end-to-end load_experiment_from_toml on the shipped vision_sft_nano recipe #
# --------------------------------------------------------------------------- #
_BASE_TOML = """\
[job]
task         = "vfm"
experiment   = "vision_sft_nano"
project      = "cosmos3"
group        = "sft"
name         = "sft_config_custom_test"
wandb_mode   = "disabled"

[model.tokenizer]
vae_path = "${oc.env:WAN_VAE_PATH}"

[checkpoint]
load_path = "${oc.env:BASE_CHECKPOINT_PATH}"
"""

_CUSTOM_TOML_BLOCK = """\

[custom]
scalar_int = 5
scalar_str = "hello"
flag       = true
ratio      = 0.3

[custom.sampling]
bug_ratio = 0.3

[custom.sampling.nested]
deep = 1

[[custom.items]]
path   = "/data/a"
weight = 1.0

[[custom.items]]
path   = "/data/b"
weight = 2.0
"""


def _load_or_skip(toml_path: Path, extra_overrides: list[str] | None = None):
    """Run the real loader, skipping if the training stack can't be imported."""
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml

    try:
        return load_experiment_from_toml(str(toml_path), extra_overrides=extra_overrides)
    except ImportError as exc:  # pragma: no cover — env-dependent
        pytest.skip(f"training stack not importable here: {exc!r}")


@pytest.fixture
def _dummy_recipe_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # vision_sft_nano interpolates these env vars into path strings at resolve time.
    monkeypatch.setenv("DATASET_PATH", "/tmp/dummy_dataset")
    monkeypatch.setenv("WAN_VAE_PATH", "/tmp/dummy_vae.pth")
    monkeypatch.setenv("BASE_CHECKPOINT_PATH", "/tmp/dummy_ckpt")


class TestEndToEndLoader:
    def test_load_edge_causal_smoke_recipe(
        self,
        _dummy_recipe_env: None,
    ) -> None:
        from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

        recipe_path = Path(__file__).parents[3] / "examples/toml/sft_config/vision_causal_smoke_edge.toml"
        config = _load_or_skip(
            recipe_path,
            extra_overrides=[
                "model=mot_causal_ddp",
                "dataloader_train.max_sequence_length=null",
                "~dataloader_train.dataloader.datasets.video.dataset.conditioning_config={0:0.7,1:0.2,2:0.1}",
                "+dataloader_train.dataloader.datasets.video.dataset.conditioning_config={0:1.0,1:0.0,2:0.0}",
                "dataloader_train.dataloader.datasets.video.dataset.num_video_frames=17",
            ],
        )

        assert config.model._target_ is OmniMoTCausalModel
        assert config.job.name == "vision_causal_smoke_edge"
        assert config.trainer.distributed_parallelism == "ddp"
        assert config.trainer.max_iter == 3
        assert config.model.config.precision == "bfloat16"
        assert config.model.config.causal_training_strategy == "teacher_forcing"
        assert config.model.config.teacher_forcing_block_size_min == 1
        assert config.model.config.teacher_forcing_block_size_max == 4
        assert config.model.config.teacher_forcing_history_blocks_min == 1
        assert config.model.config.teacher_forcing_history_blocks_max == 32
        assert config.model.config.teacher_forcing_max_sequence_length == 4096
        assert config.model.config.teacher_forcing_dense_mode == "per_sample"
        assert config.model.config.parallelism.data_parallel_shard_degree == 1
        assert config.model.config.parallelism.context_parallel_shard_degree == 1
        assert config.model.config.compile.enabled is False
        assert config.model.config.activation_checkpointing.mode == "none"
        assert config.model.config.ema.enabled is False
        assert config.optimizer.optimizer_type == "AdamW"
        assert config.optimizer.fused is False
        assert config.dataloader_train.max_samples_per_batch == 1
        assert config.dataloader_train.max_sequence_length is None
        dataset = config.dataloader_train.dataloader.datasets.video.dataset
        assert dataset.conditioning_config == {0: 1.0, 1: 0.0, 2: 0.0}
        assert dataset.num_video_frames == 17

    def test_load_causal_model_group_with_teacher_forcing_toml(
        self,
        tmp_path: Path,
        _dummy_recipe_env: None,
    ) -> None:
        from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

        toml_path = tmp_path / "causal.toml"
        toml_path.write_text(
            """\
[job]
task       = "vfm"
experiment = "vision_sft_nano"

[model]
causal_training_strategy                 = "teacher_forcing"
teacher_forcing_block_size_min           = 1
teacher_forcing_block_size_max           = 4
teacher_forcing_history_blocks_min       = 1
teacher_forcing_history_blocks_max       = 32
teacher_forcing_max_sequence_length      = 123456

[model.tokenizer]
vae_path = "${oc.env:WAN_VAE_PATH}"

[checkpoint]
load_path = "${oc.env:BASE_CHECKPOINT_PATH}"
"""
        )

        config = _load_or_skip(toml_path, extra_overrides=["model=mot_causal_fsdp"])

        assert config.model._target_ is OmniMoTCausalModel
        assert config.model.config.causal_training_strategy == "teacher_forcing"
        assert config.model.config.teacher_forcing_block_size_min == 1
        assert config.model.config.teacher_forcing_block_size_max == 4
        assert config.model.config.teacher_forcing_history_blocks_min == 1
        assert config.model.config.teacher_forcing_history_blocks_max == 32
        assert config.model.config.teacher_forcing_max_sequence_length == 123456
        assert config.model.config.teacher_forcing_dense_mode == "global"

    def test_load_with_custom_section(self, tmp_path: Path, _dummy_recipe_env: None) -> None:
        toml_path = tmp_path / "with_custom.toml"
        toml_path.write_text(_BASE_TOML + _CUSTOM_TOML_BLOCK)

        config = _load_or_skip(toml_path)

        expected = {
            "scalar_int": 5,
            "scalar_str": "hello",
            "flag": True,
            "ratio": 0.3,
            "sampling": {"bug_ratio": 0.3, "nested": {"deep": 1}},
            "items": [
                {"path": "/data/a", "weight": 1.0},
                {"path": "/data/b", "weight": 2.0},
            ],
        }
        # Injected verbatim as a plain dict after Hydra resolution, so a project
        # can run MyProjectConfig.model_validate(config.custom) directly.
        assert config.custom == expected

    def test_load_without_custom_section_defaults_empty(self, tmp_path: Path, _dummy_recipe_env: None) -> None:
        toml_path = tmp_path / "no_custom.toml"
        toml_path.write_text(_BASE_TOML)

        config = _load_or_skip(toml_path)

        assert config.custom == {}


# --------------------------------------------------------------------------- #
# 4. Every shipped example TOML validates + builds overrides                   #
# --------------------------------------------------------------------------- #
_EXAMPLE_TOML_DIR = Path(__file__).parents[3] / "examples" / "toml" / "sft_config"
_EXAMPLE_TOMLS = sorted(_EXAMPLE_TOML_DIR.glob("*.toml"))


class TestExampleTomlConfigs:
    """Every TOML shipped under ``examples/toml/sft_config/`` must both validate
    against ``SFTExperimentConfig`` (``extra="forbid"`` rejects any block the
    schema does not register) and produce a Hydra override list. Catches, e.g., a
    recipe TOML carrying a ``[model.*]`` / ``[dataloader_train.*]`` sub-block the
    schema never modelled, which would raise ``ValidationError`` at load time.
    """

    def test_examples_dir_is_nonempty(self) -> None:
        assert _EXAMPLE_TOMLS, f"no example TOMLs found under {_EXAMPLE_TOML_DIR}"

    @pytest.mark.parametrize("toml_path", _EXAMPLE_TOMLS, ids=lambda p: p.name)
    def test_example_toml_validates_and_builds_overrides(self, toml_path: Path) -> None:
        import tomllib

        raw = tomllib.loads(toml_path.read_text())

        # 1. Schema validation — extra="forbid" flags un-registered TOML blocks.
        try:
            SFTExperimentConfig.model_validate(raw)
        except ValidationError as exc:  # pragma: no cover - failure detail
            pytest.fail(f"{toml_path.name} failed schema validation:\n{exc}")

        # 2. Override construction — every leaf must route through PATH_REMAPS.
        overrides = build_hydra_overrides(raw)
        assert overrides[0] == "--"
        assert any(o.startswith("experiment=") for o in overrides), overrides
