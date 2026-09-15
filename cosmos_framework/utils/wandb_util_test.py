# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import swanlab
import wandb

from cosmos_framework.utils import wandb_util
from cosmos_framework.utils.config import JobConfig
from cosmos_framework.utils.lazy_config.lazy import LazyConfig


def test_init_wandb_places_swanlab_beside_wandb(monkeypatch, tmp_path):
    monkeypatch.setenv("IMAGINAIRE_OUTPUT_ROOT", str(tmp_path))
    job = JobConfig(project="project", group="group", name="run", wandb_mode="offline")
    config = SimpleNamespace(
        job=job,
        checkpoint=SimpleNamespace(load_from_object_store=SimpleNamespace(enabled=False)),
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n")
    sync_calls = []

    monkeypatch.setattr(wandb_util, "_read_wandb_id", lambda config_job, config_checkpoint: "run-id")
    monkeypatch.setattr(LazyConfig, "save_yaml", lambda config, path: str(config_path))
    monkeypatch.setattr(swanlab, "sync_wandb", lambda **kwargs: sync_calls.append(kwargs))
    monkeypatch.setattr(wandb, "init", lambda **kwargs: None)
    monkeypatch.setattr(wandb, "run", None)

    wandb_util.init_wandb(config, model=object())

    assert sync_calls == [
        {
            "mode": "offline",
            "wandb_run": False,
            "log_dir": str(tmp_path / "project" / "group" / "run" / "swanlab"),
        }
    ]
