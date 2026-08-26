# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
from cosmos_framework.model.generator.mot.parallelize_unified_mot import apply_fsdp
from cosmos_framework.model.generator.mot.parallelize_vfm_network import (
    build_vfm_fsdp_mixed_precision_policy,
    parallelize_vfm_network,
    resolve_vfm_parameter_storage_dtype,
)


def test_vfm_fsdp_mixed_precision_is_opt_in() -> None:
    config = ParallelismConfig(fsdp_mixed_precision_enabled=False, fsdp_master_dtype="float32")

    assert build_vfm_fsdp_mixed_precision_policy(config, "bfloat16", cast_forward_inputs=False) is None
    assert resolve_vfm_parameter_storage_dtype(torch.bfloat16, config, fsdp_enabled=True) is torch.bfloat16


def test_vfm_fsdp_mixed_precision_uses_bf16_compute_and_fp32_storage() -> None:
    config = ParallelismConfig(fsdp_mixed_precision_enabled=True, fsdp_master_dtype="float32")

    policy = build_vfm_fsdp_mixed_precision_policy(config, "bfloat16", cast_forward_inputs=False)

    assert policy is not None
    assert policy.param_dtype is torch.bfloat16
    assert policy.reduce_dtype is torch.float32
    assert policy.cast_forward_inputs is False
    assert resolve_vfm_parameter_storage_dtype(torch.bfloat16, config, fsdp_enabled=True) is torch.float32
    # The switch must not silently turn a non-FSDP run into full-fp32 training.
    assert resolve_vfm_parameter_storage_dtype(torch.bfloat16, config, fsdp_enabled=False) is torch.bfloat16


def test_parallelize_vfm_preserves_root_inputs_and_casts_nested_block_inputs() -> None:
    config = ParallelismConfig(fsdp_mixed_precision_enabled=True, fsdp_master_dtype="float32")
    model = SimpleNamespace(language_model=object())
    parallel_dims = SimpleNamespace(cp_enabled=False, dp_enabled=True, dp_mesh=object())
    compile_config = SimpleNamespace(enabled=False, compiled_region="language")

    with (
        patch(
            "cosmos_framework.model.generator.mot.parallelize_vfm_network.parallelize_unified_mot",
            side_effect=lambda language_model, **_: language_model,
        ) as parallelize_mot,
        patch(
            "cosmos_framework.model.generator.mot.parallelize_vfm_network.fully_shard",
            side_effect=lambda **kwargs: kwargs["module"],
        ) as fully_shard,
        patch("cosmos_framework.model.generator.mot.parallelize_vfm_network.register_fsdp_forward_method"),
    ):
        result = parallelize_vfm_network(
            model,
            parallel_dims=parallel_dims,
            compile_config=compile_config,
            ac_config=SimpleNamespace(),
            parallelism_config=config,
            precision="bfloat16",
        )

    nested_policy = parallelize_mot.call_args.kwargs["fsdp_mixed_precision_policy"]
    root_policy = fully_shard.call_args.kwargs["mp_policy"]
    assert result is model
    assert nested_policy is not root_policy
    assert root_policy.param_dtype is torch.bfloat16
    assert root_policy.reduce_dtype is torch.float32
    assert root_policy.cast_forward_inputs is False
    assert nested_policy.param_dtype is torch.bfloat16
    assert nested_policy.reduce_dtype is torch.float32
    assert nested_policy.cast_forward_inputs is True


class _TinyMoT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])


def test_nested_mot_fsdp_units_receive_mixed_precision_policy() -> None:
    config = ParallelismConfig(fsdp_mixed_precision_enabled=True, fsdp_master_dtype="float32")
    policy = build_vfm_fsdp_mixed_precision_policy(config, "bfloat16", cast_forward_inputs=True)
    model = _TinyMoT()
    parallel_dims = SimpleNamespace(dp_mesh=object())

    with (
        patch("cosmos_framework.model.generator.mot.parallelize_unified_mot.fully_shard") as fully_shard,
        patch("cosmos_framework.model.generator.mot.parallelize_unified_mot.register_fsdp_forward_method"),
    ):
        apply_fsdp(model, parallel_dims, mp_policy=policy)

    assert fully_shard.call_count == 2
    assert all(call.kwargs["mp_policy"] is policy for call in fully_shard.call_args_list)


def test_nested_mot_fsdp_units_omit_disabled_mixed_precision_policy() -> None:
    model = _TinyMoT()
    parallel_dims = SimpleNamespace(dp_mesh=object())

    with (
        patch("cosmos_framework.model.generator.mot.parallelize_unified_mot.fully_shard") as fully_shard,
        patch("cosmos_framework.model.generator.mot.parallelize_unified_mot.register_fsdp_forward_method"),
    ):
        apply_fsdp(model, parallel_dims, mp_policy=None)

    assert fully_shard.call_count == 2
    assert all("mp_policy" not in call.kwargs for call in fully_shard.call_args_list)


def test_parallelize_vfm_omits_disabled_root_mixed_precision_policy() -> None:
    config = ParallelismConfig(fsdp_mixed_precision_enabled=False, fsdp_master_dtype="float32")
    model = SimpleNamespace(language_model=object())
    parallel_dims = SimpleNamespace(cp_enabled=False, dp_enabled=True, dp_mesh=object())
    compile_config = SimpleNamespace(enabled=False, compiled_region="language")

    with (
        patch(
            "cosmos_framework.model.generator.mot.parallelize_vfm_network.parallelize_unified_mot",
            side_effect=lambda language_model, **_: language_model,
        ),
        patch(
            "cosmos_framework.model.generator.mot.parallelize_vfm_network.fully_shard",
            side_effect=lambda **kwargs: kwargs["module"],
        ) as fully_shard,
        patch("cosmos_framework.model.generator.mot.parallelize_vfm_network.register_fsdp_forward_method"),
    ):
        result = parallelize_vfm_network(
            model,
            parallel_dims=parallel_dims,
            compile_config=compile_config,
            ac_config=SimpleNamespace(),
            parallelism_config=config,
            precision="bfloat16",
        )

    assert result is model
    assert "mp_policy" not in fully_shard.call_args.kwargs
