# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Uniform-block conditioning contracts, including boundary and loss checks."""

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing.teacher_forcing import (
    TeacherForcingGeometry,
    build_dense_teacher_forcing_gen_mask,
    build_teacher_forcing_frame_block_ids,
    build_teacher_forcing_layout,
)
from cosmos_framework.model.generator.algorithm.loss.flow_matching import compute_flow_matching_loss
from cosmos_framework.model.generator.causal_inference import causal_total_latent_frames
from cosmos_framework.model.generator.causal_teacher_forcing import validate_teacher_forcing_conditioning
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork
from cosmos_framework.model.generator.mot.teacher_forcing_network_test import (
    _make_expanded_sequence,
    _VisionEncoderHarness,
)


@pytest.mark.parametrize("block", [1, 2, 3, 4])
@pytest.mark.parametrize("frames", [1, 2, 5, 9])
def test_uniform_blocks_include_frame_zero(block, frames):
    ids = build_teacher_forcing_frame_block_ids(frames, block)
    assert ids.tolist() == [i // block for i in range(frames)]
    assert causal_total_latent_frames(3, block) == 3 * block


@pytest.mark.parametrize("history", [1, 2, 16])
def test_uniform_first_noisy_block_cannot_see_clean_answers(history):
    layout = build_teacher_forcing_layout(
        und_token_counts=[2],
        vision_token_shapes=[(7, 1, 1)],
        geometry=TeacherForcingGeometry((3,), (history,)),
    )
    mask = build_dense_teacher_forcing_gen_mask(layout)
    # Rows are clean then noisy; columns UND + clean + noisy. True means visible.
    assert mask[7:10, :2].all()
    assert not mask[7:10, 2:9].any()
    assert mask[7:10, 9:12].all()
    assert not mask[7:10, 12:].any()


@pytest.mark.parametrize("conditions", [[], [0], [0, 1]])
def test_contiguous_conditions_allowed(conditions):
    validate_teacher_forcing_conditioning([conditions])


@pytest.mark.parametrize("attention_mode, first_embedding", [("teacher_forcing", 15.0), ("two_way", 10.0)])
def test_condition_embedding_is_zero_time_not_missing_time(attention_mode, first_embedding):
    packed = _make_expanded_sequence()
    packed.vision.condition_mask[0][0] = 1
    packed.vision.noisy_frame_indexes = [torch.tensor([1])]
    packed.vision.timesteps = torch.tensor([0.5])
    packed.vision.mse_loss_indexes = packed.vision.mse_loss_indexes[1:]
    harness = _VisionEncoderHarness()
    harness.config = SimpleNamespace(joint_attn_implementation=attention_mode)
    result = torch.zeros(packed.sequence_length, 1)
    Cosmos3VFMNetwork._encode_vision(harness, packed, result, torch.float32)
    torch.testing.assert_close(
        result[packed.teacher_forcing.layout.noisy_output_indexes].flatten(), torch.tensor([first_embedding, 25.5])
    )


@pytest.mark.parametrize("condition_count", [0, 1, 2])
def test_loss_ignores_conditions_and_normalizes_active_targets(condition_count):
    pred = torch.ones(1, 3, 1, 1, requires_grad=True)
    mask = torch.zeros(3, 1, 1)
    mask[:condition_count] = 1
    rf = SimpleNamespace(train_time_weight=lambda t, kwargs: torch.ones_like(t))
    loss, _ = compute_flow_matching_loss(
        [pred],
        [torch.zeros_like(pred)],
        [mask],
        torch.ones(1, 3),
        True,
        rf,
        {"dtype": torch.float32, "device": "cpu"},
        normalize_by_active=True,
    )
    torch.testing.assert_close(loss, torch.tensor(1.0))
    loss.backward()
    assert torch.count_nonzero(pred.grad[:, :condition_count]) == 0
    assert (pred.grad[:, condition_count:] > 0).all()


@pytest.mark.parametrize("block,history", [(1, 1), (2, 2), (3, 1), (4, 16)])
@pytest.mark.parametrize("budget", [8, 32, 131072])
def test_uniform_tnd_output_and_gradients_match_dense(block, history, budget):
    from cosmos_framework.model.generator.mot.teacher_forcing_attention import teacher_forcing_dense_attention
    from cosmos_framework.model.generator.mot.teacher_forcing_tnd import build_tnd_plan, teacher_forcing_tnd_attention

    torch.manual_seed(71)
    layout = build_teacher_forcing_layout(
        und_token_counts=[2, 3],
        vision_token_shapes=[(9, 1, 2), (5, 1, 1)],
        geometry=TeacherForcingGeometry((block, block), (history, history)),
    )
    mask = build_dense_teacher_forcing_gen_mask(layout)
    plan = build_tnd_plan(layout, device="cpu", max_kv_tokens=budget)
    inputs = [torch.randn(n, h, 8, dtype=torch.float64, requires_grad=True) for n, h in [(46, 4), (51, 2), (51, 2)]]
    reference = [x.detach().clone().requires_grad_() for x in inputs]
    actual = teacher_forcing_tnd_attention(*inputs, plan)
    expected = teacher_forcing_dense_attention(*reference, ~mask)
    weight = torch.randn_like(actual)
    gradients = torch.autograd.grad((actual * weight).sum(), inputs)
    reference_gradients = torch.autograd.grad((expected * weight).sum(), reference)
    for a, b in zip([actual, *gradients], [expected, *reference_gradients], strict=True):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)
