# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import importlib.util
import sys
from types import ModuleType

import pytest
import torch

if importlib.util.find_spec("loguru") is None:
    log_stub = ModuleType("cosmos_framework.utils.log")
    for log_method in ("critical", "debug", "exception", "info", "success", "trace", "warning"):
        setattr(log_stub, log_method, lambda *args, **kwargs: None)
    sys.modules["cosmos_framework.utils.log"] = log_stub

from cosmos_framework.data.generator.sequence_packing.modality import ModalityData, ModalitySpan
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence
from cosmos_framework.data.generator.sequence_packing.teacher_forcing import (
    expand_packed_sequence_for_teacher_forcing,
)
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import (
    Cosmos3VFMNetwork,
    Cosmos3VFMNetworkConfig,
)


class _VisionEncoderHarness:
    timestep_scale = 1.0

    @staticmethod
    def patchify_and_pack_latents(tokens, token_shapes):
        packed = torch.cat([token.flatten() for token in tokens]).unsqueeze(-1)
        original_shapes = [(shape[0], shape[1], shape[2]) for shape in token_shapes]
        return packed, original_shapes

    @staticmethod
    def vae2llm(tokens):
        return tokens

    @staticmethod
    def _embed_packed_timesteps(timesteps, packed_seq):
        del packed_seq
        return timesteps.unsqueeze(-1) + 5


def _make_expanded_sequence() -> PackedSequence:
    vision = ModalityData(
        sequence_indexes=torch.tensor([1, 2]),
        timesteps=torch.tensor([0.5, 0.5]),
        mse_loss_indexes=torch.tensor([1, 2]),
        spans=[
            ModalitySpan(1, 1, 0, 0, 1, (1, 1, 1)),
            ModalitySpan(2, 1, 0, 1, 1, (1, 1, 1)),
        ],
        token_shapes=[(2, 1, 1)],
        tokens=[torch.tensor([[[[[10.0]], [[20.0]]]]])],
        condition_mask=[torch.zeros(2, 1, 1)],
        noisy_frame_indexes=[torch.tensor([0, 1])],
    )
    source = PackedSequence(
        sample_lens=[3],
        split_lens=[1, 2],
        attn_modes=["causal", "full"],
        is_image_batch=False,
        uses_single_timestep=True,
        sequence_length=3,
        text_ids=torch.tensor([101]),
        text_indexes=torch.tensor([0]),
        position_ids=torch.arange(9).reshape(3, 3),
        vision=vision,
    )
    return expand_packed_sequence_for_teacher_forcing(
        source,
        clean_vision_tokens=[torch.tensor([[[[[1.0]], [[2.0]]]]])],
        block_size=1,
        history_blocks=1,
    )


def test_encode_vision_fills_clean_and_noisy_streams_with_distinct_timesteps():
    packed_seq = _make_expanded_sequence()
    packed_embeddings = torch.zeros(packed_seq.sequence_length, 1)

    original_shapes = Cosmos3VFMNetwork._encode_vision(
        _VisionEncoderHarness(),
        packed_seq,
        packed_embeddings,
        torch.float32,
    )

    assert packed_seq.teacher_forcing is not None
    assert original_shapes == [(2, 1, 1)]
    torch.testing.assert_close(
        packed_embeddings[packed_seq.teacher_forcing.layout.clean_token_indexes].flatten(),
        torch.tensor([6.0, 7.0]),
    )
    torch.testing.assert_close(
        packed_embeddings[packed_seq.teacher_forcing.layout.noisy_output_indexes].flatten(),
        torch.tensor([15.5, 25.5]),
    )
    torch.testing.assert_close(packed_embeddings[packed_seq.text_indexes], torch.zeros(1, 1))


def test_network_config_keeps_dense_mask_limit_explicit():
    config = Cosmos3VFMNetworkConfig(
        teacher_forcing_max_sequence_length=123,
        teacher_forcing_visualize_sdpa_mask=True,
    )

    assert config.teacher_forcing_max_sequence_length == 123
    assert config.teacher_forcing_visualize_sdpa_mask is True
    assert Cosmos3VFMNetworkConfig().teacher_forcing_max_sequence_length is None
    assert Cosmos3VFMNetworkConfig().teacher_forcing_visualize_sdpa_mask is False

    with pytest.raises(ValueError, match="teacher_forcing_max_sequence_length"):
        Cosmos3VFMNetworkConfig(teacher_forcing_max_sequence_length=0)
