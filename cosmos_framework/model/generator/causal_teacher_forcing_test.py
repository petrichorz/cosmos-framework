# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing.modality import ModalityData
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence
from cosmos_framework.model.generator.causal_teacher_forcing import (
    expand_teacher_forcing_training_sequence,
    validate_teacher_forcing_config,
)


def _config(**overrides):
    values = dict(
        causal_training_strategy="teacher_forcing",
        vision_gen=True,
        action_gen=False,
        sound_gen=False,
        video_temporal_causal=False,
        teacher_forcing_block_size_min=2,
        teacher_forcing_block_size_max=2,
        teacher_forcing_history_blocks_min=3,
        teacher_forcing_history_blocks_max=3,
        teacher_forcing_max_sequence_length=1024,
        parallelism=SimpleNamespace(context_parallel_shard_degree=1),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _packed_noisy_video() -> PackedSequence:
    noisy = torch.tensor([[[[[10.0]], [[20.0]], [[30.0]]]]])
    return PackedSequence(
        sample_lens=[4],
        split_lens=[1, 3],
        attn_modes=["causal", "full"],
        sequence_length=4,
        text_ids=torch.tensor([101]),
        text_indexes=torch.tensor([0]),
        position_ids=torch.arange(12).reshape(3, 4),
        vision=ModalityData(
            sequence_indexes=torch.tensor([1, 2, 3]),
            timesteps=torch.tensor([0.5, 0.5, 0.5]),
            mse_loss_indexes=torch.tensor([1, 2, 3]),
            token_shapes=[(3, 1, 1)],
            tokens=[noisy],
            condition_mask=[torch.zeros(3, 1, 1)],
            noisy_frame_indexes=[torch.tensor([0, 1, 2])],
        ),
    )


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"causal_training_strategy": "none"}, "teacher_forcing"),
        ({"vision_gen": False}, "vision_gen"),
        ({"action_gen": True}, "action_gen"),
        ({"sound_gen": True}, "sound_gen"),
        ({"video_temporal_causal": True}, "video_temporal_causal"),
        ({"teacher_forcing_block_size_min": 0}, "block_size"),
        ({"teacher_forcing_block_size_min": 3, "teacher_forcing_block_size_max": 2}, "block_size"),
        ({"teacher_forcing_history_blocks_min": 0}, "history_blocks"),
        ({"teacher_forcing_history_blocks_min": 4, "teacher_forcing_history_blocks_max": 3}, "history_blocks"),
        ({"teacher_forcing_max_sequence_length": None}, "max_sequence_length"),
        ({"teacher_forcing_max_sequence_length": 0}, "max_sequence_length"),
        ({"parallelism": SimpleNamespace(context_parallel_shard_degree=2)}, "context parallel"),
    ],
)
def test_validate_teacher_forcing_config_rejects_unsupported_settings(overrides, error: str):
    with pytest.raises(ValueError, match=error):
        validate_teacher_forcing_config(_config(**overrides))


def test_expand_teacher_forcing_training_sequence_uses_configured_batch_shared_geometry():
    packed = _packed_noisy_video()
    clean = [torch.tensor([[[[[1.0]], [[2.0]], [[3.0]]]]])]

    expanded = expand_teacher_forcing_training_sequence(
        packed,
        clean_vision_tokens=clean,
        config=_config(),
    )

    assert expanded.teacher_forcing is not None
    assert expanded.teacher_forcing.layout.block_size == 2
    assert expanded.teacher_forcing.layout.history_blocks == 3
    assert expanded.teacher_forcing.clean_vision_tokens == clean
    assert expanded.vision is not None
    assert expanded.vision.tokens == packed.vision.tokens
    assert expanded.vision.tokens[0].flatten().tolist() == [10.0, 20.0, 30.0]


def test_expand_teacher_forcing_training_sequence_rejects_images():
    packed = _packed_noisy_video()
    packed.is_image_batch = True

    with pytest.raises(ValueError, match="video"):
        expand_teacher_forcing_training_sequence(
            packed,
            clean_vision_tokens=[torch.zeros(1, 1, 3, 1, 1)],
            config=_config(),
        )
