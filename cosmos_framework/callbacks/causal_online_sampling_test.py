# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.callbacks.every_n_draw_sample import EveryNDrawSample
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel


@pytest.mark.parametrize("scenario", ["normal", "partial", "multiple", "empty", "failure"])
def test_causal_callback_uses_ti2v_without_mutating_training_batch(monkeypatch, scenario):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    video = (torch.arange(17.0) + 42).reshape(1, 1, 17, 1, 1)
    original_plan = SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[])
    batch = {"video": [video], "ai_caption": ["test"], "sequence_plan": [original_plan]}
    if scenario == "multiple":
        batch = {key: value * 2 for key, value in batch.items()}
    if scenario == "empty":
        batch["ai_caption"] = [""]
    calls = []
    model = SimpleNamespace(
        net=torch.nn.Linear(1, 1),
        config=SimpleNamespace(causal_training_strategy="teacher_forcing"),
        input_video_key="video",
        input_caption_key="ai_caption",
        tokenizer_vision_gen=SimpleNamespace(
            get_latent_num_frames=lambda frames: 1 + (frames - 1) // 4,
            get_pixel_num_frames=lambda frames: 1 + (frames - 1) * 4,
        ),
        get_data_and_condition=lambda data: SimpleNamespace(
            raw_state_vision=data["video"], x0_tokens_vision=data["video"]
        ),
        decode=lambda latent: latent,
    )

    def generate(data, **kwargs):
        calls.append((data, kwargs))
        assert data["sequence_plan"][0].condition_frame_indexes_vision == [0]
        assert not model.net.training
        assert len(data["video"]) == len(data["sequence_plan"]) == 1
        assert torch.count_nonzero(data["video"][0][..., 1:, :, :]) == 0
        torch.testing.assert_close(data["video"][0][..., :1, :, :], video[..., :1, :, :])
        if scenario == "failure":
            raise RuntimeError("test sampling failure")
        return {"vision": data["video"]}

    model.generate_samples_from_batch = generate
    callback = EveryNDrawSample(
        1,
        n_viz_sample=2 if scenario == "multiple" else 1,
        guidance=[1.0],
        causal_num_blocks=2,
        causal_block_size=3 if scenario == "partial" else 1,
    )
    monkeypatch.setattr(callback, "run_save", lambda rows, count, name: None)
    if scenario == "failure":
        with pytest.raises(RuntimeError, match="test sampling failure"):
            callback.sample(None, model, batch, None, None, 1)
    else:
        callback.sample(None, model, batch, None, None, 1)
    assert model.net.training
    assert batch["video"][0] is video and video.shape[-3] == 17
    assert original_plan.condition_frame_indexes_vision == []
    if scenario == "empty":
        assert not calls
        return
    assert calls[0][0]["video"][0].shape[-3] == (17 if scenario == "partial" else 5)
    assert calls[0][1]["causal_num_blocks"] == 2
    assert calls[0][1]["seed"] == [1]
    if scenario == "multiple":
        assert len(calls) == 2
        assert calls[1][1]["seed"] == [2]


@pytest.mark.parametrize("kwargs", [{"causal_num_blocks": 0}, {"causal_block_size": 0}, {"causal_history_blocks": 17}])
def test_causal_callback_rejects_invalid_geometry(kwargs):
    with pytest.raises(ValueError):
        EveryNDrawSample(1, **kwargs)


@torch.no_grad()
def test_ti2v_prefix_encoding_is_independent_of_future_ground_truth():
    encoded_lengths = []

    def encode(video, **kwargs):
        encoded_lengths.append(video.shape[-3])
        return video[..., ::4, :, :].clone()

    model = SimpleNamespace(
        tokenizer_vision_gen=SimpleNamespace(
            is_causal=True,
            get_latent_num_frames=lambda frames: 1 + (frames - 1) // 4,
            get_pixel_num_frames=lambda frames: 1 + (frames - 1) * 4,
        ),
        _encode_vision_item=encode,
    )
    first = torch.randn(1, 3, 17, 2, 2)
    second = first.clone()
    second[..., 1:, :, :] = 100
    outputs = [OmniMoTModel._encode_vision_x0_tokens(model, [video], None, [[0]])[0] for video in (first, second)]
    torch.testing.assert_close(outputs[0], outputs[1])
    assert encoded_lengths == [1, 1]
    assert outputs[0].shape[-3] == 5
    assert torch.count_nonzero(outputs[0][..., 1:, :, :]) == 0


@pytest.mark.parametrize(("latent_frames", "valid"), [(2, False), (3, True), (4, True), (5, False)])
def test_causal_request_accepts_only_nonempty_partial_final_block(latent_frames, valid):
    model = SimpleNamespace(
        config=SimpleNamespace(
            action_gen=False,
            sound_gen=False,
            teacher_forcing_block_size_min=1,
            teacher_forcing_block_size_max=4,
            teacher_forcing_history_blocks_min=1,
            teacher_forcing_history_blocks_max=64,
        ),
        parallel_dims=None,
        input_caption_key="ai_caption",
        tokenizer_vision_gen=SimpleNamespace(
            is_causal=True,
            get_pixel_num_frames=lambda frames: 1 + (frames - 1) * 4,
        ),
    )
    data = SimpleNamespace(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[torch.zeros(1, 3, latent_frames, 2, 2)],
        raw_state_vision=[torch.zeros(1, 3, 1 + (latent_frames - 1) * 4, 2, 2)],
    )
    request = dict(
        data_batch={"ai_caption": ["test"]},
        sequence_plans=[SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[0])],
        gen_data_clean=data,
        has_noisy_actions=False,
        has_velocity_postprocess_builder=False,
        causal_num_blocks=2,
        causal_block_size=2,
        causal_history_blocks=16,
    )
    if valid:
        assert OmniMoTCausalModel._validate_causal_inference_request(model, **request) == latent_frames
    else:
        with pytest.raises(ValueError, match="prepared latent length"):
            OmniMoTCausalModel._validate_causal_inference_request(model, **request)


@pytest.mark.parametrize("conditions", [0, 1, 2])
def test_uniform_preview_reuses_sample_and_restores_rng(monkeypatch, conditions):
    import random

    import numpy as np

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    original = torch.arange(29.0).reshape(1, 1, 29, 1, 1)
    model = SimpleNamespace(
        net=torch.nn.Linear(1, 1),
        config=SimpleNamespace(causal_training_strategy="teacher_forcing"),
        input_video_key="video",
        input_caption_key="ai_caption",
        tokenizer_vision_gen=SimpleNamespace(
            get_latent_num_frames=lambda n: (n + 3) // 4, get_pixel_num_frames=lambda n: 4 * n - 3
        ),
        get_data_and_condition=lambda batch: SimpleNamespace(
            raw_state_vision=batch["video"], x0_tokens_vision=batch["video"]
        ),
        decode=lambda x: x,
    )
    calls = []

    def generate(batch, **kwargs):
        random.random()
        np.random.rand()
        torch.rand(1)
        calls.append((batch, kwargs))
        return {"vision": batch["video"]}

    model.generate_samples_from_batch = generate
    callback = EveryNDrawSample(
        5000,
        n_viz_sample=1,
        guidance=[1.0],
        causal_num_blocks=3,
        causal_block_size=2,
        causal_condition_frames=conditions,
        causal_use_kv_cache=False,
    )
    monkeypatch.setattr(callback, "run_save", lambda *args: None)
    batch = {
        "video": [original],
        "ai_caption": ["fixed prompt"],
        "sequence_plan": [SequencePlan(has_text=True, has_vision=True)],
    }
    torch_before = torch.get_rng_state()
    random_before = random.getstate()
    numpy_before = np.random.get_state()
    callback.sample(None, model, batch, None, None, 5000)
    callback.sample(None, model, {**batch, "video": [original + 100], "ai_caption": ["changed"]}, None, None, 10000)
    assert torch.equal(torch_before, torch.get_rng_state())
    assert random_before == random.getstate()
    assert np.array_equal(numpy_before[1], np.random.get_state()[1])
    assert model.net.training
    assert len(calls) == 2
    for data, args in calls:
        assert data["ai_caption"] == ["fixed prompt"]
        assert data["sequence_plan"][0].condition_frame_indexes_vision == list(range(conditions))
        assert args["seed"] == [1] and not args["causal_use_kv_cache"]
        prefix = 4 * conditions - 3 if conditions else 0
        assert data["video"][0].shape[-3] == 21
        torch.testing.assert_close(data["video"][0][..., :prefix, :, :], original[..., :prefix, :, :])
        assert data["video"][0][..., prefix:, :, :].count_nonzero() == 0
