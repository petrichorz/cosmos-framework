# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Exercise the production causal loop and two-layer MoT on small CPU tensors."""

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing import PackedSequence, SequencePlan
from cosmos_framework.data.generator.sequence_packing.modality import ModalityData, ModalitySpan
from cosmos_framework.model.attention.frontend import attention
from cosmos_framework.model.generator.causal_teacher_forcing_test import _config
from cosmos_framework.model.generator.diffusion.samplers.fixed_step import FixedStepSampler
from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig
from cosmos_framework.model.generator.mot.unified_mot import Qwen3VLMoTConfig, Qwen3VLTextForCausalLM
from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean


class _Harness:
    _validate_causal_inference_request = OmniMoTCausalModel._validate_causal_inference_request
    _make_causal_current_block_template = OmniMoTCausalModel._make_causal_current_block_template
    _run_causal_prefill = OmniMoTCausalModel._run_causal_prefill
    _generate_causal_inference_from_prepared = OmniMoTCausalModel._generate_causal_inference_from_prepared
    _update_inference_pack_template = OmniMoTCausalModel._update_inference_pack_template
    _copy_timestep_to_template = OmniMoTCausalModel._copy_timestep_to_template

    def __init__(self):
        vlm = Qwen3VLMoTConfig(
            {
                "text_config": dict(
                    vocab_size=256,
                    hidden_size=64,
                    intermediate_size=128,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    head_dim=16,
                    rope_scaling=dict(mrope_interleaved=True, mrope_section=[2, 3, 3], rope_type="default"),
                )
            },
            qk_norm_for_text=False,
        )
        vlm.use_und_k_norm_for_gen = True
        config = Cosmos3VFMNetworkConfig(
            vlm_config=vlm,
            latent_patch_size=1,
            latent_channel_size=2,
            max_latent_h=1,
            max_latent_w=1,
            max_latent_t=32,
            joint_attn_implementation="teacher_forcing",
        )
        self.net = Cosmos3VFMNetwork(Qwen3VLTextForCausalLM(vlm), config).float().eval()
        # Initialize both MoT pathways explicitly: real runs copy/load GEN weights.
        with torch.no_grad():
            for name, parameter in self.net.named_parameters():
                if parameter.ndim >= 2:
                    torch.nn.init.normal_(parameter, std=0.08)
                elif "norm" in name and name.endswith("weight"):
                    parameter.fill_(1.0)
                else:
                    parameter.zero_()
        self.config = _config(
            teacher_forcing_block_size_min=1,
            teacher_forcing_block_size_max=4,
            teacher_forcing_history_blocks_min=1,
            teacher_forcing_history_blocks_max=16,
            teacher_forcing_dense_mode="global",
            action_gen=False,
            sound_gen=False,
        )
        self.parallel_dims = None
        self.precision = torch.float32
        self.tensor_kwargs = dict(device="cpu", dtype=torch.float32)
        self.input_caption_key = "ai_caption"
        self.tokenizer_vision_gen = SimpleNamespace(is_causal=True, get_pixel_num_frames=lambda t: 4 * t - 3)
        self.sampler = FixedStepSampler([1.0])
        self.calls = []

    def _derive_include_end_of_generation_token(self):
        return False

    def _pack_input_sequence(self, plans, text_tokens, data, timesteps, **kwargs):
        latent = data.x0_tokens_vision[0]
        count = latent.shape[2]
        conditions = plans[0].condition_frame_indexes_vision
        noisy = torch.tensor([i for i in range(count) if i not in conditions], dtype=torch.long)
        mask = torch.zeros(count, 1, 1)
        mask[conditions] = 1
        und = len(text_tokens[0])
        vision = ModalityData(
            sequence_indexes=torch.arange(und, und + count),
            timesteps=torch.zeros(len(noisy)),
            mse_loss_indexes=noisy + und,
            spans=[ModalitySpan(und + i, 1, 0, i, 1, (1, 1, 1)) for i in range(count)],
            token_shapes=[(count, 1, 1)],
            tokens=[latent],
            condition_mask=[mask],
            noisy_frame_indexes=[noisy],
        )
        return PackedSequence(
            sample_lens=[und + count],
            split_lens=[und, count],
            attn_modes=["causal", "full"],
            sequence_length=und + count,
            is_image_batch=False,
            uses_single_timestep=True,
            text_ids=torch.tensor(text_tokens[0]),
            text_indexes=torch.arange(und),
            position_ids=torch.arange(und + count).repeat(3, 1),
            vision=vision,
        )

    def denoise(self, *, net, data_batch_packed, memory=None):
        result = self.net(data_batch_packed, memory=memory)
        self.calls.append((data_batch_packed, result["preds_vision"][0].clone()))
        return result


@pytest.mark.parametrize("block,history,frames", [(1, 1, 5), (2, 1, 7), (3, 2, 11), (4, 1, 15)])
@pytest.mark.parametrize("conditions", [0, 1, 2])
@pytest.mark.parametrize("guidance", [1.0, 3.0])
@pytest.mark.parametrize("sampler_name", ["fixed", "unipc"])
@torch.no_grad()
def test_uniform_cached_and_recomputed_latents_agree(
    monkeypatch, block, history, frames, conditions, guidance, sampler_name
):
    def cpu_attention(*args, backend=None, **kwargs):
        return attention(*args, backend="sdpa", **kwargs)

    monkeypatch.setattr("cosmos_framework.model.generator.mot.attention.attention", cpu_attention)
    monkeypatch.setattr("cosmos_framework.model.generator.mot.gen_kv_cache.attention", cpu_attention)
    monkeypatch.setattr(PackedSequence, "to_cuda", lambda self: self)
    torch.manual_seed(321)
    model = _Harness()
    if sampler_name == "unipc":
        model.config.rectified_flow_inference_config = SimpleNamespace(scheduler_type="unipc")
        model.sampler = UniPCSampler(tensor_kwargs=model.tensor_kwargs)
    else:
        model.sampler = FixedStepSampler([1.0, 0.5])
    reference = torch.randn(1, 2, frames, 1, 1)
    noise = torch.randn_like(reference)
    mask = torch.zeros_like(reference)
    mask[:, :, :conditions] = 1
    request = dict(
        data_batch={"ai_caption": ["test"]},
        net=None,
        sampler=None,
        guidance=guidance,
        guidance_interval=None,
        velocity_postprocess_builder=None,
        seed=[1],
        n_sample=1,
        has_negative_prompt=False,
        num_steps=3,
        shift=1.0,
        sigma_max=1.0,
        skip_text_tokens_for_cfg=False,
        normalize_cfg=False,
        sequence_plans=[
            SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=list(range(conditions)))
        ],
        gen_data_clean=GenerationDataClean(batch_size=1, is_image_batch=False, x0_tokens_vision=[reference]),
        cond_tokens=[[11, 12]],
        uncond_tokens=[[13, 14]],
        initial_noise=[noise.flatten()],
        condition_reference=[reference.flatten()],
        condition_mask=[mask.flatten()],
        has_noisy_actions=False,
        causal_num_blocks=(frames + block - 1) // block,
        causal_block_size=block,
        causal_history_blocks=history,
    )
    torch.manual_seed(99)
    cached = model._generate_causal_inference_from_prepared(**request, causal_use_kv_cache=True)["vision"][0]
    torch.manual_seed(99)
    recomputed = model._generate_causal_inference_from_prepared(**request, causal_use_kv_cache=False)["vision"][0]
    torch.testing.assert_close(cached, recomputed, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(cached[..., :conditions, :, :], reference[..., :conditions, :, :])
    assert cached.shape == reference.shape
    assert (cached[..., conditions:, :, :] - noise[..., conditions:, :, :]).abs().max() > 1e-4
