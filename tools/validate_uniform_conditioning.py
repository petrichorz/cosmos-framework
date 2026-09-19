# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Small real-NPU numerical checks; run with COSMOS_DEVICE=npu and a visible NPU."""

import json
from pathlib import Path

import torch
from torch_npu.contrib import transfer_to_npu  # noqa: F401

from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.data.generator.sequence_packing.teacher_forcing import (
    TeacherForcingGeometry,
    build_dense_teacher_forcing_gen_mask,
    build_teacher_forcing_layout,
)
from cosmos_framework.model.generator.causal_uniform_inference_test import _Harness
from cosmos_framework.model.generator.mot.teacher_forcing_attention import teacher_forcing_dense_attention
from cosmos_framework.model.generator.mot.teacher_forcing_tnd import build_tnd_plan, teacher_forcing_tnd_attention
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean


def error(actual, reference):
    a, b = actual.detach().float().cpu(), reference.detach().float().cpu()
    return dict(max_abs=(a - b).abs().max().item(), relative_l2=((a - b).norm() / b.norm().clamp_min(1e-12)).item())


def main():
    torch.manual_seed(321)
    records = []
    path = Path("outputs/uniform_conditioning/npu_numerical.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    for block, history, budget in [(1, 1, 128), (2, 2, 256), (4, 16, 131072)]:
        layout = build_teacher_forcing_layout(
            und_token_counts=[3],
            vision_token_shapes=[(9, 1, 2)],
            geometry=TeacherForcingGeometry((block,), (history,)),
        )
        plan = build_tnd_plan(layout, device="npu", max_kv_tokens=budget)
        inputs = [
            torch.randn(n, h, 64, device="npu", dtype=torch.bfloat16, requires_grad=True)
            for n, h in [(36, 4), (39, 2), (39, 2)]
        ]
        refs = [x.detach().float().cpu().requires_grad_() for x in inputs]
        output = teacher_forcing_tnd_attention(*inputs, plan)
        ref = teacher_forcing_dense_attention(*refs, ~build_dense_teacher_forcing_gen_mask(layout))
        weight = torch.randn_like(output)
        grads = torch.autograd.grad((output * weight).sum(), inputs)
        ref_grads = torch.autograd.grad((ref * weight.float().cpu()).sum(), refs)
        errors = [error(a, b) for a, b in zip([output, *grads], [ref, *ref_grads], strict=True)]
        records.append(dict(kind="tnd_grad", block=block, history=history, budget=budget, errors=errors))
        assert all(e["relative_l2"] < 0.02 for e in errors), records[-1]
        path.write_text(json.dumps(records, indent=2))
    with torch.no_grad():
        model = _Harness()
        frequencies = model.net.language_model.model.rotary_emb.inv_freq.clone()
        model.net = model.net.to(device="npu", dtype=torch.bfloat16)
        model.net.language_model.model.rotary_emb.inv_freq = frequencies.to("npu")
        model.net.time_embedder.float()
        model.precision = torch.bfloat16
        model.tensor_kwargs = dict(device="npu", dtype=torch.bfloat16)
        model.config.teacher_forcing_dense_mode = "grouped_tnd"
        model.net.config.teacher_forcing_dense_mode = "grouped_tnd"
        for block, conditions, guidance in [(1, 2, 1.0), (2, 0, 1.0), (2, 1, 3.0), (3, 2, 3.0)]:
            frames = block * 4 - 1 if block > 1 else 5
            ref = torch.randn(1, 2, frames, 1, 1, device="npu")
            noise = torch.randn_like(ref)
            mask = torch.zeros_like(ref)
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
                num_steps=1,
                shift=1.0,
                sigma_max=1.0,
                skip_text_tokens_for_cfg=False,
                normalize_cfg=False,
                sequence_plans=[
                    SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=list(range(conditions)))
                ],
                gen_data_clean=GenerationDataClean(batch_size=1, is_image_batch=False, x0_tokens_vision=[ref]),
                cond_tokens=[[11, 12]],
                uncond_tokens=[[13, 14]],
                initial_noise=[noise.flatten()],
                condition_reference=[ref.flatten()],
                condition_mask=[mask.flatten()],
                has_noisy_actions=False,
                causal_num_blocks=(frames + block - 1) // block,
                causal_block_size=block,
                causal_history_blocks=1,
            )
            outputs = [
                model._generate_causal_inference_from_prepared(**request, causal_use_kv_cache=cache)["vision"][0]
                for cache in [True, False]
            ]
            e = error(*outputs)
            delta = (outputs[0][..., conditions:, :, :] - noise[..., conditions:, :, :]).abs().max().item()
            assert delta > 1e-4
            records.append(
                dict(kind="cache", block=block, conditions=conditions, guidance=guidance, error=e, sampling_delta=delta)
            )
            path.write_text(json.dumps(records, indent=2))
            assert e["relative_l2"] < 0.02, records[-1]
            torch.testing.assert_close(outputs[0][..., :conditions, :, :], ref[..., :conditions, :, :], atol=0, rtol=0)
    path = Path("outputs/uniform_conditioning/npu_numerical.json")
    path.write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
