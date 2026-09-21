# SPDX-License-Identifier: OpenMDW-1.1
"""Offline three-task causal sampling; shared by CLI and online callbacks."""

import torch

from cosmos_framework.data.generator.sequence_packing.causal_action import action_frame_ids, expand_action_sequence
from cosmos_framework.data.generator.sequence_packing.teacher_forcing import TeacherForcingGeometry
from cosmos_framework.model.generator.diffusion.samplers.fixed_step import FixedStepSampler
from cosmos_framework.model.generator.mot.causal_action_cache import action_attention_cache


@torch.no_grad()
def sample_prepared(
    model,
    *,
    net,
    sampler,
    guidance,
    guidance_interval,
    velocity_postprocess_builder,
    num_steps,
    shift,
    sigma_max,
    skip_text_tokens_for_cfg,
    normalize_cfg,
    sequence_plans,
    gen_data_clean,
    cond_tokens,
    uncond_tokens,
    initial_noise,
    condition_reference,
    condition_mask,
    has_noisy_actions,
    causal_num_blocks,
    causal_block_size,
    causal_history_blocks,
    causal_use_kv_cache=True,
    **unused,
):
    if gen_data_clean.batch_size != 1 or len(sequence_plans) != 1:
        raise ValueError("Causal action inference currently requires batch size 1")
    if velocity_postprocess_builder is not None:
        raise ValueError("Causal action v1 does not support velocity postprocessing")
    if model.config.teacher_forcing_dense_mode != "grouped_tnd" and causal_use_kv_cache:
        raise ValueError("Action KV cache requires grouped_tnd")
    if causal_block_size < 1 or causal_history_blocks < 1:
        raise ValueError("Block/history must be positive")
    data = gen_data_clean
    vshape = data.x0_tokens_vision[0].shape
    ashape = data.x0_tokens_action[0].shape
    t = vshape[2]
    nv = data.x0_tokens_vision[0].numel()
    if causal_num_blocks is not None and causal_num_blocks != (t + causal_block_size - 1) // causal_block_size:
        raise ValueError("causal_num_blocks must cover the prepared latent window")
    selected = sampler or model.sampler
    if (
        not isinstance(selected, FixedStepSampler)
        and model.config.rectified_flow_inference_config.scheduler_type != "unipc"
    ):
        raise ValueError("Causal action v1 supports UniPC and FixedStepSampler")
    initial = initial_noise[0].float()
    reference = condition_reference[0].float()
    cm = condition_mask[0].expand_as(initial).float()
    state = initial * (1 - cm) + reference * cm
    # Unknown clean targets are explicitly zeroed; only completed generations enter history.
    history_v = data.x0_tokens_vision[0].float().clone()
    history_a = data.x0_tokens_action[0].float().clone()
    geometry = TeacherForcingGeometry((causal_block_size,), (causal_history_blocks,))
    templates = {}
    branches = [("cond", cond_tokens, False)]
    if guidance != 1.0:
        branches.append(("uncond", uncond_tokens, skip_text_tokens_for_cfg))
    for name, text, skip in branches:
        packed = model._pack_input_sequence(sequence_plans, text, data, torch.zeros(1, 1), skip_text_tokens=skip)
        history_v.mul_(packed.vision.condition_mask[0].to(history_v.device))
        history_a.mul_(packed.action.condition_mask[0].to(history_a.device))
        templates[name] = expand_action_sequence(
            packed, [history_v.to(model.precision)], [history_a.to(model.precision)], geometry
        )
        templates[name].to_cuda()
    af = action_frame_ids(ashape[0], t).to(state.device)
    target_net = net or model.net
    with action_attention_cache(target_net, causal_use_kv_cache) as cache:
        for b, start in enumerate(range(0, t, causal_block_size)):
            end = min(t, start + causal_block_size)
            vm = torch.zeros(vshape, device=state.device, dtype=torch.bool)
            vm[:, :, start:end] = True
            am = ((af >= start) & (af < end))[:, None].expand(ashape)
            active = torch.cat([vm.flatten(), am.flatten()]) if has_noisy_actions else vm.flatten()
            active = active & (~cm.bool())
            if not bool(active.any()):
                history_v[:, :, start:end] = state[:nv].reshape(vshape)[:, :, start:end]
                continue
            cache["entries"].clear()  # New history/current conditions, invalidate every layer and CFG branch.
            for template in templates.values():
                template.teacher_forcing.clean_vision_tokens = [history_v.to(model.precision)]
                template.teacher_forcing.clean_action_tokens = [history_a.to(model.precision)]
            frozen = state.clone()

            def velocity_fn(noise_x, timestep):
                x = torch.where(active, noise_x[0], frozen)
                vision = x[:nv].reshape(vshape)
                action = x[nv:].reshape(ashape) if has_noisy_actions else data.x0_tokens_action[0]

                def branch(name):
                    template = templates[name]
                    model._update_inference_pack_template(template, [vision], [action], None, timestep)
                    cache["key"] = (b, name)
                    out = model.denoise(net=net, data_batch_packed=template)
                    pred = [out["preds_vision"][0].reshape(-1)]
                    if has_noisy_actions:
                        va = out["preds_action"][0].clone()
                        if data.raw_action_dim is not None:
                            va[:, int(data.raw_action_dim[0]) :] = 0
                        pred.append(va.reshape(-1))
                    return torch.cat(pred) * active

                c = branch("cond")
                use_cfg = guidance != 1.0
                if guidance_interval is not None:
                    use_cfg = use_cfg and guidance_interval[0] < float(timestep.flatten()[0]) < guidance_interval[1]
                if not use_cfg:
                    return [c]
                u = branch("uncond")
                guided = u + guidance * (c - u)
                if normalize_cfg:
                    guided *= (torch.norm(c) / (torch.norm(guided) + 1e-8)).clamp(0, 1)
                return [guided]

            options = dict(num_steps=num_steps, shift=shift, seed=[None])
            if isinstance(selected, FixedStepSampler):
                options.update(condition_reference=[frozen], condition_mask=[(~active).float()])
            sampled = selected(velocity_fn, [state], **options)[0]
            state = torch.where(active, sampled, frozen)
            history_v[:, :, start:end] = state[:nv].reshape(vshape)[:, :, start:end]
            if has_noisy_actions:
                history_a[am[:, 0]] = state[nv:].reshape(ashape)[am[:, 0]]
    # Keep padded/state slots for the standard ActionProcessor; CLI trims them explicitly.
    return {"vision": [history_v], "action": [history_a]}
