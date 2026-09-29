# SPDX-License-Identifier: OpenMDW-1.1
"""Offline three-task causal sampling; shared by CLI and online callbacks."""

import torch

from cosmos_framework.data.generator.action.block_state import prediction_block_span
from cosmos_framework.data.generator.sequence_packing.causal_action import CausalActionGeometry, expand_action_sequence
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
    data_batch=None,
    preview_history_vision=None,
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
    if causal_num_blocks is not None and causal_num_blocks != (t - 1 + causal_block_size - 1) // causal_block_size:
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
    # Accumulate action outputs only; this buffer is never a history condition.
    history_a = data.x0_tokens_action[0].float().clone()
    metadata = sequence_plans[0].causal_action_metadata
    if metadata is None:
        raise ValueError("Measured block states are required; s0 cannot be copied into future blocks")
    if metadata.block_size != causal_block_size:
        raise ValueError("Inference block size must match delta statistics and pre-packing geometry")
    options_batch = data_batch or {}
    preview = options_batch.get("causal_action_preview", False)
    current_block = int(options_batch.get("causal_action_current_block", 0))
    if not 0 <= current_block < len(metadata.states):
        raise ValueError("Requested block has no measured state")
    geometry = CausalActionGeometry((causal_block_size,), (causal_history_blocks,))
    valid_action = metadata.action_mask.to(state.device)
    if has_noisy_actions:
        state[nv:] = state[nv:].reshape(ashape).masked_fill(~valid_action, 0).flatten()
    generated_action_mask = torch.zeros(ashape[0], dtype=torch.bool, device=state.device)
    templates = {}
    branches = [("cond", cond_tokens, False)]
    if guidance != 1.0:
        branches.append(("uncond", uncond_tokens, skip_text_tokens_for_cfg))
    for name, text, skip in branches:
        packed = model._pack_input_sequence(sequence_plans, text, data, torch.zeros(1, 1), skip_text_tokens=skip)
        if not preview:
            if not metadata.state_mask[current_block].any():
                raise ValueError("The current block needs its measured state")
            vmask = packed.vision.condition_mask[0].reshape(t, -1).all(-1)
            start, _ = prediction_block_span(current_block, causal_block_size)
            past_v = torch.arange(t, device=vmask.device) < start
            if not vmask[past_v].all():
                raise ValueError("History must be explicitly supplied as observed video")
        history_v.mul_(packed.vision.condition_mask[0].to(history_v.device))
        history_a.mul_(packed.action.condition_mask[0].to(history_a.device))
        templates[name] = expand_action_sequence(
            packed, [history_v.to(model.precision)], geometry
        )
        templates[name].to_cuda()
    af = metadata.action_frame_ids.to(state.device)
    target_net = net or model.net
    with action_attention_cache(target_net, causal_use_kv_cache) as cache:
        for b, start in enumerate(range(1, t, causal_block_size)):
            if not preview and b != current_block:
                continue
            if not metadata.state_mask[b].any():
                raise ValueError(f"Missing measured state at block {b}; stopped at the block boundary")
            end = min(t, start + causal_block_size)
            vm = torch.zeros(vshape, device=state.device, dtype=torch.bool)
            vm[:, :, start:end] = True
            am = ((af >= start) & (af < end))[:, None].expand(ashape)
            active = torch.cat([vm.flatten(), am.flatten()]) if has_noisy_actions else vm.flatten()
            if has_noisy_actions:
                active[nv:] &= valid_action.flatten()
            active = active & (~cm.bool())
            generated_action_mask |= am[:, 0] & valid_action.any(-1)
            if not bool(active.any()):
                history_v[:, :, start:end] = state[:nv].reshape(vshape)[:, :, start:end]
                continue
            cache["entries"].clear()  # New history/current conditions, invalidate every layer and CFG branch.
            for template in templates.values():
                template.teacher_forcing.clean_vision_tokens = [
                    ((preview_history_vision or data.x0_tokens_vision)[0] if preview else history_v).to(model.precision)
                ]
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
                        va = va.masked_fill(~valid_action, 0)
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
    return {
        "vision": [history_v if preview else history_v[:, :, : min(t, prediction_block_span(current_block, causal_block_size)[1])]],
        "action": [history_a.masked_fill(~valid_action, 0)],
        "generated_action_mask": [generated_action_mask],
        "preview_semantics": "真值 state 条件预览" if preview else "当前块生成，等待完整执行反馈",
    }
