# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Scheme-B causal training and GenKVCache block-autoregressive inference."""

from dataclasses import replace
from typing import Any, Optional

import torch

from cosmos_framework.configs.base.defaults.model_config import OmniMoTModelConfig
from cosmos_framework.data.generator.sequence_packing import PackedSequence, SequencePlan, TeacherForcingGeometry
from cosmos_framework.model.generator.causal_inference import (
    causal_generated_block_span,
    causal_total_latent_frames,
    concat_vision_time,
    flatten_vision_5d,
    require_single_vision_5d,
    slice_vision_time,
    unflatten_vision_5d,
)
from cosmos_framework.model.generator.causal_teacher_forcing import (
    expand_teacher_forcing_training_sequence,
    prepare_teacher_forcing_geometry,
    validate_teacher_forcing_conditioning,
    validate_teacher_forcing_config,
)
from cosmos_framework.model.generator.diffusion.samplers.fixed_step import FixedStepSampler
from cosmos_framework.model.generator.mot.gen_kv_cache import (
    CachedCleanBlock,
    GenKVCache,
    GenKVCacheMemoryState,
    GenKVCacheMode,
    commit_cfg_append_pair,
    install_gen_kv_cache_attention_dispatch,
    restore_gen_kv_cache_attention_dispatch,
)
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean


class OmniMoTCausalModel(OmniMoTModel):
    """Vision-only causal model using clean/noisy streams in one forward."""

    def __init__(self, config: OmniMoTModelConfig):
        validate_teacher_forcing_config(config)
        super().__init__(config)

    def prepare_teacher_forcing_geometry(
        self,
        num_vision_latent_frames: list[int],
        condition_frame_indexes_vision: list[list[int]] | None = None,
    ) -> TeacherForcingGeometry:
        """Sample independent block/history geometry for every packed sample."""

        if condition_frame_indexes_vision is None:
            raise ValueError("causal teacher forcing requires per-sample vision conditioning metadata")
        if len(condition_frame_indexes_vision) != len(num_vision_latent_frames):
            raise ValueError(
                "vision conditioning metadata must contain one entry per sample, "
                f"got {len(condition_frame_indexes_vision)} and {len(num_vision_latent_frames)}"
            )
        validate_teacher_forcing_conditioning(condition_frame_indexes_vision)
        return prepare_teacher_forcing_geometry(
            num_samples=len(num_vision_latent_frames),
            config=self.config,
        )

    def post_noise_packing_hook(
        self,
        packed_sequence: PackedSequence,
        gen_data_clean: GenerationDataClean,
        teacher_forcing_geometry: TeacherForcingGeometry | None = None,
    ) -> PackedSequence:
        """Attach clean ``x0`` after the ordinary path has installed noisy ``xt``."""

        if gen_data_clean.x0_tokens_vision is None:
            raise ValueError("teacher-forcing causal training requires clean vision tokens")
        if teacher_forcing_geometry is None:
            teacher_forcing_geometry = prepare_teacher_forcing_geometry(
                num_samples=len(gen_data_clean.x0_tokens_vision),
                config=self.config,
            )
        clean_vision_tokens = [token.to(dtype=self.precision) for token in gen_data_clean.x0_tokens_vision]
        return expand_teacher_forcing_training_sequence(
            packed_sequence,
            clean_vision_tokens=clean_vision_tokens,
            config=self.config,
            geometry=teacher_forcing_geometry,
        )

    def _validate_causal_inference_request(
        self,
        *,
        data_batch: dict,
        sequence_plans: list[SequencePlan],
        gen_data_clean: GenerationDataClean,
        has_noisy_actions: bool,
        has_velocity_postprocess_builder: bool,
        causal_num_blocks: int | None,
        causal_block_size: int,
        causal_history_blocks: int,
    ) -> int:
        if causal_num_blocks is None:
            raise ValueError("causal inference requires causal_num_blocks")
        target_frames = causal_total_latent_frames(causal_num_blocks, causal_block_size)
        if len(sequence_plans) != 1 or gen_data_clean.batch_size != 1:
            raise ValueError("causal inference currently requires batch_size=1")
        plan = sequence_plans[0]
        if not plan.has_text or not plan.has_vision or plan.condition_frame_indexes_vision != [0]:
            raise ValueError("causal inference requires Text+Image input with condition_frame_indexes_vision=[0]")
        if plan.has_action or plan.has_sound or has_noisy_actions or self.config.action_gen or self.config.sound_gen:
            raise ValueError("causal inference does not support action or sound")
        if has_velocity_postprocess_builder:
            raise ValueError("causal inference does not support velocity postprocessing")
        if gen_data_clean.is_image_batch:
            raise ValueError("causal inference requires a video target conditioned on one image")
        if not (
            self.config.teacher_forcing_block_size_min
            <= causal_block_size
            <= self.config.teacher_forcing_block_size_max
        ):
            raise ValueError(
                "causal_block_size lies outside the checkpoint training range "
                f"[{self.config.teacher_forcing_block_size_min}, {self.config.teacher_forcing_block_size_max}]"
            )
        history_max = min(16, self.config.teacher_forcing_history_blocks_max)
        if not self.config.teacher_forcing_history_blocks_min <= causal_history_blocks <= history_max:
            raise ValueError(
                "causal_history_blocks lies outside the supported range "
                f"[{self.config.teacher_forcing_history_blocks_min}, {history_max}]"
            )
        if self.parallel_dims is not None and (
            self.parallel_dims.cp_enabled or self.parallel_dims.cfgp_enabled or self.parallel_dims.dp_shard_enabled
        ):
            raise ValueError("GenKVCache causal inference does not yet support CP, CFGP, or FSDP sharding")
        if not getattr(self.tokenizer_vision_gen, "is_causal", False):
            raise ValueError("causal inference requires a causal vision tokenizer")
        full_latent = require_single_vision_5d(gen_data_clean.x0_tokens_vision)
        if full_latent.shape[2] != target_frames:
            raise ValueError(f"prepared latent length {full_latent.shape[2]} does not match 1+B*S={target_frames}")
        expected_pixel_frames = self.tokenizer_vision_gen.get_pixel_num_frames(target_frames)
        if gen_data_clean.raw_state_vision:
            raw_vision = gen_data_clean.raw_state_vision[0]
            actual_pixel_frames = int(raw_vision.shape[raw_vision.ndim - 3])
            if actual_pixel_frames != expected_pixel_frames:
                raise ValueError(
                    "causal request pixel length must be derived from the active tokenizer: "
                    f"get_pixel_num_frames({target_frames})={expected_pixel_frames}, got {actual_pixel_frames}"
                )
        prompts = data_batch.get(self.input_caption_key)
        if (
            not isinstance(prompts, list)
            or len(prompts) != 1
            or not isinstance(prompts[0], str)
            or not prompts[0].strip()
        ):
            raise ValueError("causal inference requires exactly one non-empty text prompt")
        return target_frames

    def _make_causal_current_block_template(
        self,
        *,
        text_tokens: list[list[int]],
        skip_text_tokens: bool,
        source_plan: SequencePlan,
        source_data: GenerationDataClean,
        current_latent: torch.Tensor,
        full_position_template: PackedSequence,
        start_latent_frame: int,
    ) -> PackedSequence:
        """Pack only current vision tokens, then install their absolute request mRoPE IDs."""

        plan = replace(source_plan, condition_frame_indexes_vision=[])
        temporal_positions = None
        if source_data.temporal_positions_vision is not None:
            end = start_latent_frame + current_latent.shape[2]
            temporal_positions = [source_data.temporal_positions_vision[0][start_latent_frame:end]]
        compact_data = replace(
            source_data,
            raw_state_vision=None,
            x0_tokens_vision=[current_latent],
            temporal_positions_vision=temporal_positions,
        )
        packed = self._pack_input_sequence(
            [plan],
            text_tokens,
            compact_data,
            torch.zeros((1,), dtype=torch.float32),
            include_end_of_generation_token=self._derive_include_end_of_generation_token(),
            skip_text_tokens=skip_text_tokens,
        )
        assert packed.vision is not None and full_position_template.vision is not None
        compact_indexes = packed.vision.sequence_indexes
        full_indexes = full_position_template.vision.sequence_indexes
        total_frames = require_single_vision_5d(source_data.x0_tokens_vision).shape[2]
        if full_indexes.numel() % total_frames:
            raise ValueError("full vision token count is not divisible by latent frame count")
        tokens_per_frame = full_indexes.numel() // total_frames
        end_token = (start_latent_frame + current_latent.shape[2]) * tokens_per_frame
        source_indexes = full_indexes[start_latent_frame * tokens_per_frame : end_token]
        if compact_indexes.numel() != source_indexes.numel():
            raise ValueError("compact/full causal position geometry mismatch")
        packed.position_ids[:, compact_indexes] = full_position_template.position_ids[:, source_indexes]
        packed.to_cuda()
        return packed

    def _run_causal_prefill(
        self,
        *,
        net: torch.nn.Module | None,
        latent: torch.Tensor,
        start_latent_frame: int,
        block_id: int,
        cond_tokens: list[list[int]],
        uncond_tokens: list[list[int]],
        skip_text_tokens_for_cfg: bool,
        source_plan: SequencePlan,
        source_data: GenerationDataClean,
        cond_full_template: PackedSequence,
        uncond_full_template: PackedSequence | None,
        cond_cache: GenKVCache,
        uncond_cache: GenKVCache | None,
    ) -> None:
        cond_template = self._make_causal_current_block_template(
            text_tokens=cond_tokens,
            skip_text_tokens=False,
            source_plan=source_plan,
            source_data=source_data,
            current_latent=latent,
            full_position_template=cond_full_template,
            start_latent_frame=start_latent_frame,
        )
        assert cond_template.vision is not None
        block = CachedCleanBlock(
            block_id, start_latent_frame, latent.shape[2], cond_template.vision.sequence_indexes.numel()
        )
        zero = torch.zeros((1, 1), dtype=torch.float32, device=latent.device)
        self._update_inference_pack_template(cond_template, [latent], None, None, zero)
        cond_state = cond_cache.new_state(GenKVCacheMode.APPEND, block)
        self.denoise(net=net, data_batch_packed=cond_template, memory=cond_state)

        uncond_state: GenKVCacheMemoryState | None = None
        if uncond_cache is not None:
            assert uncond_full_template is not None
            uncond_template = self._make_causal_current_block_template(
                text_tokens=uncond_tokens,
                skip_text_tokens=skip_text_tokens_for_cfg,
                source_plan=source_plan,
                source_data=source_data,
                current_latent=latent,
                full_position_template=uncond_full_template,
                start_latent_frame=start_latent_frame,
            )
            self._update_inference_pack_template(uncond_template, [latent], None, None, zero)
            uncond_state = uncond_cache.new_state(GenKVCacheMode.APPEND, block)
            self.denoise(net=net, data_batch_packed=uncond_template, memory=uncond_state)
        commit_cfg_append_pair(cond_state, uncond_state)

    def _generate_causal_inference_from_prepared(
        self,
        *,
        data_batch: dict,
        net: torch.nn.Module | None,
        sampler: Any | None,
        guidance: float,
        guidance_interval: Optional[list[float]],
        velocity_postprocess_builder: Optional[Any],
        seed: list[int],
        n_sample: int,
        has_negative_prompt: bool,
        num_steps: int,
        shift: float,
        sigma_max: float,
        skip_text_tokens_for_cfg: bool,
        normalize_cfg: bool,
        sequence_plans: list[SequencePlan],
        gen_data_clean: GenerationDataClean,
        cond_tokens: list[list[int]],
        uncond_tokens: list[list[int]],
        initial_noise: list[torch.Tensor],
        condition_reference: list[torch.Tensor],
        condition_mask: list[torch.Tensor],
        has_noisy_actions: bool,
        causal_num_blocks: int | None,
        causal_block_size: int,
        causal_history_blocks: int,
    ) -> dict[str, list[torch.Tensor]]:
        del seed, n_sample, has_negative_prompt
        target_frames = self._validate_causal_inference_request(
            data_batch=data_batch,
            sequence_plans=sequence_plans,
            gen_data_clean=gen_data_clean,
            has_noisy_actions=has_noisy_actions,
            has_velocity_postprocess_builder=velocity_postprocess_builder is not None,
            causal_num_blocks=causal_num_blocks,
            causal_block_size=causal_block_size,
            causal_history_blocks=causal_history_blocks,
        )
        assert causal_num_blocks is not None
        if len(initial_noise) != 1 or len(condition_reference) != 1 or len(condition_mask) != 1:
            raise ValueError("causal inference requires one prepared sampler state")

        full_shape = require_single_vision_5d(gen_data_clean.x0_tokens_vision).shape
        full_initial = unflatten_vision_5d(initial_noise[0], full_shape)
        full_reference = unflatten_vision_5d(condition_reference[0], full_shape)
        full_data = replace(gen_data_clean, raw_state_vision=None)
        full_plan = replace(sequence_plans[0], condition_frame_indexes_vision=[])
        cond_full_template = self._pack_input_sequence(
            [full_plan],
            cond_tokens,
            full_data,
            torch.zeros((1,), dtype=torch.float32),
            include_end_of_generation_token=self._derive_include_end_of_generation_token(),
            skip_text_tokens=False,
        )
        use_cfg = guidance != 1.0
        uncond_full_template = None
        if use_cfg:
            uncond_full_template = self._pack_input_sequence(
                [full_plan],
                uncond_tokens,
                full_data,
                torch.zeros((1,), dtype=torch.float32),
                include_end_of_generation_token=self._derive_include_end_of_generation_token(),
                skip_text_tokens=skip_text_tokens_for_cfg,
            )

        assert cond_full_template.vision is not None
        tokens_per_frame = cond_full_template.vision.sequence_indexes.numel() // target_frames
        target_net = net or self.net
        if target_net is None:
            raise RuntimeError("causal inference requires a built network")
        num_layers = len(target_net.language_model.model.layers)
        max_clean_tokens = causal_history_blocks * causal_block_size * tokens_per_frame
        cond_cache = GenKVCache(num_layers, causal_history_blocks, max_clean_tokens)
        uncond_cache = GenKVCache(num_layers, causal_history_blocks, max_clean_tokens) if use_cfg else None
        selected_sampler = sampler or self.sampler
        finalized = [slice_vision_time(full_reference, 0, 1)]

        previous_dispatch = install_gen_kv_cache_attention_dispatch(target_net)
        try:
            self._run_causal_prefill(
                net=net,
                latent=finalized[0],
                start_latent_frame=0,
                block_id=0,
                cond_tokens=cond_tokens,
                uncond_tokens=uncond_tokens,
                skip_text_tokens_for_cfg=skip_text_tokens_for_cfg,
                source_plan=sequence_plans[0],
                source_data=gen_data_clean,
                cond_full_template=cond_full_template,
                uncond_full_template=uncond_full_template,
                cond_cache=cond_cache,
                uncond_cache=uncond_cache,
            )

            for block_id in range(1, causal_num_blocks + 1):
                start, end = causal_generated_block_span(block_id, causal_block_size)
                block_shape = slice_vision_time(full_initial, start, end).shape
                block_initial = slice_vision_time(full_initial, start, end)
                cond_template = self._make_causal_current_block_template(
                    text_tokens=cond_tokens,
                    skip_text_tokens=False,
                    source_plan=sequence_plans[0],
                    source_data=gen_data_clean,
                    current_latent=block_initial,
                    full_position_template=cond_full_template,
                    start_latent_frame=start,
                )
                uncond_template = None
                if use_cfg:
                    assert uncond_full_template is not None
                    uncond_template = self._make_causal_current_block_template(
                        text_tokens=uncond_tokens,
                        skip_text_tokens=skip_text_tokens_for_cfg,
                        source_plan=sequence_plans[0],
                        source_data=gen_data_clean,
                        current_latent=block_initial,
                        full_position_template=uncond_full_template,
                        start_latent_frame=start,
                    )

                def velocity_fn(noise_x: list[torch.Tensor], timestep: torch.Tensor) -> list[torch.Tensor]:
                    current = unflatten_vision_5d(noise_x[0], block_shape)

                    def branch(template: PackedSequence, cache: GenKVCache) -> list[torch.Tensor]:
                        self._update_inference_pack_template(template, [current], None, None, timestep)
                        output = self.denoise(
                            net=net,
                            data_batch_packed=template,
                            memory=cache.new_state(GenKVCacheMode.READONLY),
                        )
                        prediction = output["preds_vision"][0]
                        if prediction.ndim == 4:
                            prediction = prediction.unsqueeze(0)
                        return [flatten_vision_5d(prediction)]

                    needs_cfg = use_cfg
                    if needs_cfg and guidance_interval is not None:
                        t_lo, t_hi = guidance_interval
                        needs_cfg = t_lo < timestep[0].item() < t_hi
                    cond_velocity = branch(cond_template, cond_cache)
                    if not needs_cfg:
                        return cond_velocity
                    assert uncond_template is not None and uncond_cache is not None
                    uncond_velocity = branch(uncond_template, uncond_cache)
                    guided = [u + guidance * (c - u) for c, u in zip(cond_velocity, uncond_velocity, strict=True)]
                    if normalize_cfg:
                        guided = [
                            value * (torch.norm(cond) / (torch.norm(value) + 1e-8)).clamp(0.0, 1.0)
                            for value, cond in zip(guided, cond_velocity, strict=True)
                        ]
                    return guided

                sampler_state = [flatten_vision_5d(block_initial)]
                if isinstance(selected_sampler, FixedStepSampler):
                    sampled = selected_sampler(
                        velocity_fn,
                        sampler_state,
                        num_steps=len(selected_sampler.t_list) - 1,
                        shift=0.0,
                        seed=[None],
                        condition_reference=[torch.zeros_like(sampler_state[0])],
                        condition_mask=[torch.zeros_like(sampler_state[0])],
                    )
                elif self.config.rectified_flow_inference_config.scheduler_type == "unipc":
                    sampled = selected_sampler(
                        velocity_fn, sampler_state, num_steps=num_steps, shift=shift, seed=[None]
                    )
                else:

                    def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
                        timestep = sigma * float(self.config.rectified_flow_inference_config.num_train_timesteps)
                        return noise_x - sigma * velocity_fn([noise_x], timestep.reshape(1, 1))[0]

                    sampled = [
                        selected_sampler(
                            x0_fn,
                            sampler_state[0],
                            num_steps=num_steps,
                            sigma_max=sigma_max,
                            sigma_min=0.002,
                            solver_option="2ab",
                        )
                    ]
                block_clean = unflatten_vision_5d(sampled[0], block_shape)
                finalized.append(block_clean)
                if block_id != causal_num_blocks:
                    self._run_causal_prefill(
                        net=net,
                        latent=block_clean,
                        start_latent_frame=start,
                        block_id=block_id,
                        cond_tokens=cond_tokens,
                        uncond_tokens=uncond_tokens,
                        skip_text_tokens_for_cfg=skip_text_tokens_for_cfg,
                        source_plan=sequence_plans[0],
                        source_data=gen_data_clean,
                        cond_full_template=cond_full_template,
                        uncond_full_template=uncond_full_template,
                        cond_cache=cond_cache,
                        uncond_cache=uncond_cache,
                    )
        finally:
            restore_gen_kv_cache_attention_dispatch(previous_dispatch)

        result = concat_vision_time(finalized)
        if result.shape[2] != target_frames:
            raise RuntimeError(f"causal generator returned {result.shape[2]} latent frames, expected {target_frames}")
        return {"vision": [result]}
