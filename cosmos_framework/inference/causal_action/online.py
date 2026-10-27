# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Streaming observed-video encoding and receding-horizon causal policy sampling."""

import json
from dataclasses import replace

import numpy as np
import torch

from cosmos_framework.data.generator.action.block_state import BlockStateMetadata
from cosmos_framework.data.generator.action.causal_block_geometry import CausalBlockGeometry
from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline
from cosmos_framework.data.generator.action.video_view import VideoViewConfig
from cosmos_framework.data.generator.sequence_packing.causal_action import CausalActionGeometry, expand_action_sequence
from cosmos_framework.data.generator.sequence_packing.mrope import get_3d_mrope_ids_vae_tokens
from cosmos_framework.inference.causal_action.streaming_cache import ObservedVideoCache
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean


class StreamingObservationEncoder:
    """Wan2.2 encoder with explicit episode-owned causal convolution features."""

    def __init__(self, tokenizer):
        from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import Wan2pt2VAEInterface

        if not isinstance(tokenizer, Wan2pt2VAEInterface):
            raise ValueError("Streaming causal action serving requires the Wan2.2 4x16x16 VAE")
        self.vae = tokenizer.model
        self.cache = None

    def reset(self):
        self.cache = None

    @torch.no_grad()
    def encode(self, pixels):
        from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import patchify

        first = self.cache is None
        expected = 1 if first else 4
        if pixels.shape[2] != expected:
            raise ValueError(f"Expected {expected} subsampled frames per streaming VAE call")
        core = self.vae.model
        cache = core._new_enc_cache() if first else self.cache
        x = patchify(pixels.to(self.vae.dtype), patch_size=2).contiguous()
        latent, cache = core._encode_chunk_impl(x, cache, self.vae.scale)
        if latent.shape[2] != 1:
            raise RuntimeError("Each observation update must encode exactly one latent")
        self.cache = cache
        return latent.float()


class CausalPolicyEngine:
    def __init__(
        self,
        model,
        *,
        statistics,
        template,
        actions_per_block=32,
        execute_steps=None,
        video_stride=4,
        history_blocks=32,
        steps=10,
        guidance=1.0,
        shift=5.0,
        resolution="480",
        fps=30.0,
        seed=42,
    ):
        self.model, self.statistics, self.template = model, statistics, template
        self.geometry = CausalBlockGeometry(
            actions_per_block=actions_per_block,
            video_stride=video_stride,
            temporal_compression_factor=model.tokenizer_vision_gen.temporal_compression_factor,
        )
        if execute_steps is None:
            execute_steps = actions_per_block
        if type(execute_steps) is not int or execute_steps != actions_per_block:
            raise ValueError(
                "Training-block joint prefill requires execute_steps == actions_per_block; "
                "partial-block execution feedback is not supported"
            )
        if steps < 1 or history_blocks < 1 or not np.isfinite([guidance, shift]).all():
            raise ValueError("Invalid sampling/history options")
        self.execute_steps = execute_steps
        self.steps, self.guidance, self.shift = steps, guidance, shift
        self.fps = self.geometry.action_fps(fps)
        self.resolution, self.seed = resolution, seed
        self.encoder = StreamingObservationEncoder(model.tokenizer_vision_gen)
        self.transform = ActionTransformPipeline(
            max_action_dim=template.width,
            video_temporal_downsample=4,
            cfg_dropout_rate=0,
            format_prompt_as_json=True,
        )
        self.views = VideoViewConfig(cameras={"head": "cam_high", "left": "cam_left_wrist", "right": "cam_right_wrist"})
        count = len(model.net.language_model.model.layers)
        self.caches = {
            name: ObservedVideoCache(count, history_blocks * self.geometry.latent_frames_per_block)
            for name in (["cond"] if guidance == 1 else ["cond", "uncond"])
        }
        self.history_blocks = history_blocks
        self.reset()

    def reset(self):
        self.encoder.reset()
        for cache in self.caches.values():
            cache.clear()
        self.frame = -1
        self.latest = self.frame_size = self.pixel_shape = None
        self.text_tokens = None

    @torch.no_grad()
    def observe(self, cameras, *, state, state_mask, action_mask, prompt):
        # Match the training concat_view composition, then its resize/reflection pad.
        videos = {key: torch.from_numpy(value.copy()).permute(0, 3, 1, 2) for key, value in cameras.items()}
        video = self.views.compose(videos, len(next(iter(cameras.values()))), viewpoint="concat_view")
        first = self.frame == -1
        expected = 1 if first else self.execute_steps
        if video.shape[1] != expected:
            raise ValueError(f"Expected {expected} observation frames per feedback update")
        video = video[:, :1] if first else video[:, self.geometry.video_stride - 1 :: self.geometry.video_stride]
        sample = dict(
            video=video,
            action=torch.zeros(self.geometry.actions_per_block, self.template.width),
            ai_caption=prompt,
            mode="policy",
            viewpoint="concat_view",
            conditioning_fps=torch.tensor(self.geometry.video_fps(self.fps)),
            additional_view_description=self.views.describe(viewpoint="concat_view"),
        )
        # Prompt construction uses the same transform as training. It excludes duration/idle metadata.
        sample = self.transform(sample, self.resolution)
        frame_size = sample["image_size"]
        if self.frame_size is not None and not torch.equal(self.frame_size, frame_size):
            raise ValueError("Camera resolution/layout cannot change within an episode")
        self.frame_size, self.pixel_shape = frame_size, sample["video"].shape[-2:]
        pixels = sample["video"].unsqueeze(0).to(**self.model.tensor_kwargs_fp32) / 127.5 - 1
        if self.text_tokens is None:
            caption = sample["ai_caption"]
            if isinstance(caption, dict):
                caption = json.dumps(caption)
            self.text_tokens = self.model._get_inference_text_tokens({self.model.input_caption_key: [caption]}, False)
        # VAE convolutions remain causal; transformer clean attention is joint within a training block.
        chunk_frames = 1 if first else self.geometry.temporal_compression_factor
        latents = []
        for chunk in pixels.split(chunk_frames, dim=2):
            latent = self.encoder.encode(chunk)
            latents.append(self.model._remove_padding_from_latent([latent], [frame_size])[0])
        observed = torch.cat(latents, dim=2)
        start = self.frame + 1
        self.frame += observed.shape[2]
        self.latest = observed[:, :, -1:]
        for name, cache in self.caches.items():
            packed = self._pack(name, state, state_mask, action_mask, observed=None if first else observed)
            with cache.forward(
                self.model.net,
                append_frames=tuple(range(start, self.frame + 1)),
                clean_block=-1 if first else 0,
            ):
                self.model.denoise(data_batch_packed=packed)

    def _pack(self, name, state, state_mask, action_mask, *, observed=None):
        g, model = self.geometry, self.model
        device = self.latest.device
        state = state.to(device)
        state_mask, action_mask = state_mask.to(device), action_mask.to(device)
        normalized, _ = self.statistics.state.normalize(state, state_mask)
        metadata = BlockStateMetadata(
            block_size=g.latent_frames_per_block,
            history_blocks=self.history_blocks,
            action_frame_ids=torch.arange(1, 1 + g.latent_frames_per_block, device=device).repeat_interleave(
                g.actions_per_latent_frame
            ),
            action_mask=action_mask.expand(g.actions_per_block, -1),
            states=normalized[None],
            state_mask=state_mask[None],
            state_action_indexes=torch.zeros(1, dtype=torch.long, device=device),
            anchors=state[None],
            state_latent_indexes=torch.zeros(1, dtype=torch.long, device=device),
            statistics=self.statistics,
            template=self.template,
        )
        from cosmos_framework.data.generator.action.transforms import build_sequence_plan_from_mode

        plan = replace(
            build_sequence_plan_from_mode(
                "policy",
                video_length=1 + g.actions_per_block // g.video_stride,
                action_length=g.actions_per_block,
                video_temporal_downsample=4,
            ),
            causal_action_metadata=metadata,
            action_start_frame_offset=1,
        )
        if observed is None:
            vision = torch.cat(
                [
                    self.latest,
                    self.latest.new_zeros(1, self.latest.shape[1], g.latent_frames_per_block, *self.latest.shape[-2:]),
                ],
                dim=2,
            )
            frame_offset = self.frame
        else:
            if observed.shape[2] != g.latent_frames_per_block:
                raise ValueError("Observation prefill requires one complete training block")
            # The compact pack requires a prefix slot; only clean block 0 is queried/committed.
            vision = torch.cat([observed[:, :, :1], observed], dim=2)
            frame_offset = self.frame - g.latent_frames_per_block
        data = GenerationDataClean(
            batch_size=1,
            is_image_batch=False,
            x0_tokens_vision=[vision],
            x0_tokens_action=[torch.zeros(g.actions_per_block, self.template.width, device=device)],
            fps_vision=torch.tensor([g.video_fps(self.fps)], device=device),
            fps_action=torch.tensor([self.fps], device=device),
        )
        tokens = self.text_tokens[0 if name == "cond" else 1]
        packed = model._pack_input_sequence([plan], tokens, data, torch.zeros(1, 1))
        packed = expand_action_sequence(
            packed,
            [vision.to(model.precision)],
            CausalActionGeometry((g.latent_frames_per_block,), (self.history_blocks,)),
        )
        # Keep cached RoPE coordinates absolute, even when old video KVs are evicted.
        indexes = packed.vision.sequence_indexes
        temporal_offset = packed.position_ids[0, indexes[0]].item()
        config = model.config.diffusion_expert_config
        t, h, w = packed.vision.token_shapes[0]
        # Generate absolute positions with the training formula; subtracting two rounded
        # FPS-scaled positions to recover a step accumulates temporal rounding errors.
        positions, _ = get_3d_mrope_ids_vae_tokens(
            grid_t=t,
            grid_h=h,
            grid_w=w,
            temporal_offset=temporal_offset,
            reset_spatial_indices=config.unified_3d_mrope_reset_spatial_ids,
            fps=data.fps_vision.item() if config.enable_fps_modulation else None,
            base_fps=config.base_fps,
            temporal_compression_factor=g.temporal_compression_factor,
            start_frame_offset=frame_offset,
        )
        layout = packed.teacher_forcing.layout
        visual = layout.gen_query_indexes[layout.stream_ids[layout.gen_query_indexes] == 0]
        packed.position_ids[:, visual] = positions.to(packed.position_ids)
        packed.position_ids[:, indexes] = positions.to(packed.position_ids)
        packed.position_ids[:, layout.state_indexes] = positions[:, :1].to(packed.position_ids)
        action_indexes = packed.action.sequence_indexes
        action_positions, _ = get_3d_mrope_ids_vae_tokens(
            grid_t=g.actions_per_block,
            grid_h=1,
            grid_w=1,
            temporal_offset=temporal_offset,
            reset_spatial_indices=config.unified_3d_mrope_reset_spatial_ids,
            fps=data.fps_action.item() if config.enable_fps_modulation else None,
            base_fps=config.base_fps,
            temporal_compression_factor=1,
            base_temporal_compression_factor=g.temporal_compression_factor,
            start_frame_offset=1 + frame_offset * g.actions_per_latent_frame,
        )
        packed.position_ids[:, action_indexes] = action_positions.to(packed.position_ids)
        packed.to_cuda()
        return packed

    @torch.no_grad()
    def predict(self, *, state, state_mask, action_mask, revision):
        model, g = self.model, self.geometry
        if model.config.rectified_flow_inference_config.scheduler_type != "unipc":
            raise ValueError("Online causal policy currently requires the UniPC sampler")
        templates = {name: self._pack(name, state, state_mask, action_mask) for name in self.caches}
        shape = (1, self.latest.shape[1], g.latent_frames_per_block, *self.latest.shape[-2:])
        generator = torch.Generator(device=self.latest.device).manual_seed(self.seed + revision)
        vnoise = torch.randn(shape, generator=generator, device=self.latest.device)
        anoise = torch.randn((g.actions_per_block, self.template.width), generator=generator, device=self.latest.device)
        mask = action_mask.to(self.latest.device)
        anoise.masked_fill_(~mask, 0)
        nv = vnoise.numel()

        def velocity(noise, timestep):
            vision = torch.cat([self.latest, noise[0][:nv].reshape(shape)], dim=2)
            action = noise[0][nv:].reshape(anoise.shape).masked_fill(~mask, 0)
            outputs = {}
            for name, cache in self.caches.items():
                packed = templates[name]
                model._update_inference_pack_template(packed, [vision], [action], None, timestep)
                with cache.forward(model.net):
                    out = model.denoise(data_batch_packed=packed)
                outputs[name] = torch.cat(
                    [
                        out["preds_vision"][0].reshape(vision.shape)[:, :, 1:].flatten(),
                        out["preds_action"][0].masked_fill(~mask, 0).flatten(),
                    ]
                )
            result = outputs["cond"]
            if "uncond" in outputs:
                result = outputs["uncond"] + self.guidance * (result - outputs["uncond"])
            return [result]

        sampled = model.sampler(
            velocity,
            [torch.cat([vnoise.flatten(), anoise.flatten()])],
            num_steps=self.steps,
            shift=self.shift,
            seed=[None],
        )[0]
        delta = self.statistics.action.denormalize(sampled[nv:].reshape(anoise.shape), mask)
        actions = self.template.decode_action_delta(delta, state.to(delta), mask)
        video_latents = sampled[:nv].reshape(shape)
        # The caller assembles the selected latents and decodes the whole video
        # once after the episode.
        # Predicted latents never enter the observed cache or the streaming VAE.
        return dict(
            action=actions.float().cpu().numpy(),
            action_normalized=sampled[nv:].reshape(anoise.shape).float().cpu().numpy(),
            video_latents=video_latents.float().cpu().numpy(),
        )
