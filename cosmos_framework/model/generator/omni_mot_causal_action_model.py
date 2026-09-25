# SPDX-License-Identifier: OpenMDW-1.1
"""Separate action training entry; the video-only causal entry remains unchanged."""

from contextlib import contextmanager

import torch

from cosmos_framework.data.generator.sequence_packing.causal_action import CausalActionGeometry, expand_action_sequence
from cosmos_framework.model.generator.causal_teacher_forcing import validate_teacher_forcing_config
from cosmos_framework.model.generator.mot.causal_action_network import CausalActionNetwork
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel


def aligned_action_schedule(schedule, packed):
    rows = []
    for i, (vshape, actions) in enumerate(zip(packed.vision.token_shapes, packed.action.tokens, strict=True)):
        ids = packed.causal_action_metadata[i].action_frame_ids.to(schedule.device)
        row = schedule[i]
        rows.append(row.expand(ids.numel()) if row.numel() == 1 else row[ids])
    return torch.nn.utils.rnn.pad_sequence(rows, batch_first=True)


class OmniMoTCausalActionModel(OmniMoTModel):
    network_cls = CausalActionNetwork

    def __init__(self, config):
        # Reuse geometry/backend validation without relaxing the video model's contract.
        import copy

        video_config = copy.copy(config)
        video_config.action_gen = False
        validate_teacher_forcing_config(video_config)
        if not config.action_gen:
            raise ValueError("Causal action jobs require action_gen=True")
        if config.rectified_flow_training_config.independent_action_schedule:
            raise ValueError("Causal action v1 uses shared per-block video/action timesteps")
        super().__init__(config)

    def prepare_teacher_forcing_geometry_from_plans(self, num_frames, plans):
        metadata = [p.causal_action_metadata for p in plans]
        if any(m is None for m in metadata):
            raise ValueError("Causal action requires measured block-state metadata from a source adapter")
        for frames, m in zip(num_frames, metadata, strict=True):
            if int(m.action_frame_ids.max()) + 1 != frames:
                raise ValueError("Data/model latent geometry mismatch")
        return CausalActionGeometry(tuple(m.block_size for m in metadata), tuple(m.history_blocks for m in metadata))

    def _pack_input_sequence(self, sequence_plans, input_text_indexes, gen_data_clean, input_timesteps, **kwargs):
        if any(not p.has_action or not p.has_vision or p.has_sound for p in sequence_plans):
            raise ValueError("Use separate jobs for video-only and action samples")
        input_timesteps = input_timesteps.reshape(len(sequence_plans), -1)
        packed = super()._pack_input_sequence(
            sequence_plans, input_text_indexes, gen_data_clean, input_timesteps[:, :1], **kwargs
        )
        packed.causal_action_metadata = [p.causal_action_metadata for p in sequence_plans]
        if any(m is None for m in packed.causal_action_metadata):
            raise ValueError("Block geometry and current measured state metadata are required for causal action")
        action_ts = aligned_action_schedule(input_timesteps, packed)
        for mod, ts in ((packed.vision, input_timesteps), (packed.action, action_ts)):
            mod.timesteps = torch.cat(
                [
                    (ts[i, nfi.cpu()] if ts.shape[1] > 1 else ts[i].expand(nfi.numel())).repeat_interleave(
                        int(torch.tensor(mod.token_shapes[i][1:]).prod())
                    )
                    for i, nfi in enumerate(mod.noisy_frame_indexes)
                ]
            ).float()
        # First latent occupies time zero; real actions cover raw-frame intervals
        # at the source action rate; no synthetic action slots are constructed.
        action_offset = vision_offset = 0
        if not self.config.diffusion_expert_config.enable_fps_modulation:
            packed.position_ids = packed.position_ids.float()
        for vs, actions, metadata in zip(
            packed.vision.token_shapes, packed.action.tokens, packed.causal_action_metadata, strict=True
        ):
            ids = metadata.action_frame_ids.cpu()
            prefix = int((ids <= 0).sum())
            indexes = packed.action.sequence_indexes[action_offset : action_offset + actions.shape[0]]
            first_time = packed.position_ids[0, packed.vision.sequence_indexes[vision_offset]].clone()
            packed.position_ids[0, indexes[:prefix]] = first_time
            if not self.config.diffusion_expert_config.enable_fps_modulation:
                raw_times = (torch.arange(actions.shape[0]) - prefix + 1).clamp_min(0)
                packed.position_ids[0, indexes] = first_time + raw_times.float() / int((ids == 1).sum())
            action_offset += actions.shape[0]
            vision_offset += int(torch.tensor(vs).prod())
        return packed

    @contextmanager
    def _fixed_debug_rng(self, offset=0):
        seed = getattr(self.config, "causal_action_debug_noise_seed", None)
        if seed is None:
            yield
            return
        device = torch.device(self.tensor_kwargs_fp32["device"])
        devices = [] if device.type == "cpu" else [device]
        with torch.random.fork_rng(devices=devices, device_type="cuda" if device.type == "cpu" else device.type):
            torch.manual_seed(seed + offset)
            if device.type != "cpu":
                getattr(torch, device.type).manual_seed(seed + offset)
            yield

    def _get_train_noise_level_vision(self, *args, **kwargs):
        if getattr(self.config, "causal_action_debug_noise_seed", None) is not None:
            kwargs["iteration"] = None
        with self._fixed_debug_rng(offset=1):
            return super()._get_train_noise_level_vision(*args, **kwargs)

    def _add_noise_to_input(
        self, gen_data_clean, packed_sequence, sigmas, sigmas_action=None, sigmas_sound=None, iteration=None
    ):
        if getattr(self.config, "causal_action_debug_noise_seed", None) is not None:
            iteration = None
        with self._fixed_debug_rng():
            noised = super()._add_noise_to_input(
                gen_data_clean,
                packed_sequence,
                sigmas,
                sigmas_action=aligned_action_schedule(sigmas, packed_sequence),
                sigmas_sound=sigmas_sound,
                iteration=iteration,
            )

        for i, metadata in enumerate(packed_sequence.causal_action_metadata):
            for values in (noised.xt_tokens_action, noised.vt_target_action, noised.epsilon_action):
                values[i] = values[i].masked_fill(~metadata.action_mask.to(values[i].device), 0)
        return noised

    def _compute_flow_matching_loss(self, *args, **kwargs):
        kwargs["normalize_by_active"] = True
        loss, per_instance = super()._compute_flow_matching_loss(*args, **kwargs)
        # ID has a dummy vision loss. Keep its zero in FP32 so the base
        # accumulator does not round the real action loss back to BF16.
        return loss.float(), per_instance.float()

    def _compute_losses(
        self,
        out_net,
        data_batch_packed,
        gen_data_noised,
        timesteps,
        is_image_batch,
        timesteps_action=None,
        timesteps_sound=None,
    ):
        action = data_batch_packed.action
        original_mask = action.condition_mask
        action.condition_mask = [
            1.0 - (1.0 - c) * m.action_mask.to(c.device)
            for c, m in zip(original_mask, data_batch_packed.causal_action_metadata, strict=True)
        ]
        try:
            return super()._compute_losses(
                out_net,
                data_batch_packed,
                gen_data_noised,
                timesteps,
                is_image_batch,
                timesteps_action=aligned_action_schedule(timesteps, data_batch_packed),
                timesteps_sound=timesteps_sound,
            )
        finally:
            action.condition_mask = original_mask

    def post_noise_packing_hook(self, packed_sequence, gen_data_clean, teacher_forcing_geometry=None):
        if teacher_forcing_geometry is None:
            raise ValueError("Action teacher forcing requires the geometry used for noise sampling")
        return expand_action_sequence(
            packed_sequence,
            [x.to(dtype=self.precision) for x in gen_data_clean.x0_tokens_vision],
            teacher_forcing_geometry,
        )

    def _generate_causal_inference_from_prepared(self, **kwargs):
        from cosmos_framework.inference.causal_action.sampling import sample_prepared

        batch = kwargs.get("data_batch") or {}
        if batch.get("causal_action_preview", False):
            # Ordinary inference encodes conditioning pixels only. A declared
            # truth-state preview needs separately encoded GT history, while
            # the noisy stream still uses only its legitimate conditions.
            truth = batch.get("causal_action_truth_vision")
            if truth is None and getattr(self, "input_video_key", "video") in batch:
                truth = self.get_data_and_condition(batch).x0_tokens_vision
            kwargs["preview_history_vision"] = truth
        return sample_prepared(self, **kwargs)
