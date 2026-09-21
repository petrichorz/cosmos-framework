# SPDX-License-Identifier: OpenMDW-1.1
"""Separate action training entry; the video-only causal entry remains unchanged."""

import torch

from cosmos_framework.data.generator.sequence_packing.causal_action import action_frame_ids, expand_action_sequence
from cosmos_framework.data.generator.sequence_packing.teacher_forcing import sample_teacher_forcing_geometry
from cosmos_framework.model.generator.causal_teacher_forcing import validate_teacher_forcing_config
from cosmos_framework.model.generator.mot.causal_action_network import CausalActionNetwork
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel


def aligned_action_schedule(schedule, packed):
    rows = []
    for i, (vshape, actions) in enumerate(zip(packed.vision.token_shapes, packed.action.tokens, strict=True)):
        ids = action_frame_ids(actions.shape[0], vshape[0]).clamp_min(0).to(schedule.device)
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

    def prepare_teacher_forcing_geometry(self, num_vision_latent_frames, condition_frame_indexes_vision=None):
        c = self.config
        return sample_teacher_forcing_geometry(
            num_samples=len(num_vision_latent_frames),
            block_size_min=c.teacher_forcing_block_size_min,
            block_size_max=c.teacher_forcing_block_size_max,
            history_blocks_min=c.teacher_forcing_history_blocks_min,
            history_blocks_max=c.teacher_forcing_history_blocks_max,
        )

    def _pack_input_sequence(self, sequence_plans, input_text_indexes, gen_data_clean, input_timesteps, **kwargs):
        if any(not p.has_action or not p.has_vision or p.has_sound for p in sequence_plans):
            raise ValueError("Use separate jobs for video-only and action samples")
        if self.tokenizer_vision_gen.temporal_compression_factor != 4:
            raise ValueError("Causal DROID v1 requires temporal compression 4")
        input_timesteps = input_timesteps.reshape(len(sequence_plans), -1)
        packed = super()._pack_input_sequence(
            sequence_plans, input_text_indexes, gen_data_clean, input_timesteps[:, :1], **kwargs
        )
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
        # at the source action rate. State and synthetic slots share the first frame's time.
        action_offset = vision_offset = 0
        if not self.config.diffusion_expert_config.enable_fps_modulation:
            packed.position_ids = packed.position_ids.float()
        for vs, actions in zip(packed.vision.token_shapes, packed.action.tokens, strict=True):
            ids = action_frame_ids(actions.shape[0], vs[0])
            prefix = int((ids <= 0).sum())
            indexes = packed.action.sequence_indexes[action_offset : action_offset + actions.shape[0]]
            first_time = packed.position_ids[0, packed.vision.sequence_indexes[vision_offset]].clone()
            packed.position_ids[0, indexes[:prefix]] = first_time
            if not self.config.diffusion_expert_config.enable_fps_modulation:
                raw_times = (torch.arange(actions.shape[0]) - prefix + 1).clamp_min(0)
                packed.position_ids[0, indexes] = first_time + raw_times.float() / int((ids == 0).sum())
            action_offset += actions.shape[0]
            vision_offset += int(torch.tensor(vs).prod())
        return packed

    def _add_noise_to_input(
        self, gen_data_clean, packed_sequence, sigmas, sigmas_action=None, sigmas_sound=None, iteration=None
    ):
        return super()._add_noise_to_input(
            gen_data_clean,
            packed_sequence,
            sigmas,
            sigmas_action=aligned_action_schedule(sigmas, packed_sequence),
            sigmas_sound=sigmas_sound,
            iteration=iteration,
        )

    def _compute_flow_matching_loss(self, *args, **kwargs):
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
        return super()._compute_losses(
            out_net,
            data_batch_packed,
            gen_data_noised,
            timesteps,
            is_image_batch,
            timesteps_action=aligned_action_schedule(timesteps, data_batch_packed),
            timesteps_sound=timesteps_sound,
        )

    def post_noise_packing_hook(self, packed_sequence, gen_data_clean, teacher_forcing_geometry=None):
        if teacher_forcing_geometry is None:
            raise ValueError("Action teacher forcing requires the geometry used for noise sampling")
        return expand_action_sequence(
            packed_sequence,
            [x.to(dtype=self.precision) for x in gen_data_clean.x0_tokens_vision],
            [x.to(dtype=self.precision) for x in gen_data_clean.x0_tokens_action],
            teacher_forcing_geometry,
        )

    def _generate_causal_inference_from_prepared(self, **kwargs):
        from cosmos_framework.inference.causal_action.sampling import sample_prepared

        return sample_prepared(self, **kwargs)
