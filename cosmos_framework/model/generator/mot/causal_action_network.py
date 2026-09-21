# SPDX-License-Identifier: OpenMDW-1.1
"""Action TF projections with zero-time clean and condition representations."""

import torch

from .cosmos3_vfm_network import Cosmos3VFMNetwork


class CausalActionNetwork(Cosmos3VFMNetwork):
    def _encode_action(self, packed_seq, packed_sequence, target_dtype):
        super()._encode_action(packed_seq, packed_sequence, target_dtype)
        action = packed_seq.action
        if action is None:
            raise ValueError("Causal action network requires action data")
        cond = torch.cat([mask.reshape(-1).bool() for mask in action.condition_mask])
        indexes = action.sequence_indexes[cond]
        zeros = torch.zeros(indexes.numel(), device=packed_sequence.device, dtype=torch.float32)
        packed_sequence[indexes] = packed_sequence[indexes] + self._embed_packed_timesteps(zeros, packed_seq).to(
            target_dtype
        )
        tf = packed_seq.teacher_forcing
        if tf is not None:
            tokens, domains = self.pack_action(tf.clean_action_tokens, action.token_shapes, action.domain_id)
            tokens = self.action2llm(tokens.to(target_dtype), domains) + self.action_modality_embed.view(1, -1)
            zeros = torch.zeros(tokens.shape[0], device=tokens.device, dtype=torch.float32)
            packed_sequence[tf.layout.clean_action_indexes] = tokens + self._embed_packed_timesteps(
                zeros, packed_seq
            ).to(target_dtype)
