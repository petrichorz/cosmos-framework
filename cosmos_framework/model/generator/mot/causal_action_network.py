# SPDX-License-Identifier: OpenMDW-1.1
"""Action TF projections with zero-time clean and condition representations."""

import math

import torch
from torch import nn

from .cosmos3_vfm_network import Cosmos3VFMNetwork


class SharedActionLinear(nn.Linear):
    """One projection for all sources; accepts legacy caller metadata only."""

    def forward(self, x, domain_id=None):
        return super().forward(x)


class CausalActionNetwork(Cosmos3VFMNetwork):
    def _create_action_interfaces(self):
        if self.action_dim != 80:
            raise ValueError("Causal mid-training requires the shared OpenWAM 80D layout")
        self.action2llm = SharedActionLinear(80, self.hidden_size)
        self.llm2action = SharedActionLinear(self.hidden_size, 80)
        self.state2llm = nn.Linear(80, self.hidden_size)
        self.state_modality_embed = nn.Parameter(torch.zeros(self.hidden_size))

    def _init_action_interfaces(self):
        for module in (self.action2llm, self.llm2action, self.state2llm):
            std = 1.0 / math.sqrt(module.in_features)
            nn.init.trunc_normal_(module.weight, std=std, a=-3 * std, b=3 * std)
            nn.init.zeros_(module.bias)
        nn.init.normal_(self.state_modality_embed, std=1.0 / math.sqrt(self.hidden_size))

    def _encode_vision(self, packed_seq, packed_sequence, target_dtype):
        original_shapes = super()._encode_vision(packed_seq, packed_sequence, target_dtype)
        vision = packed_seq.vision
        cond = torch.cat([
            mask.reshape(-1).bool().repeat_interleave(h * w)
            for mask, (_, h, w) in zip(vision.condition_mask, vision.token_shapes, strict=True)
        ])
        indexes = vision.sequence_indexes[cond]
        zeros = torch.zeros(indexes.numel(), device=packed_sequence.device, dtype=torch.float32)
        packed_sequence[indexes] += self._embed_packed_timesteps(zeros, packed_seq).to(target_dtype)
        return original_shapes

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
            if tf.state_tokens is not None:
                states = torch.cat(tf.state_tokens).to(device=packed_sequence.device, dtype=target_dtype)
                masks = torch.cat([m.state_mask for m in packed_seq.causal_action_metadata]).to(states.device)
                states = states.masked_fill(~masks, 0)
                encoded = self.state2llm(states) + self.state_modality_embed.view(1, -1)
                zeros = torch.zeros(len(states), device=states.device, dtype=torch.float32)
                packed_sequence[tf.layout.state_indexes] = encoded + self._embed_packed_timesteps(zeros, packed_seq).to(
                    target_dtype
                )

    def _decode_action(self, packed_seq, last_hidden_state, output_dict):
        super()._decode_action(packed_seq, last_hidden_state, output_dict)
        output_dict["preds_action"] = [
            x.masked_fill(~m.action_mask.to(x.device), 0)
            for x, m in zip(output_dict["preds_action"], packed_seq.causal_action_metadata, strict=True)
        ]
