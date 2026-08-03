# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import torch
from torch import nn

from cosmos_framework.data.generator.sequence_packing.modality import ModalityData, ModalitySpan
from cosmos_framework.data.generator.sequence_packing.runtime import from_all_seq, get_all_seq
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence
from cosmos_framework.data.generator.sequence_packing.teacher_forcing import (
    expand_packed_sequence_for_teacher_forcing,
)
from cosmos_framework.model.generator.mot.attention import dispatch_attention
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import (
    Cosmos3VFMNetwork,
    Cosmos3VFMNetworkConfig,
)


class _TinyEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)


class _TinyLanguageModel(nn.Module):
    """One attention layer implementing the VFM language-model call contract."""

    def __init__(
        self,
        *,
        hidden_size: int = 16,
        num_heads: int = 4,
        num_kv_heads: int = 2,
        head_dim: int = 4,
    ) -> None:
        super().__init__()
        self.model = _TinyEmbeddings(256, hidden_size)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(self, input_pack, *, attention_mask, **kwargs):
        del kwargs
        hidden_states = get_all_seq(input_pack)
        query = self.q_proj(hidden_states).reshape(-1, self.num_heads, self.head_dim)
        key = self.k_proj(hidden_states).reshape(-1, self.num_kv_heads, self.head_dim)
        value = self.v_proj(hidden_states).reshape(-1, self.num_kv_heads, self.head_dim)
        output_pack, kv_to_store = dispatch_attention(
            from_all_seq(query, input_pack),
            from_all_seq(key, input_pack),
            from_all_seq(value, input_pack),
            attention_mask,
        )
        assert kv_to_store is None
        output = self.o_proj(get_all_seq(output_pack))
        return from_all_seq(output, input_pack), {}


def _make_network() -> Cosmos3VFMNetwork:
    text_config = SimpleNamespace(
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_hidden_layers=1,
    )
    config = Cosmos3VFMNetworkConfig(
        vlm_config=SimpleNamespace(text_config=text_config),
        vision_gen=True,
        action_gen=False,
        sound_gen=False,
        latent_patch_size=1,
        latent_channel_size=2,
        max_latent_h=1,
        max_latent_w=1,
        max_latent_t=3,
        joint_attn_implementation="two_way",
        teacher_forcing_max_sequence_length=1_000,
    )
    return Cosmos3VFMNetwork(_TinyLanguageModel(), config)


def _make_sequence() -> tuple[PackedSequence, torch.Tensor, torch.Tensor]:
    noisy = torch.tensor(
        [[[[[0.1]], [[0.2]], [[0.3]]], [[[0.4]], [[0.5]], [[0.6]]]]],
        requires_grad=True,
    )
    clean = torch.tensor(
        [[[[[1.1]], [[1.2]], [[1.3]]], [[[1.4]], [[1.5]], [[1.6]]]]],
        requires_grad=True,
    )
    vision = ModalityData(
        sequence_indexes=torch.tensor([2, 3, 4]),
        timesteps=torch.tensor([500.0, 500.0, 500.0]),
        mse_loss_indexes=torch.tensor([2, 3, 4]),
        spans=[
            ModalitySpan(2, 1, 0, 0, 1, (1, 1, 1)),
            ModalitySpan(3, 1, 0, 1, 1, (1, 1, 1)),
            ModalitySpan(4, 1, 0, 2, 1, (1, 1, 1)),
        ],
        token_shapes=[(3, 1, 1)],
        tokens=[noisy],
        condition_mask=[torch.zeros(3, 1, 1)],
        noisy_frame_indexes=[torch.tensor([0, 1, 2])],
    )
    source = PackedSequence(
        sample_lens=[5],
        split_lens=[2, 3],
        attn_modes=["causal", "full"],
        is_image_batch=False,
        uses_single_timestep=True,
        sequence_length=5,
        text_ids=torch.tensor([11, 12]),
        text_indexes=torch.tensor([0, 1]),
        position_ids=torch.arange(15).reshape(3, 5),
        vision=vision,
    )
    expanded = expand_packed_sequence_for_teacher_forcing(
        source,
        clean_vision_tokens=[clean],
        block_size=1,
        history_blocks=2,
    )
    return expanded, clean, noisy


def test_teacher_forcing_network_vertical_slice_backpropagates_through_both_streams():
    torch.manual_seed(1234)
    network = _make_network()
    packed_sequence, clean, noisy = _make_sequence()

    output = network(packed_sequence)
    prediction = output["preds_vision"][0]
    loss = prediction.square().mean()
    loss.backward()

    assert prediction.shape == noisy.shape
    assert torch.isfinite(prediction).all()
    assert clean.grad is not None
    assert torch.isfinite(clean.grad).all()
    assert clean.grad.abs().sum() > 0
    assert noisy.grad is not None
    assert torch.isfinite(noisy.grad).all()
    assert noisy.grad.abs().sum() > 0
    parameter_grads = [parameter.grad for parameter in network.parameters() if parameter.grad is not None]
    assert parameter_grads
    assert all(torch.isfinite(grad).all() for grad in parameter_grads)
    assert any(grad.abs().sum() > 0 for grad in parameter_grads)
