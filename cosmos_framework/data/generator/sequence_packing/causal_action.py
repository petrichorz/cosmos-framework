# SPDX-License-Identifier: OpenMDW-1.1
"""Action teacher forcing with visual-only history and current block state.

Stream 0 is clean visual history, 1 current targets, 2 current conditions.
Stream 3 contains one immutable measured state per block.
"""

from dataclasses import dataclass, fields, replace

import torch

from cosmos_framework.data.generator.action.block_state import prediction_block_ids

from .teacher_forcing import TeacherForcingData, TeacherForcingGeometry, TeacherForcingLayout


@dataclass(frozen=True)
class CausalActionGeometry(TeacherForcingGeometry):
    def __post_init__(self):
        if not self.block_sizes or len(self.block_sizes) != len(self.history_blocks):
            raise ValueError("One block size and history window required per sample")
        if any(b < 1 for b in self.block_sizes) or any(h < 1 for h in self.history_blocks):
            raise ValueError("Block sizes must be positive and history windows positive")


@dataclass(frozen=True)
class CausalActionLayout(TeacherForcingLayout):
    includes_action: bool = True
    state_indexes: torch.Tensor | None = None

    def to(self, device):
        return replace(
            self,
            **{
                f.name: getattr(self, f.name).to(device)
                for f in fields(self)
                if isinstance(getattr(self, f.name), torch.Tensor)
            },
        )


@dataclass
class CausalActionData(TeacherForcingData):
    state_tokens: list[torch.Tensor] | None = None

    def to_cuda(self):
        super().to_cuda()
        if self.state_tokens is not None:
            self.state_tokens = [x.cuda() for x in self.state_tokens]


def build_action_layout(
    *,
    und_counts,
    vision_shapes,
    action_lengths,
    vision_conditions,
    action_conditions,
    geometry: TeacherForcingGeometry,
    metadata,
):
    source, samples, streams, blocks, gen, cv, outputs = [], [], [], [], [], [], []
    old_lens, lens, splits, modes = [], [], [], []
    old = 0
    state_indexes = []
    for i, (u, shape, na, vc, ac, b) in enumerate(
        zip(
            und_counts,
            vision_shapes,
            action_lengths,
            vision_conditions,
            action_conditions,
            geometry.block_sizes,
            strict=True,
        )
    ):
        t, h, w = shape
        nv = t * h * w
        if u < 1:
            raise ValueError("Action teacher forcing requires UND text")
        vb = prediction_block_ids(torch.arange(t), b).repeat_interleave(h * w).tolist()
        af = metadata[i].action_frame_ids.cpu()
        if len(af) != na or (af < 0).any():
            raise ValueError("Explicit action frame IDs must match action payload")
        ab = prediction_block_ids(af, b).tolist()
        vcond = torch.as_tensor(vc).bool().reshape(t, -1).all(1).repeat_interleave(h * w).tolist()
        if not all(vcond[:h * w]):
            raise ValueError("The independent first frame must be an observed visual condition")
        acond = torch.as_tensor(ac).bool().reshape(na, -1).all(1).tolist()
        if (af < 1).any():
            raise ValueError("Actions must belong to real intervals after the independent first frame")
        start = len(source)
        source.extend(range(old, old + u))
        samples.extend([i] * u)
        streams.extend([-1] * u)
        blocks.extend([-1] * u)
        for history in (True, False):
            for count, offset, bs, cs in ((nv, u, vb, vcond), (na, u + nv, ab, acond)):
                if history and offset == u + nv:
                    continue
                indexes = list(range(len(source), len(source) + count))
                gen.extend(indexes)
                if history:
                    cv.extend(indexes)
                else:
                    outputs.extend(indexes)
                source.extend(range(old + offset, old + offset + count))
                samples.extend([i] * count)
                streams.extend([0] * count if history else [2 if c else 1 for c in cs])
                blocks.extend(bs)
        if metadata is not None:
            m = metadata[i]
            ns = (t - 1 + b - 1) // b
            if m.block_size != b or len(m.states) != ns:
                raise ValueError("State geometry differs from the pre-packing geometry")
            indexes = list(range(len(source), len(source) + ns))
            state_indexes.extend(indexes)
            gen.extend(indexes)
            # Source indexes only seed positional fields; time is set from measurements below.
            source.extend(old + u + nv + int(j) for j in m.state_action_indexes)
            samples.extend([i] * ns)
            streams.extend([3] * ns)
            blocks.extend(range(ns))
        n = len(source) - start
        old_lens.append(u + nv + na)
        lens.append(n)
        splits.extend([u, n - u])
        modes.extend(["causal", "full"])
        old += u + nv + na

    def tensor(x):
        return torch.tensor(x, dtype=torch.long)

    return CausalActionLayout(
        geometry,
        tuple(old_lens),
        tuple(lens),
        tuple(splits),
        tuple(modes),
        tensor(source),
        tensor(samples),
        tensor(streams),
        tensor(blocks),
        tensor(gen),
        tensor(cv),
        tensor(outputs),
        state_indexes=tensor(state_indexes),
    )


def dense_action_mask(layout):
    """Independent vectorized reference; never used by production grouped_tnd."""
    qi = layout.gen_query_indexes[:, None]
    qb, kb = layout.block_ids[qi], layout.block_ids[None, :]
    qr, kr = layout.stream_ids[qi], layout.stream_ids[None, :]
    qs, ks = layout.sample_ids[qi], layout.sample_ids[None, :]
    hist = torch.tensor(layout.geometry.history_blocks, device=qb.device)[qs]
    past = (kr == 0) & (kb < qb) & (kb >= qb - hist)
    same = (kb == qb) & torch.where(qr == 0, kr == 0, torch.where(qr == 2, kr == 2, (kr == 1) | (kr == 2)))
    visible = (kr == -1) | past | same | ((qr != 0) & (kr == 3) & (kb == qb))
    # Clean visual history never absorbs state/action, including through earlier layers.
    # Each state is immutable through every layer: only UND and itself.
    visible = torch.where(qr == 3, (kr == -1) | ((kr == 3) & (kb == qb)), visible)
    return (qs == ks) & visible


def expand_action_sequence(packed, clean_vision_tokens, geometry):
    v, a = packed.vision, packed.action
    if v is None or a is None or packed.sound is not None:
        raise ValueError("Causal action jobs require video and action for every sample, without audio")
    n = len(packed.sample_lens)
    if len(v.tokens) != n or len(a.tokens) != n:
        raise ValueError("Causal action jobs require one video and one action payload per sample")
    nv = [int(torch.tensor(s).prod()) for s in v.token_shapes]
    na = [int(x.shape[0]) for x in a.tokens]
    und = [length - vi - ai for length, vi, ai in zip(packed.sample_lens, nv, na, strict=True)]
    layout = build_action_layout(
        und_counts=und,
        vision_shapes=v.token_shapes,
        action_lengths=na,
        vision_conditions=v.condition_mask,
        action_conditions=a.condition_mask,
        geometry=geometry,
        metadata=packed.causal_action_metadata,
    )
    # Reject interleaved/supertoken packs instead of silently using wrong source indices.
    off = 0
    vpos = apos = 0
    for u, vi, ai in zip(und, nv, na, strict=True):
        if v.sequence_indexes[vpos : vpos + vi].cpu().tolist() != list(
            range(off + u, off + u + vi)
        ) or a.sequence_indexes[apos : apos + ai].cpu().tolist() != list(range(off + u + vi, off + u + vi + ai)):
            raise ValueError("Expected contiguous UND / video / action input layout")
        off += u + vi + ai
        vpos += vi
        apos += ai
    mapping = torch.full((off,), -1, dtype=torch.long)
    targets = torch.cat([torch.where(layout.stream_ids == -1)[0], layout.noisy_output_indexes])
    mapping[layout.source_sequence_indexes[targets]] = targets

    def remap(mod):
        return replace(
            mod,
            sequence_indexes=mapping[mod.sequence_indexes],
            mse_loss_indexes=mapping[mod.mse_loss_indexes],
            spans=[replace(s, sequence_start=int(mapping[s.sequence_start])) for s in mod.spans],
        )

    positions = packed.position_ids[:, layout.source_sequence_indexes].clone()
    action_offset = state_offset = 0
    for m, length in zip(packed.causal_action_metadata, na, strict=True):
        original_action_indexes = a.sequence_indexes[action_offset : action_offset + length]
        step = packed.position_ids[0, original_action_indexes[1]] - packed.position_ids[0, original_action_indexes[0]]
        origin = packed.position_ids[0, original_action_indexes[0]] - step
        indexes = layout.state_indexes[state_offset : state_offset + len(m.states)]
        positions[0, indexes] = origin + m.state_frame_times.to(positions) * step
        action_offset += length
        state_offset += len(m.states)

    return replace(
        packed,
        vision=remap(v),
        action=remap(a),
        text_indexes=mapping[packed.text_indexes],
        ce_loss_indexes=None if packed.ce_loss_indexes is None else mapping[packed.ce_loss_indexes],
        position_ids=positions,
        sample_lens=list(layout.sample_lens),
        split_lens=list(layout.split_lens),
        attn_modes=list(layout.attn_modes),
        sequence_length=sum(layout.sample_lens),
        uses_single_timestep=False,
        teacher_forcing=CausalActionData(
            layout,
            list(clean_vision_tokens),
            None if packed.causal_action_metadata is None else [m.states for m in packed.causal_action_metadata],
        ),
    )
