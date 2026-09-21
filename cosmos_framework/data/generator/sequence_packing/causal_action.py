# SPDX-License-Identifier: OpenMDW-1.1
"""Action-only teacher forcing: joint history and isolated current conditions.

Stream 0 is completed clean history, 1 current targets, 2 current conditions.
State uses block -1 and is persistent; it cannot absorb video/action information.
"""

from dataclasses import dataclass, fields, replace

import torch

from .teacher_forcing import TeacherForcingData, TeacherForcingGeometry, TeacherForcingLayout


@dataclass(frozen=True)
class CausalActionLayout(TeacherForcingLayout):
    clean_action_indexes: torch.Tensor
    includes_action: bool = True

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
    clean_action_tokens: list[torch.Tensor]

    def to_cuda(self):
        super().to_cuda()
        self.clean_action_tokens = [x.cuda() for x in self.clean_action_tokens]


def action_frame_ids(length: int, latent_frames: int, compression: int = 4):
    """Infer the uniform stride contract: [optional state, padding group, real groups].

    Each latent has compression * video_stride action slots. The remainder
    identifies optional state; at least two latents make this unambiguous.
    """
    if latent_frames < 2:
        raise ValueError("Causal action requires at least two video latents")
    per_latent, state = divmod(length, latent_frames)
    if state not in (0, 1) or per_latent < compression or per_latent % compression:
        raise ValueError("Expected uniform compression * video_stride action groups plus optional state")
    return torch.cat([torch.full((state,), -1), torch.arange(latent_frames).repeat_interleave(per_latent)]).long()


def action_prefix_length(length: int, latent_frames: int) -> int:
    """Number of state + synthetic slots excluded from executable actions."""
    return int((action_frame_ids(length, latent_frames) <= 0).sum())


def build_action_layout(
    *, und_counts, vision_shapes, action_lengths, vision_conditions, action_conditions, geometry: TeacherForcingGeometry
):
    source, samples, streams, blocks, gen, cv, ca, outputs = [], [], [], [], [], [], [], []
    old_lens, lens, splits, modes = [], [], [], []
    old = 0
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
        vb = (torch.arange(t) // b).repeat_interleave(h * w).tolist() # 获取vision block
        af = action_frame_ids(na, t)
        ab = torch.where(af < 0, -1, af // b).tolist()
        vcond = torch.as_tensor(vc).bool().reshape(t, -1).all(1).repeat_interleave(h * w).tolist()
        acond = torch.as_tensor(ac).bool().reshape(na, -1).all(1).tolist()
        if not all(acond[j] for j in range(na) if int(af[j]) <= 0):
            raise ValueError("Initial state and zero-group action slots must be conditioning")
        start = len(source)
        source.extend(range(old, old + u))
        samples.extend([i] * u)
        streams.extend([-1] * u)
        blocks.extend([-1] * u)
        for history in (True, False):
            for count, offset, bs, cs, clean in ((nv, u, vb, vcond, cv), (na, u + nv, ab, acond, ca)):
                indexes = list(range(len(source), len(source) + count))
                gen.extend(indexes)
                if history:
                    clean.extend(indexes)
                else:
                    outputs.extend(indexes)
                source.extend(range(old + offset, old + offset + count))
                samples.extend([i] * count)
                streams.extend([0] * count if history else [2 if c else 1 for c in cs])
                blocks.extend(bs)
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
        tensor(ca),
    )


def dense_action_mask(layout):
    """Independent vectorized reference; never used by production grouped_tnd."""
    qi = layout.gen_query_indexes[:, None]
    qb, kb = layout.block_ids[qi], layout.block_ids[None, :]
    qr, kr = layout.stream_ids[qi], layout.stream_ids[None, :]
    qs, ks = layout.sample_ids[qi], layout.sample_ids[None, :]
    hist = torch.tensor(layout.geometry.history_blocks, device=qb.device)[qs]
    past = (kr == 0) & (kb < qb) & (kb >= qb - hist)
    state = (kr == 0) & (kb == -1)
    same = (kb == qb) & torch.where(qr == 0, kr == 0, torch.where(qr == 2, kr == 2, kr >= 1))
    visible = (kr == -1) | state | past | same
    # State Q is immutable conditioning, including its duplicate current stream.
    visible = torch.where(qb == -1, (kr == -1) | ((kb == -1) & (kr == qr)), visible)
    return (qs == ks) & visible


def expand_action_sequence(packed, clean_vision_tokens, clean_action_tokens, geometry):
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

    return replace(
        packed,
        vision=remap(v),
        action=remap(a),
        text_indexes=mapping[packed.text_indexes],
        ce_loss_indexes=None if packed.ce_loss_indexes is None else mapping[packed.ce_loss_indexes],
        position_ids=packed.position_ids[:, layout.source_sequence_indexes],
        sample_lens=list(layout.sample_lens),
        split_lens=list(layout.split_lens),
        attn_modes=list(layout.attn_modes),
        sequence_length=sum(layout.sample_lens),
        uses_single_timestep=False,
        teacher_forcing=CausalActionData(layout, list(clean_vision_tokens), list(clean_action_tokens)),
    )
