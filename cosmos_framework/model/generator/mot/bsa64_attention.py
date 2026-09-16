"""Experimental exact block-mask adapter, without dense mask or padded keys."""

from dataclasses import dataclass

import torch


@dataclass
class BSA64Plan:
    kv_indexes: torch.Tensor
    mask: torch.Tensor
    query_length: int
    key_length: int
    visible_density: float


def build_bsa64_plans(layout, num_heads, device):
    streams = layout.stream_ids.detach().cpu()
    blocks = layout.block_ids.detach().cpu()
    plans = []
    offset = 0
    for i, (length, history) in enumerate(zip(layout.sample_lens, layout.geometry.history_blocks, strict=True)):
        und = layout.split_lens[2 * i]
        gen = layout.split_lens[2 * i + 1]
        if gen % 128:
            raise ValueError(f"BSA64 requires clean/noisy length divisible by 64, GEN={gen}")
        ss = streams[offset : offset + length]
        bb = blocks[offset : offset + length]
        qs, qb = ss[und:], bb[und:]
        for x in (qs, qb):
            if not torch.equal(x.reshape(-1, 64), x[::64, None].expand(-1, 64)):
                raise ValueError("stream/geometry boundary crosses BSA64 grid")
        permutation = torch.cat((torch.arange(und, length), torch.arange(und)))
        ks = ss[permutation][::64][None, :]
        kb = bb[permutation][::64][None, :]
        qs, qb = qs[::64, None], qb[::64, None]
        clean = (ks == 0) & (kb >= qb - history) & (kb <= qb)
        noisy = ((ks == 0) & (kb >= qb - history) & (kb < qb)) | ((ks == 1) & (kb == qb))
        allowed = (ks == -1) | torch.where(qs == 0, clean, noisy)
        mask = allowed.to(torch.uint8)[None, None].expand(1, num_heads, -1, -1).contiguous().to(device)
        plans.append(BSA64Plan(permutation.to(device), mask, gen, length, allowed.float().mean().item()))
        offset += length
    if offset != len(streams):
        raise ValueError("BSA64 plans do not cover packed samples")
    return tuple(plans)


def bsa64_per_sample_attention(query, key, value, plans, scale=None):
    from block_sparse_attention import block_sparse_attention

    outputs = []
    qo = ko = 0
    for plan in plans:
        q = query[qo : qo + plan.query_length].transpose(0, 1).unsqueeze(0).contiguous()
        k = key[ko : ko + plan.key_length].index_select(0, plan.kv_indexes).transpose(0, 1).unsqueeze(0).contiguous()
        v = value[ko : ko + plan.key_length].index_select(0, plan.kv_indexes).transpose(0, 1).unsqueeze(0).contiguous()
        if q.shape[1] != plan.mask.shape[1]:
            raise ValueError("BSA64 head count mismatch")
        out = block_sparse_attention(
            q,
            k,
            v,
            plan.mask,
            block_shape=(64, 64),
            input_layout="BNSD",
            num_key_value_heads=k.shape[1],
            scale_value=scale or q.shape[-1] ** -0.5,
            inner_precise=0,
        )
        outputs.append(out.squeeze(0).transpose(0, 1))
        qo += plan.query_length
        ko += plan.key_length
    if qo != query.shape[0] or ko != key.shape[0]:
        raise ValueError("BSA64 sequence lengths mismatch")
    return torch.cat(outputs, dim=0)
