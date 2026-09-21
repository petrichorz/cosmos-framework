# SPDX-License-Identifier: OpenMDW-1.1
import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention

from cosmos_framework.data.generator.sequence_packing.causal_action import (
    action_frame_ids,
    build_action_layout,
    dense_action_mask,
)
from cosmos_framework.data.generator.sequence_packing.teacher_forcing import TeacherForcingGeometry
from cosmos_framework.model.generator.mot.teacher_forcing_tnd import build_tnd_plan, teacher_forcing_tnd_attention


def make_layout(mode, block=1, history=2, batch=1, stride=1):
    t = 5
    group = 4 * stride
    na = t * group + 1
    return build_action_layout(
        und_counts=[3] * batch,
        vision_shapes=[(t, 1, 2)] * batch,
        action_lengths=[na] * batch,
        vision_conditions=[torch.tensor([1] * t if mode == "id" else [1] + [0] * (t - 1))] * batch,
        action_conditions=[torch.tensor([1] * na if mode == "fd" else [1] * (group + 1) + [0] * (4 * group))] * batch,
        geometry=TeacherForcingGeometry((block,) * batch, (history,) * batch),
    )


@pytest.mark.parametrize("mode", ["fd", "id", "policy"])
@pytest.mark.parametrize("block,history", [(1, 1), (1, 64), (2, 2), (4, 1)])
@pytest.mark.parametrize("budget", [1, 128, 131072])
@pytest.mark.parametrize("stride", [1, 4])
def test_tnd_dense_and_gradients(mode, block, history, budget, stride):
    torch.manual_seed(17)
    layout = make_layout(mode, block, history, 2, stride=stride)
    plan = build_tnd_plan(layout, device="cpu", max_kv_tokens=budget)
    mask = dense_action_mask(layout)
    q = torch.randn(plan.num_queries, 2, 8, dtype=torch.float64, requires_grad=True)
    k = torch.randn(plan.num_keys, 2, 8, dtype=torch.float64, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    ref = scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), attn_mask=mask
    ).transpose(0, 1)
    out = teacher_forcing_tnd_attention(q, k, v, plan)
    torch.testing.assert_close(out, ref, atol=1e-12, rtol=1e-12)
    grad = torch.randn_like(out)
    actual = torch.autograd.grad(out, (q, k, v), grad, retain_graph=True)
    expected = torch.autograd.grad(ref, (q, k, v), grad)
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, atol=1e-11, rtol=1e-11)
    assert torch.equal(torch.cat([c.query_indexes for c in plan.chunks]), torch.arange(plan.num_queries))


@pytest.mark.parametrize("mode", ["fd", "id", "policy"])
def test_multilayer_no_target_or_future_leak(mode):
    layout = make_layout(mode, block=1)
    mask = dense_action_mask(layout)
    n = layout.stream_ids.numel()
    # Boolean transitive dependency includes residual paths. UND sees UND only.
    dependencies = torch.eye(n, dtype=torch.bool)
    adjacency = torch.eye(n, dtype=torch.bool)
    adjacency[layout.gen_query_indexes] |= mask
    for _ in range(4):
        dependencies = (adjacency.float() @ dependencies.float()) > 0
    for q in layout.gen_query_indexes:
        b, r = int(layout.block_ids[q]), int(layout.stream_ids[q])
        assert not dependencies[q, layout.block_ids > b].any()
        if r in (1, 2) and b >= 0:
            assert not dependencies[q, (layout.stream_ids == 0) & (layout.block_ids == b)].any()
    # Task's legitimate current conditions remain visible to predicted tokens.
    for b in range(1, 5):
        targets = (layout.stream_ids == 1) & (layout.block_ids == b)
        conditions = (layout.stream_ids == 2) & (layout.block_ids == b)
        if mode != "policy":
            assert dependencies[targets][:, conditions].all()


def test_action_time_contract():
    ids = action_frame_ids(37, 9)
    assert ids.tolist() == [-1] + sum(([i] * 4 for i in range(9)), [])
    with pytest.raises(ValueError):
        action_frame_ids(33, 9)


def test_mixed_task_pack_isolation():
    layout = build_action_layout(
        und_counts=[2, 3, 4],
        vision_shapes=[(5, 1, 2)] * 3,
        action_lengths=[21] * 3,
        vision_conditions=[torch.tensor(x) for x in ([1, 0, 0, 0, 0], [1] * 5, [1, 0, 0, 0, 0])],
        action_conditions=[torch.tensor(x) for x in ([1] * 21, [1] * 5 + [0] * 16, [1] * 5 + [0] * 16)],
        geometry=TeacherForcingGeometry((1, 2, 4), (1, 3, 8)),
    )
    plan = build_tnd_plan(layout, device="cpu", max_kv_tokens=128)
    mask = dense_action_mask(layout)
    assert not mask[layout.sample_ids[layout.gen_query_indexes, None] != layout.sample_ids[None, :]].any()
    torch.manual_seed(23)
    q = torch.randn(plan.num_queries, 2, 8, dtype=torch.float64, requires_grad=True)
    k = torch.randn(plan.num_keys, 2, 8, dtype=torch.float64, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    out = teacher_forcing_tnd_attention(q, k, v, plan)
    ref = scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), attn_mask=mask
    ).transpose(0, 1)
    torch.testing.assert_close(out, ref, atol=1e-12, rtol=1e-12)
    grad = torch.randn_like(out)
    for a, e in zip(torch.autograd.grad(out, (q, k, v), grad), torch.autograd.grad(ref, (q, k, v), grad), strict=True):
        torch.testing.assert_close(a, e, atol=1e-11, rtol=1e-11)
