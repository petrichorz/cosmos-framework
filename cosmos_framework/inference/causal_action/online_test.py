# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.action.action_state_template import ActionStateTemplate55
from cosmos_framework.data.generator.action.block_state import BlockStatistics, Quantiles
from cosmos_framework.data.generator.action.causal_block_geometry import CausalBlockGeometry
from cosmos_framework.data.generator.sequence_packing.runtime import from_mode_splits, init_sequence_pack
from cosmos_framework.inference.causal_action import streaming_cache
from cosmos_framework.inference.causal_action.policy import CAMERA_KEYS, CausalPolicy
from cosmos_framework.inference.causal_action.streaming_cache import ObservedVideoCache
from cosmos_framework.inference.causal_action.wire import packb, unpackb


class FakeEngine:
    def __init__(self):
        self.template = ActionStateTemplate55()
        self.geometry = CausalBlockGeometry(temporal_compression_factor=4)
        self.execute_steps = 16
        self.steps = 10
        self.seed = 42
        self.fps = 30.0
        mask = torch.zeros(55, dtype=torch.bool)
        mask[:6] = True
        mask[7:13] = True
        mask[28:30] = True
        self.mask = mask
        q = Quantiles(torch.zeros(55), torch.ones(55), mask)
        self.statistics = BlockStatistics(
            q, q, dict(sources=[dict(state_mask=mask.tolist(), action_mask=mask.tolist())])
        )
        self.calls = []
        self.reset()

    def reset(self):
        self.frames = []

    def observe(self, cameras, **kwargs):
        self.frames.append(cameras)
        self.calls.append(kwargs)

    def predict(self, **kwargs):
        out = np.zeros((32, 55), dtype=np.float32)
        out[:, 0] = kwargs["state"][0].item()
        return dict(action=out, video=np.zeros((8, 4, 6, 3), dtype=np.uint8), video_latents=np.zeros((1, 2, 2, 1, 1)))


def request(engine, count=1, value=0):
    state = dict(
        joint_action=dict(left_arm=[value, 0, 0, 0, 0, 0], right_arm=[0] * 6, left_gripper=0, right_gripper=0),
        endpose=dict(left_endpose=[0, 0, 0, 0, 0, 0, 1], right_endpose=[0, 0, 0, 0, 0, 0, 1]),
    )
    return dict(
        obs={key: np.full((count, 4, 6, 3), value, dtype=np.uint8) for key in CAMERA_KEYS},
        state=state,
    )


def reset(policy):
    policy.infer(
        dict(
            reset=True,
            prompt="Pick up the cube.",
            control_mode="joint",
        )
    )


def test_reset_and_half_block_feedback_use_fresh_measured_state():
    engine = FakeEngine()
    policy = CausalPolicy(engine)
    reset(policy)
    result = policy.infer(request(engine, value=3))
    assert result["action"]["joint"].shape == (32, 14) and result["execute_steps"] == 16
    assert result["action"]["joint"][0, 0] == 3
    assert result["action"]["eef"] is None
    result = policy.infer(dict(request(engine, 16, 9), executed_steps=16, observation_step=16))
    assert result["action"]["joint"][0, 0] == 9
    assert len(engine.frames) == 2
    assert result["revision"] == 2
    reset(policy)
    assert not engine.frames
    result = policy.infer(request(engine, value=5))
    assert result["revision"] == 1 and result["observation_step"] == 0


def test_bad_or_duplicate_feedback_does_not_advance_cache():
    engine = FakeEngine()
    policy = CausalPolicy(engine)
    reset(policy)
    policy.infer(request(engine))
    for bad, match in [
        (dict(request(engine, 16), executed_steps=32), "exactly 16"),
        (dict(request(engine, 16), observation_step=0), "out-of-order"),
        (request(engine, 1), "RGB uint8"),
        (dict(request(engine, 16), prompt="Different task"), "prompt"),
    ]:
        with pytest.raises(ValueError, match=match):
            policy.infer(bad)
    assert len(engine.frames) == 1


def test_model_failure_requires_reset(monkeypatch):
    engine = FakeEngine()
    policy = CausalPolicy(engine)
    reset(policy)

    def fail(**kwargs):
        raise RuntimeError("failed denoise")

    monkeypatch.setattr(engine, "predict", fail)
    with pytest.raises(RuntimeError, match="failed denoise"):
        policy.infer(request(engine))
    assert policy.needs_reset and not engine.frames
    with pytest.raises(ValueError, match="send reset"):
        policy.infer(request(engine))


def test_server_mask_survives_reset_and_cannot_be_overridden():
    engine = FakeEngine()
    policy = CausalPolicy(engine)
    reset(policy)
    with pytest.raises(ValueError, match="Unsupported control_mode"):
        policy.infer(dict(reset=True, prompt="Pick up the cube.", control_mode="eef"))
    with pytest.raises(ValueError, match="server"):
        policy.infer(dict(request(engine), state_mask=engine.mask.numpy()))
    response = policy.infer(request(engine, value=7))
    assert "action_unified" not in response and "action_mask" not in response
    reset(policy)
    torch.testing.assert_close(policy.state_mask, engine.mask)
    policy.infer(request(engine, value=2))
    assert engine.calls[-1]["state"][0] == 2


def test_native_websocket_roundtrip_and_reset(monkeypatch):
    import asyncio
    from contextlib import asynccontextmanager, suppress

    import websockets.asyncio.server

    from cosmos_framework.inference.causal_action.client import CausalPolicyClient
    from cosmos_framework.inference.causal_action.server import serve

    async def scenario():
        port = asyncio.get_running_loop().create_future()
        original = websockets.asyncio.server.serve

        @asynccontextmanager
        async def capture(*args, **kwargs):
            async with original(*args, **kwargs) as server:
                port.set_result(server.sockets[0].getsockname()[1])
                yield server

        monkeypatch.setattr(websockets.asyncio.server, "serve", capture)
        policy = CausalPolicy(FakeEngine(), return_unified_action=True)
        task = asyncio.create_task(serve(policy, host="127.0.0.1", port=0))
        try:
            uri = f"ws://127.0.0.1:{await asyncio.wait_for(port, 5)}"

            def client_session():
                with CausalPolicyClient(uri, control_mode="joint") as client:
                    assert client.metadata["protocol"] == "cosmos-causal-action-v2"
                    assert client.metadata["quaternion_order"] == "xyzw"
                    assert client.metadata["eef_pose_frame"] == "dataset"
                    assert client.metadata["seed"] == policy.engine.seed
                    assert client.metadata["available_control_modes"] == ["joint"]
                    for episode in range(2):
                        assert client.reset(prompt="Pick up the cube.")["reset"]
                        for revision, count in enumerate((1, 16)):
                            req = dict(
                                request(policy.engine, count, revision + episode), observation_step=revision * 16
                            )
                            result = client.infer(req)
                            assert client.actions(result).shape == (32, 14)
                            assert result["action"]["eef"] is None
                            assert result["action_unified"].shape == (32, 55)
                            assert "video" not in result
                            assert client.metadata["video_decode_mode"] == "whole_sequence"
                            assert client.metadata["prediction_video_fps"] == 16
                            assert result["revision"] == revision + 1
                            assert client.actions(result)[0, 0] == revision + episode
                with pytest.raises(ValueError, match="does not support"):
                    CausalPolicyClient(uri, control_mode="eef")

            await asyncio.wait_for(asyncio.to_thread(client_session), 10)
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    asyncio.run(scenario())


def test_wire_is_numpy_compatible_and_rejects_object_arrays():
    value = dict(obs=np.arange(24, dtype=np.uint8).reshape(2, 3, 4), state=np.float32(0.5))
    decoded = unpackb(packb(value))
    np.testing.assert_array_equal(decoded["obs"], value["obs"])
    assert decoded["state"] == value["state"]
    with pytest.raises(ValueError, match="dtype"):
        packb(np.array([object()]))


def cpu_attention(q, k, v, *, is_causal=False, **kwargs):
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=is_causal
    ).transpose(1, 2)


@pytest.mark.parametrize("normalized_keys", [False, True])
def test_streaming_cache_attention_matches_oracle_and_is_readonly(monkeypatch, normalized_keys):
    monkeypatch.setattr(streaming_cache, "attention", cpu_attention)
    torch.manual_seed(3)
    # UND, clean prefix, unused clean future, duplicate condition prefix, targets, state.
    roles = torch.tensor([-1, -1, 0, 0, 2, 1, 1, 3])
    layout = SimpleNamespace(
        stream_ids=roles,
        block_ids=torch.tensor([-1, -1, -1, 0, -1, 0, 0, 0]),
        gen_query_indexes=torch.arange(2, 8),
    )
    meta = init_sequence_pack([8], [2, 6], ["causal", "full"], torch.device("cpu"))
    meta["is_sharded"] = False
    q, k, v = [torch.randn(8, 2, 3) for _ in range(3)]
    packs = [from_mode_splits(x[:2], x[2:], meta) for x in (q, k, v)]
    gen_k = k + 0.25 if normalized_keys else k
    kwargs = {"packed_key_states_normalized": from_mode_splits(gen_k[:2], gen_k[2:], meta)} if normalized_keys else {}
    original = object()
    attn = SimpleNamespace(dispatch_attention_fn=original)
    net = SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)]))
    )
    cache = ObservedVideoCache(1, 2)

    with torch.no_grad():
        for frame in range(3):
            with cache.forward(net, append_frame=frame):
                attn.dispatch_attention_fn(*packs, SimpleNamespace(layout=layout), **kwargs)
        assert [x[0] for x in cache.layers[0].video] == [1, 2]
        saved = [(key.clone(), value.clone()) for _, key, value in cache.layers[0].video]
        with cache.forward(net):
            output, _ = attn.dispatch_attention_fn(*packs, SimpleNamespace(layout=layout), **kwargs)
        # Current targets see text, the two observed prefix copies, state, targets.
        keys = torch.tensor([0, 1, 2, 2, 7, 5, 6])
        expected = cpu_attention(q[None, 5:7], gen_k[None, keys], v[None, keys])[0].flatten(-2)
        torch.testing.assert_close(output["full_only_seq"][3:5], expected)
        state_keys = torch.tensor([0, 1, 7])
        expected_state = cpu_attention(q[None, 7:8], gen_k[None, state_keys], v[None, state_keys])[0].flatten(-2)
        torch.testing.assert_close(output["full_only_seq"][5:6], expected_state)
        for (_, key, value), (before_k, before_v) in zip(cache.layers[0].video, saved, strict=True):
            torch.testing.assert_close(key, before_k)
            torch.testing.assert_close(value, before_v)
        with pytest.raises(RuntimeError, match="abort"):
            with cache.forward(net, append_frame=3):
                attn.dispatch_attention_fn(*packs, SimpleNamespace(layout=layout), **kwargs)
                raise RuntimeError("abort")
        assert cache.layers[0].video[-1][0] == 2
        assert attn.dispatch_attention_fn is original
        cache.clear()
        with pytest.raises(RuntimeError, match="prefill"):
            with cache.forward(net):
                pass
