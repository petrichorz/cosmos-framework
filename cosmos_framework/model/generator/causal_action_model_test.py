# SPDX-License-Identifier: OpenMDW-1.1
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.action.datasets.causal_action_sft_dataset import (
    pad_causal_actions,
    validate_joint_weights,
)
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.data.generator.sequence_packing.causal_action import expand_action_sequence
from cosmos_framework.data.generator.sequence_packing.teacher_forcing import TeacherForcingGeometry
from cosmos_framework.model.generator.omni_mot_causal_action_model import (
    OmniMoTCausalActionModel,
    aligned_action_schedule,
)
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean


def harness():
    m = OmniMoTCausalActionModel.__new__(OmniMoTCausalActionModel)
    torch.nn.Module.__init__(m)
    m.config = SimpleNamespace(
        video_temporal_causal=False,
        max_action_dim=8,
        diffusion_expert_config=SimpleNamespace(
            patch_spatial=1,
            unified_3d_mrope_reset_spatial_ids=True,
            unified_3d_mrope_temporal_modality_margin=0,
            enable_fps_modulation=True,
            base_fps=24,
            sound_base_temporal_compression_factor=None,
            vision_temporal_position_mode="latent_index",
        ),
    )
    m.tokenizer_vision_gen = SimpleNamespace(temporal_compression_factor=4)
    m.llm_special_tokens = dict(eos_token_id=1, start_of_generation=2, end_of_generation=3)
    return m


@pytest.mark.parametrize("stride", [1, 2, 4, 8])
@pytest.mark.parametrize("state", [False, True])
@pytest.mark.parametrize("mode", ["fd", "id", "policy"])
@pytest.mark.parametrize("fps_modulation", [False, True])
def test_pack_and_schedule(state, mode, fps_modulation, stride):
    group = 4 * stride
    t = 5
    na = t * group + int(state)
    data = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[torch.randn(1, 2, t, 2, 2)],
        x0_tokens_action=[torch.randn(na, 8)],
        fps_vision=torch.tensor([15.0 / stride]),
        fps_action=torch.tensor([15.0]),
        action_domain_id=[torch.tensor(0)],
        raw_action_dim=[torch.tensor(8)],
    )
    plan = SequencePlan(
        has_text=True,
        has_vision=True,
        has_action=True,
        condition_frame_indexes_vision=list(range(t)) if mode == "id" else [0],
        condition_frame_indexes_action=list(range(na)) if mode == "fd" else list(range(group + int(state))),
        action_start_frame_offset=1 - group - int(state),
    )
    ts = torch.tensor([[100.0, 200.0, 300.0, 400.0, 500.0]])
    model = harness()
    model.config.diffusion_expert_config.enable_fps_modulation = fps_modulation
    p = model._pack_input_sequence([plan], [[9, 10]], data, ts)
    a = aligned_action_schedule(ts, p)
    assert a.shape == (1, na)
    torch.testing.assert_close(a[0, -group:], ts[0, -1].expand(group))
    assert p.action.timesteps.numel() == (0 if mode == "fd" else 4 * group)
    if mode != "fd":
        torch.testing.assert_close(p.action.timesteps, ts[0, 1:].repeat_interleave(group))
    pos = p.position_ids[0, p.action.sequence_indexes]
    torch.testing.assert_close(pos[: group + int(state)], pos[0].expand(group + int(state)))
    assert float(pos[group + int(state)] - pos[0]) == pytest.approx(0.4 if fps_modulation else 1 / group, abs=1e-6)
    expanded = expand_action_sequence(
        p, data.x0_tokens_vision, data.x0_tokens_action, TeacherForcingGeometry((2,), (2,))
    )
    assert expanded.sequence_length == p.sequence_length + 20 + na
    torch.testing.assert_close(
        expanded.position_ids[:, expanded.vision.sequence_indexes], p.position_ids[:, p.vision.sequence_indexes]
    )
    assert expanded.action.mse_loss_indexes.numel() == p.action.mse_loss_indexes.numel()


def test_weights_and_padding():
    assert validate_joint_weights() == (1.0, 1.0, 1.0)
    assert validate_joint_weights(dict(forward_dynamics=0, inverse_dynamics=2, policy=1)) == (0.0, 2.0, 1.0)
    for bad in (
        {},
        dict(forward_dynamics=0, inverse_dynamics=0, policy=0),
        dict(forward_dynamics=float("nan"), inverse_dynamics=1, policy=1),
    ):
        with pytest.raises(ValueError):
            validate_joint_weights(bad)
    raw = dict(video=torch.zeros(3, 33, 2, 2), action=torch.arange(33 * 8).reshape(33, 8))
    x = pad_causal_actions(raw, use_state=True)["action"]
    torch.testing.assert_close(x[0], raw["action"][0])
    assert not x[1:5].any()
    torch.testing.assert_close(x[5:], raw["action"][1:])


def tiny_model(device="cpu", dtype=torch.float32):
    from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler
    from cosmos_framework.model.generator.mot.causal_action_network import CausalActionNetwork
    from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetworkConfig
    from cosmos_framework.model.generator.mot.unified_mot import Qwen3VLMoTConfig, Qwen3VLTextForCausalLM

    m = harness()
    vlm = Qwen3VLMoTConfig(
        {
            "text_config": dict(
                vocab_size=256,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                rope_scaling=dict(mrope_interleaved=True, mrope_section=[2, 3, 3], rope_type="default"),
            )
        },
        qk_norm_for_text=False,
    )
    vlm.use_und_k_norm_for_gen = True
    cfg = Cosmos3VFMNetworkConfig(
        vlm_config=vlm,
        latent_patch_size=1,
        latent_channel_size=2,
        max_latent_h=2,
        max_latent_w=2,
        max_latent_t=32,
        joint_attn_implementation="teacher_forcing",
        action_gen=True,
        action_dim=8,
        num_embodiment_domains=1,
        teacher_forcing_dense_mode="grouped_tnd",
    )
    m.net = CausalActionNetwork(Qwen3VLTextForCausalLM(vlm), cfg).float().eval()
    for name, p in m.net.named_parameters():
        with torch.no_grad():
            if p.ndim >= 2:
                torch.nn.init.normal_(p, std=0.05)
            elif "norm" in name and name.endswith("weight"):
                p.fill_(1)
            else:
                p.zero_()
    freq = m.net.language_model.model.rotary_emb.inv_freq.clone()
    m.net = m.net.to(device=device, dtype=dtype)
    m.net.language_model.model.rotary_emb.inv_freq = freq.to(device)
    m.net.time_embedder.float()
    m.precision = dtype
    m.tensor_kwargs = dict(device=device, dtype=dtype)
    m.tensor_kwargs_fp32 = dict(device=device, dtype=torch.float32)
    m.config.teacher_forcing_dense_mode = "grouped_tnd"
    m.config.action_gen = True
    m.config.sound_gen = False
    m.config.rectified_flow_inference_config = SimpleNamespace(scheduler_type="unipc")
    m.sampler = UniPCSampler(tensor_kwargs=m.tensor_kwargs_fp32)
    return m


def sample_inputs(model, mode, block=2, stride=1):
    t = 5
    group = 4 * stride
    na = t * group + 1
    v = torch.randn(1, 2, t, 2, 2, device=model.tensor_kwargs["device"])
    a = torch.randn(na, 8, device=v.device)
    a[1 : 1 + group] = 0
    data = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[v],
        x0_tokens_action=[a],
        fps_vision=torch.tensor([15.0 / stride]),
        fps_action=torch.tensor([15.0]),
        action_domain_id=[torch.tensor(0)],
        raw_action_dim=[torch.tensor(8)],
    )
    plan = SequencePlan(
        has_text=True,
        has_vision=True,
        has_action=True,
        condition_frame_indexes_vision=list(range(t)) if mode == "id" else [0],
        condition_frame_indexes_action=list(range(na)) if mode == "fd" else list(range(group + 1)),
        action_start_frame_offset=-group,
    )
    packed = model._pack_input_sequence([plan], [[9, 10]], data, torch.zeros(1, 1))
    vm = packed.vision.condition_mask[0].to(v.device).expand_as(v)
    am = packed.action.condition_mask[0].to(v.device).expand_as(a)
    cm = torch.cat([vm.flatten(), am.flatten()]) if mode != "fd" else vm.flatten()
    ref = torch.cat([v.flatten(), a.flatten()]) if mode != "fd" else v.flatten()
    noise = torch.randn_like(ref) * (1 - cm) + ref * cm
    return dict(
        net=None,
        sampler=None,
        guidance=1.0,
        guidance_interval=None,
        velocity_postprocess_builder=None,
        num_steps=3,
        shift=1.0,
        sigma_max=80.0,
        skip_text_tokens_for_cfg=False,
        normalize_cfg=False,
        sequence_plans=[plan],
        gen_data_clean=data,
        cond_tokens=[[9, 10]],
        uncond_tokens=[[11]],
        initial_noise=[noise],
        condition_reference=[ref],
        condition_mask=[cm],
        has_noisy_actions=mode != "fd",
        causal_num_blocks=None,
        causal_block_size=block,
        causal_history_blocks=2,
    )


@pytest.mark.parametrize("mode", ["fd", "id", "policy"])
@pytest.mark.parametrize("guidance", [1.0, 3.0])
@pytest.mark.parametrize("stride", [1, 4])
def test_real_network_cache_and_conditions(mode, guidance, stride, monkeypatch):
    from cosmos_framework.data.generator.sequence_packing import PackedSequence
    from cosmos_framework.model.attention.frontend import attention

    def _cpu_attention(*args, backend=None, **kwargs):
        return attention(*args, backend="sdpa", **kwargs)

    monkeypatch.setattr(PackedSequence, "to_cuda", lambda self: None)
    monkeypatch.setattr("cosmos_framework.model.generator.mot.attention.attention", _cpu_attention)
    torch.manual_seed(43)
    m = tiny_model()
    args = sample_inputs(m, mode, stride=stride)
    args["guidance"] = guidance
    with torch.no_grad():
        full = m._generate_causal_inference_from_prepared(**args, causal_use_kv_cache=False)
        cached = m._generate_causal_inference_from_prepared(**args, causal_use_kv_cache=True)
    for key in ("vision", "action"):
        torch.testing.assert_close(full[key][0], cached[key][0], atol=2e-5, rtol=2e-5)
    original = args["gen_data_clean"]
    torch.testing.assert_close(
        cached["action"][0][: 4 * stride + 1], original.x0_tokens_action[0][: 4 * stride + 1], atol=0, rtol=0
    )
    torch.testing.assert_close(cached["vision"][0][:, :, :1], original.x0_tokens_vision[0][:, :, :1], atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["fd", "id", "policy"])
def test_network_dense_tnd_and_no_clean_target_leak(mode, monkeypatch):
    from cosmos_framework.model.attention.frontend import attention

    def cpu_attention(*args, backend=None, **kwargs):
        return attention(*args, backend="sdpa", **kwargs)

    monkeypatch.setattr("cosmos_framework.model.generator.mot.attention.attention", cpu_attention)
    torch.manual_seed(97)
    m = tiny_model()
    args = sample_inputs(m, mode, block=1)
    data = args["gen_data_clean"]
    original = m._pack_input_sequence(args["sequence_plans"], args["cond_tokens"], data, torch.full((1, 5), 500.0))
    packed = expand_action_sequence(
        original,
        [data.x0_tokens_vision[0].clone()],
        [data.x0_tokens_action[0].clone()],
        TeacherForcingGeometry((1,), (2,)),
    )
    for mod in (packed.vision, packed.action):
        mod.tokens = [torch.randn_like(x) * (1 - c) + x * c for x, c in zip(mod.tokens, mod.condition_mask)]
    parameters = [p for p in m.net.parameters() if p.requires_grad]
    outputs = []
    gradients = []
    for backend in ("grouped_tnd", "global"):
        m.net.config.teacher_forcing_dense_mode = backend
        out = m.net(packed_seq=packed)
        outputs.append(out)
        loss = sum(x.square().sum() for key in ("preds_vision", "preds_action") for x in out[key])
        gradients.append(torch.autograd.grad(loss, parameters, allow_unused=True))
    for key in ("preds_vision", "preds_action"):
        torch.testing.assert_close(outputs[0][key][0], outputs[1][key][0], atol=2e-5, rtol=2e-5)
    for a, b in zip(*gradients, strict=True):
        if a is None or b is None:
            assert a is b
        else:
            torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
    # Current target history copies must never influence current prediction.
    packed.teacher_forcing.clean_vision_tokens[0][:, :, 1] += 10
    packed.teacher_forcing.clean_action_tokens[0][5:9] += 10
    perturbed = m.net(packed_seq=packed)
    if mode != "id":
        torch.testing.assert_close(
            outputs[1]["preds_vision"][0][..., 1, :, :], perturbed["preds_vision"][0][..., 1, :, :], atol=0, rtol=0
        )
    if mode != "fd":
        torch.testing.assert_close(
            outputs[1]["preds_action"][0][5:9], perturbed["preds_action"][0][5:9], atol=0, rtol=0
        )


def test_empty_modality_keeps_loss_accumulator_fp32():
    model = harness()
    model.tensor_kwargs_fp32 = dict(device="cpu", dtype=torch.float32)
    pred = torch.ones(1, 2, 1, 1, dtype=torch.bfloat16, requires_grad=True)
    loss, _ = model._compute_flow_matching_loss(
        [pred], [torch.zeros_like(pred)], [torch.ones(2, 1, 1)], torch.ones(1, 2), False, None, normalize_by_active=True
    )
    assert loss.dtype == torch.float32
    total = loss + torch.tensor(2.6387458) * 10
    assert float(total) == pytest.approx(26.387458, abs=1e-5)
    total.backward()
    assert not pred.grad.any()


@pytest.mark.parametrize("stride", [1, 2, 4, 8])
@pytest.mark.parametrize("state", [False, True])
def test_stride_preserves_actions_and_timestamps(stride, state):
    from cosmos_framework.data.generator.sequence_packing.causal_action import action_frame_ids, action_prefix_length

    video = torch.arange(33).reshape(1, 33, 1, 1)
    actions = torch.arange((32 + int(state)) * 8).reshape(-1, 8)
    raw = dict(video=video, action=actions, conditioning_fps=torch.tensor(15))
    result = pad_causal_actions(raw, use_state=state, video_stride=stride)
    torch.testing.assert_close(result["video"], video[:, ::stride])
    assert result["conditioning_fps"].item() == 15 / stride
    assert result["conditioning_fps_action"].item() == 15
    latent_frames = 1 + 32 // (4 * stride)
    prefix = action_prefix_length(len(result["action"]), latent_frames)
    assert prefix == 4 * stride + int(state)
    assert not result["action"][int(state) : prefix].any()
    torch.testing.assert_close(result["action"][prefix:], actions[int(state) :])
    ids = action_frame_ids(len(result["action"]), latent_frames)
    assert (ids == 1).sum() == 4 * stride
    torch.testing.assert_close(raw["action"], actions)


@pytest.mark.parametrize("stride", [0, -1, 1.5, True, 3, 16])
def test_invalid_stride_does_not_truncate(stride):
    raw = dict(video=torch.zeros(3, 33, 2, 2), action=torch.zeros(32, 8))
    with pytest.raises(ValueError):
        pad_causal_actions(raw, use_state=False, video_stride=stride)


@pytest.mark.parametrize("mode", ["policy", "inverse_dynamics", "forward_dynamics"])
@pytest.mark.parametrize("state", [False, True])
@pytest.mark.parametrize("stride", [1, 2, 4, 8])
def test_stride_dataset_conditions_and_export(mode, state, stride):
    from cosmos_framework.data.generator.action.datasets.causal_action_sft_dataset import CausalActionSFTDataset
    from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline
    from cosmos_framework.inference.causal_action.outputs import real_actions

    class Reader:
        def __getitem__(self, idx):
            return dict(
                video=torch.zeros(3, 33, 32, 32, dtype=torch.uint8),
                action=torch.arange((32 + int(state)) * 8).reshape(-1, 8).float(),
                conditioning_fps=torch.tensor(15),
                domain_id=torch.tensor(0),
                ai_caption="move the robot",
                mode="policy",
            )

        def get_action_normalizer(self):
            return None

    base = SimpleNamespace(
        _dataset=Reader(),
        _resolution="256",
        _transform=ActionTransformPipeline(
            max_action_dim=8,
            append_viewpoint_info=False,
            append_duration_fps_timestamps=False,
            append_resolution_info=False,
        ),
    )
    dataset = CausalActionSFTDataset(base, mode=mode, use_state=state, video_stride=stride)
    item = dataset[0]
    prefix = 4 * stride + int(state)
    expected = len(item["action"]) if mode == "forward_dynamics" else prefix
    assert item["sequence_plan"].condition_frame_indexes_action == list(range(expected))
    assert item["sequence_plan"].action_start_frame_offset == 1 - prefix
    result = dict(vision=[torch.zeros(1, 2, 1 + 8 // stride, 2, 2)], action=[item["action"]])
    batch = {"action_processing_record": [item["action_processing_record"]]}
    exported = real_actions(result, batch)[0]
    torch.testing.assert_close(exported, base._dataset[0]["action"][int(state) :])


def test_stride_requires_synchronized_source_and_valid_window():
    from cosmos_framework.data.generator.action.datasets.causal_action_sft_dataset import (
        get_causal_action_droid_sft_dataset,
    )

    raw = dict(
        video=torch.zeros(3, 33, 2, 2),
        action=torch.zeros(32, 8),
        conditioning_fps=torch.tensor(15),
        conditioning_fps_action=torch.tensor(30),
    )
    with pytest.raises(ValueError, match="same-rate"):
        pad_causal_actions(raw, use_state=False, video_stride=2)
    for stride in [0, True, 3, 16]:
        with pytest.raises(ValueError, match="video_stride"):
            get_causal_action_droid_sft_dataset(root="/not-opened", video_stride=stride)
