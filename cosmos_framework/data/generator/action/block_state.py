# SPDX-License-Identifier: OpenMDW-1.1
"""Template-driven fixed-block encoding and state/action quantile normalization."""

import json
from dataclasses import dataclass, field

import torch

from cosmos_framework.data.generator.action.action_state_template import ActionStateTemplate, TemplateSourceContract


def prediction_block_ids(frame_ids, block_size):
    """L0 is visual prefix (-1); L1..LB form prediction block zero."""
    return (frame_ids - 1) // block_size


def prediction_block_span(block, block_size):
    start = 1 + block * block_size
    return start, start + block_size


@dataclass
class Quantiles:
    low: torch.Tensor
    high: torch.Tensor
    valid: torch.Tensor

    def __post_init__(self):
        if (
            self.low.ndim != 1
            or not self.low.numel()
            or any(x.shape != self.low.shape for x in (self.high, self.valid))
        ):
            raise ValueError("Quantiles and validity must have the same nonempty [D] shape")
        if not torch.isfinite(self.low).all() or not torch.isfinite(self.high).all() or (self.high < self.low).any():
            raise ValueError("Quantile bounds must be finite and ordered")
        if self.valid.dtype != torch.bool:
            raise ValueError("Statistics validity must be boolean")

    @classmethod
    def fit(cls, values, masks):
        values, masks = torch.cat(values).float(), torch.cat(masks).bool()
        lo, hi = values.new_zeros(values.shape[-1]), values.new_zeros(values.shape[-1])
        valid = masks.any(0)
        for d in torch.where(valid)[0].tolist():
            lo[d], hi[d] = torch.quantile(values[masks[:, d], d], values.new_tensor([0.01, 0.99]))
        return cls(lo, hi, valid)

    def normalize(self, values, mask):
        lo, hi = self.low.to(values), self.high.to(values)
        span = hi - lo
        scaled = 2 * (values - lo) / span.clamp_min(1e-12) - 1
        clipped = mask & (span > 0) & (scaled.abs() > 1)
        out = scaled.clamp(-1, 1).masked_fill(~mask | (span == 0), 0)
        return out, clipped.sum(0) / mask.sum(0).clamp_min(1)

    def denormalize(self, values, mask):
        lo, hi = self.low.to(values), self.high.to(values)
        return ((values + 1) * 0.5 * (hi - lo) + lo).masked_fill(~mask, 0)

    def as_dict(self):
        return {k: getattr(self, k).tolist() for k in ("low", "high", "valid")}


@dataclass
class BlockStatistics:
    state: Quantiles
    action: Quantiles
    provenance: dict = field(default_factory=dict)

    def validate(self, width, state_mask, action_mask, *, allow_mock=False):
        """训练适配层首次加载时检查维度、mask 覆盖和模拟统计开关。"""
        if self.state.low.numel() != width or self.action.low.numel() != width:
            raise ValueError("Statistics width differs from the action template")
        for mask, quantiles in ((state_mask, self.state), (action_mask, self.action)):
            mask = torch.as_tensor(mask, dtype=torch.bool, device=quantiles.valid.device)
            if (mask & ~quantiles.valid).any():
                raise ValueError("Statistics do not cover valid source channels")
        if self.provenance.get("kind") == "mock" and not allow_mock:
            raise ValueError("Mock statistics require allow_mock_statistics=True")

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)

        def read(name):
            q = data[name]
            return Quantiles(torch.tensor(q["low"]), torch.tensor(q["high"]), torch.tensor(q["valid"]).bool())

        return cls(read("state"), read("action"), data.get("provenance", {}))

    def save(self, path):
        with open(path, "w") as f:
            json.dump(
                dict(state=self.state.as_dict(), action=self.action.as_dict(), provenance=self.provenance),
                f,
                indent=2,
            )


@dataclass
class BlockStateMetadata:
    block_size: int
    history_blocks: int
    action_frame_ids: torch.Tensor
    action_mask: torch.Tensor
    states: torch.Tensor
    state_mask: torch.Tensor
    state_action_indexes: torch.Tensor
    anchors: torch.Tensor
    contract: TemplateSourceContract
    state_clip_fraction: torch.Tensor | None = None
    action_clip_fraction: torch.Tensor | None = None
    statistics: BlockStatistics | None = None
    state_frame_times: torch.Tensor | None = None
    state_latent_indexes: torch.Tensor | None = None
    template: ActionStateTemplate | None = None

    def __post_init__(self):
        if self.history_blocks < 1:
            raise ValueError("history_blocks must be positive")

    def to(self, device):
        from dataclasses import replace

        return replace(self, **{k: v.to(device) for k, v in self.__dict__.items() if isinstance(v, torch.Tensor)})


def build_block_sample(raw, *, template, planner, history_blocks, statistics=None):
    """按固定 block 编码；调用方在首次加载时校验统计，统计采样与训练共用编码。"""
    # Reader 已完成目标时间对齐；这里只检查 T 个 action 与 T+1 个 state 能组成完整 block。
    geometry = planner.geometry
    state, target = raw["state_trajectory"].float(), raw["action_target"].float()
    geometry.validate_observation_frames(len(state))
    n = len(target)
    if n != len(state) - 1:
        raise ValueError("Expected T action targets and T+1 observations")
    # 两套 mask 独立；相对字段必须有有效 anchor，绝对字段不要求对应 state 有效。
    contract = raw["source_contract"]
    sm = template.validate_valid_mask(raw["state_mask"]).to(state.device)
    am = template.validate_valid_mask(raw["action_mask"]).to(target.device)
    template.validate_source_contract(contract, sm)
    template.validate_source_contract(contract, am)
    if not sm.any() or not am.any():
        raise ValueError("Block training requires valid state and action channels")
    if (template.required_anchor_mask(am) & ~sm).any():
        raise ValueError("Relative action channels require valid anchor state channels")
    state = template.sanitize(state, sm)
    target = template.sanitize(target, am)

    # 同一 geometry 决定 anchor、action 对应 latent 和 state 的视频边界。
    blocks = geometry.num_blocks(len(state))
    starts = torch.tensor([geometry.block_action_span(b)[0] for b in range(blocks)], device=state.device)
    anchors = state[starts]
    # 每个 block 内共用起始 state，下一个 block 切换 anchor。
    anchor_per_action = anchors.repeat_interleave(geometry.actions_per_block, dim=0)
    # 模板决定字段使用差值、相对旋转还是绝对目标。
    encoded = template.encode_action_delta(target, anchor_per_action, am, source_contract=contract)
    state_mask = sm.expand(blocks, -1)
    action_mask = am.expand(n, -1)
    states = anchors
    state_clip = action_clip = None
    # 先编码再分别归一化；anchors 保留原值，未传统计时可用于采集 q01/q99。
    if statistics is not None:
        states, state_clip = statistics.state.normalize(anchors, state_mask)
        encoded, action_clip = statistics.action.normalize(encoded, action_mask)
    # anchor 对应预测 block 前的视频边界 latent，默认得到 0、2、4……。
    latent_indexes = torch.tensor([geometry.block_latent_span(b)[0] - 1 for b in range(blocks)], device=state.device)
    metadata = BlockStateMetadata(
        block_size=geometry.latent_frames_per_block,
        history_blocks=history_blocks,
        # latent 0 是独立首帧，action 从 latent 1 开始映射。
        action_frame_ids=torch.arange(1, geometry.num_latent_frames(len(state)), device=state.device).repeat_interleave(
            geometry.actions_per_latent_frame
        ),
        action_mask=action_mask,
        states=states,
        state_mask=state_mask,
        state_action_indexes=starts,
        anchors=anchors,
        contract=contract,
        state_clip_fraction=state_clip,
        action_clip_fraction=action_clip,
        statistics=statistics,
        # C07a 再删除旧位置消费者；此处保留其原始时间步单位。
        state_frame_times=(raw["state_timestamps"][starts] - raw["action_timestamps"][0]) * raw["storage_fps"],
        state_latent_indexes=latent_indexes,
        template=template,
    )
    # 保留首帧并按 stride 抽视频帧；action 不抽帧，此处不执行 VAE 编码。
    video = raw.get("video")
    if video is not None:
        if video.ndim != 4 or video.shape[1] != n + 1:
            raise ValueError("Video must contain T+1 observations before subsampling")
        video = video[:, :: geometry.video_stride].contiguous()
    # 视频 FPS 随抽帧降低，action FPS 保持 Reader 提供的训练频率。
    fps = float(raw["conditioning_fps"])
    result = dict(
        raw,
        action=encoded,
        video=video,
        conditioning_fps=torch.tensor(geometry.video_fps(fps), dtype=torch.float32),
        conditioning_fps_action=torch.tensor(geometry.action_fps(fps), dtype=torch.float32),
    )
    return result, metadata
