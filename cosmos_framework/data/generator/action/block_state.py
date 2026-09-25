# SPDX-License-Identifier: OpenMDW-1.1
"""Shared 80D physical contract for causal action mid-training.

Adapters supply absolute measurements and absolute targets on an explicit
interval grid. Rotation slots contain the first two matrix columns, column-major.
Domain is metadata only; neither normalization nor network weights infer a domain.
"""

import hashlib
import json
from dataclasses import dataclass

import torch
import torch.nn.functional as F

WIDTH = 80


def prediction_block_ids(frame_ids, block_size):
    """L0 is visual prefix (-1); L1..LB form prediction block zero."""
    return (frame_ids - 1) // block_size


def prediction_block_span(block, block_size):
    start = 1 + block * block_size
    return start, start + block_size

ROTATIONS = (slice(3, 9), slice(37, 43))


def scatter_openwam_fields(fields, *, length):
    """Map independent source fields/masks, never mechanical arm joint vectors.

    Each value is (values[N,D], valid[N,D]); hand fields may have up to
    24 actual finger joints. Units, side and endpoints are adapter contracts.
    """
    slots = {
        "left_position": (0, 3),
        "left_rotation": (3, 9),
        "left_gripper": (9, 10),
        "left_hand_joints": (10, 34),
        "right_position": (34, 37),
        "right_rotation": (37, 43),
        "right_gripper": (43, 44),
        "right_hand_joints": (44, 68),
        "base_pose": (68, 71),
    }
    output = torch.zeros(length, WIDTH)
    mask = torch.zeros(length, WIDTH, dtype=torch.bool)
    for name, (value, valid) in fields.items():
        if name not in slots:
            raise ValueError(f"Unknown physical field {name}; arm joints require FK to an EEF")
        start, end = slots[name]
        value, valid = torch.as_tensor(value).float(), torch.as_tensor(valid).bool()
        if value.ndim != 2 or value.shape != valid.shape or len(value) != length:
            raise ValueError("Field values and validity must have identical [N,D] shapes")
        width = value.shape[1]
        if width != end - start and not (name.endswith("hand_joints") and 0 < width <= end - start):
            raise ValueError(f"Invalid width for {name}")
        output[:, start : start + width] = value.masked_fill(~valid, 0)
        mask[:, start : start + width] = valid
    return output, mask


def rotation_matrix(x):
    a, b = x[..., :3], x[..., 3:]
    if (a.norm(dim=-1) < 1e-7).any():
        raise ValueError("Invalid zero rot6d measurement")
    u = F.normalize(a, dim=-1)
    v = b - (u * b).sum(-1, keepdim=True) * u
    if (v.norm(dim=-1) < 1e-7).any():
        raise ValueError("Degenerate rot6d measurement")
    v = F.normalize(v, dim=-1)
    return torch.stack((u, v, torch.linalg.cross(u, v)), dim=-1)


def rotation_6d(matrix):
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), -1)


def relative_action(target, anchor, mask):
    """Translation in the declared frame; rotation in the anchor EEF frame."""
    result = target - anchor
    for slots in ROTATIONS:
        valid = mask[..., slots].all(-1)
        if (mask[..., slots].any(-1) != valid).any():
            raise ValueError("Rotation masks must cover all six components")
        if valid.any():
            result[..., slots][valid] = rotation_6d(
                rotation_matrix(anchor[..., slots][valid]).transpose(-1, -2)
                @ rotation_matrix(target[..., slots][valid])
            )
    return result.masked_fill(~mask, 0)


def absolute_action(delta, anchor, mask):
    result = delta + anchor
    for slots in ROTATIONS:
        valid = mask[..., slots].all(-1)
        if valid.any():
            result[..., slots][valid] = rotation_6d(
                rotation_matrix(anchor[..., slots][valid]) @ rotation_matrix(delta[..., slots][valid])
            )
    return result.masked_fill(~mask, 0)


@dataclass(frozen=True)
class SourceContract:
    source: str
    robot: str
    frame: str
    endpoint: str
    target_semantics: str
    data_version: str
    split: str = "train"
    units: str = "metres,radians,gripper_0_closed_1_open"
    side_mapping: str = "left=0:34,right=34:68"
    base_semantics: str = "absent"
    mapping_version: str = "openwam80-column-rot6d-v1"
    delta_definition: str = "block_anchor_position_difference_Rb_transpose_Rtarget"

    def __post_init__(self):
        if not all((self.source, self.robot, self.frame, self.endpoint, self.target_semantics, self.data_version)):
            raise ValueError("Source contract must declare provenance, frame, endpoint and target semantics")
        if self.base_semantics not in ("absent", "absolute_pose"):
            raise ValueError("Velocity-only base controls need an explicit adaptation; cannot subtract pose")

    def statistics_key(self, block_sizes, video_stride, chunk_length=32):
        payload = dict(
            self.__dict__,
            block_sizes=list(block_sizes),
            video_stride=video_stride,
            compression=4,
            chunk_length=chunk_length,
            sampling="uniform_windows_uniform_block_sizes",
            layout="independent_first_frame_no_action_padding_v1",
        )
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass
class Quantiles:
    low: torch.Tensor
    high: torch.Tensor
    valid: torch.Tensor

    def __post_init__(self):
        if any(x.shape != (WIDTH,) for x in (self.low, self.high, self.valid)):
            raise ValueError("Quantiles and validity must each have 80 channels")
        if not torch.isfinite(self.low).all() or not torch.isfinite(self.high).all() or (self.high < self.low).any():
            raise ValueError("Quantile bounds must be finite and ordered")
        if self.valid.dtype != torch.bool:
            raise ValueError("Statistics validity must be boolean")

    @classmethod
    def fit(cls, values, masks):
        values, masks = torch.cat(values).float(), torch.cat(masks).bool()
        lo, hi = torch.zeros(WIDTH), torch.zeros(WIDTH)
        valid = masks.any(0)
        for d in torch.where(valid)[0].tolist():
            lo[d], hi[d] = torch.quantile(values[masks[:, d], d], torch.tensor([0.01, 0.99]))
        return cls(lo, hi, valid)

    def normalize(self, values, mask):
        lo, hi, valid = (x.to(values.device) for x in (self.low, self.high, self.valid))
        if (mask & ~valid).any():
            raise ValueError("Statistics do not cover valid source channels")
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
    key: str
    state: Quantiles
    action: Quantiles

    @classmethod
    def load(cls, path, expected_key):
        with open(path) as f:
            data = json.load(f)
        if data["key"] != expected_key:
            raise ValueError("Statistics contract/geometry mismatch; recompute delta statistics on the training split")

        def read(name):
            q = data[name]
            return Quantiles(torch.tensor(q["low"]), torch.tensor(q["high"]), torch.tensor(q["valid"]).bool())

        return cls(data["key"], read("state"), read("action"))

    def save(self, path):
        with open(path, "w") as f:
            json.dump(dict(key=self.key, state=self.state.as_dict(), action=self.action.as_dict()), f, indent=2)


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
    contract: SourceContract
    state_clip_fraction: torch.Tensor | None = None
    action_clip_fraction: torch.Tensor | None = None
    statistics: BlockStatistics | None = None
    state_frame_times: torch.Tensor | None = None

    def __post_init__(self):
        if self.history_blocks < 1:
            raise ValueError("history_blocks must be positive")

    def to(self, device):
        from dataclasses import replace

        return replace(self, **{k: v.to(device) for k, v in self.__dict__.items() if isinstance(v, torch.Tensor)})


def build_block_sample(raw, *, block_size, history_blocks, video_stride, statistics=None):
    """Build geometry before packing. Each state_index precedes its action interval.

    Raw masks are independent [N,80] boolean masks. ``action_state_indexes``
    explicitly maps each target to its preceding measurement; timestamps are
    checked so an adapter cannot silently supply future measurements.
    """
    contract = raw["source_contract"]
    if not isinstance(contract, SourceContract):
        raise ValueError("A declared SourceContract is required")
    if block_size < 1 or history_blocks < 1 or video_stride < 1:
        raise ValueError("Invalid block geometry")
    state, target = raw["state_trajectory"].float(), raw["action_target"].float()
    sm, am = raw["state_mask"].bool(), raw["action_mask"].bool()
    if state.shape != sm.shape or target.shape != am.shape or state.shape[-1] != WIDTH or target.shape[-1] != WIDTH:
        raise ValueError("Measurements/targets and independent masks must have shape [N,80]")
    if sm[:, 71:].any() or am[:, 71:].any():
        raise ValueError("Reserved slots 71:80 must remain invalid")
    if contract.base_semantics == "absent" and (sm[:, 68:71].any() or am[:, 68:71].any()):
        raise ValueError("Base slots require an absolute-pose contract")
    if not torch.isfinite(state[sm]).all() or not torch.isfinite(target[am]).all():
        raise ValueError("Nonfinite valid measurement/target")
    n, g = len(target), 4 * video_stride
    if n < g or n % g or raw["video"].shape[1] != n + 1:
        raise ValueError("Expected synchronized source intervals divisible by compression * video_stride")
    mapping = torch.as_tensor(raw["action_state_indexes"], dtype=torch.long)
    if mapping.shape != (n,) or (mapping < 0).any() or (mapping >= len(state)).any():
        raise ValueError("Invalid action-to-measurement mapping")
    st, at = torch.as_tensor(raw["state_timestamps"]), torch.as_tensor(raw["action_timestamps"])
    if st.shape != (len(state),) or at.shape != (n,) or (st[mapping] > at).any():
        raise ValueError("State must be measured before or at the action interval start")
    if (st[1:] < st[:-1]).any() or (at[1:] <= at[:-1]).any():
        raise ValueError("Timestamps must be ordered")
    # 旧 Reader 仅提供 conditioning_fps；新 Reader 显式区分存储与训练 FPS。
    fps = float(raw.get("storage_fps", raw["conditioning_fps"]))
    if not fps > 0 or not torch.isfinite(torch.tensor(fps)):
        raise ValueError("Source fps must be finite and positive")
    if "conditioning_fps_action" in raw and float(raw["conditioning_fps_action"]) != float(raw["conditioning_fps"]):
        raise ValueError("Adapters must align video and action onto the same interval grid")
    if not torch.allclose(at[1:] - at[:-1], torch.full_like(at[1:], 1 / fps), atol=1e-5, rtol=1e-4):
        raise ValueError("Action timestamps must match the declared synchronized video interval grid")
    latent_frames = n // g + 1
    starts = torch.arange(0, latent_frames - 1, block_size) * g
    chosen = mapping[starts]
    anchors, masks = state[chosen], sm[chosen]
    if not masks.any(-1).all() or not am.any():
        raise ValueError("Causal action sources require measured block states and real action channels")
    for slots in ROTATIONS:
        valid = masks[:, slots].all(-1)
        if (masks[:, slots].any(-1) != valid).any():
            raise ValueError("State rotation validity must cover all six components")
        if valid.any():
            rotation_matrix(anchors[:, slots][valid])
    frame_ids = torch.arange(1, latent_frames).repeat_interleave(g)
    target_blocks = prediction_block_ids(frame_ids, block_size)
    anchor = anchors[target_blocks]
    if (am & ~masks[target_blocks]).any():
        raise ValueError("Every relative target channel requires a valid block-start measurement")
    delta = relative_action(target, anchor, am)
    state_clip = action_clip = None
    if statistics is not None:
        states, state_clip = statistics.state.normalize(anchors, masks)
        delta, action_clip = statistics.action.normalize(delta, am)
    else:
        states = anchors.masked_fill(~masks, 0)
    actions = delta
    action_mask = am
    metadata = BlockStateMetadata(
        block_size,
        history_blocks,
        frame_ids,
        action_mask,
        states,
        masks,
        starts,
        anchors,
        contract,
        state_clip,
        action_clip,
        statistics,
        (st[chosen] - at[0]) * fps,
    )
    result = dict(raw, action=actions, video=raw["video"][:, ::video_stride].contiguous())
    source_fps = torch.as_tensor(raw["conditioning_fps"]).float()
    result.update(conditioning_fps=source_fps / video_stride, conditioning_fps_action=source_fps)
    return result, metadata
