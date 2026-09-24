# SPDX-License-Identifier: OpenMDW-1.1
"""Versioned physical layouts, independent of dataset I/O and normalization.

55D EEF quaternions use xyzw. Relative scalars are target minus block anchor;
Gripper and dexterous-hand targets remain absolute in both encoding and decoding.
rotation deltas are inverse(anchor) * target. Decoding composes anchor * delta.
Quaternion results are unit length with a deterministic sign (q and -q are
the same rotation). No angle wrapping or coordinate-frame conversion is implicit.
"""

import hashlib
import importlib
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Mapping

import torch


@dataclass(frozen=True)
class ValidMaskSpec:
    width: int
    dtype: str = "bool"
    true_means: str = "valid"
    alignment: str = "template_dimension"
    granularity: str = "dataset"
    version: str = "v1"


@dataclass(frozen=True)
class TemplateSourceContract:
    """Explicit declaration for already-template-mapped absolute quantities.

    ``units`` and ``scalar_semantics`` are keyed by field name. Scalars can
    only use block subtraction when declared ``absolute``. In particular,
    velocity commands must not be silently interpreted as mobile pose.
    This contract does not reuse the existing OpenWAM80 SourceContract.
    """

    template_id: str
    source: str
    frame: str
    # A source must state the physical EEF endpoint for each active side.
    endpoints: Mapping[str, str]
    units: Mapping[str, str]
    scalar_semantics: Mapping[str, str]
    target_semantics: str
    quaternion_order: str = "xyzw"
    data_version: str = "processed-v1"
    split: str = "train"
    fps: float = 0.0
    valid_dimensions: tuple[int, ...] = ()
    sampling_signature: str = ""

    def statistics_key(self, block_sizes, video_stride, chunk_length):
        """统计绑定来源、模板及实际采样几何，不能跨契约误用。"""
        payload = dict(
            contract=asdict(self),
            block_sizes=list(block_sizes),
            video_stride=video_stride,
            chunk_length=chunk_length,
            compression=4,
            layout="independent_first_frame_short_final_block_v1",
            sampling="uniform_windows_uniform_block_sizes",
        )
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class ActionStateTemplate(ABC):
    template_id: str
    width: int
    fields: Mapping[str, slice]
    rotation_groups: tuple[tuple[int, ...], ...]
    valid_mask_spec: ValidMaskSpec

    @property
    def action_names(self):
        """字段名来自模板，通用 reader 不再生成特定机器人的槽位名。"""
        names = [f"channel_{i}" for i in range(self.width)]
        for name, slots in self.fields.items():
            for i, slot in enumerate(range(*slots.indices(self.width))):
                names[slot] = f"{name}_{i}"
        return names

    @abstractmethod
    def sanitize(self, values, valid_mask): ...

    @abstractmethod
    def validate_source_contract(self, contract, valid_mask): ...

    def source_contract(self, profile, *, source, info, target_semantics):
        """已有产物缺少契约时，由所选模板明确解释来源字段。"""
        raise ValueError("This template requires an explicit source contract")

    def mock_quantiles(self, state_stats, delta_stats, valid_mask):
        """返回模板表示的模拟范围；具体字段映射只能由模板定义。"""
        raise ValueError("This template does not define an AgiBot mock statistics mapping")

    @abstractmethod
    def validate_valid_mask(self, valid_mask, *, template_id=None) -> torch.Tensor:
        """校验数据集共享 mask，返回布尔向量。"""
        ...

    @abstractmethod
    def validate(self, state, action, valid_mask, source_contract) -> None:
        """检查 state/action 数值、形状及来源契约是否符合模板。"""
        ...

    @abstractmethod
    def encode_action_delta(self, absolute_action, anchor_state, valid_mask, *, source_contract) -> torch.Tensor:
        """按模板逐字段策略编码目标；相对字段使用块起点 state。"""
        ...

    @abstractmethod
    def decode_action_delta(self, action_delta, anchor_state, valid_mask, *, source_contract) -> torch.Tensor:
        """使用编码时的同一 anchor，将相对 action 还原为绝对量。"""
        ...


class ActionStateTemplate55(ActionStateTemplate):
    """55D layout. Waist/head/mobile component meanings remain source-declared.

    No dataset masks or quantiles are stored on the template. A dataset owns
    one [55] mask, broadcast at runtime over any leading tensor dimensions.
    Invalid values are cleared by sanitize/encode/decode, including NaNs.
    """

    template_id = "unified55-xyzw-absolute-gripper-hand-v3"
    width = 55
    fields = MappingProxyType(
        {
            "left_arm_joint": slice(0, 7),
            "right_arm_joint": slice(7, 14),
            "left_eef": slice(14, 21),
            "right_eef": slice(21, 28),
            "left_gripper": slice(28, 29),
            "right_gripper": slice(29, 30),
            "left_hand": slice(30, 36),
            "right_hand": slice(36, 42),
            "waist": slice(42, 46),
            "head": slice(46, 48),
            "mobile": slice(48, 51),
            "reserved": slice(51, 55),
        }
    )
    # 左 EEF 四元数占 17:21，右 EEF 四元数占 24:28，均按 xyzw 排列；每组四维必须整体有效或无效。
    rotation_groups = (tuple(range(17, 21)), tuple(range(24, 28)))
    valid_mask_spec = ValidMaskSpec(width=55)
    absolute_fields = ("left_gripper", "right_gripper", "left_hand", "right_hand")

    def validate_valid_mask(self, valid_mask, *, template_id=None):
        """校验 55D mask、模板版本、保留位及四元数组完整性。"""
        if template_id is not None and template_id != self.template_id:
            raise ValueError(f"Expected template_id={self.template_id}, got {template_id}")
        # Preserve Python float precision before checking exact 0/1 values.
        mask = valid_mask if isinstance(valid_mask, torch.Tensor) else torch.as_tensor(valid_mask, dtype=torch.float64)
        if mask.shape != (self.width,):
            raise ValueError("valid_mask must be a dataset-level [55] vector")
        if not ((mask == 0) | (mask == 1)).all():
            raise ValueError("valid_mask must contain only bool or exact 0/1 values")
        mask = mask.bool()
        if mask[self.fields["reserved"]].any():
            raise ValueError("Reserved dimensions 51:55 must be invalid")
        for group in self.rotation_groups:
            enabled = mask[list(group)]
            if enabled.any() and not enabled.all():
                raise ValueError("A quaternion rotation group must be entirely valid or invalid")
        return mask

    def validate_source_contract(self, contract, valid_mask):
        """检查有效字段的单位、坐标系、端点及绝对量语义声明。"""
        mask = self.validate_valid_mask(valid_mask)
        if not isinstance(contract, TemplateSourceContract):
            raise ValueError("An explicit TemplateSourceContract is required")
        if contract.template_id != self.template_id or contract.quaternion_order != "xyzw":
            raise ValueError("Source template version/quaternion order mismatch")
        if not all((contract.source, contract.frame, contract.target_semantics)):
            raise ValueError("Source, frame and target semantics must be declared")
        for name, slots in self.fields.items():
            if not mask[slots].any():
                continue
            if not contract.units.get(name):
                raise ValueError(f"Declare units for active field {name}")
            if name.endswith("_eef"):
                if contract.units[name] != "metres,unit_quaternion":
                    raise ValueError("EEF units must be metres,unit_quaternion")
                if not contract.endpoints.get(name):
                    raise ValueError(f"Declare physical endpoint for {name}")
            if contract.scalar_semantics.get(name) != "absolute":
                raise ValueError(f"{name} must declare absolute semantics for block-anchor deltas")

    def sanitize(self, values, valid_mask):
        """返回无效维清零后的张量，并拒绝有效维中的非有限值或零四元数。"""
        mask = self.validate_valid_mask(valid_mask)
        if not isinstance(values, torch.Tensor) or not values.is_floating_point():
            raise ValueError("Template values must be floating-point tensors")
        if values.ndim < 1 or values.shape[-1] != self.width:
            raise ValueError("Template values must have last dimension 55")
        mask = mask.to(values.device)
        result = values.masked_fill(~mask, 0)
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite value in a valid template dimension")
        for group in self.rotation_groups:
            if mask[list(group)].all():
                q = result[..., list(group)]
                if (torch.linalg.vector_norm(q, dim=-1) < 1e-8).any():
                    raise ValueError("Active quaternion must be nonzero")
        return result

    def validate(self, state, action, valid_mask, source_contract):
        """校验来源与有效数值，并确认 state/action 可通过广播对齐。"""
        self.validate_source_contract(source_contract, valid_mask)
        state, action = self.sanitize(state, valid_mask), self.sanitize(action, valid_mask)
        try:
            torch.broadcast_shapes(state.shape, action.shape)
        except RuntimeError as exc:
            raise ValueError("State and action shapes are not broadcastable") from exc

    @staticmethod
    def _unit_quaternion(q):
        """归一化 xyzw 四元数并统一符号，消除 q 与 -q 的表示歧义。"""
        q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
        # w >= 0; at exactly 180 degrees use the largest xyz component.
        pivot = torch.where(q[..., 3:4] != 0, q[..., 3:4], q.gather(-1, q.abs().argmax(-1, keepdim=True)))
        return torch.where(pivot < 0, -q, q)

    @staticmethod
    def _multiply(a, b):
        """计算 xyzw 四元数乘积 a * b，保持旋转复合顺序。"""
        av, aw, bv, bw = a[..., :3], a[..., 3:], b[..., :3], b[..., 3:]
        return torch.cat(
            (aw * bv + bw * av + torch.linalg.cross(av, bv), aw * bw - (av * bv).sum(-1, keepdim=True)), -1
        )

    def _convert(self, values, anchor, valid_mask, source_contract, *, decode):
        """相对标量做加减，夹爪与灵巧手绝对保留，四元数复合，清零无效维。"""
        self.validate(anchor, values, valid_mask, source_contract)
        values = self.sanitize(values, valid_mask)
        anchor = self.sanitize(anchor, valid_mask)
        values, anchor = torch.broadcast_tensors(values, anchor)
        result = values + anchor if decode else values - anchor
        # 目标时间由读取选项决定；夹爪和灵巧手编解码均不加减 block anchor。
        for name in self.absolute_fields:
            result[..., self.fields[name]] = values[..., self.fields[name]]
        mask = self.validate_valid_mask(valid_mask).to(values.device)
        for group in self.rotation_groups:
            indexes = list(group)
            if mask[indexes].all():
                a, b = self._unit_quaternion(anchor[..., indexes]), self._unit_quaternion(values[..., indexes])
                if not decode:
                    a = torch.cat((-a[..., :3], a[..., 3:]), -1)
                result[..., indexes] = self._unit_quaternion(self._multiply(a, b))
        return result.masked_fill(~mask, 0)

    def encode_action_delta(self, absolute_action, anchor_state, valid_mask, *, source_contract):
        """编码块相对目标；旋转采用 inverse(anchor) * target。"""
        return self._convert(absolute_action, anchor_state, valid_mask, source_contract, decode=False)

    def decode_action_delta(self, action_delta, anchor_state, valid_mask, *, source_contract):
        """还原绝对目标；旋转采用 anchor * delta，输入应已反归一化。"""
        return self._convert(action_delta, anchor_state, valid_mask, source_contract, decode=True)

    def source_contract(self, profile, *, source, info, target_semantics):
        """声明已处理产物的来源语义；不在 reader 中解释物理槽位。"""
        units = {name: "source_native" for name in self.fields if name != "reserved"}
        for name in ("left_eef", "right_eef"):
            units[name] = "metres,unit_quaternion"
        if profile == "egosuite":
            custom = info["custom"]
            if "state_unified eef quats: xyzw" not in custom["coordinate_frames"]["quat_orders"]:
                raise ValueError("EgoSuite must declare unified EEF xyzw quaternions")
            frame = custom["coordinate_frames"]["torso"]
            endpoints = dict(left_eef="left_hand_wrist_joint_0", right_eef="right_hand_wrist_joint_0")
            units.update(left_gripper="unit_interval", right_gripper="unit_interval", head="radians")
        elif profile == "agibot":
            # 先沿用预处理表示；不擅自声明 flange/TCP 或转换物理坐标。
            frame = "processed_export_reference_frame"
            endpoints = dict(left_eef="exported_left_end", right_eef="exported_right_end")
            units.update(left_arm_joint="radians", right_arm_joint="radians")
        else:
            raise ValueError(f"Unknown source profile: {profile}")
        return TemplateSourceContract(
            template_id=self.template_id,
            source=source,
            frame=frame,
            endpoints=endpoints,
            units=units,
            scalar_semantics=dict.fromkeys(units, "absolute"),
            target_semantics=target_semantics,
        )

    def mock_quantiles(self, state_stats, delta_stats, valid_mask):
        """AgiBot 具名统计映射到本模板；未确认的旋转顺序不猜测。"""
        mask = self.validate_valid_mask(valid_mask)
        sl, sh = torch.full((self.width,), -1.0), torch.ones(self.width)
        al, ah = sl.clone(), sh.clone()
        origin = {name: "placeholder" for name in self.fields}
        # 所有数字切片均属于此模板，统计加载器只接收最终 D 维范围。
        mappings = {
            "left_arm_joint": ("observation.states.joint.position", slice(0, 7)),
            "right_arm_joint": ("observation.states.joint.position", slice(7, 14)),
            "left_gripper": ("observation.states.effector.position", slice(0, 1)),
            "right_gripper": ("observation.states.effector.position", slice(1, 2)),
            "head": ("observation.states.head.position", slice(0, 2)),
        }

        def bounds(record, slots, width):
            if not record or "q01" not in record or "q99" not in record:
                return None
            lo = torch.as_tensor(record["q01"], dtype=torch.float32).flatten()[slots]
            hi = torch.as_tensor(record["q99"], dtype=torch.float32).flatten()[slots]
            if lo.shape != (width,) or hi.shape != (width,):
                return None
            if not torch.isfinite(lo).all() or not torch.isfinite(hi).all() or (hi < lo).any():
                raise ValueError("Invalid AgiBot q01/q99 bounds")
            return lo, hi

        for name, (key, source_slots) in mappings.items():
            slots = self.fields[name]
            pair = bounds(state_stats.get(key), source_slots, slots.stop - slots.start)
            if pair is None:
                continue
            sl[slots], sh[slots] = pair
            if name in self.absolute_fields:
                al[slots], ah[slots] = pair
            else:
                span = pair[1] - pair[0]
                al[slots], ah[slots] = -span, span
            origin[name] = key
        for side, source_slots in (("left", slice(0, 3)), ("right", slice(3, 6))):
            name = f"{side}_eef"
            slots = self.fields[name]
            position = slice(slots.start, slots.start + 3)
            pair = bounds(state_stats.get("observation.states.end.position"), source_slots, 3)
            if pair is not None:
                sl[position], sh[position] = pair
                span = pair[1] - pair[0]
                al[position], ah[position] = -span, span
                origin[name] = "position:state_span; quaternion:unit_component_range"
            key = f"action.{side}_arm_eef_position.delta_sub"
            pair = bounds(delta_stats.get(key), slice(None), 3)
            if pair is not None:
                al[position], ah[position] = pair
                origin[name] = f"position:{key}; quaternion:unit_component_range"
        for values in (sl, sh, al, ah):
            values.masked_fill_(~mask, 0)
        return (sl, sh), (al, ah), origin


def resolve_action_template(template=None):
    """支持实例、类或完整类路径；更换模板无需改通用 dataset。"""
    if template is None:
        return ActionStateTemplate55()
    if isinstance(template, str):
        module, name = template.rsplit(".", 1)
        template = getattr(importlib.import_module(module), name)
    if isinstance(template, type):
        template = template()
    if not isinstance(template, ActionStateTemplate):
        raise TypeError("Expected an ActionStateTemplate instance or class path")
    return template
