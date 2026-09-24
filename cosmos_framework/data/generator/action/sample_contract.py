# SPDX-License-Identifier: OpenMDW-1.1
"""Template-independent contracts for unnormalized, aligned reader output.

No dataset I/O, block partitioning, target relabeling or normalization runs here.
"""

import math
from dataclasses import dataclass, field

import torch

from cosmos_framework.data.generator.action.action_state_template import ActionStateTemplate, TemplateSourceContract


@dataclass(frozen=True)
class ActionReadOptions:
    """Per-source field selection; offsets count steps on the aligned grid.

    ``action_from_state=False, action_time_offset_steps=0`` reads the existing action.
    ``action_from_state=True, action_time_offset_steps=1`` uses the next measured state.
    The reader must reject unavailable targets instead of padding them.
    """

    state_key: str = "observation.state"
    action_key: str = "action"
    # 每个 LeRobot v3 子数据集加载一次并缓存；其全部帧共用这对 mask。
    state_mask_key: str = "mask_state"
    action_mask_key: str = "mask_action"
    # False：读取 action_key；True：读取 state_key 作为 action 目标。
    action_from_state: bool = False
    # 相对于当前对齐时间步：0 读取当前步，1 读取下一步，-1 读取前一步。
    # 对 action 列或 state 列都适用；单位是对齐后的时间步，不是秒。
    action_time_offset_steps: int = 0
    # 默认action配置 action_from_state=False action_time_offset_steps=0
    # 如果使用state差分 action_from_state=True action_time_offset_steps=1 action[t]=state[t+1]

    def __post_init__(self):
        """拒绝空字段、未知来源和非整数偏移，避免静默改变读取语义。"""
        for name in ("state_key", "action_key", "state_mask_key", "action_mask_key"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty field name")
        if not isinstance(self.action_from_state, bool):
            raise ValueError("action_from_state must be bool")
        if isinstance(self.action_time_offset_steps, bool) or not isinstance(self.action_time_offset_steps, int):
            raise ValueError("action_time_offset_steps must be an integer aligned-grid offset")

    @property
    def target_key(self) -> str:
        """返回目标读取字段；不在此执行切片或重标。"""
        return self.state_key if self.action_from_state else self.action_key

    @property
    def target_mask_key(self) -> str:
        """目标来自 state 时使用 state mask，否则使用 action mask。"""
        return self.state_mask_key if self.action_from_state else self.action_mask_key


@dataclass
class RawActionSample:
    """One aligned segment before block encoding and normalization.

    State/video contain T+1 observations; action_target contains T absolute
    targets. Masks are independent dataset-level [D] vectors, not padding masks.
    Each LeRobot v3 subdataset owns one state/action mask pair, loaded and
    validated once by its reader, then reused across all its samples.
    Do not read or compare masks per frame, or share a cache across different
    subdataset roots. Missing masks must not silently become all-valid.
    When action_from_state is true, action_mask is the state-derived target
    mask and must equal state_mask under this dataset-level mask contract.
    action_timestamps mark interval starts, not the physical target time:
    that meaning is declared in source_contract.target_semantics/read_options.
    video is optional for numeric inspection; when present it is [C,T+1,H,W].
    All timestamps use the same origin and seconds as their unit.
    """

    state_trajectory: torch.Tensor
    action_target: torch.Tensor
    state_mask: torch.Tensor
    action_mask: torch.Tensor
    state_timestamps: torch.Tensor
    action_timestamps: torch.Tensor
    action_state_indexes: torch.Tensor
    source_contract: TemplateSourceContract
    conditioning_fps: float
    read_options: ActionReadOptions = field(default_factory=ActionReadOptions)
    video: torch.Tensor | None = None
    ai_caption: str = ""

    def validate(self, template: ActionStateTemplate) -> None:
        """只校验原始样本，不清洗或修改数据，不施加 block 长度约束。"""
        if not isinstance(self.read_options, ActionReadOptions):
            raise ValueError("read_options must be ActionReadOptions")
        for name in ("state_trajectory", "action_target"):
            values = getattr(self, name)
            if not isinstance(values, torch.Tensor) or values.ndim != 2 or values.shape[-1] != template.width:
                raise ValueError(f"{name} must have shape [N, {template.width}]")
        count = len(self.action_target)
        if count < 1 or len(self.state_trajectory) != count + 1:
            raise ValueError("Expected T absolute targets and T+1 measured states, with T >= 1")
        # 轨迹长度不同，不能用要求广播对齐的 template.validate(anchor, target)。
        for values, mask in ((self.state_trajectory, self.state_mask), (self.action_target, self.action_mask)):
            template.validate_source_contract(self.source_contract, mask)
            template.sanitize(values, mask)
        if self.read_options.action_from_state:
            state_mask = template.validate_valid_mask(self.state_mask)
            action_mask = template.validate_valid_mask(self.action_mask).to(state_mask.device)
            if not torch.equal(state_mask, action_mask):
                raise ValueError("State-derived action targets must use the dataset state mask")

        if (
            isinstance(self.conditioning_fps, bool)
            or not math.isfinite(self.conditioning_fps)
            or self.conditioning_fps <= 0
        ):
            raise ValueError("conditioning_fps must be finite and positive")
        for name, length in (("state_timestamps", count + 1), ("action_timestamps", count)):
            times = getattr(self, name)
            if not isinstance(times, torch.Tensor) or times.shape != (length,) or not times.is_floating_point():
                raise ValueError(f"{name} must be a floating-point [{length}] tensor in seconds")
            if not torch.isfinite(times).all() or (times[1:] <= times[:-1]).any():
                raise ValueError(f"{name} must be finite and strictly increasing")
            dt = times[1:] - times[:-1]
            if not torch.allclose(dt, torch.full_like(dt, 1 / self.conditioning_fps), atol=1e-5, rtol=1e-4):
                raise ValueError(f"{name} must follow the declared aligned sampling rate")

        indexes = self.action_state_indexes
        if not isinstance(indexes, torch.Tensor) or indexes.dtype != torch.int64 or indexes.shape != (count,):
            raise ValueError("action_state_indexes must be an int64 [T] tensor")
        if not torch.equal(indexes, torch.arange(count, device=indexes.device)):
            raise ValueError("Aligned targets must map to their corresponding interval-start states")
        state_times = self.state_timestamps[:-1].to(self.action_timestamps)
        if not torch.allclose(state_times, self.action_timestamps, atol=1e-5, rtol=0):
            raise ValueError("Action interval starts must match the preceding observation timestamps")
        if self.video is not None and (
            not isinstance(self.video, torch.Tensor) or self.video.ndim != 4 or self.video.shape[1] != count + 1
        ):
            raise ValueError("video must have shape [C,T+1,H,W] before temporal subsampling")
