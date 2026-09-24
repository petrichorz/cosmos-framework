# SPDX-License-Identifier: OpenMDW-1.1
"""Template-independent contracts for unnormalized, aligned reader output.

No dataset I/O, block partitioning, target relabeling or normalization runs here.
"""

from dataclasses import dataclass

import torch

from cosmos_framework.data.generator.action.action_state_template import ActionStateTemplate


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


def validate_raw_action_sample(sample: dict, template: ActionStateTemplate) -> None:
    """校验单个原始片段的形状和时间对齐，不修改字典。

    固定字段/FPS 在 Reader 初始化时校验；mask/来源契约首次加载时校验。
    有效值和旋转合法性由后续模板编码检查，此处不重复 sanitize。
    action_target 为绝对目标；action 时间戳标记区间起点。
    """
    for name in ("state_trajectory", "action_target"):
        values = sample[name]
        if not isinstance(values, torch.Tensor) or values.ndim != 2 or values.shape[-1] != template.width:
            raise ValueError(f"{name} must have shape [N, {template.width}]")
    count = len(sample["action_target"])
    if count < 1 or len(sample["state_trajectory"]) != count + 1:
        raise ValueError("Expected T absolute targets and T+1 measured states, with T >= 1")
    for name, length in (("state_timestamps", count + 1), ("action_timestamps", count)):
        times = sample[name]
        if not isinstance(times, torch.Tensor) or times.shape != (length,) or not times.is_floating_point():
            raise ValueError(f"{name} must be a floating-point [{length}] tensor in seconds")
        if not torch.isfinite(times).all() or (times[1:] <= times[:-1]).any():
            raise ValueError(f"{name} must be finite and strictly increasing")
        dt = times[1:] - times[:-1]
        if not torch.allclose(dt, torch.full_like(dt, 1 / sample["conditioning_fps"]), atol=1e-5, rtol=1e-4):
            raise ValueError(f"{name} must follow the declared aligned sampling rate")

    indexes = sample["action_state_indexes"]
    if not isinstance(indexes, torch.Tensor) or indexes.dtype != torch.int64 or indexes.shape != (count,):
        raise ValueError("action_state_indexes must be an int64 [T] tensor")
    if not torch.equal(indexes, torch.arange(count, device=indexes.device)):
        raise ValueError("Aligned targets must map to their corresponding interval-start states")
    state_times = sample["state_timestamps"][:-1].to(sample["action_timestamps"])
    if not torch.allclose(state_times, sample["action_timestamps"], atol=1e-5, rtol=0):
        raise ValueError("Action interval starts must match the preceding observation timestamps")
    if sample["video"] is not None and (
        not isinstance(sample["video"], torch.Tensor)
        or sample["video"].ndim != 4
        or sample["video"].shape[1] != count + 1
    ):
        raise ValueError("video must have shape [C,T+1,H,W] before temporal subsampling")
