# SPDX-License-Identifier: OpenMDW-1.1
"""Template-independent contracts for unnormalized, aligned reader output.

No dataset I/O, block partitioning, target relabeling or normalization runs here.
"""

from dataclasses import dataclass


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
