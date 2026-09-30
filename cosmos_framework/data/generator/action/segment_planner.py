# SPDX-License-Identifier: OpenMDW-1.1
"""Plan aligned episode segments without reading parquet or video."""

import warnings

from cosmos_framework.data.generator.action.causal_block_geometry import CausalBlockGeometry


class SegmentPlanner:
    """按统一配置切分 episode/segment，返回 (observation 起点, action 数)。

    索引位于对齐后的 episode 时间网格，不是 parquet 行号。调用方保留
    来源和 episode 信息，并提前确定读取偏移下的有效范围。
    skipped_ranges / discarded_action_steps 累计本实例所有 plan 调用的结果；
    建立一份新数据集索引时创建一个实例，不在 __getitem__ 中重复规划。
    """

    def __init__(self, *, max_action_steps: int, overlap_action_steps: int, geometry: CausalBlockGeometry):
        # 只保留防止非法长度和零/负推进间隔所需的配置检查。
        if type(max_action_steps) is not int:
            raise ValueError("max_action_steps must be an integer")
        geometry.validate_observation_frames(max_action_steps + 1)
        if type(overlap_action_steps) is not int or not 0 <= overlap_action_steps < max_action_steps:
            raise ValueError("overlap_action_steps must be an integer in [0, max_action_steps)")

        # 保存切片配置；计数器累计所有 episode/segment 的跳过与丢弃情况。
        self.geometry = geometry
        self.max_action_steps = max_action_steps
        self.overlap_action_steps = overlap_action_steps
        self.skipped_ranges = 0
        self.discarded_action_steps = 0

    def plan(
        self,
        num_observation_frames: int,
        *,
        observation_start: int = 0,
        source_id: str = "",
        episode_id: int | None = None,
        segment_id: str | None = None,
        preserve_tail: bool = False,
    ) -> list[tuple[int, int]]:
        """长范围尾段右对齐；短范围保留开头；过短范围 warning 并跳过。

        每条 (start, actions) 对应 observation[start:start+actions+1]。
        来源参数仅用于 warning；不检查标识格式，也不包装输出记录。
        丢弃计数仅含未覆盖的 action 间隔，不包含 overlap 的重复部分。
        preserve_tail 用于 subtask：短范围有余量时，首尾各取一个最大合法窗口。
        """
        # 计算完整 block 能容纳的最大帧数，只算长度，此处不截断长 episode。
        usable_frames = self.geometry.max_complete_observation_frames(num_observation_frames)
        if type(observation_start) is not int or observation_start < 0:
            raise ValueError("observation_start must be a nonnegative integer")
        available_actions = max(0, num_observation_frames - 1)

        # 不足一个完整 block：告警并跳过，记录未参与训练的 action 数。
        if usable_frames == 0:
            warnings.warn(
                f"Skipping source={source_id!r}, episode={episode_id}, segment={segment_id!r}, "
                f"observations=[{observation_start}, {observation_start + num_observation_frames}): "
                f"{num_observation_frames} aligned frames available; "
                f"at least {self.geometry.min_observation_frames} required",
                UserWarning,
                stacklevel=2,
            )
            self.skipped_ranges += 1
            self.discarded_action_steps += available_actions
            return []

        # 短范围默认截尾；subtask 保尾时，尾窗与前窗等长，保留尽可能多的上下文。
        if available_actions < self.max_action_steps:
            if preserve_tail and num_observation_frames > usable_frames:
                window_actions = usable_frames - 1
                return [
                    (observation_start, window_actions),
                    (observation_start + available_actions - window_actions, window_actions),
                ]
            self.discarded_action_steps += num_observation_frames - usable_frames
            return [(observation_start, usable_frames - 1)]

        # 长 episode：按“片段长度 - overlap”推进，先生成所有放得下的完整片段。
        stride = self.max_action_steps - self.overlap_action_steps
        # 使最后一个完整片段恰好结束于有效范围末端的起点。
        tail_start = observation_start + available_actions - self.max_action_steps
        segments = [(start, self.max_action_steps) for start in range(observation_start, tail_start + 1, stride)]

        # 尾段右对齐：向前移动起点以覆盖结尾；已覆盖则不重复添加。
        # 最后一次 overlap 可以大于配置值，不要求起点与 block 对齐。
        if segments[-1][0] != tail_start:
            segments.append((tail_start, self.max_action_steps))
        return segments
