# SPDX-License-Identifier: OpenMDW-1.1
"""Shared temporal geometry, independent of action layout and dataset I/O.

Observation/action indexes are local to a segment, before video subsampling.
Latent index 0 is the independent visual prefix and has no action interval.
This module does not choose segment starts, overlap, anchors or loss masks.
"""

import math
from dataclasses import dataclass
from numbers import Real


def _check_integer(name: str, value: int, *, minimum: int) -> None:
    """拒绝 bool、浮点数和越界值，避免时间索引被隐式截断。"""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


def _checked_fps(name: str, value: float) -> float:
    """FPS 必须是有限正数，保留小数，不接受 bool 或字符串。"""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return float(value)


@dataclass(frozen=True, kw_only=True)
class CausalBlockGeometry:
    """One immutable temporal contract for indexing and causal encoding.

    temporal_compression_factor must come from the selected VAE/config; it
    deliberately has no default. The current recipe uses 4, giving 32 actions
    per block / 4 video stride / 4 compression = 2 predicted latent frames.
    actions_per_block is a temporal length, not the template channel width.
    """

    temporal_compression_factor: int
    actions_per_block: int = 32
    video_stride: int = 4

    def __post_init__(self) -> None:
        """一个完整 action block 必须对应整数个预测 latent。"""
        for name in ("temporal_compression_factor", "actions_per_block", "video_stride"):
            _check_integer(name, getattr(self, name), minimum=1)
        if self.actions_per_block % self.actions_per_latent_frame:
            raise ValueError("actions_per_block must be divisible by video_stride * temporal_compression_factor")

    @property
    def actions_per_latent_frame(self) -> int:
        """每个后续 latent 对应的 action 步数，不包含首帧。"""
        return self.video_stride * self.temporal_compression_factor

    @property
    def latent_frames_per_block(self) -> int:
        """每个预测 block 的 latent 数，由 action 数推导，不能独立配置。"""
        return self.actions_per_block // self.actions_per_latent_frame

    @property
    def min_observation_frames(self) -> int:
        """一个完整 block 所需的最少 observation 数，包含起始帧。"""
        return self.actions_per_block + 1

    def video_fps(self, aligned_fps: float) -> float:
        """抽帧后的视频 FPS；输入为 Reader 对齐后的 FPS，不是原始文件 FPS。

        不再除以 VAE 压缩倍率：mRoPE 会使用该倍率换算 latent 时间。
        调用方应与实际抽帧同步更新 metadata，不能对同一片段重复换算。
        """
        return _checked_fps("video_fps", _checked_fps("aligned_fps", aligned_fps) / self.video_stride)

    def action_fps(self, aligned_fps: float) -> float:
        """action 不随视频抽帧，保持 Reader 对齐后的采样频率。"""
        return _checked_fps("aligned_fps", aligned_fps)

    def validate_observation_frames(self, num_frames: int) -> None:
        """片段长度必须为 A*x+1，x>=1；只检查，不自动裁剪或补齐。"""
        _check_integer("num_frames", num_frames, minimum=self.min_observation_frames)
        if (num_frames - 1) % self.actions_per_block:
            raise ValueError(f"num_frames must equal {self.actions_per_block} * x + 1, with x >= 1")

    def max_complete_observation_frames(self, available_frames: int) -> int:
        """返回不超过可用长度的最大合法帧数；不足一个 block 返回 0。

        只计算长度。截哪一端、是否 warning 或跳过，由切片规划器决定。
        """
        _check_integer("available_frames", available_frames, minimum=0)
        if available_frames < self.min_observation_frames:
            return 0
        return ((available_frames - 1) // self.actions_per_block) * self.actions_per_block + 1

    def num_blocks(self, num_observation_frames: int) -> int:
        """合法片段包含的完整预测 block 数，不计视觉前缀。"""
        self.validate_observation_frames(num_observation_frames)
        return (num_observation_frames - 1) // self.actions_per_block

    def num_video_frames(self, num_observation_frames: int) -> int:
        """抽帧后送入 VAE 的视频帧数，包含独立首帧。"""
        self.validate_observation_frames(num_observation_frames)
        return (num_observation_frames - 1) // self.video_stride + 1

    def num_latent_frames(self, num_observation_frames: int) -> int:
        """VAE 输出总 latent 数，包含 L0。"""
        return self.num_blocks(num_observation_frames) * self.latent_frames_per_block + 1

    def block_action_span(self, block_index: int) -> tuple[int, int]:
        """从 0 编号的预测 block 对应的 action 区间，左闭右开。"""
        _check_integer("block_index", block_index, minimum=0)
        start = block_index * self.actions_per_block
        return start, start + self.actions_per_block

    def block_latent_span(self, block_index: int) -> tuple[int, int]:
        """预测 block 对应的 latent 区间，左闭右开，跳过 L0。"""
        _check_integer("block_index", block_index, minimum=0)
        start = 1 + block_index * self.latent_frames_per_block
        return start, start + self.latent_frames_per_block

    def action_latent_index(self, action_index: int) -> int:
        """将片段内 action 映射到后续 latent；首个 action 对应 L1。"""
        _check_integer("action_index", action_index, minimum=0)
        return 1 + action_index // self.actions_per_latent_frame

    def latent_block_index(self, latent_index: int) -> int:
        """后续 latent 对应的预测 block；L0 返回 -1，表示独立视觉条件。"""
        _check_integer("latent_index", latent_index, minimum=0)
        return (latent_index - 1) // self.latent_frames_per_block
