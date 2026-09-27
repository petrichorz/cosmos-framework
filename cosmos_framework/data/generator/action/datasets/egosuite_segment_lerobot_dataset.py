# SPDX-License-Identifier: OpenMDW-1.1
"""Thin adapter for already-template-mapped EgoSuite exports."""

from cosmos_framework.data.generator.action.datasets.segment_lerobot_dataset import SegmentLeRobotDataset
from cosmos_framework.data.generator.action.sample_contract import ActionReadOptions
from cosmos_framework.data.generator.action.video_view import VideoViewConfig

_HEAD_LEFT_KEY = "observation.images.head_left"  # 左侧头部相机的视频字段。


# 来源字段与默认目标读取规则；不从列名推断是否已重标。
_STATE_KEY = "state_unified"  # 按公共模板排布的实测状态列。
_ACTION_KEY = "action_unified"  # 按公共模板排布的绝对动作目标列，尚未计算 block delta。
_STATE_MASK_KEY = "mask_state"  # state 有效维度；每个子数据集读取一次并缓存。
_ACTION_MASK_KEY = "mask_action"  # action 有效维度；从 state 构造目标时改用 state mask。
# action 已在预处理时重标为下一步 state，直接读当前 action 行。
_DEFAULT_READ_OPTIONS = ActionReadOptions(
    state_key=_STATE_KEY,
    action_key=_ACTION_KEY,
    state_mask_key=_STATE_MASK_KEY,
    action_mask_key=_ACTION_MASK_KEY,
    action_from_state=False,  # False 读取 action 列；True 使用 state 列构造目标。
    action_time_offset_steps=0,  # 0 读取当前行，1 读取下一行；对上述选中的目标列生效。
)
_CAMERA_MAPPING = {"head": _HEAD_LEFT_KEY}  # ego_view 的 head 使用左相机，不读取右相机。


class EgoSuiteSegmentLeRobotDataset(SegmentLeRobotDataset):
    """处理后 EgoSuite：ego_view 默认只读取左侧头部相机。

    template/source_contract/planner 由调用方传入，不在 Reader 中解释通道语义。
    当前导出的 action 已是下一步 state，默认直接读取，不重复移位。
    """

    default_read_options = _DEFAULT_READ_OPTIONS

    def __init__(
        self,
        *,
        viewpoint="ego_view",  # 默认单路头部视角，具体相机由上方映射指定。
        video_view: VideoViewConfig | None = None,
        read_options=_DEFAULT_READ_OPTIONS,
        **kwargs,
    ):
        # ego_view 的 head 角色映射到左相机，不解码右相机。
        if video_view is None:
            video_view = VideoViewConfig(cameras=_CAMERA_MAPPING.copy())
        super().__init__(read_options=read_options, video_view=video_view, viewpoint=viewpoint, **kwargs)
