# SPDX-License-Identifier: OpenMDW-1.1
"""Thin adapter for already-template-mapped AgiBot exports."""

from cosmos_framework.data.generator.action.datasets.segment_lerobot_dataset import SegmentLeRobotDataset
from cosmos_framework.data.generator.action.sample_contract import ActionReadOptions
from cosmos_framework.data.generator.action.video_view import VideoViewConfig

_HEAD_KEY = "observation.images.head"  # 头部相机的视频字段。
_HAND_LEFT_KEY = "observation.images.hand_left"  # 左腕相机的视频字段。
_HAND_RIGHT_KEY = "observation.images.hand_right"  # 右腕相机的视频字段。


# 来源字段与默认目标读取规则；不从列名推断是否已重标。
_STATE_KEY = "state_unified"  # 按公共模板排布的实测状态列。
_ACTION_KEY = "action_unified"  # 按公共模板排布的绝对动作目标列，尚未计算 block delta。
_STATE_MASK_KEY = "mask_state"  # state 有效维度；每个子数据集读取一次并缓存。
_ACTION_MASK_KEY = "mask_action"  # action 有效维度；从 state 构造目标时改用 state mask。
# 默认读取当前 action 行；如需下一步 state，调用方显式覆盖 read_options。
_DEFAULT_READ_OPTIONS = ActionReadOptions(
    state_key=_STATE_KEY,
    action_key=_ACTION_KEY,
    state_mask_key=_STATE_MASK_KEY,
    action_mask_key=_ACTION_MASK_KEY,
    action_from_state=False,  # False 读取 action 列；True 使用 state 列构造目标。
    action_time_offset_steps=0,  # 0 读取当前行，1 读取下一行；对上述选中的目标列生效。
)
# 布局角色映射到来源相机；ego_view 只取 head，concat_view 使用三路。
_CAMERA_MAPPING = {"head": _HEAD_KEY, "left": _HAND_LEFT_KEY, "right": _HAND_RIGHT_KEY}


class AgiBotSegmentLeRobotDataset(SegmentLeRobotDataset):
    """处理后 AgiBot：只声明列和相机，不执行 FK、维度重排或归一化。

    template/source_contract/planner 由调用方传入，替换模板无需修改本类。
    默认使用原始 action；state 下一步目标需显式传入 read_options。
    """

    default_read_options = _DEFAULT_READ_OPTIONS

    def __init__(
        self,
        *,
        viewpoint="concat_view",  # 默认上方头部视角，下方左、右腕视角。
        video_view: VideoViewConfig | None = None,
        read_options=_DEFAULT_READ_OPTIONS,
        **kwargs,
    ):
        # 配置只声明来源相机映射；每次读取由显式 viewpoint 选择布局。
        if video_view is None:
            video_view = VideoViewConfig(cameras=_CAMERA_MAPPING.copy())
        super().__init__(read_options=read_options, video_view=video_view, viewpoint=viewpoint, **kwargs)
