# SPDX-License-Identifier: OpenMDW-1.1
"""Thin adapter for already-template-mapped AgiBot exports."""

from cosmos_framework.data.generator.action.datasets.segment_lerobot_dataset import SegmentLeRobotDataset
from cosmos_framework.data.generator.action.sample_contract import ActionReadOptions
from cosmos_framework.data.generator.action.video_view import VideoViewConfig

_HEAD_KEY = "observation.images.head"
_HAND_LEFT_KEY = "observation.images.hand_left"
_HAND_RIGHT_KEY = "observation.images.hand_right"


class AgiBotSegmentLeRobotDataset(SegmentLeRobotDataset):
    """处理后 AgiBot：只声明列和相机，不执行 FK、维度重排或归一化。

    template/source_contract/planner 由调用方传入，替换模板无需修改本类。
    默认使用原始 action；state 下一步目标需显式传入 read_options。
    """

    def __init__(
        self,
        *,
        viewpoint="concat_view",
        video_view: VideoViewConfig | None = None,
        read_options=ActionReadOptions(state_key="state_unified", action_key="action_unified"),
        **kwargs,
    ):
        # 配置只声明来源相机映射；每次读取由显式 viewpoint 选择布局。
        if video_view is None:
            video_view = VideoViewConfig(
                cameras={"head": _HEAD_KEY, "left": _HAND_LEFT_KEY, "right": _HAND_RIGHT_KEY},
            )
        super().__init__(read_options=read_options, video_view=video_view, viewpoint=viewpoint, **kwargs)
