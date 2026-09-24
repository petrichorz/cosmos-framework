# SPDX-License-Identifier: OpenMDW-1.1
"""Shared spatial camera composition; no temporal subsampling."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class VideoViewConfig:
    """相机角色映射与布局名称分开配置；布局由调用方显式传入，缺失相机或文件直接报错。"""

    cameras: Mapping[str, str]  # 布局中的 head/left/right → 来源视频字段。

    def camera_roles(self, viewpoint):
        """每种布局明确声明需要的相机，不由行列数量推断。"""
        if viewpoint == "ego_view":
            return ("head",)
        if viewpoint == "concat_view":
            return ("head", "left", "right")
        raise ValueError(f"Unsupported viewpoint: {viewpoint}")

    def camera_keys(self, *, viewpoint):
        """按角色获取来源字段并去重，只解码当前布局需要的相机。"""
        roles = self.camera_roles(viewpoint)
        for role in roles:
            if not self.cameras.get(role):
                raise ValueError(f"Missing camera mapping for {viewpoint}: {role}")
        return tuple(dict.fromkeys(self.cameras[role] for role in roles))

    def validate_features(self, features, *, viewpoint):
        """初始化时检查选中的相机是否声明为视频。"""
        for key in self.camera_keys(viewpoint=viewpoint):
            if features.get(key, {}).get("dtype") != "video":
                raise ValueError(f"Required video camera is missing: {key}")

    def describe(self, *, viewpoint):
        """描述本次实际布局，不在相机配置中缓存布局描述。"""
        if viewpoint == "ego_view":
            return "The video shows the head-mounted camera view."
        if viewpoint == "concat_view":
            return (
                "The top row shows the head-mounted camera view. "
                "The bottom row shows the left wrist camera on the left and the right wrist camera on the right."
            )
        raise ValueError(f"Unsupported viewpoint: {viewpoint}")

    def read(self, ds, episode_id, timestamps, *, viewpoint):
        """按共同 episode 时间读取，LeRobot 负责加各视频文件中的 episode 偏移。"""
        for key in self.camera_keys(viewpoint=viewpoint):
            path = ds.root / ds.meta.get_video_file_path(episode_id, key)
            if not path.is_file():
                raise FileNotFoundError(f"Required camera {key}, episode {episode_id}: {path}")
        frames = ds._query_videos({key: timestamps for key in self.camera_keys(viewpoint=viewpoint)}, episode_id)
        return self.compose(frames, len(timestamps), viewpoint=viewpoint)

    def compose(self, frames, count, *, viewpoint):
        """按 viewpoint 显式选择拼接函数，统一输出 [C,T,H,W]。"""
        for key in self.camera_keys(viewpoint=viewpoint):
            video = frames[key]
            if video.ndim != 4 or video.shape[:2] != (count, 3):
                raise ValueError(f"Camera {key} must have shape [{count},3,H,W]")
        if viewpoint == "ego_view":
            video = self._compose_ego_view(frames)
        elif viewpoint == "concat_view":
            video = self._compose_concat_view(frames)
        else:
            raise ValueError(f"Unsupported viewpoint: {viewpoint}")
        return video.permute(1, 0, 2, 3).contiguous()

    def _compose_ego_view(self, frames):
        """只返回 head，不缩放、不拼接。"""
        return frames[self.cameras["head"]]

    def _compose_concat_view(self, frames):
        """上方保留 head 原尺寸，下方左右腕部各占半宽、半高。"""
        head = frames[self.cameras["head"]]
        left = frames[self.cameras["left"]]
        right = frames[self.cameras["right"]]
        height, width = head.shape[-2:]
        if height < 2 or width < 2:
            raise ValueError("concat_view requires head height and width >= 2")
        half_height, left_width = height // 2, width // 2
        # 奇数宽度的剩余一列交给右侧，保证上下两行宽度一致。
        left = F.interpolate(left, size=(half_height, left_width), mode="bilinear", align_corners=False)
        right = F.interpolate(right, size=(half_height, width - left_width), mode="bilinear", align_corners=False)
        bottom = torch.cat((left, right), dim=-1)
        return torch.cat((head, bottom), dim=-2)
