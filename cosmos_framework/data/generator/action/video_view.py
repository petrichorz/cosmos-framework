# SPDX-License-Identifier: OpenMDW-1.1
"""Shared spatial camera composition; no temporal subsampling."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class _VideoLayout:
    """Row arrangement and layout-only resize targets, expressed as (height, width)."""

    rows: tuple[tuple[str, ...], ...]
    resize_targets: Callable[[int, int], dict[str, tuple[int, int]]] | None = None

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(role for row in self.rows for role in row))

    def sizes_to_resize(self, height: int, width: int) -> dict[str, tuple[int, int]]:
        # A role absent here keeps its original pixels on the legacy path.
        return self.resize_targets(height, width) if self.resize_targets is not None else {}


def _concat_resize_targets(height: int, width: int) -> dict[str, tuple[int, int]]:
    if height < 2 or width < 2:
        raise ValueError("concat_view requires head height and width >= 2")
    # The right camera receives the remaining column for odd head widths.
    return {"left": (height // 2, width // 2), "right": (height // 2, width - width // 2)}


def _get_layout(viewpoint: str) -> _VideoLayout:
    """Keep role selection and spatial rules in one layout definition."""
    if viewpoint == "ego_view":
        return _VideoLayout(rows=(("head",),))
    if viewpoint == "concat_view":
        return _VideoLayout(
            rows=(("head",), ("left", "right")),
            resize_targets=_concat_resize_targets,
        )
    raise ValueError(f"Unsupported viewpoint: {viewpoint}")


def _compose_prepared(frames: Mapping[str, torch.Tensor], count: int, layout: _VideoLayout) -> torch.Tensor:
    """按布局拼接已准备好尺寸的 TCHW 相机视频，返回 CTHW；不缩放、补边或转换 dtype。"""
    if not layout.rows or any(not row for row in layout.rows):
        raise ValueError("Video layout must contain nonempty rows")
    # 时间戳对齐由上游读取负责；这里仅检查帧数、通道、尺寸及 dtype 是否满足拼接条件。
    dtype = None
    for role in layout.roles:
        if role not in frames:
            raise ValueError(f"Missing prepared camera role: {role}")
        video = frames[role]
        if video.ndim != 4 or video.shape[:2] != (count, 3) or min(video.shape[-2:]) <= 0:
            raise ValueError(f"Camera {role} must have shape [{count},3,H,W] with positive dimensions")
        if dtype is not None and video.dtype != dtype:
            raise TypeError("Prepared cameras must have the same dtype")
        dtype = video.dtype
    # 每个元组是一行，例如 (("head",), ("left", "right")) 表示头部在上、双腕在下。
    rows = []
    row_width = None
    for roles in layout.rows:
        cameras = [frames[role] for role in roles]
        # 行内沿宽度拼接：高度必须相同，宽度可以不同（如奇数宽度拆分后的左右腕）。
        if len({camera.shape[-2] for camera in cameras}) != 1:
            raise ValueError("Cameras in a layout row must have the same height")
        width = sum(camera.shape[-1] for camera in cameras)
        # 各行随后沿高度拼接，因此每一行的总宽度必须相同。
        if row_width is not None and width != row_width:
            raise ValueError("Layout rows must have the same width")
        row_width = width
        # TCHW 的最后一维是宽度；单相机行直接复用，不额外执行 cat。
        rows.append(cameras[0] if len(cameras) == 1 else torch.cat(cameras, dim=-1))
    # 倒数第二维是高度：将各行从上到下拼接。
    video = rows[0] if len(rows) == 1 else torch.cat(rows, dim=-2)
    # TCHW → CTHW；contiguous 必要时复制数据以保证连续内存，不改变像素值。
    return video.permute(1, 0, 2, 3).contiguous()


@dataclass(frozen=True)
class VideoViewConfig:
    """相机角色映射与布局名称分开配置；布局由调用方显式传入，缺失相机或文件直接报错。"""

    cameras: Mapping[str, str]  # 布局中的 head/left/right → 来源视频字段。

    def camera_roles(self, viewpoint):
        """每种布局明确声明需要的相机，不由行列数量推断。"""
        return _get_layout(viewpoint).roles

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
        """Prepare cameras with the selected backend, then compose using the shared layout."""
        layout = _get_layout(viewpoint)
        keys = self.camera_keys(viewpoint=viewpoint)
        for key in keys:
            path = ds.root / ds.meta.get_video_file_path(episode_id, key)
            if not path.is_file():
                raise FileNotFoundError(f"Required camera {key}, episode {episode_id}: {path}")
        if getattr(ds, "video_backend", None) == "pyav_resize":
            prepared = self._prepare_pyav_resized(ds, episode_id, timestamps, layout)
        else:
            frames = ds._query_videos({key: timestamps for key in keys}, episode_id)
            prepared = self._prepare_legacy(frames, len(timestamps), layout)
        return _compose_prepared(prepared, len(timestamps), layout)

    def compose(self, frames, count, *, viewpoint):
        """Legacy entry point: resize decoded cameras for the layout, then compose."""
        self.camera_keys(viewpoint=viewpoint)
        layout = _get_layout(viewpoint)
        return _compose_prepared(self._prepare_legacy(frames, count, layout), count, layout)

    def _prepare_legacy(self, frames, count, layout):
        prepared = {}
        for role in layout.roles:
            key = self.cameras[role]
            video = frames[key]
            if video.ndim != 4 or video.shape[:2] != (count, 3):
                raise ValueError(f"Camera {key} must have shape [{count},3,H,W]")
            prepared[role] = video
        # Keep actual decoded dimensions and the original float interpolation order.
        height, width = prepared["head"].shape[-2:]
        for role, size in layout.sizes_to_resize(height, width).items():
            prepared[role] = F.interpolate(prepared[role], size=size, mode="bilinear", align_corners=False)
        return prepared

    def _prepare_pyav_resized(self, ds, episode_id, timestamps, layout):
        """Decode each role to its layout size; leave all concatenation to the caller."""
        from lerobot.datasets.video_utils import decode_video_frames

        feature = ds.meta.features[self.cameras["head"]]
        names, shape = feature.get("names"), feature["shape"]
        if not isinstance(names, (list, tuple)) or len(names) != len(shape):
            raise ValueError("pyav_resize requires named head video dimensions")
        if names.count("height") != 1 or names.count("width") != 1:
            raise ValueError("pyav_resize requires named height and width dimensions")
        height, width = shape[names.index("height")], shape[names.index("width")]
        if any(type(size) is not int or size <= 0 for size in (height, width)):
            raise ValueError("Invalid head video dimensions")
        sizes = {"head": (height, width), **layout.sizes_to_resize(height, width)}
        episode = ds.meta.episodes[episode_id]
        prepared = {}
        for role in layout.roles:
            key = self.cameras[role]
            offset = episode[f"videos/{key}/from_timestamp"]
            shifted = [offset + timestamp for timestamp in timestamps]
            resize_h, resize_w = sizes[role]
            video = decode_video_frames(
                ds.root / ds.meta.get_video_file_path(episode_id, key),
                shifted,
                ds.tolerance_s,
                backend="pyav_resize",
                resize_h=resize_h,
                resize_w=resize_w,
            )
            if video.dtype != torch.uint8 or video.shape != (len(timestamps), 3, resize_h, resize_w):
                raise ValueError(f"Camera {key} must return uint8 [{len(timestamps)},3,{resize_h},{resize_w}]")
            prepared[role] = video
        return prepared
