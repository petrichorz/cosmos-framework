# SPDX-License-Identifier: OpenMDW-1.1
"""Camera ordering, missing-camera policy and synchronized query checks."""

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.action.video_view import VideoViewConfig


def test_layout_preserves_camera_order_and_odd_width():
    view = VideoViewConfig(cameras={"head": "head", "left": "left", "right": "right"})
    frames = {
        key: torch.full((33, 3, 8, 11), float(i)) for i, key in enumerate(view.camera_keys(viewpoint="concat_view"))
    }
    result = view.compose(frames, 33, viewpoint="concat_view")
    assert result.shape == (3, 33, 12, 11)
    assert (result[:, :, :8] == 0).all()
    assert (result[:, :, 8:, :5] == 1).all()
    assert (result[:, :, 8:, 5:] == 2).all()


def test_ego_keeps_pixels_and_frame_order():
    view = VideoViewConfig(cameras={"head": "head"})
    frames = torch.rand(65, 3, 8, 10)
    torch.testing.assert_close(view.compose({"head": frames}, 65, viewpoint="ego_view"), frames.permute(1, 0, 2, 3))


def test_missing_camera_or_frame_is_an_error():
    view = VideoViewConfig(cameras={"head": "head"})
    with pytest.raises(ValueError, match="Required video camera"):
        view.validate_features({}, viewpoint="ego_view")
    with pytest.raises(ValueError, match="shape"):
        view.compose({"head": torch.zeros(32, 3, 8, 8)}, 33, viewpoint="ego_view")


def test_only_selected_cameras_receive_same_timestamps(tmp_path):
    view = VideoViewConfig(cameras={"head": "head", "left": "left", "right": "right"})
    meta = SimpleNamespace(get_video_file_path=lambda episode, key: f"{key}.mp4")
    calls = []

    def query(times, episode):
        calls.append((times, episode))
        return {key: torch.zeros(len(ts), 3, 8, 8) for key, ts in times.items()}

    ds = SimpleNamespace(root=tmp_path, meta=meta, _query_videos=query)
    with pytest.raises(FileNotFoundError, match="Required camera head"):
        view.read(ds, 7, [0.5, 0.6], viewpoint="concat_view")
    for key in view.camera_keys(viewpoint="concat_view"):
        (tmp_path / f"{key}.mp4").touch()
    view.read(ds, 7, [0.5, 0.6], viewpoint="concat_view")
    assert calls == [({key: [0.5, 0.6] for key in view.camera_keys(viewpoint="concat_view")}, 7)]


def test_role_mapping_and_ego_camera_selection():
    view = VideoViewConfig(cameras={"head": "robot.front", "left": "robot.wrist"})
    assert view.camera_keys(viewpoint="ego_view") == ("robot.front",)
    frames = torch.rand(2, 3, 8, 10)
    torch.testing.assert_close(
        view.compose({"robot.front": frames}, 2, viewpoint="ego_view"), frames.permute(1, 0, 2, 3)
    )


def test_unsupported_layout_and_missing_role_are_rejected():
    view = VideoViewConfig(cameras={"head": "front"})
    with pytest.raises(ValueError, match="Unsupported viewpoint"):
        view.camera_keys(viewpoint="two_view")
    with pytest.raises(ValueError, match="left"):
        view.camera_keys(viewpoint="concat_view")


def test_same_camera_config_supports_different_sample_layouts():
    view = VideoViewConfig(cameras={"head": "front", "left": "wrist_l", "right": "wrist_r"})
    frames = {key: torch.rand(2, 3, 8, 10) for key in view.cameras.values()}
    assert view.compose(frames, 2, viewpoint="ego_view").shape == (3, 2, 8, 10)
    assert view.compose(frames, 2, viewpoint="concat_view").shape == (3, 2, 12, 10)
    assert "wrist" not in view.describe(viewpoint="ego_view")
    assert "wrist" in view.describe(viewpoint="concat_view")
    assert set(vars(view)) == {"cameras"}
