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


@pytest.mark.parametrize("viewpoint", ["ego_view", "concat_view"])
@pytest.mark.parametrize(
    "names,shape", [(["height", "width", "rgb"], [8, 11, 3]), (["rgb", "height", "width"], [3, 8, 11])]
)
def test_resized_layout_offsets_and_instance_isolation(tmp_path, monkeypatch, viewpoint, names, shape):
    from lerobot.datasets import video_utils

    view = VideoViewConfig(cameras={"head": "head", "left": "left", "right": "right"})
    offsets = {"head": 1000.0, "left": 2000.0, "right": 3000.0}
    meta = SimpleNamespace(
        features={"head": {"names": names, "shape": shape}},
        episodes=[{f"videos/{key}/from_timestamp": value for key, value in offsets.items()}],
        get_video_file_path=lambda episode, key: f"{key}.mp4",
    )
    for key in offsets:
        (tmp_path / f"{key}.mp4").touch()
    calls = []

    def decode(path, timestamps, tolerance, *, backend, resize_h, resize_w):
        key = path.stem
        calls.append((key, timestamps, tolerance, backend, resize_h, resize_w))
        return torch.full((len(timestamps), 3, resize_h, resize_w), list(offsets).index(key), dtype=torch.uint8)

    monkeypatch.setattr(video_utils, "decode_video_frames", decode)
    times = [0.0, 1 / 30, 2 / 30]
    legacy_calls = []

    def query(timestamps, episode):
        legacy_calls.append((timestamps, episode))
        return {key: torch.full((3, 3, 8, 11), 0.5) for key in timestamps}

    legacy = SimpleNamespace(root=tmp_path, meta=meta, video_backend="pyav", _query_videos=query)
    fast = SimpleNamespace(root=tmp_path, meta=meta, video_backend="pyav_resize", tolerance_s=1e-4)
    before = view.read(legacy, 0, times, viewpoint=viewpoint)
    result = view.read(fast, 0, times, viewpoint=viewpoint)
    after = view.read(legacy, 0, times, viewpoint=viewpoint)
    assert torch.equal(before, after) and len(legacy_calls) == 2
    assert result.dtype == torch.uint8
    assert result.shape == (3, 3, 8 if viewpoint == "ego_view" else 12, 11)
    assert (result[:, :, :8] == 0).all()
    if viewpoint == "concat_view":
        assert (result[:, :, 8:, :5] == 1).all()
        assert (result[:, :, 8:, 5:] == 2).all()
    for key, shifted, tolerance, backend, height, width in calls:
        assert shifted == [offsets[key] + t for t in times]
        assert tolerance == 1e-4 and backend == "pyav_resize"
        assert (height, width) == {"head": (8, 11), "left": (4, 5), "right": (4, 6)}[key]
    assert [call[0] for call in calls] == list(view.camera_roles(viewpoint))


@pytest.mark.parametrize("height,width", [(None, None), (8, None), (0, 8), (8, -1), (True, 8)])
def test_resize_backend_requires_explicit_positive_dimensions(height, width):
    from lerobot.datasets.video_utils import decode_video_frames

    with pytest.raises(ValueError, match="positive integer"):
        decode_video_frames("unused.mp4", [0.0], 1e-4, "pyav_resize", height, width)


@pytest.mark.parametrize("backend", [None, "pyav", "video_reader", "torchcodec", "pyav_resize"])
@pytest.mark.parametrize("resized", [False, True])
def test_decoder_dispatch_preserves_legacy_routes(monkeypatch, backend, resized):
    from lerobot.datasets import video_utils

    calls = []
    for name in ("torchvision", "torchcodec", "pyav_resized"):

        def record(*args, _name=name, **kwargs):
            calls.append((_name, args, kwargs))
            return _name

        monkeypatch.setattr(video_utils, f"decode_video_frames_{name}", record)
    monkeypatch.setattr(video_utils, "get_safe_default_codec", lambda: "pyav")
    h, w = (8, 10) if resized else (None, None)
    if backend == "pyav_resize" and not resized:
        with pytest.raises(ValueError):
            video_utils.decode_video_frames("clip", [1.0], 1e-4, backend, h, w)
        assert not calls
        return
    expected = "torchcodec" if backend == "torchcodec" else "pyav_resized" if resized else "torchvision"
    assert video_utils.decode_video_frames("clip", [1.0], 1e-4, backend, h, w) == expected
    assert len(calls) == 1
    assert calls[0][1][:3] == ("clip", [1.0], 1e-4)


def test_resize_backend_real_video_preserves_frame_selection(tmp_path):
    import av
    import numpy as np

    from lerobot.datasets.video_utils import FrameTimestampError, decode_video_frames

    path = tmp_path / "frames.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = 16, 16, "yuv420p"
        for index in range(12):
            frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), index * 20, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    times = [7 / 30, 2 / 30, 7 / 30, 11 / 30]
    original = decode_video_frames(path, times, 1e-4, "pyav")
    resized = decode_video_frames(path, times, 1e-4, "pyav_resize", 16, 16)
    assert torch.equal((original * 255).to(torch.uint8), resized)
    # Existing video-generation resize route must keep its pixels and uint8 contract.
    existing = decode_video_frames(path, times, 1e-4, "pyav", 8, 8)
    assert torch.equal(existing, decode_video_frames(path, times, 1e-4, "pyav_resize", 8, 8))
    for backend in ("pyav", "pyav_resize"):
        with pytest.raises(FrameTimestampError):
            decode_video_frames(
                path,
                [0.015],
                1e-4,
                backend,
                8 if backend == "pyav_resize" else None,
                8 if backend == "pyav_resize" else None,
            )


@pytest.mark.parametrize("backend", ["pyav", "pyav_resize"])
def test_new_layout_only_changes_declaration(tmp_path, monkeypatch, backend):
    from cosmos_framework.data.generator.action import video_view as module
    from lerobot.datasets import video_utils

    # Wrist row first: only a layout declaration changes, not either backend.
    layout = module._VideoLayout(
        rows=(("left", "right"), ("head",)),
        resize_targets=module._concat_resize_targets,
    )
    monkeypatch.setattr(module, "_get_layout", lambda viewpoint: layout)
    view = VideoViewConfig(cameras={"head": "front", "left": "wrist", "right": "wrist"})
    for key in ("front", "wrist"):
        (tmp_path / f"{key}.mp4").touch()
    meta = SimpleNamespace(
        features={"front": {"names": ["height", "width", "rgb"], "shape": [8, 11, 3]}},
        episodes=[{"videos/front/from_timestamp": 0, "videos/wrist/from_timestamp": 1}],
        get_video_file_path=lambda episode, key: f"{key}.mp4",
    )
    decode_calls, query_calls = [], []

    def query(timestamps, episode):
        query_calls.append(timestamps)
        # Metadata is irrelevant to the legacy path; actual head dimensions win.
        return {key: torch.full((2, 3, 8, 11), 0.25 if key == "front" else 0.75) for key in timestamps}

    def decode(path, timestamps, tolerance, *, backend, resize_h, resize_w):
        decode_calls.append((path.stem, resize_h, resize_w))
        return torch.full((2, 3, resize_h, resize_w), 64 if path.stem == "front" else 192, dtype=torch.uint8)

    monkeypatch.setattr(video_utils, "decode_video_frames", decode)
    ds = SimpleNamespace(root=tmp_path, meta=meta, video_backend=backend, tolerance_s=1e-4, _query_videos=query)
    if backend == "pyav":
        meta.features = {}  # Legacy must not start depending on head metadata.
    else:

        def forbidden(*args, **kwargs):
            pytest.fail("PyAV preparation and composition must not call legacy resize")

        monkeypatch.setattr(module.F, "interpolate", forbidden)
    result = view.read(ds, 0, [0, 1 / 30], viewpoint="wrists_above_head")
    assert result.shape == (3, 2, 12, 11)
    assert (result[:, :, :4] == (0.75 if backend == "pyav" else 192)).all()
    assert (result[:, :, 4:] == (0.25 if backend == "pyav" else 64)).all()
    if backend == "pyav":
        assert len(query_calls) == 1 and set(query_calls[0]) == {"front", "wrist"}
    else:
        # Same camera field, different sizes: do not reuse one resized tensor for both wrists.
        assert decode_calls == [("wrist", 4, 5), ("wrist", 4, 6), ("front", 8, 11)]


@pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
def test_pure_composition_never_resizes_or_casts(monkeypatch, dtype):
    from cosmos_framework.data.generator.action import video_view as module

    def forbidden(*args, **kwargs):
        pytest.fail("Pure composition must not resize")

    monkeypatch.setattr(module.F, "interpolate", forbidden)
    frames = {
        "head": torch.full((2, 3, 8, 11), 2, dtype=dtype),
        "left": torch.full((2, 3, 4, 5), 3, dtype=dtype),
        "right": torch.full((2, 3, 4, 6), 4, dtype=dtype),
    }
    output = module._compose_prepared(frames, 2, module._get_layout("concat_view"))
    assert output.dtype == dtype
    assert torch.equal(output[:, :, :8], frames["head"].permute(1, 0, 2, 3))
    assert torch.equal(output[:, :, 8:, :5], frames["left"].permute(1, 0, 2, 3))
    assert torch.equal(output[:, :, 8:, 5:], frames["right"].permute(1, 0, 2, 3))


def test_legacy_resizes_declared_roles_even_at_target_size(monkeypatch):
    from cosmos_framework.data.generator.action import video_view as module

    view = VideoViewConfig(cameras={role: role for role in ("head", "left", "right")})
    frames = {"head": torch.rand(2, 3, 8, 11), "left": torch.rand(2, 3, 4, 5), "right": torch.rand(2, 3, 4, 6)}
    original = module.F.interpolate
    calls = []

    def record(tensor, **kwargs):
        calls.append((tensor, kwargs))
        return original(tensor, **kwargs)

    monkeypatch.setattr(module.F, "interpolate", record)
    view.compose(frames, 2, viewpoint="concat_view")
    assert len(calls) == 2
    for (tensor, kwargs), role, size in zip(calls, ("left", "right"), ((4, 5), (4, 6))):
        assert tensor is frames[role]
        assert kwargs == dict(size=size, mode="bilinear", align_corners=False)


@pytest.mark.parametrize("error", ["missing", "count", "channels", "height", "width", "zero", "dtype"])
def test_prepared_layout_rejects_invalid_inputs(error):
    from cosmos_framework.data.generator.action import video_view as module

    frames = {"head": torch.zeros(2, 3, 8, 11), "left": torch.zeros(2, 3, 4, 5), "right": torch.zeros(2, 3, 4, 6)}
    if error == "missing":
        del frames["left"]
    elif error == "dtype":
        frames["left"] = frames["left"].to(torch.uint8)
    else:
        shape = {
            "count": (1, 3, 4, 5),
            "channels": (2, 1, 4, 5),
            "height": (2, 3, 3, 5),
            "width": (2, 3, 4, 4),
            "zero": (2, 3, 0, 5),
        }[error]
        frames["left"] = torch.zeros(shape)
    with pytest.raises(TypeError if error == "dtype" else ValueError):
        module._compose_prepared(frames, 2, module._get_layout("concat_view"))
