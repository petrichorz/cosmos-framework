# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Small simulator-side client; requires only websockets, msgpack and NumPy."""

from cosmos_framework.inference.causal_action.wire import packb, unpackb


class H264VideoWriter:
    """Stream RGB frames to FFmpeg's libx264 encoder without buffering a video."""

    def __init__(self, path, *, fps):
        from pathlib import Path

        import numpy as np

        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Video FPS must be finite and positive")
        self.path = Path(path) if path is not None else None
        self.fps = float(fps)
        self.frames = 0
        self.writer = self.errors = self.shape = None

    def __enter__(self):
        return self

    def append(self, frame):
        if self.path is None:
            return
        import subprocess
        import tempfile

        import numpy as np

        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3 or min(frame.shape[:2]) < 2:
            raise ValueError("Video recording requires uint8 RGB [H,W,3] frames")
        height, width = frame.shape[:2]
        frame = np.pad(frame, ((0, height % 2), (0, width % 2), (0, 0)))
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.shape = frame.shape
            self.errors = tempfile.TemporaryFile()
            try:
                self.writer = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        "-s",
                        f"{frame.shape[1]}x{frame.shape[0]}",
                        "-r",
                        str(self.fps),
                        "-i",
                        "pipe:0",
                        "-an",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "fast",
                        "-crf",
                        "18",
                        "-threads",
                        "2",
                        "-pix_fmt",
                        "yuv420p",
                        "-tag:v",
                        "avc1",
                        "-movflags",
                        "+faststart",
                        str(self.path),
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=self.errors,
                )
            except OSError as error:
                self.errors.close()
                raise RuntimeError("H.264 video saving requires ffmpeg with the libx264 encoder on PATH") from error
        if frame.shape != self.shape:
            raise ValueError("Frame dimensions changed during video recording")
        try:
            self.writer.stdin.write(frame.tobytes())
        except BrokenPipeError:
            self.close()
            raise RuntimeError(f"FFmpeg stopped accepting frames: {self.path}") from None
        self.frames += 1

    def close(self):
        if self.writer is None:
            return
        writer, self.writer = self.writer, None
        try:
            writer.stdin.close()
        except BrokenPipeError:
            pass
        status = writer.wait()
        self.errors.seek(0)
        detail = self.errors.read().decode(errors="replace").strip()
        self.errors.close()
        if status:
            raise RuntimeError(f"H.264 encoding failed for {self.path}: {detail}")

    def __exit__(self, *args):
        self.close()


def save_structured_prompt(response, path):
    """Save the server's full prompt when included in the first inference response."""
    if "structured_prompt" not in response:
        return
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(response["structured_prompt"], ensure_ascii=False, indent=2))


def save_prediction_video(response, path):
    """Save one server RGB preview as H.264/AVC in an MP4 container."""
    if "video" not in response:
        return {}  # Allow clients to keep working with earlier v2 servers.
    import numpy as np

    frames = response["video"]
    fps = response["video_fps"]
    if (
        not isinstance(frames, np.ndarray)
        or frames.dtype != np.uint8
        or frames.ndim != 4
        or frames.shape[0] < 1
        or frames.shape[-1] != 3
        or min(frames.shape[1:3]) < 2
    ):
        raise ValueError("Server video must be RGB uint8 [T,H,W,3]")
    with H264VideoWriter(path, fps=fps) as writer:
        for frame in frames:
            writer.append(frame)
    return dict(prediction_video_path=str(path), prediction_video_frames=len(frames), prediction_video_fps=float(fps))


def append_prediction_video(writer, response, *, action_steps, video_stride):
    """Join the executed horizon of each prediction without overlapping time."""
    if "video" not in response or writer.path is None:
        return
    if response["video_fps"] != writer.fps:
        raise ValueError("Prediction video FPS differs from connection metadata")
    for frame in response["video"][: action_steps // video_stride]:
        writer.append(frame)


class CausalPolicyClient:
    def __init__(self, uri="ws://localhost:8000", *, control_mode="joint"):
        from websockets.sync.client import connect

        self.connection = connect(uri, compression=None, max_size=None, ping_interval=None)
        self.metadata = unpackb(self.connection.recv())
        if self.metadata.get("protocol") != "cosmos-causal-action-v2":
            self.close()
            raise ValueError("Expected a cosmos-causal-action-v2 server")
        if self.metadata.get("quaternion_order") != "xyzw" or self.metadata.get("eef_pose_frame") != "dataset":
            self.close()
            raise ValueError("Expected dataset-frame xyz+xyzw EEF poses; update the server and clients together")
        if control_mode not in self.metadata["available_control_modes"]:
            self.close()
            raise ValueError(f"Server does not support control_mode={control_mode!r}")
        self.control_mode = control_mode

    def infer(self, observation):
        self.connection.send(packb(observation))
        response = unpackb(self.connection.recv())
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

    def reset(self, *, prompt, record_video=False):
        return self.infer(dict(reset=True, prompt=prompt, control_mode=self.control_mode, record_video=record_video))

    def save_prediction_video(self, path):
        """Request one whole-sequence VAE decode and save the server-encoded MP4."""
        from pathlib import Path

        result = self.infer(dict(finish_video=True))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(result["video_mp4"])
        return dict(
            prediction_video_path=str(path),
            prediction_video_frames=result["video_frames"],
            prediction_video_fps=result["video_fps"],
        )

    def actions(self, response):
        """Select packed joint or dataset-frame xyz+xyzw EEF actions."""
        actions = response["action"][self.control_mode]
        if actions is None:
            raise ValueError(f"Server returned no valid {self.control_mode} actions")
        return actions

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
