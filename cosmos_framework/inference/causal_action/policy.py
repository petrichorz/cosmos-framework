# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Validated single-episode request lifecycle, shared by serving and replay tests."""

import numpy as np
import torch

from cosmos_framework.inference.causal_action.robotwin import RobotwinAdapter

CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


class CausalPolicy:
    def __init__(self, engine, *, adapter=None, return_video_latents=False, return_unified_action=False):
        self.engine = engine
        self.adapter = adapter or RobotwinAdapter.from_statistics(engine.template, engine.statistics)
        self.state_mask, self.action_mask = self.adapter.state_mask, self.adapter.action_mask
        engine.statistics.validate(engine.template.width, self.state_mask, self.action_mask)
        self.return_video_latents = return_video_latents
        self.return_unified_action = return_unified_action
        self.reset()

    def reset(self):
        self.engine.reset()
        self.prompt = None
        self.control_mode = None
        self.revision = self.observation_step = 0
        self.started = False
        self.needs_reset = False
        self.record_video = False
        self.video_latents = []

    def finish_video(self):
        """Decode and encode the selected prediction sequence once on the model host."""
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from cosmos_framework.inference.causal_action.video import save_latent_video

        if self.needs_reset or not self.record_video or len(self.video_latents) < 2:
            raise ValueError("No recorded prediction sequence; reset with record_video=True before inference")
        latent = torch.cat(self.video_latents, dim=2)
        with TemporaryDirectory(prefix="cosmos-prediction-") as directory:
            path = Path(directory) / "prediction.mp4"
            metrics = save_latent_video(self.engine.model, latent, path)
            result = dict(
                video_mp4=path.read_bytes(), video_frames=metrics["video_frames"], video_fps=metrics["video_fps"]
            )
        self.video_latents.clear()
        self.record_video = False
        return result

    def infer(self, request):
        if not isinstance(request, dict):
            raise ValueError("Request must be a dictionary")
        if "state_mask" in request or "action_mask" in request:
            raise ValueError("State/action masks are configured on the server; do not send masks")
        mode = request.get("control_mode", self.control_mode)
        if mode is not None and mode not in self.adapter.available_control_modes:
            raise ValueError(f"Unsupported control_mode={mode!r}; available: {self.adapter.available_control_modes}")
        if self.started and not request.get("reset", False) and mode != self.control_mode:
            raise ValueError("Changing control_mode requires reset")
        if request.get("reset", False):
            self.reset()
            self.prompt = request.get("prompt")
            self.control_mode = mode
            self.record_video = bool(request.get("record_video", False))
            if "obs" not in request:
                return {"reset": True, "observation_step": 0}
        if self.needs_reset:
            raise ValueError("Previous inference failed; send reset before continuing")
        if request.get("finish_video", False):
            try:
                return self.finish_video()
            except Exception:
                self.reset()
                self.needs_reset = True
                raise
        if "obs" not in request:
            raise ValueError("Each inference needs fresh observations and measured state")
        prompt = request.get("prompt", self.prompt)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A nonempty task prompt is required on reset or first observation")
        if self.started and prompt != self.prompt:
            raise ValueError("Changing the task prompt requires reset")
        template = self.engine.template

        sm, am = self.state_mask, self.action_mask
        state = torch.from_numpy(self.adapter.assemble_state(request.get("state")))
        state = template.sanitize(state, sm)
        count = self.engine.execute_steps if self.started else 1
        obs = request["obs"]
        if isinstance(obs, dict):
            cameras = {key: np.asarray(obs[key]) for key in CAMERA_KEYS}
            cameras = {key: value[None] if value.ndim == 3 else value for key, value in cameras.items()}
        elif isinstance(obs, list) and len(obs) == count:
            cameras = {key: np.stack([frame[key] for frame in obs]) for key in CAMERA_KEYS}
        else:
            raise ValueError(f"obs must be a camera dictionary or a list of {count} camera dictionaries")
        for key, value in cameras.items():
            if (
                value.dtype != np.uint8
                or value.ndim != 4
                or value.shape[0] != count
                or value.shape[-1] != 3
                or min(value.shape[1:3]) < 2
            ):
                raise ValueError(f"{key} must be RGB uint8 [{count},H,W,3]")
        step = self.observation_step + (self.engine.execute_steps if self.started else 0)
        if request.get("observation_step", step) != step:
            raise ValueError(f"Expected observation_step={step}; duplicate/out-of-order feedback")
        if self.started and request.get("executed_steps", self.engine.execute_steps) != self.engine.execute_steps:
            raise ValueError(f"Feedback must follow exactly {self.engine.execute_steps} executed actions")
        try:
            self.engine.observe(cameras, state=state, state_mask=sm, action_mask=am, prompt=prompt)
            result = self.engine.predict(state=state, state_mask=sm, action_mask=am, revision=self.revision)
            action = result["action"]
            if (
                action.shape != (self.engine.geometry.actions_per_block, template.width)
                or not np.isfinite(action).all()
            ):
                raise RuntimeError("Model returned invalid actions")
            video_latents = result["video_latents"]
            if (
                video_latents.ndim != 5
                or video_latents.shape[0] != 1
                or video_latents.shape[2] != self.engine.geometry.latent_frames_per_block
                or not np.isfinite(video_latents).all()
            ):
                raise RuntimeError("Model returned invalid predicted video latents")
            if self.record_video:
                if not self.video_latents:
                    self.video_latents.append(self.engine.latest.detach().float().cpu().clone())
                keep = self.engine.execute_steps // self.engine.geometry.actions_per_latent_frame
                self.video_latents.append(torch.from_numpy(video_latents[:, :, :keep].copy()))
        except Exception:
            # A VAE or CFG branch may have advanced before failure. Never reuse it.
            self.reset()
            self.needs_reset = True
            raise
        self.prompt, self.control_mode = prompt, mode
        self.started, self.observation_step = True, step
        self.revision += 1
        response = dict(
            action=self.adapter.split_actions(action),
            available_control_modes=self.adapter.available_control_modes,
            execute_steps=self.engine.execute_steps,
            observation_step=step,
            revision=self.revision,
        )
        if self.revision == 1:
            response["structured_prompt"] = self.engine.structured_prompt
        if self.return_video_latents:
            response["video_latents"] = result["video_latents"]
        if self.return_unified_action:
            response["action_unified"] = action
        return response
