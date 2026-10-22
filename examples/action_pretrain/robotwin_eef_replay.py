# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Replay one physical, absolute unified55 EEF episode without a policy server.

Run in the RoboTwin Python environment (NumPy, PyArrow, OpenCV and FFmpeg).
--dataset-root must name a LeRobot root containing meta/info.json and data/,
e.g. /path/to/robotwin_processed/RoboTwin__adjust_bottle/clean.
--prepare-only exports actions and the dataset video without importing RoboTwin/SAPIEN.
Outputs under --output/episode{episode_index:03d}/: action.npy [T,16],
observations.npz (initial state then one state per action), video.mp4,
video_data.mp4 (dataset reference), metrics.json. Simulator seed equals episode_index.
Stops on task success or step limit unless --execute-all-actions is set.

EEF actions are derived next-recorded-pose targets, not original Cartesian
commands. Each take_action call replans a path, so replay has no fixed 30 Hz
physics cadence. Using episode_index as seed is a replay convention; it does not
recover the original collection seed or guarantee the original scene layout.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cosmos_framework.inference.causal_action.client import H264VideoWriter
from cosmos_framework.inference.causal_action.robotwin_client import (
    VideoRecorder,
    camera_images,
    episode_finished,
    load_robotwin,
    native_state,
    positive_int,
)


def unit_quaternion(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.isfinite(q).all() or (norm < 1e-8).any():
        raise ValueError("Quaternion must be finite and nonzero")
    return q / norm


def quaternion_product(a, b):
    """Hamilton product of broadcastable wxyz quaternions."""
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return np.concatenate(
        [aw * bw - (av * bv).sum(axis=-1, keepdims=True), aw * bv + bw * av + np.cross(av, bv)], axis=-1
    )


def load_actions(root, episode_index, *, pose_frame, world_from_base=None):
    """Read unnormalized action_unified rows; no extra temporal shift or stride."""
    import pyarrow.dataset as ds

    info = json.loads((root / "meta/info.json").read_text())
    columns = ["frame_index", "action_unified", "mask_action"]
    table = (
        ds.dataset(root / "data", format="parquet")
        .to_table(columns=columns, filter=ds.field("episode_index") == episode_index)
        .sort_by("frame_index")
    )
    if not table.num_rows:
        raise ValueError(f"Episode {episode_index} is empty or missing in {root}")
    frames = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    if not np.array_equal(frames, np.arange(len(frames))):
        raise ValueError("Episode frame_index must be contiguous, starting at zero")
    action = np.asarray(table["action_unified"].to_pylist(), dtype=np.float64)
    mask = np.asarray(table["mask_action"].to_pylist(), dtype=np.float64)
    if action.shape != (len(frames), 55) or mask.shape != action.shape:
        raise ValueError("action_unified and mask_action must have shape [T,55]")
    slots = np.r_[14:28, 28:30]
    if not np.isfinite(mask[:, slots]).all() or not (mask[:, slots] == 1).all():
        raise ValueError("Both EEFs [14:28] and grippers [28:30] must be valid in every action row")
    if not np.isfinite(action[:, slots]).all():
        raise ValueError("EEF/gripper actions must be finite")
    if ((action[:, 28:30] < 0) | (action[:, 28:30] > 1)).any():
        raise ValueError("RoboTwin gripper targets must be normalized to [0,1]")

    if pose_frame == "base":
        if world_from_base is None:
            transform = info.get("custom", {}).get("eef_enrichment", {}).get("T_world_base")
            if transform is None:
                raise ValueError(
                    "Base-frame poses require metadata custom.eef_enrichment.T_world_base or --world-from-base"
                )
            world_from_base = transform["position"] + transform["quaternion_wxyz"]
        transform = np.asarray(world_from_base, dtype=np.float64)
        if transform.shape != (7,) or not np.isfinite(transform).all():
            raise ValueError("world-from-base must be finite [x,y,z,qw,qx,qy,qz]")
        translation, rotation = transform[:3], unit_quaternion(transform[3:])
    else:
        if world_from_base is not None:
            raise ValueError("--world-from-base only applies to --pose-frame base")
        translation, rotation = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

    parts = []
    for start, gripper in ((14, 28), (21, 29)):
        pose = action[:, start : start + 7]
        xyz = pose[:, :3]
        q = unit_quaternion(pose[:, [6, 3, 4, 5]])  # unified xyzw -> RoboTwin wxyz
        # T_world_eef = T_world_base @ T_base_eef, including orientation.
        uv = np.cross(rotation[1:], xyz)
        xyz = xyz + 2 * (rotation[0] * uv + np.cross(rotation[1:], uv)) + translation
        parts.extend([xyz, quaternion_product(rotation, q), action[:, gripper : gripper + 1]])
    return np.concatenate(parts, axis=1), frames, info


class DatasetVideoRecorder(VideoRecorder):
    """Use the simulator video's three-view layout with a separate output path."""

    def __init__(self, path, *, fps):
        H264VideoWriter.__init__(self, path, fps=fps)


def save_dataset_video(root, episode_index, info, output, *, max_frames):
    """Extract this episode from packed LeRobot v3 videos, one frame per source row."""
    import cv2
    import pyarrow.dataset as ds

    episodes = ds.dataset(root / "meta/episodes", format="parquet").to_table(
        filter=ds.field("episode_index") == episode_index
    )
    if episodes.num_rows != 1:
        raise ValueError(f"Expected one video metadata row for episode {episode_index}")
    episode = episodes.to_pylist()[0]
    fps = float(info["fps"])
    count = min(int(episode["length"]), max_frames)
    if count < 1:
        raise ValueError("Dataset episode must contain video frames")
    captures = {}
    path = output / "video_data.mp4"
    try:
        for key in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            video_key = f"observation.images.{key}"
            prefix = f"videos/{video_key}"
            source = root / info["video_path"].format(
                video_key=video_key,
                chunk_index=int(episode[f"{prefix}/chunk_index"]),
                file_index=int(episode[f"{prefix}/file_index"]),
            )
            capture = cv2.VideoCapture(str(source))
            captures[key] = capture
            if not capture.isOpened():
                raise ValueError(f"Cannot open dataset video: {source}")
            source_fps = capture.get(cv2.CAP_PROP_FPS)
            if not np.isclose(source_fps, fps):
                raise ValueError(f"Dataset video FPS {source_fps} differs from action FPS {fps}: {source}")
            begin = round(float(episode[f"{prefix}/from_timestamp"]) * fps)
            end = round(float(episode[f"{prefix}/to_timestamp"]) * fps)
            if end - begin < count or not capture.set(cv2.CAP_PROP_POS_FRAMES, begin):
                raise ValueError(f"Cannot select {count} episode frames from {source}")
        with DatasetVideoRecorder(path, fps=fps) as video:
            for _ in range(count):
                images = {}
                for key, capture in captures.items():
                    ok, frame = capture.read()
                    if not ok:
                        raise ValueError(f"Dataset video ended before all {count} episode frames were read: {key}")
                    images[key] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video.append(images)
    finally:
        for capture in captures.values():
            capture.release()
    return dict(video_data_path=str(path), video_data_frames=count, video_data_fps=fps)


def pose_errors(target, observed):
    position = np.linalg.norm(target[:, :3] - observed[:, :3], axis=-1)
    dot = np.sum(unit_quaternion(target[:, 3:7]) * unit_quaternion(observed[:, 3:7]), axis=-1)
    angle = 2 * np.arccos(np.clip(np.abs(dot), 0, 1))
    return position, angle


def replay(env, actions, frames, output, *, video_fps, execute_all_actions=False):
    """Save initial observation at row 0; action i produces observation i+1."""
    eef, measured, command, success = [], [], [], []
    executed = 0
    first_success_step = None
    success_seen = bool(env.eval_success)
    if execute_all_actions:
        original_check_success = env.check_success
        had_check_success_override = "check_success" in vars(env)
        original_step_lim = env.step_lim
        replay_step_lim = max(env.step_lim, env.take_action_cnt + len(actions))

        def check_success_without_stopping():
            nonlocal success_seen
            # Still evaluate the real task condition at every physics step.
            success_seen = bool(original_check_success()) or success_seen
            return False

        # take_action checks eval_success at entry and check_success in its loop.
        env.check_success = check_success_without_stopping
        env.eval_success = False
        env.step_lim = replay_step_lim

    def capture(video):
        nonlocal first_success_step
        observation = env.get_obs()
        images = camera_images(observation)
        video.append(images)
        state = native_state(observation, env, state_source="measured")
        eef.append(np.concatenate([observation["endpose"][f"{side}_endpose"] for side in ("left", "right")]))
        # Real arm qpos; gripper getter is still a command value, so exclude it here.
        measured.append(np.concatenate([state["joint_action"][f"{side}_arm"] for side in ("left", "right")]))
        command.append(np.asarray(observation["joint_action"]["vector"]).copy())
        achieved = success_seen if execute_all_actions else bool(env.eval_success)
        success.append(achieved)
        if achieved and first_success_step is None:
            first_success_step = executed

    try:
        with VideoRecorder(output, fps=video_fps) as video:
            capture(video)
            for action in actions:
                if not execute_all_actions and episode_finished(env):
                    break
                env.take_action(action.copy(), action_type="ee")
                executed += 1
                capture(video)
    finally:
        if execute_all_actions:
            if had_check_success_override:
                env.check_success = original_check_success
            else:
                del env.check_success
            env.step_lim = original_step_lim
            env.eval_success = success_seen
        np.savez_compressed(
            output / "observations.npz",
            eef_world_wxyz=np.asarray(eef),  # [N+1,14]: left pose, right pose
            measured_arm_qpos=np.asarray(measured),
            joint_command=np.asarray(command),  # includes commanded grippers
            eval_success=np.asarray(success),
            executed_action=actions[:executed],
            source_frame_index=frames[:executed],
        )
    error_metrics = {}
    observed = np.asarray(eef)[1:]
    if executed:
        for side, target_start, observed_start in (("left", 0, 0), ("right", 8, 7)):
            position, angle = pose_errors(
                actions[:executed, target_start : target_start + 7], observed[:, observed_start : observed_start + 7]
            )
            error_metrics[side] = dict(
                position_mean_m=float(position.mean()),
                position_max_m=float(position.max()),
                rotation_mean_rad=float(angle.mean()),
                rotation_max_rad=float(angle.max()),
            )
    stop_reason = "actions_exhausted"
    if not execute_all_actions:
        if env.eval_success:
            stop_reason = "success"
        elif episode_finished(env):
            stop_reason = "step_limit"
    return dict(
        executed_steps=executed,
        observation_frames=len(eef),
        eval_success=bool(env.eval_success),
        first_success_step=first_success_step,
        stop_reason=stop_reason,
        tracking_error=error_metrics,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--episode-index", type=int, default=0, help="Dataset episode and simulator seed (default: 0)")
    parser.add_argument("--robotwin-root", help="Required unless --prepare-only")
    parser.add_argument("--task-name", help="Defaults to meta/info.json custom.source.task")
    parser.add_argument("--task-config", default="demo_clean", help="Use demo_randomized for randomized data")
    parser.add_argument("--pose-frame", choices=["base", "world"], default="base", help="Frame of unified EEF slots")
    parser.add_argument(
        "--world-from-base", nargs=7, type=float, metavar="VALUE", help="Override base transform: xyz+wxyz"
    )
    parser.add_argument("--max-steps", type=positive_int, help="Optional action count limit")
    parser.add_argument(
        "--execute-all-actions",
        action="store_true",
        help="Execute every selected action despite task success; extend the task step limit if needed",
    )
    parser.add_argument("--video-fps", type=positive_int, default=30, help="Playback FPS; does not set physics cadence")
    parser.add_argument(
        "--prepare-only", action="store_true", help="Export native actions and dataset video without RoboTwin"
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/robotwin_eef_replay"))
    args = parser.parse_args()
    seed = args.episode_index
    if not args.prepare_only and args.robotwin_root is None:
        parser.error("--robotwin-root is required for simulator replay")
    root = args.dataset_root.expanduser().resolve()
    output = args.output.expanduser().resolve() / f"episode{args.episode_index:03d}"
    actions, frames, info = load_actions(
        root, args.episode_index, pose_frame=args.pose_frame, world_from_base=args.world_from_base
    )
    args.task_name = args.task_name or info.get("custom", {}).get("source", {}).get("task")
    if not args.task_name and not args.prepare_only:
        parser.error("--task-name is required when dataset metadata does not specify the task")
    if args.max_steps is not None:
        actions, frames = actions[: args.max_steps], frames[: args.max_steps]
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "action.npy", actions)
    summary = dict(
        dataset_root=str(root),
        episode_index=args.episode_index,
        task=args.task_name,
        task_config=args.task_config,
        seed=seed,
        seed_source="episode_index",
        source_pose_frame=args.pose_frame,
        world_from_base=(args.world_from_base or info.get("custom", {}).get("eef_enrichment", {}).get("T_world_base"))
        if args.pose_frame == "base"
        else None,
        action_format="left xyz+wxyz+gripper, right xyz+wxyz+gripper; world frame",
        action_steps=len(actions),
        execute_all_actions=args.execute_all_actions,
        video_fps=args.video_fps,
        status="prepared",
    )
    try:
        summary.update(save_dataset_video(root, args.episode_index, info, output, max_frames=len(actions) + 1))
        if not args.prepare_only:
            helpers = load_robotwin(args.robotwin_root)
            task_args, _ = helpers.load_task_args(
                dict(
                    task_name=args.task_name,
                    task_config=args.task_config,
                    policy_name="eef_replay",
                    ckpt_setting="dataset",
                )
            )
            task_args.update(
                eval_mode=True, eval_video_log=False, eval_video_save_dir=None, render_freq=0, save_data=False
            )
            task_args["camera"].update(collect_head_camera=True, collect_wrist_camera=True)
            task_args["data_type"].update(rgb=True, qpos=True, endpose=True)
            env = helpers.class_decorator(args.task_name)
            env.suc, env.test_num = 0, 0
            try:
                env.setup_demo(now_ep_num=args.episode_index, seed=seed, is_test=True, **task_args)
                summary.update(
                    replay(
                        env,
                        actions,
                        frames,
                        output,
                        video_fps=args.video_fps,
                        execute_all_actions=args.execute_all_actions,
                    )
                )
                summary["status"] = "finished"
            finally:
                helpers.safe_close_env(env)
    except Exception as error:
        summary.update(status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"{summary['status']}: {output}, actions={actions.shape}", flush=True)


if __name__ == "__main__":
    main()
