# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Exercise simulator integration with fakes; never import or run RoboTwin."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from cosmos_framework.inference.causal_action import robotwin_client as runner


class UnStableError(Exception):
    pass


class FakeEnv:
    def __init__(self, *, limit=35, success_at=None):
        self.step_lim = limit
        self.success_at = success_at
        self.eval_success = False
        self.take_action_cnt = 0
        self.executed = []
        self.buffer = np.zeros((2, 3, 3), dtype=np.uint8)
        self.robot = SimpleNamespace(
            left_arm_joints=[None] * 6,
            right_arm_joints=[None] * 6,
            get_left_arm_real_jointState=lambda: [1.0] * 6 + [0.25],
            get_right_arm_real_jointState=lambda: [2.0] * 6 + [0.75],
        )

    def get_obs(self):
        # Real renderers may reuse buffers; later steps must not overwrite history.
        self.buffer.fill(self.take_action_cnt)
        return dict(
            observation={name: dict(rgb=self.buffer) for name in ("head_camera", "left_camera", "right_camera")},
            joint_action=dict(
                left_arm=[self.take_action_cnt] * 6,
                right_arm=[self.take_action_cnt] * 6,
                left_gripper=0.0,
                right_gripper=1.0,
            ),
            endpose=dict(left_endpose=[0, 0, 0, 1, 0, 0, 0], right_endpose=[0, 0, 0, 1, 0, 0, 0]),
        )

    def take_action(self, action, *, action_type):
        assert not runner.episode_finished(self)
        self.executed.append((action.copy(), action_type))
        self.take_action_cnt += 1
        self.eval_success = self.take_action_cnt == self.success_at

    def get_instruction(self):
        return self.instruction

    def set_instruction(self, *, instruction):
        self.instruction = instruction

    def setup_demo(self, **kwargs):
        self.setup_args = kwargs


class FakeClient:
    def __init__(self, uri=None, *, control_mode="joint"):
        self.control_mode = control_mode
        self.metadata = dict(actions_per_block=32, execute_steps=16, left_arm_dof=6, right_arm_dof=6)
        self.requests = []
        self.resets = []
        self.closed = False

    def reset(self, *, prompt):
        self.resets.append(prompt)

    def infer(self, request):
        self.requests.append(request)
        width = 14 if self.control_mode == "joint" else 16
        actions = np.zeros((32, width), dtype=np.float32)
        actions[:16] = np.arange(16)[:, None]
        actions[16:] = 999  # Must be discarded, never executed.
        return dict(action={self.control_mode: actions}, execute_steps=16, server_timing=dict(infer_ms=1.0))

    def actions(self, response):
        return response["action"][self.control_mode]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True


def test_rollout_sends_post_action_history_and_discards_unexecuted_predictions(tmp_path):
    env, client = FakeEnv(), FakeClient()
    env.set_instruction(instruction="Adjust the bottle.")
    result = runner.rollout(env, client, output=tmp_path)
    assert client.resets == ["Adjust the bottle."]
    assert result["executed_steps"] == 35
    assert not result["success"]
    assert [r["observation_step"] for r in client.requests] == [0, 16, 32]
    assert [r["observation_step"] for r in result["requests"]] == [0, 16, 32]
    assert client.requests[0]["obs"]["cam_high"][0, 0, 0] == 0
    for request, start in zip(client.requests[1:], (1, 17)):
        assert request["executed_steps"] == 16
        for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            assert [int(frame[camera][0, 0, 0]) for frame in request["obs"]] == list(range(start, start + 16))
        assert request["state"]["joint_action"]["left_arm"] == [start + 15] * 6
        assert "endpose" in request["state"]
    assert [int(action[0]) for action, _ in env.executed] == list(range(16)) * 2 + [0, 1, 2]
    assert {kind for _, kind in env.executed} == {"qpos"}
    saved = sorted(tmp_path.glob("*.npy"))
    assert len(saved) == 3
    assert np.load(saved[0]).shape == (32, 14)


def test_eef_rollout_stops_on_success_mid_chunk():
    env, client = FakeEnv(success_at=3), FakeClient(control_mode="eef")
    env.set_instruction(instruction="Move.")
    result = runner.rollout(env, client)
    assert result["success"]
    assert result["executed_steps"] == 3
    assert len(client.requests) == 1
    assert {kind for _, kind in env.executed} == {"ee"}
    assert all(action.shape == (16,) for action, _ in env.executed)


def test_measured_state_replaces_joint_values_without_mutating_get_obs():
    env = FakeEnv()
    observation = env.get_obs()
    command = runner.native_state(observation, env)
    measured = runner.native_state(observation, env, state_source="measured")
    assert command["joint_action"]["left_arm"] == [0] * 6
    assert observation["joint_action"]["left_gripper"] == 0.0
    assert measured["joint_action"]["left_arm"] == [1.0] * 6
    np.testing.assert_array_equal(measured["joint_action"]["vector"], [1.0] * 6 + [0.25] + [2.0] * 6 + [0.75])
    for side in ("left", "right"):
        key = f"{side}_endpose"
        np.testing.assert_array_equal(measured["endpose"][key], [0, 0, 0, 0, 0, 0, 1])
        assert observation["endpose"][key] == [0, 0, 0, 1, 0, 0, 0]


def test_client_converts_world_state_and_dataset_actions_with_base_rotation():
    env = FakeEnv(success_at=3)
    env.set_instruction(instruction="Move.")
    client = FakeClient(control_mode="eef")
    converter = runner.RobotwinPoseConverter(
        eef_pose_frame="base",
        T_world_base=dict(position=[1, 2, 3], quaternion_wxyz=[1, 0, 0, 1]),
    )
    # R_world_base is +90 degrees about Z; the observation is world xyz=0, q=identity.
    runner.rollout(env, client, pose_converter=converter)
    for side in ("left", "right"):
        np.testing.assert_allclose(
            client.requests[0]["state"]["endpose"][f"{side}_endpose"],
            [-2, 1, -3, 0, 0, -np.sqrt(0.5), np.sqrt(0.5)],
            atol=1e-6,
        )
    # The second predicted pose has xyz=1 and xyzw=[1,1,1,1]. Both arms must convert.
    action = env.executed[1][0]
    for start in (0, 8):
        np.testing.assert_allclose(action[start : start + 7], [0, 3, 4, 0, 0, np.sqrt(2), np.sqrt(2)], atol=1e-6)
    np.testing.assert_array_equal(action[[7, 15]], [1, 1])


def test_client_loads_dataset_pose_metadata_and_requires_base_transform(tmp_path):
    info = tmp_path / "meta/info.json"
    info.parent.mkdir()
    enrichment = dict(
        unified_pose_frame="shared ALOHA dual-arm URDF root/base frame",
        unified_quaternion="xyzw",
        T_world_base=dict(position=[0, -0.65, 0], quaternion_wxyz=[0.707, 0, 0, 0.707]),
    )
    info.write_text(json.dumps(dict(custom=dict(eef_enrichment=enrichment))))
    converter = runner.RobotwinPoseConverter.from_dataset(tmp_path)
    assert converter.eef_pose_frame == "base"
    pose = [1, 2, 3, 0.5, 0.5, 0.5, 0.5]
    np.testing.assert_allclose(converter.to_world(converter.to_dataset(pose)), pose, atol=1e-6)
    del enrichment["T_world_base"]
    info.write_text(json.dumps(dict(custom=dict(eef_enrichment=enrichment))))
    with pytest.raises(ValueError, match="T_world_base"):
        runner.RobotwinPoseConverter.from_dataset(tmp_path)


def test_invalid_action_chunk_is_rejected_before_simulator_execution():
    env, client = FakeEnv(), FakeClient()
    env.set_instruction(instruction="Move.")
    client.actions = lambda response: np.full((32, 14), np.nan)
    with pytest.raises(ValueError, match="invalid native action"):
        runner.rollout(env, client)
    assert not env.executed


@pytest.mark.parametrize("ending", ["limit", "success", "error"])
def test_saved_video_contains_every_observation_and_is_closed_on_exit(tmp_path, ending):
    import cv2

    env = FakeEnv(success_at=3 if ending == "success" else None)
    env.set_instruction(instruction="Move.")
    get_obs = env.get_obs

    def observation():
        obs = get_obs()
        for camera in obs["observation"].values():
            camera["rgb"] = np.full((32, 48, 3), env.take_action_cnt * 5, dtype=np.uint8)
        return obs

    env.get_obs = observation
    client = FakeClient()
    infer = client.infer

    def failing_infer(request):
        if request["observation_step"] == 16:
            raise RuntimeError("Inference failed")
        return infer(request)

    if ending == "error":
        client.infer = failing_infer
        with pytest.raises(RuntimeError, match="Inference failed"):
            runner.rollout(env, client, output=tmp_path, video_fps=12)
        count = 17
    else:
        result = runner.rollout(env, client, output=tmp_path, video_fps=12)
        count = 4 if ending == "success" else 36
        assert result["video_frames"] == count
        assert result["video_path"] == str(tmp_path / "video.mp4")
        assert result["video_fps"] == 12

    # Decode the actual MP4: validates codec output, finalization and frame order,
    # including the last partial chunk and frames saved before an exception.
    capture = cv2.VideoCapture(str(tmp_path / "video.mp4"))
    try:
        assert capture.isOpened()
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(12)
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    assert len(frames) == count
    assert frames[0].shape == (48, 48, 3)
    np.testing.assert_allclose([frame[16, 24].mean() for frame in frames], np.arange(count) * 5, atol=4)


def test_video_preserves_rgb_colors_and_different_camera_sizes(tmp_path):
    import cv2

    images = dict(
        cam_high=np.full((33, 65, 3), [255, 0, 0], dtype=np.uint8),
        cam_left_wrist=np.full((31, 33, 3), [0, 255, 0], dtype=np.uint8),
        cam_right_wrist=np.full((29, 35, 3), [0, 0, 255], dtype=np.uint8),
    )
    with runner.VideoRecorder(tmp_path) as recorder:
        recorder.append(images)
    capture = cv2.VideoCapture(str(tmp_path / "video.mp4"))
    try:
        ok, frame = capture.read()
        assert ok
        assert frame.shape == (50, 66, 3)
        # Decoded pixels are BGR; inspect well inside each camera to avoid codec edges.
        np.testing.assert_allclose(frame[16, 32], [0, 0, 255], atol=10)
        np.testing.assert_allclose(frame[40, 16], [0, 255, 0], atol=10)
        np.testing.assert_allclose(frame[40, 50], [255, 0, 0], atol=10)
    finally:
        capture.release()


def test_robotwin_root_is_required_even_with_environment_variable(monkeypatch, capsys):
    monkeypatch.setenv("ROBOTWIN_ROOT", "/path/to/RoboTwin")
    with pytest.raises(SystemExit) as error:
        runner.parser().parse_args([])
    assert error.value.code == 2
    assert "--robotwin-root" in capsys.readouterr().err


@pytest.mark.parametrize("fail_setup", [False, True])
def test_main_loads_config_resets_each_episode_and_always_closes_env(monkeypatch, tmp_path, fail_setup, capsys):
    client, envs, closed = FakeClient(), [], []
    config = dict(camera={}, data_type={})

    def make_env(task):
        assert task == "adjust_bottle"
        env = FakeEnv()
        if fail_setup:

            def fail(**kwargs):
                raise RuntimeError("setup failed")

            env.setup_demo = fail
        envs.append(env)
        return env

    helpers = SimpleNamespace(
        UnStableError=UnStableError,
        load_task_args=lambda args: (config, "aloha-agilex"),
        class_decorator=make_env,
        safe_close_env=lambda env: closed.append(env),
    )
    monkeypatch.setattr(runner, "load_robotwin", lambda root: helpers)
    monkeypatch.setattr(runner, "CausalPolicyClient", lambda *args, **kwargs: client)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "robotwin_client.py",
            "--robotwin-root",
            "/path/to/RoboTwin",
            "--instruction",
            "Adjust.",
            "--seed",
            "42",
            "--episodes",
            "2",
            "--max-steps",
            "2",
            "--output",
            str(tmp_path),
        ],
    )
    if fail_setup:
        with pytest.raises(RuntimeError, match="setup failed"):
            runner.main()
    else:
        runner.main()
        assert [env.setup_args["seed"] for env in envs] == [42, 43]
        assert client.resets == ["Adjust."] * 2
        assert all(env.step_lim == 2 for env in envs)
        assert all(env.setup_args["is_test"] and env.setup_args["eval_mode"] for env in envs)
        assert config["camera"] == dict(collect_head_camera=True, collect_wrist_camera=True)
        assert config["data_type"] == dict(rgb=True, qpos=True, endpose=True)
        assert not config["eval_video_log"] and config["eval_video_save_dir"] is None
    assert closed == envs
    assert client.closed
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert len(metrics["episodes"]) == (1 if fail_setup else 2)
    assert metrics["episodes"][0]["status"] == ("error" if fail_setup else "finished")
    if not fail_setup:
        assert metrics["episodes"][0]["instruction"] == "Adjust."
        assert metrics["episodes"][0]["instruction_source"] == "override"
        for record in metrics["episodes"]:
            requests_path = tmp_path / f"seed{record['seed']}" / "requests.json"
            assert record["requests_path"] == str(requests_path)
            assert "requests" not in record
            requests = json.loads(requests_path.read_text())
            assert [request["observation_step"] for request in requests] == [0]
            assert requests[0]["infer_ms"] == 1.0
        stdout = capsys.readouterr().out
        assert "Episode 0, seed 42" in stdout
        assert "requests" not in stdout
        assert "infer_ms" not in stdout


@pytest.fixture
def automatic_main(monkeypatch, tmp_path):
    client, envs, events = FakeClient(), [], []
    config = dict(task_name="adjust_bottle", eval_instruction="seen", camera={}, data_type={})

    class AutoEnv(FakeEnv):
        def setup_demo(self, **kwargs):
            super().setup_demo(**kwargs)
            self.take_action_cnt, self.eval_success = 0, False
            events.append(("setup", kwargs["seed"], kwargs["render_freq"]))

        def play_once(self):
            events.append(("expert", self.setup_args["seed"]))
            assert self.step_lim == 35  # The policy-only max-steps must not truncate the expert.
            self.take_action_cnt = 35
            return dict(info={"{A}": "bottle", "{a}": "left"})

    def make_env(task):
        env = AutoEnv()
        envs.append(env)
        return env

    def build_instruction(args, episode_info, kind, count, seed):
        assert args["task_name"] == "adjust_bottle"
        assert episode_info["info"] == {"{A}": "bottle", "{a}": "left"}
        assert count == 2
        events.append(("instruction", seed, kind))
        return f"Lift the bottle with the left arm. Seed {seed}."

    helpers = SimpleNamespace(
        UnStableError=UnStableError,
        load_task_args=lambda args: (config, "aloha-agilex"),
        resolve_eval_instruction=lambda args, task: args["instruction_type"] or task["eval_instruction"],
        build_instruction=build_instruction,
        class_decorator=make_env,
        safe_close_env=lambda env: events.append(("close", env.setup_args["seed"])),
    )
    monkeypatch.setattr(runner, "load_robotwin", lambda root: helpers)
    monkeypatch.setattr(runner, "CausalPolicyClient", lambda *args, **kwargs: client)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "robotwin_client.py",
            "--robotwin-root",
            "/path/to/RoboTwin",
            "--seed",
            "42",
            "--episodes",
            "2",
            "--max-steps",
            "2",
            "--render-freq",
            "10",
            "--output",
            str(tmp_path),
        ],
    )
    return client, envs, events, helpers


@pytest.mark.parametrize("instruction_type", [None, "unseen"])
def test_automatic_instruction_uses_episode_metadata_and_recreates_scene(automatic_main, tmp_path, instruction_type):
    client, envs, events, _ = automatic_main
    if instruction_type is not None:
        sys.argv.extend(["--instruction-type", instruction_type])
    runner.main()
    kind = instruction_type or "seen"
    assert events == [
        (event, *values)
        for seed in (42, 43)
        for event, values in (
            ("setup", (seed, 0)),
            ("expert", (seed,)),
            ("instruction", (seed, kind)),
            ("close", (seed,)),
            ("setup", (seed, 10)),
            ("close", (seed,)),
        )
    ]
    assert client.resets == [f"Lift the bottle with the left arm. Seed {seed}." for seed in (42, 43)]
    assert len(client.requests) == 2
    assert all(request["obs"]["cam_high"][0, 0, 0] == 0 for request in client.requests)
    assert all(len(env.executed) == 2 for env in envs)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert [record["instruction"] for record in metrics["episodes"]] == client.resets
    assert all(record["instruction_source"] == "robotwin" for record in metrics["episodes"])
    assert metrics["instruction_type"] == kind


@pytest.mark.parametrize("instruction", ["adjust_bottle", "", None])
def test_failed_automatic_instruction_is_recorded_and_never_sent(automatic_main, tmp_path, instruction):
    client, _, events, helpers = automatic_main
    helpers.build_instruction = lambda *args: instruction
    with pytest.raises(ValueError, match="instruction generation failed"):
        runner.main()
    assert not client.resets and not client.requests
    assert events[-1] == ("close", 42)
    assert client.closed
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["episodes"][0]["status"] == "error"


@pytest.mark.parametrize(
    ("phase", "override"),
    [("setup", False), ("expert", False), ("restore", False), ("rollout", False), ("setup", True), ("rollout", True)],
)
def test_unstable_seeds_are_retried_without_consuming_episodes(automatic_main, tmp_path, phase, override):
    client, envs, events, helpers = automatic_main
    if override:
        sys.argv.extend(["--instruction", "Adjust."])
    make_env = helpers.class_decorator

    def unstable_env(task):
        env = make_env(task)
        setup, expert, take_action = env.setup_demo, env.play_once, env.take_action
        setups = 0

        def fail_if_unstable(current_phase):
            if phase == current_phase and env.setup_args["seed"] in (42, 43):
                raise UnStableError("Objects are unstable")

        def setup_demo(**kwargs):
            nonlocal setups
            setup(**kwargs)
            setups += 1
            fail_if_unstable("setup" if setups == 1 else "restore")

        def play_once():
            fail_if_unstable("expert")
            return expert()

        def execute(action, *, action_type):
            take_action(action, action_type=action_type)
            fail_if_unstable("rollout")

        env.setup_demo, env.play_once, env.take_action = setup_demo, play_once, execute
        return env

    helpers.class_decorator = unstable_env
    runner.main()

    # Two consecutive bad seeds must not exhaust --episodes=2 or be reused.
    assert [env.setup_args["seed"] for env in envs] == [42, 43, 44, 45]
    assert [env.setup_args["now_ep_num"] for env in envs] == [0, 0, 0, 1]
    assert all(("close", seed) in events for seed in (42, 43, 44, 45))
    prompts = ["Adjust."] * 2 if override else [f"Lift the bottle with the left arm. Seed {seed}." for seed in (44, 45)]
    assert client.resets[-2:] == prompts
    assert len(client.resets) == (4 if phase == "rollout" else 2)
    assert client.closed
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert [record["seed"] for record in metrics["episodes"]] == [44, 45]
    assert all(record["status"] == "finished" and record["executed_steps"] == 2 for record in metrics["episodes"])
    assert [record["seed"] for record in metrics["skipped_seeds"]] == [42, 43]
    assert all(record["status"] == "skipped_unstable" for record in metrics["skipped_seeds"])
    assert all("UnStableError" in record["error"] for record in metrics["skipped_seeds"])
    assert (tmp_path / "seed44" / "step000000_action_joint.npy").is_file()
    if phase == "rollout":
        # Failed attempts' predictions must not be overwritten by the retry.
        assert (tmp_path / "seed42" / "step000000_action_joint.npy").is_file()
        import cv2

        for seed, frames in ((42, 1), (43, 1), (44, 3), (45, 3)):
            capture = cv2.VideoCapture(str(tmp_path / f"seed{seed}" / "video.mp4"))
            try:
                assert capture.isOpened()
                assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == frames
            finally:
                capture.release()
