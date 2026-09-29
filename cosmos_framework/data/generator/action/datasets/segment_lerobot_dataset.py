# SPDX-License-Identifier: OpenMDW-1.1
"""Template-independent, variable-length LeRobot segment reader."""

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.data.generator.action.action_state_template import ActionStateTemplate, TemplateSourceContract
from cosmos_framework.data.generator.action.datasets.cosmos3_action_lerobot import (
    BaseActionLeRobotDataset,
    LeRobotDatasetMetadata,
    split_episode_ids,
)
from cosmos_framework.data.generator.action.sample_contract import ActionReadOptions
from cosmos_framework.data.generator.action.segment_planner import SegmentPlanner
from cosmos_framework.data.generator.action.video_view import VideoViewConfig


class SegmentLeRobotDataset(BaseActionLeRobotDataset):
    """一个实例对应一个 LeRobot 根目录；返回尚未编码、归一化的绝对量。

    连续读取源数据时间步，保持源 FPS；视频专用抽帧由后续 block 处理执行。
    相机组合通过 video_view 配置；未配置时只读取数值。
    """

    def __init__(
        self,
        *,
        root: str | Path,
        template: ActionStateTemplate,
        source_contract: TemplateSourceContract,
        planner: SegmentPlanner,
        read_options: ActionReadOptions = ActionReadOptions(),
        split: str = "full",
        split_seed: int = 0,
        split_val_ratio: float = 0.0,
        tolerance_s: float = 1e-4,
        video_backend: str | None = None,
        video_view: VideoViewConfig | None = None,
        viewpoint: str | None = None,
    ):
        meta = LeRobotDatasetMetadata(repo_id="local", root=root, revision="local")
        self._pyav_resize = video_backend == "pyav_resize"
        self.video_view = video_view
        if video_view is not None:
            video_view.validate_features(meta.features, viewpoint=viewpoint)
        fps = float(meta.fps)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Source FPS must be finite and positive")
        if source_contract.fps and not math.isclose(source_contract.fps, fps):
            raise ValueError("Source contract FPS must match the aligned observation rate")
        super().__init__(
            fps=fps,
            chunk_length=planner.max_action_steps,
            split_seed=split_seed,
            split_val_ratio=split_val_ratio,
            split=split,
            mode=None,  # Reader 只读取数据，FD/ID/Policy 由后续训练适配层选择。
            embodiment_type=None,
            viewpoint=viewpoint,
            tolerance_s=tolerance_s,
        )
        self.template = template
        self.read_options = read_options
        self.source_contract = replace(source_contract, fps=fps, split=self.split)
        self._episode_fps = self._load_episode_fps(Path(root), meta.total_episodes)
        # 索引计数属于当前 Reader，不修改调用方或其他 Reader 的 planner。
        self.planner = SegmentPlanner(
            max_action_steps=planner.max_action_steps,
            overlap_action_steps=planner.overlap_action_steps,
            geometry=planner.geometry,
        )
        self._segments: list[tuple[int, int, int, int]] = []  # ds、episode、源表起点、action 数
        self._masks: tuple[torch.Tensor, torch.Tensor] | None = None
        self._register_source(
            root=str(root),
            delta_timestamps={},
            tolerance_s=tolerance_s,
            video_backend=video_backend,
            prefetched_meta=meta,
        )

    def _load_episode_fps(self, root, total_episodes):
        """episode ID 从 0 连续编号；只缓存 float32 FPS，不保留 JSONL 内容。"""
        values = np.full(total_episodes, np.nan, dtype=np.float32)
        path = root / "meta" / "episodes.jsonl"
        if path.is_file():
            with path.open() as file:
                for line in file:
                    if not line.strip():
                        continue
                    episode = json.loads(line)
                    if "source_fps" not in episode:
                        continue
                    episode_id = episode["episode_index"]
                    if type(episode_id) is not int or not 0 <= episode_id < total_episodes:
                        raise ValueError(f"episode_index out of range in {path}: {episode_id}")
                    fps = episode["source_fps"]
                    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
                        raise ValueError(f"Invalid source_fps for episode {episode_id} in {path}: {fps}")
                    values[episode_id] = fps
        values.setflags(write=False)
        return values

    def _read_source_fps(self, episode_id):
        """按 episode ID 直接索引；NaN 表示未声明，回退到 meta.fps。"""
        fps = self._episode_fps[episode_id]
        return self.fps if np.isnan(fps) else float(fps)

    def _append_index_records(self, *, meta, ds_idx, dataset_label=None):
        """先排除目标偏移导致的越界，再在对齐网格上规划片段。"""
        options = self.read_options
        for key in {options.state_key, options.target_key, options.state_mask_key, options.target_mask_key}:
            if key not in meta.features:
                raise ValueError(f"Missing required field {key!r} in {meta.root}")
        for key in {options.state_key, options.target_key, options.state_mask_key, options.target_mask_key}:
            if tuple(meta.features[key]["shape"]) != (self.template.width,):
                raise ValueError(f"Field {key!r} must declare shape [{self.template.width}]")
        episode_ids = split_episode_ids(
            total_episodes=meta.total_episodes,
            seed=self._split_seed,
            val_ratio=self._split_val_ratio,
            split=self._split,
        )
        offset = options.action_time_offset_steps
        for episode_id in episode_ids:
            ep = meta.episodes[episode_id]
            begin, end = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
            count = end - begin
            # 目标只对应前 T 个区间；offset=1 恰好使用已有的终点 observation。
            first = max(0, -offset)
            stop = max(first, min(count, count + 1 - offset))
            planned = self.planner.plan(
                stop - first,
                observation_start=first,
                source_id=str(meta.root),
                episode_id=episode_id,
            )
            for start, actions in planned:
                self._segments.append((ds_idx, episode_id, begin + start, actions))
            if planned:
                self._episode_cum_ends.append(len(self._segments))
        self._num_valid_indices = len(self._segments)

    def _resolve_index(self, idx):
        """新索引空间的一项就是一段；不再把 idx 解释成滑窗起点。"""
        return self._segments[idx]

    def _read_masks(self, ds, row):
        """每个子数据集从一行加载 mask，缓存后不再逐帧查询。"""
        if self._masks is None:
            keys = {self.read_options.state_mask_key, self.read_options.target_mask_key}
            values = ds._query_hf_dataset({key: [row] for key in keys})
            masks = []
            for key in (self.read_options.state_mask_key, self.read_options.target_mask_key):
                mask = self.template.validate_valid_mask(values[key][0]).clone()
                self.template.validate_source_contract(self.source_contract, mask)
                masks.append(mask)
            self._masks = tuple(masks)
        return self._masks

    def _compute_idle_frames(self, raw_action):
        """绝对目标尚未编码，不能套用旧 action 的静止检测规则。"""
        return None

    def _read_video(self, ds, episode_id, timestamps, *, viewpoint):
        """按配置读取并组合视角，不做时间下采样。"""
        return (
            self.video_view.read(ds, episode_id, timestamps, viewpoint=viewpoint)
            if self.video_view is not None
            else None
        )

    def _convert_video(self, video_tchw):
        """The opt-in resized reader already produces uint8; preserve legacy validation otherwise."""
        if not self._pyav_resize:
            return super()._convert_video(video_tchw)
        if self._skip_video_loading or video_tchw is None:
            return None
        if video_tchw.ndim != 4 or video_tchw.shape[1] != 3:
            raise ValueError("pyav_resize expected video with shape [T,3,H,W]")
        if video_tchw.dtype != torch.uint8:
            raise TypeError("pyav_resize expected uint8 video")
        return video_tchw.permute(1, 0, 2, 3)

    def __getitem__(self, idx):
        """按实际长度批量读取，目标偏移只在这里执行一次。"""
        ds_idx, episode_id, start, actions = self._resolve_index(idx)
        # 按当前 episode 获取原始 FPS；只有字段缺失时才回退到 meta.fps。
        source_fps = self._read_source_fps(episode_id)
        ds = self._get_dataset(ds_idx)
        ds._ensure_hf_dataset_loaded()
        options = self.read_options
        rows = list(range(start, start + actions + 1))
        target_rows = [row + options.action_time_offset_steps for row in rows[:-1]]
        # 不经过 LeRobot 的边界 clamp，也不改共享 delta_timestamps。
        observations = ds._query_hf_dataset({options.state_key: rows})
        if options.action_from_state and options.action_time_offset_steps in (0, 1):
            offset = options.action_time_offset_steps
            target = observations[options.state_key][offset : offset + actions]
        else:
            target = ds._query_hf_dataset({options.target_key: target_rows})[options.target_key]
        state_mask, action_mask = self._read_masks(ds, rows[0])
        # 绕开 LeRobot 默认 torch.tensor(float) 的 float32 转换，保留长轨迹时间精度。
        timestamps = ds.hf_dataset.select_columns(["timestamp"]).with_format(None)[rows]["timestamp"]
        times = torch.tensor(timestamps, dtype=torch.float64)
        # 先用存储 FPS 检查连续采样，再将输出 FPS 切换到真实训练时间尺度。
        if not torch.allclose(times[1:] - times[:-1], torch.full_like(times[1:], 1 / self.fps), atol=1e-5, rtol=1e-4):
            raise ValueError(f"Episode {episode_id}: timestamp intervals must match meta.fps={self.fps}")
        # action 时间标记区间起点；目标实际读取时刻由 read_options 声明。
        task = ds._query_hf_dataset({"task_index": [rows[0]]})["task_index"][0].item()
        # 当前使用默认布局；后续布局增强在此选择本次样本的 viewpoint。
        viewpoint = self._viewpoint
        video = self._read_video(ds, episode_id, times.tolist(), viewpoint=viewpoint)
        # 沿用父类字典与 uint8 视频格式；action 暂为绝对目标，C06 再写入编码结果。
        sample = self._build_result(
            mode=None,
            video=video.permute(1, 0, 2, 3) if video is not None else None,
            action=target,
            state_trajectory=observations[options.state_key],
            action_target=target,
            state_mask=state_mask.clone(),
            action_mask=action_mask.clone(),
            state_timestamps=times,
            action_timestamps=times[:-1].clone(),
            action_state_indexes=torch.arange(actions),
            source_contract=self.source_contract,
            # 存储时间戳仍搭配 storage_fps；训练 FPS 在读取完成后统一切换。
            storage_fps=torch.tensor(self.fps, dtype=torch.float32),
            conditioning_fps=torch.tensor(source_fps, dtype=torch.float32),
            source_fps=torch.tensor(source_fps, dtype=torch.float32),
            read_options=options,
            ai_caption=str(ds.meta.tasks.iloc[int(task)].name),
            viewpoint=viewpoint,
            additional_view_description=self.video_view.describe(viewpoint=viewpoint) if self.video_view else "",
        )
        return sample
