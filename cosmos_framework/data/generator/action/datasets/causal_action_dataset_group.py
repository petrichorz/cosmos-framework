# SPDX-License-Identifier: OpenMDW-1.1
"""Map group-level segment indexes to independently encoded source datasets."""

from torch.utils.data import ConcatDataset

from .causal_action_sft_dataset import MODES


class CausalActionDatasetGroup(ConcatDataset):
    """共用一条来源配置；累计长度定位子集，不合并 episode 或样本内容。"""

    def __init__(self, datasets):
        super().__init__(datasets)
        self.template = self.datasets[0].template
        if any(
            (d.template.template_id, d.template.width) != (self.template.template_id, self.template.width)
            for d in self.datasets
        ):
            raise ValueError("Grouped datasets must use the same action template")

    def get_shuffle_blocks(self):
        """将各子集的 episode 范围转换为组内索引，供 mixture 统一分片。"""
        blocks = []
        for offset, dataset in zip([0, *self.cumulative_sizes[:-1]], self.datasets):
            blocks.extend((offset + start, length) for start, length in dataset.get_shuffle_blocks())
        return blocks

    def set_mode(self, mode):
        """推理显式指定任务时，覆盖所有子集的训练任务设置。"""
        if mode not in (*MODES, "joint"):
            raise ValueError(f"Unsupported causal action mode: {mode}")
        for dataset in self.datasets:
            dataset.mode = mode
