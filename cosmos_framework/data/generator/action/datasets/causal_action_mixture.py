# SPDX-License-Identifier: OpenMDW-1.1
"""Mix source contracts before max_tokens packing on every training rank."""

import math
import random

import torch
from torch.utils.data import IterableDataset, get_worker_info


class CausalActionMixture(IterableDataset):
    """Weighted sample-level mixture of transformed, map-style causal sources.

    Place this as one dataset in RankPartitionedDataLoader so a packed batch
    can contain different robots/valid slots even on a single rank. Every
    source retains its own adapter and training statistics. Segments are independent
    training units; sharding does not require one whole episode per worker.
    """

    def __init__(self, datasets, weights=None, seed=42):
        self.datasets = list(datasets)
        self.weights = [1.0] * len(self.datasets) if weights is None else list(weights)
        if not self.datasets or len(self.weights) != len(self.datasets):
            raise ValueError("Mixture requires one weight per source")
        if any(not math.isfinite(w) or w <= 0 for w in self.weights):
            raise ValueError("Source weights must be finite and positive")
        if any(isinstance(d, IterableDataset) for d in self.datasets):
            raise ValueError("Set iterable_shuffle=False for sources; mixture owns sharding")
        templates = {(d.template.template_id, d.template.width) for d in self.datasets}
        if len(templates) != 1:
            raise ValueError("Mixed sources must use the same action template")
        self.seed = seed
        self.shard_rank, self.shard_world_size = 0, 1

    def __len__(self):
        # Nominal epoch size for PackingDataLoader accounting; iteration cycles.
        return sum(len(dataset) for dataset in self.datasets)

    def __iter__(self):
        worker = get_worker_info()
        wid, nw = (worker.id, worker.num_workers) if worker else (0, 1)
        rng = random.Random(self.seed + self.shard_rank * nw + wid)
        streams = []
        shard = self.shard_rank * nw + wid
        total_shards = self.shard_world_size * nw
        for i, dataset in enumerate(self.datasets):
            if len(dataset) < total_shards:
                raise ValueError("Each source needs at least one segment per rank/worker; reduce workers or ranks")
            streams.append(self._source_stream(dataset, self.seed + i, shard, total_shards))
        while True:
            index = rng.choices(range(len(streams)), weights=self.weights, k=1)[0]
            yield next(streams[index])

    @staticmethod
    def _source_stream(dataset, seed, shard, total_shards):
        # 各 rank/worker 共用 episode 顺序，再按片段序号分片；短数据源也不复制样本。
        blocks = dataset.get_shuffle_blocks()
        epoch = 0
        while True:
            order = torch.randperm(len(blocks), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
            position = 0
            for block in order:
                start, length = blocks[block]
                first = start + (shard - position) % total_shards
                for index in range(first, start + length, total_shards):
                    yield dataset[index]
                position += length
            epoch += 1
