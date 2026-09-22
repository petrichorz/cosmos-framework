# SPDX-License-Identifier: OpenMDW-1.1
"""Mix source contracts before max_tokens packing on every training rank."""

import math
import random

from torch.utils.data import IterableDataset, get_worker_info

from cosmos_framework.data.generator.action.datasets.action_sft_dataset import ActionIterableShuffleDataset


class CausalActionMixture(IterableDataset):
    """Weighted sample-level mixture of transformed, map-style causal sources.

    Place this as one dataset in RankPartitionedDataLoader so a packed batch
    can contain different robots/valid slots even on a single rank. Every
    source retains its own adapter, training statistics and geometry sampling.
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
        for i, dataset in enumerate(self.datasets):
            if len(dataset.get_shuffle_blocks()) < self.shard_world_size * nw:
                raise ValueError("Each source needs at least one episode per rank/worker; reduce workers or ranks")
            stream = ActionIterableShuffleDataset(dataset, self.seed + i)
            stream.shard_rank, stream.shard_world_size = self.shard_rank, self.shard_world_size
            streams.append(iter(stream))
        while True:
            index = rng.choices(range(len(streams)), weights=self.weights, k=1)[0]
            yield next(streams[index])
