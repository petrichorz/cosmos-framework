# SPDX-License-Identifier: OpenMDW-1.1
from itertools import islice
from types import SimpleNamespace

import pytest

from cosmos_framework.data.generator.action.datasets.causal_action_mixture import CausalActionMixture


class Source:
    template = SimpleNamespace(template_id="test", width=3)

    def __init__(self, lengths):
        self.blocks = []
        n = 0
        for length in lengths:
            self.blocks.append((n, length))
            n += length
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return index

    def get_shuffle_blocks(self):
        return self.blocks


@pytest.mark.parametrize("lengths", [[17], [3, 7, 11], [1, 1, 1, 7]])
@pytest.mark.parametrize("shards", [1, 4, 8])
def test_segment_sharding_complete_and_disjoint(lengths, shards):
    dataset = Source(lengths)
    all_indexes = []
    for shard in range(shards):
        count = len(range(shard, len(dataset), shards))
        stream = CausalActionMixture._source_stream(dataset, 42, shard, shards)
        indexes = list(islice(stream, count))
        assert len(indexes) == len(set(indexes))
        all_indexes.extend(indexes)
    assert sorted(all_indexes) == list(range(len(dataset)))


def test_small_source_fails_instead_of_repeating():
    mixture = CausalActionMixture([Source([2])])
    mixture.shard_world_size = 4
    with pytest.raises(ValueError, match="one segment"):
        next(iter(mixture))


def test_source_weights_and_reproducible_sampling():
    class NamedSource(Source):
        def __init__(self, name):
            super().__init__([16])
            self.name = name

        def __getitem__(self, index):
            return self.name, index

    datasets = [NamedSource("agibot"), NamedSource("egosuite")]
    first = list(islice(iter(CausalActionMixture(datasets, [1, 3], seed=42)), 4000))
    second = list(islice(iter(CausalActionMixture(datasets, [1, 3], seed=42)), 4000))
    assert first == second
    assert 0.72 < sum(name == "egosuite" for name, _ in first) / len(first) < 0.78


def test_incompatible_templates_fail_before_iteration():
    other = Source([16])
    other.template = SimpleNamespace(template_id="different", width=3)
    with pytest.raises(ValueError, match="same action template"):
        CausalActionMixture([Source([16]), other])
