# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import io
import json
import tarfile
from copy import deepcopy

import numpy as np
import pytest
import soundfile as sf
from lhotse import CutSet
from lhotse.lazy import IteratorNode, get_graph_origin
from lhotse.testing.dummies import dummy_cut

from nemo.collections.common.data.lhotse.dataloader import (
    get_lhotse_sampler_from_config,
    make_structured_with_schema_warnings,
)
from nemo.collections.common.data.lhotse.nemo_adapters import LazyNeMoTarredIterator


def make_config(tmp_path, monkeypatch):
    monkeypatch.setenv('USE_AIS_GET_BATCH', 'true')
    monkeypatch.setattr('lhotse.dataset.dataloading.get_worker_partition', lambda: (0, 1))
    audio = io.BytesIO()
    sf.write(audio, np.zeros(1600, dtype=np.float32), 16000, format='WAV')
    archive = tmp_path / 'audio_0.tar'
    with tarfile.open(archive, 'w') as tar:
        entry = tarfile.TarInfo('0.wav')
        entry.size = len(audio.getvalue())
        tar.addfile(entry, io.BytesIO(audio.getvalue()))
    manifest = tmp_path / 'manifest_0.jsonl'
    manifest.write_text(
        json.dumps(dict(audio_filepath='0.wav', duration=0.1, shard_id=0, high_quality=True, tn_raw='Example.')) + '\n'
    )
    source = dict(type='nemo_tarred', manifest_filepath=str(manifest), tarred_audio_filepaths=str(archive))
    return make_structured_with_schema_warnings(
        dict(
            indexed=True,
            cache_indexed_restore_capabilities=True,
            input_sampling_rate=16000,
            text_field='tn_raw',
            keep_high_quality_only=True,
            skip_empty_text=True,
            input_cfg=[dict(source, weight=1) for _ in range(32)],
            use_bucketing=True,
            concurrent_bucketing=False,
            bucket_duration_bins=[0.1],
            bucket_buffer_size=32,
            shuffle_buffer_size=0,
            batch_size=2,
            shuffle=False,
            shard_seed=42,
            seed=42,
        )
    )


def test_checkpoint_does_not_recheck_every_leaf_for_every_buffered_cut(tmp_path, monkeypatch):
    config = make_config(tmp_path, monkeypatch)
    original = LazyNeMoTarredIterator.has_constant_time_access.fget
    calls = []

    def counted(source):
        calls.append(source)
        return original(source)

    monkeypatch.setattr(LazyNeMoTarredIterator, 'has_constant_time_access', property(counted))
    sampler, _ = get_lhotse_sampler_from_config(config, global_rank=0, world_size=1)
    next(iter(sampler))
    calls.clear()
    state = sampler.state_dict()
    assert state['bucketer_state'] is not None
    assert len(calls) <= 64, 'Checkpoint capture traverses all source leaves for each buffered cut'


def test_cached_graph_restores_exactly_and_rejects_changed_graph(tmp_path, monkeypatch):
    config = make_config(tmp_path, monkeypatch)
    sampler, _ = get_lhotse_sampler_from_config(config, global_rank=0, world_size=1)

    def identity(batch):
        return [(cut.id, cut.supervisions[0].text, get_graph_origin(cut)) for cut in batch]

    iterator = iter(sampler)
    prefix = [identity(next(iterator)) for _ in range(20)]
    state = deepcopy(sampler.state_dict())
    expected = [identity(next(iterator)) for _ in range(16)]
    restored, _ = get_lhotse_sampler_from_config(config, global_rank=0, world_size=1)
    restored.load_state_dict(deepcopy(state))
    actual = [identity(next(iter2)) for iter2 in [iter(restored)] for _ in range(16)]
    assert actual == expected

    # Native checkpoints validate node types. Changing this opt-in setting on
    # resume must fail visibly rather than silently reset or replay the data.
    config.cache_indexed_restore_capabilities = False
    unwrapped, _ = get_lhotse_sampler_from_config(config, global_rank=0, world_size=1)
    assert [identity(next(iter2)) for iter2 in [iter(unwrapped)] for _ in range(36)] == prefix + expected
    unwrapped.load_state_dict(deepcopy(state))
    with pytest.raises(RuntimeError, match='Type mismatch'):
        next(iter(unwrapped))


@pytest.mark.parametrize('restorable', [False, True])
@pytest.mark.parametrize('sized', [False, True])
def test_capability_cache_preserves_source_protocol(restorable, sized):
    from nemo.collections.common.data.lhotse.cutset import _cache_cutset_restore_capability

    class Source(IteratorNode):
        has_constant_time_access = restorable
        is_checkpointable = True
        position = 0

        def __iter__(self):
            return iter([dummy_cut(0)])

        def __getitem__(self, index):
            return dummy_cut(index)

        def state_dict(self):
            return {'position': self.position}

        def load_state_dict(self, state):
            self.position = state['position']

    class SizedSource(Source):
        def __len__(self):
            return 1

    source = SizedSource() if sized else Source()
    cuts = CutSet(source)
    cached = _cache_cutset_restore_capability(cuts)
    assert cached.has_constant_time_access == restorable
    assert hasattr(cached.data, '__len__') == sized
    if not restorable:
        assert cached is cuts  # A cache must never turn a non-restorable graph into a restorable one.
    assert cached.data.state_dict() == source.state_dict()
    cached.data.load_state_dict({'position': 3})
    assert source.position == 3
    assert list(cached)[0].id == list(cuts)[0].id
    assert cached.data[2].id == source[2].id
