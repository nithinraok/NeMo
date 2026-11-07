# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import io
import json
import tarfile

import numpy as np
import pytest
import soundfile as sf
from lhotse import CutSet
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.data.lhotse.cutset import read_cutset_from_config


class IdentityDataset:
    def __getitem__(self, cuts):
        return cuts


@pytest.fixture
def hq_manifest(tmp_path):
    """A real tarred source with conflicting text fields and non-boolean HQ flags."""
    wav = io.BytesIO()
    sf.write(wav, np.zeros(1600, dtype=np.float32), 16000, format='WAV')
    flags = [True, False, None, 'true', 1, True, True]
    rows = []
    tar_path = tmp_path / 'audio_0.tar'
    with tarfile.open(tar_path, 'w') as archive:
        for idx, flag in enumerate(flags):
            name = f'{idx}.wav'
            data = wav.getvalue()
            (tmp_path / name).write_bytes(data)
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
            row = dict(audio_filepath=name, duration=0.1, shard_id=0, text='Wrong target.', tn_raw='Right target.')
            if flag is not None:
                row['high_quality'] = flag
            if idx == 5:
                row['tn_raw'] = '  '
            if idx == 6:
                row['_skipme'] = True
            rows.append(row)
    path = tmp_path / 'manifest_0.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return dict(manifest_filepath=str(path), tarred_audio_filepaths=str(tar_path), text_field='tn_raw')


@pytest.mark.parametrize('nested', [False, True])
def test_strict_hq_original_tarred_manifest_zero_workers(hq_manifest, nested):
    config = dict(
        num_workers=0,
        batch_size=2,
        force_finite=True,
        keep_high_quality_only=True,
        skip_empty_text=True,
        shuffle=False,
        seed=42,
        shard_seed=42,
    )
    if nested:
        config['input_cfg'] = [dict(type='group', input_cfg=[dict(type='nemo_tarred', **hq_manifest)])]
    else:
        config.update(hq_manifest)
    loader = get_lhotse_dataloader_from_config(config, global_rank=0, world_size=1, dataset=IdentityDataset())
    assert loader.num_workers == 0
    cuts = [cut for batch in loader for cut in batch]
    assert len(cuts) == 1
    assert cuts[0].custom['high_quality'] is True
    assert cuts[0].supervisions[0].text == 'Right target.'
    assert cuts[0].load_audio().shape == (1, 1600)


def test_hq_filter_is_lazy_and_disabled_by_default(hq_manifest):
    cuts, _ = read_cutset_from_config(OmegaConf.create(dict(**hq_manifest, force_finite=True)))
    assert isinstance(cuts, CutSet)
    assert cuts.is_lazy
    assert len(list(cuts)) == 6  # only the explicit _skipme row is removed


def test_hq_filter_runs_before_tokenization_and_noise(hq_manifest, monkeypatch):
    from nemo.collections.common.data.lhotse import dataloader as module

    calls = []

    def tokenize(cut, tokenizer, **kwargs):
        assert cut.custom['high_quality'] is True
        assert cut.supervisions[0].text.strip()
        calls.append(cut.id)
        return cut

    monkeypatch.setattr(module, 'tokenize', tokenize)
    loader = get_lhotse_dataloader_from_config(
        dict(
            **hq_manifest,
            num_workers=0,
            batch_size=2,
            force_finite=True,
            keep_high_quality_only=True,
            skip_empty_text=True,
            noise_path=dict(manifest_filepath=hq_manifest['manifest_filepath']),
            noise_mix_prob=1.0,
            pretokenize=True,
            seed=42,
            shard_seed=42,
        ),
        global_rank=0,
        world_size=1,
        dataset=IdentityDataset(),
        tokenizer=lambda text: [1],
    )
    cuts = [cut for batch in loader for cut in batch]
    assert len(calls) == 1
    assert len(cuts) == 1
