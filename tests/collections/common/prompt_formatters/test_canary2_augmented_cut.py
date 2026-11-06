# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from lhotse import AudioSource, MonoCut, Recording, SupervisionSegment
from lhotse.cut import MixedCut, MixTrack, MultiCut

from nemo.collections.common.prompts.canary2 import canary2


@pytest.mark.parametrize('nesting', [0, 1, 2, 3, 'rir-padding'])
def test_canary2_nested_augmented_cut_preserves_speech_prompt(nesting):
    recording = Recording('speech', [AudioSource('file', [0], 'unused.wav')], 16000, 16000, 1.0)
    speech = MonoCut(
        'speech',
        0.0,
        1.0,
        0,
        recording=recording,
        supervisions=[SupervisionSegment('s', 'speech', 0.0, 1.0, text='Hello.', language='en')],
        custom={'source_lang': 'en', 'target_lang': 'en', 'pnc': '<|pnc|>'},
    )
    noise = MonoCut('noise', 0.0, 1.0, 0, recording=recording)
    cut = speech
    if nesting == 'rir-padding':
        rir = Recording('rir', [AudioSource('file', [0], 'unused-rir.wav')], 16000, 1600, 0.1)
        cut = speech.mix(noise, snr=20).reverb_rir(rir_recording=rir).pad(2.0)
        nesting = 0
    for _ in range(nesting):
        # RIR transforms followed by padding can retain a nested mixture rather than flattening it.
        cut = MixedCut('augmented', tracks=[MixTrack(cut=cut), MixTrack(cut=noise, snr=20)])
    encode = Mock(return_value={'answer_ids': torch.tensor([7, 99]), 'input_ids': torch.tensor([1, 7, 99])})
    prompt = SimpleNamespace(
        PROMPT_LANGUAGE_SLOT='prompt_language',
        tokenizer=SimpleNamespace(token_to_id=lambda _: 99),
        encode_dialog=encode,
    )
    result = canary2(cut, prompt)
    turns = encode.call_args.args[0]
    assert turns[0]['slots']['source_lang'] == 'en'
    assert turns[0]['slots']['target_lang'] == 'en'
    assert turns[1]['slots']['text'] == 'Hello.'
    assert result['answer_ids'].tolist() == [7]


def test_canary2_nested_multichannel_cut_is_rejected():
    cut = MultiCut('stereo', 0.0, 1.0, [0, 1])
    cut = MixedCut('mixed', tracks=[MixTrack(cut=cut)])
    with pytest.raises(TypeError, match='single channel'):
        canary2(cut, Mock())
