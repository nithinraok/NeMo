# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
from lhotse import CutSet, SupervisionSegment
from lhotse.testing.dummies import dummy_cut
from lhotse.testing.random import deterministic_rng  # noqa: F401
from omegaconf import DictConfig

from nemo.collections.asr.data.audio_to_text_lhotse_prompted import PromptedAudioToTextLhotseDataset
from nemo.collections.asr.models import EncDecMultiTaskCTCModel
from nemo.collections.common.prompts.canary2 import Canary2PromptFormatter

LANGS = ["en", "hi", "ta"]
SPECIAL_TOKENS = [
    "<|startofcontext|>",
    "<|startoftranscript|>",
    "<|endoftext|>",
    "<pad>",
    "<|nospeech|>",
    "<|pnc|>",
    "<|nopnc|>",
    "<|itn|>",
    "<|noitn|>",
    "<|timestamp|>",
    "<|notimestamp|>",
    "<|diarize|>",
    "<|nodiarize|>",
    "<|emo:undefined|>",
] + [f"<|{lang}|>" for lang in LANGS]


@pytest.fixture(scope="module")
def canary_bpe_tokenizer_dir(tmp_path_factory):
    import sentencepiece as spm

    root = tmp_path_factory.mktemp("canary_bpe")
    corpus = root / "corpus.txt"
    corpus.write_text("\n".join(["hello world", "a very long transcript", "short", "नमस्ते दुनिया", "வணக்கம்"] * 50))
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(root / "tokenizer"),
        vocab_size=96,
        model_type="bpe",
        character_coverage=1.0,
        hard_vocab_limit=False,
        user_defined_symbols=SPECIAL_TOKENS,
    )
    (root / "vocab.txt").write_text("")
    return str(root)


def make_model(tokenizer_dir, encoder_input=True, ctc_loss_weight=0.3):
    model_defaults = {'asr_enc_hidden': 64, 'lm_enc_hidden': 32, 'lm_dec_hidden': 32}
    cfg = DictConfig(
        {
            'sample_rate': 16000,
            'prompt_format': 'canary2',
            'prompt_defaults': [
                {
                    'role': 'user',
                    'slots': {
                        'decodercontext': '',
                        'source_lang': '<|en|>',
                        'target_lang': '<|en|>',
                        'emotion': '<|emo:undefined|>',
                        'pnc': '<|pnc|>',
                        'itn': '<|noitn|>',
                        'diarize': '<|nodiarize|>',
                        'timestamp': '<|notimestamp|>',
                    },
                }
            ],
            'model_defaults': model_defaults,
            'tokenizer': {
                'dir': tokenizer_dir,
                'type': 'bpe',
                'custom_tokenizer': {
                    '_target_': 'nemo.collections.common.tokenizers.canary_tokenizer.CanaryBPETokenizer'
                },
            },
            'preprocessor': {
                '_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor',
                'features': 64,
                'window_size': 0.02,
                'window_stride': 0.01,
                'dither': 0.0,
            },
            'encoder': {
                '_target_': 'nemo.collections.asr.modules.transformer_encoder.TransformerEncoder',
                'feat_in': 64,
                'd_model': model_defaults['asr_enc_hidden'],
                'n_heads': 4,
                'n_layers': 1,
                'drop_rate': 0.0,
                'subsampling': 'feature_stacking',
                'subsampling_factor': 4,
                'self_attention_model': 'rope',
            },
            'transf_decoder': {
                '_target_': 'nemo.collections.asr.modules.transformer.get_nemo_transformer',
                'model_name': None,
                'pretrained': False,
                'encoder': None,
                'pre_ln_final_layer_norm': True,
                'config_dict': {
                    'max_sequence_length': 256,
                    'num_token_types': 0,
                    'hidden_size': model_defaults['lm_dec_hidden'],
                    'inner_size': 4 * model_defaults['lm_dec_hidden'],
                    'num_layers': 1,
                    'num_attention_heads': 2,
                    'pre_ln': True,
                    'vocab_size': None,
                },
            },
            'head': {
                '_target_': 'nemo.collections.asr.parts.submodules.token_classifier.TokenClassifier',
                'num_layers': 1,
                'activation': 'relu',
                'log_softmax': True,
                'hidden_size': model_defaults['lm_dec_hidden'],
                'num_classes': None,
            },
            'decoding': {'strategy': 'beam', 'beam': {'beam_size': 1}},
            'loss': {
                '_target_': 'nemo.collections.common.losses.smoothed_cross_entropy.SmoothedCrossEntropyLoss',
                'label_smoothing': 0.0,
            },
            'optim': {'name': 'adamw', 'lr': 1e-4},
            'aux_ctc': {
                'ctc_loss_weight': ctc_loss_weight,
                'decoder': {
                    '_target_': 'nemo.collections.asr.modules.ConvASRDecoder',
                    'feat_in': None,
                    'num_classes': -1,
                },
                'decoding': {'strategy': 'greedy_batch'},
            },
            'lang_conditioning': {'langs': LANGS, 'encoder_input': encoder_input},
        }
    )
    return EncDecMultiTaskCTCModel(cfg=cfg)


def make_batch(model, langs=("hi", "ta")):
    texts = ["hello world", "नमस्ते दुनिया", "வணக்கம்", "short"]
    cuts = []
    for i, lang in enumerate(langs):
        duration = 1.0 + i
        cut = dummy_cut(
            i,
            duration=duration,
            recording_duration=duration,
            with_data=True,
            supervisions=[
                SupervisionSegment(
                    id=f"cut-{i}", recording_id=f"cut-{i}", start=0, duration=duration, text=texts[i % len(texts)]
                )
            ],
        )
        cut.source_lang = lang
        cut.target_lang = lang
        cut.pnc = "no"
        cuts.append(cut)
    dataset = PromptedAudioToTextLhotseDataset(
        tokenizer=model.tokenizer, prompt=Canary2PromptFormatter(model.tokenizer)
    )
    return dataset[CutSet(cuts)]


class TestEncDecMultiTaskCTCModel:
    @pytest.mark.unit
    def test_constructor(self, canary_bpe_tokenizer_dir):
        model = make_model(canary_bpe_tokenizer_dir)
        assert model.ctc_decoder.num_classes_with_blank == model.tokenizer.vocab_size + 1
        assert model.lang_input_embedding.weight.shape == (len(LANGS) + 1, 64)
        assert torch.count_nonzero(model.lang_input_embedding.weight) == 0

    @pytest.mark.unit
    def test_missing_lang_token_raises(self, canary_bpe_tokenizer_dir):
        with pytest.raises(ValueError, match="missing"):
            model = make_model(canary_bpe_tokenizer_dir)
            model.cfg.lang_conditioning.langs = LANGS + ["xx"]
            EncDecMultiTaskCTCModel(cfg=model.cfg)

    @pytest.mark.unit
    def test_prompt_lang_indices(self, canary_bpe_tokenizer_dir):
        model = make_model(canary_bpe_tokenizer_dir)
        batch = make_batch(model, langs=("ta", "en", "hi"))
        assert model.prompt_lang_indices(batch.prompt).tolist() == [2, 0, 1]
        no_lang = torch.full((2, 3), model.tokenizer.token_to_id("<|pnc|>"))
        assert model.prompt_lang_indices(no_lang).tolist() == [len(LANGS)] * 2
        assert model.prompt_lang_indices(torch.tensor([])) is None

    @pytest.mark.unit
    def test_encoder_input_conditioning(self, deterministic_rng, canary_bpe_tokenizer_dir):
        model = make_model(canary_bpe_tokenizer_dir).eval()
        batch = make_batch(model, langs=("hi", "hi"))
        audio = batch.audio[:1].repeat(2, 1)
        lens = batch.audio_lens[:1].repeat(2)
        indices = torch.tensor([1, 2])
        with torch.no_grad():
            _, _, states, _ = model(input_signal=audio, input_signal_length=lens, lang_indices=indices)
            # zero-initialised: identical outputs for every language and for the plain encoder path
            assert torch.allclose(states[0], states[1])
            model.lang_at_encoder_input = False
            _, _, plain, _ = model(input_signal=audio, input_signal_length=lens)
            assert torch.allclose(states, plain, atol=1e-5)
            model.lang_at_encoder_input = True
            torch.nn.init.normal_(model.lang_input_embedding.weight)
            _, _, states, _ = model(input_signal=audio, input_signal_length=lens, lang_indices=indices)
            assert not torch.allclose(states[0], states[1])

    @staticmethod
    def _run_training_step(model, batch, batch_nb=0):
        """training_step with a dummy optimizer; returns (loss, logged metrics)."""
        model._optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        logged = {}
        model.log_dict = lambda d, **kw: logged.update(d)
        out = model.training_step(batch, batch_nb=batch_nb)
        return out["loss"], logged

    @pytest.mark.unit
    def test_training_step_trains_heads(self, deterministic_rng, canary_bpe_tokenizer_dir):
        # FlexAttention has no CPU backward, so the encoder (and the input embedding before it) is frozen here;
        # test_training_step_trains_lang_embedding covers that path on GPU.
        model = make_model(canary_bpe_tokenizer_dir).train()
        model.encoder.requires_grad_(False)
        model.lang_input_embedding.requires_grad_(False)
        loss, _ = self._run_training_step(model, make_batch(model))
        assert torch.isfinite(loss)
        loss.backward()
        for module in (model.ctc_decoder, model.transf_decoder):
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()), module

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention backward needs CUDA")
    @pytest.mark.unit
    def test_training_step_trains_lang_embedding(self, deterministic_rng, canary_bpe_tokenizer_dir):
        model = make_model(canary_bpe_tokenizer_dir).cuda().train()
        batch = make_batch(model)
        for field in ("audio", "audio_lens", "transcript", "transcript_lens", "prompt", "prompt_lens"):
            setattr(batch, field, getattr(batch, field).cuda())
        batch.prompted_transcript = batch.prompted_transcript.cuda()
        batch.prompted_transcript_lens = batch.prompted_transcript_lens.cuda()
        loss, _ = self._run_training_step(model, batch)
        loss.backward()
        grad = model.lang_input_embedding.weight.grad
        # rows of the languages in the batch (hi=1, ta=2) get gradient; the others do not
        assert grad[1].abs().sum() > 0 and grad[2].abs().sum() > 0
        assert grad[0].abs().sum() == 0 and grad[3].abs().sum() == 0

    @pytest.mark.unit
    def test_ctc_loss_weight_blend(self, deterministic_rng, canary_bpe_tokenizer_dir):
        model = make_model(canary_bpe_tokenizer_dir).eval()
        with torch.no_grad():
            loss, log = self._run_training_step(model, make_batch(model), batch_nb=1)
        expected = 0.7 * log["train_aed_loss"] + 0.3 * log["train_ctc_loss"]
        assert torch.allclose(loss, expected)

    @pytest.mark.unit
    def test_validation_epoch_end_reports_ctc_wer(self, deterministic_rng, canary_bpe_tokenizer_dir):
        model = make_model(canary_bpe_tokenizer_dir).eval()
        batch = make_batch(model)
        with torch.no_grad():
            outputs = [model.validation_pass(batch, 0) for _ in range(2)]
        assert {"val_loss", "val_ctc_loss", "val_wer_num_ctc", "val_wer_denom_ctc"} <= set(outputs[0])
        result = model.multi_validation_epoch_end(outputs)
        assert "val_wer_ctc" in result["log"] and "val_ctc_loss" in result["log"]
        assert result["log"]["val_wer_ctc"] >= 0

    @pytest.mark.unit
    def test_predict_step_aed_and_ctc(self, deterministic_rng, canary_bpe_tokenizer_dir):
        model = make_model(canary_bpe_tokenizer_dir).eval()
        batch = make_batch(model)
        with torch.no_grad():
            aed = model.predict_step(batch)
            model.change_decoding_strategy(decoder_type="ctc")
            ctc = model.predict_step(batch)
            model.change_decoding_strategy(decoder_type="aed")
        assert len(aed) == len(ctc) == 2
        assert all(isinstance(hyp.text, str) for _, hyp in ctc)

    @pytest.mark.unit
    def test_transcribe_tensor_ctc(self, canary_bpe_tokenizer_dir):
        model = make_model(canary_bpe_tokenizer_dir).eval()
        model.change_decoding_strategy(decoder_type="ctc")
        audio = torch.randn(16000)
        hyps = model.transcribe(audio, batch_size=1, source_lang="hi", target_lang="hi")
        assert len(hyps) == 1 and isinstance(hyps[0].text, str)

    @pytest.mark.unit
    def test_oomptimizer_schema_has_ctc_targets(self, canary_bpe_tokenizer_dir):
        model = make_model(canary_bpe_tokenizer_dir)
        items = {item["name"]: item for item in model.oomptimizer_schema["inputs"]}
        assert items["transcript"]["seq_length"] == "output"
        assert items["transcript"]["vocab_size"] == model.tokenizer.vocab_size
        assert items["prompt"]["type"] == "dummy"

    @pytest.mark.unit
    def test_save_restore(self, canary_bpe_tokenizer_dir, tmp_path):
        model = make_model(canary_bpe_tokenizer_dir)
        torch.nn.init.normal_(model.lang_input_embedding.weight)
        path = tmp_path / "model.nemo"
        model.save_to(str(path))
        restored = EncDecMultiTaskCTCModel.restore_from(str(path), map_location="cpu")
        assert isinstance(restored, EncDecMultiTaskCTCModel)
        assert torch.equal(restored.lang_input_embedding.weight, model.lang_input_embedding.weight)
        assert torch.equal(restored.ctc_decoder.decoder_layers[0].weight, model.ctc_decoder.decoder_layers[0].weight)

    @pytest.mark.unit
    def test_aed_ctc_rescoring_scores(self, deterministic_rng, canary_bpe_tokenizer_dir):
        model = make_model(canary_bpe_tokenizer_dir).eval()
        batch = make_batch(model)
        with torch.no_grad():
            _, enc_len, enc, enc_mask = model(
                input_signal=batch.audio,
                input_signal_length=batch.audio_lens,
                lang_indices=model.prompt_lang_indices(batch.prompt),
            )
            seqs = [(0, batch.transcript[0, : batch.transcript_lens[0]].tolist()), (1, []), (0, [5] * 1000)]
            # teacher-forced AED score == sum of the training-forward log-probs of transcript + <eos>
            att = model._aed_log_likelihood(enc, enc_mask, batch.prompt, seqs[:2])
            tokens = seqs[0][1] + [model.tokenizer.eos]
            inp = torch.cat([batch.prompt[0], torch.tensor(seqs[0][1], dtype=torch.long)])[None]
            log_probs, _, _, _ = model(
                input_signal=batch.audio[:1],
                input_signal_length=batch.audio_lens[:1],
                transcript=inp,
                transcript_length=torch.tensor([inp.shape[1]]),
                lang_indices=model.prompt_lang_indices(batch.prompt[:1]),
            )
            P = batch.prompt.shape[1]
            ref = log_probs[0, P - 1 :].float().log_softmax(-1).gather(-1, torch.tensor(tokens)[:, None]).sum()
            assert torch.allclose(att[0], ref, atol=1e-3)
            # CTC forced alignment: finite for the transcript, -inf for a sequence longer than the encoder output
            ctc = model._ctc_log_likelihood(model.ctc_log_probs(enc), enc_len, seqs)
            assert torch.isfinite(ctc[:2]).all() and ctc[2] == float("-inf")

    @pytest.mark.unit
    def test_aed_ctc_rescoring_transcribe(self, deterministic_rng, canary_bpe_tokenizer_dir, monkeypatch):
        model = make_model(canary_bpe_tokenizer_dir).eval()
        audio = [torch.randn(16000), torch.randn(24000)]
        assert model.aed_ctc_rescoring_weight == 0.0
        plain = model.transcribe(audio, batch_size=2, source_lang="hi", target_lang="hi")
        assert all(not hasattr(h, "aed_ctc_selected") for h in plain)
        model.change_aed_ctc_rescoring(0.1)
        hyps = model.transcribe(audio, batch_size=2, source_lang="hi", target_lang="hi")
        assert [h.aed_ctc_selected in ("aed", "ctc") for h in hyps] == [True, True]
        # every output is either the AED or the greedy CTC hypothesis, consistent with the selection flag
        monkeypatch.setenv("AED_CTC_RESCORING_WEIGHT", "1.0")
        ctc_only = model.transcribe(audio, batch_size=2, source_lang="hi", target_lang="hi")
        model.change_decoding_strategy(decoder_type="ctc")
        ctc = model.transcribe(audio, batch_size=2, source_lang="hi", target_lang="hi")
        for h, p, c in zip(ctc_only, plain, ctc):
            assert h.text == (c.text if h.aed_ctc_selected == "ctc" else p.text)
