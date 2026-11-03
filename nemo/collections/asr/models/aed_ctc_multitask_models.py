# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Canary-style attention encoder-decoder (AED) model with an auxiliary CTC head and language-conditioned encoder.

Compared with :class:`EncDecMultiTaskModel`:

* **CTC head.** A ``ConvASRDecoder`` on the (projected) encoder states predicts the transcript tokens with CTC. The
  training loss is ``(1 - ctc_loss_weight) * AED + ctc_loss_weight * CTC``; both are per-token averages (label-smoothed
  cross entropy and CTC with ``mean_volume`` reduction). Validation additionally reports ``val_wer_ctc``, and
  ``change_decoding_strategy(decoder_type='ctc')`` switches transcription to greedy CTC decoding.
* **Language at the encoder input.** The AED decoder sees the language through the canary2 prompt as usual. In
  addition, a zero-initialised embedding of the utterance's source language (the first language token of its
  prompt) is added to every pre-encoder frame, so the shared encoder -- and therefore the CTC head -- is
  language-conditioned too. At initialisation this is an exact no-op.

Config additions under ``model``::

    aux_ctc:
      ctc_loss_weight: 0.3
      decoder: {_target_: nemo.collections.asr.modules.ConvASRDecoder, feat_in: null, num_classes: -1}
      decoding: {strategy: greedy_batch}
    lang_conditioning:
      langs: [en, hi, bn, ...]          # tokens <|xx|> that must exist in the tokenizer
      encoder_input: true               # add the language embedding at the encoder input
      encoder_input_dropout: 0.0        # probability of replacing the language by "unknown" during training
"""

import inspect
import os
from typing import Dict, Optional

import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.asr.data.audio_to_text_lhotse_prompted import PromptedAudioToTextMiniBatch
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.models.aed_multitask_models import (
    EncDecMultiTaskModel,
    MultiTaskTranscriptionConfig,
    lens_to_mask,
)
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCBPEDecoding, CTCBPEDecodingConfig
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.core.classes.common import typecheck
from nemo.core.neural_types import LabelsType, LengthsType, NeuralType
from nemo.utils import logging

__all__ = ["EncDecMultiTaskCTCModel"]


class EncDecMultiTaskCTCModel(EncDecMultiTaskModel):
    """AED (Canary) model with an auxiliary CTC head and a language-conditioned encoder input. See module docstring."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

        # ---- auxiliary CTC head on the projected encoder states ------------------------------------------------
        if "aux_ctc" not in self.cfg:
            raise ValueError("EncDecMultiTaskCTCModel needs a `model.aux_ctc` config section.")
        vocabulary = self.tokenizer.vocab
        with open_dict(self.cfg.aux_ctc):
            if not self.cfg.aux_ctc.decoder.get("feat_in"):
                self.cfg.aux_ctc.decoder.feat_in = self.cfg.model_defaults.lm_dec_hidden
            if self.cfg.aux_ctc.decoder.get("num_classes", -1) < 1:
                self.cfg.aux_ctc.decoder.num_classes = len(vocabulary)
            self.cfg.aux_ctc.decoder.vocabulary = list(vocabulary)
            if self.cfg.aux_ctc.get("decoding") is None:
                self.cfg.aux_ctc.decoding = OmegaConf.structured(CTCBPEDecodingConfig)
        self.ctc_decoder = EncDecMultiTaskCTCModel.from_config_dict(self.cfg.aux_ctc.decoder)
        self.ctc_loss_weight = float(self.cfg.aux_ctc.get("ctc_loss_weight", 0.3))
        self.ctc_loss = CTCLoss(
            num_classes=self.ctc_decoder.num_classes_with_blank - 1,
            zero_infinity=True,
            reduction=self.cfg.aux_ctc.get("ctc_reduction", "mean_volume"),
        )
        self.ctc_decoding = CTCBPEDecoding(self.cfg.aux_ctc.decoding, tokenizer=self.tokenizer)
        self.ctc_wer = WER(
            decoding=self.ctc_decoding,
            use_cer=self.cfg.aux_ctc.get("use_cer", False),
            dist_sync_on_step=True,
            log_prediction=False,
        )
        self.cur_decoder = "aed"

        # ---- language conditioning at the encoder input --------------------------------------------------------
        lang_cfg = self.cfg.get("lang_conditioning", {}) or {}
        self.lang_list = list(lang_cfg.get("langs", []))
        self.lang_at_encoder_input = bool(lang_cfg.get("encoder_input", False)) and bool(self.lang_list)
        self.lang_dropout = float(lang_cfg.get("encoder_input_dropout", 0.0))
        self.unknown_lang_index = len(self.lang_list)
        lang_token_ids = [self.tokenizer.token_to_id(f"<|{lang}|>") for lang in self.lang_list]
        # SentencePiece maps unknown pieces to <unk>, so check the round trip rather than the id.
        missing = [
            lang
            for lang, tid in zip(self.lang_list, lang_token_ids)
            if self.tokenizer.ids_to_tokens([tid])[0] != f"<|{lang}|>"
        ]
        if missing:
            raise ValueError(f"Language tokens missing from the tokenizer: {['<|%s|>' % m for m in missing]}")
        # token id -> language index lookup (unknown for every other token)
        lut = torch.full((self.tokenizer.vocab_size,), self.unknown_lang_index, dtype=torch.long)
        for index, token_id in enumerate(lang_token_ids):
            lut[token_id] = index
        self.register_buffer("lang_token_lut", lut, persistent=False)
        self.register_buffer("lang_token_ids", torch.tensor(lang_token_ids, dtype=torch.long), persistent=False)
        if self.lang_at_encoder_input:
            if "bypass_pre_encode" not in inspect.signature(self.encoder.forward).parameters:
                raise ValueError(
                    f"lang_conditioning.encoder_input requires an encoder whose forward() accepts bypass_pre_encode, "
                    f"but {type(self.encoder).__name__} does not."
                )
            self.lang_input_embedding = torch.nn.Embedding(len(self.lang_list) + 1, self.encoder._feat_out)
            torch.nn.init.zeros_(self.lang_input_embedding.weight)
        logging.info(
            f"EncDecMultiTaskCTCModel: ctc_loss_weight={self.ctc_loss_weight}, "
            f"lang_at_encoder_input={self.lang_at_encoder_input} ({len(self.lang_list)} langs), "
            f"lang_dropout={self.lang_dropout}"
        )

    # ------------------------------------------------------------------------------------------------------------
    # language index from the prompt
    # ------------------------------------------------------------------------------------------------------------
    def prompt_lang_indices(self, prompt: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Index of the first language token in each prompt row (the canary2 source_lang); unknown if none."""
        if prompt is None or prompt.dim() != 2 or prompt.numel() == 0 or not self.lang_list:
            return None
        is_lang = torch.isin(prompt, self.lang_token_ids)
        first = is_lang.int().argmax(dim=1)
        tokens = prompt.gather(1, first.unsqueeze(1)).squeeze(1)
        indices = self.lang_token_lut[tokens.clamp(min=0, max=self.lang_token_lut.numel() - 1)]
        return torch.where(is_lang.any(dim=1), indices, torch.full_like(indices, self.unknown_lang_index))

    # ------------------------------------------------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------------------------------------------------
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        types = dict(super().input_types)
        types["lang_indices"] = NeuralType(("B",), LabelsType(), optional=True)
        return types

    @typecheck()
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        transcript=None,
        transcript_length=None,
        lang_indices=None,
    ):
        """Like :meth:`EncDecMultiTaskModel.forward`, plus an optional per-utterance language index (``[B]``) that
        is embedded and added to the pre-encoder output when ``lang_conditioning.encoder_input`` is enabled."""
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )
        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length
            )
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        if self.lang_at_encoder_input:
            if lang_indices is None:
                lang_indices = torch.full(
                    (processed_signal.shape[0],), self.unknown_lang_index, dtype=torch.long, device=processed_signal.device
                )
            elif self.training and self.lang_dropout > 0:
                drop = torch.rand(lang_indices.shape, device=lang_indices.device) < self.lang_dropout
                lang_indices = torch.where(drop, torch.full_like(lang_indices, self.unknown_lang_index), lang_indices)
            features, features_len = self._pre_encode(processed_signal, processed_signal_length)
            features = features + self.lang_input_embedding(lang_indices).unsqueeze(1).to(features.dtype)
            encoded, encoded_len = self.encoder(audio_signal=features, length=features_len, bypass_pre_encode=True)
        else:
            encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)

        enc_states = encoded.permute(0, 2, 1)
        enc_states = self.encoder_decoder_proj(enc_states)
        enc_mask = lens_to_mask(encoded_len, enc_states.shape[1]).to(enc_states.dtype)
        if self.use_transf_encoder:
            enc_states = self.transf_encoder(encoder_states=enc_states, encoder_mask=enc_mask)

        transf_log_probs = None
        if transcript is not None:
            dec_mask = lens_to_mask(transcript_length, transcript.shape[1]).to(transcript.dtype)
            dec_states = self.transf_decoder(
                input_ids=transcript, decoder_mask=dec_mask, encoder_embeddings=enc_states, encoder_mask=enc_mask
            )
            transf_log_probs = self.log_softmax(hidden_states=dec_states)

        return transf_log_probs, encoded_len, enc_states, enc_mask

    def _pre_encode(self, processed_signal, processed_signal_length):
        """Run only the encoder's pre-encoder: (B, C, T) features -> (B, T', D) embeddings + lengths."""
        pre_encode = self.encoder.pre_encode
        module = getattr(pre_encode, "_checkpoint_wrapped_module", pre_encode)
        if type(module).__name__ == "FeatureStacking":
            features, length = pre_encode(processed_signal, processed_signal_length)
        elif isinstance(module, torch.nn.Linear):
            features, length = pre_encode(torch.transpose(processed_signal, 1, 2)), processed_signal_length
        else:
            features, length = pre_encode(x=torch.transpose(processed_signal, 1, 2), lengths=processed_signal_length)
        return features, length.to(torch.int64)

    def ctc_log_probs(self, enc_states: torch.Tensor) -> torch.Tensor:
        """CTC log-probabilities ``[B, T, V+1]`` from the projected encoder states ``[B, T, H]``."""
        return self.ctc_decoder(encoder_output=enc_states.transpose(1, 2))

    # ------------------------------------------------------------------------------------------------------------
    # training / validation
    # ------------------------------------------------------------------------------------------------------------
    def training_step(self, batch: PromptedAudioToTextMiniBatch, batch_nb):
        if batch is None:
            return torch.tensor([0.0])

        input_ids, labels = batch.get_decoder_inputs_outputs()
        input_ids_lens = batch.prompted_transcript_lens - 1
        num_frames = batch.audio_lens.sum().float()
        num_tokens = batch.prompted_transcript_lens.sum().float()
        tot_frames = torch.as_tensor(batch.audio.numel(), device=num_frames.device, dtype=torch.float)
        tot_tokens = torch.as_tensor(batch.prompted_transcript.numel(), device=num_frames.device, dtype=torch.float)

        transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
            input_signal=batch.audio,
            input_signal_length=batch.audio_lens,
            transcript=input_ids,
            transcript_length=input_ids_lens,
            lang_indices=self.prompt_lang_indices(batch.prompt),
        )
        if self.cfg.get("use_loss_mask_for_prompt", False):
            maxlen = batch.prompted_transcript.shape[1] - 1
            loss_mask = lens_to_mask(input_ids_lens, maxlen) & ~lens_to_mask(batch.prompt_lens - 1, maxlen)
        else:
            loss_mask = None
        aed_loss = self.loss(log_probs=transf_log_probs, labels=labels, output_mask=loss_mask)

        ctc_log_probs = self.ctc_log_probs(enc_states)
        ctc_loss = self.ctc_loss(
            log_probs=ctc_log_probs,
            targets=batch.transcript,
            input_lengths=encoded_len,
            target_lengths=batch.transcript_lens,
        )
        loss = (1.0 - self.ctc_loss_weight) * aed_loss + self.ctc_loss_weight * ctc_loss

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
        else:
            log_every_n_steps = 1
        metric_dict = (
            self.metric.eval(
                batch=batch,
                predictions=enc_states,
                predictions_lengths=encoded_len,
                predictions_mask=enc_mask,
                prefix="training_batch",
            )
            if (batch_nb + 1) % log_every_n_steps == 0
            else {}
        )
        metric_dict.update(
            {
                'train_loss': loss,
                'train_aed_loss': aed_loss,
                'train_ctc_loss': ctc_loss,
                'learning_rate': torch.as_tensor(self._optimizer.param_groups[0]['lr']),
                'batch_size': torch.as_tensor(batch.audio.shape[0]),
                'num_frames': num_frames,
                'num_tokens': num_tokens,
                'input_to_padding_ratio': num_frames / tot_frames,
                'output_to_padding_ratio': num_tokens / tot_tokens,
            }
        )
        return {"loss": loss, "log": metric_dict}

    def validation_pass(self, batch: PromptedAudioToTextMiniBatch, batch_idx, dataloader_idx=0, eval_mode="val"):
        input_ids, labels = batch.get_decoder_inputs_outputs()
        input_ids_lens = batch.prompted_transcript_lens - 1

        transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
            input_signal=batch.audio,
            input_signal_length=batch.audio_lens,
            transcript=input_ids,
            transcript_length=batch.prompted_transcript_lens,
            lang_indices=self.prompt_lang_indices(batch.prompt),
        )
        if self.cfg.get("use_loss_mask_for_prompt", False):
            maxlen = batch.prompted_transcript.shape[1] - 1
            loss_mask = lens_to_mask(input_ids_lens, maxlen) & ~lens_to_mask(batch.prompt_lens - 1, maxlen)
            num_measurements = loss_mask.long().sum()
        else:
            loss_mask = None
            num_measurements = transf_log_probs.shape[0] * transf_log_probs.shape[1]
        aed_loss = self.loss(log_probs=transf_log_probs, labels=labels, output_mask=loss_mask)
        self.val_loss(loss=aed_loss, num_measurements=num_measurements)

        metric_dict = self.metric.eval(
            batch=batch,
            predictions=enc_states,
            predictions_lengths=encoded_len,
            predictions_mask=enc_mask,
            prefix=eval_mode,
            return_all_metrics=True,
        )
        metric_dict[f"{eval_mode}_loss"] = aed_loss

        ctc_log_probs = self.ctc_log_probs(enc_states)
        metric_dict[f"{eval_mode}_ctc_loss"] = self.ctc_loss(
            log_probs=ctc_log_probs,
            targets=batch.transcript,
            input_lengths=encoded_len,
            target_lengths=batch.transcript_lens,
        )
        self.ctc_wer.update(
            predictions=ctc_log_probs,
            targets=batch.transcript,
            targets_lengths=batch.transcript_lens,
            predictions_lengths=encoded_len,
        )
        _, wer_num, wer_denom = self.ctc_wer.compute()
        self.ctc_wer.reset()
        metric_dict[f"{eval_mode}_wer_num_ctc"] = wer_num
        metric_dict[f"{eval_mode}_wer_denom_ctc"] = wer_denom
        return metric_dict

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        result = super().multi_validation_epoch_end(outputs, dataloader_idx)
        return self._add_ctc_epoch_metrics(result, outputs, "val")

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        result = super().multi_test_epoch_end(outputs, dataloader_idx)
        return self._add_ctc_epoch_metrics(result, outputs, "test")

    @staticmethod
    def _add_ctc_epoch_metrics(result, outputs, mode):
        if not outputs or f"{mode}_wer_num_ctc" not in outputs[0]:
            return result
        num = torch.stack([x[f"{mode}_wer_num_ctc"] for x in outputs]).sum()
        denom = torch.stack([x[f"{mode}_wer_denom_ctc"] for x in outputs]).sum()
        ctc_loss = torch.stack([x[f"{mode}_ctc_loss"] for x in outputs]).mean()
        log = result.setdefault("log", {})
        log[f"{mode}_wer_ctc"] = num / denom
        log[f"{mode}_ctc_loss"] = ctc_loss
        return result

    # ------------------------------------------------------------------------------------------------------------
    # decoding / transcription
    # ------------------------------------------------------------------------------------------------------------
    def change_decoding_strategy(self, decoding_cfg: DictConfig = None, decoder_type: str = None, verbose: bool = True):
        """``decoder_type='ctc'`` switches transcription to the CTC head; ``'aed'``/None configures the AED decoder."""
        if decoder_type in (None, "aed", "multitask"):
            self.cur_decoder = "aed"
            if decoding_cfg is not None:
                super().change_decoding_strategy(decoding_cfg)
            return
        if decoder_type != "ctc":
            raise ValueError(f"decoder_type must be 'aed' or 'ctc', got {decoder_type!r}")
        if decoding_cfg is None:
            decoding_cfg = self.cfg.aux_ctc.decoding
        self.ctc_decoding = CTCBPEDecoding(decoding_cfg=decoding_cfg, tokenizer=self.tokenizer)
        self.ctc_wer = WER(
            decoding=self.ctc_decoding,
            use_cer=self.ctc_wer.use_cer,
            dist_sync_on_step=True,
            log_prediction=self.ctc_wer.log_prediction,
        )
        with open_dict(self.cfg.aux_ctc):
            self.cfg.aux_ctc.decoding = decoding_cfg
        self.cur_decoder = "ctc"
        if verbose:
            logging.info(f"Changed decoding strategy to the CTC head: {OmegaConf.to_yaml(decoding_cfg)}")

    def _transcribe_forward(self, batch, trcfg: MultiTaskTranscriptionConfig) -> dict:
        """Builds the prompt first (so the encoder can be language-conditioned), then runs the encoder."""
        if isinstance(batch, PromptedAudioToTextMiniBatch):
            audio, audio_lens, decoder_input_ids = batch.audio, batch.audio_lens, batch.prompt
        else:
            audio, audio_lens = batch[0], batch[1]
            decoder_input_ids = batch[4] if len(batch) == 6 else None
        if decoder_input_ids is None:
            decoder_input_ids = self._default_prompt_ids(audio.shape[0], trcfg).to(audio.device)

        log_probs, encoded_len, enc_states, enc_mask = self.forward(
            input_signal=audio,
            input_signal_length=audio_lens,
            lang_indices=self.prompt_lang_indices(decoder_input_ids),
        )
        return dict(
            log_probs=log_probs,
            encoded_lengths=encoded_len,
            encoder_states=enc_states,
            encoder_mask=enc_mask,
            decoder_input_ids=decoder_input_ids,
            batch=batch,
        )

    def _default_prompt_ids(self, batch_size: int, trcfg: MultiTaskTranscriptionConfig) -> torch.Tensor:
        """The prompt from ``trcfg.prompt`` with defaults filled in (same logic as the parent class)."""
        default_turns = self.prompt.get_default_dialog_slots()
        if not trcfg.prompt:
            turns = default_turns
        else:
            turns = trcfg.prompt.copy()
            for turn in turns:
                defaults = [t for t in default_turns if t["role"] == turn["role"]]
                if defaults:
                    turn["slots"] = turn["slots"].copy()
                    for slot, value in defaults[0]["slots"].items():
                        if turn["slots"].get(slot) is None:
                            turn["slots"][slot] = value
        return self.prompt.encode_dialog(turns=turns)["context_ids"].unsqueeze(0).repeat(batch_size, 1)

    # ------------------------------------------------------------------------------------------------------------
    # AED/CTC hypothesis selection
    # ------------------------------------------------------------------------------------------------------------
    @property
    def aed_ctc_rescoring_weight(self) -> float:
        """Weight ``w`` of the AED/CTC hypothesis selection (0 = off).

        For every utterance the AED hypothesis and the greedy CTC hypothesis are both scored with
        ``(1 - w) * log p_aed(y | x) + w * log p_ctc(y | x)`` (teacher-forced decoder log-likelihood incl. <eos>;
        CTC forced-alignment log-likelihood) and the better one is returned. This removes AED repetition loops (a
        looping hypothesis cannot be aligned by CTC) at the cost of one teacher-forced decoder pass. Set
        ``model.aux_ctc.aed_ctc_rescoring.ctc_weight``, call :meth:`change_aed_ctc_rescoring`, or (for scripts such as
        transcribe_speech.py) export ``AED_CTC_RESCORING_WEIGHT``, which takes precedence.
        """
        env = os.environ.get("AED_CTC_RESCORING_WEIGHT")
        if env:
            return float(env)
        return float((self.cfg.aux_ctc.get("aed_ctc_rescoring") or {}).get("ctc_weight", 0.0))

    def change_aed_ctc_rescoring(self, ctc_weight: float = 0.1):
        """Enable (``ctc_weight > 0``) or disable (0) the AED/CTC hypothesis selection used with AED decoding."""
        with open_dict(self.cfg.aux_ctc):
            self.cfg.aux_ctc.aed_ctc_rescoring = {"ctc_weight": float(ctc_weight)}

    @torch.inference_mode()
    def _aed_log_likelihood(self, enc_states, enc_mask, prompt, sequences, chunk: int = 128) -> torch.Tensor:
        """Teacher-forced ``log p_aed(seq + <eos> | x, prompt)`` for each ``(utterance index, token list)`` pair."""
        tok = self.tokenizer
        max_len = self.transf_decoder.max_sequence_length
        P = prompt.shape[1]
        device = enc_states.device
        scores = []
        for start in range(0, len(sequences), chunk):
            part = [(i, list(seq)[: max_len - P]) for i, seq in sequences[start : start + chunk]]
            L = P + max(len(seq) for _, seq in part)
            inp = torch.full((len(part), L), tok.pad, dtype=torch.long, device=device)
            tgt = torch.full((len(part), L), -1, dtype=torch.long, device=device)
            dmask = torch.zeros((len(part), L), dtype=torch.long, device=device)
            for k, (i, seq) in enumerate(part):
                inp[k, :P] = prompt[i]
                if seq:
                    inp[k, P : P + len(seq)] = torch.tensor(seq, device=device)
                dmask[k, : P + len(seq)] = 1
                target = seq + ([tok.eos] if P + len(seq) < max_len else [])
                tgt[k, P - 1 : P - 1 + len(target)] = torch.tensor(target, device=device)
            index = torch.tensor([i for i, _ in part], device=device)
            hidden = self.transf_decoder(
                input_ids=inp, decoder_mask=dmask, encoder_embeddings=enc_states[index], encoder_mask=enc_mask[index]
            )
            log_probs = torch.log_softmax(self.log_softmax(hidden_states=hidden).float(), dim=-1)
            picked = log_probs.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            scores.append((picked * (tgt >= 0)).sum(-1))
        return torch.cat(scores)

    @torch.inference_mode()
    def _ctc_log_likelihood(self, ctc_log_probs, encoded_len, sequences) -> torch.Tensor:
        """``log p_ctc(seq | x)`` by forced alignment (-inf when the sequence cannot be aligned)."""
        device = ctc_log_probs.device
        index = torch.tensor([i for i, _ in sequences], device=device)
        lengths = torch.tensor([len(seq) for _, seq in sequences], device=device)
        targets = torch.zeros((len(sequences), max(1, int(lengths.max()))), dtype=torch.long, device=device)
        for k, (_, seq) in enumerate(sequences):
            if len(seq):
                targets[k, : len(seq)] = torch.tensor(list(seq), device=device)
        loss = torch.nn.functional.ctc_loss(
            ctc_log_probs[index].float().transpose(0, 1),
            targets,
            encoded_len[index],
            lengths,
            blank=ctc_log_probs.shape[-1] - 1,
            reduction="none",
            zero_infinity=False,
        )
        return -loss

    @torch.inference_mode()
    def _select_aed_or_ctc(self, hypotheses, enc_states, enc_mask, encoded_len, prompt, ctc_weight: float):
        """Replace AED hypotheses by the greedy CTC hypothesis where the latter has the better combined score."""
        ctc_log_probs = self.ctc_log_probs(enc_states)
        ctc_hyps = self.ctc_decoding.ctc_decoder_predictions_tensor(
            ctc_log_probs, decoder_lengths=encoded_len, return_hypotheses=False
        )
        if isinstance(ctc_hyps, tuple):
            ctc_hyps = ctc_hyps[0]
        # greedy CTC token sequences (collapse repeats, drop blanks)
        blank = ctc_log_probs.shape[-1] - 1
        best_path = ctc_log_probs.argmax(-1).cpu().tolist()
        ctc_tokens = []
        for i, path in enumerate(best_path):
            seq, prev = [], -1
            for t in path[: int(encoded_len[i])]:
                if t != prev and t != blank:
                    seq.append(t)
                prev = t
            ctc_tokens.append(seq)
        aed_tokens = [
            [int(t) for t in (h.y_sequence.tolist() if torch.is_tensor(h.y_sequence) else h.y_sequence)]
            for h in hypotheses
        ]
        pairs = list(enumerate(aed_tokens)) + list(enumerate(ctc_tokens))
        att = self._aed_log_likelihood(enc_states, enc_mask, prompt, pairs)
        ctc = self._ctc_log_likelihood(ctc_log_probs, encoded_len, pairs)
        ctc = torch.where(torch.isfinite(ctc), ctc, torch.full_like(ctc, -1e9))
        combined = ((1.0 - ctc_weight) * att + ctc_weight * ctc).view(2, -1)
        # identical token sequences can score slightly differently in different batch rows: only switch when they differ
        use_ctc = [u and c != a for u, c, a in zip((combined[1] > combined[0]).cpu().tolist(), ctc_tokens, aed_tokens)]
        for i, hyp in enumerate(hypotheses):
            hyp.aed_ctc_selected = "ctc" if use_ctc[i] else "aed"
            if use_ctc[i]:
                hyp.text = ctc_hyps[i].text if hasattr(ctc_hyps[i], "text") else ctc_hyps[i]
                hyp.y_sequence = torch.tensor(ctc_tokens[i], dtype=torch.long)
                hyp.token_confidence = None
                hyp.xatt_scores = None
        return hypotheses

    def _transcribe_output_processing(self, outputs, trcfg: MultiTaskTranscriptionConfig):
        if self.cur_decoder != "ctc":
            ctc_weight = self.aed_ctc_rescoring_weight
            chunked = trcfg.enable_chunking and self.timestamps_asr_model is not None
            if ctc_weight <= 0 or trcfg.timestamps or chunked:
                return super()._transcribe_output_processing(outputs, trcfg)
            enc_states, enc_mask = outputs["encoder_states"], outputs["encoder_mask"]
            encoded_len, prompt = outputs["encoded_lengths"], outputs["decoder_input_ids"]
            hypotheses = super()._transcribe_output_processing(outputs, trcfg)
            if not hypotheses or not all(isinstance(h, Hypothesis) for h in hypotheses):
                return hypotheses  # N-best lists: leave untouched
            return self._select_aed_or_ctc(hypotheses, enc_states, enc_mask, encoded_len, prompt, ctc_weight)
        enc_states = outputs.pop("encoder_states")
        encoded_len = outputs.pop("encoded_lengths")
        log_probs = self.ctc_log_probs(enc_states)
        hypotheses = self.ctc_decoding.ctc_decoder_predictions_tensor(
            log_probs, decoder_lengths=encoded_len, return_hypotheses=trcfg.return_hypotheses
        )
        return hypotheses

    def predict_step(
        self,
        batch: PromptedAudioToTextMiniBatch,
        batch_idx=0,
        dataloader_idx=0,
        has_processed_signal=False,
        timestamps=False,
    ):
        if timestamps:
            raise NotImplementedError("EncDecMultiTaskCTCModel.predict_step does not support timestamps.")
        signal_key = "processed_signal" if has_processed_signal else "input_signal"
        _, encoded_len, enc_states, enc_mask = self.forward(
            **{signal_key: batch.audio, f"{signal_key}_length": batch.audio_lens},
            lang_indices=self.prompt_lang_indices(batch.prompt),
        )
        if self.cur_decoder == "ctc":
            hypotheses = self.ctc_decoding.ctc_decoder_predictions_tensor(
                self.ctc_log_probs(enc_states), decoder_lengths=encoded_len, return_hypotheses=False
            )
        else:
            hypotheses = self.decoding.decode_predictions_tensor(
                encoder_hidden_states=enc_states,
                encoder_input_mask=enc_mask,
                decoder_input_ids=batch.prompt,
                return_hypotheses=False,
            )
        if batch.cuts:
            return list(zip(batch.cuts, hypotheses))
        return hypotheses

    @property
    def oomptimizer_schema(self) -> dict:
        """As in the parent class, but ``transcript`` is real (it is the CTC target) instead of a dummy."""
        schema = super().oomptimizer_schema
        for item in schema["inputs"]:
            if item["name"] == "transcript":
                item.update(
                    type=NeuralType(("B", "T"), LabelsType()),
                    seq_length="output",
                    vocab_size=self.tokenizer.vocab_size,
                )
            elif item["name"] == "transcript_lens":
                item.update(type=NeuralType(("B",), LengthsType()), seq_length="output")
        return schema
