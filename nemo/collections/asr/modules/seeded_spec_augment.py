# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SpecAugment with a checkpointed RNG stream independent of model dropout."""
import torch

from nemo.collections.asr.modules.audio_preprocessing import SpectrogramAugmentation


class SeededSpectrogramAugmentation(SpectrogramAugmentation):
    """Seed vectorized frequency/time masks by call count and distributed rank.

    The counter is checkpointed so wall-time resumes preserve augmentation order.
    Model depth and dropout do not affect this RNG stream. Rectangle masks and the
    legacy Python RNG implementation are intentionally unsupported.
    """

    def __init__(self, augmentation_seed: int = 42, **kwargs):
        if not kwargs.get("use_vectorized_spec_augment", True) or kwargs.get("rect_masks", 0):
            raise ValueError("Seeded augmentation requires vectorized frequency/time masking only.")
        super().__init__(**kwargs)
        self.augmentation_seed = augmentation_seed
        self.register_buffer("_augmentation_calls", torch.zeros((), dtype=torch.int64))

    def forward(self, input_spec, length=None):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        seed = self.augmentation_seed + 1000003 * int(self._augmentation_calls.item()) + 10007 * rank
        devices = [input_spec.device.index] if input_spec.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if input_spec.is_cuda:
                with torch.cuda.device(input_spec.device):
                    torch.cuda.manual_seed(seed)
            output = super().forward(input_spec=input_spec, length=length)
        self._augmentation_calls.add_(1)
        return output
