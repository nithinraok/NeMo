# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline locality ablations: masks, narrow width, initialization, and gradients."""
import pytest
import torch
import torch.nn.functional as F

from nemo.collections.asr.modules import transformer_encoder as module


def encoder(**kwargs):
    return module.TransformerEncoder(
        feat_in=8,
        d_model=64,
        n_heads=4,
        n_layers=2,
        subsampling_factor=1,
        drop_rate=0.0,
        self_attention_model='rope',
        qk_norm=True,
        rotary_fraction=0.5,
        pos_emb_max_len=32,
        sync_max_audio_length=False,
        **kwargs,
    )


@pytest.mark.unit
def test_shared_initialization_and_added_width():
    models, next_random = [], []
    for kwargs in (
        {},
        {'local_attention_heads': 1},
        {'extra_attention_dim': 16, 'extra_attention_heads': 1, 'extra_attention_context': (1, 1)},
        {'extra_attention_dim': 16, 'extra_attention_heads': 1},
    ):
        torch.manual_seed(42)
        models.append(encoder(**kwargs))
        next_random.append(torch.rand(3))
    baseline = models[0].state_dict()
    for model, rng in zip(models[1:], next_random[1:]):
        assert torch.equal(next_random[0], rng)
        state = model.state_dict()
        for name, value in baseline.items():
            assert torch.equal(value, state[name]), name
    for name, value in models[2].state_dict().items():
        assert torch.equal(value, models[3].state_dict()[name]), name
    counts = [sum(p.numel() for p in m.parameters()) for m in models]
    assert counts[0] == counts[1]
    assert counts[2] == counts[3] == counts[0] + 2 * (4 * 64 * 16 + 64 + 2 * 64 + 4 * 16)
    assert models[2].layers[0].extra_attn.w_qkv.weight.shape == (48, 64)
    assert models[2].layers[0].extra_attn.out_proj.weight.shape == (64, 16)


@pytest.mark.unit
@pytest.mark.parametrize('variant', ['baseline', 'mixed', 'local', 'global'])
def test_flex_matches_dense_reference_and_backward(monkeypatch, variant):
    options = {
        'baseline': {},
        'mixed': dict(local_attention_heads=1, local_attention_context=(1, 1)),
        'local': dict(extra_attention_dim=16, extra_attention_heads=1, extra_attention_context=(1, 1)),
        'global': dict(extra_attention_dim=16, extra_attention_heads=1),
    }
    torch.manual_seed(0)
    model = encoder(**options[variant]).eval()
    x = torch.randn(2, 8, 11, requires_grad=True)
    lengths = torch.tensor([11, 7])
    # CPU FlexAttention supports forward only; validate its output against a differentiable dense oracle.
    with torch.no_grad():
        actual, actual_lengths = model(x, lengths)
    calls = []

    def dense(q, k, v, block_mask=None, score_mod=None):
        assert score_mod is None
        batch = torch.arange(q.shape[0])[:, None, None, None]
        head = torch.arange(q.shape[1])[None, :, None, None]
        query = torch.arange(q.shape[2])[None, None, :, None]
        key = torch.arange(k.shape[2])[None, None, None, :]
        mask = block_mask.mask_mod(batch, head, query, key)
        calls.append(mask)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    monkeypatch.setattr(module._transformer_utils, '_get_flex_attention', lambda _: dense)
    expected, _ = model(x, lengths)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    assert torch.equal(actual_lengths, lengths)
    if variant == 'mixed':
        assert calls[0][0, 0, 5, 0]  # Global head sees distant keys.
        assert not calls[0][0, 3, 5, 0]
        assert calls[0][0, 3, 5, 4] and calls[0][0, 3, 5, 6]
    if variant == 'local':
        assert not calls[1][0, 0, 5, 0]
        assert calls[1][0, 0, 5, 4] and calls[1][0, 0, 5, 6]
    for mask in calls:
        assert not mask[1, ..., 7:].any()  # Every stage excludes padded keys.
    expected.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    for name, param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name


@pytest.mark.unit
@pytest.mark.parametrize(
    'kwargs',
    [
        dict(local_attention_heads=5),
        dict(local_attention_context=(-1, 4)),
        dict(extra_attention_dim=24, extra_attention_heads=1),
        dict(extra_attention_dim=16, extra_attention_context=(-1, 4)),
    ],
)
def test_reject_invalid_locality_config(kwargs):
    with pytest.raises(ValueError):
        encoder(**kwargs)


@pytest.mark.unit
def test_unsupported_execution_paths_fail_explicitly():
    model = encoder(local_attention_heads=1)
    with pytest.raises(NotImplementedError, match='padded offline'):
        model.forward_sequence_packed(torch.randn(1, 8, 10), torch.tensor([10]))
    with pytest.raises(NotImplementedError, match='streaming'):
        module.StreamingTransformerEncoder(d_model=64, n_heads=4, n_layers=1, local_attention_heads=1)


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason='Compiled FlexAttention backward requires CUDA')
@pytest.mark.parametrize('variant', ['baseline', 'mixed', 'local', 'global'])
def test_cuda_backward_with_production_head_dimensions(variant):
    options = {
        'baseline': {},
        'mixed': dict(local_attention_heads=4, local_attention_context=(4, 4)),
        'local': dict(extra_attention_dim=320, extra_attention_heads=4, extra_attention_context=(4, 4)),
        'global': dict(extra_attention_dim=320, extra_attention_heads=4),
    }
    model = (
        module.TransformerEncoder(
            feat_in=128,
            d_model=1280,
            n_heads=16,
            n_layers=2,
            subsampling_factor=8,
            self_attention_model='rope',
            qk_norm=True,
            rotary_fraction=0.5,
            pos_emb_max_len=512,
            sync_max_audio_length=False,
            **options[variant],
        )
        .cuda()
        .train()
    )
    x = torch.randn(2, 128, 2048, device='cuda', requires_grad=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output, _ = model(x, torch.tensor([2048, 1800], device='cuda'))
        loss = (output * torch.randn_like(output)).mean()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    for name, param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
