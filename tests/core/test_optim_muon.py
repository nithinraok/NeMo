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

import copy
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from nemo.core.optim.lr_scheduler import CosineAnnealing
from nemo.core.optim.muon import (
    NS_COEFFICIENT_SETS,
    MuonWithAdamW,
    split_param_groups_for_muon,
    zeropower_via_newtonschulz,
)
from nemo.core.optim.optimizers import get_optimizer


class ToyASR(nn.Module):
    """Module names mirror the NeMo transducer models (encoder / decoder.prediction / joint)."""

    def __init__(self, d=16, vocab=11, pred=8):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.pre_encode = nn.Module()
        self.encoder.pre_encode.proj = nn.Linear(2 * d, d, bias=False)
        self.encoder.norm = nn.LayerNorm(d)
        self.encoder.attn = nn.Module()
        self.encoder.attn.w_qkv = nn.Linear(d, 3 * d, bias=False)
        self.encoder.attn.out_proj = nn.Linear(d, d)
        self.encoder.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.prompt_input_embedding = nn.Embedding(4, d)
        self.decoder = nn.Module()
        self.decoder.prediction = nn.ModuleDict({"embed": nn.Embedding(vocab, pred), "dec_rnn": nn.LSTM(pred, pred)})
        self.joint = nn.Module()
        self.joint.enc = nn.Linear(d, pred)
        self.joint.pred = nn.Linear(pred, pred)
        self.joint.joint_net = nn.Sequential(nn.ReLU(), nn.Linear(pred, vocab))

    def forward(self, x, tokens):
        h = self.encoder.pre_encode.proj(x)
        q, k, v = self.encoder.attn.w_qkv(self.encoder.norm(h)).chunk(3, dim=-1)
        att = torch.softmax(q @ k.transpose(-1, -2) / 4.0, dim=-1) @ v
        h = h + self.encoder.attn.out_proj(att) + self.prompt_input_embedding.weight[0]
        h = h + self.encoder.ffn(h)
        g, _ = self.decoder.prediction["dec_rnn"](self.decoder.prediction["embed"](tokens))
        logits = self.joint.joint_net(self.joint.enc(h).unsqueeze(2) + self.joint.pred(g).unsqueeze(1))
        return logits.log_softmax(-1)


def _groups(model, **kwargs):
    return split_param_groups_for_muon(
        [{"params": list(model.parameters())}],
        named_parameters=model.named_parameters(),
        named_modules=model.named_modules(),
        **kwargs,
    )


def _loss(model, batch):
    x, tokens = batch
    return -model(x, tokens)[..., 0].mean()


def _batch(seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(2, 5, 32, generator=gen), torch.randint(0, 11, (2, 3), generator=gen)


class TestNewtonSchulz:
    @pytest.mark.unit
    @pytest.mark.parametrize("shape", [(64, 128), (128, 64), (96, 96)])
    def test_output_is_approximately_orthogonal(self, shape):
        torch.manual_seed(0)
        out = zeropower_via_newtonschulz(torch.randn(*shape))
        assert out.dtype == torch.bfloat16 and out.shape == shape
        singular_values = torch.linalg.svdvals(out.float())
        # Keller Jordan's coefficients trade exactness for speed: singular values land in roughly [0.7, 1.2].
        assert singular_values.min() > 0.5 and singular_values.max() < 1.5

    @pytest.mark.unit
    def test_polar_express_is_closer_to_orthogonal(self):
        torch.manual_seed(0)
        grad = torch.randn(64, 128)
        keller = torch.linalg.svdvals(zeropower_via_newtonschulz(grad, "keller").float())
        polar = torch.linalg.svdvals(zeropower_via_newtonschulz(grad, "polar_express").float())
        assert (polar - 1).abs().max() < (keller - 1).abs().max()
        # An explicit list of per-iteration triples is the same as the named set.
        explicit = zeropower_via_newtonschulz(grad, [list(t) for t in NS_COEFFICIENT_SETS["polar_express"]])
        torch.testing.assert_close(explicit, zeropower_via_newtonschulz(grad, "polar_express"), atol=0, rtol=0)

    @pytest.mark.unit
    def test_batched_equals_per_matrix(self):
        torch.manual_seed(0)
        blocks = torch.randn(3, 16, 32)
        batched = zeropower_via_newtonschulz(blocks)
        for i in range(3):
            single = zeropower_via_newtonschulz(blocks[i]).float()
            torch.testing.assert_close(batched[i].float(), single, atol=2e-2, rtol=0)

    @pytest.mark.unit
    @pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="torch.optim.Muon needs torch >= 2.9")
    def test_matches_torch_reference(self):
        from torch.optim._muon import _zeropower_via_newtonschulz

        torch.manual_seed(0)
        grad = torch.randn(48, 24)
        ours = zeropower_via_newtonschulz(grad)
        ref = _zeropower_via_newtonschulz(grad, (3.4445, -4.7750, 2.0315), 5, 1e-7)
        torch.testing.assert_close(ours, ref, atol=0, rtol=0)


class TestParamSplit:
    @pytest.mark.unit
    def test_every_parameter_exactly_once(self):
        model = ToyASR()
        groups = _groups(model)
        seen = [id(p) for g in groups for p in g["params"]]
        assert sorted(seen) == sorted(id(p) for p in model.parameters())
        # torch.optim.Optimizer also rejects duplicates across groups.
        MuonWithAdamW(groups, lr=1e-3)

    @pytest.mark.unit
    def test_routing(self):
        model = ToyASR()
        groups = _groups(model)
        names = {id(p): n for n, p in model.named_parameters()}
        muon = {names[id(p)]: g["ns_splits"] for g in groups if g["use_muon"] for p in g["params"]}
        adamw = {names[id(p)] for g in groups if not g["use_muon"] for p in g["params"]}
        assert muon == {
            "encoder.pre_encode.proj.weight": 1,
            "encoder.attn.w_qkv.weight": 3,
            "encoder.attn.out_proj.weight": 1,
            "encoder.ffn.0.weight": 1,
            "encoder.ffn.2.weight": 1,
            "joint.enc.weight": 1,
            "joint.pred.weight": 1,
        }
        assert "joint.joint_net.1.weight" in adamw  # output head
        assert "prompt_input_embedding.weight" in adamw  # embedding
        assert "decoder.prediction.embed.weight" in adamw
        assert "decoder.prediction.dec_rnn.weight_hh_l0" in adamw  # LSTM stays on AdamW by default
        assert "encoder.norm.weight" in adamw and "encoder.attn.out_proj.bias" in adamw

    @pytest.mark.unit
    def test_patterns_force_lstm_to_muon_with_gate_split(self):
        model = ToyASR()
        groups = _groups(
            model,
            muon_patterns=[r"dec_rnn\.weight_(ih|hh)_l\d+$"],
            split_patterns={r"w_qkv\.weight$": 3, r"dec_rnn\.weight_(ih|hh)_l\d+$": 4},
        )
        names = {id(p): n for n, p in model.named_parameters()}
        muon = {names[id(p)]: g["ns_splits"] for g in groups if g["use_muon"] for p in g["params"]}
        assert muon["decoder.prediction.dec_rnn.weight_ih_l0"] == 4
        assert muon["decoder.prediction.dec_rnn.weight_hh_l0"] == 4

    @pytest.mark.unit
    def test_group_options_are_preserved(self):
        model = ToyASR()
        groups = split_param_groups_for_muon(
            [{"params": list(model.encoder.parameters()), "lr": 0.5}, {"params": list(model.joint.parameters())}],
            named_parameters=model.named_parameters(),
            named_modules=model.named_modules(),
        )
        assert {g["name"] for g in groups} == {"muon_g0", "muon_split3_g0", "adamw_g0", "muon_g1", "adamw_g1"}
        assert all(g["lr"] == 0.5 for g in groups if g["name"].endswith("_g0"))
        assert all("lr" not in g for g in groups if g["name"].endswith("_g1"))

    @pytest.mark.unit
    def test_registry_and_plain_param_fallback(self):
        model = ToyASR()
        opt = get_optimizer("muon", lr=1e-3)(model.parameters())
        assert isinstance(opt, MuonWithAdamW)
        assert all(p.ndim == 2 for g in opt.param_groups if g["use_muon"] for p in g["params"])


class TestMuonWithAdamW:
    @pytest.mark.unit
    @pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="torch.optim.Muon needs torch >= 2.9")
    @pytest.mark.parametrize("adjust_lr_fn", ["original", "match_rms_adamw"])
    def test_muon_group_matches_torch_muon(self, adjust_lr_fn):
        torch.manual_seed(0)
        ours, ref = nn.Parameter(torch.randn(24, 40)), nn.Parameter(torch.randn(24, 40))
        ref.data.copy_(ours.data)
        opt = MuonWithAdamW(
            [{"params": [ours], "use_muon": True}], lr=0.02, weight_decay=0.1, adjust_lr_fn=adjust_lr_fn
        )
        ref_opt = torch.optim.Muon([ref], lr=0.02, weight_decay=0.1, adjust_lr_fn=adjust_lr_fn)
        for step in range(4):
            grad = torch.randn(24, 40, generator=torch.Generator().manual_seed(step))
            ours.grad, ref.grad = grad.clone(), grad.clone()
            opt.step()
            ref_opt.step()
        torch.testing.assert_close(ours.data, ref.data, atol=1e-6, rtol=0)

    @pytest.mark.unit
    def test_adamw_group_matches_torch_adamw(self):
        torch.manual_seed(0)
        ours = [nn.Parameter(torch.randn(7)), nn.Parameter(torch.randn(3, 5))]
        ref = [nn.Parameter(p.detach().clone()) for p in ours]
        opt = MuonWithAdamW([{"params": ours, "use_muon": False}], lr=1e-2, weight_decay=0.05, betas=(0.9, 0.95))
        ref_opt = torch.optim.AdamW(ref, lr=1e-2, weight_decay=0.05, betas=(0.9, 0.95), eps=1e-8)
        for step in range(5):
            gen = torch.Generator().manual_seed(step)
            for p, r in zip(ours, ref):
                p.grad = torch.randn(p.shape, generator=gen)
                r.grad = p.grad.clone()
            opt.step()
            ref_opt.step()
        for p, r in zip(ours, ref):
            torch.testing.assert_close(p.data, r.data, atol=1e-6, rtol=1e-6)

    @pytest.mark.unit
    def test_split_update_is_blockwise(self):
        torch.manual_seed(0)
        weight = nn.Parameter(torch.zeros(48, 16))
        weight.grad = torch.randn(48, 16)
        opt = MuonWithAdamW(
            [{"params": [weight], "use_muon": True, "ns_splits": 3}],
            lr=1.0,
            weight_decay=0.0,
            nesterov=False,
            momentum=0.0,
            adjust_lr_fn="none",
        )
        grad = weight.grad.clone()
        opt.step()
        expected = zeropower_via_newtonschulz(grad.view(3, 16, 16)).reshape(48, 16).float()
        torch.testing.assert_close(weight.data, -expected, atol=0, rtol=0)
        whole = zeropower_via_newtonschulz(grad).float()
        assert (weight.data + whole).abs().max() > 1e-2  # orthogonalising the fused matrix is different

    @pytest.mark.unit
    def test_loss_decreases(self):
        torch.manual_seed(0)
        model = ToyASR()
        opt = MuonWithAdamW(_groups(model), lr=2e-2, weight_decay=0.0, betas=(0.9, 0.95))
        batch = _batch()
        initial = _loss(model, batch).item()
        for _ in range(30):
            opt.zero_grad()
            _loss(model, batch).backward()
            opt.step()
        assert _loss(model, batch).item() < 0.5 * initial

    @pytest.mark.unit
    def test_state_dict_round_trip(self):
        torch.manual_seed(0)
        model = ToyASR()
        opt = MuonWithAdamW(_groups(model), lr=1e-2, weight_decay=0.01)

        def train(m, o, steps, seed):
            for i in range(steps):
                o.zero_grad()
                _loss(m, _batch(seed + i)).backward()
                o.step()

        train(model, opt, 3, seed=0)
        resumed_model = copy.deepcopy(model)
        resumed_opt = MuonWithAdamW(_groups(resumed_model), lr=1e-2, weight_decay=0.01)
        resumed_opt.load_state_dict(copy.deepcopy(opt.state_dict()))

        train(model, opt, 2, seed=3)
        train(resumed_model, resumed_opt, 2, seed=3)
        for p, q in zip(model.parameters(), resumed_model.parameters()):
            torch.testing.assert_close(p, q, atol=0, rtol=0)

    @pytest.mark.unit
    def test_nemo_scheduler_scales_every_group(self):
        model = ToyASR()
        opt = MuonWithAdamW(_groups(model), lr=1e-3, adamw_lr=4e-3)
        sched = CosineAnnealing(opt, warmup_steps=4, max_steps=20, min_lr=0.0)
        for _ in range(10):
            opt.step()
            sched.step()
        lrs = {g["name"]: g["lr"] for g in opt.param_groups}
        assert lrs["adamw"] == pytest.approx(4 * lrs["muon"])
        assert 0 < lrs["muon"] < 1e-3


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _sharded_worker(rank, world_size, port, result):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(0)
        model = ToyASR()
        opt = MuonWithAdamW(_groups(model), lr=1e-2, distributed_mode="world")
        for i in range(3):
            opt.zero_grad()
            _loss(model, _batch(i)).backward()  # same batch on every rank stands in for DDP's all-reduce
            opt.step()
        result[rank] = [p.detach().clone() for p in model.parameters()]
    finally:
        dist.destroy_process_group()


class TestShardedMuon:
    @pytest.mark.unit
    def test_sharded_matches_replicated(self):
        torch.manual_seed(0)
        model = ToyASR()
        opt = MuonWithAdamW(_groups(model), lr=1e-2)
        for i in range(3):
            opt.zero_grad()
            _loss(model, _batch(i)).backward()
            opt.step()

        with mp.Manager() as manager:
            result = manager.dict()
            mp.spawn(_sharded_worker, args=(2, _free_port(), result), nprocs=2, join=True)
            for rank in range(2):
                for p, q in zip(model.parameters(), result[rank]):
                    torch.testing.assert_close(p.detach(), q, atol=0, rtol=0)


class TestBatchedMuon:
    @pytest.mark.unit
    def test_stacked_buckets_match_per_matrix(self):
        """Equally oriented matrices (incl. a transposed one) share one batched call and equal per-matrix NS."""
        torch.manual_seed(0)
        shapes = [(16, 64), (64, 16), (16, 64), (48, 16)]  # (64, 16) is transposed into the (16, 64) bucket
        params = [nn.Parameter(torch.zeros(shape)) for shape in shapes]
        grads = [torch.randn(shape) for shape in shapes]
        for p, g in zip(params, grads):
            p.grad = g.clone()
        groups = [
            {"params": params[:3], "use_muon": True},
            {"params": params[3:], "use_muon": True, "ns_splits": 3},
        ]
        opt = MuonWithAdamW(groups, lr=1.0, weight_decay=0.0, nesterov=False, momentum=0.0, adjust_lr_fn="none")
        opt.step()
        for p, g in zip(params[:3], grads[:3]):
            expected = zeropower_via_newtonschulz(g).float()
            torch.testing.assert_close(p.data, -expected, atol=2e-2, rtol=0)
        expected = zeropower_via_newtonschulz(grads[3].view(3, 16, 16)).reshape(48, 16).float()
        torch.testing.assert_close(params[3].data, -expected, atol=0, rtol=0)

    @pytest.mark.unit
    def test_small_batch_limit_matches_one_call(self):
        torch.manual_seed(0)
        model_a, model_b = ToyASR(), ToyASR()
        model_b.load_state_dict(model_a.state_dict())
        opt_a = MuonWithAdamW(_groups(model_a), lr=1e-2)
        opt_b = MuonWithAdamW(_groups(model_b), lr=1e-2, ns_max_batch_numel=1)  # one matrix per call
        for i in range(2):
            for model, opt in ((model_a, opt_a), (model_b, opt_b)):
                opt.zero_grad()
                _loss(model, _batch(i)).backward()
                opt.step()
        for p, q in zip(model_a.parameters(), model_b.parameters()):
            torch.testing.assert_close(p, q, atol=1e-5, rtol=0)

    @pytest.mark.unit
    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="needs torch.compile")
    def test_compiled_newton_schulz_matches_eager(self):
        """Inductor keeps bf16 intermediates in fp32 inside fused kernels, so the compiled result is not bitwise equal
        to eager; it must be as orthogonal and within bf16 Newton-Schulz noise of it."""
        torch.manual_seed(0)
        grads = torch.randn(4, 32, 96)
        eager = zeropower_via_newtonschulz(grads).float()
        compiled = torch.compile(zeropower_via_newtonschulz, dynamic=False)(grads).float()
        rel = (compiled - eager).norm() / eager.norm()
        assert rel < 0.05, rel
        for out in (eager, compiled):
            singular = torch.linalg.svdvals(out)
            assert singular.min() > 0.5 and singular.max() < 1.6, singular

        model = ToyASR()
        opt = MuonWithAdamW(_groups(model), lr=1e-2, compile_ns=True)
        before = _loss(model, _batch(0)).item()
        for i in range(10):
            opt.zero_grad()
            _loss(model, _batch(0)).backward()
            opt.step()
        assert _loss(model, _batch(0)).item() < before
