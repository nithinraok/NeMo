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

"""Muon for hidden-layer matrices, with an auxiliary AdamW for every other parameter, in one optimizer.

Muon (https://kellerjordan.github.io/posts/muon/) replaces the update of a 2D weight by the (approximate)
orthogonalisation of its Nesterov momentum, computed with a few Newton-Schulz iterations. It is meant for hidden
weight matrices only; embeddings, output heads, norms and biases stay on AdamW. Keeping both in a single
``torch.optim.Optimizer`` lets Lightning, NeMo LR schedulers, gradient clipping and checkpointing treat it like
any other optimizer.

Parameter groups should carry ``use_muon``; build them with :func:`split_param_groups_for_muon` (``ModelPT`` does
this automatically for ``optim.name: muon``). A plain parameter list falls back to routing by ``ndim``.
"""

import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.optim.optimizer import Optimizer

from nemo.utils import logging

__all__ = [
    "DEFAULT_ADAMW_PATTERNS",
    "DEFAULT_SPLIT_PATTERNS",
    "MuonWithAdamW",
    "NS_COEFFICIENT_SETS",
    "split_param_groups_for_muon",
    "zeropower_via_newtonschulz",
]

# Quintic Newton-Schulz coefficients (a, b, c) per iteration; the last entry repeats if ns_steps is larger.
# "keller": Keller Jordan's reference implementation, also torch.optim.Muon's default.
# "polar_express": Polar Express (arXiv 2505.16932), per-iteration minimax coefficients incl. the 1.01 safety factor,
# as tabulated in Transformer Engine's transformer_engine/pytorch/newton_schulz.py.
NS_COEFFICIENT_SETS = {
    "keller": ((3.4445, -4.7750, 2.0315),),
    "polar_express": (
        (8.2051, -22.9019, 16.4607),
        (4.0664, -2.8612, 0.5184),
        (3.9096, -2.8234, 0.5250),
        (3.2856, -2.4153, 0.4853),
        (2.2779, -1.6198, 0.3985),
        (1.8726, -1.2307, 0.3585),
        (1.8564, -1.2132, 0.3568),
        (1.8750, -1.2500, 0.3750),
    ),
}

# Output layers of NeMo ASR heads (RNN-T/TDT joint output, CTC decoder) are nn.Linear/Conv1d but are not hidden
# layers, so they stay on AdamW.
DEFAULT_ADAMW_PATTERNS = (r"(^|\.)joint_net\.", r"(^|\.)ctc_decoder\.")

# Fused projections orthogonalised block by block: TransformerEncoder's w_qkv stacks Q, K and V along the rows.
DEFAULT_SPLIT_PATTERNS = ({"pattern": r"(^|\.)w_qkv\.weight$", "num_splits": 3},)

_ADJUST_LR_FNS = ("original", "match_rms_adamw", "none")
_DISTRIBUTED_MODES = ("none", "world", "node")


class MuonWithAdamW(Optimizer):
    """Muon for groups with ``use_muon=True``, AdamW for groups with ``use_muon=False``.

    Muon update for a weight ``W`` (rows x cols) with gradient ``G``::

        B <- momentum * B + (1 - momentum) * G
        U <- (1 - momentum) * G + momentum * B            (nesterov) or B
        O <- NewtonSchulz(U)  (per row block if ns_splits > 1)
        W <- W * (1 - lr * weight_decay) - lr * scale(rows, cols) * O

    ``scale`` is ``0.2 * sqrt(max(rows, cols))`` for ``adjust_lr_fn='match_rms_adamw'`` (Moonlight, arXiv
    2502.16982: update RMS ~0.2 like AdamW, so the AdamW ``lr``/``weight_decay`` can be reused) or
    ``sqrt(max(1, rows / cols))`` for ``'original'`` (Keller Jordan's, typically with ``lr ~ 0.02``).

    Args:
        params: parameter groups, each with a ``use_muon`` key (see :func:`split_param_groups_for_muon`).
        lr: learning rate of the Muon groups, and of the AdamW groups unless ``adamw_lr`` is given.
        weight_decay: decoupled weight decay of the Muon groups, and of the AdamW groups unless
            ``adamw_weight_decay`` is given.
        momentum: Muon momentum.
        nesterov: use Nesterov momentum for Muon.
        ns_steps: number of Newton-Schulz iterations.
        ns_coefficients: a name from :data:`NS_COEFFICIENT_SETS` (``'keller'``, ``'polar_express'``), one
            ``(a, b, c)`` triple, or a list of per-iteration triples (the last one repeats).
        ns_eps: epsilon of the Frobenius normalisation before Newton-Schulz.
        adjust_lr_fn: ``'match_rms_adamw'``, ``'original'`` or ``'none'``.
        betas: AdamW betas.
        eps: AdamW epsilon.
        adamw_lr: learning rate of the AdamW groups (default: ``lr``).
        adamw_weight_decay: weight decay of the AdamW groups (default: ``weight_decay``).
        distributed_mode: ``'none'`` computes Newton-Schulz for every matrix on every rank (gradients are already
            all-reduced by DDP, so all ranks compute the same update). ``'world'`` / ``'node'`` assign each matrix to
            one rank of the whole job / of the node, which orthogonalises it and all-gathers the bf16 update.
            Momentum buffers stay replicated on every rank, so checkpoints saved by rank 0 are complete.
        compile_ns: run Newton-Schulz through ``torch.compile`` (fuses the normalisation and polynomial
            element-wise work; compiled once per matrix-stack shape).
        ns_max_batch_numel: upper bound on the elements of one batched Newton-Schulz call (bounds the temporary
            memory of a stack of equally shaped matrices).

    Matrices are orthogonalised in stacks: every Muon matrix (or row block, with ``ns_splits``) is oriented so that
    rows <= cols, and all matrices with the same oriented shape and Newton-Schulz settings go through one batched
    call. This replaces ~100 small per-matrix launches per step by a handful of large batched matmuls.
    """

    def __init__(
        self,
        params: Iterable[Dict[str, Any]],
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_coefficients: Union[str, Sequence[float], Sequence[Sequence[float]]] = "keller",
        ns_eps: float = 1e-7,
        adjust_lr_fn: str = "match_rms_adamw",
        betas: Sequence[float] = (0.9, 0.999),
        eps: float = 1e-8,
        adamw_lr: Optional[float] = None,
        adamw_weight_decay: Optional[float] = None,
        distributed_mode: str = "none",
        compile_ns: bool = False,
        ns_max_batch_numel: int = 2**30,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if adjust_lr_fn not in _ADJUST_LR_FNS:
            raise ValueError(f"adjust_lr_fn must be one of {_ADJUST_LR_FNS}, got {adjust_lr_fn!r}")
        if distributed_mode not in _DISTRIBUTED_MODES:
            raise ValueError(f"distributed_mode must be one of {_DISTRIBUTED_MODES}, got {distributed_mode!r}")
        ns_coefficients = _ns_coefficient_schedule(ns_coefficients)

        params = list(params)
        if params and all(isinstance(p, torch.Tensor) for p in params):
            logging.warning(
                "MuonWithAdamW got a plain parameter list: routing every 2D tensor to Muon and the rest to AdamW. "
                "Embeddings and output layers should be on AdamW; use split_param_groups_for_muon instead."
            )
            params = [
                {"params": [p for p in params if p.ndim == 2], "use_muon": True},
                {"params": [p for p in params if p.ndim != 2], "use_muon": False},
            ]
            params = [group for group in params if group["params"]]
        if not params or not all(isinstance(group, dict) and "use_muon" in group for group in params):
            raise ValueError(
                "MuonWithAdamW expects parameter groups with a 'use_muon' key; build them with "
                "nemo.core.optim.muon.split_param_groups_for_muon."
            )
        for group in params:
            if not group["use_muon"]:
                if adamw_lr is not None:
                    group.setdefault("lr", adamw_lr)
                if adamw_weight_decay is not None:
                    group.setdefault("weight_decay", adamw_weight_decay)

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_coefficients=ns_coefficients,
            ns_eps=ns_eps,
            ns_splits=1,
            adjust_lr_fn=adjust_lr_fn,
            betas=tuple(betas),
            eps=eps,
        )
        super().__init__(params, defaults)
        self.distributed_mode = distributed_mode
        self._shard_group = None
        self._shard_group_ready = False
        self.ns_max_batch_numel = int(ns_max_batch_numel)
        self._ns_fn = torch.compile(zeropower_via_newtonschulz, dynamic=False) if compile_ns else None

        for group in self.param_groups:
            if not group["use_muon"]:
                continue
            for p in group["params"]:
                if p.ndim < 2:
                    raise ValueError(f"Muon needs parameters with ndim >= 2, got shape {tuple(p.shape)}")
                if p.shape[0] % group["ns_splits"] != 0:
                    raise ValueError(
                        f"Shape {tuple(p.shape)} has rows not divisible by ns_splits={group['ns_splits']}"
                    )

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        muon_work = []
        for group in self.param_groups:
            if group["use_muon"]:
                params = [p for p in group["params"] if p.grad is not None]
                if any(p.grad.is_sparse for p in params):
                    raise RuntimeError("Muon does not support sparse gradients")
                for p in params:
                    if "momentum_buffer" not in self.state[p]:
                        self.state[p]["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if params:
                    torch._foreach_lerp_(
                        [self.state[p]["momentum_buffer"] for p in params],
                        [p.grad for p in params],
                        1 - group["momentum"],
                    )
                muon_work.extend((p, group) for p in params)
            else:
                self._adamw_step(group)

        if muon_work:
            updates = self._orthogonalised_updates(muon_work, self._get_shard_group())
            for (p, group), update in zip(muon_work, updates):
                self._apply_muon_update(p, group, update)
        return loss

    def _muon_input(self, p: torch.Tensor, group: Dict[str, Any]) -> Tuple[torch.Tensor, bool]:
        """(Nesterov) momentum of ``p`` as a bf16 stack ``(ns_splits, rows, cols)`` with rows <= cols.

        Returns the stack and whether it was transposed to get rows <= cols.
        """
        buf = self.state[p]["momentum_buffer"]
        update = p.grad.lerp(buf, group["momentum"]) if group["nesterov"] else buf
        matrix = update.reshape(group["ns_splits"], p.shape[0] // group["ns_splits"], -1).bfloat16()
        transposed = matrix.shape[1] > matrix.shape[2]
        return (matrix.mT if transposed else matrix), transposed

    def _orthogonalised_updates(
        self, muon_work: List[Tuple[torch.Tensor, Dict[str, Any]]], shard_group
    ) -> List[torch.Tensor]:
        """Orthogonalised momentum of every Muon parameter (bf16, shape of the parameter), in ``muon_work`` order.

        Parameters are bucketed by their oriented stack shape and Newton-Schulz settings; each bucket is
        orthogonalised in batched calls. With a ``shard_group``, the parameters of a bucket are dealt round-robin to
        its ranks, each rank orthogonalises its share, and one all-gather per bucket collects the results. The
        bucket order follows ``muon_work``, which is identical on all ranks.
        """
        buckets = defaultdict(list)  # (stack shape, NS settings) -> indices into muon_work
        oriented = []
        for idx, (p, group) in enumerate(muon_work):
            rows = p.shape[0] // group["ns_splits"]
            cols = p[0].numel()
            shape = (group["ns_splits"], min(rows, cols), max(rows, cols))
            key = (shape, group["ns_coefficients"], group["ns_steps"], group["ns_eps"])
            buckets[key].append(idx)
            oriented.append(rows > cols)

        world_size = dist.get_world_size(shard_group) if shard_group is not None else 1
        rank = dist.get_rank(shard_group) if shard_group is not None else 0
        results: List[Optional[torch.Tensor]] = [None] * len(muon_work)
        pending = []
        for (shape, coefficients, ns_steps, ns_eps), indices in buckets.items():
            mine = indices[rank::world_size]
            stacks = [self._muon_input(*muon_work[i])[0] for i in mine]
            # A bucket holding a single matrix uses the 2D path (same kernels as torch.optim.Muon); every other bucket
            # uses the batched path on every rank, so sharded and replicated runs compute identical updates.
            single = len(indices) * shape[0] == 1
            local = self._newton_schulz_stacked(
                stacks, shape, coefficients, ns_steps, ns_eps, muon_work[0][0].device, single
            )
            if shard_group is None:
                for i, ortho in zip(mine, local):
                    results[i] = ortho
                continue
            # Pad so every rank contributes the same number of stacks to the all-gather.
            per_rank = -(-len(indices) // world_size)
            padded = torch.zeros((per_rank,) + shape, dtype=torch.bfloat16, device=muon_work[0][0].device)
            if mine:
                padded[: len(mine)] = local
            gathered = [torch.empty_like(padded) for _ in range(world_size)]
            handle = dist.all_gather(gathered, padded, group=shard_group, async_op=True)
            pending.append((handle, indices, gathered))

        for handle, indices, gathered in pending:
            handle.wait()
            for position, i in enumerate(indices):
                results[i] = gathered[position % world_size][position // world_size]

        updates = []
        for (p, _), ortho, transposed in zip(muon_work, results, oriented):
            updates.append((ortho.mT if transposed else ortho).reshape(p.shape))
        return updates

    def _newton_schulz_stacked(
        self,
        stacks: List[torch.Tensor],
        shape: Tuple[int, int, int],
        coefficients,
        ns_steps: int,
        ns_eps: float,
        device: torch.device,
        single: bool = False,
    ) -> torch.Tensor:
        """Orthogonalise equally shaped ``(splits, rows, cols)`` stacks; returns ``(len(stacks), *shape)`` bf16."""
        if not stacks:
            return torch.empty((0,) + shape, dtype=torch.bfloat16, device=device)
        ns_fn = self._ns_fn or zeropower_via_newtonschulz
        matrices = torch.cat(stacks) if len(stacks) > 1 else stacks[0]  # (n * splits, rows, cols)
        per_call = max(1, self.ns_max_batch_numel // (shape[1] * shape[2]))
        if single:
            out = ns_fn(matrices[0], coefficients, ns_steps, ns_eps).unsqueeze(0)
        else:
            out = torch.cat(
                [
                    ns_fn(matrices[start : start + per_call], coefficients, ns_steps, ns_eps)
                    for start in range(0, matrices.shape[0], per_call)
                ]
            )
        return out.view((len(stacks),) + shape)

    @staticmethod
    def _apply_muon_update(p: torch.Tensor, group: Dict[str, Any], update: torch.Tensor) -> None:
        rows = p.shape[0] // group["ns_splits"]
        cols = p[0].numel()
        lr = group["lr"]
        p.mul_(1 - lr * group["weight_decay"])
        p.add_(update.to(p.dtype), alpha=-lr * _lr_scale(rows, cols, group["adjust_lr_fn"]))

    def _get_shard_group(self):
        """Process group used to shard Newton-Schulz, created lazily (every rank reaches the first step)."""
        if self.distributed_mode == "none" or not (dist.is_available() and dist.is_initialized()):
            return None
        if dist.get_world_size() == 1:
            return None
        if not self._shard_group_ready:
            if self.distributed_mode == "world":
                self._shard_group = dist.group.WORLD
            else:
                local_size = _local_world_size()
                if dist.get_world_size() % local_size != 0:
                    raise RuntimeError(f"World size {dist.get_world_size()} is not a multiple of {local_size}")
                self._shard_group, _ = dist.new_subgroups(group_size=local_size)
            self._shard_group_ready = True
        return self._shard_group

    def _adamw_step(self, group: Dict[str, Any]) -> None:
        """AdamW (decoupled weight decay, bias correction), same math as ``torch.optim.AdamW``."""
        params, grads, exp_avgs, exp_avg_sqs, steps = [], [], [], [], []
        for p in group["params"]:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise RuntimeError("AdamW does not support sparse gradients")
            state = self.state[p]
            if len(state) == 0:
                state["step"] = torch.tensor(0.0)
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state["step"] += 1
            params.append(p)
            grads.append(p.grad)
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            steps.append(state["step"].item())
        if not params:
            return

        lr, weight_decay, eps = group["lr"], group["weight_decay"], group["eps"]
        beta1, beta2 = group["betas"]
        torch._foreach_mul_(params, 1 - lr * weight_decay)
        torch._foreach_lerp_(exp_avgs, grads, 1 - beta1)
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, 1 - beta2)
        denom = torch._foreach_sqrt(exp_avg_sqs)
        torch._foreach_div_(denom, [math.sqrt(1 - beta2**step) for step in steps])
        torch._foreach_add_(denom, eps)
        torch._foreach_addcdiv_(params, exp_avgs, denom, [-lr / (1 - beta1**step) for step in steps])


def split_param_groups_for_muon(
    param_groups: Sequence[Dict[str, Any]],
    named_parameters: Iterable[Tuple[str, torch.Tensor]],
    named_modules: Optional[Iterable[Tuple[str, nn.Module]]] = None,
    adamw_patterns: Optional[Sequence[str]] = None,
    muon_patterns: Optional[Sequence[str]] = None,
    split_patterns: Optional[Union[Sequence[Dict[str, Any]], Dict[str, int]]] = None,
) -> List[Dict[str, Any]]:
    """Split existing parameter groups into Muon and AdamW groups.

    Routing, in order: a name matching ``adamw_patterns`` goes to AdamW; a name matching ``muon_patterns`` goes to
    Muon; the ``weight`` of an ``nn.Linear`` goes to Muon; everything else (embeddings, norms, biases, RNN and conv
    weights, buffers-like scalars) goes to AdamW. Patterns are ``re.search``-ed against the full parameter name.
    Muon parameters whose name matches a ``split_patterns`` entry are orthogonalised per equal row block.

    Group-level options of the input groups (e.g. a per-module ``lr`` from ``optim_param_groups``) are preserved.

    Args:
        param_groups: groups as built by ``ModelPT.setup_optimizer_param_groups`` (dicts with ``params``).
        named_parameters: ``model.named_parameters()``, to name the parameters.
        named_modules: ``model.named_modules()``, to find ``nn.Linear`` owners (default: inferred from names only,
            in which case every 2D ``*.weight`` counts as a Linear weight).
        adamw_patterns: regexes forcing parameters to AdamW (default: :data:`DEFAULT_ADAMW_PATTERNS`).
        muon_patterns: regexes forcing parameters (ndim >= 2) to Muon, e.g. LSTM ``weight_hh_l0``.
        split_patterns: ``[{pattern: regex, num_splits: n}, ...]`` or ``{regex: n}``
            (default: :data:`DEFAULT_SPLIT_PATTERNS`, which splits the fused ``w_qkv`` into Q, K, V).

    Returns:
        A list of groups, each with ``use_muon``, ``name`` and, for Muon, ``ns_splits``.
    """
    adamw_res = [re.compile(p) for p in (DEFAULT_ADAMW_PATTERNS if adamw_patterns is None else adamw_patterns)]
    muon_res = [re.compile(p) for p in (muon_patterns or ())]
    split_res = [(re.compile(pattern), int(n)) for pattern, n in _normalize_split_patterns(split_patterns)]

    names = {id(p): n for n, p in named_parameters}
    modules = dict(named_modules) if named_modules is not None else None

    new_groups = []
    for group_idx, group in enumerate(param_groups):
        options = {k: v for k, v in group.items() if k != "params"}
        buckets = defaultdict(list)  # (use_muon, ns_splits) -> params
        for p in group["params"]:
            if id(p) not in names:
                raise ValueError(f"Parameter of shape {tuple(p.shape)} in group {group_idx} is not in the model")
            name = names[id(p)]
            use_muon = _route_to_muon(name, p, modules, adamw_res, muon_res)
            ns_splits = 1
            if use_muon:
                ns_splits = next((n for regex, n in split_res if regex.search(name)), 1)
            buckets[(use_muon, ns_splits)].append(p)

        for (use_muon, ns_splits), params in sorted(buckets.items(), key=lambda kv: (not kv[0][0], kv[0][1])):
            new_group = dict(options, params=params, use_muon=use_muon)
            suffix = f"_g{group_idx}" if len(param_groups) > 1 else ""
            if use_muon:
                new_group["ns_splits"] = ns_splits
                new_group["name"] = f"muon{'_split' + str(ns_splits) if ns_splits > 1 else ''}{suffix}"
            else:
                new_group["name"] = f"adamw{suffix}"
            new_groups.append(new_group)

    for group in new_groups:
        numel = sum(p.numel() for p in group["params"])
        logging.info(f"Muon param group '{group['name']}': {len(group['params'])} tensors, {numel / 1e6:.2f}M params")
    adamw_matrices = [
        names[id(p)] for g in new_groups if not g["use_muon"] for p in g["params"] if p.ndim >= 2 and p.numel() > 0
    ]
    logging.info(f"Matrices kept on AdamW: {adamw_matrices}")
    return new_groups


def zeropower_via_newtonschulz(
    grad: torch.Tensor,
    ns_coefficients: Union[str, Sequence[float], Sequence[Sequence[float]]] = "keller",
    ns_steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Approximately orthogonalise ``grad`` (``U S V^T -> U S' V^T``, ``S'`` near 1) with quintic Newton-Schulz.

    Same iteration as ``torch.optim.Muon`` (for ``'keller'`` coefficients), extended to per-iteration coefficients
    and to a batch of matrices (leading dimension). Runs in bf16 and returns bf16.
    """
    if grad.ndim not in (2, 3):
        raise ValueError(f"Expected a matrix or a batch of matrices, got shape {tuple(grad.shape)}")
    schedule = _ns_coefficient_schedule(ns_coefficients)
    x = grad.bfloat16()
    transposed = grad.size(-2) > grad.size(-1)
    if transposed:
        x = x.mT
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    for step in range(ns_steps):
        a, b, c = schedule[min(step, len(schedule) - 1)]
        if x.ndim == 2:
            gram = x @ x.T
            poly = torch.addmm(gram, gram, gram, beta=b, alpha=c)
            x = torch.addmm(x, poly, x, beta=a)
        else:
            gram = x @ x.mT
            poly = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
            x = torch.baddbmm(x, poly, x, beta=a)
    if transposed:
        x = x.mT
    return x.contiguous()


def _ns_coefficient_schedule(ns_coefficients) -> Tuple[Tuple[float, float, float], ...]:
    if isinstance(ns_coefficients, str):
        if ns_coefficients not in NS_COEFFICIENT_SETS:
            raise ValueError(f"Unknown ns_coefficients {ns_coefficients!r}; known: {list(NS_COEFFICIENT_SETS)}")
        return NS_COEFFICIENT_SETS[ns_coefficients]
    ns_coefficients = list(ns_coefficients)
    if len(ns_coefficients) == 3 and all(isinstance(v, (int, float)) for v in ns_coefficients):
        return (tuple(float(v) for v in ns_coefficients),)
    schedule = tuple(tuple(float(v) for v in triple) for triple in ns_coefficients)
    if not schedule or any(len(triple) != 3 for triple in schedule):
        raise ValueError(f"ns_coefficients must be (a, b, c) triples, got {ns_coefficients}")
    return schedule


def _route_to_muon(name, param, modules, adamw_res, muon_res) -> bool:
    if param.ndim < 2:
        return False
    if any(regex.search(name) for regex in adamw_res):
        return False
    if any(regex.search(name) for regex in muon_res):
        return True
    module_name, _, attr = name.rpartition(".")
    if attr != "weight" or param.ndim != 2:
        return False
    if modules is None:
        return True
    return isinstance(modules.get(module_name), nn.Linear)


def _normalize_split_patterns(split_patterns) -> List[Tuple[str, int]]:
    if split_patterns is None:
        split_patterns = DEFAULT_SPLIT_PATTERNS
    if isinstance(split_patterns, dict) or hasattr(split_patterns, "items"):
        return [(str(k), int(v)) for k, v in split_patterns.items()]
    return [(str(entry["pattern"]), int(entry["num_splits"])) for entry in split_patterns]


def _lr_scale(rows: int, cols: int, adjust_lr_fn: str) -> float:
    if adjust_lr_fn == "match_rms_adamw":
        return 0.2 * math.sqrt(max(rows, cols))
    if adjust_lr_fn == "original":
        return math.sqrt(max(1.0, rows / cols))
    return 1.0


def _local_world_size() -> int:
    if "LOCAL_WORLD_SIZE" in os.environ:
        return int(os.environ["LOCAL_WORLD_SIZE"])
    if "SLURM_NTASKS_PER_NODE" in os.environ:
        return int(os.environ["SLURM_NTASKS_PER_NODE"].split("(")[0])
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1
