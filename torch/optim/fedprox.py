# mypy: allow-untyped-defs
from typing import cast

import torch
from torch import Tensor

from .optimizer import (
    _default_to_fused_or_foreach,
    _differentiable_doc,
    _foreach_doc,
    _maximize_doc,
    _params_doc,
    _use_grad_for_differentiable,
    Optimizer,
    ParamsT,
)

__all__ = ["FedProx", "fedprox"]


class FedProx(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        mu: float = 0.01,
        weight_decay: float = 0,
        *,
        maximize: bool = False,
        foreach: bool | None = None,
        differentiable: bool = False,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= mu:
            raise ValueError(f"Invalid proximal term strength (mu): {mu}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr, mu=mu, weight_decay=weight_decay,
            maximize=maximize, foreach=foreach, differentiable=differentiable,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("differentiable", False)

    def set_global_params(self, global_params) -> None:
        r"""Set the global model parameters for the proximal term."""
        global_params_list = list(global_params)
        param_idx = 0

        for group in self.param_groups:
            for p in group["params"]:
                if param_idx >= len(global_params_list):
                    raise ValueError(
                        f"Not enough global parameters. Expected >= {param_idx + 1}, "
                        f"got {len(global_params_list)}."
                    )
                global_p = global_params_list[param_idx]
                if p.shape != global_p.shape:
                    raise ValueError(
                        f"Shape mismatch at parameter {param_idx}: "
                        f"{p.shape} vs {global_p.shape}."
                    )
                self.state[p]["global_param"] = global_p.detach().clone().to(p.device)
                param_idx += 1

        if param_idx != len(global_params_list):
            raise ValueError(
                f"Too many global parameters. Expected {param_idx}, "
                f"got {len(global_params_list)}."
            )

    def _init_group(self, group, params, grads, global_params_list):
        has_sparse_grad = False
        for p in group["params"]:
            if p.grad is None:
                continue
            params.append(p)
            grads.append(p.grad)
            has_sparse_grad |= p.grad.is_sparse
            if "global_param" not in self.state[p]:
                raise RuntimeError(
                    "Global parameters not set. Call set_global_params() before step()."
                )
            global_params_list.append(self.state[p]["global_param"])
        return has_sparse_grad

    @_use_grad_for_differentiable
    def step(self, closure=None):
        self._cuda_graph_capture_health_check()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params, grads, global_params_list = [], [], []
            has_sparse_grad = self._init_group(group, params, grads, global_params_list)
            fedprox(
                params, grads, global_params_list,
                weight_decay=group["weight_decay"],
                mu=group["mu"],
                lr=group["lr"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                has_sparse_grad=has_sparse_grad,
            )
        return loss


FedProx.__doc__ = (
    r"""Implements FedProx optimizer for federated learning.

    FedProx adds a proximal term to SGD to prevent client drift in heterogeneous
    federated settings.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \theta_0 \text{ (params)},
                w^t \text{ (global params)}, \mu \text{ (prox strength)},                        \\
            &\hspace{13mm} \lambda \text{ (weight decay)}, \textit{maximize}                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: k=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_k \leftarrow \nabla_{\theta} f_k(\theta_{k-1})                        \\
            &\hspace{5mm}\textbf{if} \: \lambda \neq 0                                           \\
            &\hspace{10mm} g_k \leftarrow g_k + \lambda \theta_{k-1}                             \\
            &\hspace{5mm}\textbf{if} \: \mu \neq 0                                               \\
            &\hspace{10mm} g_k \leftarrow g_k + \mu (\theta_{k-1} - w^t)                         \\
            &\hspace{5mm}\theta_k \leftarrow \theta_{k-1} - \gamma g_k                           \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_k                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}
    """
    + rf"""
    Args:
        {_params_doc}
        lr (float, optional): learning rate (default: 1e-3)
        mu (float, optional): proximal strength controlling deviation penalty
            from global model. Range [0.001, 1.0] typical. (default: 0.01)
        weight_decay (float, optional): weight decay / L2 penalty (default: 0)
        {_maximize_doc}
        {_foreach_doc}
        {_differentiable_doc}

    Example:
        >>> # xdoctest: +SKIP
        >>> optimizer = torch.optim.FedProx(model.parameters(), lr=0.1, mu=0.01)
        >>> optimizer.set_global_params(global_model.parameters())
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    .. note::
        Call :meth:`set_global_params` before :meth:`step` at each federated round.
        When :math:`\mu = 0`, FedProx reduces to standard SGD.
    """
)


def fedprox(
    params: list[Tensor],
    grads: list[Tensor],
    global_params: list[Tensor],
    *,
    weight_decay: float,
    mu: float,
    lr: float,
    maximize: bool,
    foreach: bool | None = None,
    has_sparse_grad: bool = False,
) -> None:
    r"""Functional API for FedProx."""
    if not params:
        return

    if foreach is None and not torch.jit.is_scripting():
        _, foreach = _default_to_fused_or_foreach(params, False, False)

    if foreach and not torch.jit.is_scripting():
        _multi_tensor_fedprox(params, grads, global_params,
                              weight_decay=weight_decay, mu=mu, lr=lr,
                              maximize=maximize, has_sparse_grad=has_sparse_grad)
    else:
        _single_tensor_fedprox(params, grads, global_params,
                               weight_decay=weight_decay, mu=mu, lr=lr,
                               maximize=maximize, has_sparse_grad=has_sparse_grad)


def _single_tensor_fedprox(
    params: list[Tensor],
    grads: list[Tensor],
    global_params: list[Tensor],
    *,
    weight_decay: float,
    mu: float,
    lr: float,
    maximize: bool,
    has_sparse_grad: bool,
) -> None:
    for i, param in enumerate(params):
        grad = -grads[i] if maximize else grads[i]
        global_p = global_params[i]

        if grad.is_sparse:
            param.add_(grad, alpha=-lr)
            if weight_decay != 0:
                param.add_(param, alpha=-lr * weight_decay)
            if mu != 0:
                param.add_(param - global_p, alpha=-lr * mu)
        else:
            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)
            if mu != 0:
                grad = grad.add(param - global_p, alpha=mu)
            param.add_(grad, alpha=-lr)


def _multi_tensor_fedprox(
    params: list[Tensor],
    grads: list[Tensor],
    global_params: list[Tensor],
    *,
    weight_decay: float,
    mu: float,
    lr: float,
    maximize: bool,
    has_sparse_grad: bool,
) -> None:
    if not params:
        return

    grouped = Optimizer._group_tensors_by_device_and_dtype([params, grads, global_params])
    for (dev_params_, dev_grads_, dev_globals_), _ in grouped.values():
        dev_params = cast(list[Tensor], dev_params_)
        dev_grads = cast(list[Tensor], dev_grads_)
        dev_globals = cast(list[Tensor], dev_globals_)

        if maximize:
            dev_grads = torch._foreach_neg(dev_grads)

        if weight_decay != 0:
            dev_grads = torch._foreach_add(dev_grads, dev_params, alpha=weight_decay)

        if mu != 0:
            prox = torch._foreach_sub(dev_params, dev_globals)
            dev_grads = torch._foreach_add(dev_grads, prox, alpha=mu)

        if has_sparse_grad and any(g.is_sparse for g in dev_grads):
            for j in range(len(dev_params)):
                dev_params[j].add_(dev_grads[j], alpha=-lr)
        else:
            torch._foreach_add_(dev_params, dev_grads, alpha=-lr)
