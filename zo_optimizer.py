"""
zo_optimizer.py — SPSA + Adam zero-order optimizer.

Algorithm
---------
Gradient estimator: SPSA (Simultaneous Perturbation Stochastic Approximation)
  - Perturb ALL active parameters simultaneously with a single Bernoulli ±1 vector δ
  - grad_i ≈ (f(x + ε·δ) - f(x - ε·δ)) / (2ε) · δ_i
  - Cost: exactly 2 forward passes regardless of parameter count
  - K independent estimates are averaged per step to reduce variance

Update rule: Adam (adaptive first/second moment estimates)

Layer schedule (curriculum):
  - Steps 1 … PHASE2_START-1 : only fc.weight, fc.bias
  - Steps PHASE2_START … end  : fc + last ResNet block (layer4.1)
  SPSA cost is independent of parameter count, so expanding layers is free.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class ZeroOrderOptimizer:
    """SPSA + Adam gradient-free optimizer for ResNet18 fine-tuning.

    Args:
        model:        The nn.Module to optimize.
        lr:           Adam learning rate.
        eps:          SPSA perturbation magnitude.
        k_estimates:  Number of independent SPSA estimates averaged per step.
        beta1:        Adam first-moment decay.
        beta2:        Adam second-moment decay.
        eps_adam:     Adam numerical stability constant.
    """

    # Layers unlocked in phase 2 (entire last residual block)
    _PHASE2_LAYERS: list[str] = [
        "layer4.1.conv2.weight",
        "layer4.1.bn2.weight",
        "layer4.1.bn2.bias",
        "layer4.1.conv1.weight",
        "layer4.1.bn1.weight",
        "layer4.1.bn1.bias",
    ]
    _PHASE2_START: int = 100  # switch after this many steps

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-2,
        eps: float = 1e-3,
        k_estimates: int = 5,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps_adam: float = 1e-8,
    ) -> None:
        self.model = model
        self.lr = lr
        self.eps = eps
        self.k_estimates = k_estimates
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps_adam = eps_adam

        # Adam state — initialised lazily on first update
        self._m: dict[str, torch.Tensor] = {}
        self._v: dict[str, torch.Tensor] = {}
        self._t: int = 0       # global Adam step counter
        self._step: int = 0    # optimizer step counter (for curriculum)

        # Phase 1: tune only the classification head
        self.layer_names: list[str] = ["fc.weight", "fc.bias"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _active_params(self) -> dict[str, nn.Parameter]:
        named = dict(self.model.named_parameters())
        missing = [n for n in self.layer_names if n not in named]
        if missing:
            raise KeyError(
                f"Layer names not found in model: {missing}. "
                f"Valid names: {list(named.keys())}"
            )
        return {n: named[n] for n in self.layer_names}

    def _estimate_grad(
        self,
        loss_fn: Callable[[], float],
        params: dict[str, nn.Parameter],
    ) -> dict[str, torch.Tensor]:
        """Average k_estimates SPSA gradient estimates (2 forward passes each).

        SPSA uses ONE simultaneous Bernoulli ±1 perturbation across all params:
            δ_i ~ Bernoulli({-1, +1})
            ĝ_i = (f(x + ε·δ) - f(x - ε·δ)) / (2ε) · δ_i

        Since δ_i ∈ {-1, +1}, δ_i⁻¹ = δ_i, so the formula simplifies to:
            ĝ_i = coeff · δ_i,   coeff = (f_plus - f_minus) / (2ε)

        E[ĝ_i] = ∂f/∂x_i + O(ε²) — unbiased in the ε→0 limit.
        """
        acc: dict[str, torch.Tensor] = {
            name: torch.zeros_like(param) for name, param in params.items()
        }

        for _ in range(self.k_estimates):
            # Sample simultaneous Bernoulli ±1 perturbation for every param
            deltas = {
                name: torch.randint(
                    0, 2, param.shape, device=param.device, dtype=param.dtype
                ) * 2.0 - 1.0
                for name, param in params.items()
            }

            with torch.no_grad():
                # f(x + ε·δ)
                for name, param in params.items():
                    param.data.add_(self.eps * deltas[name])
                f_plus = loss_fn()

                # f(x - ε·δ)
                for name, param in params.items():
                    param.data.sub_(2.0 * self.eps * deltas[name])
                f_minus = loss_fn()

                # Restore x
                for name, param in params.items():
                    param.data.add_(self.eps * deltas[name])

            coeff = (f_plus - f_minus) / (2.0 * self.eps)
            for name, delta in deltas.items():
                acc[name].add_(delta, alpha=coeff)

        # Average over k_estimates
        for name in acc:
            acc[name].div_(self.k_estimates)

        return acc

    def _update_params(
        self,
        params: dict[str, nn.Parameter],
        grads: dict[str, torch.Tensor],
    ) -> None:
        """Adam update: p ← p - lr · m̂ / (√v̂ + ε_adam)."""
        self._t += 1
        with torch.no_grad():
            for name, param in params.items():
                g = grads[name]

                # Lazily initialise Adam state for new layers
                if name not in self._m:
                    self._m[name] = torch.zeros_like(param)
                    self._v[name] = torch.zeros_like(param)

                # Moment updates (in-place)
                self._m[name].mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
                self._v[name].mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)

                # Bias-corrected moments
                m_hat = self._m[name] / (1.0 - self.beta1 ** self._t)
                v_hat = self._v[name] / (1.0 - self.beta2 ** self._t)

                param.data.addcdiv_(
                    m_hat, v_hat.sqrt().add_(self.eps_adam), value=-self.lr
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, loss_fn: Callable[[], float]) -> float:
        """One SPSA + Adam optimisation step.

        Args:
            loss_fn: Callable returning a scalar float loss on the current
                     mini-batch. Called 2·k_estimates + 1 times per step
                     (all on the same fixed batch).

        Returns:
            Loss value before the update.
        """
        self._step += 1

        # Curriculum: unlock layer4.1 after PHASE2_START steps
        if self._step == self._PHASE2_START:
            self.layer_names = ["fc.weight", "fc.bias"] + self._PHASE2_LAYERS

        params = self._active_params()

        with torch.no_grad():
            loss_before = loss_fn()

        grads = self._estimate_grad(loss_fn, params)
        self._update_params(params, grads)

        return float(loss_before)
