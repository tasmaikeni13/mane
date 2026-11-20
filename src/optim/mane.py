"""MANE: Micro-batch Agreement Noise Equilibration.

MANE is a sign-based momentum optimizer that stabilizes Lion-style updates by
dynamically allocating a per-step noise budget. In standard sign-momentum
methods (like Lion), every parameter coordinate is updated by exactly the learning
rate regardless of signal quality, which injects excessive noise in low-SNR
dimensions and triggers training instability.

MANE solves this by consuming two independent gradient estimates per step (derived
from two disjoint data folds) to directly measure coordinate-wise noise variance.
It then allocates a global noise budget across all parameters, scaling updates
proportionally to their signal-to-noise ratio (z-score) and capping the maximum
trust level.
"""

import math
import torch
from torch.optim.optimizer import Optimizer


class MANE(Optimizer):
    """MANE: Micro-batch Agreement Noise Equilibration optimizer.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr: learning rate (default: 1e-4).
        betas: coefficients used for running averages of direction momentum (default: (0.95, 0.98)).
        beta_v: coefficient used for running average of fold noise variance (default: 0.99).
        weight_decay: weight decay coefficient (default: 0.0).
        budget_ratio: ratio of the total noise budget relative to vanilla Lion (default: 1.0).
        trust_cap: maximum trust multiplier Γ for reliable coordinates (default: 3.0).
        eps: term added to the denominator to improve numerical stability (default: 1e-16).
    """

    def __init__(self, params, lr: float = 1e-4, betas=(0.95, 0.98),
                 beta_v: float = 0.99, weight_decay: float = 0.0,
                 budget_ratio: float = 1.0, trust_cap: float = 3.0,
                 eps: float = 1e-16):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if not 0.0 <= beta_v < 1.0:
            raise ValueError(f"Invalid beta_v: {beta_v}")
        if budget_ratio < 0.0:
            raise ValueError(f"Invalid budget_ratio: {budget_ratio}")
        if trust_cap < 1.0:
            raise ValueError(f"Invalid trust_cap: {trust_cap} (must be >= 1)")
        defaults = dict(lr=lr, betas=betas, beta_v=beta_v, weight_decay=weight_decay,
                        budget_ratio=budget_ratio, trust_cap=trust_cap, eps=eps)
        super().__init__(params, defaults)
        self.last_stats = {}
        self.last_param_gate_means = []

    @staticmethod
    def _omega_c(beta1: float, beta2: float) -> float:
        """Variance-transfer factor from per-step noise to the direction momentum c."""
        return beta1 ** 2 * (1.0 - beta2) / (1.0 + beta2) + (1.0 - beta1) ** 2

    @torch.no_grad()
    def step(self, grads_a=None, closure=None):
        """Perform a single optimization step.

        Args:
            grads_a: list of gradients from fold A. If None, degrades to vanilla Lion.
            closure: A closure that re-evaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if grads_a is None:
            # Fallback to vanilla Lion
            for group in self.param_groups:
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                wd = group["weight_decay"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g_mean = p.grad
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                    m = state["exp_avg"]

                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)

                    c = m.mul(beta1).add_(g_mean, alpha=1.0 - beta1)
                    p.add_(c.sign_(), alpha=-lr)
                    m.mul_(beta2).add_(g_mean, alpha=1.0 - beta2)
            return loss

        # Run budget-allocated MANE step
        entries = []
        zsq_sum, n_coords = None, 0

        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            beta_v = group["beta_v"]
            eps = group["eps"]
            omega_c = self._omega_c(beta1, beta2)
            for p in group["params"]:
                g_a = grads_a[idx]
                idx += 1
                if p.grad is None or g_a is None:
                    continue
                g_b = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["noise_ema"] = torch.zeros_like(p)
                    state["t"] = 0
                m, v = state["exp_avg"], state["noise_ema"]
                state["t"] += 1
                bias_c = 1.0 - beta_v ** state["t"]

                diff = g_a.sub_(g_b)
                g_mean = g_b.add_(diff, alpha=0.5)
                v.mul_(beta_v).addcmul_(diff, diff, value=0.25 * (1.0 - beta_v))
                c = diff.copy_(m).mul_(beta1).add_(g_mean, alpha=1.0 - beta1)
                z = c.abs().div_(v.mul(omega_c / bias_c).add_(eps).sqrt_())

                zs = z.square().sum()
                zsq_sum = zs if zsq_sum is None else zsq_sum + zs
                n_coords += z.numel()
                entries.append((p, group, m, g_mean, c, z))

        if n_coords == 0:
            return loss

        # Compute scaling factor to equilibrate noise budget: lambda_t = budget_ratio / RMS(z)
        z_rms = (zsq_sum / n_coords).sqrt().clamp_(min=1e-12)
        budget_ratio = self.param_groups[0]["budget_ratio"]
        lam = budget_ratio / z_rms

        gate_sum = gate_sq_sum = gate_lt5 = gate_lt1 = gate_gt2 = None
        param_gate_means = []
        for p, group, m, g_mean, c, z in entries:
            lr = group["lr"]
            beta2 = group["betas"][1]
            wd = group["weight_decay"]
            gamma = z.mul(lam).clamp_(max=group["trust_cap"])

            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            p.addcmul_(gamma, c.sign_(), value=-lr)
            m.mul_(beta2).add_(g_mean, alpha=1.0 - beta2)

            gs = gamma.sum()
            gss = gamma.square().sum()
            l5 = (gamma < 0.5).sum()
            l1 = (gamma < 0.1).sum()
            g2 = (gamma > 2.0).sum()
            gate_sum = gs if gate_sum is None else gate_sum + gs
            gate_sq_sum = gss if gate_sq_sum is None else gate_sq_sum + gss
            gate_lt5 = l5 if gate_lt5 is None else gate_lt5 + l5
            gate_lt1 = l1 if gate_lt1 is None else gate_lt1 + l1
            gate_gt2 = g2 if gate_gt2 is None else gate_gt2 + g2
            param_gate_means.append(gs / gamma.numel())

        self.last_stats = {
            "gate_mean": gate_sum / n_coords,
            "gate_rms": (gate_sq_sum / n_coords).sqrt(),
            "gate_frac_lt_05": gate_lt5.float() / n_coords,
            "gate_frac_lt_01": gate_lt1.float() / n_coords,
            "gate_frac_gt_2": gate_gt2.float() / n_coords,
            "gate_lambda": lam,
        }
        self.last_param_gate_means = param_gate_means

        return loss
