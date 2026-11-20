"""Lion optimizer (EvoLved Sign Momentum), Chen et al. 2023 (arXiv:2302.06675).

Repo-local implementation so the Lion experiments have no external dependency.
Update rule, per parameter:

    c_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    theta_t = theta_{t-1} - lr * (sign(c_t) + wd * theta_{t-1})   # decoupled wd
    m_t = beta2 * m_{t-1} + (1 - beta2) * g_t

Because the update is the sign of an interpolated momentum, Lion typically wants a
learning rate a few times smaller and a weight decay a few times larger than
AdamW.
"""

import torch
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    def __init__(self, params, lr: float = 1e-4, betas=(0.9, 0.99),
                 weight_decay: float = 0.0):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]

                # Decoupled weight decay.
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # Update = sign(beta1 * m + (1 - beta1) * g); does not mutate m.
                update = exp_avg.clone().mul_(beta1).add_(grad, alpha=1.0 - beta1).sign_()
                p.add_(update, alpha=-lr)

                # Momentum update with beta2.
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        return loss
