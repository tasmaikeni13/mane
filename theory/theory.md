# MANE: Micro-batch Agreement Noise Equilibration

This document presents the mathematical foundation for **MANE** (**M**icro-batch **A**greement **N**oise **E**quilibration), a sign-based momentum optimizer designed to stabilize and enhance the generalization of sign-update methods (such as Lion) under noise-dominated training regimes.

Prose companion to the Lean 4 sources in [`ManeLean/`](ManeLean) (Lean 4.31 + Mathlib v4.31.0; **all files compile — no `sorry`, no `axiom` declarations**).

---

## 1. The Stability Problem in Sign-Momentum Optimizers

Sign-momentum methods like Lion update every parameter coordinate by exactly $\pm \eta$:

$$\theta_t = \theta_{t-1} - \eta (\operatorname{sign}(c_t) + \lambda \theta_{t-1})$$

where $c_t$ is the running momentum of gradients. For a parameter coordinate whose direction estimate $c$ has expected value $\mu$ and noise standard deviation $\sigma$ (gradient noise per batch), the expected movement is:

$$\mathbb{E}[\operatorname{sign}(c)] = \operatorname{erf}\left( \frac{\mu}{\sigma \sqrt{2}} \right)$$

When the signal-to-noise ratio is low ($|\mu| \ll \sigma$), the expected movement is close to zero, but the update power remains exactly $\eta^2$ at every step. This means **low-SNR coordinates buy zero expected descent at full noise cost**, which acts as a primary source of training instability and collapse at higher learning rates or smaller batch sizes.

In contrast, AdamW stabilizes these coordinates by implicitly gating the update magnitude:

$$\frac{m}{\sqrt{\mathbb{E}[g^2]}} = \operatorname{sign}(m) \cdot \frac{|\mu|}{\sqrt{\mu^2 + \sigma^2}} = \operatorname{sign}(m) \cdot \sqrt{\frac{\rho}{\rho + 1}}$$

where $\rho = \mu^2 / \sigma^2$ is the coordinate-wise $\text{SNR}^2$. MANE explicitly measures this SNR and applies optimal budget redistribution.

---

## 2. The MANE Optimization Algorithm

MANE estimates coordinate-wise noise variance in real time using two independent data folds (each consuming half of the step's batch, keeping the total tokens per step identical to the baseline).

For each parameter coordinate at step $t$:

1. **Unbiased Noise Measurement:**
   Using the gradient estimates $g_a, g_b$ from the two independent data folds, we compute the sample mean $\bar{g}$ and the sample variance $n$:
   $$\bar{g} = \frac{g_a + g_b}{2}$$
   $$n = \frac{(g_a - g_b)^2}{4}$$

2. **Noise EMA Tracking:**
   We maintain a running exponential moving average $v$ of the per-coordinate noise variance, bias-corrected at step $t$:
   $$v_t = \beta_v v_{t-1} + (1 - \beta_v) n$$
   $$\hat{v}_t = \frac{v_t}{1 - \beta_v^t}$$

3. **Direction and Reliability Estimation:**
   We update the momentum direction $c$ and estimate the coordinate-wise reliability z-score $z$:
   $$c = \beta_1 m_{t-1} + (1 - \beta_1) \bar{g}$$
   $$z = \frac{|c|}{\sqrt{\omega_c \hat{v}_t + \epsilon}}$$
   where $\omega_c = \beta_1^2 \frac{1 - \beta_2}{1 + \beta_2} + (1 - \beta_1)^2$ is the stationary variance transfer coefficient from per-step gradient noise to the direction momentum $c$.

4. **Equilibrated Noise Budget Allocation:**
   To spend a target noise budget optimally without introducing overall scale drift, we compute the global scaling factor $\lambda_t$:
   $$\lambda_t = \frac{r}{\operatorname{RMS}(z)}$$
   where $r$ is the target budget ratio relative to vanilla Lion (default: $1.0$). The per-coordinate gate $\gamma$ is then computed by scaling $z$ and clamping at the trust cap $\Gamma$:
   $$\gamma = \min\left(\Gamma, \lambda_t z\right)$$

5. **Parameter and Momentum Update:**
   $$\theta_t = (1 - \eta \lambda) \theta_{t-1} - \eta \cdot \gamma \odot \operatorname{sign}(c)$$
   $$m_t = \beta_2 m_{t-1} + (1 - \beta_2) \bar{g}$$

When all coordinates have uniform reliability, $\gamma \equiv r$, recovering vanilla Lion exactly. In heterogeneous noise regimes, MANE automatically dampens unreliable updates and redistributes step size to high-signal coordinates.

---

## 3. Machine-Checked Theorems (Lean 4)

The mathematical properties of the MANE estimator and gate are verified in the accompanying Lean files:

*   **T1 — Two-Fold Unbiasedness (`Variance.lean`):**
    For independent folds with equal means, the sample variance of the difference is an unbiased estimator of the variance of the sample mean:
    $$\mathbb{E}\left[ \left(\frac{g_a - g_b}{2}\right)^2 \right] = \operatorname{Var}\left( \frac{g_a + g_b}{2} \right)$$
    This licenses tracking fold disagreement as the core coordinate-wise noise metric.

*   **T2 — Wiener Gate Optimality (`OptimalGate.lean`):**
    Under coordinate estimation $\gamma \cdot \hat{m}$ where $\hat{m} \sim \mathcal{N}(\mu, \sigma^2)$, the mean-squared error $\mathbb{E}[(\gamma \hat{m} - \mu)^2]$ is uniquely minimized at the shrinkage factor $\gamma^* = \rho / (\rho + 1)$.

*   **T3 — Gated Descent Lemma (`Descent.lean`):**
    Under $L$-smoothness, a gated sign update decreases $f$ subject to a quadratic noise penalty:
    $$f(\theta - \eta \gamma \odot s) \leq f(\theta) - \eta \sum_i \gamma_i (s_i \partial_i f) + \frac{L \eta^2}{2} \sum_i \gamma_i^2$$
    Vanilla Lion is the ungated limit ($\gamma \equiv 1$), paying the full noise penalty $n \frac{L \eta^2}{2}$ regardless of signal quality.

*   **T4 & T4' — Trust-Capped Optimality (`OptimalGate.lean`):**
    For descent $a \cdot \gamma$ vs noise $b \cdot \gamma^2$ under $\gamma \leq \Gamma$, the objective is maximized by $\gamma^* = \min(\Gamma, a/2b)$. Trust allocation beyond 1 is optimal when expected descent exceeds the noise curvature.

*   **T7 — Budget Allocation (`Allocation.lean`):**
    Under a global noise budget $\sum \gamma_i^2 \leq B$, expected descent $\sum \gamma_i a_i$ is maximized by the proportional allocation $\gamma_i \propto a_i$ (Cauchy-Schwarz). This establishes that **redistribution of step size is optimal** under a fixed noise budget.

*   **T5 & T6 — Mirror Identity and Gap-Growth (`Mirror.lean`, `GapGrowth.lean`):**
    Relates generalization gap evolution under empirical gradient flow to the gradient noise energy $\langle \nabla L_S - \nabla L_D, \nabla L_S \rangle$. It shows that training-validation drift is driven directly by the noise component, which the MANE gate measures and suppresses.
