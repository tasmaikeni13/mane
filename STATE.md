# STATE — MANE: Micro-batch Agreement Noise Equilibration

## Problem Definition
Lion (sign-momentum, Chen et al. 2023) updates every coordinate by exactly $\pm \eta$, which causes it to inject high noise power in low-SNR dimensions. This noise fragility leads to training instability and collapse at higher learning rates. 

**Goal:** Stabilize sign-momentum optimization by applying a dynamic noise budget allocation (MANE) to beat tuned AdamW on WikiText-103 (39M parameter decoder-only Transformer, 50k steps, batch 32 × seq 256).

---

## Active Constraints & Protocol
- **Turing target hardware:** Quadro RTX 6000 (FP32 only, no BF16).
- **Fairness contract:** Identical model initialization per seed; equal token budget per step (split folds of 2×16 vs baseline batch of 32); individual learning rate tuning for each optimizer; conserved $lr \cdot wd$; fixed validation windows.
- **Formal Verification:** All theoretical lemmas verified using Lean 4.31 + Mathlib with zero warnings or `sorry` statements.

---

## Proven Results

### 1. Mathematical Proofs (Lean 4)
- **`Variance.lean` (Unbiasedness):** Proves that sample variance of difference of independent folds is unbiased for the variance of the sample mean: $\mathbb{E}[((g_a - g_b)/2)^2] = \operatorname{Var}((g_a + g_b)/2)$.
- **`OptimalGate.lean` (Optimal Gating):** Proves that Wiener shrinkage minimizes coordinate-wise MSE, and capped gate $\min(\Gamma, a/2b)$ is optimal for local curvature.
- **`Allocation.lean` (Budget Allocation):** Proves that proportional allocation ($\gamma_i \propto z_i$) under a fixed noise budget maximizes expected descent.
- **`Descent.lean` (Descent Lemma):** Proves target reduction under coordinate-wise gated sign steps.
- **`Mirror.lean` & `GapGrowth.lean` (Generalization dynamics):** Formulates the mirror identity and growth of generalization gap along empirical gradient flows.

### 2. Empirical Verification (WikiText-103, 39M, 50k steps, seed 0)
MANE successfully beats both vanilla Lion and tuned AdamW:

- **AdamW (Baseline):**
  - Learning Rate: $1.0 \times 10^{-3}$
  - Weight Decay: $0.06$
  - Val Full Loss: $3.6160$ (Perplexity: $37.19$)
  - Test Full Loss: $3.6234$ (Perplexity: $37.46$)

- **Lion (Baseline):**
  - Learning Rate: $5.0 \times 10^{-5}$ (Larger lrs collapse immediately)
  - Weight Decay: $1.20$
  - Val Full Loss: $3.5724$ (Perplexity: $35.60$)
  - Test Full Loss: $3.5749$ (Perplexity: $35.69$)

- **MANE (Ours):**
  - Learning Rate: $1.4 \times 10^{-4}$ (Stable at $2.8\times$ vanilla Lion's maximum lr)
  - Weight Decay: $0.43$
  - Budget Ratio ($r$): $1.0$
  - Trust Cap ($\Gamma$): $3.0$
  - Val Full Loss: **3.4578** (Perplexity: **31.75**)
  - Test Full Loss: **3.4617** (Perplexity: **31.87**)

---

## Next Steps & Scaling Agenda
1. **Replicate across seeds:** Execute runs for seeds 1 and 2 to confirm statistical significance of the perplexity improvements.
2. **Scale to 1B+ parameter models:** Formulate fold parallelism across Data Parallel (DP) ranks to run MANE steps without sequential backward overhead in a distributed multi-GPU setting.
