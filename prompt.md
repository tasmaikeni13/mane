### Objective & Strict Execution Order
Inspect this repository, deepen the theoretical and formal foundations of the **`Mane`** optimizer, and refactor the codebase to run high-throughput LLM pretraining and optimizer benchmarks on a **Google Cloud TPU v4-32 pod slice** (4 host workers, 4 TPU v4 chips per worker; 16 chips total).

**CRITICAL — Execute strictly in the sequential phases below and obey the final authorization gate:**
1. **Phase 1:** Mathematical, Statistical & Monte Carlo Analysis of `Mane` (in `analysis/` using Python)
2. **Phase 2:** Formal Mathematical Verification of `Mane` (in `analysis/` using Lean 4)
3. **Phase 3:** Rigorous Theoretical Paper for `Mane`
4. **Phase 4:** Repository Refactoring for the TPU v4-32 Pod
5. **Phase 5:** End-to-End Smoke Testing **ONLY — STOP after smoke tests pass. Do NOT launch any hyperparameter sweeps or main training runs until I explicitly authorize you to proceed.**

Aside from creating the `analysis/` directory specified below, keep the rest of the repository architecture clean, modular, and idiomatic to the existing framework—do not enforce rigid file names or directory layouts; organize refactored components however best fits the codebase.

---

### Phase 1: Quantitative, Statistical & Monte Carlo Analysis (`analysis/` — Python)
The current theoretical treatment of **`Mane`** in this repository is superficial. Before touching the training pipeline, create a directory named `analysis/` and build a rigorous Python numerical, statistical, and Monte Carlo simulation suite to analyze `Mane` prior to real LLM pretraining:
* **Synthetic & Stress-Test Landscapes:** Simulate `Mane`'s update dynamics across convex, non-convex, ill-conditioned (high condition number), saddle-point, and anisotropic loss landscapes under varying batch sizes, heavy-tailed/noisy gradients, and curvature shifts.
* **Statistical & Monte Carlo Characterization:** Run Monte Carlo trials to measure trajectory variance, step-size adaptation, spectral/directional alignment, stability boundaries, and hyperparameter sensitivity compared to standard baselines (`AdamW`, `Lion`, `Muon`) so we establish principled priors and high-confidence bets before spending TPU compute.
* **Iterative Theoretical Refinement:** If numerical simulations or statistical stress-tests expose instabilities, poor conditioning behavior, or suboptimal scaling laws in `Mane`, diagnose the weakness, strengthen `Mane`'s mathematical formulation or update rule, and re-run the Python validation suite until its behavior is demonstrably sound.

---

### Phase 2: Formal Verification (`analysis/` — Lean 4)
Once `Mane`'s mathematical formulation passes Phase 1 simulations, formalize and verify its core theoretical guarantees in **Lean 4** (leveraging `Mathlib` where appropriate) inside `analysis/`:
* **Formal Definitions & Invariants:** Encode `Mane`'s state transitions, update operators, normalization/preconditioning bounds, and step-size constraints in Lean 4.
* **Machine-Checked Proofs:** Prove `Mane`'s foundational mathematical properties—such as descent lemmas, bounded update norms, Lyapunov/energy decay, or convergence rate bounds under explicit smoothness and variance assumptions—ensuring all proofs type-check cleanly with zero `sorry` placeholders.
* **Closed-Loop Feedback:** If Lean formalization reveals hidden assumptions, edge cases, or unprovable claims, loop back to refine the mathematics and re-verify in both Python and Lean until the theory and formal proofs are completely airtight.

---

### Phase 3: Theoretical Paper on `Mane`
Synthesize the validated theory, formal proofs, and simulation insights into a rigorous, publication-grade technical theory document (focused entirely on mathematical depth and substance rather than filler):
* **Complete Mathematical Exposition:** Present formal problem setup, explicit assumptions, exact definitions, step-by-step derivations, lemmas, theorems, and complete proofs corresponding to the Lean 4 formalization.
* **Algorithmic Specification:** Provide clear, unambiguous pseudocode for `Mane` (including any refinements discovered during Phases 1–2), complexity/memory analysis, and theoretical comparisons against `AdamW`, `Lion`, and `Muon`.
* **Simulation & Prior Synthesis:** Integrate key quantitative findings from the Python Monte Carlo analysis to explain `Mane`'s expected pretraining dynamics and justify the hyperparameter search regions for real LLM runs.

---

### Phase 4: TPU v4-32 Pod Refactoring & Benchmark Suite
After completing Phases 1–3, inspect the rest of the repository, remove obsolete low-compute code and workarounds, and refactor the codebase for the **TPU v4-32 pod** (4 workers × 4 TPU v4 chips = 16 chips):

#### 1. Model, Data & Optimizer Specifications
* **Model Architecture:** Modern LLaMA-style Transformer (RoPE, RMSNorm, SwiGLU) configurable across two parameter scales:
  * **Small Scale:** ~15M parameters
  * **Medium Scale:** ~125M parameters
* **Tokenizer:** 32k-vocabulary modern LLaMA-style tokenizer.
* **Dataset:** **FineWeb-Edu** with deterministic tokenization, sharding/streaming, and a dedicated held-out validation split, supporting small-scale splits (150M tokens) and multi-billion-token runs (up to 3B+ tokens).
* **Optimizers to Benchmark (5 total):**
  1. **Mane** (incorporating all theoretical/algorithmic refinements validated in Phases 1–3)
  2. **AdamW**
  3. **Vanilla Lion**
  4. **C-Lion**
  5. **Vanilla Muon**

#### 2. Hardware & Parallelization Topologies
Support two distinct execution modes on the TPU v4-32 pod:
* **Worker-Isolated Mode (for 15M runs):** Each host worker (using its 4 local TPU v4 chips) executes an independent training job, enabling up to 4 concurrent runs across the pod.
* **Full-Pod Multi-Host Mode (for 125M runs):** All 4 workers and all 16 TPU v4 chips operate together simultaneously as a single coordinated multi-host job (SPMD data/model parallelism across the entire pod slice).

#### 3. Configured Experimental Tracks (Ready to Launch Upon Authorization)
Prepare clean configurations, orchestration scripts, and metric logging (training/validation loss, perplexity, tokens/sec, MFU, and stability diagnostics) for both tracks:
* **Track A — 15M Parameters (1 Worker / 4 TPUs per run):**
  * *Sweep:* 4-arm hyperparameter sweep (2 seeds per arm) on the **150M-token** FineWeb-Edu split (1 arm/seed per worker; 4 concurrent runs across the pod).
  * *Main Runs:* Multi-seed training on **125M tokens** of FineWeb-Edu per optimizer using the winning sweep hyperparameters (1 worker / 4 TPUs per seed).
* **Track B — 125M Parameters (All 4 Workers / 16 TPUs simultaneously):**
  * *Sweep:* 4-arm hyperparameter sweep (1 seed per arm) on **1 billion tokens** of FineWeb-Edu per arm per optimizer across all 16 TPUs simultaneously.
  * *Main Runs:* Single-seed training on **3 billion tokens** of FineWeb-Edu per optimizer across all 16 TPUs simultaneously using the winning sweep hyperparameters (parameterized cleanly so additional seeds can be run later).

---

### Phase 5: Smoke Testing & Authorization Gate
* Run short, lightweight **smoke tests** verifying that:
  1. Both **Worker-Isolated Mode (4 TPUs)** and **Full-Pod Multi-Host Mode (16 TPUs)** initialize, compile, step, checkpoint, and log cleanly without deadlocks or NaNs across all 5 optimizers (`Mane`, `AdamW`, `Vanilla Lion`, `C-Lion`, `Vanilla Muon`) and both model sizes (15M and 125M).
  2. The FineWeb-Edu data pipeline and 32k tokenizer stream deterministically across hosts.
* **STOP IMMEDIATELY AFTER SMOKE TESTS COMPLETE.** Present a concise report summarizing: (a) key findings and refinements from the Python & Lean 4 `analysis/`, (b) the completed `Mane` theory paper, and (c) smoke-test verification results. **Do NOT start any hyperparameter sweeps or main training runs until I review your report and give explicit authorization.**
