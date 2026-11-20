/-
# Optimal gates for sign updates (MANE core algebra)

Two fully-proved results about the MANE gate γ:

1. `wiener_gate_optimal` — Estimating a signal μ by `g · m̂`, where the observation
   m̂ has mean μ and variance σ², has mean-squared error
   `E[(g·m̂ − μ)²] = g²(μ² + σ²) − 2gμ² + μ²`. Over all constant gates g, this is
   minimized exactly at the Wiener/James–Stein shrinkage
   `g* = μ²/(μ² + σ²) = SNR²/(SNR² + 1)` — the MANE gate with scale 1.

2. `clipped_gate_optimal` — In the smooth-descent bound, a gated sign update buys
   descent `a·γ` per coordinate at quadratic noise cost `b·γ²`. Over γ ∈ [0,1] the
   trade `a·γ − b·γ²` is maximized by the clipped gate `min 1 (a/(2b))`: coordinates
   with weak expected descent (small a — low SNR) get proportionally small gates.

Both are pure algebra over ℝ: no axioms, no `sorry`.
-/
import Mathlib.Tactic

namespace Mane

/-- Bias–variance decomposition of the gated estimator's MSE around the
Wiener gate `g* = μ² / (μ² + σ²)`:
`g²(μ²+σ²) − 2gμ² + μ² = (μ²+σ²)(g − g*)² + μ²σ²/(μ²+σ²)`. -/
theorem wiener_decomp (μ σ g : ℝ) (hσ : σ ≠ 0) :
    g ^ 2 * (μ ^ 2 + σ ^ 2) - 2 * g * μ ^ 2 + μ ^ 2
      = (μ ^ 2 + σ ^ 2) * (g - μ ^ 2 / (μ ^ 2 + σ ^ 2)) ^ 2
        + μ ^ 2 * σ ^ 2 / (μ ^ 2 + σ ^ 2) := by
  have hs : μ ^ 2 + σ ^ 2 ≠ 0 := by positivity
  field_simp
  ring

/-- **Wiener-gate optimality.** The MSE of the gated estimator is minimized at
`g* = μ²/(μ²+σ²)`: for every gate `g`,
`MSE(g*) ≤ MSE(g)`. This is the MANE gate `γ = ρ/(ρ+1)` with `ρ = μ²/σ²`. -/
theorem wiener_gate_optimal (μ σ g : ℝ) (hσ : σ ≠ 0) :
    (μ ^ 2 / (μ ^ 2 + σ ^ 2)) ^ 2 * (μ ^ 2 + σ ^ 2)
        - 2 * (μ ^ 2 / (μ ^ 2 + σ ^ 2)) * μ ^ 2 + μ ^ 2
      ≤ g ^ 2 * (μ ^ 2 + σ ^ 2) - 2 * g * μ ^ 2 + μ ^ 2 := by
  have hs : (0:ℝ) < μ ^ 2 + σ ^ 2 := by positivity
  rw [wiener_decomp μ σ g hσ, wiener_decomp μ σ (μ ^ 2 / (μ ^ 2 + σ ^ 2)) hσ]
  have hz : (μ ^ 2 / (μ ^ 2 + σ ^ 2) - μ ^ 2 / (μ ^ 2 + σ ^ 2)) ^ 2 = 0 := by ring
  rw [hz, mul_zero, zero_add]
  nlinarith [sq_nonneg (g - μ ^ 2 / (μ ^ 2 + σ ^ 2))]

/-- The Wiener gate equals `SNR²/(SNR²+1)` where `SNR² = μ²/σ²`. -/
theorem wiener_gate_snr_form (μ σ : ℝ) (hσ : σ ≠ 0) :
    μ ^ 2 / (μ ^ 2 + σ ^ 2) = (μ ^ 2 / σ ^ 2) / (μ ^ 2 / σ ^ 2 + 1) := by
  have h1 : σ ^ 2 ≠ 0 := pow_ne_zero 2 hσ
  have h2 : μ ^ 2 + σ ^ 2 ≠ 0 := by positivity
  field_simp

/-- **Clipped-gate optimality.** Over gates `g ≤ 1`, the descent-vs-noise
objective `a·g − b·g²` (a = expected per-coordinate descent, b = curvature·η/2 > 0)
is maximized at `min 1 (a/(2b))`. Low-SNR coordinates (small `a`) receive
proportionally small optimal gates — pure Lion (`g ≡ 1`) is optimal only when
`a ≥ 2b` on every coordinate. (For a ≥ 0 the maximizer lies in [0,1], i.e. is a
valid gate; the inequality needs neither `0 ≤ a` nor `0 ≤ g`.) -/
theorem clipped_gate_optimal (a b g : ℝ) (hb : 0 < b) (hg1 : g ≤ 1) :
    a * g - b * g ^ 2 ≤ a * min 1 (a / (2 * b)) - b * (min 1 (a / (2 * b))) ^ 2 := by
  rcases le_total (a / (2 * b)) 1 with h | h
  · -- interior optimum g* = a/(2b) ≤ 1
    rw [min_eq_right h]
    have h2b : (0:ℝ) < 2 * b := by positivity
    have key : a * (a / (2 * b)) - b * (a / (2 * b)) ^ 2 - (a * g - b * g ^ 2)
        = b * (g - a / (2 * b)) ^ 2 := by
      field_simp
      ring
    nlinarith [sq_nonneg (g - a / (2 * b)), mul_nonneg hb.le (sq_nonneg (g - a / (2 * b)))]
  · -- boundary optimum g* = 1 (a ≥ 2b)
    rw [min_eq_left h]
    have h2b : (0:ℝ) < 2 * b := by positivity
    have h2 : 2 * b ≤ a := by
      have := (le_div_iff₀ h2b).mp h
      linarith
    -- a·1 − b·1² − (a·g − b·g²) = (1−g)(a−2b) + b(1−g)² ≥ 0
    nlinarith [mul_nonneg (sub_nonneg.mpr hg1) (sub_nonneg.mpr h2),
               mul_nonneg hb.le (sq_nonneg (1 - g))]

/-- **Trust-capped gate optimality (MANE v2).** Over gates `g ≤ Γ`, the
descent-vs-noise objective is maximized at `min Γ (a/(2b))`: the optimal gate
grows LINEARLY with expected descent up to the trust cap. Nothing in the
argument requires Γ = 1 — allocating trust beyond 1 to a coordinate is optimal
precisely when its expected descent exceeds `2b`. `Γ = 1` recovers
`clipped_gate_optimal`; pure Lion (`g ≡ 1`) is the Γ-saturated limit. -/
theorem capped_gate_optimal (a b g Γ : ℝ) (hb : 0 < b) (hg : g ≤ Γ) :
    a * g - b * g ^ 2 ≤ a * min Γ (a / (2 * b)) - b * (min Γ (a / (2 * b))) ^ 2 := by
  rcases le_total (a / (2 * b)) Γ with h | h
  · -- interior optimum g* = a/(2b) ≤ Γ
    rw [min_eq_right h]
    have h2b : (0:ℝ) < 2 * b := by positivity
    have key : a * (a / (2 * b)) - b * (a / (2 * b)) ^ 2 - (a * g - b * g ^ 2)
        = b * (g - a / (2 * b)) ^ 2 := by
      field_simp
      ring
    nlinarith [sq_nonneg (g - a / (2 * b)), mul_nonneg hb.le (sq_nonneg (g - a / (2 * b)))]
  · -- boundary optimum g* = Γ (a ≥ 2bΓ)
    rw [min_eq_left h]
    have h2b : (0:ℝ) < 2 * b := by positivity
    have h2 : 2 * b * Γ ≤ a := by
      have := (le_div_iff₀ h2b).mp h
      linarith
    -- a·Γ − b·Γ² − (a·g − b·g²) = (Γ−g)(a−2bΓ) + b(Γ−g)² ≥ 0
    nlinarith [mul_nonneg (sub_nonneg.mpr hg) (sub_nonneg.mpr h2),
               mul_nonneg hb.le (sq_nonneg (Γ - g))]

end Mane
