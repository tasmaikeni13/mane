/-
# The budget-allocation theorem (MANE v2 core)

Fix a per-step noise budget: the descent lemma (Descent.lean) charges a gated
sign step `(Lη²/2) ∑ γᵢ²` of noise for `η ∑ γᵢ aᵢ` of expected descent
(aᵢ = per-coordinate expected descent). Question: how should trust γ be
allocated across coordinates for a fixed budget `∑ γᵢ² ≤ B`?

`budget_allocation`      —  ∑ γᵢ aᵢ ≤ √(B · ∑ aᵢ²)  for EVERY allocation γ.
`budget_allocation_proportional` — the proportional allocation γᵢ = t·aᵢ
(t = √(B/∑a²)) meets the budget exactly and ACHIEVES the bound.

Consequence: **redistribution is optimal** — trust proportional to expected
descent. A UNIFORM allocation (vanilla Lion, γ ≡ const) is optimal only when
every coordinate carries identical signal. This is the formal license for
gates γ > 1 on reliable coordinates (MANE v2 "budgeted trust"), complementing
`capped_gate_optimal` (per-coordinate view with a trust cap).

Proof: Cauchy–Schwarz on finite sums. No axioms, no `sorry`.
-/
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Analysis.Real.Sqrt

open Finset

namespace Mane

variable {ι : Type*}

/-- **Upper bound.** Any trust allocation obeying the noise budget `∑ γ² ≤ B`
buys at most `√(B · ∑ a²)` of descent. -/
theorem budget_allocation (s : Finset ι) (a γ : ι → ℝ) (B : ℝ)
    (hbudget : ∑ i ∈ s, γ i ^ 2 ≤ B) :
    ∑ i ∈ s, γ i * a i ≤ Real.sqrt (B * ∑ i ∈ s, a i ^ 2) := by
  refine Real.le_sqrt_of_sq_le ?_
  have hA : (0:ℝ) ≤ ∑ i ∈ s, a i ^ 2 := Finset.sum_nonneg fun i _ => sq_nonneg _
  calc (∑ i ∈ s, γ i * a i) ^ 2
      ≤ (∑ i ∈ s, γ i ^ 2) * ∑ i ∈ s, a i ^ 2 :=
        Finset.sum_mul_sq_le_sq_mul_sq s γ a
    _ ≤ B * ∑ i ∈ s, a i ^ 2 := mul_le_mul_of_nonneg_right hbudget hA

/-- **Achievability.** The proportional allocation `γᵢ = √(B/∑a²) · aᵢ`
spends the budget exactly and attains the bound — trust ∝ expected descent
is the optimal policy. -/
theorem budget_allocation_proportional (s : Finset ι) (a : ι → ℝ) (B : ℝ)
    (hB : 0 ≤ B) (hA : 0 < ∑ j ∈ s, a j ^ 2) :
    (∑ i ∈ s, (Real.sqrt (B / ∑ j ∈ s, a j ^ 2) * a i) ^ 2 = B) ∧
    (∑ i ∈ s, (Real.sqrt (B / ∑ j ∈ s, a j ^ 2) * a i) * a i
      = Real.sqrt (B * ∑ j ∈ s, a j ^ 2)) := by
  set A := ∑ j ∈ s, a j ^ 2 with hAdef
  set t := Real.sqrt (B / A) with htdef
  have ht2 : t ^ 2 = B / A := Real.sq_sqrt (by positivity)
  have hAne : A ≠ 0 := ne_of_gt hA
  constructor
  · have hsum : ∑ i ∈ s, (t * a i) ^ 2 = t ^ 2 * A := by
      rw [hAdef, Finset.mul_sum]
      exact Finset.sum_congr rfl fun i _ => by ring
    rw [hsum, ht2]
    field_simp
  · have hsum : ∑ i ∈ s, (t * a i) * a i = t * A := by
      rw [hAdef, Finset.mul_sum]
      exact Finset.sum_congr rfl fun i _ => by ring
    rw [hsum]
    have h1 : (t * A) ^ 2 = B * A := by
      rw [mul_pow, ht2]
      field_simp
    calc t * A = Real.sqrt ((t * A) ^ 2) := (Real.sqrt_sq (by positivity)).symm
      _ = Real.sqrt (B * A) := by rw [h1]

end Mane
