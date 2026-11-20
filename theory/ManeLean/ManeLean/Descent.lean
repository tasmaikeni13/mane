/-
# Descent lemma for gated sign updates

The step analysed here is Lion-MANE's:  θ' = θ − η (γ ⊙ s),  with per-coordinate
gates γ_i ∈ [0,1] and a sign-like direction |s_i| ≤ 1 (s = sign(c) gives
s_i ∈ {−1,0,1}).

`gated_sign_descent`: if `f` satisfies the quadratic upper bound of L-smoothness
(taken as a hypothesis — it is the standard consequence of a Lipschitz gradient),
then

    f(θ') ≤ f(θ) − η ∑ i γ_i (s_i · ∂_i f) + (L η²/2) ∑ i γ_i².

Reading: coordinate i buys descent η·γ_i·(s_i ∂_i f) — which in expectation is
η·γ_i·|∂_i f|·(2·P(sign correct) − 1) — at a noise cost (Lη²/2)·γ_i², INDEPENDENT
of the gradient. Vanilla Lion (γ ≡ 1, `lion_descent`) pays the full noise price
n·Lη²/2 even on coordinates whose expected descent is zero. Optimizing the
per-coordinate trade-off is exactly `Mane.clipped_gate_optimal`
(OptimalGate.lean): γ_i* = min(1, aᵢ/(2bᵢ)) — small expected descent ⇒ small gate.

No axioms, no `sorry`.
-/
import Mathlib.Tactic

namespace Mane

open Finset

variable {n : ℕ}

/-- One gated sign step on the coordinates: `θ'_i = θ_i − η γ_i s_i`. -/
def gatedStep (θ γ s : Fin n → ℝ) (η : ℝ) : Fin n → ℝ :=
  fun i => θ i - η * (γ i * s i)

/-- **Gated sign descent.** Under the quadratic upper bound of `L`-smoothness,
a gated sign step decreases `f` up to the gated noise term `(Lη²/2) ∑ γ_i²`. -/
theorem gated_sign_descent
    (f : (Fin n → ℝ) → ℝ) (θ grad γ s : Fin n → ℝ) (η L : ℝ)
    (hL : 0 ≤ L)
    (hsmooth : ∀ d : Fin n → ℝ,
      f (fun i => θ i + d i) ≤ f θ + (∑ i, grad i * d i) + L / 2 * ∑ i, d i ^ 2)
    (hs : ∀ i, |s i| ≤ 1) :
    f (gatedStep θ γ s η)
      ≤ f θ - η * (∑ i, γ i * (s i * grad i)) + L * η ^ 2 / 2 * ∑ i, γ i ^ 2 := by
  have h := hsmooth fun i => -(η * (γ i * s i))
  have hstep : gatedStep θ γ s η = fun i => θ i + -(η * (γ i * s i)) := by
    funext i; simp [gatedStep, sub_eq_add_neg]
  rw [hstep]
  refine h.trans ?_
  have e1 : (∑ i, grad i * -(η * (γ i * s i)))
      + η * (∑ i, γ i * (s i * grad i)) = 0 := by
    rw [Finset.mul_sum, ← Finset.sum_add_distrib]
    exact Finset.sum_eq_zero fun i _ => by ring
  have e2 : (∑ i, (-(η * (γ i * s i))) ^ 2) ≤ η ^ 2 * ∑ i, γ i ^ 2 := by
    rw [Finset.mul_sum]
    refine Finset.sum_le_sum fun i _ => ?_
    have hs2 : s i ^ 2 ≤ 1 := (sq_le_one_iff_abs_le_one (s i)).mpr (hs i)
    nlinarith [mul_nonneg (sq_nonneg η) (sq_nonneg (γ i)), hs2]
  have e3 : L / 2 * (∑ i, (-(η * (γ i * s i))) ^ 2)
      ≤ L * η ^ 2 / 2 * ∑ i, γ i ^ 2 := by
    calc L / 2 * (∑ i, (-(η * (γ i * s i))) ^ 2)
        ≤ L / 2 * (η ^ 2 * ∑ i, γ i ^ 2) := by
          exact mul_le_mul_of_nonneg_left e2 (by positivity)
      _ = L * η ^ 2 / 2 * ∑ i, γ i ^ 2 := by ring
  linarith [e3, e1]

/-- **Vanilla Lion as the ungated special case** (γ ≡ 1): the noise term is the
full `n·Lη²/2`, paid on every coordinate regardless of signal quality. -/
theorem lion_descent
    (f : (Fin n → ℝ) → ℝ) (θ grad s : Fin n → ℝ) (η L : ℝ)
    (hL : 0 ≤ L)
    (hsmooth : ∀ d : Fin n → ℝ,
      f (fun i => θ i + d i) ≤ f θ + (∑ i, grad i * d i) + L / 2 * ∑ i, d i ^ 2)
    (hs : ∀ i, |s i| ≤ 1) :
    f (gatedStep θ (fun _ => 1) s η)
      ≤ f θ - η * (∑ i, s i * grad i) + L * η ^ 2 / 2 * n := by
  have h := gated_sign_descent f θ grad (fun _ => 1) s η L hL hsmooth hs
  simpa using h

end Mane
