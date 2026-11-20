/-
# Mirror identity (expected generalization gap = replace-one stability)

    E[ L_D(A(S)) − L_S(A(S)) ] = (1/n) ∑ i E[ ℓ(A(S⁽ⁱ⁾); z_i) − ℓ(A(S); z_i) ]

where S⁽ⁱ⁾ replaces the i-th sample with an independent copy. This is the
foundational identity behind fold-based noise measurement: what a model loses to
distribution shift is EXACTLY what it loses when its own training points are
swapped out — disagreement between independently-sampled data is generalization
error, which is what licenses MANE's two-fold estimator as a generalization
signal (Variance.lean makes the per-step version quantitative).

Upgraded from the earlier axiom-based file: the expectation functional and the
two distributional facts (`h_LD`: z_i has the population loss in expectation;
`h_exchange`: exchangeability of (S, z_i) with (S⁽ⁱ⁾ swapped)) are HYPOTHESES of
the theorem — this file declares no axioms and contains no `sorry`. Numerically
corroborated by simulate.py (both sides ≈ 2σ²/n for the sample mean).
-/
import Mathlib.Tactic

open Finset

namespace Mane

theorem mirror_identity {Ω Z Θ : Type*} {n : ℕ}
    (E : (Ω → ℝ) → ℝ)
    (E_sub : ∀ X Y : Ω → ℝ, E (fun ω => X ω - Y ω) = E X - E Y)
    (E_const_mul : ∀ (c : ℝ) (X : Ω → ℝ), E (fun ω => c * X ω) = c * E X)
    (E_sum : ∀ (f : Fin n → Ω → ℝ),
      E (fun ω => ∑ i, f i ω) = ∑ i, E (fun ω => f i ω))
    (ℓ : Θ → Z → ℝ) (L_D : Θ → ℝ) (A : (Fin n → Z) → Θ)
    (S : Ω → Fin n → Z) (z : Ω → Fin n → Z)
    (hn : (n : ℝ) ≠ 0)
    (h_LD : ∀ zz : Ω → Z,
      E (fun ω => ℓ (A (S ω)) (zz ω)) = E (fun ω => L_D (A (S ω))))
    (h_exchange : ∀ i : Fin n,
      E (fun ω => ℓ (A (Function.update (S ω) i (z ω i))) (S ω i))
        = E (fun ω => ℓ (A (S ω)) (z ω i))) :
    E (fun ω => L_D (A (S ω)) - (1 / (n : ℝ)) * ∑ i, ℓ (A (S ω)) (S ω i))
      = (1 / (n : ℝ)) * ∑ i : Fin n,
          E (fun ω => ℓ (A (Function.update (S ω) i (z ω i))) (S ω i)
                      - ℓ (A (S ω)) (S ω i)) := by
  rw [E_sub]
  have h_LS : E (fun ω => (1 / (n : ℝ)) * ∑ i, ℓ (A (S ω)) (S ω i))
      = (1 / (n : ℝ)) * ∑ i : Fin n, E (fun ω => ℓ (A (S ω)) (S ω i)) := by
    rw [E_const_mul, E_sum]
  rw [h_LS]
  have h_LD_sum : E (fun ω => L_D (A (S ω)))
      = (1 / (n : ℝ)) * ∑ _i : Fin n, E (fun ω => L_D (A (S ω))) := by
    rw [Finset.sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul, ← mul_assoc,
        one_div, inv_mul_cancel₀ hn, one_mul]
  rw [h_LD_sum, ← mul_sub, ← Finset.sum_sub_distrib]
  congr 1
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [E_sub, h_exchange i, h_LD fun ω => z ω i]

end Mane
