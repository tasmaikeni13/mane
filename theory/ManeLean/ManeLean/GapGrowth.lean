/-
# Gap-growth law under gradient flow

Under gradient flow on the empirical loss, θ'(t) = −∇L_S(θ), the
population/empirical gap evolves as

    d/dt [ L_D(θ(t)) − L_S(θ(t)) ] = ⟨ ∇L_S(θ) − ∇L_D(θ), ∇L_S(θ) ⟩.

The gap grows exactly when the sampled descent direction disagrees with the
population one — i.e. it is driven by gradient NOISE, the quantity MANE measures
per-coordinate via fold disagreement and suppresses via the gate. Proved from
the chain rule and inner-product algebra; the chain-rule facts are hypotheses.
No axioms, no `sorry`.
-/
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.InnerProductSpace.Calculus

namespace Mane

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem gap_growth_law
    (L_D L_S : V → ℝ) (θ : ℝ → V) (grad_L_D grad_L_S : V → V) (t : ℝ)
    (h_diff_D : Differentiable ℝ (fun t => L_D (θ t)))
    (h_diff_S : Differentiable ℝ (fun t => L_S (θ t)))
    (h_grad_D : ∀ t, deriv (fun t => L_D (θ t)) t = inner ℝ (grad_L_D (θ t)) (deriv θ t))
    (h_grad_S : ∀ t, deriv (fun t => L_S (θ t)) t = inner ℝ (grad_L_S (θ t)) (deriv θ t))
    (h_flow : ∀ t, deriv θ t = -(grad_L_S (θ t))) :
    deriv (fun t => L_D (θ t) - L_S (θ t)) t
      = inner ℝ (grad_L_S (θ t) - grad_L_D (θ t)) (grad_L_S (θ t)) := by
  have h_deriv : deriv (fun t => L_D (θ t) - L_S (θ t)) t
      = deriv (fun t => L_D (θ t)) t - deriv (fun t => L_S (θ t)) t :=
    deriv_sub (h_diff_D t) (h_diff_S t)
  rw [h_deriv, h_grad_D t, h_grad_S t, h_flow t, inner_sub_left,
      inner_neg_right, inner_neg_right, sub_neg_eq_add, add_comm,
      ← sub_eq_add_neg, ← inner_sub_left]

end Mane
