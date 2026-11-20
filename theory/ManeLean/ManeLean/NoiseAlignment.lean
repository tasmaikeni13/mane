/-
# Noise–gradient alignment

Write the sampled gradient as ∇L_S = ξ + ∇L_D (noise + population gradient) with
E⟨ξ, ∇L_D⟩ = 0 (noise uncorrelated with the true direction). Then

    E⟨ξ, ∇L_S⟩ = E⟨ξ, ξ⟩ = noise energy.

Pairing with the gap-growth law (GapGrowth.lean): the expected gap growth along
empirical gradient flow equals E⟨∇L_S − ∇L_D, ∇L_S⟩ = E⟨ξ, ∇L_S⟩ — i.e. the gap
grows at exactly the rate of the noise energy, which is the quantity the MANE
gate measures (Variance.lean) and suppresses. Expectation and inner product enter
as hypothesis-level structure (bilinearity of the composite E⟪·,·⟫); no axioms,
no `sorry`.
-/
import Mathlib.Tactic

namespace Mane

/-- If `Eip` (the expected inner product) is additive in its second slot and the
noise is uncorrelated with the population gradient, then testing the noise
against the SAMPLED gradient returns exactly the noise energy. -/
theorem noise_alignment {V : Type*} [Add V]
    (Eip : V → V → ℝ)
    (ξ gradD : V)
    (Eip_add_right : ∀ x y z : V, Eip x (y + z) = Eip x y + Eip x z)
    (h_orth : Eip ξ gradD = 0) :
    Eip ξ (ξ + gradD) = Eip ξ ξ := by
  rw [Eip_add_right, h_orth, add_zero]

end Mane
