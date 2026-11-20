/-
# The two-fold noise estimator is unbiased

MANE's noise measurement: from two independent, equal-mean gradient estimates
`X` (fold A) and `Y` (fold B) of the same true gradient coordinate,

    E[ ((X − Y)/2)² ] = Var[ (X + Y)/2 ].

The left side is the quantity MANE tracks in its noise EMA `v`; the right side is
exactly the variance of the mean gradient ḡ the optimizer steps with. So `v` is
an unbiased estimator of the per-coordinate noise power of ḡ — the σ² feeding the
Wiener gate (OptimalGate.lean).

Proved from Mathlib's probability library (independence ⇒ zero covariance,
variance algebra). No axioms, no `sorry`. Note only EQUAL MEANS of the folds is
needed — equal fold variances is NOT required.
-/
import Mathlib.Probability.Moments.Variance

open MeasureTheory ProbabilityTheory

namespace Mane

variable {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω} [IsProbabilityMeasure μ]

theorem two_fold_noise_unbiased (X Y : Ω → ℝ)
    (hX : MemLp X 2 μ) (hY : MemLp Y 2 μ)
    (hindep : IndepFun X Y μ)
    (hmean : ∫ ω, X ω ∂μ = ∫ ω, Y ω ∂μ) :
    ∫ ω, ((X ω - Y ω) / 2) ^ 2 ∂μ
      = variance (fun ω => (X ω + Y ω) / 2) μ := by
  have hXi : Integrable X μ := hX.integrable one_le_two
  have hYi : Integrable Y μ := hY.integrable one_le_two
  have hDfun : (fun ω => (X ω - Y ω) / 2) = (2⁻¹ : ℝ) • fun ω => X ω - Y ω := by
    funext ω; simp [Pi.smul_apply, smul_eq_mul, div_eq_inv_mul]
  have hMfun : (fun ω => (X ω + Y ω) / 2) = (2⁻¹ : ℝ) • fun ω => X ω + Y ω := by
    funext ω; simp [Pi.smul_apply, smul_eq_mul, div_eq_inv_mul]
  have hD2 : MemLp (fun ω => (X ω - Y ω) / 2) 2 μ := by
    rw [hDfun]; exact (hX.sub hY).const_smul _
  -- E[D] = 0 from equal means
  have hED : ∫ ω, (X ω - Y ω) / 2 ∂μ = 0 := by
    rw [integral_div, integral_sub hXi hYi, hmean, sub_self, zero_div]
  -- E[D²] = Var[D] + E[D]² = Var[D]
  have hED2 : ∫ ω, ((X ω - Y ω) / 2) ^ 2 ∂μ
      = variance (fun ω => (X ω - Y ω) / 2) μ := by
    have h := variance_eq_sub hD2
    simp only [Pi.pow_apply] at h
    rw [hED] at h
    simp at h
    exact h.symm
  -- independence kills the covariance
  have hcov : cov[X, Y; μ] = 0 := hindep.covariance_eq_zero hX hY
  -- Var[D] = ¼(Var X + Var Y) = Var[M]
  have hVD : variance (fun ω => (X ω - Y ω) / 2) μ
      = (variance X μ + variance Y μ) / 4 := by
    rw [hDfun, variance_smul, variance_fun_sub hX hY, hcov]
    ring
  have hVM : variance (fun ω => (X ω + Y ω) / 2) μ
      = (variance X μ + variance Y μ) / 4 := by
    rw [hMfun, variance_smul, variance_fun_add hX hY, hcov]
    ring
  rw [hED2, hVD, hVM]

end Mane
