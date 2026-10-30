import Mathlib.Data.Complex.Exponential
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

/-!
Contratos reais do gate e das derivadas entre observações válidas.
As hipóteses de sinais e flags são explícitas; ver README.md para o vínculo Python.
-/

noncomputable section
namespace GateFall

def sigmoid (x : ℝ) : ℝ := 1 / (1 + Real.exp (-x))

theorem sigmoid_pos (x : ℝ) : 0 < sigmoid x := by
  unfold sigmoid
  exact div_pos zero_lt_one (by linarith [Real.exp_pos (-x)])

theorem sigmoid_lt_one (x : ℝ) : sigmoid x < 1 := by
  unfold sigmoid
  apply (div_lt_iff₀ (by linarith [Real.exp_pos (-x)] : 0 < 1 + Real.exp (-x))).2
  linarith [Real.exp_pos (-x)]

theorem sigmoid_monotone : Monotone sigmoid := by
  intro x y hxy
  unfold sigmoid
  apply one_div_le_one_div_of_le (by linarith [Real.exp_pos (-y)])
  have hexp : Real.exp (-y) ≤ Real.exp (-x) := Real.exp_le_exp.mpr (neg_le_neg hxy)
  linarith

def score (b α β γ δ qp qv rp rv : ℝ) : ℝ :=
  b + α * qp - β * qv - γ * rp + δ * rv

def gate (b α β γ δ qp qv rp rv : ℝ) : ℝ := sigmoid (score b α β γ δ qp qv rp rv)

theorem gate_pose_quality (b α β γ δ qp qp' qv rp rv : ℝ)
    (hα : 0 ≤ α) (hq : qp ≤ qp') :
    gate b α β γ δ qp qv rp rv ≤ gate b α β γ δ qp' qv rp rv := by
  apply sigmoid_monotone
  unfold score
  nlinarith

theorem gate_visual_quality (b α β γ δ qp qv qv' rp rv : ℝ)
    (hβ : 0 ≤ β) (hq : qv ≤ qv') :
    gate b α β γ δ qp qv' rp rv ≤ gate b α β γ δ qp qv rp rv := by
  apply sigmoid_monotone
  unfold score
  nlinarith

theorem gate_pose_inconsistency (b α β γ δ qp qv rp rp' rv : ℝ)
    (hγ : 0 ≤ γ) (hr : rp ≤ rp') :
    gate b α β γ δ qp qv rp' rv ≤ gate b α β γ δ qp qv rp rv := by
  apply sigmoid_monotone
  unfold score
  nlinarith

theorem gate_visual_inconsistency (b α β γ δ qp qv rp rv rv' : ℝ)
    (hδ : 0 ≤ δ) (hr : rv ≤ rv') :
    gate b α β γ δ qp qv rp rv ≤ gate b α β γ δ qp qv rp rv' := by
  apply sigmoid_monotone
  unfold score
  nlinarith

def maskedGate (b α β γ δ qp qv rp rv rpValid rvValid : ℝ) : ℝ :=
  gate b α β (γ * rpValid) (δ * rvValid) qp qv rp rv

theorem maskedGate_bounds (b α β γ δ qp qv rp rv rpValid rvValid : ℝ) :
    0 < maskedGate b α β γ δ qp qv rp rv rpValid rvValid ∧
      maskedGate b α β γ δ qp qv rp rv rpValid rvValid < 1 := by
  exact ⟨sigmoid_pos _, sigmoid_lt_one _⟩

theorem maskedGate_pose_inconsistency
    (b α β γ δ qp qv rp rp' rv rpValid rvValid : ℝ)
    (hγ : 0 ≤ γ) (hvalid : 0 ≤ rpValid) (hr : rp ≤ rp') :
    maskedGate b α β γ δ qp qv rp' rv rpValid rvValid ≤
      maskedGate b α β γ δ qp qv rp rv rpValid rvValid := by
  exact gate_pose_inconsistency _ _ _ _ _ _ _ _ _ _ (mul_nonneg hγ hvalid) hr

theorem maskedGate_visual_inconsistency
    (b α β γ δ qp qv rp rv rv' rpValid rvValid : ℝ)
    (hδ : 0 ≤ δ) (hvalid : 0 ≤ rvValid) (hr : rv ≤ rv') :
    maskedGate b α β γ δ qp qv rp rv rpValid rvValid ≤
      maskedGate b α β γ δ qp qv rp rv' rpValid rvValid := by
  exact gate_visual_inconsistency _ _ _ _ _ _ _ _ _ _ (mul_nonneg hδ hvalid) hr

theorem invalid_pose_residual_ignored
    (b α β γ δ qp qv rp rp' rv rvValid : ℝ) :
    maskedGate b α β γ δ qp qv rp rv 0 rvValid =
      maskedGate b α β γ δ qp qv rp' rv 0 rvValid := by
  simp [maskedGate, gate, score]

theorem invalid_visual_residual_ignored
    (b α β γ δ qp qv rp rv rv' rpValid : ℝ) :
    maskedGate b α β γ δ qp qv rp rv rpValid 0 =
      maskedGate b α β γ δ qp qv rp rv' rpValid 0 := by
  simp [maskedGate, gate, score]

def orderHinge (margin clean corrupted : ℝ) : ℝ := max 0 (margin + corrupted - clean)

theorem orderHinge_nonneg (margin clean corrupted : ℝ) :
    0 ≤ orderHinge margin clean corrupted := le_max_left _ _

theorem constant_gate_positive_loss (margin g : ℝ) (hm : 0 < margin) :
    0 < orderHinge margin g g := by
  unfold orderHinge
  have heq : margin + g - g = margin := by ring
  rw [heq, max_eq_right hm.le]
  exact hm

theorem zero_hinge_requires_margin (margin clean corrupted : ℝ)
    (hzero : orderHinge margin clean corrupted = 0) : corrupted + margin ≤ clean := by
  have h := le_max_right (0 : ℝ) (margin + corrupted - clean)
  change margin + corrupted - clean ≤ orderHinge margin clean corrupted at h
  rw [hzero] at h
  linarith

def meanVelocity (x0 x1 t0 t1 : ℝ) : ℝ := (x1 - x0) / (t1 - t0)

def midpointAcceleration (x0 x1 x2 t0 t1 t2 : ℝ) : ℝ :=
  (meanVelocity x1 x2 t1 t2 - meanVelocity x0 x1 t0 t1) /
    ((t1 + t2) / 2 - (t0 + t1) / 2)

theorem uniform_velocity (x v t0 t1 : ℝ) (ht : t0 < t1) :
    meanVelocity (x + v * t0) (x + v * t1) t0 t1 = v := by
  unfold meanVelocity
  apply (div_eq_iff (ne_of_gt (sub_pos.mpr ht))).2
  ring

theorem uniform_motion_zero_acceleration (x v t0 t1 t2 : ℝ)
    (h01 : t0 < t1) (h12 : t1 < t2) :
    midpointAcceleration (x + v * t0) (x + v * t1) (x + v * t2) t0 t1 t2 = 0 := by
  unfold midpointAcceleration
  rw [uniform_velocity x v t1 t2 h12, uniform_velocity x v t0 t1 h01]
  simp

theorem midpoint_interval_positive (t0 t1 t2 : ℝ)
    (h01 : t0 < t1) (h12 : t1 < t2) :
    0 < (t1 + t2) / 2 - (t0 + t1) / 2 := by
  linarith

theorem uniform_acceleration_exact (x v a t0 t1 t2 : ℝ)
    (h01 : t0 < t1) (h12 : t1 < t2) :
    midpointAcceleration
      (x + v * t0 + a / 2 * t0 ^ 2)
      (x + v * t1 + a / 2 * t1 ^ 2)
      (x + v * t2 + a / 2 * t2 ^ 2) t0 t1 t2 = a := by
  unfold midpointAcceleration meanVelocity
  have hd01 : t1 - t0 ≠ 0 := ne_of_gt (sub_pos.mpr h01)
  have hd12 : t2 - t1 ≠ 0 := ne_of_gt (sub_pos.mpr h12)
  have hdm : (t1 + t2) / 2 - (t0 + t1) / 2 ≠ 0 :=
    ne_of_gt (midpoint_interval_positive t0 t1 t2 h01 h12)
  apply (div_eq_iff hdm).2
  field_simp [hd01, hd12]
  ring

end GateFall


#print axioms GateFall.sigmoid_pos
#print axioms GateFall.sigmoid_lt_one
#print axioms GateFall.sigmoid_monotone
#print axioms GateFall.gate_pose_quality
#print axioms GateFall.gate_visual_quality
#print axioms GateFall.gate_pose_inconsistency
#print axioms GateFall.gate_visual_inconsistency
#print axioms GateFall.constant_gate_positive_loss
#print axioms GateFall.uniform_motion_zero_acceleration
#print axioms GateFall.uniform_acceleration_exact
#print axioms GateFall.maskedGate_bounds
#print axioms GateFall.maskedGate_pose_inconsistency
#print axioms GateFall.maskedGate_visual_inconsistency
#print axioms GateFall.invalid_pose_residual_ignored
#print axioms GateFall.invalid_visual_residual_ignored
#print axioms GateFall.orderHinge_nonneg
#print axioms GateFall.zero_hinge_requires_margin
#print axioms GateFall.uniform_velocity
#print axioms GateFall.midpoint_interval_positive
