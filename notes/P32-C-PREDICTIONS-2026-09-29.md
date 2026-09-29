# P32 (c): swing locking — predictions fixed before any code

Claude Fable 5.1, ec5ece87, 2026-09-29. For the book's §1.5 Synchronization item, second sentence:
"given inertia, they would keep in step the way generators on a power grid do." The library's swing
arrangement (`kuramoto_arrangement(..., parameter=phase)`) is undamped and `test_kuramoto.py`'s
phase test shows swinging without decay, not locking. This run adds damping through the existing
`coalgebra.damped(Q, c)` integrator, which removes a fraction c of the momentum each step, on the same
two-oscillator lens data as the locking test (natural frequencies 0.8 and −0.2, so Δω = 1, threshold
K_c = 0.5).

The map, derived by hand before coding (unit step; sharp 1/m; the forcing on oscillator i is
ξ_i = ω_i + K sin(θ_j − θ_i) evaluated at the new angles): θ_i' = θ_i + ξ_i/m; ξ_i' = (1−c)ξ_i + ξ_Q,i(θ').
A locked state has constant momenta, cξ_i = ξ_Q,i, and equal per-step advances ξ_1 = ξ_2, so
ω_1 − K sin δ = ω_2 + K sin δ: the locked difference is δ* = arcsin(Δω/2K), Kuramoto's own, and the
common advance per step is mean(ω)/(c m). Linearized at δ* with a = 2K cos δ*, the (ε, η) map has
trace 2 − c − a/m and determinant 1 − c, so it is stable whenever it is a rest, and its eigenvalues
are complex (an oscillatory, overshooting approach) iff c + a/m < 2√(a/m), real (monotone) otherwise.

Parameters: m = 20, K = 0.8 (above threshold) or 0.3 (below), c = 0.05 (small) or 0.9 (large).
At K = 0.8: sin δ* = 0.625, cos δ* = 0.7806, a = 1.249, a/m = 0.0624, 2√(a/m) = 0.4997; so c = 0.05
is complex and c = 0.9 is real.

## Predictions, each able to fail

1. **Lock above threshold.** K = 0.8, c = 0.05: δ settles to arcsin(Δω/2K) = arcsin(0.625) to 1e-8
   within 4000 steps and is constant over the last 500 steps to 1e-8; both oscillators advance by
   mean(ω)/(c m) = 0.3 per step, measured over the last 1000 steps, to 1e-6.
2. **Drift below threshold.** K = 0.3, c = 0.05: δ grows without bound (by more than 4π over 6000
   steps) and the two measured per-step advances differ by more than 0.2.
3. **No damping, no lock.** K = 0.8, c = 0: over the last 1000 of 4000 steps δ still varies by more
   than 0.1 rad (the existing swinging test's behaviour), so inertia alone does not keep them in step.
4. **Overshoot or not, as the linearization says.** With c = 0.05 the sign of δ − δ* changes at least
   once after step 50; with c = 0.9 it does not change after step 50.

If (1) fails, the sentence in §1.5 goes. If (3) fails, the sentence stands even without friction and the
book's claim is stronger than I expect.

## Domain objection (Claude's judgment)

A power engineer would say generator damping acts on the frequency deviation through the damper
windings and the governor, not as viscous friction on a momentum, and that real grids lock through
active frequency control. The model survives as the classical second-order (inertial) Kuramoto model
with linear damping, which is the textbook first model of grid synchronization; it does not model
governors or loads. Source to cite with a locator before any prose: not yet checked.

## Verdicts (`tests/test_swing_locking.py`, 4 passed in 1.9 s; run 2026-09-29, ec5ece87)

1. **Held.** δ settles to arcsin(0.625) to 1e-8 and is constant over the last 500 steps; both advance
   0.3 rad per step to 1e-6, which is mean(ω)/(c m) exactly as derived.
2. **Held.** Below threshold δ grows past 4π over 6000 steps; the advances differ by more than 0.2.
3. **Held.** With c = 0 the difference still swings by more than 0.1 rad over the last 1000 steps.
4. **Held.** c = 0.05 (complex eigenvalues) overshoots δ*; c = 0.9 (real) approaches monotonically.

Which could have come out otherwise: (1) and (4) are closed forms of the discrete map, so once the
derivation is right they had to hold; the run checks the derivation, not the world. (3) could have
failed if the undamped map happened to settle numerically; it did not.

**Reasonable** (Claude's judgment): at the grid scale, the textbook swing equation locks only with
damping, and the locked phase difference is Kuramoto's, so the second-order model adds nothing to
where they lock and everything to how (overshoot). At the firefly scale inertia is not the model.
**Weird**: with c = 0 and unequal natural frequencies the mean momentum grows without bound (the
natural frequencies act as constant torques with nothing to spend them on), so the undamped pair's
angles run to about 1e5 rad by step 4000; a rotor with no load. Physically right, numerically ugly.

**For the book.** §1.5's "given inertia, they would keep in step the way generators on a power
grid do" holds in the model for every damping short of none, and fails at none. Candidate, not
applied: "given inertia and a little friction, they would keep in step …".
