"""Swing locking: inertia alone does not keep two oscillators in step; inertia with damping does.

Book: §1.5, the Synchronization item's second sentence ("given inertia, they would keep in step
the way generators on a power grid do"); plan P32 (c). The library's swing arrangement is
``kuramoto_arrangement(..., parameter=phase)``, the same oscillator and coupling boxes as
Kuramoto's with the phase parameter and sharp ``1/m``; test_kuramoto.py section 4 shows it swings
without decay. This file runs the same lens data with ``coalgebra.damped(Q, c)``, the phase
coalgebra that removes a fraction ``c`` of the momentum each step, so the closed map is (unit step;
the forcing on oscillator i is ``xi_Q,i = omega_i + K sin(theta_j - theta_i)`` at the new angles)

    theta_i' = theta_i + xi_i / m,      xi_i' = (1 - c) xi_i + xi_Q,i(theta').

Predictions, fixed in ``notes/P32-C-PREDICTIONS-2026-09-29.md`` before this file existed
(natural frequencies 0.8 and -0.2, so dw = 1 and the Kuramoto threshold is K_c = 0.5; m = 20).
At a locked state the momenta are constant, ``c xi_i = xi_Q,i``, and the advances equal, so
``omega_1 - K sin(delta) = omega_2 + K sin(delta)``: ``delta* = arcsin(dw / 2K)``, Kuramoto's own
locked difference, and the common advance per step is ``mean(omega) / (c m)``.

1. Lock above threshold (K = 0.8, c = 0.05): delta -> arcsin(dw/2K) to 1e-8, constant over the last
   500 of 4000 steps; both oscillators advance mean(omega)/(c m) = 0.3 per step (last 1000 steps).
2. Drift below threshold (K = 0.3, c = 0.05): delta grows by more than 4 pi over 6000 steps and the
   two per-step advances differ by more than 0.2.
3. No damping, no lock (K = 0.8, c = 0): delta still varies by more than 0.1 rad over the last 1000
   of 4000 steps.
4. Overshoot as the linearization says: at delta* the (eps, eta) map has trace ``2 - c - a/m`` and
   determinant ``1 - c`` with ``a = 2K cos(delta*)``, complex eigenvalues iff
   ``c + a/m < 2 sqrt(a/m)``; so c = 0.05 overshoots (the sign of delta - delta* changes after
   step 50, while |delta - delta*| > 1e-6) and c = 0.9 does not.

What each test does: tests 1 and 4 REPRODUCE CLOSED FORMS derived by hand for the discrete map (the
fixed point is exact, not O(step)-near); tests 2 and 3 CHECK the discrete analogue of the classical
behaviour (drift below threshold; an undamped driven rotor pair never settles) and illustrate it.
Conventions as in test_kuramoto.py: unit step, ``1/m`` in the sharp, the book's sign (the covector
is added). Claude Fable 5.1, ec5ece87, 2026-09-29.
"""
from functools import partial

import jax.numpy as jnp
import numpy as np

from arrangements.coalgebra import damped
from arrangements.functors import Phi
from arrangements.kuramoto import kuramoto_arrangement
from tests import test_kuramoto as tk

OMEGAS = tk._TWO_OMEGAS  # (0.8, -0.2): dw = 1.0, mean = 0.3, K_c = 0.5
DW = OMEGAS[0] - OMEGAS[1]
M = 20.0


def _run(K, c, steps, theta0=(0.0, 0.0)):
    """Angles along ``steps`` closed steps of the damped swing pair from rest at ``theta0``."""
    arr = kuramoto_arrangement(
        list(OMEGAS), [(0, 1)], K, sharp_scale=1.0 / M, parameter=partial(damped, damping=c)
    )
    step = tk._stepper(Phi(arr))
    s = tk._phase_state(jnp.asarray(theta0, float), jnp.zeros(2), 1)
    thetas = np.empty((steps + 1, 2))
    thetas[0] = np.asarray(theta0, float)
    for t in range(steps):
        s = step(s)
        q, _ = tk._phase_flat(s, 2, 1)
        thetas[t + 1] = np.asarray(q)
    return thetas


def test_prediction_1_lock_above_threshold_with_damping():
    """CLOSED FORM: delta -> arcsin(dw/2K) exactly; both advance mean(omega)/(c m) per step."""
    K, c, steps = 0.8, 0.05, 4000
    th = _run(K, c, steps)
    delta = th[:, 0] - th[:, 1]
    delta_star = np.arcsin(DW / (2 * K))
    np.testing.assert_allclose(delta[-1], delta_star, rtol=0, atol=1e-8)
    assert np.max(np.abs(delta[-500:] - delta[-1])) < 1e-8
    advance = (th[-1] - th[-1001]) / 1000.0
    np.testing.assert_allclose(advance, np.full(2, np.mean(OMEGAS) / (c * M)), rtol=0, atol=1e-6)


def test_prediction_2_drift_below_threshold():
    """DISCRETE ANALOGUE: below K_c the difference runs away and the advances stay distinct."""
    K, c, steps = 0.3, 0.05, 6000
    th = _run(K, c, steps)
    delta = th[:, 0] - th[:, 1]
    assert delta[-1] - delta[0] > 4 * np.pi
    advance = (th[-1] - th[-1001]) / 1000.0
    assert advance[0] - advance[1] > 0.2


def test_prediction_3_inertia_alone_does_not_lock():
    """DISCRETE ANALOGUE: with c = 0 the difference keeps swinging; nothing settles."""
    K, steps = 0.8, 4000
    th = _run(K, 0.0, steps)
    delta = th[:, 0] - th[:, 1]
    assert np.max(delta[-1000:]) - np.min(delta[-1000:]) > 0.1


def test_prediction_4_overshoot_follows_the_linearization():
    """CLOSED FORM: complex eigenvalues iff c + a/m < 2 sqrt(a/m); overshoot iff complex."""
    K = 0.8
    delta_star = np.arcsin(DW / (2 * K))
    a = 2 * K * np.cos(delta_star)
    for c, expect_complex in ((0.05, True), (0.9, False)):
        assert (c + a / M < 2 * np.sqrt(a / M)) == expect_complex
        th = _run(K, c, 4000)
        d = th[:, 0] - th[:, 1] - delta_star
        d = d[50:]
        signs = np.sign(d[np.abs(d) > 1e-6])
        changes = int(np.sum(np.diff(signs) != 0))
        if expect_complex:
            assert changes >= 1, "small damping should overshoot"
        else:
            assert changes == 0, f"large damping should approach monotonically, saw {changes}"
