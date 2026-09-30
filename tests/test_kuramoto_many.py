"""More than two oscillators: all-to-all falls into step; a ring can stay out of step.

Book: §1.5, the Synchronization item ("Coupled strongly enough, the oscillators would fall into
step, as fireflies come to flash together; given some inertia and friction, they would keep in step
the way generators on a power grid do"). test_kuramoto.py and test_swing_locking.py run two
oscillators only; the Ch 1 domain check (book repo, notes/DOMAIN-CHECK-CH1-2026-09-29.md, Group C)
found that "the oscillators" in general is false on a ring, and David asked for the run rather than
the narrowing (2026-09-29: "If the test hasn't been run, just run it and then be willing to say more
than two"). Predictions fixed first in ``notes/KURAMOTO-MANY-PREDICTIONS-2026-09-29.md``.

The model is the library's composite ``kuramoto_arrangement`` stepped by ``Phi``; all-to-all puts
``kappa / N`` on each edge, so the closed map is forward-Euler (step ``dt``) of the classical
``theta_i' = omega_i + (kappa/N) sum_j sin(theta_j - theta_i)``; natural frequencies evenly spaced
over ``[-0.5, 0.5]``; "in step" means equal average rates over the last window.

What each test does: tests 1, 2 and 4 CHECK the discrete analogue of the classical behaviour
(locking above the coupling threshold, none below, locking with inertia and damping); test 3
REPRODUCES a closed form, the twisted ring state being an exact fixed point of the discrete map,
and checks that it is stable to a small nudge. Claude Opus 5.5 (1M), 16311c9d, 2026-09-29.
"""
from functools import partial

import jax.numpy as jnp
import numpy as np

from arrangements.coalgebra import damped
from arrangements.functors import Phi
from arrangements.kuramoto import kuramoto_arrangement
from tests import test_kuramoto as tk

DT, STEPS, WINDOW = 0.05, 4000, 1000


def omegas(N):
    return list(np.linspace(-0.5, 0.5, N))


def all_edges(N):
    return [(i, j) for i in range(N) for j in range(i + 1, N)]


def run_conf(om, edges, K, theta0, steps=STEPS, dt=DT):
    N, E = len(om), len(edges)
    step = tk._stepper(Phi(kuramoto_arrangement(om, edges, K, sharp_scale=dt)))
    s = tk._conf_state(jnp.asarray(theta0, float), E)
    th = np.empty((steps + 1, N))
    th[0] = theta0
    for t in range(steps):
        s = step(s)
        th[t + 1] = np.asarray(tk._conf_flat(s, N, E))
    return th


def rates(th, dt=DT, window=WINDOW):
    return (th[-1] - th[-1 - window]) / (window * dt)


def order(theta):
    return abs(np.mean(np.exp(1j * np.asarray(theta))))


def test_1_all_to_all_falls_into_step_above_threshold():
    """kappa = 2: every oscillator ends up at the mean rate, for N = 3, 5, 10, from random angles."""
    rng = np.random.default_rng(0)
    for N in (3, 5, 10):
        th = run_conf(omegas(N), all_edges(N), 2.0 / N, rng.uniform(0, 2 * np.pi, N))
        r = rates(th)
        assert np.ptp(r) < 1e-6, (N, r)
        np.testing.assert_allclose(r, np.mean(omegas(N)), atol=1e-6)


def test_2_all_to_all_stays_apart_below_threshold():
    """kappa = 0.3: the average rates stay apart (spread above 0.05) for each N."""
    rng = np.random.default_rng(1)
    for N in (3, 5, 10):
        th = run_conf(omegas(N), all_edges(N), 0.3 / N, rng.uniform(0, 2 * np.pi, N))
        assert np.ptp(rates(th)) > 0.05, N


def test_3_a_ring_can_stay_out_of_step():
    """Ten identical oscillators on a ring, K = 1, started twisted and nudged: the twist returns."""
    N = 10
    ring = [(i, (i + 1) % N) for i in range(N)]
    twist = 2 * np.pi * np.arange(N) / N
    # CLOSED FORM: the twist is an exact fixed point (each oscillator's two pulls cancel).
    th = run_conf([0.0] * N, ring, 1.0, twist, steps=10)
    np.testing.assert_allclose(th[-1], twist, atol=1e-12)
    # ...and it is stable: a small nudge dies away and the oscillators never gather.
    rng = np.random.default_rng(2)
    th = run_conf([0.0] * N, ring, 1.0, twist + 0.01 * rng.standard_normal(N))
    gaps = np.mod(np.diff(np.r_[th[-1], th[-1][0] + 2 * np.pi]), 2 * np.pi)
    np.testing.assert_allclose(gaps, 2 * np.pi / N, atol=0.02)
    assert max(order(t) for t in th[::100]) < 0.05


def test_4_with_inertia_and_friction_many_keep_in_step():
    """Swing with damping, all-to-all N = 5, kappa = 2, from rest: the average rates agree."""
    N, m, c, steps = 5, 20.0, 0.05, 6000
    om, edges = omegas(N), all_edges(N)
    arr = kuramoto_arrangement(om, edges, 2.0 / N, sharp_scale=1.0 / m,
                               parameter=partial(damped, damping=c))
    step = tk._stepper(Phi(arr))
    rng = np.random.default_rng(3)
    theta0 = rng.uniform(0, 2 * np.pi, N)
    s = tk._phase_state(jnp.asarray(theta0), jnp.zeros(N), len(edges))
    th = np.empty((steps + 1, N))
    th[0] = theta0
    for t in range(steps):
        s = step(s)
        th[t + 1] = np.asarray(tk._phase_flat(s, N, len(edges))[0])
    advance = (th[-1] - th[-1 - WINDOW]) / WINDOW          # per unit step
    assert np.ptp(advance) < 1e-6, advance
