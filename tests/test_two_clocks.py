"""Two clocks and the furniture room (book Ch 1 l.565, ``exc.furniture``; plan P41 (d), Claude Opus 5.5, b94c8c22).

Predictions were written before the run in the book's ``notes/runs/p41d-b94c8c22/RESULTS.md`` §0; the run's record
is §2 there. What runs in ``arr``: every room below is an arrangement built from parts and couplings and stepped by
the dynamics functor (``routine_step`` and ``open_step`` are checked against ``Phi``); the Euler recurrence in
``test_rooms_match_euler`` is an ordinary calculation beside them, a check only.

Laws (a test of a structure tests each of its laws): ``slowed`` on the unit integrator, at 1, composed, and on a
tensor; the fill of a two-clock arrangement on a composite with two inputs (the room's wiring hears both units), in
three orders.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arrangements import two_clocks as tc
from arrangements.coalgebra import configuration
from arrangements.rvect import euclidean
from arrangements.wiring import trivial_parameter

jax.config.update("jax_enable_x64", True)

TAU = 0.05
Q0, W0 = (-2.4, 2.45), (-2.5, 2.55)

# position orders: flat (q1, q2, w1, w2); para (w1, w2, q1, q2); unitwise (w1, q1, w2, q2)
TO_FLAT = {"flat": [0, 1, 2, 3], "para": [2, 3, 0, 1], "unitwise": [1, 3, 0, 2]}
BUILD = {"flat": tc.room_flat, "para": tc.room_para, "unitwise": tc.room_unitwise}


def _path(kind, eps, inputs):
    arr = BUILD[kind](eps, Q0, W0)
    step = tc.routine_step(arr, TAU)
    return tc.run(arr, step, inputs, record=True)[:, TO_FLAT[kind]]


def test_slowed_laws():
    rng = np.random.default_rng(0)
    P = configuration(euclidean(1, 1.0), init=jnp.array([0.3]))
    P2 = configuration(euclidean(2, 0.7), init=jnp.array([0.1, -0.2]))
    for _ in range(5):
        s, xi, tau = jnp.array([rng.normal()]), jnp.array([rng.normal()]), float(rng.uniform(0, 2))
        assert np.allclose(tc.slowed(P, 1.0).timed_step(s, xi, tau), P.timed_step(s, xi, tau))
        assert np.allclose(tc.slowed(tc.slowed(P, 0.3), 0.5).timed_step(s, xi, tau),
                           tc.slowed(P, 0.15).timed_step(s, xi, tau))
        assert np.allclose(tc.slowed(P, 0.0).timed_step(s, xi, tau), s)          # frozen
        s2, xi2 = (s, jnp.array(rng.normal(size=2))), jnp.array(rng.normal(size=3))
        lhs = tc.slowed(P.tensor(P2), 0.2).timed_step(s2, xi2, tau)
        rhs = tc.slowed(P, 0.2).tensor(tc.slowed(P2, 0.2)).timed_step(s2, xi2, tau)
        for x, y in zip(jax.tree_util.tree_leaves(lhs), jax.tree_util.tree_leaves(rhs)):
            assert np.allclose(x, y)
    U = trivial_parameter()
    assert tc.slowed(U, 0.4).timed_step(U.init, jnp.zeros(0), 1.0).shape == (0,)  # the unit stays the unit


@pytest.mark.parametrize("eps", [0.01, 0.3])
def test_fill_commutes_with_composition(eps):
    """Flat, Para-then-fill and fill-then-compose step alike, routine on then off."""
    inputs = [1.0] * 40 + [0.0] * 40
    flat = _path("flat", eps, inputs)
    for kind in ("para", "unitwise"):
        assert np.allclose(_path(kind, eps, inputs), flat, rtol=1e-12, atol=1e-12), kind


def test_rooms_match_euler():
    """The flat room against the hand recurrence (beside arr, a check): people response 1, chairs eps."""
    eps, a, c, b, w0 = 0.01, 1.0, 3.0, 2.6, (-2.5, 2.5)
    inputs = [1.0] * 30 + [0.0] * 30
    q1, q2, w1, w2 = Q0[0], Q0[1], W0[0], W0[1]
    hand = []
    for u in inputs:
        e = np.exp(-(q1 - q2) ** 2 / 2)
        f1 = -a * (q1 - w1) + c * (q2 - q1) * e + u * (q2 - q1)
        f2 = -a * (q2 - w2) + c * (q1 - q2) * e + u * (q1 - q2)
        g1 = a * (q1 - w1) - b * (w1 - w0[0])
        g2 = a * (q2 - w2) - b * (w2 - w0[1])
        q1, q2, w1, w2 = q1 + TAU * f1, q2 + TAU * f2, w1 + eps * TAU * g1, w2 + eps * TAU * g2
        hand.append([q1, q2, w1, w2])
    assert np.allclose(_path("flat", eps, inputs), np.array(hand), rtol=1e-12, atol=1e-12)


def test_open_person_matches_phi():
    arr = tc.person_open(0.3)
    tc.open_step(arr, TAU, check_at=[-0.3, 1.9])
