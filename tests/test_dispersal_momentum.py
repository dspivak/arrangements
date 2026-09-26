"""Dispersal with momentum: does a push's energy spread irrecoverably through a collective?

The second half of the book plan's P17 (AGENTDOCPLAN.md §8.2 in the book repository), in
David's words (2026-09-26): "in a ratchet, i want the click to be the sound of energy
spreading irrecoverably through the handle of the ratchet", and "if it remained right
there, the spring would bounce back up and the ratchet would go backwards".
``tests/test_dispersal.py`` ran the same collective under descent, where a displacement
carries no energy. Here the parent's integrator is the phase coalgebra (and its damped
variant): each placement carries a momentum, and a part is *pushed* by starting it with
momentum ``xi_0`` outward at a common equilibrium. Nothing else changes: frozen parts
sending ``(1, 1)``, the parent's cost ``(1/2)(nu - sum_j kappa_j)^2`` on the total it sends.

**The predictions, fixed in advance and written here before any construction code for this
file existed. Each must be able to fail.**

Linear analysis first. Near a common equilibrium the cost is ``(1/2) g^2 (sum_j x_j)^2``
with ``g = kappa'(p*)``: its Hessian has rank one. The one direction it sees, the total,
is a spring; the other ``n - 1`` directions are free. A push ``xi_0`` on part 0 splits into
``xi_0 / n`` on every part along the spring and a free remainder, ``xi_0 (1 - 1/n)`` on
part 0 and ``-xi_0 / n`` on each other part; the free remainder carries the fraction
``1 - 1/n`` of the push's kinetic energy.

(5) *Undamped, a small push: the spring's share comes back and the rest never does.* Over
many oscillations each part's momentum averages to its free share: ``xi_0 (1 - 1/n)`` for
the pushed part, ``-xi_0 / n`` for each other, within 5%; for ``n = 1`` it averages to zero.
So the fraction ``1 - 1/n`` of the energy put in spreads into motions the cost cannot see
and does not return to the spring. Fails if the averages differ from the free shares.

(6) *Damped, a small push lands the configuration somewhere new.* With the damped phase
integrator (a fraction ``c`` of momentum removed per step), the spring's share rings down
and returns, and the free share coasts a distance ``mu xi_0 / c`` before it stops: the pushed
part ends displaced by ``(mu xi_0 / c)(1 - 1/n)`` and each other by ``-(mu xi_0 / c) / n``,
within 5%, with the onward message back at ``nu``; for ``n = 1`` it ends where it started.
This is the click in this model: the push's energy is gone, dissipated, and the
configuration keeps the part of the push the cost could not see.

(7) *Undamped, redundancy decides whether a small push loses a part.* At target
``nu = 1.5`` and push ``xi_0 = 0.05`` (kinetic energy ``~6e-4``): with ``n = 3`` (the other
two can cover 1.5) the pushed part coasts out of reach, past distance 5 within 5000 steps,
and does not come back; with ``n = 2`` (the other can send at most 1) it moves along the
one-dimensional family of minima until the other part reaches the kernel's peak, turns, and
never gets past distance 1.2 in 100,000 steps. This goes beyond the linear analysis, which
says only that the free share coasts; whether the others compensate fast enough for the
pushed part to leave, and whether the ``n = 2`` part is turned back rather than escaping, are
the nonlinear run's. Fails if the ``n = 3`` part is turned back or the ``n = 2`` part escapes.
"""

import jax.numpy as jnp
import numpy as np

from arrangements.attention import signal
from arrangements.coalgebra import damped, phase
from arrangements.collective import (
    EXT, collective, frozen_part, parent_state, position, run, sent_onward, total_cost, trace,
)

from tests.test_rendezvous import run_phi

TAU, MU, SIG = 0.8, 0.5, 1.0     # as in tests/test_dispersal.py


def distance(k):
    return float(np.sqrt(-2 * TAU * np.log(k)))


def build(positions, nu, xi0, damping=None):
    """``n`` frozen parts on the spatial parent, whose integrator is the phase coalgebra
    (or the damped one) started with momentum ``xi0`` on part 0's placement."""
    n = len(positions)
    xi = jnp.zeros(2 * n).at[0].set(xi0)          # (p_1..p_n, V_1..V_n); V frozen

    def integrator(Q, init):
        if damping is None:
            return phase(Q, init=(init, xi))
        return damped(Q, damping, init=(init, xi))

    return collective(
        [frozen_part(SIG, 1.0, f"C{j}") for j in range(n)],
        tau=TAU, cost=total_cost(n, TAU, nu), positions=positions,
        integrator=integrator, mobility=MU, value_mobility=0.0)


def placements(C, state):
    return np.asarray(parent_state(C, state)[0][:C.n])


def momenta(state, n):
    return parent_state(None, state)[1][:n]


def onward(C, state):
    return float(signal(sent_onward(C, position(C, state)))[0])


K_STAR = 0.5
P_STAR = distance(K_STAR)


def test_0_the_run_is_the_dynamics_functors():
    """LICENSES ``collective.run`` for the phase parameter; establishes nothing else."""
    C = build([P_STAR, P_STAR, 2.0], 1.5, 0.3)
    via_phi = run_phi(C.arr, 6, EXT)[-1]
    s = run(C, 6)
    ours = np.concatenate([np.ravel(x) for x in (parent_state(C, s)[0], parent_state(C, s)[1])])
    np.testing.assert_allclose(ours, via_phi, atol=1e-14)


def test_5_undamped_the_springs_share_returns_and_the_rest_never_does():
    xi0, T = 1e-3, 400
    rows = []
    for n in (1, 2, 3, 5):
        C = build([P_STAR] * n, n * K_STAR * SIG, xi0)
        _, xis = trace(C, T, lambda s: momenta(s, n))
        avg = np.asarray(xis).mean(axis=0)
        free = np.array([xi0 * (1 - 1 / n)] + [-xi0 / n] * (n - 1))
        rows.append((n, avg / xi0, free / xi0))
        if n == 1:
            assert abs(avg[0]) < 0.05 * xi0, avg
        else:
            np.testing.assert_allclose(avg, free, rtol=0.05)
    print("\n  n  time-averaged momentum / xi_0     free share / xi_0    energy kept out = 1 - 1/n")
    for n, a, f in rows:
        print(f"  {n}  {np.round(a, 4)}  {np.round(f, 4)}  {1 - 1 / n:.3f}")


def test_6_damped_a_push_lands_the_configuration_somewhere_new():
    xi0, c = 0.01, 0.1
    coast = MU * xi0 / c
    rows = []
    for n in (1, 2, 3, 5):
        nu = n * K_STAR * SIG
        C = build([P_STAR] * n, nu, xi0, damping=c)
        s = run(C, 20000)
        moved = placements(C, s) - P_STAR
        want = np.array([coast * (1 - 1 / n)] + [-coast / n] * (n - 1))
        rows.append((n, moved, want))
        assert abs(onward(C, s) - nu) < 1e-10
        if n == 1:
            assert abs(moved[0]) < 1e-8, moved
        else:
            np.testing.assert_allclose(moved, want, rtol=0.05)
    print("\n  n  where the parts came to rest (minus p*)     predicted")
    for n, m, w in rows:
        print(f"  {n}  {np.round(m, 5)}  {np.round(w, 5)}")


def test_7_undamped_redundancy_decides_whether_a_small_push_loses_a_part():
    nu, xi0 = 1.5, 0.05
    C3 = build([distance(nu / 3)] * 3, nu, xi0)
    _, p3 = trace(C3, 5000, lambda s: parent_state(C3, s)[0][0])
    p3 = np.asarray(p3)
    C2 = build([distance(nu / 2)] * 2, nu, xi0)
    _, p2 = trace(C2, 100000, lambda s: parent_state(C2, s)[0][0])
    p2 = np.asarray(p2)
    print(f"\n  n=3: pushed part at {p3[-1]:.2f} after 5000 steps (first past 5 at step "
          f"{int(np.argmax(p3 > 5)) + 1 if (p3 > 5).any() else None}); "
          f"n=2: farthest in 100000 steps {p2.max():.4f}")
    assert (p3 > 5).any() and p3[-1] > 5, p3[-1]
    assert p2.max() < 1.2, p2.max()
