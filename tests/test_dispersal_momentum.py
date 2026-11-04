"""A push carries off a part the others can do without (book Chapter 1, ``sec.check_claims``: "a part the others can do
without is barely held onto: a push carries it off for good").

For a reader of the book. The collective is ``arrangements/collective.py``: ``n`` parts, each sending the same signal ``1``, placed on a line
under a parent. A part counts for ``kappa(p) = exp(-p^2 / 2 tau)`` by its placement ``p``, so one placed far away is out
of reach, and the parent's cost is the gap between the total it sends onward and what the collective needs, ``nu``:
``(nu - sum_j kappa_j)^2 / 2``. Only the placements move. Here each placement also carries a momentum, and a part is pushed by starting it
with momentum outward. With need ``nu = 1.5`` and three parts, any two can supply it, so the third can be done without;
with two parts the other sends at most 1, so it cannot. Run from the library's folder with it on the path (``PYTHONPATH=. .venv/bin/python example.py``):

    import jax; jax.config.update("jax_enable_x64", True)
    import numpy as np
    from tests.test_dispersal_momentum import build, distance, parent_state, trace
    nu, push = 1.5, 0.05
    for n in (3, 2):
        C = build([distance(nu / n)] * n, nu, push)
        _, p = trace(C, 5000, lambda s: parent_state(C, s)[0][0])
        print(n, round(float(p[-1]), 3), round(float(np.max(np.asarray(p))), 3))   # n, where it is, farthest it got
    # 3 109.075 109.075
    # 2 -0.682 1.09

Of three, the pushed part coasts away for good. Of two, it is turned back, and in 100,000 steps gets no farther than 1.2.

Why: the cost sees only the total, so near a common rest it is a spring in one direction, the total, and free in the other
``n - 1``. A push splits accordingly. The spring's share, ``1/n``, comes back; the rest does not.

What the tests check. Test 5, with no damping: time-averaged momenta are the free shares, ``1 - 1/n`` of the push's energy
never returning. Test 6, damped: the configuration comes to rest somewhere new, with the onward signal back at ``nu``.
Test 7: the runs above.

For agents: the file's earlier head, with the predictions fixed before the code, what happened and the
weirdness checks, is in ``notes/P31-B-CH1-HISTORY-2026-11-03.md``.
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
