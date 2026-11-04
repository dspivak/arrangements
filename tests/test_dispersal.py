"""A part the collective needs is pulled back by that need (book Chapter 1, ``sec.check_claims``: "without it, the others
send on less than the collective's cost asks for, and only bringing the part back makes up the difference, so that is
where the collective settles").

For a reader of the book. The collective is ``arrangements/collective.py``: ``n`` parts, each sending the same signal ``1``, placed on a line
under a parent. A part counts for ``kappa(p) = exp(-p^2 / 2 tau)`` by its placement ``p``, so one placed far away is out
of reach, and the parent's cost is the gap between the total it sends onward and what the collective needs, ``nu``:
``(nu - sum_j kappa_j)^2 / 2``. Only the placements move. A part is displaced by starting it displaced, at distance 3, far out of reach. With
need 1.5, the one other part of a pair sends at most 1, so the cost stays unmet and pulls the far part back; two others
can cover the need, so nothing asks for the far part, and it stays out. Run from the library's folder with it on the path (``PYTHONPATH=. .venv/bin/python example.py``):

    import jax; jax.config.update("jax_enable_x64", True)
    from arrangements.collective import split
    from tests.test_dispersal import build, distance, onward, settled
    for n, steps in ((2, 20000), (3, 100000)):
        C = build([3.0] + [distance(1.5 / n)] * (n - 1), 1.5)
        q = settled(C, steps)
        print(n, round(float(split(C, q)[1][0]), 3), round(onward(C, q), 6))   # n, far part's placement, signal sent onward
    # 2 1.053 1.5
    # 3 2.992 1.5

Either way the collective sends on what it needs. In the pair, that took bringing the part back.

The pull is weak at a distance: it is the unmet need times the slope of ``kappa``, about .007 at distance 3, so a standing
push of that order would hold the part out (``notes/COMPOSITION-AUDIT.md``, entry 8).

What the tests check. Test 1: the signal returns to ``nu`` whenever the others can supply it. Test 2: a small displacement
is recovered only in the fraction ``1/n``, since the cost sees only the total, and the configuration keeps the rest. Test
3: the runs above. Test 4: a whole collective displaced comes back, in a time set by the distance, growing like
``exp(d^2 / 2 tau)``, and not by the number of parts.

Nothing here drives a part out; every run starts displaced. ``tests/test_dispersal_momentum.py`` is the run in which a
pushed part leaves by the motion itself.

For agents: the file's earlier head, with the predictions fixed before the code, what happened and the
weirdness checks, is in ``notes/P31-B-CH1-HISTORY-2026-11-03.md``.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.attention import signal
from arrangements.coalgebra import configuration
from arrangements.collective import (
    EXT, collective, frozen_part, kappas, position, run, sent_onward, split, total_cost, trace,
)

from tests.test_rendezvous import run_phi

TAU = 0.8          # the kernel scale, as in tests/test_turnover.py
MU = 0.5           # the placements' mobility, as there
SIG = 1.0          # what each part signals, at weight 1


def kappa(p):
    return float(np.exp(-p ** 2 / (2 * TAU)))


def distance(k):
    """The placement at which the kernel is ``k``."""
    return float(np.sqrt(-2 * TAU * np.log(k)))


def build(positions, nu):
    """``n`` frozen parts sending ``(SIG, 1)`` on a spatial parent with the total cost."""
    n = len(positions)
    return collective(
        [frozen_part(SIG, 1.0, f"C{j}") for j in range(n)],
        tau=TAU, cost=total_cost(n, TAU, nu), positions=positions,
        integrator=lambda Q, init: configuration(Q, init=init),
        mobility=MU, value_mobility=0.0)


def settled(C, steps):
    return np.asarray(position(C, run(C, steps)))


def onward(C, q):
    return float(signal(sent_onward(C, q))[0])


# ---------------------------------------------------------------------------
# Test 0: the runs are Phi's.
# ---------------------------------------------------------------------------


def test_0_the_run_is_the_dynamics_functors():
    """LICENSES ``collective.run``; establishes nothing about the predictions."""
    C = build([1.0, 3.0, -0.5], nu=1.5)
    via_phi = run_phi(C.arr, 6, EXT)[-1]
    np.testing.assert_allclose(settled(C, 6), via_phi, atol=1e-14)


# ---------------------------------------------------------------------------
# Predictions (1) and (2): a small displacement of one part.
# ---------------------------------------------------------------------------

K_STAR = 0.5
P_STAR = distance(K_STAR)
DELTA = 0.05


def test_1_2_the_message_returns_and_the_configuration_keeps_all_but_one_nth():
    rows = []
    for n in (1, 2, 3, 5, 8):
        nu = n * K_STAR * SIG
        C = build([P_STAR + DELTA] + [P_STAR] * (n - 1), nu)
        q1, q2 = settled(C, 20000), settled(C, 40000)
        _, pos, _ = split(C, q2)
        pos = np.asarray(pos)
        recovered = (DELTA - (pos[0] - P_STAR)) / DELTA
        others = P_STAR - pos[1:]
        rows.append((n, recovered, others))
        assert abs(onward(C, q2) - nu) < 1e-10                       # (1)
        assert abs(n * recovered - 1.0) < 0.05, (n, recovered)       # (2)
        np.testing.assert_allclose(others, DELTA / n, rtol=0.05)
        assert np.max(np.abs(q2 - q1)) < 1e-12                       # nothing further
    print("\n  n  recovered by the displaced part  (1/n)  others moved in")
    for n, r, o in rows:
        print(f"  {n}  {r:.4f}  ({1 / n:.4f})  {np.round(o, 5)}")


# ---------------------------------------------------------------------------
# Prediction (3): a far part is called back only if the rest cannot cover it.
# ---------------------------------------------------------------------------

NU_3, D_FAR = 1.5, 3.0


def test_3_a_far_part_is_called_back_only_when_the_rest_cannot_cover_it():
    # n = 2: the other part sends at most SIG = 1 < 1.5.
    C2 = build([D_FAR, distance(NU_3 / 2)], NU_3)
    q2 = settled(C2, 20000)
    k2 = np.asarray(kappas(C2, q2))
    assert abs(onward(C2, q2) - NU_3) < 1e-8
    assert k2[0] > 0.45, k2                      # back inside the relations
    # n = 3: the other two can send 1.5 between them.
    C3 = build([D_FAR] + [distance(NU_3 / 3)] * 2, NU_3)
    q3 = settled(C3, 100000)
    _, pos3, _ = split(C3, q3)
    assert abs(onward(C3, q3) - NU_3) < 1e-8
    assert float(pos3[0]) > 0.9 * D_FAR, pos3    # still out
    print(f"\n  n=2: far part's kernel {k2[0]:.4f}; n=3: far part at {float(pos3[0]):.4f} "
          f"(started {D_FAR}), kernel {kappa(float(pos3[0])):.2e}")


# ---------------------------------------------------------------------------
# Prediction (4): the whole collective displaced.
# ---------------------------------------------------------------------------

NU_4, P_NEAR = 0.9, 1.0


def return_time(n, d, steps):
    C = build([d] * n, NU_4)
    _, sig = trace(C, steps, lambda s: signal(sent_onward(C, position(C, s)))[0])
    hit = np.nonzero(np.asarray(sig) >= 0.99 * NU_4)[0]
    return int(hit[0]) + 1 if hit.size else None


def quadrature(d):
    """``(tau / mu nu) int_{P_NEAR}^{d} e^{p^2 / 2 tau} / p dp``: the lone far part's time."""
    p = np.linspace(P_NEAR, d, 200001)
    return TAU / (MU * NU_4) * np.trapezoid(np.exp(p ** 2 / (2 * TAU)) / p, p)


def test_4_total_dispersal_returns_at_a_rate_set_by_distance_not_by_count():
    rows = []
    for d in (3.0, 3.5, 4.0):
        times = {n: return_time(n, d, 6000) for n in (1, 2, 4)}
        rows.append((d, times, quadrature(d)))
    print("\n  d    T(n=1)  T(n=2)  T(n=4)  quadrature")
    for d, t, qd in rows:
        print(f"  {d}  {t[1]}  {t[2]}  {t[4]}  {qd:.1f}")
    d, t, qd = rows[-1]
    assert all(v is not None for v in t.values()), t
    assert abs(t[1] / qd - 1) < 0.15, (t, qd)
    assert max(t.values()) / min(t.values()) < 1.15, t
