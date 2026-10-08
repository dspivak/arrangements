"""Dispersal: can a collective's parts be driven out of the relations, and do they come back?

The book plan's P17 (AGENTDOCPLAN.md §8.2 in the book repository), David, 2026-09-26: "in
a ratchet, i want the click to be the sound of energy spreading irrecoverably through the
handle of the ratchet. is that testable? if so would it handle the destruction idea?", and
§1.4's "the destruction of a building is just a dis-integration": no matter disappears, it
stops being in the relations. Here a part is in the relations while the parent's kernel
reaches it, and the collective is ``arrangements.collective``: ``n`` frozen contributors,
each sending ``(SIG, WGT)``, on a spatial parent whose cost is on the total it sends onward,
``(1/2)(nu - h~)^2`` with ``h~ = SIG sum_j kappa_j``, and whose only mobile coordinates are
the placements. A part is displaced by starting it displaced; nothing else is done to it.

**The predictions, fixed in advance and written here before the collective module or any
construction existed. Each must be able to fail.**

(1) *The collective's message returns.* After one part is displaced, small or far, the
onward signal returns to ``nu`` whenever the other parts can supply it. Fails if descent
stalls short of ``nu`` with the others able to supply it.

(2) *Its configuration does not, and what stays out is shared evenly.* With all parts at one
equilibrium distance, a small outward displacement ``delta`` of one part is recovered by
that part only in the fraction ``1/n``; each other part moves ``delta/n`` inward, and nothing
further happens however long the run. This is a theorem in the linear regime, not a
discovery: the cost sees only the total, its minima form an ``(n-1)``-dimensional family,
and descent from a displaced point moves along the single direction the cost sees,
``(kappa'_1, ..., kappa'_n)``, which is uniform here, so it removes the displacement's
projection on that direction, ``delta/n`` in each coordinate, and never touches the rest.
What the run checks is that the nonlinear kernel and the discrete step keep it (fraction
within 5% of ``1/n`` at ``delta = 0.05``), and it fails if they do not. It refutes the plan's
earlier guess that return *time* grows with the number of parts: for ``n >= 2`` the
configuration never returns at all, and for ``n = 1`` it returns completely. So the
irreversibility here is not slowness; it is displacement in directions the cost cannot see.

(3) *Whether a far part is called back is decided by whether the rest can cover it.* With
``nu = 1.5`` and a part displaced to ``d = 3`` (kernel ``~ 0.004``): for ``n = 2`` the other
part alone sends at most ``SIG = 1 < nu``, a residual persists, and the far part is pulled
all the way back; for ``n = 3`` the other two can send ``1.5``, the residual vanishes, and
the far part stays out, within 10% of ``d``, for ``10^5`` steps. Fails if the ``n = 3`` part
returns or the ``n = 2`` part does not.

(4) *Total dispersal comes back, at a rate set by distance and not by the number of parts.*
All ``n`` parts displaced together to ``d``: the onward signal falls to ``n kappa(d)``, near
zero, and returns to within 1% of ``nu``. Far out ``kappa`` is small and each part moves as a
lone part would, ``dp/dt = -(mu nu / tau) p kappa(p)``, so the return time is the quadrature
``(tau / mu nu) int e^{p^2 / 2 tau} / p dp``, which grows like ``e^{d^2 / 2 tau}``. The
measured return time agrees with that quadrature within 15% at the largest ``d``, and the
times for ``n = 1, 2, 4`` agree with each other within 15% there. Fails if the time depends
on ``n`` at large ``d``, or does not track the quadrature.

If (2)-(4) hold, the answer to P17 in this model is: a part started far from a collective that
can do without it stays out for good, deterministically, and the configuration keeps the
displacement the cost cannot see; a collective started far out as a whole comes back as long
as the need persists, at a time exponential in distance squared. Nothing here drives a part
out: every run starts from the displaced state (``test_dispersal_momentum.py`` is the run in
which a pushed part coasts out by the dynamics). Irrecoverable dispersal needs
redundancy, not noise.
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
