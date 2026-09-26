"""Seeing and pulling together: a monitor, redundancy, and a spare in one arrangement.

The final stage of David's monitor test, 2026-09-25, from his question: "there can't be a
monitor who is able to see dead parts and also able to pull from far? maybe that's
something else to test separately and then we can put them together and test that too?"
Seeing is ``tests/test_monitor_seeing.py`` and pulling is ``tests/test_monitor_pulling.py``;
this file **puts them together**. Success is the turnover test's criterion: the
collective's outgoing message returns to its pre-failure value.

David also described a fuller loop: "if i call out 'i need X', then something should fall
into the X basin, and then we should start crafting the plan to go get X and actually get
X and bring it back. can that be made?" Four pieces -- the **call**, **falling into the
basin**, **running a plan to go and get X**, and **bringing it back**. This stage is
scoped to call, basin and delivery; plan-execution is not built here and is not a piece
this construction contains. That scoping is the agent's, not David's.

**The four predictions, fixed in advance and written here before any construction code
existed. Each must be able to fail. None is revised after a number was seen.**

**(C1) The composite restores the whole outgoing message, not only the signal.** The
turnover run found the signal returns exactly while the total weight stays raised when the
failed part keeps reporting; the seeing run found a monitor holding an expectation of the
total weight drives the dead part's kernel down until the expected weight arrives. So with
both a monitor and a spare, both the signal ``h~`` and the weight ``w~`` return to ``nu``.
*Able to fail:* the monitor's push on the dead part and the spare's arrival both raise the
total weight, so they may overshoot.

**(C2) Delivery stops itself.** The seeing run found the monitor's forcing is exactly
``-s * grad(w~)``, inert to the last bit iff the surprise ``s`` is zero; the pulling run
found a doodad whose alarm is *held* at a constant overshoots, holding the collective at
``h~ = 2.529935`` instead of ``nu = 2.5``. So an alarm **driven by the surprise** rather
than held should end delivery exactly when the surprise closes, with no stopping rule
added. *Able to fail:* if the overshoot survives, the composition needs a stop the
mathematics does not supply.

**(C3) The loop is one arrangement, and the only stipulation left is the supply.** The
*call* is the monitor's nonzero surprise; the *basin* is the spare descending into kernel
reach; the *delivery* is the doodad's pull. *Able to fail:* if the composite needs
anything further -- a schedule, a hand-set threshold, a switch -- then it is a controller
wrapped around an arrangement, not an arrangement.

**(C4) The monitor is idle where it is not needed.** A silent failure is already repaired
without a monitor (the turnover run: in the silent mode the whole message returns to
``(2.5, 2.5)``), and the seeing run found the monitor cannot move a silent part, which
carries no weight. So adding the monitor to the silent case should change nothing. *Able
to fail:* the seeing run found even a **free** monitor's five-step transient leaves the
collective at a different resting place, so the monitor may spoil a case it cannot help.

The three prior stages are imported rather than copied, so the collective here is
literally the one run there: ``tests/test_turnover.py`` supplies the contributors, the
spatial parent and its cost on the weighted signal; ``tests/test_monitor_seeing.py``
supplies the monitor, one more child of the same parent, frozen at the origin, holding a
Gaussian prior on the total arriving weight; ``tests/test_monitor_pulling.py``'s route (b)
supplies the doodad's pull law, a bounded ``tanh`` ramp on the spare's published placement
with nothing returning. The one thing new here is that the doodad's alarm is not a held
constant but the monitor's own surprise.

Every docstring below says which of three things its test does (STATUS.md §3.7): reproduce
a closed form derived in the docstring, check a discrete analogue, or correct a claim made
on paper. The numbers, and which verdicts the parameters decide rather than discover:
``notes/MONITOR-COMPOSED.md``.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement
from arrangements.attention import signal, weight
from arrangements.wiring import compose_seq, trivial_parameter

from tests.chain import flat_state
from tests.test_rendezvous import run_closed, run_phi
from tests.test_turnover import (
    EDIM, EXT, MDIM, MU_P, NU, TAU, collective, equilibrium_position, healthy, kappa_at,
    kappas, sent_onward,
)
from tests.test_monitor_seeing import PI0, THETA, heard, held, watched

EMPTY = jnp.zeros(0)

# the composite configuration of ``watched([C0, C1, C_F, SPARE], ...)``:
#   0..3  the four contributors' healths, 4 the monitor's expectation pi,
#   5..9  the five placements (C0, C1, C_F, SPARE, MON), 10..14 the value maps.
FAIL, SPARE = 2, 3          # which contributor is dead, and which is the spare
MON_PI = 4                  # the monitor's expectation of the total arriving weight
P_SPARE = 8                 # the spare's placement in the composite configuration

ALPHA, WID = 0.02, 0.25     # the doodad's pull: strength, and the width of its ramp
A_HELD = 1.0                # the pulling test's held alarm, for the comparison in test 3
W_PI = 1.0                  # the monitor's prior precision, the seeing test's headline
D_NEAR, D_FAR = 2.0, 20.0   # a spare inside the parent's own reach, and one far outside
STEPS = 8000
BUDGET = 40000              # the budget every reach number here is read at
SETTLED = 1e-9


# ---------------------------------------------------------------------------
# The composite: the watched collective, publishing two more coordinates, pulled.
# ---------------------------------------------------------------------------


def publishing(b, idxs):
    """The same arrangement, additionally publishing configuration coordinates ``idxs``
    at its port (``tests/test_monitor_pulling.py``'s wrapper, widened to several). Only
    ``out_f`` changes: the forcing is untouched, so every non-exactness measured below is
    the coupler's and not an artefact of the wrapper."""
    take = jnp.asarray(idxs)
    return replace(b, out_dim_N=b.out_dim_N + len(idxs),
                   out_f=lambda q, l_out: jnp.concatenate([b.out_f(q, l_out),
                                                           jnp.take(q, take)]))


def pull_F(alpha=ALPHA, wid=WID, a_held=None):
    """The returned covector of the recruiting coupler on the ports
    ``e = (h~, w~, p, pi)``: the collective's outgoing message, the spare's placement, and
    the monitor's expectation.

        alarm   a(e) = pi - w~      = -s, the monitor's surprise, sign-flipped
        F_p(e)  = -alpha a tanh(p / wid),   F_h~ = F_w~ = F_pi = 0.

    The pulling test's law with one change: its alarm was a configuration coordinate of
    its own held at ``a0`` by a restoring cost, and here the alarm **is the surprise**.
    Nothing returns to the monitor -- ``F_pi = 0`` -- which is that test's stipulation
    kept, and by the reciprocity lemma (§5.5) is what makes the coupler non-exact.

    ``a_held`` replaces the driven alarm by a constant, which is the pulling test's doodad
    wired into this composite; test 3 runs the two side by side.
    """

    def F(e):
        h, w, p, pi = e[0], e[1], e[2], e[3]
        a = (0.0 * pi + a_held) if a_held is not None else (pi - w)
        return jnp.array([0.0 * h, 0.0 * w, -alpha * a * jnp.tanh(p / wid), 0.0 * pi])

    return F


def coupler(F, ports):
    """A stateless coupler whose forcing on its ``ports`` is the returned covector ``F``
    and whose ``in_f`` closes the collective on the exterior input ``EXT``."""
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=EDIM, out_dim_M=ports, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, e: EMPTY,
        in_f=lambda q, e, n_in: EXT,
        omega=lambda q, e, n_in: (EMPTY, F(e), EMPTY),
        label="pull")


def specs_for(d, mode, nu=NU):
    """The failed state with a spare at ``d``: two contributors in full health at ``+-p``,
    a dead one at ``+p``, a spare in full health at ``d``, with ``p`` chosen so that
    *before* the failure the collective sent exactly ``nu`` -- the turnover test's device
    for starting at the pre-failure fixed point. The failure is a **state** here, the dead
    part's health held at ``0`` and immobile, as in ``tests/test_monitor_seeing.py``."""
    p = equilibrium_position(3, kappa_at(d), nu)
    return [healthy(p), healthy(-p), held(0.0, mode, pos=p), healthy(d)], p


def watched_collective(d, mode, w_pi=W_PI, healthy_fail=False):
    """The turnover collective with a spare at ``d`` and the seeing test's monitor added
    as one more child of the same parent, at ``p = 0`` and frozen."""
    specs, p = specs_for(d, mode)
    if healthy_fail:
        specs[FAIL] = healthy(p, mode)
    return watched(specs, w_pi, "total", mu_pos=MU_P) + (p,)


def bare_collective(d, mode):
    """The same collective with no monitor: the turnover test's four-child composite."""
    specs, p = specs_for(d, mode)
    parts, a, b = collective(specs, "total", mu_pos=MU_P)
    return parts, a, b, p


def looped(d, mode, w_pi=W_PI, a_held=None, alpha=ALPHA, healthy_fail=False):
    """The whole loop as one closed arrangement:

        compose_seq(watched collective publishing (p_spare, pi), coupler(F))

    Its configuration is the watched collective's, the coupler being stateless, so the
    indices above read it unchanged.
    """
    all_parts, a, b, p = watched_collective(d, mode, w_pi, healthy_fail)
    F = pull_F(alpha, a_held=a_held)
    return all_parts, a, compose_seq(publishing(b, (P_SPARE, MON_PI)),
                                     coupler(F, MDIM + 2)), p, F


# ---------------------------------------------------------------------------
# Reading a run.
# ---------------------------------------------------------------------------


def traj(b, steps, ext=EMPTY):
    """The whole trajectory of a closed composite, as ``run_closed`` computes it but
    inside a ``lax.scan`` (test 0 checks the two agree)."""
    P, omega = b.P, b.omega

    @jax.jit
    def step(s, _):
        xi_q, _, _ = omega(P.position(s), EMPTY, ext)
        s2 = P.step(s, xi_q)
        return s2, flat_state(s2)

    return np.asarray(jax.lax.scan(step, P.init, None, length=steps)[1])


def final(b, steps, ext=EMPTY):
    """The configuration after ``steps``, without keeping the trajectory."""
    P, omega = b.P, b.omega

    def step(_i, s):
        xi_q, _, _ = omega(P.position(s), EMPTY, ext)
        return P.step(s, xi_q)

    return np.asarray(flat_state(jax.lax.fori_loop(0, steps, step, P.init)))


def msg(all_parts, a, q):
    """``(h~, w~)``, the message the collective sends onward at configuration ``q``."""
    m = sent_onward(all_parts, a, jnp.asarray(q, float))
    return float(signal(m)[0]), float(weight(m))


def surprise(all_parts, a, q):
    """``s = w(heard) - pi``: the monitor's prediction error on the total arriving
    weight, which is the whole of the force its error term exerts on ``pi``."""
    return float(weight(heard(all_parts, a, q)) - jnp.asarray(q, float)[MON_PI])


def settled(all_parts, a, b, steps, extra=500, ext=EMPTY):
    """``(h~, w~, surprise, kappas, drift, q)`` at ``steps``, the drift over ``extra``
    more. Settling is read off the *motion*, as the turnover test reads it: a frozen
    coordinate carries a nonzero forcing forever without moving."""
    q1, q2 = final(b, steps, ext), final(b, steps + extra, ext)
    h, w = msg(all_parts, a, q2)
    return h, w, surprise(all_parts, a, q2), kappas(q2, len(all_parts)), \
        float(np.max(np.abs(q2 - q1))), q2


VALUES = [1.0, 1.0, 0.0, 1.0, 0.0]                 # what each child signals
WEIGHTS = {"reporting": [1.0, 1.0, 1.0, 1.0, 0.0],  # ... and the weight it claims
           "silent": [1.0, 1.0, 0.0, 1.0, 0.0]}


def sent(rows, mode, values=None):
    """``(h~, w~)`` along a whole trajectory, in numpy: ``sum_j kappa(p_j) v_j`` and
    ``sum_j kappa(p_j) w_j``, which is ``sigma_a`` when every ``V_j`` is the identity.
    Test 1 licenses this against the library; it is used instead of the library call so
    that a whole run's worth of placements can be read at once."""
    rows = np.atleast_2d(rows)
    kap = np.exp(-rows[:, 5:10] ** 2 / (2.0 * TAU))
    v = np.array(VALUES if values is None else values)
    w = np.array(WEIGHTS[mode]) if values is None else np.array(values)
    return kap @ v, kap @ w


def arrival(rows, mode, target=NU, tol=1e-6):
    """The step at which the outgoing *signal* first reaches ``target``, or ``None``."""
    hit = np.nonzero(sent(rows, mode)[0] > target - tol)[0]
    return (int(hit[0]) + 1) if len(hit) else None


def d_omega(F):
    """``(d omega)_{ij} = d_i F_j - d_j F_i``, the antisymmetric part of the
    returned-covector Jacobian (the reciprocity lemma, §5.5), exactly as
    ``tests/test_mutual_need.py`` and ``tests/test_monitor_pulling.py`` compute it."""

    def at(e):
        J = jax.jacobian(F)(jnp.asarray(e, float))       # J[j, i] = d F_j / d e_i
        return np.asarray(J.T - J)

    return at


def attention_force(rows, mode):
    """The parent's *own* inward force on the spare along a trajectory,
    ``(nu - h~) v_spare dkappa/dp = -(nu - h~)(p/tau) kappa(p)``, in numpy. Compared in
    test 4 against the doodad's pull, to locate where the parent takes the spare over."""
    p = np.atleast_2d(rows)[:, P_SPARE]
    return -(NU - sent(rows, mode)[0]) * (p / TAU) * np.exp(-p ** 2 / (2.0 * TAU))


def pull_force(rows, mode, a_held=None):
    """The doodad's pull along a trajectory, ``-alpha a tanh(p/wid)``, in numpy."""
    rows = np.atleast_2d(rows)
    p, pi = rows[:, P_SPARE], rows[:, MON_PI]
    a = np.full(len(rows), a_held, float) if a_held is not None else pi - sent(rows, mode)[1]
    return -ALPHA * a * np.tanh(p / WID)


P_REST = float(np.sqrt(2.0 * TAU * np.log(2.0)))   # = 1.053108, where a spare comes to rest
DELTA = 0.5        # nu - w~ while the live pair is home and the spare is still out


# ---------------------------------------------------------------------------
# Test 0: the runs are Phi's.
# ---------------------------------------------------------------------------


def test_0_the_runs_are_the_dynamics_functors():
    """LICENSES the run forms used below; it establishes nothing about the predictions.
    For the loop and for the watched collective without the doodad, ``Phi``, ``run_closed``
    and the ``lax.scan`` of ``traj`` agree over fifty steps -- to ``5.6e-17``, and the scan
    against ``run_closed`` bit for bit -- so every trajectory here is ``Phi``'s."""
    _, _, loop, _, _ = looped(D_FAR, "silent")
    _, _, watched_only, _ = watched_collective(D_FAR, "silent")
    for b, ext in ((loop, EMPTY), (watched_only, EXT)):
        by_phi = run_phi(b, 50, ext)
        by_closed = run_closed(b, 50, lambda t: ext)
        assert float(np.max(np.abs(by_phi - by_closed))) < 1e-15
        np.testing.assert_array_equal(traj(b, 50, ext)[-1], by_closed[-1])


# ---------------------------------------------------------------------------
# Test 1: the baseline, and what the four ports carry.
# ---------------------------------------------------------------------------


def test_1_the_baseline_is_a_fixed_point_and_the_ports_are_what_they_say():
    """REPRODUCES the closed forms derived here for the ports and for the pre-failure
    fixed point, and LICENSES the numpy reader ``sent``.

    The collective's four children in full health at ``+-p`` and ``d``, with
    ``3 kappa(p) + kappa(d) = nu``, send ``m~ = (nu, nu)``: the signal because that is how
    ``p`` was chosen, the weight because in full health every contributor claims ``WGT``
    and the same kernels apply. So the parent's cost is at its minimum, the monitor's
    expectation ``pi_0 = nu`` is met, its surprise is ``0`` to the last bit, and therefore
    the alarm ``a = pi - w~`` is ``0`` and the doodad's pull ``-alpha a tanh(p/wid)`` is
    ``0``. Every coordinate of the composite forcing vanishes and 200 steps move nothing.

    **This is the first half of (C3).** The loop has no switch to be off: it is quiescent
    because the forcing it is made of vanishes where nothing is wrong, and the same
    expression that vanishes here is what pulls in test 3.

    The ports the coupler reads are ``(h~, w~, p_spare, pi)`` -- the collective's own
    outgoing message, which ``publishing`` leaves untouched, followed by the two published
    configuration coordinates."""
    all_parts, a, b, _, F = looped(D_NEAR, "reporting", healthy_fail=True)
    q0 = jnp.asarray(flat_state(b.P.init))

    np.testing.assert_allclose(msg(all_parts, a, q0), [NU, NU], atol=1e-14)
    assert surprise(all_parts, a, q0) == 0.0

    # the ports, and the covector returned on them
    _, _, b_open, _ = watched_collective(D_NEAR, "reporting", healthy_fail=True)
    e = publishing(b_open, (P_SPARE, MON_PI)).out_f(q0, EMPTY)
    np.testing.assert_allclose(np.asarray(e), [NU, NU, D_NEAR, PI0], atol=1e-14)
    np.testing.assert_array_equal(np.asarray(F(e)), np.zeros(4))

    xi, _, _ = b.omega(q0, EMPTY, EMPTY)
    assert float(np.max(np.abs(np.asarray(xi)))) == 0.0
    np.testing.assert_array_equal(final(b, 200), np.asarray(q0))

    # the numpy reader against the library, at the baseline and at the failed state
    np.testing.assert_allclose([x[0] for x in sent(np.asarray(q0), "reporting",
                                                   values=[1.0, 1.0, 1.0, 1.0, 0.0])],
                               msg(all_parts, a, q0), rtol=1e-14)
    for mode in ("reporting", "silent"):
        ps, aa, bb, _, _ = looped(D_FAR, mode)
        q = np.asarray(flat_state(bb.P.init))
        np.testing.assert_allclose([x[0] for x in sent(q, mode)], msg(ps, aa, q), rtol=1e-14)


# ---------------------------------------------------------------------------
# Test 2: (C1) -- does the whole outgoing message come back?
# ---------------------------------------------------------------------------


def test_2_C1_the_whole_message_returns_in_one_mode_and_not_in_the_other():
    """CORRECTS (C1), which holds in the silent mode and fails in the reporting one, and
    REPRODUCES the closed forms for every cell of the table. Twelve settled runs: two
    failure modes, a spare inside the parent's own reach (``d = 2``) and one far outside
    it (``d = 20``), and three arrangements -- the bare collective, the collective with
    the monitor, and the whole loop.

    | mode | d | arrangement | h~ | w~ |
    |---|---|---|---|---|
    | silent | 2 | bare | 2.5 | 2.5 |
    | silent | 2 | + monitor | 2.5 | 2.5 |
    | silent | 2 | loop | 2.5 | 2.5 |
    | silent | 20 | bare | 2.0 | 2.0 |
    | silent | 20 | + monitor | 2.0 | 2.0 (surprise open at -0.25) |
    | silent | 20 | **loop** | **2.5** | **2.5** |
    | reporting | 2 | bare | 2.5 | 3.305972 |
    | reporting | 2 | + monitor | 2.498156 | 2.503688 (still moving) |
    | reporting | 2 | loop | 2.498108 | 2.503672 (still moving) |
    | reporting | 20 | bare | 2.0 | 2.833333 |
    | reporting | 20 | + monitor | 2.0 | 2.5 |
    | reporting | 20 | loop | 2.0 | 2.5 (the spare pushed *out*, to 20.016) |

    *The silent mode, and the cell the composition is for.* A silent part sends ``0_M``,
    so ``h~ = w~`` identically and the whole message is one number. At ``d = 2`` the bare
    collective already restores it -- the turnover run's result, and (C4)'s premise -- and
    neither the monitor nor the doodad is needed. At ``d = 20`` the parent's own inward
    force on the spare is ``-(nu - h~)(d/tau) kappa(d) ~ 1e-108``, so the bare collective
    sits at ``2.0`` forever and the monitor alone sits there with its surprise open at
    ``-0.25``, seeing the drop and unable to close it. **The loop closes it exactly**,
    ``h~ = w~ = 2.5`` with drift ``0`` and the spare at rest at
    ``p = sqrt(2 tau log 2) = 1.053108``. So (C1) holds in the silent mode, and the
    composition earns its keep in exactly the cell where neither half alone does.

    *The reporting mode, where the monitor's own repair is the whole story.* A reporting
    part keeps claiming ``WGT`` at a zero signal, so ``w~ - h~ = kappa_F``: the two
    coordinates of the message can both equal ``nu`` only if ``kappa_F = 0``, which no
    finite placement gives. At ``d = 20`` the monitor pushes the dead part out until
    ``w~ = nu`` exactly (``kappa_F = 0.5``, the seeing run's repair) and the signal stays
    at ``2.0``: **(C1) is refuted there** -- the weight returns and the signal does not,
    which is the exact opposite of the turnover run's reporting failure. At ``d = 2`` both
    coordinates do approach ``nu``, but only as the dead part is pushed to infinity, and
    the approach is **algebraic**: ``kappa_F`` and both deficits fall by a factor
    ``2.12--2.17`` for each fourfold in the run length, about ``t^{-0.55}``, with the
    exponent still drifting at 512 000 steps. No budget closes it. What the monitor buys
    in that cell is real but is not what (C1) claimed: it turns the turnover run's
    *permanent* mark on the outgoing weight (``3.305972``, drift ``0``) into a decaying
    one.
    """
    rows = {}
    for mode in ("silent", "reporting"):
        for d, steps in ((D_NEAR, STEPS), (D_FAR, BUDGET)):
            bp, ba, bb, _ = bare_collective(d, mode)
            rows[mode, d, "bare"] = settled(bp, ba, bb, steps, ext=EXT)
            mp, ma, mb, _ = watched_collective(d, mode)
            rows[mode, d, "monitor"] = settled(mp, ma, mb, steps, ext=EXT)
            lp, la, lb, _, _ = looped(d, mode)
            rows[mode, d, "loop"] = settled(lp, la, lb, steps)

    # the silent mode: h~ = w~ identically, and the loop closes the far cell exactly
    for which in ("bare", "monitor", "loop"):
        h, w, s, k, drift = rows["silent", D_NEAR, which][:5]
        np.testing.assert_allclose([h, w], [NU, NU], atol=1e-15)   # already repaired
        assert drift < SETTLED
    for which in ("bare", "monitor"):
        h, w, s, k, drift = rows["silent", D_FAR, which][:5]
        np.testing.assert_allclose([h, w], [2.0, 2.0], atol=1e-9)  # the spare never comes
        assert drift < SETTLED
    np.testing.assert_allclose(rows["silent", D_FAR, "monitor"][2], -0.25, atol=1e-9)
    assert abs(rows["silent", D_FAR, "monitor"][2]) >= THETA       # it sees and cannot act
    h, w, s, k, drift, q = rows["silent", D_FAR, "loop"]
    np.testing.assert_allclose([h, w], [NU, NU], atol=1e-14)       # ... and the loop acts
    assert drift == 0.0 and s == 0.0
    np.testing.assert_allclose([q[P_SPARE], k[SPARE]], [P_REST, 0.5], atol=1e-6)

    # the reporting mode: w~ - h~ = kappa_F, so both cannot be nu at any finite placement
    for d in (D_NEAR, D_FAR):
        for which in ("bare", "monitor", "loop"):
            h, w, s, k, _ = rows["reporting", d, which][:5]
            np.testing.assert_allclose(w - h, k[FAIL], atol=1e-9)

    h_b, w_b, _, k_b, drift_b = rows["reporting", D_NEAR, "bare"][:5]
    np.testing.assert_allclose([h_b, w_b], [NU, 3.305971667], atol=1e-8)
    assert drift_b == 0.0                      # the turnover run's permanent mark
    for which in ("monitor", "loop"):
        h, w, _, k, drift = rows["reporting", D_NEAR, which][:5]
        np.testing.assert_allclose([h, w], [2.4981, 2.5037], atol=1e-4)
        assert drift > SETTLED                 # ... turned into a decaying one
        assert k[FAIL] < 0.01 and k[SPARE] > 0.49

    np.testing.assert_allclose(rows["reporting", D_FAR, "bare"][:2], [2.0, 2.8333333],
                               atol=1e-7)
    for which in ("monitor", "loop"):
        h, w, s, k, drift, q = rows["reporting", D_FAR, which]
        np.testing.assert_allclose([h, w], [2.0, NU], atol=1e-9)   # (C1) refuted here
        np.testing.assert_allclose(k[FAIL], 0.5, atol=1e-9)        # the seeing run's repair
        assert drift < SETTLED and abs(s) < 1e-12
    assert rows["reporting", D_FAR, "loop"][5][P_SPARE] > D_FAR + 1e-3   # pushed away
    assert abs(rows["reporting", D_FAR, "monitor"][5][P_SPARE] - D_FAR) < 1e-12

    # the rate of the reporting near cell, so that "approaches nu" is not a budget
    _, _, lb, _, _ = looped(D_NEAR, "reporting")
    ks = []
    for n in (STEPS, 4 * STEPS, 16 * STEPS, 64 * STEPS):
        q = final(lb, n)
        ks.append(float(kappas(q, 5)[FAIL]))
    ratios = np.array(ks[:-1]) / np.array(ks[1:])
    assert np.all((ratios > 2.11) & (ratios < 2.18))               # algebraic, about t^-0.55
    np.testing.assert_allclose(np.log(ratios) / np.log(4.0), [0.5578, 0.5485, 0.5421],
                               atol=1e-3)


# ---------------------------------------------------------------------------
# Test 3: (C2) -- does delivery stop itself?
# ---------------------------------------------------------------------------


DISTANCES = (5.0, 10.0, 20.0, 50.0)
ARRIVALS = (791, 2791, 6791, 18791)


def test_3_C2_a_driven_alarm_stops_itself_and_a_held_one_never_does():
    """REPRODUCES the closed forms derived here for the arrival law and the resting place,
    and so tests (C2).

    *The driven alarm.* While the live pair is home and the spare is still out, the total
    arriving weight is ``w~ = 2`` and the monitor settles at
    ``pi* = (w~ + w_pi pi_0)/(1 + w_pi)``, so the alarm is constant along the haul at

        a* = pi* - w~ = (w_pi/(1 + w_pi)) Delta,   Delta = nu - w~ = 0.5,

    which at ``w_pi = 1`` is ``0.25``. The pull is then ``alpha a*`` once the spare is past
    the ramp, the placement's mobility is ``MU_P``, so the trip is **linear in the
    distance** at ``1/(MU_P alpha a*) = 400`` steps per unit: the measured arrivals at
    ``d = 5, 10, 20, 50`` are ``791, 2791, 6791, 18791``, successive slopes ``400.0`` per
    unit at ``atol = 0``. Compare route (a) at the same ``tau``, where
    ``notes/TURNOVER-TEST.md`` found a spare at ``d = 5`` needs two million steps and one
    at ``d = 6`` never arrives within that.

    *And it stops.* From every distance the spare comes to rest at the same place,
    ``p = sqrt(2 tau log 2) = 1.053108``, which is where ``kappa = nu - 2`` -- the share
    the dead part gave up, and the resting place ``notes/TURNOVER-TEST.md`` found for a
    spare recruited by attention alone. There the alarm is ``0.0`` and the pull is
    ``-0.0``, both to the last bit: **the delivery ends exactly when the surprise closes,
    and nothing was added to stop it.** (C2) holds.

    *The held alarm*, which is ``tests/test_monitor_pulling.py``'s doodad wired into this
    composite unchanged, never rests with zero force, at any of the three distances run.
    From ``d = 20`` it holds the collective at ``h~ = 2.5200424`` -- above its target, as
    the pulling run found at ``2.529935`` -- with ``-0.01999`` of pull still on the spare
    at settlement, balanced by the parent pushing back. From ``d = 5`` and ``d = 10`` it
    drags the spare the whole way to the parent, ``p ~ 1e-306``, and the collective pays
    by holding its two live contributors out at ``+-0.678450`` instead of at ``0``. So the
    held alarm's resting place depends on where the spare started and the driven alarm's
    does not.
    """
    ends = []
    for d in DISTANCES:
        all_parts, a, b, _, _ = looped(d, "silent")
        rows = traj(b, BUDGET)
        ends.append((arrival(rows, "silent"), rows[-1]))
    assert [e[0] for e in ends] == list(ARRIVALS)
    np.testing.assert_allclose(np.diff(ARRIVALS) / np.diff(DISTANCES),
                               1.0 / (MU_P * ALPHA * (W_PI / (1.0 + W_PI)) * DELTA),
                               atol=0.0)

    for q in (e[1] for e in ends):
        h, w = (float(x[0]) for x in sent(q, "silent"))
        np.testing.assert_allclose([h, w, q[P_SPARE]], [NU, NU, P_REST], atol=1e-6)
        assert float(q[MON_PI]) - w == 0.0                 # the alarm, to the last bit
        assert float(pull_force(q, "silent")[0]) == 0.0    # ... and the pull with it

    # the held alarm, the pulling test's doodad unchanged, at the same three distances
    for d, h_end, p_end in ((5.0, NU, 0.0), (10.0, NU, 0.0), (20.0, 2.5200424048, 1.022816)):
        all_parts, a, b, _, _ = looped(d, "silent", a_held=A_HELD)
        q1, q2 = final(b, BUDGET), final(b, BUDGET + 500)
        assert float(np.max(np.abs(q2 - q1))) < SETTLED
        h, w = msg(all_parts, a, q2)
        np.testing.assert_allclose([h, w], [h_end, h_end], atol=1e-8)
        np.testing.assert_allclose(q2[P_SPARE], p_end, atol=1e-6)
        assert float(pull_force(q2, "silent", a_held=A_HELD)[0]) != 0.0   # never at rest
        if d == 20.0:
            np.testing.assert_allclose(pull_force(q2, "silent", a_held=A_HELD)[0],
                                       -0.0199893, atol=1e-6)
            assert h > NU + 1e-3                            # held past what it asked for
        else:
            np.testing.assert_allclose(np.abs(q2[5:7]), [0.678450, 0.678450], atol=1e-5)


# ---------------------------------------------------------------------------
# Test 4: (C3) -- the call, the basin, the delivery, and what is stipulated.
# ---------------------------------------------------------------------------


W_PIS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def test_4_C3_the_call_the_basin_and_the_one_thing_stipulated():
    """REPRODUCES the closed forms for the call, the delivery rate and the coupler's
    non-exactness, and CORRECTS (C3) in one place: the loop is one arrangement and needs
    no switch, but which failure *sounds* the call is not the agent's to choose.

    *The call is the surprise, and it does not sound for every failure.* At the failed
    state with the spare still out, a **silent** part has taken its weight with it, so
    ``w~ = 2 kappa(p) = 1.6666667`` against ``pi_0 = 2.5``: the surprise is
    ``-kappa(p) = -0.8333333``, past the seeing test's threshold ``theta = 0.05``, and the
    alarm is ``+0.8333333`` with a pull of ``-0.0166667`` already on the spare at step
    ``0``. A **reporting** part keeps claiming ``WGT``, so ``w~ = 2.5`` exactly and the
    surprise is ``0.0`` to the last bit: the alarm is zero, the pull is ``-0.0``, and the
    doodad never goes. **The failure that sounds the call is not the failure the monitor
    can repair**, which is the seeing run's reporting failure, and test 2's table is what
    that costs.

    *The basin is a crossing, not a chosen radius.* From ``d = 20``, the doodad carries
    the spare for 6671 steps while the parent's own inward force on it is below the pull
    -- ``4e-108`` at the start. At step **6671**, placement **3.081955**,
    ``kappa = 0.0026412``, the parent's force overtakes the doodad's, and the last 120
    steps of the trip are the collective's own attention: arrival at 6791. Nothing chose
    that number; it is where two forces cross.

    *The delivery rate is the prior precision.* The haul runs at
    ``1/(MU_P alpha a*)`` steps per unit with ``a* = (w_pi/(1 + w_pi)) Delta``, so the same
    dial the seeing run found sets the *detectable* drop sets the *delivery speed*, and
    the measured slope matches at ``atol = 0`` for every precision run: 1000, 600, 400,
    300 and 250 steps per unit at ``w_pi = 0.25, 0.5, 1, 2, 4``. **At ``w_pi = 0`` the
    spare never arrives at any budget**: a free prediction's settled surprise is zero
    (the seeing run's closed form), so a free monitor has nothing to pull with. This was
    not predicted, and it is the strongest thing in the stage: seeing and pulling are not
    two capacities here that happen to be composed, they are one quantity.

    *What is stipulated.* The coupler returns nothing at the monitor's port, ``F_pi = 0``,
    so ``(d omega)_{pi p} = -alpha tanh(p/wid)`` and ``(d omega)_{w~ p} = +alpha
    tanh(p/wid)`` -- the only nonzero entries, ``-0.02`` and ``+0.02`` once the ramp
    saturates -- and the composite's own antisymmetric part on ``Q`` carries ``-0.02`` at
    the alarm and the spare's placement. By the reciprocity lemma (§5.5) the pull is
    therefore not ``dU`` for any ``U``: inside the book's category, outside the attention
    operad, exactly as in ``tests/test_monitor_pulling.py``. Nothing depletes: the
    delivered work, ``0.0916588`` from ``d = 20``, is finite because the job ends, not
    because anything was charged for it. **The supply is stipulated away here too**, and
    (C3)'s "the only stipulation left is the supply" holds in that sense and in no
    stronger one.
    """
    # the call, at the failed state, before anything has moved
    calls = {}
    for mode in ("silent", "reporting"):
        all_parts, a, b, p, F = looped(D_FAR, mode)
        q0 = np.asarray(flat_state(b.P.init))
        h, w = msg(all_parts, a, q0)
        calls[mode] = (h, w, surprise(all_parts, a, q0), float(q0[MON_PI]) - w,
                       float(pull_force(q0, mode)[0]))
    kap = kappa_at(equilibrium_position(3, kappa_at(D_FAR)))
    np.testing.assert_allclose(calls["silent"][:4], [2.0 * kap, 2.0 * kap, -kap, kap],
                               atol=1e-12)
    assert abs(calls["silent"][2]) >= THETA                  # the call sounds
    np.testing.assert_allclose(calls["silent"][4], -ALPHA * kap * np.tanh(D_FAR / WID),
                               rtol=1e-12)
    np.testing.assert_allclose(calls["reporting"][:2], [2.0 * kap, NU], atol=1e-12)
    assert calls["reporting"][2] == 0.0 and calls["reporting"][3] == 0.0
    assert calls["reporting"][4] == 0.0                      # ... and here it does not

    # the basin: where the parent's own attention overtakes the doodad
    _, _, b, _, _ = looped(D_FAR, "silent")
    rows = traj(b, STEPS)
    att, pull = attention_force(rows, "silent"), pull_force(rows, "silent")
    cross = int(np.nonzero(np.abs(att) > np.abs(pull))[0][0])
    assert cross + 1 == 6671 and arrival(rows, "silent") == 6791
    np.testing.assert_allclose(rows[cross, P_SPARE], 3.081955, atol=1e-5)
    np.testing.assert_allclose(np.exp(-rows[cross, P_SPARE] ** 2 / (2.0 * TAU)),
                               0.00264118, atol=1e-7)
    assert abs(att[0]) < 1e-100                              # at the start: nothing at all

    # the delivery rate against the prior precision, and the free monitor
    for w_pi in W_PIS:
        arrs = []
        for d in (5.0, 10.0):
            _, _, bb, _, _ = looped(d, "silent", w_pi=w_pi)
            arrs.append(arrival(traj(bb, BUDGET), "silent"))
        if w_pi == 0.0:
            assert arrs == [None, None]                      # nothing to pull with
            continue
        a_star = (w_pi / (1.0 + w_pi)) * DELTA
        np.testing.assert_allclose((arrs[1] - arrs[0]) / 5.0,
                                   1.0 / (MU_P * ALPHA * a_star), atol=0.0)

    # the coupler is not exact, and nothing returns to the monitor
    F = pull_F()
    for p in (0.1, 0.4, 1.0, 5.0):
        M = d_omega(F)([2.0, 2.0, p, PI0])
        np.testing.assert_allclose(M[3, 2], -ALPHA * np.tanh(p / WID), rtol=1e-12)
        np.testing.assert_allclose(M[1, 2], +ALPHA * np.tanh(p / WID), rtol=1e-12)
        np.testing.assert_allclose(np.abs(M).max(), abs(M[3, 2]), rtol=0.0)
    np.testing.assert_allclose(d_omega(F)([2.0, 2.0, 5.0, PI0])[3, 2], -ALPHA, atol=1e-16)

    _, _, b5, _, _ = looped(5.0, "silent")
    q0 = jnp.asarray(flat_state(b5.P.init))
    J = jax.jacobian(lambda q: b5.omega(q, EMPTY, EMPTY)[0])(q0)
    comp = np.asarray(J.T - J)
    np.testing.assert_allclose(comp[MON_PI, P_SPARE], -ALPHA, rtol=1e-12)
    np.testing.assert_allclose(np.abs(comp).max(), ALPHA, rtol=1e-12)

    # the supply: the work delivered is finite because the job ends, not because it is paid
    work = float(np.sum(pull[:-1] * np.diff(rows[:, P_SPARE])))
    np.testing.assert_allclose(work, 0.0916588, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 5: (C4) -- is the monitor idle where it is not needed?
# ---------------------------------------------------------------------------


def test_5_C4_the_monitor_changes_the_message_by_nothing_and_the_placements_by_a_hundredth():
    """CHECKS THE DISCRETE ANALOGUE of (C4) and CORRECTS it in one half. The silent
    failure with the spare inside the parent's own reach, ``d = 2``, is the case
    ``notes/TURNOVER-TEST.md`` found repairs itself: four arrangements, the bare
    collective, the collective with a free monitor (``w_pi = 0``), with a held one
    (``w_pi = 1``), and the whole loop.

    | | h~ - nu | w~ - nu | the live pair's placement |
    |---|---|---|---|
    | bare | 0.0 | 0.0 | +-0.01932436 |
    | + free monitor | -4.4e-16 | -4.4e-16 | +-0.01019096 |
    | + held monitor | 0.0 | 0.0 | +-0.00845884 |
    | loop | 0.0 | 0.0 | +-0.00916135 |

    **On the message (C4) holds exactly**, which is the criterion this whole stage is
    measured by: adding the monitor, and the doodad with it, changes the collective's
    outgoing message by ``0`` in the two held cases and by one unit in the last place with
    a free prediction. **On the configuration it does not**: the live pair rests
    ``0.0109`` nearer the parent with the monitor than without, and the spare
    correspondingly further out, the seeing run's warning materializing. The reason it
    costs nothing here is the turnover run's: the fixed set is the whole level manifold
    ``sum_j kappa_j = nu``, so a monitor's transient moves the collective *along* it. A
    collective whose fixed point were isolated would pay for the same transient.
    """
    rows = {}
    bp, ba, bb, _ = bare_collective(D_NEAR, "silent")
    rows["bare"] = settled(bp, ba, bb, STEPS, ext=EXT)
    for lab, w_pi in (("free", 0.0), ("held", W_PI)):
        mp, ma, mb, _ = watched_collective(D_NEAR, "silent", w_pi=w_pi)
        rows[lab] = settled(mp, ma, mb, STEPS, ext=EXT)
    lp, la, lb, _, _ = looped(D_NEAR, "silent")
    rows["loop"] = settled(lp, la, lb, STEPS)

    for lab, (h, w, _, _, drift, _) in rows.items():
        assert drift < SETTLED
        np.testing.assert_allclose([h, w], [NU, NU], atol=1e-15)
    for lab in ("bare", "held", "loop"):
        assert rows[lab][0] - NU == 0.0 and rows[lab][1] - NU == 0.0
    assert abs(rows["free"][0] - NU) <= 4.5e-16

    # ... and the placements, which are not the same
    live = {lab: float(v[5][4 if lab == "bare" else 5]) for lab, v in rows.items()}
    np.testing.assert_allclose(
        [live["bare"], live["free"], live["held"], live["loop"]],
        [0.01932436, 0.01019096, 0.00845884, 0.00916135], atol=1e-7)
    assert abs(live["bare"] - live["held"]) > 0.010
