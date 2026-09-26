"""A monitor that sees a dead part: stage A of David's monitor test, **seeing**.

David, 2026-09-25: "there can't be a monitor who is able to see dead parts and also able
to pull from far? maybe that's something else to test separately and then we can put them
together and test that too?" This file is stage A alone; pulling is stage B, and composing
the two is a later stage. Nothing here is about reach.

**The prediction, David-confirmed ("yes: expectation about what should be arriving!"),
written into this file before any construction code existed and not revised after any
number was seen:**

    A silent failure is invisible to listening alone, so detection needs a **held**
    expectation of what should be arriving -- the curious-reader result that nothing
    surprises a free prediction and surprise needs a view that lags what is heard. So a
    monitor with a prior on the total weight detects the drop and one without does not,
    and the prior precision ``w_pi`` sets how large a drop it takes. **Able to fail:** if
    listening alone suffices, the curious-reader mechanism is not needed here.

The parent construction is ``tests/test_turnover.py`` (its numbers, ``notes/TURNOVER-TEST.md``),
whose classes and helpers are imported rather than copied, so that the collective watched
here is literally the collective run there. The mechanism is
``tests/test_curious_reader.py`` part 4: a prediction carried as a configuration coordinate,
an error term ``(1/2)(heard - pi)^2``, and a Gaussian prior ``(w_pi/2)(pi - pi_0)^2`` whose
precision is what makes the prediction lag. The monitor here holds an expectation of the
**total arriving weight** ``w~``, which is the quantity the turnover run found the required
cost to be blind to.

Every docstring below says which of three things its test does (STATUS.md §3.7): reproduce
a closed form derived in the docstring, check a discrete analogue, or correct a claim made
on paper. The numbers, and which verdicts the parameters decide rather than discover:
``notes/MONITOR-SEEING.md``.
"""

import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact
from arrangements.attention import (
    message, signal, spatial_arrangement, spatial_config, split_input, weight,
)
from arrangements.coalgebra import configuration
from arrangements.rvect import constant, euclidean
from arrangements.wiring import compose_seq, tensor_arrangements

from tests.chain import flat_state
from tests.test_rendezvous import run_closed, run_phi
from tests.test_turnover import (
    COSTS, DH, DP, EDIM, EXT, MDIM, MU_P, NU, NU_BAR, SIG, TAU, WGT,
    Part, Spec, collective, contributor, emitted, equilibrium_position, forcing,
    healthy, kappa_at, kappas, sent_onward, settle,
)

P_LIVE = equilibrium_position(3, 0.0, NU)   # 3 kappa(p) = nu, so p = 0.5401060
KAP = kappa_at(P_LIVE)                      # = nu/3 = 0.8333333
PI0 = NU                                    # what the monitor expects to arrive
THETA = 0.05                                # STIPULATED: the detection threshold
STEPS = 4000
SETTLED = 1e-9
W_PIS = (0.0, 0.0625, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
HEALTHS = (1.0, 0.9, 0.75, 0.5, 0.25, 0.0)


# ---------------------------------------------------------------------------
# The monitor: a held expectation of the total arriving weight.
# ---------------------------------------------------------------------------


def monitor_cost(w_pi, pi0=PI0):
    """``k(pi, e) = (1/2)(w(e) - pi)^2 + (w_pi/2)(pi - pi_0)^2``.

    The first term is the curious reader's error term with the *weight* of what arrives
    in place of its mean readout; the second is that test's prior, the same Gaussian
    precision ``w_pi``. Both are quadratic in the received message, so unlike the mean
    readout the monitor stays inside the affine-quadratic family whatever the parent's
    cost is.
    """

    def k(q, e):
        m, _ = split_input(e, DH)
        return 0.5 * (weight(m) - q[0]) ** 2 + 0.5 * w_pi * (q[0] - pi0) ** 2

    return k


def monitor(w_pi, mu_pi=None, pi0=PI0, label="MON"):
    """A component ``I -> B`` that sends ``0_M`` and carries ``monitor_cost``.

    Sending ``0_M`` keeps the monitor out of what the collective sends onward: it adds
    nothing to either coordinate of ``m~``, so the *message* of ``tests/test_turnover.py``
    is untouched. It does not make the monitor passive -- its cost is a force on whatever
    its input depends on, which is test 1's correction. Its mobility is ``1/(2 L)`` with ``L = 1 + w_pi`` the exact Hessian of its cost in
    ``pi``, so every run below contracts by the same factor ``1/2`` per step and the
    sweep over ``w_pi`` is not also a sweep over the step.
    """
    mu_pi = 0.5 / (1.0 + w_pi) if mu_pi is None else mu_pi
    k = monitor_cost(w_pi, pi0)
    return Arrangement(
        P=configuration(euclidean(1, mu_pi), init=jnp.array([float(pi0)])),
        in_dim_M=0, out_dim_M=0, in_dim_N=EDIM, out_dim_N=MDIM,
        out_f=lambda q, m_out: message(jnp.zeros(DH), 0.0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -k(q, n_in)),
        label=label,
    )


def held(c, mode="silent", pos=None):
    """A contributor whose health is held at ``c`` and does not move: the failure as a
    state of a given severity, so that the size of the drop is a swept parameter."""
    return Spec(mode, c, c, 0.0, P_LIVE if pos is None else pos)


def watched(specs, w_pi, cost_kind="total", mu_pos=0.0, mu_pi=None, pi0=PI0):
    """The turnover collective with a monitor added as one more child of the same parent.

    The monitor sits at ``p = 0``, the parent's own reference position (``l_0 = 0_P`` in
    eq:spatial-transfers), and its position is frozen. So its kernel to contributor ``j``
    is ``kappa(0 - p_j)``, which is the parent's own kernel to ``j``, and what it receives
    is exactly the message ``m~`` the collective sends onward. Listening from the parent's
    vantage point is a stipulation, not a derivation; test 1 checks it does what it says.
    """
    n = len(specs) + 1
    parts = [contributor(s, f"C{i}") for i, s in enumerate(specs)]
    mon = monitor(w_pi, mu_pi, pi0)
    all_parts = parts + [Part(mon, held(pi0, pos=0.0), "MON")]
    mobilities = jnp.array([mu_pos] * len(specs) + [0.0] + [0.0] * (n * DH * DH))
    a = spatial_arrangement(
        n, DH, DP, TAU, cost=COSTS[cost_kind](n),
        parameter=configuration(
            constant(jnp.diag(mobilities)),
            init=spatial_config([s.pos for s in specs] + [0.0], [1.0] * n)),
        label=f"watched({n})")
    b = compose_seq(tensor_arrangements([p.arr for p in all_parts]), a.arr())
    return all_parts, a, b


def failed(c=0.0, mode="silent", w_pi=1.0, cost_kind="total", mu_pos=0.0):
    """Two contributors in full health at ``+p`` and ``-p`` and one at health ``c``, all
    at the positions where a healthy trio sends exactly ``nu``, plus the monitor."""
    return watched([healthy(P_LIVE), healthy(-P_LIVE), held(c, mode)],
                   w_pi, cost_kind, mu_pos)


MON = 3            # the monitor's child index, and its configuration coordinate
FAIL = 2           # the contributor whose health is held down


def heard(all_parts, a, q):
    """``rho_a(q, m, e)_MON``'s message: what the monitor receives."""
    q = jnp.asarray(q, float)
    n = len(all_parts)
    e = a.received(q[n:], emitted(all_parts, q), EXT, n)
    return split_input(e, DH)[0]


def surprise(all_parts, a, q):
    """``w(heard) - pi``: the monitor's prediction error on the total arriving weight.

    This is the whole of the force the error term exerts on ``pi``; at settlement it is
    what the prior holds open, and it is the quantity a downstream part would read.
    """
    return float(weight(heard(all_parts, a, q)) - jnp.asarray(q, float)[MON])


def detectable_drop(w_pi, theta=THETA):
    """``theta (1 + w_pi) / w_pi``: the smallest drop whose settled surprise reaches the
    threshold, derived in test 3. Infinite at ``w_pi = 0``."""
    return np.inf if w_pi == 0.0 else theta * (1.0 + w_pi) / w_pi


# ---------------------------------------------------------------------------
# Test 0: the runs are Phi's.
# ---------------------------------------------------------------------------


def test_0_the_runs_are_the_dynamics_functors():
    """LICENSES the run forms used below; it establishes nothing about the prediction.
    ``Phi`` of the composite, ``run_closed`` and the ``fori_loop`` of ``settle`` agree
    over fifty steps to one unit in the last place, so every trajectory here is ``Phi``'s.
    """
    _, _, b = failed(mu_pos=MU_P)
    by_phi = run_phi(b, 50, EXT)
    by_closed = run_closed(b, 50, lambda t: EXT)
    assert float(np.max(np.abs(by_phi - by_closed))) < 1e-15
    assert float(np.max(np.abs(settle(b, 50) - by_closed[-1]))) < 1e-15


# ---------------------------------------------------------------------------
# Test 1: the monitor hears m~ and does not perturb the collective.
# ---------------------------------------------------------------------------


def test_1_the_monitor_hears_m_tilde_and_acts_on_the_collective_exactly_when_surprised():
    """REPRODUCES the closed form derived here for what the monitor hears, and CORRECTS an
    assumption of my own construction -- that a part which sends ``0_M`` is a passive
    observer. It is not: a held expectation is a cost, and a cost is a force.

    *What it hears.* ``rho_a(q, m, e)_i`` has message ``T_{i0} m_0 + sum_j T_{ij} m_j``
    with ``T_{ij} = (kappa_ij, kappa_ij V_j)``, ``kappa_ij = exp(-||p_i - p_j||^2/2 tau)``.
    At ``p_MON = 0`` this is ``kappa_{0j}``, the parent's own kernel to ``j``, and the
    exterior message is ``0_M``, so the monitor's input message is
    ``sum_j T_{0j} m_j = sigma_a(q, m) = m~`` bit for bit. Its own ``0_M`` contributes
    nothing to ``m~``, so it is excluded from its own input for free. Pre-failure every
    contributor sends ``(SIG, WGT) = (1, 1)`` and ``3 kappa(p) = nu``, so ``m~ = (nu, nu)``,
    the monitor at ``pi = pi_0 = nu`` is unsurprised, and the whole composite forcing
    vanishes.

    *What it does.* The forcing the monitor adds to the composite is ``-s grad w_recv``
    with ``s = w_recv - pi`` its surprise, so on contributor ``j``

        health:    -s kappa_j dw_j/dc_j   (``WGT`` in the silent mode, ``0`` reporting)
        position:  -s w_j dkappa_j/dp_j = s w_j (p_j/tau) kappa_j.

    At the silent failed state ``s = -kappa(p) = -0.833333``, and the added forces are
    ``+0.694444`` on the dead part's health -- which is exactly the force the parent's own
    deficit already exerts, so the monitor **doubles the prop-up that keeps a failing part
    alive** -- and ``-0.468842`` on each live part's position, again doubling the parent's.
    Under the mean readout, where the parent's forcing is identically zero, these are the
    *only* forces in the composite. In the reporting mode ``s = 0`` at this state and the
    monitor is inert to the last bit. **So the monitor is passive exactly when it is
    unsurprised**, and the seeing mechanism cannot be separated from an intervention.
    """
    all_parts, a, b = watched([healthy(P_LIVE), healthy(-P_LIVE), healthy(P_LIVE)],
                              1.0, mu_pos=MU_P)
    q0 = jnp.asarray(flat_state(b.P.init))
    m_out = sent_onward(all_parts, a, q0)

    np.testing.assert_allclose(heard(all_parts, a, q0), m_out, atol=0.0)
    np.testing.assert_allclose([float(signal(m_out)[0]), float(weight(m_out))],
                               [NU, NU], atol=1e-14)
    assert surprise(all_parts, a, q0) == 0.0
    assert float(np.max(np.abs(forcing(b, q0)))) < 1e-14
    np.testing.assert_array_equal(settle(b, 200), np.asarray(q0))

    dkappa = -(P_LIVE / TAU) * KAP
    for cost_kind in ("total", "mean"):
        for mode in ("reporting", "silent"):
            specs = [healthy(P_LIVE), healthy(-P_LIVE), held(0.0, mode)]
            all_parts, a, b = watched(specs, 1.0, cost_kind, mu_pos=MU_P)
            _, _, b3 = collective(specs, cost_kind, mu_pos=MU_P)
            q4 = flat_state(b.P.init)
            with_mon, without = forcing(b, q4), forcing(b3, flat_state(b3.P.init))
            s = surprise(all_parts, a, q4)
            dw_dc = WGT if mode == "silent" else 0.0
            w_fail = 0.0 if mode == "silent" else WGT

            np.testing.assert_allclose(with_mon[MON], s, atol=1e-14)   # the force on pi
            added_c = with_mon[:3] - without[:3]
            added_p = with_mon[4:7] - without[3:6]
            np.testing.assert_allclose(added_c, [0.0, 0.0, -s * KAP * dw_dc], atol=1e-14)
            np.testing.assert_allclose(
                added_p, [s * WGT * -dkappa, -s * WGT * -dkappa, s * w_fail * -dkappa],
                atol=1e-14)
            if mode == "reporting":
                np.testing.assert_allclose(with_mon[:3], without[:3], atol=0.0)
                np.testing.assert_allclose(with_mon[4:7], without[3:6], atol=0.0)
            else:
                np.testing.assert_allclose(added_c[2], 0.694444444, rtol=1e-8)
                np.testing.assert_allclose(added_p[0], -0.468842, rtol=1e-5)
                if cost_kind == "total":                   # the prop-up, exactly doubled
                    np.testing.assert_allclose(added_c[2], without[2], rtol=1e-12)
                    np.testing.assert_allclose(added_p[0], without[3], rtol=1e-12)
    # under the mean readout the monitor is the only force there is
    specs = [healthy(P_LIVE), healthy(-P_LIVE), held(0.0, "silent")]
    _, _, b3 = collective(specs, "mean", mu_pos=MU_P)
    assert float(np.max(np.abs(forcing(b3, flat_state(b3.P.init))))) == 0.0


# ---------------------------------------------------------------------------
# Test 2: listening alone -- the prediction's premise, under both costs.
# ---------------------------------------------------------------------------


def test_2_listening_alone_sees_the_silent_failure_under_the_required_cost():
    """CORRECTS the prediction's premise for the cost this construction is required to
    carry, and confirms it for the mean readout. This is the **Able to fail** clause.

    At the silent failed state the collective sends ``h~ = 2 kappa(p) = 2 nu / 3``, so the
    parent's deficit is ``nu - h~ = nu/3 = kappa(p)`` -- the dead part's whole share. The
    force on a live contributor's position is ``(nu - h~) v_j dkappa_j/dp_j`` with
    ``dkappa/dp = -(p/tau) kappa``, which is nonzero: **a silent failure is not invisible
    to a cost on the weighted signal.** It is invisible to a cost on the mean readout, where
    ``h_j = w_j`` for every contributor makes the readout exactly ``nu_bar`` and the entire
    forcing on the collective zero.
    """
    rows = {}
    for cost_kind in ("total", "mean"):
        specs = [healthy(P_LIVE), healthy(-P_LIVE), held(0.0, "silent")]
        _, a, b3 = collective(specs, cost_kind, mu_pos=MU_P)
        parts = [contributor(s, f"C{i}") for i, s in enumerate(specs)]
        q0 = jnp.asarray(flat_state(b3.P.init))
        m = sent_onward(parts, a, q0)
        tr = run_closed(b3, 300, lambda t: EXT)
        rows[cost_kind] = (forcing(b3, q0), float(signal(m)[0]), float(weight(m)), tr)

    xi_tot, h_tot, w_tot, tr_tot = rows["total"]
    np.testing.assert_allclose([h_tot, w_tot], [2.0 * KAP, 2.0 * KAP], atol=1e-14)
    np.testing.assert_allclose(xi_tot[3], -KAP * SIG * (P_LIVE / TAU) * KAP, rtol=1e-12)
    assert abs(xi_tot[3]) > 0.4                       # listening alone: a real force
    assert np.max(np.abs(tr_tot[-1] - tr_tot[0])) > 0.5   # and the collective moves

    xi_mean, h_mean, w_mean, tr_mean = rows["mean"]
    np.testing.assert_allclose(h_mean / w_mean, NU_BAR, atol=1e-15)
    assert float(np.max(np.abs(xi_mean))) == 0.0      # listening alone: nothing at all
    np.testing.assert_array_equal(tr_mean, np.repeat(tr_mean[:1], len(tr_mean), axis=0))


# ---------------------------------------------------------------------------
# Test 3: the prediction, with the collective frozen -- w_pi against the drop.
# ---------------------------------------------------------------------------


def test_3_the_prior_precision_sets_the_detectable_drop():
    """REPRODUCES the closed form derived here, and so tests the prediction's two
    quantitative halves with the collective held still, where the drop is a dial.

    Every contributor's health and every position is frozen -- STIPULATED, and it is what
    suppresses the back-action test 1 found, so that the monitor here only watches. The
    only moving coordinate is the monitor's ``pi``, ``w~`` is a constant of the run, and the
    step
    ``pi_{t+1} = pi_t + mu[(w~ - pi_t) - w_pi(pi_t - pi_0)]`` is affine. With
    ``mu = 1/(2(1 + w_pi))`` it contracts by exactly ``1/2`` per step toward

        pi* = (w~ + w_pi pi_0) / (1 + w_pi),   s* = w~ - pi* = -(w_pi/(1 + w_pi)) Delta,

    where ``Delta = pi_0 - w~`` is the drop. In the silent mode ``w~ = (2 + c) kappa(p)``
    at held health ``c``, so ``Delta = (1 - c) kappa(p)``, and the surprise along the run is
    ``s_t = s* - 2^{-t} (Delta + s*)``.

    **At ``w_pi = 0``, ``s* = 0`` exactly: a free prediction is not surprised**, and the
    rate is ``s_t = -Delta 2^{-t}`` -- the whole drop at step 0, under the threshold from
    step ``ceil(log2(Delta/theta)) = 5`` on. So "one without does not detect" is a statement
    about the settled run with a rate attached, not about a budget: at full silence the free
    monitor is above threshold at steps ``0..4`` and below it forever after.

    Detection at threshold ``theta`` therefore needs ``Delta >= theta (1 + w_pi)/w_pi``, so
    **the prior precision sets the detectable drop**, and the largest available drop,
    ``kappa(p) = nu/3``, puts a floor under the useful precision at
    ``w_pi = theta/(kappa - theta)``.
    """
    # the settled surprise, and the run's rate, for every (w_pi, drop) pair
    settled, traces = {}, {}
    for w_pi in W_PIS:
        for c in HEALTHS:
            all_parts, a, b = failed(c=c, w_pi=w_pi)
            tr = run_closed(b, 60, lambda t: EXT)
            s = [surprise(all_parts, a, q) for q in tr]
            w_tilde = float(weight(heard(all_parts, a, tr[0])))
            settled[w_pi, c] = (w_tilde, s[-1])
            traces[w_pi, c] = np.array(s)

    for w_pi in W_PIS:
        for c in HEALTHS:
            drop = (1.0 - c) * KAP
            w_tilde, s_end = settled[w_pi, c]
            s_star = -(w_pi / (1.0 + w_pi)) * drop
            np.testing.assert_allclose(w_tilde, (2.0 + c) * KAP, atol=1e-14)
            np.testing.assert_allclose(s_end, s_star, atol=1e-13)
            # the rate: contraction by exactly one half per step
            s = traces[w_pi, c]
            np.testing.assert_allclose(s[0], -drop, atol=1e-14)
            steps = np.arange(len(s))
            np.testing.assert_allclose(s, s_star - 0.5 ** steps * (drop + s_star), atol=1e-12)

    # a free prediction is not surprised -- with a rate, at the largest available drop
    free = traces[0.0, 0.0]
    np.testing.assert_allclose(free, -KAP * 0.5 ** np.arange(len(free)), atol=1e-14)
    above = np.abs(free) >= THETA
    n_above = int(np.ceil(np.log2(KAP / THETA)))
    assert n_above == 5 and above[:n_above].all() and not above[n_above:].any()
    assert abs(free[-1]) < 1e-15

    # the detection table: which (w_pi, drop) pairs are detected
    detected = {k: abs(v[1]) >= THETA for k, v in settled.items()}
    for w_pi in W_PIS:
        for c in HEALTHS:
            assert detected[w_pi, c] == ((1.0 - c) * KAP >= detectable_drop(w_pi) - 1e-15)
        assert not detected[w_pi, 1.0]                   # no drop, no detection, any w_pi
    for c in HEALTHS:
        assert not detected[0.0, c]                      # no prior, no detection, any drop

    # the floor under a useful precision: the largest drop is kappa(p)
    w_min = THETA / (KAP - THETA)
    np.testing.assert_allclose(w_min, 0.0638297872, rtol=1e-9)
    assert not detected[0.0625, 0.0] and detected[0.1, 0.0]
    assert 0.0625 < w_min < 0.1


# ---------------------------------------------------------------------------
# Test 4: the prediction with the collective live, under both costs.
# ---------------------------------------------------------------------------


def test_4_the_verdict_under_both_costs_and_both_failure_modes():
    """CHECKS THE DISCRETE ANALOGUE of the prediction in the coupled system, where the
    collective responds to the failure while the monitor watches -- and CORRECTS the
    reading of the monitor as a detector. Four settled states at 4000 steps, the monitor
    at ``w_pi = 0`` and ``w_pi = 1``; ``kappa`` below is ``(C0, C1, failed, MON)``.

    | cost | mode | w_pi | w~ arriving | settled surprise | kappa of the failed part |
    |---|---|---|---|---|---|
    | weighted signal | reporting | 0 | 2.7456255 | 0 | 0.7456255 |
    | weighted signal | reporting | 1 | **2.5000000** | **0** | **0.5000000** |
    | weighted signal | silent | 0 | 2.0000000 | 0 | 0.8333333 |
    | weighted signal | silent | 1 | 2.0000000 | **-0.25** | 0.8333333 |
    | mean readout | reporting | 0 | 1.9926702 | -1.9e-6 | 0.0094600 (still moving) |
    | mean readout | reporting | 1 | 2.3862888 | **-0.0568556** | 0.3862888 |
    | mean readout | silent | 0 | 1.9230078 | 0 | 0.8333333 |
    | mean readout | silent | 1 | 2.0000000 | **-0.25** | 0.8333333 |

    Three things, of which only the second was predicted.

    *The free monitor's settled surprise is zero in every row* -- to one unit in the last
    place, ``2.2e-16`` -- except in the one run that never settles (the mean readout with a reporting failure, drift ``2e-2``
    per 500 steps, ``kappa_F`` still falling), where it lags by ``1.8e-6``, a thousandth of
    the way to the threshold. So a free prediction is surprised only by what is still
    changing, and only in proportion to how fast.

    *The monitor with a prior detects the silent failure under both costs*: ``-0.25``
    against a threshold of ``0.05``, and the closed form
    ``s = -(w_pi/(1+w_pi))(pi_0 - w~)`` holds in every settled row. Under the mean readout
    it is the only thing in the composite that registers anything, since test 2 shows the
    collective's own forcing there is identically zero.

    *But it does not only detect: it acts, and what it does is repair.* Under the
    weighted-signal cost with a **reporting** failure -- the case ``notes/TURNOVER-TEST.md``
    found the required cost cannot push out, because a zero signal at full weight and a
    silent part are the same input to it -- the monitor drives the dead part's kernel from
    ``0.8333333`` to ``0.5000000``, exactly until the weight it expects arrives, and its own
    surprise closes to zero. **The monitor supplies the sense the parent's cost lacks, and
    the collective then uses it.** It cannot do this to a silent failure: a silent part
    carries no weight, so moving it changes nothing, and the surprise stays open at
    ``-0.25`` forever. Under the mean readout with a reporting failure the prior does the
    opposite service: turnover found that run has no settled state, the failed part pushed
    away at every step; with the prior it settles at ``kappa_F = 0.386``, the monitor's
    surprise stopping the runaway.

    So the drop is detected, but "detected" and "acted on" are not separable here, and the
    largest effect in the table is one the prediction does not mention.
    """
    rows = {}
    for cost_kind in ("total", "mean"):
        for mode in ("reporting", "silent"):
            for w_pi in (0.0, 1.0):
                all_parts, a, b = failed(mode=mode, w_pi=w_pi, cost_kind=cost_kind,
                                         mu_pos=MU_P)
                q1, q2 = settle(b, STEPS), settle(b, STEPS + 500)
                rows[cost_kind, mode, w_pi] = (
                    float(weight(heard(all_parts, a, q2))),
                    surprise(all_parts, a, q2),
                    float(np.max(np.abs(q2 - q1))),
                    kappas(q2, len(all_parts)))

    unsettled = ("mean", "reporting", 0.0)
    for key, (w_tilde, s, drift, _) in rows.items():
        if key == unsettled:
            assert drift > 1e-3                          # the one run with no settled state
            assert abs(s) < 1e-5 and abs(s) < THETA / 1000.0
            continue
        assert drift < SETTLED
        w_pi = key[2]
        np.testing.assert_allclose(s, -(w_pi / (1.0 + w_pi)) * (PI0 - w_tilde), atol=1e-9)
        if w_pi == 0.0:
            assert abs(s) <= 2.3e-16                     # a free prediction: not surprised

    # what arrives, and whether the monitor detects it
    detected = {k: abs(v[1]) >= THETA for k, v in rows.items()}
    assert {k for k, v in detected.items() if v} == {
        ("total", "silent", 1.0), ("mean", "silent", 1.0), ("mean", "reporting", 1.0)}
    assert not any(detected[k] for k in rows if k[2] == 0.0)          # no prior, no detection
    assert detected["total", "silent", 1.0] and detected["mean", "silent", 1.0]
    np.testing.assert_allclose(rows["total", "silent", 1.0][1], -0.25, atol=1e-9)
    np.testing.assert_allclose(rows["mean", "silent", 1.0][1], -0.25, atol=1e-9)
    np.testing.assert_allclose(rows["mean", "reporting", 1.0][1], -0.0568556, atol=1e-7)
    assert detected["mean", "reporting", 1.0]
    assert not detected["total", "reporting", 1.0]       # because it was repaired, below

    # the repair: the reporting failure the required cost cannot push out
    w_rep, s_rep, _, k_rep = rows["total", "reporting", 1.0]
    np.testing.assert_allclose(w_rep, PI0, atol=1e-5)
    assert abs(s_rep) < 1e-6
    np.testing.assert_allclose(k_rep[FAIL], 0.5, atol=1e-5)
    assert k_rep[FAIL] < rows["total", "reporting", 0.0][3][FAIL] - 0.2   # pushed out
    np.testing.assert_allclose(rows["total", "reporting", 0.0][3][FAIL], 0.7456255, atol=1e-7)
    # and the silent failure it cannot repair: the dead part is not moved at all
    np.testing.assert_allclose(rows["total", "silent", 1.0][3][FAIL], KAP, atol=1e-9)
    np.testing.assert_allclose(rows["mean", "silent", 1.0][3][FAIL], KAP, atol=1e-9)

    # under the mean readout the collective moves only because the monitor moves it
    all_parts, a, b = failed(mode="silent", w_pi=1.0, cost_kind="mean", mu_pos=MU_P)
    moved = np.max(np.abs(run_closed(b, 300, lambda t: EXT) - np.asarray(flat_state(b.P.init))),
                   axis=0)
    assert moved[MON] > 0.2 and moved[4] > 0.5 and moved[5] > 0.5     # pi, and the live pair
    assert moved[FAIL] == 0.0 and moved[6] == 0.0                     # the dead part: nothing
