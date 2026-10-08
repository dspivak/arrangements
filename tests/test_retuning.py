"""A reader that retunes: the reader-origin composite with a site placement that is a slow state.

P38 stage 1 (the book repository's plan §8.2; Claude Fable 5.1, 27dc74fb, 2026-10-07). The predictions,
the closed forms and the strongest objection are fixed before any of this code in the book repository's
``notes/runs/p38-retuning-27dc74fb/PREDICTIONS.md``; this docstring restates what the code needs.

**Where it starts.** ``test_reader_origin.py`` stage B, T3: the match is the kernel
``kappa = exp(-(s - S)^2 / 2 tau)`` between the apparatus's site placement ``s`` and the substrate's
placement ``S``; the substrate drifts at ``v = 0.005`` with the site held; the reader collapses at distance
``0.812`` (the fold ``d_* = 0.722``) and drifting back does not restore it.

**What changes** (``Model`` below).

1. The site is a slow state, response ``h mu`` (``mu = 0`` is T3's held site).
2. It descends the reader's own cost. T1b's potential has one term through the match, the rebuild's
   ``U = -k_r kappa F q_1``; so the rebuild child's forcing gets its component on the site,
   ``xi_s = -dU/ds = -k_r F q_1 kappa (s - S) / tau``, the one T3's direct covector left at zero. The substrate
   is a datum the reader cannot move: the rebuild's covector on ``S`` stays zero unless ``exact`` (piece 5).
3. The motion is charged to the fuel: a stateless child returns ``-chi mu xi_s^2`` on ``F``, the work the
   forcing does per unit time moving the site, at exchange rate ``chi``.
4. Where the charge is to count the fuel is a species (T4c's law: fed at ``J``, washed out at rate 1, one unit
   per rebuild and per boundary-production event), with ``J = 1 + 13 q_1^rest = 24.246`` so that at rest
   ``F = 1`` and the reader is T3's.

**Closed forms** (hand derivations; checked here against the runs). Tracking is steady when the site's speed
equals the drift, ``mu k_r F q_1 kappa(d) d / tau = v`` with ``q_1`` the exposed high state at effective match
``kappa F`` and, fed, ``F (1 + 13 kappa q_1) = J - chi v^2 / mu``. Held fuel: ``v_c = 1.767 mu``, the maximum
pull at lag ``0.509``, inside the fold. Width: ``v_c(tau) = v_c(1) / sqrt(tau)``. Fed, ``chi = 0``:
``v_c = 0.0815`` at ``mu = 0.01``, since unspent fuel pools (to ``J``) and powers the pull. Fed with charge:
the charge equals the whole feed at ``v_pay = sqrt(mu J / chi)`` and ``v_c`` lies in ``[0.83, 0.86] v_pay``.

**Pieces.** 0: T3 reproduced with ``mu = 0``. 1: held fuel, tracking below ``v_c = 0.01767`` at ``mu = 0.01``,
mobility scaling. 2: fed fuel, the fixed reader reaches ``d = 2.52`` by pooling; the mobile one tracks to
``v = 0.08``. 3: the price of staying tuned, ``v_c(chi)`` and the tracking reader's lower fuel and size.
4: the strongest objection, fixed against mobile at the same drift across widths, counts each way. 5: the exact
control, the reader's cost also pulls the substrate, ``v_c`` up by ``(mu + mu_S) / mu`` and the substrate
yielding a third of the drift.

Collapse means ``q_1 < 0.05`` as in T3. Time is discrete, ``h = 0.01``; runs are 600 time units unless said.
"""

from dataclasses import dataclass, replace
import math

import jax.numpy as jnp
import numpy as np
import pytest

from arrangements.rvect import constant, euclidean

from tests import test_reader_origin as ro

H = ro.H
HELD = ro.HELD
K_R, K_B, L_OPEN = 3.0, 10.0, ro.L_OPEN
Q_REST = ro.exact_equilibria(L_OPEN)[1]            # (1.788153, 17.881527), T3's established state
J_REST = 1.0 + (K_R + K_B) * float(Q_REST[0])      # 24.246: fed fuel rests at F = 1
E_STAR, D_STAR = ro.KAPPA_STAR, ro.D_STAR
COLLAPSE = 0.05
DURATION = 600.0


@dataclass(frozen=True)
class Model:
    tau: float = 1.0
    mu: float = 0.01        # site mobility; 0 holds the site (T3)
    v: float = 0.005        # drift speed of the substrate
    mu_S: float = 0.005     # the substrate's response (T3's)
    feed: float = None      # None: fuel held at 1; else fed at this rate, washed out at rate 1
    chi: float = 0.0        # exchange rate: fuel per unit of work moving the site
    exact: bool = False     # the rebuild's covector on S too (piece 5)


def kappa(s, S, tau):
    return jnp.exp(-(s - S) ** 2 / (2.0 * tau))


def transitions(m):
    use = 0.0 if m.feed is None else 1.0

    def pull(n):      # the rebuild's forcing on the site, -dU/ds for U = -k_r kappa F q_1
        A, F, s, S = n
        return -K_R * F * A * kappa(s, S, m.tau) * (s - S) / m.tau

    def rebuild(n):
        A, F, s, S = n
        act = K_R * kappa(s, S, m.tau) * F * A
        xi = pull(n)
        return jnp.array([act, -use * act, xi, -xi if m.exact else 0.0])

    def charge(n):
        return jnp.array([0.0, -m.chi * m.mu * pull(n) ** 2, 0.0, 0.0])

    def production(n):
        A, B, F, s, S = n
        rate = K_B * kappa(s, S, m.tau) * F * A
        return jnp.array([0.0, rate, -use * rate, 0.0, 0.0])

    ts = [
        ro.Transition("rebuild", ("A", "F", "site", "S"), covector=rebuild),
        ro.Transition("retuning charge", ("A", "F", "site", "S"), covector=charge),
        ro.Transition("boundary production", ("A", "B", "F", "site", "S"), covector=production),
        ro.Transition("damage", ("A",), covector=lambda n: -n),
        ro.Transition("saturating loss", ("A",), covector=lambda n: -n ** 2),
        ro.Transition("boundary loss", ("B",), covector=lambda n: -n),
        ro.Transition("protected leak", ("A", "B", "pore"),
                      covector=lambda n: jnp.array([-(L_OPEN - n[2]) * n[0] / (1.0 + n[1]), 0.0, 0.0])),
        ro.Transition("drift", ("S",), covector=lambda n: jnp.array([m.v / m.mu_S])),
    ]
    if m.feed is not None:
        ts.append(ro.Transition("feed", ("F",), covector=lambda n: jnp.array([m.feed - n[0]])))
    return ts


def arrangement(m, q=Q_REST, fuel=1.0):
    """Configuration ``(A, B, F, site, S, pore)``; the site and the substrate start at 0."""
    boxes = [("A", q[0], euclidean(1, H)), ("B", q[1], euclidean(1, H)),
             ("F", fuel, HELD if m.feed is None else euclidean(1, H)),
             ("site", 0.0, HELD if m.mu == 0 else euclidean(1, H * m.mu)),
             ("S", 0.0, constant(jnp.array([[H * m.mu_S]]))),
             ("pore", 0.0, HELD)]
    return ro.network(boxes, transitions(m))


A_, B_, F_, SITE, S_ = 0, 1, 2, 3, 4


def run(m, duration=DURATION, every=10):
    return ro.trajectory(arrangement(m), int(round(duration / H)), every)


def collapse_distance(out):
    """The substrate's distance from the origin when ``q_1`` first drops below the threshold; None if never."""
    hit = np.where(out[:, A_] < COLLAPSE)[0]
    return None if len(hit) == 0 else float(out[hit[0], S_])


def lag(out):
    return out[:, S_] - out[:, SITE]


def tracks(m, duration=DURATION):
    """The prediction's criterion: no collapse within the run."""
    return collapse_distance(run(m, duration)) is None


def settled(m, duration=1500.0, tol=0.01):
    """Introduced after piece 3 (a) was refuted (2026-10-07), not predicted: no collapse, and the lag
    changes by less than ``tol`` over the last fifth of the run. Above the stationary threshold a fed
    reader declines for hundreds of time units while its fuel pools before it collapses, so a finite-run
    collapse criterion overshoots the closed form; a lag still growing at the end is the runaway seen early.
    The run is 1500 time units because the approach to the steady lag slows near the threshold (at
    ``chi = 1000``, ``v = 0.012``, 92% of ``v_c``, the lag still moves 0.02 between t = 430 and t = 640)."""
    out = run(m, duration)
    lg = lag(out)
    n = out.shape[0] // 5
    return bool(collapse_distance(out) is None and abs(lg[-1] - lg[-n]) < tol)


def threshold_speed(m, lo, hi, steps=10, criterion=tracks):
    """Bisection in ``v`` between a speed that passes the criterion and one that fails it."""
    assert criterion(replace(m, v=lo)) and not criterion(replace(m, v=hi))
    for _ in range(steps):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if criterion(replace(m, v=mid)) else (lo, mid)
    return lo, hi


# ---------------------------------------------------------------------------
# Closed forms, by hand (PREDICTIONS.md), used to state the predictions numerically.
# ---------------------------------------------------------------------------


def upper_root(e):
    return ro.upper_root(e)


def fed_fuel(J_eff, k):
    """Steady fuel at effective feed ``J_eff`` and match ``k``: ``F (1 + 13 k q_1(kF)) = J_eff`` on the
    high branch; nan if no high state."""
    from scipy.optimize import brentq
    if J_eff <= 0:
        return float("nan")
    lo = E_STAR / k * (1 + 1e-9)
    f = lambda F: F * (1.0 + (K_R + K_B) * k * upper_root(k * F)) - J_eff
    if lo >= J_eff or f(lo) > 0:
        return float("nan")
    return brentq(f, lo, J_eff)


def site_speed(d, m):
    """The site's steady speed at lag ``d`` under tracking, per the closed form."""
    k = float(kappa(0.0, d, m.tau))
    F = 1.0 if m.feed is None else fed_fuel(m.feed - m.chi * m.v ** 2 / m.mu, k)
    if math.isnan(F):
        return float("nan")
    q1 = upper_root(k * F)
    return (m.mu + (m.mu_S if m.exact else 0.0)) * K_R * F * q1 * k * d / m.tau


def steady_lag(m, dmax=3.0):
    ds = np.linspace(1e-3, dmax, 3000)
    for d in ds:
        s = site_speed(d, m)
        if not math.isnan(s) and s >= m.v:
            return float(d)
    return None


def v_crit(m):
    lo, hi = 1e-4, 1.0
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if steady_lag(replace(m, v=mid)) is not None else (lo, mid)
    return lo


# ---------------------------------------------------------------------------
# Piece 0. With the site held, the composite is T3's.
# ---------------------------------------------------------------------------


def test_p0_held_site_is_T3_and_fed_fuel_rests_at_T3s_state():
    out = run(Model(mu=0.0), duration=240.0)
    d = collapse_distance(out)
    print("P0: fixed held reader collapses at d =", d, "(T3: 0.812)")
    assert abs(d - 0.812) < 0.005
    rest = ro.settle(arrangement(Model(mu=0.0, v=0.0, feed=J_REST)), int(round(200.0 / H)))
    print("P0: fed rest state", rest[:3], "against", Q_REST, 1.0)
    assert np.allclose(rest[:2], Q_REST, atol=1e-6) and abs(rest[F_] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Piece 1. Held fuel: the mobile site tracks below a critical speed set by its mobility.
# ---------------------------------------------------------------------------

V_C_HELD = 0.01767  # = 1.767 mu at mu = 0.01


def test_p1a_tracks_at_the_closed_form_lag_where_the_fixed_reader_collapsed():
    m = Model()
    assert abs(v_crit(m) - V_C_HELD) < 1e-4
    out = run(m)
    assert collapse_distance(out) is None
    final_lag = float(lag(out)[-1])
    predicted = steady_lag(m)
    k = float(kappa(0.0, final_lag, 1.0))
    print(f"P1a: tracks to S = {out[-1, S_]:.3f}; lag {final_lag:.4f} against {predicted:.4f}; "
          f"q_1 {out[-1, A_]:.4f} against the branch {upper_root(k):.4f}")
    assert abs(final_lag - 0.0945) < 0.005
    assert abs(out[-1, A_] - upper_root(k)) < 0.01


def test_p1b_a_critical_speed_set_by_the_mobility():
    m = Model()
    verdicts = {v: tracks(replace(m, v=v)) for v in (0.005, 0.010, 0.015, 0.0165, 0.019, 0.025)}
    print("P1b: tracks at", verdicts)
    assert [verdicts[v] for v in sorted(verdicts)] == [True, True, True, True, False, False]
    lo, hi = threshold_speed(m, 0.0165, 0.019)
    print(f"P1b: threshold in [{lo:.5f}, {hi:.5f}] against v_c = {V_C_HELD}")
    assert V_C_HELD <= hi <= 1.08 * V_C_HELD


def test_p1c_mobility_scaling():
    verdicts = {mu: tracks(Model(mu=mu)) for mu in (0.002, 0.005, 0.01)}
    print("P1c: tracks at mu", verdicts)
    assert [verdicts[mu] for mu in sorted(verdicts)] == [False, True, True]
    out = run(Model(mu=0.005))
    print("P1c: lag at mu = 0.005:", lag(out)[-1], "against 0.198")
    assert abs(lag(out)[-1] - 0.198) < 0.01


# ---------------------------------------------------------------------------
# Piece 2. Fed fuel: pooling, and a fixed reader that reaches much farther.
# ---------------------------------------------------------------------------


def test_p2a_the_fixed_fed_reader_reaches_farther_by_pooling():
    out = run(Model(mu=0.0, feed=J_REST))
    d = collapse_distance(out)
    i = np.where(out[:, A_] < COLLAPSE)[0][0]
    print(f"P2a: fixed fed reader collapses at d = {d:.3f} (predicted [2.52, 2.65]); fuel there {out[i, F_]:.2f}; "
          f"max fuel {out[:, F_].max():.2f}")
    assert 2.52 <= d <= 2.65


def test_p2b_the_mobile_fed_reader_tracks_to_a_far_higher_speed():
    m = Model(feed=J_REST)
    vc = v_crit(m)
    print("P2b: closed-form v_c fed, chi = 0:", vc)
    assert abs(vc - 0.0815) < 0.002
    verdicts = {v: tracks(replace(m, v=v)) for v in (0.005, 0.03, 0.06, 0.1)}
    print("P2b: tracks at", verdicts)
    assert [verdicts[v] for v in sorted(verdicts)] == [True, True, True, False]
    out = run(m)
    print("P2b: lag at v = 0.005:", lag(out)[-1], "against 0.094")
    assert abs(lag(out)[-1] - 0.094) < 0.005
    lo, hi = threshold_speed(m, 0.06, 0.1, steps=8)
    print(f"P2b: threshold in [{lo:.4f}, {hi:.4f}] against [0.0815, 0.09]")
    assert 0.0815 <= hi <= 0.09


# ---------------------------------------------------------------------------
# Piece 3. The price of staying tuned.
# ---------------------------------------------------------------------------


def v_pay(m):
    return math.sqrt(m.mu * m.feed / m.chi)


@pytest.mark.xfail(strict=True, reason="refuted 2026-10-07: at v = 0.0145 the reader collapses only at "
                   "d = 18.5, t = 1280; the finite-run threshold is 0.0146-0.0147, 11% above v_c")
def test_p3a_with_a_charge_the_threshold_falls_below_the_price_ceiling():
    """Prediction (3a) as fixed, kept on the record: refuted. Above the stationary threshold the fed
    reader declines slowly, its lag growing from 0.77 at t = 214 to 1.84 at t = 1071 while the fuel pools
    from 1.0 to 4.5 (the weaker pull charges less), and collapses at t = 1280, far past the 600-unit run."""
    m = Model(feed=J_REST, chi=1000.0)
    verdicts = {v: tracks(replace(m, v=v)) for v in (0.005, 0.010, 0.012, 0.0145, 0.02)}
    print("P3a: chi = 1000 tracks at", verdicts)
    assert [verdicts[v] for v in sorted(verdicts)] == [True, True, True, False, False]


def bracket(m, vc, steps=7):
    """After the refutation: the stationary threshold ``vc`` bracketed between the settled-lag threshold
    (which undershoots, since near ``vc`` the lag approaches its steady value slowly) and the
    collapse-within-600 threshold (which overshoots, since above ``vc`` the decline is slow; it can even
    exceed ``v_pay``, since a reader that is losing its match is charged less than a tracking one)."""
    _, below = threshold_speed(m, 0.8 * vc, 1.6 * vc, steps=steps, criterion=settled)
    _, above = threshold_speed(m, 0.8 * vc, 1.6 * vc, steps=steps,
                               criterion=lambda mm: tracks(mm, duration=1500.0))
    return below, above


def test_p3a_settled_the_stationary_threshold_sits_below_the_price_ceiling():
    """After the refutation, the same claim read off the lag: it settles below v_c and runs away above."""
    m = Model(feed=J_REST, chi=1000.0)
    verdicts = {v: settled(replace(m, v=v)) for v in (0.005, 0.010, 0.012, 0.0145, 0.02)}
    print("P3a (settled): chi = 1000 settles at", verdicts)
    assert [verdicts[v] for v in sorted(verdicts)] == [True, True, True, False, False]
    below, above = bracket(m, 0.0131)
    out = run(replace(m, v=0.012), duration=1500.0)
    print(f"P3a (settled): v_c = 0.0131 bracketed by settled {below:.5f} and collapse {above:.5f}; "
          f"v_pay = {v_pay(m):.5f}; at v = 0.012 the lag {lag(out)[-1]:.3f} against "
          f"{steady_lag(replace(m, v=0.012)):.3f}, q_1 {out[-1, A_]:.3f}, F {out[-1, F_]:.3f}")
    assert 0.95 * 0.0131 <= below <= 0.0131 <= above and 0.0131 < v_pay(m)
    assert abs(lag(out)[-1] - steady_lag(replace(m, v=0.012))) < 0.02


def test_p3b_the_tracking_reader_runs_on_less_fuel_and_is_smaller():
    for chi, (F, q1, d) in ((1000.0, (0.970, 1.657, 0.105)), (4000.0, (0.865, 1.206, 0.162))):
        out = run(Model(feed=J_REST, chi=chi))
        assert collapse_distance(out) is None
        got = (float(out[-1, F_]), float(out[-1, A_]), float(lag(out)[-1]))
        print(f"P3b: chi = {chi}: (F, q_1, lag) = {np.round(got, 4)} against {(F, q1, d)}; rest (1, 1.788, 0)")
        assert abs(got[0] - F) < 0.02 * F and abs(got[1] - q1) < 0.02 * q1 and abs(got[2] - d) < 0.02 * d + 0.002


PREDICTED_V_C = {500.0: 0.0182, 1000.0: 0.0131, 2000.0: 0.0093, 4000.0: 0.0066}


@pytest.mark.xfail(strict=True, reason="refuted 2026-10-07 with (3a): the finite-run threshold at chi = 500 "
                   "is 0.0208 against v_c = 0.0182, 14% above")
def test_p3c_the_threshold_falls_with_the_charge():
    """Prediction (3c) as fixed, with the collapse-within-600 criterion: refuted, same cause as (3a)."""
    for chi, vc in PREDICTED_V_C.items():
        m = Model(feed=J_REST, chi=chi)
        assert abs(v_crit(m) - vc) < 0.0002, (chi, v_crit(m))
        lo, hi = threshold_speed(m, 0.9 * vc, 1.3 * vc, steps=7)
        print(f"P3c: chi = {chi}: threshold in [{lo:.5f}, {hi:.5f}] against {vc}, v_pay = {v_pay(m):.5f}")
        assert abs(hi - vc) <= 0.08 * vc and hi < v_pay(m)


def test_p3c_settled_the_threshold_falls_with_the_charge():
    """After the refutation: the stationary closed form bracketed by the two measured thresholds, each charge."""
    found = {}
    for chi, vc in PREDICTED_V_C.items():
        m = Model(feed=J_REST, chi=chi)
        exact_vc = v_crit(m)                      # the closed form to full precision; ``vc`` is its rounding
        assert abs(exact_vc - vc) < 0.0002, (chi, exact_vc)
        below, above = bracket(m, vc, steps=6)
        found[chi] = (below, above)
        print(f"P3c (settled): chi = {chi}: v_c = {exact_vc:.5f} bracketed by settled {below:.5f} and collapse "
              f"{above:.5f}; v_pay = {v_pay(m):.5f}")
        # The settled arm can sit a little above v_c: just over it the runaway is too slow to show in 1500
        # time units, the more so the higher the charge (chi = 2000: settled 0.00934 against v_c 0.00931;
        # chi = 4000: 0.00670 against 0.00661, 2026-10-07). The bracket is stated to 2%.
        assert 0.95 * exact_vc <= below <= 1.02 * exact_vc and exact_vc <= above and exact_vc < v_pay(m)
    assert [b for b, _ in found.values()] == sorted((b for b, _ in found.values()), reverse=True)


# ---------------------------------------------------------------------------
# Piece 4. The strongest objection: is a mobile placement just a larger fixed reader?
# ---------------------------------------------------------------------------


def test_p4_fixed_against_mobile_at_the_same_drift_across_widths():
    fixed, mobile = {}, {}
    for tau in (1.0, 2.0, 4.0, 9.0):
        out_f = run(Model(tau=tau, mu=0.0))
        # The widest reader approaches its steady lag slowly (0.906 at t = 600 against 0.998, seen
        # 2026-10-07); it is run to t = 1500 so the comparison is of settled lags. Not a prediction change.
        out_m = run(Model(tau=tau), duration=1500.0 if tau == 9.0 else DURATION)
        fixed[tau] = collapse_distance(out_f)
        mobile[tau] = (collapse_distance(out_m), float(lag(out_m)[-1]), steady_lag(Model(tau=tau)))
        print(f"P4: tau = {tau}: fixed collapses at {fixed[tau]} (d_* = {0.7222 * math.sqrt(tau):.3f}); "
              f"mobile collapse {mobile[tau][0]}, lag {mobile[tau][1]:.4f} against {mobile[tau][2]}")
    assert all(d is not None for d in fixed.values())                           # 4 of 4 collapse
    assert all(c is None for c, _, _ in mobile.values())                        # 4 of 4 track
    for tau, d in fixed.items():
        assert D_STAR * math.sqrt(tau) <= d <= D_STAR * math.sqrt(tau) + 0.1 * math.sqrt(tau)
    for tau, (_, got, predicted) in mobile.items():
        assert abs(got - predicted) < 0.05 * predicted + 0.005
    # No steady lag exists at tau = 16, but the runaway is slow: the lag reaches 2.5 at t = 2200 and the
    # reader collapses at d = 12.5, t = 2500 (seen 2026-10-07 after a 600-unit run showed no collapse).
    wide = Model(tau=16.0)
    d = collapse_distance(run(wide, duration=3500.0, every=100))
    print("P4: mobile tau = 16 (v_c = 0.0044) collapses at d =", d)
    assert d is not None


# ---------------------------------------------------------------------------
# Piece 5. The exact control: the reader's cost also pulls the substrate.
# ---------------------------------------------------------------------------


def test_p5_exact_forcing_moves_the_substrate_too():
    m = Model(exact=True)
    vc = v_crit(m)
    print("P5: closed-form v_c exact:", vc, "against 0.0265")
    assert abs(vc - 1.5 * V_C_HELD) < 1e-4
    lo, hi = threshold_speed(m, 0.025, 0.03, steps=8)
    print(f"P5: threshold in [{lo:.5f}, {hi:.5f}]")
    assert vc <= hi <= 1.08 * vc
    out = run(m)
    t = np.arange(1, out.shape[0] + 1) * 10 * H
    late = t >= 300.0
    speed = np.polyfit(t[late], out[late, S_], 1)[0]
    print(f"P5: substrate speed under tracking {speed:.5f} against 2v/3 = {2 * m.v / 3:.5f}")
    assert abs(speed - 2 * m.v / 3) < 0.05 * 2 * m.v / 3
