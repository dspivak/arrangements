"""A listener who hears accounts and decides whom to count on (book plan P37; ``arrangements/listener.py``).

Predictions fixed before any construction code, in the book repo's
``notes/runs/p37-listener-f2836b6c/PREDICTIONS.md`` (Claude Fable 5.1, f2836b6c, 2026-10-07), verbatim in
short here; results in the same folder's ``RESULTS.md``. Every test says which of three things it does:
check the composite against a closed form (the construction), check a claim about the model's behaviour,
or exhibit a comparison arm. The closed form, ``closed_form`` below, is 30 lines of numpy with the same
forces written by hand; nothing builds the composite from it.

1. The composite is the closed form: placement and belief over 100 steps to 1e-8 on every script; the
   placement force at the initial state is -0.926 (toward the source).
2. Whom the listener keeps, and whom it leaves: A ends at p < 0.01, belief (0.761, 0.198, 0.042); B at
   p < 0.01, belief (0.0001, 0.998, 0.002); C at p = 2.93 +- 0.05, kappa < 0.01, belief (0.002, 0.032,
   0.966); C's kappa below 0.25 by step 30 and 0.05 by step 40. (Narrowed 2026-10-08: that B is kept rests
   on the stipulated loss ``ell_2 < ell_3``, "one can plan around an honest late source," which David does
   not hold; what holds is that the belief tells B and C apart and the placement follows the losses.)
3. Leaving stops the learning: C's third and fourth X outcomes arrive at kappa about 0.21 and 0.044, so
   its final belief in z_3 is 0.966 where full evidence gives 0.99998.
4. The valuation adds nothing to the belief on the one-way wire: at every outcome step the log-heights
   move by kappa (log pi_o - <y, log pi_o>) to 1e-10, and not at all on a no-news step.
5. On the exact wire the belief goes where it is cheapest to go: on C the belief in z_1 exceeds 0.999 by
   step 10 and at step 100; the listener ends at p < 0.01 on the unaccountable source. The same on A.
6. The exact likelihood forcing pushes the listener away from a surprising account: one step from the
   start on outcome L, the two arms' placements differ by 0.1328 away from the source, to 1e-8.
7. A repeated account counts twice: C with the round-2 X sent again on step 11 moves the z_3 : z_2
   log-odds by 5.543 at step 12 against 2.771, to 1e-8.
8. (Added after David's answer, before code; PREDICTIONS.md's last section.) The checker that knows the push
   gives the same belief as the wire that carries none: ``wire="subtract"`` agrees with ``wire="oneway"`` on
   every script to 1e-10 over 100 steps.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arrangements.belief import belief_of
from arrangements.listener import (
    DEFAULT, EMPTY, EXT, LINED_UP, NOT_LINED_UP, OUTCOME, listener_system, log_heights, one_hot,
    placement, run_script, set_signal, step_fn,
)

jax.config.update("jax_enable_x64", True)

SCRIPTS = {"A": "KKKKLKKKKK", "B": "KLKLKKLKLK", "C": "KXKXKKXKXK"}
PI = np.asarray(DEFAULT.pi)
LOGPI = np.log(PI)
ELL = np.asarray(DEFAULT.ell)
C, TAU, MU = DEFAULT.c, DEFAULT.tau, DEFAULT.mu_p


def kappa(p):
    return np.exp(-p ** 2 / (2.0 * TAU))


def dkappa(p):
    return -p / TAU * kappa(p)


def softmax(l):
    e = np.exp(l - l.max())
    return e / e.sum()


def closed_form(script, wire="oneway", likelihood="unilateral", repeats=(), K=10):
    """The same forces by hand: tempered Bayes on the belief, descent of the expected loss on the
    placement; the two comparison arms add the valuation's pull on the belief and the surprisal's push on
    the placement. Returns (placements, beliefs) at steps 0..T."""
    extra = dict(repeats)
    l, p = np.zeros(3), DEFAULT.p0
    ps, ys = [], []
    for t in range(K * len(script)):
        r, s = divmod(t, K)
        letter = extra.get(t, script[r] if s == 0 else None)
        h = np.zeros(3)
        if letter:
            h[OUTCOME[letter]] = 1.0
        y, k = softmax(l), kappa(p)
        ps.append(p)
        ys.append(y)
        xi_y = k * (LOGPI @ h)
        if wire == "exact":
            xi_y = xi_y - k * ELL
        xi_p = -dkappa(p) * (y @ ELL - C)
        if likelihood == "exact":
            xi_p = xi_p - dkappa(p) * (y @ (-LOGPI @ h))
        l = l + (xi_y - y @ xi_y)
        p = p + MU * xi_p
    ps.append(p)
    ys.append(softmax(l))
    return np.array(ps), np.array(ys)


@pytest.fixture(scope="module")
def runs():
    arr = listener_system()
    return {name: run_script(arr, sc) for name, sc in SCRIPTS.items()}


def test_1_the_composite_is_the_closed_form(runs):
    """Prediction 1 (the construction): composition in arr gives the forces written by hand."""
    arr = listener_system()
    s = set_signal(arr.P.init, one_hot(LINED_UP))
    xi_q, _, _ = arr.omega(arr.P.position(s), EMPTY, EXT)
    force = float(xi_q[6])          # the listener's placement coordinate in the parent's configuration
    expected = -dkappa(DEFAULT.p0) * (np.full(3, 1 / 3) @ ELL - C)
    print(f"placement force at start {force:.6f}, by hand {expected:.6f}")
    assert force == pytest.approx(expected, abs=1e-10)
    assert force == pytest.approx(-0.926, abs=1e-3)
    for name, (ps, ys, _) in runs.items():
        cp, cy = closed_form(SCRIPTS[name])
        print(f"script {name}: max |p - closed| {np.abs(ps - cp).max():.2e}, "
              f"max |y - closed| {np.abs(ys - cy).max():.2e}")
        np.testing.assert_allclose(ps, cp, atol=1e-8, rtol=0)
        np.testing.assert_allclose(ys, cy, atol=1e-8, rtol=0)


def test_2_whom_the_listener_keeps_and_whom_it_leaves(runs):
    """Prediction 2 (the claim): the listener leaves the source whose accounts do not line up and stays
    with the one that is as unreliable but accountable."""
    for name, (ps, ys, _) in runs.items():
        print(f"script {name}: end p {ps[-1]:.4f}, kappa {kappa(ps[-1]):.4f}, belief {np.round(ys[-1], 4)}")
    pA, yA, _ = runs["A"]
    pB, yB, _ = runs["B"]
    pC, yC, _ = runs["C"]
    assert pA[-1] < 0.01 and pB[-1] < 0.01
    np.testing.assert_allclose(yA[-1], [0.761, 0.198, 0.042], atol=5e-4)
    np.testing.assert_allclose(yB[-1], [0.0001, 0.998, 0.002], atol=5e-4)
    assert pC[-1] == pytest.approx(2.93, abs=0.05) and kappa(pC[-1]) < 0.01
    np.testing.assert_allclose(yC[-1], [0.002, 0.032, 0.966], atol=5e-4)
    print(f"script C: kappa at step 30 {kappa(pC[30]):.4f}, at step 40 {kappa(pC[40]):.4f}")
    assert kappa(pC[30]) < 0.25 and kappa(pC[40]) < 0.05


def test_3_leaving_stops_the_learning(runs):
    """Prediction 3 (a consequence of the tempering): the listener that has left learns little more.

    **The kernel-weight clause failed as written** (2026-10-07): the prediction said the third and fourth X
    outcomes arrive at kappa about 0.21 and 0.044, numbers read off the closed form at steps 30 and 40;
    but script C's X outcomes are on steps 10, 30, 60 and 80, so 0.21 is the SECOND X's weight and step
    40 carries no outcome. Measured: the second X at 0.2086, the third at 0.0130, the fourth at 0.0070.
    The claim the numbers were attached to held: final z_3 0.96588 against 0.99998 with full evidence.
    The assertion below is the measured clause; the prediction file is untouched."""
    pC, yC, _ = runs["C"]
    k2, k3, k4 = kappa(pC[30]), kappa(pC[60]), kappa(pC[80])
    full = softmax(np.array([LOGPI[i] @ np.array([6, 0, 4]) for i in range(3)]))
    print(f"script C: second X at kappa {k2:.4f}, third at {k3:.4f}, fourth at {k4:.4f}; "
          f"final z_3 {yC[-1][2]:.5f}, with full evidence {full[2]:.5f}")
    assert k2 == pytest.approx(0.2086, abs=5e-4)
    assert k3 == pytest.approx(0.0130, abs=5e-4) and k4 == pytest.approx(0.0070, abs=5e-4)
    assert yC[-1][2] == pytest.approx(0.966, abs=5e-4) and full[2] > 0.9999


def test_4_the_valuation_adds_nothing_to_the_belief(runs):
    """Prediction 4 (REUSE check (i)): on the one-way wire the belief's step is Bayes tempered by kappa
    at an outcome step and zero at a no-news step, whatever the losses."""
    arr = listener_system()
    step = step_fn(arr)
    worst_outcome, worst_quiet = 0.0, 0.0
    for name, sc in SCRIPTS.items():
        s = arr.P.init
        for t in range(100):
            r, k = divmod(t, 10)
            letter = sc[r] if k == 0 else None
            s = set_signal(s, one_hot(OUTCOME[letter]) if letter else jnp.zeros(3))
            l0, p0 = np.asarray(log_heights(s)), float(placement(s))
            s, _ = step(s)
            dl = np.asarray(log_heights(s)) - l0
            if letter:
                lp = LOGPI[:, OUTCOME[letter]]
                expected = kappa(p0) * (lp - softmax(l0) @ lp)
                worst_outcome = max(worst_outcome, np.abs(dl - expected).max())
            else:
                worst_quiet = max(worst_quiet, np.abs(dl).max())
    print(f"worst outcome-step error {worst_outcome:.2e}; worst no-news move {worst_quiet:.2e}")
    assert worst_outcome < 1e-10 and worst_quiet < 1e-12


def test_5_on_the_exact_wire_the_belief_goes_where_it_is_cheapest(runs):
    """Prediction 5 (comparison arm): a listener whose valuation pushes on its belief comes to believe whatever
    is cheapest to believe, the push proportional to how much it already counts on the source."""
    arr = listener_system(wire="exact")
    for name in ("C", "A"):
        ps, ys, _ = run_script(arr, SCRIPTS[name])
        cp, cy = closed_form(SCRIPTS[name], wire="exact")
        np.testing.assert_allclose(ps, cp, atol=1e-8, rtol=0)
        np.testing.assert_allclose(ys, cy, atol=1e-8, rtol=0)
        print(f"exact wire, script {name}: z_1 at step 10 {ys[10][0]:.6f}, at 100 {ys[-1][0]:.6f}, "
              f"end p {ps[-1]:.4f}")
        assert ys[10][0] > 0.999 and ys[-1][0] > 0.999 and ps[-1] < 0.01
    pC, yC, _ = runs["C"]
    print(f"one-way wire, script C, for comparison: z_1 at step 100 {yC[-1][0]:.6f}, end p {pC[-1]:.4f}")


def test_6_the_exact_likelihood_pushes_the_listener_away_from_a_surprise():
    """Prediction 6 (REUSE check (iii), comparison arm): the surprisal's push on the placement, through
    the transfer, is mu |kappa'(p_0)| <y, -log pi_L> on the step the account arrives."""
    s_uni = set_signal(listener_system().P.init, one_hot(LINED_UP))
    s_ex = set_signal(listener_system(likelihood="exact").P.init, one_hot(LINED_UP))
    p_uni = float(placement(step_fn(listener_system())(s_uni)[0]))
    p_ex = float(placement(step_fn(listener_system(likelihood="exact"))(s_ex)[0]))
    S = np.full(3, 1 / 3) @ (-LOGPI[:, LINED_UP])
    expected = MU * abs(dkappa(DEFAULT.p0)) * S
    print(f"one step on L from p = 0.5: unilateral {p_uni:.6f}, exact {p_ex:.6f}, "
          f"difference {p_ex - p_uni:.6f}, by hand {expected:.6f}")
    assert p_ex - p_uni == pytest.approx(expected, abs=1e-8)
    assert expected == pytest.approx(0.1328, abs=5e-4)
    assert p_uni < DEFAULT.p0 < p_ex


def test_7_a_repeated_account_counts_twice():
    """Prediction 7 (REUSE check (ii)): the arrangement has no notion of the same report."""
    arr = listener_system()
    ps1, ys1, _ = run_script(arr, SCRIPTS["C"])
    ps2, ys2, _ = run_script(arr, SCRIPTS["C"], repeats=[(11, "X")])
    lo = lambda y: np.log(y[2] / y[1])
    single, double = lo(ys1[12]) - lo(ys1[10]), lo(ys2[12]) - lo(ys2[10])
    expected = (kappa(ps2[10]) + kappa(ps2[11])) * (LOGPI[2, NOT_LINED_UP] - LOGPI[1, NOT_LINED_UP])
    print(f"z_3 : z_2 log-odds moved {single:.4f} by one report, {double:.4f} by the report twice; "
          f"(kappa_10 + kappa_11) log-likelihood ratio {expected:.4f}")
    assert single == pytest.approx(2.771, abs=5e-4)
    assert double == pytest.approx(expected, abs=1e-8) and double == pytest.approx(5.543, abs=5e-4)


def test_8_the_checker_that_knows_the_push_matches_the_one_way_wire(runs):
    """Prediction 8 (built to hold): David's internal checker, a part that notices the preference's push and
    cancels it, is the cascade's subtracting wire; it needs to know the losses and the reliance, and then its
    belief is the one-way wire's, to 1e-10. The exact wire is printed beside them for the size of the push.

    The checker is its own box since P43 (b) job 8 (``listener.checker_box``, between the belief, sent forward
    plainly, and the valuation; de71554a); before, it was folded into the belief box's forcing
    (``wire="subtract"``), checked equal to it here."""
    folded = listener_system(wire="subtract")
    arr = listener_system(wire="checker")
    for sc in SCRIPTS.values():
        np.testing.assert_allclose(run_script(arr, sc)[1], run_script(folded, sc)[1], atol=1e-12, rtol=0)
    for name, sc in SCRIPTS.items():
        ps, ys, _ = run_script(arr, sc)
        p1, y1, _ = runs[name]
        print(f"script {name}: checker vs one-way, max |p| diff {np.abs(ps - p1).max():.2e}, "
              f"max |y| diff {np.abs(ys - y1).max():.2e}")
        np.testing.assert_allclose(ps, p1, atol=1e-10, rtol=0)
        np.testing.assert_allclose(ys, y1, atol=1e-10, rtol=0)
