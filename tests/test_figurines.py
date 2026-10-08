"""A composed belief paid in arrears (book plan P37 (b)(i); ``arrangements/figurines.py``).

Predictions fixed before any construction code, in the book repo's
``notes/runs/p37-composed-belief-3f16cdb9/PREDICTIONS.md`` (Claude Fable 5.1, 3f16cdb9, 2026-10-08), the
numbers from ``figurines_closed_form.py`` run before ``figurines.py`` existed; results in the same folder's
``RESULTS.md``. Each test says which of three things it does: check the composite against the closed form
(the construction), check a claim about the model's behaviour, or exhibit a comparison arm.

1. The composite is the closed form, to 1e-10, on every arm and scenario over 300 steps; each figurine's run
   equals its closed form; S1's first outcome by hand: t_hat 22.5, xi_t 4, state forces (-8, 2.03, 20).
2. Neither use nor blame: the vehicle figurine at p = 4 contributes and is pushed e^-10 times as much.
3. Shared blame (the objection, run): S1 under arm A, the bet converges but the route figurine, which was
   right, absorbs 88% of the driver's 4 minutes; end guesses (2.45, 0.53, 23.52).
4. The account separates: S2, each figurine within 0.031 of its truth by round 30.
5. Reuse: S1-trained vehicle and route with a fresh driver predict a new bus 3.55 minutes late; S2-trained,
   0.02.
6. The invitation goes by contribution: S5's first-step placement forces (1.216, 2.128, 12.16), route over
   vehicle 5.714; the route loses more invitation than the vehicle; S6 the same first-step forces.
7. S0: arm A nothing moves; arm B the belief 0.955 late at the round's end, total cost 7.323 < 7.501.
8. S0: arm C the belief true to 1e-12 throughout, margin 0.945, total cost 7.048; arm D the belief runs
   from -0.301 to +0.208 within a round, margin 0.661.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arrangements import figurines as F
from tests import figurines_closed_form as CF

jax.config.update("jax_enable_x64", True)

T_STAR = {name: float(sum(sc.truth)) for name, sc in CF.SCENARIOS.items()}


def model_of(sc: CF.Scenario, theta0=None, p0=None) -> F.Model:
    th = CF.initial_thetas(sc) if theta0 is None else np.asarray(theta0, float)
    return F.Model(theta0=tuple(float(x) for x in th), p0=tuple(sc.p0 if p0 is None else p0),
                   mu_theta=sc.mu_theta, mu_p=sc.mu_p)


def run_both(arm_key: str, sc_key: str, theta0=None, rounds=None):
    """The composite's trajectory and the closed form's, on the same arm and scenario."""
    sc, arm = CF.SCENARIOS[sc_key], CF.ARMS[arm_key]
    R = sc.rounds if rounds is None else rounds
    flags = sc.account if any(sc.account) else None
    arr = F.belief_system(model_of(sc, theta0), F.ARMS[arm_key])
    got, state = F.run_rounds(arr, sc.truth, R, account_flags=flags)
    want = CF.run(arm, sc, theta0=theta0, rounds=R)
    return got, want, state


@pytest.fixture(scope="module")
def runs():
    keys = [("A", "S0"), ("A", "S1"), ("A", "S2"), ("A", "S5"), ("A", "S6"), ("B", "S0"), ("C", "S0"),
            ("D", "S0")]
    return {k: run_both(*k) for k in keys}


def guesses(theta: np.ndarray) -> np.ndarray:
    return np.array([[CF.GUESS[j](th[j])[0] for j in range(3)] for th in np.atleast_2d(theta)])


def test_1_the_composite_is_the_closed_form(runs):
    """Prediction 1 (the construction): states, placements, margin and prediction agree to 1e-10 on every
    arm and scenario; each run equals its closed form; S1's first outcome step by hand."""
    for (arm, sc), (got, want, _) in runs.items():
        for key in ("theta", "p", "m", "t_hat"):
            d = np.abs(got[key] - want[key]).max()
            print(f"arm {arm} {sc}: max |{key}| difference {d:.2e}")
            np.testing.assert_allclose(got[key], want[key], atol=1e-10, rtol=0)
    th0 = CF.initial_thetas(CF.SCENARIOS["S1"])
    for j in range(3):
        g_run, g_closed = float(F.RUNS[j](th0[j])), CF.GUESS[j](th0[j])[0]
        print(f"{F.NAMES[j]}: run {g_run:.12f}, closed form {g_closed:.12f}")
        assert g_run == pytest.approx(g_closed, abs=1e-10)
    arr = F.belief_system(model_of(CF.SCENARIOS["S1"]), F.ARMS["A"])
    s = arr.P.init
    _, xi = F.step_fn(arr)(s, F.exterior(1.0, T_STAR["S1"]))
    t_hat = float(F.prediction_at(s))
    print(f"S1 first outcome: t_hat {t_hat:.4f}, t* {T_STAR['S1']}, xi_t {T_STAR['S1'] - t_hat:.4f}, "
          f"state forces {np.asarray(xi[:3])}")
    assert t_hat == pytest.approx(22.5, abs=1e-10)
    np.testing.assert_allclose(np.asarray(xi[:3]), [-7.99892716, 2.03392537, 20.0], atol=1e-6)


def test_2_neither_use_nor_blame():
    """Prediction 2 (a claim): a figurine out of reach contributes nothing and is pushed by nothing."""
    sc = CF.SCENARIOS["S1"]
    near = F.belief_system(model_of(sc), F.ARMS["A"])
    far = F.belief_system(model_of(sc, p0=(0.0, 4.0, 0.0)), F.ARMS["A"])
    ext = F.exterior(1.0, T_STAR["S1"])
    _, xi_near = F.step_fn(near)(near.P.init, ext)
    _, xi_far = F.step_fn(far)(far.P.init, ext)
    t_near, t_far = float(F.prediction_at(near.P.init)), float(F.prediction_at(far.P.init))
    ratio = float(xi_far[1] / xi_near[1])
    contribution = t_far - (t_near - 0.5)
    print(f"vehicle at p = 4: force ratio {ratio:.6e} (closed form 5.107e-5; e^-10 = {np.exp(-10):.3e}), "
          f"contribution to t_hat {contribution:.6e}, placement force {float(xi_far[4]):.3e}")
    assert ratio == pytest.approx(5.107466e-05, rel=1e-6)
    assert contribution == pytest.approx(0.5 * np.exp(-10), rel=1e-6)
    assert abs(float(xi_far[4])) < 1e-3


def blame_report(sc: CF.Scenario, theta: np.ndarray, p: np.ndarray, xi_theta: np.ndarray) -> str:
    """What each figurine was wrong about, and what it was passed, after one failure."""
    g = guesses(theta)[0]
    dg = np.array([CF.GUESS[j](theta[j])[1] for j in range(3)])
    k = CF.kappa(p)
    t_hat, t_star = float(np.sum(k * g)), float(sum(sc.truth))
    push = t_star - t_hat
    lines = [f"the bus came at {t_star:.2f}; the listener predicted {t_hat:.2f}; the push on the "
             f"prediction is {push:+.3f}"]
    for j, name in enumerate(F.NAMES):
        wrong = g[j] - sc.truth[j]
        lines.append(
            f"  the {name} figurine guessed {g[j]:.3f} for an actual {sc.truth[j]:.2f}: wrong by {wrong:+.3f}; "
            f"its contribution (invitation {k[j]:.3f} times the run's sensitivity {dg[j]:+.3f}) is "
            f"{k[j] * dg[j]:+.3f}; its responsibility, the push times its contribution, is "
            f"{xi_theta[j]:+.3f}; it moves its state by {sc.mu_theta * xi_theta[j]:+.4f}")
    return "\n".join(lines)


def test_3_shared_blame(runs):
    """Prediction 3 (the objection, run): the bet converges, and the right figurine absorbs the mistake."""
    got, _, _ = runs[("A", "S1")]
    sc = CF.SCENARIOS["S1"]
    arr = F.belief_system(model_of(sc), F.ARMS["A"])
    _, xi = F.step_fn(arr)(arr.P.init, F.exterior(1.0, T_STAR["S1"]))
    print("after the first failure (the driver left 4 minutes later than the figurine had it):")
    print(blame_report(sc, got["theta"][0], got["p"][0], np.asarray(xi[:3])))
    g_end = guesses(got["theta"][-1])[0]
    drift = g_end - np.array(sc.truth)
    e_end = got["t_hat"][-1] - T_STAR["S1"]
    print(f"round 30: |e| {abs(e_end):.2e}; end guesses {g_end}; drift from truth {drift}; "
          f"shares of the 4 minutes {drift / 4.0}")
    assert abs(e_end) < 0.05
    np.testing.assert_allclose(g_end, [2.4522, 0.5265, 23.5213], atol=5e-4)
    assert abs(g_end[0] - 6.0) > 0.5 and abs(g_end[2] - 20.0) > 0.5
    assert drift[2] / 4.0 > 0.85


def test_4_the_account_separates(runs):
    """Prediction 4 (a claim): with the account, each figurine comes to its own truth."""
    got, _, _ = runs[("A", "S2")]
    sc = CF.SCENARIOS["S2"]
    g = guesses(got["theta"])
    g_end, drift = g[-1], g[-1] - np.array(sc.truth)
    e_end = got["t_hat"][-1] - T_STAR["S2"]
    print(f"S2 round 30: end guesses {g_end}, drift from truth {drift}, |e| {abs(e_end):.3f}, "
          f"the route's maximal transient drift {np.abs(g[:, 2] - 20.0).max():.3f}")
    np.testing.assert_allclose(g_end, [5.9898, 0.5304, 19.9898], atol=5e-4)
    assert np.abs(drift).max() < 0.1


def test_5_only_the_accounts_figurines_can_be_counted_on_elsewhere(runs):
    """Prediction 5 (a claim): reuse with a fresh driver on a different bus."""
    fresh = CF.SCENARIOS["S0"]
    th_fresh = CF.initial_thetas(fresh)
    for key, expect in (("S1", 3.5478), ("S2", 0.0202)):
        got, _, _ = runs[("A", key)]
        th = np.array([th_fresh[0], got["theta"][-1][1], got["theta"][-1][2]])
        arr = F.belief_system(model_of(fresh, theta0=th), F.ARMS["A"])
        t_hat = float(F.prediction_at(arr.P.init))
        print(f"{key}-trained vehicle and route with a fresh driver: predict {t_hat:.4f} for a bus at 22.5, "
              f"error {t_hat - 22.5:+.4f} (closed form {expect:+.4f})")
        assert t_hat - 22.5 == pytest.approx(expect, abs=5e-4)
    assert runs[("A", "S1")][0]["t_hat"][-1] == pytest.approx(T_STAR["S1"], abs=0.05)


def test_6_the_invitation_goes_by_contribution(runs):
    """Prediction 6 (a claim): the figurine that contributed most is uninvited hardest, right or wrong."""
    for key in ("S5", "S6"):
        got, _, _ = runs[("A", key)]
        sc = CF.SCENARIOS[key]
        arr = F.belief_system(model_of(sc), F.ARMS["A"])
        _, xi = F.step_fn(arr)(arr.P.init, F.exterior(1.0, T_STAR[key],
                                                     (sc.truth, sc.account) if any(sc.account) else None))
        xi_p = np.asarray(xi[3:6])
        k_end, k0 = CF.kappa(got["p"][-1]), CF.kappa(got["p"][0])
        g_end = guesses(got["theta"][-1])[0]
        print(f"{key}: first outcome e {got['t_hat'][0] - T_STAR[key]:+.4f}, placement forces {xi_p}, "
              f"route over vehicle {xi_p[2] / xi_p[1]:.4f}; end invitations {k_end} (from {k0[0]:.4f}); "
              f"end guesses {g_end}; |e| after round 5 at most "
              f"{np.abs(got['t_hat'][50:] - T_STAR[key]).max():.4f}")
        np.testing.assert_allclose(xi_p, [1.216, 2.128, 12.16], atol=2e-3)
        assert xi_p[2] / xi_p[1] == pytest.approx(20.0 / 3.5, rel=1e-6)
        if key == "S5":
            assert got["p"][-1][2] > got["p"][-1][1] > got["p"][-1][0]
            np.testing.assert_allclose(k_end, [0.9597, 0.9582, 0.9397], atol=5e-4)
            assert abs(g_end[1] - 0.5) > 2.0
        else:
            np.testing.assert_allclose(k_end, [0.9622, 0.9603, 0.9680], atol=5e-4)
            assert k_end[2] > k0[2]


def cost_per_round(got, t_star: float, last: int = 5, steps: int = 10) -> float:
    """Inconvenience summed over a round's steps plus the realized loss at its outcome, averaged over the
    last ``last`` rounds."""
    R = (len(got["t_hat"]) - 1) // steps
    total = 0.0
    for r in range(R - last, R):
        sl = slice(steps * r, steps * (r + 1))
        inc = CF.C_INC * CF.softplus(CF.T_PREF - got["t_hat"][sl] - got["m"][sl])
        end = steps * (r + 1) - 1
        total += inc.sum() + 0.5 * (got["t_hat"][end] + got["m"][end] - t_star) ** 2
    return total / last


def test_7_a_belief_that_is_the_bet_is_bent_by_the_plan(runs):
    """Prediction 7 (a claim and a comparison arm): arm A nothing moves; arm B the plan bends the belief
    late, and the listener's total cost is lower for it."""
    true_cost = 10 * CF.C_INC * CF.softplus(CF.T_PREF - 22.5)
    a, _, _ = runs[("A", "S0")]
    e_a = a["t_hat"] - 22.5
    print(f"arm A on S0: max |e| {np.abs(e_a).max():.2e}, max |state change| "
          f"{np.abs(a['theta'] - a['theta'][0]).max():.2e}")
    assert np.abs(e_a).max() < 1e-12
    b, _, _ = runs[("B", "S0")]
    e_b = b["t_hat"] - 22.5
    ends, starts = CF.outcome_steps()[25:], CF.outcome_steps()[25:] - 9
    cost_b = cost_per_round(b, 22.5)
    print(f"arm B on S0: belief late by {e_b[ends].mean():.4f} at the round's end, {e_b[starts].mean():.4f} "
          f"at its start; total cost per round {cost_b:.4f} against {true_cost:.4f} for the true belief")
    assert e_b[ends].mean() == pytest.approx(0.9547, abs=5e-4)
    assert e_b[starts].mean() == pytest.approx(0.3302, abs=5e-4)
    assert cost_b == pytest.approx(7.3226, abs=5e-4) and cost_b < true_cost


def test_8_a_choice_with_matter_of_its_own(runs):
    """Prediction 8 (a comparison arm): with a margin of its own, read one-way, the belief is true at every
    step and the margin takes the lateness; on the exact wire the belief is true only on the round's
    average."""
    ends, starts = CF.outcome_steps()[25:], CF.outcome_steps()[25:] - 9
    c, _, _ = runs[("C", "S0")]
    e_c = c["t_hat"] - 22.5
    cost_c = cost_per_round(c, 22.5)
    print(f"arm C on S0: max |e| {np.abs(e_c).max():.2e}; margin at the round's end {c['m'][ends].mean():.4f}; "
          f"total cost per round {cost_c:.4f}")
    assert np.abs(e_c).max() < 1e-12
    assert c["m"][ends].mean() == pytest.approx(0.9453, abs=5e-4)
    assert cost_c == pytest.approx(7.0478, abs=5e-4)
    d, _, _ = runs[("D", "S0")]
    e_d = d["t_hat"] - 22.5
    print(f"arm D on S0: belief {e_d[starts].mean():+.4f} at the round's start, {e_d[ends].mean():+.4f} at "
          f"its end; margin {d['m'][ends].mean():.4f}; bet late by {(e_d + d['m'])[ends].mean():.4f}; "
          f"total cost per round {cost_per_round(d, 22.5):.4f}")
    assert e_d[starts].mean() == pytest.approx(-0.3013, abs=5e-4)
    assert e_d[ends].mean() == pytest.approx(0.2077, abs=5e-4)
    assert d["m"][ends].mean() == pytest.approx(0.6614, abs=5e-4)
    assert abs(e_d[starts].mean() + e_d[ends].mean()) < 0.1
