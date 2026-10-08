"""The coordinator that distributes the mistake (book plan P37 (b)(ii); ``arrangements/coordinator.py``).

Predictions fixed before any construction code, in the book repo's
``notes/runs/p37-coordinator-70fcc5f2/PREDICTIONS.md`` (Claude Fable 5.1, 70fcc5f2, 2026-10-08), the numbers
from ``coordinator_closed_form.py`` run before ``coordinator.py`` existed; results in the same folder's
``RESULTS.md``. Each test says which of three things it does: check the composite against the closed form (the
construction), check a claim about the model's behaviour, or exhibit a comparison arm. A figurine's SHARE is its
drift from where it started over the run, divided by the four minutes the driver was late.

1. The composite is the closed form, to 1e-10, on every arm and scenario over 60 rounds; the slope figurines under
   the chain rule reproduce item 65's round-30 guesses (2.452, 0.527, 23.521).
2. A sighting of the driver every round: told, the driver's share 0.917 (the prediction's 0.95 failed: round 1's
   report precedes round 1's sighting), the vehicle and route 0.043 and 0.039; slope, the driver 0.792.
3. The prior alone: told, shares (0.703, 0.099, 0.196), the prior's; slope, (0.400, 0.002, 0.598), the prior
   reweighted by the figurines' sensitivities.
4. Comparison clears the route (under 0.01 by round 3) and cannot tell the driver from the vehicle: told
   (0.475, 0.492, 0.031); with the prior added, (0.857, 0.121, 0.020).
5. The driver's share ranks S > PC > P > C > U, told.
6. No evidence: told, even thirds; slope, item 65's shares within 0.011.
7. The objection: the wrong prior alone sends 0.799 to the route and nothing in the bus's data moves the belief;
   with comparison the route is cleared by round 4 but keeps 0.065, and bus 2's own figurines cover the rest.
8. Reuse is 4 (1 - the driver's share), told, within 0.01, in every scenario.
"""

import jax
import numpy as np
import pytest

from arrangements import coordinator as C
from tests import coordinator_closed_form as CF

jax.config.update("jax_enable_x64", True)

ROUNDS = CF.N_ROUNDS


def arm_of(sc: CF.Scenario) -> C.Arm:
    return C.Arm(sc.arm.response, sc.arm.B, sc.arm.blame)


def t_star_of(B: int):
    return [sum(CF.TRUTH_1) if b == 0 else sum(CF.TRUTH_2) for b in range(B)]


def run_both(key: str, rounds: int = ROUNDS):
    """The composite's trajectory and the closed form's, on the same scenario."""
    sc = CF.SCENARIOS[key]
    arm = arm_of(sc)
    built = C.coordinated(arm, CF.initial_thetas(arm.n), sc.priors)
    sighting = (0, sc.sighting) if sc.sighting is not None else None
    got, state = C.run_rounds(built, arm, t_star_of(arm.B), rounds, sighting=sighting)
    want = CF.run(sc, rounds=rounds)
    return got, want, state


@pytest.fixture(scope="module")
def runs():
    return {key: run_both(key) for key in CF.SCENARIOS}


def shares(got) -> np.ndarray:
    return (got["g"][-1] - got["g"][0]) / 4.0


def reuse_error(got) -> float:
    """Bus 1's trained vehicle and route with a fresh driver on a bus (2, 1/2, 20): the first prediction's
    error, read off the guesses (a fresh driver figurine guesses 2)."""
    g = got["g"][-1]
    return float(CF.GUESS0[0] + g[1] + g[2] - sum(CF.TRUTH_2))


def test_1_the_composite_is_the_closed_form(runs):
    """Prediction 1 (the construction): states, beliefs, predictions and reports agree to 1e-10 on every arm
    and scenario; the slope figurines under the chain rule are item 65 at round 30."""
    for key, (got, want, _) in runs.items():
        for k in ("theta", "y", "t_hat", "report"):
            d = np.abs(got[k] - want[k]).max()
            print(f"{key}: max |{k}| difference {d:.2e}")
            np.testing.assert_allclose(got[k], want[k], atol=1e-10, rtol=0)
    got, _, _ = run_both("R-slope", rounds=30)
    print(f"R-slope at round 30: guesses {got['g'][-1]} (item 65: 2.4522, 0.5265, 23.5213)")
    np.testing.assert_allclose(got["g"][-1], [2.4522, 0.5265, 23.5213], atol=5e-4)


def report_lines(got, sc: CF.Scenario, t: int) -> str:
    """What the depot believed, and what it told each of bus 1's figurines, at round ``t``."""
    y, rep, e = got["y"][t][0], got["report"][t], got["e"][t][0]
    lines = [f"  round {t}: bus 1's mistake {e:+.3f}; the depot believes (driver, vehicle, route) wrong with "
             f"({y[0]:.3f}, {y[1]:.3f}, {y[2]:.3f}) and tells them they were wrong by "
             f"({rep[0]:+.3f}, {rep[1]:+.3f}, {rep[2]:+.3f}) minutes"]
    return "\n".join(lines)


def test_2_a_sighting_gives_the_driver_nearly_all_of_it(runs):
    """Prediction 2 (a claim; its thresholds failed in the closed form, the numbers here are the closed form's):
    with the listener sighting the driver every round the belief is 0.9 on the driver after one round and 0.994
    after two, and the driver takes 0.917 of the four minutes, told; the first round's third each went out
    before the first sighting landed."""
    got, _, _ = runs["S-told"]
    sc = CF.SCENARIOS["S-told"]
    print("S-told, a sighting of the driver every round:")
    for t in range(3):
        print(report_lines(got, sc, t))
    sh = shares(got)
    print(f"  shares at round 60 {sh}; reuse error {reuse_error(got):+.3f}")
    np.testing.assert_allclose(got["y"][1][0], [0.9, 0.05, 0.05], atol=1e-6)
    np.testing.assert_allclose(got["y"][2][0][0], 0.9939, atol=5e-4)
    np.testing.assert_allclose(sh, [0.917, 0.043, 0.039], atol=2e-3)
    assert sh[0] < 0.95   # the prediction's threshold, failed as written
    got_s, _, _ = runs["S-slope"]
    sh_s = shares(got_s)
    print(f"S-slope: shares {sh_s}; reuse error {reuse_error(got_s):+.3f}")
    np.testing.assert_allclose(sh_s, [0.792, 0.002, 0.206], atol=2e-3)


def test_3_the_prior_alone(runs):
    """Prediction 3 (a claim and a comparison arm): told, the shares are the prior's; slope, the prior is
    reweighted by the sensitivities and the route takes most."""
    got, _, _ = runs["P-told"]
    sh = shares(got)
    print(f"P-told: shares {sh} (prior {CF.PRIOR}); reuse error {reuse_error(got):+.3f}")
    np.testing.assert_allclose(sh, CF.PRIOR, atol=0.04)
    np.testing.assert_allclose(sh, [0.703, 0.099, 0.196], atol=2e-3)
    got_s, _, _ = runs["P-slope"]
    sh_s = shares(got_s)
    print(f"P-slope: shares {sh_s}; reuse error {reuse_error(got_s):+.3f}")
    np.testing.assert_allclose(sh_s, [0.400, 0.002, 0.598], atol=2e-3)
    assert sh_s[0] < 0.5 < sh_s[2]


def test_4_comparison_clears_the_route(runs):
    """Prediction 4 (a claim): with a second bus on the same route and on time, the belief on the route is under
    0.01 by round 3, and the rest is split between the driver and the vehicle, which the comparison cannot
    tell apart; the prior then splits them."""
    for key, expect, expect_y in (("C-told", [0.475, 0.492, 0.031], [0.5, 0.5, 0.0]),
                                  ("PC-told", [0.857, 0.121, 0.020], [0.875, 0.125, 0.0])):
        got, _, _ = runs[key]
        yr = got["y"][:, 0, 2]
        first = int(np.argmax(yr < 0.01))
        sh = shares(got)
        print(f"{key}: belief on the route by round {yr[:5]}, under 0.01 at round {first}; shares {sh[:3]}, "
              f"bus 2's (driver, vehicle) {sh[3:]}; belief at the end {got['y'][-1][0]}; reuse error "
              f"{reuse_error(got):+.3f}")
        assert first <= 3
        np.testing.assert_allclose(sh[:3], expect, atol=2e-3)
        np.testing.assert_allclose(got["y"][-1][0], expect_y, atol=1e-3)


def test_5_the_kinds_rank_as_david_ranked_them(runs):
    """Prediction 5 (a claim): the driver's share, told: S > PC > P > C > U, and SPC within 0.02 of S or above."""
    d = {k: shares(runs[f"{k}-told"][0])[0] for k in ("S", "PC", "P", "C", "U", "SPC")}
    print("the driver's share, told: " + ", ".join(f"{k} {v:.3f}" for k, v in d.items()))
    assert d["S"] > d["PC"] > d["P"] > d["C"] > d["U"]
    assert d["SPC"] >= d["S"] - 0.02


def test_6_no_evidence_is_an_even_split_and_item_65_is_the_slopes(runs):
    """Prediction 6 (a claim and a comparison arm): a coordinator with nothing to go on splits the mistake
    evenly; the slope figurines turn the even split into item 65's shares."""
    got, _, _ = runs["U-told"]
    sh = shares(got)
    print(f"U-told: shares {sh}; reuse error {reuse_error(got):+.3f}")
    assert np.all((sh > 0.30) & (sh < 0.37))
    got_s, _, _ = runs["U-slope"]
    sh_s = shares(got_s)
    print(f"U-slope: shares {sh_s} (item 65 at round 30: 0.113, 0.007, 0.880); reuse error "
          f"{reuse_error(got_s):+.3f}")
    np.testing.assert_allclose(sh_s, [0.113, 0.007, 0.880], atol=0.02)
    assert sh_s[2] > 0.8


def test_7_the_objection_run(runs):
    """Prediction 7 (the objection, run): a wrong prior blames the route and nothing in the bus's own data moves
    the belief; comparison clears the route but does not restore it, and bus 2's own figurines cover what it
    kept."""
    got, _, _ = runs["W-told"]
    sh = shares(got)
    y = got["y"][:, 0, :]
    print(f"W-told: shares {sh}; |e| at the end {abs(got['e'][-1][0]):.1e}; the belief's largest move "
          f"{np.abs(y - y[0]).max():.1e}; reuse error {reuse_error(got):+.3f}")
    np.testing.assert_allclose(sh, [0.099, 0.100, 0.799], atol=2e-3)
    assert abs(got["e"][-1][0]) < 0.05
    assert np.abs(y - y[0]).max() < 1e-12
    assert reuse_error(got) > 3.4
    got, _, _ = runs["WC-told"]
    sh = shares(got)
    yr = got["y"][:, 0, 2]
    print(f"WC-told: belief on the route by round {yr[:6]}; shares {sh[:3]}, bus 2's (driver, vehicle) {sh[3:]}; "
          f"reuse error {reuse_error(got):+.3f}")
    assert int(np.argmax(yr < 0.05)) <= 6
    np.testing.assert_allclose(sh, [0.460, 0.473, 0.065, -0.032, -0.032], atol=2e-3)
    assert sh[2] > 0.03 and sh[2] < 0.10        # kept, not restored; the prediction's 0.10 failed as written
    assert np.all(np.abs(sh[3:]) * 4 > 0.05)
    got_s, _, _ = runs["WC-slope"]
    print(f"WC-slope: shares {shares(got_s)}; reuse error {reuse_error(got_s):+.3f}")


def test_8_reuse_goes_by_what_the_driver_was_not_given(runs):
    """Prediction 8 (a claim): told, the reuse error is 4 (1 - the driver's share) in every scenario."""
    for key in ("S", "PC", "P", "C", "U", "W", "SPC", "WC"):
        got, _, _ = runs[f"{key}-told"]
        err, by_share = reuse_error(got), 4.0 * (1.0 - shares(got)[0])
        print(f"{key}-told: reuse error {err:+.3f}, 4 (1 - driver's share) {by_share:+.3f}")
        assert err == pytest.approx(by_share, abs=0.05)
    errs = [reuse_error(runs[f"{k}-told"][0]) for k in ("S", "PC", "P", "C", "U", "W")]
    assert errs == sorted(errs)
