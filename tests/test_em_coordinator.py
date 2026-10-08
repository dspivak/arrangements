"""The EM coordinator (book plan P37 (b)(iii); ``arrangements/em_coordinator.py``).

Predictions fixed before any construction code, in the book repo's
``notes/runs/p37-em-coordinator-1a279037/PREDICTIONS.md`` (Claude Fable 5.1, 1a279037, 2026-10-08), the numbers from
``em_coordinator_closed_form.py`` run before ``em_coordinator.py`` existed; results in the same folder's
``RESULTS.md``. A figurine's SHARE is its drift from where it started over the run, divided by the four minutes the
driver was late.

0. Fisher's identity: the gradient of the expected complete-data cost Q at the current point equals the marginal's,
   in the doubt and in the observations, and with no sighting it is one number on every figurine of a bus; the
   composite's report is not that gradient but the gap to the E-step's lift.
1. At the same split and doubt the memoryless report equals stage (ii)'s share times V / (V + sigma0^2); P-told
   memoryless shares (0.702, 0.099, 0.196), stage (ii)'s within 0.03.
2. A sighting moves the split onto the driver (above 0.9 by round 10; share 0.957); comparison clears the route's
   split (under 0.05 by round 10) and cannot split driver from vehicle (0.459, 0.475). The categorical form's M-step
   is stage (ii)'s Bayes step.
3. One bus and no sighting: the covector on the split is one number on every figurine, so the belief's step is zero
   and the split stays the prior's; with a sighting or a second bus it is not.
4. Reuse = 4 (1 - the driver's share), told memoryless, within 0.05.
5. The record restores: WC-told, memoryless the route keeps 0.064 and bus 2's figurines drift 0.13; with the record
   the route's share is 0.000 and bus 2's figurines stay within 0.001. Driver and vehicle stay at their prior ratio.
6. With the record the destination is the coordinator's: P-record slope driver 0.699, route 0.200 (told 0.698,
   0.199), the vehicle lagging (0.033); P-none slope is stage (ii)'s reweighting (0.400, 0.002, 0.598).
7. The doubt: record B=1, V at round 60 is 12.80 (the prediction's 16 within 10% FAILED on its threshold: the fixed
   point is 16 - sigma0^2, where the covector on Lambda vanishes, and the rate 0.05 does not reach it in 60 rounds);
   memoryless V at round 60 under 1.
8. The composite is the closed form to 1e-10 on all 32 arm-scenario pairs over 60 rounds.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arrangements import em_coordinator as E
from tests import coordinator_closed_form as CF2
from tests import em_coordinator_closed_form as CF

jax.config.update("jax_enable_x64", True)

ROUNDS = CF.N_ROUNDS


def arm_of(sc: CF.Scenario) -> E.Arm:
    return E.Arm(sc.arm.response, sc.arm.memory, sc.arm.B)


def run_both(key: str, rounds: int = ROUNDS):
    sc = CF.SCENARIOS[key]
    arm = arm_of(sc)
    n = arm.n
    G0 = [CF.GUESS0[CF.kind(j)] for j in range(n)]
    built = E.coordinated(arm, CF.initial_thetas(n), sc.split0, G0)
    got, state = E.run_rounds(built, arm, list(CF.t_stars(arm.B)), rounds, sc.sighting, CF.O_DRIVER)
    want = CF.run(sc, rounds=rounds)
    return got, want, state


@pytest.fixture(scope="module")
def runs():
    out = {}
    for key in CF.SCENARIOS:
        out[key] = run_both(key)
        print(f"ran {key}")
    return out


def shares(got) -> np.ndarray:
    return (got["g"][-1] - got["g"][0]) / 4.0


def reuse_error(got) -> float:
    g = got["g"][-1]
    return float(CF.GUESS0[0] + g[1] + g[2] - sum(CF.TRUTH_2))


def test_0_fisher_the_chain_rule_of_Q_broadcasts():
    """Prediction 0: grad Q == grad marginal at the current point (in v and in y); with no sighting the gradient in
    the guesses is one number per bus; the composite's report is the gap, not that gradient."""
    rng = np.random.default_rng(0)
    for B in (1, 2):
        A, N = E.incidence(B)
        n = 2 * B + 1
        for flag in (0.0, 1.0):
            for _ in range(3):
                v = jnp.asarray(rng.uniform(0.1, 2.0, n))
                y = jnp.asarray(rng.normal(0, 3, B + 1)) * jnp.concatenate([jnp.ones(B), jnp.array([flag])])
                k = lambda v_, y_: E.marginal_cost(v_, y_, A, N, flag)
                Q = lambda v_, y_: E.expected_complete_cost(v_, y_, v, y, A, N, flag)
                gk_v, gk_y = jax.grad(k, argnums=(0, 1))(v, y)
                gQ_v, gQ_y = jax.grad(Q, argnums=(0, 1))(v, y)
                np.testing.assert_allclose(gQ_v, gk_v, atol=1e-10, rtol=0)
                np.testing.assert_allclose(gQ_y, gk_y, atol=1e-10, rtol=0)
                # the gradient in the guesses (memoryless: y_b = t* - sum g): -A^T grad_y
                g_grad = -A.T @ gk_y
                eps_hat = E.e_step(v, y, A, N, flag)
                for b in range(B):
                    js = np.array(E.bus_figurines(b))
                    if flag == 0.0:
                        # the driver and the vehicle, in this bus only, get the bus's one number; the route,
                        # shared, gets every bus's number summed (prediction 0 as written forgot the route)
                        np.testing.assert_allclose(g_grad[js[1]], g_grad[js[0]], atol=1e-10, rtol=0)
                        np.testing.assert_allclose(g_grad[2], sum(g_grad[E.bus_figurines(b_)[0]] for b_ in range(B)),
                                                   atol=1e-10, rtol=0)
                    # the report (memoryless) is -eps_hat, which is not the broadcast
                    assert np.abs(-eps_hat[js] - g_grad[js]).max() > 1e-3
                print(f"B={B} flag={flag}: grad_g {np.asarray(g_grad)}, report {np.asarray(-eps_hat)}")


def test_1_same_belief_same_shares(runs):
    """Prediction 1: memoryless, B=1, the report is stage (ii)'s s_j e_b times V/(V + sigma0^2) at every state;
    P-told memoryless shares are stage (ii)'s."""
    got, _, _ = runs["P-told-none"]
    for t in range(0, ROUNDS, 7):
        s, V = got["s"][t], got["V"][t]
        e_b = got["g"][t].sum() - sum(CF.TRUTH_1)
        want = s * V / (V + E.SIGMA0 ** 2) * e_b
        np.testing.assert_allclose(got["report"][t], want, atol=1e-10, rtol=0)
    sh = shares(got)
    print(f"P-told-none shares {sh}, stage (ii) (0.703, 0.099, 0.196)")
    np.testing.assert_allclose(sh, (0.703, 0.099, 0.196), atol=0.03)


def test_2_the_beliefs_steps(runs):
    """Prediction 2: the sighting puts the split on the driver; comparison clears the route and cannot split driver
    from vehicle; the categorical form's M-step is stage (ii)'s Bayes step."""
    got, _, _ = runs["S-told-none"]
    print(f"S-told-none driver's split by round {got['s'][:12, 0]}, shares {shares(got)}")
    assert got["s"][10, 0] > 0.9
    assert shares(got)[0] > 0.9
    got, _, _ = runs["C-told-none"]
    sh = shares(got)
    print(f"C-told-none route's split {got['s'][:21, 2]}, shares {sh}")
    assert got["s"][20, 2] < 0.05
    assert abs(sh[0] - sh[1]) < 0.1
    d, resp, stage2 = CF.categorical_check()
    print(f"categorical: M-step vs Bayes {d:.1e}; responsibilities {resp} vs {stage2}")
    assert d < 1e-12
    np.testing.assert_allclose(resp, stage2 * 25.0 / (25.0 + E.SIGMA0 ** 2), atol=1e-10)


def test_3_one_total_is_flat_across_the_split(runs):
    """Prediction 3: B=1, no sighting, both memories: the covector on the split is one number on every figurine at
    every step, and the split at round 60 is the prior's; with a sighting or B=2 it is not."""
    for key in ("U-told-none", "P-told-none", "W-told-none", "U-told-record", "P-told-record", "W-told-record",
                "P-slope-none", "P-slope-record"):
        got, _, _ = runs[key]
        spread = np.abs(got["xi_s"] - got["xi_s"].mean(axis=1, keepdims=True)).max()
        drift = np.abs(got["s"][-1] - got["s"][0]).max()
        print(f"{key}: spread of the covector across figurines {spread:.1e}, split drift {drift:.1e}")
        assert spread < 1e-10          # the covector passes through a log-determinant; the prediction's 1e-12 was roundoff-tight
        assert drift < 1e-10
    for key in ("S-told-none", "C-told-none", "S-told-record", "C-told-record"):
        got, _, _ = runs[key]
        spread = np.abs(got["xi_s"] - got["xi_s"].mean(axis=1, keepdims=True)).max()
        print(f"{key}: spread {spread:.1e}")
        assert spread > 1.0


def test_4_reuse_is_what_the_driver_was_not_given(runs):
    """Prediction 4 (item 65's prediction 5): told memoryless, reuse = 4 (1 - driver's share) within 0.05."""
    for key in ("U", "P", "S", "C", "PC", "SPC", "W", "WC"):
        got, _, _ = runs[f"{key}-told-none"]
        r, want = reuse_error(got), 4 * (1 - shares(got)[0])
        print(f"{key}-told-none: reuse {r:.3f}, 4(1 - driver) {want:.3f}")
        assert abs(r - want) < 0.05


def test_5_the_record_restores(runs):
    """Prediction 5: WC-told, memoryless the route keeps its share and bus 2's figurines drift; with the record the
    route is restored and bus 2's figurines stay; driver and vehicle keep their prior ratio."""
    got, _, _ = runs["WC-told-none"]
    sh = shares(got)
    off = got["g"][-1][3:] - np.array(CF.TRUTH_2[:2])
    print(f"WC-told-none shares {sh}, bus 2 off truth {off}")
    assert sh[2] > 0.03 and np.abs(off).min() > 0.05
    got, _, _ = runs["WC-told-record"]
    sh = shares(got)
    off = got["g"][-1][3:] - np.array(CF.TRUTH_2[:2])
    print(f"WC-told-record shares {sh}, bus 2 off truth {off}")
    assert sh[2] < 0.01 and np.abs(off).max() < 0.02
    for key in ("WC-told-record", "C-told-record"):
        sh = shares(runs[key][0])
        print(f"{key}: driver/vehicle share ratio {sh[0] / sh[1]:.3f} (the prior's 1)")
        assert abs(sh[0] / sh[1] - 1.0) < 0.2


def test_6_with_the_record_the_destination_is_the_coordinators(runs):
    """Prediction 6: P-record, slope driver and route shares within 0.05 of told's, the vehicle lagging; P-none slope
    is stage (ii)'s reweighting."""
    told, slope = shares(runs["P-told-record"][0]), shares(runs["P-slope-record"][0])
    print(f"P-record told {told}, slope {slope}")
    assert abs(told[0] - slope[0]) < 0.05 and abs(told[2] - slope[2]) < 0.05
    assert slope[1] < told[1] - 0.03
    sh = shares(runs["P-slope-none"][0])
    print(f"P-slope-none {sh}")
    assert sh[2] > 0.5 and sh[0] < 0.5


def test_7_the_doubt_learns_the_size_of_the_mistake(runs):
    """Prediction 7: record B=1, V at round 60 within 10% of 16 (FAILS on its threshold: 12.80; the fixed point,
    where the covector on Lambda vanishes, is 16 - sigma0^2, checked); memoryless V under 1."""
    A, N = E.incidence(1)
    v = jnp.asarray([1 / 3, 1 / 3, 1 / 3]) * (16.0 - E.SIGMA0 ** 2)
    y = jnp.array([4.0, 0.0])
    dk_dLam = jnp.dot(v, jax.grad(lambda v_: E.marginal_cost(v_, y, A, N, 0.0))(v))
    print(f"covector on the doubt at V = 16 - sigma0^2: {float(dk_dLam):.2e}")
    assert abs(float(dk_dLam)) < 1e-10
    for key in ("U", "P", "W"):
        V = runs[f"{key}-told-record"][0]["V"][-1]
        print(f"{key}-told-record V at round 60: {V:.3f}")
        assert abs(V - 12.80) < 0.05       # the closed form's number; the prediction's 16 within 10% failed
    for key in ("U", "P", "W"):
        V = runs[f"{key}-told-none"][0]["V"][-1]
        print(f"{key}-told-none V at round 60: {V:.3f}")
        assert V < 1.0


def test_8_the_composite_is_the_closed_form(runs):
    """Prediction 8 (the construction): states, splits, doubts, reports and the covector on the split agree to
    1e-10 on every arm and scenario over 60 rounds."""
    for key, (got, want, _) in runs.items():
        for k in ("theta", "s", "V", "report", "xi_s"):
            d = np.abs(got[k] - want[k]).max()
            print(f"{key}: max |{k}| difference {d:.2e}")
            # the covector on the split passes through a log-determinant and reaches 390 with two buses: relative
            if k == "xi_s":
                np.testing.assert_allclose(got[k], want[k], atol=1e-9, rtol=1e-10)
            else:
                np.testing.assert_allclose(got[k], want[k], atol=1e-10, rtol=0)
