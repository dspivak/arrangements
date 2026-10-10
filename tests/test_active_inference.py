"""Active inference: three tests (book ``AGENTDOCPLAN.md`` §8.2 P5 (b)).

Predictions fixed in writing, with their numbers, before this file or
``arrangements/active_inference.py`` existed: ``notes/ACTIVE-INFERENCE-TESTS.md`` §§1-3. Each test
below asserts a prediction at the tolerance it was stated with; a failure is a refuted
prediction, reported in that note's §4, not repaired here.

WHAT THE RUN FOUND (verdicts and numbers in the note, §§4-6). Test (i) held in all four parts:
precision settles at the inverse ML variance, collapses to zero without the log-det term, learns
the inverse covariance in 2-D, and the learned precisions of two sources are the weights on the
mean. Test (ii): the chooser's cost is the policy free energy (held), but the chooser does NOT
settle on the policy posterior (refuted, strict xfail): with values ``-G`` the log-partition is
below 1, the simplex repels at the factor ``1 - dt T (log Z - 1)`` per step, and the state goes to
the unnormalized Gibbs weights. Addendum A, written after that failure and before further code:
a constant shift of the values, which leaves the logit unchanged, restores it. Test (iii): the
rollout by ``Phi`` equals the closed form and the information gain is the book note's term
(held); with the unshifted chooser the choices are refuted by the same drift (strict xfail);
shifted, the switch from looking to pushing sits where the closed form puts it, and exact wiring
bends the belief 0.384 toward the preferred outcome while unilateral wiring leaves it on the
evidence (held).

G BY PARTS (plan P43 (b) job 6, Claude Opus 5.5, de71554a, 2026-10-10, at David's "A seems much better …
is the looking part using figurines? Can we do it that way?"). Every look-to-push run now composes
``ai.figurine_planner``: the belief box, copied to a figurine of the world per policy (the world's own step
from the belief under the policy's push), a score box per step (risk plus ambiguity), and the chooser.
Until then the planner was one box running those futures in its sending map, and the look-to-push
switch fed it ``G`` computed in test code. The one-box agent is kept and checked equal (``one_box_agent``).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from arrangements import active_inference as ai


X1 = 0.5 * jnp.array([0.3, -1.1, 0.8, 2.0, -0.4, 0.9, 1.5])
X2 = jnp.array([[0.2, 1.0], [1.3, 0.4], [-0.7, -0.2], [0.9, 1.8],
                [0.1, -0.9], [1.6, 0.7], [-0.3, 0.5], [0.5, 1.1]])
XA = 1 + 0.3 * jnp.array([0.2, -0.9, 1.1, 0.4, -1.3, 0.6])
XB = 1 + jnp.array([0.6, -1.1, 0.9, -0.5, 1.4, -1.2])


def rel(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# ---------------------------------------------------------------------------
# Test (i): precision learned by descent.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("eta", [0.2, 0.1])
def test_i_a_precision_settles_at_inverse_ml_variance(eta):
    """(i-a) mu* = xbar, Pi* = 1/S (ML variance), rel. error < 1e-10, independent of eta."""
    s = ai.run(ai.precision_learner(X1, eta), 4000)
    assert rel(s[0], 0.2857142857142857) < 1e-10
    assert rel(jnp.exp(s[1]), 4.073150457190358) < 1e-10
    # not the unbiased variance's inverse
    n = X1.shape[0]
    assert abs(float(jnp.exp(s[1])) - 4.073150457190358 * (n - 1) / n) > 0.5


def test_i_b_without_normalizer_precision_collapses():
    """(i-b) Pi decreases at every step; Pi_n eta S n / 2 within 2% of 1 at n = 20000; mu -> xbar."""
    eta, n = 0.2, 20000
    s, ells = ai.run(ai.precision_learner(X1, eta, normalizer=False), n,
                     record=lambda s: float(s[1]))
    ells = np.array(ells)
    assert np.all(np.diff(ells) < 0)
    S = 0.24551020408163265
    ratio = float(np.exp(ells[-1])) * eta * S * n / 2
    print("ratio", ratio, "Pi_n", np.exp(ells[-1]), "mu_n", float(s[0]))
    assert abs(ratio - 1) < 0.02
    assert abs(float(s[0]) - 0.2857142857142857) < 1e-3


def test_i_b_prime_two_dimensional_log_det():
    """(i-b') mu* = (0.45, 0.55), Pi* = Shat^{-1}, rel. Frobenius error < 1e-10."""
    s = ai.run(ai.precision_learner_2d(X2, 0.1), 20000)
    Pi = ai.cholesky_precision(s[2:])
    print("mu", s[:2], "Pi", Pi)
    assert rel(s[:2], [0.45, 0.55]) < 1e-10
    target = [[2.259419664750335, -0.9122761407883151],
              [-0.9122761407883151, 2.0419859731634826]]
    assert rel(Pi, target) < 1e-10


def test_i_c_two_sources_learned_precisions_are_attention_weights():
    """(i-c) mu* = 1.0056988365525892, Pi_A = 15.618779730059675, Pi_B = 0.9951808214273963."""
    s = ai.run(ai.two_source_learner(XA, XB, 0.05), 40000)
    print("mu", float(s[0]), "PiA", float(jnp.exp(s[1])), "PiB", float(jnp.exp(s[2])))
    assert rel(s[0], 1.0056988365525892) < 1e-9
    assert rel(jnp.exp(s[1]), 15.618779730059675) < 1e-9
    assert rel(jnp.exp(s[2]), 0.9951808214273963) < 1e-9
    wA = float(jnp.exp(s[1]) / (jnp.exp(s[1]) + jnp.exp(s[2])))
    assert rel(s[0], wA * XA.mean() + (1 - wA) * XB.mean()) < 1e-9
    assert abs(float(s[0]) - 1.0108333333333335) > 1e-3


# ---------------------------------------------------------------------------
# Test (ii): policy selection is the chooser of book §5.9 item 28.
# ---------------------------------------------------------------------------

from arrangements import choice  # noqa: E402
from arrangements.wiring import compose_seq  # noqa: E402

G_II = jnp.array([1.3, 0.4, 2.1])
E_II = jnp.array([0.2, 0.5, 0.3])
Q_HABIT = [0.06087552357274065, 0.9206886420666024, 0.01843583436065685]
Q_FLAT = [0.13790166651764385, 0.8342564671867838, 0.02784186629557244]


REFUTED = ("refuted as written (notes/ACTIVE-INFERENCE-TESTS.md §4): log Z < 1, so the simplex "
           "repels and the chooser settles on the unnormalized Gibbs weights")


@pytest.mark.xfail(strict=True, reason=REFUTED)
@pytest.mark.parametrize("E,target", [(E_II, Q_HABIT), (None, Q_FLAT)])
def test_ii_a_chooser_reproduces_policy_posterior(E, target):
    """(ii-a) logit_arrangement(v = -G + T log E, T = 1/gamma) settles at q, rel. error < 1e-12."""
    v, T = ai.policy_values(G_II, 2.0, E)
    s = ai.run(choice.logit_arrangement(v, T, dt=0.3), 3000)
    print("q", s)
    assert rel(s, target) < 1e-12


def test_ii_b_chooser_cost_is_policy_free_energy():
    """(ii-b) k(p) - k(q) = T KL(p || q) at five seeded interior p, to 1e-12."""
    T = 0.5
    k = ai.policy_free_energy(G_II, T, E_II)
    q = jnp.array(Q_HABIT)
    rng = np.random.default_rng(0)
    for _ in range(5):
        p = jnp.array(rng.dirichlet(np.ones(3)))
        kl = float(jnp.sum(p * jnp.log(p / q)))
        assert abs(float(k(p) - k(q)) - T * kl) < 1e-12


@pytest.mark.xfail(strict=True, reason=REFUTED)
def test_ii_c_opened_chooser_after_stateless_publisher():
    """(ii-c) publisher(G) then open_chooser settles at the same q, rel. error < 1e-12."""
    arr = compose_seq(ai.publisher(G_II), ai.open_chooser(3, 0.5, dt=0.3))
    s = ai.run(arr, 3000)
    p = s[1]
    print("q", p)
    assert rel(p, Q_FLAT) < 1e-12


# ---------------------------------------------------------------------------
# Addendum A (notes/ACTIVE-INFERENCE-TESTS.md §5), written after (ii)'s refutation.
# ---------------------------------------------------------------------------


def test_A1_simplex_repels_at_rate_and_gibbs_weights_attract():
    """(A1) |1 - sum p| grows by 1.2428178386074 per step (to 1%); final state exp(v/T - 1)."""
    v, T = ai.policy_values(G_II, 2.0)
    s, sums = ai.run(choice.logit_arrangement(v, T, dt=0.3), 3000,
                     record=lambda s: float(jnp.sum(s)))
    eps = np.abs(1 - np.array(sums))
    window = [(a, b) for a, b in zip(eps[:-1], eps[1:]) if 1e-12 < a < 1e-3 and b > 0]
    factors = np.array([b / a for a, b in window])
    print("steps in window", len(window), "median factor", np.median(factors))
    assert len(window) > 10
    assert np.all(np.abs(factors / 1.2428178386074 - 1) < 0.01)
    assert rel(s, jnp.exp(v / T - 1)) < 1e-10


@pytest.mark.parametrize("E,target", [(E_II, Q_HABIT), (None, Q_FLAT)])
def test_A2_shifted_chooser_keeps_its_normalization(E, target):
    """(A2) values v + 3: the same logit, and the chooser settles there, below 1e-12."""
    v, T = ai.policy_values(G_II, 2.0, E)
    s = ai.run(choice.logit_arrangement(v + 3.0, T, dt=0.3), 3000)
    assert rel(s, target) < 1e-12


def test_A2_shifted_open_chooser_after_stateless_publisher():
    """(A2) opened, the shift a constant c0 <1, p> in the potential: settles at q, below 1e-12."""
    arr = compose_seq(ai.publisher(G_II), ai.open_chooser(3, 0.5, dt=0.3, shift=3.0))
    assert rel(ai.run(arr, 3000)[1], Q_FLAT) < 1e-12


# ---------------------------------------------------------------------------
# Test (iii): expected free energy by a part that runs futures.
# ---------------------------------------------------------------------------

A_W, B_W, Q_W, OSTAR, P_PREF = 0.8, 0.5, 0.005, 1.0, 0.5
PUSH, LOOK, RETREAT = (1.0, 0.3), (0.0, 0.02), (-1.0, 0.3)
POLICIES = [[PUSH], [LOOK], [RETREAT]]
SIG_FLIP = 0.050968866365422215
STEP = ai.phi_step(ai.world())


def closed_G(m, Sig, seq):
    """The closed form, by hand, with no arrangement: x' = 0.8 x + 0.5 u."""
    G = 0.0
    for u, R in seq:
        mu = A_W * m + B_W * u
        V = A_W * A_W * Sig + Q_W
        so = V + R
        G += 0.5 * (so / P_PREF + (mu - OSTAR) ** 2 / P_PREF - 1 - np.log(so / P_PREF))
        G += 0.5 * np.log(2 * np.pi * np.e * R)
        m, Sig = mu, 1 / (1 / V + 1 / R)
    return G


def test_figurines_are_the_one_box_planner():
    """The planner by parts equals the one-box planner: G from the figurines equals the closed form
    at horizon 1 and 2 (1e-12), and the agent with the shifted chooser, exact and unilateral, settles
    where the one-box agent does (1e-12) over 6000 steps. The prediction, not a premise."""
    for Sig in [2.0, 0.05]:
        for a in [PUSH, LOOK, RETREAT]:
            for b in [None, PUSH, LOOK, RETREAT]:
                pol = [a] if b is None else [a, b]
                A = ai.figurine_planner([pol], Sig, 0.0, 0.0, 0.0, Q_W, OSTAR, P_PREF)
                g = float(A.out_f(A.P.position(A.P.init), jnp.zeros(0))[0])
                assert abs(g - closed_G(0.0, Sig, pol)) < 1e-12
    for uni in (False, True):
        m, p = settle(agent(uni, 2.0), 6000)
        m0, p0 = ai.run(one_box_agent(uni, 2.0), 6000)
        assert abs(float(m[0]) - float(m0[0])) < 1e-12
        assert float(jnp.max(jnp.abs(p - p0))) < 1e-12


def test_iii_a_rollout_by_phi_equals_closed_form():
    """(iii-a) horizon 1 at three Sigma, two m; horizon 2, nine sequences, two Sigma; to 1e-12."""
    for Sig in [2.0, 0.5, 0.05]:
        for m in [0.0, -0.5]:
            for pol in POLICIES:
                g = float(ai.expected_free_energy(STEP, m, Sig, pol, Q_W, OSTAR, P_PREF))
                assert abs(g - closed_G(m, Sig, pol)) < 1e-12
    for Sig in [2.0, 0.05]:
        for a in [PUSH, LOOK, RETREAT]:
            for b in [PUSH, LOOK, RETREAT]:
                g = float(ai.expected_free_energy(STEP, 0.0, Sig, [a, b], Q_W, OSTAR, P_PREF))
                assert abs(g - closed_G(0.0, Sig, [a, b])) < 1e-12
    G = ai.G_vector(STEP, 0.0, 0.2, POLICIES, Q_W, OSTAR, P_PREF)
    assert rel(G, [1.0718873162515554, 0.7080121190054778, 3.0718873162515554]) < 1e-12


def test_iii_b_risk_plus_ambiguity_is_cost_less_information_gain():
    """(iii-b) G = -E log p~(o) - I, I = 1/2 log(1 + w_obs/w_pred), to 1e-12."""
    for Sig in [2.0, 0.5, 0.05]:
        for m in [0.0, -0.5]:
            for u, R in [PUSH, LOOK, RETREAT]:
                mu, V = ai.predict(STEP, jnp.full(1, m), jnp.full((1, 1), Sig), jnp.full(1, u), Q_W)
                mu, V = float(mu[0]), float(V[0, 0])
                G = float(ai.risk(mu, V + R, OSTAR, P_PREF) + ai.ambiguity(R))
                cost = 0.5 * np.log(2 * np.pi * P_PREF) + 0.5 * (V + R + (mu - OSTAR) ** 2) / P_PREF
                I = float(ai.information_gain(V, R))
                assert abs(G - (cost - I)) < 1e-12
                assert abs(I - 0.5 * np.log(1 + V / R)) < 1e-12


def figurine_agent(chooser, Sig=0.5, x_obs=-0.5, w_ev=4.0, eta=0.1):
    """The planner by parts (belief, figurines, score boxes) wired to ``chooser``."""
    return compose_seq(ai.figurine_planner(POLICIES, Sig, x_obs=x_obs, w_ev=w_ev, eta=eta, Q=Q_W,
                                           ostar=OSTAR, P=P_PREF), chooser)


def settle(arr, steps, k=3):
    """Run to ``steps``; the belief's mean and the chooser's state, read off the composite's position."""
    pos = arr.P.position(ai.run(arr, steps))
    return pos[:1], pos[-k:]


def chooser_after(Sig, T, shift=0.0, steps=3000):
    """The chooser after the figurines, the belief held at 0 with spread ``Sig``."""
    return settle(figurine_agent(ai.open_chooser(3, T, dt=0.3, shift=shift), Sig=Sig, x_obs=0.0,
                                 w_ev=0.0, eta=0.0), steps)[1]


@pytest.mark.parametrize("shift", [pytest.param(0.0, marks=pytest.mark.xfail(strict=True, reason=REFUTED)), 2.0])
def test_iii_c_look_while_unsure_push_once_sure(shift):
    """(iii-c) shares at Sigma = 0.2 and 0.02 (1e-10); ratio push/look = 1 at the switch (1e-9)."""
    got = {}
    for Sig in [0.2, 0.02, SIG_FLIP]:
        got[Sig] = chooser_after(Sig, 0.2, shift)
        print("shift", shift, "Sigma", Sig, "p", got[Sig], "sum", float(jnp.sum(got[Sig])))
    ratio = float(got[SIG_FLIP][0] / got[SIG_FLIP][1])
    print("ratio at switch", ratio)
    assert abs(ratio - 1) < 1e-9
    assert abs(float(got[0.2][1]) - 0.86048578343851367) < 1e-10
    assert abs(float(got[0.2][0]) - 0.13950788291340072) < 1e-10
    assert abs(float(got[0.02][0]) - 0.71147230606178846) < 1e-10
    assert abs(float(got[0.02][1]) - 0.28849539314548839) < 1e-10


def agent(unilateral, shift):
    return figurine_agent(ai.open_chooser(3, 0.5, dt=0.3, unilateral=unilateral, shift=shift))


def one_box_agent(unilateral, shift):
    return compose_seq(
        ai.planner(POLICIES, 0.5, x_obs=-0.5, w_ev=4.0, eta=0.1, Q=Q_W, ostar=OSTAR, P=P_PREF),
        ai.open_chooser(3, 0.5, dt=0.3, unilateral=unilateral, shift=shift))


@pytest.mark.parametrize("shift", [pytest.param(0.0, marks=pytest.mark.xfail(strict=True, reason=REFUTED)), 2.0])
def test_iii_d_exact_wiring_bends_belief_toward_preference(shift):
    """(iii-d) exact: m* = -0.11620980279789729, p* as written; to 1e-9."""
    m, p = settle(agent(False, shift), 6000)
    print("exact shift", shift, "m", float(m[0]), "p", p, "sum", float(jnp.sum(p)))
    assert abs(float(m[0]) - (-0.11620980279789729)) < 1e-9
    assert rel(p, [0.2703991878894675, 0.7261863226871871, 0.00341448942334541]) < 1e-9


@pytest.mark.parametrize("shift", [pytest.param(0.0, marks=pytest.mark.xfail(strict=True, reason=REFUTED)), 2.0])
def test_iii_d_unilateral_wiring_leaves_belief_on_evidence(shift):
    """(iii-d) unilateral: m* = -0.5, p* as written; to 1e-9."""
    m, p = settle(agent(True, shift), 6000)
    print("unilateral shift", shift, "m", float(m[0]), "p", p, "sum", float(jnp.sum(p)))
    assert abs(float(m[0]) - (-0.5)) < 1e-9
    assert rel(p, [0.4069992837643438, 0.5914956883515895, 0.00150502788406671]) < 1e-9


def test_A3_unshifted_switch_ratio_survives_the_drift():
    """(A3) unshifted, the shares drift off the simplex but stay proportional: ratio 1 at the switch."""
    p = chooser_after(SIG_FLIP, 0.2)
    assert float(jnp.sum(p)) < 0.01
    assert abs(float(p[0] / p[1]) - 1) < 1e-9


def test_A3_unshifted_unilateral_belief_still_on_evidence():
    """Not predicted (A3 said all of (iii-d)'s numbers would fail unshifted): the unilateral
    chooser pulls nothing back, so the belief settles on the evidence whatever the chooser does."""
    m, p = settle(agent(True, 0.0), 6000)
    assert abs(float(m[0]) - (-0.5)) < 1e-9
    assert float(jnp.sum(p)) < 0.05


# ---------------------------------------------------------------------------
# Addendum B (notes §9): a chooser whose every state is a distribution.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("E,target", [(E_II, Q_HABIT), (None, Q_FLAT)])
def test_B1_natural_chooser_settles_on_policy_posterior(E, target):
    """(B1) settles at q (1e-12), one step at dt = 1/T, sum p = 1 throughout, shift-blind."""
    T = 0.5
    arr = compose_seq(ai.publisher(G_II), ai.natural_chooser(3, T, dt=0.3, E=E))
    s, sums = ai.run(arr, 3000, record=lambda s: float(jnp.sum(ai.shares(s[1]))))
    assert rel(ai.shares(s[1]), target) < 1e-12
    assert max(abs(x - 1) for x in sums) < 1e-15
    one = ai.run(compose_seq(ai.publisher(G_II), ai.natural_chooser(3, T, dt=1 / T, E=E)), 1)
    assert rel(ai.shares(one[1]), target) < 1e-12
    _, a = ai.run(arr, 50, record=lambda s: s[1])
    _, b = ai.run(compose_seq(ai.publisher(G_II + 7.0), ai.natural_chooser(3, T, dt=0.3, E=E)),
                  50, record=lambda s: s[1])
    assert max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(a, b)) < 1e-12


def natural_agent(unilateral):
    return figurine_agent(ai.natural_chooser(3, 0.5, dt=0.3, unilateral=unilateral))


def test_B2_natural_chooser_in_the_planner_composite_unshifted():
    """(B2) exact: m* = -0.11620980279789729; unilateral: m* = -0.5; p* as in (iii-d); 1e-9."""
    m, lam = settle(natural_agent(False), 6000, k=2)
    print("natural exact m", float(m[0]), "p", ai.shares(lam))
    assert abs(float(m[0]) - (-0.11620980279789729)) < 1e-9
    assert rel(ai.shares(lam), [0.2703991878894675, 0.7261863226871871, 0.00341448942334541]) < 1e-9
    m, lam = settle(natural_agent(True), 6000, k=2)
    assert abs(float(m[0]) - (-0.5)) < 1e-9
    assert rel(ai.shares(lam), [0.4069992837643438, 0.5914956883515895, 0.00150502788406671]) < 1e-9


# ---------------------------------------------------------------------------
# Addendum C (notes §9): David's picture, nested.
# ---------------------------------------------------------------------------


def settle_hierarchy(kappa, P=4.0, push1=True, push2=True, steps=80000):
    eta = 1.0 / (kappa + 3 * P)
    (hm, mu2) = ai.run(ai.hierarchy(kappa, P, P, P, 1.0, eta, push1, push2), steps)
    h, mu1 = hm
    return np.array([float(h[0]), float(mu1[0]), float(mu2[0])])


@pytest.mark.parametrize("kappa,target", [
    (1.0, [0.5714285714285715, 0.7142857142857144, 0.8571428571428572]),
    (100.0, [0.01315789473684211, 0.3421052631578948, 0.6710526315789473]),
    (0.01, [0.9925558312655088, 0.9950372208436725, 0.9975186104218362]),
])
def test_C1_C2_goal_reaches_the_hand_by_pullback_alone(kappa, target):
    """(C1, C2) the spring-chain rest, ordered 0 < h < mu_1 < mu_2 < g; 1e-10."""
    x = settle_hierarchy(kappa)
    print("kappa", kappa, "rest", x)
    assert np.max(np.abs(x - np.array(target))) < 1e-10
    assert 0 < x[0] < x[1] < x[2] < 1


def test_C3_each_level_must_push_the_one_below():
    """(C3) level 1 unilateral: (0, 1/3, 2/3); level 2 unilateral: (0, 0, 1/2); 1e-10."""
    assert np.max(np.abs(settle_hierarchy(1.0, push1=False) - [0, 1 / 3, 2 / 3])) < 1e-10
    assert np.max(np.abs(settle_hierarchy(1.0, push2=False) - [0, 0, 0.5])) < 1e-10


def test_C4_offset_and_only_the_ratio_counts():
    """(C4) h* < g, mu_1* > h*; precisions x100 at kappa = 1 equal kappa = 0.01's rest; 1e-10."""
    x = settle_hierarchy(1.0, P=400.0, steps=20000)
    assert np.max(np.abs(x - [0.9925558312655088, 0.9950372208436725, 0.9975186104218362])) < 1e-10
    assert x[0] < 1 and x[1] > x[0]
