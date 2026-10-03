"""Science at two scales, tests (a)-(f).

Test (a): one mind and a population are one arrangement drawn two ways.

Predictions fixed before code, in ``notes/SCIENCE-SCALES-PREDICTIONS-2026-10-01.md`` §1, quoted:

  (a1) At every step, ``softmax(lam, 0) = softmax(y)`` to round-off (< 1e-12), with
  ``lam_i = y_i - y_n``, for a data stream of at least 200 draws and ``n >= 3``.
  (a2) The composite of ``n`` one-dimensional holder boxes and the single ``n``-dimensional box
  give the same trajectory to round-off (clustering is free).
  (a3) Misspecified world (no ``h_i`` true; e.g. Gaussian hypotheses with means on a grid, truth
  off the grid): both readings concentrate on the hypothesis minimizing ``KL(truth || h_i)``, the
  same one, with weight > 0.99 by step 200.

Settings, fixed in that note's "Numbers fixed before code" paragraph: ``h_i = N(mu_i, 1)``,
``mu in {0,1,2,3,4}``, 200 draws from ``default_rng(0)``; misspecified truth ``N(2.3, 1)``,
well-specified truth ``N(2, 1)``. Verdicts in the note's §5.1.

Test (b), keeping an idea costs; confirmation pays for it, or explaining away does. Quoted from §2:

  (b1) *Sanity, built in, not a finding:* a hypothesis that changes no prediction (``K_hold =
  K_drop`` before upkeep) has hold share ``1/(1+e^{c/T})`` at rest, which falls to 0 as ``c/T``
  grows, and stays at 1/2 when ``c = 0``. (b2) Hold share at rest equals
  ``1/(1 + exp(-(K_drop - K_hold* - c)/T))`` (with ``ell`` at its optimum in the learned arm), to
  1e-8. (b3) The disconfirmation the holder survives: hold beats drop iff ``S_B < S*`` with
  ``S*_fixed = 2(K_drop - log 2pi - c) - S_A`` and ``S*_learned = exp(S*_fixed - 1)``: exponentially
  more disconfirmation survives when trust in the critic can be learned. (b4) With learned trust,
  what drops ``h`` is a lack of confirmation, not disconfirmation: lowering A's confirmation (raising
  ``S_A`` by moving A's mean off 0) drops ``h`` at a rate set by ``S_A`` one-for-one in both arms,
  while ``d`` moves the decision only through ``1/2 log S_B``.

(b4) is too vague to test as worded; the run's sharper (b4'), stated in §2's "Numbers fixed before
code" before running: in both arms the rested hold logit moves by exactly ``-Delta S_A / 2T`` when A's
mean moves (same in both arms, to 1e-8), and by ``-Delta S_B / 2T`` (fixed) or ``-Delta log S_B / 2T``
(learned) when ``d`` moves. Settings for (b): ``L = 10``, ``c = 1.6``, ``T = 1``, ``n = 200`` per
source, ``default_rng(0)``, grid ``d = 0, 0.125, ..., 5``.

Test (c), holders who read each other and not the world. Quoted from §3:

  (c1) ``beta = 0``, agree: consensus at the ``W``-weighted average of the starting ``lam`` (plain
  average for doubly stochastic ``W``), at no more than the starting confidence: a community fed only
  by itself agrees and learns nothing. (c2) ``beta = 0``, evidence: consensus in the direction of
  ``W``'s leading eigenvector applied to the start, with ``|lam|`` growing like ``(1+alpha)^t``: sure
  of something no one observed. (c3) ``beta > 0``, agree: every holder concentrates on the true
  hypothesis (weight > 0.99) in 20 of 20 seeds. (c4) ``beta > 0``, evidence: over 20 seeds at
  ``alpha/beta in {0, 0.1, 1, 10}``, the count locking onto a wrong hypothesis is 0 at 0 and rises with
  ``alpha/beta``; at 10, more than half of the seeds whose starting average favoured a wrong
  hypothesis lock onto it.

Settings, definitions and the derivation of both steps under the inverse-Fisher response are in §3's
"Derivation" and "Numbers fixed before code" paragraphs; realized in the gauge-free log-weight chart.

Test (d), nothing new without variation, at both scales. Quoted from §4:

  (d1) No variation: both readings settle on ``mu = 1`` forever; ``3.5`` never gains weight.
  (d2) Mutation at rate ``eps`` (each step a fraction ``eps`` of each type's abundance moves to its grid
  neighbours, half each side, then selection): the population reaches a peak at 3.5; stationary spread
  increases with ``eps``. (d3) The same update on the mind is Bayes' filter with the assumption that
  the truth may drift by one grid step (transition matrix ``M``): identical trajectories to round-off
  when mutation and selection are applied in the same order. (d4) **No prediction** on whether the
  mutation step is an exact forcing of the book's kind in some chart; the run reports what it finds.

Charts, ``eps``, steps, seeds and the definitions of "settle", "reaches a peak" and "spread" are in §4's
"Numbers fixed before code".

Test (e), an idea against a rival (second round, §6). Quoted:

  (e1) *fixed:* the rival overtakes (share of ``h1`` passes 1/2) at the B count ``n_B*`` where the
  sums of squares equalize; with population values, ``n_B* = n_A``. (e2) *both learn:* the same
  crossing, ``n_B* ~ n_A`` (each hypothesis discounts the other's confirming source by the same
  ``1/2 log(1+d^2)`` per draw, so learning trust protects neither); the switch is softer, its logit
  growing per B draw by ``1/2 log(1+d^2)`` instead of ``d^2/2``. (e3) *only the incumbent learns:* the
  rival needs ``n_B* ~ n_A d^2 / log(1+d^2)`` draws (at ``d = 3``, about 3.9 times as many as the
  incumbent's confirmations). Explaining away protects an idea against a rival only when the rival is
  denied the same move. (e4) Contrast with (b): at the same ``d`` where (b)'s learned-trust holder kept
  its idea against a know-nothing default, the rival here overtakes it after a finite B count in every
  arm.

Settings and the meaning of "held" are in §6's "Numbers fixed before code".

Test (f), a community against one mind holding every datum (second round, §7). Quoted:

  (f1) Summaries with provenance: every holder's belief equals the one mind's on the data that has
  reached it; once every paper has reached everyone, equal to the pooled mind to round-off. Sufficient
  statistics add. (f2) Summaries without provenance: each holder is more confident than the pooled mind
  at every step (data counted many times); over 20 seeds, some lock onto a wrong hypothesis when early
  draws mislead; the count is above the with-provenance arm's (which is 0 when the truth is on the
  grid). (f3) Conclusions with replication: when the truth lies in some holder's starting slice, every
  holder ends holding it (weight > 0.99) in 20 of 20 seeds, reaching each holder after roughly its ring
  distance from the first finder times ``r``; when it lies in no slice, never (test (d)'s gap at the
  community's scale); the one bounded mind finds it only if its own ``K`` contains it. (f4) Conclusions
  without replication: wrong conclusions spread; over 20 seeds the count ending on a wrong hypothesis
  is above zero and above (f3)'s, at the same settings.

Settings and definitions: §7, "Numbers fixed before code".

Each test asserts a prediction at the tolerance it was stated with; a refuted one is a strict
xfail, not repaired. (f2) is refuted on the ring and (f3) as written (18 of 20); see §5.6.
(d3) is refuted at the tolerance fixed before code (1.1e-11 against 1e-12).
(c3) is refuted at ``alpha/beta = 10`` (both wirings); the rest of (c) held.
(a1) is REFUTED on both streams and (a3) is refuted for the mind: the
library's mind step is gauge-fixed with its reference value at ``mu = 4``, and once that
reference's weight is thin the step loses the exact translation (from draw 9 and 6; about
``1e-16`` over the reference's height), the error compounds, and the misspecified run is NaN
from step 110. (Cause corrected 2026-10-03, ``notes/BAYES-EVERY-BELIEF.md`` §3: not the dense
inverse and not ``cond(G)``.) (a2) and the population's half of (a3) hold.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arrangements import science as sc
from arrangements.choice import belief_to_distribution

MUS = jnp.arange(5.0)
N_DRAWS = 200
STREAMS = {"misspecified": 2.3, "well-specified": 2.0}


def _stream(true_mu):
    return np.random.default_rng(0).normal(true_mu, 1.0, N_DRAWS)


@pytest.fixture(scope="module")
def runs():
    return {name: sc.run_two_scales(_stream(m), MUS) for name, m in STREAMS.items()}


REFUTED_A1 = ("refuted (notes/SCIENCE-SCALES-PREDICTIONS-2026-10-01.md §5.1): the gauge's thin "
              "reference value breaks the mind's exact translation; max |dq| 2.2e-4 "
              "(well-specified), NaN from step 110 (misspecified)")


@pytest.mark.xfail(strict=True, reason=REFUTED_A1)
@pytest.mark.parametrize("name", list(STREAMS))
def test_a1_mind_equals_population(runs, name):
    lams, ys, _ = runs[name]
    q_mind = jax.vmap(belief_to_distribution)(lams)
    q_pop = jax.vmap(sc.abundance_share)(ys)
    dq = float(jnp.max(jnp.abs(q_mind - q_pop)))
    dlam = jnp.abs(lams - (ys[:, :-1] - ys[:, -1:]))
    dlam_rel = float(jnp.max(dlam / jnp.maximum(1.0, jnp.abs(lams))))
    print(f"\n(a1) {name}: max_t |q_mind - q_pop| = {dq:.3e}; "
          f"max_t |lam - (y_i - y_n)| = {float(jnp.max(dlam)):.3e} (relative {dlam_rel:.3e})")
    assert lams.shape[0] == N_DRAWS + 1 and len(MUS) >= 3
    assert dq < 1e-12
    assert dlam_rel < 1e-12


@pytest.mark.parametrize("name", list(STREAMS))
def test_a2_clustering_is_free(runs, name):
    _, yt, ys = runs[name]
    d = float(jnp.max(jnp.abs(yt - ys)))
    print(f"\n(a2) {name}: max_t |y_tensor - y_single| = {d:.3e}")
    assert d < 1e-12


def _a3(runs):
    lams, ys, _ = runs["misspecified"]
    kl = 0.5 * (STREAMS["misspecified"] - MUS) ** 2
    target = int(jnp.argmin(kl))
    qs = {"mind": jax.vmap(belief_to_distribution)(lams), "population": jax.vmap(sc.abundance_share)(ys)}
    for who, q in qs.items():
        print(f"\n(a3) {who}: KL-closest mu = {float(MUS[target])}; argmax at 200 mu = "
              f"{float(MUS[int(jnp.argmax(q[-1]))])}; weight {float(q[-1, target]):.17f}; "
              f"first step > 0.99 = {int(np.argmax(np.asarray(q[:, target]) > 0.99))}")
    return target, qs


def test_a3_population_concentrates_on_kl_closest(runs):
    target, qs = _a3(runs)
    q = qs["population"][-1]
    assert int(jnp.argmax(q)) == target and float(q[target]) > 0.99


@pytest.mark.xfail(strict=True, reason="refuted for the mind (§5.1): NaN from step 110, same cause as (a1)")
def test_a3_mind_concentrates_on_the_same_hypothesis(runs):
    target, qs = _a3(runs)
    q = qs["mind"][-1]
    assert int(jnp.argmax(q)) == target == int(jnp.argmax(qs["population"][-1]))
    assert float(q[target]) > 0.99


# ---------------------------------------------------------------------------
# Test (b)
# ---------------------------------------------------------------------------

L_, C_, T_, N_ = 10.0, 1.6, 1.0, 200
_z = np.random.default_rng(0).normal(0.0, 1.0, 2 * N_)
Z_A, Z_B = _z[:N_], _z[N_:]
K_DROP = 2.0 * float(np.log(2.0 * L_))
D_GRID = np.arange(0.0, 5.0 + 1e-9, 0.125)


def _S(draws):
    return float(np.mean(np.asarray(draws) ** 2))


def _rest(S_A, S_B, learn):
    arr = sc.holder_chooser(lambda ell: sc.k_hold(S_A, S_B, ell), K_DROP, C_, T_, learn)
    return np.asarray(sc.rest(arr))


def _k_star(S_A, S_B, learn):
    return float(sc.k_hold(S_A, S_B, -np.log(S_B) if learn else 0.0))


@pytest.fixture(scope="module")
def grid():
    S_A = _S(Z_A)
    out = {}
    for learn in (False, True):
        rows = []
        for d in D_GRID:
            S_B = _S(d + Z_B)
            r = _rest(S_A, S_B, learn)
            rows.append((d, S_B, r))
        out[learn] = rows
    return S_A, out


def test_b1_idle_hypothesis_costs_its_upkeep():
    errs = []
    for c in (0.0, 1.0, 3.0, 10.0):
        arr = sc.holder_chooser(lambda ell: K_DROP, K_DROP, c, T_, False)
        a = float(sc.rest(arr)[0])
        errs.append(abs(a - 1.0 / (1.0 + np.exp(c / T_))))
        print(f"\n(b1) c/T = {c:g}: hold share {a:.10g}, closed form {1/(1+np.exp(c/T_)):.10g}")
    assert max(errs) < 1e-8


@pytest.mark.parametrize("learn", [False, True], ids=["fixed", "learned"])
def test_b2_rest_share_is_the_logit(grid, learn):
    S_A, out = grid
    errs, ell_errs = [], []
    for d, S_B, r in out[learn]:
        errs.append(abs(r[0] - sc.hold_share_closed(K_DROP, _k_star(S_A, S_B, learn), C_, T_)))
        if learn:
            ell_errs.append(abs(r[2] + np.log(S_B)))
    print(f"\n(b2) {'learned' if learn else 'fixed'}: max |share - closed form| = {max(errs):.3e}"
          + (f"; max |ell - log(1/S_B)| = {max(ell_errs):.3e}" if learn else ""))
    assert max(errs) < 1e-8


def test_b3_learned_trust_survives_exponentially_more(grid):
    S_A, out = grid
    S_f = 2.0 * (K_DROP - sc.LOG2PI - C_) - S_A
    S_l = float(np.exp(S_f - 1.0))
    tallies, mismatches = {}, 0
    for learn, S_star in ((False, S_f), (True, S_l)):
        hold = [r[0] > 0.5 for _, _, r in out[learn]]
        rule = [S_B < S_star for _, S_B, _ in out[learn]]
        mismatches += sum(h != q for h, q in zip(hold, rule))
        last_hold = max((d for (d, _, _), h in zip(out[learn], hold) if h), default=None)
        tallies["learned" if learn else "fixed"] = (sum(hold), len(hold) - sum(hold), last_hold)
    print(f"\n(b3) S*_fixed = {S_f:.6f}, S*_learned = {S_l:.6f}; over {len(D_GRID)} values of d "
          f"(hold, drop, largest d held): fixed {tallies['fixed']}, learned {tallies['learned']}; "
          f"mismatches with the S_B < S* rule: {mismatches}")
    assert mismatches == 0
    assert 0.0 < S_f < S_l


@pytest.mark.parametrize("learn", [False, True], ids=["fixed", "learned"])
def test_b4_prime_confirmation_moves_both_arms_alike(learn):
    T = T_
    logit = lambda a: float(np.log(a / (1.0 - a)))  # noqa: E731
    S_B0 = _S(2.0 + Z_B)  # d = 2, inside both arms' interesting range
    base_SA = _S(Z_A)
    base = logit(_rest(base_SA, S_B0, learn)[0])
    errs_A = []
    for m in (0.5, 1.0, 1.5):
        S_A = _S(m + Z_A)
        got = logit(_rest(S_A, S_B0, learn)[0]) - base
        errs_A.append(abs(got - (-(S_A - base_SA) / (2 * T))))
    errs_d = []
    for d in (0.0, 1.0, 3.0):
        S_B = _S(d + Z_B)
        got = logit(_rest(base_SA, S_B, learn)[0]) - base
        want = -(np.log(S_B) - np.log(S_B0)) / (2 * T) if learn else -(S_B - S_B0) / (2 * T)
        errs_d.append(abs(got - want))
    print(f"\n(b4') {'learned' if learn else 'fixed'}: max err, A's mean moved = {max(errs_A):.3e}; "
          f"d moved = {max(errs_d):.3e}")
    assert max(errs_A) < 1e-8 and max(errs_d) < 1e-8


# ---------------------------------------------------------------------------
# Test (c)
# ---------------------------------------------------------------------------

M_, NH_, TRUE_ = 10, 5, 2
SEEDS = range(20)
ALPHAS = (0.0, 0.1, 1.0, 10.0)
WIRINGS = {"ring": sc.ring_W(M_), "all-to-all": sc.all_to_all_W(M_)}


def _seed(s, T):
    rng = np.random.default_rng(s)
    Y0 = rng.normal(size=(M_, NH_))
    X = rng.normal(2.0, 1.0, size=(T, M_))
    LL = -0.5 * (X[:, :, None] - np.arange(NH_)[None, None, :]) ** 2 - 0.5 * np.log(2 * np.pi)
    return Y0, LL


def _lam(Y):
    Y = np.asarray(Y)
    return Y[..., :-1] - Y[..., -1:]


def _q(Y):
    Y = np.asarray(Y)
    e = np.exp(Y - Y.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _favourite_wrong(s):
    Y0, _ = _seed(s, 1)
    return int(np.argmax(np.append(_lam(Y0).mean(0), 0.0))) != TRUE_


def _classify(Y):
    q = _q(Y)
    for i in range(NH_):
        if np.all(q[:, i] > 0.99):
            return "true" if i == TRUE_ else "wrong"
    return "neither"


@pytest.mark.parametrize("wiring", list(WIRINGS))
def test_c1_agreeing_alone_learns_nothing(wiring):
    W, errs, conf_ok = WIRINGS[wiring], [], 0
    for s in SEEDS:
        Y0, LL = _seed(s, 400)
        Ys = sc.community_run(sc.account_community(Y0, W, 0.5, 0.0, LL, "agree"), 400, M_, NH_)
        target = _lam(Y0).mean(0)
        errs.append(float(np.max(np.abs(_lam(Ys[-1]) - target))))
        cons = _q(np.append(target, 0.0)).max()
        conf_ok += int(cons <= _q(Y0).max())
    print(f"\n(c1) {wiring}: max over seeds |lam_j(400) - mean lam_0| = {max(errs):.3e}; "
          f"consensus no more confident than the most confident start in {conf_ok} of 20")
    assert max(errs) < 1e-8 and conf_ok == 20


@pytest.mark.parametrize("wiring", list(WIRINGS))
def test_c2_reading_accounts_as_evidence_is_sure_of_nothing_observed(wiring):
    W, rel, growth, sure = WIRINGS[wiring], [], [], 0
    for s in SEEDS:
        Y0, LL = _seed(s, 100)
        Ys = sc.community_run(sc.account_community(Y0, W, 1.0, 0.0, LL, "evidence"), 100, M_, NH_)
        target = _lam(Y0).mean(0)
        lam_T, lam_T1 = _lam(Ys[-1]), _lam(Ys[-2])
        rel.append(float(np.max(np.abs(lam_T / 2.0 ** 100 - target)) / np.max(np.abs(target))))
        growth.append(abs(np.linalg.norm(lam_T) / np.linalg.norm(lam_T1) - 2.0))
        sure += int(np.all(_q(Ys[-1]).max(-1) > 0.99))
    print(f"\n(c2) {wiring}: max relative |lam(100)/2^100 - mean lam_0| = {max(rel):.3e}; "
          f"max |growth - 2| = {max(growth):.3e}; every holder > 0.99 sure in {sure} of 20")
    assert max(rel) < 1e-3 and max(growth) < 1e-6 and sure == 20


@pytest.fixture(scope="module")
def community_grid():
    out = {}
    for arm in ("agree", "evidence"):
        for wname, W in WIRINGS.items():
            for alpha in ALPHAS:
                res = []
                for s in SEEDS:
                    Y0, LL = _seed(s, 100)
                    arr = sc.account_community(Y0, W, alpha, 1.0, LL, arm)
                    Y = np.asarray(sc.run(arr, 100))[:-1].reshape(M_, NH_)
                    res.append(_classify(Y) if np.all(np.isfinite(Y)) else "nonfinite")
                out[(arm, wname, alpha)] = res
                print(f"\n(c grid) {arm:8s} {wname:10s} alpha/beta={alpha:<4g} true {res.count('true'):2d}, "
                      f"wrong {res.count('wrong'):2d}, neither {res.count('neither'):2d}, "
                      f"nonfinite {res.count('nonfinite'):2d}")
    return out


REFUTED_C3 = ("refuted at alpha/beta = 10 (§5.3): the synchronous agree step's disagreement modes have "
              "eigenvalues -19 (ring) and -10.1 (all-to-all), so holders diverge; 0 of 20 find the truth")


@pytest.mark.parametrize("alpha", [0.0, 0.1, 1.0, pytest.param(
    10.0, marks=pytest.mark.xfail(strict=True, reason=REFUTED_C3))])
@pytest.mark.parametrize("wiring", list(WIRINGS))
def test_c3_agreeing_with_data_finds_the_truth(community_grid, wiring, alpha):
    res = community_grid[("agree", wiring, alpha)]
    assert res.count("true") == 20


@pytest.mark.parametrize("wiring", list(WIRINGS))
def test_c4_double_counting_locks_onto_the_start(community_grid, wiring):
    counts = [community_grid[("evidence", wiring, a)].count("wrong") for a in ALPHAS]
    fav_wrong = [s for s in SEEDS if _favourite_wrong(s)]
    at10 = sum(community_grid[("evidence", wiring, 10.0)][s] == "wrong" for s in fav_wrong)
    agree = [community_grid[("agree", wiring, a)].count("wrong") for a in ALPHAS]
    print(f"\n(c4) {wiring}: locked wrong over alpha/beta {ALPHAS}: evidence {counts}, agree {agree}; "
          f"at 10, {at10} of the {len(fav_wrong)} seeds whose start favoured a wrong hypothesis")
    assert counts[0] == 0
    assert all(a <= b for a, b in zip(counts, counts[1:])) and counts[-1] > counts[0]
    assert at10 > len(fav_wrong) / 2


# ---------------------------------------------------------------------------
# Test (d)
# ---------------------------------------------------------------------------

MU_D = np.arange(0.0, 5.0 + 1e-9, 0.5)
I35, I1 = 7, 2
EPS = (0.0, 1e-3, 1e-2, 1e-1)
D_SEEDS = range(5)
T_D = 120
START = np.zeros(len(MU_D)); START[:3] = 1.0 / 3.0


@pytest.fixture(scope="module")
def variation():
    out = {}
    for seed in D_SEEDS:
        xs = np.random.default_rng(seed).normal(3.7, 1.0, T_D)
        for eps in EPS:
            Ns, ps = sc.run_variation(xs, MU_D, eps, START)
            out[(seed, eps)] = (np.asarray(Ns), np.asarray(ps))
    return out


def _norm(Ns):
    return Ns / Ns.sum(1, keepdims=True)


def test_d1_no_variation_nothing_new(variation):
    for seed in D_SEEDS:
        Ns, ps = variation[(seed, 0.0)]
        for name, w in (("population", _norm(Ns)), ("mind", ps)):
            modes = w[1:].argmax(1)
            assert np.all(w[:, I35] == 0.0) and np.all(w[:, 3:] == 0.0), (seed, name)
            assert modes[-1] == I1 and np.all(modes[T_D // 2:] == I1), (seed, name)
    print(f"\n(d1) eps = 0, seeds 0-4: largest weight on mu = 1 through the second half in both readings; "
          f"weight off {{0, 0.5, 1}} exactly 0 at every step")


def test_d2_variation_finds_the_kl_closest(variation):
    ok = True
    for seed in D_SEEDS:
        rows, spreads = [], []
        for eps in EPS[1:]:
            w = _norm(variation[(seed, eps)][0])[1:]
            modes = w.argmax(1)
            first = int(np.argmax(modes == I35)) + 1 if np.any(modes == I35) else None
            sd = np.sqrt(np.sum(w * (MU_D - (w @ MU_D)[:, None]) ** 2, axis=1))
            spreads.append(float(sd[-40:].mean()))
            rows.append(f"eps={eps:g}: first {first}, mode@120 {MU_D[modes[-1]]}, "
                        f"2nd-half at 3.5 {np.mean(modes[T_D // 2:] == I35):.2f}, spread {spreads[-1]:.4f}")
            ok &= first is not None
        ok &= all(a < b for a, b in zip(spreads, spreads[1:]))
        print(f"\n(d2) seed {seed}: " + "; ".join(rows))
    assert ok


REFUTED_D3 = ("refuted at the 1e-12 fixed before code (§5.4): 4e-12 to 1.1e-11; the error is the population "
              "realization's cancellation N + D_l (M - D_l^{-1}) N, not the filter (mind vs plain filter 5e-16)")


@pytest.mark.xfail(strict=True, reason=REFUTED_D3)
@pytest.mark.parametrize("eps", EPS)
def test_d3_population_and_mind_one_filter(variation, eps):
    worst = max(float(np.max(np.abs(_norm(variation[(s, eps)][0]) - variation[(s, eps)][1])))
                for s in D_SEEDS)
    print(f"\n(d3) eps={eps:g}: max over seeds, steps, types |N/sum N - p| = {worst:.3e}")
    assert worst < 1e-12


def test_d4_what_kind_of_forcing():
    M = sc.mutation_matrix(len(MU_D), 0.01)
    L = np.exp(np.asarray(sc.gaussian_loglik(3.7, MU_D)))
    ell = L / L.max()
    N = np.random.default_rng(7).random(len(MU_D))
    pop = sc.mutation_selection(M, ell, N)
    J_pop = np.asarray(jax.jacfwd(lambda q: pop.omega(q, jnp.zeros(0), jnp.zeros(0))[0])(jnp.asarray(N)))
    landed = np.asarray(sc._one_step(pop))
    mind = sc.drift_filter(M, L, N / N.sum())
    J_mind = np.asarray(jax.jacfwd(lambda q: mind.omega(q, jnp.zeros(0), jnp.zeros(0))[0])(jnp.asarray(N / N.sum())))
    asym_pop = float(np.max(np.abs(J_pop - J_pop.T)))
    asym_mind = float(np.max(np.abs(J_mind - J_mind.T)) / np.max(np.abs(J_mind)))
    land = float(np.max(np.abs(landed - ell * (np.asarray(M) @ N))) / np.max(np.abs(ell * (np.asarray(M) @ N))))
    print(f"\n(d4) population forcing Jacobian asymmetry {asym_pop:.3e} (exact); one step vs D_l M N "
          f"relative {land:.3e}; mind's direct forcing Jacobian relative asymmetry {asym_mind:.3e} (not exact)")
    assert asym_pop < 1e-12 and land < 1e-12 and asym_mind > 1e-3


# ---------------------------------------------------------------------------
# Test (e)
# ---------------------------------------------------------------------------

NA_E, NB_E, E_SEEDS, E_DS = 20, 150, range(10), (3.0, 2.0)
E_ARMS = {"fixed": (), "both learn": ((0, 0), (0, 1), (1, 0), (1, 1)),
          "only the incumbent learns": ((0, 0), (0, 1))}


def _e_draws(seed, d):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, NA_E), rng.normal(d, 1.0, NB_E)


def _e_stats(a, b, d, k):
    SB = (lambda v: float(np.mean((b[:k] - v) ** 2))) if k else (lambda v: 1.0)
    return np.array([NA_E, k], float), np.array([[np.mean(a ** 2), SB(0.0)], [np.mean((a - d) ** 2), SB(d)]])


def _e_closed_logit(n, S, learned):
    K = np.zeros(2)
    for i in (0, 1):
        for X in (0, 1):
            if n[X]:
                K[i] += 0.5 * n[X] * ((1 + np.log(S[i, X])) if (i, X) in learned else S[i, X])
    return K[0] - K[1]


def _rival_grid(adaptive):
    """``adaptive=False``: the construction fixed in §6 (dt 0.01 x 5000 cold, 0.1 x 300 warm).
    ``adaptive=True``: the re-run's step-size rule (§5.5), ``dt = min(0.1, 0.5 / max(1, |jump|))``
    with ``jump`` the change in the closed-form ``K_0 - K_1``, and at B count 0 the gap at the
    starting state (every trust at 1), and ``ceil(30 / dt)`` steps."""
    out = {}
    for arm, learned in E_ARMS.items():
        if adaptive:
            settle = sc.rival_settle_fn(learned, 1.0)
        else:
            cold = sc.rival_settle_fn(learned, 1.0, 5000)
            warm = sc.rival_settle_fn(learned, 1.0, 300)
        for d in E_DS:
            for seed in E_SEEDS:
                a, b = _e_draws(seed, d)
                st = (jnp.full(2, 0.5), jnp.zeros(len(learned))) if learned else jnp.full(2, 0.5)
                lg_arr, lg_cf = [], []
                prev = 0.0
                for k in range(NB_E + 1):
                    n, S = _e_stats(a, b, d, k)
                    eta = jnp.array([1.0 / n[X] if n[X] else 1.0 for (_, X) in learned])
                    Bs = float(sc.rival_closed_K(n, S, learned).min()) + 5.0
                    cf = _e_closed_logit(n, S, learned)
                    args = (st, jnp.asarray(n), jnp.asarray(S))
                    if adaptive:
                        # the jump the chooser must absorb: at B count 0 the gap at the STARTING state
                        # (every ell = 0, i.e. fixed trust); afterwards the change in the closed form
                        jump = abs(_e_closed_logit(n, S, ())) if k == 0 else abs(cf - prev)
                        dt = min(0.1, 0.5 / max(1.0, jump))
                        st = settle(*args, dt, eta, Bs, int(np.ceil(30.0 / dt)))
                    else:
                        st = (cold if k == 0 else warm)(*args, 0.01 if k == 0 else 0.1, eta, Bs)
                    prev = cf
                    p = np.asarray(st[0] if learned else st)
                    lg_arr.append(float(np.log(p[1]) - np.log(p[0])))
                    lg_cf.append(cf)
                out[(arm, d, seed)] = (np.array(lg_arr), np.array(lg_cf))
    return out


@pytest.fixture(scope="module")
def rival_runs():
    return _rival_grid(adaptive=False)


@pytest.fixture(scope="module")
def rival_rerun():
    return _rival_grid(adaptive=True)


def _first(lg):
    over = np.where(lg > 0)[0]
    return int(over[0]) if len(over) else None


def _e_summary(rival_runs, arm, d):
    firsts, firsts_cf, stays, slopes, gap, nonfinite = [], [], 0, [], 0.0, 0
    for seed in E_SEEDS:
        lg, cf = rival_runs[(arm, d, seed)]
        firsts.append(_first(lg)); firsts_cf.append(_first(cf))
        f = firsts[-1]
        stays += int(f is not None and np.all(lg[f:] > 0))
        slopes.append((lg[-1] - lg[1]) / (NB_E - 1))
        gap = float(np.nanmax([gap, np.nanmax(np.abs(lg - cf))]))
        nonfinite += int(not np.all(np.isfinite(lg)))
    done = [f for f in firsts if f is not None]
    med = float(np.median(done)) if done else None
    print(f"\n(e) {arm}, d={d:g}: first overtake by seed {firsts} (closed form {firsts_cf}); "
          f"overtook by B count {max(done) if done else None} in {len(done)} of 10 seeds; stayed ahead in "
          f"{stays}; median {med}; mean logit slope {np.mean(slopes):.4f}; max |logit - closed form| {gap:.2e} "
          f"(finite entries); non-finite runs {nonfinite}")
    return firsts, firsts_cf, med, float(np.mean(slopes))


E_FIRST = ("first run, construction as fixed in §6 (§5.5): the step size let a share go negative, NaN "
           "in seed 7 at B count 0 (and after the crossing in the fixed arm at d = 3)")


def _xf_d3(d):
    return pytest.param(d, marks=pytest.mark.xfail(strict=True, reason=E_FIRST)) if d == 3.0 else d


@pytest.mark.parametrize("d", [_xf_d3(d) for d in E_DS])
def test_e1_fixed_trust_crosses_at_n_A(rival_runs, d):
    firsts, cf, med, slope = _e_summary(rival_runs, "fixed", d)
    assert firsts == cf and abs(med - NA_E) <= 0.25 * NA_E
    assert abs(slope - d * d / 2) <= 0.1 * d * d / 2


@pytest.mark.parametrize("d", E_DS)
def test_e2_both_learn_cross_at_n_A_softer(rival_runs, d):
    firsts, cf, med, slope = _e_summary(rival_runs, "both learn", d)
    want = 0.5 * np.log(1 + d * d)
    assert firsts == cf and abs(med - NA_E) <= 0.25 * NA_E
    assert abs(slope - want) <= 0.1 * want


@pytest.mark.parametrize("d", [_xf_d3(d) for d in E_DS])
def test_e3_only_the_incumbent_explains_away(rival_runs, d):
    firsts, cf, med, _ = _e_summary(rival_runs, "only the incumbent learns", d)
    want = NA_E * d * d / np.log(1 + d * d)
    assert firsts == cf and abs(med - want) <= 0.25 * want


@pytest.mark.xfail(strict=True, reason=E_FIRST)
def test_e4_a_rival_wins_where_a_default_lost(rival_runs):
    tally = {arm: sum(_first(rival_runs[(arm, 3.0, s)][0]) is not None for s in E_SEEDS) for arm in E_ARMS}
    print(f"\n(e4) d=3: seeds in which the rival overtook by B count {NB_E}: {tally}")
    assert all(v == 10 for v in tally.values())


# The re-run (§5.5): the same predictions and definitions, with only the step-size rule changed.


@pytest.mark.parametrize("d", E_DS)
def test_e1_rerun(rival_rerun, d):
    _e1(rival_rerun, d)


def _e1(runs, d):
    firsts, cf, med, slope = _e_summary(runs, "fixed", d)
    assert firsts == cf and abs(med - NA_E) <= 0.25 * NA_E
    assert abs(slope - d * d / 2) <= 0.1 * d * d / 2


@pytest.mark.parametrize("d", E_DS)
def test_e2_rerun(rival_rerun, d):
    firsts, cf, med, slope = _e_summary(rival_rerun, "both learn", d)
    want = 0.5 * np.log(1 + d * d)
    assert firsts == cf and abs(med - NA_E) <= 0.25 * NA_E and abs(slope - want) <= 0.1 * want


@pytest.mark.parametrize("d", E_DS)
def test_e3_rerun(rival_rerun, d):
    firsts, cf, med, _ = _e_summary(rival_rerun, "only the incumbent learns", d)
    want = NA_E * d * d / np.log(1 + d * d)
    assert firsts == cf and abs(med - want) <= 0.25 * want


def test_e4_rerun(rival_rerun):
    tally = {arm: sum(_first(rival_rerun[(arm, 3.0, s)][0]) is not None for s in E_SEEDS) for arm in E_ARMS}
    print(f"\n(e4, re-run) d=3: seeds in which the rival overtook by B count {NB_E}: {tally}")
    assert all(v == 10 for v in tally.values())


# ---------------------------------------------------------------------------
# Test (f)
# ---------------------------------------------------------------------------

M_F, MUS_F, F_SEEDS = 8, np.arange(20) * 0.5, range(20)
T_SUM, D_EXTRA, T_CON = 60, 4, 150
SLICES = [(2 * j, 2 * j + 1) for j in range(M_F)]
_ring = np.zeros((M_F, M_F), int)
for _j in range(M_F):
    _ring[_j, (_j + 1) % M_F] = _ring[_j, (_j - 1) % M_F] = 1
F_WIRINGS = {"ring": _ring, "all-to-all": np.ones((M_F, M_F), int) - np.eye(M_F, dtype=int)}
EXTRA = {"ring": 4, "all-to-all": 1}


def _f_draws(seed, truth, T):
    return np.random.default_rng(seed).normal(truth, 1.0, size=(T, M_F))


def _entropy(w):
    w = np.clip(w, 1e-300, 1.0)
    return -np.sum(w * np.log(w), axis=-1)


def _ends_wrong(W, right):
    """(any holder, all holders) with weight > 0.99 on one hypothesis other than ``right``."""
    top = W.argmax(-1)
    wrong = (W.max(-1) > 0.99) & (top != right)
    return bool(wrong.any()), bool(wrong.all())


@pytest.fixture(scope="module")
def summaries():
    out = {}
    for truth, right in ((5.0, 10), (5.2, 10)):
        for wname, A in F_WIRINGS.items():
            for prov in (True, False):
                rows = []
                for seed in F_SEEDS:
                    dr = _f_draws(seed, truth, T_SUM)
                    Ys, held = sc.run_summaries(dr, MUS_F, A, prov, EXTRA[wname])
                    rows.append((dr, Ys, held))
                out[(truth, wname, prov)] = rows
    return out


F1_SIDE = "side variant (truth 5.2): per-step identity 1.4e-12, over the 1e-12 fixed in §7 (§5.6)"
F2_RING = ("refuted (§5.6): on the ring a holder holds fewer draws than the pooled mind early on; less "
           "confident at some step in 20 of 20 seeds (more confident from step 9)")
F2_SIDE = "side variant all-to-all (truth 5.2): 1 of 20 seeds less confident at an early step (§5.6)"
F3_REF = ("refuted as written (§5.6): all holders above 0.99 on the truth in 18 of 20; in seeds 10, 18 holder 4 "
          "cannot separate 5.0 from 4.5 on its own draws (0.984, 0.877)")


def _xf(value, reason):
    return pytest.param(value, marks=pytest.mark.xfail(strict=True, reason=reason))


@pytest.mark.parametrize("truth", [5.0, _xf(5.2, F1_SIDE)])
@pytest.mark.parametrize("wname", list(F_WIRINGS))
def test_f1_summaries_with_provenance_are_the_pooled_mind(summaries, wname, truth):
    worst_i, worst_ii, wrong = 0.0, 0.0, 0
    for dr, Ys, held in summaries[(truth, wname, True)]:
        for Y, papers in zip(Ys, held):
            for j in range(M_F):
                xs = np.array([dr[t, a] for (a, t) in papers[j]])
                mind = sc.weights(sc.stats_inc(MUS_F, len(xs), xs.sum()))
                worst_i = max(worst_i, float(np.max(np.abs(sc.weights(Y[j]) - mind))))
        pooled = sc.weights(sc.stats_inc(MUS_F, dr.size, dr.sum()))
        worst_ii = max(worst_ii, float(np.max(np.abs(sc.weights(Ys[-1]) - pooled))))
        wrong += _ends_wrong(sc.weights(Ys[-1]), 10)[0]
    print(f"\n(f1) truth {truth}, {wname}: max |holder - one mind on its draws| = {worst_i:.2e}; "
          f"final max |holder - pooled| = {worst_ii:.2e}; seeds ending on a wrong hypothesis {wrong} of 20")
    assert worst_i < 1e-12 and worst_ii < 1e-12
    if truth == 5.0:
        assert wrong == 0


@pytest.mark.parametrize("truth,wname", [
    pytest.param(5.0, "ring", marks=pytest.mark.xfail(strict=True, reason=F2_RING)),
    pytest.param(5.2, "ring", marks=pytest.mark.xfail(strict=True, reason=F2_RING)),
    (5.0, "all-to-all"),
    pytest.param(5.2, "all-to-all", marks=pytest.mark.xfail(strict=True, reason=F2_SIDE))])
def test_f2_summaries_without_provenance_overcount(summaries, wname, truth):
    violations, first_ok, wrong_any, wrong_all = 0, [], 0, 0
    for dr, Ys, NS in summaries[(truth, wname, False)]:
        bad_steps = []
        for t, Y in enumerate(Ys):
            upto = dr[: min(t + 1, T_SUM)]
            pooled = sc.weights(sc.stats_inc(MUS_F, upto.size, upto.sum()))
            if np.any(_entropy(sc.weights(Y)) > _entropy(pooled) + 1e-12):
                bad_steps.append(t + 1)
        violations += bool(bad_steps)
        first_ok.append(max(bad_steps) + 1 if bad_steps else 1)
        a, b = _ends_wrong(sc.weights(Ys[-1]), 10)
        wrong_any += a; wrong_all += b
    prov_wrong = sum(_ends_wrong(sc.weights(Ys[-1]), 10)[0] for _, Ys, _ in summaries[(truth, wname, True)])
    print(f"\n(f2) truth {truth}, {wname}: seeds with a step where some holder is LESS confident than the "
          f"pooled mind {violations} of 20 (more confident from step {max(first_ok)} on in all); ending on a "
          f"wrong hypothesis: without provenance any-holder {wrong_any}, all-holders {wrong_all} of 20; "
          f"with provenance {prov_wrong} of 20")
    assert violations == 0
    assert wrong_any > prov_wrong


@pytest.fixture(scope="module")
def conclusions():
    out = {}
    for variant, truth, right in (("in slice", 5.0, 10), ("no slice", 8.5, 17)):
        for wname, A in F_WIRINGS.items():
            for rep in (True, False):
                out[(variant, wname, rep)] = [
                    sc.run_conclusions(_f_draws(seed, truth, T_CON), MUS_F, A, SLICES, rep)
                    for seed in F_SEEDS]
        out[(variant, "bounded")] = {
            sl: [sc.run_bounded_mind(_f_draws(seed, truth, T_CON), MUS_F, sl) for seed in F_SEEDS]
            for sl in ((0, 1), (10, 11))}
    return out


def _holding(W_last, right):
    return W_last[:, right] > 0.99


def _arrivals(W, right):
    arr = np.full(M_F, -1)
    for t, Wt in enumerate(W):
        for j in range(M_F):
            if arr[j] < 0 and Wt[j, right] > 0.5:
                arr[j] = t + 1
    return arr


@pytest.mark.parametrize("wname", [_xf(w, F3_REF) for w in F_WIRINGS])
def test_f3_replication_spreads_the_truth(conclusions, wname):
    runs = conclusions[("in slice", wname, True)]
    all_hold = sum(bool(_holding(W[-1], 10).all()) for W, _ in runs)
    wrong = sum(_ends_wrong(W[-1], 10)[0] for W, _ in runs)
    msg = f"\n(f3) {wname}, truth in slice: all holders hold the truth in {all_hold} of 20; any holder ends wrong in {wrong}"
    ok = all_hold == 20
    if wname == "ring":
        by_dist = {d: [] for d in range(5)}
        finders = []
        for W, _ in runs:
            arr = _arrivals(W, 10)
            if np.any(arr < 0):
                continue
            f = int(np.argmin(arr)); finders.append((f, int(arr[f])))
            for j in range(M_F):
                d = min((j - f) % M_F, (f - j) % M_F)
                by_dist[d].append(int(arr[j] - arr[f]))
        med = {d: float(np.median(v)) for d, v in by_dist.items() if v}
        ds = np.array(sorted(med)); ms = np.array([med[d] for d in ds])
        slope = float(np.polyfit(ds, ms, 1)[0])
        msg += (f"; first finders (holder, step) {finders}; median arrival delay by ring distance {med}; "
                f"slope {slope:.2f} per hop (window [5, 10])")
        ok &= 5.0 <= slope <= 10.0
    none = conclusions[("no slice", wname, True)]
    any_hold_none = sum(bool(_holding(W[-1], 17).any()) for W, _ in none)
    ends_on = [int(np.bincount(W[-1].argmax(-1), minlength=20).argmax()) for W, _ in none]
    msg += f"; truth in no slice: any holder holds it in {any_hold_none} of 20 (most common final conclusion mu = {MUS_F[max(set(ends_on), key=ends_on.count)]})"
    print(msg)
    ok &= any_hold_none == 0
    assert ok


def test_f3_bounded_mind_finds_only_what_it_holds(conclusions):
    res = {}
    for variant, right in (("in slice", 10), ("no slice", 17)):
        for sl, runs in conclusions[(variant, "bounded")].items():
            res[(variant, sl)] = sum(bool(W[-1][right] > 0.99) for W in runs)
    print(f"\n(f3) bounded mind (K = 2, every draw): holds the truth in {res} of 20")
    assert res[("in slice", (10, 11))] == 20 and res[("in slice", (0, 1))] == 0
    assert res[("no slice", (10, 11))] == 0 and res[("no slice", (0, 1))] == 0


@pytest.mark.parametrize("wname", list(F_WIRINGS))
def test_f4_without_replication_wrong_conclusions_spread(conclusions, wname):
    out = {}
    for rep in (True, False):
        runs = conclusions[("in slice", wname, rep)]
        ends = [_ends_wrong(W[-1], 10) for W, _ in runs]
        out[rep] = (sum(a for a, _ in ends), sum(b for _, b in ends),
                    sum(bool(_holding(W[-1], 10).all()) for W, _ in runs))
    print(f"\n(f4) {wname}, truth in slice: ending on a wrong hypothesis (any holder, all holders) / all "
          f"holders hold the truth: with replication {out[True]}, without {out[False]} (of 20)")
    assert out[False][0] > 0 and out[False][0] > out[True][0]


def test_f3_the_parts_that_held(conclusions):
    """(f3)'s other clauses, asserted separately so the strict xfail above covers only the 20-of-20 clause."""
    for wname in F_WIRINGS:
        runs = conclusions[("in slice", wname, True)]
        assert all(bool(np.all(W[-1].argmax(-1) == 10)) for W, _ in runs)        # conclusion = truth, 20 of 20
        assert sum(bool(_holding(W[-1], 17).any()) for W, _ in conclusions[("no slice", wname, True)]) == 0
    by_dist = {d: [] for d in range(5)}
    for W, _ in conclusions[("in slice", "ring", True)]:
        arr = _arrivals(W, 10)
        f = int(np.argmin(arr))
        for j in range(M_F):
            by_dist[min((j - f) % M_F, (f - j) % M_F)].append(int(arr[j] - arr[f]))
    ds = np.arange(5)
    slope = float(np.polyfit(ds, [np.median(by_dist[d]) for d in ds], 1)[0])
    print(f"\n(f3, held parts) conclusion = truth in 20 of 20 both wirings; truth in no slice held by no one; "
          f"ring slope {slope:.2f} per hop")
    assert 5.0 <= slope <= 10.0
