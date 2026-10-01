"""Test (h): bettors paid by being right, natively (``arrangements/science_payment.py``).

Predictions fixed before code in ``notes/SCIENCE-PAYMENT-PREDICTIONS-2026-10-01.md`` §2 and its "Numbers
fixed before code" (quoted in the module docstring); results in §6 of that note. Each test prints its
counts (run with ``PYTEST_ADDOPTS=-s`` to see them).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp

from arrangements.science_payment import (all_to_all, hub, hypotheses, listener_sensitivity, locked_wrong,
                                          market_gated_run, market_run, ring, seeds)

M, T, C, S0 = 10, 200, 0.02, 1.0
H = hypotheses()
PI = H[0]
ZERO = np.zeros((M, M))


def softmax(Y):
    return np.exp(Y - logsumexp(Y, axis=-1, keepdims=True))


@pytest.fixture(scope="module")
def data():
    return seeds()


@pytest.fixture(scope="module")
def paid(data):
    Y0s, Xs = data
    return {"dogmatic": market_run(Y0s, Xs, ZERO, c=C), "updating": market_run(Y0s, Xs, ZERO, beta=1.0, c=C)}


H1_TOL = ("failed its fixed check in the recorded run (worst 1.14e-12 > 1e-12, §6); diagnosed after: the gap "
           "is the float64 closed form's (exp of a cumulative log sum), not the arrangement's; see the next test")


@pytest.mark.xfail(strict=True, reason=H1_TOL)
def test_h1_wealth_is_the_likelihood(data):
    Y0s, Xs = data
    worst = 0.0
    for arm, beta in (("dogmatic", 0.0), ("updating", 1.0)):
        _, S = market_run(Y0s, Xs, ZERO, beta=beta, c=0.0)
        for s in range(20):
            q0 = softmax(Y0s[s])
            if arm == "dogmatic":
                logL = np.cumsum(np.log(q0 @ H)[:, Xs[s]], axis=1)              # (m, T)
            else:                                                               # marginal likelihood
                logLi = np.cumsum(np.log(H[:, Xs[s]]), axis=1)                  # (K, T)
                logL = logsumexp(np.log(q0)[:, :, None] + logLi[None], axis=1)
            t = np.arange(1, T + 1)
            closed = S0 * np.exp(t * np.log(4.0) + logL).T                      # (T, m)
            rel = np.max(np.abs(S[s, 1:] - closed) / closed)
            worst = max(worst, rel)
            if arm == "dogmatic":                                               # test (a)'s population
                y = np.zeros(M)
                for k, x in enumerate(Xs[s]):
                    y = y + np.log(q0 @ H[:, x])
                    share = S[s, k + 1] / S[s, k + 1].sum()
                    assert np.max(np.abs(share - softmax(y))) < 1e-12
    print(f"\n(h1) worst relative supply error {worst:.2e}")
    assert worst < 1e-12


def test_h1_against_exact_rationals_post_hoc(data):
    """Post hoc (disclosed, §6): the native supply against the identity computed in exact rational
    arithmetic from the same float64 predictives and ``H``; 5 seeds to keep it fast."""
    from fractions import Fraction
    Y0s, Xs = data
    worst = {}
    for arm, beta in (("dogmatic", 0.0), ("updating", 1.0)):
        _, S = market_run(Y0s, Xs, ZERO, beta=beta, c=0.0)
        w = 0.0
        for s in range(5):
            q0 = softmax(Y0s[s])
            for j in range(M):
                if arm == "dogmatic":
                    p = [Fraction(float(v)) for v in q0[j] @ H]
                    ex = Fraction(1)
                    for t, x in enumerate(Xs[s], 1):
                        ex *= 4 * p[x]
                        w = max(w, abs(float((Fraction(float(S[s, t, j])) - ex) / ex)))
                else:
                    Li = [Fraction(1)] * 4
                    for t, x in enumerate(Xs[s], 1):
                        Li = [Li[i] * Fraction(float(H[i, x])) for i in range(4)]
                        ex = 4 ** t * sum(Fraction(float(q0[j, i])) * Li[i] for i in range(4))
                        w = max(w, abs(float((Fraction(float(S[s, t, j])) - ex) / ex)))
        worst[arm] = w
    print(f"\n(h1, post hoc) native vs exact rationals: {worst}")
    assert max(worst.values()) < 1e-12


def exact_death(q0, X):
    """Dogmatic bettor: first step ``t`` with ``c sum_{k<=t} 1/Pi_k >= s0`` (0 if none)."""
    G = 4.0 * (q0 @ H)[X]
    part = C * np.cumsum(1.0 / np.cumprod(G))
    hit = np.nonzero(part >= S0)[0]
    return int(hit[0]) + 1 if hit.size else 0


def test_h2_dogmatic_die_with_their_idea(data, paid):
    Y0s, Xs = data
    _, S = paid["dogmatic"]
    below = below_dead = above = above_dead = 0
    for s in range(20):
        q0 = softmax(Y0s[s])
        for j in range(M):
            died = int(np.argmax(S[s, :, j] == 0.0)) if S[s, -1, j] == 0.0 else 0
            assert died == exact_death(q0[j], Xs[s])                              # the exact condition
            g = np.log(4.0) + PI @ np.log(q0[j] @ H)
            if g <= 0:
                below += 1; below_dead += died > 0
            else:
                above += 1; above_dead += died > 0
    print(f"\n(h2) g<=0: {below_dead} of {below} dead; g>0: {above_dead} of {above} dead")
    assert below_dead == below == 136 and (above, above_dead) == (64, 14)


def test_h3_updating_let_the_idea_die(data, paid):
    Y0s, Xs = data
    _, Sd = paid["dogmatic"]
    Y, S = paid["updating"]
    dead_d, dead_u = int((Sd[:, -1] == 0).sum()), int((S[:, -1] == 0).sum())
    agree = before = 0
    for s in range(20):
        qt = softmax(Y[s])[:, :, 0]                                             # weight on the truth
        for j in range(M):
            tau = int(np.argmax(qt[:, j] > 0.99))
            assert qt[tau, j] > 0.99
            need = C * np.sum(1.0 / np.cumprod(4.0 * PI[Xs[s, tau:]]))
            dead = S[s, -1, j] == 0.0
            agree += (S[s, tau, j] < need) == dead
            before += dead and S[s, tau, j] == 0.0
    print(f"\n(h3) dead: dogmatic {dead_d} of 200, updating {dead_u} of 200; "
          f"mechanism rule agrees {agree} of 200; {before} dead before their belief moved")
    assert (dead_d, dead_u) == (150, 76) and agree == 199


NETS = {"ring": ring(M), "all": all_to_all(M), "hub": hub(M)}


def community(data, W, alpha_agree=0.0, alpha_evid=0.0, beta=1.0):
    Y0s, Xs = data
    Y, S = market_run(Y0s, Xs, W, alpha_agree=alpha_agree, alpha_evid=alpha_evid, beta=beta, c=C)
    lock = [locked_wrong(Y[s, -1]) for s in range(20)]
    dead = (S[:, -1] == 0.0).sum(1)
    finite = bool(np.all(np.isfinite(Y)) and np.all(np.isfinite(S)))
    return lock, dead, finite


@pytest.mark.parametrize("net", ["ring", "all", "hub"])
def test_h4_echo_chamber_falls_apart_together(data, net):
    lock, dead, finite = community(data, NETS[net], alpha_evid=10.0)
    locked = [s for s in range(20) if lock[s]]
    print(f"\n(h4) evidence alpha=10, {net}: locked wrong {len(locked)} of 20; all ten dead in "
          f"{sum(dead[s] == M for s in locked)} of those; total collapse {int((dead == M).sum())} of 20; "
          f"dead per seed {dead.tolist()}; finite {finite}")
    assert finite and all(dead[s] == M for s in locked)


H4_AGREE10 = ("predicted before code to fail: the synchronous agree step at alpha = 10 diverges (eigenvalues -19, "
              "-10.1, -15.3); 20 of 20 collapse on every network")
H4_AGREE05 = ("refuted (§6): 3, 4, 4 of 20 collapse at the stable alpha = 0.5; the no-talk updating arm already "
              "collapses in 3 of 20, since every bettor faces the same races")


@pytest.mark.parametrize("alpha", [pytest.param(10.0, marks=pytest.mark.xfail(strict=True, reason=H4_AGREE10)),
                                   pytest.param(0.5, marks=pytest.mark.xfail(strict=True, reason=H4_AGREE05))])
@pytest.mark.parametrize("net", ["ring", "all", "hub"])
def test_h4_agree_no_collapse(data, net, alpha):
    lock, dead, finite = community(data, NETS[net], alpha_agree=alpha)
    print(f"\n(h4) agree alpha={alpha}, {net}: total collapse {int((dead == M).sum())} of 20; locked wrong "
          f"{sum(lock)}; dead bettors {int(dead.sum())} of 200; finite {finite}")
    assert int((dead == M).sum()) == 0


def test_h4_shared_races_collapse_without_talk(data):
    """Found after the recorded run (§6): with no talk at all, updating bettors collapse together in the
    seeds where the shared early races go against the favourite."""
    lock, dead, finite = community(data, ZERO)
    collapsed = [s for s in range(20) if dead[s] == M]
    print(f"\n(h4) no talk, updating: total collapse {len(collapsed)} of 20 (seeds {collapsed}); locked wrong {sum(lock)}")
    assert finite and collapsed == [1, 9, 12] and sum(lock) == 0


def perron(W):
    ev, L = np.linalg.eig(W.T)
    v = np.real(L[:, np.argmax(ev.real)])
    return v / v.sum()


@pytest.mark.parametrize("net", ["ring", "all", "hub"])
def test_h5_consensus_is_the_centrality_weighted_start(data, net):
    Y0s, Xs = data
    W = NETS[net]
    v = perron(W)
    assert np.allclose(v, [0.25] + [1 / 12] * 9 if net == "hub" else np.full(M, 0.1), atol=1e-12)
    Y, _ = market_run(Y0s, Xs, W, alpha_agree=0.5, beta=0.0, c=C)
    lam0 = Y0s - Y0s[:, :, -1:]
    lamT = Y[:, -1] - Y[:, -1, :, -1:]
    target = np.einsum("j,sjk->sk", v, lam0)
    err = np.max(np.abs(lamT - target[:, None, :]))
    off = np.max(np.abs(target - lam0.mean(1)))
    print(f"\n(h5) agree, no outcomes, {net}: max |lam_j(200) - sum v lam(0)| = {err:.2e}; "
          f"distance of that consensus from the plain mean {off:.3f}")
    assert err < 1e-8


H5_AGREE = ("its lock half fails as stated before code (0 and 0 of 20 locked wrong); its collapse half holds, "
            "ring 3 and hub 9 of 20")


@pytest.mark.parametrize("rule,alpha", [("evidence", 1.0),
                                        pytest.param("agree", 0.5, marks=pytest.mark.xfail(strict=True, reason=H5_AGREE))])
def test_h5_hub_wrong(rule, alpha):
    d = seeds(hub_wrong=True)
    kw = {"alpha_evid": alpha} if rule == "evidence" else {"alpha_agree": alpha}
    out = {net: community(d, NETS[net], **kw) for net in ("ring", "hub")}
    lk = {net: sum(out[net][0]) for net in out}
    col = {net: int((out[net][1] == M).sum()) for net in out}
    dd = {net: int(out[net][1].sum()) for net in out}
    print(f"\n(h5) hub wrong, {rule} alpha={alpha}: locked wrong ring {lk['ring']}, hub {lk['hub']} of 20; "
          f"total collapse ring {col['ring']}, hub {col['hub']}; dead bettors ring {dd['ring']}, hub {dd['hub']} "
          f"of 200")
    assert lk["hub"] > lk["ring"] and col["hub"] > col["ring"]


# --- §7 correction: the dead fall apart (results in the note's §8) -----------------------------------

R, S1 = 0.1, 0.01
SIX = {("evidence", 10.0): {"ring": 18, "all": 18, "hub": 17},
       ("agree", 10.0): {"ring": 20, "all": 20, "hub": 20},
       ("agree", 0.5): {"ring": 3, "all": 4, "hub": 4}}


def death_steps(S):
    """First step with supply 0, per seed and bettor (-1 if alive at the end)."""
    return np.where(S[:, -1] == 0.0, np.argmax(S == 0.0, axis=1), -1)


def gated(data, W, rule=None, alpha=0.0, beta=1.0):
    Y0s, Xs = data
    kw = {} if rule is None else ({"alpha_evid": alpha} if rule == "evidence" else {"alpha_agree": alpha})
    return market_gated_run(Y0s, Xs, W, beta=beta, c=C, r=R, s1=S1, **kw)


@pytest.mark.parametrize("beta", [0.0, 1.0])
def test_h7_dissolution_changes_no_death_step(data, beta):
    Y0s, Xs = data
    _, S6 = market_run(Y0s, Xs, ZERO, beta=beta, c=C)
    _, S7 = gated(data, ZERO, beta=beta)
    d6, d7 = death_steps(S6), death_steps(S7)
    print(f"\n(§7) beta={beta}: death steps differing from §6 in {int((d6 != d7).sum())} of 200; dead {int((d7 >= 0).sum())}")
    assert np.array_equal(d6, d7)


H6_CASES = [("evidence", 10.0, "ring"), ("agree", 0.5, "ring"), ("agree", 0.5, "hub"), ("evidence", 10.0, "hub")]


@pytest.mark.parametrize("rule,alpha,net", H6_CASES)
def test_h6_dead_are_not_heard(data, rule, alpha, net):
    Y, S = gated(data, NETS[net], rule, alpha)
    Y0s, Xs = data
    worst, pairs = 0.0, 0
    for s in range(0, 20, 4):
        dead = np.nonzero(S[s, -1] == 0.0)[0]
        if dead.size == 0:
            continue
        x1h = jax.nn.one_hot(jnp.asarray(Xs[s]), 4, dtype=float)
        kw = {"alpha_evid": alpha, "alpha_agree": 0.0} if rule == "evidence" else {"alpha_agree": alpha, "alpha_evid": 0.0}
        J = listener_sensitivity(Y[s, -1], S[s, -1], x1h, NETS[net], beta=1.0, c=C, r=R, s1=S1, **kw)
        for k in dead:
            for j in range(M):
                if j != k:
                    worst = max(worst, float(np.max(np.abs(J[j, :, k, :]))))
                    pairs += 1
    print(f"\n(h6) {rule} alpha={alpha}, {net}: max |d(listener forcing)/d(dead state)| = {worst:.1e} over {pairs} pairs")
    assert pairs > 0 and worst < 1e-12


H6_UNIFORM = "predicted before code to fail (§7 numbers): |lam| at death is far above 0.8 for most dead bettors"


@pytest.mark.xfail(strict=True, reason=H6_UNIFORM)
@pytest.mark.parametrize("rule,alpha,net", [(None, 0.0, "ring"), ("evidence", 10.0, "ring"), ("agree", 0.5, "ring")])
def test_h6_dead_dissolve_to_uniform(data, rule, alpha, net):
    W = ZERO if rule is None else NETS[net]
    Y, S = gated(data, W, rule, alpha)
    d = death_steps(S)
    ok = tot = 0
    for s in range(20):
        for j in range(M):
            if 0 <= d[s, j] <= T - 50:
                q = softmax(Y[s, d[s, j] + 50, j])
                tot += 1
                ok += np.max(np.abs(q - 0.25)) < 1e-3
    print(f"\n(h6) {rule or 'no talk'} alpha={alpha}: within 1e-3 of uniform 50 steps after death: {ok} of {tot}")
    assert tot > 0 and ok == tot


H4P = "refuted (§8): agree alpha = 10 collapses {} of 20 on the {}, against §6's 20"


@pytest.mark.parametrize("rule,alpha", [("evidence", 10.0), ("agree", 10.0), ("agree", 0.5)])
@pytest.mark.parametrize("net", ["ring", "all", "hub"])
def test_h4prime_collapse_counts(data, rule, alpha, net, request):
    if rule == "agree" and alpha == 10.0 and net in ("ring", "hub"):
        request.applymarker(pytest.mark.xfail(strict=True, reason=H4P.format(12 if net == "ring" else 17, net)))
    Y, S = gated(data, NETS[net], rule, alpha)
    dead = (S[:, -1] == 0.0).sum(1)
    lock = sum(locked_wrong(Y[s, -1]) for s in range(20))
    col = int((dead == M).sum())
    print(f"\n(h4') {rule} alpha={alpha}, {net}: total collapse {col} of 20 (§6: {SIX[(rule, alpha)][net]}); "
          f"locked wrong {lock}; dead bettors {int(dead.sum())}")
    assert abs(col - SIX[(rule, alpha)][net]) <= 2


def lock_step(Y, wrong_only=True):
    """First step at which bettors 1..9 are all above 0.99 on one wrong hypothesis (-1 if never)."""
    q = softmax(Y[:, 1:])                                   # (T+1, 9, K)
    top = q.argmax(-1)
    ok = np.all(top == top[:, :1], axis=1) & (top[:, 0] != 0) & np.all(q.max(-1) > 0.99, axis=1)
    return int(np.argmax(ok)) if ok.any() else -1


H5P = ("refuted as fixed (§8): bettor 0 dies after the others lock in 20 of 20 on the hub network, yet 0 of 20 "
       "end locked wrong, since the dead dissolve; the end-state lock no longer measures a lock")


@pytest.mark.parametrize("rule,alpha", [pytest.param("evidence", 1.0, marks=pytest.mark.xfail(strict=True, reason=H5P)),
                                        ("agree", 0.5)])
def test_h5prime_hub_wrong_falls_apart(rule, alpha):
    d = seeds(hub_wrong=True)
    res = {}
    for net in ("ring", "hub"):
        Y, S = gated(d, NETS[net], rule, alpha)
        dead = (S[:, -1] == 0.0).sum(1)
        dstep = death_steps(S)[:, 0]
        before = after = never_lock = alive = 0
        for s in range(20):
            L = lock_step(Y[s])
            if dstep[s] < 0:
                alive += 1
            elif L < 0:
                never_lock += 1
            elif dstep[s] < L:
                before += 1
            else:
                after += 1
        res[net] = (sum(locked_wrong(Y[s, -1]) for s in range(20)), int((dead == M).sum()), int(dead.sum()),
                    before, after, never_lock, alive)
        print(f"\n(h5') hub wrong, {rule} alpha={alpha}, {net}: locked wrong {res[net][0]}, total collapse "
              f"{res[net][1]}, dead {res[net][2]} of 200; bettor 0 dies before the others lock {before}, after "
              f"{after}, dies and they never lock {never_lock}, survives {alive}")
    if rule == "evidence":                                  # §6: hub 20 locked wrong
        assert res["hub"][0] < 20 if res["hub"][3] > 0 else res["hub"][0] == 20


@pytest.mark.xfail(strict=True, reason="not testable at §2's settings (§8): with no outcomes read every bettor "
                   "dies, so no seed has bettor 0 dead and the nine alive")
def test_h5prime_consensus_after_hub_dies():
    """Agree 0.5, no outcomes read, hub wrong: after bettor 0 dies, the nine living agree on the plain mean of
    their beliefs at its death (in seeds where no one else dies)."""
    Y0s, Xs = seeds(hub_wrong=True)
    Y, S = market_gated_run(Y0s, Xs, hub(M), alpha_agree=0.5, beta=0.0, c=C, r=R, s1=S1)
    d = death_steps(S)
    worst, used = 0.0, 0
    for s in range(20):
        if d[s, 0] < 0 or np.any(d[s, 1:] >= 0) or d[s, 0] > T - 150:
            continue
        lam = Y[s, :, 1:] - Y[s, :, 1:, -1:]
        target = lam[d[s, 0], :].mean(0)
        worst = max(worst, float(np.max(np.abs(lam[-1] - target))))
        used += 1
    print(f"\n(h5') agree, no outcomes, hub wrong: hub dead with the rest alive in {used} of 20; "
          f"max |lam_j(200) - mean of the nine at hub death| = {worst:.1e}")
    assert used > 0 and worst < 1e-8


def ever_locked_living(Y, S):
    """Post hoc (§8): first step at which every living bettor is above 0.99 on one wrong hypothesis (-1 never)."""
    q = softmax(Y)
    for t in range(Y.shape[0]):
        alive = S[t] > 0
        if not alive.any():
            return -1
        top = q[t, alive].argmax(-1)
        if np.all(top == top[0]) and top[0] != 0 and np.all(q[t, alive].max(-1) > 0.99):
            return t
    return -1


@pytest.mark.parametrize("case,expect", [(("evidence", 10.0, "ring", False), 17), (("evidence", 10.0, "all", False), 18),
                                         (("evidence", 10.0, "hub", False), 17), (("evidence", 1.0, "ring", True), 12),
                                         (("evidence", 1.0, "hub", True), 20)])
def test_h4prime_locks_among_the_living_post_hoc(case, expect):
    """Post hoc (§8): a community that ever locks wrong among its living members ends all dead."""
    rule, alpha, net, wrong = case
    d = seeds(hub_wrong=wrong)
    Y, S = gated(d, NETS[net], rule, alpha)
    L = [ever_locked_living(Y[s], S[s]) for s in range(20)]
    locked = [s for s in range(20) if L[s] >= 0]
    alldead = sum(S[s, -1].max() == 0.0 for s in locked)
    print(f"\n(h4', post hoc) {rule} alpha={alpha}, {net}{' hub wrong' if wrong else ''}: ever locked wrong among "
          f"the living {len(locked)} of 20; all dead in {alldead} of those")
    assert len(locked) == expect and alldead == expect
