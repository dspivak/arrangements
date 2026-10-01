"""Third round, stage 1: tests (c) and (f)'s summaries natively in the book's mathematics.

Predictions, fixed before ``arrangements/science_native.py`` existed
(``notes/SCIENCE-SCALES-PREDICTIONS-2026-10-01.md`` §8 and its "Numbers fixed before code (stage 1)";
verdicts in §5.7):

  (n1) (c) rebuilt as ``m`` holder boxes and a world box composed by wiring gives round two's
  trajectories to round-off (< 1e-10), in both arms; the composite equals the monolithic box.

  (n2) Journal arm: a holder reading the journal equals the pooled mind on the data published so far,
  to round-off. No-provenance arm: the same trajectories as round two's bookkeeping version (same
  arithmetic), hence the same count, 7 of 20 seeds locked wrong.

Stage-1 numbers: (n1) holds iff at every step ``max |Y_native - Y_mono| <= 1e-10 max(1, max |Y_mono|)``
and every seed's round-two classification is reproduced. (n2) no provenance: a holder box cannot
publish what arrives in the same step, so it carries a register and native state ``s + 2`` is round
two's step ``s``; holds iff weights agree to 1e-10 and ``Y`` relatively to 1e-12 at every step, and the
counts are 7 (truth 5.0) and 9 (truth 5.2), any and all, both wirings. The lag variant (no register)
was computed in numpy before code at the same counts, 7 and 9. Journal: native state ``s + 3`` holds
the pooled mind on draw rows ``0..s``; holds iff weights equal the pooled closed form to 1e-12 at every
step and 0 of 20 seeds end wrong at truth 5.0.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arrangements import science as sc
from arrangements import science_native as nv
from tests import test_science_scales as r2


MUS_C = np.arange(r2.NH_, dtype=float)


@partial(jax.jit, static_argnames=("W_key", "alpha", "beta", "arm", "steps"))
def _mono_run(Y0, LL, W_key, alpha, beta, arm, steps):
    """Round two's monolithic box, run by ``Phi`` (traced data, so one compile per setting)."""
    arr = sc.account_community(Y0, jnp.asarray(np.asarray(W_key)), alpha, beta, LL, arm)
    m, n = Y0.shape
    return nv.run_positions(arr, steps)[:, :-1].reshape(steps + 1, m, n)


def _mono(Y0, LL, W, alpha, beta, arm, steps):
    W_key = tuple(tuple(float(v) for v in row) for row in np.asarray(W))
    return np.asarray(_mono_run(jnp.asarray(Y0), jnp.asarray(LL), W_key, float(alpha), float(beta), arm, steps))


def _draws_c(s, T):
    rng = np.random.default_rng(s)
    Y0 = rng.normal(size=(r2.M_, r2.NH_))
    X = rng.normal(2.0, 1.0, size=(T, r2.M_))
    _, LL = r2._seed(s, T)
    return Y0, X, LL


N1_SETTINGS = ([(a, 1.0, 100) for a in r2.ALPHAS] + [(0.5, 0.0, 400), (1.0, 0.0, 100)])


@pytest.mark.parametrize("arm", ["agree", "evidence"])
@pytest.mark.parametrize("wiring", list(r2.WIRINGS))
def test_n1_wired_holders_are_the_monolithic_box(arm, wiring):
    W = np.asarray(r2.WIRINGS[wiring])
    worst, same, total, exact = 0.0, 0, 0, 0
    for alpha, beta, T in N1_SETTINGS:
        if beta == 0.0 and (arm, alpha) not in (("agree", 0.5), ("evidence", 1.0)):
            continue
        for s in r2.SEEDS:
            Y0, X, LL = _draws_c(s, T)
            Yn = np.asarray(nv.community_native_run(Y0, X, W, alpha, beta, MUS_C, arm, T))
            Ym = _mono(Y0, LL, W, alpha, beta, arm, T)
            scale = np.maximum(1.0, np.abs(Ym).max(axis=(1, 2)))
            err = np.abs(Yn - Ym).max(axis=(1, 2)) / scale
            worst = max(worst, float(np.nanmax(err)) if np.all(np.isfinite(Ym)) else worst)
            exact += bool(np.array_equal(Yn, Ym, equal_nan=True))
            cn = r2._classify(Yn[-1]) if np.all(np.isfinite(Yn[-1])) else "nonfinite"
            cm = r2._classify(Ym[-1]) if np.all(np.isfinite(Ym[-1])) else "nonfinite"
            same += cn == cm
            total += 1
    print(f"\n(n1) {arm} {wiring}: {total} runs; worst relative |native - monolithic| = {worst:.2e}; "
          f"bit-identical {exact} of {total}; same classification {same} of {total}")
    assert worst <= 1e-10 and same == total


def _summ(truth, wname, arm, steps):
    A = r2.F_WIRINGS[wname]
    out = []
    for seed in r2.F_SEEDS:
        dr = r2._f_draws(seed, truth, r2.T_SUM)
        Y, J = nv.summaries_native_run(dr, A, r2.MUS_F, arm, steps)
        out.append((dr, Y, J))
    return out


@pytest.mark.parametrize("truth,expected", [(5.0, 7), (5.2, 9)])
@pytest.mark.parametrize("wname", list(r2.F_WIRINGS))
def test_n2_no_provenance_is_round_two(wname, truth, expected):
    extra = r2.EXTRA[wname]
    worst_w, worst_y, wrong_any, wrong_all = 0.0, 0.0, 0, 0
    for dr, Y, _ in _summ(truth, wname, "noprov", r2.T_SUM + extra + 1):
        Ys, _ = sc.run_summaries(dr, r2.MUS_F, r2.F_WIRINGS[wname], False, extra)
        Ys = np.asarray(Ys)
        Yn = Y[2:]
        worst_w = max(worst_w, float(np.max(np.abs(sc.weights(Yn) - sc.weights(Ys)))))
        scale = np.abs(Ys).max(axis=(1, 2), keepdims=True)
        worst_y = max(worst_y, float(np.max(np.abs(Yn - Ys) / scale)))
        a, b = r2._ends_wrong(sc.weights(Yn[-1]), 10)
        wrong_any += a; wrong_all += b
    print(f"\n(n2 no provenance) truth {truth}, {wname}: worst |weights native - round two| = {worst_w:.2e}, "
          f"worst relative |Y| = {worst_y:.2e}; ending on a wrong hypothesis any/all {wrong_any}/{wrong_all} of 20")
    assert worst_w <= 1e-10 and worst_y <= 1e-12 and wrong_any == expected and wrong_all == expected


@pytest.mark.parametrize("truth,expected", [(5.0, 7), (5.2, 9)])
@pytest.mark.parametrize("wname", list(r2.F_WIRINGS))
def test_n2_lag_variant_counts(wname, truth, expected):
    """The plain box (no register) publishes its stock before its draw: numpy fixed 7/7 and 9/9."""
    extra = r2.EXTRA[wname]
    wrong_any, wrong_all, counts = 0, 0, None
    for dr, Y, _ in _summ(truth, wname, "lag", r2.T_SUM + extra):
        a, b = r2._ends_wrong(sc.weights(Y[-1]), 10)
        wrong_any += a; wrong_all += b
    print(f"\n(n2 lag) truth {truth}, {wname}: ending on a wrong hypothesis any/all {wrong_any}/{wrong_all} of 20")
    assert wrong_any == expected and wrong_all == expected


@pytest.mark.parametrize("truth", [5.0, 5.2])
@pytest.mark.parametrize("wname", list(r2.F_WIRINGS))
def test_n2_journal_reader_is_the_pooled_mind(wname, truth):
    steps = r2.T_SUM + 2
    worst, wrong, last_k = 0.0, 0, 0
    for dr, Y, J in _summ(truth, wname, "journal", steps):
        for idx in range(steps + 1):
            k = max(0, idx - 2)                             # rows 0..idx-3 published, read at idx
            k = min(k, r2.T_SUM)
            rows = dr[:k]
            pooled = sc.weights(sc.stats_inc(r2.MUS_F, rows.size, rows.sum()))
            worst = max(worst, float(np.max(np.abs(sc.weights(Y[idx]) - pooled))))
        last_k = k
        wrong += r2._ends_wrong(sc.weights(Y[-1]), 10)[0]
    print(f"\n(n2 journal) truth {truth}, {wname}: worst |holder weights - pooled closed form| = {worst:.2e} "
          f"over every step; rows read at the end {last_k} of {r2.T_SUM}; seeds ending wrong {wrong} of 20")
    assert worst <= 1e-12 and last_k == r2.T_SUM
    if truth == 5.0:
        assert wrong == 0


# ---------------------------------------------------------------------------
# Stage 2: (f)'s conclusions arms natively, (n3)-(n5)
# ---------------------------------------------------------------------------
#
#   (n3) Seed arm, truth in one holder's slice: every holder's top hypothesis is the truth at the final
#   step in 20 of 20 seeds (ring and all-to-all); arrival time grows with ring distance, roughly
#   log(1/eps) / (fitness gap) per hop. Truth in no slice: 0 of 20, settling on the KL-closest
#   hypothesis in the union of slices (round two: 7.5).
#   (n4) Evidence arm at the same settings: some holders end on a wrong hypothesis, count above zero and
#   above the seed arm's (round two: 11 and 12 of 20).
#   (n5) The seed and evidence forcings are each exact (the costs above are functions); no non-exact
#   forcing is needed anywhere in (f).
#
# Stage-2 numbers (§8): m = 8, n = 20, slices {2j, 2j+1}, eps = 0.05, alpha = 0.5, tau = 0.05,
# dt = 1/160, each draw held 160 steps, 150 draws, seeds 0-19. Analysis before code: run 1 (Shahshahani,
# as designed) is NaN in the seed arm; no finite forcing plants support under that response, so (n5) is
# predicted refuted; the evidence arm as specified cannot spread anything. Run 2 (disclosed, decided
# before code): Euclidean response, replicator field as the (non-exact) data forcing, exact seed.

EPS, ALPHA, TAU, DT, HOLD, T_N = 0.05, 0.5, 0.05, 1.0 / 160, 160, 150
TRUE_IDX = {5.0: 10, 8.5: 17}


@pytest.fixture(scope="module")
def native_conclusions():
    out = {}
    runs = [("shah", arm, w, 5.0) for arm in ("seed", "evidence") for w in r2.F_WIRINGS]
    runs += [("euc", arm, w, t) for arm in ("seed", "evidence", "both") for w in r2.F_WIRINGS
             for t in (5.0, 8.5)]
    for frame, arm, w, truth in runs:
        dr = np.stack([r2._f_draws(s, truth, T_N) for s in r2.F_SEEDS])
        out[(frame, arm, w, truth)] = nv.conclusions_native_run(
            dr, r2.F_WIRINGS[w], r2.MUS_F, r2.SLICES, DT, HOLD, arm, frame, EPS, ALPHA, TAU)
        tr, lo, dev = out[(frame, arm, w, truth)]
        last = tr[:, -1]
        print(f"\n(native conclusions) run {'1' if frame == 'shah' else '2'} {frame} {arm:8s} {w:10s} "
              f"truth {truth}: NaN in {int(np.isnan(tr).any(axis=(1, 2, 3)).sum())} of 20 seeds; "
              f"min weight {np.nanmin(lo):.2e}; max |sum p - 1| {np.nanmax(dev):.2e}")
    return out


def _wrong(last, right):
    """Per seed (any holder, all holders) above 0.99 on one hypothesis other than ``right``."""
    top = last.argmax(-1)
    bad = (last.max(-1) > 0.99) & (top != right)
    return bad.any(-1), bad.all(-1)


def test_n5_run1_seed_is_nan_where_it_is_needed(native_conclusions):
    """Prediction (i) before code: the seed's differential is infinite on the face, NaN after the sharp."""
    for w in r2.F_WIRINGS:
        tr, _, _ = native_conclusions[("shah", "seed", w, 5.0)]
        assert np.isnan(tr[:, 0]).all()
        tr, _, _ = native_conclusions[("shah", "evidence", w, 5.0)]
        assert not np.isnan(tr).any()


def _probe(frame, arm, seed=3):
    rng = np.random.default_rng(seed)
    n, d = 20, 2
    p0 = np.zeros(n); p0[[8, 9]] = 0.5
    h = nv.conclusion_holder(jnp.asarray(p0), jnp.full(d, 0.5), jnp.asarray(r2.MUS_F), DT, arm, frame,
                             EPS, ALPHA, int(1 / TAU))
    c = rng.dirichlet(np.ones(n), size=d)
    n_in = jnp.asarray(np.concatenate([c.ravel(), [5.3]]))
    return h, n_in, c


def test_n5_findings_which_pieces_are_exact():
    """Run 2's seed is the differential of eps(<c̄,p> - |p|^2/2); the Shahshahani evidence forcing is the
    differential of <f + alpha c̄, p>; the replicator field is NOT a differential (asymmetric Jacobian);
    and the Shahshahani sharp of any finite covector has zero component on a face."""
    rng = np.random.default_rng(0)
    p = jnp.asarray(rng.dirichlet(np.ones(20)))
    h, n_in, c = _probe("euc", "seed")
    hd, _, _ = _probe("euc", "evidence")
    cbar = 0.5 * (c[0] + c[1])
    ll = -0.5 * (5.3 - r2.MUS_F) ** 2
    f = ll - ll.min() + 1.0
    xi_seed = h.omega(p, jnp.zeros(0), n_in)[0] - p * (f - jnp.dot(f, p))
    g_seed = jax.grad(lambda q: EPS * (jnp.dot(cbar, q) - 0.5 * jnp.dot(q, q)))(p)
    err_seed = float(jnp.max(jnp.abs(xi_seed - g_seed)))
    J = jax.jacfwd(lambda q: hd.omega(q, jnp.zeros(0), n_in)[0])(p)
    asym = float(jnp.max(jnp.abs(J - J.T)))
    hs, _, _ = _probe("shah", "evidence")
    g_ev = jax.grad(lambda q: jnp.dot(f + ALPHA * cbar, q))(p)
    err_ev = float(jnp.max(jnp.abs(hs.omega(p, jnp.zeros(0), n_in)[0] - g_ev)))
    S = np.asarray(nv.shahshahani(20, DT).sharp_fn(jnp.asarray(np.where(np.arange(20) < 10, 0.1, 0.0))))
    face = float(np.max(np.abs(S[10:] @ rng.normal(size=(20, 50)))))
    print(f"\n(n5) euc seed vs d(eps(<c,p> - |p|^2/2)) {err_seed:.1e}; shah evidence vs d<f + alpha c, p> "
          f"{err_ev:.1e}; replicator field Jacobian asymmetry {asym:.2e}; Shahshahani sharp on a face, "
          f"largest normal component over 50 random covectors {face:.1e}")
    assert err_seed < 1e-15 and err_ev < 1e-12 and asym > 1e-3 and face == 0.0


@pytest.mark.xfail(strict=True, reason="refuted before code and confirmed (§5.7 stage 2): a seed that plants "
                   "support needs a non-exact data forcing (run 2) or an infinite one (run 1, NaN)")
def test_n5_no_non_exact_forcing_is_needed(native_conclusions):
    tr, _, _ = native_conclusions[("shah", "seed", "ring", 5.0)]
    assert not np.isnan(tr).any()


def _arrivals(tr, right):
    """Per seed and holder, first draw index (1-based) after which weight on ``right`` > 1/2, else inf."""
    above = tr[..., right] > 0.5
    first = np.where(above.any(1), above.argmax(1) + 1, np.inf)
    return first


def _ring_slope(first):
    m = first.shape[1]
    delays = {r: [] for r in range(m // 2 + 1)}
    for row in first:
        if not np.isfinite(row).any():
            continue
        f0 = int(np.argmin(row))
        for j in range(m):
            r = min((j - f0) % m, (f0 - j) % m)
            if np.isfinite(row[j]):
                delays[r].append(row[j] - row[f0])
    med = [(r, float(np.median(v))) for r, v in delays.items() if v]
    rs, ds = np.array(med).T
    return float(np.polyfit(rs, ds, 1)[0]), med


N3_RING = ("refuted (§5.7 stage 2): on the ring every holder's top is the truth in 19 of 20 seeds (seed 14: "
           "holders 0-2 still on 4.5 at the end), and the slope is 10.0 draws per hop, outside [12, 48]")


@pytest.mark.parametrize("wname", [pytest.param("ring", marks=pytest.mark.xfail(strict=True, reason=N3_RING)),
                                   "all-to-all"])
def test_n3_a_seed_spreads_the_truth(native_conclusions, wname):
    tr, _, _ = native_conclusions[("euc", "seed", wname, 5.0)]
    top_truth = (tr[:, -1].argmax(-1) == 10)
    all_top = int(top_truth.all(-1).sum())
    all_99 = int((tr[:, -1, :, 10] > 0.99).all(-1).sum())
    first = _arrivals(tr, 10)
    slope, med = _ring_slope(first)
    finders = np.argmin(first, axis=1)
    print(f"\n(n3) seed {wname}: every holder's top is the truth in {all_top} of 20 seeds "
          f"(holders not: {[(s, list(np.nonzero(~top_truth[s])[0]), list(tr[s, -1].argmax(-1)[~top_truth[s]])) for s in range(20) if not top_truth[s].all()]}); "
          f"every holder > 0.99 on it in {all_99}; first finder holder 5 in {int((finders == 5).sum())}; "
          f"median delay by distance {med}; slope {slope:.1f} draws per hop; "
          f"min final weight on the truth {tr[:, -1, :, 10].min():.3f}")
    assert all_top == 20
    if wname == "ring":
        assert 12 <= slope <= 48


@pytest.mark.parametrize("wname", list(r2.F_WIRINGS))
def test_n3_truth_in_no_slice_is_never_found(native_conclusions, wname):
    tr, _, _ = native_conclusions[("euc", "seed", wname, 8.5)]
    top = tr[:, -1].argmax(-1)
    none_truth = int((~(top == 17).any(-1)).sum())
    all_75 = int((top == 15).all(-1).sum())
    print(f"\n(n3, no slice) seed {wname}: no holder's top is the truth in {none_truth} of 20; every holder's "
          f"top is 7.5 in {all_75} of 20; max weight on the truth anywhere {np.nanmax(tr[..., 17]):.1e}")
    assert none_truth == 20 and all_75 == 20


@pytest.mark.parametrize("wname", list(r2.F_WIRINGS))
def test_n4_evidence_ends_wrong_more_than_seed(native_conclusions, wname):
    counts = {}
    for key in (("euc", "seed"), ("euc", "evidence"), ("euc", "both"), ("shah", "evidence")):
        tr, _, _ = native_conclusions[key + (wname, 5.0)]
        a, b = _wrong(tr[:, -1], 10)
        top = int((tr[:, -1].argmax(-1) == 10).all(-1).sum())
        counts[key] = (int(a.sum()), int(b.sum()), top)
    print(f"\n(n4) {wname}, truth 5.0, seeds ending on a wrong hypothesis any/all, and every holder's top the "
          f"truth: " + "; ".join(f"{k[0]} {k[1]} {v[0]}/{v[1]}, top {v[2]}" for k, v in counts.items()))
    ev, sd = counts[("euc", "evidence")][0], counts[("euc", "seed")][0]
    assert ev > 0 and ev > sd


# ---------------------------------------------------------------------------
# Test (g): explaining away by building excuses from figures (§9)
# ---------------------------------------------------------------------------
#
#   (g1) While some unexcluded figure has a drawing cost below the cost of moving theta by the needed
#   amount, an anomaly is absorbed by attaching it, and theta moves by less than a stated small bound.
#   (g2) Each ruled-out figure forces the next excuse to come from further away: the per-incident
#   explaining cost rises along the sequence of anomalies.
#   (g3) Once no unexcluded figure is cheaper than the update, theta moves; prediction to test: the move
#   is abrupt, a jump at one incident larger than the sum of all earlier moves.
#
# Numbers fixed before code (§9): K = 8, delta_k = 0.1 k, D = 1, theta_0 = 1, w = 4, v = 100, T = 10,
# incident i rules out figures 1..i-1; eta = 0.05, dt = 0.05, a0 = 0.01, 4000 steps per incident;
# convex (kappa = 0) and double well (kappa = 0.5). Called before running: convex gradual, double well
# abrupt. Scratch theta: convex 0.975, 0.925, 0.85, 0.75, 0.625, 0.504, ...; double well 1, 1, 1, 0.876,...

G_K, G_T, G_D, G_TH0, G_W, G_V = 8, 10, 1.0, 1.0, 4.0, 100.0
G_DELTA = 0.1 * np.arange(1, G_K + 1)
G_SCRATCH = {0.0: [0.975, 0.925, 0.85, 0.75, 0.625, 0.5038, 0.4316, 0.3807, 0.3425, 0.3113],
             0.5: [1.0, 1.0, 1.0, 0.8756, 0.7783, 0.7003, 0.5455, 0.5, 0.4615, 0.4286]}


def _g_run(kappa):
    ruled = np.zeros((G_T, G_K))
    for i in range(G_T):
        ruled[i, :min(i, G_K)] = 1
    th, A = nv.excuses_native_run(G_T, G_K, jnp.asarray(G_DELTA), jnp.full(G_T, G_TH0 - G_D),
                                  jnp.asarray(ruled), 4000, G_D, G_TH0, G_W, G_V, kappa, 0.05, 0.05, 0.01)
    th, A = np.asarray(th), np.asarray(A)
    own = np.array([A[i, i] for i in range(G_T)])                      # incident i's attachments at its end
    expl = np.array([own[i] @ G_DELTA + 0.5 * G_V * np.sum(ruled[i] * own[i] ** 2) for i in range(G_T)])
    moves = -np.diff(np.concatenate([[G_TH0], th]))
    excused = own.max(1) >= 0.5
    return th, A, own, expl, moves, excused


@pytest.fixture(scope="module")
def excuses():
    out = {}
    for kappa in (0.0, 0.5):
        th, A, own, expl, moves, excused = out[kappa] = _g_run(kappa)
        print(f"\n(g) kappa {kappa}: theta {np.round(th, 4).tolist()}; moves {np.round(moves, 4).tolist()}; "
              f"incident's own largest attachment {np.round(own.max(1), 3).tolist()}; explaining cost "
              f"{np.round(expl, 3).tolist()}; max |theta - scratch| "
              f"{np.max(np.abs(th - np.array(G_SCRATCH[kappa]))):.1e}; final attachments of incidents 1-3 "
              f"{np.round(A[-1, :3].max(1), 3).tolist()}; min/max attachment {A.min():.1e}/{A.max():.6f}")
    return out


@pytest.mark.parametrize("kappa", [0.0, 0.5])
def test_g1_an_excuse_absorbs_the_anomaly(excuses, kappa):
    th, A, own, expl, moves, excused = excuses[kappa]
    idx = np.nonzero(excused)[0]
    bound = G_DELTA[np.minimum(idx, G_K - 1)] / (G_D * G_W) + 1e-3
    assert np.all(moves[idx] <= bound) and np.all(moves[idx] < G_D / (G_W + 1))


@pytest.mark.parametrize("kappa", [0.0, 0.5])
def test_g2_each_ruling_out_makes_the_next_excuse_dearer(excuses, kappa):
    th, A, own, expl, moves, excused = excuses[kappa]
    e = expl[excused]
    assert len(e) >= 2 and np.all(np.diff(e) > 0)


G3_CONVEX = ("refuted, as called before code (§9): the cost is convex, so theta's move grows with the excuse's "
             "distance and the threshold move (0.1) is less than the sum of the earlier ones (0.15)")


@pytest.mark.parametrize("kappa", [pytest.param(0.0, marks=pytest.mark.xfail(strict=True, reason=G3_CONVEX)), 0.5])
def test_g3_the_update_is_abrupt(excuses, kappa):
    th, A, own, expl, moves, excused = excuses[kappa]
    first = int(np.argmin(excused))
    print(f"\n(g3) kappa {kappa}: threshold incident {first + 1}, move {moves[first]:.4f} against the sum of "
          f"earlier moves {moves[:first].sum():.4f}")
    assert moves[first] > moves[:first].sum()


def _g_scratch(kappa, S=4000, eta=0.05, dt=0.05, a0=0.01):
    """The numpy scratch of §9 (no arrangement): the same steps, written out."""
    y = np.full(G_T, G_TH0 - G_D)
    R = np.zeros((G_T, G_K))
    for i in range(G_T):
        R[i, :min(i, G_K)] = 1
    th, A, out = G_TH0, np.full((G_T, G_K), a0), []
    for t in range(G_T):
        seen = (np.arange(G_T) <= t).astype(float)
        for _ in range(S):
            r = y - th + G_D * A.sum(1)
            gA = seen[:, None] * (G_D * r[:, None] + G_V * R * A + G_DELTA + kappa * (1 - 2 * A))
            gth = G_W * (th - G_TH0) - np.sum(seen * r)
            th, A = th - eta * gth, A - dt * A * (1 - A) * gA
        out.append(th)
    return np.array(out)


@pytest.mark.parametrize("kappa", [0.0, 0.5])
def test_g_native_reproduces_the_scratch_values(excuses, kappa):
    err = float(np.max(np.abs(excuses[kappa][0] - _g_scratch(kappa))))
    print(f"\n(g) kappa {kappa}: max |native theta - numpy scratch| = {err:.1e}")
    assert err < 1e-6
