"""Pain: a tracked probability of coming apart, and the resources it pulls in. Stage (a).

The book plan's P22 (AGENTDOCPLAN.md §8.2 in the book repository). David, 2026-09-26: "a pain
indicator which is a tracking of disintegration probability and a drive to change it and stop all
other projects", and, asked what it stops, "i just mean that pain pulls in all resources to stop
it, because otherwise they will be dispersed anyway … we're always working on growing, learning,
getting more influence to protect against shocks. but when we're in a shock, we pull in resources.
i wouldn't say its workers leave … they don't stop working, they just try to apply their knowledge
to fix the situation so they can go back to what they were doing." Growth, he said, is learning
and a reserve "in some combination"; this stage builds learning, held in the parts. The reserve
and the combination are stage (b).

**The whole.** The cascade's one level (``tests/test_cascade.py``): four workers, a source at
2.5 offering 1.0, a fund paying a gated wage through a one-way wire, a market; the same numbers
(``alpha = b = beta = 1``, amplitude and placement mobilities 0.1, ``tau = 0.8``; the fund's
mobility 0.05, ``wmax = K = 1``, overhead 2, leak 0.1). Three channels: supply, product, pain.
Three additions, in ``arrangements/pain.py``:

* **Growth.** A worker splits its effort ``a`` into a share ``s = sigmoid(theta)`` spent on
  learning and the rest on product. It sends the product ``(1 + sigma)(1 - s) a``; its skill
  ``sigma`` moves by the non-exact flow ``eta s a - delta sigma``, learning with forgetting
  (``eta = 1/1500``, ``delta = 1/5000``, so ``sigma`` settles near 1 at ``s = 0.3``: a grown
  worker produces twice what a young one does). Its allocation descends on
  ``(1/2)(theta - theta*)^2`` with mobility 0.2, ``theta* = logit(0.3) - Lambda h_3``. The market
  pays ``r = 2 / 0.7`` per unit of product, so a young whole (``sigma = 0``, ``s = 0.3``) sits
  at the cascade's balance, about 20.
* **Pain.** The fund carries a second coordinate ``pi`` moving by ``0.1 (phi(f) - pi)``,
  ``phi(f) = 1 / (1 + exp((f - 10) / 2))``: a lagged reading of its own balance, even odds at
  half a young whole's balance. ``phi`` is fixed here, before any run, from the whole's own
  structure; it is never fitted to the probability it is compared with (the trap of P20 (b)).
  The fund sends ``pi`` down on channel 3; a worker receives it weighted by the kernel, as it
  receives the wage, so a part out of reach does not feel the whole's pain.
* **The drive.** ``Lambda = 10``: at ``pi = 0.3`` a part's learning share is already below 2%,
  so pain above that pulls in all of it (David's "all"), and graded pain below it pulls in some.
  The control is ``Lambda = 0``: the tracker runs and nothing reads it. Pain is *read*: the part's
  cost sees ``h_3`` through ``stop_gradient``, so no push crosses back up the pain wire (between
  readers only the front crosses, `AGENTS.md`), as the wage wire is one-way in the cascade.

Shocks drain the balance: ``rand.kicked_configuration`` on the fund, ``-S`` with probability
``q`` a step, paired with a sampler; the same seed gives the same shock times with pain and
without. A whole **comes apart** when its workers' mean placement passes 1.25, halfway to the
source (recorded as ever, not at the end). Skill **in reach** is ``sum_j kappa_j sigma_j / 4``;
a whole that came apart holds almost none, by the kernel, not by a rule.

**The predictions, fixed in advance and written here before ``pain.py`` or any construction
code existed. Each must be able to fail.** Numbers in brackets are the agent's estimates from
the cascade's recorded behaviour, not measurements.

(a1) *Pain shortens the danger after a shock.* A young whole, no random shocks: drain 15 from
equilibrium; the window is the longest delay at which a second drain of 15 still brings it apart.
Pain shortens the window by at least a factor 2 [about 200 steps without, about 55 with]: the
learning share turned to product raises income by 1/0.7 while the balance is low. Fails if less.

(a2) *Pain cannot buy the moment.* The largest single drain a young whole survives rises with pain
by less than 10% [about 6%]. Once the wage stops, unpaid workers' amplitudes fall within about
fifteen steps; a 43% rise in product per unit of effort, over those steps, is small. Fails at 10%
or more. (This is the agents' objection that pulling in saves only against what the pulled-in
resources can fix in time.)

(a3) *Over a run of shocks, a whole with pain keeps more* (David's "otherwise they will be
dispersed anyway"). Drains of 15 at rate ``q``, 200 seeds, 20,000 steps. ``q`` is set on the
pain-free whole alone, before any run with pain: from ``q in {1, 2, 4, 8, 16} x 10^-3``, the one
whose pain-free fraction coming apart is nearest 50%. At that ``q``, with pain: (i) the fraction
coming apart is lower by at least 10 points; (ii) the mean skill in reach at the end is higher.
Fails if pain's forgone learning outweighs the collapses it prevents.

(a4) *Pain's price in growth is small here.* Drains of 9 at ``q = 10^-3`` (a drain leaves a young
whole near 11, where ``phi`` reads 0.4; two cannot bring it apart); if more than 5% of pain-free
runs come apart, ``S`` is lowered by 1 until not, on the pain-free whole alone. With pain, the mean
skill in reach at 10,000 steps is within 5% of the pain-free whole's. The reason given in advance:
learning raises the balance (``f_eq`` rises by about 78 per unit of skill), so a growing whole
leaves the range where these drains hurt within about 500 steps. Fails if pain costs 5% or more.

(a5) *A permanent loss, young whole: pain saves it, and can trap it.* Cut ``r`` by the fraction
``c`` at the young equilibrium; run 40,000 steps. Without pain the whole comes apart for ``c``
above ``c_0`` [0.25]; with pain, only above ``c_1`` [0.47], and ``c_1 - c_0 >= 0.1``. Inside that
band two outcomes, each over at least 0.03 of ``c``: for the smaller cuts pain fades and learning
resumes (``pi < 0.1`` and skill in reach above 0.3 at the end); for the larger ones the whole holds
together in lasting pain and does not learn (``pi > 0.5``, skill in reach below 0.05), although a
little learning would have lifted it out: the pain that stops the learning is what keeps it in
pain. Fails if either outcome is missing or the band is narrower.

(a6) *A permanent loss, grown whole: pain can bring apart what it would have survived.* The same
cut from the grown equilibrium (skill settled). There is a band of ``c`` at least 0.05 wide
[about 0.53 to 0.63] where the pain-free whole holds together for 40,000 steps and the whole with
pain comes apart, more than 2,000 steps after the cut: pain stops the learning, the skill decays,
and the income it paid for goes with it. Fails if no such band exists, or if the pain whole comes
apart within 2,000 steps (then it is the moment, not the learning, that kills).

(a7) *Pain read off the balance misreads a grown whole.* Young and grown wholes, each brought to
balance 5 from equilibrium; the true probability of coming apart within 1,000 steps on the
current course (``Lambda = 0``), under (a3)'s shocks, by 400 seeds each. ``phi`` reads the two the
same (it sees only the balance). The grown whole's true probability is at most half the young
whole's [nearer a tenth]: at the same balance it recovers faster, since its income is twice as
large. Fails if the ratio exceeds one half.

**What happened** (2026-09-26, recorded after the runs; the predictions above untouched). Found
in building: nothing changed the design; ``logit`` had to be taken in Python and the whole closed
at another market rate outside the collective's construction, both mechanical.

(a1) Held: the window for a second drain of 15 is 205 steps without pain and 36 with (x5.7).
(a2) **Refuted**: pain raised the single-drain threshold from 20.25 to 22.91 (x1.131), not under
10%. Part of it the agent did not foresee: ``phi`` reads 0.004 at rest, so a whole with pain rests
at balance 20.85 against 19.74, its parts leaning slightly to product. Measured from each whole's
own balance, pain survives a drain to -2.06 where the pain-free whole survives -0.51.
(a3) Held: ``q = 0.004`` (pain-free fractions apart 0.07, 0.29, 0.68, 0.99, 1.00 over the grid);
with pain 38% come apart against 68%, and the mean skill in reach at the end is 0.619 against 0.330.
(a4) Held: at ``S = 9`` (no lowering needed) none of either comes apart; skill in reach 0.8683
with pain against 0.8690 without.
(a5) Held, narrowly: ``c_0 = 0.24``, ``c_1 = 0.41``. Learning resumes after cuts 0.24 to 0.37;
cuts 0.38 to 0.40, exactly the three grid points the prediction required, leave the whole
together in lasting pain (pain 0.52, 0.74, 0.89, skill in reach under 0.01), a fixed point
unchanged from 20,000 to 80,000 steps. The clause "a little learning would have lifted it out"
was not asserted in advance; checked after (``test_a5_after``): with ``Lambda = 3`` or 5 the same
cuts survive *and* learning resumes, and the survival edge is 0.41 for ``Lambda = 3, 5, 10``
alike. So pulling in all buys no survival that pulling in most does not; what traps is pulling in
all at moderate pain (``Lambda = 10`` stops learning by ``pi = 0.3``), not pulling in all at full pain.
(a6) Held in kind, **refuted in width**: the pain-free grown whole survives cuts up to 0.54; the
whole with pain comes apart from 0.52, at 31,405, 22,977 and 17,943 steps after the cut (the
learning's decay, not the moment). The band is 0.52 to 0.54, three grid points, not five. Beyond
it both come apart and pain buys time (at 0.60: 6,493 steps against 1,104). Checked after
(``test_a6_after``): less steep pain (``Lambda = 3``) narrows the band to 0.53-0.54; it does not
remove it.
(a7) Held: from balance 5, 80.0% of young wholes come apart within 1,000 steps against 5.5% of
grown ones (ratio 0.069); ``phi(5) = 0.92`` for both, near the young whole's truth and seventeen
times the grown whole's.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.pain import closed_step, source, whole, worker
from arrangements.rand import key_sampler, seed

S0, ETA, DELTA = 0.3, 1 / 1500, 1 / 5000
R = 2 / 0.7
LAM = 10.0
APART = 1.25          # a whole has come apart when its workers' mean placement passes this
PRE = 4000            # steps to the young equilibrium, learning frozen
GROW = 40000          # steps to the grown equilibrium
POST = 6000           # steps after a drain to see what happens


def build(Lam, *, eta=ETA, r=R, shocks=None):
    ws = [worker(alpha=1.0, b=1.0, beta=1.0, mu_a=0.1, mu_theta=0.2, eta=eta, delta=DELTA,
                 s0=S0, Lam=Lam, label=f"w{i}") for i in range(4)]
    return whole(ws, fund=dict(f0=20.0, mu=0.05, wmax=1.0, K=1.0, c0=2.0, leak=0.1, k_pi=0.1,
                               f_half=10.0, width=2.0, shocks=shocks), r=r, src=source(1.0))


def loop(step, s, n):
    return jax.lax.fori_loop(0, n, lambda i, x: step(x), s)


def young(Lam):
    w = build(Lam, eta=0.0)
    return jax.jit(lambda s: loop(closed_step(w.arr), s, PRE))(w.arr.P.init)


def grown(Lam):
    w = build(Lam)
    return jax.jit(lambda s: loop(closed_step(w.arr), s, GROW))(young(Lam))


def drain(s, S):
    """Drain ``S`` from the balance of a deterministic whole's state ``((C, fund), market)``."""
    (c, fund), mk = s
    return ((c, fund.at[0].add(-S)), mk)


def watch(w, step, s, n):
    """Run ``n`` steps; return the final state, the largest mean placement on the way, and the
    first step at which it passed ``APART`` (-1 if never)."""
    pos = lambda x: jnp.mean(w.read(w.arr.P.position(x))["placements"])

    def body(i, carry):
        x, m, t = carry
        x = step(x)
        p = pos(x)
        t = jnp.where((t < 0) & (p > APART), i + 1, t)
        return x, jnp.maximum(m, p), t

    return jax.lax.fori_loop(0, n, body, (s, pos(s), jnp.asarray(-1)))


def in_reach(d):
    return float(jnp.mean(d["kappa"] * d["sigma"]))


def bisect(apart, lo, hi, iters=14):
    """The boundary of a monotone predicate: ``apart(lo)`` is False, ``apart(hi)`` True."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if apart(mid) else (mid, hi)
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# (a1), (a2): one young whole, drains by hand.
# ---------------------------------------------------------------------------


def drains(Lam):
    """Jitted ``(S1, delay, S2) |-> came apart``: drain ``S1``, wait ``delay``, drain ``S2``."""
    w = build(Lam)
    step, s0 = closed_step(w.arr), young(Lam)

    @jax.jit
    def go(S1, delay, S2):
        s = jax.lax.fori_loop(0, delay, lambda i, x: step(x), drain(s0, S1))
        _, m, _ = watch(w, step, drain(s, S2), POST)
        return m > APART

    return go


def test_a1_a2_the_window_and_the_moment():
    rows = {}
    for Lam in (0.0, LAM):
        go = drains(Lam)
        th = bisect(lambda S: bool(go(S, 0, 0.0)), 5.0, 60.0)
        assert bool(go(15.0, 0, 15.0)) and not bool(go(15.0, 3000, 15.0))
        lo, hi = 0, 3000
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if bool(go(15.0, mid, 15.0)) else (lo, mid)
        rows[Lam] = (th, lo)
    (th0, W0), (th1, W1) = rows[0.0], rows[LAM]
    print(f"\n  single-drain threshold: without pain {th0:.3f}, with {th1:.3f} (x{th1 / th0:.4f})"
          f"\n  window for a second drain of 15: without pain {W0} steps, with {W1} (x{W0 / max(W1, 1):.2f})")
    assert W0 >= 2 * W1                                   # (a1)
    assert th1 / th0 > 1.10                               # (a2), refuted: recorded as run


# ---------------------------------------------------------------------------
# (a3), (a4), (a7): random drains, many seeds.
# ---------------------------------------------------------------------------


def shocked(Lam, q, S, s0, runs, steps, seed0=0):
    """``runs`` seeds from state ``s0``; returns (came apart, first step apart, final reads)."""
    w = seed(build(Lam, shocks=(q, S)).arr, key_sampler(0))
    W = build(Lam, shocks=(q, S))
    step = closed_step(w)
    keys = jax.random.split(jax.random.PRNGKey(seed0), runs)
    pos = lambda x: jnp.mean(W.read(w.P.position(x))["placements"])

    def one(k):
        def body(i, carry):
            x, m = carry
            x = step(x)
            return x, jnp.maximum(m, pos(x))
        x, m = jax.lax.fori_loop(0, steps, body, ((s0, k), pos((s0, k))))
        return m > APART, w.P.position(x)

    apart, q_end = jax.jit(jax.vmap(one))(keys)
    reads = [W.read(qe) for qe in np.asarray(q_end)]
    return np.asarray(apart), reads


_Q3 = {}


def q_for_half():
    """(a3)'s ``q``, set on the pain-free whole alone."""
    if not _Q3:
        s0 = young(0.0)
        fr = {}
        for q in (1e-3, 2e-3, 4e-3, 8e-3, 16e-3):
            fr[q] = shocked(0.0, q, 15.0, s0, 200, 20000)[0].mean()
        _Q3["q"] = min(fr, key=lambda q: abs(fr[q] - 0.5))
        _Q3["table"] = fr
    return _Q3["q"]


def test_a3_over_a_run_of_shocks_a_whole_with_pain_keeps_more():
    q = q_for_half()
    out = {}
    for Lam in (0.0, LAM):
        apart, reads = shocked(Lam, q, 15.0, young(Lam), 200, 20000)
        out[Lam] = (apart.mean(), np.mean([in_reach(d) for d in reads]))
    print("\n  pain-free fraction apart by q: "
          + ", ".join(f"{k:g}: {v:.2f}" for k, v in _Q3["table"].items()) + f"  -> q = {q:g}")
    for Lam, (fa, sk) in out.items():
        print(f"  Lambda {Lam:4.1f}: apart {fa:.3f}, mean skill in reach at the end {sk:.4f}")
    assert out[0.0][0] - out[LAM][0] >= 0.10              # (a3) (i)
    assert out[LAM][1] > out[0.0][1]                      # (a3) (ii)


def test_a4_pains_price_in_growth():
    S, q = 9.0, 1e-3
    while shocked(0.0, q, S, young(0.0), 200, 10000)[0].mean() > 0.05:
        S -= 1.0
    out = {}
    for Lam in (0.0, LAM):
        apart, reads = shocked(Lam, q, S, young(Lam), 200, 10000)
        out[Lam] = (apart.mean(), np.mean([in_reach(d) for d in reads]))
    for Lam, (fa, sk) in out.items():
        print(f"\n  S {S:g}, Lambda {Lam:4.1f}: apart {fa:.3f}, mean skill in reach {sk:.4f}", end="")
    print()
    assert abs(out[LAM][1] / out[0.0][1] - 1) < 0.05       # (a4)


def test_a7_pain_read_off_the_balance_misreads_a_grown_whole():
    q = q_for_half()
    frac = {}
    for name, s in (("young", young(0.0)), ("grown", grown(0.0))):
        (c, fund), mk = s
        s5 = ((c, fund.at[0].set(5.0)), mk)
        frac[name] = shocked(0.0, q, 15.0, s5, 400, 1000, seed0=7)[0].mean()
    print(f"\n  true probability of coming apart within 1000 steps from balance 5: "
          f"young {frac['young']:.4f}, grown {frac['grown']:.4f}")
    assert frac["grown"] <= 0.5 * frac["young"]            # (a7)


# ---------------------------------------------------------------------------
# (a5), (a6): a permanent cut of the market.
# ---------------------------------------------------------------------------


def cut(Lam, s0, cs, steps=40000):
    """For each cut ``c``: came apart, first step apart, final pain, final skill in reach."""
    w = build(Lam)

    def one(c):
        arr = w.closed(R * (1.0 - c))
        x, m, t = watch(w, closed_step(arr), s0, steps)
        d = w.read(arr.P.position(x))
        return m > APART, t, d["pi"], jnp.mean(d["kappa"] * d["sigma"])

    return [np.asarray(v) for v in jax.jit(jax.vmap(one))(jnp.asarray(cs))]


def show(cs, res):
    for c, a, t, p, s in zip(cs, *res):
        print(f"  c {c:.2f}: apart {bool(a)!s:5} at {int(t):6d}, pain {float(p):.3f}, skill in reach {float(s):.3f}")


def test_a5_young_whole_permanent_cut():
    cs = np.round(np.arange(0.15, 0.60, 0.01), 2)
    res = {Lam: cut(Lam, young(Lam), cs) for Lam in (0.0, LAM)}
    for Lam in (0.0, LAM):
        print(f"\n  Lambda {Lam}")
        show(cs, res[Lam])
    a0, a1 = res[0.0][0], res[LAM][0]
    c0 = cs[np.argmax(a0)]                                 # first cut the pain-free whole does not survive
    c1 = cs[np.argmax(a1)]
    band = (cs >= c0) & (cs < c1)
    pi, sk = res[LAM][2], res[LAM][3]
    resumed = band & (pi < 0.1) & (sk > 0.3)
    trapped = band & (pi > 0.5) & (sk < 0.05)
    print(f"  c_0 {c0:.2f}, c_1 {c1:.2f}; learning resumed at {cs[resumed]}; trapped at {cs[trapped]}")
    assert c1 - c0 >= 0.1                                  # (a5) the band
    assert resumed.sum() >= 3 and trapped.sum() >= 3       # (a5) both outcomes


def test_a6_grown_whole_permanent_cut():
    cs = np.round(np.arange(0.40, 0.80, 0.01), 2)
    res = {Lam: cut(Lam, grown(Lam), cs) for Lam in (0.0, LAM)}
    for Lam in (0.0, LAM):
        print(f"\n  Lambda {Lam}")
        show(cs, res[Lam])
    killed = (~res[0.0][0]) & res[LAM][0]
    late = killed & (res[LAM][1] > 2000)
    print(f"  pain brings apart what the pain-free whole survives at {cs[killed]}; after 2000 steps at {cs[late]}")
    assert late.sum() >= 3 and np.array_equal(late, killed)  # (a6), in kind
    assert killed.sum() < 5                                # (a6), refuted in width: recorded as run


def test_a5_after_all_at_moderate_pain_is_the_trap():
    """Checked after (a5), not predicted: the trapped cuts under pain that pulls in less steeply.
    ``Lambda = 3`` survives the same cuts as ``Lambda = 10`` and resumes learning where it traps."""
    cs = np.array([0.38, 0.39, 0.40, 0.41])
    res = {Lam: cut(Lam, young(Lam), cs, steps=80000) for Lam in (3.0, LAM)}
    for Lam in (3.0, LAM):
        print(f"\n  Lambda {Lam}")
        show(cs, res[Lam])
    apart3, _, pi3, sk3 = res[3.0]
    apart10, _, pi10, sk10 = res[LAM]
    assert list(apart3) == list(apart10) == [False, False, False, True]
    assert np.all(pi10[:3] > 0.5) and np.all(sk10[:3] < 0.05)
    assert np.all(pi3[:3] < 0.1) and np.all(sk3[:3] > 0.9)


def test_a6_after_less_steep_pain_narrows_the_grown_cost_but_keeps_it():
    """Checked after (a6), not predicted: ``Lambda = 3`` on the grown whole."""
    cs = np.array([0.52, 0.53, 0.54])
    free = cut(0.0, grown(0.0), cs)[0]
    three = cut(3.0, grown(3.0), cs)[0]
    print(f"\n  cuts {cs}: pain-free apart {free}, Lambda 3 apart {three}")
    assert not free.any()
    assert list(three) == [False, True, True]
