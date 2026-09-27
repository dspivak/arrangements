"""Pain, stage (b): growth as a reserve held in the fund, against learning held in the parts.

The book plan's P22 (AGENTDOCPLAN.md §8.2 in the book repository); stage (a) is
``tests/test_pain.py``, whose whole, numbers and shocks this stage keeps. David, 2026-09-26, of
growth: "1 and 2 in some combination? i'm fine with creating them seperately, but i think it's
really a combination and getting that combination right is a lot of life." The objection sent to
him and standing: in this model which mix is right is set by rates the build chooses, so this
stage predicts which threats each kind answers, not which mix wins.

**Pain's steepness, from David** (asked before this stage whether pain pulls in everything at the
first twinge or only when it gets bad): "the level of pain is exactly how distracting (or
re-tracking) it is, how hard it is to ignore. that is what pain is: hard to ignore." So here the
share pain pulls in *is* its level: a part's learning share moves toward ``s0 (1 - h_3)``
(``drive="level"``), all of it pulled only at full pain. Stage (a) pulled all of it by pain 0.3
(``Lambda = 10``); its run found that this is what traps a young whole (``test_pain.py`` (a5)).

**The reserve.** The fund gains a third coordinate ``rho``. A share ``eps (1 - pi)`` of income
goes into the reserve instead of the balance (saving is a project of growth, so pain pulls it in
as it pulls in learning); pain draws the reserve into the balance at rate ``k pi rho``; the
reserve leaks at ``leak_rho rho``. Chosen: ``eps = 0.05``, ``k = 2``, ``leak_rho = 0.01``, a tenth
of the balance's leak, with the fund's mobility 0.05 on both. The leak is the one reason to keep a
store apart in this model: the wage barely depends on the balance above a few units, so with equal
leaks a reserve is the balance relabelled. The pain reading ``phi`` stays a reading of the
balance alone, unchanged from stage (a).

**The control** (``control=True``): the same reserve, set aside the same way, drawn at rate
``k rho sigmoid(-f / 0.25)``, only once the balance is gone. It separates acting on the probability
of coming apart from acting on the event.

**The wholes**, every one with David's drive: ``N`` neither kind of growth (learning frozen,
``eta = 0``, no reserve); ``L`` learning (``eta = 1/1500``), stage (a)'s growth; ``R`` the reserve,
learning frozen; ``Rc`` the control reserve, learning frozen; ``LR`` both. **From zero** means
the young equilibrium (``N``'s, reached with learning frozen) with skill 0 and an empty reserve.
**Grown** means 40,000 unshocked steps from zero. Come apart, skill in reach, drains and cuts are
stage (a)'s.

**The predictions, fixed in advance and written here before any construction code for this stage
existed. Each must be able to fail.** Numbers in brackets are the agent's estimates, not
measurements.

(b0) *David's drive, on stage (a)'s cuts.* ``L`` from its young equilibrium, cut ``r`` by ``c``,
40,000 steps: (i) the survival edge is 0.41 +- 0.01, as for ``Lambda = 3, 5, 10``; (ii) no cut it
survives from 0.24 up leaves it trapped: at every one, pain below 0.1 and skill in reach above
0.3 at the end. ``L`` grown: (iii) a band of cuts the pain-free grown whole survives and ``L``
does not is still there, at least one grid point [0.53-0.54, as for ``Lambda = 3``]: the cost of
pain to a grown whole comes from a loss outlasting the learning pain stops, which a gentler drive
slows and does not remove.

(b1) *Both kinds answer drains.* Stage (a)'s shocks (drains of 15 at ``q = 0.004``, the rate
(a3) set on the pain-free whole; 200 seeds, 20,000 steps), each whole from zero. Fractions coming
apart: (i) ``R`` at least 10 points below ``N``; (ii) ``L`` at least 10 points below ``N``
[``N`` about 0.55, ``L`` about 0.35, ``R`` about 0.25]. Learning answers a drain through the stock
its income builds: the balance at rest rises about 78 per unit of skill. So by kind of threat,
nothing among drains and cuts is answered by the reserve and not by learning.

(b2) *A stock does not answer a lasting loss; a flow does.* Each whole grown, then cut ``c`` on a
grid of 0.01, 40,000 steps. Survival edges: (i) ``R``'s within 0.02 of ``N``'s; (ii) ``L``'s at
least 0.08 above ``N``'s; (iii) the reserve buys time: at each cut beyond ``R``'s edge, up to 0.10
past it, ``R`` comes apart later than ``N``.

(b3) *What counts is drawing on the probability, not the event.* Under (b1)'s drains, ``Rc`` comes
apart at least 5 points more often than ``R``. Reason given in advance: by the time the balance is
gone the wage has stopped and the parts have begun to leave, and the control, drawing only as the
balance goes below zero, holds it near zero where the wage is weak (``g(0.5)`` about a third of
its most). Can fail: the control never draws at rest, so it holds a larger reserve when a drain
comes.

(b4) *Between drains, the reserve is fast and learning slow.* The largest single drain survived,
measured at ``t`` steps after starting from zero, ``t`` in {1000, 20000}: ``R``'s exceeds ``L``'s
at 1000 and ``L``'s exceeds ``R``'s at 20000 [the reserve fills with a time constant of about
1/(0.05 (0.01 + 2 pi)), under 1,000 steps; skill with 1/delta = 5,000]. Fails if one exceeds the
other at both. Which is ahead at a given ``t`` is set by the chosen rates (the objection), so this
predicts a crossing, not where it falls.

(b5) *The combination.* In this build the two kinds draw on different things, the reserve on
income and learning on effort, so they share no budget and the combination is both, not a split;
a mix that trades one for the other would need a shared budget, a choice this stage does not make.
(i) They do not interfere at rest: ``LR``'s grown skill is within 2% of ``L``'s. (ii) Drains:
``LR``'s fraction apart is at most the smaller of ``L``'s and ``R``'s plus 0.02. (iii) Cuts:
``LR``'s edge is at least ``L``'s. (iv) If (b0)(iii) holds, the band of cuts the pain-free grown
whole survives and ``LR`` does not is narrower than ``L``'s: the drawn reserve holds the balance up
while the loss sets in, so pain is lower and less of the learning is stopped. This is the one
place predicted where the combination answers what neither kind answers alone.

**What happened** (2026-09-26, recorded after the runs; the predictions above untouched). Found in
building: nothing changed the design; ``Whole`` had to learn where its fund's coordinates sit.

(b0) Held. From young, ``L`` survives every cut to 0.41 and comes apart from 0.42 (stage (a)'s
steeper drives from 0.41), and at every cut it survives learning resumes: pain 0.000, skill in
reach 0.98-1.01 at the end. No trap. Grown, the pain-free whole survives to 0.54 and ``L`` to
0.53: the band is the one cut 0.54, narrower than ``Lambda = 3``'s two and ``Lambda = 10``'s three.
(b1) **(i) Refuted, (ii) held.** Under (a3)'s drains ``N`` and ``R`` both come apart in all 200
runs; ``L`` in 53.5% (skill in reach at the end 0.470). *Checked after, not predicted*
(``test_b1_b3_after``): the reserve does answer a drain, in the moment and at lower rates. The
largest single drain survived from rest is 21.47 for ``N`` and 30.75 for ``R``; at drains of 15
one per 1,000 steps (100 seeds) 68% of ``N`` come apart and 45% of ``R``, at one per 500 98% and
97%. At (a3)'s rate, one per 250 steps, the drains come faster than a reserve refills, and every
whole that does not learn comes apart. Learning raises the income that refills, so it answers
drains at a rate a stock cannot.
(b2) Held. Grown edges: ``N`` 0.39, ``R`` 0.39, ``L`` 0.53. Past ``R``'s edge the reserve buys
about 100 steps at every cut (0.40: 1,598 against 1,482; 0.49: 380 against 296).
(b3) **Refuted, and the other way round.** At (a3)'s rate ``R`` and ``Rc`` both come apart in all
runs. Checked after: the control holds far more at rest, 37.77 against 11.58, and survives a single
drain of 53.57 against 30.75; at one drain per 1,000 steps 3% of ``Rc`` come apart against 45% of
``R``, at one per 500 33% against 97%. The reason, not foreseen: ``phi`` is never zero (it reads
0.012 at ``R``'s rest), and a draw in proportion to pain draws the reserve at rest, in this build
about two thirds of what is set aside. A reserve drawn by the level of pain is spent by the pain
that is always there; one drawn by the event is kept for it.
(b4) **Refuted.** Learning is ahead early too: at 1,000 steps ``L`` survives a single drain of
34.09 and ``R`` 26.94; at 20,000 steps 107.75 and 30.75. No crossing. A thousand steps of learning
(skill about 0.18) already raise the balance more than the reserve holds, since the balance rises
about 78 per unit of skill, and setting income aside lowers the balance the drain hits.
(b5) (i) Held: grown skill 1.0112 for ``LR`` against 1.0120 for ``L``. (ii) Held: under (a3)'s
drains 44.5% of ``LR`` come apart, against 53.5% of ``L`` and 100% of ``R``. (iii) Held: both
edges 0.53. (iv) **Refuted**: the band is still the one cut 0.54. The grown whole with pain comes
apart there tens of thousands of steps after the cut, as stopped learning decays; the reserve is
drawn in the first few hundred, so it is gone long before the loss does its work.

*Reasonable or weird* (the agent's judgment, not evidence). (b2) is reasonable at every scale:
savings bridge a pay cut, retraining answers it. (b3) is reasonable for a household that dips into
savings at every worry, and weird as biology, where a store is drawn on a threshold, not in
proportion to a standing reading; what makes it so is ``phi`` never being zero and the draw exactly
proportional. (b4) is weird in size: a unit of skill doubles income and raises the balance about
78, a stage (a) choice, so learning's lead is the rates', as the objection said.

**(b6), fixed after the runs above and before any code for it** (2026-09-27). David, asked whether a
good day has a little pain or none: "if nothing is wrong, there's a little anxiety to make you want
to learn and grow, but it's a good day. i'd say no pain in the distracting sense. just motivation to
learn." The build's pain read about 0.01 on a good day, which is what spent the reserve at rest
(b3). So pain is rebuilt as the part of the reading a whole cannot ignore: ``fund_layer(floor=15)``
reads ``max(0, phi(f) - phi(15)) / (1 - phi(15))``, zero at any balance of 15 or more. 15 is chosen
so that setting income aside does not itself read as pain: the reserve wholes rest near 16 before
any draw. Every whole below uses it; the results above stand as run with the old reading.

(i) At rest the pain-drawn reserve ``R`` holds within 5% of the control ``Rc`` [both about 39: with
no pain neither draws]. (ii) ``R`` survives a single drain from rest at least as large as ``Rc``'s.
(iii) Under drains of 15 at one per 1,000 and one per 500 steps (100 seeds, 20,000 steps, from
zero), ``R`` comes apart no more often than ``Rc`` plus 0.02: it opens the jar once the balance falls
below 15, before the till is empty. Can fail: ``R`` opens the jar for drains the balance would have
survived alone, and what it moves into the balance leaks ten times faster there, so less may be
left for the next drain. (iv) At one per 1,000 steps ``R`` comes apart at least 10 points less often
than ``N``.

*Not in this build, David's (2026-09-27)*: "the workers might start looking for a job (increasing
their chances of leaving with stochasticicy) when the jar is low because they see the possibility of
not getting wages growing." Here workers leave only when unpaid; (b6) cannot speak to it.

**What happened** (2026-09-27). At rest every whole reads pain 0.0000. (i) Held: ``R`` and ``Rc``
both hold 38.35 at balance 15.39. (ii) Held: ``R`` survives a single drain of 54.48, ``Rc`` 52.74,
``N`` 21.22. (iii) **Refuted, by the failure named in advance**: at one drain per 1,000 steps 40% of
``R`` come apart against 10% of ``Rc``; at one per 500, 99% against 42%. (iv) Held: 40% against
``N``'s 78%. *Checked after*: after one drain of 15 from rest ``R`` pours more than half its jar into
the balance within 100 steps (reserve 38.4 to 16.6, balance overshooting to 21.8), and 1,000 steps
later holds 40.1 in all against the control's 52.5; the control drew almost nothing, and its
balance recovered from 0.4 on its own. With no pain on a good day the reserve is no longer spent at
rest, but a draw in proportion to pain still opens the jar wide for a drain the balance would have
survived, and money moved into the balance leaks there ten times faster. That leak is the one
reason to keep a jar apart in this build, so drawing on the event beats drawing on the level for
any store that keeps better than the till; how much better is the rates'.
"""


import functools

import jax
import jax.numpy as jnp
import numpy as np

from tests import test_pain as a
from arrangements.pain import closed_step, source, whole, worker
from arrangements.rand import key_sampler, seed

APART, R, S0, ETA, DELTA = a.APART, a.R, a.S0, a.ETA, a.DELTA
PRE, GROW, POST = a.PRE, a.GROW, a.POST
Q3, S3 = 0.004, 15.0          # (a3)'s drains, the rate set there on the pain-free whole
RESERVE = dict(eps=0.05, k=2.0, leak=0.01)
KINDS = ("N", "L", "R", "Rc", "LR")
CUTS = np.round(np.arange(0.30, 0.80, 0.01), 2)


def build(kind, *, r=R, shocks=None, floor=None):
    eta = ETA if kind in ("L", "LR") else 0.0
    res = None
    if kind in ("R", "Rc", "LR"):
        res = dict(RESERVE, control=(kind == "Rc"))
    ws = [worker(alpha=1.0, b=1.0, beta=1.0, mu_a=0.1, mu_theta=0.2, eta=eta, delta=DELTA,
                 s0=S0, Lam=0.0, drive="level", label=f"w{i}") for i in range(4)]
    return whole(ws, fund=dict(f0=20.0, mu=0.05, wmax=1.0, K=1.0, c0=2.0, leak=0.1, k_pi=0.1,
                               f_half=10.0, width=2.0, shocks=shocks, reserve=res, floor=floor),
                 r=r, src=source(1.0))


@functools.lru_cache(None)
def zero(kind, floor=None):
    """The young equilibrium, learning frozen, with skill 0 and (for a reserve) an empty one."""
    w = build("N", floor=floor)
    (c, fund), mk = jax.jit(lambda s: a.loop(closed_step(w.arr), s, PRE))(w.arr.P.init)
    if kind in ("R", "Rc", "LR"):
        fund = jnp.concatenate([fund, jnp.zeros(1)])
    return ((c, fund), mk)


def run(kind, s, n, floor=None):
    w = build(kind, floor=floor)
    return jax.jit(lambda x: a.loop(closed_step(w.arr), x, n))(s)


@functools.lru_cache(None)
def grown(kind, floor=None):
    return run(kind, zero(kind, floor), GROW, floor)


def read(kind, s, floor=None):
    w = build(kind, floor=floor)
    return w.read(w.arr.P.position(s))


def shocked(kind, q, S, s0, runs, steps, seed0=0, floor=None):
    """``runs`` seeds from ``s0`` under drains ``S`` at rate ``q``; fraction apart, final reads."""
    W = build(kind, shocks=(q, S), floor=floor)
    w = seed(W.arr, key_sampler(0))
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
    return np.asarray(apart).mean(), [W.read(qe) for qe in np.asarray(q_end)]


def cut(kind, s0, cs, steps=40000):
    """For each cut ``c``: came apart, first step apart, final pain, final skill in reach."""
    w = build(kind)

    def one(c):
        arr = w.closed(R * (1.0 - c))
        x, m, t = a.watch(w, closed_step(arr), s0, steps)
        d = w.read(arr.P.position(x))
        return m > APART, t, d["pi"], jnp.mean(d["kappa"] * d["sigma"])

    return [np.asarray(v) for v in jax.jit(jax.vmap(one))(jnp.asarray(cs))]


def edge(apart, cs):
    """The last cut survived before the first one that is not."""
    i = int(np.argmax(apart)) if apart.any() else len(cs)
    return cs[i - 1] if i > 0 else None


def threshold(kind, s, floor=None):
    """The largest single drain ``s`` survives (``POST`` steps to see)."""
    w = build(kind, floor=floor)
    step = closed_step(w.arr)
    go = jax.jit(lambda S: a.watch(w, step, a.drain(s, S), POST)[1] > APART)
    return a.bisect(lambda S: bool(go(S)), 1.0, 200.0)


# ---------------------------------------------------------------------------
# (b0): David's drive on stage (a)'s cuts.
# ---------------------------------------------------------------------------


@functools.lru_cache(None)
def grown_cuts(kind):
    return cut(kind, grown(kind), CUTS)


@functools.lru_cache(None)
def painfree_grown_cuts():
    return a.cut(0.0, a.grown(0.0), CUTS)


def test_b0_davids_drive_on_stage_a_cuts():
    cs = np.round(np.arange(0.15, 0.60, 0.01), 2)
    ap, t, pi, sk = cut("L", zero("L"), cs)
    print("\n  L from young, David's drive")
    a.show(cs, (ap, t, pi, sk))
    e = edge(ap, cs)
    surv = (cs >= 0.24) & ~ap
    resumed = surv & (pi < 0.1) & (sk > 0.3)
    print(f"  edge {e}; survived from 0.24 at {cs[surv]}; learning resumed at {cs[resumed]}")
    free, L = painfree_grown_cuts()[0], grown_cuts("L")[0]
    band = (~free) & L
    print(f"  grown: pain-free edge {edge(free, CUTS)}, L edge {edge(L, CUTS)}, band {CUTS[band]}")
    assert abs(e - 0.41) <= 0.011                          # (b0)(i)
    assert np.array_equal(resumed, surv)                   # (b0)(ii)
    assert band.sum() >= 1                                 # (b0)(iii)


# ---------------------------------------------------------------------------
# (b1), (b3), (b5)(ii): stage (a)'s drains, each whole from zero.
# ---------------------------------------------------------------------------


@functools.lru_cache(None)
def drained(kind):
    fa, reads = shocked(kind, Q3, S3, zero(kind), 200, 20000)
    return fa, float(np.mean([a.in_reach(d) for d in reads]))


def test_b1_both_kinds_answer_drains():
    out = {k: drained(k) for k in ("N", "L", "R")}
    for k, (fa, sk) in out.items():
        print(f"\n  {k:2}: apart {fa:.3f}, skill in reach {sk:.4f}", end="")
    print()
    assert out["R"][0] == out["N"][0] == 1.0                # (b1)(i), refuted: recorded as run
    assert out["L"][0] <= out["N"][0] - 0.10                # (b1)(ii)


def test_b3_drawing_on_the_probability_not_the_event():
    fR, fc = drained("R")[0], drained("Rc")[0]
    print(f"\n  R apart {fR:.3f}, Rc apart {fc:.3f}")
    assert fc == fR == 1.0                                  # (b3), refuted: recorded as run


# ---------------------------------------------------------------------------
# (b2), (b5)(iii)-(iv): permanent cuts from each grown whole.
# ---------------------------------------------------------------------------


def test_b2_a_stock_does_not_answer_a_lasting_loss():
    res = {k: grown_cuts(k) for k in ("N", "L", "R")}
    e = {k: edge(res[k][0], CUTS) for k in res}
    print(f"\n  edges: {e}")
    eR = e["R"]
    beyond = (CUTS > eR + 1e-9) & (CUTS <= eR + 0.10 + 1e-9)
    tN, tR = res["N"][1][beyond], res["R"][1][beyond]
    for c, x, y in zip(CUTS[beyond], tN, tR):
        print(f"  c {c:.2f}: N apart at {int(x)}, R at {int(y)}")
    assert abs(e["R"] - e["N"]) <= 0.02 + 1e-9               # (b2)(i)
    assert e["L"] >= e["N"] + 0.08 - 1e-9                    # (b2)(ii)
    assert np.all(tR > tN)                                   # (b2)(iii)


# ---------------------------------------------------------------------------
# (b4): the largest single drain survived, t steps into growth.
# ---------------------------------------------------------------------------


def test_b4_the_reserve_fast_and_learning_slow():
    th = {(k, t): threshold(k, run(k, zero(k), t)) for k in ("L", "R") for t in (1000, 20000)}
    for (k, t), v in th.items():
        print(f"\n  {k} at {t:5d}: largest single drain survived {v:.2f}", end="")
    print()
    assert th[("L", 1000)] > th[("R", 1000)]                 # (b4) early, refuted: recorded as run
    assert th[("L", 20000)] > th[("R", 20000)]               # (b4) late


# ---------------------------------------------------------------------------
# (b5): the combination.
# ---------------------------------------------------------------------------


def test_b5_the_combination():
    sL = float(np.mean(read("L", grown("L"))["sigma"]))
    sLR = float(np.mean(read("LR", grown("LR"))["sigma"]))
    d = {k: drained(k)[0] for k in ("L", "R", "LR")}
    free = painfree_grown_cuts()[0]
    res = {k: grown_cuts(k)[0] for k in ("L", "LR")}
    e = {k: edge(v, CUTS) for k, v in res.items()}
    band = {k: CUTS[(~free) & v] for k, v in res.items()}
    print(f"\n  grown skill L {sL:.4f}, LR {sLR:.4f}; drains apart {d}; edges {e}; bands {band}")
    assert abs(sLR / sL - 1) <= 0.02                          # (b5)(i)
    assert d["LR"] <= min(d["L"], d["R"]) + 0.02             # (b5)(ii)
    assert e["LR"] >= e["L"] - 1e-9                          # (b5)(iii)
    assert list(band["LR"]) == list(band["L"]) == [0.54]      # (b5)(iv), refuted: recorded as run


def test_b1_b3_after_the_reserve_answers_slower_drains_and_the_control_more():
    """Checked after (b1) and (b3), not predicted: single drains from rest, and drains of 15 at
    one per 1,000 and one per 500 steps (100 seeds)."""
    th = {k: threshold(k, grown(k)) for k in ("N", "R", "Rc")}
    fr = {(k, q): shocked(k, q, S3, zero(k), 100, 20000)[0]
          for k in ("N", "R", "Rc") for q in (0.001, 0.002)}
    rest = {k: read(k, grown(k)) for k in ("R", "Rc")}
    print(f"\n  single drain survived from rest: {th}\n  apart: {fr}"
          f"\n  reserve at rest: R {float(rest['R']['rho']):.2f}, Rc {float(rest['Rc']['rho']):.2f}")
    assert th["N"] < th["R"] < th["Rc"]
    assert fr[("Rc", 0.001)] < fr[("R", 0.001)] < fr[("N", 0.001)]
    assert fr[("Rc", 0.002)] < fr[("R", 0.002)] <= fr[("N", 0.002)]


# ---------------------------------------------------------------------------
# (b6): pain zero on a good day.
# ---------------------------------------------------------------------------

FLOOR = 15.0


def test_b6_with_no_pain_on_a_good_day():
    rest = {k: read(k, grown(k, FLOOR), FLOOR) for k in ("N", "R", "Rc")}
    th = {k: threshold(k, grown(k, FLOOR), FLOOR) for k in ("N", "R", "Rc")}
    fr = {(k, q): shocked(k, q, S3, zero(k, FLOOR), 100, 20000, floor=FLOOR)[0]
          for k in ("N", "R", "Rc") for q in (0.001, 0.002)}
    for k, d in rest.items():
        print(f"\n  {k:2} rest: f {float(d['f']):.2f}, pain {float(d['pi']):.4f}, "
              f"reserve {float(d.get('rho', 0.0)):.2f}; single drain survived {th[k]:.2f}; "
              f"apart at 1/1000 {fr[(k, 0.001)]:.2f}, at 1/500 {fr[(k, 0.002)]:.2f}", end="")
    print()
    rR, rc = float(rest["R"]["rho"]), float(rest["Rc"]["rho"])
    assert abs(rR / rc - 1) <= 0.05                          # (b6)(i)
    assert th["R"] >= th["Rc"]                               # (b6)(ii)
    for q in (0.001, 0.002):
        assert fr[("R", q)] >= fr[("Rc", q)] + 0.25          # (b6)(iii), refuted: recorded as run
    assert fr[("R", 0.001)] <= fr[("N", 0.001)] - 0.10       # (b6)(iv)
