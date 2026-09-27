"""An account restores the push, as data (the book plan's P20 (a), AGENTDOCPLAN.md §8.2).

David, 2026-09-26: "what drives us to account is the failure for our gradient to push." The
cascade (``tests/test_cascade.py``) found the quantity: when the wage wire from a fund to its
workers is one-way, the workers' push on the fund's balance, ``g'(f) sum_j kappa_j (alpha_j a_j +
beta_j)``, no longer arrives. This test gives each worker a third channel, a **report**, and lets
the fund read it.

**Two changes of construction from the cascade, said first.** (i) The cascade made its wire one-way
by subtracting the pullback, which needed the fund to know every worker's ``(alpha, beta)``: the
fund knew its workers' wants without being told. Here the wire is made one-way by
``jax.lax.stop_gradient`` on the balance in the fund's sending map, so composition (``jax.vjp`` in
``wiring.py``) carries nothing back along it and the fund knows nothing of its workers. (ii) The
workers are heterogeneous (their ``alpha_j, beta_j`` differ), so that a fund that did not read the
reports could not reconstruct the push from what it already receives.

**The predictions, fixed in advance and written here before ``account.py`` or any construction
code existed.**

(20) *The two one-way wires agree.* With identical workers, the stop-gradient wire and the
cascade's subtracting wire give the same trajectory to 1e-10. A check of (i), not a result.

(21) *A report carries the whole stranded push.* Each worker reports ``r_j = alpha_j a_j + beta_j``,
its want per unit of supply, computed from its own state and its own constants; the collective
carries ``sum_j kappa_j r_j`` onward on the report channel, as it carries every channel; the fund,
on the one-way wire, adds ``g'(f)`` times what arrives to its flow. Then the composite reproduces
the exact composite's trajectory, through equilibrium and a shock, to 1e-8. **Built to hold**: the
pullback through a wire is the sender's derivative applied to the receivers' covectors on their
inputs, and each receiver's covector is its own (the chain rule). It could fail only if the
kernel's aggregation of the reports differed from the kernel's aggregation of the pullback. So
this establishes that the construction is right, and that a report a worker can compute alone
suffices; it is not evidence about the world.

(22) *Acting on the report as a push creates money.* A fund whose balance moves by the report, as
the exact composite's does, gains ``mu g'(f) sum kappa r`` per step from nothing: need is not
income. Predicted: over equilibrium the created money is small against the balance (under 5% of
it over the run), and through a shock just above the one-way threshold it is what separates
survival from collapse (the exact fund survives, the one-way fund does not, and the difference in
the balance at the shock's low point is at least half the money created by then). Fails if the
exact fund's survival is not accounted for by the money created. So no fund that reproduces the
exact composite keeps money conserved; what a conserving fund may do with a report is a choice,
not settled here.

**What happened** (2026-09-26, recorded after the runs; the predictions above untouched).

(20) Held: the two one-way wires agree to 3.6e-15.
(21) Held, as built to: with four workers whose ``(alpha_j, beta_j)`` differ and are unknown to the
fund, the fund that reads the reports reproduces the exact composite to 3.6e-15 through a shock,
where the fund that does not is off by 0.62.
(22) Clause 1 held: at a drain 2% above the one-way threshold (20.22; the exact one is 21.77) the
exact fund recovers to 19.93 and the one-way fund falls to -18.39. Clause 2 held only per 1000
steps (0.885 created against a balance of 19.94, 4.4%); "over the run" was ambiguous, and over the
4000-step approach it is about 18%. Clause 3 was **ill-posed and fails as written**: the drain
takes the balance below 0 at once, so the low point is the instant of the drain, before anything is
created. Measured after the fact in its place: twenty steps on, the exact fund stands 1.355 above
the one-way fund and has created 1.268; and the one-way fund started from the exact fund's own
equilibrium, drained the same, still falls to -18.39. So what saves the exact fund is the money its
workers' need creates during the crisis (the gate's slope is largest near an empty fund), not its
higher balance going in.

**Stage (b): where a part settles, with a want that levels off at a need** (David, 2026-09-26,
after two scratch runs: "you can run whatever tests you think are appropriate"). No report
channel, no audit, no learner. A **needful worker** holds ``a`` on received supply ``h`` with cost
``(b/2) a^2 - alpha a h + (gamma/2)(nu - h)^2``: supply up to its need ``nu`` is wanted, and beyond
it the work that comes with supply costs more than it gives. Its placement moves by its own
descent, as every worker's does. The fund and the market are stage (a)'s, one-way wire, no reports
read; the fund sees a worker only through its weight, the kernel at its placement. The comparison
is the cascade's **linear** worker, ``(b/2) a^2 - (alpha a + beta) h``, which always wants more.

**The predictions, fixed in advance and written here before any stage (b) construction code.**

(23) *Where a part settles carries its need.* Four needful workers with needs 0.3, 0.6, 0.9, 1.2,
the fund's wage cap lowered in steps: at every cap where they do not all stand together, their
placements are ordered by need (neediest farthest out), and the neediest leaves first as the cap
falls. Four linear workers with standing wants 0.2 to 2 settle within 0.01 of each other at every
cap. Fails if the needful placements are not ordered by need, or the linear ones spread.

(24) *Departure is graded, not all or none, when wants level off.* Drain the fund from equilibrium
by growing amounts: the number of needful workers that end at the source rises through at least
two intermediate values (1, 2 or 3) as the drain grows, where the linear workers go from none to
all (the cascade's result). The agent's guess, unsure: each needful worker that leaves takes its
income with it, so the departures might cascade anyway. Fails if the needful workers also go
from none to all.

(25) *Does a collective whose wants level off hold through larger shocks?* (David's question,
"when wants are closer to needs, the workers do a better job of working together?".) The two
collectives are set to the same equilibrium balance within 5% by the linear workers' standing
want. Predicted, a guess: the needful collective's threshold (the drain past which its balance
ends below 0) is higher, because its least needy workers stay and keep some income. Fails if it
is lower or within 2%.

**What happened** (2026-09-26, recorded after the runs; the predictions above untouched; market
``r = 4``, overhead 1, ``alpha = 0.5``, chosen before the runs only for a solvent fund).

(23) **Refuted.** At every cap from 1.4 to 0.6 the farther a needful worker stands from the fund,
the *less* its need: where the wage exceeds a need, the worker backs away from the excess (cap 1.0:
placements -1.17, -0.51, 0.06, 0.06 for needs 0.3 to 1.2; cap 1.4: 1.43, 1.43, 0.51, 0.04). Workers
the wage underpays stand together at the fund, as the linear ones do. No one reaches the source at
any cap down to 0.45, where the least needy stands nearest (0.15 against 0.25). The linear workers
stand within 0.01 of each other at every cap (held). The two scratch runs that suggested the
prediction had a fund near 1, paying less than every need. So what the fund can read from where a
part stands is whom it overpays and by how much, not whom it underpays. (Later traced to an
error in the cost, need written as a target; the rerun below replaces this reading.)
(24) **Refuted as written.** The needful workers go from none to two at a drain of 15.5 (the two
neediest reach the source, the fund ends at -4.70, the other two stop partway at 1.29 and 1.90),
with no other count: not all or none, and not graded through two values. The linear workers are not
all or none either: at a drain of 32.5 the one with the largest standing want reaches the source
while the fund recovers (to 20.09), then all four go. A standing want sets how fast a worker moves,
so a passing shock catches the fastest.
(25) **Ill-posed as designed**: a linear worker's standing want does not move the fund's balance,
so it cannot match the two. Measured after the fact, matching instead by the linear fund's wage cap
(0.613; both balances 14.62): the needful collective's threshold is 15.46 (margin 0.84), the linear
one's 14.87 (margin 0.25), higher by 0.59, 4%. The guess holds for this matching and these
parameters only.

**Reasonable or weird, at the scales these might be read at** (the agent's judgment, 2026-09-26,
not evidence).
* Reasonable: a worker can state its own want from its own state (a tissue short of oxygen signals
  for supply); a fund learns of underpayment only when parts leave (an employer learns it at the
  resignation); the most wanting move first on a passing shock.
* Weird: *need creating money* (22) -- right for physical loading, where a downstream load does
  act back on a source, wrong for anything conserved, and it is largest when the fund is emptiest,
  an artifact of the gate's slope. *A balance at -18 with the fund still running*: ruin here is not
  an end, since the boundary is left out. *The overpaid backing away* (23): the quadratic makes
  supply beyond need exactly as costly as supply short of it; a person may cut hours when pay
  rises, and a cell downregulates a receptor when oversupplied, but excess is usually cheaper than
  shortage, and a worker walking off to the side away from both fund and source goes nowhere
  real. *No one leaving though underpaid*: the source's kernel at the fund is about 0.02, so an
  underpaid worker cannot feel the elsewhere; that is a world with no information about
  alternatives, which is a choice of geometry, not a finding.

**Stage (b), rerun with two fixes** (David, 2026-09-26, "yes" to rerunning after the weirdnesses
above were traced). (i) Need is a **floor**, not a target: a **floor worker** has cost
``(b/2) a^2 - alpha a h - lam log(sp(h - nu))``, ``sp(x) = eps softplus(x / eps)``, David's log with
the need subtracted (Stone-Geary's form, from memory): below its need it is desperate, above it
more is wanted less but never refused. (ii) The **going rate is felt**: the elsewhere, offering 1.0,
stands at 1.5 instead of 2.5, where its kernel at the fund is about 0.25 instead of 0.02.

**The predictions, fixed in advance and written here before any floor-worker construction code.**
The math first: a floor worker's want per unit of supply, ``alpha a + lam sp'/sp``, is positive
everywhere, and every worker in the collective faces the same supply over placements, so every
worker's placement settles at the same local maximum. Need can change how fast a worker moves, not
where it settles.

(26) *No one backs away.* At every cap tried, the four floor workers stand within 0.01 of each other.
Built to hold by the argument above; a check of fix (i).

(27) *They leave together, the neediest first.* As the cap falls, all four are at the fund or all
four are at the elsewhere at every cap (no cap splits them). Released from the fund's equilibrium
at a cap low enough that they leave, they reach the elsewhere in order of need, neediest first.
Fails if some cap splits them or the order is not by need.

(28) *A passing shock sorts them by need.* Drain the fund from equilibrium by growing amounts: the
number that end at the elsewhere takes at least two values strictly between none and all, and
whenever some but not all leave, those who leave are the neediest. Fails otherwise; the agent's
guess, unsure, since the linear workers of (24) took only one intermediate value.

**What happened** (2026-09-26, recorded after the runs; ``lam = 0.3``, chosen after a check that
equilibria are finite and the fund solvent, where ``lam = 0.1`` gave the same placements).

(26) Held: at every cap from 1.4 to 0.2 the four stand within 0.001 of each other. Fix (i) works.
(27) First clause **refuted**, by the geometry: with the elsewhere at 1.5 its supply and the fund's
overlap into one hill, so as the cap falls the four drift together from 0.42 to 1.43 (at the fund
at no cap, at the elsewhere from cap 0.45 down); there is no leaving, only drift. Second clause
held: released from the cap-1.0 equilibrium into cap 0.2 they pass 1.25 after 17, 11, 3 and 1
steps, neediest first.
(28) **Refuted**: no drain up to 80 (twice the balance) moves anyone for good; the fund recovers to
39.27 every time. The reason is a weirdness of the construction since the cascade: the fund is paid
for all its collective's product, including what workers make on the elsewhere's supply, so with
the elsewhere felt the elsewhere keeps a broke fund's income running.

**Reasonable or weird** (the agent's judgment). Reasonable: need as a floor, and the neediest
moving first. Reasonable-ish: the overlap, a worker between the two taking supply from both, as
someone holding two jobs. **Weird: a fund earning on product it did not pay for**, which also
drove the cascade's rise of balance with the number of workers (``tests/test_cascade.py`` (12));
not fixed here, since the product would have to be attributed to whoever supplied it.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.account import (
    EMPTY, floor_worker, gate, needful_worker, one_level, source, worker,
)
from arrangements.cascade import closed_step

TAU, MU_A, MU_P, MU_F = 0.8, 0.1, 0.1, 0.05
L_SRC, S_SRC, R_MKT = 2.5, 1.0, 2.0
PRE, POST = 4000, 6000
N = 4
HETERO = [(1.0, 1.0), (1.2, 0.5), (0.8, 1.5), (1.0, 0.8)]      # (alpha_j, beta_j)


def build(wire, read_reports=False, consts=HETERO, known=None):
    ws = [worker(0.8, alpha=al, b=1.0, beta=be, mu=MU_A, label=f"w{j}")
          for j, (al, be) in enumerate(consts)]
    return one_level(ws, positions=[0.06] * N, mobility=MU_P, tau=TAU,
                     fund=dict(f0=20.0, mu=MU_F, wmax=1.0, K=1.0, c0=2.0, leak=0.1, wire=wire,
                               read_reports=read_reports, known=known),
                     src=source(S_SRC), src_position=L_SRC, r=R_MKT)


def fund_of(s):
    return s[0][1]


def trajectory(arr, S=0.0, pre=PRE, post=POST):
    """Balances through ``pre`` steps to equilibrium, a drain of ``S``, and ``post`` more."""
    step = closed_step(arr, EMPTY)

    @jax.jit
    def go(S):
        def body(s, _):
            s = step(s)
            return s, jnp.concatenate([jnp.ravel(fund_of(s)), jnp.ravel(arr.P.position(s))])
        s1, xs1 = jax.lax.scan(body, arr.P.init, None, length=pre)
        s1 = ((s1[0][0], s1[0][1] - S), s1[1])
        _, xs2 = jax.lax.scan(body, s1, None, length=post)
        return jnp.concatenate([xs1, xs2])

    return np.asarray(go(S))


def test_20_the_two_one_way_wires_agree():
    same = [(1.0, 1.0)] * N
    a = trajectory(build("oneway", consts=same), S=10.0)
    b = trajectory(build("subtract", consts=same, known=(1.0, 1.0)), S=10.0)
    err = np.max(np.abs(a - b))
    print(f"\n  stop-gradient against subtracting wire, max difference {err:.2e}")
    assert err < 1e-10


def test_21_a_report_carries_the_whole_stranded_push():
    exact = trajectory(build("exact"), S=15.0)
    oneway = trajectory(build("oneway"), S=15.0)
    told = trajectory(build("oneway", read_reports=True), S=15.0)
    err, gap = np.max(np.abs(told - exact)), np.max(np.abs(oneway - exact))
    print(f"\n  reports read against exact {err:.2e};  one-way unread against exact {gap:.2e}")
    assert gap > 1e-2 and err < 1e-8


def threshold(arr):
    """The smallest drain after which the balance ends below 0, by bisection."""
    step = closed_step(arr, EMPTY)
    s_eq = jax.jit(lambda s: jax.lax.fori_loop(0, PRE, lambda i, x: step(x), s))(arr.P.init)
    after = jax.jit(lambda S: fund_of(jax.lax.fori_loop(
        0, POST, lambda i, x: step(x), ((s_eq[0][0], s_eq[0][1] - S), s_eq[1])))[0])
    lo, hi = 0.0, 60.0
    for _ in range(18):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if after(mid) > 0 else (lo, mid)
    return float(fund_of(s_eq)[0]), (lo + hi) / 2


def created(exact, oneway, S, pre=PRE, post=POST):
    """Along the exact composite's run: its balance, the one-way balance, and the money the
    exact fund gains per step beyond the one-way flow at the same state (``mu`` times the
    difference of the two forcings on the balance)."""
    step = closed_step(exact, EMPTY)

    def extra(s):
        q = exact.P.position(s)
        return MU_F * (exact.omega(q, EMPTY, EMPTY)[0][-1] - oneway.omega(q, EMPTY, EMPTY)[0][-1])

    @jax.jit
    def go(S):
        def body(s, _):
            return step(s), (fund_of(s)[0], extra(s))
        s1, (f1, m1) = jax.lax.scan(body, exact.P.init, None, length=pre)
        s1 = ((s1[0][0], s1[0][1] - S), s1[1])
        _, (f2, m2) = jax.lax.scan(body, s1, None, length=post)
        return jnp.concatenate([f1, f2]), jnp.concatenate([m1, m2])

    return [np.asarray(x) for x in go(S)]


def test_22_acting_on_the_report_as_a_push_creates_money():
    ex, ow = build("exact"), build("oneway")
    (fe, Se), (fo, So) = threshold(ex), threshold(ow)
    print(f"\n  exact: balance {fe:.3f}, threshold {Se:.3f};  one-way: balance {fo:.3f}, threshold {So:.3f}")
    S = 1.02 * So
    f_ex, m_ex = created(ex, ow, S)
    f_ow = trajectory(ow, S)[:, 0]
    after = slice(PRE, None)
    low = PRE + int(np.argmin(f_ex[after]))
    eq_created = float(np.sum(m_ex[PRE - 1000:PRE]))              # the last 1000 steps before
    by_low = float(np.sum(m_ex[PRE:low + 1]))
    diff_low = float(f_ex[low] - f_ow[low])
    print(f"  at equilibrium: created per 1000 steps {eq_created:.3f} against balance {fe:.3f}")
    print(f"  shock {S:.3f}: exact ends {f_ex[-1]:.2f}, one-way ends {f_ow[-1]:.2f}; at the exact low "
          f"(step {low - PRE}) balances {f_ex[low]:.3f} vs {f_ow[low]:.3f}, "
          f"difference {diff_low:.3f}, created since the shock {by_low:.3f}")
    assert f_ex[-1] > 0 > f_ow[-1]                              # held
    assert eq_created < 0.05 * fe                               # held per 1000 steps only
    assert low == PRE and by_low < 1e-5                         # clause 3 ill-posed: the low is the drain
    # Measured after the fact, in place of clause 3: twenty steps on, the created money is most
    # of the gap, and the one-way fund started from the exact fund's own equilibrium, with the
    # same drain, still collapses, so the higher start is not what saves it.
    k = PRE + 20
    gap20, made20 = float(f_ex[k] - f_ow[k - 1]), float(np.sum(m_ex[PRE:k]))
    step_o, step_e = closed_step(ow, EMPTY), closed_step(ex, EMPTY)
    s = jax.jit(lambda s: jax.lax.fori_loop(0, PRE, lambda i, x: step_e(x), s))(ex.P.init)
    s = ((s[0][0], s[0][1] - S), s[1])
    end = float(fund_of(jax.jit(lambda s: jax.lax.fori_loop(
        0, POST, lambda i, x: step_o(x), s))(s))[0])
    print(f"  20 steps on: gap {gap20:.3f}, created {made20:.3f};  one-way from the exact "
          f"equilibrium ends {end:.2f}")
    assert made20 >= 0.5 * gap20 and end < 0



# ---------------------------------------------------------------------------
# Stage (b): a want that levels off at a need.
# ---------------------------------------------------------------------------

NEEDS = [0.3, 0.6, 0.9, 1.2]
WANTS = [0.2, 0.8, 1.4, 2.0]
AL_B, R_B, C0_B = 0.5, 4.0, 1.0       # chosen only so that the fund is solvent at full wage


def collective_b(kind, wmax, betas=WANTS):
    if kind == "needful":
        ws = [needful_worker(0.4, alpha=AL_B, b=1.0, gamma=1.0, nu=n, mu=MU_A, label=f"w{j}")
              for j, n in enumerate(NEEDS)]
    else:
        ws = [worker(0.4, alpha=AL_B, b=1.0, beta=be, mu=MU_A, label=f"w{j}")
              for j, be in enumerate(betas)]
    return one_level(ws, positions=[0.06] * N, mobility=MU_P, tau=TAU,
                     fund=dict(f0=20.0, mu=MU_F, wmax=wmax, K=1.0, c0=C0_B, leak=0.1,
                               wire="oneway"),
                     src=source(S_SRC), src_position=L_SRC, r=R_B)


def settle(arr, steps=10000):
    step = closed_step(arr, EMPTY)
    return jax.jit(lambda s: jax.lax.fori_loop(0, steps, lambda i, x: step(x), s))(arr.P.init)


def placements(arr, s):
    return np.asarray(arr.P.position(s))[N:2 * N]


CAPS = (1.4, 1.2, 1.0, 0.8, 0.6, 0.45)


def test_23_where_a_part_settles_carries_its_need():
    """Recorded as run: refuted. A worker the wage overpays stands back; those it underpays stand
    together at the fund, and no one leaves for the source at any cap tried."""
    rows = []
    for kind in ("needful", "linear"):
        for wmax in CAPS:
            arr = collective_b(kind, wmax)
            s = settle(arr)
            rows.append((kind, wmax, placements(arr, s), float(fund_of(s)[0])))
    print("\n  kind     cap   placements (needs or wants rising)     fund")
    for kind, wmax, p, f in rows:
        print(f"  {kind:8} {wmax:4}  {np.round(p, 2)}   {f:.2f}")
    R = {(k, w): p for k, w, p, _ in rows}
    for wmax in CAPS:
        assert np.ptp(R["linear", wmax]) < 0.01                     # linear: together (held)
        p = R["needful", wmax]
        assert np.all(p < 2.0)                                      # no one at the source
        if wmax >= 0.6:                           # farther from the fund the less the need
            assert np.all(np.diff(np.abs(p)) <= 1e-3) and abs(p[0]) - abs(p[3]) > 0.3
        else:                                     # at 0.45 the least needy is nearest
            assert np.argmin(np.abs(p)) == 0


def drainer(arr):
    """``S |-> (placements, balance)`` after draining ``S`` from equilibrium; and the balance."""
    step = closed_step(arr, EMPTY)
    s_eq = settle(arr, 2 * PRE)

    @jax.jit
    def after(S):
        sT = jax.lax.fori_loop(0, POST, lambda i, x: step(x), ((s_eq[0][0], s_eq[0][1] - S), s_eq[1]))
        return arr.P.position(sT)[N:2 * N], fund_of(sT)[0]

    return after, float(fund_of(s_eq)[0])


def bisect_threshold(after, hi=60.0):
    lo = 0.0
    for _ in range(16):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if float(after(mid)[1]) > 0 else (lo, mid)
    return (lo + hi) / 2


def test_24_25_departure_and_the_threshold():
    """Recorded as run. (24) refuted as written: the needful workers go from none to two, the
    neediest two, with no other count. (25) ill-posed as designed; measured after the fact with
    the linear collective matched in balance by its wage cap instead."""
    nf, e_nf = drainer(collective_b("needful", 1.0))
    ln, e_ln = drainer(collective_b("linear", 1.0))
    counts = {}
    for name, after in (("needful", nf), ("linear", ln)):
        counts[name] = []
        for S in np.linspace(0.0, 40.0, 81):
            p, f = after(S)
            counts[name].append((float(S), int(np.sum(np.asarray(p) > 2.0)), float(f), np.asarray(p)))
    for name in counts:
        seen = sorted({c for _, c, _, _ in counts[name]})
        print(f"\n  {name}: counts seen {seen}")
        first = next(r for r in counts[name] if r[1] > 0)
        print(f"  first departures at drain {first[0]:.1f}: placements {np.round(first[3], 2)}, fund {first[2]:.2f}")
    assert sorted({c for _, c, _, _ in counts["linear"]}) == [0, 1, 4]
    assert sorted({c for _, c, _, _ in counts["needful"]}) == [0, 2]
    first = next(r for r in counts["needful"] if r[1] > 0)
    assert np.all(first[3][2:] > 2.0) and np.all(first[3][:2] < 2.0)   # the neediest two leave
    # (25), after the fact: the linear cap that matches the needful balance, by bisection.
    lo, hi = 0.45, 1.0
    for _ in range(12):
        mid = (lo + hi) / 2
        e = float(fund_of(settle(collective_b("linear", mid)))[0])
        lo, hi = (mid, hi) if e < e_nf else (lo, mid)
    lm, e_lm = drainer(collective_b("linear", (lo + hi) / 2))
    th_nf, th_lm = bisect_threshold(nf), bisect_threshold(lm)
    print(f"  (25) needful: balance {e_nf:.2f}, threshold {th_nf:.2f}, margin {th_nf - e_nf:.2f};  "
          f"linear at cap {(lo + hi) / 2:.3f}: balance {e_lm:.2f}, threshold {th_lm:.2f}, "
          f"margin {th_lm - e_lm:.2f}")
    assert abs(e_lm - e_nf) < 0.05 * e_nf
    print(f"  threshold difference, needful minus linear: {th_nf - th_lm:.3f}")
    assert th_nf - th_lm > 0.02 * th_lm



# ---------------------------------------------------------------------------
# Stage (b), rerun: need as a floor, the going rate felt.
# ---------------------------------------------------------------------------

L_NEAR, LAM = 1.5, 0.3
CAPS_F = (1.4, 1.2, 1.0, 0.8, 0.6, 0.45, 0.3, 0.2)


def collective_f(wmax):
    ws = [floor_worker(0.4, alpha=AL_B, b=1.0, lam=LAM, nu=n, mu=MU_A, label=f"w{j}")
          for j, n in enumerate(NEEDS)]
    return one_level(ws, positions=[0.06] * N, mobility=MU_P, tau=TAU,
                     fund=dict(f0=20.0, mu=MU_F, wmax=wmax, K=1.0, c0=C0_B, leak=0.1,
                               wire="oneway"),
                     src=source(S_SRC), src_position=L_NEAR, r=R_B)


def where(p):
    return np.where(np.abs(p) < 0.25, "fund", np.where(np.abs(p - L_NEAR) < 0.25, "else", "between"))


def test_26_27_floor_need_and_a_felt_going_rate():
    rows = []
    for wmax in CAPS_F:
        arr = collective_f(wmax)
        s = settle(arr)
        rows.append((wmax, placements(arr, s), float(fund_of(s)[0])))
    print("\n  cap   placements (needs rising)     where          fund")
    for wmax, p, f in rows:
        print(f"  {wmax:4}  {np.round(p, 3)}   {list(where(p))}  {f:.2f}")
    for _, p, _ in rows:
        assert np.ptp(p) < 0.01                                        # (26)
    RESULT_27a = [len(set(where(p))) == 1 and where(p)[0] != "between" for _, p, _ in rows]
    print(f"  (27) every worker at the fund or every one at the elsewhere, by cap: {RESULT_27a}")
    # Released from the cap-1.0 equilibrium into cap 0.2: when does each pass 1.25?
    s0 = settle(collective_f(1.0))
    arr = collective_f(0.2)
    step = closed_step(arr, EMPTY)

    def body(s, _):
        s = step(s)
        return s, placements_j(arr, s)

    placements_j = lambda arr, s: arr.P.position(s)[N:2 * N]
    _, ps = jax.lax.scan(body, s0, None, length=POST)
    ps = np.asarray(ps)
    arrive = [int(np.argmax(ps[:, j] > 1.25)) if np.any(ps[:, j] > 1.25) else -1 for j in range(N)]
    print(f"  released at cap 0.2: steps to pass 1.25, needs rising: {arrive}")
    assert RESULT_27a == [False] * 5 + [True] * 3        # (27) first clause refuted: hills merge
    assert all(a >= 0 for a in arrive) and np.all(np.diff(arrive) < 0)   # neediest first (held)


def test_28_a_passing_shock_with_floor_need():
    after, e = drainer(collective_f(1.0))
    rows = []
    for S in np.linspace(0.0, 80.0, 161):
        p, f = after(S)
        rows.append((float(S), np.asarray(p), float(f)))
    counts = [int(np.sum(np.abs(p - L_NEAR) < 0.25)) for _, p, _ in rows]
    print(f"\n  equilibrium balance {e:.2f}; counts at the elsewhere seen {sorted(set(counts))}")
    for S, p, f in rows[::10]:
        print(f"  drain {S:5.1f}: placements {np.round(p, 2)}  fund {f:.2f}")
    mixed = [(S, p) for (S, p, _), c in zip(rows, counts) if 0 < c < N]
    by_need = all(np.all(np.diff((np.abs(p - L_NEAR) < 0.25).astype(int)) >= 0) for _, p in mixed)
    print(f"  mixed outcomes {len(mixed)}; the leavers always the neediest: {by_need}")
    assert sorted(set(counts)) == [0]                    # (28) refuted: the fund always recovers
    assert all(abs(f - e) < 0.01 for _, _, f in rows)
