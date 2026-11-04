"""The wage that brings parts back is several times the wage at which they left (book Chapter 1, ``sec.check_claims``: "In
a town, where parts do not feel each other's push directly, the collective's pull has to reach a part as what it offers").

For a reader of the book. This is the collective of ``tests/test_cascade.py``: workers, a fund paying a wage ``g``, and a
source offering ``s`` at distance ``d``. A worker's placement ``p`` climbs the supply it receives,

    h_1(p) = g kappa(p) + s kappa(p - d),     kappa(x) = exp(-x^2 / 2 tau),

and no other worker enters this: supply and product travel on separate channels, so the parts feel each other only
through what the fund can pay. That is the paragraph's "do not feel each other's push directly". With the wage ratio
``rho = g / s``, ``h_1`` has a rest at ``p`` where ``rho = phi(p) = ((d - p) / p) exp(d (2p - d) / 2 tau)``. Between
``phi``'s local minimum and maximum, ``phi_min < rho < phi_max``, both home and the source are rests. So a worker at
home leaves only when the wage falls to ``phi_min s``, and one at the source returns only when it rises to
``phi_max s``. The two wages differ, and their ratio is the "several times". Run from the library's folder with it on the path (``PYTHONPATH=. .venv/bin/python example.py``):

    import jax; jax.config.update("jax_enable_x64", True)
    import numpy as np
    from tests.test_ratchet import critical_wages, phi_bounds
    print(np.round(phi_bounds(), 4))                       # the turning points, and phi_min, phi_max in closed form
    g_out, g_in = critical_wages(1, 2.5, 0.8)              # measured in the run, about 40 seconds
    print(round(g_out, 3), round(g_in, 3), round(g_in / g_out, 2))
    # [0.3768 2.1232 0.368  2.7177]
    # 0.368 2.717 7.38

The ratio is one geometry's, not a fact about towns: it depends on ``d^2 / tau`` alone, from 1 at ``d^2 / tau = 4``, below
which there is no such band, to about 5,400 at 16 (``notes/COMPOSITION-AUDIT.md``, entry 10).

What the tests check. R1: a fund that cannot pay more than the outside offer cannot call anyone back, whatever is paid in.
R2: the two wages match the closed form, in two geometries. R3: they are the same for one worker and four, so parts do not
call each other back. R4, R5: where the fund can outbid the offer the whole loop closes, and a collective re-formed whole
is the original, remembering nothing. R6: if it re-forms short of a part it is weaker, a scar and not a ratchet. R7:
lowering the offer and raising the wage are one condition.

Re-forming comes from outside the run: money added to the balance by hand, or the offer lowered by hand. This collective is
held together by what it is paid; its parts do not supply one another. The file takes about eleven minutes.

For agents: the file's earlier head, with the predictions fixed before the code, what happened and the
weirdness checks, is in ``notes/P31-B-CH1-HISTORY-2026-11-03.md``.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.cascade import (
    EMPTY, closed_step, gate, leaf, market, read, source, unit, worker,
)
from arrangements.wiring import compose_seq

TAU, D_SRC, S_SRC = 0.8, 2.5, 1.0        # kernel width; where the source is, and its offer
ALPHA, B, BETA = 1.0, 1.0, 1.0           # a worker, as in the cascade run
MU_A, K_GATE, C0, LEAK, R_MKT = 0.1, 1.0, 2.0, 0.1, 2.0
WM_LO, MU_LO = 1.0, 0.1                  # the cascade run's own setting: wage cap = the offer
WM_HI, MU_HI = 4.0, 0.02                 # a fund that can outbid the source (and a stable step)
PRE, POST, REV = 60_000, 60_000, 300_000


def setup(n=4, *, wmax, s=S_SRC, d=D_SRC, tau=TAU, mu_f=0.05, mu_p=MU_LO, f0=20.0, pos=0.06):
    """One level of ``cascade.py``: ``n`` workers, a fund, a source at ``d``, a market."""
    ws = [leaf(worker(0.8, alpha=ALPHA, b=B, beta=BETA, mu=MU_A, label=f"w{i}")) for i in range(n)]
    u = unit(ws,
             positions=[pos] * n if np.isscalar(pos) else list(pos),
             mobilities=[mu_p] * n if np.isscalar(mu_p) else list(mu_p),
             tau=tau,
             fund=dict(f0=f0, mu=mu_f, wmax=wmax, K=K_GATE, c0=C0, leak=LEAK,
                       record=(ALPHA, BETA)),
             src=source(s), src_position=d)
    return u, compose_seq(u.arr, market(R_MKT))


def add_f(state, r):
    """Pay ``r`` into the fund (or drain it, for ``r < 0``)."""
    return ((state[0][0], state[0][1] + r), state[1])


def runner(arr, steps):
    step = closed_step(arr, EMPTY)
    return jax.jit(lambda s: jax.lax.fori_loop(0, steps, lambda i, x: step(x), s))


def readout(u, arr, state):
    """``(balance, placements, amplitudes)``."""
    d = read(u, arr.P.position(state))
    return (float(d["fund"]), np.asarray(d["positions"]),
            np.array([float(c["a"]) for c in d["children"]]))


def one_step_move(arr, state):
    """The largest coordinate change in one further step: 0 at a fixed point, large on a
    limit cycle. Every equilibrium this file reports is guarded by it (docstring, (i))."""
    q1 = np.asarray(arr.P.position(state))
    q2 = np.asarray(arr.P.position(jax.jit(closed_step(arr, EMPTY))(state)))
    return float(np.max(np.abs(q2 - q1)))


def bisect(pred, lo, hi, iters=14):
    """``pred(lo)`` false, ``pred(hi)`` true: the switch point."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if pred(mid) else (mid, hi)
    return (lo + hi) / 2


def phi_bounds(d=D_SRC, tau=TAU):
    """``(p_1, p_2, phi_min, phi_max)`` of the docstring: the turning points of ``phi`` and
    its two values there, the closed form derived by hand before the run."""
    root = np.sqrt(d * d - 4 * tau)
    p1, p2 = (d - root) / 2, (d + root) / 2
    e = np.exp(d * (p2 - p1) / (2 * tau))
    return p1, p2, (p2 / p1) / e, (p1 / p2) * e


def wage(f, wmax):
    return float(gate(jnp.array(float(f)), wmax, K_GATE))


def home(u, arr, state):
    """How many placements are back within reach of the fund."""
    return int(np.sum(np.abs(readout(u, arr, state)[1]) < 1.0))


def collapsed(wmax, mu_p, f0, iters=12):
    """The end state of a run from the equilibrium with a drain above the threshold taken
    from the balance by hand: every worker at the source, fund dead.
    Returns ``(u, arr, equilibrium, threshold, collapsed state)``."""
    u, arr = setup(wmax=wmax, mu_p=mu_p, f0=f0)
    eq = runner(arr, PRE)(arr.P.init)
    post = runner(arr, POST)
    f_eq = readout(u, arr, eq)[0]
    S = bisect(lambda x: readout(u, arr, post(add_f(eq, -x)))[0] < 0, 0.0, 3 * f_eq, iters)
    return u, arr, eq, S, runner(arr, REV)(add_f(eq, -1.05 * S))


# ---------------------------------------------------------------------------
# The landscape, derived by hand (docstring) and checked here.
# ---------------------------------------------------------------------------


def test_the_landscape_is_what_the_hand_derivation_says():
    """Not a prediction: the closed forms of the docstring, checked against ``phi`` itself
    and against where a placement actually settles."""
    for d, tau in ((D_SRC, TAU), (2.0, 0.5)):
        p1, p2, lo, hi = phi_bounds(d, tau)
        ps = np.linspace(1e-4, d - 1e-4, 2_000_001)
        phi = ((d - ps) / ps) * np.exp(d * (2 * ps - d) / (2 * tau))
        inner = phi[(ps > p1 / 2) & (ps < p2)]
        assert abs(np.min(inner) - lo) < 1e-6 and abs(np.max(phi[ps > p1]) - hi) < 1e-6
        assert abs(p1 * (d - p1) - tau) < 1e-12 and abs(p2 * (d - p2) - tau) < 1e-12
    # and the placement settles at the root of h', with the wage the fund happens to pay
    u, arr = setup(wmax=WM_LO)
    eq = runner(arr, 6000)(arr.P.init)
    f, p, a = readout(u, arr, eq)
    g = wage(f, WM_LO)
    hp = (-(g * p[0] / TAU) * np.exp(-p[0] ** 2 / (2 * TAU))
          - (S_SRC * (p[0] - D_SRC) / TAU) * np.exp(-(p[0] - D_SRC) ** 2 / (2 * TAU)))
    print(f"\n  equilibrium: fund {f:.4f} wage {g:.4f} placement {p[0]:.6f} amplitude {a[0]:.6f}"
          f"  h'(p) = {hp:.2e}  one-step move {one_step_move(arr, eq):.1e}")
    assert abs(hp) < 1e-6 and one_step_move(arr, eq) < 1e-6
    # the amplitude is the supply it receives, alpha h_1 / b
    h1 = g * np.exp(-p[0] ** 2 / (2 * TAU)) + S_SRC * np.exp(-(p[0] - D_SRC) ** 2 / (2 * TAU))
    assert abs(a[0] - h1) < 1e-5


# ---------------------------------------------------------------------------
# (R1) Money cannot outbid the elsewhere.
# ---------------------------------------------------------------------------


def test_R1_money_cannot_outbid_the_elsewhere():
    u, arr, eq, S, coll = collapsed(WM_LO, MU_LO, 20.0)
    print(f"\n  wmax = s = 1: equilibrium {readout(u, arr, eq)[0]:.4f}, threshold {S:.4f},"
          f" collapsed {readout(u, arr, coll)[0]:.4f}")
    long = runner(arr, 100_000)
    for R in (10.0, 100.0, 1000.0, 10000.0):
        f, p, a = readout(u, arr, long(add_f(coll, R)))
        print(f"    inject {R:8.0f}: fund {f:9.3f}  placements {np.round(p, 4)}  amplitudes {np.round(a, 3)}")
        assert np.all(p > 2.0) and np.all(np.abs(a - 1.0) < 1e-3)
        assert abs(f + 18.391) < 1e-2          # and the money drains away


# ---------------------------------------------------------------------------
# (R2), (R3) The two wages, their ratio, and the number of parts.
# ---------------------------------------------------------------------------


WM_CLAMP, T_CLAMP, MU_CLAMP = 10.0, 300_000, 0.03


def clamped_scan(n, d, tau, start):
    """``f0s |-> the placements each run settles at``, with the fund's balance clamped (fund
    mobility 0), so the wage ``g(f0)`` is a dial and nothing feeds back; vmapped over the
    balances. The placement mobility is 0.03 here: the critical wage is a property of the
    landscape, not of the speed, and 0.1 is unstable at the wages this sweep reaches."""
    u, arr = setup(n=n, wmax=WM_CLAMP, d=d, tau=tau, mu_f=0.0, f0=0.0, pos=start, mu_p=MU_CLAMP)
    step = closed_step(arr, EMPTY)

    def go(f0):
        state = ((arr.P.init[0][0], jnp.array([f0])), arr.P.init[1])
        state = jax.lax.fori_loop(0, T_CLAMP, lambda i, x: step(x), state)
        return jnp.asarray(read(u, arr.P.position(state))["positions"])

    return jax.jit(jax.vmap(go))


def transition(scan, pred, lo, hi, points=48, passes=2):
    """The balance at which ``pred`` on the settled placements first turns true, by two
    passes of a grid rather than a bisection: one vmapped call does a whole grid."""
    for _ in range(passes):
        fs = np.linspace(lo, hi, points)
        flags = np.array([pred(v) for v in np.asarray(scan(jnp.array(fs)))])
        i = int(np.argmax(flags))
        assert 0 < i, "the transition is outside the grid"
        lo, hi = float(fs[i - 1]), float(fs[i])
    return (lo + hi) / 2


def critical_wages(n, d, tau):
    """The highest wage at which a worker at home leaves, and the lowest at which one at the
    source comes back. Which basin is read in the geometry's own units."""
    out, back = 0.8 * d, 0.3 * d
    g_out = wage(transition(clamped_scan(n, d, tau, 0.06),
                            lambda p: not np.all(p > out), 1e-4, 0.1), WM_CLAMP)
    g_in = wage(transition(clamped_scan(n, d, tau, d),
                           lambda p: bool(np.all(np.abs(p) < back)), 0.05, 0.6), WM_CLAMP)
    return g_out, g_in


def test_R2_R3_the_two_wages_their_ratio_and_the_number_of_parts():
    print("\n  d    tau   n   g_out (phi_min)      g_in (phi_max)       ratio (predicted)")
    rows = {}
    for d, tau, n in ((D_SRC, TAU, 1), (D_SRC, TAU, 4), (2.0, 0.5, 1)):
        g_out, g_in = critical_wages(n, d, tau)
        _, _, lo, hi = phi_bounds(d, tau)
        rows[d, tau, n] = (g_out, g_in)
        print(f"  {d}  {tau}   {n}   {g_out:.5f} ({lo:.5f})   {g_in:.5f} ({hi:.5f})   "
              f"{g_in / g_out:.4f} ({hi / lo:.4f})")
        assert abs(g_out / lo - 1) < 0.02 and abs(g_in / hi - 1) < 0.02          # (R2)
        assert abs((g_in / g_out) / (hi / lo) - 1) < 0.02
    for i in (0, 1):                                                             # (R3)
        assert abs(rows[D_SRC, TAU, 1][i] / rows[D_SRC, TAU, 4][i] - 1) < 0.01


# ---------------------------------------------------------------------------
# (R4), (R5) The loop, and what the re-formed collective remembers.
# ---------------------------------------------------------------------------


def f_for_wage(g, wmax):
    return bisect(lambda f: wage(f, wmax) > g, 0.0, 1e4, 50)


def test_R4_R5_the_loop_and_what_it_remembers():
    u, arr, eq, S1, coll = collapsed(WM_HI, MU_HI, 100.0)
    f_eq, p_eq, a_eq = readout(u, arr, eq)
    f_c = readout(u, arr, coll)[0]
    print(f"\n  equilibrium: fund {f_eq:.4f} wage {wage(f_eq, WM_HI):.4f} placements {np.round(p_eq, 6)}"
          f" amplitudes {np.round(a_eq, 6)} | one-step move {one_step_move(arr, eq):.1e}")
    print(f"  first threshold {S1:.4f}; collapsed: fund {f_c:.4f} placements"
          f" {np.round(readout(u, arr, coll)[1], 4)} | one-step move {one_step_move(arr, coll):.1e}")
    assert one_step_move(arr, eq) < 1e-9 and one_step_move(arr, coll) < 1e-9
    assert np.all(readout(u, arr, coll)[1] > 2.0) and f_c < 0

    # (R4) the critical injection, against the quasi-static landscape value
    rev = runner(arr, REV)
    R_c = bisect(lambda R: home(u, arr, rev(add_f(coll, R))) == 4, 0.0, 120.0, 12)
    R_hand = f_for_wage(phi_bounds()[3] * S_SRC, WM_HI) - f_c
    print(f"  critical injection R_c = {R_c:.4f} (balance after {f_c + R_c:.4f}, wage"
          f" {wage(f_c + R_c, WM_HI):.4f}); quasi-static {R_hand:.4f}; error {100 * (R_c / R_hand - 1):+.1f}%")
    assert R_c > 0 and 1.20 < R_c / R_hand         # refuted as stated: it needs far more, not 20%

    # the explanation: the balance decays while they walk, so a faster walk needs less money
    u2, arr2, _, _, coll2 = collapsed(WM_HI, 0.05, 100.0)
    rev2 = runner(arr2, REV)
    R_fast = bisect(lambda R: home(u2, arr2, rev2(add_f(coll2, R))) == 4, 0.0, 120.0, 12)
    print(f"  the same at placement mobility 0.05 (a walk 2.5 times faster): R_c = {R_fast:.4f}")
    assert R_fast < R_c

    # (R5) all four return: is the re-formed collective the original?
    ref = runner(arr, REV)(rev(add_f(coll, 1.02 * R_c)))
    f_r, p_r, a_r = readout(u, arr, ref)
    print(f"  re-formed:  fund {f_r:.6f} placements {np.round(p_r, 6)} amplitudes {np.round(a_r, 6)}"
          f" | one-step move {one_step_move(arr, ref):.1e}")
    print(f"  original:   fund {f_eq:.6f} placements {np.round(p_eq, 6)} amplitudes {np.round(a_eq, 6)}")
    assert one_step_move(arr, ref) < 1e-9
    assert (abs(f_r - f_eq) < 1e-4 and np.max(np.abs(p_r - p_eq)) < 1e-4
            and np.max(np.abs(a_r - a_eq)) < 1e-4)
    post = runner(arr, POST)
    S2 = bisect(lambda x: readout(u, arr, post(add_f(ref, -x)))[0] < 0, 0.0, 3 * f_eq, 12)
    print(f"  second threshold {S2:.4f} against the first {S1:.4f}: {100 * (S2 / S1 - 1):+.4f}%")
    assert abs(S2 / S1 - 1) < 0.005


# ---------------------------------------------------------------------------
# (R6) Why the injection route is all-or-none: the door, and who holds it open.
# ---------------------------------------------------------------------------


def test_R6_the_first_ones_home_hold_the_door_open():
    """A surviving collective of ``m`` pays a wage; if that wage is above ``g_in`` the
    landscape stays open and every straggler follows, however slow."""
    g_in = phi_bounds()[3] * S_SRC
    print(f"\n  the wage a surviving collective pays, against g_in = {g_in:.4f}")
    wages = {}
    for wmax, mu_p, f0 in ((WM_HI, MU_HI, 100.0), (WM_LO, MU_LO, 20.0)):
        for m in (1, 2, 3, 4):
            um, am = setup(n=m, wmax=wmax, mu_p=mu_p, f0=f0)
            st = runner(am, 200_000)(am.P.init)
            f, p, a = readout(um, am, st)
            wages[wmax, m] = (f, wage(f, wmax))
            print(f"    wmax {wmax}  m = {m}: fund {f:9.4f}  wage {wage(f, wmax):.4f}"
                  f"  placements {np.round(p, 4)}  {'alive' if f > 0 else 'dead'}"
                  f" | one-step move {one_step_move(am, st):.1e}")
            assert one_step_move(am, st) < 1e-9
    assert all(wages[WM_HI, m][1] > g_in for m in (1, 2, 3, 4))      # the door never closes
    assert wages[WM_LO, 1][0] < 0 and wages[WM_LO, 2][0] < 0         # and at wmax = 1 it does
    assert wages[WM_LO, 3][1] < g_in < WM_HI

    # so no band of injections re-forms it short a part, even at eight-fold speeds
    u, arr, eq, S, coll = collapsed(WM_HI, MU_HI, 100.0)
    u6, arr6 = setup(wmax=WM_HI, mu_p=[0.04, 0.02, 0.01, 0.005], f0=100.0)
    eq6 = runner(arr6, PRE)(arr6.P.init)
    coll6 = runner(arr6, REV)(add_f(eq6, -1.05 * S))
    rev6 = runner(arr6, REV)
    counts = []
    for R in np.linspace(20.0, 95.0, 11):
        m = home(u6, arr6, rev6(add_f(coll6, float(R))))
        counts.append((round(float(R), 2), m))
    print(f"  injections and how many of the four came home: {counts}")
    assert set(m for _, m in counts) <= {0, 4}

    # a worker twenty times slower than the rest still walks home
    u7, arr7 = setup(wmax=WM_HI, mu_p=[0.02, 0.02, 0.02, 0.001], f0=100.0)
    eq7 = runner(arr7, PRE)(arr7.P.init)
    coll7 = runner(arr7, REV)(add_f(eq7, -1.05 * S))
    st7 = runner(arr7, 600_000)(add_f(coll7, 60.0))
    print(f"  with one worker at mobility 0.001: returned {home(u7, arr7, st7)}/4,"
          f" placements {np.round(readout(u7, arr7, st7)[1], 4)}")
    assert home(u7, arr7, st7) == 4


# ---------------------------------------------------------------------------
# (R6), (R7) The second route: the elsewhere dries up, for a while.
# ---------------------------------------------------------------------------


MOBS_W = [0.2, 0.1, 0.05, 0.025]


def window(coll, s_low, W, mobs=MU_LO, refill=1e5, settle=200_000):
    """Refill the fund, drop the source's offer to ``s_low`` for ``W`` steps, restore it, and
    settle. The two arrangements differ only in the source's offer, so the state carries."""
    _, arr_lo = setup(wmax=WM_LO, s=s_low, mu_p=mobs)
    u_hi, arr_hi = setup(wmax=WM_LO, s=S_SRC, mu_p=mobs)
    st = runner(arr_lo, W)(add_f(coll, refill))
    return u_hi, arr_hi, runner(arr_hi, settle)(st)


def test_R6_R7_the_offer_the_window_and_the_scar():
    u, arr, eq, S1, coll = collapsed(WM_LO, MU_LO, 20.0)
    f_eq = readout(u, arr, eq)[0]

    # (R7) how far the offer must fall, against g / phi_max
    g_cap = wage(1e5, WM_LO)
    def stays_out(s_low):
        u2, arr2, st = window(coll, s_low, 20_000)
        return home(u2, arr2, st) < 4

    s_star = bisect(stays_out, 0.0, 1.0)
    s_hand = g_cap / phi_bounds()[3]
    print(f"\n  critical offer s* = {s_star:.5f} against g / phi_max = {s_hand:.5f}:"
          f" {100 * (s_star / s_hand - 1):+.2f}%")
    assert abs(s_star / s_hand - 1) < 0.05

    # (R6)(i) the window: who gets home before the elsewhere recovers
    print(f"  the window, offer 0.2, refill 200, mobilities {MOBS_W}")
    partial = None
    for W in (100, 200, 300, 400, 800, 2000):
        u2, arr2, st = window(coll, 0.2, W, mobs=MOBS_W, refill=200.0)
        f, p, a = readout(u2, arr2, st)
        m = home(u2, arr2, st)
        print(f"    W {W:5d}: returned {m}/4  fund {f:8.4f}  placements {np.round(p, 4)}"
              f" | one-step move {one_step_move(arr2, st):.1e}")
        assert one_step_move(arr2, st) < 1e-9
        if 0 < m < 4 and partial is None:
            partial = (W, m, u2, arr2, st, f)
        if W >= 400:
            assert m == 4 and abs(f - f_eq) < 1e-3          # (R5) again, in this route
    assert partial is not None                               # the band is not empty
    W, m, u2, arr2, st, f_m = partial

    # (R6)(ii) the scar: the re-formed collective, short a part, breaks sooner
    post2 = runner(arr2, POST)
    S_m = bisect(lambda x: readout(u2, arr2, post2(add_f(st, -x)))[0] < 0, 0.0, 60.0)
    print(f"  re-formed short one part (W = {W}, m = {m}): fund {f_m:.4f}, threshold {S_m:.4f}"
          f" against {S1:.4f} ({100 * (S_m / S1 - 1):+.2f}%); buffer {S_m - f_m:+.4f} against"
          f" {S1 - f_eq:+.4f}")
    assert S_m < 0.99 * S1                                   # a scar, never a ratchet
    assert S_m - f_m < 0 < S1 - f_eq                         # and it has lost its buffer

    # the part left out is not gone: it still sends its product in from the source
    u3, arr3 = setup(n=m, wmax=WM_LO, mu_p=MU_LO)
    f3 = readout(u3, arr3, runner(arr3, 200_000)(arr3.P.init))[0]
    trickle = R_MKT * np.exp(-readout(u2, arr2, st)[1][-1] ** 2 / (2 * TAU)) / LEAK
    print(f"  three that came home hold {f_m:.4f}; three that never left hold {f3:.4f};"
          f" the distant part's trickle accounts for {trickle:.4f}")
    assert abs((f_m - f3) - trickle) < 0.05
