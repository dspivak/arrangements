"""A part that gets nothing moves to where it can get something (book Chapter 1, ``sec.check_claims``: "a part that doesn't
get anything from the collective moves away to wherever it can get something").

For a reader of the book. The pieces are in ``arrangements/cascade.py``:

* a *worker* holds an amplitude ``a`` on the supply ``h_1`` it receives, with cost ``(b/2) a^2 - (alpha a + beta) h_1``, so
  it settles at ``alpha h_1 / b``, and its placement moves toward whatever supplies it (``beta`` is a standing want, so
  a part with no supply still looks for it);
* a *source* is a fixed part off to the side, offering supply: the elsewhere a worker can go;
* a *fund* pays every part in reach a wage set by its balance, is paid what arrives, and pays a fixed overhead, without
  which nothing could tip; it is a record, read one way;
* a *market* pays for the product. In the nested runs, a fixed supply arrives instead.

Each is a box. ``one_level()`` wires four workers, a source, a fund and a market into a closed composite. A shock is the
balance reduced by hand and the run taken from there; nothing in the run produces it. Run from the library's folder with it on the path (``PYTHONPATH=. .venv/bin/python example.py``):

    import jax; jax.config.update("jax_enable_x64", True)
    import numpy as np
    from tests.test_cascade import one_level, outcome, shocker
    u, arr = one_level()
    settled, after = shocker(u, arr)
    print(round(outcome(u, arr, settled)[0], 2), np.round(outcome(u, arr, settled)[1], 2))    # balance, placements
    for shock in (19.0, 21.0):
        balance, placements, amplitudes = outcome(u, arr, after(shock))
        print(shock, round(balance, 2), np.round(placements, 2), np.round(amplitudes, 2))
    # 19.74 [0.06 0.06 0.06 0.06]
    # 19.0 19.74 [0.06 0.06 0.06 0.06] [0.97 0.97 0.97 0.97]
    # 21.0 -18.39 [2.5 2.5 2.5 2.5] [1. 1. 1. 1.]

Below the threshold, about 20.22, the fund recovers and everyone stays. Above it the fund dies, no wage arrives, and every
worker walks to the source, where it keeps holding its form.

What the tests check. Tests 10 and 11: the threshold, and with no source the same shock moves no one; the workers fade
where they are. Test 12: more workers raise the threshold only by raising the balance; the margin stays flat. Tests 13
and 14, nested, one to four levels: a permanent cut in the top's supply runs to every depth, and a passing shock is
filtered by funds that hold reserves. One retroactivity test: with an exact wage wire the workers' want pulls back on the
balance, and the threshold rises.

All-or-none here is forced by symmetry: a worker's leaving depends only on the wage against the outside offer, the same
for each. The file takes about ten minutes; the example above takes seconds.

For agents: the file's earlier head, with the predictions fixed before the code, what happened and the
weirdness checks, is in ``notes/P31-B-CH1-HISTORY-2026-11-03.md``.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.cascade import (
    EMPTY, closed_step, gate, leaf, market, read, source, supplied, unit, worker,
)
from arrangements.wiring import compose_seq

from tests.chain import flat_state
from tests.test_rendezvous import run_phi

TAU = 0.8
ALPHA, B, BETA = 1.0, 1.0, 1.0     # a worker: amplitude alpha h_1 / b, standing want beta
MU_A, MU_P = 0.1, 0.1              # worker amplitude and placement mobilities
L_SRC, S_SRC = 2.5, 1.0            # the source: where, and how much
PRE, POST = 4000, 6000             # steps to equilibrium, and after the shock


def workers(n):
    return [leaf(worker(0.8, alpha=ALPHA, b=B, beta=BETA, mu=MU_A, label=f"w{i}")) for i in range(n)]


# ---------------------------------------------------------------------------
# One level: a unit of n workers and a fund, closed by a market.
# ---------------------------------------------------------------------------


def one_level(n=4, src=True, record=True, wmax=1.0):
    u = unit(workers(n), positions=[0.06] * n, mobilities=[MU_P] * n, tau=TAU,
             fund=dict(f0=20.0, mu=0.05, wmax=wmax, K=1.0, c0=2.0, leak=0.1,
                       record=(ALPHA, BETA) if record else None),
             src=source(S_SRC) if src else None, src_position=L_SRC)
    return u, compose_seq(u.arr, market(2.0))


def shocker(u, arr):
    """The equilibrium, and a jitted map ``S |-> state POST steps after draining S``."""
    step = closed_step(arr, EMPTY)
    s_eq = jax.jit(lambda s: jax.lax.fori_loop(0, PRE, lambda i, x: step(x), s))(arr.P.init)

    @jax.jit
    def after(S):
        s = ((s_eq[0][0], s_eq[0][1] - S), s_eq[1])
        return jax.lax.fori_loop(0, POST, lambda i, x: step(x), s)

    return s_eq, after


def outcome(u, arr, s):
    d = read(u, arr.P.position(s))
    return (float(d["fund"]), np.asarray(d["positions"]),
            np.array([float(c["a"]) for c in d["children"]]))


def threshold(u, arr):
    s_eq, after = shocker(u, arr)
    f_eq = outcome(u, arr, s_eq)[0]
    lo, hi = 0.0, 3 * f_eq
    for _ in range(16):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if outcome(u, arr, after(mid))[0] > 0 else (lo, mid)
    return f_eq, (lo + hi) / 2, after


def test_0_the_run_is_the_dynamics_functors():
    """LICENSES ``closed_step``; establishes nothing about the predictions."""
    u, arr = one_level()
    step, s = closed_step(arr, EMPTY), arr.P.init
    for _ in range(5):
        s = step(s)
    np.testing.assert_allclose(np.asarray(flat_state(s)), run_phi(arr, 5, EMPTY)[-1], atol=1e-13)


def test_retroactivity_the_needy_prop_up_the_purse_unless_the_wire_is_one_way():
    """A finding of the construction, not a prediction. With exact composition the fund's
    forcing exceeds its declared flow by ``g'(f)(alpha sum kappa a + beta sum kappa)``: the
    workers' want, pulled back through the wage wire. ``record`` removes exactly that."""
    for record in (False, True):
        u, arr = one_level(record=record)
        s_eq, _ = shocker(u, arr)
        q = arr.P.position(s_eq)
        xi_f = float(arr.omega(q, EMPTY, EMPTY)[0][-1])
        d = read(u, q)
        f = d["fund"]
        a = jnp.array([c["a"] for c in d["children"]])
        k = jnp.exp(-d["positions"] ** 2 / (2 * TAU))
        declared = float(2.0 * jnp.sum(k * a) - gate(f, 1.0, 1.0) * jnp.sum(k) - 2.0 - 0.1 * f)
        back = float(jax.grad(gate)(f, 1.0, 1.0) * (ALPHA * jnp.sum(k * a) + BETA * jnp.sum(k)))
        if record:
            assert abs(xi_f - declared) < 1e-12
        else:
            assert abs(xi_f - declared - back) < 1e-12 and back > 1e-3
    # and it moves the threshold: the exact fund survives larger shocks
    th_exact = threshold(*one_level(record=False))[1]
    th_record = threshold(*one_level(record=True))[1]
    print(f"\n  threshold, exact wire {th_exact:.3f}; one-way wire {th_record:.3f}")
    assert th_exact > th_record


def test_10_11_a_threshold_and_nowhere_to_go():
    rows = []
    for src in (True, False):
        u, arr = one_level(src=src)
        f_eq, Sc, after = threshold(u, arr)
        below, above = outcome(u, arr, after(0.95 * Sc)), outcome(u, arr, after(1.05 * Sc))
        rows.append((src, f_eq, Sc, below, above))
        assert below[0] > 0 and np.all(np.abs(below[1]) < 0.5)            # all stay
        assert above[0] < 0
        if src:                                                            # (10) all leave
            assert np.all(above[1] > 2.0) and np.all(above[2] > 0.9)
        else:                                                              # (11) none move
            assert np.all(np.abs(above[1] - below[1]) < 0.1) and np.all(above[2] < 0.01)
    print("\n  source  balance  threshold   below: fund, placements, amplitudes   above: the same")
    for src, f_eq, Sc, b_, a_ in rows:
        print(f"  {src!s:6}  {f_eq:.2f}  {Sc:.3f}   {b_[0]:.2f} {np.round(b_[1], 2)} {np.round(b_[2], 3)}"
              f"   {a_[0]:.2f} {np.round(a_[1], 2)} {np.round(a_[2], 3)}")


def test_12_the_threshold_and_the_number_of_workers():
    """At fixed total output (``wmax = 4/n``). Recorded as run: P19's rise holds in letter
    and the agent's flat counter-guess fails; the rise is the balance's, not the margin's."""
    rows = {}
    for src in (True, False):
        for n in (2, 4, 8):
            f_eq, Sc, _ = threshold(*one_level(n=n, src=src, wmax=4.0 / n))
            rows[src, n] = (f_eq, Sc)
    print("\n  n   with source: balance, threshold, margin     without: balance, threshold, margin")
    for n in (2, 4, 8):
        (f1, s1), (f0, s0) = rows[True, n], rows[False, n]
        print(f"  {n}   {f1:.3f} {s1:.3f} {s1 - f1:.3f}     {f0:.3f} {s0:.3f} {s0 - f0:.3f}")
    s_with = [rows[True, n][1] for n in (2, 4, 8)]
    assert s_with[2] / s_with[0] > 1.10                       # P19's letter; the counter-guess fails
    margins = [rows[True, n][1] - rows[True, n][0] for n in (2, 4, 8)]
    assert max(margins) - min(margins) < 0.25                  # the margin does not grow


# ---------------------------------------------------------------------------
# Nested: a binary tree of funds, workers and their sources at the bottom.
# ---------------------------------------------------------------------------

WB = 0.8          # the wage a bottom worker is paid at equilibrium


def tree(D, K_mid):
    """A unit of height ``h`` pays each of its two children about ``WB 2^(h-1)``, with
    ``wmax`` 1.25 times that and the gate's half-saturation ``K 2^(h-1)``, so every fund's
    equilibrium balance is its reserve ``4 K 2^(h-1)``: ``K_mid`` below the top, 1 at the top."""

    def build(h, top):
        if h == 0:
            return workers(1)[0]
        need = WB * 2 ** (h - 1)
        K = (1.0 if top else K_mid) * 2 ** (h - 1)
        return unit([build(h - 1, False) for _ in range(2)],
                    positions=[0.06, 0.06] if h == 1 else [0.0, 0.0],
                    mobilities=[MU_P] * 2 if h == 1 else [0.0] * 2, tau=TAU,
                    fund=dict(f0=4 * K, mu=0.05, wmax=1.25 * need, K=K, c0=0.0, leak=0.0,
                              record=(ALPHA, BETA) if h == 1 else None),
                    src=source(S_SRC) if h == 1 else None, src_position=L_SRC, label=f"u{h}")

    return build(D, True)


def first_branch(u, K_mid, q):
    """The wage each level pays down the first branch, top first, and the bottom placements."""
    d, node, ws = read(u, q), u, []
    while True:
        h = node.depth
        K = (1.0 if node is u else K_mid) * 2 ** (h - 1)
        ws.append(gate(d["fund"], 1.25 * WB * 2 ** (h - 1), K))
        if h == 1:
            return jnp.array(ws), d["positions"]
        d, node = d["children"][0], node.children[0]


def depth_runner(D, K_mid, post=4000):
    """A jitted map ``(S, cut) |-> (lowest wage / equilibrium by level, bottom placements at
    the end)``: drain ``S`` from the top fund and scale the top supply by ``cut``."""
    u = tree(D, K_mid)
    top = WB * 2 ** D
    step = closed_step(u.arr, supplied(top))
    s_eq = jax.jit(lambda s: jax.lax.fori_loop(0, 3000, lambda i, x: step(x), s))(u.arr.P.init)
    eq, _ = first_branch(u, K_mid, u.arr.P.position(s_eq))

    @jax.jit
    def go(S, cut):
        step2 = closed_step(u.arr, supplied(top * cut))

        def body(s, _):
            s = step2(s)
            return s, first_branch(u, K_mid, u.arr.P.position(s))[0]

        sT, ws = jax.lax.scan(body, (s_eq[0], s_eq[1] - S), None, length=post)
        return jnp.min(ws, axis=0) / eq, first_branch(u, K_mid, u.arr.P.position(sT))[1]

    return go


FAST, SLOW = 0.05, 5.0


def test_13_14_how_deep_a_shock_runs():
    rows = []
    for K_mid in (FAST, SLOW):
        for D in (1, 2, 3, 4):
            go = depth_runner(D, K_mid)
            reserve = 4 * 2 ** (D - 1)
            for kind, S, cut in (("cut", 0.0, 0.3), ("deep", 4 * reserve, 1.0),
                                 ("moderate", 0.95 * reserve, 1.0)):
                mins, pos = go(S, cut)
                rows.append((K_mid, D, kind, np.asarray(mins), bool(np.all(np.asarray(pos) > 2.0))))
    print("\n  funds  D  shock      lowest wage / equilibrium, top first      workers left")
    for K_mid, D, kind, mins, left in rows:
        print(f"  {'fast' if K_mid == FAST else 'slow'}   {D}  {kind:9}  {np.round(mins, 3)}  {left}")
    R = {(k, D, kind): (m, left) for k, D, kind, m, left in rows}
    for K_mid in (FAST, SLOW):
        for D in (1, 2, 3, 4):
            assert R[K_mid, D, "cut"][1]                                    # (13)
    for D in (1, 2, 3, 4):                                                  # (14), deep shock
        m, left = R[FAST, D, "deep"]
        assert left and m[-1] - m[0] < 0.1
    assert R[SLOW, 1, "deep"][1]
    for D in (2, 3, 4):
        assert not R[SLOW, D, "deep"][1]
    bottoms = [R[SLOW, D, "deep"][0][-1] for D in (2, 3, 4)]
    assert bottoms[0] < bottoms[1] < bottoms[2]
    # Refuted in part: a moderate shock is attenuated level by level even through fast funds.
    m4 = R[FAST, 4, "moderate"][0]
    assert np.all(np.diff(m4) > 0) and m4[-1] - m4[0] > 0.2
    assert not any(R[FAST, D, "moderate"][1] for D in (1, 2, 3, 4))
