"""A figurine as a net, its belief a market, run by the runner (book plan P37 (b)(iii′) stage (A);
``arrangements/market_scene.py``, the kit and nets of ``arrangements/runner.py`` enlarged).

Predictions fixed before any construction code, in the book repo's
``notes/runs/p37-market-of-scenes-3819a169/PREDICTIONS.md`` (Claude Fable 5.1, 3819a169, 2026-10-08), their
numbers from ``by_hand.log``; results in the same folder's ``RESULTS.md``.

A1. The two boxes against their terms: the Euler term ``(1 + v/N)^N`` falls short of ``exp`` by 4.93e-4, 4.93e-5,
    2.47e-5 at N = 1e3, 1e4, 2e4 (v = 0.7); the Newton term is within 3.9e-14 of ``recip`` after four steps at
    v = 1.7 from 0.5; through the library a cost on ``exp(x)`` moves ``x`` by ``-e^v``, on ``recip(x)`` by ``+1/v^2``.
A2. Ports' backward direction: ``couple`` (library), ``couple_by_hand`` and ``couple_net`` (one net, by hand and
    through the library) agree in states and covectors to 1e-13 on the two-tank pair, the figurine and the world,
    and random pairs; the relayed two tanks are (2.9, 1.0), (2.805, 1.1), (2.71475, 1.195) after three steps,
    the direct net (2.9, 1.1), (2.81, 1.19), (2.729, 1.271); ``v1 + v2 + r = 4`` exactly; both reach 2.
A3. The market is Bayes: through the library the belief after each round equals P36's direct posterior to 1e-12;
    after six outcomes (0, 0.9546, 0.0454, 0, 0); the guess before each round (5.068, 5.829, 5.748, 6.113, 6.204,
    6.234); the covector returned on the outcome's port is ``G - x`` (P36's pushback), and it lands on the
    world's drum cell after that cell is presented.
A4. The runner runs it: the six-drum runner's decoded market heights equal ``ref_step``'s to 1e-12 and the direct
    posterior to 1e-10 after each pass; the runner is an arrangement over the enlarged kit; the net needs between
    500 and 5,000 runner steps a round (the by-hand count is 37,824: see RESULTS).
A5. P34's tests pass unchanged (``tests/test_runner.py``, run separately).

Run with ``PYTEST_ADDOPTS=-s`` to see the numbers. The library's compiled step of the 141-node coupled net takes
about 80 s to trace; the module fixture takes it once.
"""
import math
import random
from fractions import Fraction as F

import jax
import numpy as np
import pytest

from arrangements import market_scene as S
from arrangements import runner as R

jax.config.update("jax_enable_x64", True)


def _softmax(l):
    l = np.asarray(l, float)
    q = np.exp(l - l.max())
    return q / q.sum()


def _floats(st):
    return {w: ([float(x) for x in v] if isinstance(v, list) else float(v)) for w, v in st.items()}


# ---------------------------------------------------------------------------
# A1: the two boxes against their terms.
# ---------------------------------------------------------------------------


def test_A1a_the_euler_term_approaches_exp():
    errs = {}
    for N in (1000, 10000, 20000):
        net = S.euler_exp_net(0.7, N)
        st = net.init_state()
        for _ in range(N):
            st = R.ref_step(net, st)
        errs[N] = math.exp(0.7) - st[0]
    print(f"\nA1a Euler: e^0.7 - (1 + 0.7/N)^N = {errs}")
    assert abs(errs[1000] - 4.93e-4) < 0.02e-4
    assert abs(errs[10000] - 4.93e-5) < 0.02e-5
    assert abs(errs[20000] - 2.47e-5) < 0.02e-5


def test_A1b_the_newton_term_reaches_recip():
    net = S.newton_recip_net(1.7, 0.5)
    st = net.init_state()
    errs = []
    for _ in range(5):
        st = R.ref_step(net, st)
        errs.append(abs(st[0] - 1 / 1.7))
    print(f"\nA1b Newton: |y_k - 1/1.7| = {errs}")
    assert errs[3] < 1e-13 and errs[4] == 0.0


def test_A1c_the_boxes_backward_direction_through_the_library():
    for mk, v, want in ((S.exp_cost_net, 0.7, -math.exp(0.7)), (S.recip_cost_net, 1.7, 1 / 1.7**2)):
        net = mk(v)
        lib = R.library_run(net, 2)
        st = net.init_state()
        for k in range(2):
            st = R.ref_step(net, st)
            assert lib[k] == pytest.approx(R.flat_ref(net, st), abs=1e-12)
        print(f"\nA1c {mk.__name__}: one step moved x by {lib[0][0] - v:.12f} (want {want:.12f})")
        assert abs((lib[0][0] - v) - want) < 1e-12


# ---------------------------------------------------------------------------
# A2: ports' backward direction.
# ---------------------------------------------------------------------------


def _flat_coupled(net_a, net_b, sa, sb):
    """The two nets' states in the order the library's `couple` keeps them (a's leaves, then b's)."""
    return R.flat_ref(net_a, sa) + R.flat_ref(net_b, sb)


def _flat_one_net(c, sc, net_a, net_b):
    """`couple_net`'s state in the library's order for the coupled pair: a's leaves, b's leaves, then the relays
    (an inbox's [x, r] in a's or b's place)."""
    off = len(net_a.nodes)
    out = []
    relays = [w for w, nd in enumerate(c.nodes) if nd[0] == "relay"]
    ib = net_a.inboxes() + [w + off for w in net_b.inboxes()]
    at = dict(zip(ib, relays))
    for shift, net in ((0, net_a), (off, net_b)):
        for w in net.leaves():
            nd = net.nodes[w]
            if nd[0] == "inbox":
                out.extend(sc[at[w + shift]])
            elif w + shift in sc:
                v = sc[w + shift]
                out.extend(v if isinstance(v, list) else [v])
    return out


def _check_pair(net_a, net_b, a_to_b, b_to_a, steps, tol=1e-13, library=True):
    """`couple_by_hand` == `couple_net` by `ref_step` exactly; both == the library's `couple` and the library's
    run of `couple_net`, states (values and the inboxes' stored covectors) to `tol`."""
    sa, sb = net_a.init_state(), net_b.init_state()
    c = R.couple_net(net_a, net_b, a_to_b, b_to_a)
    sc = c.init_state()
    hand, one, own = [], [], []
    for _ in range(steps):
        sa, sb = R.couple_by_hand(net_a, net_b, a_to_b, b_to_a, sa, sb)
        sc = R.ref_step(c, sc)
        hand.append(_flat_coupled(net_a, net_b, sa, sb))
        one.append(_flat_one_net(c, sc, net_a, net_b))
        own.append(R.flat_ref(c, sc))
    for h, o in zip(hand, one):
        assert h == pytest.approx(o, abs=tol, rel=tol)
    if library:
        L = R.couple(net_a, net_b, a_to_b, b_to_a)
        lib = R.arrangement_run_jit(L, steps, check=1, every=1)
        for h, l in zip(hand, lib):
            assert l == pytest.approx(h, abs=tol, rel=tol)
        lib1 = R.library_run_jit(c, steps, check=0, every=1)   # the one net's own order: leaves, relays last
        for o, l in zip(own, lib1):
            assert l == pytest.approx(o, abs=tol, rel=tol)
    return hand, one


def test_A2a_two_tanks_split_with_a_port():
    A, B, a2b, b2a = S.tank_pair(3.0, 1.0, 0.05, 1.0)
    hand, one = _check_pair(A, B, a2b, b2a, 200)
    levels = [(round(h[0], 6), round(h[3], 6)) for h in hand[:3]]
    print(f"\nA2a relayed levels after 1,2,3 steps: {levels}")
    assert levels == [(2.9, 1.0), (2.805, 1.1), (2.71475, 1.195)]
    direct = R.two_tanks(3.0, 1.0, 0.05)
    st = direct.init_state()
    dl = []
    for _ in range(3):
        st = R.ref_step(direct, st)
        dl.append((round(st[0], 6), round(st[1], 6)))
    print(f"A2a direct levels after 1,2,3 steps: {dl}")
    assert dl == [(2.9, 1.1), (2.81, 1.19), (2.729, 1.271)]
    conserved = [h[0] + h[3] + h[2] for h in hand]   # v1 + v2 + r (the flat order is v1, x, r, v2)
    assert max(abs(s - 4.0) for s in conserved) < 1e-12
    assert abs(hand[99][0] - 2) < 1e-5 and abs(hand[99][3] - 2) < 1e-5
    assert abs(hand[199][0] - 2) < 1e-6 and abs(hand[199][3] - 2) < 1e-6
    print(f"A2a v1 + v2 + r stays {conserved[0]:.12f}; levels at 100: ({hand[99][0]:.7f}, {hand[99][3]:.7f}), "
          f"at 200: ({hand[199][0]:.8f}, {hand[199][3]:.8f})")


def test_A2a_two_tanks_pair_exact_in_fractions():
    A, B, a2b, b2a = S.tank_pair()
    sa, sb = A.init_state(), B.init_state()
    for _ in range(3):
        sa, sb = R.couple_by_hand(A, B, a2b, b2a, sa, sb)
    assert (sa[0], sb[0], sa[1]) == (F(10859, 4000), F(239, 200), [F(11, 10), F(361, 4000)])


def _random_port_net(rng, n_in, n_out):
    """A random net with `n_in` inboxes and `n_out` outputs over the enlarged kit, its values kept tame."""
    n = R.Net()
    wires, stateful = [], []
    for _ in range(rng.randint(1, 2)):
        w = n.register(rng.uniform(-1, 1)); wires.append(w); stateful.append(w)
    for _ in range(rng.randint(0, 1)):
        w = n.drum([rng.uniform(-1, 1) for _ in range(rng.randint(1, 3))]); wires.append(w); stateful.append(w)
    if rng.random() < 0.7:
        q = n.market([rng.uniform(-1, 1) for _ in range(3)]); wires += q; stateful += q
    for _ in range(n_in):
        w = n.inbox(rng.uniform(-1, 1)); wires.append(w); stateful.append(w)
    for _ in range(rng.randint(0, 2)):
        wires.append(n.const(rng.uniform(-1, 1)))
    one = n.const(1.0)
    for _ in range(rng.randint(1, 5)):
        i, j = rng.choice(wires), rng.choice(wires)
        r = rng.random()
        if r < 0.35:
            wires.append(n.add(i, j))
        elif r < 0.7:
            wires.append(n.mul(i, j))
        elif r < 0.85:
            wires.append(n.exp(n.mul(n.const(0.1), i)))
        else:
            wires.append(n.recip(n.add(one, n.mul(i, i))))
    for _ in range(rng.randint(0, 2)):
        if rng.random() < 0.5:
            n.cost(n.mul(n.const(0.1), rng.choice(wires)))
        else:
            n.push(n.mul(n.const(0.1), rng.choice(wires)), rng.choice(stateful))
    for _ in range(n_out):
        n.out(rng.choice(wires))
    return n


def test_A2c_random_pairs():
    rng = random.Random(3819)
    for k in range(8):
        a = _random_port_net(rng, 1, 2)
        b = _random_port_net(rng, 2, 1)
        _check_pair(a, b, [0, 1], [0], 4, tol=1e-12, library=(k < 3))
    print("\nA2c eight random pairs: by hand == one net exactly; the first three also through the library")


def test_A2d_random_nets_over_the_enlarged_kit_translate_exactly():
    """The translation on closed random nets with markets, relays, exp and recip: `machine_run` decodes to
    `ref_step` to 1e-12 for three steps."""
    rng = random.Random(20261008)
    kinds = set()
    for _ in range(100):
        a = _random_port_net(rng, 1, 1)
        b = _random_port_net(rng, 1, 1)
        net = R.couple_net(a, b, [0], [0])
        prog = R.compile_net(net, zero=0.0, one=1.0)
        st = _floats(net.init_state())
        data, acc = prog.encode(st), 0.0
        for _ in range(3):
            st = R.ref_step(net, st)
            acc = R.machine_run(prog, data, acc, 1)
            dec = prog.decode(data)
            for w, v in st.items():
                assert np.asarray(dec[w]) == pytest.approx(np.asarray(v), abs=1e-12, rel=1e-12)
        kinds |= {nd[0] for nd in net.nodes}
    assert {"market", "relay", "exp", "recip", "slot"} <= kinds
    print(f"\nA2d 100 random coupled nets, node kinds {sorted(kinds)}: the runner's arithmetic decodes to ref_step")


# ---------------------------------------------------------------------------
# A3, A4: the figurine, through the library and through the runner.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def figurine_runs():
    fig, wires, world, a2b, b2a = S.figurine_and_world()
    beliefs, guesses, g = S.direct_bayes(S.LAMS, S.PRIOR, S.OUTCOMES)
    steps = len(S.OUTCOMES) + 1
    sa, sb = fig.init_state(), world.init_state()
    hand = []
    for _ in range(steps):
        sa, sb = R.couple_by_hand(fig, world, a2b, b2a, sa, sb)
        hand.append((dict(sa), dict(sb)))
    L = R.couple(fig, world, a2b, b2a)
    lib = R.arrangement_run_jit(L, steps, check=1, every=1)
    return dict(fig=fig, wires=wires, world=world, a2b=a2b, b2a=b2a, beliefs=beliefs, guesses=guesses, g=g,
                hand=hand, lib=lib, steps=steps)


def test_A3_the_market_is_bayes(figurine_runs):
    r = figurine_runs
    mk = \
        [w for w, nd in enumerate(r["fig"].nodes) if nd[0] == "market"][0]
    worst_hand = worst_lib = 0.0
    returned = []
    for t in range(r["steps"]):
        want = np.array(r["beliefs"][t])     # after step t+1 the belief has seen outcomes 1..t
        sa, _ = r["hand"][t]
        q_hand = _softmax(sa[mk])
        q_lib = _softmax(r["lib"][t][:5])
        worst_hand = max(worst_hand, np.abs(q_hand - want).max())
        worst_lib = max(worst_lib, np.abs(q_lib - want).max())
        returned.append(sa[r["wires"]["x"]][1])
        assert abs(r["lib"][t][6] - sa[r["wires"]["x"]][1]) < 1e-12   # the library's stored covector too
    print(f"\nA3 belief vs direct Bayes: worst by hand {worst_hand:.1e}, through the library {worst_lib:.1e}")
    print(f"A3 belief after six outcomes: {np.round(_softmax(r['lib'][-1][:5]), 4)}")
    print(f"A3 covector returned on the outcome's port each step: {np.round(returned, 4)}")
    assert worst_hand < 1e-12 and worst_lib < 1e-12
    assert _softmax(r["lib"][-1][:5]) == pytest.approx([0, 0.9546, 0.0454, 0, 0], abs=5e-5)
    # the covector stored after step t+1 is G(t) - x_t, the guess before round t minus round t's reading, which the
    # cost received at step t+1 (the inbox held x_t then); it is published at step t+2 onto the cell presented then
    G, xs = r["guesses"], S.OUTCOMES
    want_ret = [0.0] + [G[t] - xs[t] for t in range(len(xs))]
    assert returned == pytest.approx(want_ret, abs=1e-12)
    assert np.round(G, 3).tolist() == [5.068, 5.829, 5.748, 6.113, 6.204, 6.234]
    # the pushed-on drum cell: after seven steps the outcome drum has rotated once; its cell for x_3 holds
    # x_3 + G(1) - x_1, round 1's pushback, which arrived two steps after x_1 was presented
    drum = r["lib"][-1][9:15]
    print(f"A3 the world's outcome drum after seven steps: {np.round(drum, 4)}")
    assert drum[1] == pytest.approx(xs[2] + G[0] - xs[0], abs=1e-12)
    assert drum[0] == pytest.approx(xs[1], abs=1e-12)   # x_2 untouched: round 0's flag was 0


def test_A4_the_runner_runs_the_figurine(figurine_runs):
    r = figurine_runs
    c = R.couple_net(r["fig"], r["world"], r["a2b"], r["b2a"])
    mk = [w for w, nd in enumerate(c.nodes) if nd[0] == "market"][0]
    prog = R.compile_net(c, zero=0.0, one=1.0)
    print(f"\nA4 the translation: {prog.L} cells, {prog.Lp} runner steps a round, {prog.n_slp} instructions, "
          f"{len(prog.events)} events")
    sc = c.init_state()
    data, a = prog.encode(sc), 0.0
    worst_ref = worst_direct = 0.0
    for t in range(r["steps"]):
        sc = R.ref_step(c, sc)
        a = R.machine_run(prog, data, a, 1)
        dec = prog.decode(data)
        worst_ref = max(worst_ref, np.abs(np.array(dec[mk]) - np.array(sc[mk])).max())
        worst_direct = max(worst_direct, np.abs(_softmax(dec[mk]) - np.array(r["beliefs"][t])).max())
    print(f"A4 runner vs ref_step heights {worst_ref:.1e}; runner belief vs direct Bayes {worst_direct:.1e}")
    assert worst_ref < 1e-12 and worst_direct < 1e-10
    assert not (500 <= prog.Lp <= 5000), "the prediction's count was wrong; see RESULTS"


@pytest.mark.skipif(not __import__("os").environ.get("RUN_SLOW"),
                    reason="one round of the figurine's program through the library takes about five minutes")
def test_A4c_the_runner_as_an_arrangement_at_the_figurines_settings():
    """P43 (b) job 8 (Claude Opus 5.5, de71554a, 2026-10-10): A4 checks the runner on the figurine's program only
    by `machine_run` (by hand); here the runner itself, `machine_net` denoted in the library and stepped by its own
    forcing and integrator (compiled), runs one whole round of that program, 49,946 steps, and its data drum equals
    `machine_run`'s exactly (0.0 on 2026-10-10, 314 s). Set RUN_SLOW=1 to run it."""
    fig, wires, world, a2b, b2a = S.figurine_and_world()
    c = R.couple_net(fig, world, a2b, b2a)
    prog = R.compile_net(c, zero=0.0, one=1.0)
    data = [float(x) for x in prog.encode(c.init_state())]
    M = R.machine_net(prog.flags(), data, a0=0.0, minus_one=-1.0)
    traj = R.library_run_jit(M, prog.Lp, check=0, every=prog.Lp)
    hand = list(data)
    R.machine_run(prog, hand, 0.0, 1)
    assert np.abs(np.asarray(traj[0][6 * prog.Lp: 6 * prog.Lp + prog.L]) - np.asarray(hand)).max() == 0.0


def _small_full_kit_net():
    """A small closed net using every new kind: a market over two guesses (one through exp, one through recip),
    paid by a world drum through a relay, and a register pushed by a relayed value."""
    fig = R.Net()
    q = fig.market([0.0, 0.0])
    g = [fig.exp(fig.const(0.5)), fig.recip(fig.const(0.8))]      # guesses e^0.5 and 1/0.8
    x = fig.inbox(0.0)
    neg, half = fig.const(-1.0), fig.const(0.5)
    k = None
    for qi, gi in zip(q, g):
        d = fig.add(x, fig.mul(neg, gi))
        t = fig.mul(qi, fig.mul(half, fig.mul(d, d)))
        k = t if k is None else fig.add(k, t)
    fig.cost(k)
    y = fig.register(0.3)
    fig.push(fig.mul(fig.const(0.1), x), y)
    fig.out(fig.add(fig.mul(q[0], g[0]), fig.mul(q[1], g[1])))
    world = R.Net()
    world.out(world.drum([1.5, 1.2, 1.8, 1.4]))
    return R.couple_net(fig, world, [], [0])


def test_A4b_the_six_drum_runner_is_an_arrangement_over_the_kit():
    """One pass of the six-drum runner, taken through the library's own composition, decodes to the net's step:
    on a small net using every new kind (the figurine's program is 49,946 steps a round, too long for the
    library's uncompiled check in the time limit; its runner is checked by `machine_run` in test A4)."""
    c = _small_full_kit_net()
    assert {nd[0] for nd in c.nodes} >= {"market", "slot", "relay", "exp", "recip"}
    prog = R.compile_net(c, zero=0.0, one=1.0)
    st = c.init_state()
    data = [float(x) for x in prog.encode(st)]
    M = R.machine_net(prog.flags(), data, a0=0.0, minus_one=-1.0)
    assert {nd[0] for nd in M.nodes} <= {"drum", "register", "const", "add", "mul", "exp", "recip"}
    assert len([nd for nd in M.nodes if nd[0] == "drum"]) == 7
    Lp = prog.Lp
    traj = R.library_run_jit(M, 2 * Lp, check=2, every=Lp)
    worst = 0.0
    for k in range(2):
        st = R.ref_step(c, st)
        decoded = prog.decode(traj[k][6 * Lp: 6 * Lp + prog.L])
        for w, v in st.items():
            worst = max(worst, np.abs(np.asarray(decoded[w]) - np.asarray(v)).max())
    print(f"\nA4b the six-drum runner ({len(M.nodes)} nodes, {Lp} steps a pass, {prog.L} cells) through the "
          f"library, two passes, decodes to the net's steps to {worst:.1e}")
    assert worst < 1e-10
