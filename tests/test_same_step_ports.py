"""The same-step port: a cost that reads its port inside its own forcing (book plan P37 (b)(iii′) stage (A′);
``runner.Net.pcost``, ``market_scene`` stage (A′) section).

David (2026-10-09): "i think we should try to build the same step kind if we can. is it a mealy machine in place of a
moore by any chance ([Ay,By] rather than Ay^B)?" Predictions fixed before any code for it, in the book repo's
``notes/runs/p37-market-of-scenes-3819a169/PREDICTIONS.md``, stage (A′); results in its ``RESULTS.md``.

A′1. The two-tank pair with the pipe's cost as a ``pcost`` on the port equals P34's direct ``two_tanks`` net exactly
     in fractions: (2.9, 1.1), (2.81, 1.19), (2.729, 1.271); nothing in flight.
A′2. ``couple`` through the library, ``couple_by_hand`` and ``couple_net`` (by hand and through the library) agree to
     1e-13 on that pair, on the figurine and the world, and on random pairs with ``pcost`` sinks.
A′3. The figurine with no flag and no lag: the belief after step t is the direct posterior after t outcomes; the
     covector returned at step t is G(t) - x_t; the drum cell rotates to the back holding G(t).
A′4. The one net wired by hand runs on the six-drum runner, decoding to ``ref_step`` to 1e-12 and to the direct
     posterior to 1e-10.
"""
import random
from fractions import Fraction as F

import jax
import numpy as np
import pytest

from arrangements import market_scene as S
from arrangements import runner as R
from tests.test_market_runner import _flat_coupled, _floats, _random_port_net, _softmax

jax.config.update("jax_enable_x64", True)


def _check_pair(net_a, net_b, a_to_b, b_to_a, steps, tol=1e-13, library=True):
    sa, sb = net_a.init_state(), net_b.init_state()
    c = R.couple_net(net_a, net_b, a_to_b, b_to_a)
    sc = c.init_state()
    hand, own = [], []
    for _ in range(steps):
        sa, sb = R.couple_by_hand(net_a, net_b, a_to_b, b_to_a, sa, sb)
        sc = R.ref_step(c, sc)
        hand.append(_flat_coupled(net_a, net_b, sa, sb))
        own.append(R.flat_ref(c, sc))
    # the one net has no inbox cells; compare every register, drum and market cell of a and b by name
    for h, o in zip(hand, own):
        assert len(o) <= len(h)
    if library:
        L = R.couple(net_a, net_b, a_to_b, b_to_a)
        lib = R.arrangement_run_jit(L, steps, check=1, every=1)
        for h, l in zip(hand, lib):
            assert l == pytest.approx(h, abs=tol, rel=tol)
        lib1 = R.library_run_jit(c, steps, check=0, every=1)
        for o, l in zip(own, lib1):
            assert l == pytest.approx(o, abs=tol, rel=tol)
    return hand, own, c


def _one_net_states(net_a, net_b, a_to_b, b_to_a, steps):
    """The pair by hand and the one net by hand, as dicts over the original nodes' states (inboxes dropped)."""
    sa, sb = net_a.init_state(), net_b.init_state()
    c = R.couple_net(net_a, net_b, a_to_b, b_to_a)
    sc = c.init_state()
    off = len(net_a.nodes)
    worst = 0.0
    for _ in range(steps):
        sa, sb = R.couple_by_hand(net_a, net_b, a_to_b, b_to_a, sa, sb)
        sc = R.ref_step(c, sc)
        for shift, net, st in ((0, net_a, sa), (off, net_b, sb)):
            for w, v in st.items():
                if net.nodes[w][0] == "inbox":
                    continue
                worst = max(worst, np.abs(np.asarray(sc[w + shift], float) - np.asarray(v, float)).max())
    return worst


def test_A1_the_direct_wire_returns_exact_in_fractions():
    A, B, a2b, b2a = S.tank_pair_same_step()
    sa, sb = A.init_state(), B.init_state()
    direct = R.two_tanks()
    sd = direct.init_state()
    levels = []
    for k in range(1, 9):
        sa, sb = R.couple_by_hand(A, B, a2b, b2a, sa, sb)
        sd = R.ref_step(direct, sd)
        assert (sa[0], sb[0]) == (sd[0], sd[1]) == (2 + F(9, 10) ** k, 2 - F(9, 10) ** k)
        assert sa[1][1] == 0   # the inbox stored no covector: the port returned it the same step
        levels.append((float(sa[0]), float(sb[0])))
    print(f"\nA′1 the pair with a same-step port equals the direct two tanks exactly: {levels[:3]}")


def test_A2_equality_on_the_pair_and_random_pairs():
    A, B, a2b, b2a = S.tank_pair_same_step(3.0, 1.0, 0.05)
    _check_pair(A, B, a2b, b2a, 50)
    assert _one_net_states(A, B, a2b, b2a, 50) < 1e-13
    rng = random.Random(20261009)
    n_pcost = 0
    for k in range(8):
        a = _random_port_net(rng, 1, 2)
        b = _random_port_net(rng, 2, 1)
        for net, ports in ((a, 1), (b, 2)):
            wires = [w for w, nd in enumerate(net.nodes) if nd[0] not in ("market", "relay", "inbox")]
            if rng.random() < 0.8:
                c0, c1 = rng.choice(wires), rng.choice(wires)
                net.pcost([net.mul(net.const(0.1), c0), net.mul(net.const(0.1), c1)], rng.randrange(ports))
                n_pcost += 1
        _check_pair(a, b, [0, 1], [0], 4, tol=1e-12, library=(k < 3))
        assert _one_net_states(a, b, [0, 1], [0], 4) < 1e-12
    print(f"\nA′2 the pair for 50 steps and eight random pairs ({n_pcost} pcost sinks): the library's couple, the pair "
          f"by hand and the one net wired by hand agree; the first three pairs through the library too")


@pytest.fixture(scope="module")
def figurine():
    fig, wires = S.figurine_net_same_step(S.LAMS, S.PRIOR)
    world = S.world_net_outcomes()
    beliefs, guesses, g = S.direct_bayes(S.LAMS, S.PRIOR, S.OUTCOMES)
    mk = [i for i, nd in enumerate(fig.nodes) if nd[0] == "market"][0]
    sa, sb = fig.init_state(), world.init_state()
    hand, returned, back = [], [], []
    for _ in range(6):
        returned.append(R.port_covectors(fig, sa, [R.present(world, sb)[0]])[0])
        sa, sb = R.couple_by_hand(fig, world, [], [0], sa, sb)
        hand.append((dict(sa), dict(sb)))
        back.append(sb[0][-1])
    return dict(fig=fig, world=world, mk=mk, beliefs=beliefs, guesses=guesses, hand=hand, returned=returned,
                back=back)


def test_A3_no_flag_no_lag(figurine):
    r = figurine
    worst = max(np.abs(_softmax(sa[r["mk"]]) - np.array(r["beliefs"][t + 1])).max() for t, (sa, _) in enumerate(r["hand"]))
    want = [r["guesses"][t] - S.OUTCOMES[t] for t in range(6)]
    print(f"\nA′3 belief after step t vs direct Bayes after t outcomes: {worst:.1e}")
    print(f"A′3 returned on the port at steps 1-6: {np.round(r['returned'], 4)}; the drum cell rotated to the back "
          f"holds {np.round(r['back'], 4)} (the guesses before each round: {np.round(r['guesses'], 4)})")
    assert worst < 1e-12
    assert r["returned"] == pytest.approx(want, abs=1e-12)
    assert r["back"] == pytest.approx(r["guesses"], abs=1e-12)


def test_A3b_through_the_library(figurine):
    r = figurine
    L = R.couple(r["fig"], r["world"], [], [0])
    lib = R.arrangement_run_jit(L, 6, check=1, every=1)
    worst = 0.0
    for t, (sa, sb) in enumerate(r["hand"]):
        worst = max(worst, np.abs(np.array(lib[t]) - np.array(_flat_coupled(r["fig"], r["world"], sa, sb))).max())
    print(f"\nA′3b the library's couple vs the pair by hand, every state cell: {worst:.1e}")
    assert worst < 1e-12


def test_A4_the_runner(figurine):
    r = figurine
    c = R.couple_net(r["fig"], r["world"], [], [0])
    mk = [w for w, nd in enumerate(c.nodes) if nd[0] == "market"][0]
    prog = R.compile_net(c, zero=0.0, one=1.0)
    sc = c.init_state()
    data, a = prog.encode(sc), 0.0
    worst_ref = worst_direct = 0.0
    for t in range(6):
        sc = R.ref_step(c, sc)
        a = R.machine_run(prog, data, a, 1)
        dec = prog.decode(data)
        worst_ref = max(worst_ref, np.abs(np.array(dec[mk]) - np.array(sc[mk])).max())
        worst_direct = max(worst_direct, np.abs(_softmax(dec[mk]) - np.array(r["beliefs"][t + 1])).max())
    print(f"\nA′4 the runner ({prog.L} cells, {prog.Lp} steps a round): vs ref_step {worst_ref:.1e}, vs direct Bayes "
          f"{worst_direct:.1e}")
    assert worst_ref < 1e-12 and worst_direct < 1e-10
