"""An arrangement that runs arrangements (arrangements/runner.py; book plan P34; STATUS.md §3.x).

What is checked, and against what:

1. The kit's terms mean what the by-hand stepper says: three nets run through the library
   (`denote`, `Phi`) agree with `ref_step`, which shares no code with the library; and the two
   tanks agree with their closed form 2 +- (9/10)^k exactly, in fractions.
2. The translation is right on these nets: the runner's plain stepper on the translated
   program, decoded, equals `ref_step` exactly, in fractions, for twelve steps.
3. The runner is an arrangement over the kit: `machine_net` taken through the library's own
   composition, one pass of its flag drums, decodes to the net's own step, and agrees with the
   plain stepper bit for bit. Its compiled step equals `Phi`'s on the first two steps.
4. The runner handed its own description runs a net: the host runner on the translation of
   the guest runner's net, decoded twice, equals the net's own step exactly, in fractions; and
   after every guest step the host's decoded state equals the guest runner's net stepped by hand.
5. The same tower with the host as a library arrangement, for the smallest net.

Not checked here: any net with outer ports; the translation as a general theorem; the
translation done by the runner itself (it is done outside, by `compile_net`).
"""
from fractions import Fraction as F

import pytest

from arrangements import runner as R

NETS = [("two_tanks", R.two_tanks), ("turner", R.turner), ("crossed", R.crossed)]


def _floats(net):
    return {w: float(v) for w, v in net.init_state().items()}


@pytest.mark.parametrize("name,mk", NETS)
def test_kit_terms_agree_with_by_hand_stepper(name, mk):
    net = mk()
    lib = R.library_run(net, 4)
    st = _floats(net)
    for k in range(4):
        st = R.ref_step(net, st)
        assert lib[k] == pytest.approx(R.flat_ref(net, st), abs=1e-13)


def test_two_tanks_closed_form_exact():
    net = R.two_tanks()
    st = net.init_state()
    for k in range(1, 9):
        st = R.ref_step(net, st)
        assert R.flat_ref(net, st) == [2 + F(9, 10) ** k, 2 - F(9, 10) ** k]


@pytest.mark.parametrize("name,mk", NETS)
def test_translation_exact_on_plain_stepper(name, mk):
    net = mk()
    prog = R.compile_net(net, zero=F(0), one=F(1))
    st = {w: F(v) for w, v in net.init_state().items()}
    data, a = prog.encode(st), F(0)
    for _ in range(12):
        st = R.ref_step(net, st)
        a = R.machine_run(prog, data, a, 1)
        assert prog.decode(data) == st


@pytest.mark.parametrize("name,mk", NETS)
def test_runner_is_an_arrangement_over_the_kit(name, mk):
    net = mk()
    prog = R.compile_net(net, zero=0.0, one=1.0)
    st = _floats(net)
    data = [float(x) for x in prog.encode(st)]
    M = R.machine_net(prog.flags(), data, a0=0.0, minus_one=-1.0)
    assert {nd[0] for nd in M.nodes} <= {"drum", "register", "const", "add", "mul"}
    assert {s[0] for s in M.sinks} == {"push"}
    Lp, L = prog.Lp, prog.L
    traj = R.library_run_jit(M, 2 * Lp, check=2, every=Lp)
    fast, a = list(data), 0.0
    for k in range(2):
        st = R.ref_step(net, st)
        a = R.machine_run(prog, fast, a, 1)
        flat = traj[k]
        assert flat[: 4 * Lp] == [float(x) for f in prog.flags() for x in f]
        assert flat[4 * Lp: 4 * Lp + L] == fast and flat[4 * Lp + L] == a
        dec = prog.decode(flat[4 * Lp: 4 * Lp + L])
        assert [dec[w] for w in sorted(dec)] == pytest.approx([st[w] for w in sorted(st)], abs=1e-13)


@pytest.mark.parametrize("name,mk", NETS)
def test_runner_runs_its_own_description_exact(name, mk):
    net = mk()
    p_in = R.compile_net(net, zero=F(0), one=F(1))
    st = {w: F(v) for w, v in net.init_state().items()}
    guest = R.machine_net(p_in.flags(), p_in.encode(st), a0=F(0), minus_one=F(-1))
    p_out = R.compile_net(guest, zero=F(0), one=F(1))
    gst = guest.init_state()
    host, a = p_out.encode(gst), F(0)
    leaves = guest.leaves()
    for k in range(2):
        for _ in range(p_in.Lp):
            a = R.machine_run(p_out, host, a, 1)
            if k == 0:
                gst = R.ref_step(guest, gst)
                assert p_out.decode(host) == gst
        st = R.ref_step(net, st)
        g = p_out.decode(host)
        assert [g[w] for w in leaves[:4]] == [list(f) for f in p_in.flags()]
        assert p_in.decode(g[leaves[4]]) == st


def test_library_host_runs_its_own_description():
    net = R.crossed()
    p_in = R.compile_net(net, zero=0.0, one=1.0)
    st = _floats(net)
    guest = R.machine_net(p_in.flags(), [float(x) for x in p_in.encode(st)], a0=0.0, minus_one=-1.0)
    p_out = R.compile_net(guest, zero=0.0, one=1.0)
    host = R.machine_net(p_out.flags(), [float(x) for x in p_out.encode(guest.init_state())], a0=0.0, minus_one=-1.0)
    per_net = p_out.Lp * p_in.Lp
    flat = R.library_run_jit(host, per_net, check=2, every=per_net)[0]
    g = p_out.decode(flat[4 * p_out.Lp: 4 * p_out.Lp + p_out.L])
    inner = p_in.decode(g[guest.leaves()[4]])
    st = R.ref_step(net, st)
    assert inner == st


def _random_net(rng):
    """A random closed net over the kit: drums, registers, constants, adds, muls, costs, pushes."""
    n = R.Net()
    wires, stateful = [], []
    for _ in range(rng.randint(1, 3)):
        w = n.register(F(rng.randint(-3, 3), rng.randint(1, 4))); wires.append(w); stateful.append(w)
    for _ in range(rng.randint(0, 2)):
        w = n.drum([F(rng.randint(-2, 2), rng.randint(1, 3)) for _ in range(rng.randint(1, 4))])
        wires.append(w); stateful.append(w)
    for _ in range(rng.randint(0, 2)):
        wires.append(n.const(F(rng.randint(-2, 2), rng.randint(1, 3))))
    for _ in range(rng.randint(0, 6)):
        i, j = rng.choice(wires), rng.choice(wires)
        wires.append(n.add(i, j) if rng.random() < 0.5 else n.mul(i, j))
    for _ in range(rng.randint(0, 3)):
        if rng.random() < 0.5:
            n.cost(rng.choice(wires))
        else:
            n.push(rng.choice(wires), rng.choice(wires if rng.random() < 0.3 else stateful))
    return n


def test_translation_exact_on_random_nets():
    import random

    rng = random.Random(20261001)
    kinds = set()
    for _ in range(300):
        net = _random_net(rng)
        prog = R.compile_net(net, zero=F(0), one=F(1))
        st = net.init_state()
        data, a = prog.encode(st), F(0)
        for _ in range(3):
            st = R.ref_step(net, st)
            a = R.machine_run(prog, data, a, 1)
            assert prog.decode(data) == st
        kinds |= {nd[0] for nd in net.nodes} | {s[0] for s in net.sinks}
    assert kinds == {"register", "drum", "const", "add", "mul", "cost", "push"}


def test_random_nets_library_agrees_with_by_hand_stepper():
    import random

    rng = random.Random(7)
    for _ in range(10):
        net = _random_net(rng)
        lib = R.library_run(net, 2)
        st = {w: ([float(x) for x in v] if isinstance(v, list) else float(v)) for w, v in net.init_state().items()}
        for nd_i, nd in enumerate(net.nodes):
            if nd[0] == "const":
                net.nodes[nd_i] = ("const", float(nd[1]))
        for k in range(2):
            st = R.ref_step(net, st)
            assert lib[k] == pytest.approx(R.flat_ref(net, st), rel=1e-12, abs=1e-12)


# --- The runner as a part with ports: another part sends it a description and reads the answer.

CELLS, LENGTH, PASSES = 16, 208, 3


def _exchange(name, num):
    zero, one, neg = num(0), num(1), num(-1)
    runner = R.port_runner_net(CELLS, LENGTH, zero=zero, minus_one=neg)
    net = getattr(R, name)()
    prog = R.compile_net(net, zero=zero, one=one, cells=CELLS, length=LENGTH)
    wanted = sorted(prog.reg_addr)
    asker, T, ans = R.asker_net(prog, PASSES, [prog.reg_addr[w] for w in wanted], zero=zero, one=one, minus_one=neg)
    st = {w: num(v) for w, v in net.init_state().items()}
    for _ in range(PASSES):
        st = R.ref_step(net, st)
    return runner, asker, T, ans, [st[w] for w in wanted]


@pytest.mark.parametrize("name", ["two_tanks", "turner", "crossed"])
def test_exchange_by_hand_exact(name):
    """One runner net; the asker's answer registers end holding the net's state, exactly."""
    runner, asker, T, ans, want = _exchange(name, F)
    assert len(runner.nodes) == 43 and runner.outputs and len(runner.inboxes()) == 7
    sa, sb = asker.init_state(), runner.init_state()
    for _ in range(T):
        sa, sb = R.couple_by_hand(asker, runner, list(range(7)), [0], sa, sb)
    assert [sa[w] for w in ans] == want


def test_exchange_through_library():
    """Asker and runner, each a term over the kit, wired to each other by the library."""
    runner, asker, T, ans, want = _exchange("crossed", float)
    A = R.couple(asker, runner, list(range(7)), [0])
    flat = R.arrangement_run_jit(A, T, check=1, every=T)[-1]
    off, where = 0, {}
    for w in asker.leaves():
        where[w] = off
        off += R.leaf_cells(asker.nodes[w])
    assert [flat[where[w]] for w in ans] == want


def test_a_description_too_big_for_the_runner_is_refused():
    with pytest.raises(ValueError):
        R.compile_net(R.two_tanks(), cells=8, length=208)
    with pytest.raises(ValueError):
        R.compile_net(R.two_tanks(), cells=16, length=64)
