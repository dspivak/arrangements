"""One runner for every closed term over the kit and a tape (arrangements/tape_runner.py; book plan P34 (i)).

What is checked, and against what:

1. The tape's update by hand (`runner._tape_step`): a write then a move; a fractional move blends.
2. The tape is a kit part in the library: a net with tapes run through `denote` and `Phi` agrees
   with `ref_step`, which shares no code with the library.
3. The translation is right: U's fast step (`tape_runner_run`) on the translated program, decoded
   after each pass, equals `ref_step` of the guest exactly, in fractions: two tanks, the turner,
   the crossed pair, the trail, a tape moved by fractions, two tapes and a drum, and random nets
   with tapes.
4. The fast step is U's own: at every phase boundary of two guest steps it equals `ref_step` of
   `tape_runner_net`, exactly. U is one net: its boxes and pushes are the same for every guest.
5. U is an arrangement over the kit and the tape: its term, run through the library on tape
   windows wide enough for the horizon, agrees with `ref_step` of U.
6. U handed its own description and a state holding two tanks runs three steps of itself exactly.

Not checked here: the long run at depth two (U running U running two tanks through a whole tank
step; `notes/runs/p34-tape-95e16660/` in the book repo); a proof for every closed term.
"""
from fractions import Fraction as F
import random

import pytest

from arrangements import runner as R
from arrangements import tape_runner as T

GUESTS = [("two_tanks", R.two_tanks), ("turner", R.turner), ("crossed", R.crossed),
          ("trail", T.trail), ("blend", T.blend), ("two_tapes", T.two_tapes), ("busy_beaver_2", T.busy_beaver_2)]


def test_tape_step_by_hand():
    s = {0: 2, 1: -1, -2: 3}
    assert R._tape_step(s, 5, 0, 0) == {0: 7, 1: -1, -2: 3}
    assert R._tape_step(s, 0, 0, 1) == {-1: 2, 0: -1, -3: 3}       # the head moves right
    assert R._tape_step(s, 0, 1, 0) == {1: 2, 2: -1, -1: 3}        # the head moves left
    h = R._tape_step(s, 0, F(1, 2), F(1, 2))                        # each cell the mean of its neighbors
    assert h == {-1: F(1, 2) * 2 + F(1, 2) * 3, 0: F(-1, 2), 1: 1, 2: F(-1, 2), -3: F(3, 2)}


def test_tapes_through_library_agree_with_by_hand_stepper():
    for mk in (T.trail, T.blend, T.two_tapes):
        net = mk()
        for i, nd in enumerate(net.nodes):
            if nd[0] == "tape":
                net.nodes[i] = ("tape", nd[1], 21)
        lib = R.library_run(net, 5)
        st = net.init_state()
        for k in range(5):
            st = R.ref_step(net, st)
            assert lib[k] == pytest.approx([float(x) for x in R.flat_ref(net, st)], abs=1e-12)


@pytest.mark.parametrize("name,mk", GUESTS)
def test_tape_runner_runs_guests_exact(name, mk):
    net = mk()
    prog = T.compile_tape(net, zero=F(0), one=F(1))
    st = net.init_state()
    u = T.runner_init(prog, st)
    for _ in range(8):
        T.tape_runner_run(u, 1)
        st = R.ref_step(net, st)
        assert prog.decode(u.tapes["V"]) == st
        assert u.phase() == "FETCH" and (u.hp, u.hv, u.hb) == (0, 0, 0)


def _random_tape_net(rng):
    """A random closed net over the kit and the tape: registers, drums, tapes, constants, adds,
    muls, costs, pushes; a tape's levers pushed by fractions or by wires."""
    n = R.Net()
    wires, stateful = [], []
    for _ in range(rng.randint(0, 2)):
        w = n.register(F(rng.randint(-3, 3), rng.randint(1, 4))); wires.append(w); stateful.append(w)
    for _ in range(rng.randint(0, 1)):
        w = n.drum([F(rng.randint(-2, 2), rng.randint(1, 3)) for _ in range(rng.randint(1, 3))])
        wires.append(w); stateful.append(w)
    for _ in range(rng.randint(1, 2)):
        cells = {rng.randint(-3, 3): F(rng.randint(-2, 2), rng.randint(1, 3)) for _ in range(rng.randint(0, 3))}
        f, l, r = n.tape(cells)
        wires.append(f); stateful += [f, l, r]
    for _ in range(rng.randint(0, 2)):
        wires.append(n.const(F(rng.randint(-2, 2), rng.randint(1, 4))))
    for _ in range(rng.randint(0, 5)):
        i, j = rng.choice(wires), rng.choice(wires)
        wires.append(n.add(i, j) if rng.random() < 0.5 else n.mul(i, j))
    for _ in range(rng.randint(1, 4)):
        if rng.random() < 0.3:
            n.cost(rng.choice(wires))
        else:
            n.push(rng.choice(wires), rng.choice(stateful))
    return n


def test_tape_runner_exact_on_random_nets():
    rng = random.Random(20261015)
    kinds = set()
    for _ in range(200):
        net = _random_tape_net(rng)
        prog = T.compile_tape(net, zero=F(0), one=F(1))
        st = net.init_state()
        u = T.runner_init(prog, st)
        for _ in range(3):
            T.tape_runner_run(u, 1)
            st = R.ref_step(net, st)
            assert prog.decode(u.tapes["V"]) == st
        kinds |= {nd[0] for nd in net.nodes} | {s[0] for s in net.sinks}
    assert kinds == {"register", "drum", "tape", "slot", "const", "add", "mul", "cost", "push"}


def _phase_marks(U, prog, gstate, passes):
    u = T.runner_init(prog, gstate)
    marks = []
    T.tape_runner_run(u, passes, boundary=lambda v: marks.append((v.steps, T.to_net_state(U, v))))
    return marks


@pytest.mark.parametrize("name,mk", [("crossed", R.crossed), ("trail", T.trail), ("blend", T.blend),
                                     ("two_tapes", T.two_tapes)])
def test_fast_step_is_U_stepped_by_hand(name, mk):
    net = mk()
    prog = T.compile_tape(net)
    U = T.tape_runner_net(prog, net.init_state())
    assert {nd[0] for nd in U.nodes} == {"tape", "slot", "register", "const", "add", "mul"}
    assert {s[0] for s in U.sinks} == {"push"}
    st, k = U.init_state(), 0
    for steps, want in _phase_marks(U, prog, net.init_state(), 2):
        while k < steps:
            st = R.ref_step(U, st)
            k += 1
        assert st == want


def test_U_is_one_net_for_every_guest():
    def shape(U):
        return tuple(nd[0] if nd[0] in ("tape", "register") else nd for nd in U.nodes), tuple(U.sinks)

    shapes = {shape(T.tape_runner_net())}
    for _name, mk in GUESTS:
        net = mk()
        shapes.add(shape(T.tape_runner_net(T.compile_tape(net), net.init_state())))
    assert len(shapes) == 1


def test_U_through_library():
    """U's term, its tapes on windows of 101 cells, run through the library for two guest steps (342
    of U's steps). About 5 minutes, nearly all JAX compiling the step of a term 178 boxes deep; the
    compiled step is the closed arrangement's own (`runner.library_run_jit`), not also stepped by `Phi`."""
    net = R.crossed()
    prog = T.compile_tape(net)
    U = T.tape_runner_net(prog, net.init_state(), window=101)
    H = 342
    traj = R.library_run_jit(U, H, check=0, every=1)
    st = U.init_state()
    for k in range(H):
        st = R.ref_step(U, st)
        assert traj[k] == pytest.approx([float(x) for x in R.flat_ref(U, st)], abs=1e-12)


def test_U_runs_its_own_description_exact():
    """U handed its own description, holding two tanks: three steps of the inner U, exact."""
    net = R.two_tanks()
    p_in = T.compile_tape(net)
    inner = T.tape_runner_net(p_in, net.init_state())
    p_out = T.compile_tape(inner)
    gst = inner.init_state()
    u = T.runner_init(p_out, gst)
    for _ in range(3):
        T.tape_runner_run(u, 1)
        gst = R.ref_step(inner, gst)
        assert p_out.decode(u.tapes["V"]) == gst


def test_a_turing_machine_is_a_term_and_U_runs_it():
    """Rado's two-state busy beaver: halts after 6 steps with four 1s, by the net's own step and by U."""
    net = T.busy_beaver_2()
    tape = next(w for w, nd in enumerate(net.nodes) if nd[0] == "tape")
    halt = [w for w, nd in enumerate(net.nodes) if nd[0] == "register"][2]
    st = net.init_state()
    for _ in range(6):
        st = R.ref_step(net, st)
    assert st[tape] == {-2: 1, -1: 1, 0: 1, 1: 1} and st[halt] == 1
    assert R.ref_step(net, st) == st
