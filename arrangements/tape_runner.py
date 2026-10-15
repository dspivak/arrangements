"""One runner for every closed term over the kit and a tape (book plan P34 (i)).

P34 (d)'s runners are a family: the runner of each size runs the terms that fit its data drum, and
none fits its own translation. Here one part is added to the kit, a TAPE (`runner.Net.tape`): its
state the finitely supported real sequences, its configuration R^3, the cell under the head and
two levers, so a covector (xw, xl, xr) writes xw into the cell and then moves the tape by xl, xr
(at 0 or 1, one place or none; at a fraction, a blend of the shifted and unshifted tape). Ch 3's
integrator asks a cotangent only of the configuration, so a state of no fixed dimension is allowed
(the design, `notes/runs/p34-tape-7d46b5bb/DESIGN.md` §1, in the book repo).

THE RUNNER U (`tape_runner_net`) is one closed net over the kit and the tape: 17 tapes, 34 registers,
two constants, boxes and pushes, the same for every guest; only its initial state differs.

* The PROGRAM bundle, moved together: one flag tape per instruction (LOAD, STORE, ADD, MUL, XCHG,
  SETL, SETR, SWEEP, END), a digit tape `DIG` for unary numbers, a start mark `Z`.
* The DATA bundle, moved together: `V` the guest's numbers, `D` scratch, `H` the home mark, `B`
  the block starts, `EL`, `ER` the two ends of the extent. `B` alone also moves by itself.
* Registers: 15 one-hot phases, 7 latched instructions, the accumulator `a`, the guest tape's two
  moves `rL`, `rR`, and two carries `c`, `e`.

Every register, write and move is a push whose amount is a polynomial in the registers and the
cells under the heads (`_build`, one block per phase), and the control branches only on 0/1 marks
it reads, never on the guest's numbers, which is why polynomials suffice.

THE DESCRIPTION (`compile_tape`): a guest's straight-line program as in `runner.compile_net` (its
forward values, its backward pass, its updates), each instruction an op cell then its address in
unary; then for each guest tape SETL, SETR (its two moves) and SWEEP (its track); then END. Its
data: blocks of W cells on `V`; guest tape j's cell i at block i, track j; every other number at
block 0, track j >= the number of tapes. One guest step is one pass from the program's start to
END's rewind:

* an address op walks `V` out by its address, acts as (d)'s runner acts on the front cell, walks
  home to `H`;
* SWEEP t moves `B` alone left by t, so its 1s fall on track t, then makes two passes over the
  extent: right to left, each track-t cell becomes `rR * (old right neighbor) + (1 - rL - rR) * old`
  and its old value is saved on `D`; left to right, `rL * (old left neighbor)` is added from `D`;
  then home, and `B` back;
* END widens the extent by a block at each end and rewinds the program.

This departs from the design's SWEEP (two delay lines held back by W and 2W): two passes and one
scratch tape need no relative shift but `B`'s (Claude Opus 5.5, 95e16660, 2026-10-15).

`tape_runner_run` is U's step done a phase at a time, skipping the steps that only walk; it is
checked against `runner.ref_step` of `tape_runner_net` at every phase boundary of short stretches.
All exact over `Fraction`.
"""
from __future__ import annotations

from collections import defaultdict

from . import runner as R

ADDR_OPS = ("LOAD", "STORE", "ADD", "MUL", "XCHG", "SETL", "SETR")
OPS = ADDR_OPS + ("SWEEP", "END")
PHASES = ("FETCH", "WALK", "RETURN", "SHIFTB", "S_GOER", "S_P1", "S_P2", "S_HOME", "UNSHIFTB",
          "EL_GO", "EL_WALK", "ER_GO", "ER_WALK", "E_HOME", "REWIND")
P_TAPES = tuple("F_" + k for k in OPS) + ("DIG", "Z")
V_TAPES = ("V", "D", "H", "B", "EL", "ER")
TAPES = P_TAPES + V_TAPES
REGS = PHASES + tuple("o_" + k for k in ADDR_OPS) + ("a", "rL", "rR", "c", "e")


# ---------------------------------------------------------------------------
# The description: a closed net over the kit and the tape, as U's program and data.
# ---------------------------------------------------------------------------


class TapeProgram:
    """A guest's program for U and its data layout. `instrs` are `(op, field)`; `W` the block width;
    `tapes` the guest's tape nodes, tape j on track j."""

    def __init__(self, net, instrs, W, tapes, data0, reg_addr, drum_base):
        self.net, self.instrs, self.W, self.tapes = net, instrs, W, tapes
        self.data0, self.reg_addr, self.drum_base = data0, reg_addr, drum_base

    def program_cells(self) -> list:
        """The program bundle, cell by cell: `(op or None, digit)`. An instruction is its op cell,
        then its field in unary closed by an empty cell; SWEEP's and END's field is written twice."""
        out = []
        for op, x in self.instrs:
            out.append((op, 0))
            for _ in range(2 if op in ("SWEEP", "END") else 1):
                out.extend([(None, 1)] * x)
                out.append((None, 0))
        return out

    def encode(self, state: dict):
        """The guest's state as `V`'s cells (a dict from place to value), and the extent's blocks
        `[eL, eR)`: the tapes' support and block 0, two blocks of zeros beyond each end."""
        W, data = self.W, list(self.data0)
        for w, a in self.reg_addr.items():
            data[a] = state[w]
        for w, base in self.drum_base.items():
            n = len(state[w])
            for g in range(n):
                data[base + n - 1 - g] = state[w][g]
        for j, w in enumerate(self.tapes):
            data[j] = state[w].get(0, 0)
        V = {k: v for k, v in enumerate(data) if v != 0}
        lo = hi = 0
        for j, w in enumerate(self.tapes):
            for i, v in state[w].items():
                if v != 0:
                    lo, hi = min(lo, i), max(hi, i)
                    if i != 0:
                        V[i * W + j] = v
        return V, lo - 2, hi + 3

    def decode(self, V: dict) -> dict:
        W, m, state = self.W, len(self.tapes), {}
        for w, a in self.reg_addr.items():
            state[w] = V.get(a, 0)
        for w, base in self.drum_base.items():
            n = len(self.net.nodes[w][1])
            state[w] = [V.get(base + n - 1 - g, 0) for g in range(n)]
        for w in self.tapes:
            state[w] = {}
        for p, v in V.items():
            j = p % W
            if j < m and v != 0:
                state[self.tapes[j]][(p - j) // W] = v
        return state


def compile_tape(net: R.Net, zero=0, one=1) -> TapeProgram:
    """Translate a closed net over the kit and the tape (const, register, drum, tape, add, mul,
    cost, push) into U's program: `runner.compile_net`'s straight-line part, its cells addressed by
    walking instead of by timing, then each tape's two moves and its sweep, then END."""
    nodes = net.nodes
    for nd in nodes:
        if nd[0] not in ("const", "register", "drum", "tape", "slot", "add", "mul"):
            raise ValueError(f"the tape runner's kit has no {nd[0]}")
        if nd[0] == "slot" and nodes[nd[1]][0] != "tape":
            raise ValueError("the tape runner's kit has no wide leaf but the tape")
    if any(s[0] not in ("cost", "push") for s in net.sinks) or net.outputs or net.inboxes():
        raise ValueError("the tape runner runs closed nets over the kit and the tape")
    data0: list = []

    def alloc(v=zero):
        data0.append(v)
        return len(data0) - 1

    tapes = [w for w, nd in enumerate(nodes) if nd[0] == "tape"]
    track = {w: alloc(nodes[w][1].get(0, zero)) for w in tapes}   # tape j's front: track j of block 0
    val, reg_addr, drum_base, state_cells = {}, {}, {}, set(track.values())
    for w, nd in enumerate(nodes):
        if nd[0] == "const":
            val[w] = alloc(nd[1])
        elif nd[0] == "register":
            val[w] = reg_addr[w] = alloc(nd[1])
            state_cells.add(val[w])
        elif nd[0] == "drum":
            n = len(nd[1])
            base = len(data0)
            for p in range(n):
                alloc(nd[1][n - 1 - p])
            drum_base[w] = base
            val[w] = base + n - 1
            state_cells.update(range(base, base + n))
    zc = []

    def zero_cell():
        if not zc:
            zc.append(alloc(zero))
        return zc[0]

    slots = {w: [] for w in tapes}
    for w, nd in enumerate(nodes):
        if nd[0] == "slot":
            slots[nd[1]].append(w)
            val[w] = track[nd[1]] if nd[2] == 0 else zero_cell()   # the levers present 0
    neg1 = alloc(-one) if any(s[0] == "cost" for s in net.sinks) else None
    stateful = [nd[0] in ("register", "drum", "slot") for nd in nodes]
    for w, nd in enumerate(nodes):
        if nd[0] in ("add", "mul"):
            stateful[w] = stateful[nd[1]] or stateful[nd[2]]

    slp = []
    for w, nd in enumerate(nodes):   # forward values
        if nd[0] in ("add", "mul"):
            val[w] = alloc()
            slp.append((nd[0], val[w], val[nd[1]], val[nd[2]]))
    xi = {}

    def acc(w, c):
        if not stateful[w]:
            return
        if w not in xi:
            xi[w] = c
        else:
            z = alloc()
            slp.append(("add", z, xi[w], c))
            xi[w] = z

    for s in net.sinks:
        if s[0] == "cost":
            acc(s[1], neg1)
        else:
            acc(s[2], val[s[1]])
    for w in range(len(nodes) - 1, -1, -1):   # backward pass
        nd = nodes[w]
        if w not in xi or nd[0] not in ("add", "mul"):
            continue
        if nd[0] == "add":
            acc(nd[1], xi[w])
            acc(nd[2], xi[w])
        else:
            i, j = nd[1], nd[2]
            for x, y in ((i, j), (j, i)):
                if stateful[x]:
                    z = alloc()
                    slp.append(("mul", z, xi[w], val[y]))
                    acc(x, z)

    def safe(c):   # a covector held in a cell that is state is copied before the updates
        if c in state_cells:
            z = alloc()
            slp.append(("copy", z, c))
            return z
        return c

    updates = []
    for w, nd in enumerate(nodes):
        if nd[0] in ("register", "drum") and w in xi:
            updates.append((val[w], xi[w]))
    moves = []
    for w in tapes:
        f, l, r = sorted(slots[w], key=lambda s: nodes[s][2])
        if f in xi:
            updates.append((track[w], xi[f]))
        moves.append((track[w], xi[l] if l in xi else zero_cell(), xi[r] if r in xi else zero_cell()))
    updates = [(t, safe(c)) for t, c in updates]
    moves = [(t, safe(cl), safe(cr)) for t, cl, cr in moves]
    for t, c in updates:
        slp.append(("add", t, t, c))
    for w, base in drum_base.items():
        n = len(nodes[w][1])
        if n > 1:
            slp.append(("rot", base, n))
    if not data0:
        alloc(zero)

    instrs = []
    for ins in slp:
        if ins[0] in ("add", "mul"):
            _, z, x, y = ins
            instrs += [("LOAD", x), ("ADD" if ins[0] == "add" else "MUL", y), ("STORE", z)]
        elif ins[0] == "copy":
            instrs += [("LOAD", ins[2]), ("STORE", ins[1])]
        else:
            _, base, n = ins
            instrs.append(("LOAD", base + n - 1))
            instrs += [("XCHG", base + p) for p in range(n)]
    for t, cl, cr in moves:
        instrs += [("SETL", cl), ("SETR", cr), ("SWEEP", t)]
    W = len(data0)
    instrs.append(("END", W))
    return TapeProgram(net, instrs, W, tapes, data0, reg_addr, drum_base)


# ---------------------------------------------------------------------------
# U's state, and U as a net.
# ---------------------------------------------------------------------------


class UState:
    """A state of U kept by absolute place: each tape a dict from place to value, three heads (`hp`
    the program bundle's, `hv` the data bundle's, `hb` the block marks'), the registers, and the
    count of U's steps taken."""

    def __init__(self, tapes, regs, hp=0, hv=0, hb=0, steps=0):
        self.tapes, self.regs = tapes, regs
        self.hp, self.hv, self.hb, self.steps = hp, hv, hb, steps

    def head(self, name):
        return self.hp if name in P_TAPES else (self.hb if name == "B" else self.hv)

    def phase(self) -> str:
        on = [k for k in PHASES if self.regs[k] != 0]
        if len(on) != 1 or self.regs[on[0]] != 1:
            raise ValueError(f"U's phase registers are not one-hot: {on}")
        return on[0]


def runner_init(prog: TapeProgram, gstate: dict) -> UState:
    """U handed `prog` and the guest state `gstate`: the program written from place 0, the data at
    home, the extent's marks, phase FETCH."""
    tapes = {k: {} for k in TAPES}
    for pos, (op, dig) in enumerate(prog.program_cells()):
        if op:
            tapes["F_" + op][pos] = 1
        if dig:
            tapes["DIG"][pos] = 1
    tapes["Z"][0] = 1
    V, eL, eR = prog.encode(gstate)
    W = prog.W
    tapes.update(V=V, H={0: 1}, B={b * W: 1 for b in range(eL, eR)}, EL={eL * W: 1}, ER={eR * W: 1})
    regs = {k: 0 for k in REGS}
    regs["FETCH"] = 1
    return UState(tapes, regs)


class _Poly:
    """Polynomials over the net's wires as add and mul boxes, each built once."""

    def __init__(self, net, one, minus_one):
        self.n, self.memo = net, {}
        self.one, self.m1 = net.const(one), net.const(minus_one)

    def _box(self, kind, x, y):
        key = (kind, min(x, y), max(x, y))
        if key not in self.memo:
            self.memo[key] = getattr(self.n, kind)(x, y)
        return self.memo[key]

    def add(self, x, y):
        return self._box("add", x, y)

    def mul(self, *xs):
        out = xs[0]
        for y in xs[1:]:
            out = self._box("mul", out, y)
        return out

    def neg(self, x):
        return self.mul(self.m1, x)

    def sub(self, x, y):
        return self.add(x, self.neg(y))

    def nt(self, x):   # 1 - x
        return self.sub(self.one, x)

    def sum(self, xs):
        out = xs[0]
        for y in xs[1:]:
            out = self.add(out, y)
        return out


def tape_runner_net(prog: TapeProgram = None, gstate: dict = None, window: int = 0, one=1, minus_one=-1) -> R.Net:
    """U as one closed net over the kit and the tape. Its boxes and pushes are the same for every
    guest; `prog` and `gstate` set only its leaves' initial values (blank without them). `window` is
    the width of each tape in the library (`runner._kit`). `net.names` gives the wires: for each
    tape its `(front, left, right)`, for each register its wire."""
    u = runner_init(prog, gstate) if prog is not None else UState({k: {} for k in TAPES}, dict({k: 0 for k in REGS}, FETCH=1))
    n = R.Net()
    P = _Poly(n, one, minus_one)
    T = {k: n.tape(u.tapes[k], window) for k in TAPES}
    reg = {k: n.register(u.regs[k]) for k in REGS}
    _build(P, T, reg)
    n.names = dict(tapes=T, regs=reg)
    return n


def _build(P, T, reg):
    """U's control: for each phase, its pushes. `f[k]` is the cell under tape k's head."""
    mul, add, sub, neg, nt = P.mul, P.add, P.sub, P.neg, P.nt
    f = {k: T[k][0] for k in TAPES}
    ph = {k: reg[k] for k in PHASES}
    o = {k: reg["o_" + k] for k in ADDR_OPS}
    a, rL, rR, c, e = (reg[k] for k in ("a", "rL", "rR", "c", "e"))
    V = f["V"]
    push = defaultdict(list)
    pR, pL, vR, vL, bR, bL = [], [], [], [], [], []
    wr = defaultdict(list)

    def go(src, dst, h):
        push[ph[src]].append(neg(h))
        push[ph[dst]].append(h)

    nD, nH, nEL, nER, nZ = (nt(f[k]) for k in ("DIG", "H", "EL", "ER", "Z"))

    X = ph["FETCH"]   # read the op cell: latch an address op, or start a sweep or the end
    for k in ADDR_OPS:
        push[o[k]].append(mul(X, sub(f["F_" + k], o[k])))
    sa = P.sum([f["F_" + k] for k in ADDR_OPS])
    push[X].append(neg(mul(X, add(sa, add(f["F_SWEEP"], f["F_END"])))))
    push[ph["WALK"]].append(mul(X, sa))
    push[ph["SHIFTB"]].append(mul(X, f["F_SWEEP"]))
    push[ph["EL_GO"]].append(mul(X, f["F_END"]))
    pR.append(X)

    X = ph["WALK"]    # walk out by the address; at its end act on the cell, as (d)'s runner does
    pR.append(X)
    vR.append(mul(X, f["DIG"]))
    g = mul(X, nD)
    Va = sub(V, a)
    da = P.sum([mul(add(o["LOAD"], o["XCHG"]), Va), mul(o["ADD"], V), mul(o["MUL"], sub(mul(a, V), a))])
    push[a].append(mul(g, da))
    wr["V"].append(mul(g, add(o["STORE"], o["XCHG"]), neg(Va)))
    push[rL].append(mul(g, o["SETL"], sub(V, rL)))
    push[rR].append(mul(g, o["SETR"], sub(V, rR)))
    go("WALK", "RETURN", g)

    X = ph["RETURN"]  # walk home
    vL.append(mul(X, nH))
    go("RETURN", "FETCH", mul(X, f["H"]))

    X = ph["SHIFTB"]  # hold the block marks back by the track
    pR.append(X)
    bL.append(mul(X, f["DIG"]))
    go("SHIFTB", "S_GOER", mul(X, nD))

    X = ph["S_GOER"]  # go to the extent's right end
    vR.append(mul(X, nER))
    push[c].append(neg(mul(X, c)))
    go("S_GOER", "S_P1", mul(X, f["ER"]))

    X = ph["S_P1"]    # right to left: rR (right neighbor) + (1 - rL - rR) (self); the old value to D
    q = mul(X, f["B"])
    wr["V"].append(mul(q, sub(mul(rR, c), mul(add(rL, rR), V))))
    wr["D"].append(mul(q, sub(V, f["D"])))
    push[c].append(mul(q, sub(V, c)))
    push[e].append(neg(mul(X, e)))
    vL.append(mul(X, nEL))
    go("S_P1", "S_P2", mul(X, f["EL"]))

    X = ph["S_P2"]    # left to right: add rL (left neighbor), read off D
    q = mul(X, f["B"])
    wr["V"].append(mul(q, rL, e))
    push[e].append(mul(q, sub(f["D"], e)))
    vR.append(mul(X, nER))
    go("S_P2", "S_HOME", mul(X, f["ER"]))

    X = ph["S_HOME"]
    vL.append(mul(X, nH))
    go("S_HOME", "UNSHIFTB", mul(X, f["H"]))

    X = ph["UNSHIFTB"]  # the block marks back
    pR.append(X)
    bR.append(mul(X, f["DIG"]))
    go("UNSHIFTB", "FETCH", mul(X, nD))

    X = ph["EL_GO"]   # END: move the left end out a block, marking its start
    vL.append(mul(X, nEL))
    h = mul(X, f["EL"])
    wr["EL"].append(neg(h))
    go("EL_GO", "EL_WALK", h)

    X = ph["EL_WALK"]
    pR.append(X)
    vL.append(mul(X, f["DIG"]))
    g = mul(X, nD)
    wr["EL"].append(mul(g, nEL))
    wr["B"].append(mul(g, nt(f["B"])))
    go("EL_WALK", "ER_GO", g)

    X = ph["ER_GO"]   # and the right end, marking the block it adds
    vR.append(mul(X, nER))
    h = mul(X, f["ER"])
    wr["ER"].append(neg(h))
    wr["B"].append(mul(h, nt(f["B"])))
    go("ER_GO", "ER_WALK", h)

    X = ph["ER_WALK"]
    pR.append(X)
    vR.append(mul(X, f["DIG"]))
    g = mul(X, nD)
    wr["ER"].append(mul(g, nER))
    go("ER_WALK", "E_HOME", g)

    X = ph["E_HOME"]
    vL.append(mul(X, nH))
    go("E_HOME", "REWIND", mul(X, f["H"]))

    X = ph["REWIND"]  # back to the program's start
    pL.append(mul(X, nZ))
    go("REWIND", "FETCH", mul(X, f["Z"]))

    for k in P_TAPES:
        push[T[k][1]] += pL
        push[T[k][2]] += pR
    for k in V_TAPES:
        push[T[k][1]] += vL + (bL if k == "B" else [])
        push[T[k][2]] += vR + (bR if k == "B" else [])
        push[T[k][0]] += wr[k]
    for target in sorted(push):
        if push[target]:
            P.n.push(P.sum(push[target]), target)


# ---------------------------------------------------------------------------
# U's step a phase at a time.
# ---------------------------------------------------------------------------


def to_net_state(net: R.Net, u: UState) -> dict:
    """`u` as the state `runner.ref_step` keeps for `net` (each tape relative to its head)."""
    st = {}
    for k, (fw, _l, _r) in net.names["tapes"].items():
        h = u.head(k)
        st[net.nodes[fw][1]] = {p - h: v for p, v in u.tapes[k].items() if v != 0}
    for k, w in net.names["regs"].items():
        st[w] = u.regs[k]
    return st


def _set(d, p, v):
    if v == 0:
        d.pop(p, None)
    else:
        d[p] = v


def _only(d, name):
    ks = [p for p, v in d.items() if v != 0]
    if len(ks) != 1 or d[ks[0]] != 1:
        raise ValueError(f"mark {name} is not a single 1: {ks}")
    return ks[0]


def tape_runner_run(u: UState, guest_steps: int = 1, boundary=None, max_steps: int = None) -> UState:
    """Run U from `u` (changed in place) until `guest_steps` passes have ended (REWIND reaching the
    start), or past `max_steps` of U's steps. One phase at a time: the steps that only walk are
    counted, not taken. `boundary(u)` is called after every phase."""
    t = u.tapes
    Pcells = set().union(*(t[k] for k in P_TAPES))
    lo, hi = min(Pcells), max(Pcells)
    run = {}
    for p in range(hi + 1, lo - 1, -1):   # the program never changes: runs of digits, read once
        run[p] = run.get(p + 1, 0) + 1 if t["DIG"].get(p, 0) == 1 else 0
    flag = {}
    for k in OPS:
        for p, v in t["F_" + k].items():
            if v != 0:
                flag[p] = k
    V, D, B, reg = t["V"], t["D"], t["B"], u.regs
    done = 0
    while done < guest_steps and (max_steps is None or u.steps < max_steps):
        ph = u.phase()
        if ph == "FETCH":
            op = flag.get(u.hp)
            for k in ADDR_OPS:
                reg["o_" + k] = 1 if op == k else 0
            nxt = "WALK" if op in ADDR_OPS else {"SWEEP": "SHIFTB", "END": "EL_GO"}.get(op)
            if nxt is None:
                raise ValueError(f"no op at the program's place {u.hp}")
            u.hp += 1
            u.steps += 1
        elif ph == "WALK":
            x = run.get(u.hp, 0)
            u.hv += x
            u.hb += x
            u.hp += x + 1
            u.steps += x + 1
            cell, a = V.get(u.hv, 0), reg["a"]
            o = {k: reg["o_" + k] for k in ADDR_OPS}
            reg["a"] = a + (o["LOAD"] + o["XCHG"]) * (cell - a) + o["ADD"] * cell + o["MUL"] * (a * cell - a)
            _set(V, u.hv, cell + (o["STORE"] + o["XCHG"]) * (a - cell))
            reg["rL"] = reg["rL"] + o["SETL"] * (cell - reg["rL"])
            reg["rR"] = reg["rR"] + o["SETR"] * (cell - reg["rR"])
            nxt = "RETURN"
        elif ph in ("RETURN", "S_HOME", "E_HOME"):
            home = _only(t["H"], "H")
            if u.hv < home:
                raise ValueError("U is left of home")
            d = u.hv - home
            u.hv -= d
            u.hb -= d
            u.steps += d + 1
            nxt = {"RETURN": "FETCH", "S_HOME": "UNSHIFTB", "E_HOME": "REWIND"}[ph]
        elif ph in ("SHIFTB", "UNSHIFTB"):
            x = run.get(u.hp, 0)
            u.hb += -x if ph == "SHIFTB" else x
            u.hp += x + 1
            u.steps += x + 1
            nxt = "S_GOER" if ph == "SHIFTB" else "FETCH"
        elif ph in ("S_GOER", "ER_GO"):
            er = _only(t["ER"], "ER")
            d = er - u.hv
            if d < 0:
                raise ValueError("U is right of the extent")
            u.hv += d
            u.hb += d
            u.steps += d + 1
            if ph == "S_GOER":
                reg["c"] = 0
                nxt = "S_P1"
            else:
                del t["ER"][er]
                _set(B, u.hb, 1)
                nxt = "ER_WALK"
        elif ph in ("S_P1", "S_P2", "EL_GO"):
            if ph == "S_P2":
                end = _only(t["ER"], "ER")
                d = end - u.hv
            else:
                end = _only(t["EL"], "EL")
                d = u.hv - end
            if d < 0:
                raise ValueError("U is outside the extent")
            if ph != "EL_GO":
                off = u.hb - u.hv
                lo_p, hi_p = (end, u.hv) if ph == "S_P1" else (u.hv, end)
                marked = sorted((p - off, v) for p, v in B.items() if lo_p <= p - off <= hi_p and v != 0)
                rL, rR = reg["rL"], reg["rR"]
                if ph == "S_P1":
                    c = reg["c"]
                    for p, q in reversed(marked):
                        old = V.get(p, 0)
                        _set(V, p, old + q * (rR * c - (rL + rR) * old))
                        dold = D.get(p, 0)
                        _set(D, p, dold + q * (old - dold))
                        c = c + q * (old - c)
                    reg["c"] = c
                    reg["e"] = 0
                else:
                    e = reg["e"]
                    for p, q in marked:
                        _set(V, p, V.get(p, 0) + q * rL * e)
                        e = e + q * (D.get(p, 0) - e)
                    reg["e"] = e
            step = -d if ph != "S_P2" else d
            u.hv += step
            u.hb += step
            u.steps += d + 1
            if ph == "EL_GO":
                del t["EL"][end]
            nxt = {"S_P1": "S_P2", "S_P2": "S_HOME", "EL_GO": "EL_WALK"}[ph]
        elif ph in ("EL_WALK", "ER_WALK"):
            x = run.get(u.hp, 0)
            s = -x if ph == "EL_WALK" else x
            u.hv += s
            u.hb += s
            u.hp += x + 1
            u.steps += x + 1
            if ph == "EL_WALK":
                _set(t["EL"], u.hv, 1)
                _set(B, u.hb, 1)
                nxt = "ER_GO"
            else:
                _set(t["ER"], u.hv, 1)
                nxt = "E_HOME"
        elif ph == "REWIND":
            z = _only(t["Z"], "Z")
            d = u.hp - z
            u.hp = z
            u.steps += d + 1
            nxt = "FETCH"
            done += 1
        reg[ph] = 0
        reg[nxt] = 1
        if boundary is not None:
            boundary(u)
    return u


# ---------------------------------------------------------------------------
# Guests with tapes.
# ---------------------------------------------------------------------------


def trail() -> R.Net:
    """A tape that writes 1 and moves right every step. From blank, after k steps its cells
    -k, ..., -1 hold 1 (the design's §4)."""
    n = R.Net()
    f, _l, r = n.tape()
    one = n.const(1)
    n.push(one, f)
    n.push(one, r)
    return n


def blend(third=None, quarter=None) -> R.Net:
    """A tape moved by fractions, so every cell blends with its neighbors: left by 1/3, right by
    1/4. A register x is pushed by the cell under the head, and a cost x * cell charges both, so the
    tape is written through the chain rule as well as by a push."""
    from fractions import Fraction as F
    third = F(1, 3) if third is None else third
    quarter = F(1, 4) if quarter is None else quarter
    n = R.Net()
    f, l, r = n.tape({0: 2, 1: -1, -2: 3})
    x = n.register(1)
    n.push(n.const(third), l)
    n.push(n.const(quarter), r)
    n.push(f, x)
    n.push(x, f)
    n.cost(n.mul(f, x))
    return n


def two_tapes() -> R.Net:
    """Two tapes and a drum: tape A moves left each step and is written by the drum's front; tape B
    is written by A's cell and moves right by a half."""
    from fractions import Fraction as F
    n = R.Net()
    fa, la, _ra = n.tape({0: 1, 2: 5})
    fb, _lb, rb = n.tape()
    d = n.drum([1, 2, 3])
    one = n.const(1)
    n.push(one, la)
    n.push(d, fa)
    n.push(fa, fb)
    n.push(n.const(F(1, 2)), rb)
    return n


def busy_beaver_2() -> R.Net:
    """Rado's two-state busy beaver as a closed net: A0 -> 1RB, A1 -> 1LB, B0 -> 1LA, B1 -> 1RH.
    A Turing machine is a term over the kit and the tape: the state one-hot registers, and the write,
    the two moves and the next state polynomials in them and the symbol under the head (a Boolean
    function of 0/1 inputs is a polynomial). From blank it halts after 6 steps holding four 1s, and
    stays: the halting state pushes nothing."""
    n = R.Net()
    s, left, right = n.tape()
    qA, qB, qH = n.register(1), n.register(0), n.register(0)
    one, m1 = n.const(1), n.const(-1)
    ns = n.add(one, n.mul(m1, s))
    n.push(n.mul(n.add(qA, qB), ns), s)                     # every rule writes 1
    n.push(n.add(n.mul(qA, ns), n.mul(qB, s)), right)       # A0, B1 move right
    n.push(n.add(n.mul(qA, s), n.mul(qB, ns)), left)        # A1, B0 move left
    n.push(n.add(n.mul(m1, qA), n.mul(qB, ns)), qA)         # to A from B0
    n.push(n.add(qA, n.mul(m1, qB)), qB)                    # to B from A
    n.push(n.mul(qB, s), qH)                                # B1 halts
    return n
