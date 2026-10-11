"""An arrangement that runs arrangements: the kit, its terms, and the runner (book plan P34).

The question (D. Spivak, 2026-10-01): "something like a universal Turing machine, except a
universal arrangement runner", for the *constructible* arrangements, those built from a finite
kit by composition and tensor. Statement aimed at: there is a finite kit and one closed
arrangement built from that kit which, handed the description of any closed term over the kit,
runs it, taking many of its own steps for each step of the term.

THE KIT (each a library `Arrangement`; `KIT` below).

* `const(c)`    -- presents the number c. No state.
* `register(x)` -- one real, unit response: x' = x + (covector received).
* `drum(cells)` -- presents its front cell; the covector received is added to the front cell and
                   then the cells rotate one place, front to back. Nothing received can change its
                   length.
* `add`, `mul`  -- stateless, hear two values, present their sum / product; no forcing.
* `cost`        -- stateless, hears v; forcing -dv (the potential -v).
* `push`        -- stateless, hears (d, r); returns the covector d on r's wire and ZERO on d's:
                   a one-way, non-exact forcing.
* `fan(n, idx)` -- a wiring: hears n values, presents the listed ones (copy, permute, drop).

A NET is a closed term over the kit written as a netlist (`Net`): leaves, then boxes each
reading two earlier wires, then sinks. `to_term` writes it with 'seq' and 'tensor'; `denote`
turns the term into the library's arrangement. `ref_step` is the same semantics in plain
Python arithmetic (it runs on `fractions.Fraction`), sharing no code with the library.

THE RUNNER is one net, `machine_net`: four flag drums (load, store, add, multiply), one data
drum, one register `a` (the accumulator), the constant -1, twelve boxes and two pushes. Each of
its steps, with `cell` the data drum's front,

    a'    = a + L (cell - a) + A cell + M (a cell - a)
    cell' = cell + S (a - cell)                 (L and S together exchange a and cell)

then every drum rotates. `compile_net` translates any net into flag-drum contents and a data
layout: the net's forward values, its backward pass (the backward pass of add is a copy, of
mul two muls, so the kit can do its own), the updates of its registers and drums. A drum of
the net is rotated by one pass of exchanges. One pass of the flag drums is one step of the net.

`machine_run` is the runner's step in plain Python, skipping the steps whose flags are all zero
(they only rotate). The test checks it against the library's arrangement of `machine_net`.

WITH PORTS (`port_runner_net`, `asker_net`, `couple`): the runner as one part among others. An
asker sends a translated description over seven wires, one piece a step; the runner loads it,
runs it, and presents its data cells, which the asker samples into registers. `inbox`, a
register that becomes what arrives at its port, is the one part added to the kit for this.

THE PORTS' BACKWARD DIRECTION (book plan P37 (b)(iii′) stage (A); David, 2026-10-08: "give P34's
net ports their backward direction, so a push arriving at a net's output reaches the parts
inside and an inbox returns the covector it receives on its port, with couple then composing
forces as well as values"). A covector arriving at an output wire goes back through the chain
rule to the parts (`ref_step`'s `pushes`; the library does it by pullback-and-sum). An inbox's
state is `[x, r]`: `x` becomes what arrived at the port and `r` the covector that landed on its
wire, which it publishes on the port the next step (`port_covectors`). A box presents, then
hears, so a port's value and its covector each cross one step late. `couple` composes forces as
well as values; `couple_net` is the same pair as one net wired by hand, with a `relay` (an inbox
whose port is a wire of the same net, via two pushes) where each port was.

THE KIT, ENLARGED (the same stage):

* `exp`, `recip`     -- stateless unary boxes, e^v and 1/v; the Euler term (1 + v/N)^N and
                        Newton's iteration y <- y (2 - v y) over the old kit approach them.
* `market(cells)`    -- the belief integrator of P36 (b) (`belief.belief_coalgebra`): state the
                        log-heights, presents the belief on one wire per value, steps by Bayes'
                        rule on the covector it receives.
* `relay(x0)`        -- see above; three wires: the value, the stored covector, and an ear.
* `pcost(d)`         -- a sink whose forcing reads a port THE SAME STEP (David, 2026-10-09: "build the same
                        step kind"): hears d coefficient wires from inside and the port's value n, cost
                        sum_j c_j n^j, returns -n^j on wire j and -sum_j j c_j n^(j-1) on the port. In arr a
                        box's output is Moore (it cannot read its input the same step) but its forcing is
                        not: the 1-form it publishes on its input side is a function of the input it will
                        receive. `portsplit` in the term copies each port's value to the sinks that read it.

A runner over the enlarged kit has six flag drums (exp and reciprocal added), each new term gated
so a zero flag computes exp(0) and 1/1 and adds nothing; `compile_net` lays out a market's
position and update, a relay's two copies, and the unary boxes' backward passes. P34's four-drum
runner is unchanged and still exact in fractions; the six-drum one is checked in floats.

WHAT THE DESCRIPTION IS: the compiled flags and layout, not the raw term. The translation
(`compile_net`) is done outside the runner. See STATUS.md for what this does and does not show.
"""
from __future__ import annotations

from fractions import Fraction
from typing import List, Sequence

# ---------------------------------------------------------------------------
# Nets: closed terms over the kit, as netlists.
# ---------------------------------------------------------------------------

LEAF_KINDS = ("const", "register", "drum", "inbox", "market", "relay")
BOX_KINDS = ("add", "mul", "exp", "recip")
UNARY_KINDS = ("exp", "recip")


def leaf_width(nd) -> int:
    """How many wires a leaf presents: a market one per value, a relay three (value, stored
    covector, ear), every other leaf one."""
    if nd[0] == "market":
        return len(nd[1])
    if nd[0] == "relay":
        return 3
    return 1


class Net:
    """A term over the kit as a netlist. A node's wire id is its index in `nodes`.

    Ports: an `inbox` hears one value at the net's outer input and presents it a step late; `out`
    names the wires presented at the outer output. Both carry a backward direction (2026-10-08,
    David: "give P34's net ports their backward direction"): a covector arriving at an output wire
    reaches the parts inside through the chain rule, and an inbox returns the covector it received
    on its wire to its port, one step late, as it delivered the value. A wide leaf (market, relay)
    is one node followed by its `slot` nodes, one per wire.
    """

    def __init__(self):
        self.nodes: list = []
        self.sinks: list = []
        self.outputs: list = []   # wires presented at the net's outer port, in order

    def inbox(self, x0=0):
        """A port. Its state is `[x, r]`: `x` becomes what arrives at the port (one step late) and
        `r` the covector received on its wire, which it publishes on the port the next step."""
        return self._node("inbox", x0)

    def out(self, w):
        self.outputs.append(w)

    def inboxes(self):
        return [w for w, nd in enumerate(self.nodes) if nd[0] == "inbox"]

    def _node(self, *node):
        self.nodes.append(tuple(node))
        return len(self.nodes) - 1

    def const(self, c):
        return self._node("const", c)

    def register(self, x0):
        return self._node("register", x0)

    def drum(self, cells):
        return self._node("drum", list(cells))

    def market(self, cells):
        """The belief integrator (P36 (b), `belief.belief_coalgebra`): state the log-heights
        `cells`, presents the belief `softmax(cells)` on one wire per value, and on the covector
        `xi` it receives on them steps `l <- l + xi - <q, xi> 1`, Bayes' rule. Returns the wires."""
        w = self._node("market", [c for c in cells])
        return [self._node("slot", w, i) for i in range(len(cells))]

    def relay(self, x0=0):
        """An inbox whose port is a wire of the same net: state `[x, r]`, three wires (`x`, `r`,
        and an ear whose value is 0). `push(src, ear)` makes `x` become `src` next step, and
        `push(r, src)` returns the covector received on `x`'s wire to `src`. Returns
        `(x, r, ear)`."""
        w = self._node("relay", x0)
        return tuple(self._node("slot", w, i) for i in range(3))

    def hear(self, src, x0=0):
        """A relay listening to wire `src`: the one-net form of an inbox wired to `src`."""
        x, r, ear = self.relay(x0)
        self.push(src, ear)
        self.push(r, src)
        return x

    def add(self, i, j):
        return self._node("add", i, j)

    def mul(self, i, j):
        return self._node("mul", i, j)

    def exp(self, i):
        return self._node("exp", i)

    def recip(self, i):
        return self._node("recip", i)

    def cost(self, w):
        self.sinks.append(("cost", w))

    def push(self, d, r):
        self.sinks.append(("push", d, r))

    def pcost(self, coeffs, port):
        """A cost polynomial in the value at port `port` (the net's `port`-th inbox's port), with the
        coefficients `coeffs` wires from inside: `sum_j c_j n^j`. Its forcing reads the port the same
        step and returns `-sum_j j c_j n^(j-1)` on it, and `-n^j` on coefficient wire `j` (David,
        2026-10-09: "build the same step kind"). The port's inbox may be left unread."""
        self.sinks.append(("pcost", int(port)) + tuple(coeffs))

    def leaves(self):
        return [w for w, nd in enumerate(self.nodes) if nd[0] in LEAF_KINDS]

    def init_state(self):
        st = {}
        for w, nd in enumerate(self.nodes):
            if nd[0] == "register":
                st[w] = nd[1]
            elif nd[0] in ("inbox", "relay"):
                st[w] = [nd[1], 0]
            elif nd[0] in ("drum", "market"):
                st[w] = list(nd[1])
        return st


def _softmax(l: Sequence) -> list:
    import math
    m = max(l)
    e = [math.exp(x - m) for x in l]
    s = sum(e)
    return [x / s for x in e]


def _forward(net: Net, state: dict) -> list:
    import math
    val = [None] * len(net.nodes)
    # leaves and their slots first (their values are state), then the boxes in order: a relay's
    # value wire may be read by a box that precedes it in the list (`couple_net`)
    order = [w for w, nd in enumerate(net.nodes) if nd[0] not in BOX_KINDS] + \
            [w for w, nd in enumerate(net.nodes) if nd[0] in BOX_KINDS]
    for w in order:
        nd = net.nodes[w]
        k = nd[0]
        if k == "const":
            val[w] = nd[1]
        elif k == "register":
            val[w] = state[w]
        elif k in ("drum", "inbox", "relay"):
            val[w] = state[w][0]
        elif k == "market":
            val[w] = _softmax(state[w])
        elif k == "slot":
            src, i = nd[1], nd[2]
            if net.nodes[src][0] == "market":
                val[w] = val[src][i]
            else:   # a relay: x, r, ear
                val[w] = state[src][i] if i < 2 else 0
        elif k == "add":
            val[w] = val[nd[1]] + val[nd[2]]
        elif k == "mul":
            val[w] = val[nd[1]] * val[nd[2]]
        elif k == "exp":
            val[w] = math.exp(val[nd[1]])
        else:
            val[w] = 1 / val[nd[1]]
    return val


def present(net: Net, state: dict) -> list:
    """What the net presents at its outer port in this state."""
    val = _forward(net, state)
    return [val[w] for w in net.outputs]


def port_covectors(net: Net, state: dict, inputs: Sequence = None) -> list:
    """What the net publishes on its ports this step: each inbox's stored covector `r`, plus, given the
    values `inputs` arriving at the ports this step, what each `pcost` returns on its port."""
    out = [state[w][1] for w in net.inboxes()]
    if inputs is not None:
        val = None
        for sk in net.sinks:
            if sk[0] == "pcost":
                if val is None:
                    val = _forward(net, state)
                port, coeffs = sk[1], sk[2:]
                n_ = inputs[port]
                out[port] = out[port] - sum(j * val[c] * n_ ** (j - 1) for j, c in enumerate(coeffs) if j > 0)
    return out


def ref_step(net: Net, state: dict, inputs: Sequence = (), pushes: Sequence = ()) -> dict:
    """One step of the net's dynamics, by hand: forward values, backward covectors, updates.

    `inputs` are the values arriving at the net's inboxes, in order; `pushes` the covectors
    arriving at the net's outputs, in order (the backward direction of its ports).
    """
    n = len(net.nodes)
    val = _forward(net, state)
    xi = [0] * n
    for w, p in zip(net.outputs, pushes):
        xi[w] = xi[w] + p
    for s in net.sinks:
        if s[0] == "cost":
            xi[s[1]] = xi[s[1]] + (-1)
        elif s[0] == "pcost":
            n_ = inputs[s[1]]
            for j, c in enumerate(s[2:]):
                xi[c] = xi[c] - n_ ** j
        else:
            xi[s[2]] = xi[s[2]] + val[s[1]]
    wide = {}   # a market's or relay's covector, one per slot
    for w in range(n - 1, -1, -1):
        nd = net.nodes[w]
        if nd[0] == "add":
            xi[nd[1]] = xi[nd[1]] + xi[w]
            xi[nd[2]] = xi[nd[2]] + xi[w]
        elif nd[0] == "mul":
            i, j = nd[1], nd[2]
            gi, gj = xi[w] * val[j], xi[w] * val[i]
            xi[i] = xi[i] + gi
            xi[j] = xi[j] + gj
        elif nd[0] == "exp":
            xi[nd[1]] = xi[nd[1]] + xi[w] * val[w]
        elif nd[0] == "recip":
            xi[nd[1]] = xi[nd[1]] - xi[w] * val[w] * val[w]
    for w, nd in enumerate(net.nodes):   # a wide leaf's covectors, once every reader has pushed back
        if nd[0] == "slot":
            wide.setdefault(nd[1], {})[nd[2]] = xi[w]
    new = {}
    inputs = list(inputs)
    for w, nd in enumerate(net.nodes):
        if nd[0] == "register":
            new[w] = state[w] + xi[w]
        elif nd[0] == "inbox":
            new[w] = [inputs.pop(0), xi[w]]
        elif nd[0] == "drum":
            cells = list(state[w])
            cells[0] = cells[0] + xi[w]
            new[w] = cells[1:] + cells[:1]
        elif nd[0] == "market":
            q, l = val[w], state[w]
            xs = [wide.get(w, {}).get(i, 0) for i in range(len(l))]
            c = sum(qi * x for qi, x in zip(q, xs))
            new[w] = [li + x - c for li, x in zip(l, xs)]
        elif nd[0] == "relay":
            got = wide.get(w, {})
            new[w] = [got.get(2, 0), got.get(0, 0)]
    return new


def to_term(net: Net):
    """The net as a term: nested tuples over the kit with 'seq' and 'tensor'.

    A net with inboxes carries their k input wires down through every layer ('inwire', k);
    its outputs are presented at the end.
    """
    leaves = net.leaves()
    k = len(net.inboxes())
    thread = (("inwire", k),) if k else ()
    pos, n = {}, 0
    for w in leaves:
        nd = net.nodes[w]
        if leaf_width(nd) == 1:
            pos[w] = n
        else:
            for s, snd in enumerate(net.nodes):
                if snd[0] == "slot" and snd[1] == w:
                    pos[s] = n + snd[2]
        n += leaf_width(nd)
    term = ("tensor",) + tuple(net.nodes[w] for w in leaves)
    for w, nd in enumerate(net.nodes):
        if nd[0] in BOX_KINDS:
            ident = list(range(n))
            operands = [pos[x] for x in nd[1:]]
            route = ("tensor", ("fan", n, ident + operands)) + thread
            layer = ("tensor", ("fan", n, ident), (nd[0],)) + thread
            term = ("seq", term, ("seq", route, layer))
            pos[w] = n
            n += 1
    idx, parts, ports = [], [], []
    for s in net.sinks:
        if s[0] == "pcost":
            idx += [pos[x] for x in s[2:]]
            parts.append(("pcost", len(s) - 2))
            ports.append(s[1])
        else:
            idx += [pos[x] for x in s[1:]]
            parts.append((s[0],))
    j = len(net.outputs)
    idx += [pos[w] for w in net.outputs]
    if j:
        parts.append(("fan", j, list(range(j))))
    if parts:
        term = ("seq", term, ("seq", ("tensor", ("fan", n, idx)) + thread, ("tensor",) + tuple(parts) + thread))
    else:
        term = ("seq", term, ("tensor", ("fan", n, [])) + thread)
    if ports:   # each port's value copied to the sinks that read it, and down to the inboxes
        term = ("seq", term, ("portsplit", k, ports + list(range(k)), j))
    return term


# ---------------------------------------------------------------------------
# The kit as library arrangements, and the denotation of a term.
# ---------------------------------------------------------------------------


def _kit():
    import jax.numpy as jnp

    from .arrangement import Arrangement, exact, zero_form
    from .coalgebra import CotCoalg, configuration
    from .rvect import ReactiveVectorSpace
    from .wiring import trivial_parameter

    Z = lambda: jnp.zeros(0)

    def const(c):
        return Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
                           out_f=lambda q, m: jnp.array([float(c)]), in_f=lambda q, m, n: Z(),
                           omega=zero_form, label="const")

    def register(x0):
        Q = ReactiveVectorSpace(dim=1, sharp_fn=lambda q: jnp.eye(1))
        return Arrangement(P=configuration(Q, init=jnp.array([float(x0)])), in_dim_M=0, out_dim_M=0,
                           in_dim_N=0, out_dim_N=1, out_f=lambda q, m: q, in_f=lambda q, m, n: Z(),
                           omega=zero_form, label="register")

    def drum(cells):
        cells = jnp.asarray([float(c) for c in cells])
        P = CotCoalg(Q_dim=int(cells.size), init=cells, position=lambda s: s,
                     step=lambda s, xi: jnp.roll(s + xi, -1), label="drum",
                     timed_step=lambda s, xi, t: jnp.roll(s + xi, -1))   # one cell per step, any duration (book P9.5)
        return Arrangement(P=P, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
                           out_f=lambda q, m: q[:1], in_f=lambda q, m, n: Z(), omega=zero_form, label="drum")

    def add():
        return Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=2, in_dim_N=0, out_dim_N=1,
                           out_f=lambda q, m: m[:1] + m[1:], in_f=lambda q, m, n: Z(),
                           omega=zero_form, label="add")

    def mul():
        return Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=2, in_dim_N=0, out_dim_N=1,
                           out_f=lambda q, m: m[:1] * m[1:], in_f=lambda q, m, n: Z(),
                           omega=zero_form, label="mul")

    def cost():
        return Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=1, in_dim_N=0, out_dim_N=0,
                           out_f=lambda q, m: Z(), in_f=lambda q, m, n: Z(),
                           omega=exact(lambda q, m, n: -m[0]), label="cost")

    def push():
        def omega(q, m, n):
            return Z(), jnp.stack([jnp.zeros(()), m[0]]), Z()
        return Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=2, in_dim_N=0, out_dim_N=0,
                           out_f=lambda q, m: Z(), in_f=lambda q, m, n: Z(), omega=omega, label="push")

    def copier(x0):
        """The integrator of an inbox and a relay: state `[x, r]`, base R^3 presenting
        `(x, r, 0)`; on the covector `xi` it receives it becomes `[xi_2, xi_0]`: `x` what was
        delivered to its ear, `r` what landed on its value wire."""
        return CotCoalg(Q_dim=3, init=jnp.array([float(x0), 0.0]),
                        position=lambda s: jnp.concatenate([s, jnp.zeros(1)]),
                        step=lambda s, xi: jnp.stack([xi[2], xi[0]]), label="copier",
                        timed_step=lambda s, xi, t: jnp.stack([xi[2], xi[0]]))   # one copy per step (book P9.5)

    def inbox(x0):
        def omega(q, m, n):
            return jnp.stack([jnp.zeros(()), jnp.zeros(()), n[0]]), Z(), q[1:2]
        return Arrangement(P=copier(x0), in_dim_M=0, out_dim_M=0, in_dim_N=1, out_dim_N=1,
                           out_f=lambda q, m: q[:1], in_f=lambda q, m, n: Z(), omega=omega, label="inbox")

    def relay(x0):
        return Arrangement(P=copier(x0), in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=3,
                           out_f=lambda q, m: q, in_f=lambda q, m, n: Z(), omega=zero_form, label="relay")

    def market(cells):
        from .belief import belief_coalgebra
        n = len(cells)
        return Arrangement(P=belief_coalgebra(n, jnp.asarray([float(c) for c in cells])), in_dim_M=0,
                           out_dim_M=0, in_dim_N=0, out_dim_N=n, out_f=lambda q, m: q,
                           in_f=lambda q, m, n_: Z(), omega=zero_form, label="market")

    def exp_():
        return Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=1, in_dim_N=0, out_dim_N=1,
                           out_f=lambda q, m: jnp.exp(m), in_f=lambda q, m, n: Z(),
                           omega=zero_form, label="exp")

    def recip():
        return Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=1, in_dim_N=0, out_dim_N=1,
                           out_f=lambda q, m: 1.0 / m, in_f=lambda q, m, n: Z(),
                           omega=zero_form, label="recip")

    def inwire(k):
        return Arrangement(P=trivial_parameter(), in_dim_M=int(k), out_dim_M=0, in_dim_N=int(k), out_dim_N=0,
                           out_f=lambda q, m: Z(), in_f=lambda q, m, n: n, omega=zero_form, label="inwire")

    def fan(n, idx):
        ix = jnp.asarray(list(idx), dtype=int)
        return Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=int(n), in_dim_N=0,
                           out_dim_N=len(idx), out_f=lambda q, m: m[ix], in_f=lambda q, m, n_: Z(),
                           omega=zero_form, label="fan")

    def pcost(d):
        """A cost polynomial in the port's value, `sum_j m_j n^j`, exact; its forcing reads N^- the same step."""
        def U(q, m, n):
            return -sum(m[j] * n[0] ** j for j in range(int(d)))
        return Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=int(d), in_dim_N=1, out_dim_N=0,
                           out_f=lambda q, m: Z(), in_f=lambda q, m, n: Z(), omega=exact(U), label="pcost")

    def portsplit(k, idx, j):
        """A wiring on the input side: the net's `k` port values copied to the slots `idx` below (the sinks
        that read ports, then the inboxes' thread); the `j` output wires pass through."""
        ix = jnp.asarray(list(idx), dtype=int)
        return Arrangement(P=trivial_parameter(), in_dim_M=len(idx), out_dim_M=int(j), in_dim_N=int(k),
                           out_dim_N=int(j), out_f=lambda q, m: m, in_f=lambda q, m, n: n[ix],
                           omega=zero_form, label="portsplit")

    return dict(const=const, register=register, drum=drum, add=add, mul=mul, cost=cost, push=push, fan=fan,
                inbox=inbox, inwire=inwire, relay=relay, market=market, exp=exp_, recip=recip, pcost=pcost,
                portsplit=portsplit)


def denote(term):
    """The arrangement a term denotes: one clause per kit part, one for 'seq', one for 'tensor'."""
    from .wiring import compose_seq, tensor_arrangements

    kit = _kit()

    def go(t):
        head, rest = t[0], t[1:]
        if head == "seq":
            return compose_seq(go(rest[0]), go(rest[1]))
        if head == "tensor":
            parts = [go(x) for x in rest]
            return parts[0] if len(parts) == 1 else tensor_arrangements(parts)
        return kit[head](*rest)

    return go(term)


def flat_state(state) -> list:
    """A library state (nested pairs of arrays) as a flat list: the leaves' states in order."""
    import numpy as np

    out = []

    def rec(x):
        if isinstance(x, (tuple, list)):
            for y in x:
                rec(y)
        else:
            out.extend(np.asarray(x).ravel().tolist())

    rec(state)
    return out


def library_run(net: Net, steps: int, inputs=None, pushes=None, covectors: bool = False):
    """Run the net through the library (`denote`, `Phi`); the flat state after each step.

    `inputs[t]` are the values at the net's ports and `pushes[t]` the covectors at its outputs on
    step `t` (zeros when omitted). With `covectors`, also return per step what the net published
    on its ports (its inboxes' stored covectors, read off the state before the step).
    """
    import jax.numpy as jnp

    from .functors import Phi
    from .interpretation import trivial_omega

    k, j = len(net.inboxes()), len(net.outputs)
    O = Phi(denote(to_term(net)))
    state, IN = O.state, (jnp.zeros(0), trivial_omega(0))
    out, pub = [], []
    for t in range(steps):
        n_in = jnp.asarray(inputs[t] if inputs is not None else [0.0] * k, dtype=float)
        xi_N = jnp.asarray(pushes[t] if pushes is not None else [0.0] * j, dtype=float)
        pub.append([flat_state(state)[p] for p in _inbox_r_positions(net)])
        _, _, state = O.with_state(state).run_one(IN, lambda _o: (xi_N, n_in))
        out.append(flat_state(state))
    return (out, pub) if covectors else out


def _inbox_r_positions(net: Net) -> list:
    """Where each inbox's stored covector sits in the flat state (leaves in order)."""
    off, where = 0, []
    for w in net.leaves():
        nd = net.nodes[w]
        if nd[0] == "inbox":
            where.append(off + 1)
        off += leaf_cells(nd)
    return where


def leaf_cells(nd) -> int:
    """How many cells of state a leaf holds."""
    if nd[0] == "const":
        return 0
    if nd[0] in ("drum", "market"):
        return len(nd[1])
    if nd[0] in ("inbox", "relay"):
        return 2
    return 1


def library_run_jit(net: Net, steps: int, check: int = 2, every: int = 1) -> List[list]:
    """As `library_run`, with the step compiled by JAX; the first `check` steps are also taken
    through `Phi` and compared, so the compiled step is the library's own. Only every
    `every`-th state is kept (and the first `check`).

    For a closed arrangement `Phi`'s step is: present q, take the forcing's component on Q at
    q, hand it to the integrator. That is what is compiled here.
    """
    import jax
    import jax.numpy as jnp

    from .functors import Phi

    A = denote(to_term(net))
    z = jnp.zeros(0)
    step = jax.jit(lambda s: A.P.step(s, A.omega(A.P.position(s), z, z)[0]))
    state, out, head = A.P.init, [], []
    for k in range(1, steps + 1):
        state = step(state)
        if k <= check:
            head.append(flat_state(state))
        if k % every == 0:
            out.append(flat_state(state))
    if check:
        assert library_run(net, check) == head, "compiled step differs from Phi's"
    return out


def couple(net_a: Net, net_b: Net, a_to_b: Sequence[int], b_to_a: Sequence[int]):
    """Two nets with ports, wired to each other: b's i-th inbox hears a's output a_to_b[i], and
    a's i-th inbox hears b's output b_to_a[i]. Returns the closed library arrangement."""
    import jax.numpy as jnp

    from .arrangement import Arrangement, zero_form
    from .wiring import compose_seq, tensor_arrangements, trivial_parameter

    A, B = denote(to_term(net_a)), denote(to_term(net_b))
    ja = len(net_a.outputs)
    route = jnp.asarray([ja + i for i in b_to_a] + list(a_to_b), dtype=int)
    W = Arrangement(P=trivial_parameter(), in_dim_M=A.in_dim_N + B.in_dim_N, out_dim_M=A.out_dim_N + B.out_dim_N,
                    in_dim_N=0, out_dim_N=0, out_f=lambda q, m: jnp.zeros(0),
                    in_f=lambda q, m, n: m[route], omega=zero_form, label="wires")
    return compose_seq(tensor_arrangements([A, B]), W)


def arrangement_run_jit(A, steps: int, check: int = 2, every: int = 1) -> List[list]:
    """Run a closed library arrangement with its step compiled, as `library_run_jit` does for a net."""
    import jax
    import jax.numpy as jnp

    from .functors import Phi
    from .interpretation import trivial_omega

    z = jnp.zeros(0)
    step = jax.jit(lambda s: A.P.step(s, A.omega(A.P.position(s), z, z)[0]))
    state, out, head = A.P.init, [], []
    for k in range(1, steps + 1):
        state = step(state)
        if k <= check:
            head.append(flat_state(state))
        if k % every == 0:
            out.append(flat_state(state))
    if check:
        O = Phi(A)
        slow, IN = O.state, (z, trivial_omega(0))
        for k in range(check):
            _, _, slow = O.with_state(slow).run_one(IN, lambda _o: (z, z))
            assert flat_state(slow) == head[k], "compiled step differs from Phi's"
    return out


def couple_by_hand(net_a: Net, net_b: Net, a_to_b, b_to_a, sa: dict, sb: dict):
    """One step of the two coupled nets, by hand: each presents, and publishes on its ports the
    covectors its inboxes stored; then each steps on what it hears, values at its inboxes and
    covectors at its outputs."""
    oa, ob = present(net_a, sa), present(net_b, sb)
    in_a, in_b = [ob[i] for i in b_to_a], [oa[i] for i in a_to_b]
    ca, cb = port_covectors(net_a, sa, in_a), port_covectors(net_b, sb, in_b)
    pa, pb = [0] * len(net_a.outputs), [0] * len(net_b.outputs)
    for i, j in enumerate(a_to_b):   # b's inbox i hears a's output j, and returns its covector there
        pa[j] = pa[j] + cb[i]
    for i, j in enumerate(b_to_a):
        pb[j] = pb[j] + ca[i]
    return (ref_step(net_a, sa, [ob[i] for i in b_to_a], pa),
            ref_step(net_b, sb, [oa[i] for i in a_to_b], pb))


def couple_net(net_a: Net, net_b: Net, a_to_b, b_to_a) -> Net:
    """The two coupled nets as one net wired by hand: b's nodes after a's, and each inbox made a
    relay listening to the output wire it was coupled to (`Net.hear`). Closed: no ports, no outputs.
    An inbox's node is left in place as the constant 0, so every other index is unchanged."""
    c = Net()
    off = len(net_a.nodes)
    for shift, net in ((0, net_a), (off, net_b)):
        for nd in net.nodes:
            if nd[0] == "inbox":
                c.nodes.append(("const", 0))
            elif nd[0] in BOX_KINDS:
                c.nodes.append((nd[0],) + tuple(x + shift for x in nd[1:]))
            elif nd[0] == "slot":
                c.nodes.append(("slot", nd[1] + shift, nd[2]))
            else:
                c.nodes.append(nd)
    rename, relays = {}, []
    for i, w in enumerate(net_a.inboxes()):
        x, r, ear = c.relay(net_a.nodes[w][1])
        rename[w] = x
        relays.append((r, ear, net_b.outputs[b_to_a[i]] + off))
    for i, w in enumerate(net_b.inboxes()):
        x, r, ear = c.relay(net_b.nodes[w][1])
        rename[w + off] = x
        relays.append((r, ear, net_a.outputs[a_to_b[i]]))
    for k_, nd in enumerate(c.nodes):   # readers of an inbox read its relay's value wire
        if nd[0] in BOX_KINDS:
            c.nodes[k_] = (nd[0],) + tuple(rename.get(x, x) for x in nd[1:])
    src_a = [net_b.outputs[i] + off for i in b_to_a]      # what a's port p hears
    src_b = [net_a.outputs[i] for i in a_to_b]
    for shift, net, srcs in ((0, net_a, src_a), (off, net_b, src_b)):
        for s in net.sinks:
            if s[0] == "pcost":   # the same polynomial, of the wire the port hears, under a cost
                w = rename.get(srcs[s[1]], srcs[s[1]])
                tot, pw = None, None
                for j, cw in enumerate(s[2:]):
                    cw = rename.get(cw + shift, cw + shift)
                    pw = w if j == 1 else (c.mul(pw, w) if j > 1 else None)
                    term = cw if j == 0 else c.mul(cw, pw)
                    tot = term if tot is None else c.add(tot, term)
                c.cost(tot)
            else:
                c.sinks.append((s[0],) + tuple(rename.get(x + shift, x + shift) for x in s[1:]))
    for r, ear, src in relays:
        src = rename.get(src, src)
        c.push(src, ear)
        c.push(r, src)
    return c


def flat_ref(net: Net, state: dict) -> list:
    out = []
    for w in net.leaves():
        if w in state:
            out.extend(state[w] if isinstance(state[w], list) else [state[w]])
    return out


# ---------------------------------------------------------------------------
# The translation: a net to the runner's flags and data layout.
# ---------------------------------------------------------------------------

NOP, LOAD, STORE, ADD, MUL, XCHG, EXP, RECIP = 0, 1, 2, 3, 4, 5, 6, 7


class Program:
    """Flag-drum contents and data layout for one net. `events` are `(step, op, address)`."""

    def __init__(self, net, L, Lp, events, data0, reg_addr, drum_base, n_slp, market_base=None,
                 relay_addr=None):
        self.net, self.L, self.Lp, self.events = net, L, Lp, events
        self.data0, self.reg_addr, self.drum_base, self.n_slp = data0, reg_addr, drum_base, n_slp
        self.market_base = market_base or {}
        self.relay_addr = relay_addr or {}

    def unary(self) -> bool:
        """Does the program use the exponential or the reciprocal?"""
        return any(op in (EXP, RECIP) for _t, op, _a in self.events)

    def flags(self, full: bool = None):
        """The flag drums (load, store, add, multiply; with `full`, also exp and reciprocal), each
        of length Lp. Four drums are P34's runner; a program using a unary instruction needs six."""
        if full is None:
            full = self.unary()
        if not full and self.unary():
            raise ValueError("the net uses exp or recip; its runner needs the six flag drums")
        FL, FS, FA, FM, FX, FR = ([0] * self.Lp for _ in range(6))
        for tau, op, _addr in self.events:
            if op in (LOAD, XCHG):
                FL[tau] = 1
            if op in (STORE, XCHG):
                FS[tau] = 1
            if op == ADD:
                FA[tau] = 1
            if op == MUL:
                FM[tau] = 1
            if op == EXP:
                FX[tau] = 1
            if op == RECIP:
                FR[tau] = 1
        return (FL, FS, FA, FM, FX, FR) if full else (FL, FS, FA, FM)

    def encode(self, state: dict, zero=0) -> list:
        """The data drum holding a state of the net (work cells as in `data0`)."""
        data = list(self.data0)
        for w, a in self.reg_addr.items():
            data[a] = state[w]
        for w, base in self.drum_base.items():
            n = len(state[w])
            for g in range(n):
                data[base + n - 1 - g] = state[w][g]
        for w, base in self.market_base.items():
            for i, v in enumerate(state[w]):
                data[base + i] = v
        for w, (ax, ar) in self.relay_addr.items():
            data[ax], data[ar] = state[w][0], state[w][1]
        return data

    def decode(self, data: Sequence) -> dict:
        state = {}
        for w, a in self.reg_addr.items():
            state[w] = data[a]
        for w, base in self.drum_base.items():
            n = len(self.net.nodes[w][1])
            state[w] = [data[base + n - 1 - g] for g in range(n)]
        for w, base in self.market_base.items():
            state[w] = [data[base + i] for i in range(len(self.net.nodes[w][1]))]
        for w, (ax, ar) in self.relay_addr.items():
            state[w] = [data[ax], data[ar]]
        return state


def compile_net(net: Net, zero=0, one=1, cells: int = None, length: int = None) -> Program:
    """Translate a net into the runner's instructions. `zero`, `one` fix the number type.

    `cells` and `length` fit the translation to a runner whose data drum has `cells` cells and
    whose flag drums have `length` (a multiple of `cells`): spare cells and idle steps pad it.

    Beyond P34's kit: a market's position is computed each pass (exp of each height, their sum,
    its reciprocal, the products) and its update `l += xi - <q, xi>` laid out as multiplies and
    adds; a relay's update is two copies (its value becomes what its ear received, its stored
    covector what its value wire received); the unary boxes' backward passes are multiplies by
    their derivatives, `e^v` and `-1/v^2`.
    """
    nodes = net.nodes
    data0: list = []

    def alloc(v=zero):
        data0.append(v)
        return len(data0) - 1

    val, reg_addr, drum_base, market_base, relay_addr, state_cells = {}, {}, {}, {}, {}, set()
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
            val[w] = base + n - 1  # the front cell
            state_cells.update(range(base, base + n))
        elif nd[0] == "market":
            base = len(data0)
            for c in nd[1]:
                alloc(c)
            market_base[w] = base
            state_cells.update(range(base, base + len(nd[1])))
        elif nd[0] == "relay":
            ax, ar = alloc(nd[1]), alloc()
            relay_addr[w] = (ax, ar)
            state_cells.update((ax, ar))
    zero_cell = None
    for w, nd in enumerate(nodes):   # the slots of markets and relays
        if nd[0] == "slot":
            src, i = nd[1], nd[2]
            if nodes[src][0] == "relay":
                if i < 2:
                    val[w] = relay_addr[src][i]
                else:
                    if zero_cell is None:
                        zero_cell = alloc(zero)
                    val[w] = zero_cell
            else:
                val[w] = alloc()   # the belief's height, computed each pass
    needs_neg = any(s[0] == "cost" for s in net.sinks) or any(nd[0] in ("recip", "market") for nd in nodes)
    neg1 = alloc(-one) if needs_neg else None

    stateful = [nd[0] in ("register", "drum", "slot") for nd in nodes]  # a register, drum, market or relay behind it?
    for w, nd in enumerate(nodes):   # boxes in order; a box may read a relay's slot that comes later in the list
        if nd[0] in BOX_KINDS:
            stateful[w] = any(stateful[x] for x in nd[1:])

    slp = []
    for w, nd in enumerate(nodes):  # the markets' positions
        if nd[0] == "market":
            slots = [s for s, snd in enumerate(nodes) if snd[0] == "slot" and snd[1] == w]
            es = []
            for i, s in enumerate(slots):
                z = alloc()
                slp.append(("exp", z, market_base[w] + i))
                es.append(z)
            tot = es[0]
            for z in es[1:]:
                t = alloc()
                slp.append(("add", t, tot, z))
                tot = t
            rho = alloc()
            slp.append(("recip", rho, tot))
            for s, e in zip(slots, es):
                slp.append(("mul", val[s], e, rho))
    for w, nd in enumerate(nodes):  # forward values
        if nd[0] in ("add", "mul"):
            val[w] = alloc()
            slp.append((nd[0], val[w], val[nd[1]], val[nd[2]]))
        elif nd[0] in UNARY_KINDS:
            val[w] = alloc()
            slp.append((nd[0], val[w], val[nd[1]]))

    xi = {}

    def acc(w, c):  # add the covector held in cell c to wire w's total
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
        elif s[0] == "pcost":
            raise ValueError("a net with a pcost has a port; the runner runs closed nets (couple it first)")
        else:
            acc(s[2], val[s[1]])
    for w in range(len(nodes) - 1, -1, -1):  # backward pass
        nd = nodes[w]
        if w not in xi or nd[0] not in BOX_KINDS:
            continue
        if nd[0] == "add":
            acc(nd[1], xi[w])
            acc(nd[2], xi[w])
        elif nd[0] == "mul":
            i, j = nd[1], nd[2]
            if stateful[i]:
                z = alloc()
                slp.append(("mul", z, xi[w], val[j]))
                acc(i, z)
            if stateful[j]:
                z = alloc()
                slp.append(("mul", z, xi[w], val[i]))
                acc(j, z)
        elif nd[0] == "exp":
            if stateful[nd[1]]:
                z = alloc()
                slp.append(("mul", z, xi[w], val[w]))
                acc(nd[1], z)
        else:  # recip: xi_in = -xi v^2, v the reciprocal presented
            if stateful[nd[1]]:
                t, u, z = alloc(), alloc(), alloc()
                slp.append(("mul", t, val[w], val[w]))
                slp.append(("mul", u, neg1, t))
                slp.append(("mul", z, xi[w], u))
                acc(nd[1], z)

    # the updates: first every covector that is another part's state is copied, then applied
    updates = []   # (kind, target, cell)
    for w, nd in enumerate(nodes):
        if nd[0] in ("register", "drum") and w in xi:
            updates.append(("add", val[w], xi[w]))
        elif nd[0] == "relay":
            slots = [s for s, snd in enumerate(nodes) if snd[0] == "slot" and snd[1] == w]
            ax, ar = relay_addr[w]
            updates.append(("set", ax, xi.get(slots[2], zero_cell if zero_cell is not None else None)))
            updates.append(("set", ar, xi.get(slots[0], None)))
        elif nd[0] == "market":
            slots = [s for s, snd in enumerate(nodes) if snd[0] == "slot" and snd[1] == w]
            xs = [xi.get(s) for s in slots]
            if all(x is None for x in xs):
                continue
            prods = []
            for s, x in zip(slots, xs):
                if x is not None:
                    z = alloc()
                    slp.append(("mul", z, val[s], x))
                    prods.append(z)
            tot = prods[0]
            for z in prods[1:]:
                t = alloc()
                slp.append(("add", t, tot, z))
                tot = t
            nc = alloc()
            slp.append(("mul", nc, neg1, tot))
            for i, x in enumerate(xs):
                if x is None:
                    updates.append(("add", market_base[w] + i, nc))
                else:
                    d = alloc()
                    slp.append(("add", d, x, nc))
                    updates.append(("add", market_base[w] + i, d))
    safe = []
    for kind, target, c in updates:
        if c is None:   # a relay cell receiving nothing becomes 0
            if zero_cell is None:
                zero_cell = alloc(zero)
            c = zero_cell
        if c in state_cells:
            z = alloc()
            slp.append(("copy", z, c))
            c = z
        safe.append((kind, target, c))
    for kind, target, c in safe:
        if kind == "add":
            slp.append(("add", target, target, c))
        else:
            slp.append(("copy", target, c))
    for w, base in drum_base.items():
        n = len(nodes[w][1])
        if n > 1:
            slp.append(("rot", base, n))

    if cells is not None:
        if len(data0) > cells:
            raise ValueError(f"the net needs {len(data0)} cells; the runner has {cells}")
        data0.extend([zero] * (cells - len(data0)))
    L = max(len(data0), 1)
    if not data0:
        data0.append(zero)
    events, tau = [], -1

    def at(op, addr):
        nonlocal tau
        tau = tau + 1 + ((addr - (tau + 1)) % L)
        events.append((tau, op, addr))

    def now(op):   # a unary instruction on the accumulator: the next step, whatever cell shows
        nonlocal tau
        tau = tau + 1
        events.append((tau, op, tau % L))

    for ins in slp:
        if ins[0] in ("add", "mul"):
            _, z, x, y = ins
            first = (tau + 1) % L  # read first the operand that comes round sooner
            if ((y - first) % L) < ((x - first) % L):
                x, y = y, x
            at(LOAD, x)
            at(ADD if ins[0] == "add" else MUL, y)
            at(STORE, z)
        elif ins[0] in UNARY_KINDS:
            _, z, x = ins
            at(LOAD, x)
            now(EXP if ins[0] == "exp" else RECIP)
            at(STORE, z)
        elif ins[0] == "copy":
            at(LOAD, ins[2])
            at(STORE, ins[1])
        else:
            _, base, n = ins
            at(LOAD, base + n - 1)
            for p in range(n):
                at(XCHG, base + p)
    Lp = L * max(1, -(-(tau + 1) // L))
    if length is not None:
        if Lp > length or length % L:
            raise ValueError(f"the net needs {Lp} steps a pass; the runner's flag drums hold {length}")
        Lp = length
    return Program(net, L, Lp, events, data0, reg_addr, drum_base, len(slp), market_base, relay_addr)


# ---------------------------------------------------------------------------
# The runner: one net over the kit, and its step in plain Python.
# ---------------------------------------------------------------------------


def machine_net(flags, data, a0=0, minus_one=-1) -> Net:
    """The runner as a net. With four flag drums it is P34's: five drums, one register, one
    constant, twelve boxes, two pushes. With six (exp and reciprocal added), two more terms move
    the accumulator, each gated so that a flag of 0 computes `exp(0)` and `1/1` and adds nothing:

        a' = a + ... + X (exp(X a) - a) + R (recip(1 + R (a - 1)) - a).
    """
    full = len(flags) == 6
    FL, FS, FA, FM = flags[:4]
    m = Net()
    L, S, A, M = m.drum(FL), m.drum(FS), m.drum(FA), m.drum(FM)
    if full:
        X, Rc = m.drum(flags[4]), m.drum(flags[5])
    cell = m.drum(data)
    a = m.register(a0)
    neg = m.const(minus_one)
    na = m.mul(neg, a)
    ncell = m.mul(neg, cell)
    t1 = m.add(cell, na)          # cell - a
    u1 = m.mul(L, t1)
    u2 = m.mul(A, cell)
    ac = m.mul(a, cell)
    t2 = m.add(ac, na)            # a cell - a
    u3 = m.mul(M, t2)
    da = m.add(m.add(u1, u2), u3)
    if full:
        one = m.const(-minus_one)
        ex = m.exp(m.mul(X, a))                                   # exp(X a)
        u4 = m.mul(X, m.add(ex, na))                              # X (exp(X a) - a)
        arg = m.add(one, m.mul(Rc, m.add(a, neg)))                # 1 + R (a - 1)
        u5 = m.mul(Rc, m.add(m.recip(arg), na))                   # R (1/arg - a)
        da = m.add(da, m.add(u4, u5))
    t3 = m.add(a, ncell)          # a - cell
    dc = m.mul(S, t3)
    m.push(da, a)
    m.push(dc, cell)
    return m


def machine_run(prog: Program, data: list, a, passes: int = 1):
    """`passes` passes of the flag drums over `data` (changed in place); returns the accumulator.

    Steps whose flags are all zero only rotate the drums and are skipped; the arithmetic of a
    step with a flag set is the runner's own (the formulas in the module docstring).
    """
    import math
    for _ in range(passes):
        for _tau, op, i in prog.events:
            cell = data[i]
            if op == LOAD:
                a = a + (cell - a)
            elif op == STORE:
                data[i] = cell + (a - cell)
            elif op == ADD:
                a = a + cell
            elif op == MUL:
                a = a + (a * cell - a)
            elif op == EXP:
                a = a + (math.exp(a) - a)
            elif op == RECIP:
                a = a + (1 / a - a)
            else:  # XCHG: load and store together
                a, data[i] = a + (cell - a), cell + (a - cell)
    return a


# ---------------------------------------------------------------------------
# Three small nets used by the test and the notes.
# ---------------------------------------------------------------------------


def two_tanks(v1=3, v2=1, kappa=Fraction(1, 20)) -> Net:
    """Two registers joined by the cost (kappa/2)(v1 - v2)^2: two tanks and a pipe, in the units
    where a tank's response is one. Closed form: v1, v2 = 2 +- (1 - 2 kappa)^k from 3, 1."""
    n = Net()
    x, y = n.register(v1), n.register(v2)
    neg, hk = n.const(-1), n.const(kappa / 2)
    d = n.add(x, n.mul(neg, y))
    n.cost(n.mul(hk, n.mul(d, d)))
    return n


def turner(x0=1, y0=0, eps=Fraction(1, 8)) -> Net:
    """x' = x + eps y, y' = y - eps x: two pushes, a forcing with no potential."""
    n = Net()
    x, y = n.register(x0), n.register(y0)
    e, ne = n.const(eps), n.const(-eps)
    n.push(n.mul(e, y), x)
    n.push(n.mul(ne, x), y)
    return n


def crossed(x0=1, y0=2) -> Net:
    """x' = x + y, y' = y + x, each pushed by the other's state itself."""
    n = Net()
    x, y = n.register(x0), n.register(y0)
    n.push(y, x)
    n.push(x, y)
    return n


# ---------------------------------------------------------------------------
# The runner as a part with ports, and a part that asks it.
# ---------------------------------------------------------------------------


def port_runner_net(cells: int, length: int, zero=0, minus_one=-1) -> Net:
    """The runner with an outer port. It hears seven values a step: a load gate, a run gate, four
    flags and a number. While the load gate is 1 the front cell of each of its drums becomes
    what arrived; while the run gate is 1 it runs what its drums hold; it always presents the
    front cell of its data drum. Its drums' sizes are fixed: nothing it hears can change them.
    """
    m = Net()
    g, e = m.inbox(zero), m.inbox(zero)
    vL, vS, vA, vM, vd = (m.inbox(zero) for _ in range(5))
    L, S, A, M = (m.drum([zero] * length) for _ in range(4))
    cell = m.drum([zero] * cells)
    a = m.register(zero)
    neg = m.const(minus_one)
    for F, v in ((L, vL), (S, vS), (A, vA), (M, vM)):       # loading a flag drum
        m.push(m.mul(g, m.add(v, m.mul(neg, F))), F)
    na, ncell = m.mul(neg, a), m.mul(neg, cell)
    u1 = m.mul(L, m.add(cell, na))
    u2 = m.mul(A, cell)
    u3 = m.mul(M, m.add(m.mul(a, cell), na))
    m.push(m.mul(e, m.add(m.add(u1, u2), u3)), a)
    run_c = m.mul(e, m.mul(S, m.add(a, ncell)))
    load_c = m.mul(g, m.add(vd, ncell))
    m.push(m.add(run_c, load_c), cell)
    m.out(cell)
    return m


def asker_net(prog: Program, passes: int, addresses: Sequence[int], zero=0, one=1, minus_one=-1):
    """A part that sends `prog` to a port runner, has it run `passes` passes, and keeps the answer.

    It is seven drums it sends from (load gate, run gate, four flags, the numbers), one inbox for
    what comes back, and for each wanted cell a sampling drum and a register `ans`. Returns the
    net, the number of steps the exchange takes, and the wires of the answer registers.
    """
    Lp, L = prog.Lp, prog.L
    start_read = Lp + passes * Lp + 1
    times = [start_read + ((a + 1 - start_read) % L) for a in addresses]   # runner step showing cell a
    T = max(times) + 3
    FL, FS, FA, FM = prog.flags()
    n = Net()
    G = n.drum([one if t < Lp else zero for t in range(T)])
    E = n.drum([one if Lp <= t < Lp + passes * Lp else zero for t in range(T)])
    fl, fs, fa, fm = (n.drum([one if x else zero for x in F]) for F in (FL, FS, FA, FM))
    d = n.drum(list(prog.data0))
    heard = n.inbox(zero)
    neg = n.const(minus_one)
    answers = []
    for t in times:
        samp = n.drum([one if k == t + 1 else zero for k in range(T)])
        ans = n.register(zero)
        n.push(n.mul(samp, n.add(heard, n.mul(neg, ans))), ans)
        answers.append(ans)
    for w in (G, E, fl, fs, fa, fm, d):
        n.out(w)
    return n, T, answers
