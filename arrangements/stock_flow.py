"""Stock-and-flow diagrams as arrangements in ``arr`` (book Ch 4 ``sec.stock_flow``).

Any stock-and-flow diagram with smooth rates and acyclic auxiliaries is an arrangement whose internal
boxes are its stocks, its auxiliaries and its flows, and nothing else (``prop.stock_flow``):

  stock      state = its level, reported on its one output port; no forcing of its own; the
             configuration coalgebra with sharp ``dt`` (``euclidean(1, dt)``), so it moves by ``dt``
             times the force on it.
  auxiliary  a gray box (trivial parameter, zero forcing): a smooth function of what it reads.
  flow       trivial parameter; reads its links and its pipe ends; forcing ``-rate`` on the port of
             the stock it drains, ``+rate`` on the stock it fills, ``0`` on every other port. A cloud end
             is no wire.

A link is a wire whose reader pushes nothing back (its forcing is ``0`` there); a pipe end carries the
stock's level forward and the flow's push back; copying a value is wiring (``_wire``, a lens of
``Lens(FinSet^op)``), not a box, and pushes add where a wire splits. One step of the closed arrangement
is the diagram's forward-Euler step with time step ``dt`` (``euler``), the step stock-and-flow software
takes by default; the forcing is generally not exact (no cost function), which ``arr`` allows.

Origin: the book repo's ``notes/runs/stock-flow-d14d8d10/`` (Claude Opus 5.5, d14d8d10, 2026-10-08, at
David's "I want some kind of general formula for stock and flow, how to convert them into this
formalism").
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .arrangement import Arrangement, zero_form
from .coalgebra import configuration
from .rvect import euclidean, trivial
from .wiring import compose_seq, tensor_arrangements
from .functors import closed_step


def _wire(idx, d_in):  # a wiring: route/copy/drop values; pushes come back summed
    idx = jnp.array(idx, dtype=int)
    return Arrangement(P=configuration(trivial()), in_dim_M=0, out_dim_M=d_in, in_dim_N=0,
        out_dim_N=len(idx), out_f=lambda q, m: m[idx] if len(idx) else jnp.zeros(0),
        in_f=lambda q, m, n: jnp.zeros(0), omega=zero_form, label="wire")

def stock_box(level0, dt):
    return Arrangement(P=configuration(euclidean(1, dt), init=jnp.array([float(level0)])),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m: q, in_f=lambda q, m, n: jnp.zeros(0), omega=zero_form, label="stock")

def aux_box(fn, d_in):
    return Arrangement(P=configuration(trivial()), in_dim_M=0, out_dim_M=d_in, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m: jnp.atleast_1d(fn(m)), in_f=lambda q, m, n: jnp.zeros(0),
        omega=zero_form, label="aux")

def flow_box(rate, d_in, src_slot, tgt_slot):
    def omega(q, m, n):
        r, xi = rate(m), jnp.zeros(d_in)
        if src_slot is not None: xi = xi.at[src_slot].add(-r)
        if tgt_slot is not None: xi = xi.at[tgt_slot].add(r)
        return jnp.zeros(0), xi, jnp.zeros(0)
    return Arrangement(P=configuration(trivial()), in_dim_M=0, out_dim_M=d_in, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m: jnp.zeros(0), in_f=lambda q, m, n: jnp.zeros(0), omega=omega, label="flow")


def arrangement_of(diagram, dt):
    """diagram = (stocks, auxes, flows):
         stocks: [(name, level0)]
         auxes:  [(name, [names read], fn)]        in topological order
         flows:  [(name, src or None, tgt or None, [names read], rate)]   rate(values read)
    Returns the closed arrangement; its state's nonempty leaves are the stocks in order."""
    stocks, auxes, flows = diagram
    names = [s for s, _ in stocks]
    a = tensor_arrangements([stock_box(v, dt) for _, v in stocks])
    for nm, reads, fn in auxes:  # pass everything on, and feed this auxiliary what it reads
        k = len(names)
        a = compose_seq(a, _wire(list(range(k)) + [names.index(r) for r in reads], k))
        a = compose_seq(a, tensor_arrangements([_wire(list(range(k)), k), aux_box(fn, len(reads))]))
        names.append(nm)
    boxes, route = [], []
    for nm, src, tgt, reads, rate in flows:  # inputs: its links, then its pipe ends
        ins = list(reads) + [e for e in (src, tgt) if e is not None and e not in reads]
        nr = len(reads)
        boxes.append(flow_box(lambda m, rate=rate, nr=nr: rate(m[:nr]), len(ins),
                              ins.index(src) if src else None, ins.index(tgt) if tgt else None))
        route += [names.index(e) for e in ins]
    a = compose_seq(a, _wire(route, len(names)))
    return compose_seq(a, tensor_arrangements(boxes))


def euler(diagram, dt, steps):  # the diagram's own update, as stock-and-flow software does it
    stocks, auxes, flows = diagram
    x = np.array([v for _, v in stocks], float); out = []
    for _ in range(steps):
        val = {s: x[i] for i, (s, _) in enumerate(stocks)}
        for nm, reads, fn in auxes: val[nm] = float(fn(jnp.array([val[r] for r in reads])))
        dx = np.zeros_like(x); idx = {s: i for i, (s, _) in enumerate(stocks)}
        for nm, src, tgt, reads, rate in flows:
            r = float(rate(jnp.array([val[e] for e in reads])))
            if src: dx[idx[src]] -= r
            if tgt: dx[idx[tgt]] += r
        x = x + dt * dx; out.append(x.copy())
    return np.array(out)


def run(diagram, dt, steps):
    """The stock levels after each of ``steps`` steps of the closed arrangement."""
    a = arrangement_of(diagram, dt); step = closed_step(a, check=1); s = a.P.init; out = []
    for _ in range(steps):
        s = step(s)
        out.append(np.concatenate([np.asarray(l) for l in jax.tree_util.tree_leaves(s) if l.size]))
    return np.array(out)


def random_diagram(rng):
    n, na, nf = rng.integers(1, 6), rng.integers(0, 5), rng.integers(1, 7)
    stocks = [(f"s{i}", float(rng.uniform(1, 10))) for i in range(n)]
    pool, auxes = [s for s, _ in stocks], []
    for j in range(na):
        reads = list(rng.choice(pool, size=rng.integers(1, len(pool) + 1), replace=False))
        w = jnp.array(rng.normal(size=len(reads)))
        auxes.append((f"a{j}", reads, lambda m, w=w: jnp.tanh(w @ m / 10) * jnp.prod(1 + 0.01 * m)))
        pool.append(f"a{j}")
    flows = []
    for j in range(nf):
        ends = [None] + [s for s, _ in stocks]
        src, tgt = rng.choice(ends), rng.choice(ends)
        if src == tgt: tgt = None if src else stocks[0][0]
        reads = list(rng.choice(pool, size=rng.integers(0, min(3, len(pool)) + 1), replace=False))
        w = jnp.array(rng.normal(size=len(reads))); c = float(rng.normal())
        flows.append((f"f{j}", src, tgt, reads, lambda m, w=w, c=c: jax.nn.softplus(c + w @ m / 5)))
    return stocks, auxes, flows
