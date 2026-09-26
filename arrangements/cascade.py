"""A fund that pays its parts, nested to any depth (STATUS.md §3.14).

The pieces of the cascade test (``tests/test_cascade.py``), on ``collective.py`` with two
channels, ``H = R^2``: channel 1 carries *supply* (a wage, a source's offer), channel 2 carries
*product*. A message is ``(h_1, h_2, w)``; an input adds a reference position.

* ``worker``: a part holding an amplitude ``a`` on the supply it receives, cost
  ``(b/2) a^2 - (alpha a + beta) h_1``; it sends ``(0, a, 1)``.
* ``source``: a fixed part offering supply ``s`` at weight 0 (it is not a head the fund pays).
* ``fund_layer``: a box ``B -> B`` wrapping a collective into a unit ``I -> B``. Its balance
  ``f`` sends the collective the wage ``g(f)`` on channel 1 (each part in reach receives
  ``kappa_j g(f)``), passes the collective's product up as ``(0, h_2, 1)`` (the unit is one
  head to its own parent), and moves by the non-exact flow
  ``income - g(f) w~ - c_0 - leak f``: what arrives on the unit's own channel 1, less the wage bill
  (``w~``, the collective's onward weight, counts the heads in reach), less an overhead,
  less the fraction ``leak`` of its balance spent.
* ``market``: closes a unit, paying ``r`` times its product back on its channel 1.
* ``unit``: a collective of sub-units (and a source, if any) wrapped by a fund layer.

Everything is an ``Arrangement``; ``Unit`` records where each fund and each placement sits in
the composite configuration so that runs can be read level by level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import jax
import jax.numpy as jnp

from .arrangement import Arrangement
from .attention import input_dim, make_input, message, msg_dim
from .coalgebra import configuration
from .collective import DP, Part, collective, frozen_part, part
from .rvect import euclidean, trivial
from .wiring import compose_seq

DH2 = 2
MDIM2, EDIM2 = msg_dim(DH2), input_dim(DH2, DP)     # dim M = 3, dim E = 4
EMPTY = jnp.zeros(0)


def worker(a0: float, *, alpha: float, b: float, beta: float, mu: float, label: str) -> Part:
    """A worker: amplitude ``a`` (mobility ``mu``) held on received supply ``h_1``."""
    return part(
        send=lambda q: message(jnp.array([0.0, q[0]]), 1.0),
        cost=lambda q, e: 0.5 * b * q[0] ** 2 - (alpha * q[0] + beta) * e[0],
        P=configuration(euclidean(1, mu), init=jnp.array([float(a0)])),
        label=label, dim_H=DH2)


def source(s: float, label: str = "source") -> Part:
    """An elsewhere: supply ``s`` on channel 1, weight 0."""
    return frozen_part([s, 0.0], 0.0, label, dim_H=DH2)


def gate(f, wmax: float, K: float, eps: float = 0.05):
    """``g(f) = wmax u / (K + u)``, ``u = eps softplus(f / eps)``: smooth, near 0 for
    ``f <= 0``, saturating at ``wmax``."""
    u = eps * jax.nn.softplus(f / eps)
    return wmax * u / (K + u)


def fund_layer(f0: float, *, mu: float, wmax: float, K: float, c0: float, leak: float,
               record: Optional[tuple] = None, label: str = "fund") -> Arrangement:
    """The fund box ``B -> B`` (module docstring). ``leak`` is the fraction of its balance
    it spends per unit time; without it a fund in surplus grows without bound.

    **Retroactivity.** The wage ``g(f)`` enters the collective's input, so composition pulls
    the collective's forcing on that input back onto ``f``: workers with cost
    ``-(alpha a + beta) h_1`` push the balance up at the rate
    ``g'(f) (alpha sum_j kappa_j a_j + beta sum_j kappa_j)`` -- exact coupling is mutual, and
    an account paid out by gate is propped up by the need of those it pays. ``record =
    (alpha, beta)`` cancels that term exactly, reading the two sums off the collective's
    onward message ``(0, sum kappa a, sum kappa)``: the composite is then non-exact, the wage
    wire one-way, and the balance moves by the declared flow alone (``tests/test_cascade.py``
    checks both)."""

    def xi(q, m_C, e_out):
        f = q[0]
        flow = e_out[0] - gate(f, wmax, K) * m_C[2] - c0 - leak * f
        if record is not None:
            alpha, beta = record
            flow = flow - jax.grad(gate)(f, wmax, K) * (alpha * m_C[1] + beta * m_C[2])
        return (jnp.array([flow]), jnp.zeros(MDIM2), jnp.zeros(EDIM2))

    return Arrangement(
        P=configuration(euclidean(1, mu), init=jnp.array([float(f0)])),
        in_dim_M=EDIM2, out_dim_M=MDIM2, in_dim_N=EDIM2, out_dim_N=MDIM2,
        out_f=lambda q, m_C: message(jnp.array([0.0, m_C[1]]), 1.0),
        in_f=lambda q, m_C, e_out: make_input(
            message(jnp.array([gate(q[0], wmax, K), 0.0]), 1.0), jnp.zeros(DP)),
        omega=xi, label=label)


def market(r: float) -> Arrangement:
    """Closes a unit ``I -> B``: pays ``r`` times its product back on channel 1."""
    return Arrangement(
        P=configuration(trivial()),
        in_dim_M=EDIM2, out_dim_M=MDIM2, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m: EMPTY,
        in_f=lambda q, m, n: make_input(message(jnp.array([r * m[1], 0.0]), 1.0), jnp.zeros(DP)),
        omega=lambda q, m, n: (jnp.zeros(0), jnp.zeros(MDIM2), jnp.zeros(0)),
        label="market")


def supplied(level: float) -> jnp.ndarray:
    """A fixed exterior input: supply ``level`` on channel 1."""
    return make_input(message(jnp.array([level, 0.0]), 1.0), jnp.zeros(DP))


@dataclass
class Unit:
    """A unit ``I -> B`` and where its pieces sit in its own configuration."""
    part: Part
    children: List["Unit"] = field(default_factory=list)
    n_children: int = 0            # sub-units, not counting a source
    has_source: bool = False
    depth: int = 0                 # 0 for a worker

    @property
    def arr(self) -> Arrangement:
        return self.part.arr


def leaf(w: Part) -> Unit:
    return Unit(part=w, depth=0)


def unit(children: List[Unit], *, positions, mobilities, tau: float, fund: dict,
         src: Optional[Part] = None, src_position: float = 0.0, label: str = "unit") -> Unit:
    """A collective of ``children`` (and ``src`` at ``src_position``, fixed), wrapped by a fund
    layer with parameters ``fund`` (``f0, mu, wmax, K, c0``)."""
    parts = [c.part for c in children] + ([src] if src is not None else [])
    pos = list(positions) + ([src_position] if src is not None else [])
    mob = list(mobilities) + ([0.0] if src is not None else [])
    C = collective(parts, tau=tau, cost=lambda q, m, e: 0.0 * jnp.sum(q), positions=pos,
                   integrator=lambda Q, init: configuration(Q, init=init),
                   mobility=mob, value_mobility=0.0, dim_H=DH2, label=label)
    arr = compose_seq(C.arr, fund_layer(label=f"fund:{label}", **fund))
    return Unit(part=Part(arr, arr.P.Q_dim, label), children=list(children),
                n_children=len(children), has_source=src is not None,
                depth=1 + max(c.depth for c in children))


def read(u: Unit, q) -> dict:
    """The configuration of ``u`` read level by level: ``{'fund', 'positions', 'children'}``
    for a unit, ``{'a'}`` for a worker. Layout: the collective's parts in order, then its
    parent's placements and value maps, then the fund."""
    q = jnp.asarray(q)
    if u.depth == 0:
        return {"a": q[0]}
    i, kids = 0, []
    for c in u.children:
        kids.append(read(c, q[i:i + c.part.q_dim]))
        i += c.part.q_dim
    n = u.n_children + (1 if u.has_source else 0)
    return {"positions": q[i:i + n * DP][:u.n_children], "children": kids,
            "fund": q[-1]}


def closed_step(arr: Arrangement, e_ext):
    """One step of the arrangement on a constant exterior input (``e_ext`` empty when closed)."""
    P, omega = arr.P, arr.omega

    def step(s):
        xi_q, _, _ = omega(P.position(s), EMPTY, e_ext)
        return P.step(s, xi_q)

    return step
