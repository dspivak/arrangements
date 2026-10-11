"""Workers who report what they want, and a fund that reads it (``tests/test_account.py``).

Historical subsidy model: the outside source has no contractual claim on the
product it supports. Its no-ruin result is about that subsidized construction.
Use ``contracts.piecework(needs=...)`` for the contracted wage/output comparison
(book P24); the original tests remain reproducible as historical controls.

The cascade's pieces (``cascade.py``) with a third channel, ``H = R^3``: supply, product, and a
**report**. A message is ``(h_1, h_2, h_3, w)``.

* ``worker``: as in the cascade, amplitude ``a`` held on received supply ``h_1`` with cost
  ``(b/2) a^2 - (alpha a + beta) h_1``; it sends ``(0, a, r, 1)`` with ``r = report(a)``. The
  report is a value sent forward; by default it is ``alpha a + beta``, the worker's want per unit
  of supply, which is the number its forcing puts on its input.
* ``fund_layer``: the cascade's fund, with the wire chosen by ``wire``: ``"exact"`` (composition
  pulls the workers' want back onto the balance), ``"oneway"`` (``stop_gradient`` on the balance,
  so nothing comes back and the fund needs to know nothing of its workers), or ``"subtract"`` (the
  cascade's wire, which cancels the pullback using known ``(alpha, beta)``). ``read_reports``
  adds ``g'(f)`` times the report channel that arrives to the balance's flow.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import jax
import jax.numpy as jnp

from .arrangement import Arrangement
from .attention import input_dim, make_input, message, msg_dim
from .cascade import gate
from .coalgebra import configuration
from .collective import DP, Part, collective, frozen_part, part
from .rvect import euclidean, trivial
from .wiring import compose_seq

DH3 = 3
MDIM3, EDIM3 = msg_dim(DH3), input_dim(DH3, DP)     # dim M = 4, dim E = 5
EMPTY = jnp.zeros(0)
SUPPLY, PRODUCT, REPORT, WEIGHT = 0, 1, 2, 3


def worker(a0: float, *, alpha: float, b: float, beta: float, mu: float, label: str,
           report: Optional[Callable] = None) -> Part:
    """A worker sending ``(0, a, report(a), 1)``; ``report`` defaults to ``alpha a + beta``."""
    rep = report if report is not None else (lambda a: alpha * a + beta)
    return part(
        send=lambda q: message(jnp.array([0.0, q[0], rep(q[0])]), 1.0),
        cost=lambda q, e: 0.5 * b * q[0] ** 2 - (alpha * q[0] + beta) * e[SUPPLY],
        P=configuration(euclidean(1, mu), init=jnp.array([float(a0)])),
        label=label, dim_H=DH3)


def needful_worker(a0: float, *, alpha: float, b: float, gamma: float, nu: float, mu: float,
                   label: str) -> Part:
    """A worker whose want levels off at its need ``nu``: cost
    ``(b/2) a^2 - alpha a h_1 + (gamma/2)(nu - h_1)^2``; it sends ``(0, a, 0, 1)``, no report."""
    return part(
        send=lambda q: message(jnp.array([0.0, q[0], 0.0]), 1.0),
        cost=lambda q, e: (0.5 * b * q[0] ** 2 - alpha * q[0] * e[SUPPLY]
                           + 0.5 * gamma * (nu - e[SUPPLY]) ** 2),
        P=configuration(euclidean(1, mu), init=jnp.array([float(a0)])),
        label=label, dim_H=DH3)



def floor_worker(a0: float, *, alpha: float, b: float, lam: float, nu: float, mu: float,
                 label: str, eps: float = 0.05) -> Part:
    """A worker whose need ``nu`` is a floor: cost
    ``(b/2) a^2 - alpha a h_1 - lam log(sp(h_1 - nu))``, ``sp(x) = eps softplus(x / eps)``;
    desperate below its need, wanting more above it, less and less. Sends ``(0, a, 0, 1)``."""
    def cost(q, e):
        sp = eps * jax.nn.softplus((e[SUPPLY] - nu) / eps)
        return 0.5 * b * q[0] ** 2 - alpha * q[0] * e[SUPPLY] - lam * jnp.log(sp)

    return part(
        send=lambda q: message(jnp.array([0.0, q[0], 0.0]), 1.0),
        cost=cost, P=configuration(euclidean(1, mu), init=jnp.array([float(a0)])),
        label=label, dim_H=DH3)

def source(s: float, label: str = "source") -> Part:
    """An elsewhere: supply ``s`` on channel 1, weight 0, no report."""
    return frozen_part([s, 0.0, 0.0], 0.0, label, dim_H=DH3)


def fund_layer(f0: float, *, mu: float, wmax: float, K: float, c0: float, leak: float,
               wire: str, read_reports: bool = False, known: Optional[tuple] = None,
               label: str = "fund") -> Arrangement:
    """The fund box ``B -> B``: flow ``income - g(f) w~ - c0 - leak f`` on the balance, plus
    ``g'(f) sum kappa r`` when it reads the reports. ``known = (alpha, beta)`` is required by
    ``wire="subtract"`` and used by nothing else."""
    assert wire in ("exact", "oneway", "subtract")
    dgate = jax.grad(gate)

    def xi(q, m_C, e_out):
        f = q[0]
        flow = e_out[SUPPLY] - gate(f, wmax, K) * m_C[WEIGHT] - c0 - leak * f
        if wire == "subtract":
            alpha, beta = known
            flow = flow - dgate(f, wmax, K) * (alpha * m_C[PRODUCT] + beta * m_C[WEIGHT])
        if read_reports:
            flow = flow + dgate(f, wmax, K) * m_C[REPORT]
        return (jnp.array([flow]), jnp.zeros(MDIM3), jnp.zeros(EDIM3))

    def paid(f):
        return jax.lax.stop_gradient(f) if wire == "oneway" else f

    return Arrangement(
        P=configuration(euclidean(1, mu), init=jnp.array([float(f0)])),
        in_dim_M=EDIM3, out_dim_M=MDIM3, in_dim_N=EDIM3, out_dim_N=MDIM3,
        out_f=lambda q, m_C: message(jnp.array([0.0, m_C[PRODUCT], 0.0]), 1.0),
        in_f=lambda q, m_C, e_out: make_input(
            message(jnp.array([gate(paid(q[0]), wmax, K), 0.0, 0.0]), 1.0), jnp.zeros(DP)),
        omega=xi, label=label)


def market(r: float) -> Arrangement:
    """Closes a unit ``I -> B``: pays ``r`` times its product back on channel 1."""
    return Arrangement(
        P=configuration(trivial()),
        in_dim_M=EDIM3, out_dim_M=MDIM3, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m: EMPTY,
        in_f=lambda q, m, n: make_input(
            message(jnp.array([r * m[PRODUCT], 0.0, 0.0]), 1.0), jnp.zeros(DP)),
        omega=lambda q, m, n: (jnp.zeros(0), jnp.zeros(MDIM3), jnp.zeros(0)),
        label="market")


def one_level(workers: List[Part], *, positions, mobility: float, tau: float, fund: dict,
              src: Optional[Part], src_position: float, r: float) -> Arrangement:
    """``market(r)`` after a fund layer after a collective of ``workers`` (and ``src``)."""
    parts = list(workers) + ([src] if src is not None else [])
    pos = list(positions) + ([src_position] if src is not None else [])
    mob = [mobility] * len(workers) + ([0.0] if src is not None else [])
    C = collective(parts, tau=tau, cost=lambda q, m, e: 0.0 * jnp.sum(q), positions=pos,
                   integrator=lambda Q, init: configuration(Q, init=init),
                   mobility=mob, value_mobility=0.0, dim_H=DH3, label="collective")
    return compose_seq(compose_seq(C.arr, fund_layer(**fund)), market(r))
