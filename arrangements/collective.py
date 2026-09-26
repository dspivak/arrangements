"""A collective: many parts on one spatial-attention parent (STATUS.md §3.13).

Until this module, each test that needed a collective built its own (turnover, member,
mutual need, the monitor), with different choices each time. The choices that decided
those runs' outcomes -- whether the parent's cost is on the total it sends or on the mean,
its target, the kernel scale, the placements' integrator and mobility -- are required
arguments here, never defaults, so a run cannot inherit one silently.

The composite is ``compose_seq(tensor(parts), parent.arr())``, an arrangement ``I -> B``
with ``B = (E, M)``, ``M = H x W``, ``E = M x P`` (``attention.py``), here with
``dim H = dim P = 1``. Its configuration is the parts' configurations in order, then the
parent's ``(p_1..p_n, V_1..V_n)`` (eq:spatial-state). A part is *in the relations* to the
degree the parent's kernel reaches it, ``kappa_j = exp(-p_j^2 / 2 tau)``, its weight in
what the collective sends onward, ``m~ = (sum_j kappa_j V_j h_j, sum_j kappa_j w_j)``.

The parent's integrator is any coalgebra on ``Q_n`` built by ``integrator(Q, init)``:
``coalgebra.configuration`` for descent, or a stochastic one from ``rand.py`` (then pair it
with a sampler through ``rand.seed`` before running). The response is diagonal,
``mobility`` on the placements and ``value_mobility`` on the value maps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp
from jax import Array

from .arrangement import Arrangement, exact
from .attention import (
    AttentionArrangement, input_dim, make_input, message, msg_dim, signal,
    spatial_arrangement, spatial_config, spatial_kappa, spatial_split, weight,
)
from .coalgebra import configuration
from .rvect import constant, trivial
from .wiring import compose_seq, tensor_arrangements

DH, DP = 1, 1                                   # the default H = R; P = R always
MDIM, EDIM = msg_dim(DH), input_dim(DH, DP)     # dim M = 2, dim E = 3
EMPTY = jnp.zeros(0)


def ext(dim_H: int = DH) -> Array:
    """The exterior input ``(0_M, 0)``: nothing outside."""
    return make_input(message(jnp.zeros(dim_H), 0.0), jnp.zeros(DP))


EXT = ext()


# ---------------------------------------------------------------------------
# Parts.
# ---------------------------------------------------------------------------


class Part(NamedTuple):
    arr: Arrangement
    q_dim: int
    label: str


def part(send: Callable, cost: Callable, P, label: str, dim_H: int = DH) -> Part:
    """A part ``I -> B``: parameter ``P``, sending ``send(q) in M``, receiving ``e in E``
    through the unique map to the unit's input, and carrying its own cost ``cost(q, e)``
    as the exact forcing ``-dk`` (the idiom of ``tests/test_member.py``)."""
    arr = Arrangement(
        P=P,
        in_dim_M=0, out_dim_M=0, in_dim_N=input_dim(dim_H, DP), out_dim_N=msg_dim(dim_H),
        out_f=lambda q, m_out: send(q),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -cost(q, n_in)),
        label=label,
    )
    return Part(arr, P.Q_dim, label)


def frozen_part(h, w: float, label: str, dim_H: int = DH) -> Part:
    """A part with no configuration and no cost, always sending ``(h, w)``."""
    h = jnp.broadcast_to(jnp.asarray(h, float), (dim_H,))
    return part(lambda q: message(h, w), lambda q, e: 0.0, configuration(trivial()), label, dim_H)


# ---------------------------------------------------------------------------
# The parent's cost, on what the collective sends onward.
# ---------------------------------------------------------------------------


def _onward(n: int, tau: float, dim_H: int = DH) -> Callable:
    """``(q_parent, m) |-> sigma_a(q, m)``, the library's own sending map (eq:maps) of the
    spatial parent, taken from a cost-free copy so that a cost can be written on it."""
    return spatial_arrangement(n, dim_H, DP, tau).sigma


def total_cost(n: int, tau: float, nu: float) -> Callable:
    """``(1/2)|nu - h~|^2`` on the *weighted* signal sent onward: in the affine-quadratic
    family, and blind to weight (``tests/test_turnover.py``, ``total_cost``)."""
    onward = _onward(n, tau)
    return lambda q, m_out, n_in: 0.5 * jnp.sum((nu - signal(onward(q, m_out))) ** 2)


def mean_cost(n: int, tau: float, nu: float) -> Callable:
    """``(1/2)|nu - h~/w~|^2`` on the *mean* readout, extended by ``0_M |-> 0``; it leaves
    the affine-quadratic family (``tests/test_turnover.py``, ``mean_cost``)."""
    onward = _onward(n, tau)

    def readout(m):
        w = weight(m)
        safe = jnp.where(w > 0.0, w, 1.0)
        return jnp.where(w > 0.0, signal(m) / safe, jnp.zeros_like(signal(m)))

    return lambda q, m_out, n_in: 0.5 * jnp.sum((nu - readout(onward(q, m_out))) ** 2)


# ---------------------------------------------------------------------------
# The collective.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Collective:
    parts: tuple
    parent: AttentionArrangement
    arr: Arrangement
    n: int
    tau: float
    dim_H: int = DH

    @property
    def parts_dim(self) -> int:
        return sum(p.q_dim for p in self.parts)


def collective(parts: Sequence[Part], *, tau: float, cost: Callable, positions,
               integrator: Callable, mobility, value_mobility: float,
               values=None, dim_H: int = DH, label: str = "") -> Collective:
    """The composite ``compose_seq(tensor(parts), parent.arr())``.

    ``cost`` is the parent's, a function of ``(q_parent, m_out, n_in)``: build it with
    ``total_cost`` or ``mean_cost`` for this ``n`` and ``tau``. ``integrator(Q, init)``
    gives the parent's parameter on ``Q_n`` with the diagonal response. ``mobility`` is one
    number for every placement or one per placement. ``values`` are the initial value
    maps, the identity if omitted.
    """
    parts = tuple(parts)
    n = len(parts)
    values = [jnp.eye(dim_H)] * n if values is None else values
    mob = jnp.broadcast_to(jnp.asarray(mobility, float), (n * DP,))
    diag = jnp.concatenate([mob, jnp.full(n * dim_H * dim_H, float(value_mobility))])
    parent = spatial_arrangement(
        n, dim_H, DP, tau, cost=cost,
        parameter=integrator(constant(jnp.diag(diag)), spatial_config(positions, values)),
        label=label or f"parent({n})")
    arr = compose_seq(tensor_arrangements([p.arr for p in parts]), parent.arr())
    return Collective(parts, parent, arr, n, float(tau), dim_H)


# ---------------------------------------------------------------------------
# Instruments.
# ---------------------------------------------------------------------------


def position(C: Collective, state) -> Array:
    """The flat composite configuration ``q`` a state presents."""
    return C.arr.P.position(state)


def split(C: Collective, q):
    """``q |-> (the parts' configurations, placements (n,), value maps (n,))``."""
    q = jnp.asarray(q, float)
    qs, i = [], 0
    for p in C.parts:
        qs.append(q[i:i + p.q_dim])
        i += p.q_dim
    pos, V = spatial_split(q[i:], C.n, C.dim_H, DP)
    return qs, jnp.ravel(pos), V if C.dim_H > 1 else jnp.ravel(V)


def emitted(C: Collective, q) -> Array:
    """The parts' messages at ``q``, in order."""
    qs, _, _ = split(C, q)
    return jnp.concatenate([p.arr.out_f(qp, EMPTY) for p, qp in zip(C.parts, qs)])


def sent_onward(C: Collective, q) -> Array:
    """``m~``, the message the collective sends onward at ``q``."""
    q = jnp.asarray(q, float)
    return C.parent.sigma(q[C.parts_dim:], emitted(C, q))


def kappas(C: Collective, q) -> Array:
    """``kappa_j = exp(-p_j^2 / 2 tau)``: how far the parent's kernel reaches each part."""
    _, pos, _ = split(C, q)
    return jnp.array([spatial_kappa(pos[j:j + 1], jnp.zeros(DP), C.tau) for j in range(C.n)])


# ---------------------------------------------------------------------------
# Running.
# ---------------------------------------------------------------------------


def step_fn(C: Collective, e_ext=None) -> Callable:
    """One step of ``Phi(C.arr)`` on the constant exterior input ``e_ext`` with zero
    exterior covector, specialized to the configuration component of the forcing (the
    form ``tests/test_rendezvous.run_closed`` uses and checks against ``Phi``)."""
    P, omega = C.arr.P, C.arr.omega
    e_ext = ext(C.dim_H) if e_ext is None else e_ext

    def step(s):
        xi_q, _, _ = omega(P.position(s), EMPTY, e_ext)
        return P.step(s, xi_q)

    return step


def run(C: Collective, steps: int, state=None, e_ext=None):
    """The state after ``steps`` steps from ``state`` (the initial state if omitted)."""
    f = step_fn(C, e_ext)
    s0 = C.arr.P.init if state is None else state
    return jax.lax.fori_loop(0, steps, lambda _i, s: f(s), s0)


def trace(C: Collective, steps: int, observe: Callable, state=None, e_ext=None):
    """``(final state, observe(s_t) for t = 1..steps)``, by ``lax.scan``; ``observe``
    reads a state, so a momentum is as visible as a position."""
    f = step_fn(C, e_ext)
    s0 = C.arr.P.init if state is None else state

    def body(s, _):
        s = f(s)
        return s, observe(s)

    return jax.lax.scan(body, s0, None, length=steps)


def with_sampler(C: Collective, sampler) -> Collective:
    """A collective with a stochastic parent paired with ``sampler`` (``rand.seed``); its
    state is ``(composite state, sampler state)``, and ``run`` and ``trace`` apply as before."""
    from dataclasses import replace
    from .rand import seed
    return replace(C, arr=seed(C.arr, sampler))


def parent_state(C: Collective, state):
    """The parent's own state inside a composite state: the composite's parameter is
    ``tensor(parts) (x) parent``, so it is the second factor. For the phase coalgebra it is
    ``(q, xi)``, positions and momenta."""
    return state[1]
