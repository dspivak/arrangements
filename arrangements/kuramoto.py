"""Kuramoto oscillators via edge-box wiring (compositional, in ``arr``).

The claim under test: the Kuramoto model

    theta_i' = omega_i + K * sum_{j ~ i} sin(theta_j - theta_i)

is the dynamics ``Phi`` (sec.dynamics_functor) of a closed arrangement whose
parameter is the configuration coalgebra and whose total forcing is
``omega = -dk`` for the cost

    k(theta) = - K * sum_{{i,j} in edges} cos(theta_i - theta_j)
               - sum_i omega_i * theta_i,

built by *genuine composition* in ``arr`` (no hand-written total forcing):

* one **oscillator box** per node ``i``: parameter ``parameter(Q_i)`` on
  ``Q_i = R`` with constant scalar sharp ``c`` (``configuration`` reads ``c`` as
  the time step ``dt``; ``phase`` reads it as the inverse inertia ``1/m``, giving
  the swing equation), output the reported angle ``theta_i`` broadcast to the
  ``deg(i)`` incident edge ports (the diagonal, as in ``wiring.harmonic_vertex``),
  forcing the drift term ``omega_i * dtheta_i`` (``= -d(-omega_i theta_i)``);
* one stateless **coupling box** per edge ``{i,j}``: trivial parameter ``R^0``,
  two input ports, no outputs, forcing ``-dk_e`` for the edge cost
  ``k_e = -K * cos(theta_i - theta_j)`` read off its inputs;
* one **wiring**: the static arrangement (image under ``R^-``, CDLM lem.lens_pow,
  of a finite-set lens exactly as ``wiring.graph_wire``) that routes each
  oscillator's broadcast out-ports bijectively onto the edge boxes' in-ports.

``kuramoto_arrangement`` is ``compose_seq(tensor(boxes), wire)``; the total
forcing above *emerges* from ``compose_seq``'s pullback-and-sum (sec.forcing).
Because the integrator is part of the arrangement (def.arr), the Kuramoto and
the swing-equation systems are two *different* arrangements on the same boxes,
selected by the ``parameter`` keyword -- there is no second dynamics functor.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

import jax.numpy as jnp
from jax import Array

from .arrangement import Arrangement, exact, zero_form
from .coalgebra import configuration
from .rvect import diagonal
from .wiring import compose_seq, tensor_arrangements, trivial_parameter


# ---------------------------------------------------------------------------
# Port bookkeeping: incidences and the routing bijection.
# ---------------------------------------------------------------------------


def _incidences(
    num_oscillators: int, edges: Sequence[Tuple[int, int]]
) -> Tuple[List[int], List[int]]:
    """Degrees and the routing permutation for the Kuramoto wire.

    Oscillator ``v`` gets one out-port per incidence ``(v, e)`` with an edge
    ``e``; edge box ``e = {i, j}`` has two in-ports, slot 0 for ``theta_i`` and
    slot 1 for ``theta_j``. Out-ports are numbered oscillator-major (all of
    oscillator 0's ports, then oscillator 1's, ...), in-ports edge-major.
    Since every edge has exactly two incidences, both sides have ``2E`` ports
    and the routing ``in-port |-> out-port`` is a bijection ``perm`` with
    ``in_f(m_out) = m_out[perm]``.
    """
    for e, (i, j) in enumerate(edges):
        if i == j:
            raise ValueError(f"edge {e} = ({i}, {j}) is a self-loop; not allowed")
        if not (0 <= i < num_oscillators and 0 <= j < num_oscillators):
            raise ValueError(f"edge {e} = ({i}, {j}) out of range")

    out_port: Dict[Tuple[int, int], int] = {}
    degrees: List[int] = []
    idx = 0
    for v in range(num_oscillators):
        deg = 0
        for e, (i, j) in enumerate(edges):
            if v in (i, j):
                out_port[(v, e)] = idx
                idx += 1
                deg += 1
        degrees.append(deg)

    perm: List[int] = []
    for e, (i, j) in enumerate(edges):
        perm.append(out_port[(i, e)])
        perm.append(out_port[(j, e)])
    return degrees, perm


# ---------------------------------------------------------------------------
# The three kinds of box.
# ---------------------------------------------------------------------------


def kuramoto_oscillator(
    omega: float, degree: int, sharp_scale: float = 1.0, parameter=configuration
) -> Arrangement:
    """The oscillator box ``Osc_i : <R^0|R^0> -> <R^0 | R^{deg}>``.

    Parameter ``parameter(Q_i)`` on ``Q_i = R`` holding the angle ``theta_i``,
    with constant scalar sharp ``sharp_scale`` (``dt`` for ``configuration``,
    ``1/m`` for ``phase``). Output map the diagonal broadcast of ``theta_i`` to
    the ``degree`` incident edge ports; input map vacuous; forcing the drift
    term, ``omega = -dk`` for the cost ``k = -omega_i * theta_i`` (so the
    parameter covector is ``+omega_i``, the natural frequency).
    """

    def out_f(q: Array, m_out: Array) -> Array:
        return jnp.tile(q, degree)

    def in_f(q: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.zeros(0)

    def k(q: Array, m_out: Array, n_in: Array) -> Array:
        return -omega * q[0]

    return Arrangement(
        P=parameter(diagonal(jnp.full(1, sharp_scale))),
        in_dim_M=0,
        out_dim_M=0,
        in_dim_N=0,
        out_dim_N=degree,
        out_f=out_f,
        in_f=in_f,
        omega=exact(lambda q, m_out, n_in: -k(q, m_out, n_in)),
        label=f"Osc(omega={omega:g})",
    )


def kuramoto_coupling(coupling: float) -> Arrangement:
    """The stateless coupling box ``Edge_e : <R^0|R^0> -> <R^2 | R^0>``.

    Trivial parameter ``R^0``; two input ports carrying the two reported
    angles, no outputs; forcing ``-dk_e`` for the edge cost
    ``k_e = -coupling * cos(theta_i - theta_j)``. All the dynamics it
    contributes flows back through the wiring as the cotangent pullback of this
    one-form to the two oscillators' parameter covectors.
    """

    def out_f(q: Array, m_out: Array) -> Array:
        return jnp.zeros(0)

    def in_f(q: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.zeros(0)

    def k(q: Array, m_out: Array, n_in: Array) -> Array:
        return -coupling * jnp.cos(n_in[0] - n_in[1])

    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0,
        out_dim_M=0,
        in_dim_N=2,
        out_dim_N=0,
        out_f=out_f,
        in_f=in_f,
        omega=exact(lambda q, m_out, n_in: -k(q, m_out, n_in)),
        label=f"Edge(K={coupling:g})",
    )


def kuramoto_wire(
    num_oscillators: int, edges: Sequence[Tuple[int, int]]
) -> Arrangement:
    """The routing wire as a static arrangement (cf. ``wiring.graph_wire``).

    Image under ``R^-`` (CDLM lem.lens_pow) of the finite-set lens that feeds
    each edge box's two in-ports from its endpoints' broadcast out-ports. Source
    ``<R^{2E} | R^{2E}>`` (the tensor of all boxes' interfaces: ``2E``
    oscillator out-ports, ``2E`` edge in-ports), target the unit ``<R^0|R^0>``,
    trivial parameter, zero forcing. The routing is the bijection ``perm`` of
    ``_incidences``.
    """
    E = len(edges)
    _, perm = _incidences(num_oscillators, edges)
    perm = jnp.asarray(perm, dtype=int)

    def out_f(q_wire: Array, m_out: Array) -> Array:
        return jnp.zeros(0)

    def in_f(q_wire: Array, m_out: Array, n_in: Array) -> Array:
        return m_out[perm]

    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=2 * E,
        out_dim_M=2 * E,
        in_dim_N=0,
        out_dim_N=0,
        out_f=out_f,
        in_f=in_f,
        omega=zero_form,
        label=f"kuramoto_wire(N={num_oscillators}, E={E})",
    )


# ---------------------------------------------------------------------------
# The composite arrangement.
# ---------------------------------------------------------------------------


def kuramoto_arrangement(
    omegas: Sequence[float],
    edges: Sequence[Tuple[int, int]],
    coupling: float,
    sharp_scale: float = 1.0,
    parameter=configuration,
) -> Arrangement:
    """The closed Kuramoto arrangement, by genuine composition in ``arr``:

        compose_seq( tensor(Osc_1, ..., Osc_N, Edge_1, ..., Edge_E),
                     kuramoto_wire(N, edges) )  :  <R^0|R^0> -> <R^0|R^0>,

    with parameter base ``R^N`` (the angles, in oscillator order; the stateless
    edge boxes contribute nothing) and sharp ``sharp_scale * I``. The total
    forcing ``omega = -dk``,

        k(theta) = - sum_i omegas[i] * theta_i
                   - coupling * sum_{(i,j) in edges} cos(theta_i - theta_j),

    emerges from ``compose_seq`` -- it is not written anywhere by hand. Then
    ``Phi`` of the ``configuration`` arrangement (with ``sharp_scale = dt``)
    steps forward-Euler Kuramoto, and ``Phi`` of the ``phase`` one (with
    ``sharp_scale = 1/m``) steps the swing equation
    ``m * ddot(theta)_i = omega_i + K sum_j sin(theta_j - theta_i)`` by
    semi-implicit Euler (unit time step, as in the graph-wave test). They are
    two arrangements on one wiring (def.arr), selected by ``parameter``.
    """
    N = len(omegas)
    degrees, _ = _incidences(N, edges)
    boxes = [
        kuramoto_oscillator(float(omegas[v]), degrees[v], sharp_scale, parameter)
        for v in range(N)
    ] + [kuramoto_coupling(coupling) for _ in edges]
    wired = compose_seq(tensor_arrangements(boxes), kuramoto_wire(N, edges))
    return replace(
        wired, label=f"kuramoto(N={N}, E={len(edges)}, K={coupling:g})"
    )
