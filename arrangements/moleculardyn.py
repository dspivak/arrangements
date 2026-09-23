"""Molecular dynamics as leapfrog phase dynamics of a wired arrangement.

EXTENSION (an *instance*, not new machinery). The claim under test: the
standard MD simulation loop -- velocity-Verlet integration of point particles
under pairwise Lennard-Jones forces -- is the two-stage dynamics ``Phi2``
(pc2.py, CDLM rmk.multistage) of an arrangement whose particles carry the
leapfrog coalgebra ``leapfrog(Q)`` (leapfrog.py) as parameter, wired from

* **particle boxes** ``Part_i : <R^0|R^0> -> <R^{d*c_i}|R^0>`` -- parameter
  the coalgebra ``parameter(Q_i)`` (``leapfrog`` by default) on ``Q_i = R^d``
  (the position, ``d in {1, 2}``) with *constant* mass-like sharp
  ``sharpR(xi) = (dt^2/m) xi`` (see below), output map the diagonal broadcast
  of the position to the ``c_i`` pair boxes the particle participates in,
  vacuous input map, zero forcing;
* **stateless LJ pair boxes** ``LJ_p : <R^0|R^0> -> <R^0|R^{2d}>`` -- trivial
  parameter ``R^0`` (no state), no output, input the two reported positions
  ``(r, r')``, and forcing exact, ``omega = -dU_LJ`` for the Lennard-Jones
  pair energy

      U_LJ(r, r') = 4 eps ((sigma/|r - r'|)^12 - (sigma/|r - r'|)^6);

* the **prism wiring** ``graph_wire`` (CDLM eqn.prism_f) on the bipartite graph
  particles -> pair-boxes, one edge per membership, routing each particle's
  broadcast out-port onto the pair boxes' in-ports.

The closed composite (``compose_seq(tensor(parts), graph_wire(...))``, exactly
the ``compose_graph`` pattern of CDLM sec.graph_laplacian) is a 0-ary arrangement
(def:arr) with parameter base ``Q = (R^d)^N`` whose total forcing

    omega(q) = -d sum_{pairs p = (i,j)} U_LJ(q_i, q_j)

*emerges* from the composition (pullback-and-sum, sec.forcing) -- it is not
written by hand. The potential it is exact for, ``md_potential``, is written out
here only to define the total energy; the tests check that the composite forcing
is its negative differential.

Where the timestep lives
------------------------

The framework's integrators take a *unit* step; there is no ``dt`` slot. The
physical timestep therefore enters through the reactive metric: the particle
sharp is the constant ``(dt^2/m) I`` and the stored momentum coordinate is
``xi = (m/dt) v``. Under this identification one ``Phi2`` macro-tick of the
leapfrog coalgebra (in the book's sign, the covector ``xi_Q = -dU`` is added)

    xi_half = xi + (1/2) xi_Q(q);  q' = q + sharpR(xi_half);
    xi'     = xi_half + (1/2) xi_Q(q')

is *literally* one velocity-Verlet step of timestep ``dt``,

    v_half = v - (dt/2m) dU(q);  q' = q + dt v_half;
    v'     = v_half - (dt/2m) dU(q'),

with the two force evaluations at ``q`` and ``q'`` being the two ``pc^(2)``
rounds; and the kinetic energy *via the sharp* is exact:

    (1/2) <xi, sharpR(xi)> = (1/2) (m/dt)^2 |v|^2 (dt^2/m) = (1/2) m |v|^2.

Equivalently: the sharp is ``xi / m_eff`` with effective mass ``m_eff = m/dt^2``
(time measured in ticks). The arrangement's forcing is ``-d`` of the *physical*
LJ energy, unscaled, so the parameter covector ``xi_Q`` that the interpretation
hands the coalgebra (CDLM eqn.bigtheta) is exactly ``-dU``, the physical force.

State shape
-----------

The composite's parameter is the tensor of the particles' coalgebras with the
trivial parameters of the pair boxes and the wiring, so its state is
``(nest_left([s_1, ..., s_N, z, ..., z]), z)`` with ``z = R^0``;
``md_state`` / ``md_flat`` convert between that shape and flat arrays ``q``
(and ``xi``) over ``(R^d)^N``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, List, Sequence, Tuple

import jax.numpy as jnp
from jax import Array

from .arrangement import Arrangement, exact, zero_form
from .interpretation import interpretation, trivial_omega
from .leapfrog import leapfrog
from .polynomial import nest_left, unnest_left
from .rvect import diagonal
from .wiring import compose_seq, graph_wire, tensor_arrangements, trivial_parameter


# ---------------------------------------------------------------------------
# The two generators: particle box and stateless LJ pair box.
# ---------------------------------------------------------------------------


def particle_box(
    mass: float, dt: float, n_pairs: int, dim: int = 1, parameter=leapfrog
) -> Arrangement:
    """The particle ``Part_i : <R^0|R^0> -> <R^{dim*n_pairs}|R^0>``.

    Parameter the coalgebra ``parameter(Q_i)`` (``leapfrog`` by default;
    ``phase`` or ``configuration`` for the single-stage dynamics) on
    ``Q_i = R^dim`` (the position) with constant sharp ``(dt^2/mass) I``
    (mass-like; see module docstring for why ``dt^2`` lives here). Output map
    broadcasts the position to the ``n_pairs`` pair boxes the particle feeds;
    input map vacuous; forcing zero -- all interaction energy lives in the
    pair boxes.
    """

    def out_f(q: Array, m_out: Array) -> Array:
        return jnp.tile(q, n_pairs)  # diagonal broadcast, as in harmonic_vertex

    def in_f(q: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.zeros(0)  # bang

    return Arrangement(
        P=parameter(diagonal(jnp.full(dim, dt * dt / mass))),
        in_dim_M=0,
        out_dim_M=0,
        in_dim_N=0,
        out_dim_N=dim * n_pairs,
        out_f=out_f,
        in_f=in_f,
        omega=zero_form,
        label=f"Part(m={mass:g})",
    )


def lj_pair_energy(r: Array, rp: Array, eps: float, sigma: float) -> Array:
    """The Lennard-Jones pair energy ``4 eps ((sigma/|r-r'|)^12 - (sigma/|r-r'|)^6)``.

    Computed via ``s6 = (sigma^2/|r-r'|^2)^3`` so no square root appears
    (smooth away from the ``r = r'`` singularity, which LJ genuinely has).
    """
    d = r - rp
    r2 = jnp.sum(d * d)
    s6 = (sigma * sigma / r2) ** 3
    return 4.0 * eps * (s6 * s6 - s6)


def lj_pair_box(eps: float, sigma: float, dim: int = 1) -> Arrangement:
    """The stateless pair-interaction box ``LJ_p : <R^0|R^0> -> <R^0|R^{2*dim}>``.

    Trivial parameter ``R^0`` (no state, nothing for a coalgebra to update),
    no output ports, ``2*dim`` input ports receiving the two reported particle
    positions, and forcing ``omega = -dU_LJ`` for the Lennard-Jones energy of
    their separation. The entire content of the box is its forcing.
    """

    def out_f(q: Array, m_out: Array) -> Array:
        return jnp.zeros(0)  # bang

    def in_f(q: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.zeros(0)  # bang

    def U(q: Array, m_out: Array, n_in: Array) -> Array:
        return lj_pair_energy(n_in[:dim], n_in[dim:], eps, sigma)

    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0,
        out_dim_M=0,
        in_dim_N=2 * dim,
        out_dim_N=0,
        out_f=out_f,
        in_f=in_f,
        omega=exact(lambda q, m_out, n_in: -U(q, m_out, n_in)),
        label=f"LJ(eps={eps:g}, sigma={sigma:g})",
    )


# ---------------------------------------------------------------------------
# The wired composite: particles + pair boxes + prism wiring.
# ---------------------------------------------------------------------------


def _edges(num_particles: int, pairs: Sequence[Tuple[int, int]]):
    N = num_particles
    counts = [0] * N
    edges: List[Tuple[int, int]] = []
    for p, (i, j) in enumerate(pairs):
        if not (0 <= i < N and 0 <= j < N and i != j):
            raise ValueError(f"pair {p} = ({i}, {j}) is not a pair of distinct particles")
        edges.append((i, N + p))
        edges.append((j, N + p))
        counts[i] += 1
        counts[j] += 1
    return counts, edges


def md_arrangement(
    num_particles: int,
    pairs: Sequence[Tuple[int, int]],
    mass: float,
    eps: float,
    sigma: float,
    dt: float,
    dim: int = 1,
    parameter=leapfrog,
) -> Arrangement:
    """The closed MD arrangement: particles wired into LJ pair boxes.

    Vertices of the wiring graph are the ``N = num_particles`` particle boxes
    followed by the ``P = len(pairs)`` pair boxes; for pair ``p = (i, j)`` the
    edges ``(i, N+p)`` and ``(j, N+p)`` carry the two reported positions (each
    a ``dim``-block) into the pair box. The composite

        compose_seq( tensor(Part_1, ..., Part_N, LJ_1, ..., LJ_P),
                     graph_wire(N + P, edges, vdim=dim) )

    is a closed 0-ary arrangement with parameter the tensor of the particles'
    ``parameter(Q_i)`` (``leapfrog`` by default: its dynamics is ``pc2.Phi2``;
    with ``phase`` or ``configuration`` it is ``functors.Phi``) on base
    ``(R^dim)^N`` with block sharp ``(dt^2/mass) I``, and emergent forcing
    ``-d sum_p U_LJ(q_i, q_j)``.
    """
    N, P = num_particles, len(pairs)
    counts, edges = _edges(N, pairs)
    parts = [particle_box(mass, dt, counts[i], dim=dim, parameter=parameter) for i in range(N)]
    ljs = [lj_pair_box(eps, sigma, dim=dim) for _ in range(P)]
    wired = compose_seq(
        tensor_arrangements(parts + ljs), graph_wire(N + P, edges, vdim=dim)
    )
    return replace(wired, label=f"md(N={N}, P={P}, dim={dim})")


def md_potential(
    num_particles: int, pairs: Sequence[Tuple[int, int]], eps: float, sigma: float, dim: int = 1
) -> Callable[[Array], Array]:
    """The total LJ energy ``U(q) = sum_p U_LJ(q_i, q_j)`` on ``(R^dim)^N``, as a function of the flat ``q``.

    The composite arrangement carries the forcing ``-dU``, not ``U`` (def:arr);
    this is the potential written out, for the energy ``md_energy`` and for the
    tests that check the composite forcing against it.
    """
    _edges(num_particles, pairs)

    def U(q: Array) -> Array:
        qs = q.reshape(num_particles, dim)
        return sum(lj_pair_energy(qs[i], qs[j], eps, sigma) for i, j in pairs)

    return U


# ---------------------------------------------------------------------------
# The composite's state shape.
# ---------------------------------------------------------------------------


def md_state(num_particles: int, num_pairs: int, q: Array, xi: Array = None):
    """The composite's state from flat ``q`` (and ``xi``) over ``(R^dim)^N``.

    ``(nest_left([s_1, ..., s_N, z, ..., z]), z)`` with ``s_i = q_i`` for the
    configuration coalgebra and ``s_i = (q_i, xi_i)`` for the phase and leapfrog
    coalgebras, ``z = R^0`` the trivial state of each pair box and of the wiring.
    """
    q = jnp.asarray(q)
    dim = q.shape[0] // num_particles
    blocks = [q[i * dim:(i + 1) * dim] for i in range(num_particles)]
    if xi is not None:
        xi = jnp.asarray(xi)
        blocks = [(b, xi[i * dim:(i + 1) * dim]) for i, b in enumerate(blocks)]
    z = jnp.zeros(0)
    return (nest_left(blocks + [z] * num_pairs), z)


def md_flat(state, num_particles: int, num_pairs: int):
    """Inverse of ``md_state``: flat ``q`` (configuration) or ``(q, xi)`` (phase/leapfrog)."""
    parts, _ = state
    blocks = unnest_left(parts, num_particles + num_pairs)[:num_particles]
    if isinstance(blocks[0], tuple):
        return jnp.concatenate([b for b, _ in blocks]), jnp.concatenate([x for _, x in blocks])
    return jnp.concatenate(blocks)


# ---------------------------------------------------------------------------
# Framework-level observables: the force covector and the total energy.
# ---------------------------------------------------------------------------


def md_force(arr: Arrangement, q: Array) -> Array:
    """The physical force ``xi_Q`` at ``q``: ONE covector evaluation through
    the framework.

    Runs the direction action of the interpretation (sec.lens_internalization,
    CDLM eqn.bigtheta) of the closed arrangement at parameter position ``q``
    with the trivial boundary data, and returns ``xi_Q`` -- the parameter
    covector every coalgebra consumes. Since the arrangement's forcing is
    ``-dU`` for the physical LJ energy, this is the physical force, with no
    ``dt`` factor (CDLM's integrator subtracted, so its force was ``-xi_Q``;
    STATUS.md §1).
    """
    _, direction_action = interpretation(arr)(jnp.asarray(q))
    xi_Q, _, _ = direction_action(
        jnp.zeros(0), trivial_omega(0), jnp.zeros(0), jnp.zeros(0)
    )
    return xi_Q


def md_energy(q: Array, xi: Array, mass: float, dt: float, U: Callable[[Array], Array]) -> Array:
    """Total energy: kinetic via the sharp plus the potential.

        H(q, xi) = (1/2) <xi, sharpR_q(xi)> + U(q)

    With the particle sharp ``(dt^2/m) I`` and ``xi = (m/dt) v`` the kinetic
    term equals ``(1/2) m |v|^2`` exactly, and ``U`` (``md_potential``) is the
    sum of LJ pair energies whose ``-d`` is the emergent forcing, so ``H`` is
    the physical total energy. The sharp lives inside the particle coalgebra's
    step and the composite carries a forcing, not a potential (def:arr), so both
    are passed here explicitly rather than read off the arrangement.
    """
    ke = 0.5 * (dt * dt / mass) * jnp.dot(xi, xi)
    return ke + U(q)
