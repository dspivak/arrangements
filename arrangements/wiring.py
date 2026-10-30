"""Composition and wiring constructors in ``arr`` (def.arr, sec.forces).

Provides:

* ``compose_seq(f, g)``    -- sequential composition ``g . f`` in ``arr``: lens
  composition on the maps, **pullback-and-sum** on the forcings (Kleisli
  composition for the writer monad ``Omega``, sec.forcing), tensor on the
  parameters.
* ``parallel_arrangements(p1, p2)`` / ``tensor_arrangements(parts)`` -- the
  monoidal product in ``arr``: direct sum of forcings, tensor of parameters.
* ``finset_chain_wire(K)`` -- the lens ``phi_K`` of CDLM ex.lens_finsetop /
  CDLM sec.spring_first_pass, as the underlying pair of finite-set functions.
* ``chain_wire(K)``        -- its image under ``R^-`` (CDLM lem.lens_pow),
  realized as a static arrangement: a ``K``-ary wiring ``box^K -> box`` with
  trivial parameter and zero forcing.
* ``compose_chain(parts)`` -- the ``K``-ary composite
  ``chain_wire(K)(Part_1, ..., Part_K)`` directly in ``arr`` (the book's
  sec.heat_waves chain; CDLM sec.spring_first_pass).
* ``held_end(...)`` / ``cap()`` / ``held_chain(parts, ...)`` -- a chain with its two ends held in
  place by boxes, closed (book sec.heat_waves; plan P43 (b) job 4).
* ``graph_wire(V, edges)`` / ``harmonic_vertex(...)`` / ``compose_graph(...)``
  -- the prism wiring ``R^{varphi_G}``, the vertex particle ``Part_v``, and their
  composite ``wire_G`` (CDLM sec.graph_laplacian).
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from .arrangement import Arrangement, exact, zero_form
from .coalgebra import CotCoalg, configuration
from .rvect import diagonal, trivial


def trivial_parameter() -> CotCoalg:
    """The unit parameter: the configuration coalgebra on ``R^0`` (state ``R^0``)."""
    return configuration(trivial())


# ---------------------------------------------------------------------------
# Sequential composition in arr: pullback-and-sum (sec.forcing, notes/DESIGN.md §2).
# ---------------------------------------------------------------------------


def compose_seq(f: Arrangement, g: Arrangement) -> Arrangement:
    """Sequential composition ``g . f`` in ``arr`` (def.arr).

    For ``f : L -> M`` and ``g : M -> N`` (so ``f.out_dim_N == g.out_dim_M`` and
    ``f.in_dim_N == g.in_dim_M``), the composite ``g . f : L -> N`` has the lens
    composite's forward and backward maps, parameter ``P_f (x) P_g`` (state a
    pair), and forcing the Kleisli composite for the writer monad ``Omega``:
    each factor's one-form pulled back along the map that locates its point, then
    summed. On coordinates, with ``m = out_f(q_f, l)`` the intermediate value,

        out    : n    = out_g(q_g, m)
        in     : l_in = in_f(q_f, l, in_g(q_g, m, n_in))
        omega  : A^* omega_g + B^* omega_f,
                 A : (q, l, n) |-> (q_g, m, n),
                 B : (q, l, n) |-> (q_f, l, in_g(q_g, m, n)).

    When both forcings are exact, ``omega_f = dU_f`` and ``omega_g = dU_g``, the
    chain rule makes this ``d(U_f . B + U_g . A)``, CDLM's "sum the two potentials
    at the composed points"; when they are not, only the pullback form is right.
    """

    if f.out_dim_N != g.out_dim_M or f.in_dim_N != g.in_dim_M:
        raise ValueError(
            f"compose_seq: codomain of f <{f.in_dim_N}|{f.out_dim_N}> "
            f"!= domain of g <{g.in_dim_M}|{g.out_dim_M}>"
        )

    P = f.P.tensor(g.P)
    nf = f.P.Q_dim

    def split(q: Array) -> Tuple[Array, Array]:
        return q[:nf], q[nf:]

    def out_f(q: Array, l_out: Array) -> Array:
        q_f, q_g = split(q)
        m = f.out_f(q_f, l_out)
        return g.out_f(q_g, m)

    def in_f(q: Array, l_out: Array, n_in: Array) -> Array:
        q_f, q_g = split(q)
        m = f.out_f(q_f, l_out)
        m_in = g.in_f(q_g, m, n_in)
        return f.in_f(q_f, l_out, m_in)

    def locate_g(q: Array, l_out: Array, n_in: Array):
        q_f, q_g = split(q)
        return q_g, f.out_f(q_f, l_out), n_in

    def locate_f(q: Array, l_out: Array, n_in: Array):
        q_f, q_g = split(q)
        m = f.out_f(q_f, l_out)
        return q_f, l_out, g.in_f(q_g, m, n_in)

    def omega(q: Array, l_out: Array, n_in: Array):
        at_g, pull_g = jax.vjp(locate_g, q, l_out, n_in)
        at_f, pull_f = jax.vjp(locate_f, q, l_out, n_in)
        a = pull_g(g.omega(*at_g))
        b = pull_f(f.omega(*at_f))
        return tuple(x + y for x, y in zip(a, b))

    return Arrangement(
        P=P,
        in_dim_M=f.in_dim_M,
        out_dim_M=f.out_dim_M,
        in_dim_N=g.in_dim_N,
        out_dim_N=g.out_dim_N,
        out_f=out_f,
        in_f=in_f,
        omega=omega,
        label=f"({g.label} . {f.label})",
    )


# ---------------------------------------------------------------------------
# Monoidal (parallel) composition in arr.
# ---------------------------------------------------------------------------


def parallel_arrangements(p1: Arrangement, p2: Arrangement) -> Arrangement:
    """Monoidal product in ``arr``: side-by-side independent boxes.

    Parameter ``P_1 (x) P_2`` (state a pair), maps componentwise, forcing the
    direct sum of the two one-forms (no pullback).
    """

    P = p1.P.tensor(p2.P)
    n1 = p1.P.Q_dim

    def split(q: Array) -> Tuple[Array, Array]:
        return q[:n1], q[n1:]

    def parts(q, m_out, n_in):
        q1, q2 = split(q)
        m1, m2 = m_out[: p1.out_dim_M], m_out[p1.out_dim_M:]
        n1_in, n2_in = n_in[: p1.in_dim_N], n_in[p1.in_dim_N:]
        return (q1, m1, n1_in), (q2, m2, n2_in)

    def out_f(q: Array, m_out: Array) -> Array:
        q1, q2 = split(q)
        m1, m2 = m_out[: p1.out_dim_M], m_out[p1.out_dim_M:]
        return jnp.concatenate([p1.out_f(q1, m1), p2.out_f(q2, m2)])

    def in_f(q: Array, m_out: Array, n_in: Array) -> Array:
        (q1, m1, n1_in), (q2, m2, n2_in) = parts(q, m_out, n_in)
        return jnp.concatenate([p1.in_f(q1, m1, n1_in), p2.in_f(q2, m2, n2_in)])

    def omega(q: Array, m_out: Array, n_in: Array):
        (q1, m1, n1_in), (q2, m2, n2_in) = parts(q, m_out, n_in)
        a = p1.omega(q1, m1, n1_in)
        b = p2.omega(q2, m2, n2_in)
        return tuple(jnp.concatenate([x, y]) for x, y in zip(a, b))

    return Arrangement(
        P=P,
        in_dim_M=p1.in_dim_M + p2.in_dim_M,
        out_dim_M=p1.out_dim_M + p2.out_dim_M,
        in_dim_N=p1.in_dim_N + p2.in_dim_N,
        out_dim_N=p1.out_dim_N + p2.out_dim_N,
        out_f=out_f,
        in_f=in_f,
        omega=omega,
        label=f"({p1.label} || {p2.label})",
    )


def tensor_arrangements(parts: Sequence[Arrangement]) -> Arrangement:
    """The ``K``-fold monoidal product ``Part_1 (x) ... (x) Part_K`` in ``arr``,
    by folding ``parallel_arrangements`` (state left-nested, as ``nest_left``)."""

    if len(parts) < 1:
        raise ValueError("tensor_arrangements: need at least one factor")
    acc = parts[0]
    for P in parts[1:]:
        acc = parallel_arrangements(acc, P)
    return acc


# ---------------------------------------------------------------------------
# The finset-op lens phi_K (CDLM ex.lens_finsetop, CDLM sec.spring_first_pass).
# ---------------------------------------------------------------------------


def finset_chain_wire(K: int) -> Tuple[List[int], List[int]]:
    """Return ``(out_f, in_f)`` for ``phi_K : <K|K> -> <1|1>``.

    With the inner finite set ``{1,...,K}`` and the outer box ports labeled by
    ``{K}`` (output, right) and ``{0}`` (input, left):

        out_f : {K} -> {1,...,K}        the inclusion, out_f(0) = K-1 (0-indexed),
        in_f  : {1,...,K} -> {1,...,K}+{0}, n |-> n-1 (feed each from the previous wire).
    """
    out_f = [K - 1]               # singleton; selects index of last small box
    in_f = list(range(K))         # picks the first K of K+1 wires
    return out_f, in_f


# ---------------------------------------------------------------------------
# chain_wire(K) as an Arrangement, the image under R^- (CDLM lem.lens_pow).
# ---------------------------------------------------------------------------


def chain_wire(K: int) -> Arrangement:
    """Image of ``phi_K`` under ``R^-``, as a static arrangement.

    Source ``M = box^K = <R^K | R^K>``, target ``N = box = <R | R>``, trivial
    parameter, zero forcing. On coordinates (CDLM sec.spring_first_pass):

        out_f(q_1, ..., q_K)      = q_K
        in_f((q_1, ..., q_K), q_0) = (q_0, q_1, ..., q_{K-1}).
    """

    if K < 1:
        raise ValueError("K must be >= 1")

    def out_f(q_wire: Array, m_out: Array) -> Array:
        return m_out[K - 1:K]  # shape (1,); the last coordinate q_K

    def in_f(q_wire: Array, m_out: Array, n_in: Array) -> Array:
        q_0 = n_in[0]
        return jnp.concatenate([jnp.array([q_0]), m_out[: K - 1]])

    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=K,
        out_dim_M=K,
        in_dim_N=1,
        out_dim_N=1,
        out_f=out_f,
        in_f=in_f,
        omega=zero_form,
        label=f"chain_wire({K})",
    )


def compose_chain(parts: Sequence[Arrangement]) -> Arrangement:
    """The operad composite ``chain_wire(K)(Part_1, ..., Part_K)``, built by genuine
    composition in ``arr``:

        chain_wire(K)(Parts) = compose_seq( tensor(Parts), chain_wire(K) ),

    i.e. tensor the ``K`` particles (``box^0 -> box^K``) and post-compose with the
    wiring (``box^K -> box``). No hand-specialization -- the closed harmonic
    forcing ``-d sum_i (kappa/2)(q_i - q_{i-1})^2`` emerges from ``compose_seq``.
    Each ``Part_i`` must have source ``<R^0|R^0>`` and target ``<R|R>``. The
    composite's state is ``(nest_left([s_1, ..., s_K]), s_wire)`` with
    ``s_wire`` the wiring's trivial state ``R^0``.
    """

    K = len(parts)
    if K < 1:
        raise ValueError("Need at least one particle in chain")
    for i, P in enumerate(parts):
        if not (P.out_dim_M == 0 and P.in_dim_M == 0):
            raise ValueError(f"Part {i}: expected source <R^0|R^0>")
        if not (P.out_dim_N == 1 and P.in_dim_N == 1):
            raise ValueError(f"Part {i}: expected target <R|R>")

    return compose_seq(tensor_arrangements(list(parts)), chain_wire(K))


# ---------------------------------------------------------------------------
# A chain with its ends held (book sec.heat_waves; plan P43 (b) job 4).
# ---------------------------------------------------------------------------


def held_end(parameter, kappa: float = 0.0) -> Arrangement:
    """A held end ``<R^0|R^0> -> <R|R>``: a particle at ``0`` with response ``0``, so it does not move.

    Its integrator is ``parameter`` (the chain's own kind: ``configuration``, ``phase``, ``leapfrog``)
    on ``R`` with the zero sharp, started at rest at ``0``; it reads out its position and pays
    ``(kappa/2)(q - y)^2`` to the neighbor ``y`` on its input, the spring joining it to the chain
    (``kappa = 0``: no spring on that side). The force the chain returns to it moves nothing.
    """
    return Arrangement(
        P=parameter(diagonal(jnp.zeros(1))), in_dim_M=0, out_dim_M=0, in_dim_N=1, out_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -0.5 * kappa * (q[0] - n_in[0]) ** 2),
        label="held",
    )


def cap() -> Arrangement:
    """``<R|R> -> <R^0|R^0>``: closes a chain's two outer ports. The value it presents to the chain's
    input is ``0`` and the force it returns to the chain's output is ``0``; a held end on each side
    makes both inert."""
    return Arrangement(
        P=trivial_parameter(), in_dim_M=1, out_dim_M=1, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(1),
        omega=zero_form, label="cap",
    )


def held_chain(parts: Sequence[Arrangement], kappa: float, parameter=configuration) -> Arrangement:
    """The closed chain ``cap . chain_wire(K+2)(held_left, Part_1, ..., Part_K, held_right)``.

    The left end is a held particle at ``0`` with no spring of its own (the first part's spring
    reaches it); the right end is a held particle at ``0`` joined to the last part by a spring of
    constant ``kappa``. Both ends are boxes of the chain's integrator kind with zero response. The
    composite's position is ``(0, q_1, ..., q_K, 0)``.
    """
    ends = [held_end(parameter)], [held_end(parameter, kappa)]
    return compose_seq(compose_chain(ends[0] + list(parts) + ends[1]), cap())


# ---------------------------------------------------------------------------
# A wiring diagram as the book draws it: boxes, dots, couplers (book Ch 2, sec.springs).
# ---------------------------------------------------------------------------


def copy_wire(src_dim: int, picks: Sequence[int], label: str = "copy") -> Arrangement:
    """The dots of a wiring diagram as one static arrangement ``<R^0|R^src_dim> -> <R^0|R^len(picks)>``,
    ``m |-> m[picks]``.

    A value copied to several ports comes back as the sum of the forces returned to them, since the
    transpose of copying ``x |-> (x, x)`` is addition (book Ch 2, ``sec.springs``). That sum is produced
    by ``compose_seq``'s pullback, not written here. Trivial parameter, zero forcing. (``commons.fan_wire``
    and ``network.fan_box`` are the same wiring, written where they were first needed.)
    """
    idx = jnp.asarray(list(picks), dtype=int)
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0, out_dim_M=src_dim, in_dim_N=0, out_dim_N=len(idx),
        out_f=lambda q, m_out: m_out[idx],
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=zero_form,
        label=label,
    )


def boxes_and_couplers(
    boxes: Sequence[Arrangement], pairs: Sequence[Tuple[int, int]], couplers: Sequence[Arrangement]
) -> Arrangement:
    """The closed arrangement the book's Chapter 2 pictures draw: parts in boxes, each reading out one
    value, wired through dots to two-port couplers (a pipe, a spring, a joint, a resistor) that return a
    force to each end.

    ``boxes[i]`` is a part ``<R^0|R^0> -> <R^0|R>`` (a tank, a rod segment, a mass, a capacitor);
    ``couplers[j]`` is a stateless ``<R^0|R^2> -> <R^0|R^0>`` whose two ports receive the readouts of
    ``boxes[pairs[j][0]]`` and ``boxes[pairs[j][1]]``. The composite is

        compose_seq(compose_seq(tensor(boxes), copy_wire(picks)), tensor(couplers)),

    so a box named in several pairs has its readout copied to each of those couplers and receives the
    sum of the forces they return (``copy_wire``). Its state is the boxes' states in order.
    Two tanks and a pipe: ``boxes_and_couplers([tank_1, tank_2], [(0, 1)], [pipe])``; the three-mass chain:
    ``pairs = [(0, 1), (1, 2)]``; the triangle of ``exc.kcl_loop``: ``pairs = [(0, 1), (1, 2), (0, 2)]``.
    """
    if len(pairs) != len(couplers):
        raise ValueError(f"{len(pairs)} pairs for {len(couplers)} couplers")
    for i, b in enumerate(boxes):
        if not (b.in_dim_M == 0 and b.out_dim_M == 0 and b.in_dim_N == 0 and b.out_dim_N == 1):
            raise ValueError(f"box {i}: expected <R^0|R^0> -> <R^0|R>")
    for j, c in enumerate(couplers):
        if not (c.in_dim_M == 0 and c.out_dim_M == 2 and c.in_dim_N == 0 and c.out_dim_N == 0):
            raise ValueError(f"coupler {j}: expected <R^0|R^2> -> <R^0|R^0>")
    picks = [i for pair in pairs for i in pair]
    wired = compose_seq(tensor_arrangements(list(boxes)), copy_wire(len(boxes), picks))
    return compose_seq(wired, tensor_arrangements(list(couplers)))


# ---------------------------------------------------------------------------
# Graph wiring: the prism varphi_G and the vertex particle (CDLM sec.graph_laplacian).
# ---------------------------------------------------------------------------


def _graph_ports(
    num_vertices: int, edges: Sequence[Tuple[int, int]]
) -> Tuple[List[List[int]], List[List[int]], List[int]]:
    """Out-edge / in-edge index lists per vertex, and the prism permutation.

    For ``edges[e] = (src, tgt)`` and each ``v``, ``out(v) = src^{-1}(v)`` and
    ``in(v) = tgt^{-1}(v)``. Concatenating over ``v`` in order gives two orderings
    of ``E``, ``OUT = sum_v out(v)`` and ``IN = sum_v in(v)`` (each a permutation
    of ``E``). The prism input map -- ``sum_v in(v) -> sum_v out(v)`` via ``tgt``
    then ``src`` -- is the permutation ``perm`` with ``OUT[perm[j]] == IN[j]``
    (same edge at in-port ``j``).
    """
    out_lists = [[e for e, (s, t) in enumerate(edges) if s == v] for v in range(num_vertices)]
    in_lists = [[e for e, (s, t) in enumerate(edges) if t == v] for v in range(num_vertices)]
    OUT = [e for v in range(num_vertices) for e in out_lists[v]]
    IN = [e for v in range(num_vertices) for e in in_lists[v]]
    pos_in_out = {e: k for k, e in enumerate(OUT)}
    perm = [pos_in_out[e] for e in IN]  # in_f(m_out)[j] = m_out[perm[j]]
    return out_lists, in_lists, perm


def graph_wire(
    num_vertices: int, edges: Sequence[Tuple[int, int]], vdim: int = 1
) -> Arrangement:
    """The prism wiring ``R^{varphi_G}`` (CDLM sec.graph_laplacian) as a static arrangement.

    Image under ``R^-`` of the prism ``varphi_G : (x)_v <in(v)/out(v)> -> <0/0>``.
    Each edge port carries ``vdim`` components. Source
    ``(x)_v <R^{vdim*in(v)}|R^{vdim*out(v)}> = <R^{vdim*E}|R^{vdim*E}>``, target the
    unit ``<R^0|R^0>``, trivial parameter, zero forcing. The input map routes
    each in-edge's ``vdim``-block from its source's out-edge block: ``in_f(m_out) =
    m_out[comp_perm]`` where ``comp_perm`` is the edge permutation ``perm``
    (``_graph_ports``) expanded over the ``vdim`` components.
    """
    E = len(edges)
    _, _, perm = _graph_ports(num_vertices, edges)
    perm = jnp.asarray(perm, dtype=int)
    # expand the edge permutation to act block-wise on vdim components per edge:
    comp_perm = (perm[:, None] * vdim + jnp.arange(vdim)).reshape(-1)

    def out_f(q_wire: Array, m_out: Array) -> Array:
        return jnp.zeros(0)  # bang : 0 -> sum_v out(v)

    def in_f(q_wire: Array, m_out: Array, n_in: Array) -> Array:
        return m_out[comp_perm]  # route each in-edge block from its source's out-edge block

    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=E * vdim,
        out_dim_M=E * vdim,
        in_dim_N=0,
        out_dim_N=0,
        out_f=out_f,
        in_f=in_f,
        omega=zero_form,
        label=f"graph_wire(V={num_vertices}, E={E}, vdim={vdim})",
    )


def harmonic_vertex(
    d_in: int, d_out: int, m: float, kappas: Sequence[float], vdim: int = 1, onsite=None,
    parameter=configuration,
) -> Arrangement:
    """The harmonic particle ``Part_v : <1/1> -> <R^{vdim*in(v)}|R^{vdim*out(v)}>`` at a vertex.

    Parameter the coalgebra ``parameter(Q_v)`` (``configuration`` by default,
    ``phase`` for the wave regime) on ``Q_v = R^{vdim}`` with constant
    ``sharp(xi) = xi/m`` (per component); output map the diagonal
    ``q |-> (q)_{e in out(v)}`` broadcasting the ``vdim``-vector position to the
    ``d_out`` out-edges; input map vacuous; forcing exact, ``omega = -dk`` for the
    cost ``k_v(q, (q'_e)) = sum_e (kappa_e/2)|q - q'_e|^2`` (Euclidean per edge)
    over the ``d_in`` in-edges, plus an optional on-site term ``onsite(q)``.
    """
    kappas = jnp.asarray(kappas, float)

    def out_f(q: Array, m_out: Array) -> Array:
        return jnp.tile(q, d_out)  # diagonal: broadcast the vdim-vector to each out-edge

    def in_f(q: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.zeros(0)  # bang

    def k(q: Array, m_out: Array, n_in: Array) -> Array:
        nbr = n_in.reshape(d_in, vdim)  # in-edge neighbor positions
        springs = 0.5 * jnp.sum(kappas * jnp.sum((q[None, :] - nbr) ** 2, axis=1))
        return springs + (onsite(q) if onsite is not None else 0.0)

    return Arrangement(
        P=parameter(diagonal(jnp.full(vdim, 1.0 / m))),
        in_dim_M=0,
        out_dim_M=0,
        in_dim_N=d_in * vdim,
        out_dim_N=d_out * vdim,
        out_f=out_f,
        in_f=in_f,
        omega=exact(lambda q, m_out, n_in: -k(q, m_out, n_in)),
        label="Part_v",
    )


def compose_graph(
    num_vertices: int,
    edges: Sequence[Tuple[int, int]],
    m: float,
    kappa: float,
    vdim: int = 1,
    onsite=None,
    parameter=configuration,
) -> Arrangement:
    """``wire_G = R^{varphi_G}((Part_v)_v)`` (CDLM eqn.graph_wire), by genuine composition.

        wire_G = compose_seq( tensor(Part_v for v), graph_wire(G) ) : <1/1> -> <1/1>.

    A closed 0-ary arrangement with parameter the tensor of the vertices'
    ``parameter(Q_v)`` (state ``(nest_left([s_v]), s_wire)``), and composite
    forcing ``-d sum_e (kappa/2)|q_tgt(e) - q_src(e)|^2`` (Euclidean per edge) plus
    the optional on-site term -- the whole forcing *emerges* from the prism wiring
    composed with the vertex particles, it is not written by hand. Uniform mass
    ``m`` and spring constant ``kappa``.
    """
    out_lists, in_lists, _ = _graph_ports(num_vertices, edges)
    vertices = [
        harmonic_vertex(
            len(in_lists[v]), len(out_lists[v]), m, [kappa] * len(in_lists[v]),
            vdim=vdim, onsite=onsite, parameter=parameter,
        )
        for v in range(num_vertices)
    ]
    wired = compose_seq(tensor_arrangements(vertices), graph_wire(num_vertices, edges, vdim=vdim))
    return replace(wired, label=f"graph(V={num_vertices}, E={len(edges)}, vdim={vdim})")
