"""Light on a lattice as an arrangement built by composition (book plan P32 (a), §5.9 item 31).

The spring chain one dimension up. On an ``N x N`` periodic grid each EDGE is a box holding one
number ``A_e`` (the vector potential along that edge) with the ``phase`` parameter and unit
mass; each FACE is a stateless box whose cost is ``(kappa/2) B_f^2`` with
``B_f = A_bottom + A_right - A_top - A_left`` the signed sum of its four edges (``B = dA``, the
discrete curl); a routing wire copies each edge's value to the two faces it borders. The
composite

    light_lattice(N) = compose_seq( tensor(edges + faces), lattice_wire(N) )

is closed, and its forcing ``-d sum_f (kappa/2) B_f^2 = -kappa C^T C A`` EMERGES from
``compose_seq``'s pullback-and-sum through the copy wire: it is written nowhere by hand, the
face cost being the only formula in this file (the ``compose_graph`` pattern of wiring.py, with
a wire that is a copy rather than a permutation). ``Phi`` of it is the phase rule
``A' = A + xi; xi' = xi - kappa C^T C A'``, which is ``scripts/maxwell2d.py``'s step at
``dt = sqrt(kappa)`` with momenta ``xi = dt * x_script`` (the library has no ``dt``; STATUS.md
§1). The predictions checked against it are in ``notes/P32-A-PREDICTIONS-2026-09-29.md`` and
``tests/test_light_lattice.py``.

Edge order in the composite's flat position ``q`` (and in ``pack_state``): the ``N^2``
x-edges ``Ax[i, j]`` (from ``(i, j)`` to ``(i+1, j)``) row-major, then the ``N^2`` y-edges
``Ay[i, j]`` (from ``(i, j)`` to ``(i, j+1)``), matching the script's two arrays.
Face ``(i, j)`` reads, in order, ``Ax[i, j]`` (+), ``Ay[i+1, j]`` (+), ``Ax[i, j+1]`` (-),
``Ay[i, j]`` (-), the script's ``curl``.

Written by Claude Fable 5.1, session ec5ece87, 2026-09-29.
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from .arrangement import Arrangement, exact, zero_form
from .coalgebra import phase
from .hopfield import _tensor_balanced
from .rvect import diagonal
from .wiring import compose_seq, trivial_parameter


# ---------------------------------------------------------------------------
# Indexing: the script's two arrays flattened, x-edges then y-edges.
# ---------------------------------------------------------------------------


def edge_ax(N: int, i: int, j: int) -> int:
    """Flat index of the x-edge ``Ax[i, j]`` (periodic)."""
    return (i % N) * N + (j % N)


def edge_ay(N: int, i: int, j: int) -> int:
    """Flat index of the y-edge ``Ay[i, j]`` (periodic)."""
    return N * N + (i % N) * N + (j % N)


def face_edges(N: int, i: int, j: int) -> List[int]:
    """The four edges face ``(i, j)`` reads, in the sign order ``(+, +, -, -)`` of the script's
    ``curl``: ``Ax[i, j] + Ay[i+1, j] - Ax[i, j+1] - Ay[i, j]``."""
    return [edge_ax(N, i, j), edge_ay(N, i + 1, j), edge_ax(N, i, j + 1), edge_ay(N, i, j)]


# ---------------------------------------------------------------------------
# The boxes.
# ---------------------------------------------------------------------------


def edge_box(m: float = 1.0) -> Arrangement:
    """An edge: ``<R^0|R^0> -> <R^0 | R^1>``, parameter ``phase(Q)`` on ``Q = R`` with
    ``sharp(xi) = xi/m``; it emits its value ``A_e`` and reads nothing; zero forcing (every
    force on it arrives by pullback from the faces it feeds)."""
    return Arrangement(
        P=phase(diagonal(jnp.array([1.0 / m]))),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=zero_form,
        label="Edge",
    )


def face_box(kappa: float = 1.0) -> Arrangement:
    """A face: ``<R^0|R^0> -> <R^4 | R^0>``, trivial parameter (stateless), four in-ports
    ``n = (A_bottom, A_right, A_top, A_left)``, cost ``k = (kappa/2)(n_0 + n_1 - n_2 - n_3)^2
    = (kappa/2) B^2``, forcing ``omega = -dk`` (STATUS.md §1). The one formula in the model."""
    def k(q: Array, m_out: Array, n_in: Array) -> Array:
        B = n_in[0] + n_in[1] - n_in[2] - n_in[3]
        return 0.5 * kappa * B * B

    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0, out_dim_M=0, in_dim_N=4, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -k(q, m_out, n_in)),
        label="Face",
    )


def lattice_route(N: int) -> jnp.ndarray:
    """``route[4 f + s]`` = the edge whose value face ``f = i N + j`` reads at its slot ``s``."""
    return jnp.asarray([e for i in range(N) for j in range(N) for e in face_edges(N, i, j)], dtype=int)


def lattice_wire(N: int) -> Arrangement:
    """The routing wire ``<R^{4 N^2} | R^{2 N^2}> -> <R^0 | R^0>``: trivial parameter, zero
    forcing, input map ``m_out |-> m_out[route]`` copying each edge's out-port to the in-ports
    of the two faces it borders. Unlike ``graph_wire`` this is a copy, not a permutation, so
    the transposed routing in ``compose_seq`` SUMS the two faces' pulls on each edge."""
    route = lattice_route(N)
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=4 * N * N, out_dim_M=2 * N * N, in_dim_N=0, out_dim_N=0,
        out_f=lambda q_wire, m_out: jnp.zeros(0),
        in_f=lambda q_wire, m_out, n_in: m_out[route],
        omega=zero_form,
        label=f"lattice_wire(N={N})",
    )


def light_lattice(N: int = 8, kappa: float = 1.0, m: float = 1.0) -> Arrangement:
    """The closed composite ``lattice_wire(N)(edges, faces)`` by genuine composition in ``arr``:
    ``2 N^2`` edge boxes then ``N^2`` face boxes tensored (balanced fold, hopfield.py's
    ``_tensor_balanced``, the same product as ``tensor_arrangements``), post-composed with the
    wire. Position ``q`` = the ``2 N^2`` edge values in the order above; the forcing is
    ``-kappa C^T C q``, produced by pullback-and-sum. The script's step at ``dt`` is
    ``kappa = dt^2, m = 1``."""
    parts = [edge_box(m) for _ in range(2 * N * N)] + [face_box(kappa) for _ in range(N * N)]
    wired = compose_seq(_tensor_balanced(parts), lattice_wire(N))
    return replace(wired, label=f"light_lattice(N={N}, kappa={kappa:g}, m={m:g})")


# ---------------------------------------------------------------------------
# State helpers: the composite's pytree state <-> flat (A, xi).
# ---------------------------------------------------------------------------


def pack_state(arr: Arrangement, A, xi):
    """The composite's state with edge values ``A`` and momenta ``xi`` (flat, edge order).
    The state's leaves in tree order are ``(q_e, xi_e)`` per edge, then the faces' and the
    wire's empty states, so the flat leaf vector interleaves ``A`` and ``xi``."""
    flat = jnp.stack([jnp.asarray(A, float), jnp.asarray(xi, float)], axis=1).reshape(-1)
    leaves, treedef = jax.tree_util.tree_flatten(arr.P.init)
    pieces, i = [], 0
    for leaf in leaves:
        n = int(jnp.size(leaf))
        pieces.append(jnp.reshape(flat[i:i + n], jnp.shape(leaf)))
        i += n
    if i != flat.shape[0]:
        raise ValueError(f"pack_state: {flat.shape[0]} coordinates for a state of size {i}")
    return jax.tree_util.tree_unflatten(treedef, pieces)


def unpack_state(state) -> Tuple[Array, Array]:
    """Inverse of ``pack_state``: ``(A, xi)`` flat in edge order."""
    flat = jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(state)])
    pairs = flat.reshape(-1, 2)
    return pairs[:, 0], pairs[:, 1]
