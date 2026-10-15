"""A spot that feeds and divides: a ring of cells fed from a reservoir, composed (book P27, copying).

David, 2026-10-12: "a cell collects the nutrients it needs. And then when there's enough … it divides … we're in a medium
where there's like lots of stuff floating around." The chemistry is Gray and Scott's, whose spots grow and split in two
(Pearson 1993; in one dimension Reynolds, Pearson and Ponce-Dawson 1994; recalled, not read). Each piece is a box:

* a CELL per site, a small volume of the medium: state ``(u, v)``, the amounts of food and of builder, with the
  configuration integrator (response ``dt``); it sends ``(u, v)`` four times, to its exchange, its reaction and its two
  joints;
* an EXCHANGE per cell, the volume's traffic with the reservoir (the gel fed from a stirred tank, as in the experiments):
  stateless, cost ``(F/2)(u - 1)^2 + ((F + k)/2) v^2``, so food flows in toward the reservoir's 1 and builder is washed
  out and decays;
* a REACTION per cell, ``U + 2V -> 3V`` at rate ``u v^2``, one-way: ``reaction_network.reaction_box``, a 1-form written
  directly, not the differential of any cost;
* a JOINT per neighboring pair, the boundary molecules cross: stateless, cost
  ``(D_u/2)(u_i - u_j)^2 + (D_v/2)(v_i - v_j)^2``, so each amount moves down its difference.

``ring_wire`` routes each cell's copies to their boxes; ``spot_ring`` is ``ring_wire . (cells (x) boxes)``, closed, and
``Phi`` runs it. Its position is ``(u_0, v_0, u_1, v_1, ...)``. Predictions and verdicts: book repo,
``notes/runs/p27-copying-c2fac8e7/``; tests in ``tests/test_dividing_spot.py``.
"""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
import numpy as np

from .arrangement import Arrangement, exact, zero_form
from .coalgebra import configuration
from .reaction_network import reaction_box
from .rvect import diagonal
from .wiring import compose_seq, parallel_arrangements, trivial_parameter

DEGREE = 4  # exchange, reaction, joint to the left, joint to the right


def cell_box(u0: float, v0: float, dt: float) -> Arrangement:
    """A cell ``I -> <0 | 8>``: state ``(u, v)``, response ``dt``, sending ``(u, v)`` to each of its four boxes."""
    return Arrangement(
        P=configuration(diagonal(jnp.array([dt, dt])), init=jnp.array([float(u0), float(v0)])),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=2 * DEGREE,
        out_f=lambda q, m: jnp.tile(q, DEGREE), in_f=lambda q, m, n: jnp.zeros(0), omega=zero_form,
        label="cell",
    )


def exchange_box(F: float, k: float) -> Arrangement:
    """The exchange ``I -> <2 | 0>`` on ``(u, v)``: cost ``(F/2)(u - 1)^2 + ((F + k)/2) v^2``."""
    return Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=0, in_dim_N=2, out_dim_N=0,
        out_f=lambda q, m: jnp.zeros(0), in_f=lambda q, m, n: jnp.zeros(0),
        omega=exact(lambda q, m, n: -(0.5 * F * (n[0] - 1.0) ** 2 + 0.5 * (F + k) * n[1] ** 2)),
        label="exchange",
    )


def joint_box(Du: float, Dv: float) -> Arrangement:
    """The joint ``I -> <4 | 0>`` on ``(u_i, v_i, u_j, v_j)``: cost ``(D_u/2)(u_i - u_j)^2 + (D_v/2)(v_i - v_j)^2``."""
    return Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=0, in_dim_N=4, out_dim_N=0,
        out_f=lambda q, m: jnp.zeros(0), in_f=lambda q, m, n: jnp.zeros(0),
        omega=exact(lambda q, m, n: -(0.5 * Du * (n[0] - n[2]) ** 2 + 0.5 * Dv * (n[1] - n[3]) ** 2)),
        label="joint",
    )


def growth_box() -> Arrangement:
    """The reaction ``U + 2V -> 3V`` on ``(u, v)``, one-way, rate constant 1."""
    return reaction_box(jnp.array([1.0, 2.0]), jnp.array([0.0, 3.0]), 1.0, 0.0)


def ring_wire(N: int) -> Arrangement:
    """The wiring ``<8N | 8N> -> I``: the inputs of the exchanges, reactions and joints, in that order, fed from the cells.

    Cell ``i``'s copies are slots ``8i + 2s + (0, 1)`` for ``s`` = 0 exchange, 1 reaction, 2 the joint to its left
    (joint ``i - 1``, where it is the second cell), 3 the joint to its right (joint ``i``, where it is the first).
    Static, zero forcing; its backward map is the transpose, so each box's push returns to the cell it read.
    """
    def slot(i, s):
        b = 2 * DEGREE * (i % N) + 2 * s
        return [b, b + 1]

    perm = []
    for i in range(N):
        perm += slot(i, 0)
    for i in range(N):
        perm += slot(i, 1)
    for i in range(N):
        perm += slot(i, 3) + slot(i + 1, 2)
    perm = jnp.asarray(perm, dtype=int)
    P = 2 * DEGREE * N
    return Arrangement(
        P=trivial_parameter(), in_dim_M=P, out_dim_M=P, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m: jnp.zeros(0), in_f=lambda q, m, n: m[perm], omega=zero_form, label=f"ring_wire({N})",
    )


def balanced_tensor(parts: Sequence[Arrangement]) -> Arrangement:
    """``parts[0] (x) ... (x) parts[-1]``, bracketed as a balanced tree so the nesting is ``log`` deep, not linear."""
    parts = list(parts)
    if len(parts) == 1:
        return parts[0]
    mid = len(parts) // 2
    return parallel_arrangements(balanced_tensor(parts[:mid]), balanced_tensor(parts[mid:]))


def spot_ring(u0, v0, *, F: float = 0.04, k: float = 0.06, Du: float = 0.512, Dv: float = 0.256,
              dt: float = 0.5) -> Arrangement:
    """The closed ring ``ring_wire . (cells (x) exchanges (x) reactions (x) joints)``, cells at ``(u0, v0)``."""
    N = len(u0)
    boxes = ([cell_box(u0[i], v0[i], dt) for i in range(N)]
             + [exchange_box(F, k) for _ in range(N)]
             + [growth_box() for _ in range(N)]
             + [joint_box(Du, Dv) for _ in range(N)])
    return compose_seq(balanced_tensor(boxes), ring_wire(N))


def seed(N: int, width: int = 10):
    """The run's start: ``width`` cells at the ring's middle at ``(0.5, 0.25)``, the rest at the reservoir's ``(1, 0)``."""
    u, v = np.ones(N), np.zeros(N)
    c = N // 2
    u[c - width // 2:c + width // 2] = 0.5
    v[c - width // 2:c + width // 2] = 0.25
    return u, v


def amounts(arr: Arrangement, state):
    """``(u, v)``, each an array over the cells, read off the ring's position."""
    q = np.asarray(arr.P.position(state))
    return q[0::2], q[1::2]


def spots(v, threshold: float = 0.1):
    """The spots: maximal runs of cells around the ring with builder above ``threshold``, as ``(start, end)`` pairs."""
    m = np.asarray(v) > threshold
    N = len(m)
    if m.all() or not m.any():
        return [(0, N - 1)] if m.all() else []
    off = int(np.argmin(m))  # a cell below threshold, so no run wraps past it
    runs, start = [], None
    for j in range(1, N + 1):
        i = (off + j) % N
        if m[i] and start is None:
            start = off + j
        if not m[i] and start is not None:
            runs.append((start % N, (off + j - 1) % N))
            start = None
    return runs


def numpy_step(u, v, *, F=0.04, k=0.06, Du=0.512, Dv=0.256, dt=0.5):
    """The same step written as one numpy formula, for checking the composite against (not a model of its own)."""
    lu = np.roll(u, 1) + np.roll(u, -1) - 2 * u
    lv = np.roll(v, 1) + np.roll(v, -1) - 2 * v
    r = u * v * v
    return u + dt * (Du * lu - r + F * (1 - u)), v + dt * (Dv * lv + r - (F + k) * v)


def state_at(arr: Arrangement, u, v):
    """The ring's state with the cells at ``(u, v)`` (the cells' leaves come first, in order; the rest are empty)."""
    import jax

    leaves, tree = jax.tree_util.tree_flatten(arr.P.init)
    N = len(u)
    new = [jnp.array([float(u[i]), float(v[i])]) for i in range(N)] + leaves[N:]
    return jax.tree_util.tree_unflatten(tree, new)


def run_counting(step, state, steps: int, every: int):
    """``step`` taken ``steps`` times by ``jax.lax.scan`` in blocks of ``every``; returns the final state and the state
    after each block (as a list)."""
    import jax

    def body(s, _):
        return step(s), None

    block = jax.jit(lambda s: jax.lax.scan(body, s, None, length=every)[0])
    out = []
    for _ in range(steps // every):
        state = block(state)
        out.append(state)
    return state, out


# --- The pattern: heredity from relaxation alone (book P52 (b)) ---------------------------------------------------------
#
# Each cell carries a third chemical, the pattern ``p``, and gets one more box, the WELL: a 1-form pushing ``p`` by
# ``-v p (p^2 - 1)``, the ``p``-derivative of ``(v/4)(p^2 - 1)^2`` with the builder ``v`` held, so the two wells exist only where
# a spot is. Written directly, not exact: the exact cost would also push ``v`` by ``-(p^2 - 1)^2 / 4``, whatever ``v`` is.
# ``p`` diffuses across each joint like the builder and has no exchange with the reservoir. The reaction may be biased by the
# pattern, rate ``(1 + eps p) u v^2``. Predictions and verdicts: book repo ``notes/runs/p52-heredity-49346cc0/``;
# tests in ``tests/test_heredity.py``.

PDEGREE = 5  # exchange, reaction, well, joint to the left, joint to the right


def pattern_cell_box(u0: float, v0: float, p0: float, dt: float) -> Arrangement:
    """A cell ``I -> <0 | 15>``: state ``(u, v, p)``, response ``dt``, sending ``(u, v, p)`` to each of its five boxes."""
    return Arrangement(
        P=configuration(diagonal(jnp.array([dt, dt, dt])), init=jnp.array([float(u0), float(v0), float(p0)])),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=3 * PDEGREE,
        out_f=lambda q, m: jnp.tile(q, PDEGREE), in_f=lambda q, m, n: jnp.zeros(0), omega=zero_form,
        label="cell",
    )


def pattern_exchange_box(F: float, k: float) -> Arrangement:
    """The exchange ``I -> <3 | 0>`` on ``(u, v, p)``: cost ``(F/2)(u - 1)^2 + ((F + k)/2) v^2``; no term in ``p``."""
    return Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=0, in_dim_N=3, out_dim_N=0,
        out_f=lambda q, m: jnp.zeros(0), in_f=lambda q, m, n: jnp.zeros(0),
        omega=exact(lambda q, m, n: -(0.5 * F * (n[0] - 1.0) ** 2 + 0.5 * (F + k) * n[1] ** 2)),
        label="exchange",
    )


def pattern_reaction_box(eps: float = 0.0) -> Arrangement:
    """The reaction ``U + 2V -> 3V`` on ``(u, v, p)``, one-way, at rate ``(1 + eps p) u v^2``; it pushes ``p`` by nothing.

    With ``eps = 0`` it is ``growth_box`` with a third, unused input. A 1-form written directly, as ``reaction_box``."""
    def omega(q, m, n):
        r = (1.0 + eps * n[2]) * n[0] * n[1] ** 2
        return jnp.zeros_like(q), jnp.zeros_like(m), jnp.array([-r, r, 0.0])

    return Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=0, in_dim_N=3, out_dim_N=0,
        out_f=lambda q, m: jnp.zeros(0), in_f=lambda q, m, n: jnp.zeros(0), omega=omega,
        label="reaction",
    )


def well_box() -> Arrangement:
    """The well ``I -> <3 | 0>`` on ``(u, v, p)``: pushes ``p`` by ``-v p (p^2 - 1)`` and ``u``, ``v`` by nothing.

    Not exact: its ``p``-component depends on ``v`` while its ``v``-component is 0."""
    def omega(q, m, n):
        return jnp.zeros_like(q), jnp.zeros_like(m), jnp.array([0.0, 0.0, -n[1] * n[2] * (n[2] ** 2 - 1.0)])

    return Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=0, in_dim_N=3, out_dim_N=0,
        out_f=lambda q, m: jnp.zeros(0), in_f=lambda q, m, n: jnp.zeros(0), omega=omega,
        label="well",
    )


def pattern_joint_box(Du: float, Dv: float, Dp: float) -> Arrangement:
    """The joint ``I -> <6 | 0>`` on ``(u_i, v_i, p_i, u_j, v_j, p_j)``: cost the sum of ``(D/2)(x_i - x_j)^2`` per chemical."""
    D = jnp.array([Du, Dv, Dp])
    return Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=0, in_dim_N=6, out_dim_N=0,
        out_f=lambda q, m: jnp.zeros(0), in_f=lambda q, m, n: jnp.zeros(0),
        omega=exact(lambda q, m, n: -0.5 * jnp.sum(D * (n[:3] - n[3:]) ** 2)),
        label="joint",
    )


def pattern_ring_wire(N: int) -> Arrangement:
    """The wiring ``<15N | 15N> -> I``: the inputs of the exchanges, reactions, wells and joints, in that order, from the cells.

    Cell ``i``'s copies are slots ``15i + 3s + (0, 1, 2)`` for ``s`` = 0 exchange, 1 reaction, 2 well, 3 the joint to its
    left (joint ``i - 1``, where it is the second cell), 4 the joint to its right (joint ``i``, where it is the first).
    Static, zero forcing; its backward map is the transpose, as ``ring_wire``."""
    def slot(i, s):
        b = 3 * PDEGREE * (i % N) + 3 * s
        return [b, b + 1, b + 2]

    perm = []
    for s in (0, 1, 2):
        for i in range(N):
            perm += slot(i, s)
    for i in range(N):
        perm += slot(i, 4) + slot(i + 1, 3)
    perm = jnp.asarray(perm, dtype=int)
    P = 3 * PDEGREE * N
    return Arrangement(
        P=trivial_parameter(), in_dim_M=P, out_dim_M=P, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m: jnp.zeros(0), in_f=lambda q, m, n: m[perm], omega=zero_form,
        label=f"pattern_ring_wire({N})",
    )


def pattern_ring(u0, v0, p0, *, F: float = 0.04, k: float = 0.06, Du: float = 0.512, Dv: float = 0.256,
                 Dp: float = 0.256, eps: float = 0.0, dt: float = 0.5) -> Arrangement:
    """The closed ring ``pattern_ring_wire . (cells (x) exchanges (x) reactions (x) wells (x) joints)``."""
    N = len(u0)
    boxes = ([pattern_cell_box(u0[i], v0[i], p0[i], dt) for i in range(N)]
             + [pattern_exchange_box(F, k) for _ in range(N)]
             + [pattern_reaction_box(eps) for _ in range(N)]
             + [well_box() for _ in range(N)]
             + [pattern_joint_box(Du, Dv, Dp) for _ in range(N)])
    return compose_seq(balanced_tensor(boxes), pattern_ring_wire(N))


def pattern_seed(N: int, signs, centers=None, width: int = 10, medium: float = 0.0):
    """The run's start: a ``seed`` spot of ``width`` cells at each center, its cells at ``p = sign``; every other cell at
    the reservoir's ``(1, 0)`` and ``p = medium``. One center defaults to the ring's middle, two to opposite sides."""
    if centers is None:
        centers = [N // 2] if len(signs) == 1 else [N // 4 + j * N // len(signs) for j in range(len(signs))]
    u, v, p = np.ones(N), np.zeros(N), np.full(N, float(medium))
    for c, s in zip(centers, signs):
        idx = np.arange(c - width // 2, c + width // 2) % N
        u[idx], v[idx], p[idx] = 0.5, 0.25, float(s)
    return u, v, p


def pattern_amounts(arr: Arrangement, state):
    """``(u, v, p)``, each an array over the cells, read off the ring's position."""
    q = np.asarray(arr.P.position(state))
    return q[0::3], q[1::3], q[2::3]


def pattern_state_at(arr: Arrangement, u, v, p):
    """The ring's state with the cells at ``(u, v, p)``."""
    import jax

    leaves, tree = jax.tree_util.tree_flatten(arr.P.init)
    N = len(u)
    new = [jnp.array([float(u[i]), float(v[i]), float(p[i])]) for i in range(N)] + leaves[N:]
    return jax.tree_util.tree_unflatten(tree, new)


def spot_signs(v, p, threshold: float = 0.1):
    """Each spot's mean pattern over its cells, in the order ``spots`` gives them."""
    N = len(v)
    return [float(np.mean(np.take(p, range(a, a + (b - a) % N + 1), mode="wrap"))) for a, b in spots(v, threshold)]


def numpy_pattern_step(u, v, p, *, F=0.04, k=0.06, Du=0.512, Dv=0.256, Dp=0.256, eps=0.0, dt=0.5):
    """The same step as one numpy formula, for checking the composite against (not a model of its own)."""
    lap = lambda x: np.roll(x, 1) + np.roll(x, -1) - 2 * x
    r = (1 + eps * p) * u * v * v
    return (u + dt * (Du * lap(u) - r + F * (1 - u)),
            v + dt * (Dv * lap(v) + r - (F + k) * v),
            p + dt * (Dp * lap(p) - v * p * (p * p - 1)))
