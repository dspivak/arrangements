"""What moves and makes light on a lattice: a charge in contact with the field, and a clock part driving one edge (book plan §5.9
item 90, proposed; moved from the book's run scripts ``notes/runs/light-12b2681e/charge_on_lattice.py`` and ``sun.py``, whose RESULTS.md
holds the predictions).

*The charge on the lattice.* On the ``N x N`` periodic grid each edge is the two-port capacitor stepped by Ch 7 ``def.runge_kutta``, state
``(A, E)``; a charge is a mass reading out position and velocity; one contact part, ``field_and_contact``, hears every edge and the charge. It
returns to the charge ``e (E(x) + v x B(x))``, with ``E(x)``, ``B(x)`` by Whitney's shapes from the edge voltages and the flux around the cell,
and to each edge the current ``-e v . W_e(x)``. The field's springs are gathered into that one box, for speed: they are
``light_lattice.py``'s face boxes, summed by hand. The push and the current are the two sides of one exchange of power, so the box is not
exact, and sits outside Ch 5's rest-is-Lagrangian and potential functors.

*The source.* ``source_lattice_3d`` is the mass-and-spring lattice one dimension up, with ``clock_part`` (time its state) returning a current to
one edge: an input varying with time arrives through a part with a clock. ``loop_step`` is the plain loop that stands beside it, checked equal to
it at ``N = 4`` (``tests/test_light_sources.py``).

Claude Opus 5.5, session 5c9967d2, 2026-11-05.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .arrangement import Arrangement, zero_form
from .coalgebra import CotCoalgK
from .hopfield import _tensor_balanced
from .light_lattice import edge_ax, edge_ay, face_edges
from .light_tank import flux_symplectic_edge, rk4, spring_face
from .wiring import compose_seq, trivial_parameter

_z = jnp.zeros(0)
SIGN = jnp.array([1.0, 1.0, -1.0, -1.0])


# ---------------------------------------------------------------------------
# A charge and the field moving each other.
# ---------------------------------------------------------------------------


def rk4_coalg(read, v, tau, s0, Q_dim, label) -> CotCoalgK:
    """Ch 7 ``def.runge_kutta`` for an affine integrand on an array state."""
    reads, advances = rk4(read, v, tau)
    return CotCoalgK(Q_dim=Q_dim, init=jnp.asarray(s0, float), reads=reads, advances=advances, label=label)


def field_edge(a0, e0, tau) -> Arrangement:
    """The two-port capacitor as an edge, under Runge--Kutta: state ``(A, E)``, rates ``(E, load on either port)``; it reads out ``(A, E)``."""
    read = lambda s: s
    v = lambda s, xi: jnp.array([s[1], xi[0] + xi[1]])
    P = rk4_coalg(read, v, tau, [a0, e0], 2, "edge")
    return Arrangement(P=P, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=2,
                       out_f=lambda q, m: q, in_f=lambda q, m, n: _z, omega=zero_form, label="edge")


def charge_part(x0, v0, tau, m_ch: float = 1.0) -> Arrangement:
    """A mass reading out position and velocity, under Runge--Kutta; a push on the position coordinates changes the momentum."""
    read = lambda s: jnp.concatenate([s[:2], s[2:] / m_ch])
    v = lambda s, xi: jnp.concatenate([s[2:] / m_ch, xi[:2]])
    P = rk4_coalg(read, v, tau, list(x0) + [m_ch * v0[0], m_ch * v0[1]], 4, "charge")
    return Arrangement(P=P, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=4,
                       out_f=lambda q, m: q, in_f=lambda q, m, n: _z, omega=zero_form, label="charge")


def face_table(N: int):
    """The four edges of each face, in ``face_edges``'s order."""
    return jnp.asarray([face_edges(N, i, j) for i in range(N) for j in range(N)], dtype=int)


def faces_cost_load(table, A):
    """Springs on ``B = dA``: their load on the flux ports, ``-d^T dA``, and ``B``."""
    B = (A[table] * SIGN).sum(axis=1)
    g = jnp.zeros(A.shape[0])
    g = g.at[table.ravel()].add((B[:, None] * SIGN).ravel())
    return -g, B


def whitney(N: int, x):
    """Cell ``(i, j)``, fractions ``(fx, fy)``. The x-edges ``Ax[i, j]``, ``Ax[i, j+1]`` weigh ``(1-fy)``, ``fy``; the y-edges ``Ay[i, j]``, ``Ay[i+1, j]``
    weigh ``(1-fx)``, ``fx``. These are Whitney's 1-forms in a square."""
    xm = jnp.mod(x, N)
    i, j = jnp.floor(xm[0]).astype(int), jnp.floor(xm[1]).astype(int)
    fx, fy = xm[0] - i, xm[1] - j
    return i, j, fx, fy


def field_and_contact(N: int, e_ch: float = 1.0) -> Arrangement:
    """The contact part with the field's springs gathered in. It hears, for each edge, its flux and its voltage, and the charge's ``(x, y, vx, vy)``.
    It returns on the flux ports the springs' load; on the voltage ports the charge's current ``-e v . W_e(x)``; and on the charge's position
    coordinates the push ``e (E(x) + v x B(x))``. Not exact: the push and the current are the two sides of one exchange of power."""
    NE = 2 * N * N
    n_in = 2 * NE + 4
    table = face_table(N)

    def omega(q, m, n):
        A = m[0:2 * NE:2]
        E = m[1:2 * NE:2]
        x = m[2 * NE:2 * NE + 2]
        v = m[2 * NE + 2:2 * NE + 4]
        i, j, fx, fy = whitney(N, x)
        idx_x = jnp.array([edge_ax(N, i, j), edge_ax(N, i, j + 1)])
        w_x = jnp.array([1 - fy, fy])
        idx_y = jnp.array([edge_ay(N, i, j), edge_ay(N, i + 1, j)])
        w_y = jnp.array([1 - fx, fx])
        Ex_at = (E[idx_x] * w_x).sum()
        Ey_at = (E[idx_y] * w_y).sum()
        spring, B = faces_cost_load(table, A)
        # E here is the rate of A, so the physical potential is -A and the physical field is B_z = -(dA)_f (RESULTS's D0).
        Bz = -B[(i % N) * N + (j % N)]
        fxx = e_ch * (Ex_at + v[1] * Bz)
        fyy = e_ch * (Ey_at - v[0] * Bz)
        load_v = jnp.zeros(NE)
        load_v = load_v.at[idx_x].add(-e_ch * v[0] * w_x)
        load_v = load_v.at[idx_y].add(-e_ch * v[1] * w_y)
        load_m = jnp.zeros(n_in)
        load_m = load_m.at[0:2 * NE:2].set(spring)
        load_m = load_m.at[1:2 * NE:2].set(load_v)
        load_m = load_m.at[2 * NE].set(fxx)
        load_m = load_m.at[2 * NE + 1].set(fyy)
        return (q, load_m, _z)

    return Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=n_in, in_dim_N=0, out_dim_N=0,
                       out_f=lambda q, m: _z, in_f=lambda q, m, n: _z, omega=omega, label="field_and_contact")


def charge_on_lattice(N: int, x0, v0, tau, e_ch: float = 1.0, m_ch: float = 1.0) -> Arrangement:
    """The closed composite: every edge at rest with ``E = 0``, the charge, and ``field_and_contact`` over them."""
    NE = 2 * N * N
    parts = [field_edge(0.0, 0.0, tau) for _ in range(NE)] + [charge_part(x0, v0, tau, m_ch)]
    return compose_seq(_tensor_balanced(parts), field_and_contact(N, e_ch))


def charge_read(N: int, s):
    """``(A, E, x, p)`` off a state of ``charge_on_lattice``."""
    NE = 2 * N * N
    flat = np.concatenate([np.ravel(np.asarray(x, float)) for x in jax.tree_util.tree_leaves(s) if np.asarray(x).size])
    AE = flat[: 2 * NE].reshape(-1, 2)
    return AE[:, 0], AE[:, 1], flat[2 * NE:2 * NE + 2], flat[2 * NE + 2:2 * NE + 4]


def charge_energy(N: int, A, E, p, m_ch: float = 1.0):
    """The field's energy and the charge's kinetic energy. A check, read off a state."""
    B = (A[np.asarray(face_table(N))] * np.asarray(SIGN)).sum(axis=1)
    return 0.5 * (E @ E) + 0.5 * (B @ B) + 0.5 * (p @ p) / m_ch


# ---------------------------------------------------------------------------
# A source on a 3D grid: the clock part, and the plain loop beside it.
# ---------------------------------------------------------------------------


def clock_part(tau, w, label: str = "clock") -> Arrangement:
    """A part with a clock: its state is the time; it returns the current ``sin(w t)`` to the edge it hears."""
    P = CotCoalgK(Q_dim=1, init=jnp.array([0.0]), reads=(lambda t: t,), advances=(lambda t, xi: t + tau,), label=label)

    def omega(q, m, n):
        return (jnp.zeros(1), _z, jnp.sin(w * q))

    return Arrangement(P=P, in_dim_M=0, out_dim_M=0, in_dim_N=1, out_dim_N=0,
                       out_f=lambda q, m: _z, in_f=lambda q, m, n: _z, omega=omega, label=label)


def edge_3d(N: int, family: int, i: int, j: int, k: int) -> int:
    """Flat index of the edge of ``family`` (0, 1, 2 for x, y, z) at ``(i, j, k)``, periodic."""
    return family * N ** 3 + (i % N) * N * N + (j % N) * N + (k % N)


def faces_3d(N: int) -> list:
    """The four edges of each face, in the sign order ``(+, +, -, -)``: the xy family, then yz, then zx, each row-major."""
    ax = lambda i, j, k: edge_3d(N, 0, i, j, k)
    ay = lambda i, j, k: edge_3d(N, 1, i, j, k)
    az = lambda i, j, k: edge_3d(N, 2, i, j, k)
    cells = [(i, j, k) for i in range(N) for j in range(N) for k in range(N)]
    faces = [(ax(i, j, k), ay(i + 1, j, k), ax(i, j + 1, k), ay(i, j, k)) for i, j, k in cells]
    faces += [(ay(i, j, k), az(i, j + 1, k), ay(i, j, k + 1), az(i, j, k)) for i, j, k in cells]
    faces += [(az(i, j, k), ax(i, j, k + 1), az(i + 1, j, k), ax(i, j, k)) for i, j, k in cells]
    return faces


def lattice3d_wire(N: int, src: int, faces) -> Arrangement:
    """The copy wire: the clock hears the edge ``src``, each face its four edges."""
    route = jnp.asarray([src] + [e for f in faces for e in f], dtype=int)
    return Arrangement(P=trivial_parameter(), in_dim_M=int(route.shape[0]), out_dim_M=3 * N ** 3, in_dim_N=0, out_dim_N=0,
                       out_f=lambda q, m: _z, in_f=lambda q, m, n: m[route], omega=zero_form, label="lattice3d")


def source_lattice_3d(N: int, A0, E0, src: int, tau, w) -> Arrangement:
    """The closed composite: the clock part, then ``3 N^3`` edges (Ch 3's mass, one round), then ``3 N^3`` spring faces, under the copy wire. The state's
    leaves are the clock's time and then each edge's ``(A, E)``. The clock goes first: the library's tensor takes a many-round part on the left of a
    plain one, not the reverse."""
    faces = faces_3d(N)
    parts = [clock_part(tau, w)] + [flux_symplectic_edge(float(A0[e]), float(E0[e]), tau) for e in range(3 * N ** 3)]
    parts += [spring_face() for _ in faces]
    return compose_seq(_tensor_balanced(parts), lattice3d_wire(N, src, faces))


def curl_3d(A):
    """``B = dA`` on the three families of faces, the arrays ``A = (Ax, Ay, Az)``, by ``np.roll``. The plain loop's."""
    Ax, Ay, Az = A
    Bxy = Ax + np.roll(Ay, -1, 0) - np.roll(Ax, -1, 1) - Ay
    Byz = Ay + np.roll(Az, -1, 1) - np.roll(Ay, -1, 2) - Az
    Bzx = Az + np.roll(Ax, -1, 2) - np.roll(Az, -1, 0) - Ax
    return Bxy, Byz, Bzx


def curl_T_3d(B):
    """``d^T`` on the three families of faces. The plain loop's."""
    Bxy, Byz, Bzx = B
    gx = Bxy - np.roll(Bxy, 1, 1) + np.roll(Bzx, 1, 2) - Bzx
    gy = Byz - np.roll(Byz, 1, 2) + np.roll(Bxy, 1, 0) - Bxy
    gz = Bzx - np.roll(Bzx, 1, 0) + np.roll(Byz, 1, 1) - Byz
    return gx, gy, gz


def loop_step(A, E, t, tau, J):
    """A plain loop's one step, beside ``source_lattice_3d``, for large ``N``: the edge reads out where it is headed, the springs push back there, the clock
    returns its current ``J(E, t, tau)`` (which adds to ``E`` in place); then the charge takes it. Not an arrangement: it stands only where checked
    equal to the composite (``tests/test_light_sources.py``, S1)."""
    A = [a + tau * e for a, e in zip(A, E)]
    g = curl_T_3d(curl_3d(A))
    E = [e - tau * x for e, x in zip(E, g)]
    J(E, t, tau)
    return A, E, t + tau
