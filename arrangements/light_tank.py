"""Light typed as the coil is: capacitor edges and coil faces on the lattice wire of ``light_lattice.py`` (book plan §5.9 item 90,
proposed; moved from the book's run script ``notes/runs/light-12b2681e/light_tank.py``, whose RESULTS.md holds the predictions).

On the ``N x N`` periodic grid each edge is Ch 2 ``sec.kcl``'s capacitor at capacitance 1: state ``E_e``, reads out its voltage
``E_e``. Each face is a coil with four voltage ports: state its flux ``B_f``, internal inclination the signed sum of the four
voltages (``(dE)_f``, Faraday's law as Kirchhoff's voltage law around the square), pushing back ``-+B_f`` on the four ports. The copy
wire is ``lattice_wire``, unchanged, so the pushbacks at an edge add (``-d^T B``, Ampere--Maxwell as Kirchhoff's current law). The
differential equation is ``E' = -d^T B``, ``B' = dE``.

The map to the flux lattice, ``g(A, xi) = (E, B) = (xi, dA)`` (``flux_to_tank``), commutes with waiting: it is a morphism of closed
systems, not of integrands, since ``B`` is not a function of a part. It is not injective (``A`` with ``dA = 0``: gauge) nor onto (on
the torus every ``dA`` has mean 0, while the tank lattice can hold a uniform ``B``).

The rules are parts: ``euler`` (one round), ``rk4`` (Ch 7 ``def.runge_kutta``, four rounds), ``midpoint`` (P76's rule, K rounds), each
lifted by ``lift`` to a K-round integrator from an affine integrand; and the two-round stagger, ``tank_lattice_stagger``, coil first,
capacitors second, which is Yee's scheme and equals the flux lattice's one-round step under ``g``.

Edge order is ``light_lattice.py``'s; a state's leaves are its edges in order and then its faces.

Claude Opus 5.5, session 5c9967d2, 2026-11-05.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .arrangement import Arrangement, exact, zero_form
from .coalgebra import CotCoalgK
from .hopfield import _tensor_balanced
from .light_lattice import face_edges, lattice_wire
from .wiring import compose_seq, trivial_parameter

_z = jnp.zeros(0)
SIGN = jnp.array([1.0, 1.0, -1.0, -1.0])


# ---------------------------------------------------------------------------
# The rules: lifting an affine integrand (read, tangent response) to a K-round integrator. ``read(s)`` is the configuration,
# ``v(s, load)`` the tangent response.
# ---------------------------------------------------------------------------


def euler(read, v, tau):
    """One round, the plain step: ``(reads, advances)``."""
    return ((read,), (lambda s, xi: s + tau * v(s, xi),))


def rk4(read, v, tau):
    """Ch 7 ``def.runge_kutta``, four rounds: ``(reads, advances)``. The intermediate is (s, the rates so far)."""
    A = (0.5, 0.5, 1.0)

    def r0(s):
        return read(s)

    def a0(s, xi):
        return (s, v(s, xi), None, None)

    def r1(m):
        return read(m[0] + A[0] * tau * m[1])

    def a1(m, xi):
        return (m[0], m[1], v(m[0] + A[0] * tau * m[1], xi), None)

    def r2(m):
        return read(m[0] + A[1] * tau * m[2])

    def a2(m, xi):
        return (m[0], m[1], m[2], v(m[0] + A[1] * tau * m[2], xi))

    def r3(m):
        return read(m[0] + A[2] * tau * m[3])

    def a3(m, xi):
        k4 = v(m[0] + A[2] * tau * m[3], xi)
        return m[0] + tau / 6.0 * (m[1] + 2 * m[2] + 2 * m[3] + k4)

    return ((r0, r1, r2, r3), (a0, a1, a2, a3))


def midpoint(read, v, tau, K):
    """P76's rule, K rounds: read out a trial midpoint, receive there, improve the trial; the last round steps. The intermediate is
    (s, trial). K = 1 is ``euler``."""
    if K == 1:
        return euler(read, v, tau)

    def r0(s):
        return read(s)

    def a0(s, xi):
        return (s, s + 0.5 * tau * v(s, xi))

    def r(m):
        return read(m[1])

    def a(m, xi):
        return (m[0], m[0] + 0.5 * tau * v(m[1], xi))

    def last(m, xi):
        return m[0] + tau * v(m[1], xi)

    return ((r0,) + (r,) * (K - 1), (a0,) + (a,) * (K - 2) + (last,))


def lift(rule, read, v, s0, Q_dim, tau, K=1) -> CotCoalgK:
    """The integrator of ``rule`` (``"euler"``, ``"rk4"`` or ``"mid"``; ``K`` is the midpoint's rounds) on an affine integrand."""
    if rule == "euler":
        reads, advances = euler(read, v, tau)
    elif rule == "rk4":
        reads, advances = rk4(read, v, tau)
    elif rule == "mid":
        reads, advances = midpoint(read, v, tau, K)
    else:
        raise ValueError(rule)
    return CotCoalgK(Q_dim=Q_dim, init=jnp.asarray(s0, float), reads=reads, advances=advances, label=rule)


# ---------------------------------------------------------------------------
# The tank lattice: capacitor edges, coil faces.
# ---------------------------------------------------------------------------


def tank_edge(P) -> Arrangement:
    """``sec.kcl``'s capacitor at capacitance 1 on an edge: it reads out its voltage and receives a current."""
    return Arrangement(P=P, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
                       out_f=lambda q, m: q, in_f=lambda q, m, n: _z, omega=zero_form, label="cap")


def coil_face(P) -> Arrangement:
    """A coil with four voltage ports: state its flux ``B``, internal inclination the signed sum of the voltages, pushback ``-+B`` on
    the ports. At two ports with signs (+, -) it is ``magnetism.coil_draft(1, now=True)``."""

    def omega(q, m, n):
        return (jnp.array([jnp.dot(SIGN, n)]), _z, -SIGN * q[0])

    return Arrangement(P=P, in_dim_M=0, out_dim_M=0, in_dim_N=4, out_dim_N=0,
                       out_f=lambda q, m: _z, in_f=lambda q, m, n: _z, omega=omega, label="coil4")


READ_1 = lambda s: s
V_1 = lambda s, xi: xi


def tank_lattice(N, E0, B0, rule, tau, K=1) -> Arrangement:
    """The closed composite ``lattice_wire(N)(edges, faces)`` with each part stepped by ``rule``."""
    parts = [tank_edge(lift(rule, READ_1, V_1, [E0[e]], 1, tau, K)) for e in range(2 * N * N)]
    parts += [coil_face(lift(rule, READ_1, V_1, [B0[f]], 1, tau, K)) for f in range(N * N)]
    return compose_seq(_tensor_balanced(parts), lattice_wire(N))


def tank_edge_stagger(s0, tau) -> Arrangement:
    """Two rounds: round 1 reads out the voltage and discards the load; round 2 reads out the voltage and steps by the current that
    came back."""
    P = CotCoalgK(Q_dim=1, init=jnp.asarray([s0], float), reads=(READ_1, READ_1),
                  advances=((lambda s, xi: s), (lambda s, xi: s + tau * xi)), label="edge2")
    return tank_edge(P)


def coil_face_stagger(s0, tau) -> Arrangement:
    """Two rounds: round 1 reads out ``B``, moves it by the inclination; round 2 reads out the new ``B``, whose pushback is the new
    current, and keeps it."""
    P = CotCoalgK(Q_dim=1, init=jnp.asarray([s0], float), reads=(READ_1, READ_1),
                  advances=((lambda s, xi: s + tau * xi), (lambda s, xi: s)), label="face2")
    return coil_face(P)


def tank_lattice_stagger(N, E0, B0, tau) -> Arrangement:
    """Coil first, capacitors second: Yee's scheme. Equal under ``g`` to the one-round flux lattice at the same ``tau``."""
    parts = [tank_edge_stagger(E0[e], tau) for e in range(2 * N * N)]
    parts += [coil_face_stagger(B0[f], tau) for f in range(N * N)]
    return compose_seq(_tensor_balanced(parts), lattice_wire(N))


# ---------------------------------------------------------------------------
# The mass-and-spring lattice with a duration and rounds: light_lattice.py's typing.
# ---------------------------------------------------------------------------


def flux_edge(P) -> Arrangement:
    """An edge of the flux lattice, on an integrator with state ``(A, xi)``; it reads out one value."""
    return Arrangement(P=P, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
                       out_f=lambda q, m: q, in_f=lambda q, m, n: _z, omega=zero_form, label="mass")


def spring_face() -> Arrangement:
    """A stateless face, cost ``B^2 / 2`` on ``B = dA``: ``light_lattice.face_box`` at ``kappa = 1``."""

    def k(q, m, n):
        B = jnp.dot(SIGN, n)
        return 0.5 * B * B

    return Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=0, in_dim_N=4, out_dim_N=0,
                       out_f=lambda q, m: _z, in_f=lambda q, m, n: _z,
                       omega=exact(lambda q, m, n: -k(q, m, n)), label="spring4")


READ_2 = lambda s: s[:1]
V_2 = lambda s, xi: jnp.array([s[1], xi[0]])


def flux_symplectic_edge(a0, x0, tau) -> Arrangement:
    """Ch 3's mass, ``eqn.mass_duration_update``, one round."""
    P = CotCoalgK(Q_dim=1, init=jnp.asarray([a0, x0], float),
                  reads=(lambda s: jnp.array([s[0] + tau * s[1]]),),
                  advances=(lambda s, xi: jnp.array([s[0] + tau * s[1], s[1] + tau * xi[0]]),), label="mass1")
    return flux_edge(P)


def flux_lattice(N, A0, X0, rule, tau, K=1) -> Arrangement:
    """The closed mass-and-spring composite, each edge stepped by ``rule`` (or ``"symplectic"``, the mass's own one round)."""
    if rule == "symplectic":
        edges = [flux_symplectic_edge(A0[e], X0[e], tau) for e in range(2 * N * N)]
    else:
        edges = [flux_edge(lift(rule, READ_2, V_2, [A0[e], X0[e]], 1, tau, K)) for e in range(2 * N * N)]
    faces = [spring_face() for _ in range(N * N)]
    return compose_seq(_tensor_balanced(edges + faces), lattice_wire(N))


# ---------------------------------------------------------------------------
# The operators, the map, and reading states off. Checks only: nothing here steps a system.
# ---------------------------------------------------------------------------


def curl_matrix(N) -> np.ndarray:
    """``d`` from edges to faces, ``B = dA``, in ``face_edges``'s signs."""
    C = np.zeros((N * N, 2 * N * N))
    for i in range(N):
        for j in range(N):
            f = i * N + j
            for s, e in zip((1, 1, -1, -1), face_edges(N, i, j)):
                C[f, e] += s
    return C


def grad_matrix(N) -> np.ndarray:
    """``d`` from nodes to edges. Gauss: ``(div E)_n`` is the transpose of this applied to ``E``; ``curl_matrix @ grad_matrix = 0``."""
    D = np.zeros((2 * N * N, N * N))
    for i in range(N):
        for j in range(N):
            ax = i * N + j
            ay = N * N + i * N + j
            D[ax, ((i + 1) % N) * N + j] += 1
            D[ax, i * N + j] -= 1
            D[ay, i * N + (j + 1) % N] += 1
            D[ay, i * N + j] -= 1
    return D


def flux_to_tank(N, A, xi):
    """The map ``g(A, xi) = (E, B) = (xi, dA)`` from the flux lattice's states to the tank lattice's."""
    return np.asarray(xi), curl_matrix(N) @ np.asarray(A)


def leaves(s) -> np.ndarray:
    """The state's nonempty leaves, flat."""
    return np.concatenate([np.ravel(np.asarray(x, float)) for x in jax.tree_util.tree_leaves(s) if np.asarray(x).size])


def tank_read(N, s):
    """``(E, B)`` off a state of the tank lattice."""
    x = leaves(s)
    return x[: 2 * N * N], x[2 * N * N:]


def flux_read(s):
    """``(A, xi)`` off a state of the flux lattice."""
    x = leaves(s).reshape(-1, 2)
    return x[:, 0], x[:, 1]


def energy_tank(E, B):
    """``E^2 / 2 + B^2 / 2``."""
    return 0.5 * E @ E + 0.5 * B @ B


def mode_start(N, kx_index=1, ky_index=0):
    """An eigenmode's flux ``B = cos(k . r)``, with ``E = 0``, and its ``omega = 2 sqrt(sin^2(kx/2) + sin^2(ky/2))``."""
    k = (2 * np.pi * kx_index / N, 2 * np.pi * ky_index / N)
    B = np.array([np.cos(k[0] * i + k[1] * j) for i in range(N) for j in range(N)])
    w = 2 * np.sqrt(np.sin(k[0] / 2) ** 2 + np.sin(k[1] / 2) ** 2)
    return B, w
