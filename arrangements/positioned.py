"""The positioned operad: parts that send their own positions (book plan P77 (b)).

Everything here is the agents' and *proposed*; the predictions and results are in the book repository,
``notes/runs/p77b-b62b56fe/RESULTS.md``.

Fix a position space ``Pos = R^e``. A position is sent as a *message* of ``Pos``: ``(m z, m)`` in ``R^{e+1}``, mass
times position and the mass, so that positions add as messages and the value of a sum is the centre of mass (Ch 4
``prop.positive_messages`` (c)).

**Colors** are value dimensions ``d``. The interface of a box of color ``d`` receives a message in ``R^{d+1}`` and sends
its position message in ``R^{e+1}`` and a message in ``R^{d+1}``. This is the addressed operad of ``addressed_one.py``
at one place per slot, the place being the position (``iota`` below), on one-slot colors.

**Operations** (``PosOp``) ``d_1, ..., d_n -> d_0``: an integrator; a position readout ``place(q, Z)``, where ``Z`` lists
the members' sent position messages; transfers ``T_ij(q, Z)`` for ``i, j`` in ``0..n`` (``T_00 = 0``) and the matter's
messages ``T_iV(q, Z)``, cone maps as in Ch 4 ``def.attention_arrangement``, here also reading ``Z``; a cost.

**Substitution** (``substitute``) is ``eqn.attention_substitution`` row by row, each part's data read at its own
members' position messages and the parent's at the positions its parts send.

``cone_readout`` gives the readouts of the form ``g_V(q) + sum_i a_i(q) Z_i`` with ``a_i >= 0``: the group's matter's
own position message plus its members' (David, 2026-10-29: "maybe we should do both?"). All ``a_i = 0`` is a door, a
place of the group's own; all ``a_i = 1`` with silent matter is the centre of mass. ``ear_op`` hears by attenuation
``beta * d_eps`` between the positions of an operation's boxes, ``d_eps(x) = sqrt(|x|^2 + eps^2)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import jax.numpy as jnp
from jax import Array

from .addressed_one import AddrOp, OpData
from .arrangement import Arrangement, exact
from .coalgebra import CotCoalg, configuration
from .rvect import trivial

V = "v"   # the index of the matter as a sender


def zero_cost(q: Array, m_out: Array, n_in: Array) -> Array:
    return jnp.asarray(0.0)


def _zeros(rows, cols=None):
    return jnp.zeros(rows) if cols is None else jnp.zeros((rows, cols))


def value(g: Array) -> Array:
    """The position a position message ``(m z, m)`` names, ``z``."""
    return g[:-1] / g[-1]


def at(z, mass=1.0) -> Array:
    """The position message of mass ``mass`` at ``z``."""
    z = jnp.asarray(z, dtype=float)
    return jnp.concatenate([mass * z, jnp.asarray([mass], dtype=float)])


@dataclass(frozen=True)
class PosOp:
    """``phi : d_1, ..., d_n -> d_0`` (module docstring); ``transfers(q, Z)`` a dict ``(i, j) -> matrix`` with ``(i, V)``
    a vector; a missing key is zero."""

    dims: Tuple[int, ...]
    e: int
    P: CotCoalg
    place: Callable[[Array, List[Array]], Array]
    transfers: Callable[[Array, List[Array]], Dict]
    cost: Callable[[Array, Array, Array], Array] = zero_cost
    label: str = ""

    @property
    def n(self) -> int:
        return len(self.dims) - 1

    def T(self, q, Z):
        T = self.transfers(q, Z)
        d = self.dims

        def get(i, j):
            if j == V:
                v = T.get((i, V))
                return _zeros(d[i] + 1) if v is None else v
            v = T.get((i, j))
            return _zeros(d[i] + 1, d[j] + 1) if v is None else v
        return get

    def split(self, m_out):
        """Each member's sent position message and message."""
        Z, ms, c = [], [], 0
        for d in self.dims[1:]:
            Z.append(m_out[c:c + self.e + 1])
            ms.append(m_out[c + self.e + 1:c + self.e + d + 2])
            c += self.e + d + 2
        return Z, ms

    def sigma(self, q, m_out):
        Z, ms = self.split(m_out)
        T = self.T(q, Z)
        out = T(0, V)
        for j in range(1, self.n + 1):
            out = out + T(0, j) @ ms[j - 1]
        return jnp.concatenate([self.place(q, Z), out])

    def rho(self, q, m_out, n_in):
        if self.n == 0:
            return jnp.zeros(0)
        Z, ms = self.split(m_out)
        T = self.T(q, Z)
        out = []
        for i in range(1, self.n + 1):
            m = T(i, V) + T(i, 0) @ n_in
            for j in range(1, self.n + 1):
                m = m + T(i, j) @ ms[j - 1]
            out.append(m)
        return jnp.concatenate(out)

    def arr(self) -> Arrangement:
        return Arrangement(P=self.P, in_dim_M=sum(d + 1 for d in self.dims[1:]),
                           out_dim_M=sum(self.e + d + 2 for d in self.dims[1:]),
                           in_dim_N=self.dims[0] + 1, out_dim_N=self.e + self.dims[0] + 2,
                           out_f=self.sigma, in_f=self.rho,
                           omega=exact(lambda q, m_out, n_in: -self.cost(q, m_out, n_in)),
                           label=self.label or f"pos{self.dims}")


def unit(d: int, e: int) -> PosOp:
    """One member, forwarded: its own position, ``T_01 = T_10 = id``."""
    I = jnp.eye(d + 1)
    return PosOp(dims=(d, d), e=e, P=configuration(trivial()), place=lambda q, Z: Z[0],
                 transfers=lambda q, Z: {(0, 1): I, (1, 0): I}, label=f"1_{d}")


def substitute(phi: PosOp, psis: Sequence[PosOp], label: str = "") -> PosOp:
    """``phi(psi_1, ..., psi_n)``; configuration order ``Q_{psi_1} x ... x Q_{psi_n} x Q_phi`` (as ``compose_seq``)."""
    psis = list(psis)
    n = phi.n
    if len(psis) != n:
        raise ValueError("arity")
    for i, p in enumerate(psis):
        if p.dims[0] != phi.dims[i + 1] or p.e != phi.e:
            raise ValueError("colors")
    boxes = [(i, ii) for i in range(n) for ii in range(1, psis[i].n + 1)]
    dims = (phi.dims[0],) + tuple(psis[i].dims[ii] for i, ii in boxes)
    qd = [p.P.Q_dim for p in psis]
    qcut = [sum(qd[:i]) for i in range(n + 1)]
    zcut = [sum(p.n for p in psis[:i]) for i in range(n + 1)]
    Pint = None
    for p in psis:
        Pint = p.P if Pint is None else Pint.tensor(p.P)
    Pint = phi.P if Pint is None else Pint.tensor(phi.P)

    def split(q, Z):
        qs = [q[qcut[i]:qcut[i + 1]] for i in range(n)]
        Zs = [list(Z[zcut[i]:zcut[i + 1]]) for i in range(n)]
        return qs, Zs, q[qcut[-1]:]

    def parts_at(q, Z):
        qs, Zs, qp = split(q, Z)
        Za = [psis[i].place(qs[i], Zs[i]) for i in range(n)]
        return qs, Zs, qp, Za

    def place(q, Z):
        _, _, qp, Za = parts_at(q, Z)
        return phi.place(qp, Za)

    def transfers(q, Z):
        qs, Zs, qp, Za = parts_at(q, Z)
        F = phi.T(qp, Za)
        S = [psis[i].T(qs[i], Zs[i]) for i in range(n)]
        T = {}
        T[(0, V)] = F(0, V) + sum((F(0, i + 1) @ S[i](0, V) for i in range(n)), _zeros(dims[0] + 1))
        for a, (i, ii) in enumerate(boxes, start=1):
            T[(0, a)] = F(0, i + 1) @ S[i](0, ii)
            T[(a, 0)] = S[i](ii, 0) @ F(i + 1, 0)
            inner = F(i + 1, V) + sum((F(i + 1, j + 1) @ S[j](0, V) for j in range(n)), _zeros(phi.dims[i + 1] + 1))
            T[(a, V)] = S[i](ii, V) + S[i](ii, 0) @ inner
            for b, (j, jj) in enumerate(boxes, start=1):
                M = S[i](ii, 0) @ F(i + 1, j + 1) @ S[j](0, jj)
                if i == j:
                    M = M + S[i](ii, jj)
                T[(a, b)] = M
        return T

    def cost(q, m_out, n_in):
        qs = [q[qcut[i]:qcut[i + 1]] for i in range(n)]
        qp = q[qcut[-1]:]
        cuts = [0]
        for p in psis:
            cuts.append(cuts[-1] + sum(p.e + d + 2 for d in p.dims[1:]))
        locs = [m_out[cuts[i]:cuts[i + 1]] for i in range(n)]
        tilde = jnp.concatenate([psis[i].sigma(qs[i], locs[i]) for i in range(n)]) if n else jnp.zeros(0)
        heard = phi.rho(qp, tilde, n_in)
        hcuts = [0]
        for p in psis:
            hcuts.append(hcuts[-1] + p.dims[0] + 1)
        total = phi.cost(qp, tilde, n_in)
        for i in range(n):
            total = total + psis[i].cost(qs[i], locs[i], heard[hcuts[i]:hcuts[i + 1]])
        return total

    return PosOp(dims=dims, e=phi.e, P=Pint, place=place, transfers=transfers, cost=cost,
                 label=label or f"{phi.label}({', '.join(p.label for p in psis)})")


def iota(phi: PosOp) -> AddrOp:
    """The addressed operation at one place per slot: address space ``R^{e+1}``, a position message ``g`` sent as the
    place ``(g, g)``; transfers copied; the cost reads the address and the message."""
    n, e = phi.n, phi.e

    def data(q, Zaddr):
        Z = [Zaddr[(i, 0)][0] for i in range(n)]
        T = phi.T(q, Z)
        g = phi.place(q, Z)
        return OpData(place=[(g, g)],
                      U={(0, (i, 0)): T(0, i + 1) for i in range(n)}, Uv={0: T(0, V)},
                      D={((i, 0), 0): T(i + 1, 0) for i in range(n)},
                      T={((i, 0), (j, 0)): T(i + 1, j + 1) for i in range(n) for j in range(n)},
                      Tv={(i, 0): T(i + 1, V) for i in range(n)})

    def cost(q, m_out, n_in):
        parts, c = [], 0
        for d in phi.dims[1:]:
            parts += [m_out[c:c + e + 1], m_out[c + 2 * (e + 1):c + 2 * (e + 1) + d + 1]]
            c += 2 * (e + 1) + d + 1
        return phi.cost(q, jnp.concatenate(parts) if parts else jnp.zeros(0), n_in)

    return AddrOp(members=tuple((d,) for d in phi.dims[1:]), shown=(phi.dims[0],), dim_P=e + 1, P=phi.P,
                  data=data, cost=cost, label=f"iota({phi.label})")


def cone_readout(g_V: Callable[[Array], Array], a: Callable[[Array], Sequence]) -> Callable:
    """``place(q, Z) = g_V(q) + sum_i a_i(q) Z_i``: the matter's own position message plus its members', each scaled
    by ``a_i(q) >= 0``; a cone map ``C(Pos)^n -> C(Pos)`` at each ``q``, plus the matter's message."""
    def place(q, Z):
        out = g_V(q)
        for ai, z in zip(a(q), Z):
            out = out + ai * z
        return out
    return place


def door(z_door: Callable[[Array], Array], mass: float = 1.0) -> Callable:
    """A place of the group's own, stored in its matter: ``a = 0``."""
    return cone_readout(lambda q: at(z_door(q), mass), lambda q: ())


def centre_of_mass(e: int) -> Callable:
    """The members' messages summed, the matter silent: ``a = 1``, ``g_V = 0``."""
    return lambda q, Z: sum(Z, jnp.zeros(e + 1))


def d_eps(x: Array, eps: float) -> Array:
    """The softened distance ``sqrt(|x|^2 + eps^2)``; ``eps = 0`` is the plain distance."""
    return jnp.sqrt(jnp.sum(x * x) + eps ** 2)


def ear_op(d: int, n: int, e: int, P: CotCoalg, place: Callable, beta: float, eps: float,
           cost: Callable = zero_cost, label: str = "") -> PosOp:
    """Every box hears every other box (not itself) at loudness ``exp(-beta d_eps(z_i - z_j))``, unchanged in value;
    ``z_i`` the value of member ``i``'s position message, ``z_0`` that of the operation's own readout; matter silent."""
    I = jnp.eye(d + 1)

    def transfers(q, Z):
        z = [value(place(q, Z))] + [value(Zi) for Zi in Z]
        return {(i, j): jnp.exp(-beta * d_eps(z[i] - z[j], eps)) * I
                for i in range(n + 1) for j in range(n + 1) if i != j}

    return PosOp(dims=(d,) * (n + 1), e=e, P=P, place=place, transfers=transfers, cost=cost,
                 label=label or f"ear{n}")


def space_op(d: int, n: int, e: int, P: CotCoalg, place: Callable, loudness: Callable,
             cost: Callable = zero_cost, label: str = "") -> PosOp:
    """Every box hears every other box (not itself) at ``loudness(z_i, z_j)``, unchanged in value; ``z_i`` the value
    of member ``i``'s position message, ``z_0`` that of the operation's own readout; matter silent. ``ear_op`` is the
    case ``loudness = exp(-beta d_eps(z_i - z_j))``; a loudness summed over several paths is a sum of cone maps."""
    I = jnp.eye(d + 1)

    def transfers(q, Z):
        z = [value(place(q, Z))] + [value(Zi) for Zi in Z]
        return {(i, j): loudness(z[i], z[j]) * I for i in range(n + 1) for j in range(n + 1) if i != j}

    return PosOp(dims=(d,) * (n + 1), e=e, P=P, place=place, transfers=transfers, cost=cost,
                 label=label or f"space{n}")
