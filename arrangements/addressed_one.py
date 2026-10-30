"""The addressed operad after the matter's own messages and P77, and the book's attention operad (book plan P79 (b)).

The book's Ch 4 attention operad (``def.attention_arrangement``, ``eqn.attention_substitution``, 2026-10-29) differs
from the paper's operad of ``attention.py`` in three ways: a box's interface carries messages only (P77, no
positions); a box may hear itself through the arrangement (``T_ii``); and the matter sends messages of its own
(``T_{i voice}``, P73 (c)). ``addressed.py`` was built against the paper's operad, so it has none of these. This module
is the addressed operad with all three, as proposed in the book repository's ``notes/runs/p79-b430b0e2/RESULTS.md`` §2;
everything here is the agents' and *proposed*.

**Colors** are lists of value dimensions ``(d_1, ..., d_r)``, one per slot. The interface of a color has, per slot,
input the heard message ``(h, w)`` in ``R^{d+1}`` and output ``(k, p, h, w)``: an address and a query in ``P = R^dim_P``
and a message. No position is received.

**Operations** (``AddrOp``): an integrator; and, at each configuration ``q`` and each list ``Z`` of every member slot's
sent ``(k, p)``, the data ``OpData``: for each shown slot ``u`` a place, transfers up ``U[(u, y)]`` and the matter's
message out ``Uv[u]``; for each member slot ``x`` transfers down ``D[(x, u)]``, a hearing ``T[(x, y)]`` for *every*
member slot ``y`` (its own member's and itself included) and the matter's message to it ``Tv[x]``; and a cost, or
any effort field ``omega`` in its place (``AddrOp.effort``). A
transfer from slot ``y`` to slot ``x`` is a cone map, a ``(d_x+1, d_y+1)`` matrix with last row ``(0, ..., 0, gamma)``,
``gamma > 0``, or zero. A missing key is the zero map.

**Substitution** (``substitute``): no frames, no placements; each part's data at its own members' places, the parent's
at the places its parts show; a member hears another of its own part inside the part *and* around through the parent.

``AttOp`` is the book's attention arrangement, ``j`` the map of operads into the addressed operad, ``forget_places`` the
lens ``p : B_(V) -> <M(V) | M(V)>`` that relates the two maps into ``arr``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from .arrangement import Arrangement, exact, zero_form
from .coalgebra import CotCoalg, configuration
from .rvect import trivial

Slot = Tuple[int, int]
Places = Dict[Slot, Tuple[Array, Array]]


def zero_cost(q: Array, m_out: Array, n_in: Array) -> Array:
    return jnp.asarray(0.0)


@dataclass
class OpData:
    """An operation's data at one ``(q; Z)``; missing transfers are zero."""

    place: List[Tuple[Array, Array]]
    U: Dict = field(default_factory=dict)
    Uv: Dict = field(default_factory=dict)
    D: Dict = field(default_factory=dict)
    T: Dict = field(default_factory=dict)
    Tv: Dict = field(default_factory=dict)


def _get(d, key, rows, cols):
    v = d.get(key)
    return jnp.zeros((rows, cols)) if v is None else v


def _getv(d, key, rows):
    v = d.get(key)
    return jnp.zeros(rows) if v is None else v


@dataclass(frozen=True)
class AddrOp:
    """An operation ``members -> shown`` of the addressed operad (module docstring)."""

    members: Tuple[Tuple[int, ...], ...]
    shown: Tuple[int, ...]
    dim_P: int
    P: CotCoalg
    data: Callable[[Array, Places], OpData]
    cost: Callable[[Array, Array, Array], Array] = zero_cost
    label: str = ""
    omega: Optional[Callable] = None

    def effort(self) -> Callable:
        """The effort field on ``Q x M^+ x N^-``: ``omega`` if given (any 1-form, Ch 4 ``def.addressed_arrangement``
        since 2026-10-30, David: "yah there's no reason to restrict to exact effort fields"), else ``-d cost``."""
        if self.omega is not None:
            return self.omega
        return exact(lambda q, m_out, n_in: -self.cost(q, m_out, n_in))

    @property
    def slots(self) -> List[Slot]:
        return [(i, a) for i, c in enumerate(self.members) for a in range(len(c))]

    def dim(self, x: Slot) -> int:
        return self.members[x[0]][x[1]]

    def out_dim_M(self) -> int:
        return sum(2 * self.dim_P + self.dim(x) + 1 for x in self.slots)

    def in_dim_M(self) -> int:
        return sum(self.dim(x) + 1 for x in self.slots)

    def out_dim_N(self) -> int:
        return sum(2 * self.dim_P + d + 1 for d in self.shown)

    def in_dim_N(self) -> int:
        return sum(d + 1 for d in self.shown)

    def sent(self, m_out: Array):
        """Each member slot's ``(k, p)`` and message."""
        Z, msgs, c, P = {}, {}, 0, self.dim_P
        for x in self.slots:
            d = self.dim(x)
            v = m_out[c:c + 2 * P + d + 1]
            Z[x], msgs[x] = (v[:P], v[P:2 * P]), v[2 * P:]
            c += 2 * P + d + 1
        return Z, msgs

    def heard_outside(self, n_in: Array) -> List[Array]:
        out, c = [], 0
        for d in self.shown:
            out.append(n_in[c:c + d + 1])
            c += d + 1
        return out

    def sigma(self, q: Array, m_out: Array) -> Array:
        """Shown slot ``u`` sends its place and ``Uv_u + sum_y U_uy m_y``."""
        if not self.shown:
            return jnp.zeros(0)
        Z, msgs = self.sent(m_out)
        D = self.data(q, Z)
        out = []
        for u, du in enumerate(self.shown):
            m = _getv(D.Uv, u, du + 1)
            for y in self.slots:
                m = m + _get(D.U, (u, y), du + 1, self.dim(y) + 1) @ msgs[y]
            k, p = D.place[u]
            out.append(jnp.concatenate([k, p, m]))
        return jnp.concatenate(out)

    def rho(self, q: Array, m_out: Array, n_in: Array) -> Array:
        """Member slot ``x`` hears ``Tv_x + sum_u D_xu m'_u + sum_y T_xy m_y``."""
        if not self.slots:
            return jnp.zeros(0)
        Z, msgs = self.sent(m_out)
        D = self.data(q, Z)
        ext = self.heard_outside(n_in)
        out = []
        for x in self.slots:
            dx = self.dim(x) + 1
            m = _getv(D.Tv, x, dx)
            for u, du in enumerate(self.shown):
                m = m + _get(D.D, (x, u), dx, du + 1) @ ext[u]
            for y in self.slots:
                m = m + _get(D.T, (x, y), dx, self.dim(y) + 1) @ msgs[y]
            out.append(m)
        return jnp.concatenate(out)

    def arr(self) -> Arrangement:
        return Arrangement(
            P=self.P, in_dim_M=self.in_dim_M(), out_dim_M=self.out_dim_M(),
            in_dim_N=self.in_dim_N(), out_dim_N=self.out_dim_N(),
            out_f=self.sigma, in_f=self.rho,
            omega=self.effort(),
            label=self.label or f"addr{self.members}->{self.shown}",
        )


def unit(color: Tuple[int, ...], dim_P: int) -> AddrOp:
    """One member of ``color``, each slot forwarded: its own place, ``U = D = id``."""
    color = tuple(color)

    def data(q, Z):
        return OpData(place=[Z[(0, u)] for u in range(len(color))],
                      U={(u, (0, u)): jnp.eye(d + 1) for u, d in enumerate(color)},
                      D={((0, u), u): jnp.eye(d + 1) for u, d in enumerate(color)})

    return AddrOp(members=(color,), shown=color, dim_P=dim_P, P=configuration(trivial()), data=data,
                  label=f"1{color}")


def substitute(a: AddrOp, parts: Sequence[AddrOp], label: str = "", old_rule: bool = False) -> AddrOp:
    """``b = a(a_1, ..., a_n)``; configuration factors ordered ``Q_{a_1} x ... x Q_{a_n} x Q_a`` (as ``compose_seq``).

    ``old_rule=True`` is the control: ``addressed.py``'s rule, in which members of one part hear each other only
    inside the part, and no matter's message travels.
    """
    parts = list(parts)
    if len(parts) != len(a.members):
        raise ValueError("arity")
    for i, c in enumerate(parts):
        if tuple(c.shown) != tuple(a.members[i]):
            raise ValueError(f"part {i} shows {c.shown}, the parent's member {i} is {a.members[i]}")
    P_ = a.dim_P
    owner = [(i, j) for i, c in enumerate(parts) for j in range(len(c.members))]
    starts = [sum(len(c.members) for c in parts[:i]) for i in range(len(parts) + 1)]
    qd = [c.P.Q_dim for c in parts]
    qcut = [sum(qd[:i]) for i in range(len(parts) + 1)]
    Pint = None
    for c in parts:
        Pint = c.P if Pint is None else Pint.tensor(c.P)
    Pint = a.P if Pint is None else Pint.tensor(a.P)
    members = tuple(m for c in parts for m in c.members)

    def split_q(q):
        return [q[qcut[i]:qcut[i + 1]] for i in range(len(parts))], q[qcut[-1]:]

    def local(x):
        i, j = owner[x[0]]
        return i, (j, x[1])

    def dimx(x):
        return members[x[0]][x[1]] + 1

    def data(q, Z):
        qs, qa = split_q(q)
        Zi = [dict() for _ in parts]
        for x, kp in Z.items():
            i, xl = local(x)
            Zi[i][xl] = kp
        Di = [parts[i].data(qs[i], Zi[i]) for i in range(len(parts))]
        Za = {(i, v): Di[i].place[v] for i in range(len(parts)) for v in range(len(parts[i].shown))}
        Da = a.data(qa, Za)
        bslots = [(c, s) for c, m in enumerate(members) for s in range(len(m))]
        U, Uv, D, T, Tv = {}, {}, {}, {}, {}
        for u, du in enumerate(a.shown):
            uv = _getv(Da.Uv, u, du + 1)
            if not old_rule:
                for r, c in enumerate(parts):
                    for w, dw in enumerate(c.shown):
                        uv = uv + _get(Da.U, (u, (r, w)), du + 1, dw + 1) @ _getv(Di[r].Uv, w, dw + 1)
            Uv[u] = uv
            for y in bslots:
                r, yl = local(y)
                M = jnp.zeros((du + 1, dimx(y)))
                for w, dw in enumerate(parts[r].shown):
                    M = M + _get(Da.U, (u, (r, w)), du + 1, dw + 1) @ _get(Di[r].U, (w, yl), dw + 1, dimx(y))
                U[(u, y)] = M
        for x in bslots:
            i, xl = local(x)
            dx = dimx(x)
            for u, du in enumerate(a.shown):
                M = jnp.zeros((dx, du + 1))
                for v, dv in enumerate(parts[i].shown):
                    M = M + _get(Di[i].D, (xl, v), dx, dv + 1) @ _get(Da.D, ((i, v), u), dv + 1, du + 1)
                D[(x, u)] = M
            for y in bslots:
                r, yl = local(y)
                dy = dimx(y)
                M = _get(Di[i].T, (xl, yl), dx, dy) if i == r else jnp.zeros((dx, dy))
                if not (old_rule and i == r):
                    for v, dv in enumerate(parts[i].shown):
                        Dv = _get(Di[i].D, (xl, v), dx, dv + 1)
                        for w, dw in enumerate(parts[r].shown):
                            M = M + Dv @ _get(Da.T, ((i, v), (r, w)), dv + 1, dw + 1) \
                                @ _get(Di[r].U, (w, yl), dw + 1, dy)
                T[(x, y)] = M
            tv = _getv(Di[i].Tv, xl, dx)
            if not old_rule:
                for v, dv in enumerate(parts[i].shown):
                    inner = _getv(Da.Tv, (i, v), dv + 1)
                    for r, c in enumerate(parts):
                        for w, dw in enumerate(c.shown):
                            inner = inner + _get(Da.T, ((i, v), (r, w)), dv + 1, dw + 1) @ _getv(Di[r].Uv, w, dw + 1)
                    tv = tv + _get(Di[i].D, (xl, v), dx, dv + 1) @ inner
            Tv[x] = tv
        return OpData(place=list(Da.place), U=U, Uv=Uv, D=D, T=T, Tv=Tv)

    def cost(q, m_out, n_in):
        qs, qa = split_q(q)
        cuts = [0]
        for c in parts:
            cuts.append(cuts[-1] + c.out_dim_M())
        locs = [m_out[cuts[i]:cuts[i + 1]] for i in range(len(parts))]
        tilde = jnp.concatenate([parts[i].sigma(qs[i], locs[i]) for i in range(len(parts))]) \
            if parts else jnp.zeros(0)
        e = a.rho(qa, tilde, n_in)
        ecuts = [0]
        for c in parts:
            ecuts.append(ecuts[-1] + c.in_dim_N())
        total = a.cost(qa, tilde, n_in)
        for i in range(len(parts)):
            total = total + parts[i].cost(qs[i], locs[i], e[ecuts[i]:ecuts[i + 1]])
        return total

    omega = None
    if a.omega is not None or any(c.omega is not None for c in parts):
        def points(q, m_out, n_in):
            qs, qa = split_q(q)
            cuts = [0]
            for c in parts:
                cuts.append(cuts[-1] + c.out_dim_M())
            locs = [m_out[cuts[i]:cuts[i + 1]] for i in range(len(parts))]
            tilde = jnp.concatenate([parts[i].sigma(qs[i], locs[i]) for i in range(len(parts))]) \
                if parts else jnp.zeros(0)
            e = a.rho(qa, tilde, n_in)
            ecuts = [0]
            for c in parts:
                ecuts.append(ecuts[-1] + c.in_dim_N())
            return [(qa, tilde, n_in)] + [(qs[i], locs[i], e[ecuts[i]:ecuts[i + 1]]) for i in range(len(parts))]

        def omega(q, m_out, n_in):
            """Each constituent's effort field at what it reads and receives, pulled back and summed."""
            pts, pull = jax.vjp(points, q, m_out, n_in)
            covs = [tuple(a.effort()(*pts[0]))] + [tuple(parts[i].effort()(*pts[i + 1])) for i in range(len(parts))]
            return pull(covs)

    return AddrOp(members=members, shown=a.shown, dim_P=P_, P=Pint, data=data, cost=cost,
                  label=label or f"{a.label}({', '.join(c.label for c in parts)})", omega=omega)


# ---------------------------------------------------------------------------
# The book's attention operad (Ch 4 def.attention_arrangement, eqn.attention_arrangement,
# eqn.attention_substitution): no positions, self-hearing allowed, the matter's own messages.
# ---------------------------------------------------------------------------


V = "v"   # the index of the matter as a sender


@dataclass(frozen=True)
class AttOp:
    """``phi : V_1, ..., V_n -> V_0`` with ``dims = (d_0, d_1, ..., d_n)``; ``transfers(q)`` a dict ``(i, j) -> matrix``
    for ``i`` in ``0..n`` and ``j`` in ``V, 0..n`` (``(i, V)`` a vector, the matter's message), ``T_00 = 0``."""

    dims: Tuple[int, ...]
    P: CotCoalg
    transfers: Callable[[Array], Dict]
    cost: Callable[[Array, Array, Array], Array] = zero_cost
    label: str = ""

    @property
    def n(self) -> int:
        return len(self.dims) - 1

    def T(self, q):
        T = self.transfers(q)
        d = self.dims

        def get(i, j):
            if j == V:
                return _getv(T, (i, V), d[i] + 1)
            return _get(T, (i, j), d[i] + 1, d[j] + 1)
        return get

    def split(self, m_out):
        out, c = [], 0
        for d in self.dims[1:]:
            out.append(m_out[c:c + d + 1])
            c += d + 1
        return out

    def sigma(self, q, m_out):
        T, ms = self.T(q), self.split(m_out)
        out = T(0, V)
        for j in range(1, self.n + 1):
            out = out + T(0, j) @ ms[j - 1]
        return out

    def rho(self, q, m_out, n_in):
        if self.n == 0:
            return jnp.zeros(0)
        T, ms = self.T(q), self.split(m_out)
        out = []
        for i in range(1, self.n + 1):
            m = T(i, V) + T(i, 0) @ n_in
            for j in range(1, self.n + 1):
                m = m + T(i, j) @ ms[j - 1]
            out.append(m)
        return jnp.concatenate(out)

    def arr(self) -> Arrangement:
        s = sum(d + 1 for d in self.dims[1:])
        return Arrangement(P=self.P, in_dim_M=s, out_dim_M=s, in_dim_N=self.dims[0] + 1,
                           out_dim_N=self.dims[0] + 1, out_f=self.sigma, in_f=self.rho,
                           omega=exact(lambda q, m_out, n_in: -self.cost(q, m_out, n_in)),
                           label=self.label or f"att{self.dims}")


def att_unit(d: int) -> AttOp:
    I = jnp.eye(d + 1)
    return AttOp(dims=(d, d), P=configuration(trivial()), transfers=lambda q: {(0, 1): I, (1, 0): I}, label=f"1_{d}")


def att_substitute(phi: AttOp, psis: Sequence[AttOp], label: str = "") -> AttOp:
    """``eqn.attention_substitution``, row by row; configuration order ``Q_{psi_1} x ... x Q_{psi_n} x Q_phi``."""
    psis = list(psis)
    n = phi.n
    for i, p in enumerate(psis):
        if p.dims[0] != phi.dims[i + 1]:
            raise ValueError("colors")
    boxes = [(i, ii) for i in range(n) for ii in range(1, psis[i].n + 1)]
    dims = (phi.dims[0],) + tuple(psis[i].dims[ii] for i, ii in boxes)
    qd = [p.P.Q_dim for p in psis]
    qcut = [sum(qd[:i]) for i in range(n + 1)]
    Pint = None
    for p in psis:
        Pint = p.P if Pint is None else Pint.tensor(p.P)
    Pint = phi.P if Pint is None else Pint.tensor(phi.P)

    def split_q(q):
        return [q[qcut[i]:qcut[i + 1]] for i in range(n)], q[qcut[-1]:]

    def transfers(q):
        qs, qp = split_q(q)
        F = phi.T(qp)
        S = [psis[i].T(qs[i]) for i in range(n)]
        T = {}
        T[(0, V)] = F(0, V) + sum((F(0, i + 1) @ S[i](0, V) for i in range(n)), jnp.zeros(dims[0] + 1))
        for a, (i, ii) in enumerate(boxes, start=1):
            T[(0, a)] = F(0, i + 1) @ S[i](0, ii)
            T[(a, 0)] = S[i](ii, 0) @ F(i + 1, 0)
            inner = F(i + 1, V) + sum((F(i + 1, j + 1) @ S[j](0, V) for j in range(n)), jnp.zeros(phi.dims[i + 1] + 1))
            T[(a, V)] = S[i](ii, V) + S[i](ii, 0) @ inner
            for b, (j, jj) in enumerate(boxes, start=1):
                M = S[i](ii, 0) @ F(i + 1, j + 1) @ S[j](0, jj)
                if i == j:
                    M = M + S[i](ii, jj)
                T[(a, b)] = M
        return T

    def cost(q, m_out, n_in):
        qs, qp = split_q(q)
        cuts = [0]
        for p in psis:
            cuts.append(cuts[-1] + sum(d + 1 for d in p.dims[1:]))
        locs = [m_out[cuts[i]:cuts[i + 1]] for i in range(n)]
        tilde = jnp.concatenate([psis[i].sigma(qs[i], locs[i]) for i in range(n)]) if n else jnp.zeros(0)
        e = phi.rho(qp, tilde, n_in)
        ecuts = [0]
        for p in psis:
            ecuts.append(ecuts[-1] + p.dims[0] + 1)
        total = phi.cost(qp, tilde, n_in)
        for i in range(n):
            total = total + psis[i].cost(qs[i], locs[i], e[ecuts[i]:ecuts[i + 1]])
        return total

    return AttOp(dims=dims, P=Pint, transfers=transfers, cost=cost,
                 label=label or f"{phi.label}({', '.join(p.label for p in psis)})")


def j(phi: AttOp, dim_P: int, place: str = "sum") -> AddrOp:
    """The map of operads ``j : Att -> Addr_P``: one shown slot showing the sum of its members' sent places (the plain
    sum, David 2026-10-21), every transfer copied, the cost copied. ``place="zero"`` is the control that fails the
    unit (P64 §1)."""
    n = phi.n

    def data(q, Z):
        T = phi.T(q)
        if place == "sum":
            k = sum((Z[(i, 0)][0] for i in range(n)), jnp.zeros(dim_P))
            p = sum((Z[(i, 0)][1] for i in range(n)), jnp.zeros(dim_P))
        else:
            k = p = jnp.zeros(dim_P)
        return OpData(place=[(k, p)],
                      U={(0, (i, 0)): T(0, i + 1) for i in range(n)}, Uv={0: T(0, V)},
                      D={((i, 0), 0): T(i + 1, 0) for i in range(n)},
                      T={((i, 0), (jj, 0)): T(i + 1, jj + 1) for i in range(n) for jj in range(n)},
                      Tv={(i, 0): T(i + 1, V) for i in range(n)})

    def cost(q, m_out, n_in):
        msgs, c = [], 0
        for d in phi.dims[1:]:
            msgs.append(m_out[c + 2 * dim_P:c + 2 * dim_P + d + 1])
            c += 2 * dim_P + d + 1
        return phi.cost(q, jnp.concatenate(msgs) if msgs else jnp.zeros(0), n_in)

    return AddrOp(members=tuple((d,) for d in phi.dims[1:]), shown=(phi.dims[0],), dim_P=dim_P, P=phi.P,
                  data=data, cost=cost, label=f"j({phi.label})")


def forget_places(d: int, dim_P: int) -> Arrangement:
    """The lens ``p : B_(d) -> <M | M>`` of ``arr``: reads out the message a slot sends, forgetting its address and
    query; delivers what it hears unchanged; no matter, no effort."""
    return Arrangement(P=configuration(trivial()), in_dim_M=d + 1, out_dim_M=2 * dim_P + d + 1,
                       in_dim_N=d + 1, out_dim_N=d + 1,
                       out_f=lambda q, m_out: m_out[2 * dim_P:], in_f=lambda q, m_out, n_in: n_in,
                       omega=zero_form, label=f"p{d}")
