"""Addressed attention: messages that carry their address (book plan P5 (g), §5.4 *Two ways to move*).

The paper's operad (``attention.py``, ``background/affine_quadratic_operad.tex``) moves a part
only the slow way: its transfers ``T_ij(q)`` depend on the parent's configuration alone, so what a
part sends sets its weight, never whom it hears. Here a part also sends *where* its message is
(an address) and *where it listens* (a query), and the parent's hearing may depend on them, so a
part can change whom it hears through its own outputs (the jump), as a transformer's keys and
queries do. The definition and its proof are in the book repository,
``notes/runs/p5g-f519e448/RESULTS.md``; everything here is the agents' and *proposed*.

**Colors and interfaces.** A color is a number ``r`` of *slots*. A slot sends an address
``k in P``, a query ``p in P`` and a message ``m = (h, w) in M = H x R``, and receives one message,
what it hears. The interface of color ``r`` is ``B_r = <M^r x P | (P x P x M)^r>``: what each slot
hears and a reference position, over what each slot sends. Coordinates: a slot's output is
``[k..., p..., h..., w]`` (length ``2 dim_P + dim_H + 1``); the input is the ``r`` heard messages,
then the position.

**Operations.** ``AddressedArrangement`` with ``slots = (r_1, ..., r_n)``: an integrator ``P``; a
placement ``l : Q -> P^n``; a *hearing* ``T_xy(q; Z)``, a message transfer for each slot ``x`` of
child ``i`` and slot ``y`` of child ``j != i``; the slots it *shows* outward, each a ``Shown``; and
a cost. ``Z`` is every child slot's address and query in the group's frame: a slot of child ``i``
that sends ``(k, p, m)`` sits at ``l_i(q) + k`` and listens at ``l_i(q) + p``. Messages enter
linearly; places, and the configuration, smoothly.

A ``Shown`` slot has a place ``(key, query)(q; Z)``, sends ``sum_y up(q, y; Z) m_y`` and hands what it
hears down to slot ``x`` as ``down(q, x; Z)`` of it. Two special cases: ``forward(x)``, the member's
own place, message and hearing passing the wall unchanged; ``speaker(...)``, a place the group sets
from its configuration alone, speaking for its members (the paper's group is one speaker at its
center, so the paper's operad sits inside). Anything between composes as well, e.g. a group speaking
from its members' mean place (``centroid_speaker``).

**Substitution** (``substitute``): two members of different parts hear each other through every
pair of slots their parts show, ``sum_{v,w} down_v(x) . T^a_{vw} . up_w(y)``, at the places those
shown slots have; members of one part, by that part's own hearing in its frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple, Union

import jax.numpy as jnp
from jax import Array

from .arrangement import Arrangement, exact, zero_form
from .attention import (
    Transfer, add_transfers, apply_transfer, compose_transfers, identity_transfer, zero_transfer,
)
from .coalgebra import CotCoalg, configuration
from .rvect import trivial

Slot = Tuple[int, int]                    # (child, slot of that child), both counted from 0
Places = Dict[Slot, Tuple[Array, Array]]  # every child slot's (address, query), in the group's frame
Hearing = Callable[[Array, Slot, Slot, Places], Transfer]   # (q, x, y, Z) |-> T_xy(q; Z)


@dataclass(frozen=True)
class Shown:
    """A slot a group shows outward (module docstring)."""

    place: Callable[[Array, Places], Tuple[Array, Array]]    # (q, Z) |-> (address, query)
    up: Callable[[Array, Slot, Places], Transfer]             # (q, y, Z) |-> how member y's message goes out
    down: Callable[[Array, Slot, Places], Transfer]           # (q, x, Z) |-> how what it hears reaches x
    label: str = ""


def forward(x: Slot, dim_H: int) -> Shown:
    """The member slot ``x`` shown as it is: its own place, its message, its hearing."""
    def one(q, z, Z):
        return identity_transfer(dim_H) if z == x else zero_transfer(dim_H)
    return Shown(place=lambda q, Z: Z[x], up=one, down=one, label=f"fwd{x}")


def speaker(key, query, up, down, label: str = "speaker") -> Shown:
    """A slot at a place the group sets from its configuration: ``key(q)``, ``query(q)``; it sends
    ``sum_y up(q, y) m_y`` and hands down ``down(q, x)``."""
    return Shown(place=lambda q, Z: (key(q), query(q)), up=lambda q, y, Z: up(q, y),
                 down=lambda q, x, Z: down(q, x), label=label)


def centroid_speaker(members: Sequence[Slot], dim_H: int, offset=None, label: str = "centroid") -> Shown:
    """Between the two: the group speaks for ``members`` from their mean address and mean query (plus
    a configuration-set ``offset(q)``), sending the sum of their messages and handing what it hears to
    each of them."""
    members = tuple(members)

    def place(q, Z):
        k = sum(Z[x][0] for x in members) / len(members)
        p = sum(Z[x][1] for x in members) / len(members)
        if offset is not None:
            k, p = k + offset(q), p + offset(q)
        return k, p

    def one(q, z, Z):
        return identity_transfer(dim_H) if z in members else zero_transfer(dim_H)

    return Shown(place=place, up=one, down=one, label=label)


def slot_dim(dim_H: int, dim_P: int) -> int:
    """A slot's output ``(k, p, h, w)``."""
    return 2 * dim_P + dim_H + 1


def out_dim(r: int, dim_H: int, dim_P: int) -> int:
    return r * slot_dim(dim_H, dim_P)


def in_dim(r: int, dim_H: int, dim_P: int) -> int:
    """What ``r`` slots hear, then the reference position."""
    return r * (dim_H + 1) + dim_P


def slot_output(k, p, h, w) -> Array:
    return jnp.concatenate([jnp.ravel(jnp.asarray(k, float)), jnp.ravel(jnp.asarray(p, float)),
                            jnp.ravel(jnp.asarray(h, float)), jnp.reshape(jnp.asarray(w, float), (1,))])


def split_slot(v: Array, dim_H: int, dim_P: int) -> Tuple[Array, Array, Array]:
    """``(k, p, m)`` of one slot's output."""
    return v[:dim_P], v[dim_P:2 * dim_P], v[2 * dim_P:]


def zero_cost(q: Array, m_out: Array, n_in: Array) -> Array:
    return jnp.asarray(0.0)


def _as_shown(e, dim_H: int) -> Shown:
    return e if isinstance(e, Shown) else forward(tuple(e), dim_H)


@dataclass(frozen=True)
class AddressedArrangement:
    """An operation of ``Addr(r_1, ..., r_n; r)``, ``r = len(exposure)`` (module docstring).

    ``exposure`` lists the shown slots; a bare ``(i, a)`` in it means ``forward((i, a))``.
    """

    slots: Tuple[int, ...]
    exposure: Tuple[Union[Shown, Slot], ...]
    dim_H: int
    dim_P: int
    P: CotCoalg
    placement: Callable[[Array], Array]          # q |-> (n, dim_P)
    hearing: Hearing
    cost: Callable[[Array, Array, Array], Array] = zero_cost
    label: str = ""

    @property
    def n(self) -> int:
        return len(self.slots)

    @property
    def r(self) -> int:
        return len(self.exposure)

    @property
    def shown(self) -> Tuple[Shown, ...]:
        return tuple(_as_shown(e, self.dim_H) for e in self.exposure)

    def child_slots(self):
        return [(i, a) for i, ri in enumerate(self.slots) for a in range(ri)]

    def _sent(self, q: Array, m_out: Array):
        """Each child slot's places in the group's frame, ``Z``, and its message."""
        sd, ell, Z, msgs, c = slot_dim(self.dim_H, self.dim_P), self.placement(q), {}, {}, 0
        for x in self.child_slots():
            k, p, m = split_slot(m_out[c:c + sd], self.dim_H, self.dim_P)
            Z[x], msgs[x] = (ell[x[0]] + k, ell[x[0]] + p), m
            c += sd
        return Z, msgs

    def sigma(self, q: Array, m_out: Array) -> Array:
        """The shown slots, in order, each at its place, sending its members' messages through ``up``."""
        if not self.exposure:
            return jnp.zeros(0)
        Z, msgs = self._sent(q, m_out)
        out = []
        for e in self.shown:
            k, p = e.place(q, Z)
            m = sum((apply_transfer(e.up(q, y, Z), msgs[y]) for y in self.child_slots()),
                    jnp.zeros(self.dim_H + 1))
            out.append(jnp.concatenate([k, p, m]))
        return jnp.concatenate(out)

    def rho(self, q: Array, m_out: Array, n_in: Array) -> Array:
        """What each child receives: per slot, what it hears; then its position ``p_0 + l_i``."""
        hd = self.dim_H + 1
        ell = self.placement(q)
        Z, msgs = self._sent(q, m_out)
        shown = self.shown
        p0 = n_in[self.r * hd:]
        parts = []
        for i, ri in enumerate(self.slots):
            for a in range(ri):
                x = (i, a)
                heard = jnp.zeros(hd)
                for u, e in enumerate(shown):
                    heard = heard + apply_transfer(e.down(q, x, Z), n_in[u * hd:(u + 1) * hd])
                for y in self.child_slots():
                    if y[0] != i:
                        heard = heard + apply_transfer(self.hearing(q, x, y, Z), msgs[y])
                parts.append(heard)
            parts.append(p0 + ell[i])
        return jnp.concatenate(parts) if parts else jnp.zeros(0)

    def arr(self) -> Arrangement:
        """The arrangement ``B_{r_1} (x) ... (x) B_{r_n} -> B_r`` of ``arr``, forcing ``-dk``."""
        h, p = self.dim_H, self.dim_P
        return Arrangement(
            P=self.P,
            in_dim_M=sum(in_dim(ri, h, p) for ri in self.slots),
            out_dim_M=sum(out_dim(ri, h, p) for ri in self.slots),
            in_dim_N=in_dim(self.r, h, p),
            out_dim_N=out_dim(self.r, h, p),
            out_f=self.sigma,
            in_f=self.rho,
            omega=exact(lambda q, m_out, n_in: -self.cost(q, m_out, n_in)),
            label=self.label or f"addr{self.slots}->{self.r}",
        )


def unit(r: int, dim_H: int, dim_P: int) -> AddressedArrangement:
    """The unit of color ``r``: one child, every slot forwarded in order, no placement, no cost."""
    return AddressedArrangement(
        slots=(r,), exposure=tuple((0, a) for a in range(r)), dim_H=dim_H, dim_P=dim_P,
        P=configuration(trivial()), placement=lambda q: jnp.zeros((1, dim_P)),
        hearing=lambda q, x, y, Z: zero_transfer(dim_H), label=f"1_{r}",
    )


def substitute(a: AddressedArrangement, parts: Sequence[AddressedArrangement],
               label: str = "") -> AddressedArrangement:
    """``b = a(a_1, ..., a_n)`` (module docstring; RESULTS.md).

    Configuration factors are ordered ``Q_{a_1} x ... x Q_{a_n} x Q_a``, the order ``compose_seq``
    produces, so the two composites can be compared coordinatewise. Grandchild ``(i, j)`` is child
    ``starts[i] + j`` of ``b``.
    """
    parts = list(parts)
    if len(parts) != a.n:
        raise ValueError(f"substitute: arity {a.n}, got {len(parts)} parts")
    for i, c in enumerate(parts):
        if c.r != a.slots[i]:
            raise ValueError(f"part {i} has color {c.r}, the parent's child {i} has {a.slots[i]}")
        if (c.dim_H, c.dim_P) != (a.dim_H, a.dim_P):
            raise ValueError(f"part {i}: signal/place spaces differ from the parent's")
    dH, dP = a.dim_H, a.dim_P
    starts = [sum(c.n for c in parts[:i]) for i in range(a.n + 1)]
    owner = [(i, j) for i, c in enumerate(parts) for j in range(c.n)]   # b's child -> (i, j)
    qd = [c.P.Q_dim for c in parts]
    qcut = [sum(qd[:i]) for i in range(a.n + 1)]
    a_shown = a.shown
    p_shown = [c.shown for c in parts]

    P = None
    for c in parts:
        P = c.P if P is None else P.tensor(c.P)
    P = a.P if P is None else P.tensor(a.P)

    def split_q(q):
        return [q[qcut[i]:qcut[i + 1]] for i in range(a.n)], q[qcut[a.n]:]

    def local(x):
        """b's slot ``x`` as ``(part, that part's own child slot)``."""
        i, j = owner[x[0]]
        return i, (j, x[1])

    def frames(q, Z):
        """From b's places ``Z``: each part's members' places in its own frame, and ``a``'s ``Z_a``,
        the places of the slots each part shows, in ``a``'s frame (which is b's)."""
        qs, qa = split_q(q)
        la = a.placement(qa)
        Zi = [dict() for _ in range(a.n)]
        for x, (k, p) in Z.items():
            i, xl = local(x)
            Zi[i][xl] = (k - la[i], p - la[i])
        Za = {}
        for i in range(a.n):
            for v, e in enumerate(p_shown[i]):
                k, p = e.place(qs[i], Zi[i])
                Za[(i, v)] = (la[i] + k, la[i] + p)
        return qs, qa, Zi, Za

    def placement(q):
        qs, qa = split_q(q)
        la = a.placement(qa)
        rows = [la[i] + parts[i].placement(qs[i])[j] for i, j in owner]
        return jnp.stack(rows) if rows else jnp.zeros((0, dP))

    def climb(q, y, Z, qs, Zi, coeff):
        """``sum_w coeff((r, w)) . up^{a_r}_w(y)``: member ``y`` of part ``r`` going out through every
        slot its part shows."""
        r, yl = local(y)
        T = zero_transfer(dH)
        for w, e in enumerate(p_shown[r]):
            T = add_transfers(T, compose_transfers(coeff((r, w)), e.up(qs[r], yl, Zi[r])))
        return T

    def descend(q, x, Z, qs, Zi, coeff):
        """``sum_v down^{a_i}_v(x) . coeff((i, v))``."""
        i, xl = local(x)
        T = zero_transfer(dH)
        for v, e in enumerate(p_shown[i]):
            T = add_transfers(T, compose_transfers(e.down(qs[i], xl, Zi[i]), coeff((i, v))))
        return T

    def lifted(e):
        """``a``'s shown slot ``e`` seen from ``b``."""
        def place(q, Z):
            qs, qa, Zi, Za = frames(q, Z)
            return e.place(qa, Za)

        def up(q, y, Z):
            qs, qa, Zi, Za = frames(q, Z)
            return climb(q, y, Z, qs, Zi, lambda s: e.up(qa, s, Za))

        def down(q, x, Z):
            qs, qa, Zi, Za = frames(q, Z)
            return descend(q, x, Z, qs, Zi, lambda s: e.down(qa, s, Za))

        return Shown(place=place, up=up, down=down, label=e.label)

    def hearing(q, x, y, Z):
        qs, qa, Zi, Za = frames(q, Z)
        (i, xl), (r, yl) = local(x), local(y)
        if i == r:
            return parts[i].hearing(qs[i], xl, yl, Zi[i])
        T = zero_transfer(dH)
        for v, ev in enumerate(p_shown[i]):
            D = ev.down(qs[i], xl, Zi[i])
            for w, ew in enumerate(p_shown[r]):
                U = ew.up(qs[r], yl, Zi[r])
                T = add_transfers(T, compose_transfers(compose_transfers(D, a.hearing(qa, (i, v), (r, w), Za)), U))
        return T

    def cost(q, m_out, n_in):
        qs, qa = split_q(q)
        sd = slot_dim(dH, dP)
        cuts = [sum(sum(c.slots) for c in parts[:i]) * sd for i in range(a.n + 1)]
        locs = [m_out[cuts[i]:cuts[i + 1]] for i in range(a.n)]
        tilde = jnp.concatenate([parts[i].sigma(qs[i], locs[i]) for i in range(a.n)]) \
            if a.n else jnp.zeros(0)
        e_all = a.rho(qa, tilde, n_in)
        ecuts = [sum(in_dim(ri, dH, dP) for ri in a.slots[:i]) for i in range(a.n + 1)]
        total = a.cost(qa, tilde, n_in)
        for i in range(a.n):
            total = total + parts[i].cost(qs[i], locs[i], e_all[ecuts[i]:ecuts[i + 1]])
        return total

    return AddressedArrangement(
        slots=tuple(s for c in parts for s in c.slots), exposure=tuple(lifted(e) for e in a_shown),
        dim_H=dH, dim_P=dP, P=P, placement=placement, hearing=hearing, cost=cost,
        label=label or f"{a.label or 'a'}({', '.join(c.label or '?' for c in parts)})",
    )


# ---------------------------------------------------------------------------
# Kernel hearings: T_xy(q; Z) = kappa(p_x, k_y) (V_xy(q) h, w).
# ---------------------------------------------------------------------------


def kernel_hearing(kernel: Callable[[Array, Array], Array], dim_H: int,
                   value_map: Callable[[Array, Slot, Slot], Array] = None) -> Hearing:
    """The hearing ``(h, w) |-> kappa(p_x, k_y) (V h, w)``, ``p_x`` the receiver's query and ``k_y`` the
    sender's address: weight ``kappa > 0``, signal map ``kappa V``. It depends on those two places only."""

    def hearing(q, x, y, Z):
        g = kernel(Z[x][1], Z[y][0])
        V = jnp.eye(dim_H) if value_map is None else value_map(q, x, y)
        return Transfer(gamma=g, L=g * V)

    return hearing


def gaussian(tau: float) -> Callable[[Array, Array], Array]:
    """``kappa(p, k) = exp(-|p - k|^2 / 2 tau)``, the paper's spatial kernel (eq:spatial-transfers)."""
    return lambda p, k: jnp.exp(-jnp.sum((p - k) ** 2) / (2.0 * tau))


def dot(scale: float = 1.0) -> Callable[[Array, Array], Array]:
    """``kappa(p, k) = exp(scale <p, k>)``, softmax attention's kernel."""
    return lambda p, k: jnp.exp(scale * jnp.dot(p, k))


# ---------------------------------------------------------------------------
# Systems that fill a box of color r, and the ground that closes a group.
# ---------------------------------------------------------------------------


def system(r: int, dim_H: int, dim_P: int, P: CotCoalg, emit: Callable[[Array], Array],
           cost: Callable[[Array, Array], Array] = None, label: str = "") -> Arrangement:
    """A system ``I -> B_r``: its matter ``P``; ``emit(q)`` its slots' outputs; ``cost(q, e)`` a cost
    on its configuration and what it receives ``e in M^r x P``, forcing ``-dk``."""
    c = (lambda q, e: jnp.asarray(0.0)) if cost is None else cost
    return Arrangement(
        P=P, in_dim_M=0, out_dim_M=0,
        in_dim_N=in_dim(r, dim_H, dim_P), out_dim_N=out_dim(r, dim_H, dim_P),
        out_f=lambda q, m_out: emit(q),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -c(q, n_in)),
        label=label or f"system{r}",
    )


def ground(dim_P: int) -> Arrangement:
    """``B_0 -> I``: closes a group that shows nothing, giving it the reference position ``0``."""
    return Arrangement(
        P=configuration(trivial()), in_dim_M=dim_P, out_dim_M=0, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(dim_P),
        omega=zero_form, label="ground",
    )
