"""Addressed attention over several rounds in one pass (book plan P53 (g)).

David, 2026-10-20: keys and queries as "functions … from the set of earlier attention outputs to the set of
keys/queries", and "a box's output kind of can depend on its input. What you do is you make a function from input to
output as the output. … an (A,B) mealy machine is a coalgebra on (B^A)y^A." The definitions and the proof are in the
book repository, ``notes/runs/p53g-53d025a9/RESULTS.md``; everything here is the agents' and *proposed*.

**Rounds.** A pass has rounds ``1, ..., L``. A box of color ``r`` sends at round 1 its slots' outputs ``s^1`` (as in
``addressed.py``) and at round ``l`` a function of everything its ``r`` slots heard at rounds ``1, ..., l - 1``: its
*strategy*. The interface ``B^L_r`` has input side ``(M^r)^L x P`` (what each slot hears at each round, then the
reference position) and output side ``S_r x prod_l C^oo((M^r)^{l-1}, S_r)``.

**Supplied maps.** A function space is not a finite-dimensional manifold, so here a strategy is a parameter ``theta`` and
a smooth map fixed by the box's ``Color``: ``strategy(s1, theta, l, H)`` returns the round-``l`` sends from the round-1
sends ``s1``, the parameter, and ``H``, an ``(l - 1, r (dim_H + 1))`` array of what the box heard. A box's output is
``[s1, theta]``. A leaf chooses its own map; a group's is its round recursion applied to its children's maps, with
parameter ``(q, its children's outputs)``.

**The operations do not change.** ``rounds(a, colors, L)`` sends an operation ``a`` of ``Addr`` (``addressed.py``) to
the arrangement that runs ``a``'s one-round sending and receiving maps round by round: at round ``l`` each child sends
its strategy at what it heard before, and hears ``rho_a`` of those sends and of what the group hears from outside at
round ``l``. The forcing is ``-dk_a`` at the last round's local interaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import jax.numpy as jnp
from jax import Array

from .addressed import AddressedArrangement, slot_dim
from .arrangement import Arrangement, exact, zero_form
from .coalgebra import CotCoalg, configuration
from .rvect import trivial

Strategy = Callable[[Array, Array, int, Array], Array]   # (s1, theta, l, H) |-> round-l sends


@dataclass(frozen=True)
class Color:
    """The type of what a box sends: ``r`` slots, a parameter of dimension ``theta_dim``, and its supplied map."""

    r: int
    theta_dim: int
    strategy: Strategy
    label: str = ""


def moore(r: int) -> Color:
    """A box that sends the same at every round: no parameter, the round-1 sends again."""
    return Color(r, 0, lambda s1, theta, l, H: s1, label=f"moore{r}")


def heard_dim(r: int, dim_H: int) -> int:
    return r * (dim_H + 1)


def out_dim(c: Color, dim_H: int, dim_P: int) -> int:
    """``[s1, theta]``."""
    return c.r * slot_dim(dim_H, dim_P) + c.theta_dim


def in_dim(c: Color, L: int, dim_H: int, dim_P: int) -> int:
    """What the box's slots hear at rounds ``1..L``, then the reference position."""
    return L * heard_dim(c.r, dim_H) + dim_P


def _split_outputs(colors: Sequence[Color], m_out: Array, dim_H: int, dim_P: int) -> List[Tuple[Array, Array]]:
    sd, parts, c = slot_dim(dim_H, dim_P), [], 0
    for col in colors:
        s1 = m_out[c:c + col.r * sd]
        theta = m_out[c + col.r * sd:c + col.r * sd + col.theta_dim]
        parts.append((s1, theta))
        c += out_dim(col, dim_H, dim_P)
    return parts


def realize(a: AddressedArrangement, colors: Sequence[Color], q: Array, m_out: Array, H_ext: Sequence[Array],
            p0: Array, upto: int):
    """Run rounds ``1..upto``. ``H_ext[l - 1]`` is what the group's shown slots hear from outside at round ``l``.
    Returns, per round, the children's sends (concatenated, as ``a.sigma`` and ``a.rho`` read them); per child, its
    hearing at each round; and per child, its position."""
    dH, dP = a.dim_H, a.dim_P
    parts = _split_outputs(colors, m_out, dH, dP)
    hist: List[List[Array]] = [[] for _ in colors]
    sends, pos = [], None
    for l in range(1, upto + 1):
        if l == 1:
            s_l = jnp.concatenate([s1 for s1, _ in parts]) if parts else jnp.zeros(0)
        else:
            s_l = jnp.concatenate([col.strategy(s1, th, l, jnp.stack(hist[i]))
                                   for i, (col, (s1, th)) in enumerate(zip(colors, parts))])
        sends.append(s_l)
        rho = a.rho(q, s_l, jnp.concatenate([H_ext[l - 1], p0]))
        c, pos = 0, []
        for i, col in enumerate(colors):
            hd = heard_dim(col.r, dH)
            hist[i].append(rho[c:c + hd])
            pos.append(rho[c + hd:c + hd + dP])
            c += hd + dP
    return sends, hist, pos


def rounds(a: AddressedArrangement, colors: Sequence[Color], L: int, label: str = "") -> Tuple[Arrangement, Color]:
    """``Phi_L(a)``: the arrangement ``B^L_{c_1} (x) ... (x) B^L_{c_n} -> B^L_c`` and its output color ``c``."""
    colors = tuple(colors)
    if tuple(c.r for c in colors) != a.slots:
        raise ValueError(f"rounds: colors {[c.r for c in colors]} do not match the operation's slots {a.slots}")
    dH, dP, r = a.dim_H, a.dim_P, a.r
    hd = heard_dim(r, dH)
    Qd = a.P.Q_dim
    m_dim = sum(out_dim(c, dH, dP) for c in colors)
    zp = jnp.zeros(dP)

    def strategy(s1, theta, l, H):
        q, m_out = theta[:Qd], theta[Qd:]
        if l == 1:
            return s1
        H_ext = [H[j] for j in range(l - 1)]
        _, hist, _ = realize(a, colors, q, m_out, H_ext, zp, l - 1)
        parts = _split_outputs(colors, m_out, dH, dP)
        s_l = jnp.concatenate([col.strategy(s1_i, th_i, l, jnp.stack(hist[i]))
                               for i, (col, (s1_i, th_i)) in enumerate(zip(colors, parts))])
        return a.sigma(q, s_l)

    out_color = Color(r, Qd + m_dim, strategy, label=label or f"rounds({a.label})")

    def ext(n_in):
        return [n_in[l * hd:(l + 1) * hd] for l in range(L)], n_in[L * hd:]

    def out_f(q, m_out):
        s1 = jnp.concatenate([s for s, _ in _split_outputs(colors, m_out, dH, dP)]) if colors else jnp.zeros(0)
        return jnp.concatenate([a.sigma(q, s1), q, m_out])

    def in_f(q, m_out, n_in):
        H_ext, p0 = ext(n_in)
        _, hist, pos = realize(a, colors, q, m_out, H_ext, p0, L)
        return jnp.concatenate([jnp.concatenate(hist[i] + [pos[i]]) for i in range(len(colors))]) \
            if colors else jnp.zeros(0)

    def cost(q, m_out, n_in):
        H_ext, p0 = ext(n_in)
        sends, _, _ = realize(a, colors, q, m_out, H_ext, p0, L)
        return a.cost(q, sends[-1], jnp.concatenate([H_ext[-1], p0]))

    arr = Arrangement(
        P=a.P,
        in_dim_M=sum(in_dim(c, L, dH, dP) for c in colors), out_dim_M=m_dim,
        in_dim_N=in_dim(out_color, L, dH, dP), out_dim_N=out_dim(out_color, dH, dP),
        out_f=out_f, in_f=in_f, omega=exact(lambda q, m, n: -cost(q, m, n)),
        label=label or f"Phi{L}({a.label})",
    )
    return arr, out_color


def strategy_sends(c: Color, out: Array, l: int, H: Array, dim_H: int, dim_P: int) -> Array:
    """The round-``l`` sends of a box whose output is ``out = [s1, theta]``, having heard ``H`` (``l - 1`` rows)."""
    sd = slot_dim(dim_H, dim_P)
    s1, theta = out[:c.r * sd], out[c.r * sd:c.r * sd + c.theta_dim]
    return s1 if l == 1 else c.strategy(s1, theta, l, H)


# ---------------------------------------------------------------------------
# Leaves, the comparison lens, and what closes a group.
# ---------------------------------------------------------------------------


def leaf(c: Color, L: int, dim_H: int, dim_P: int, P: CotCoalg, emit: Callable[[Array], Array],
         cost: Callable[[Array, Array], Array] = None, label: str = "") -> Arrangement:
    """A system ``I -> B^L_c``: ``emit(q) = [s1, theta]``; ``cost(q, e)``, ``e`` what it hears at every round and its
    position; forcing ``-dk``."""
    k = (lambda q, e: jnp.asarray(0.0)) if cost is None else cost
    return Arrangement(
        P=P, in_dim_M=0, out_dim_M=0, in_dim_N=in_dim(c, L, dim_H, dim_P), out_dim_N=out_dim(c, dim_H, dim_P),
        out_f=lambda q, m_out: emit(q), in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -k(q, n_in)), label=label or f"leaf{c.r}",
    )


def iota(r: int, L: int, dim_H: int, dim_P: int) -> Arrangement:
    """The static lens ``B_r -> B^L_{moore(r)}``: forward the sends (the same at every round), backward what the slots
    hear at the last round, with the position."""
    hd = heard_dim(r, dim_H)
    return Arrangement(
        P=configuration(trivial()), in_dim_M=hd + dim_P, out_dim_M=r * slot_dim(dim_H, dim_P),
        in_dim_N=L * hd + dim_P, out_dim_N=r * slot_dim(dim_H, dim_P),
        out_f=lambda q, s: s,
        in_f=lambda q, s, n_in: jnp.concatenate([n_in[(L - 1) * hd:L * hd], n_in[L * hd:]]),
        omega=zero_form, label=f"iota{r}^{L}",
    )


def ground(c: Color, L: int, dim_H: int, dim_P: int) -> Arrangement:
    """``B^L_c -> I`` for a color with no slots: drops its parameter, gives the reference position ``0``."""
    if c.r != 0:
        raise ValueError("ground: the color shows slots")
    return Arrangement(
        P=configuration(trivial()), in_dim_M=dim_P, out_dim_M=out_dim(c, dim_H, dim_P), in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0), in_f=lambda q, m_out, n_in: jnp.zeros(dim_P),
        omega=zero_form, label="ground",
    )
