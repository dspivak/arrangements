"""A collective whose parts supply each other, on placements (STATUS.md §3.21).

The book plan's P21 stage (b) (AGENTDOCPLAN.md §8.2), from David, 2026-09-26: "i can only do
my job if i get what i need from you". Two earlier constructions each had one half of that.
``tests/test_mutual_need.py`` (§5.9 item 12) has two parts, each of whose held form is
supplied by the other, but no placements, so nothing in it can disperse; ``collective.py``
with ``cascade.py`` (items 18, 22) has placements and an elsewhere, but its parts are supplied
by the fund and by the elsewhere and never by one another, since supply and product ride
separate channels there.

This module is the two halves in one construction, and adds no mechanism to the library: the
spatial parent's transfers already deliver to each box the kernel-weighted sum of *every
other* box's message and of the exterior's (``attention.spatial_transfers``, ``T_ii = 0``), so
a part that sends on the supply channel is supplying its neighbours already. What is new is
one part, ``supplier``, and it is the cascade's ``worker`` with two changes: it passes a share
``sigma`` of its own activity on as supply, and what it can do with what arrives is gated,

    send(a) = ((sigma a, a), 1),    k(a, e) = (b/2) a^2 - (alpha a + beta) I(e_1),
    I(r) = A r / (K_s + r),

``I`` being ``test_mutual_need.py``'s Michaelis-Menten gate moved onto the receiving side.
With ``beta = 0`` a part with nothing arriving has no drive and only its dissipation remains,
which is item 12's wiring condition -- no ungated path to a part -- inside a collective: the
only ungated supplies are the fund's wage at the origin and the elsewhere's offer at distance
``d``, and either can be removed. (With ``beta > 0`` it is not: composition drives a part by
its neighbours' need for what it sends, and that term survives at zero activity.
``tests/test_mutual_ratchet.py``, ``test_why_beta_is_zero``.)

``H = R^2`` as in ``cascade.py``: channel 1 supply, channel 2 product. The parent is
cost-free, so a placement moves only by what its own part's cost says about what arrives; since
``I`` is increasing, *the stationary placements are the stationary points of the arriving
supply* -- the gate changes the rate, never the place (``tests/test_mutual_ratchet.py``).

Two ways to run it. ``funded`` wraps the collective in the cascade's fund layer and market,
unchanged, so the wage is a balance that can be drained and killed; ``bare`` is the collective
alone, driven by a constant exterior supply at the origin (``cascade.supplied``), which is a
clamped wage, and by ``0`` when the paymaster is gone altogether.
"""

from __future__ import annotations

from typing import List, Sequence

import jax.numpy as jnp
from jax import Array

from .attention import make_input, message, split_inputs
from .cascade import DH2, leaf, market, source, unit
from .coalgebra import configuration
from .collective import Collective, Part, collective, part, split
from .rvect import euclidean
from .wiring import compose_seq


def intake(r, A: float, K_s: float):
    """``I(r) = A r / (K_s + r)``: what a part can do with the supply ``r`` that arrives."""
    return A * r / (K_s + r)


def supplier(a0: float, *, alpha: float, b: float, beta: float, sigma: float, A: float,
             K_s: float, mu: float, label: str) -> Part:
    """A part that passes a share ``sigma`` of its activity on as supply (module docstring)."""
    return part(
        send=lambda q: message(jnp.array([sigma * q[0], q[0]]), 1.0),
        cost=lambda q, e: 0.5 * b * q[0] ** 2 - (alpha * q[0] + beta) * intake(e[0], A, K_s),
        P=configuration(euclidean(1, mu), init=jnp.array([float(a0)])),
        label=label, dim_H=DH2)


def suppliers(n: int, a0, **kw) -> List[Part]:
    """``n`` suppliers ``w0..w{n-1}``; ``a0`` is one number or one per part."""
    a0 = [float(a0)] * n if isinstance(a0, (int, float)) else list(a0)
    return [supplier(a0[i], label=f"w{i}", **kw) for i in range(n)]


def bare(parts: Sequence[Part], *, positions, mobilities, tau: float, s: float, d: float,
         label: str = "mutual") -> Collective:
    """The collective alone: ``parts`` at ``positions``, and the elsewhere, a frozen source
    offering ``s`` at distance ``d``. Run it with ``collective.run`` on the exterior input
    ``cascade.supplied(g)``: ``g`` is a clamped wage at the origin, ``0`` no paymaster."""
    ps = list(positions) + [d]
    mob = list(mobilities) + [0.0]
    return collective(list(parts) + [source(s)], tau=tau,
                      cost=lambda q, m, e: 0.0 * jnp.sum(q), positions=ps,
                      integrator=lambda Q, init: configuration(Q, init=init),
                      mobility=mob, value_mobility=0.0, dim_H=DH2, label=label)


def funded(parts: Sequence[Part], *, positions, mobilities, tau: float, s: float, d: float,
           fund: dict, r_market: float, label: str = "mutual"):
    """The same collective wrapped in the cascade's fund layer and closed by its market.
    Returns ``(unit, arrangement)``; read it with ``cascade.read`` on the unit.

    ``fund`` is the cascade's, and ``record`` should be ``None`` here: the exact cancellation
    of the wage's retroactive pull needs each part's arriving total, and the fund sees only the
    collective's onward message, so the balance is propped up by the need of those it pays
    (``cascade.py``, ``fund_layer``). The gate's slope is what that term rides on, so it is
    negligible wherever the balance is far from zero, which is every run reported."""
    u = unit([leaf(p) for p in parts], positions=list(positions), mobilities=list(mobilities),
             tau=tau, fund=fund, src=source(s), src_position=d, label=label)
    return u, compose_seq(u.arr, market(r_market))


# ---------------------------------------------------------------------------
# Reading a state: who is where, doing what, on what.
# ---------------------------------------------------------------------------


def amplitudes(C: Collective, q) -> Array:
    """The parts' amplitudes, the source excluded (it has no configuration)."""
    qs, _, _ = split(C, q)
    return jnp.array([qp[0] for qp in qs[:-1]])


def placements(C: Collective, q) -> Array:
    """The parts' placements, the source's (fixed at ``d``) excluded."""
    _, pos, _ = split(C, q)
    return pos[:-1]


def arriving(C: Collective, q, e_ext) -> Array:
    """What arrives on the supply channel at each part, from the library's own ``rho``:
    ``r_i = kappa_i0 g + kappa_{i,src} s + sigma sum_{j != i} kappa_ij a_j``."""
    from .collective import emitted
    q = jnp.asarray(q, float)
    es = split_inputs(C.parent.rho(q[C.parts_dim:], emitted(C, q), e_ext),
                      C.n, C.dim_H, 1)
    return es[:-1, 0]


def supplied(level: float) -> Array:
    """A constant exterior supply ``level`` at the origin (the cascade's, re-exported)."""
    return make_input(message(jnp.array([float(level), 0.0]), 1.0), jnp.zeros(1))
