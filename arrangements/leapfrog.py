"""Leapfrog (velocity Verlet) as a two-stage coalgebra (rmk.multistage).

EXTENSION (beyond the paper). Leapfrog is ONE ``CotCoalg2`` — a single
``Store∘S ⇒ cot^{◁2}`` natural transformation, the multi-stage integrator the
remark proposes — and an arrangement whose parameter it is has ``pc2.Phi2``
dynamics, an ``pc^(2)`` morphism. This is exactly parallel to the 1-stage story
(a 1-stage ``CotCoalg`` parameter + ``functors.Phi`` gives an ``pc`` morphism).
The kick/drift/kick lives once, in the coalgebra, in the book's sign (the
covector is *added*, STATUS.md §1):

    read1(q, ξ)   = q
    advance       : ξ½ = ξ + ½ ξ_Q(q),   q' = q + sharp_q(ξ½)     (mid = (q', ξ½))
    read2(q', ξ½) = q'
    finish        : ξ' = ξ½ + ½ ξ_Q(q')                            (new = (q', ξ'))

so the two force evaluations ξ_Q(q) and ξ_Q(q') are the two ``pc^(2)`` rounds.
Constant (mass-like) ``sharp`` — the physics case. Whether ``arr → pc^(2)`` is
lax monoidal is open (pc2.py).
"""

from __future__ import annotations

import jax.numpy as jnp

from .coalgebra import CotCoalg2
from .rvect import ReactiveVectorSpace


def leapfrog(Q: ReactiveVectorSpace, init=None) -> CotCoalg2:
    """The leapfrog (velocity Verlet) two-stage coalgebra on ``(Q, sharp_Q)``,
    ``Store∘S ⇒ cot^{◁2}``; state ``(q, xi)``."""

    def advance(s, xi_Q1):
        q, xi = s
        xi_half = xi + 0.5 * xi_Q1            # half-kick
        return (q + Q.apply_sharp(q, xi_half), xi_half)   # drift -> mid = (q', ξ½)

    def finish(s, mid, xi_Q2):
        q2, xi_half = mid
        return (q2, xi_half + 0.5 * xi_Q2)    # half-kick -> (q', ξ')

    s0 = (jnp.zeros(Q.dim), jnp.zeros(Q.dim)) if init is None else init
    return CotCoalg2(
        Q_dim=Q.dim,
        init=s0,
        read1=lambda s: s[0],
        advance=advance,
        read2=lambda mid: mid[0],
        finish=finish,
        label="leapfrog",
    )
