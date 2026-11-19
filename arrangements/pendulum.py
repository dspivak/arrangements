"""The n-fold pendulum by nesting one link in itself (book Ch 3 ``sec.imc_arrangements``, ``eqn.link``).

A *link* is a 1-ary arrangement ``<0|2> -> <0|2>``, a bar hanging from a support with a bob at its bottom:
its matter is the bob, a mass ``m`` in the plane (``phase``, symplectic Euler with its look-ahead); its open
internal box, the support, reads out the position ``x`` the bar hangs from; it reads out its bob's position
``q``; and its cost is the bar from ``x`` to ``q`` plus gravity on the bob. ``pivot`` is the stateless
system reading out the fixed point 0, ``drop`` the static arrangement that drops the last bob's position.
The n-fold pendulum is ``drop . link^n . pivot`` (David, 2026-10-14, "yes" to turning the first draft's
bob-above-bar link right side up).

The bar of length ``ell`` and spring constant ``kappa`` has the cost of the book's ``eqn.bar_cost``,
``kappa/(8 ell^2) (|q-x|^2 - ell^2)^2``, smooth everywhere and ``kappa/2`` times the squared stretch to
second order. A real bar's ``kappa`` is Young's modulus times area over length (steel, 1 m by 1 cm^2:
``2e7`` N/m). Claude Opus 5.5, fa95b8ae, 2026-10-14; the book's run is
``notes/runs/pendulum-fa95b8ae/`` there.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import jax.numpy as jnp
from jax import Array

from .arrangement import Arrangement, Numbers, exact, zero_form
from .coalgebra import CotCoalg, phase
from .rvect import diagonal
from .wiring import compose_seq, trivial_parameter

_Z = jnp.zeros(0)


def bar_cost(kappa: float, ell: float, x: Array, q: Array) -> Array:
    d = q - x
    return kappa / (8 * ell**2) * (jnp.dot(d, d) - ell**2) ** 2


def bob(m: float, q0: Numbers, p0: Optional[Numbers] = None, dt: float = 1.0) -> CotCoalg:
    """The mass ``m`` in the plane: state ``(q, p)``, look-ahead ``q + dt p/m``."""
    return phase(diagonal(jnp.full(2, 1.0 / m)), init=(q0, jnp.zeros(2) if p0 is None else p0), dt=dt)


def link(m: float, kappa: float, ell: float, q0: Numbers, g: float = 9.81, dt: float = 1.0) -> Arrangement:
    return Arrangement(
        P=bob(m, q0, dt=dt), in_dim_M=0, out_dim_M=2, in_dim_N=0, out_dim_N=2,
        out_f=lambda q, x: q, in_f=lambda q, x, n: _Z,
        omega=exact(lambda q, x, n: -(bar_cost(kappa, ell, x, q) + m * g * q[1])),
        label="link")


def pivot() -> Arrangement:
    """The stateless system reading out the fixed point 0, forcing 0."""
    return Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=2,
        out_f=lambda q, x: jnp.zeros(2), in_f=lambda q, x, n: _Z,
        omega=zero_form, label="pivot")


def drop() -> Arrangement:
    """The static arrangement dropping a position."""
    return Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=2, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, x: _Z, in_f=lambda q, x, n: _Z,
        omega=zero_form, label="drop")


def positions(angles: Sequence[float], ell: float) -> List[Array]:
    """Bob positions, top first, for bars at the given angles from straight down."""
    pts, p = [], jnp.zeros(2)
    for a in angles:
        p = p + ell * jnp.array([jnp.sin(a), -jnp.cos(a)])
        pts.append(p)
    return pts


def pendulum(
    angles: Sequence[float], m: float = 1.0, kappa: float = 1e4, ell: float = 1.0, g: float = 9.81, dt: float = 1e-3,
) -> Arrangement:
    """``drop . link^n . pivot``, at rest with its bars at rest length; the composite's state lists the
    bobs top first, read off by ``bob_states``."""
    pts = positions(angles, ell)
    a = pivot()
    for p in pts:
        a = compose_seq(a, link(m, kappa, ell, p, g, dt))
    return compose_seq(a, drop())


def bob_states(s: Any) -> List[Tuple[Array, Array]]:
    """The bobs' ``(q, p)``, top first, from a state of ``pendulum`` (the pivot and drop store nothing).
    ``s`` is the composite's pytree of nested tuples, so ``Any``."""
    import jax
    leaves = [a for a in jax.tree_util.tree_leaves(s) if a.size]
    return [(leaves[2 * i], leaves[2 * i + 1]) for i in range(len(leaves) // 2)]
