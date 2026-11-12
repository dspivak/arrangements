"""Cautious integrators and Euler integrators (book Ch 3 ``def.cautious_integrator``, Ch 5 ``def.euler_integrator``).

A cautious integrator ``(Q, S, r, u)`` is an integrator whose update is defined only on an open set ``dom(u)`` of
triples ``(s, tau, load)``, its *allowed steps*, containing every triple of duration ``0``. In the library the update
off ``dom(u)`` returns a state of ``nan``: a step is allowed exactly when every leaf of the new state is finite
(``allowed``). Since ``nan`` stays in the part that made it, the step of a tensor, or of any composite, is allowed
exactly when each part's is, which is the domain of the tensor in ``prop.cautious_tensor``; no code says so beyond the
tensor ``CotCoalg.tensor`` already has. (A ``nan`` that arithmetic makes, a blow-up, reads the same; the parts here
make none on their domains.)

``euler_integrator(read, v, ...)`` is the Euler integrator of an integrand ``(Q, S, rbar, v)`` with the affine
stepping structure ``s + tau w`` of Ch 5 ``ex.affine_stepping``, on all of ``S = R^n`` or, given ``domain``, on the
open subset ``U`` where ``domain`` is positive: look-ahead ``rbar(s)`` at every duration, update ``s + tau v_s(load)``,
allowed where it lands in ``U`` (``eqn.euler_integrator``). Two of the book's examples are such integrators on
``U = R_{>0}``, and are built here as such:

* ``kelvin_rod(C, theta0)``: Ch 3 ``ex.kelvin_rod``, a rod segment whose temperature in kelvin stays positive, the
  Euler integrator of the rod segment's integrand ``v_theta(load) = load / C``;
* ``species(q0, degree)``: each species of Ch 3 ``ex.reaction_network``, the integrand ``v_q(load) = load``, in a box
  that transfers out its concentration once per arrow at it.

Everything is on ``R^n`` (the admissible subcategory of plan §5.3 (c)); ``R_{>0}`` is the open subset of ``R``.
Claude Opus 5.5, session 00f2d232, 2026-11-12.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from jax import Array

from .arrangement import Arrangement, zero_form
from .coalgebra import CotCoalg

_z = jnp.zeros(0)


def cautious(P: CotCoalg, allows: Callable[[Any, Array, Any], Array], label: Optional[str] = None) -> CotCoalg:
    """``P`` with allowed steps ``{(s, tau, load) | allows(s, load, tau)}``: off them the update returns ``nan``.

    ``allows`` must hold at every duration-zero triple (``def.cautious_integrator``); ``P`` must have a duration law.
    """
    if P.timed_step is None:
        raise TypeError("cautious: the integrator needs a duration law (timed_step)")

    def timed_step(s, xi, tau):
        new = P.timed_step(s, xi, tau)
        ok = allows(s, xi, tau)
        return jax.tree_util.tree_map(lambda x: jnp.where(ok, x, jnp.nan), new)

    return CotCoalg(Q_dim=P.Q_dim, init=P.init, position=P.position, step=lambda s, xi: timed_step(s, xi, 1.0),
                    label=label or f"cautious({P.label})", timed_step=timed_step, look_ahead=P.look_ahead)


def allowed(state) -> bool:
    """Whether the step that produced ``state`` was allowed: no leaf is ``nan``."""
    return all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(state))


def euler_integrator(read: Callable[[Array], Array], v: Callable[[Array, Array], Array], Q_dim: int, init: Array,
                     domain: Optional[Callable[[Array], Array]] = None, label: str = "euler") -> CotCoalg:
    """The Euler integrator of the integrand ``(Q, S, read, v)`` with the affine stepping ``s + tau w``
    (``eqn.euler_integrator``, ``eqn.affine_stepping``): look-ahead ``read(s)`` at every duration, update
    ``s + tau v(s, load)``. With ``domain``, the state space is the open set ``U = {s | domain(s) > 0}`` (every
    coordinate), and a step is allowed when it lands in ``U``; without, every step is allowed."""
    s0 = jnp.asarray(init, dtype=float)
    timed = lambda s, xi, tau: s + tau * v(s, xi)
    P = CotCoalg(Q_dim=Q_dim, init=s0, position=read, step=lambda s, xi: timed(s, xi, 1.0), label=label,
                 timed_step=timed, look_ahead=lambda s, tau: read(s))
    if domain is None:
        return P
    return cautious(P, lambda s, xi, tau: jnp.all(domain(timed(s, xi, tau)) > 0), label=label)


def positive(s: Array) -> Array:
    """``R_{>0}``, coordinatewise: the open set where every coordinate is positive."""
    return s


def kelvin_rod(C: float, theta0: float) -> Arrangement:
    """Ch 3 ``ex.kelvin_rod`` in a box: a rod segment of heat capacity ``C`` whose temperature ``theta`` in kelvin must
    stay positive; it transfers out ``theta``; a step is allowed when ``theta + tau load / C > 0``."""
    P = euler_integrator(lambda s: s, lambda s, xi: xi / C, 1, [theta0], domain=positive, label=f"kelvin_rod({C:g})")
    return Arrangement(P=P, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1, out_f=lambda q, m: q,
                       in_f=lambda q, m, n: _z, omega=zero_form, label=f"kelvin_rod({C:g})")


def species(q0: float, degree: int, label: str = "species") -> Arrangement:
    """A species of Ch 3 ``ex.reaction_network``: the output-only uninclined configuration system on ``R_{>0}``, update
    ``q + tau load``, allowed where it lands in ``R_{>0}``; its output wire split once per arrow at it (``degree``)."""
    P = euler_integrator(lambda s: s, lambda s, xi: xi, 1, [q0], domain=positive, label=label)
    return Arrangement(P=P, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=degree, out_f=lambda q, m: jnp.tile(q, degree),
                       in_f=lambda q, m, n: _z, omega=zero_form, label=label)
