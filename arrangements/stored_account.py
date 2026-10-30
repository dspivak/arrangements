"""A stored account, averaged over its bursts (plan P41 (f), Claude Opus 5.5, f0638d8e, 2026-10-30; the book's
``notes/runs/p41f-f0638d8e/RESULTS.md`` §3).

The burst room (``acceptable.py``) moves its chairs with an account stored and released in bursts: every coupling
there is one-way. The candidate theorem: averaged over its bursts, it is the pushback a two-way seat would give,
received through chairs whose matter has a dead zone (static friction). This module builds that right-hand side:

* ``friction_chairs``: the two chairs as one configuration integrator on ``R^2`` whose response to the force ``xi`` is
  ``r(xi) = k d(|xi|^2 / 2a) xi``, ``d`` a function of the account (zero below its threshold). With ``d``
  nondecreasing, ``r`` is the gradient of a convex function that is zero on a ball: a matter with a dead zone.
* ``friction_room``: the people as in the burst room, the seat two-way (``two_clocks.seat``), talk and the routine,
  and the chairs filled by ``friction_chairs``. Its forcing is exact (every coupling is a cost); only the chairs'
  integrator differs from a room with two-way seats. Position ``(q1, q2, w1, w2)``; ``I -> <1|0>``, the routine on
  its port.
* ``duty_singular``: the singular-limit duty ``d(s) = T_m / (T_c + T_m)`` of the store and switch (RESULTS §3 (ii)).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import jax.numpy as jnp
import numpy as np

from .acceptable import BURST
from .arrangement import Arrangement
from .coalgebra import CotCoalg
from .two_clocks import matter, person, route, routine, seat, talk
from .wiring import compose_seq, tensor_arrangements


def duty_singular(s, sigma=BURST["sigma"], lam=BURST["lam"], rho=BURST["rho"], z_on=40.0, z_off=20.0):
    """The fraction of time the switch is on, at a held account ``s``, with the switch instantaneous."""
    s = np.asarray(s, dtype=float)
    up = sigma * s - lam * z_on
    Tc = np.where(up > 0, np.log(np.maximum(sigma * s - lam * z_off, 1e-300) / np.maximum(up, 1e-300)) / lam, np.inf)
    Tm = np.log((rho - sigma * s + lam * z_on) / (rho - sigma * s + lam * z_off)) / lam
    return np.where(up > 0, Tm / (Tc + Tm), 0.0)


def tabulated(s_grid, d_grid) -> Callable:
    """``d`` as the piecewise-linear interpolation of a table, ``0`` below the table's first account."""
    sg, dg = jnp.asarray(np.asarray(s_grid, float)), jnp.asarray(np.asarray(d_grid, float))
    return lambda s: jnp.where(s < sg[0], 0.0, jnp.interp(s, sg, dg))


def friction_chairs(w0, duty: Callable, k: float = BURST["G"] / BURST["a"], a: float = BURST["a"]) -> CotCoalg:
    """Both chairs, one integrator on ``R^2``: state the positions; told ``xi`` for a duration ``tau``, they move
    ``tau k d(|xi|^2 / 2a) xi``. One account ``|xi|^2 / 2a`` for the pair: they stick or slide together."""
    def r(xi):
        return k * duty(jnp.sum(xi ** 2) / (2.0 * a)) * xi
    return CotCoalg(
        Q_dim=2, init=jnp.asarray(np.asarray(w0, float)), position=lambda s: s, step=lambda s, xi: s + r(xi),
        label="friction_chairs", timed_step=lambda s, xi, tau: s + tau * r(xi),
    )


def friction_room(duty: Callable, q=(-2.5, 2.5), w=(-2.5, 2.5), **kw) -> Arrangement:
    """``I -> <1|0>``: people, two-way seats, talk and the routine, chairs with the frictional response."""
    p = {**BURST, **kw}
    parts = tensor_arrangements([
        person(q[0], 3), person(q[1], 3),
        matter(friction_chairs(w, duty, p["G"] / p["a"], p["a"]), 1, "chairs"),
    ])
    # wires: q1 0-2, q2 3-5, w1 6, w2 7 -> seat1 (q1, w1), seat2 (q2, w2), talk (q1, q2), routine (q1, q2)
    wired = compose_seq(parts, route(8, [0, 6, 3, 7, 1, 4, 2, 5]))
    couplings = tensor_arrangements([seat(p["a"]), seat(p["a"]), talk(p["c"], p["ell"]), routine()])
    return replace(compose_seq(wired, couplings), label="friction_room")


# ---------------------------------------------------------------------------
# Static friction above sliding friction: a chair that is a mass, pulled steadily by the stored account.
# ---------------------------------------------------------------------------


def _smooth_step(t):
    """``0`` for ``t <= 0``, ``1`` for ``t >= 1``, C^infinity between (flat at both ends)."""
    t = jnp.clip(t, 1e-12, 1.0 - 1e-12)
    a, b = jnp.exp(-1.0 / t), jnp.exp(-1.0 / (1.0 - t))
    return a / (a + b)


def _shrink(x, theta, eps=1e-9):
    """``sign(x) max(|x| - theta, 0)`` made C^infinity: exactly ``0`` for ``|x| <= theta`` (a smooth function can be
    flat, so sticking needs no non-smooth step), and ``|x| - theta`` beyond ``theta + eps``."""
    u = jnp.abs(x) - theta
    return jnp.sign(x) * jnp.where(u > 0, u * _smooth_step(u / eps), 0.0)


def friction_mass(w0: float, m: float, c: float, F_s: float, F_k: float, v_s: float) -> CotCoalg:
    """A chair that is a mass ``m`` on a floor: state ``(w, p)``, look-ahead ``w + tau p / m``; told the force ``xi`` for
    a duration ``tau``, its momentum takes ``tau (xi - c p / m)`` and then loses up to ``tau F(p)`` to friction,
    ``F(p) = F_k + (F_s - F_k) exp(-(p / m v_s)^2)``: static friction ``F_s`` at rest, sliding ``F_k`` once moving.
    A chair at rest told less than ``F_s`` stays at rest exactly."""
    def F(p):
        return F_k + (F_s - F_k) * jnp.exp(-(p / (m * v_s)) ** 2)

    def timed_step(s, xi, tau):
        w, p = s
        x = p + tau * (xi - c * p / m)
        return (w + tau * p / m, _shrink(x, tau * F(p)))

    return CotCoalg(
        Q_dim=1, init=(jnp.array([float(w0)]), jnp.array([0.0])), position=lambda s: s[0],
        step=lambda s, xi: timed_step(s, xi, 1.0), label="friction_mass",
        look_ahead=lambda s, tau: s[0] + tau * s[1] / m, timed_step=timed_step,
    )


def _force(k: int, force, label: str) -> Arrangement:
    from .acceptable import _force_coupling
    return _force_coupling(k, force, label)


def store_pull(kappa: float) -> Arrangement:
    """``(z, q1, w1, q2, w2)``: force ``kappa z (q_i - w_i)`` on chair ``i``; nothing on the store or the people. The
    stored discomfort pulls steadily; no switch."""
    def f(m):
        return jnp.array([0.0, 0.0, kappa * m[0] * (m[1] - m[2]), 0.0, kappa * m[0] * (m[3] - m[4])])
    return _force(5, f, f"store_pull({kappa:g})")


STICK = dict(m=20.0, c_v=40.0, F_k=0.9, v_s=0.002)   # c_v: the floor's viscous drag (BURST's c is talk)


def stick_slip_room(q=(-2.5, 2.5), w=(-2.5, 2.5), z0=0.0, **kw) -> Arrangement:
    """``I -> <1|0>``: the people as in the burst room, the store charged by their strain and leaking, the store
    pulling each chair steadily (``store_pull``), each chair a ``friction_mass``. No switch: the threshold is the
    chairs' static friction ``F_s = kappa * sigma * a / lam`` by default, so that, as in the burst room, nothing moves
    while each person sits within 1 of their chair. Position ``(q1, q2, w1, p1, w2, p2, z)``."""
    from .acceptable import seat_oneway, strain
    p = {**BURST, **STICK, "kappa": 0.0125, **kw}
    F_s = p.get("F_s", p["kappa"] * p["sigma"] * p["a"] / p["lam"])
    chair = lambda w0: friction_mass(w0, p["m"], p["c_v"], F_s, p["F_k"] * F_s, p["v_s"])
    parts = tensor_arrangements([
        person(q[0], 5), person(q[1], 5),
        matter(chair(w[0]), 3, "chair"), matter(chair(w[1]), 3, "chair"),
        matter(CotCoalg(Q_dim=1, init=jnp.array([float(z0)]), position=lambda s: s, step=lambda s, xi: s + xi,
                        label="store", timed_step=lambda s, xi, tau: s + tau * xi), 3, "store"),
    ])
    # wires: q1 0-4, q2 5-9, w1 10-12, w2 13-15, z 16-18
    idx = [0, 10, 5, 13,            # seat1 (q1, w1), seat2 (q2, w2)
           1, 6, 2, 7,              # talk, routine (q1, q2)
           3, 11, 8, 14, 16,        # strain (q1, w1, q2, w2, z)
           17,                      # leak (z)
           18, 4, 12, 9, 15]        # store_pull (z, q1, w1, q2, w2)
    wired = compose_seq(parts, route(19, idx))
    from .two_clocks import _cost_coupling
    couplings = tensor_arrangements([
        seat_oneway(p["a"]), seat_oneway(p["a"]), talk(p["c"], p["ell"]), routine(),
        strain(p["a"], p["sigma"]), _cost_coupling(1, lambda mm: 0.5 * p["lam"] * mm[0] ** 2, "leak"),
        store_pull(p["kappa"]),
    ])
    return replace(compose_seq(wired, couplings), label="stick_slip_room")
