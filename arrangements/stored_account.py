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
