"""A rod tied by springs to data points, composed (book plan P43 (b) job 7; ``tests/test_regression_springs.py``).

The picture (David, 2026-09-27): pins at the data points ``(x_i, y_i)``; a stiff rod on a vertical rail at
``x = 0`` with height ``h`` and tilt ``theta``; at each ``x_i`` a zero-length spring from the pin to the rod.
The test ran it as plain numpy (P43 (a), cb545383). Here each segment is a box:

* the ROD, state ``(h, theta)``, its integrator saying how it moves: in syrup the configuration integrator
  with response ``diag(dt / drag_h, dt / drag_theta)``; in air the damped phase integrator with mass and
  moment of inertia folded with the step (``coalgebra.damped``, friction ``c = dt friction / mass``); it
  presents ``(h, theta)`` to every slider or ring;
* a PIN per point, a box with no state presenting ``(x_i, y_i)``;
* a SLIDER per point, a gray box presenting the rod's point above ``x_i``, ``(x_i, h + x_i tan theta)``; or a
  free RING, a gray box presenting the foot of the perpendicular from the pin to the rod;
* a SPRING per point, a 2-ary closed box with cost ``(k/2)|a - b|^2`` between the slider (or ring) and the pin.

``rod_arrangement`` composes them and is closed. ``Phi`` runs it. The springs' total cost is
``U(h, theta)``, the squared error of the line ``h + x tan theta`` (vertical sliders) or of its perpendicular
distances (rings), so the rod comes to rest on the least-squares or the total-least-squares line.
"""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp

from .arrangement import Arrangement, exact, zero_form
from .coalgebra import configuration, damped
from .network import fan_box, map_box
from .rvect import diagonal
from .wiring import compose_seq, tensor_arrangements, trivial_parameter


def rod_box(h0: float, th0: float, *, dt: float, drags=None, mass_inertia=None, friction: float = 0.0,
            n_out: int = 1) -> Arrangement:
    """The rod ``<0|0> -> <0|2 n_out>``: in syrup (``drags``) or in air (``mass_inertia``, ``friction``)."""
    q0 = jnp.array([h0, th0], dtype=float)
    if drags is not None:
        P = configuration(diagonal(dt / jnp.asarray(drags, dtype=float)), init=q0)
    else:
        mi = jnp.asarray(mass_inertia, dtype=float)
        if not bool(jnp.allclose(mi, mi[0])):
            raise ValueError("rod_box: one friction coefficient needs equal mass and inertia")
        P = damped(diagonal(dt / mi), dt * friction / float(mi[0]), init=(q0, jnp.zeros(2)))
    return Arrangement(
        P=P, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=2 * n_out,
        out_f=lambda q, m: jnp.tile(q, n_out), in_f=lambda q, m, n: jnp.zeros(0), omega=zero_form,
        label="rod",
    )


def pin_box(x: float, y: float) -> Arrangement:
    """A pin ``<0|0> -> <0|2>`` at ``(x, y)``: no state; the force that reaches it moves nothing."""
    return map_box(lambda v: jnp.array([x, y]), 0, 2, "pin")


def slider_box(x: float) -> Arrangement:
    """``<0|2> -> <0|2>``: from ``(h, theta)``, the rod's point above ``x``, ``(x, h + x tan theta)``."""
    return map_box(lambda v: jnp.stack([jnp.asarray(x, float), v[0] + x * jnp.tan(v[1])]), 2, 2, "slider")


def ring_box() -> Arrangement:
    """``<0|4> -> <0|2>``: from ``(h, theta)`` and the pin ``(x, y)``, the foot of the perpendicular from
    the pin to the rod (where a free ring on a zero-length spring comes to rest)."""
    def foot(v):
        h, th, p = v[0], v[1], v[2:4]
        base, d = jnp.array([0.0, h]), jnp.array([jnp.cos(th), jnp.sin(th)])
        return base + jnp.dot(p - base, d) * d

    return map_box(foot, 4, 2, "ring")


def spring_box(k: float, scale: float = 1.0) -> Arrangement:
    """``<0|2> (x) <0|2> -> I``: a zero-length spring of constant ``k`` (cost ``scale (k/2)|a - b|^2``;
    ``scale`` folds a step into the cost for the damped integrator)."""
    return Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=4, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m: jnp.zeros(0), in_f=lambda q, m, n: jnp.zeros(0),
        omega=exact(lambda q, m, n: -scale * 0.5 * k * jnp.sum((m[:2] - m[2:]) ** 2)),
        label="spring",
    )


def rod_arrangement(x: Sequence[float], y: Sequence[float], rod: Arrangement, *, k: float = 1.0,
                    rings: bool = False, scale: float = 1.0) -> Arrangement:
    """The closed arrangement: ``rod`` (built with ``n_out = len(x)``), a pin per point, a slider (or a
    ring) per point, a spring per point between them. Its position is ``(h, theta)``."""
    n = len(x)
    pins = [pin_box(float(a), float(b)) for a, b in zip(x, y)]
    parts = tensor_arrangements([rod] + pins)          # wires: (h, th) x n, then pins (x_i, y_i)
    if rings:
        # each ring hears (h, th) and its pin; each spring hears the ring's foot and the pin
        idx = [w for i in range(n) for w in (2 * i, 2 * i + 1, 2 * n + 2 * i, 2 * n + 2 * i + 1)]
        idx += [w for i in range(n) for w in (2 * n + 2 * i, 2 * n + 2 * i + 1)]
        feet = tensor_arrangements([ring_box() for _ in range(n)] + [fan_box(2 * n, range(2 * n), "pins")])
    else:
        idx = list(range(4 * n))
        feet = tensor_arrangements([slider_box(float(a)) for a in x] + [fan_box(2 * n, range(2 * n), "pins")])
    routed = compose_seq(compose_seq(parts, fan_box(4 * n, idx, "route")), feet)
    # wires now: n feet (2 each), then n pins (2 each); pair foot i with pin i
    pair = [w for i in range(n) for w in (2 * i, 2 * i + 1, 2 * n + 2 * i, 2 * n + 2 * i + 1)]
    springs = tensor_arrangements([spring_box(k, scale) for _ in range(n)])
    return compose_seq(compose_seq(routed, fan_box(4 * n, pair, "pair")), springs)
