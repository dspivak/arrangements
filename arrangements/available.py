"""Forms available over a slow motion, along one-way nesting (book plan P41 (j); Claude Opus 5.5, 836d12ce, 2026-11-05).

The furniture room of ``two_clocks`` with a *guest*: a quick part that reads the people's separation one-way, its
cost ``(z^2 - 1)^2 / 4 - t z`` with the tilt ``t = mu (d - d_c)``, the force on ``z`` only, so nothing returns to the
people. The guest can take either stance while ``|t| < 2 / (3 sqrt 3)`` and only ``z > 0`` past it. With ``lamp`` the
guest also reads a quick double well ``y``, ``t = mu (d - d_c) + nu y``, a reader with two sources.

``one_way_room`` is the control: the chairs have their own well ``kappa (D - D_a)^2 (D - D_b)^2 / 2`` and the people
are seated one-way, so the slow motion descends to the chairs, which is what the furniture room's does not.

These are arrangements ``I -> <1|0>``, the routine on the one input port, run by ``two_clocks.routine_step``.
Predictions and results: the book repo's ``notes/runs/p41j-836d12ce/``.

A forward run cannot stay on a repelling sheet: rounding leaves it first, so a connection along one is missed and an
order read off forward runs comes out wrong with no warning (the room and lamp's first count, 49 against the hand's 48;
that file's RESULTS, "Where a forward run could not see the order", says what stood in).
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp

from .arrangement import Arrangement
from .coalgebra import configuration
from .rvect import euclidean
from .two_clocks import _none, chair_integrator, matter, person, route, routine, seat, slowed, talk, tidy
from .wiring import compose_seq, tensor_arrangements, trivial_parameter

D_C = 0.646


def _part(z0: float, degree: int, label: str) -> Arrangement:
    return matter(configuration(euclidean(1, 1.0), init=jnp.array([z0])), degree, label)


def _reading_coupling(k: int, n_read: int, cost, label: str) -> Arrangement:
    """``<0|k> -> I``: cost on all ``k`` wires, with force only on the wires after the first ``n_read``. The first
    ``n_read`` are read one-way: zero load returns along them."""
    def omega(s, m, n):
        g = jax.grad(lambda rest: cost(jnp.concatenate([jax.lax.stop_gradient(m[:n_read]), rest])))(m[n_read:])
        return jnp.zeros_like(s), jnp.concatenate([jnp.zeros(n_read), -g]), jnp.zeros_like(n)
    return Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=k, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0), in_f=_none, omega=omega, label=label,
    )


def _well(z, t):
    return 0.25 * (z ** 2 - 1.0) ** 2 - t * z


def guest(mu: float, d_c: float = D_C) -> Arrangement:
    """``<0|3> -> I`` on ``(q1, q2, z)``: the people read one-way."""
    return _reading_coupling(3, 2, lambda m: _well(m[2], mu * (m[1] - m[0] - d_c)), f"guest({mu:g})")


def lamp_guest(mu: float, nu: float, d_c: float = D_C) -> Arrangement:
    """``<0|4> -> I`` on ``(q1, q2, y, z)``: the people and the lamp read one-way."""
    return _reading_coupling(4, 3, lambda m: _well(m[3], mu * (m[1] - m[0] - d_c) + nu * m[2]),
                             f"lamp_guest({mu:g},{nu:g})")


def lamp_well() -> Arrangement:
    """``<0|1> -> I`` on ``y``: the lamp's own double well."""
    return _reading_coupling(1, 0, lambda m: 0.25 * (m[0] ** 2 - 1.0) ** 2, "lamp_well")


def _chairs(eps: float, w, degree: int = 2):
    return [matter(slowed(chair_integrator(w[i]), eps), degree, "chair") for i in range(2)]


def room_with_guest(eps: float, mu: float, q=(-2.5, 2.5), w=(-2.5, 2.5), z=1.0, a=1.0, c=3.0, ell=1.0, b=2.6,
                    w0=(-2.5, 2.5)) -> Arrangement:
    """The furniture room and its guest. State position ``(q1, q2, w1, w2, z)``."""
    parts = tensor_arrangements([person(q[0], 4), person(q[1], 4), *_chairs(eps, w), _part(z, 1, "guest part")])
    # wires: p1 x4 (0-3), p2 x4 (4-7), c1 x2 (8,9), c2 x2 (10,11), z (12)
    wired = compose_seq(parts, route(13, [0, 8, 4, 10, 1, 5, 2, 6, 9, 11, 3, 7, 12]))
    couplings = tensor_arrangements([seat(a), seat(a), talk(c, ell), routine(), tidy(b, w0[0]), tidy(b, w0[1]),
                                     guest(mu)])
    return replace(compose_seq(wired, couplings), label=f"room_with_guest(eps={eps:g},mu={mu:g})")


def room_lamp(eps: float, q=(-2.5, 2.5), w=(-2.5, 2.5), y=1.0, a=1.0, c=3.0, ell=1.0, b=2.6,
              w0=(-2.5, 2.5)) -> Arrangement:
    """The furniture room beside the lamp. State position ``(q1, q2, w1, w2, y)``."""
    parts = tensor_arrangements([person(q[0], 3), person(q[1], 3), *_chairs(eps, w), _part(y, 1, "lamp")])
    wired = compose_seq(parts, route(11, [0, 6, 3, 8, 1, 4, 2, 5, 7, 9, 10]))
    couplings = tensor_arrangements([seat(a), seat(a), talk(c, ell), routine(), tidy(b, w0[0]), tidy(b, w0[1]),
                                     lamp_well()])
    return replace(compose_seq(wired, couplings), label=f"room_lamp(eps={eps:g})")


def room_lamp_guest(eps: float, mu: float, nu: float, q=(-2.5, 2.5), w=(-2.5, 2.5), y=1.0, z=1.0, a=1.0, c=3.0,
                    ell=1.0, b=2.6, w0=(-2.5, 2.5)) -> Arrangement:
    """The guest reading two sources, the room and the lamp. State position ``(q1, q2, w1, w2, y, z)``."""
    parts = tensor_arrangements([person(q[0], 4), person(q[1], 4), *_chairs(eps, w), _part(y, 2, "lamp"),
                                 _part(z, 1, "guest part")])
    # wires: p1 x4 (0-3), p2 x4 (4-7), c1 x2 (8,9), c2 x2 (10,11), y x2 (12,13), z (14)
    wired = compose_seq(parts, route(15, [0, 8, 4, 10, 1, 5, 2, 6, 9, 11, 12, 3, 7, 13, 14]))
    couplings = tensor_arrangements([seat(a), seat(a), talk(c, ell), routine(), tidy(b, w0[0]), tidy(b, w0[1]),
                                     lamp_well(), lamp_guest(mu, nu)])
    return replace(compose_seq(wired, couplings), label=f"room_lamp_guest(eps={eps:g},mu={mu:g},nu={nu:g})")


def chair_well(kappa: float = 1.0, D_a: float = 3.8, D_b: float = 5.0) -> Arrangement:
    """``<0|2> -> I`` on ``(w1, w2)``: the chairs' own cost on their separation, and their centre held at 0."""
    def cost(m):
        D = m[1] - m[0]
        return 0.5 * kappa * (D - D_a) ** 2 * (D - D_b) ** 2 + 0.5 * (m[0] + m[1]) ** 2
    return _reading_coupling(2, 0, cost, f"chair_well({kappa:g},{D_a:g},{D_b:g})")


def one_way_seat(a: float = 1.0) -> Arrangement:
    """``<0|2> -> I`` on ``(w, q)``: the person is pulled to the chair; the chair is read one-way."""
    return _reading_coupling(2, 1, lambda m: 0.5 * a * (m[1] - m[0]) ** 2, f"one_way_seat({a:g})")


def one_way_room(eps: float, q=(-2.5, 2.5), w=(-2.5, 2.5), a=1.0, c=3.0, ell=1.0) -> Arrangement:
    """The control: chairs with their own well, read one-way by the people. State position ``(q1, q2, w1, w2)``."""
    parts = tensor_arrangements([person(q[0], 3), person(q[1], 3), *_chairs(eps, w)])
    # wires: p1 x3 (0-2), p2 x3 (3-5), c1 x2 (6,7), c2 x2 (8,9)
    wired = compose_seq(parts, route(10, [6, 0, 8, 3, 1, 4, 2, 5, 7, 9]))
    couplings = tensor_arrangements([one_way_seat(a), one_way_seat(a), talk(c, ell), routine(), chair_well()])
    return replace(compose_seq(wired, couplings), label=f"one_way_room(eps={eps:g})")


def chairs_alone(eps: float, w=(-2.5, 2.5)) -> Arrangement:
    """The one-way room's source: the chairs and their well, ``I -> <1|0>`` with the routine's port unused.
    State position ``(w1, w2)``."""
    parts = tensor_arrangements(_chairs(eps, w, degree=1))
    unused = Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=0, in_dim_N=1, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0), in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=lambda s, m, n: (jnp.zeros_like(s), jnp.zeros(0), jnp.zeros_like(n)), label="unused routine port")
    couplings = tensor_arrangements([chair_well(), unused])
    return replace(compose_seq(parts, couplings), label=f"chairs_alone(eps={eps:g})")
