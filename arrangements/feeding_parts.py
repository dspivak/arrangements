"""The general parts of the feeding note, each a library arrangement.

The note is the book repo's ``background/feeding-imc/feeding-imc.tex``; a name in
double backquotes after "note" below is a label there. Everything here is a morphism of
IMC's Cartesian tangent subcategory in the array core: an ``Arrangement`` whose integrator
responds to the load it receives and whose inclination field is any one-form, exact or not.

Two kinds of primitive, as in the note's ``sec.integrands`` and ``sec.push``:

* an integrand, ``rod_segment`` or ``mass``: stores a state, sends its readout through one
  output, and responds to the load that comes back along that output;
* ``push``: stores nothing; inputs ``(F, x)``, inclination field ``F dx``.

Everything else is composed from these by the library's own ``compose_seq`` and
``tensor_arrangements`` (``fill``), so each part records its factors and
``explore.inspect`` shows them:

* ``pushes``    -- a finite sum of pullbacks of ``F dx`` (note ``prop.push_generates``: the
  homomorphism of commutative monoids from the free commutative monoid on smooth maps
  ``V -> R^2`` to one-forms on ``V``, ``(c, h) |-> c dh``; the empty list goes to the zero
  form and a concatenation to the sum, which is what pullback-and-sum computes);
* ``spring``    -- note ``eqn.spring``, two pushes;
* ``dragger``   -- note ``eqn.dragger``, a memory and a spring;
* ``damper``    -- note ``eqn.damper``, a dragger and a return spring;
* ``motor``     -- note ``eqn.motor``, two pushes; ``motor_at_rate`` is it sent the push that
  moves its store at a chosen rate;
* ``sweep``     -- the note's sweep, two motors;
* ``send`` and ``discard`` -- a part with a constant sent in on some input wires, or with
  some outputs unread (the first paragraph of the note's ``sec.general``).

Built 2026-11-19 for the book's P109 (e) (Claude Fable 5.2, 6a32afbb). Nothing here is used
by ``feeding.py`` yet: its casing and receiver are still single boxes with typed-out fields.
"""
from __future__ import annotations

from typing import Callable, Mapping, Sequence, Tuple

import jax.numpy as jnp
from jax import Array

from .arrangement import Arrangement, zero_form
from .coalgebra import CotCoalg
from .wiring import compose_seq, tensor_arrangements, trivial_parameter

EMPTY = jnp.zeros(0)

# A term of a sum of pushes: a name, a coefficient c : V -> R and a target h : V -> R.
Term = Tuple[str, Callable[[Array], Array], Callable[[Array], Array]]


# ---------------------------------------------------------------------------
# Integrands (note sec.integrands).
# ---------------------------------------------------------------------------


def rod_segment(capacity: float, initial: float, label: str = "rod segment") -> Arrangement:
    """A rod-segment integrand ``1 -> <0|1>``: stores one number, sends it out, and moves it
    at ``1/capacity`` times the load that comes back (book Ch 6 ``ex.tangent_heat``)."""
    if capacity <= 0:
        raise ValueError("a rod segment's capacity is positive")

    def advance(s: Array, xi: Array, duration: float) -> Array:
        return s + duration * xi / capacity

    P = CotCoalg(1, jnp.array([float(initial)]), lambda s: s,
                 lambda s, xi: advance(s, xi, 1.0), label, timed_step=advance)
    return Arrangement(P, 0, 0, 0, 1, lambda q, m: q, lambda q, m, n: EMPTY, zero_form, label)


def mass(m: float, position: float, momentum: float = 0.0, label: str = "mass") -> Arrangement:
    """A mass ``1 -> <0|1>``: stores ``(q, p)``, sends out ``q`` and forgets ``p``, and
    responds to a load ``xi`` by ``(p/m, xi)`` (book Ch 6 ``exc.tangent_mass``)."""
    if m <= 0:
        raise ValueError("a mass is positive")

    def advance(s: Array, xi: Array, duration: float) -> Array:
        return s + duration * jnp.array([s[1] / m, xi[0]])

    P = CotCoalg(1, jnp.array([float(position), float(momentum)]), lambda s: s[:1],
                 lambda s, xi: advance(s, xi, 1.0), label, timed_step=advance)
    return Arrangement(P, 0, 0, 0, 1, lambda q, m_out: q, lambda q, m_out, n: EMPTY, zero_form, label)


# ---------------------------------------------------------------------------
# The one stateless box, and the lens that feeds parts (note sec.push).
# ---------------------------------------------------------------------------


def push(label: str = "push") -> Arrangement:
    """The stateless box ``1 -> <2|0>`` with inputs ``(F, x)`` and inclination field ``F dx``:
    it pushes back ``F`` through ``x`` and ``0`` through ``F`` (note ``sec.push``)."""
    return Arrangement(trivial_parameter(), 0, 0, 2, 0, lambda q, m: EMPTY, lambda q, m, n: EMPTY,
                       lambda q, m, n: (EMPTY, EMPTY, jnp.stack([jnp.zeros_like(n[0]), n[0]])), label)


def transfer(label: str, inner_in: int, inner_out: int, outer_in: int, outer_out: int,
             route: Callable[[Array, Array], Array],
             publish: Callable[[Array], Array] = lambda m: EMPTY) -> Arrangement:
    """A stateless lens with zero inclination field, ``<inner_in|inner_out> -> <outer_in|outer_out>``.

    ``route(m, n)`` computes the parts' inputs from the parts' outputs ``m`` and the outer
    inputs ``n`` (the note's circles: any smooth map); ``publish(m)`` computes the outer
    outputs from the parts' outputs. Loads come back along the transposes of their
    derivatives, which ``compose_seq`` takes; none is written here.
    """
    return Arrangement(trivial_parameter(), inner_in, inner_out, outer_in, outer_out,
                       lambda q, m: publish(m), lambda q, m, n: route(m, n), zero_form, label)


def fill(parts: Sequence[Arrangement], wiring: Arrangement) -> Arrangement:
    """The parts side by side, their interfaces filled into ``wiring``: ``wiring . (part_1 (x) ...)``."""
    return compose_seq(tensor_arrangements(list(parts)), wiring)


def pushes(label: str, dim: int, terms: Sequence[Term]) -> Arrangement:
    """The stateless part ``1 -> <dim|0>`` with inclination field ``sum_i c_i dh_i``.

    One ``push`` per term, fed by the circles ``c_i`` and ``h_i`` (note ``eqn.push_term``,
    ``eqn.pullback_term``). No derivative of a coefficient appears, because ``push`` is
    load-free at ``F``. With no terms it is the zero field on ``R^dim``.
    """
    if not terms:
        return Arrangement(trivial_parameter(), 0, 0, dim, 0, lambda q, m: EMPTY,
                           lambda q, m, n: EMPTY, zero_form, label)
    boxes = [push(f"push: {name}") for name, _, _ in terms]

    def route(m: Array, v: Array) -> Array:
        return jnp.stack([f(v) for _, c, h in terms for f in (c, h)])

    return fill(boxes, transfer(label, 2 * len(terms), 0, dim, 0, route))


# ---------------------------------------------------------------------------
# A constant sent in on a wire; an output left unread (note sec.general, first paragraph).
# ---------------------------------------------------------------------------


def send(part: Arrangement, constants: Mapping[int, float], label: str | None = None) -> Arrangement:
    """``part`` with a constant sent in on each input wire named in ``constants`` (index -> value).

    The field is pulled back along the map that inserts the constants, so the load at those
    inputs is dropped. The remaining inputs keep their order.
    """
    d = part.in_dim_N
    if part.in_dim_M or part.out_dim_M:
        raise ValueError("send: expected a part 1 -> <in|out>")
    if any(i < 0 or i >= d for i in constants):
        raise ValueError(f"send: the part has inputs 0..{d - 1}")
    free = [i for i in range(d) if i not in constants]
    fixed = sorted(constants)
    values = jnp.array([float(constants[i]) for i in fixed])
    order = jnp.argsort(jnp.array(fixed + free)) if d else jnp.zeros(0, dtype=int)

    def route(m: Array, n: Array) -> Array:
        return jnp.concatenate([values, n])[order]

    name = label or f"{part.label}[{', '.join(f'{i}:={constants[i]:g}' for i in fixed)}]"
    return compose_seq(part, transfer(name, d, part.out_dim_N, len(free), part.out_dim_N, route, lambda m: m))


def discard(part: Arrangement, outputs: Sequence[int], label: str | None = None) -> Arrangement:
    """``part`` with the named outputs unread: nothing reads them and their load is ``0``."""
    if part.in_dim_M or part.out_dim_M:
        raise ValueError("discard: expected a part 1 -> <in|out>")
    kept = jnp.array([i for i in range(part.out_dim_N) if i not in set(outputs)], dtype=int)
    name = label or f"{part.label}, unread"
    return compose_seq(part, transfer(name, part.in_dim_N, part.out_dim_N, part.in_dim_N, len(kept),
                                      lambda m, n: n, lambda m: m[kept]))


# ---------------------------------------------------------------------------
# The general parts (note sec.general).
# ---------------------------------------------------------------------------


def spring(label: str = "spring") -> Arrangement:
    """The spring ``1 -> <2|0>`` with inputs a stiffness ``k`` and a stretch ``u`` and field
    ``-d(k u^2 / 2) = -k u du - (u^2 / 2) dk`` (note ``eqn.spring``): the pull, and the energy
    it takes to change the stiffness, which comes back through the wire that sets ``k``."""
    return pushes(label, 2, [("pull", lambda v: -v[0] * v[1], lambda v: v[1]),
                             ("stiffening", lambda v: -0.5 * v[1] ** 2, lambda v: v[0])])


def dragger(beta: float, memory: float = 0.0, label: str = "dragger") -> Arrangement:
    """The dragger ``1 -> <2|1>`` (note ``eqn.dragger``): a memory of capacity ``beta`` and a
    spring stretched between the position ``x`` it reads and the memory ``z``.

    Inputs ``(k, x)``, output ``z``. The circle computes the stretch ``x - z``; the field is
    ``-d(k (x - z)^2 / 2)`` and the memory moves at ``k (x - z) / beta``.
    """
    wiring = transfer(label, 2, 1, 2, 1, lambda m, n: jnp.stack([n[0], n[1] - m[0]]), lambda m: m)
    return fill([rod_segment(beta, memory, "memory"), spring()], wiring)


def damper(beta: float, kappa_return: float, stroke: float = 0.0, label: str = "damper") -> Arrangement:
    """The damper ``1 -> <2|0>`` (note ``eqn.damper``): a dragger whose memory, its stroke
    ``z``, a spring sent the constant stiffness ``kappa_return`` pulls back to ``0``.

    Inputs a stiffness ``k`` (``0`` out of contact) and a gap ``u``. The field is
    ``-d(k (u - z)^2 / 2 + kappa_return z^2 / 2)`` and the stroke moves at
    ``(k (u - z) - kappa_return z) / beta``.
    """
    back = send(spring("return spring"), {0: kappa_return})
    wiring = transfer(label, 3, 1, 2, 0, lambda m, n: jnp.stack([n[0], n[1], m[0]]))
    return fill([dragger(beta, stroke), back], wiring)


def motor(capacity: float, label: str = "motor") -> Arrangement:
    """The motor ``1 -> <4|0>`` (note ``eqn.motor``) for a store of the given capacity.

    Inputs: the battery ``b``, the store ``q``, the push ``tau``, and the load ``P`` the rest
    of the arrangement puts on the store. Field ``tau dq - tau (tau + P) / capacity db``: the
    store moves at ``(tau + P) / capacity``, so the battery pays the power ``tau`` times that
    rate. Load-free at ``tau`` and ``P``. It must be sent the true ``P``.
    """
    if capacity <= 0:
        raise ValueError("the driven store's capacity is positive")
    return pushes(label, 4, [("drive", lambda v: v[2], lambda v: v[1]),
                             ("pay", lambda v: -v[2] * (v[2] + v[3]) / capacity, lambda v: v[0])])


def motor_at_rate(capacity: float, label: str = "motor at a rate") -> Arrangement:
    """A motor that drives its store at a chosen rate ``u``: inputs ``(b, q, u, P)``, the motor
    sent the push ``capacity * u - P``. The battery pays ``capacity u^2 - P u`` (note
    ``sec.general``, the paragraph after ``eqn.motor``)."""
    wiring = transfer(label, 4, 0, 4, 0,
                      lambda m, n: jnp.stack([n[0], n[1], capacity * n[2] - n[3], n[3]]))
    return fill([motor(capacity)], wiring)


def sweep(capacity: float, radius: float, nu: float, gamma: float, label: str = "sweep") -> Arrangement:
    """The sweep ``1 -> <5|0>`` (note ``sec.general``): two motors with ``P = 0`` driving two
    stores round the circle of the given radius, at ``nu`` radians per unit of duration,
    returning to that circle at the rate ``gamma``.

    Inputs: the battery ``b``, the stores ``q1, q2``, a gate ``g`` and an extra rate ``e`` on
    the first store. Field ``C r1 dq1 + C r2 dq2 - C (r1^2 + r2^2) db`` with
    ``r1 = g (nu q2 + gamma (A^2 - q1^2 - q2^2) q1) + e`` and
    ``r2 = g (-nu q1 + gamma (A^2 - q1^2 - q2^2) q2)``.
    """
    def route(m: Array, n: Array) -> Array:
        b, q1, q2, g, e = n
        ring = radius ** 2 - q1 ** 2 - q2 ** 2
        r1 = g * (nu * q2 + gamma * ring * q1) + e
        r2 = g * (-nu * q1 + gamma * ring * q2)
        zero = jnp.zeros_like(b)
        return jnp.stack([b, q1, capacity * r1, zero, b, q2, capacity * r2, zero])

    return fill([motor(capacity, "motor 1"), motor(capacity, "motor 2")],
                transfer(label, 8, 0, 5, 0, route))
