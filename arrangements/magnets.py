"""One magnet, duplicated: bodies that emit fields, and one arrangement that carries their positions (book plan P76 (h), (i)).

David, 2026-11-10 (session 2524b69e): "i want to see a single magnet, and then duplicate it, and then see the phenomena without it
being hand-entered into the thing"; "using smooth sets, i think you can emit a field, right? and then just have the arrangment carry
their positions ?"; "what about a field that makes things not want to run into each other".

THE WIRES CARRY FIELDS. This file is built on ``smooth.py``, the core whose interface sides may be mapping spaces (the book's
arrangements with smooth interfaces, Ch 4). Write ``F`` for ``Maps(2)``, the smooth set of smooth maps from the plane to ``R``: a
field, held as a function of a point. EVERY BODY HAS ONE INTERFACE, ``<(F, F) | (F, F)>``: it puts out the magnetic field and the
contact field it makes, in its own frame, and receives the two fields around it, in its own frame; its cost is a function of what it
receives. The magnetic field is the potential along wires that run straight out of the plane, so a magnet is a long coil seen end-on
and everything is per unit of length.

ONE ARRANGEMENT SERVES ANY BODIES. ``plane(n, ...)`` is told how many bodies there are, which cart each rides, and the carts' masses
and start. It is not told what the bodies are: its in-transfer gives each body the sum of the other bodies' fields, each moved from
its body's frame to the receiver's by the two carts' configurations. A body's shape is in the body: ``magnet(1, 0.25)`` and
``magnet(1, 0.4)`` are different systems on the same interface, and a body made of bodies, ``magnet_of_wires``, has a body's
interface. (The version of P76 (h), on ``R^n``, where a wire carried the numbers that fix a field and the arrangement held every
body's shape, is kept as ``magnets_rn.py``; ``tests/test_magnets.py`` checks the two step for step.)

Entered: the field of one long straight wire, ``g``; the contact field of one bumper, ``contact``; each part's own cost. Nothing about
a pair of bodies is written anywhere; what two bodies do to each other comes out of ``smooth.compose_seq``.

ASSUMED. ``g`` and ``contact`` are singular at the body's own point, so they are smooth maps on the plane only away from it: a point
of ``F`` should be the field made smooth inside a small disc around the body, or the bodies kept apart. The code evaluates a body's
field only where another body reads, and no body reads its own field.

Three ways the positions are carried, all the same step (``tests/test_magnets.py``): ``plane``, whose integrator is one frictionless
mass per cart; ``field`` with ``cart`` boxes, each cart a part of its own and the arrangement without state; and a body made of bodies,
``placed``, a static arrangement that holds its bodies at fixed places and is itself a body. Several bodies may ride one cart
(``cart_of``).

Claude Fable 5.2, sessions 2524b69e and 7328b8c7, 2026-11-10.
"""

from __future__ import annotations

from typing import Optional, Sequence

import jax.numpy as jnp

from .light_tank import lift
from .smooth import Maps, R, SmoothArrangement, compose_seq, exact, tensor, zero_form
from .wiring import trivial_parameter

F = Maps(2)                 # a field: a smooth map from the plane to R
FIELDS = (F, F)             # what a body puts out, and what it receives: (magnetic, contact)
MAGNETIC, CONTACT = 0, 1
_origin = jnp.zeros(2)


def g(d):
    """The magnetic field of one long straight wire carrying unit current, at the displacement ``d`` from it: ``-(1/2 pi) ln |d|``."""
    return -jnp.log(jnp.sum(d * d, axis=-1)) / (4.0 * jnp.pi)


def contact(d, sigma):
    """The contact field of one bumper of unit strength and size ``sigma``, at the displacement ``d`` from it: ``(sigma / |d|)^6``. A
    chosen shape, steep and short: it stands for two things not fitting in one place, and is not taken from a source."""
    return (sigma * sigma / jnp.sum(d * d, axis=-1)) ** 3


def nothing(r):
    """The zero field."""
    return jnp.zeros(())


def body(magnetic, touch, minus_cost, label: str) -> SmoothArrangement:
    """A stateless body ``1 -> <FIELDS|FIELDS>``: it puts out the fields ``magnetic`` and ``touch``, functions of a point of its own
    frame, and has the cost ``-minus_cost(received)`` on the pair of fields it receives."""
    return SmoothArrangement(P=trivial_parameter(), M_in=(), M_out=(), N_in=FIELDS, N_out=FIELDS,
                             out_f=lambda q, m: (magnetic, touch), in_f=lambda q, m, n: (),
                             omega=exact(lambda q, m, n: minus_cost(n)), label=label)


def wire(current: float) -> SmoothArrangement:
    """A long straight wire: it emits the magnetic field ``current * g`` and has the cost ``-current f(0) / 2`` on the magnetic field
    ``f`` it receives."""
    return body(lambda r: current * g(r), nothing, lambda n: 0.5 * current * n[MAGNETIC](_origin), f"wire({current:g})")


def magnet(current: float, h: float) -> SmoothArrangement:
    """A magnet as one part: it emits the field of two wires with opposite currents at ``+-h`` on its axis, and its cost is minus half
    its current times the flux of the received field through it, ``f(h) - f(-h)``."""
    e = jnp.array([h, 0.0])
    return body(lambda r: current * (g(r - e) - g(r + e)), nothing,
                lambda n: 0.5 * current * (n[MAGNETIC](e) - n[MAGNETIC](-e)), f"magnet({current:g},h={h:g})")


def dead_loop(ell: float, h: float) -> SmoothArrangement:
    """A closed loop with no resistance and no flux through it: it emits nothing, and its cost is ``(f(h) - f(-h))^2 / 2 ell``, the
    energy of the current ``-(f(h) - f(-h)) / ell`` that keeps its flux at 0."""
    e = jnp.array([h, 0.0])
    return body(nothing, nothing, lambda n: -(n[MAGNETIC](e) - n[MAGNETIC](-e)) ** 2 / (2.0 * ell),
                f"deadLoop(l={ell:g},h={h:g})")


def bumper(strength: float, at: float = 0.0, sigma: float = 0.25) -> SmoothArrangement:
    """Something that will not share its place: at the point ``at`` of its cart's axis it emits the contact field
    ``strength * contact`` and has the cost ``+strength f(at) / 2`` on the contact field ``f`` it receives. The sign is the opposite
    of a wire's, so two bumpers push each other apart."""
    p = jnp.array([at, 0.0])
    return body(nothing, lambda r: strength * contact(r - p, sigma), lambda n: -0.5 * strength * n[CONTACT](p),
                f"bumper({strength:g},at={at:g})")


# ---------------------------------------------------------------------------
# Moving a field between frames, and adding fields: all an arrangement of bodies does.
# ---------------------------------------------------------------------------


def _frame(x):
    """A frame ``(x, y, angle)`` as its origin and its rotation."""
    cs, sn = jnp.cos(x[2]), jnp.sin(x[2])
    return x[:2], jnp.array([[cs, -sn], [sn, cs]])


def _between(frames, i, j):
    """The map taking a point of frame ``i`` to the same point of the plane in frame ``j``."""
    (O_i, R_i), (O_j, R_j) = frames[i], frames[j]
    return lambda r: R_j.T @ (O_i + R_i @ r - O_j)


def _total(fields):
    def f(r):
        out = jnp.zeros(())
        for field, move in fields:
            out = out + field(move(r))
        return out
    return f


def _received(frames, emitted, outer=None):
    """For bodies in the frames ``frames`` (a body's index into ``emitted`` is its place in that list) that emit ``emitted``, a flat
    tuple ``(magnetic_1, contact_1, magnetic_2, ...)``: what each body receives, in its own frame, the sum of the other bodies'
    fields moved into it. ``outer``, if given, is a pair of fields in the frame the bodies' frames are written in, added to each."""
    n, out = len(frames), []
    for i in range(n):
        for k in (MAGNETIC, CONTACT):
            terms = [(emitted[2 * j + k], _between(frames, i, j)) for j in range(n) if j != i]
            if outer is not None:
                (O_i, R_i) = frames[i]
                terms.append((outer[k], lambda r, O=O_i, Rot=R_i: O + Rot @ r))
            out.append(_total(terms))
    return tuple(out)


def placed(places: Sequence[Sequence[float]], label: str = "") -> SmoothArrangement:
    """A static arrangement that holds bodies at the fixed places ``(x, y, angle)`` of a frame and is itself a body: it puts out the
    sum of their fields, each moved into that frame, and gives each the fields arriving from outside plus the other bodies' fields,
    moved into its own frame. No state, no inclination field."""
    frames = [_frame(jnp.asarray(x, float)) for x in places]
    n = len(frames)

    def out_f(q, m):
        parent = (_origin, jnp.eye(2))
        return tuple(_total([(m[2 * j + k], lambda r, j=j: _between([parent, frames[j]], 0, 1)(r)) for j in range(n)])
                     for k in (MAGNETIC, CONTACT))

    return SmoothArrangement(P=trivial_parameter(), M_in=FIELDS * n, M_out=FIELDS * n, N_in=FIELDS, N_out=FIELDS,
                             out_f=out_f, in_f=lambda q, m, n_in: _received(frames, m, outer=n_in), omega=zero_form,
                             label=label or f"placed({n})")


def pair(h: float) -> SmoothArrangement:
    """``placed`` for two bodies at ``+h`` and ``-h`` on the axis."""
    return placed([[h, 0.0, 0.0], [-h, 0.0, 0.0]], label=f"pair(h={h:g})")


def magnet_of_wires(current: float, h: float) -> SmoothArrangement:
    """A magnet composed of smaller things: ``pair(h) . (wire(current) || wire(-current))``. It has a body's interface, puts out the
    field ``magnet(current, h)`` does, and has the same inclination field: its cost differs from the magnet's by a constant, the two
    wires' energy in each other's field."""
    return compose_seq(tensor([wire(current), wire(-current)]), pair(h))


# ---------------------------------------------------------------------------
# The arrangements that carry the positions.
# ---------------------------------------------------------------------------


def _carts(cart_of, n):
    cart_of = list(range(n)) if cart_of is None else list(cart_of)
    if len(cart_of) != n:
        raise ValueError(f"cart_of names {len(cart_of)} bodies, not {n}")
    return cart_of, 1 + max(cart_of)


def _mass_rule(mass, drag, k):
    """The tangent response of ``k`` frictionless masses with drag: positions move at ``p / mass``, momenta at the load minus
    ``drag * p``."""
    return lambda s, xi: jnp.concatenate([s[k:] / mass, xi - drag * s[k:]])


def plane(n: int, q0, p0, mass, rule: str = "rk4", tau: float = 0.01, K: int = 1,
          cart_of: Optional[Sequence[int]] = None, drag: float = 0.0) -> SmoothArrangement:
    """The arrangement that carries the positions of ``n`` bodies, whatever they are, in its own integrator: one frictionless mass
    per cart, with configuration ``(x, y, angle)``; ``mass`` lists each cart's ``(m, m, inertia)``, and ``drag`` is the rate at which
    a cart loses momentum to what it rolls on. Body ``i`` rides cart ``cart_of[i]`` (its own, by default). Its in-transfer gives each
    body the sum of the other bodies' fields, moved between their carts' frames. It has no inclination field of its own."""
    cart_of, c = _carts(cart_of, n)
    mass = jnp.asarray(mass, float).reshape(3 * c)
    s0 = jnp.concatenate([jnp.asarray(q0, float).reshape(3 * c), jnp.asarray(p0, float).reshape(3 * c)])
    P = lift(rule, lambda s: s[:3 * c], _mass_rule(mass, drag, 3 * c), s0, 3 * c, tau, K)

    def in_f(q, m, n_in):
        X = q.reshape(c, 3)
        return _received([_frame(X[cart_of[i]]) for i in range(n)], m)

    return SmoothArrangement(P=P, M_in=FIELDS * n, M_out=FIELDS * n, N_in=(), N_out=(),
                             out_f=lambda q, m: (), in_f=in_f, omega=zero_form, label=f"plane({c})")


def system(bodies: Sequence[SmoothArrangement], q0, p0, mass, rule: str = "rk4", tau: float = 0.01, K: int = 1,
           cart_of: Optional[Sequence[int]] = None, drag: float = 0.0) -> SmoothArrangement:
    """The closed composite ``plane . (body_1 || ... || body_n)``."""
    return compose_seq(tensor(list(bodies)), plane(len(bodies), q0, p0, mass, rule, tau, K, cart_of, drag))


def cart(mass, q0, p0, rule: str = "rk4", tau: float = 0.01, K: int = 1, drag: float = 0.0) -> SmoothArrangement:
    """One cart as a part of its own: a frictionless mass with configuration ``(x, y, angle)`` and ``mass = (m, m, inertia)``. It
    reads out its configuration on one output; what comes back there is the force and the torque on it."""
    mass = jnp.asarray(mass, float)
    s0 = jnp.concatenate([jnp.asarray(q0, float), jnp.asarray(p0, float)])
    P = lift(rule, lambda s: s[:3], _mass_rule(mass, drag, 3), s0, 3, tau, K)
    return SmoothArrangement(P=P, M_in=(), M_out=(), N_in=(), N_out=(R(3),),
                             out_f=lambda q, m: (q,), in_f=lambda q, m, n: (), omega=zero_form, label="cart")


def field(n: int, cart_of: Optional[Sequence[int]] = None) -> SmoothArrangement:
    """The arrangement with no state: it hears ``n`` bodies' fields and then every cart's configuration, and gives each body the sum
    of the other bodies' fields. ``plane`` with its integrator drawn as boxes."""
    cart_of, c = _carts(cart_of, n)

    def in_f(q, m, n_in):
        return _received([_frame(m[2 * n + cart_of[i]]) for i in range(n)], m[:2 * n])

    return SmoothArrangement(P=trivial_parameter(), M_in=FIELDS * n, M_out=FIELDS * n + (R(3),) * c, N_in=(), N_out=(),
                             out_f=lambda q, m: (), in_f=in_f, omega=zero_form, label=f"field({c})")


def system_of_carts(bodies: Sequence[SmoothArrangement], q0, p0, mass, rule: str = "rk4", tau: float = 0.01, K: int = 1,
                    cart_of: Optional[Sequence[int]] = None, drag: float = 0.0) -> SmoothArrangement:
    """The closed composite ``field . (body_1 || ... || body_n || cart_1 || ... || cart_c)``: every cart independent."""
    q0, p0, mass = jnp.asarray(q0, float), jnp.asarray(p0, float), jnp.asarray(mass, float)
    carts = [cart(mass[i], q0[i], p0[i], rule, tau, K, drag) for i in range(q0.shape[0])]
    return compose_seq(tensor(list(bodies) + carts), field(len(bodies), cart_of))
