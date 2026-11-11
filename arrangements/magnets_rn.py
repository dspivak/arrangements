"""One magnet, duplicated, on ``R^n``: the version of book plan P76 (h), kept beside ``magnets.py`` to compare against.

SINCE P76 (i) (session 7328b8c7, 2026-11-10) ``magnets.py`` is this construction with the fields themselves on the wires, built on
``smooth.py``; ``tests/test_magnets.py`` checks that the two step alike, and that this file's ``_transfer`` is the field-valued
arrangement with each body's two maps composed in, the equality the paragraph below says was argued by hand. What follows is this
file's own account, as written for P76 (h).

One magnet, duplicated: bodies that emit fields, and an arrangement that carries their positions (book plan P76 (h)).

David, 2026-11-10 (session 2524b69e): "i want to see a single magnet, and then duplicate it, and then see the phenomena without it
being hand-entered into the thing"; "using smooth sets, i think you can emit a field, right? and then just have the arrangment carry
their positions ?"; "what about a field that makes things not want to run into each other".

WHAT THE MATHEMATICS IS, AND WHAT THIS FILE CARRIES OF IT. In the book's arrangements with smooth interfaces (Ch 4, ``\\smoothimc``) a
body's output is the fields it makes, in its own frame: a point of ``F x F``, ``F`` the smooth set of maps from the plane to ``R``, one
factor for the magnetic field (the potential along wires that run straight out of the plane, so a magnet is a long coil seen end-on)
and one for the contact field. Its input is the fields around it, in its own frame, and its cost is a function of those.

This library is on ``R^n``, so no wire here carries a field. A wire out of a body carries the numbers that fix its fields, and a wire
into it carries the values of the fields around it at the points its cost reads. Each ``Body`` owns the two maps between the two
descriptions: ``field``, from its numbers to its fields (a function of a point in its own frame), and ``reads``, the points and
fields its cost evaluates. Call them ``iota_b : R^p -> F x F`` and ``ev_b : F x F -> R^k``. The code of ``_transfer`` names neither
``g`` nor any body's shape: it moves each body's fields from that body's frame to another's and adds them. But it is handed every
body's two maps, so the arrangement it builds holds them: ON ``R^n`` A BODY'S SHAPE IS IN THE ARRANGEMENT, NOT IN THE PART. The parts
``magnet(1, 0.25).box`` and ``magnet(1, 0.4).box`` are the same arrangement, and the arrangement must be built again for each choice
of what fills it. With field-valued interfaces the shape would be in the body's own output and one arrangement would serve every
body. So what the code composes is the field-valued arrangement with each body's lens ``(ev_b, iota_b)`` composed into the
arrangement that carries the positions. That the two composites are equal is associativity of composition in ``\\smoothimc``; it is argued in the book repo's
``notes/runs/magnets-2524b69e/RESULTS.md`` and is not independently checked.

Entered: the field of one long straight wire, ``g``; the contact field of one bumper, ``contact``; each part's own cost. Nothing about
a pair of bodies is written anywhere; what two bodies do to each other comes out of ``compose_seq``.

Three ways the positions are carried, all the same step (``tests/test_magnets.py``): ``plane``, whose integrator is one frictionless
mass per cart; ``field`` with ``cart`` boxes, each cart a part of its own and the arrangement without state; and a body made of bodies,
``magnet_of_wires``, two ``wire`` boxes placed by the static arrangement ``pair``. Several bodies may ride one cart (``cart_of``).

Claude Fable 5.2, session 2524b69e, 2026-11-10.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import jax.numpy as jnp

from .arrangement import Arrangement, exact, zero_form
from .light_tank import lift
from .wiring import compose_seq, tensor_arrangements, trivial_parameter

_z = jnp.zeros(0)
MAGNETIC, CONTACT = 0, 1


def g(d):
    """The magnetic field of one long straight wire carrying unit current, at the displacement ``d`` from it: ``-(1/2 pi) ln |d|``."""
    return -jnp.log(jnp.sum(d * d, axis=-1)) / (4.0 * jnp.pi)


def contact(d, sigma):
    """The contact field of one bumper of unit strength and size ``sigma``, at the displacement ``d`` from it: ``(sigma / |d|)^6``. A
    chosen shape, steep and short: it stands for two things not fitting in one place, and is not taken from a source."""
    return (sigma * sigma / jnp.sum(d * d, axis=-1)) ** 3


@dataclass(frozen=True)
class Body:
    """A part together with the two maps between its numbers and its fields. ``field(m, r)``: the fields it emits when its outputs
    are ``m``, at the points ``r`` (rows) of its own frame, one column per field (magnetic, contact); ``None`` if it emits nothing.
    ``reads``: for each input, ``(x, y, which field)``, the point of its own frame and the field its cost reads there."""
    box: Arrangement
    reads: Tuple[Tuple[float, float, int], ...]
    field: Optional[Callable]
    n_out: int = 1


def _part(out, n_in: int, minus_cost, label: str) -> Arrangement:
    """A stateless part: it puts out the numbers ``out`` and has the cost ``-minus_cost`` on what it receives."""
    out = jnp.asarray(out, float)
    return Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=0, in_dim_N=n_in, out_dim_N=int(out.shape[0]),
                       out_f=lambda q, m: out, in_f=lambda q, m, n: _z,
                       omega=exact(lambda q, m, n: minus_cost(n)), label=label)


def _cols(magnetic, touch):
    return jnp.stack([magnetic, touch], axis=-1)


def wire(current: float) -> Body:
    """A long straight wire: it emits the magnetic field ``current * g`` and has the cost ``-current f(0) / 2`` on the magnetic field
    ``f`` it receives."""
    return Body(_part([current], 1, lambda n: 0.5 * current * n[0], f"wire({current:g})"), ((0.0, 0.0, MAGNETIC),),
                lambda m, r: _cols(m[0] * g(r), 0.0 * r[:, 0]))


def magnet(current: float, h: float) -> Body:
    """A magnet as one part: it emits the field of two wires with opposite currents at ``+-h`` on its axis, and its cost is minus half
    its current times the flux of the received field through it, ``f(h) - f(-h)``."""
    e = jnp.array([h, 0.0])
    return Body(_part([current], 2, lambda n: 0.5 * current * (n[0] - n[1]), f"magnet({current:g},h={h:g})"),
                ((h, 0.0, MAGNETIC), (-h, 0.0, MAGNETIC)), lambda m, r: _cols(m[0] * (g(r - e) - g(r + e)), 0.0 * r[:, 0]))


def dead_loop(ell: float, h: float) -> Body:
    """A closed loop with no resistance and no flux through it: it emits nothing, and its cost is ``(f(h) - f(-h))^2 / 2 ell``, the
    energy of the current ``-(f(h) - f(-h)) / ell`` that keeps its flux at 0."""
    return Body(_part([0.0], 2, lambda n: -(n[0] - n[1]) ** 2 / (2.0 * ell), f"deadLoop(l={ell:g},h={h:g})"),
                ((h, 0.0, MAGNETIC), (-h, 0.0, MAGNETIC)), None)


def bumper(strength: float, at: float = 0.0, sigma: float = 0.25) -> Body:
    """Something that will not share its place: at the point ``at`` of its cart's axis it emits the contact field
    ``strength * contact`` and has the cost ``+strength f(at) / 2`` on the contact field ``f`` it receives. The sign is the opposite
    of a wire's, so two bumpers push each other apart."""
    p = jnp.array([at, 0.0])
    return Body(_part([strength], 1, lambda n: -0.5 * strength * n[0], f"bumper({strength:g},at={at:g})"), ((at, 0.0, CONTACT),),
                lambda m, r: _cols(0.0 * r[:, 0], m[0] * contact(r - p, sigma)))


def pair(h: float) -> Arrangement:
    """The static arrangement that places two one-point magnetic bodies at ``+h`` and ``-h`` on an axis and is itself a body: it passes
    their two currents out, and gives each the field arriving at its end plus the other's field there, ``current * g`` at distance
    ``2 h``."""
    far = g(jnp.array([2.0 * h, 0.0]))
    return Arrangement(P=trivial_parameter(), in_dim_M=2, out_dim_M=2, in_dim_N=2, out_dim_N=2,
                       out_f=lambda q, m: m, in_f=lambda q, m, n: jnp.array([n[0] + m[1] * far, n[1] + m[0] * far]),
                       omega=zero_form, label=f"pair(h={h:g})")


def magnet_of_wires(current: float, h: float) -> Body:
    """A magnet composed of smaller things: ``pair(h) . (wire(current) || wire(-current))``. It has two outputs, the two wires'
    currents, and emits the sum of their fields from ``+-h``."""
    e = jnp.array([h, 0.0])
    box = compose_seq(tensor_arrangements([wire(current).box, wire(-current).box]), pair(h))
    return Body(box, ((h, 0.0, MAGNETIC), (-h, 0.0, MAGNETIC)),
                lambda m, r: _cols(m[0] * g(r - e) + m[1] * g(r + e), 0.0 * r[:, 0]), n_out=2)


def _transfer(bodies: Sequence[Body], cart_of: Optional[Sequence[int]] = None):
    """``(number of inputs, number of outputs, received)``. ``received(X, m)`` lists, for each body and each point its cost reads, the
    sum of the other bodies' fields there: each other body's field, a function on that body's frame, is moved to the frame of the
    reading body by the two carts' configurations ``X`` (one row ``(x, y, angle)`` per cart). It uses each body's ``field`` and
    ``reads`` and knows nothing else of them."""
    n = len(bodies)
    cart_of = list(range(n)) if cart_of is None else list(cart_of)
    starts = [0]
    for b in bodies:
        starts.append(starts[-1] + b.n_out)

    def frame(X, c):
        cs, sn = jnp.cos(X[c, 2]), jnp.sin(X[c, 2])
        return X[c, :2], jnp.array([[cs, -sn], [sn, cs]])

    def received(X, m):
        out = []
        for i, b in enumerate(bodies):
            if not b.reads:
                continue
            pts = jnp.array([[x, y] for x, y, _ in b.reads])
            which = jnp.array([k for _, _, k in b.reads], dtype=int)
            O_i, R_i = frame(X, cart_of[i])
            world = O_i + pts @ R_i.T
            total = jnp.zeros(len(b.reads))
            for j, other in enumerate(bodies):
                if j == i or other.field is None:
                    continue
                O_j, R_j = frame(X, cart_of[j])
                theirs = other.field(m[starts[j]:starts[j + 1]], (world - O_j) @ R_j)      # the same points, in j's frame
                total = total + theirs[jnp.arange(len(b.reads)), which]
            out.append(total)
        return jnp.concatenate(out) if out else _z

    return sum(len(b.reads) for b in bodies), starts[-1], received, 1 + max(cart_of)


def _mass_rule(mass, drag, k):
    """The tangent response of ``k`` frictionless masses with drag: positions move at ``p / mass``, momenta at the load minus
    ``drag * p``."""
    return lambda s, xi: jnp.concatenate([s[k:] / mass, xi - drag * s[k:]])


def plane(bodies: Sequence[Body], q0, p0, mass, rule: str = "rk4", tau: float = 0.01, K: int = 1,
          cart_of: Optional[Sequence[int]] = None, drag: float = 0.0) -> Arrangement:
    """The arrangement that carries the positions in its own integrator: one frictionless mass per cart, with configuration
    ``(x, y, angle)``; ``mass`` lists each cart's ``(m, m, inertia)``, and ``drag`` is the rate at which a cart loses momentum to what
    it rolls on. Its in-transfer is ``_transfer``'s. It has no inclination field of its own."""
    n_in, n_out, received, c = _transfer(bodies, cart_of)
    mass = jnp.asarray(mass, float).reshape(3 * c)
    s0 = jnp.concatenate([jnp.asarray(q0, float).reshape(3 * c), jnp.asarray(p0, float).reshape(3 * c)])
    P = lift(rule, lambda s: s[:3 * c], _mass_rule(mass, drag, 3 * c), s0, 3 * c, tau, K)
    return Arrangement(P=P, in_dim_M=n_in, out_dim_M=n_out, in_dim_N=0, out_dim_N=0,
                       out_f=lambda q, m: _z, in_f=lambda q, m, n_: received(q.reshape(c, 3), m), omega=zero_form,
                       label=f"plane({c})")


def system(bodies: Sequence[Body], q0, p0, mass, rule: str = "rk4", tau: float = 0.01, K: int = 1,
           cart_of: Optional[Sequence[int]] = None, drag: float = 0.0) -> Arrangement:
    """The closed composite ``plane . (body_1 || ... || body_n)``."""
    return compose_seq(tensor_arrangements([b.box for b in bodies]), plane(bodies, q0, p0, mass, rule, tau, K, cart_of, drag))


def cart(mass, q0, p0, rule: str = "rk4", tau: float = 0.01, K: int = 1, drag: float = 0.0) -> Arrangement:
    """One cart as a part of its own: a frictionless mass with configuration ``(x, y, angle)`` and ``mass = (m, m, inertia)``. It
    reads out its configuration on one output; what comes back there is the force and the torque on it."""
    mass = jnp.asarray(mass, float)
    s0 = jnp.concatenate([jnp.asarray(q0, float), jnp.asarray(p0, float)])
    P = lift(rule, lambda s: s[:3], _mass_rule(mass, drag, 3), s0, 3, tau, K)
    return Arrangement(P=P, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=3,
                       out_f=lambda q, m: q, in_f=lambda q, m, n: _z, omega=zero_form, label="cart")


def field(bodies: Sequence[Body], cart_of: Optional[Sequence[int]] = None) -> Arrangement:
    """The arrangement with no state: it hears every body's numbers and then every cart's configuration, and gives each body the sum
    of the other bodies' fields. ``plane`` with its integrator drawn as boxes."""
    n_in, n_out, received, c = _transfer(bodies, cart_of)
    return Arrangement(P=trivial_parameter(), in_dim_M=n_in, out_dim_M=n_out + 3 * c, in_dim_N=0, out_dim_N=0,
                       out_f=lambda q, m: _z, in_f=lambda q, m, n_: received(m[n_out:].reshape(c, 3), m[:n_out]),
                       omega=zero_form, label=f"field({c})")


def system_of_carts(bodies: Sequence[Body], q0, p0, mass, rule: str = "rk4", tau: float = 0.01, K: int = 1,
                    cart_of: Optional[Sequence[int]] = None, drag: float = 0.0) -> Arrangement:
    """The closed composite ``field . (body_1 || ... || body_n || cart_1 || ... || cart_c)``: every cart independent."""
    q0, p0, mass = jnp.asarray(q0, float), jnp.asarray(p0, float), jnp.asarray(mass, float)
    carts = [cart(mass[i], q0[i], p0[i], rule, tau, K, drag) for i in range(q0.shape[0])]
    return compose_seq(tensor_arrangements([b.box for b in bodies] + carts), field(bodies, cart_of))
