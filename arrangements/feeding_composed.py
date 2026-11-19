"""Feeding's casing, composed by the library from the feeding note's parts.

The note is the book repo's ``background/feeding-imc/feeding-imc.tex``; ``source`` here is
its ``eqn.source``: three integrands (food, opening, the casing's mass) and three parts that
read the casing's height (gravity, the air's dragger, the ground), with circles for the
access the opening gives, how red the casing is, and its place. Each part is one of
``feeding_parts``; the composite is ``feeding_parts.fill``, that is ``compose_seq`` after
``tensor_arrangements``, so it records its factors and ``explore.inspect`` shows them.

It is the same morphism of IMC's Cartesian tangent subcategory as ``feeding.source``, which
types the summed field out by hand: same interface ``1 -> <0|6>``, same configuration
``(f, a, x, z, z_g)``, same readouts, same inclination field, same integrator scheme
(``tests/test_feeding_composed.py``). ``feeding.py`` does not use it yet.

Built 2026-11-19 for the book's P109 (e) (Claude Fable 5.2, 6a32afbb).
"""
from __future__ import annotations

from typing import Mapping

import jax.numpy as jnp
from jax import Array

from . import feeding_parts as fp
from .arrangement import Arrangement
from .feeding import CENTRES, DEFAULTS, _destination, _food_memory, _search_map, _variate, theta


def gravity(m: float, g: float) -> Arrangement:
    """``push`` with constant coefficient ``-m g`` and target the height: field ``-m g dx``."""
    return fp.pushes("gravity", 1, [("weight", lambda v: -m * g * jnp.ones_like(v[0]), lambda v: v[0])])


def ground(xg: float, kg: float, eps: float, kd: float, epsd: float, bd: float, kr: float,
           stroke: float = 0.0, *, stages: int = 1, h: float = 1.0) -> Arrangement:
    """The ground ``1 -> <1|0>`` (note ``sec.source``): a spring and a damper sent the gap
    ``v = x - xg``, the spring the stiffness ``kg theta(-v/eps)`` and the damper
    ``kd theta(-v/epsd)``. Its integrand is the damper's stroke; its field is minus the
    differential of ``kg theta(-v/eps) v^2/2 + kd theta(-v/epsd) (v - z_g)^2/2 + kr z_g^2/2``.
    """
    def route(m: Array, n: Array) -> Array:
        v = n[0] - xg
        return jnp.stack([kg * theta(-v / eps), v, kd * theta(-v / epsd), v])

    return fp.fill([fp.spring("ground spring"), fp.damper(bd, kr, stroke, "ground damper", stages=stages, h=h)],
                   fp.transfer("ground", 4, 0, 1, 0, route))


def source(place: float, food: float = 10., *, parameters: Mapping[str, float] | None = None,
           stages: int = 1, h: float = 1.) -> Arrangement:
    """The casing ``1 -> <0|6>`` of the note's ``eqn.source``, composed from its parts.

    Outputs: food, opening, access, height, red, place. Arguments as ``feeding.source``.
    """
    p = {**DEFAULTS, **(parameters or {})}
    step = dict(stages=stages, h=h)
    air = fp.discard(fp.send(fp.dragger(p['beta'], p['xg'], "air dragger", **step), {0: p['kappa']}), [0],
                     "air drag")
    parts = [fp.rod_segment(1., food, "food", **step),
             fp.rod_segment(1., 0., "opening", **step),
             fp.mass(p['m'], p['xg'], 0., "casing", **step),
             gravity(p['m'], p['g']),
             air,
             ground(p['xg'], p['kg'], p['eps'], p['kd'], p['epsd'], p['bd'], p['kr'], **step)]

    def publish(m: Array) -> Array:
        f, a, x = m
        return jnp.stack([f, a, theta((a - p['a1']) / (p['a2'] - p['a1'])), x,
                          theta((f - p['fr']) / p['fr']), jnp.asarray(place, dtype=float)])

    wiring = fp.transfer(f"casing({place:g})", 3, 3, 0, 6,
                         lambda m, n: jnp.stack([m[2], m[2], m[2]]), publish)
    return fp.fill(parts, wiring)


# ---------------------------------------------------------------------------
# The body and its motors (note sec.receiver), behind the seam of ``feeding._body``.
# ---------------------------------------------------------------------------


def _grip_rate(b: Array, grip: Array, keep: Array, p: Mapping[str, float]) -> Array:
    """The note's ``grip``: toward 2 while a casing worth keeping is held, toward -1 otherwise."""
    return p['lgrip'] * theta(b / p['b0']) * (keep * (2 - grip) - (1 - keep) * (grip + 1))


def gripper(p: Mapping[str, float]) -> Arrangement:
    """The gripper ``1 -> <4|0>`` (note ``eqn.gripper``): the motor of capacity ``cgrip`` sent
    the push ``cgrip * grip - P``. Inputs: battery, grip, the encounter's ``K``, and the
    attachment's pull on the grip, which is the motor's other load."""
    def route(m: Array, n: Array) -> Array:
        b, grip, keep, pull = n
        return jnp.stack([b, grip, p['cgrip'] * _grip_rate(b, grip, keep, p) - pull, pull])

    return fp.fill([fp.motor(p['cgrip'])], fp.transfer("gripper", 4, 0, 4, 0, route))


def lever(p: Mapping[str, float]) -> Arrangement:
    """The lever ``1 -> <4|0>``: the motor of capacity ``clift`` driving the tip toward the
    mouth while the grip holds and toward the ground otherwise. Inputs: battery, tip, grip,
    and the attachment's pull on the tip."""
    def route(m: Array, n: Array) -> Array:
        b, tip, grip, pull = n
        hold = theta(grip)
        lift = p['llift'] * theta(b / p['b0']) * (hold * (p['x0'] - tip) + (1 - hold) * (p['xg'] - tip))
        return jnp.stack([b, tip, p['clift'] * lift - pull, pull])

    return fp.fill([fp.motor(p['clift'])], fp.transfer("lever", 4, 0, 4, 0, route))


def wheels(p: Mapping[str, float], cart: float = 0., *, stages: int = 1, h: float = 1.) -> Arrangement:
    """The wheels ``1 -> <2|1>`` (note ``eqn.cart``): the cart's mass on a dragger whose memory
    is the wheel, driven by a motor sent the torque and the dragger's own pull on the wheel.
    Inputs: battery, torque. Output: the cart's place."""
    step = dict(stages=stages, h=h)
    parts = [fp.mass(p['M'], cart, 0., "cart", **step),
             fp.send(fp.dragger(p['betac'], cart, "wheel dragger", **step), {0: p['kappac']}, "wheel"),
             fp.motor(p['betac'], "drive")]

    def route(m: Array, n: Array) -> Array:
        y, wheel = m
        b, torque = n
        return jnp.stack([y, b, wheel, torque, p['kappac'] * (y - wheel)])

    return fp.fill(parts, fp.transfer("wheels", 5, 2, 2, 1, route, lambda m: m[:1]))


def _memory_start(p: Mapping[str, float], memory) -> list[tuple[str, Array]]:
    """The learning receiver's memory stores and their starting values, as ``feeding._body``."""
    zeros = jnp.zeros(len(CENTRES))
    saved = memory if isinstance(memory, dict) else {'usual': zeros if memory is None else memory}
    usual = jnp.asarray(saved.get('usual', zeros), dtype=float)
    weights = jnp.asarray(saved.get('returnWeights', zeros), dtype=float)
    if usual.shape != zeros.shape or weights.shape != zeros.shape:
        raise ValueError("expected one yield and one return parameter per center")
    fresh = jnp.ones_like(usual)
    _, masses = _search_map(usual, fresh)
    target = _destination(masses, _variate(p['searchSeed'], jnp.array(0)))
    return [("meal", jnp.zeros(1)), ("usual yield", usual), ("unexplored", fresh),
            ("destination", jnp.atleast_1d(target)), ("clock", jnp.array([1.7])),
            ("return rate", weights), ("eye contact", zeros), ("last empty", zeros), ("time away", zeros)]


def memory_motors(capacity: float, widths: list[int], names: list[str]) -> Arrangement:
    """One motor per memory store, each driving its store at a chosen rate with other load 0,
    which is the true one: nothing else acts on these stores. Inputs, per store: battery, the
    store, its rate."""
    motors = [fp.send(fp.motor_at_rate(capacity, f"{name} motor", n), {1 + 2 * n + j: 0. for j in range(n)},
                      f"{name} motor, P=0") for name, n in zip(names, widths)]
    size = sum(1 + 2 * n for n in widths)
    return fp.fill(motors, fp.transfer("memory motors", size, 0, size, 0, lambda m, n: n))


def body(*, parameters: Mapping[str, float] | None = None, stages: int = 1, h: float = 1.,
         learning: bool = False, memory=None, cart: float = 0.) -> Arrangement:
    """The body and its motors ``1 -> <7|5>``, composed from the note's parts.

    Inputs, from the encounter: seen, climb, K, the attachment's pulls on grip and tip, an
    unused slot, and the food eaten. Outputs: battery, grip, tip, cart, look. Arguments as
    ``feeding._body``, of which this is the same morphism: same configuration
    ``(b, grip, tip, cart, wheel, look, scan, memory...)``, readouts and inclination field.

    Stores: battery, engagement, tip, the wheels' cart and wheel, look, scan, and the memory
    (with ``learning``: meal, usual yield, unexplored, destination, clock, return rate, eye
    contact, last empty, time away, the nineteen-wide ones one box each; without: the two
    patrol stores). Stateless: gripper, lever, the wheels' dragger spring and drive motor, the
    eye's sweep, and the memory's motors or the patrol's sweep.
    """
    p = {**DEFAULTS, **(parameters or {})}
    step = dict(stages=stages, h=h)
    if learning:
        start, capacity = _memory_start(p, memory), p['memoryCost']
    else:
        start, capacity = [("patrol", jnp.zeros(1)), ("patrol scan", jnp.array([p['B']]))], p['cpat']
    names = [name for name, _ in start]
    widths = [int(v.shape[0]) for _, v in start]
    extra = sum(widths)

    stores = [fp.rod_segment(1., 6. if learning else 3., "battery", **step),
              fp.rod_segment(p['cgrip'], -1., "engagement", **step),
              fp.rod_segment(p['clift'], p['xg'], "tip", **step),
              wheels(p, cart, **step),
              fp.rod_segment(p['csee'], 0., "look", **step),
              fp.rod_segment(p['csee'], p['A'], "scan", **step),
              *[fp.rod_segment(capacity, v, name, **step) for name, v in start]]
    stateless = [gripper(p), lever(p), fp.sweep(p['csee'], p['A'], p['omega'], p['gam'], "eye"),
                 memory_motors(capacity, widths, names) if learning
                 else fp.sweep(p['cpat'], p['B'], p['nup'], p['gamp'], "patrol")]

    def route(m: Array, n: Array) -> Array:
        b, grip, tip, y, look, scan = m[:6]
        rest = m[6:]
        seen, climb, keep, pull_grip, pull_tip, _, eaten = n
        live, free = theta(b / p['b0']), theta(1 - 2 * seen)
        if learning:
            q = jnp.concatenate([jnp.stack([b, grip, tip, y, jnp.zeros_like(b), look, scan]), rest])
            rates, search = _food_memory(q, _grip_rate(b, grip, keep, p), live, seen, eaten, p)
            cuts = [0]
            for n_k in widths:
                cuts.append(cuts[-1] + n_k)
            tail = [x for i in range(len(widths))
                    for x in (b[None], rest[cuts[i]:cuts[i + 1]], rates[cuts[i]:cuts[i + 1]])]
        else:
            search = p['kexp'] * (rest[0] - y)
            tail = [jnp.stack([b, rest[0], rest[1], live * free, jnp.zeros_like(b)])]
        torque = live * (p['ktau'] * seen * look + free * search)
        return jnp.concatenate([jnp.stack([b, torque]),                               # wheels
                                jnp.stack([b, grip, keep, pull_grip]),                # gripper
                                jnp.stack([b, tip, grip, pull_tip]),                  # lever
                                jnp.stack([b, look, scan, live * free, p['lam'] * live * climb]),  # eye
                                *tail])

    inner_in = 2 + 4 + 4 + 5 + (sum(1 + 2 * n_k for n_k in widths) if learning else 5)
    wiring = fp.transfer("body and motors", inner_in, 6 + extra, 7, 5, route, lambda m: m[:5])
    return fp.fill(stores + stateless, wiring)
