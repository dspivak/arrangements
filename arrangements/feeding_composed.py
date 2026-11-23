"""Feeding's constructors, composed by the library from the feeding note's parts.

The note is the book repo's ``background/feeding-imc/feeding-imc.tex``; its composite is
``feed_n = W_n . (R (x) S_1 (x) ... (x) S_n)`` (``eqn.feed``), with the separator mouth of the
recursive note (``recursive-feeding.tex``, book P109 (i)):

* ``source``, the casing ``S_k`` (``eqn.source``): two integrands (food, the casing's mass)
  and three parts that read the casing's height (gravity, the air's dragger, the ground),
  with circles for how red the casing is and its place. It has no hinge: the mouth tears the
  food out and pays per gram;
* ``seeker_with_slot``, the receiver ``R``: ``body`` (its stores and motors) beside the mouth
  (the chamber, ``feeding_recursive.settle``), the husk collector and an open slot, filled by
  any feeder (``absorber_filler``, the gut wall, by default; ``hopper_filler``; ``robot_style``);
* ``mouth_coupling``, ``W_n``: per casing a contact and a latch, the casing side of taking,
  and one static map; the pile on the ground is the outermost discard.

``feeding_with_slot`` composes them; ``feeding.feeding`` is it. Each part is one of
``feeding_parts``; a composite is ``feeding_parts.fill``, that is ``compose_seq`` after
``tensor_arrangements``, so it records its factors and ``explore.inspect`` shows them. Every
inclination field here is a ``push``. The body is the same morphism as ``feeding._body_by_hand``,
its field typed out by hand (``tests/test_feeding_composed.py``).

Built 2026-11-19 for the book's P109 (e) (Claude Fable 5.2, 6a32afbb), the casing and the
body; 2026-11-21 for P109 (h) (Claude Fable 5.2, 86739fa7), the organs and the coupling; the
hinged casing, the opener and digestor and their coupling left on 2026-11-23 (P109 (i) act (5),
Claude Opus 5.5, subagent of c5f9ecc9).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import jax
import jax.numpy as jnp
from jax import Array

from . import feeding_parts as fp
from .arrangement import Arrangement
from .feeding import (CENTRES, DEFAULTS, _destination, _memory_rules, _memory_values, _search_map, _variate,
                      near, theta)
from .wiring import compose_seq, parallel_arrangements, tensor_arrangements


def gravity(m: float, g: float) -> Arrangement:
    """``push`` with constant coefficient ``-m g`` and target the height: field ``-m g dx``."""
    return fp.pushes("gravity", 1, [("weight", lambda v: -m * g * jnp.ones_like(v[0]), lambda v: v[0])],
                     inputs=[("x", "height", 1)], forms=[("-m*g", "x")], constants={"m": m, "g": g})


def ground(xg: float, kg: float, eps: float, kd: float, epsd: float, bd: float, kr: float,
           stroke: float = 0.0, *, stages: int = 1, h: float = 1.0) -> Arrangement:
    """The ground ``1 -> <1|0>`` (note ``sec.source``): the ``contact`` of stiffnesses ``kg``
    and ``kd`` sent the gap ``v = x - xg`` as its stretch and the gates ``theta(-v/eps)`` for
    its spring and ``theta(-v/epsd)`` for its damper, its stroke unread. Its integrand is
    the damper's stroke; its field is minus the differential of
    ``kg theta(-v/eps) v^2/2 + kd theta(-v/epsd) (v - z_g)^2/2 + kr z_g^2/2``.
    """
    def route(m: Array, n: Array) -> Array:
        v = n[0] - xg
        return jnp.stack([theta(-v / eps), theta(-v / epsd), v])

    return fp.fill([fp.contact(kg, kd, bd, kr, stroke, "ground", stages=stages, h=h)],
                   fp.transfer("ground", 3, 1, 1, 0, route))


def source(place: float, food: float = 10., *, parameters: Mapping[str, float] | None = None,
           stages: int = 1, h: float = 1., hinge: bool = False) -> Arrangement:
    """The casing ``1 -> <0|4>`` a separator mouth eats from (book P109 (i)), composed from
    its parts: outputs food, height, red, place; configuration ``(f, x, z_air, z_g)``. It has
    no opening: prying is paid per gram, by the mouth. (The hinged casing, with an opening
    store and an access readout, left the library on 2026-11-23; ``hinge=True`` says so.)
    """
    if hinge:
        raise ValueError("the hinged casing left the library on 2026-11-23 (book P109 (i), act (5))")
    p = {**DEFAULTS, **(parameters or {})}
    step = dict(stages=stages, h=h)
    air = fp.discard(fp.send(fp.dragger(p['beta'], p['xg'], "air dragger", **step), {0: p['kappa']}), [0],
                     "air drag")
    body = [fp.mass(p['m'], p['xg'], 0., "casing", **step),
            gravity(p['m'], p['g']),
            air,
            ground(p['xg'], p['kg'], p['eps'], p['kd'], p['epsd'], p['bd'], p['kr'], **step)]
    def publish(m: Array) -> Array:
        f, x = m
        return jnp.stack([f, x, theta((f - p['fr']) / p['fr']), jnp.asarray(place, dtype=float)])

    wiring = fp.transfer(f"casing({fp.number(place)})", 3, 2, 0, 4,
                         lambda m, n: jnp.stack([m[1], m[1], m[1]]), publish)
    return fp.with_meta(fp.fill([fp.rod_segment(1., food, "food", **step), *body], wiring),
                        outputs=[("f", "food", 1), ("x", "height", 1), ("r", "red", 1), ("y", "place", 1)])


# ---------------------------------------------------------------------------
# The body and its motors (note sec.receiver), behind the seam of ``feeding._body``.
# ---------------------------------------------------------------------------


def _grip_rate(grip: Array, keep: Array, p: Mapping[str, float]) -> Array:
    """The rate the gripper tells its motor: toward 2 while a casing worth keeping is held,
    toward -1 otherwise. The motor, reading the battery, drives the grip at this times
    ``live(b)``, the note's ``grip``."""
    return p['lgrip'] * (keep * (2 - grip) - (1 - keep) * (grip + 1))


def gripper(p: Mapping[str, float]) -> Arrangement:
    """The gripper ``1 -> <4|0>`` (note ``eqn.gripper``): the motor of capacity ``cgrip`` told
    the rate ``_grip_rate`` and the true other load ``P``. Inputs: battery, grip, the
    encounter's ``K``, and the attachment's pull on the grip, which is the motor's other load."""
    def route(m: Array, n: Array) -> Array:
        b, grip, keep, pull = n
        return jnp.stack([b, grip, _grip_rate(grip, keep, p), pull])

    return fp.fill([fp.motor_at_rate(p['cgrip'], p['b0'], "grip motor")], fp.transfer("gripper", 4, 0, 4, 0, route))


def lever(p: Mapping[str, float]) -> Arrangement:
    """The lever ``1 -> <4|0>``: the motor of capacity ``clift`` told the rate that takes the
    tip toward the mouth while the grip holds and toward the ground otherwise. Inputs:
    battery, tip, grip, and the attachment's pull on the tip."""
    def route(m: Array, n: Array) -> Array:
        b, tip, grip, pull = n
        hold = theta(grip)
        lift = p['llift'] * (hold * (p['x0'] - tip) + (1 - hold) * (p['xg'] - tip))
        return jnp.stack([b, tip, lift, pull])

    return fp.fill([fp.motor_at_rate(p['clift'], p['b0'], "lift motor")], fp.transfer("lever", 4, 0, 4, 0, route))


def wheels(p: Mapping[str, float], cart: float = 0., *, stages: int = 1, h: float = 1.) -> Arrangement:
    """The wheels ``1 -> <2|1>`` (note ``eqn.cart``): the cart's mass on a dragger whose memory
    is the wheel, driven by a motor sent the torque and the dragger's own pull on the wheel.
    Inputs: battery, torque. Output: the cart's place."""
    step = dict(stages=stages, h=h)
    parts = [fp.mass(p['M'], cart, 0., "cart", **step),
             fp.send(fp.dragger(p['betac'], cart, "wheel dragger", **step), {0: p['kappac']}, "wheel"),
             fp.motor(p['betac'], p['bDrive'], "drive")]

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


def learner(name: str, rule, reads: list[int], width: int, capacity: float, wake: float,
            read_names: list[str] | None = None) -> Arrangement:
    """One store's learner ``1 -> <1 + width + sum(reads)|0>``: its circle, the rule, and the
    motor it tells the rate, with other load 0 (nothing else acts on the store). Inputs: the
    battery, the store, then what the rule reads, each at the width in ``reads`` (named
    ``read_names`` in the part's ``meta``).

    ``rule`` takes the store and what it reads as one list, in that order, and returns the
    rate. The motor reads the battery and drives the store at ``live(b)`` times the rate."""
    motor = fp.send(fp.motor_at_rate(capacity, wake, f"{name} motor", width),
                    {1 + 2 * width + j: 0. for j in range(width)}, f"{name} motor, P=0")
    cuts = [1 + width]
    for k in reads:
        cuts.append(cuts[-1] + k)

    def route(m: Array, n: Array) -> Array:
        b, store = n[:1], n[1:1 + width]
        read = [n[cuts[i]:cuts[i + 1]] for i in range(len(reads))]
        return jnp.concatenate([b, store, rule(store, *read)])

    read_names = read_names or [f"input {i + 1}" for i in range(len(reads))]
    return fp.with_meta(fp.fill([motor], fp.transfer(f"{name} rule", 1 + 2 * width, 0, cuts[-1], 0, route)),
                        part="learner", kind="push of some inputs",
                        inputs=[("b", "battery", 1), ("s", name, width),
                                *[(f"r_{i + 1}", read, k) for i, (read, k) in enumerate(zip(read_names, reads))]],
                        constants={"C": capacity, "b_0": wake},
                        formula="a motor told the rule's rate u: (C*g*u)*ds - Σ(C*g*u*g*u)*db, g = θ(b/b_0)")


def memory_motors(capacity: float, wake: float, widths: list[int], names: list[str]) -> Arrangement:
    """One motor per memory store, each told a rate with other load 0, which is the true one:
    nothing else acts on these stores. Each reads the battery. Inputs, per store: battery,
    the store, its rate."""
    motors = [fp.send(fp.motor_at_rate(capacity, wake, f"{name} motor", n), {1 + 2 * n + j: 0. for j in range(n)},
                      f"{name} motor, P=0") for name, n in zip(names, widths)]
    size = sum(1 + 2 * n for n in widths)
    return fp.fill(motors, fp.transfer("memory motors", size, 0, size, 0, lambda m, n: n))


def body(*, parameters: Mapping[str, float] | None = None, stages: int = 1, h: float = 1.,
         learning: bool = False, memory=None, cart: float = 0., battery: float | None = None) -> Arrangement:
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

    battery = (6. if learning else 3.) if battery is None else battery
    stores = [fp.rod_segment(1., battery, "battery", **step),
              fp.rod_segment(p['cgrip'], -1., "engagement", **step),
              fp.rod_segment(p['clift'], p['xg'], "tip", **step),
              wheels(p, cart, **step),
              fp.rod_segment(p['csee'], 0., "look", **step),
              fp.rod_segment(p['csee'], p['A'], "scan", **step),
              *[fp.rod_segment(capacity, v, name, **step) for name, v in start]]
    # One learner per memory store, its circle reading only what its rule names (book P109 (f),
    # act (3)); the width of each value a rule may read.
    rules = _memory_rules(p) if learning else []
    width_of = dict(zip([key for key, _, _ in rules], widths), y=1, gaze=1, grip=1, grip_rate=1, seen=1, eaten=1)

    def as_rule(key, reads, rule):
        def rate(store, *read):
            values = {k: (x[0] if width_of[k] == 1 else x) for k, x in zip((key, *reads), (store, *read))}
            return rule(values)
        return rate

    # What each value a rule reads is called on its wire (the part's meta; drawing only).
    called = {**dict(zip([key for key, _, _ in rules], names)), 'y': 'cart', 'gaze': 'gaze', 'grip': 'grip',
              'grip_rate': 'grip rate', 'seen': 'red at gaze', 'eaten': 'food taken'}
    learners = [learner(name, as_rule(key, reads, rule), [width_of[k] for k in reads], width_of[key], capacity, p['b0'],
                        [called.get(k, k) for k in reads])
                for (key, reads, rule), name in zip(rules, names)]
    stateless = [gripper(p), lever(p), fp.sweep(p['csee'], p['b0'], p['A'], p['omega'], p['gam'], "eye"),
                 *(learners if learning
                   else [fp.sweep(p['cpat'], p['b0'], p['B'], p['nup'], p['gamp'], "patrol")])]

    # Every motor reads the battery itself, so no rate told to a motor carries live(b), and
    # nothing here reads the battery except to pass it to the motors. The memory's rules read
    # the rate the gripper tells its motor.
    def route(m: Array, n: Array) -> Array:
        b, grip, tip, y, look, scan = m[:6]
        rest = m[6:]
        seen, climb, keep, pull_grip, pull_tip, _, eaten = n
        free = theta(1 - 2 * seen)
        if learning:
            q = jnp.concatenate([jnp.stack([b, grip, tip, y, jnp.zeros_like(b), look, scan]), rest])
            v = _memory_values(q, _grip_rate(grip, keep, p), seen, eaten)
            search = p['memoryDrive'] * jnp.tanh((v['target'] - y) / 1.5)
            tail = [jnp.concatenate([b[None], *(jnp.atleast_1d(v[k]) for k in (key, *reads))])
                    for key, reads, _ in rules]
        else:
            search = p['kexp'] * (rest[0] - y)
            tail = [jnp.stack([b, rest[0], rest[1], free, jnp.zeros_like(b)])]
        # Toward what the eye holds, the most torque ktau until within rDrive of it (P109 (c)).
        torque = p['ktau'] * seen * jnp.tanh(look / p['rDrive']) + free * search
        return jnp.concatenate([jnp.stack([b, torque]),                               # wheels
                                jnp.stack([b, grip, keep, pull_grip]),                # gripper
                                jnp.stack([b, tip, grip, pull_tip]),                  # lever
                                jnp.stack([b, look, scan, free, p['lam'] * climb]),   # eye
                                *tail])

    inner_in = 2 + 4 + 4 + 5 + (sum(1 + width_of[key] + sum(width_of[k] for k in reads) for key, reads, _ in rules)
                                if learning else 5)
    wiring = fp.transfer("body and motors", inner_in, 6 + extra, 7, 5, route, lambda m: m[:5])
    return fp.with_meta(fp.fill(stores + stateless, wiring),
                        inputs=[(f"n_{i + 1}", name, 1) for i, name in
                                enumerate(("seen", "climb", "K", "pull on grip", "pull on tip", "unused", "eaten"))],
                        outputs=[(f"o_{i + 1}", name, 1) for i, name in
                                 enumerate(("battery", "grip", "tip", "cart", "look"))])


# ---------------------------------------------------------------------------
# The separator mouth and its slot (book P109 (i), act (3); the recursive note's page
# *A separator whose rate is set from outside*): the mouth is the half that settles, inside the
# seeker; the half that reaches is the spatial coupling, which computes the rate at the held
# casing and sends it in. The slot inside is the recursive note's feeder interface
# F = <food, reserve, parent mass, parent acceptance | >.
# ---------------------------------------------------------------------------

SLOT_SEEKER_INPUTS = ("seen", "climb", "K", "pull on grip", "pull on tip", "taken", "parent mass", "parent acceptance")
SLOT_SEEKER_OUTPUTS = ("battery", "grip", "tip", "cart", "look", "effort")


def seeker_with_slot(*, parameters: Mapping[str, float] | None = None, stages: int = 1, h: float = 1.,
                     learning: bool = False, memory=None, cart: float = 0., battery: float | None = None) -> Arrangement:
    """The seeker with a separator mouth and an open slot, ``F -> <8|6>``.

    Parts: the body and its motors; the mouth, ``feeding_recursive.settle`` (the chamber, a
    store of what the mouth has torn off, which receives the pass share of each gram taken,
    puts the rest in the collector and charges the battery ``mouthCost`` per gram); the
    collector of husks, ``feeding_recursive.collector``; and the identity on ``F``, the slot.

    Inputs: seen, climb, K, the attachment's pulls on grip and tip (as ``body``'s), the rate
    ``taken`` at which the spatial coupling realizes eating, and the parent's discard pair
    (the pile on the ground, for the outermost seeker). Outputs: battery, grip, tip, cart,
    look, and the effort the mouth offers, ``leat live(b) hold(grip) admit(chamber) accept``,
    ``accept`` the collector's admission. The slot is handed the chamber's mass as its food,
    the battery as its reserve and the collector's pair as its parent's.
    """
    from . import feeding_recursive as fr
    p = {**DEFAULTS, **(parameters or {})}
    step = dict(stages=stages, h=h)
    chamber_admits = fr.admit(p['chamberCap'])
    parts = [body(parameters=p, stages=stages, h=h, learning=learning, memory=memory, cart=cart, battery=battery),
             fr.settle(p['mouthPass'], p['mouthCost'], 0., "mouth", **step),
             fr.collector(p['clearance'], p['collectorCap'], 0., "husk collector", **step),
             fp.identity(4, 0, "seeker slot")]

    def route(m: Array, n: Array) -> Array:
        b, z, local = m[0], m[5], m[6:8]
        seen, climb, keep, pull_grip, pull_tip, taken, parent = n[0], n[1], n[2], n[3], n[4], n[5], n[6:8]
        return jnp.concatenate([jnp.stack([seen, climb, keep, pull_grip, pull_tip, jnp.zeros_like(taken), taken]),
                                jnp.stack([taken, b]), local,
                                parent,
                                jnp.stack([z, b]), local])

    def publish(m: Array) -> Array:
        b, grip, z, accept = m[0], m[1], m[5], m[7]
        effort = p['leat'] * theta(b / p['b0']) * theta(grip) * chamber_admits(z) * accept
        return jnp.concatenate([m[:5], effort[None]])

    wiring = fp.transfer("seeker with a slot", 7 + 4 + 2 + 4, 5 + 1 + 2, 8, 6, route, publish)
    return fp.with_meta(replace(fp.fill(parts, wiring), label="seeker with a slot"),
                        inputs=[(f"n_{i + 1}", name, 1) for i, name in enumerate(SLOT_SEEKER_INPUTS)],
                        outputs=[(f"o_{i + 1}", name, 1) for i, name in enumerate(SLOT_SEEKER_OUTPUTS)])


def absorber_filler(parameters: Mapping[str, float] | None = None) -> Arrangement:
    """The default filler of the slot: the recursive note's absorber, ``1 -> F``, a gut wall
    that takes up ``uptake`` times what the chamber holds per unit of duration and gives the
    battery ``absorbYield`` per gram."""
    from . import feeding_recursive as fr
    p = {**DEFAULTS, **(parameters or {})}
    return fr.absorber(p['absorbYield'], p['uptake'], "gut wall")


def hopper_filler(pass_fraction: float = .5, cost: float = .02, speed: float = 2., capacity: float = 6.,
                  clearance: float = 1., inner: Arrangement | None = None, *, parameters=None,
                  stages: int = 1, h: float = 1.) -> Arrangement:
    """A hopper around a feeder (the absorber by default): the recursive note's layer
    ``L_{p,c}``, a collector and a separator around the slot, filled. A gut."""
    from . import feeding_recursive as fr
    step = dict(stages=stages, h=h)
    inner = absorber_filler(parameters) if inner is None else inner
    return fr.wrap(fr.layer(fr.collector(clearance, capacity, label="gut collector", **step),
                            fr.separator(pass_fraction, cost, fr.process_rate(speed, capacity), label="gut", **step),
                            label="hopper"), inner)


def mouth_coupling(count: int, *, parameters: Mapping[str, float] | None = None,
                   stages: int = 1, h: float = 1., feeder: bool = False) -> Arrangement:
    """The spatial coupling of a seeker with a slot, ``count`` hingeless casings and the pile:
    ``<8|6> (x) <0|4>^count (x) <0|2> -> <0|2>``.

    Its parts, beside the identity on the seeker's, casings' and pile's interfaces: per casing
    a ``contact`` (the lever's attachment: both gates sent ``G_k``, the stretch ``x_k - h``)
    and a ``latch``, and one wide ``push`` ("taking"), all fed by one static map, which also
    computes the latch law's gates (note ``sec.gates``), what the eye sees, and the loads the
    attachment puts on the grip, the tip and each latch. (Until 2026-11-23 the hinged model's
    coupling had two pushes, opening and eating.) The coupling is the half of the separator
    that reaches. At casing ``k`` it realizes
    ``J_k = effort s_k near_Rh(cart - place_k) near_R(height_k - x0) food_k``, pushes
    ``-J_k d food_k``, and tells the seeker ``sum_k J_k`` at its load-free input ``taken``,
    with the pile's pair as its parent's.

    With ``feeder`` it is the coupling of a robot inside another's slot (book P109 (j)),
    ``<8|6> (x) <0|4>^count -> F``, ``F`` the feeder interface ``<food, reserve, parent
    mass, parent acceptance | >``: no pile, the parent's pair arriving from outside; the mass
    handed in on ``food`` laid out among the casings by one Petri transition per casing,
    ``layRate food / count (d food_k - d food)``; and the robot's own battery joined to the
    reserve by a cord, a spring of conductance ``cord`` between the two, whose field
    ``-cord (b - reserve) d(b - reserve)`` moves charge from the fuller to the emptier.
    """
    p = {**DEFAULTS, **(parameters or {})}
    n = count
    step = dict(stages=stages, h=h)
    cost = fp.contact_cost(p['kappa0'], p['kd'])
    width = 6 + 4 * n + (0 if feeder else 2)

    def route(m: Array, outer: Array) -> Array:
        b, grip, tip, cart, look, effort = m[:6]
        food, height, red, place = m[6:6 + 4 * n].reshape((n, 4)).T
        pile = outer[2:4] if feeder else m[6 + 4 * n:width]
        stroke, latch = m[width:width + n], m[width + n:]
        along = near(cart - place, p['Rh'])

        def gate(grip, tip, latch):
            return fp.engagement(latch) * theta(grip) * near(height - tip, p['R']) * along

        def attachment(grip, tip, latch):
            g = gate(grip, tip, latch)
            return jnp.sum(cost(g, g, height - tip, stroke))

        pulls = jax.grad(attachment, argnums=(0, 1, 2))(grip, tip, latch)
        g = gate(grip, tip, latch)
        s = fp.engagement(latch)
        offered = near(height - tip, p['R']) * along * theta((food - p['f0']) / p['f0'])
        score = offered / (1 + ((height - tip) / p['R']) ** 2 + ((cart - place) / p['Rh']) ** 2)
        others = ~jnp.eye(n, dtype=bool)
        first = jnp.prod(jnp.where(others, theta((score[:, None] - score[None, :]) / .05), 1.), axis=1)
        free = jnp.prod(jnp.where(others, 1 - theta(latch / .3), 1.), axis=1)
        held = theta((grip + .8) / .4)
        acquire = theta((offered - .15) / .15) * first * (1 - held)
        retain = 1 - (1 - offered) * (1 - held)
        taken = effort * s * along * near(height - p['x0'], p['R']) * food
        offset = (cart + look - place) / p['w']
        signal = red * jnp.exp(-offset ** 2)
        told = jnp.concatenate([jnp.stack([signal.sum(), jnp.sum(-2 * offset * signal / p['w']), jnp.sum(s * offered),
                                           -pulls[0], -pulls[1], taken.sum()]), pile])
        inner = [told,
                 jnp.stack([g, g, height - tip], axis=1).ravel(),
                 jnp.stack([jnp.broadcast_to(b, (n,)), free, acquire, retain, -pulls[2]], axis=1).ravel(),
                 -taken, food]
        if feeder:
            handed, reserve = outer[0], outer[1]
            laid = p['layRate'] * handed / n * jnp.ones(n)
            inner += [laid, food - handed, (b - reserve)[None]]
        return jnp.concatenate(inner)

    pieces = [*[fp.contact(p['kappa0'], p['kd'], p['bd'], p['kr'], 0., f"attachment {k + 1}", **step)
                for k in range(n)],
              *[fp.latch(p['latchCost'], p['latchRate'], p['b0'], 0., f"contact latch {k + 1}", **step)
                for k in range(n)],
              fp.push("push: taking", n)]
    if feeder:
        # The cord is a spring of constant stiffness between the robot's battery and the reserve.
        pieces += [fp.push("push: laying out", n), fp.send(fp.spring("cord"), {0: p['cord']}, "cord")]
    pieces_in = 10 * n + (2 * n + 1 if feeder else 0)
    inside = parallel_arrangements(fp.identity(8, width, "seeker and casings" if feeder else "seeker, casings and pile"),
                                   tensor_arrangements(pieces))
    if feeder:
        wiring = fp.transfer("inner spatial coupling", 8 + pieces_in, width + 2 * n, 4, 0, route)
        return replace(compose_seq(inside, wiring), label="inner spatial coupling")
    wiring = fp.transfer("spatial coupling", 8 + pieces_in, width + 2 * n, 0, 2, route, lambda m: m[:2])
    return replace(compose_seq(inside, wiring), label="spatial coupling")


def robot_style(places=(2.,), *, parameters: Mapping[str, float] | None = None, stages: int = 1, h: float = 1.,
                learning: bool = False, memory=None, cart: float = 0.) -> Arrangement:
    """A robot-style seeker, an arrow ``F -> F`` of the tangent subcategory (book P109 (j);
    David, 2026-11-21: "i think the seeker may be a copy of the whole robot, but i'm not sure
    it needs to accomplish eerything the same way. a hopper is also a seeker. can we leave it
    open to be filled by any seeker style?").

    A whole robot, every organ kept (the run's call, his to change): a seeker with its own
    separator mouth, husk collector, battery and slot, and ``len(places)`` hingeless casings
    in its own arena, under ``mouth_coupling(..., feeder=True)``. Handed a mass on its food
    wire, it lays it out among its casings and seeks it; its battery starts empty and is
    joined to the reserve by a cord; its husks go to the parent's collector; its slot is open,
    so ``compose_seq(filler, robot_style())`` is a feeder ``1 -> F`` for any feeder
    ``filler``, a hopper, an absorber or another robot.
    """
    p = {**DEFAULTS, **(parameters or {})}
    step = dict(stages=stages, h=h)
    seeker = seeker_with_slot(parameters=p, learning=learning, memory=memory, cart=cart, battery=0., **step)
    casings = [source(x, 0., parameters=p, hinge=False, **step) for x in places]
    world = tensor_arrangements([seeker, *casings])
    return replace(compose_seq(world, mouth_coupling(len(places), parameters=p, feeder=True, **step)),
                   label="robot style")


def feeding_with_slot(places=(4., -8., 8.), foods=(.05, 10., 10.), *, filler: Arrangement | None = None,
                      parameters: Mapping[str, float] | None = None, stages: int = 1, h: float = 1.,
                      learning: bool = False, memory=None, cart: float = 0., pile: float = 0.,
                      battery: float | None = None) -> Arrangement:
    """Feeding with a separator mouth: ``mouth_coupling(n) . ((seeker_with_slot . filler)
    (x) casing_1 (x) ... (x) casing_n (x) pile)``, the casings without their hinge.
    ``filler`` is any ``1 -> F`` (the absorber by default; a hopper; a seeker style).
    ``battery`` is the battery's starting charge (``body``'s default when ``None``)."""
    from . import feeding_recursive as fr
    if len(places) != len(foods):
        raise ValueError('One food per casing')
    p = {**DEFAULTS, **(parameters or {})}
    step = dict(stages=stages, h=h)
    filler = absorber_filler(p) if filler is None else filler
    robot = compose_seq(filler, seeker_with_slot(parameters=p, learning=learning, memory=memory, cart=cart,
                                                 battery=battery, **step))
    casings = [source(x, f, parameters=p, hinge=False, **step) for x, f in zip(places, foods)]
    world = tensor_arrangements([robot, *casings, fr.pile(pile, "pile on the ground", **step)])
    return compose_seq(world, mouth_coupling(len(casings), parameters=p, **step))
