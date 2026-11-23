"""The recursive feeding note's operation and its class of feeders, as library arrangements.

The note is the book repo's ``background/feeding-imc/recursive-feeding.tex``. Its value
types are ``M = R`` (grams), ``E = R`` (energy) and ``D = M x R``, a discard pair (the mass
in a collector, and how much of an incoming rate it admits, between 0 and 1); a feeder's
input is ``H = M x E x D``, here four numbers in the order food, reserve, parent mass,
parent acceptance. Its interfaces, written ``<inputs|outputs>``:

* ``F = <4|0>``, a feeder: it is handed food, a reserve and its parent's discard pair;
* ``C = <2|2>``, a collector: in, its parent's discard pair; out, its own;
* ``Sep = <4|1>``, a separator: in, as a feeder with its own collector's pair as the
  discard; out, the ready stock.

Everything is a morphism of IMC's Cartesian tangent subcategory, composed from
``feeding_parts`` by ``compose_seq`` and ``tensor_arrangements``; every inclination field is
one ``push``, a rate times the differential of a fixed signed sum of stocks, which is the
book's Petri transition (Ch 3 ``eqn.transition_effort``):

* ``absorber``        -- ``1 -> F``, field ``uptake (-d food + epsilon d reserve)``;
* ``collector``       -- ``1 -> C``, a store ``m`` and ``export (d parentMass - dm)``;
* ``separator``       -- ``1 -> Sep``, a store ``z`` and
  ``rate (p dz + (1 - p) d localMass - d food - c d reserve)``;
* ``ternary_wiring``  -- the note's ``W : C (x) Sep (x) F -> F``, static, zero field;
* ``feed``            -- ``W`` filled with three; ``layer`` -- ``W`` filled with a collector
  and a separator, its feeder slot open, the note's ``L_{p,c} : F -> F``; ``wrap`` -- a
  layer around a feeder;
* ``settle`` and ``reach`` -- the note's page *A separator whose rate is set from outside*:
  the half that stores the ready mass and is sent its rate, and the static half that
  computes the rate behind a gate, sends it in and pushes back on the food wires;
  ``gated_wiring`` is ``W`` with those food wires and the gate's values in place of the
  one food wire;
* ``pile``            -- a collector that exports nothing and admits everything, its own
  parent a wall: the ground under the outermost feeder;
* ``wrap_yield``      -- the note's action of a layer's settings on a yield pair,
  ``(a, e) |-> (p a, p e - c)``.

Built 2026-11-21 for the book's P109 (i) (Claude Fable 5.2, 9e1759d5). The tests are
``tests/test_feeding_recursive.py``.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Tuple

import jax.numpy as jnp
from jax import Array

from . import feeding_parts as fp
from .arrangement import Arrangement
from .feeding import theta
from .wiring import compose_seq, parallel_arrangements, tensor_arrangements

# A rate law of a separator: (ready stock z, food, reserve, local mass, local acceptance) -> grams per unit of duration.
RateLaw = Callable[[Array, Array, Array, Array, Array], Array]


def admit(capacity: float | None) -> Callable[[Array], Array]:
    """A store's admission function ``M -> [0, 1]``: ``1`` up to half the capacity, falling
    smoothly to ``0`` at the capacity. With no capacity it is the constant ``1``."""
    if capacity is None:
        return lambda m: jnp.ones_like(m)
    if fp.nonpositive(capacity):
        raise ValueError("a store's capacity is positive")
    return lambda m: theta((capacity - m) / (capacity / 2))


def process_rate(speed: float, capacity: float | None = None) -> RateLaw:
    """The note's example rate law, ``speed * food * reserve * admit(z) * localAccept``."""
    accept = admit(capacity)
    return lambda z, food, reserve, local_mass, local_accept: speed * food * reserve * accept(z) * local_accept


def absorber(energy_yield: float, uptake: float | Callable[[Array, Array, Array, Array], Array],
             label: str = "absorber") -> Arrangement:
    """The physical absorber ``1 -> F``: stateless, field ``j (-d food + epsilon d reserve)``
    with ``j = uptake(food, reserve, parentMass, parentAccept)``; a number ``k`` for
    ``uptake`` is the law ``k * food``. It puts no load on the discard pair."""
    rate = uptake if callable(uptake) else (lambda food, reserve, mass, accept: uptake * food)
    return fp.pushes(label, 4, [("absorb", lambda v: rate(v[0], v[1], v[2], v[3]),
                                 lambda v: -v[0] + energy_yield * v[1])],
                     inputs=[("f", "food", 1), ("e", "reserve", 1), ("m_p", "parent mass", 1),
                             ("a_p", "parent acceptance", 1)],
                     forms=[(None if callable(uptake) else "k*f", "-f + ε*e")],
                     constants={"ε": energy_yield, **({} if callable(uptake) else {"k": uptake})})


def collector(clearance: float, capacity: float | None = None, mass: float = 0., label: str = "collector", *,
              stages: int = 1, h: float = 1.) -> Arrangement:
    """The collector ``1 -> C``: a store ``m`` of discard, sending out ``(m, admit(m))``, with
    field ``export (d parentMass - dm)`` and ``export = clearance * m * parentAccept``."""
    accept = admit(capacity)
    field = fp.pushes(f"{label}: export", 3, [("export", lambda v: clearance * v[0] * v[2], lambda v: v[1] - v[0])],
                      inputs=[("m", "store", 1), ("m_p", "parent mass", 1), ("a_p", "parent acceptance", 1)],
                      forms=[("κ*m*a_p", "m_p - m")], constants={"κ": clearance})
    wiring = fp.transfer(label, 3, 1, 2, 2, lambda m, n: jnp.concatenate([m, n]),
                         lambda m: jnp.stack([m[0], accept(m[0])]))
    return fp.with_meta(fp.fill([fp.rod_segment(1., mass, f"{label} store", stages=stages, h=h), field], wiring),
                        inputs=[("m_p", "parent mass", 1), ("a_p", "parent acceptance", 1)],
                        outputs=[("m", "mass", 1), ("a", "acceptance", 1)])


def pile(mass: float = 0., label: str = "pile", *, stages: int = 1, h: float = 1.) -> Arrangement:
    """The ground's pile ``1 -> <0|2>``: a collector that exports nothing and admits
    everything, its own parent a wall. It sends out ``(mass, 1)``."""
    return fp.send(collector(0., None, mass, label, stages=stages, h=h), {0: 0., 1: 0.}, label)


def separator(pass_fraction: float, cost: float, rate_law: RateLaw, ready: float = 0., label: str = "separator", *,
              stages: int = 1, h: float = 1.) -> Arrangement:
    """The separator ``1 -> Sep``: a store ``z`` of ready mass, sent out, with field
    ``rate (p dz + (1 - p) d localMass - d food - c d reserve)``: of each gram it takes
    from the food, ``p`` lands in its store, the rest in its collector, and the reserve
    pays ``c``."""
    p, c = pass_fraction, cost
    field = fp.pushes(f"{label}: separate", 5,
                      [("separate", lambda v: rate_law(v[0], v[1], v[2], v[3], v[4]),
                        lambda v: p * v[0] + (1 - p) * v[3] - v[1] - c * v[2])],
                      inputs=[("z", "ready store", 1), ("f", "food", 1), ("e", "reserve", 1),
                              ("m_l", "local mass", 1), ("a_l", "local acceptance", 1)],
                      forms=[(None, "p*z + (1 - p)*m_l - f - c*e")], constants={"p": p, "c": c})
    wiring = fp.transfer(label, 5, 1, 4, 1, lambda m, n: jnp.concatenate([m, n]), lambda m: m)
    return fp.fill([fp.rod_segment(1., ready, f"{label} ready store", stages=stages, h=h), field], wiring)


def ternary_wiring(label: str = "feeding wiring") -> Arrangement:
    """The note's ``W : C (x) Sep (x) F -> F``: static, zero field. The collector is sent the
    parent's pair; the separator the food, the reserve and the local pair; the feeder the
    ready stock, the reserve and the local pair."""
    def route(m: Array, n: Array) -> Array:
        local, ready = m[:2], m[2:3]
        food, reserve, parent = n[0:1], n[1:2], n[2:4]
        return jnp.concatenate([parent, food, reserve, local, ready, reserve, local])

    return fp.transfer(label, 10, 3, 4, 0, route)


def feed(collect: Arrangement, separate: Arrangement, feeder: Arrangement, label: str = "feed") -> Arrangement:
    """The ternary operation on three filled slots: ``W . (collect (x) separate (x) feeder) : 1 -> F``."""
    return replace(fp.fill([collect, separate, feeder], ternary_wiring()), label=label)


def layer(collect: Arrangement, separate: Arrangement, label: str = "layer") -> Arrangement:
    """The unary layer ``F -> F``: ``W . (collect (x) separate (x) id_F)``, its feeder slot open."""
    return replace(fp.fill([collect, separate, fp.identity(4, 0, "feeder slot")], ternary_wiring()), label=label)


def wrap(outer: Arrangement, feeder: Arrangement) -> Arrangement:
    """A layer around a feeder: ``outer . feeder``."""
    return compose_seq(feeder, outer)


def wrap_yield(settings: Tuple[float, float], pair: Tuple[float, float]) -> Tuple[float, float]:
    """The note's ``T_{p,c}``: a layer with pass fraction ``p`` and cost ``c`` sends the yield
    pair ``(a, e)``, grams absorbed and net energy per gram, to ``(p a, p e - c)``."""
    (p, c), (a, e) = settings, pair
    return p * a, p * e - c


def settle(pass_fraction: float, cost: float, ready: float = 0., label: str = "mouth", *,
           stages: int = 1, h: float = 1.) -> Arrangement:
    """The half of a separator that settles, ``1 -> <4|1>``: a store ``z`` of ready mass,
    sent out. Inputs: the rate ``taken`` at which food is being taken, the reserve, and the
    local discard pair. Field ``taken (p dz + (1 - p) d localMass - c d reserve)``,
    load-free at ``taken``."""
    p, c = pass_fraction, cost
    field = fp.pushes(f"{label}: settle", 5,
                      [("settle", lambda v: v[1], lambda v: p * v[0] + (1 - p) * v[3] - c * v[2])],
                      inputs=[("z", "ready store", 1), ("J", "taken", 1), ("e", "reserve", 1),
                              ("m_l", "local mass", 1), ("a_l", "local acceptance", 1)],
                      forms=[("J", "p*z + (1 - p)*m_l - c*e")], constants={"p": p, "c": c})
    wiring = fp.transfer(label, 5, 1, 4, 1, lambda m, n: jnp.concatenate([m, n]), lambda m: m)
    return fp.fill([fp.rod_segment(1., ready, f"{label} ready store", stages=stages, h=h), field], wiring)


def reach(gate: Callable[[Array], Array], rate_law: RateLaw, count: int = 1, further: int = 0,
          label: str = "reach") -> Arrangement:
    """The half of a separator that reaches, static ``<4|1> -> <further + count + 3 | 1>``.

    Its inputs: ``further`` values the gate reads, ``count`` food wires, the reserve and the
    local discard pair. With ``J_k = gate(values)_k * rate_law(z, food_k, reserve, local)``
    it sends the half that settles the rate ``sum_k J_k`` and has the field
    ``-sum_k J_k d food_k``, one ``push`` of width ``count``. Composed after ``settle`` it is
    a separator whose rate law is the gate times ``rate_law`` (the note's Proposition 1)."""
    def route(m: Array, n: Array) -> Array:
        values, food = n[:further], n[further:further + count]
        reserve, local_mass, local_accept = n[further + count], n[further + count + 1], n[further + count + 2]
        rates = gate(values) * rate_law(m[0], food, reserve, local_mass, local_accept)
        return jnp.concatenate([jnp.stack([rates.sum(), reserve, local_mass, local_accept]), -rates, food])

    inside = parallel_arrangements(fp.identity(4, 1, "separator sent its rate"), fp.push("push: taking", count))
    wiring = fp.transfer(label, 4 + 2 * count, 1, further + count + 3, 1, route, lambda m: m)
    return replace(compose_seq(inside, wiring), label=label)


def gated_wiring(count: int = 1, further: int = 0, label: str = "feeding wiring") -> Arrangement:
    """The note's ``W_{G,n}``: the ternary wiring with ``further`` gate values and ``count``
    food wires in place of the one food wire, ``C (x) <further + count + 3 | 1> (x) F ->
    <further + count + 3 | 0>``."""
    k = further + count

    def route(m: Array, n: Array) -> Array:
        local, ready = m[:2], m[2:3]
        front, reserve, parent = n[:k], n[k:k + 1], n[k + 1:k + 3]
        return jnp.concatenate([parent, front, reserve, local, ready, reserve, local])

    return fp.transfer(label, 2 + (k + 3) + 4, 3, k + 3, 0, route)
