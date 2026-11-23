"""Feeding with a separator mouth and an open slot (book P109 (i), act (3); (k)).

``feeding_composed.feeding_with_slot``: the seeker's mouth is the half of a separator that
settles (``feeding_recursive.settle``, the chamber), the spatial coupling is the half that
reaches (it realizes the rate at the held casing and sends it in), the husks go to a
collector and from it to a pile on the ground (David, 2026-11-22: "pile on the ground."), and
the slot is filled by any feeder ``1 -> F``: the absorber, or a hopper around it.

Two tests, from the design ``notes/runs/separator-mouth-9e1759d5/DESIGN.md`` in the book repo:

* at the gate fully open, the field is the separator's plus the filler's on (food, chamber,
  collector, the batteries) and nothing else, against the twin that does not eat;
* the recursive note's yield rule with the motors' payments apart, pointwise for both fills,
  and over a meal: the husks are ``1 - p`` of what was taken, and the charge's gain, plus the
  chamber's pending energy, plus what the motors paid, is ``p epsilon - c`` per gram.

Since P109 (k) the robot's charge is its battery's plus each organ's (``ORGANS``, the mouth's
and the hands'), every organ battery a store of capacity ``cLocal``, so its charge is
``cLocal`` times its level; the mouth pays its tearing from its own; the gut wall slows near
the cap and is slow (``uptake`` 0.15), so a meal is not wholly digested in 20 units of
duration and its pending energy, the chamber's, is counted. Spilling is off in the meal (``spillRate`` 0): a held casing's tremor
spills, and spilled food is neither taken nor a husk; ``test_spilling`` checks it apart.

What runs in IMC: every world is a library composite; fields are its own ``omega``; the meal
is stepped by ``pcK.PhiK``. Beside it, in ordinary arithmetic: the inventories, linear in the
stores, and the sum over the run of the twin's load on the charge (the motors' payments).
Stores are found by label (``export_step_js.stores``), never by a number typed here.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from arrangements import feeding_composed as fc
from arrangements import feeding_recursive as fr
from arrangements.feeding import DEFAULTS
from arrangements.interpretation import trivial_omega
from arrangements.pcK import PhiK
from tests.feeding_layout import configuration_index, flat_index

jax.config.update("jax_enable_x64", True)
EMPTY = jnp.zeros(0)
RNG = np.random.default_rng(11)
STILL = {'kexp': 0., 'ktau': 0.}                  # the cart stays put, as tests/test_explore.py
P, C, EPS, CL = DEFAULTS['mouthPass'], DEFAULTS['mouthCost'], DEFAULTS['absorbYield'], DEFAULTS['cLocal']
ORGAN_BATTERIES = [f"{name} battery" for name in (*fc.ORGANS, "mouth", "hands")]


def _field(world):
    """The load the composite's field puts on its stores, ``omega``'s first component. A store
    of capacity ``c`` moves at ``1/c`` times its load, so the load on a store is the rate of
    what it holds: grams on a mass store of capacity 1, charge on a battery of any capacity."""
    return jax.jit(lambda q: world.omega(q, EMPTY, EMPTY)[0])


def _gate_open_state(world, at):
    """Every battery at 3, grip 1.5, the tip and the casing at the mouth, the air's memory
    with the casing (speed 0, so nothing spills), the cart at the casing's place, the latch
    full, the lid open, the robot hungry: every gate on the mouth's rate is 1."""
    q = np.asarray(world.P.position(world.P.init), dtype=float).copy()
    for label in ("battery", *ORGAN_BATTERIES):
        q[at(label)] = 3.
    q[at("engagement")], q[at("tip")], q[at("cart")] = 1.5, DEFAULTS['x0'], 0.
    q[at("casing")] = q[at("memory", 1)] = DEFAULTS['x0']      # the air dragger's memory, after the wheel's
    q[at("contact latch 1 store")], q[at("lid")], q[at("satiety")] = 1., 1., 0.
    return q


def test_the_field_at_an_open_gate_is_the_separator_and_the_absorber():
    world = fc.feeding_with_slot([0.], [10.], parameters=STILL)
    twin = fc.feeding_with_slot([0.], [10.], parameters={**STILL, 'leat': 0., 'uptake': 0.})
    assert (world.in_dim_N, world.out_dim_N) == (0, 2)
    at = configuration_index(world)
    B, CHAMBER, COLLECTOR, FOOD, MOUTH = (at("battery"), at("mouth ready store"), at("husk collector store"),
                                          at("food"), at("mouth battery"))
    f, g = _field(world), _field(twin)
    accept = fr.admit(DEFAULTS['collectorCap'])
    admit = fr.admit(DEFAULTS['chamberCap'])
    for _ in range(20):
        q = _gate_open_state(world, at)
        q[FOOD], q[CHAMBER], q[COLLECTOR] = RNG.uniform(.5, 10.), RNG.uniform(0., 7.), RNG.uniform(0., 3.5)
        q = jnp.asarray(q)
        diff = np.asarray(f(q) - g(q))
        rate = DEFAULTS['leat'] * q[FOOD] * admit(q[CHAMBER]) * accept(q[COLLECTOR])
        uptake = DEFAULTS['uptake'] * q[CHAMBER]        # the battery at 3 is far below the cap
        expected = np.zeros_like(diff)
        expected[FOOD] = -rate
        expected[CHAMBER] = P * rate - uptake
        expected[COLLECTOR] = (1 - P) * rate
        expected[MOUTH] = -C * rate                     # the mouth tears from its own battery
        expected[B] = EPS * uptake
        np.testing.assert_allclose(diff, expected, rtol=0, atol=1e-11)
        assert rate > 1e-3


def test_the_gut_wall_stops_at_the_cap():
    """The gut wall's uptake is ``uptake`` times the chamber below ``battCap - capWidth`` and
    nothing at the cap: the battery cannot charge beyond it (David, 2026-11-23)."""
    world = fc.feeding_with_slot([0.], [10.], parameters={**STILL, 'leat': 0.})
    twin = fc.feeding_with_slot([0.], [10.], parameters={**STILL, 'leat': 0., 'uptake': 0.})
    at = configuration_index(world)
    f, g = _field(world), _field(twin)
    cap, width = DEFAULTS['battCap'], DEFAULTS['capWidth']
    for b, share in ((cap - 3., 1.), (cap - width, 1.), (cap - width / 2, .5), (cap, 0.), (cap + 1., 0.)):
        q = _gate_open_state(world, at)
        q[at("battery")], q[at("mouth ready store")] = b, 4.
        q = jnp.asarray(q)
        gained = float(f(q)[at("battery")] - g(q)[at("battery")])
        np.testing.assert_allclose(gained, EPS * DEFAULTS['uptake'] * 4. * share, rtol=0, atol=1e-12)


def _fills():
    """The two fills and their inventories, by store label: the absorber, and a hopper (a gut)
    around it, whose stores are the filler's own (``'gut collector'`` first, then its ready
    store), the first two of the configuration."""
    gut = (.5, .02)
    inner_pair = fr.wrap_yield(gut, (1., EPS))
    yield {'filler': None, 'twin': None, 'pair': fr.wrap_yield((P, C), (1., EPS)),
           'U': {"husk collector store": 1.}, 'V': {"mouth ready store": EPS}, 'own': {}}
    # Around the absorber the gut's inventories are its collector and nothing of its ready store
    # (the absorber's pair is (1, EPS)); its net energy still to deliver is EPS per ready gram.
    # Its twin neither separates nor absorbs.
    yield {'filler': fc.hopper_filler(*gut), 'twin': fc.hopper_filler(*gut, speed=0., parameters={'uptake': 0.}),
           'pair': fr.wrap_yield((P, C), inner_pair),
           'U': {"husk collector store": 1., "mouth ready store": 1. - inner_pair[0]},
           'V': {"mouth ready store": inner_pair[1]},
           'own': {'U': {0: 1., 1: 0.}, 'V': {1: EPS}}}


def _batteries(at):
    """The configuration indices of the robot's battery and its organs' (``ORGAN_BATTERIES``)."""
    return [at("battery"), *(at(label) for label in ORGAN_BATTERIES)]


def _charge(at):
    """The robot's charge, as coefficients on the configuration's levels: its battery's
    level (capacity 1), and ``cLocal`` times each organ battery's."""
    return {at("battery"): 1., **{at(label): CL for label in ORGAN_BATTERIES}}


def test_the_yield_rule_with_the_motors_payments_apart():
    for fill in _fills():
        world = fc.feeding_with_slot([0.], [10.], filler=fill['filler'], parameters=STILL)
        twin = fc.feeding_with_slot([0.], [10.], filler=fill['twin'],
                                    parameters={**STILL, 'leat': 0., 'uptake': 0.})
        at = configuration_index(world)
        (a, e), own = fill['pair'], fill['own']
        u = {at(label): c for label, c in fill['U'].items()} | own.get('U', {})
        v = {at(label): c for label, c in fill['V'].items()} | own.get('V', {})
        batteries, FOOD, PILE = _batteries(at), at("food"), at("pile on the ground store")
        f, g = _field(world), _field(twin)
        taken = 0.
        for _ in range(30):
            q = _gate_open_state(world, at)
            q[FOOD] = RNG.uniform(.5, 10.)
            for i in (*u, *v):
                q[i] = RNG.uniform(0., 3.)
            q = jnp.asarray(q)
            rate, twin_rate = np.asarray(f(q)), np.asarray(g(q))
            discard = rate[PILE] + sum(c * rate[i] for i, c in u.items()) + (1 - a) * rate[FOOD]
            energy = rate[batteries].sum() + sum(c * rate[i] for i, c in v.items()) + e * rate[FOOD]
            np.testing.assert_allclose(discard, 0., rtol=0, atol=1e-11)
            np.testing.assert_allclose(energy, twin_rate[batteries].sum(), rtol=0, atol=1e-11)
            taken -= rate[FOOD]
        assert taken > 1.


def test_a_meal_yields_the_pair_per_gram():
    """One full casing at the mouth; Runge-Kutta steps of 0.001 for 20 units of duration; no
    spilling. (0.001, not 0.002 as before P109 (k): the organs' batteries are small, so their
    cords are fast, and the trapezoid sum of the payments needs the shorter step; the per-gram
    error is 1.0e-4, 1.8e-5 and 8.7e-7 at 0.004, 0.002 and 0.001, so it goes to zero.) The chamber still holds food at the end (the gut wall is slow since P109 (k),
    about 0.47 g of the 7.9 g it received): its pending energy ``EPS z`` is counted."""
    h, steps = .001, 20000
    quiet = {**STILL, 'spillRate': 0.}
    world = fc.feeding_with_slot([0.], [10.], parameters=quiet, stages=4, h=h)
    one = fc.feeding_with_slot([0.], [10.], parameters=quiet)
    at = configuration_index(one)
    charge, batteries = _charge(at), _batteries(at)
    CHAMBER, COLLECTOR, FOOD, PILE = (at("mouth ready store"), at("husk collector store"), at("food"),
                                      at("pile on the ground store"))
    twin_field = _field(fc.feeding_with_slot([0.], [10.], parameters={**quiet, 'leat': 0., 'uptake': 0.}))
    dynamics = PhiK(world)
    tick = lambda _, s: dynamics.with_state(s).run_one((EMPTY, trivial_omega(0)), lambda _: (jnp.zeros(2), EMPTY))[1]
    # Between steps the four-round world's state has the one-round world's shape; read it there.
    position = one.P.position
    total = lambda q: sum(c * q[i] for i, c in charge.items())
    paying = lambda s: twin_field(position(s))[jnp.asarray(batteries)].sum()

    def tick_and_pay(_, carry):
        # The motors' payments, by the trapezoid rule on each step of 0.002 (beside the run).
        s, paid = carry
        after = tick(_, s)
        return after, paid - h * (paying(s) + paying(after)) / 2

    advance = jax.jit(lambda carry: jax.lax.fori_loop(0, 50, tick_and_pay, carry))
    state, paid = world.P.init, 0.
    q0 = np.asarray(position(state))
    for _ in range(steps // 50):
        state, paid = advance((state, paid))
        q = np.asarray(position(state))
        husks = q[PILE] + q[COLLECTOR]
        np.testing.assert_allclose(husks, (1 - P) * (q0[FOOD] - q[FOOD]), rtol=0, atol=1e-9)
    taken = q0[FOOD] - q[FOOD]
    assert taken > 9. and q[CHAMBER] > .1            # eaten, and not yet all digested
    per_gram = (total(q) - total(q0) + EPS * (q[CHAMBER] - q0[CHAMBER]) + paid) / taken
    np.testing.assert_allclose(per_gram, P * EPS - C, rtol=0, atol=1e-5)


def test_spilling():
    """A casing lifted to the mouth's height away from the cart and let fall: with its lid
    closed it spills nothing; open, its food goes to the pile while it falls (a Petri
    transition at ``spillRate`` times the lid's opening, the food and the speed above
    ``vSpill``). The lid is held as set (``lidRate`` 0)."""
    h = .004
    spilt = []
    for cover in (0., 1.):
        params = {**STILL, 'lidRate': 0.}
        world = fc.feeding_with_slot([6.], [10.], parameters=params, stages=4, h=h)
        one = fc.feeding_with_slot([6.], [10.], parameters=params)
        at = flat_index(one)
        flat, unravel = ravel_pytree(world.P.init)
        flat = np.asarray(flat).copy()
        flat[at("casing")], flat[at("lid")] = 0., cover
        dynamics = PhiK(world)
        tick = lambda _, s: dynamics.with_state(s).run_one((EMPTY, trivial_omega(0)), lambda _: (jnp.zeros(2), EMPTY))[1]
        end = np.asarray(ravel_pytree(jax.jit(lambda s: jax.lax.fori_loop(0, 750, tick, s))(unravel(jnp.asarray(flat))))[0])
        food, pile = end[at("food")], end[at("pile on the ground store")]
        np.testing.assert_allclose(food + pile, 10., rtol=0, atol=1e-9)       # spilled food lands on the pile
        spilt.append(pile)
    assert spilt[0] < 1e-9 and spilt[1] > 1.
