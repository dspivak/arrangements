"""Feeding with a separator mouth and an open slot (book P109 (i), act (3)).

``feeding_composed.feeding_with_slot``: the seeker's mouth is the half of a separator that
settles (``feeding_recursive.settle``, the chamber), the spatial coupling is the half that
reaches (it realizes the rate at the held casing and sends it in), the husks go to a
collector and from it to a pile on the ground (David, 2026-11-22: "pile on the ground."), and
the slot is filled by any feeder ``1 -> F``: the absorber, or a hopper around it.

Two tests, from the design ``notes/runs/separator-mouth-9e1759d5/DESIGN.md`` in the book repo:

* at the gate fully open, the field is the separator's plus the filler's on (food, chamber,
  collector, battery) and nothing else, against the twin that does not eat;
* the recursive note's yield rule with the motors' payments apart, pointwise for both fills,
  and over a meal: the husks are ``1 - p`` of what was taken, and the battery's gain plus
  what the motors paid is ``p epsilon - c`` per gram once the chamber and collector are empty.

What runs in IMC: every world is a library composite; fields are its own ``omega``; the meal
is stepped by ``pcK.PhiK``. Beside it, in ordinary arithmetic: the inventories, linear in the
stores, and the sum over the run of the twin's load on the battery (the motors' payments).
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

jax.config.update("jax_enable_x64", True)
EMPTY = jnp.zeros(0)
RNG = np.random.default_rng(11)
STILL = {'kexp': 0., 'ktau': 0.}                  # the cart stays put, as tests/test_explore.py
P, C, EPS = DEFAULTS['mouthPass'], DEFAULTS['mouthCost'], DEFAULTS['absorbYield']
# Configuration indices without learning, absorber fill, one casing (the design's order).
B, CHAMBER, COLLECTOR, FOOD, PILE, LATCH = 0, 9, 10, 11, 15, 17


def _field(world):
    return jax.jit(lambda q: world.omega(q, EMPTY, EMPTY)[0])


def _gate_open_state(world):
    """Battery 3, grip 1.5, the tip and the casing at the mouth, the cart at the casing's
    place, the latch full: every gate on the mouth's rate is 1."""
    q = np.asarray(world.P.position(world.P.init), dtype=float).copy()
    q[B], q[1], q[2], q[3] = 3., 1.5, DEFAULTS['x0'], 0.
    q[12] = DEFAULTS['x0']
    q[LATCH] = 1.
    return q


def test_the_field_at_an_open_gate_is_the_separator_and_the_absorber():
    world = fc.feeding_with_slot([0.], [10.], parameters=STILL)
    twin = fc.feeding_with_slot([0.], [10.], parameters={**STILL, 'leat': 0., 'uptake': 0.})
    assert (world.in_dim_N, world.out_dim_N, world.P.Q_dim) == (0, 2, 18)
    f, g = _field(world), _field(twin)
    accept = fr.admit(DEFAULTS['collectorCap'])
    admit = fr.admit(DEFAULTS['chamberCap'])
    for _ in range(20):
        q = _gate_open_state(world)
        q[FOOD], q[CHAMBER], q[COLLECTOR] = RNG.uniform(.5, 10.), RNG.uniform(0., 7.), RNG.uniform(0., 3.5)
        q = jnp.asarray(q)
        diff = np.asarray(f(q) - g(q))
        rate = DEFAULTS['leat'] * q[FOOD] * admit(q[CHAMBER]) * accept(q[COLLECTOR])
        uptake = DEFAULTS['uptake'] * q[CHAMBER]
        expected = np.zeros_like(diff)
        expected[FOOD] = -rate
        expected[CHAMBER] = P * rate - uptake
        expected[COLLECTOR] = (1 - P) * rate
        expected[B] = -C * rate + EPS * uptake
        np.testing.assert_allclose(diff, expected, rtol=0, atol=1e-11)
        assert rate > 1e-3


def _fills():
    """The two fills and their inventories, as coefficients on the configuration (indices
    before the filler's own stores, which come first), with the outermost yield pair: the
    absorber, and a hopper (a gut) around it. A filler's stores are named ``'gut collector'``
    and ``'gut'`` (its ready store): the hopper's are configuration 0 and 1."""
    gut = (.5, .02)
    inner_pair = fr.wrap_yield(gut, (1., EPS))
    yield {'filler': None, 'twin': None, 'pair': fr.wrap_yield((P, C), (1., EPS)),
           'U': {COLLECTOR: 1., CHAMBER: 0.}, 'V': {CHAMBER: EPS}, 'own': {}}
    # Around the absorber the gut's inventories are its collector and nothing of its ready store
    # (the absorber's pair is (1, EPS)); its net energy still to deliver is EPS per ready gram.
    # Its twin neither separates nor absorbs.
    yield {'filler': fc.hopper_filler(*gut), 'twin': fc.hopper_filler(*gut, speed=0., parameters={'uptake': 0.}),
           'pair': fr.wrap_yield((P, C), inner_pair),
           'U': {COLLECTOR: 1., CHAMBER: 1. - inner_pair[0]}, 'V': {CHAMBER: inner_pair[1]},
           'own': {'U': {0: 1., 1: 0.}, 'V': {1: EPS}}}


def test_the_yield_rule_with_the_motors_payments_apart():
    for fill in _fills():
        world = fc.feeding_with_slot([0.], [10.], filler=fill['filler'], parameters=STILL)
        twin = fc.feeding_with_slot([0.], [10.], filler=fill['twin'],
                                    parameters={**STILL, 'leat': 0., 'uptake': 0.})
        s = world.P.Q_dim - 18                      # the filler's stores come first
        (a, e), own = fill['pair'], fill['own']
        u = {i + s: c for i, c in fill['U'].items()} | own.get('U', {})
        v = {i + s: c for i, c in fill['V'].items()} | own.get('V', {})
        f, g = _field(world), _field(twin)
        taken = 0.
        for _ in range(30):
            q = np.asarray(world.P.position(world.P.init), dtype=float).copy()
            q[[B + s, 1 + s, 2 + s, 3 + s, 12 + s, LATCH + s]] = 3., 1.5, DEFAULTS['x0'], 0., DEFAULTS['x0'], 1.
            q[FOOD + s] = RNG.uniform(.5, 10.)
            for i in (*u, *v):
                q[i] = RNG.uniform(0., 3.)
            q = jnp.asarray(q)
            rate, twin_rate = np.asarray(f(q)), np.asarray(g(q))
            discard = rate[PILE + s] + sum(c * rate[i] for i, c in u.items()) + (1 - a) * rate[FOOD + s]
            energy = rate[B + s] + sum(c * rate[i] for i, c in v.items()) + e * rate[FOOD + s]
            np.testing.assert_allclose(discard, 0., rtol=0, atol=1e-11)
            np.testing.assert_allclose(energy, twin_rate[B + s], rtol=0, atol=1e-11)
            taken -= rate[FOOD + s]
        assert taken > 1.


def test_a_meal_yields_the_pair_per_gram():
    """One full casing at the mouth; Runge-Kutta steps of 0.002 for 20 units of duration."""
    h, steps = .002, 10000
    world = fc.feeding_with_slot([0.], [10.], parameters=STILL, stages=4, h=h)
    twin_field = _field(fc.feeding_with_slot([0.], [10.], parameters={**STILL, 'leat': 0., 'uptake': 0.}))
    dynamics = PhiK(world)
    tick = lambda _, s: dynamics.with_state(s).run_one((EMPTY, trivial_omega(0)), lambda _: (jnp.zeros(2), EMPTY))[1]
    # Between steps the four-round world's state has the one-round world's shape; read it there.
    position = fc.feeding_with_slot([0.], [10.], parameters=STILL).P.position
    paying = lambda s: twin_field(position(s))[B]

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
    assert taken > 9. and q[CHAMBER] < 1e-6 and q[COLLECTOR] < 1e-6
    per_gram = (q[B] - q0[B] + paid) / taken
    np.testing.assert_allclose(per_gram, P * EPS - C, rtol=0, atol=1e-5)
