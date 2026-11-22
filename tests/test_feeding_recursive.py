"""The recursive feeding note's class of feeders, tested on library arrangements.

The note is the book repo's ``background/feeding-imc/recursive-feeding.tex``. A feeder
``b : 1 -> F`` is in its class when the load it pushes back on its input (food, reserve,
parent mass, parent acceptance) is ``-j d food + v d reserve + w d parentMass`` with
``j, w >= 0`` and nothing on the acceptance, and two functions of its state, ``U`` (discard
still to export) and ``V`` (net energy still to deliver), move at ``(1 - a) j - w`` and
``e j - v`` for a yield pair ``(a, e)``.

Tested here: the absorber (a nullary operation), one layer around it, and the ternary
wiring filled with a collector, a separator and a feeder (three inputs), against the unary
route to the same morphism; the yields at the note's numbers; and the note's two
propositions on a separator whose rate is set from outside.

What runs in IMC: every feeder is a library composite and the fields are its own ``omega``;
the meal is that composite beside three stores, stepped by ``pcK.PhiK``. Beside it, in
ordinary arithmetic: the inventories ``U`` and ``V``, which are linear in the stores.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from arrangements import feeding_parts as fp
from arrangements import feeding_recursive as fr
from arrangements.interpretation import trivial_omega
from arrangements.pcK import PhiK
from arrangements.wiring import compose_seq, tensor_arrangements

jax.config.update("jax_enable_x64", True)
EMPTY = jnp.zeros(0)
RNG = np.random.default_rng(7)

EPSILON, UPTAKE = 4., 2.
GUT, MOUTH = (.5, .2), (.8, .1)          # the note's numbers: (pass fraction, cost per gram)


def a_layer(settings, speed, clearance, capacity=6., **step):
    p, c = settings
    return fr.layer(fr.collector(clearance, capacity, **step),
                    fr.separator(p, c, fr.process_rate(speed, capacity), **step))


def inventories(stack, pair=(1., EPSILON)):
    """The note's ``U`` and ``V`` for layers wrapped innermost first around the absorber, as
    coefficients on the stores ``(m_1, z_1, m_2, z_2, ...)``, and the outermost yield pair:
    ``U' = U + m + (1 - a) z`` and ``V' = V + e z`` with ``(a, e)`` the pair one layer in."""
    u, v = [], []
    for settings in stack:
        a, e = pair
        u += [1., 1. - a]
        v += [0., e]
        pair = fr.wrap_yield(settings, pair)
    return np.array(u), np.array(v), pair


def passes(feeder, u, v, pair, points=40, order=None):
    """The note's membership test at sampled physical states and inputs."""
    a, e = pair
    n = feeder.P.Q_dim
    field = jax.jit(lambda q, h: feeder.omega(q, EMPTY, h))
    took = 0.
    for _ in range(points):
        q = jnp.asarray(RNG.uniform(0., 5., size=n))
        h = jnp.asarray(RNG.uniform(0., 3., size=4) * np.array([1., 1., 1., 1 / 3]))
        dq, _, dh = field(q, h)
        rate = np.asarray(dq) if order is None else np.asarray(dq)[order]   # unit capacities: each store moves at its load
        j, energy, w = -dh[0], dh[1], dh[2]
        assert j >= 0 and w >= 0 and dh[3] == 0.
        np.testing.assert_allclose(u @ rate, (1 - a) * j - w, rtol=0, atol=1e-12)
        np.testing.assert_allclose(v @ rate, e * j - energy, rtol=0, atol=1e-12)
        took += float(j)
    return took


def test_the_absorber_is_in_the_class():
    """The nullary case: yield pair ``(1, epsilon)``, both inventories zero."""
    absorb = fr.absorber(EPSILON, UPTAKE)
    assert (absorb.in_dim_M, absorb.out_dim_M, absorb.in_dim_N, absorb.out_dim_N) == (0, 0, 4, 0)
    assert passes(absorb, np.zeros(0), np.zeros(0), (1., EPSILON)) > 1.
    h = jnp.array([1.5, .7, 2., .4])
    np.testing.assert_allclose(absorb.omega(EMPTY, EMPTY, h)[2], [-UPTAKE * 1.5, EPSILON * UPTAKE * 1.5, 0., 0.])


def test_one_layer_keeps_the_class():
    feeder = fr.wrap(a_layer(GUT, 1.5, .7), fr.absorber(EPSILON, UPTAKE))
    assert (feeder.in_dim_N, feeder.out_dim_N, feeder.P.Q_dim) == (4, 0, 2)
    u, v, pair = inventories([GUT])
    np.testing.assert_allclose(pair, (.5, 1.8))
    assert passes(feeder, u, v, pair) > 1.


def test_the_ternary_wiring_filled_with_three_keeps_the_class():
    """``W . (collector (x) separator (x) feeder)`` with the feeder a gut around the absorber:
    three filled inputs. It is the same morphism as the mouth's layer after that feeder, with
    its stores in another order."""
    inner = fr.wrap(a_layer(GUT, 1.5, .7), fr.absorber(EPSILON, UPTAKE))
    ternary = fr.feed(fr.collector(.4, 6.), fr.separator(*MOUTH, fr.process_rate(1.2, 6.)), inner)
    unary = fr.wrap(a_layer(MOUTH, 1.2, .4), inner)
    assert (ternary.in_dim_N, ternary.out_dim_N, ternary.P.Q_dim) == (4, 0, 4)
    u, v, pair = inventories([GUT, MOUTH])
    np.testing.assert_allclose(pair, (.4, 1.34))
    order = np.array([2, 3, 0, 1])                       # ternary stores (m_2, z_2, m_1, z_1)
    assert passes(ternary, u, v, pair, order=order) > 1.
    assert passes(unary, u, v, pair) > 1.
    for _ in range(10):
        q = jnp.asarray(RNG.uniform(0., 5., size=4))
        h = jnp.asarray(RNG.uniform(0., 2., size=4))
        a, b = ternary.omega(q[order], EMPTY, h), unary.omega(q, EMPTY, h)
        np.testing.assert_allclose(np.asarray(a[0])[order], b[0], rtol=0, atol=1e-13)
        np.testing.assert_allclose(a[2], b[2], rtol=0, atol=1e-13)


def test_yields_at_the_notes_numbers_over_a_meal():
    """A gut inside a mouth around the absorber, fed one gram of whole seed from a store
    with a unit reserve and the ground's pile: per gram taken, 0.4 grams absorbed, so 0.6 on
    the pile, and 1.34 of net energy. The books balance at every time with the inventories
    counted; once the stores have drained, they balance without them."""
    step = dict(stages=4, h=.01)
    feeder = fr.wrap(a_layer(MOUTH, 2., 1., **step),
                     fr.wrap(a_layer(GUT, 2., 1., **step), fr.absorber(EPSILON, UPTAKE)))
    stores = [fp.rod_segment(1., 1., "seed", **step), fp.rod_segment(1., 1., "reserve", **step), fr.pile(**step)]
    world = fp.fill(stores + [feeder],
                    fp.transfer("meal", 4, 4, 0, 3, lambda m, n: m, lambda m: m[:3]))
    u, v, (a, e) = inventories([GUT, MOUTH])
    assert (a, e) == fr.wrap_yield(MOUTH, fr.wrap_yield(GUT, (1., EPSILON)))
    dynamics = PhiK(world)
    tick = lambda _, s: dynamics.with_state(s).run_one((EMPTY, trivial_omega(0)), lambda _: (jnp.zeros(3), EMPTY))[1]
    advance = jax.jit(lambda s, steps: jax.lax.fori_loop(0, steps, tick, s))
    start = world.P.init
    read = lambda s: np.asarray(ravel_pytree(s)[0])     # seed, reserve, pile, then (m, z) gut, (m, z) mouth

    def books(state):
        seed, reserve, heap, *inside = read(state)
        taken = 1. - seed
        return taken, reserve - 1. + v @ np.array(inside), heap + u @ np.array(inside)

    part = advance(start, 60)                            # mid-meal: stores full, the books still balance
    taken, energy, discard = books(part)
    assert .2 < taken < .95 and max(read(part)[3:]) > .02
    np.testing.assert_allclose([energy, discard], [e * taken, (1 - a) * taken], rtol=0, atol=1e-8)
    done = advance(part, 5940)                           # 60 units of duration in all
    taken, energy, discard = books(done)
    assert taken > .99 and max(read(done)[3:]) < 1e-6    # the seed is eaten and every store has drained
    np.testing.assert_allclose([energy, discard], [e * taken, (1 - a) * taken], rtol=0, atol=1e-8)
    np.testing.assert_allclose([(read(done)[1] - 1.) / taken, read(done)[2] / taken], [1.34, .6], rtol=0, atol=1e-5)


def test_settle_after_reach_is_the_separator():
    """The note's Proposition 1 at one food wire, no gate: the same interface, readout and field."""
    law = fr.process_rate(1.3, 5.)
    whole = fr.separator(.8, .1, law, ready=.5)
    halves = compose_seq(fr.settle(.8, .1, ready=.5), fr.reach(lambda values: jnp.ones(1), law))
    for arr in (whole, halves):
        assert (arr.in_dim_M, arr.out_dim_M, arr.in_dim_N, arr.out_dim_N, arr.P.Q_dim) == (0, 0, 4, 1, 1)
    np.testing.assert_allclose(ravel_pytree(halves.P.init)[0], ravel_pytree(whole.P.init)[0])
    for _ in range(30):
        q, n = jnp.asarray(RNG.uniform(0., 5., size=1)), jnp.asarray(RNG.uniform(0., 3., size=4))
        np.testing.assert_allclose(halves.out_f(q, EMPTY), whole.out_f(q, EMPTY))
        for x, y in zip(halves.omega(q, EMPTY, n), whole.omega(q, EMPTY, n)):
            np.testing.assert_allclose(x, y, rtol=0, atol=1e-13)


def test_a_gated_rate_keeps_the_class():
    """The note's Proposition 2 with two food wires and one gate value, which sends the mouth
    to the first wire, the second, or neither: nothing comes back on the gate's value, each
    food wire loses its own rate, and the two inventories move by the total."""
    law = fr.process_rate(1.2, 6.)
    gate = lambda values: jnp.stack([jnp.cos(values[0]) ** 2, jnp.sin(values[0]) ** 2]) * jnp.exp(-values[0] ** 2 / 9)
    mouth = compose_seq(fr.settle(*MOUTH), fr.reach(gate, law, count=2, further=1))
    inner = fr.wrap(a_layer(GUT, 1.5, .7), fr.absorber(EPSILON, UPTAKE))
    feeder = compose_seq(tensor_arrangements([fr.collector(.4, 6.), mouth, inner]), fr.gated_wiring(2, 1))
    assert (feeder.in_dim_N, feeder.out_dim_N, feeder.P.Q_dim) == (6, 0, 4)
    u, v, (a, e) = inventories([GUT, MOUTH])
    order = np.array([2, 3, 0, 1])                       # stores (m_2, z_2, m_1, z_1)
    field = jax.jit(lambda q, h: feeder.omega(q, EMPTY, h))
    shares = []
    for _ in range(40):
        q = jnp.asarray(RNG.uniform(0., 5., size=4))
        h = jnp.asarray(RNG.uniform(0., 3., size=6) * np.array([1., 1., 1., 1., 1., 1 / 3]))
        dq, _, dh = field(q, h)
        rate = np.asarray(dq)[order]
        j1, j2, energy, w = -dh[1], -dh[2], dh[3], dh[4]
        local = jnp.stack([q[0], fr.admit(6.)(q[0])])
        expected = gate(h[:1]) * law(q[1], h[1:3], h[3], *local)
        np.testing.assert_allclose([j1, j2], expected, rtol=0, atol=1e-13)
        assert dh[0] == 0. and dh[5] == 0. and j1 >= 0 and j2 >= 0 and w >= 0
        np.testing.assert_allclose(u @ rate, (1 - a) * (j1 + j2) - w, rtol=0, atol=1e-12)
        np.testing.assert_allclose(v @ rate, e * (j1 + j2) - energy, rtol=0, atol=1e-12)
        shares.append(float(j1 / (j1 + j2)))
    assert min(shares) < .2 and max(shares) > .8         # the gate sent the mouth to each wire
