"""The casing composed from the feeding note's parts is the casing typed out by hand.

``feeding_composed.source`` (integrands, gravity, the air's dragger, the ground's spring and
damper, composed by the library) against ``feeding.source_by_hand`` (one box, its summed field typed
out): same interface, readouts and inclination field at sampled configurations, the same
Euler step, and the same four-round Runge-Kutta run. What runs in IMC: both are library
arrangements; the fields are their own ``omega``; the steps are ``functors.open_step`` and
``pcK.PhiK``.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from arrangements import explore, feeding, feeding_composed
from arrangements.functors import open_step
from arrangements.interpretation import trivial_omega
from arrangements.pcK import PhiK

jax.config.update("jax_enable_x64", True)
EMPTY = jnp.zeros(0)
RNG = np.random.default_rng(1)
# Loads on the six outputs (food, opening, access, height, red, place): food taken, the
# casing pried open, a lift stronger than its weight m g = 2.
LOADS = jnp.array([-0.5, 0.3, 0.0, 3.0, 0.0, 0.7])


def flat(state):
    return np.asarray(ravel_pytree(state)[0])


def test_same_interface_readouts_and_field():
    old, new = feeding.source_by_hand(4., 10.), feeding_composed.source(4., 10.)
    for a in ("in_dim_M", "out_dim_M", "in_dim_N", "out_dim_N"):
        assert getattr(old, a) == getattr(new, a)
    assert old.P.Q_dim == new.P.Q_dim == 5
    np.testing.assert_allclose(flat(new.P.init), flat(old.P.init))
    # Configurations (f, a, x, z, z_g) on both sides of the ground at x_g = -0.4, through both gates.
    for _ in range(40):
        q = jnp.asarray(RNG.normal(size=5) * np.array([3., .5, .3, .3, .05]) + np.array([5., .4, -.4, -.4, 0.]))
        np.testing.assert_allclose(new.out_f(q, EMPTY), old.out_f(q, EMPTY), rtol=0, atol=1e-14)
        np.testing.assert_allclose(new.omega(q, EMPTY, EMPTY)[0], old.omega(q, EMPTY, EMPTY)[0], rtol=0, atol=1e-11)
        pull = lambda arr: jax.vjp(lambda qq: arr.out_f(qq, EMPTY), q)[1](LOADS)[0]
        np.testing.assert_allclose(pull(new), pull(old), rtol=0, atol=1e-13)


def test_same_euler_step():
    old, new = feeding.source_by_hand(-8., 2.), feeding_composed.source(-8., 2.)
    a = open_step(old, EMPTY, LOADS, duration=1e-3)(old.P.init)
    b = open_step(new, EMPTY, LOADS, duration=1e-3)(new.P.init)
    np.testing.assert_allclose(flat(b), flat(a), rtol=0, atol=1e-14)


def test_same_runge_kutta_run():
    """500 steps of 0.004: the casing is lifted off the ground against its weight and the air."""
    inputs = (EMPTY, trivial_omega(0))
    loads = lambda _: (LOADS, EMPTY)
    ends = []
    for build in (feeding.source_by_hand, feeding_composed.source):
        dynamics = PhiK(build(4., 10., stages=4, h=.004))

        @jax.jit
        def run(state):
            return jax.lax.fori_loop(0, 500, lambda _, s: dynamics.with_state(s).run_one(inputs, loads)[1], state)

        ends.append(flat(run(dynamics.state)))
    assert abs(ends[0][2] + 0.4) > 0.5          # it moved: the height left the ground
    np.testing.assert_allclose(ends[1], ends[0], rtol=0, atol=1e-10)


def _bodies(learning, **kw):
    args = dict(parameters=None, stages=1, h=1., learning=learning, memory=None, cart=0.5)
    args.update(kw)
    return feeding._body_by_hand(**args), feeding_composed.body(**args)


def _body_points(old, learning, count):
    """Configurations near the start, with the battery on both sides of its gate, and inputs
    (seen, climb, K, the pulls on grip and tip, unused, eaten) over their ranges."""
    q0 = np.asarray(old.P.position(old.P.init))
    for _ in range(count):
        q = q0 + RNG.normal(size=q0.shape) * 0.3
        q[0] = RNG.uniform(0.1, 6.0)
        n = RNG.normal(size=7) * np.array([.6, .6, .4, .5, .5, 1., 1.]) + np.array([.3, 0., .5, 0., 0., 0., 1.])
        yield jnp.asarray(q), jnp.asarray(n)


# Loads on the body's five outputs (battery, grip, tip, cart, look): the mouth charging the
# battery, the attachment pulling grip and tip, a push on the cart.
BODY_LOADS = jnp.array([0.8, -0.2, 0.4, 0.3, 0.0])


def _same_body(learning, count):
    old, new = _bodies(learning)
    for a in ("in_dim_M", "out_dim_M", "in_dim_N", "out_dim_N"):
        assert getattr(old, a) == getattr(new, a)
    assert old.P.Q_dim == new.P.Q_dim
    np.testing.assert_allclose(flat(new.P.init), flat(old.P.init), rtol=0, atol=1e-15)
    field = lambda arr: jax.jit(lambda q, n: (arr.out_f(q, EMPTY), arr.omega(q, EMPTY, n)[0], arr.omega(q, EMPTY, n)[2],
                                              jax.vjp(lambda qq: arr.out_f(qq, EMPTY), q)[1](BODY_LOADS)[0]))
    f_old, f_new = field(old), field(new)
    for q, n in _body_points(old, learning, count):
        for a, b in zip(f_new(q, n), f_old(q, n)):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-11)


def test_same_body_patrol():
    _same_body(False, 30)


def test_same_body_learning():
    _same_body(True, 30)


def test_same_body_runs():
    """Told a red sighting and a meal, 300 Runge-Kutta steps of 0.004, both receivers."""
    told = jnp.array([.7, .2, 1., 0., 0., 0., 1.5])
    inputs = (EMPTY, trivial_omega(0))
    loads = lambda _: (BODY_LOADS, told)
    for learning in (False, True):
        ends = []
        for arr in _bodies(learning, stages=4, h=.004):
            dynamics = PhiK(arr)

            @jax.jit
            def run(state):
                return jax.lax.fori_loop(0, 300, lambda _, s: dynamics.with_state(s).run_one(inputs, loads)[1], state)

            ends.append(flat(run(dynamics.state)))
        assert np.abs(ends[0] - flat(arr.P.init)).max() > 0.1
        np.testing.assert_allclose(ends[1], ends[0], rtol=0, atol=1e-9)


def test_the_explorer_shows_the_body_parts():
    def leaves(record):
        kids = record["children"]
        return [record["label"]] if not kids else [x for k in kids for x in leaves(k)]

    told = jnp.array([.7, .2, 1., 0., 0., 0., 1.5])
    old, new = _bodies(True)
    assert not explore.inspect(old, inputs=told, loads=BODY_LOADS)["children"]
    record = explore.inspect(new, inputs=told, loads=BODY_LOADS)
    names = leaves(record)
    stores = ["battery", "engagement", "tip", "cart", "memory", "look", "scan", "meal", "usual yield",
              "unexplored", "destination", "clock", "return rate", "eye contact", "last empty", "time away"]
    for name in stores:
        assert names.count(name) == 1, name
    # gripper, lever, drive, two in the eye, nine memory motors: fourteen motors, two pushes
    # each; and the wheel dragger's spring, two more.
    assert sum(n.startswith("push: ") for n in names) == 30
    explore.scene(record)


def test_the_explorer_shows_the_casing_parts():
    def leaves(record):
        kids = record["children"]
        return [record["label"]] if not kids else [x for k in kids for x in leaves(k)]

    record = explore.inspect(feeding_composed.source(4., 10.), loads=LOADS)
    names = leaves(record)
    for name in ("food", "opening", "casing", "memory", "push: weight"):
        assert name in names, name
    assert names.count("memory") == 2            # the air's memory and the ground damper's stroke
    assert sum(n.startswith("push: ") for n in names) == 9
    assert len(explore.inspect(feeding.source_by_hand(4., 10.), loads=LOADS)["children"]) == 0
    assert feeding.source(4., 10.).construction and feeding.receiver().construction[1].construction
    explore.scene(record)
