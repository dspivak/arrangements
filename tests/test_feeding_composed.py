"""Feeding composed from the feeding note's parts is feeding typed out by hand.

``feeding_composed.source`` (the casing without a hinge: integrands, gravity, the air's dragger,
the ground's contact, composed by the library) against ``feeding.source_by_hand`` (one box, its
summed field typed out): same interface, readouts and inclination field at sampled
configurations, the same Euler step, and the same four-round Runge-Kutta run. Likewise the body
and its motors against ``feeding._body_by_hand``. What runs in IMC: both sides are library
arrangements; the fields are their own ``omega``; the steps are ``functors.open_step`` and
``pcK.PhiK``. (Until 2026-11-23 the hinged casing, the receiver with its opener and digestor
and their spatial coupling were checked here too against their typed-out boxes; that model left
the library then, book P109 (i) act (5).)
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
# Loads on the four outputs (food, height, red, place): food taken, a lift stronger than its
# weight m g = 2.
LOADS = jnp.array([-0.5, 3.0, 0.0, 0.7])


def flat(state):
    return np.asarray(ravel_pytree(state)[0])


def _unlidded(*a, **kw):
    """The casing as it was before P109 (k), without its lid and speed: the one ``source_by_hand``
    types out."""
    return feeding_composed.source(*a, lid=False, **kw)


def test_same_interface_readouts_and_field():
    old, new = feeding.source_by_hand(4., 10.), _unlidded(4., 10.)
    for a in ("in_dim_M", "out_dim_M", "in_dim_N", "out_dim_N"):
        assert getattr(old, a) == getattr(new, a)
    assert old.P.Q_dim == new.P.Q_dim == 4 and new.out_dim_N == 4
    np.testing.assert_allclose(flat(new.P.init), flat(old.P.init))
    # Configurations (f, x, z, z_g) on both sides of the ground at x_g = -0.4, through both gates.
    for _ in range(40):
        q = jnp.asarray(RNG.normal(size=4) * np.array([3., .3, .3, .05]) + np.array([5., -.4, -.4, 0.]))
        np.testing.assert_allclose(new.out_f(q, EMPTY), old.out_f(q, EMPTY), rtol=0, atol=1e-14)
        np.testing.assert_allclose(new.omega(q, EMPTY, EMPTY)[0], old.omega(q, EMPTY, EMPTY)[0], rtol=0, atol=1e-11)
        pull = lambda arr: jax.vjp(lambda qq: arr.out_f(qq, EMPTY), q)[1](LOADS)[0]
        np.testing.assert_allclose(pull(new), pull(old), rtol=0, atol=1e-13)


def test_the_lid_adds_a_store_and_two_readouts():
    """The casing since P109 (k), ``1 -> <0|6>``: configuration (f, lid, x, z_air, z_g). With no
    load on the lid and the speed, its field is the lidless casing's on (f, x, z_air, z_g) and
    the lid is still; its fifth output is the lid, its sixth the air dragger's stretch times
    kappa / beta, which is the casing's speed once the air's memory has caught up."""
    old, new = _unlidded(4., 10.), feeding_composed.source(4., 10.)
    assert (new.in_dim_N, new.out_dim_N, new.P.Q_dim) == (0, 6, 5)
    p = feeding.DEFAULTS
    keep = jnp.array([0, 2, 3, 4])
    for _ in range(20):
        q = jnp.asarray(RNG.normal(size=5) * np.array([3., .3, .3, .3, .05]) + np.array([5., .3, -.4, -.4, 0.]))
        out = new.out_f(q, EMPTY)
        np.testing.assert_allclose(out[:4], old.out_f(q[keep], EMPTY), rtol=0, atol=1e-14)
        np.testing.assert_allclose(out[4:], [q[1], p['kappa'] * (q[2] - q[3]) / p['beta']], rtol=0, atol=1e-12)
        loads = jnp.concatenate([LOADS, jnp.zeros(2)])
        rate = new.omega(q, EMPTY, loads)[0]
        assert rate[1] == 0.
        np.testing.assert_allclose(rate[keep], old.omega(q[keep], EMPTY, LOADS)[0], rtol=0, atol=1e-11)


def test_same_euler_step():
    old, new = feeding.source_by_hand(-8., 2.), _unlidded(-8., 2.)
    a = open_step(old, EMPTY, LOADS, duration=1e-3)(old.P.init)
    b = open_step(new, EMPTY, LOADS, duration=1e-3)(new.P.init)
    np.testing.assert_allclose(flat(b), flat(a), rtol=0, atol=1e-14)


def test_same_runge_kutta_run():
    """500 steps of 0.004: the casing is lifted off the ground against its weight and the air."""
    inputs = (EMPTY, trivial_omega(0))
    loads = lambda _: (LOADS, EMPTY)
    ends = []
    for build in (feeding.source_by_hand, _unlidded):
        dynamics = PhiK(build(4., 10., stages=4, h=.004))

        @jax.jit
        def run(state):
            return jax.lax.fori_loop(0, 500, lambda _, s: dynamics.with_state(s).run_one(inputs, loads)[1], state)

        ends.append(flat(run(dynamics.state)))
    assert abs(ends[0][1] + 0.4) > 0.5          # it moved: the height left the ground
    np.testing.assert_allclose(ends[1], ends[0], rtol=0, atol=1e-10)


def test_the_hinged_casing_has_left():
    try:
        feeding_composed.source(4., 10., hinge=True)
    except ValueError as error:
        assert "left the library" in str(error)
    else:
        raise AssertionError("a hinged casing was built")
    for name in ("seeker", "receiver", "receiver_by_hand", "spatial_coupling", "_encounter_by_hand", "couple",
                 "add_casing", "pack_state", "unpack_state"):
        assert not hasattr(feeding, name), name
    for name in ("seeker", "mouth_organ", "spatial_coupling"):
        assert not hasattr(feeding_composed, name), name


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


def test_every_field_of_feeding_is_a_push():
    """Every primitive of ``feeding()`` is an integrand, a static map or a ``push``: no
    inclination field is typed out as a whole. (The memory's rates are nine circles, one per
    store, each in its learner's box; a circle has no field.)"""
    from arrangements.arrangement import zero_form

    def primitives(arr):
        if arr.construction:
            _, a, b = arr.construction
            yield from primitives(a)
            yield from primitives(b)
        else:
            yield arr

    for learning in (False, True):
        parts = list(primitives(feeding.feeding(learning=learning)))
        typed = [a.label for a in parts if a.omega is not zero_form and not a.label.startswith("push")]
        assert not typed, typed
        assert sum(a.label.startswith("push") for a in parts) > 40
    # The checks themselves are single boxes with typed-out fields.
    assert not feeding.source_by_hand(1., 1.).construction


def _leaves(record):
    kids = record["children"]
    return [record["label"]] if not kids else [x for k in kids for x in _leaves(k)]


def test_the_explorer_shows_the_coupling_parts():
    """``mouth_coupling(2)``: per casing a contact and a latch, and the one wide push that takes."""
    new = feeding_composed.mouth_coupling(2)
    values = jnp.asarray(np.concatenate([[3., 1.5, -.2, 0., .4, 1., 1.],
                                         [10., -.2, 1., 0., .8, 0., 8., -.4, 0., .2, 0., .3], [0., 1.]]))
    record = explore.inspect(new, values=values)
    assert record["label"] == "spatial coupling"
    names = _leaves(record)
    for name in ("seeker, casings and pile", "contact latch 1 store", "contact latch 2 store", "attachment 1",
                 "attachment 2", "push: taking", "push: spilling", "hands battery"):
        assert names.count(name) == 1, name
    assert names.count("memory") == 2                    # the two attachments' strokes
    # Per contact: spring 2, damper 4, latch motor 2; the wide push that takes; since P109 (k)
    # per casing a lid motor 2, the wide push that spills, and the hands' cord, a spring, 2.
    assert sum(n.startswith("push") for n in names) == 2 * 8 + 1 + 2 * 2 + 1 + 2
    explore.scene(record)


def test_the_explorer_shows_the_body_parts():
    told = jnp.array([.7, .2, 1., 0., 0., 0., 1.5])
    old, new = _bodies(True)
    assert not explore.inspect(old, inputs=told, loads=BODY_LOADS)["children"]
    record = explore.inspect(new, inputs=told, loads=BODY_LOADS)
    names = _leaves(record)
    stores = ["battery", "engagement", "tip", "cart", "memory", "look", "scan", "meal", "usual yield",
              "unexplored", "destination", "clock", "return rate", "eye contact", "last empty", "time away"]
    for name in stores:
        assert names.count(name) == 1, name
    # gripper, lever, drive, two in the eye, nine memory motors: fourteen motors, two pushes
    # each; and the wheel dragger's spring, two more.
    assert sum(n.startswith("push: ") for n in names) == 30
    explore.scene(record)


def test_the_explorer_shows_the_casing_parts():
    record = explore.inspect(feeding_composed.source(4., 10.), loads=jnp.concatenate([LOADS, jnp.zeros(2)]))
    names = _leaves(record)
    for name in ("food", "lid", "casing", "memory", "push: weight"):
        assert name in names, name
    assert "opening" not in names
    assert names.count("memory") == 2            # the air's memory and the ground damper's stroke
    assert sum(n.startswith("push: ") for n in names) == 9
    assert len(explore.inspect(feeding.source_by_hand(4., 10.), loads=LOADS)["children"]) == 0
    assert feeding.source(4., 10.).construction and feeding.feeding().construction[1].construction
    explore.scene(record)
