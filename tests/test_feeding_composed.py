"""Feeding composed from the feeding note's parts is feeding typed out by hand.

``feeding_composed.source`` (integrands, gravity, the air's dragger, the ground's contact,
composed by the library) against ``feeding.source_by_hand`` (one box, its summed field typed
out): same interface, readouts and inclination field at sampled configurations, the same
Euler step, and the same four-round Runge-Kutta run. Likewise the body, the receiver with
its two organs, the spatial coupling, and the whole composite on one casing and on two. What
runs in IMC: both sides are library arrangements; the fields are their own ``omega``; the
steps are ``functors.open_step`` and ``pcK.PhiK``.
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


# ---------------------------------------------------------------------------
# The receiver's organs and the spatial coupling (book P109 (h)): regrouping changes no equation.
# ---------------------------------------------------------------------------


def _close(a, b, rel):
    """Equal to ``rel`` of the larger of 1 and the largest entry."""
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    if a.size:
        np.testing.assert_allclose(a, b, rtol=0, atol=rel * max(1., np.abs(b).max()))


def test_same_receiver():
    """``seeker`` (body, opener and digestor side by side, each organ a ``push``) against
    ``receiver_by_hand`` (each organ one typed-out box): readouts, field, loads on the inputs,
    and the pullback of loads on the seven outputs."""
    loads = jnp.array([0.8, -0.2, 0.4, 0.3, 0.0, 0.5, -0.7])
    for learning, count in ((False, 20), (True, 8)):
        args = dict(parameters=None, stages=1, h=1., learning=learning, memory=None, cart=0.5)
        old, new = feeding.receiver_by_hand(**args), feeding.seeker(**args)
        for a in ("in_dim_M", "out_dim_M", "in_dim_N", "out_dim_N"):
            assert getattr(old, a) == getattr(new, a)
        assert (new.in_dim_N, new.out_dim_N) == (7, 7) and old.P.Q_dim == new.P.Q_dim
        np.testing.assert_allclose(flat(new.P.init), flat(old.P.init), rtol=0, atol=1e-15)
        field = lambda arr: jax.jit(lambda q, n: (arr.out_f(q, EMPTY), arr.omega(q, EMPTY, n)[0], arr.omega(q, EMPTY, n)[2],
                                                  jax.vjp(lambda qq: arr.out_f(qq, EMPTY), q)[1](loads)[0]))
        f_old, f_new = field(old), field(new)
        for q, n in _body_points(feeding._body_by_hand(**args), learning, count):
            for a, b in zip(f_new(q, n), f_old(q, n)):
                np.testing.assert_allclose(a, b, rtol=0, atol=1e-11)
    assert feeding.receiver is feeding.seeker


def _coupling_point(count, held):
    """Values at a spatial coupling: the receiver's seven outputs and each casing's six, near
    a casing at the tip, with gates partly open; then the strokes and the latches. With
    ``held`` the first latch is full and the others empty, every casing within reach."""
    b, grip, tip = RNG.uniform(.1, 6.), RNG.uniform(-1., 2.), RNG.uniform(-.4, 0.)
    cart, look = RNG.normal() * 2, RNG.normal() * 3
    if held:
        grip = RNG.uniform(.5, 2.)
    values = [b, grip, tip, cart, look, 2. * RNG.uniform(), 2. * RNG.uniform()]
    spread = .1 if held else .25
    for _ in range(count):
        food = RNG.uniform(0., .3) + (10. * RNG.uniform() if held or RNG.integers(0, 2) else 0.)
        values += [food, RNG.uniform(0., 1.), RNG.uniform(0., 1.), tip + RNG.normal() * spread,
                   RNG.uniform(0., 1.), cart + RNG.normal() * spread]
    latches = RNG.uniform(-.1, 1.1, size=count)
    if held:
        latches = np.where(np.arange(count) == 0, RNG.uniform(.9, 1.), RNG.uniform(0., .05, size=count))
    return jnp.asarray(np.concatenate([RNG.normal(size=count) * .05, latches])), jnp.asarray(values)


def test_same_spatial_coupling():
    """``spatial_coupling`` (contacts, latches and two pushes beside the identity, under one
    static map) against ``_encounter_by_hand`` (one box, its field typed out), for no casing,
    one, and two with the latch holding the first: interface, readouts, what the receiver is
    told, and the field on the strokes, the latches and every wire.

    The tolerance is 1e-9 of the largest entry. The latch's motor recovers its store's rate as
    ``(tau + P) / C`` with ``C = 1e-4``, which loses digits where the attachment's load ``P`` on
    the latch is large; the typed-out box multiplies by the rate itself.
    """
    for count, held, points in ((0, False, 5), (1, False, 40), (2, False, 40), (2, True, 40)):
        old = feeding._encounter_by_hand(count, feeding.DEFAULTS, 1, 1.)
        new = feeding.spatial_coupling(count)
        for a in ("in_dim_M", "out_dim_M", "in_dim_N", "out_dim_N"):
            assert getattr(old, a) == getattr(new, a)
        assert (new.in_dim_M, new.out_dim_M, new.in_dim_N, new.out_dim_N) == (7, 7 + 6 * count, 0, 2)
        assert old.P.Q_dim == new.P.Q_dim == 2 * count
        np.testing.assert_allclose(flat(new.P.init), flat(old.P.init))
        both = lambda arr: jax.jit(lambda q, m: (arr.out_f(q, m), arr.in_f(q, m, EMPTY), *arr.omega(q, m, EMPTY)[:2]))
        f_old, f_new = both(old), both(new)
        engaged = 0
        for _ in range(points):
            q, m = _coupling_point(count, held)
            a, b = f_new(q, m), f_old(q, m)
            for x, y in zip(a, b):
                _close(x, y, 1e-9)
            if count and np.abs(np.asarray(b[3])[7:]).max() > 1e-3:
                engaged += 1
        assert count == 0 or engaged > points // 2      # most points put a load on a casing
        if held:                                         # the first casing is fed from, the second is not
            assert np.asarray(b[1])[2] > .5 and abs(np.asarray(b[3])[7 + 6]) < 1e-12


def _to(world, state):
    """``state`` in the nesting of ``world``'s own states: the two composites group the same
    stores differently."""
    return ravel_pytree(world.P.init)[1](ravel_pytree(state)[0])


def test_feeding_is_the_same_before_and_after_regrouping():
    """The whole composite, built from ``seeker``, ``source`` and ``spatial_coupling``, against
    the one built from the typed-out receiver organs and coupling: the same Runge-Kutta run on
    one casing, through the grip closing and the latch filling, and then on two, the second
    set down while the latch holds the first. What runs in IMC: both are library composites,
    stepped by ``pcK.PhiK``."""
    params = {'kexp': 0., 'ktau': 0.}                    # as tests/test_explore.py: the cart stays put
    step = dict(parameters=params, stages=4, h=.002)
    new_body, old_body = feeding.seeker(**step), feeding.receiver_by_hand(**step)
    apple = feeding.source(0., 10., **step)
    worlds = {count: (feeding.couple(new_body, [apple] * count, **step),
                      feeding.couple(old_body, [apple] * count, by_hand=True, **step)) for count in (1, 2)}

    def advance(world, state, steps):
        dynamics = PhiK(world)
        tick = lambda _, s: dynamics.with_state(s).run_one((EMPTY, trivial_omega(0)), lambda _: (jnp.zeros(2), EMPTY))[1]
        return jax.jit(lambda s: jax.lax.fori_loop(0, steps, tick, s))(state)

    new, old = worlds[1]
    np.testing.assert_allclose(flat(new.P.init), flat(old.P.init))
    held_new, held_old = advance(new, new.P.init, 500), advance(old, old.P.init, 500)
    assert feeding.unpack_state(held_new, 1)[-1] > .99            # the latch is full
    np.testing.assert_allclose(flat(held_new), flat(held_old), rtol=0, atol=1e-9)

    # Two casings, the latch holding the first: the same state in each world's nesting.
    new, old = worlds[2]
    both, inserted = feeding.add_casing(new_body, [apple], held_new, apple, **step)
    start = feeding.unpack_state(inserted, 2)
    assert start[-9] > .99 and start[-1] == 0.                    # first latch full, second empty
    during_new = advance(new, _to(new, inserted), 500)
    during_old = advance(old, _to(old, inserted), 500)
    out = feeding.unpack_state(during_new, 2)
    assert out[-8] == 10. and out[-1] == 0. and out[-16] < start[-16]   # second untouched, first eaten from
    np.testing.assert_allclose(flat(during_new), flat(during_old), rtol=0, atol=1e-9)


def test_every_field_of_feeding_is_a_push():
    """Every primitive of ``feeding()`` is an integrand, a static map or a ``push``: no
    inclination field is typed out as a whole. (The memory's rates are typed out inside one
    static map, ``feeding._food_memory``, which is a circle and has no field.)"""
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
    assert not feeding._encounter_by_hand(1, feeding.DEFAULTS, 1, 1.).construction


def test_the_explorer_shows_the_coupling_parts():
    def leaves(record):
        kids = record["children"]
        return [record["label"]] if not kids else [x for k in kids for x in leaves(k)]

    q, m = _coupling_point(2, True)
    new = feeding.spatial_coupling(2)
    record = explore.inspect(new, _to(new, q), values=m)
    assert record["label"] == "spatial coupling"
    names = leaves(record)
    for name in ("receiver and casings", "contact latch 1 store", "contact latch 2 store", "attachment 1", "attachment 2",
                 "push: opening", "push: eating"):
        assert names.count(name) == 1, name
    assert names.count("memory") == 2                    # the two attachments' strokes
    # Per contact: spring 2, damper 4, latch motor 2; and the two wide pushes.
    assert sum(n.startswith("push") for n in names) == 2 * 8 + 2
    old = feeding._encounter_by_hand(2, feeding.DEFAULTS, 1, 1.)
    assert not explore.inspect(old, q, values=m)["children"]
    explore.scene(record)


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
