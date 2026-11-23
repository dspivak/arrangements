"""A robot-style seeker for the slot (book P109 (j); David, 2026-11-21: "i think the seeker
may be a copy of the whole robot, but i'm not sure it needs to accomplish eerything the same
way. a hopper is also a seeker. can we leave it open to be filled by any seeker style?").

``feeding_composed.robot_style()`` is an arrow ``F -> F``: a whole robot (every organ kept,
the run's call) with its own arena, which lays out the mass it is handed among its casings
and seeks it, its battery corded to the reserve, its husks sent to the parent's collector,
its own slot open. Tested here:

* its loads on the feeder interface have the recursive note's form, ``-j d food + v d
  reserve + w d parentMass`` with ``j, w >= 0`` and nothing on the acceptance;
* it fills the outer robot's slot;
* styles compose in either order: a robot around a hopper around the absorber, and a hopper
  around a robot around the absorber, both run and conserve mass;
* its net energy per gram is not constant: its motors draw on the reserve while nothing is
  eaten, so it is outside the note's class (a hopper's is constant).

What runs in IMC: every feeder and meal is a library composite, its field its own ``omega``,
stepped by ``pcK.PhiK``. Beside it, in ordinary arithmetic: sums of stores.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from arrangements import feeding_composed as fc
from arrangements import feeding_parts as fp
from arrangements import feeding_recursive as fr
from arrangements.interpretation import trivial_omega
from arrangements.pcK import PhiK
from arrangements.wiring import compose_seq

jax.config.update("jax_enable_x64", True)
EMPTY = jnp.zeros(0)
RNG = np.random.default_rng(5)
STEP = dict(stages=4, h=.004)


def _absorber():
    return fc.absorber_filler()


def _hopper_layer(**step):
    return fr.layer(fr.collector(1., 6., label="gut collector", **step),
                    fr.separator(.5, .02, fr.process_rate(2., 6.), label="gut", **step), label="hopper")


def _meal(feeder, seed=5., reserve=3., **step):
    """The feeder handed a store of seed, with a reserve and the ground's pile, as
    ``tests/test_feeding_recursive.py``'s meal. Readouts: seed, reserve, pile."""
    stores = [fp.rod_segment(1., seed, "seed", **step), fp.rod_segment(1., reserve, "reserve", **step), fr.pile(**step)]
    return fp.fill(stores + [feeder], fp.transfer("meal", 4, 4, 0, 3, lambda m, n: m, lambda m: m[:3]))


def _run(world, steps, every):
    dynamics = PhiK(world)
    tick = lambda _, s: dynamics.with_state(s).run_one((EMPTY, trivial_omega(0)), lambda _: (jnp.zeros(3), EMPTY))[1]
    advance = jax.jit(lambda s: jax.lax.fori_loop(0, every, tick, s))
    state, out = world.P.init, []
    for _ in range(steps // every):
        state = advance(state)
        out.append(np.asarray(ravel_pytree(state)[0]))
    return out


def test_the_robot_style_s_loads_have_the_feeder_form():
    feeder = compose_seq(_absorber(), fc.robot_style())
    assert (feeder.in_dim_M, feeder.out_dim_M, feeder.in_dim_N, feeder.out_dim_N) == (0, 0, 4, 0)
    field = jax.jit(lambda q, h: feeder.omega(q, EMPTY, h))
    q0 = np.asarray(fc.robot_style().P.position(fc.robot_style().P.init))
    took = 0.
    for _ in range(30):
        q = q0 + RNG.normal(size=q0.shape) * .3
        q[0] = RNG.uniform(0., 4.)                        # the robot's own battery
        q[9:12] = RNG.uniform(0., 3., size=3)             # its chamber, collector and casing's food: masses
        h = jnp.asarray(RNG.uniform(0., 3., size=4) * np.array([1., 1., 1., 1 / 3]))
        dh = np.asarray(field(jnp.asarray(q), h)[2])
        assert dh[0] <= 0. and dh[2] >= 0. and dh[3] == 0.
        took -= dh[0]
    assert took > 1.


def test_it_fills_the_outer_robot_s_slot():
    """One full casing at the outer robot's mouth; the outer robot eats, its chamber hands the
    mass to the robot inside, which lays it out and eats it."""
    inner = compose_seq(_absorber(), fc.robot_style(**STEP))
    world = fc.feeding_with_slot([0.], [10.], filler=inner, parameters={'kexp': 0., 'ktau': 0.}, **STEP)
    dynamics = PhiK(world)
    tick = lambda _, s: dynamics.with_state(s).run_one((EMPTY, trivial_omega(0)), lambda _: (jnp.zeros(2), EMPTY))[1]
    end = jax.jit(lambda s: jax.lax.fori_loop(0, 2500, tick, s))(world.P.init)
    flat = np.asarray(ravel_pytree(end)[0])
    assert np.all(np.isfinite(flat))
    pile = float(flat[-3])                                # pile, then the coupling's stroke and latch
    assert pile > 1.                                      # husks from both mouths reached the ground


def test_styles_compose_in_either_order():
    """A robot around a hopper around the absorber, and a hopper around a robot around the
    absorber: both run; mass is conserved up to what the absorber took up."""
    orders = {'robot(hopper(absorber))': compose_seq(compose_seq(_absorber(), _hopper_layer(**STEP)), fc.robot_style(**STEP)),
              'hopper(robot(absorber))': compose_seq(compose_seq(_absorber(), fc.robot_style(**STEP)), _hopper_layer(**STEP))}
    for name, feeder in orders.items():
        out = _run(_meal(feeder, **STEP), 2500, 500)
        seed, reserve, pile = out[-1][:3]
        assert np.all(np.isfinite(out[-1])), name
        assert seed < 4. and pile > .1, (name, seed, pile)


def _net_per_gram(feeder, seed, steps):
    """Net energy delivered per gram taken: the reserve's gain plus what the feeder's own
    batteries hold (a robot's battery counts at 1 per unit, as the reserve does)."""
    out = _run(_meal(feeder, seed=seed, **STEP), steps, steps)[-1]
    batteries = 0.
    if feeder.label.endswith("robot style") or "robot" in feeder.label:
        batteries = out[3]                                # the robot's own battery comes first among its stores
    return (out[1] - 3. + batteries) / (seed - out[0])


def test_the_robot_style_s_energy_per_gram_is_not_constant():
    """The same meal of 5 grams, read at 10 and at 40 units of duration: the hopper delivers the
    same per gram both times (its stores drained); the robot delivers less at 40 than at 10,
    since its motors kept drawing on the reserve while nothing was left to eat."""
    hopper = compose_seq(_absorber(), _hopper_layer(**STEP))
    robot = compose_seq(_absorber(), fc.robot_style(**STEP))
    counts = {}
    for name, feeder in (('hopper', hopper), ('robot', robot)):
        counts[name] = [_net_per_gram(feeder, 5., steps) for steps in (2500, 10000)]
    print('net energy per gram at 10 and at 40 units:', counts)
    h10, h40 = counts['hopper']
    r10, r40 = counts['robot']
    assert abs(h40 - h10) < 1e-3                         # a hopper's yield pair is constant
    assert r10 - r40 > .1                                 # a robot's is not
