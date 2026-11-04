"""Book Ch 5 ``sec.poise``: leverage, balance points, and what one-way nesting does to them (Claude Opus 5.5, ccc7e1fe,
2026-11-04).

Every system here is an arrangement run by the dynamics functor: each is built with ``compose_seq`` and
``parallel_arrangements``, its output dropped by a wire, and stepped by ``closed_step``, which is ``Phi``'s step checked
against it. What is measured beside the runs is only the largest gap between two streams of readouts.

Predictions, written before the first run:
(a) stiffness: on a bowl a press of 0.1 never gets farther than the press (leverage 1 at most); on a hill it runs off
    to a well whatever the press; a zero stiffness decides nothing: a flat bowl stays, a flat top leaves.
(b) a rim: inside its trough the leverage is at most 1; just past the rim the gap is the distance to the far well.
    A smaller trough has a smaller press that is enough, and the far well hardly moves.
(c) the one-way chain handle -> switch -> switch: nudging the handle by a small press moves the last readout by about
    g^2 times the press, the bound of the proposition; a static map after the readout never increases a jump, and a
    dropped output has no gap at all.
(d) a branching one-way nest, two handles into one two-input switch: the bound holds, and nothing jumps.
(e) the loop of two switches: each alone has one rest for each input value, and wired each to the other, (0,0) is at
    rest, and presses of +-1e-3 end near +-q*, where q* = tanh(g q*).
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact
from arrangements.coalgebra import configuration
from arrangements.functors import closed_step, open_step
from arrangements.rvect import diagonal
from arrangements.wiring import compose_seq, parallel_arrangements, trivial_parameter

jax.config.update("jax_enable_x64", True)

Z = jnp.zeros(0)
DT = 0.05
G = 3.0


def conf(q0, dim=1):
    return configuration(diagonal(jnp.ones(dim)), init=jnp.asarray(q0, dtype=float).reshape(dim))


def drop(dim=1):
    """The wire that drops an output: the system's output closed with zero load."""
    return Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=dim, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m: Z, in_f=lambda q, m, n: Z,
        omega=lambda q, m, n: (Z, jnp.zeros(dim), Z), label="drop",
    )


def cost_system(cost, q0):
    """A system 1 -> <1|R> whose output is its configuration, effort field -d cost, update q + duration * load."""
    return Arrangement(
        P=conf(q0), in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m: q, in_f=lambda q, m, n: Z,
        omega=exact(lambda q, m, n: -cost(q[0])), label="cost",
    )


def handle(h0):
    """A flat part: effort field 0, so it stays where it is put."""
    return cost_system(lambda q: 0.0 * q, h0)


def switch_reader(g, q0, inputs=1):
    """A one-way reader <1|R^inputs> -> <1|R>: load tanh(g * sum of inputs) - q on q, pushback 0."""
    return Arrangement(
        P=conf(q0), in_dim_M=0, out_dim_M=inputs, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m: q, in_f=lambda q, m, n: Z,
        omega=lambda q, m, n: (jnp.tanh(g * jnp.sum(m)) - q, jnp.zeros(inputs), Z), label=f"switch({g:g})",
    )


def switch_box(g, q0):
    """The same switch as a system 1 -> <R|R>, its input to be wired."""
    return Arrangement(
        P=conf(q0), in_dim_M=0, out_dim_M=0, in_dim_N=1, out_dim_N=1,
        out_f=lambda q, m: q, in_f=lambda q, m, n: Z,
        omega=lambda q, m, n: (jnp.tanh(g * n) - q, Z, jnp.zeros(1)), label=f"switch({g:g})",
    )


def loop_wire():
    """Each of two boxes reads the other's output."""
    return Arrangement(
        P=trivial_parameter(), in_dim_M=2, out_dim_M=2, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m: Z, in_f=lambda q, m, n: m[::-1],
        omega=lambda q, m, n: (Z, jnp.zeros(2), Z), label="loop",
    )


def stream(arr, read, n=4000, duration=DT):
    """The readouts of a run of ``arr``: ``read`` takes the state's leaves to the output's value."""
    step = closed_step(arr, check=2, duration=duration)
    s, out = arr.P.init, []
    for _ in range(n):
        out.append(np.asarray(read(jax.tree_util.tree_leaves(s))))
        s = step(s)
    return np.array(out)


def gap(a, b):
    return float(np.max(np.abs(a - b)))


def closed(system):
    return compose_seq(system, drop())


first = lambda leaves: leaves[0]
last = lambda leaves: [x for x in leaves if np.size(x)][-1]  # the drop wire's own state is empty


def press_gap(cost, press, q0=0.0):
    rest = stream(closed(cost_system(cost, q0)), first)
    return gap(stream(closed(cost_system(cost, q0 + press)), first), rest)


def test_stiffness_decides_when_it_is_not_zero():
    bowl = lambda q: q * q / 2
    well = lambda q: (q * q - 1) ** 2 / 4
    assert press_gap(bowl, 0.1) <= 0.1 + 1e-12
    for press in (0.1, 1e-3):
        assert abs(press_gap(well, press) - 1.0) < 1e-2
        assert abs(press_gap(well, -press) - 1.0) < 1e-2


def test_zero_stiffness_decides_nothing():
    assert press_gap(lambda q: q ** 4 / 4, 0.1) <= 0.1 + 1e-12
    assert press_gap(lambda q: 0.0 * q, 0.1) <= 0.1 + 1e-12
    assert abs(press_gap(lambda q: -q ** 4 / 4 + q ** 6 / 6, 0.2) - 1.0) < 1e-2


def rim_cost(a):
    return lambda q: a * q ** 2 / 2 - q ** 4 / 4 + q ** 6 / 96


def test_a_rim_keeps_small_presses_and_a_smaller_trough_needs_a_smaller_press():
    rims = {1.0: 1.035, 0.04: 0.200}
    wells = []
    for a, rim in rims.items():
        cost = rim_cost(a)
        assert press_gap(cost, 0.8 * rim) <= 0.8 * rim + 1e-12
        far = press_gap(cost, 1.1 * rim)
        assert far > 3.5
        wells.append(far)
    assert abs(wells[0] - wells[1]) < 0.2


def chain(h0, g=G):
    src = handle(h0)
    one = compose_seq(src, switch_reader(g, np.tanh(g * 0.0)))
    two = compose_seq(one, switch_reader(g, 0.0))
    return closed(two)


def test_one_way_chain_multiplies_at_small_presses():
    press = 1e-3
    print(chain(0.0).label)
    a, b = stream(chain(0.0), last, n=6000), stream(chain(press), last, n=6000)
    leverage = gap(a, b) / press
    assert leverage <= G * G + 1e-6
    assert abs(leverage - G * G) < 0.05 * G * G
    # the handle's own leverage is 1, and the first switch alone has about g
    one = closed(compose_seq(handle(press), switch_reader(G, 0.0)))
    base = closed(compose_seq(handle(0.0), switch_reader(G, 0.0)))
    assert abs(gap(stream(one, last), stream(base, last)) / press - G) < 0.05 * G
    # no jump: the gap shrinks with the press
    small = closed(compose_seq(compose_seq(handle(1e-5), switch_reader(G, 0.0)), switch_reader(G, 0.0)))
    assert gap(stream(chain(0.0), last, n=6000), stream(small, last, n=6000)) < 2e-4


def test_a_static_map_after_the_readout_cannot_raise_a_jump_and_a_drop_has_no_gap():
    well = lambda q: (q * q - 1) ** 2 / 4
    a = stream(closed(cost_system(well, 0.0)), first)
    b = stream(closed(cost_system(well, 1e-3)), first)
    half = lambda x: 0.5 * x
    assert gap(half(a), half(b)) <= gap(a, b)
    assert gap(0.0 * a, 0.0 * b) == 0.0


def test_branching_one_way_nest_keeps_the_bound():
    press = 1e-3
    def nest(h1, h2):
        srcs = parallel_arrangements(handle(h1), handle(h2))
        return closed(compose_seq(srcs, switch_reader(G, 0.0, inputs=2)))
    base = stream(nest(0.0, 0.0), last)
    # press both handles: the input moves by 2 * press in the sum, so at most 2 g times the press
    both = stream(nest(press, press), last)
    assert gap(base, both) / press <= 2 * G + 1e-6
    assert abs(gap(base, both) / press - 2 * G) < 0.05 * 2 * G
    # press one handle only
    assert abs(gap(base, stream(nest(press, 0.0), last)) / press - G) < 0.05 * G


def q_star(g=G):
    q = 1.0
    for _ in range(200):
        q = float(np.tanh(g * q))
    return q


def test_one_switch_alone_has_one_rest_for_each_input():
    for u in (-0.01, 0.0, 0.01):
        ends = []
        for q0 in (-2.0, 0.0, 2.0):
            sw = switch_box(G, q0)
            step = open_step(sw, [u], None, check=2, duration=DT)
            s = sw.P.init
            for _ in range(2000):
                s = step(s)
            ends.append(float(s[0]))
        assert np.allclose(ends, np.tanh(G * u), atol=1e-6)


def test_the_loop_of_two_switches_has_a_balance_point():
    def loop(a, b):
        return compose_seq(parallel_arrangements(switch_box(G, a), switch_box(G, b)), loop_wire())
    read = lambda leaves: np.concatenate([np.ravel(x) for x in leaves])
    print(loop(0.0, 0.0).label)
    rest = stream(loop(0.0, 0.0), read)
    assert np.allclose(rest, 0.0)
    qs = q_star()
    for press in (1e-1, 1e-3, 1e-6):
        plus = stream(loop(press, press), read, n=8000)
        minus = stream(loop(-press, -press), read, n=8000)
        assert np.allclose(plus[-1], qs, atol=1e-3)
        assert np.allclose(minus[-1], -qs, atol=1e-3)
        assert gap(plus, np.zeros_like(plus)) >= qs - 1e-3
