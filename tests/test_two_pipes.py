"""The two other pipes of the book's Chapter 2: the compressible-fluid pipe (``ex.gas_pipe``, with ``exc.gas_pipe_halves`` and
``exc.gas_pipe_rod``) and the inertial pipe (``ex.pipe_momentum``, with ``exc.pipe_momentum_settles``).

For a reader of the book. ``tests/test_two_tanks.py`` has the tank and the resistance-only pipe; this file adds what those two
examples add, built as the book draws them, and runs the book's own numbers.

* **The compressible-fluid pipe.** The gas in the pipe is a third tank, and each half of the pipe is a ``joint``: a part without
  state between two tanks, with a flow law ``I(q_a, q_b)``, ``I(q, q) = 0``, returning ``-I`` to its first end and ``+I`` to its
  second. ``gas_pipe`` wires tank, pipe's gas, tank through two joints, the pipe's gas reading out its pressure to both
  (``wiring.boxes_and_couplers``, the split wire of the book's picture). A gas tank's ``A`` is the gas it takes in per unit of
  pressure.
* **The inertial pipe.** ``inertial_pipe(m, kappa)`` is a part with a state, the momentum ``p`` of its water, and two input ports
  receiving the pressures. Its response is ``1``; its internal inclination is ``q_1 - q_2 - p / (m kappa)``; it returns ``-p/m`` to
  the first tank and ``+p/m`` to the second. It has no output port.

What the tests check: the gas pipe with the linear flow law is the rod of ``sec.heat``, step for step (``exc.gas_pipe_rod`` (a),
(b)); with the pipe's gas halfway between the tanks' pressures it takes in what it lets out and passes what a resistance-only pipe of
half the conductance does (``exc.gas_pipe_halves``); otherwise it gains the difference; any flow law keeps the total
(``exc.gas_pipe_rod`` (c)); the inertial pipe's five steps printed in ``ex.pipe_momentum``, the second level passing the first; and at
the momentum ``m kappa (q_1 - q_2)`` it passes the resistance-only pipe's flow and its momentum stays (``exc.pipe_momentum_settles``).

Every test runs a composed arrangement through ``Phi``. Everything is on ``R^n``. Written for the book's plan §7 item 82 at David's
"please do 3" (Claude Fable 5.2, session 2524b69e, 2026-11-11).
"""

import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement
from arrangements.coalgebra import configuration
from arrangements.functors import Phi
from arrangements.rvect import diagonal
from arrangements.wiring import boxes_and_couplers, compose_seq, tensor_arrangements

from tests.chain import IN_POS, book_chain, closed_run, flat_state, one_port_box
from tests.test_two_tanks import closed, coupler, cylinder, pipe

_z = jnp.zeros(0)


def joint(flow_law, label="joint"):
    """Half of a compressible-fluid pipe (``ex.gas_pipe``): no state; it passes the flow ``I(q_a, q_b)`` from its end at the pressure
    ``q_a`` to its end at ``q_b``, so it returns ``-I`` to the first and ``+I`` to the second."""
    return coupler(lambda e: jnp.array([-flow_law(e[0], e[1]), flow_law(e[0], e[1])]), label)


def linear(kappa):
    """The linear flow law ``I(q_a, q_b) = kappa (q_a - q_b)``, the water pipe's."""
    return lambda a, b: kappa * (a - b)


def gas_pipe(A, s, flow_law):
    """Two tanks of gas joined by a pipe full of the same gas (``ex.gas_pipe``): ``A = (A_1, A, A_2)`` and the amounts
    ``s = (s_1, s, s_2)`` for the first tank, the gas in the pipe, and the second tank; both joints have ``flow_law``."""
    boxes = [cylinder(A[i], s[i]) for i in range(3)]
    return boxes_and_couplers(boxes, [(0, 1), (1, 2)], [joint(flow_law), joint(flow_law)])


def inertial_pipe(m, kappa, p0=0.0):
    """The inertial pipe (``ex.pipe_momentum``): state the momentum ``p`` of its water, response ``1``, internal inclination
    ``q_1 - q_2 - p / (m kappa)``; it returns ``-p/m`` to the first tank and ``+p/m`` to the second."""
    def omega(p, e, n_in):
        return (jnp.array([e[0] - e[1] - p[0] / (m * kappa)]), jnp.array([-p[0] / m, p[0] / m]), _z)

    return Arrangement(P=configuration(diagonal(jnp.array([1.0])), init=jnp.array([float(p0)])),
                       in_dim_M=0, out_dim_M=2, in_dim_N=0, out_dim_N=0,
                       out_f=lambda p, e: _z, in_f=lambda p, e, n_in: _z, omega=omega, label=f"inertialPipe(m={m:g},kappa={kappa:g})")


def steps(arr, T):
    """The flat states of ``T`` steps of ``Phi(arr)``, the start first."""
    O = Phi(arr)
    state, out = O.state, [np.asarray(flat_state(O.state))]
    for _ in range(T):
        _, _, state = O.with_state(state).run_one(IN_POS, lambda _o: (_z, _z))
        out.append(np.asarray(flat_state(state)))
    return np.array(out)


# ---------------------------------------------------------------------------
# The compressible-fluid pipe.
# ---------------------------------------------------------------------------


def test_gas_pipe_label():
    assert gas_pipe((1.0, 1.0, 1.0), (8.0, 0.0, 0.0), linear(0.25)).label == \
        "((joint || joint) . (copy . ((cyl(1.0) || cyl(1.0)) || cyl(1.0))))"


def test_the_gas_pipe_is_like_the_rod():
    """``exc.gas_pipe_rod`` (a), (b): the pressures step as ``q_i' = q_i + load_i / A_i`` with the chain's loads, which is the rod of
    ``sec.heat`` with ``A_1, A, A_2`` as the heat capacities and both joints of conductance ``kappa``."""
    A, kappa, q0 = (2.0, 1.0, 4.0), 0.25, np.array([8.0, 2.0, 0.0])
    S = steps(gas_pipe(A, tuple(q0 * np.array(A)), linear(kappa)), 6)
    pressures = S / np.array(A)
    rod = book_chain([one_port_box(configuration, 1.0 / A[i], jnp.array([q0[i]]), "segment") for i in range(3)], [kappa, kappa])
    np.testing.assert_allclose(pressures, np.array(closed_run(rod, 6)[0]).reshape(7, 3), atol=1e-12)
    q1, q, q2 = q0                                           # the first step, by the formulas of the exercise's solution
    by_hand = [q1 + kappa * (q - q1) / A[0], q + (kappa * (q1 - q) + kappa * (q2 - q)) / A[1], q2 + kappa * (q - q2) / A[2]]
    np.testing.assert_allclose(pressures[1], by_hand, atol=1e-12)


def test_two_halves_make_one_pipe():
    """``exc.gas_pipe_halves``: with the gas in the pipe at the pressure halfway between the tanks', it takes in as much as it lets
    out, and the tanks change as through a resistance-only pipe of conductance ``kappa / 2``."""
    kappa, q1, q2 = 0.25, 6.0, 2.0
    S = steps(gas_pipe((1.0, 1.0, 1.0), (q1, 0.5 * (q1 + q2), q2), linear(kappa)), 1)
    assert S[1, 1] == S[0, 1]
    one = steps(closed([cylinder(1.0, q1), cylinder(1.0, q2)], pipe(kappa / 2)), 1)
    np.testing.assert_allclose(S[1, [0, 2]], one[1], atol=1e-12)
    np.testing.assert_allclose(S[0, 0] - S[1, 0], 0.5 * kappa * (q1 - q2), atol=1e-12)


def test_a_gas_pipe_gains_what_it_takes_in_and_does_not_let_out():
    """``ex.gas_pipe``: "such a pipe can take in more than it lets out, and the difference is the gas it gains"."""
    kappa, s = 0.25, (8.0, 1.0, 2.0)
    S = steps(gas_pipe((1.0, 1.0, 1.0), s, linear(kappa)), 1)
    taken_in, let_out = S[0, 0] - S[1, 0], S[1, 2] - S[0, 2]
    assert taken_in == kappa * (s[0] - s[1]) and let_out == kappa * (s[1] - s[2]) and taken_in > let_out
    np.testing.assert_allclose(S[1, 1] - S[0, 1], taken_in - let_out, atol=1e-12)


def test_any_flow_law_keeps_the_total():
    """``exc.gas_pipe_rod`` (c): a joint returns equal and opposite flows whatever its flow law, so the total gas stays the same."""
    cubic = lambda a, b: 0.01 * (a - b) ** 3
    S = steps(gas_pipe((2.0, 1.0, 4.0), (16.0, 2.0, 0.0), cubic), 8)
    np.testing.assert_allclose(S.sum(axis=1), 18.0, atol=1e-12)
    assert abs(S[8, 0] - S[0, 0]) > 0.1


# ---------------------------------------------------------------------------
# The inertial pipe.
# ---------------------------------------------------------------------------


def two_tanks_inertial(s, m, kappa, p0=0.0):
    """The two tanks of ``sec.pipe``, of unit cross-sections, joined by the inertial pipe (``ex.pipe_momentum``'s picture)."""
    return compose_seq(tensor_arrangements([cylinder(1.0, s[0]), cylinder(1.0, s[1])]), inertial_pipe(m, kappa, p0))


def test_inertial_pipe_label():
    assert two_tanks_inertial((4.0, 0.0), 8.0, 0.25).label == "(inertialPipe(m=8,kappa=0.25) . (cyl(1.0) || cyl(1.0)))"


def test_the_book_steps_of_the_inertial_pipe():
    """``ex.pipe_momentum``, ``m = 8``, ``kappa = 1/4``, from the volumes ``(4, 0)`` with the water at rest: "The first step moves no
    water and gives it the momentum 4. After the next four steps the volumes are (3.5, 0.5), (2.75, 1.25), (2, 2), and
    (1.4375, 2.5625), with the momenta 6, 6, 4.5, and 2.25." The state is ``(s_1, s_2, p)``."""
    S = steps(two_tanks_inertial((4.0, 0.0), 8.0, 0.25), 5)
    np.testing.assert_allclose(S, [[4, 0, 0], [4, 0, 4], [3.5, 0.5, 6], [2.75, 1.25, 6], [2, 2, 4.5], [1.4375, 2.5625, 2.25]],
                               atol=1e-12)
    assert S[4, 0] == S[4, 1] and S[4, 2] > 0 and S[5, 1] > S[5, 0]          # even levels, water still moving, then past


def test_where_the_momentum_settles():
    """``exc.pipe_momentum_settles``: at the momentum ``m kappa (q_1 - q_2)`` the pipe's internal inclination is 0, so its momentum
    stays, and it passes the flow ``kappa (q_1 - q_2)`` of the resistance-only pipe of ``sec.pipe``."""
    m, kappa, s = 8.0, 0.25, (6.0, 2.0)
    S = steps(two_tanks_inertial(s, m, kappa, p0=m * kappa * (s[0] - s[1])), 1)
    assert S[1, 2] == S[0, 2]
    np.testing.assert_allclose(S[1, :2], steps(closed([cylinder(1.0, s[0]), cylinder(1.0, s[1])], pipe(kappa)), 1)[1], atol=1e-12)
