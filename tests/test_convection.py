"""A convection roll as a ring of stirred heat tanks (book sec.driven; ``notes/CONVECTION-RESULTS.md``).

Four cells around a loop -- bottom, right, top, left -- each a *tank of heat*: state its temperature ``T_i``
(unit heat capacity), response the step size ``s``, output ``T_i``, zero forcing. Two *plates* are tanks whose
response is zero, so their temperatures ``+Delta/2`` (hot, under the bottom cell) and ``-Delta/2`` (cold, over the
top cell) never move: the maintained difference (plan §5.6). Every other box touches only neighbours (redrawn
2026-10-02, Claude Opus 5.5, 9dd117f1, at David's worry that one box receiving all six temperatures held all the
knowledge): a stateless *buoyancy box* reads the right and left tanks and sends the flow ``Q = g (T_right - T_left)``,
set at each moment by how much warmer the rising side is (no inertia); it is one number round the ring because what
leaves one tank enters the next. Each cell receives the sum of

* conduction ``D (T_up + T_down - 2 T_i)`` along the loop (exact);
* plate exchange ``k (T_plate - T_i)`` at the bottom and top cells, two of Heat's joints (exact);
* heat carried by the flow, ``Q (T_upstream - T_i)``: a stirred tank takes fluid in at its upstream neighbour's
  temperature and lets it out at its own. Not exact: one-way, like the waterfall.

Conduction and carrying between two neighbouring tanks are one *ring joint*, Heat's joint with the flow as a third
input, on which it returns no force.

Closed forms (derived in the results note, checked here): the still state loses stability at
``Delta_c = 2D(k + 2D)/(gk)``; above it the steady flow solves ``2Q^2 + (4D + k) Q = gk (Delta - Delta_c)``, so it
grows in a straight line from onset (the stirring inside each tank is mixing proportional to the flow), the bottom
cell sits at ``T_0 = (2D + Q)/(2g) = -T_2``, and the heat through the loop ``J = k (Delta/2 - T_0)`` exceeds
conduction's ``kD Delta/(k + 2D)`` exactly when ``Delta > Delta_c``. More, smaller stirred tanks recover a
continuous fluid's square-root onset (``notes/convection_many_tanks.py``; not a test).

Status: model. Omits momentum (so no oscillation or chaos under strong drive), the roll's shape within a cell,
and a hurricane's swirl.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact, zero_form
from arrangements.coalgebra import configuration
from arrangements.functors import Phi
from arrangements.rvect import ReactiveVectorSpace
from arrangements.wiring import compose_seq, tensor_arrangements, trivial_parameter

from arrangements.commons import stateless_coupler

from tests.chain import IN_POS

D, K, G, S = 0.1, 1.0, 1.0, 0.05
DELTA_C = 2 * D * (K + 2 * D) / (G * K)


def heat_tank(T0, s, label):
    """State the temperature, response ``s`` (zero for a plate), output the temperature."""
    Q = ReactiveVectorSpace(dim=1, sharp_fn=lambda v: jnp.array([[s]]))
    return Arrangement(
        P=configuration(Q, init=jnp.array([T0])),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda v, m_out: v,
        in_f=lambda v, m_out, n_in: jnp.zeros(0),
        omega=zero_form,
        label=label,
    )


def flow_F(e):
    """The hand formula: the force all the joints return at each cell (four cells, then hot, cold; plates never move)."""
    T, Th, Tc = e[:4], e[4], e[5]
    Q = G * (T[1] - T[3])
    up, dn = jnp.roll(T, 1), jnp.roll(T, -1)
    carried = jnp.maximum(Q, 0) * (up - T) + jnp.maximum(-Q, 0) * (dn - T)
    conducted = D * (up + dn - 2 * T)
    plates = jnp.array([K * (Th - T[0]), 0.0, K * (Tc - T[2]), 0.0])
    return jnp.concatenate([carried + conducted + plates, jnp.zeros(2)])


def buoyancy_box():
    """Reads the left and right tanks and sends the flow ``Q = g (T_right - T_left)``; returns no force."""
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0, out_dim_M=2, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m: G * (m[1:2] - m[0:1]),
        in_f=lambda q, m, n_in: jnp.zeros(0),
        omega=zero_form,
        label="buoyancy",
    )


def ring_joint(sign):
    """Heat's joint between neighbouring tanks ``a`` and ``b``, plus the flow: receiving ``(T_a, T_b, Q)``, it
    conducts ``D (T_a - T_b)`` (exact) and carries water into whichever tank the flow points into at the other's
    temperature (one-way, like the waterfall; no force on the flow). ``sign Q > 0`` means the flow goes a to b."""

    def omega(q, e, n_in):
        Ta, Tb, Q = e[0], e[1], sign * e[2]
        F = jnp.array([(D + jnp.maximum(-Q, 0)) * (Tb - Ta), (D + jnp.maximum(Q, 0)) * (Ta - Tb), 0.0])
        return jnp.zeros_like(q), F, jnp.zeros_like(n_in)

    return stateless_coupler(3, omega, "joint")


def heat_joint(c, label):
    """Heat's joint with conductance ``c``: exact, ``-(c/2)(T_a - T_b)^2``."""
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0, out_dim_M=2, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, e: jnp.zeros(0),
        in_f=lambda q, e, n_in: jnp.zeros(0),
        omega=exact(lambda q, e, n_in: -0.5 * c * (e[0] - e[1]) ** 2),
        label=label,
    )


def copy(n, idx):
    """Copies and routes wires: output ``k`` is input ``idx[k]``; forces at copies add."""
    idx = jnp.asarray(idx, dtype=int)
    return Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=n, in_dim_N=0, out_dim_N=len(idx),
        out_f=lambda q, m: m[idx], in_f=lambda q, m, n_in: jnp.zeros(0), omega=zero_form, label="copy",
    )


RING = [(2, 3, 1), (2, 1, -1), (3, 0, 1), (1, 0, -1)]   # (a, b, sign): top-left, top-right, left-bottom,
PLATES = [(5, 2), (0, 4)]                                # right-bottom; positive flow goes bottom, right, top, left;
                                                         # plates: cold with top, bottom with hot


def loop(Delta, cells):
    """Every box touches only neighbours except the buoyancy box, which reads the left and right tanks; the flow
    it sends is one number because what leaves one tank enters the next. Cells 0..3 are bottom, right, top, left."""
    tanks = [heat_tank(cells[i], S, f"cell{i}") for i in range(4)]
    tanks += [heat_tank(Delta / 2, 0.0, "hot"), heat_tank(-Delta / 2, 0.0, "cold")]
    to_buoyancy = copy(6, [3, 1, 0, 1, 2, 3, 4, 5])                 # (left, right) and all six temperatures
    flow = tensor_arrangements([buoyancy_box(), copy(6, range(6))])  # (Q, T_0..T_3, hot, cold)
    wires = [w for (a, b, _) in RING for w in (1 + a, 1 + b, 0)]
    wires += [1 + k for (a, b) in PLATES for k in (a, b)]
    joints = [ring_joint(sign) for (_, _, sign) in RING] + [heat_joint(K, "joint") for _ in PLATES]
    arr = compose_seq(tensor_arrangements(tanks), to_buoyancy)
    arr = compose_seq(arr, flow)
    arr = compose_seq(arr, copy(7, wires))
    return Phi(compose_seq(arr, tensor_arrangements(joints)))


def temperatures(state):
    """(T_0..T_3, hot, cold): the only nonempty leaves of the composite's state."""
    return np.concatenate([np.ravel(x) for x in jax.tree_util.tree_leaves(state)])


def step_phi(O, state, n):
    for _ in range(n):
        _, _, state = O.with_state(state).run_one(IN_POS, lambda _o: (jnp.zeros(0), jnp.zeros(0)))
    return state


@jax.jit
def _hand_run(T, Th, Tc):
    """20000 hand steps ``T' = T + s F(T)``, which one ``Phi`` step equals (first test)."""
    return jax.lax.fori_loop(
        0, 20000, lambda i, T: T + S * flow_F(jnp.concatenate([T, jnp.array([Th, Tc])]))[:4], T)


def settle(Delta, cells=(0.0, 1e-3, 0.0, 0.0)):
    return np.asarray(_hand_run(jnp.array(cells), Delta / 2, -Delta / 2))


def flow(T):
    return G * (T[1] - T[3])


def steady(Delta, cells=(0.0, 1e-3, 0.0, 0.0)):
    """Polish ``settle`` by Newton on ``F = 0`` (near onset the roll settles slowly)."""
    f = lambda T: flow_F(jnp.concatenate([T, jnp.array([Delta / 2, -Delta / 2])]))[:4]
    T = jnp.array(settle(Delta, cells))
    for _ in range(30):
        T = T - jnp.linalg.solve(jax.jacobian(f)(T), f(T))
    assert float(jnp.abs(f(T)).max()) < 1e-12
    return np.asarray(T)


# ---------------------------------------------------------------------------


def test_phi_step_is_the_hand_step_and_plates_hold():
    O = loop(0.5, [0.1, 0.02, -0.1, -0.03])
    x = temperatures(O.state)
    y = temperatures(step_phi(O, O.state, 5))
    for _ in range(5):
        x = np.concatenate([x[:4] + S * np.asarray(flow_F(jnp.array(x)))[:4], x[4:]])
    np.testing.assert_allclose(y, x, atol=1e-14)
    np.testing.assert_allclose(y[4:], [0.25, -0.25], atol=0)


def test_wired_loop_is_the_hand_step_at_random_states_and_a_wrong_wire_breaks_it():
    rng = np.random.default_rng(1)
    for _ in range(3):
        O = loop(0.5, list(0.2 * rng.standard_normal(4)))
        x = temperatures(O.state)
        y = temperatures(step_phi(O, O.state, 1))
        np.testing.assert_allclose(y[:4], x[:4] + S * np.asarray(flow_F(jnp.array(x)))[:4], atol=1e-14)
    RING[1], saved = (2, 1, 1), RING[1]   # the top-right joint facing the wrong way
    try:
        O = loop(0.5, [0.1, 0.02, -0.1, -0.03])
        x = temperatures(O.state)
        y = temperatures(step_phi(O, O.state, 1))
        assert np.abs(y[:4] - x[:4] - S * np.asarray(flow_F(jnp.array(x)))[:4]).max() > 1e-4
    finally:
        RING[1] = saved


def test_wired_loop_itself_stills_below_onset_and_rolls_above():
    """The stir through the composite, not the hand formula: 60 steps."""
    stir = [0.0, 1e-3, 0.0, 0.0]
    below = flow(temperatures(step_phi(loop(0.8 * DELTA_C, stir), loop(0.8 * DELTA_C, stir).state, 60)))
    above = flow(temperatures(step_phi(loop(2.0 * DELTA_C, stir), loop(2.0 * DELTA_C, stir).state, 60)))
    assert below < 1e-3 * G < above


def test_flow_and_conduction_move_heat_without_making_any():
    """Summed over the four cells, the returned covector is only what the plates exchange."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        e = jnp.asarray(rng.standard_normal(6))
        F = flow_F(e)
        np.testing.assert_allclose(float(F[:4].sum()), float(K * (e[4] - e[0]) + K * (e[5] - e[2])), atol=1e-12)


def test_heat_carried_is_one_way_not_exact():
    """Like the waterfall: the summed pushback's Jacobian is not symmetric (d omega != 0)."""
    e = jnp.array([0.3, 0.2, -0.3, -0.1, 0.5, -0.5])
    J = jax.jacobian(flow_F)(e)
    assert np.abs(np.asarray(J - J.T)).max() > 1e-2


def test_still_below_onset_roll_above():
    """A small stir dies out below ``Delta_c`` and grows above it."""
    stir = (0.0, 1e-3, 0.0, 0.0)
    for r, grows in ((0.5, False), (0.9, False), (0.95, False), (1.05, True), (1.1, True), (2.0, True)):
        T = settle(r * DELTA_C, stir)
        assert (abs(flow(T)) > abs(flow(np.array(stir)))) == grows, r


def test_steady_roll_matches_closed_form_and_carries_more_heat():
    for r in (1.1, 2.0, 4.0, 8.0):
        Delta = r * DELTA_C
        T = steady(Delta)
        Q = flow(T)
        b = 4 * D + K
        Q_formula = (-b + np.sqrt(b * b + 8 * G * K * (Delta - DELTA_C))) / 4
        np.testing.assert_allclose(Q, Q_formula, rtol=1e-6)
        np.testing.assert_allclose(T[0], (2 * D + Q) / (2 * G), rtol=1e-6)
        np.testing.assert_allclose(T[2], -T[0], atol=1e-9)
        J = K * (Delta / 2 - T[0])
        assert J > K * D * Delta / (K + 2 * D) * (1 + 1e-3)


def test_onset_is_a_straight_line_for_stirred_tanks():
    """Just above onset the flow grows in proportion to the excess drive, slope gk/(4D + k)."""
    q1, q2 = flow(steady(1.01 * DELTA_C)), flow(steady(1.02 * DELTA_C))
    assert abs(np.log(q2 / q1) / np.log(2) - 1) < 0.03


def test_withdrawing_the_difference_kills_the_roll():
    T = settle(4 * DELTA_C)
    T_after = np.asarray(_hand_run(jnp.array(T), 0.0, 0.0))
    assert abs(flow(T_after)) < 1e-12


def test_the_stir_picks_the_direction():
    assert flow(settle(4 * DELTA_C, (0.0, -1e-3, 0.0, 0.0))) < -1e-3
