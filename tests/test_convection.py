"""A convection roll as a ring of stirred heat tanks (book sec.driven; ``notes/CONVECTION-RESULTS.md``).

Four cells around a loop -- bottom, right, top, left -- each a *tank of heat*: state its temperature ``T_i``
(unit heat capacity), response the step size ``s``, output ``T_i``, zero forcing. Two *plates* are tanks whose
response is zero, so their temperatures ``+Delta/2`` (hot, under the bottom cell) and ``-Delta/2`` (cold, over the
top cell) never move: the maintained difference (plan §5.6). One stateless *flow box* receives all six temperatures;
its flow ``Q = g (T_right - T_left)`` is set at each moment by how much warmer the rising side is (no inertia), and it
returns at each cell the sum of

* conduction ``D (T_up + T_down - 2 T_i)`` along the loop (exact, a pipe for heat);
* plate exchange ``k (T_plate - T_i)`` at the bottom and top cells (exact);
* heat carried by the flow, ``Q (T_upstream - T_i)``: a stirred tank takes fluid in at its upstream neighbour's
  temperature and lets it out at its own. Not exact: one-way, like the waterfall.

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

from arrangements.arrangement import Arrangement, zero_form
from arrangements.coalgebra import configuration
from arrangements.functors import Phi
from arrangements.rvect import ReactiveVectorSpace
from arrangements.wiring import compose_seq, tensor_arrangements, trivial_parameter

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
    """The flow box's returned covector at its six ports (four cells, then hot, cold); plates get nothing."""
    T, Th, Tc = e[:4], e[4], e[5]
    Q = G * (T[1] - T[3])
    up, dn = jnp.roll(T, 1), jnp.roll(T, -1)
    carried = jnp.maximum(Q, 0) * (up - T) + jnp.maximum(-Q, 0) * (dn - T)
    conducted = D * (up + dn - 2 * T)
    plates = jnp.array([K * (Th - T[0]), 0.0, K * (Tc - T[2]), 0.0])
    return jnp.concatenate([carried + conducted + plates, jnp.zeros(2)])


def flow_box():
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0, out_dim_M=6, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, e: jnp.zeros(0),
        in_f=lambda q, e, n_in: jnp.zeros(0),
        omega=lambda q, e, n_in: (jnp.zeros(0), flow_F(e), jnp.zeros(0)),
        label="flow",
    )


def loop(Delta, cells):
    parts = [heat_tank(cells[i], S, f"cell{i}") for i in range(4)]
    parts += [heat_tank(Delta / 2, 0.0, "hot"), heat_tank(-Delta / 2, 0.0, "cold")]
    return Phi(compose_seq(tensor_arrangements(parts), flow_box()))


def temperatures(state):
    """Unnest the left-nested state into (T_0..T_3, hot, cold)."""
    out, x = [], state[0]
    while isinstance(x, tuple):
        out.append(x[1])
        x = x[0]
    out.append(x)
    return np.concatenate(out[::-1])


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


def test_flow_and_conduction_move_heat_without_making_any():
    """Summed over the four cells, the returned covector is only what the plates exchange."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        e = jnp.asarray(rng.standard_normal(6))
        F = flow_F(e)
        np.testing.assert_allclose(float(F[:4].sum()), float(K * (e[4] - e[0]) + K * (e[5] - e[2])), atol=1e-12)


def test_heat_carried_is_one_way_not_exact():
    """Like the waterfall: the flow box's covector Jacobian is not symmetric (d omega != 0)."""
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
