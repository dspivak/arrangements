"""A Kirchhoff node: the current law falls out of the wiring (P32 (e)).

The derivation under test is the book repo's ``background/note-retroactivity.tex``
l.94-103, *Kirchhoff's current law is the chain rule*: a circuit node is a capacitor
component; the wiring COPIES its voltage to ``k`` element ports; the pullback of
covectors along a copy is their SUM; hence ``q' = -sum_j I_j`` (CDLM's sign). In this
library the step adds the forcing (STATUS.md §1), so the claim reads
``Delta q = -dt sum_j I_j`` with ``I_j = (e_node - e_j)/R_j`` positive leaving the node.

Components, in the idiom of ``tests/test_two_tanks.py`` under the hydraulic-electrical
dictionary (tank = capacitor, pressure = voltage, pipe = resistor):

* capacitor ``C``: state the charge ``q``, response ``sharp = C``, output ``e = q/C``,
  zero forcing; the port covector is pulled back by ``1/C`` and the response multiplies
  by ``C``, so ``q' = q + F(e)``: the returned covector is the charge increment;
* resistor ``R``: a stateless two-port coupler, ``omega = -dV``,
  ``V = (dt/2R)(e_1 - e_2)^2``, so ``F = (-dt I, +dt I)``;
* ideal source ``E``: trivial parameter, constant output; the vjp of a constant is zero,
  so it absorbs any current;
* copy: ``fan_wire`` (``arrangements/commons.py``), ``m |-> m[picks]``, zero forcing.
  Nothing in any component sums over branches.

Predictions (``notes/P32-E-PREDICTIONS-2026-09-29.md``, fixed before this file):

1. one capacitor copied to three resistors to three sources: the ``Phi`` step's charge
   increment is ``-dt sum_j (e - E_j)/R_j`` to ``rtol 1e-12``; the closed forcing on
   ``Q`` is the same sum; two equal resistors to opposite offsets cancel exactly
   (closed form);
2. a triangle of three capacitors: the closed forcing is ``-dt L C^{-1} q`` with ``L``
   the hand-written weighted Laplacian (closed form); total charge is constant to
   ``rtol 1e-12`` at every step (discrete analogue); the voltages converge to the
   capacitance-weighted mean of the initial voltages (closed form, ``atol 1e-8``);
3. one capacitor to two sources through unequal resistors rests at the
   conductance-weighted mean of the sources, the nodal-analysis result (closed form).

What is guaranteed by construction and what is not is written in the predictions note.
"""

import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact, zero_form
from arrangements.coalgebra import configuration
from arrangements.commons import EMPTY, fan_wire, stateless_coupler
from arrangements.functors import Phi
from arrangements.rvect import ReactiveVectorSpace
from arrangements.wiring import compose_seq, tensor_arrangements, trivial_parameter

from tests.chain import IN_POS

_Z0 = jnp.zeros(0)


# ---------------------------------------------------------------------------
# The components.
# ---------------------------------------------------------------------------


def capacitor(C, q0, label="cap"):
    """State ``q``, response ``sharp = C``, output ``e = q / C``, zero forcing."""
    Q = ReactiveVectorSpace(dim=1, sharp_fn=lambda q: jnp.array([[C]]))
    return Arrangement(
        P=configuration(Q, init=jnp.array([q0])),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m_out: q / C,
        in_f=lambda q, m_out, n_in: EMPTY,
        omega=zero_form,
        label=f"{label}({C})",
    )


def source(E, label="src"):
    """An ideal voltage source: no state, constant output ``E``."""
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m_out: jnp.array([E]),
        in_f=lambda q, m_out, n_in: EMPTY,
        omega=zero_form,
        label=f"{label}({E})",
    )


def resistor(R, dt, label="R"):
    """``omega = -dV``, ``V = (dt/2R)(e_1 - e_2)^2``: returns ``(-dt I, +dt I)``, ``I = (e_1 - e_2)/R``."""
    V = lambda e: 0.5 * dt / R * (e[0] - e[1]) ** 2
    return stateless_coupler(2, exact(lambda q, e, n: -V(e)), f"{label}({R})")


def network(boxes, pairs, resistors):
    """``compose_seq(compose_seq(tensor(boxes), copy), tensor(resistors))``: box ``i``'s output
    is copied to one port of each resistor whose pair names ``i``. Closed."""
    picks = [i for pair in pairs for i in pair]
    wired = compose_seq(tensor_arrangements(boxes), fan_wire(len(boxes), picks))
    return compose_seq(wired, tensor_arrangements(resistors))


def run(arr, T):
    """``T`` steps of ``Phi(arr)``; returns the flat configurations (the capacitor charges)."""
    O = Phi(arr)
    state = O.state
    traj = [np.asarray(arr.P.position(state))]
    for _ in range(T):
        _, _, state = O.with_state(state).run_one(IN_POS, lambda _o: (_Z0, _Z0))
        traj.append(np.asarray(arr.P.position(state)))
    return traj


def q_forcing(arr, q):
    """The closed composite's forcing on ``Q`` at configuration ``q``."""
    xi_q, xi_m, xi_n = arr.omega(jnp.asarray(q), _Z0, _Z0)
    assert xi_m.shape == (0,) and xi_n.shape == (0,)
    return np.asarray(xi_q)


# ---------------------------------------------------------------------------
# 1. One node, three branches.
# ---------------------------------------------------------------------------


def _star(C, q0, E, R, dt):
    boxes = [capacitor(C, q0)] + [source(Ej, f"src{j + 1}") for j, Ej in enumerate(E)]
    pairs = [(0, 1 + j) for j in range(len(E))]
    return network(boxes, pairs, [resistor(Rj, dt, f"R{j + 1}") for j, Rj in enumerate(R)])


def test_node_step_is_minus_dt_times_the_sum_of_branch_currents():
    """Prediction 1 (closed form): ``Delta q = -dt sum_j (e - E_j)/R_j`` from the ``Phi`` step,
    the sum produced by the composite, not written in any component."""
    rng = np.random.default_rng(32)
    dt = 0.01
    for C in (0.5, 2.0):
        E = rng.uniform(-2.0, 2.0, size=3)
        R = rng.uniform(0.5, 3.0, size=3)
        q0 = float(rng.uniform(-1.0, 1.0))
        arr = _star(C, q0, E, R, dt)
        assert (arr.out_dim_M, arr.in_dim_M, arr.out_dim_N, arr.in_dim_N) == (0, 0, 0, 0)
        q_before, q_after = run(arr, 1)
        e = q0 / C
        currents = (e - E) / R                      # I_j, positive leaving the node
        want = -dt * currents.sum()
        got = float(q_after[0] - q_before[0])
        np.testing.assert_allclose(got, want, rtol=1e-12)
        assert abs(want) > 1e-4                      # the branches do not happen to cancel


def test_closed_forcing_on_Q_is_the_summed_pullback():
    """Prediction 1, at the level of one-forms: the closed composite's ``Q``-forcing at a random
    charge is ``-dt sum_j (q/C - E_j)/R_j``, with the response not yet applied."""
    rng = np.random.default_rng(33)
    C, dt = 1.7, 0.02
    E = rng.uniform(-2.0, 2.0, size=3)
    R = rng.uniform(0.5, 3.0, size=3)
    arr = _star(C, 0.0, E, R, dt)
    for _ in range(5):
        q = rng.uniform(-3.0, 3.0, size=1)
        want = -dt * ((q[0] / C - E) / R).sum() / C   # pullback along e = q/C multiplies by 1/C
        np.testing.assert_allclose(q_forcing(arr, q), [want], atol=1e-12)


def test_equal_and_opposite_branches_cancel_at_the_node():
    """Prediction 1, the sum's signature: two equal resistors to sources at ``e +/- a`` return
    opposite currents and the node stands still. A wire that copied to one port only would move it."""
    C, dt, a = 1.0, 0.05, 0.8
    q0 = 0.6
    e = q0 / C
    arr = _star(C, q0, [e + a, e - a], [2.0, 2.0], dt)
    q_before, q_after = run(arr, 1)
    assert abs(float(q_after[0] - q_before[0])) < 1e-14
    # and either branch alone does move it, by dt a / R
    lone = _star(C, q0, [e + a], [2.0], dt)
    q_before, q_after = run(lone, 1)
    np.testing.assert_allclose(float(q_after[0] - q_before[0]), dt * a / 2.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# 2. A triangle of capacitors.
# ---------------------------------------------------------------------------


PAIRS = [(0, 1), (1, 2), (0, 2)]


def _triangle(Cs, q0s, Rs, dt):
    boxes = [capacitor(C, q0, f"cap{i + 1}") for i, (C, q0) in enumerate(zip(Cs, q0s))]
    return network(boxes, PAIRS, [resistor(R, dt, f"R{i}{j}") for (i, j), R in zip(PAIRS, Rs)])


def _laplacian(Rs):
    L = np.zeros((3, 3))
    for (i, j), R in zip(PAIRS, Rs):
        g = 1.0 / R
        L[i, i] += g; L[j, j] += g; L[i, j] -= g; L[j, i] -= g
    return L


def test_triangle_forcing_is_the_nodal_equations():
    """Prediction 2a (closed form): the closed ``Q``-forcing is ``-dt C^{-1} L C^{-1} q``, the
    hand-written weighted Laplacian pulled back through each ``e_i = q_i / C_i``."""
    rng = np.random.default_rng(34)
    Cs, Rs, dt = np.array([0.5, 1.0, 2.0]), np.array([1.0, 2.0, 0.5]), 0.05
    arr = _triangle(Cs, [0.0, 0.0, 0.0], Rs, dt)
    L = _laplacian(Rs)
    for _ in range(5):
        q = rng.uniform(-2.0, 2.0, size=3)
        want = -dt * (L @ (q / Cs)) / Cs
        np.testing.assert_allclose(q_forcing(arr, q), want, atol=1e-12)


def test_triangle_conserves_charge_and_relaxes_to_the_weighted_mean():
    """Prediction 2b (discrete analogue) and 2c (closed form): ``sum_i q_i`` constant to rounding
    at every step; voltages converge to ``sum_i C_i e_i(0) / sum_i C_i``."""
    Cs, Rs, dt = np.array([0.5, 1.0, 2.0]), np.array([1.0, 2.0, 0.5]), 0.05
    e0 = np.array([3.0, -1.0, 0.5])
    q0 = Cs * e0
    # stability of the discrete step: dt * lambda_max(C^{-1} L) < 2
    lam = np.linalg.eigvals(np.diag(1.0 / Cs) @ _laplacian(Rs)).real
    assert dt * lam.max() < 2.0 and dt * lam.max() > 0.1
    traj = run(_triangle(Cs, q0, Rs, dt), 400)
    totals = np.array([t.sum() for t in traj])
    np.testing.assert_allclose(totals, totals[0], rtol=1e-12)
    e_star = (Cs * e0).sum() / Cs.sum()
    e_final = traj[-1] / Cs
    np.testing.assert_allclose(e_final, e_star, atol=1e-8)
    assert np.ptp(traj[0] / Cs) > 1.0                     # it had somewhere to go


# ---------------------------------------------------------------------------
# 3. Nodal analysis: the rest voltage.
# ---------------------------------------------------------------------------


def test_rest_voltage_is_the_conductance_weighted_mean():
    """Prediction 3 (closed form): ``e* = (G_1 E_1 + G_2 E_2)/(G_1 + G_2)``, not the plain mean."""
    C, dt = 1.0, 0.1
    E, R = np.array([1.0, -0.5]), np.array([1.0, 1.0 / 3.0])
    G = 1.0 / R
    assert abs(1.0 - dt * G.sum() / C) < 1.0             # a contraction
    traj = run(_star(C, 0.0, E, R, dt), 200)
    e_star = (G * E).sum() / G.sum()
    np.testing.assert_allclose(traj[-1][0] / C, e_star, atol=1e-10)
    assert abs(e_star - E.mean()) > 1e-2
