"""Capacitors joined by resistors, and Kirchhoff's current law from the wiring (book Chapter 2,
``sec.kcl`` in ``sec.heat_waves``; linked from the footnote after the current law's paragraph).

For a reader of the book. A capacitor is the tank of ``sec.one_tank`` with charge in place of water:
its state is the charge on its plate, it reads out its voltage, charge over capacitance
(``eqn.capacitor_voltage``), and the current that comes back is added to its charge
(``eqn.circuit_step``). A resistor is the spring of ``sec.springs`` with voltages in place of positions:
no state, the cost function ``(kappa/2)(q_1 - q_2)^2`` with ``kappa`` its conductance, so it returns
``-kappa (q_1 - q_2)`` to its first end and ``+kappa (q_1 - q_2)`` to its second, the current through
it. Where a capacitor's voltage is copied to several resistors, the dot's transpose adds the currents
they return: that sum is Kirchhoff's current law, and no part in the wiring adds anything. Names: the
book writes the charge ``s_i`` and the voltage ``q_i``; this file, following the tank, writes the charge
``q`` and the voltage ``e``, and a resistor by its resistance ``R = 1/kappa`` with a step ``dt``, the book's
being ``dt = 1``. In code: ``capacitor(C, q0)``, ``resistor(R, dt)``, ``source(E)`` (a part held at a fixed
voltage, like the segment on a heater of ``exc.heat_held``: constant readout, response ``0``), and
``network(boxes, pairs, resistors)``, which copies each box's voltage to the resistors its pairs name
(``wiring.boxes_and_couplers``). The loop of ``exc.kcl_loop`` by hand (unit capacitances, conductances
``1/4``, charges ``(8, 0, 0)``),
run from the library's folder with it on the path, as the README's *Run it* says
(``PYTHONPATH=. .venv/bin/python example.py``):

    from tests.test_kirchhoff_node import capacitor, resistor, network, run
    caps = [capacitor(1.0, q) for q in (8.0, 0.0, 0.0)]
    loop = network(caps, [(0, 1), (1, 2), (0, 2)], [resistor(4.0, 1.0) for _ in range(3)])
    print(run(loop, 2))   # arrays [8, 0, 0], [4, 2, 2], [3, 2.5, 2.5]

What the tests check. ``test_book_circuits_by_hand``: the chain of ``sec.kcl`` steps as the rod did,
``(8, 0, 0)`` to ``(6, 2, 0)`` to ``(5, 2.5, 0.5)`` to ``(4.375, 2.625, 1)``, and the loop as above. One
capacitor copied to three resistors, their far ends held at fixed voltages: its charge changes by the
sum of the three currents, produced by the composite and written in no part; two equal resistors to
voltages the same distance above and below cancel exactly. The loop of three capacitors
(``exc.kcl_loop``): the currents are the weighted graph Laplacian of the voltages; the total charge
stays the same at every step while the voltages even out, to the capacitance-weighted mean. One
capacitor between two sources through unequal resistors rests at the conductance-weighted mean of the
two voltages, as nodal analysis says.

For agents: the derivation is ``background/note-retroactivity.tex`` l.94-103 (*Kirchhoff's current law
is the chain rule*); the predictions fixed before this file are ``notes/P32-E-PREDICTIONS-2026-09-29.md``;
the sign of forcings is STATUS.md §1. The file's earlier head is in
``notes/P31-B-CH2-HISTORY-2026-10-30.md``.
"""

import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact, zero_form
from arrangements.coalgebra import configuration
from arrangements.commons import EMPTY, stateless_coupler
from arrangements.functors import Phi
from arrangements.rvect import ReactiveVectorSpace
from arrangements.wiring import boxes_and_couplers, trivial_parameter

from tests.chain import IN_POS

_Z0 = jnp.zeros(0)


# ---------------------------------------------------------------------------
# The components.
# ---------------------------------------------------------------------------


def capacitor(C, q0, label="cap"):
    """A capacitor of capacitance ``C`` holding the charge ``q0``: state the charge ``q``, readout the
    voltage ``e = q/C`` (eqn.capacitor_voltage), response ``C``, no internal inclination; one step adds the
    current to the charge (eqn.circuit_step)."""
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
    """A part held at the voltage ``E`` whatever current comes back: no state, constant readout, like the
    ground of sec.kcl or the segment on a heater of exc.heat_held."""
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m_out: jnp.array([E]),
        in_f=lambda q, m_out, n_in: EMPTY,
        omega=zero_form,
        label=f"{label}({E})",
    )


def resistor(R, dt, label="R"):
    """A resistor of resistance ``R`` (conductance ``kappa = 1/R``) for a step of length ``dt``: the cost
    function ``(dt/2R)(e_1 - e_2)^2``, so it returns ``(-dt I, +dt I)`` with the current ``I = (e_1 - e_2)/R``
    (sec.kcl; the book's step has ``dt = 1``)."""
    V = lambda e: 0.5 * dt / R * (e[0] - e[1]) ** 2
    return stateless_coupler(2, exact(lambda q, e, n: -V(e)), f"{label}({R})")


def network(boxes, pairs, resistors):
    """The circuit as the book draws it: box ``i``'s voltage is copied to one port of each resistor whose
    pair names ``i``, and the currents they return add at the dot (``wiring.boxes_and_couplers``). Closed."""
    return boxes_and_couplers(boxes, pairs, resistors)


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


PAIRS = [(0, 1), (1, 2), (0, 2)]   # the triangle of exc.kcl_loop: each capacitor joined to each other


# ---------------------------------------------------------------------------
# 0. The book's hand steps (sec.kcl, exc.kcl_loop).
# ---------------------------------------------------------------------------


def test_book_circuits_by_hand():
    """Unit capacitances, conductances 1/4 (``R = 4``, ``dt = 1``), charges from (8, 0, 0). The chain of
    sec.kcl steps as the rod of sec.heat did: (6,2,0), (5,2.5,0.5), (4.375,2.625,1), the total 8 at every
    step. The loop of exc.kcl_loop, a third resistor joining the first and third capacitors: (4,2,2),
    (3,2.5,2.5); the first capacitor's distance from 8/3 shrinks by 4 each step, faster than in the chain."""
    caps = lambda: [capacitor(1.0, q, f"cap{i + 1}") for i, q in enumerate((8.0, 0.0, 0.0))]
    chain = network(caps(), [(0, 1), (1, 2)], [resistor(4.0, 1.0, f"R{j}") for j in (1, 2)])
    traj = np.stack(run(chain, 3))
    np.testing.assert_allclose(traj, [[8, 0, 0], [6, 2, 0], [5, 2.5, 0.5], [4.375, 2.625, 1]], atol=1e-12)
    np.testing.assert_allclose(traj.sum(axis=1), 8.0, atol=1e-12)
    loop = network(caps(), PAIRS, [resistor(4.0, 1.0, f"R{i}{j}") for (i, j) in PAIRS])
    traj = np.stack(run(loop, 2))
    np.testing.assert_allclose(traj, [[8, 0, 0], [4, 2, 2], [3, 2.5, 2.5]], atol=1e-12)
    np.testing.assert_allclose((traj[:, 0] - 8.0 / 3.0), [16.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0], atol=1e-12)


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
