"""Heat along a rod of three segments, and the chain of any length with its ends held (book Chapter 2,
``sec.heat`` in ``sec.heat_waves``; linked, with ``test_wave_equation.py``, from the footnote in ``sec.waves``
on what makes the difference between heat and waves).

For a reader of the book. The rod's three segments fill the boxes of the chain ``eqn.chain_wd``. A
segment is a part whose state is its temperature ``theta_i``, which it reads out; it has no internal
inclination, and its response function multiplies the flow of heat that comes back to it by ``1/C_i``,
its heat capacity's inverse, so one step is ``theta_i' = theta_i + xi_i / C_i`` (``eqn.heat_step``). A
joint between two segments is the spring of ``sec.springs`` with temperatures in place of positions,
Fourier's law as the cost function ``(kappa/2)(theta_1 - theta_2)^2`` with ``kappa`` its conductance;
the dot copies the middle temperature to both joints and the flows they return add, giving the forces
``eqn.chain_forces``. In code (``tests/chain.py``) a segment is
``one_port_box(configuration, 1/C, [theta0], "segment")``, a joint is ``spring(kappa)``, and the rod is
``book_chain(segments, [kappa_1, kappa_2])``, its ends free as in the book, so the total heat stays
put. The book's hand steps (unit heat capacities, conductances ``1/4``, from ``(8, 0, 0)``),
run from the library's folder with it on the path, as the README's *Run it* says
(``PYTHONPATH=. .venv/bin/python example.py``):

    from arrangements.coalgebra import configuration
    from tests.chain import one_port_box, book_chain, closed_run
    segments = [one_port_box(configuration, 1.0, [t], "segment") for t in (8.0, 0.0, 0.0)]
    temperatures, _ = closed_run(book_chain(segments, [0.25, 0.25]), 3)
    print(temperatures)   # arrays [8, 0, 0], [6, 2, 0], [5, 2.5, 0.5], [4.375, 2.625, 1]

What the tests check. ``test_book_rod_by_hand``: those steps; the sum ``8`` at every step and the
approach of all three temperatures to ``8/3`` (``exc.heat_middle``); every point of the plot
``eqn.heat_plot``; and the step that is too long, ``kappa = 1`` (``exc.heat_long_step``). The rest run
the chain of ``K`` harmonic particles with both ends held at ``0`` by walls (``wiring.held_chain``; a
wall is a part with response ``0``, as the segment on a heater in ``exc.heat_held``), each particle a
box carrying its own spring to its left neighbour: the step is
``m (q(t+1) - q(t)) = kappa Lap q(t)`` with the walls' Laplacian; each standing wave decays by the
factor ``1 - (kappa/m) 4 sin^2(n pi / 2(K+1))``; the peak decays to ``0`` when ``kappa/m`` is small; and
the same wiring filled with ``phase`` particles is the wave equation (``test_wave_equation.py``: one
wiring, filled two ways).

For agents: ``regress/heat.npz`` holds the CDLM trajectories this reproduces; the pinned ends once
computed in code each step are kept as ``_step_by_port`` and checked equal to the held chain (the sign
of forcings is STATUS.md §1). The file's earlier head is in ``notes/P31-B-CH2-HISTORY-2026-10-30.md``.
"""

import jax.numpy as jnp
import numpy as np

from arrangements.coalgebra import configuration
from arrangements.functors import Phi
from arrangements.wiring import compose_chain

from tests.chain import (
    IN_POS, HeldChain, book_chain, closed_run, conf_flat, conf_state, harmonic_particle, load_regress,
    one_port_box,
)


def _heat_chain(K, m, kappa):
    return compose_chain([harmonic_particle(configuration, m, kappa)] * K)


def _held(K, m, kappa):
    """The chain with its ends held by boxes, closed."""
    return HeldChain([harmonic_particle(configuration, m, kappa)] * K, kappa, configuration)


def _step_by_port(O, q, kappa):
    """The old pinned step, ends held in code: q_prev = 0 fed in, xi_N = -kappa * q_K computed."""
    K = q.shape[0]
    q_prev = jnp.array([0.0])
    xi_N = jnp.array([-kappa * float(q[K - 1])])
    _, _, new_state = O.with_state(conf_state(q)).run_one(IN_POS, lambda _o: (xi_N, q_prev))
    return conf_flat(new_state, K)


# ---------------------------------------------------------------------------
# The rod as the book draws it (sec.heat): three segments, two joints, free ends.
# ---------------------------------------------------------------------------


def _rod(thetas, kappas=(0.25, 0.25), C=1.0):
    """The three segments at temperatures ``thetas``, heat capacity ``C`` each, joints of conductances
    ``kappas``: ``book_chain`` of ``one_port_box(configuration, 1/C, ...)``."""
    segments = [one_port_box(configuration, 1.0 / C, jnp.array([t]), "segment") for t in thetas]
    return book_chain(segments, list(kappas))


HEAT_PLOT = np.array([   # eqn.heat_plot: steps 0..12 of the first, middle and last segment, 3 decimals
    [8, 0, 0], [6, 2, 0], [5, 2.5, 0.5], [4.375, 2.625, 1], [3.938, 2.656, 1.406], [3.617, 2.664, 1.719],
    [3.379, 2.666, 1.955], [3.201, 2.667, 2.133], [3.067, 2.667, 2.266], [2.967, 2.667, 2.366],
    [2.892, 2.667, 2.441], [2.836, 2.667, 2.498], [2.793, 2.667, 2.54],
])


def test_book_rod_by_hand():
    """sec.heat's hand steps, unit heat capacities and conductances 1/4: (8,0,0) -> (6,2,0) -> (5,2.5,0.5)
    -> (4.375,2.625,1); each step is eqn.heat_step with the free-ended forces eqn.chain_forces; the sum
    stays 8; all three approach 8/3 (exc.heat_middle); the plot eqn.heat_plot, point by point; and with
    conductances 1 the step overshoots, (8,0,0) -> (0,8,0) -> (8,-8,8) (exc.heat_long_step)."""
    temperatures, _ = closed_run(_rod((8.0, 0.0, 0.0)), 80)
    arr = np.stack(temperatures)
    np.testing.assert_allclose(arr[:4], [[8, 0, 0], [6, 2, 0], [5, 2.5, 0.5], [4.375, 2.625, 1]], atol=1e-12)
    L = 0.25 * np.array([[-1.0, 1.0, 0.0], [1.0, -2.0, 1.0], [0.0, 1.0, -1.0]])   # the forces, free ends
    for t in range(12):
        np.testing.assert_allclose(arr[t + 1], arr[t] + L @ arr[t], atol=1e-12)
    np.testing.assert_allclose(arr.sum(axis=1), 8.0, atol=1e-12)
    np.testing.assert_allclose(arr[-1], 8.0 / 3.0, atol=1e-8)
    np.testing.assert_allclose(arr[:13], HEAT_PLOT, atol=6e-4)
    too_long, _ = closed_run(_rod((8.0, 0.0, 0.0), kappas=(1.0, 1.0)), 2)
    np.testing.assert_allclose(np.stack(too_long), [[8, 0, 0], [0, 8, 0], [8, -8, 8]], atol=1e-12)


# ---------------------------------------------------------------------------
# The chain of any length with its ends held by walls (wiring.held_chain).
# ---------------------------------------------------------------------------


def test_held_ends_are_the_port_fed_run():
    """The held chain's ``Phi`` equals the old run that held the ends in code, over 500 steps at
    ``K = 9``, ``kappa = 0.2`` and 10 steps at ``K = 5``, ``m = 1.5``, ``kappa = 0.9``."""
    for K, T, m, kappa, seed in ((9, 500, 1.0, 0.2, 4), (5, 10, 1.5, 0.9, 3)):
        H, O = _held(K, m, kappa), Phi(_heat_chain(K, m, kappa))
        q = r = jnp.asarray(np.random.default_rng(seed).standard_normal(K))
        for _ in range(T):
            q, r = H.step(q), _step_by_port(O, r, kappa)
        np.testing.assert_allclose(np.asarray(q), np.asarray(r), atol=1e-12)


def test_discrete_heat_equation_identity():
    """m (q(t+1) - q(t)) = kappa * Laplacian q(t) at every step (exact, any kappa, m)."""
    K, T = 5, 10
    m, kappa = 1.5, 0.9
    H = _held(K, m, kappa)

    rng = np.random.default_rng(3)
    q = jnp.asarray(rng.standard_normal(K))
    traj = [q]
    for _ in range(T):
        q = H.step(q)
        traj.append(q)

    arr = np.stack([np.asarray(x) for x in traj])
    for t in range(T):
        dq = arr[t + 1] - arr[t]
        aug = np.concatenate([[0.0], arr[t], [0.0]])
        laplacian = aug[:-2] - 2.0 * aug[1:-1] + aug[2:]
        np.testing.assert_allclose(m * dq - kappa * laplacian, np.zeros(K), atol=1e-10)


def test_heat_eigenmode_decays_geometrically():
    """A Dirichlet mode decays by the exact factor 1 - (kappa/m) 4 sin^2(n pi / 2(K+1))."""
    K = 7
    m, kappa = 1.0, 0.15  # kappa/m small => stable
    H = _held(K, m, kappa)

    js = np.arange(1, K + 1)
    for n in range(1, K + 1):
        q0 = jnp.asarray(np.sin(js * n * np.pi / (K + 1)))
        q1 = H.step(q0)
        factor = 1.0 - (kappa / m) * 4.0 * np.sin(n * np.pi / (2.0 * (K + 1))) ** 2
        np.testing.assert_allclose(np.asarray(q1), factor * np.asarray(q0), atol=1e-10)


def test_heat_dissipates_to_equilibrium():
    """For real: in the stable regime the peak displacement decays monotonically to ~0."""
    K, T = 9, 500
    m, kappa = 1.0, 0.2  # sub-stochastic update => max-norm non-increasing
    H = _held(K, m, kappa)

    rng = np.random.default_rng(4)
    q = jnp.asarray(rng.standard_normal(K))
    peaks = [float(np.max(np.abs(np.asarray(q))))]
    for _ in range(T):
        q = H.step(q)
        peaks.append(float(np.max(np.abs(np.asarray(q)))))

    assert peaks[-1] < 1e-3 * peaks[0]                                  # dissipated
    assert all(peaks[i + 1] <= peaks[i] + 1e-12 for i in range(T))      # monotone decay


def test_regression_matches_cdlm():
    """The CDLM trajectories (``regress/heat.npz``) are reproduced to rtol 1e-10 (STATUS.md §1)."""
    R = load_regress("heat")
    for tag in ("a", "b"):
        K, T, m, kappa, seed = R[f"params_{tag}"]
        K, T = int(K), int(T)
        H = _held(K, m, kappa)
        q = jnp.asarray(np.random.default_rng(int(seed)).standard_normal(K))
        traj = [q]
        for _ in range(T):
            q = H.step(q)
            traj.append(q)
        new = np.stack([np.asarray(x) for x in traj])
        np.testing.assert_allclose(new, R[f"traj_{tag}"], rtol=1e-10, atol=1e-12)
