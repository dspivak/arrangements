"""The heat equation: the chain of harmonic particles with *configuration* parameters.

One wiring, two arrangements (book sec.heat_waves, def:arr): the chain wiring
applied to particles whose parameter is ``configuration(Q)`` is the dissipative
first-order discrete heat equation; the same wiring applied to particles with
``phase(Q)`` is the wave equation (test_wave_equation.py). There is one dynamics
functor ``Phi`` and no integrator to choose. With particle sharp ``p |-> p/m`` and
harmonic forcing ``omega = -dk``, the displacement ``q -> q + sharp_q(xi_Q)`` is
forward Euler for

    m (q_i(t+1) - q_i(t)) = kappa (q_{i-1} - 2 q_i + q_{i+1})(t),

the graph Laplacian of the chain. In the stable regime ``kappa/m`` small it
dissipates to the pinned equilibrium ``q = 0``.

The ends are held by boxes (plan P43 (b) job 4, Claude Opus 5.5, de71554a, 2026-10-10): every run
below is ``Phi`` of ``wiring.held_chain``, the chain with a held particle at ``0`` on each end, the
right one joined to the last part by a spring, closed. Until then the ends were held in code: the
run fed ``q_prev = 0`` in and computed the wall's pull ``xi_N = -kappa * q_K`` from the state each
step (CDLM seeded ``+kappa * q_K`` and its integrator subtracted; STATUS.md §1). That port-fed run is
kept as ``_step_by_port`` and checked equal to the held chain.
"""

import jax.numpy as jnp
import numpy as np

from arrangements.coalgebra import configuration
from arrangements.functors import Phi
from arrangements.wiring import compose_chain

from tests.chain import IN_POS, HeldChain, conf_flat, conf_state, harmonic_particle, load_regress


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
