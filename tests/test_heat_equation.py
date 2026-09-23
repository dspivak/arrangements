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

The environment's covector is a forcing in the book's sign: pinning
``q_{K+1} = 0`` means ``xi_N = -kappa * q_K`` (CDLM seeded ``+kappa * q_K`` and
its integrator subtracted; STATUS.md §1).
"""

import jax.numpy as jnp
import numpy as np

from arrangements.coalgebra import configuration
from arrangements.functors import Phi
from arrangements.wiring import compose_chain

from tests.chain import IN_POS, conf_flat, conf_state, harmonic_particle, load_regress


def _heat_chain(K, m, kappa):
    return compose_chain([harmonic_particle(configuration, m, kappa)] * K)


def _step_pinned(O, q, kappa):
    """One pinned-end step: q_prev = 0, xi_N = -kappa * q_K (so q_{K+1} = 0)."""
    K = q.shape[0]
    q_prev = jnp.array([0.0])
    xi_N = jnp.array([-kappa * float(q[K - 1])])
    _, _, new_state = O.with_state(conf_state(q)).run_one(IN_POS, lambda _o: (xi_N, q_prev))
    return conf_flat(new_state, K)


def test_discrete_heat_equation_identity():
    """m (q(t+1) - q(t)) = kappa * Laplacian q(t) at every step (exact, any kappa, m)."""
    K, T = 5, 10
    m, kappa = 1.5, 0.9
    O = Phi(_heat_chain(K, m, kappa))

    rng = np.random.default_rng(3)
    q = jnp.asarray(rng.standard_normal(K))
    traj = [q]
    for _ in range(T):
        q = _step_pinned(O, q, kappa)
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
    O = Phi(_heat_chain(K, m, kappa))

    js = np.arange(1, K + 1)
    for n in range(1, K + 1):
        q0 = jnp.asarray(np.sin(js * n * np.pi / (K + 1)))
        q1 = _step_pinned(O, q0, kappa)
        factor = 1.0 - (kappa / m) * 4.0 * np.sin(n * np.pi / (2.0 * (K + 1))) ** 2
        np.testing.assert_allclose(np.asarray(q1), factor * np.asarray(q0), atol=1e-10)


def test_heat_dissipates_to_equilibrium():
    """For real: in the stable regime the peak displacement decays monotonically to ~0."""
    K, T = 9, 500
    m, kappa = 1.0, 0.2  # sub-stochastic update => max-norm non-increasing
    O = Phi(_heat_chain(K, m, kappa))

    rng = np.random.default_rng(4)
    q = jnp.asarray(rng.standard_normal(K))
    peaks = [float(np.max(np.abs(np.asarray(q))))]
    for _ in range(T):
        q = _step_pinned(O, q, kappa)
        peaks.append(float(np.max(np.abs(np.asarray(q)))))

    assert peaks[-1] < 1e-3 * peaks[0]                                  # dissipated
    assert all(peaks[i + 1] <= peaks[i] + 1e-12 for i in range(T))      # monotone decay


def test_regression_matches_cdlm():
    """The CDLM trajectories (``regress/heat.npz``) are reproduced to rtol 1e-10 (STATUS.md §1)."""
    R = load_regress("heat")
    for tag in ("a", "b"):
        K, T, m, kappa, seed = R[f"params_{tag}"]
        K, T = int(K), int(T)
        O = Phi(_heat_chain(K, m, kappa))
        q = jnp.asarray(np.random.default_rng(int(seed)).standard_normal(K))
        traj = [q]
        for _ in range(T):
            q = _step_pinned(O, q, kappa)
            traj.append(q)
        new = np.stack([np.asarray(x) for x in traj])
        np.testing.assert_allclose(new, R[f"traj_{tag}"], rtol=1e-10, atol=1e-12)
