"""End-to-end test: the chain-of-K-particles Klein-Gordon equation, phase parameters.

The Klein-Gordon chain extends the harmonic chain (book sec.heat_waves; CDLM
sec.wave_equation) by adding an on-site mass term ``(m_0^2/2) q^2`` to each
particle's cost:

    k(q, n_in) = (kappa/2)(q - n_in)^2 + (m_0^2/2) q^2,   omega = -dk.

The discrete recurrence is

    m (q_i(t+2) - 2 q_i(t+1) + q_i(t)) = kappa (q_{i-1} - 2 q_i + q_{i+1})(t) - m_0^2 q_i(t),

and for a pinned chain of length K the Dirichlet eigenmodes are
``q_j^{(n)} = sin(j n pi / (K+1))`` with
``omega_n^2 = m_0^2/m + (kappa/m) 4 sin^2(n pi / (2(K+1)))``.

Environment covectors are forcings in the book's sign: the pin ``q~_{K+1} = 0``
is ``xi_N = -kappa * q~_K`` (CDLM seeded ``+kappa * q~_K`` and its integrator
subtracted; DESIGN.md §3). States are unchanged.
"""

import os

import jax.numpy as jnp
import numpy as np
import pytest

from arrangements.arrangement import Arrangement, exact
from arrangements.coalgebra import phase
from arrangements.functors import Phi
from arrangements.rvect import diagonal
from arrangements.wiring import compose_chain

from tests.chain import IN_POS, load_regress, phase_flat, phase_state


def _kg_particle(m: float, kappa: float, m0: float, parameter=phase) -> Arrangement:
    """The Klein-Gordon particle: harmonic spring plus on-site mass, forcing ``-dk``."""
    return Arrangement(
        P=parameter(diagonal(jnp.array([1.0 / m]))),
        in_dim_M=0, out_dim_M=0, in_dim_N=1, out_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in:
                    -(0.5 * kappa * (q[0] - n_in[0]) ** 2 + 0.5 * (m0 * m0) * q[0] * q[0])),
        label="KGPart",
    )


def _kg_chain(K, m, kappa, m0):
    return compose_chain([_kg_particle(m, kappa, m0)] * K)


def _kg_composite_final(q, p, q_prev_extern, xi_N_extern, m, kappa, m0):
    """Reference for the KG analogue of CDLM eqn.potlens_composite_final (on-site
    ``-m_0^2 q_i``), with ``xi_N_extern`` a forcing (added, not subtracted)."""
    K = q.shape[0]
    qt = q + p / m  # presented positions q~ (forces are evaluated here)
    q_new = qt
    p_new = []
    for i in range(K):
        if i < K - 1:
            q_prev = q_prev_extern if i == 0 else qt[i - 1]
            q_next = qt[i + 1]
            p_i_new = p[i] + kappa * (q_prev + q_next - 2.0 * qt[i]) - (m0 * m0) * qt[i]
        else:
            q_prev = qt[i - 1] if K > 1 else q_prev_extern
            p_i_new = p[i] + kappa * (q_prev - qt[i]) + xi_N_extern - (m0 * m0) * qt[i]
        p_new.append(p_i_new)
    return q_new, jnp.stack(p_new)


def _run(O, q, p, xi_N, q_prev):
    K = q.shape[0]
    _, _, new_state = O.with_state(phase_state(q, p)).run_one(
        IN_POS, lambda _o: (jnp.atleast_1d(xi_N), jnp.atleast_1d(q_prev)))
    return phase_flat(new_state, K)


def _step_pinned(O, q, p, kappa, m):
    """One pinned-end step: q_prev = 0, xi_N = -kappa * q~_K (presented), q~_{K+1} = 0."""
    K = q.shape[0]
    xi_N = -kappa * float(q[K - 1] + p[K - 1] / m)
    return _run(O, q, p, xi_N, 0.0)


def test_kg_state_update():
    K = 5
    m, kappa, m0 = 1.7, 2.3, 0.4
    O = Phi(_kg_chain(K, m, kappa, m0))

    rng = np.random.default_rng(20260508)
    for _ in range(5):
        q = jnp.asarray(rng.standard_normal(K))
        p = jnp.asarray(rng.standard_normal(K))
        q_prev = float(rng.standard_normal())
        xi_N = float(rng.standard_normal())

        new_q, new_p = _run(O, q, p, xi_N, q_prev)

        ref_q, ref_p = _kg_composite_final(q, p, q_prev, xi_N, m, kappa, m0)
        np.testing.assert_allclose(np.asarray(new_q), np.asarray(ref_q), atol=1e-10)
        np.testing.assert_allclose(np.asarray(new_p), np.asarray(ref_p), atol=1e-10)


def test_kg_recurrence():
    K, T = 5, 10
    m, kappa, m0 = 1.5, 0.9, 0.3
    O = Phi(_kg_chain(K, m, kappa, m0))

    rng = np.random.default_rng(20260508)
    q, p = jnp.asarray(rng.standard_normal(K)), jnp.zeros(K)
    q_traj = [q]
    for _ in range(T + 2):
        q, p = _step_pinned(O, q, p, kappa, m)
        q_traj.append(q)

    q_arr = np.stack([np.asarray(x) for x in q_traj], axis=0)
    for t in range(T):
        ddot_q = q_arr[t + 2] - 2.0 * q_arr[t + 1] + q_arr[t]
        q_aug = np.concatenate([[0.0], q_arr[t + 1], [0.0]])
        laplacian = q_aug[:-2] - 2.0 * q_aug[1:-1] + q_aug[2:]
        residual = m * ddot_q - kappa * laplacian + (m0 * m0) * q_arr[t + 1]
        np.testing.assert_allclose(residual, np.zeros(K), atol=1e-10)


def test_kg_dispersion():
    """Initialize in Dirichlet eigenmode n, evolve two steps, check ddot q = -omega_n^2 q."""
    K = 7
    m, kappa, m0 = 1.3, 0.8, 0.5
    O = Phi(_kg_chain(K, m, kappa, m0))

    js = np.arange(1, K + 1)
    for n in range(1, K + 1):
        q0 = jnp.asarray(np.sin(js * n * np.pi / (K + 1)))
        q, p = q0, jnp.zeros(K)
        q_traj = [q0]
        for _ in range(2):
            q, p = _step_pinned(O, q, p, kappa, m)
            q_traj.append(q)
        q_arr = np.stack([np.asarray(x) for x in q_traj], axis=0)
        ddot_q = q_arr[2] - 2.0 * q_arr[1] + q_arr[0]
        omega2 = (m0 * m0) / m + (kappa / m) * 4.0 * np.sin(n * np.pi / (2.0 * (K + 1))) ** 2
        np.testing.assert_allclose(ddot_q, -omega2 * np.asarray(q0), atol=1e-10)


@pytest.mark.skipif(not os.environ.get("RUN_SLOW"),
                    reason="long-horizon stability diagnostic; set RUN_SLOW=1 to enable")
def test_kg_long_horizon_bounded():
    """The phase parameter is symplectic (presented-position readout), so over a
    long horizon the amplitude stays bounded rather than diverging."""
    K, T = 5, 2000
    m, kappa, m0 = 1.5, 0.9, 0.3
    O = Phi(_kg_chain(K, m, kappa, m0))

    rng = np.random.default_rng(20260508)
    q0 = jnp.asarray(1e-3 * rng.standard_normal(K))
    peak0 = float(np.max(np.abs(np.asarray(q0))))
    q, p = q0, jnp.zeros(K)
    peak = peak0
    for _ in range(T):
        q, p = _step_pinned(O, q, p, kappa, m)
        peak = max(peak, float(np.max(np.abs(np.asarray(q)))))
    assert peak < 50.0 * peak0


def test_regression_matches_cdlm():
    """``regress/misc.npz``: the one-step samples and the pinned trajectory reproduce
    CDLM, with ``xi_N`` negated on the way in (DESIGN.md §3)."""
    R = load_regress("misc")

    K, m, kappa, m0 = int(R["kg_step_params"][0]), *(float(v) for v in R["kg_step_params"][1:])
    O = Phi(_kg_chain(K, m, kappa, m0))
    for i in range(R["kg_q"].shape[0]):
        new_q, new_p = _run(O, jnp.asarray(R["kg_q"][i]), jnp.asarray(R["kg_p"][i]),
                            -float(R["kg_xiN"][i]), float(R["kg_qprev"][i]))
        np.testing.assert_allclose(np.asarray(new_q), R["kg_nq"][i], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(np.asarray(new_p), R["kg_np"][i], rtol=1e-10, atol=1e-12)

    K2, T2 = int(R["kg_pin_params"][0]), int(R["kg_pin_params"][1])
    m2, kappa2, m02 = (float(v) for v in R["kg_pin_params"][2:])
    O2 = Phi(_kg_chain(K2, m2, kappa2, m02))
    q, p = jnp.asarray(R["kg_pin_q"][0]), jnp.asarray(R["kg_pin_p"][0])
    qs, ps = [np.asarray(q)], [np.asarray(p)]
    for _ in range(T2 + 2):
        q, p = _step_pinned(O2, q, p, kappa2, m2)
        qs.append(np.asarray(q)); ps.append(np.asarray(p))
    np.testing.assert_allclose(np.stack(qs), R["kg_pin_q"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.stack(ps), R["kg_pin_p"], rtol=1e-10, atol=1e-12)
