"""The dynamics functor ``Phi`` on the standard coalgebras, and the cotangent functor (sec.dynamics_functor)."""

import jax.numpy as jnp
import numpy as np

from arrangements.coalgebra import configuration, phase
from arrangements.functors import Phi, cot_map

from tests.chain import IN_POS, harmonic_particle, load_regress


def test_cot_map_pullback():
    """cot(f).pos = f, cot(f).dir at x = (T_x f)^T (via jax.vjp)."""
    f = lambda x: jnp.array([x[0] + 2.0 * x[1]])
    fp = cot_map(f, 2, 1)
    x = jnp.array([3.0, 4.0])
    np.testing.assert_allclose(fp.on_position(x), jnp.array([11.0]), atol=1e-10)
    np.testing.assert_allclose(fp.on_direction(x, jnp.array([7.0])),
                               jnp.array([7.0, 14.0]), atol=1e-10)


def test_phase_single_particle_update():
    """(q,p) -> (q~, p + xi_Q), q~ = q + p/m, xi_Q = xi_N + kappa(q_prev - q~): the
    book's phase step adds the forcing (NOTATION.md "Displacement map")."""
    m, kappa = 2.0, 3.0
    O = Phi(harmonic_particle(phase, m, kappa))
    q0, p0 = jnp.array([0.7]), jnp.array([1.1])
    xi_N, q_prev = jnp.array([0.4]), jnp.array([0.55])
    qt = q0 + p0 / m  # presented position; forces are evaluated here
    _, _, (new_q, new_p) = O.with_state((q0, p0)).run_one(IN_POS, lambda _o: (xi_N, q_prev))
    np.testing.assert_allclose(new_q, qt, atol=1e-10)
    np.testing.assert_allclose(new_p, p0 + xi_N + kappa * (q_prev - qt), atol=1e-10)


def test_conf_single_particle_displacement():
    """sec.displacement: q -> q + sharp_q(xi_Q), the forcing -dk descends the spring cost."""
    m, kappa = 2.0, 3.0
    O = Phi(harmonic_particle(configuration, m, kappa))
    q0, xi_N, q_prev = jnp.array([0.7]), jnp.array([0.4]), jnp.array([0.55])
    _, _, new_q = O.with_state(q0).run_one(IN_POS, lambda _o: (xi_N, q_prev))
    xi_Q = xi_N - kappa * (q0 - q_prev)
    np.testing.assert_allclose(new_q, q0 + xi_Q / m, atol=1e-10)


def test_conf_phase_share_readout():
    """The configuration and phase arrangements share the readout *formulas*; the
    phase one evaluates them at the presented position q~ = q + sharp(p)."""
    m, kappa = 1.3, 0.8
    q = jnp.array([0.3])
    p = jnp.array([0.5])
    qt = q + p / m  # presented position
    in_dir = (jnp.array([0.1]), jnp.array([0.05]))
    pc, dc, _ = Phi(harmonic_particle(configuration, m, kappa)).with_state(qt).run_one(IN_POS, lambda _o: in_dir)
    pp, dp, _ = Phi(harmonic_particle(phase, m, kappa)).with_state((q, p)).run_one(IN_POS, lambda _o: in_dir)
    np.testing.assert_allclose(pc[0], pp[0], atol=1e-10)        # out_n
    z = jnp.array([0.42])                                       # probe the one-form
    np.testing.assert_allclose(pc[1](z), pp[1](z), atol=1e-10)  # omega_N
    np.testing.assert_allclose(dc[0], dp[0], atol=1e-10)        # returned xi_M


def test_regression_matches_cdlm():
    """``regress/functors.npz``: CDLM's single-particle updates, with xi_N negated (DESIGN.md §3)."""
    R = load_regress("functors")
    m, kappa = 2.0, 3.0
    q0, p0 = jnp.array([0.7]), jnp.array([1.1])
    xi_N, q_prev = jnp.array([-0.4]), jnp.array([0.55])
    _, _, (nq, np_) = Phi(harmonic_particle(phase, m, kappa)).with_state((q0, p0)).run_one(
        IN_POS, lambda _o: (xi_N, q_prev))
    np.testing.assert_allclose(nq, R["phase_new_q"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np_, R["phase_new_p"], rtol=1e-10, atol=1e-12)
    _, _, cq = Phi(harmonic_particle(configuration, m, kappa)).with_state(q0).run_one(
        IN_POS, lambda _o: (xi_N, q_prev))
    np.testing.assert_allclose(cq, R["conf_new_q"], rtol=1e-10, atol=1e-12)

    m2, kappa2 = 1.3, 0.8
    in_dir = (jnp.array([-0.1]), jnp.array([0.05]))
    pp, dp, _ = Phi(harmonic_particle(phase, m2, kappa2)).with_state(
        (jnp.array([0.3]), jnp.array([0.5]))).run_one(IN_POS, lambda _o: in_dir)
    np.testing.assert_allclose(pp[0], R["share_out_n"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(pp[1](jnp.array([0.42])), -R["share_omega_z"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(dp[0], -R["share_xi_M"], rtol=1e-10, atol=1e-12)
