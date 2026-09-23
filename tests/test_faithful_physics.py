"""Faithful-machine physics pieces (Phase 3): nonlinear rod gravity + per-gyro gamma.

Two small in-framework closures of blog deviations:

* ``rod_gravity(g, L)`` -- the nonlinear rod-geometry on-site potential
  ``-g sqrt(L^2 - |q|^2)``: a restoring well, *nonlinear* (stiffens toward horizontal),
  whose small-tilt limit is the harmonic well the surrogate used, and which is continued
  past the rod's horizontal limit rather than returning NaN. Plugs into the ``onsite``
  hook of ``compose_graph`` (as a cost; the vertex forcing is ``-d`` of it).
* per-gyro ``gamma``: a per-component vector gamma makes the ``gyro`` coalgebra precess
  each gyro at its own rate (the step just broadcasts it against ``J @ v``).
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, zero_form
from arrangements.coalgebra import gyro
from arrangements.functors import Phi
from arrangements.gyroscope import complex_structure, rod_gravity
from arrangements.interpretation import trivial_omega
from arrangements.rvect import constant

from tests.chain import load_regress

_IN_POS = (jnp.zeros(0), trivial_omega(0))
_TRIV = lambda op: (jnp.zeros(0), jnp.zeros(0))


def test_rod_gravity_is_a_restoring_well_with_harmonic_limit():
    """Minimum upright (r=0), and the small-tilt force is the harmonic ``(g/L) q``."""
    g, L = 2.0, 3.0
    U = rod_gravity(g, L)
    assert abs(float(U(jnp.zeros(2))) - (-g * L)) < 1e-12          # minimum value -gL
    assert float(U(jnp.array([0.5, 0.3]))) > float(U(jnp.zeros(2)))  # a well
    q_small = jnp.array([1e-3, -2e-3])
    grad = jax.grad(U)(q_small)
    np.testing.assert_allclose(np.asarray(grad), np.asarray((g / L) * q_small), rtol=1e-3)


def test_rod_gravity_stiffens_nonlinearly():
    """The restoring force per unit displacement grows toward the rod's horizontal limit
    -- genuinely nonlinear, unlike the constant-stiffness harmonic surrogate."""
    g, L = 1.0, 1.0
    U = rod_gravity(g, L)
    near = float(jnp.linalg.norm(jax.grad(U)(jnp.array([0.1, 0.0])))) / 0.1   # ~ g/L
    far = float(jnp.linalg.norm(jax.grad(U)(jnp.array([0.9, 0.0])))) / 0.9    # >> g/L
    assert far > 1.5 * near


def test_rod_gravity_is_finite_past_horizontal():
    """Past the rod's horizontal limit ``|q| = L`` the potential is continued by its
    tangent line, so value and gradient stay finite and still restoring -- a bare
    ``sqrt`` would return NaN there and (through Adam's moments) poison every
    parameter of a training run for good."""
    g, L = 2.0, 3.0
    U = rod_gravity(g, L)
    for r in (L, 1.5 * L, 10.0 * L):
        q = jnp.array([r, 0.0])
        assert np.isfinite(float(U(q)))
        grad = np.asarray(jax.grad(U)(q))
        assert np.all(np.isfinite(grad))
        assert grad[0] > 0.0            # -dU points back toward upright
    # ...and inside the rod's range it is still exactly the rod potential
    for r in (0.0, 0.5, 2.0, 2.9):
        q = jnp.array([r, 0.0])
        np.testing.assert_allclose(
            float(U(q)), -g * float(np.sqrt(L ** 2 - r ** 2)), rtol=1e-9, atol=1e-9
        )


def _free_gyros(V, m=1.0, parameter=gyro):
    """A closed V-gyro free system (Q = R^{2V}, sharp = (1/m) I, forcing 0 = -d0)."""
    return Arrangement(
        P=parameter(constant(jnp.eye(2 * V) / m)),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=zero_form,
        label="free_gyros",
    )


def test_per_gyro_gamma_precesses_independently():
    """A per-gyro gamma vector precesses each gyro at its own rate: with gamma=(g0,0),
    gyro 0 gets the skew kick -g0 J v_0 and gyro 1 (gamma=0) is untouched."""
    V, m, g0 = 2, 1.0, 0.5
    J = complex_structure(V)
    gamma = jnp.array([g0, g0, 0.0, 0.0])  # per-gyro: gyro 0 -> g0, gyro 1 -> 0
    O = Phi(_free_gyros(V, m, parameter=lambda Q: gyro(Q, gamma=gamma, J=J)))

    xi = jnp.array([0.0, 1.0, 0.0, 1.0])  # both gyros start with the same momentum
    _, _, (q1, xi1) = O.with_state((jnp.zeros(2 * V), xi)).run_one(_IN_POS, _TRIV)

    Jb = np.array([[0.0, -1.0], [1.0, 0.0]])
    v = np.asarray(xi) / m
    exp0 = np.asarray(xi[0:2]) - g0 * (Jb @ v[0:2])  # gyro 0 precesses
    np.testing.assert_allclose(np.asarray(xi1[0:2]), exp0, atol=1e-10)
    np.testing.assert_allclose(np.asarray(xi1[2:4]), np.asarray(xi[2:4]), atol=1e-10)  # gyro 1 fixed


def test_regression_matches_cdlm():
    """CDLM's rod-gravity values/gradients and the per-gyro-gamma step
    (``regress/gyroscope.npz``) are reproduced to rtol 1e-10 (STATUS.md §1)."""
    R = load_regress("gyroscope")
    g, L = R["rod_params"]
    U = rod_gravity(g, L)
    pts = jnp.asarray(R["rod_pts"])
    np.testing.assert_allclose(np.stack([np.asarray(U(p)) for p in pts]), R["rod_U"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.stack([np.asarray(jax.grad(U)(p)) for p in pts]), R["rod_dU"], rtol=1e-10, atol=1e-12)

    V, m = R["pg_params"]
    V = int(V)
    gamma = jnp.asarray(R["pg_gamma"])
    J = complex_structure(V)
    O = Phi(_free_gyros(V, m, parameter=lambda Q: gyro(Q, gamma=gamma, J=J)))
    _, _, (q1, xi1) = O.with_state((jnp.zeros(2 * V), jnp.asarray(R["pg_xi"]))).run_one(_IN_POS, _TRIV)
    np.testing.assert_allclose(np.asarray(q1), R["pg_new_q"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.asarray(xi1), R["pg_new_xi"], rtol=1e-10, atol=1e-12)
