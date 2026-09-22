"""Published one-forms are general (CDLM eqn.Phi_on_obs: ``Omega(M)``), not affine.

For a box with a *nonlinear* forcing, the published one-form ``omega_N``
(sec.lens_internalization; CDLM eqn.omegaprime) is nonlinear in the input. The
interpretation returns it as the exact form (a callable); the old affine
``(A, b)`` reconstruction from value and Jacobian at zero would be wrong away
from zero. We check exactness here.

Sign: CDLM's cost ``U(q, m, n) = sum cos(n)`` becomes the forcing
``omega = -dU``, so the published field is minus CDLM's, ``+sin(n_in)``
(DESIGN.md §3).
"""

import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact
from arrangements.coalgebra import configuration
from arrangements.interpretation import interpretation, trivial_omega
from arrangements.rvect import euclidean


def test_nonlinear_omega_field_is_exact():
    # An open box <R^0|R> -> <R|R> with NONLINEAR cost k(q, m_out, n_in) = sum cos(n_in),
    # i.e. forcing omega = -dk.
    arr = Arrangement(
        P=configuration(euclidean(1)),
        in_dim_M=0, out_dim_M=0, in_dim_N=1, out_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -jnp.sum(jnp.cos(n_in))),
    )
    q = jnp.array([0.3])
    position_action, _ = interpretation(arr)(q)
    out_n, omega_N = position_action(jnp.zeros(0), trivial_omega(0))

    np.testing.assert_allclose(np.asarray(out_n), np.asarray(q), atol=1e-12)
    # omega_N(n_in) = -d/d(n_in) sum cos(n_in) = +sin(n_in): exact at every point,
    # including ones the affine-at-zero reconstruction (slope 1, i.e. n_in) gets wrong.
    for x in (-1.3, 0.0, 0.7, 2.1):
        n_in = jnp.array([x])
        np.testing.assert_allclose(np.asarray(omega_N(n_in)), np.asarray(jnp.sin(n_in)), atol=1e-10)
    assert abs(float(omega_N(jnp.array([2.1]))[0]) - 2.1) > 0.5  # genuinely nonlinear
