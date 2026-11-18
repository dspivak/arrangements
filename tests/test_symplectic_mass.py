"""Book Ch 3 ``ex.symplectic_form``: the frictionless mass and its regrouping have the same dynamics.

Both systems are ``Arrangement``s run through the dynamics functor (``open_step`` and ``closed_step``
check their compiled step against ``Phi(arr).run_one``). Runs on ``R^1`` and ``R^2`` only.

* Open, told a load ``xi`` at the output: from the same state ``(q, p)`` and at the same duration, the
  mass and the regrouped system (the ?-box with the kinetic box ``-dE``) step to the same state.
* Closed by a spring: the regrouped closed step is symplectic Euler and preserves area (Jacobian
  determinant 1); with the kinetic box's mass different from the look-ahead's it does not.
* The mass is uninclined and the regrouped system is not: different arrangements, one dynamics.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.functors import closed_step, open_step
from arrangements.symplectic_mass import J, mass_system, regrouped_system


def _as_pair(s):
    if isinstance(s, tuple):
        return np.concatenate([np.asarray(s[0]).ravel(), np.asarray(s[1]).ravel()])
    return np.asarray(s).ravel()


def test_same_open_step():
    rng = np.random.default_rng(0)
    for m in (1.0, 2.5):
        for tau in (None, 0.37, 1.0):
            for _ in range(4):
                q, p, xi = rng.standard_normal(3)
                left = mass_system(m, init=(jnp.array([q]), jnp.array([p])))
                right = regrouped_system(m, init=jnp.array([q, p]))
                a = open_step(left, n_in=[], xi_out=[xi], duration=tau)(left.P.init)
                b = open_step(right, n_in=[], xi_out=[xi], duration=tau)(right.P.init)
                np.testing.assert_allclose(_as_pair(a), _as_pair(b), atol=1e-12)


def test_small_case_from_the_book():
    """m = tau = 1, state (1, 0), load -1: both go to (1, -1)."""
    left = mass_system(1.0, init=(jnp.array([1.0]), jnp.array([0.0])))
    right = regrouped_system(1.0, init=jnp.array([1.0, 0.0]))
    a = open_step(left, n_in=[], xi_out=[-1.0], duration=1.0)(left.P.init)
    b = open_step(right, n_in=[], xi_out=[-1.0], duration=1.0)(right.P.init)
    np.testing.assert_allclose(_as_pair(a), [1.0, -1.0])
    np.testing.assert_allclose(_as_pair(b), [1.0, -1.0])


def test_response_is_the_inverse_of_the_symplectic_pairing():
    omega = jnp.array([[0.0, 1.0], [-1.0, 0.0]])        # <(q,p),(q',p')> = q p' - p q'
    np.testing.assert_allclose(np.asarray(J @ omega), np.eye(2))
    np.testing.assert_allclose(np.asarray(J + J.T), np.zeros((2, 2)))


def test_closed_step_preserves_area_only_with_one_mass():
    kappa, tau = 1.3, 0.4
    s0 = jnp.array([0.7, -0.2])
    for m, mk, det_is_one in ((1.5, 1.5, True), (1.0, 2.0, False)):
        right = regrouped_system(m, init=s0, kappa=kappa, m_kinetic=mk)
        step = closed_step(right, duration=tau)
        det = float(jnp.linalg.det(jax.jacobian(step)(s0)))
        expected = 1.0 if det_is_one else 1.0 - tau**2 * kappa / m + tau**2 * kappa / mk
        np.testing.assert_allclose(det, expected, atol=1e-12)
        if det_is_one:
            left = mass_system(m, init=(jnp.array([0.7]), jnp.array([-0.2])), kappa=kappa)
            a = closed_step(left, duration=tau)(left.P.init)
            np.testing.assert_allclose(_as_pair(a), np.asarray(step(s0)), atol=1e-12)


def test_uninclined_left_not_right():
    left, right = mass_system(1.0), regrouped_system(1.0, init=jnp.array([0.0, 2.0]))
    z = jnp.zeros(0)
    assert float(jnp.abs(left.omega(jnp.array([0.3]), z, z)[0]).sum()) == 0.0
    assert float(jnp.abs(right.omega(jnp.array([0.3, 2.0]), z, z)[0]).sum()) > 0.0
