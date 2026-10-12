"""The standard ``cot(Q)``-coalgebras and their tensor (sec.matter)."""

import jax.numpy as jnp
import numpy as np

from arrangements.coalgebra import configuration, phase
from arrangements.rvect import diagonal


def test_configuration_adds_the_forcing():
    """s |-> s + sharp_s(xi): the displacement is *along* the forcing (sec.differential_equations)."""
    P = configuration(diagonal(jnp.array([0.5, 2.0])))
    s = jnp.array([1.0, -1.0])
    np.testing.assert_allclose(P.position(s), s)
    np.testing.assert_allclose(P.step(s, jnp.array([2.0, 1.0])), jnp.array([2.0, 1.0]))


def test_phase_presents_and_adds():
    """(q, xi) |-> (q + sharp_q xi, xi + xi_Q), position the presented q~."""
    P = phase(diagonal(jnp.array([0.5])))
    s = (jnp.array([1.0]), jnp.array([2.0]))
    np.testing.assert_allclose(P.position(s), jnp.array([2.0]))
    q1, xi1 = P.step(s, jnp.array([-0.5]))
    np.testing.assert_allclose(q1, jnp.array([2.0]))
    np.testing.assert_allclose(xi1, jnp.array([1.5]))


def test_phase_with_a_time_step_scales_both_halves():
    """(q, xi) |-> (q + dt sharp_q xi, xi + dt xi_Q); dt = 1 is the unit step."""
    P = phase(diagonal(jnp.array([0.5])), dt=0.1)
    s = (jnp.array([1.0]), jnp.array([2.0]))
    np.testing.assert_allclose(P.position(s), jnp.array([1.1]))
    q1, xi1 = P.step(s, jnp.array([-0.5]))
    np.testing.assert_allclose(q1, jnp.array([1.1]))
    np.testing.assert_allclose(xi1, jnp.array([1.95]))

def test_tensor_is_the_product_coalgebra():
    """Base Q x Q', state S x S', position concatenated, covector split."""
    P = configuration(diagonal(jnp.array([1.0]))).tensor(phase(diagonal(jnp.array([2.0, 3.0]))))
    assert P.Q_dim == 3
    s = (jnp.array([0.5]), (jnp.array([1.0, 1.0]), jnp.array([0.0, 1.0])))
    np.testing.assert_allclose(P.position(s), jnp.array([0.5, 1.0, 4.0]))
    s1, (q2, xi2) = P.step(s, jnp.array([1.0, 10.0, 20.0]))
    np.testing.assert_allclose(s1, jnp.array([1.5]))
    np.testing.assert_allclose(q2, jnp.array([1.0, 4.0]))
    np.testing.assert_allclose(xi2, jnp.array([10.0, 21.0]))
