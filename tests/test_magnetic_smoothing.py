"""Whole-plane field morphisms in smooth sets: numerical regression checks.

The flat correction proves smoothness analytically; finite derivative samples
and exact exterior agreement test its implementation, not all derivative orders.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from arrangements import magnets as M
from magnetic_checks import checked_jit, outside_smoothing_discs


@pytest.mark.parametrize('field,bare', [
    (M.g, lambda d: -jnp.log(jnp.sum(d*d))/(4*jnp.pi)),
    (lambda d: M.contact(d,.25), lambda d: (.25**2/jnp.sum(d*d))**3),
])
def test_fields_equal_bare_formulas_outside_disc(field,bare):
    for radius in [M.r0, 1.001*M.r0, .1, .25, 1., 4.]:
        d=jnp.array([radius,0.])
        for order in range(3):
            a,b=field,bare
            for _ in range(order): a,b=jax.jacfwd(a),jax.jacfwd(b)
            np.testing.assert_array_equal(a(d),b(d))


@pytest.mark.parametrize('field',[M.g,lambda d:M.contact(d,.25)])
def test_finite_at_origin_inside_and_on_boundary(field):
    points=jnp.array([[0.,0.],[M.r0*.2,M.r0*.3],[M.r0*.9,0.],
                      [M.r0*(1-1e-10),0.],[M.r0,0.],[M.r0*(1+1e-10),0.]])
    derivative=field
    for order in range(5):
        assert np.isfinite(jax.jit(jax.vmap(derivative))(points)).all(),order
        derivative=jax.jacfwd(derivative)
    np.testing.assert_array_equal(jax.grad(field)(jnp.zeros(2)),jnp.zeros(2))


def test_regression_distance_guard_rejects_inside_disc(outside_smoothing_discs):
    read=checked_jit(M.g)
    assert np.isfinite(read(jnp.array([2*M.r0,0.])))
    with pytest.raises(Exception,match='entered smoothing disc'):
        read(jnp.array([M.r0/2,0.]))
