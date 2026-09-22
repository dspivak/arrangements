"""Gradient descent and backpropagation via ``Phi`` (book sec.two_layer).

Three checks, plus the CDLM regression:
* the lens backward pass equals reverse-mode autodiff (backprop = chain rule),
  with the sign of the forcing: ``xi_Q = -grad k``;
* one ``Phi`` step on a Euclidean configuration parameter is vanilla gradient descent;
* the training loop actually fits a realizable regression (it runs, for real).
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.coalgebra import configuration
from arrangements.functors import Phi
from arrangements.interpretation import trivial_omega
from arrangements.learning import (
    forward_backward,
    parameterized_map,
    squared_error,
    squared_error_forcing,
    train,
)
from arrangements.rvect import euclidean

from tests.chain import load_regress


def _two_layer(h, m, n):
    """A tanh two-layer net F(q, x) = W2 tanh(W1 x + b1) + b2, params flattened into q."""
    shapes = [(h, m), (h,), (n, h), (n,)]
    sizes = [int(np.prod(s)) for s in shapes]
    dim = sum(sizes)

    def unpack(q):
        out, i = [], 0
        for s, k in zip(shapes, sizes):
            out.append(q[i:i + k].reshape(s))
            i += k
        return out

    def F(q, x):
        W1, b1, W2, b2 = unpack(q)
        return W2 @ jnp.tanh(W1 @ x + b1) + b2

    return F, dim


def _linear(m, n):
    def F(q, x):
        return q[: n * m].reshape(n, m) @ x + q[n * m:]
    return F, n * m + n


def test_backprop_matches_autodiff():
    """(xi_Q, xi_M) = (T F)^T omega equals minus jax.grad of the cost (the forcing is -dk)."""
    F, dim = _two_layer(4, 3, 2)
    arr = parameterized_map(F, configuration(euclidean(dim)), in_dim=3, out_dim=2)

    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.standard_normal(dim))
    x = jnp.asarray(rng.standard_normal(3))
    lam = jnp.asarray(rng.standard_normal(2))

    y0 = F(q, x)
    y, xi_Q, xi_M = forward_backward(arr, q, x, squared_error_forcing(y0, lam))

    np.testing.assert_allclose(np.asarray(y), np.asarray(y0), atol=1e-12)
    grad_q = jax.grad(lambda qq: squared_error(F(qq, x), lam))(q)
    grad_x = jax.grad(lambda xx: squared_error(F(q, xx), lam))(x)
    np.testing.assert_allclose(np.asarray(xi_Q), -np.asarray(grad_q), atol=1e-9)
    np.testing.assert_allclose(np.asarray(xi_M), -np.asarray(grad_x), atol=1e-9)


def test_phi_step_is_gradient_descent():
    """sec.displacement with sharp = eta * I: new q = q + eta * (-grad k) = q - eta * grad k."""
    F, dim = _two_layer(4, 3, 2)
    eta = 0.05
    arr = parameterized_map(F, configuration(euclidean(dim, eta)), in_dim=3, out_dim=2)
    O = Phi(arr)

    rng = np.random.default_rng(1)
    q = jnp.asarray(rng.standard_normal(dim))
    x = jnp.asarray(rng.standard_normal(3))
    lam = jnp.asarray(rng.standard_normal(2))

    in_pos = (x, trivial_omega(0))
    _, _, new_q = O.with_state(q).run_one(
        in_pos, lambda op: (squared_error_forcing(op[0], lam), jnp.zeros(0))
    )
    grad_q = jax.grad(lambda qq: squared_error(F(qq, x), lam))(q)
    np.testing.assert_allclose(np.asarray(new_q), np.asarray(q - eta * grad_q), atol=1e-9)


def test_training_fits_realizable_regression():
    """The training loop converges: a linear model fits realizable data to ~0 loss."""
    m, n = 3, 2
    F, dim = _linear(m, n)

    rng = np.random.default_rng(2)
    W_true = jnp.asarray(rng.standard_normal((n, m)))
    b_true = jnp.asarray(rng.standard_normal(n))
    data = []
    for _ in range(20):
        x = jnp.asarray(rng.standard_normal(m))
        data.append((x, W_true @ x + b_true))

    arr = parameterized_map(F, configuration(euclidean(dim, eta=0.05)), in_dim=m, out_dim=n)
    q, history = train(arr, jnp.zeros(dim), data, steps=5000)

    full_batch_loss = float(np.mean([float(squared_error(F(q, x), lam)) for x, lam in data]))
    assert full_batch_loss < 1e-4
    assert np.mean(history[-50:]) < np.mean(history[:50])


def test_regression_matches_cdlm():
    """``regress/gradient_descent.npz``: backprop covectors negated, parameters identical."""
    R = load_regress("gradient_descent")
    F, dim = _two_layer(4, 3, 2)

    arr = parameterized_map(F, configuration(euclidean(dim)), in_dim=3, out_dim=2)
    q, x, lam = (jnp.asarray(R[k]) for k in ("bp_q", "bp_x", "bp_lam"))
    y, xi_Q, xi_M = forward_backward(arr, q, x, squared_error_forcing(F(q, x), lam))
    np.testing.assert_allclose(y, R["bp_y"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(xi_Q, -R["bp_xi_Q"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(xi_M, -R["bp_xi_M"], rtol=1e-10, atol=1e-12)

    eta = float(R["step_eta"])
    arr2 = parameterized_map(F, configuration(euclidean(dim, eta)), in_dim=3, out_dim=2)
    q, x, lam = (jnp.asarray(R[k]) for k in ("step_q", "step_x", "step_lam"))
    _, _, new_q = Phi(arr2).with_state(q).run_one(
        (x, trivial_omega(0)), lambda op: (squared_error_forcing(op[0], lam), jnp.zeros(0)))
    np.testing.assert_allclose(new_q, R["step_new_q"], rtol=1e-10, atol=1e-12)

    m, n = 3, 2
    F3, dim3 = _linear(m, n)
    data = list(zip(jnp.asarray(R["train_xs"]), jnp.asarray(R["train_lams"])))
    arr3 = parameterized_map(F3, configuration(euclidean(dim3, eta=0.05)), in_dim=m, out_dim=n)
    q, history = train(arr3, jnp.zeros(dim3), data, steps=5000)
    np.testing.assert_allclose(q, R["train_q"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.array(history), R["train_history"], rtol=1e-10, atol=1e-12)
