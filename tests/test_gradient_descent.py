"""Gradient descent and backpropagation via ``Phi`` (book sec.neural_network).

The book's run is the arrangement it draws (``network.drawn_network``; plan P43 (b) job 1): a white box
per parameter, the datum's and the label's boxes stepping through a list of examples, gray ``x``, ``+``
and ``tanh`` boxes, the wiring's copies, and the loss as a box, composed by ``compose_seq`` and
``tensor_arrangements`` and run by ``Phi``. Nothing is computed outside it. Its checks:
* the book's step, box by box: each box's own ``Phi`` passes the force back, and every wire of the
  book's picture carries the force printed on it; the composite's ``Phi`` step is eqn.nn_step;
* the drawn network's forces are minus the partial derivatives a standard library (``jax.grad``)
  computes, and equal the one-box run's, for a two-layer network with several parameters per layer;
* repeated ``Phi`` steps of the drawn network equal the one-box training loop step for step, and fit
  the two-layer network to its examples.

The one-box run (``learning.parameterized_map`` of the whole map, its loss's covector computed in code
by ``learning.train_by_port``) is kept as the check; its own tests follow, with the CDLM regression.
``learning.train`` itself now runs the closed training arrangement, the learner wired to datum, label
and loss boxes (P43 (b) job 5); ``test_train_is_the_port_fed_loop`` checks it equal to the old loop.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.coalgebra import configuration
from arrangements.functors import Phi, closed_step
from arrangements.interpretation import trivial_omega
from arrangements.learning import (
    forward_backward,
    parameterized_map,
    squared_error,
    squared_error_forcing,
    train,
    train_by_port,
)
from arrangements.network import drawn_network, loss_box, parameter_box, example_box
from arrangements.network import parameters, plus_box, tanh_box, times_box
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


_IN = (jnp.zeros(0), trivial_omega(0))
_NONE = jnp.zeros(0)


def _closed_step(A):
    """One ``Phi`` step of a closed arrangement."""
    _, _, s = Phi(A).run_one(_IN, lambda _o: (_NONE, _NONE))
    return s


def _open(box, state, values):
    """A box's own ``Phi`` at its state, heard ``values``: (what it presents, its backward pass)."""
    _act, fiber = Phi(box).step(state)
    out_pos, at_pos = fiber((jnp.asarray(values, dtype=float), trivial_omega(0)))
    return out_pos[0], lambda xi: at_pos((jnp.asarray(xi, dtype=float), _NONE))


# ---------------------------------------------------------------------------
# The network as drawn (the book's run).
# ---------------------------------------------------------------------------


def test_book_step_box_by_box():
    """sec.neural_network's step: weights (1, -1, 2), datum 1, label 1, learning rate 1/10.

    Each box is run by its own ``Phi``, values forward and then forces back one box at a time; every
    wire of the book's picture carries the force printed on it, the loss's (1, -1) to the answer and
    the label, then (2, 2, 0) on the three parameters. The composite's ``Phi`` step agrees, moving the
    parameters to (1.2, -0.8, 2), and the answer on the same datum becomes 2 tanh(0.4) ~ 0.76.
    """
    eta = 0.1
    w = [parameter_box(v, eta) for v in (1.0, -1.0, 2.0)]
    datum, label = example_box([[1.0]]), example_box([[1.0]])
    times1, plus, tanh, times2, loss = times_box(), plus_box(), tanh_box(), times_box(), loss_box()

    # values forward
    ws, at_w = zip(*[_open(b, b.P.init, _NONE) for b in w])
    x, _ = _open(datum, datum.P.init, _NONE)
    lab, _ = _open(label, label.P.init, _NONE)
    triv = times1.P.init
    p1, at_times1 = _open(times1, triv, jnp.concatenate([ws[0], x]))
    z, at_plus = _open(plus, triv, jnp.concatenate([ws[1], p1]))
    a, at_tanh = _open(tanh, triv, z)
    y, at_times2 = _open(times2, triv, jnp.concatenate([ws[2], a]))
    _, at_loss = _open(loss, triv, jnp.concatenate([y, lab]))
    np.testing.assert_allclose([p1[0], z[0], a[0], y[0]], [1.0, 0.0, 0.0, 0.0], atol=1e-15)

    # forces back, box by box
    (xi_loss, _), _ = at_loss(_NONE)
    np.testing.assert_allclose(xi_loss, [1.0, -1.0], atol=1e-15)          # answer, label
    (xi_t2, _), _ = at_times2(xi_loss[:1])
    np.testing.assert_allclose(xi_t2, [0.0, 2.0], atol=1e-15)             # w3, tanh's output
    (xi_tanh, _), _ = at_tanh(xi_t2[1:])
    np.testing.assert_allclose(xi_tanh, [2.0], atol=1e-15)                # +'s output
    (xi_plus, _), _ = at_plus(xi_tanh)
    np.testing.assert_allclose(xi_plus, [2.0, 2.0], atol=1e-15)           # w2, first x's output
    (xi_t1, _), _ = at_times1(xi_plus[1:])
    np.testing.assert_allclose(xi_t1, [2.0, 2.0], atol=1e-15)             # w1, datum

    forces = [xi_t1[0], xi_plus[0], xi_t2[0]]
    stepped = [at(jnp.array([f]))[1] for at, f in zip(at_w, forces)]
    np.testing.assert_allclose(jnp.concatenate(stepped), [1.2, -0.8, 2.0], atol=1e-15)

    # the composite, run by Phi
    A = drawn_network(1, 1, 1, [1.0, -1.0, 2.0], eta, [[1.0]], [[1.0]], out_bias=False)
    s = _closed_step(A)
    np.testing.assert_allclose(parameters(A, s, 1, 3), [1.2, -0.8, 2.0], atol=1e-15)
    w1, w2, w3 = parameters(A, s, 1, 3)
    np.testing.assert_allclose(w3 * jnp.tanh(w1 * 1.0 + w2), 2 * np.tanh(0.4), atol=1e-15)
    assert abs(float(2 * np.tanh(0.4)) - 0.76) < 0.005


def test_drawn_forces_are_minus_partials_and_match_one_box():
    """A two-layer network with several parameters per layer (h=4, m=3, n=2: 26 parameter boxes).

    With learning rate 1 one ``Phi`` step of the drawn arrangement moves each parameter by the force
    passed back to it; that force is minus the partial derivative ``jax.grad`` computes, and the
    one-box run's ``xi_Q``. Also at the CDLM recording's point and step.
    """
    h, m, n = 4, 3, 2
    F, dim = _two_layer(h, m, n)
    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.standard_normal(dim))
    x = jnp.asarray(rng.standard_normal(m))
    lam = jnp.asarray(rng.standard_normal(n))

    A = drawn_network(h, m, n, q, 1.0, x[None], lam[None])
    xi = parameters(A, _closed_step(A), m, dim) - q
    grad_q = jax.grad(lambda qq: squared_error(F(qq, x), lam))(q)
    np.testing.assert_allclose(np.asarray(xi), -np.asarray(grad_q), atol=1e-12)

    one_box = parameterized_map(F, configuration(euclidean(dim)), in_dim=m, out_dim=n)
    _, xi_Q, _ = forward_backward(one_box, q, x, squared_error_forcing(F(q, x), lam))
    np.testing.assert_allclose(np.asarray(xi), np.asarray(xi_Q), atol=1e-12)

    R = load_regress("gradient_descent")
    q, x, lam = (jnp.asarray(R[k]) for k in ("bp_q", "bp_x", "bp_lam"))
    A = drawn_network(h, m, n, q, 1.0, x[None], lam[None])
    np.testing.assert_allclose(parameters(A, _closed_step(A), m, dim) - q, -R["bp_xi_Q"],
                               rtol=1e-10, atol=1e-12)
    eta = float(R["step_eta"])
    q, x, lam = (jnp.asarray(R[k]) for k in ("step_q", "step_x", "step_lam"))
    A = drawn_network(h, m, n, q, eta, x[None], lam[None])
    np.testing.assert_allclose(parameters(A, _closed_step(A), m, dim), R["step_new_q"],
                               rtol=1e-10, atol=1e-12)


def test_drawn_linear_training_equals_one_box_and_cdlm():
    """The CDLM recording's training run (linear model, 20 examples, rate 0.05, 5000 steps), drawn.

    The drawn arrangement (an ``x`` box per entry of ``W``, a ``+`` per output, datum, label and loss
    boxes), run by ``Phi`` (compiled, checked against ``Phi`` on its first steps), ends at the one-box
    run's parameters and the CDLM recording's, and its cost at each step, read off its state, is
    the recorded history.
    """
    R = load_regress("gradient_descent")
    m, n = 3, 2
    F3, dim3 = _linear(m, n)
    xs, lams = np.asarray(R["train_xs"]), np.asarray(R["train_lams"])
    A = drawn_network(0, m, n, jnp.zeros(dim3), 0.05, xs, lams, hidden=False)
    step = closed_step(A)

    def body(s, _):
        pos = A.P.position(s)
        x, q, lam = pos[:m], pos[m:m + dim3], pos[m + dim3:]
        return step(s), squared_error(F3(q, x), lam)

    s, history = jax.lax.scan(body, A.P.init, None, length=5000)
    np.testing.assert_allclose(parameters(A, s, m, dim3), R["train_q"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.asarray(history), R["train_history"], rtol=1e-10, atol=1e-12)


def test_drawn_two_layer_trains_like_one_box_and_fits():
    """Repeated ``Phi`` steps fit the two-layer network (h=4, m=3, n=2) to its examples.

    Twenty examples from a teacher network of the same shape; start 0.5 x normal; rate 0.05. The
    drawn arrangement's first 2000 steps equal the one-box training loop's to 1e-12, and after
    20000 steps its full-batch cost is below 1% of where it started (1.84 to 0.0069 at seed 3).
    """
    h, m, n = 4, 3, 2
    F, dim = _two_layer(h, m, n)
    rng = np.random.default_rng(3)
    teacher = jnp.asarray(rng.standard_normal(dim))
    xs = rng.standard_normal((20, m))
    lams = np.stack([np.asarray(F(teacher, jnp.asarray(x))) for x in xs])
    q0 = jnp.asarray(0.5 * rng.standard_normal(dim))
    eta = 0.05

    A = drawn_network(h, m, n, q0, eta, xs, lams)
    step = closed_step(A)
    run = jax.jit(lambda s, k: jax.lax.fori_loop(0, k, lambda _i, t: step(t), s), static_argnums=1)
    s_2000 = run(A.P.init, 2000)

    one_box = parameterized_map(F, configuration(euclidean(dim, eta)), in_dim=m, out_dim=n)
    q_one, _ = train_by_port(one_box, q0, list(zip(xs, lams)), steps=2000)
    np.testing.assert_allclose(parameters(A, s_2000, m, dim), q_one, atol=1e-12)

    def cost(q):
        return float(np.mean([float(squared_error(F(q, jnp.asarray(x)), jnp.asarray(l)))
                              for x, l in zip(xs, lams)]))

    s_end = run(s_2000, 18000)
    assert cost(parameters(A, s_end, m, dim)) < 0.01 * cost(q0)


# ---------------------------------------------------------------------------
# The one-box run, kept as the check.
# ---------------------------------------------------------------------------


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


def test_train_is_the_port_fed_loop():
    """``train`` (datum, label and loss boxes, run by ``Phi``) equals ``train_by_port`` (the loss's
    covector in code) step for step: the CDLM recording's linear run, 5000 steps, and 300 steps of a
    two-layer net. The prediction, not a premise."""
    R = load_regress("gradient_descent")
    F3, dim3 = _linear(3, 2)
    data = list(zip(jnp.asarray(R["train_xs"]), jnp.asarray(R["train_lams"])))
    arr3 = parameterized_map(F3, configuration(euclidean(dim3, eta=0.05)), in_dim=3, out_dim=2)
    q_a, h_a = train(arr3, jnp.zeros(dim3), data, steps=5000)
    q_b, h_b = train_by_port(arr3, jnp.zeros(dim3), data, steps=5000)
    np.testing.assert_allclose(q_a, q_b, atol=1e-12)
    np.testing.assert_allclose(h_a, h_b, atol=1e-12)
    F, dim = _two_layer(4, 3, 2)
    arr = parameterized_map(F, configuration(euclidean(dim, eta=0.05)), in_dim=3, out_dim=2)
    q0 = jnp.asarray(np.random.default_rng(5).standard_normal(dim))
    np.testing.assert_allclose(train(arr, q0, data, steps=300)[0],
                               train_by_port(arr, q0, data, steps=300)[0], atol=1e-12)


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
