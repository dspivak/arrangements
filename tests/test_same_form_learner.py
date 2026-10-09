"""Same form for a learner (book P41 (b), notes/SAME-FORM-LEARNER-PREDICTIONS-2026-10-09.md).

A line-learner holding what dataset D1 taught it is fed another dataset through its input port. The
form is the same for it when it does not have to move. Predictions are in the note, written first.
"""

import jax.numpy as jnp
import numpy as np

from arrangements.coalgebra import configuration
from arrangements.learning import forward_backward, parameterized_map, squared_error_forcing, train
from arrangements.rvect import euclidean


def _F(q, x):
    return q[:1].reshape(1, 1) @ x + q[1:]


def _learner(eta):
    return parameterized_map(_F, configuration(euclidean(2, eta)), in_dim=1, out_dim=1)


def _ols(x, y):
    b, a = np.polyfit(x, y, 1)
    return np.array([b, a])  # q = (slope, intercept)


def _residual_orth(x, r):
    """r made orthogonal to 1 and x."""
    A = np.column_stack([np.ones_like(x), x])
    return r - A @ np.linalg.lstsq(A, r, rcond=None)[0]


def _datasets():
    rng = np.random.default_rng(0)
    x1 = np.linspace(-2.0, 2.0, 12)
    y1 = 0.7 + 1.3 * x1 + 0.5 * rng.standard_normal(12)
    w1 = _ols(x1, y1)
    x2 = np.linspace(-1.5, 2.5, 9)
    y2 = w1[0] * x2 + w1[1] + _residual_orth(x2, 1.5 * rng.standard_normal(9))
    x3 = np.linspace(-2.0, 2.0, 11)
    y3 = w1[0] * x3 + w1[1] + 0.8 * (x3 ** 2 - (x3 ** 2).mean())
    x4 = np.linspace(-2.0, 2.0, 12)
    y4 = -0.5 + 0.4 * x4 + 0.5 * rng.standard_normal(12)
    return w1, {"D1": (x1, y1), "D2": (x2, y2), "D3": (x3, y3), "D4": (x4, y4)}


def _data(xy):
    x, y = xy
    return [(jnp.array([a]), jnp.array([b])) for a, b in zip(x, y)]


def _summed_xi_Q(arr, q, xy):
    """Sum over the dataset of the forcing pulled back to the parameter, by the arrangement's backward pass."""
    tot = np.zeros(2)
    q = jnp.asarray(q, dtype=float)
    for x, lam in _data(xy):
        y, _, _ = forward_backward(arr, q, x, jnp.zeros(1))
        _, xi_Q, _ = forward_backward(arr, q, x, squared_error_forcing(y, lam))
        tot += np.asarray(xi_Q)
    return tot


def _centre(arr, q, xy, passes):
    data = _data(xy)
    q, _ = train(arr, q, data, steps=passes * len(data))
    path = []
    for t in range(len(data)):
        q, _ = train(arr, q, [data[t]], steps=1)
        path.append(np.asarray(q))
    return q, np.mean(path, axis=0)


def test_lines_agree():
    w1, D = _datasets()
    for k in ("D2", "D3"):
        assert np.allclose(_ols(*D[k]), w1, atol=1e-10)
    assert np.linalg.norm(_ols(*D["D4"]) - w1) > 0.5


def test_full_pass_at_w1():
    """Prediction 1: the summed pulled-back forcing at w1 is zero for D2 and D3, not for D4."""
    w1, D = _datasets()
    arr = _learner(0.05)
    for k in ("D2", "D3"):
        f = _summed_xi_Q(arr, w1, D[k])
        print(k, "summed xi_Q at w1:", f)
        assert np.linalg.norm(f) < 1e-9
    f4 = _summed_xi_Q(arr, w1, D["D4"])
    print("D4 summed xi_Q at w1:", f4)
    assert np.linalg.norm(f4) > 1.0


def _switch(eta, k, passes=400):
    w1, D = _datasets()
    arr = _learner(eta)
    q_held, c1 = _centre(arr, jnp.zeros(2), D["D1"], passes)
    data_k = _data(D[k])
    q_after_one, _ = train(arr, q_held, data_k, steps=len(data_k))
    first_pass = np.linalg.norm(np.asarray(q_after_one) - np.asarray(q_held))
    _, ck = _centre(arr, q_held, D[k], passes)
    return w1, _ols(*D[k]), c1, ck, first_pass


def test_switch_same_line_does_not_move_the_centre():
    """Predictions 2 and 3, D2 and D3: the centre's shift and the first pass's motion shrink with eta."""
    for k in ("D2", "D3"):
        shifts, firsts = [], []
        for eta in (0.05, 0.01):
            w1, wk, c1, ck, first = _switch(eta, k, passes=400 if eta == 0.05 else 2000)
            shifts.append(np.linalg.norm(ck - c1))
            firsts.append(first)
            print(k, "eta", eta, "centre before", c1, "after", ck, "shift", shifts[-1], "first-pass motion", first)
        assert shifts[0] < 0.1
        assert shifts[1] < shifts[0]


def test_switch_other_line_moves():
    """Predictions 2 and 3, D4: the centre moves by about |w4 - w1| whatever eta."""
    for eta in (0.05, 0.01):
        w1, w4, c1, c4, first = _switch(eta, "D4", passes=400 if eta == 0.05 else 2000)
        print("D4 eta", eta, "centre before", c1, "after", c4, "|w4-w1|", np.linalg.norm(w4 - w1), "first-pass motion", first)
        assert abs(np.linalg.norm(c4 - c1) - np.linalg.norm(w4 - w1)) < 0.15
