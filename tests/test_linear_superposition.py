"""Book Ch 5 ``sec.linear_tangent``: solutions of linear arrangements add (Claude Opus 5.5, 18e9ed1c, 2026-11-05).

The library has no tangent arrangements, so a solution is approached by running an arrangement at shrinking
durations, as in ``tests/test_solutions_waiting.py``: each arrangement here is built in ``arr`` and run by the
dynamics functor ``Phi``. The integrators are the rod segment and the mass; the coupling is linear. One step of such an
arrangement is already linear in the state and the port data, so superposition is checked exactly at each duration.

Predictions, written before the first run: (a) on a composite with two inputs, a rod segment and a mass under one
coupler, the run from ``a s + b s'`` under the push ``a l + b l'`` is ``a`` times the first run plus ``b`` times the
second, at every step, up to rounding; (b) the same with an identity composed in, so the unit changes nothing;
(c) the closed composite's state approaches ``expm(tau G)`` with error that halves with the duration, where ``G`` is
its differential equation as a matrix; (d) a gain, out-transfer ``(x, q) |-> q x``, is linear in ``x`` at each ``q`` and
fails (a), by a factor of ``2`` when both ``x`` and the load are doubled; (e) a quadratic cost ``x^2 / 2`` under an
in-transfer ``n |-> n^2`` has a composite effort field that is not linear, ``8`` times at twice the point.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact, zero_form
from arrangements.coalgebra import configuration, phase
from arrangements.functors import Phi, closed_step
from arrangements.interpretation import trivial_omega
from arrangements.rvect import diagonal
from arrangements.wiring import compose_seq, tensor_arrangements, trivial_parameter

Z = jnp.zeros(0)
IN = (Z, trivial_omega(0))
C, M, K = 2.0, 0.5, 3.0


def rod():
    return Arrangement(
        P=configuration(diagonal(jnp.array([1.0 / C])), init=jnp.array([0.0])),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m: q, in_f=lambda q, m, n: Z, omega=zero_form, label="rod",
    )


def mass():
    return Arrangement(
        P=phase(diagonal(jnp.array([1.0 / M])), init=(jnp.array([0.0]), jnp.array([0.0]))),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m: q, in_f=lambda q, m, n: Z, omega=zero_form, label="mass",
    )


def spring_and_port():
    """The coupler: a spring between its two inputs, and an output port showing the mass's position."""
    return Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=2, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m: m[1:], in_f=lambda q, m, n: Z,
        omega=exact(lambda q, m, n: -0.5 * K * (m[0] - m[1]) ** 2), label="spring_and_port",
    )


def identity():
    return Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=1, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m: m, in_f=lambda q, m, n: Z, omega=zero_form, label="id",
    )


def body():
    return compose_seq(tensor_arrangements([rod(), mass()]), spring_and_port())


def lin(a, s, b, t):
    return jax.tree_util.tree_map(lambda x, y: a * x + b * y, s, t)


def flat(s):
    return np.concatenate([np.ravel(x) for x in jax.tree_util.tree_leaves(s)])


def like(arr, values):
    """The state of ``arr`` with the given coordinates, in the order ``flat`` reads them."""
    leaves, tree = jax.tree_util.tree_flatten(arr.P.init)
    out, i = [], 0
    for x in leaves:
        out.append(jnp.array(values[i:i + x.size]).reshape(x.shape))
        i += x.size
    return jax.tree_util.tree_unflatten(tree, out)


def run_pushed(arr, s, pushes, dur):
    """States and output values along a run under the given loads at the output port."""
    O = Phi(arr)
    states, outs = [], []
    for load in pushes:
        out, _, s = O.with_state(s).run_one(IN, lambda _o: (load, Z), dur)
        states.append(flat(s))
        outs.append(np.ravel(out[0] if isinstance(out, tuple) else out))
    return np.array(states), np.array(outs)


def check_superposition(arr):
    a, b, n = 2.0, -3.0, 25
    s0, s1 = like(arr, [0.3, -1.0, 0.7]), like(arr, [1.1, 0.4, -0.2])
    t = np.arange(n)
    push0 = [jnp.array([np.sin(0.3 * k)]) for k in t]
    push1 = [jnp.array([np.cos(0.11 * k) - 0.2]) for k in t]
    both = [a * p + b * r for p, r in zip(push0, push1)]
    for dur in (0.01, 0.1):
        st0, out0 = run_pushed(arr, s0, push0, dur)
        st1, out1 = run_pushed(arr, s1, push1, dur)
        st, out = run_pushed(arr, lin(a, s0, b, s1), both, dur)
        np.testing.assert_allclose(st, a * st0 + b * st1, atol=1e-9)
        np.testing.assert_allclose(out, a * out0 + b * out1, atol=1e-9)
        assert np.abs(st0 - st1).max() > 1e-3  # the two runs differ, so the check says something


def test_branching_composite_superposes_at_every_step():
    arr = body()
    assert "rod" in arr.label and "mass" in arr.label and "spring_and_port" in arr.label
    check_superposition(arr)


def test_the_unit_keeps_superposition_and_changes_nothing():
    plain, with_unit = body(), compose_seq(body(), identity())
    check_superposition(with_unit)
    pushes = [jnp.array([0.5])] * 10
    st_a, _ = run_pushed(plain, like(plain, [0.3, -1.0, 0.7]), pushes, 0.05)
    st_b, _ = run_pushed(with_unit, like(with_unit, [0.3, -1.0, 0.7]), pushes, 0.05)
    np.testing.assert_allclose(st_a, st_b, atol=1e-12)


def test_closed_composite_approaches_the_exponential_of_its_matrix():
    """The rod's temperature, the mass's position, the mass's momentum, under one spring."""
    G = jnp.array([[-K / C, K / C, 0.0],
                   [0.0, 0.0, 1.0 / M],
                   [K, -K, 0.0]])
    s0, tau = np.array([0.3, -1.0, 0.7]), 1.3
    target = np.array(jax.scipy.linalg.expm(tau * G) @ s0)
    spring = Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=2, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m: Z, in_f=lambda q, m, n: Z,
        omega=exact(lambda q, m, n: -0.5 * K * (m[0] - m[1]) ** 2), label="spring",
    )
    arr = compose_seq(tensor_arrangements([rod(), mass()]), spring)
    errors = []
    for n in (800, 1600, 3200):
        step = closed_step(arr, check=2, duration=tau / n)
        s = like(arr, s0)
        for _ in range(n):
            s = step(s)
        errors.append(np.linalg.norm(flat(s) - target))
    assert errors[2] < errors[1] < errors[0]
    np.testing.assert_allclose(errors[0] / errors[1], 2.0, atol=0.15)
    np.testing.assert_allclose(errors[1] / errors[2], 2.0, atol=0.15)


def test_a_gain_is_not_in_the_class():
    """Linear in x at each q, not jointly: doubling the input and the load quadruples the response."""
    gain = Arrangement(
        P=configuration(diagonal(jnp.array([1.0])), init=jnp.array([0.0])),
        in_dim_M=0, out_dim_M=1, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m: q * m, in_f=lambda q, m, n: Z, omega=zero_form, label="gain",
    )
    O, dur = Phi(gain), 0.1

    def after(x, load):
        _, _, s = O.with_state(jnp.array([0.0])).run_one((jnp.array([x]), trivial_omega(0)),
                                                        lambda _o: (jnp.array([load]), Z), dur)
        return flat(s)[0]

    one, two = after(1.0, 1.0), after(2.0, 2.0)
    assert abs(one) > 1e-6
    np.testing.assert_allclose(two, 4 * one, atol=1e-12)


def test_a_square_in_transfer_makes_a_quadratic_cost_quartic():
    """``x^2 / 2`` through ``n |-> n^2`` is ``n^4 / 2``, whose effort field at ``2 n`` is ``8`` times that at ``n``."""
    cost = Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=0, in_dim_N=1, out_dim_N=0,
        out_f=lambda q, m: Z, in_f=lambda q, m, n: Z,
        omega=exact(lambda q, m, n: -0.5 * n[0] ** 2), label="cost",
    )
    square = Arrangement(
        P=trivial_parameter(), in_dim_M=1, out_dim_M=0, in_dim_N=1, out_dim_N=0,
        out_f=lambda q, m: Z, in_f=lambda q, m, n: n ** 2, omega=zero_form, label="square",
    )
    composite = compose_seq(cost, square)
    at = lambda v: float(composite.omega(Z, Z, jnp.array([v]))[-1][0])
    np.testing.assert_allclose(at(1.0), -2.0, atol=1e-12)
    np.testing.assert_allclose(at(2.0), 8 * at(1.0), atol=1e-12)
