"""Book Ch 5 ``sec.modes``: one chain with two matters, and the swing (Claude Opus 5.5, d4e65d6e, 2026-11-05).

The library has no tangent arrangements, so a solution is approached by running an arrangement at shrinking durations.
Every arrangement here is built in ``arr`` and run by the dynamics functor (``Phi`` and ``PhiK``); the expected curves
are the book's formulas. The push arrives through a part with a clock, label ``(ports . (clock || swing))``.

Predictions, written before the first run: (a) three rod segments started at a shape with stiffness eigenvalue ``mu``
keep the shape and scale by ``e^{-mu t}``; (b) three masses on the same wiring started at that shape keep it and swing as
``cos(sqrt(mu) t)``; (c) a mass on a spring from rest, pushed by ``cos(nu t)`` at its own rate, follows
``t sin(nu t) / (2 m nu)``; (d) pushed at ``0.7 nu`` from the mode's state, it follows ``cos(w t) / (kappa - m w^2)``;
(e) with drag ``gamma``, pushed at ``nu``, it settles to amplitude ``1 / (gamma nu)``. In each, the error against the
formula halves with the duration, except (e), where it is read at the settled amplitude.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact, zero_form
from arrangements.coalgebra import CotCoalgK, configuration, phase
from arrangements.functors import closed_step
from arrangements.hopfield import _tensor_balanced
from arrangements.interpretation import trivial_omega
from arrangements.pcK import PhiK
from arrangements.rvect import diagonal
from arrangements.wiring import boxes_and_couplers, compose_seq, trivial_parameter

Z = jnp.zeros(0)
IN = (Z, trivial_omega(0))
TRIV = lambda _o: (jnp.zeros(0), jnp.zeros(0))

SHAPES = {1.0: np.array([1.0, 0.0, -1.0]), 3.0: np.array([1.0, -2.0, 1.0])}


def flat(s):
    return np.concatenate([np.ravel(x) for x in jax.tree_util.tree_leaves(s)])


def joint():
    return Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=2, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m: Z, in_f=lambda q, m, n: Z,
        omega=exact(lambda q, m, n: -0.5 * (m[0] - m[1]) ** 2), label="joint",
    )


def rod(th0):
    return Arrangement(
        P=configuration(diagonal(jnp.array([1.0])), init=jnp.array([th0])),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m: q, in_f=lambda q, m, n: Z, omega=zero_form, label="rod",
    )


def mass(q0):
    return Arrangement(
        P=phase(diagonal(jnp.array([1.0])), init=(jnp.array([q0]), jnp.array([0.0]))),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m: q, in_f=lambda q, m, n: Z, omega=zero_form, label="mass",
    )


def chain(part, shape):
    """One wiring, one cost: three parts and two joints of stiffness 1."""
    return boxes_and_couplers([part(float(x)) for x in shape], [(0, 1), (1, 2)], [joint(), joint()])


def run_chain(arr, total, n):
    step = closed_step(arr, check=2, duration=total / n)
    s = arr.P.init
    for _ in range(n):
        s = step(s)
    return s


def test_rods_keep_the_shape_and_decay():
    total = 1.2
    for mu, shape in SHAPES.items():
        expected = np.exp(-mu * total) * shape
        errors = [np.linalg.norm(flat(run_chain(chain(rod, shape), total, n)) - expected) for n in (200, 400, 800)]
        assert errors[2] < 1e-2
        np.testing.assert_allclose(errors[0] / errors[1], 2.0, atol=0.1)
        np.testing.assert_allclose(errors[1] / errors[2], 2.0, atol=0.1)


def test_masses_keep_the_same_shape_and_swing():
    total = 1.2
    for mu, shape in SHAPES.items():
        expected = np.cos(np.sqrt(mu) * total) * shape
        errors = []
        for n in (200, 400, 800):
            arr = chain(mass, shape)
            s = run_chain(arr, total, n)
            q = flat(s)[[0, 2, 4]] if flat(s).size == 6 else flat(s)[:3]
            errors.append(np.linalg.norm(q - expected))
        assert errors[2] < 1e-2
        np.testing.assert_allclose(errors[0] / errors[1], 2.0, atol=0.25)
        np.testing.assert_allclose(errors[1] / errors[2], 2.0, atol=0.25)


def swing_with_a_push(x0, p0, m, kappa, gamma, rate, tau):
    """A mass on a spring, with drag ``gamma``, whose output port receives ``cos(rate t)`` from a part with a clock."""
    matter = CotCoalgK(Q_dim=1, init=jnp.array([x0, p0]), reads=(lambda s: s[:1] + tau * s[1:] / m,),
                       advances=(lambda s, xi: jnp.concatenate([s[:1] + tau * s[1:] / m,
                                                                s[1:] + tau * (xi - gamma * s[1:] / m)]),),
                       label="swing")
    swing = Arrangement(P=matter, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1, out_f=lambda q, m_: q,
                        in_f=lambda q, m_, n: Z, omega=exact(lambda q, m_, n: -0.5 * kappa * q[0] ** 2), label="swing")
    clock_P = CotCoalgK(Q_dim=1, init=jnp.array([0.0]), reads=(lambda t: t + tau,), advances=(lambda t, xi: t + tau,),
                        label="clock")

    def clock_omega(q, m_, n):
        return (jnp.zeros(1), Z, jnp.cos(rate * q[0]) * jnp.ones(1))

    clock = Arrangement(P=clock_P, in_dim_M=0, out_dim_M=0, in_dim_N=1, out_dim_N=0, out_f=lambda q, m_: Z,
                        in_f=lambda q, m_, n: Z, omega=clock_omega, label="clock")
    wire = Arrangement(P=trivial_parameter(), in_dim_M=1, out_dim_M=1, in_dim_N=0, out_dim_N=0, out_f=lambda q, m_: Z,
                       in_f=lambda q, m_, n: m_, omega=zero_form, label="ports")
    arr = compose_seq(_tensor_balanced([clock, swing]), wire)
    assert arr.label == "(ports . (clock || swing))"
    return arr


def positions(arr, n):
    """The swing's position at each round: the state is (clock, position, momentum), flat."""
    O = PhiK(arr)
    f = jax.jit(lambda s: O.with_state(s).run_one(IN, TRIV)[1])
    s = O.state
    out = []
    for _ in range(n + 1):
        out.append(np.concatenate([np.ravel(np.asarray(x, float)) for x in jax.tree_util.tree_leaves(s)
                                   if np.asarray(x).size])[1])
        s = f(s)
    return np.array(out)


M, KAPPA = 2.0, 3.0
NU = float(np.sqrt(KAPPA / M))


def test_pushed_at_its_own_rate_it_goes_higher_and_higher():
    total = 40.0
    errors = []
    for n in (4000, 8000, 16000):
        tau = total / n
        x = positions(swing_with_a_push(0.0, 0.0, M, KAPPA, 0.0, NU, tau), n)
        t = np.arange(n + 1) * tau
        errors.append(np.abs(x - t * np.sin(NU * t) / (2 * M * NU)).max())
    assert errors[2] < 0.1
    np.testing.assert_allclose(errors[0] / errors[1], 2.0, atol=0.2)
    np.testing.assert_allclose(errors[1] / errors[2], 2.0, atol=0.2)
    # the amplitude keeps rising: the largest swing in each of two windows, against tau / (2 m nu)
    t = np.arange(16001) * (total / 16000)
    assert np.abs(x[t > 30]).max() > 2.5 * np.abs(x[t < 15]).max()
    np.testing.assert_allclose(np.abs(x).max(), total / (2 * M * NU), rtol=0.03)


def test_pushed_at_the_wrong_rate_it_stays_low():
    rate = 0.7 * NU
    gain = 1.0 / (KAPPA - M * rate**2)
    total = 40.0
    errors = []
    for n in (4000, 8000, 16000):
        tau = total / n
        x = positions(swing_with_a_push(gain, 0.0, M, KAPPA, 0.0, rate, tau), n)
        t = np.arange(n + 1) * tau
        errors.append(np.abs(x - gain * np.cos(rate * t)).max())
    assert errors[2] < 0.05
    np.testing.assert_allclose(errors[0] / errors[1], 2.0, atol=0.3)
    np.testing.assert_allclose(errors[1] / errors[2], 2.0, atol=0.3)
    assert np.abs(x).max() < 1.1 * gain < 40.0 / (2 * M * NU)


def test_drag_makes_the_gain_finite():
    gamma = 0.4
    total, n = 200.0, 40000
    tau = total / n
    x = positions(swing_with_a_push(0.0, 0.0, M, KAPPA, gamma, NU, tau), n)
    t = np.arange(n + 1) * tau
    np.testing.assert_allclose(np.abs(x[t > 170]).max(), 1.0 / (gamma * NU), rtol=0.01)
