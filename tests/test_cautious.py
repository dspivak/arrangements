"""Cautious integrators: Ch 3 ``ex.kelvin_rod`` (with ``exc.kelvin_tensor``), Ch 3 ``ex.reaction_network`` as the book
draws it, and Ch 5 ``ex.affine_stepping`` (``arrangements/cautious.py``, ``reaction_network.book_reaction_network``).

Every system is an arrangement run by the dynamics functor: ``open_step`` and ``closed_step`` are ``Phi``'s step,
checked against it on the steps that are allowed. The reaction network is built with ``compose_seq`` from species boxes,
transition boxes and a wire carrying each species' value to each of its arrows, and its label is read.

Predictions, written before the first run (Claude Opus 5.5, 00f2d232, 2026-11-12):
(a) a rod segment in kelvin, heat capacity 4 at 2 kelvin, told an outflow of 1, allows a step of 6 (to 1/2 kelvin)
    and not one of 10, the bound being ``C theta / (-load) = 8``; told an inflow of 1 it allows 10 and 100;
(b) beside a second segment, capacity 1 at 3 kelvin, also told an outflow of 1, the tensor allows 2.5 and not 5:
    the second segment, whose bound 3 is the smaller, decides (``exc.kelvin_tensor``);
(c) the net of the book's picture, ``A + B -> C`` at ``kappa_1`` and ``C -> A`` at ``kappa_2``, from ``(1, 0.8, 0.5)``
    with ``kappa = (0.6, 0.3)``, takes the step ``eqn.reaction_network_step`` at durations 1/4 and 1, keeps
    ``q_A + q_C``, agrees with encoding 7 of ``reaction_network.py`` there, and does not allow a step of 3, at which
    ``q_B`` would be negative;
(d) the Euler integrator of the integrand ``(q, load -> load)`` on ``R_{>0}`` is each species, step for step; its
    step at duration 0 is the state and its derivative there is the tangent response (``prop.euler_derivative``); and
    an affine map carries ``s + tau w`` to ``g(s) + tau A w`` (``ex.affine_stepping``).

*Reasonable or weird* (the agent's judgment): a temperature that may not reach absolute zero, and concentrations that
may not go negative, are reasonable at every scale they are read at; what is weird is that the refusal is the step's,
not the world's: a long enough step is refused although the true flow stays positive for all time (the footnote of
``ex.reaction_network``), so a refusal says the step is too long, not that the chemistry fails.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.cautious import allowed, euler_integrator, kelvin_rod, positive, species
from arrangements.functors import closed_step, open_step
from arrangements.reaction_network import book_reaction_network, composed_mass_action_arrangement
from arrangements.wiring import tensor_arrangements


def step_rods(rods, loads, tau, ok):
    arr = rods[0] if len(rods) == 1 else tensor_arrangements(rods)
    return open_step(arr, [], loads, check=1 if ok else 0, duration=tau)(arr.P.init)


def test_kelvin_rod_allows_only_steps_that_stay_above_zero():
    """Prediction (a)."""
    rod = lambda: kelvin_rod(4.0, 2.0)
    s = step_rods([rod()], [-1.0], 6.0, True)
    assert allowed(s) and np.allclose(s, 0.5)
    assert not allowed(step_rods([rod()], [-1.0], 10.0, False))
    for tau in (10.0, 100.0):
        s = step_rods([rod()], [1.0], tau, True)
        assert allowed(s) and np.allclose(s, 2.0 + tau / 4.0)


def test_tensor_allows_a_step_when_both_do():
    """Prediction (b), ``prop.cautious_tensor`` and ``exc.kelvin_tensor``."""
    rods = lambda: [kelvin_rod(4.0, 2.0), kelvin_rod(1.0, 3.0)]
    s = step_rods(rods(), [-1.0, -1.0], 2.5, True)
    assert allowed(s)
    np.testing.assert_allclose([float(s[0][0]), float(s[1][0])], [2.0 - 2.5 / 4.0, 3.0 - 2.5], atol=1e-12)
    s = step_rods(rods(), [-1.0, -1.0], 5.0, False)
    assert not allowed(s)
    assert allowed(s[0]) and not allowed(s[1])                 # the first alone allows 5; the second decides
    assert allowed(step_rods(rods()[:1], [-1.0], 5.0, True))


K1, K2, Q0 = 0.6, 0.3, (1.0, 0.8, 0.5)


def book_net():
    """The net of the book's picture: species ``A, B, C``; ``t_1 : A + B -> C``, ``t_2 : C -> A``."""
    return book_reaction_network(Q0, [([0, 1], [2], K1), ([2], [0], K2)], names=["A", "B", "C"])


def flat(s):
    return np.concatenate([np.ravel(np.asarray(x)) for x in jax.tree_util.tree_leaves(s)])


def test_book_reaction_network_step():
    """Prediction (c): ``eqn.reaction_network_step``."""
    arr = book_net()
    assert all(n in arr.label for n in ("A", "B", "C")) and arr.label.count("transition") == 2
    qA, qB, qC = Q0
    alpha = jnp.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    beta = jnp.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    for tau in (0.25, 1.0):
        s = closed_step(arr, duration=tau)(arr.P.init)
        assert allowed(s)
        want = [qA + tau * (K2 * qC - K1 * qA * qB), qB - tau * K1 * qA * qB, qC + tau * (K1 * qA * qB - K2 * qC)]
        np.testing.assert_allclose(flat(s), want, atol=1e-12)
        assert abs(flat(s)[0] + flat(s)[2] - (qA + qC)) < 1e-12
        seven = composed_mass_action_arrangement(alpha, beta, [K1, K2], [0.0, 0.0], Q0)
        np.testing.assert_allclose(flat(closed_step(seven, duration=tau)(seven.P.init)), want, atol=1e-12)
    assert not allowed(closed_step(arr, check=0, duration=3.0)(arr.P.init))


def test_transition_loads_are_the_pictures():
    """The loads written on the picture's wires: ``-kappa_1 q_A q_B`` on ``t_1``'s input arrows, ``+kappa_1 q_A q_B``
    on its output arrow; ``-kappa_2 q_C`` and ``+kappa_2 q_C`` on ``t_2``'s."""
    from arrangements.reaction_network import book_transition
    qA, qB, qC = Q0
    z = jnp.zeros(0)
    _, _, l1 = book_transition(2, 1, K1).omega(z, z, jnp.array([qA, qB, qC]))
    _, _, l2 = book_transition(1, 1, K2).omega(z, z, jnp.array([qC, qA]))
    np.testing.assert_allclose(l1, [-K1 * qA * qB, -K1 * qA * qB, K1 * qA * qB], atol=1e-12)
    np.testing.assert_allclose(l2, [-K2 * qC, K2 * qC], atol=1e-12)


def test_euler_integrator_is_each_species_and_differentiates_back():
    """Prediction (d), ``ex.affine_stepping`` and ``prop.euler_derivative``."""
    E = euler_integrator(lambda s: s, lambda s, xi: xi, 1, [0.7], domain=positive)
    P = species(0.7, 1).P
    for tau in (0.25, 1.0):
        for xi in (0.3, -0.5):
            a = E.step_over(E.init, jnp.array([xi]), tau)
            b = P.step_over(P.init, jnp.array([xi]), tau)
            assert allowed(a) == allowed(b) == (0.7 + tau * xi > 0)
            if allowed(a):
                np.testing.assert_allclose(a, b, atol=1e-12)
    xi = jnp.array([-0.5])
    zero, rate = jax.jvp(lambda t: E.step_over(E.init, xi, t), (0.0,), (1.0,))
    np.testing.assert_allclose(zero, E.init, atol=1e-12)
    np.testing.assert_allclose(rate, xi, atol=1e-12)
    rng = np.random.default_rng(3)
    A, b = rng.normal(size=(3, 3)), rng.normal(size=3)
    s, w = rng.normal(size=3), rng.normal(size=3)
    for tau in (0.25, 1.0):
        np.testing.assert_allclose(A @ (s + tau * w) + b, (A @ s + b) + tau * (A @ w), atol=1e-12)
