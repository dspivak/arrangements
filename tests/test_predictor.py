"""A predictor beside a physical system, one-way and active (book Ch 4 ``sec.predictive_coding``; book plan
P84 (g), Claude Opus 5.5, 4eacd80b).

A free mass beside a predictor whose model says the hidden state decays to 0. Heard through the load-free,
unilateral sensor, the mass never feels the predictor; with the sensor removed, the predictor's prediction error
comes back along the wire and pulls the mass to where the model says it goes. With learning on, the one-way
predictor's model learns that the mass does not decay, while the active one makes the mass fit its model.
"""

import jax
import jax.numpy as jnp
import pytest

from arrangements.functors import Phi
from arrangements.predictor import _IN, active, free_mass, one_way, predictor, run, sensor


def decay(theta, tau, mu):          # f(theta, tau, mu) = e^{-theta tau} mu, f(theta, 0, mu) = mu
    return jnp.exp(-theta[0] * tau) * mu


ident = lambda x: x


def pair(kind, *, o=ident, q0=1.0, p0=0.0, mu0=1.0, eta=0.0):
    sys = free_mass(1.0, q0, p0)
    pred = predictor(decay, o, jnp.eye(1), eta, mu0, 1.0, dy=1)
    if kind == "active":
        return active(sys, pred, o)
    return one_way(sys, sensor(o, 1.0, 1, 1, y0=[q0]), pred)


def mass_of(kind, s):               # (q, p) of the mass
    return s[0][0] if kind == "active" else s[0][0][0]


def theta_of(kind, s):
    return (s[0][1] if kind == "active" else s[0][1])[1][0]


def test_predictor_sends_its_prediction():
    """The modeling predictor sends out msg(o(xhat)) at its forecast xhat (book def.model_predictor)."""
    pred = predictor(decay, jnp.sin, jnp.eye(1), 0.0, 0.4, 1.0, dy=1)
    xhat = pred.P.look_ahead(pred.P.init, 0.1)
    assert jnp.allclose(xhat, jnp.exp(-0.1) * 0.4)
    assert jnp.allclose(pred.out_f(xhat, jnp.zeros(0)), jnp.array([jnp.sin(xhat[0]), 1.0]))


@pytest.mark.parametrize("kind", ["one_way", "active"])
def test_grouping_transfers_out_the_prediction(kind):
    """Both material groupings transfer out the predictor's prediction msg(o(xhat))."""
    arr, tau = pair(kind, o=jnp.sin, mu0=0.4), 0.1
    tell = lambda _o: (jnp.zeros(arr.out_dim_N), jnp.zeros(arr.in_dim_N))
    out_pos = Phi(arr).with_state(arr.P.init).run_one(_IN, tell, tau)[0]
    xhat = jnp.exp(-tau) * 0.4
    assert jnp.allclose(jnp.ravel(jax.tree_util.tree_leaves(out_pos)[0]), jnp.array([jnp.sin(xhat), 1.0]), atol=1e-6)


def test_force_on_system_is_the_prediction_error_pulled_back():
    """One active step: the mass's momentum gains tau (T_x o)^T (o(xhat) - o(x))."""
    o = jnp.sin
    q0, p0, mu0, tau = 0.7, 0.3, 0.2, 0.1
    s = run(pair("active", o=o, q0=q0, p0=p0, mu0=mu0), tau, 1)
    x = q0 + tau * p0                       # the mass's look-ahead
    xhat = jnp.exp(-tau) * mu0              # the predictor's forecast
    force = jnp.cos(x) * (jnp.sin(xhat) - jnp.sin(x))
    assert jnp.allclose(mass_of("active", s)[1][0], p0 + tau * force, atol=1e-6)


@pytest.mark.parametrize("tau", [0.1, 0.05])
def test_one_way_leaves_the_system_alone(tau):
    """Through the sensor the mass coasts exactly as it would alone."""
    n = int(round(6.0 / tau))
    q, p = mass_of("one_way", run(pair("one_way", p0=0.2), tau, n))
    assert jnp.allclose(p[0], 0.2) and jnp.allclose(q[0], 1.0 + n * tau * 0.2, atol=1e-5)


@pytest.mark.parametrize("tau", [0.1, 0.05])
def test_active_pulls_the_system_to_the_prediction(tau):
    """With the sensor removed the mass is pulled to where the model says it goes; through the sensor it stays."""
    n = int(round(60.0 / tau))
    q_active = mass_of("active", run(pair("active"), tau, n))[0][0]
    q_one_way = mass_of("one_way", run(pair("one_way"), tau, n))[0][0]
    assert abs(q_active) < 1e-2 and jnp.allclose(q_one_way, 1.0)


def test_duration_zero_is_the_identity():
    arr = pair("active", p0=0.3, mu0=0.4)
    s = run(arr, 0.0, 5)
    assert jnp.allclose(mass_of("active", s)[0], mass_of("active", arr.P.init)[0])
    assert jnp.allclose(s[0][1][0], arr.P.init[0][1][0])


@pytest.mark.parametrize("tau", [0.1, 0.05])
def test_learning_one_way_learns_the_world_active_moves_it(tau):
    """With learning on, the one-way model's decay rate goes to 0 (the mass does not decay); the active model keeps a
    decay, having pulled the mass to 0 to fit it."""
    n = int(round(60.0 / tau))
    s_one_way, s_active = run(pair("one_way", eta=1.0), tau, n), run(pair("active", eta=1.0), tau, n)
    assert abs(theta_of("one_way", s_one_way)) < 1e-3 and theta_of("active", s_active) > 0.5
    assert jnp.allclose(mass_of("one_way", s_one_way)[0][0], 1.0) and abs(mass_of("active", s_active)[0][0]) < 1e-2
