"""A predictor beside a physical system, one-way and active (book Ch 4 ``sec.predictive_coding``).

The book's predictor (``eqn.predictor_look_ahead``, ``eqn.predictor_residual``, ``eqn.predictor_update``):
an integrator with state ``(mu, theta)``, configuration space ``X``, look-ahead the forecast
``f(theta, tau, mu)``, cost ``1/2 |h - w o(xhat)|^2`` at the message ``(h, w)`` it hears, and update

    (mu, theta) |-> (xhat + tau G xi,  theta + eta (d_theta f(theta, tau, mu))^T G xi),   xhat = f(theta, tau, mu).

Two closed arrangements, run by ``Phi`` at any duration, with nothing computed between steps:

* ``one_way``: the predictor hears the system through the book's sensor (``eqn.sensor_update``), a part storing a
  reading and a report, load-free and unilateral (Ch 3 ``def.load_free_unilateral``), so the system never feels the
  predictor.
* ``active``: the sensor is removed; the wiring sends ``(x, v)`` to the message ``(o(x) + v, 1)``. The predictor's
  force on that message comes back along the wire, pulled back through ``o``, so the system receives
  ``(T_x o)^T (o(xhat) - o(x) - v)``: active inference (Friston et al. 2009, Results, "Active inference").

Book plan P84 (g), Claude Opus 5.5, session 4eacd80b; *proposed*. Tests: ``tests/test_predictor.py``.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

from .arrangement import Arrangement, exact, zero_form
from .coalgebra import CotCoalg, phase
from .functors import Phi
from .interpretation import trivial_omega
from .rvect import diagonal
from .wiring import compose_seq, tensor_arrangements, trivial_parameter

Model = Callable[[Array, float, Array], Array]   # f(theta, tau, mu) -> X, with f(theta, 0, mu) = mu


def free_mass(m: float, q0, p0) -> Arrangement:
    """A physical system ``<0|0> -> <0|dim>`` with no cost of its own: a mass that reads out its look-ahead."""
    q0, p0 = jnp.atleast_1d(jnp.asarray(q0, float)), jnp.atleast_1d(jnp.asarray(p0, float))
    d = q0.shape[0]
    return Arrangement(
        P=phase(diagonal(jnp.full(d, 1.0 / m)), init=(q0, p0)),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=d,
        out_f=lambda q, m_out: q, in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=zero_form, label="sys",
    )


def noise(v) -> Arrangement:
    """A source ``<0|0> -> <0|dim>`` reading out the sensor error ``v``; it has no state, so it ignores every force."""
    v = jnp.atleast_1d(jnp.asarray(v, float))
    return Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=v.shape[0],
        out_f=lambda q, m_out: v, in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=zero_form, label="noise",
    )


def predictor(f: Model, o: Callable[[Array], Array], G: Array, eta: float, mu0, theta0, dy: int) -> Arrangement:
    """The predictor ``<0|0> -> <dy+1|0>``: it hears a message ``(h, w)`` in ``msg(Y)`` and reads out nothing."""
    mu0, theta0 = jnp.atleast_1d(jnp.asarray(mu0, float)), jnp.atleast_1d(jnp.asarray(theta0, float))
    G = jnp.atleast_2d(jnp.asarray(G, float))

    def look_ahead(s, tau):
        mu, theta = s
        return f(theta, tau, mu)

    def timed_step(s, xi, tau):
        mu, theta = s
        xhat = f(theta, tau, mu)
        J = jax.jacobian(lambda th: f(th, tau, mu))(theta)   # d_theta f, an X x Theta matrix
        return (xhat + tau * G @ xi, theta + eta * J.T @ (G @ xi))

    P = CotCoalg(
        Q_dim=mu0.shape[0], init=(mu0, theta0),
        position=lambda s: look_ahead(s, 1.0), step=lambda s, xi: timed_step(s, xi, 1.0),
        look_ahead=look_ahead, timed_step=timed_step, label="predictor",
    )

    def neg_cost(q, m_out, n_in):          # the effort field is -d of 1/2 |h - w o(xhat)|^2
        h, w = n_in[:dy], n_in[dy]
        r = h - w * o(q)
        return -0.5 * jnp.dot(r, r)

    return Arrangement(
        P=P, in_dim_M=0, out_dim_M=0, in_dim_N=dy + 1, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0), in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(neg_cost), label="predictor",
    )


def sensor(o: Callable[[Array], Array], k: float, dx: int, dy: int, y0=None) -> Arrangement:
    """The book's sensor ``<0|0> -> <dy+1|dx+dy>``: stores a reading ``z`` and a report ``y``, reads out ``msg(y)``.

    Its effort field is ``o(x) + v - z`` on the reading and ``0`` on the report and on ``(x, v)`` (load-free); its
    update uses only the reading's component (unilateral): ``z' = z + k tau xi_z``, ``y' = y + k tau (z' - y)``."""
    s0 = jnp.zeros(2 * dy) if y0 is None else jnp.concatenate([jnp.asarray(y0, float)] * 2)

    def timed_step(s, xi, tau):
        y, z = s[:dy], s[dy:]
        z2 = z + k * tau * xi[dy:]
        return jnp.concatenate([y + k * tau * (z2 - y), z2])

    P = CotCoalg(
        Q_dim=2 * dy, init=s0, position=lambda s: s, step=lambda s, xi: timed_step(s, xi, 1.0),
        timed_step=timed_step, label="sensor",
    )

    def omega(q, m_out, n_in):
        x, v = n_in[:dx], n_in[dx:]
        return (jnp.concatenate([jnp.zeros(dy), o(x) + v - q[dy:]]), jnp.zeros_like(m_out), jnp.zeros_like(n_in))

    return Arrangement(
        P=P, in_dim_M=0, out_dim_M=0, in_dim_N=dx + dy, out_dim_N=dy + 1,
        out_f=lambda q, m_out: jnp.concatenate([q[:dy], jnp.ones(1)]), in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=omega, label="sensor",
    )


def _route(in_M: int, out_M: int, in_f: Callable) -> Arrangement:
    return Arrangement(
        P=trivial_parameter(), in_dim_M=in_M, out_dim_M=out_M, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0), in_f=lambda q, m_out, n_in: in_f(m_out),
        omega=zero_form, label="route",
    )


def one_way(sys: Arrangement, v: Arrangement, sense: Arrangement, pred: Arrangement) -> Arrangement:
    """The closed arrangement of ``eqn.predictive_coding_wiring``: sys and noise into the sensor, its report to the
    predictor. State: ``((((sys, noise), sensor), predictor), route)``."""
    dx, dy = sys.out_dim_N, v.out_dim_N
    parts = tensor_arrangements([sys, v, sense, pred])
    return compose_seq(parts, _route(dx + dy + dy + 1, dx + dy + dy + 1, lambda m: m))


def active(sys: Arrangement, v: Arrangement, pred: Arrangement, o: Callable[[Array], Array]) -> Arrangement:
    """The sensor removed: the wiring sends ``(x, v)`` to ``(o(x) + v, 1)`` on the predictor's input.
    State: ``(((sys, noise), predictor), route)``."""
    dx, dy = sys.out_dim_N, v.out_dim_N
    parts = tensor_arrangements([sys, v, pred])
    return compose_seq(parts, _route(dy + 1, dx + dy, lambda m: jnp.concatenate([o(m[:dx]) + m[dx:], jnp.ones(1)])))


_IN = (jnp.zeros(0), trivial_omega(0))


def run(arr: Arrangement, tau: float, steps: int, s=None):
    """Iterate ``Phi(arr)`` with every step held for duration ``tau``; return the final state."""
    O = Phi(arr)
    step = jax.jit(lambda s: O.with_state(s).run_one(_IN, lambda _o: (jnp.zeros(0), jnp.zeros(0)), tau)[2])
    s = arr.P.init if s is None else s
    for _ in range(steps):
        s = step(s)
    return s
