"""A predictor beside a physical system, one-way and active (book Ch 4 ``sec.prediction_learning``).

The book's predictor (``eqn.predictor_look_ahead``, ``eqn.predictor_residual``, ``eqn.predictor_update``):
an integrator with state ``(x, theta)`` (the estimate ``x``; ``mu`` in the code), configuration space ``X``,
look-ahead the forecast ``f(theta, tau, x)``, cost ``1/2 |h - w o(xhat)|^2`` at the message ``(h, w)`` it hears, and
update

    (x, theta) |-> (xhat + tau G xi,  theta + eta (d_theta f(theta, tau, x))^T G xi),   xhat = f(theta, tau, x).

Two systems ``1 -> <0|msg(Y)>`` transferring out the prediction, run by ``Phi`` at any duration, with nothing
computed between steps (book Ch 4 ``sec.predictor_active``):

The system reads out ``q`` and is observed by its own map ``g``, which need not be the predictor's ``o``
(Friston et al. 2009, Methods, "Active inference," eqs. (7)-(8)); the tests pass the same map for both.

* ``one_way``: the predictor hears the system through the book's sensor (``sensor`` below; the book says only
  that it takes a snapshot of ``g(q)`` and reads it out while ignoring every force on its readout), a part storing
  a reading and a report, load-free and unilateral (Ch 3 ``def.load_free_unilateral``), so the system never feels
  the predictor.
* ``active``: the sensor is removed; the wiring sends ``q`` to the message ``(g(q), 1)``. The predictor's force on
  that message comes back along the wire, pulled back through ``g``, so the system receives
  ``(T_q g)^T (o(xhat) - g(q))``: active inference (Friston et al. 2009, Methods, "Active inference," eq. (9)).

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


def predictor(f: Model, o: Callable[[Array], Array], G: Array, eta: float, mu0, theta0, dy: int) -> Arrangement:
    """The modeling predictor (book ``def.model_predictor``), ``<0|0> -> <dy+1|dy+1>``: it hears a message
    ``(h, w)`` in ``msg(Y)`` and sends out its prediction ``msg(o(xhat)) = (o(xhat), 1)``."""
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
        P=P, in_dim_M=0, out_dim_M=0, in_dim_N=dy + 1, out_dim_N=dy + 1,
        out_f=lambda q, m_out: jnp.concatenate([o(q), jnp.ones(1)]), in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(neg_cost), label="predictor",
    )


def sensor(g: Callable[[Array], Array], k: float, dx: int, dy: int, y0=None) -> Arrangement:
    """The book's sensor ``<0|0> -> <dy+1|dx>``: stores a reading ``z`` and a report ``y``, reads out ``msg(y)``.

    Its effort field is ``g(q) - z`` on the reading and ``0`` on the report and on ``q`` (load-free); its update uses
    only the reading's component (unilateral): ``z' = z + k tau xi_z``, ``y' = y + k tau (z' - y)``."""
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
        return (jnp.concatenate([jnp.zeros(dy), g(n_in) - q[dy:]]), jnp.zeros_like(m_out), jnp.zeros_like(n_in))

    return Arrangement(
        P=P, in_dim_M=0, out_dim_M=0, in_dim_N=dx, out_dim_N=dy + 1,
        out_f=lambda q, m_out: jnp.concatenate([q[:dy], jnp.ones(1)]), in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=omega, label="sensor",
    )


def _route(in_M: int, out_M: int, in_f: Callable, dy: int) -> Arrangement:
    """The wiring: ``in_f`` sends the parts' outputs to their inputs, and the last ``dy + 1`` outputs, the
    predictor's prediction ``msg(o(xhat))``, are transferred out."""
    return Arrangement(
        P=trivial_parameter(), in_dim_M=in_M, out_dim_M=out_M, in_dim_N=0, out_dim_N=dy + 1,
        out_f=lambda q, m_out: m_out[out_M - dy - 1:], in_f=lambda q, m_out, n_in: in_f(m_out),
        omega=zero_form, label="route",
    )


def one_way(sys: Arrangement, sense: Arrangement, pred: Arrangement) -> Arrangement:
    """The system ``1 -> <0|msg(Y)>`` of ``eqn.predictive_coding_wiring``: sys into the sensor, its report to the
    predictor, whose prediction is transferred out. State: ``(((sys, sensor), predictor), route)``."""
    dx, dy = sys.out_dim_N, pred.in_dim_N - 1
    parts = tensor_arrangements([sys, sense, pred])
    n = dx + dy + 1
    return compose_seq(parts, _route(n, n + dy + 1, lambda m: m[:n], dy))


def active(sys: Arrangement, pred: Arrangement, g: Callable[[Array], Array]) -> Arrangement:
    """The sensor removed: the wiring's in-transfer ``msg o g`` sends ``q`` to ``(g(q), 1)`` on the predictor's
    input, and the prediction is transferred out. State: ``((sys, predictor), route)``."""
    dx, dy = sys.out_dim_N, pred.in_dim_N - 1
    parts = tensor_arrangements([sys, pred])
    return compose_seq(parts, _route(dy + 1, dx + dy + 1, lambda m: jnp.concatenate([g(m[:dx]), jnp.ones(1)]), dy))


_IN = (jnp.zeros(0), trivial_omega(0))


def run(arr: Arrangement, tau: float, steps: int, s=None):
    """Iterate ``Phi(arr)`` with every step held for duration ``tau``, nothing pushing on the transferred-out
    prediction; return the final state."""
    O = Phi(arr)
    tell = lambda _o: (jnp.zeros(arr.out_dim_N), jnp.zeros(arr.in_dim_N))
    step = jax.jit(lambda s: O.with_state(s).run_one(_IN, tell, tau)[2])
    s = arr.P.init if s is None else s
    for _ in range(steps):
        s = step(s)
    return s
