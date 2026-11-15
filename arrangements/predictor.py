"""A predictor beside a physical system, one-way and active (book Ch 4 ``sec.prediction_learning``).

The book's predictor (``eqn.predictor_look_ahead``, ``eqn.predictor_residual``, ``eqn.predictor_update``):
an integrator with state ``(x, theta)`` (the estimate ``x``; ``mu`` in the code), configuration space ``X``,
look-ahead the forecast ``f(theta, tau, x)``, cost ``1/2 |y - o(xhat)|^2`` at the value ``y`` it hears, and
update

    (x, theta) |-> (xhat + tau G xi,  theta + eta (d_theta f(theta, tau, x))^T G xi),   xhat = f(theta, tau, x).

``G`` is the book's response function at the forecast (``def.model_predictor``; the book writes it as a sharp),
taken constant here.

Two systems ``1 -> <0|Y>`` transferring out the prediction, run by ``Phi`` at any duration, with nothing
computed between steps (book Ch 4 ``sec.predictor_active``). Every wire carries a value in ``Y``, as the book's
do since its Ch 4 was rebuilt on values (until 2026-11-14 they carried messages, a value beside a weight 1):

The system reads out ``q`` and is observed by its own map ``g``, which need not be the predictor's ``o``
(Friston et al. 2009, Methods, "Active inference," eqs. (7)-(8)); the tests pass the same map for both.

* ``one_way``: the predictor hears the system through the book's sensor (``sensor`` below; the book says only
  is the follower of Ch 4 ``ex.follower``, its report pulled toward ``g(q)``), load-free and unilateral at its output
  (Ch 3 ``def.arr_vocabulary``), so the system never feels the predictor.
* ``active``: the sensor is removed; the wiring sends ``q`` to the value ``g(q)``. The predictor's load on
  that value comes back along the wire, pulled back through ``g``, so the system receives
  ``(T_q g)^T (o(xhat) - g(q))``: active inference (Friston et al. 2009, Methods, "Active inference," eq. (9)).

Book plan P84 (g), Claude Opus 5.5, session 4eacd80b; *proposed*. Tests: ``tests/test_predictor.py``.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

from .arrangement import Arrangement, exact, zero_form
from .coalgebra import CotCoalg, configuration, phase
from .functors import Phi
from .interpretation import trivial_omega
from .rvect import constant, diagonal
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
    """The modeling predictor (book ``def.model_predictor``), ``<0|0> -> <dy|dy>``: it hears a value ``y`` in
    ``Y`` and sends out its prediction ``o(xhat)``."""
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

    def neg_cost(q, m_out, n_in):          # the inclination field is -d of 1/2 |y - o(xhat)|^2
        r = n_in - o(q)
        return -0.5 * jnp.dot(r, r)

    return Arrangement(
        P=P, in_dim_M=0, out_dim_M=0, in_dim_N=dy, out_dim_N=dy,
        out_f=lambda q, m_out: jnp.atleast_1d(o(q)), in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(neg_cost), label="predictor",
    )


def sensor(g: Callable[[Array], Array], k: float, dx: int, dy: int, y0=None) -> Arrangement:
    """The book's sensor ``<0|0> -> <dy|dx>``: the follower of Ch 4 ``ex.follower``, a report ``y`` that follows a
    leader ``z``, reading out ``y``.

    Its integrator is the response-integrator on ``Y x Y`` with constant response ``(xi_y, xi_z) |-> k (xi_z, xi_z)``
    (Ch 3 ``def.response_integrator``): a load on the leader moves both, a load on the report moves neither. Its
    inclination field is ``g(x) - y`` on the leader and ``0`` on the report and on the input ``x`` (load-free), so
    ``y' = y + k tau (g(x) - y)``: at ``k tau = 1`` the report is a snapshot of ``g(x)``. A load on the readout ``y``
    is pulled back to ``(xi, 0)``, which the response sends to ``0``: unilateral at its output (Ch 3
    ``def.arr_vocabulary`` (x)). (Until 2026-11-14 the update was two-stage, ``y' = y + k tau (z' - y)``, not a
    response-integrator, so not unilateral in the book's sense; book plan P103 (a), Claude Opus 5.5, e1aedebb.)"""
    s0 = jnp.zeros(2 * dy) if y0 is None else jnp.concatenate([jnp.asarray(y0, float)] * 2)
    I = jnp.eye(dy)
    O = jnp.zeros((dy, dy))
    P = configuration(constant(k * jnp.block([[O, I], [O, I]])), init=s0)

    def omega(q, m_out, n_in):
        return (jnp.concatenate([jnp.zeros(dy), g(n_in) - q[:dy]]), jnp.zeros_like(m_out), jnp.zeros_like(n_in))

    return Arrangement(
        P=P, in_dim_M=0, out_dim_M=0, in_dim_N=dx, out_dim_N=dy,
        out_f=lambda q, m_out: q[:dy], in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=omega, label="sensor",
    )


def _route(in_M: int, out_M: int, in_f: Callable, dy: int) -> Arrangement:
    """The wiring: ``in_f`` sends the parts' outputs to their inputs, and the last ``dy`` outputs, the
    predictor's prediction ``o(xhat)``, are transferred out."""
    return Arrangement(
        P=trivial_parameter(), in_dim_M=in_M, out_dim_M=out_M, in_dim_N=0, out_dim_N=dy,
        out_f=lambda q, m_out: m_out[out_M - dy:], in_f=lambda q, m_out, n_in: in_f(m_out),
        omega=zero_form, label="route",
    )


def one_way(sys: Arrangement, sense: Arrangement, pred: Arrangement) -> Arrangement:
    """The system ``1 -> <0|Y>`` of ``eqn.predictive_coding_wiring``: sys into the sensor, its report to the
    predictor, whose prediction is transferred out. State: ``(((sys, sensor), predictor), route)``."""
    dx, dy = sys.out_dim_N, pred.in_dim_N
    parts = tensor_arrangements([sys, sense, pred])
    n = dx + dy
    return compose_seq(parts, _route(n, n + dy, lambda m: m[:n], dy))


def active(sys: Arrangement, pred: Arrangement, g: Callable[[Array], Array]) -> Arrangement:
    """The sensor removed: the wiring's in-transfer ``g`` sends ``q`` to the value ``g(q)`` on the predictor's
    input, and the prediction is transferred out. State: ``((sys, predictor), route)``."""
    dx, dy = sys.out_dim_N, pred.in_dim_N
    parts = tensor_arrangements([sys, pred])
    return compose_seq(parts, _route(dy, dx + dy, lambda m: jnp.atleast_1d(g(m[:dx])), dy))


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
