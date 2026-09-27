"""Pain: a fund that tracks how near its whole is to coming apart, and parts that read it
(``tests/test_pain.py``).

The cascade's one level (``cascade.py``) with growth and pain, on three channels, ``H = R^3``:
supply, product, pain. A message is ``(h_1, h_2, h_3, w)``.

* ``worker``: configuration ``(a, theta, sigma)``. It holds the amplitude ``a`` on received
  supply as in the cascade, splits it into a learning share ``s = sigmoid(theta)`` and a product
  share, and sends ``(0, (1 + sigma)(1 - s) a, 0, 1)``. Its skill ``sigma`` moves by the non-exact
  flow ``eta s a - delta sigma``. Its allocation descends on ``(1/2)(theta - theta*)^2``,
  ``theta* = logit(s0) - Lambda h_3``, where ``h_3`` is the pain it receives, read through
  ``stop_gradient``: reading returns no push, so the pain wire is one-way.
* ``source``: supply ``s`` on channel 1, weight 0.
* ``fund_layer``: configuration ``(f, pi)``. The balance moves by the cascade's flow
  ``income - g(f) w~ - c0 - leak f``, the wage wire one-way (``stop_gradient`` on the balance, as
  in ``account.py``'s ``"oneway"``); ``pi`` moves by ``k_pi (phi(f) - pi)``, ``phi`` the fund's
  fixed reading of its own balance. It sends ``(g(f), 0, pi, 1)`` down and ``(0, h_2, 0, 1)`` up.
  ``shocks = (q, S)`` makes its parameter ``rand.kicked_configuration``: the balance drained by
  ``S`` with probability ``q`` a step.
* ``market``: closes the whole, paying ``r`` times its product on channel 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp

from .arrangement import Arrangement, exact
from .attention import input_dim, make_input, message, msg_dim
from .cascade import gate
from .coalgebra import configuration
from .collective import DP, Part, collective, frozen_part
from .rand import kicked_configuration
from .rvect import constant, trivial
from .wiring import compose_seq

DH3 = 3
MDIM3, EDIM3 = msg_dim(DH3), input_dim(DH3, DP)     # dim M = 4, dim E = 5
EMPTY = jnp.zeros(0)
SUPPLY, PRODUCT, PAIN, WEIGHT = 0, 1, 2, 3


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def worker(*, alpha: float, b: float, beta: float, mu_a: float, mu_theta: float, eta: float,
           delta: float, s0: float, Lam: float, a0: float = 0.8, sigma0: float = 0.0,
           label: str = "w") -> Part:
    """A worker that learns with share ``sigmoid(theta)`` and reads pain (module docstring)."""
    theta0 = logit(s0)

    def cost(q, e):
        a, th = q[0], q[1]
        h3 = jax.lax.stop_gradient(e[PAIN])
        return 0.5 * b * a ** 2 - (alpha * a + beta) * e[SUPPLY] + 0.5 * (th - theta0 + Lam * h3) ** 2

    dk = exact(lambda q, m, n: -cost(q, n))

    def omega(q, m_out, n_in):
        xq, xm, xn = dk(q, m_out, n_in)
        learn = eta * jax.nn.sigmoid(q[1]) * q[0] - delta * q[2]
        return (xq.at[2].add(learn), xm, xn)

    def send(q):
        a, s, sig = q[0], jax.nn.sigmoid(q[1]), q[2]
        return message(jnp.array([0.0, (1.0 + sig) * (1.0 - s) * a, 0.0]), 1.0)

    P = configuration(constant(jnp.diag(jnp.array([mu_a, mu_theta, 1.0]))),
                      init=jnp.array([float(a0), theta0, float(sigma0)]))
    arr = Arrangement(
        P=P, in_dim_M=0, out_dim_M=0, in_dim_N=EDIM3, out_dim_N=MDIM3,
        out_f=lambda q, m_out: send(q),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=omega, label=label)
    return Part(arr, 3, label)


def source(s: float, label: str = "source") -> Part:
    return frozen_part([s, 0.0, 0.0], 0.0, label, dim_H=DH3)


def phi(f, f_half: float, width: float):
    """The fund's fixed reading of its balance: even odds at ``f_half``."""
    return jax.nn.sigmoid(-(f - f_half) / width)


def fund_layer(f0: float, *, mu: float, wmax: float, K: float, c0: float, leak: float,
               k_pi: float, f_half: float, width: float, shocks: Optional[tuple] = None,
               label: str = "fund") -> Arrangement:
    """The fund box ``B -> B`` with its pain coordinate (module docstring)."""

    def xi(q, m_C, e_out):
        f, pi = q[0], q[1]
        flow_f = e_out[SUPPLY] - gate(f, wmax, K) * m_C[WEIGHT] - c0 - leak * f
        flow_pi = k_pi * (phi(f, f_half, width) - pi)
        return (jnp.array([flow_f, flow_pi]), jnp.zeros(MDIM3), jnp.zeros(EDIM3))

    Q = constant(jnp.diag(jnp.array([mu, 1.0])))
    init = jnp.array([float(f0), 0.0])
    if shocks is None:
        P = configuration(Q, init=init)
    else:
        q, S = shocks
        P = kicked_configuration(Q, [1.0 - q, q], [[0.0, 0.0], [-S, 0.0]], init=init)
    return Arrangement(
        P=P, in_dim_M=EDIM3, out_dim_M=MDIM3, in_dim_N=EDIM3, out_dim_N=MDIM3,
        out_f=lambda q, m_C: message(jnp.array([0.0, m_C[PRODUCT], 0.0]), 1.0),
        in_f=lambda q, m_C, e_out: make_input(
            message(jnp.array([gate(jax.lax.stop_gradient(q[0]), wmax, K), 0.0, q[1]]), 1.0),
            jnp.zeros(DP)),
        omega=xi, label=label)


def market(r: float) -> Arrangement:
    """Closes the whole: pays ``r`` times its product back on channel 1."""
    return Arrangement(
        P=configuration(trivial()),
        in_dim_M=EDIM3, out_dim_M=MDIM3, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m: EMPTY,
        in_f=lambda q, m, n: make_input(
            message(jnp.array([r * m[PRODUCT], 0.0, 0.0]), 1.0), jnp.zeros(DP)),
        omega=lambda q, m, n: (jnp.zeros(0), jnp.zeros(MDIM3), jnp.zeros(0)),
        label="market")


@dataclass(frozen=True)
class Whole:
    """The closed whole and where its pieces sit in the flat configuration. ``unit`` is the
    whole before its market, so that ``closed(r)`` can close it at another rate (inside ``jit``,
    where the collective itself cannot be built)."""
    arr: Arrangement
    n: int              # workers
    unit: Arrangement = None

    def closed(self, r) -> Arrangement:
        return compose_seq(self.unit, market(r))

    def read(self, q) -> dict:
        """``{'a', 'theta', 'sigma', 'placements', 'kappa', 'f', 'pi'}`` at ``q``. Layout: the
        workers' ``(a, theta, sigma)``, then the parent's placements (workers, then the source)
        and value maps, then the fund's ``(f, pi)``."""
        q = jnp.asarray(q)
        W = q[:3 * self.n].reshape(self.n, 3)
        pos = q[3 * self.n:3 * self.n + self.n]
        return {"a": W[:, 0], "theta": W[:, 1], "sigma": W[:, 2], "placements": pos,
                "kappa": jnp.exp(-pos ** 2 / (2 * TAU)), "f": q[-2], "pi": q[-1]}


TAU = 0.8


def whole(workers, *, fund: dict, r: float, src: Part, src_position: float = 2.5,
          positions=None, mobility: float = 0.1, tau: float = TAU) -> Whole:
    """``market(r)`` after a fund layer after a collective of ``workers`` and ``src``."""
    n = len(workers)
    positions = [0.06] * n if positions is None else list(positions)
    C = collective(list(workers) + [src], tau=tau, cost=lambda q, m, e: 0.0 * jnp.sum(q),
                   positions=positions + [src_position],
                   integrator=lambda Q, init: configuration(Q, init=init),
                   mobility=[mobility] * n + [0.0], value_mobility=0.0, dim_H=DH3, label="whole")
    u = compose_seq(C.arr, fund_layer(**fund))
    return Whole(compose_seq(u, market(r)), n, u)


def closed_step(arr: Arrangement):
    """One step of the closed whole."""
    P, omega = arr.P, arr.omega

    def step(s):
        xi_q, _, _ = omega(P.position(s), EMPTY, EMPTY)
        return P.step(s, xi_q)

    return step
