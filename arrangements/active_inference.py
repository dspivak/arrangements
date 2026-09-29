"""Active inference in three tests: precision by descent, policy choice, expected free energy.

Book plan §8.2 P5 (b); predictions, fixed before this file existed, and verdicts in
``notes/ACTIVE-INFERENCE-TESTS.md``. The book's mapping (its ``notes/ACTIVE-INFERENCE.md``) says
that at FIXED precision the attention paper's cost is the precision-weighted prediction error, and
names three things missing: the log-determinant term, stochasticity, and expected free energy.
This module builds the first and the third inside ``arr`` and leaves stochasticity to ``rand``.

TEST (i), PRECISION BY DESCENT. A closed arrangement on the belief ``(mu, ell)``, ``ell = log Pi``,
Euclidean response, exact forcing ``dU`` for ``U = -k``,

    k(mu, ell) = 1/2 e^ell S(mu) - 1/2 ell,      S(mu) = mean_j (x_j - mu)^2.

The forcing's components are ``Pi (xbar - mu)``, the precision-weighted prediction error, and
``1/2 (1 - Pi S(mu))``: precision rises when errors are smaller than expected. Without the
``-1/2 ell`` (``normalizer=False``) the second component is negative everywhere.

TEST (ii), POLICY CHOICE, needs no new construction: ``choice.logit_arrangement`` with
``T = 1/gamma`` and values ``-G + T log E``. ``open_chooser`` opens it to receive ``G`` forward.

TEST (iii), EXPECTED FREE ENERGY. ``world`` is an open arrangement (input a push ``u``, output its
position); ``simulate`` runs futures by calling ``Phi(world)`` for the mean and the Jacobian of that
same step for the variance; ``expected_free_energy`` is risk plus ambiguity for Gaussian beliefs.
``planner`` is the part that runs futures as a box with a belief state, publishing ``G``;
``compose_seq(planner, open_chooser)`` is the wired agent, exact or unilateral.
"""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from .arrangement import Arrangement, exact
from .choice import _closed
from .coalgebra import configuration
from .functors import Phi
from .interpretation import trivial_omega
from .replicator import shahshahani
from .rvect import euclidean


# ---------------------------------------------------------------------------
# Running a closed arrangement, jitted (``choice.settle`` is the unjitted twin).
# ---------------------------------------------------------------------------


_IN_POS = (jnp.zeros(0), trivial_omega(0))
_IN_DIR = (jnp.zeros(0), jnp.zeros(0))


def stepper(arr: Arrangement) -> Callable:
    """One step of the closed system ``Phi(arr)``, jitted."""
    O = Phi(arr)

    def step(s):
        _, _, new_s = O.with_state(s).run_one(_IN_POS, lambda _o: _IN_DIR)
        return new_s

    return jax.jit(step)


def run(arr: Arrangement, steps: int, s=None, record: Callable = None):
    """Iterate ``Phi(arr)`` ``steps`` times from ``s`` (default the parameter's init).

    With ``record``, also return ``[record(s_n) for n = 0..steps]``."""
    step = stepper(arr)
    s = arr.P.init if s is None else s
    rec = [record(s)] if record else None
    for _ in range(steps):
        s = step(s)
        if record:
            rec.append(record(s))
    return (s, rec) if record else s


# ---------------------------------------------------------------------------
# Test (i): precision learned by descent.
# ---------------------------------------------------------------------------


def precision_cost(x: Array, normalizer: bool = True) -> Callable[[Array], Array]:
    """``k(mu, ell) = 1/2 e^ell mean (x - mu)^2 - 1/2 ell`` (the last term iff ``normalizer``)."""
    x = jnp.asarray(x, dtype=float)

    def k(q):
        mu, ell = q[0], q[1]
        S = jnp.mean((x - mu) ** 2)
        return 0.5 * jnp.exp(ell) * S - (0.5 * ell if normalizer else 0.0)

    return k


def precision_learner(x: Array, eta: float, normalizer: bool = True,
                      init: Array = None) -> Arrangement:
    """The 1-D learner of ``(mu, log Pi)``: Euclidean response, exact forcing ``-dk``."""
    k = precision_cost(x, normalizer)
    q0 = jnp.zeros(2) if init is None else jnp.asarray(init, dtype=float)
    return _closed(configuration(euclidean(2, eta), init=q0), lambda q: -k(q),
                   f"precision({'with' if normalizer else 'no'} logdet)")


def cholesky_precision(q: Array) -> Array:
    """``Pi = L L^T`` from ``(a1, a2, b)``: ``L = [[e^{a1}, 0], [b, e^{a2}]]``."""
    a1, a2, b = q[0], q[1], q[2]
    L = jnp.array([[jnp.exp(a1), 0.0], [b, jnp.exp(a2)]])
    return L @ L.T


def precision_learner_2d(X: Array, eta: float, init: Array = None) -> Arrangement:
    """State ``(mu_1, mu_2, a1, a2, b)``; ``k = 1/2 tr(Pi Shat(mu)) - 1/2 log det Pi``,
    ``log det Pi = 2 (a1 + a2)``."""
    X = jnp.asarray(X, dtype=float)

    def k(q):
        mu, Pi = q[:2], cholesky_precision(q[2:])
        D = X - mu
        Shat = D.T @ D / X.shape[0]
        return 0.5 * jnp.trace(Pi @ Shat) - (q[2] + q[3])

    q0 = jnp.zeros(5) if init is None else jnp.asarray(init, dtype=float)
    return _closed(configuration(euclidean(5, eta), init=q0), lambda q: -k(q), "precision-2d")


def two_source_learner(xA: Array, xB: Array, eta: float, init: Array = None) -> Arrangement:
    """One mean, one log-precision per source: ``k = sum_s 1/2 Pi_s S_s(mu) - 1/2 ell_s``."""
    xA, xB = jnp.asarray(xA, dtype=float), jnp.asarray(xB, dtype=float)

    def k(q):
        mu, lA, lB = q[0], q[1], q[2]
        return (0.5 * jnp.exp(lA) * jnp.mean((xA - mu) ** 2) - 0.5 * lA
                + 0.5 * jnp.exp(lB) * jnp.mean((xB - mu) ** 2) - 0.5 * lB)

    q0 = jnp.zeros(3) if init is None else jnp.asarray(init, dtype=float)
    return _closed(configuration(euclidean(3, eta), init=q0), lambda q: -k(q), "two-source")


# ---------------------------------------------------------------------------
# Test (ii): policy choice is the chooser of choice.py, opened to receive G.
# ---------------------------------------------------------------------------


def policy_values(G: Array, gamma: float, E: Array = None) -> Tuple[Array, float]:
    """``(v, T)`` for ``choice.logit_arrangement``: ``T = 1/gamma``, ``v = -G + T log E``."""
    G = jnp.asarray(G, dtype=float)
    T = 1.0 / gamma
    logE = jnp.zeros_like(G) if E is None else jnp.log(jnp.asarray(E, dtype=float))
    return -G + T * logE, T


def policy_free_energy(G: Array, T: float, E: Array = None) -> Callable[[Array], Array]:
    """The chooser's cost ``k(p) = <G, p> - T <log E, p> - T H(p)``."""
    G = jnp.asarray(G, dtype=float)
    logE = jnp.zeros_like(G) if E is None else jnp.log(jnp.asarray(E, dtype=float))
    return lambda p: jnp.dot(G, p) - T * jnp.dot(logE, p) + T * jnp.sum(p * jnp.log(p))


def open_chooser(n: int, T: float, dt: float = 1.0, init: Array = None,
                 unilateral: bool = False, shift: float = 0.0) -> Arrangement:
    """The chooser ``<0|n> -> I`` receiving ``G`` forward, with potential ``-<G, p> + T H(p)``.

    Exact by default, so its forcing has the component ``-p`` on the incoming ``G`` and
    ``compose_seq`` pulls that back onto whatever produced ``G`` (retroactivity; book §5.9 item
    30). ``unilateral`` zeroes that component, which leaves the forcing non-exact.

    ``shift`` adds ``c0 <1, p>`` to the potential: the same logit, and no component on ``G``,
    but it decides whether the simplex attracts. Near the logit ``1 - sum p`` is multiplied by
    ``1 - dt T (log Z - 1)`` per step, so with ``log Z < 1`` (values ``-G``, ``G > 0``: the normal
    case) round-off grows and the chooser settles on ``exp(v/T - 1)`` instead
    (``notes/ACTIVE-INFERENCE-TESTS.md`` §4)."""
    p0 = jnp.full(n, 1.0 / n) if init is None else jnp.asarray(init, dtype=float)

    def U(q, m_out, n_in):
        return -jnp.dot(m_out, q) + shift * jnp.sum(q) - T * jnp.sum(q * jnp.log(q))

    grad = exact(U)

    def one_sided(q, m_out, n_in):
        xi_q, xi_m, xi_n = grad(q, m_out, n_in)
        return xi_q, jnp.zeros_like(xi_m), xi_n

    return Arrangement(
        P=configuration(shahshahani(n, dt), init=p0),
        in_dim_M=0, out_dim_M=n, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=one_sided if unilateral else grad,
        label=f"chooser(T={T:g}{',unilateral' if unilateral else ''})",
    )


def publisher(G: Array) -> Arrangement:
    """A stateless box ``I -> <0|n>`` publishing the constant ``G``, with no forcing."""
    G = jnp.asarray(G, dtype=float)
    return Arrangement(
        P=configuration(euclidean(0)),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=int(G.shape[0]),
        out_f=lambda q, m_out: G,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=lambda q, m_out, n_in: (jnp.zeros_like(q), jnp.zeros_like(m_out),
                                      jnp.zeros_like(n_in)),
        label="publish(G)",
    )


# ---------------------------------------------------------------------------
# Test (iii): expected free energy, computed by a part that runs futures.
# ---------------------------------------------------------------------------


def world(kappa: float = 0.4, eta: float = 0.5, init: float = 0.0) -> Arrangement:
    """A pushed spring ``<0|1> -> <0|1>``: state ``x``, Euclidean response ``eta``, potential
    ``-1/2 kappa x^2 + u x`` for the incoming push ``u``, publishing ``x``. One step of ``Phi`` is
    ``x' = (1 - eta kappa) x + eta u``."""
    return Arrangement(
        P=configuration(euclidean(1, eta), init=jnp.full(1, init)),
        in_dim_M=0, out_dim_M=1, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -0.5 * kappa * q[0] ** 2 + m_out[0] * q[0]),
        label="world",
    )


def phi_step(arr: Arrangement) -> Callable[[Array, Array], Array]:
    """``(x, u) |-> `` the state after one step of ``Phi(arr)`` with input ``u`` and no push from
    downstream: running the arrangement, which is what a simulated future is."""
    O = Phi(arr)

    def step(x, u):
        _, _, new = O.with_state(x).run_one((u, trivial_omega(0)),
                                            lambda _o: (jnp.zeros(arr.out_dim_N), jnp.zeros(0)))
        return new

    return step


def predict(step: Callable, m: Array, Sig: Array, u: Array, Q: float) -> Tuple[Array, Array]:
    """The predicted belief after one simulated step: mean by running the step, variance
    ``J Sig J^T + Q`` with ``J`` the step's Jacobian (exact for a linear world)."""
    mu = step(m, u)
    J = jax.jacfwd(lambda x: step(x, u))(m)
    return mu, J @ Sig @ J.T + Q * jnp.eye(m.shape[0])


def risk(mu_o: Array, var_o: Array, ostar: float, P: float) -> Array:
    """``KL(N(mu_o, var_o) || N(o*, P))`` in one dimension."""
    return 0.5 * (var_o / P + (mu_o - ostar) ** 2 / P - 1.0 - jnp.log(var_o / P))


def ambiguity(R: float) -> Array:
    """``E_q H[p(o | s)] = 1/2 log(2 pi e R)``: the observation's entropy, independent of belief."""
    return 0.5 * jnp.log(2 * jnp.pi * jnp.e * R)


def information_gain(V: Array, R: float) -> Array:
    """``I(o; s) = 1/2 log(1 + w_obs / w_pred)``, ``w_obs = 1/R``, ``w_pred = 1/V``."""
    return 0.5 * jnp.log(1.0 + (1.0 / R) / (1.0 / V))


def expected_free_energy(step: Callable, m: Array, Sig: Array, policy: Sequence[Tuple[float, float]],
                         Q: float, ostar: float, P: float) -> Array:
    """``G`` of a policy ``((u_1, R_1), ...)``: risk plus ambiguity summed over the simulated steps,
    the belief carried forward by ``predict`` and, between steps, the expected posterior variance
    ``(1/V + 1/R)^{-1}`` (a Gaussian's posterior variance does not depend on the value seen)."""
    m = jnp.atleast_1d(jnp.asarray(m, dtype=float))
    Sig = jnp.atleast_2d(jnp.asarray(Sig, dtype=float))
    G = 0.0
    for u, R in policy:
        m, V = predict(step, m, Sig, jnp.full(1, u), Q)
        G = G + risk(m[0], V[0, 0] + R, ostar, P) + ambiguity(R)
        Sig = jnp.atleast_2d(1.0 / (1.0 / V[0, 0] + 1.0 / R))
    return G


def G_vector(step, m, Sig, policies, Q, ostar, P) -> Array:
    return jnp.stack([expected_free_energy(step, m, Sig, pol, Q, ostar, P) for pol in policies])


def planner(policies, Sig: float, x_obs: float, w_ev: float, eta: float, Q: float,
            ostar: float, P: float, init: float = 0.0, world_arr: Arrangement = None) -> Arrangement:
    """The part that runs futures, as a box ``I -> <0|n>``: state its belief mean ``m``
    (Euclidean response ``eta``), its own evidence potential ``-1/2 w_ev (m - x_obs)^2``, and it
    publishes ``G(m)`` for each policy by running ``Phi(world)`` (``expected_free_energy``); the
    belief variance ``Sig`` is held fixed."""
    step = phi_step(world() if world_arr is None else world_arr)
    return Arrangement(
        P=configuration(euclidean(1, eta), init=jnp.full(1, init)),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=len(policies),
        out_f=lambda q, m_out: G_vector(step, q, Sig, policies, Q, ostar, P),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -0.5 * w_ev * (q[0] - x_obs) ** 2),
        label="planner",
    )


# ---------------------------------------------------------------------------
# Addendum B: a chooser whose every state is a distribution (the natural chart).
# ---------------------------------------------------------------------------


def natural_chooser(n: int, T: float, dt: float = 1.0, E: Array = None, init: Array = None,
                    unilateral: bool = False) -> Arrangement:
    """The chooser ``<0|n> -> I`` in the gauge-fixed natural chart: state ``lambda in R^{n-1}``,
    ``p = softmax(lambda, 0)``, response ``dt`` times the inverse Fisher metric, exact forcing
    ``-dk`` for ``k = <G, p> - T <log E, p> - T H(p)`` with ``G`` incoming. Every state is a
    distribution, so the drift of ``open_chooser`` has no direction to happen in; the step is
    ``lambda' = (1 - dt T) lambda - dt (Delta G - T Delta log E)``."""
    from .choice import fisher
    from .rvect import ReactiveVectorSpace

    F = fisher(n)
    Q = ReactiveVectorSpace(dim=n - 1, sharp_fn=lambda lam: dt * jnp.linalg.inv(F(lam)))
    logE = jnp.zeros(n) if E is None else jnp.log(jnp.asarray(E, dtype=float))
    lam0 = jnp.zeros(n - 1) if init is None else jnp.asarray(init, dtype=float)

    def U(lam, G, n_in):
        p = jax.nn.softmax(jnp.concatenate([lam, jnp.zeros(1)]))
        return -(jnp.dot(G, p) - T * jnp.dot(logE, p) + T * jnp.sum(p * jnp.log(p)))

    grad = exact(U)

    def one_sided(q, m_out, n_in):
        xi_q, xi_m, xi_n = grad(q, m_out, n_in)
        return xi_q, jnp.zeros_like(xi_m), xi_n

    return Arrangement(
        P=configuration(Q, init=lam0),
        in_dim_M=0, out_dim_M=n, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=one_sided if unilateral else grad,
        label=f"natural-chooser(T={T:g})",
    )


def shares(lam: Array) -> Array:
    """``softmax(lambda, 0)``: the distribution a natural-chart state is."""
    return jax.nn.softmax(jnp.concatenate([lam, jnp.zeros(1)]))


# ---------------------------------------------------------------------------
# Addendum C: David's picture, nested -- a prediction discharged by the level below.
# ---------------------------------------------------------------------------


def hand(kappa: float, eta: float, init: float = 0.0) -> Arrangement:
    """The outside thing, ``I -> <0|1>``: state ``h``, Euclidean response, its own potential
    ``-1/2 kappa h^2``, publishing ``h``. It receives no command; only pullbacks reach it."""
    return Arrangement(
        P=configuration(euclidean(1, eta), init=jnp.full(1, init)),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -0.5 * kappa * q[0] ** 2),
        label="hand",
    )


def predictor(Pi: float, eta: float, goal: Tuple[float, float] = None, top: bool = False,
              unilateral: bool = False, init: float = 0.0) -> Arrangement:
    """A level ``<0|1> -> <0|1>`` (``-> I`` if ``top``): state its prediction ``mu`` of what
    arrives, cost ``1/2 Pi (m - mu)^2`` on the incoming ``m``, plus ``1/2 pi_g (mu - g)^2`` if
    ``goal = (pi_g, g)``; publishes ``mu`` upward unless ``top``. Exact, so its forcing on ``m`` is
    pulled back onto the level below -- the push that makes the input more like the prediction --
    unless ``unilateral``."""
    pi_g, g = (0.0, 0.0) if goal is None else goal

    def U(q, m_out, n_in):
        return -(0.5 * Pi * (m_out[0] - q[0]) ** 2 + 0.5 * pi_g * (q[0] - g) ** 2)

    grad = exact(U)

    def one_sided(q, m_out, n_in):
        xi_q, xi_m, xi_n = grad(q, m_out, n_in)
        return xi_q, jnp.zeros_like(xi_m), xi_n

    return Arrangement(
        P=configuration(euclidean(1, eta), init=jnp.full(1, init)),
        in_dim_M=0, out_dim_M=1, in_dim_N=0, out_dim_N=0 if top else 1,
        out_f=(lambda q, m_out: jnp.zeros(0)) if top else (lambda q, m_out: q),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=one_sided if unilateral else grad,
        label="predictor",
    )


def hierarchy(kappa: float, Pi1: float, Pi2: float, pi_g: float, g: float, eta: float,
              push1: bool = True, push2: bool = True) -> Arrangement:
    """``hand -> level 1 -> level 2``, closed; state ``((h, mu_1), mu_2)``."""
    from .wiring import compose_seq

    return compose_seq(
        compose_seq(hand(kappa, eta), predictor(Pi1, eta, unilateral=not push1)),
        predictor(Pi2, eta, goal=(pi_g, g), top=True, unilateral=not push2),
    )
