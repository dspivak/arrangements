"""Exponential-family costs in the attention operad (book plan P5 (h), b25567c3, 2026-10-19).

A cost of the form ``k(q, d) = A(theta(q, d)) - <theta(q, d), x(q, d)>``, with ``theta`` and ``x``
affine in the interaction ``d = (m, e)`` at each configuration ``q`` and ``A`` a log-partition
function (Gaussian, Poisson, Bernoulli, categorical), on the arrangements of
``background/affine_quadratic_operad.tex`` with its ``k >= 0`` and degree-two conditions dropped.
The claims checked: substitution (eq:composite-cost) of such costs is again one such cost, for the
direct sum of the ``A``'s and the stacked, pulled-back ``theta`` and ``x``; it agrees with the
composite in ``arr``; the force on ``theta`` is ``x - grad A(theta)``; and a cost whose
precision is a received weight is not defined on admissible interactions.
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from arrangements.attention import (
    attention_arrangement, spatial_arrangement, spatial_placement, spatial_Q_dim,
    spatial_transfers, substitute,
)
from arrangements.coalgebra import configuration
from arrangements.rvect import euclidean
from tests.test_attention_operad import DH, DP, EDIM, MDIM, TAU, _agrees_with_composition


# ---------------------------------------------------------------------------
# Log-partition functions and exponential-family terms.
# ---------------------------------------------------------------------------

def gauss_A(t):            # N(theta, I): A = |theta|^2 / 2, Theta = R^k
    return 0.5 * jnp.sum(t * t)


def poisson_A(t):          # Poisson with log-rate theta: A = sum exp(theta)
    return jnp.sum(jnp.exp(t))


def bernoulli_A(t):        # Bernoulli with log-odds theta: A = sum log(1 + exp(theta))
    return jnp.sum(jnp.logaddexp(0.0, t))


def categorical_A(t):      # categorical with logits theta: A = log sum exp(theta)
    return jax.scipy.special.logsumexp(t)


class Term(NamedTuple):
    """``A(theta(q, m, e)) - <theta, x(q, m, e)>``; ``A`` sees the whole vector ``theta``."""
    A: Callable
    theta: Callable
    x: Callable

    def cost(self, q, m, e):
        t = self.theta(q, m, e)
        return self.A(t) - jnp.dot(t, self.x(q, m, e))


def direct_sum(terms, dims):
    """``A_1 (+) ... (+) A_r`` on stacked natural parameters, cut by ``dims``."""
    cut = np.cumsum([0] + list(dims))

    def A(t):
        return sum(T.A(t[cut[i]:cut[i + 1]]) for i, T in enumerate(terms))

    return A


def affine_term(A, k, n, seed, scale=0.4):
    """A term whose ``theta`` and ``x`` in ``R^k`` are affine in ``(m, e)`` with smooth ``q``-dependence."""
    r = np.random.default_rng(seed)
    L1 = jnp.asarray(r.standard_normal((k, n * MDIM + EDIM)) * scale)
    L2 = jnp.asarray(r.standard_normal((k, n * MDIM + EDIM)) * scale)
    c1, c2 = jnp.asarray(r.standard_normal(k) * scale), jnp.asarray(r.standard_normal(k) * scale)
    theta = lambda q, m, e: L1 @ jnp.concatenate([m, e]) + c1 * jnp.cos(jnp.sum(q))
    x = lambda q, m, e: L2 @ jnp.concatenate([m, e]) + c2 * jnp.sin(jnp.sum(q))
    return Term(A, theta, x)


def gaussian_prediction_term(n, seed):
    """``(1/2)|e - pi(q, m)|^2`` as one Gaussian term: ``theta = (pi, e)``, ``x = (e, 0)``."""
    r = np.random.default_rng(seed)
    Pm = jnp.asarray(r.standard_normal((EDIM, n * MDIM)) * 0.3)
    c = jnp.asarray(r.standard_normal(EDIM) * 0.3)
    pi = lambda q, m: Pm @ m + c + 0.2 * jnp.sum(q)
    theta = lambda q, m, e: jnp.concatenate([pi(q, m), e])
    x = lambda q, m, e: jnp.concatenate([e, jnp.zeros(EDIM)])
    return Term(gauss_A, theta, x), pi


def part(n, term, seed, mobility=0.05):
    r = np.random.default_rng(seed)
    init = jnp.asarray(r.standard_normal(spatial_Q_dim(n, DH, DP)) * 0.6)
    a = spatial_arrangement(n, DH, DP, TAU, cost=term.cost, init=init, mobility=mobility,
                            label=f"x{seed}/{n}")
    return a


# ---------------------------------------------------------------------------
# The composite's stacked term, from eq:local-interactions.
# ---------------------------------------------------------------------------

def composite_term(a, ta, parts, tparts, kdims):
    """The one term the prediction says ``substitute(a, parts)`` has: ``A_b`` the direct sum, ``theta_b``
    and ``x_b`` the constituents' evaluated at their local interactions (eq:local-interactions),
    stacked. Configuration order as in ``substitute``: ``Q_{a_1} ... Q_{a_n} Q_a``."""
    n, ar = a.n, [p.n for p in parts]
    starts = np.cumsum([0] + ar)
    qd = [p.P.Q_dim for p in parts]
    qcut = np.cumsum([0] + qd)

    def local(q, m, e):
        qs = [q[qcut[i]:qcut[i + 1]] for i in range(n)]
        qa = q[qcut[n]:qcut[n] + a.P.Q_dim]
        ms = [m[starts[i] * MDIM:starts[i + 1] * MDIM] for i in range(n)]
        mt = jnp.concatenate([parts[i].sigma(qs[i], ms[i]) for i in range(n)])
        ei = a.rho(qa, mt, e)
        return [(qa, mt, e)] + [(qs[i], ms[i], ei[i * EDIM:(i + 1) * EDIM]) for i in range(n)]

    terms = [ta] + list(tparts)
    theta = lambda q, m, e: jnp.concatenate([T.theta(*z) for T, z in zip(terms, local(q, m, e))])
    x = lambda q, m, e: jnp.concatenate([T.x(*z) for T, z in zip(terms, local(q, m, e))])
    return Term(direct_sum(terms, kdims), theta, x)


def build():
    """Parent 3-ary (Gaussian prediction (+) Bernoulli); children 1-ary Poisson, 2-ary categorical,
    0-ary Gaussian prediction."""
    tg, _ = gaussian_prediction_term(3, 11)
    tb = affine_term(bernoulli_A, 2, 3, 12)
    ta = Term(direct_sum([tg, tb], [2 * EDIM, 2]),
              lambda q, m, e: jnp.concatenate([tg.theta(q, m, e), tb.theta(q, m, e)]),
              lambda q, m, e: jnp.concatenate([tg.x(q, m, e), tb.x(q, m, e)]))
    t1 = affine_term(poisson_A, 2, 1, 13)
    t2 = affine_term(categorical_A, 3, 2, 14)
    t3, _ = gaussian_prediction_term(0, 15)
    a = part(3, ta, 21)
    parts = [part(1, t1, 22), part(2, t2, 23), part(0, t3, 24)]
    return a, ta, parts, [t1, t2, t3], [2 * EDIM + 2, 2, 3, 2 * EDIM]


def random_point(b, seed):
    r = np.random.default_rng(seed)
    return (jnp.asarray(r.standard_normal(b.P.Q_dim) * 0.5),
            jnp.asarray(r.standard_normal(b.n * MDIM)),
            jnp.asarray(r.standard_normal(EDIM)))


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------

def test_gaussian_prediction_cost_is_one_gaussian_term():
    """``(1/2)|e - pi|^2 = A(pi, e) - <(pi, e), (e, 0)>`` for ``A = |.|^2/2``: the paper's prediction
    cost is in the family, via a stacked term."""
    t, pi = gaussian_prediction_term(2, 1)
    r = np.random.default_rng(2)
    for _ in range(5):
        q, m, e = (jnp.asarray(r.standard_normal(4)), jnp.asarray(r.standard_normal(2 * MDIM)),
                   jnp.asarray(r.standard_normal(EDIM)))
        d = e - pi(q, m)
        np.testing.assert_allclose(t.cost(q, m, e), 0.5 * jnp.dot(d, d), atol=1e-12)


def test_composite_cost_is_one_stacked_term_with_affine_theta():
    """The prediction's item 1: the composite cost of exponential-family costs is one term, for the
    direct sum of log-partition functions, and its ``theta_b``, ``x_b`` are affine in ``(m, e)``."""
    a, ta, parts, tparts, kdims = build()
    b = substitute(a, parts)
    tb = composite_term(a, ta, parts, tparts, kdims)
    worst_cost, worst_aff = 0.0, 0.0
    for s in range(6):
        q, m, e = random_point(b, 100 + s)
        worst_cost = max(worst_cost, abs(float(b.cost(q, m, e) - tb.cost(q, m, e))))
        _, m2, e2 = random_point(b, 200 + s)
        for f in (tb.theta, tb.x):
            mid = f(q, (m + m2) / 2, (e + e2) / 2)
            worst_aff = max(worst_aff, float(jnp.max(jnp.abs(f(q, m, e) + f(q, m2, e2) - 2 * mid))))
    print(f"composite cost vs stacked term: {worst_cost:.2e}; second difference of theta_b, x_b: {worst_aff:.2e}")
    assert worst_cost < 1e-12 and worst_aff < 1e-12
    # and it is not of degree two: the fourth difference along a line in d is not zero
    q, m, e = random_point(b, 300)
    _, dm, de = random_point(b, 301)
    f = lambda s: b.cost(q, m + s * dm, e + s * de)
    fourth = float(f(2.0) - 4 * f(1.0) + 6 * f(0.0) - 4 * f(-1.0) + f(-2.0))
    print(f"fourth difference of the composite cost along a line: {fourth:.3e}")
    assert abs(fourth) > 1e-3


def test_exponential_family_substitution_agrees_with_composition_in_arr():
    """``substitute`` against ``compose_seq(tensor(parts), a)`` in ``arr``: sending, receiving, the whole
    forcing, and 40 steps of ``Phi``. ``substitute``'s cost code is cost-agnostic, so this checks the
    port on non-quadratic costs, not new mathematics."""
    a, _, parts, _, _ = build()
    _agrees_with_composition(a, parts, seed=401)


def test_force_on_theta_is_observed_minus_expected():
    """``-dk = J_theta^T (x - grad A(theta)) + J_x^T theta`` on the interaction, for the composite: what
    is pulled back to ``theta`` is the prediction error in mean coordinates, ``x - E_theta[x]``."""
    a, ta, parts, tparts, kdims = build()
    b = substitute(a, parts)
    tb = composite_term(a, ta, parts, tparts, kdims)
    worst = 0.0
    for s in range(4):
        q, m, e = random_point(b, 500 + s)
        _, xi_m, xi_e = b.arr().omega(q, m, e)
        d = jnp.concatenate([m, e])
        nm = m.shape[0]
        th = lambda d: tb.theta(q, d[:nm], d[nm:])
        xx = lambda d: tb.x(q, d[:nm], d[nm:])
        t, x = th(d), xx(d)
        pred = jax.jacfwd(th)(d).T @ (x - jax.grad(tb.A)(t)) + jax.jacfwd(xx)(d).T @ t
        worst = max(worst, float(jnp.max(jnp.abs(jnp.concatenate([xi_m, xi_e]) - pred))))
    print(f"force on the interaction vs observed-minus-expected pulled back: {worst:.2e}")
    assert worst < 1e-11


def test_precision_from_a_received_weight_is_not_defined_on_admissible_interactions():
    """The domain condition. A Gaussian term whose precision is the received weight ``w``
    (``theta = (h, -w/2)``, ``x = (y, y^2)``, ``A = -theta_1^2/(4 theta_2) - log(-2 theta_2)/2``) needs
    ``w > 0``. In ``a(p, z)``, ``z`` nullary, the child ``p`` receives ``T_10 m_0 + T_12 0_M``: with the
    admissible exterior message ``0_M`` its weight is ``0`` and the cost is not finite."""
    def A(t):
        return -t[0] ** 2 / (4 * t[1]) - 0.5 * jnp.log(-2 * t[1])

    y = 0.7
    prec = Term(A, lambda q, m, e: jnp.stack([e[0], -0.5 * e[DH]]),
                lambda q, m, e: jnp.array([y, y * y]))
    p = part(1, prec, 31)
    z = part(0, Term(gauss_A, lambda q, m, e: jnp.zeros(1), lambda q, m, e: jnp.zeros(1)), 32)
    a = part(2, Term(gauss_A, lambda q, m, e: jnp.zeros(1), lambda q, m, e: jnp.zeros(1)), 33)
    b = substitute(a, [p, z])
    q, m, _ = random_point(b, 600)
    e_ok = jnp.concatenate([jnp.array([0.3, 0.0, 1.0]), jnp.zeros(DP)])     # m_0 = (h, w=1)
    e_zero = jnp.zeros(EDIM)                                                 # m_0 = 0_M, admissible
    k_ok, k_zero = float(b.cost(q, m, e_ok)), float(b.cost(q, m, e_zero))
    print(f"cost with exterior weight 1: {k_ok:.4f}; with the admissible zero message: {k_zero}")
    assert np.isfinite(k_ok) and not np.isfinite(k_zero)
