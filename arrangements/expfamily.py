"""Exponential-family operads as data (book P53 (h), continuing P5 (h2)).

The mathematics is the book's ``notes/runs/p53h-82601336/RESULTS.md`` §1. An operation is attention data
``(I, l, T)`` (``attention.py``, with zero cost) and a list of *terms* ``(A, T, theta, y)``: an exponential family
(log-partition ``A``, sufficient statistic ``T``), an *expectation* ``theta(q, m, e)`` in natural coordinates and an
*observation* ``y(q, m, e)``. The cost is ``sum_t A_t(theta_t) - <theta_t, T_t(y_t)>``. ``substitute`` composes term
lists by pulling each term back along its local interaction (the paper's eq:local-interactions), using only the
constituents' sending and receiving maps; the attention operad's own ``substitute`` composes costs independently, and the
two are compared in the tests.

``gaussian_term`` is the Gaussian with a precision function: ``T(y) = (y, -1/2 y y^T)``, ``theta = (Pi mu, Pi)``,
``A(eta, Lam) = 1/2 eta^T Lam^{-1} eta - 1/2 log det Lam``, so the cost is
``1/2 (y - mu)^T Pi (y - mu) - 1/2 log det Pi``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, List, NamedTuple, Sequence

import jax.numpy as jnp
from jax import Array

from .attention import AttentionArrangement, input_dim, msg_dim, split_messages, substitute as att_substitute, zero_cost


class Term(NamedTuple):
    """One term: ``A(theta) - <theta, T(y)>`` with ``theta = theta(q, m, e)``, ``y = y(q, m, e)``."""

    A: Callable[[Array], Array]
    T: Callable[[Array], Array]
    theta: Callable[[Array, Array, Array], Array]
    y: Callable[[Array, Array, Array], Array]


def term_cost(t: Term, q: Array, m: Array, e: Array) -> Array:
    th = t.theta(q, m, e)
    return t.A(th) - th @ t.T(t.y(q, m, e))


@dataclass(frozen=True)
class ExpFamilyOperation:
    """Attention data ``base`` (its cost ignored) and a list of terms."""

    base: AttentionArrangement
    terms: List[Term]

    def cost(self, q: Array, m: Array, e: Array) -> Array:
        return sum((term_cost(t, q, m, e) for t in self.terms), jnp.zeros(()))

    def attention(self) -> AttentionArrangement:
        """Forget the presentation: the attention arrangement with this cost (any smooth cost)."""
        return replace(self.base, cost=self.cost)


def substitute(a: ExpFamilyOperation, parts: Sequence[ExpFamilyOperation]) -> ExpFamilyOperation:
    """``a(a_1, ..., a_n)``: transfers and placements by ``attention.substitute`` (with zero costs), terms by
    pulling back along the local interactions, in the order ``a``'s terms, then ``a_1``'s, and so on."""
    parts = list(parts)
    n, dim_H, dim_P = a.base.n, a.base.dim_H, a.base.dim_P
    arities = [p.base.n for p in parts]
    starts = [sum(arities[:i]) for i in range(n + 1)]
    N = starts[-1]
    qd = [p.base.P.Q_dim for p in parts]
    qcut = [sum(qd[:i]) for i in range(n + 1)]
    Qa, edim = a.base.P.Q_dim, input_dim(dim_H, dim_P)

    def local(q, m, e):
        """``(q_a, m~, e)`` and, per part, ``(q_i, m_i, e_i)`` (eq:local-interactions)."""
        qs = [q[qcut[i]:qcut[i + 1]] for i in range(n)]
        qa = q[qcut[n]:qcut[n] + Qa]
        ms = split_messages(m, N, dim_H)
        mi = [jnp.ravel(ms[starts[i]:starts[i + 1]]) for i in range(n)]
        tilde = [parts[i].base.sigma(qs[i], mi[i]) for i in range(n)]
        mt = jnp.concatenate(tilde) if tilde else jnp.zeros(0)
        ei = a.base.rho(qa, mt, e)
        return (qa, mt, e), [(qs[i], mi[i], ei[i * edim:(i + 1) * edim]) for i in range(n)]

    def pull_a(t: Term) -> Term:
        return Term(t.A, t.T, lambda q, m, e: t.theta(*local(q, m, e)[0]),
                    lambda q, m, e: t.y(*local(q, m, e)[0]))

    def pull_i(t: Term, i: int) -> Term:
        return Term(t.A, t.T, lambda q, m, e: t.theta(*local(q, m, e)[1][i]),
                    lambda q, m, e: t.y(*local(q, m, e)[1][i]))

    terms = [pull_a(t) for t in a.terms]
    for i, p in enumerate(parts):
        terms += [pull_i(t, i) for t in p.terms]
    base = att_substitute(replace(a.base, cost=zero_cost), [replace(p.base, cost=zero_cost) for p in parts])
    return ExpFamilyOperation(base, terms)


# ---------------------------------------------------------------------------
# The Gaussian with a precision function.
# ---------------------------------------------------------------------------


def gaussian_A(r: int) -> Callable[[Array], Array]:
    """``A(eta, Lam) = 1/2 eta^T Lam^{-1} eta - 1/2 log det Lam`` on ``theta = (eta, vec Lam)``."""

    def A(theta):
        eta, Lam = theta[:r], jnp.reshape(theta[r:], (r, r))
        return 0.5 * eta @ jnp.linalg.solve(Lam, eta) - 0.5 * jnp.linalg.slogdet(Lam)[1]

    return A


def gaussian_T(r: int) -> Callable[[Array], Array]:
    """``T(y) = (y, -1/2 vec(y y^T))``."""
    return lambda y: jnp.concatenate([y, -0.5 * jnp.ravel(jnp.outer(y, y))])


def gaussian_term(r: int, mu, Pi, y) -> Term:
    """A Gaussian term from a prediction ``mu(q, m, e)``, a precision function ``Pi(q) -> PD_r`` and an observation
    ``y(q, m, e)``: ``theta = (Pi mu, vec Pi)``."""

    def theta(q, m, e):
        P = Pi(q)
        return jnp.concatenate([P @ mu(q, m, e), jnp.ravel(P)])

    return Term(gaussian_A(r), gaussian_T(r), theta, y)


def cholesky_precision(r: int, offset: int) -> Callable[[Array], Array]:
    """``Pi(q) = L L^T`` with ``L`` lower triangular read from ``q[offset:offset + r(r+1)/2]``, its diagonal
    exponentiated: a smooth map onto the positive-definite matrices."""
    rows, cols = jnp.tril_indices(r)

    def Pi(q):
        v = q[offset:offset + rows.shape[0]]
        L = jnp.zeros((r, r)).at[rows, cols].set(v)
        L = L.at[jnp.diag_indices(r)].set(jnp.exp(jnp.diag(L)))
        return L @ L.T

    return Pi
