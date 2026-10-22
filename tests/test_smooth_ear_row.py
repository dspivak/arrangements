"""The smooth-cost row of the book's Ch 4 diagram (plan P68 (a), 2ade836b).

The row is Gauss^infty -> Breg^infty -> Fall^infty -> Att^infty: each ear
condition on transfers, with costs any smooth real function. The hand proof is
``notes/runs/p68-a-2ade836b/RESULTS.md`` in the book repository: the paper's
substitution (``background/affine_quadratic_operad.tex`` eq:composite-state,
eq:composite-transfers) computes a composite's transfers without consulting
costs, and each condition is on transfers only.

What these tests do: build operations with presentations and with costs that are
not quadratic, substitute them in an uneven branching nest (arities 2, 1, 0) and
in a three-level nest, and check that the composite presentation of RESULTS §1
(stacked sites, direct sums of generators), computed here without ``substitute``,
presents the composite's transfers; that the composite is the lens composite in
``arr`` (forcing included, and ``Phi`` trajectories); and the unit laws. They
check the derivation and its code, not any interpretation of it.
"""

from typing import Callable, Dict, List, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arrangements.attention import (
    Transfer, attention_arrangement, msg_dim, input_dim, substitute,
    unit_arrangement, zero_transfer,
)
from arrangements.coalgebra import configuration
from arrangements.rvect import euclidean

from tests.test_attention_operad import _agrees_with_composition, assert_same_operation

jax.config.update("jax_enable_x64", True)

DH, DP = 2, 2
MDIM, EDIM = msg_dim(DH), input_dim(DH, DP)
QD = 3          # configuration dimension of each generator


# ---------------------------------------------------------------------------
# Presentations (RESULTS §1): a site map and, per ordered pair, (phi, L, L').
# ---------------------------------------------------------------------------


def half_sq(x):
    return 0.5 * jnp.vdot(x, x)


def entropy(x):          # convex on (0, inf)^c
    return jnp.sum(x * jnp.log(x) - x)


def logsumexp(x):        # convex, not strictly convex (constant along (1, ..., 1))
    return jnp.log(jnp.sum(jnp.exp(x)))


def bregman(phi, x, y):
    """D_phi(x, y) = phi(x) - phi(y) - <grad phi(y), x - y>."""
    return phi(x) - phi(y) - jnp.vdot(jax.grad(phi)(y), x - y)


Array = jnp.ndarray


class Gen(NamedTuple):
    phi: Callable
    L: Array
    Lp: Array


class Pres(NamedTuple):
    """A Bregman presentation: site map ``s : Q -> R^d`` and generators per pair."""
    d: int
    site: Callable
    gens: Dict[Tuple[int, int], Gen]


def exponent(p: Pres, q, i, j):
    g = p.gens[(i, j)]
    s = p.site(q)
    return bregman(g.phi, g.L @ s, g.Lp @ s)


# Falloff needs no presentation: the exponent g_ij = -log gamma_ij is determined.
FallPres = Dict[Tuple[int, int], Callable]


# ---------------------------------------------------------------------------
# Generators: operations of Att^infty whose transfers meet a condition, with a
# cost that is not quadratic in the messages and position.
# ---------------------------------------------------------------------------


def smooth_cost(seed):
    r = np.random.default_rng(seed)
    c = jnp.asarray(r.standard_normal(QD))
    u = jnp.asarray(r.standard_normal(EDIM))

    def k(q, m_out, n_in):
        return (0.3 * jnp.log1p(jnp.vdot(m_out, m_out) + jnp.vdot(n_in, n_in))
                + 0.2 * jnp.cos(jnp.vdot(c, q)) * jnp.tanh(jnp.vdot(u, n_in))
                + 0.01 * jnp.sum(n_in ** 4))
    return k


def value_maps(n, seed):
    r = np.random.default_rng(seed)
    A = jnp.asarray(r.standard_normal((n + 1, n + 1, DH, DH)) * 0.5)
    B = jnp.asarray(r.standard_normal((n + 1, n + 1, DH, DH)) * 0.3)
    return lambda q, i, j: A[i, j] + jnp.sin(q[0]) * B[i, j]     # noncommuting


def operation(n, exponent_fn, seed, label):
    """An n-ary operation of Att^infty with multiplier exp(-exponent_fn(q, i, j))."""
    r = np.random.default_rng(seed)
    P = jnp.asarray(r.standard_normal((n, DP, QD)) * 0.7)
    W = value_maps(n, seed + 1)

    def placement(q):
        return jnp.tanh(jnp.einsum("nkq,q->nk", P, q)) if n else jnp.zeros((0, DP))

    def transfers(q):
        rows = []
        for i in range(n + 1):
            row = []
            for j in range(n + 1):
                if i == j:
                    row.append(zero_transfer(DH))
                else:
                    g = jnp.exp(-exponent_fn(q, i, j))
                    row.append(Transfer(gamma=g, L=g * W(q, i, j)))
            rows.append(tuple(row))
        return tuple(rows)

    init = jnp.asarray(r.standard_normal(QD) * 0.5)
    return attention_arrangement(
        n, DH, DP, placement, transfers, cost=smooth_cost(seed + 2),
        parameter=configuration(euclidean(QD, 0.05), init=init), label=label,
    )


def pairs(n):
    return [(i, j) for i in range(n + 1) for j in range(n + 1) if i != j]


def bregman_part(n, seed, kind):
    """kind: 'gauss' (phi = |.|^2/2, L' = 0), 'entropy', or 'lse'."""
    r = np.random.default_rng(seed)
    d = 2
    M = jnp.asarray(r.standard_normal((d, QD)) * 0.6)
    if kind == "entropy":
        site = lambda q: jnp.exp(jnp.tanh(M @ q))                 # positive sites
    else:
        site = lambda q: jnp.tanh(M @ q) + 0.3 * (M @ q)
    gens = {}
    for (i, j) in pairs(n):
        c = int(r.integers(1, 3))
        if kind == "gauss":
            gens[(i, j)] = Gen(half_sq, jnp.asarray(r.standard_normal((c, d))), jnp.zeros((c, d)))
        elif kind == "entropy":
            gens[(i, j)] = Gen(entropy, jnp.asarray(r.uniform(0.2, 1.2, (c, d))),
                               jnp.asarray(r.uniform(0.2, 1.2, (c, d))))
        else:
            gens[(i, j)] = Gen(logsumexp, jnp.asarray(r.standard_normal((c, d))),
                               jnp.asarray(r.standard_normal((c, d))))
    pres = Pres(d, site, gens)
    a = operation(n, lambda q, i, j: exponent(pres, q, i, j), seed, f"{kind}{seed}/{n}")
    return a, pres


def falloff_part(n, seed):
    r = np.random.default_rng(seed)
    U = {p: jnp.asarray(r.standard_normal(QD)) for p in pairs(n)}
    V = {p: jnp.asarray(r.standard_normal(QD)) for p in pairs(n)}
    g = {p: (lambda q, u=U[p], v=V[p]: jnp.log1p(jnp.vdot(u, q) ** 2) + jnp.sin(jnp.vdot(v, q)) ** 2)
         for p in pairs(n)}
    a = operation(n, lambda q, i, j: g[(i, j)](q), seed, f"fall{seed}/{n}")
    return a, g


def unit_pres():
    return Pres(0, lambda q: jnp.zeros(0),
                {(0, 1): Gen(half_sq, jnp.zeros((0, 0)), jnp.zeros((0, 0))),
                 (1, 0): Gen(half_sq, jnp.zeros((0, 0)), jnp.zeros((0, 0)))})


# ---------------------------------------------------------------------------
# The composite presentation (RESULTS §1), computed without ``substitute``.
# ---------------------------------------------------------------------------


def path(arities, x, y):
    """The communication path of eq:composite-transfers for the composite pair
    (x, y), as a list of (constituent, (i, j)); constituent n means the outer one.
    Composite indices: 0 the exterior, children (i, j) flattened in order."""
    n = len(arities)
    flat = [(i, j) for i in range(n) for j in range(1, arities[i] + 1)]
    if x == 0:
        i, j = flat[y - 1]
        return [(n, (0, i + 1)), (i, (0, j))]
    if y == 0:
        i, j = flat[x - 1]
        return [(i, (j, 0)), (n, (i + 1, 0))]
    (i, j), (r, s) = flat[x - 1], flat[y - 1]
    if i == r:
        return [(i, (j, s))]
    return [(i, (j, 0)), (n, (i + 1, r + 1)), (r, (0, s))]


def direct_sum(phis, dims):
    cuts = np.cumsum([0] + dims)

    def phi(x):
        return sum(f(x[cuts[t]:cuts[t + 1]]) for t, f in enumerate(phis))
    return phi


def compose_pres(pa: Pres, parts: List[Pres], arities, qdims) -> Pres:
    """Stacked site in the library's factor order (parts, then the outer), and for
    each pair the direct sum of its path's generators, each L precomposed with the
    projection of the composite site onto its constituent's block."""
    cons = list(parts) + [pa]
    ds = [p.d for p in cons]
    dcut = np.cumsum([0] + ds)
    qcut = np.cumsum([0] + list(qdims))
    D = int(dcut[-1])

    def site(q):
        blocks = [c.site(q[qcut[t]:qcut[t + 1]]) for t, c in enumerate(cons)]
        return jnp.concatenate(blocks) if blocks else jnp.zeros(0)

    def proj(t):
        Pm = np.zeros((ds[t], D))
        Pm[:, dcut[t]:dcut[t + 1]] = np.eye(ds[t])
        return jnp.asarray(Pm)

    N = sum(arities)
    gens = {}
    for x in range(N + 1):
        for y in range(N + 1):
            if x == y:
                continue
            steps = path(arities, x, y)
            gs = [(t, cons[t].gens[ij]) for t, ij in steps]
            phi = direct_sum([g.phi for _, g in gs], [g.L.shape[0] for _, g in gs])
            L = jnp.concatenate([g.L @ proj(t) for t, g in gs], axis=0)
            Lp = jnp.concatenate([g.Lp @ proj(t) for t, g in gs], axis=0)
            gens[(x, y)] = Gen(phi, L, Lp)
    return Pres(D, site, gens)


def qdims_of(a, parts):
    return [p.P.Q_dim for p in parts] + [a.P.Q_dim]


def assert_presents(b, pres, seed, tol=1e-11):
    """``pres`` presents ``b``'s transfers: gamma = exp(-D) in (0, 1] for every pair."""
    r = np.random.default_rng(seed)
    for _ in range(3):
        q = jnp.asarray(r.standard_normal(b.P.Q_dim) * 0.8)
        T = b.transfers(q)
        for (x, y) in pairs(b.n):
            gam = float(T[x][y].gamma)
            assert 0.0 < gam <= 1.0 + 1e-15
            np.testing.assert_allclose(gam, float(jnp.exp(-exponent(pres, q, x, y))), rtol=tol, atol=tol)


def assert_not_quadratic(b, seed):
    """The composite's cost has a nonzero fourth difference along a line in the
    messages and position: the test is in Att^infty, not in Att."""
    r = np.random.default_rng(seed)
    q = jnp.asarray(r.standard_normal(b.P.Q_dim) * 0.5)
    m = jnp.asarray(r.standard_normal(b.n * MDIM))
    e = jnp.asarray(r.standard_normal(EDIM))
    vm, ve = jnp.asarray(r.standard_normal(b.n * MDIM)), jnp.asarray(r.standard_normal(EDIM))
    h = 0.5
    f = [float(b.cost(q, m + t * h * vm, e + t * h * ve)) for t in range(5)]
    fourth = f[4] - 4 * f[3] + 6 * f[2] - 4 * f[1] + f[0]
    assert abs(fourth) > 1e-4


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------

KINDS = ["gauss", "entropy", "lse"]


@pytest.mark.parametrize("kind", KINDS)
def test_bregman_presentations_compose_on_an_uneven_nest(kind):
    """PROVES BY EXAMPLE what RESULTS §2 (1) proves: for arities 2, 1, 0 under a
    ternary outer operation, all with costs that are not quadratic, the composite
    presentation computed from the constituents' presents the composite's
    transfers, and the composite is the lens composite in arr (forcing and
    trajectories). 'gauss' is the Gaussian row; 'lse' has a generator that is
    convex and not strictly convex."""
    a, pa = bregman_part(3, 10, kind)
    built = [bregman_part(2, 11, kind), bregman_part(1, 12, kind), bregman_part(0, 13, kind)]
    parts, pparts = [x for x, _ in built], [p for _, p in built]
    b = _agrees_with_composition(a, parts, seed=201)
    pres = compose_pres(pa, pparts, [p.n for p in parts], qdims_of(a, parts))
    assert_presents(b, pres, seed=202)
    assert_not_quadratic(b, seed=203)


def test_gaussian_composite_is_gaussian():
    """RESULTS §2 (1), Gaussian clause: the direct sum of the generators |.|^2/2 is
    |.|^2/2 on the stacked space, and L' stays zero, so the composite presentation
    of Gaussian presentations is Gaussian, and Gauss -> Breg commutes with
    composites."""
    a, pa = bregman_part(2, 20, "gauss")
    built = [bregman_part(2, 21, "gauss"), bregman_part(1, 22, "gauss")]
    parts, pparts = [x for x, _ in built], [p for _, p in built]
    pres = compose_pres(pa, pparts, [2, 1], qdims_of(a, parts))
    r = np.random.default_rng(23)
    for g in pres.gens.values():
        x = jnp.asarray(r.standard_normal(g.L.shape[0]))
        np.testing.assert_allclose(float(g.phi(x)), float(half_sq(x)), rtol=1e-13)
        assert float(jnp.max(jnp.abs(g.Lp))) == 0.0


def test_falloff_composes_on_an_uneven_nest():
    """RESULTS §2 (1), falloff: every composite multiplier lies in (0, 1], and its
    exponent -log gamma is the sum of the constituent exponents along the path."""
    a, ga = falloff_part(3, 30)
    built = [falloff_part(2, 31), falloff_part(1, 32), falloff_part(0, 33)]
    parts, gparts = [x for x, _ in built], [g for _, g in built]
    b = _agrees_with_composition(a, parts, seed=301)
    arities = [p.n for p in parts]
    cons = gparts + [ga]
    qcut = np.cumsum([0] + qdims_of(a, parts))
    r = np.random.default_rng(302)
    for _ in range(3):
        q = jnp.asarray(r.standard_normal(b.P.Q_dim) * 0.8)
        T = b.transfers(q)
        for (x, y) in pairs(b.n):
            gam = float(T[x][y].gamma)
            assert 0.0 < gam <= 1.0 + 1e-15
            want = sum(float(cons[t][ij](q[qcut[t]:qcut[t + 1]])) for t, ij in path(arities, x, y))
            np.testing.assert_allclose(-np.log(gam), want, rtol=1e-11, atol=1e-12)
    assert_not_quadratic(b, seed=303)


@pytest.mark.parametrize("kind", ["entropy", "lse"])
def test_three_level_nest_presented_both_ways(kind):
    """Associativity at the level of presentations: a three-level nest, bracketed
    either way, is presented by the composite presentation built the same way."""
    a, pa = bregman_part(2, 40, kind)
    m1, pm1 = bregman_part(2, 41, kind)
    m2, pm2 = bregman_part(1, 42, kind)
    l1, pl1 = bregman_part(1, 43, kind)
    l2, pl2 = bregman_part(0, 44, kind)
    l3, pl3 = bregman_part(2, 45, kind)
    # inner first: a(m1(l1, l2), m2(l3))
    i1, i2 = substitute(m1, [l1, l2]), substitute(m2, [l3])
    pi1 = compose_pres(pm1, [pl1, pl2], [1, 0], qdims_of(m1, [l1, l2]))
    pi2 = compose_pres(pm2, [pl3], [2], qdims_of(m2, [l3]))
    inner = substitute(a, [i1, i2])
    assert_presents(inner, compose_pres(pa, [pi1, pi2], [1, 2], qdims_of(a, [i1, i2])), seed=401)
    # outer first: (a(m1, m2))(l1, l2, l3)
    o = substitute(a, [m1, m2])
    po = compose_pres(pa, [pm1, pm2], [2, 1], qdims_of(a, [m1, m2]))
    outer = substitute(o, [l1, l2, l3])
    assert_presents(outer, compose_pres(po, [pl1, pl2, pl3], [1, 0, 2], qdims_of(o, [l1, l2, l3])), seed=402)


@pytest.mark.parametrize("kind", KINDS)
def test_unit_laws_with_presentations(kind):
    """The unit is in every family (multiplier 1, the empty presentation), and
    a(1, ..., 1) = a = 1(a) as operations, the composite presentation presenting
    each."""
    one, p1 = unit_arrangement(DH, DP), unit_pres()
    a, pa = bregman_part(2, 50, kind)
    right = substitute(a, [one, one])
    left = substitute(one, [a])
    q = jnp.asarray(np.random.default_rng(51).standard_normal(a.P.Q_dim) * 0.5)
    assert_same_operation(right, a, q, q, seed=52)
    assert_same_operation(left, a, q, q, seed=53)
    assert_presents(right, compose_pres(pa, [p1, p1], [1, 1], qdims_of(a, [one, one])), seed=54)
    assert_presents(left, compose_pres(p1, [pa], [2], qdims_of(one, [a])), seed=55)
