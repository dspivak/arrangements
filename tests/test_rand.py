"""Randomness: the pairing with a sampler against the monad's exact law (STATUS.md §3.12).

A closed particle with cost ``k(q) = (a/2)|q|^2`` and response ``eta`` moves by
``q' = c q + noise``, ``c = 1 - eta a``. Four checks, each against an
independent computation:

* the unit: a deterministic parameter lifted into ``rand`` or ``rand_G`` and
  seeded runs exactly as the deterministic one;
* finite kicks: the exact law of ``K`` steps three ways -- ``Phi`` of the
  stochastic arrangement iterated by the monad's ``bind``, the seeded arrangement
  run on every replayed index sequence and weighted, and the closed form;
* Gaussian kicks: the law of ``K`` steps by ``bind`` is affine in the seed, so its
  mean and covariance are exact and match the closed form; a replayed seed gives
  the bound random variable's value pathwise; and a long pseudo-random run has
  the discrete chain's stationary variance ``(T/a)/(1 - eta a/2)``, not Gibbs's ``T/a``;
* the asymmetry: ``A (x) B`` and ``B (x) A`` have the same law, and on one shared
  seed they give ``A`` different paths; seeding each box separately does not.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact
from arrangements.coalgebra import configuration
from arrangements.functors import Phi
from arrangements.rand import (
    FiniteRV, GaussianRV, euler_maruyama, key_sampler, kicked_configuration,
    lift, replay_sampler, seed,
)
from arrangements.rvect import euclidean
from arrangements.wiring import parallel_arrangements

from tests.chain import IN_POS

_TRIV = lambda _o: (jnp.zeros(0), jnp.zeros(0))


def particle(P, a, label="particle"):
    """A closed arrangement on the parameter ``P`` with cost ``(a/2)|q|^2``."""
    return Arrangement(
        P=P, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m, n: -0.5 * a * jnp.sum(q ** 2)),
        label=label,
    )


def step_of(arr):
    """One closed-system step of ``Phi(arr)``; for a stochastic ``arr`` the new
    state is a random variable (the stochastic dynamics functor)."""
    O = Phi(arr)
    return lambda s: O.with_state(s).run_one(IN_POS, _TRIV)[-1]


def run(arr, K):
    f, s = step_of(arr), arr.P.init
    for _ in range(K):
        s = f(s)
    return s


def exact_law(arr, K):
    """The law of ``K`` steps by the monad: ``bind`` the step ``K`` times."""
    f, rv = step_of(arr), arr.P.kind.unit(arr.P.init)
    for _ in range(K):
        rv = rv.bind(f)
    return rv


# ---------------------------------------------------------------------------
# The unit.
# ---------------------------------------------------------------------------


def test_unit_lift_and_seed_recovers_the_deterministic_run():
    Q, q0 = euclidean(2, eta=0.3), jnp.array([1.0, -2.0])
    det = run(particle(configuration(Q, init=q0), a=1.5), 7)
    for kind in (FiniteRV, GaussianRV):
        stoch = particle(lift(configuration(Q, init=q0), kind), a=1.5)
        s, _ = run(seed(stoch, key_sampler(0)), 7)
        np.testing.assert_allclose(s, det, atol=1e-14)


# ---------------------------------------------------------------------------
# Finite kicks: three computations of one law.
# ---------------------------------------------------------------------------

ETA, A_COST, Q0 = 0.3, 1.0, 1.5
W, KICKS = jnp.array([0.5, 0.3, 0.2]), jnp.array([[-1.0], [0.0], [2.0]])
C = 1.0 - ETA * A_COST


def _kicked():
    return particle(kicked_configuration(euclidean(1, eta=ETA), W, KICKS, init=[Q0]), A_COST)


def test_finite_law_by_monad_by_replay_and_in_closed_form():
    K = 4
    law = exact_law(_kicked(), K)
    assert law.n == 3 ** K
    m1_monad = law.expect(lambda q: q[0])
    m2_monad = law.expect(lambda q: q[0] ** 2)

    m1_replay = m2_replay = 0.0
    for seq in itertools.product(range(3), repeat=K):
        q, _ = run(seed(_kicked(), replay_sampler(jnp.array(seq))), K)
        w = np.prod([float(W[i]) for i in seq])
        m1_replay += w * q[0]
        m2_replay += w * q[0] ** 2

    mu = float(W @ KICKS[:, 0])
    var_kick = float(W @ KICKS[:, 0] ** 2) - mu ** 2
    mean = C ** K * Q0 + mu * sum(C ** j for j in range(K))
    var = var_kick * sum(C ** (2 * j) for j in range(K))

    np.testing.assert_allclose([m1_monad, m2_monad], [mean, var + mean ** 2], rtol=1e-12)
    np.testing.assert_allclose([m1_replay, m2_replay], [mean, var + mean ** 2], rtol=1e-12)


# ---------------------------------------------------------------------------
# Gaussian kicks.
# ---------------------------------------------------------------------------

T, ETA_G, A_G = 0.7, 0.5, 1.0
C_G = 1.0 - ETA_G * A_G


def _em(d=2, q0=(1.0, -0.5)):
    return particle(euler_maruyama(euclidean(d, eta=ETA_G), T, init=list(q0)), A_G)


def test_gaussian_law_by_monad_is_the_closed_form():
    K, d, q0 = 6, 2, jnp.array([1.0, -0.5])
    law = exact_law(_em(), K)
    assert law.n == K * d
    z0 = jnp.zeros(law.n)
    J = jax.jacfwd(law.ret)(z0)                   # the law is affine in the seed
    np.testing.assert_allclose(law.ret(z0), C_G ** K * q0, atol=1e-14)
    var = 2 * T * ETA_G * sum(C_G ** (2 * j) for j in range(K))
    np.testing.assert_allclose(J @ J.T, var * jnp.eye(d), atol=1e-14)


def test_gaussian_replay_is_the_bound_variable_pathwise():
    K, d = 6, 2
    zs = jax.random.normal(jax.random.PRNGKey(3), (K, d))
    q, _ = run(seed(_em(), replay_sampler(zs)), K)
    np.testing.assert_allclose(q, exact_law(_em(), K).ret(zs.reshape(-1)), atol=1e-14)


def test_stationary_variance_is_the_chains_not_gibbs():
    arr = seed(_em(d=1, q0=(0.0,)), key_sampler(11))
    f = jax.jit(step_of(arr))

    def body(s, _):
        s = f(s)
        return s, s[0][0]

    _, qs = jax.lax.scan(body, arr.P.init, None, length=40000)
    v = float(jnp.var(qs[100:]))
    chain, gibbs = (T / A_G) / (1 - ETA_G * A_G / 2), T / A_G
    assert abs(v / chain - 1) < 0.04, (v, chain)
    assert abs(v / gibbs - 1) > 0.2, (v, gibbs)


# ---------------------------------------------------------------------------
# The asymmetry of the independent product.
# ---------------------------------------------------------------------------


def _box(weights, kicks, q0, label):
    return particle(kicked_configuration(euclidean(1, eta=0.4), weights, kicks, init=[q0]), 1.0, label)


def _A():
    return _box([0.5, 0.5], [[-1.0], [1.0]], 0.2, "A")


def _B():
    return _box([0.9, 0.1], [[0.0], [3.0]], -0.7, "B")


def test_swap_keeps_the_law():
    K = 3
    ab, ba = exact_law(parallel_arrangements(_A(), _B()), K), exact_law(parallel_arrangements(_B(), _A()), K)
    for g in (lambda qa, qb: qa, lambda qa, qb: qb, lambda qa, qb: qa * qb ** 2 + 2 * qa ** 3):
        np.testing.assert_allclose(ab.expect(lambda s: g(s[0][0], s[1][0])),
                                   ba.expect(lambda s: g(s[1][0], s[0][0])), rtol=1e-12)


def test_one_shared_seed_decides_which_box_gets_which_draw():
    # The key sampler inverts one uniform draw through the cumulative weights. In
    # A (x) B they are (.45, .05, .45, .05) and A reads k // 2; in B (x) A they are
    # (.45, .45, .05, .05) and A reads k % 2; A's draw differs only when the uniform
    # falls in (.45, .5) or (.9, .95), probability .1 per step. So run long enough.
    K = 40
    (a_ab, _), _ = run(seed(parallel_arrangements(_A(), _B()), key_sampler(5)), K)
    (_, a_ba), _ = run(seed(parallel_arrangements(_B(), _A()), key_sampler(5)), K)
    assert not np.allclose(a_ab, a_ba)
    # Seeded box by box, the order does not matter: A reads its own seed.
    (a1, _), _ = run(parallel_arrangements(seed(_A(), key_sampler(1)), seed(_B(), key_sampler(2))), K)
    _, (a2, _) = run(parallel_arrangements(seed(_B(), key_sampler(2)), seed(_A(), key_sampler(1))), K)
    np.testing.assert_allclose(a1[0], a2[0], atol=1e-14)
