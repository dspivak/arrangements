"""Noisy integrators over durations (plan P6 (d) (1); STATUS.md §3.12).

A noisy integrator is a ``(y^D <| cot(Q) <| R)``-coalgebra for a noise polynomial
``R``: told a duration, it shows a deterministic look-ahead, and on a covector there
returns a random variable over next states. Four checks, predictions written first in
``notes/runs/p6d-aa61c0a2/RESULTS.md`` (book repo):

* P1 the unit: a noisy part beside the unit integrator steps to the same random
  variable at every duration;
* P2 two parts side by side, run through ``Phi`` at one shared duration: the seed is
  the concatenation, the covariance the block sum, both scaled by the same duration;
* P3 Euler--Maruyama on ``rand_N`` (variance in the position) has the law of the
  ``rand_G`` timed step (``sqrt(duration)`` in the return map) at a positive duration;
* P4 at duration zero the ``rand_G`` return map's duration-difference quotient grows
  like ``h^{-1/2}``, while ``rand_N``'s position and return map have the derivative
  ``(2 T sharp, sharp xi)``: only the second is smooth up to duration zero.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.coalgebra import CotCoalg
from arrangements.functors import Phi
from arrangements.rand import CovGaussianRV, euler_maruyama, euler_maruyama_cov
from arrangements.rvect import euclidean
from arrangements.wiring import parallel_arrangements

from tests.chain import IN_POS
from tests.test_rand import particle

jax.config.update("jax_enable_x64", True)

_TRIV = lambda _o: (jnp.zeros(0), jnp.zeros(0))


def step_at(arr, tau):
    """One closed-system step of ``Phi(arr)`` of duration ``tau``, from ``arr``'s initial state."""
    O = Phi(arr)
    told = lambda _o: (jnp.zeros(arr.out_dim_N), jnp.zeros(arr.in_dim_N))   # no force, no value from outside
    return O.with_state(arr.P.init).run_one(IN_POS, told, tau)[-1]


def moments(rv):
    """Mean and covariance of an affine return map on ``N(0, Sigma)``."""
    flat = lambda z: jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(rv.ret(z))])
    z0 = jnp.zeros(rv.n)
    J = jax.jacfwd(flat)(z0)
    return flat(z0), J @ rv.cov @ J.T


def unit_integrator():
    """The unit integrator ``(1, 1, !, !)``."""
    return CotCoalg(Q_dim=0, init=jnp.zeros(0), position=lambda s: jnp.zeros(0),
                    step=lambda s, xi: s, timed_step=lambda s, xi, tau: s, label="unit")


T1, ETA1, A1, Q1 = 0.7, 0.5, 1.0, (1.0,)
T2, ETA2, A2, Q2 = 0.2, 0.3, 2.0, (-0.5, 2.0)


def _em1():
    return particle(euler_maruyama_cov(euclidean(1, eta=ETA1), T1, init=list(Q1)), A1, "EM1")


def _em2():
    return particle(euler_maruyama_cov(euclidean(2, eta=ETA2), T2, init=list(Q2)), A2, "EM2")


def test_unit_changes_nothing():
    pair = parallel_arrangements(_em1(), particle(unit_integrator(), 0.0, "unit"))
    for tau in (0.0, 0.3, 1.0):
        alone, beside = step_at(_em1(), tau), step_at(pair, tau)
        assert isinstance(beside, CovGaussianRV) and beside.n == alone.n == 1
        np.testing.assert_allclose(beside.cov, alone.cov, atol=1e-15)
        z = jnp.array([0.37])
        np.testing.assert_allclose(beside.ret(z)[0], alone.ret(z), atol=1e-15)
        assert beside.ret(z)[1].shape == (0,)


def test_two_parts_share_one_duration_and_independent_noise():
    tau = 0.3
    pair = parallel_arrangements(_em1(), _em2())
    print(pair.label)
    rv = step_at(pair, tau)
    assert rv.n == 3
    mean, cov = moments(rv)
    q1, q2 = jnp.array(Q1), jnp.array(Q2)
    drift = jnp.concatenate([q1 - tau * ETA1 * A1 * q1, q2 - tau * ETA2 * A2 * q2])
    expected = jnp.diag(jnp.array([2 * T1 * tau * ETA1, 2 * T2 * tau * ETA2, 2 * T2 * tau * ETA2]))
    np.testing.assert_allclose(mean, drift, atol=1e-14)
    np.testing.assert_allclose(cov, expected, atol=1e-14)


def test_same_law_as_the_square_root_step():
    tau, d = 0.3, 2
    Q = euclidean(d, eta=ETA2)
    cov_rv = step_at(particle(euler_maruyama_cov(Q, T2, init=list(Q2)), A2), tau)
    g_rv = step_at(particle(euler_maruyama(Q, T2, init=list(Q2)), A2), tau)
    m_cov, c_cov = moments(cov_rv)
    J = jax.jacfwd(g_rv.ret)(jnp.zeros(g_rv.n))
    np.testing.assert_allclose(m_cov, g_rv.ret(jnp.zeros(g_rv.n)), atol=1e-14)
    np.testing.assert_allclose(c_cov, J @ J.T, atol=1e-14)


def test_only_the_variance_in_the_position_is_smooth_at_duration_zero():
    Q, z = euclidean(1, eta=ETA1), jnp.array([1.0])
    g_arr = particle(euler_maruyama(Q, T1, init=list(Q1)), A1)
    c_arr = particle(euler_maruyama_cov(Q, T1, init=list(Q1)), A1)
    g0, c0 = step_at(g_arr, 0.0), step_at(c_arr, 0.0)
    xi = -A1 * Q1[0]
    quotients_g, quotients_c = [], []
    for h in (1e-2, 1e-4, 1e-6):
        gh, ch = step_at(g_arr, h), step_at(c_arr, h)
        quotients_g.append(float((gh.ret(z) - g0.ret(z))[0] / h))
        quotients_c.append((float((ch.cov - c0.cov)[0, 0] / h), float((ch.ret(z) - c0.ret(z))[0] / h)))
    print("rand_G return-map quotients:", quotients_g)
    print("rand_N (position, return-map) quotients:", quotients_c)
    for a, b in zip(quotients_g, quotients_g[1:]):
        assert 9.0 < b / a < 11.0                    # grows like h^{-1/2}
    for pos, ret in quotients_c:
        np.testing.assert_allclose([pos, ret], [2 * T1 * ETA1, ETA1 * xi], rtol=1e-8)


# ---------------------------------------------------------------------------
# Symmetry and interchange (plan P6 (d) (2)): the order of the noise, and renaming.
# ---------------------------------------------------------------------------

from arrangements.arrangement import Arrangement, exact
from arrangements.coalgebra import configuration
from arrangements.rand import lift
from arrangements.wiring import compose_seq

TAU = 0.3


def em_part(T, eta, q0):
    return euler_maruyama_cov(euclidean(1, eta=eta), T, init=[q0])


def source(P, a, label):
    """A system showing its configuration, with cost ``(a/2) q^2``."""
    return Arrangement(P=P, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
                       out_f=lambda q, m: q, in_f=lambda q, m, n: jnp.zeros(0),
                       omega=exact(lambda q, m, n: -0.5 * a * jnp.sum(q ** 2)), label=label)


def reader(P, k, label):
    """An open arrangement reading one value ``m``, showing its configuration, with cost ``(k/2)(q - m)^2``."""
    return Arrangement(P=P, in_dim_M=0, out_dim_M=1, in_dim_N=0, out_dim_N=1,
                       out_f=lambda q, m: q, in_f=lambda q, m, n: jnp.zeros(0),
                       omega=exact(lambda q, m, n: -0.5 * k * jnp.sum((q - m) ** 2)), label=label)


def flat(tree):
    return jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(tree)])


def _four(y_noisy=True):
    x = source(em_part(0.7, 0.5, 1.0), 1.0, "x")
    yP = em_part(0.2, 0.3, -0.5) if y_noisy else lift(configuration(euclidean(1, eta=0.3), init=jnp.array([-0.5])), CovGaussianRV)
    y = source(yP, 2.0, "y")
    a = reader(em_part(1.1, 0.4, 0.3), 1.5, "a")
    b = reader(em_part(0.05, 0.9, 2.0), 0.5, "b")
    return x, y, a, b


def _sides(y_noisy=True):
    x, y, a, b = _four(y_noisy)
    lhs = compose_seq(parallel_arrangements(x, y), parallel_arrangements(a, b))     # (a (x) b) . (x (x) y)
    rhs = parallel_arrangements(compose_seq(x, a), compose_seq(y, b))               # (a . x) (x) (b . y)
    print(lhs.label, "|", rhs.label)
    return step_at(lhs, TAU), step_at(rhs, TAU)


def test_interchange_holds_only_up_to_renaming():
    L, R = _sides()
    assert L.n == R.n == 4
    perm = jnp.array([0, 2, 1, 3])                     # seeds (x, y, a, b) -> (x, a, y, b)
    print("lhs cov diag", jnp.diag(L.cov), "rhs cov diag", jnp.diag(R.cov))
    assert not np.allclose(L.cov, R.cov)               # presented differently
    np.testing.assert_allclose(R.cov, L.cov[perm][:, perm], atol=1e-14)
    z = jax.random.normal(jax.random.PRNGKey(0), (4,))
    (sx, sy), (sa, sb) = L.ret(z)
    (tx, ta), (ty, tb) = R.ret(z[perm])
    np.testing.assert_allclose(flat((sx, sa, sy, sb)), flat((tx, ta, ty, tb)), atol=1e-14)


def test_interchange_on_the_nose_when_a_middle_part_has_no_noise():
    L, R = _sides(y_noisy=False)
    assert L.n == R.n == 3
    np.testing.assert_allclose(L.cov, R.cov, atol=1e-14)
    z = jax.random.normal(jax.random.PRNGKey(1), (3,))
    (sx, sy), (sa, sb) = L.ret(z)
    (tx, ta), (ty, tb) = R.ret(z)
    np.testing.assert_allclose(flat((sx, sa, sy, sb)), flat((tx, ta, ty, tb)), atol=1e-14)


def test_symmetry_is_the_block_swap():
    A, B = _em1(), _em2()                              # seeds of dimension 1 and 2
    ab, ba = step_at(parallel_arrangements(A, B), TAU), step_at(parallel_arrangements(B, A), TAU)
    beta = jnp.array([2, 0, 1])                        # beta_{1,2}: (zA, zB1, zB2) read from (zB1, zB2, zA)
    assert not np.allclose(ab.cov, ba.cov)
    np.testing.assert_allclose(ab.cov, ba.cov[beta][:, beta], atol=1e-14)
    z = jax.random.normal(jax.random.PRNGKey(2), (3,))
    sa, sb = ab.ret(z)
    tb, ta = ba.ret(z[jnp.array([1, 2, 0])])
    np.testing.assert_allclose(flat((sa, sb)), flat((ta, tb)), atol=1e-14)


# ---------------------------------------------------------------------------
# Seed or category (plan P6 (d) (3)): a sampler's draw must be smooth in the demand.
# ---------------------------------------------------------------------------

from arrangements.rand import FiniteRV, key_sampler, replay_sampler, seed


def test_fair_finite_draw_jumps_in_the_demanded_probabilities():
    smp = key_sampler(4)
    ps = jnp.linspace(0.01, 0.99, 99)
    tickets = [int(smp.draw(smp.init, jnp.array([p, 1 - p]))) for p in ps]
    jumps = sum(a != b for a, b in zip(tickets, tickets[1:]))
    print("tickets along p:", "".join(map(str, tickets)))
    assert jumps == 1                                  # a smooth draw on Delta_2 would be constant


def test_seeded_square_root_step_is_not_smooth_at_duration_zero():
    Q, w = euclidean(1, eta=ETA1), jnp.array([[0.8], [0.0]])
    arr = seed(particle(euler_maruyama(Q, T1, init=list(Q1)), A1), replay_sampler(w))
    s0 = step_at(arr, 0.0)[0]
    quotients = [float((step_at(arr, h)[0] - s0)[0] / h) for h in (1e-2, 1e-4, 1e-6)]
    print("seeded rand_G quotients:", quotients)
    for a, b in zip(quotients, quotients[1:]):
        assert 9.0 < b / a < 11.0
    # at the unit duration the seeded part is the bound random variable on the replayed draw
    bound = step_at(particle(euler_maruyama(Q, T1, init=list(Q1)), A1), 1.0)
    np.testing.assert_allclose(step_at(arr, 1.0)[0], bound.ret(w[0]), atol=1e-14)


# ---------------------------------------------------------------------------
# Laws (plan P6 (d) (4)): taking laws is symmetric monoidal, renamings vanish.
# ---------------------------------------------------------------------------


def state_law(rv):
    """Mean and covariance of the next state, in the state's own coordinate order."""
    return moments(rv)


def test_law_of_a_tensor_is_the_product_and_symmetric_on_the_nose():
    A, B = _em1(), _em2()
    mA, cA = state_law(step_at(A, TAU))
    mB, cB = state_law(step_at(B, TAU))
    mAB, cAB = state_law(step_at(parallel_arrangements(A, B), TAU))
    mBA, cBA = state_law(step_at(parallel_arrangements(B, A), TAU))
    np.testing.assert_allclose(mAB, jnp.concatenate([mA, mB]), atol=1e-14)
    np.testing.assert_allclose(cAB, jax.scipy.linalg.block_diag(cA, cB), atol=1e-14)
    swap = jnp.array([2, 0, 1])                        # states (B1, B2, A) -> (A, B1, B2)
    np.testing.assert_allclose(mAB, mBA[swap], atol=1e-14)
    np.testing.assert_allclose(cAB, cBA[swap][:, swap], atol=1e-14)


def test_interchange_holds_at_laws_by_the_state_map_alone():
    L, R = _sides()
    mL, cL = state_law(L)                              # states (x, y, a, b)
    mR, cR = state_law(R)                              # states (x, a, y, b)
    g = jnp.array([0, 2, 1, 3])
    np.testing.assert_allclose(mL, mR[g], atol=1e-14)
    np.testing.assert_allclose(cL, cR[g][:, g], atol=1e-14)


def test_stationary_variance_of_the_kernel_read_off_phi():
    T, eta, a = 0.7, 0.5, 1.0
    for tau in (1.0, 0.3, 0.01):
        def kernel(q0):
            arr = particle(euler_maruyama_cov(euclidean(1, eta=eta), T, init=[q0]), a)
            return state_law(step_at(arr, tau))
        (m0, v0), (m1, _) = kernel(0.0), kernel(1.0)
        c, v = float(m1[0] - m0[0]), float(v0[0, 0])
        np.testing.assert_allclose([c, v], [1 - tau * eta * a, 2 * T * tau * eta], rtol=1e-12)
        V = v / (1 - c ** 2)
        print(f"tau={tau}: stationary variance {V:.6f}, closed form {(T / a) / (1 - tau * eta * a / 2):.6f}, Gibbs {T / a}")
        np.testing.assert_allclose(V, (T / a) / (1 - tau * eta * a / 2), rtol=1e-10)
