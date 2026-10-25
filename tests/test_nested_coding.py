"""Predictive coding nested in the attention operad with signed costs (book plan P5 (i)).

Predictions, fixed before ``arrangements/nested_coding.py`` existed, and verdicts: the book's
``notes/runs/p5i-1e43f5fb/RESULTS.md``. Each docstring says which of three things its test does:
reproduce what the paper proves (``prop:operad``, here with negative costs), check a closed form,
or merely illustrate. Every run is ``Phi`` of a closed arrangement: the composite closed by
``data_box`` (and, for ``belief_first``, its slot filled by ``feeder``).
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.active_inference import run
from arrangements.attention import attention_arrangement, substitute
from arrangements.nested_coding import (
    data_box, dims, feeder, free_energy, laplace_free_energy, laplace_learner, level1,
    level1_publishing, level2, level2_reading, level3, _zeros,
)
from arrangements.wiring import compose_seq

from tests.chain import flat_state
from tests.test_attention_operad import _agrees_with_composition

jax.config.update("jax_enable_x64", True)

J, K = 4, 8
DH, DP = dims(J, K)


def data(centers, seed):
    r = np.random.default_rng(seed)
    return jnp.asarray(np.asarray(centers, float)[:, None] + r.standard_normal((J, K)))


SEPARATED = data([-3, -1, 1, 3], 0)
CLOSE = data([0, 0, 0, 0], 1)


def spreads(x):
    """``(W, B)``: mean within-group squared deviation, and mean squared deviation of group means."""
    x = np.asarray(x)
    xb = x.mean(1)
    return ((x - xb[:, None]) ** 2).mean(), ((xb - xb.mean()) ** 2).mean()


def stationary(x):
    """The interior stationary points of ``F``, from ``(W+B) r^2 + (2W - KB) r + W = 0``,
    ``r = Pi2 / (Pi1 K)``: a list of ``(Pi1, Pi2, c)``, ``c = 1/(1+r)`` the shrinkage."""
    W, B = spreads(x)
    a, b, c = W + B, 2 * W - K * B, W
    disc = b * b - 4 * a * c
    if disc < 0 or b >= 0:
        return []
    out = []
    for r in sorted([(-b - np.sqrt(disc)) / (2 * a), (-b + np.sqrt(disc)) / (2 * a)]):
        cc = 1 / (1 + r)
        out.append((1 / (W + (1 - cc) ** 2 * B), 1 / (cc ** 2 * B), cc))
    return out


def data_first(eta, x, init1=None, init2=None):
    """``L1(L2)`` closed by the data: state ``((nu, l2), (mu, l1))`` flattened in that order."""
    b = substitute(level1(J, K, eta, init1), [level2(J, K, eta, 0, init2)])
    return b, compose_seq(b.arr(), data_box(x, DP))


def unpack_df(s):
    """``(mu, l1, nu, l2)`` from a flat ``data_first`` state."""
    return s[2:2 + J], s[2 + J], s[0], s[1]


def random_q(n, seed, scale=0.5):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(n) * scale)


# ---------------------------------------------------------------------------
# Piece A: the operad with signed costs.
# ---------------------------------------------------------------------------


def test_composite_cost_is_the_free_energy_and_goes_negative():
    """REPRODUCES eq:composite-cost with signed costs: ``validate`` accepts the levels (nothing
    guards the sign), and at the data the composite cost is ``F`` written by hand, negative at
    some configurations."""
    b, _ = data_first(0.01, SEPARATED)
    e = jnp.concatenate([jnp.ravel(SEPARATED), jnp.ones(1), jnp.zeros(DP)])
    seen_negative = False
    for seed in range(6):
        q = random_q(b.P.Q_dim, seed, 1.5)
        nu, l2, mu, l1 = q[0], q[1], q[2:2 + J], q[2 + J]
        k = b.cost(q, jnp.zeros(0), e)
        np.testing.assert_allclose(k, free_energy(SEPARATED, mu, l1, nu, l2), atol=1e-12, rtol=1e-13)
        seen_negative |= bool(k < 0)
    assert not seen_negative   # at these scales the data term dominates
    # down the funnel: every belief at the grand mean, the level-2 precision large
    xbar = float(jnp.mean(SEPARATED))
    q = jnp.concatenate([jnp.array([xbar, 100.0]), jnp.full(J, xbar), jnp.zeros(1)])
    k = b.cost(q, jnp.zeros(0), e)
    np.testing.assert_allclose(k, free_energy(SEPARATED, q[2:2 + J], 0.0, xbar, 100.0), rtol=1e-13)
    assert k < -100


def test_substitution_agrees_with_composition():
    """REPRODUCES prop:operad for the signed levels: ``substitute(L1, [L2])`` equals
    ``compose_seq(L2.arr(), L1.arr())`` in sending, receiving, all three forcing components and
    ``Phi`` trajectories (the helper of ``test_attention_operad``)."""
    _agrees_with_composition(level1(J, K, 0.01), [level2(J, K, 0.01)], seed=7)
    _agrees_with_composition(level2(J, K, 0.01, arity=1), [level3(J, K, 2.0, 0.3)], seed=8)


def test_three_levels_are_associative():
    """REPRODUCES prop:operad's associativity with signed costs: ``(L1(L2))(L3)`` and
    ``L1(L2(L3))`` have the same configuration order and are the same operation: placement,
    sending, receiving, cost and the whole forcing."""
    L1, L2, L3 = level1(J, K, 0.01), level2(J, K, 0.01, arity=1), level3(J, K, 2.0, 0.3)
    left = substitute(substitute(L1, [L2]), [L3])
    right = substitute(L1, [substitute(L2, [L3])])
    assert left.n == right.n == 0 and left.P.Q_dim == right.P.Q_dim == J + 3
    for seed in range(4):
        q = random_q(left.P.Q_dim, 20 + seed, 1.0)
        e = random_q(left.input_dim, 30 + seed, 1.0)
        m = jnp.zeros(0)
        np.testing.assert_allclose(left.placement(q), right.placement(q), atol=1e-12)
        np.testing.assert_allclose(left.sigma(q, m), right.sigma(q, m), atol=1e-12)
        np.testing.assert_allclose(left.rho(q, m, e), right.rho(q, m, e), atol=1e-12)
        np.testing.assert_allclose(left.cost(q, m, e), right.cost(q, m, e), atol=1e-11)
        for u, v in zip(left.arr().omega(q, m, e), right.arr().omega(q, m, e)):
            np.testing.assert_allclose(u, v, atol=1e-11)
        nu, l2, mu, l1 = q[0], q[1], q[2:2 + J], q[2 + J]
        # the three-level free energy, by hand; positions are relative, so each level reads the
        # exterior position p0 plus what the levels above it placed
        x, p0 = jnp.reshape(e[:DH], (J, K)), e[DH + 1:]
        by_hand = (0.5 * jnp.exp(l1) * jnp.sum((x - mu[:, None]) ** 2) - 0.5 * J * K * l1
                   + 0.5 * jnp.exp(l2) * jnp.sum((p0[:J] + mu - nu) ** 2) - 0.5 * J * l2
                   + 0.5 * 2.0 * (p0[J] + nu - 0.3) ** 2)
        np.testing.assert_allclose(left.cost(q, m, e), by_hand, atol=1e-11)


def test_both_orientations_give_one_free_energy_and_one_run():
    """CHECKS the prediction that nesting either way gives one free energy and one run: L1 outer
    (state placed down) and L2 outer (state published up through a configuration-dependent
    transfer, its slot fed by ``feeder``) agree on ``F`` and on 300 steps of ``Phi``."""
    eta = 0.005
    _, A = data_first(eta, SEPARATED, init1=random_q(J + 1, 1), init2=random_q(2, 2))
    bB = substitute(level2_reading(J, K, eta, random_q(2, 2)),
                    [level1_publishing(J, K, eta, random_q(J + 1, 1))])
    B = compose_seq(compose_seq(feeder(J, K), bB.arr()), data_box(SEPARATED, DP))
    # B's state: feeder (empty), (mu, l1), (nu, l2), data (empty)
    m_fed = jnp.zeros(DH + 1).at[0].set(1.0).at[DH].set(1.0)
    e = jnp.concatenate([jnp.ravel(SEPARATED), jnp.ones(1), jnp.zeros(DP)])
    q = random_q(J + 3, 5, 1.0)
    np.testing.assert_allclose(bB.cost(q, m_fed, e),
                               free_energy(SEPARATED, q[:J], q[J], q[J + 1], q[J + 2]), atol=1e-11)
    _, ra = run(A, 300, record=flat_state)
    _, rb = run(B, 300, record=flat_state)
    for sa, sb in zip(ra, rb):
        np.testing.assert_allclose(jnp.concatenate([sa[2:], sa[:2]]), sb, atol=1e-10)
    assert float(jnp.max(jnp.abs(ra[-1] - ra[0]))) > 1e-2


# ---------------------------------------------------------------------------
# Piece B: where the precisions settle, and the funnel.
# ---------------------------------------------------------------------------


def settle(arr, steps, every=None):
    """Run ``Phi(arr)``; the flat final state, and with ``every`` the flat states on the way."""
    s, rec = run(arr, steps, record=flat_state)
    return rec[-1], (rec[::every] if every else None)


def test_one_level_settles_at_the_inverse_within_variance():
    """CHECKS A CLOSED FORM (prediction 4): L1 with an empty child of zero cost settles at
    ``mu_j = xbar_j`` and ``e^l1 = 1/W``."""
    empty = attention_arrangement(0, DH, DP, placement=lambda q: jnp.zeros((0, DP)),
                                  transfers=_zeros(0, DH), label="empty")
    b = substitute(level1(J, K, 0.005), [empty])
    f, _ = settle(compose_seq(b.arr(), data_box(SEPARATED, DP)), 6000)
    W, _ = spreads(SEPARATED)
    np.testing.assert_allclose(f[:J], SEPARATED.mean(1), atol=1e-9)
    np.testing.assert_allclose(jnp.exp(f[J]), 1 / W, rtol=1e-9)


def test_separated_data_settle_at_the_interior_minimum_from_both_starts():
    """CHECKS A CLOSED FORM (prediction 5, held; prediction 6, refuted): from rest, and from a
    level-2 precision of 33, above the saddle's 13.9, the run settles at the lower root ``r_-``:
    ``nu = xbar``, ``mu_j - nu = c (xbar_j - xbar)``, precisions to ``1e-9``."""
    P1, P2, c = stationary(SEPARATED)[0]
    xb = SEPARATED.mean(1)
    for init2 in (None, jnp.array([0.0, 3.5])):
        f, _ = settle(data_first(0.005, SEPARATED, init2=init2)[1], 20000)
        mu, l1, nu, l2 = unpack_df(f)
        np.testing.assert_allclose([jnp.exp(l1), jnp.exp(l2)], [P1, P2], rtol=1e-9)
        np.testing.assert_allclose(mu - nu, c * (xb - xb.mean()), atol=1e-9)
        np.testing.assert_allclose(nu, xb.mean(), atol=1e-9)


def test_the_upper_root_is_a_saddle_whose_branches_both_end_at_the_minimum():
    """CHECKS A CLOSED FORM, then ILLUSTRATES: ``F``'s Hessian at ``r_+`` has one negative
    eigenvalue; a step of 0.05 either way along its eigenvector, run, ends at ``r_-``."""
    P1, P2, c = stationary(SEPARATED)[1]
    lo = stationary(SEPARATED)[0]
    xb, g = SEPARATED.mean(1), float(SEPARATED.mean())
    q = jnp.concatenate([g + c * (xb - g), jnp.array([np.log(P1), g, np.log(P2)])])
    F = lambda q: free_energy(SEPARATED, q[:J], q[J], q[J + 1], q[J + 2])
    assert float(jnp.max(jnp.abs(jax.grad(F)(q)))) < 1e-12
    ev, V = np.linalg.eigh(np.asarray(jax.hessian(F)(q)))
    assert ev[0] < 0 < ev[1]
    for s in (-0.05, 0.05):
        q1 = q + s * V[:, 0]
        f, _ = settle(data_first(0.005, SEPARATED, q1[:J + 1], q1[J + 1:])[1], 20000)
        np.testing.assert_allclose(jnp.exp(unpack_df(f)[3]), lo[1], rtol=1e-9)


def test_a_start_down_the_funnel_stays_there_at_a_small_step():
    """CHECKS prediction 6b (held): at ``eta = 0.0005``, from every belief at the grand mean, ``Pi1 =
    1/(W+B)`` and ``Pi2 = 40``, the level-2 precision climbs to the step's stability edge,
    ``eta (J+1) Pi2 ~ 2``, and stays there; ``F`` stays above its value at ``r_-``."""
    W, B = spreads(SEPARATED)
    g = float(SEPARATED.mean())
    _, A = data_first(0.0005, SEPARATED, jnp.concatenate([jnp.full(J, g), jnp.array([np.log(1 / (W + B))])]),
                      jnp.array([g, np.log(40.0)]))
    f, path = settle(A, 40000, every=4000)
    l2s = [float(unpack_df(s)[3]) for s in path[1:]]
    edge = np.log(2 / (0.0005 * (J + 1)))
    assert min(l2s) > edge - 0.5 and max(l2s) < edge + 0.3
    assert free_energy(SEPARATED, *unpack_df(f)) > 20


def test_close_data_level_two_precision_never_settles():
    """CHECKS prediction 7 (partly held): with no interior stationary point, the beliefs gather at
    the grand mean, while ``l2`` wanders below the step's stability edge and does not settle; the edge
    rises as the step shrinks. ``Pi1`` swings with ``l2``'s sawtooth, jumping after each crash and falling
    back as the beliefs regather (1.21-1.30 at eta = 0.005 against ``1/(W+B)`` = 1.214), so it is read over
    the path, never at one instant: the swing stays above ``1/(W+B)`` less 1% and narrows as the step
    shrinks, and at eta = 0.00125 every reading is within 1% of ``1/(W+B)`` (book P5 (l) stage B,
    fcf80b7b: the instant at t = 200 fell after a crash and read 1.281)."""
    W, B = spreads(CLOSE)
    assert stationary(CLOSE) == []
    tops, swings = [], []
    for eta in (0.005, 0.00125):
        f, path = settle(data_first(eta, CLOSE)[1], int(200 / eta), every=int(2 / eta))
        l2s = np.array([float(unpack_df(s)[3]) for s in path[20::10]])   # t = 40, 60, ..., 200
        pi1 = np.array([float(jnp.exp(unpack_df(s)[1])) for s in path[10:]])  # t = 20, 22, ..., 200
        mu, l1, nu, l2 = unpack_df(f)
        assert float(jnp.std(mu - nu)) < 0.15
        assert pi1.min() > 0.99 / (W + B)
        assert l2s.max() < np.log(2 / (eta * (J + 1))) + 0.7
        assert l2s.max() - l2s.min() > 0.4          # it does not settle
        tops.append(l2s.max())
        swings.append(pi1.max() - pi1.min())
    np.testing.assert_allclose(pi1, 1 / (W + B), rtol=0.01)   # eta = 0.00125, every reading
    assert swings[1] < 0.25 * swings[0]
    assert tops[1] > tops[0] + 0.5


# ---------------------------------------------------------------------------
# Piece C: the curvature term, one closed box outside the operad.
# ---------------------------------------------------------------------------


def neg_log_marginal(theta, x):
    """``-log int prod_j N(x_j; nu 1, s2 I + t2 1 1^T) d nu``, flat in ``nu``, constants dropped;
    ``theta = (log s2, log t2)``. An ordinary calculation, the independent check."""
    s2, t2 = np.exp(theta)
    x = np.asarray(x)
    C = s2 * np.eye(K) + t2 * np.ones((K, K))
    Ci, one = np.linalg.inv(C), np.ones(K)
    a = J * one @ Ci @ one
    b = sum(one @ Ci @ xj for xj in x)
    c = sum(xj @ Ci @ xj for xj in x)
    return 0.5 * J * np.linalg.slogdet(C)[1] + 0.5 * np.log(a) + 0.5 * (c - b * b / a)


def test_curvature_term_settles_at_type_two_maximum_likelihood_from_both_starts():
    """CHECKS prediction 8 (held): ``F + 1/2 log det H`` run as one closed box settles at the
    precisions that maximize the marginal likelihood, from rest and from the funnel start that the
    plain ``F`` keeps (``Pi2 = 40``, beliefs at the grand mean)."""
    from scipy.optimize import minimize

    ml = min((minimize(neg_log_marginal, t0, args=(SEPARATED,), method="Nelder-Mead",
                       options=dict(xatol=1e-12, fatol=1e-14, maxiter=20000))
              for t0 in ([0, 0], [0, 3], [-1, -3])), key=lambda r: r.fun)
    want = 1 / np.exp(ml.x)
    W, B = spreads(SEPARATED)
    g = float(SEPARATED.mean())
    for q0 in (None, jnp.concatenate([jnp.full(J, g), jnp.array([np.log(1 / (W + B)), g, np.log(40.0)])])):
        f, _ = settle(laplace_learner(SEPARATED, 0.005, q0), 20000)
        np.testing.assert_allclose([jnp.exp(f[J]), jnp.exp(f[J + 2])], want, rtol=1e-6)
    # the plain F's minimum overstates both precisions
    P1, P2, _ = stationary(SEPARATED)[0]
    assert P1 > 1.1 * want[0] and P2 > 1.3 * want[1]


# ---------------------------------------------------------------------------
# Piece D: the precision sent forward, three boxes in arr.
# ---------------------------------------------------------------------------


def _evidence_F(q, x, integrate_nu):
    """The per-level costs summed by hand at ``q = (l1, nu, l2)``."""
    from arrangements.nested_coding import evidence_cost

    X = jnp.asarray(x)
    ss = jnp.sum((X - X.mean(1)[:, None]) ** 2)
    w = K * jnp.exp(q[0])
    msg = jnp.concatenate([w * X.mean(1), jnp.full(J, w)])
    return 0.5 * jnp.exp(q[0]) * ss - 0.5 * J * (K - 1) * q[0] + evidence_cost(q[1:], msg, J, integrate_nu)


def test_forward_evidence_sums_to_the_marginal_likelihood():
    """CHECKS A CLOSED FORM (piece D, prediction 1): the closed composite's forcing is ``-d`` of the per-level sum,
    and that sum, at ``nu = xbar``, is minus the log marginal likelihood up to one constant."""
    from arrangements.nested_coding import evidence_hierarchy

    A = evidence_hierarchy(SEPARATED, 0.005)
    q, z = jnp.array([0.4, 0.1, -0.7]), jnp.zeros(0)
    np.testing.assert_allclose(A.omega(q, z, z)[0], -jax.grad(_evidence_F)(q, SEPARATED, True), atol=1e-12)
    for x in (SEPARATED, CLOSE):
        d = [float(_evidence_F(jnp.array([l1, float(x.mean()), l2]), x, True))
             - neg_log_marginal(np.array([-l1, -l2]), x)
             for l1, l2 in ((0.0, 0.0), (0.7, -1.8), (-1.6, 3.7), (-2.3, -6.9), (1.1, 9.2))]
        np.testing.assert_allclose(d, -4.158883083359669, atol=1e-10)


def test_forward_evidence_settles_at_maximum_likelihood_from_both_starts():
    """CHECKS prediction 2 (held): from rest and from the funnel start, the separated data settle at the
    marginal-likelihood precisions (``nu`` integrated), equal to ``laplace_learner``'s; with ``nu`` a state only,
    the same ``Pi1`` and a finite ``Pi2`` of 0.209103 (prediction 3)."""
    from arrangements.nested_coding import evidence_hierarchy

    g = float(SEPARATED.mean())
    for integ, want in ((True, (2.145187, 0.156351)), (False, (2.145187, 0.209103))):
        for i1, i2 in ((0.0, None), (-1.66, jnp.array([g, np.log(40.0)]))):
            f, _ = settle(evidence_hierarchy(SEPARATED, 0.005, integ, i1, i2), 10000)
            np.testing.assert_allclose([jnp.exp(f[0]), jnp.exp(f[2])], want, rtol=3e-6)


def test_forward_evidence_close_data_send_the_upper_precision_up_slowly_and_bounded():
    """CHECKS prediction 4 (held): on the close data ``Pi1`` nears the marginal likelihood's 1.1757 while ``l2``
    climbs ever more slowly and the summed cost falls ever less, bounded below."""
    from arrangements.nested_coding import evidence_hierarchy

    _, path = settle(evidence_hierarchy(CLOSE, 0.005), 40000, every=10000)
    l2 = np.array([float(s[2]) for s in path])
    Fs = np.array([float(_evidence_F(s, CLOSE, True)) for s in path])
    assert np.all(np.diff(l2) > 0) and np.all(np.diff(np.diff(l2[1:])) < 0)
    assert np.all(np.diff(Fs) < 0) and Fs[-2] - Fs[-1] < 0.01
    np.testing.assert_allclose(np.exp(float(path[-1][0])), 1.1757, rtol=2e-3)
