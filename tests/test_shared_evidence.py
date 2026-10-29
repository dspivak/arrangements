"""Shared evidence: messages read as Gaussian evidence, pooling, learning whom to trust, the echo
(book Ch 4, plan P5 (m); predictions in the book repo's ``notes/runs/p5m-d2348c43/RESULTS.md``,
written before this file).

The reading under test (P5 (m) claim 1): a positive-weight message ``(h, w)`` on ``V = R^r`` is the
likelihood ``exp(<h, x> - w |x|^2 / 2)``, the bell curve ``N(h/w, I/w)`` up to a constant, so weight is
precision and value is the mean. Part 1 checks the reading on numbers. Parts 2-5 run attention
arrangements of ``attention.py`` composed in arr with frozen scouts (the idiom of
tests/test_fire.py): the scouts' reports are their frozen configurations, and the truth, where a
cost needs it, arrives from outside as the external message.

Transfers: the library's ``Transfer(gamma, L)`` is ``(h, w) |-> (L h, gamma w)``; the book's
``gamma C(id)`` is ``Transfer(gamma, gamma I)``.

Batch form: ``K`` trials sit in the coordinates of ``V = R^K``, each scout sending its ``K``
reports with weight 1. The cost is the mean over trials of the per-trial cost, so gradient descent
here is batch gradient descent on Ch 4's per-trial cost (the shared configuration is the
stipulation, as in tests/test_fire.py part 4).

Not the book's run (2026-10-29, Claude Opus 5.5, b0471e87): David called the batch form "seems like a total
violation of the model. how could this be at all reasonable? the whole thing is a time-based physics. now you're
cheating that? untenable.", since it runs an operation on the color ``R^K``, with one weight for all
trials and no clock, not Ch 4's ``phi : R, R, R -> R``. Ch 4 ``ex.learning_trust`` now reports
``tests/test_trust_in_time.py``, where a day lasts duration 1 and the reports arrive one day at a time.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, zero_form
from arrangements.attention import (
    attention_arrangement, identity_transfer, input_dim, make_input, message, msg_dim,
    signal, transfer, weight, zero_message, zero_transfer,
)
from arrangements.coalgebra import configuration
from arrangements.rvect import euclidean
from arrangements.wiring import compose_seq, tensor_arrangements

from tests.chain import flat_state

EMPTY = jnp.zeros(0)
SIGMAS = np.array([0.5, 1.0, 2.0])     # the sources' noise; precisions 4, 1, 1/4
K = 400                                # training trials
TAU = 1.0                              # loudness exp(-p^2 / 2 tau)


def data(K=K, sigmas=SIGMAS, seed=0, copy=None):
    """Truths ``z ~ N(0, 1)`` and reports ``y_i = z + sigma_i eps_i``; ``copy=(i, j)`` makes source
    ``j`` repeat source ``i``'s reports exactly."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(K)
    Y = z[None, :] + sigmas[:, None] * rng.standard_normal((len(sigmas), K))
    if copy is not None:
        Y[copy[1]] = Y[copy[0]]
    return z, Y


def loglik(m, x):
    """``<h, x> - w |x|^2 / 2`` on a grid of scalars ``x``."""
    h, w = m
    return h * x - 0.5 * w * x ** 2


def gauss_logpdf(x, mean, var):
    return -0.5 * (x - mean) ** 2 / var - 0.5 * np.log(2 * np.pi * var)


def same_up_to_constant(f, g):
    d = f - g
    return np.max(np.abs(d - d.mean())) < 1e-9


# ---------------------------------------------------------------------------
# Part 1: the reading, on numbers.
# ---------------------------------------------------------------------------


def test_1_messages_are_gaussian_evidence():
    """Prediction (1): sums, scaling and the zero message read exactly; a transfer
    ``gamma C(g)``, ``g(v) = a v + b``, is pushforward along ``g`` then tempering by ``gamma a^2``
    whenever ``a != 0``; a constant ``g`` has no such reading; in ``R^2`` a non-conformal linear part
    has none either."""
    x = np.linspace(-6, 6, 241)
    m1, m2 = (3 * 4.0, 3.0), (8.0, 1.0)            # 3 msg(4), msg(8) (exc.message_weight)
    s = (m1[0] + m2[0], m1[1] + m2[1])
    # sum = product of likelihoods; the product is the bell curve with mean = value, precision = weight
    assert same_up_to_constant(loglik(s, x), loglik(m1, x) + loglik(m2, x))
    assert same_up_to_constant(loglik(s, x), gauss_logpdf(x, s[0] / s[1], 1 / s[1]))
    # scaling = tempering
    g = 0.3
    assert same_up_to_constant(loglik((g * s[0], g * s[1]), x), g * loglik(s, x))
    # zero message = flat
    assert np.all(loglik((0.0, 0.0), x) == 0)
    # transfers gamma C(g), g(v) = a v + b, on N(mu, 1/w): book formula (gamma(a h + b w), gamma w)
    mu, w, gam = 1.5, 2.0, 0.7
    for a, b in [(1.0, 0.5), (-1.0, 2.0), (2.0, -1.0), (0.25, 0.0)]:
        T = (gam * (a * mu * w + b * w), gam * w)
        push_mean, push_var = a * mu + b, a ** 2 / w          # pushforward of N(mu, 1/w)
        tempered = gam * a ** 2 * gauss_logpdf(x, push_mean, push_var)
        assert same_up_to_constant(loglik(T, x), tempered), (a, b)
    # a = 0: the transfer says "value b, precision gamma w"; the pushforward is a point mass at b
    T0 = (gam * 0.5 * w, gam * w)
    assert np.isclose(T0[0] / T0[1], 0.5) and np.isclose(T0[1], gam * w)
    # R^2, A = diag(1, 2): pushforward of N(mu, I/w) has covariance A A^T / w = diag(1, 4)/w,
    # not a multiple of I, so no message (h, w') in R^2 x R (always isotropic) equals it
    A = np.diag([1.0, 2.0])
    cov = A @ A.T / w
    assert not np.allclose(cov, cov[0, 0] * np.eye(2))
    # A = 2U, U a rotation: conformal, cov = 4 I / w, isotropic again
    th = 0.4
    U = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    cov = (2 * U) @ (2 * U).T / w
    assert np.allclose(cov, cov[0, 0] * np.eye(2))


# ---------------------------------------------------------------------------
# Scouts and hubs.
# ---------------------------------------------------------------------------


def scout(reports, label, dim_P=1):
    """A component ``I -> B`` with frozen configuration ``reports`` (a vector over trials); it sends
    ``(reports, 1)``, one weight-1 report per trial, and has zero forcing."""
    dim_H = len(reports)
    return Arrangement(
        P=configuration(euclidean(dim_H, 0.0), init=jnp.asarray(reports, float)),
        in_dim_M=0, out_dim_M=0,
        in_dim_N=input_dim(dim_H, dim_P), out_dim_N=msg_dim(dim_H),
        out_f=lambda q, m_out: message(q, 1.0),
        in_f=lambda q, m_out, n_in: EMPTY,
        omega=zero_form,
        label=label,
    )


def hub(n, dim_H, loudness, placement, cost, mu, init):
    """An attention arrangement ``V, ..., V -> V`` whose external box hears box ``i`` through
    ``loudness(q)[i] C(id)`` and nothing else is heard."""
    Z = zero_transfer(dim_H)

    def T(q):
        g = loudness(q)
        row0 = [Z] + [transfer(g[i], g[i] * jnp.eye(dim_H)) for i in range(n)]
        return tuple([tuple(row0)] + [tuple(Z for _ in range(n + 1)) for _ in range(n)])

    return attention_arrangement(
        n, dim_H, 1, placement=placement, transfers=T, cost=cost,
        parameter=configuration(euclidean(len(init), mu), init=jnp.asarray(init, float)),
        label="hub")


def run(system, e, steps):
    P, omega = system.P, system.omega

    @jax.jit
    def step(s, _):
        xi_q, _, _ = omega(P.position(s), EMPTY, e)
        return P.step(s, xi_q), None

    s, _ = jax.lax.scan(step, P.init, None, length=steps)
    return np.asarray(flat_state(s))


# ---------------------------------------------------------------------------
# Part 2: pooling.
# ---------------------------------------------------------------------------


def test_2_pooling():
    """Prediction (2): ``n`` sources of precision 1 heard at full loudness: the readout is
    ``(sum y_i, n)``, value the mean, weight ``n``; over trials the value's error variance is
    ``1/n``."""
    n, Kmc = 5, 20000
    z, Y = data(Kmc, np.ones(n), seed=1)
    a = hub(n, Kmc, lambda q: jnp.ones(n), lambda q: jnp.zeros((n, 1)),
            lambda q, m, e: jnp.asarray(0.0), 0.0, [0.0])
    m = jnp.concatenate([message(Y[i], 1.0) for i in range(n)])
    out = a.sigma(jnp.zeros(1), m)
    assert np.isclose(float(weight(out)), n)
    val = np.asarray(signal(out)) / float(weight(out))
    assert np.allclose(val, Y.mean(0))
    err_var = np.var(val - z)
    print("pooling: weight", float(weight(out)), "error variance", err_var, "1/n", 1 / n)
    assert abs(err_var - 1 / n) < 0.01


# ---------------------------------------------------------------------------
# Part 3: the echo (exc.self_through_group).
# ---------------------------------------------------------------------------


def test_3_echo_counts_a_report_twice():
    """Prediction (4): in the nest of exc.self_through_group with ``T^phi_11 = id``, the part in
    ``B_11`` receives ``m_0 + m_11 + 2 m_12 + m_21``: weight 5 from 4 independent reports, value
    error variance ``7/25`` against the ``1/5`` its weight claims and the ``1/4`` of pooling once."""
    Z, I = zero_transfer(1), identity_transfer(1)
    psi1 = attention_arrangement(2, 1, 1, lambda q: jnp.zeros((2, 1)),
                                 lambda q: ((Z, I, I), (I, Z, I), (I, I, Z)), check=True)
    psi2 = attention_arrangement(1, 1, 1, lambda q: jnp.zeros((1, 1)),
                                 lambda q: ((Z, I), (I, Z)), check=True)
    phi = attention_arrangement(2, 1, 1, lambda q: jnp.zeros((2, 1)),
                                lambda q: ((Z, I, I), (I, I, I), (I, I, Z)), check=False)
    comp = compose_seq(tensor_arrangements([psi1.arr(), psi2.arr()]), phi.arr())
    rng = np.random.default_rng(2)
    Kmc = 20000
    z = rng.standard_normal(Kmc)
    ys = z[None, :] + rng.standard_normal((4, Kmc))      # m_0, m_11, m_12, m_21
    q = jnp.zeros(comp.P.position(comp.P.init).shape)
    f = jax.jit(lambda m3, e: comp.in_f(q, m3, e)[:2])   # what B_11 receives, as a message
    recv = np.array([np.asarray(f(jnp.array([ys[1, k], 1, ys[2, k], 1, ys[3, k], 1.0]),
                                  jnp.array([ys[0, k], 1.0, 0.0]))) for k in range(2000)])
    assert np.allclose(recv[:, 1], 5.0)
    assert np.allclose(recv[:, 0], ys[0, :2000] + ys[1, :2000] + 2 * ys[2, :2000] + ys[3, :2000])
    val = (ys[0] + ys[1] + 2 * ys[2] + ys[3]) / 5
    ev = np.var(val - z)
    print("echo: weight 5, error variance", ev, "claimed 0.2, pooled-once 0.25, predicted 0.28")
    assert abs(ev - 0.28) < 0.012


# ---------------------------------------------------------------------------
# Part 4: learning whom to trust.
# ---------------------------------------------------------------------------


def trust_system(cost_kind, Y, mu, init):
    n = len(Y)
    if cost_kind == "prediction":
        # Ch 4's eqn.private_prediction, per trial ½|(z,1,p_0) - (sum_i g_i (y_i, 1), 0)|^2,
        # averaged over trials; loudness Gaussian in the placement (ex.gaussian_attention)
        loud = lambda q: jnp.exp(-q ** 2 / (2 * TAU))
        place = lambda q: q.reshape(n, 1)

        def cost(q, m_out, e):
            ms = m_out.reshape(n, K + 1)
            pred = (loud(q)[:, None] * ms).sum(0)
            d = e[:K + 1] - pred
            return 0.5 * jnp.mean(d[:K] ** 2) + 0.5 * d[K] ** 2 + 0.5 * e[K + 1] ** 2
    else:
        # per-source negative log likelihood of the truth: sum_i ½ e^{l_i} (h_0 - h_i)^2 - ½ l_i
        loud = lambda q: jnp.exp(q)
        place = lambda q: jnp.zeros((n, 1))

        def cost(q, m_out, e):
            ms = m_out.reshape(n, K + 1)
            sq = jnp.mean((e[None, :K] - ms[:, :K]) ** 2, axis=1)
            return jnp.sum(0.5 * jnp.exp(q) * sq - 0.5 * q)
    a = hub(n, K, loud, place, cost, mu, init)
    kids = [scout(Y[i], f"s{i}") for i in range(n)]
    return a, loud, compose_seq(tensor_arrangements(kids), a.arr())


def held_out(g, sigmas, seed=3, Kmc=200000):
    z, Y = data(Kmc, sigmas, seed)
    val = (g[:, None] * Y).sum(0) / g.sum()
    return np.var(val - z)


def test_4a_private_prediction_learns_relative_trust():
    """Prediction (3), Ch 4's cost: the settled loudnesses solve the least squares
    ``(Y Y^T / K + 1 1^T) g = Y z / K + 1``; relative trust is near inverse variance, the pooled
    value near the precision-weighted mean, and the pooled weight ``s < 1`` is not the precision."""
    z, Y = data()
    a, loud, sysm = trust_system("prediction", Y, 1.0, [1.0, 1.0, 1.0])
    e = make_input(message(jnp.asarray(z), 1.0), 0.0)
    q = run(sysm, e, 60000)[-3:]
    g = np.asarray(loud(jnp.asarray(q)))
    G = Y @ Y.T / K + np.ones((3, 3))
    g_ls = np.linalg.solve(G, Y @ z / K + np.ones(3))
    prec = 1 / SIGMAS ** 2
    print("positions", q, "loudness", g, "least squares", g_ls)
    print("relative trust", g / g.sum(), "inverse variance", prec / prec.sum())
    print("pooled weight", g.sum(), "precision", prec.sum())
    print("held-out error variance", held_out(g, SIGMAS), "1/precision", 1 / prec.sum())
    assert np.all(g_ls <= 1) and np.allclose(g, g_ls, atol=1e-5)
    assert np.abs(q[0]) < np.abs(q[1]) < np.abs(q[2])        # reliable near, unreliable far
    assert g.sum() < 1


def test_4b_likelihood_learns_precision():
    """Prediction (3), per-source likelihood: ``e^{l_i} -> 1 / mean (z - y_i)^2``, so the pooled
    weight is the precision and held-out error variance matches ``1 / weight``."""
    z, Y = data()
    a, loud, sysm = trust_system("likelihood", Y, 0.5, [0.0, 0.0, 0.0])
    e = make_input(message(jnp.asarray(z), 1.0), 0.0)
    q = run(sysm, e, 4000)[-3:]
    g = np.exp(q)
    g_ml = 1 / np.mean((z[None] - Y) ** 2, axis=1)
    hv = held_out(g, SIGMAS)
    print("precisions", g, "ML", g_ml, "pooled weight", g.sum(), "held-out var", hv, "1/weight", 1 / g.sum())
    assert np.allclose(g, g_ml, rtol=1e-8)
    assert abs(hv * g.sum() - 1) < 0.05


# ---------------------------------------------------------------------------
# Part 5: copied reports.
# ---------------------------------------------------------------------------


def test_5_copies():
    """Prediction (4), under learning: with source 2 a copy of source 1 (all ``sigma = 1``), the
    likelihood cost trusts each copy fully (weight 3 claimed, true precision 2; held-out variance
    ``5/9``); the private-prediction cost splits trust between the copies (relative trust
    ``(1/4, 1/4, 1/2)``, held-out variance ``1/2``)."""
    sig = np.ones(3)
    z, Y = data(sigmas=sig, copy=(0, 1), seed=4)
    e = make_input(message(jnp.asarray(z), 1.0), 0.0)
    _, _, s_lik = trust_system("likelihood", Y, 0.5, [0.0, 0.0, 0.0])
    g_lik = np.exp(run(s_lik, e, 4000)[-3:])
    _, loud, s_pred = trust_system("prediction", Y, 1.0, [1.0, 1.0, 1.0])
    g_pred = np.asarray(loud(jnp.asarray(run(s_pred, e, 60000)[-3:])))

    def held(g, Kmc=200000):
        zz, YY = data(Kmc, sig, seed=5, copy=(0, 1))
        return np.var((g[:, None] * YY).sum(0) / g.sum() - zz)

    print("likelihood: trust", g_lik, "weight", g_lik.sum(), "held-out", held(g_lik))
    print("prediction: relative", g_pred / g_pred.sum(), "weight", g_pred.sum(), "held-out", held(g_pred))
    assert g_lik.sum() > 2.5 and abs(held(g_lik) - 5 / 9) < 0.02
    assert np.allclose(g_pred / g_pred.sum(), [0.25, 0.25, 0.5], atol=0.03)
    assert abs(held(g_pred) - 0.5) < 0.02


# ---------------------------------------------------------------------------
# Part 6: learning whom to trust by softmin lookup (book Ch 4 sec.learning_where, cut 2026-10-28; the version with Bob's query fixed, which Ch 4 ex.learning_trust, rebuilt 2026-10-28, no longer is).
# ---------------------------------------------------------------------------


def test_6_softmin_lookup_learns_inverse_variance():
    """The trust example of sec.learning_where (cut 2026-10-28), before arrangements: Bob's query fixed at ``0``,
    three sources at addresses ``k_i`` with softmin weights ``P = softmin(k^2 / 2)``, his cost the
    average over the days of ``½ (P . y - z)^2``; gradient descent on the addresses. Prediction: the
    weights settle at the least squares with ``sum P = 1``, close to the inverse average squared errors, normalized, and Bob's
    average squared error falls below the best source's."""
    z, Y = data()
    cost = jax.jit(lambda k: 0.5 * jnp.mean((jax.nn.softmax(-0.5 * k ** 2) @ Y - z) ** 2))
    g = jax.jit(jax.grad(cost))

    @jax.jit
    def many(k):
        return jax.lax.fori_loop(0, 200000, lambda _, k: k - 2.0 * g(k), k)

    k = many(jnp.array([1.0, 1.2, 1.4]))
    P = np.asarray(jax.nn.softmax(-0.5 * k ** 2))
    C, b = Y @ Y.T / K, Y @ z / K
    Ci, one = np.linalg.inv(C), np.ones(3)
    P_ls = Ci @ (b + (1 - one @ Ci @ b) / (one @ Ci @ one) * one)
    err = np.mean((P @ Y - z) ** 2)
    best = np.min(np.mean((Y - z) ** 2, axis=1))
    print("addresses", np.asarray(k), "weights", P, "least squares", P_ls, "Bob's error", err, "best source", best)
    assert np.allclose(P, P_ls, atol=1e-8)
    # close to inverse average squared error in this sample (not the population sigmas: at K = 400 the
    # sample's errors are slightly correlated, which is all that separates P from this)
    prec = 1 / np.mean((Y - z) ** 2, axis=1)
    assert np.allclose(P, prec / prec.sum(), atol=0.01)
    assert err < best


def test_6b_softmin_lookup_is_an_arrangement():
    """Test 6's hand loop beside the arrangement it replaces: the hub of part 4 with loudness
    ``exp(-k_i^2 / 2)`` at placement ``k_i`` and Bob's cost on the *value* of what it reads out, the
    average over days of ``½ (h_d / w - z_d)^2`` (``w > 0`` throughout, every scout sending weight 1
    and every loudness positive). Its value is the softmin average, so its gradient-descent steps are
    test 6's; checked equal over 2000 steps at rate 2."""
    z, Y = data()
    n = len(Y)
    loud = lambda q: jnp.exp(-q ** 2 / 2)

    def cost(q, m_out, e):
        ms = m_out.reshape(n, K + 1)
        h = (loud(q)[:, None] * ms[:, :K]).sum(0)
        w = (loud(q) * ms[:, K]).sum()
        return 0.5 * jnp.mean((h / w - e[:K]) ** 2)

    a = hub(n, K, loud, lambda q: q.reshape(n, 1), cost, 2.0, [1.0, 1.2, 1.4])
    sysm = compose_seq(tensor_arrangements([scout(Y[i], f"s{i}") for i in range(n)]), a.arr())
    e = make_input(message(jnp.asarray(z), 1.0), 0.0)
    q_arr = run(sysm, e, 2000)[-3:]
    g = jax.jit(jax.grad(lambda k: 0.5 * jnp.mean((jax.nn.softmax(-0.5 * k ** 2) @ Y - z) ** 2)))
    k = jax.lax.fori_loop(0, 2000, lambda _, k: k - 2.0 * g(k), jnp.array([1.0, 1.2, 1.4]))
    print("arrangement", q_arr, "hand loop", np.asarray(k), "label", sysm.label)
    assert np.allclose(q_arr, np.asarray(k), atol=1e-10)
