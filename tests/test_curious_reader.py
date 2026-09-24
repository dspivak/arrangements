"""A curious reader: an information-gain term in the reader's cost.

The construction is the book's note ``notes/ideas-locke-sensorimotor-2026-09-24.md`` §3
("Can the book's cost carry an information-gain term?"), on the three-child spatial
arrangement of ``tests/test_reader.py``. **Every docstring below says which of three
things its test does** -- reproduce a closed form the note derives, check a discrete-time
analogue the book still has to prove (STATUS.md §3.2), or, as STATUS.md §3.5 records the
commons model doing, correct a claim that was made on paper.

The reader is midway between a weak confirming source ``A`` (signal ``pi``, weight
``w_A = 1``, at ``-D``) and a strong disconfirming source ``B`` (signal ``pi + d``,
weight ``w_B = 4``, at ``+D``), with ``D = sqrt(2 tau log 2)`` so that both kernel values
are ``1/2`` at the midpoint. Its cost is **not** eq:prediction on the weighted input but

    k = (1/2) (hbar - pi)^2  -  lam (1/2) log(1 + w_recv / w_pi),

with ``hbar`` the mean readout of what it receives (eq:spatial-mean) and ``w_recv`` the
received weight, that readout's denominator. The note calls the second term
``k_info``: Lindley's expected information gain of a Gaussian channel, deterministic in
the precisions. The mean readout inside the cost leaves the paper's affine-quadratic
family, as the paper says it would; the Hessian is therefore not constant, and every run
below chooses its mobility from a bound on the Hessian over a named region.

The sources are frozen (mobility ``0`` on their positions and on every value map), so the
reader does all the moving, and the sources' signals are fixed numbers: ``A`` sends
``(w_A pi_0, w_A)``, ``B`` sends ``(w_B (pi_0 + d), w_B)``. The composite is
``compose_seq(tensor(reader, A, B), a.arr())``; ``k_reader`` below is only the closed
form it is checked against.

Part 4 (stage B') adds the prior term ``(w_pi/2) (pi - pi_0)^2`` that part 3 found
missing, with the same ``w_pi`` in both places, and checks three predictions written
before the run. Part 5 (stage B'') runs the window ``(lam*/2, lam_A)`` that part 4 left
unrun, against a prediction written before the run.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact
from arrangements.attention import (
    input_dim, make_input, mean, message, msg_dim, split_input, spatial_arrangement,
    spatial_config, weight, zero_cost,
)
from arrangements.coalgebra import configuration
from arrangements.rvect import constant, euclidean, trivial
from arrangements.wiring import compose_seq, tensor_arrangements

from tests.chain import flat_state

DH, DP = 1, 1
MDIM, EDIM = msg_dim(DH), input_dim(DH, DP)
TAU = 0.8                                       # the Gaussian scale of eq:spatial-transfers
D = float(np.sqrt(2.0 * TAU * np.log(2.0)))     # the paper's d = sqrt(2 tau log 2)
EXT = make_input(message(0.0, 0.0), 0.0)        # the exterior input, fixed to (0_M, 0)

W_A, W_B = 1.0, 4.0        # the sources' weights (precisions)
DELTA = 1.0                # B's signal is pi_0 + DELTA
W_PI = 1.0                 # the reader's prior precision, in k_info
PI0 = 0.0                  # the reader's prediction (fixed, or its starting value)
KAPPA = 0.5                # both kernel values at the midpoint
POSITIONS = (0.0, -D, D)   # reader, A, B


def lam_star():
    """The note's crossover,

        lam* = 4 w_A w_B^2 d^2 (w_pi + kappa (w_A + w_B)) / (kappa (w_A + w_B)^3 (w_B - w_A)),

    about 1.19 with these numbers; the kernel slope ``kappa'`` cancels, so ``tau`` does
    not enter."""
    return (4.0 * W_A * W_B ** 2 * DELTA ** 2 * (W_PI + KAPPA * (W_A + W_B))
            / (KAPPA * (W_A + W_B) ** 3 * (W_B - W_A)))


# ---------------------------------------------------------------------------
# The cost, the three components, and the composite.
# ---------------------------------------------------------------------------


def curious_cost(lam, w_pi=W_PI):
    """``k(pi, e) = (1/2)(hbar - pi)^2 - lam (1/2) log(1 + w_recv / w_pi)`` on an input
    ``e = ((h, w), p)``: ``hbar = mean(h, w) = h / w`` (eq:mean) and ``w_recv = w``."""

    def k(pi, e):
        m, _ = split_input(e, DH)
        return (0.5 * jnp.sum((mean(m) - pi) ** 2)
                - 0.5 * lam * jnp.log1p(weight(m) / w_pi))

    return k


def component(m, cost, label, P=None):
    """A component ``I -> B``: it sends the fixed message ``m in M``, receives an input
    ``e in E``, and carries the cost ``cost(q, e)`` on its own configuration ``q`` (by
    default none) and its input."""
    return Arrangement(
        P=configuration(trivial()) if P is None else P,
        in_dim_M=0, out_dim_M=0, in_dim_N=EDIM, out_dim_N=MDIM,
        out_f=lambda q, m_out: m,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -cost(q, n_in)),
        label=label,
    )


def fixed_reader(lam, pi=PI0):
    """The reader with its prediction ``pi`` held fixed: message ``0_M``, no
    configuration of its own."""
    k = curious_cost(lam)
    return component(message(0.0, 0.0), lambda q, e: k(pi, e), "reader")


def free_reader(lam, mu_pi, pi0=PI0):
    """The reader with its prediction as a configuration coordinate ``q = (pi,)`` of its
    own, Euclidean mobility ``mu_pi``, starting at ``pi0``; the same cost."""
    k = curious_cost(lam)
    return component(message(0.0, 0.0), lambda q, e: k(q[0], e), "reader",
                     P=configuration(euclidean(1, mu_pi), init=jnp.array([pi0])))


def source(sig, w, label):
    """A source of signal ``sig`` with weight ``w``: message ``(w sig, w)``, zero cost."""
    return component(message(w * sig, w), lambda q, e: jnp.asarray(0.0), label)


def curious_system(rdr, mu_reader):
    """``a(reader, A, B) : I -> B`` by composition in ``arr``, the spatial arrangement of
    eq:spatial-transfers with zero cost. Only the reader's position responds; the
    sources' positions and all three value maps have mobility ``0``."""
    a = spatial_arrangement(
        3, DH, DP, TAU, cost=zero_cost,
        parameter=configuration(
            constant(jnp.diag(jnp.array([mu_reader, 0.0, 0.0, 0.0, 0.0, 0.0]))),
            init=spatial_config(POSITIONS, (1.0, 1.0, 1.0))),
        label="spatial(3)")
    return compose_seq(
        tensor_arrangements([rdr, source(PI0, W_A, "A"), source(PI0 + DELTA, W_B, "B")]),
        a.arr())


def kernels(p):
    """``(kappa_A, kappa_B)`` at reader position ``p``, the sources at ``-D`` and ``D``."""
    return jnp.exp(-(p + D) ** 2 / (2.0 * TAU)), jnp.exp(-(p - D) ** 2 / (2.0 * TAU))


def w_recv(p):
    """The received weight ``kappa_A w_A + kappa_B w_B`` (eq:spatial-mean's denominator)."""
    kA, kB = kernels(p)
    return kA * W_A + kB * W_B


def hbar(p):
    """The mean readout ``(kappa_A w_A pi_0 + kappa_B w_B (pi_0 + d)) / w_recv``."""
    kA, kB = kernels(p)
    return (kA * W_A * PI0 + kB * W_B * (PI0 + DELTA)) / w_recv(p)


def k_reader(p, pi, lam):
    """The closed form the composite is checked against: the reader's cost pulled back
    to its position ``p`` and prediction ``pi``. Nothing builds the composite from it."""
    return 0.5 * (hbar(p) - pi) ** 2 - 0.5 * lam * jnp.log1p(w_recv(p) / W_PI)


def midpoint_force(lam):
    """The composite forcing's component on the reader's position at the midpoint;
    positive means toward ``B`` (at ``+D``)."""
    b = curious_system(fixed_reader(lam), 1.0)
    q = spatial_config(POSITIONS, (1.0, 1.0, 1.0))
    xi_q, _, _ = b.omega(q, jnp.zeros(0), EXT)
    return float(xi_q[0])


# ---------------------------------------------------------------------------
# Running to settlement, and choosing the mobility.
# ---------------------------------------------------------------------------


def run(arr, steps):
    """``Phi(arr)`` on the constant exterior input, jitted as in
    ``tests/test_rendezvous.py``'s ``run_closed``; returns the flat configurations and
    the last forcing on ``Q``."""
    P, omega, empty = arr.P, arr.omega, jnp.zeros(0)

    @jax.jit
    def step(s):
        xi_q, _, _ = omega(P.position(s), empty, EXT)
        return P.step(s, xi_q), xi_q

    s, out = P.init, [np.asarray(flat_state(P.init))]
    for _ in range(steps):
        s, xi = step(s)
        out.append(np.asarray(flat_state(s)))
    return np.array(out), np.asarray(xi)


REGION = (-3.0 * D, 3.0 * D)     # the position region the Hessian is bounded over


def mobility_fixed(lam):
    """``1 / L`` with ``L = max |k''(p)|`` over ``REGION`` at fixed ``pi``: for an
    ``L``-smooth cost a step of at most ``1 / L`` decreases it every step (the descent
    lemma). The region is named because the constant is region-dependent (STATUS.md
    §3.2 item 6); the runs stay inside it and the tests check that."""
    ps = jnp.linspace(*REGION, 2001)
    kpp = jax.vmap(jax.hessian(lambda p: k_reader(p, PI0, lam)))(ps)
    return 1.0 / float(jnp.max(jnp.abs(kpp)))


def mobility_free(lam):
    """``1 / L`` with ``L`` the largest ``|eigenvalue|`` of the ``(p, pi)`` Hessian over
    ``REGION x [pi_0 - 1, pi_0 + d + 1]``; the same bound for both coordinates."""
    ps = jnp.linspace(*REGION, 401)
    pis = jnp.linspace(PI0 - 1.0, PI0 + DELTA + 1.0, 101)
    H = jax.vmap(lambda p: jax.vmap(
        lambda pi: jax.hessian(lambda y: k_reader(y[0], y[1], lam))(jnp.array([p, pi])))(pis))(ps)
    return 1.0 / float(jnp.max(jnp.abs(jnp.linalg.eigvalsh(H))))


def local_minimizer(lam, lo, hi):
    """The zero of ``k'(p)`` in ``[lo, hi]`` by bisection, at fixed ``pi = pi_0``: the
    independent computation the settled position is compared with."""
    kp = jax.grad(lambda p: k_reader(p, PI0, lam))
    assert float(kp(lo)) < 0.0 < float(kp(hi))
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if float(kp(mid)) < 0.0 else (lo, mid)
    return 0.5 * (lo + hi)


def decreases_every_step(costs):
    return bool(np.all(np.diff(costs) <= 1e-15))       # to rounding, once settled


# ---------------------------------------------------------------------------
# The composite is the note's cost.
# ---------------------------------------------------------------------------


def test_the_mean_readout_cost_emerges_from_substitution():
    """REPRODUCES A CLOSED FORM: substitution pulls the reader's cost back to
    ``k_reader(p_1, pi_0, lam)`` on the reader's position. The composite's forcing on
    ``p_1`` is ``-d k_reader / d p_1`` at random positions, so the identity is the whole
    function and not one value; the forcing on the frozen coordinates is not checked,
    since nothing here responds to it."""
    lam = 0.7
    b = curious_system(fixed_reader(lam), 1.0)
    kp = jax.grad(lambda p: k_reader(p, PI0, lam))
    r = np.random.default_rng(302)
    for _ in range(6):
        p = float(r.uniform(-2.0 * D, 2.0 * D))
        q = spatial_config((p, -D, D), (1.0, 1.0, 1.0))
        xi_q, _, _ = b.omega(q, jnp.zeros(0), EXT)
        np.testing.assert_allclose(float(xi_q[0]), -float(kp(p)), atol=1e-13)


# ---------------------------------------------------------------------------
# Part 1: the crossover.
# ---------------------------------------------------------------------------


def test_the_midpoint_force_crosses_zero_at_lambda_star():
    """REPRODUCES A CLOSED FORM the note derives. At the midpoint the error term pushes
    the reader away from ``B`` with force ``2 w_A w_B^2 d^2 kappa' / (kappa (w_A+w_B)^3)``
    and the information term toward ``B`` with force
    ``lam kappa' (w_B - w_A) / (2 (w_pi + kappa (w_A+w_B)))``, ``kappa' = D / 2 tau``;
    both are read off the autodiff forcing (its value at ``lam = 0`` and its slope in
    ``lam``). The sign change, found by bisection, is at ``lam*`` to rounding: the
    measured crossover and the formula agree to better than 1e-12 relative.

    Below ``lam*`` the reader is pushed toward ``A``, the confirming source; above it,
    toward ``B``, the precise disconfirming one. This is a statement about one
    configuration with the prediction held fixed -- a local response."""
    kappa_p = D / (2.0 * TAU)
    error_force = -2.0 * W_A * W_B ** 2 * DELTA ** 2 * kappa_p / (KAPPA * (W_A + W_B) ** 3)
    info_slope = kappa_p * (W_B - W_A) / (2.0 * (W_PI + KAPPA * (W_A + W_B)))
    np.testing.assert_allclose(midpoint_force(0.0), error_force, rtol=1e-13)
    np.testing.assert_allclose(midpoint_force(1.0) - midpoint_force(0.0), info_slope, rtol=1e-12)

    lo, hi = 0.0, 3.0
    assert midpoint_force(lo) < 0.0 < midpoint_force(hi)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if midpoint_force(mid) < 0.0 else (lo, mid)
    crossover = 0.5 * (lo + hi)
    assert abs(crossover - lam_star()) / lam_star() < 1e-12
    np.testing.assert_allclose(lam_star(), 1.1946666666666668, rtol=1e-15)

    assert midpoint_force(1.1) < 0.0 < midpoint_force(1.3)


# ---------------------------------------------------------------------------
# Part 2: descent from the midpoint with the prediction fixed.
# ---------------------------------------------------------------------------


def test_descent_from_the_midpoint_selects_a_basin():
    """CHECKS THE DISCRETE ANALOGUE (STATUS.md §3.2) of the note's run with the
    prediction held fixed. ``k_reader(-, pi_0, lam)`` has two local minima, one near
    each source, with the midpoint a local maximum between them; the sign of the
    midpoint force therefore selects the basin. With mobility ``1 / L`` (``L`` bounding
    ``|k''|`` over ``REGION``) the cost decreases every step, the run stays in the
    region, and it settles -- forcing below 1e-12 -- at the basin's minimizer computed
    independently: at ``lam = 1.1`` near ``A`` (``p = -0.835``, ``w_recv = 1.40``), at
    ``lam = 1.3`` near ``B`` (``p = 0.925``, ``w_recv = 4.05``). Neither minimizer is at
    the source itself: the other source's tail still pulls."""
    expected = {1.1: (-0.8351057, 1.4016, (-1.5 * D, 0.0)),
                1.3: (0.9251135, 4.0459, (0.0, 1.5 * D))}
    for lam, (p_star, w_star, bracket) in expected.items():
        mu = mobility_fixed(lam)
        assert 1.5 < mu < 2.5                       # 2.10 at 1.1, 1.78 at 1.3
        tr, xi = run(curious_system(fixed_reader(lam), mu), 300)
        p = tr[:, 0]
        costs = np.asarray(jax.vmap(lambda p: k_reader(p, PI0, lam))(jnp.asarray(p)))

        assert decreases_every_step(costs)
        assert REGION[0] < p.min() and p.max() < REGION[1]
        assert abs(xi[0]) < 1e-12
        np.testing.assert_allclose(p[-1], local_minimizer(lam, *bracket), atol=1e-10)
        np.testing.assert_allclose(p[-1], p_star, atol=1e-6)
        np.testing.assert_allclose(float(w_recv(p[-1])), w_star, atol=1e-4)
    assert expected[1.1][0] < 0.0 < expected[1.3][0]


# ---------------------------------------------------------------------------
# Part 3: the prediction as a descended coordinate.
# ---------------------------------------------------------------------------


def test_a_free_prediction_ends_in_the_precise_sources_basin_at_both_lambdas():
    """CORRECTS A CLAIM made on paper, as STATUS.md §3.5 records the commons model
    doing. The note predicts that with the prediction a descended coordinate the reader
    "at lam = 1.1 goes to A and never revises it; at lam = 1.3 goes to B and revises pi
    toward B's signal". The second half holds; the first does not, and cannot with this
    cost: ``dk/dpi = -(hbar(p) - pi)`` vanishes only at ``pi = hbar(p)``, which zeroes the
    error term wherever the reader is, so on the stationary set only
    ``-lam/2 log(1 + w_recv(p) / w_pi)`` remains and its minimizer is where ``w_recv``
    peaks -- ``B``'s side, for every ``lam > 0``, since ``w_B > w_A``. ``lam*`` governs
    the initial force at a *fixed* prediction, not where a reader with a free prediction
    settles. Nothing anchors ``pi`` to ``pi_0``: a prior term ``(w_pi/2)(pi - pi_0)^2``
    is not in the note's cost, and this test does not add one.

    Both regimes settle at the same point, ``p = 1.018``, ``pi = 0.983 = hbar(p)``, with
    ``w_recv'(p) = 0`` there; the cost decreases every step under the ``(p, pi)`` Hessian
    bound. With the prior slow (``mu_pi = mu_p / 50``) the ``lam = 1.1`` reader does first
    move toward ``A`` -- to ``p = -0.49``, nearly half way -- and turns back once its
    prediction has caught up with what it hears; with equal mobilities it barely moves
    toward ``A`` at all, because the force on ``pi`` (``0.8``) dwarfs the net force on
    ``p`` (``-0.027``)."""
    # the stationary point, computed independently: w_recv'(p) = 0 on B's side, pi = hbar(p)
    wp = jax.grad(w_recv)
    lo, hi = 0.0, 1.5 * D
    assert float(wp(lo)) > 0.0 > float(wp(hi))
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if float(wp(mid)) > 0.0 else (lo, mid)
    p_star = 0.5 * (lo + hi)
    pi_star = float(hbar(p_star))
    np.testing.assert_allclose([p_star, pi_star], [1.0176, 0.9831], atol=1e-4)

    for lam in (1.1, 1.3):
        mu = mobility_free(lam)
        assert 0.6 < mu < 0.7                       # 0.651 at 1.1, 0.656 at 1.3
        for ratio in (1.0, 0.02):
            tr, xi = run(curious_system(free_reader(lam, mu * ratio), mu), 4000)
            pi, p = tr[:, 0], tr[:, 1]
            costs = np.asarray(jax.vmap(lambda y: k_reader(y[1], y[0], lam))(jnp.asarray(tr[:, :2])))

            assert decreases_every_step(costs)
            assert REGION[0] < p.min() and p.max() < REGION[1]
            assert np.max(np.abs(xi[:2])) < 1e-12
            np.testing.assert_allclose([p[-1], pi[-1]], [p_star, pi_star], atol=1e-10)
            np.testing.assert_allclose(pi[-1], float(hbar(p[-1])), atol=1e-12)
            assert abs(float(wp(p[-1]))) < 1e-11
            assert pi[-1] > PI0 + 0.9 * DELTA        # revised almost all the way to B's signal

            if lam == 1.1 and ratio == 0.02:
                assert p.min() < -0.45               # first toward A, nearly half way
            if lam == 1.1 and ratio == 1.0:
                assert p.min() > -0.02               # barely toward A at all
            if lam == 1.3:
                assert p.min() >= 0.0                # never toward A


# ---------------------------------------------------------------------------
# Part 4 (stage B'): the prediction as a descended coordinate with a prior.
# ---------------------------------------------------------------------------
#
# The cost is part 3's with the prior term the note left out,
#
#     k = (1/2) (hbar - pi)^2 - lam (1/2) log(1 + w_recv / w_pi) + (w_pi/2) (pi - pi_0)^2,
#
# one ``w_pi`` in both places: the Gaussian prior precision on the prediction. Every run
# starts at ``p = 0``, ``pi = pi_0``. The three predictions, written before the run:
#
#   (i)   eliminating ``pi`` at its posterior mean ``(hbar + w_pi pi_0) / (1 + w_pi)``
#         leaves the held-prediction cost with the error term scaled by
#         ``w_pi / (1 + w_pi)``, so the reduced midpoint force changes sign at ``lam*/2``;
#   (ii)  the initial force on ``p`` in the full system is the held-prediction force,
#         sign change at ``lam*``;
#   (iii) the settled basin has three regimes: ``lam < lam*/2`` A's side for any mobility
#         ratio, ``lam > lam*`` B's side for any, and in between mobility-dependent.


def k_prior(p, pi, lam, w_pi=W_PI):
    """The closed form for part 4: ``k_reader`` plus the prior ``(w_pi/2)(pi - pi_0)^2``."""
    return k_reader(p, pi, lam) + 0.5 * w_pi * (pi - PI0) ** 2


def pi_post(p, w_pi=W_PI):
    """The posterior mean ``(hbar(p) + w_pi pi_0) / (1 + w_pi)``: the minimizer of
    ``k_prior`` over ``pi`` at every ``p``, since the cost is quadratic in ``pi``."""
    return (hbar(p) + w_pi * PI0) / (1.0 + w_pi)


def k_reduced(p, lam, w_pi=W_PI):
    """``k_prior`` with ``pi`` eliminated at its posterior mean."""
    return k_prior(p, pi_post(p, w_pi), lam, w_pi)


def prior_reader(lam, mu_pi, pi0=PI0, w_pi=W_PI):
    """``free_reader`` with the prior term added to its cost; ``w_pi`` is the same
    number that ``curious_cost`` uses in ``k_info``."""
    k = curious_cost(lam, w_pi)
    return component(message(0.0, 0.0),
                     lambda q, e: k(q[0], e) + 0.5 * w_pi * (q[0] - pi0) ** 2, "reader",
                     P=configuration(euclidean(1, mu_pi), init=jnp.array([pi0])))


def prior_force(lam, pi):
    """The composite forcing's ``(pi, p)`` components with the reader at the midpoint
    and its prediction at ``pi``, for the reader with a prior."""
    b = curious_system(prior_reader(lam, 1.0), 1.0)
    q = jnp.concatenate([jnp.array([pi]), spatial_config(POSITIONS, (1.0, 1.0, 1.0))])
    xi_q, _, _ = b.omega(q, jnp.zeros(0), EXT)
    return float(xi_q[0]), float(xi_q[1])


def mobility_prior(lam, ratio):
    """``mu_p`` for the pair ``(mu_p, mu_pi) = (mu_p, ratio mu_p)``, from the descent
    lemma in the mobility-weighted metric: in ``y = M^{-1/2} (p, pi)`` the step is unit
    gradient descent and the Hessian is ``mu_p S H S`` with ``S = diag(1, sqrt ratio)``,
    so ``mu_p = 1 / L`` with ``L`` the largest ``|eigenvalue|`` of ``S H S`` over
    ``REGION x [pi_0 - 1, pi_0 + d + 1]`` makes the cost decrease every step."""
    ps = jnp.linspace(*REGION, 401)
    pis = jnp.linspace(PI0 - 1.0, PI0 + DELTA + 1.0, 101)
    S = jnp.diag(jnp.array([1.0, jnp.sqrt(ratio)]))
    H = jax.vmap(lambda p: jax.vmap(
        lambda pi: S @ jax.hessian(lambda y: k_prior(y[0], y[1], lam))(jnp.array([p, pi])) @ S)(pis))(ps)
    return 1.0 / float(jnp.max(jnp.abs(jnp.linalg.eigvalsh(H))))


def reduced_minimizer(lam, lo, hi):
    """The zero of ``k_reduced'(p)`` in ``[lo, hi]`` by bisection: the independent
    computation the settled position is compared with (``local_minimizer`` for part 4)."""
    kp = jax.grad(k_reduced)
    assert float(kp(lo, lam)) < 0.0 < float(kp(hi, lam))
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if float(kp(mid, lam)) < 0.0 else (lo, mid)
    return 0.5 * (lo + hi)


def lam_A():
    """The ``lam`` above which ``k_reduced`` has no local minimum on ``A``'s side. Far
    to the left only the information term's tail pulls inward, so ``k_reduced' < 0``
    there; ``A``'s basin exists iff ``k_reduced'`` turns positive somewhere in
    ``[-2D, 0]``. Bisection on ``lam`` of that maximum over a grid."""
    kp = jax.grad(k_reduced)
    ps = jnp.linspace(-2.0 * D, 0.0, 4001)
    left_max = lambda lam: float(jnp.max(jax.vmap(lambda p: kp(p, lam))(ps)))
    lo, hi = 0.4, 0.9
    assert left_max(lo) > 0.0 > left_max(hi)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if left_max(mid) > 0.0 else (lo, mid)
    return 0.5 * (lo + hi)


def test_eliminating_the_prediction_scales_the_error_term_and_halves_the_crossover():
    """REPRODUCES A CLOSED FORM, prediction (i). ``k_reduced(p)`` equals
    ``(w_pi/(1+w_pi)) (1/2)(hbar - pi_0)^2 - (lam/2) log(1 + w_recv/w_pi)`` -- at every
    ``p``, not only at stationary points, since the elimination is a quadratic
    minimization. The reduced midpoint force is computed two ways, by autodiff of
    ``k_reduced`` and as the composite's force on ``p`` at ``(0, pi_post(0))`` (the
    envelope theorem: the ``pi``-gradient vanishes there), and they agree to 1e-13. Its
    sign change, by bisection, is at ``lam*/2 = 0.597333`` to 1e-12 relative; the
    prediction's ``0.597`` differs from it by 5.6e-4 relative."""
    r = np.random.default_rng(303)
    for _ in range(6):
        p = float(r.uniform(-2.0 * D, 2.0 * D))
        scaled = (W_PI / (1.0 + W_PI)) * 0.5 * (hbar(p) - PI0) ** 2 - 0.5 * 0.9 * jnp.log1p(w_recv(p) / W_PI)
        np.testing.assert_allclose(float(k_reduced(p, 0.9)), float(scaled), atol=1e-15)

    def reduced_force(lam):
        return -float(jax.grad(k_reduced)(0.0, lam))

    for lam in (0.0, 0.5, 1.0, 2.0):
        _, f_p = prior_force(lam, float(pi_post(0.0)))
        np.testing.assert_allclose(f_p, reduced_force(lam), atol=1e-13)

    lo, hi = 0.0, 3.0
    assert reduced_force(lo) < 0.0 < reduced_force(hi)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if reduced_force(mid) < 0.0 else (lo, mid)
    crossover = 0.5 * (lo + hi)
    assert abs(crossover - 0.5 * lam_star()) / (0.5 * lam_star()) < 1e-12
    np.testing.assert_allclose(crossover, 0.5973333333, rtol=1e-9)
    assert abs(crossover - 0.597) / 0.597 < 1e-3


def test_the_initial_force_with_a_prior_is_the_held_prediction_force():
    """REPRODUCES A CLOSED FORM, prediction (ii). At ``(p, pi) = (0, pi_0)`` the prior's
    gradient vanishes in both coordinates, so the composite's force on ``p`` is
    ``midpoint_force(lam)`` of part 1 to 1e-13 and its force on ``pi`` is
    ``hbar(0) - pi_0 = 0.8``, as in part 3. The sign change of the force on ``p`` is at
    ``lam* = 1.194667`` to 1e-12 relative."""
    for lam in (0.0, 0.5, 1.0, 2.0):
        f_pi, f_p = prior_force(lam, PI0)
        np.testing.assert_allclose(f_p, midpoint_force(lam), atol=1e-13)
        np.testing.assert_allclose(f_pi, float(hbar(0.0)) - PI0, atol=1e-13)
        np.testing.assert_allclose(f_pi, 0.8, atol=1e-13)

    lo, hi = 0.0, 3.0
    assert prior_force(lo, PI0)[1] < 0.0 < prior_force(hi, PI0)[1]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if prior_force(mid, PI0)[1] < 0.0 else (lo, mid)
    crossover = 0.5 * (lo + hi)
    assert abs(crossover - lam_star()) / lam_star() < 1e-12


def test_with_a_prior_the_middle_lambda_does_not_depend_on_the_mobility_ratio():
    """CORRECTS A CLAIM made on paper, prediction (iii). The outer regimes hold: at
    ``lam = 0.4`` all three mobility ratios settle on ``A``'s side, at ``lam = 1.4`` all
    three on ``B``'s. The middle row does not: at ``lam = 0.9`` all three ratios settle
    at the same point on ``B``'s side, ``(p, pi) = (0.956, 0.490)``, to 1e-10. The
    reason is computed independently: the stationary points of the full cost are those
    of ``k_reduced``, and ``k_reduced`` has no local minimum on ``A``'s side above
    ``lam_A = 0.8074`` -- with ``w_pi = 1`` it is half the held cost at ``2 lam``, whose
    ``A`` basin closes at ``2 lam_A = 1.615``. So the second threshold is ``lam_A``,
    not ``lam*``, and the window where the ratio could matter is at most
    ``(lam*/2, lam_A) = (0.597, 0.807)``; ``0.9`` lies above it. What happens inside
    that window is not tested here.

    The slow-prediction reader at ``lam = 0.9`` does go most of the way to ``A`` --
    ``p = -0.78``, against the held ``A`` minimizer ``-0.916`` -- and turns back once its
    prediction catches up, as in part 3; the fast one never moves toward ``A``. Each
    run's cost decreases every step under the mobility-weighted Hessian bound, stays in
    the named region, and settles (forcing below 1e-12) at the minimizer of
    ``k_reduced`` computed by bisection, with ``pi = pi_post(p)`` there."""
    threshold = lam_A()
    np.testing.assert_allclose(threshold, 0.8074, atol=1e-3)
    assert 0.5 * lam_star() < threshold < 0.9

    expected = {0.4: (-0.956346, 0.121938, (-1.5 * D, 0.0)),
                0.9: (0.956022, 0.490112, (0.0, 1.5 * D)),
                1.4: (0.980475, 0.490717, (0.0, 1.5 * D))}
    mu_range = {50.0: (0.009, 0.011), 1.0: (0.4, 0.5), 0.02: (0.9, 1.0)}
    settled = {}
    for lam, (p_star, pi_star, bracket) in expected.items():
        p_red = reduced_minimizer(lam, *bracket)
        np.testing.assert_allclose([p_red, float(pi_post(p_red))], [p_star, pi_star], atol=1e-6)
        for ratio in (50.0, 1.0, 0.02):
            mu = mobility_prior(lam, ratio)
            assert mu_range[ratio][0] < mu < mu_range[ratio][1]
            tr, xi = run(curious_system(prior_reader(lam, mu * ratio), mu), 20000)
            pi, p = tr[:, 0], tr[:, 1]
            costs = np.asarray(jax.vmap(lambda y: k_prior(y[1], y[0], lam))(jnp.asarray(tr[:, :2])))

            assert decreases_every_step(costs)
            assert REGION[0] < p.min() and p.max() < REGION[1]
            assert PI0 - 1.0 < pi.min() and pi.max() < PI0 + DELTA + 1.0
            assert np.max(np.abs(xi[:2])) < 1e-12
            np.testing.assert_allclose([p[-1], pi[-1]], [p_red, float(pi_post(p_red))], atol=1e-10)
            settled[lam, ratio] = (p[-1], pi[-1])

            if lam == 0.9 and ratio == 0.02:
                assert p.min() < -0.7                # most of the way to A, then back
            if lam == 0.9 and ratio == 50.0:
                assert p.min() > -0.01               # never toward A
            if lam == 1.4:
                assert p.min() >= 0.0

    for ratio in (50.0, 1.0, 0.02):
        assert settled[0.4, ratio][0] < 0.0          # A's side, every ratio
        assert settled[0.9, ratio][0] > 0.0          # B's side, every ratio: not as predicted
        assert settled[1.4, ratio][0] > 0.0          # B's side, every ratio
    middle = np.array([settled[0.9, r] for r in (50.0, 1.0, 0.02)])
    assert np.ptp(middle, axis=0).max() < 1e-10      # the middle row is one point


# ---------------------------------------------------------------------------
# Part 5 (stage B''): inside the window (lam*/2, lam_A) the mobility ratio selects the basin.
# ---------------------------------------------------------------------------
#
# Part 4 left the window ``(lam*/2, lam_A) = (0.597, 0.807)`` unrun. Inside it both reduced
# basins exist, the reduced midpoint force points toward ``B``, and the initial full-system
# force at ``(0, pi_0)`` points toward ``A`` (``lam < lam*``). The prediction, written
# before the run: a fast prediction (``mu_pi / mu_p = 50``) settles on ``B``'s side; a slow
# one (``1/50``) is carried toward ``A`` by the initial held force, reaches ``A``'s reduced
# basin before its prediction catches up, and settles on ``A``'s side. Equal mobilities were
# not predicted.


def test_inside_the_window_the_mobility_ratio_selects_the_basin():
    """CHECKS THE DISCRETE ANALOGUE (STATUS.md §3.2) of a prediction written before the
    run, and the prediction holds. At ``lam = 0.65`` and ``0.75`` the slow-prediction
    reader settles on ``A``'s side and the fast one on ``B``'s; equal mobilities, not
    predicted, settle on ``B``'s side. Every run's cost decreases every step under the
    mobility-weighted Hessian bound, stays in the named region, and settles (forcing
    below 1e-12) at the minimizer of ``k_reduced`` on its side, with ``pi = pi_post(p)``.

    The boundary of ``A``'s reduced basin is the local maximum of ``k_reduced`` between
    the basins, found by bisection of ``k_reduced'`` from ``+`` to ``-``: ``p = -0.0799``
    at ``lam = 0.65`` and ``-0.2471`` at ``0.75``. ``B``'s basin is the deeper one at both
    ``lam`` (``k_reduced`` at ``A``, saddle, ``B``: ``-0.262, -0.247, -0.287`` and
    ``-0.308, -0.305, -0.368``), so the slow reader settles in a local, not the global,
    minimum. The slow reader does more than reach ``A``'s basin: with its prediction still
    near ``pi_0`` (``0.07``) it overshoots ``A``'s reduced minimizer -- least ``p`` of
    ``-0.917`` against ``-0.747`` at ``0.65``, ``-0.865`` against ``-0.629`` at ``0.75``
    -- and relaxes back to it as the prediction rises, never leaving the basin once in.
    The fast reader never moves toward ``A`` (least ``p`` above ``-0.002``). The
    equal-mobility reader moves toward ``A`` and stops short of the saddle -- ``-0.073``
    against ``-0.080`` at ``0.65``, by a margin of ``0.007``; ``-0.055`` against
    ``-0.247`` at ``0.75`` -- and then goes to ``B``."""
    threshold = lam_A()
    kp = jax.grad(k_reduced)
    expected = {0.65: (-0.079937, (-0.746580, 0.179549), (0.925114, 0.489291)),
                0.75: (-0.247063, (-0.629214, 0.216419), (0.940660, 0.489712))}
    mu_range = {50.0: (0.009, 0.011), 1.0: (0.4, 0.5), 0.02: (0.9, 1.0)}
    settled, least = {}, {}
    for lam, (saddle_star, a_star, b_star) in expected.items():
        assert 0.5 * lam_star() < lam < threshold
        assert -float(kp(0.0, lam)) > 0.0           # reduced midpoint force toward B
        assert prior_force(lam, PI0)[1] < 0.0        # initial full-system force toward A

        # the saddle: k_reduced' peaks on [-2D, 0] (as in lam_A) and is negative at 0
        ps = jnp.linspace(-2.0 * D, 0.0, 4001)
        vals = jax.vmap(lambda p: kp(p, lam))(ps)
        p_top = float(ps[jnp.argmax(vals)])
        assert float(vals.max()) > 0.0               # A's basin exists
        lo, hi = p_top, 0.0
        assert float(kp(lo, lam)) > 0.0 > float(kp(hi, lam))
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if float(kp(mid, lam)) > 0.0 else (lo, mid)
        saddle = 0.5 * (lo + hi)
        p_A = reduced_minimizer(lam, -1.5 * D, p_top)
        p_B = reduced_minimizer(lam, 0.0, 1.5 * D)
        assert p_A < saddle < 0.0 < p_B
        assert float(k_reduced(saddle, lam)) > max(float(k_reduced(p_A, lam)), float(k_reduced(p_B, lam)))
        assert float(k_reduced(p_B, lam)) < float(k_reduced(p_A, lam))     # B's is the deeper basin
        np.testing.assert_allclose(saddle, saddle_star, atol=1e-6)
        np.testing.assert_allclose([p_A, float(pi_post(p_A))], a_star, atol=1e-6)
        np.testing.assert_allclose([p_B, float(pi_post(p_B))], b_star, atol=1e-6)

        for ratio in (50.0, 1.0, 0.02):
            mu = mobility_prior(lam, ratio)
            assert mu_range[ratio][0] < mu < mu_range[ratio][1]
            tr, xi = run(curious_system(prior_reader(lam, mu * ratio), mu), 20000)
            pi, p = tr[:, 0], tr[:, 1]
            costs = np.asarray(jax.vmap(lambda y: k_prior(y[1], y[0], lam))(jnp.asarray(tr[:, :2])))

            assert decreases_every_step(costs)
            assert REGION[0] < p.min() and p.max() < REGION[1]
            assert PI0 - 1.0 < pi.min() and pi.max() < PI0 + DELTA + 1.0
            assert np.max(np.abs(xi[:2])) < 1e-12
            side = p_A if p[-1] < saddle else p_B
            np.testing.assert_allclose([p[-1], pi[-1]], [side, float(pi_post(side))], atol=1e-10)
            settled[lam, ratio] = (p[-1], pi[-1])
            least[lam, ratio] = p.min()

            if ratio == 0.02:
                assert p.min() < p_A                 # overshoots A's reduced minimizer
                entered = int(np.argmax(p < saddle))
                assert np.all(p[entered:] < saddle)  # once in A's basin, stays in it
            if ratio == 50.0:
                assert p.min() > -0.01               # never toward A
            if ratio == 1.0:
                assert saddle < p.min() < 0.0        # toward A, short of the saddle

    for lam in (0.65, 0.75):
        assert settled[lam, 0.02][0] < 0.0           # slow prediction: A's side, as predicted
        assert settled[lam, 50.0][0] > 0.0           # fast prediction: B's side, as predicted
        assert settled[lam, 1.0][0] > 0.0            # equal mobilities: B's side (not predicted)
    np.testing.assert_allclose(least[0.65, 0.02], -0.916837, atol=1e-5)
    np.testing.assert_allclose(least[0.75, 0.02], -0.865396, atol=1e-5)
    np.testing.assert_allclose(least[0.65, 1.0], -0.073386, atol=1e-5)
    np.testing.assert_allclose(least[0.75, 1.0], -0.055389, atol=1e-5)
