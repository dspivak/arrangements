"""Bayes' rule for a whole curve (book ``AGENTDOCPLAN.md`` §8.2 P36 (b)).

Stage (a) (``test_bayes_every_belief.py``) showed one step of an arrangement is Bayes' rule for
every belief over finitely many values. Stage (b) asks for a belief that is a whole curve over the
real line. The proof by hand is Part II of the book repo's ``notes/bayes-every-belief.md``: with
the state the log-height curve, the readout the belief, the likelihood's cost the surprisal
averaged over the belief, and the response "divide by the height", one step is
``l' = l + xi - E_q xi``, Bayes' rule, whenever ``E_q |xi|`` and ``integral q Pr(x | .)`` are
finite.

A machine holds a grid, so the run is the stage (a) box on ``n`` grid values, and the tests are
about how that box stands to the curve as the grid grows. The eight predictions were fixed in
writing before ``belief.belief_coalgebra`` existed (``notes/BAYES-WHOLE-CURVE.md`` §2), and are
quoted in each test's docstring; the verdicts, with the numbers, are that note's §3.

Each test prints what it measured (run with ``PYTEST_ADDOPTS=-s``).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate
from scipy.special import log_ndtr, logsumexp

from arrangements.belief import (
    belief_of,
    column_nll,
    log_height_response,
    log_heights,
    run,
    updating,
    updating_on_beliefs,
)

X = 1.0                      # the reading, unless a test says otherwise
MU = np.array([-1.5, 2.5])   # the two bumps, half the mass on each
SD = np.array([0.6, 0.8])    # G: bell curves
B = 0.5                      # L: Laplace scale
GAMMA = 0.5                  # C: Cauchy scale
LOG2PI = float(np.log(2.0 * np.pi))


# ---------------------------------------------------------------------------
# The three beliefs, as log-densities on the real line.
# ---------------------------------------------------------------------------


def log_q_G(z):
    z = np.asarray(z, dtype=float)[..., None]
    return logsumexp(np.log(0.5) - 0.5 * ((z - MU) / SD) ** 2 - np.log(SD) - 0.5 * LOG2PI, axis=-1)


def log_q_L(z):
    z = np.asarray(z, dtype=float)[..., None]
    return logsumexp(np.log(0.5) - np.abs(z - MU) / B - np.log(2.0 * B), axis=-1)


def log_q_C(z):
    z = np.asarray(z, dtype=float)[..., None]
    return logsumexp(np.log(0.5) + np.log(GAMMA / np.pi) - np.log(GAMMA**2 + (z - MU) ** 2), axis=-1)


# ---------------------------------------------------------------------------
# Closed forms for G with bell-curve noise of variance 1 (arithmetic, not the construction).
# ---------------------------------------------------------------------------


def closed_form_G(readings):
    """The posterior of the two-bump prior after the readings: weights, means, variances of the
    two bell curves, and ``log Pr(readings)``. Each component is updated conjugately."""
    logw, m, v = np.log(np.array([0.5, 0.5])), MU.copy(), SD.copy() ** 2
    for x in readings:
        logw = logw - 0.5 * (x - m) ** 2 / (v + 1.0) - 0.5 * np.log(2.0 * np.pi * (v + 1.0))
        m, v = (m / v + x) / (1.0 / v + 1.0), 1.0 / (1.0 / v + 1.0)
    logev = logsumexp(logw)
    return np.exp(logw - logev), m, v, float(logev)


def log_mixture(z, w, m, v):
    z = np.asarray(z, dtype=float)[..., None]
    return logsumexp(np.log(w) - 0.5 * (z - m) ** 2 / v - 0.5 * np.log(2.0 * np.pi * v), axis=-1)


def mixture_moments(w, m, v):
    mean = float(np.sum(w * m))
    return mean, float(np.sum(w * (v + m**2)) - mean**2)


def avg_force(mean, second, x=X):
    """``E[xi]`` for ``xi = log N(x; z, 1)`` under a belief with the given first two moments."""
    return -0.5 * (second - 2.0 * x * mean + x**2) - 0.5 * LOG2PI


PRIOR_MEAN = float(np.sum(0.5 * MU))
PRIOR_SECOND_G = float(np.sum(0.5 * (MU**2 + SD**2)))
PRIOR_SECOND_L = float(np.sum(0.5 * (MU**2 + 2.0 * B**2)))


# ---------------------------------------------------------------------------
# Grids, the force, and the two boxes.
# ---------------------------------------------------------------------------


def grid(a, n):
    return np.linspace(-a, a, n)


def grid_h(a, h):
    return grid(a, int(round(2.0 * a / h)) + 1)


def bell_nll_normed(z):
    """``nll(x)_i = -log N(x; z_i, 1)``, the constant kept, so that areas and evidences compare."""
    zz = jnp.asarray(z, dtype=float)
    return lambda x: 0.5 * (x[0] - zz) ** 2 + 0.5 * LOG2PI


def force(z, x=X):
    return -0.5 * (x - np.asarray(z)) ** 2 - 0.5 * LOG2PI


def dense_step(z, l0, x=X):
    """One step of stage (a)'s box: ``Q`` the log-heights, response ``diag(1 / q)``."""
    n = len(z)
    arr = updating(log_height_response(n), bell_nll_normed(z), n, 1, jnp.asarray(l0))
    return np.asarray(log_heights(run(arr, [x])))


def beliefs_steps(z, l0, readings, tau=1.0):
    """Steps of the box whose configuration is the belief (``updating_on_beliefs``)."""
    n = len(z)
    arr = updating_on_beliefs(bell_nll_normed(z), n, 1, jnp.asarray(l0), tau)
    return np.asarray(log_heights(run(arr, list(readings))))


def moments(z, l):
    y = np.asarray(belief_of(jnp.asarray(l)))
    mean = float(np.sum(y * z))
    return mean, float(np.sum(y * (z - mean) ** 2))


def spread(d):
    d = np.asarray(d)
    return float(np.max(d) - np.min(d))


def log_area(z, l):
    return float(logsumexp(l) + np.log(z[1] - z[0]))


# ---------------------------------------------------------------------------
# Prediction 1: restriction is exact at every grid size.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [5, 9, 17, 33, 65, 129, 257, 513, 1025])
def test_prediction_1_restriction_is_exact_at_every_grid_size(n):
    """Belief G, window 8. One step of stage (a)'s box from ``l_i = log q(z_i)``: the differences
    ``l'_i - log q'(z_i)`` have spread below ``1e-10`` at every ``n``, the five-point grid
    included, and the belief after equals ``q'(z_i) / sum_j q'(z_j)`` to relative error ``1e-10``."""
    z = grid(8.0, n)
    l1 = dense_step(z, log_q_G(z))
    w, m, v, _ = closed_form_G([X])
    lp = log_mixture(z, w, m, v)
    s = spread(l1 - lp)
    want = np.exp(lp - logsumexp(lp))
    rel = float(np.max(np.abs(np.asarray(belief_of(jnp.asarray(l1))) - want) / want))
    print(f"\nP1 n={n}: spread of l' - log q' = {s:.2e}; belief rel err = {rel:.2e}")
    assert s < 1e-10
    assert rel < 1e-10


# ---------------------------------------------------------------------------
# Prediction 2: the grid's numbers converge to the curve's, faster than any power.
# ---------------------------------------------------------------------------


def _p2_errors(h):
    z = grid_h(12.0, h)
    l0 = log_q_G(z)
    l1 = dense_step(z, l0)
    w, m, v, logev = closed_form_G([X])
    c_true = avg_force(PRIOR_MEAN, PRIOR_SECOND_G)
    mean_true, var_true = mixture_moments(w, m, v)
    c_grid = float(np.mean(force(z) - (l1 - l0)))
    mean, var = moments(z, l1)
    return {
        "constant": abs(c_grid - c_true),
        "mean": abs(mean - mean_true),
        "variance": abs(var - var_true),
        "log area": abs(log_area(z, l1) - (logev - c_true)),
    }


def test_prediction_2_the_grids_numbers_converge_to_the_curves():
    """Belief G, window 12, ``h = 2, 1, 0.5, 0.25, 0.125, 0.0625``. The constant the step
    subtracts, the posterior mean and variance, and ``log(h sum exp(l'))`` (tending to
    ``KL(q || q')``) differ from the closed forms by more than ``1e-4`` at ``h = 1``, less than
    ``1e-6`` at ``h = 0.5``, and less than ``1e-11`` from ``h = 0.25`` on."""
    w, m, v, logev = closed_form_G([X])
    c_true = avg_force(PRIOR_MEAN, PRIOR_SECOND_G)
    print(f"\nP2 closed forms: E_q xi = {c_true:.10f}, log Pr(x) = {logev:.10f}, "
          f"KL = {logev - c_true:.10f}, weights = {w}, means = {m}, variances = {v}, "
          f"posterior mean, variance = {mixture_moments(w, m, v)}")
    errs = {h: _p2_errors(h) for h in [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]}
    for h, e in errs.items():
        print(f"P2 h={h}: " + ", ".join(f"{k} {val:.2e}" for k, val in e.items()))
    assert max(errs[1.0].values()) > 1e-4
    assert max(errs[0.5].values()) < 1e-6
    for h in [0.25, 0.125, 0.0625]:
        assert max(errs[h].values()) < 1e-11


# ---------------------------------------------------------------------------
# Prediction 3: the response alone has no limit; the step does.
# ---------------------------------------------------------------------------


def test_prediction_3_the_response_alone_has_no_limit_and_the_step_does():
    """Window 8, ``n = 65, ..., 1025``. The smallest entry of ``diag(1 / q)`` times ``h`` tends to
    ``1 / max_z q(z)`` (within 1% at ``n = 1025`` of its value at ``n = 513``) and the largest
    exceeds ``1e25`` at every ``n``; the largest entry of the responsibility
    ``q (xi - <q, xi>)`` halves, within 5%, each time ``n - 1`` doubles; the step ``l' - l``
    equals ``xi - <q, xi>`` to ``1e-10`` and its largest entry is the same at every ``n`` to 1%."""
    rows = []
    for n in [65, 129, 257, 513, 1025]:
        z = grid(8.0, n)
        h = z[1] - z[0]
        l0 = log_q_G(z)
        xi = force(z)
        y = np.asarray(belief_of(jnp.asarray(l0)))
        sharp = np.diag(np.asarray(log_height_response(n).sharp_at(jnp.asarray(l0))))
        # the library's own pullback of xi along the readout
        _, vjp = jax.vjp(belief_of, jnp.asarray(l0))
        resp = np.asarray(vjp(jnp.asarray(xi))[0])
        c = float(np.sum(y * xi))
        assert np.max(np.abs(resp - y * (xi - c))) < 1e-12
        step = dense_step(z, l0) - l0
        rows.append((n, h, sharp.min() * h, sharp.max(), np.max(np.abs(resp)),
                     float(np.max(np.abs(step - (xi - c)))), float(np.max(np.abs(step)))))
        print(f"\nP3 n={n}: h*min(1/q) = {rows[-1][2]:.6f}, max(1/q) = {rows[-1][3]:.3e}, "
              f"max|responsibility| = {rows[-1][4]:.6e}, step err = {rows[-1][5]:.2e}, "
              f"max|step| = {rows[-1][6]:.6f}")
    qmax = float(np.exp(np.max(log_q_G(np.linspace(-8, 8, 200001)))))
    print(f"P3 1 / max q = {1.0 / qmax:.6f}")
    assert abs(rows[-1][2] / rows[-2][2] - 1.0) < 0.01
    assert abs(rows[-1][2] * qmax - 1.0) < 0.01
    for (_, _, _, big, r, err, s), prev in zip(rows[1:], rows[:-1]):
        assert big > 1e25 and prev[3] > 1e25
        assert abs(prev[4] / r - 2.0) < 0.1
        assert err < 1e-10
        assert abs(s / prev[6] - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Prediction 4: a thin tail kills the box that divides; the box on beliefs holds.
# ---------------------------------------------------------------------------


def test_prediction_4_the_dividing_box_is_exact_while_every_height_is_a_float():
    """Window 24, ``h = 0.125``: every height is above float64's floor and stage (a)'s box is
    exact (spread below ``1e-9``)."""
    z = grid_h(24.0, 0.125)
    l0 = log_q_G(z)
    y = np.asarray(belief_of(jnp.asarray(l0)))
    w, m, v, _ = closed_form_G([X])
    s = spread(dense_step(z, l0) - log_mixture(z, w, m, v))
    print(f"\nP4 a=24: smallest height = {y.min():.3e}; dense box spread = {s:.2e}")
    assert y.min() > 0.0
    assert s < 1e-9


def test_prediction_4_a_thin_tail_kills_the_box_that_divides():
    """Window 40: heights underflow and stage (a)'s box returns a non-finite state."""
    z = grid_h(40.0, 0.125)
    l0 = log_q_G(z)
    y = np.asarray(belief_of(jnp.asarray(l0)))
    l1 = dense_step(z, l0)
    bad = int(np.sum(~np.isfinite(l1)))
    print(f"\nP4 a=40: heights held as 0: {int(np.sum(y == 0.0))} of {len(z)}; "
          f"non-finite log-heights after one step of the dividing box: {bad}; "
          f"first zero height at z = {z[y == 0.0][np.argmin(np.abs(z[y == 0.0]))]:.3f}")
    assert np.sum(y == 0.0) > 0
    assert bad > 0


def test_prediction_4_the_box_on_beliefs_holds_on_the_same_grid():
    """``updating_on_beliefs`` on the window 40 gives finite log-heights with spread against
    ``log q'(z_i)`` below ``1e-9``; and from ``N(0, 1)`` with reading ``2`` it lands on
    ``-(z_i - 1)^2`` up to a constant, to ``1e-9``."""
    z = grid_h(40.0, 0.125)
    l1 = beliefs_steps(z, log_q_G(z), [X])
    w, m, v, _ = closed_form_G([X])
    s = spread(l1 - log_mixture(z, w, m, v))
    l1b = beliefs_steps(z, -0.5 * z**2, [2.0])
    sb = spread(l1b + (z - 1.0) ** 2)
    print(f"\nP4 a=40, box on beliefs: all finite = {bool(np.all(np.isfinite(l1)))}; "
          f"spread = {s:.2e}; bell curve N(0,1), reading 2: spread against -(z-1)^2 = {sb:.2e}; "
          f"mean, variance = {moments(z, l1b)}")
    assert np.all(np.isfinite(l1))
    assert s < 1e-9
    assert sb < 1e-9


def test_the_two_boxes_take_the_same_step_where_both_are_finite():
    """Not a numbered prediction: the derivation's 'mathematically the two boxes take the same
    step', on the window 8."""
    z = grid(8.0, 257)
    l0 = log_q_G(z)
    d = float(np.max(np.abs(dense_step(z, l0) - beliefs_steps(z, l0, [X]))))
    print(f"\nsame step, window 8, n=257: max abs difference = {d:.2e}")
    assert d < 1e-10


# ---------------------------------------------------------------------------
# Prediction 5: three tails, one noise, the window widening.
# ---------------------------------------------------------------------------

WINDOWS = [10.0, 20.0, 40.0, 80.0, 160.0, 320.0]


def _p5(log_q):
    out = []
    for a in WINDOWS:
        z = grid_h(a, 0.05)
        l0 = log_q(z)
        l1 = beliefs_steps(z, l0, [X])
        core = np.abs(z) <= 10.0 + 1e-9
        c = float(np.mean((force(z) - (l1 - l0))[core]))
        d = l1[core]
        out.append((a, c, d - d[len(d) // 2], moments(z, l1)))
    return out


@pytest.mark.parametrize("name", ["G", "L", "C"])
def test_prediction_5_the_belief_converges_in_all_three(name):
    """The differences ``l'_i - l'_j`` over the grid points in ``[-10, 10]`` are the same at every
    window to ``1e-9``, and the posterior mean and variance at ``a >= 20`` agree with those at
    ``a = 10`` to ``1e-9``."""
    rows = _p5({"G": log_q_G, "L": log_q_L, "C": log_q_C}[name])
    base = rows[0]
    for a, c, d, (mean, var) in rows:
        dd = float(np.max(np.abs(d - base[2])))
        print(f"\nP5 {name} a={a:g}: constant <q, xi> = {c:.8f}; log-height differences vs a=10: "
              f"{dd:.2e}; mean = {mean:.12f}, variance = {var:.12f}")
        assert dd < 1e-9
        assert abs(mean - base[3][0]) < 1e-9 and abs(var - base[3][1]) < 1e-9


def test_prediction_5_the_constant_converges_for_G_and_L():
    """To the closed form ``-(E z^2 - 2x E z + x^2)/2 - log(2 pi)/2`` (the same number for G and L
    by the choice of ``b``): within ``1e-6`` at every window for G; within ``1e-3`` at
    ``a >= 20`` for L."""
    c_true = avg_force(PRIOR_MEAN, PRIOR_SECOND_G)
    assert abs(PRIOR_SECOND_G - PRIOR_SECOND_L) < 1e-12
    for a, c, _, _ in _p5(log_q_G):
        print(f"\nP5 G a={a:g}: constant error = {abs(c - c_true):.2e}")
        assert abs(c - c_true) < 1e-6
    for a, c, _, _ in _p5(log_q_L):
        print(f"P5 L a={a:g}: constant error = {abs(c - c_true):.2e}")
        if a >= 20.0:
            assert abs(c - c_true) < 1e-3


def test_prediction_5_the_constant_diverges_for_C():
    """Linearly: ``(c(2a) - c(a)) / a`` tends to ``-gamma / pi = -0.1592``, within 2% for
    ``a = 80 -> 160`` and ``160 -> 320``."""
    rows = _p5(log_q_C)
    want = -GAMMA / np.pi
    for (a0, c0, _, _), (a1, c1, _, _) in zip(rows[:-1], rows[1:]):
        slope = (c1 - c0) / a0
        print(f"\nP5 C a={a0:g}->{a1:g}: c = {c0:.6f} -> {c1:.6f}; slope = {slope:.6f} "
              f"(-gamma/pi = {want:.6f})")
        if a0 >= 80.0:
            assert abs(slope / want - 1.0) < 0.02


# ---------------------------------------------------------------------------
# Prediction 6: H1 without H2, a force that flattens.
# ---------------------------------------------------------------------------


def test_prediction_6_a_force_that_flattens():
    """Belief C, force ``xi = -log q``. The belief after is uniform on the window at every window
    (largest height over smallest within ``1e-9`` of 1), its variance is ``a^2 / 3`` to 1%; the
    constant ``<q, xi>`` converges to the entropy of C, within ``0.05`` at ``a = 320`` and closer
    at each doubling."""
    f = lambda z: float(-np.exp(log_q_C(z)) * log_q_C(z))
    entropy = sum(integrate.quad(f, lo, hi, limit=400, epsabs=1e-13, epsrel=1e-13)[0]
                  for lo, hi in [(-np.inf, -1.5), (-1.5, 2.5), (2.5, np.inf)])
    print(f"\nP6 entropy of C by quadrature = {entropy:.8f}")
    gaps = []
    for a in WINDOWS:
        z = grid_h(a, 0.05)
        n = len(z)
        l0 = log_q_C(z)
        arr = updating_on_beliefs(column_nll, n, n, jnp.asarray(l0))
        l1 = np.asarray(log_heights(run(arr, [l0])))     # nll = log q, so xi = -log q
        y = np.asarray(belief_of(jnp.asarray(l1)))
        c = float(np.mean(-l0 - (l1 - l0)))
        _, var = moments(z, l1)
        gaps.append(abs(c - entropy))
        print(f"P6 a={a:g}: max/min height - 1 = {y.max() / y.min() - 1.0:.2e}; "
              f"variance / (a^2/3) = {var / (a * a / 3.0):.6f}; constant = {c:.6f}, "
              f"gap to entropy = {gaps[-1]:.5f}")
        assert abs(y.max() / y.min() - 1.0) < 1e-9
        assert abs(var / (a * a / 3.0) - 1.0) < 0.01
    assert gaps[-1] < 0.05
    assert all(g1 < g0 for g0, g1 in zip(gaps[:-1], gaps[1:]))


# ---------------------------------------------------------------------------
# Prediction 7: duration.
# ---------------------------------------------------------------------------


def test_prediction_7_one_step_leaves_area_exp_KL():
    """Belief G, window 12, ``h = 0.125``, from ``l_i = log q(z_i)``: ``log(h sum exp(l'))``
    equals ``KL(q || q')`` to ``1e-9``; two half steps equal one whole step in belief."""
    z = grid_h(12.0, 0.125)
    l0 = log_q_G(z)
    _, _, _, logev = closed_form_G([X])
    kl = logev - avg_force(PRIOR_MEAN, PRIOR_SECOND_G)
    l1 = beliefs_steps(z, l0, [X])
    lh = beliefs_steps(z, l0, [X, X], tau=0.5)
    y1, yh = belief_of(jnp.asarray(l1)), belief_of(jnp.asarray(lh))
    rel = float(np.max(np.abs(np.asarray(yh) - np.asarray(y1)) / np.asarray(y1)))
    print(f"\nP7 one step: log area = {log_area(z, l1):.12f}, KL = {kl:.12f}; "
          f"two half steps vs one whole, belief rel err = {rel:.2e}; "
          f"log area after two half steps = {log_area(z, lh):.12f}")
    assert abs(log_area(z, l1) - kl) < 1e-9
    assert rel < 1e-10


def test_prediction_7_many_small_steps_leave_area_one():
    """``N`` steps of duration ``1/N``, ``N = 1, 2, 4, ..., 1024``: the belief after is the same at
    every ``N`` (to ``1e-10`` relative); ``log(h sum exp(l))`` after is positive, falls by a factor
    between ``1.8`` and ``2.2`` at each doubling from ``N = 16`` on, and at ``N = 1024`` is within
    10% of ``(E_{q'} xi - E_q xi) / 2048``."""
    z = grid_h(12.0, 0.125)
    l0 = log_q_G(z)
    w, m, v, _ = closed_form_G([X])
    want = np.exp(log_mixture(z, w, m, v) - logsumexp(log_mixture(z, w, m, v)))
    mean1, var1 = mixture_moments(w, m, v)
    lead = (avg_force(mean1, var1 + mean1**2) - avg_force(PRIOR_MEAN, PRIOR_SECOND_G)) / 2.0
    print(f"\nP7 (E_q' xi - E_q xi) / 2 = {lead:.8f}")
    las = {}
    for N in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        lN = beliefs_steps(z, l0, [X] * N, tau=1.0 / N)
        rel = float(np.max(np.abs(np.asarray(belief_of(jnp.asarray(lN))) - want) / want))
        las[N] = log_area(z, lN)
        factor = las[N // 2] / las[N] if N > 1 else float("nan")
        print(f"P7 N={N}: log area = {las[N]:.10f}; N * log area = {N * las[N]:.8f}; "
              f"factor from N/2 = {factor:.5f}; belief rel err = {rel:.2e}")
        assert rel < 1e-10
        assert las[N] > 0.0
        if N >= 32:
            assert 1.8 < factor < 2.2
    assert abs(1024 * las[1024] / lead - 1.0) < 0.10


# ---------------------------------------------------------------------------
# Prediction 8: a stream on the wide window.
# ---------------------------------------------------------------------------


def test_prediction_8_a_stream_on_the_wide_window():
    """Belief G, window 40, ``h = 0.125``, 30 readings about ``2.0`` (``RandomState(7)``), the box
    on beliefs: mean and variance equal the closed-form mixture posterior's to ``1e-9``, the
    log-heights have spread against the closed form's log-density below ``1e-8``, and the same in
    reversed order to ``1e-9``."""
    readings = list(2.0 + np.random.RandomState(7).randn(30))
    z = grid_h(40.0, 0.125)
    l0 = log_q_G(z)
    w, m, v, _ = closed_form_G(readings)
    mean_true, var_true = mixture_moments(w, m, v)
    l1 = beliefs_steps(z, l0, readings)
    l2 = beliefs_steps(z, l0, readings[::-1])
    mean, var = moments(z, l1)
    s = spread(l1 - log_mixture(z, w, m, v))
    rev = float(np.max(np.abs((l1 - l1[len(z) // 2]) - (l2 - l2[len(z) // 2]))))
    print(f"\nP8: closed form weights = {w}, mean = {mean_true:.12f}, variance = {var_true:.12f}; "
          f"box mean err = {abs(mean - mean_true):.2e}, variance err = {abs(var - var_true):.2e}; "
          f"spread = {s:.2e}; reversed order, log-height differences = {rev:.2e}; "
          f"smallest log-height = {l1.min():.1f}")
    assert np.all(np.isfinite(l1))
    assert abs(mean - mean_true) < 1e-9 and abs(var - var_true) < 1e-9
    assert s < 1e-8
    assert rev < 1e-9


# ---------------------------------------------------------------------------
# Prediction 9 (added after 1-8 had run, before this test existed): "heavier", held on a curve.
# ---------------------------------------------------------------------------


def test_prediction_9_heavier_held_on_a_curve():
    """Belief ``N(0, 1)``; only the sign of a unit-noise reading is told, ``Pr(heavier | z) =
    Phi(z)``. The log-heights after one step equal ``-z^2/2 + log Phi(z)`` up to a constant (spread
    below ``1e-10``); mean ``1 / sqrt(pi)`` and variance ``1 - 1/pi`` to ``1e-9``; the constant
    agrees with quadrature to ``1e-9``; the new log-height's best quadratic fit over ``[-3, 3]``
    misses by more than ``0.01`` where the prior's misses by less than ``1e-12``; a second
    "heavier" gives ``phi Phi^2`` up to scale."""
    z = grid_h(12.0, 0.05)
    n = len(z)
    l0 = -0.5 * z**2
    nll = -log_ndtr(z)                                    # the column -log Pr(heavier | z)
    arr = updating_on_beliefs(column_nll, n, n, jnp.asarray(l0))
    l1 = np.asarray(log_heights(run(arr, [nll])))
    l2 = np.asarray(log_heights(run(arr, [nll, nll])))
    s1 = spread(l1 - (l0 + log_ndtr(z)))
    s2 = spread(l2 - (l0 + 2.0 * log_ndtr(z)))
    mean, var = moments(z, l1)
    c = float(np.mean(log_ndtr(z) - (l1 - l0)))
    c_true = integrate.quad(lambda t: float(np.exp(-0.5 * t * t) / np.sqrt(2 * np.pi) * log_ndtr(t)),
                            -np.inf, np.inf, epsabs=1e-13, epsrel=1e-13)[0]
    core = np.abs(z) <= 3.0 + 1e-9

    def misfit(l):
        coef = np.polyfit(z[core], l[core], 2)
        return float(np.max(np.abs(np.polyval(coef, z[core]) - l[core])))

    print(f"\nP9: spread = {s1:.2e}; mean = {mean:.10f} (1/sqrt(pi) = {1 / np.sqrt(np.pi):.10f}); "
          f"variance = {var:.10f} (1 - 1/pi = {1 - 1 / np.pi:.10f}); constant = {c:.10f} "
          f"(quadrature {c_true:.10f}); quadratic misfit on [-3, 3]: prior {misfit(l0):.2e}, "
          f"after {misfit(l1):.4f}; second 'heavier' spread = {s2:.2e}")
    assert s1 < 1e-10 and s2 < 1e-10
    assert abs(mean - 1 / np.sqrt(np.pi)) < 1e-9
    assert abs(var - (1 - 1 / np.pi)) < 1e-9
    assert abs(c - c_true) < 1e-9
    assert misfit(l0) < 1e-12 and misfit(l1) > 0.01
