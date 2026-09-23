"""Replicator dynamics as Phi of the Shahshahani configuration arrangement (replicator.py).

Four checks:
* one Phi step from a simplex point IS the forward-Euler replicator step
  ``x + dt * (x_i ((A x)_i - x^T A x))_i``, to machine precision (keystone);
* the Shahshahani sharp is symmetric with the constants in its kernel exactly
  on the simplex, and ``sum(x) = 1`` is preserved to rounding level (measured
  max drift ~3e-16 over 5000 steps, non-accumulating -- the drift is
  self-correcting, see the module docstring);
* convergence to the ESS: a diagonal coordination game reaches the pure
  strategy of the largest payoff from the barycenter, and (column-symmetrized)
  Hawk-Dove reaches the mixed ESS ``x_Hawk = v/c`` from interior starts.

Non-symmetric games (rock-paper-scissors) are OUT OF SCOPE of the TRICK
realization by construction -- they are not potential games -- and its
constructor rejects them; we test the rejection, not the dynamics.

LOG COORDINATES (module docstring): one Phi step of the log arrangement
is exactly the multiplicative-weights update ``x * exp(dt * ((A x)_i -
x^T A x))``; fractions stay positive at a ``dt`` where the standard tick
overshoots a coordinate below 0; the sum, exactly invariant for the standard
tick, drifts and the drift shrinks with ``dt``; the two ticks converge to
each other as ``dt -> 0``.

TRICK versus DIRECT: the book's arrangement carries a forcing that need not be
exact, so the replicator field is also realized directly, as a hand-written
non-exact ``omega`` with the plain ``configuration(euclidean(n))`` parameter
(``direct_replicator_arrangement``). We check that it agrees with the trick
step for step for symmetric ``A``, that its ``omega`` is genuinely not closed,
and that it runs rock-paper-scissors, which the trick refuses.

The system is closed (no environment covector), so the CDLM regression
(``regress/replicator.npz``) needs no sign translation: states are identical.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arrangements.replicator import (
    direct_replicator_arrangement,
    direct_replicator_dynamics,
    log_replicator_dynamics,
    log_shahshahani,
    replicator_arrangement,
    replicator_dynamics,
    replicator_step,
    shahshahani,
)

from tests.chain import load_regress


def _random_symmetric(rng, n):
    B = rng.standard_normal((n, n))
    return jnp.asarray((B + B.T) / 2.0)


def _random_simplex(rng, n):
    return jnp.asarray(rng.dirichlet(np.ones(n)))


def test_one_step_is_euler_replicator():
    """Keystone: Phi step = x + dt * (x_i((Ax)_i - x^T A x))_i, machine precision."""
    rng = np.random.default_rng(0)
    for n in (2, 3, 5):
        A = _random_symmetric(rng, n)
        for dt in (1.0, 0.05):
            O = replicator_dynamics(A, dt)
            for _ in range(3):
                x = _random_simplex(rng, n)
                new_x = replicator_step(O, x)
                Ax = A @ x
                exact = x + dt * (x * (Ax - jnp.dot(x, Ax)))
                np.testing.assert_allclose(
                    np.asarray(new_x), np.asarray(exact), rtol=0.0, atol=1e-14
                )


def test_sharp_kernel_and_simplex_invariance():
    """sharpR_x is symmetric; ker contains constants iff sum(x)=1; sum drifts only at rounding."""
    rng = np.random.default_rng(1)
    n, dt = 3, 0.05
    Q = shahshahani(n, dt)

    # Symmetric, and constants in the kernel exactly on the simplex:
    # sharpR_x(1) = dt * x * (1 - sum x).
    x_on = _random_simplex(rng, n)
    S = Q.sharp_at(x_on)
    np.testing.assert_allclose(np.asarray(S), np.asarray(S.T), atol=0.0)
    np.testing.assert_allclose(np.asarray(S @ jnp.ones(n)), np.zeros(n), atol=1e-15)
    x_off = 1.7 * x_on  # sum = 1.7: constants NOT in the kernel off the simplex
    S_off = Q.sharp_at(x_off)
    np.testing.assert_allclose(
        np.asarray(S_off @ jnp.ones(n)), np.asarray(dt * x_off * (1.0 - 1.7)), atol=1e-14
    )
    assert float(jnp.max(jnp.abs(S_off @ jnp.ones(n)))) > 1e-3

    # Invariance of sum(x) = 1 along the Euler orbit. Not exact in floats: the
    # per-step increment is dt * (x^T A x) * (1 - sum x) plus rounding, so the
    # drift stays at rounding level and does not accumulate (measured max
    # ~3e-16 over 5000 steps); we assert the rounding-level bound, not exactness.
    A = _random_symmetric(rng, n) + 2.0 * jnp.eye(n) + 1.0  # positive on the orthant
    O = replicator_dynamics(A, dt)
    x = _random_simplex(rng, n)
    max_drift = 0.0
    for _ in range(2000):
        x = replicator_step(O, x)
        max_drift = max(max_drift, abs(float(jnp.sum(x)) - 1.0))
    assert max_drift < 1e-13


def test_coordination_game_converges_to_dominant_pure_strategy():
    """A = diag(3,2,1): from the barycenter the largest-payoff pure strategy wins."""
    A = jnp.diag(jnp.array([3.0, 2.0, 1.0]))
    O = replicator_dynamics(A, dt=0.1)
    x = jnp.ones(3) / 3.0
    fitness = [float(jnp.dot(x, A @ x))]
    for _ in range(2000):
        x = replicator_step(O, x)
        fitness.append(float(jnp.dot(x, A @ x)))
    np.testing.assert_allclose(np.asarray(x), np.array([1.0, 0.0, 0.0]), atol=1e-8)
    # ascent of U = mean fitness (omega = dU), monotone in this run
    assert all(f2 >= f1 - 1e-12 for f1, f2 in zip(fitness, fitness[1:]))


def test_hawk_dove_converges_to_mixed_ess():
    """Column-symmetrized Hawk-Dove (v=2, c=3) reaches the mixed ESS x_H = v/c = 2/3.

    The textbook matrix [[(v-c)/2, v], [0, v/2]] is not symmetric, but replicator
    dynamics is invariant under column shifts A -> A + 1 c^T; the shift c = (v, 0)
    gives the symmetric representative [[(v-c)/2 + v, v], [v, v/2]] with the same
    dynamics, and THAT is the partnership game fed to the construction.
    """
    v, c = 2.0, 3.0
    A = jnp.array([[(v - c) / 2.0 + v, v], [v, v / 2.0]])
    O = replicator_dynamics(A, dt=0.1)
    for x0 in (jnp.array([0.15, 0.85]), jnp.array([0.9, 0.1])):
        x = x0
        for _ in range(3000):
            x = replicator_step(O, x)
        np.testing.assert_allclose(
            np.asarray(x), np.array([v / c, 1.0 - v / c]), atol=1e-10
        )


def test_rock_paper_scissors_is_rejected():
    """RPS is not a partnership game; the trick constructor refuses it (scope limit)."""
    rps = jnp.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
    with pytest.raises(ValueError):
        replicator_arrangement(rps, dt=0.1)


# ---------------------------------------------------------------------------
# Log coordinates (module docstring): the multiplicative-weights reading.
# ---------------------------------------------------------------------------


def _excess(A, x):
    return A @ x - jnp.dot(x, A @ x)


def test_log_one_step_is_multiplicative_weights():
    """Keystone: one Phi step of the log arrangement, read back through
    x = e^y, is exactly x * exp(dt * ((A x)_i - x^T A x))."""
    rng = np.random.default_rng(11)
    for dt in (1.0, 0.05):
        for _ in range(5):
            A = _random_symmetric(rng, 4)
            x = _random_simplex(rng, 4)
            O = log_replicator_dynamics(A, dt)
            y_new = replicator_step(O, jnp.log(x))
            np.testing.assert_allclose(
                np.asarray(jnp.exp(y_new)),
                np.asarray(x * jnp.exp(dt * _excess(A, x))),
                rtol=1e-12,
            )


def test_log_positive_where_standard_tick_overshoots():
    """Coordination game A = diag(1,1), x = (0.9, 0.1), dt = 2: the standard
    tick sends x_2 below 0; the log tick keeps every fraction positive."""
    A = jnp.eye(2)
    x = jnp.array([0.9, 0.1])
    dt = 2.0
    x_std = replicator_step(replicator_dynamics(A, dt), x)
    assert bool(jnp.any(x_std < 0))
    y = replicator_step(log_replicator_dynamics(A, dt), jnp.log(x))
    assert bool(jnp.all(jnp.exp(y) > 0)) and bool(jnp.all(jnp.isfinite(y)))


def test_log_sum_drifts_and_shrinks_with_dt():
    """The trade: the standard tick preserves sum(x) = 1 exactly on the
    simplex; the log tick's sum drifts, and the drift shrinks with dt over a
    fixed horizon."""
    rng = np.random.default_rng(12)
    A = _random_symmetric(rng, 3)
    x0 = _random_simplex(rng, 3)
    horizon = 1.0
    drift = {}
    for dt in (0.2, 0.02):
        O = log_replicator_dynamics(A, dt)
        y = jnp.log(x0)
        for _ in range(int(round(horizon / dt))):
            y = replicator_step(O, y)
        drift[dt] = abs(float(jnp.sum(jnp.exp(y))) - 1.0)
    assert drift[0.2] > 1e-6
    assert drift[0.02] < drift[0.2]


def test_log_and_standard_ticks_converge_together():
    """As dt -> 0 the two readings track the same replicator flow: over a
    fixed horizon their endpoints approach each other at first order."""
    rng = np.random.default_rng(13)
    A = _random_symmetric(rng, 3)
    x0 = _random_simplex(rng, 3)
    horizon = 0.5
    gap = {}
    for dt in (0.05, 0.01):
        Os = replicator_dynamics(A, dt)
        Ol = log_replicator_dynamics(A, dt)
        xs, y = x0, jnp.log(x0)
        for _ in range(int(round(horizon / dt))):
            xs, y = replicator_step(Os, xs), replicator_step(Ol, y)
        gap[dt] = float(jnp.max(jnp.abs(xs - jnp.exp(y))))
    assert gap[0.01] < gap[0.05]
    assert gap[0.01] < 0.02


def test_log_sharp_is_pushforward_and_psd_on_simplex():
    """sharpR^log_y = J sharpR_x J^T with J = diag(1/x): the same geometry in
    the new chart; and it is symmetric PSD at simplex points."""
    rng = np.random.default_rng(14)
    for _ in range(5):
        x = _random_simplex(rng, 4)
        J = jnp.diag(1.0 / x)
        S_std = shahshahani(4, dt=0.3).sharp_at(x)
        S_log = log_shahshahani(4, dt=0.3).sharp_at(jnp.log(x))
        np.testing.assert_allclose(np.asarray(S_log), np.asarray(J @ S_std @ J), atol=1e-10)
        np.testing.assert_allclose(np.asarray(S_log), np.asarray(S_log).T, atol=1e-12)
        eigs = np.linalg.eigvalsh(np.asarray(S_log))
        assert eigs.min() > -1e-10


# ---------------------------------------------------------------------------
# Trick versus direct: the replicator field as a non-exact forcing.
# ---------------------------------------------------------------------------


def _orbit(O, x, T):
    traj = [x]
    for _ in range(T):
        x = replicator_step(O, x)
        traj.append(x)
    return np.stack([np.asarray(v) for v in traj])


def test_direct_one_form_realization_agrees_with_trick():
    """The hand-written forcing on ``configuration(euclidean(n))`` (direct) and the
    Shahshahani sharp with ``d((1/2) x^T A x)`` (trick) give the same orbit, step
    for step, for symmetric ``A``; and the direct ``omega`` is not closed."""
    rng = np.random.default_rng(21)
    for n, dt, T in ((2, 0.1, 200), (3, 0.05, 300), (5, 0.02, 100)):
        A = _random_symmetric(rng, n)
        x0 = _random_simplex(rng, n)
        a = _orbit(replicator_dynamics(A, dt), x0, T)
        b = _orbit(direct_replicator_dynamics(A, dt), x0, T)
        np.testing.assert_allclose(b, a, rtol=1e-12, atol=1e-14)

        # not exact: the Jacobian of omega|_x is not symmetric (so omega is not dU)
        arr = direct_replicator_arrangement(A, dt)
        z = jnp.zeros(0)
        Jw = np.asarray(jax.jacfwd(lambda q: arr.omega(q, z, z)[0])(x0))
        assert np.max(np.abs(Jw - Jw.T)) > 1e-3 * np.max(np.abs(Jw))


def test_direct_realization_runs_rock_paper_scissors():
    """Where the trick stops (scope limit), the direct realization proceeds: RPS is
    accepted, one step is the Euler replicator step, the barycenter is fixed, and
    the sum stays 1 (the field is tangent to the simplex)."""
    rps = jnp.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
    dt = 0.05
    O = direct_replicator_dynamics(rps, dt)
    rng = np.random.default_rng(22)
    for _ in range(3):
        x = _random_simplex(rng, 3)
        new_x = replicator_step(O, x)
        Ax = rps @ x
        np.testing.assert_allclose(
            np.asarray(new_x), np.asarray(x + dt * (x * (Ax - jnp.dot(x, Ax)))), rtol=0.0, atol=1e-14
        )
    bary = jnp.ones(3) / 3.0
    np.testing.assert_allclose(np.asarray(replicator_step(O, bary)), np.asarray(bary), atol=1e-15)
    traj = _orbit(O, jnp.array([0.5, 0.3, 0.2]), 500)
    np.testing.assert_allclose(traj.sum(axis=1), np.ones(traj.shape[0]), atol=1e-13)
    assert np.all(traj > 0)


# ---------------------------------------------------------------------------
# CDLM regression.
# ---------------------------------------------------------------------------


def test_regression_matches_cdlm():
    """The CDLM states (``regress/replicator.npz``) are reproduced to rtol 1e-10 (STATUS.md §1):
    one-step (standard and log charts), the simplex orbit, coordination, Hawk-Dove, and the
    standard/log convergence run. Closed system: no sign translation."""
    R = load_regress("replicator")
    tol = dict(rtol=1e-10, atol=1e-12)

    for n in (2, 3, 5):
        A = jnp.asarray(R[f"step_A_{n}"])
        for dt in (1.0, 0.05):
            O = replicator_dynamics(A, dt)
            tag = f"{n}_{dt:g}"
            new = np.stack([np.asarray(replicator_step(O, jnp.asarray(x))) for x in R[f"step_x_{tag}"]])
            np.testing.assert_allclose(new, R[f"step_new_{tag}"], **tol)

    inv = _orbit(replicator_dynamics(jnp.asarray(R["inv_A"]), float(R["inv_dt"])),
                 jnp.asarray(R["inv_x0"]), R["inv_traj"].shape[0] - 1)
    np.testing.assert_allclose(inv, R["inv_traj"], **tol)

    coord = _orbit(replicator_dynamics(jnp.asarray(R["coord_A"]), dt=0.1), jnp.ones(3) / 3.0,
                   R["coord_traj"].shape[0] - 1)
    np.testing.assert_allclose(coord, R["coord_traj"], **tol)

    O = replicator_dynamics(jnp.asarray(R["hd_A"]), dt=0.1)
    for k, x0 in enumerate((jnp.array([0.15, 0.85]), jnp.array([0.9, 0.1]))):
        hd = _orbit(O, x0, R[f"hd_traj_{k}"].shape[0] - 1)
        np.testing.assert_allclose(hd, R[f"hd_traj_{k}"], **tol)

    for A, x, dt, y_old in zip(R["log_A"], R["log_x"], R["log_dts"], R["log_ynew"]):
        y_new = replicator_step(log_replicator_dynamics(jnp.asarray(A), float(dt)), jnp.log(jnp.asarray(x)))
        np.testing.assert_allclose(np.asarray(y_new), y_old, **tol)

    A, x0 = jnp.asarray(R["conv_A"]), jnp.asarray(R["conv_x0"])
    for dt in (0.05, 0.01):
        T = R[f"conv_std_{dt:g}"].shape[0] - 1
        np.testing.assert_allclose(_orbit(replicator_dynamics(A, dt), x0, T), R[f"conv_std_{dt:g}"], **tol)
        np.testing.assert_allclose(_orbit(log_replicator_dynamics(A, dt), jnp.log(x0), T),
                                   R[f"conv_log_{dt:g}"], **tol)
