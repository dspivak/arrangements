"""Impedance (P-)control as a virtual spring (control.py) -- audit tests.

The claim under test: a stateless controller box whose forcing is
``omega_c = -d (k/2)(q_rep - s)^2``, wired to a zero-forcing plant, with the
setpoint ``s`` a genuine run-time input on the outer boundary, gives

* with the ``configuration`` parameter: geometric convergence of the plant
  state to ``s``;
* with the ``phase`` parameter: undamped oscillation about ``s`` at frequency
  ``~ sqrt(k/m)`` (the classical stiffness-only impedance controller).

One wiring, two arrangements (def:arr): the parameter -- the coalgebra inside
the arrangement -- is what differs, and there is one dynamics functor ``Phi``.

Timestep convention: the framework has no separate ``dt``; it lives in the
sharp (CDLM eqn.learning_sharp: ``euclidean(1, dt)`` makes one configuration
step ``q <- q + dt * (-k(q - s))``) and, for phase dynamics, one coalgebra step
is one unit of time with ``sharp = xi/m`` (as in test_wave_equation), so "small
dt" means small ``k/m``.

Signs (STATUS.md §1): the covector the coalgebra consumes is the forcing
``xi_Q = -k(q_eval - s)`` and the coalgebra *adds* it, where CDLM's was the cost
gradient ``+k(q_eval - s)`` and its integrator subtracted; the field published on
the setpoint port is likewise negated. States are identical.
"""

import jax.numpy as jnp
import numpy as np

from arrangements.coalgebra import configuration, phase
from arrangements.control import impedance_arrangement, impedance_flat, impedance_state, run_tracking
from arrangements.functors import Phi
from arrangements.rvect import diagonal, euclidean

from tests.chain import IN_POS, load_regress


def _one_step(O, state, s):
    """One coalgebra step with setpoint ``s`` and zero boundary covector."""
    in_dir = (jnp.zeros(1), jnp.array([float(s)]))
    return O.with_state(state).run_one(IN_POS, lambda _op: in_dir)


# ---------------------------------------------------------------------------
# 1. One-step exactness with the configuration parameter.
# ---------------------------------------------------------------------------


def test_conf_one_step_is_exact_virtual_spring_descent():
    """One configuration step equals ``q + dt * (-k (q - s))`` to numerical precision.

    The plant's forcing is zero (the honest simplest choice), so the *entire*
    parameter covector is the controller's ``xi_Q = -k(q - s)`` and the
    configuration coalgebra applies ``q <- q + sharpR_q(xi_Q)`` with
    ``sharpR = dt * I`` (sec.displacement).
    """
    dt, k = 0.07, 1.3
    arr = impedance_arrangement(euclidean(1, dt), k, parameter=configuration)
    O = Phi(arr)

    rng = np.random.default_rng(20260820)
    for _ in range(5):
        q0 = jnp.array([rng.standard_normal()])
        s = float(rng.standard_normal())
        out_pos, _out_dir, new_state = _one_step(O, impedance_state(q0), s)
        expected = q0 + dt * (-k * (q0 - s))
        np.testing.assert_allclose(
            np.asarray(impedance_flat(new_state)), np.asarray(expected), atol=1e-12)
        # the configuration readout is the current parameter position q itself
        np.testing.assert_allclose(np.asarray(out_pos[0]), np.asarray(q0), atol=1e-12)


# ---------------------------------------------------------------------------
# 2. Step response: convergence, and a mid-run setpoint change.
# ---------------------------------------------------------------------------


def test_conf_step_response_tracks_runtime_setpoint():
    """Constant setpoint => exact geometric convergence, ratio (1 - dt k) per
    step; changing the setpoint mid-run (same coalgebra, same threaded state,
    only the per-step input changes) re-converges to the new value."""
    dt, k = 0.05, 2.0          # contraction factor 1 - dt k = 0.9
    rho = 1.0 - dt * k
    s1, s2, T = 1.5, -0.7, 120
    arr = impedance_arrangement(euclidean(1, dt), k, parameter=configuration)
    O = Phi(arr)

    setpoints = [s1] * T + [s2] * T
    _, states = run_tracking(O, impedance_state(jnp.array([0.0])), setpoints)
    q = np.array([float(impedance_flat(st)[0]) for st in states])  # q_0 .. q_{2T}

    # phase 1: exact geometric contraction toward s1 at every step
    e1 = q[: T + 1] - s1
    np.testing.assert_allclose(e1[1:], rho * e1[:-1], atol=1e-12)
    assert abs(q[T] - s1) < 1e-4          # |e| = 1.5 * 0.9^120 ~ 4.7e-6

    # phase 2: the SAME run re-converges to the new setpoint s2
    e2 = q[T: 2 * T + 1] - s2
    np.testing.assert_allclose(e2[1:], rho * e2[:-1], atol=1e-12)
    assert abs(q[2 * T] - s2) < 1e-4


# ---------------------------------------------------------------------------
# 3. The phase parameter: undamped oscillation about the setpoint at ~ sqrt(k/m).
# ---------------------------------------------------------------------------


def test_phase_oscillates_about_setpoint_at_sqrt_k_over_m():
    """Same construction, phase coalgebra: semi-implicit Euler on the
    harmonic oscillator ``m qdd = -k(q - s)``.

    One step is ``q <- q + p/m``, ``p <- p - k(q~ - s)`` (forces at the
    presented position ``q~``), whose exact rotation angle per step is
    ``theta = arccos(1 - k/(2m))``; ``k_eff = k`` (the only stiffness in the
    construction is the controller's), so for small ``k/m`` the period is
    ``~ 2 pi / sqrt(k/m)`` steps. Tolerances: measured period vs the exact
    discrete angle to 0.5% (zero-crossing interpolation error over ~38
    averaged periods is far below this); vs the continuum ``sqrt(k/m)`` to 1%
    (the discretization offset is ``theta/sqrt(k/m) - 1 ~ k/(24 m) ~ 0.17%``).
    Amplitude bounds +-15%: symplectic Euler preserves a modified quadratic
    invariant, so the amplitude wobbles by O(theta) ~ 20% at most per phase of
    the modified ellipse but has no secular drift; we check both no decay and
    no growth between the first and last quarters of the run.
    """
    m, k = 1.0, 0.04           # k_eff/m = 0.04, theta ~ 0.2 rad/step
    s_star, A = 0.5, 1.0
    T = 1200                   # ~38 periods
    arr = impedance_arrangement(diagonal(jnp.array([1.0 / m])), k, parameter=phase)
    O = Phi(arr)

    q0 = jnp.array([s_star + A])
    p0 = jnp.array([0.0])
    outs, states = run_tracking(O, impedance_state((q0, p0)), [s_star] * T)

    x = np.array([float(o[0]) for o in outs]) - s_star  # presented positions
    q = np.array([float(impedance_flat(st)[0][0]) for st in states]) - s_star  # true positions

    # -- period from zero crossings of the reported (presented) position --
    idx = np.nonzero(x[:-1] * x[1:] < 0)[0]
    assert len(idx) >= 60      # it genuinely oscillates
    t_cross = idx + x[idx] / (x[idx] - x[idx + 1])  # linear interpolation
    period = 2.0 * (t_cross[-1] - t_cross[0]) / (len(t_cross) - 1)

    theta_exact = np.arccos(1.0 - k / (2.0 * m))
    assert abs(period - 2.0 * np.pi / theta_exact) / period < 0.005
    assert abs(period - 2.0 * np.pi / np.sqrt(k / m)) / period < 0.01

    # -- no convergence, no divergence: amplitude neither decays nor grows --
    first, last = np.max(np.abs(q[: T // 4])), np.max(np.abs(q[-T // 4:]))
    assert 0.85 * A < first < 1.15 * A
    assert 0.85 * A < last < 1.15 * A
    assert 0.9 < last / first < 1.1


# ---------------------------------------------------------------------------
# 4. Two-sided accounting: equal-and-opposite covectors on the setpoint port.
# ---------------------------------------------------------------------------


def test_setpoint_port_receives_equal_and_opposite_covector():
    """The composite EMITS, in its outer position, the one-form
    ``omega_N(s) = omega_c|_{N^-} = k(q_eval - s)`` on the setpoint port
    (sec.forcing; CDLM eqn.omegaprime negated, STATUS.md §1) -- equal and
    opposite to the parameter covector ``xi_Q = -k(q_eval - s)`` the coalgebra
    consumes, because both are components of the one shared exact forcing
    (Newton's third law of the virtual spring). ``q_eval`` is the coalgebra's
    presented position: ``q`` for ``configuration``, ``q~ = q + p/m`` for
    ``phase``.

    Our runs (tests 2-3) DISCARD this one-form: physically, the setpoint
    holder is an infinitely stiff agent whose work against the spring is
    untracked by the closed loop.
    """
    # --- configuration: xi_Q recovered from the state update q' = q + dt * xi_Q ---
    dt, k = 0.05, 2.0
    q0, s = 1.3, 0.4
    O = Phi(impedance_arrangement(euclidean(1, dt), k, parameter=configuration))
    out_pos, _out_dir, new_state = _one_step(O, impedance_state(jnp.array([q0])), s)
    omega_N = out_pos[1]                       # the emitted one-form
    omega_at_s = float(omega_N(jnp.array([s]))[0])
    xi_Q = float((impedance_flat(new_state) - jnp.array([q0]))[0]) / dt
    np.testing.assert_allclose(xi_Q, -k * (q0 - s), atol=1e-10)
    np.testing.assert_allclose(omega_at_s, k * (q0 - s), atol=1e-10)
    np.testing.assert_allclose(omega_at_s + xi_Q, 0.0, atol=1e-10)

    # --- phase: xi_Q recovered from the momentum update p' = p + xi_Q ---
    m, k = 0.8, 0.6
    q0, p0, s = -0.9, 0.35, 0.25
    O = Phi(impedance_arrangement(diagonal(jnp.array([1.0 / m])), k, parameter=phase))
    out_pos, _out_dir, new_state = _one_step(
        O, impedance_state((jnp.array([q0]), jnp.array([p0]))), s)
    new_q, new_p = impedance_flat(new_state)
    q_tilde = q0 + p0 / m                      # forces evaluated at q~
    omega_at_s = float(out_pos[1](jnp.array([s]))[0])
    xi_Q = float((new_p - jnp.array([p0]))[0])
    np.testing.assert_allclose(xi_Q, -k * (q_tilde - s), atol=1e-10)
    np.testing.assert_allclose(omega_at_s, k * (q_tilde - s), atol=1e-10)
    np.testing.assert_allclose(omega_at_s + xi_Q, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 5. Regression against CDLM.
# ---------------------------------------------------------------------------


def test_regression_matches_cdlm():
    """The CDLM runs (``regress/control.npz``) are reproduced to rtol 1e-10
    (STATUS.md §1): states and reported positions identical, the emitted field
    on the setpoint port negated."""
    R = load_regress("control")
    tol = dict(rtol=1e-10, atol=1e-12)

    # 1. the five one-step draws
    dt, k = R["one_params"]
    O = Phi(impedance_arrangement(euclidean(1, dt), k, parameter=configuration))
    news, outs = [], []
    for q0, s in zip(R["one_q0"], R["one_s"]):
        out_pos, _, new_state = _one_step(O, impedance_state(jnp.asarray(q0)), s)
        news.append(np.asarray(impedance_flat(new_state))); outs.append(np.asarray(out_pos[0]))
    np.testing.assert_allclose(np.stack(news), R["one_new_q"], **tol)
    np.testing.assert_allclose(np.stack(outs), R["one_out"], **tol)

    # 2. the step response with the mid-run setpoint change
    dt2, k2, s1, s2, T = R["track_params"]
    T = int(T)
    O2 = Phi(impedance_arrangement(euclidean(1, dt2), k2, parameter=configuration))
    _, states = run_tracking(O2, impedance_state(jnp.array([0.0])), [s1] * T + [s2] * T)
    np.testing.assert_allclose(
        np.array([float(impedance_flat(st)[0]) for st in states]), R["track_q"], **tol)

    # 3. the phase oscillation
    m, k3, s_star, A, T3 = R["osc_params"]
    T3 = int(T3)
    O3 = Phi(impedance_arrangement(diagonal(jnp.array([1.0 / m])), k3, parameter=phase))
    outs, states = run_tracking(
        O3, impedance_state((jnp.array([s_star + A]), jnp.array([0.0]))), [s_star] * T3)
    np.testing.assert_allclose(np.array([float(o[0]) for o in outs]), R["osc_out"], **tol)
    np.testing.assert_allclose(
        np.array([float(impedance_flat(st)[0][0]) for st in states]), R["osc_q"], **tol)
    np.testing.assert_allclose(
        np.array([float(impedance_flat(st)[1][0]) for st in states]), R["osc_p"], **tol)

    # 4. the emitted field on the setpoint port: CDLM's, negated
    dt4, k4, q0, s = R["fld_conf_params"]
    Oc = Phi(impedance_arrangement(euclidean(1, dt4), k4, parameter=configuration))
    out_pos, _, new_state = _one_step(Oc, impedance_state(jnp.array([q0])), s)
    np.testing.assert_allclose(
        np.asarray(out_pos[1](jnp.array([s]))), -R["fld_conf_omega"], **tol)
    np.testing.assert_allclose(np.asarray(impedance_flat(new_state)), R["fld_conf_new_q"], **tol)

    m5, k5, q05, p05, s5 = R["fld_phase_params"]
    Op = Phi(impedance_arrangement(diagonal(jnp.array([1.0 / m5])), k5, parameter=phase))
    out_pos, _, new_state = _one_step(
        Op, impedance_state((jnp.array([q05]), jnp.array([p05]))), s5)
    nq, np_ = impedance_flat(new_state)
    np.testing.assert_allclose(
        np.asarray(out_pos[1](jnp.array([s5]))), -R["fld_phase_omega"], **tol)
    np.testing.assert_allclose(np.asarray(nq), R["fld_phase_new_q"], **tol)
    np.testing.assert_allclose(np.asarray(np_), R["fld_phase_new_p"], **tol)
