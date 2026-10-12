"""Audit: the port-Hamiltonian DC motor (van der Schaft 2014, Ex. 2.5) in arr.

Target ODE (states phi = flux linkage, p = angular momentum; i = phi/L,
omega = p/J):

    phi' = -(R/L) phi - K omega + V
    p'   =  K i - (b/J) p - tau

The claims audited here (see dc_motor.py for the full verdict). Everything
runs through the one dynamics functor ``Phi`` (sec.dynamics_functor) applied to
arrangements whose parameter is the configuration coalgebra (sec.matter) --
CDLM's ``Phiconf`` -- with each cell's cost ``U`` entering as the forcing
``omega = -dU`` (sec.forcing, STATUS.md §1):

1-2.  MONOLITHIC control: one closed box, sharp dt*[[R,K],[-K,b]], cost U = H
      plus the source term c.q with c = -A^{-1}(V,-tau).  One ``Phi`` step is
      the exact Euler step; the sketch's naive source potential -V phi + tau p
      is off by exactly dt*(A - I)(V,-tau).
3-4.  Lossless case R = b = 0: the energy obeys the EXACT geometric law
      H_n = H_0 (1 + (omega0 dt)^2)^n, omega0 = K/sqrt(LJ), while trading
      fully between phi^2/(2L) and p^2/(2J).
5-7.  COMPOSITIONAL (the paper's actual assertion): electrical cell (x)
      mechanical cell composed with a stateless gyrator; the composite data
      emerge from compose_seq, one ``Phi`` step is the exact Euler step, and
      the trajectory coincides with the monolithic control.
8.    The opposite-signed scalar sharps are FORCED: the one-step Jacobian
      factors (through the framework) as I - S.Hess(U_tot) with Hess
      symmetric, so the cross entries' signs are the sharps'; the motor's
      cross terms (-K/J, +K/L) have opposite signs.  Same-signed sharps
      reverse the torque.
9-10. The quoted sentence's own encoding (two T^*R boxes with -sharpS,
      electrical reports momentum, mechanical reports position) FAILS on the
      nose: (9) making (phi, p) autonomous forces the flux update to lose all
      mechanical dependence -- no back-EMF term can be produced; (10) the flow
      is divergence-free for EVERY potential, while the motor contracts.
11.   What survives of the sentence: the Bateman-style embedding -- tuned
      quadratic potentials carry the motor EXACTLY on a 2-dim invariant
      subspace of the 4-dim composite, but only from embedded initial states,
      and off-subspace errors grow at the mirror (expanding) rate.
12-14. Physics on the true encoding: spin-up to omega_ss = K V/(K^2 + R b)
      (the discrete fixed point coincides with the ODE's); back-EMF K*omega
      and torque K*i read off the framework's own covector xi_Q (in the
      forcing sign, so CDLM's covector negated); stall (rotor locked by the
      degenerate sharp 0) gives i -> V/R and stall torque K V/R.

State shapes: a composite's parameter is the tensor of its factors', so the
state of the two-cell motor is ``((phi, p), ())`` and of the sentence pair
``((z1, z2), ())``. ``_nest`` / ``_flat`` convert to and from the flat vector
the old test used, so the assertions below are on the same arrays.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import exact
from arrangements.dc_motor import (
    bateman_coefficients,
    bateman_motor,
    dc_motor,
    electrical_cell,
    gyrator,
    mechanical_cell,
    motor_monolithic,
    sentence_pair,
)
from arrangements.functors import Phi
from arrangements.interpretation import interpretation, trivial_omega
from arrangements.wiring import compose_seq, tensor_arrangements

from tests.chain import load_regress

_IN_POS = (jnp.zeros(0), trivial_omega(0))
_TRIV = lambda _o: (jnp.zeros(0), jnp.zeros(0))

# Generic parameter set used throughout (all constants distinct and non-round).
_P = dict(R=1.0, L=0.5, K=2.0, J=0.3, b=0.1)


# ---------------------------------------------------------------------------
# Flat <-> nested states: the composites' parameter is a tensor of coalgebras.
# ---------------------------------------------------------------------------


def _nest(z, cell_dim=1):
    """Flat parameter state as the composite's ``((s_1, s_2), s_coupling))``."""
    z = jnp.asarray(z, float)
    return ((z[:cell_dim], z[cell_dim:]), jnp.zeros(0))


def _flat(s):
    (s1, s2), _ = s
    return jnp.concatenate([s1, s2])


def _step_fn(arr, cell_dim=1):
    """The framework's own closed-system ``Phi`` step as a flat state map."""
    O = Phi(arr)

    def f(z):
        _, _, s2 = O.with_state(_nest(z, cell_dim)).run_one(_IN_POS, _TRIV)
        return _flat(s2)

    return f


def _mono_step_fn(arr):
    """The monolithic box's step: its parameter is a single coalgebra on R^2."""
    O = Phi(arr)

    def f(z):
        _, _, s2 = O.with_state(jnp.asarray(z, float)).run_one(_IN_POS, _TRIV)
        return s2

    return f


def _run(step_fn, q0, steps):
    """Iterate a flat state map (jitted for speed; nothing re-derived)."""
    step = jax.jit(step_fn)
    s = jnp.asarray(q0, float)
    traj = [np.asarray(s)]
    for _ in range(steps):
        s = step(s)
        traj.append(np.asarray(s))
    return np.stack(traj)


def _euler(z, R, L, K, J, b, V=0.0, tau=0.0, dt=0.01):
    """Oracle: one forward-Euler step of the target ODE."""
    phi, p = float(z[0]), float(z[1])
    return np.array(
        [
            phi + dt * (-(R / L) * phi - K * (p / J) + V),
            p + dt * (K * (phi / L) - (b / J) * p - tau),
        ]
    )


def _sharp_matrix(P, state, dim):
    """The response the parameter coalgebra applies to a covector: the Jacobian of
    ``xi |-> position(step(s, xi))``, which for the configuration coalgebra
    (``s -> s + sharp_s(xi)``) is the sharp matrix at ``s``."""
    return np.asarray(jax.jacobian(lambda xi: P.position(P.step(state, xi)))(jnp.zeros(dim)))


def _xi_Q(arr, z):
    """The parameter covector the coalgebra consumes at position ``z`` (the
    forcing sign: CDLM's ``d U_total`` negated)."""
    _, direction_action = interpretation(arr)(jnp.asarray(z, float))
    xi_Q, _, _ = direction_action(jnp.zeros(0), trivial_omega(0), jnp.zeros(0), jnp.zeros(0))
    return np.asarray(xi_Q)


# ---------------------------------------------------------------------------
# 1-2. Monolithic control.
# ---------------------------------------------------------------------------


def test_monolithic_one_step_exact():
    """One ``Phi`` step of the single-box motor == Euler step, to ~1e-14."""
    dt, V, tau = 0.01, 1.3, 0.2
    arr = motor_monolithic(**_P, V=V, tau=tau, dt=dt)
    step = _mono_step_fn(arr)
    rng = np.random.default_rng(0)
    for _ in range(5):
        z = rng.standard_normal(2)
        got = np.asarray(step(jnp.asarray(z)))
        want = _euler(z, **_P, V=V, tau=tau, dt=dt)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-14)


def test_monolithic_naive_sources_fail_by_exactly_dt_A_minus_I_u():
    """The sketch's source potential -V phi + tau p does NOT give the motor.

    Every potential term reaches the state through the sharp dt*A, so the
    naive linear coefficient c = -(V, -tau) produces the flow dt*A u instead
    of dt*u: a constant deviation dt*(A - I)u, nonzero whenever A u != u.
    The correct coefficient is c = -A^{-1} u (test 1 above passes with it).
    """
    dt, V, tau = 0.01, 1.3, 0.2
    R, L, K, J, b = _P["R"], _P["L"], _P["K"], _P["J"], _P["b"]
    A = np.array([[R, K], [-K, b]])
    u = np.array([V, -tau])
    dev = dt * (A - np.eye(2)) @ u
    assert np.linalg.norm(dev) > 1e-3  # the failure is not marginal here

    arr = motor_monolithic(**_P, V=V, tau=tau, dt=dt, naive_sources=True)
    step = _mono_step_fn(arr)
    rng = np.random.default_rng(1)
    for _ in range(3):
        z = rng.standard_normal(2)
        got = np.asarray(step(jnp.asarray(z)))
        want = _euler(z, **_P, V=V, tau=tau, dt=dt)
        np.testing.assert_allclose(got - want, dev, rtol=0, atol=1e-14)


# ---------------------------------------------------------------------------
# 3-4. Lossless case: exact energy law, full exchange.
# ---------------------------------------------------------------------------


def _lossless():
    L, J, K = _P["L"], _P["J"], _P["K"]
    omega0 = K / np.sqrt(L * J)
    dt = 0.05 / omega0  # omega0 * dt = 0.05 in tick units
    return L, J, K, omega0, dt


def test_lossless_energy_exact_geometric_growth():
    """R = b = 0: H_n = H_0 (1 + (omega0 dt)^2)^n EXACTLY (machine precision).

    Derived from the code's step matrix I + M: Q M is antisymmetric (so the
    O(dt) energy terms cancel identically) and M^T Q M = (omega0 dt)^2 Q, so
    every step multiplies H by exactly 1 + (omega0 dt)^2.  This is sharper
    than a bound: explicit Euler's energy error is a known geometric growth,
    O(dt) in the exponent over a fixed physical horizon, with no drift beyond
    it.  (Contrast the phase coalgebra's bounded wobble in test_lc_circuit.py
    -- here the claim under audit is the configuration coalgebra, whose
    lossless motor grows at this rate.)
    """
    L, J, K, omega0, dt = _lossless()
    arr = motor_monolithic(R=0.0, L=L, K=K, J=J, b=0.0, dt=dt)
    traj = _run(_mono_step_fn(arr), [0.3, 0.0], 800)
    H = traj[:, 0] ** 2 / (2 * L) + traj[:, 1] ** 2 / (2 * J)
    n = np.arange(len(H))
    growth = (1.0 + (omega0 * dt) ** 2) ** n
    np.testing.assert_allclose(H / H[0], growth, rtol=1e-12)


def test_lossless_energy_exchanges_between_flux_and_momentum():
    """The energy visibly trades: each term passes below 1% and above 99% of H."""
    L, J, K, omega0, dt = _lossless()
    arr = motor_monolithic(R=0.0, L=L, K=K, J=J, b=0.0, dt=dt)
    traj = _run(_mono_step_fn(arr), [0.3, 0.0], 300)  # > 2 periods (period ~ 126 steps)
    E_phi = traj[:, 0] ** 2 / (2 * L)
    E_p = traj[:, 1] ** 2 / (2 * J)
    H = E_phi + E_p
    assert np.min(E_phi / H) < 0.01 and np.max(E_phi / H) > 0.99
    assert np.min(E_p / H) < 0.01 and np.max(E_p / H) > 0.99


# ---------------------------------------------------------------------------
# 5-7. The compositional encoding (the paper's actual assertion).
# ---------------------------------------------------------------------------


def test_compositional_structure_emerges():
    """The composite's data emerge from compose_seq, not by hand: parameter
    R^2, sharp diag(+dt L, -dt J) (OPPOSITE signs), gyrator stateless, total
    forcing -d[(R/2)i^2 - Vi - (b/2)omega^2 - tau*omega + K i omega]."""
    dt, V, tau = 0.01, 1.3, 0.2
    R, L, K, J, b = _P["R"], _P["L"], _P["K"], _P["J"], _P["b"]
    arr = dc_motor(**_P, V=V, tau=tau, dt=dt)
    assert arr.P.Q_dim == 2
    assert (arr.out_dim_M, arr.in_dim_M, arr.out_dim_N, arr.in_dim_N) == (0, 0, 0, 0)
    assert gyrator(K).P.Q_dim == 0  # the coupling box is stateless
    np.testing.assert_allclose(
        _sharp_matrix(arr.P, _nest(jnp.zeros(2)), 2),
        np.diag([dt * L, -dt * J]),
        atol=1e-15,
    )
    U = lambda z: (R / 2) * (z[0] / L) ** 2 - V * (z[0] / L) - (b / 2) * (z[1] / J) ** 2 \
        - tau * (z[1] / J) + K * (z[0] / L) * (z[1] / J)
    want_omega = exact(lambda q, m_out, n_in: -U(q))
    rng = np.random.default_rng(2)
    for _ in range(5):
        z = jnp.asarray(rng.standard_normal(2))
        got = arr.omega(z, jnp.zeros(0), jnp.zeros(0))[0]
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(want_omega(z, jnp.zeros(0), jnp.zeros(0))[0]),
            rtol=0, atol=1e-12,
        )


def test_compositional_one_step_exact():
    """One ``Phi`` step of the two-cell + gyrator composite == Euler step."""
    dt, V, tau = 0.01, 1.3, 0.2
    step = _step_fn(dc_motor(**_P, V=V, tau=tau, dt=dt))
    rng = np.random.default_rng(3)
    for _ in range(5):
        z = rng.standard_normal(2)
        got = np.asarray(step(jnp.asarray(z)))
        want = _euler(z, **_P, V=V, tau=tau, dt=dt)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-14)


def test_compositional_matches_monolithic():
    """The two encodings produce the same trajectory (same exact Euler map)."""
    dt, V, tau = 0.01, 1.3, 0.2
    t1 = _run(_step_fn(dc_motor(**_P, V=V, tau=tau, dt=dt)), [0.4, -0.2], 300)
    t2 = _run(_mono_step_fn(motor_monolithic(**_P, V=V, tau=tau, dt=dt)), [0.4, -0.2], 300)
    np.testing.assert_allclose(t1, t2, rtol=0, atol=1e-11)


# ---------------------------------------------------------------------------
# 8. Opposite-signed sharps are forced (the Lotka-Volterra sign mechanism).
# ---------------------------------------------------------------------------


def test_opposite_sharps_forced():
    """The step Jacobian factors as I - S.Hess(U_tot) with Hess SYMMETRIC, so
    cross entries are (-s_e H12, -s_m H12): their sign ratio is the sharps'.
    The motor needs (-dt K/J, +dt K/L) -- opposite signs -- so s_e s_m < 0 is
    forced for ANY potential-mediated coupling of two 1-dim boxes.  The
    same-signed control produces a REVERSED torque.  (The Hessian is read off
    the forcing: ``omega|_Q = -dU``, so ``Hess(U) = -d(omega|_Q)``, and its
    symmetry is exactness of the composite forcing.)"""
    dt = 0.01
    R, L, K, J, b = _P["R"], _P["L"], _P["K"], _P["J"], _P["b"]
    arr = dc_motor(**_P, dt=dt)
    z0 = jnp.array([0.4, -0.2])

    # (a) The factorization, all through the framework: jac(step) = I - S H.
    jac = np.asarray(jax.jacobian(_step_fn(arr))(z0))
    S = _sharp_matrix(arr.P, _nest(z0), 2)
    H = -np.asarray(jax.jacobian(lambda q: arr.omega(q, jnp.zeros(0), jnp.zeros(0))[0])(z0))
    np.testing.assert_allclose(jac, np.eye(2) - S @ H, atol=1e-13)
    np.testing.assert_allclose(H, H.T, atol=1e-13)  # symmetric: the mechanism

    # (b) The true encoding's cross terms are the motor's, with opposite signs.
    np.testing.assert_allclose(jac[0, 1], -dt * K / J, atol=1e-14)  # back-EMF
    np.testing.assert_allclose(jac[1, 0], +dt * K / L, atol=1e-14)  # torque
    assert jac[0, 1] * jac[1, 0] < 0.0

    # (c) Same-signed sharps (negative control): the torque comes out reversed.
    same_sign = compose_seq(
        tensor_arrangements(
            [electrical_cell(R, L, 0.0, dt), mechanical_cell(b, J, 0.0, dt, sharp_sign=+1.0)]
        ),
        gyrator(K),
    )
    jac2 = np.asarray(jax.jacobian(_step_fn(same_sign))(z0))
    np.testing.assert_allclose(jac2[0, 1], -dt * K / J, atol=1e-14)  # unchanged
    np.testing.assert_allclose(jac2[1, 0], -dt * K / L, atol=1e-14)  # REVERSED
    assert jac2[0, 1] * jac2[1, 0] > 0.0  # symmetric-type coupling, not gyroscopic


# ---------------------------------------------------------------------------
# 9-10. The quoted sentence's encoding fails on the nose.
# ---------------------------------------------------------------------------


def test_sentence_encoding_cannot_produce_back_emf():
    """T^*R boxes, -sharpS, electrical reports momentum y1, mechanical reports
    position x2.  The flux update is y1' = y1 - dt*dU1/dx1(x1, y1): the
    coupling (which reads y1, x2) never reaches it.  For (y1, y2) = (phi, p)
    to evolve autonomously the x1-dependence must vanish (a1 = b1 = 0 below),
    and then the y1-row of the Jacobian has NO mechanical entries at all: the
    back-EMF term -dt*K/J, which the motor requires there, cannot be produced
    by ANY choice of the remaining coefficients.  Conversely a1 != 0 makes
    y1' depend on the hidden x1, so (phi, p) is not autonomous."""
    dt = 0.01
    rng = np.random.default_rng(4)
    for _ in range(5):
        d1, a2, b2, d2, c = rng.standard_normal(5)
        arr = sentence_pair((0.0, 0.0, d1), (a2, b2, d2), c, dt=dt)
        z = jnp.asarray(rng.standard_normal(4))  # state (x1, y1, x2, y2)
        jac = np.asarray(jax.jacobian(_step_fn(arr, cell_dim=2))(z))
        # y1-row (index 1): no dependence on the mechanical box (x2, y2)...
        np.testing.assert_allclose(jac[1, 2], 0.0, atol=1e-15)
        np.testing.assert_allclose(jac[1, 3], 0.0, atol=1e-15)
        # ...and none on x1 either (that was the autonomy requirement a1 = 0).
        np.testing.assert_allclose(jac[1, 0], 0.0, atol=1e-15)

    # Conversely: a1 != 0 couples y1' to the hidden coordinate x1.
    arr = sentence_pair((0.7, 0.0, 0.3), (0.4, 0.0, -0.2), 0.5, dt=dt)
    jac = np.asarray(jax.jacobian(_step_fn(arr, cell_dim=2))(jnp.zeros(4)))
    assert abs(jac[1, 0]) > 1e-4  # = dt * a1


def test_sentence_encoding_is_volume_preserving():
    """With EVERY box carrying the antisymmetric -sharpS, the composite field
    (S (+) S) dU is divergence-free for every potential: tr(jac - I) = 0.
    The motor contracts at rate R/L + b/J > 0, so no potentials, reports, or
    wiring in this all-antisymmetric shape yield the lossy motor as the
    dynamics of the box states (only the subspace salvage of test 11)."""
    dt = 0.01
    R, L, J, b = _P["R"], _P["L"], _P["J"], _P["b"]
    rng = np.random.default_rng(5)
    for _ in range(5):
        ce = tuple(rng.standard_normal(3))
        cm = tuple(rng.standard_normal(3))
        arr = sentence_pair(ce, cm, float(rng.standard_normal()), dt=dt)
        z = jnp.asarray(rng.standard_normal(4))
        jac = np.asarray(jax.jacobian(_step_fn(arr, cell_dim=2))(z))
        np.testing.assert_allclose(np.trace(jac - np.eye(4)), 0.0, atol=1e-13)
    # ...whereas the motor's Euler map has trace 2 - dt*(R/L + b/J) < 2.
    assert dt * (R / L + b / J) > 1e-3


# ---------------------------------------------------------------------------
# 11. What survives of the sentence: the Bateman-style embedding.
# ---------------------------------------------------------------------------


def test_bateman_salvage_exact_on_the_invariant_subspace():
    """With tuned quadratic potentials the sentence encoding carries the motor
    EXACTLY on the invariant subspace: from embedded initial states, (y1, y2)
    reproduces the motor's Euler trajectory and the hidden coordinates stay
    slaved.  Off the subspace the mirror modes EXPAND (spectrum of a
    Hamiltonian flow is {lam, -lam}-symmetric), so the emulation requires a
    measure-zero initialization and is exponentially non-robust -- the precise
    sense in which the sentence's encoding is weaker than the two-cell one."""
    dt, steps = 0.002, 3000
    arr, embed, project = bateman_motor(**_P, dt=dt)
    C = bateman_coefficients(**_P)

    # (a) One-step exactness on the subspace, including the hidden coordinates.
    rng = np.random.default_rng(6)
    step = _step_fn(arr, cell_dim=2)
    for _ in range(5):
        y0 = rng.standard_normal(2)
        z1 = step(embed(jnp.asarray(y0)))
        y1 = _euler(y0, **_P, dt=dt)
        np.testing.assert_allclose(np.asarray(project(z1)), y1, rtol=0, atol=1e-14)
        np.testing.assert_allclose(  # still on the subspace
            np.asarray(z1), np.asarray(embed(jnp.asarray(y1))), rtol=0, atol=1e-14
        )

    # (b) Long run from an embedded state: exact match to the Euler trajectory.
    y = np.array([0.7, -0.3])
    traj = _run(_step_fn(arr, cell_dim=2), np.asarray(embed(jnp.asarray(y))), steps)
    oracle = np.empty((steps + 1, 2))
    oracle[0] = y
    for t in range(steps):
        oracle[t + 1] = _euler(oracle[t], **_P, dt=dt)
    np.testing.assert_allclose(traj[:, [1, 3]], oracle, rtol=0, atol=1e-9)

    # (c) Non-robustness: a 1e-6 kick on the hidden x1 grows at the mirror
    # rate (|1 - dt*Re(lam)| per step) and wrecks the physical coordinates.
    z0 = np.asarray(embed(jnp.asarray(y))).copy()
    z0[0] += 1e-6
    traj_p = _run(_step_fn(arr, cell_dim=2), z0, steps)
    err_tuned = np.max(np.abs(traj[:, [1, 3]] - oracle))
    err_pert = np.max(np.abs(traj_p[:, [1, 3]] - oracle))
    assert err_tuned < 1e-9
    assert err_pert > 1e-4  # ~ 1e-6 * exp(steps * dt * (R/L + b/J)/2) mixing in
    assert err_pert > 1e4 * err_tuned


# ---------------------------------------------------------------------------
# 12-14. Physics sanity on the true encoding.
# ---------------------------------------------------------------------------


def test_spin_up_to_steady_state():
    """From rest with constant V, no load: the motor spins up and settles at
    omega_ss = K V / (K^2 + R b).  The discrete fixed point solves the SAME
    linear equation as the ODE's equilibrium (z + dt(Ez + u) = z iff
    Ez + u = 0), so the check is exact up to convergence, not O(dt)-loose."""
    dt, V = 0.002, 1.0
    R, L, K, J, b = _P["R"], _P["L"], _P["K"], _P["J"], _P["b"]
    traj = _run(_step_fn(dc_motor(**_P, V=V, dt=dt)), [0.0, 0.0], 12000)
    omega = traj[:, 1] / J
    omega_ss = K * V / (K**2 + R * b)
    i_ss = b * omega_ss / K  # equilibrium current balances friction torque

    assert omega[200] > 0.0  # it spins up (and in the +V direction)
    np.testing.assert_allclose(omega[-1], omega_ss, rtol=0, atol=1e-10)
    np.testing.assert_allclose(traj[-1, 0] / L, i_ss, rtol=0, atol=1e-10)


def test_back_emf_and_torque_via_framework_covectors():
    """Along the spin-up, the covector xi_Q the coalgebra consumes (computed by
    the framework's own interpretation, eqn.bigtheta with ``dU`` replaced by
    ``omega``) decomposes as

        L * xi_Q[phi] = -(R i - V + K omega)   (Ohmic + source + BACK-EMF)
        J * xi_Q[p]   = -(-b omega - tau + K i)  (friction + load + TORQUE)

    i.e. the coupling contributes exactly K*(reported omega) to the electrical
    covector and K*(reported i) to the mechanical one: voltage proportional to
    angular velocity, torque proportional to current, at every visited state.
    The overall minus is the forcing sign (``omega = -dU``, STATUS.md §1), which
    the coalgebra's ADDING step undoes: the trajectory is CDLM's."""
    dt, V, tau = 0.002, 1.0, 0.05
    R, L, K, J, b = _P["R"], _P["L"], _P["K"], _P["J"], _P["b"]
    arr = dc_motor(**_P, V=V, tau=tau, dt=dt)
    traj = _run(_step_fn(arr), [0.0, 0.0], 400)
    for z in traj[::50]:
        xi_Q = _xi_Q(arr, z)
        i, om = z[0] / L, z[1] / J
        np.testing.assert_allclose(L * float(xi_Q[0]) + (R * i - V), -K * om, atol=1e-13)
        np.testing.assert_allclose(J * float(xi_Q[1]) + (-b * om - tau), -K * i, atol=1e-13)


def test_stall_current_and_stall_torque():
    """Rotor locked by the degenerate mechanical sharp 0 (omega stays 0, so no
    back-EMF): the current settles at i = V/R exactly (the discrete fixed
    point), and the torque the gyrator exerts on the rotor -- the mechanical
    sharp's per-tick flow J * xi_Q[p] / dt read off the framework covector --
    is the stall torque K V / R, proportional to V (with the forcing's sign,
    so the recorded covector is CDLM's negated)."""
    dt = 0.002
    R, L, K, J = _P["R"], _P["L"], _P["K"], _P["J"]
    for V in (0.8, 1.6):
        arr = dc_motor(**_P, V=V, dt=dt, locked_rotor=True)
        traj = _run(_step_fn(arr), [0.0, 0.0], 8000)
        assert np.all(traj[:, 1] == 0.0)  # the rotor is truly held
        i_end = traj[-1, 0] / L
        np.testing.assert_allclose(i_end, V / R, rtol=0, atol=1e-12)

        xi_Q = _xi_Q(arr, traj[-1])
        stall_torque = -J * float(xi_Q[1])  # the unlocked sharp -dt*J would step p by -dt*this
        np.testing.assert_allclose(stall_torque, K * V / R, rtol=0, atol=1e-11)


# ---------------------------------------------------------------------------
# Flow-coordinate variant (the paper's ex.further_reach "DC motor" item).
# ---------------------------------------------------------------------------


def test_flow_variant_one_tick_is_euler_in_i_omega():
    """The (i, omega)-state encoding steps EXACTLY by

        i     -> i     + dt (V - R i - K omega) / L,
        omega -> omega + dt (K i - b omega - tau) / J,

    i.e. the momentum encoding's Euler step conjugated by the constant
    rescaling phi = L i, p = J omega -- same motor, LC-item convention."""
    from arrangements.dc_motor import dc_motor_flow

    Rr, L, K, J, b, V, tau, dt = 1.3, 0.7, 0.9, 2.1, 0.4, 5.0, 0.6, 0.01
    step = _step_fn(dc_motor_flow(Rr, L, K, J, b, V, tau, dt))
    rng = np.random.default_rng(5)
    for _ in range(5):
        i0, w0 = rng.standard_normal(2)
        got = np.asarray(step(jnp.array([i0, w0])))
        expect = np.array(
            [i0 + dt * (V - Rr * i0 - K * w0) / L, w0 + dt * (K * i0 - b * w0 - tau) / J]
        )
        np.testing.assert_allclose(got, expect, atol=1e-13)


# ---------------------------------------------------------------------------
# Regression: the CDLM trajectories, sign-translated (STATUS.md §1).
# ---------------------------------------------------------------------------


def test_regression_matches_cdlm():
    """``regress/dc_motor.npz`` is reproduced to rtol 1e-10: states, sharps and
    Jacobians identical, the covector ``xi_Q`` negated (STATUS.md §1)."""
    Reg = load_regress("dc_motor")
    R, L, K, J, b = Reg["P"]
    Pm = dict(R=R, L=L, K=K, J=J, b=b)
    close = lambda new, old: np.testing.assert_allclose(new, old, rtol=1e-10, atol=1e-12)

    # monolithic (correct and naive sources)
    dt, V, tau = Reg["mono_params"]
    step = _mono_step_fn(motor_monolithic(**Pm, V=V, tau=tau, dt=dt))
    close(np.stack([np.asarray(step(jnp.asarray(z))) for z in Reg["mono_z"]]), Reg["mono_next"])
    step = _mono_step_fn(motor_monolithic(**Pm, V=V, tau=tau, dt=dt, naive_sources=True))
    close(np.stack([np.asarray(step(jnp.asarray(z))) for z in Reg["naive_z"]]), Reg["naive_next"])

    # lossless trajectory
    dt_l, _omega0 = Reg["lossless_params"]
    close(_run(_mono_step_fn(motor_monolithic(R=0.0, L=L, K=K, J=J, b=0.0, dt=dt_l)), [0.3, 0.0], 800),
          Reg["lossless_traj"])

    # compositional: step, sharp, forcing (= -dU at the recorded points), trajectory
    arr = dc_motor(**Pm, V=V, tau=tau, dt=dt)
    step = _step_fn(arr)
    close(np.stack([np.asarray(step(jnp.asarray(z))) for z in Reg["comp_z"]]), Reg["comp_next"])
    close(_sharp_matrix(arr.P, _nest(jnp.zeros(2)), 2), Reg["comp_sharp"])
    # the old composite potential's differential is the new composite forcing negated
    new_omega_Q = np.stack([
        np.asarray(arr.omega(jnp.asarray(z), jnp.zeros(0), jnp.zeros(0))[0]) for z in Reg["comp_U_z"]
    ])
    close(new_omega_Q, -Reg["comp_dU"])
    close(_run(_step_fn(arr), [0.4, -0.2], 300), Reg["comp_traj"])
    close(_run(_mono_step_fn(motor_monolithic(**Pm, V=V, tau=tau, dt=dt)), [0.4, -0.2], 300),
          Reg["mono_traj"])

    # Jacobians: true encoding and the same-signed control
    z0 = jnp.asarray(Reg["jac_z0"])
    close(np.asarray(jax.jacobian(_step_fn(dc_motor(**Pm, dt=dt)))(z0)), Reg["jac_true"])
    same_sign = compose_seq(
        tensor_arrangements(
            [electrical_cell(R, L, 0.0, dt), mechanical_cell(b, J, 0.0, dt, sharp_sign=+1.0)]
        ),
        gyrator(K),
    )
    close(np.asarray(jax.jacobian(_step_fn(same_sign))(z0)), Reg["jac_same_sign"])
    assert gyrator(K).P.Q_dim == int(Reg["gyrator_dim"])

    # sentence encoding: both Jacobian batteries
    for coeffs, z, jac_old in zip(Reg["sent_coeffs"], Reg["sent_z"], Reg["sent_jac"]):
        a1, b1, d1, a2, b2, d2, c = coeffs
        arr_s = sentence_pair((a1, b1, d1), (a2, b2, d2), c, dt=dt)
        close(np.asarray(jax.jacobian(_step_fn(arr_s, cell_dim=2))(jnp.asarray(z))), jac_old)
    arr_s = sentence_pair((0.7, 0.0, 0.3), (0.4, 0.0, -0.2), 0.5, dt=dt)
    close(np.asarray(jax.jacobian(_step_fn(arr_s, cell_dim=2))(jnp.zeros(4))), Reg["sent_a1_jac"])
    for coeffs, z, jac_old in zip(Reg["vol_coeffs"], Reg["vol_z"], Reg["vol_jac"]):
        arr_s = sentence_pair(tuple(coeffs[:3]), tuple(coeffs[3:6]), float(coeffs[6]), dt=dt)
        close(np.asarray(jax.jacobian(_step_fn(arr_s, cell_dim=2))(jnp.asarray(z))), jac_old)

    # Bateman: one step from embedded states, and both long runs
    dt_b, steps_b = Reg["bat_params"]
    steps_b = int(steps_b)
    arr_b, embed, _ = bateman_motor(**Pm, dt=dt_b)
    step_b = _step_fn(arr_b, cell_dim=2)
    close(np.stack([np.asarray(step_b(embed(jnp.asarray(y)))) for y in Reg["bat_y0"]]), Reg["bat_next"])
    close(np.asarray(embed(jnp.array([0.7, -0.3]))), Reg["bat_embed"])
    close(_run(_step_fn(arr_b, cell_dim=2), Reg["bat_embed"], steps_b), Reg["bat_traj"])
    z_pert = np.asarray(Reg["bat_embed"]).copy(); z_pert[0] += 1e-6
    close(_run(_step_fn(arr_b, cell_dim=2), z_pert, steps_b), Reg["bat_traj_pert"])

    # spin-up, covectors (negated), stall
    dt_s, V_s = Reg["spin_params"]
    close(_run(_step_fn(dc_motor(**Pm, V=V_s, dt=dt_s)), [0.0, 0.0], 12000), Reg["spin_traj"])

    dt_c, V_c, tau_c = Reg["cov_params"]
    arr_c = dc_motor(**Pm, V=V_c, tau=tau_c, dt=dt_c)
    traj_c = _run(_step_fn(arr_c), [0.0, 0.0], 400)
    close(traj_c, Reg["cov_traj"])
    close(np.stack([_xi_Q(arr_c, z) for z in traj_c[::50]]), -Reg["cov_xi_Q"])

    for k in range(2):
        dt_st, V_st = Reg[f"stall{k}_params"]
        arr_st = dc_motor(**Pm, V=V_st, dt=dt_st, locked_rotor=True)
        traj_st = _run(_step_fn(arr_st), [0.0, 0.0], 8000)
        close(traj_st, Reg[f"stall{k}_traj"])
        close(_xi_Q(arr_st, traj_st[-1]), -Reg[f"stall{k}_xi_Q"])

    # flow-coordinate variant
    from arrangements.dc_motor import dc_motor_flow

    Rr, Lf, Kf, Jf, bf, Vf, tauf, dtf = Reg["flow_params"]
    step_f = _step_fn(dc_motor_flow(Rr, Lf, Kf, Jf, bf, Vf, tauf, dtf))
    close(np.stack([np.asarray(step_f(jnp.asarray(z))) for z in Reg["flow_z"]]), Reg["flow_next"])
