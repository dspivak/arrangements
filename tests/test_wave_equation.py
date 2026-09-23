"""The wave equation: the chain of harmonic particles with *phase* parameters (book sec.heat_waves).

The same chain wiring as test_heat_equation.py; the particles' parameter is
``phase(Q)`` instead of ``configuration(Q)``, which makes a *different*
arrangement (def:arr), and ``Phi`` of it is the conservative second-order
discrete wave equation. References in CDLM: sec.spring_first_pass
(the particle and wiring), eqn.potlens_composite_final (the state update),
eqn.discrete_wave / eqn.wave_bc (pinned ends).

Environment covectors are forcings in the book's sign: the pin ``q_{K+1} = 0``
is ``xi_N = -kappa * q~_K``; the published field is ``omega_N = pullback + omega|_N``,
i.e. minus CDLM's ``dU``-based field (STATUS.md §1).
"""

import jax.numpy as jnp
import numpy as np

from arrangements.coalgebra import configuration, phase
from arrangements.functors import Phi
from arrangements.wiring import compose_chain

from tests.chain import IN_POS, harmonic_particle, load_regress, phase_flat, phase_state


def _wave_chain(K, m, kappa):
    return compose_chain([harmonic_particle(phase, m, kappa)] * K)


def _composite_final(q, p, q_prev_extern, xi_N_extern, m, kappa):
    """Reference implementation of CDLM eqn.potlens_composite_final, with the
    external covector ``xi_N_extern`` a forcing (added, not subtracted)."""
    K = q.shape[0]
    qt = q + p / m  # presented positions q~ (forces are evaluated here)
    q_new = qt
    p_new = []
    for i in range(K):
        if i < K - 1:
            q_prev = q_prev_extern if i == 0 else qt[i - 1]
            q_next = qt[i + 1]
            p_i_new = p[i] + kappa * (q_prev + q_next - 2.0 * qt[i])
        else:
            q_prev = qt[i - 1] if K > 1 else q_prev_extern
            p_i_new = p[i] + kappa * (q_prev - qt[i]) + xi_N_extern
        p_new.append(p_i_new)
    return q_new, jnp.stack(p_new)


def _run(O, q, p, xi_N, q_prev):
    K = q.shape[0]
    out_pos, out_dir, new_state = O.with_state(phase_state(q, p)).run_one(
        IN_POS, lambda _o: (jnp.atleast_1d(xi_N), jnp.atleast_1d(q_prev)))
    return out_pos, out_dir, phase_flat(new_state, K)


def test_state_update_matches_composite_final():
    """Apply Phi to a 5-particle phase chain; check eqn.potlens_composite_final."""
    K = 5
    m, kappa = 1.7, 2.3
    O = Phi(_wave_chain(K, m, kappa))

    rng = np.random.default_rng(20260507)
    for _ in range(5):
        q = jnp.asarray(rng.standard_normal(K))
        p = jnp.asarray(rng.standard_normal(K))
        q_prev = float(rng.standard_normal())
        xi_N = float(rng.standard_normal())

        _, _, (new_q, new_p) = _run(O, q, p, xi_N, q_prev)

        ref_q, ref_p = _composite_final(q, p, q_prev, xi_N, m, kappa)
        np.testing.assert_allclose(np.asarray(new_q), np.asarray(ref_q), atol=1e-10)
        np.testing.assert_allclose(np.asarray(new_p), np.asarray(ref_p), atol=1e-10)


def test_omega_N_is_harmonic_spring_forcing():
    """The published one-form on N^-: omega_N(q_0) = kappa(q_1 - q_0), i.e. the
    forcing ``-dk`` of the first spring seen from the input port (minus CDLM's eqn.omegaprime)."""
    K = 5
    m, kappa = 1.0, 0.7
    O = Phi(_wave_chain(K, m, kappa))

    q = jnp.array([0.4, -0.2, 1.1, 0.0, 2.3])
    p = jnp.zeros(K)
    _, fiber = O.step(phase_state(q, p))
    out_pos, _ = fiber(IN_POS)
    out_n, omega_N = out_pos  # omega_N is the one-form (a callable)
    np.testing.assert_allclose(np.asarray(out_n), np.asarray(q[K - 1:K]), atol=1e-10)
    # omega_N(q_0) = kappa(q_1 - q_0): value at 0 is b, the 1D slope is A.
    b_N = omega_N(jnp.zeros(1))
    A_N = omega_N(jnp.ones(1)) - b_N
    np.testing.assert_allclose(np.asarray(A_N), np.array([-kappa]), atol=1e-10)
    np.testing.assert_allclose(np.asarray(b_N), np.array([kappa * float(q[0])]), atol=1e-10)


def test_discrete_wave_equation_with_pinned_ends():
    """eqn.discrete_wave: m * ddot q_i = kappa (q_{i-1} - 2 q_i + q_{i+1}) at every step.

    Boundary conditions: q_0 := 0, xi_N := -kappa * q~_K (presented), a forcing.
    The recurrence is the centered one: m ddot q_i(t) = kappa Lap(q(t+1))_i.
    """
    K = 5
    T = 10
    m, kappa = 1.5, 0.9
    O = Phi(_wave_chain(K, m, kappa))

    rng = np.random.default_rng(42)
    q = jnp.asarray(rng.standard_normal(K))
    p = jnp.zeros(K)

    q_traj = [q]
    for _ in range(T + 2):
        xi_N = -kappa * float(q[K - 1] + p[K - 1] / m)
        _, _, (q, p) = _run(O, q, p, xi_N, 0.0)
        q_traj.append(q)

    q_arr = np.stack([np.asarray(x) for x in q_traj], axis=0)
    for t in range(T):
        ddot_q = q_arr[t + 2] - 2.0 * q_arr[t + 1] + q_arr[t]
        q_aug = np.concatenate([[0.0], q_arr[t + 1], [0.0]])
        laplacian = q_aug[:-2] - 2.0 * q_aug[1:-1] + q_aug[2:]
        residual = m * ddot_q - kappa * laplacian
        np.testing.assert_allclose(residual, np.zeros(K), atol=1e-10)


def test_heat_and_wave_are_two_arrangements_on_one_wiring():
    """notes/DESIGN.md §5: the integrator is part of the arrangement. The same wiring
    ``compose_chain`` on the same particle data gives two arrangements, one with
    ``configuration`` parameters and one with ``phase``; ``Phi`` takes no
    integrator, and the two differ in their parameter, not in their maps."""
    K, m, kappa = 3, 1.5, 0.9
    heat = compose_chain([harmonic_particle(configuration, m, kappa)] * K)
    wave = compose_chain([harmonic_particle(phase, m, kappa)] * K)

    # same interface and same lens data
    assert (heat.in_dim_M, heat.out_dim_M, heat.in_dim_N, heat.out_dim_N) == \
           (wave.in_dim_M, wave.out_dim_M, wave.in_dim_N, wave.out_dim_N)
    q = jnp.array([0.3, -1.0, 2.0]); n_in = jnp.array([0.7])
    np.testing.assert_allclose(heat.out_f(q, jnp.zeros(0)), wave.out_f(q, jnp.zeros(0)))
    for a, b in zip(heat.omega(q, jnp.zeros(0), n_in), wave.omega(q, jnp.zeros(0), n_in)):
        np.testing.assert_allclose(a, b)

    # different parameters: state R^K vs T^*R^K, hence different dynamics from Phi alone
    O_heat, O_wave = Phi(heat), Phi(wave)
    s_heat, s_wave = O_heat.state, O_wave.state
    assert heat.P.Q_dim == wave.P.Q_dim == K
    assert isinstance(s_heat[0][0][0], jnp.ndarray) and isinstance(s_wave[0][0][0], tuple)
    import inspect
    assert list(inspect.signature(Phi).parameters) == ["arr"]


def test_regression_matches_cdlm():
    """The CDLM outputs (``regress/wave.npz``) are reproduced to rtol 1e-10 (STATUS.md §1):
    states identical, environment covectors negated, published field negated."""
    R = load_regress("wave")

    K, m, kappa = R["cf_params"]; K = int(K)
    O = Phi(_wave_chain(K, m, kappa))
    for i in range(R["cf_q"].shape[0]):
        _, _, (nq, np_) = _run(O, jnp.asarray(R["cf_q"][i]), jnp.asarray(R["cf_p"][i]),
                               -float(R["cf_xiN"][i]), float(R["cf_qprev"][i]))
        np.testing.assert_allclose(np.asarray(nq), R["cf_new_q"][i], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(np.asarray(np_), R["cf_new_p"][i], rtol=1e-10, atol=1e-12)

    K2, m2, kappa2 = R["field_params"]; K2 = int(K2)
    O2 = Phi(_wave_chain(K2, m2, kappa2))
    _, fiber = O2.step(phase_state(jnp.asarray(R["field_q"]), jnp.zeros(K2)))
    (out_n, omega_N), _ = fiber(IN_POS)
    np.testing.assert_allclose(np.asarray(out_n), R["field_out_n"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.asarray(omega_N(jnp.zeros(1))), -R["field_at_0"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.asarray(omega_N(jnp.ones(1))), -R["field_at_1"], rtol=1e-10, atol=1e-12)

    K3, T3, m3, kappa3 = R["pin_params"]; K3, T3 = int(K3), int(T3)
    O3 = Phi(_wave_chain(K3, m3, kappa3))
    q, p = jnp.asarray(R["pin_q"][0]), jnp.asarray(R["pin_p"][0])
    qs, ps = [q], [p]
    for _ in range(T3 + 2):
        xi_N = -kappa3 * float(q[K3 - 1] + p[K3 - 1] / m3)
        _, _, (q, p) = _run(O3, q, p, xi_N, 0.0)
        qs.append(q); ps.append(p)
    np.testing.assert_allclose(np.stack([np.asarray(x) for x in qs]), R["pin_q"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.stack([np.asarray(x) for x in ps]), R["pin_p"], rtol=1e-10, atol=1e-12)
