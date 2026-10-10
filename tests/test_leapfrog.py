"""Leapfrog as a two-stage coalgebra, dynamics in pc^(2) (leapfrog.py; cf. rmk.multistage).

Same harmonic chain as the wave test, with the particles' parameter the symplectic
two-stage coalgebra ``leapfrog(Q)`` (velocity Verlet) -- a third arrangement on the
one chain wiring, next to heat (``configuration``) and wave (``phase``):
* eliminating momentum gives the centered discrete wave recurrence
      m (q_{n+1} - 2 q_n + q_{n-1}) = kappa * Lap(q_n);   [force at the center]
* the energy stays in a bounded band (symplectic).

The single-stage phase coalgebra is itself symplectic (presented-position
readout), so it stays bounded too; leapfrog is a higher-order alternative.
The ends are held by boxes (plan P43 (b) job 4, Claude Opus 5.5, de71554a, 2026-10-10): every run is
``Phi2`` of ``wiring.held_chain`` with leapfrog parts and leapfrog held ends, closed. Until then the
pin was computed in code at each stage, ``xi_N = -kappa q_K`` from the emitted position (CDLM's
``+kappa q_K`` with a subtracting integrator; STATUS.md §1); that run is kept as ``_boundary`` and
checked equal. The phase chain in the boundedness test is held the same way.
"""

import jax.numpy as jnp
import numpy as np

from arrangements.coalgebra import phase
from arrangements.functors import Phi
from arrangements.leapfrog import leapfrog
from arrangements.pc2 import Phi2
from arrangements.wiring import compose_chain

from tests.chain import IN_POS, HeldChain, harmonic_particle, load_regress, phase_flat, phase_state


def _boundary(kappa):
    # pinned ends: q_prev = 0, xi_N = -kappa * (emitted q_K).  Reads the emitted position,
    # so it is re-evaluated correctly at both leapfrog stages.
    return lambda out_pos: (-kappa * out_pos[0], jnp.array([0.0]))


def _lap_pinned(v):
    aug = np.concatenate([[0.0], np.asarray(v), [0.0]])
    return aug[:-2] - 2.0 * aug[1:-1] + aug[2:]


def _energy(q, xi, m, kappa):
    ke = 0.5 * float(np.sum(np.asarray(xi) ** 2)) / m
    aug = np.concatenate([[0.0], np.asarray(q), [0.0]])
    pe = 0.5 * kappa * float(np.sum(np.diff(aug) ** 2))
    return ke + pe


def _leap_chain(K, m, kappa):
    return Phi2(compose_chain([harmonic_particle(leapfrog, m, kappa)] * K))


def _held(K, m, kappa, parameter=leapfrog):
    """The chain with its ends held by boxes, closed."""
    return HeldChain([harmonic_particle(parameter, m, kappa)] * K, kappa, parameter,
                     two_stage=parameter is leapfrog)


def test_held_ends_are_the_port_fed_run():
    """The held chain's ``Phi2`` equals the old run that held the ends in code, 300 steps."""
    K, m, kappa = 5, 1.5, 0.9
    O, H, bdy = _leap_chain(K, m, kappa), _held(K, m, kappa), _boundary(kappa)
    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.standard_normal(K))
    p = jnp.asarray(rng.standard_normal(K))
    state = phase_state(q, p)
    for _ in range(300):
        _, _, state = O.with_state(state).run_one(IN_POS, bdy)
        q, p = H.step(q, p)
    a, b = phase_flat(state, K)
    np.testing.assert_allclose(np.asarray(q), np.asarray(a), atol=1e-12)
    np.testing.assert_allclose(np.asarray(p), np.asarray(b), atol=1e-12)


def test_leapfrog_and_phase_are_both_bounded():
    """The presented-position readout makes the single-stage phase coalgebra
    symplectic, so it stays bounded like leapfrog (kappa/m = 0.6)."""
    K, m, kappa = 5, 1.5, 0.9
    rng = np.random.default_rng(0)
    q0 = jnp.asarray(rng.standard_normal(K))
    peak0 = float(np.max(np.abs(np.asarray(q0))))
    bdy = _boundary(kappa)

    H = _held(K, m, kappa)
    q, p = q0, jnp.zeros(K)
    leap_peak = peak0
    for _ in range(300):
        q, p = H.step(q, p)
        leap_peak = max(leap_peak, float(np.max(np.abs(np.asarray(q)))))

    He = _held(K, m, kappa, phase)
    q, p = q0, jnp.zeros(K)
    phase_peak = peak0
    for _ in range(300):
        q, p = He.step(q, p)
        phase_peak = max(phase_peak, float(np.max(np.abs(np.asarray(q)))))

    assert leap_peak < 10.0 * peak0
    assert phase_peak < 10.0 * peak0


def test_leapfrog_centered_wave_recurrence():
    """m (q_{n+1} - 2 q_n + q_{n-1}) = kappa * Lap(q_n), force at the CENTER point."""
    K, m, kappa = 5, 1.5, 0.9
    H = _held(K, m, kappa)
    rng = np.random.default_rng(1)
    q, p = jnp.asarray(rng.standard_normal(K)), jnp.asarray(rng.standard_normal(K))

    traj = [np.asarray(q)]
    for _ in range(12):
        q, p = H.step(q, p)
        traj.append(np.asarray(q))
    a = np.stack(traj)
    for t in range(len(a) - 2):
        centered = m * (a[t + 2] - 2.0 * a[t + 1] + a[t])
        np.testing.assert_allclose(centered, kappa * _lap_pinned(a[t + 1]), atol=1e-10)


def test_leapfrog_energy_is_bounded():
    """Symplectic: the energy stays in a bounded band (here well within a factor of 2)."""
    K, m, kappa = 5, 1.5, 0.9
    chain = _held(K, m, kappa)
    rng = np.random.default_rng(2)
    q, p = jnp.asarray(rng.standard_normal(K)), jnp.asarray(rng.standard_normal(K))

    H0 = _energy(q, p, m, kappa)
    for _ in range(150):
        q, p = chain.step(q, p)
        H = _energy(q, p, m, kappa)
        assert 0.5 * H0 < H < 2.0 * H0


def test_regression_matches_cdlm():
    """The CDLM leapfrog trajectory (``regress/multistage.npz``) to rtol 1e-10."""
    R = load_regress("multistage")
    K, m, kappa = R["leap_params"]
    K = int(K)
    H = _held(K, m, kappa)
    q, p = jnp.asarray(R["leap_q"][0]), jnp.asarray(R["leap_p"][0])
    qs, ps = [R["leap_q"][0]], [R["leap_p"][0]]
    for _ in range(R["leap_q"].shape[0] - 1):
        q, p = H.step(q, p)
        qs.append(np.asarray(q)); ps.append(np.asarray(p))
    np.testing.assert_allclose(np.stack(qs), R["leap_q"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.stack(ps), R["leap_p"], rtol=1e-10, atol=1e-12)
