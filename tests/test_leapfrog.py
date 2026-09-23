"""Leapfrog as a two-stage coalgebra, dynamics in pc^(2) (leapfrog.py; cf. rmk.multistage).

Same harmonic chain as the wave test, with the particles' parameter the symplectic
two-stage coalgebra ``leapfrog(Q)`` (velocity Verlet) -- a third arrangement on the
one chain wiring, next to heat (``configuration``) and wave (``phase``):
* eliminating momentum gives the centered discrete wave recurrence
      m (q_{n+1} - 2 q_n + q_{n-1}) = kappa * Lap(q_n);   [force at the center]
* the energy stays in a bounded band (symplectic).

The single-stage phase coalgebra is itself symplectic (presented-position
readout), so it stays bounded too; leapfrog is a higher-order alternative.
The boundary covector is a forcing: pinning ``q_{K+1} = 0`` is ``xi_N = -kappa q_K``
(CDLM's ``+kappa q_K`` with a subtracting integrator; STATUS.md §1).
"""

import jax.numpy as jnp
import numpy as np

from arrangements.coalgebra import phase
from arrangements.functors import Phi
from arrangements.leapfrog import leapfrog
from arrangements.pc2 import Phi2
from arrangements.wiring import compose_chain

from tests.chain import IN_POS, harmonic_particle, load_regress, phase_flat, phase_state


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


def test_leapfrog_and_phase_are_both_bounded():
    """The presented-position readout makes the single-stage phase coalgebra
    symplectic, so it stays bounded like leapfrog (kappa/m = 0.6)."""
    K, m, kappa = 5, 1.5, 0.9
    rng = np.random.default_rng(0)
    q0 = jnp.asarray(rng.standard_normal(K))
    peak0 = float(np.max(np.abs(np.asarray(q0))))
    bdy = _boundary(kappa)

    O = _leap_chain(K, m, kappa)
    state = phase_state(q0, jnp.zeros(K))
    leap_peak = peak0
    for _ in range(300):
        _, _, state = O.with_state(state).run_one(IN_POS, bdy)
        leap_peak = max(leap_peak, float(np.max(np.abs(np.asarray(phase_flat(state, K)[0])))))

    Oe = Phi(compose_chain([harmonic_particle(phase, m, kappa)] * K))
    es = phase_state(q0, jnp.zeros(K))
    phase_peak = peak0
    for _ in range(300):
        _, _, es = Oe.with_state(es).run_one(IN_POS, bdy)
        phase_peak = max(phase_peak, float(np.max(np.abs(np.asarray(phase_flat(es, K)[0])))))

    assert leap_peak < 10.0 * peak0
    assert phase_peak < 10.0 * peak0


def test_leapfrog_centered_wave_recurrence():
    """m (q_{n+1} - 2 q_n + q_{n-1}) = kappa * Lap(q_n), force at the CENTER point."""
    K, m, kappa = 5, 1.5, 0.9
    O = _leap_chain(K, m, kappa)
    rng = np.random.default_rng(1)
    state = phase_state(jnp.asarray(rng.standard_normal(K)), jnp.asarray(rng.standard_normal(K)))
    bdy = _boundary(kappa)

    traj = [np.asarray(phase_flat(state, K)[0])]
    for _ in range(12):
        _, _, state = O.with_state(state).run_one(IN_POS, bdy)
        traj.append(np.asarray(phase_flat(state, K)[0]))
    a = np.stack(traj)
    for t in range(len(a) - 2):
        centered = m * (a[t + 2] - 2.0 * a[t + 1] + a[t])
        np.testing.assert_allclose(centered, kappa * _lap_pinned(a[t + 1]), atol=1e-10)


def test_leapfrog_energy_is_bounded():
    """Symplectic: the energy stays in a bounded band (here well within a factor of 2)."""
    K, m, kappa = 5, 1.5, 0.9
    O = _leap_chain(K, m, kappa)
    rng = np.random.default_rng(2)
    state = phase_state(jnp.asarray(rng.standard_normal(K)), jnp.asarray(rng.standard_normal(K)))
    bdy = _boundary(kappa)

    H0 = _energy(*phase_flat(state, K), m, kappa)
    for _ in range(150):
        _, _, state = O.with_state(state).run_one(IN_POS, bdy)
        H = _energy(*phase_flat(state, K), m, kappa)
        assert 0.5 * H0 < H < 2.0 * H0


def test_regression_matches_cdlm():
    """The CDLM leapfrog trajectory (``regress/multistage.npz``) to rtol 1e-10."""
    R = load_regress("multistage")
    K, m, kappa = R["leap_params"]
    K = int(K)
    O = _leap_chain(K, m, kappa)
    state = phase_state(jnp.asarray(R["leap_q"][0]), jnp.asarray(R["leap_p"][0]))
    bdy = _boundary(kappa)
    qs, ps = [R["leap_q"][0]], [R["leap_p"][0]]
    for _ in range(R["leap_q"].shape[0] - 1):
        _, _, state = O.with_state(state).run_one(IN_POS, bdy)
        q, p = phase_flat(state, K)
        qs.append(np.asarray(q)); ps.append(np.asarray(p))
    np.testing.assert_allclose(np.stack(qs), R["leap_q"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.stack(ps), R["leap_p"], rtol=1e-10, atol=1e-12)
