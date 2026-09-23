"""Record CDLM (``dap-core``) trajectories for the multi-stage regression tests.

Run from the OLD package, like ``generate_cdlm.py``:

    cd ../dynamic-algebra-potentials/dap-core && PYTHONPATH=. uv run --no-sync python ../../arrangements/tests/regress/generate_multistage.py

Writes ``multistage.npz``: leapfrog on the pinned harmonic chain, RK4 on a
quadratic well, RK4 on the gyro phase system with drag and precession, and the
damped / gyro single-stage phase integrators on an oscillator. All in the old sign
convention (environment covectors are costs' gradients; the new tests negate them).
"""

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from dap.arrangement import SmoothArrangement
from dap.functors import Phi, Phirk4, Phirk4gyro
from dap.gyroscope import complex_structure
from dap.integrator import damped_phase_integrator, gyro_phase_integrator
from dap.interpretation import trivial_omega
from dap.leapfrog import Phileap
from dap.rvect import constant, diagonal, euclidean
from dap.wiring import compose_chain

assert "dap-core" in os.path.abspath(sys.modules["dap"].__file__)

HERE = os.path.dirname(os.path.abspath(__file__))
_IN_POS = (jnp.zeros(0), trivial_omega(0))
_TRIV = lambda op: (jnp.zeros(0), jnp.zeros(0))


def _harmonic(m, kappa):
    return SmoothArrangement(
        diagonal(jnp.array([1.0 / m])), 0, 0, 1, 1,
        out_f=lambda q, m_out: q, in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: 0.5 * kappa * (q[0] - n_in[0]) ** 2, label="Part")


def _well(A):
    A = jnp.asarray(A)
    return SmoothArrangement(
        euclidean(A.shape[0], 1.0), 0, 0, 0, 0,
        out_f=lambda q, m_out: jnp.zeros(0), in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: 0.5 * q @ (A @ q), label="well")


def _oscillator(d, m, k):
    return SmoothArrangement(
        constant(jnp.eye(d) / m), 0, 0, 0, 0,
        out_f=lambda q, mo: jnp.zeros(0), in_f=lambda q, mo, ni: jnp.zeros(0),
        U=lambda q, mo, ni: 0.5 * k * jnp.sum(q ** 2), label="osc")


def main():
    out = {}
    # leapfrog: pinned chain, seed 1, 12 steps (test_leapfrog_centered_wave_recurrence)
    K, m, kappa = 5, 1.5, 0.9
    O = Phileap(compose_chain([_harmonic(m, kappa)] * K))
    rng = np.random.default_rng(1)
    state = (jnp.asarray(rng.standard_normal(K)), jnp.asarray(rng.standard_normal(K)))
    bdy = lambda out_pos: (kappa * out_pos[0], jnp.array([0.0]))
    qs, ps = [state[0]], [state[1]]
    for _ in range(12):
        _, _, state = O.with_state(state).run_one(_IN_POS, bdy)
        qs.append(state[0]); ps.append(state[1])
    out.update(leap_q=np.stack(qs), leap_p=np.stack(ps), leap_params=np.array([K, m, kappa]))

    # rk4: quadratic well, 25 steps of h = 0.1
    A = np.array([[1.0, 0.2, 0.0], [0.2, 3.0, 0.1], [0.0, 0.1, 2.0]])
    O = Phirk4(_well(jnp.asarray(A)), 0.1)
    x = jnp.array([0.6, -0.9, 0.3])
    xs = [x]
    for _ in range(25):
        _, x = O.with_state(x).run_one(_IN_POS, _TRIV)
        xs.append(x)
    out.update(rk4_A=A, rk4_x=np.stack(xs), rk4_h=np.array(0.1))

    # rk4_gyro: oscillator on R^4 with drag and per-gyro precession, 10 steps
    V, mm, k, h = 2, 1.0, 0.5, 0.1
    J = complex_structure(V)
    gamma = jnp.array([0.4, 0.4, 0.0, 0.0])
    O = Phirk4gyro(_oscillator(2 * V, mm, k), h, drag=0.3, gamma=gamma, J=J, gyro_block=2)
    s = (jnp.array([0.3, -0.2, 0.1, 0.5]), jnp.array([0.2, 0.1, -0.3, 0.4]))
    qs, ps = [s[0]], [s[1]]
    for _ in range(10):
        _, s = O.with_state(s).run_one(_IN_POS, _TRIV)
        qs.append(s[0]); ps.append(s[1])
    out.update(gyro4_q=np.stack(qs), gyro4_p=np.stack(ps), gyro4_J=np.asarray(J), gyro4_gamma=np.asarray(gamma),
               gyro4_params=np.array([V, mm, k, h, 0.3]))

    # damped and gyro single-stage phase integrators on the same oscillator, 30 steps
    arr = _oscillator(2 * V, mm, k)
    for tag, intg in (("damped", damped_phase_integrator(0.1)),
                      ("gyro", gyro_phase_integrator(damping=0.1, gamma=gamma, J=J, drag=0.3, gyro_block=2))):
        O = Phi(arr, intg)
        s = (jnp.array([0.3, -0.2, 0.1, 0.5]), jnp.array([0.2, 0.1, -0.3, 0.4]))
        qs, ps = [s[0]], [s[1]]
        for _ in range(30):
            _, _, s = O.with_state(s).run_one(_IN_POS, _TRIV)
            qs.append(s[0]); ps.append(s[1])
        out[f"{tag}_q"] = np.stack(qs); out[f"{tag}_p"] = np.stack(ps)

    path = os.path.join(HERE, "multistage.npz")
    np.savez(path, **{k: np.asarray(v) for k, v in out.items()})
    print("wrote", path, sorted(out))


if __name__ == "__main__":
    main()
