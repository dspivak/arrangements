"""Record CDLM (``dap-core-old``) replicator trajectories for the regression test.

Run from the OLD package, like ``generate_cdlm.py``:

    cd ../dynamic-algebra-potentials/dap-core-old && PYTHONPATH=. uv run --no-sync python ../../arrangements/tests/regress/generate_replicator.py

Writes ``replicator.npz``: for the scenarios of the old ``test_replicator.py`` --
the keystone one-step check (standard and log charts), the simplex-invariance
orbit, the coordination game, Hawk-Dove from two starts, and the standard/log
convergence run -- the *inputs* (payoff matrices and initial states, with the
seeds replayed here) and the *outputs* ``Phiconf`` produces. The system is
closed (no environment covector), so there is nothing to sign-translate: states
are identical under DESIGN.md §3.
"""

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from dap.replicator import log_replicator_dynamics, replicator_dynamics, replicator_step

assert "dap-core-old" in os.path.abspath(sys.modules["dap"].__file__), \
    "run this from dap-core-old with PYTHONPATH=. so that `dap` is the old package"

HERE = os.path.dirname(os.path.abspath(__file__))


def _random_symmetric(rng, n):
    B = rng.standard_normal((n, n))
    return jnp.asarray((B + B.T) / 2.0)


def _random_simplex(rng, n):
    return jnp.asarray(rng.dirichlet(np.ones(n)))


def _orbit(O, x, T):
    traj = [x]
    for _ in range(T):
        x = replicator_step(O, x)
        traj.append(x)
    return np.stack([np.asarray(v) for v in traj])


def replicator():
    out = {}

    # one step (test_one_step_is_euler_replicator, seed 0): n in (2, 3, 5), dt in (1, 0.05), 3 states
    rng = np.random.default_rng(0)
    for n in (2, 3, 5):
        A = _random_symmetric(rng, n)
        out[f"step_A_{n}"] = A
        for dt in (1.0, 0.05):
            O = replicator_dynamics(A, dt)
            xs, nxs = [], []
            for _ in range(3):
                x = _random_simplex(rng, n)
                xs.append(x)
                nxs.append(replicator_step(O, x))
            tag = f"{n}_{dt:g}"
            out[f"step_x_{tag}"] = np.stack(xs)
            out[f"step_new_{tag}"] = np.stack(nxs)

    # simplex invariance orbit (test_sharp_kernel_and_simplex_invariance, seed 1)
    rng = np.random.default_rng(1)
    n, dt = 3, 0.05
    _ = _random_simplex(rng, n)  # x_on (consumed by the sharp checks)
    A = _random_symmetric(rng, n) + 2.0 * jnp.eye(n) + 1.0
    x0 = _random_simplex(rng, n)
    out["inv_A"], out["inv_x0"], out["inv_dt"] = A, x0, np.array(dt)
    out["inv_traj"] = _orbit(replicator_dynamics(A, dt), x0, 2000)

    # coordination game (test_coordination_game_converges_to_dominant_pure_strategy)
    A = jnp.diag(jnp.array([3.0, 2.0, 1.0]))
    out["coord_A"] = A
    out["coord_traj"] = _orbit(replicator_dynamics(A, dt=0.1), jnp.ones(3) / 3.0, 2000)

    # Hawk-Dove (test_hawk_dove_converges_to_mixed_ess), two starts
    v, c = 2.0, 3.0
    A = jnp.array([[(v - c) / 2.0 + v, v], [v, v / 2.0]])
    out["hd_A"] = A
    O = replicator_dynamics(A, dt=0.1)
    for k, x0 in enumerate((jnp.array([0.15, 0.85]), jnp.array([0.9, 0.1]))):
        out[f"hd_traj_{k}"] = _orbit(O, x0, 3000)

    # log one step (test_log_one_step_is_multiplicative_weights, seed 11)
    rng = np.random.default_rng(11)
    As, xs, ys = [], [], []
    for dt in (1.0, 0.05):
        for _ in range(5):
            A = _random_symmetric(rng, 4)
            x = _random_simplex(rng, 4)
            O = log_replicator_dynamics(A, dt)
            As.append(A); xs.append(x); ys.append(replicator_step(O, jnp.log(x)))
    out["log_A"], out["log_x"], out["log_ynew"] = np.stack(As), np.stack(xs), np.stack(ys)
    out["log_dts"] = np.array([1.0] * 5 + [0.05] * 5)

    # standard and log ticks together (test_log_and_standard_ticks_converge_together, seed 13)
    rng = np.random.default_rng(13)
    A = _random_symmetric(rng, 3)
    x0 = _random_simplex(rng, 3)
    out["conv_A"], out["conv_x0"] = A, x0
    for dt in (0.05, 0.01):
        T = int(round(0.5 / dt))
        out[f"conv_std_{dt:g}"] = _orbit(replicator_dynamics(A, dt), x0, T)
        out[f"conv_log_{dt:g}"] = _orbit(log_replicator_dynamics(A, dt), jnp.log(x0), T)

    path = os.path.join(HERE, "replicator.npz")
    np.savez(path, **{k: np.asarray(v) for k, v in out.items()})
    print("wrote", path, sorted(out))


if __name__ == "__main__":
    replicator()
