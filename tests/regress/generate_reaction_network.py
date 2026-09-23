"""Record CDLM (``dap-core``) reaction-network trajectories for the regression test.

Run from the OLD package, like ``generate_cdlm.py``:

    cd ../dynamic-algebra-potentials/dap-core && PYTHONPATH=. uv run --no-sync python ../../arrangements/tests/regress/generate_reaction_network.py

Writes ``reaction_network.npz``: the scenarios of the old
``test_reaction_network.py`` -- one-step Onsager (monolithic) steps on the
chain and the bimolecular network, the two convergence trajectories, the
compositional unimolecular steps, the Petri-net (extent) and log-Petri ticks
and long runs -- recording the inputs (replayed seeds, constants) and the
states the old package produces. Closed systems: no environment covector, so
the only sign-translated quantities are the composite potentials' gradients
(``grad_U_*``, CDLM's ``dU``), which the new package's forcing negates.
"""

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from dap.reaction_network import (
    crn_dynamics, crn_step, petri_arrangement, petri_dynamics, petri_initial_state,
    petri_log_arrangement, petri_log_dynamics, petri_log_initial_state, petri_step,
    unimolecular_arrangement, unimolecular_dynamics, unimolecular_stoichiometry,
)

assert "dap-core" in os.path.abspath(sys.modules["dap"].__file__), \
    "run this from dap-core with PYTHONPATH=. so that `dap` is the old package"

HERE = os.path.dirname(os.path.abspath(__file__))

CHAIN_EDGES = [(0, 1), (1, 2)]
CHAIN_KF = [2.0, 0.75]
CHAIN_KB = [1.0, 1.0]
CHAIN_XEQ = jnp.array([1.0, 2.0, 1.5])
CHAIN_ALPHA, CHAIN_BETA = unimolecular_stoichiometry(3, CHAIN_EDGES)

BI_ALPHA = jnp.array([[1.0, 1.0, 0.0]])
BI_BETA = jnp.array([[0.0, 0.0, 1.0]])
BI_KF = [2.0]
BI_KB = [1.0]
BI_XEQ = jnp.array([1.0, 0.5, 1.0])

PETRI_M = jnp.array([[1., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 1., 0.], [0., 0., 0., 2.], [1., 0., 0., 1.]])
PETRI_N = jnp.array([[0., 0., 1., 0.], [1., 1., 0., 0.], [0., 0., 0., 2.], [0., 0., 1., 0.], [0., 1., 0., 0.]])
PETRI_R = jnp.array([1.3, 0.7, 0.9, 0.4, 0.6])
PETRI_X0 = jnp.array([1.4, 0.8, 0.5, 1.1])
PETRI_T = int(PETRI_M.shape[0])


def _random_positive(rng, n, lo=0.2, hi=3.0):
    return jnp.asarray(rng.uniform(lo, hi, size=n))


def _one_steps(make_O, seed, dts, n=3):
    rng = np.random.default_rng(seed)
    xs, news = [], []
    for dt in dts:
        O = make_O(dt)
        for _ in range(5):
            x = _random_positive(rng, n)
            xs.append(x)
            news.append(crn_step(O, x))
    return np.stack(xs), np.stack(news)


def _run(O, s, T, step):
    traj = [s]
    for _ in range(T):
        s = step(O, s)
        traj.append(s)
    return np.stack([np.asarray(v) for v in traj])


def main():
    out = {}
    none = jnp.zeros(0)

    # (i) monolithic one-step (seeds 0, 1; dts as in the old test)
    out["mono_chain_dts"] = np.array([1.0, 0.05])
    out["mono_chain_x"], out["mono_chain_new"] = _one_steps(
        lambda dt: crn_dynamics(CHAIN_ALPHA, CHAIN_BETA, CHAIN_KF, CHAIN_KB, CHAIN_XEQ, dt), 0, (1.0, 0.05))
    out["mono_bi_dts"] = np.array([1.0, 0.02])
    out["mono_bi_x"], out["mono_bi_new"] = _one_steps(
        lambda dt: crn_dynamics(BI_ALPHA, BI_BETA, BI_KF, BI_KB, BI_XEQ, dt), 1, (1.0, 0.02))

    # (ii) convergence trajectories
    out["conv_chain_traj"] = _run(crn_dynamics(CHAIN_ALPHA, CHAIN_BETA, CHAIN_KF, CHAIN_KB, CHAIN_XEQ, 0.05),
                                  jnp.array([3.0, 0.2, 1.0]), 6000, crn_step)
    out["conv_bi_traj"] = _run(crn_dynamics(BI_ALPHA, BI_BETA, BI_KF, BI_KB, BI_XEQ, 0.02),
                               jnp.array([2.0, 1.2, 0.3]), 8000, crn_step)

    # (iv) compositional unimolecular: steps (seed 4) and the composite dU (seed 3, dt 0.1)
    out["uni_dts"] = np.array([1.0, 0.05])
    out["uni_x"], out["uni_new"] = _one_steps(
        lambda dt: unimolecular_dynamics(3, CHAIN_EDGES, CHAIN_KF, CHAIN_KB, CHAIN_XEQ, dt), 4, (1.0, 0.05))
    arr = unimolecular_arrangement(3, CHAIN_EDGES, CHAIN_KF, CHAIN_KB, CHAIN_XEQ, 0.1)
    rng = np.random.default_rng(3)
    xs = [_random_positive(rng, 3) for _ in range(5)]
    out["uni_grad_x"] = np.stack(xs)
    out["uni_grad_U"] = np.stack([jax.grad(lambda q: arr.U(q, none, none))(x) for x in xs])

    # 4. Petri (extent) encoding
    for dt in (1.0, 0.05, 0.001):
        out[f"petri_traj_{dt:g}"] = _run(petri_dynamics(PETRI_M, PETRI_N, PETRI_R, PETRI_X0, dt),
                                         petri_initial_state(PETRI_T), 6, petri_step)
    out["petri_long_traj"] = _run(petri_dynamics(PETRI_M, PETRI_N, PETRI_R, PETRI_X0, 0.01),
                                  petri_initial_state(PETRI_T), 300, petri_step)
    out["petri_epsp_traj"] = _run(petri_dynamics(PETRI_M, PETRI_N, PETRI_R, PETRI_X0, 0.01),
                                  petri_initial_state(PETRI_T, eps_prime=-3.7), 50, petri_step)
    arr = petri_arrangement(PETRI_M, PETRI_N, PETRI_R, PETRI_X0, dt=0.1)
    rng = np.random.default_rng(3)
    ss = [jnp.asarray(rng.uniform(-0.3, 0.3, size=2 * PETRI_T)) for _ in range(5)]
    out["petri_grad_s"] = np.stack(ss)
    out["petri_grad_U"] = np.stack([jax.grad(lambda q: arr.U(q, none, none))(s) for s in ss])

    # 5. log-Petri encoding
    for dt in (1.0, 0.05, 0.001):
        out[f"plog_traj_{dt:g}"] = _run(petri_log_dynamics(PETRI_M, PETRI_N, PETRI_R, dt),
                                        petri_log_initial_state(PETRI_X0), 6, petri_step)
    m1, n1, r1, x01 = jnp.array([[1.0]]), jnp.array([[0.0]]), jnp.array([3.0]), jnp.array([1.0])
    out["plog_stiff_traj"] = _run(petri_log_dynamics(m1, n1, r1, 0.5), petri_log_initial_state(x01), 100, petri_step)
    for dt in (0.05, 0.01):
        out[f"plog_conv_{dt:g}"] = _run(petri_log_dynamics(PETRI_M, PETRI_N, PETRI_R, dt),
                                        petri_log_initial_state(PETRI_X0), int(round(0.3 / dt)), petri_step)
    out["plog_yp_traj"] = _run(petri_log_dynamics(PETRI_M, PETRI_N, PETRI_R, 0.01),
                               petri_log_initial_state(PETRI_X0, y_prime=-3.7), 50, petri_step)
    arr = petri_log_arrangement(PETRI_M, PETRI_N, PETRI_R, dt=0.1)
    rng = np.random.default_rng(7)
    ss = []
    for _ in range(5):
        y = jnp.asarray(rng.uniform(-0.5, 0.5, size=4))
        yp = jnp.asarray(rng.uniform(-1.0, 1.0, size=4))
        ss.append(jnp.stack([y, yp], axis=1).reshape(-1))
    out["plog_grad_s"] = np.stack(ss)
    out["plog_grad_U"] = np.stack([jax.grad(lambda q: arr.U(q, none, none))(s) for s in ss])

    path = os.path.join(HERE, "reaction_network.npz")
    np.savez(path, **{k: np.asarray(v) for k, v in out.items()})
    print("wrote", path, sorted(out))


if __name__ == "__main__":
    main()
