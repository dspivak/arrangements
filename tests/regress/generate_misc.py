"""Record CDLM (``dap-core``) trajectories for the module-less regression tests.

Run from the OLD package so that ``import dap`` is CDLM's:

    cd ../dynamic-algebra-potentials/dap-core && PYTHONPATH=. uv run --no-sync python ../../arrangements/tests/regress/generate_misc.py

Writes ``misc.npz``, covering the four old test files that use only the core
(``test_newton.py``, ``test_optimizers.py``, ``test_klein_gordon.py``,
``test_graph_laplacian.py``): Newton orbits (position-dependent inverse-Hessian
sharp), the three optimizer trajectories of one quadratic well (``Phiconf``,
``Phiphase``, ``Phidamped``), the Klein-Gordon chain's pinned phase trajectory
and its one-step samples, and the graph-Laplacian wave and heat trajectories.

Everything recorded is in the old sign convention: the potentials ``U`` are costs
and the integrators subtract, and the environment covector ``xi_N`` seeded for the
pinned Klein-Gordon chain is CDLM's ``+kappa * q~_K``. The new tests seed
``-kappa * q~_K`` and build ``omega = exact(-U)``; STATES (the trajectories here)
are unchanged (STATUS.md §1).
"""

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from dap.arrangement import SmoothArrangement
from dap.functors import Phiconf, Phidamped, Phiphase
from dap.interpretation import trivial_omega
from dap.rvect import diagonal, euclidean, inverse_hessian
from dap.wiring import compose_chain, compose_graph

assert "dap-core" in os.path.abspath(sys.modules["dap"].__file__), \
    "run this from dap-core with PYTHONPATH=. so that `dap` is the old package"

HERE = os.path.dirname(os.path.abspath(__file__))
_IN_POS = (jnp.zeros(0), trivial_omega(0))
_TRIV = lambda _o: (jnp.zeros(0), jnp.zeros(0))


def _save(name, **arrays):
    path = os.path.join(HERE, name + ".npz")
    np.savez(path, **{k: np.asarray(v) for k, v in arrays.items()})
    print("wrote", path, sorted(arrays))


# ---------------------------------------------------------------------------
# Newton (test_newton.py): the inverse-Hessian sharp, one step and the orbit.
# ---------------------------------------------------------------------------


def _scalar(U_scalar, dim):
    return SmoothArrangement(
        Q=inverse_hessian(U_scalar, dim), out_dim_M=0, in_dim_M=0, out_dim_N=0, in_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: U_scalar(q),
        label="newton",
    )


def _newton_orbit(U_scalar, q0, steps):
    O = Phiconf(_scalar(U_scalar, 1))
    q = jnp.asarray(q0, float)
    out = [q]
    for _ in range(steps):
        _, _, q = O.with_state(q).run_one(_IN_POS, _TRIV)
        out.append(q)
    return np.stack([np.asarray(x) for x in out])


def newton(out):
    a = 2.0
    out["newton_quad"] = _newton_orbit(lambda q: (q[0] - a) ** 2 + 1.0, jnp.array([5.0]), 3)
    out["newton_quad_a"] = np.array(a)
    out["newton_exp"] = _newton_orbit(lambda q: jnp.exp(q[0]) - q[0], jnp.array([2.0]), 40)


# ---------------------------------------------------------------------------
# Optimizers (test_optimizers.py): one quadratic well, three integrators.
# ---------------------------------------------------------------------------


def _quadratic_well(A, eta=1.0):
    A = jnp.asarray(A, float)
    d = int(A.shape[0])
    return SmoothArrangement(
        euclidean(d, eta), 0, 0, 0, 0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: 0.5 * q @ (A @ q),
        label="quadratic_well",
    )


def optimizers(out):
    A = jnp.diag(jnp.array([1.0, 3.0, 9.0]))
    q0 = jnp.array([1.0, 1.0, 1.0])
    eta, damping, steps = 0.1, 0.15, 600
    arr = _quadratic_well(A, eta=eta)

    q = q0
    conf = [np.asarray(q)]
    Oc = Phiconf(arr)
    for _ in range(steps):
        _, _, q = Oc.with_state(q).run_one(_IN_POS, _TRIV)
        conf.append(np.asarray(q))

    for tag, O in (("phase", Phiphase(arr)), ("damped", Phidamped(arr, damping))):
        state = (q0, jnp.zeros(3))
        qs, ps = [np.asarray(state[0])], [np.asarray(state[1])]
        for _ in range(steps):
            _, _, state = O.with_state(state).run_one(_IN_POS, _TRIV)
            qs.append(np.asarray(state[0])); ps.append(np.asarray(state[1]))
        out[f"opt_{tag}_q"] = np.stack(qs)
        out[f"opt_{tag}_p"] = np.stack(ps)

    out["opt_conf_q"] = np.stack(conf)
    out["opt_diagA"] = np.asarray(jnp.diag(A))
    out["opt_params"] = np.array([eta, damping, steps])


# ---------------------------------------------------------------------------
# Klein-Gordon (test_klein_gordon.py): one-step samples and the pinned trajectory.
# ---------------------------------------------------------------------------


def _kg_particle(m, kappa, m0):
    return SmoothArrangement(
        Q=diagonal(jnp.array([1.0 / m])), out_dim_M=0, in_dim_M=0, out_dim_N=1, in_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: 0.5 * kappa * (q[0] - n_in[0]) ** 2 + 0.5 * (m0 * m0) * q[0] * q[0],
        label="KGPart",
    )


def klein_gordon(out):
    # one-step samples on random (q, p) with random external q_prev, xi_N (CDLM's sign)
    K, m, kappa, m0 = 5, 1.7, 2.3, 0.4
    O = Phiphase(compose_chain([_kg_particle(m, kappa, m0)] * K))
    rng = np.random.default_rng(20260508)
    rows = {k: [] for k in ("q", "p", "qprev", "xiN", "nq", "np")}
    for _ in range(5):
        q = jnp.asarray(rng.standard_normal(K))
        p = jnp.asarray(rng.standard_normal(K))
        q_prev = float(rng.standard_normal())
        xi_N = float(rng.standard_normal())
        in_dir = (jnp.array([xi_N]), jnp.array([q_prev]))
        _, _, (new_q, new_p) = O.with_state((q, p)).run_one(_IN_POS, lambda _o: in_dir)
        rows["q"].append(q); rows["p"].append(p); rows["qprev"].append(q_prev)
        rows["xiN"].append(xi_N); rows["nq"].append(new_q); rows["np"].append(new_p)
    for k, v in rows.items():
        out[f"kg_{k}"] = np.stack([np.asarray(x) for x in v])
    out["kg_step_params"] = np.array([K, m, kappa, m0])

    # pinned trajectory: xi_N = kappa * q~_K (CDLM's sign), K = 5, T + 2 = 12 steps
    K2, T2, m2, kappa2, m02 = 5, 10, 1.5, 0.9, 0.3
    O2 = Phiphase(compose_chain([_kg_particle(m2, kappa2, m02)] * K2))
    rng2 = np.random.default_rng(20260508)
    state = (jnp.asarray(rng2.standard_normal(K2)), jnp.zeros(K2))
    q_traj, p_traj = [state[0]], [state[1]]
    for _ in range(T2 + 2):
        q, p = state
        xi_N = jnp.array([kappa2 * float(q[K2 - 1] + p[K2 - 1] / m2)])
        _, _, state = O2.with_state(state).run_one(_IN_POS, lambda _o: (xi_N, jnp.array([0.0])))
        q_traj.append(state[0]); p_traj.append(state[1])
    out["kg_pin_q"] = np.stack([np.asarray(x) for x in q_traj])
    out["kg_pin_p"] = np.stack([np.asarray(x) for x in p_traj])
    out["kg_pin_params"] = np.array([K2, T2, m2, kappa2, m02])


# ---------------------------------------------------------------------------
# Graph Laplacian (test_graph_laplacian.py): the wave and heat trajectories.
# ---------------------------------------------------------------------------


V = 3
EDGES = [(0, 1), (1, 2), (2, 0), (0, 2)]


def graph_laplacian(out):
    m, kappa, T = 1.5, 0.9, 12
    O = Phiphase(compose_graph(V, EDGES, m, kappa))
    rng = np.random.default_rng(42)
    state = (jnp.asarray(rng.standard_normal(V)), jnp.zeros(V))
    qs, ps = [np.asarray(state[0])], [np.asarray(state[1])]
    for _ in range(T):
        _, _, state = O.with_state(state).run_one(_IN_POS, _TRIV)
        qs.append(np.asarray(state[0])); ps.append(np.asarray(state[1]))
    out.update(gl_wave_q=np.stack(qs), gl_wave_p=np.stack(ps),
               gl_wave_params=np.array([m, kappa, T]))

    m2, kappa2, T2 = 1.0, 0.2, 50
    O2 = Phiconf(compose_graph(V, EDGES, m2, kappa2))
    q = jnp.asarray(np.random.default_rng(7).standard_normal(V))
    traj = [np.asarray(q)]
    for _ in range(T2):
        _, _, q = O2.with_state(q).run_one(_IN_POS, _TRIV)
        traj.append(np.asarray(q))
    out.update(gl_heat_q=np.stack(traj), gl_heat_params=np.array([m2, kappa2, T2]))


if __name__ == "__main__":
    data = {}
    newton(data); optimizers(data); klein_gordon(data); graph_laplacian(data)
    _save("misc", **data)
