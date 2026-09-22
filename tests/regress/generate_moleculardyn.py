"""Record CDLM (``dap-core-old``) molecular-dynamics trajectories for the regression test.

Run from the OLD package, like ``generate_cdlm.py``:

    cd ../dynamic-algebra-potentials/dap-core-old && PYTHONPATH=. uv run --no-sync python ../../arrangements/tests/regress/generate_moleculardyn.py

Writes ``moleculardyn.npz``: the scenarios of ``test_moleculardyn.py`` -- the LJ force
covector at three separations, the 1-D pair under ``Phileap``, one ``Phileap``
macro-tick on a 2-D pair, the 2-D trimer under ``Phileap`` and ``Phiphase`` (with
the energies), the 1-D and 3-D pairs relaxing under ``Phiconf``. The MD arrangement
is closed (0-ary), so no environment covector crosses a port: the recorded force is
CDLM's ``-xi_Q`` (the new package's ``xi_Q`` is the force itself, DESIGN.md §3);
states are identical in both conventions.
"""

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from dap.functors import Phiconf, Phiphase
from dap.interpretation import trivial_omega
from dap.leapfrog import Phileap
from dap.moleculardyn import md_arrangement, md_energy, md_force

assert "dap-core-old" in os.path.abspath(sys.modules["dap"].__file__), \
    "run this from dap-core-old with PYTHONPATH=. so that `dap` is the old package"

HERE = os.path.dirname(os.path.abspath(__file__))
_IN_POS = (jnp.zeros(0), trivial_omega(0))
_TRIV = lambda _o: (jnp.zeros(0), jnp.zeros(0))

EPS, SIGMA, MASS = 1.0, 1.0, 1.0
R0 = 2.0 ** (1.0 / 6.0)
DT = 0.005


def _run(morphism, state, n_steps):
    def one(s):
        *_, new = morphism.with_state(s).run_one(_IN_POS, _TRIV)
        return new

    step = jax.jit(one)
    out = [state]
    for _ in range(n_steps):
        state = step(state)
        out.append(state)
    return out


def _save(name, **arrays):
    path = os.path.join(HERE, name + ".npz")
    np.savez(path, **{k: np.asarray(v) for k, v in arrays.items()})
    print("wrote", path, sorted(arrays))


def moleculardyn():
    out = {"constants": np.array([EPS, SIGMA, MASS, DT])}

    # forces at R0, 0.99 R0, 1.01 R0 (1-D pair), CDLM's -xi_Q
    pair1 = md_arrangement(2, [(0, 1)], MASS, EPS, SIGMA, DT, dim=1)
    seps = np.array([R0, 0.99 * R0, 1.01 * R0])
    out["force_sep"] = seps
    out["force"] = np.stack([np.asarray(md_force(pair1, jnp.array([0.0, r]))) for r in seps])

    # 1-D pair under Phileap from R0 + 0.02, 2000 steps
    delta = 0.02
    states = _run(Phileap(pair1), (jnp.array([0.0, R0 + delta]), jnp.zeros(2)), 2000)
    out["pair_delta"] = np.array(delta)
    out["pair_q"] = np.stack([np.asarray(s[0]) for s in states])
    out["pair_xi"] = np.stack([np.asarray(s[1]) for s in states])

    # one Phileap macro-tick on a 2-D pair
    pair2 = md_arrangement(2, [(0, 1)], MASS, EPS, SIGMA, DT, dim=2)
    q0 = np.array([0.0, 0.0, 1.25, 0.3])
    v0 = np.array([0.1, -0.2, 0.05, 0.15])
    _, _, (q_new, xi_new) = Phileap(pair2).with_state(
        (jnp.asarray(q0), (MASS / DT) * jnp.asarray(v0))
    ).run_one(_IN_POS, _TRIV)
    out.update(verlet_q0=q0, verlet_v0=v0, verlet_q1=np.asarray(q_new), verlet_xi1=np.asarray(xi_new))

    # 2-D trimer under Phileap and Phiphase, 1000 steps, with energies
    pairs = [(0, 1), (0, 2), (1, 2)]
    tri_arr = md_arrangement(3, pairs, MASS, EPS, SIGMA, DT, dim=2)
    tri = R0 * np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
    rng = np.random.default_rng(0)
    tq0 = jnp.asarray((tri + 0.005 * rng.standard_normal(tri.shape)).reshape(-1))
    out["trimer_q0"] = np.asarray(tq0)
    energy = jax.jit(lambda s: md_energy(tri_arr, s[0], s[1]))
    for tag, O in (("leap", Phileap(tri_arr)), ("phase", Phiphase(tri_arr))):
        states = _run(O, (tq0, jnp.zeros(6)), 1000)
        out[f"trimer_{tag}_q"] = np.stack([np.asarray(s[0]) for s in states])
        out[f"trimer_{tag}_xi"] = np.stack([np.asarray(s[1]) for s in states])
        out[f"trimer_{tag}_H"] = np.array([float(energy(s)) for s in states])

    # Phiconf relaxation of the 1-D pair from 1.05 R0 and 0.97 R0, 2000 steps each
    O = Phiconf(pair1)
    starts = np.array([1.05 * R0, 0.97 * R0])
    out["conf_starts"] = starts
    for i, r_start in enumerate(starts):
        states = _run(O, jnp.array([0.0, r_start]), 2000)
        out[f"conf1_{i}_q"] = np.stack([np.asarray(s) for s in states])

    # Phiconf relaxation of the 3-D pair off-axis, 2000 steps each
    axis = np.ones(3) / np.sqrt(3.0)
    pair3 = md_arrangement(2, [(0, 1)], MASS, EPS, SIGMA, DT, dim=3)
    O3 = Phiconf(pair3)
    for i, r_start in enumerate(starts):
        q0 = jnp.asarray(np.concatenate([np.zeros(3), r_start * axis]))
        states = _run(O3, q0, 2000)
        out[f"conf3_{i}_q"] = np.stack([np.asarray(s) for s in states])

    _save("moleculardyn", **out)


if __name__ == "__main__":
    moleculardyn()
