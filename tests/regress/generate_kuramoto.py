"""Record CDLM (``dap-core``) Kuramoto trajectories for the regression test.

Run from the OLD package so that ``import dap`` is CDLM's:

    cd ../dynamic-algebra-potentials/dap-core && PYTHONPATH=. uv run --no-sync python ../../arrangements/tests/regress/generate_kuramoto.py

Writes ``kuramoto.npz``. The system is closed (no environment covector), so no
sign translation is needed on the inputs (STATUS.md §1 affects only covectors
crossing a port); the recorded *states* -- angles and momenta -- are what the
new package must reproduce. Covered dynamics: ``Phiconf`` one step on the
4-oscillator graph, ``Phiconf`` locking above threshold and drift below it on
the two-oscillator system, and ``Phiphase`` (the swing equation) on the same
two-oscillator data.
"""

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from dap.functors import Phiconf, Phiphase
from dap.interpretation import trivial_omega
from dap.kuramoto import kuramoto_arrangement

assert "dap-core" in os.path.abspath(sys.modules["dap"].__file__), \
    "run this from dap-core with PYTHONPATH=. so that `dap` is the old package"

HERE = os.path.dirname(os.path.abspath(__file__))
_IN_POS = (jnp.zeros(0), trivial_omega(0))
_TRIV = lambda _o: (jnp.zeros(0), jnp.zeros(0))

_N = 4
_EDGES = [(0, 1), (1, 2), (2, 3), (0, 2)]
_TWO_OMEGAS = (0.8, -0.2)


def _stepper(O):
    def f(s):
        _, _, s2 = O.with_state(s).run_one(_IN_POS, _TRIV)
        return s2

    return jax.jit(f)


def _save(name, **arrays):
    path = os.path.join(HERE, name + ".npz")
    np.savez(path, **{k: np.asarray(v) for k, v in arrays.items()})
    print("wrote", path, sorted(arrays))


def kuramoto():
    out = {}

    # --- one Phiconf step on the 4-oscillator graph (replayed seed 1).
    K, dt = 0.7, 0.07
    omegas = [0.9, -0.4, 0.2, 1.1]
    O = Phiconf(kuramoto_arrangement(omegas, _EDGES, K, sharp_scale=dt))
    rng = np.random.default_rng(1)
    ins, outs = [], []
    for _ in range(5):
        theta = rng.standard_normal(_N)
        _, _, new = O.with_state(jnp.asarray(theta)).run_one(_IN_POS, _TRIV)
        ins.append(theta)
        outs.append(np.asarray(new))
    out.update(step_params=np.array([K, dt]), step_omegas=np.array(omegas),
               step_edges=np.array(_EDGES), step_in=np.stack(ins), step_out=np.stack(outs))

    # --- two oscillators: locking above threshold, drift below (Phiconf).
    out["two_omegas"] = np.array(_TWO_OMEGAS)
    for tag, (Kc, dtc, steps) in {"lock": (0.8, 0.02, 5000), "drift": (0.3, 0.02, 3000)}.items():
        Oc = Phiconf(kuramoto_arrangement(list(_TWO_OMEGAS), [(0, 1)], Kc, sharp_scale=dtc))
        step = _stepper(Oc)
        s = jnp.asarray((0.1, 0.0))
        traj = np.empty((steps + 1, 2))
        traj[0] = np.asarray(s)
        for t in range(steps):
            s = step(s)
            traj[t + 1] = np.asarray(s)
        out[f"{tag}_params"] = np.array([Kc, dtc, steps])
        out[f"{tag}_traj"] = traj

    # --- Phiphase (swing equation) on the same two-oscillator data.
    Kp, m, steps = 1.0, 400.0, 2500
    Op = Phiphase(kuramoto_arrangement([0.0, 0.0], [(0, 1)], Kp, sharp_scale=1.0 / m))
    step = _stepper(Op)
    s = (jnp.array([0.5, -0.5]), jnp.zeros(2))
    qs = np.empty((steps + 1, 2))
    ps = np.empty((steps + 1, 2))
    qs[0], ps[0] = np.asarray(s[0]), np.asarray(s[1])
    for t in range(steps):
        s = step(s)
        qs[t + 1], ps[t + 1] = np.asarray(s[0]), np.asarray(s[1])
    out.update(swing_params=np.array([Kp, m, steps]), swing_q=qs, swing_p=ps)

    _save("kuramoto", **out)


if __name__ == "__main__":
    kuramoto()
