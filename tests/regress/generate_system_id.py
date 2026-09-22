"""Record CDLM (``dap-core-old``) system-identification outputs for the regression test.

Run from the OLD package so that ``import dap`` is CDLM's:

    cd ../dynamic-algebra-potentials/dap-core-old && PYTHONPATH=. uv run --no-sync python ../../arrangements/tests/regress/generate_system_id.py

Writes ``system_id.npz``: the pendulum's closed phase trajectory (``Phiphase``)
from a fixed initial state, and the trained MLP weights and loss history after
a short ``identify`` run (``Phiconf`` on the learner). The training data are
consecutive-state pairs, so no covector crosses a port in the old sign; the new
test rebuilds the same run with the new package and compares states and
parameters directly (they are unchanged by DESIGN.md §3).
"""

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from dap.functors import Phiphase
from dap.system_id import PENDULUM, identify, oscillator, trajectory

assert "dap-core-old" in os.path.abspath(sys.modules["dap"].__file__), \
    "run this from dap-core-old with PYTHONPATH=. so that `dap` is the old package"

HERE = os.path.dirname(os.path.abspath(__file__))


def _save(name, **arrays):
    path = os.path.join(HERE, name + ".npz")
    np.savez(path, **{k: np.asarray(v) for k, v in arrays.items()})
    print("wrote", path, sorted(arrays))


def system_id():
    out = {}
    # the closed phase dynamics of the pendulum (m = 1.3, 40 steps)
    m, T = 1.3, 40
    O = Phiphase(oscillator(PENDULUM[0], PENDULUM[1], m))
    q0, p0 = jnp.array([0.8]), jnp.array([-0.4])
    traj = trajectory(O, q0, p0, T)
    out.update(traj_m=np.array(m), traj_T=np.array(T), traj_q0=q0, traj_p0=p0,
               traj_states=np.stack([np.asarray(s) for s in traj]))

    # a short identify run: seeds replayed, trained parameters after `steps` steps
    n_traj, T2, hidden, steps, seed = 4, 10, 8, 300, 1
    m2, ic_scale, eta = 1.0, 1.0, 6e-3
    params, _, hist, _, _, _ = identify(
        PENDULUM, m=m2, n_traj=n_traj, T=T2, ic_scale=ic_scale, hidden=hidden,
        eta=eta, steps=steps, seed=seed,
    )
    out.update(id_int_params=np.array([n_traj, T2, hidden, steps, seed]),
               id_float_params=np.array([m2, ic_scale, eta]),
               id_params=params, id_history=np.array(hist))
    _save("system_id", **out)


if __name__ == "__main__":
    system_id()
