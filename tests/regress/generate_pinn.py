"""Record CDLM (``dap-core``) deep-Ritz outputs for the regression test.

Run from the OLD package so that ``import dap`` is CDLM's:

    cd ../dynamic-algebra-potentials/dap-core && PYTHONPATH=. uv run --no-sync python ../../arrangements/tests/regress/generate_pinn.py

Writes ``pinn.npz``: the grid, the exact discrete Poisson solution, the learned
field and the per-step energy of a short ``solve_deep_ritz`` run (``Phiconf`` on
the field network, the environment seeding CDLM's ``dE/du = L u - b``). The new
test seeds the forcing ``b - L u`` instead (STATUS.md §1); the weights' trajectory,
hence the field and the energy history, are identical.
"""

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from dap.pinn import solve_deep_ritz

assert "dap-core" in os.path.abspath(sys.modules["dap"].__file__), \
    "run this from dap-core with PYTHONPATH=. so that `dap` is the old package"

HERE = os.path.dirname(os.path.abspath(__file__))


def _save(name, **arrays):
    path = os.path.join(HERE, name + ".npz")
    np.savez(path, **{k: np.asarray(v) for k, v in arrays.items()})
    print("wrote", path, sorted(arrays))


def pinn():
    V, hidden, steps, seed = 15, 8, 300, 0
    eta = 0.3
    u, u_star, hist, x = solve_deep_ritz(V, hidden=hidden, eta=eta, steps=steps, seed=seed)
    _save("pinn", int_params=np.array([V, hidden, steps, seed]), eta=np.array(eta),
          x=x, u_star=u_star, u=u, history=np.array(hist))


if __name__ == "__main__":
    pinn()
