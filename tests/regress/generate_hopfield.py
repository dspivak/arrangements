"""Record CDLM (``dap-core``) Hopfield trajectories for the regression test (STATUS.md §1).

Run from the OLD package so that ``import dap`` is CDLM's:

    cd ../dynamic-algebra-potentials/dap-core && PYTHONPATH=. uv run --no-sync python ../../arrangements/tests/regress/generate_hopfield.py

Writes ``hopfield.npz`` into this directory: the inputs the old test file
(``test_hopfield.py``) drew (seeds replayed, so the new test need not reproduce
the draw order) and the outputs the old package produced. The Hopfield
arrangement is closed, so no environment covector crosses a port and the
states are the only outputs; they are identical in the two sign conventions.
"""

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from dap.functors import Phiconf
from dap.hopfield import hebbian_weights, hopfield_arrangement
from dap.interpretation import trivial_omega

assert "dap-core" in os.path.abspath(sys.modules["dap"].__file__), \
    "run this from dap-core with PYTHONPATH=. so that `dap` is the old package"

HERE = os.path.dirname(os.path.abspath(__file__))
_IN_POS = (jnp.zeros(0), trivial_omega(0))
_TRIV = lambda _o: (jnp.zeros(0), jnp.zeros(0))


def _stepper(O):
    def f(s):
        _, _, s2 = O.with_state(s).run_one(_IN_POS, _TRIV)
        return s2
    return jax.jit(f)


def _random_symmetric(rng, N):
    A = rng.standard_normal((N, N))
    W = (A + A.T) / 2.0
    np.fill_diagonal(W, 0.0)
    return W


def _save(name, **arrays):
    path = os.path.join(HERE, name + ".npz")
    np.savez(path, **{k: np.asarray(v) for k, v in arrays.items()})
    print("wrote", path, sorted(arrays))


def hopfield():
    out = {}

    # one-step exactness (test_phiconf_one_step_is_euler_hopfield): N=4, dt=0.07, seed 1
    N, dt = 4, 0.07
    rng = np.random.default_rng(1)
    W = _random_symmetric(rng, N)
    I = rng.standard_normal(N)
    O = Phiconf(hopfield_arrangement(W, I, dt))
    us, news = [], []
    for _ in range(5):
        u = rng.standard_normal(N) * 1.5
        _, _, new = O.with_state(jnp.asarray(u)).run_one(_IN_POS, _TRIV)
        us.append(u); news.append(new)
    out.update(step_W=W, step_I=I, step_dt=np.array(dt), step_u=np.stack(us), step_new=np.stack(news))

    # energy trajectory (test_energy_decreases_along_trajectory): N=8, dt=0.05, 1500 steps, seed 11
    N, dt, steps = 8, 0.05, 1500
    rng = np.random.default_rng(11)
    W = _random_symmetric(rng, N)
    I = 0.3 * rng.standard_normal(N)
    step = _stepper(Phiconf(hopfield_arrangement(W, I, dt)))
    u = jnp.asarray(rng.standard_normal(N) * 1.5)
    traj = [u]
    for _ in range(steps):
        u = step(u)
        traj.append(u)
    out.update(traj_W=W, traj_I=I, traj_dt=np.array(dt), traj_u=np.stack([np.asarray(x) for x in traj]))

    # associative memory (test_associative_memory_retrieval): seed 0, 10 trials of 600 steps
    _N, _P, _GAIN, _DT, _STEPS, _FLIPS = 40, 3, 2.0, 0.1, 600, 7
    rng = np.random.default_rng(0)
    patterns = rng.choice([-1.0, 1.0], size=(_P, _N))
    W = hebbian_weights(patterns, gain=_GAIN)
    step = _stepper(Phiconf(hopfield_arrangement(W, np.zeros(_N), _DT)))
    u0s, finals, targets = [], [], []
    for _ in range(10):
        xi = patterns[rng.integers(_P)].copy()
        u0 = xi.copy()
        u0[rng.choice(_N, size=_FLIPS, replace=False)] *= -1
        s = jnp.asarray(u0)
        for _ in range(_STEPS):
            s = step(s)
        u0s.append(u0); finals.append(np.asarray(s)); targets.append(xi)
    out.update(mem_patterns=patterns, mem_gain=np.array(_GAIN), mem_dt=np.array(_DT), mem_steps=np.array(_STEPS),
               mem_u0=np.stack(u0s), mem_final=np.stack(finals), mem_target=np.stack(targets))
    _save("hopfield", **out)


if __name__ == "__main__":
    hopfield()
