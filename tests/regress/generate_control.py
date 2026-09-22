"""Record CDLM (``dap-core-old``) impedance-control trajectories for the regression test.

Run from the OLD package, like ``generate_cdlm.py``:

    cd ../dynamic-algebra-potentials/dap-core-old && PYTHONPATH=. uv run --no-sync python ../../arrangements/tests/regress/generate_control.py

Writes ``control.npz``: the scenarios of ``test_control.py`` -- the one-step
``Phiconf`` update on five replayed random draws, the step response with a mid-run
setpoint change, the ``Phiphase`` oscillation about a setpoint, and the emitted
covector field on the setpoint port in both dynamics. All in the old sign
convention: the setpoint port's emitted field is CDLM's ``omega_N = dU_c/ds``
(the new package publishes its negative) and ``xi_Q`` is the cost gradient the old
integrator subtracted (the new one adds its negative); states are identical.
"""

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from dap.control import impedance_arrangement, run_tracking
from dap.functors import Phiconf, Phiphase
from dap.interpretation import trivial_omega
from dap.rvect import diagonal, euclidean

assert "dap-core-old" in os.path.abspath(sys.modules["dap"].__file__), \
    "run this from dap-core-old with PYTHONPATH=. so that `dap` is the old package"

HERE = os.path.dirname(os.path.abspath(__file__))
_IN_POS = (jnp.zeros(0), trivial_omega(0))


def _one_step(O, state, s):
    in_dir = (jnp.zeros(1), jnp.array([float(s)]))
    return O.with_state(state).run_one(_IN_POS, lambda _op: in_dir)


def _save(name, **arrays):
    path = os.path.join(HERE, name + ".npz")
    np.savez(path, **{k: np.asarray(v) for k, v in arrays.items()})
    print("wrote", path, sorted(arrays))


def control():
    out = {}

    # 1. one Phiconf step on five replayed draws (seed 20260820)
    dt, k = 0.07, 1.3
    O = Phiconf(impedance_arrangement(euclidean(1, dt), k))
    rng = np.random.default_rng(20260820)
    q0s, ss, news, reported = [], [], [], []
    for _ in range(5):
        q0 = jnp.array([rng.standard_normal()])
        s = float(rng.standard_normal())
        out_pos, _out_dir, new_q = _one_step(O, q0, s)
        q0s.append(q0); ss.append(s); news.append(new_q); reported.append(out_pos[0])
    out.update(one_params=np.array([dt, k]), one_q0=np.stack(q0s), one_s=np.array(ss),
               one_new_q=np.stack(news), one_out=np.stack(reported))

    # 2. step response with a mid-run setpoint change
    dt2, k2, s1, s2, T = 0.05, 2.0, 1.5, -0.7, 120
    O2 = Phiconf(impedance_arrangement(euclidean(1, dt2), k2))
    _, states = run_tracking(O2, jnp.array([0.0]), [s1] * T + [s2] * T)
    out.update(track_params=np.array([dt2, k2, s1, s2, T]),
               track_q=np.array([float(st[0]) for st in states]))

    # 3. Phiphase oscillation about a setpoint
    m, k3, s_star, A, T3 = 1.0, 0.04, 0.5, 1.0, 1200
    O3 = Phiphase(impedance_arrangement(diagonal(jnp.array([1.0 / m])), k3))
    outs, states = run_tracking(O3, (jnp.array([s_star + A]), jnp.array([0.0])), [s_star] * T3)
    out.update(osc_params=np.array([m, k3, s_star, A, T3]),
               osc_out=np.array([float(o[0]) for o in outs]),
               osc_q=np.array([float(st[0][0]) for st in states]),
               osc_p=np.array([float(st[1][0]) for st in states]))

    # 4. the emitted covector field on the setpoint port, conf and phase
    dt4, k4, q0, s = 0.05, 2.0, 1.3, 0.4
    Oc = Phiconf(impedance_arrangement(euclidean(1, dt4), k4))
    out_pos, _out_dir, new_q = _one_step(Oc, jnp.array([q0]), s)
    out.update(fld_conf_params=np.array([dt4, k4, q0, s]),
               fld_conf_omega=np.asarray(out_pos[1](jnp.array([s]))),
               fld_conf_new_q=np.asarray(new_q))

    m5, k5, q05, p05, s5 = 0.8, 0.6, -0.9, 0.35, 0.25
    Op = Phiphase(impedance_arrangement(diagonal(jnp.array([1.0 / m5])), k5))
    out_pos, _out_dir, (nq, np_) = _one_step(Op, (jnp.array([q05]), jnp.array([p05])), s5)
    out.update(fld_phase_params=np.array([m5, k5, q05, p05, s5]),
               fld_phase_omega=np.asarray(out_pos[1](jnp.array([s5]))),
               fld_phase_new_q=np.asarray(nq), fld_phase_new_p=np.asarray(np_))

    _save("control", **out)


if __name__ == "__main__":
    control()
