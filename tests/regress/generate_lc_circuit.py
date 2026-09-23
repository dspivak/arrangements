"""Record CDLM (``dap-core``) LC-circuit trajectories for the regression test.

Run from the OLD package so that ``import dap`` is CDLM's:

    cd ../dynamic-algebra-potentials/dap-core && PYTHONPATH=. uv run --no-sync python ../../arrangements/tests/regress/generate_lc_circuit.py

Writes ``lc_circuit.npz``. Each array records either an *input* the old test
supplied (constants, replayed seeds, drive frequencies) or an *output* the old
package produced, in the old sign convention: the environment covector ``xi_N``
is CDLM's (the integrator subtracts it). The new test translates per
STATUS.md §1 -- ``xi_N_new = -xi_N_old`` (the radio drive), states identical.

Covered dynamics: ``Phiphase`` on ``lc_cell`` (three (L,C) pairs), ``Phileap``
on ``lc_cell``, ``Phiconf`` on ``lc_cell`` (RC decay), ``Phiphase`` on the
coupled pair (both normal modes), ``Phiconf`` on ``lc_conf`` (one-tick states,
the energy-law trajectory, and the driven radio sweep).
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
from dap.lc_circuit import coupled_lc_pair, lc_cell, lc_conf

assert "dap-core" in os.path.abspath(sys.modules["dap"].__file__), \
    "run this from dap-core with PYTHONPATH=. so that `dap` is the old package"

HERE = os.path.dirname(os.path.abspath(__file__))
_IN_POS = (jnp.zeros(0), trivial_omega(0))
_DIR_CELL = lambda _o: (jnp.zeros(1), jnp.zeros(0))
_DIR_PAIR = lambda _o: (jnp.zeros(0), jnp.zeros(0))
_DIR_CONF = lambda _o: (jnp.zeros(1), jnp.zeros(0))


def _run(O, q0, xi0, steps, dir_from):
    state = (jnp.asarray(q0, float), jnp.asarray(xi0, float))
    qs, xis = [state[0]], [state[1]]
    for _ in range(steps):
        _, _, state = O.with_state(state).run_one(_IN_POS, dir_from)
        qs.append(state[0])
        xis.append(state[1])
    return np.stack([np.asarray(x) for x in qs]), np.stack([np.asarray(x) for x in xis])


def _save(name, **arrays):
    path = os.path.join(HERE, name + ".npz")
    np.savez(path, **{k: np.asarray(v) for k, v in arrays.items()})
    print("wrote", path, sorted(arrays))


def lc_circuit():
    out = {}

    # --- Phiphase on the single cell: the three (L, C) pairs of test 1, and the
    # 300-step flux-linkage run of test 2 (L = C = 20 is the first pair, run long).
    pairs = [(20.0, 20.0), (80.0, 20.0), (100.0, 100.0)]
    out["phase_LC"] = np.array(pairs)
    steps = []
    for k, (L, C) in enumerate(pairs):
        w = 1.0 / np.sqrt(L * C)
        n = int(5.2 * 2.0 * np.pi / w)
        qs, xis = _run(Phiphase(lc_cell(L, C)), [1.0], [0.0], n, _DIR_CELL)
        out[f"phase_q_{k}"] = qs
        out[f"phase_xi_{k}"] = xis
        steps.append(n)
    out["phase_steps"] = np.array(steps)

    # --- Phileap on the single cell (L = C = 20), the run of tests 2 and 3.
    L, C = 20.0, 20.0
    n = int(5.2 * 2.0 * np.pi * np.sqrt(L * C))
    qs, xis = _run(Phileap(lc_cell(L, C)), [1.0], [0.0], n, _DIR_CELL)
    out.update(leap_params=np.array([L, C, n]), leap_q=qs, leap_xi=xis)

    # --- Phiconf on the single cell: RC relaxation (test 4).
    Lc, Cc_, nc = 4.0, 25.0, 400
    O = Phiconf(lc_cell(Lc, Cc_))
    state = jnp.array([1.0])
    conf_q = [np.asarray(state)]
    for _ in range(nc):
        _, _, state = O.with_state(state).run_one(_IN_POS, _DIR_CELL)
        conf_q.append(np.asarray(state))
    out.update(conf_params=np.array([Lc, Cc_, nc]), conf_q=np.stack(conf_q))

    # --- Phiphase on the coupled pair: both normal modes (test 5).
    Lp, Cp, Ccp = 20.0, 20.0, 5.0
    pair = coupled_lc_pair(Lp, Cp, Ccp)
    out["pair_params"] = np.array([Lp, Cp, Ccp])
    for tag, q0, w in (("plus", [1.0, 1.0], 1.0 / np.sqrt(Lp * Cp)),
                       ("minus", [1.0, -1.0], np.sqrt(1.0 / (Lp * Cp) + 2.0 / (Lp * Ccp)))):
        n = int(5.2 * 2.0 * np.pi / w)
        qs, xis = _run(Phiphase(pair), q0, [0.0, 0.0], n, _DIR_PAIR)
        out[f"pair_q_{tag}"] = qs
        out[f"pair_xi_{tag}"] = xis
        out[f"pair_steps_{tag}"] = np.array(n)

    # --- Phiconf on lc_conf: one-tick states on replayed random inputs (test 6).
    rng = np.random.default_rng(3)
    tick_LC, tick_in, tick_out = [], [], []
    for L2, C2 in [(5.0, 20.0), (2.0, 2.0), (30.0, 7.0)]:
        O2 = Phiconf(lc_conf(L2, C2))
        for _ in range(4):
            iv = rng.standard_normal(2)
            _, _, s = O2.with_state(jnp.asarray(iv)).run_one(_IN_POS, _DIR_CONF)
            tick_LC.append([L2, C2])
            tick_in.append(iv)
            tick_out.append(np.asarray(s))
    out.update(tick_LC=np.array(tick_LC), tick_in=np.stack(tick_in), tick_out=np.stack(tick_out))

    # --- Phiconf on lc_conf: the energy-law / period trajectory (test 6).
    L3, C3, n3 = 20.0, 20.0, 400
    O3 = Phiconf(lc_conf(L3, C3))
    state = jnp.array([0.0, 1.0])
    ivs = [np.asarray(state)]
    for _ in range(n3):
        _, _, state = O3.with_state(state).run_one(_IN_POS, _DIR_CONF)
        ivs.append(np.asarray(state))
    out.update(energy_params=np.array([L3, C3, n3]), energy_iv=np.stack(ivs))

    # --- Phiconf on lc_conf with a series resistance: the radio sweep (test 6).
    # The drive covector ``xi_N(t) = sin(W t)`` is CDLM's sign; the new test
    # feeds ``-sin(W t)`` (STATUS.md §1).
    L4, C4, R4, n4 = 20.0, 20.0, 0.2, 2000
    w4 = 1.0 / np.sqrt(L4 * C4)
    freqs = np.array([0.6 * w4, 0.8 * w4, w4, 1.25 * w4, 1.6 * w4])
    O4 = Phiconf(lc_conf(L4, C4, R4))
    rows = []
    for W in freqs:
        state = jnp.zeros(2)
        vs = []
        for t in range(n4):
            drive = lambda _o, t=t, W=W: (jnp.array([np.sin(W * t)]), jnp.zeros(0))
            _, _, state = O4.with_state(state).run_one(_IN_POS, drive)
            vs.append(float(state[1]))
        rows.append(np.asarray(vs))
    out.update(radio_params=np.array([L4, C4, R4, n4]), radio_freqs=freqs,
               radio_v=np.stack(rows))

    _save("lc_circuit", **out)


if __name__ == "__main__":
    lc_circuit()
