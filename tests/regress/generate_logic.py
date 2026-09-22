"""Record CDLM (``dap-core-old``) CMOS-logic trajectories for the regression test (DESIGN.md §3).

Run from the OLD package so that ``import dap`` is CDLM's:

    cd ../dynamic-algebra-potentials/dap-core-old && PYTHONPATH=. uv run --no-sync python ../../arrangements/tests/regress/generate_logic.py

Writes ``logic.npz`` into this directory: the inputs the old test file
(``test_logic.py``) used and the outputs the old package produced, in the old
sign convention. The new test translates per DESIGN.md §3: the ``load``
presented at the output ports and the port currents ``run`` returns are
negated (they are covectors crossing a port), states are identical.

Covers both dynamics the old tests run: ``Phiconf`` (devices, NAND, drain
swing, latch, inverter chain) and ``Phirk4`` (the four-stage ``pc^(4)``
rollout of ``test_rk4_buys_1_4x_the_step_for_4x_the_work``).
"""

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from dap.functors import Phiconf, Phirk4
from dap.interpretation import trivial_omega
from dap.logic import (
    chain_state,
    inverter,
    inverter_chain,
    inverter_state,
    latch_state,
    mosfet,
    nand_gate,
    nand_state,
    run,
    sr_latch,
    stepper,
)

assert "dap-core-old" in os.path.abspath(sys.modules["dap"].__file__), \
    "run this from dap-core-old with PYTHONPATH=. so that `dap` is the old package"

HERE = os.path.dirname(os.path.abspath(__file__))
_IN_POS = (jnp.zeros(0), trivial_omega(0))
HI, LO = 1.0, 0.0


def _save(name, **arrays):
    path = os.path.join(HERE, name + ".npz")
    np.savez(path, **{k: np.asarray(v) for k, v in arrays.items()})
    print("wrote", path, sorted(arrays))


def _settle_fet(fet, e, n=300):
    O, q = Phiconf(fet), jnp.zeros(2)
    for _ in range(n):
        _, fiber = O.with_state(q).step(q)
        _, at_pos = fiber(_IN_POS)
        _, q = at_pos((jnp.zeros(0), e))
    return q


def _fet_currents(fet, q, e):
    _, fiber = Phiconf(fet).with_state(q).step(q)
    (_, omega_N), _ = fiber(_IN_POS)
    return omega_N(e)


def logic():
    out = {}

    # 1. the single device: settled gate states and port currents.
    fet = mosfet("n")
    vds = np.array([0.2, 0.5, 0.9, 2.0])
    qs = [_settle_fet(fet, jnp.array([0.8, float(vd), 0.0])) for vd in vds]
    e = jnp.array([0.8, 0.5, 0.1])
    out.update(fet_vds=vds, fet_q=np.stack([np.asarray(q) for q in qs]),
               fet_e=np.asarray(e), fet_currents=np.asarray(_fet_currents(fet, _settle_fet(fet, e), e)))

    # 2. NAND truth table: 600 ticks from nand_state(0.5), zero load.
    finals, lasts, currs = [], [], []
    for a, b in ((LO, LO), (LO, HI), (HI, LO), (HI, HI)):
        st, outs, currents = run(nand_gate(), nand_state(0.5), lambda t: (a, b), 600)
        finals.append(np.asarray(st)); lasts.append(np.asarray(outs[-1])); currs.append(np.asarray(currents[-1]))
    out.update(nand_inputs=np.array([[LO, LO], [LO, HI], [HI, LO], [HI, HI]]),
               nand_state0=np.asarray(nand_state(0.5)), nand_steps=np.array(600),
               nand_final=np.stack(finals), nand_out=np.stack(lasts), nand_current=np.stack(currs))

    # 3. drain swing under three loads (the loads are CDLM-sign covectors).
    loads = np.array([0.0, 0.5, 2.0])
    swing = []
    for load in loads:
        st, _, _ = run(nand_gate(), nand_state(0.5), lambda t: (HI, HI), 400, load=[float(load)])
        swing.append(np.asarray(st))
    out.update(swing_loads=loads, swing_steps=np.array(400), swing_final=np.stack(swing))

    # 4. SR latch: set / hold / reset / hold / set / hold, 500 ticks each.
    L, st = sr_latch(), latch_state(0.5, 0.5)
    seq = [(LO, HI), (HI, HI), (HI, LO), (HI, HI), (LO, HI), (HI, HI)]
    seen = []
    for sbar, rbar in seq:
        st, outs, _ = run(L, st, lambda t: (sbar, rbar), 500)
        seen.append(np.asarray(outs[-1]))
    out.update(latch_state0=np.asarray(latch_state(0.5, 0.5)), latch_seq=np.array(seq),
               latch_steps=np.array(500), latch_outs=np.stack(seen), latch_final=np.asarray(st))

    # 5. inverter chain of depth 4, 400 ticks with input HI.
    st, outs, _ = run(inverter_chain(4), chain_state(4), lambda t: (HI,), 400)
    out.update(chain_K=np.array(4), chain_steps=np.array(400), chain_state0=np.asarray(chain_state(4)),
               chain_final=np.asarray(st), chain_outs=np.stack([np.asarray(o) for o in outs[::50]]))

    # 6. Phirk4 (pc^(4)): the inverter stepped by classical RK4, h = 5.0, 50 macro-ticks.
    h, T = 5.0, 50
    O = Phirk4(inverter(c=0.02, kappa=0.5), h=h)
    triv = lambda _o: (jnp.zeros(1), jnp.array([HI]))
    s = inverter_state(0.5)
    traj = [np.asarray(s)]
    for _ in range(T):
        s = O.with_state(s).run_one(_IN_POS, triv)[-1]
        traj.append(np.asarray(s))
    out.update(rk4_h=np.array(h), rk4_steps=np.array(T), rk4_state0=np.asarray(inverter_state(0.5)),
               rk4_traj=np.stack(traj))

    # 7. one plain tick of the inverter, with its port covector (pins the port sign).
    one = stepper(inverter(c=0.02))
    o1, cur1, s1 = one(inverter_state(0.5), jnp.array([HI]), jnp.zeros(1))
    out.update(tick_out=np.asarray(o1), tick_current=np.asarray(cur1), tick_state=np.asarray(s1))

    _save("logic", **out)


if __name__ == "__main__":
    logic()
