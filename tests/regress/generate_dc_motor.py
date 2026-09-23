"""Record CDLM (``dap-core-old``) DC-motor trajectories for the regression test (STATUS.md §1).

Run from the OLD package so that ``import dap`` is CDLM's:

    cd ../dynamic-algebra-potentials/dap-core-old && PYTHONPATH=. uv run --no-sync python ../../arrangements/tests/regress/generate_dc_motor.py

Writes ``dc_motor.npz`` into this directory: the inputs of the scenarios of
``dap-core-old/dap/tests/test_dc_motor.py`` (constants, seeds replayed into the
recorded initial states) and the outputs the old package produces -- the
``Phiconf`` trajectories of the monolithic, compositional, sentence-encoding,
Bateman and flow-coordinate arrangements, and the covector ``xi_Q`` the old
interpretation hands the integrator. All in the old sign convention: ``xi_Q``
is CDLM's ``d(U_total)``-covector, which the integrator SUBTRACTS. The new test
translates per STATUS.md §1: ``xi_Q_new = -xi_Q_old``, states identical.
"""

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from dap.dc_motor import (
    bateman_motor,
    dc_motor,
    dc_motor_flow,
    electrical_cell,
    gyrator,
    mechanical_cell,
    motor_monolithic,
    sentence_pair,
)
from dap.functors import Phiconf
from dap.interpretation import smooth_interpretation, trivial_omega
from dap.wiring import compose_seq, tensor_arrangements

assert "dap-core-old" in os.path.abspath(sys.modules["dap"].__file__), \
    "run this from dap-core-old with PYTHONPATH=. so that `dap` is the old package"

HERE = os.path.dirname(os.path.abspath(__file__))
_IN_POS = (jnp.zeros(0), trivial_omega(0))
_TRIV = lambda _o: (jnp.zeros(0), jnp.zeros(0))
_P = dict(R=1.0, L=0.5, K=2.0, J=0.3, b=0.1)


def _save(name, **arrays):
    path = os.path.join(HERE, name + ".npz")
    np.savez(path, **{k: np.asarray(v) for k, v in arrays.items()})
    print("wrote", path, sorted(arrays))


def _step_fn(arr):
    O = Phiconf(arr)

    def f(s):
        _, _, s2 = O.with_state(s).run_one(_IN_POS, _TRIV)
        return s2

    return f


def _run(arr, q0, steps):
    step = jax.jit(_step_fn(arr))
    s = jnp.asarray(q0, float)
    traj = [np.asarray(s)]
    for _ in range(steps):
        s = step(s)
        traj.append(np.asarray(s))
    return np.stack(traj)


def _xi_Q(arr, q):
    _, direction_action = smooth_interpretation(arr)(jnp.asarray(q))
    xi_Q, _, _ = direction_action(jnp.zeros(0), trivial_omega(0), jnp.zeros(0), jnp.zeros(0))
    return np.asarray(xi_Q)


def dc_motor_regress():
    out = dict(P=np.array([_P["R"], _P["L"], _P["K"], _P["J"], _P["b"]]))

    # monolithic, one step on 5 random states (+ the naive-source control)
    dt, V, tau = 0.01, 1.3, 0.2
    z_mono = np.random.default_rng(0).standard_normal((5, 2))
    step = _step_fn(motor_monolithic(**_P, V=V, tau=tau, dt=dt))
    out["mono_params"] = np.array([dt, V, tau])
    out["mono_z"] = z_mono
    out["mono_next"] = np.stack([np.asarray(step(jnp.asarray(z))) for z in z_mono])

    z_naive = np.random.default_rng(1).standard_normal((3, 2))
    step = _step_fn(motor_monolithic(**_P, V=V, tau=tau, dt=dt, naive_sources=True))
    out["naive_z"] = z_naive
    out["naive_next"] = np.stack([np.asarray(step(jnp.asarray(z))) for z in z_naive])

    # lossless: 800-step trajectory from (0.3, 0)
    L, J, K = _P["L"], _P["J"], _P["K"]
    omega0 = K / np.sqrt(L * J)
    dt_l = 0.05 / omega0
    out["lossless_params"] = np.array([dt_l, omega0])
    out["lossless_traj"] = _run(motor_monolithic(R=0.0, L=L, K=K, J=J, b=0.0, dt=dt_l), [0.3, 0.0], 800)

    # compositional: one step on 5 random states, the composite potential and sharp,
    # and the 300-step trajectory from (0.4, -0.2)
    arr = dc_motor(**_P, V=V, tau=tau, dt=dt)
    z_comp = np.random.default_rng(3).standard_normal((5, 2))
    step = _step_fn(arr)
    out["comp_z"] = z_comp
    out["comp_next"] = np.stack([np.asarray(step(jnp.asarray(z))) for z in z_comp])
    out["comp_sharp"] = np.asarray(arr.Q.sharp_at(jnp.zeros(2)))
    z_U = np.random.default_rng(2).standard_normal((5, 2))
    out["comp_U_z"] = z_U
    out["comp_U"] = np.array([float(arr.U(jnp.asarray(z), jnp.zeros(0), jnp.zeros(0))) for z in z_U])
    dU = jax.grad(lambda q: arr.U(q, jnp.zeros(0), jnp.zeros(0)))
    out["comp_dU"] = np.stack([np.asarray(dU(jnp.asarray(z))) for z in z_U])
    out["comp_traj"] = _run(arr, [0.4, -0.2], 300)
    out["mono_traj"] = _run(motor_monolithic(**_P, V=V, tau=tau, dt=dt), [0.4, -0.2], 300)

    # step Jacobians: the true encoding and the same-signed negative control
    z0 = jnp.array([0.4, -0.2])
    arr0 = dc_motor(**_P, dt=dt)
    out["jac_z0"] = np.asarray(z0)
    out["jac_true"] = np.asarray(jax.jacobian(_step_fn(arr0))(z0))
    R, L2, K2, J2, b = _P["R"], _P["L"], _P["K"], _P["J"], _P["b"]
    same_sign = compose_seq(
        tensor_arrangements(
            [electrical_cell(R, L2, 0.0, dt), mechanical_cell(b, J2, 0.0, dt, sharp_sign=+1.0)]
        ),
        gyrator(K2),
    )
    out["jac_same_sign"] = np.asarray(jax.jacobian(_step_fn(same_sign))(z0))
    out["gyrator_dim"] = np.array(gyrator(K2).Q.dim)

    # sentence encoding: Jacobians at random coefficients/states (tests 9 and 10)
    rng = np.random.default_rng(4)
    coeffs, states, jacs = [], [], []
    for _ in range(5):
        d1, a2, b2, d2, c = rng.standard_normal(5)
        arr_s = sentence_pair((0.0, 0.0, d1), (a2, b2, d2), c, dt=dt)
        z = jnp.asarray(rng.standard_normal(4))
        coeffs.append([0.0, 0.0, d1, a2, b2, d2, c]); states.append(np.asarray(z))
        jacs.append(np.asarray(jax.jacobian(_step_fn(arr_s))(z)))
    out.update(sent_coeffs=np.stack(coeffs), sent_z=np.stack(states), sent_jac=np.stack(jacs))
    arr_s = sentence_pair((0.7, 0.0, 0.3), (0.4, 0.0, -0.2), 0.5, dt=dt)
    out["sent_a1_jac"] = np.asarray(jax.jacobian(_step_fn(arr_s))(jnp.zeros(4)))

    rng = np.random.default_rng(5)
    coeffs, states, jacs = [], [], []
    for _ in range(5):
        ce = tuple(rng.standard_normal(3))
        cm = tuple(rng.standard_normal(3))
        cc = float(rng.standard_normal())
        arr_s = sentence_pair(ce, cm, cc, dt=dt)
        z = jnp.asarray(rng.standard_normal(4))
        coeffs.append(list(ce) + list(cm) + [cc]); states.append(np.asarray(z))
        jacs.append(np.asarray(jax.jacobian(_step_fn(arr_s))(z)))
    out.update(vol_coeffs=np.stack(coeffs), vol_z=np.stack(states), vol_jac=np.stack(jacs))

    # Bateman: one step from 5 embedded states, and two 3000-step runs
    dt_b, steps_b = 0.002, 3000
    arr_b, embed, _ = bateman_motor(**_P, dt=dt_b)
    rng = np.random.default_rng(6)
    step_b = _step_fn(arr_b)
    y0s = rng.standard_normal((5, 2))
    out["bat_params"] = np.array([dt_b, steps_b])
    out["bat_y0"] = y0s
    out["bat_next"] = np.stack([np.asarray(step_b(embed(jnp.asarray(y)))) for y in y0s])
    y = np.array([0.7, -0.3])
    z_embed = np.asarray(embed(jnp.asarray(y)))
    out["bat_embed"] = z_embed
    out["bat_traj"] = _run(arr_b, z_embed, steps_b)
    z_pert = z_embed.copy(); z_pert[0] += 1e-6
    out["bat_traj_pert"] = _run(arr_b, z_pert, steps_b)

    # spin-up (12000 steps), covectors along a 400-step run, and the stall runs
    dt_s, V_s = 0.002, 1.0
    out["spin_params"] = np.array([dt_s, V_s])
    out["spin_traj"] = _run(dc_motor(**_P, V=V_s, dt=dt_s), [0.0, 0.0], 12000)

    dt_c, V_c, tau_c = 0.002, 1.0, 0.05
    arr_c = dc_motor(**_P, V=V_c, tau=tau_c, dt=dt_c)
    traj_c = _run(arr_c, [0.0, 0.0], 400)
    out["cov_params"] = np.array([dt_c, V_c, tau_c])
    out["cov_traj"] = traj_c
    out["cov_xi_Q"] = np.stack([_xi_Q(arr_c, z) for z in traj_c[::50]])

    for k, V_st in enumerate((0.8, 1.6)):
        arr_st = dc_motor(**_P, V=V_st, dt=dt_s, locked_rotor=True)
        traj_st = _run(arr_st, [0.0, 0.0], 8000)
        out[f"stall{k}_params"] = np.array([dt_s, V_st])
        out[f"stall{k}_traj"] = traj_st
        out[f"stall{k}_xi_Q"] = _xi_Q(arr_st, traj_st[-1])

    # flow-coordinate variant: one step on 5 random states
    Rr, Lf, Kf, Jf, bf, Vf, tauf, dtf = 1.3, 0.7, 0.9, 2.1, 0.4, 5.0, 0.6, 0.01
    O = Phiconf(dc_motor_flow(Rr, Lf, Kf, Jf, bf, Vf, tauf, dtf))
    rng = np.random.default_rng(5)
    zs, nexts = [], []
    for _ in range(5):
        i0, w0 = rng.standard_normal(2)
        state = jnp.array([i0, w0])
        *_, state = O.with_state(state).run_one(_IN_POS, _TRIV)
        zs.append([i0, w0]); nexts.append(np.asarray(state))
    out.update(flow_params=np.array([Rr, Lf, Kf, Jf, bf, Vf, tauf, dtf]),
               flow_z=np.array(zs), flow_next=np.stack(nexts))

    _save("dc_motor", **out)


if __name__ == "__main__":
    dc_motor_regress()
