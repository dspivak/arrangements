"""Record CDLM (``dap-core-old``) amplifier trajectories for the regression test (DESIGN.md §3).

Run from the OLD package so that ``import dap`` is CDLM's:

    cd ../dynamic-algebra-potentials/dap-core-old && PYTHONPATH=. uv run --no-sync python ../../arrangements/tests/regress/generate_amplifier.py

Writes ``amplifier.npz`` into this directory: the inputs of the main scenarios of
``dap-core-old/dap/tests/test_amplifier.py`` (constants; states) and the outputs
the old package produces (reported ``v``, the emitted reaction ``react`` on the
drive port, updated states, settle trajectories), all in the old sign
convention: ``react`` is CDLM's reaction covector ``omega_N(e_src)`` (published
fields are ``pullback + dU``), and the load ``xi_N`` is the covector CDLM's
integrator subtracts.  The new test translates per DESIGN.md §3:
``react_new = -react_old``, ``xi_N_new = -xi_N_old``, states identical.
"""

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from dap.amplifier import amp_state, black_amplifier, settle, stepper

assert "dap-core-old" in os.path.abspath(sys.modules["dap"].__file__), \
    "run this from dap-core-old with PYTHONPATH=. so that `dap` is the old package"

HERE = os.path.dirname(os.path.abspath(__file__))
BETA = 0.1


def _tanh_g(s):
    return lambda x: s * jnp.tanh(x / s)


def _save(name, **arrays):
    path = os.path.join(HERE, name + ".npz")
    np.savez(path, **{k: np.asarray(v) for k, v in arrays.items()})
    print("wrote", path, sorted(arrays))


def _settle_traj(one, state, e_src, steps):
    """``settle`` recording ``v`` and ``react`` at every tick."""
    e = jnp.atleast_1d(jnp.asarray(e_src, dtype=float))
    xi = jnp.zeros(1)
    vs, reacts = [], []
    for _ in range(steps):
        v, react, state = one(state, e, xi)
        vs.append(float(v)); reacts.append(float(react))
    return np.array(vs), np.array(reacts), state


def amplifier():
    out = {}

    # one-step, linear stage, loaded and unloaded (test_output_load_rejection,
    # test_source_reaction_emitted_and_discarded, test_one_step_exactness_linear)
    A, kappa, dt = 50.0, 0.7, 0.03
    one = stepper(black_amplifier(A, BETA, kappa, dt))
    s0, e, load = jnp.array([0.3, 0.0]), jnp.array([0.5]), jnp.array([7.3])
    v_out, react, s1 = one(s0, e, jnp.zeros(1))
    _, _, s1_loaded = one(s0, e, load)
    out.update(lin_params=np.array([A, BETA, kappa, dt]), lin_s0=s0, lin_e=e, lin_load=load,
               lin_out=v_out, lin_react=react, lin_s1=s1, lin_s1_loaded=s1_loaded)

    # one-step, saturating stage (test_one_step_exactness_nonlinear)
    A, kappa, dt, s = 1000.0, 1.0, 0.008, 0.01
    one = stepper(black_amplifier(A, BETA, kappa, dt, g=_tanh_g(s)))
    s0, e = jnp.array([2.0, 0.0]), jnp.array([0.4])
    v_out, react, s1 = one(s0, e, jnp.zeros(1))
    out.update(nl_params=np.array([A, BETA, kappa, dt, s]), nl_s0=s0, nl_e=e,
               nl_out=v_out, nl_react=react, nl_s1=s1)

    # one-step, bilateral control (test_bilateral_one_step_feels_the_loop)
    A, kappa, dt = 50.0, 0.7, 1e-4
    one = stepper(black_amplifier(A, BETA, kappa, dt, unilateral=False))
    s0, e = jnp.array([0.3]), jnp.array([0.5])
    v_out, react, s1 = one(s0, e, jnp.zeros(1))
    out.update(bi_params=np.array([A, BETA, kappa, dt]), bi_s0=s0, bi_e=e,
               bi_out=v_out, bi_react=react, bi_s1=s1)

    # Black's gain: settle 300 ticks for A in (1e2, 1e3, 1e4) (test_black_gain_formula)
    e_src, kappa = 2.0, 1.0
    for k, A in enumerate((1e2, 1e3, 1e4)):
        dt = 0.5 / (kappa * (1.0 + A * BETA))
        one = stepper(black_amplifier(A, BETA, kappa, dt))
        vs, reacts, state = _settle_traj(one, amp_state(), e_src, 300)
        out[f"gain{k}_params"] = np.array([A, BETA, kappa, dt, e_src])
        out[f"gain{k}_v"] = vs
        out[f"gain{k}_react"] = reacts
        out[f"gain{k}_state"] = state

    # distortion: quasi-static closed and open loop (test_distortion_suppression)
    A, kappa, s = 1000.0, 1.0, 0.01
    g = _tanh_g(s)
    b = 0.6 * s
    Vp = A * s * np.tanh(b / s)
    u0 = b + BETA * Vp
    N = 64
    theta = 2 * np.pi * np.arange(N) / N
    one_cl = stepper(black_amplifier(A, BETA, kappa, dt=0.008 / kappa, g=g))
    state, y_cl = amp_state(), []
    for th in theta:
        v, _, state = settle(one_cl, state, u0 * np.sin(th), 600)
        y_cl.append(float(v))
    one_ol = stepper(black_amplifier(A, 0.0, kappa, dt=0.8 / kappa, g=g))
    state, y_ol = amp_state(), []
    for th in theta:
        v, _, state = settle(one_ol, state, b * np.sin(th), 100)
        y_ol.append(float(v))
    out.update(thd_params=np.array([A, BETA, kappa, s, N]), thd_y_cl=np.array(y_cl), thd_y_ol=np.array(y_ol))

    # stability boundary: 400 ticks just inside and just outside (test_stability_boundary)
    A, kappa, e_src = 1e3, 1.0, 2.0
    vstar = A / (1.0 + A * BETA) * e_src
    for k, prod in enumerate((1.95, 2.05)):
        dt = prod / (kappa * (1.0 + A * BETA))
        one = stepper(black_amplifier(A, BETA, kappa, dt))
        vs, _, state = _settle_traj(one, jnp.array([vstar + 1.0, 0.0]), e_src, 400)
        out[f"stab{k}_params"] = np.array([A, BETA, kappa, dt, e_src, vstar + 1.0])
        out[f"stab{k}_v"] = vs
        out[f"stab{k}_state"] = state

    # bilateral divergence, 6 ticks (test_bilateral_diverges_where_unilateral_converges)
    dt = 1.95 / (kappa * (1.0 + A * BETA))
    one = stepper(black_amplifier(A, BETA, kappa, dt, unilateral=False))
    vs, _, state = _settle_traj(one, jnp.array([vstar + 1.0]), e_src, 6)
    out.update(bidiv_params=np.array([A, BETA, kappa, dt, e_src, vstar + 1.0]), bidiv_v=vs, bidiv_state=state)

    # bilateral same equilibrium, 300 ticks (test_bilateral_same_equilibrium)
    dt = 0.5 / (kappa * (1.0 + A * BETA) ** 2)
    one = stepper(black_amplifier(A, BETA, kappa, dt, unilateral=False))
    vs, reacts, state = _settle_traj(one, amp_state(unilateral=False), e_src, 300)
    out.update(bieq_params=np.array([A, BETA, kappa, dt, e_src]), bieq_v=vs, bieq_react=reacts, bieq_state=state)

    _save("amplifier", **out)


if __name__ == "__main__":
    amplifier()
