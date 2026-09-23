"""Record CDLM (``dap-core-old``) outputs for the gyroscope regression tests (STATUS.md §1).

Run from the OLD package, like ``generate_cdlm.py``:

    cd ../dynamic-algebra-potentials/dap-core-old && PYTHONPATH=. uv run --no-sync python ../../arrangements/tests/regress/generate_gyroscope.py

Writes ``gyroscope.npz`` (the harmonic surrogate, ``dap/gyroscope.py``: one-gyro
``Phigyro`` step, the damped free rollout, the drag rollout, the per-gyro-gamma
step, the ``compose_graph`` potentials, and the trained classifier after 15 epochs)
and ``gyroscope_faithful.npz`` (``dap/gyroscope_faithful.py``: the faithful
potential's gradients, the springs-on/off output readouts, the hard-drive readout,
one stroke's logits and loss gradients, and the trained machine after 10 epochs).

All in the old sign convention. Every environment covector these scenarios send is
zero (the drives enter as *positions* ``n_in``, which are not sign-translated), so
the new tests translate only the potentials: the old ``dU`` recorded here is minus
the new forcing ``omega``. States, readouts, logits and trained parameters are
identical.
"""

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from dap.arrangement import SmoothArrangement
from dap.functors import Phigyro
from dap.gyroscope import (
    accuracy, batch_loss, classify, coalgebra, complex_structure, gyro_arrangement,
    hex_graph, init_params, make_config, rod_gravity, train,
)
from dap import gyroscope_faithful as gf
from dap.interpretation import trivial_omega
from dap.rvect import constant
from dap.wiring import compose_graph

assert "dap-core-old" in os.path.abspath(sys.modules["dap"].__file__), \
    "run this from dap-core-old with PYTHONPATH=. so that `dap` is the old package"

HERE = os.path.dirname(os.path.abspath(__file__))
_IN_POS = (jnp.zeros(0), trivial_omega(0))
_TRIV = lambda op: (jnp.zeros(0), jnp.zeros(0))


def _save(name, **arrays):
    path = os.path.join(HERE, name + ".npz")
    np.savez(path, **{k: np.asarray(v) for k, v in arrays.items()})
    print("wrote", path, sorted(arrays))


def _params_dict(prefix, p):
    return {f"{prefix}_{k}": np.asarray(v) for k, v in p.items()}


# ---------------------------------------------------------------------------
# gyroscope.py (test_gyroscope.py, test_drag_one_form.py, test_faithful_physics.py,
# test_vector_graph.py).
# ---------------------------------------------------------------------------


def _free_rollout(O, cfg, q0, xi0, steps):
    state = (q0, xi0)
    qs, xis = [q0], [xi0]
    zero_force = jnp.zeros(cfg.n_in)
    for _ in range(steps):
        _, _, state = O.with_state(state).run_one(_IN_POS, lambda op: (jnp.zeros(cfg.n_out), zero_force))
        qs.append(state[0]); xis.append(state[1])
    return np.stack([np.asarray(x) for x in qs]), np.stack([np.asarray(x) for x in xis])


def _synthetic(n, T, rng):
    y = rng.integers(0, 2, n)
    sign = np.where(y == 0, 1.0, -1.0)[:, None, None]
    x = 0.5 * sign + 0.25 * rng.standard_normal((n, T, 2))
    return jnp.asarray(x), jnp.asarray(y)


def gyroscope():
    out = {}

    # (a) one-gyro Phigyro step (test_gyro_step_matches_closed_form)
    cfg = make_config(rows=1, cols=1)
    m, k, gamma = 2.0, 0.7, 0.4
    arr = gyro_arrangement(cfg, inv_mass=jnp.array([1.0 / m]), kappa=jnp.zeros(0), grav_k=jnp.array(k))
    O = Phigyro(arr, damping=0.0, gamma=gamma, J=complex_structure(1))
    q, xi, f = jnp.array([0.3, -0.5]), jnp.array([0.1, 0.2]), jnp.array([0.05, -0.02])
    _, _, (q_new, xi_new) = O.with_state((q, xi)).run_one(_IN_POS, lambda op: (jnp.zeros(2), f))
    out.update(one_q=q, one_xi=xi, one_f=f, one_new_q=q_new, one_new_xi=xi_new,
               one_params=np.array([m, k, gamma]))

    # (b) damped free rollout, 2x3 grid, 60 steps (test_damping_dissipates_energy)
    rng = np.random.default_rng(0)
    cfg = make_config(rows=2, cols=3, damping=0.1)
    params = init_params(cfg, seed=0, kappa0=0.2, grav0=0.1, gamma0=0.0)
    q0 = jnp.asarray(0.5 * rng.standard_normal(2 * cfg.V))
    xi0 = jnp.asarray(0.5 * rng.standard_normal(2 * cfg.V))
    qs, xis = _free_rollout(coalgebra(cfg, params), cfg, q0, xi0, 60)
    out.update(damp_q=qs, damp_xi=xis, **_params_dict("damp", params))

    # (c) drag rollout, 2x3 grid, drag 0.3, 60 steps (test_drag_dissipates_energy)
    cfg = make_config(rows=2, cols=3)
    kappa = jnp.full(len(cfg.edges), 0.2)
    grav = jnp.array(0.1)
    arr = gyro_arrangement(cfg, inv_mass=jnp.ones(cfg.V), kappa=kappa, grav_k=grav)
    O = Phigyro(arr, damping=0.0, gamma=0.0, J=None, drag=0.3, gyro_block=2)
    rng = np.random.default_rng(0)
    q0 = jnp.asarray(0.5 * rng.standard_normal(2 * cfg.V))
    xi0 = jnp.asarray(0.8 * rng.standard_normal(2 * cfg.V))
    qs, xis = _free_rollout(O, cfg, q0, xi0, 60)
    out.update(drag_q=qs, drag_xi=xis, drag_params=np.array([0.2, 0.1, 0.3]))

    # (d) per-gyro gamma on free gyros (test_per_gyro_gamma_precesses_independently)
    V, mm, g0 = 2, 1.0, 0.5
    free = SmoothArrangement(
        constant(jnp.eye(2 * V) / mm), 0, 0, 0, 0,
        out_f=lambda q, m_out: jnp.zeros(0), in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: jnp.array(0.0), label="free_gyros")
    gam = jnp.array([g0, g0, 0.0, 0.0])
    O = Phigyro(free, gamma=gam, J=complex_structure(V))
    xi = jnp.array([0.0, 1.0, 0.0, 1.0])
    _, _, (q1, xi1) = O.with_state((jnp.zeros(2 * V), xi)).run_one(_IN_POS, _TRIV)
    out.update(pg_xi=xi, pg_gamma=gam, pg_new_q=q1, pg_new_xi=xi1, pg_params=np.array([V, mm]))

    # (e) rod gravity values and gradients (test_faithful_physics.py)
    g, L = 2.0, 3.0
    U = rod_gravity(g, L)
    pts = jnp.array([[0.0, 0.0], [0.5, 0.3], [1e-3, -2e-3], [L, 0.0], [1.5 * L, 0.0], [10.0 * L, 0.0], [2.9, 0.0]])
    out.update(rod_pts=pts, rod_U=jnp.stack([U(p) for p in pts]),
               rod_dU=jnp.stack([jax.grad(U)(p) for p in pts]), rod_params=np.array([g, L]))

    # (f) compose_graph potentials: dU at random points (test_vector_graph.py)
    edges, _, _ = hex_graph(3, 3)
    arr = compose_graph(9, edges, 1.3, 0.7, vdim=2)
    rng = np.random.default_rng(0)
    qs = jnp.asarray(rng.standard_normal((5, 18)))
    out.update(vg_hex_q=qs, vg_hex_dU=jnp.stack([jax.grad(arr.U)(q, jnp.zeros(0), jnp.zeros(0)) for q in qs]))
    edges, _, _ = hex_graph(2, 3)
    onsite = lambda qv: 0.5 * 0.3 * jnp.sum(qv ** 2)
    arr = compose_graph(6, edges, 1.0, 0.5, vdim=2, onsite=onsite)
    rng = np.random.default_rng(1)
    qs = jnp.asarray(rng.standard_normal((5, 12)))
    out.update(vg_on_q=qs, vg_on_dU=jnp.stack([jax.grad(arr.U)(q, jnp.zeros(0), jnp.zeros(0)) for q in qs]))
    arr = compose_graph(3, [(0, 1), (1, 2), (2, 0), (0, 2)], 1.3, 2.1)
    rng = np.random.default_rng(2)
    qs = jnp.asarray(rng.standard_normal((5, 3)))
    out.update(vg_sc_q=qs, vg_sc_dU=jnp.stack([jax.grad(arr.U)(q, jnp.zeros(0), jnp.zeros(0)) for q in qs]))

    # (g) the classifier trained 15 epochs on the synthetic task (test_classifier_trains_and_separates)
    rng = np.random.default_rng(1)
    cfg = make_config(rows=2, cols=3, n_classes=2, settle=6)
    Xtr, Ytr = _synthetic(400, 6, rng)
    Xva, Yva = _synthetic(200, 6, rng)
    p0 = init_params(cfg, seed=0)
    loss0 = float(batch_loss(p0, Xtr[:128], Ytr[:128], cfg))
    logits0 = classify(p0, Xtr[0], cfg)
    params, hist = train(cfg, Xtr, Ytr, Xva, Yva, epochs=15, batch=64, lr=5e-3, seed=0, verbose=False)
    out.update(cls_Xtr=Xtr, cls_Ytr=Ytr, cls_Xva=Xva, cls_Yva=Yva, cls_loss0=np.array(loss0),
               cls_logits0=logits0,
               cls_hist_loss=np.array([h["loss"] for h in hist]),
               cls_hist_val=np.array([h["val_acc"] for h in hist]),
               cls_final_acc=np.array(accuracy(params, Xva, Yva, cfg)),
               **_params_dict("cls_p", params))
    _save("gyroscope", **out)


# ---------------------------------------------------------------------------
# gyroscope_faithful.py (test_faithful_machine.py).
# ---------------------------------------------------------------------------


def faithful():
    out = {}

    # (a) the faithful potential's gradients at small tilts (test_faithful_potential_...)
    cfg = make_hex = gf.make_hex_config(3, 3, n_classes=4)
    m, kappa, g, L = 1.2, 0.6, 0.3, 3.0
    arr = gf.faithful_arrangement(cfg, m, kappa, g, L)
    rng = np.random.default_rng(0)
    qs, fs, dUq, dUn = [], [], [], []
    for _ in range(4):
        q = jnp.asarray(0.3 * rng.standard_normal(2 * cfg.V))
        force = jnp.asarray(rng.standard_normal(cfg.n_in))
        gq, gn = jax.grad(arr.U, argnums=(0, 2))(q, jnp.zeros(0), force)
        qs.append(q); fs.append(force); dUq.append(gq); dUn.append(gn)
    out.update(pot_q=np.stack(qs), pot_force=np.stack(fs), pot_dU_q=np.stack(dUq), pot_dU_n=np.stack(dUn),
               pot_params=np.array([m, kappa, g, L]))

    # (b) springs on / off output readouts (test_springs_zero_collapses_information_flow)
    cfg = gf.make_hex_config(3, 3, n_classes=4, settle=10)
    p = gf.init_params(cfg, seed=0)
    rng = np.random.default_rng(1)
    driveA = jnp.asarray(0.2 * rng.standard_normal((8, cfg.n_in)))
    driveB = jnp.asarray(0.2 * rng.standard_normal((8, cfg.n_in)))
    p0 = {**p, "log_kappa": jnp.array(-100.0)}
    out.update(abl_driveA=driveA, abl_driveB=driveB,
               abl_outA=gf.output_readout(p, driveA, cfg), abl_outB=gf.output_readout(p, driveB, cfg),
               abl_outA0=gf.output_readout(p0, driveA, cfg), abl_outB0=gf.output_readout(p0, driveB, cfg),
               **_params_dict("abl_p", p))

    # (c) one stroke's logits and the batch-loss gradients (test_classifier_runs_and_is_differentiable)
    cfg = gf.make_hex_config(2, 3, n_classes=3, settle=6)
    p = gf.init_params(cfg, seed=0)
    X, Y = gf.make_strokes(8, T=10, n_classes=3, seed=0)
    logits = gf.classify(p, X[0], cfg)
    loss, grads = jax.value_and_grad(lambda pp: gf.batch_loss(pp, X, Y, cfg))(p)
    out.update(dif_X=X, dif_Y=Y, dif_logits=logits, dif_loss=np.array(float(loss)),
               **_params_dict("dif_p", p), **_params_dict("dif_g", grads))

    # (d) the hard-drive readout (test_hard_drive_and_fast_training_stay_finite, first half)
    cfg = gf.make_hex_config(3, 3, n_classes=4, settle=8)
    p = gf.init_params(cfg, seed=0)
    rng = np.random.default_rng(2)
    hard = jnp.asarray(3.0 * rng.standard_normal((16, cfg.n_in)))
    out.update(hard_drive=hard, hard_out=gf.output_readout(p, hard, cfg))

    # (e) trained 10 epochs (test_faithful_machine_trains_loss_decreases)
    cfg = gf.make_hex_config(3, 3, n_classes=3, settle=8)
    Xtr, Ytr = gf.make_strokes(300, T=12, n_classes=3, seed=0)
    pt, hist = gf.train(cfg, Xtr, Ytr, epochs=10, batch=60, lr=5e-3, seed=0)
    out.update(tr_Xtr=Xtr, tr_Ytr=Ytr, tr_hist=np.array(hist), **_params_dict("tr_p", pt))
    _save("gyroscope_faithful", **out)


if __name__ == "__main__":
    gyroscope(); faithful()
