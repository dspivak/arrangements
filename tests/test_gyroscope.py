"""Gyroscopes-on-springs as a classifier (EXTENSION; gyroscope.py).

A network of 2-D harmonic gyros with the ``gyro`` coalgebra as parameter, run by
``Phi``, is a harmonic surrogate of the system of Bull & Achour (unconv.ai, 2026). We
check four things: the gyroscopic phase step matches the closed-form Hamilton update;
damping dissipates energy out of a free network; the skew gyroscopic momentum kick is
perpendicular to the velocity, so the *force* does no work (``(J v).v = 0``) -- though
its explicit discretization still needs the damping to stay bounded; and the whole
classifier trains -- the loss drops and a learnable synthetic task is classified well
above chance, exercising backprop through the full ``Phi`` rollout. Then: the CDLM
outputs (``regress/gyroscope.npz``), trained parameters included, are reproduced; and
the DIRECT realization of the gyro/drag terms as a non-exact forcing on ``T^*Q``
(``gyro_arrangement_on_cotangent``) agrees step for step with the TRICK
(``gyro_arrangement`` under ``gyro``).

The drives ``f`` in these tests enter as *positions* ``n_in`` on the input port, and
every environment covector ``xi_N`` is zero, so no input is sign-translated from CDLM
(STATUS.md §1); only the module's cost became the forcing ``omega = -dU``.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.coalgebra import gyro, phase_on_cotangent
from arrangements.functors import Phi
from arrangements.gyroscope import (
    complex_structure,
    gyro_arrangement,
    gyro_arrangement_on_cotangent,
    make_config,
    init_params,
    classify,
    batch_loss,
    accuracy,
    coalgebra,
    train,
)
from arrangements.interpretation import trivial_omega

from tests.chain import load_regress

_IN_POS = (jnp.zeros(0), trivial_omega(0))


def test_gyro_step_matches_closed_form():
    """One ``Phi`` step of a single 2-D gyro with the ``gyro`` parameter equals the
    hand-written symplectic Euler + gyroscopic update ``(q + v, xi - (k*q~ - f) - gamma*J*v)``."""
    cfg = make_config(rows=1, cols=1)  # one gyro; in == out
    m, k, gamma = 2.0, 0.7, 0.4
    J = complex_structure(1)
    arr = gyro_arrangement(cfg, inv_mass=jnp.array([1.0 / m]), kappa=jnp.zeros(0), grav_k=jnp.array(k),
                           parameter=lambda Q: gyro(Q, damping=0.0, gamma=gamma, J=J))
    O = Phi(arr)

    q = jnp.array([0.3, -0.5])
    xi = jnp.array([0.1, 0.2])
    f = jnp.array([0.05, -0.02])
    _, _, (q_new, xi_new) = O.with_state((q, xi)).run_one(_IN_POS, lambda op: (jnp.zeros(2), f))

    v = xi / m
    q_tilde = q + v
    xi_Q = k * q_tilde - f
    q_exp = q + v
    xi_exp = xi - xi_Q - gamma * (J @ v)
    assert np.allclose(np.asarray(q_new), np.asarray(q_exp), atol=1e-10)
    assert np.allclose(np.asarray(xi_new), np.asarray(xi_exp), atol=1e-10)


def _free_rollout(O, cfg, q0, xi0, steps):
    """The unforced rollout's ``(q, xi)`` trajectory (both include the initial state)."""
    state = (q0, xi0)
    qs, xis = [q0], [xi0]
    zero_force = jnp.zeros(cfg.n_in)
    for _ in range(steps):
        _, _, state = O.with_state(state).run_one(_IN_POS, lambda op: (jnp.zeros(cfg.n_out), zero_force))
        qs.append(state[0]); xis.append(state[1])
    return np.stack([np.asarray(x) for x in qs]), np.stack([np.asarray(x) for x in xis])


def _free_energy_trace(cfg, params, q0, xi0, steps):
    """Energy ``H = 1/2 xi.M^{-1}.xi + U(q)`` along the unforced rollout."""
    O = coalgebra(cfg, params)
    inv_diag = jnp.repeat(jnp.exp(params["log_inv_mass"]), 2)
    kappa = jnp.exp(params["log_kappa"])
    grav = jnp.exp(params["log_grav"])
    E = jnp.asarray(cfg.edges)

    def energy(q, xi):
        qv = q.reshape(cfg.V, 2)
        springs = 0.5 * jnp.sum(kappa * jnp.sum((qv[E[:, 0]] - qv[E[:, 1]]) ** 2, axis=1))
        return 0.5 * jnp.sum(inv_diag * xi ** 2) + springs + 0.5 * grav * jnp.sum(qv ** 2)

    qs, xis = _free_rollout(O, cfg, q0, xi0, steps)
    return np.array([float(energy(jnp.asarray(q), jnp.asarray(xi))) for q, xi in zip(qs, xis)])


def test_damping_dissipates_energy():
    """Damping bleeds energy out of a free (unforced) network."""
    rng = np.random.default_rng(0)
    cfg = make_config(rows=2, cols=3, damping=0.1)
    params = init_params(cfg, seed=0, kappa0=0.2, grav0=0.1, gamma0=0.0)
    q0 = jnp.asarray(0.5 * rng.standard_normal(2 * cfg.V))
    xi0 = jnp.asarray(0.5 * rng.standard_normal(2 * cfg.V))

    damped = _free_energy_trace(cfg, params, q0, xi0, steps=60)
    assert damped[-1] < 0.5 * damped[0]  # damping dissipates


def test_gyroscopic_force_does_no_work():
    """The gyroscopic momentum kick ``-gamma*J*v`` is perpendicular to the velocity
    ``v`` (its defining no-work property, rmk.symplectic_perpendicular): per 2-D gyro
    block, ``(J v) . v = 0``. (The explicit discretization still needs damping to stay
    bounded -- see the coalgebra docstring -- but the *force* itself does no work.)"""
    rng = np.random.default_rng(2)
    V = 6
    J = complex_structure(V)
    for _ in range(20):
        v = jnp.asarray(rng.standard_normal(2 * V))
        assert abs(float(jnp.dot(J @ v, v))) < 1e-10


def _synthetic(n, T, rng):
    """A 2-class task: class 0 strokes drift right (+x), class 1 drift left (-x)."""
    y = rng.integers(0, 2, n)
    sign = np.where(y == 0, 1.0, -1.0)[:, None, None]
    x = 0.5 * sign + 0.25 * rng.standard_normal((n, T, 2))
    return jnp.asarray(x), jnp.asarray(y)


def test_classifier_trains_and_separates():
    """Backprop through the full ``Phi`` rollout reduces the loss and learns a
    linearly-separable synthetic stroke task well above chance."""
    rng = np.random.default_rng(1)
    cfg = make_config(rows=2, cols=3, n_classes=2, settle=6)
    Xtr, Ytr = _synthetic(400, 6, rng)
    Xva, Yva = _synthetic(200, 6, rng)

    p0 = init_params(cfg, seed=0)  # cfg.n_classes=2 sizes the decoder head
    loss0 = float(batch_loss(p0, Xtr[:128], Ytr[:128], cfg))

    params, hist = train(cfg, Xtr, Ytr, Xva, Yva, epochs=15, batch=64, lr=5e-3, seed=0, verbose=False)
    best = max(h["val_acc"] for h in hist)

    assert hist[-1]["loss"] < 0.7 * loss0    # training drove the loss down
    assert best > 0.8                        # well above chance (0.5) on a learnable task


def test_online_classifier_is_one_arrangement_and_separates():
    """P43 (b) job 7 (Claude Opus 5.5, de71554a, 2026-10-10). ``gyroscope.online_classifier``: stroke and
    label boxes, the encoder as the drive lens, the gyros, and the decoder's flagged cross-entropy read one
    way, pushing a weights box; one closed arrangement run by ``Phi`` over five passes of the 400 training
    strokes, the physics and encoder held at their initial values. Validation accuracy 0.535 before (the
    untrained decoder, chance), 1.0 after; the old Adam-through-the-rollout training also reaches 1.0 (the
    CDLM recording, every epoch), so this is no better, only a run of one arrangement. Damping 0.2, not the default
    0.05: at the default the network is unstable (next test), which twelve-step rollouts from rest never
    showed. **Could have come out otherwise**: a fixed network's readout might not have separated."""
    rng = np.random.default_rng(1)
    cfg = make_config(rows=2, cols=3, n_classes=2, settle=6, damping=0.2)
    Xtr, Ytr = _synthetic(400, 6, rng)
    Xva, Yva = _synthetic(200, 6, rng)
    p = init_params(cfg, seed=0)
    W0 = jnp.concatenate([p["W_dec"].ravel(), p["b_dec"]])
    from arrangements.functors import closed_step
    from arrangements.gyroscope import online_accuracy, online_classifier, online_weights
    A = online_classifier(cfg, p, np.concatenate([Xtr] * 5), np.concatenate([Ytr] * 5), lr=0.2)
    step = closed_step(A)
    s = jax.lax.fori_loop(0, 5 * 400 * (6 + 6 + 100), lambda _i, u: step(u), A.P.init)
    before, after = online_accuracy(cfg, p, W0, Xva, Yva), online_accuracy(cfg, p, online_weights(A, s), Xva, Yva)
    print(f"\nonline classifier: validation accuracy {before:.3f} -> {after:.3f}")
    assert before < 0.6 and after > 0.8


def test_the_default_network_is_unstable_beyond_its_rollouts():
    """Found by P43 (b) job 7: at the initial physics (gyroscopic coefficient 0.2, damping 0.05) the
    undriven network grows, a 0.1 kick to 5e7 in 300 steps; the explicit gyroscopic kick outruns the
    damping. At damping 0.2, or with no precession, it decays. The trained classifier's rollouts are
    18 steps from rest, too short to show it."""
    from arrangements.gyroscope import _IN_POS as IN
    out = {}
    for gamma, damping in ((0.2, 0.05), (0.2, 0.2), (0.0, 0.05)):
        cfg = make_config(rows=2, cols=3, n_classes=2, settle=6, damping=damping)
        p = dict(init_params(cfg, seed=0))
        p["gamma"] = jnp.array(gamma)
        O = coalgebra(cfg, p)
        f, z = jnp.zeros(len(cfg.in_idx)), jnp.zeros(len(cfg.out_idx))
        step = jax.jit(lambda s, O=O, f=f, z=z: O.with_state(s).run_one(IN, lambda _o: (z, f))[2])
        s = (0.1 * jnp.asarray(np.random.default_rng(0).standard_normal(2 * cfg.V)), jnp.zeros(2 * cfg.V))
        for _ in range(300):
            s = step(s)
        out[(gamma, damping)] = float(jnp.max(jnp.abs(s[0])))
    print("\n|q| after 300 undriven steps:", out)
    assert out[(0.2, 0.05)] > 1e3 and out[(0.2, 0.2)] < 0.1 and out[(0.0, 0.05)] < 0.1


def test_regression_matches_cdlm():
    """The CDLM outputs (``regress/gyroscope.npz``) are reproduced to rtol 1e-10
    (STATUS.md §1): the one-gyro step, the damped free rollout, and the classifier's
    initial loss, training history and trained parameters after 15 epochs."""
    R = load_regress("gyroscope")

    # one-gyro step
    m, k, gamma = R["one_params"]
    cfg = make_config(rows=1, cols=1)
    J = complex_structure(1)
    arr = gyro_arrangement(cfg, inv_mass=jnp.array([1.0 / m]), kappa=jnp.zeros(0), grav_k=jnp.array(k),
                           parameter=lambda Q: gyro(Q, damping=0.0, gamma=gamma, J=J))
    f = jnp.asarray(R["one_f"])
    _, _, (q_new, xi_new) = Phi(arr).with_state((jnp.asarray(R["one_q"]), jnp.asarray(R["one_xi"]))).run_one(
        _IN_POS, lambda op: (jnp.zeros(2), f))
    np.testing.assert_allclose(np.asarray(q_new), R["one_new_q"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.asarray(xi_new), R["one_new_xi"], rtol=1e-10, atol=1e-12)

    # damped free rollout
    cfg = make_config(rows=2, cols=3, damping=0.1)
    params = {k: jnp.asarray(R[f"damp_{k}"]) for k in init_params(cfg, seed=0)}
    qs, xis = _free_rollout(coalgebra(cfg, params), cfg, jnp.asarray(R["damp_q"][0]), jnp.asarray(R["damp_xi"][0]),
                            R["damp_q"].shape[0] - 1)
    np.testing.assert_allclose(qs, R["damp_q"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(xis, R["damp_xi"], rtol=1e-10, atol=1e-12)

    # the classifier: initial loss and logits, then trained parameters and history
    cfg = make_config(rows=2, cols=3, n_classes=2, settle=6)
    Xtr, Ytr, Xva, Yva = (jnp.asarray(R[k]) for k in ("cls_Xtr", "cls_Ytr", "cls_Xva", "cls_Yva"))
    p0 = init_params(cfg, seed=0)
    np.testing.assert_allclose(float(batch_loss(p0, Xtr[:128], Ytr[:128], cfg)), R["cls_loss0"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.asarray(classify(p0, Xtr[0], cfg)), R["cls_logits0"], rtol=1e-10, atol=1e-12)
    params, hist = train(cfg, Xtr, Ytr, Xva, Yva, epochs=15, batch=64, lr=5e-3, seed=0, verbose=False)
    np.testing.assert_allclose([h["loss"] for h in hist], R["cls_hist_loss"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose([h["val_acc"] for h in hist], R["cls_hist_val"], rtol=1e-10, atol=1e-12)
    for k, v in params.items():
        np.testing.assert_allclose(np.asarray(v), R[f"cls_p_{k}"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(accuracy(params, Xva, Yva, cfg), R["cls_final_acc"], rtol=1e-10, atol=1e-12)


def test_direct_one_form_realization_agrees_with_trick():
    """TRICK vs DIRECT (STATUS.md §2, trick vs direct): ``gyro_arrangement`` with the ``gyro``
    parameter smuggles damping, precession and drag into the coalgebra's step, its
    forcing being the exact ``-dU`` on ``Q``; ``gyro_arrangement_on_cotangent`` with
    ``phase_on_cotangent`` writes precession and drag into a non-exact forcing on
    ``T^*Q`` (``-dU(q) + gyro_kick(sharp(xi))``). On the same driven 2x3 network (a
    stroke of encoder forces, then free settling) the two agree step for step."""
    cfg = make_config(rows=2, cols=3, damping=0.05)
    rng = np.random.default_rng(5)
    inv_mass = jnp.asarray(np.exp(0.2 * rng.standard_normal(cfg.V)))
    kappa = jnp.asarray(0.3 * (1.0 + 0.2 * rng.random(len(cfg.edges))))
    grav = jnp.array(0.1)
    gamma = jnp.asarray(np.repeat(0.3 * rng.random(cfg.V), 2))  # per-gyro precession
    kw = dict(gamma=gamma, J=cfg.J, drag=0.3, gyro_block=2)

    trick = Phi(gyro_arrangement(cfg, inv_mass, kappa, grav,
                                 parameter=lambda Q: gyro(Q, damping=cfg.damping, **kw)))
    direct = Phi(gyro_arrangement_on_cotangent(cfg, inv_mass, kappa, grav, **kw,
                                               parameter=lambda Q: phase_on_cotangent(Q, cfg.damping)))

    forces = jnp.asarray(0.3 * rng.standard_normal((8, cfg.n_in)))
    forces = jnp.concatenate([forces, jnp.zeros((12, cfg.n_in))], axis=0)
    zeros_out = jnp.zeros(cfg.n_out)
    s_t = s_d = (jnp.zeros(2 * cfg.V), jnp.zeros(2 * cfg.V))
    for force in forces:
        (out_t, _), _, s_t = trick.with_state(s_t).run_one(_IN_POS, lambda op: (zeros_out, force))
        (out_d, _), _, s_d = direct.with_state(s_d).run_one(_IN_POS, lambda op: (zeros_out, force))
        np.testing.assert_allclose(np.asarray(out_d), np.asarray(out_t), rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(np.asarray(s_d[0]), np.asarray(s_t[0]), rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(np.asarray(s_d[1]), np.asarray(s_t[1]), rtol=1e-12, atol=1e-13)
    assert float(jnp.linalg.norm(s_t[0])) > 1e-3  # the network actually moved
