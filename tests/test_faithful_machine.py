"""The faithful gyroscope machine (EXTENSION, Phase 3; gyroscope_faithful.py).

The Bull & Achour classifier assembled entirely from the paper's constructions. The
point (GYRO_BUILD_HANDOFF goal) is the *factorization* and the *springs ablation*, not
accuracy. We check:

* **factorization** -- the classifier's forcing is ``-d`` of the *wired* R^2 graph
  Laplacian plus rod gravity minus the open-port drive, matching an independent oracle
  (the spring coupling EMERGES from ``compose_graph``, it is not a hand-written omega);
* **the springs->0 ablation (the headline)** -- with springs the input signal reaches the
  output gyros and depends on the input; freeze the stiffness to ~0 and *nothing* reaches
  the output (it stays at rest regardless of the input). The blog's "Take 1 failed",
  here a one-line categorical wiring fact;
* the classifier **runs end-to-end and is differentiable** (backprop through the full
  ``PhiK`` / ``rk4_gyro`` rollout gives finite gradients);
* the CDLM outputs (``regress/gyroscope_faithful.npz``), including the machine trained
  for 10 epochs, are reproduced.

The drives enter as *positions* ``n_in`` and every environment covector is zero, so no
input is sign-translated (STATUS.md §1); CDLM's ``U`` became the forcing ``-dU``.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.gyroscope_faithful import (
    batch_loss,
    classify,
    faithful_arrangement,
    init_params,
    make_hex_config,
    make_strokes,
    output_readout,
    train,
)

from tests.chain import load_regress


def _oracle_U(cfg, kappa, g, L):
    """The classifier's cost, written by hand: springs + rod gravity - drive."""
    def U(q, force):
        qv = q.reshape(cfg.V, 2)
        springs = sum(0.5 * kappa * jnp.sum((qv[t] - qv[s]) ** 2) for (s, t) in cfg.edges)
        gravity = sum(-g * jnp.sqrt(L ** 2 - jnp.sum(qv[v] ** 2)) for v in range(cfg.V))
        drive = jnp.dot(force, q[jnp.asarray(cfg.in_idx)])
        return springs + gravity - drive
    return U


def test_faithful_potential_is_wired_laplacian_plus_gravity_minus_drive():
    """The classifier's forcing = -d[(wired R^2 spring Laplacian) + (rod gravity) - (drive)]."""
    cfg = make_hex_config(3, 3, n_classes=4)
    m, kappa, g, L = 1.2, 0.6, 0.3, 3.0
    arr = faithful_arrangement(cfg, m, kappa, g, L)
    assert (arr.out_dim_M, arr.in_dim_M) == (0, 0)
    assert (arr.out_dim_N, arr.in_dim_N) == (cfg.n_out, cfg.n_in)  # open readout + drive ports
    assert arr.P.Q_dim == 2 * cfg.V

    U = _oracle_U(cfg, kappa, g, L)
    rng = np.random.default_rng(0)
    for _ in range(4):
        q = jnp.asarray(0.3 * rng.standard_normal(2 * cfg.V))  # small tilts (|q| < L)
        force = jnp.asarray(rng.standard_normal(cfg.n_in))
        xi_q, xi_m, xi_n = arr.omega(q, jnp.zeros(0), force)
        dU_q, dU_n = jax.grad(U, argnums=(0, 1))(q, force)
        assert xi_m.shape == (0,)
        np.testing.assert_allclose(np.asarray(xi_q), -np.asarray(dU_q), atol=1e-9)
        np.testing.assert_allclose(np.asarray(xi_n), -np.asarray(dU_n), atol=1e-9)


def test_springs_zero_collapses_information_flow():
    """Headline: springs carry the input signal to the output; freeze them and nothing
    reaches the output. Input gyros = left column, output = right column (disjoint), so
    with no spring path the output gyros stay at rest regardless of the input drive."""
    cfg = make_hex_config(3, 3, n_classes=4, settle=10)
    p = init_params(cfg, seed=0)
    rng = np.random.default_rng(1)
    T = 8
    driveA = jnp.asarray(0.2 * rng.standard_normal((T, cfg.n_in)))
    driveB = jnp.asarray(0.2 * rng.standard_normal((T, cfg.n_in)))

    outA = np.asarray(output_readout(p, driveA, cfg))
    outB = np.asarray(output_readout(p, driveB, cfg))
    assert np.linalg.norm(outA) > 1e-3                       # with springs: signal reaches the output
    assert not np.allclose(outA, outB, atol=1e-5)            # ...and it is input-dependent

    p0 = {**p, "log_kappa": jnp.array(-100.0)}               # freeze stiffness to ~0
    outA0 = np.asarray(output_readout(p0, driveA, cfg))
    outB0 = np.asarray(output_readout(p0, driveB, cfg))
    np.testing.assert_allclose(outA0, 0.0, atol=1e-9)        # springs off: nothing reaches the output
    np.testing.assert_allclose(outB0, 0.0, atol=1e-9)        # ...independent of the input


def test_classifier_runs_and_is_differentiable():
    """One stroke -> (n_classes,) logits; backprop through the full ``PhiK`` / ``rk4_gyro``
    rollout gives finite gradients (the classifier is trainable end-to-end)."""
    cfg = make_hex_config(2, 3, n_classes=3, settle=6)
    p = init_params(cfg, seed=0)
    X, Y = make_strokes(8, T=10, n_classes=3, seed=0)

    logits = classify(p, X[0], cfg)
    assert logits.shape == (3,)

    loss, grads = jax.value_and_grad(lambda pp: batch_loss(pp, X, Y, cfg))(p)
    assert np.isfinite(float(loss))
    assert all(bool(np.all(np.isfinite(np.asarray(g)))) for g in grads.values())


def test_hard_drive_and_fast_training_stay_finite():
    """Over-driving the machine must not produce NaN, by either of two routes.

    *Geometry*: the rod potential is only defined for tilts ``|q| < L``; before
    ``rod_gravity`` continued it past that, a large drive returned NaN, which Adam's
    moments then carried through every remaining step -- ``train`` reported a NaN history
    and did not raise.

    *Sign*: drag is the only dissipative term ``rk4_gyro`` carries, so while it was a
    free scalar Adam drove it negative (``-0.68`` by epoch 3 here) and the resulting
    anti-damping blew the rollout up a few epochs later. ``log_drag`` keeps it positive.
    The epoch budget below is past where that run diverged, so it is the sign fix -- not
    a short run -- that keeps the history finite.
    """
    cfg = make_hex_config(3, 3, n_classes=4, settle=8)
    p = init_params(cfg, seed=0)

    rng = np.random.default_rng(2)
    hard = jnp.asarray(3.0 * rng.standard_normal((16, cfg.n_in)))  # drive amplitude ~ L
    assert np.all(np.isfinite(np.asarray(output_readout(p, hard, cfg))))

    Xtr, Ytr = make_strokes(256, T=12, n_classes=4, seed=0)
    for seed in (0, 1):
        _p, hist = train(cfg, Xtr, Ytr, epochs=12, batch=64, lr=3e-2, seed=seed)
        assert np.all(np.isfinite(hist))


def test_faithful_machine_trains_loss_decreases():
    """It trains end-to-end: minibatch Adam through the full ``PhiK`` / ``rk4_gyro`` rollout
    reduces the loss. Per the goal this is only a trainability sanity -- NO accuracy claim
    (the machine exists for the factorization + the springs ablation, not to hit a
    benchmark number)."""
    cfg = make_hex_config(3, 3, n_classes=3, settle=8)
    Xtr, Ytr = make_strokes(300, T=12, n_classes=3, seed=0)
    _p, hist = train(cfg, Xtr, Ytr, epochs=10, batch=60, lr=5e-3, seed=0)
    assert np.isfinite(hist[-1])
    assert hist[-1] < hist[0]  # the loss genuinely decreases under training


def test_regression_matches_cdlm():
    """The CDLM outputs (``regress/gyroscope_faithful.npz``) are reproduced to rtol 1e-10
    (STATUS.md §1): the forcing (minus CDLM's ``dU``), the springs on/off readouts, one
    stroke's logits and loss gradients, the hard-drive readout, and the training history
    and parameters after 10 epochs."""
    R = load_regress("gyroscope_faithful")
    tol = dict(rtol=1e-10, atol=1e-12)

    # forcing = -dU
    cfg = make_hex_config(3, 3, n_classes=4)
    m, kappa, g, L = R["pot_params"]
    arr = faithful_arrangement(cfg, m, kappa, g, L)
    for q, force, dU_q, dU_n in zip(R["pot_q"], R["pot_force"], R["pot_dU_q"], R["pot_dU_n"]):
        xi_q, _, xi_n = arr.omega(jnp.asarray(q), jnp.zeros(0), jnp.asarray(force))
        np.testing.assert_allclose(np.asarray(xi_q), -dU_q, **tol)
        np.testing.assert_allclose(np.asarray(xi_n), -dU_n, **tol)

    # springs on / off readouts
    cfg = make_hex_config(3, 3, n_classes=4, settle=10)
    p = {k: jnp.asarray(R[f"abl_p_{k}"]) for k in init_params(cfg, seed=0)}
    p0 = {**p, "log_kappa": jnp.array(-100.0)}
    for tag, pp in (("", p), ("0", p0)):
        for d in ("A", "B"):
            got = output_readout(pp, jnp.asarray(R[f"abl_drive{d}"]), cfg)
            np.testing.assert_allclose(np.asarray(got), R[f"abl_out{d}{tag}"], **tol)

    # logits and loss gradients
    cfg = make_hex_config(2, 3, n_classes=3, settle=6)
    p = {k: jnp.asarray(R[f"dif_p_{k}"]) for k in init_params(cfg, seed=0)}
    X, Y = jnp.asarray(R["dif_X"]), jnp.asarray(R["dif_Y"])
    np.testing.assert_allclose(np.asarray(classify(p, X[0], cfg)), R["dif_logits"], **tol)
    loss, grads = jax.value_and_grad(lambda pp: batch_loss(pp, X, Y, cfg))(p)
    np.testing.assert_allclose(float(loss), R["dif_loss"], **tol)
    for k, gk in grads.items():
        np.testing.assert_allclose(np.asarray(gk), R[f"dif_g_{k}"], **tol)

    # hard drive
    cfg = make_hex_config(3, 3, n_classes=4, settle=8)
    p = init_params(cfg, seed=0)
    np.testing.assert_allclose(np.asarray(output_readout(p, jnp.asarray(R["hard_drive"]), cfg)), R["hard_out"], **tol)

    # trained 10 epochs
    cfg = make_hex_config(3, 3, n_classes=3, settle=8)
    Xtr, Ytr = jnp.asarray(R["tr_Xtr"]), jnp.asarray(R["tr_Ytr"])
    pt, hist = train(cfg, Xtr, Ytr, epochs=10, batch=60, lr=5e-3, seed=0)
    np.testing.assert_allclose(hist, R["tr_hist"], **tol)
    for k, v in pt.items():
        np.testing.assert_allclose(np.asarray(v), R[f"tr_p_{k}"], **tol)
