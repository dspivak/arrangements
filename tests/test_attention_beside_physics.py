"""A two-layer attention predictor wired beside pendulums as one closed arrangement (``attention_beside_physics.py``;
book plan P5 (l) stage C; the predictions written before this code and the results are in the book repository,
``notes/runs/p5lC-a9e113da/RESULTS.md``). Each docstring says which of three things its test does: CHECKS what the
construction must do exactly, COMPARES with a plain implementation beside it, or ILLUSTRATES.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.attention_beside_physics import (
    Block, direct_loss, wired_attention, wired_presented, wired_weights, wired_windows,
)
from arrangements.functors import Phi, closed_step
from arrangements.system_id import PENDULUM, oscillator, trajectory

jax.config.update("jax_enable_x64", True)

BK = Block()


def _run(A, steps):
    step = closed_step(A)
    run = jax.jit(lambda s, k: jax.lax.fori_loop(0, k, lambda _i, x: step(x), s), static_argnums=1)
    return run(A.P.init, steps)


def _starts(K, seed=1):
    rng = np.random.default_rng(seed)
    return [tuple(rng.uniform(-1.0, 1.0, 2)) for _ in range(K)]


def test_learner_does_not_move_the_pendulums():
    """CHECKS (C1, prediction 1). Three pendulums wired to the predictor, 50 ``Phi`` steps: each presented position
    equals the open pendulum's ``q + p`` to 1e-9, and the effort field at each pendulum's configuration is its own
    potential's, ``-sin a``, exactly: the label's share is 0 and the delays pull nothing."""
    K, N = 3, 50
    starts = _starts(K)
    A = wired_attention(starts, BK, eta=1.5e-4, seed=1)
    s = _run(A, N)
    O = Phi(oscillator(PENDULUM[0], 1, 1.0))
    for i in range(K):
        tr = trajectory(O, jnp.array([starts[i][0]]), jnp.array([starts[i][1]]), N)
        assert abs(float(wired_presented(A, s, K, BK)[i]) - float(tr[-1][0] + tr[-1][1])) < 1e-9
    z = jnp.zeros(0)
    force = A.omega(A.P.position(s), z, z)[0]
    a = wired_presented(A, s, K, BK)
    np.testing.assert_allclose(force[:(1 + 2 * BK.h) * K].reshape(K, -1)[:, 0], -jnp.sin(a), atol=1e-14)


def test_forces_at_the_weights_are_backpropagation():
    """COMPARES with ``jax.grad`` of the plain causal transformer (C1, prediction 2). After 50 steps, the closed
    arrangement's effort field at the weights is minus the gradient of the summed squared error over the windows its
    tokens hear, to 1e-10. With layer 2's keys and queries frozen the gradient differs; prediction 2's bound on that
    gap, 1e-3, failed at these weights (3.9e-6, RESULTS C1), so the test asserts what discriminates: the forces
    match the full gradient at least 1e6 times more closely than the frozen one."""
    K, N = 3, 50
    A = wired_attention(_starts(K), BK, eta=1.5e-4, seed=1)
    s = _run(A, N)
    z = jnp.zeros(0)
    force = A.omega(A.P.position(s), z, z)[0]
    o = (1 + 2 * BK.h) * K
    fW = force[o:o + BK.n]
    W, win, lab = wired_weights(A, s, K, BK), wired_windows(A, s, K, BK), wired_presented(A, s, K, BK)
    g = jax.grad(lambda W_: direct_loss(BK, W_, win, lab))(W)
    gf = jax.grad(lambda W_: direct_loss(BK, W_, win, lab, freeze_keys=True))(W)
    err, gap = float(jnp.max(jnp.abs(fW + g))), float(jnp.max(jnp.abs(fW + gf)))
    print(f"max |force + grad| = {err:.3g}; max |force + frozen grad| = {gap:.4g}; max |force| = "
          f"{float(jnp.max(jnp.abs(fW))):.4g}")
    assert err < 1e-10
    assert gap > 1e6 * max(err, 1e-16)


def test_attention_learner_predicts_the_pendulums():
    """ILLUSTRATES (C2). Forty pendulums (starts as ``test_wired_pendulums_teach_their_learner``) wired to the attention
    predictor, 5000 ``Phi`` steps at rate 1.5e-4 per pendulum, the MLP's settings: on eight held-out runs the one-step
    error falls below B1's bounds for the MLP (0.15, and a quarter of "no change"). Recorded (RESULTS C2): 0.203 at
    3000 steps, where the MLP has 0.045; 0.043 at 4000; both settle at 0.045, the error of the best linear predictor
    ``a_t ~ 1.16 a_{t-1} - 0.99 a_{t-2}`` (0.044), neither learning the ``sin`` term. ``closed_step``'s check
    against ``Phi`` is skipped here (it costs 125 s at K = 40); the C1 tests run it on the same construction."""
    from arrangements.attention_beside_physics import held_out_error

    K = 40
    A = wired_attention(_starts(K), BK, eta=1.5e-4, seed=1)
    step = closed_step(A, check=0)
    s = jax.jit(lambda s: jax.lax.fori_loop(0, 5000, lambda _i, x: step(x), s))(A.P.init)
    err, base = held_out_error(BK, wired_weights(A, s, K, BK))
    print(f"held-out one-step error {err:.4f}, no change {base:.4f}")
    assert err < 0.15 and err < 0.25 * base
