"""System identification (CDLM rmk.graph_environment, "Predicting a closed system").

``Phi`` of the nonlinear pendulum with a phase parameter runs the system; a
network trained on its consecutive-state pairs -- the ``<o, o.c>`` environment
with ``o = id`` -- learns the system's one-step flow map. We check the held-out
one-step error is small and far below the naive "no change" baseline, so the net
genuinely identified the (nonlinear) dynamics.

**Wired** (plan P43 (b) job 5, Claude Opus 5.5, de71554a, 2026-10-10): ``system_id.wired_identification``
runs forty pendulums and their learner as one closed arrangement, with no data passed in code. Each
pendulum reads out its presented position; two delay boxes per pendulum hold the two before it; one
parameter box's weights are copied to a net per pendulum, which predicts the presented position from
the two delayed ones; a one-way loss per pendulum pushes the answer and not the pendulum. It cannot
equal the recorded run below (the learner sees positions, not momenta, and every pendulum at once,
not shuffled pairs), so the check is the pendulums' own trajectories, equal to ``trajectory``'s, and
the recorded run's claim, an accurate one-step prediction far below "no change," on held-out runs.
The recorded run (``identify``: trajectories run, paired and shuffled in code, then fed to the learner
through ``learning.train``, whose loss is now a box) is kept below with its CDLM regression.
"""

import jax.numpy as jnp
import numpy as np

import jax

from arrangements.functors import Phi, closed_step
from arrangements.system_id import (
    PENDULUM, identify, mlp, one_step_error, oscillator, trajectory, wired_identification,
    wired_presented, wired_weights,
)

from tests.chain import load_regress


def test_wired_pendulums_teach_their_learner():
    """Forty pendulums (starts uniform in [-1, 1]^2, seed 1) wired to one learner, 3000 ``Phi`` steps at
    rate 1.5e-4 per pendulum (40 x 1.5e-4 = the recorded run's 6e-3): each pendulum's presented position
    equals the recorded run's ``q + p/m`` (the learner does not move it), and on eight held-out runs
    the one-step error is 0.045 against "no change"'s 0.56. **Could have come out otherwise**: online,
    correlated data from forty orbits might not have fitted."""
    K = 40
    rng = np.random.default_rng(1)
    starts = [tuple(rng.uniform(-1.0, 1.0, 2)) for _ in range(K)]
    A = wired_identification(starts, eta=1.5e-4, seed=1)
    step = closed_step(A)
    run = jax.jit(lambda s, k: jax.lax.fori_loop(0, k, lambda _i, x: step(x), s), static_argnums=1)
    s = run(A.P.init, 3000)

    O = Phi(oscillator(PENDULUM[0], 1, 1.0))
    for i in (0, 17, 39):
        tr = trajectory(O, jnp.array([starts[i][0]]), jnp.array([starts[i][1]]), 3000)
        assert abs(float(wired_presented(A, s, K)[i]) - float(tr[-1][0] + tr[-1][1])) < 1e-9

    F, _ = mlp(32, 2, 1)
    th = wired_weights(A, s, K)
    held = np.random.default_rng(101)
    errs, base = [], []
    for _ in range(8):
        q0, p0 = held.uniform(-1.0, 1.0, 2)
        a = [float(x[0] + x[1]) for x in trajectory(O, jnp.array([q0]), jnp.array([p0]), 40)]
        for t in range(2, len(a)):
            errs.append(abs(float(F(th, jnp.array([a[t - 2], a[t - 1]]))[0]) - a[t]))
            base.append(abs(a[t] - a[t - 1]))
    err, b = float(np.mean(errs)), float(np.mean(base))
    assert err < 0.15 and err < 0.25 * b


def test_identifies_nonlinear_pendulum_one_step_map():
    params, F, hist, O, dim, m = identify(PENDULUM, steps=3000, seed=1)
    err, base = one_step_error(params, F, O, dim, m, seed=101)

    assert np.mean(hist[-200:]) < 0.5 * np.mean(hist[:200])  # training drove the loss down
    assert err < 0.15                                        # accurate one-step prediction
    assert err < 0.25 * base                                 # far better than 'no change'


def test_regression_matches_cdlm():
    """``regress/system_id.npz``: the pendulum's phase trajectory, and the weights
    and loss history after a short ``identify`` run, reproduce CDLM (states and
    parameters identical; only the sign of covectors crossed the port)."""
    R = load_regress("system_id")

    # the closed phase dynamics of the pendulum
    m, T = float(R["traj_m"]), int(R["traj_T"])
    O = Phi(oscillator(PENDULUM[0], PENDULUM[1], m))
    traj = trajectory(O, R["traj_q0"], R["traj_p0"], T)
    new = np.stack([np.asarray(s) for s in traj])
    np.testing.assert_allclose(new, R["traj_states"], rtol=1e-10, atol=1e-12)

    # the learner: trained parameters after `steps` displacement steps
    n_traj, T, hidden, steps, seed = (int(v) for v in R["id_int_params"])
    m, ic_scale, eta = (float(v) for v in R["id_float_params"])
    params, _, hist, _, _, _ = identify(
        PENDULUM, m=m, n_traj=n_traj, T=T, ic_scale=ic_scale, hidden=hidden,
        eta=eta, steps=steps, seed=seed,
    )
    np.testing.assert_allclose(np.asarray(params), R["id_params"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.array(hist), R["id_history"], rtol=1e-10, atol=1e-12)
