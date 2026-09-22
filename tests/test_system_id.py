"""System identification (CDLM rmk.graph_environment, "Predicting a closed system").

``Phi`` of the nonlinear pendulum with a phase parameter runs the system; a
network trained on its consecutive-state pairs -- the ``<o, o.c>`` environment
with ``o = id`` -- learns the system's one-step flow map. We check the held-out
one-step error is small and far below the naive "no change" baseline, so the net
genuinely identified the (nonlinear) dynamics.
"""

import jax.numpy as jnp
import numpy as np

from arrangements.functors import Phi
from arrangements.system_id import PENDULUM, identify, one_step_error, oscillator, trajectory

from tests.chain import load_regress


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
