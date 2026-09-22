"""A physics-informed network (deep Ritz), via ``Phi`` of a configuration arrangement.

A field network ``u_theta(x) = x(1-x) g_theta(x)`` (hard Dirichlet BC) is trained
under ``Phi`` (configuration parameter) to minimize the discrete Dirichlet energy
of ``-u'' = f`` on ``[0,1]``. We check the energy decreases and the learned field
matches the discrete Poisson solution ``L u = b`` -- i.e. the network solved the BVP.
"""

import numpy as np

from arrangements.pinn import solve_deep_ritz

from tests.chain import load_regress


def test_deep_ritz_solves_poisson():
    u, u_star, hist, x = solve_deep_ritz(steps=4000, seed=0)
    rel = float(
        np.linalg.norm(np.asarray(u) - np.asarray(u_star))
        / np.linalg.norm(np.asarray(u_star))
    )

    assert hist[-1] < hist[0]  # the Dirichlet energy decreased
    assert rel < 0.08          # the field matches the discrete Poisson solution


def test_regression_matches_cdlm():
    """``regress/pinn.npz``: the learned field, the exact solution, the grid and the
    per-step energy after a short deep-Ritz run reproduce CDLM (the forcing
    ``b - L u`` replaces CDLM's seeded gradient ``L u - b``; the weights' trajectory
    is identical, DESIGN.md §3)."""
    R = load_regress("pinn")
    V, hidden, steps, seed = (int(v) for v in R["int_params"])
    eta = float(R["eta"])
    u, u_star, hist, x = solve_deep_ritz(V, hidden=hidden, eta=eta, steps=steps, seed=seed)
    np.testing.assert_allclose(np.asarray(x), R["x"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.asarray(u_star), R["u_star"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.asarray(u), R["u"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.array(hist), R["history"], rtol=1e-10, atol=1e-12)
