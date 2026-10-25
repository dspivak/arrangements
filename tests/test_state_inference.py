"""P5 (l) stage D: state inference on a hidden physical state against the Kalman posterior
(book ``notes/runs/ch4-redraft-baff60f0/TESTS.md`` §Stage D; results and predictions in
``notes/runs/p5lD-98fc949b/RESULTS.md``). Each test is one piece, run as its own command, and prints
what it asserts (``PYTEST_ADDOPTS="-s -v"``).

What runs in ``arr``: ``state_inference.closed_system`` (the oscillator, the noise counter, the latch and the
belief, closed by a wiring), stepped by ``closed_step`` at durations 1 and 0 (``frame_fn``). Beside it: the
steady-state Kalman filter in numpy on the same observations, the Riccati and Lyapunov recursions.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements import state_inference as si
from arrangements.attention import substitute, unit_arrangement


def _hidden(S, rec):
    """``x[t]`` the oscillator's stored state before frame ``t`` (``x_0`` the start, then ``x_after[t-1]``)."""
    return np.concatenate([S["x0"][None, :], rec["x_after"][:-1]])


def _recursion(S, D, T):
    """``x_{t+1} = F x_t + e_2 w_{c_t}``, ``c_t = t (D + 1)``: the hidden state by hand, and ``H x_t``."""
    F, H, w = S["kalman"]["F"], S["kalman"]["H"], S["w"]
    x = np.asarray(S["x0"], float)
    qs = []
    for t in range(T):
        qs.append(H @ x)
        x = F @ x + np.array([0.0, w[t * (D + 1)]])
    return np.array(qs)


# ---------------------------------------------------------------------------
# D1. Exact: the effort field by blocks, the one-way sensor, the open run, and beta as an operation.
# ---------------------------------------------------------------------------


def test_D1_effort_field_blocks_and_one_way_sensor():
    """At states after 5 and 50 frames, the closed system's effort field at its configuration is, block by
    block: ``-d cost`` at the belief; ``-k q~ + w`` at the oscillator, nothing from the belief; the pushback
    ``Pi (H mu - h)`` at the latch's ``h`` (nonzero, ignored by the latch); ``q~ + v - z`` at ``z``; ``q~`` at the
    noise's ``w`` (the kick's own pullback, ignored by the counter) and ``0`` at ``v``. All to 1e-12."""
    S = si.build(D=0, T=60, newton=True)
    A, b, kd = S["closed"], S["belief"], S["kalman"]
    print("label:", A.label)
    frame = si.frame_fn(A, 0)
    Pi = 1.0 / S["sigma_v"] ** 2
    k, H = S["k"], kd["H"]
    z = jnp.zeros(0)
    cost = b.cost
    s = A.P.init
    for t in range(1, 51):
        s = frame(s)
        if t in (5, 50):
            q = A.P.position_at(s, 1.0)
            xi = np.asarray(A.omega(q, z, z)[0])
            e = jnp.array([q[si.I_H], 1.0])
            want_b = -np.asarray(jax.grad(lambda qb: cost(qb, z, e))(q[5:11]))
            want = np.zeros(11)
            want[5:11] = want_b
            want[si.I_Q] = -k * q[si.I_Q] + q[si.I_W]
            want[si.I_H] = Pi * (H @ np.asarray(q[si.I_MU]) - q[si.I_H])
            want[si.I_Z] = q[si.I_Q] + q[si.I_V] - q[si.I_Z]
            want[si.I_W] = q[si.I_Q]
            want[si.I_V] = 0.0
            gap = np.abs(xi - want)
            print(f"frame {t}: max |effort - expected| by block: osc {gap[0]:.2e}, noise {gap[1:3].max():.2e}, "
                  f"latch {gap[3:5].max():.2e}, belief {gap[5:11].max():.2e}; pushback on h = {xi[si.I_H]:+.4f}, "
                  f"effort at the oscillator = {xi[si.I_Q]:+.4f}")
            assert gap.max() < 1e-12
            assert abs(xi[si.I_H]) > 1e-3        # the pushback exists; the latch ignores it

    # the belief as an operation: validated at build; degree two in e; the unit substitution 1(beta) = beta
    q0 = jnp.asarray(np.random.default_rng(3).standard_normal(6))
    e0 = jnp.array([0.3, 1.2])
    third = jax.jacfwd(jax.hessian(lambda e: cost(q0, z, e)))(e0)
    u = substitute(unit_arrangement(1, 0), [b])
    gaps = [float(jnp.abs(u.sigma(q0, z) - b.sigma(q0, z)).max()),
            float(jnp.abs(u.cost(q0, z, e0) - b.cost(q0, z, e0)))]
    print(f"beta: third derivative in e {float(jnp.abs(third).max()):.1e}; 1(beta) vs beta: readout {gaps[0]:.1e}, cost {gaps[1]:.1e}")
    assert float(jnp.abs(third).max()) < 1e-12 and max(gaps) < 1e-12


def test_D1_physics_unmoved_closed_equals_open_equals_recursion():
    """Over 200 frames at D = 0 and D = 12 (the belief descending at the inverse-Hessian response), the oscillator's
    presented positions in the closed system equal the open run's (oscillator and noise alone) and the
    recursion's, to 1e-9."""
    for D in (0, 12):
        S = si.build(D=D, T=200, newton=True)
        _, rc = si.run_frames(S["closed"], D, 200)
        _, ro = si.run_frames(S["open"], D, 200)
        q_closed, q_open = rc["before"][:, si.I_Q], ro["before"][:, 0]
        q_hand = _recursion(S, D, 200)
        g1, g2 = np.abs(q_closed - q_open).max(), np.abs(q_closed - q_hand).max()
        print(f"D={D}: max |closed - open| = {g1:.2e}, max |closed - recursion| = {g2:.2e}")
        assert g1 < 1e-9 and g2 < 1e-9


# ---------------------------------------------------------------------------
# D2. Fixed known precision: the minimizer is the Kalman posterior; one step at the gain and off it.
# ---------------------------------------------------------------------------


def _compare(S, rec, lo=1):
    r = si.posterior_record(rec)
    kd = S["kalman"]
    xh = si.kalman(r["y"], S["mu0"], kd["F"], kd["H"], kd["K"])      # xh[t] estimates x_t
    x = _hidden(S, rec)
    T = S["T"]
    gap = np.abs(r["mu_p"][lo:T] - xh[lo - 1:T - 1]).max()           # mu_p after frame t <-> xh[t-1]
    err_b = np.sqrt(np.mean((kd["H"] @ (r["mu_p"][lo:T] - x[lo - 1:T - 1]).T) ** 2))
    err_k = np.sqrt(np.mean((kd["H"] @ (xh[lo - 1:T - 1] - x[lo - 1:T - 1]).T) ** 2))
    err_y = np.sqrt(np.mean((r["y"][lo - 1:T - 1] - kd["H"] @ x[lo - 1:T - 1].T) ** 2))
    return gap, err_b, err_k, err_y


def test_D2_fixed_precision_minimizer_is_kalman_posterior():
    """(a) 12 descent steps per observation at the scalar response 1/lambda_max(H' Pi H + P^{-1}): the belief's
    posterior equals the steady-state Kalman mean to 1e-6 over 200 frames. (b) One step at the response P+: to 1e-10.
    (c) One step at P+/2 and P+/4: fixed-gain observers; their gap to the Kalman mean grows as the response shrinks.
    Root-mean-square errors against the hidden presented position reported for all, with the raw observation's."""
    kd0 = si.riccati(0.1, 1.0, 0.05, 0.1)
    A = np.outer(kd0["H"], kd0["H"]) / 0.1 ** 2 + np.linalg.inv(kd0["P"])
    lam = np.linalg.eigvalsh(A)
    g = 1.0 / lam[-1]
    print(f"Hessian eigenvalues {lam}, scalar response g = {g:.5f}, contraction {1 - lam[0] * g:.3f} per step")

    S = si.build(D=12, T=200, G=g)
    _, rec = si.run_frames(S["closed"], 12, 200)
    gap_a, eb, ek, ey = _compare(S, rec)
    print(f"(a) D=12, G=gI: max |belief - Kalman| = {gap_a:.2e}; RMS error vs hidden: belief {eb:.4f}, Kalman {ek:.4f}, observation {ey:.4f}")
    assert gap_a < 1e-6

    S = si.build(D=0, T=200, newton=True)
    _, rec = si.run_frames(S["closed"], 0, 200)
    gap_b, eb, ek, ey = _compare(S, rec)
    print(f"(b) D=0, G=P+: max |belief - Kalman| = {gap_b:.2e}; RMS error vs hidden: belief {eb:.4f}, Kalman {ek:.4f}, observation {ey:.4f}")
    assert gap_b < 1e-10
    assert ek < ey

    gaps = {}
    for c in (0.5, 0.25):
        S = si.build(D=0, T=200, G=c * S["kalman"]["P_plus"])
        _, rec = si.run_frames(S["closed"], 0, 200)
        gap_c, eb, ek, ey = _compare(S, rec, lo=21)
        gaps[c] = gap_c
        r = si.posterior_record(rec)
        xh = si.kalman(r["y"], S["mu0"], S["kalman"]["F"], S["kalman"]["H"], S["kalman"]["K"])
        rms_gap = np.sqrt(np.mean((r["mu_p"][21:200] - xh[20:199]) ** 2))
        print(f"(c) D=0, G={c} P+: frames 21-200: max |belief - Kalman| = {gap_c:.4f}, RMS {rms_gap:.4f}; "
              f"RMS error vs hidden: belief {eb:.4f}, Kalman {ek:.4f}, observation {ey:.4f}")
    assert gaps[0.25] > gaps[0.5] > 1e-3


# ---------------------------------------------------------------------------
# D3. One unknown model parameter, the stiffness.
# ---------------------------------------------------------------------------


def test_D3_unknown_stiffness_is_learned_from_two_starts():
    """D = 1, response P+, eta_k = 5e-5, 4000 frames, from k_hat = 0.05 and 0.2 (true 0.1): each ends within
    0.015 of 0.1, its mean over the last 1000 frames within 0.01, and enters [0.09, 0.11] before frame 3000."""
    for k0 in (0.05, 0.2):
        S = si.build(D=1, T=4000, newton=True, k_hat0=k0, eta_k=5e-5)
        _, rec = si.run_frames(S["closed"], 1, 4000)
        kh = si.posterior_record(rec)["k_hat"]
        inside = np.where(np.abs(kh - 0.1) < 0.01)[0]
        first = int(inside[0]) if len(inside) else -1
        print(f"k_hat0={k0}: final {kh[-1]:.4f}, mean of last 1000 {kh[-1000:].mean():.4f} "
              f"(sd {kh[-1000:].std():.4f}), first within 0.01 of 0.1 at frame {first}; "
              f"at frames 500/1000/2000/3000: {kh[499]:.4f} {kh[999]:.4f} {kh[1999]:.4f} {kh[2999]:.4f}")
        assert abs(kh[-1] - 0.1) < 0.015
        assert abs(kh[-1000:].mean() - 0.1) < 0.01
        assert 0 <= first < 3000


# ---------------------------------------------------------------------------
# D4. Learned precision at one level: a biased rest when the sensor is noisy, a funnel when it is sharp.
# ---------------------------------------------------------------------------


def _precision_run(sigma_v, lambda0, T=6000, eta=0.005):
    S = si.build(D=1, T=T, newton=True, sigma_v=sigma_v, lambda0=lambda0, eta_lambda=eta)
    _, rec = si.run_frames(S["closed"], 1, T)
    r = si.posterior_record(rec)
    H = S["kalman"]["H"]
    resid = r["y"][:-1] - r["mu_p"][1:] @ H            # posterior residual r_t = y_t - H mu_post_t
    return S, r["lam"], resid


def test_D4_learned_precision_one_level():
    """(a) sigma_v = 1 from lambda = 0: e^lambda settles at the stable fixed point of RESULTS §2 (iv) (1.24, above
    the true 1.0), its mean over the last 3000 frames within 10% of it and above 1.10, and equal within 10% to the
    inverse mean squared posterior residual. (b) sigma_v = 1 from lambda = log 100, above the unstable point: lambda
    rises by more than 10. (c) sigma_v = 0.1 from the true lambda = log 100: no fixed point; lambda rises by more
    than 10 and Pi r^2 falls below 0.05 in the last 1000 frames. The funnel at one level."""
    fp = si.precision_fixed_points(0.1, 1.0, 0.05, 1.0)
    fp01 = si.precision_fixed_points(0.1, 1.0, 0.05, 0.1)
    print(f"fixed points: sigma_v=1.0 -> {fp}; sigma_v=0.1 -> {fp01}")
    assert len(fp) == 2 and len(fp01) == 0

    S, lam, resid = _precision_run(1.0, 0.0)
    Pi_mean = np.exp(lam[-3000:]).mean()
    inv_r2 = 1.0 / np.mean(resid[-3000:] ** 2)
    print(f"(a) sigma_v=1, lambda0=0: mean e^lambda over last 3000 = {Pi_mean:.4f} (sd of lambda {lam[-3000:].std():.3f}); "
          f"1/mean r^2 = {inv_r2:.4f}; predicted {fp[0]:.4f}; true 1.0; lambda at 0/1000/3000/6000: "
          f"{lam[0]:.3f} {lam[999]:.3f} {lam[2999]:.3f} {lam[-1]:.3f}")
    assert abs(Pi_mean - fp[0]) < 0.1 * fp[0] and Pi_mean > 1.10
    assert abs(inv_r2 - Pi_mean) < 0.1 * Pi_mean

    S, lam, resid = _precision_run(1.0, float(np.log(100.0)))
    print(f"(b) sigma_v=1, lambda0=log 100: lambda rises {lam[-1] - lam[0]:.2f} over 6000 frames "
          f"(rate {(lam[-1] - lam[0]) / 6000:.5f} per frame; eta/2 = 0.0025)")
    assert lam[-1] - lam[0] > 10

    S, lam, resid = _precision_run(0.1, float(np.log(100.0)))
    Pr2 = np.mean(np.exp(lam[-1000:]) * resid[-1000:] ** 2)
    print(f"(c) sigma_v=0.1, lambda0=log 100 (true): lambda rises {lam[-1] - lam[0]:.2f} "
          f"(rate {(lam[-1] - lam[0]) / 6000:.5f} per frame); mean Pi r^2 over last 1000 = {Pr2:.4f}")
    assert lam[-1] - lam[0] > 10 and Pr2 < 0.05
