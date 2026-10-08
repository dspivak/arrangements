"""The closed form of the composed belief paid in arrears (book plan P37 (b)(i)): the forces written by
hand in numpy, with the figurines' guesses in their closed forms. Written and run BEFORE
``arrangements/figurines.py`` existed, to fix the numbers in the book repo's
``notes/runs/p37-composed-belief-3f16cdb9/PREDICTIONS.md``; the tests then check the composite built in
``arr`` against it. Nothing here builds the composite.

Minutes throughout. Three figurines (driver, vehicle, route) with one real state each; the composite's
prediction ``t_hat = sum_j kappa_j g_j(theta_j)``, ``kappa_j = exp(-p_j^2 / 2 tau)`` the invitation; a
bettor with a margin ``m``; the plan's inconvenience every step, the realized loss ``e^2/2`` on the round's
last step, an optional observation cost, an optional per-component account read by each figurine.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence, Tuple

import numpy as np

TAU = 0.8
DRIVER_DT, DRIVER_K = 0.5, 40
VEHICLE_K = 20
ROUTE_BASE, ROUTE_DOWNTOWN = 0.75, 5          # minutes per km off downtown; downtown km (k = 8..12 of 20)
C_INC, T_PREF = 0.1, 30.0
A_OBS = 0.5
N_STEPS, N_ROUNDS = 10, 30


def softplus(x):
    return np.logaddexp(0.0, x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# The figurines' guesses, closed forms, and their derivatives in the state.
# ---------------------------------------------------------------------------


def driver_guess(theta: float) -> Tuple[float, float]:
    """``g = (1 - (1 - dt lam)^K) / lam``, ``lam = e^theta``; returns ``(g, dg/dtheta)``."""
    lam = np.exp(theta)
    r = 1.0 - DRIVER_DT * lam
    g = (1.0 - r ** DRIVER_K) / lam
    dg_dlam = (DRIVER_K * DRIVER_DT * r ** (DRIVER_K - 1) * lam - (1.0 - r ** DRIVER_K)) / lam ** 2
    return float(g), float(dg_dlam * lam)


def vehicle_guess(theta: float) -> Tuple[float, float]:
    """``g = sum_{k<K} ((1 - phi)^{-k} - 1)``, ``phi = softplus(theta)``; returns ``(g, dg/dtheta)``."""
    phi = softplus(theta)
    rho = 1.0 - phi
    k = np.arange(VEHICLE_K)
    g = np.sum(rho ** (-k) - 1.0)
    dg_dphi = np.sum(k * rho ** (-k - 1))
    return float(g), float(dg_dphi * sigmoid(theta))


def route_guess(theta: float) -> Tuple[float, float]:
    """``g = 20 * 0.75 + 5 c``, ``c = e^theta``; returns ``(g, dg/dtheta)``."""
    c = np.exp(theta)
    return float(20 * ROUTE_BASE + ROUTE_DOWNTOWN * c), float(ROUTE_DOWNTOWN * c)


GUESS = (driver_guess, vehicle_guess, route_guess)


BRACKET = ((-12.0, 0.69), (-12.0, 0.5), (-12.0, 4.0))   # the driver needs dt lam < 1, the vehicle phi < 1


def theta_for(which: int, target: float) -> float:
    """The state at which figurine ``which`` guesses ``target`` (bisection on a monotone guess)."""
    lo, hi = BRACKET[which]
    increasing = GUESS[which](hi)[0] > GUESS[which](lo)[0]
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        g = GUESS[which](mid)[0]
        if (g < target) == increasing:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def kappa(p):
    return np.exp(-np.asarray(p, float) ** 2 / (2.0 * TAU))


def dkappa(p):
    p = np.asarray(p, float)
    return -p / TAU * kappa(p)


# ---------------------------------------------------------------------------
# Arms and scenarios.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    """Which copies of the prediction are one-way, what pays the belief, and the margin's mobility."""
    plan_plain: bool
    bet_plain: bool
    observe: bool
    mu_m: float
    name: str = ""


ARMS = {
    "A": Arm(plan_plain=False, bet_plain=True, observe=False, mu_m=0.0, name="A, one-way"),
    "B": Arm(plan_plain=True, bet_plain=True, observe=False, mu_m=0.0, name="B, exact"),
    "C": Arm(plan_plain=False, bet_plain=False, observe=True, mu_m=0.1, name="C, a margin of its own"),
    "D": Arm(plan_plain=True, bet_plain=True, observe=True, mu_m=0.1, name="D, a margin, exact"),
}


@dataclass(frozen=True)
class Scenario:
    truth: Tuple[float, float, float]           # (d*, v*, r*)
    guesses0: Tuple[float, float, float]        # the figurines' initial guesses
    p0: Tuple[float, float, float]              # initial placements
    mu_theta: float = 0.02
    mu_p: float = 0.0
    account: Tuple[int, int, int] = (0, 0, 0)   # which components the account reports
    rounds: int = N_ROUNDS


SCENARIOS = {
    "S0": Scenario(truth=(2.0, 0.5, 20.0), guesses0=(2.0, 0.5, 20.0), p0=(0.0, 0.0, 0.0)),
    "S1": Scenario(truth=(6.0, 0.5, 20.0), guesses0=(2.0, 0.5, 20.0), p0=(0.0, 0.0, 0.0)),
    "S2": Scenario(truth=(6.0, 0.5, 20.0), guesses0=(2.0, 0.5, 20.0), p0=(0.0, 0.0, 0.0), account=(1, 1, 1)),
    "S5": Scenario(truth=(2.0, 0.5, 20.0), guesses0=(2.0, 3.5, 20.0), p0=(0.25, 0.25, 0.25), mu_p=0.005),
    "S6": Scenario(truth=(2.0, 0.5, 20.0), guesses0=(2.0, 3.5, 20.0), p0=(0.25, 0.25, 0.25), mu_p=0.005,
                   account=(1, 1, 1)),
}


def initial_thetas(sc: Scenario) -> np.ndarray:
    return np.array([theta_for(j, sc.guesses0[j]) for j in range(3)])


def forces(theta, p, m, arm: Arm, sc: Scenario, outcome: bool):
    """The forces on (theta, p, m) at one step, the forcing's configuration component by hand."""
    g = np.array([GUESS[j](theta[j])[0] for j in range(3)])
    dg = np.array([GUESS[j](theta[j])[1] for j in range(3)])
    k, dk = kappa(p), dkappa(p)
    t_hat = float(np.sum(k * g))
    t_star = float(sum(sc.truth))
    plan_push = C_INC * sigmoid(T_PREF - t_hat - m)              # +d/dt of -inconvenience
    xi_t = (plan_push if arm.plan_plain else 0.0)
    xi_m = plan_push                                              # the margin moves only where mu_m > 0
    xi_theta = np.zeros(3)
    if outcome:
        e_bet = t_hat + m - t_star
        xi_m += -e_bet
        if arm.bet_plain:
            xi_t += -e_bet
        if arm.observe:
            xi_t += -A_OBS * (t_hat - t_star)
        f = np.asarray(sc.account, float)
        xi_theta += -k * f * (g - np.asarray(sc.truth)) * dg       # the account, read unilaterally
    xi_theta += k * dg * xi_t
    xi_p = dk * g * xi_t
    return xi_theta, xi_p, xi_m, t_hat, g, dg


def run(arm: Arm, sc: Scenario, theta0: Optional[np.ndarray] = None, p0=None, m0: float = 0.0,
        rounds: Optional[int] = None):
    """The trajectory: arrays of theta (T+1, 3), p (T+1, 3), m (T+1), t_hat (T+1), guesses (T+1, 3)."""
    theta = initial_thetas(sc) if theta0 is None else np.array(theta0, float)
    p = np.array(sc.p0 if p0 is None else p0, float)
    m = float(m0)
    R = sc.rounds if rounds is None else rounds
    T = N_STEPS * R
    out = {"theta": [], "p": [], "m": [], "t_hat": [], "g": []}
    for t in range(T + 1):
        outcome = (t % N_STEPS == N_STEPS - 1) and t < T
        xi_theta, xi_p, xi_m, t_hat, g, _ = forces(theta, p, m, arm, sc, outcome)
        for key, val in (("theta", theta.copy()), ("p", p.copy()), ("m", m), ("t_hat", t_hat), ("g", g)):
            out[key].append(val)
        if t == T:
            break
        theta = theta + sc.mu_theta * xi_theta
        p = p + sc.mu_p * xi_p
        m = m + arm.mu_m * xi_m
    return {key: np.array(val) for key, val in out.items()}


def outcome_steps(rounds: int = N_ROUNDS) -> np.ndarray:
    return np.arange(N_STEPS - 1, N_STEPS * rounds, N_STEPS)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    th0 = initial_thetas(SCENARIOS["S1"])
    print("initial states", th0, "guesses", [GUESS[j](th0[j])[0] for j in range(3)],
          "sensitivities", [GUESS[j](th0[j])[1] for j in range(3)])
    print("vehicle state for a guess of 3.5:", theta_for(1, 3.5))

    # Prediction 1: S1's first outcome step by hand.
    sc, arm = SCENARIOS["S1"], ARMS["A"]
    xi_theta, xi_p, xi_m, t_hat, g, dg = forces(th0, np.zeros(3), 0.0, arm, sc, True)
    print(f"S1 first outcome: t_hat {t_hat:.4f}, t* {sum(sc.truth)}, xi_t {sum(sc.truth) - t_hat:.4f}, "
          f"state forces {xi_theta}")

    # Prediction 2: the vehicle at p = 4.
    xi_far, xi_pfar, _, t_far, _, _ = forces(th0, np.array([0.0, 4.0, 0.0]), 0.0, arm, sc, True)
    print(f"vehicle at p=4: force ratio {xi_far[1] / xi_theta[1]:.6e} (e^-10 = {np.exp(-10):.6e}), "
          f"contribution {t_far - (t_hat - g[1]):.6e}, placement force {xi_pfar[1]:.6e}")

    # Predictions 3, 4: S1 and S2 under arm A.
    for name in ("S1", "S2"):
        r = run(arm, SCENARIOS[name])
        gT = r["g"][-1]
        truth = np.array(SCENARIOS[name].truth)
        drift = r["g"] - truth
        print(f"{name}: final t_hat {r['t_hat'][-1]:.4f} (t* {truth.sum()}), end guesses {gT}, "
              f"drift from truth {gT - truth}, route max |drift| {np.abs(drift[:, 2]).max():.4f}, "
              f"|e| at round 30 {abs(r['t_hat'][-1] - truth.sum()):.2e}")
        if name == "S1":
            print("   shares of the 4 minutes:", (gT - truth) / 4.0)
        # Prediction 5: reuse with a fresh driver on a bus (2, 0.5, 20).
        fresh = SCENARIOS["S0"]
        th_reuse = np.array([initial_thetas(fresh)[0], r["theta"][-1][1], r["theta"][-1][2]])
        rr = run(arm, fresh, theta0=th_reuse, rounds=1)
        print(f"   reuse: first-round prediction {rr['t_hat'][0]:.4f} vs t* 22.5, "
              f"error {rr['t_hat'][0] - 22.5:.4f}")

    # Prediction 6: S5 and S6, placements mobile.
    for name in ("S5", "S6"):
        sc5 = SCENARIOS[name]
        th5 = initial_thetas(sc5)
        xi_th, xi_p5, _, t5, g5, _ = forces(th5, np.array(sc5.p0), 0.0, arm, sc5, True)
        r5 = run(arm, sc5)
        kT = kappa(r5["p"][-1])
        print(f"{name}: first outcome t_hat {t5:.4f}, e {t5 - sum(sc5.truth):.4f}, placement forces {xi_p5}, "
              f"ratio route/vehicle {xi_p5[2] / xi_p5[1]:.4f}; end placements {r5['p'][-1]}, "
              f"end kappa {kT}, end guesses {r5['g'][-1]}, end t_hat {r5['t_hat'][-1]:.4f}, "
              f"max |e| after round 5 {np.abs(r5['t_hat'][50:] - sum(sc5.truth)).max():.4f}")

    # Predictions 7, 8: S0 under the four arms.
    sc0 = SCENARIOS["S0"]
    for key in "ABCD":
        a = ARMS[key]
        r0 = run(a, sc0)
        e = r0["t_hat"] - 22.5
        last = slice(N_STEPS * 25, None)
        steps_end = outcome_steps()[25:]                   # the round's last step, the plan's step
        steps_start = steps_end - (N_STEPS - 1)
        tot_true = N_STEPS * C_INC * softplus(T_PREF - 22.5)
        # total cost per round at the late rest: inconvenience at each step + loss at the outcome
        inc = C_INC * softplus(T_PREF - r0["t_hat"] - r0["m"])
        loss = 0.5 * (r0["t_hat"][steps_end] + r0["m"][steps_end] - 22.5) ** 2
        per_round = sum(inc[N_STEPS * k: N_STEPS * (k + 1)].sum() + loss[k - 25] for k in range(25, 30)) / 5
        print(f"arm {a.name}: max |e| {np.abs(e).max():.4e}; last 5 rounds: e at round end "
              f"{e[steps_end].mean():.4f}, at round start {e[steps_start].mean():.4f}, "
              f"m at end {r0['m'][steps_end].mean():.4f}, bet lateness at end "
              f"{(e + r0['m'])[steps_end].mean():.4f}; total cost per round {per_round:.4f} "
              f"vs true-belief, zero-margin {tot_true:.4f}")
