"""The closed form of the EM coordinator (book plan P37 (b)(iii)): the forces written by hand in numpy. Written
and run BEFORE ``arrangements/em_coordinator.py`` existed, to fix the numbers in the book repo's
``notes/runs/p37-em-coordinator-1a279037/PREDICTIONS.md``; the tests then check the composite built in ``arr``
against it. Nothing here builds the composite.

Minutes throughout; a round is one outcome step; the figurines and their indexing are ``coordinator_closed_form``'s
(bus 0 is (0, 1, 2), bus ``b >= 1`` is (2b + 1, 2b + 2, 2); the route shared).

THE COORDINATOR'S MODEL. Figurine ``j``'s true minutes are ``t_j = G_j + eps_j``, one LATENT ERROR per figurine,
``eps_j ~ N(0, v_j)`` with ``v_j = V s_j``: the SPLIT ``s = softmax(l)``, a belief over the ``n`` figurines about where
the doubt lies (the belief integrator, rate ``TAU_S``), and the DOUBT ``V = e^Lambda`` (Euclidean on ``Lambda``, rate
``MU_LAMBDA``). Bus ``b`` arrives at ``t*_b = sum_{j in b} t_j + n_b``, ``n_b ~ N(0, SIGMA0^2)`` (the accident); a
sighting of figurine ``j`` is ``o_j = t_j + m``, ``m ~ N(0, SIGMA_S^2)``. ``G_j`` is the guess AS FIRST HEARD (the
RECORD, a frozen state) or the CURRENT guess (MEMORYLESS). Observations ``y = A eps + eta``: a bus row
``y_b = t*_b - sum_{j in b} G_j``, a sighting row ``y_s = o_j - G_j``. With ``D = diag(v)``, ``Sigma = A D A^T + N``:

* the E-step ``eps_hat = D A^T Sigma^{-1} y``, the lift of what was seen into the product of the stories;
* the cost on the coordinator's own state, the marginal ``k = y^T Sigma^{-1} y / 2 + log det Sigma / 2``, which the
  split and the doubt descend (``dk/dv_j = -(a_j^T Sigma^{-1} y)^2 / 2 + a_j^T Sigma^{-1} a_j / 2``);
* the REPORT ``r_j = g_j - G_j - eps_hat_j``, the gap between the guess and the M-step's target, sent down slot ``j``
  and read by the figurine as a push on its guess: force ``-r_j g_j'`` on its parameter, response ``told``
  (``MU_G / g'^2``) or ``slope`` (``MU_THETA``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from tests.coordinator_closed_form import (GUESS0, PRIOR, TRUTH_1, TRUTH_2, UNIFORM, WRONG_PRIOR, bus_figurines,
                                           initial_thetas, kind, softmax)
from tests.figurines_closed_form import GUESS

MU_G, MU_THETA = 0.1, 0.02
SIGMA0, SIGMA_S = 0.1, 0.5
TAU_S, MU_LAMBDA = 0.05, 0.05
V0 = 1.0
O_DRIVER = 6.0            # the sighting: the driver did take six minutes
N_ROUNDS = 60


@dataclass(frozen=True)
class Arm:
    response: str          # "told" or "slope"
    memory: str            # "none" or "record"
    B: int

    @property
    def n(self) -> int:
        return 2 * self.B + 1


@dataclass(frozen=True)
class Scenario:
    arm: Arm
    split0: Tuple[float, ...]        # the prior split over the n figurines (normalized here)
    sighting: bool = False           # bus 0's driver seen every round
    rounds: int = N_ROUNDS
    name: str = ""


def split_for(B: int, per_bus) -> Tuple[float, ...]:
    """A per-bus prior (driver, vehicle, route) laid over the ``n`` figurines and normalized."""
    s = np.zeros(2 * B + 1)
    for b in range(B):
        d, v, r = bus_figurines(b)
        s[d], s[v] = per_bus[0], per_bus[1]
    s[2] = per_bus[2]
    return tuple(s / s.sum())


SCENARIOS = {}
for resp in ("told", "slope"):
    for mem in ("none", "record"):
        key = f"{resp}-{mem}"
        SCENARIOS[f"U-{key}"] = Scenario(Arm(resp, mem, 1), split_for(1, UNIFORM), name="U, uniform")
        SCENARIOS[f"P-{key}"] = Scenario(Arm(resp, mem, 1), split_for(1, PRIOR), name="P, the prior")
        SCENARIOS[f"S-{key}"] = Scenario(Arm(resp, mem, 1), split_for(1, UNIFORM), sighting=True, name="S, a sighting")
        SCENARIOS[f"C-{key}"] = Scenario(Arm(resp, mem, 2), split_for(2, UNIFORM), name="C, comparison")
        SCENARIOS[f"PC-{key}"] = Scenario(Arm(resp, mem, 2), split_for(2, PRIOR), name="PC, prior and comparison")
        SCENARIOS[f"SPC-{key}"] = Scenario(Arm(resp, mem, 2), split_for(2, PRIOR), sighting=True, name="SPC, all three")
        SCENARIOS[f"W-{key}"] = Scenario(Arm(resp, mem, 1), split_for(1, WRONG_PRIOR), name="W, the wrong prior")
        SCENARIOS[f"WC-{key}"] = Scenario(Arm(resp, mem, 2), split_for(2, WRONG_PRIOR), name="WC, wrong prior, comparison")


def t_stars(B: int) -> np.ndarray:
    return np.array([sum(TRUTH_1) if b == 0 else sum(TRUTH_2) for b in range(B)])


def observations(G: np.ndarray, t_star: np.ndarray, B: int, sighting: bool):
    """``(y, A, N)``: a row per bus, then the sighting's row."""
    n = 2 * B + 1
    rows, ys, noise = [], [], []
    for b in range(B):
        a = np.zeros(n)
        a[list(bus_figurines(b))] = 1.0
        rows.append(a)
        ys.append(t_star[b] - a @ G)
        noise.append(SIGMA0 ** 2)
    if sighting:
        a = np.zeros(n)
        a[0] = 1.0
        rows.append(a)
        ys.append(O_DRIVER - G[0])
        noise.append(SIGMA_S ** 2)
    return np.array(ys), np.array(rows), np.array(noise)


def em(v: np.ndarray, y: np.ndarray, A: np.ndarray, N: np.ndarray):
    """The E-step ``eps_hat``, the marginal cost ``k`` and ``dk/dv``."""
    Sigma = A @ np.diag(v) @ A.T + np.diag(N)
    Si_y = np.linalg.solve(Sigma, y)
    eps_hat = v * (A.T @ Si_y)
    k = 0.5 * y @ Si_y + 0.5 * np.linalg.slogdet(Sigma)[1]
    Si_A = np.linalg.solve(Sigma, A)
    dk_dv = -0.5 * (A.T @ Si_y) ** 2 + 0.5 * np.einsum("ij,ij->j", A, Si_A)
    return eps_hat, k, dk_dv


def forces(theta, l, Lam, G_state, sc: Scenario, t_star):
    """One step: the report, the covector on the split (before the belief's projection) and on the doubt."""
    a = sc.arm
    n = a.n
    g = np.array([GUESS[kind(j)](theta[j])[0] for j in range(n)])
    dg = np.array([GUESS[kind(j)](theta[j])[1] for j in range(n)])
    G = G_state if a.memory == "record" else g
    s = softmax(l)
    V = np.exp(Lam)
    v = V * s
    y, A, N = observations(G, t_star, a.B, sc.sighting)
    eps_hat, k, dk_dv = em(v, y, A, N)
    report = g - G - eps_hat
    xi_s = -V * dk_dv                     # the covector on the split, s as free coordinates
    xi_Lam = -np.sum(v * dk_dv)           # on log V
    return dict(g=g, dg=dg, s=s, V=V, v=v, eps_hat=eps_hat, report=report, xi_s=xi_s, xi_Lam=xi_Lam, k=k, y=y)


def run(sc: Scenario, rounds: Optional[int] = None):
    a = sc.arm
    n = a.n
    theta = initial_thetas(n)
    l = np.log(np.asarray(sc.split0, float))
    Lam = np.log(V0)
    G_state = np.array([GUESS0[kind(j)] for j in range(n)])      # the record: the guesses as first heard
    t_star = t_stars(a.B)
    R = sc.rounds if rounds is None else rounds
    keys = ("theta", "g", "s", "V", "eps_hat", "report", "xi_s", "k")
    out = {kk: [] for kk in keys}
    for t in range(R + 1):
        f = forces(theta, l, Lam, G_state, sc, t_star)
        f["theta"] = theta.copy()
        for kk in keys:
            out[kk].append(np.array(f[kk]))
        if t == R:
            break
        xi_theta = -f["report"] * f["dg"]
        if a.response == "told":
            theta = theta + MU_G / f["dg"] ** 2 * xi_theta
        else:
            theta = theta + MU_THETA * xi_theta
        l = l + TAU_S * (f["xi_s"] - np.dot(f["s"], f["xi_s"]))
        Lam = Lam + MU_LAMBDA * f["xi_Lam"]
    return {kk: np.array(vv) for kk, vv in out.items()}


def shares(r) -> np.ndarray:
    """Each figurine's drift from where it started over the run, divided by the four minutes."""
    return (r["g"][-1] - r["g"][0]) / 4.0


def reuse_error(r) -> float:
    """Bus 0's trained vehicle and route with a fresh driver on a bus (2, 1/2, 20)."""
    g = r["g"][-1]
    return float(GUESS0[0] + g[1] + g[2] - sum(TRUTH_2))


def categorical_check():
    """The categorical form, by hand: stage (ii)'s Bayes step is EM's M-step for the mixing weight, and the
    responsibilities are stage (ii)'s shares with the posterior in place of the prior. Returns the max differences."""
    from tests.coordinator_closed_form import sighting_nll
    y0 = np.array(PRIOR)
    lik = np.exp(-sighting_nll()[:, 0])          # saw the driver wrong
    bayes = y0 * lik / np.sum(y0 * lik)          # stage (ii)'s step
    # EM in the one-wrong model: the total is flat across z, so the posterior over z is the prior times the
    # sighting's likelihood; the M-step for the mixing weight sets it to that posterior.
    rho = y0 * lik / np.sum(y0 * lik)
    e_b, s2 = -4.0, 25.0
    resp = -rho * s2 / (s2 + SIGMA0 ** 2) * e_b   # the responsibilities (posterior mean error per figurine)
    return float(np.abs(rho - bayes).max()), resp, -rho * e_b


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    d, resp, stage2 = categorical_check()
    print(f"categorical: M-step == stage (ii)'s Bayes to {d:.1e}; responsibilities {resp} vs stage (ii) with the posterior {stage2}")
    for key, sc in SCENARIOS.items():
        r = run(sc)
        B = sc.arm.B
        sh = shares(r)
        line = (f"{key:16s} {sc.name:28s} shares bus0 ({sh[0]:+.3f}, {sh[1]:+.3f}, {sh[2]:+.3f})")
        if B == 2:
            line += f"; bus1 ({sh[3]:+.3f}, {sh[4]:+.3f})"
        s_end = r["s"][-1]
        line += (f"; split end {s_end}; V end {r['V'][-1]:.3f}; max|xi_s - mean| {np.abs(r['xi_s'] - r['xi_s'].mean(axis=1, keepdims=True)).max():.1e}; "
                 f"reuse {reuse_error(r):+.3f}")
        print(line)
        if sc.sighting:
            print(f"                 driver's split by round {r['s'][:12, 0]}")
        if B == 2:
            s2 = r["s"][:, 2]
            below = np.argmax(s2 < 0.05) if (s2 < 0.05).any() else None
            print(f"                 route's split by round {s2[:8]} ... first under 0.05 at round {below}; bus1 end guesses {r['g'][-1][3:]}")
