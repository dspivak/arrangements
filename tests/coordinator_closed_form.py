"""The closed form of the coordinator that distributes the mistake (book plan P37 (b)(ii)): the forces
written by hand in numpy. Written and run BEFORE ``arrangements/coordinator.py`` existed, to fix the numbers in
the book repo's ``notes/runs/p37-coordinator-70fcc5f2/PREDICTIONS.md``; the tests then check the composite built
in ``arr`` against it. Nothing here builds the composite.

Minutes throughout. ``B`` buses share one route figurine; bus 0 has figurines 0 (its driver), 1 (its vehicle) and 2 (the
route), item 65's three in its order, and bus ``b >= 1`` has ``2b + 1`` (its driver), ``2b + 2`` (its vehicle)
and 2 (the route). A round is one
outcome step. The depot holds a belief per bus over which of its three figurines was wrong and sends figurine ``j``
the report ``r_j = sum_{b contains j} s^b_j e_b`` with ``e_b = t_hat_b - t*_b`` and the share ``s^b_j`` the belief
(``blame="belief"``) or ``1`` (``blame="chain"``, the chain rule: everyone the whole push). A figurine reads the
report as a push on its guess: force ``-r_j g_j'`` on its parameter, and its response is ``told``
(``mu_g / g'^2``: the guess moves by ``mu_g`` times the push) or ``slope`` (``mu_theta``, item 65's).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

from tests.figurines_closed_form import GUESS, theta_for

MU_G, MU_THETA = 0.1, 0.02
SIGMA = 2.0
SEE_WRONG = (0.9, 0.05)       # Pr(saw j wrong | z_j), Pr(saw j wrong | z_i), i != j
SEE_RIGHT = (0.1, 0.95)       # Pr(saw j right | z_j), Pr(saw j right | z_i), i != j
N_ROUNDS = 60
TRUTH_1, TRUTH_2 = (6.0, 0.5, 20.0), (2.0, 0.5, 20.0)
GUESS0 = (2.0, 0.5, 20.0)
PRIOR, WRONG_PRIOR, UNIFORM = (0.7, 0.1, 0.2), (0.1, 0.1, 0.8), (1 / 3, 1 / 3, 1 / 3)


def softmax(l):
    z = np.exp(l - l.max())
    return z / z.sum()


def kind(j: int) -> int:
    """Which of the three mechanisms figurine ``j`` is: 0 driver, 1 vehicle, 2 route."""
    return 2 if j == 2 else (j if j < 2 else (j - 3) % 2)


def bus_figurines(b: int) -> Tuple[int, int, int]:
    """Bus ``b``'s (driver, vehicle, route): bus 0 is (0, 1, 2), bus ``b >= 1`` is (2b + 1, 2b + 2, 2)."""
    return (0, 1, 2) if b == 0 else (2 * b + 1, 2 * b + 2, 2)


def sighting_nll() -> np.ndarray:
    """``N[i, o] = -log Pr(o | z_i)`` for the six sightings ``o``: ``o = j`` saw ``j`` wrong, ``o = 3 + j`` saw
    ``j`` right."""
    N = np.zeros((3, 6))
    for i in range(3):
        for j in range(3):
            N[i, j] = -np.log(SEE_WRONG[0] if i == j else SEE_WRONG[1])
            N[i, 3 + j] = -np.log(SEE_RIGHT[0] if i == j else SEE_RIGHT[1])
    return N


def comparison_nll(e_b: float, e_other: float) -> np.ndarray:
    """``-log`` likelihood of the other bus's mistake under bus ``b``'s three hypotheses: the route shared,
    ``e_other ~ N(e_b, sigma^2)``; a figurine of its own, the other bus right, ``e_other ~ N(0, sigma^2)``."""
    own = (e_other - 0.0) ** 2 / (2 * SIGMA ** 2)
    shared = (e_other - e_b) ** 2 / (2 * SIGMA ** 2)
    return np.array([own, own, shared])


@dataclass(frozen=True)
class Arm:
    response: str          # "told" or "slope"
    B: int                 # buses
    blame: str             # "belief" or "chain"
    comparison: bool = False

    @property
    def n(self) -> int:
        return 2 * self.B + 1


@dataclass(frozen=True)
class Scenario:
    arm: Arm
    priors: Tuple[Tuple[float, float, float], ...]     # one belief per bus
    sighting: Optional[int] = None                     # a sighting for bus 0 every round: 0..5, or None
    rounds: int = N_ROUNDS
    name: str = ""


def arm(response, B, blame="belief"):
    return Arm(response, B, blame, comparison=(B == 2))


SCENARIOS = {}
for resp in ("told", "slope"):
    SCENARIOS[f"R-{resp}"] = Scenario(arm(resp, 1, "chain"), (UNIFORM,), name="R, the chain rule")
    SCENARIOS[f"U-{resp}"] = Scenario(arm(resp, 1), (UNIFORM,), name="U, uniform, no evidence")
    SCENARIOS[f"S-{resp}"] = Scenario(arm(resp, 1), (UNIFORM,), sighting=0, name="S, a sighting of the driver")
    SCENARIOS[f"P-{resp}"] = Scenario(arm(resp, 1), (PRIOR,), name="P, the prior")
    SCENARIOS[f"C-{resp}"] = Scenario(arm(resp, 2), (UNIFORM, UNIFORM), name="C, comparison")
    SCENARIOS[f"PC-{resp}"] = Scenario(arm(resp, 2), (PRIOR, PRIOR), name="PC, prior and comparison")
    SCENARIOS[f"SPC-{resp}"] = Scenario(arm(resp, 2), (PRIOR, PRIOR), sighting=0, name="SPC, all three")
    SCENARIOS[f"W-{resp}"] = Scenario(arm(resp, 1), (WRONG_PRIOR,), name="W, the wrong prior")
    SCENARIOS[f"WC-{resp}"] = Scenario(arm(resp, 2), (WRONG_PRIOR, WRONG_PRIOR), name="WC, wrong prior, comparison")


def truths(B: int) -> np.ndarray:
    """The world's values per figurine: bus 0 the late driver, bus 1 on time, the route shared."""
    t = np.zeros(2 * B + 1)
    t[2] = TRUTH_1[2]
    for b in range(B):
        tr = TRUTH_1 if b == 0 else TRUTH_2
        d, v, _ = bus_figurines(b)
        t[d], t[v] = tr[0], tr[1]
    return t


def initial_thetas(n: int) -> np.ndarray:
    return np.array([theta_for(kind(j), GUESS0[kind(j)]) for j in range(n)])


def initial_logs(sc: Scenario) -> np.ndarray:
    return np.array([np.log(np.asarray(p, float)) for p in sc.priors])


def forces(theta: np.ndarray, logs: np.ndarray, sc: Scenario, t_star: np.ndarray):
    """One outcome step: the report each figurine is sent, the covector on each belief, and what was read."""
    a = sc.arm
    n = a.n
    g = np.array([GUESS[kind(j)](theta[j])[0] for j in range(n)])
    dg = np.array([GUESS[kind(j)](theta[j])[1] for j in range(n)])
    t_hat = np.array([sum(g[j] for j in bus_figurines(b)) for b in range(a.B)])
    e = t_hat - t_star
    y = np.array([softmax(l) for l in logs])
    report = np.zeros(n)
    for b in range(a.B):
        for i, j in enumerate(bus_figurines(b)):
            report[j] += (y[b, i] if a.blame == "belief" else 1.0) * e[b]
    xi_theta = -report * dg
    xi_y = np.zeros((a.B, 3))
    if sc.sighting is not None:
        xi_y[0] -= sighting_nll()[:, sc.sighting]
    if a.comparison:
        for b in range(a.B):
            other = 1 - b
            xi_y[b] -= comparison_nll(e[b], e[other])
    return xi_theta, xi_y, g, dg, t_hat, e, y, report


def run(sc: Scenario, theta0: Optional[np.ndarray] = None, rounds: Optional[int] = None):
    a = sc.arm
    theta = initial_thetas(a.n) if theta0 is None else np.array(theta0, float)
    logs = initial_logs(sc)
    t_star = np.array([sum(TRUTH_1) if b == 0 else sum(TRUTH_2) for b in range(a.B)])
    R = sc.rounds if rounds is None else rounds
    out = {"theta": [], "logs": [], "y": [], "g": [], "t_hat": [], "e": [], "report": []}
    for t in range(R + 1):
        xi_theta, xi_y, g, dg, t_hat, e, y, report = forces(theta, logs, sc, t_star)
        for key, val in (("theta", theta.copy()), ("logs", logs.copy()), ("y", y), ("g", g),
                         ("t_hat", t_hat), ("e", e), ("report", report)):
            out[key].append(val)
        if t == R:
            break
        if a.response == "told":
            theta = theta + MU_G / dg ** 2 * xi_theta
        else:
            theta = theta + MU_THETA * xi_theta
        logs = np.array([l + (xi - np.dot(softmax(l), xi)) for l, xi in zip(logs, xi_y)])
    return {key: np.array(val) for key, val in out.items()}


def shares(r, B: int) -> np.ndarray:
    """Each figurine's drift from where it started over the run, divided by the four minutes: its share of the
    correction (bus 0's driver started four minutes under its truth; every other figurine started at its truth,
    so its share is what it wrongly absorbed)."""
    return (r["g"][-1] - r["g"][0]) / 4.0


def reuse_error(r) -> float:
    """Bus 0's trained vehicle and route with a fresh driver on a bus (2, 1/2, 20): the first prediction's
    error."""
    g = r["g"][-1]
    return float(GUESS0[0] + g[1] + g[2] - sum(TRUTH_2))


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    for key, sc in SCENARIOS.items():
        r = run(sc)
        B = sc.arm.B
        sh = shares(r, B)
        y_end = r["y"][-1]
        line = (f"{key:10s} {sc.name:30s} shares bus0 (driver, vehicle, route) "
                f"({sh[0]:+.3f}, {sh[1]:+.3f}, {sh[2]:+.3f})")
        if B == 2:
            line += f"; bus1 (driver, vehicle) ({sh[3]:+.3f}, {sh[4]:+.3f})"
        line += (f"; |e| at the end {np.abs(r['e'][-1]).max():.1e}; belief bus0 at the end {y_end[0]}; "
                 f"reuse error {reuse_error(r):+.3f}")
        print(line)
        if sc.arm.comparison or sc.sighting is not None:
            yr = r["y"][:, 0, 2]
            below = np.argmax(yr < 0.01) if (yr < 0.01).any() else None
            below5 = np.argmax(yr < 0.05) if (yr < 0.05).any() else None
            print(f"             bus0's belief on the route by round: {yr[:8]} ... first under 0.05 at round "
                  f"{below5}, under 0.01 at round {below}; belief on the driver by round {r['y'][:6, 0, 0]}")
        if key == "R-slope":
            r30 = run(sc, rounds=30)
            print(f"             at round 30: end guesses {r30['g'][-1]} (item 65: 2.452, 0.527, 23.521)")
        if key.startswith("U-slope"):
            print(f"             item 65 at round 30 shares (-0.887, 0.007, 0.880)")
