"""A population tracking a moving optimum: Bürger and Lynch (1995) as an arrangement.

P38, second attempt (the book repository's plan §8.2; Claude Fable 5.1, 27dc74fb, 2026-10-08), at David's "find a
case in the world that yours matches and start using those words and those values for the test, or abandon it."
Predictions and the source's equations, read and checked against the paper's own numbers, are fixed before this
code in the book repository's ``notes/runs/p38-retuning-27dc74fb/PREDICTIONS-moving-optimum.md``.

**The source.** Bürger, R. and Lynch, M. (1995), "Evolution and extinction in a changing environment: a
quantitative-genetic analysis," *Evolution* 49(1):151-163; its deterministic part, pp. 152-154: viability
``W = exp(-(z - theta)^2 / 2 omega^2)`` (eq. 1), the optimum ``theta_t = k t`` (eq. 2), mean viability
``W_bar = sqrt(omega^2 / (V_s + sigma_g^2)) exp(-(g_bar - theta)^2 / 2 (V_s + sigma_g^2))`` with
``V_s = omega^2 + 1`` (eq. 3a), growth ``B W_bar`` with a ceiling at the carrying capacity ``K`` (eq. 3b, p. 152),
``s = sigma_g^2 / (sigma_g^2 + V_s)`` and ``g_bar' = g_bar + s (theta - g_bar)`` (eq. 4), the lag settling at ``k / s``
(eq. 8a), the growth rate ``lambda = B_O exp(-lag^2 / 2 V_lambda)``, ``B_O = B omega / sqrt(V_lambda)``,
``V_lambda = V_s + sigma_g^2`` (eq. 9), and the critical rate ``k_c = s sqrt(2 V_lambda ln B_O)`` (eq. 10).

**The arrangement, in their words.** Three number boxes: the population ``N`` (mobility 1), the mean phenotype
``g_bar`` (mobility ``sigma_g^2``, the additive genetic variance: Lande's equation is
``Delta g_bar = sigma_g^2 d ln W_bar / d g_bar``), the optimum ``theta`` (mobility 1). Four term boxes: viability
and birth (``N (B W_bar - 1)`` on ``N``), carrying capacity (``-max(0, B W_bar N - K)`` on ``N``, their ceiling),
selection on the mean (``-d/d g_bar`` of the cost ``(g_bar - theta)^2 / 2 (V_s + sigma_g^2)``, which is ``-ln W_bar``
up to a constant, on ``g_bar`` and nothing on ``theta``: the environment is a datum the population cannot move),
and the environmental trend (``k`` on ``theta``). One generation is one tick.

**Values** (Fig. 1B and Fig. 3 captions): ``omega^2 = 9``, ``B = 2``, ``K = 32``; ``sigma_g^2 = 0.0703``, the value
eq. (10) implies from the caption's ``k_c = 0.025``, which then predicts Fig. 3's ``k_c = 0.0390, 0.0469, 0.0668`` at
``B = 5, 10, 100`` against the paper's ``0.039, 0.047, 0.067``.

**What this adds to the paper**: only that its model is an arrangement, the mean phenotype a placement descending
its own cost at a mobility the genetic variance sets; the runs reproduce (4), (8a), (9), (10) and Fig. 3, and count
fixed against evolving populations at the same rate of change.
"""

from dataclasses import dataclass, replace
import math

import jax.numpy as jnp
import numpy as np
import pytest

from arrangements.rvect import constant, euclidean

from tests import test_reader_origin as ro

HELD = ro.HELD
EXTINCT = 1.0


@dataclass(frozen=True)
class Model:
    omega2: float = 9.0       # width of the fitness function, eq. (1)
    B: float = 2.0            # offspring per individual
    K: float = 32.0           # carrying capacity
    sigma_g2: float = 0.0703  # additive genetic variance, held constant (p. 153)
    k: float = 0.0125         # rate of environmental change per generation, eq. (2)
    ceiling: bool = True

    @property
    def V_s(self):
        return self.omega2 + 1.0          # sigma_e^2 = 1 (p. 152)

    @property
    def V_lambda(self):
        return self.V_s + self.sigma_g2

    @property
    def s(self):
        return self.sigma_g2 / (self.sigma_g2 + self.V_s)

    @property
    def B_O(self):
        return self.B * math.sqrt(self.omega2) / math.sqrt(self.V_lambda)

    @property
    def k_c(self):
        return self.s * math.sqrt(2.0 * self.V_lambda * math.log(self.B_O)) if self.B_O > 1 else 0.0

    def lag(self):
        return self.k / self.s if self.s > 0 else float("inf")


def mean_viability(m, g, theta):
    return jnp.sqrt(m.omega2 / m.V_lambda) * jnp.exp(-(g - theta) ** 2 / (2.0 * m.V_lambda))   # eq. (3a)


def transitions(m):
    def growth(n):            # viability and birth, eq. (3b): N' = B W_bar N
        N, g, th = n
        return jnp.array([N * (m.B * mean_viability(m, g, th) - 1.0), 0.0, 0.0])

    def ceiling(n):           # the carrying capacity, p. 152
        N, g, th = n
        return jnp.array([-jnp.maximum(0.0, m.B * mean_viability(m, g, th) * N - m.K), 0.0, 0.0])

    def selection(n):         # -d/d g_bar of (g_bar - theta)^2 / 2 V_lambda; nothing on theta
        g, th = n
        return jnp.array([(th - g) / m.V_lambda, 0.0])

    ts = [ro.Transition("viability and birth", ("N", "g", "theta"), covector=growth),
          ro.Transition("selection on the mean", ("g", "theta"), covector=selection),
          ro.Transition("environmental trend", ("theta",), covector=lambda n: jnp.array([m.k]))]
    if m.ceiling:
        ts.insert(1, ro.Transition("carrying capacity", ("N", "g", "theta"), covector=ceiling))
    return ts


def arrangement(m, N0=None, g0=0.0, theta0=0.0):
    boxes = [("N", m.K if N0 is None else N0, euclidean(1, 1.0)),
             ("g", g0, HELD if m.sigma_g2 == 0 else euclidean(1, m.sigma_g2)),
             ("theta", theta0, euclidean(1, 1.0))]
    return ro.network(boxes, transitions(m))


N_, G_, TH_ = 0, 1, 2


def run(m, generations, every=1, **kw):
    return ro.trajectory(arrangement(m, **kw), generations, every)


def lag_of(out):
    return out[:, TH_] - out[:, G_]


def extinction_generation(out, every=1):
    hit = np.where(out[:, N_] < EXTINCT)[0]
    return None if len(hit) == 0 else int((hit[0] + 1) * every)


def holds_K(m, generations=None):
    """``N`` still at ``K`` once the lag has settled: the horizon is ``8 / s`` generations, since the lag reaches
    ``1 - e^-8`` of ``k / s`` by then (at ``omega^2 = 36``, ``s = 0.0019``, a fixed 2000 left the lag at 97.8% and
    read the threshold 2.3% high, 2026-10-08)."""
    if generations is None:
        generations = max(2000, int(math.ceil(8.0 / m.s))) if m.s > 0 else 2000
    out = run(m, generations, every=generations)
    return bool(abs(out[-1, N_] - m.K) < 1e-9)


def threshold_rate(m, lo, hi, steps=12):
    assert holds_K(replace(m, k=lo)) and not holds_K(replace(m, k=hi))
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if holds_K(replace(m, k=mid)) else (lo, mid)
    return lo, hi


# ---------------------------------------------------------------------------


def test_p0_one_generation_is_the_papers_step():
    m = Model(k=0.0, ceiling=False)
    out = run(m, 1, N0=1.0)
    print("P0: growth factor at zero lag", out[-1, N_], "against B_O =", m.B_O)
    assert abs(out[-1, N_] - m.B_O) < 1e-12
    rng = np.random.default_rng(0)
    for _ in range(5):
        g0, th0 = rng.uniform(-3, 3, 2)
        out = run(Model(), 1, g0=float(g0), theta0=float(th0))
        assert abs(out[-1, G_] - (g0 + Model().s * (th0 - g0))) < 1e-12          # eq. (4)
        assert abs(out[-1, TH_] - (th0 + Model().k)) < 1e-12                     # eq. (2)


def test_p1_the_lag_settles_at_k_over_s():
    m = Model(k=0.0125)
    out = run(m, 5000)
    lag = lag_of(out)
    t95 = int(np.argmax(lag >= 0.95 * m.lag())) + 1
    print(f"P1: lag at 5000 = {lag[-1]:.4f} against k/s = {m.lag():.4f}; 95% at generation {t95} "
          f"against {math.log(0.05) / math.log(1 - m.s):.1f}; N = {out[-1, N_]}")
    assert abs(lag[-1] - m.lag()) < 0.005 * m.lag()
    assert abs(t95 - math.log(0.05) / math.log(1 - m.s)) <= 5
    assert np.all(np.abs(out[:, N_] - m.K) < 1e-9)


def test_p2_the_critical_rate_is_equation_10():
    m = Model()
    lo, hi = threshold_rate(m, 0.5 * m.k_c, 1.5 * m.k_c)
    print(f"P2: threshold in [{lo:.5f}, {hi:.5f}] against k_c = {m.k_c:.5f}")
    assert abs(hi - m.k_c) < 0.005 * m.k_c
    out = run(replace(m, k=2 * m.k_c), 400)
    t = extinction_generation(out)
    print(f"P2: at 2 k_c extinct at generation {t} with lag {lag_of(out)[t - 1]:.3f} (predicted 128, 4.22)")
    assert abs(t - 128) <= 2


def test_p3_fixed_against_evolving_at_the_same_rate():
    predicted = {0.005: 778, 0.0125: 326, 0.02: 210}
    fixed, evolving = {}, {}
    for k, t_pred in predicted.items():
        out_f = run(Model(sigma_g2=0.0, k=k), 1000)
        fixed[k] = extinction_generation(out_f)
        out_e = run(Model(k=k), 5000, every=5000)
        evolving[k] = float(out_e[-1, N_])
        print(f"P3: k = {k}: fixed extinct at {fixed[k]} (predicted {t_pred}); evolving N at 5000 = {evolving[k]}")
        assert fixed[k] is not None and abs(fixed[k] - t_pred) <= 2
        assert abs(evolving[k] - Model().K) < 1e-9
    # 3 of 3 extinct against 3 of 3 holding K


def test_p4_width_of_the_fitness_function():
    predicted = {0.5: 0.0276, 1: 0.0396, 2: 0.0393, 4: 0.0335, 9: 0.0250, 16: 0.0196, 36: 0.0134, 99: 0.0082}
    found = {}
    for om2, kc in predicted.items():
        m = Model(omega2=float(om2))
        assert abs(m.k_c - kc) < 0.0001, (om2, m.k_c)
        _, hi = threshold_rate(m, 0.5 * m.k_c, 1.5 * m.k_c, steps=10)
        found[om2] = hi
        print(f"P4: omega^2 = {om2}: threshold {hi:.5f} against k_c = {m.k_c:.5f}")
        assert abs(hi - m.k_c) < 0.01 * m.k_c
    best = max(found, key=found.get)
    print("P4: maximal at omega^2 =", best)
    assert best == 1 and found[0.5] < found[1] and all(found[a] > found[b] for a, b in ((1, 2), (2, 4), (4, 9), (9, 16), (16, 36), (36, 99)))


def test_p5_the_standing_load_is_the_price_of_mobility():
    for g, BO in ((0.0, 1.8974), (0.0703, 1.8907), (1.0, 1.8091), (5.0, 1.5492)):
        m = Model(sigma_g2=g, k=0.0, ceiling=False)
        out = run(m, 1, N0=1.0)
        print(f"P5: sigma_g^2 = {g}: growth factor at rest {out[-1, N_]:.4f} against {BO}")
        assert abs(out[-1, N_] - BO) < 0.0001


def test_p6_birth_rate_reproduces_figure_3():
    for B, kc_paper, kc_eq in ((5.0, 0.039, 0.0390), (10.0, 0.047, 0.0469), (100.0, 0.067, 0.0668)):
        m = Model(B=B)
        assert abs(m.k_c - kc_eq) < 0.0001
        _, hi = threshold_rate(m, 0.5 * m.k_c, 1.5 * m.k_c, steps=10)
        print(f"P6: B = {B}: threshold {hi:.5f} against eq. (10) {m.k_c:.5f} and the paper's {kc_paper}")
        assert abs(hi - m.k_c) < 0.01 * m.k_c and abs(hi - kc_paper) < 0.02 * kc_paper
