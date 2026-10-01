"""Test (i): what is built on an idea pays for its excuses, natively (``arrangements/science_incumbent.py``).

Predictions, fixed before code in ``notes/SCIENCE-INCUMBENT-PREDICTIONS-2026-10-01.md`` §2, quoted:

  (i1) ``D = 0`` reproduces test (g4)'s incumbent-only overtake count (26-30 at its settings) exactly.
  (i2) With ``R`` infinite, the overtake count rises with ``D`` (monotone in the four values); the run
  derives the closed form in the regime where the fund never runs dry.
  (i3) With ``R`` small, dependents rebuild early ... [deferred; no test].
  (i4) *Funerals.* ... the gap between the last dependent's death and the overtake; predicted within 3
  steps when death comes first.
  (i5) Who pays: in every arm, the total drawn from the fund equals the total paid in by dependents, to
  round-off.

Test (j), in §5, quoted:

  (j1) ``F = 0`` reproduces (i1)'s overtake counts seed by seed.
  (j2) The overtake count rises with ``F`` (monotone over at least four values spanning two orders of magnitude).
  (j3) *Epicycles.* ... the delay grows like ``log F``, not like ``F``. The run derives the closed form before
  code and fits the slope.
  (j4) The admitted excuses grow more specific ... and, if distance is tied to specificity, further away, so
  their drawing cost rises along the run.
  (j5) With the fund paid by test (i)'s dependents instead of a fixed ``F`` ... [not run unless fixed in §5].

Verdict rules and closed forms are the note's "Numbers fixed before code". (i2) and (i4) were predicted
refuted by those closed forms before any construction code, and are strict xfail from the first run.
"""

import numpy as np
import pytest

from arrangements import science_incumbent as si

DS = (0, 2, 5, 10)
# Closed forms at rest (the note's fixed numbers), n_B* per seed 0-9.
CLOSED = {0: [26, 27, 27, 28, 29, 37, 19, 33, 31, 28], 2: [26, 27, 27, 28, 29, 37, 19, 33, 31, 28],
          5: [26, 27, 27, 28, 29, 37, 19, 33, 31, 28], 10: [26, 29, 27, 28, 29, 37, 20, 34, 31, 28]}
G4_HARD_INCUMBENT = [26, 27, 27, 28, 29, 37, 19, 33, 31, 28]
_RUNS = {}


def arm(D):
    if D not in _RUNS:
        _RUNS[D] = si.summarize(si.run_arm(D), D)
    return _RUNS[D]


@pytest.mark.parametrize("D", DS)
def test_construction_matches_closed_form(D):
    got = [r["nstar"] for r in arm(D)]
    agree = sum(g == c for g, c in zip(got, CLOSED[D]))
    print(f"\n(construction) D={D}: native n* {got}; closed {CLOSED[D]}; agree {agree}/10")
    assert agree >= 9


def test_i1_D0_is_g4():
    got = [r["nstar"] for r in arm(0)]
    print(f"\n(i1) D=0 native n* {got}; (g4) hard incumbent-only {G4_HARD_INCUMBENT}")
    assert got == G4_HARD_INCUMBENT


@pytest.mark.xfail(strict=True, reason="predicted refuted by the closed form before code: the half-subsidy "
                   "raises n* at D=10 in 3 of 10 seeds only (rule: at least 5)")
def test_i2_rises_with_D():
    N = np.array([[r["nstar"] for r in arm(D)] for D in DS])
    mono = int(np.sum(np.all(np.diff(N, axis=0) >= 0, axis=0)))
    rise = int(np.sum(N[-1] > N[0]))
    print(f"\n(i2) n* by D (rows {DS}):\n{N}\nmedians {np.median(N, 1)}; non-decreasing in {mono}/10; "
          f"D=10 > D=0 in {rise}/10")
    assert mono == 10 and rise >= 5


@pytest.mark.parametrize("D", DS)
def test_i5_fund_bookkeeping(D):
    rs = arm(D)
    worst = max(abs(r["F"] - (r["paid"] - r["drawn"])) / max(r["paid"], 1.0) for r in rs)
    literal = sum(r["F"] <= 1e-3 * r["paid"] for r in rs) if D else 10
    print(f"\n(i5) D={D}: paid {[round(r['paid'], 3) for r in rs]}\n     drawn {[round(r['drawn'], 3) for r in rs]}"
          f"\n     final F {[round(r['F'], 4) for r in rs]}; worst |F - (paid - drawn)|/paid {worst:.2e}; "
          f"drawn = paid (F <= 1e-3 paid) in {literal}/10")
    assert worst <= 1e-9


@pytest.mark.xfail(strict=True, reason="predicted refuted by the closed form before code: deaths at count 15, "
                   "overtakes at 19-37, gap 4-22")
def test_i4_funerals():
    ok = 0
    for D in (2, 5, 10):
        rs = arm(D)
        rows = [(r["death"], r["dry"], r["nstar"]) for r in rs]
        first = ["death" if (r["dry"] is None or (r["death"] is not None and r["death"] <= r["dry"])) else "dry"
                 for r in rs]
        gaps = [r["nstar"] - r["death"] for r in rs]
        ok += sum(f == "death" and g <= 3 for f, g in zip(first, gaps))
        print(f"\n(i4) D={D}: (last death, dry, overtake) {rows}\n     first: {first}; gap death->overtake {gaps}")
    assert ok == 30


# --- test (j) -------------------------------------------------------------------------------------
J_CLOSED = {0: [26, 27, 27, 28, 29, 37, 19, 33, 31, 28], 4: [26, 28, 28, 28, 29, 37, 19, 34, 32, 28],
            16: [27, 28, 28, 28, 32, 37, 20, 34, 32, 28], 64: [27, 31, 30, 33, 32, 38, 22, 36, 36, 28],
            256: [30, 33, 35, 35, 32, 38, 23, 36, 36, 33], 1024: [30, 35, 36, 36, 33, 39, 25, 39, 36, 33],
            4096: [31, 37, 36, 37, 39, 39, 28, 39, 39, 36]}
_J = {}


def jarm(F):
    if F not in _J:
        lam, att, pairs = si.run_fixer_arm(F)
        _J[F] = ([int(np.argmax(l < 0)) if (l < 0).any() else None for l in lam], att, pairs, lam)
    return _J[F]


@pytest.mark.parametrize("F", si.J_FS)
def test_j_construction_matches_closed_form(F):
    got, _, pairs, lam = jarm(F)
    agree = sum(g == c for g, c in zip(got, J_CLOSED[F]))
    print(f"\n(j construction) F={F}: native n* {got}; closed {J_CLOSED[F]}; agree {agree}/10; "
          f"pairs {[len(p) for p in pairs]}; NaN {bool(np.isnan(lam).any())}")
    assert agree >= 9 and not np.isnan(lam).any()


def test_j_no_draw_needs_a_fourth_figure():
    big = [float(np.max(np.random.default_rng(sd).normal(3.0, 1, 150)[:60])) for sd in range(10)]
    print(f"\n(j reduction) max B draw per seed {np.round(big, 2)} against 3d = 9")
    assert max(big) < 9.0


def test_j1_F0_is_i1():
    got = jarm(0)[0]
    print(f"\n(j1) F=0 native n* {got}")
    assert got == CLOSED[0]


def test_j2_rises_with_F():
    N = np.array([jarm(F)[0] for F in si.J_FS])
    mono = int(np.sum(np.all(np.diff(N, axis=0) >= 0, axis=0)))
    print(f"\n(j2) n* by F (rows {si.J_FS}):\n{N}\nmedians {np.median(N, 1)}; non-decreasing in {mono}/10")
    assert mono == 10 and np.median(N[-1]) > np.median(N[0])


def test_j3_log_not_linear():
    Fs = np.array(si.J_FS[1:], float)
    m = np.array([np.mean(jarm(int(F))[0]) for F in Fs])
    def fit(x):
        c = np.polyfit(x, m, 1); r = m - np.polyval(c, x)
        return c[0], 1 - np.sum(r ** 2) / np.sum((m - m.mean()) ** 2)
    sl, r2l = fit(np.log2(Fs)); sF, r2F = fit(Fs)
    gains = np.diff(m) / 2.0
    print(f"\n(j3) mean n* {np.round(m, 2)} at F {Fs.astype(int)}; slope per doubling {sl:.3f} (R2 {r2l:.3f}); "
          f"linear in F R2 {r2F:.3f}; gain per doubling between arms {np.round(gains, 2)}")
    assert 0.5 <= sl <= 1.0 and r2l > r2F


def test_j4_drawing_cost_rises_along_run():
    _, att, pairs, _ = jarm(1024)
    pos = 0
    rhos = []
    for sd in range(10):
        pr = pairs[sd]
        js, ws = [], []
        for j in sorted(set(p[0] for p in pr)):
            idx = [i for i, p in enumerate(pr) if p[0] == j]
            a = att[sd, idx]; dl = np.array([pr[i][2] for i in idx])
            if a.sum() > 0.1:
                js.append(j); ws.append(float(a @ dl / a.sum()))
        rj = np.argsort(np.argsort(js)); rw = np.argsort(np.argsort(ws))
        rho = float(np.corrcoef(rj, rw)[0, 1]) if len(js) > 2 else float("nan")
        rhos.append(round(rho, 2)); pos += rho > 0
        spec = [p[1] for p in pr]
        if sd < 3:
            print(f"\n(j4) seed {sd}: excused draws {js}\n     weighted delta {np.round(ws, 2)}; "
                  f"admission counts of pairs {min(spec)}-{max(spec)}")
    print(f"\n(j4) Spearman(draw, weighted delta) per seed {rhos}; positive in {pos}/10")
    assert pos >= 8
