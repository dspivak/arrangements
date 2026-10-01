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
