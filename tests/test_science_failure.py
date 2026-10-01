"""Test (k): publishing failed replications, natively (``arrangements/science_failure.py``).

Predictions, fixed before code in ``notes/SCIENCE-FAILURE-REPORTS-PREDICTIONS-2026-10-01.md`` §2, quoted:

  (k1) Lower-the-seed: 0 of 20 seeds end with any holder on a wrong hypothesis (silent: 0); the truth reaches
  every holder in at least as many seeds as silent (19 of 20 ring, 20 of 20 all-to-all); arrival time per ring
  hop within 10% of silent's; the seed mass planted on hypotheses later rejected is lower than silent's in at
  least 15 of 20 seeds.
  (k2) Counted as evidence: at beta = 0.1 within 2 seeds of silent; as beta grows, the count of seeds in which
  the truth never reaches some holder rises, above zero at beta = 10.
  (k3) In (k2)'s failures, the first report against the truth comes from a holder whose first few local draws
  favoured a neighbouring hypothesis.
  (k4) A report used to lower a seed is safe, one counted as evidence is not.

Settings and verdict rules: that note's "Numbers fixed before code" (stage 2's settings, truth 5.0; report
``exp(-0.5/z)``; ``g(R) = R``; ``beta in {0.1, 1, 10}``).
"""

from __future__ import annotations

import numpy as np
import pytest

from arrangements import science_failure as sf
from arrangements import science_native as nv
from tests import test_science_native as tn
from tests import test_science_scales as r2

BETAS = (0.1, 1.0, 10.0)
TRUE = 10


@pytest.fixture(scope="module")
def runs():
    out = {}
    for w in r2.F_WIRINGS:
        dr = np.stack([r2._f_draws(s, 5.0, tn.T_N) for s in r2.F_SEEDS])
        out[("draws", w)] = dr
        out[("stage2", w)] = nv.conclusions_native_run(dr, r2.F_WIRINGS[w], r2.MUS_F, r2.SLICES, tn.DT, tn.HOLD,
                                                       "seed", "euc", tn.EPS, tn.ALPHA, tn.TAU)[0]
        arms = [("silent", 0.0), ("lower", 0.0)] + [("evidence", b) for b in BETAS]
        for arm, beta in arms:
            res = sf.failure_run(dr, r2.F_WIRINGS[w], r2.MUS_F, r2.SLICES, tn.DT, tn.HOLD, arm, tn.EPS, beta,
                                 tn.TAU)
            out[(arm, beta, w)] = res
            p, z, a, b_, lo, dev = res
            print(f"\n(k run) {w:10s} {arm:8s} beta {beta:4}: NaN {int(np.isnan(p).any(axis=(1, 2, 3)).sum())}; "
                  f"min weight {lo.min():.2e}; max |sum p - 1| {dev.max():.2e}")
    return out


def _stats(res):
    p, z, a, b_, lo, dev = res
    first = tn._arrivals(p, TRUE)
    never = ~np.isfinite(first).all(1)
    wrong_any, wrong_all = tn._wrong(p[:, -1], TRUE)
    top = (p[:, -1].argmax(-1) == TRUE).all(-1)
    rejected = p[:, -1] < 0.01
    planted = (a[:, -1] * rejected).sum((1, 2))
    return dict(first=first, never=never, wrong=wrong_any, wrong_all=wrong_all, top=top, planted=planted,
                lo=float(lo.min()), dev=float(dev.max()))


@pytest.mark.parametrize("wname", list(r2.F_WIRINGS))
def test_silent_is_stage_two_and_every_arm_stays_on_the_simplex(runs, wname):
    p = runs[("silent", 0.0, wname)][0]
    err = float(np.max(np.abs(p - runs[("stage2", wname)])))
    worst = {k[:2]: (_stats(v)["lo"], _stats(v)["dev"]) for k, v in runs.items()
             if k[0] not in ("draws", "stage2") and k[2] == wname}
    print(f"\n(silent) {wname}: max |p - stage 2| {err:.1e}; per arm (min weight, max |sum p - 1|) {worst}")
    assert err <= 1e-12
    assert all(lo >= 0 and dev <= 1e-12 for lo, dev in worst.values())


@pytest.mark.parametrize("wname", list(r2.F_WIRINGS))
def test_k1_lowering_the_seed_is_safe(runs, wname):
    s0, s1 = _stats(runs[("silent", 0.0, wname)]), _stats(runs[("lower", 0.0, wname)])
    sl0, med0 = tn._ring_slope(s0["first"])
    sl1, med1 = tn._ring_slope(s1["first"])
    lower = int((s1["planted"] < s0["planted"]).sum())
    print(f"\n(k1) {wname}: wrong any/all silent {s0['wrong'].sum()}/{s0['wrong_all'].sum()}, lower "
          f"{s1['wrong'].sum()}/{s1['wrong_all'].sum()}; truth reaches every holder silent {20 - s0['never'].sum()}, "
          f"lower {20 - s1['never'].sum()} (never: lower seeds {list(np.nonzero(s1['never'])[0])}); top-at-end "
          f"silent {s0['top'].sum()}, lower {s1['top'].sum()}; slope silent {sl0:.2f} {med0}, lower {sl1:.2f} "
          f"{med1}; planted on rejected lower than silent in {lower} of 20 (median silent "
          f"{np.median(s0['planted']):.3f}, lower {np.median(s1['planted']):.3f})")
    assert s1["wrong"].sum() == 0
    assert 20 - s1["never"].sum() >= 20 - s0["never"].sum()
    if wname == "ring":
        assert 0.9 * sl0 <= sl1 <= 1.1 * sl0
    assert lower >= 15


def _never_counts(runs, wname):
    c = {"silent": int(_stats(runs[("silent", 0.0, wname)])["never"].sum())}
    for b in BETAS:
        c[b] = int(_stats(runs[("evidence", b, wname)])["never"].sum())
    return c


K2 = ("refuted (failure-reports note §3): the truth never reaches some holder in 1, 0, 0, 0 seeds (silent, beta 0.1, "
      "1, 10) on the ring and 0, 0, 0, 0 all-to-all; counted as evidence, reports never kept the truth out")


@pytest.mark.xfail(strict=True, reason=K2)
@pytest.mark.parametrize("wname", list(r2.F_WIRINGS))
def test_k2_reports_counted_as_evidence_keep_the_truth_out(runs, wname):
    c = _never_counts(runs, wname)
    detail = {b: (list(np.nonzero(_stats(runs[("evidence", b, wname)])["never"])[0]),
                  int(_stats(runs[("evidence", b, wname)])["wrong"].sum()),
                  int(_stats(runs[("evidence", b, wname)])["top"].sum())) for b in BETAS}
    print(f"\n(k2) {wname}: seeds where the truth never reaches some holder {c}; per beta (those seeds, wrong any, "
          f"top-at-end every holder) {detail}")
    assert abs(c[0.1] - c["silent"]) <= 2
    assert c[0.1] <= c[1.0] <= c[10.0]
    assert c[10.0] > c["silent"]


def test_k3_first_report_against_the_truth(runs):
    rows, holds = [], []
    for wname in r2.F_WIRINGS:
        dr = runs[("draws", wname)]
        silent_never = _stats(runs[("silent", 0.0, wname)])["never"]
        for b in BETAS:
            p, z, a, _, _, _ = runs[("evidence", b, wname)]
            never = _stats(runs[("evidence", b, wname)])["never"] & ~silent_never
            R = np.asarray(sf.report(z[..., TRUE]))
            for s in np.nonzero(never)[0]:
                hit = R[s] > 0.01
                t = int(np.nonzero(hit.any(1))[0][0])
                j = int(np.argmax(R[s, t]))
                x = dr[s, max(0, t - 4): t + 1, j]
                llr = max(np.sum(-0.5 * (x - mu) ** 2 + 0.5 * (x - 5.0) ** 2) for mu in (4.5, 5.5))
                rows.append((wname, b, int(s), t + 1, j, round(float(llr), 2), bool(llr > 0)))
                holds.append(llr > 0)
    print(f"\n(k3) failing (wiring, beta, seed, draw of first report, holder, best neighbour llr over its 5 draws, "
          f"favours a neighbour): {rows}; holds in {sum(holds)} of {len(holds)} (base rate 0.59)")
    if not holds:
        pytest.skip("no failing seed: (k3) untested")
    assert all(holds)


@pytest.mark.xfail(strict=True, reason="refuted (failure-reports note §3): lowering the seed is safe in both wirings, "
                   "but counting reports as evidence at beta 10 loses the truth in no more seeds than silent")
def test_k4_lower_is_safe_evidence_is_not(runs):
    ok_lower, ev_fails = True, False
    for wname in r2.F_WIRINGS:
        s0, s1 = _stats(runs[("silent", 0.0, wname)]), _stats(runs[("lower", 0.0, wname)])
        ok_lower &= s1["wrong"].sum() == 0 and s1["never"].sum() <= s0["never"].sum()
        ev_fails |= _never_counts(runs, wname)[10.0] > s0["never"].sum()
    print(f"\n(k4) lower safe in both wirings: {ok_lower}; evidence at beta 10 loses the truth more than silent in "
          f"some wiring: {ev_fails}")
    assert ok_lower and ev_fails


# ---------------------------------------------------------------------------
# §4 correction: the report reads the current shortfall b - p (form="shortfall")
# ---------------------------------------------------------------------------
#
#   (k5) Lower-the-seed: §3's (k1) counts within 1 seed each way. (k6) Counted as evidence: the truth, once
#   reached, is lost in 0 of 20 seeds at every beta on both networks (§3: 1 and 2 of 20 at beta 10), since
#   holders who hold the truth stop reporting against it. (k7) The never-reached counts stay within 1 of §3's.
#
# Numbers fixed before code (§4): "lost after reach" = a holder whose weight on the truth exceeded 1/2 after some
# draw and whose top at the last step is not the truth. §3's counts: ring silent 1, lower 0, evidence 0, 0, 1;
# all-to-all 0, 0, 0, 0, 2.


@pytest.fixture(scope="module")
def runs4():
    out = {}
    for w in r2.F_WIRINGS:
        dr = np.stack([r2._f_draws(s, 5.0, tn.T_N) for s in r2.F_SEEDS])
        for arm, beta in [("silent", 0.0), ("lower", 0.0)] + [("evidence", b) for b in BETAS]:
            res = sf.failure_run(dr, r2.F_WIRINGS[w], r2.MUS_F, r2.SLICES, tn.DT, tn.HOLD, arm, tn.EPS, beta,
                                 tn.TAU, form="shortfall")
            out[(arm, beta, w)] = res
            p, lo, dev = res[0], res[4], res[5]
            print(f"\n(§4 run) {w:10s} {arm:8s} beta {beta:4}: NaN {int(np.isnan(p).any(axis=(1, 2, 3)).sum())}; "
                  f"min weight {lo.min():.2e}; max |sum p - 1| {dev.max():.2e}")
    return out


def _lost(res):
    p = res[0]
    lost = np.isfinite(tn._arrivals(p, TRUE)) & (p[:, -1].argmax(-1) != TRUE)
    return lost.any(1)


def _fossils(res, form):
    """(seed, holder) pairs at > 0.99 on the truth at the end that still report it failed (> 0.01)."""
    p, z, a, b = res[:4]
    R = np.asarray(sf.report(z[:, -1, :, TRUE]) if form == "tally"
                   else sf.report_shortfall(b[:, -1, :, TRUE] - p[:, -1, :, TRUE]))
    return int(((p[:, -1, :, TRUE] > 0.99) & (R > 0.01)).sum())


@pytest.mark.parametrize("wname", list(r2.F_WIRINGS))
def test_s4_silent_unchanged_and_simplex_kept(runs, runs4, wname):
    err = float(np.max(np.abs(runs4[("silent", 0.0, wname)][0] - runs[("silent", 0.0, wname)][0])))
    worst = {k[:2]: (float(v[4].min()), float(v[5].max()), bool(np.isnan(v[0]).any()))
             for k, v in runs4.items() if k[2] == wname}
    print(f"\n(§4 silent) {wname}: max |p - §3 silent| {err:.1e}; per arm (min weight, max |sum p - 1|, NaN) {worst}")
    assert err == 0.0
    assert all(lo >= 0 and dev <= 1e-12 and not nan for lo, dev, nan in worst.values())


K5 = ("refuted all-to-all (note §5): planted-on-rejected lower than silent in 17 of 20 against §3's 19; wrong 0 and "
      "reach 20 unchanged")


@pytest.mark.parametrize("wname", ["ring", pytest.param("all-to-all", marks=pytest.mark.xfail(strict=True, reason=K5))])
def test_k5_lower_the_seed_unchanged(runs, runs4, wname):
    s0, s3, s4 = (_stats(r[("silent", 0.0, wname)]) for r in (runs, runs, runs4))
    s3, s4 = _stats(runs[("lower", 0.0, wname)]), _stats(runs4[("lower", 0.0, wname)])
    c3 = (int(s3["wrong"].sum()), int(20 - s3["never"].sum()), int((s3["planted"] < s0["planted"]).sum()))
    c4 = (int(s4["wrong"].sum()), int(20 - s4["never"].sum()), int((s4["planted"] < s0["planted"]).sum()))
    sl = tn._ring_slope(s4["first"])
    print(f"\n(k5) {wname}: lower-the-seed (wrong, reach every holder, planted-on-rejected lower than silent) §3 {c3}, "
          f"§4 {c4}; slope §4 {sl[0]:.2f} {sl[1]}; median planted on rejected §4 {np.median(s4['planted']):.3f} "
          f"(silent {np.median(s0['planted']):.3f})")
    assert all(abs(x - y) <= 1 for x, y in zip(c3, c4))


K6 = ("refuted (note §5): at beta 10 the truth is lost after reach in 1 of 20 (ring, seed 3) and 2 of 20 (all-to-all, "
      "seeds 4 and 8), the whole community collapsing onto 4.5 or 5.5; the fossil reports are gone (0)")


@pytest.mark.xfail(strict=True, reason=K6)
@pytest.mark.parametrize("wname", list(r2.F_WIRINGS))
def test_k6_holding_the_truth_stops_the_reports(runs, runs4, wname):
    rows = {}
    for arm, b in [("silent", 0.0), ("lower", 0.0)] + [("evidence", b) for b in BETAS]:
        l3, l4 = _lost(runs[(arm, b, wname)]), _lost(runs4[(arm, b, wname)])
        rows[(arm, b)] = (int(l3.sum()), int(l4.sum()), list(np.nonzero(l4)[0]),
                          _fossils(runs[(arm, b, wname)], "tally"), _fossils(runs4[(arm, b, wname)], "shortfall"))
    print(f"\n(k6) {wname}: per arm (lost after reach §3, §4, §4 seeds, fossil reports §3, §4) {rows}")
    assert all(rows[("evidence", b)][1] == 0 for b in BETAS)


@pytest.mark.parametrize("wname", list(r2.F_WIRINGS))
def test_k7_never_reached_unchanged(runs, runs4, wname):
    keys = [("lower", 0.0)] + [("evidence", b) for b in BETAS]
    c3 = {k: int(_stats(runs[k + (wname,)])["never"].sum()) for k in keys}
    c4 = {k: int(_stats(runs4[k + (wname,)])["never"].sum()) for k in keys}
    top4 = {k: int(_stats(runs4[k + (wname,)])["top"].sum()) for k in keys}
    wrong4 = {k: int(_stats(runs4[k + (wname,)])["wrong"].sum()) for k in keys}
    print(f"\n(k7) {wname}: never-reaches §3 {c3}, §4 {c4}; §4 every holder's top the truth {top4}; §4 wrong {wrong4}")
    assert all(abs(c3[k] - c4[k]) <= 1 for k in keys)
