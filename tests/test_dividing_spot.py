"""A spot that feeds and divides (book P27, copying): ``spot.spot_ring``, a ring of 120 cells fed from a reservoir.

Predictions were written first, in the book repo's ``notes/runs/p27-copying-c2fac8e7/PREDICTIONS.md``; verdicts in
``RESULTS.md`` beside it. Settings: ``F = 0.04``, ``k = 0.06``, ``D_u = 0.512``, ``D_v = 0.256``, ``dt = 0.5`` unless said.
Each test says whether it could have come out otherwise.
"""

import jax
import numpy as np

from arrangements.functors import closed_step
from arrangements.spot import amounts, numpy_step, run_counting, seed, spot_ring, spots, state_at

jax.config.update("jax_enable_x64", True)

N = 120
_BUILT = {}


def _built(F=0.04):
    if F not in _BUILT:
        W = spot_ring(*seed(N), F=F)
        _BUILT[F] = (W, closed_step(W, check=2))
    return _BUILT[F]


def _counts(W, hist, every):
    out, last = [], None
    for j, st in enumerate(hist):
        n = len(spots(amounts(W, st)[1]))
        if n != last:
            out.append(((j + 1) * every * 0.5, n))
            last = n
    return out


def _profile(W, state):
    u, v = amounts(W, state)
    return [(float(np.max(np.take(v, range(a, a + (b - a) % N + 1), mode="wrap"))), (b - a) % N + 1) for a, b in spots(v)]


def test_composite_is_the_chemistry():
    """Prediction 1. Could fail: a cell's copy routed to the wrong joint, a doubled joint or a sign."""
    W, step = _built()
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(5):
        u, v = rng.uniform(0, 1, N), rng.uniform(0, 0.5, N)
        a, b = amounts(W, step(state_at(W, u, v)))
        c, d = numpy_step(u, v)
        worst = max(worst, np.abs(a - c).max(), np.abs(b - d).max())
    print("worst difference from the numpy step", worst)
    assert worst < 1e-12


def test_one_spot_becomes_six_and_they_are_the_same():
    """Predictions 2 and 3. Could fail: the spot could die, spread into a band, or split into unequal spots."""
    W, step = _built()
    s, hist = run_counting(step, W.P.init, 24000, 50)
    counts = _counts(W, hist, 50)
    print("spot count by time", counts)
    assert [n for _, n in counts] == [1, 2, 4, 6]
    for (t, _), want in zip(counts[1:], (400, 1150, 2325)):
        assert abs(t - want) <= 50
    prof = _profile(W, s)
    print("each spot's peak builder and width at t = 12000", prof)
    peaks = [p for p, _ in prof]
    widths = [w for _, w in prof]
    assert max(peaks) / min(peaks) - 1 < 0.01
    assert max(widths) - min(widths) <= 1


def test_the_form_is_the_mediums():
    """Prediction 4: a random blob, not the seed, comes to the same spot and divides the same way. Could fail: the blob
    could leave spots of another height, or a different number."""
    W, step = _built()
    _, ref = run_counting(step, W.P.init, 24000, 24000)
    ref_peak = np.mean([p for p, _ in _profile(W, ref[-1])])
    rng = np.random.default_rng(1)
    u, v = np.ones(N), np.zeros(N)
    u[40:52], v[40:52] = rng.uniform(0.2, 0.8, 12), rng.uniform(0.0, 0.5, 12)
    s, hist = run_counting(step, state_at(W, u, v), 24000, 50)
    counts = _counts(W, hist, 50)
    prof = _profile(W, s)
    print("random blob: count by time", counts, "final", prof, "seeded run's peak", ref_peak)
    assert counts[-1][1] == 6
    assert all(abs(p / ref_peak - 1) < 0.01 for p, _ in prof)


def test_no_supply_no_division():
    """Prediction 5. Could fail: the spot could live on its own store and split anyway."""
    W, step = _built(F=0.0)
    s, _ = run_counting(step, W.P.init, 2000, 2000)
    u, v = amounts(W, s)
    print("F = 0: largest builder at t = 1000", v.max())
    assert v.max() < 1e-6


def test_how_the_middle_goes():
    """Prediction 6: food at the middle falls first, builder there after, and then food returns while builder stays low.
    Could fail: the middle could stay starved of food to the split (the run's first account to David)."""
    W, step = _built()
    c = N // 2
    _, hist = run_counting(step, W.P.init, 800, 50)  # t = 25, 50, ..., 400
    mid = [(float(np.mean(amounts(W, st)[0][c - 1:c + 1])), float(np.mean(amounts(W, st)[1][c - 1:c + 1]))) for st in hist]
    for j, (u, v) in enumerate(mid):
        print(f"t = {(j + 1) * 25:4d}: middle food {u:.3f}, builder {v:.3f}")
    u50, v50 = mid[1]
    u400, v400 = mid[-1]
    assert u50 < 0.35
    assert u400 > 0.6
    assert v400 < 0.5 * v50
