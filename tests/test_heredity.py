"""Heredity from relaxation alone (book P52 (b)): item 75's dividing spot with one more chemical, the pattern ``p``.

``spot.pattern_ring`` is item 75's ring of 120 cells (food ``u``, builder ``v``, an exchange with the reservoir, the one-way
reaction ``U + 2V -> 3V``, a diffusion joint per neighboring pair) with ``p`` added to every cell, diffusing like the builder
(``D_p = D_v = 0.256``), and one more box per cell, the WELL, which pulls ``p`` toward ``+1`` or ``-1`` at a strength set by the
builder: its push on ``p`` is ``-v p (p^2 - 1)``, the ``p``-derivative of the two-well ``(v/4)(p^2 - 1)^2`` with ``v`` held.
The well is a 1-form written directly, pushing ``p`` only: written as an exact cost on ``(v, p)`` it would also push the
builder by ``-(p^2 - 1)^2 / 4``, a quarter per unit time at ``p = 0`` whatever the builder, driving it negative in the empty
medium (the hand step that chose the non-exact form; a switch catalyzed by the builder does not consume it). Nothing else:
no copier, no write-enable, no source deeper than its copy (item 74's three hand-built conditions). ``p`` has no exchange with
the reservoir, so in the empty medium it only diffuses. For (B3) the reaction's rate is ``(1 + eps p) u v^2``.

The starting state is set by the experimenter and is where each run starts, nothing that happened before it: item 75's seed
(ten cells at ``(u, v) = (0.5, 0.25)``, the rest at the reservoir's ``(1, 0)``), the seed's cells at ``p = +1`` or ``-1``, every
other cell at ``p = 0`` unless a test says otherwise. Settings: ``F = 0.04``, ``k = 0.06``, ``D_u = 0.512``, ``D_v = D_p = 0.256``,
``dt = 0.5``, run to ``t = 12000`` (24000 steps), as item 75. A spot is a run of cells with builder above 0.1; its SIGN is
the sign of its cells' mean ``p``; it is CARRIED if that mean is beyond 0.5 in size.

PREDICTIONS, fixed before any construction was written (Claude Opus 5.5, 49346cc0, 2026-10-15; P52 (b)'s block, sharpened):

(B0) With ``eps = 0`` the pattern never pushes food or builder, so ``(u, v)`` is item 75's run exactly: the composite equals
     its numpy formula to 1e-12, and the spots go 1 -> 2 -> 4 -> 6 at t = 400, 1150, 2325 as there.
(B1) One +1 seed ends with six spots, each carrying +1 (mean p > 0.9); one -1 seed, six carrying -1. Sharpened: (B1') a
     +1 seed in a medium whose every other cell starts at p = -1 ends with six spots carrying +1, so the sign travels with
     the spots against the medium, not with the medium.
(B2) Two seeds of opposite sign at opposite sides of the ring: the ring ends holding spots of both signs, every spot
     carried at every record (every 25 time units), and the number of spots of each sign never falls (no spot flips).
     Since the run is symmetric under p -> -p, the two signs end equal in number.
(B3) B2's ring with the reaction at rate (1 + eps p) u v^2, eps = 0.03: the +1 lineage divides first, and when the ring
     stops changing, the +1 spots are more than half of all spots. Checked also at eps = 0.01 and 0.05, direction only.
     The direction is the agent's guess (a +1 spot holds its builder at lower food), unchecked by hand.

(B3c), ADDED AFTER (B3) RAN (2026-10-15, before this control ran): at eps = 0.03 the -1 lineage died out, its one spot
     never dividing, hemmed between +1 spots, its builder falling 0.335 -> 0.255 before it died at t = 2600; at eps = 0.05
     the ring ended as one band, no spots. The control runs each sign ALONE at each eps, from the one-seed start.
     Predicted: a lone -1 seed at eps = 0.03 divides (ends with at least two spots), so (B3)'s extinction is the +1
     lineage's doing, competition for the food both eat, with no selection step; a lone +1 seed at eps = 0.05 ends as a
     band, so the band is the rate's and not the meeting's.

HOW IT FAILS, the two ways named before the run: (F1) the spot divides because its middle dies; if ``p`` decays to 0 there,
the children start blank and nothing is carried (by hand it should not: the well's pull at builder 0.075 still points to
the parent's sign, and diffusion from both shoulders pushes the same way); (F2) the well is too shallow against diffusion,
so neighboring spots of opposite sign average to 0 (by hand, a spot 14 cells wide with builder near 0.3 holds against the
flux across a gap of six cells, ``0.256 * 2 / 6``, with its mean p moved about 0.01).
Each test says whether it could have come out otherwise. Results: book repo ``notes/runs/p52-heredity-49346cc0/RESULTS.md``.
"""

import jax
import numpy as np

from arrangements.functors import closed_step
from arrangements.spot import (amounts, numpy_pattern_step, pattern_amounts, pattern_ring, pattern_seed,
                               pattern_state_at, run_counting, seed, spot_ring, spot_signs, spots)

jax.config.update("jax_enable_x64", True)

N = 120
EVERY = 50  # steps between records: 25 time units
STEPS = 24000  # t = 12000
_BUILT = {}


def _built(eps=0.0):
    if eps not in _BUILT:
        W = pattern_ring(*pattern_seed(N, [1.0]), eps=eps)
        _BUILT[eps] = (W, closed_step(W, check=2))
    return _BUILT[eps]


def _run(signs, eps=0.0, medium=0.0, steps=STEPS):
    """Run from the seeds; return the ring, the final state, and per record ``(t, signs of the spots, p range off spots)``."""
    W, step = _built(eps)
    s, hist = run_counting(step, pattern_state_at(W, *pattern_seed(N, signs, medium=medium)), steps, EVERY)
    rec = []
    for j, st in enumerate(hist):
        u, v, p = pattern_amounts(W, st)
        off = p[v <= 0.1]
        rec.append(((j + 1) * EVERY * 0.5, spot_signs(v, p), (float(off.min()), float(off.max())) if off.size else None))
    return W, s, rec


def _changes(rec):
    """The records at which the spots' sign tally changes: ``(t, n_plus, n_minus, n_blank)``."""
    out, last = [], None
    for t, sg, _ in rec:
        tally = (sum(x > 0.5 for x in sg), sum(x < -0.5 for x in sg), sum(abs(x) <= 0.5 for x in sg))
        if tally != last:
            out.append((t,) + tally)
            last = tally
    return out


def _report(name, rec):
    ch = _changes(rec)
    t, sg, off = rec[-1]
    worst = min(abs(x) for r in rec for x in r[1]) if any(r[1] for r in rec) else None
    print(f"{name}: tally (t, +, -, blank) {ch}")
    print(f"{name}: final spot means {[round(x, 4) for x in sg]}; p off the spots {off}; least |mean p| over all records {worst}")
    return ch


def test_composite_is_the_chemistry():
    """(B0), first half. Could fail: a copy routed to the wrong box, the well's sign, the joint's third term."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for eps in (0.0, 0.03):
        W, step = _built(eps)
        for _ in range(5):
            u, v, p = rng.uniform(0, 1, N), rng.uniform(0, 0.5, N), rng.uniform(-1.5, 1.5, N)
            a, b, c = pattern_amounts(W, step(pattern_state_at(W, u, v, p)))
            d, e, f = numpy_pattern_step(u, v, p, eps=eps)
            worst = max(worst, np.abs(a - d).max(), np.abs(b - e).max(), np.abs(c - f).max())
    print("worst difference from the numpy step", worst)
    assert worst < 1e-12


def test_no_pattern_feedback_is_item_75():
    """(B0), second half: at eps = 0 food and builder run as item 75's ring, from the same start. Could fail: any push of
    the well or the joint's pattern term reaching u or v."""
    W, step = _built(0.0)
    s, hist = run_counting(step, pattern_state_at(W, *pattern_seed(N, [1.0])), STEPS, STEPS)
    u, v, _ = pattern_amounts(W, s)
    W75 = spot_ring(*seed(N))
    s75, _ = run_counting(closed_step(W75, check=2), W75.P.init, STEPS, STEPS)
    u75, v75 = amounts(W75, s75)
    diff = max(np.abs(u - u75).max(), np.abs(v - v75).max())
    print("t = 12000: largest difference in (u, v) from item 75's ring", diff, "; spots", len(spots(v)), "against", len(spots(v75)))
    assert diff < 1e-10
    assert len(spots(v)) == 6


def test_one_seed_carries_its_sign():
    """(B1): one +1 seed ends with six spots carrying +1; one -1 seed, six carrying -1. Could fail by (F1): the children
    start where the middle died, and p there could fall toward 0."""
    for sign in (1.0, -1.0):
        _, _, rec = _run([sign])
        ch = _report(f"seed {sign:+.0f}", rec)
        assert ch[-1][1:] == ((6, 0, 0) if sign > 0 else (0, 6, 0))
        assert all(abs(x) > 0.5 and np.sign(x) == sign for r in rec for x in r[1])
        assert all(abs(x - sign) < 0.1 for x in rec[-1][1])


def test_sign_travels_against_the_medium():
    """(B1'): a +1 seed in a medium every other cell of which starts at p = -1 ends with six spots carrying +1. Could fail:
    the medium holds 106 cells of -1 against the seed's 10 of +1, and the children move out into it."""
    _, _, rec = _run([1.0], medium=-1.0)
    ch = _report("seed +1 in a -1 medium", rec)
    assert ch[-1][1:] == (6, 0, 0)
    assert all(x > 0.5 for r in rec for x in r[1])


def test_two_lineages_hold_their_signs():
    """(B2): seeds of opposite sign at opposite sides. Both signs at the end, every spot carried at every record, neither
    sign's count ever falls, and the two counts end equal. Could fail by (F2): where lineages meet, neighbors of opposite
    sign could average toward 0, or a child could form where p is near 0."""
    _, _, rec = _run([1.0, -1.0])
    ch = _report("two seeds", rec)
    assert all(abs(x) > 0.5 for r in rec for x in r[1])
    assert all(b[1] >= a[1] and b[2] >= a[2] for a, b in zip(ch, ch[1:]))
    assert ch[-1][1] == ch[-1][2] > 0


def test_a_faster_sign_takes_more_of_the_ring():
    """(B3): B2's ring with the reaction at rate (1 + eps p) u v^2. At eps = 0.03 the +1 lineage divides first and holds
    more than half the spots at the end; the direction also at eps = 0.01 and 0.05. Could fail: a faster reaction could
    instead make a +1 spot stretch and divide later, or the -1 lineage could die out (a sharper result than predicted,
    and reported as such)."""
    results = {}
    for eps in (0.01, 0.03, 0.05):
        _, _, rec = _run([1.0, -1.0], eps=eps)
        ch = _report(f"eps {eps}", rec)
        first = next(c for c in ch if c[1] != c[2])
        results[eps] = (first, ch[-1])
    print("first record where the counts differ, and the end:", results)
    # Held at 0.01 (4 to 2) and 0.03 (6 to 0, the -1 lineage gone by t = 2600); at 0.05 REFUTED in its terms: the ring
    # ends as one band of builder, no spots (``spots`` reads the whole ring as one run), so no sign holds "spots" at all.
    for eps in (0.01, 0.03, 0.05):
        assert results[eps][0][1] > results[eps][0][2]  # the +1 lineage divides first at every eps
    assert results[0.01][1][1:] == (4, 2, 0)
    assert results[0.03][1][1:] == (6, 0, 0)
    assert results[0.05][1][1:] == (1, 0, 0)  # the band, read as one run of cells


def test_each_sign_alone():
    """(B3c), the control: each sign alone at each eps, from one seed. Could fail: a lone -1 seed at eps = 0.03 could die
    or never divide (then (B3)'s extinction is the slower rate's, not the meeting's)."""
    out = {}
    for eps in (0.01, 0.03, 0.05):
        for sign in (1.0, -1.0):
            W, _, rec = _run([sign], eps=eps)
            ch = _changes(rec)
            band = rec[-1][2] is None
            out[(eps, sign)] = (ch, band)
            print(f"eps {eps} alone {sign:+.0f}: tally (t, +, -, blank) {ch}; band at the end: {band}")
    # REFUTED in its first half: a lone -1 seed at eps = 0.03 never divides (one spot, alive, to t = 12000). So in (B3) its
    # failure to divide is the slower rate's, and only its death is the +1 lineage's (alone it lives). Held in its second
    # half: a lone +1 seed at 0.05 ends as a band.
    assert out[(0.03, -1.0)] == ([(25.0, 0, 1, 0)], False)
    assert out[(0.05, 1.0)][1]
    assert out[(0.01, 1.0)][0][-1] == (1700.0, 6, 0, 0)
    assert out[(0.01, -1.0)][0][-1] == (1850.0, 0, 4, 0)
