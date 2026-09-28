"""A form played long enough carves a rest that was not there.

The claim is David's (2026-09-28: "i think a new form can start to carve out a new capacity. is
that not something we've imagined yet? this is how memory works, and how river beds work"), and
this file is its first run. The predictions were fixed before any of this code existed, in
``notes/CARVING-PREDICTIONS-2026-09-28.md``; the book's plan holds them as open-mathematics item 8a.
**Every docstring below says whether its test could have come out otherwise**; the ones that could
not are closed-form checks of the machinery, and only the others are results.

The reader is one component ``I -> <R^2 | R^2>`` whose state is a fast coordinate ``q`` and a slow
one ``w``, its response the diagonal sharp ``diag(eta_q, eta_w)`` with ``eta_w << eta_q``. The cost,
with the note ``u`` arriving on the exterior input, is

    k(q, w; u) = q^4/4 - w q^2/2 - u q + gamma (w - w0)^2 / 2 + beta (w - w0)^4 / 4,

so ``w`` is the curvature of the fast landscape at the origin: for ``w < 0`` the reader has one rest,
``q = 0``; for ``w > 0`` it has two, ``q = +- sqrt(w)``. Both coordinates descend this one cost, so
the slow rule is ``w' = w + eta_w (q^2/2 - gamma (w - w0) - beta (w - w0)^3)``: activity deepens,
disuse decays. The reader starts at ``w0 = -1/2``, one rest and no memory.

**A finding from the construction, before any prediction was tested.** The first cost had no quartic
term in ``w``, only ``gamma (w - w0)^2 / 2``. With a quadratic deepening against a linear decay there
is no stable memory at all: for ``gamma > 1/2`` the only rest is the blank one, and for
``gamma < 1/2`` the carved rest exists but is unstable, since on the slaved branch ``q^2 = w`` the
slow flow has ``d/dw [q^2/2 - gamma(w - w0)] = 1/2 - gamma > 0``, and ``w`` ran away to overflow. So a
carved memory needs the deepening to saturate; the quartic is the least that does it. With
``beta (w - w0)^4 / 4`` added, the slaved slow flow ``g(w) = -w/2 + gamma(w - w0) + beta(w - w0)^3``
has a stable carved rest at ``w* = 1/2``, ``q* = +- sqrt(1/2)``, whenever ``gamma + beta = 1/4`` and
``gamma + 3 beta > 1/2``; this file takes ``beta = 1/5``, ``gamma = 1/20``. The blank rest
``(0, w0)`` stays stable, and an unstable rest near ``w = 0.22`` separates the two basins.

Results, against the five predictions (Claude Opus 5 (1M), session 1a70010f, 2026-09-28):

1. **Held.** The hold threshold is 1337 steps and is sharp to one: a note held 1336 steps leaves the
   reader at the blank rest, 1337 leaves it at the carved one, ``(0.7051, 0.4972)`` against the
   predicted ``(sqrt(1/2), 1/2)``, and it stays. **Unpredicted, and a result:** below the threshold
   the tuning is not left untouched but holding a *fading* trace — ``w = -0.402`` where it started at
   ``-0.5`` — which goes back on the tuning's own decay time, ``1/(eta_w gamma) = 10^4`` steps. So a
   note too short to carve still leaves a mark that has to wear off, and what the threshold decides
   is whether the mark holds itself up.
2. **Held.** The fast landscape's curvature at ``q = 0`` flips sign, so the reader ends with a rest
   where it had none, and a kick that does not cross ``q = 0`` decays back into it.
3. **Closed form, as predicted.** The carved rest is a joint fixed point at ``u = 0`` and was one
   before the note played. Carving is basin selection at the joint level; what is new is the fast
   landscape at the slow coordinate's new value. The book must say it that way.
4. **Held.** The joint cost never rises along a run, with the note on or off, so dissipation stays
   non-negative and the accounting is unharmed by the tuning moving.
5. **Held, and it is the warning.** A tuning driven by ``q`` rather than ``q^2 / 2`` is non-reciprocal
   and no longer descends the cost: the joint cost rises on 2915 of 8000 steps, by as much as 0.80,
   and the reader ends at ``(1.10, 1.22)``, at no rest of the exact model. So a carving tuning is
   accountable only while it descends the same cost its fast coordinate does.

6. **Held.** With the note arriving as rain rather than held once — on for a percentage of each
   period — what decides the outcome is the percentage, not the elapsed time: 9% carves and 8% does
   not, at periods 100 and 200 alike. (David, 2026-09-28, asked for this: "should have been something
   like a weekly rainfall … if you get to a certain amount, then something happens.")
7. **Refuted in both halves, and the refutation is the better statement.** I predicted bursts would
   beat drizzle at short periods, the deepening being convex, and drizzle would beat bursts at long
   ones, the tuning decaying between. At a fixed 9%: drizzle carves at periods 100 and 200, the same
   water in bigger helpings every 2000 steps does not, and every 20000 steps it carves again —
   because a single 1800-step burst is longer than the held threshold of 1337 and carves on its own.
   So spacing does not act monotonically, and the long-period win is the held note rather than
   burstiness. For the book: regular light rain keeps a channel that the same water in occasional
   storms does not, unless one storm is big enough to cut the channel by itself.

**Weirdness, read at each scale** (the agent's judgment, not evidence). A cell: a self-holding mark
that needs no rehearsal is ordinary, a bistable circuit or a methylation kept by its own feedback,
and the saturation the construction forced is what any real deepening has. A body: a habit that
sticks only after enough repetition, and a shorter run that fades over its own timescale, is
reasonable; what is weird is the sharpness, one step deciding, which is an artifact of one
dimension and no noise — with noise the threshold would become a probability, as in
``test_spells.py``. A society: the same shape is a practice that becomes self-funding once enough
of the town holds it, which is the commons runs' territory, and the weird part there is that
nothing in this file's cost is anybody's, so the "supply" is implicit. Across all three, the
non-reciprocal case is the warning that a tuning which carves by its own rule, rather than by the
cost the parts pay, cannot be billed to anybody.
"""

import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact
from arrangements.coalgebra import configuration
from arrangements.functors import Phi
from arrangements.rvect import diagonal

from tests.chain import IN_POS

ETA_Q = 0.05          # the fast response
ETA_W = 0.002         # the slow one, twenty-five times slower
GAMMA = 0.05          # the slow coordinate's linear decay
BETA = 0.2            # its saturation; GAMMA + BETA = 1/4 puts the carved rest at w = 1/2
W0 = -0.5             # the reader starts with one rest, at q = 0
U = 0.6               # the note
W_CARVED = 0.5        # by the choice of GAMMA + BETA; the fast rests are then at +-sqrt(1/2)
Q_CARVED = np.sqrt(W_CARVED)


def k(q, w, u):
    return (0.25 * q ** 4 - 0.5 * w * q ** 2 - u * q
            + 0.5 * GAMMA * (w - W0) ** 2 + 0.25 * BETA * (w - W0) ** 4)


def dk_q(q, w, u):
    return q ** 3 - w * q - u


def dk_w(q, w):
    return -0.5 * q ** 2 + GAMMA * (w - W0) + BETA * (w - W0) ** 3


def reader(q0=0.0, w0=W0):
    """The reader component ``I -> <R^2 | R^2>``: it sends ``(q, w)`` and pays ``k`` on the note."""
    Q = diagonal(jnp.array([ETA_Q, ETA_W]))
    init = jnp.array([float(q0), float(w0)])
    return Arrangement(
        P=configuration(Q, init=init), in_dim_M=0, out_dim_M=0, in_dim_N=2, out_dim_N=2,
        out_f=lambda s, m_out: s,
        in_f=lambda s, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda s, m_out, n_in: -k(s[0], s[1], n_in[0])),
        label="carving reader",
    )


def run_phi(arr, notes):
    """``Phi(arr)`` on a note schedule, one exterior input per step; returns the states."""
    O = Phi(arr)
    s, zero_xi, out = O.state, jnp.zeros(2), []
    for u in notes:
        _, _, s = O.with_state(s).run_one(
            IN_POS, lambda _o, u=u: (zero_xi, jnp.array([u, 0.0]))
        )
        out.append(np.array(arr.P.position(s)))
    return np.array(out)


def fast(notes, q0=0.0, w0=W0, slow=None):
    """The same step by hand. ``slow(q, w)`` overrides the slow coordinate's descent rule."""
    q, w, out = float(q0), float(w0), []
    for u in notes:
        dq = dk_q(q, w, u)
        dw = dk_w(q, w) if slow is None else slow(q, w)
        q, w = q - ETA_Q * dq, w - ETA_W * dw
        out.append((q, w))
    return np.array(out)


def schedule(hold, after):
    """``hold`` steps of the note, then ``after`` steps of silence."""
    return np.concatenate([np.full(hold, U), np.zeros(after)])


def outcome(hold, after=40000):
    """Where the reader is left once the note has stopped and everything has settled."""
    return fast(schedule(hold, after))[-1]


def is_carved(state):
    """Whether the reader was left in the carved well rather than the blank rest."""
    return abs(state[0] - Q_CARVED) < 5e-3


# ---------------------------------------------------------------------------
# the machinery


def test_the_fast_step_is_Phi():
    """The by-hand step is the ``Phi`` of the arrangement, pathwise. Cannot come out otherwise."""
    notes = schedule(60, 40)
    assert np.allclose(run_phi(reader(), notes), fast(notes), atol=1e-6)


def test_the_carved_rest_is_a_joint_fixed_point():
    """Prediction 3, closed form: the carved rest solves both fixed-point equations at ``u = 0``,
    so it was a rest of the joint dynamics before any note played. Cannot come out otherwise; it is
    why the book says carving is basin selection at the joint level and a new landscape at the fast
    one."""
    assert abs(dk_q(Q_CARVED, W_CARVED, 0.0)) < 1e-12
    assert abs(dk_w(Q_CARVED, W_CARVED)) < 1e-12
    step = fast(np.zeros(1), q0=Q_CARVED, w0=W_CARVED)[-1]
    assert np.allclose(step, [Q_CARVED, W_CARVED], atol=1e-12)


# ---------------------------------------------------------------------------
# the results


def test_a_hold_threshold_decides_whether_anything_is_carved():
    """Prediction 1: there is a hold time above which the reader keeps a rest it did not have, and
    below which it returns to ``(0, w0)``; the threshold is sharp. **Could have come out otherwise**:
    a gradual drift with no threshold, or no reachable carved state at all."""
    step = 300
    holds = np.arange(step, 9001, step)
    carved = np.array([is_carved(outcome(h)) for h in holds])
    assert carved[-1] and not carved[0], "neither outcome occurs in the swept range"
    assert np.all(carved[np.argmax(carved):]), "carving is not monotone in the hold time"
    lo, hi = int(holds[np.argmax(carved)]) - step, int(holds[np.argmax(carved)])
    while hi - lo > 1:                     # bisect to the step
        mid = (lo + hi) // 2
        if is_carved(outcome(mid)):
            hi = mid
        else:
            lo = mid
    assert is_carved(outcome(hi)) and not is_carved(outcome(lo))
    q_short, w_short = outcome(lo)
    assert abs(q_short) < 1e-2, "a note one step too short left the reader away from the blank rest"
    # Unpredicted, and a result: below the threshold the tuning is left holding a *fading* trace,
    # not nothing. It goes back on the tuning's own decay time, 1 / (eta_w gamma) = 10^4 steps.
    later = outcome(lo, after=320000)[1]
    assert w_short > W0 and later > W0
    assert later < w_short - 0.05, "the sub-threshold trace is not fading back toward w0"


def test_the_reader_ends_with_a_rest_it_did_not_have():
    """Prediction 2: the fast landscape's curvature at ``q = 0`` is positive before the note and
    negative after, so a rest at ``+-sqrt(w)`` exists that did not; and a kick that does not cross
    ``q = 0`` decays back into the carved well. **Could have come out otherwise**: ``w`` could have
    settled at or below zero, leaving the landscape's shape untouched."""
    assert -W0 > 0                                   # curvature at q = 0 before the note
    q_end, w_end = outcome(4000)
    assert -w_end < 0                                # and after
    assert abs(w_end - W_CARVED) < 1e-3
    kicked = fast(np.zeros(40000), q0=q_end * 0.3, w0=w_end)[-1]
    assert abs(kicked[0] - Q_CARVED) < 1e-3, "the carved well does not hold the reader"


def test_the_accounting_survives_the_landscape_moving():
    """Prediction 4: the joint cost never rises along the run, note on or off, so dissipation
    ``-Delta k`` stays non-negative while the tuning itself moves. **Could have come out otherwise**:
    a step large enough to overshoot would raise the cost."""
    notes = schedule(4000, 4000)
    path = fast(notes)
    q = np.concatenate([[0.0], path[:, 0]])
    w = np.concatenate([[W0], path[:, 1]])
    rises = [t for t in range(len(notes)) if k(q[t + 1], w[t + 1], notes[t]) > k(q[t], w[t], notes[t]) + 1e-12]
    assert not rises, f"the cost rose at steps {rises[:5]}"


def rain_outcome(period, pct, steps=60000, after=120000):
    """Rain: the note on for ``pct`` percent of each ``period``, then a long dry spell. The on-steps
    are ``period * pct / 100`` exactly, so two periods at the same percentage really do deliver the
    same water (rounding here is what first made a short period look better than a long one)."""
    on = period * pct // 100
    assert on * 100 == period * pct, "choose a period and percentage with an exact on-time"
    one = np.concatenate([np.full(on, U), np.zeros(period - on)])
    notes = np.concatenate([np.tile(one, max(1, steps // period)), np.zeros(after)])
    return fast(notes)[-1]


def test_rain_decides_by_its_rate_not_its_duration():
    """Prediction 6: with rain arriving on a cycle short against the tuning's own time, what decides
    the outcome is the percentage of the time it rains, and that percentage is the same for two
    different short periods. **Could have come out otherwise**: the threshold could have moved with
    the period, or cycled rain could have failed to carve at any rate."""
    for period in (100, 200):
        assert is_carved(rain_outcome(period, 9)), f"9% did not carve at period {period}"
        assert not is_carved(rain_outcome(period, 8)), f"8% carved at period {period}"


def test_at_one_rate_the_spacing_of_the_rain_decides():
    """Prediction 7, the one I was unsure of, and it is **refuted in both halves**. I predicted bursts
    would beat drizzle at short periods (the deepening is convex in ``q``) and drizzle would beat
    bursts at long ones (the tuning decays between bursts). What happens at a fixed 9%: drizzle carves
    at periods 100 and 200, the same water in bigger helpings every 2000 steps does not, and every
    20000 steps it carves again — because a single 1800-step burst is longer than the held-note
    threshold of 1337 steps and so carves on its own. So the dependence on spacing is not monotone,
    and the long-period win is the held note, not burstiness. **Could have come out otherwise** in
    either direction; the reading for the book is that regular light rain keeps a channel that the
    same water in occasional storms does not, unless one storm is big enough to cut the channel
    by itself."""
    assert is_carved(rain_outcome(200, 9))
    assert not is_carved(rain_outcome(2000, 9)), "intermediate spacing carved after all"
    assert is_carved(rain_outcome(20000, 9)), "the long burst did not carve"
    # the mechanism at the long period: one burst on its own is already a held note
    assert is_carved(outcome(1800)), "a single 1800-step burst did not carve"
    assert not is_carved(outcome(180)), "a single 180-step burst carved"


def test_non_reciprocal_tuning_is_not_accountable():
    """Prediction 5: a tuning driven by ``q`` rather than ``q^2 / 2`` is non-reciprocal — the two
    rules are not one cost's gradient — and then the joint cost rises on some steps, so the spend
    can no longer be read off the cost. **Could have come out otherwise**: the mismatched pair might
    still have descended this cost. The lesson: carving is accountable while the tuning descends the
    same cost its fast coordinate does."""
    lam = 1.0

    def slow(q, w):
        return -lam * q + GAMMA * (w - W0)

    # non-reciprocal: d(force_q)/dw = q, while d(force_w)/dq = lam, equal only where q = lam
    assert lam != Q_CARVED

    notes = schedule(4000, 4000)
    path = fast(notes, q0=-0.8, slow=slow)
    q = np.concatenate([[-0.8], path[:, 0]])
    w = np.concatenate([[W0], path[:, 1]])
    rises = [t for t in range(len(notes)) if k(q[t + 1], w[t + 1], notes[t]) > k(q[t], w[t], notes[t]) + 1e-12]
    assert rises, "the mismatched tuning still descended the cost"
