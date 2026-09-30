"""A carved rest that remembers which note was held.

**Question.** The book's Ch 1 (``sec.held_form``) says a form played long enough may "create a place
to rest that it did not have before, which is what memory is." The domain check of 2026-09-29
(``notes/DOMAIN-CHECK-CH1-2026-09-29.md`` in the book repo, Group E2 item 5) found that the carving
model of ``test_carving.py`` is symmetric under ``(q, u) -> (-q, -u)``: a note of ``-0.6`` carves the
identical ``w``, so its slow coordinate records *that* something was held, not *which* note, and a
kick of the fast state across 0 swaps the remembered side for good. This file asks whether a slow
variable tuned by the note's sign, inside one exact cost, remembers which.

**Model.** The reader of ``test_carving.py`` with one more slow coordinate, a bias ``b`` coupled to
the fast state by ``-b q`` and held by its own restoring cost:

    k(q, w, b; u) = q^4/4 - w q^2/2 - (u + b) q
                    + gamma (w - w0)^2/2 + beta (w - w0)^4/4 + rho b^2/2,

``gamma = 1/20``, ``beta = 1/5``, ``w0 = -1/2`` as there; ``rho = 5/2``; the response is the diagonal
sharp ``diag(eta_q, eta_w, eta_b)`` with ``eta_w = eta_b = eta_q / 25``. The bias descends the same
cost, ``b' = b + eta_b (q - rho b)``, so the cross term is reciprocal (the book's accounting holds)
and the note's work reaches ``b`` through ``q``. By hand: the blank rest ``(0, w0, 0)`` is stable iff
``rho > 2``; at ``u = 0`` the carved rests solve ``q^2 = w + 1/rho``, ``b = q/rho``,
``q^2/2 = gamma x + beta x^3`` with ``x = w - w0``, so ``x^3 - 9x/4 + 1/4 = 0`` and
``(q*, w*, b*) = (+-1.15802, 0.94101, +-0.46321)``. At ``(w*, b*)`` the fast landscape has a rest on
the mirror side only if ``b* < 2 (w*/3)^{3/2} = 0.351``, and ``b*`` exceeds that.

**Prediction** (written before any code ran, in the book repo's
``notes/ch1-tests-2026-09-29/carving-which.md``; subagent of session 5765f5dc, Claude Opus 5.5 (1M),
2026-09-29): (1) ``+0.6`` held long enough leaves ``b > 0`` and the rest at ``q > 0``, ``-0.6`` the
mirror; (2) after carving, a kick of ``q`` across 0 no longer swaps the memory, since ``b`` tilts the
fast landscape past the threshold, where the old model swaps; (3) a hold threshold remains, below
which the trace fades; (4) the carved rest is a joint fixed point before any note plays.

**Result.**

1. **Held.** ``+0.6`` settles at ``(1.15802, 0.94101, 0.46321)``, ``-0.6`` at the exact mirror. The
   slow variables now carry the note's sign. (The mirror being exact cannot come out otherwise, by
   the symmetry ``(q, u, b) -> (-q, -u, -b)``; that ``b`` ends on the note's side could have.)
2. **Failed as stated; held in part.** A kick landing above ``q = -0.543`` (0.47 ``q*``) returns to
   the note's side, where the old model swaps on *any* crossing; a deeper kick (``-0.5 q*``, ``-q*``,
   ``-3 q*``) swaps this model's memory too, permanently. The mechanism I had not accounted for:
   while ``q`` is far negative, ``b`` is driven toward ``q/rho`` fast (``eta_b (q - rho b)`` is large),
   falls below the tilting threshold after 34 steps, when ``q`` is at ``-0.72`` and has not left the mirror side (it never rises above ``-0.70``),
   being slowed by the ghost of the vanished mirror rest; so the mirror rest comes back. Tilt is
   necessary for kick-robustness, not sufficient. **Post hoc**, not a prediction: with the bias four
   times slower (``eta_b = 0.0005``) the full-mirror and ``-3 q*`` kicks return, so the missing
   ingredient is timescale separation between the fast escape and the bias's following.
3. **Held, and the number is a surprise.** The threshold is 123 steps (122 fades), against 1337 in
   the old model, and at the threshold ``w`` has barely moved (``-0.453`` at release): the note pushes
   ``(q, b)`` over a low ridge and the reader carves ``w`` after the note stops. The ridge is low
   because the coupling that stores the sign also flattens the blank rest (with ``b`` slaved its
   curvature in ``q`` is ``-w0 - 1/rho = 0.1``, not ``0.5``). A stronger coupling (smaller ``rho``)
   makes the memory hold more firmly and the blank rest easier to tip: ``rho`` 2.2, 2.5, 3, 4, 6 give
   thresholds of about 50, 130, 240, 430, 660 steps.
4. **Held, closed form.** Both carved rests are stable fixed points of the joint dynamics at ``u = 0``
   (Hessian eigenvalues ``0.61, 2.12, 4.16``) and were before any note played. So at the joint level
   carving is still basin selection, now among three basins; what is new is the fast landscape at the
   new slow values, which is now tilted toward the note.

**What this does not show.** The memory is one bit, a sign; a vector of weights storing a pattern
(Hopfield-like) and retrieval from a partial cue are not run. No noise, so thresholds are sharp. The
rest's "newness" is only at the fast level (point 4). A second, opposite note overwriting the memory
is not tested here, though result 2 says a deep enough displacement of the fast state does it.
"""

import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact
from arrangements.coalgebra import configuration
from arrangements.functors import Phi
from arrangements.rvect import diagonal

from tests.chain import IN_POS

ETA_Q = 0.05
ETA_W = 0.002
ETA_B = 0.002
GAMMA = 0.05
BETA = 0.2
RHO = 2.5
W0 = -0.5
U = 0.6
# the carved rest at u = 0, from x^3 - 9x/4 + 1/4 = 0 (module docstring)
X_CARVED = float(max(np.roots([1.0, 0.0, -2.25, 0.25]).real))
W_CARVED = X_CARVED + W0
Q_CARVED = float(np.sqrt(W_CARVED + 1 / RHO))
B_CARVED = Q_CARVED / RHO
CARVED = np.array([Q_CARVED, W_CARVED, B_CARVED])
MIRROR = np.array([-Q_CARVED, W_CARVED, -B_CARVED])


def k(q, w, b, u):
    return (0.25 * q ** 4 - 0.5 * w * q ** 2 - (u + b) * q
            + 0.5 * GAMMA * (w - W0) ** 2 + 0.25 * BETA * (w - W0) ** 4 + 0.5 * RHO * b ** 2)


def grad(q, w, b, u):
    return np.array([q ** 3 - w * q - u - b,
                     -0.5 * q ** 2 + GAMMA * (w - W0) + BETA * (w - W0) ** 3,
                     -q + RHO * b])


def hessian(q, w, b):
    return np.array([[3 * q ** 2 - w, -q, -1.0],
                     [-q, GAMMA + 3 * BETA * (w - W0) ** 2, 0.0],
                     [-1.0, 0.0, RHO]])


def tilt_threshold(w):
    """The bias above which the fast landscape at curvature ``w`` has no rest on the mirror side."""
    return 2 * (max(w, 0.0) / 3) ** 1.5


def reader(s0=(0.0, W0, 0.0)):
    """The reader ``I -> <R^1 | R^3>``: it sends ``(q, w, b)`` and pays ``k`` on the note."""
    Q = diagonal(jnp.array([ETA_Q, ETA_W, ETA_B]))
    return Arrangement(
        P=configuration(Q, init=jnp.array([float(x) for x in s0])),
        in_dim_M=0, out_dim_M=0, in_dim_N=1, out_dim_N=3,
        out_f=lambda s, m_out: s,
        in_f=lambda s, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda s, m_out, n_in: -k(s[0], s[1], s[2], n_in[0])),
        label="carving reader with a bias",
    )


def run_phi(arr, notes):
    O = Phi(arr)
    s, zero_xi, out = O.state, jnp.zeros(3), []
    for u in notes:
        _, _, s = O.with_state(s).run_one(IN_POS, lambda _o, u=u: (zero_xi, jnp.array([u])))
        out.append(np.array(arr.P.position(s)))
    return np.array(out)


def fast(notes, s0=(0.0, W0, 0.0), eta_b=ETA_B, record=False):
    """The same step by hand."""
    q, w, b = (float(x) for x in s0)
    out = []
    for u in notes:
        dq, dw, db = grad(q, w, b, u)
        q, w, b = q - ETA_Q * dq, w - ETA_W * dw, b - eta_b * db
        if record:
            out.append((q, w, b))
    return np.array(out) if record else np.array([q, w, b])


def old_fast(notes, q0, w0):
    """``test_carving.py``'s reader, for the comparison: no bias."""
    q, w = float(q0), float(w0)
    for u in notes:
        dq = q ** 3 - w * q - u
        dw = -0.5 * q ** 2 + GAMMA * (w - W0) + BETA * (w - W0) ** 3
        q, w = q - ETA_Q * dq, w - ETA_W * dw
    return np.array([q, w])


OLD_Q_CARVED, OLD_W_CARVED = np.sqrt(0.5), 0.5


def schedule(hold, after, u=U):
    return np.concatenate([np.full(hold, u), np.zeros(after)])


def outcome(hold, after=40000, u=U):
    return fast(schedule(hold, after, u))


def at(state, target, tol=5e-3):
    return np.allclose(state, target, atol=tol)


def kicked(q_kick, steps=40000, **kw):
    """From the carved rest, the fast state displaced to ``q_kick``, then silence."""
    return fast(np.zeros(steps), s0=(q_kick, W_CARVED, B_CARVED), **kw)


# ---------------------------------------------------------------------------
# the machinery


def test_the_fast_step_is_Phi():
    """The by-hand step is the ``Phi`` of the arrangement, pathwise. Cannot come out otherwise."""
    notes = schedule(60, 40)
    assert np.allclose(run_phi(reader(), notes), fast(notes, record=True), atol=1e-6)


def test_the_accounting_holds():
    """The joint cost never rises along a run, note on or off: the bias is tuned by the same cost
    (reciprocal cross term). Could have come out otherwise with a step too large."""
    notes = schedule(400, 4000)
    path = np.vstack([[0.0, W0, 0.0], fast(notes, record=True)])
    rises = [t for t in range(len(notes))
             if k(*path[t + 1], notes[t]) > k(*path[t], notes[t]) + 1e-12]
    assert not rises, f"the cost rose at steps {rises[:5]}"


def test_the_carved_rests_are_joint_fixed_points():
    """Prediction 4, closed form: both carved rests are stable fixed points at ``u = 0`` and so were
    rests of the joint dynamics before any note played; the blank rest is stable too (``rho > 2``).
    Carving is basin selection at the joint level. Cannot come out otherwise."""
    for s in (CARVED, MIRROR):
        assert np.abs(grad(*s, 0.0)).max() < 1e-12
        assert np.linalg.eigvalsh(hessian(*s)).min() > 0.5
    assert np.linalg.eigvalsh(hessian(0.0, W0, 0.0)).min() > 0      # blank rest, eigenvalue 0.05
    assert RHO > 2
    # the fast landscape at the carved slow values is tilted past the mirror rest
    assert B_CARVED > tilt_threshold(W_CARVED)
    real = [r.real for r in np.roots([1.0, 0.0, -W_CARVED, -B_CARVED]) if abs(r.imag) < 1e-9]
    assert len(real) == 1 and abs(real[0] - Q_CARVED) < 1e-9


# ---------------------------------------------------------------------------
# the results


def test_the_rest_carries_the_notes_sign():
    """Prediction 1: ``+0.6`` held long enough leaves ``b > 0`` and the rest at ``q > 0``; ``-0.6``
    the mirror. **Could have come out otherwise**: ``b`` could have ended at 0 or on either side."""
    plus, minus = outcome(4000, u=U), outcome(4000, u=-U)
    assert at(plus, CARVED, 1e-6) and plus[2] > 0
    assert at(minus, MIRROR, 1e-6) and minus[2] < 0
    # the old model carves the same w from either note: it does not know which
    assert np.allclose(old_fast(schedule(4000, 40000, U), 0, W0)[1],
                       old_fast(schedule(4000, 40000, -U), 0, W0)[1])


def test_a_hold_threshold_remains_and_the_short_trace_fades():
    """Prediction 3: below a hold time the reader returns to the blank rest; above, it carves. Found:
    the threshold is 123 steps (the old model's is 1337), and the note at threshold leaves ``w``
    almost untouched, so the carving happens after the note stops. **Could have come out
    otherwise**: no threshold, or none in reach."""
    lo, hi = 100, 150
    assert not at(outcome(lo), CARVED) and at(outcome(hi), CARVED)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if at(outcome(mid), CARVED):
            hi = mid
        else:
            lo = mid
    assert (lo, hi) == (122, 123)
    assert at(outcome(lo), [0.0, W0, 0.0], 1e-2)
    released = outcome(hi, after=0)
    assert released[1] < -0.45, "the note itself carved w"
    trace = outcome(lo, after=2000)
    assert trace[1] > W0 + 0.05 and trace[2] > 0.03, "no trace left after a short note"


def test_a_shallow_kick_across_zero_no_longer_swaps():
    """Prediction 2, the part that held: a kick of the fast state just across 0 returns to the
    note's side, where in the old model any crossing swaps the memory for good. **Could have come out
    otherwise**."""
    for f in (0.01, 0.2, 0.4):
        assert at(kicked(-f * Q_CARVED), CARVED), f"kick to {-f} q* swapped"
        old = old_fast(np.zeros(40000), -f * OLD_Q_CARVED, OLD_W_CARVED)
        assert abs(old[0] + OLD_Q_CARVED) < 1e-3, f"old model did not swap at {-f} q*"
    # the critical depth, found: between 0.46 q* and 0.48 q*
    assert at(kicked(-0.46 * Q_CARVED), CARVED) and at(kicked(-0.48 * Q_CARVED), MIRROR)


def test_a_deep_kick_still_swaps():
    """Prediction 2, the part that **failed**: although ``b*`` exceeds the tilting threshold, a kick to
    ``-0.5 q*``, ``-q*`` or ``-3 q*`` swaps the memory permanently. Mechanism, found: ``b`` falls
    below the threshold after 34 steps, with ``q`` at ``-0.72`` and still on the mirror side, slowed by the
    ghost of the vanished mirror rest."""
    for f in (0.5, 1.0, 3.0):
        assert at(kicked(-f * Q_CARVED), MIRROR, 1e-6), f"kick to {-f} q* did not swap"
    path = fast(np.zeros(60), s0=(-Q_CARVED, W_CARVED, B_CARVED), record=True)
    first_below = next(t for t, (q, w, b) in enumerate(path) if b < tilt_threshold(w))
    assert first_below < 40 and path[first_below, 0] < -0.5
    assert path[:, 0].max() < -0.7


def test_post_hoc_a_slower_bias_makes_the_memory_kick_proof():
    """Post hoc, not a prediction and not this file's parameters: with the bias four times slower the
    full-mirror and ``-3 q*`` kicks return to the note's side, while at ``eta_b = 0.001`` they still
    swap. What kick-robustness needs beyond the tilt is timescale separation. Could have come out
    otherwise."""
    for f in (1.0, 3.0):
        assert at(kicked(-f * Q_CARVED, steps=60000, eta_b=0.0005), CARVED)
        assert at(kicked(-f * Q_CARVED, steps=60000, eta_b=0.001), MIRROR)
