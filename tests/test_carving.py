"""A form held long enough carves a place to rest (book Chapter 1, ``sec.held_form`` and the footnote in
``sec.check_claims``).

For a reader of the book. ``carving.reader`` is the code's name, not the book's word: this model does not run or pay, and it is not
poised, so it is not a reader in the sense of ``sec.held_form``, and the book's footnote says "two parts". It is a material
arrangement of a quick part and a slow part, as ``sec.held_form`` has it: "where the slower parts sit sets out the shape that decides where the quick ones can settle". The quick part has
state ``q``, one degree of freedom, and cost ``q^4/4 - u q``, where ``u`` is the note arriving on the reader's port. The
slow part, the tuning, has state ``w``, twenty-five times slower, and cost ``gamma (w - w0)^2/2 + beta (w - w0)^4/4``. A
stateless coupling with cost ``-w q^2/2`` joins them. So ``w`` is the curvature of the quick part's landscape at 0: for
``w < 0`` the quick part has one rest, ``q = 0``, and for ``w > 0`` two, ``q = +-sqrt(w)``. Activity deepens the tuning
and disuse lets it decay. The reader starts at ``w0 = -1/2``, with one rest. Hold the note, then stop, run from the library's folder with it on the path (``PYTHONPATH=. .venv/bin/python example.py``):

    import jax; jax.config.update("jax_enable_x64", True)
    import numpy as np
    from tests.test_carving import is_carved, outcome
    for hold in (1336, 1337):
        state = outcome(hold)                      # (q, w) after the note has stopped and everything has settled
        print(hold, np.round(state, 4), is_carved(state))
    # 1336 [ 0.     -0.4016] False
    # 1337 [0.7051 0.4972] True

One step decides. Held for 1337 steps the note leaves the quick part a rest away from 0, which it keeps with no note
playing. A step less, and it leaves only a mark on the tuning, which fades.

**A new capacity is a newly available one.** The whole reader's rests are fixed by its arrangement: the carved rest was
already a rest of the quick and slow parts together before any note played (the test
``test_the_carved_rest_is_a_joint_fixed_point``). What the note changes is which rests are *available*, meaning rests of
the quick part with the tuning held where it now sits: only ``q = 0`` at ``w = -1/2``, and ``q = +-sqrt(w)`` once ``w > 0``.
Settling moves among the rests available at one ``w``; carving moves ``w`` so that another is. A rest can be available and
not kept: the note of 1336 steps stops at ``w = .211``, where ``q = +-sqrt(w)`` are available, and the tuning falls back and
closes them again; "long enough" is the tuning passing the point, near ``w = .212``, from which its own motion keeps them
open (``test_a_rest_can_be_available_and_not_kept``).

What the tests check, and each test's docstring says whether it could have come out otherwise. The composed reader equals
the one-box reader and the step by hand. There is a sharp hold threshold. The curvature flips sign, and the carved rest
holds against a kick. The joint cost never rises along a run, so the accounting survives the landscape moving. Rain decides
by its rate and not its duration, and spacing does not act monotonically. A tuning that does not descend the same cost
makes the joint cost rise: it is not accountable.

The tests above feed the note as a schedule, ordinary code on the port. The last three do not: the note comes from a part, a
finger, and the whole is one closed composite of the book's category of arrangements. How long the note lasts is how
slowly the finger lets go; one box of states traps for every note; and it survives a two-way spring to a partner in the
same box, and fails for a partner outside it.

Artificial: the sharpness of one step comes from one dimension and no noise; with noise the threshold would be a
probability. The deepening had to saturate, hence the quartic, or there would be no stable memory at all.

For agents: the file's earlier head, with the predictions fixed before the code, what happened and the
weirdness checks, is in ``notes/P31-B-CH1-HISTORY-2026-11-03.md``; predictions in ``notes/CARVING-PREDICTIONS-2026-09-28.md``.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements import carving
from arrangements.arrangement import Arrangement, exact, zero_form
from arrangements.coalgebra import configuration
from arrangements.functors import Phi
from arrangements.interpretation import trivial_omega
from arrangements.rvect import diagonal, euclidean
from arrangements.wiring import compose_seq, tensor_arrangements, trivial_parameter

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


def one_box_reader(q0=0.0, w0=W0):
    """The old reader, one component ``I -> <R^2 | R^2>``: it sends ``(q, w)`` and pays ``k`` on the note."""
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


_COMPOSED = {}


def composed(nonreciprocal=None):
    """The composed reader and its compiled ``Phi`` step: the tuning with the curvature coupling, or,
    with ``nonreciprocal = lam``, the same tuning with the non-reciprocal coupling (only the coupling
    changes)."""
    if nonreciprocal not in _COMPOSED:
        if nonreciprocal is None:
            A = carving.reader([carving.tuning_box(ETA_W, W0, W0, GAMMA, BETA)],
                               [carving.curvature_coupling()], 0.0, ETA_Q)
        else:
            A = carving.reader([carving.tuning_box(ETA_W, W0, W0, GAMMA, BETA)],
                               [carving.nonreciprocal_coupling(nonreciprocal)], 0.0, ETA_Q)
        _COMPOSED[nonreciprocal] = (A, carving.note_step(A))
    return _COMPOSED[nonreciprocal]


def fast(notes, q0=0.0, w0=W0, nonreciprocal=None):
    """``Phi`` of the composed reader on the note schedule from ``(q0, w0)``; every step's ``(q, w)``."""
    A, step = composed(nonreciprocal)
    return carving.play(A, step, notes, record=True, start=[q0, w0])


def by_hand(notes, q0=0.0, w0=W0, slow=None):
    """The one-box reader's step by hand. ``slow(q, w)`` overrides the slow coordinate's descent rule."""
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


def test_the_composed_reader_is_the_one_box_reader():
    """The composed reader's ``Phi`` equals the one-box reader's ``Phi`` pathwise, and the hand loop
    at the runs' settings (4000 steps of the note, 4000 of silence, both arms, from the blank rest and
    from ``q = -0.8``). The prediction, not a premise: the parts' and coupling's costs sum to ``k``."""
    notes = schedule(60, 40)
    np.testing.assert_allclose(fast(notes), run_phi(one_box_reader(), notes), atol=1e-12)
    notes = schedule(4000, 4000)
    np.testing.assert_allclose(fast(notes), by_hand(notes), atol=1e-12)
    np.testing.assert_allclose(fast(notes, q0=-0.8, nonreciprocal=1.0),
                               by_hand(notes, q0=-0.8, slow=lambda q, w: dk_w(q, w) + 0.5 * q ** 2 - q),
                               atol=1e-12)


def test_the_hand_step_is_Phi():
    """The by-hand step is the ``Phi`` of the one-box arrangement, pathwise. Cannot come out otherwise."""
    notes = schedule(60, 40)
    assert np.allclose(run_phi(one_box_reader(), notes), by_hand(notes), atol=1e-6)


def test_the_carved_rest_is_a_joint_fixed_point():
    """Prediction 3, closed form: the carved rest solves both fixed-point equations at ``u = 0``,
    so it was a rest of the joint dynamics before any note played. Cannot come out otherwise; it is
    why a new capacity here is a newly available one: what the note changes is not the whole reader's
    rests but which of them are available with the tuning where it sits."""
    assert abs(dk_q(Q_CARVED, W_CARVED, 0.0)) < 1e-12
    assert abs(dk_w(Q_CARVED, W_CARVED)) < 1e-12
    step = fast(np.zeros(1), q0=Q_CARVED, w0=W_CARVED)[-1]
    assert np.allclose(step, [Q_CARVED, W_CARVED], atol=1e-12)


def test_a_rest_can_be_available_and_not_kept():
    """With the tuning held at ``w > 0`` the quick part has rests at ``q = +-sqrt(w)``: they are available.
    Started on such a rest with no note and the tuning free, the reader keeps it only if the tuning's own motion
    keeps it open: from ``w = .2`` the tuning falls back and the rest closes, from ``w = .25`` it stays. So
    availability opens at ``w = 0``, and the hold threshold is the later point, near ``w = .212``, past which it
    is kept. **Could have come out otherwise**: every available rest might have been kept."""
    for w, kept in ((.05, False), (.1, False), (.2, False), (.25, True), (.3, True)):
        assert abs(dk_q(np.sqrt(w), w, 0.0)) < 1e-12              # a rest of the quick part at this w
        assert is_carved(fast(np.zeros(60000), q0=np.sqrt(w), w0=w)[-1]) == kept, w
    assert fast(schedule(1336, 1))[1335, 1] > 0                    # the note too short still opened it


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
    same cost its fast coordinate does.

    Rebuilt as parts (P43 (b) job 3): the tuning keeps its own cost and only the coupling changes, to
    ``carving.nonreciprocal_coupling``. The hand rule this replaced, ``-lam q + gamma (w - w0)``, had
    also dropped the tuning's saturation, so its ``w`` ran away (to 32) and the quick part's step went
    unstable after step 4859; the numbers recorded for this test (2915 rises, the end ``(1.10, 1.22)``)
    were of the saturated run, which is what this now runs. The biggest rise is 1.0e-4, not 0.80 as
    recorded."""
    lam = 1.0
    # the coupling returning lam q to the tuning (carving.nonreciprocal_coupling):
    # non-reciprocal, d(force_q)/dw = q while d(force_w)/dq = lam, equal only where q = lam
    assert lam != Q_CARVED

    notes = schedule(4000, 4000)
    path = fast(notes, q0=-0.8, nonreciprocal=lam)
    q = np.concatenate([[-0.8], path[:, 0]])
    w = np.concatenate([[W0], path[:, 1]])
    rises = [t for t in range(len(notes)) if k(q[t + 1], w[t + 1], notes[t]) > k(q[t], w[t], notes[t]) + 1e-12]
    assert rises, "the mismatched tuning still descended the cost"
    assert 2900 < len(rises) < 2930
    np.testing.assert_allclose(path[-1], [1.10481607, 1.2206185], atol=1e-6)


# ---------------------------------------------------------------------------
# No schedule: the note comes from a part, and the whole is one closed composite.
# ---------------------------------------------------------------------------

LAM = 10.0                                    # the finger's stiffness
K_Q, K_W, K_F = (0.54, 1.4), (0.3, 1.2), (-0.001, 1.01)   # the one box: every note in [0, .606]
_EMPTY = jnp.zeros(0)


def _finger(eta_f, f0=1.0):
    """State ``f``, cost ``LAM f^2/2``, its out-transfer the note ``U f``. It closes the port, and the quick part pushes back on it."""
    return Arrangement(P=configuration(euclidean(1, eta_f), init=jnp.array([float(f0)])),
                       in_dim_M=1, out_dim_M=0, in_dim_N=0, out_dim_N=0,
                       out_f=lambda q, m: _EMPTY, in_f=lambda q, m, e: U * q,
                       omega=exact(lambda q, m, e: -0.5 * LAM * q[0] ** 2), label="finger")


def _one(eta_f):
    r = carving.reader([carving.tuning_box(ETA_W, W0, W0, GAMMA, BETA)], [carving.curvature_coupling()], 0.0, ETA_Q)
    return compose_seq(r, _finger(eta_f))


def _spring(kappa):
    return Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=2, in_dim_N=0, out_dim_N=0,
                       out_f=lambda q, m: _EMPTY, in_f=lambda q, m, e: _EMPTY,
                       omega=exact(lambda s, m, n: -0.5 * kappa * (m[0] - m[1]) ** 2), label=f"spring({kappa:g})")


def _two(kappa, eta_f):
    """Two quick parts, two tunings, two curvature couplings, a spring between the quick parts, the note copied to both."""
    parts = tensor_arrangements([carving.quick_box(ETA_Q, 0.0, 2), carving.tuning_box(ETA_W, W0, W0, GAMMA, BETA),
                                 carving.quick_box(ETA_Q, 0.0, 2), carving.tuning_box(ETA_W, W0, W0, GAMMA, BETA)])
    idx = jnp.array([0, 2, 3, 5, 1, 4])       # wires out q1, q1, w1, q2, q2, w2; the couplings hear (q1,w1), (q2,w2), (q1,q2)
    route = Arrangement(P=trivial_parameter(), in_dim_M=2, out_dim_M=6, in_dim_N=1, out_dim_N=6,
                        out_f=lambda q, m: m[idx], in_f=lambda q, m, e: jnp.array([e[0], e[0]]),
                        omega=zero_form, label="route, the note copied")
    wired = compose_seq(compose_seq(parts, route),
                        tensor_arrangements([carving.curvature_coupling(), carving.curvature_coupling(), _spring(kappa), carving._port()]))
    return compose_seq(wired, _finger(eta_f))


def _end(arr, steps):
    P, omega = arr.P, arr.omega
    s = jax.lax.fori_loop(0, steps, lambda i, x: P.step(x, omega(P.position(x), _EMPTY, _EMPTY)[0]), P.init)
    return np.asarray(P.position(s), dtype=float)


def _violations(arr, boxes, n=20000, seed=0):
    """How many sampled states of the box, half of them on a face, step out of its interior."""
    P, omega = arr.P, arr.omega
    leaves, treedef = jax.tree_util.tree_flatten(P.init)
    sizes = [int(jnp.size(l)) for l in leaves]
    where = np.cumsum([0] + sizes)

    def one(y):
        s = jax.tree_util.tree_unflatten(treedef, [y[where[i]:where[i] + sizes[i]] for i in range(len(leaves))])
        return P.position(P.step(s, omega(P.position(s), _EMPTY, _EMPTY)[0]))
    rng = np.random.default_rng(seed)
    lo, hi = np.array([b[0] for b in boxes]), np.array([b[1] for b in boxes])
    pts = lo + (hi - lo) * rng.random((n, len(boxes)))
    for i in range(n // 2):
        j = rng.integers(len(boxes))
        pts[i, j] = boxes[j][rng.integers(2)]
    after = np.asarray(jax.jit(jax.vmap(one))(jnp.asarray(pts)))
    return int((np.minimum(after - lo, hi - after).min(axis=1) <= 0).sum())


def test_the_finger_is_a_part_and_no_schedule_is_needed():
    """One closed composite, equal to ``Phi``. How long the note lasts is how slowly the finger lets go. Could fail: the finger
    pushed back on by the quick part might never let go, or let go too fast either way."""
    arr = _one(1 / (LAM * 3000))
    assert arr.label.startswith("(finger .")
    P, omega = arr.P, arr.omega
    s, s_phi, O = P.init, P.init, Phi(arr)
    for _ in range(5):
        _, _, s_phi = O.with_state(s_phi).run_one((_EMPTY, trivial_omega(0)), lambda _o: (_EMPTY, _EMPTY))
        s = P.step(s, omega(P.position(s), _EMPTY, _EMPTY)[0])
    np.testing.assert_allclose(P.position(s), P.position(s_phi), atol=1e-12)
    np.testing.assert_allclose(_end(_one(1 / (LAM * 300)), 80000), [0.0, W0, 0.0], atol=1e-3)
    e = _end(arr, 80000)
    assert K_Q[0] < e[0] < K_Q[1] and K_W[0] < e[1] < K_W[1]


def test_one_box_traps_for_every_note():
    """A single certificate replaces a note-on map and a note-off map: the product of three intervals steps into itself, the
    finger included. Could fail on a face; the faces are checked by hand in ``notes/runs/p31b-ch1-d477d422/CARVING-IN-MG.md``, in the book's repository."""
    assert _violations(_one(1e-3), [K_Q, K_W, K_F]) == 0


def test_the_box_survives_a_two_way_spring_and_fails_for_a_partner_outside_it():
    """Two readers, a spring. In the same box, the spring only helps at each face, at every stiffness tried. A partner in the
    mirror box breaks it. So what is kept is kept under a contract on the partner. Could fail either way."""
    for kappa in (0.1, 1.0, 10.0):
        assert _violations(_two(kappa, 1e-3), [K_Q, K_W, K_Q, K_W, K_F]) == 0
        assert _violations(_two(kappa, 1e-3), [K_Q, K_W, (-1.4, -0.54), K_W, K_F]) > 0
