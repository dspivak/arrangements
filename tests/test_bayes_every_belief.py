"""Bayes' rule for every belief over finitely many values (book ``AGENTDOCPLAN.md`` §8.2 P36 (a)).

David's sentence in the book's Ch 2 ``sec.bayes`` (2026-10-03): "a belief in general is a whole
curve ... So we do not get Bayes' rule for every belief in this chapter." Stage (a) asks for the
case of finitely many values: is one step of an arrangement Bayes' rule for EVERY belief and EVERY
likelihood, with no family chosen?

The derivation was fixed in writing before ``arrangements/belief.py`` existed
(``notes/BAYES-EVERY-BELIEF.md`` §1; the proof in the book's letters is the book repo's
``notes/bayes-every-belief.md``). State ``l in R^n``, the log-heights; readout ``q = softmax(l)``;
zero forcing; a downstream likelihood box with cost ``k(y, x) = <y, nll(x)>``, linear in the
belief, which pushes ``xi = log Pr(x | .)`` back whatever the belief; response
``sharp_l = diag(1 / q)``. Then ``l' = l + xi - <q, xi> 1`` and ``softmax(l')`` is the posterior.

THE NINE PREDICTIONS, verbatim from the note's §2, written before any construction code:

1. **Exactness, every belief and every likelihood.** For ``n = 2, 3, 7, 50``, random beliefs with
   every height positive and random positive likelihood columns, one step of
   ``Phi(likelihood . belief)`` gives ``softmax(l')`` equal to the posterior computed directly,
   ``q p / <q, p>``, to relative error below ``1e-12``. The hand example: belief ``(1/2, 1/2)``,
   likelihoods ``(3/4, 1/4)``, posterior ``(3/4, 1/4)``, and ``l' = (log(3)/2, -log(3)/2)``.
2. **The constant is the belief's average of the force.** On ``R^n`` the step is
   ``l + xi - <q, xi> 1`` to ``1e-12``, so ``<q, l' - l> = 0``; it is *not* the normalizing
   constant ``log <q, p>``, and ``sum exp(l')`` exceeds ``1`` after a step from normalized
   log-heights by the factor ``exp(log <q,p> - <q, log p>) >= 1``.
3. **Three ways to hold the constant, one belief.** The ``R^n`` box with the diagonal response,
   the gauge-fixed box with the closed-form inverse, and the gauge-fixed box with the dense
   ``inv(G)`` (``choice.inverse_fisher``) land on the same belief to ``1e-12`` from a belief with
   every height above ``1e-3``. For ``n = 2`` the gauge state is the log-odds and the step adds
   the log-likelihood ratio.
4. **A stream of observations; order does not matter.** With ``z`` on a grid, bell-curve noise of
   variance 1 and a two-bump prior (in no bell-curve family), 30 real observations stepped one at
   a time give the directly computed posterior to ``1e-10``, the same in reversed and shuffled
   order, and the same as one step with the summed cost.
5. **The response is what does the work.** With the Euclidean response ``eta I`` in the same
   arrangement: from the uniform belief ``eta = n`` is Bayes exactly (there ``diag(1/q) = n I``),
   and from a non-uniform belief no ``eta in {0.1, 0.5, 1, n, 2n}`` gives the posterior (belief
   error above ``1e-3``); and with ``eta = n`` a second step from the uniform belief misses.
6. **Thin beliefs.** In the mathematics the step is exact at every positive height. In float64,
   the diagonal response stays exact (step error against ``xi - <q, xi>`` below ``1e-9``) with a
   height as small as ``1e-300``, where the dense ``inv(G)`` has lost it by ``1e-8`` (the
   science-scales run found loss below about ``1e-5``,
   ``SCIENCE-SCALES-PREDICTIONS-2026-10-01.md`` §5.1). Over a 400-draw stream that drives heights
   below ``1e-100`` the diagonal box tracks the directly accumulated log-posterior differences to
   ``1e-8``. It fails only when a height underflows to exactly ``0`` in float64 (a log-height more
   than about 745 below the largest): then ``0 / 0``, a non-finite state.
7. **A zero likelihood is outside.** ``Pr(x | z_i) = 0`` gives a non-finite state after one step;
   ``1e-300`` in its place gives the direct posterior to ``1e-12``.
8. **Duration is tempering.** A step at ``tau`` gives ``q p^tau / <q, p^tau>``; two half steps
   equal one whole step.
9. **The forcing may sit in the belief box.** The belief box carrying the cost as its own forcing
   (item 27's shape, no likelihood box) moves the state identically, to ``1e-12``.

Each test prints what it measured (run with ``PYTEST_ADDOPTS=-s``); the verdicts, with the
numbers, are the note's §3.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from arrangements.belief import (
    bell_nll,
    belief_of,
    column_nll,
    gauge_belief_of,
    gauge_response,
    log_height_response,
    log_heights,
    posterior,
    run,
    self_forced_belief,
    updating,
)
from arrangements.choice import inverse_fisher
from arrangements.functors import Phi
from arrangements.interpretation import interpretation, trivial_omega
from arrangements.rvect import euclidean


def _rng(seed):
    return np.random.RandomState(seed)


def _random_belief(rng, n, spread=2.0):
    """Log-heights of a random belief with every height positive, normalized."""
    l = spread * rng.randn(n)
    return jnp.asarray(l - np.log(np.sum(np.exp(l))))


def _random_likelihood(rng, n, spread=2.0):
    """A random positive likelihood column, not normalized (any column whatever)."""
    return jnp.asarray(np.exp(spread * rng.randn(n)))


def _step(l, p, response=None):
    """One step of the composite from log-heights ``l`` on the likelihood column ``p``."""
    n = int(l.shape[0])
    response = log_height_response(n) if response is None else response
    arr = updating(response, column_nll, n, n, l)
    return log_heights(run(arr, [-jnp.log(p)]))


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b) / np.abs(b)))


def _abs(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


# ---------------------------------------------------------------------------
# Prediction 1: exactness, every belief and every likelihood.
# ---------------------------------------------------------------------------


def test_prediction_1_the_hand_example():
    """Belief ``(1/2, 1/2)``, likelihoods ``(3/4, 1/4)``: ``l' = (log 3 / 2, -log 3 / 2)``."""
    l1 = _step(jnp.zeros(2), jnp.array([0.75, 0.25]))
    want = np.array([np.log(3.0) / 2, -np.log(3.0) / 2])
    print(f"\nP1 hand: l' = {np.asarray(l1)}, belief = {np.asarray(belief_of(l1))}")
    assert _abs(l1, want) < 1e-12
    assert _rel(belief_of(l1), [0.75, 0.25]) < 1e-12


@pytest.mark.parametrize("n", [2, 3, 7, 50])
def test_prediction_1_every_belief_and_every_likelihood(n):
    """Random beliefs, random positive columns: one step is the direct posterior."""
    rng = _rng(100 + n)
    worst = 0.0
    for _ in range(6):
        l = _random_belief(rng, n)
        for _ in range(4):
            p = _random_likelihood(rng, n)
            err = _rel(belief_of(_step(l, p)), posterior(belief_of(l), p))
            worst = max(worst, err)
    print(f"\nP1 n={n}: worst relative error over 24 (belief, likelihood) pairs = {worst:.2e}")
    assert worst < 1e-12


def test_prediction_1_the_books_bell_curve_step_on_a_grid():
    """Not in §2: the book's worked step, ``N(0,1)`` and the reading ``x = 2``, on a grid.

    A bell curve restricted to a grid times a bell-curve likelihood is a bell curve restricted to
    the grid, so the belief after is the restriction of ``N(1, 1/2)`` exactly, with no family told
    to the box."""
    grid = jnp.linspace(-8.0, 8.0, 161)
    l0 = -0.5 * grid**2
    arr = updating(log_height_response(161), bell_nll(grid), 161, 1, l0)
    q1 = belief_of(log_heights(run(arr, [jnp.array([2.0])])))
    want = belief_of(-((grid - 1.0) ** 2))
    mean = float(jnp.dot(q1, grid))
    var = float(jnp.dot(q1, (grid - mean) ** 2))
    print(f"\nP1 grid: max abs belief error = {_abs(q1, want):.2e}, mean = {mean:.12f}, var = {var:.12f}")
    assert _abs(q1, want) < 1e-12
    assert abs(mean - 1.0) < 1e-9 and abs(var - 0.5) < 1e-9


def test_the_pushback_on_the_observation_is_the_beliefs_mean_minus_the_reading():
    """Plumbing (the book's ``y_1 - x``): the covector sent back to the observation's slot."""
    grid = jnp.linspace(-3.0, 3.0, 7)
    l0 = _random_belief(_rng(7), 7)
    arr = updating(log_height_response(7), bell_nll(grid), 7, 1, l0)
    q = jnp.concatenate([l0, jnp.zeros(0)])
    _, direction_action = interpretation(arr)(q)
    x = jnp.array([0.8])
    _, xi_M, _ = direction_action(x, trivial_omega(0), jnp.zeros(0), jnp.zeros(0))
    want = float(jnp.dot(belief_of(l0), grid)) - 0.8
    print(f"\npushback on the observation = {float(xi_M[0]):.12f}, E_q[z] - x = {want:.12f}")
    assert abs(float(xi_M[0]) - want) < 1e-12


# ---------------------------------------------------------------------------
# Prediction 2: the constant is the belief's average of the force.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 5, 20])
def test_prediction_2_the_constant_is_the_beliefs_average_of_the_force(n):
    rng = _rng(200 + n)
    l = _random_belief(rng, n)  # normalized: sum exp(l) = 1
    p = _random_likelihood(rng, n)
    q, xi = belief_of(l), jnp.log(p)
    l1 = _step(l, p)
    c = float(jnp.dot(q, xi))
    step_err = _abs(l1 - l, xi - c)
    mean_move = float(jnp.dot(q, l1 - l))
    total = float(jnp.sum(jnp.exp(l1)))
    jensen = float(jnp.exp(jnp.log(jnp.dot(q, p)) - c))
    print(
        f"\nP2 n={n}: |step - (xi - <q,xi>)| = {step_err:.2e}, <q, l'-l> = {mean_move:.2e}, "
        f"sum exp(l') = {total:.6f}, exp(log<q,p> - <q,log p>) = {jensen:.6f}"
    )
    assert step_err < 1e-12
    assert abs(mean_move) < 1e-12
    assert total > 1.0 + 1e-6 and abs(total - jensen) / jensen < 1e-12


# ---------------------------------------------------------------------------
# Prediction 3: three ways to hold the constant, one belief.
# ---------------------------------------------------------------------------


def _gauge_step(l, p, response):
    """One step of the gauge-fixed composite (state: log-odds against the last value)."""
    n = int(l.shape[0])
    lam = l[:-1] - l[-1]
    arr = updating(response, column_nll, n, n, lam, readout=gauge_belief_of)
    return log_heights(run(arr, [-jnp.log(p)]))


@pytest.mark.parametrize("n", [2, 3, 7])
def test_prediction_3_three_ways_to_hold_the_constant(n):
    rng = _rng(300 + n)
    worst = 0.0
    for _ in range(5):
        while True:
            l = _random_belief(rng, n, spread=1.0)
            if float(jnp.min(belief_of(l))) > 1e-3:
                break
        p = _random_likelihood(rng, n)
        direct = posterior(belief_of(l), p)
        full = belief_of(_step(l, p))
        closed = gauge_belief_of(_gauge_step(l, p, gauge_response(n)))
        dense = gauge_belief_of(_gauge_step(l, p, inverse_fisher(n)))
        worst = max(worst, _rel(full, direct), _rel(closed, direct), _rel(dense, direct))
    print(f"\nP3 n={n}: worst relative error of the three against the direct posterior = {worst:.2e}")
    assert worst < 1e-12


def test_prediction_3_two_values_the_log_odds_add_the_log_likelihood_ratio():
    """``n = 2`` in the gauge: one number, and the step is odds times likelihood ratio."""
    for lam0, p in ((0.0, (0.75, 0.25)), (-1.3, (0.2, 0.9)), (2.1, (0.05, 0.6))):
        l = jnp.array([lam0, 0.0])
        lam1 = _gauge_step(l, jnp.array(p), gauge_response(2))
        want = lam0 + np.log(p[0] / p[1])
        print(f"\nP3 log-odds: {lam0} -> {float(lam1[0]):.12f}, prior + log LR = {want:.12f}")
        assert abs(float(lam1[0]) - want) < 1e-12


# ---------------------------------------------------------------------------
# Prediction 4: a stream of observations; order does not matter.
# ---------------------------------------------------------------------------

GRID = jnp.linspace(-4.0, 6.0, 41)


def _two_bump_prior():
    w = 0.5 * jnp.exp(-0.5 * ((GRID + 1.5) / 0.4) ** 2) / 0.4 + 0.5 * jnp.exp(
        -0.5 * ((GRID - 2.5) / 0.6) ** 2
    ) / 0.6
    return jnp.log(w / jnp.sum(w))


def test_prediction_4_a_stream_on_a_two_bump_prior_and_order_does_not_matter():
    rng = _rng(4)
    data = 2.0 + rng.randn(30)
    l0 = _two_bump_prior()
    n = int(GRID.shape[0])

    log_post = l0 - jnp.sum(0.5 * (jnp.asarray(data)[:, None] - GRID[None, :]) ** 2, axis=0)
    direct = belief_of(log_post)

    def stream(xs):
        arr = updating(log_height_response(n), bell_nll(GRID), n, 1, l0)
        return belief_of(log_heights(run(arr, [jnp.array([x]) for x in xs])))

    seq = stream(data)
    rev = stream(data[::-1])
    shuf = stream(rng.permutation(data))
    batch_nll = lambda xs: jnp.sum(0.5 * (xs[:, None] - GRID[None, :]) ** 2, axis=0)
    batch = belief_of(log_heights(run(
        updating(log_height_response(n), batch_nll, n, 30, l0), [jnp.asarray(data)]
    )))
    errs = {k: _abs(v, direct) for k, v in (("seq", seq), ("rev", rev), ("shuf", shuf), ("batch", batch))}
    print(
        f"\nP4: max abs belief error against the direct posterior: {errs}; "
        f"smallest height after = {float(jnp.min(direct)):.2e}; "
        f"prior mass on the left bump = {float(jnp.sum(belief_of(l0)[GRID < 0.5])):.4f}, "
        f"after = {float(jnp.sum(direct[GRID < 0.5])):.2e}"
    )
    assert all(e < 1e-10 for e in errs.values())


# ---------------------------------------------------------------------------
# Prediction 5: the response is what does the work.
# ---------------------------------------------------------------------------


def test_prediction_5_a_learning_rate_of_n_is_bayes_at_the_uniform_belief_only():
    n = 3
    p = jnp.array([0.2, 0.5, 0.9])
    uniform = jnp.zeros(n)
    hit = _abs(belief_of(_step(uniform, p, euclidean(n, float(n)))), posterior(belief_of(uniform), p))

    l = jnp.log(jnp.array([0.6, 0.3, 0.1]))
    direct = posterior(belief_of(l), p)
    misses = {
        eta: _abs(belief_of(_step(l, p, euclidean(n, eta))), direct)
        for eta in (0.1, 0.5, 1.0, float(n), 2.0 * n)
    }

    # the second step from the uniform belief, same learning rate n
    arr = updating(euclidean(n, float(n)), column_nll, n, n, uniform)
    two = belief_of(log_heights(run(arr, [-jnp.log(p), -jnp.log(p)])))
    direct_two = posterior(posterior(belief_of(uniform), p), p)
    second = _abs(two, direct_two)
    print(
        f"\nP5: eta = n from uniform, error {hit:.2e}; from (0.6, 0.3, 0.1), errors {misses}; "
        f"second step from uniform at eta = n, error {second:.2e}"
    )
    assert hit < 1e-12
    assert all(e > 1e-3 for e in misses.values())
    assert second > 1e-3


# ---------------------------------------------------------------------------
# Prediction 6: thin beliefs.
# ---------------------------------------------------------------------------


def _thin_belief(h):
    """Three values, the first with height ``h``."""
    return jnp.log(jnp.array([h, 0.3, 0.7 - h]))


THIN_P = jnp.array([0.9, 0.2, 0.5])


@pytest.mark.parametrize("h", [1e-5, 1e-8, 1e-12, 1e-100, 1e-300])
def test_prediction_6_the_diagonal_response_is_exact_on_a_thin_belief(h):
    l = _thin_belief(h)
    q, xi = belief_of(l), jnp.log(THIN_P)
    err = _abs(_step(l, THIN_P) - l, xi - jnp.dot(q, xi))
    print(f"\nP6 diagonal, height {h:g}: step error = {err:.2e}")
    assert err < 1e-9


@pytest.mark.parametrize(
    "h",
    [1e-5]
    + [
        pytest.param(h, marks=pytest.mark.xfail(strict=True, reason="prediction 6's dense-inverse clause, REFUTED"))
        for h in (1e-8, 1e-12, 1e-100, 1e-300)
    ],
)
def test_prediction_6_the_dense_inverse_has_lost_it_by_1e_minus_8(h):
    """The gauge-fixed dense ``inv(G)``; the thin value is not the gauge's reference value.

    REFUTED (strict xfail, kept as the record): with the reference value thick, the dense inverse
    is as exact as the diagonal response down to a height of ``1e-300``. What loses the step is a
    thin REFERENCE value, ``test_the_gauge_needs_a_thick_reference_value``."""
    l = _thin_belief(h)
    xi = jnp.log(THIN_P)
    want = xi[:-1] - xi[-1]
    lam = l[:-1] - l[-1]
    dense = _abs(_gauge_step(l, THIN_P, inverse_fisher(3)) - lam, want)
    closed = _abs(_gauge_step(l, THIN_P, gauge_response(3)) - lam, want)
    print(f"\nP6 gauge, height {h:g}: dense inv(G) step error = {dense:.2e}, closed form = {closed:.2e}")
    if h <= 1e-8:
        assert not dense < 1e-9  # lost (or non-finite)


@pytest.mark.parametrize("h", [1e-5, 1e-8, 1e-12, 1e-100, 1e-300])
def test_the_gauge_needs_a_thick_reference_value(h):
    """Not in §2: the closed-form gauge response when the THIN value is the reference."""
    l = jnp.log(jnp.array([0.3, 0.7 - h, h]))
    xi = jnp.log(jnp.array([0.2, 0.5, 0.9]))
    want = xi[:-1] - xi[-1]
    lam = l[:-1] - l[-1]
    closed = _abs(_gauge_step(l, jnp.exp(xi), gauge_response(3)) - lam, want)
    dense = _abs(_gauge_step(l, jnp.exp(xi), inverse_fisher(3)) - lam, want)
    full = _abs(_step(l, jnp.exp(xi)) - l, xi - jnp.dot(belief_of(l), xi))
    print(
        f"\nthin reference value, height {h:g}: gauge step error, closed form = {closed:.2e}, "
        f"dense inv(G) = {dense:.2e}; the R^n diagonal box on the same belief = {full:.2e}"
    )
    assert full < 1e-9


STREAM_GRID = jnp.array([-0.25, 0.5, 1.0])


def _stream(draws, seed=6):
    return 1.0 + _rng(seed).randn(draws)


def test_prediction_6_a_400_draw_stream_below_1e_minus_100():
    data = _stream(400)
    l0 = jnp.zeros(3)
    arr = updating(log_height_response(3), bell_nll(STREAM_GRID), 3, 1, l0)
    l = log_heights(run(arr, [jnp.array([x]) for x in data]))
    direct = -jnp.sum(0.5 * (jnp.asarray(data)[:, None] - STREAM_GRID[None, :]) ** 2, axis=0)
    err = _abs(l - l[-1], direct - direct[-1])
    smallest = float(jnp.min(belief_of(direct)))
    print(
        f"\nP6 stream, 400 draws: smallest height = {smallest:.2e}, "
        f"max error in log-height differences = {err:.2e}, largest |log-height| = {float(jnp.max(jnp.abs(l))):.2f}, "
        f"sum exp(l) = {float(jnp.sum(jnp.exp(l))):.6f}"
    )
    assert smallest < 1e-100
    assert err < 1e-8


def test_prediction_10_the_science_scales_stream():
    """Prediction 10 (the note's §2, added before this piece was run): on
    ``tests/test_science_scales.py``'s misspecified stream the loss is the gauge's reference
    value going thin, not the response and not the dense inverse."""
    mus = jnp.arange(5.0)
    data = np.random.default_rng(0).normal(2.3, 1.0, 200)
    cum = np.cumsum(-0.5 * (data[:, None] - np.asarray(mus)[None, :]) ** 2, axis=0)  # direct log-posterior

    def track(order, response, gauge):
        """Max error in log-height differences against the direct ones, by draw."""
        grid = mus[jnp.asarray(order)]
        ref = cum[:, order]
        if gauge:
            arr = updating(response, bell_nll(grid), 5, 1, jnp.zeros(4), readout=gauge_belief_of)
        else:
            arr = updating(response, bell_nll(grid), 5, 1, jnp.zeros(5))
        O = Phi(arr)
        s, errs = O.state, []
        for t, x in enumerate(data):
            _, _, s = O.with_state(s).run_one(
                (jnp.array([x]), trivial_omega(0)), lambda _o: (jnp.zeros(0), jnp.zeros(0))
            )
            l = np.asarray(log_heights(s))
            diffs = l if gauge else l[:-1] - l[-1]
            errs.append(float(np.max(np.abs(diffs - (ref[t, :-1] - ref[t, -1])))))
        return np.asarray(errs)

    as_run = [0, 1, 2, 3, 4]       # reference mu = 4, as science.mind_step has it
    winner_last = [0, 1, 3, 4, 2]  # reference mu = 2, the winner
    full = track(as_run, log_height_response(5), gauge=False)
    ref4_closed = track(as_run, gauge_response(5), gauge=True)
    ref4_dense = track(as_run, inverse_fisher(5), gauge=True)
    ref2_closed = track(winner_last, gauge_response(5), gauge=True)
    ref2_dense = track(winner_last, inverse_fisher(5), gauge=True)

    def first_above(e, tol):
        bad = np.nonzero(~(e < tol))[0]
        return int(bad[0]) + 1 if len(bad) else None

    q200 = np.exp(cum[-1] - np.max(cum[-1]))
    print(
        f"\nP10: smallest height at draw 200 = {float(np.min(q200 / np.sum(q200))):.2e}; "
        f"R^n diagonal max error over 200 draws = {np.max(full):.2e}; "
        f"gauge, reference the winner: closed {np.max(ref2_closed):.2e}, dense {np.max(ref2_dense):.2e}; "
        f"gauge, reference mu = 4: error at draw 30 closed {ref4_closed[29]:.2e}, dense {ref4_dense[29]:.2e}, "
        f"first draw above 1e-12 closed {first_above(ref4_closed, 1e-12)}, dense {first_above(ref4_dense, 1e-12)}, "
        f"first non-finite closed {first_above(np.where(np.isfinite(ref4_closed), 0.0, 1.0), 0.5)}, "
        f"dense {first_above(np.where(np.isfinite(ref4_dense), 0.0, 1.0), 0.5)}"
    )
    assert np.max(full) < 1e-9
    assert np.max(ref2_closed) < 1e-9 and np.max(ref2_dense) < 1e-9
    assert not ref4_closed[29] < 1e-6 and not ref4_dense[29] < 1e-6


def test_prediction_6_it_fails_when_a_height_underflows_to_zero():
    data = _stream(1300)
    l0 = jnp.zeros(3)
    arr = updating(log_height_response(3), bell_nll(STREAM_GRID), 3, 1, l0)
    first_bad, height_before = None, None
    O = Phi(arr)
    s = O.state
    for t, x in enumerate(data):
        q_before = belief_of(log_heights(s))
        _, _, s = O.with_state(s).run_one(
            (jnp.array([x]), trivial_omega(0)), lambda _o: (jnp.zeros(0), jnp.zeros(0))
        )
        if not bool(jnp.all(jnp.isfinite(log_heights(s)))):
            first_bad, height_before = t, float(jnp.min(q_before))
            break
    print(f"\nP6 underflow: first non-finite state at draw {first_bad}, smallest height just before = {height_before}")
    assert first_bad is not None
    assert height_before == 0.0


def test_below_the_smallest_normal_float_a_height_is_zero():
    """Not in §2: jax flushes subnormal numbers to zero, so a height below about ``2.2e-308``
    (a log-height about 708 below the largest, not 745) is exactly ``0`` and the step is
    non-finite there; at ``1e-307`` it is still exact."""
    xi = jnp.log(THIN_P)
    out = {}
    for h in (1e-300, 1e-307, 1e-310, 1e-320):
        l = jnp.array([float(np.log(1e-300) + np.log(h / 1e-300)), np.log(0.3), np.log(0.7)])
        q = belief_of(l)
        out[h] = (float(q[0]), _abs(_step(l, THIN_P) - l, xi - jnp.dot(q, xi)))
    print(f"\nbelow the smallest normal float: (height as held, step error) by height = {out}")
    assert out[1e-300][1] < 1e-9 and out[1e-307][1] < 1e-9
    assert out[1e-310][0] == 0.0 and not np.isfinite(out[1e-310][1])


# ---------------------------------------------------------------------------
# Prediction 7: a zero likelihood is outside.
# ---------------------------------------------------------------------------


def test_prediction_7_a_zero_likelihood_is_outside():
    l = jnp.log(jnp.array([0.6, 0.3, 0.1]))
    with np.errstate(divide="ignore"):
        l_zero = _step(l, jnp.array([0.0, 0.5, 0.9]))
    tiny = jnp.array([1e-300, 0.5, 0.9])
    err = _rel(belief_of(_step(l, tiny)), posterior(belief_of(l), tiny))
    print(f"\nP7: state after a zero likelihood = {np.asarray(l_zero)}; with 1e-300, relative error {err:.2e}")
    assert not bool(jnp.all(jnp.isfinite(l_zero)))
    assert err < 1e-12


# ---------------------------------------------------------------------------
# Prediction 8: duration is tempering.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tau", [0.0, 0.25, 0.5, 1.0, 2.0])
def test_prediction_8_duration_tempers_the_likelihood(tau):
    l = jnp.log(jnp.array([0.6, 0.3, 0.1]))
    p = jnp.array([0.2, 0.5, 0.9])
    got = belief_of(_step(l, p, log_height_response(3, tau)))
    err = _rel(got, posterior(belief_of(l), p**tau))
    print(f"\nP8 tau={tau}: relative error against q p^tau / <q, p^tau> = {err:.2e}")
    assert err < 1e-12


def test_prediction_8_two_half_steps_equal_one_whole_step():
    l = jnp.log(jnp.array([0.6, 0.3, 0.1]))
    p = jnp.array([0.2, 0.5, 0.9])
    half = updating(log_height_response(3, 0.5), column_nll, 3, 3, l)
    two = belief_of(log_heights(run(half, [-jnp.log(p), -jnp.log(p)])))
    one = belief_of(_step(l, p))
    print(f"\nP8: two half steps against one whole step, relative error {_rel(two, one):.2e}")
    assert _rel(two, one) < 1e-12


# ---------------------------------------------------------------------------
# Prediction 9: the forcing may sit in the belief box.
# ---------------------------------------------------------------------------


def test_prediction_9_the_forcing_may_sit_in_the_belief_box():
    rng = _rng(9)
    worst = 0.0
    for n in (2, 5, 12):
        l = _random_belief(rng, n)
        p = _random_likelihood(rng, n)
        own = run(self_forced_belief(log_height_response(n), column_nll, n, l), [-jnp.log(p)])
        worst = max(worst, _abs(own, _step(l, p)))
    print(f"\nP9: self-forced box against the composite, max abs state difference {worst:.2e}")
    assert worst < 1e-12
