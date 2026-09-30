"""Many parts, three wirings: does a part that needs *every* other survive a cut?

**The question.** The book's Ch 1 §1.7 footnote on ``tests/test_many_parts.py`` says a
collective "in which each needs all the others, given enough supply, loses only the part that
was cut but otherwise survives." The domain check of 2026-09-29 (book repo,
``notes/DOMAIN-CHECK-CH1-2026-09-29.md``, Group B, run 2) found that the surviving wiring feeds
each part the *mean* of the others, which is pooled and substitutable, and that the "same supply
condition at every size" comes from the averaging. This file runs the same parts under three
wirings and asserts what each does.

**The model.** Exactly ``test_many_parts.py``'s construction (``collective``: ``n`` parts from
``test_mutual_need.part``, one supply box, a stateless coupler returning
``F_i = alpha s m_i g(r_i, q_i)``, ``g = u/(K+u)``, ``u = r_i q_i``, ``beta = 0``,
``alpha = b = K = 1``, ``h = 0.01``, 20000 steps; "held" iff ``|q| > 1e-3``; a cut sets
``m_i = 0``). Only ``r_i``, what part ``i`` receives from the others, changes:

* ``mean``: ``r_i = mean_{j != i} q_j`` (the existing test's all-to-all);
* ``sum``:  ``r_i = sum_{j != i} q_j``;
* ``min``:  ``r_i = min_{j != i} q_j``, each part needing every other (a Liebig/Leontief need);
* ``softmin``: ``r_i = -(1/k) log sum_{j != i} exp(-k q_j)``, a smooth ``min``, ``k`` given.

The coupler is a returned covector, not an exact cost (``test_many_parts`` test 4 shows it is
non-exact), so the hard ``min`` is used as is; ``softmin`` is run only to show the verdict does
not rest on the kink.

**The prediction** (written first, book repo ``notes/ch1-tests-2026-09-29/many-parts-wirings.md``).
Mean: threshold ``s_* = 2`` at every ``n``, a cut kills only the cut part when
``s > 2 sqrt((n-1)/(n-2))``. Sum: at the symmetric state ``r = c q``, ``c = n-1``, the nonzero
fixed points solve ``b c q^2 - alpha s c q + b K = 0``, real iff ``s >= s_* = 2 (b/alpha)
sqrt(K/(n-1))``: ``2, 1.414, 1, 0.756`` at ``n = 2, 3, 5, 8``. Min: intact threshold ``2`` (the
symmetric state has ``r = q``); **one cut kills every part, at every ``n`` and supply**. The
mechanism: the cut part has no drive and decays as ``e^{-bt}``; each other part's input is at
most the cut part's amplitude, so its drive is at most ``alpha s q_cut(t) q_i / K``, whose time
integral is finite, and ``q_i <= q_i(0) exp(alpha s q_cut(0) / (K b) - b t) -> 0``. The cut
part's collapse is every other part's input.

**The result.** All three predictions hold (tests 1-4). Sum: ``s_* = 2, 1.414, 1, 0.756`` at
``n = 2, 3, 5, 8``, held at ``1.05 s_*``, dead at ``0.95 s_*``. Min: one cut kills every part at
every ``n`` and at ``s`` up to 20; not a front, since at ``n = 5, s = 4`` the cut part falls below
``1e-3`` at ``t = 8.2`` and all four others together at ``t = 11.3``. Soft-min at ``k = 5, 20,
80`` gives the same verdicts. So "needs every other" is the wiring that dies from one cut, like
the ring; the wiring that survives is the one where a part draws on what the others send
between them; and "the same supply at every size" is a property of averaging (summed help makes
a bigger collective easier to hold). Numbers: book repo
``notes/ch1-tests-2026-09-29/many-parts-wirings.md``.

**What it does not show.** The supply is still per part and never depletes, so none of this
says what a shared finite supply does to size. The ``min`` wiring is the extreme of
non-substitutable need; real needs sit between it and the mean. A cut is the supply wire; a part
that fails some other way is not run. The parts have one scalar each; "needs every other" is a
statement about this gate, not about organs.
"""

import jax
import jax.numpy as jnp
import numpy as np

from tests import test_mutual_need as mn
from tests import test_many_parts as mp
from arrangements.wiring import compose_seq, tensor_arrangements

ALPHA, DISS, KGATE = mn.ALPHA, mn.DISS, mn.KGATE
STEPS = mn.STEPS
NS = (2, 3, 5, 8)


def receive(q, wiring, k=None):
    """``r_i`` from the published amplitudes ``q`` (the module docstring's four wirings)."""
    n = q.shape[0]
    if wiring == "mean":
        return (jnp.sum(q) - q) / (n - 1)
    if wiring == "sum":
        return jnp.sum(q) - q
    others = jnp.where(jnp.eye(n, dtype=bool), jnp.inf, q[None, :])   # row i: q_j, j != i
    if wiring == "min":
        return jnp.min(others, axis=1)
    if wiring == "softmin":
        return -jax.scipy.special.logsumexp(-k * others, axis=1) / k
    raise ValueError(wiring)


def F_wired(n, wiring, s_mask=None, k=None, alpha=ALPHA, K=KGATE):
    """``test_many_parts.F_many`` at ``beta = 0`` with ``r_i`` from ``receive``."""
    s_mask = jnp.ones(n) if s_mask is None else jnp.asarray(s_mask, float)

    def F(e):
        q, s = e[:n], e[n]
        u = receive(q, wiring, k) * q
        drive = alpha * s * s_mask * u / (K + u)
        return jnp.concatenate([drive, jnp.zeros(1)])

    return F


def collective(n, wiring, q0, s, s_mask=None, k=None):
    q0 = [float(q0)] * n if isinstance(q0, (int, float)) else list(q0)
    parts = [mn.part(q0[i], label=f"P{i}") for i in range(n)] + [mn.supply(s)]
    return compose_seq(tensor_arrangements(parts),
                       mn.coupler(F_wired(n, wiring, s_mask, k), n + 1))


def sum_roots(n, s, cut=False, alpha=ALPHA, b=DISS, K=KGATE):
    """Sum wiring's symmetric nonzero fixed points ``(q-, q+, s_*)``, with ``c = n-1``
    intact or ``c = n-2`` for the survivors of one cut."""
    c = (n - 2.0) if cut else (n - 1.0)
    s_star = 2.0 * (b / alpha) * np.sqrt(K / c)
    disc = (alpha * s * c) ** 2 - 4.0 * b * b * K * c
    if disc < 0.0:
        return float("nan"), float("nan"), s_star
    return ((alpha * s * c - np.sqrt(disc)) / (2 * b * c),
            (alpha * s * c + np.sqrt(disc)) / (2 * b * c), s_star)


def settle(n, wiring, q0, s, cut=None, k=None, steps=STEPS):
    mask = None
    if cut is not None:
        mask = np.ones(n)
        mask[cut] = 0.0
    return mn.settle(collective(n, wiring, q0, s, mask, k), steps)[:n]


# ---------------------------------------------------------------------------


def test_0_mean_here_is_test_many_parts_all_to_all():
    """LICENSES the reuse: the ``mean`` coupler is ``test_many_parts.F_many(n, "all")``,
    value for value, at random points and with a cut."""
    rng = np.random.default_rng(0)
    for n in NS:
        mask = np.ones(n)
        mask[0] = 0.0
        for _ in range(5):
            e = jnp.asarray(np.r_[rng.uniform(0, 4, n), 3.0])
            np.testing.assert_allclose(F_wired(n, "mean", mask)(e),
                                       mp.F_many(n, "all", s_mask=mask)(e), atol=1e-12)
            if n == 2:   # one other part: the three wirings coincide
                for w in ("sum", "min"):
                    np.testing.assert_allclose(F_wired(n, w, mask)(e),
                                               F_wired(n, "mean", mask)(e), atol=1e-12)


def test_1_intact_thresholds_mean_and_min_two_sum_two_over_root_n_minus_1():
    """REPRODUCES the closed forms. Mean and min: nothing held at ``s = 1.5``, every part
    at ``q+ = 3.732051`` at ``s = 4`` (from ``q0 = 1``). Sum: every part held at
    ``1.05 s_*`` and at ``q+`` (checked against ``sum_roots``, with every Jacobian eigenvalue
    negative there), nothing held at ``0.95 s_*``, where ``s_* = 2/sqrt(n-1)``."""
    _, q_hi, _ = mp.symmetric_roots(4.0)
    for n in NS:
        for w in ("mean", "min"):
            assert float(np.max(np.abs(settle(n, w, 1.0, 1.5)))) < mn.HELD, (n, w)
            np.testing.assert_allclose(settle(n, w, 1.0, 4.0), q_hi, atol=1e-6)
        _, _, s_star = sum_roots(n, 1.0)
        np.testing.assert_allclose(s_star, 2.0 / np.sqrt(n - 1), rtol=1e-12)
        q_lo, q_up, _ = sum_roots(n, 1.05 * s_star)
        q = settle(n, "sum", 2.0 * q_up, 1.05 * s_star)
        np.testing.assert_allclose(q, q_up, atol=1e-5)
        arr = collective(n, "sum", q_up, 1.05 * s_star)
        J = np.asarray(jax.jacobian(mp.field(arr, n))(jnp.full(n, q_up)))
        assert np.max(np.linalg.eigvals(J).real) < 0.0
        for q0 in (1.0, 5.0):
            q = settle(n, "sum", q0, 0.95 * s_star, steps=2 * STEPS)
            assert float(np.max(np.abs(q))) < mn.HELD, (n, q0, q)
        print("sum", n, "s_*", s_star, "q+ at 1.05 s_*", q_up)


def test_2_one_cut_min_kills_every_part_at_every_supply():
    """CORRECTS the footnote's reading of ``test_many_parts``: under ``min``, cutting one
    part's supply from the settled state kills every part, at ``n = 2, 3, 5, 8`` and
    ``s = 2.5, 4, 8, 20``; the module docstring's bound
    ``q_i <= q_i(0) exp(alpha s q_cut(0)/(K b) - b t)`` holds along the whole run."""
    for s in (2.5, 4.0, 8.0, 20.0):
        _, q_hi, _ = mp.symmetric_roots(s)
        for n in NS:
            q = settle(n, "min", q_hi, s, cut=0)
            assert float(np.max(np.abs(q))) < 1e-30, (s, n, q)
    # the bound, along a trajectory
    s, n = 8.0, 5
    _, q_hi, _ = mp.symmetric_roots(s)
    mask = np.ones(n)
    mask[0] = 0.0
    tr = mp.trajectory(collective(n, "min", q_hi, s, mask), 3000)[:, :n]
    t = mn.MOB * np.arange(1, 3001)
    bound = q_hi * np.exp(ALPHA * s * q_hi / (KGATE * DISS) - DISS * t)
    assert np.all(tr[:, 1:] <= bound[:, None] * 1.01)
    print("min, n=5, s=8: t at which the last part falls below HELD",
          mn.MOB * (np.argmax(np.all(np.abs(tr) < mn.HELD, axis=1)) + 1))


def test_3_one_cut_mean_and_sum_lose_only_the_cut_part_above_their_thresholds():
    """REPRODUCES the survivors' closed forms. Mean at ``s = 4``: the cut part dies, the
    others hold at ``test_many_parts.survivors_roots`` (``2 + sqrt 2`` at ``n = 3``). Sum:
    survivors of one cut hold iff ``s > 2/sqrt(n-2)``; run at ``s = 1.2 * 2/sqrt(n-2)``
    (hold, at ``sum_roots(cut=True)``) and at the geometric midpoint of the intact and the
    survivors' thresholds, ``2/sqrt(n-1) < s < 2/sqrt(n-2)``, where the intact collective
    holds and one cut kills all, ``n = 3, 5, 8``."""
    _, q_hi, _ = mp.symmetric_roots(4.0)
    for n in (3, 5, 8):
        q = settle(n, "mean", q_hi, 4.0, cut=0)
        assert not mn.held(q[0])
        np.testing.assert_allclose(q[1:], mp.survivors_roots(n, 4.0)[0], atol=1e-6)
        _, _, s_cut = sum_roots(n, 1.0, cut=True)
        s_int = sum_roots(n, 1.0)[2]
        for s, holds in ((1.2 * s_cut, True), (np.sqrt(s_int * s_cut), False)):
            q_start = sum_roots(n, s)[1]
            q = settle(n, "sum", q_start, s, cut=0, steps=2 * STEPS)
            assert not mn.held(q[0])
            if holds:
                np.testing.assert_allclose(q[1:], sum_roots(n, s, cut=True)[1], atol=1e-5)
            else:
                assert float(np.max(np.abs(q))) < mn.HELD, (n, s, q)


def test_4_softmin_gives_the_min_verdict_as_k_grows():
    """CHECKS the verdict does not rest on the kink: under ``softmin`` with ``k = 5, 20,
    80``, the intact collective at ``s = 4`` holds every part (its level tending to
    ``3.732051`` as ``k`` grows) and one cut kills every part, ``n = 3, 5, 8``."""
    _, q_hi, _ = mp.symmetric_roots(4.0)
    for n in (3, 5, 8):
        levels = []
        for k in (5.0, 20.0, 80.0):
            q = settle(n, "softmin", q_hi, 4.0, k=k)
            assert all(mn.held(x) for x in q), (n, k, q)
            levels.append(float(np.mean(q)))
            q = settle(n, "softmin", q_hi, 4.0, cut=0, k=k)
            assert float(np.max(np.abs(q))) < mn.HELD, (n, k, q)
        assert abs(levels[-1] - q_hi) < abs(levels[0] - q_hi)
        assert abs(levels[-1] - q_hi) < 1e-2
        print("softmin", n, "intact levels at k=5,20,80", levels)
