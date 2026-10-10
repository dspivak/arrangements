"""Template copying: a row of two-well parts, each tipped by its partner in a source row (book P27).

The arrangement is ``copying.template``: source parts of depth ``alpha``, copy parts of depth 1, and one contact per pair whose
cost ``(g K / 2)(a_i - b_i)^2`` conducts in proportion to the write-enable ``K``, the environment's schedule through the
external port. Predictions were written first, in the book repo's ``notes/runs/p27-copying-dfb252e2/TEMPLATE-PREDICTIONS.md``;
verdicts in ``TEMPLATE-RESULTS.md`` beside it. Each test says whether it could have come out otherwise.
Settings: ``eta = 0.05``, ``g = 1``, ``alpha = 4`` unless said.
"""

import itertools

import jax
import numpy as np

from arrangements.copying import driven_step, run_scan, template, template_init, template_states

jax.config.update("jax_enable_x64", True)


_BUILT = {}


def _built(n, alpha, one_way=False):
    if (n, alpha, one_way) not in _BUILT:
        W = template([1.0] * n, [-1.0] * n, alpha=alpha, one_way=one_way)
        _BUILT[(n, alpha, one_way)] = (W, driven_step(W, check_inputs=[[1.0]] * 2))
    return _BUILT[(n, alpha, one_way)]


def _write(a0, b0, K, on, off, alpha=4.0, one_way=False):
    W, step = _built(len(a0), alpha, one_way)
    s = run_scan(step, template_init(W, a0, b0), [[K]] * on)
    during = template_states(W, s, len(a0))
    if off:
        s = run_scan(step, s, [[0.0]] * off)
    return during, template_states(W, s, len(a0))


def test_threshold():
    """Prediction 1. Could fail: the threshold could sit far from 1/4, where the copy's wrong well vanishes."""
    lo, hi = 0.2, 0.4
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        _, (a, b) = _write([1.0], [-1.0], mid, 2000, 2000)
        lo, hi = (lo, mid) if b[0] > 0 else (mid, hi)
    print("write-enable threshold", lo, hi)
    (a, b), _ = _write([1.0], [-1.0], 0.2, 2000, 0)
    print("K = 0.2 while enabled: a", a[0], "b", b[0])
    assert 0.25 < lo < 0.3
    assert abs(b[0] + 0.72) < 0.03


def test_every_pattern_copied_and_held():
    """Predictions 2 and 3, all 256 patterns of 8 bits, the copy row from a random start. Could fail: a bit could fail to
    cross in 200 steps, or a source bit could flip."""
    rng = np.random.default_rng(0)
    worst_copy, worst_source, worst_pull = 0.0, 0.0, 0.0
    for bits in itertools.product((-1.0, 1.0), repeat=8):
        a0 = np.array(bits)
        b0 = rng.choice((-1.0, 1.0), 8)
        (a_d, _), (a, b) = _write(a0, b0, 1.0, 200, 2000)
        worst_copy = max(worst_copy, np.abs(b - a0).max())
        worst_source = max(worst_source, np.abs(a - a0).max())
        worst_pull = max(worst_pull, np.abs(a_d - a0).max())
    print("worst copy error", worst_copy, "worst source change after", worst_source, "largest source pull while enabled",
          worst_pull)
    assert worst_copy < 1e-9 and worst_source < 1e-9
    assert worst_pull < 0.2


def test_equal_depth_copies_nothing():
    """Prediction 4. Could fail: the pair could break its symmetry and one side win. Found beside it: at K = 1 the pair falls
    toward the middle, each keeping its sign only by a remainder that shrinks like exp(-eta t), and climbs back only in about
    as many steps as the write lasted (2000 on, 2000 off leaves a at 0.005); 4000 off returns it."""
    for K, off in ((1.0, 2000), (1.0, 4000), (0.3, 2000)):
        (a_d, b_d), (a, b) = _write([1.0], [-1.0], K, 2000, off, alpha=1.0)
        print("alpha 1, K", K, "while enabled", a_d[0], b_d[0], "after", off, "off:", a[0], b[0])
        assert a_d[0] == -b_d[0] and a[0] == -b[0] and a[0] > 0
        if off == 4000 or K == 0.3:
            assert abs(a[0] - 1.0) < 1e-9


def test_write_time():
    """Prediction 5. Could fail: the shortest write that copies could be far from the quadrature's 17, 43, 140 steps."""
    for K, by_hand in ((1.0, 17), (0.5, 43), (0.3, 140)):
        lo, hi = 1, 4000
        while hi - lo > 1:
            mid = (lo + hi) // 2
            _, (a, b) = _write([1.0], [-1.0], K, mid, 2000)
            lo, hi = (lo, mid) if b[0] > 0 else (mid, hi)
        print("K", K, "shortest write that copies", hi, "by hand", by_hand)
        assert by_hand <= hi < 1.6 * by_hand


def test_source_pulled_but_not_flipped():
    """Prediction 3, the pull at first. Predicted 0.125, **missed**: that estimate, g K / (2 alpha), was for a contact cost
    -g K a b, and the contact written is the spring (g K / 2)(a - b)^2, whose first push is g K |a - b| = 2, so the same
    linear estimate is 0.25. Observed 0.239, at step 7, the copy bit then half way. Could fail: the source could flip."""
    W, step = _built(1, 4.0)
    s, a_path, b_path = template_init(W, [1.0], [-1.0]), [], []
    for _ in range(200):
        s = step(s, np.array([1.0]))
        a, b = template_states(W, s, 1)
        a_path.append(a[0])
        b_path.append(b[0])
    pull = 1.0 - min(a_path)
    print("largest pull on the source", pull, "at step", int(np.argmin(a_path)), "copy bit then", b_path[int(np.argmin(a_path))])
    assert 0.2 < pull < 0.3 and min(a_path) > 0


def test_one_way_contact():
    """Addendum predictions 6 and 7: a one-way contact (non-exact forcing) moves no source, and copies at equal depth with
    threshold exactly 1/4. Could fail: only by a wrong build; it checks that arr has the one-way wire, not a result."""
    W, step = _built(1, 1.0, True)
    s, a_path = template_init(W, [1.0], [-1.0]), []
    for _ in range(200):
        s = step(s, np.array([1.0]))
        a_path.append(template_states(W, s, 1)[0][0])
    print("one-way, equal depth: largest pull on the source", 1.0 - min(a_path))
    assert min(a_path) == 1.0
    (_, _), (a, b) = _write([1.0], [-1.0], 0.26, 4000, 2000, alpha=1.0, one_way=True)
    (_, _), (a2, b2) = _write([1.0], [-1.0], 0.24, 4000, 2000, alpha=1.0, one_way=True)
    print("one-way, equal depth: K 0.26 copy bit", b[0], "K 0.24 copy bit", b2[0])
    assert abs(b[0] - 1.0) < 1e-9 and abs(b2[0] + 1.0) < 1e-9 and a[0] == 1.0
