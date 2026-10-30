"""Acceptable durations and the burst room (plan P41 (e), Claude Opus 5.5, dd3daf04; the book's
``notes/runs/p41e-dd3daf04/RESULTS.md``, predictions in its §0).

What runs in ``arr``: every room and pair below is an arrangement stepped by the dynamics functor (``timed_closed_step``,
``open_step``, ``routine_step`` are checked against ``Phi``); the defect, a distance between two runs' results, is
arithmetic beside them. Laws of the acceptable-duration subcategory, tested: the tensor (two parts not coupled: the
pair's acceptable durations are those acceptable to both) and a composite with two inputs (two parts coupled by a
spring: the composite's defect is the open part's under the contract the composite guarantees).
"""

import jax
import numpy as np

from arrangements import acceptable as ac
from arrangements import two_clocks as tc

jax.config.update("jax_enable_x64", True)

R, EPS = 2.0, 1e-3
GRID = np.linspace(-R, R, 21)
BOX = np.array([[x, y] for x in GRID for y in GRID])
TAUS = np.round(np.arange(0.002, 0.6, 0.002), 6)


def _tstar(arr, starts, coords=None):
    return ac.acceptable_duration(ac.defect_curve(arr, ac.timed_closed_step(arr), TAUS, starts, coords), TAUS, EPS)


def test_one_part_matches_hand():
    arr = ac.one_part(1.0)
    d = ac.defect_curve(arr, ac.timed_closed_step(arr), [0.01], GRID[:, None])[0]
    assert abs(d - R * 0.01 ** 2 / 4) < 1e-12


def test_tensor_is_intersection():
    a, b, pair = ac.one_part(1.0), ac.one_part(0.1), ac.two_parts(1.0, 0.1, 0.0)
    assert _tstar(pair, BOX) == min(_tstar(a, GRID[:, None]), _tstar(b, GRID[:, None]))


def test_composite_equals_part_under_contract():
    pair, part = ac.two_parts(1.0, 1.0, 1.0), ac.part_open(1.0, 1.0)
    for t in (0.01, 0.0148):
        comp = ac.defect_curve(pair, ac.timed_closed_step(pair), [t], BOX)[0]
        assert abs(comp - ac.open_defect(part, t, GRID, GRID, 6.0)) < 1e-12


def test_one_chair_position_two_slow_motions():
    """Chairs 4.6 apart, store 30, routine off: holding apart, no burst; holding together, the chairs move in."""
    arr = ac.burst_room()
    step = ac.input_step(arr, 0.004)
    ends = {}
    for name, q in (("apart", (-2.3, 2.3)), ("together", (-0.4825, 0.4825))):
        p = tc.run(arr, step, np.zeros(50000), start=[q[0], q[1], -2.3, 2.3, 30.0, -1.2])
        ends[name] = p[3] - p[2]
    assert abs(ends["apart"] - 4.6) < 1e-3 and ends["together"] < 4.0
