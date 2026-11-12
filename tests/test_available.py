"""Forms available over a slow motion, along one-way nesting (book plan P41 (j); ``arrangements/available.py``). The room,
the guest at ``mu = 9/20`` and the routine are Ch 5 ``ex.guest``.

Predictions fixed before the code: the book repo's ``notes/runs/p41j-836d12ce/PREDICTIONS.md``; the lattices and their
squares are read in that folder's ``piece*.py``. Claude Opus 5.5, 836d12ce, 2026-11-05.

*Reasonable or weird* (the agent's judgment): a third person whose choice of stance is open only when two others sit
together is reasonable at the scale of a room, and the same shape at a cell (a target gene with two stable states only
while a slow chromatin mark is set) and a society (an option that exists only once two institutions have settled
together). Weird: the guest is quick and reads for free, which no real reader does; the sharp 80/81 threshold is one
dimension and no noise, as in P41 (d).
"""
import numpy as np
import pytest

from arrangements import available as av, two_clocks as tc

TAU = 0.05
EPS = 0.01


def _hand_guest(x, mu, u=0.0, a=1.0, c=3.0, b=2.6, w0=(-2.5, 2.5), d_c=av.D_C):
    q1, q2, w1, w2, z = x
    e = np.exp(-(q1 - q2) ** 2 / 2)
    f1 = -a * (q1 - w1) + c * (q2 - q1) * e + u * (q2 - q1)
    f2 = -a * (q2 - w2) + c * (q1 - q2) * e + u * (q1 - q2)
    g1 = a * (q1 - w1) - b * (w1 - w0[0])
    g2 = a * (q2 - w2) - b * (w2 - w0[1])
    t = mu * (q2 - q1 - d_c)
    return np.array([q1 + TAU * f1, q2 + TAU * f2, w1 + EPS * TAU * g1, w2 + EPS * TAU * g2,
                     z + TAU * (-(z ** 3 - z) + t)])


@pytest.mark.parametrize("mu", [0.2, 0.45])
def test_guest_room_is_the_hand_recurrence(mu):
    arr = av.room_with_guest(EPS, mu)
    step = tc.routine_step(arr, TAU)
    u = [1.0] * 30 + [0.0] * 30
    path = tc.run(arr, step, u, start=[-2.0, 2.2, -2.5, 2.5, 0.3], record=True)
    x = np.array([-2.0, 2.2, -2.5, 2.5, 0.3])
    for row, ui in zip(path, u):
        x = _hand_guest(x, mu, ui)
        assert np.allclose(row, x, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("mu", [0.2, 0.45])
def test_the_guest_moves_nothing_in_the_room(mu):
    """One-way, exactly: a skew product over the room at every step, routine on or off."""
    room = tc.room_flat(EPS, (-2.5, 2.5), (-2.5, 2.5))
    whole = av.room_with_guest(EPS, mu)
    s_room, s_whole = tc.routine_step(room, TAU), tc.routine_step(whole, TAU)
    rng = np.random.default_rng(1)
    for _ in range(3):
        start = rng.uniform(-3, 3, 4)
        u = rng.uniform(0, 1.2, 300)
        a = tc.run(room, s_room, u, start=start, record=True)
        b = tc.run(whole, s_whole, u, start=np.r_[start, rng.uniform(-2, 2)], record=True)
        assert np.array_equal(a, b[:, :4])


def test_newly_available_through_a_one_way_wire():
    """Room apart: the guest's stance at -1 falls back. Past the routine's threshold it holds."""
    whole = av.room_with_guest(EPS, 0.45)
    st = tc.routine_step(whole, TAU)
    s = tc.run(whole, st, np.zeros(4000), start=[-2.5, 2.5, -2.5, 2.5, 1.0])

    def nudged_end(state):
        n = np.array(state)
        n[4] = -1.0
        return tc.run(whole, st, np.zeros(3000), start=n)

    assert nudged_end(s)[4] > 1.0
    for hold, together in ((80, False), (81, True)):
        t = tc.run(whole, st, np.r_[np.ones(hold), np.zeros(20000)], start=s)
        e = nudged_end(t)
        assert (t[1] - t[0] < 1.0) == together
        assert (e[4] < 0.0) == together


def _force_on_chairs(arr, position):
    step = tc.routine_step(arr, TAU)
    p = tc.run(arr, step, [0.0], start=position)
    return ((p[3] - p[2]) - (position[3] - position[2])) / TAU


def test_the_slow_motion_descends_only_one_way():
    """Chairs held 4.6 apart. The two-way room's force on them depends on the form held, by sign; the one-way room's
    does not depend on it at all."""
    def forms(build):
        held = build(0.0)
        step = tc.routine_step(held, TAU)
        return [tc.run(held, step, np.zeros(6000), start=[-d / 2, d / 2, -2.3, 2.3]) for d in (0.5, 4.5)]

    t, a = forms(lambda e: tc.room_flat(e, (-2.5, 2.5), (-2.3, 2.3)))
    assert t[1] - t[0] < 1.5 < 3 < a[1] - a[0]
    room = tc.room_flat(1.0, (-2.5, 2.5), (-2.3, 2.3))
    ft, fa = _force_on_chairs(room, t), _force_on_chairs(room, a)
    assert ft < -1 and fa > 0.5

    t, a = forms(lambda e: av.one_way_room(e, w=(-2.3, 2.3)))
    room = av.one_way_room(1.0, w=(-2.3, 2.3))
    assert _force_on_chairs(room, t) == _force_on_chairs(room, a) != 0.0


def test_a_rest_of_the_whole_over_each_slow_rest():
    """Each minimal attractor of the slow motion has a rest of the whole over it, so the shadow is onto (its lattice is
    the subsets of the three). Started near each, the whole stays and comes to rest there."""
    whole = av.room_with_guest(EPS, 0.45)
    st = tc.routine_step(whole, TAU)
    for d, D, z in ((0.6456, 3.7904, -1.0), (0.6456, 3.7904, 1.0), (4.9998, 5.0, 1.5145)):
        end = tc.run(whole, st, np.zeros(20000), start=[-d / 2, d / 2, -D / 2, D / 2, z])
        step = tc.run(whole, st, [0.0], start=end)
        assert abs((end[1] - end[0]) - d) < 1e-3 and abs((end[3] - end[2]) - D) < 1e-3 and abs(end[4] - z) < 1e-3
        assert np.abs(step - end).max() < 1e-9


def test_the_slow_motion_runs_away_from_both_folds():
    """Near where *together* ends the load pulls the chairs in; near where *apart* ends it pushes them out. So no fold
    is reached and the slow motion is a continuous step."""
    room = tc.room_flat(1.0, (-2.5, 2.5), (-2.5, 2.5))

    def settled(D, d0):
        held = tc.room_flat(0.0, (-2.5, 2.5), (-D / 2, D / 2))
        return tc.run(held, tc.routine_step(held, TAU), np.zeros(8000), start=[-d0 / 2, d0 / 2, -D / 2, D / 2])

    t, a = settled(4.70, 0.5), settled(3.13, 5.0)
    assert t[1] - t[0] < 1.3 and a[1] - a[0] > 2.5
    assert _force_on_chairs(room, t) < 0 < _force_on_chairs(room, a)


def test_two_sources():
    """A reader with two inputs: the room and the lamp are not moved by it."""
    base = av.room_lamp(EPS)
    whole = av.room_lamp_guest(EPS, 0.45, 0.2)
    s_b, s_w = tc.routine_step(base, TAU), tc.routine_step(whole, TAU)
    u = np.r_[np.ones(50), np.zeros(250)]
    a = tc.run(base, s_b, u, start=[-2.0, 2.2, -2.5, 2.5, 0.4], record=True)
    b = tc.run(whole, s_w, u, start=[-2.0, 2.2, -2.5, 2.5, 0.4, -0.7], record=True)
    assert np.array_equal(a, b[:, :5])


def test_chairs_alone_is_the_one_way_rooms_source():
    source = av.chairs_alone(EPS, w=(-2.1, 2.2))
    whole = av.one_way_room(EPS, w=(-2.1, 2.2))
    s_s, s_w = tc.routine_step(source, TAU), tc.routine_step(whole, TAU)
    a = tc.run(source, s_s, np.zeros(200), start=[-2.1, 2.2], record=True)
    b = tc.run(whole, s_w, np.zeros(200), start=[-1.0, 0.7, -2.1, 2.2], record=True)
    assert np.array_equal(a, b[:, 2:4])
