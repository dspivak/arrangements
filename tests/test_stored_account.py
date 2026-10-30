"""A stored account averaged over its bursts, and the forms over a transit (plan P41 (f), Claude Opus 5.5, f0638d8e;
the book's ``notes/runs/p41f-f0638d8e/RESULTS.md``, predictions in its §§0, 3, 5, 7).

What runs in ``arr``: the burst room, the same room with its chairs held (``eta_w=0``), the friction room and the
stick-slip room, each stepped by the dynamics functor (``input_step``/``routine_step`` are checked against ``Phi``).
Burst detection and the comparisons are arithmetic on recorded paths. The duty table is the one measured in the run
(RESULTS §4), at chairs held still with the people under the routine.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements import acceptable as ac
from arrangements import stored_account as sa
from arrangements import two_clocks as tc

jax.config.update("jax_enable_x64", True)

TAU = 0.004
DUTY_S = [1.039, 1.1307, 1.2261, 1.3254, 1.5352, 1.7598, 2.0613, 2.3853, 2.7315, 3.0992, 3.4879, 3.8967, 4.3246,
          4.7704]
DUTY_D = [0.001181, 0.001883, 0.00247, 0.003035, 0.004169, 0.005343, 0.006887, 0.00853, 0.010279, 0.012129, 0.01408,
          0.01613, 0.018272, 0.020503]
gate = lambda y: 0.5 * (1.0 + np.tanh(10.0 * y))


def _outcome(hold):
    room = ac.burst_room()
    nh, n = int(round(hold / TAU)), int(round((hold + 300.0) / TAU))
    p = tc.run(room, ac.input_step(room, TAU), np.r_[np.ones(nh), np.zeros(n - nh)])
    return "together" if p[1] - p[0] < 2.0 else "apart"


def test_release_then_burst_is_a_race():
    """Chairs and people are at one place at release for holds 32.8 and 33; only the store differs, and decides."""
    assert _outcome(32.8) == "apart" and _outcome(33.0) == "together"


def test_held_chairs_do_not_move():
    room = ac.burst_room(eta_w=0.0)
    p = tc.run(room, ac.input_step(room, TAU), np.ones(int(round(40.0 / TAU))))
    assert p[2] == -2.5 and p[3] == 2.5


def test_friction_response_is_a_dead_zone_and_monotone():
    """r(xi) = 0 on the ball |xi| <= sqrt(2 a s_0), s_0 the first account with bursts; <r(x)-r(x'), x-x'> >= 0."""
    duty = sa.tabulated(DUTY_S, DUTY_D)
    P = sa.friction_chairs([0.0, 0.0], duty)
    r = lambda xi: P.timed_step(P.init, jnp.asarray(xi), 1.0) - P.init
    F = np.sqrt(2.0 * DUTY_S[0])
    for th in np.linspace(0, 2 * np.pi, 13):
        assert np.all(np.asarray(r(0.999 * F * np.array([np.cos(th), np.sin(th)]))) == 0.0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        x, y = rng.uniform(-3.5, 3.5, 2), rng.uniform(-3.5, 3.5, 2)
        assert float(jnp.dot(r(x) - r(y), x - y)) >= -1e-12


def test_averaging_burst_room_and_friction_room():
    """From the burst room's state after its first burst, routine held to t=120: the two rooms' chair separations at
    every burst end agree to 0.03 (one burst moves them 0.18)."""
    room = ac.burst_room()
    n = int(round(120.0 / TAU))
    pA = tc.run(room, ac.input_step(room, TAU), np.ones(n), record=True)
    on = gate(pA[:, 5]) > 0.5
    ends = np.flatnonzero(~on[1:] & on[:-1]) + 1
    e1 = ends[0]
    B = sa.friction_room(sa.tabulated(DUTY_S, DUTY_D), q=pA[e1, :2], w=pA[e1, 2:4])
    pB = tc.run(B, tc.routine_step(B, TAU), np.ones(n - e1 - 1), record=True)
    DA, DB = pA[:, 3] - pA[:, 2], np.r_[pA[e1, 3] - pA[e1, 2], pB[:, 3] - pB[:, 2]]
    assert len(ends) >= 5
    assert max(abs(DA[e] - DB[e - e1]) for e in ends) < 0.03


def test_friction_mass_sticks_below_static_friction():
    P = sa.friction_mass(0.0, m=20.0, c=40.0, F_s=0.5, F_k=0.45, v_s=0.002)
    s = P.init
    for _ in range(1000):
        s = P.timed_step(s, jnp.array([0.499]), TAU)
    assert float(s[0][0]) == 0.0 and float(s[1][0]) == 0.0
    for _ in range(1000):
        s = P.timed_step(s, jnp.array([0.501]), TAU)
    assert float(s[0][0]) > 0.0


def test_stick_slip_people_match_burst_room():
    """The stick-slip room's people under the routine are the burst room's (regression: a name clash once set the
    talk strength to the floor's drag)."""
    A, C = ac.burst_room(), sa.stick_slip_room()
    n = int(round(5.0 / TAU))
    pa = tc.run(A, ac.input_step(A, TAU), np.ones(n))
    pc = tc.run(C, tc.routine_step(C, TAU), np.ones(n))
    assert abs((pa[1] - pa[0]) - (pc[1] - pc[0])) < 1e-9
