"""Getting stronger: two wholes feeding each other, each with a slow coordinate, its skill, grown by the play they
hold together; a capacity becomes newly available to both by playing. Book plan P73 (David, 2026-11-05, d5eec4fb:
"getting stronger means a newly available capacity. both soccer teams get new capacities by playing each oteher";
"as rec'd" to running this stage). Everything not quoted from him is the agent's.

**The whole.** ``test_wholes_cooperate.py``'s two gated pairs feeding each other through the parent (group 1 receives
``s + k' out2``, group 2 receives ``k out1``), with two changes. (1) Each group's coupler has its own integrator, a
*skill* ``kap`` (mobility ``h ETA``), which lowers the group's threshold of living: its gate's half-saturation is
``K(kap) = 1/(1 + kap)``, so a pair alone lives at a sun above ``2.1213 sqrt(K(kap))``. (2) The parent computes the
*play*, the form the two wholes hold together, the gate ``g_p = y1 y2/(1 + y1 y2)`` on their joint activity, and
returns it to both on a second input port; each group's effort on its skill is ``play (1 - kap) - LAM kap``: skill
grows with play and decays without. Configuration ``(q_A1, q_B1, kap_1, q_A2, q_B2, kap_2, s)``; ``w = 1/2``,
``k = 1``, ``k' = 0.4``, ``LAM = 1``, ``ETA = 0.05``. This is ``test_hearing_kept.py``'s mechanism one level up:
there a pair's hearing was kept up by the pair's joint activity; here two wholes' skills are grown by their joint play.

**The structure** (P41 (e)'s category of elements, with skill as the slow part): an object is a slow state, the two
skills, with a form of the quick parts available over it; a morphism is a slow path carrying the form into another's
basin. Playing drives the slow path; a capacity "newly available" is an object reached along it whose form was not
available at the start. The ceiling theorem gives, at each skill, the joint ceiling by the trace of the two level maps.

**Predictions, written before the arrangement was built** (`.scratch/p73-structures-d5eec4fb/stronger_predictions.py`):
  S0  ``Phi``, ``run_closed`` and the loop agree to machine precision over fifty steps.
  S1  (sun 2, from an alive unskilled start) the whole settles at skills ``(0.497, 0.497)`` and levels
      ``(8.464, 4.232, 11.165, 5.583)``, the slow motion's rest on the alive sheet, within ``5e-3``; play ``0.99``.
  S2  (the earned capacity) the two-way whole's threshold is ``1.375`` unskilled and ``1.124`` at skill ``0.497``.
      From the skilled rest, lowering the sun to ``1.2``: alive, near ``(3.81, 1.91, 4.81, 2.40)`` with skills near
      ``0.49``, within 15% (the rest moves a little as play changes). From the alive but unskilled start at sun ``1.2``:
      dead, and the skills stay below ``0.05``, since there is no play to grow them. The capacity "alive at sun 1.2" was
      always in the whole and is available only over skill the whole earns by playing.
      [The second half REFUTED on the first run: that start is active, so it plays from the first step and earns
      its skill before it fades; replaced, before the rerun, by a low start that dies with skills below ``0.1`` and
      a threshold start on the ray ``(a, a/2; a, a/2)`` between ``0.5`` and ``5``: the race between earning and fading.]
  S3  (one way) from the dead state at sun 2 the whole stays dead and unskilled: no play, no skill, no form.
  S4  (losing it) from the skilled rest at sun ``1.2``, with the play's return to the skills cut (the league shut, the
      feeding kept), skill decays, the threshold rises back past ``1.2``, and the whole dies within 600 units, skills
      ending below ``0.05``: a capacity made available by play goes when the play stops.
  S5  (both get stronger) from one whole at skill ``0.6`` and the other at ``0``, at sun 2, both end at ``0.497``
      within ``5e-3``: playing each other brings both to the same skill.
  S6  (Bekic at a skill) at sun 2 with the skills frozen at ``0.497``, the flat greatest self-sustaining level of the
      parts equals the trace of the two skilled level maps, ``(8.464, 4.232, 11.165, 5.583)``, within ``1e-3``.

Everything here is one nested closed arrangement stepped by the dynamics functor (S0 licenses ``settle``); the
predictions come from the reduced slow motion on the sheets.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, zero_form
from arrangements.coalgebra import configuration
from arrangements.rvect import euclidean
from arrangements.wiring import compose_seq, tensor_arrangements, trivial_parameter

from tests.test_bees_shadow import EMPTY, MOB, W, gfp_below, part, run_closed, run_phi, settle, steps_for, sun

K, KP, LAM, ETA = 1.0, 0.4, 1.0, 0.05
SUN = 2.0
HELD = 1e-3
TOPS = np.array([28.0, 14.0, 40.0, 20.0])
K_REST = 0.4974
LEVELS_REST = np.array([8.4636, 4.2318, 11.1652, 5.5826])


def gate_k(q_A, q_B, kap):
    u = q_A * q_B
    return u / (1.0 / (1.0 + kap) + u)


def group_k(q_A, q_B, k0, lam=LAM, eta=ETA, label="pair"):
    """A pair as a group with its own skill: ``<R^0|R^0> -> <R^2|R^1>``, two input ports (supply, play), one output
    (A's activity). The coupler's integrator is the skill; its gate's half-saturation falls with skill; its effort on
    the skill is the play's growth minus decay."""

    def coupler_omega(q, e, n_in):
        kap = q[0]
        q_A, q_B = e[0], e[1]
        s_in, play = n_in[0], n_in[1]
        F = jnp.array([s_in * gate_k(q_A, q_B, kap) + W * q_B, W * q_A])
        xi_k = jnp.array([play * (1.0 - kap) - lam * kap])
        return (xi_k, F, jnp.zeros(2))

    coupler = Arrangement(
        P=configuration(euclidean(1, MOB * eta), init=jnp.array([float(k0)])),
        in_dim_M=0, out_dim_M=2, in_dim_N=2, out_dim_N=1,
        out_f=lambda q, e: e[:1],
        in_f=lambda q, e, n_in: EMPTY,
        omega=coupler_omega,
        label=f"{label}_wiring_with_skill",
    )
    return compose_seq(tensor_arrangements([part(q_A, f"A_{label}"), part(q_B, f"B_{label}")]), coupler)


def parent(k=K, kp=KP, play_on=True):
    """``<R^4|R^3> -> <R^0|R^0>``: reads (out1, out2, s); supplies group 1 with s + kp out2 and group 2 with k out1,
    and returns to both the play, the gate on their joint activity (or 0 when the league is shut)."""

    def in_f(q, m_out, n_in):
        y1, y2, s = m_out[0], m_out[1], m_out[2]
        play = y1 * y2 / (1.0 + y1 * y2) if play_on else 0.0 * y1
        return jnp.array([s + kp * y2, play, k * y1, play])

    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=4, out_dim_M=3, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: EMPTY,
        in_f=in_f,
        omega=zero_form,
        label="each_feeds_the_other_and_they_play",
    )


def whole(q=(5.0, 2.5, 0.0, 6.0, 3.0, 0.0), s=SUN, eta=ETA, play_on=True):
    """Configuration (q_A1, q_B1, kap_1, q_A2, q_B2, kap_2, s)."""
    boxes = [group_k(q[0], q[1], q[2], eta=eta, label="bees"), group_k(q[3], q[4], q[5], eta=eta, label="us"), sun(s)]
    return compose_seq(tensor_arrangements(boxes), parent(play_on=play_on))


PARTS = [0, 1, 3, 4]
SKILLS = [2, 5]


def parts_step_at(arr, kaps, s):
    """The step on the four parts' coordinates with the skills frozen at ``kaps`` and the sun at ``s``."""
    omega = arr.omega

    @jax.jit
    def F_jnp(x):
        q = jnp.array([x[0], x[1], kaps[0], x[2], x[3], kaps[1], s])
        xi_q, _, _ = omega(q, EMPTY, EMPTY)
        return (q + MOB * xi_q)[jnp.array(PARTS)]

    def F(x):
        return np.asarray(F_jnp(jnp.asarray(x, float)))
    return F


def level_k(u, kap, tops=(40.0, 20.0)):
    """The skilled pair's level map, Kleene from the top on the pair alone at supply u and skill kap."""
    def Fp(x):
        q_A, q_B = x
        return np.array([q_A + MOB * (u * gate_k(q_A, q_B, kap) + W * q_B - q_A), q_B + MOB * (W * q_A - q_B)])
    return gfp_below(Fp, np.array(tops), tol=1e-9)


def loop_trace_k(s, kaps, tops=(28.0, 40.0)):
    y1, y2 = tops
    for _ in range(100000):
        l1 = level_k(s + KP * y2, kaps[0])
        y1n = min(l1[0], tops[0])
        l2 = level_k(K * y1n, kaps[1])
        y2n = min(l2[0], tops[1])
        if abs(y1n - y1) < 1e-7 and abs(y2n - y2) < 1e-7:
            return np.array([l1[0], l1[1], l2[0], l2[1]])
        y1, y2 = y1n, y2n
    return np.array([l1[0], l1[1], l2[0], l2[1]])


def test_S0_the_runs_are_phis():
    b = whole()
    print("label:", b.label)
    by_phi = run_phi(b, 50, EMPTY)
    by_closed = run_closed(b, 50, lambda t: EMPTY)
    gap = float(np.max(np.abs(by_phi - by_closed)))
    print("S0 Phi vs run_closed:", gap)
    assert gap < 1e-13
    np.testing.assert_allclose(settle(b, 50), by_closed[-1], atol=1e-13)


def test_S1_skills_grow_by_play():
    end = settle(whole(), steps_for(600.0))
    print("S1 rest:", end)
    np.testing.assert_allclose(end[SKILLS], [K_REST, K_REST], atol=5e-3)
    np.testing.assert_allclose(end[PARTS], LEVELS_REST, rtol=5e-3)
    play = end[0] * end[3] / (1 + end[0] * end[3])
    print("S1 play at the rest:", play)
    assert play > 0.98


def test_S2_the_earned_capacity():
    skilled = settle(whole(), steps_for(600.0))
    low = settle(whole(tuple(skilled[:6]), s=1.2), steps_for(600.0))
    print("S2 skilled rest, sun lowered to 1.2:", low)
    assert np.min(low[PARTS]) > HELD
    np.testing.assert_allclose(low[PARTS], [3.81, 1.91, 4.81, 2.40], rtol=0.15)
    np.testing.assert_allclose(low[SKILLS], [0.49, 0.49], atol=0.05)
    # Predicted dead; REFUTED on the first run: the unskilled start (5, 2.5; 6, 3) at sun 1.2 ended at the skilled
    # alive rest. Mechanism: the start is active, so there is play from the first step, and skill grows (rate ETA)
    # faster than the unfed parts decay, lowering the threshold under 1.2 before the form fades: a transient of play
    # earns the capacity. The prediction replacing it, written before the rerun: a low unskilled start,
    # (0.5, 0.25; 0.5, 0.25), has play 0.2 at most, dies, and its skills end below 0.1; and along the ray
    # (a, a/2; a, a/2) the start from which the whole earns its skill in time is a threshold a* between 0.5 and 5.
    high = settle(whole(s=1.2), steps_for(600.0))
    print("S2 unskilled but active start at sun 1.2 (refuted prediction, now a race):", high)
    assert np.min(high[PARTS]) > HELD
    low_start = settle(whole((0.5, 0.25, 0.0, 0.5, 0.25, 0.0), s=1.2), steps_for(600.0))
    print("S2 low unskilled start at sun 1.2:", low_start)
    assert np.max(np.abs(low_start[PARTS])) < HELD and np.max(low_start[SKILLS]) < 0.1
    lo, hi = 0.5, 5.0
    for _ in range(10):
        mid = 0.5 * (lo + hi)
        e = settle(whole((mid, mid / 2, 0.0, mid, mid / 2, 0.0), s=1.2), steps_for(400.0))
        if np.min(e[PARTS]) > HELD:
            hi = mid
        else:
            lo = mid
    print("S2 the start from which play earns the skill in time, on the ray (a, a/2; a, a/2):", 0.5 * (lo + hi))
    assert 0.5 < 0.5 * (lo + hi) < 5.0


def test_S3_no_play_no_skill_no_form():
    dead = settle(whole((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), s=SUN), steps_for(600.0))
    print("S3 from the dead state at sun 2:", dead)
    assert np.max(np.abs(dead[PARTS])) < HELD and np.max(dead[SKILLS]) < 1e-6


def test_S4_the_capacity_goes_when_the_play_stops():
    skilled = settle(whole(), steps_for(600.0))
    low = settle(whole(tuple(skilled[:6]), s=1.2), steps_for(600.0))
    assert np.min(low[PARTS]) > HELD
    shut = settle(whole(tuple(low[:6]), s=1.2, play_on=False), steps_for(600.0))
    print("S4 league shut at sun 1.2, from the skilled alive rest:", shut)
    assert np.max(np.abs(shut[PARTS])) < HELD and np.max(shut[SKILLS]) < 0.05


def test_S5_both_get_stronger():
    end = settle(whole((5.0, 2.5, 0.6, 6.0, 3.0, 0.0)), steps_for(600.0))
    print("S5 one skilled, one not, after playing:", end)
    np.testing.assert_allclose(end[SKILLS], [K_REST, K_REST], atol=5e-3)


def test_S6_bekic_at_a_skill():
    b = whole(eta=0.0, q=(5.0, 2.5, K_REST, 6.0, 3.0, K_REST))
    F = parts_step_at(b, (K_REST, K_REST), SUN)
    nuF = gfp_below(F, TOPS)
    staged = loop_trace_k(SUN, (K_REST, K_REST))
    print("S6 nu F at skill 0.497:", nuF, " trace of skilled level maps:", staged)
    np.testing.assert_allclose(nuF, LEVELS_REST, atol=1e-3)
    np.testing.assert_allclose(staged, nuF, atol=1e-3)
