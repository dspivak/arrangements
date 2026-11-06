"""Two open wholes feeding each other: the bees and us, two-way. The check of the open theorem (book plan P73;
`notes/runs/p73-structures-d5eec4fb/RESULTS.md`, "The open theorem"). David, 2026-11-05 (d5eec4fb): "it's not
tenable to have it be closed. wholes need to cooperate and you kill it by making it closed. can you try again?"
Everything not quoted from him is the agent's.

**The whole.** ``test_bees_shadow.py``'s two groups, each a gated pair with an open boundary (its supply in, A's
activity out), and the sun; the parent now feeds group 1 with ``s + k' out2`` and group 2 with ``k out1``: each
whole receives from the other. Configuration ``(q_A1, q_B1, q_A2, q_B2, s)``; hearing in each pair a fixed loudness
``w = 1/2``; ``k = 1``, ``k' = 0.4`` (below the loop-gain bound ``1.78 k k' < 1`` that keeps the levels finite).

**The open theorem's objects.** Each group's *level map* ``L(u)``: its greatest self-sustaining output under a constant
supply ``u``, by Kleene from the top on the group alone. The composite's greatest self-sustaining level is the
trace of the two level maps around the loop, ``nu (y1, y2). (L(s + k' y2), L(k y1))`` by Kleene from the tops
(Bekic), and it must equal ``nu F`` of the flat whole. Tops ``M = (28, 14, 40, 20)``, threshold ``1/2`` (a first
run with tops ``(24, 12, 30, 15)`` failed C1's box check, since with the feedback the step leaves that box: my
arithmetic, corrected before the rerun; the predictions' numbers are unchanged).

**Predictions, written before the arrangement was built** (`.scratch/p73-structures-d5eec4fb/cooperate_predictions.py`,
the level maps from the pair's closed form):
  C0  ``Phi``, ``run_closed`` and the loop agree to machine precision over fifty steps.
  C1  (sun ``s = 4``) the step is monotone (every Jacobian entry nonnegative at 40 random states) and keeps the box;
      ``nu F = (17.9201, 8.9601, 23.8095, 11.9048)`` to ``1e-3``, and the trace of the two level maps equals it;
      from an alive start the whole settles at ``nu F``.
  C2  (sun ``s = 1.5``, below a pair's own threshold ``2.1213``) ``nu F = (4.9272, 2.4636, 6.2496, 3.1248)``, the
      trace of the level maps equals it, and from an alive start the whole settles there: **the two wholes live at a
      sun on which neither lives alone**; from a dead start it stays dead (``mu F = 0``, bistable).
  C3  (``s = 1.5``, the feedback cut, ``k' = 0``) ``nu F = 0`` and from the alive start both die: cutting the
      cooperation kills both.
  C4  (sun off, ``k' = 0.4``) ``nu F = 0`` and both die: the loop needs the sun (no free lunch).
  C5  the sun threshold of the two-way whole is ``1.3754`` by the level maps, against ``2.1213`` for a pair alone:
      from the alive start the whole survives at ``s = 1.45`` and dies at ``s = 1.30``.

Everything here is one nested closed arrangement stepped by the dynamics functor (C0 licenses ``settle``); the
theorem's objects are calculations on its step and on each group's own step.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, zero_form
from arrangements.wiring import compose_seq, tensor_arrangements, trivial_parameter

from tests.test_bees_shadow import EMPTY, MOB, W, gate, gfp_below, group, lfp_above, run_closed, run_phi, settle, steps_for, sun
from tests.test_mutual_need import gated_roots

K, KP = 1.0, 0.4
SUN = 4.0
TOPS = np.array([28.0, 14.0, 40.0, 20.0])   # a box the step keeps: F(M) <= M (checked in C1); (24, 12, 30, 15) was not
HELD = 1e-3


def parent(k=K, kp=KP):
    """``<R^2|R^3> -> <R^0|R^0>``: reads (out1, out2, s); supplies group 1 with s + kp out2, group 2 with k out1."""
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=2, out_dim_M=3, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: EMPTY,
        in_f=lambda q, m_out, n_in: jnp.array([m_out[2] + kp * m_out[1], k * m_out[0]]),
        omega=zero_form,
        label="each_feeds_the_other",
    )


def whole(q=(18.0, 9.0, 24.0, 12.0), s=SUN, k=K, kp=KP):
    boxes = [group(q[0], q[1], label="bees"), group(q[2], q[3], label="us"), sun(s)]
    return compose_seq(tensor_arrangements(boxes), parent(k, kp))


def parts_step(arr, s):
    omega = arr.omega

    @jax.jit
    def F_jnp(x):
        q = jnp.concatenate([x, jnp.array([s])])
        xi_q, _, _ = omega(q, EMPTY, EMPTY)
        return (q + MOB * xi_q)[:4]

    def F(x):
        return np.asarray(F_jnp(jnp.asarray(x, float)))
    return F


def pair_level(u, tops=(40.0, 20.0)):
    """A group's level map at the input level ``u``: its greatest self-sustaining state under the constant supply
    ``u``, the pair's larger rest from its closed form (``gated_roots``), ``0`` below the pair's threshold; equal
    to Kleene from the top on the group alone (checked once below, ``test_C0``), and fast."""
    hi = gated_roots(u)[1] if u > 0 else float("nan")
    if not np.isfinite(hi):
        return np.array([0.0, 0.0])
    return np.array([min(hi, tops[0]), min(W * hi, tops[1])])


def pair_level_kleene(u, tops=(40.0, 20.0)):
    def Fp(x):
        q_A, q_B = x
        return np.array([q_A + MOB * (u * gate(q_A, q_B) + W * q_B - q_A), q_B + MOB * (W * q_A - q_B)])
    return gfp_below(Fp, np.array(tops), tol=1e-9)


def loop_trace(s, k=K, kp=KP, tops=(28.0, 40.0)):
    """The trace of the two level maps around the loop, Kleene from the tops."""
    y1, y2 = tops
    for _ in range(100000):
        l1 = pair_level(s + kp * y2)
        y1n = min(l1[0], tops[0])
        l2 = pair_level(k * y1n)
        y2n = min(l2[0], tops[1])
        if abs(y1n - y1) < 1e-9 and abs(y2n - y2) < 1e-9:
            return np.array([l1[0], l1[1], l2[0], l2[1]])
        y1, y2 = y1n, y2n
    return np.array([l1[0], l1[1], l2[0], l2[1]])


def test_C0_the_runs_are_phis():
    b = whole()
    print("label:", b.label)
    by_phi = run_phi(b, 50, EMPTY)
    by_closed = run_closed(b, 50, lambda t: EMPTY)
    gap = float(np.max(np.abs(by_phi - by_closed)))
    print("C0 Phi vs run_closed:", gap)
    assert gap < 1e-13
    np.testing.assert_allclose(settle(b, 50), by_closed[-1], atol=1e-13)
    # the closed-form level map is Kleene from the top on the group alone (one check, at two supplies)
    for u in (4.0, 1.5):
        np.testing.assert_allclose(pair_level(u), pair_level_kleene(u), atol=1e-6)


def test_C1_sun_on_two_way():
    b = whole()
    F = parts_step(b, SUN)
    rng = np.random.default_rng(20261105)

    def step5(y):
        q = jnp.concatenate([y, jnp.array([SUN])])
        return (q + MOB * b.omega(q, EMPTY, EMPTY)[0])[:4]

    worst = 0.0
    for _ in range(40):
        x = rng.uniform(0.0, 1.0, 4) * TOPS
        worst = min(worst, float(np.asarray(jax.jacobian(step5)(jnp.asarray(x))).min()))
    print("C1 smallest Jacobian entry:", worst)
    assert worst >= 0.0 and np.all(F(TOPS) <= TOPS + 1e-12)
    nuF = gfp_below(F, TOPS)
    staged = loop_trace(SUN)
    print("C1 nu F:", nuF, " trace of level maps:", staged)
    np.testing.assert_allclose(nuF, [17.9201, 8.9601, 23.8095, 11.9048], atol=1e-3)
    np.testing.assert_allclose(staged, nuF, atol=1e-5)
    end = settle(whole((18.0, 9.0, 24.0, 12.0)), steps_for(600.0))[:4]
    print("C1 from an alive start:", end)
    np.testing.assert_allclose(end, nuF, atol=1e-5)


def test_C2_together_at_a_sun_too_weak_for_either_alone():
    s = 1.5
    F = parts_step(whole(s=s), s)
    nuF = gfp_below(F, TOPS)
    muF = lfp_above(F, np.zeros(4))
    staged = loop_trace(s)
    print("C2 nu F:", nuF, " mu F:", muF, " trace:", staged, " a pair alone at 1.5:", pair_level(s))
    np.testing.assert_allclose(nuF, [4.9272, 2.4636, 6.2496, 3.1248], atol=1e-3)
    np.testing.assert_allclose(staged, nuF, atol=1e-5)
    assert np.max(muF) < 1e-6 and np.max(pair_level(s)) < 1e-6
    alive = settle(whole((5.0, 2.5, 6.0, 3.0), s=s), steps_for(600.0))[:4]
    dead = settle(whole((0.1, 0.1, 0.1, 0.1), s=s), steps_for(600.0))[:4]
    print("C2 from an alive start:", alive, " from a dead start:", dead)
    np.testing.assert_allclose(alive, nuF, atol=1e-5)
    assert np.max(np.abs(dead)) < HELD


def test_C3_cutting_the_cooperation_kills():
    s = 1.5
    F = parts_step(whole(s=s, kp=0.0), s)
    nuF = gfp_below(F, TOPS)
    end = settle(whole((5.0, 2.5, 6.0, 3.0), s=s, kp=0.0), steps_for(600.0))[:4]
    print("C3 nu F with the feedback cut:", nuF, " from the alive start:", end)
    assert np.max(nuF) < 1e-6 and np.max(np.abs(end)) < HELD


def test_C4_the_loop_needs_the_sun():
    F = parts_step(whole(s=0.0), 0.0)
    nuF = gfp_below(F, TOPS)
    end = settle(whole((18.0, 9.0, 24.0, 12.0), s=0.0), steps_for(600.0))[:4]
    print("C4 nu F, sun off:", nuF, " from the alive start:", end)
    assert np.max(nuF) < 1e-6 and np.max(np.abs(end)) < HELD


def test_C5_the_sun_threshold_drops_with_cooperation():
    lo, hi = 0.0, 2.2
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if loop_trace(mid)[0] > 1e-6:
            hi = mid
        else:
            lo = mid
    thr = 0.5 * (lo + hi)
    print("C5 sun threshold by the level maps:", thr, " (a pair alone: 2.1213)")
    assert abs(thr - 1.3754) < 2e-3
    live = settle(whole((5.0, 2.5, 6.0, 3.0), s=1.45), steps_for(900.0))[:4]
    die = settle(whole((5.0, 2.5, 6.0, 3.0), s=1.30), steps_for(900.0))[:4]
    print("C5 at s = 1.45:", live, " at s = 1.30:", die)
    assert np.min(live) > HELD and np.max(np.abs(die)) < HELD
