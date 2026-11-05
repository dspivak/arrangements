"""The bees: two gated pairs chained, a nested closed arrangement through Phi, as the check of the theorem
"a form is a coalgebra, and no false negatives" (book plan P73; `notes/runs/p73-structures-d5eec4fb/RESULTS.md`,
"Piece 3, the theorem"). David, 2026-11-05 (d5eec4fb): "the theorem should be beautiful, categorically elegant ...
go for it". Everything not quoted from him is the agent's.

**The whole.** Pair 1 (the bees) and pair 2 (us) are each ``test_mutual_need.py``'s gated pair, built here as a
*group* ``G = compose_seq(tensor(A, B), coupler_G)`` with an open boundary ``<R^1|R^1>``: it receives its supply
at its one input port and sends A's activity out at its one output port, and returns no load on its input (one-way
feeding). The parent coupler ``<R^2|R^3> -> <R^0|R^0>`` reads the two groups' outputs and the sun and supplies
group 1 with the sun and group 2 with ``kappa`` times group 1's output. ``whole = compose_seq(tensor(G1, G2, S),
parent)``, configuration ``(q_A1, q_B1, q_A2, q_B2, s)``; the hearing in each pair is a fixed loudness ``w``.
Each pair's intake is ``supply * g + leak + w q_B`` with the gate ``g = q_A q_B/(1 + q_A q_B)`` and an optional
ungated ``leak`` (the free lunch).

**The theorem's objects, computed from the whole's step.** The step ``F(q) = q + h xi(q)`` on the parts' box
``prod [0, M_i]`` (tops ``M = (8, 4, 12, 6)``, chosen so that ``F(M) <= M``); the Galois connection ``alpha`` (who
is above threshold ``theta = 1/2``) and ``gamma`` (alive at the top, dead at the threshold); the greatest
self-sustaining level ``nu F`` by Kleene from the top; the shadow ``Sigma(A) = alpha(nu(F /\\ gamma(A)))`` and its
greatest fixed point by descent on subsets. The sun coordinate is a parameter, not a part.

**Predictions, written before the arrangement was built** (``kappa = 1``, ``w = 1/2``, ``h = 0.01``, 600 time units
per settle; the pair's rests from its closed form, ``gated_roots``):
  E0  ``Phi``, ``run_closed`` and the ``fori_loop`` agree exactly over fifty steps.
  E1  (sun ``s = 4``) the step is monotone: every entry of its Jacobian in the parts' coordinates is nonnegative at
      40 random states of the box; ``F(M) <= M``. Three forms are reached: from an alive start, all alive at
      ``(4.9274, 2.4637, 6.2500, 3.1250)`` (pair 2 fed at ``4.9274``); from pair 2 low, ``(4.9274, 2.4637, 0, 0)``;
      from everyone low, dead. ``nu F`` equals the all-alive rest to ``1e-6``; each form's envelope is a
      post-fixed point and lies below ``nu F``; the shadow's greatest fixed point is all four parts and every form's
      support is a fixed point of ``Sigma``; ``Sigma({}) = {}``.
  E2  (sun off) ``nu F = 0`` and the shadow's greatest fixed point is empty; from the all-alive rest the whole
      dies, every part below ``1e-3``: death forced by the cut.
  E3  (sun off, a free lunch ``leak = 1`` to pair 2's A) ``nu F = (0, 0, 4/3, 2/3)``, the shadow's greatest fixed
      point ``{A2, B2}`` with ``Sigma({}) = {A2}``; from a dead start the whole settles at that rest (arising).
      [Refuted as written on the first run: ``Sigma({}) = {}``, since the shadow caps the dead at their thresholds
      and so never raises a dead part; arising is the least fixed point from the dead state, and the prediction
      replacing it, written before the rerun: ``mu F = nu F`` here, ``alpha(mu F) = {A2, B2}``, while at E1
      ``mu F = 0`` with ``nu F`` alive, the bistability of the whole.]
  E4  (sun ``s = 1.5``, below the pair's threshold ``2.1213``) ``nu F = 0``: the joint object sees the loop gain
      and calls everyone dead, while piece 2's Boolean trace with parts settled alone at their neighbours' tops
      calls everyone alive (the first version's false positive, by hand); from the alive start the whole dies.
  E5  (nesting) at ``s = 4`` the groups' level maps composed along the chain, group 1's output level at the sun
      then group 2's at ``kappa`` times it, equal ``nu F`` of the flat whole to ``1e-6`` (Bekic).

Everything here is one closed arrangement built by nesting and stepped by the dynamics functor (E0 licenses
``settle``); the theorem's objects are calculations on its step. Reasonable or weird, by scale: RESULTS.md.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact, zero_form
from arrangements.coalgebra import configuration
from arrangements.rvect import constant, euclidean
from arrangements.wiring import compose_seq, tensor_arrangements, trivial_parameter

from tests.chain import flat_state
from tests.test_mutual_need import gated_roots
from tests.test_rendezvous import run_closed, run_phi

MOB = 0.01
W = 0.5
KAPPA = 1.0
SUN = 4.0
THETA = 0.5
TOPS = np.array([8.0, 4.0, 12.0, 6.0])
HELD = 1e-3
EMPTY = jnp.zeros(0)
PARTS = ["A1", "B1", "A2", "B2"]


def part(q0, label):
    return Arrangement(
        P=configuration(euclidean(1, MOB), init=jnp.array([float(q0)])),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: EMPTY,
        omega=exact(lambda q, m_out, n_in: -0.5 * q[0] ** 2),
        label=label,
    )


def sun(s):
    return Arrangement(
        P=configuration(constant(jnp.zeros((1, 1))), init=jnp.array([float(s)])),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: EMPTY,
        omega=zero_form,
        label="S",
    )


def gate(q_A, q_B):
    u = q_A * q_B
    return u / (1.0 + u)


def group(q_A, q_B, w=W, leak=0.0, label="pair"):
    """A pair as a group with one input port (its supply) and one output port (A's activity):
    ``compose_seq(tensor(A, B), coupler_G)``, ``<R^0|R^0> -> <R^1|R^1>``. The coupler returns the pair's wiring
    as covectors on the parts' outputs, reads its supply from the input port, returns no load there."""

    def coupler_omega(q, e, n_in):
        q_A, q_B = e[0], e[1]
        s_in = n_in[0]
        F = jnp.array([s_in * gate(q_A, q_B) + leak + w * q_B, w * q_A])
        return (EMPTY, F, jnp.zeros(1))

    coupler = Arrangement(
        P=trivial_parameter(),
        in_dim_M=0, out_dim_M=2, in_dim_N=1, out_dim_N=1,
        out_f=lambda q, e: e[:1],
        in_f=lambda q, e, n_in: EMPTY,
        omega=coupler_omega,
        label=f"{label}_wiring",
    )
    return compose_seq(tensor_arrangements([part(q_A, f"A_{label}"), part(q_B, f"B_{label}")]), coupler)


def parent(kappa=KAPPA):
    """``<R^2|R^3> -> <R^0|R^0>``: reads (out1, out2, s), supplies group 1 with s and group 2 with kappa out1."""
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=2, out_dim_M=3, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: EMPTY,
        in_f=lambda q, m_out, n_in: jnp.array([m_out[2], kappa * m_out[0]]),
        omega=zero_form,
        label="chain",
    )


def whole(q=(4.9, 2.45, 4.9, 2.45), s=SUN, leak2=0.0, kappa=KAPPA):
    boxes = [group(q[0], q[1], label="bees"), group(q[2], q[3], leak=leak2, label="us"), sun(s)]
    return compose_seq(tensor_arrangements(boxes), parent(kappa))


def step_fn(arr):
    """One step of the closed whole on its configuration, ``q + mob xi(q)``, as a function of the configuration."""
    P = arr.P

    def F(qcfg):
        st = P.init  # the state is the configuration here (configuration coalgebras)
        xi_q, _, _ = arr.omega(jnp.asarray(qcfg, float), EMPTY, EMPTY)
        return np.asarray(P.position(P.step(st_from(P, qcfg), xi_q)))
    return F


def st_from(P, qcfg):
    """A state with the given configuration (the configuration coalgebra's state is its configuration)."""
    return jnp.asarray(qcfg, float)


def settle(arr, steps, init=None):
    P, omega = arr.P, arr.omega
    s0 = P.init if init is None else jnp.asarray(init, float)

    def step(_i, st):
        xi_q, _, _ = omega(P.position(st), EMPTY, EMPTY)
        return P.step(st, xi_q)

    return np.asarray(flat_state(jax.lax.fori_loop(0, steps, step, s0)))


def steps_for(T):
    return int(round(T / MOB))


# ---------------------------------------------------------------------------
# The theorem's objects, from the step.
# ---------------------------------------------------------------------------


def parts_step(arr, s):
    """The step on the four parts' coordinates with the sun fixed at ``s``: a monotone map of the box."""
    omega = arr.omega

    @jax.jit
    def F_jnp(x):
        q = jnp.concatenate([x, jnp.array([s])])
        xi_q, _, _ = omega(q, EMPTY, EMPTY)
        return (q + MOB * xi_q)[:4]

    def F(x):
        return np.asarray(F_jnp(jnp.asarray(x, float)))
    return F


def gfp_below(F, cap, tol=1e-10, max_iter=400000):
    """Kleene from the cap: the greatest fixed point of ``F /\\ cap`` on the box."""
    x = np.asarray(cap, float)
    for _ in range(max_iter):
        y = np.minimum(F(x), cap)
        if np.max(np.abs(y - x)) < tol:
            return y
        x = y
    return x


def lfp_above(F, floor, tol=1e-10, max_iter=400000):
    """Kleene from the floor: the least fixed point of ``F`` above ``floor`` (``floor <= F(floor)``)."""
    x = np.asarray(floor, float)
    for _ in range(max_iter):
        y = np.maximum(F(x), floor)
        if np.max(np.abs(y - x)) < tol:
            return y
        x = y
    return x


def alpha(x):
    return frozenset(i for i in range(4) if x[i] > THETA)


def gamma(A):
    return np.array([TOPS[i] if i in A else THETA for i in range(4)])


def shadow(F):
    def Sigma(A):
        return alpha(gfp_below(F, gamma(A)))
    return Sigma


def gfp_sets(Sigma):
    A = frozenset(range(4))
    while True:
        B = Sigma(A)
        if B == A:
            return A
        A = B


def names(A):
    return "{" + ", ".join(PARTS[i] for i in sorted(A)) + "}"


# ---------------------------------------------------------------------------
# E0
# ---------------------------------------------------------------------------


def test_E0_the_runs_are_phis():
    b = whole()
    print("label:", b.label)
    by_phi = run_phi(b, 50, EMPTY)
    by_closed = run_closed(b, 50, lambda t: EMPTY)
    gap = float(np.max(np.abs(by_phi - by_closed)))
    print("E0 Phi vs run_closed over fifty steps:", gap)
    # Predicted exact; refuted as written: 8.9e-16, one ulp, since the nested composite's pullback sums in a
    # different order in the two run forms. Agreement to machine precision is what licenses ``settle``.
    assert gap < 1e-14
    np.testing.assert_allclose(settle(b, 50), by_closed[-1], atol=1e-14)


# ---------------------------------------------------------------------------
# E1: the class, the forms, the object, the shadow.
# ---------------------------------------------------------------------------


def test_E1_sun_on():
    b = whole()
    F = parts_step(b, SUN)
    # the class: monotone step on the box, and the box is kept
    rng = np.random.default_rng(20261105)

    def step5(y):
        q = jnp.concatenate([y, jnp.array([SUN])])
        return (q + MOB * b.omega(q, EMPTY, EMPTY)[0])[:4]

    worst = 0.0
    for _ in range(40):
        x = rng.uniform(0.0, 1.0, 4) * TOPS
        J = np.asarray(jax.jacobian(step5)(jnp.asarray(x)))
        worst = min(worst, float(J.min()))
    print("E1 smallest Jacobian entry of the step over 40 states:", worst)
    assert worst >= 0.0
    assert np.all(F(TOPS) <= TOPS + 1e-12)

    # the forms
    qA1, qB1 = gated_roots(SUN)[1], W * gated_roots(SUN)[1]
    qA2 = gated_roots(KAPPA * qA1)[1]
    predicted_AA = np.array([qA1, qB1, qA2, W * qA2])
    AA = settle(whole((4.9, 2.45, 4.9, 2.45)), steps_for(600.0))[:4]
    AD = settle(whole((4.9, 2.45, 0.1, 0.1)), steps_for(600.0))[:4]
    DD = settle(whole((0.1, 0.1, 0.1, 0.1)), steps_for(600.0))[:4]
    print("E1 forms: AA", AA, " AD", AD, " DD", DD)
    np.testing.assert_allclose(AA, predicted_AA, atol=1e-5)
    np.testing.assert_allclose(AD[:2], predicted_AA[:2], atol=1e-5)
    assert np.max(np.abs(AD[2:])) < HELD and np.max(np.abs(DD)) < HELD

    # the object and the theorem
    nuF = gfp_below(F, TOPS)
    print("E1 nu F:", nuF)
    np.testing.assert_allclose(nuF, predicted_AA, atol=1e-5)
    Sigma = shadow(F)
    top = gfp_sets(Sigma)
    muF = lfp_above(F, np.zeros(4))
    print("E1 shadow gfp:", names(top), " Sigma({}) =", names(Sigma(frozenset())), " mu F:", muF)
    assert top == frozenset(range(4)) and Sigma(frozenset()) == frozenset()
    assert np.max(muF) < 1e-6                        # bistable: the floor stays dead while the ceiling is alive
    for name, H in (("AA", AA), ("AD", AD), ("DD", DD)):
        env = np.maximum(H, 0.0)
        assert np.all(env <= F(env) + 1e-6)          # a form (a rest) is a coalgebra
        assert np.all(env <= nuF + 1e-6)             # below the greatest self-sustaining level
        A = alpha(env)
        print(f"E1 form {name}: support {names(A)}, Sigma(support) = {names(Sigma(A))}")
        assert A <= Sigma(A) and A <= top


# ---------------------------------------------------------------------------
# E2 and E3: the cut, and the free lunch.
# ---------------------------------------------------------------------------


def test_E2_sun_off_death_forced():
    b = whole(s=0.0)
    F = parts_step(b, 0.0)
    nuF = gfp_below(F, TOPS)
    Sigma = shadow(F)
    print("E2 nu F:", nuF, " shadow gfp:", names(gfp_sets(Sigma)))
    assert np.max(nuF) < 1e-6 and gfp_sets(Sigma) == frozenset()
    qA1 = gated_roots(SUN)[1]
    qA2 = gated_roots(KAPPA * qA1)[1]
    end = settle(whole((qA1, W * qA1, qA2, W * qA2), s=0.0), steps_for(600.0))[:4]
    print("E2 from the alive rest, sun off:", end)
    assert np.max(np.abs(end)) < HELD


def test_E3_free_lunch_arises():
    b = whole(s=0.0, leak2=1.0)
    F = parts_step(b, 0.0)
    nuF = gfp_below(F, TOPS)
    Sigma = shadow(F)
    muF = lfp_above(F, np.zeros(4))
    print("E3 nu F:", nuF, " mu F:", muF, " shadow gfp:", names(gfp_sets(Sigma)), " Sigma({}) =", names(Sigma(frozenset())))
    np.testing.assert_allclose(nuF, [0.0, 0.0, 1.0 / (1 - W ** 2), W / (1 - W ** 2)], atol=1e-6)
    assert gfp_sets(Sigma) == frozenset({2, 3})
    # Predicted ``Sigma({}) = {A2}``; refuted as written: the shadow caps the dead at their thresholds, so it is
    # deflationary, ``Sigma(A) <= A``, and ``Sigma({}) = {}`` always. Arising is the least fixed point from the
    # dead state: ``mu F = nu F`` here (the free lunch raises the floor to the ceiling), ``alpha(mu F) = {A2, B2}``.
    assert Sigma(frozenset()) == frozenset()
    np.testing.assert_allclose(muF, nuF, atol=1e-6)
    assert alpha(muF) == frozenset({2, 3})
    end = settle(whole((0.0, 0.0, 0.0, 0.0), s=0.0, leak2=1.0), steps_for(600.0))[:4]
    print("E3 from the dead state:", end)
    np.testing.assert_allclose(end, nuF, atol=1e-6)


# ---------------------------------------------------------------------------
# E4: the joint object sees the loop gain; the Boolean trace of parts settled alone does not.
# ---------------------------------------------------------------------------


def test_E4_below_threshold_supply():
    s = 1.5
    b = whole(s=s)
    F = parts_step(b, s)
    nuF = gfp_below(F, TOPS)
    print("E4 nu F at s = 1.5:", nuF)
    assert np.max(nuF) < 1e-6

    # the first version's shadow: each part settled alone with the others frozen at gamma(A)
    def settled_alone(i, x):
        y = x.copy()
        v = TOPS[i]
        for _ in range(200000):
            y[i] = v
            v2 = min(F(y)[i], TOPS[i])
            if abs(v2 - v) < 1e-10:
                return v2
            v = v2
        return v

    def Sigma_alone(A):
        g = gamma(A)
        return frozenset(i for i in range(4) if settled_alone(i, g) > THETA)

    print("E4 parts-alone shadow gfp:", names(gfp_sets(Sigma_alone)))
    assert gfp_sets(Sigma_alone) == frozenset(range(4))
    end = settle(whole((4.9, 2.45, 4.9, 2.45), s=s), steps_for(600.0))[:4]
    print("E4 from the alive start at s = 1.5:", end)
    assert np.max(np.abs(end)) < HELD


# ---------------------------------------------------------------------------
# E5: nesting, the groups' level maps composed along the chain equal the flat object.
# ---------------------------------------------------------------------------


def test_E5_level_maps_compose():
    def pair_level(s_in, leak=0.0):
        """A group's level map at one input level: its greatest self-sustaining output, by Kleene from the top
        on the group alone (the group closed by a constant supply)."""
        def Fp(x):
            q_A, q_B = x
            return np.array([q_A + MOB * (s_in * gate(q_A, q_B) + leak + W * q_B - q_A),
                             q_B + MOB * (W * q_A - q_B)])
        return gfp_below(Fp, np.array([TOPS[0], TOPS[1]]))

    out1 = pair_level(SUN)
    out2 = pair_level(KAPPA * out1[0])
    staged = np.concatenate([out1, out2])
    F = parts_step(whole(), SUN)
    flat = gfp_below(F, TOPS)
    print("E5 staged:", staged, " flat:", flat)
    np.testing.assert_allclose(staged, flat, atol=1e-6)
