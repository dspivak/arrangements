"""Word of mouth: does a collective recover when the leaver carried the recruiting channel?

The book plan's §5.9 item 38 (AGENTDOCPLAN.md in the book repository; the construction
and the four predictions Claude's, 2026-09-28, *proposed*, from David's words: "because
other members of the town rely on those people, they wear down … there is stress, and
people move away. word of mouth can make them hard to replace"; "bad juju builds up and
that can knock things out of their equilibrium, so that a lot of effort can (if the
accounting backs it up) aim at the problem joint"). P30 stage (d).

**The predictions, fixed before any construction code in
``notes/WORD-OF-MOUTH-PREDICTIONS-2026-09-29.md``**, with the run's own R0-R5 beside them.
The plan's four: (1) no exact return: the settled outgoing signal keeps a positive deficit;
(2) successive failures compound smoothly, with no jump; (3) mutual worsening is ordered by
the wiring; (4) the deficit recovers when the joint's cost reaches a part that can act on
it and not when it reaches no actor, and that actor's forcing is non-exact.

**The construction** is the turnover run's, ``tests/test_turnover.py`` (imported as ``tv``):
a spatial-attention parent with cost ``(1/2)(nu - h~)^2`` on the weighted signal it sends
onward, three contributors at the positions that make the pre-failure state a fixed point,
one contributor's health driven to zero by its own cost, a spare off to the side at ``d``.
One change, the plan's: **the forcing that places a spare is multiplied by the word**

    g(m) = sum_j rho_j w_j / sum_j rho_j WGT,

``w_j`` the weight in contributor ``j``'s message, ``rho_j`` its share of the recruiting
channel (a wiring datum; a spare's is ``0`` unless it inherits the leaver's share once
recruited). In the reporting mode ``w_j = WGT`` whatever the health, so ``g = 1``; in the
silent mode ``w_j = c_j WGT``, so a leaver's word dies with it. The parent's arrangement is
wrapped: its one-form is the operad's ``-dk`` with the spare placements' components scaled
by ``g(m_out)``. That is no longer ``-dk``, so the wrapped parent is inside ``arr`` and
outside the attention operad (test 5 measures its ``d omega``).

For predictions 3 and 4 two more terms enter the wrapper's one-form, both on the message
space and both pulled back to the children's configurations by ``compose_seq``: a **supply
cost** ``(lam_h/2)(c_j - t_j)^2`` on part ``j`` with target
``t_j = (1 - a_j) + a_j c_F + act_j``, a share ``a_j`` of its supply arriving through the
failing part ``F`` (read off ``F``'s message) and the rest from outside; and a **doodad**
(item 14 route (b)), a part whose alarm ``a`` is its configuration, sent as the weight of
a message with zero signal so the parent's cost cannot see it, driven toward
``gain_r * delta`` by a cost whose reading ``delta`` is insulated (the reading enters the
one-form as a value, not as a variable it is differentiated in: the doodad reads and
nothing returns), and acting at the joint as ``act_j = gain * a``, likewise insulated.
Cutting the reading wire (``delta = 0``) is the control. Because the insulation is written
into the one-form's components and not hidden from differentiation, the Jacobian of the
composite forcing is the true one and its antisymmetric part is what test 4 measures.

Every docstring below says which of three things its test does (STATUS.md §3.7):
reproduce a closed form derived in the docstring, check a discrete analogue, or correct a
claim made on paper. The numbers: ``notes/WORD-OF-MOUTH-TEST.md``.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact
from arrangements.attention import (
    message, signal, spatial_arrangement, spatial_config, split_messages, weight,
)
from arrangements.coalgebra import configuration
from arrangements.rvect import constant, euclidean
from arrangements.wiring import compose_seq, tensor_arrangements

from tests import test_turnover as tv
from tests.chain import flat_state
from tests.test_rendezvous import run_closed, run_phi

DH, DP, MDIM, EDIM, EXT, EMPTY = tv.DH, tv.DP, tv.MDIM, tv.EDIM, tv.EXT, tv.EMPTY
TAU, SIG, WGT, NU, LAM, MU_C, MU_P = tv.TAU, tv.SIG, tv.WGT, tv.NU, tv.LAM, tv.MU_C, tv.MU_P
D_SPARE, FAIL = tv.D_SPARE, tv.FAIL
BUDGET = tv.STEPS          # 8000 steps, item 13's budget
TOL = 1e-3                 # the collective has returned when h~ > nu - TOL
LAM_H = 50.0               # the stiffness of a supply cost
LAM_A = 50.0               # the stiffness of the doodad's alarm toward its reading
MU_A = 0.001               # the alarm's mobility: LAM_A * MU_A = 1/20 per step, see test 4
A_SHARE = 0.6              # the share of part 0's supply that runs through the failing part
CHUNK = 20                 # steps per jitted chunk when reading the time to return
CAP = 400000               # the largest budget any return is waited for


# ---------------------------------------------------------------------------
# The doodad, the word, and the wrapped parent.
# ---------------------------------------------------------------------------


def alarm(q):
    """The doodad's message: zero signal, its alarm as the weight. The parent's cost on
    the weighted *signal* cannot see it (item 13: that cost cannot see weight)."""
    return message(0.0 * q[0], q[0])


def doodad_part(label="D", a0=0.0, mu=MU_A):
    """A part whose configuration is its alarm ``a``, carrying no cost of its own: what
    drives it is the wrapper's reading cost. Interface a contributor's, ``I -> B``."""
    P = configuration(euclidean(1, mu), init=jnp.array([float(a0)]))
    arr = Arrangement(
        P=P, in_dim_M=0, out_dim_M=0, in_dim_N=EDIM, out_dim_N=MDIM,
        out_f=lambda q, m_out: alarm(q),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: 0.0 * q[0]),
        label=label)
    return tv.Part(arr, tv.Spec("alarm", a0, 0.0, mu, 0.0), label)


def healths_from(ms):
    """``c_j = signal(m_j) / SIG`` for every slot: exact in the silent mode, where a
    contributor sends ``(c SIG, c WGT)``; a doodad slot reads ``0``."""
    return [signal(m)[0] / SIG for m in ms]


def build(specs, rho, spare_slots=(), supply=None, doodad=None, doodad_pos=0.0):
    """The composite ``compose_seq(tensor(children), wrapped parent)``.

    ``specs`` are ``tv.Spec``s for the contributors in slot order; ``rho`` their shares of
    the word (one number per contributor); ``spare_slots`` the slots whose placement
    forcing the word scales. ``supply`` is ``{j: (a_j, F)}``: part ``j``'s target is
    ``(1 - a_j) + a_j c_F``. ``doodad`` is ``dict(reads=..., joint=j, gain=..., gain_r=...)``
    with ``reads`` one of ``"joint"`` (``delta = 1 - c_j``), ``"collective"``
    (``delta = nu - h~``) or ``None`` (the reading wire cut); it adds one more child, in the
    last slot, at ``doodad_pos``. Returns ``(parts, a, b, n)``.
    """
    parts = [tv.contributor(s, f"C{i}") for i, s in enumerate(specs)]
    positions = [s.pos for s in specs]
    rho = list(rho)
    if doodad is not None:
        parts.append(doodad_part())
        positions.append(doodad_pos)
        rho.append(0.0)
    n = len(parts)
    rho = np.asarray(rho, float)
    assert rho.sum() > 0.0
    mobilities = jnp.array([MU_P] * n + [0.0] * (n * DH * DH))
    a = spatial_arrangement(
        n, DH, DP, TAU, cost=tv.total_cost(n, NU),
        parameter=configuration(constant(jnp.diag(mobilities)),
                                init=spatial_config(positions, [1.0] * n)),
        label=f"parent({n})")
    base = a.arr()
    spare_slots = list(spare_slots)
    d_slot = n - 1 if doodad is not None else None
    supply = supply or {}

    def word(m_out):
        ms = split_messages(m_out, n, DH)
        return sum(float(rho[j]) * weight(ms[j]) for j in range(n)) / float(rho.sum() * WGT)

    def extra_cost(m_out, act, delta):
        """The supply and reading costs on the message space, with the doodad's action
        ``act`` and its reading ``delta`` entering as given values (the insulators)."""
        ms = split_messages(m_out, n, DH)
        c = healths_from(ms)
        k = 0.0
        if doodad is not None:
            k = k + 0.5 * LAM_A * (weight(ms[d_slot]) - doodad["gain_r"] * delta) ** 2
        for j, (a_j, F) in supply.items():
            here = act if (doodad is not None and doodad["joint"] == j) else 0.0
            t_j = (1.0 - a_j) + a_j * c[F] + here
            k = k + 0.5 * LAM_H * (c[j] - t_j) ** 2
        return k

    def readings(q, m_out):
        """The doodad's action and reading, as values computed from the ports."""
        ms = split_messages(m_out, n, DH)
        c = healths_from(ms)
        act = doodad["gain"] * weight(ms[d_slot])
        if doodad["reads"] == "joint":
            delta = 1.0 - c[doodad["joint"]]
        elif doodad["reads"] == "collective":
            delta = NU - signal(tv.outgoing(n, q, m_out))[0]
        else:
            delta = 0.0 * act
        return act, delta

    def omega(q, m_out, n_in):
        xi_q, xi_m, xi_n = base.omega(q, m_out, n_in)
        if spare_slots:
            scale = jnp.ones(q.shape[0]).at[jnp.array(spare_slots)].set(word(m_out))
            xi_q = xi_q * scale
        if supply or doodad is not None:
            act, delta = readings(q, m_out) if doodad is not None else (0.0, 0.0)
            xi_m = xi_m - jax.grad(lambda m: extra_cost(m, act, delta))(m_out)
        return xi_q, xi_m, xi_n

    parent = replace(base, omega=omega)
    b = compose_seq(tensor_arrangements([p.arr for p in parts]), parent)
    return parts, a, b, n


# ---------------------------------------------------------------------------
# Running, and reading a state.
# ---------------------------------------------------------------------------


def chunk_runner(b, chunk=CHUNK):
    """One jitted function advancing the composite's state by ``chunk`` steps."""
    P, omega = b.P, b.omega

    @jax.jit
    def run(state):
        def step(_i, s):
            xi_q, _, _ = omega(P.position(s), EMPTY, EXT)
            return P.step(s, xi_q)
        return jax.lax.fori_loop(0, chunk, step, state)

    return run


def sent(parts, a, q):
    return float(signal(tv.sent_onward(parts, a, q))[0])


def time_to_return(parts, a, b, cap, chunk=CHUNK, tol=TOL):
    """``(steps, q)``: the first multiple of ``chunk`` at which the collective sends more
    than ``nu - tol``, or ``(None, q at cap)`` if it has not by ``cap`` steps."""
    run = chunk_runner(b, chunk)
    state = b.P.init
    for k in range(1, cap // chunk + 1):
        state = run(state)
        q = np.asarray(flat_state(state))
        if sent(parts, a, q) > NU - tol:
            return k * chunk, q
    return None, q


def at_budget(parts, a, b, steps=BUDGET):
    q = tv.settle(b, steps)
    return sent(parts, a, q), q


def anti_part(b, q):
    """The antisymmetric part of the composite forcing's Jacobian at ``q``."""
    J = np.asarray(jax.jacobian(lambda x: b.omega(x, EMPTY, EXT)[0])(jnp.asarray(q, float)))
    return J.T - J


def shares(n_inc, rho_F, fail=FAIL):
    """The three incumbents' shares of the word: the leaver ``rho_F``, the rest equal."""
    rho = [(1.0 - rho_F) / (n_inc - 1)] * n_inc
    rho[fail] = rho_F
    return rho


def after_failure(rho_F=1.0 / 3.0, mode="silent", d=D_SPARE, with_spare=True, leaver="failing"):
    """Item 13's ``after_failure`` with the word: three incumbents, ``FAIL`` failing (or,
    ``leaver="dead"``, already at zero health and frozen), a spare at ``d`` with no word."""
    extra = tv.kappa_at(d) if with_spare else 0.0
    p = tv.equilibrium_position(3, extra)
    third = tv.failing(p, mode, LAM) if leaver == "failing" else tv.dead(p, mode)
    specs = [tv.healthy(p, mode), tv.healthy(-p, mode), third]
    rho = shares(3, rho_F)
    spare_slots = ()
    if with_spare:
        specs.append(tv.healthy(d, mode))
        rho.append(0.0)
        spare_slots = (3,)
    return build(specs, rho, spare_slots)


# ---------------------------------------------------------------------------
# Test 0: the runs are Phi's, and with the word intact this is item 13 (R0).
# ---------------------------------------------------------------------------


def test_0_the_runs_are_phis_and_the_reporting_mode_is_item_13():
    """LICENSES the run forms and CHECKS R0. ``Phi`` of the wrapped composite,
    ``run_closed`` and the chunked ``fori_loop`` agree to one ulp over fifty steps. In the
    reporting mode every incumbent keeps claiming ``WGT``, so ``g = 1`` throughout and the
    trajectory equals item 13's to ``1e-12`` over 2000 steps: the construction reduces to
    the turnover run when the word is intact. So does the silent mode when the leaver
    carried no word (``rho_F = 0``)."""
    parts, a, b, n = after_failure(mode="reporting")
    by_phi = run_phi(b, 50, EXT)
    by_closed = run_closed(b, 50, lambda t: EXT)
    assert float(np.max(np.abs(by_phi - by_closed))) < 1e-15
    run = chunk_runner(b, 50)
    np.testing.assert_allclose(np.asarray(flat_state(run(b.P.init))), by_closed[-1], atol=1e-15)

    _, _, b13 = tv.after_failure(mode="reporting")
    ours = run_closed(b, 2000, lambda t: EXT)
    theirs = run_closed(b13, 2000, lambda t: EXT)
    assert float(np.max(np.abs(ours - theirs))) < 1e-12
    parts, a, b, n = after_failure(rho_F=0.0, mode="silent")
    _, _, b13 = tv.after_failure(mode="silent")
    assert float(np.max(np.abs(run_closed(b, 2000, lambda t: EXT)
                               - run_closed(b13, 2000, lambda t: EXT)))) < 1e-12


# ---------------------------------------------------------------------------
# Test 1: prediction 1 and R1 -- the return is exact and slower, unless the word is gone.
# ---------------------------------------------------------------------------


RHO_F = (0.0, 1.0 / 3.0, 2.0 / 3.0, 0.9)


def test_1_prediction_1_the_return_is_exact_but_slower_and_never_when_the_word_is_gone():
    """CORRECTS prediction 1 and REPRODUCES the closed form for the word-gone state.

    Silent mode, spare at ``d = 2``. With the leaver's share ``rho_F`` the word after the
    failure is ``g = 1 - rho_F`` (the leaver's health, and so its weight, goes to ``0``).
    For every ``g > 0`` run the collective returns to ``nu`` within the cap, so the
    deficit's floor is zero and prediction 1 is refuted; the time to return grows as the
    word shrinks (R1: reach is a rate, item 13). With the leaver dead and frozen at zero
    health and the whole word its own (``rho_F = 1``), ``g = 0``: the spare's placement
    forcing is zero, the spare never moves, and the collective settles at item 13's
    no-spare value for a dead part plus the spare's own kernel from where it sits,
    ``2 + kappa(d) = 2.082085``: prediction 1 holds only where the leaver was the
    whole channel, and then as a total loss, not a floor. And with the leaver *failing*
    rather than dead, ``g`` is not zero: the parent's deficit props the leaver's health at
    ``(nu - h~) kappa_F / lam`` (item 13), and that whisper of health is a whisper of word,
    so the spare creeps.
    """
    times = {}
    for rho_F in RHO_F:
        parts, a, b, n = after_failure(rho_F=rho_F)
        steps, q = time_to_return(parts, a, b, CAP)
        times[rho_F] = steps
        assert steps is not None, (rho_F, sent(parts, a, q))
        np.testing.assert_allclose(sent(parts, a, tv.settle(b, steps + 20000)), NU, atol=TOL)
    t = [times[r] for r in RHO_F]
    assert t[0] < t[1] < t[2] < t[3]

    # the whole word the leaver's, and the leaver dead: nothing moves the spare
    parts, a, b, n = after_failure(rho_F=1.0, leaver="dead")
    steps, q = time_to_return(parts, a, b, 4 * BUDGET)
    assert steps is None
    h, q = at_budget(parts, a, b, 2 * BUDGET)
    np.testing.assert_allclose(h, 2.0 + tv.kappa_at(D_SPARE), atol=1e-6)   # survivors home, spare at d
    _, pos, _ = tv.split_composite(q, n)
    np.testing.assert_allclose(pos[3], D_SPARE, atol=0.0)
    np.testing.assert_allclose(tv.kappas(q, n)[:2], 1.0, atol=1e-6)  # survivors pulled home

    # the leaver failing instead: a whisper of word, and the spare creeps
    parts, a, b, n = after_failure(rho_F=1.0, leaver="failing")
    steps_w, q = time_to_return(parts, a, b, CAP, chunk=1000)
    c, pos, _ = tv.split_composite(q, n)
    times["1.0 failing"] = steps_w
    print("time to return by rho_F", times, "creeping spare: c_F", float(c[FAIL]),
          "p_s", float(pos[3]), "h~", sent(parts, a, q))
    assert pos[3] < D_SPARE                                          # it moved
    print("h~ at 8000 steps by rho_F",
          {r: at_budget(*after_failure(rho_F=r)[:3])[0] for r in RHO_F})


# ---------------------------------------------------------------------------
# Test 2: prediction 2 and R2 -- compounding is the channel running out.
# ---------------------------------------------------------------------------


def spec_for(status, pos):
    if status == "healthy":
        return tv.healthy(pos, "silent")
    if status == "failing":
        return tv.failing(pos, "silent", LAM)
    return tv.dead(pos, "silent")


def sequence(inherit, cap=CAP):
    """Three failures in turn, each followed by a fresh spare at ``d``, each stage started
    from the previous one's settled state. A dead incumbent keeps its share of the word
    (silent: its weight is ``0``, so the share is dead weight in the denominator); with
    ``inherit`` the recruit takes the leaver's share once in, and the leaver's goes to
    zero. Returns ``[(steps to return, h~ at the budget, word at the start)]``."""
    p = tv.equilibrium_position(3, tv.kappa_at(D_SPARE))
    positions, status, rho = [p, -p, p], ["healthy"] * 3, [1.0 / 3.0] * 3
    out = []
    for k in range(3):
        status[k] = "failing"
        specs = [spec_for(status[i], positions[i]) for i in range(len(positions))]
        specs.append(tv.healthy(D_SPARE, "silent"))
        rho_now = rho + [0.0]
        parts, a, b, n = build(specs, rho_now, spare_slots=(len(specs) - 1,))
        g0 = sum(r for r, s in zip(rho, status) if s == "healthy") / sum(rho_now)
        steps, q = time_to_return(parts, a, b, cap)
        h8, _ = at_budget(parts, a, b)
        out.append((steps, h8, g0))
        if steps is None:
            break
        q, _ = tv.drifted(b, steps + 20000)
        _, pos, _ = tv.split_composite(q, n)
        positions = [float(x) for x in pos]
        status[k] = "dead"
        status.append("healthy")
        if inherit:
            rho.append(rho[k])
            rho[k] = 0.0
        else:
            rho.append(0.0)
    return out


def test_2_prediction_2_successive_failures_compound_as_the_word_runs_out():
    """CHECKS the discrete analogue of R2 and so reads prediction 2.

    Three failures in turn, each followed by a spare, the dead keeping their (now zero)
    weight and the recruits carrying no word: the word is ``2/3, 1/3, 0`` and the times to
    return increase, the last not within the cap, so at any fixed budget the deficits are
    monotone and the last step is a jump to no recovery at all (up to the whisper of test
    1). The jump is the channel running out -- the last carrier's share was the last of the
    word -- and not a threshold reached by accumulation: with ``inherit`` (a recruit takes
    the leaver's share once in) the word is ``2/3`` at every failure and the three times
    are within one chunk of each other.
    """
    plain = sequence(False)
    inh = sequence(True, cap=100000)
    print("no inherit (steps, h~ at 8000, word)", plain)
    print("inherit    (steps, h~ at 8000, word)", inh)
    np.testing.assert_allclose([r[2] for r in plain], [2.0 / 3.0, 1.0 / 3.0, 0.0], atol=1e-12)
    np.testing.assert_allclose([r[2] for r in inh], [2.0 / 3.0] * 3, atol=1e-12)
    t = [r[0] for r in plain]
    assert t[0] is not None and t[1] is not None
    assert t[0] < t[1]
    assert t[2] is None or t[2] > 3 * t[1]                            # the whisper, if any
    d8 = [NU - r[1] for r in plain]
    assert d8[0] <= d8[1] + 1e-9 <= d8[2] + 1e-9                      # monotone at the budget
    ti = [r[0] for r in inh]
    assert None not in ti
    assert max(ti) - min(ti) <= 0.1 * max(ti)


# ---------------------------------------------------------------------------
# Test 3: prediction 3 and R3 -- worsening ordered by the wiring, and what props it.
# ---------------------------------------------------------------------------


def fed(a_share=A_SHARE, with_spare=False, d=D_SPARE, doodad=None, rho_F=1.0 / 3.0):
    """Part 0 fed through the failing part ``FAIL`` by a share ``a_share``, part 1 fed
    from outside; both with mobile health and no cost of their own (their integrity is what
    arrives), so the supply cost alone, and the parent's propping, set their health."""
    extra = tv.kappa_at(d) if with_spare else 0.0
    p = tv.equilibrium_position(3, extra)
    mobile = lambda pos: tv.Spec("silent", 1.0, 1.0, MU_C, pos, 0.0)
    specs = [mobile(p), mobile(-p), tv.failing(p, "silent", LAM)]
    rho = shares(3, rho_F)
    spare_slots = ()
    if with_spare:
        specs.append(tv.healthy(d, "silent"))
        rho.append(0.0)
        spare_slots = (3,)
    return build(specs, rho, spare_slots, supply={0: (a_share, FAIL), 1: (0.0, FAIL)},
                 doodad=doodad)


def test_3_prediction_3_the_starved_part_falls_by_its_wiring_and_is_propped_by_the_deficit():
    """REPRODUCES the closed forms derived here, and so tests prediction 3 and R3.

    No spare. After ``F`` dies, part 0's target is ``1 - a = 0.4`` and part 1's stays ``1``.
    The parent's deficit ``nu - h~`` props every health by ``(nu - h~) kappa_j SIG / lam_h``
    (in the silent mode the cost reads ``c_j SIG kappa_j``), so the settled healths are
    ``c_j = t_j + (nu - h~) kappa_j / lam_h`` with ``kappa_j -> 1`` (everyone pulled home);
    and the dead part is propped by both the deficit and its dependent's need:
    ``c_F = [(nu - h~) kappa_F + lam_h a (c_0 - t_0)] / lam``. Solved together with
    ``h~ = sum_j kappa_j c_j``. Prediction 3 holds -- part 0 falls further than part 1 -- and
    by its wiring, since the propping is the same for both. Part 0 is not pushed out: its
    contribution is low, not wrong, so the force on its placement is inward.
    """
    parts, a, b, n = fed()
    q, drift = tv.drifted(b, BUDGET)
    assert drift < 1e-9
    c, pos, _ = tv.split_composite(q, n)
    kap = tv.kappas(q, n)
    h = sent(parts, a, q)
    np.testing.assert_allclose(kap[:2], 1.0, atol=1e-6)
    t0, t1 = 1.0 - A_SHARE + A_SHARE * c[FAIL], 1.0
    np.testing.assert_allclose(c[0], t0 + (NU - h) * kap[0] / LAM_H, atol=1e-9)
    np.testing.assert_allclose(c[1], t1 + (NU - h) * kap[1] / LAM_H, atol=1e-9)
    np.testing.assert_allclose(
        c[FAIL], ((NU - h) * kap[FAIL] + LAM_H * A_SHARE * (c[0] - t0)) / LAM, atol=1e-9)
    np.testing.assert_allclose(h, float(np.sum(kap * c)), atol=1e-9)
    assert c[0] < c[1] - 0.5                                        # ordered by the wiring
    assert c[0] > 1.0 - A_SHARE                                     # propped above its target
    xi = tv.forcing(b, q)
    assert xi[n + 0] * pos[0] <= 0.0                                # inward, not pushed out
    print("fed, no spare: healths", c, "h~", h, "kappas", kap)


# ---------------------------------------------------------------------------
# Test 4: prediction 4 and R4 -- the doodad at the joint.
# ---------------------------------------------------------------------------


GAINS = (1.0, 4.0, 16.0)


def test_4_prediction_4_the_deficit_closes_only_if_the_joints_cost_reaches_an_actor_and_then_not_quite():
    """REPRODUCES the closed form for the proportional offset and so tests prediction 4
    and R4; MEASURES the doodad's non-exactness.

    Part 0 fed through ``F`` by ``0.6``, a spare at ``d = 2`` with the word ``2/3``, and a
    doodad reading the joint's deficit ``delta = 1 - c_0`` (its alarm settles at
    ``gain_r delta``) and adding ``gain * a`` to part 0's supply. Settled, with the
    propping ``pi = (nu - h~) kappa_0 / lam_h`` and ``c_F ~ 0``:
    ``c_0 = 0.4 + gain gain_r (1 - c_0) + pi``, so the joint's residual is
    ``1 - c_0 = (0.6 - pi) / (1 + gain gain_r)``: it shrinks as ``1/gain`` and never closes
    (R4). The collective's deficit closes iff the survivors and the spare can reach ``nu``,
    ``1 + c_0 + 1 >= nu``, i.e. ``c_0 >= 0.5``: with the reading wire cut ``a = 0``,
    ``c_0 = 0.4 + pi < 0.5`` and the collective keeps a deficit near ``0.1`` however long
    it runs; wired, at every gain here, it returns to ``nu``. The doodad's coupling is
    non-exact: the antisymmetric part of the composite forcing's Jacobian between ``a``
    and ``c_0`` is ``lam_h gain + lam_a gain_r`` wired (the alarm moves the joint and the
    joint moves the alarm, neither by the other's potential) and ``lam_h gain`` with the
    reading cut.

    *Found while running, and the reason the alarm is slow.* Time is discrete here
    (STATUS.md §1), and the joint-and-alarm loop is a linear map per step,
    ``[[1 - mu_h lam_h, mu_h lam_h gain], [-mu_a lam_a gain_r, 1 - mu_a lam_a]]``. With both
    at a half step (``mu lam = 1/2``) its eigenvalues have modulus ``(1/2) sqrt(1 + gain)``,
    so the loop rings itself apart above ``gain = 3``: at ``gain = 4`` the first run
    diverged to ``nan`` within 3000 steps, having passed through ``nu`` on the way. The
    continuous-time loop is stable at every gain (trace negative, determinant positive), so
    this is the step's, and the fix is the alarm moving a twentieth per step
    (``MU_A = 0.001``): then the modulus is below one at every gain run. The fixed points
    do not depend on the mobilities, so the settled values are the same. A fast alarm on a
    stiff joint is a control loop that overshoots, and the book's integrator sees it.
    """
    rows = {}
    for reads, gain in [(None, 4.0)] + [("joint", g) for g in GAINS]:
        dd = dict(reads=reads, joint=0, gain=gain, gain_r=1.0)
        parts, a, b, n = fed(with_spare=True, doodad=dd)
        steps, q = time_to_return(parts, a, b, 200000)
        q, drift = tv.drifted(b, (steps or 200000) + 20000)
        c, pos, _ = tv.split_composite(q, n)
        kap = tv.kappas(q, n)
        h = sent(parts, a, q)
        pi = (NU - h) * kap[0] / LAM_H
        t0 = 0.4 + A_SHARE * c[FAIL]
        rows[(reads, gain)] = (steps, h, float(c[0]), float(c[n - 1]), drift)
        if reads is None:
            assert steps is None
            np.testing.assert_allclose(c[n - 1], 0.0, atol=0.0)      # the alarm never rose
            np.testing.assert_allclose(c[0], t0 + pi, atol=1e-8)
            assert NU - h > 0.05
            expect = LAM_H * gain
        else:
            assert steps is not None
            np.testing.assert_allclose(h, NU, atol=TOL)
            np.testing.assert_allclose(c[n - 1], 1.0 - c[0], atol=1e-8)   # a = gain_r delta
            np.testing.assert_allclose(1.0 - c[0], (1.0 - t0 - pi) / (1.0 + gain), atol=1e-8)
            assert 1.0 - c[0] > 1e-3                                      # never closes
            expect = LAM_H * gain + LAM_A * 1.0
        anti = anti_part(b, q)
        np.testing.assert_allclose(abs(anti[0, n - 1]), expect, rtol=1e-9)
    print("doodad (reads, gain) -> (steps, h~, c_0, alarm, drift)", rows)
    res = [1.0 - rows[("joint", g)][2] for g in GAINS]
    assert res[0] > res[1] > res[2]

    # a doodad that reads the collective's deficit instead aims at the joint it cannot tell
    dd = dict(reads="collective", joint=0, gain=4.0, gain_r=1.0)
    parts, a, b, n = fed(with_spare=True, doodad=dd)
    steps, q = time_to_return(parts, a, b, 200000)
    q, drift = tv.drifted(b, (steps or 200000) + 20000)
    c, _, _ = tv.split_composite(q, n)
    print("collective-reading doodad: steps", steps, "h~", sent(parts, a, q),
          "c_0", float(c[0]), "alarm", float(c[n - 1]), "drift", drift)


# ---------------------------------------------------------------------------
# Test 5: R5 -- the word-gated parent is non-exact.
# ---------------------------------------------------------------------------


def test_5_R5_the_word_gated_parent_is_non_exact():
    """MEASURES R5: at the failed state with the leaver's health at ``1/2``, the composite
    forcing's antisymmetric part between the leaver's health ``c_F`` and the spare's
    placement ``p_s`` is nonzero for ``rho_F > 0`` -- the leaver's word moves the spare and
    the spare's arrival returns nothing through the word -- and zero for ``rho_F = 0``,
    where only the exact cost couples them. Its value, derived here: with ``xi_s`` the
    unwrapped parent's pull on the spare and ``g`` the word at the state,
    ``d xi_{p_s}/d c_F - d xi_{c_F}/d p_s = (rho_F / sum rho) xi_s - (1 - g) d xi_s / d c_F``,
    the first term the word's, the second because the spare's placement is scaled by ``g``
    while the leaver's health is not."""
    for rho_F in (0.0, 1.0 / 3.0, 1.0):
        parts, a, b, n = after_failure(rho_F=rho_F)
        q0 = np.array(flat_state(b.P.init), dtype=float)
        q0[FAIL] = 0.5
        entry = anti_part(b, q0)[FAIL, n + 3]
        # the unwrapped composite's pull on the spare and its mixed derivative at this state
        _, _, b13 = tv.after_failure(mode="silent")
        xi_s = tv.forcing(b13, q0)[n + 3]
        J13 = np.asarray(jax.jacobian(lambda x: b13.omega(x, EMPTY, EXT)[0])(jnp.asarray(q0)))
        g = (1.0 - rho_F) + rho_F * 0.5
        expect = rho_F * xi_s - (1.0 - g) * J13[n + 3, FAIL]
        print("rho_F", rho_F, "(d omega)(c_F, p_s)", entry, "expected", expect)
        np.testing.assert_allclose(entry, expect, atol=1e-12)
        assert (abs(entry) > 1e-6) == (rho_F > 0.0)
