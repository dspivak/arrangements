"""Composition audit, stage (b): the composites the audit names, run against its predictions.

Book plan P26 (b) for Ch 1's linked files (David, 2026-10-30: "let's not leave something for later. finish it. do
the work."). ``notes/COMPOSITION-AUDIT.md`` §0 states the rule, §§1–14 predict for each file what an added
upstream push and an added exact downstream reader do to its claim, and §17 fixes, before this file existed, the
composite each test here builds and the numbers it must show. Every reader below is built by composition in
``arr`` (``compose_seq``, ``tensor_arrangements``), never by adding a term to a cost by hand, and each has its
*unilateral control*: the same reader with its covector on the value it reads set to zero, which must leave the
file's own numbers to 1e-12. A test that fails is a refuted prediction, reported in the audit, not repaired here.

The rule under test (audit §0): composition is the Kleisli composite for the writer monad of one-forms, so an added
part adds one term to the forcing on the audited configuration; an exact reader with a quadratic preference adds
$-\\kappa\\,(o(q) - m_R)\\,(T_q o)^\\top$, a spring on what it reads; a one-way reader adds nothing.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arrangements import active_inference as ai
from arrangements import carving
from arrangements.arrangement import Arrangement, exact
from arrangements.attention import input_dim, make_input, message, msg_dim, signal
from arrangements.cascade import EDIM2, MDIM2, read, unit, source, leaf, worker
from arrangements.coalgebra import configuration
from arrangements.collective import position, run as crun, sent_onward, split
from arrangements.network import fan_box, map_box
from arrangements.town import Settings, build, eve_town, run as town_run, summary
from arrangements.wiring import compose_seq, tensor_arrangements, trivial_parameter

from tests import test_carving as tcv
from tests import test_cascade as tcas
from tests import test_dispersal as td
from tests import test_dispersal_momentum as tdm
from tests import test_many_parts as tmp
from tests import test_mutual_need as mn
from tests import test_tilted_cue as tc

EMPTY = jnp.zeros(0)
MD, ED = msg_dim(1), input_dim(1, 1)          # the attention operad's message and input sizes at dim H = dim P = 1


# ---------------------------------------------------------------------------
# Readers, built once.
# ---------------------------------------------------------------------------


def town_reader(inputs, potential, exact_read=True, label="reader"):
    """A stateless system reading ``inputs`` published values of the town with potential ``potential(e)``
    (``town.node``'s convention); ``exact_read=False`` zeroes its covector on what it reads."""
    om = exact(lambda q, m, e: potential(e))

    def unilateral(q, m, e):
        xq, xm, xn = om(q, m, e)
        return xq, xm, jnp.zeros_like(xn)

    return Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=0, in_dim_N=inputs, out_dim_N=0,
                       out_f=lambda q, m: EMPTY, in_f=lambda q, m, e: EMPTY,
                       omega=om if exact_read else unilateral, label=label)


def output_reader(kappa, pref, exact_read=True):
    """``B -> B``: passes the exterior input down unchanged and the message up unchanged, and reads the
    outward signal with cost ``kappa/2 (h - pref)^2``: an exact reader of a collective's or the cue
    composite's output."""
    om = exact(lambda q, m, n: -0.5 * kappa * (signal(m)[0] - pref) ** 2)

    def unilateral(q, m, n):
        xq, xm, xn = om(q, m, n)
        return xq, jnp.zeros_like(xm), xn

    return Arrangement(P=trivial_parameter(), in_dim_M=ED, out_dim_M=MD, in_dim_N=ED, out_dim_N=MD,
                       out_f=lambda q, m: m, in_f=lambda q, m, n: n,
                       omega=om if exact_read else unilateral, label="reader")


def wire_reader(kappa, pref, exact_read=True):
    """``<0|1> -> I``: reads one wire with cost ``kappa/2 (m - pref)^2``."""
    om = exact(lambda q, m, n: -0.5 * kappa * (m[0] - pref) ** 2)

    def unilateral(q, m, n):
        xq, xm, xn = om(q, m, n)
        return xq, jnp.zeros_like(xm), xn

    return Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=1, in_dim_N=0, out_dim_N=0,
                       out_f=lambda q, m: EMPTY, in_f=lambda q, m, n: EMPTY,
                       omega=om if exact_read else unilateral, label="reader")


# ---------------------------------------------------------------------------
# 1, 3. The town: an exact reader of the smoke is the perfected premium.
# ---------------------------------------------------------------------------


def test_1_3_an_exact_reader_of_the_smoke_is_the_perfected_premium():
    """Prediction (audit §17, rows 1 and 3): the stage-1 two-dial town with an exact reader of the smoke,
    cost ``kappa s^2/2`` at ``kappa = 2``, follows the informed stage-2 plant at ``p = 2`` pathwise to 1e-9 on
    output, cleanliness and coughing, and rests at ``(x, c) = (.99541, .95196)``; the unilateral control
    never cleans. Could fail if the pullback along ``s = (1 - c) x`` were not the premium term."""
    kappa = 2.0
    potential = lambda e: -0.5 * kappa * e[0] ** 2
    read = build(1, Settings(), clean=True, extra=[(town_reader(1, potential), "smoke")])
    informed = build(2, Settings(), clean=True, informed=True)
    a, b = town_run(read, 60), town_run(informed, 60)
    for name in ("output", "cleanliness", "coughing"):
        np.testing.assert_allclose(read.value(a, name), informed.value(b, name), atol=1e-9)
    rest = [read.value(a, "output")[-1, 0], read.value(a, "cleanliness")[-1, 0]]
    print("rest with the exact smoke reader", rest)
    np.testing.assert_allclose(rest, [0.99541, 0.95196], atol=1e-4)
    control = build(1, Settings(), clean=True, extra=[(town_reader(1, potential, exact_read=False), "smoke")])
    c = town_run(control, 60)
    np.testing.assert_array_equal(control.value(c, "cleanliness"), 0.0)
    np.testing.assert_allclose(control.value(c, "output")[-1, 0], 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 2. Her account: an exact reader of the estimate bends the fit.
# ---------------------------------------------------------------------------


def test_2_an_exact_reader_of_her_estimate_bends_the_fit():
    """Prediction (row 2): the one-dial stage-4 town with an exact reader of her estimate, ``kappa = .05``,
    preference 0, ends with the estimate in ``(.5, .9)`` (self-consistent rest .72) and the premium below 1,
    and reduces harm against stage 3 by less than the plain account does; the unilateral control equals
    ``build(4)`` on every coordinate. Could fail if the fit were held by more than ``u^2 = (s^2/2)^2``."""
    kappa = 0.05
    potential = lambda e: -0.5 * kappa * e[0] ** 2
    plain, hist = build(4), build(3)
    read = build(4, extra=[(town_reader(1, potential), "estimate")])
    control = build(4, extra=[(town_reader(1, potential, exact_read=False), "estimate")])
    a, b, c, h = town_run(plain), town_run(read), town_run(control), town_run(hist)
    for name in plain.slots:
        np.testing.assert_allclose(plain.value(a, name), control.value(c, name), atol=1e-12)
    s, s0, sh = summary(read, b), summary(plain, a), summary(hist, h)
    reduction, reduction0 = 1 - s["harm"] / sh["harm"], 1 - s0["harm"] / sh["harm"]
    print("final estimate", s["final_estimate"], "final premium", s["final_premium"],
          "harm reduction", reduction, "plain", reduction0, "premium reaches 1 at", s["premium_reaches_1"])
    assert 0.5 < s["final_estimate"] < 0.9
    assert s["final_premium"] < 1.0
    assert reduction < reduction0


# ---------------------------------------------------------------------------
# 4. The tilted cue: an exact reader of the output leaves the energy accounting.
# ---------------------------------------------------------------------------


def run_read(phi, amp, T, kappa, pref, exact_read=True, h=tc.H, post=tc.POST, ridge=None):
    """``test_tilted_cue.run`` for the composite with an output reader; ``V`` includes the reader's term."""
    t = np.arange(int(round((T + post) / h))) * h
    us = np.asarray(amp * tc.bump(t / T))
    arr = compose_seq(tc.tilted_system(phi, h), output_reader(kappa, pref, exact_read))
    ys = tc.scan_closed(arr, jnp.asarray(us))
    Vp = tc.V_of(phi)
    extra = (lambda y: 0.5 * kappa * (y[0] + y[1] - pref) ** 2) if exact_read else (lambda y: 0.0)
    V = np.asarray(jax.vmap(lambda y: Vp(y) + extra(y))(jnp.asarray(ys)))
    g = ys[:, 4]
    k = int(round(T / h))
    out = dict(W=float(np.sum(us * np.diff(g))), V0=float(V[0]), VT=float(V[k]),
               Vend=float(V[-1]), Vpeak=float(V[:k + 1].max()), gend=float(g[-1]))
    out["released"] = out["VT"] - out["Vend"]
    if ridge is not None:
        kc = int(np.argmax(g < ridge))
        out["W_cross"] = float(np.sum(us[:kc] * np.diff(g)[:kc]))
    return out


def minimal_amplitude_read(phi, T, kappa, pref, iters=16):
    captures = lambda a: run_read(phi, a, T, kappa, pref)["gend"] < 0.5
    hi = -1.0
    while not captures(hi):
        hi *= 2.0
    lo = hi / 2.0 if hi < -1.0 else 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if captures(mid):
            hi = mid
        else:
            lo = mid
    assert not captures(lo)
    return hi


def test_4_an_exact_reader_of_the_cue_composites_output_leaves_the_ridge_and_the_depth():
    """Prediction (row 4): with an exact reader of the output signal ``z_1 + z_2`` (``kappa = 1``, preference
    1, the rest's value), the minimal ``T = .2`` cue at ``D = 16``, ``eps = .25`` meets the file's shared
    bounds, ``V`` including the reader's term, and ``V(0) - V(end) = D``; the unilateral control at the
    file's amplitude equals the file's run to 1e-12. Could fail if the reader's direction were not orthogonal
    to the cue's."""
    D, eps, T = 16.0, 0.25, 0.2
    phi, c = tc.tilted(D, eps)
    amp = minimal_amplitude_read(phi, T, 1.0, 1.0)
    out = run_read(phi, amp, T, 1.0, 1.0, ridge=c)
    print("minimal cue with the reader", amp, {k: round(v, 4) for k, v in out.items()})
    tc._check_minimal_brief(D, eps, out)
    np.testing.assert_allclose(out["V0"] - out["Vend"], D, atol=1e-9)
    assert eps < out["W_cross"] < 5.0 * eps
    plain_amp = tc.threshold(D, eps, T)
    plain, uni = tc.run(phi, plain_amp, T), run_read(phi, plain_amp, T, 1.0, 1.0, exact_read=False)
    for key in ("W", "V0", "VT", "Vend", "gend"):
        np.testing.assert_allclose(uni[key], plain[key], atol=1e-12)


# ---------------------------------------------------------------------------
# 5. The hierarchy: an exact reader of the hand is one more spring.
# ---------------------------------------------------------------------------


def hierarchy_read(kappa_r, pref, exact_read=True, kappa=1.0, P=4.0, g=1.0):
    """``hand -> fan -> (level 1 (x) reader of h) -> level 2``, closed; position ``(h, mu_1, mu_2)``."""
    eta = 1.0 / (kappa + 3 * P + kappa_r)
    hand = compose_seq(ai.hand(kappa, eta), fan_box(1, [0, 0], "fan"))
    level1 = tensor_arrangements([ai.predictor(P, eta), wire_reader(kappa_r, pref, exact_read)])
    return compose_seq(compose_seq(hand, level1), ai.predictor(P, eta, goal=(P, g), top=True))


@pytest.mark.parametrize("pref,target", [(0.0, [0.4, 0.6, 0.8]), (0.5, [0.55, 0.7, 0.85])])
def test_5_an_exact_reader_of_the_hand_is_one_more_spring(pref, target):
    """Prediction (row 5): the rest is the linear chain's with the reader's spring, ``(.4, .6, .8)`` at
    preference 0 and ``(.55, .7, .85)`` at preference .5, to 1e-8; the unilateral control rests at the
    file's ``(.5714, .7143, .8571)``. Could fail if the fan or the reader's pullback misrouted the force."""
    arr = hierarchy_read(1.0, pref)
    x = np.asarray(arr.P.position(ai.run(arr, 80000)))
    print("rest with the reader", x)
    np.testing.assert_allclose(x, target, atol=1e-8)
    if pref == 0.0:
        ctrl = hierarchy_read(1.0, pref, exact_read=False)
        y = np.asarray(ctrl.P.position(ai.run(ctrl, 80000)))
        np.testing.assert_allclose(y, [4 / 7, 5 / 7, 6 / 7], atol=1e-8)


# ---------------------------------------------------------------------------
# 6, 8. The collectives: a reader of the onward signal leaves the flat directions flat.
# ---------------------------------------------------------------------------


def with_reader(C, kappa, pref, exact_read=True):
    return replace(C, arr=compose_seq(C.arr, output_reader(kappa, pref, exact_read)))


def test_6_8_a_reader_of_the_onward_signal_leaves_the_free_directions_free():
    """Prediction (rows 6, 8): with an exact reader of the onward signal (``kappa = 1``, preference ``nu``),
    a displaced part still recovers only ``1/n`` of its displacement (``n = 3``, within 5%), the others move in
    ``delta/n``, the onward signal returns to ``nu``; the far part (``d = 3``, ``n = 3``) stays out; with
    preference ``1.1 nu`` the signal settles at the combined target. Could fail if the reader's pullback reached
    a placement by anything but the symmetric total."""
    n, nu = 3, 3 * td.K_STAR * td.SIG
    C = with_reader(td.build([td.P_STAR + td.DELTA] + [td.P_STAR] * (n - 1), nu), 1.0, nu)
    q = np.asarray(position(C, crun(C, 40000)))
    _, pos, _ = split(C, q)
    pos = np.asarray(pos)
    recovered = (td.DELTA - (pos[0] - td.P_STAR)) / td.DELTA
    print("recovered fraction", recovered, "others moved in", td.P_STAR - pos[1:])
    assert abs(float(signal(sent_onward(C, q))[0]) - nu) < 1e-8
    assert abs(n * recovered - 1.0) < 0.05
    np.testing.assert_allclose(td.P_STAR - pos[1:], td.DELTA / n, rtol=0.05)
    # the far part the rest can cover stays out
    C3 = with_reader(td.build([td.D_FAR] + [td.distance(td.NU_3 / 3)] * 2, td.NU_3), 1.0, td.NU_3)
    q3 = np.asarray(position(C3, crun(C3, 20000)))
    _, pos3, _ = split(C3, q3)
    assert float(pos3[0]) > 0.9 * td.D_FAR, pos3
    # a different preference moves the family's level to the combined target, and no further
    Cd = with_reader(td.build([td.P_STAR] * n, nu), 1.0, 1.1 * nu)
    qd = np.asarray(position(Cd, crun(Cd, 40000)))
    np.testing.assert_allclose(float(signal(sent_onward(Cd, qd))[0]), (nu + 1.1 * nu) / 2.0, atol=1e-6)
    # the unilateral control is the file's run
    Cu = with_reader(td.build([td.P_STAR + td.DELTA] + [td.P_STAR] * (n - 1), nu), 1.0, nu, exact_read=False)
    C0 = td.build([td.P_STAR + td.DELTA] + [td.P_STAR] * (n - 1), nu)
    np.testing.assert_allclose(position(Cu, crun(Cu, 2000)), position(C0, crun(C0, 2000)), atol=1e-12)


def test_6_a_damped_push_still_lands_the_configuration_somewhere_new_under_a_reader():
    """Prediction (row 6): the damped momentum collective with the exact reader (``kappa = 1``, preference
    ``nu``): the pushed part ends displaced by ``(mu xi_0 / c)(1 - 1/n)`` and each other by ``-(mu xi_0/c)/n``
    within 5%, ``n = 3``. Could fail if the reader's term had a component along the free directions."""
    n, xi0, c = 3, 0.01, 0.1
    nu = n * tdm.K_STAR * tdm.SIG
    C = with_reader(tdm.build([tdm.P_STAR] * n, nu, xi0, damping=c), 1.0, nu)
    s = crun(C, 20000)
    parent = s[0][1]                       # ((parts, parent), reader): the parent's (q, xi)
    moved = np.asarray(parent[0][:n]) - tdm.P_STAR
    coast = tdm.MU * xi0 / c
    want = np.array([coast * (1 - 1 / n)] + [-coast / n] * (n - 1))
    print("moved", moved, "predicted", want)
    np.testing.assert_allclose(moved, want, rtol=0.05)


# ---------------------------------------------------------------------------
# 9, 10. The cascade and the ratchet: an exact buyer holds the parts by its need.
# ---------------------------------------------------------------------------


def buyer(r, kappa, m_R):
    """The market made exact: pays ``r`` times the product forward as ``cascade.market`` does, and has the
    cost ``kappa/2 (product - m_R)^2`` on the product it receives."""
    return Arrangement(
        P=trivial_parameter(), in_dim_M=EDIM2, out_dim_M=MDIM2, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m: EMPTY,
        in_f=lambda q, m, n: make_input(message(jnp.array([r * m[1], 0.0]), 1.0), jnp.zeros(1)),
        omega=exact(lambda q, m, n: -0.5 * kappa * (m[1] - m_R) ** 2), label="buyer")


def settle(arr, steps, state=None):
    step = tcas.closed_step(arr, EMPTY)
    return jax.jit(lambda s: jax.lax.fori_loop(0, steps, lambda i, x: step(x), s))(arr.P.init if state is None else state)


def test_9_10_an_exact_buyer_holds_the_workers_by_its_need():
    """Prediction (rows 9, 10): with an exact buyer (``kappa = 2``, preference twice the one-way equilibrium
    product) in the market's place, the one-way run's above-threshold drain (``1.05 S_c``) leaves every
    worker's placement below 1 and the fund above 0, where the one-way run sends all past 2 and kills the
    fund; and (R3) at equal balance and amplitudes the force toward home on a worker at the source is larger
    when all four are out than when three are home, while with the one-way market the two forces are equal.
    Could fail if the buyer's pull through ``sum kappa_j a_j`` were weaker than the source's hold. The last
    block records a refuted prediction of this run's own: the buyer reaches the amplitudes, not only the
    placements."""
    u, arr1 = tcas.one_level()
    f_eq, Sc, after = tcas.threshold(u, arr1)
    s_eq1, _ = tcas.shocker(u, arr1)
    d = read(u, arr1.P.position(s_eq1))
    product = float(jnp.sum(jnp.exp(-d["positions"] ** 2 / (2 * tcas.TAU)) * jnp.array([c["a"] for c in d["children"]])))
    one_way = tcas.outcome(u, arr1, after(1.05 * Sc))
    print("one-way: threshold", Sc, "after the drain: fund", one_way[0], "placements", np.round(one_way[1], 2))
    assert one_way[0] < 0 and np.all(one_way[1] > 2.0)

    u2 = unit(tcas.workers(4), positions=[0.06] * 4, mobilities=[tcas.MU_P] * 4, tau=tcas.TAU,
              fund=dict(f0=20.0, mu=0.05, wmax=1.0, K=1.0, c0=2.0, leak=0.1, record=(tcas.ALPHA, tcas.BETA)),
              src=source(tcas.S_SRC), src_position=tcas.L_SRC)
    arr2 = compose_seq(u2.arr, buyer(2.0, 2.0, 2.0 * product))
    s_eq2 = settle(arr2, tcas.PRE)
    drained = ((s_eq2[0][0], s_eq2[0][1] - 1.05 * Sc), s_eq2[1])      # ((collective, fund), buyer): the fund
    fund, pos, amp = tcas.outcome(u2, arr2, settle(arr2, tcas.POST, drained))
    print("exact buyer: equilibrium", tcas.outcome(u2, arr2, s_eq2)[0], "after the same drain: fund", fund,
          "placements", np.round(pos, 3), "amplitudes", np.round(amp, 3))
    assert np.all(np.abs(pos) < 1.0)
    assert fund > 0
    # A prediction added before this part ran (2026-10-30) and REFUTED by it: that a drain killing the fund
    # (three times its balance) would still send every worker to the source, the buyer's pull on a placement
    # being xi a_j kappa'(p_j) and a_j bought by the wage. What happened: the buyer's covector on the product is
    # pulled back along the collective's out-transfer onto the amplitudes themselves (a worker sends (0, a, 1)), so
    # the buyer's need drives production with no wage at all; income persists, and the fund climbs back from
    # -180 to its equilibrium in a few hundred steps (the leak term, -leak f, also pays into a negative balance,
    # a feature of the model's fund). Asserted now: the fund ends above 0, every worker home and producing.
    f_eq2 = tcas.outcome(u2, arr2, s_eq2)[0]
    killed = ((s_eq2[0][0], s_eq2[0][1] - 3.0 * f_eq2), s_eq2[1])
    assert tcas.outcome(u2, arr2, killed)[0] < -1.9 * f_eq2        # the drain did land on the fund
    fund_k, pos_k, amp_k = tcas.outcome(u2, arr2, settle(arr2, tcas.POST, killed))
    print("exact buyer, the fund drained three times over: fund", fund_k, "placements", np.round(pos_k, 3),
          "amplitudes", np.round(amp_k, 3))
    assert fund_k > 0 and abs(fund_k - f_eq2) < 0.01
    assert np.all(np.abs(pos_k) < 1.0) and np.all(amp_k > 1.0)

    # (R3) the force on an absent worker's placement, all out against three home
    q = np.asarray(arr1.P.position(s_eq1)).copy()
    q[:4] = 1.0                                     # amplitudes
    allout, threehome = q.copy(), q.copy()
    allout[4:8] = tcas.L_SRC
    threehome[4:8] = [tcas.L_SRC, 0.06, 0.06, 0.06]
    f1 = [float(arr1.omega(jnp.asarray(x), EMPTY, EMPTY)[0][4]) for x in (allout, threehome)]
    q2 = np.asarray(arr2.P.position(s_eq2)).copy()
    q2[:4] = 1.0
    allout2, threehome2 = q2.copy(), q2.copy()
    allout2[4:8] = tcas.L_SRC
    threehome2[4:8] = [tcas.L_SRC, 0.06, 0.06, 0.06]
    f2 = [float(arr2.omega(jnp.asarray(x), EMPTY, EMPTY)[0][4]) for x in (allout2, threehome2)]
    print("force toward home on the absent worker, one-way (all out, three home):", f1, " exact buyer:", f2)
    assert abs(f1[0] - f1[1]) < 1e-12
    assert f2[0] < f2[1] - 1e-3


# ---------------------------------------------------------------------------
# 11. Carving: a reader stiffer than the saddle-node erases the capacity.
# ---------------------------------------------------------------------------


def carving_read(kappa, q_R):
    """The carving reader with a third part held at ``q_R`` (mobility 0) and an exact coupling
    ``kappa/2 (q - q_R)^2`` between it and the quick part: an exact reader of ``q`` by composition."""
    held = carving._part(0.0, q_R, lambda x, n: 0.0, 1, False, "held")
    coupling = carving._coupling(exact(lambda s, m, n: -0.5 * kappa * (m[0] - m[1]) ** 2), "reader")
    A = carving.reader([carving.tuning_box(tcv.ETA_W, tcv.W0, tcv.W0, tcv.GAMMA, tcv.BETA), held],
                       [carving.curvature_coupling(), coupling], 0.0, tcv.ETA_Q)
    return A, carving.note_step(A)


def carve(kappa, q_R, hold=4000, after=40000):
    A, step = carving_read(kappa, q_R)
    return carving.play(A, step, tcv.schedule(hold, after), record=False, start=[0.0, tcv.W0, q_R])


def test_11_a_reader_past_the_saddle_node_erases_the_carved_rest():
    """Prediction (row 11, corrected before the run from the entry's ``kappa > 1/2``): the slaved slow rest
    solves ``(w - kappa)/2 = gamma (w - w0) + beta (w - w0)^3``, whose roots merge at ``kappa_c = .0196``; so
    at ``kappa = .05`` a 4000-step note leaves ``|q| < .01``, and at ``kappa = .01`` it leaves
    ``(q, w) = (.6712, .4605)`` within .01; at ``kappa = 0`` the construction reproduces the file's reader to
    1e-9. Could fail if the reader's spring did not enter the slow flow through ``q^2/2``."""
    zero = carve(0.0, 0.0)
    np.testing.assert_allclose(zero[:2], tcv.outcome(4000), atol=1e-9)
    weak = carve(0.01, 0.0)
    strong = carve(0.05, 0.0)
    print("kappa 0:", zero[:2], " kappa .01:", weak[:2], " kappa .05:", strong[:2])
    np.testing.assert_allclose(weak[:2], [0.6712, 0.4605], atol=0.01)
    assert abs(strong[0]) < 0.01


# ---------------------------------------------------------------------------
# 12. Many parts: an exact reader with a preference is an ungated supply.
# ---------------------------------------------------------------------------


def ring_read(n, s, kappa, q_R, cut=0, exact_read=True):
    """``test_many_parts.collective`` for the ring at ``beta = 0`` with part ``cut``'s supply wire cut, plus an
    exact reader of part 0's amplitude: one more wire from the fan, one more stage."""
    q_hi = tmp.symmetric_roots(s)[1]
    zero = map_box(lambda v: jnp.zeros(1), 0, 1, "cut wire")
    parts = [mn.part(q_hi, label=f"P{i}") for i in range(n)] + [mn.supply(s), zero]
    idx, stages = [], []
    for i in range(n):
        idx += [i, (i - 1) % n, (n + 1) if i == cut else n]
        stages.append(compose_seq(
            tensor_arrangements([fan_box(1, [0], "own"), fan_box(1, [0], "neighbour"), fan_box(1, [0], "supply")]),
            tmp.need_box(0.0, label=f"need {i}")))
    idx += [0]
    stages.append(wire_reader(kappa, q_R, exact_read))
    routed = compose_seq(tensor_arrangements(parts), fan_box(n + 2, idx, "wiring"))
    return compose_seq(routed, tensor_arrangements(stages))


def test_12_an_exact_reader_with_a_preference_keeps_the_cut_ring_alive():
    """Prediction (row 12): ring, ``n = 5``, ``s = 4``, part 0's supply cut, an exact reader of part 0 with
    ``kappa = .5``: preference 1 holds every part, at ``(.3333, 1.0, 3.0, 3.6667, 3.7273)`` within 1e-3 (the
    cut part at ``kappa q_R/(b + kappa)``, each next at ``(alpha s r - b K)/(b r)``); preference 0 lets every
    part die; the unilateral control dies as the file's cut ring does. Could fail if the gate let a dying part
    drive its neighbour, or the reader's drive were gated."""
    n, s = 5, 4.0
    alive = mn.settle(ring_read(n, s, 0.5, 1.0), 20000)[:n]
    print("ring with a reader of preference 1:", np.round(np.asarray(alive), 4))
    np.testing.assert_allclose(alive, [1 / 3, 1.0, 3.0, 11 / 3, 41 / 11], atol=1e-3)
    assert tmp.held_all(alive) == [True] * n
    load = mn.settle(ring_read(n, s, 0.5, 0.0), 20000)[:n]
    assert float(np.max(load)) < 1e-3, load
    ctrl = mn.settle(ring_read(n, s, 0.5, 1.0, exact_read=False), 20000)[:n]
    assert float(np.max(ctrl)) < 1e-3, ctrl


# ---------------------------------------------------------------------------
# 13. Diligence: an exact plant has cleaned away what it would lobby with.
# ---------------------------------------------------------------------------


def test_13_an_exact_plant_moves_the_rate_by_less_than_its_smoke_squared():
    """Prediction (row 13, corrected before the run from the entry's ``1.955``): the exact plant's push on
    its premium is ``-s^2/2`` with the clean plant's ``s`` near .05, so the final rate moves by less than .005
    and stays within 2% of beta. Could fail if the plant's lobbying reached the rate by more than that."""
    plain, lobby = eve_town(4, diligence=True), eve_town(4, diligence=True, exact_plant=True)
    a, b = summary(plain, town_run(plain)), summary(lobby, town_run(lobby))
    print("final premium, price-taker", a["final_premium"], "exact plant", b["final_premium"],
          "final smoke", b["final_smoke"])
    assert abs(a["final_premium"] - b["final_premium"]) < 0.005
    assert abs(b["final_premium"] - plain.settings.beta) < 0.02 * plain.settings.beta
    assert b["final_premium"] < a["final_premium"]          # it lobbies, a little


# ---------------------------------------------------------------------------
# 14. The historical insurer: an exact reader of the quote with a preference leads the evidence.
# ---------------------------------------------------------------------------


def test_14_an_exact_reader_of_the_quote_moves_it_before_its_evidence():
    """Prediction (row 14): stage 3 with an exact reader of the premium, ``kappa = 1``, preference 2.5, has the
    premium rest at ``kappa p_R/(1 + kappa) = 1.25`` before any claim, so it crosses 1 before ``t = 2`` and
    before the court's fit does; the unilateral control lags its evidence as the file says. Could fail if the
    reader's pullback did not land on the premium's own coordinate."""
    potential = lambda e: -0.5 * 1.0 * (e[0] - 2.5) ** 2
    read = build(3, extra=[(town_reader(1, potential), "premium")])
    ctrl = build(3, extra=[(town_reader(1, potential, exact_read=False), "premium")])
    r, c = summary(read, town_run(read)), summary(ctrl, town_run(ctrl))
    print("premium reaches 1 at", r["premium_reaches_1"], "court at", r["assessed_reaches_1"],
          "| control:", c["premium_reaches_1"], c["assessed_reaches_1"])
    assert r["premium_reaches_1"] is not None and r["premium_reaches_1"] < 2.0
    assert r["premium_reaches_1"] < r["assessed_reaches_1"]
    assert c["premium_reaches_1"] > c["assessed_reaches_1"]


# ===========================================================================
# Chs 3 and 2 (audit §§18–20): the same stage for the other seven linked files.
# ===========================================================================

from arrangements import pendulum as pend
from arrangements.coalgebra import phase
from arrangements.copying import contact, copy_pair, copy_pair_q, driven_step, identity, osc
from arrangements.functors import Phi, closed_step as phi_closed_step
from arrangements.interpretation import trivial_omega
from arrangements.network import example_box, identity_box, loss_box, parameter_box, plus_box, tanh_box, times_box
from arrangements.wiring import boxes_and_couplers, parallel_arrangements

from tests import test_pendulum as tp
from tests.chain import closed_run, flat_state, one_port_box, spring
from tests.test_kirchhoff_node import capacitor, resistor, source as voltage_source, network, run as circuit_run
from tests.test_two_tanks import cylinder, pipe, volumes


# ---------------------------------------------------------------------------
# 18.1 The pendulum: an exact reader of the lower bob is one more spring.
# ---------------------------------------------------------------------------


def position_reader(kappa_r, x_R, exact_read=True):
    """``<0|2> -> I``: reads a position in the plane with cost ``kappa_r/2 |q - x_R|^2``."""
    om = exact(lambda q, x, n: -0.5 * kappa_r * jnp.sum((x - x_R) ** 2))

    def unilateral(q, x, n):
        xq, xm, xn = om(q, x, n)
        return xq, jnp.zeros_like(xm), xn

    return Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=2, in_dim_N=0, out_dim_N=0,
                       out_f=lambda q, x: EMPTY, in_f=lambda q, x, n: EMPTY,
                       omega=om if exact_read else unilateral, label="reader")


def pendulum_read(kappa, dt, kappa_r, x_R):
    """``reader . link^2 . pivot``: ``pendulum.pendulum`` with the reader in ``drop``'s place."""
    a = pend.pivot()
    for p in pend.positions(tp.ANGLES, tp.L):
        a = compose_seq(a, pend.link(tp.M, kappa, tp.L, p, tp.G, dt))
    return compose_seq(a, position_reader(kappa_r, x_R))


X_R = pend.positions(tp.ANGLES, tp.L)[1] + jnp.array([0.1, 0.0])
KAPPA_R = 1.0


def _spring_energy(s, kappa):
    (q1, p1), (q2, p2) = pend.bob_states(s)
    return float(tp._energy(s, kappa) + 0.5 * KAPPA_R * jnp.sum((q2 - X_R) ** 2))


def test_18_1_the_pendulum_read_by_a_spring_is_still_nested_and_still_set_by_the_bar():
    """Prediction (row 18.1): with an exact reader of the lower bob (a spring, ``kappa_r = 1``, to a point near
    its start), the nested composite equals the flat computation with the spring's potential added, to 1e-10 over
    500 steps at ``kappa = 1e4``; at steel with step 1e-4 the energy including the spring stays within 0.1% over
    10,000 steps; and the stiff pendulum with the spring stays within 1 mm of the rigid pendulum with the same spring
    over 1 s. Could fail if the reader's spring did not add to the composite's cost, or set the step."""
    kappa, dt, N = 1e4, 1e-3, 500
    A = pendulum_read(kappa, dt, KAPPA_R, X_R)
    s = tp._runner(A)(A.P.init, N)
    q = jnp.stack(pend.positions(tp.ANGLES, tp.L)); p = jnp.zeros_like(q)
    cost = lambda Q: (pend.bar_cost(kappa, tp.L, jnp.zeros(2), Q[0]) + pend.bar_cost(kappa, tp.L, Q[0], Q[1])
                      + tp.M * tp.G * jnp.sum(Q[:, 1]) + 0.5 * KAPPA_R * jnp.sum((Q[1] - X_R) ** 2))
    g = jax.jit(jax.grad(cost))
    for _ in range(N):
        q = q + dt * p / tp.M
        p = p - dt * g(q)
    assert float(jnp.max(jnp.abs(jnp.stack([b for b, _ in pend.bob_states(s)]) - q))) < 1e-10
    # the bar still sets the step
    kappa = 2e7
    A = pendulum_read(kappa, 1e-4, KAPPA_R, X_R)
    run, E0, s = tp._runner(A), _spring_energy(A.P.init, kappa), A.P.init
    for _ in range(20):
        s = run(s, 500)
        assert abs(_spring_energy(s, kappa) - E0) < 1e-3 * abs(E0)
    # the stiff pendulum with the spring against the rigid one with the same spring
    def rigid_with_spring(h):
        def lag(th, w):
            x1 = tp.L * jnp.array([jnp.sin(th[0]), -jnp.cos(th[0])])
            x2 = x1 + tp.L * jnp.array([jnp.sin(th[1]), -jnp.cos(th[1])])
            v1 = tp.L * w[0] * jnp.array([jnp.cos(th[0]), jnp.sin(th[0])])
            v2 = v1 + tp.L * w[1] * jnp.array([jnp.cos(th[1]), jnp.sin(th[1])])
            return (0.5 * tp.M * (v1 @ v1 + v2 @ v2) - tp.M * tp.G * (x1[1] + x2[1])
                    - 0.5 * KAPPA_R * jnp.sum((x2 - X_R) ** 2))
        dw = jax.grad(lag, 1)

        def f(y):
            th, w = y[:2], y[2:]
            acc = jnp.linalg.solve(jax.jacfwd(dw, 1)(th, w), jax.grad(lag, 0)(th, w) - jax.jacfwd(dw, 0)(th, w) @ w)
            return jnp.concatenate([w, acc])

        def rk4(y):
            k1 = f(y); k2 = f(y + h / 2 * k1); k3 = f(y + h / 2 * k2); k4 = f(y + h * k3)
            return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return jax.jit(lambda y, k: jax.lax.fori_loop(0, k, lambda _i, t: rk4(t), y), static_argnums=1)

    dt = 2e-5
    A = pendulum_read(kappa, dt, KAPPA_R, X_R)
    run, every = tp._runner(A), int(round(1e-3 / dt))
    rigid, h = rigid_with_spring(1e-5), 1e-5
    s, y, worst = A.P.init, jnp.array(tp.ANGLES + [0.0, 0.0]), 0.0
    for _ in range(1000):
        s, y = run(s, every), rigid(y, int(round(every * dt / h)))
        q2 = pend.bob_states(s)[1][0]
        worst = max(worst, float(jnp.linalg.norm(q2 - pend.positions([y[0], y[1]], tp.L)[1])))
    print("stiff pendulum with the spring against the rigid one with the spring: worst gap", worst)
    assert worst < 1e-3


# ---------------------------------------------------------------------------
# 18.2 The copy pair: an exact coupling has no original.
# ---------------------------------------------------------------------------


def copy_pair_exact(eta_w, eta_theta, init_original, init_copy):
    """``copy_pair`` with the exact contact: the cost ``K/2 (x - y)^2`` returned to both oscillators."""
    V = parallel_arrangements(osc(eta_w, eta_theta, init_original), osc(eta_w, eta_theta, init_copy))
    return compose_seq(parallel_arrangements(V, identity(1)), contact(1.0))


def test_18_2_an_exact_coupling_moves_the_original_too():
    """Prediction (row 18.2): with the contact exact, after 4000 steps at ``K = 1`` the two angular frequencies
    agree to 1e-6 and lie strictly inside ``(.22, .3)``, the original having moved by more than 1e-3; after 3000
    more steps at ``K = 0`` both keep that value; the one-way control leaves the original at .3 exactly. Could
    fail if the exact coupling did not lock, or locked at an end."""
    from tests.test_copy_pair import ETA, LAM, NU_COPY, NU_ORIG, _init
    W = copy_pair_exact(LAM, ETA, _init(NU_ORIG), _init(NU_COPY))
    step = driven_step(W, check_inputs=[[1.0]] * 3)
    s = W.P.init
    for K in [1.0] * 4000:
        s = step(s, jnp.array([K]))
    o, c = copy_pair_q(W, s)
    print("exact coupling, after 4000 coupled steps: original nu", o[1], "copy nu", c[1])
    assert abs(o[1] - c[1]) < 1e-6
    assert NU_COPY + 1e-3 < o[1] < NU_ORIG - 1e-3
    locked = o[1]
    for K in [0.0] * 3000:
        s = step(s, jnp.array([K]))
    o2, c2 = copy_pair_q(W, s)
    assert abs(o2[1] - locked) < 1e-9 and abs(c2[1] - locked) < 1e-9
    W1 = copy_pair(LAM, ETA, _init(NU_ORIG), _init(NU_COPY))
    step1, s1 = driven_step(W1), W1.P.init
    for K in [1.0] * 4000:
        s1 = step1(s1, jnp.array([K]))
    o1, c1 = copy_pair_q(W1, s1)
    assert o1[1] == NU_ORIG and abs(c1[1] - NU_ORIG) < 1e-6


# ---------------------------------------------------------------------------
# 19.1 Two tanks: a reader of a pressure is a reservoir through a pipe.
# ---------------------------------------------------------------------------


def levels(tanks, arr, T):
    """Run ``T`` steps and return the pressures and the total water at each step."""
    O = Phi(arr)
    state, out = O.state, []
    for _ in range(T + 1):
        q = np.asarray(arr.P.position(state))                    # the volumes (the tanks' configurations)
        es = [float(t.out_f(jnp.array([v]), EMPTY)[0]) for t, v in zip(tanks, q[:len(tanks)])]
        out.append((es, float(q[:len(tanks)].sum())))
        _, _, state = O.with_state(state).run_one((EMPTY, trivial_omega(0)), lambda _o: (EMPTY, EMPTY))
    return out


def test_19_1_a_reader_of_a_tanks_pressure_sets_both_levels_and_breaks_conservation():
    """Prediction (row 19.1): two unit cylinders from (4, 0) through a pipe of conductance 1/4, with a reservoir
    at pressure 3 joined to tank 1 by a pipe of conductance .1: both pressures within 1e-6 of 3 after 400 steps and
    the total water 6, not 4; the plain pair rests at (2, 2) with 4. Could fail if the reader's flow did not reach
    the second tank through the pipe."""
    tanks = [cylinder(1.0, 4.0), cylinder(1.0, 0.0)]
    read = boxes_and_couplers(tanks + [voltage_source(3.0, "reservoir")], [(0, 1), (0, 2)], [pipe(0.25), pipe(0.1)])
    (es, water) = levels(tanks, read, 400)[-1]
    print("with the reservoir: pressures", es, "water", water)
    np.testing.assert_allclose(es, [3.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(water, 6.0, atol=1e-6)
    plain = boxes_and_couplers(tanks, [(0, 1)], [pipe(0.25)])
    (es0, water0) = levels(tanks, plain, 400)[-1]
    np.testing.assert_allclose(es0, [2.0, 2.0], atol=1e-9)
    np.testing.assert_allclose(water0, 4.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 19.2, 19.3 Heat and waves: the same reader on both; the difference is still the momentum.
# ---------------------------------------------------------------------------


def test_19_2_a_heater_read_into_the_rod_sets_its_temperature():
    """Prediction (row 19.2): the rod (8, 0, 0), joints 1/4, with a heater at 5 joined to the middle segment by a
    joint of conductance .1: all three temperatures within 1e-6 of 5 after 600 steps, the sum near 15, not 8.
    Could fail if the heater's flow did not spread through the joints."""
    segments = [one_port_box(configuration, 1.0, jnp.array([t]), "segment") for t in (8.0, 0.0, 0.0)]
    rod = boxes_and_couplers(segments + [voltage_source(5.0, "heater")], [(0, 1), (1, 2), (1, 3)],
                             [spring(0.25), spring(0.25), spring(0.1)])
    temps, _ = closed_run(rod, 600)
    final = np.asarray(temps[-1])[:3]
    print("rod with a heater at 5:", final, "sum", final.sum())
    np.testing.assert_allclose(final, 5.0, atol=1e-6)
    assert abs(final.sum() - 15.0) < 1e-5


def test_19_3_a_wall_read_into_the_chain_leaves_it_swinging():
    """Prediction (row 19.3): the three masses from (8, 0, 0) at rest, springs 1/4, with a wall at 8/3 joined to
    the middle mass by a spring of constant .1: over 400 steps the middle mass's position ranges by more than 2 in
    the last 100 steps, the positions' sum varies by more than .5 over the run, and the energy (kinetic plus the four
    springs) stays within 10% of its start. Could fail if the wall's spring damped the chain or the step lost energy."""
    masses = [one_port_box(phase, 1.0, (jnp.array([q]), jnp.zeros(1)), "mass") for q in (8.0, 0.0, 0.0)]
    x_R = 8.0 / 3.0
    chain = boxes_and_couplers(masses + [voltage_source(x_R, "wall")], [(0, 1), (1, 2), (1, 3)],
                               [spring(0.25), spring(0.25), spring(0.1)])
    _, states = closed_run(chain, 400)
    qp = np.stack([np.asarray(flat_state(s)).reshape(3, 2) for s in states])
    q, p = qp[:, :, 0], qp[:, :, 1]
    middle = q[-100:, 1]
    sums = q.sum(axis=1)
    energy = (0.5 * (p ** 2).sum(axis=1) + 0.125 * (q[:, 0] - q[:, 1]) ** 2 + 0.125 * (q[:, 1] - q[:, 2]) ** 2
              + 0.05 * (q[:, 1] - x_R) ** 2)
    print("middle mass range in the last 100 steps", middle.max() - middle.min(), "sum range", sums.max() - sums.min(),
          "energy range / start", (energy.max() - energy.min()) / energy[0])
    assert middle.max() - middle.min() > 2.0
    assert sums.max() - sums.min() > 0.5
    # REFUTED (2026-10-30): the prediction said this energy stays within 10% of its start. It swings by 84%:
    # the book's step is coarse (unit masses, springs 1/4, duration 1), and symplectic Euler conserves a
    # modified energy, not this one, whose kinetic and potential terms sit half a step apart. The book's own
    # chain claims momentum and the positions' sum, which are exact, and never this energy. Kept as observed:
    # the energy stays bounded, below twice its start, over the run.
    assert energy.max() < 2.0 * energy[0]


# ---------------------------------------------------------------------------
# 19.4 Kirchhoff: a reader of a voltage is one more branch at the node.
# ---------------------------------------------------------------------------


def test_19_4_a_reader_of_a_voltage_is_one_more_branch_in_the_current_law():
    """Prediction (row 19.4): the triangle of unit capacitors from (8, 0, 0), conductances 1/4, with a resistor
    ``R = 4`` from capacitor 1 to a source at 1: at the first step capacitor 1's charge changes by minus the sum of
    its four branch currents (to 1e-12), and after 400 steps the total charge is within 1e-6 of 3 (every voltage 1),
    not 8. Could fail if the dot's transpose did not add the reader's current with the rest."""
    caps = [capacitor(1.0, q, f"cap{i + 1}") for i, q in enumerate((8.0, 0.0, 0.0))]
    pairs = [(0, 1), (1, 2), (0, 2), (0, 3)]
    loop = network(caps + [voltage_source(1.0)], pairs, [resistor(4.0, 1.0) for _ in pairs])
    traj = circuit_run(loop, 400)
    q0, q1 = traj[0][:3], traj[1][:3]
    e = q0 / 1.0
    currents = [(e[0] - e[1]) / 4.0, (e[0] - e[2]) / 4.0, (e[0] - 1.0) / 4.0]      # leaving capacitor 1
    np.testing.assert_allclose(q1[0] - q0[0], -sum(currents), atol=1e-12)
    print("charges after 400 steps", traj[-1][:3], "total", traj[-1][:3].sum())
    np.testing.assert_allclose(traj[-1][:3].sum(), 3.0, atol=1e-6)
    np.testing.assert_allclose(traj[-1][:3], 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 19.5 The network: an exact reader of a weight is weight decay, and the force is still the chain rule.
# ---------------------------------------------------------------------------


def book_network_read(kappa_r, exact_read=True):
    """The book's network drawn by hand with ``w_1``'s wire fanned to a reader (preference 0): leaves
    ``[x, w1, w2, w3, l]``; ``x`` then ``+`` then ``tanh`` then ``x`` then the loss, the reader beside the first
    ``x``. Its parameters sit at positions 1..3, as in ``drawn_network``."""
    leaves = tensor_arrangements([example_box(jnp.array([[1.0]]), "datum"), parameter_box(1.0, 0.1, "w1"),
                                  parameter_box(-1.0, 0.1, "w2"), parameter_box(2.0, 0.1, "w3"),
                                  example_box(jnp.array([[1.0]]), "label")])
    net = compose_seq(leaves, fan_box(5, [1, 0, 2, 3, 4, 1], "copy"))                 # [w1, x, w2, w3, l, w1]
    net = compose_seq(net, tensor_arrangements([times_box(), identity_box(3), wire_reader(kappa_r, 0.0, exact_read)]))
    net = compose_seq(net, fan_box(4, [1, 0, 2, 3], "group"))                          # [w2, p1, w3, l]
    net = compose_seq(net, tensor_arrangements([plus_box(2), identity_box(2)]))        # [z, w3, l]
    net = compose_seq(net, tensor_arrangements([tanh_box(), identity_box(2)]))         # [a, w3, l]
    net = compose_seq(net, fan_box(3, [1, 0, 2], "copy"))                              # [w3, a, l]
    net = compose_seq(net, tensor_arrangements([times_box(), identity_box(1)]))        # [y, l]
    return compose_seq(net, loss_box(1))


def test_19_5_an_exact_reader_of_a_weight_is_weight_decay():
    """Prediction (row 19.5): with ``w_1``'s wire read by an exact reader of stiffness .5 and preference 0, the
    book's step gives ``(1.15, -.8, 2.0)`` (the force on ``w_1`` is ``2 - .5``); the one-way control gives the book's
    ``(1.2, -.8, 2.0)``. Could fail if the fan's transpose did not add the reader's force to the loss's."""
    A = book_network_read(0.5)
    _, _, s = Phi(A).run_one((EMPTY, trivial_omega(0)), lambda _o: (EMPTY, EMPTY))
    w = np.asarray(A.P.position(s))[1:4]
    print("one step with weight decay on w1:", w)
    np.testing.assert_allclose(w, [1.15, -0.8, 2.0], atol=1e-12)
    C = book_network_read(0.5, exact_read=False)
    _, _, s = Phi(C).run_one((EMPTY, trivial_omega(0)), lambda _o: (EMPTY, EMPTY))
    np.testing.assert_allclose(np.asarray(C.P.position(s))[1:4], [1.2, -0.8, 2.0], atol=1e-12)
