"""A figurine as a net over the kit, its belief a market over its parameter values (book plan P37
(b)(iii′) stage (A); ``AGENTDOCPLAN.md`` §8.2).

David (book repo, 2026-10-08): "I should be able to modularly plop in figurines, have some kind of
specification language for them, and then be able to put them in and that should work right ... it should
be fully compositional"; and, mid-run: "give P34's net ports their backward direction ... enlarge the kit by a
reciprocal and an exponential as kit boxes, each checked against its Newton or Euler term; write one figurine
as such a net, its belief a market over its parameter values by the belief integrator; run it through the
runner, paid by the outcome's likelihood, against P36's direct Bayes." Predictions fixed before this file
existed: the book repo's ``notes/runs/p37-market-of-scenes-3819a169/PREDICTIONS.md``; the tests are
``tests/test_market_runner.py``.

THE FIGURINE (``figurine_net``) is item 65's driver (``figurines.run_driver``) written as a net: its guess in
minutes at readiness rate ``lam`` is the closed form of that run, ``g(lam) = (1 - (1 - dt lam)^K) / lam``
(``K = 40`` half-minute steps; the power by six squarings and a multiply), with ``lam = exp(theta)`` through the
kit's ``exp`` box and the division through its ``recip``. Its BELIEF is a ``market`` over ``n`` values of the
rate, the belief integrator of P36 (b) as a kit leaf: it presents the belief ``q`` and steps by Bayes' rule on
the covector it receives. The figurine presents the mixture guess ``G = sum_i q_i g_i`` at its output, hears
the outcome ``x`` and a flag ``f`` at two ports (inboxes), and carries the likelihood as its cost, gated by the
flag: ``k = f sum_i q_i (x - g_i)^2 / 2`` (bell noise of variance 1). The cost's covector on the belief's wires
is ``-f (x - g_i)^2 / 2 = f log Pr(x | lam_i)`` up to a constant, so the market's step is Bayes' rule exactly;
its covector on the outcome's wire, ``f (G - x)``, the belief's guess minus the reading, is what the inbox
returns on its port the next step (P36's pushback on the observation).

THE WORLD (``world_net``) is two drums, the outcomes and the flags, presented. ``couple`` wires the world's two
outputs to the figurine's two ports; the figurine's guess goes nowhere. ``couple_net`` is the same system as one
closed net (relays where the ports were), which the runner is handed.

TIMING. The world presents outcome ``t`` at step ``t`` (from 1); the figurine's inbox holds it at step ``t + 1``,
when its cost pays the belief; so after step ``t + 1`` the belief has seen outcomes ``1..t``. The covector the
figurine returns lands on the drum's front cell after that cell is presented and rotates away with it; within
one pass of the drum no later round reads a cell the belief pushed on.

THE TWO BOXES AGAINST THEIR TERMS: ``euler_exp_net(v, N)`` is a register ``y_0 = 1`` pushed by ``(v / N) y`` a
step, so ``N`` steps give ``(1 + v/N)^N``; ``newton_recip_net(v, y_0)`` a register pushed by ``y (1 - v y)``, Newton's
iteration for ``1/v``. ``tank_pair`` splits P34's two tanks into a tank with the pipe's cost, hearing the other's
level at a port, and a bare tank: the pair whose coupling by the library is checked against the one net wired
by hand, values and forces.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Sequence

from . import runner as R

DT, K_STEPS = 0.5, 40


# ---------------------------------------------------------------------------
# The two boxes against their terms.
# ---------------------------------------------------------------------------


def euler_exp_net(v: float, N: int) -> R.Net:
    """``y' = y + (v / N) y`` from ``y_0 = 1``: after ``N`` steps, ``(1 + v/N)^N``."""
    n = R.Net()
    y = n.register(1.0)
    n.push(n.mul(n.const(v / N), y), y)
    n.out(y)
    return n


def newton_recip_net(v: float, y0: float) -> R.Net:
    """``y' = y + y (1 - v y)``: Newton's iteration for ``1/v`` from ``y_0``."""
    n = R.Net()
    y = n.register(y0)
    one, neg, c = n.const(1.0), n.const(-1.0), n.const(v)
    n.push(n.mul(y, n.add(one, n.mul(neg, n.mul(c, y)))), y)
    n.out(y)
    return n


def exp_cost_net(v: float) -> R.Net:
    """A register at ``v`` under the cost ``exp(x)``: one step moves it by ``-e^v``."""
    n = R.Net()
    x = n.register(v)
    n.cost(n.exp(x))
    return n


def recip_cost_net(v: float) -> R.Net:
    """A register at ``v`` under the cost ``1/x``: one step moves it by ``+1/v^2``."""
    n = R.Net()
    x = n.register(v)
    n.cost(n.recip(x))
    return n


# ---------------------------------------------------------------------------
# Two tanks as two nets with ports.
# ---------------------------------------------------------------------------


def tank_pair(v1=3, v2=1, kappa=Fraction(1, 20), x0=1):
    """Net A: tank 1 and the pipe's cost ``(kappa/2)(v1 - x)^2``, ``x`` an inbox hearing tank 2; net B: tank
    2 bare. Returns ``(A, B, a_to_b, b_to_a)``: B hears nothing, A's inbox hears B's output 0."""
    A = R.Net()
    v = A.register(v1)
    x = A.inbox(x0)
    neg, hk = A.const(-1), A.const(kappa / 2)
    d = A.add(v, A.mul(neg, x))
    A.cost(A.mul(hk, A.mul(d, d)))
    A.out(v)
    B = R.Net()
    u = B.register(v2)
    B.out(u)
    return A, B, [], [0]


# ---------------------------------------------------------------------------
# The figurine and the world.
# ---------------------------------------------------------------------------


def driver_guess(lam: float, dt: float = DT, K: int = K_STEPS) -> float:
    """The closed form of ``figurines.run_driver``: ``(1 - (1 - dt lam)^K) / lam``."""
    return (1 - (1 - dt * lam) ** K) / lam


def _guess_wires(n: R.Net, theta: float, dt: float, K: int):
    """Wires computing ``g(exp(theta))`` in the net: ``lam`` by the exp box, the power by squarings, the
    division by the recip box. Returns the guess's wire."""
    assert K == 40, "the squaring ladder below is written for K = 40"
    lam = n.exp(n.const(theta))
    a = n.add(n.const(1.0), n.mul(n.const(-dt), lam))       # 1 - dt lam
    a2 = n.mul(a, a)
    a4 = n.mul(a2, a2)
    a8 = n.mul(a4, a4)
    a16 = n.mul(a8, a8)
    a32 = n.mul(a16, a16)
    a40 = n.mul(a32, a8)
    top = n.add(n.const(1.0), n.mul(n.const(-1.0), a40))     # 1 - a^40
    return n.mul(top, n.recip(lam))


def figurine_net(lams: Sequence[float], prior: Sequence[float], dt: float = DT, K: int = K_STEPS):
    """The driver figurine with a market over the rates ``lams`` (prior ``prior``). Returns the net and the
    wires of interest: the market's slots, the guesses, the mixture guess and the two inboxes."""
    n = R.Net()
    q = n.market([math.log(p) for p in prior])
    x, f = n.inbox(0.0), n.inbox(0.0)
    neg, half = n.const(-1.0), n.const(0.5)
    gs = [_guess_wires(n, math.log(lam), dt, K) for lam in lams]
    G = None
    for qi, gi in zip(q, gs):
        t = n.mul(qi, gi)
        G = t if G is None else n.add(G, t)
    n.out(G)
    k = None
    for qi, gi in zip(q, gs):
        d = n.add(x, n.mul(neg, gi))
        term = n.mul(qi, n.mul(half, n.mul(d, d)))
        k = term if k is None else n.add(k, term)
    n.cost(n.mul(f, k))
    return n, dict(q=q, g=gs, G=G, x=x, f=f)


def world_net(outcomes: Sequence[float], flags: Sequence[float] = None) -> R.Net:
    """Two drums, the outcomes and the flags (all 1 unless given), both presented."""
    n = R.Net()
    xs = n.drum([float(v) for v in outcomes])
    fs = n.drum([1.0] * len(outcomes) if flags is None else [float(v) for v in flags])
    n.out(xs)
    n.out(fs)
    return n


def direct_bayes(lams: Sequence[float], prior: Sequence[float], outcomes: Sequence[float]):
    """P36's direct Bayes: the belief after each outcome, ``q p / <q, p>`` with
    ``p_i = exp(-(x - g_i)^2 / 2)``, and the mixture guess before each."""
    g = [driver_guess(l) for l in lams]
    q = [float(p) for p in prior]
    beliefs, guesses = [list(q)], []
    for x in outcomes:
        guesses.append(sum(qi * gi for qi, gi in zip(q, g)))
        w = [qi * math.exp(-0.5 * (x - gi) ** 2) for qi, gi in zip(q, g)]
        s = sum(w)
        q = [wi / s for wi in w]
        beliefs.append(list(q))
    return beliefs, guesses, g


LAMS = (0.1, 0.15, 0.2, 0.3, 0.5)
PRIOR = (0.2, 0.2, 0.2, 0.2, 0.2)
OUTCOMES = (6.0, 5.5, 6.5, 6.0, 5.8, 6.2)


def figurine_and_world(lams=LAMS, prior=PRIOR, outcomes=OUTCOMES):
    """The figurine net, the world net, and the coupling ``(a_to_b, b_to_a)``: the world's two outputs to
    the figurine's two ports."""
    fig, wires = figurine_net(lams, prior)
    world = world_net(outcomes)
    return fig, wires, world, [], [0, 1]


# ===========================================================================
# Stage (B): a scene of two figurines, the coordinator a market over the one-part-altered scenes.
# ===========================================================================
#
# David's snake and cat (2026-10-08: "what a snake and a cat might do in a scene together"). The snake at ``s``
# moves toward the cat with aggressiveness ``a``; the cat at ``c`` keeps a preferred distance ``d`` from the snake
# with wariness ``b``; one inner step is ``s' = s + a (c - s)``, ``c' = c + b (s + d - c)``. A scene starts at
# ``s_0 = 0``, ``c_0 = d_0`` and runs ``T`` inner steps UNROLLED INSIDE ONE STEP OF THE NET: the scene is imagined,
# so the chain rule passes through its whole run. Its outcome is the final distance ``D = c_T - s_T``; a sighting
# sees the snake's final position ``s_T``. The world is a recording (drums): each round a starting distance, the
# true scene's ``D*`` and ``s*``, the true snake's ``D_2*`` with a second cat, and a flag.
#
# THE FAMILY is the one-part-altered family of the starting base: the snake altered over ``A_VALUES`` with the cat
# at ``B0``, the cat altered over ``B_VALUES`` with the snake at ``A0``. THE COORDINATOR is one ``market`` over the
# six scenes, paid by the likelihood of the recording under each; the nested form (an outer trust over the two
# families, inner trusts within, the prediction market dynamic operad's ``mu o nu``) is ``nested_market`` in the
# library. THE PUSH ARMS move the base parameters (registers, log-parameters) by the loss on the base scene's
# outcome: bare (the chain rule through the run, both pushed) or routed by trust (the base scene run twice, each
# copy with one parameter an insulated relay copy, each copy's loss weighted by the trust in that part's family
# read through a relay). Predictions: the book repo's ``notes/runs/p37-market-of-scenes-3819a169/PREDICTIONS.md``,
# stage (B); the by-hand model is ``scene_closed_form`` below, written with them.

from typing import NamedTuple

T_INNER, D_PREF, SIGMA_B = 5, 3.0, 0.25
A_VALUES, B_VALUES = (0.1, 0.2, 0.3), (0.2, 0.4, 0.6)
A0, B0 = 0.3, 0.2            # the base: the snake right, the cat wrong
A_TRUE, B_TRUE, B2 = 0.3, 0.4, 0.6
D0S = (5.0, 2.0, 6.0, 4.0, 3.0, 7.0)
ETA = 0.01


class SceneArm(NamedTuple):
    sighting: bool = False
    comparison: bool = False
    push: str = "none"       # "none", "bare", "routed"


SCENE_ARMS = {
    "O": SceneArm(), "S": SceneArm(sighting=True), "C": SceneArm(comparison=True),
    "O-bare": SceneArm(push="bare"), "O-routed": SceneArm(push="routed"),
}


def scene_by_hand(a, b, d0, T=T_INNER, d=D_PREF):
    """The scene's recurrence in plain arithmetic: ``(D, s_T)``."""
    s, c = 0.0, d0
    for _ in range(T):
        s, c = s + a * (c - s), c + b * (s + d - c)
    return c - s, s


def scene_wires(n: R.Net, a, b, d0, T=T_INNER, d=D_PREF):
    """The scene unrolled in the net from the wires ``a``, ``b`` (parameters) and ``d0``: ``(D, s_T)`` wires."""
    neg, dc = n.const(-1.0), n.const(d)
    s, c = n.const(0.0), d0
    for _ in range(T):
        gap = n.add(c, n.mul(neg, s))                 # c - s
        want = n.add(n.add(s, dc), n.mul(neg, c))     # s + d - c
        s, c = n.add(s, n.mul(a, gap)), n.add(c, n.mul(b, want))
    return n.add(c, n.mul(neg, s)), s


def scene_family(A=A_VALUES, B=B_VALUES, a0=A0, b0=B0):
    """``(who, a, b)`` for the six one-part-altered scenes: the snake's three, then the cat's three."""
    return [("snake", a, b0) for a in A] + [("cat", a0, b) for b in B]


def scene_world_net(d0s=D0S, a_true=A_TRUE, b_true=B_TRUE, b2=B2) -> R.Net:
    """The recording: drums of ``d_0``, ``D*``, ``s*``, ``D_2*`` and the flag, presented in that order."""
    n = R.Net()
    truths = [scene_by_hand(a_true, b_true, d0) for d0 in d0s]
    second = [scene_by_hand(a_true, b2, d0)[0] for d0 in d0s]
    for cells in ([float(x) for x in d0s], [D for D, _ in truths], [s for _, s in truths], second, [1.0] * len(d0s)):
        n.out(n.drum(cells))
    return n


def scene_market_net(arm: SceneArm, A=A_VALUES, B=B_VALUES, a0=A0, b0=B0, b2=B2, sigma=SIGMA_B, eta=ETA,
                     T=T_INNER, prior=None):
    """The figurine world as one net with five ports (``d_0``, ``D*``, ``s*``, ``D_2*``, flag). Returns the net
    and the wires of interest."""
    n = R.Net()
    d0, Dst, sst, D2st, flag = (n.inbox(0.0) for _ in range(5))
    neg, c_half = n.const(-1.0), n.const(0.5 / sigma ** 2)
    fam = scene_family(A, B, a0, b0)
    if prior is None:
        prior = [0.5 / len(A)] * len(A) + [0.5 / len(B)] * len(B)
    q = n.market([math.log(p) for p in prior])
    a0_w, b0_w = n.exp(n.const(math.log(a0))), n.exp(n.const(math.log(b0)))
    A_w = [n.exp(n.const(math.log(a))) for a in A]
    B_w = [n.exp(n.const(math.log(b))) for b in B]
    params = [(aw, b0_w) for aw in A_w] + [(a0_w, bw) for bw in B_w]
    outs = [scene_wires(n, aw, bw, d0, T) for aw, bw in params]

    def sq_err(target, wire):
        e = n.add(target, n.mul(neg, wire))
        return n.mul(c_half, n.mul(e, e))

    def weighted_sum(terms):
        tot = None
        for qi, t in terms:
            x = n.mul(qi, t)
            tot = x if tot is None else n.add(tot, x)
        return tot

    k = weighted_sum([(qi, sq_err(Dst, D)) for qi, (D, _) in zip(q, outs)])
    if arm.sighting:
        k = n.add(k, weighted_sum([(qi, sq_err(sst, s)) for qi, (_, s) in zip(q, outs)]))
    if arm.comparison:
        b2_w = n.exp(n.const(math.log(b2)))
        D2_cat = scene_wires(n, a0_w, b2_w, d0, T)[0]        # the cat-altered scenes share the base's second scene
        D2 = [scene_wires(n, aw, b2_w, d0, T)[0] for aw in A_w] + [D2_cat] * len(B)
        k = n.add(k, weighted_sum([(qi, sq_err(D2st, D2i)) for qi, D2i in zip(q, D2)]))
    n.cost(n.mul(flag, k))
    wires = dict(q=q, outs=outs, d0=d0, Dst=Dst, sst=sst, D2st=D2st, flag=flag)
    if arm.push != "none":
        th_s, th_c = n.register(math.log(a0)), n.register(math.log(b0))
        a_live, b_live = n.exp(th_s), n.exp(th_c)
        eta_w = n.const(eta)
        if arm.push == "bare":
            D, _ = scene_wires(n, a_live, b_live, d0, T)
            n.cost(n.mul(flag, n.mul(eta_w, sq_err(Dst, D))))
        else:
            xb, _, ear_b = n.relay(b0)
            n.push(b_live, ear_b)                              # an insulated copy of b, one step late
            xa, _, ear_a = n.relay(a0)
            n.push(a_live, ear_a)
            D_S, _ = scene_wires(n, a_live, xb, d0, T)        # the snake's copy: its parameter live
            D_C, _ = scene_wires(n, xa, b_live, d0, T)        # the cat's copy
            mu_s = n.add(n.add(q[0], q[1]), q[2])
            mu_c = n.add(n.add(q[3], q[4]), q[5])
            ws, _, ear_s = n.relay(0.5)
            n.push(mu_s, ear_s)                                # the trust in the snake's family, read through a relay
            wc, _, ear_c = n.relay(0.5)
            n.push(mu_c, ear_c)
            routed = n.add(n.mul(ws, sq_err(Dst, D_S)), n.mul(wc, sq_err(Dst, D_C)))
            n.cost(n.mul(flag, n.mul(eta_w, routed)))
        wires.update(th_s=th_s, th_c=th_c)
    return n, wires


def scene_closed_form(arm: SceneArm, A=A_VALUES, B=B_VALUES, a0=A0, b0=B0, a_true=A_TRUE, b_true=B_TRUE,
                      b2=B2, d0s=D0S, sigma=SIGMA_B, eta=ETA, T=T_INNER):
    """The by-hand model: the flat trust after each round, and the base log-parameters after each round under
    the push arm (the pushes by ``jax.grad`` of the same recurrence; routed by the trust after ``r - 2`` payments).
    Returns ``dict(trust=[q_0, ..., q_R], theta=[th_0, ..., th_R])`` (theta only with a push arm)."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    fam = scene_family(A, B, a0, b0)
    nA = len(A)
    q = np.array([0.5 / nA] * nA + [0.5 / len(B)] * len(B))
    trust = [q.copy()]
    lik = lambda x, y: np.exp(-0.5 * ((x - y) / sigma) ** 2)
    for d0 in d0s:
        Dst, sst = scene_by_hand(a_true, b_true, d0, T)
        D2st = scene_by_hand(a_true, b2, d0, T)[0]
        p = np.ones(len(fam))
        for i, (who, a, b) in enumerate(fam):
            D, s = scene_by_hand(a, b, d0, T)
            p[i] *= lik(Dst, D)
            if arm.sighting:
                p[i] *= lik(sst, s)
            if arm.comparison:
                D2 = scene_by_hand(a if who == "snake" else a0, b2, d0, T)[0]
                p[i] *= lik(D2st, D2)
        q = q * p / (q @ p)
        trust.append(q.copy())
    out = dict(trust=trust)
    if arm.push != "none":
        def D_of(th, d0):
            s, c = 0.0, d0
            a, b = jnp.exp(th[0]), jnp.exp(th[1])
            for _ in range(T):
                s, c = s + a * (c - s), c + b * (s + D_PREF - c)
            return c - s
        loss = lambda th, d0, Dst: 0.5 * ((Dst - D_of(th, d0)) / sigma) ** 2
        th = jnp.array([math.log(a0), math.log(b0)])
        thetas = [th]
        for r, d0 in enumerate(d0s, 1):
            Dst = scene_by_hand(a_true, b_true, d0, T)[0]
            if arm.push == "bare":
                th = th - eta * jax.grad(loss)(th, d0, Dst)
            else:
                # each copy holds the other part's parameter through a relay, one step staler than the live one:
                # at round r the live parameter has taken r - 1 pushes, the copy r - 2; so has the trust
                lag = thetas[max(0, r - 2)]
                g_s = jax.grad(loss)(jnp.array([th[0], lag[1]]), d0, Dst)[0]
                g_c = jax.grad(loss)(jnp.array([lag[0], th[1]]), d0, Dst)[1]
                mu_c = trust[max(0, r - 2)][nA:].sum()
                th = th - eta * jnp.array([(1 - mu_c) * g_s, mu_c * g_c])
            thetas.append(th)
        out["theta"] = [np.array(t) for t in thetas]
    return out


def nested_market(gammas, prior_outer=(0.5, 0.5), nA=3, nB=3):
    """The prediction market dynamic operad's nested form in the library: an outer belief over the two families
    and an inner belief in each (three belief integrators, tensored), paid by one likelihood box whose cost is
    ``-sum_j mu_j log sum_v sg(nu_jv) gamma_jv - sum_j sum_v nu_jv log gamma_jv``, the inner beliefs read one-way
    in the outer's term (``stop_gradient``): the outer is paid by each family's aggregate likelihood, the inner by
    its own. ``gammas[r]`` are the six likelihoods of round ``r``. Returns the flat trust ``mu o nu`` after each
    round, from the uniform inner prior."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from .arrangement import Arrangement, exact
    from .belief import belief_coalgebra, belief_of
    from .functors import Phi
    from .interpretation import trivial_omega
    from .wiring import compose_seq, trivial_parameter

    n_tot = nA + nB
    P = belief_coalgebra(2, jnp.log(jnp.asarray(prior_outer))).tensor(
        belief_coalgebra(nA, jnp.zeros(nA))).tensor(belief_coalgebra(nB, jnp.zeros(nB)))
    beliefs = Arrangement(
        P=P, in_dim_M=0, out_dim_M=n_tot, in_dim_N=0, out_dim_N=2 + n_tot + n_tot,
        out_f=lambda y, g: jnp.concatenate([y, g]), in_f=lambda y, g, n_in: jnp.zeros(0),
        omega=lambda y, g, n_in: (jnp.zeros(2 + n_tot), jnp.zeros(n_tot), jnp.zeros(0)), label="beliefs")

    def cost(q, w, n_in):
        mu, nu_s, nu_c, g = w[:2], w[2:2 + nA], w[2 + nA:2 + n_tot], w[2 + n_tot:]
        g_s, g_c = g[:nA], g[nA:]
        agg = jnp.stack([jnp.dot(jax.lax.stop_gradient(nu_s), g_s), jnp.dot(jax.lax.stop_gradient(nu_c), g_c)])
        return -jnp.dot(mu, jnp.log(agg)) - jnp.dot(nu_s, jnp.log(g_s)) - jnp.dot(nu_c, jnp.log(g_c))

    pay = Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=2 + n_tot + n_tot, in_dim_N=0, out_dim_N=0,
                      out_f=lambda q, w: jnp.zeros(0), in_f=lambda q, w, n_in: jnp.zeros(0),
                      omega=exact(lambda q, w, n_in: -cost(q, w, n_in)), label="pay")
    O = Phi(compose_seq(beliefs, pay))
    s = O.state
    out = []

    def flat(s):
        (l_mu, l_s), l_c = s[0]   # the composite's state is (the beliefs', the likelihood box's)
        mu, nu_s, nu_c = belief_of(l_mu), belief_of(l_s), belief_of(l_c)
        return np.concatenate([np.asarray(mu[0] * nu_s), np.asarray(mu[1] * nu_c)])

    out.append(flat(s))
    for g in gammas:
        _, _, s = O.with_state(s).run_one((jnp.asarray(g, dtype=float), trivial_omega(0)),
                                          lambda _o: (jnp.zeros(0), jnp.zeros(0)))
        out.append(flat(s))
    return out


def nested_market_composed(arm: SceneArm, prior_outer=(0.5, 0.5), A=A_VALUES, B=B_VALUES, a0=A0, b0=B0,
                           a_true=A_TRUE, b_true=B_TRUE, b2=B2, d0s=D0S, sigma=SIGMA_B, T=T_INNER):
    """``nested_market`` with its likelihoods made by boxes (plan P43 (b) job 7, Claude Opus 5.5, de71554a):
    a box stepping through the rounds' starting distances ``d_0`` (the environment's schedule); the world, a
    gray box running the true scene from ``d_0`` (and the second scene of the comparison); a likelihood box per
    scene, running that scene from the same ``d_0`` and reading the world's outcome by a Gaussian; and the
    nested market (beliefs and the paying box) hearing the six likelihoods. One closed arrangement, run by
    ``Phi``. Before, ``scene_likelihoods`` computed the likelihoods in numpy and fed them in. Returns the flat
    trust after each round, as ``nested_market``."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from .arrangement import Arrangement, exact
    from .belief import belief_coalgebra, belief_of
    from .functors import Phi
    from .interpretation import trivial_omega
    from .network import example_box, fan_box, map_box
    from .wiring import compose_seq, tensor_arrangements, trivial_parameter

    def scene(a, b, d0):
        s, c = 0.0, d0
        for _ in range(T):
            s, c = s + a * (c - s), c + b * (s + D_PREF - c)
        return c - s, s

    def world(v):
        D, sT = scene(a_true, b_true, v[0])
        return jnp.stack([v[0], D, sT, scene(a_true, b2, v[0])[0]])

    lik = lambda x, y: jnp.exp(-0.5 * ((x - y) / sigma) ** 2)
    fam = scene_family(A, B, a0, b0)

    def likelihood_box(who, a, b):
        def g(v):
            d0, Dst, sst, D2st = v[0], v[1], v[2], v[3]
            D, sT = scene(a, b, d0)
            p = lik(Dst, D)
            if arm.sighting:
                p = p * lik(sst, sT)
            if arm.comparison:
                p = p * lik(D2st, scene(a if who == "snake" else a0, b2, d0)[0])
            return jnp.reshape(p, (1,))
        return map_box(g, 4, 1, f"likelihood({who}, {a:g}, {b:g})")

    nA, nB = len(A), len(B)
    n_tot = nA + nB
    rounds = example_box(jnp.asarray(d0s, dtype=float).reshape(-1, 1), "d0")
    seen = compose_seq(compose_seq(rounds, map_box(world, 1, 4, "world")),
                       fan_box(4, list(range(4)) * n_tot, "copy"))
    gammas = compose_seq(seen, tensor_arrangements([likelihood_box(*f) for f in fam]))

    P = belief_coalgebra(2, jnp.log(jnp.asarray(prior_outer))).tensor(
        belief_coalgebra(nA, jnp.zeros(nA))).tensor(belief_coalgebra(nB, jnp.zeros(nB)))
    beliefs = Arrangement(
        P=P, in_dim_M=0, out_dim_M=n_tot, in_dim_N=0, out_dim_N=2 + n_tot + n_tot,
        out_f=lambda y, g: jnp.concatenate([y, g]), in_f=lambda y, g, n_in: jnp.zeros(0),
        omega=lambda y, g, n_in: (jnp.zeros(2 + n_tot), jnp.zeros(n_tot), jnp.zeros(0)), label="beliefs")

    def cost(q, w, n_in):
        mu, nu_s, nu_c, g = w[:2], w[2:2 + nA], w[2 + nA:2 + n_tot], w[2 + n_tot:]
        g_s, g_c = g[:nA], g[nA:]
        agg = jnp.stack([jnp.dot(jax.lax.stop_gradient(nu_s), g_s), jnp.dot(jax.lax.stop_gradient(nu_c), g_c)])
        return -jnp.dot(mu, jnp.log(agg)) - jnp.dot(nu_s, jnp.log(g_s)) - jnp.dot(nu_c, jnp.log(g_c))

    pay = Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=2 + n_tot + n_tot, in_dim_N=0, out_dim_N=0,
                      out_f=lambda q, w: jnp.zeros(0), in_f=lambda q, w, n_in: jnp.zeros(0),
                      omega=exact(lambda q, w, n_in: -cost(q, w, n_in)), label="pay")
    closed = compose_seq(gammas, compose_seq(beliefs, pay))
    O = Phi(closed)
    s = O.state
    z = jnp.zeros(0)

    def flat(s):
        (l_mu, l_s), l_c = s[1][0]   # (the likelihood stage's, (the beliefs', the paying box's))
        mu, nu_s, nu_c = belief_of(l_mu), belief_of(l_s), belief_of(l_c)
        return np.concatenate([np.asarray(mu[0] * nu_s), np.asarray(mu[1] * nu_c)])

    out = [flat(s)]
    for _ in d0s:
        _, _, s = O.with_state(s).run_one((z, trivial_omega(0)), lambda _o: (z, z))
        out.append(flat(s))
    return out


def scene_likelihoods(arm: SceneArm, A=A_VALUES, B=B_VALUES, a0=A0, b0=B0, a_true=A_TRUE, b_true=B_TRUE, b2=B2,
                      d0s=D0S, sigma=SIGMA_B, T=T_INNER):
    """The six scenes' likelihoods of the recording, round by round (what pays the market)."""
    import numpy as np
    fam = scene_family(A, B, a0, b0)
    lik = lambda x, y: np.exp(-0.5 * ((x - y) / sigma) ** 2)
    rows = []
    for d0 in d0s:
        Dst, sst = scene_by_hand(a_true, b_true, d0, T)
        D2st = scene_by_hand(a_true, b2, d0, T)[0]
        p = np.ones(len(fam))
        for i, (who, a, b) in enumerate(fam):
            D, s = scene_by_hand(a, b, d0, T)
            p[i] *= lik(Dst, D)
            if arm.sighting:
                p[i] *= lik(sst, s)
            if arm.comparison:
                p[i] *= lik(D2st, scene_by_hand(a if who == "snake" else a0, b2, d0, T)[0])
        rows.append(p)
    return rows


# ===========================================================================
# Stage (A′): the same-step port (David, 2026-10-09: "build the same step kind if we can").
# ===========================================================================
#
# A ``pcost`` sink's forcing reads the port the same step, so a figurine's cost can read what the world presents
# now, with no flag and no lag, and the covector it returns on the port lands this step. The inbox stays for a
# value that must be presented or stored.


def tank_pair_same_step(v1=3, v2=1, kappa=Fraction(1, 20)):
    """Net A: tank 1 with the pipe's cost ``(kappa/2)(v1 - x)^2`` written as the polynomial
    ``(kappa/2) v1^2 - kappa v1 x + (kappa/2) x^2`` in the port's value ``x``; net B: tank 2 bare."""
    A = R.Net()
    v = A.register(v1)
    A.inbox(v2)                                   # the port; its register is left unread
    hk, nk = A.const(kappa / 2), A.const(-kappa)
    c0 = A.mul(hk, A.mul(v, v))
    c1 = A.mul(nk, v)
    A.pcost([c0, c1, hk], 0)
    A.out(v)
    B = R.Net()
    u = B.register(v2)
    B.out(u)
    return A, B, [], [0]


def figurine_net_same_step(lams: Sequence[float], prior: Sequence[float], dt: float = DT, K: int = K_STEPS):
    """The driver figurine with a market over the rates, its likelihood ``sum_i q_i (x - g_i)^2 / 2`` as the
    polynomial ``(½ sum q_i g_i^2) - (sum q_i g_i) x + (½ sum q_i) x^2`` in the outcome heard at its one port."""
    n = R.Net()
    q = n.market([math.log(p) for p in prior])
    n.inbox(0.0)                                  # the outcome's port
    half = n.const(0.5)
    gs = [_guess_wires(n, math.log(lam), dt, K) for lam in lams]
    G = H = Q = None
    for qi, gi in zip(q, gs):
        t = n.mul(qi, gi)
        G = t if G is None else n.add(G, t)
        h = n.mul(half, n.mul(qi, n.mul(gi, gi)))
        H = h if H is None else n.add(H, h)
        s = n.mul(half, qi)
        Q = s if Q is None else n.add(Q, s)
    n.out(G)
    n.pcost([H, n.mul(n.const(-1.0), G), Q], 0)
    return n, dict(q=q, g=gs, G=G)


def world_net_outcomes(outcomes: Sequence[float] = OUTCOMES) -> R.Net:
    """One drum of outcomes, presented."""
    n = R.Net()
    n.out(n.drum([float(v) for v in outcomes]))
    return n
