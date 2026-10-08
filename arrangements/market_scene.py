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
