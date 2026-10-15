"""The swap, piece 1: each part on its own evidence (S6) beside the fixed family (S0), and a trigger box that
stores each scene's summed miss (book plan P37 (b)(vii); ``AGENTDOCPLAN.md`` §8.2, "The build, decided").

Predictions fixed before this file existed: the book repo's ``notes/runs/p37-swap-build-4f4ee268/PREDICTIONS.md``;
the tests are ``tests/test_swap.py``; the hand design is ``notes/runs/p37-swap-396b8ee2/DESIGN.md`` (Claude Opus
5.5, 396b8ee2). Claude Opus 5.5, 4f4ee268, 2026-10-15.

THE SCENE is stage (B)'s snake and cat (``market_scene``): the gap steps ``g' = g (1 - a - b) + 3 b`` five inner
steps from ``d_0``; the outcome is the final gap ``D``.

S0 is stage (B)'s nested market over the six one-part-altered scenes of a start ``(a0, b0)``, outcome only, as one
closed arrangement (``market_scene.nested_market_composed``'s wiring), here with a box per scene that reads out
both its likelihood of the recording and its MISS ``D* - D``, so a trigger can be wired beside the market.

THE TRIGGER is a box whose state is each scene's summed miss and the count of rounds (a configuration on
``R^7``, response 1); its 1-form adds this round's misses and ``1`` to the state and is zero on its inputs, so it
reads the misses and the beliefs one-way. It reads out ``fire``: ``1`` when every scene of trust at least
``trusted`` has ``|sum| > 3 sigma sqrt(t)``, else ``0``; the trust it reads is the flat trust ``mu_j nu_jv``.

S6 is two belief integrators over the library, uniform priors. The snake's is paid by the comparison (the second
cat's outcome under each snake value); the cat's by ``log sum_a nu_s(a) gamma_ab``, the likelihood of the scene's
outcome under the snake's mixture, the snake's belief read one-way (the paying box's 1-form is the derivative of
its cost holding the snake's belief fixed in the cat's term). The boxes step together, so round ``t``'s cat is paid
with the snake's belief after ``t - 1`` comparisons. Each round the cat's term imagines every snake with every cat.

Everything is boxes composed by ``compose_seq`` and ``tensor_arrangements`` and run by ``Phi``: the rounds'
starting distances (and, in the noisy arm, the recording's noise) an example box, the environment's schedule;
the world a gray box; nothing computed between steps. ``*_by_hand`` are the numpy recurrences they are checked
against.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from .arrangement import Arrangement, exact, zero_form
from .belief import belief_coalgebra, belief_of
from .coalgebra import configuration
from .functors import Phi
from .interpretation import trivial_omega
from .network import example_box, fan_box, identity_box, map_box
from .rvect import euclidean
from .wiring import compose_seq, tensor_arrangements, trivial_parameter

T_INNER, D_PREF = 5, 3.0
A_VALUES, B_VALUES = (0.1, 0.2, 0.3), (0.2, 0.4, 0.6)
B2 = 0.6
D0S = (5.0, 2.0, 6.0, 4.0, 3.0, 7.0) * 2
PROBLEMS = {"P1": ((0.3, 0.2), (0.3, 0.4)), "P2": ((0.1, 0.2), (0.3, 0.4)), "P3": ((0.3, 0.2), (0.3, 0.5))}


def scene(a, b, d0, T=T_INNER):
    """The scene's final gap ``D``."""
    s, c = 0.0, d0
    for _ in range(T):
        s, c = s + a * (c - s), c + b * (s + D_PREF - c)
    return c - s


def family(a0, b0, A=A_VALUES, B=B_VALUES):
    """The six one-part-altered scenes of the start: the snake's three, then the cat's three."""
    return [(a, b0) for a in A] + [(a0, b) for b in B]


def _bell(sigma):
    return lambda x, y: jnp.exp(-0.5 * ((x - y) / sigma) ** 2)


def _rounds_box(d0s, noise):
    noise = np.zeros(len(d0s)) if noise is None else np.asarray(noise, float)
    return example_box(jnp.asarray(np.stack([np.asarray(d0s, float), noise], axis=1)), "rounds")


def _beliefs(P, n_heard, n_belief):
    """The beliefs: present their configurations and pass on what they hear."""
    return Arrangement(
        P=P, in_dim_M=0, out_dim_M=n_heard, in_dim_N=0, out_dim_N=n_belief + n_heard,
        out_f=lambda y, g: jnp.concatenate([y, g]), in_f=lambda y, g, n_in: jnp.zeros(0),
        omega=lambda y, g, n_in: (jnp.zeros(n_belief), jnp.zeros(n_heard), jnp.zeros(0)), label="beliefs")


def _closed_run(closed, rounds, read):
    """Run a closed arrangement ``<0|0> -> <0|k>`` by ``Phi`` for ``rounds`` steps; ``read(state, out)`` after each,
    with the output shown at that step (computed from the state before it)."""
    O = Phi(closed)
    s = O.state
    z = jnp.zeros(0)
    rows = []
    for _ in range(rounds):
        out, _, s = O.with_state(s).run_one((z, trivial_omega(0)),
                                            lambda o: (jnp.zeros_like(jnp.asarray(o[0])), z))
        rows.append(read(s, out))
    return rows


# ---------------------------------------------------------------------------
# S0, with the trigger beside it.
# ---------------------------------------------------------------------------


def s0_arrangement(a0, b0, truth, sigma, d0s=D0S, noise=None, trigger=True, trusted=0.1, A=A_VALUES, B=B_VALUES):
    """S0 as one closed arrangement; with ``trigger`` its external output is the trigger's ``fire``."""
    lik = _bell(sigma)
    fam = family(a0, b0, A, B)
    nA, nB = len(A), len(B)
    n = nA + nB

    def world(v):
        return jnp.stack([v[0], scene(*truth, v[0]) + v[1]])

    def scene_box(a, b):
        def g(v):
            D = scene(a, b, v[0])
            return jnp.stack([lik(v[1], D), v[1] - D])
        return map_box(g, 2, 2, f"scene({a:g}, {b:g})")

    seen = compose_seq(compose_seq(_rounds_box(d0s, noise), map_box(world, 2, 2, "world")),
                       fan_box(2, [0, 1] * n, "copy"))
    # each scene box reads out (gamma_i, miss_i); regroup as (gammas, misses)
    scenes = compose_seq(compose_seq(seen, tensor_arrangements([scene_box(*c) for c in fam])),
                         fan_box(2 * n, [2 * i for i in range(n)] + [2 * i + 1 for i in range(n)], "regroup"))
    P = belief_coalgebra(2, jnp.log(jnp.asarray([0.5, 0.5]))).tensor(
        belief_coalgebra(nA, jnp.zeros(nA))).tensor(belief_coalgebra(nB, jnp.zeros(nB)))
    nb = 2 + n
    beliefs = _beliefs(P, 2 * n, nb)

    def U(q, w, n_in):
        mu, nu_s, nu_c, g = w[:2], w[2:2 + nA], w[2 + nA:nb], w[nb:nb + n]
        agg = jnp.stack([jnp.dot(jax.lax.stop_gradient(nu_s), g[:nA]),
                         jnp.dot(jax.lax.stop_gradient(nu_c), g[nA:])])
        return jnp.dot(mu, jnp.log(agg)) + jnp.dot(nu_s, jnp.log(g[:nA])) + jnp.dot(nu_c, jnp.log(g[nA:]))

    pay = Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=nb + n, in_dim_N=0, out_dim_N=0,
                      out_f=lambda q, w: jnp.zeros(0), in_f=lambda q, w, n_in: jnp.zeros(0),
                      omega=exact(U), label="pay")
    if not trigger:
        tail = compose_seq(fan_box(nb + 2 * n, list(range(nb + n)), "drop misses"), pay)
        return compose_seq(scenes, compose_seq(beliefs, tail))

    def fire(q, w):
        mu, nu_s, nu_c, = w[:2], w[2:2 + nA], w[2 + nA:nb]
        trust = jnp.concatenate([mu[0] * nu_s, mu[1] * nu_c])
        sums, t = q[:n], q[n]
        past = jnp.abs(sums) > 3.0 * sigma * jnp.sqrt(jnp.maximum(t, 1.0))
        ok = jnp.all(jnp.where(trust >= trusted, past, True)) & (t > 0)
        return jnp.reshape(ok.astype(float), (1,))

    trig = Arrangement(
        P=configuration(euclidean(n + 1, 1.0)), in_dim_M=0, out_dim_M=nb + n, in_dim_N=0, out_dim_N=1,
        out_f=fire, in_f=lambda q, w, n_in: jnp.zeros(0),
        omega=lambda q, w, n_in: (jnp.concatenate([w[nb:nb + n], jnp.ones(1)]), jnp.zeros(nb + n), jnp.zeros(0)),
        label="trigger")
    split = fan_box(nb + 2 * n, list(range(nb + n)) + list(range(nb)) + list(range(nb + n, nb + 2 * n)), "split")
    return compose_seq(scenes, compose_seq(beliefs, compose_seq(split, tensor_arrangements([pay, trig]))))


def s0_run(problem, sigma, rounds=12, noise=None, trigger=True, trusted=0.1):
    """Run S0 on a problem: the flat trust after each round, and, with the trigger, each scene's summed miss, the
    count and the ``fire`` shown at each step (from the state before it)."""
    (a0, b0), truth = PROBLEMS[problem]
    arr = s0_arrangement(a0, b0, truth, sigma, noise=noise, trigger=trigger, trusted=trusted)

    def read(s, out):
        bel = s[1][0]                                   # (scenes' state, (beliefs', tail's))
        (l_mu, l_s), l_c = bel
        mu, nu_s, nu_c = belief_of(l_mu), belief_of(l_s), belief_of(l_c)
        row = dict(trust=np.concatenate([np.asarray(mu[0] * nu_s), np.asarray(mu[1] * nu_c)]))
        if trigger:
            tr = s[1][1][1][1]                          # (split's, (pay's, trigger's))
            row.update(sums=np.asarray(tr[:-1]), count=float(tr[-1]), fire=float(np.asarray(out[0])[0]))
        return row

    return _closed_run(arr, rounds + (1 if trigger else 0), read)


def s0_by_hand(problem, sigma, rounds=12, noise=None):
    """The flat trust after each round by Bayes' rule, and each scene's summed miss."""
    (a0, b0), truth = PROBLEMS[problem]
    fam = family(a0, b0)
    q = np.full(len(fam), 1.0 / len(fam))
    sums = np.zeros(len(fam))
    noise = np.zeros(rounds) if noise is None else np.asarray(noise, float)
    out = []
    for t in range(rounds):
        d0 = D0S[t]
        y = scene(*truth, d0) + noise[t]
        D = np.array([scene(a, b, d0) for a, b in fam])
        q = q * np.exp(-0.5 * ((y - D) / sigma) ** 2)
        q = q / q.sum()
        sums = sums + (y - D)
        out.append(dict(trust=q.copy(), sums=sums.copy()))
    return out


# ---------------------------------------------------------------------------
# S6: each part on its own evidence.
# ---------------------------------------------------------------------------


def s6_arrangement(truth, sigma, d0s=D0S, A=A_VALUES, B=B_VALUES, b2=B2):
    lik = _bell(sigma)
    nA, nB = len(A), len(B)

    def world(v):
        return jnp.stack([v[0], scene(*truth, v[0]), scene(truth[0], b2, v[0])])

    def comparison_box(a):
        return map_box(lambda v: jnp.reshape(lik(v[2], scene(a, b2, v[0])), (1,)), 3, 1, f"comparison({a:g})")

    def cat_box(a, b):
        return map_box(lambda v: jnp.reshape(lik(v[1], scene(a, b, v[0])), (1,)), 3, 1, f"scene({a:g}, {b:g})")

    boxes = [comparison_box(a) for a in A] + [cat_box(a, b) for b in B for a in A]
    seen = compose_seq(compose_seq(_rounds_box(d0s, None), map_box(world, 2, 3, "world")),
                       fan_box(3, [0, 1, 2] * len(boxes), "copy"))
    gammas = compose_seq(seen, tensor_arrangements(boxes))
    n_heard = nA + nA * nB
    P = belief_coalgebra(nA, jnp.zeros(nA)).tensor(belief_coalgebra(nB, jnp.zeros(nB)))
    beliefs = _beliefs(P, n_heard, nA + nB)

    def U(q, w, n_in):
        nu_s, nu_c, g = w[:nA], w[nA:nA + nB], w[nA + nB:]
        g2, gab = g[:nA], jnp.reshape(g[nA:], (nB, nA))        # row b, column a
        mix = gab @ jax.lax.stop_gradient(nu_s)
        return jnp.dot(nu_s, jnp.log(g2)) + jnp.dot(nu_c, jnp.log(mix))

    pay = Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=nA + nB + n_heard, in_dim_N=0, out_dim_N=0,
                      out_f=lambda q, w: jnp.zeros(0), in_f=lambda q, w, n_in: jnp.zeros(0),
                      omega=exact(U), label="pay")
    return compose_seq(gammas, compose_seq(beliefs, pay))


def s6_run(problem, sigma, rounds=12):
    """The two beliefs after each round. S6 reads the problem's truth only; its start is never read."""
    _, truth = PROBLEMS[problem]
    arr = s6_arrangement(truth, sigma)

    def read(s, out):
        l_s, l_c = s[1][0]
        return dict(snake=np.asarray(belief_of(l_s)), cat=np.asarray(belief_of(l_c)))

    return _closed_run(arr, rounds, read)


def s6_by_hand(problem, sigma, rounds=12, A=A_VALUES, B=B_VALUES, b2=B2):
    _, truth = PROBLEMS[problem]
    ns, nc = np.full(len(A), 1 / len(A)), np.full(len(B), 1 / len(B))
    bell = lambda x, y: math.exp(-0.5 * ((x - y) / sigma) ** 2)
    out = []
    for t in range(rounds):
        d0 = D0S[t]
        y, y2 = scene(*truth, d0), scene(truth[0], b2, d0)
        g2 = np.array([bell(y2, scene(a, b2, d0)) for a in A])
        mix = np.array([sum(ns[i] * bell(y, scene(a, b, d0)) for i, a in enumerate(A)) for b in B])
        ns, nc = ns * g2 / (ns @ g2), nc * mix / (nc @ mix)    # the cat's mix uses the snake's belief before
        out.append(dict(snake=ns.copy(), cat=nc.copy()))
    return out


# ---------------------------------------------------------------------------
# S6g: David's "generic snake" (2026-10-15): the cat imagined with one snake, the average under the belief.
# ---------------------------------------------------------------------------


def s6g_arrangement(truth, sigma, d0s=D0S, A=A_VALUES, B=B_VALUES, b2=B2):
    """S6 with the cat's scenes run with the average snake ``sum_a nu_s(a) a``, read one-way: the scene box holds
    the snake's belief fixed in its backward map, so no force reaches it from the cat's payment."""
    lik = _bell(sigma)
    nA, nB = len(A), len(B)
    A_arr = jnp.asarray(A)

    def world(v):
        return jnp.stack([v[0], scene(*truth, v[0]), scene(truth[0], b2, v[0])])

    def comparison_box(a):
        return map_box(lambda v: jnp.reshape(lik(v[2], scene(a, b2, v[0])), (1,)), 3, 1, f"comparison({a:g})")

    seen = compose_seq(compose_seq(_rounds_box(d0s, None), map_box(world, 2, 3, "world")),
                       fan_box(3, [0, 1, 2] * nA + [0, 1], "copy"))
    heard = compose_seq(seen, tensor_arrangements([comparison_box(a) for a in A] + [identity_box(2)]))
    P = belief_coalgebra(nA, jnp.zeros(nA)).tensor(belief_coalgebra(nB, jnp.zeros(nB)))
    beliefs = _beliefs(P, nA + 2, nA + nB)          # presents nu_s, nu_c; passes gamma2 (nA), d0, D*

    def cat_box(b):
        def g(v):
            abar = jnp.dot(jax.lax.stop_gradient(v[:nA]), A_arr)     # the generic snake, read one-way
            return jnp.reshape(lik(v[nA + 1], scene(abar, b, v[nA])), (1,))
        return map_box(g, nA + 2, 1, f"scene(avg, {b:g})")

    keep = list(range(nA + nB + nA))                # nu_s, nu_c, gamma2
    to_cat = list(range(nA)) + [2 * nA + nB, 2 * nA + nB + 1]
    route = fan_box(2 * nA + nB + 2, keep + to_cat * nB, "route")
    cats = tensor_arrangements([identity_box(len(keep))] + [cat_box(b) for b in B])

    def U(q, w, n_in):
        nu_s, nu_c, g2, gc = w[:nA], w[nA:nA + nB], w[nA + nB:2 * nA + nB], w[2 * nA + nB:]
        return jnp.dot(nu_s, jnp.log(g2)) + jnp.dot(nu_c, jnp.log(gc))

    pay = Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=2 * nA + 2 * nB, in_dim_N=0, out_dim_N=0,
                      out_f=lambda q, w: jnp.zeros(0), in_f=lambda q, w, n_in: jnp.zeros(0),
                      omega=exact(U), label="pay")
    return compose_seq(heard, compose_seq(beliefs, compose_seq(route, compose_seq(cats, pay))))


def s6g_run(problem, sigma, rounds=12):
    _, truth = PROBLEMS[problem]
    arr = s6g_arrangement(truth, sigma)

    def read(s, out):
        l_s, l_c = s[1][0]
        return dict(snake=np.asarray(belief_of(l_s)), cat=np.asarray(belief_of(l_c)))

    return _closed_run(arr, rounds, read)


def s6g_by_hand(problem, sigma, rounds=12, A=A_VALUES, B=B_VALUES, b2=B2):
    _, truth = PROBLEMS[problem]
    ns, nc = np.full(len(A), 1 / len(A)), np.full(len(B), 1 / len(B))
    bell = lambda x, y: math.exp(-0.5 * ((x - y) / sigma) ** 2)
    out = []
    for t in range(rounds):
        d0 = D0S[t]
        y, y2 = scene(*truth, d0), scene(truth[0], b2, d0)
        abar = float(ns @ np.asarray(A))
        g2 = np.array([bell(y2, scene(a, b2, d0)) for a in A])
        gc = np.array([bell(y, scene(abar, b, d0)) for b in B])
        ns, nc = ns * g2 / (ns @ g2), nc * gc / (nc @ gc)
        out.append(dict(snake=ns.copy(), cat=nc.copy(), abar=abar))
    return out


def settled(tops, truth):
    """The first round (from 1) from which the top cast is the truth at every later round, else ``None``."""
    for t in range(len(tops)):
        if all(x == truth for x in tops[t:]):
            return t + 1
    return None
