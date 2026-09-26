"""Scattering: random kicks on a collective's placements.

The book plan's P6 stage (c) (AGENTDOCPLAN.md §8.2 in the book repository; David,
2026-09-25, "if it can, then maybe we should include that test too?"), whose claim, Claude's
and *proposed*, was: "with the supply cut, a bounded finite random kick carries parts out of
kernel reach, where kernel and gradient both vanish so nothing returns them." The collective
is ``tests/test_dispersal.py``'s (frozen parts sending ``(1, 1)`` on a spatial parent with
the total cost), and the parent's integrator is ``rand.kicked_configuration``: the descent
step, then an independent kick of ``+a`` or ``-a`` on every placement, each with probability
1/2. It is paired with a pseudo-random sampler by ``rand.seed``; many runs are one ``vmap``
over seeds. A part is *in reach* while ``kappa > 0.05``, i.e. ``|p| < r = 2.19``.

**The predictions, fixed in advance and written here before any construction code for this
file existed. Each must be able to fail.**

(8) *With the need present, noise does what a push does: redundancy decides.* Target 1.5,
kick ``a = 0.05``, 50 runs of 20,000 steps. With ``n = 3`` (any two can cover the target)
the kicks drive a random walk along the family of minima, which is unbounded in the
direction of losing a part, and at least 80% of runs end with a part out of reach. With
``n = 2`` (the other can send at most 1) the family is bounded, and no part in any run ever
passes ``|p| = 1.5``. Fails if the ``n = 3`` collective keeps its parts or the ``n = 2``
one loses one.

(9) *With nothing holding them, scattered parts do come back -- ever more rarely.* This
corrects the plan's "nothing returns them". With the parent's cost removed (the supply cut),
each placement is a simple random walk: no gradient returns a part, but a walk on the line
is recurrent and returns by itself. The probability that a part is in reach at step ``t`` is
exactly the binomial ``P(|p_0 + a (2B - t)| < r)``, ``B ~ Bin(t, 1/2)``; the fraction of
runs with the part in reach matches it within three standard errors at ``t = 100, 1000,
10000``, and between ``t = 1000`` and ``10000`` it falls by ``sqrt(10) ~ 3.16`` within 15%.
Fails if the frequencies miss the binomial or the decay is not ``t^{-1/2}``. On a plane the
walk is still recurrent, only more slowly; in three or more dimensions it is transient and a
scattered part would, with positive probability, never return. The collective module places
parts on a line, so that case is not run.

**What happened** (2026-09-26, recorded after the run, the predictions above untouched).
(8) held, narrowly: 82% of ``n = 3`` runs lost a part (threshold 80%); the farthest any
``n = 2`` part got was 1.319. (9) held in its first half -- the frequencies match the
binomial at every ``t`` -- and was **refuted in its second as stated**: from ``t = 1000`` to
``10000`` the fraction in reach fell by 2.145, not 3.16. The error was the window, not the
law: at ``t = 1000`` the walk's spread ``a sqrt(t) = 1.6`` is still inside the reach
``r = 2.19``, so the ``t^{-1/2}`` tail has not set in; the exact binomial itself falls by 2.24
there, by 3.02 from ``10^4`` to ``10^5``, and by 3.15 from ``10^5`` to ``10^6``. The test below
asserts the corrected claim, added after the result: the runs match the binomial through
``t = 10^5``, and between ``10^4`` and ``10^5``, where ``a sqrt(t) >> r``, they fall by
``sqrt(10)`` within 15%.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import binom

from arrangements.collective import (
    collective, frozen_part, position, run, split, total_cost, with_sampler,
)
from arrangements.rand import key_sampler, kicked_configuration

TAU, MU, SIG = 0.8, 0.5, 1.0     # as in tests/test_dispersal.py
A = 0.05                         # the kick
R = float(np.sqrt(-2 * TAU * np.log(0.05)))   # in reach: kappa > 0.05


def distance(k):
    return float(np.sqrt(-2 * TAU * np.log(k)))


def kicks(n, a):
    """Every sign pattern of ``+-a`` on the ``n`` placements, zero on the value maps."""
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
    K = np.concatenate([a * signs, np.zeros_like(signs)], axis=1)
    return jnp.full(len(K), 1.0 / len(K)), jnp.asarray(K)


def build(positions, cost, a=A):
    n = len(positions)
    w, K = kicks(n, a)
    return with_sampler(collective(
        [frozen_part(SIG, 1.0, f"C{j}") for j in range(n)],
        tau=TAU, cost=cost, positions=positions,
        integrator=lambda Q, init: kicked_configuration(Q, w, K, init=init),
        mobility=MU, value_mobility=0.0), key_sampler(0))


def many(C, runs, seed=0):
    """``runs`` initial states differing only in the sampler's key."""
    s0, _ = C.arr.P.init
    keys = jax.random.split(jax.random.PRNGKey(seed), runs)
    return jax.vmap(lambda k: (s0, k))(keys)


def advance(C, states, steps):
    return jax.jit(jax.vmap(lambda s: run(C, steps, s)))(states)


def placements(C, states):
    return np.asarray(jax.vmap(lambda s: split(C, position(C, s))[1])(states))


def test_8_with_the_need_present_noise_does_what_a_push_does():
    nu, runs, T = 1.5, 50, 20000
    C3 = build([distance(nu / 3)] * 3, total_cost(3, TAU, nu))
    p3 = placements(C3, advance(C3, many(C3, runs), T))
    lost3 = (np.abs(p3) > R).any(axis=1).mean()

    C2 = build([distance(nu / 2)] * 2, total_cost(2, TAU, nu))

    def body(s, _):
        s = jax.vmap(lambda x: run(C2, 1, x))(s)
        return s, jnp.max(jnp.abs(jax.vmap(lambda x: split(C2, position(C2, x))[1])(s)))

    _, far2 = jax.jit(lambda s: jax.lax.scan(body, s, None, length=T))(many(C2, runs))
    far2 = float(jnp.max(far2))
    print(f"\n  n=3: {lost3:.0%} of runs end with a part out of reach; "
          f"n=2: farthest any part got in any run {far2:.3f}")
    assert lost3 >= 0.8, lost3
    assert far2 < 1.5, far2


def test_9_with_nothing_holding_them_scattered_parts_come_back_ever_more_rarely():
    n, runs, p0 = 4, 500, distance(0.5)
    C = build([p0] * n, lambda q, m, e: 0.0 * jnp.sum(q))
    s, t, rows = many(C, runs), 0, []
    for t_next in (100, 1000, 10000, 100000):
        s = advance(C, s, t_next - t)
        t = t_next
        p = placements(C, s).ravel()
        freq = float((np.abs(p) < R).mean())
        B = np.arange(t + 1)
        exact = float(binom.pmf(B, t, 0.5)[np.abs(p0 + A * (2 * B - t)) < R].sum())
        se = np.sqrt(exact * (1 - exact) / p.size)
        rows.append((t, freq, exact, se))
    print("\n  t      in reach (runs)   binomial   3 s.e.")
    for t, f, e, se in rows:
        print(f"  {t:<6} {f:.4f}           {e:.4f}     {3 * se:.4f}")
    for t, f, e, se in rows:
        assert abs(f - e) < 3 * se, (t, f, e, se)
    early, late = rows[1][1] / rows[2][1], rows[2][1] / rows[3][1]
    print(f"  fall 10^3 -> 10^4: {early:.3f} (the refuted window); "
          f"10^4 -> 10^5: {late:.3f} (sqrt 10 = {np.sqrt(10):.3f})")
    assert abs(early / np.sqrt(10) - 1) > 0.15, early      # the window as predicted: refuted
    assert abs(late / np.sqrt(10) - 1) < 0.15, late
