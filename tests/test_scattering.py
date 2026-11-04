"""Noise can carry a part away (book Chapter 1, ``sec.check_claims``: "and even a random noise can carry it away").

For a reader of the book. The collective is ``arrangements/collective.py``: ``n`` parts, each sending the same signal ``1``, placed on a line
under a parent. A part counts for ``kappa(p) = exp(-p^2 / 2 tau)`` by its placement ``p``, so one placed far away is out
of reach, and the parent's cost is the gap between the total it sends onward and what the collective needs, ``nu``:
``(nu - sum_j kappa_j)^2 / 2``. Only the placements move. Here, at each step, every placement also gets a kick of ``+a`` or ``-a`` at random
(``rand.kicked_configuration``). With three parts and need 1.5 the kicks drive a random walk along the placements that
keep the total, and nothing stops that walk from losing a part; with two parts those placements are bounded. A part is
out of reach once ``kappa < 0.05``, ``|p| > R``. Fifty runs of each, run from the library's folder with it on the path (``PYTHONPATH=. .venv/bin/python example.py``):

    import jax; jax.config.update("jax_enable_x64", True)
    import numpy as np
    from tests.test_scattering import R, TAU, advance, build, distance, many, placements
    from arrangements.collective import total_cost
    for n in (3, 2):
        C = build([distance(1.5 / n)] * n, total_cost(n, TAU, 1.5))
        p = placements(C, advance(C, many(C, 50), 20000))
        print(n, float((np.abs(p) > R).any(axis=1).mean()), round(float(np.abs(p).max()), 3))   # n, share of runs out of reach, farthest
    # 3 0.82 16.62
    # 2 0.0 1.219

In 82% of the runs with three parts a part ends out of reach; with two, never.

What the tests check. Test 8: at least 80% against none. That is a narrow pass of the bar fixed in advance, and a count of
one model's runs, not a rate of anything in the world. Test 9: with the cost removed, nothing holds the parts, and each is
a plain random walk: it does come back, more and more rarely, as the exact binomial says, so "carried away" here means
out of reach at the time looked at, not gone forever.

For agents: the file's earlier head, with the predictions fixed before the code, what happened and the
weirdness checks, is in ``notes/P31-B-CH1-HISTORY-2026-11-03.md``.
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
