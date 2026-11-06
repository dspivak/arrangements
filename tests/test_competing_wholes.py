"""Competition with no odd cycle is one monotone theory, with an odd cycle it is frustrated: the check of book plan P73's competition theorems
(``notes/runs/p73-competition-3bf2ce2f/RESULTS.md``, whose predictions K0-K9 were written before this file). David,
2026-11-06: "no, let's do it now, but only as a theorem, a structure bearing thing. not an example." The theorem is in
that file; this is its check, not its evidence. Everything not quoted from him is the agent's.

The whole: ``test_bees_shadow.py``'s gated pairs and the sun, in one nested closed arrangement; the parent reads the
pairs' activities and the sun and gives each camp its share of the sun. The theorem's objects are calculations on its
step (K0 licenses that); ``settle`` runs the arrangement.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact, zero_form
from arrangements.coalgebra import configuration
from arrangements.rvect import euclidean
from arrangements.wiring import compose_seq, tensor_arrangements, trivial_parameter

from tests.test_bees_shadow import EMPTY, MOB, W, gfp_below, group, lfp_above, run_closed, run_phi, settle, steps_for, sun

DELTA = 0.01
TOP = 8.0


def hi(u):
    """The pair's larger rest at supply ``u``, by hand; 0 below its threshold."""
    d = u ** 2 - 4.5
    return 0.0 if d < 0 or u <= 0 else (u + np.sqrt(d)) / 1.5


def shares(ys, camps=None):
    """A row of the network, as the listener's proportions. ``camps=None``: proportional to activity, every pair
    competing. ``camps``: for each pair its camp (0 or 1) and frozen share inside it: the row factors through 2."""
    if camps is None:
        return (ys + DELTA) / (jnp.sum(ys) + len(ys) * DELTA)
    labels = jnp.array([c for c, _ in camps])
    inside = jnp.array([f for _, f in camps])
    a0 = (jnp.sum(ys * (labels == 0)) + DELTA) / (jnp.sum(ys) + 2 * DELTA)
    return jnp.where(labels == 0, a0, 1.0 - a0) * inside


def parent(n, camps=None):
    def in_f(q, m_out, n_in):
        return m_out[n] * shares(m_out[:n], camps)
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=n, out_dim_M=n + 1, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: EMPTY,
        in_f=in_f,
        omega=zero_form,
        label=f"one_sun_{n}" if camps is None else f"one_sun_{n}_through_2",
    )


def whole(q, s, camps=None, leaks=None):
    n = len(q) // 2
    leaks = leaks or [0.0] * n
    boxes = [group(q[2 * i], q[2 * i + 1], leak=leaks[i], label=f"camp{i}") for i in range(n)] + [sun(s)]
    return compose_seq(tensor_arrangements(boxes), parent(n, camps))


def parts_step(arr, s):
    omega = arr.omega

    def F_jnp(x):
        q = jnp.concatenate([x, jnp.array([s])])
        return (q + MOB * omega(q, EMPTY, EMPTY)[0])[:-1]
    return F_jnp


class Polar:
    """A step read in the order of a polarity: the inverter ``x -> TOP - x`` on each minus coordinate turns it into a
    monotone map in the plain order (the functor (L, eps) -> L^eps), so plain Kleene gives nu^eps and mu^eps."""

    def __init__(self, F, eps):
        self.F, self.eps = F, np.asarray(eps, float)
        self.G = jax.jit(lambda z: self.flip(F(self.flip(z))))

    def flip(self, x):
        return jnp.where(jnp.asarray(self.eps) > 0, x, TOP - x)

    def nu(self):
        cap = np.full(len(self.eps), TOP)
        return np.asarray(self.flip(gfp_below(lambda z: np.asarray(self.G(jnp.asarray(z))), cap, tol=1e-9)))

    def mu(self):
        return np.asarray(self.flip(lfp_above(lambda z: np.asarray(self.G(jnp.asarray(z))), np.zeros(len(self.eps)), tol=1e-9)))


def leq(a, b, eps, tol=1e-6):
    return bool(np.all(np.asarray(eps) * (np.asarray(b) - np.asarray(a)) >= -tol))


def worst_entry(F, eps, n_states=40, seed=20261106):
    """The smallest ``eps_i eps_j J_ij``, over random states of the box: nonnegative when ``eps`` is a polarity."""
    eps = np.asarray(eps, float)
    rng = np.random.default_rng(seed)
    jac = jax.jit(jax.jacobian(F))
    worst = np.inf
    for _ in range(n_states):
        x = rng.uniform(0.01, 1.0, len(eps)) * np.array([TOP, TOP / 2] * (len(eps) // 2))
        worst = min(worst, float((np.outer(eps, eps) * np.asarray(jac(jnp.asarray(x)))).min()))
    return worst


EPS = (1, 1, -1, -1)
A3 = (3.400, 1.700, 0.0, 0.0)


def test_K0_the_runs_are_phis():
    b = whole((3.0, 1.5, 2.0, 1.0), 5.0)
    print("label:", b.label)
    by_phi = run_phi(b, 50, EMPTY)
    by_closed = run_closed(b, 50, lambda t: EMPTY)
    assert float(np.max(np.abs(by_phi - by_closed))) < 1e-13
    F = parts_step(b, 5.0)
    x = jnp.array([3.0, 1.5, 2.0, 1.0])
    for _ in range(50):
        x = F(x)
    np.testing.assert_allclose(np.asarray(x), by_closed[-1][:4], atol=1e-5)


def test_K1_the_mixed_order_is_a_polarity():
    F = parts_step(whole((3.0, 1.5, 2.0, 1.0), 5.0), 5.0)
    w_eps, w_plain = worst_entry(F, EPS), worst_entry(F, (1, 1, 1, 1))
    print("K1 worst entry, polarity:", w_eps, " all plus:", w_plain)
    assert w_eps >= -1e-9
    assert w_plain < -1e-6
    P = Polar(F, EPS)
    jac = jax.jit(jax.jacobian(P.G))
    rng = np.random.default_rng(1)
    for _ in range(10):
        z = rng.uniform(0.01, 0.5, 4) * TOP
        assert float(np.asarray(jac(jnp.asarray(z))).min()) >= -1e-9


def test_K2_two_winners_below_twice_the_threshold():
    s = 3.0
    F = parts_step(whole(A3, s), s)
    P = Polar(F, EPS)
    nu, mu = P.nu(), P.mu()
    print("K2 nu:", nu, " mu:", mu)
    np.testing.assert_allclose(nu, A3, atol=2e-3)
    np.testing.assert_allclose(mu, (0.0, 0.0, 3.400, 1.700), atol=2e-3)
    dead = np.zeros(4)
    np.testing.assert_allclose(np.asarray(F(jnp.asarray(dead))), dead, atol=1e-12)
    assert leq(mu, dead, EPS) and leq(dead, nu, EPS)
    end = settle(whole((3.0, 1.5, 3.0, 1.5), s), steps_for(200.0))[:4]
    print("K2 equal sharing ends:", end)
    assert np.max(np.abs(end)) < 1e-3


def test_K3_no_join_pointwise_a_chain_in_the_polar_order():
    s = 5.0
    F = parts_step(whole(A3, s), s)
    P = Polar(F, EPS)
    nu, mu = P.nu(), P.mu()
    print("K3 nu:", nu, " mu:", mu)
    np.testing.assert_allclose(nu, (6.341, 3.170, 0.0, 0.0), atol=2e-3)
    q = hi(2.5)
    both = np.array([q, W * q, q, W * q])
    np.testing.assert_allclose(np.asarray(F(jnp.asarray(both))), both, atol=1e-6)
    # pointwise: neither above the other, and their pointwise join is no fixed point and flows down
    assert not leq(both, nu, (1, 1, 1, 1)) and not leq(nu, both, (1, 1, 1, 1))
    join = np.maximum(both, nu)
    assert np.max(np.abs(np.asarray(F(jnp.asarray(join))) - join)) > 1e-4
    # polar: a chain
    assert leq(mu, both, EPS) and leq(both, nu, EPS)
    # both alive is unstable: a small push goes to a winner
    end = settle(whole((q + 0.05, W * q, q - 0.05, W * q), s), steps_for(400.0))[:4]
    print("K3 pushed end:", end)
    np.testing.assert_allclose(end, nu, atol=2e-2)


def test_K4_kept_sets_lie_in_the_interval():
    s = 5.0
    F = parts_step(whole(A3, s), s)
    P = Polar(F, EPS)
    nu, mu = P.nu(), P.mu()
    rng = np.random.default_rng(4)
    run = jax.jit(lambda x: jax.lax.fori_loop(0, steps_for(300.0), lambda _i, y: F(y), x))
    ends = []
    for _ in range(12):
        x = rng.uniform(0.0, 1.0, 4) * np.array([TOP, TOP / 2, TOP, TOP / 2])
        end = np.asarray(run(jnp.asarray(x)))
        ends.append(end)
        assert leq(mu, end, EPS, tol=1e-3) and leq(end, nu, EPS, tol=1e-3), end
    print("K4 ends:", np.round(np.array(ends), 3))


def test_K5_level_maps_compose_in_the_polar_order():
    for s in (3.0, 5.0):
        F = parts_step(whole(A3, s), s)
        nu = Polar(F, EPS).nu()
        ya, yb = TOP, 0.0
        for _ in range(100000):
            a = (ya + DELTA) / (ya + yb + 2 * DELTA)
            na, nb = hi(s * a), hi(s * (1 - a))
            if abs(na - ya) + abs(nb - yb) < 1e-10:
                break
            ya, yb = na, nb
        staged = np.array([ya, W * ya, yb, W * yb])
        print("K5", s, "flat:", nu, " staged:", staged)
        np.testing.assert_allclose(staged, nu, atol=1e-4)


def test_K6_a_free_lunch_to_one_camp_moves_up_the_polar_order():
    s = 5.0
    F0 = parts_step(whole(A3, s), s)
    F1 = parts_step(whole(A3, s, leaks=[0.5, 0.0]), s)
    P0, P1 = Polar(F0, EPS), Polar(F1, EPS)
    print("K6 mu:", P0.mu(), "->", P1.mu(), " nu:", P0.nu(), "->", P1.nu())
    assert leq(P0.mu(), P1.mu(), EPS) and leq(P0.nu(), P1.nu(), EPS)
    assert not np.allclose(P0.mu(), P1.mu(), atol=1e-3)


def test_K7_the_sun_has_no_polarity():
    def both(u):
        q = hi(u / 2)
        return np.array([q, W * q, q, W * q])
    a, b = both(5.0), both(6.0)
    print("K7 both alive at 5, 6:", a, b)
    assert not leq(a, b, EPS) and not leq(b, a, EPS)


def test_K8_three_camps_are_frustrated():
    n = 3
    s = 3.0
    q0 = (3.0, 1.5, 0.0, 0.0, 0.0, 0.0)
    F = parts_step(whole(q0, s), s)
    # every polarity up to a flip fails, each camp's two parts having one sign
    worst = {}
    for signs in itertools.product((1, -1), repeat=n):
        if signs[0] < 0:
            continue
        eps = sum(([e, e] for e in signs), [])
        worst[signs] = worst_entry(F, eps)
    print("K8 worst entry by polarity:", worst)
    assert len(worst) == 4 and all(v < -1e-6 for v in worst.values())
    # three winners, fixed points
    winners = []
    for i in range(n):
        x = np.zeros(6)
        x[2 * i], x[2 * i + 1] = 3.4, 1.7
        run = jax.jit(lambda y: jax.lax.fori_loop(0, steps_for(200.0), lambda _i, v: F(v), y))
        winners.append(np.asarray(run(jnp.asarray(x))))
    print("K8 winners:", np.round(np.array(winners), 3))
    for i, w in enumerate(winners):
        assert abs(w[2 * i] - 3.4) < 2e-2 and np.max(np.delete(w, [2 * i, 2 * i + 1])) < 1e-6
    # any polarity gives two camps one sign, and their winners are incomparable
    for signs in worst:
        eps = sum(([e, e] for e in signs), [])
        i, j = next((i, j) for i, j in itertools.combinations(range(n), 2) if signs[i] == signs[j])
        assert not leq(winners[i], winners[j], eps) and not leq(winners[j], winners[i], eps)
    # what survives: the majorant's ceiling, every camp at the full supply
    rng = np.random.default_rng(8)
    run = jax.jit(lambda y: jax.lax.fori_loop(0, steps_for(200.0), lambda _i, v: F(v), y))
    for _ in range(8):
        x = rng.uniform(0.0, 1.0, 6) * np.array([TOP, TOP / 2] * 3)
        end = np.asarray(run(jnp.asarray(x)))
        assert np.max(end[::2]) <= hi(s) + 1e-3, end


def tilted_step(s, t, leaks=None):
    """The two-camp step with an outside listener's fraction ``t`` of the row given to camp A, ``a' = (1-t) a + t``,
    the share as an input port; a parent of its own, one nested closed arrangement again."""
    def in_f(q, m_out, n_in):
        a = (1.0 - m_out[3]) * shares(m_out[:2])[0] + m_out[3]
        return m_out[2] * jnp.array([a, 1.0 - a])
    par = Arrangement(
        P=trivial_parameter(), in_dim_M=2, out_dim_M=4, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: EMPTY, in_f=in_f, omega=zero_form, label="one_sun_tilted",
    )
    boxes = [group(3.0, 1.5, label="camp0"), group(2.0, 1.0, label="camp1"), sun(s), sun(t)]
    arr = compose_seq(tensor_arrangements(boxes), par)
    omega = arr.omega

    def F(x, tt=t):
        q = jnp.concatenate([x, jnp.array([s, tt])])
        return (q + MOB * omega(q, EMPTY, EMPTY)[0])[:4]
    return F


def test_K10_the_share_is_a_port():
    s, t = 5.0, 0.1
    F = tilted_step(s, t)
    # monotone in the state and in the share, in the polar order, with the share of polarity +
    G = lambda xt: jnp.concatenate([F(xt[:4], xt[4]), xt[4:]])
    eps = np.array([1, 1, -1, -1, 1.0])
    jac = jax.jit(jax.jacobian(G))
    rng = np.random.default_rng(10)
    worst = np.inf
    for _ in range(40):
        x = np.append(rng.uniform(0.01, 1.0, 4) * np.array([TOP, TOP / 2, TOP, TOP / 2]), rng.uniform(0.0, 1.0))
        worst = min(worst, float((np.outer(eps, eps) * np.asarray(jac(jnp.asarray(x)))).min()))
    print("K10 worst entry:", worst)
    assert worst >= -1e-9
    P, P0 = Polar(lambda x: F(x), EPS), Polar(tilted_step(s, 0.0), EPS)
    nu, mu, nu0, mu0 = P.nu(), P.mu(), P0.nu(), P0.mu()
    print("K10 nu:", nu0, "->", nu, " mu:", mu0, "->", mu)
    np.testing.assert_allclose(nu, (6.343, 3.172, 0.0, 0.0), atol=3e-3)
    np.testing.assert_allclose(mu, (0.0, 0.0, 5.635, 2.817), atol=3e-3)
    assert leq(nu0, nu, EPS) and leq(mu0, mu, EPS)


def graph_step(n, edges, s=5.0):
    """Each edge a competition with a listener and a sun of its own; a part's supply sums its shares."""
    def in_f(q, m_out, n_in):
        ys = m_out[:n]
        supply = jnp.zeros(n)
        for e, (i, j) in enumerate(edges):
            a = (ys[i] + DELTA) / (ys[i] + ys[j] + 2 * DELTA)
            supply = supply.at[i].add(m_out[n + e] * a).at[j].add(m_out[n + e] * (1.0 - a))
        return supply
    par = Arrangement(
        P=trivial_parameter(), in_dim_M=n, out_dim_M=n + len(edges), in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: EMPTY, in_f=in_f, omega=zero_form, label=f"competition_graph_{n}",
    )
    boxes = [group(3.0 - 0.3 * i, 1.5, label=f"camp{i}") for i in range(n)] + [sun(s) for _ in edges]
    arr = compose_seq(tensor_arrangements(boxes), par)
    omega = arr.omega
    suns = jnp.full(len(edges), s)

    def F(x):
        q = jnp.concatenate([x, suns])
        return (q + MOB * omega(q, EMPTY, EMPTY)[0])[:2 * n]
    return F


def polarities_that_pass(n, edges):
    F = graph_step(n, edges)
    good = []
    for signs in itertools.product((1, -1), repeat=n):
        if signs[0] < 0:
            continue
        if worst_entry(F, sum(([e, e] for e in signs), []), n_states=15) >= -1e-9:
            good.append(signs)
    return good


def test_K11_the_obstruction_is_an_odd_cycle_not_a_count():
    line = polarities_that_pass(3, [(0, 1), (1, 2)])
    ring4 = polarities_that_pass(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    ring3 = polarities_that_pass(3, [(0, 1), (1, 2), (2, 0)])
    ring5 = polarities_that_pass(5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
    print("K11 line:", line, " ring4:", ring4, " ring3:", ring3, " ring5:", ring5)
    assert line == [(1, -1, 1)]
    assert ring4 == [(1, -1, 1, -1)]
    assert ring3 == [] and ring5 == []


def test_K12_the_budget_bounds_a_frustrated_row_at_rest():
    theta = np.sqrt(4.5)
    assert 6.3 < 3 * theta < 6.5
    ends = {}
    for s in (6.3, 6.5):
        q0 = (5.0, 2.5, 5.0, 2.5, 5.0, 2.5)
        F = parts_step(whole(q0, s), s)
        run = jax.jit(lambda y: jax.lax.fori_loop(0, steps_for(800.0), lambda _i, v: F(v), y))
        ends[s] = np.asarray(run(jnp.asarray(q0, float)))
    print("K12 ends:", ends)
    assert np.max(ends[6.3]) < 1e-2
    np.testing.assert_allclose(ends[6.5][0], hi(6.5 / 3), atol=1e-2)
    np.testing.assert_allclose(ends[6.5], [ends[6.5][0], W * ends[6.5][0]] * 3, atol=1e-6)


def entropy_listener(T, v, s0):
    """Speakers with fixed values ``v``; a row, softmax of a state of scores; a listener whose output is the affine
    map ``<P, v> + T H(P)``, the functor H_T on the transfer; a unit cost on that output. No effort field on the scores:
    any entropy term in their inclination comes back through the transfer."""
    n = len(v)

    def out_f(q, m):
        P = jax.nn.softmax(q)
        return jnp.array([P @ m - T * jnp.sum(P * jnp.log(P))])

    listener = Arrangement(
        P=configuration(euclidean(n, MOB), init=jnp.asarray(s0, float)),
        in_dim_M=0, out_dim_M=n, in_dim_N=0, out_dim_N=1,
        out_f=out_f,
        in_f=lambda q, m_out, n_in: EMPTY,
        omega=zero_form,
        label=f"listener_T{T}",
    )
    unit = Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=1, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: EMPTY, in_f=lambda q, m_out, n_in: EMPTY,
        omega=exact(lambda q, m_out, n_in: m_out[0]), label="unit_cost",
    )
    return compose_seq(compose_seq(tensor_arrangements([sun(x) for x in v]), listener), unit)


def test_K13_entropy_comes_back_through_the_transfer():
    v = (0.0, 1.0, 2.0)
    s0 = (0.1, -0.2, 0.05)
    for T, want in ((1.0, (0.0900, 0.2447, 0.6652)), (0.5, (0.0159, 0.1173, 0.8668))):
        b = entropy_listener(T, v, s0)
        print("label:", b.label)
        by_phi = run_phi(b, 50, EMPTY)
        by_closed = run_closed(b, 50, lambda t: EMPTY)
        assert float(np.max(np.abs(by_phi - by_closed))) < 1e-13
        end = settle(b, steps_for(1500.0))
        # the configuration lists the three speakers' fixed values first, then the scores
        np.testing.assert_allclose(end[:3], v, atol=1e-12)
        P = np.asarray(jax.nn.softmax(jnp.asarray(end[3:])))
        print("K13 T", T, "scores", end[3:], "row", P)
        np.testing.assert_allclose(P, want, atol=2e-3)
    end = settle(entropy_listener(0.0, v, s0), steps_for(1500.0))
    P = np.asarray(jax.nn.softmax(jnp.asarray(end[3:])))
    print("K13 T 0 row", P)
    assert P[2] > 0.99


def test_K14_a_nest_at_rest_is_the_flat_softmax():
    """Prediction, written before it ran: two groups of speakers, each with its own listener whose output is the affine
    map of H_T, and a top listener of the same kind hearing the two groups, a unit cost on its output. At rest each group
    sends T log sum e^{v/T}, and the product of the two rows along each route is softmax(v/T), with no group score
    written anywhere. The load reaching a group is scaled by its share, which does not change the rest."""
    T = 0.7
    v1, v2 = (0.0, 1.0), (0.5, 2.0, -1.0)

    def listener(n, s0, label):
        def out_f(q, m):
            P = jax.nn.softmax(q)
            return jnp.array([P @ m - T * jnp.sum(P * jnp.log(P))])
        return Arrangement(
            P=configuration(euclidean(n, MOB), init=jnp.asarray(s0, float)),
            in_dim_M=0, out_dim_M=n, in_dim_N=0, out_dim_N=1,
            out_f=out_f, in_f=lambda q, m_out, n_in: EMPTY, omega=zero_form, label=label,
        )

    unit = Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=1, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: EMPTY, in_f=lambda q, m_out, n_in: EMPTY,
        omega=exact(lambda q, m_out, n_in: m_out[0]), label="unit_cost",
    )
    group1 = compose_seq(tensor_arrangements([sun(x) for x in v1]), listener(2, (0.2, -0.1), "group1"))
    group2 = compose_seq(tensor_arrangements([sun(x) for x in v2]), listener(3, (0.1, 0.0, -0.2), "group2"))
    b = compose_seq(compose_seq(tensor_arrangements([group1, group2]), listener(2, (0.0, 0.3), "top")), unit)
    print("label:", b.label)
    by_phi = run_phi(b, 50, EMPTY)
    by_closed = run_closed(b, 50, lambda t: EMPTY)
    assert float(np.max(np.abs(by_phi - by_closed))) < 1e-13
    end = settle(b, steps_for(4000.0))
    print("K14 end:", end)
    # configuration, in order: group1's speakers and scores, group2's speakers and scores, the top's scores
    s1, s2, top = end[2:4], end[7:10], end[10:12]
    np.testing.assert_allclose(end[:2], v1, atol=1e-12)
    np.testing.assert_allclose(end[4:7], v2, atol=1e-12)
    P1, P2, Pt = (np.asarray(jax.nn.softmax(jnp.asarray(x))) for x in (s1, s2, top))
    routes = np.concatenate([Pt[0] * P1, Pt[1] * P2])
    flat = np.asarray(jax.nn.softmax(jnp.asarray(v1 + v2) / T))
    print("K14 routes:", routes, " flat:", flat)
    np.testing.assert_allclose(routes, flat, atol=2e-3)


def test_K9_a_row_through_two_is_balanced():
    camps = [(0, 0.5), (1, 1.0), (0, 0.5)]
    q0 = (3.0, 1.5, 2.0, 1.0, 3.0, 1.5)
    b = whole(q0, 5.0, camps=camps)
    F = parts_step(b, 5.0)
    w = worst_entry(F, (1, 1, -1, -1, 1, 1))
    print("K9 worst entry, camps {A,C} against {B}:", w)
    assert w >= -1e-9
    # and the frozen shares are what makes it so: the same three with a proportional row fail this polarity
    F3 = parts_step(whole(q0, 5.0), 5.0)
    assert worst_entry(F3, (1, 1, -1, -1, 1, 1)) < -1e-6
