"""Four people in the positioned operad: couples that settle when near, and whether Bob walks (book plan P77 (c)).

The predictions, written before this file, and the results are in the book repository,
``notes/runs/p77c-8edc1bdb/RESULTS.md``; each docstring names the prediction it checks. What runs in ``arr``: each
person is an ``Arrangement`` with the interface of a nullary operation of ``positioned.py``; each group is an
``ear_op``; a whole is ``compose_seq(tensor_arrangements(parts), parent)``, closed by the outside sending the zero
message, and run by ``Phi``. The matrices ``step_matrix`` builds are calculations beside the runs, to compare with.

A person cannot be a cost: Alice's push on herself from what she hears, ``1.2 h``, and Bob's, ``-1.2 h``, form a
coupling with a non-symmetric Jacobian, which no gradient has. So a person's effort field is not exact: a force on its
own state, from its state and what it hears, and none on what it hears, unless it also has a cost on what it hears
(Bob in the second piece).
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement
from arrangements.coalgebra import configuration
from arrangements.functors import Phi
from arrangements.positioned import at, centre_of_mass, ear_op, substitute, unit
from arrangements.rvect import diagonal, trivial
from arrangements.wiring import compose_seq, tensor_arrangements

from tests.chain import flat_state

jax.config.update("jax_enable_x64", True)

E = 2                 # people stand in the plane
BETA, EPS = 1.0, 0.1  # hearing fades as exp(-BETA * d_eps)
KIND = {"alice": 1, "bob": -1, "carol": 1, "dan": -1}   # enthusiasts +1, skeptics -1
START = {"alice": 0.02, "bob": -0.01, "carol": -0.015, "dan": 0.025}
A_STAR = float(np.sqrt(0.21 / 1.44))


def person(name, z, mobility=0.0, cost=None):
    """A system storing where it stands and its state ``x``: sends ``at(z)`` and the message ``(x, 1)``, hears
    ``(h, w)``. Its force on ``x`` is ``0.1 x + 1.2 h`` (an enthusiast) or ``-2.1 x - 1.2 h`` (a skeptic): multiplier
    ``+-1.1`` alone at duration 1, pushed ``+-1.2`` by what it hears. ``cost(n_in)``, if given, is a cost on what it
    hears, whose force goes back along the wire (to its own position through the hearing, and to the speakers)."""
    k = KIND[name]
    P = configuration(diagonal([mobility] * E + [1.0]), init=jnp.asarray([*z, START[name]], dtype=float))

    def out_f(q, m):
        return jnp.concatenate([at(q[:E]), jnp.stack([q[E], jnp.asarray(1.0)])])

    def in_f(q, m, n):
        return jnp.zeros(0)

    def omega(q, m_out, n_in):
        x, h = q[E], n_in[0]
        xi_q = jnp.concatenate([jnp.zeros(E), jnp.stack([(1.1 * k - 1.0) * x + 1.2 * k * h])])
        xi_n = jnp.zeros(2) if cost is None else -jax.grad(cost)(n_in)
        return xi_q, jnp.zeros(0), xi_n

    return Arrangement(P=P, in_dim_M=0, out_dim_M=0, in_dim_N=2, out_dim_N=E + 3, out_f=out_f, in_f=in_f,
                       omega=omega, label=name)


def group(n, label):
    return ear_op(1, n, E, configuration(trivial()), centre_of_mass(E), BETA, EPS, label=label)


def tensor(parts):
    return tensor_arrangements(list(parts))


def fill(op, people):
    """``op`` filled by ``people``: the composite in ``arr``."""
    return compose_seq(tensor(people), op.arr())


def run(arr, steps):
    """The closed whole run by ``Phi``, the outside sending the zero message; flattened states."""
    O = Phi(arr)
    state, out = O.state, [flat_state(O.state)]
    for _ in range(steps):
        _, _, state = O.with_state(state).run_one((jnp.zeros(0), lambda x: x), lambda _o: (
            jnp.zeros(arr.out_dim_N), jnp.zeros(arr.in_dim_N)))
        out.append(flat_state(state))
    return np.asarray(out)


def xs(path, n):
    return path[:, [3 * i + E for i in range(n)]]


def d_eps(v):
    return float(np.sqrt(np.dot(v, v) + EPS ** 2))


def loud(u, v):
    return float(np.exp(-BETA * d_eps(np.subtract(u, v))))


def step_matrix(names, pos, groups=None):
    """Beside the runs: the one-step matrix of the people's states, loudness by the straight line, or, with
    ``groups`` (a group index per person), by the route through the groups' centres of mass."""
    n = len(names)
    com = None
    if groups is not None:
        com = [np.mean([pos[i] for i in range(n) if groups[i] == g], axis=0) for g in sorted(set(groups))]
    M = np.zeros((n, n))
    for i in range(n):
        k = KIND[names[i]]
        M[i, i] = 1.1 * k
        for j in range(n):
            if i == j:
                continue
            if groups is None or groups[i] == groups[j]:
                a = loud(pos[i], pos[j])
            else:
                gi, gj = com[groups[i]], com[groups[j]]
                a = float(np.exp(-BETA * (d_eps(np.subtract(pos[i], gi)) + d_eps(gi - gj) + d_eps(np.subtract(gj, pos[j])))))
            M[i, j] = 1.2 * k * a
    return M


def radius(M):
    return float(max(abs(np.linalg.eigvals(M))))


def check_against_matrix(path, M, n):
    x = xs(path, n)
    pred = [x[0]]
    for _ in range(len(x) - 1):
        pred.append(M @ pred[-1])
    return float(np.max(np.abs(x - np.asarray(pred))))


# ---------------------------------------------------------------------------
# Piece (i): positions held.
# ---------------------------------------------------------------------------

def couple(s, steps=20):
    path = run(fill(group(2, "couple"), [person("alice", (0.0, 0.0)), person("bob", (s, 0.0))]), steps)
    return xs(path, 2)


def test_couple_threshold():
    """Prediction 1: the couple alone; two steps multiply both states by 1.21 - 1.44 a(s)^2; settles iff s < 0.9574."""
    s_star = np.sqrt((np.log(1 / A_STAR) / BETA) ** 2 - EPS ** 2)
    print(f"a* = {A_STAR:.5f}, s* = {s_star:.4f}")
    for s in (0.5, 0.8, 0.9, 0.95, 0.97, 1.0, 1.1, 1.5):
        x = couple(s)
        a = loud((0, 0), (s, 0))
        factor = 1.21 - 1.44 * a * a
        ratios = x[2:] / x[:-2]
        gap = float(np.max(np.abs(ratios - factor)))
        print(f"s = {s:.2f}: loudness {a:.4f}, two-step factor {factor:.5f}, run ratio gap {gap:.1e}, "
              f"|x| at 0, 20: {np.abs(x[0]).max():.4f}, {np.abs(x[-1]).max():.4f}, "
              f"{'settles' if factor < 1 else 'runs away'}")
        assert gap < 1e-10
        assert (factor < 1) == (s < s_star)


FAR = {"alice": (0.0, 0.0), "bob": (0.5, 0.0), "carol": (5.0, 0.0), "dan": (5.5, 0.0)}
NAMES = ("alice", "bob", "carol", "dan")


def people(where):
    return [person(n, where[n]) for n in NAMES]


def test_two_couples_far_apart():
    """Prediction 2: flat and nested both settle; nested bracketed both ways and with a unit over Dan run the same."""
    steps = 30
    pos = [FAR[n] for n in NAMES]
    flat = run(fill(group(4, "room"), people(FAR)), steps)
    ps = people(FAR)
    c1, c2, parent = group(2, "couple1"), group(2, "couple2"), group(2, "parent")
    nested_a = compose_seq(tensor([fill(c1, ps[:2]), fill(c2, ps[2:])]), parent.arr())
    nested_b = fill(substitute(parent, [c1, c2]), ps)
    dan_u = compose_seq(ps[3], unit(1, E).arr())
    nested_u = compose_seq(tensor([fill(c1, ps[:2]), fill(c2, [ps[2], dan_u])]), parent.arr())
    print("nested composite:", nested_a.label)
    print("with a unit:", nested_u.label)
    runs = {k: run(a, steps) for k, a in (("a", nested_a), ("b", nested_b), ("u", nested_u))}
    gap_ab = float(np.max(np.abs(runs["a"] - runs["b"])))
    gap_au = float(np.max(np.abs(runs["a"] - runs["u"])))
    Mf, Mn = step_matrix(NAMES, pos), step_matrix(NAMES, pos, groups=[0, 0, 1, 1])
    print(f"radius flat {radius(Mf):.4f}, nested {radius(Mn):.4f}, couple alone {np.sqrt(1.21 - 1.44 * loud((0, 0), (0.5, 0)) ** 2):.4f}")
    print(f"flat vs its matrix {check_against_matrix(flat, Mf, 4):.1e}; nested vs its matrix "
          f"{check_against_matrix(runs['a'], Mn, 4):.1e}; brackets {gap_ab:.1e}; unit {gap_au:.1e}")
    print("|x| at 0, 30: flat", np.abs(xs(flat, 4)[0]).max(), np.abs(xs(flat, 4)[-1]).max(),
          "nested", np.abs(xs(runs['a'], 4)[-1]).max())
    assert check_against_matrix(flat, Mf, 4) < 1e-12 and check_against_matrix(runs["a"], Mn, 4) < 1e-12
    assert gap_ab < 1e-12 and gap_au < 1e-12
    assert radius(Mf) < 1 and radius(Mn) < 1


def pairs_alone(where):
    out = {}
    for i, j in ((0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)):
        names = (NAMES[i], NAMES[j])
        x = xs(run(fill(group(2, "pair"), [person(n, where[n]) for n in names]), 20), 2)
        M = step_matrix(names, [where[n] for n in names])
        grow = float(np.abs(x[-1]).max() / np.abs(x[0]).max())
        out[names] = (radius(M), grow)
        print(f"{names[0]:>5} + {names[1]:<5}: radius {radius(M):.4f}, |x| grew x{grow:.3g} over 20 steps, "
              f"{'vouches' if radius(M) < 1 else 'runs away'}")
    return out


def test_which_pairs_vouch():
    """Prediction 3: at FAR, the couples settle alone, every crossed pair runs away."""
    out = pairs_alone(FAR)
    assert out[("alice", "bob")][0] < 1 and out[("carol", "dan")][0] < 1
    for k in (("alice", "carol"), ("bob", "dan"), ("alice", "dan"), ("bob", "carol")):
        assert out[k][0] > 1 and out[k][1] > 1


def test_moving_people_changes_the_wholes():
    """Prediction 4: Bob and Dan swap places; now Alice+Dan and Bob+Carol vouch, the old couples run away."""
    swapped = dict(FAR, bob=FAR["dan"], dan=FAR["bob"])
    out = pairs_alone(swapped)
    assert out[("alice", "dan")][0] < 1 and out[("bob", "carol")][0] < 1
    assert out[("alice", "bob")][0] > 1 and out[("carol", "dan")][0] > 1
    whole = run(fill(group(4, "room"), people(swapped)), 30)
    M = step_matrix(NAMES, [swapped[n] for n in NAMES])
    print(f"swapped whole: radius {radius(M):.4f}, vs matrix {check_against_matrix(whole, M, 4):.1e}")
    assert radius(M) < 1 and check_against_matrix(whole, M, 4) < 1e-12


# ---------------------------------------------------------------------------
# Piece (ii): Bob walks under the chapter's cost (ex.learning_trust), the form x = 0 as his truth.
# ---------------------------------------------------------------------------

KAPPA = 1e-3
S_STAR = float(np.sqrt((np.log(1 / A_STAR) / BETA) ** 2 - EPS ** 2))


def bob_cost(n):
    """``kappa/2 |msg(0) - heard|^2``: what he hears compared with the form, both at 0."""
    return 0.5 * KAPPA * ((n[0] - 0.0) ** 2 + (n[1] - 1.0) ** 2)


def walking_couple(s0, r, with_unit=False):
    bob = person("bob", (s0, 0.0), mobility=r / KAPPA, cost=bob_cost)
    if with_unit:
        bob = compose_seq(bob, unit(1, E).arr())
    return fill(group(2, "couple"), [person("alice", (0.0, 0.0)), bob])


def recurrence(s0, r, steps):
    """Beside the run: the separation and the two states, by the formulas of RESULTS.md section 4."""
    xa, xb, s = START["alice"], START["bob"], s0
    out = [(s, xa, xb)]
    for _ in range(steps):
        d = np.sqrt(s * s + EPS * EPS)
        a = np.exp(-BETA * d)
        xa, xb, s = (1.1 * xa + 1.2 * a * xb - KAPPA * a * a * xa, -1.1 * xb - 1.2 * a * xa,
                     s + r * BETA * a * (s / d) * (a * xa * xa + a - 1))
        out.append((s, xa, xb))
    return np.asarray(out)


def walk(s0, r, steps, with_unit=False):
    path = run(walking_couple(s0, r, with_unit), steps)
    got = np.stack([path[:, 3], path[:, E], path[:, 3 + E]], axis=1)   # Bob's first coordinate, x_A, x_B
    want = recurrence(s0, r, steps)
    ok = np.allclose(got, want, rtol=1e-8, atol=1e-14)
    first_in = next((t for t, s in enumerate(got[:, 0]) if s < S_STAR), None)
    marks = [k for k in (0, 25, 50, 82, 100, 200, 300) if k <= steps]
    print(f"s0 = {s0}, r = {r}{' (unit over Bob)' if with_unit else ''}: matches recurrence {ok}; "
          f"crosses s* at step {first_in}; least s {got[:, 0].min():.4f} at step {int(got[:, 0].argmin())}; "
          f"s at {marks}: {[float(f'{got[k, 0]:.4g}') for k in marks]}; |x_A| at the end {abs(got[-1, 1]):.3g}; "
          f"Bob's off-line coordinate {abs(path[-1, 4]):.1e}")
    return got, ok


def test_fast_bob_walks_in_and_stops_on_alice():
    """Prediction (ii) 1 and 5: from 1.5, r = 1, he crosses s* at step 3 and is within 1e-3 of Alice by step 50."""
    got, ok = walk(1.5, 1.0, 60)
    assert ok and got[3, 0] < S_STAR < got[2, 0] and np.all(got[50:, 0] < 1e-3)
    u, ok_u = walk(1.5, 1.0, 15, with_unit=True)
    assert ok_u and np.allclose(u, got[:16], rtol=1e-12, atol=1e-16)


def test_close_bob_still_walks_to_alice():
    """Prediction (ii) 4: from 0.5, already settling, he still walks to Alice."""
    got, ok = walk(0.5, 1.0, 60)
    assert ok and np.all(got[50:, 0] < 1e-3)


def test_slower_bob_gets_in():
    """Prediction (ii) 2: r = 0.05 crosses at step 54, s about 0.41 at step 100, the pair settles."""
    got, ok = walk(1.5, 0.05, 150)
    assert ok and got[54, 0] < S_STAR < got[53, 0] and abs(got[100, 0] - 0.4064) < 1e-3
    assert abs(got[-1, 1]) < 1e-6


def test_too_slow_bob_turns_away():
    """Prediction (ii) 3: r = 0.02 reaches 1.231 at step 82, never the threshold, then walks away from the runaway."""
    got, ok = walk(1.5, 0.02, 300)
    assert ok and int(got[:, 0].argmin()) == 82 and abs(got[:, 0].min() - 1.231) < 1e-3
    assert got[-1, 0] > 18 and abs(got[-1, 1]) > 1e8
