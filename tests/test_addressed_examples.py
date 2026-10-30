"""The positioned examples of the book's Ch 4 ``sec.addressed_messages``, in the addressed operad (book plan P79 (c)).

The predictions, written before this file, and the results are in the book repository,
``notes/runs/p79c-71075bb8/RESULTS.md``. Built with ``addressed_one.py``'s ``AddrOp`` at one address space ``R^2``, each
part sending its position as both its address and its query, so the interface is the book's
``addrint_((R^2, R))`` exactly. P77 (b) and (c) ran the same models in ``positioned.py``, which sends a position as one
cone message instead (``tests/test_positioned.py``, ``tests/test_positioned_people.py``).

Since 2026-10-30 an addressed arrangement may have any effort field (David: "yah there's no reason to restrict to exact
effort fields"), and ``AddrOp`` takes one (``omega``) beside its cost; ``substitute`` pulls each constituent's back and
sums them. A whole is filled both ways, ``substitute`` in the operad and ``compose_seq`` of the images in ``arr``, and
the two are checked to run the same. What runs in ``arr``: every whole, run by ``Phi``.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.addressed_one import AddrOp, OpData, substitute, unit
from arrangements.coalgebra import configuration
from arrangements.functors import Phi
from arrangements.rvect import diagonal, euclidean, trivial
from arrangements.wiring import compose_seq, tensor_arrangements

from tests.chain import flat_state

jax.config.update("jax_enable_x64", True)

E = 2
I2 = jnp.eye(2)


def d_eps(v, eps):
    return jnp.sqrt(jnp.sum(v * v) + eps ** 2)


def run(arr, steps, keep=None):
    """The closed whole run by ``Phi``; flattened states (or the coordinates ``keep``)."""
    O = Phi(arr)
    state = O.state
    out = [flat_state(state)]
    for _ in range(steps):
        _, _, state = O.with_state(state).run_one((jnp.zeros(0), lambda x: x), lambda _o: (
            jnp.zeros(arr.out_dim_N), jnp.zeros(arr.in_dim_N)))
        out.append(flat_state(state))
    out = np.asarray(out)
    return out if keep is None else out[:, keep]


def group(n, loud, shown=(), place=None, label="group"):
    """A group of ``n`` one-slot members of value dimension 1, every slot hearing every other (not itself) at
    ``loud(query, address)`` times the identity; with ``shown=(1,)`` it shows ``place(Z)`` and its outside hears and is
    heard by the same law at that place. Matter silent and stateless."""
    members = ((1,),) * n

    def data(q, Z):
        T = {((i, 0), (j, 0)): loud(Z[(i, 0)][1], Z[(j, 0)][0]) * I2
             for i in range(n) for j in range(n) if i != j}
        if not shown:
            return OpData(place=[], T=T)
        k0, p0 = place(Z)
        U = {(0, (j, 0)): loud(p0, Z[(j, 0)][0]) * I2 for j in range(n)}
        D = {((i, 0), 0): loud(Z[(i, 0)][1], k0) * I2 for i in range(n)}
        return OpData(place=[(k0, p0)], U=U, D=D, T=T)

    return AddrOp(members=members, shown=tuple(shown), dim_P=E, P=configuration(trivial()), data=data, label=label)


def mean_place(n):
    return lambda Z: tuple(sum(Z[(i, 0)][r] for i in range(n)) / n for r in (0, 1))


# ---------------------------------------------------------------------------
# Four parts: couples that settle when near (RESULTS §1, predictions 1-5).
# ---------------------------------------------------------------------------

BETA, EPS = 1.0, 0.1
LAM, K = 1.1, 1.2
KIND = {"alice": 1, "bob": -1, "carol": 1, "dan": -1}   # enthusiasts +1, skeptics -1
START = {"alice": 0.02, "bob": -0.01, "carol": -0.015, "dan": 0.025}


def ear(u, v):
    return jnp.exp(-BETA * d_eps(u - v, EPS))


def part(name, p):
    """A nullary addressed arrangement of color ((R^2, R)): its matter stores a position ``p``,
    which does not move, and a number ``x``; it sends ``(p, p)`` and ``msg(x) = (x, 1)``, hears ``(h, w)``, and its
    effort field is ``((lam - 1) x + k h) dx`` (enthusiast) or ``(-(lam + 1) x - k h) dx`` (skeptic), none on what it
    hears: not exact, since its exterior derivative is ``+-k dh ^ dx``."""
    s = KIND[name]
    P = configuration(diagonal([0.0] * E + [1.0]), init=jnp.asarray([*p, START[name]], dtype=float))

    def data(q, Z):
        return OpData(place=[(q[:E], q[:E])], Uv={0: jnp.stack([q[E], jnp.asarray(1.0)])})

    def omega(q, m_out, n_in):
        x, h = q[E], n_in[0]
        xi = (LAM * s - 1.0) * x + K * s * h
        return jnp.concatenate([jnp.zeros(E), jnp.stack([xi])]), jnp.zeros(0), jnp.zeros(2)

    return AddrOp(members=(), shown=(1,), dim_P=E, P=P, data=data, omega=omega, label=name)


def fill(op_arr, parts):
    """``op_arr`` filled by the images of ``parts`` in ``arr``: the composite in ``arr``."""
    return compose_seq(tensor_arrangements([p.arr() for p in parts]), op_arr)


def xs(path, n):
    return path[:, [3 * i + E for i in range(n)]]


def growth(x, lag=2):
    """The largest ratio |x(t + lag)| / |x(t)| over the last steps, per lag: a settling whole has it below 1."""
    nrm = np.linalg.norm(x, axis=1)
    return float(np.max(nrm[-5:] / nrm[-5 - lag:-lag]))


def radius(names, pos, steps=40):
    """Beside the runs: the spectral radius of the one-step map of the states, loudness by the straight line."""
    n = len(names)
    M = np.zeros((n, n))
    for i in range(n):
        M[i, i] = LAM * KIND[names[i]]
        for j in range(n):
            if i != j:
                M[i, j] = K * KIND[names[i]] * float(ear(jnp.asarray(pos[i]), jnp.asarray(pos[j])))
    return float(max(abs(np.linalg.eigvals(M)))), M


def test_couple_threshold():
    """Prediction 1: the couple's two-step ratio is lam^2 - k^2 a^2 at every separation; it settles below the
    threshold, near 0.957, and runs away above it."""
    for s in (0.5, 0.8, 0.9, 0.95, 0.97, 1.0, 1.1, 1.5):
        x = xs(run(fill(group(2, ear).arr(), [part("alice", (0.0, 0.0)), part("bob", (s, 0.0))]), 20), 2)
        a = float(ear(jnp.zeros(2), jnp.array([s, 0.0])))
        want = LAM ** 2 - K ** 2 * a ** 2
        ratio = x[2:] / x[:-2]
        print(f"s={s}: two-step ratio {ratio[-1]}, predicted {want:.4f}")
        assert np.max(np.abs(ratio - want)) < 1e-10
        assert (abs(want) < 1) == (s < 0.9574)


POS = {"alice": (0.0, 0.0), "bob": (0.5, 0.0), "carol": (5.0, 0.0), "dan": (5.5, 0.0)}
NAMES = ["alice", "bob", "carol", "dan"]


def test_two_couples_flat_unit_and_nested():
    """Prediction 2: two couples far apart settle, flat (radius 0.8338), with a unit over Dan the same, and nested in
    two groups showing the mean of their places (radius 0.8332), both bracketings agreeing."""
    parts = [part(n, POS[n]) for n in NAMES]
    flat = xs(run(fill(group(4, ear).arr(), parts), 30), 4)
    r, M = radius(NAMES, [POS[n] for n in NAMES])
    pred = [flat[0]]
    for _ in range(30):
        pred.append(M @ pred[-1])
    print(f"flat: radius {r:.4f}, |x| {np.linalg.norm(flat[0]):.3e} -> {np.linalg.norm(flat[-1]):.3e}, "
          f"against the matrix {np.max(np.abs(flat - np.asarray(pred))):.1e}")
    assert abs(r - 0.8338) < 5e-4 and np.max(np.abs(flat - np.asarray(pred))) < 1e-14
    with_unit = substitute(group(4, ear), [unit((1,), E), unit((1,), E), unit((1,), E), unit((1,), E)])
    u = xs(run(fill(with_unit.arr(), parts), 30), 4)
    print("four units inserted:", np.max(np.abs(u - flat)))
    assert np.max(np.abs(u - flat)) < 1e-14
    couple1 = group(2, ear, shown=(1,), place=mean_place(2), label="couple1")
    couple2 = group(2, ear, shown=(1,), place=mean_place(2), label="couple2")
    parent = group(2, ear, label="parent")
    nested_op = substitute(parent, [couple1, couple2])
    one = fill(nested_op.arr(), parts)
    two = compose_seq(tensor_arrangements([fill(couple1.arr(), parts[:2]), fill(couple2.arr(), parts[2:])]),
                      parent.arr())
    three = substitute(nested_op, parts).arr()
    print("bracketings:", one.label, "|", two.label, "|", three.label)
    n1, n2 = xs(run(one, 30), 4), xs(run(two, 30), 4)
    n3 = xs(run(three, 30), 4)
    print(f"substituted in the operad, whole and run: against the composite in arr {np.max(np.abs(n3 - n1)):.1e}")
    assert np.max(np.abs(n3 - n1)) < 1e-12
    rate = (np.linalg.norm(n1[-1]) / np.linalg.norm(n1[-21])) ** (1 / 20)
    print(f"nested: bracketings agree to {np.max(np.abs(n1 - n2)):.1e}, |x| {np.linalg.norm(n1[0]):.3e} -> "
          f"{np.linalg.norm(n1[-1]):.3e}, rate per step {rate:.4f}")
    assert np.max(np.abs(n1 - n2)) < 1e-12 and np.linalg.norm(n1[-1]) < 1e-2 * np.linalg.norm(n1[0])


def pairs(pos):
    out = {}
    for i in range(4):
        for j in range(i + 1, 4):
            a, b = NAMES[i], NAMES[j]
            x = xs(run(fill(group(2, ear).arr(), [part(a, pos[a]), part(b, pos[b])]), 20), 2)
            out[(a, b)] = (radius([a, b], [pos[a], pos[b]])[0], np.linalg.norm(x[-1]) / np.linalg.norm(x[0]))
    return out


def test_which_pairs_settle_and_moving_swaps_them():
    """Predictions 3 and 4: at the positions above only the couples settle alone; with Bob and Dan swapped, Alice+Dan
    and Bob+Carol settle and the old couples do not, and the whole still settles."""
    before = pairs(POS)
    swapped = dict(POS, bob=POS["dan"], dan=POS["bob"])
    after = pairs(swapped)
    for k in before:
        print(k, "radius %.3f, grew x%.3g;" % before[k], "swapped: radius %.3f, grew x%.3g" % after[k])
    settle_before = {k for k, v in before.items() if v[1] < 1}
    settle_after = {k for k, v in after.items() if v[1] < 1}
    assert settle_before == {("alice", "bob"), ("carol", "dan")}
    assert settle_after == {("alice", "dan"), ("bob", "carol")}
    whole = xs(run(fill(group(4, ear).arr(), [part(n, swapped[n]) for n in NAMES]), 30), 4)
    print("swapped whole: |x|", np.linalg.norm(whole[0]), "->", np.linalg.norm(whole[-1]))
    assert np.linalg.norm(whole[-1]) < 1e-2 * np.linalg.norm(whole[0])


# ---------------------------------------------------------------------------
# Two houses on a call (RESULTS §1, predictions 6-8): space-like inside each house, zoom-like between them.
# ---------------------------------------------------------------------------

BETA_H, ALPHA, SIG = 0.3, 3.0, 0.2
DOORWAY = jnp.array([0.0, 0.0])


def house_sound(alpha=ALPHA, doorway=True):
    """Loudness in a house with a wall along x = 0 and a doorway at the origin: the direct path, damped by ``alpha``
    when the two places are on opposite sides (a smooth indicator), plus the path through the doorway."""
    def side(u):
        return jax.nn.sigmoid(u[0] / SIG)

    def loud(u, v):
        across = side(u) * (1 - side(v)) + (1 - side(u)) * side(v)
        direct = jnp.exp(-BETA_H * d_eps(u - v, EPS) - alpha * across)
        if not doorway:
            return direct
        return direct + jnp.exp(-BETA_H * (d_eps(u - DOORWAY, EPS) + d_eps(DOORWAY - v, EPS)))
    return loud


def house(n, laptop, loud, label):
    """A house: its matter stores where its laptop is, the place it shows; inside, every slot and the laptop hear one
    another by ``loud``."""
    lap = jnp.asarray(laptop, dtype=float)
    return group(n, loud, shown=(1,), place=lambda Z: (lap, lap), label=label)


def stander(z, speaks=False, label=""):
    """A nullary operation standing at ``z``: it sends ``(z, z)``, and ``msg(1)`` if it speaks, else nothing."""
    z = jnp.asarray(z, dtype=float)
    voice = jnp.array([1.0, 1.0]) if speaks else jnp.zeros(2)
    return AddrOp(members=(), shown=(1,), dim_P=E, P=configuration(trivial()),
                  data=lambda q, Z: OpData(place=[(z, z)], Uv={0: voice}), label=label)


def walker(z, eta=0.5):
    """Bob: his configuration is his position, sent as his address and query; his cost is minus the log of the weight
    he hears, whose force reaches his position back through the hearing's dependence on his query."""
    return AddrOp(members=(), shown=(1,), dim_P=E, P=configuration(euclidean(E, eta), init=jnp.asarray(z, dtype=float)),
                  data=lambda q, Z: OpData(place=[(q, q)]), cost=lambda q, m_out, n_in: -jnp.log(n_in[-1]),
                  label="bob")


def call_run(alpha, steps, alice=(1.0, -1.0), lens_steps=0, doorway=True):
    house_a = house(2, (3.0, 1.0), house_sound(alpha, doorway), "houseA")
    house_b = house(1, (0.0, 2.0), house_sound(), "houseB")
    call = AddrOp(members=((1,), (1,)), shown=(), dim_P=E, P=configuration(trivial()),
                  data=lambda q, Z: OpData(place=[], T={((0, 0), (1, 0)): I2, ((1, 0), (0, 0)): I2}), label="call")
    people = [walker((-3.0, 2.0)), stander((-2.0, -2.0), label="chuck"), stander(alice, speaks=True, label="alice")]
    whole = substitute(substitute(call, [house_a, house_b]), people).arr()
    lens = compose_seq(tensor_arrangements([compose_seq(tensor_arrangements([p.arr() for p in people[:2]]),
                                                        house_a.arr()),
                                            compose_seq(people[2].arr(), house_b.arr())]), call.arr())
    path = run(whole, steps, keep=slice(0, E))
    gap = 0.0
    if lens_steps:
        gap = float(np.max(np.abs(path[:lens_steps + 1] - run(lens, lens_steps, keep=slice(0, E)))))
    return lens.label, path, gap


def crossing(path):
    for a, b in zip(path, path[1:]):
        if a[0] < 0 <= b[0]:
            t = -a[0] / (b[0] - a[0])
            return float(a[1] + t * (b[1] - a[1]))
    return None


def test_bob_walks_through_his_doorway_to_the_laptop():
    """Prediction 6: Bob crosses the wall near the doorway (y about 0.17) and rests about 0.016 from the laptop; the
    explicit composite and the composite in arr agree."""
    label, path, gap = call_run(ALPHA, 90, lens_steps=15)
    y = crossing(path)
    rest = float(np.linalg.norm(path[-1] - path[-11]))
    off = float(np.linalg.norm(path[-1] - np.array([3.0, 1.0])))
    print("lens composite:", label)
    print("Bob at steps 0, 10, 20, 30, 45, 90:", [np.round(path[k], 3).tolist() for k in (0, 10, 20, 30, 45, 90)],
          f"crosses x=0 at y={y}, explicit vs lens {gap:.1e}, from the laptop {off:.4f}, last 10 steps {rest:.1e}")
    assert gap < 1e-12 and y is not None and abs(y - 0.17) < 0.02 and abs(off - 0.016) < 0.002 and rest < 1e-3


def test_open_air_bob_walks_straight():
    """Prediction 7: the direct path alone, he crosses x = 0 at y = 1.5, the straight line to the laptop."""
    _, path, _ = call_run(0.0, 30, doorway=False)
    y = crossing(path)
    print("open air: crosses x=0 at y =", y)
    assert y is not None and abs(y - 1.5) < 0.01


def test_the_call_hides_house_b_inside():
    """Prediction 8: moving Alice inside house B leaves Bob's path unchanged."""
    _, p1, _ = call_run(ALPHA, 25)
    _, p2, _ = call_run(ALPHA, 25, alice=(-2.0, 3.0))
    gap = float(np.max(np.abs(p1 - p2)))
    print("Alice moved inside house B: Bob's path changes by", gap)
    assert gap < 1e-12
