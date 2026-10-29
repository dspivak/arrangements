"""The addressed operad after the matter's own messages and P77, the book's attention operad, and ``j`` between them
(``addressed_one.py``; book plan P79 (b)).

The definition, the predictions written before this file and the results are in the book repository,
``notes/runs/p79-b430b0e2/RESULTS.md``. Each docstring says which prediction it checks. What runs in ``arr``: every
operation is built as an ``Arrangement``; the composite in ``arr`` is ``compose_seq(tensor_arrangements(parts), parent)``,
and the explicit and lens composites are run through ``Phi`` side by side.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements import addressed as old
from arrangements.addressed_one import (
    AddrOp, AttOp, OpData, V, att_substitute, att_unit, forget_places, j, substitute, unit,
)
from arrangements.coalgebra import configuration
from arrangements.functors import Phi
from arrangements.rvect import euclidean, trivial
from arrangements.wiring import compose_seq, tensor_arrangements

from tests.chain import flat_state
from tests.test_addressed import random_op as old_random_op, DH as ODH, DP as ODP

jax.config.update("jax_enable_x64", True)

DP, TAU = 2, 0.8


# ---------------------------------------------------------------------------
# Random data: cone maps smooth in (q; Z).
# ---------------------------------------------------------------------------


def cone_map(r, dx, dy, qdim, kernel=None):
    """``(q, Z) |-> e^{s} [[A(q), b], [0, 1]]``, ``s`` linear in ``q`` plus an optional kernel term in ``Z``."""
    A0, A1 = r.standard_normal((dx, dy)) * 0.5, r.standard_normal((dx, dy)) * 0.3
    b, c = r.standard_normal(dx) * 0.5, r.standard_normal(qdim) * 0.3

    def f(q, Z):
        A = A0 + (A1 * q[0] if qdim else 0.0)
        M = jnp.zeros((dx + 1, dy + 1))
        M = M.at[:dx, :dy].set(A).at[:dx, dy].set(b).at[dx, dy].set(1.0)
        s = (jnp.dot(c, q) if qdim else 0.0) + (kernel(Z) if kernel is not None else 0.0)
        return jnp.exp(s) * M
    return f


def cone_point(r, d, qdim):
    h, c = r.standard_normal(d) * 0.5, r.standard_normal(qdim) * 0.3

    def f(q, Z):
        g = jnp.exp(jnp.dot(c, q)) if qdim else 1.0
        return g * jnp.concatenate([jnp.asarray(h), jnp.ones(1)])
    return f


def gauss(x, y):
    """The hearing's kernel on sent places: the receiver's query against the sender's address."""
    return lambda Z: -jnp.sum((Z[x][1] - Z[y][0]) ** 2) / (2 * TAU)


def random_addr(members, shown, seed, forwards=(), qdim=3, mobility=0.05, label=None):
    """An operation ``members -> shown``. ``forwards`` maps a shown slot to the member slot it forwards; every other
    shown slot is a speaker whose place is a smooth function of ``q`` and of the members' places, and whose up and down
    transfers read ``q`` and ``Z``. Every member slot hears every member slot (itself included) by a Gaussian of the
    sent places; the matter sends a message to every slot; the cost reads everything."""
    r = np.random.default_rng(seed)
    members, shown = tuple(tuple(m) for m in members), tuple(shown)
    fw = dict(forwards)
    slots = [(i, a) for i, m in enumerate(members) for a in range(len(m))]
    dim = {x: members[x[0]][x[1]] for x in slots}
    P = configuration(euclidean(qdim, mobility), init=jnp.asarray(r.standard_normal(qdim) * 0.5))
    W = {u: r.standard_normal((2, DP, qdim)) * 0.3 for u in range(len(shown))}
    U = {(u, y): cone_map(r, shown[u], dim[y], qdim, kernel=lambda Z, y=y: 0.2 * jnp.tanh(jnp.sum(Z[y][0])))
         for u in range(len(shown)) if u not in fw for y in slots}
    D = {(x, u): cone_map(r, dim[x], shown[u], qdim, kernel=lambda Z, x=x: 0.2 * jnp.sin(jnp.sum(Z[x][1])))
         for u in range(len(shown)) if u not in fw for x in slots}
    T = {(x, y): cone_map(r, dim[x], dim[y], qdim, kernel=gauss(x, y)) for x in slots for y in slots}
    Uv = {u: cone_point(r, shown[u], qdim) for u in range(len(shown))}
    Tv = {x: cone_point(r, dim[x], qdim) for x in slots}

    def data(q, Z):
        place = []
        for u in range(len(shown)):
            if u in fw:
                place.append(Z[fw[u]])
            else:
                mean_k = sum((Z[y][0] for y in slots), jnp.zeros(DP)) / max(len(slots), 1)
                place.append((W[u][0] @ q + jnp.tanh(mean_k), W[u][1] @ q + 0.5 * mean_k))
        Ud = {k: f(q, Z) for k, f in U.items()}
        Dd = {k: f(q, Z) for k, f in D.items()}
        for u, x in fw.items():
            Ud[(u, x)] = jnp.eye(dim[x] + 1)
            Dd[(x, u)] = jnp.eye(dim[x] + 1)
        return OpData(place=place, U=Ud, Uv={u: f(q, Z) for u, f in Uv.items()}, D=Dd,
                      T={k: f(q, Z) for k, f in T.items()}, Tv={k: f(q, Z) for k, f in Tv.items()})

    od = sum(2 * DP + dim[x] + 1 for x in slots)
    idim = sum(d + 1 for d in shown)
    A = jnp.asarray(r.standard_normal((idim, od)) * 0.2)
    c = jnp.asarray(r.standard_normal(idim) * 0.2)

    def cost(q, m_out, n_in):
        dd = n_in - A @ m_out - c - 0.1 * jnp.sum(jnp.sin(q))
        return 0.5 * jnp.sum(dd * dd) + 0.05 * jnp.sum(jnp.tanh(m_out)) ** 2

    return AddrOp(members=members, shown=shown, dim_P=DP, P=P, data=data, cost=cost, label=label or f"a{seed}")


def random_att(dims, seed, qdim=3, mobility=0.05, label=None):
    """A book attention arrangement with every transfer nonzero (``T_ii`` included, ``T_00 = 0``) and voices."""
    r = np.random.default_rng(seed)
    n = len(dims) - 1
    P = configuration(euclidean(qdim, mobility), init=jnp.asarray(r.standard_normal(qdim) * 0.5)) if qdim \
        else configuration(trivial())
    maps = {(i, k): cone_map(r, dims[i], dims[k], qdim) for i in range(n + 1) for k in range(n + 1) if (i, k) != (0, 0)}
    voices = {i: cone_point(r, dims[i], qdim) for i in range(n + 1)}
    s = sum(d + 1 for d in dims[1:])
    A = jnp.asarray(r.standard_normal((dims[0] + 1, s)) * 0.3)

    def transfers(q):
        T = {k: f(q, None) for k, f in maps.items()}
        T.update({(i, V): f(q, None) for i, f in voices.items()})
        return T

    def cost(q, m_out, n_in):
        dd = n_in - A @ m_out - (0.1 * jnp.sum(jnp.sin(q)) if qdim else 0.0)
        return 0.5 * jnp.sum(dd * dd)

    return AttOp(dims=tuple(dims), P=P, transfers=transfers, cost=cost, label=label or f"att{seed}")


# ---------------------------------------------------------------------------
# Comparisons.
# ---------------------------------------------------------------------------


def random_places(op, r):
    d = op.dim_P
    return {x: (jnp.asarray(r.standard_normal(d)), jnp.asarray(r.standard_normal(d))) for x in op.slots}


def data_gap(x: AddrOp, y: AddrOp, qx, qy, seed, draws=2):
    """The largest difference in any datum (place, U, Uv, D, T, Tv) at random places, and in sending, delivery, cost."""
    assert x.members == y.members and x.shown == y.shown
    r = np.random.default_rng(seed)
    gap = 0.0
    for _ in range(draws):
        Z = random_places(x, r)
        Dx, Dy = x.data(qx, Z), y.data(qy, Z)
        for (a, b), (c, d) in zip(Dx.place, Dy.place):
            if a.size:
                gap = max(gap, float(jnp.max(jnp.abs(a - c))), float(jnp.max(jnp.abs(b - d))))
        for name in ("U", "Uv", "D", "T", "Tv"):
            A, B = getattr(Dx, name), getattr(Dy, name)
            for k in set(A) | set(B):
                a, b = A.get(k), B.get(k)
                ref = a if a is not None else b
                a = jnp.zeros_like(ref) if a is None else a
                b = jnp.zeros_like(ref) if b is None else b
                gap = max(gap, float(jnp.max(jnp.abs(a - b))) if a.size else 0.0)
    for _ in range(3):
        m = jnp.asarray(r.standard_normal(x.out_dim_M()))
        e = jnp.asarray(r.standard_normal(x.in_dim_N()))
        gap = max(gap, float(jnp.max(jnp.abs(x.sigma(qx, m) - y.sigma(qy, m)))) if x.shown else 0.0,
                  float(jnp.max(jnp.abs(x.rho(qx, m, e) - y.rho(qy, m, e)))) if x.slots else 0.0,
                  float(jnp.abs(x.cost(qx, m, e) - y.cost(qy, m, e))))
    return gap


def arr_gap(f, g, seed, draws=4):
    """Same readout, delivery and forcing (all three components) at random points; returns the gaps and the sizes."""
    assert (f.in_dim_M, f.out_dim_M, f.in_dim_N, f.out_dim_N) == (g.in_dim_M, g.out_dim_M, g.in_dim_N, g.out_dim_N)
    r = np.random.default_rng(seed)
    gaps, size = {"out": 0.0, "in": 0.0, "omega": 0.0}, {"out": 0.0, "in": 0.0, "omega": 0.0}
    for _ in range(draws):
        q = jnp.asarray(r.standard_normal(f.P.Q_dim) * 0.5)
        m = jnp.asarray(r.standard_normal(f.out_dim_M))
        e = jnp.asarray(r.standard_normal(f.in_dim_N))
        for key, a, b in [("out", f.out_f(q, m), g.out_f(q, m)), ("in", f.in_f(q, m, e), g.in_f(q, m, e))]:
            if a.size:
                gaps[key] = max(gaps[key], float(jnp.max(jnp.abs(a - b))))
                size[key] = max(size[key], float(jnp.max(jnp.abs(a))))
        for a, b in zip(f.omega(q, m, e), g.omega(q, m, e)):
            if a.size:
                gaps["omega"] = max(gaps["omega"], float(jnp.max(jnp.abs(a - b))))
                size["omega"] = max(size["omega"], float(jnp.max(jnp.abs(a))))
    return gaps, size


def trajectory(arr, steps, seed=0):
    """``Phi(arr)`` on fixed exterior data (as in test_addressed.py); flattened states."""
    r = np.random.default_rng(seed)
    m_out = jnp.asarray(r.standard_normal(arr.out_dim_M) * 0.5)
    B = jnp.asarray(r.standard_normal((arr.in_dim_M, arr.in_dim_M)) * 0.05)
    c = jnp.asarray(r.standard_normal(arr.in_dim_M) * 0.05)
    xi_N = jnp.asarray(r.standard_normal(arr.out_dim_N) * 0.1)
    e = jnp.asarray(r.standard_normal(arr.in_dim_N) * 0.5)
    O = Phi(arr)
    state, out = O.state, [flat_state(O.state)]
    for _ in range(steps):
        _, _, state = O.with_state(state).run_one((m_out, lambda x: B @ x + c), lambda _o: (xi_N, e))
        out.append(flat_state(state))
    return out


def permute_blocks(q, sizes, order):
    cut = [sum(sizes[:i]) for i in range(len(sizes) + 1)]
    return jnp.concatenate([q[cut[i]:cut[i + 1]] for i in order])


# The two-input nest of prediction 1: a parent of two members, colors (1,) and (2, 1); part 1 shows (1,), part 2
# shows (2, 1), one slot forwarded and one a speaker.
def two_input_nest(seed=1):
    a = random_addr([(1,), (2, 1)], (2, 1), seed, forwards={1: (1, 1)}, label="a")
    a1 = random_addr([(1,), (2,)], (1,), seed + 1, label="a1")
    a2 = random_addr([(2,), (1, 2)], (2, 1), seed + 2, forwards={0: (0, 0)}, label="a2")
    return a, a1, a2


# ---------------------------------------------------------------------------
# Prediction 1: unit and a two-input composite, in arr.
# ---------------------------------------------------------------------------


def test_two_input_composite_is_composition_in_arr():
    """Prediction 1: the explicit composite equals the composite in ``arr`` (readout, delivery, forcing), and the two
    run identically through ``Phi``; voices and same-member hearing nonzero at both levels."""
    a, a1, a2 = two_input_nest()
    b = substitute(a, [a1, a2])
    explicit, lens = b.arr(), compose_seq(tensor_arrangements([a1.arr(), a2.arr()]), a.arr())
    print("lens composite:", lens.label)
    gaps, size = arr_gap(explicit, lens, seed=10)
    print("two-input composite vs arr:", gaps, "sizes:", size)
    assert max(gaps.values()) < 1e-12 and min(size.values()) > 0.5
    t1, t2 = trajectory(explicit, 15, seed=7), trajectory(lens, 15, seed=7)
    gap = max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(t1, t2))
    moved = float(jnp.max(jnp.abs(t1[-1] - t1[0])))
    print(f"Phi, 15 steps: gap {gap:.2e}, moved {moved:.3f}")
    assert gap < 1e-12 and moved > 1e-2


def test_unit_laws():
    """Prediction 1: ``1(a) = a`` and ``a(1, ..., 1) = a``, as data and as arrangements."""
    a, _, _ = two_input_nest()
    q = a.P.init
    left = substitute(unit(a.shown, DP), [a])
    right = substitute(a, [unit(m, DP) for m in a.members])
    g1, g2 = data_gap(left, a, q, q, seed=3), data_gap(right, a, q, q, seed=4)
    print("unit laws, data:", g1, g2)
    assert max(g1, g2) < 1e-14
    for here in (left, right):
        gaps, _ = arr_gap(here.arr(), a.arr(), seed=5)
        assert max(gaps.values()) < 1e-13


# ---------------------------------------------------------------------------
# Prediction 2: associativity.
# ---------------------------------------------------------------------------


def test_associative_on_an_uneven_three_level_nest():
    """Prediction 2: both bracketings agree in every datum, configurations reordered."""
    a, a1, a2 = two_input_nest(seed=21)
    g = [random_addr([(2,)], (1,), 31, label="g0"),
         random_addr([(1,), (1,)], (2,), 32, label="g1"),
         random_addr([], (2,), 33, label="g2"),                     # nullary: a system
         random_addr([(1, 1)], (1,), 34, forwards={0: (0, 1)}, label="g3"),
         random_addr([(2,), (1,)], (2,), 35, label="g4")]
    # a1 has members (1,), (2,): g0 shows (1,), g2 shows (2,); a2 has members (2,), (1, 2): g4 shows (2,), ... need (1, 2)
    g_last = random_addr([(1,), (2,)], (1, 2), 36, forwards={1: (1, 0)}, label="g5")
    left = substitute(substitute(a, [a1, a2]), [g[0], g[2], g[4], g_last])          # Q: g0 g2 g4 g5 a1 a2 a
    right = substitute(a, [substitute(a1, [g[0], g[2]]), substitute(a2, [g[4], g_last])])  # g0 g2 a1 g4 g5 a2 a
    sizes = [g[0].P.Q_dim, g[2].P.Q_dim, g[4].P.Q_dim, g_last.P.Q_dim, a1.P.Q_dim, a2.P.Q_dim, a.P.Q_dim]
    ql = jnp.asarray(np.random.default_rng(37).standard_normal(left.P.Q_dim) * 0.5)
    qr = permute_blocks(ql, sizes, [0, 1, 4, 2, 3, 5, 6])
    gap = data_gap(left, right, ql, qr, seed=38)
    print("associativity, data gap:", gap, "member slots:", len(left.slots))
    assert gap < 1e-12
    r = np.random.default_rng(39)
    m, e = jnp.asarray(r.standard_normal(left.out_dim_M())), jnp.asarray(r.standard_normal(left.in_dim_N()))
    xl, xr = left.arr().omega(ql, m, e), right.arr().omega(qr, m, e)
    assert float(jnp.max(jnp.abs(permute_blocks(xl[0], sizes, [0, 1, 4, 2, 3, 5, 6]) - xr[0]))) < 1e-12
    assert float(jnp.max(jnp.abs(xl[1] - xr[1]))) < 1e-12 and float(jnp.max(jnp.abs(xl[2] - xr[2]))) < 1e-12


# ---------------------------------------------------------------------------
# Prediction 3: the extension is needed.
# ---------------------------------------------------------------------------


def test_the_old_rule_misses_the_composite_in_arr():
    """Prediction 3 (control): ``addressed.py``'s rule, no route through the parent within a part and no voices
    travelling, misses the composite in ``arr``."""
    a, a1, a2 = two_input_nest()
    lens = compose_seq(tensor_arrangements([a1.arr(), a2.arr()]), a.arr())
    gaps, _ = arr_gap(substitute(a, [a1, a2], old_rule=True).arr(), lens, seed=10)
    print("old rule vs arr:", gaps)
    assert gaps["in"] > 1e-2


# ---------------------------------------------------------------------------
# Prediction 4: the book's attention operad.
# ---------------------------------------------------------------------------


def test_book_attention_substitution_is_composition_in_arr():
    """Prediction 4: ``eqn.attention_substitution`` is the composite in ``arr`` of ``eqn.attention_arrangement``,
    with voices and ``T_ii`` nonzero, uneven, with a nullary part; and the unit laws."""
    phi = random_att((1, 2, 1, 1), 41, label="phi")
    psis = [random_att((2, 1, 1), 42, label="psi1"), random_att((1,), 43, label="psi2"),
            random_att((1, 2), 44, label="psi3")]
    b = att_substitute(phi, psis)
    lens = compose_seq(tensor_arrangements([p.arr() for p in psis]), phi.arr())
    gaps, size = arr_gap(b.arr(), lens, seed=45)
    print("book attention composite vs arr:", gaps, "sizes:", size)
    assert max(gaps.values()) < 1e-12 and min(size.values()) > 0.1
    for here in (att_substitute(att_unit(phi.dims[0]), [phi]), att_substitute(phi, [att_unit(d) for d in phi.dims[1:]])):
        gaps, _ = arr_gap(here.arr(), phi.arr(), seed=46)
        assert max(gaps.values()) < 1e-13


# ---------------------------------------------------------------------------
# Prediction 5: j is a map of operads.
# ---------------------------------------------------------------------------


def test_j_preserves_units():
    """Prediction 5: ``j(1_V)`` is the addressed unit at ``(V,)``, exactly; the control with place ``0`` is not."""
    q0 = jnp.zeros(0)
    for d in (1, 2):
        gap = data_gap(j(att_unit(d), DP), unit((d,), DP), q0, q0, seed=51)
        bad = data_gap(j(att_unit(d), DP, place="zero"), unit((d,), DP), q0, q0, seed=51)
        print(f"j(unit_{d}) vs unit: {gap}; place 0: {bad:.3f}")
        assert gap == 0.0 and bad > 0.1


def test_j_preserves_substitution():
    """Prediction 5: ``j(phi(psi_1, psi_2, psi_3)) = j(phi)(j(psi_1), j(psi_2), j(psi_3))`` in every datum, a nullary
    part, voices and self-hearing; and twice nested."""
    phi = random_att((1, 2, 1, 1), 61, label="phi")
    psis = [random_att((2, 1, 1), 62, label="psi1"), random_att((1,), 63, label="psi2"),
            random_att((1, 2), 64, label="psi3")]
    grand = [random_att((1, 2), 65), random_att((1,), 66), random_att((2, 1, 2), 67)]   # the composite's 3 boxes
    once_here = substitute(j(phi, DP), [j(p, DP) for p in psis])
    once_there = j(att_substitute(phi, psis), DP)
    q = jnp.asarray(np.random.default_rng(69).standard_normal(once_here.P.Q_dim) * 0.5)
    g1 = data_gap(once_here, once_there, q, q, seed=70)
    twice_here = substitute(once_here, [j(c, DP) for c in grand])
    twice_there = j(att_substitute(att_substitute(phi, psis), grand), DP)
    q2 = jnp.asarray(np.random.default_rng(71).standard_normal(twice_here.P.Q_dim) * 0.5)
    g2 = data_gap(twice_here, twice_there, q2, q2, seed=72)
    print("j substitution, once:", g1, "twice:", g2)
    assert max(g1, g2) < 1e-12
    gaps, _ = arr_gap(once_here.arr(), compose_seq(tensor_arrangements([j(p, DP).arr() for p in psis]),
                                                   j(phi, DP).arr()), seed=73)
    print("j composite vs arr:", gaps)
    assert max(gaps.values()) < 1e-12


# ---------------------------------------------------------------------------
# Prediction 6: forgetting places relates the two maps into arr.
# ---------------------------------------------------------------------------


def test_forgetting_places_is_a_transformation():
    """Prediction 6: ``p . j(phi) = phi . (p (x) ... (x) p)`` in ``arr``."""
    phi = random_att((1, 2, 1, 1), 81, label="phi")
    lhs = compose_seq(j(phi, DP).arr(), forget_places(phi.dims[0], DP))
    rhs = compose_seq(tensor_arrangements([forget_places(d, DP) for d in phi.dims[1:]]), phi.arr())
    gaps, size = arr_gap(lhs, rhs, seed=82)
    print("p . j(phi) vs phi . p^n:", gaps, size)
    assert max(gaps.values()) < 1e-13


# ---------------------------------------------------------------------------
# Prediction 7: in addressed.py, a placement is absorbed except in the delivered position.
# ---------------------------------------------------------------------------


def absorbed(a: old.AddressedArrangement) -> old.AddressedArrangement:
    """The same operation with placement 0 and the placement folded into its places, transfers and hearing."""
    def shift(q, Z):
        ell = a.placement(q)
        return {x: (k + ell[x[0]], p + ell[x[0]]) for x, (k, p) in Z.items()}

    def lift(e):
        return old.Shown(place=lambda q, Z: e.place(q, shift(q, Z)), up=lambda q, y, Z: e.up(q, y, shift(q, Z)),
                         down=lambda q, x, Z: e.down(q, x, shift(q, Z)), label=e.label)

    return old.AddressedArrangement(
        slots=a.slots, exposure=tuple(lift(e) for e in a.shown), dim_H=a.dim_H, dim_P=a.dim_P, P=a.P,
        placement=lambda q: jnp.zeros((a.n, a.dim_P)), hearing=lambda q, x, y, Z: a.hearing(q, x, y, shift(q, Z)),
        cost=a.cost, label=a.label + "~")


def test_placement_is_absorbed_but_for_the_delivered_position():
    """Prediction 7: same sending and same heard messages; only the delivered positions differ, by ``l_i(q)``."""
    a = old_random_op((2, 1), [(0, 1), (1, 0)], 91, speakers=1)
    b = absorbed(a)
    r = np.random.default_rng(92)
    hd = ODH + 1
    heard_gap, send_gap, pos_gap = 0.0, 0.0, 0.0
    for _ in range(4):
        q = jnp.asarray(r.standard_normal(a.P.Q_dim) * 0.5)
        m = jnp.asarray(r.standard_normal(sum(old.out_dim(ri, ODH, ODP) for ri in a.slots)))
        e = jnp.asarray(r.standard_normal(old.in_dim(a.r, ODH, ODP)))
        send_gap = max(send_gap, float(jnp.max(jnp.abs(a.sigma(q, m) - b.sigma(q, m)))))
        ra, rb, c = a.rho(q, m, e), b.rho(q, m, e), 0
        for ri in a.slots:
            heard_gap = max(heard_gap, float(jnp.max(jnp.abs(ra[c:c + ri * hd] - rb[c:c + ri * hd]))))
            pos_gap = max(pos_gap, float(jnp.max(jnp.abs(ra[c + ri * hd:c + ri * hd + ODP] - rb[c + ri * hd:c + ri * hd + ODP]))))
            c += ri * hd + ODP
    print(f"absorbed placement: sending {send_gap:.1e}, heard {heard_gap:.1e}, delivered positions {pos_gap:.3f}")
    assert send_gap < 1e-13 and heard_gap < 1e-13 and pos_gap > 0.1


# ---------------------------------------------------------------------------
# Prediction 8: the eye's Gaussian and the ear's, in one operad.
# ---------------------------------------------------------------------------


def scalar(g):
    return g * jnp.eye(2)


def test_eye_endpoint_and_ear_route():
    """Prediction 8: members at 0 and 3 on a line, tau = 1, group places 1 and 2. Forwarding groups with Gaussian
    hearing on sent places: weight e^{-4.5}. ``j`` of the spatial nest: e^{-1.5}."""
    one = lambda Z: None  # noqa: E731
    triv = configuration(trivial())
    # the eye: the parent hears by the sent places; each group forwards its one member
    def eye_parent(q, Z):
        T = {}
        for x in [(0, 0), (1, 0)]:
            for y in [(0, 0), (1, 0)]:
                if x != y:
                    T[(x, y)] = scalar(jnp.exp(-jnp.sum((Z[x][1] - Z[y][0]) ** 2) / 2))
        return OpData(place=[], T=T)
    parent = AddrOp(members=((1,), (1,)), shown=(), dim_P=1, P=triv, data=eye_parent, label="eye")
    eye = substitute(parent, [unit((1,), 1), unit((1,), 1)])
    Z = {(0, 0): (jnp.zeros(1), jnp.zeros(1)), (1, 0): (3 * jnp.ones(1), 3 * jnp.ones(1))}
    w_eye = float(eye.data(jnp.zeros(0), Z).T[((1, 0), (0, 0))][1, 1])
    # the ear: phi hears between group places 1 and 2; each psi hears its member at offset 1 from its place
    g = lambda d2: float(np.exp(-d2 / 2))  # noqa: E731
    phi = AttOp(dims=(1, 1, 1), P=triv, transfers=lambda q: {(1, 2): scalar(g(1.0)), (2, 1): scalar(g(1.0))})
    psi = AttOp(dims=(1, 1), P=triv, transfers=lambda q: {(0, 1): scalar(g(1.0)), (1, 0): scalar(g(1.0))})
    ear = substitute(j(phi, 1), [j(psi, 1), j(psi, 1)])
    w_ear = float(ear.data(jnp.zeros(0), Z).T[((1, 0), (0, 0))][1, 1])
    print(f"eye (forwarding, endpoint): {w_eye:.4f} = e^-4.5 = {np.exp(-4.5):.4f}; "
          f"ear (j, route): {w_ear:.4f} = e^-1.5 = {np.exp(-1.5):.4f}")
    assert abs(w_eye - np.exp(-4.5)) < 1e-12 and abs(w_ear - np.exp(-1.5)) < 1e-12


def test_eye_composes_with_any_kernel():
    """Addressing check (P79 (b), David's "check whether all our addressing stuff really was semantically right"):
    with forwarding groups, hearing by a kernel that is not an exponential, ``1 / (1 + |p - k|^2)``, the composite hears
    by the same kernel at the endpoints' query and address, and its unit laws hold; so composition alone does not force
    the exponential on content addressing. Two levels: three members in two forwarding groups."""
    triv = configuration(trivial())
    kern = lambda p, k: 1.0 / (1.0 + jnp.sum((p - k) ** 2))  # noqa: E731

    def parent(q, Z):
        xs = list(Z)
        return OpData(place=[], T={(x, y): scalar(kern(Z[x][1], Z[y][0])) for x in xs for y in xs if x[0] != y[0]})
    a = AddrOp(members=((1, 1), (1,)), shown=(), dim_P=1, P=triv, data=parent, label="eye")
    g1 = AddrOp(members=((1,), (1,)), shown=(1, 1), dim_P=1, P=triv, label="fwd2",
                data=lambda q, Z: OpData(place=[Z[(0, 0)], Z[(1, 0)]],
                                         U={(0, (0, 0)): jnp.eye(2), (1, (1, 0)): jnp.eye(2)},
                                         D={((0, 0), 0): jnp.eye(2), ((1, 0), 1): jnp.eye(2)}))
    b = substitute(a, [g1, unit((1,), 1)])
    Z = {(0, 0): (jnp.zeros(1), jnp.asarray([0.5])), (1, 0): (jnp.asarray([2.0]), jnp.asarray([1.0])),
         (2, 0): (jnp.asarray([3.0]), jnp.asarray([-1.0]))}
    T = b.data(jnp.zeros(0), Z).T
    w = float(T[((2, 0), (0, 0))][1, 1])            # member 2's query -1 against member 0's address 0
    print(f"eye, kernel 1/(1+d^2): composite {w:.6f}, endpoint kernel {1 / (1 + 1.0):.6f}")
    assert abs(w - 0.5) < 1e-14
    assert ((0, 0), (1, 0)) not in T or float(jnp.max(jnp.abs(T[((0, 0), (1, 0))]))) == 0.0   # same group: parent says nothing
    gaps, _ = arr_gap(b.arr(), compose_seq(tensor_arrangements([g1.arr(), unit((1,), 1).arr()]), a.arr()), seed=99)
    print("eye composite vs arr:", gaps)
    assert max(gaps.values()) < 1e-14


def test_at_a_point_the_addressed_operad_is_attention():
    """Chapter order (P79 (b)): with address space R^0 and one-slot colors, an addressed operation's data is an attention
    arrangement's (one shown slot, the external box), and substitution is eqn.attention_substitution: j at dim_P = 0 is
    the identification, its composite equal to the attention composite in every datum and as arrangements in arr."""
    phi = random_att((1, 2, 1, 1), 101, label="phi")
    psis = [random_att((2, 1, 1), 102), random_att((1,), 103), random_att((1, 2), 104)]
    here, there = substitute(j(phi, 0), [j(p, 0) for p in psis]), j(att_substitute(phi, psis), 0)
    q = jnp.asarray(np.random.default_rng(105).standard_normal(here.P.Q_dim) * 0.5)
    g = data_gap(here, there, q, q, seed=106)
    a_here = here.arr()
    a_att = att_substitute(phi, psis).arr()
    gaps, size = arr_gap(a_here, a_att, seed=107)
    print("address space a point: data gap", g, "addressed vs attention composite in arr:", gaps, size)
    assert g < 1e-12 and max(gaps.values()) < 1e-12
