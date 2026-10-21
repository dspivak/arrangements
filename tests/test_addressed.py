"""Addressed attention (``addressed.py``): the operad laws, the paper's spatial attention and softmax
attention as its cases, and the force that reaches a sent address.

Book plan P5 (g); the definition, the prediction written before this run, and the results are in the
book repository, ``notes/runs/p5g-f519e448/RESULTS.md``. Each docstring says which of three things its
test does: CHECKS A PROOF (the agents' proof in RESULTS.md, not yet read by David), COMPARES with the
paper's operad, or ILLUSTRATES.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.addressed import (
    AddressedArrangement, Shown, centroid_speaker, dot, forward, gaussian, ground, in_dim,
    kernel_hearing, out_dim, slot_dim, slot_output, speaker, split_slot, substitute, system, unit,
)
from arrangements.attention import (
    Transfer, apply_transfer, message, prediction_cost, spatial_arrangement, spatial_split,
    substitute as paper_substitute, zero_message,
)
from arrangements.coalgebra import configuration
from arrangements.functors import Phi, closed_step
from arrangements.rvect import diagonal, euclidean, trivial
from arrangements.wiring import compose_seq, tensor_arrangements

from tests.chain import flat_state

jax.config.update("jax_enable_x64", True)

DH, DP, TAU = 2, 2, 0.8


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


def random_speaker(slots, qdim, r):
    """A speaker whose place, and whose transfers up and down, depend smoothly on the configuration."""
    A, b = jnp.asarray(r.standard_normal((2, DP, qdim)) * 0.3), jnp.asarray(r.standard_normal((2, DP)))
    cu, cd = jnp.asarray(r.standard_normal((sum(slots), qdim)) * 0.2), jnp.asarray(r.standard_normal((sum(slots), qdim)) * 0.2)
    Mu, Md = jnp.asarray(r.standard_normal((sum(slots), DH, DH))), jnp.asarray(r.standard_normal((sum(slots), DH, DH)))
    flat = {x: t for t, x in enumerate((i, a) for i, ri in enumerate(slots) for a in range(ri))}

    def tr(c, M, x):
        def T(q, y):
            g = jnp.exp(jnp.dot(c[flat[y]], q))
            return Transfer(gamma=g, L=g * M[flat[y]])
        return T

    return speaker(key=lambda q: A[0] @ q + b[0], query=lambda q: A[1] @ q + b[1],
                   up=tr(cu, Mu, None), down=tr(cd, Md, None))


def random_op(slots, exposure, seed, mobility=0.05, kernel=None, label=None, speakers=0, extra=()):
    """An operation of ``Addr(slots; len(exposure) + speakers)``: configuration the placements and
    one log-gain per child; a Gaussian hearing with a configuration-dependent value map; a smooth
    cost on the configuration and the whole interaction (sent places included); ``speakers`` random
    speakers shown after the forwarded slots."""
    r = np.random.default_rng(seed)
    n = len(slots)
    qdim = n * DP + n
    P = configuration(euclidean(qdim, mobility), init=jnp.asarray(r.standard_normal(qdim) * 0.6))
    Ms = [jnp.asarray(r.standard_normal((DH, DH)) * 0.4) + jnp.eye(DH) for _ in range(n)]
    exposure = list(exposure) + [random_speaker(slots, qdim, r) for _ in range(speakers)] + list(extra)

    def placement(q):
        return jnp.reshape(q[:n * DP], (n, DP))

    def value_map(q, x, y):
        return jnp.exp(0.3 * q[n * DP + y[0]]) * Ms[y[0]]

    hearing = kernel_hearing(kernel or gaussian(TAU), DH, value_map)
    od = sum(out_dim(ri, DH, DP) for ri in slots)
    idim = in_dim(len(exposure), DH, DP)
    A = jnp.asarray(r.standard_normal((idim, od)) * 0.2)
    c = jnp.asarray(r.standard_normal(idim) * 0.2)

    def cost(q, m_out, n_in):
        d = n_in - A @ m_out - c - 0.1 * jnp.sum(jnp.sin(q))
        return 0.5 * jnp.sum(d * d)

    return AddressedArrangement(
        slots=tuple(slots), exposure=tuple(exposure), dim_H=DH, dim_P=DP, P=P,
        placement=placement, hearing=hearing, cost=cost, label=label or f"op{seed}",
    )


def random_point(a, seed):
    r = np.random.default_rng(seed)
    od = sum(out_dim(ri, DH, DP) for ri in a.slots)
    return (jnp.asarray(r.standard_normal(a.P.Q_dim) * 0.5), jnp.asarray(r.standard_normal(od)),
            jnp.asarray(r.standard_normal(in_dim(a.r, DH, DP))))


def permute_blocks(q, sizes, order):
    cut = [sum(sizes[:i]) for i in range(len(sizes) + 1)]
    return jnp.concatenate([q[cut[i]:cut[i + 1]] for i in order])


def random_places(slots, r, dP=DP):
    """Random places ``Z`` for every child slot of an operation with these ``slots``."""
    return {(i, a): (jnp.asarray(r.standard_normal(dP)), jnp.asarray(r.standard_normal(dP)))
            for i, ri in enumerate(slots) for a in range(ri)}


def assert_same_transfer(S, T, atol):
    np.testing.assert_allclose(S.gamma, T.gamma, atol=atol)
    np.testing.assert_allclose(S.L, T.L, atol=atol)


def assert_same_op(x, y, q_x, q_y, seed, atol=1e-11, draws=2):
    """Same slots; each shown slot the same place and the same transfers up and down to every
    slot; same placement, every hearing, all at random places ``Z``; same sending, receiving, cost."""
    assert x.slots == y.slots and x.r == y.r
    slots = x.child_slots()
    r = np.random.default_rng(seed)
    pairs = [(s, t) for s in slots for t in slots if s[0] != t[0]]
    for _ in range(draws):
        Z = random_places(x.slots, r)
        for ex, ey in zip(x.shown, y.shown):
            for a, b in zip(ex.place(q_x, Z), ey.place(q_y, Z)):
                np.testing.assert_allclose(a, b, atol=atol)
            for z in slots:
                assert_same_transfer(ex.up(q_x, z, Z), ey.up(q_y, z, Z), atol)
                assert_same_transfer(ex.down(q_x, z, Z), ey.down(q_y, z, Z), atol)
        for s, t in pairs:
            assert_same_transfer(x.hearing(q_x, s, t, Z), y.hearing(q_y, s, t, Z), atol)
    np.testing.assert_allclose(x.placement(q_x), y.placement(q_y), atol=atol)
    od = sum(out_dim(ri, DH, DP) for ri in x.slots)
    for _ in range(3):
        m = jnp.asarray(r.standard_normal(od))
        e = jnp.asarray(r.standard_normal(in_dim(x.r, DH, DP)))
        np.testing.assert_allclose(x.sigma(q_x, m), y.sigma(q_y, m), atol=atol)
        np.testing.assert_allclose(x.rho(q_x, m, e), y.rho(q_y, m, e), atol=atol)
        np.testing.assert_allclose(x.cost(q_x, m, e), y.cost(q_y, m, e), atol=atol)
    return len(pairs)


def trajectory(arr, steps, seed=0):
    """``Phi(arr)`` on fixed exterior data (as in test_attention_operad.py); flattened states."""
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


# ---------------------------------------------------------------------------
# Prediction 1: substitution is composition in arr.
# ---------------------------------------------------------------------------


def test_substitution_is_composition_in_arr():
    """CHECKS A PROOF (RESULTS.md §1, prediction 1): the explicit composite equals
    ``compose_seq(tensor(parts), parent)`` in its sending map, receiving map and all three
    components of its forcing, and the two run identically through ``Phi``. Hidden slots, a
    reordered exposure and an uneven nest are included."""
    a = random_op((2, 1), [(0, 1), (1, 0)], 1)                       # child 0's slot 0 hidden
    a1 = random_op((1, 2), [(1, 0), (0, 0)], 2)                      # color 2, order reversed
    a2 = random_op((1, 1), [(1, 0)], 3)                              # color 1
    b = substitute(a, [a1, a2])
    explicit = b.arr()
    composite = compose_seq(tensor_arrangements([a1.arr(), a2.arr()]), a.arr())
    assert (explicit.in_dim_M, explicit.out_dim_M, explicit.in_dim_N, explicit.out_dim_N) == \
        (composite.in_dim_M, composite.out_dim_M, composite.in_dim_N, composite.out_dim_N)
    for s in range(5):
        q, m, e = random_point(b, 100 + s)
        np.testing.assert_allclose(explicit.out_f(q, m), composite.out_f(q, m), atol=1e-12)
        np.testing.assert_allclose(explicit.in_f(q, m, e), composite.in_f(q, m, e), atol=1e-12)
        for x, y in zip(explicit.omega(q, m, e), composite.omega(q, m, e)):
            np.testing.assert_allclose(x, y, atol=1e-11)
    t1, t2 = trajectory(explicit, 25, seed=7), trajectory(composite, 25, seed=7)
    for x, y in zip(t1, t2):
        np.testing.assert_allclose(x, y, atol=1e-11)
    assert float(jnp.max(jnp.abs(t1[-1] - t1[0]))) > 1e-3


# ---------------------------------------------------------------------------
# Prediction 2: associativity and units.
# ---------------------------------------------------------------------------


def test_substitution_is_associative_on_a_three_level_nest():
    """CHECKS A PROOF (prediction 2): the two bracketings of a three-level nest agree as
    operations, up to the reordering of configuration factors (the coherence isomorphism)."""
    a = random_op((2, 1), [(0, 1), (1, 0)], 11)
    a1 = random_op((1, 2), [(1, 0), (0, 0)], 12)
    a2 = random_op((1, 1), [(1, 0)], 13)
    g = [random_op((1, 1), [(0, 0)], 14), random_op((2,), [(0, 1), (0, 0)], 15),
         random_op((1,), [(0, 0)], 16), random_op((2, 1), [(1, 0)], 17)]
    left = substitute(substitute(a, [a1, a2]), g)                    # Q: g0 g1 g2 g3 a1 a2 a
    right = substitute(a, [substitute(a1, g[:2]), substitute(a2, g[2:])])   # g0 g1 a1 g2 g3 a2 a
    sizes = [c.P.Q_dim for c in g] + [a1.P.Q_dim, a2.P.Q_dim, a.P.Q_dim]
    q_left = jnp.asarray(np.random.default_rng(18).standard_normal(left.P.Q_dim) * 0.5)
    q_right = permute_blocks(q_left, sizes, [0, 1, 4, 2, 3, 5, 6])
    npairs = assert_same_op(left, right, q_left, q_right, seed=19)
    assert npairs > 30
    # the forcings agree, the configuration covector reordered the same way
    r = np.random.default_rng(20)
    m = jnp.asarray(r.standard_normal(sum(out_dim(ri, DH, DP) for ri in left.slots)))
    e = jnp.asarray(r.standard_normal(in_dim(left.r, DH, DP)))
    xl, xr = left.arr().omega(q_left, m, e), right.arr().omega(q_right, m, e)
    np.testing.assert_allclose(permute_blocks(xl[0], sizes, [0, 1, 4, 2, 3, 5, 6]), xr[0], atol=1e-11)
    np.testing.assert_allclose(xl[1], xr[1], atol=1e-11)
    np.testing.assert_allclose(xl[2], xr[2], atol=1e-11)


def test_unit_laws():
    """CHECKS A PROOF (prediction 2): ``1_r(a) = a`` and ``a(1_{r_1}, ..., 1_{r_n}) = a``; the
    unit's configuration space is a point, so the configurations are the same."""
    a = random_op((2, 1), [(1, 0), (0, 0)], 21)
    q = a.P.init
    assert_same_op(substitute(unit(a.r, DH, DP), [a]), a, q, q, seed=22)
    assert_same_op(substitute(a, [unit(ri, DH, DP) for ri in a.slots]), a, q, q, seed=23)


# ---------------------------------------------------------------------------
# Predictions 3 and 4: the paper's spatial attention.
# ---------------------------------------------------------------------------


def spatial_addr(n, q_layout_n, tau=TAU, slots=None):
    """One level, every slot shown (one per child unless ``slots`` says), placement and value maps
    read off the paper's spatial configuration ``q = (p_1..p_n, V_1..V_n)``."""
    slots = (1,) * n if slots is None else tuple(slots)
    def placement(q):
        return spatial_split(q, n, DH, DP)[0]

    def value_map(q, x, y):
        return spatial_split(q, n, DH, DP)[1][y[0]]

    return AddressedArrangement(
        slots=slots, exposure=tuple((i, a) for i in range(n) for a in range(slots[i])), dim_H=DH,
        dim_P=DP, P=configuration(euclidean(q_layout_n, 1.0)), placement=placement,
        hearing=kernel_hearing(gaussian(tau), DH, value_map), label=f"spatial{n}",
    )


def at_origin(messages):
    """Children's outputs with address and query 0 (placement sets every place)."""
    return jnp.concatenate([slot_output(jnp.zeros(DP), jnp.zeros(DP), m[:DH], m[DH]) for m in messages])


def test_one_level_siblings_hear_as_in_the_paper_and_the_wall_differs():
    """COMPARES with the paper (prediction 3). With address and query 0, what each child hears from
    its siblings is the paper's spatial attention (eq:spatial-transfers) exactly. What crosses the
    group's wall is not: the paper's group listens at its center and attenuates by
    ``kappa(l_i, 0)`` (``T_{i0}``), and speaks once, ``sum_j kappa(0, l_j) V_j m_j`` (``T_{0j}``);
    here the exterior reaches each shown slot as it is, and the group shows each slot at its place."""
    n = 3
    paper = spatial_arrangement(n, DH, DP, TAU)
    addr = spatial_addr(n, paper.P.Q_dim)
    r = np.random.default_rng(31)
    q = jnp.asarray(r.standard_normal(paper.P.Q_dim) * 0.7)
    msgs = [message(r.standard_normal(DH), w) for w in (0.7, 1.3, 2.1)]
    p0 = jnp.asarray(r.standard_normal(DP))
    m_paper = jnp.concatenate(msgs)
    rho_paper = paper.rho(q, m_paper, jnp.concatenate([zero_message(DH), p0]))
    rho_addr = addr.rho(q, at_origin(msgs), jnp.concatenate([jnp.zeros(n * (DH + 1)), p0]))
    np.testing.assert_allclose(rho_addr, rho_paper, atol=1e-12)     # same layout: message, position

    m0 = message(r.standard_normal(DH), 0.9)
    rho_paper_ext = paper.rho(q, m_paper, jnp.concatenate([m0, p0]))
    rho_addr_ext = addr.rho(q, at_origin(msgs), jnp.concatenate([jnp.tile(m0, n), p0]))
    ell = spatial_split(q, n, DH, DP)[0]
    for i in range(n):
        blk = slice(i * (DH + 1 + DP), i * (DH + 1 + DP) + DH + 1)
        att = float(jnp.exp(-jnp.sum(ell[i] ** 2) / (2 * TAU)))
        np.testing.assert_allclose(rho_paper_ext[blk] - rho_paper[blk], att * m0, atol=1e-12)
        np.testing.assert_allclose(rho_addr_ext[blk] - rho_addr[blk], m0, atol=1e-12)
    assert paper.sigma(q, m_paper).shape == (DH + 1,)                 # one message, at the center
    assert addr.sigma(q, at_origin(msgs)).shape == (n * slot_dim(DH, DP),)   # each slot, at its place


def test_nested_spatial_hears_by_one_gaussian_the_paper_by_a_product():
    """COMPARES with the paper (prediction 4). Two groups of two inside a parent, value maps the
    identity. Between groups, the composite weight here is one Gaussian of the exposed places; the
    paper's is the product of three (eq:composite-transfers). So the Gaussian family is closed under
    nesting here (the composite equals the one-level spatial operation on the exposed places), and
    in the paper it is not ("it is not itself closed under substitution")."""
    qd = lambda n: n * DP + n * DH * DH
    eye = lambda n, pos: jnp.concatenate([jnp.ravel(pos), jnp.tile(jnp.ravel(jnp.eye(DH)), n)])
    pa, pa1, pa2 = jnp.array([[0.0, 0.0], [1.5, 0.3]]), jnp.array([[-0.4, 0.2], [0.5, -0.6]]), \
        jnp.array([[0.3, 0.8], [-0.7, 0.1]])
    paper_b = paper_substitute(spatial_arrangement(2, DH, DP, TAU),
                               [spatial_arrangement(2, DH, DP, TAU), spatial_arrangement(2, DH, DP, TAU)])
    q_b = jnp.concatenate([eye(2, pa1), eye(2, pa2), eye(2, pa)])
    addr_b = substitute(spatial_addr(2, qd(2), slots=(2, 2)), [spatial_addr(2, qd(2)), spatial_addr(2, qd(2))])
    ell_b = addr_b.placement(q_b)
    np.testing.assert_allclose(ell_b, paper_b.placement(q_b), atol=1e-12)
    g = lambda d: float(jnp.exp(-jnp.sum(d ** 2) / (2 * TAU)))

    # receiver (0,0) = composite child 0, sender (1,0) = composite child 2
    Zb = {(t, 0): (ell_b[t], ell_b[t]) for t in range(4)}            # every place set by placement
    w_addr = float(addr_b.hearing(q_b, (0, 0), (2, 0), Zb).gamma)
    w_paper = float(paper_b.transfers(q_b)[1][3].gamma)
    np.testing.assert_allclose(w_addr, g(ell_b[0] - ell_b[2]), atol=1e-12)
    np.testing.assert_allclose(w_paper, g(pa1[0]) * g(pa[0] - pa[1]) * g(pa2[0]), atol=1e-12)
    assert abs(w_addr - w_paper) > 1e-2
    print(f"\nbetween groups: one Gaussian {w_addr:.4f}, the paper's product {w_paper:.4f}")

    flat = spatial_addr(4, qd(4))
    q_flat = eye(4, ell_b)
    for x in range(4):
        for y in range(4):
            if x != y:
                Tb = addr_b.hearing(q_b, (x, 0), (y, 0), Zb)
                Tf = flat.hearing(q_flat, (x, 0), (y, 0), Zb)
                np.testing.assert_allclose(Tb.gamma, Tf.gamma, atol=1e-12)
                np.testing.assert_allclose(Tb.L, Tf.L, atol=1e-12)


# ---------------------------------------------------------------------------
# Prediction 5: softmax attention.
# ---------------------------------------------------------------------------


def test_softmax_attention_is_the_case_where_parts_send_their_own_places():
    """COMPARES with transformer attention (prediction 5). Four tokens with no placement each send
    a key, a query and a value of weight 1; the parent hears by ``exp <p, k>``. A token hears every
    other token; adding its own term (the zero diagonal makes hearing oneself the token's own
    business) and taking the mean readout gives ``sum_y softmax_y(<p_x, k_y>) v_y``. The reference
    is a plain formula beside the arrangement, checked equal."""
    dH, dP, n = 3, 2, 4
    r = np.random.default_rng(41)
    K, Qs, Vv = (jnp.asarray(r.standard_normal((n, d))) for d in (dP, dP, dH))
    parent = AddressedArrangement(
        slots=(1,) * n, exposure=(), dim_H=dH, dim_P=dP, P=configuration(trivial()),
        placement=lambda q: jnp.zeros((n, dP)), hearing=kernel_hearing(dot(), dH), label="softmax",
    )
    m_out = jnp.concatenate([slot_output(K[t], Qs[t], Vv[t], 1.0) for t in range(n)])
    rho = parent.rho(jnp.zeros(0), m_out, jnp.zeros(dP))
    blk = dH + 1 + dP
    scores = Qs @ K.T
    ref = jax.nn.softmax(scores, axis=1) @ Vv
    for x in range(n):
        heard = rho[x * blk:x * blk + dH + 1] + jnp.exp(scores[x, x]) * jnp.concatenate([Vv[x], jnp.ones(1)])
        np.testing.assert_allclose(heard[:dH] / heard[dH], ref[x], atol=1e-12)


# ---------------------------------------------------------------------------
# Prediction 6: the force that reaches a sent address.
# ---------------------------------------------------------------------------


def reader_and_sources(tau=1.0, keys=(-1.0, 2.0), values=(0.0, 1.0), target=(1.0, 1.0), home=0.0,
                       reader_mobility=0.0, place_mobility=0.0, key_mobility=0.0):
    """One parent, three one-slot children on ``H = P = R``: a reader whose configuration is its
    query offset, two sources whose configurations are their address offsets. The reader's cost
    predicts what it hears, ``(1/2)|heard - target|^2`` (the paper's private prediction on the
    weighted input), plus ``(home/2)|position|^2`` on its position input. The parent's configuration
    is the three placements, of which only the reader's is mobile (``place_mobility``). Returns the
    closed system and the parent."""
    dH = dP = 1
    t = jnp.asarray(target)

    def reader_cost(q, e):
        heard, pos = e[:2], e[2:]
        return 0.5 * jnp.sum((heard - t) ** 2) + 0.5 * home * jnp.sum(pos ** 2)

    reader = system(1, dH, dP, configuration(euclidean(1, reader_mobility), init=jnp.zeros(1)),
                    emit=lambda q: slot_output(jnp.zeros(1), q, jnp.zeros(1), 0.0),
                    cost=reader_cost, label="reader")
    sources = [system(1, dH, dP, configuration(euclidean(1, key_mobility), init=jnp.asarray([k])),
                      emit=(lambda v: (lambda q: slot_output(q, jnp.zeros(1), jnp.asarray([v]), 1.0)))(v),
                      label=f"source{s}")
               for s, (k, v) in enumerate(zip(keys, values))]
    parent = AddressedArrangement(
        slots=(1, 1, 1), exposure=(), dim_H=dH, dim_P=dP,
        P=configuration(diagonal(jnp.asarray([place_mobility, 0.0, 0.0])), init=jnp.zeros(3)),
        placement=lambda q: jnp.reshape(q, (3, 1)), hearing=kernel_hearing(gaussian(tau), dH),
        label="parent",
    )
    closed = compose_seq(compose_seq(tensor_arrangements([reader] + sources), parent.arr()), ground(dP))
    return closed, parent


def test_force_on_a_sent_address():
    """CHECKS A PROOF (prediction 6). The force the composite returns to a source's address is
    ``<xi, (h_y, w_y)> d/dk kappa(p_x, k_y)``, ``xi = -(heard - target)`` the reader's covector on what
    it hears; and the force on the parent's placement of each child is that child's covector on its
    position input plus the forces on its slots' address and query offsets (``l_j + k``). Read off
    the closed composite's forcing, the configuration order being reader, source 0, source 1,
    parent placements."""
    tau = 1.0
    closed, parent = reader_and_sources(tau=tau, home=0.5)
    z = jnp.zeros(0)
    q = jnp.asarray([0.3, -1.0, 2.0, 0.1, 0.4, -0.2])     # reader query, two keys, three placements
    xi_q, _, _ = closed.omega(q, z, z)
    p_hat = q[0] + q[3]
    k_hat = [q[1] + q[4], q[2] + q[5]]
    kap = [float(jnp.exp(-(p_hat - k) ** 2 / (2 * tau))) for k in k_hat]
    values = (0.0, 1.0)
    heard = jnp.asarray([kap[0] * values[0] + kap[1] * values[1], kap[0] + kap[1]])
    xi = -(heard - jnp.asarray([1.0, 1.0]))
    for s in range(2):
        dk = kap[s] * (p_hat - k_hat[s]) / tau                # d/dk of the Gaussian at the address
        force = float(jnp.dot(xi, jnp.asarray([values[s], 1.0]))) * dk
        np.testing.assert_allclose(xi_q[1 + s], force, atol=1e-12)
        np.testing.assert_allclose(xi_q[4 + s], force, atol=1e-12)   # the placement gets the same
    # the reader: placement force = its position covector (-home * position) + the force on its query
    np.testing.assert_allclose(xi_q[3], -0.5 * q[3] + xi_q[0], atol=1e-12)


# ---------------------------------------------------------------------------
# Prediction 7: the jump and the slow move, run.
# ---------------------------------------------------------------------------


def run(closed, steps):
    step = closed_step(closed)
    s = closed.P.init
    for _ in range(steps):
        s = step(s)
    return flat_state(s)


def test_the_jump_listens_far_from_home_and_the_slow_move_compromises():
    """ILLUSTRATES (prediction 7), run through ``Phi`` (``closed_step``, checked against ``Phi`` on
    its first steps). The reader wants to hear the second source's value (target ``(1, 1)``) and
    pays ``(1/2)|position|^2`` for being away from home. Jump: the reader's query offset is its
    matter, the placements are fixed; the query settles just short of the second source's address
    and the body stays home. Slow move: the query offset is fixed at 0 and the parent's placement of the reader
    is mobile; the body is pulled toward the source and held back by home, so it settles between,
    hearing less well. The first step is the same in both (the query and the placement enter at
    the same place); what differs is whose matter moves and what moves with it."""
    eta = 0.5
    jump, _ = reader_and_sources(home=1.0, reader_mobility=eta)
    slow, _ = reader_and_sources(home=1.0, place_mobility=eta)
    s_jump = run(jump, 4000)
    s_slow = run(slow, 4000)
    p_400 = float(run(jump, 400)[0])
    p_jump, pos_jump = float(s_jump[0]), 0.0
    pos_slow = float(s_slow[3])

    def heard_mean(listen):
        kap = [np.exp(-(listen - k) ** 2 / 2) for k in (-1.0, 2.0)]
        return kap[1] / (kap[0] + kap[1]), kap[0] + kap[1]

    m_jump, w_jump = heard_mean(p_jump + pos_jump)
    m_slow, w_slow = heard_mean(pos_slow)
    print(f"\njump: query {p_400:.4f} after 400 steps, {p_jump:.4f} after 4000")
    print(f"jump: query {p_jump:.4f}, home position {pos_jump}, hears mean {m_jump:.4f} weight {w_jump:.4f}"
          f"\nslow: position {pos_slow:.4f}, hears mean {m_slow:.4f} weight {w_slow:.4f}"
          f"\nslow, the sources' placements: {float(s_slow[4]):.4f}, {float(s_slow[5]):.4f}")
    # it settles just short of the second source's address (1.894 in the run of 2026-10-19), where the
    # first source's small weight and the second's sum to the weight 1 the reader predicts
    assert 1.8 < p_400 < p_jump < 2.0
    assert abs(p_jump - float(run(jump, 3000)[0])) < 1e-6
    assert 0.0 < pos_slow < p_jump - 0.3
    assert m_jump > m_slow
    # first steps agree
    one_jump, one_slow = run(jump, 1), run(slow, 1)
    np.testing.assert_allclose(one_jump[0], one_slow[3], atol=1e-12)


# ---------------------------------------------------------------------------
# Speakers: a group answering for its members from a place it sets (RESULTS.md §4).
# ---------------------------------------------------------------------------


def test_speakers_substitution_is_composition_and_associative():
    """CHECKS A PROOF (RESULTS.md §4): with speakers mixed among forwarded slots, at both levels, the
    explicit composite is the composite in ``arr`` (sending, receiving, forcing, a ``Phi`` run), and
    the two bracketings of a three-level nest agree."""
    a = random_op((2, 2), [(0, 1)], 51, speakers=1)                  # shows a member and a speaker
    a1 = random_op((1, 2), [(1, 0)], 52, speakers=1)                 # color 2: member, speaker
    a2 = random_op((1, 1), [], 53, speakers=2)                       # color 2: two speakers
    b = substitute(a, [a1, a2])
    explicit = b.arr()
    composite = compose_seq(tensor_arrangements([a1.arr(), a2.arr()]), a.arr())
    for t in range(4):
        q, m, e = random_point(b, 200 + t)
        np.testing.assert_allclose(explicit.out_f(q, m), composite.out_f(q, m), atol=1e-11)
        np.testing.assert_allclose(explicit.in_f(q, m, e), composite.in_f(q, m, e), atol=1e-11)
        for x, y in zip(explicit.omega(q, m, e), composite.omega(q, m, e)):
            np.testing.assert_allclose(x, y, atol=1e-10)
    t1, t2 = trajectory(explicit, 15, seed=8), trajectory(composite, 15, seed=8)
    for x, y in zip(t1, t2):
        np.testing.assert_allclose(x, y, atol=1e-10)

    g = [random_op((1, 1), [(0, 0)], 54), random_op((2,), [(0, 1)], 55, speakers=1),
         random_op((1,), [], 56, speakers=1), random_op((2, 1), [(1, 0)], 57)]
    left = substitute(b, g)
    right = substitute(a, [substitute(a1, g[:2]), substitute(a2, g[2:])])
    sizes = [c.P.Q_dim for c in g] + [a1.P.Q_dim, a2.P.Q_dim, a.P.Q_dim]
    q_left = jnp.asarray(np.random.default_rng(58).standard_normal(left.P.Q_dim) * 0.5)
    q_right = permute_blocks(q_left, sizes, [0, 1, 4, 2, 3, 5, 6])
    assert_same_op(left, right, q_left, q_right, seed=59, atol=1e-10)


def from_paper(att):
    """The paper's attention arrangement as an operation here: one slot per child (its places
    ignored), one speaker at the group's center sending ``sum_j T_{0j} m_j`` and handing down
    ``T_{i0}``; the hearing ``T_{ij}(q)``, blind to places."""
    sd = slot_dim(att.dim_H, att.dim_P)
    n = att.n

    def cost(q, m_out, n_in):
        msgs = jnp.concatenate([split_slot(m_out[t * sd:(t + 1) * sd], att.dim_H, att.dim_P)[2]
                                for t in range(n)]) if n else jnp.zeros(0)
        return att.cost(q, msgs, n_in)

    sp = speaker(key=lambda q: jnp.zeros(att.dim_P), query=lambda q: jnp.zeros(att.dim_P),
                 up=lambda q, y: att.transfers(q)[0][y[0] + 1],
                 down=lambda q, x: att.transfers(q)[x[0] + 1][0])
    return AddressedArrangement(
        slots=(1,) * n, exposure=(sp,), dim_H=att.dim_H, dim_P=att.dim_P, P=att.P,
        placement=att.placement, hearing=lambda q, x, y, Z: att.transfers(q)[x[0] + 1][y[0] + 1],
        cost=cost, label=f"paper:{att.label}",
    )


def paper_part(n, seed):
    r = np.random.default_rng(seed)
    od, ed = n * (DH + 1), DH + 1 + DP
    A, c = jnp.asarray(r.standard_normal((ed, od)) * 0.3), jnp.asarray(r.standard_normal(ed) * 0.3)
    init = jnp.asarray(r.standard_normal(n * DP + n * DH * DH) * 0.6)
    return spatial_arrangement(n, DH, DP, TAU, cost=prediction_cost(lambda q, m: A @ m + c + 0.2 * jnp.sum(q)),
                               init=init, mobility=0.05, label=f"p{seed}")


def test_the_papers_operad_is_inside_as_groups_that_speak_from_their_center():
    """CHECKS SUBSTITUTION ONLY (RESULTS.md §4): ``from_paper`` is nonunital; its image of the paper unit resets places (P53(j), b2b2dfd4). The composite here of
    the paper's arrangements equals the image of the paper's own composite (eq:composite-transfers):
    the hearing between every two children is the paper's ``T_{(i,j),(r,s)}``, the speaker's up and
    down are its ``T_{0,(i,j)}`` and ``T_{(i,j),0}``, and receiving and cost agree. Nested twice, uneven."""
    a, a1, a2 = paper_part(2, 61), paper_part(2, 62), paper_part(1, 63)
    g = [paper_part(1, 64), paper_part(2, 65), paper_part(0, 66)]
    for here, there in [
        (substitute(from_paper(a), [from_paper(a1), from_paper(a2)]), from_paper(paper_substitute(a, [a1, a2]))),
        (substitute(substitute(from_paper(a), [from_paper(a1), from_paper(a2)]), [from_paper(c) for c in g]),
         from_paper(paper_substitute(paper_substitute(a, [a1, a2]), g))),
    ]:
        q = jnp.asarray(np.random.default_rng(67).standard_normal(here.P.Q_dim) * 0.5)
        assert_same_op(here, there, q, q, seed=68)
    # and as arrangements: the receiving map is the paper's, read on the paper's coordinates
    b_paper = paper_substitute(a, [a1, a2])
    b_here = substitute(from_paper(a), [from_paper(a1), from_paper(a2)])
    r = np.random.default_rng(69)
    q = jnp.asarray(r.standard_normal(b_here.P.Q_dim) * 0.5)
    msgs = [jnp.asarray(r.standard_normal(DH + 1)) for _ in range(b_here.n)]
    places = [jnp.asarray(r.standard_normal(2 * DP)) for _ in range(b_here.n)]
    m_here = jnp.concatenate([jnp.concatenate([pl, m]) for pl, m in zip(places, msgs)])
    e = jnp.asarray(r.standard_normal(DH + 1 + DP))
    np.testing.assert_allclose(b_here.rho(q, m_here, e), b_paper.rho(q, jnp.concatenate(msgs), e), atol=1e-12)
    np.testing.assert_allclose(b_here.sigma(q, m_here)[2 * DP:], b_paper.sigma(q, jnp.concatenate(msgs)), atol=1e-12)
    np.testing.assert_allclose(b_here.cost(q, m_here, e), b_paper.cost(q, jnp.concatenate(msgs), e), atol=1e-12)


# ---------------------------------------------------------------------------
# Combinations: forwarding and speaking are the two ends of one rule (RESULTS.md §9).
# ---------------------------------------------------------------------------


def test_forwarding_and_speaking_are_two_cases_of_one_rule():
    """CHECKS A PROOF (RESULTS.md §9). A group that speaks for two members from their mean place is
    neither forwarding nor a speaker at a place the group sets; with it at both levels, mixed with
    forwarded slots and speakers, substitution is composition in ``arr`` and both bracketings of a
    three-level nest agree. Forwarding is the centroid of one member; a speaker is a place that
    ignores the members. Between groups, a centroid makes two members' hearing depend on a third
    member's place, which forwarding and speaking never do."""
    a = random_op((2, 2), [(0, 1)], 71, speakers=1)
    a1 = random_op((1, 2), [], 72, extra=[centroid_speaker([(0, 0), (1, 1)], DH), (1, 0)])
    a2 = random_op((1, 1), [], 73, speakers=1,
                   extra=[centroid_speaker([(0, 0), (1, 0)], DH, offset=lambda q: 0.5 * q[:DP])])
    b = substitute(a, [a1, a2])
    explicit = b.arr()
    composite = compose_seq(tensor_arrangements([a1.arr(), a2.arr()]), a.arr())
    for t in range(4):
        q, m, e = random_point(b, 300 + t)
        np.testing.assert_allclose(explicit.out_f(q, m), composite.out_f(q, m), atol=1e-11)
        np.testing.assert_allclose(explicit.in_f(q, m, e), composite.in_f(q, m, e), atol=1e-11)
        for x, y in zip(explicit.omega(q, m, e), composite.omega(q, m, e)):
            np.testing.assert_allclose(x, y, atol=1e-10)

    g = [random_op((1, 1), [(0, 0)], 74),
         random_op((2,), [], 75, speakers=1, extra=[centroid_speaker([(0, 0), (0, 1)], DH)]),
         random_op((1,), [], 76, speakers=1), random_op((2, 1), [(1, 0)], 77)]
    left = substitute(b, g)
    right = substitute(a, [substitute(a1, g[:2]), substitute(a2, g[2:])])
    sizes = [c.P.Q_dim for c in g] + [a1.P.Q_dim, a2.P.Q_dim, a.P.Q_dim]
    q_left = jnp.asarray(np.random.default_rng(78).standard_normal(left.P.Q_dim) * 0.5)
    q_right = permute_blocks(q_left, sizes, [0, 1, 4, 2, 3, 5, 6])
    assert_same_op(left, right, q_left, q_right, seed=79, atol=1e-10, draws=1)

    # the two ends: a centroid of one member is forwarding; a speaker's place ignores the members
    r = np.random.default_rng(80)
    Z = random_places((2, 1), r)
    q0 = jnp.asarray(r.standard_normal(4))
    c1, f1 = centroid_speaker([(0, 1)], DH), forward((0, 1), DH)
    for u, v in zip(c1.place(q0, Z), f1.place(q0, Z)):
        np.testing.assert_allclose(u, v, atol=1e-14)
    for z in [(0, 0), (0, 1), (1, 0)]:
        assert_same_transfer(c1.up(q0, z, Z), f1.up(q0, z, Z), 1e-14)

    # a centroid makes b's hearing between members (0,0) of a1 and (0,0) of a2's first child depend on
    # where a1's other member (1,1) sits, through a1's mean place
    q = random_point(b, 81)[0]
    Zb = random_places(b.slots, np.random.default_rng(82))
    x, y, third = (0, 0), (3, 0), (1, 1)          # b's children: a1's 0, 1, then a2's 0, 1
    T0 = b.hearing(q, x, y, Zb)
    Zm = dict(Zb)
    Zm[third] = (Zb[third][0] + 0.7, Zb[third][1] + 0.7)
    T1 = b.hearing(q, x, y, Zm)
    assert abs(float(T0.gamma) - float(T1.gamma)) > 1e-4
