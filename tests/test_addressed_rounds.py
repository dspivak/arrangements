"""Addressed attention over several rounds in one pass (``addressed_rounds.py``): substitution is composition, a
two-layer transformer as one operation, why a strategy is a function of the whole box's hearing, and the comparison
lenses to one round and to the paper's operad.

Book plan P53 (g); the definitions, the prediction written before this run, and the results are in the book
repository, ``notes/runs/p53g-53d025a9/RESULTS.md``. Each docstring says which of three things its test does: CHECKS A
PROOF (the agents' proof in RESULTS.md, not yet read by David), COMPARES with a plain implementation beside it, or
ILLUSTRATES.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.addressed import AddressedArrangement, slot_dim, slot_output, substitute
from arrangements.addressed_rounds import (
    Color, ground, heard_dim, in_dim, iota, leaf, moore, out_dim, rounds, strategy_sends,
)
from arrangements.arrangement import Arrangement, exact, zero_form
from arrangements.attention import Transfer, zero_transfer
from arrangements.coalgebra import configuration
from arrangements.rvect import euclidean, trivial
from arrangements.wiring import compose_seq, tensor_arrangements

from tests.test_addressed import DH, DP, from_paper, paper_part, random_op

jax.config.update("jax_enable_x64", True)

SD, HD = slot_dim(DH, DP), DH + 1


# ---------------------------------------------------------------------------
# Fixtures: boxes whose later sends depend smoothly on what they heard.
# ---------------------------------------------------------------------------


def mealy(r, seed, theta_dim=3, L_max=3):
    """A color of ``r`` slots whose round-``l`` sends are a smooth function of its parameter and of everything its
    slots heard before: ``s1 / 2 + tanh(B_l [theta, H] + c_l)``."""
    g = np.random.default_rng(seed)
    B = {l: jnp.asarray(g.standard_normal((r * SD, theta_dim + (l - 1) * r * HD)) * 0.4) for l in range(2, L_max + 1)}
    c = {l: jnp.asarray(g.standard_normal(r * SD) * 0.3) for l in range(2, L_max + 1)}

    def strategy(s1, theta, l, H):
        return 0.5 * s1 + jnp.tanh(B[l] @ jnp.concatenate([theta, jnp.ravel(H)]) + c[l])

    return Color(r, theta_dim, strategy, label=f"mealy{r}:{seed}")


def random_outputs(colors, seed):
    g = np.random.default_rng(seed)
    return jnp.asarray(g.standard_normal(sum(out_dim(c, DH, DP) for c in colors)) * 0.5)


def random_leaf(c, L, seed, qdim=3):
    """A system ``I -> B^L_c``: sends and parameter affine in its configuration; a smooth cost on all it hears."""
    g = np.random.default_rng(seed)
    od = out_dim(c, DH, DP)
    A, b = jnp.asarray(g.standard_normal((od, qdim)) * 0.5), jnp.asarray(g.standard_normal(od) * 0.3)
    idim = in_dim(c, L, DH, DP)
    E, f = jnp.asarray(g.standard_normal((idim, idim)) * 0.3), jnp.asarray(g.standard_normal(idim) * 0.3)
    P = configuration(euclidean(qdim, 0.05), init=jnp.asarray(g.standard_normal(qdim) * 0.5))
    return leaf(c, L, DH, DP, P, emit=lambda q: A @ q + b,
                cost=lambda q, e: 0.5 * jnp.sum((E @ e - f) ** 2) + 0.1 * jnp.sum(jnp.sin(q)), label=f"leaf{seed}")


def assert_same_arrangement_up_to_strategy(x, cx, y, cy, L, seed, atol=1e-11):
    """Same round-1 sends; the same strategy at random earlier hearing (the parameters may differ); same receiving
    map and effort field, at random points."""
    g = np.random.default_rng(seed)
    r = cx.r
    assert r == cy.r and x.in_dim_M == y.in_dim_M and x.out_dim_M == y.out_dim_M and x.in_dim_N == y.in_dim_N
    assert x.P.Q_dim == y.P.Q_dim
    for _ in range(3):
        q = jnp.asarray(g.standard_normal(x.P.Q_dim) * 0.5)
        m = jnp.asarray(g.standard_normal(x.out_dim_M) * 0.5)
        e = jnp.asarray(g.standard_normal(x.in_dim_N) * 0.5)
        ox, oy = x.out_f(q, m), y.out_f(q, m)
        np.testing.assert_allclose(ox[:r * SD], oy[:r * SD], atol=atol)
        for l in range(2, L + 1):
            H = jnp.asarray(g.standard_normal((l - 1, heard_dim(r, DH))) * 0.5)
            np.testing.assert_allclose(strategy_sends(cx, ox, l, H, DH, DP), strategy_sends(cy, oy, l, H, DH, DP),
                                       atol=atol)
        np.testing.assert_allclose(x.in_f(q, m, e), y.in_f(q, m, e), atol=atol)
        for u, v in zip(x.omega(q, m, e), y.omega(q, m, e)):
            np.testing.assert_allclose(u, v, atol=atol * 10)


# ---------------------------------------------------------------------------
# Prediction 1: Phi_L preserves substitution.
# ---------------------------------------------------------------------------


def test_substitution_is_composition_over_rounds():
    """CHECKS A PROOF (RESULTS.md §2, prediction 1): for P5 (g)'s explicit composite ``b = a(a1, a2)``,
    ``Phi_L(b) = Phi_L(a) . (Phi_L(a1) (x) Phi_L(a2))`` at two and three rounds, with a speaker, a hidden slot and
    uneven arities; and closed by a group that reads the shown strategy, the forces on every configuration agree,
    though the two represent that strategy by different parameters."""
    a = random_op((2, 1), [(0, 1), (1, 0)], 1, speakers=1)          # child 0's slot 0 hidden; a speaker
    a1 = random_op((1, 2), [(1, 0), (0, 0)], 2)
    a2 = random_op((1, 1), [(1, 0)], 3)
    b = substitute(a, [a1, a2])
    grand = [mealy(1, 10), mealy(2, 11), mealy(1, 12), mealy(1, 13)]
    for L in (2, 3):
        explicit, cb = rounds(b, grand, L)
        p1, c1 = rounds(a1, grand[:2], L)
        p2, c2 = rounds(a2, grand[2:], L)
        pa, ca = rounds(a, [c1, c2], L)
        composite = compose_seq(tensor_arrangements([p1, p2]), pa)
        assert_same_arrangement_up_to_strategy(explicit, cb, composite, ca, L, seed=20 + L)

        top = random_op((a.r,), [], 5)
        tb, ctb = rounds(top, [cb], L)
        tc, ctc = rounds(top, [ca], L)
        leaves = tensor_arrangements([random_leaf(c, L, 30 + i) for i, c in enumerate(grand)])
        closed_b = compose_seq(compose_seq(compose_seq(leaves, explicit), tb), ground(ctb, L, DH, DP))
        closed_c = compose_seq(compose_seq(compose_seq(leaves, composite), tc), ground(ctc, L, DH, DP))
        z = jnp.zeros(0)
        q = jnp.asarray(np.random.default_rng(40 + L).standard_normal(closed_b.P.Q_dim) * 0.5)
        fb, fc = closed_b.omega(q, z, z)[0], closed_c.omega(q, z, z)[0]
        np.testing.assert_allclose(fb, fc, atol=1e-10)
        assert float(jnp.max(jnp.abs(fb))) > 1e-2


def test_units_over_rounds():
    """CHECKS A PROOF (prediction 1, units). ``Phi_L`` of the unit of color ``r`` is the identity on ``B^L_c``: the
    same sends at every round, what it hears passed through, the position ``p_0 + 0``, no force."""
    from arrangements.addressed import unit
    for L, c in [(2, mealy(2, 91)), (3, mealy(1, 92))]:
        arr, cu = rounds(unit(c.r, DH, DP), [c], L)
        g = np.random.default_rng(93 + L)
        q = jnp.zeros(0)
        m = random_outputs([c], 94 + L)
        e = jnp.asarray(g.standard_normal(arr.in_dim_N))
        out = arr.out_f(q, m)
        np.testing.assert_allclose(out[:c.r * SD], m[:c.r * SD], atol=1e-14)
        for l in range(2, L + 1):
            H = jnp.asarray(g.standard_normal((l - 1, heard_dim(c.r, DH))))
            np.testing.assert_allclose(strategy_sends(cu, out, l, H, DH, DP), strategy_sends(c, m, l, H, DH, DP),
                                       atol=1e-13)
        np.testing.assert_allclose(arr.in_f(q, m, e), e, atol=1e-14)
        assert all(float(jnp.max(jnp.abs(x))) == 0.0 for x in arr.omega(q, m, e) if x.size)


# ---------------------------------------------------------------------------
# Prediction 2: a two-layer transformer.
# ---------------------------------------------------------------------------


N_TOK, D, HEADS, DK, DV, D_MLP, D_OUT = 4, 4, 2, 2, 2, 3, 2


def unpack(W):
    """The weights, flat: per layer and head W^Q, W^K, W^V, W^O; the MLP of layer 1; the readout U."""
    shapes = []
    for l in range(2):
        for a in range(HEADS):
            shapes += [(f"Q{l}{a}", (DK, D)), (f"K{l}{a}", (DK, D)), (f"V{l}{a}", (DV, D)), (f"O{l}{a}", (D, DV))]
    shapes += [("W1", (D_MLP, D)), ("b1", (D_MLP,)), ("W2", (D, D_MLP)), ("U", (D_OUT, D))]
    out, c = {}, 0
    for name, sh in shapes:
        k = int(np.prod(sh))
        out[name] = jnp.reshape(W[c:c + k], sh)
        c += k
    return out, c


W_DIM = unpack(jnp.zeros(10_000))[1]


def kqv(w, l, a, x):
    return w[f"K{l}{a}"] @ x, w[f"Q{l}{a}"] @ x / jnp.sqrt(DK), w[f"V{l}{a}"] @ x


def layer_one(w, x0, means):
    """``x^1`` from ``x^0`` and the two heads' layer-1 means: residual, then a one-layer MLP."""
    xh = x0 + sum(w[f"O0{a}"] @ means[a] for a in range(HEADS))
    return xh + w["W2"] @ jnp.tanh(w["W1"] @ xh + w["b1"])


def transformer(W, X0, freeze_keys=False):
    """The plain implementation: two layers of causal multi-head softmax attention (Vaswani et al. §§3.2.1–3.2.3,
    the scale folded into the query), a residual and an MLP after the first. Returns ``x^2`` per token and the two
    layers' per-head outputs. ``freeze_keys`` stops the gradient through ``x^1`` into layer 2's keys and queries."""
    w, _ = unpack(W)
    mask = jnp.tril(jnp.ones((N_TOK, N_TOK)))

    def attend(l, X, Xkq):
        outs = []
        for a in range(HEADS):
            K = jnp.stack([kqv(w, l, a, x)[0] for x in Xkq])
            Q = jnp.stack([kqv(w, l, a, x)[1] for x in Xkq])
            V = jnp.stack([kqv(w, l, a, x)[2] for x in X])
            S = jnp.where(mask > 0, Q @ K.T, -jnp.inf)
            outs.append(jax.nn.softmax(S, axis=1) @ V)
        return outs

    o1 = attend(0, X0, X0)
    X1 = jnp.stack([layer_one(w, X0[t], [o1[a][t] for a in range(HEADS)]) for t in range(N_TOK)])
    o2 = attend(1, X1, jax.lax.stop_gradient(X1) if freeze_keys else X1)
    X2 = jnp.stack([X1[t] + sum(w[f"O1{a}"] @ o2[a][t] for a in range(HEADS)) for t in range(N_TOK)])
    return X2, o1, o2


def transformer_loss(W, X0, Y, freeze_keys=False):
    w, _ = unpack(W)
    X2, _, _ = transformer(W, X0, freeze_keys)
    return 0.5 * jnp.sum((X2 @ w["U"].T - Y) ** 2)


def token_sends(w, l, x):
    return jnp.concatenate([slot_output(*kqv(w, l, a, x), 1.0) for a in range(HEADS)])


def head_means(w, l, x, heard):
    """Mean on nonnegative received weights, plus the positive self-term.

    The restricted domain is essential: arbitrary real histories can cancel the
    denominator. P53(j), b2b2dfd4, checks that pole and proves domain preservation.
    """
    means = []
    for a in range(HEADS):
        k, p, v = kqv(w, l, a, x)
        tot = heard[a * (DV + 1):(a + 1) * (DV + 1)] + jnp.exp(jnp.dot(p, k)) * jnp.concatenate([v, jnp.ones(1)])
        means.append(tot[:DV] / tot[DV])
    return means


def token_color():
    """A token: two slots (heads); parameter its embedding and the weights; round 2 sends the keys, queries and values
    of ``x^1``, computed from both heads' round-1 hearing."""

    def strategy(s1, theta, l, H):
        x0, w = theta[:D], unpack(theta[D:])[0]
        x1 = layer_one(w, x0, head_means(w, 0, x0, H[0]))
        return token_sends(w, 1, x1)

    return Color(HEADS, D + W_DIM, strategy, label="token")


def token(c, x0, y, mobility=0.1):
    """The token as an arrangement ``<1 | R^W> -> B^2_c``: its open box reads the shared weights; its matter is its
    embedding; its cost is ``(1/2)|U x^2 - y|^2``, ``x^2`` computed from what its heads heard at both rounds."""
    hd = heard_dim(HEADS, DV)

    def emit(x, W):
        return jnp.concatenate([token_sends(unpack(W)[0], 0, x), x, W])

    def cost(x, W, e):
        w = unpack(W)[0]
        x1 = layer_one(w, x, head_means(w, 0, x, e[:hd]))
        o2 = head_means(w, 1, x1, e[hd:2 * hd])
        x2 = x1 + sum(w[f"O1{a}"] @ o2[a] for a in range(HEADS))
        return 0.5 * jnp.sum((w["U"] @ x2 - y) ** 2)

    return Arrangement(
        P=configuration(euclidean(D, mobility), init=x0), in_dim_M=0, out_dim_M=W_DIM,
        in_dim_N=in_dim(c, 2, DV, DK), out_dim_N=out_dim(c, DV, DK),
        out_f=emit, in_f=lambda x, W, n_in: jnp.zeros(0), omega=exact(lambda x, W, e: -cost(x, W, e)),
        label="token",
    )


def attention_group():
    """One operation of ``Addr``: ``N_TOK`` children of color ``HEADS``, no placement, nothing shown; head ``a`` of
    token ``x`` hears head ``a`` of every earlier token ``y`` by ``exp <p_x, k_y>`` (the causal mask and the heads are
    fixed zeros of the hearing)."""

    def hearing(q, x, y, Z):
        if x[1] != y[1] or y[0] > x[0]:
            return zero_transfer(DV)
        g = jnp.exp(jnp.dot(Z[x][1], Z[y][0]))
        return Transfer(gamma=g, L=g * jnp.eye(DV))

    return AddressedArrangement(
        slots=(HEADS,) * N_TOK, exposure=(), dim_H=DV, dim_P=DK, P=configuration(trivial()),
        placement=lambda q: jnp.zeros((N_TOK, DK)), hearing=hearing, label="causal-softmax",
    )


def two_layer(W, X0, Y):
    """The closed system ``ground . Phi_2(group) . (tokens) . copy . weights``; its configuration ``(W, x^0_1..n)``."""
    c = token_color()
    weights = Arrangement(P=configuration(euclidean(W_DIM, 0.1), init=W), in_dim_M=0, out_dim_M=0, in_dim_N=0,
                          out_dim_N=W_DIM, out_f=lambda q, m: q, in_f=lambda q, m, n: jnp.zeros(0), omega=zero_form,
                          label="weights")
    copy = Arrangement(P=configuration(trivial()), in_dim_M=0, out_dim_M=W_DIM, in_dim_N=0, out_dim_N=N_TOK * W_DIM,
                       out_f=lambda q, w: jnp.tile(w, N_TOK), in_f=lambda q, w, n: jnp.zeros(0), omega=zero_form,
                       label="copy")
    toks = tensor_arrangements([token(c, X0[t], Y[t]) for t in range(N_TOK)])
    grp, cg = rounds(attention_group(), [c] * N_TOK, 2)
    closed = compose_seq(compose_seq(compose_seq(compose_seq(weights, copy), toks), grp), ground(cg, 2, DV, DK))
    return closed, grp, c


def transformer_data(seed=7):
    g = np.random.default_rng(seed)
    return (jnp.asarray(g.standard_normal(W_DIM) * 0.5), jnp.asarray(g.standard_normal((N_TOK, D))),
            jnp.asarray(g.standard_normal((N_TOK, D_OUT))))


def test_two_layer_transformer_forward():
    """COMPARES with the plain implementation (prediction 2, forward). What each head of each token hears at each
    round, with the token's own term and its mean, is that layer's masked multi-head attention output."""
    W, X0, Y = transformer_data()
    _, grp, c = two_layer(W, X0, Y)
    m_out = jnp.concatenate([jnp.concatenate([token_sends(unpack(W)[0], 0, X0[t]), X0[t], W]) for t in range(N_TOK)])
    e = grp.in_f(jnp.zeros(0), m_out, jnp.zeros(DK))
    _, o1, o2 = transformer(W, X0)
    w = unpack(W)[0]
    blk = in_dim(c, 2, DV, DK)
    hd = heard_dim(HEADS, DV)
    for t in range(N_TOK):
        et = e[t * blk:(t + 1) * blk]
        m1 = head_means(w, 0, X0[t], et[:hd])
        x1 = layer_one(w, X0[t], m1)
        m2 = head_means(w, 1, x1, et[hd:2 * hd])
        for a in range(HEADS):
            np.testing.assert_allclose(m1[a], o1[a][t], atol=1e-12)
            np.testing.assert_allclose(m2[a], o2[a][t], atol=1e-12)


def test_two_layer_transformer_forces_are_backpropagation():
    """COMPARES with ``jax.grad`` of the plain implementation (prediction 2, forces). The closed system's effort field
    at the weights and the embeddings is minus the loss's gradient; with layer 2's keys and queries frozen the
    gradient differs, so the path from earlier hearing into later keys is carried."""
    W, X0, Y = transformer_data()
    closed, _, _ = two_layer(W, X0, Y)
    z = jnp.zeros(0)
    q = jnp.concatenate([W, jnp.ravel(X0)])
    force = closed.omega(q, z, z)[0]
    gW, gX = jax.grad(transformer_loss, argnums=(0, 1))(W, X0, Y)
    ref = -jnp.concatenate([gW, jnp.ravel(gX)])
    np.testing.assert_allclose(force, ref, atol=1e-10)
    fW, fX = jax.grad(lambda W_, X_: transformer_loss(W_, X_, Y, freeze_keys=True), argnums=(0, 1))(W, X0)
    frozen = -jnp.concatenate([fW, jnp.ravel(fX)])
    gap = float(jnp.max(jnp.abs(force - frozen)))
    print(f"max |force - frozen-keys force| = {gap:.4g}; max |force| = {float(jnp.max(jnp.abs(force))):.4g}")
    assert gap > 1e-3


# ---------------------------------------------------------------------------
# Prediction 3: a function per slot does not survive.
# ---------------------------------------------------------------------------


def test_strategy_is_a_function_of_the_whole_box():
    """ILLUSTRATES (prediction 3). (a) A token's head-0 key at round 2 depends on what head 1 heard at round 1.
    (b) Two one-slot members, both shown: at round 2 a shown slot's send depends only on what that slot heard; at
    round 3 it depends on what the other shown slot heard at round 1, through the members' round-2 hearing inside."""
    W, X0, _ = transformer_data()
    c = token_color()
    s1 = token_sends(unpack(W)[0], 0, X0[0])
    theta = jnp.concatenate([X0[0], W])
    H = jnp.asarray(np.random.default_rng(3).standard_normal((1, heard_dim(HEADS, DV)))) + jnp.asarray(
        [0, 0, 2.0, 0, 0, 2.0])
    J = jax.jacobian(lambda H_: c.strategy(s1, theta, 2, H_))(H)          # (sends, 1, heard)
    key0_vs_head1 = J[:DK, 0, DV + 1:]
    print("d key(head 0, round 2) / d heard(head 1, round 1):", np.asarray(key0_vs_head1).round(4).tolist())
    assert float(jnp.max(jnp.abs(key0_vs_head1))) > 1e-3

    a = random_op((1, 1), [(0, 0), (1, 0)], 51)
    arr, ca = rounds(a, [mealy(1, 52), mealy(1, 53)], 3)
    g = np.random.default_rng(54)
    q = jnp.asarray(g.standard_normal(a.P.Q_dim) * 0.5)
    out = arr.out_f(q, random_outputs([mealy(1, 52), mealy(1, 53)], 55))
    H3 = jnp.asarray(g.standard_normal((2, heard_dim(2, DH))) * 0.5)
    J2 = jax.jacobian(lambda H_: strategy_sends(ca, out, 2, H_[:1], DH, DP))(H3)
    J3 = jax.jacobian(lambda H_: strategy_sends(ca, out, 3, H_, DH, DP))(H3)
    other2 = J2[:SD, 0, HD:]           # slot 0's round-2 send against slot 1's round-1 hearing
    other3 = J3[:SD, 0, HD:]           # slot 0's round-3 send against slot 1's round-1 hearing
    print(f"round 2: {float(jnp.max(jnp.abs(other2))):.3g}; round 3: {float(jnp.max(jnp.abs(other3))):.3g}")
    assert float(jnp.max(jnp.abs(other2))) == 0.0
    assert float(jnp.max(jnp.abs(other3))) > 1e-3


# ---------------------------------------------------------------------------
# Prediction 4: comparison lenses.
# ---------------------------------------------------------------------------


def test_one_round_sits_inside_by_iota():
    """CHECKS A PROOF (prediction 4a). With Moore children, ``Phi_L(a) . (iota (x) iota) = iota . Phi_1(a)``: same
    sends at every round, same receiving map, same effort field; the last-round cost is what makes the last agree."""
    a = random_op((2, 1), [(0, 1), (1, 0)], 61, speakers=1)
    for L in (2, 3):
        lhs, cl = rounds(a, [moore(2), moore(1)], L)
        lhs = compose_seq(tensor_arrangements([iota(2, L, DH, DP), iota(1, L, DH, DP)]), lhs)
        rhs = compose_seq(a.arr(), iota(a.r, L, DH, DP))
        assert_same_arrangement_up_to_strategy(lhs, cl, rhs, moore(a.r), L, seed=62 + L)


def test_the_papers_operad_by_dropping_places():
    """CHECKS A PROOF (prediction 4b). With ``pi : B_1 -> B`` the static lens that drops a slot's places,
    ``Phi_paper(a) . pi^n = pi . Phi_1(j(a))``, ``j`` P5 (g)'s map (each group one speaker at its center)."""
    from arrangements.addressed import in_dim as addr_in_dim

    def pi():
        return Arrangement(P=configuration(trivial()), in_dim_M=addr_in_dim(1, DH, DP), out_dim_M=SD,
                           in_dim_N=HD + DP, out_dim_N=HD, out_f=lambda q, s: s[2 * DP:],
                           in_f=lambda q, s, n: n, omega=zero_form, label="pi")

    for n, seed in [(2, 71), (3, 72)]:
        att = paper_part(n, seed)
        lhs = compose_seq(tensor_arrangements([pi() for _ in range(n)]), att.arr())
        rhs = compose_seq(from_paper(att).arr(), pi())
        g = np.random.default_rng(seed + 1)
        for _ in range(3):
            q = jnp.asarray(g.standard_normal(lhs.P.Q_dim) * 0.5)
            m = jnp.asarray(g.standard_normal(lhs.out_dim_M) * 0.5)
            e = jnp.asarray(g.standard_normal(lhs.in_dim_N) * 0.5)
            np.testing.assert_allclose(lhs.out_f(q, m), rhs.out_f(q, m), atol=1e-12)
            np.testing.assert_allclose(lhs.in_f(q, m, e), rhs.in_f(q, m, e), atol=1e-12)
            for u, v in zip(lhs.omega(q, m, e), rhs.omega(q, m, e)):
                np.testing.assert_allclose(u, v, atol=1e-11)


# ---------------------------------------------------------------------------
# Prediction 5: normalizing inside the group is not an operation.
# ---------------------------------------------------------------------------


def test_normalized_hearing_is_not_linear_in_messages():
    """ILLUSTRATES (prediction 5). Every operation's receiving map is linear in the messages at fixed configuration and
    places; the normalized hearing ``sum_y softmax_y v_y`` is unchanged when every weight doubles, so it is not."""
    g = np.random.default_rng(81)
    k, p = jnp.asarray(g.standard_normal((3, DP))), jnp.asarray(g.standard_normal(DP))
    v, w = jnp.asarray(g.standard_normal((3, DH))), jnp.asarray([1.0, 2.0, 0.5])
    kap = jnp.exp(k @ p)

    def normalized(w_):
        return (kap * w_) @ v / jnp.sum(kap * w_)

    np.testing.assert_allclose(normalized(2 * w), normalized(w), atol=1e-14)
    assert float(jnp.max(jnp.abs(normalized(w)))) > 1e-2
