"""P92 (book plan §8.2), *The grouping does the addressing*: a transformer whose every weight is the arrangement's.

Claude Opus 5.5, 2fb1d833, 2026-11-02; everything here is the agents' and *proposed* until David's answer is recorded
in P92's block. The construction (book repository, ``notes/runs/p92-2fb1d833/CONSTRUCTION.md``):

* A token *just speaks*: at round ``l`` it sends its residual stream ``x^l`` as its address and
  ``msg(x^l) = (x^l, 1)`` as its one message, and it receives one message per head and a rule message; the arrangement
  reads both the key and the query off the address. (Book Ch 4 ``sec.addressed_messages`` since 2026-11-03, 586b6a53:
  an object is an address space with a list of vector spaces sent in and a list received in, no slots.) Its parameter
  is its initial embedding ``x^0`` alone: it holds no weight.
* The arrangement's configuration ``q`` holds every layer's weights. Round ``l``'s transfer maps read layer ``l``'s
  factor of ``q`` only (rounds are a power of the base category, so each round has its own lens on one matter):
  head ``a`` of token ``i`` hears head ``a`` of every token ``j`` (``j = i`` included) by
  ``exp(<W_Q^{la} x_i, W_K^{la} x_j> / sqrt(d)) . msgfun(W_O^{la} W_V^{la})``; and from the matter's voice the rule slot
  of every token hears ``msg(rho^l)``, layer ``l``'s feed-forward and layer-normalization weights.
* The token's strategy is the fixed architecture: ``x~ = LN(x^l + sum_a val(e^{l,a}); rho^l)``,
  ``x^{l+1} = LN(x~ + FFN(x~; rho^l); rho^l)``, with ``rho^l = val(e^{l, rule})``.

Checks: the forward pass is a reference post-LN transformer's with separate weights per layer; the closed system's
effort field at the weights and the embeddings is minus the reference loss's gradient; and, untying ``rho^l`` into one
copy per token's rule slot, the load on the shared ``rho^l`` is the sum of the copies' loads (the transpose of the copy).
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact, zero_form
from arrangements.coalgebra import configuration
from arrangements.rvect import euclidean, trivial
from arrangements.wiring import compose_seq, tensor_arrangements

jax.config.update("jax_enable_x64", True)

N, L, HEADS, M, DK, DV, DFF = 2, 2, 2, 3, 2, 2, 4
EPS = 1e-5

# ---------------------------------------------------------------------------
# Weights: per layer, attention (W_Q, W_K, W_V, W_O per head) and rho (FFN and both LNs).
# ---------------------------------------------------------------------------

ATT_SHAPES = [(f"{w}{a}", s) for a in range(HEADS)
              for w, s in (("Q", (DK, M)), ("K", (DK, M)), ("V", (DV, M)), ("O", (M, DV)))]
RHO_SHAPES = [("W1", (DFF, M)), ("c1", (DFF,)), ("W2", (M, DFF)), ("c2", (M,)),
              ("g1", (M,)), ("b1", (M,)), ("g2", (M,)), ("b2", (M,))]


def size(shapes):
    return sum(int(np.prod(s)) for _, s in shapes)


ATT, RHO = size(ATT_SHAPES), size(RHO_SHAPES)
LAYER = ATT + RHO
QDIM = L * LAYER


def unpack(v, shapes):
    out, c = {}, 0
    for name, s in shapes:
        k = int(np.prod(s))
        out[name] = jnp.reshape(v[c:c + k], s)
        c += k
    return out


def layer_of(q, l):
    """Layer ``l``'s factor of the configuration: its attention weights and its rho."""
    blk = q[l * LAYER:(l + 1) * LAYER]
    return unpack(blk[:ATT], ATT_SHAPES), blk[ATT:]


def ln(x, g, b):
    mu = x.mean()
    return g * (x - mu) / jnp.sqrt(((x - mu) ** 2).mean() + EPS) + b


def ffn(r, x):
    return r["W2"] @ jax.nn.gelu(r["W1"] @ x + r["c1"]) + r["c2"]


# ---------------------------------------------------------------------------
# The reference: a post-LN transformer (Vaswani et al. §3), separate weights per layer, shared across tokens.
# ---------------------------------------------------------------------------


def reference(q, X):
    outs = []
    for l in range(L):
        w, rho = layer_of(q, l)
        r = unpack(rho, RHO_SHAPES)
        heads = []
        for a in range(HEADS):
            S = (X @ w[f"Q{a}"].T) @ (X @ w[f"K{a}"].T).T / jnp.sqrt(DK)
            heads.append(jax.nn.softmax(S, axis=1) @ (X @ w[f"V{a}"].T) @ w[f"O{a}"].T)
        outs.append(heads)
        Xt = jnp.stack([ln(X[i] + sum(h[i] for h in heads), r["g1"], r["b1"]) for i in range(N)])
        X = jnp.stack([ln(Xt[i] + ffn(r, Xt[i]), r["g2"], r["b2"]) for i in range(N)])
    return X, outs


def ref_loss(q, X, Y):
    return 0.5 * jnp.sum((reference(q, X)[0] - Y) ** 2)


# ---------------------------------------------------------------------------
# The token: what it sends, and its strategy (the fixed architecture, no weights of its own).
# ---------------------------------------------------------------------------

SEND = M + M                 # the token's send: address x, value x (Ch 4 sec.softmax_attention)
HEAR = HEADS * M + (RHO + HEADS * M * M)   # per round: each head value, the rule value (rho and every W_O W_V)


def sends_of(x):
    return jnp.concatenate([x, x])


def step(x, e):
    """The strategy's update from one round's hearing ``e``: apply the rule's W_O W_V to each head value, residual,
    LN, FFN, residual, LN, with ``rho`` and the maps W_O W_V told in the rule value (Ch 4 lem.values_natural)."""
    raw_heads = e[:HEADS * M].reshape(HEADS, M)
    rule = e[HEADS * M:]
    r = unpack(rule[:RHO], RHO_SHAPES)
    maps = rule[RHO:].reshape(HEADS, M, M)
    heads = [maps[a] @ raw_heads[a] for a in range(HEADS)]
    xt = ln(x + sum(heads), r["g1"], r["b1"])
    return ln(xt + ffn(r, xt), r["g2"], r["b2"])


def stream(x0, E):
    """``x^0, ..., x^k`` from the parameter and the hearings ``E = (e^0, ..., e^{k-1})``."""
    xs = [x0]
    for e in E:
        xs.append(step(xs[-1], e))
    return xs


# ---------------------------------------------------------------------------
# The operation phi_n, round by round, and Phi^fin of it as an arrangement.
# ---------------------------------------------------------------------------


def hear_round(q, l, S, rho_copies=None):
    """Round ``l``'s in-transfer: what each token's slots hear from the round-``l`` sends ``S`` (rows = tokens).
    ``rho_copies`` (rows = tokens), when given, replaces the one ``rho^l`` by a copy per token, untying them."""
    w, rho = layer_of(q, l)
    addrs = [S[j, :M] for j in range(N)]                      # each token's address
    vals = [S[j, M:SEND] for j in range(N)]                   # each token's value
    out = []
    for i in range(N):
        e = []
        for a in range(HEADS):
            scores = jnp.stack([jnp.dot(w[f"Q{a}"] @ addrs[i], w[f"K{a}"] @ addrs[j]) / jnp.sqrt(DK) for j in range(N)])
            P = jax.nn.softmax(scores)                                          # the head row, a distribution (eqn.softmax_attention_hearing)
            e.append(sum(P[j] * vals[j] for j in range(N)))                    # the mixture of the streams
        r_i = rho if rho_copies is None else rho_copies[i]
        maps = [jnp.ravel(w[f"O{a}"] @ w[f"V{a}"]) for a in range(HEADS)]
        e.append(jnp.concatenate([r_i] + maps))                                # the rule value: rho^l and every W_O W_V, from the voice
        out.append(jnp.concatenate(e))
    return jnp.stack(out)


def realize(q, m_out, untie=None):
    """All ``L`` rounds in one pass: round 0 sends from ``m_out``; round ``l`` sends by each token's strategy.
    ``untie = (l, copies)`` gives round ``l``'s rule slots one copy of ``rho^l`` per token."""
    S0 = jnp.reshape(m_out[:N * SEND], (N, SEND))
    X0 = jnp.reshape(m_out[N * SEND:], (N, M))
    hist, S = [], S0
    for l in range(L):
        Ecur = hear_round(q, l, S, untie[1] if untie is not None and untie[0] == l else None)
        hist.append(Ecur)
        if l + 1 < L:
            S = jnp.stack([sends_of(stream(X0[i], [h[i] for h in hist])[-1]) for i in range(N)])
    return hist


def group():
    """``Phi^fin_L(phi_n)``: tokens' outputs ``[s^0_i]_i, [x^0_i]_i`` in; it shows no slot and sends its parameter."""
    def out_f(q, m_out):
        return jnp.concatenate([q, m_out])

    def in_f(q, m_out, n_in):
        hist = realize(q, m_out)
        return jnp.concatenate([jnp.concatenate([h[i] for h in hist]) for i in range(N)])

    return Arrangement(P=configuration(euclidean(QDIM, 0.1)), in_dim_M=N * L * HEAR, out_dim_M=N * (SEND + M),
                       in_dim_N=0, out_dim_N=QDIM + N * (SEND + M), out_f=out_f, in_f=in_f, omega=zero_form,
                       label="phi_n")


def token(i, y):
    """Token ``i``: matter its embedding; emits its round-0 sends and its parameter; cost on its final stream."""
    def cost(x, e):
        E = [e[l * HEAR:(l + 1) * HEAR] for l in range(L)]
        return 0.5 * jnp.sum((stream(x, E)[-1] - y) ** 2)

    return Arrangement(P=configuration(euclidean(M, 0.1)), in_dim_M=0, out_dim_M=0, in_dim_N=L * HEAR,
                       out_dim_N=SEND + M, out_f=lambda x, _: jnp.concatenate([sends_of(x), x]),
                       in_f=lambda x, _, n: jnp.zeros(0), omega=exact(lambda x, _, e: -cost(x, e)),
                       label=f"token{i}")


def reorder(n_tokens_arr):
    """The tokens' tensor emits ``[s^0_1, x_1, s^0_2, x_2]``; the group reads ``[s^0_1, s^0_2, x_1, x_2]``."""
    def out_f(q, m):
        blocks = [m[i * (SEND + M):(i + 1) * (SEND + M)] for i in range(N)]
        return jnp.concatenate([b[:SEND] for b in blocks] + [b[SEND:] for b in blocks])

    return Arrangement(P=configuration(trivial()), in_dim_M=N * L * HEAR, out_dim_M=N * (SEND + M),
                       in_dim_N=N * L * HEAR, out_dim_N=N * (SEND + M), out_f=out_f, in_f=lambda q, m, n: n,
                       omega=zero_form, label="reorder")


def ground():
    return Arrangement(P=configuration(trivial()), in_dim_M=0, out_dim_M=QDIM + N * (SEND + M), in_dim_N=0,
                       out_dim_N=0, out_f=lambda q, m: jnp.zeros(0), in_f=lambda q, m, n: jnp.zeros(0),
                       omega=zero_form, label="ground")


def closed_system(Y):
    toks = tensor_arrangements([token(i, Y[i]) for i in range(N)])
    return compose_seq(compose_seq(compose_seq(toks, reorder(N)), group()), ground())


def data(seed=3):
    g = np.random.default_rng(seed)
    q = []
    for l in range(L):
        att = g.standard_normal(ATT) * 0.6
        r = {n: g.standard_normal(s) * 0.5 for n, s in RHO_SHAPES}
        for n in ("g1", "g2"):
            r[n] = 1.0 + 0.2 * g.standard_normal(M)
        q.append(np.concatenate([att] + [np.ravel(r[n]) for n, _ in RHO_SHAPES]))
    return jnp.asarray(np.concatenate(q)), jnp.asarray(g.standard_normal((N, M))), jnp.asarray(g.standard_normal((N, M)))


# ---------------------------------------------------------------------------
# The checks.
# ---------------------------------------------------------------------------


def test_forward_is_the_reference_transformer():
    """COMPARES: each head's normalized hearing at round l is the reference's layer-l head output, and each token's
    final stream is the reference's x^L; the two layers' weights differ, so round l reads layer l's."""
    q, X0, _ = data()
    m_out = jnp.concatenate([sends_of(X0[i]) for i in range(N)] + [X0[i] for i in range(N)])
    hist = realize(q, m_out)
    XL, outs = reference(q, X0)
    for i in range(N):
        xs = stream(X0[i], [h[i] for h in hist])
        np.testing.assert_allclose(xs[-1], XL[i], atol=1e-12)
        for l in range(L):
            w, _ = layer_of(q, l)
            for a in range(HEADS):
                mixture = hist[l][i][a * M:(a + 1) * M]                       # the head value: the streams mixed by the softmax row
                np.testing.assert_allclose(w[f"O{a}"] @ w[f"V{a}"] @ mixture, outs[l][a][i], atol=1e-12)   # the token applies W_O W_V
    assert float(jnp.max(jnp.abs(q[:LAYER] - q[LAYER:]))) > 0.1


def test_effort_field_is_backpropagation():
    """COMPARES with jax.grad of the reference: the closed system's effort field on (embeddings, weights) is minus the
    gradient of the reference loss."""
    q, X0, Y = data()
    closed = closed_system(Y)
    z = jnp.zeros(0)
    force = closed.omega(jnp.concatenate([jnp.ravel(X0), q]), z, z)[0]
    gq, gX = jax.grad(ref_loss, argnums=(0, 1))(q, X0, Y)
    np.testing.assert_allclose(force, -jnp.concatenate([jnp.ravel(gX), gq]), atol=1e-10)
    assert float(jnp.max(jnp.abs(force))) > 1e-2


def test_load_on_rho_is_the_sum_over_tokens():
    """CHECKS the copy statement: untie layer l's rho into one copy per token's rule slot; the load on the shared rho
    is the sum of the copies' loads, and the copies' loads differ (untied copies would drift apart)."""
    q, X0, Y = data()
    m_out = jnp.concatenate([sends_of(X0[i]) for i in range(N)] + [X0[i] for i in range(N)])
    gq = jax.grad(ref_loss)(q, X0, Y)

    def loss_untied(copies, l):
        hist = realize(q, m_out, untie=(l, copies))
        return sum(0.5 * jnp.sum((stream(X0[i], [h[i] for h in hist])[-1] - Y[i]) ** 2) for i in range(N))

    for l in range(L):
        copies = jnp.tile(layer_of(q, l)[1], (N, 1))
        loads = -jax.grad(loss_untied)(copies, l)
        np.testing.assert_allclose(loads.sum(axis=0), -gq[l * LAYER + ATT:(l + 1) * LAYER], atol=1e-10)
        assert float(jnp.max(jnp.abs(loads[0] - loads[1]))) > 1e-3


# ---------------------------------------------------------------------------
# The laws: operations whose transfer differs by round (the power C^{L} of the base category) still compose.
# ``rounds_by_round`` is ``addressed_rounds.rounds`` with round l reading ops[l]: one matter, one lens per round, and
# the effort field the sum of each round's cost (one 1-form on all rounds).
# ---------------------------------------------------------------------------

from dataclasses import replace  # noqa: E402

from arrangements.addressed import substitute, unit  # noqa: E402
from arrangements.addressed_rounds import Color, _split_outputs, heard_dim, in_dim as r_in_dim  # noqa: E402
from arrangements.addressed_rounds import ground as r_ground, out_dim as r_out_dim, strategy_sends  # noqa: E402
from arrangements.attention import Transfer  # noqa: E402
from tests.test_addressed import DH, DP, random_op  # noqa: E402
from tests.test_addressed_rounds import (  # noqa: E402
    SD, assert_same_arrangement_up_to_strategy, mealy, random_leaf, random_outputs,
)


def realize_by_round(ops, colors, q, m_out, H_ext, p0, upto):
    a = ops[0]
    dH, dP = a.dim_H, a.dim_P
    parts = _split_outputs(colors, m_out, dH, dP)
    hist, sends, pos = [[] for _ in colors], [], None
    for l in range(1, upto + 1):
        if l == 1:
            s_l = jnp.concatenate([s1 for s1, _ in parts]) if parts else jnp.zeros(0)
        else:
            s_l = jnp.concatenate([col.strategy(s1, th, l, jnp.stack(hist[i]))
                                   for i, (col, (s1, th)) in enumerate(zip(colors, parts))])
        sends.append(s_l)
        rho = ops[l - 1].rho(q, s_l, jnp.concatenate([H_ext[l - 1], p0]))
        c, pos = 0, []
        for i, col in enumerate(colors):
            hd = heard_dim(col.r, dH)
            hist[i].append(rho[c:c + hd])
            pos.append(rho[c + hd:c + hd + dP])
            c += hd + dP
    return sends, hist, pos


def rounds_by_round(ops, colors, L_):
    ops, colors = tuple(ops), tuple(colors)
    a = ops[0]
    dH, dP, r = a.dim_H, a.dim_P, a.r
    hd, Qd = heard_dim(r, dH), a.P.Q_dim
    m_dim = sum(r_out_dim(c, dH, dP) for c in colors)
    zp = jnp.zeros(dP)

    def strategy(s1, theta, l, H):
        q, m_out = theta[:Qd], theta[Qd:]
        if l == 1:
            return s1
        _, hist, _ = realize_by_round(ops, colors, q, m_out, [H[j] for j in range(l - 1)], zp, l - 1)
        parts = _split_outputs(colors, m_out, dH, dP)
        s_l = jnp.concatenate([col.strategy(s1_i, th_i, l, jnp.stack(hist[i]))
                               for i, (col, (s1_i, th_i)) in enumerate(zip(colors, parts))])
        return ops[l - 1].sigma(q, s_l)

    out_color = Color(r, Qd + m_dim, strategy, label="rbr")

    def ext(n_in):
        return [n_in[l * hd:(l + 1) * hd] for l in range(L_)], n_in[L_ * hd:]

    def out_f(q, m_out):
        s1 = jnp.concatenate([s for s, _ in _split_outputs(colors, m_out, dH, dP)]) if colors else jnp.zeros(0)
        return jnp.concatenate([ops[0].sigma(q, s1), q, m_out])

    def in_f(q, m_out, n_in):
        H_ext, p0 = ext(n_in)
        _, hist, pos = realize_by_round(ops, colors, q, m_out, H_ext, p0, L_)
        return jnp.concatenate([jnp.concatenate(hist[i] + [pos[i]]) for i in range(len(colors))])

    def cost(q, m_out, n_in):
        H_ext, p0 = ext(n_in)
        sends, _, _ = realize_by_round(ops, colors, q, m_out, H_ext, p0, L_)
        return sum(ops[l].cost(q, sends[l], jnp.concatenate([H_ext[l], p0])) for l in range(L_))

    arr = Arrangement(P=a.P, in_dim_M=sum(r_in_dim(c, L_, dH, dP) for c in colors), out_dim_M=m_dim,
                      in_dim_N=r_in_dim(out_color, L_, dH, dP), out_dim_N=r_out_dim(out_color, dH, dP),
                      out_f=out_f, in_f=in_f, omega=exact(lambda q, m, n: -cost(q, m, n)), label="rbr")
    return arr, out_color


def by_round(op, L_):
    """Round l's copy of ``op``: the same matter, its hearing scaled by a positive factor read off a round-l
    coordinate of the configuration, so each round has its own transfer maps."""
    def at(l):
        def hearing(q, x, y, Z):
            T = op.hearing(q, x, y, Z)
            s = jnp.exp(0.5 * jnp.sin(q[l % q.shape[0]] + 1.3 * l))
            return Transfer(gamma=s * T.gamma, L=s * T.L)
        return replace(op, hearing=hearing, label=f"{op.label}@{l}")
    return [at(l) for l in range(L_)]


def test_round_by_round_substitution_is_composition():
    """CHECKS A PROOF (rounds as a power of the base category): for a two-input composite b = a(a1, a2), with uneven
    arities, a speaker and a hidden slot, and each of a, a1, a2 having its own transfer at each round, Phi^fin(b) =
    Phi^fin(a) . (Phi^fin(a1) (x) Phi^fin(a2)), b substituted round by round; closed, the forces agree."""
    a = random_op((2, 1), [(0, 1), (1, 0)], 1, speakers=1)
    a1 = random_op((1, 2), [(1, 0), (0, 0)], 2)
    a2 = random_op((1, 1), [(1, 0)], 3)
    grand = [mealy(1, 10), mealy(2, 11), mealy(1, 12), mealy(1, 13)]
    for L_ in (2, 3):
        A, A1, A2 = by_round(a, L_), by_round(a1, L_), by_round(a2, L_)
        B = [substitute(A[l], [A1[l], A2[l]]) for l in range(L_)]
        explicit, cb = rounds_by_round(B, grand, L_)
        p1, c1 = rounds_by_round(A1, grand[:2], L_)
        p2, c2 = rounds_by_round(A2, grand[2:], L_)
        pa, ca = rounds_by_round(A, [c1, c2], L_)
        composite = compose_seq(tensor_arrangements([p1, p2]), pa)
        assert_same_arrangement_up_to_strategy(explicit, cb, composite, ca, L_, seed=20 + L_)

        top = random_op((a.r,), [], 5)
        T = by_round(top, L_)
        tb, ctb = rounds_by_round(T, [cb], L_)
        tc, ctc = rounds_by_round(T, [ca], L_)
        leaves = tensor_arrangements([random_leaf(c, L_, 30 + i) for i, c in enumerate(grand)])
        closed_b = compose_seq(compose_seq(compose_seq(leaves, explicit), tb), r_ground(ctb, L_, DH, DP))
        closed_c = compose_seq(compose_seq(compose_seq(leaves, composite), tc), r_ground(ctc, L_, DH, DP))
        z = jnp.zeros(0)
        qq = jnp.asarray(np.random.default_rng(40 + L_).standard_normal(closed_b.P.Q_dim) * 0.5)
        fb, fc = closed_b.omega(qq, z, z)[0], closed_c.omega(qq, z, z)[0]
        np.testing.assert_allclose(fb, fc, atol=1e-10)
        assert float(jnp.max(jnp.abs(fb))) > 1e-2
        # the rounds really differ: round 0's and round 1's hearings of one pair are not equal
        Z = {x: (jnp.ones(DP) * 0.1, jnp.ones(DP) * 0.2) for x in a.child_slots()}
        qa = a.P.init
        assert abs(float(A[0].hearing(qa, (0, 0), (1, 0), Z).gamma - A[1].hearing(qa, (0, 0), (1, 0), Z).gamma)) > 1e-3


def test_round_by_round_units():
    """CHECKS A PROOF (units): the unit at every round is the identity on B^L_c."""
    for L_, c in [(2, mealy(2, 91)), (3, mealy(1, 92))]:
        arr, cu = rounds_by_round([unit(c.r, DH, DP)] * L_, [c], L_)
        g = np.random.default_rng(93 + L_)
        qq, m = jnp.zeros(0), random_outputs([c], 94 + L_)
        e = jnp.asarray(g.standard_normal(arr.in_dim_N))
        out = arr.out_f(qq, m)
        np.testing.assert_allclose(out[:c.r * SD], m[:c.r * SD], atol=1e-14)
        for l in range(2, L_ + 1):
            H = jnp.asarray(g.standard_normal((l - 1, heard_dim(c.r, DH))))
            np.testing.assert_allclose(strategy_sends(cu, out, l, H, DH, DP), strategy_sends(c, m, l, H, DH, DP),
                                       atol=1e-13)
        np.testing.assert_allclose(arr.in_f(qq, m, e), e, atol=1e-14)
