"""A learner beside a physical arrangement: a two-layer attention predictor wired to pendulums (book plan P5 (l)
stage C; the design and the predictions written before this code are in the book repository,
``notes/runs/p5lC-a9e113da/RESULTS.md``; everything here is the agents' and *proposed*).

One closed arrangement, run by ``Phi`` with nothing computed between steps. Per start ``(q0, p0)``: a pendulum
(``system_id.observed_oscillator``, phase integrator, reading out its presented position ``a = q + p/m``) and ``h``
delay boxes in a chain (``system_id.delay_box``: ``d_1`` hears ``a``, ``d_k`` hears ``d_{k-1}``; each hears without
pulling and ignores the force on what it presents). One weights box (configuration integrator on ``(R^n, eta I)``)
holds every weight of the predictor. A static route copies the weights and the delayed positions forward and adds the
forces back. Per pendulum, one predictor ``ground . Phi_2(causal group) . (token_1 (x) ... (x) token_h)``: the
two-layer causal block of the library's ``tests/test_addressed_rounds.py``, with ``h`` tokens. Token ``i`` is
stateless; it hears the weights and the delayed position ``a_{t-h+i}`` (oldest first), and its embedding is
``x0_i = a_{t-h+i} e + pi_i``, ``e`` and ``pi_i`` among the weights.

**The loss is the last token's cost.** The answer ``U x2_h + c`` is computed from what the last token hears at round 2,
and a box reads out only from its state and what it hears from inside, so no readout can carry the answer to a separate
loss box. The last token also hears the label ``a_t`` from the pendulum, and its 1-form is ``-d`` of
``(1/2)(U x2_h + c - a_t)^2`` in the weights, its delayed position and its round hearings, and ``0`` on the label:
``system_id.one_way_squared_error``'s convention, so the pendulum is not moved. The other tokens carry no cost.

Three clocks: the physical step (one ``Phi`` step, duration 1, moves each pendulum once); the two rounds inside one
evaluation of the predictor, with no duration; the learner's step (the weights' response ``eta`` times the summed
force, once per ``Phi`` step).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .addressed import AddressedArrangement, slot_output
from .addressed_rounds import Color, ground, heard_dim, in_dim, out_dim, rounds
from .arrangement import Arrangement, zero_form
from .attention import Transfer, zero_transfer
from .coalgebra import configuration
from .rvect import euclidean, trivial
from .system_id import PENDULUM, delay_box, observed_oscillator


@dataclass(frozen=True)
class Block:
    """The predictor's sizes: ``h`` tokens, width ``d``, ``heads`` heads of key/query width ``dk`` and value width
    ``dv``, a hidden layer of ``d_mlp`` after layer 1."""

    h: int = 2
    d: int = 4
    heads: int = 2
    dk: int = 2
    dv: int = 2
    d_mlp: int = 3

    def shapes(self) -> List[Tuple[str, Tuple[int, ...]]]:
        s = [("e", (self.d,)), ("pi", (self.h, self.d))]
        for l in range(2):
            for a in range(self.heads):
                s += [(f"Q{l}{a}", (self.dk, self.d)), (f"K{l}{a}", (self.dk, self.d)),
                      (f"V{l}{a}", (self.dv, self.d)), (f"O{l}{a}", (self.d, self.dv))]
        return s + [("W1", (self.d_mlp, self.d)), ("b1", (self.d_mlp,)), ("W2", (self.d, self.d_mlp)),
                    ("U", (1, self.d)), ("c", (1,))]

    @property
    def n(self) -> int:
        return sum(int(np.prod(sh)) for _, sh in self.shapes())

    def unpack(self, W: Array) -> Dict[str, Array]:
        out, c = {}, 0
        for name, sh in self.shapes():
            k = int(np.prod(sh))
            out[name] = jnp.reshape(W[c:c + k], sh)
            c += k
        return out


# ---------------------------------------------------------------------------
# The block's pieces (as in tests/test_addressed_rounds.py, sized by a Block).
# ---------------------------------------------------------------------------


def kqv(bk: Block, w, l: int, a: int, x: Array):
    return w[f"K{l}{a}"] @ x, w[f"Q{l}{a}"] @ x / jnp.sqrt(bk.dk), w[f"V{l}{a}"] @ x


def embed(w, i: int, obs: Array) -> Array:
    """``x0_i = obs e + pi_i``."""
    return obs * w["e"] + w["pi"][i]


def layer_one(bk: Block, w, x0: Array, means) -> Array:
    """``x1`` from ``x0`` and the heads' layer-1 means: residual, then a one-layer MLP."""
    xh = x0 + sum(w[f"O0{a}"] @ means[a] for a in range(bk.heads))
    return xh + w["W2"] @ jnp.tanh(w["W1"] @ xh + w["b1"])


def token_sends(bk: Block, w, l: int, x: Array) -> Array:
    return jnp.concatenate([slot_output(*kqv(bk, w, l, a, x), 1.0) for a in range(bk.heads)])


def head_means(bk: Block, w, l: int, x: Array, heard: Array):
    """Each head's mean over what it heard plus the token's own positive self-term (A1's suboperad form: the group
    never makes a token hear itself). The cone domain of P53 (j) keeps the denominator positive."""
    means = []
    for a in range(bk.heads):
        k, p, v = kqv(bk, w, l, a, x)
        tot = heard[a * (bk.dv + 1):(a + 1) * (bk.dv + 1)] + jnp.exp(jnp.dot(p, k)) * jnp.concatenate([v, jnp.ones(1)])
        means.append(tot[:bk.dv] / tot[bk.dv])
    return means


def token_color(bk: Block) -> Color:
    """A token: ``heads`` slots; parameter its embedding and the weights; round 2 sends the keys, queries and values
    of ``x1``, computed from every head's round-1 hearing."""

    def strategy(s1, theta, l, H):
        x0, w = theta[:bk.d], bk.unpack(theta[bk.d:])
        return token_sends(bk, w, 1, layer_one(bk, w, x0, head_means(bk, w, 0, x0, H[0])))

    return Color(bk.heads, bk.d + bk.n, strategy, label="token")


def answer(bk: Block, w, x0: Array, e: Array) -> Array:
    """``U x2 + c`` from the token's embedding and what its heads heard at both rounds."""
    hd = heard_dim(bk.heads, bk.dv)
    x1 = layer_one(bk, w, x0, head_means(bk, w, 0, x0, e[:hd]))
    o2 = head_means(bk, w, 1, x1, e[hd:2 * hd])
    x2 = x1 + sum(w[f"O1{a}"] @ o2[a] for a in range(bk.heads))
    return (w["U"] @ x2 + w["c"])[0]


def token(bk: Block, i: int) -> Arrangement:
    """Token ``i`` as a stateless arrangement ``<0 | R^n x R> -> B^2_c`` (the last, ``<0 | R^n x R x R> -> B^2_c``):
    it hears the weights and its delayed position (and the last, the label); it sends ``[s1, x0, W]``. The last
    token's 1-form is ``-d`` of the squared error in everything it hears but the label, and ``0`` on the label."""
    c, n, last = token_color(bk), bk.n, i == bk.h - 1

    def emit(q, m):
        w = bk.unpack(m[:n])
        x0 = embed(w, i, m[n])
        return jnp.concatenate([token_sends(bk, w, 0, x0), x0, m[:n]])

    def cost(W, obs, lab, e):
        w = bk.unpack(W)
        return 0.5 * (answer(bk, w, embed(w, i, obs), e) - lab) ** 2

    def omega(q, m, e):
        gW, gobs, ge = jax.grad(cost, argnums=(0, 1, 3))(m[:n], m[n], m[n + 1], e)
        return jnp.zeros_like(q), jnp.concatenate([-gW, -gobs[None], jnp.zeros(1)]), -ge

    return Arrangement(
        P=configuration(trivial()), in_dim_M=0, out_dim_M=n + 1 + int(last),
        in_dim_N=in_dim(c, 2, bk.dv, bk.dk), out_dim_N=out_dim(c, bk.dv, bk.dk),
        out_f=emit, in_f=lambda q, m, e: jnp.zeros(0), omega=omega if last else zero_form,
        label=f"token{i}" + ("+loss" if last else ""),
    )


def attention_group(bk: Block) -> AddressedArrangement:
    """One operation of ``Addr``: ``h`` children of color ``heads``, no placement, nothing shown; head ``a`` of token
    ``x`` hears head ``a`` of every earlier token ``y`` by ``exp <p_x, k_y>`` (the causal mask and the heads are fixed
    zeros of the hearing; a token never hears itself through the group)."""

    def hearing(q, x, y, Z):
        if x[1] != y[1] or y[0] > x[0]:
            return zero_transfer(bk.dv)
        g = jnp.exp(jnp.dot(Z[x][1], Z[y][0]))
        return Transfer(gamma=g, L=g * jnp.eye(bk.dv))

    return AddressedArrangement(
        slots=(bk.heads,) * bk.h, exposure=(), dim_H=bk.dv, dim_P=bk.dk, P=configuration(trivial()),
        placement=lambda q: jnp.zeros((bk.h, bk.dk)), hearing=hearing, label="causal-softmax",
    )


def predictor(bk: Block) -> Arrangement:
    """``ground . Phi_2(group) . (token_0 (x) ... (x) token_{h-1})``: from what the tokens hear to nothing."""
    from .wiring import compose_seq, tensor_arrangements

    c = token_color(bk)
    grp, cg = rounds(attention_group(bk), [c] * bk.h, 2)
    toks = tensor_arrangements([token(bk, i) for i in range(bk.h)])
    return compose_seq(compose_seq(toks, grp), ground(cg, 2, bk.dv, bk.dk))


# ---------------------------------------------------------------------------
# The closed arrangement.
# ---------------------------------------------------------------------------


def init_weights(bk: Block, seed: int, scale: float = 0.2) -> Array:
    return scale * jnp.asarray(np.random.default_rng(seed).standard_normal(bk.n))


def wired_attention(
    starts: Sequence[Tuple[float, float]],
    bk: Block = Block(),
    *,
    potential: Callable[[Array], Array] = PENDULUM[0],
    m: float = 1.0,
    eta: float = 1.5e-4,
    seed: int = 0,
    init: Array = None,
) -> Arrangement:
    """The closed arrangement: per start a pendulum and ``h`` delays, one weights box, a route, a predictor per
    pendulum. Leaves in order: per pendulum ``[pendulum, delay_1, ..., delay_h]``, then the weights; its position is
    ``(a_i, h_1, z_1, ..., h_h, z_h)`` per pendulum, then the weights."""
    from .wiring import compose_seq, tensor_arrangements, trivial_parameter

    K, h, n = len(starts), bk.h, bk.n
    W0 = init_weights(bk, seed) if init is None else jnp.asarray(init)
    weights = Arrangement(
        P=configuration(euclidean(n, eta), init=W0), in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=n,
        out_f=lambda q, m_out: q, in_f=lambda q, m_out, n_in: jnp.zeros(0), omega=zero_form, label="weights",
    )
    leaves = []
    for q0, p0 in starts:
        leaves += [observed_oscillator(potential, m, q0, p0)] + [delay_box() for _ in range(h)]
    parts = tensor_arrangements(leaves + [weights])
    B = 1 + h                       # outputs per pendulum: a, d_1, ..., d_h
    Wi = list(range(K * B, K * B + n))
    out_idx = []
    for i in range(K):
        for t in range(h):          # token t hears d_{h-t}: oldest first
            out_idx += Wi + [B * i + (h - t)]
        out_idx += [B * i]          # the last token also hears the label a
    out_idx = jnp.asarray(out_idx, dtype=int)
    in_idx = jnp.asarray([B * i + k for i in range(K) for k in range(h)], dtype=int)  # d_1 hears a, d_k hears d_{k-1}
    route = Arrangement(
        P=trivial_parameter(), in_dim_M=h * K, out_dim_M=K * B + n, in_dim_N=0, out_dim_N=len(out_idx),
        out_f=lambda q, m_out: m_out[out_idx], in_f=lambda q, m_out, n_in: m_out[in_idx],
        omega=zero_form, label="route",
    )
    preds = tensor_arrangements([predictor(bk) for _ in range(K)])
    return compose_seq(compose_seq(parts, route), preds)


def wired_weights(arr: Arrangement, state, K: int, bk: Block = Block()) -> Array:
    return arr.P.position(state)[(1 + 2 * bk.h) * K:(1 + 2 * bk.h) * K + bk.n]


def wired_presented(arr: Arrangement, state, K: int, bk: Block = Block()) -> Array:
    return arr.P.position(state)[:(1 + 2 * bk.h) * K].reshape(K, 1 + 2 * bk.h)[:, 0]


def wired_windows(arr: Arrangement, state, K: int, bk: Block = Block()) -> Array:
    """Each pendulum's delayed positions, oldest first, ``(K, h)``: what its tokens hear."""
    pos = arr.P.position(state)[:(1 + 2 * bk.h) * K].reshape(K, 1 + 2 * bk.h)
    return pos[:, [1 + 2 * (bk.h - 1 - t) for t in range(bk.h)]]


# ---------------------------------------------------------------------------
# The direct evaluator: the same predictor as a plain causal transformer, beside the arrangement.
# ---------------------------------------------------------------------------


def predict(bk: Block, W: Array, window: Array, freeze_keys: bool = False) -> Array:
    """Two layers of causal multi-head softmax attention on ``x0_t = window_t e + pi_t`` (Vaswani et al.
    §§3.2.1–3.2.3, the scale folded into the query), a residual and an MLP after the first, readout ``U x2 + c`` at
    the last token. ``freeze_keys`` stops the gradient through ``x1`` into layer 2's keys and queries."""
    w = bk.unpack(W)
    mask = jnp.tril(jnp.ones((bk.h, bk.h)))
    X0 = jnp.stack([embed(w, t, window[t]) for t in range(bk.h)])

    def attend(l, X, Xkq):
        outs = []
        for a in range(bk.heads):
            K = jnp.stack([kqv(bk, w, l, a, x)[0] for x in Xkq])
            Q = jnp.stack([kqv(bk, w, l, a, x)[1] for x in Xkq])
            V = jnp.stack([kqv(bk, w, l, a, x)[2] for x in X])
            S = jnp.where(mask > 0, Q @ K.T, -jnp.inf)
            outs.append(jax.nn.softmax(S, axis=1) @ V)
        return outs

    o1 = attend(0, X0, X0)
    X1 = jnp.stack([layer_one(bk, w, X0[t], [o1[a][t] for a in range(bk.heads)]) for t in range(bk.h)])
    o2 = attend(1, X1, jax.lax.stop_gradient(X1) if freeze_keys else X1)
    x2 = X1[-1] + sum(w[f"O1{a}"] @ o2[a][-1] for a in range(bk.heads))
    return (w["U"] @ x2 + w["c"])[0]


def direct_loss(bk: Block, W: Array, windows: Array, labels: Array, freeze_keys: bool = False) -> Array:
    """``sum_i (1/2)(predict(window_i) - label_i)^2`` over the pendulums."""
    preds = jnp.stack([predict(bk, W, windows[i], freeze_keys) for i in range(windows.shape[0])])
    return 0.5 * jnp.sum((preds - labels) ** 2)


def held_out_error(bk: Block, W: Array, *, n_traj: int = 8, T: int = 40, seed: int = 101, m: float = 1.0,
                   q_amp: float = 1.0, p_amp: float = 1.0):
    """Mean one-step error of the trained predictor on held-out runs of the open pendulum (starts ``q0`` uniform in
    ``[-q_amp, q_amp]`` and ``p0`` in ``[-p_amp, p_amp]``, ``seed``; by default ``[-1, 1]^2``, as
    ``tests/test_system_id.py`` computes them for the MLP), and "no change"'s: an ordinary calculation beside the
    arrangement."""
    from .functors import Phi
    from .system_id import oscillator, trajectory

    O = Phi(oscillator(PENDULUM[0], 1, m))
    rng = np.random.default_rng(seed)
    f = jax.jit(lambda win: predict(bk, W, win))
    errs, base = [], []
    for _ in range(n_traj):
        q0, p0 = rng.uniform(-q_amp, q_amp), rng.uniform(-p_amp, p_amp)
        a = [float(x[0] + x[1] / m) for x in trajectory(O, jnp.array([q0]), jnp.array([p0]), T)]
        for t in range(bk.h, len(a)):
            errs.append(abs(float(f(jnp.asarray(a[t - bk.h:t]))) - a[t]))
            base.append(abs(a[t] - a[t - 1]))
    return float(np.mean(errs)), float(np.mean(base))
