"""Science at two scales, third round: the tests natively in the book's mathematics.

Predictions, fixed before this file existed, in ``notes/SCIENCE-SCALES-PREDICTIONS-2026-10-01.md``
§8 (with the stage-1 numbers at its end); verdicts in §5.7.

NATIVE means: every part is a box with ports, values go forward on wires, the community is the
composite built by the library's wiring (``compose_seq`` of a tensor of boxes with a static wiring
arrangement), and ``Phi`` runs it. No Python decides anything between steps.

THE PIECES.
* ``port_wire(sources, out_dim)``: the static arrangement ``R^-`` of a function from in-port
  coordinates to out-port coordinates (``in_f(m_out) = m_out[sources]``), trivial parameter, zero
  forcing. ``wiring.graph_wire`` is the case of one port dimension throughout; here a world's port
  (one draw) and a holder's port (a whole account) differ, so the general function is used.
* ``draw_source(table)``: the world, a box with no input. State a clock (Euclidean, exact forcing
  ``d(clock)``, so ``+1`` per step); readout the table row at the clock. GAP: that readout is a lookup,
  not a smooth map of the clock, so the world is admissible only because the library never
  differentiates it in a way that matters (every reader's forcing is zero on its in-ports, so no
  cotangent reaches the lookup). The library has no smooth source of a fixed sequence.
* A reader's forcing is the partial differential of its own cost in its own state, with component
  ZERO on its in-ports: composition (``compose_seq``'s pullback) then moves nothing back onto the
  sender. An account is a value sent forward (plan §4.1).

GAP, the readout. A box ``I -> <in | out>`` reads out ``out_f(q)``, a function of its state alone. So a
box cannot publish what arrives in the same step (round two's (f) publishes a holder's stock after its
own draw of that step). The closest native thing is a REGISTER, a pending value in the state that
steps onto whatever arrived (forcing ``x_in - pend``), published one step later.

TEST (c) NATIVE (``community_native``): ``m`` holders, holder ``j`` a box ``I -> <d_j n + 1 | n>``,
state ``y_j in R^n`` (Euclidean), out-port its account ``log_softmax(y_j)``, in-ports its neighbours'
accounts and its own draw; cost as round two's ``account_community`` per holder.

TEST (f) SUMMARIES NATIVE (``summaries_native``). Each holder's state is ``(y, NS, pend)`` or
``(y, pend)``; ``"noprov"``: out-port ``NS + pend`` (its stock), forcing ``(stats(add), add,
x_in - pend)`` with ``add = pend + sum of in-ports``; ``"lag"``: no register, out-port ``NS``, ``add =
x_in + sum of in-ports``; ``"journal"``: out-port ``pend`` (its registered draw), a journal box adds what
arrives, a holder's ``y`` steps onto the journal's log-likelihood (cost ``1/2 |y - stats(J)|^2``).

TEST (f) CONCLUSIONS NATIVE (``conclusions_native``, stage 2). Each holder is a simplex state ``p`` with
exact zeros outside its slice, out-port its sharpened conclusion ``p^{1/tau}/sum``. Under the Shahshahani
response (``frame="shah"``, as designed) the seed ``eps sum c log p`` has an infinite differential on the
face ``p_h = 0`` and the step is NaN: no finite forcing plants support under a response that keeps every
fitness's faces invariant. ``frame="euc"`` (the disclosed second run): Euclidean response, the replicator
field ``p * (g - <g, p>)`` as a NON-EXACT data forcing, and the exact seed ``d(eps(<c̄, p> - |p|^2/2))``.
"""

from __future__ import annotations

from functools import partial
from typing import List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .arrangement import Arrangement, zero_form
from .coalgebra import configuration
from .functors import Phi
from .active_inference import _IN_DIR, _IN_POS
from .replicator import shahshahani
from .rvect import euclidean
from .wiring import compose_seq, tensor_arrangements, trivial_parameter


LOG2PI = float(np.log(2.0 * np.pi))


# --- wiring ------------------------------------------------------------------


def port_wire(sources: Sequence[int], out_dim: int) -> Arrangement:
    """``R^-`` of the function ``i |-> sources[i]`` from in-port to out-port coordinates."""
    src = jnp.asarray(sources, dtype=int)
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=len(sources), out_dim_M=out_dim, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: m_out[src],
        omega=zero_form,
        label=f"port_wire({len(sources)} <- {out_dim})",
    )


def wire_boxes(boxes: Sequence[Arrangement], feeds: Sequence[Sequence[Tuple[int, int]]]) -> Arrangement:
    """The closed composite of ``boxes`` (each ``I -> <in_b | out_b>``), in-port coordinate ``c`` of box
    ``b`` fed by ``feeds[b][c] = (source box, source out coordinate)``."""
    offs = np.cumsum([0] + [b.out_dim_N for b in boxes])
    sources = []
    for b, box in enumerate(boxes):
        if box.in_dim_M or box.out_dim_M or len(feeds[b]) != box.in_dim_N:
            raise ValueError(f"box {b}: bad interface or feed")
        sources += [int(offs[k] + c) for (k, c) in feeds[b]]
    return compose_seq(tensor_arrangements(list(boxes)), port_wire(sources, int(offs[-1])))


def _reader(dim: int, init, in_dim: int, out_dim: int, readout, force, label: str) -> Arrangement:
    """A box ``I -> <in_dim | out_dim>`` with Euclidean response 1 whose forcing is ``force(q, n_in)``
    on its state and zero on its in-ports (nothing pushed back)."""
    return Arrangement(
        P=configuration(euclidean(dim, 1.0), init=init),
        in_dim_M=0, out_dim_M=0, in_dim_N=in_dim, out_dim_N=out_dim,
        out_f=lambda q, m_out: readout(q),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=lambda q, m_out, n_in: (force(q, n_in), jnp.zeros_like(m_out), jnp.zeros_like(n_in)),
        label=label,
    )


def draw_source(table: Array, hold: int = 1) -> Arrangement:
    """The world: clock state, exact forcing ``d(clock)``, readout ``table[clock // hold]`` (clipped at
    the end), so each row is emitted for ``hold`` steps."""
    table = jnp.asarray(table, dtype=float)
    last = table.shape[0] - 1
    return _reader(1, jnp.zeros(1), 0, table.shape[1],
                   lambda q: table[jnp.minimum(jax.lax.stop_gradient(q[0]).astype(int) // hold, last)],
                   lambda q, n_in: jnp.ones(1), "world")


def run_positions(arr: Arrangement, steps: int) -> Array:
    """Positions of the closed ``Phi(arr)`` at steps ``0..steps`` (a ``lax.scan``; call under ``jit``)."""
    O = Phi(arr)

    def body(s, _):
        _, _, s2 = O.with_state(s).run_one(_IN_POS, lambda _o: _IN_DIR)
        return s2, arr.P.position(s2)

    _, traj = jax.lax.scan(body, arr.P.init, None, length=steps)
    return jnp.concatenate([arr.P.position(arr.P.init)[None], traj])


def neighbours(W: np.ndarray) -> List[List[int]]:
    W = np.asarray(W)
    return [list(np.nonzero(W[j])[0]) for j in range(W.shape[0])]


# --- test (c) native -----------------------------------------------------------


def community_native(Y0: Array, X: Array, W: np.ndarray, alpha, beta, mus: Array, arm: str) -> Arrangement:
    """World plus ``m`` holder boxes, wired by ``W`` (``W[j, k] > 0``: ``j`` reads ``k``'s account at
    weight ``W[j, k]``). ``X[t, j]`` is holder ``j``'s draw at step ``t``."""
    m, n = Y0.shape
    mus = jnp.asarray(mus, dtype=float)
    nb = neighbours(W)
    boxes, feeds = [draw_source(X)], [[]]
    for j in range(m):
        d = len(nb[j])
        wj = jnp.asarray(np.asarray(W)[j, nb[j]], dtype=float)

        def cost(y, n_in, d=d, wj=wj):
            acc = n_in[: d * n].reshape(d, n)
            ll = -0.5 * (n_in[d * n] - mus) ** 2 - 0.5 * LOG2PI
            if arm == "agree":
                c = 0.5 * alpha * jnp.sum(wj[:, None] * (y[None, :] - acc) ** 2)
            elif arm == "evidence":
                c = -alpha * jnp.dot(wj @ acc, y)
            else:
                raise ValueError(arm)
            return c - beta * jnp.dot(ll, y)

        dcost = jax.grad(cost, argnums=0)
        boxes.append(_reader(n, Y0[j], d * n + 1, n, jax.nn.log_softmax,
                             lambda y, n_in, dc=dcost: -dc(y, n_in), f"holder {j}"))
        feeds.append([(k + 1, c) for k in nb[j] for c in range(n)] + [(0, j)])
    return wire_boxes(boxes, feeds)


@partial(jax.jit, static_argnames=("W_key", "alpha", "beta", "arm", "steps"))
def _community_native_run(Y0, X, mus, W_key, alpha, beta, arm, steps):
    W = np.asarray(W_key)
    m, n = Y0.shape
    arr = community_native(Y0, X, W, alpha, beta, mus, arm)
    return run_positions(arr, steps)[:, 1:].reshape(steps + 1, m, n)


def community_native_run(Y0, X, W, alpha, beta, mus, arm, steps):
    """``Y`` at steps ``0..steps``, shape ``(steps + 1, m, n)``."""
    W_key = tuple(tuple(float(v) for v in row) for row in np.asarray(W))
    return _community_native_run(jnp.asarray(Y0, float), jnp.asarray(X, float), jnp.asarray(mus, float),
                                 W_key, float(alpha), float(beta), arm, int(steps))


# --- test (f) summaries native ---------------------------------------------------


def _stats(mus, N, S):
    return mus * S - N * mus ** 2 / 2.0


def summaries_native(draws: Array, A: np.ndarray, mus: Array, arm: str, steps: int) -> Arrangement:
    """World (emitting ``(1, x_{t,j})`` per holder while ``t < T``, then ``(0, 0)``) plus holders, and in
    the ``"journal"`` arm a journal box; ``A`` the 0/1 adjacency (``j`` reads ``k`` iff ``A[j, k]``)."""
    T, m = draws.shape
    n = mus.shape[0]
    rows = jnp.stack([jnp.ones((T, m)), draws], axis=-1).reshape(T, 2 * m)
    table = jnp.concatenate([rows, jnp.zeros((steps + 1, 2 * m))])
    nb = neighbours(A)
    boxes, feeds = [draw_source(table)], [[]]
    for j in range(m):
        d = len(nb[j])
        world_in = [(0, 2 * j), (0, 2 * j + 1)]
        if arm == "noprov":                     # state (y, NS, pend); out NS + pend
            def force(q, n_in, d=d):
                pend, x_in = q[n + 2:], n_in[2 * d:]
                add = pend + n_in[: 2 * d].reshape(d, 2).sum(0)
                return jnp.concatenate([_stats(mus, add[0], add[1]), add, x_in - pend])
            boxes.append(_reader(n + 4, jnp.zeros(n + 4), 2 * d + 2, 2,
                                 lambda q: q[n:n + 2] + q[n + 2:], force, f"holder {j}"))
            feeds.append([(k + 1, c) for k in nb[j] for c in (0, 1)] + world_in)
        elif arm == "lag":                      # state (y, NS); out NS; no register
            def force(q, n_in, d=d):
                add = n_in[2 * d:] + n_in[: 2 * d].reshape(d, 2).sum(0)
                return jnp.concatenate([_stats(mus, add[0], add[1]), add])
            boxes.append(_reader(n + 2, jnp.zeros(n + 2), 2 * d + 2, 2,
                                 lambda q: q[n:], force, f"holder {j}"))
            feeds.append([(k + 1, c) for k in nb[j] for c in (0, 1)] + world_in)
        elif arm == "journal":                  # state (y, pend); out pend; reads the journal
            def force(q, n_in):
                J, x_in = n_in[:2], n_in[2:]
                return jnp.concatenate([_stats(mus, J[0], J[1]) - q[:n], x_in - q[n:]])
            boxes.append(_reader(n + 2, jnp.zeros(n + 2), 4, 2, lambda q: q[n:], force, f"holder {j}"))
            feeds.append([(m + 1, 0), (m + 1, 1)] + world_in)
        else:
            raise ValueError(arm)
    if arm == "journal":
        boxes.append(_reader(2, jnp.zeros(2), 2 * m, 2, lambda q: q,
                             lambda q, n_in: n_in.reshape(m, 2).sum(0), "journal"))
        feeds.append([(k + 1, c) for k in range(m) for c in (0, 1)])
    return wire_boxes(boxes, feeds)


@partial(jax.jit, static_argnames=("A_key", "arm", "steps"))
def _summaries_native_run(draws, mus, A_key, arm, steps):
    A = np.asarray(A_key)
    arr = summaries_native(draws, A, mus, arm, steps)
    return run_positions(arr, steps)


def summaries_native_run(draws, A, mus, arm, steps):
    """Holders' beliefs ``y`` at steps ``0..steps``, shape ``(steps + 1, m, n)``, and the journal's
    ``(N, S)`` (``None`` outside the journal arm)."""
    draws = jnp.asarray(draws, float)
    mus = jnp.asarray(mus, float)
    T, m = draws.shape
    n = mus.shape[0]
    A_key = tuple(tuple(int(v) for v in row) for row in np.asarray(A))
    pos = np.asarray(_summaries_native_run(draws, mus, A_key, arm, int(steps)))
    width = {"noprov": n + 4, "lag": n + 2, "journal": n + 2}[arm]
    H = pos[:, 1:1 + m * width].reshape(steps + 1, m, width)
    J = pos[:, 1 + m * width:] if arm == "journal" else None
    return H[:, :, :n], J


# --- test (f) conclusions native (stage 2) -----------------------------------------


def conclusion_holder(p0: Array, wj: Array, mus: Array, dt: float, arm: str, frame: str,
                      eps: float, alpha: float, tau_inv: int, label: str = "holder") -> Arrangement:
    """A holder ``I -> <d n + 1 | n>`` on the simplex: state ``p``, out-port its conclusion
    ``c = p^{1/tau} / sum`` (``tau_inv = 1/tau``), in-ports its ``d`` neighbours' conclusions and its draw.
    ``arm``: ``"seed"`` (``eps sum c̄ log p``), ``"evidence"`` (fitness bonus ``alpha c̄``) or ``"both"``.
    ``frame="shah"``: ``choice.py``'s chooser, Shahshahani response ``dt``, forcing the differential of
    ``<f + alpha c̄, p> + eps sum c̄ log p``. ``frame="euc"``: Euclidean response ``dt``, data forcing the
    replicator field ``p * (g - <g, p>)``, ``g = f + alpha c̄`` (not exact), seed the differential of
    ``eps (<c̄, p> - |p|^2 / 2)``. Every forcing is zero on the in-ports."""
    n = p0.shape[0]
    d = wj.shape[0]
    seed = arm in ("seed", "both")
    evid = arm in ("evidence", "both")

    def inputs(n_in):
        cbar = wj @ n_in[: d * n].reshape(d, n)
        ll = -0.5 * (n_in[d * n] - mus) ** 2
        f = ll - jnp.min(ll) + 1.0
        return cbar, f + (alpha * cbar if evid else 0.0)

    if frame == "shah":
        def U(p, n_in):
            cbar, g = inputs(n_in)
            return jnp.dot(g, p) + (eps * jnp.sum(cbar * jnp.log(p)) if seed else 0.0)
        dU = jax.grad(U, argnums=0)
        force, Q = dU, shahshahani(n, dt)
    elif frame == "euc":
        def V_seed(p, n_in):
            cbar, _ = inputs(n_in)
            return eps * (jnp.dot(cbar, p) - 0.5 * jnp.dot(p, p))
        dV = jax.grad(V_seed, argnums=0)

        def force(p, n_in):
            _, g = inputs(n_in)
            xi = p * (g - jnp.dot(g, p))
            return xi + dV(p, n_in) if seed else xi
        Q = euclidean(n, dt)
    else:
        raise ValueError(frame)

    def readout(p):
        r = p ** tau_inv
        return r / jnp.sum(r)

    return Arrangement(
        P=configuration(Q, init=p0),
        in_dim_M=0, out_dim_M=0, in_dim_N=d * n + 1, out_dim_N=n,
        out_f=lambda q, m_out: readout(q),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=lambda q, m_out, n_in: (force(q, n_in), jnp.zeros_like(m_out), jnp.zeros_like(n_in)),
        label=label,
    )


def conclusions_native(draws: Array, A: np.ndarray, mus: Array, slices, dt: float, hold: int, arm: str,
                       frame: str, eps: float, alpha: float, tau_inv: int) -> Arrangement:
    """World (each draw held ``hold`` steps) plus ``m`` conclusion holders wired by adjacency ``A``,
    neighbours averaged."""
    T, m = draws.shape
    n = mus.shape[0]
    nb = neighbours(A)
    boxes, feeds = [draw_source(draws, hold)], [[]]
    for j in range(m):
        p0 = np.zeros(n); p0[list(slices[j])] = 1.0 / len(slices[j])
        wj = jnp.full(len(nb[j]), 1.0 / len(nb[j]))
        boxes.append(conclusion_holder(jnp.asarray(p0), wj, mus, dt, arm, frame, eps, alpha, tau_inv,
                                       f"holder {j}"))
        feeds.append([(k + 1, c) for k in nb[j] for c in range(n)] + [(0, j)])
    return wire_boxes(boxes, feeds)


@partial(jax.jit, static_argnames=("A_key", "slices", "hold", "arm", "frame", "eps", "alpha", "tau_inv"))
def _conclusions_run(draws, mus, dt, A_key, slices, hold, arm, frame, eps, alpha, tau_inv):
    A = np.asarray(A_key)
    T, m = draws.shape
    n = mus.shape[0]
    arr = conclusions_native(draws, A, mus, slices, dt, hold, arm, frame, eps, alpha, tau_inv)
    O = Phi(arr)

    def one(carry, _):
        s, lo, dev = carry
        _, _, s2 = O.with_state(s).run_one(_IN_POS, lambda _o: _IN_DIR)
        P = arr.P.position(s2)[1:].reshape(m, n)
        return (s2, jnp.minimum(lo, jnp.min(P)), jnp.maximum(dev, jnp.max(jnp.abs(P.sum(1) - 1.0)))), None

    def per_draw(carry, _):
        carry, _ = jax.lax.scan(one, carry, None, length=hold)
        return carry, arr.P.position(carry[0])[1:].reshape(m, n)

    (_, lo, dev), traj = jax.lax.scan(per_draw, (arr.P.init, jnp.inf, 0.0), None, length=T)
    return traj, lo, dev


def conclusions_native_run(draws, A, mus, slices, dt, hold, arm, frame, eps, alpha, tau):
    """Over a batch of seeds (``draws`` of shape ``(seeds, T, m)``): each holder's weights after each draw
    ``(seeds, T, m, n)``, the minimum weight and ``max |sum p - 1|`` over every step of ``Phi``."""
    A_key = tuple(tuple(int(v) for v in row) for row in np.asarray(A))
    sl = tuple(tuple(int(i) for i in s) for s in slices)
    f = lambda dr: _conclusions_run(dr, jnp.asarray(mus, float), dt, A_key, sl, int(hold), arm, frame,
                                    float(eps), float(alpha), int(round(1.0 / tau)))
    traj, lo, dev = jax.vmap(f)(jnp.asarray(draws, float))
    return np.asarray(traj), np.asarray(lo), np.asarray(dev)


# --- test (g): explaining away by building excuses from figures ------------------------


def excuse_world(T: int, K: int, y: Array, ruled: Array, hold: int) -> Arrangement:
    """The world for (g): row ``r`` (emitted while incident ``r`` is current, ``hold`` steps each) is
    ``(seen (T), y (T), ruled-out mask (T*K))``, ``seen_i = 1`` for ``i <= r``."""
    seen = jnp.tril(jnp.ones((T, T)))
    rows = jnp.concatenate([seen, jnp.tile(jnp.asarray(y, float), (T, 1)),
                            jnp.tile(jnp.asarray(ruled, float).ravel(), (T, 1))], axis=1)
    return draw_source(rows, hold)


def excuse_reader(T: int, K: int, delta: Array, D: float, theta0: float, w: float, v: float,
                  kappa: float, eta: float, dt: float, a0: float) -> Arrangement:
    """The reader ``I -> <2T + TK | 1>``: state ``(theta, A)``, ``A[i, k]`` figure ``k``'s attachment for
    incident ``i``; response ``diag(eta, dt a (1 - a))`` (Euclidean on ``theta``, ``choice.py``'s
    ``{attached, not}`` chooser on each attachment); exact forcing ``-d_state`` of the cost (the note's §9
    numbers), zero on the in-ports; out-port ``theta``."""
    from .rvect import ReactiveVectorSpace
    delta = jnp.asarray(delta, float)
    Q = ReactiveVectorSpace(dim=1 + T * K, sharp_fn=lambda q: jnp.diag(
        jnp.concatenate([jnp.full(1, eta), dt * q[1:] * (1.0 - q[1:])])))

    def cost(q, n_in):
        th, A = q[0], q[1:].reshape(T, K)
        seen, y, R = n_in[:T], n_in[T:2 * T], n_in[2 * T:].reshape(T, K)
        r = y - th + D * A.sum(1)
        per = 0.5 * r ** 2 + 0.5 * v * jnp.sum(R * A ** 2, 1) + A @ delta + kappa * jnp.sum(A * (1 - A), 1)
        return 0.5 * w * (th - theta0) ** 2 + jnp.sum(seen * per)

    dcost = jax.grad(cost, argnums=0)
    return Arrangement(
        P=configuration(Q, init=jnp.concatenate([jnp.full(1, theta0), jnp.full(T * K, a0)])),
        in_dim_M=0, out_dim_M=0, in_dim_N=2 * T + T * K, out_dim_N=1,
        out_f=lambda q, m_out: q[:1],
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=lambda q, m_out, n_in: (-dcost(q, n_in), jnp.zeros_like(m_out), jnp.zeros_like(n_in)),
        label="excuse reader",
    )


@partial(jax.jit, static_argnames=("T", "K", "hold", "kappa"))
def excuses_native_run(T, K, delta, y, ruled, hold, D, theta0, w, v, kappa, eta, dt, a0):
    """World + reader wired and run by ``Phi``; ``(theta, A)`` at the end of each incident."""
    arr = wire_boxes([excuse_world(T, K, y, ruled, hold),
                      excuse_reader(T, K, delta, D, theta0, w, v, kappa, eta, dt, a0)],
                     [[], [(0, c) for c in range(2 * T + T * K)]])
    O = Phi(arr)

    def one(s, _):
        _, _, s2 = O.with_state(s).run_one(_IN_POS, lambda _o: _IN_DIR)
        return s2, None

    def per_incident(s, _):
        s, _ = jax.lax.scan(one, s, None, length=hold)
        return s, arr.P.position(s)[1:]

    _, traj = jax.lax.scan(per_incident, arr.P.init, None, length=T)
    return traj[:, 0], traj[:, 1:].reshape(T, T, K)


__all__ = ["excuse_world", "excuse_reader", "excuses_native_run", "conclusion_holder", "conclusions_native", "conclusions_native_run", "port_wire", "wire_boxes", "draw_source", "run_positions", "neighbours",
           "community_native", "community_native_run", "summaries_native", "summaries_native_run"]
