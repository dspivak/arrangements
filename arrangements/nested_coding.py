"""Predictive coding nested in the attention operad with signed costs.

Book plan §8.2 P5 (i); predictions, fixed before this file existed, and verdicts in the book's
``notes/runs/p5i-1e43f5fb/RESULTS.md``. The claim under test (plan §5.8): the attention operad of
``attention.py`` with its condition ``k >= 0`` dropped is still an operad mapping to ``arr``, and it
holds predictive coding with learned precisions.

THE MODEL. ``J`` groups of ``K`` samples ``x_jk``; level 1 holds a belief ``mu_j`` per group and one
log-precision ``l1``, level 2 a belief ``nu`` and a log-precision ``l2``. The free energy is

    F = 1/2 e^l1 sum_jk (x_jk - mu_j)^2 - JK/2 l1 + 1/2 e^l2 sum_j (mu_j - nu)^2 - J/2 l2,

unbounded below in the configuration: with every ``mu_j = nu``, ``F -> -inf`` as ``l2 -> inf``.

TWO ORIENTATIONS. A sending map is linear in child messages and a leaf sends ``0_M``, so a level
cannot publish its state as a message on its own; a placement is any smooth map of the
configuration, so a parent can hand its state to a child as the child's position.

* ``data_first``: L1 outer (arity 1), reading the data as its exterior message and placing ``mu``
  in position coordinates ``0..J-1``; L2 inside it (arity 0, or 1 with L3 inside), predicting its
  received position. All transfers are zero; the force on L2's position is pulled back through the
  placement onto ``mu``.
* ``belief_first``: L2 outer, the data passed down to L1 by the identity transfer; L1 has one open
  slot and publishes ``mu`` upward by a transfer whose signal map has ``mu`` as its first column, so
  the slot must be fed the constant message ``(e_0, 1)`` by ``feeder``, a box outside the operad.

``data_box`` closes either composite: a stateless box ``B -> I`` sending the data as the exterior
input. ``laplace_learner`` is the candidate repair, one closed box, outside the operad: ``F`` plus
the curvature term ``1/2 log det`` of the beliefs' precision.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
from jax import Array

from .arrangement import Arrangement, exact
from .attention import (
    AttentionArrangement, Transfer, attention_arrangement, identity_transfer, input_dim, msg_dim,
    split_input, zero_transfer,
)
from .choice import _closed
from .coalgebra import configuration
from .rvect import euclidean


def dims(J: int, K: int) -> Tuple[int, int]:
    """``(dim_H, dim_P)``: the signal carries the ``J K`` samples; positions have ``J + 1``
    coordinates, ``0..J-1`` for ``mu`` and ``J`` for ``nu``."""
    return J * K, J + 1


def _signal(e: Array, dim_H: int) -> Array:
    m, _ = split_input(e, dim_H)
    return m[:dim_H]


def _position(e: Array, dim_H: int) -> Array:
    return split_input(e, dim_H)[1]


def _zeros(n: int, dim_H: int):
    Z = zero_transfer(dim_H)
    return lambda q: tuple(tuple(Z for _ in range(n + 1)) for _ in range(n + 1))


# ---------------------------------------------------------------------------
# data_first: the level that reads the data is the outer arrangement.
# ---------------------------------------------------------------------------


def level1(J: int, K: int, eta: float, init: Array = None) -> AttentionArrangement:
    """L1, arity 1: configuration ``(mu_1..mu_J, l1)``; cost
    ``1/2 e^l1 ||h - mu (x) 1_K||^2 - JK/2 l1`` on the exterior signal ``h``; places ``mu`` in
    coordinates ``0..J-1`` of its child's position."""
    dim_H, dim_P = dims(J, K)

    def cost(q, m_out, n_in):
        r = _signal(n_in, dim_H) - jnp.repeat(q[:J], K)
        return 0.5 * jnp.exp(q[J]) * jnp.sum(r * r) - 0.5 * J * K * q[J]

    q0 = jnp.zeros(J + 1) if init is None else jnp.asarray(init, dtype=float)
    return attention_arrangement(
        1, dim_H, dim_P,
        placement=lambda q: jnp.concatenate([q[:J], jnp.zeros(1)])[None, :],
        transfers=_zeros(1, dim_H), cost=cost,
        parameter=configuration(euclidean(J + 1, eta), init=q0), label="L1",
    )


def level2(J: int, K: int, eta: float, arity: int = 0, init: Array = None) -> AttentionArrangement:
    """L2: configuration ``(nu, l2)``; cost ``1/2 e^l2 sum_j (p_j - nu)^2 - J/2 l2`` on its received
    position ``p``. With ``arity = 1`` it places ``nu`` in coordinate ``J`` of its child's position."""
    dim_H, dim_P = dims(J, K)

    def cost(q, m_out, n_in):
        d = _position(n_in, dim_H)[:J] - q[0]
        return 0.5 * jnp.exp(q[1]) * jnp.sum(d * d) - 0.5 * J * q[1]

    def placement(q):
        return jnp.zeros((0, dim_P)) if arity == 0 else jnp.concatenate([jnp.zeros(J), q[:1]])[None, :]

    q0 = jnp.zeros(2) if init is None else jnp.asarray(init, dtype=float)
    return attention_arrangement(
        arity, dim_H, dim_P, placement=placement, transfers=_zeros(arity, dim_H), cost=cost,
        parameter=configuration(euclidean(2, eta), init=q0), label="L2",
    )


def level3(J: int, K: int, pi0: float, eta0: float) -> AttentionArrangement:
    """L3, arity 0, no configuration: the fixed prior ``1/2 pi0 (p_J - eta0)^2`` on ``nu``."""
    dim_H, dim_P = dims(J, K)

    def cost(q, m_out, n_in):
        return 0.5 * pi0 * (_position(n_in, dim_H)[J] - eta0) ** 2

    return attention_arrangement(0, dim_H, dim_P, placement=lambda q: jnp.zeros((0, dim_P)),
                                 transfers=_zeros(0, dim_H), cost=cost, label="L3")


# ---------------------------------------------------------------------------
# belief_first: the level that predicts is the outer arrangement.
# ---------------------------------------------------------------------------


def level1_publishing(J: int, K: int, eta: float, init: Array = None) -> AttentionArrangement:
    """L1 with one open slot: the same cost on its received signal, and ``T_01(q) = (1, L(q))``
    with ``L(q) e_0 = (mu, 0, ...)``, so a slot fed ``(e_0, 1)`` makes it send ``mu`` upward."""
    dim_H, dim_P = dims(J, K)
    Z = zero_transfer(dim_H)

    def L(q):
        col = jnp.concatenate([q[:J], jnp.zeros(dim_H - J)])
        return jnp.zeros((dim_H, dim_H)).at[:, 0].set(col)

    def transfers(q):
        return ((Z, Transfer(gamma=jnp.asarray(1.0), L=L(q))), (Z, Z))

    def cost(q, m_out, n_in):
        r = _signal(n_in, dim_H) - jnp.repeat(q[:J], K)
        return 0.5 * jnp.exp(q[J]) * jnp.sum(r * r) - 0.5 * J * K * q[J]

    q0 = jnp.zeros(J + 1) if init is None else jnp.asarray(init, dtype=float)
    return attention_arrangement(
        1, dim_H, dim_P, placement=lambda q: jnp.zeros((1, dim_P)), transfers=transfers, cost=cost,
        parameter=configuration(euclidean(J + 1, eta), init=q0), label="L1-publishing",
    )


def level2_reading(J: int, K: int, eta: float, init: Array = None) -> AttentionArrangement:
    """L2 outer, arity 1: passes the exterior message down by the identity transfer; cost
    ``1/2 e^l2 sum_j (h~_j - nu)^2 - J/2 l2`` on its child's message ``h~``."""
    dim_H, dim_P = dims(J, K)
    Z, I = zero_transfer(dim_H), identity_transfer(dim_H)

    def cost(q, m_out, n_in):
        d = m_out[:J] - q[0]
        return 0.5 * jnp.exp(q[1]) * jnp.sum(d * d) - 0.5 * J * q[1]

    q0 = jnp.zeros(2) if init is None else jnp.asarray(init, dtype=float)
    return attention_arrangement(
        1, dim_H, dim_P, placement=lambda q: jnp.zeros((1, dim_P)),
        transfers=lambda q: ((Z, Z), (I, Z)), cost=cost,
        parameter=configuration(euclidean(2, eta), init=q0), label="L2-reading",
    )


def feeder(J: int, K: int) -> Arrangement:
    """A stateless box ``I -> B`` sending the constant message ``(e_0, 1)`` and ignoring its input:
    not an attention arrangement (an arity-0 one sends ``0_M``)."""
    dim_H, _ = dims(J, K)
    m = jnp.zeros(msg_dim(dim_H)).at[0].set(1.0).at[dim_H].set(1.0)
    return Arrangement(
        P=configuration(euclidean(0)),
        in_dim_M=0, out_dim_M=0, in_dim_N=input_dim(*dims(J, K)), out_dim_N=msg_dim(dim_H),
        out_f=lambda q, m_out: m,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: 0.0 * jnp.sum(n_in)),
        label="feed(e0,1)",
    )


# ---------------------------------------------------------------------------
# Closing the system, the free energy written by hand, and the repair.
# ---------------------------------------------------------------------------


def data_box(x: Array, dim_P: int) -> Arrangement:
    """A stateless box ``B -> I``: it sends the input ``((x, 1), 0_P)`` to its inner box and ignores
    the message it receives. ``x`` is ``(J, K)``, sent row by row."""
    x = jnp.ravel(jnp.asarray(x, dtype=float))
    dim_H = x.shape[0]
    e = jnp.concatenate([x, jnp.ones(1), jnp.zeros(dim_P)])
    return Arrangement(
        P=configuration(euclidean(0)),
        in_dim_M=input_dim(dim_H, dim_P), out_dim_M=msg_dim(dim_H), in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: e,
        omega=exact(lambda q, m_out, n_in: 0.0 * jnp.sum(m_out)),
        label="data",
    )


def free_energy(x: Array, mu: Array, l1, nu, l2) -> Array:
    """``F`` of the module docstring, written by hand."""
    x = jnp.asarray(x, dtype=float)
    J, K = x.shape
    return (0.5 * jnp.exp(l1) * jnp.sum((x - mu[:, None]) ** 2) - 0.5 * J * K * l1
            + 0.5 * jnp.exp(l2) * jnp.sum((mu - nu) ** 2) - 0.5 * J * l2)


def laplace_free_energy(x: Array, q: Array) -> Array:
    """``F`` plus ``1/2 log det H``, ``H`` the Hessian of ``F`` in the beliefs ``(mu, nu)``: for this
    Gaussian model, minus the log marginal likelihood of ``(l1, l2)`` up to a constant, at the
    minimizing beliefs. ``q = (mu_1..mu_J, l1, nu, l2)``."""
    x = jnp.asarray(x, dtype=float)
    J, K = x.shape
    mu, l1, nu, l2 = q[:J], q[J], q[J + 1], q[J + 2]
    a, b = K * jnp.exp(l1), jnp.exp(l2)
    # H = [[(a+b) I_J, -b 1], [-b 1^T, J b]]; det = (a+b)^J (J b - J b^2/(a+b)) = (a+b)^(J-1) J a b
    logdet = (J - 1) * jnp.log(a + b) + jnp.log(J * a * b)
    return free_energy(x, mu, l1, nu, l2) + 0.5 * logdet


def laplace_learner(x: Array, eta: float, init: Array = None) -> Arrangement:
    """The repair as one closed box: configuration ``(mu, l1, nu, l2)``, Euclidean response,
    exact forcing ``-d(laplace_free_energy)``."""
    J = jnp.asarray(x).shape[0]
    q0 = jnp.zeros(J + 3) if init is None else jnp.asarray(init, dtype=float)
    return _closed(configuration(euclidean(J + 3, eta), init=q0),
                   lambda q: -laplace_free_energy(x, q), "laplace-learner")


# ---------------------------------------------------------------------------
# The precision sent forward (book P5 (i), piece D; David: "can they send each other something forward to let them
# know about the proper precision in a compositional way?").
# ---------------------------------------------------------------------------
#
# Each lower level sends forward its evidence, not its belief: per group the weighted message
# ``(K Pi1 xbar_j, K Pi1)``, the attention paper's shape (signal over weight is the mean, weight the precision). The
# upper level passes each weight through its own precision, ``w' = w Pi2 / (w + Pi2)`` (variances add, as
# resistances in series; merged evidence adds weights, as conductances in parallel), and scores its prediction ``nu``
# against the message. The per-level costs then sum to minus the log marginal likelihood, up to a constant; the
# upper level's force on each weight, pulled back through L1's readout, is what tells L1 the proper precision.


def samples(x: Array) -> Arrangement:
    """A stateless box ``I -> <0|JK>`` publishing the samples, row by row."""
    x = jnp.ravel(jnp.asarray(x, dtype=float))
    return Arrangement(
        P=configuration(euclidean(0)),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=int(x.shape[0]),
        out_f=lambda q, m_out: x,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: 0.0 * jnp.sum(q)),
        label="samples",
    )


def evidence_sender(J: int, K: int, eta: float, init: float = 0.0) -> Arrangement:
    """L1 as a box ``<0|JK> -> <0|2J>``: state ``l1``; reads the samples, sends ``(K e^l1 xbar_j, K e^l1)`` per
    group (signals first, then weights); cost ``sum_j 1/2 e^l1 sum_k (x_jk - xbar_j)^2 - J(K-1)/2 l1``."""

    def send(q, x):
        X = jnp.reshape(x, (J, K))
        w = K * jnp.exp(q[0])
        return jnp.concatenate([w * X.mean(1), jnp.full(J, w)])

    def U(q, x, n_in):
        X = jnp.reshape(x, (J, K))
        ss = jnp.sum((X - X.mean(1)[:, None]) ** 2)
        return -(0.5 * jnp.exp(q[0]) * ss - 0.5 * J * (K - 1) * q[0])

    return Arrangement(
        P=configuration(euclidean(1, eta), init=jnp.full(1, init)),
        in_dim_M=0, out_dim_M=J * K, in_dim_N=0, out_dim_N=2 * J,
        out_f=send,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(U),
        label="L1-evidence",
    )


def evidence_cost(q: Array, msg: Array, J: int, integrate_nu: bool = True) -> Array:
    """The upper level's cost on the messages ``msg = (h_1..h_J, w_1..w_J)`` at ``q = (nu, l2)``."""
    h, w = msg[:J], msg[J:]
    P2 = jnp.exp(q[1])
    wp = w * P2 / (w + P2)
    c = jnp.sum(0.5 * wp * (h / w - q[0]) ** 2 - 0.5 * jnp.log(wp))
    return c + (0.5 * jnp.log(jnp.sum(wp)) if integrate_nu else 0.0)


def evidence_reader(J: int, eta: float, integrate_nu: bool = True, init: Array = None) -> Arrangement:
    """L2 as a box ``<0|2J> -> I``: state ``(nu, l2)``, cost ``evidence_cost`` on the arriving messages; exact, so
    its force on each message's weight is pulled back onto whatever sent it."""
    q0 = jnp.zeros(2) if init is None else jnp.asarray(init, dtype=float)
    return Arrangement(
        P=configuration(euclidean(2, eta), init=q0),
        in_dim_M=0, out_dim_M=2 * J, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -evidence_cost(q, m_out, J, integrate_nu)),
        label="L2-evidence",
    )


def evidence_hierarchy(x: Array, eta: float, integrate_nu: bool = True, init1: float = 0.0,
                       init2: Array = None) -> Arrangement:
    """``samples -> L1 -> L2``, closed; state ``((empty, l1), (nu, l2))``."""
    from .wiring import compose_seq

    J, K = jnp.asarray(x).shape
    return compose_seq(compose_seq(samples(x), evidence_sender(J, K, eta, init1)),
                       evidence_reader(J, eta, integrate_nu, init2))
