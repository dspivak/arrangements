"""Science at two scales, test (h): bettors paid by being right, natively in the book's mathematics.

Predictions, fixed before this file existed, in ``notes/SCIENCE-PAYMENT-PREDICTIONS-2026-10-01.md`` §2
(with "Numbers fixed before code" at its end); results in its §6. Quoted:

  (h1) *Wealth is the likelihood.* With proportional betting and no upkeep, a bettor's supply after ``t``
  races is its starting supply times ``n^t`` times the likelihood its belief gave the outcomes; supply
  shares across silent dogmatic bettors evolve exactly as test (a)'s population, to round-off.

  (h2) *Dogmatic bettors die with their idea.* With upkeep ``c > 0``, every dogmatic bettor whose belief
  puts less on the favourite than the expected-growth threshold dies (closed form ... the run derives the
  exact one before code); count each way over 20 seeds.

  (h3) *Updating bettors let the idea die instead.* Same settings: far fewer deaths than the dogmatic
  arm; deaths only among bettors whose early losses exhaust supply before their belief moves (a race
  between belief speed and supply). Counts each way.

  (h4) *The echo chamber falls apart together.* With talk taken as evidence at ten times the weight of
  reading outcomes (test (c)'s worst setting), in the seeds where the community locks onto a wrong horse,
  all bettors die (correlated wrongness is correlated death); in the agree rule, total collapse 0 of 20.

  (h5) *Who hears whom shows with a hub.* Agree rule, no outcomes read: the consensus is the
  centrality-weighted average of the starts, with the hub's weight its eigenvector centrality ..., unlike
  ring and all-to-all where every weight is equal. Hub wrong, with outcomes and paid: more seeds lock
  wrong and collapse than on the ring at the same settings.

NATIVE. A world box (``science_native.draw_source``: a clock whose readout is the race's one-hot outcome)
and ``m`` bettor boxes, composed by ``science_native.wire_boxes`` (a tensor and a static ``port_wire``)
and run by ``Phi`` under one ``lax.scan``; nothing is decided in Python between steps. Every bettor's
in-ports are all ``m`` accounts (its own included) and the outcome; who hears whom is the weight row
``W[j]`` (zero for a bettor it does not hear), so one compiled composite serves every network and arm.

A BETTOR is a box ``I -> <m K + n | K>``: state ``(y, s)`` (log-weights over the ``K`` hypotheses, and
its supply), Euclidean response 1, out-port its account ``log_softmax(y)``. Its forcing, zero on the
in-ports (an account is a value sent forward; nothing is pushed back onto a speaker):
* on ``y``: ``beta log H(x)`` (test (a)'s update) ``+ alpha_agree sum_k W_jk (log q_k - y)``
  ``+ alpha_evid sum_k W_jk log q_k`` (test (c)'s two talk rules), each the negative differential of
  test (c)'s cost in ``y``, written out rather than autodiffed so that a diverging agree step (``|y|`` to
  ``1e256``) does not overflow a squared cost;
* on ``s``: ``max(s n p(x) - c, 0) - s`` with ``p = softmax(y) H`` the predictive held before the race,
  so a step gives ``s' = max(s n p(x) - c, 0)``: proportional betting at fair odds ``n``, then upkeep.

NON-NATIVE PIECE (disclosed). The floor at zero supply is a ``max`` inside the supply forcing, so the
one-form has a kink (continuous, not smooth) where ``s n p(x) = c``. It is computed by ``Phi``, not decided
between steps, but the library's arrangements are otherwise smooth. Item 53's starving part needs no
floor because its drive is multiplicative in its own amplitude (zero is a fixed point, death a
threshold); a fixed upkeep has no such smooth fixed point. A dead bettor's belief still moves and its
account is still published; only its supply stops.
"""

from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .arrangement import Arrangement
from .coalgebra import configuration
from .rvect import euclidean
from .science_native import draw_source, run_positions, wire_boxes

N_HORSES = 4
FAV = 0.5


def hypotheses(n: int = N_HORSES, a: float = FAV) -> np.ndarray:
    """``H[i, x]``: hypothesis ``i`` says horse ``i`` wins with probability ``a``, each other ``(1-a)/(n-1)``."""
    H = np.full((n, n), (1.0 - a) / (n - 1))
    np.fill_diagonal(H, a)
    return H


def ring(m: int) -> np.ndarray:
    W = np.zeros((m, m))
    for j in range(m):
        W[j, (j - 1) % m] = W[j, (j + 1) % m] = 0.5
    return W


def all_to_all(m: int) -> np.ndarray:
    return (np.ones((m, m)) - np.eye(m)) / (m - 1)


def hub(m: int) -> np.ndarray:
    """Bettor 0 hears the others at ``1/(m-1)``; bettor ``j >= 1`` hears its two ring neighbours among
    ``1..m-1`` and the hub, ``1/3`` each."""
    W = np.zeros((m, m))
    W[0, 1:] = 1.0 / (m - 1)
    r = m - 1
    for i in range(r):
        j = i + 1
        W[j, 0] = W[j, 1 + (i - 1) % r] = W[j, 1 + (i + 1) % r] = 1.0 / 3.0
    return W


def bettor(y0, s0: float, wj, H, logH, n_bettors: int, alpha_agree, alpha_evid, beta, c,
           label: str = "bettor") -> Arrangement:
    """A bettor box ``I -> <m K + n | K>`` (module docstring)."""
    K, n = H.shape
    m = n_bettors

    def force(q, n_in):
        y, s = q[:K], q[K]
        acc = n_in[: m * K].reshape(m, K)
        onehot = n_in[m * K:]
        heard = wj @ acc
        xi_y = beta * (logH @ onehot) + alpha_agree * (heard - jnp.sum(wj) * y) + alpha_evid * heard
        p_x = jax.nn.softmax(y) @ (H @ onehot)
        xi_s = jnp.maximum(s * n * p_x - c, 0.0) - s
        return jnp.concatenate([xi_y, xi_s[None]])

    return Arrangement(
        P=configuration(euclidean(K + 1, 1.0), init=jnp.concatenate([y0, jnp.full(1, s0)])),
        in_dim_M=0, out_dim_M=0, in_dim_N=m * K + n, out_dim_N=K,
        out_f=lambda q, m_out: jax.nn.log_softmax(q[:K]),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=lambda q, m_out, n_in: (force(q, n_in), jnp.zeros_like(m_out), jnp.zeros_like(n_in)),
        label=label,
    )


def market(Y0, outcomes_onehot, W, alpha_agree, alpha_evid, beta, c, s0=1.0) -> Arrangement:
    """The closed composite: world plus ``m`` bettors, every bettor wired to every account and the race."""
    m, K = Y0.shape
    n = outcomes_onehot.shape[1]
    H = jnp.asarray(hypotheses(n), float)
    logH = jnp.log(H)
    boxes, feeds = [draw_source(outcomes_onehot)], [[]]
    for j in range(m):
        boxes.append(bettor(Y0[j], s0, W[j], H, logH, m, alpha_agree, alpha_evid, beta, c, f"bettor {j}"))
        feeds.append([(k + 1, i) for k in range(m) for i in range(K)] + [(0, x) for x in range(n)])
    return wire_boxes(boxes, feeds)


@partial(jax.jit, static_argnames=("steps",))
def _market_run(Y0, X1h, W, alpha_agree, alpha_evid, beta, c, s0, steps):
    m, K = Y0.shape
    arr = market(Y0, X1h, W, alpha_agree, alpha_evid, beta, c, s0)
    pos = run_positions(arr, steps)[:, 1:].reshape(steps + 1, m, K + 1)
    return pos[:, :, :K], pos[:, :, K]


def market_run(Y0s, Xs, W, alpha_agree=0.0, alpha_evid=0.0, beta=0.0, c=0.0, s0=1.0,
               steps=None) -> Tuple[np.ndarray, np.ndarray]:
    """Over seeds (``Y0s (S, m, K)``, outcome indices ``Xs (S, T)``): ``Y (S, steps+1, m, K)`` and supplies
    ``S (S, steps+1, m)``, by ``Phi`` on the wired composite (vmapped over seeds)."""
    Xs = np.asarray(Xs)
    T = Xs.shape[1]
    steps = T if steps is None else steps
    X1h = jax.nn.one_hot(jnp.asarray(Xs), N_HORSES, dtype=float)
    f = lambda y0, x1h: _market_run(y0, x1h, jnp.asarray(W, float), float(alpha_agree), float(alpha_evid),
                                    float(beta), float(c), float(s0), int(steps))
    Y, S = jax.vmap(f)(jnp.asarray(Y0s, float), X1h)
    return np.asarray(Y), np.asarray(S)


def seeds(n_seeds: int = 20, m: int = 10, T: int = 200, hub_wrong: bool = False):
    """The fixed seeds: ``Y0 = rng.normal(size=(m, 4))``, then ``rng.choice(4, T, p=H_0)``."""
    pi = hypotheses()[0]
    Y0s, Xs = [], []
    for s in range(n_seeds):
        rng = np.random.default_rng(s)
        Y0 = rng.normal(size=(m, N_HORSES))
        X = rng.choice(N_HORSES, size=T, p=pi)
        if hub_wrong:
            Y0[0] = np.array([0.0, 10.0, 0.0, 0.0])
        Y0s.append(Y0)
        Xs.append(X)
    return np.stack(Y0s), np.stack(Xs)


def locked_wrong(Y_final: np.ndarray, truth: int = 0) -> bool:
    """Every bettor above 0.99 on one and the same wrong hypothesis."""
    q = np.asarray(jax.nn.softmax(jnp.asarray(Y_final), axis=-1))
    top = q.argmax(-1)
    return bool(np.all(top == top[0]) and top[0] != truth and np.all(q.max(-1) > 0.99))


# --- §7 correction: the dead fall apart -------------------------------------------------------------


def gate(s, s1: float):
    """Smooth step: 0 for ``s <= 0``, 1 for ``s >= s1``, ``C^infinity``, all derivatives 0 at 0."""
    x = s / s1
    a = jnp.where(x > 0, jnp.exp(-1.0 / jnp.where(x > 0, x, 1.0)), 0.0)
    b = jnp.where(x < 1, jnp.exp(-1.0 / jnp.where(x < 1, 1.0 - x, 1.0)), 0.0)
    return a / (a + b)


def bettor_gated(y0, s0, wj, H, logH, n_bettors: int, alpha_agree, alpha_evid, beta, c, r, s1,
                 label: str = "bettor") -> Arrangement:
    """A §7 bettor ``I -> <m (K+1) + n | K+1>``: out-port ``(g(s) log q, g(s))``; its reading and talk gated
    by its own ``g``, its belief dissolving toward uniform at ``(1 - g) r``; listeners average over the
    living (the note's §7 "Numbers fixed before code")."""
    K, n = H.shape
    m = n_bettors

    def force(q, n_in):
        y, s = q[:K], q[K]
        acc = n_in[: m * (K + 1)].reshape(m, K + 1)
        onehot = n_in[m * (K + 1):]
        num, den = wj @ acc[:, :K], wj @ acc[:, K]
        alive = den > 0
        avg = jnp.where(alive, num / jnp.where(alive, den, 1.0), 0.0)
        talk = alpha_agree * jnp.where(alive, avg - y, 0.0) + alpha_evid * avg
        gj = gate(s, s1)
        xi_y = gj * (beta * (logH @ onehot) + talk) - (1.0 - gj) * r * (y - jnp.mean(y))
        p_x = jax.nn.softmax(y) @ (H @ onehot)
        xi_s = jnp.maximum(s * n * p_x - c, 0.0) - s
        return jnp.concatenate([xi_y, xi_s[None]])

    def readout(q):
        gj = gate(q[K], s1)
        return jnp.concatenate([gj * jax.nn.log_softmax(q[:K]), gj[None]])

    return Arrangement(
        P=configuration(euclidean(K + 1, 1.0), init=jnp.concatenate([y0, jnp.reshape(s0, (1,))])),
        in_dim_M=0, out_dim_M=0, in_dim_N=m * (K + 1) + n, out_dim_N=K + 1,
        out_f=lambda q, m_out: readout(q),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=lambda q, m_out, n_in: (force(q, n_in), jnp.zeros_like(m_out), jnp.zeros_like(n_in)),
        label=label,
    )


def market_gated(Y0, S0, outcomes_onehot, W, alpha_agree, alpha_evid, beta, c, r, s1) -> Arrangement:
    """World plus ``m`` §7 bettors (starting supplies ``S0``, one per bettor), every bettor wired to every
    gated account and the race."""
    m, K = Y0.shape
    n = outcomes_onehot.shape[1]
    H = jnp.asarray(hypotheses(n), float)
    boxes, feeds = [draw_source(outcomes_onehot)], [[]]
    for j in range(m):
        boxes.append(bettor_gated(Y0[j], S0[j], W[j], H, jnp.log(H), m, alpha_agree, alpha_evid, beta, c, r, s1,
                                  f"bettor {j}"))
        feeds.append([(k + 1, i) for k in range(m) for i in range(K + 1)] + [(0, x) for x in range(n)])
    return wire_boxes(boxes, feeds)


@partial(jax.jit, static_argnames=("steps",))
def _market_gated_run(Y0, X1h, W, alpha_agree, alpha_evid, beta, c, r, s1, s0, steps):
    m, K = Y0.shape
    arr = market_gated(Y0, jnp.full(m, s0), X1h, W, alpha_agree, alpha_evid, beta, c, r, s1)
    pos = run_positions(arr, steps)[:, 1:].reshape(steps + 1, m, K + 1)
    return pos[:, :, :K], pos[:, :, K]


def market_gated_run(Y0s, Xs, W, alpha_agree=0.0, alpha_evid=0.0, beta=0.0, c=0.0, r=0.1, s1=0.01, s0=1.0):
    """``market_run`` for the §7 bettors."""
    Xs = np.asarray(Xs)
    X1h = jax.nn.one_hot(jnp.asarray(Xs), N_HORSES, dtype=float)
    f = lambda y0, x1h: _market_gated_run(y0, x1h, jnp.asarray(W, float), float(alpha_agree), float(alpha_evid),
                                          float(beta), float(c), float(r), float(s1), float(s0), Xs.shape[1])
    Y, S = jax.vmap(f)(jnp.asarray(Y0s, float), X1h)
    return np.asarray(Y), np.asarray(S)


def listener_sensitivity(Y, S, x1h, W, alpha_agree, alpha_evid, beta, c, r, s1):
    """Jacobian of the composite's forcing in the composite's state at bettor states ``(Y, S)``, as an
    ``(m, K, m, K+1)`` array: ``[j, :, k, :]`` is how bettor ``j``'s belief forcing moves with bettor ``k``'s
    state."""
    m, K = Y.shape
    arr = market_gated(jnp.asarray(Y), jnp.asarray(S), x1h, jnp.asarray(W, float), alpha_agree, alpha_evid, beta,
                       c, r, s1)
    q0 = arr.P.position(arr.P.init)
    J = jax.jacobian(lambda q: arr.omega(q, jnp.zeros(0), jnp.zeros(0))[0])(q0)
    J = np.asarray(J)[1:, 1:].reshape(m, K + 1, m, K + 1)
    return J[:, :K]


__all__ = ["gate", "bettor_gated", "market_gated", "market_gated_run", "listener_sensitivity", "hypotheses", "ring", "all_to_all", "hub", "bettor", "market", "market_run", "seeds", "locked_wrong"]
