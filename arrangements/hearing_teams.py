"""Addressed hearing teams: members that send a record as their address, teams whose trust-levels read it
(book plan P104, *Addressed hearing teams: roles nobody assigned*; Ch 4 ``def.addressed_arrangement``,
``def.predictive_attention``; Ch 5 ``sec.teams_learn``, hearing teams and ``prop.hearing_mixes``).

Everything here is built from ``Arrangement``, ``compose_seq`` and ``tensor_arrangements``; the dynamics functor
(``functors.closed_step``) runs the composite, so every load reaches every configuration by the library's own
pullback-and-sum. Nothing is a hand-written gradient. The book's objects, in its words (the mathematics is the agents',
*proposed*, P104's block):

**A member** is a system of the predictive addressing arrangements, interface ``(1, t) -> (R, t)``: it hears one
value ``y`` (address space ``1``) and sends an address and one value. Its configuration is ``q = (x, a, b)``: ``x``
its *record*, ``(a, b)`` its line. Its out-transfer at ``q`` sends the address ``x`` and the value ``a x + b`` (its
word, the voice of a voiced network ``0 -> t`` of share 1). Its residual is ``sqrt(K) (y - x)``, so its cost is
``K/2 (y - x)^2``: it is graded on recording what it hears, not on forecasting. Its integrator is the identity
look-ahead with update ``x + tau xi_x / K``, ``(a, b) + tau eta (xi_a, xi_b)``: with no other load, a step of
duration 1 lands the record on ``y`` exactly; a load ``L`` from outside bends it to ``y + L / K``.

**A team** of ``n`` is a hearing team of the predictive addressing arrangements, ``(1,t)->(R,t), ..., -> (1,t)->(R,t)``,
so a member and a team have one interface and a team of teams is composition. Its configuration is the scores'
coefficients ``(C, D)`` (and, with a voice, ``(C_0, D_0, a_0, b_0)``). Its out-transfer at ``q`` is the morphism
``(f, g)`` of ``Fam(VNet)``: ``f`` sends the members' addresses to their mean ``abar``, and ``g(a)`` is the voiced
network whose trust-levels are ``softmax(C_k a_k + D_k)`` over the members (and the voice, score ``C_0 abar + D_0``),
voice constant ``share_0 (a_0 abar + b_0)``. Its in-transfer delivers what comes in to each member in proportion 1,
and no member hears another. Its residual, if it carries one, is ``y - prediction``.

**The world** is the outside of the outermost team: a clock that delivers ``y[floor t]`` to the team's input and
ignores what the team sends and pushes back (as in ``tests/test_trust_in_time.py``). Changing the series halfway is
the environment's schedule through a port the team has (STATUS.md §2, "Not evidence of an event").
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .arrangement import Arrangement, exact, zero_form
from .coalgebra import CotCoalg
from .functors import closed_step
from .wiring import compose_seq, tensor_arrangements

EMPTY = jnp.zeros(0)


def _euclidean(init, rates, label):
    """An integrator with identity look-ahead whose update is ``s + tau rates * xi`` (a diagonal response)."""
    rates = jnp.asarray(rates, float)
    return CotCoalg(Q_dim=len(rates), init=jnp.asarray(init, float), position=lambda s: s,
                    step=lambda s, xi: s + rates * xi, label=label,
                    timed_step=lambda s, xi, tau: s + tau * rates * xi)


def member(K: float, eta: float, init, label: str = "member", obs_dim: int = 1) -> Arrangement:
    """A member ``(1, t) -> (R, t)``: configuration ``(x, a, b)``, sends ``(x, a x + b)``, residual ``sqrt(K)(c - x)``.

    It hears an observation of ``obs_dim`` coordinates and records the last, ``c``. With ``obs_dim = 1`` the
    observation is the series itself, so the record is the last observation; with ``obs_dim = 2`` the world sends
    ``(y_t, x_{t+1})``, the outcome and the next context, and the member records the context a step before the
    outcome it is used to predict arrives.
    """
    def cost(q, m_out, n_in):
        return 0.5 * K * (n_in[obs_dim - 1] - q[0]) ** 2

    return Arrangement(
        P=_euclidean(init, [1.0 / K, eta, eta], label),
        in_dim_M=0, out_dim_M=0, in_dim_N=obs_dim, out_dim_N=2,
        out_f=lambda q, m_out: jnp.array([q[0], q[1] * q[0] + q[2]]),
        in_f=lambda q, m_out, n_in: EMPTY,
        omega=exact(lambda q, m, n: -cost(q, m, n)), label=label)


def team_shares(q, addresses, voice: bool):
    """The trust-levels (and the voice's share, last) at configuration ``q`` and the members' addresses."""
    n = addresses.shape[0]
    s = q[:n] * addresses + q[n:2 * n]
    if voice:
        abar = addresses.mean()
        s = jnp.append(s, q[2 * n] * abar + q[2 * n + 1])
    return jax.nn.softmax(s)


def team_out(q, m_out, n: int, voice: bool):
    """The out-transfer's value action: ``(abar, prediction)`` from the members' ``(address, value)`` pairs."""
    a, w = m_out[0::2], m_out[1::2]
    g = team_shares(q, a, voice)
    abar = a.mean()
    pred = g[:n] @ w
    if voice:
        pred = pred + g[n] * (q[2 * n + 2] * abar + q[2 * n + 3])
    return jnp.array([abar, pred])


def team(n: int, eta: float, init, voice: bool = False, residual: bool = True, label: str = "team",
         obs_dim: int = 1) -> Arrangement:
    """A hearing team of ``n`` members; with ``residual``, its residual is ``y - prediction``, ``y`` the first
    coordinate of what comes in. It delivers what comes in, all ``obs_dim`` coordinates, to each member."""
    def cost(q, m_out, n_in):
        return 0.5 * (n_in[0] - team_out(q, m_out, n, voice)[1]) ** 2

    return Arrangement(
        P=_euclidean(init, [eta] * len(init), label),
        in_dim_M=n * obs_dim, out_dim_M=2 * n, in_dim_N=obs_dim, out_dim_N=2,
        out_f=lambda q, m_out: team_out(q, m_out, n, voice),
        in_f=lambda q, m_out, n_in: jnp.tile(n_in, n),
        omega=exact(lambda q, m, nn: -cost(q, m, nn)) if residual else zero_form, label=label)


def accountable_member(K: float, eta: float, init, label: str = "member", obs_dim: int = 1) -> Arrangement:
    """A member whose input carries an address: ``(R, t) -> (R, t)``. It hears ``(s, y[, c])``: ``s`` the share in
    which the whole hears it, ``y`` the outcome, ``c`` the context it records (``y`` itself when ``obs_dim = 1``).
    Its residual is ``(sqrt(K)(c - x), sqrt(s)(y - a x - b))``: graded on its own word, weighted by how much it is
    heard (Jacobs, Jordan, Nowlan, Hinton 1991's competitive error, as a residual affine in the values at each
    configuration and address)."""
    def cost(q, m_out, n_in):
        s, y, c = n_in[0], n_in[1], n_in[obs_dim]
        return 0.5 * K * (c - q[0]) ** 2 + 0.5 * s * (y - q[1] * q[0] - q[2]) ** 2

    return Arrangement(
        P=_euclidean(init, [1.0 / K, eta, eta], label),
        in_dim_M=0, out_dim_M=0, in_dim_N=1 + obs_dim, out_dim_N=2,
        out_f=lambda q, m_out: jnp.array([q[0], q[1] * q[0] + q[2]]),
        in_f=lambda q, m_out, n_in: EMPTY,
        omega=exact(lambda q, m, n: -cost(q, m, n)), label=label)


def accountable_team(n: int, eta: float, init, label: str = "team", obs_dim: int = 1) -> Arrangement:
    """A hearing team with no residual of its own whose in-transfer tells each member its share: what comes in is
    ``(s, y[, c])``, and member ``k`` is delivered ``(s g_k(a), y[, c])``, ``g`` the trust-levels at the members'
    addresses ``a``. Its out-transfer is ``team_out`` (no voice). Nested, a member's address in is the product of the
    trust-levels on its path, its share in the composite."""
    d = 1 + obs_dim

    def in_f(q, m_out, n_in):
        g = team_shares(q, m_out[0::2], False)
        rows = jnp.concatenate([(n_in[0] * g)[:, None], jnp.tile(n_in[1:], (n, 1))], 1)
        return rows.reshape(-1)

    return Arrangement(
        P=_euclidean(init, [eta] * len(init), label),
        in_dim_M=n * d, out_dim_M=2 * n, in_dim_N=d, out_dim_N=2,
        out_f=lambda q, m_out: team_out(q, m_out, n, False),
        in_f=in_f, omega=zero_form, label=label)


def expert_team(n: int, K: float, eta: float, init, label: str = "team", obs_dim: int = 1) -> Arrangement:
    """An expert team of ``n >= 0`` members, interface ``(R, t) -> (R, t)`` like each member's (book plan P104,
    *proposed*). Configuration ``(x, a_0, b_0, C_1..C_n, D_1..D_n, C_0, D_0)``: its record ``x``, its own guess
    ``a_0 x + b_0``, the scores' coefficients. Trust-levels ``g = softmax(C_k a_k + D_k, C_0 x + D_0)`` over the
    members' addresses ``a_k`` and its own record; it sends ``(x, sum_k g_k w_k + g_0 (a_0 x + b_0))``. It hears
    ``(s, y[, c])`` and delivers ``(s g_k, y[, c])`` to member ``k``. Residual ``(sqrt(K)(c - x), sqrt(s g_0)(y -
    a_0 x - b_0))``: its record, and its own guess weighted by the share in which the whole hears that guess. With
    ``n = 0``, ``g_0 = 1``: a predictor that records the context and guesses a line of it."""
    d = 1 + obs_dim

    def shares(q, m_out):
        s = jnp.concatenate([q[3:3 + n] * m_out[0::2] + q[3 + n:3 + 2 * n],
                             jnp.array([q[3 + 2 * n] * q[0] + q[4 + 2 * n]])])
        return jax.nn.softmax(s)

    def out_f(q, m_out):
        g = shares(q, m_out)
        return jnp.array([q[0], g[:n] @ m_out[1::2] + g[n] * (q[1] * q[0] + q[2])])

    def in_f(q, m_out, n_in):
        g = shares(q, m_out)
        rows = jnp.concatenate([(n_in[0] * g[:n])[:, None], jnp.tile(n_in[1:], (n, 1))], 1)
        return rows.reshape(-1)

    def cost(q, m_out, n_in):
        g0 = shares(q, m_out)[n]
        s, y, c = n_in[0], n_in[1], n_in[obs_dim]
        return 0.5 * K * (c - q[0]) ** 2 + 0.5 * s * g0 * (y - q[1] * q[0] - q[2]) ** 2

    return Arrangement(
        P=_euclidean(init, [1.0 / K] + [eta] * (len(init) - 1), label),
        in_dim_M=n * d, out_dim_M=2 * n, in_dim_N=d, out_dim_N=2,
        out_f=out_f, in_f=in_f, omega=exact(lambda q, m, nn: -cost(q, m, nn)), label=label)


def expert_init(n: int, rng, scale: float = 0.1):
    """Record 0, a small random own guess, small random score slopes, zero intercepts."""
    return [0.0, rng.normal(0, scale), rng.normal(0, scale)] + list(rng.normal(0, scale, n)) + [0.0] * n + \
        [rng.normal(0, scale), 0.0]


def team_init(n: int, rng, voice: bool = False, scale: float = 0.1):
    """Small random score slopes, zero intercepts; with a voice, its score and its line too."""
    C = rng.normal(0, scale, n + (1 if voice else 0))
    D = np.zeros(n + (1 if voice else 0))
    q = list(C[:n]) + list(D[:n])
    if voice:
        q += [C[n], D[n], rng.normal(0, scale), rng.normal(0, scale)]
    return q


def member_init(rng, scale: float = 0.1):
    return [0.0, rng.normal(0, scale), rng.normal(0, scale)]


def fill(members: Sequence[Arrangement], parent: Arrangement) -> Arrangement:
    """The parent with its boxes filled: composition in arr, ``parent . (m_1 (x) ... (x) m_n)``."""
    return compose_seq(tensor_arrangements(list(members)), parent)


def world(series, inner: Arrangement) -> Arrangement:
    """The outside of ``inner``: a clock delivering ``series[floor t]`` (a number, or a row of ``inner``'s input
    dimension) to its input; it reads nothing back."""
    series = jnp.asarray(series, float).reshape(len(series), -1)

    def today(t):
        return series[jnp.minimum(jnp.floor(t[0]).astype(int), len(series) - 1)]

    clock = CotCoalg(Q_dim=1, init=jnp.zeros(1), position=lambda s: s, step=lambda s, xi: s + 1.0, label="world",
                     timed_step=lambda s, xi, tau: s + tau)
    return Arrangement(
        P=clock, in_dim_M=inner.in_dim_N, out_dim_M=inner.out_dim_N, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: EMPTY,
        in_f=lambda q, m_out, n_in: today(q),
        omega=lambda q, m, n: (jnp.zeros(1), jnp.zeros_like(m), EMPTY), label="world")


def run(whole: Arrangement, series, duration: float = 1.0):
    """Close ``whole`` by the world on ``series`` and run one step per observation, at the given duration.

    Returns the positions before each step (``(T, Q_dim)``, the world's clock last) and the prediction ``whole``
    sends at each step, read off its out-transfer at that position, before the observation arrives.
    """
    system = compose_seq(whole, world(series, whole))
    step = closed_step(system, duration=duration)
    k = whole.P.Q_dim
    readout = lambda q: whole.out_f(q[:k], EMPTY)[1]

    def one(s, _):
        q = system.P.position_at(s, duration)
        return step(s), (q, readout(q))

    _, (qs, preds) = jax.lax.scan(one, system.P.init, None, length=len(series))
    return system, np.asarray(qs), np.asarray(preds)


def ar_series(f, T: int, sig: float, seed: int, y0: float = 0.0):
    """``y_t = f(y_{t-1}) + sig eps_t``, the world's arriving observations."""
    rng = np.random.default_rng(seed)
    y = np.zeros(T)
    prev = y0
    noise = sig * rng.standard_normal(T)
    for t in range(T):
        prev = float(f(prev)) + noise[t]
        y[t] = prev
    return y
