"""The joint readout: a prime-maker whose two parts make its number only together (book plan P73 (b)).

The book's ``notes/runs/p75-foundation-d734ed3e/RESULTS.md`` built the recursive scatter: a listener that needs
primes pushes away the finders that send composites, and the same force, reaching a finder's own placements through
its readout, drives that finder's parts apart. David's objection, 2026-10-26: "there's nothing in them that makes
those two parts actually working together resposneible for the output". Each part sent the finder's number ``N``,
so either part alone made ``N`` and breaking up only lowered the weight. This module is the fix by route 2, the
book's *need*: each part's correct piece requires receiving the other's through the wiring.

Three levels, each an arrangement, composed by ``compose_seq`` and run by ``Phi`` (``step_fn``):

* a **part** (``piece_part``), a box ``I -> B`` with state ``s``: it sends ``(s, 1)`` and carries the cost
  ``(k/2)(s - pi - w_in)^2``, ``w_in`` the weight of what it receives. So it completes its number with how loudly
  it hears its partner. ``redundant_part`` is p75's part, sending ``(N, 1)`` with no state.
* a **finder** (``finder``), an attention arrangement with two inputs: placements ``(l_a, l_b)`` in its matter,
  ``T_0j = kappa(l_j)`` (it reads out each part at the kernel to its centre), ``T_ab = T_ba = kappa(l_a - l_b)``
  (the parts hear each other), every other transfer ``0``, cost ``0``. A prime-maker for ``P`` has ``pi = P - 1``.
* a **listener** (``listener``), an attention arrangement with one input per finder: placements ``p_f``,
  ``T_0f = kappa(p_f)``, every other transfer ``0``, and the cost ``sum_f wt(T_0f m_f) c(val m_f)``: how loudly it
  hears each finder, times how far that finder's value is from a prime. ``c`` is ``not_prime``; with
  ``fixed_c`` the cost is p75's, a constant per finder.

``kappa(x) = exp(-x^2 / 2 tau)`` (``attention.spatial_kappa``). The predictions were written before this file, in the
book's ``notes/runs/p73-b-56518228/PREDICTIONS.md``; the results are in ``RESULTS.md`` beside them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
from jax import Array

from .attention import (
    Transfer, attention_arrangement, signal, spatial_kappa, split_messages, weight, zero_transfer,
)
from .coalgebra import configuration
from .collective import EMPTY, ext, frozen_part, part
from .rvect import euclidean
from .wiring import compose_seq, tensor_arrangements

DH, DP = 1, 1


# ---------------------------------------------------------------------------
# How far a value is from a prime.
# ---------------------------------------------------------------------------

N_MAX = 256


def _primes(n: int) -> Array:
    flags = [False, False] + [True] * (n - 2)
    for i in range(2, int(n ** 0.5) + 1):
        if flags[i]:
            for j in range(i * i, n, i):
                flags[j] = False
    return jnp.array(flags)


IS_PRIME = _primes(N_MAX)


def is_prime(n: int) -> bool:
    return bool(IS_PRIME[int(n)])


def _f(x):
    """``e^{-1/x}`` for ``x > 0``, ``0`` otherwise: smooth, and flat at ``0``."""
    pos = x > 0.0
    return jnp.where(pos, jnp.exp(-1.0 / jnp.where(pos, x, 1.0)), 0.0)


def smooth_step(x):
    """The ``C^infty`` step: ``0`` for ``x <= 0``, ``1`` for ``x >= 1``, ``f(x) / (f(x) + f(1 - x))`` between."""
    return _f(x) / (_f(x) + _f(1.0 - x))


def not_prime(v):
    """``c(v)``: ``0`` within a quarter of a prime, ``1`` within a quarter of any other whole number, and the smooth
    step between two neighbouring whole numbers. A smooth function on the line, so a cost can be written with it."""
    n = jnp.floor(v)
    i = jnp.clip(n.astype(jnp.int32), 0, N_MAX - 2)
    c0 = jnp.where(IS_PRIME[i], 0.0, 1.0)
    c1 = jnp.where(IS_PRIME[i + 1], 0.0, 1.0)
    return c0 + (c1 - c0) * smooth_step((v - n - 0.25) / 0.5)


# ---------------------------------------------------------------------------
# Parts.
# ---------------------------------------------------------------------------


def piece_part(pi: float, s0: float, *, k: float, mobility: float, label: str):
    """A part that completes its number ``pi`` with how loudly it hears its partner: sends ``(s, 1)``, cost
    ``(k/2)(s - pi - w_in)^2`` with ``w_in = e[1]``, the weight of the message it receives."""
    return part(
        send=lambda q: jnp.array([q[0], 1.0]),
        cost=lambda q, e: 0.5 * k * (q[0] - pi - e[1]) ** 2,
        P=configuration(euclidean(1, mobility), init=jnp.array([float(s0)])),
        label=label)


def redundant_part(N: float, label: str):
    """p75's part: no state, always sending ``(N, 1)``."""
    return frozen_part(N, 1.0, label)


# ---------------------------------------------------------------------------
# The finder and the listener.
# ---------------------------------------------------------------------------


def _scalar(k) -> Transfer:
    return Transfer(gamma=k, L=jnp.reshape(k, (1, 1)))


def finder_transfers(tau: float):
    """``T_0j = kappa(l_j)``, ``T_ab = T_ba = kappa(l_a - l_b)``, the rest ``0``; ``q = (l_a, l_b)``."""
    Z = zero_transfer(DH)

    def T(q):
        la, lb = q[0:1], q[1:2]
        o = jnp.zeros(DP)
        ka, kb, kab = spatial_kappa(la, o, tau), spatial_kappa(lb, o, tau), spatial_kappa(la, lb, tau)
        return ((Z, _scalar(ka), _scalar(kb)),
                (Z, Z, _scalar(kab)),
                (Z, _scalar(kab), Z))

    return T


def finder(parts: Sequence, *, tau: float, placements, mobility: float, label: str):
    """The finder with its two parts filled in: ``compose_seq(parts_a (x) part_b, finder.arr())``, an arrangement
    ``I -> B``. Returns ``(arrangement, the attention arrangement)``."""
    a = attention_arrangement(
        2, DH, DP,
        placement=lambda q: jnp.reshape(q, (2, DP)),
        transfers=finder_transfers(tau),
        parameter=configuration(euclidean(2, mobility), init=jnp.asarray(placements, float)),
        label=label)
    return compose_seq(tensor_arrangements([p.arr for p in parts]), a.arr()), a


def listener_transfers(n: int, tau: float):
    Z = zero_transfer(DH)

    def T(q):
        o = jnp.zeros(DP)
        row0 = (Z,) + tuple(_scalar(spatial_kappa(q[f:f + 1], o, tau)) for f in range(n))
        rest = tuple(tuple(Z for _ in range(n + 1)) for _ in range(n))
        return (row0,) + rest

    return T


def listener_cost(n: int, tau: float, fixed_c: Optional[Sequence[float]] = None, alpha: float = 1.0):
    """``alpha sum_f wt(T_0f m_f) c_f``, with ``c_f = not_prime(val m_f)``, or ``fixed_c[f]`` (p75's cost)."""
    T = listener_transfers(n, tau)

    def cost(q, m_out, n_in):
        ms = split_messages(m_out, n, DH)
        Tq = T(q)
        total = 0.0
        for f in range(n):
            g = Tq[0][f + 1].gamma
            w = weight(ms[f])
            if fixed_c is None:
                safe = jnp.where(w > 0.0, w, 1.0)
                c = not_prime(signal(ms[f])[0] / safe)
            else:
                c = fixed_c[f]
            total = total + g * w * c
        return alpha * total

    return cost


@dataclass(frozen=True)
class Scene:
    arr: object            # the composite I -> B, run by Phi
    finders: tuple         # each finder's composite arrangement I -> B
    q_dims: tuple          # each finder's configuration dimension (its parts', then its placements)
    part_dims: tuple       # each finder's parts' configuration dimension
    n: int


def scene(finders_: Sequence, *, tau: float, positions, mobility: float, fixed_c=None, alpha: float = 1.0) -> Scene:
    """The listener with its finders filled in: ``compose_seq(finder_1 (x) ... (x) finder_n, listener.arr())``.
    ``finders_`` are ``(arrangement, attention arrangement)`` pairs from ``finder``."""
    n = len(finders_)
    lis = attention_arrangement(
        n, DH, DP,
        placement=lambda q: jnp.reshape(q, (n, DP)),
        transfers=listener_transfers(n, tau),
        cost=listener_cost(n, tau, fixed_c, alpha),
        parameter=configuration(euclidean(n, mobility), init=jnp.asarray(positions, float)),
        label="listener")
    arrs = [f[0] for f in finders_]
    top = compose_seq(tensor_arrangements(arrs), lis.arr())
    return Scene(top, tuple(arrs), tuple(a.P.Q_dim for a in arrs),
                 tuple(a.P.Q_dim - 2 for a in arrs), n)


# ---------------------------------------------------------------------------
# Running and reading.
# ---------------------------------------------------------------------------


EXT = ext()


def step_fn(S: Scene):
    """One step of ``Phi(S.arr)`` on the empty exterior input with zero exterior covector (``collective.step_fn``)."""
    P, omega = S.arr.P, S.arr.omega

    def step(s):
        xi_q, _, _ = omega(P.position(s), EMPTY, EXT)
        return P.step(s, xi_q)

    return step


def read(S: Scene, q) -> dict:
    """From the flat configuration: each finder's parts' states, its placements, its readout ``(h, w)`` from its own
    ``out_f``, and the listener's placements."""
    q = jnp.asarray(q, float)
    out, i = {"states": [], "inner": [], "readout": []}, 0
    for f in range(S.n):
        qf = q[i:i + S.q_dims[f]]
        out["states"].append(qf[:S.part_dims[f]])
        out["inner"].append(qf[S.part_dims[f]:])
        out["readout"].append(S.finders[f].out_f(qf, EMPTY))
        i += S.q_dims[f]
    out["outer"] = q[i:i + S.n]
    return out


def trace(S: Scene, steps: int):
    """``(q_0, q_1, ..., q_steps)``, the flat configurations along the run."""
    f = step_fn(S)
    P = S.arr.P

    def body(s, _):
        s = f(s)
        return s, P.position(s)

    _, qs = jax.lax.scan(body, P.init, None, length=steps)
    return jnp.concatenate([P.position(P.init)[None, :], qs])
