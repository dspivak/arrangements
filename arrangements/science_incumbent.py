"""Science at two scales, test (i): what is built on an idea pays for its excuses, natively.

Predictions, fixed before this file existed, in ``notes/SCIENCE-INCUMBENT-PREDICTIONS-2026-10-01.md`` §2
(numbers and closed forms under "Numbers fixed before code"); results in its §4. Quoted:

  (i1) ``D = 0`` reproduces test (g4)'s incumbent-only overtake count (26-30 at its settings) exactly.
  (i2) With ``R`` infinite, the overtake count rises with ``D`` (monotone in the four values); the run
  derives the closed form in the regime where the fund never runs dry.
  (i3) With ``R`` small, dependents rebuild early and the overtake count returns to within 2 of (i1)'s,
  whatever ``D``. With ``R`` large, it lies between (i1)'s and (i2)'s. [Deferred; not built here.]
  (i4) *Funerals.* With ``R`` infinite and stipends falling as ``h0``'s share falls, the incumbent holds until
  the fund cannot pay for the next excuse ... the gap between the last dependent's death and the overtake;
  predicted within 3 steps when death comes first.
  (i5) Who pays: in every arm, the total drawn from the fund equals the total paid in by dependents, to
  round-off.

NATIVE. Boxes, wires, ``Phi``; nothing decided in Python between steps. ``wire_boxes`` from
``science_native`` composes a world, a reader, a fund and ten dependents (absent ones start at supply 0 and
are silent from the start, so one compiled composite serves every ``D``):
* WORLD (``draw_source``): row ``c`` = (A draws, B draws, ``seen_j = 1`` for ``j < c``), ``hold`` steps each.
* READER ``I -> <140 + 1 | 9>``: (g4)'s incumbent-only chooser in the natural chart (``lambda = log p0/p1``,
  response ``1/(p0 p1)`` with (g4)'s stable ``p0 p1``) plus attachments for B draws 0-7 (hard variant: figures
  ``k < j`` absent at draw ``j``; draws 8 on have no figures, an exact reduction of (g4)). Its own cost
  ``K0 = sum 1/2 a^2 + sum_j seen_j [1/2 (b_j - s . a_j)^2 + (1 - sigma gF) delta . a_j]`` (``j < 8``; else
  ``1/2 b_j^2``), ``gF`` the fund's published solvency. Forcing ``-dK0`` on attachments, ``p0 p1 (K1 - K0 -
  lambda)`` on ``lambda``, zero on in-ports. Out-port ``(p0, delta . a_j for j < 8)``.
* FUND ``I -> <8 + 8 + 10 | 1>``: state ``F``; forcing ``(sum_d pay_d - sigma gF sum_j seen_j E_j) / hold``
  (Euclidean, response 1, so exact bookkeeping); out-port ``gF = gate(F, F1)``.
* DEPENDENT ``I -> <1 | 1>``: state ``s``; forcing ``gate(s, s1) (pi p0 - c - f) / hold``; out-port its
  payment ``gate(s, s1) f``. Dead dependents fall silent and pay nothing (test (h)'s correction).

NOT EXACT OR STIPULATED (disclosed). The fund pays half (``sigma``) of each held excuse's drawing cost per
count, a maintenance reading; the reader's relief is ``sigma gF delta . a``. Payments are rates per count
applied as rate/``hold`` per step. The world's readout is a lookup (``science_native``'s GAP). The reader's
attachment forcing is ``-dK0`` with ``gF`` treated as data (it is an in-port), the fund's draw is computed
from published values, so the pair is a bookkeeping, not one potential. No floor: death and insolvency are
smooth through ``science_payment.gate`` (per-step changes are far below the gates' widths). No rebuild.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .arrangement import Arrangement
from .coalgebra import configuration
from .functors import Phi
from .active_inference import _IN_DIR, _IN_POS
from .rvect import ReactiveVectorSpace, euclidean
from .science_native import draw_source, wire_boxes
from .science_payment import gate

K_FIG, NX = 8, 8                     # figures per sign; excusable B draws (hard variant)
N_DEP = 10


def _p0p1(lam):
    return jnp.exp(-jax.nn.softplus(lam) - jax.nn.softplus(-lam))


def reader(d, nA, nB, delta, sigma) -> Arrangement:
    """The incumbent's reader (module docstring)."""
    delta = jnp.asarray(delta, float)
    s = jnp.concatenate([jnp.full(K_FIG, d), jnp.full(K_FIG, -d)])
    dl = jnp.concatenate([delta, delta])
    allowed = jnp.concatenate([(jnp.arange(K_FIG)[None, :] >= jnp.arange(NX)[:, None])] * 2, 1).astype(float)
    na = NX * 2 * K_FIG

    def costs(q, n_in):
        a = q[1:].reshape(NX, 2 * K_FIG) * allowed
        A, B, seen, gF = n_in[:nA], n_in[nA:nA + nB], n_in[nA + nB:nA + 2 * nB], n_in[-1]
        r = B[:NX] - a @ s
        exc = 0.5 * r ** 2 + (1.0 - sigma * gF) * (a @ dl)
        K0 = jnp.sum(0.5 * A ** 2) + jnp.sum(seen[:NX] * exc) + jnp.sum(seen[NX:] * 0.5 * B[NX:] ** 2)
        K1 = jnp.sum(0.5 * (A - d) ** 2) + jnp.sum(seen * 0.5 * (B - d) ** 2)
        return K0, K1

    dK0 = jax.grad(lambda q, n_in: costs(q, n_in)[0], argnums=0)

    def force(q, n_in):
        K0, K1 = costs(q, n_in)
        g_lam = _p0p1(q[0]) * (K1 - K0 - q[0])
        return jnp.concatenate([g_lam[None], -dK0(q, n_in)[1:]])

    def readout(q):
        a = q[1:].reshape(NX, 2 * K_FIG) * allowed
        return jnp.concatenate([jax.nn.sigmoid(q[:1]), a @ dl])

    Q = ReactiveVectorSpace(dim=1 + na, sharp_fn=lambda q: jnp.diag(jnp.concatenate([
        1.0 / _p0p1(q[:1]), 0.02 * q[1:] * (1.0 - q[1:])])))
    return Arrangement(
        P=configuration(Q, init=jnp.concatenate([jnp.zeros(1), jnp.full(na, 0.01)])),
        in_dim_M=0, out_dim_M=0, in_dim_N=nA + 2 * nB + 1, out_dim_N=1 + NX,
        out_f=lambda q, m_out: readout(q),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=lambda q, m_out, n_in: (force(q, n_in), jnp.zeros_like(m_out), jnp.zeros_like(n_in)),
        label="incumbent reader",
    )


def _box(dim_in, init, readout, force, label) -> Arrangement:
    return Arrangement(
        P=configuration(euclidean(1, 1.0), init=jnp.reshape(jnp.asarray(init, float), (1,))),
        in_dim_M=0, out_dim_M=0, in_dim_N=dim_in, out_dim_N=1,
        out_f=lambda q, m_out: readout(q),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=lambda q, m_out, n_in: (force(q, n_in), jnp.zeros_like(m_out), jnp.zeros_like(n_in)),
        label=label,
    )


def fund(sigma, F1, hold) -> Arrangement:
    """``F`` moves by ``(sum pay - sigma gF sum seen E) / hold``; publishes ``gF``."""
    def force(q, n_in):
        E, seen, pay = n_in[:NX], n_in[NX:2 * NX], n_in[2 * NX:]
        return jnp.reshape((jnp.sum(pay) - sigma * gate(q[0], F1) * jnp.sum(seen * E)) / hold, (1,))
    return _box(2 * NX + N_DEP, 0.0, lambda q: gate(q, F1), force, "excuse fund")


def dependent(s0, pi, c, f, s1, hold, label) -> Arrangement:
    """Supply ``s`` moves by ``g(s) (pi p0 - c - f) / hold``; publishes its payment ``g(s) f``."""
    return _box(1, s0, lambda q: gate(q, s1) * f,
                lambda q, n_in: gate(q, s1) * (pi * n_in[:1] - c - f) / hold, label)


@partial(jax.jit, static_argnames=("hold", "counts"))
def incumbent_run(A, B, S0, d, delta, sigma, pi, c, f, s1, F1, hold, counts):
    """Wire world, reader, fund and dependents (starting supplies ``S0``, length ``N_DEP``); run by ``Phi``.
    Returns per count end: ``lambda``, ``F``, supplies, and the cumulative paid-in and drawn (observed from
    the states, summed step by step; nothing fed back)."""
    nA, nB = A.shape[0], B.shape[0]
    seen = jnp.tril(jnp.ones((nB + 1, nB)), -1)
    rows = jnp.concatenate([jnp.tile(A, (nB + 1, 1)), jnp.tile(B, (nB + 1, 1)), seen], axis=1)
    boxes = [draw_source(rows, hold), reader(d, nA, nB, delta, sigma), fund(sigma, F1, hold)]
    boxes += [dependent(S0[i], pi, c, f, s1, hold, f"dependent {i}") for i in range(N_DEP)]
    feeds = [[], [(0, k) for k in range(nA + 2 * nB)] + [(2, 0)],
             [(1, 1 + j) for j in range(NX)] + [(0, nA + nB + j) for j in range(NX)]
             + [(3 + i, 0) for i in range(N_DEP)]]
    feeds += [[(1, 0)] for _ in range(N_DEP)]
    arr = wire_boxes(boxes, feeds)
    O = Phi(arr)
    na = 1 + NX * 2 * K_FIG
    dl = jnp.concatenate([jnp.asarray(delta, float)] * 2)
    allowed = jnp.concatenate([(jnp.arange(K_FIG)[None, :] >= jnp.arange(NX)[:, None])] * 2, 1).astype(float)

    def one(carry, _):
        st, paid, drawn = carry
        x = arr.P.position(st)
        cnt = jnp.minimum(jax.lax.stop_gradient(x[0]).astype(int) // hold, nB)
        E = (x[2:1 + na].reshape(NX, 2 * K_FIG) * allowed) @ dl
        F, S = x[1 + na], x[2 + na:]
        paid = paid + jnp.sum(gate(S, s1) * f) / hold
        drawn = drawn + sigma * gate(F, F1) * jnp.sum(seen[cnt, :NX] * E) / hold
        _, _, st2 = O.with_state(st).run_one(_IN_POS, lambda _o: _IN_DIR)
        return (st2, paid, drawn), None

    def per_count(carry, _):
        carry, _ = jax.lax.scan(one, carry, None, length=hold)
        x = arr.P.position(carry[0])
        return carry, jnp.concatenate([x[1:2], x[1 + na:], jnp.stack([carry[1], carry[2]])])

    _, out = jax.lax.scan(per_count, (arr.P.init, 0.0, 0.0), None, length=counts)
    return out          # columns: lambda, F, S (N_DEP), paid, drawn


def run_arm(D, seeds=range(10), d=3.0, nA=20, nB=60, s0=7.5, pi=1.0, c=0.5, f=1.0, sigma=0.5, s1=0.1,
            F1=0.5, hold=400, counts=61):
    """All seeds of one arm (vmapped); ``D`` dependents present, the rest at supply 0."""
    AB = [np.random.default_rng(sd) for sd in seeds]
    AB = [(r.normal(0, 1, nA), r.normal(d, 1, 150)[:nB]) for r in AB]
    A = jnp.asarray(np.stack([x[0] for x in AB])); B = jnp.asarray(np.stack([x[1] for x in AB]))
    S0 = jnp.asarray([s0] * D + [0.0] * (N_DEP - D), float)
    delta = jnp.asarray(0.5 * np.arange(1, K_FIG + 1) * (d * d / 9))
    g = lambda a, b: incumbent_run(a, b, S0, d, delta, sigma, pi, c, f, s1, F1, hold, counts)
    return np.asarray(jax.vmap(g)(A, B))


def summarize(out, D, s1=0.1, F1=0.5):
    """Per seed: overtake count, last death count, dry count, final F, paid, drawn."""
    res = []
    for o in out:
        lam, F, S, paid, drawn = o[:, 0], o[:, 1], o[:, 2:2 + N_DEP], o[:, -2], o[:, -1]
        nstar = int(np.argmax(lam < 0)) if (lam < 0).any() else None
        dead = None
        if D:
            alive = np.asarray(gate(jnp.asarray(S[:, :D]), s1)) >= 0.01
            dd = np.nonzero(~alive.any(1))[0]
            dead = int(dd[0]) if len(dd) else None
        gF = np.asarray(gate(jnp.asarray(F), F1))
        dr = np.nonzero((gF < 0.5) & (np.arange(len(F)) >= 2))[0]
        dry = int(dr[0]) if (D and len(dr)) else None
        res.append(dict(nstar=nstar, death=dead, dry=dry, F=float(F[-1]), paid=float(paid[-1]),
                        drawn=float(drawn[-1])))
    return res




# --- test (j): a fixer paid to search for new excuses ------------------------------------------------
#
# Numbers and closed forms fixed before this code in the note's §5 ("Numbers fixed before code (j)").
# NOT NATIVE OR STIPULATED (disclosed): the fixer's search is a SEEDED CANDIDATE SEQUENCE computed before the
# run (each candidate: checks survived G, geometric; distance U(0.5, 4)); which candidates are admitted, and
# when, is a lookup the world box emits as a mask, as it emits the draws. PLANTING: every candidate pair's
# attachment is held at a0 = 0.01 from the start and masked out of the cost until admitted (zero forcing),
# so the spec's support-creating push is replaced by (g4)'s masking. Per draw, only the 3 cheapest usable
# figures admitted by some count get attachments (exact at rest unless b > 3d).

J_FS = (0, 4, 16, 64, 256, 1024, 4096)
J_PMAX = 260


def j_candidates(seed, F, q=0.5, counts=61, fmax=4096):
    """(admission count n, checks survived G, distance) of every admitted candidate of arm ``F``."""
    r = np.random.default_rng(1000 + seed)
    G = r.geometric(q, size=(counts, fmax)) - 1
    dl = r.uniform(0.5, 4.0, size=(counts, fmax))
    out = []
    for n in range(counts):
        for i in np.nonzero(G[n, :F] >= n)[0]:
            out.append((n, int(G[n, i]), float(dl[n, i])))
    return out


def j_pairs(cands, nB=60, R=3):
    """Per draw ``j``, the figures among the ``R`` cheapest usable ones admitted by some count:
    ``(draw, admission count, delta, G)``."""
    pairs = []
    for j in range(nB):
        best = []
        for c in sorted([c for c in cands if c[1] > j], key=lambda c: c[0]):
            if len(best) < R or c[2] < max(b[2] for b in best):
                pairs.append((j, c[0], c[2], c[1]))
                best = sorted(best + [c], key=lambda b: b[2])[:R]
    return pairs


def fixer_reader(d, nA, nB, delta, pair_draw, pair_delta) -> Arrangement:
    """(i)'s reader with no fund, plus one attachment per (draw, new figure) pair, masked by the in-port."""
    delta = jnp.asarray(delta, float)
    s = jnp.concatenate([jnp.full(K_FIG, d), jnp.full(K_FIG, -d)])
    dl = jnp.concatenate([delta, delta])
    allowed = jnp.concatenate([(jnp.arange(K_FIG)[None, :] >= jnp.arange(NX)[:, None])] * 2, 1).astype(float)
    na, P = NX * 2 * K_FIG, pair_draw.shape[0]
    onto = jax.nn.one_hot(pair_draw, nB, dtype=float)          # (P, nB): pair -> its draw

    def costs(q, n_in):
        a = q[1:1 + na].reshape(NX, 2 * K_FIG) * allowed
        A, B, seen, m = n_in[:nA], n_in[nA:nA + nB], n_in[nA + nB:nA + 2 * nB], n_in[nA + 2 * nB:]
        e = q[1 + na:] * m
        shift = (e * d) @ onto + jnp.concatenate([a @ s, jnp.zeros(nB - NX)])
        draw = (e * pair_delta) @ onto + jnp.concatenate([a @ dl, jnp.zeros(nB - NX)])
        K0 = jnp.sum(0.5 * A ** 2) + jnp.sum(seen * (0.5 * (B - shift) ** 2 + draw))
        K1 = jnp.sum(0.5 * (A - d) ** 2) + jnp.sum(seen * 0.5 * (B - d) ** 2)
        return K0, K1

    dK0 = jax.grad(lambda q, n_in: costs(q, n_in)[0], argnums=0)

    def force(q, n_in):
        K0, K1 = costs(q, n_in)
        return jnp.concatenate([(_p0p1(q[0]) * (K1 - K0 - q[0]))[None], -dK0(q, n_in)[1:]])

    dim = 1 + na + P
    Q = ReactiveVectorSpace(dim=dim, sharp_fn=lambda q: jnp.diag(jnp.concatenate([
        1.0 / _p0p1(q[:1]), 0.02 * q[1:] * (1.0 - q[1:])])))
    return Arrangement(
        P=configuration(Q, init=jnp.concatenate([jnp.zeros(1), jnp.full(na + P, 0.01)])),
        in_dim_M=0, out_dim_M=0, in_dim_N=nA + 2 * nB + P, out_dim_N=1,
        out_f=lambda q, m_out: jax.nn.sigmoid(q[:1]),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=lambda q, m_out, n_in: (force(q, n_in), jnp.zeros_like(m_out), jnp.zeros_like(n_in)),
        label="incumbent reader with a fixer",
    )


@partial(jax.jit, static_argnames=("hold", "counts"))
def fixer_run(A, B, pair_draw, pair_adm, pair_delta, d, delta, hold, counts):
    """World (row ``c``: A, B, seen, admitted mask) wired to the reader, run by ``Phi``; ``lambda`` per count
    and the new pairs' final attachments."""
    nA, nB = A.shape[0], B.shape[0]
    seen = jnp.tril(jnp.ones((nB + 1, nB)), -1)
    mask = (pair_adm[None, :] <= jnp.arange(nB + 1)[:, None]).astype(float)
    rows = jnp.concatenate([jnp.tile(A, (nB + 1, 1)), jnp.tile(B, (nB + 1, 1)), seen, mask], axis=1)
    P = pair_draw.shape[0]
    arr = wire_boxes([draw_source(rows, hold), fixer_reader(d, nA, nB, delta, pair_draw, pair_delta)],
                     [[], [(0, k) for k in range(nA + 2 * nB + P)]])
    O = Phi(arr)

    def one(st, _):
        _, _, st2 = O.with_state(st).run_one(_IN_POS, lambda _o: _IN_DIR)
        return st2, None

    def per_count(st, _):
        st, _ = jax.lax.scan(one, st, None, length=hold)
        return st, arr.P.position(st)[1]

    st, lam = jax.lax.scan(per_count, arr.P.init, None, length=counts)
    return lam, arr.P.position(st)[2 + NX * 2 * K_FIG:]


def run_fixer_arm(F, seeds=range(10), d=3.0, nA=20, nB=60, hold=400, counts=61):
    """All seeds of arm ``F`` (vmapped, pairs padded to ``J_PMAX`` with never-admitted dummies)."""
    As, Bs, PD, PA, PL, pairs_all = [], [], [], [], [], []
    for sd in seeds:
        r = np.random.default_rng(sd)
        As.append(r.normal(0, 1, nA)); Bs.append(r.normal(d, 1, 150)[:nB])
        pr = j_pairs(j_candidates(sd, F), nB) if F else []
        if len(pr) > J_PMAX:
            raise ValueError(f"{len(pr)} pairs > {J_PMAX}")
        pairs_all.append(pr)
        pad = J_PMAX - len(pr)
        PD.append([p[0] for p in pr] + [0] * pad); PA.append([p[1] for p in pr] + [10 ** 6] * pad)
        PL.append([p[2] for p in pr] + [0.0] * pad)
    delta = jnp.asarray(0.5 * np.arange(1, K_FIG + 1) * (d * d / 9))
    g = lambda a, b, pd, pa, pl: fixer_run(a, b, pd, pa, pl, d, delta, hold, counts)
    lam, att = jax.vmap(g)(*(jnp.asarray(np.array(x)) for x in (As, Bs, PD, PA, PL)))
    return np.asarray(lam), np.asarray(att), pairs_all


__all__ = ["reader", "fund", "dependent", "incumbent_run", "run_arm", "summarize", "j_candidates", "j_pairs",
           "fixer_reader", "fixer_run", "run_fixer_arm"]
