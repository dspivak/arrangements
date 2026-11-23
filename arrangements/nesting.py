"""Nesting a composite: the factorization of a wiring through a grouping of its boxes.

Book task ``well-factored-2ac0e890`` (David, 2026-11-23: "good factorizations should have 2-7
boxes inside. more of course may be inside of those ... make sure they are well-factored, so
that the parts make sense and there are not too many parts in any given arrangement").

Arrangements are the morphisms of ``arr``, and a composite ``w . (x)_i a_i`` with ``w`` a
static wiring is an operad composite: the wiring ``w`` is an operation with one input per part.
Given a grouping of the parts into consecutive blocks, ``w`` factors as

    w  =  u . (x)_b v_b,

``v_b`` the wiring inside block ``b`` and ``u`` the wiring between blocks, so that by
associativity of composition in ``arr``

    w . (x)_i a_i  =  u . (x)_b ( v_b . (x)_{i in b} a_i ),

the right side a composite of blocks, each itself a composite. ``v_b`` keeps inside the wires
that run between parts of the block and exposes, on the block's own interface, exactly the
outputs that are read outside the block and the inputs that read anything outside it.

``factor_linear`` does this for a wiring whose two maps are linear (every wiring the library
builds from index maps -- copies, dots and their transposes, sums at a ``+`` dot -- is one); it
reads the maps' matrices off ``jax.jacfwd`` and checks linearity at random points first. The
state of the nested composite is the flat one's, in the same order, since blocks are
consecutive and wirings carry no state; ``tests/test_nesting.py`` checks the two are equal.
"""
from __future__ import annotations

from dataclasses import replace
from typing import List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .arrangement import Arrangement, zero_form
from .wiring import compose_seq, tensor_arrangements, trivial_parameter


def _matrices(w: Arrangement):
    """The matrices of a static wiring's maps: ``out = Wo m``, ``in = Wm m + Wn n``."""
    q = jnp.zeros(0)
    m0, n0 = jnp.zeros(w.out_dim_M), jnp.zeros(w.in_dim_N)
    Wo = np.asarray(jax.jacfwd(lambda m: w.out_f(q, m))(m0)).reshape(w.out_dim_N, w.out_dim_M)
    Wm = np.asarray(jax.jacfwd(lambda m: w.in_f(q, m, n0))(m0)).reshape(w.in_dim_M, w.out_dim_M)
    Wn = np.asarray(jax.jacfwd(lambda n: w.in_f(q, m0, n))(n0)).reshape(w.in_dim_M, w.in_dim_N)
    rng = np.random.default_rng(11)
    for _ in range(3):
        m, n = rng.normal(size=w.out_dim_M), rng.normal(size=w.in_dim_N)
        ok = (np.allclose(np.asarray(w.out_f(q, jnp.asarray(m))), Wo @ m, atol=1e-9)
              and np.allclose(np.asarray(w.in_f(q, jnp.asarray(m), jnp.asarray(n))), Wm @ m + Wn @ n, atol=1e-9))
        if not ok:
            raise ValueError(f"factor_linear: the wiring {w.label!r} is not linear")
    return Wo, Wm, Wn


def _linear(in_M: int, out_M: int, in_N: int, out_N: int, Wo, Wm, Wn, label: str) -> Arrangement:
    """The static wiring ``<in_M|out_M> -> <in_N|out_N>`` with ``out = Wo m``, ``in = Wm m + Wn n``.

    A matrix that is a selection (one 1 per row) is applied by indexing, so a copy stays a copy."""

    def apply(W, x, rows):
        if rows == 0:
            return jnp.zeros(0)
        W = np.asarray(W)
        sel = (W != 0).sum(axis=1) == 1
        if np.all(sel) and np.all(W[W != 0] == 1.):
            return x[jnp.asarray(np.argmax(W != 0, axis=1))]
        return jnp.asarray(W) @ x

    def out_f(q, m):
        return apply(Wo, m, out_N)

    def in_f(q, m, n):
        if in_M == 0:
            return jnp.zeros(0)
        a = apply(Wm, m, in_M) if Wm.shape[1] and np.any(Wm) else jnp.zeros(in_M)
        b = apply(Wn, n, in_M) if Wn.shape[1] and np.any(Wn) else jnp.zeros(in_M)
        return a + b

    return Arrangement(trivial_parameter(), in_M, out_M, in_N, out_N, out_f, in_f, zero_form, label)


def _factor(parts: Sequence[Arrangement], w: Arrangement, blocks: Sequence[Tuple[int, str]]):
    """The blocks ``v_b . (x)_{i in b} parts_i`` and the wiring ``u`` between them."""
    parts = list(parts)
    if sum(s for s, _ in blocks) != len(parts):
        raise ValueError("factor_linear: block sizes must sum to the number of parts")
    Wo, Wm, Wn = _matrices(w)
    a = [p.in_dim_N for p in parts]          # each part's inputs
    b = [p.out_dim_N for p in parts]         # each part's outputs
    a_at, b_at = np.cumsum([0] + a), np.cumsum([0] + b)

    pieces: List[Arrangement] = []
    exp_in: List[np.ndarray] = []            # for each block, the global input coords it exposes
    exp_out: List[np.ndarray] = []
    start = 0
    for size, name in blocks:
        I = np.arange(a_at[start], a_at[start + size])
        O = np.arange(b_at[start], b_at[start + size])
        notO = np.setdiff1d(np.arange(Wm.shape[1]), O)
        notI = np.setdiff1d(np.arange(Wm.shape[0]), I)
        # Outputs read outside the block: by the outer readout or by another block's inputs.
        read_out = (np.any(Wo[:, O] != 0, axis=0) | np.any(Wm[np.ix_(notI, O)] != 0, axis=0)) if len(O) else np.zeros(0, bool)
        # Inputs that read anything outside the block: another block's outputs or the outer inputs.
        reads_out = (np.any(Wm[np.ix_(I, notO)] != 0, axis=1) | np.any(Wn[I] != 0, axis=1)) if len(I) else np.zeros(0, bool)
        Eo, Ei = O[read_out], I[reads_out]
        exp_out.append(Eo)
        exp_in.append(Ei)
        if size == 1 and len(Eo) == len(O) and len(Ei) == len(I):
            pieces.append(parts[start])
        else:
            v_out = np.zeros((len(Eo), len(O)))
            v_out[np.arange(len(Eo)), np.searchsorted(O, Eo)] = 1.
            v_m = Wm[np.ix_(I, O)]
            v_n = np.zeros((len(I), len(Ei)))
            v_n[np.searchsorted(I, Ei), np.arange(len(Ei))] = 1.
            v = _linear(len(I), len(O), len(Ei), len(Eo), v_out, v_m, v_n, f"{name} wiring")
            inner = compose_seq(tensor_arrangements(parts[start:start + size]), v)
            pieces.append(replace(inner, label=name))
        start += size

    Eo_all, Ei_all = np.concatenate(exp_out), np.concatenate(exp_in)
    # Between blocks: each exposed input gets what reads it from outside its block.
    u_m = np.zeros((len(Ei_all), len(Eo_all)))
    u_n = Wn[Ei_all] if len(Ei_all) else np.zeros((0, Wn.shape[1]))
    owner_in = np.concatenate([np.full(len(e), k) for k, e in enumerate(exp_in)]) if len(Ei_all) else np.zeros(0, int)
    owner_out = np.concatenate([np.full(len(e), k) for k, e in enumerate(exp_out)]) if len(Eo_all) else np.zeros(0, int)
    for r, (i, kb) in enumerate(zip(Ei_all, owner_in)):
        for c, (j, kc) in enumerate(zip(Eo_all, owner_out)):
            if kb != kc:
                u_m[r, c] = Wm[i, j]
    u_o = Wo[:, Eo_all] if len(Eo_all) else np.zeros((Wo.shape[0], 0))
    u = _linear(len(Ei_all), len(Eo_all), w.in_dim_N, w.out_dim_N, u_o, u_m, u_n, w.label)
    return pieces, u


def factor_linear(parts: Sequence[Arrangement], w: Arrangement,
                  blocks: Sequence[Tuple[int, str]], label: str | None = None) -> Arrangement:
    """``w . (x) parts`` regrouped: ``blocks`` is a list of ``(size, name)``, consecutive runs of
    parts, sizes summing to ``len(parts)``; a block of size one is the part itself.

    Returns ``u . (x)_b block_b`` with ``block_b = v_b . (x)_{i in b} parts_i`` labeled ``name``,
    and the whole labeled ``label`` (``w.label`` if not given).
    """
    pieces, u = _factor(parts, w, blocks)
    return replace(compose_seq(tensor_arrangements(pieces), u), label=label or w.label)


def carry_state(flat_state, nested_init):
    """A state of the flat composite as a state of the nested one: the same numbers, part by part,
    in the nested composite's bracketing (a tensor's state is a pair, so the bracketing of the
    parts shows in the state's shape; the wirings' states are empty)."""
    leaves = [x for x in jax.tree_util.tree_leaves(flat_state) if jnp.size(x)]
    target, tree = jax.tree_util.tree_flatten(nested_init)
    it = iter(leaves)
    out = [next(it) if jnp.size(t) else t for t in target]
    if next(it, None) is not None:
        raise ValueError("carry_state: the flat state has more numbers than the nested one")
    return jax.tree_util.tree_unflatten(tree, out)


def even_blocks(n: int, most: int = 7) -> List[int]:
    """Sizes of ``ceil(n / most)`` nearly equal consecutive blocks covering ``n`` things, each at
    most ``most`` (one block, of size ``n``, when ``n <= most``)."""
    k = -(-n // most)
    base, extra = divmod(n, k)
    return [base + (1 if i < extra else 0) for i in range(k)]


def nest(parts: Sequence[Arrangement], w: Arrangement, name: str = "parts", most: int = 7,
         label: str | None = None, _spans=None) -> Arrangement:
    """``w . (x) parts`` with every box holding at most ``most`` things: consecutive parts are
    grouped into blocks of nearly equal size, named ``name i-j`` by the parts they hold, and the
    blocks grouped again while there are more than ``most`` of them."""
    parts = list(parts)
    spans = _spans or [(i + 1, i + 1) for i in range(len(parts))]
    if len(parts) <= most:
        return replace(compose_seq(tensor_arrangements(parts), w), label=label or w.label)
    sizes = even_blocks(len(parts), most)
    blocks, new_spans, at = [], [], 0
    for s in sizes:
        lo, hi = spans[at][0], spans[at + s - 1][1]
        blocks.append((s, f"{name} {lo}-{hi}" if hi > lo else f"{name} {lo}"))
        new_spans.append((lo, hi))
        at += s
    pieces, u = _factor(parts, w, blocks)
    return nest(pieces, u, name, most, label, new_spans)
