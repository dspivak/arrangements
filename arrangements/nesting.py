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
from typing import Any, List, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from .arrangement import Arrangement, zero_form
from .wiring import compose_seq, tensor_arrangements, trivial_parameter


def _matrices(w: Arrangement):
    """The matrices of a static wiring's maps: ``out = Wo m``, ``in = Wm m + Wn n``. Read
    eagerly even when called while tracing (a composite built inside a jitted function), since
    the wiring is static."""
    with jax.ensure_compile_time_eval():
        return _matrices_now(w)


def _matrices_now(w: Arrangement):
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


def _supports(fn, sizes, rows):
    """For each result coordinate of ``fn``, the argument coordinates it may read (traced; an
    upper bound when the trace cannot tell, so never too few)."""
    from ._explorer_inputs import coordinate_inputs
    if rows == 0:
        return []
    args = [jnp.zeros(s) for s in sizes]
    (found,) = coordinate_inputs(lambda *xs: jnp.ravel(fn(*xs)), tuple(args), [rows])
    return [set(support) for support, _ in found]


def factor_route(parts: Sequence[Arrangement], w: Arrangement,
                 blocks: Sequence[Tuple[Sequence[int], str]], label: str | None = None,
                 fast: bool = True, block_meta: Sequence[dict] | None = None) -> Arrangement:
    """``w . (x) parts`` regrouped by meaning, for any static wiring ``w`` (its maps may compute:
    gates, rates, sums), with each computation moved into the block whose parts it feeds.

    ``blocks`` lists ``(indices, name)``: the parts (by index into ``parts``) each block holds, in
    the order they are to sit; every part in exactly one block. A block of one part is the part.
    Inside block ``b`` the wiring ``v_b`` computes every input of the block's parts from the
    block's own outputs and from copies of what it reads outside, which are its exposed inputs;
    its exposed outputs are the outputs read outside it. Between the blocks ``u`` only copies
    (each exposed input is one outer input or one exposed output; its transpose adds the loads)
    and computes the outer outputs. Equal to the flat composite because ``v_b`` evaluates ``w``'s
    own input map on the values its rows read (traced supports, so nothing a row reads is
    missing), and ``u`` passes each of those values unchanged.

    The configuration is the parts' in block order. With ``fast`` the result computes by the
    flat composite of the parts in that order (``run_flat``). ``block_meta`` adds meta to each
    block (port names for the explorer).
    """
    parts = list(parts)
    order = [i for idx, _ in blocks for i in idx]
    if sorted(order) != list(range(len(parts))):
        raise ValueError("factor_route: every part in exactly one block")
    a = [p.in_dim_N for p in parts]
    b = [p.out_dim_N for p in parts]
    a_at, b_at = np.cumsum([0] + a), np.cumsum([0] + b)
    A, B, C = int(a_at[-1]), int(b_at[-1]), w.in_dim_N
    q0 = jnp.zeros(0)
    in_sup = _supports(lambda m, n: w.in_f(q0, m, n), [B, C], A)       # indices < B: m; >= B: n
    out_sup = _supports(lambda m: w.out_f(q0, m), [B], w.out_dim_N)
    read_out = set().union(*out_sup) if out_sup else set()

    infos = []
    for idx, name in blocks:
        I = np.concatenate([np.arange(a_at[i], a_at[i + 1]) for i in idx]).astype(int)
        O = np.concatenate([np.arange(b_at[i], b_at[i + 1]) for i in idx]).astype(int)
        Oset = set(O.tolist())
        reads = set().union(*[in_sup[i] for i in I]) if len(I) else set()
        single = len(idx) == 1
        # A block of one part is the part itself; ``u`` computes its inputs, so everything they
        # read is exposed, its own outputs included. A larger block computes its own.
        sources = sorted(reads) if single else sorted(reads - Oset)
        infos.append(dict(idx=list(idx), name=name, I=I, O=O, single=single, sources=sources))
    for k, x in enumerate(infos):
        others = set().union(*[set(y["sources"]) for j, y in enumerate(infos) if j != k or y["single"]])
        x["exposed"] = [j for j in x["O"].tolist() if j in others or j in read_out]
        if x["single"] and len(x["exposed"]) != len(x["O"]):
            x["exposed"] = x["O"].tolist()      # a bare part shows every output it has

    def embed(pairs, size):
        z = jnp.zeros(size)
        for where, values in pairs:
            if len(where) and jnp.size(values):
                z = z.at[jnp.asarray(where)].set(values)
        return z

    def full(x, size):
        """``x``, or zeros of ``size`` when called with no arguments: the explorer evaluates a
        coordinate that reads nothing at an empty argument list (``explore_composite.readings``)."""
        return x if jnp.size(x) == size else jnp.zeros(size)

    pieces = []
    for x in infos:
        if x["single"]:
            pieces.append(parts[x["idx"][0]])
            continue
        O, I, src, exp = x["O"], x["I"], x["sources"], x["exposed"]
        src_m = [s for s in src if s < B]
        src_n = [s - B for s in src if s >= B]
        local_exp = jnp.asarray([int(np.where(O == j)[0][0]) for j in exp], dtype=int)

        def v_out(q, m, local_exp=local_exp, O=O):
            return full(m, len(O))[local_exp] if len(local_exp) else jnp.zeros(0)

        def v_in(q, m, e, O=O, I=I, src_m=src_m, src_n=src_n, src=src):
            m, e = full(m, len(O)), full(e, len(src))
            full_m = embed([(O, m), (src_m, e[:len(src_m)])], B)
            full_n = embed([(src_n, e[len(src_m):])], C)
            return w.in_f(q0, full_m, full_n)[jnp.asarray(I)] if len(I) else jnp.zeros(0)

        v = Arrangement(trivial_parameter(), len(I), len(O), len(src), len(exp), v_out, v_in, zero_form,
                        f"{x['name']} wiring")
        inner = replace(compose_seq(tensor_arrangements([parts[i] for i in x["idx"]]), v), label=x["name"])
        meta = (block_meta or [{}] * len(infos))[len(pieces)]
        pieces.append(replace(inner, meta={**(inner.meta or {}), **meta}) if meta else inner)

    exp_all = [j for x in infos for j in x["exposed"]]
    pos = {j: k for k, j in enumerate(exp_all)}
    width = sum(len(x["I"]) if x["single"] else len(x["sources"]) for x in infos)

    def u_out(q, m):
        return w.out_f(q0, embed([(exp_all, m)], B))

    def u_in(q, m, n):
        m, n = full(m, len(exp_all)), full(n, C)
        vals = []
        for x in infos:
            if x["single"]:     # a bare part's inputs, computed here from what they read
                rows = w.in_f(q0, embed([(exp_all, m)], B), n)
                vals.append(rows[jnp.asarray(x["I"])] if len(x["I"]) else jnp.zeros(0))
            else:               # a block's exposed inputs: copies
                vals.append(jnp.stack([m[pos[s]] if s < B else n[s - B] for s in x["sources"]])
                            if x["sources"] else jnp.zeros(0))
        return jnp.concatenate(vals) if vals else jnp.zeros(0)

    u = Arrangement(trivial_parameter(), width, len(exp_all), C, w.out_dim_N, u_out, u_in, zero_form, w.label)
    nested = replace(compose_seq(tensor_arrangements(pieces), u), label=label or w.label, meta=w.meta)
    if not fast:
        return nested
    return run_flat(nested, _flat_in_order(parts, w, order))


def _leaves(tree):
    if isinstance(tree, int):
        return [tree]
    return [i for child in tree[1] for i in _leaves(child)]


Tree = Union[int, Tuple[str, list]]     # a part's index, or (a box's name, its children)


def factor_tree(parts: Sequence[Arrangement], w: Arrangement, tree: Tree, fast: bool = True) -> Arrangement:
    """``w . (x) parts`` nested as ``tree`` says: a tree is a part's index, or ``(name, children)``
    for a box named ``name`` holding its children; the root's name labels the whole. Each level
    is ``factor_route``; the configuration is the parts' in the tree's left-to-right order."""
    parts = list(parts)

    def build(ps, wiring, node):
        name, children = node
        local = {i: k for k, i in enumerate(ps_index[id(ps)])} if id(ps) in ps_index else None
        blocks = [([local[i] if local else i for i in _leaves(c)], c[0] if not isinstance(c, int) else "")
                  for c in children]
        whole = factor_route(ps, wiring, blocks, label=name, fast=False)
        _, inner, u = whole.construction
        pieces = []
        _flatten_tensor(inner, pieces, len(children))
        out = []
        for piece, child in zip(pieces, children):
            if isinstance(child, int) or all(isinstance(g, int) for g in child[1]):
                out.append(piece)
                continue
            _, sub_inner, v = piece.construction
            sub_parts = []
            _flatten_tensor(sub_inner, sub_parts, len(_leaves(child)))
            ps_index[id(sub_parts)] = _leaves(child)
            out.append(build(sub_parts, v, child))
        return replace(compose_seq(tensor_arrangements(out), u), label=name, meta=whole.meta)

    ps_index = {}
    nested = build(parts, w, tree)
    if not fast:
        return nested
    return run_flat(nested, _flat_in_order(parts, w, _leaves(tree)))


def _flatten_tensor(a, out, count):
    """The ``count`` factors of a left-nested ``tensor_arrangements``, in order."""
    for _ in range(count - 1):
        _, a, last = a.construction
        out.insert(0, last)
    out.insert(0, a)


def _flat_in_order(parts, w, order):
    """``w . (x) parts`` with the parts taken in ``order``: the same composite, its configuration
    and ports permuted (the wiring reads its arguments back in the original order)."""
    a = [p.in_dim_N for p in parts]
    b = [p.out_dim_N for p in parts]
    a_at, b_at = np.cumsum([0] + a), np.cumsum([0] + b)
    perm_b = np.concatenate([np.arange(b_at[i], b_at[i + 1]) for i in order]).astype(int)
    perm_a = np.concatenate([np.arange(a_at[i], a_at[i + 1]) for i in order]).astype(int)
    back_b = np.argsort(perm_b)
    q0 = jnp.zeros(0)
    pb, pa = jnp.asarray(perm_b), jnp.asarray(perm_a)
    bb = jnp.asarray(back_b)
    wp = Arrangement(trivial_parameter(), w.in_dim_M, w.out_dim_M, w.in_dim_N, w.out_dim_N,
                     lambda q, m: w.out_f(q0, m[bb]),
                     lambda q, m, n: w.in_f(q0, m[bb], n)[pa], zero_form, w.label)
    return compose_seq(tensor_arrangements([parts[i] for i in order]), wp)


def run_flat(nested: Arrangement, flat: Arrangement) -> Arrangement:
    """``nested`` computed by ``flat``'s maps: the 'parts explosion' David allowed, 2026-11-23,
    2ac0e890: "i think it's ok to compile the entire thing to one "parts explosion" ODE befroe
    running it if the parentheses are costly to keep around".

    The two are the same morphism of ``arr`` (associativity; ``tests/test_nesting.py`` checks the
    nested composite's own maps against the flat one's), and both take the configuration as one
    vector in the same order, so the flat maps can stand in for the nested ones. The
    construction, which the explorer draws, stays the nested one; the parameter stays the nested
    one, so its state has the nested bracketing."""
    if (nested.P.Q_dim, nested.in_dim_M, nested.out_dim_M, nested.in_dim_N, nested.out_dim_N) != \
            (flat.P.Q_dim, flat.in_dim_M, flat.out_dim_M, flat.in_dim_N, flat.out_dim_N):
        raise ValueError("run_flat: the two arrangements have different shapes")
    return replace(nested, out_f=flat.out_f, in_f=flat.in_f, omega=flat.omega)


def carry_state(flat_state: Any, nested_init: Any) -> Any:
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


# ---------------------------------------------------------------------------
# Counting the boxes inside each box, as the explorer draws them.
# ---------------------------------------------------------------------------

_MARK_PARTS = ("push", "pushes", "spring", "motor")


def _is_wiring(a: Arrangement) -> bool:
    return not a.construction and a.P.Q_dim == 0 and a.omega is zero_form and not (a.meta or {}).get("part")


def _default_label(a: Arrangement) -> bool:
    _, f, g = a.construction
    return a.label == f"({g.label} . {f.label})"


def _is_leaf(a: Arrangement) -> bool:
    return not a.construction or (a.meta or {}).get("part") in _MARK_PARTS


def is_box(a: Arrangement) -> bool:
    """A box is a ``compose_seq`` node whose second factor is a wiring, or a compose node given
    its own label; a compose of two non-wirings under its default label is looked through."""
    if _is_leaf(a) or a.construction[0] != "compose" or (a.meta or {}).get("wrapper"):
        return False
    return (not _default_label(a)) or _is_wiring(a.construction[2])


def _flatten(a: Arrangement, out: list) -> None:
    if _is_leaf(a):
        if not _is_wiring(a):
            out.append(a)
        return
    if is_box(a):
        out.append(a)
        return
    _, f, g = a.construction
    _flatten(f, out)
    _flatten(g, out)


def contents(a: Arrangement) -> List[Arrangement]:
    """What is drawn directly inside box ``a``: boxes and primitive parts, wirings left out."""
    _, f, g = a.construction
    out: List[Arrangement] = []
    _flatten(f, out)
    if not _is_wiring(g):
        _flatten(g, out)
    return out


def drawn(a: Arrangement) -> List[Arrangement]:
    """What ``a`` is drawn as where it sits: itself if it is a box or a part, else the boxes and
    parts it is looked through to (wirings left out)."""
    out: List[Arrangement] = []
    _flatten(a, out)
    return out


def box_name(a: Arrangement) -> str:
    if is_box(a) and _default_label(a):
        return a.construction[2].label
    return a.label or "(unnamed)"


def box_counts(a: Arrangement) -> List[Tuple[str, int]]:
    """Every box under ``a`` (``a`` itself if it is one, else the boxes at its top level, which
    is counted too, as ``"(top level)"``) with how many boxes and parts it holds directly (book task
    well-factored, David, 2026-11-23: "good factorizations should have 2-7 boxes inside"). This is
    not all the explorer draws: it also draws each computation of a box's wiring (a gate, a sum, a
    choice) as an operation node, which this does not count; inside the feeding spatial coupling it
    counted 4 where the explorer drew 16 (3bd2e654, 2026-11-24; David: "many boxes have more than 7
    things in them")."""
    found: List[Tuple[str, int]] = []
    seen = set()

    def visit(b):
        if id(b) in seen:
            return
        seen.add(id(b))
        kids = contents(b)
        found.append((box_name(b), len(kids)))
        for k in kids:
            if is_box(k):
                visit(k)

    if is_box(a):
        visit(a)
    elif a.construction:
        top: List[Arrangement] = []
        _flatten(a, top)
        found.append(("(top level)", len(top)))
        for x in top:
            if is_box(x):
                visit(x)
    return found


def even_blocks(n: int, most: int = 7) -> List[int]:
    """Sizes of ``ceil(n / most)`` nearly equal consecutive blocks covering ``n`` things, each at
    most ``most`` (one block, of size ``n``, when ``n <= most``)."""
    k = -(-n // most)
    base, extra = divmod(n, k)
    return [base + (1 if i < extra else 0) for i in range(k)]


def nest(parts: Sequence[Arrangement], w: Arrangement, name: str = "parts", most: int = 7,
         label: str | None = None, fast: bool = True) -> Arrangement:
    """``w . (x) parts`` with every box holding at most ``most`` things: consecutive parts are
    grouped into blocks of nearly equal size, named ``name i-j`` by the parts they hold, and the
    blocks grouped again while there are more than ``most`` of them.

    With ``fast`` (the default) the result computes by the flat composite's maps (``run_flat``):
    the nesting costs about 1.7 times the flat composite's time a step on a ring of 120 cells and
    a complete graph of 40 when run through its own maps, and nothing when run flat (run,
    2026-11-23, ``.scratch/well-factored-2ac0e890/timing.py`` in the book repo)."""
    nested = _nest(list(parts), w, name, most, label, None)
    if not fast or len(parts) <= most:
        return nested
    return run_flat(nested, compose_seq(tensor_arrangements(list(parts)), w))


def _nest(parts, w, name, most, label, _spans):
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
    return _nest(pieces, u, name, most, label, new_spans)
