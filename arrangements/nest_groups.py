"""Nesting a composite whose parts come in kinds: each kind its own box, nested inside.

Book task ``well-factored-2ac0e890`` (David, 2026-11-23: "good factorizations should have 2-7
boxes inside ... make sure they are well-factored, so that the parts make sense"). Many of the
library's composites are a few kinds of parts in consecutive runs on one linear wiring: a
Hopfield net's neurons then its synapses, Kuramoto's oscillators then its edges, a commons'
resource then its extractors. ``nest_groups`` makes each run of more than one part a box named
for its kind (``neurons``, ``synapses``), nests each such box with ``nesting.nest`` so that it holds
at most seven things (blocks ``neuron 1-7``, ...), and leaves a run of one part as the part. It is
``nesting.factor_linear`` at the top and ``nesting.nest`` inside each block; equal to the flat
composite by associativity in ``arr`` (``tests/test_well_factored_regular.py`` checks the nested
maps against the flat ones). With ``fast`` it computes by the flat composite's maps
(``nesting.run_flat``); ``flat`` may supply that flat composite (built with a balanced tensor
where hundreds of parts would pass Python's recursion limit).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Sequence, Tuple

import jax.numpy as jnp
import numpy as np

from .arrangement import Arrangement, zero_form
from .nesting import factor_linear, nest, run_flat
from .wiring import compose_seq, tensor_arrangements, trivial_parameter


def _tensor_factors(a: Arrangement, count: int):
    out = []
    for _ in range(count - 1):
        _, a, last = a.construction
        out.insert(0, last)
    out.insert(0, a)
    return out


def nest_groups(parts: Sequence[Arrangement], w: Arrangement,
                groups: Sequence[Tuple[int, str, str]], label: str | None = None,
                most: int = 7, fast: bool = True, flat: Arrangement | None = None) -> Arrangement:
    """``w . (x) parts`` regrouped: ``groups`` lists ``(count, box name, part name)``, consecutive
    runs of parts with counts summing to ``len(parts)``. A run of one part stays the part; a longer
    run is a box ``box name`` whose inside is ``nest(run, v, part name)``. A single group is
    ``nest`` itself. At most ``most`` groups."""
    parts = list(parts)
    groups = [g for g in groups if g[0] > 0]
    if sum(c for c, _, _ in groups) != len(parts):
        raise ValueError("nest_groups: counts must sum to the number of parts")
    if len(groups) > most:
        raise ValueError(f"nest_groups: {len(groups)} groups, more than {most}")
    if len(groups) == 1:
        nested = nest(parts, w, groups[0][2], most=most, label=label or groups[0][1], fast=False)
    else:
        top = factor_linear(parts, w, [(c, name) for c, name, _ in groups], label=label)
        _, inner, u = top.construction
        pieces = _tensor_factors(inner, len(groups))
        out, at = [], 0
        for piece, (c, name, part_name) in zip(pieces, groups):
            if c > 1:
                _, block_inner, v = piece.construction
                sub = _tensor_factors(block_inner, c)
                piece = nest(sub, v, part_name, most=most, label=name, fast=False)
            out.append(piece)
            at += c
        nested = replace(compose_seq(tensor_arrangements(out), u), label=label or w.label)
    if not fast:
        return nested
    return run_flat(nested, flat if flat is not None else compose_seq(tensor_arrangements(parts), w))


def _permuted_wiring(parts, w, order):
    """``w`` reading its parts in ``order``: the same static wiring with its two sources permuted."""
    a = [p.in_dim_N for p in parts]
    b = [p.out_dim_N for p in parts]
    a_at, b_at = np.cumsum([0] + a), np.cumsum([0] + b)
    perm_b = np.concatenate([np.arange(b_at[i], b_at[i + 1]) for i in order]).astype(int)
    perm_a = np.concatenate([np.arange(a_at[i], a_at[i + 1]) for i in order]).astype(int)
    bb, pa = jnp.asarray(np.argsort(perm_b)), jnp.asarray(perm_a)
    q0 = jnp.zeros(0)
    return Arrangement(trivial_parameter(), w.in_dim_M, w.out_dim_M, w.in_dim_N, w.out_dim_N,
                       lambda q, m: w.out_f(q0, m[bb]),
                       lambda q, m, n: w.in_f(q0, m[bb], n)[pa], zero_form, w.label)


def nest_sites(parts: Sequence[Arrangement], w: Arrangement, sites: Sequence[Sequence[int]],
               site_name: str, arc_name: str, label: str | None = None, most: int = 7,
               fast: bool = True, flat: Arrangement | None = None) -> Arrangement:
    """``w . (x) parts`` for a linear ``w``, regrouped by site: ``sites[i]`` lists the parts (by
    index) that make site ``i`` (a cell with the boxes that serve it), which becomes a box
    ``"{site_name} {i+1}"``; the sites are then nested in consecutive runs (``nest``, named
    ``"{arc_name} i-j"``). The configuration is the parts' in site order. With ``fast`` the result
    computes by ``flat`` (default the flat composite in site order); a ``flat`` passed in must
    have the same configuration in the same order, which holds when only one part per site
    carries state and those parts keep their order."""
    parts = list(parts)
    order = [i for s in sites for i in s]
    if sorted(order) != list(range(len(parts))):
        raise ValueError("nest_sites: every part in exactly one site")
    ps = [parts[i] for i in order]
    wp = _permuted_wiring(parts, w, order)
    top = factor_linear(ps, wp, [(len(s), f"{site_name} {i + 1}") for i, s in enumerate(sites)])
    _, inner, u = top.construction
    pieces = _tensor_factors(inner, len(sites))
    nested = nest(pieces, u, arc_name, most=most, label=label or w.label, fast=False)
    if not fast:
        return nested
    return run_flat(nested, flat if flat is not None else compose_seq(tensor_arrangements(ps), wp))
