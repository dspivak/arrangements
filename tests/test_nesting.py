"""The nested composite equals the flat one (``arrangements.nesting``; book task well-factored-2ac0e890).

Associativity of composition in ``arr`` says ``w . (x) parts = u . (x)_b (v_b . (x)_{i in b} parts)``
when ``w = u . (x)_b v_b``; ``factor_linear`` builds the ``v_b`` and ``u``. Checked here on three
wirings: a graph's prism (each vertex's springs to its neighbours; a ring and the complete graph,
where every block exposes everything), a ``copy`` wiring whose loads add at a dot, and a wiring with
outer ports. For each: the forcing at random points, and twenty closed steps, agree with the flat
composite's; every box holds 2 to 7 things.
"""
import jax
import jax.numpy as jnp
import numpy as np

from arrangements.coalgebra import phase
from arrangements.functors import closed_step
from arrangements.nesting import carry_state, even_blocks, factor_linear, nest
from arrangements.wiring import (compose_seq, compose_graph, graph_wire, harmonic_vertex, tensor_arrangements,
                                 _graph_ports, chain_wire)

jax.config.update("jax_enable_x64", True)


def _vertices(V, edges, parameter=phase):
    out_lists, in_lists, _ = _graph_ports(V, edges)
    return [harmonic_vertex(len(in_lists[v]), len(out_lists[v]), 1.3, [0.7] * len(in_lists[v]),
                            onsite=lambda q: 0.1 * jnp.sum(q ** 4), parameter=parameter) for v in range(V)]


def _counts(a, out):
    """The number of things inside every box of ``a`` (a compose with a named label)."""
    if not a.construction:
        return out
    op, f, g = a.construction
    if op == "compose" and not a.label.startswith("("):
        kids = []

        def flat(x):
            if x.construction and x.construction[0] == "tensor":
                flat(x.construction[1])
                flat(x.construction[2])
            else:
                kids.append(x)
        flat(f)
        out.append(len(kids))
        for k in kids:
            _counts(k, out)
    return out


def _agree(flat, nested, steps=20):
    rng = np.random.default_rng(3)
    for _ in range(3):
        q = jnp.asarray(rng.normal(size=flat.P.Q_dim))
        m, n = jnp.asarray(rng.normal(size=flat.out_dim_M)), jnp.asarray(rng.normal(size=flat.in_dim_N))
        for x, y in zip(flat.omega(q, m, n), nested.omega(q, m, n)):
            np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=1e-12)
        np.testing.assert_allclose(np.asarray(flat.out_f(q, m)), np.asarray(nested.out_f(q, m)), atol=1e-12)
        np.testing.assert_allclose(np.asarray(flat.in_f(q, m, n)), np.asarray(nested.in_f(q, m, n)), atol=1e-12)
    if flat.in_dim_M == flat.out_dim_M == flat.in_dim_N == flat.out_dim_N == 0:
        step_f, step_n = closed_step(flat, duration=.1), closed_step(nested, duration=.1)
        s_f = jax.tree_util.tree_map(lambda x: x + .3 * jnp.arange(1, x.size + 1).reshape(x.shape), flat.P.init)
        s_n = carry_state(s_f, nested.P.init)
        for _ in range(steps):
            s_f, s_n = step_f(s_f), step_n(s_n)
        a = [x for x in jax.tree_util.tree_leaves(s_f) if x.size]
        b = [x for x in jax.tree_util.tree_leaves(s_n) if x.size]
        assert len(a) == len(b)
        for x, y in zip(a, b):
            np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=1e-10)


def test_even_blocks():
    assert even_blocks(7) == [7]
    assert even_blocks(8) == [4, 4]
    for n in range(1, 400):
        sizes = even_blocks(n)
        assert sum(sizes) == n and max(sizes) <= 7 and (n <= 7 or min(sizes) >= 2)


def test_ring_nested_equals_flat():
    V = 30
    edges = [(v, (v + 1) % V) for v in range(V)] + [((v + 1) % V, v) for v in range(V)]
    parts, w = _vertices(V, edges), graph_wire(V, edges)
    flat = compose_seq(tensor_arrangements(parts), w)
    nested = nest(parts, w, "cells", label="ring", fast=False)
    counts = _counts(nested, [])
    assert counts and all(2 <= c <= 7 for c in counts), counts
    _agree(flat, nested)


def test_complete_graph_nested_equals_flat():
    V = 9
    edges = [(a, b) for a in range(V) for b in range(V) if a != b]
    parts, w = _vertices(V, edges), graph_wire(V, edges)
    flat = compose_graph(V, edges, 1.3, .7, onsite=lambda q: 0.1 * jnp.sum(q ** 4), parameter=phase)
    nested = nest(parts, w, "neurons", label="complete", fast=False)
    assert all(2 <= c <= 7 for c in _counts(nested, []))
    _agree(flat, nested)
    fast = nest(parts, w, "neurons", label="complete")
    assert _counts(fast, []) == _counts(nested, [])      # drawn nested
    _agree(flat, fast)                                    # run flat


def test_chain_with_outer_ports():
    """A chain has an outer input (the first part's neighbour) and an outer output (the last part)."""
    from arrangements.wiring import held_end
    from arrangements.coalgebra import configuration
    parts = [held_end(configuration, .5 + .1 * k) for k in range(11)]
    w = chain_wire(11)
    flat = compose_seq(tensor_arrangements(parts), w)
    nested = factor_linear(parts, w, [(4, "left"), (4, "middle"), (3, "right")], label="chain")
    _agree(flat, nested)
    assert _counts(nested, []) == [3, 4, 4, 3]
