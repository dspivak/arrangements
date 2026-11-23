"""The library's regular composites are well-factored and equal their flat forms (book task
well-factored-2ac0e890: every box holds 2 to 7 things, at every level).

Many copies of one part on an index wiring -- a chain, a graph, a Hopfield net, a Kuramoto
network, ... -- are built nested by their library builders (``nesting.nest``,
``nest_groups.nest_groups``). Each check here: every box of the built composite holds 2 to 7
things; the nested composite's own maps (``fast=False``) equal the flat composite's at random
points and over closed steps (``tests/test_nesting.py``'s ``_agree``); and the built (``fast``)
composite, which runs by the flat maps, equals it too.
"""
import jax
import jax.numpy as jnp
import numpy as np

from arrangements.coalgebra import configuration, phase
from arrangements.nesting import nest
from arrangements.wiring import (cap, held_end, chain_wire,compose_chain, compose_graph, compose_seq, graph_wire, held_chain,
                                 tensor_arrangements, _graph_ports, harmonic_vertex)
from tests.chain import harmonic_particle
from tests.test_nesting import _agree, _counts

jax.config.update("jax_enable_x64", True)


def _well(a):
    counts = _counts(a, [])
    assert counts and all(2 <= c <= 7 for c in counts), counts
    return counts


def test_chain():
    parts = [harmonic_particle(phase, 1.0 + .1 * k, .7) for k in range(11)]
    flat = compose_seq(tensor_arrangements(parts), chain_wire(11))
    built = compose_chain(parts)
    assert _well(built) == [2, 6, 5]
    _agree(flat, built)
    _agree(flat, nest(parts, chain_wire(11), "particles", fast=False))


def test_held_chain_closed_steps():
    parts = [harmonic_particle(configuration, 1.3, .4)] * 9
    built = held_chain(parts, .4, configuration)
    _, nested, _ = built.construction          # cap . chain
    _well(nested)
    ends = [held_end(configuration)] + parts + [held_end(configuration, .4)]
    flat = compose_seq(compose_seq(tensor_arrangements(ends), chain_wire(11)), cap())
    _agree(flat, built)


def test_graph():
    V = 10
    edges = [(v, (v + 1) % V) for v in range(V)] + [((v + 1) % V, v) for v in range(V)] + [(0, 5), (5, 0)]
    built = compose_graph(V, edges, 1.3, .7, vdim=2, onsite=lambda q: .1 * jnp.sum(q ** 4))
    _well(built)
    out_lists, in_lists, _ = _graph_ports(V, edges)
    parts = [harmonic_vertex(len(in_lists[v]), len(out_lists[v]), 1.3, [.7] * len(in_lists[v]), vdim=2,
                             onsite=lambda q: .1 * jnp.sum(q ** 4)) for v in range(V)]
    flat = compose_seq(tensor_arrangements(parts), graph_wire(V, edges, vdim=2))
    _agree(flat, built)
    _agree(flat, nest(parts, graph_wire(V, edges, vdim=2), "vertices", fast=False))


def test_hopfield():
    from arrangements.hopfield import hopfield_arrangement
    rng = np.random.default_rng(5)
    N = 8
    W = rng.normal(size=(N, N)); W = (W + W.T) / 2; np.fill_diagonal(W, 0.)
    I = rng.normal(size=N) * .3
    built = hopfield_arrangement(W, I, .05)
    nested = hopfield_arrangement(W, I, .05, fast=False)
    assert _well(built) == _well(nested)
    assert _counts(nested, [])[0] == 2           # neurons, synapses
    _agree(built, nested)                         # ``built`` computes by the flat maps


def test_spot_rings():
    """Sites (a cell with the boxes that serve it) in arcs; nested maps = flat, and the fast ring is the flat one."""
    from arrangements import spot
    N = 10
    u, v = spot.seed(N, 4)
    rings = [(spot.spot_ring, (u, v), spot.ring_wire, [4] * N),
             (spot.pattern_ring, spot.pattern_seed(N, [1.0], width=4), spot.pattern_ring_wire, [5] * N)]
    for build, args, _, site_sizes in rings:
        built, nested = build(*args), build(*args, fast=False)
        counts = _well(nested)
        assert _counts(built, []) == counts and sorted(counts) == sorted([2, 5, 5] + site_sizes), counts
        _agree(built, nested)


def test_light_lattice():
    from arrangements.light_lattice import edge_box, face_box, lattice_of, light_lattice
    N = 3
    parts = [edge_box(1.0) for _ in range(2 * N * N)] + [face_box(.3) for _ in range(N * N)]
    built, nested = light_lattice(N, kappa=.3), lattice_of(N, parts, fast=False)
    counts = _well(nested)
    assert _counts(built, []) == counts and counts[0] == 2, counts
    _agree(built, nested)


def test_commons_open_access():
    from arrangements.commons import Params, catch_coupler, commons, extractor, fan_wire, resource
    from arrangements.nest_groups import nest_groups
    par, N = Params(), 12
    boxes = [resource(par, 90.)] + [extractor(par, .1, f"Ext{i + 1}") for i in range(N)]
    picks = [j for i in range(N) for j in (0, 1 + i)]
    flat_in = compose_seq(tensor_arrangements(boxes), fan_wire(1 + N, picks))
    nested_in = nest_groups(boxes, fan_wire(1 + N, picks), [(1, "resource", "resource"), (N, "extractors", "Ext")],
                            fast=False, label="resource and extractors")
    assert _well(nested_in)[0] == 2
    _agree(flat_in, nested_in)
    flat = compose_seq(flat_in, tensor_arrangements([catch_coupler(f"C{i + 1}") for i in range(N)]))
    _agree(flat, commons(par, N))


def test_network_parameters_box():
    from arrangements.network import drawn_network, fan_box, parameter_box
    ws = [parameter_box(.1 * i, .05, f"w{i}") for i in range(11)]
    flat = compose_seq(tensor_arrangements(ws), fan_box(11, list(range(11)), "copy"))
    _agree(flat, nest(ws, fan_box(11, list(range(11)), "copy"), "parameters", label="parameters", fast=False))
    net = drawn_network(2, 2, 1, jnp.linspace(-.5, .5, 9), .05, jnp.ones((3, 2)), jnp.ones(3))
    a = net
    while a.construction[0] == "compose":                                  # down the layers to the leaves
        a = a.construction[1]
    kids = []

    def flat(x):
        if x.construction and x.construction[0] == "tensor":
            flat(x.construction[1]); flat(x.construction[2])
        else:
            kids.append(x)
    flat(a)
    assert [k.label for k in kids] == ["datum", "parameters", "label"]
    assert _well(kids[1]) == [2, 5, 4]
    assert net.P.Q_dim == 2 + 9 + 1


def test_kuramoto():
    from arrangements.kuramoto import kuramoto_arrangement
    N = 10
    om = np.linspace(-.5, .5, N)
    edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    for parameter, scale in ((configuration, .05), (phase, 1.)):
        built = kuramoto_arrangement(om, edges, .2, sharp_scale=scale, parameter=parameter)
        nested = kuramoto_arrangement(om, edges, .2, sharp_scale=scale, parameter=parameter, fast=False)
        assert _well(built) == _well(nested)
        _agree(built, nested)
