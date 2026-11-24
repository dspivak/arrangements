"""Feeding's nested composites are the flat ones they regroup (book task well-factored, David,
2026-11-23: "good factorizations should have 2-7 boxes inside").

* ``feeding_composed.mouth_coupling``, nested by hand as ``choose . (handling casing 1 (x) ... (x)
  hands power) . telling the seeker``, against ``_mouth_coupling_flat``: the same interface, readout, inputs
  sent and inclination field, the configuration matched store by store by label; and a closed
  run of ``feeding_with_slot`` with each, the same at every store.
* ``body`` and ``seeker_with_slot``, nested by ``nesting.factor_tree`` and run by the flat maps
  (``run_flat``): their own nested construction gives the same maps.
* Every box of the composite feeding holds at most seven boxes, and every box of the two
  explorer data files draws at most seven things (boxes, stores, marks, static maps).

What runs in IMC: every side is a library arrangement; fields are their own ``omega``; the run is
``pcK.PhiK``.
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from arrangements import feeding_composed as fc
from arrangements.interpretation import trivial_omega
from arrangements.pcK import PhiK
from arrangements.wiring import compose_seq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from export_step_js import stores  # noqa: E402

jax.config.update("jax_enable_x64", True)
EMPTY = jnp.zeros(0)
SKIP = {"spatial coupling", "inner spatial coupling", "hands power", "choose",
        # the boxes inside a handling box (book task well-factored, aee89bf9)
        "telling the seeker", "attaching", "latching", "taking", "the lid", "spilling", "spilling and laying out"}


def _key(s):
    return tuple(x for x in s["path"] if x not in SKIP and not x.startswith("handling casing")
                 and not x.startswith("casings ") and not x.endswith(" wiring")) + (s["label"],)


def _permutation(old, new):
    """``new``'s configuration as entries of ``old``'s, matched store by store by label path."""
    where = {}
    for s in stores(old):
        where.setdefault(_key(s), []).append(s)
    perm = []
    for s in stores(new):
        o = where[_key(s)].pop(0)
        assert o["size"] == s["size"]
        perm += list(range(o["start"], o["start"] + o["size"]))
    assert not any(where.values())
    return np.asarray(perm)


def _same_coupling(n, feeder):
    new, old = fc.mouth_coupling(n, feeder=feeder), fc._mouth_coupling_flat(n, feeder=feeder)
    for a in ("in_dim_M", "out_dim_M", "in_dim_N", "out_dim_N"):
        assert getattr(old, a) == getattr(new, a)
    perm = jnp.asarray(_permutation(old, new))
    rng = np.random.default_rng(5 + n)
    for _ in range(4):
        q = jnp.asarray(rng.normal(size=old.P.Q_dim))
        m = jnp.asarray(rng.normal(size=old.out_dim_M)).at[0].set(rng.uniform(.5, 6.))
        x = jnp.asarray(rng.normal(size=old.in_dim_N))
        a, b = old.omega(q, m, x), new.omega(q[perm], m, x)
        np.testing.assert_allclose(b[0], a[0][perm], rtol=0, atol=1e-12)
        for u, v in zip(a[1:], b[1:]):
            np.testing.assert_allclose(v, u, rtol=0, atol=1e-12)
        np.testing.assert_allclose(new.in_f(q[perm], m, x), old.in_f(q, m, x), rtol=0, atol=1e-12)
        np.testing.assert_allclose(new.out_f(q[perm], m), old.out_f(q, m), rtol=0, atol=1e-12)


def test_the_nested_coupling_is_the_flat_one():
    """One, two, three casings, and seven, where the handling boxes are grouped."""
    for n in (1, 2, 3, 7):
        _same_coupling(n, False)


def test_the_nested_inner_coupling_is_the_flat_one():
    for n in (1, 2, 3):
        _same_coupling(n, True)


def test_a_closed_run_is_the_same():
    """``feeding_with_slot`` with two casings, 150 Runge-Kutta steps of 0.004, nested coupling
    against flat: every store ends the same."""
    places, foods = (1., -3.), (10., 10.)
    new = fc.feeding_with_slot(places, foods, learning=True, stages=4, h=.004)
    _, world, _ = new.construction
    old = compose_seq(world, fc._mouth_coupling_flat(2, stages=4, h=.004, battery=6.))
    perm = _permutation(old, new)
    inputs, loads = (EMPTY, trivial_omega(0)), (lambda _: (jnp.zeros(2), EMPTY))
    ends = []
    for arr in (old, new):
        # The flat coupling is the comparison, over seven by design: run unwatched.
        dynamics = PhiK(arr) if arr is new else getattr(PhiK, "__wrapped__", PhiK)(arr)

        @jax.jit
        def run(state):
            return jax.lax.fori_loop(0, 150, lambda _, s: dynamics.with_state(s).run_one(inputs, loads)[1], state)

        ends.append(np.asarray(ravel_pytree(run(dynamics.state))[0]))
    start = np.asarray(ravel_pytree(old.P.init)[0])
    assert np.abs(ends[0] - start).max() > 1e-3
    np.testing.assert_allclose(ends[1], ends[0][perm], rtol=0, atol=1e-10)


def test_casings_boxed_at_the_top_are_the_flat_world():
    """At four and six casings the top level would hold eight and ten (absorber, seeker, the
    casings, pile, coupling), so the casings sit in one box ``casings 1-n``: every box then holds
    at most seven, and 100 steps of the closed run end the same at every store as with the
    casings side by side at the top."""
    from arrangements import feeding_recursive as fr
    from arrangements.feeding import DEFAULTS
    from arrangements.nesting import box_counts
    from arrangements.wiring import tensor_arrangements
    places, foods = (1., -3., 5., -7., 9., -11.), (10.,) * 6
    inputs, loads = (EMPTY, trivial_omega(0)), (lambda _: (jnp.zeros(2), EMPTY))
    step = dict(stages=4, h=.004)
    for n in (4, 6):
        new = fc.feeding_with_slot(places[:n], foods[:n], **step)
        counts = box_counts(new)
        assert not [(label, k) for label, k in counts if k > 7], counts
        assert dict(counts)[f"casings 1-{n}"] == n and dict(counts)["(top level)"] == 5
        p = dict(DEFAULTS)
        robot = compose_seq(fc.absorber_filler(p), fc.seeker_with_slot(parameters=p, battery=3., **step))
        casings = [fc.source(x, f, parameters=p, hinge=False, **step) for x, f in zip(places[:n], foods[:n])]
        _, _, coupling = new.construction
        old = compose_seq(tensor_arrangements([robot, *casings, fr.pile(0., "pile on the ground", **step)]),
                          coupling)
        perm = _permutation(old, new)
        assert (perm == np.arange(len(perm))).all()
        ends = []
        for arr in (old, new):
            # The flat world is the comparison, over seven at the top by design: run unwatched.
            dynamics = PhiK(arr) if arr is new else getattr(PhiK, "__wrapped__", PhiK)(arr)

            @jax.jit
            def run(state):
                return jax.lax.fori_loop(0, 100, lambda _, s: dynamics.with_state(s).run_one(inputs, loads)[1], state)

            ends.append(np.asarray(ravel_pytree(run(dynamics.state))[0]))
        assert np.abs(ends[0] - np.asarray(ravel_pytree(old.P.init)[0])).max() > 1e-3
        np.testing.assert_allclose(ends[1], ends[0], rtol=0, atol=1e-12)


def _same_as_its_construction(arr):
    _, f, g = arr.construction
    nested = compose_seq(f, g)
    rng = np.random.default_rng(2)
    for _ in range(3):
        q = jnp.asarray(rng.normal(size=arr.P.Q_dim))
        m = jnp.asarray(rng.normal(size=arr.out_dim_M))
        x = jnp.asarray(rng.normal(size=arr.in_dim_N))
        for u, v in zip(arr.omega(q, m, x), nested.omega(q, m, x)):
            np.testing.assert_allclose(v, u, rtol=0, atol=1e-12)
        np.testing.assert_allclose(nested.out_f(q, m), arr.out_f(q, m), rtol=0, atol=1e-12)


def test_the_nested_body_and_seeker_are_the_flat_ones():
    _same_as_its_construction(fc.body(learning=True, organs=True))
    _same_as_its_construction(fc.seeker_with_slot(learning=True))


def test_no_feeding_box_holds_more_than_seven():
    """Boxes and parts, as ``nesting.box_counts`` counts them (the operations the explorer also
    draws for the wirings' computations are counted by the next test): the spatial coupling holds
    telling the seeker, the three handling boxes and the hands' power, a handling box its six
    boxes, the body five boxes, the seeker four (its slot, a bare identity, is drawn as wires)."""
    from arrangements.nesting import box_counts
    counts = box_counts(fc.feeding_with_slot(learning=True))
    over = [(label, k) for label, k in counts if k > 7]
    assert not over, over
    labels = dict(counts)
    assert labels["spatial coupling"] == 5 and labels["handling casing 1"] == 6
    assert labels["body and motors"] == 5 and labels["seeker with a slot"] == 4


VIEWER = Path(__file__).resolve().parent.parent / "viewer"


def drawn_counts(path):
    """For each box of an explorer data file, how many nodes it draws directly inside: boxes,
    states, fields and operations (static maps), as David sees them (book task well-factored,
    David, 2026-11-24: "many boxes have more than 7 things in them")."""
    import json
    from collections import Counter
    text, decoder, at = path.read_text(), json.JSONDecoder(), 0
    kids, labels = {}, {}
    while (at := text.find('{"id": ', at)) >= 0:
        try:
            node, _ = decoder.raw_decode(text, at)
        except json.JSONDecodeError:
            at += 1
            continue
        if "kind" in node and "parent" in node:
            kids.setdefault(node["parent"], Counter())[node["kind"]] += 1
            labels[node["id"]] = node.get("label", node["id"])
        at += 1
    return {labels.get(p, p): c for p, c in kids.items()}


def test_the_explorer_draws_at_most_seven_in_a_box():
    """Both explorer data files, as generated (``scripts/export_feeding_explorer.py``): no box
    draws more than seven things."""
    for name in ("feeding-explorer-data.js", "feeding-explorer-robot-data.js"):
        counts = drawn_counts(VIEWER / name)
        over = {box: dict(c) for box, c in counts.items() if sum(c.values()) > 7}
        assert not over, (name, over)
