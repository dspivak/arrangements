"""The feeding explorer's diagram, read from the library composite (book P109 (f), act (4)).

``explore_composite.diagram`` flattens the composite's construction into boxes, stores, static
maps and field marks; ``readings`` gives every wire's value and load. These tests hold it to
the composite: the marks' one-forms, pulled back along the wires, add up to the composite's own
inclination at every store (so no field is missing, doubled or misplaced); every push mark is
wired to exactly the inputs its traced code reads; every written formula is the code's; and a
spring is drawn as one spring, its two ends the wires whose difference its stretch is.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arrangements import explore_composite as ec
from arrangements import feeding_composed as fc
from arrangements import feeding_parts as fp
from arrangements.explore import _read
from arrangements.wiring import compose_seq


def inclination_from_wires(d, q, r):
    """The loads on each store's wires, pulled back to its configuration."""
    xi = np.zeros(q.shape[0])
    for x in d.branches:
        s = d.signals[x["sid"]]
        if s["kind"] != "state":
            continue
        prim, start = d.tree[s["path"]], d.qstart[s["path"]]
        load = np.zeros(s["width"])
        load[x["indices"]] += np.asarray(r["loads"][x["eps"]:x["eps"] + x["width"]])
        qs = q[start:start + prim.P.Q_dim]
        xi[start:start + prim.P.Q_dim] += np.asarray(jax.vjp(lambda z: prim.out_f(z, ec.EMPTY), qs)[1](jnp.asarray(load))[0])
    return xi


def sampled(arr, count, scale=.2, seed=0):
    q0 = _read(arr.P, arr.P.init, None, 0)
    rng = np.random.default_rng(seed)
    return [q0 + jnp.asarray(rng.normal(scale=scale, size=q0.shape)) for _ in range(count)]


def check_adds_up(arr, count):
    d = ec.diagram(arr)
    read = jax.jit(lambda q: ec.readings(d, arr, q))
    for q in sampled(arr, count):
        r = read(q)
        got = inclination_from_wires(d, q, r)
        want = np.asarray(arr.omega(q, jnp.zeros(arr.out_dim_M), jnp.zeros(arr.in_dim_N))[0])
        assert np.allclose(got, want, rtol=1e-9, atol=1e-9 * max(1., np.abs(want).max())), \
            np.abs(got - want).max()
    return d


def test_a_casings_marks_add_up_to_its_field():
    d = check_adds_up(fc.source(4., 10., hinge=False), 3)
    kinds = [(n["label"], n.get("rubric")) for n in d.nodes if n["kind"] == "field"]
    assert ("Gravity", "cost") in kinds
    assert sum(1 for _, k in kinds if k == "spring") == 4   # the air's, the ground's, its damper's two


@pytest.fixture(scope="module")
def feeding():
    arr = fc.feeding_with_slot(learning=True)
    return arr, check_adds_up(arr, 2)


def test_feedings_marks_add_up_to_its_field(feeding):
    arr, d = feeding
    assert len(d.marks) > 50


def test_every_push_mark_reads_what_its_code_reads(feeding):
    arr, d = feeding
    by_node = {n["id"]: n for n in d.nodes}
    for mark in d.marks:
        if mark["term"] is None or mark.get("mask") is not None:
            continue
        meta = d.tree[mark["path"]].meta
        reads = fp.term_reads(meta, mark["term"])
        assert set(mark["read"]) == set(reads["c"]) | set(reads["h"]), by_node[mark["node"]]["label"]
        wired = {d.branches[it[1]]["target"] for it in mark["items"] if it[0] == "br"}
        assert wired <= {mark["node"]}
        # a letter on the drawn form for each input it reads that arrives on a wire, and none other
        form = by_node[mark["node"]].get("form")
        if form:
            drawn = set(form["letters"].values()) | set(form["constants"])
            assert set(mark["read"]) <= drawn, by_node[mark["node"]]["label"]
            assert set(form["letters"].values()) <= set(mark["read"])


def test_every_written_formula_is_the_codes(feeding):
    arr, d = feeding
    checked = 0
    for a in ec.nodes_by_path(arr).values():
        if (a.meta or {}).get("terms"):
            fp.check_formulas(a.meta)
            checked += 1
    assert checked > 30


def test_a_spring_is_drawn_as_one_spring_between_its_ends(feeding):
    arr, d = feeding
    springs = [n for n in d.nodes if n.get("glyph") == "spring"]
    assert len(springs) == 22    # per casing 4, per attachment 3, the wheel's dragger
    attachment = next(n for n in springs if n["label"] == "Attachment 1 spring")
    # its stretch is the casing's height minus the tip: the two wires, and the gate on its third terminal
    assert set(attachment["form"]["letters"].values()) == {"x", "y", "g"}
    assert attachment["third"].startswith("Spring gate")
    assert not [n for n in d.nodes if n["parent"] == attachment["id"]]


def test_the_boxes_are_the_ones_the_code_names(feeding):
    arr, d = feeding
    top = {n["id"] for n in d.nodes if n["parent"] == "root"}
    assert {"seeker-with-a-slot", "spatial-coupling", "pile-on-the-ground"} <= top
    assert sum(1 for i in top if i.startswith("casing")) == 3
    inside = {n["id"] for n in d.nodes if n["parent"] == "seeker-with-a-slot"}
    assert {"body-and-motors", "mouth", "husk-collector", "gut-wall"} <= inside
    motors = [n for n in d.nodes if n.get("glyph") == "motor"]
    assert len(motors) == 3 + 1 + 2 + 2 + 9      # latches, drive, grip and lift, the eye's two, the learners
    for m in motors:
        assert m["turns"] and m["powered"] == "Battery"


def test_a_robot_in_the_slot_is_drawn_inside_the_seeker():
    arr = fc.feeding_with_slot(learning=True, filler=compose_seq(fc.absorber_filler(), fc.robot_style(learning=True)))
    d = ec.diagram(arr)
    parent = {n["id"]: n["parent"] for n in d.nodes}
    assert parent["robot-style"] == "seeker-with-a-slot"
    inner = {n["id"] for n in d.nodes if n["parent"] == "robot-style"}
    assert {"seeker-with-a-slot-2", "inner-spatial-coupling"} <= inner
    # the gut wall fills the inner seeker's slot, and is drawn inside it, as the outer seeker holds the robot
    assert parent["gut-wall"] == "seeker-with-a-slot-2"
    # the inner seeker is drawn with the same parts as the outer one
    outer_kinds = sorted(n["label"] for n in d.nodes if n["parent"] == "seeker-with-a-slot" and n["kind"] == "box"
                         and n["id"] != "robot-style")
    inner_kinds = sorted(n["label"] for n in d.nodes if n["parent"] == "seeker-with-a-slot-2" and n["kind"] == "box")
    assert outer_kinds == inner_kinds
    q = sampled(arr, 1)[0]
    r = jax.jit(lambda q: ec.readings(d, arr, q))(q)
    want = np.asarray(arr.omega(q, jnp.zeros(0), jnp.zeros(0))[0])
    assert np.allclose(inclination_from_wires(d, q, r), want, rtol=1e-9, atol=1e-9 * max(1., np.abs(want).max()))
