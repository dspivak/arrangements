"""Feeding's nested composites are the flat ones they regroup (book task well-factored, David,
2026-11-23: "good factorizations should have 2-7 boxes inside").

* ``feeding_composed.mouth_coupling``, nested by hand as ``choose . (handling casing 1 (x) ... (x)
  hands power) . split``, against ``_mouth_coupling_flat``: the same interface, readout, inputs
  sent and inclination field, the configuration matched store by store by label; and a closed
  run of ``feeding_with_slot`` with each, the same at every store.
* ``body`` and ``seeker_with_slot``, nested by ``nesting.factor_tree`` and run by the flat maps
  (``run_flat``): their own nested construction gives the same maps.
* Every box of the composite feeding holds at most seven boxes.

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
SKIP = {"spatial coupling", "inner spatial coupling", "hands power", "choose"}


def _key(s):
    return tuple(x for x in s["path"] if x not in SKIP and not x.startswith("handling casing")
                 and not x.startswith("casings ")) + (s["label"],)


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
    _same_coupling(2, False)


def test_the_nested_inner_coupling_is_the_flat_one():
    _same_coupling(2, True)


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
        dynamics = PhiK(arr)

        @jax.jit
        def run(state):
            return jax.lax.fori_loop(0, 150, lambda _, s: dynamics.with_state(s).run_one(inputs, loads)[1], state)

        ends.append(np.asarray(ravel_pytree(run(dynamics.state))[0]))
    start = np.asarray(ravel_pytree(old.P.init)[0])
    assert np.abs(ends[0] - start).max() > 1e-3
    np.testing.assert_allclose(ends[1], ends[0][perm], rtol=0, atol=1e-10)


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
    """As the explorer draws it (``nesting.box_counts``): the spatial coupling holds the three
    handling boxes and the hands' power, the body five boxes, the seeker four (its slot, a bare
    identity, is drawn as wires)."""
    from arrangements.nesting import box_counts
    counts = box_counts(fc.feeding_with_slot(learning=True))
    over = [(label, k) for label, k in counts if k > 7]
    assert not over, over
    labels = dict(counts)
    assert labels["spatial coupling"] == 4
    assert labels["body and motors"] == 5 and labels["seeker with a slot"] == 4
