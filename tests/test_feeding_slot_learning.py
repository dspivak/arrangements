"""Paired revisit learning on the library composite with a separator mouth (``feeding_with_slot``).

Ported from the retired hand-typed page model's check (``viewer/food-learning.test.js``, 2026-11-23)
onto the composite itself, since it needs settings the page does not export: the cart held at
zero (``ktau``, ``memoryDrive`` zero) and the eye still (``A``, ``omega``, ``gam``, ``lam`` zero),
its look set by hand toward +4 and away. One casing at +4. The eye looks at it for two units,
away for ten, and back: if the casing is empty again, the learned chance of finding food after
that wait falls; if it was refilled while the eye was away, it rises. Only the return rate at +4
changes; the usual yield does not, and nothing changes while the eye keeps watching. A fresh
outing that carries each branch's memory puts more search weight near +4 after the refill.

What runs in IMC: every run is ``PhiK`` of ``feeding_composed.feeding_with_slot``, four
Runge-Kutta rounds of 0.004, compiled once; the look is set between runs by writing its store.
Beside it: the return rate read off the weights by ``feeding._return_rate`` and the search mass
over [3.5, 4.5] by ``feeding.density_at``, both library functions.
"""
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from arrangements.feeding import CENTRES, DEFAULTS, _return_rate, _search_map, density_at
from arrangements.feeding_composed import feeding_with_slot
from arrangements.interpretation import trivial_omega
from arrangements.pcK import PhiK

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("export_step_js", ROOT / "scripts" / "export_step_js.py")
exporter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exporter)

FIXED = {'A': 0., 'omega': 0., 'gam': 0., 'lam': 0., 'ktau': 0., 'memoryDrive': 0.}
SITE, H = 4., .004
SITE_INDEX = CENTRES.index(int(SITE))


def test_paired_revisit_learning():
    build = lambda memory=None: feeding_with_slot([SITE], [.05], learning=True, stages=4, h=H,
                                                  parameters=FIXED, memory=memory)
    arr = build()
    flat0, unravel = ravel_pytree(arr.P.init)
    found = exporter.stores(arr)
    layout = {s["label"]: s for s in reversed(found)}           # the first store of each name
    at = lambda label: slice(layout[label]["start"], layout[label]["start"] + layout[label]["size"])
    food = layout["food"]["start"]
    look = layout["look"]["start"]
    dynamics = PhiK(arr)
    inputs, loads = (jnp.zeros(0), trivial_omega(0)), (lambda _: (jnp.zeros(2), jnp.zeros(0)))

    @jax.jit
    def run(flat, steps):
        tick = lambda _, s: ravel_pytree(dynamics.with_state(unravel(s)).run_one(inputs, loads)[1])[0]
        return jax.lax.fori_loop(0, steps, tick, flat)

    units = lambda time: round(time / H)
    rate = lambda flat: float(_return_rate(np.asarray(flat)[at("return rate")][SITE_INDEX]))

    def looking(flat, gaze, time):
        flat = np.asarray(flat).copy()
        flat[look] = gaze
        return np.asarray(run(jnp.asarray(flat), units(time)))

    observed = looking(flat0, SITE, 2)
    before = rate(observed)
    assert before == rate(flat0)
    observed = looking(observed, -SITE, 10)
    probability = []
    for refill in (.05, 10.):
        begin = observed.copy()
        begin[food] = refill
        assert rate(begin) == before
        end = looking(begin, SITE, 1)
        after = rate(end)
        assert (after < before) if refill < 1 else (after > before)
        np.testing.assert_array_equal(end[at("usual yield")], observed[at("usual yield")])
        assert end[at("return rate")][1] == 0
        watching = looking(end, SITE, 2)
        assert abs(rate(watching) - after) < 1e-13
        memory = {'usual': end[at("usual yield")], 'returnWeights': end[at("return rate")]}
        fresh = np.asarray(ravel_pytree(build(memory).P.init)[0])
        np.testing.assert_array_equal(fresh[at("return rate")], memory['returnWeights'])
        cleared = looking(fresh, SITE, 2)
        assert rate(cleared) == after
        cleared = looking(cleared, -SITE, 10)
        _, masses = _search_map(jnp.asarray(cleared[at("usual yield")]), jnp.asarray(cleared[at("unexplored")]))
        xs = np.linspace(3.5, 4.5, 101)
        density = np.asarray(density_at(masses, jnp.asarray(xs)))
        probability.append(float(np.sum((density[1:] + density[:-1]) / 2 * np.diff(xs))))
    assert probability[0] < probability[1]
