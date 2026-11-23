"""Write the feeding explorer's diagram and its live readings, read from the library composite.

Book P109 (f), act (4). The diagram is ``arrangements.explore_composite.diagram`` of
``feeding_composed.feeding_with_slot(..., learning=True)`` (with ``--robot``, its slot filled by
the absorber inside a robot-style seeker, P109 (j)); ``readings(state, places, params)`` is the
library's ``explore_composite.readings`` exported by ``export_step_js.javascript``, and
``step(state, places, params)`` the composite's four Runge-Kutta rounds, as ``slot_model``'s.
Nothing on the page that loads the file is typed by hand: the boxes, wires, marks, their kinds
and formulas, every value and every load come from here.

    .venv/bin/python scripts/export_feeding_explorer.py --output viewer/feeding-explorer-data.js
    .venv/bin/python scripts/export_feeding_explorer.py --robot --output viewer/feeding-explorer-robot-data.js
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))


def build_model(count=3, places=None, foods=None, robot=False, h=None, knobs=None):
    """``build(places, params)``, the example arguments and the metadata for the page."""
    import jax.numpy as jnp
    from export_step_js import H, KNOBS, PAGE_FOODS, PAGE_PLACES
    from arrangements.feeding import DEFAULTS
    from arrangements import feeding_composed as fc
    from arrangements.wiring import compose_seq

    h = H if h is None else h
    knobs = KNOBS if knobs is None else knobs
    places = np.asarray(PAGE_PLACES[:count] if places is None else places, dtype=float)
    foods = tuple(PAGE_FOODS[:count] if foods is None else foods)
    params = np.asarray([DEFAULTS[k] for k in knobs], dtype=float)

    def build(places, params):
        p = {**DEFAULTS, **{k: params[i] for i, k in enumerate(knobs)}}
        filler = None
        if robot:
            filler = compose_seq(fc.absorber_filler(p), fc.robot_style(parameters=p, learning=True, stages=4, h=h))
        return fc.feeding_with_slot(tuple(places[i] for i in range(count)), foods, filler=filler, learning=True,
                                    stages=4, h=h, parameters=p)

    metadata = dict(casings=count, places=places.tolist(), foods=list(foods), h=h, knobs=list(knobs),
                    defaults=params.tolist(), robot=robot)
    return build, (places, params), metadata


def export(path, count=3, robot=False, with_step=True):
    from export_step_js import javascript, module
    from arrangements.explore_composite import live_function
    from arrangements.interpretation import trivial_omega
    from arrangements.pcK import PhiK
    from jax.flatten_util import ravel_pytree
    import jax.numpy as jnp

    build, example, metadata = build_model(count, robot=robot)
    began = time.time()
    d, observe, flat0 = live_function(build, example, "Feeding")
    metadata["diagramSeconds"] = round(time.time() - began, 1)
    data = d.static()
    metadata.update(initial=flat0.tolist(), **data)
    parts = [("readings", javascript(observe, (flat0, *example), "readings", fields=("values", "loads", "q")))]
    metadata["readingsSeconds"] = round(time.time() - began, 1)
    if with_step:
        _, unravel = ravel_pytree(build(*example).P.init)
        empty = jnp.zeros(0)
        inputs, loads = (empty, trivial_omega(0)), (lambda _: (jnp.zeros(2), empty))

        def step(flat, places, params):
            dynamics = PhiK(build(places, params))
            return ravel_pytree(dynamics.with_state(unravel(flat)).run_one(inputs, loads)[1])[0]
        parts.append(("step", javascript(step, (flat0, *example), "step")))
    metadata["exportSeconds"] = round(time.time() - began, 1)
    name = "FeedingExplorerRobot" if robot else "FeedingExplorer"
    text = module(parts, name, metadata,
                  "Generated from arrangements.feeding_composed.feeding_with_slot by "
                  f"scripts/export_feeding_explorer.py{' --robot' if robot else ''}: the diagram is "
                  "explore_composite.diagram of the composite; readings and step are the library's own code. "
                  "Nothing here is typed by hand.")
    Path(path).write_text(text)
    return metadata


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--casings", type=int, default=3)
    parser.add_argument("--robot", action="store_true", help="the slot filled by a robot-style seeker")
    parser.add_argument("--no-step", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    m = export(a.output, a.casings, a.robot, not a.no_step)
    print(json.dumps({k: m[k] for k in ("diagramSeconds", "readingsSeconds", "exportSeconds")}),
          "nodes", len(m["scene"]["nodes"]), "wires", len(m["scene"]["wires"]), "bytes", a.output.stat().st_size)
