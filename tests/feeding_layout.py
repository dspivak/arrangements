"""Where a feeding composite's stores sit, read by label (book P109 (k)).

Feeding's stores move whenever a part is added (since P109 (k): the organs' batteries, the
satiety store, each casing's lid, the hands' battery), so the feeding tests find a store by
its label, never by a number typed in the test. ``stores`` is ``scripts/export_step_js.py``'s:
the composite's primitive stores in the order of its flat state, read off its construction.
"""
import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

_spec = importlib.util.spec_from_file_location("export_step_js",
                                               Path(__file__).resolve().parents[1] / "scripts" / "export_step_js.py")
exporter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exporter)
stores = exporter.stores


def flat_index(world):
    """``at(label, k)``: where the ``k``-th store so labelled starts in the flat state."""
    layout = stores(world)

    def at(label, k=0):
        return [s for s in layout if s['label'] == label][k]['start']
    return at


def configuration_index(world):
    """``at(label, k)``: the configuration index of the ``k``-th store so labelled.
    ``P.position`` drops momenta, so each flat entry is followed through it."""
    flat, unravel = ravel_pytree(world.P.init)
    marks = np.asarray(world.P.position(unravel(jnp.arange(flat.size, dtype=float))))
    where = {int(round(v)): i for i, v in enumerate(marks)}
    start = flat_index(world)
    return lambda label, k=0: where[start(label, k)]
