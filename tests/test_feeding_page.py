"""The feeding page's committed model files are the library composite's, as it is now.

``viewer/feeding-slot-N.js`` (N = 1 .. 6 casings) are written by ``scripts/export_step_js.py --slot
--casings N`` from ``feeding_composed.feeding_with_slot``; ``viewer/feeding.html`` steps with them.
This test runs each committed file under node at three states (the composite's start, the state
after 300 library steps, and a perturbed one at other places and knobs) and requires its step to
equal the library's to 1e-12, its ``start`` to give the composite's own starting state, and (at
three casings, the page's) every field ``observe`` returns to equal the library's. A change to
the library's parts that the files do not carry fails here: rerun the exporter for each N.
(``tests/test_export_step_js.py`` checks the exporter itself on fresh exports.)
"""
import importlib.util
import json
import subprocess
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "viewer"
DRIVER = VIEWER / "exported-step-driver.js"
_spec = importlib.util.spec_from_file_location("export_step_js", ROOT / "scripts" / "export_step_js.py")
exporter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exporter)


def node(path, calls, fn, tmp_path):
    data = tmp_path / f"{fn}.json"
    data.write_text(json.dumps([[np.asarray(a).tolist() for a in c] for c in calls]))
    out = subprocess.check_output(["node", str(DRIVER), str(path), str(data), "--args", "--fn", fn], text=True)
    return json.loads(out)["results"]


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 6])
def test_committed_module_is_the_library_step(count, tmp_path):
    functions, examples, metadata = exporter.slot_model(count)
    path = VIEWER / f"feeding-slot-{count}.js"
    initial, places, params = examples["step"]
    step = jax.jit(functions["step"])
    s = initial
    for _ in range(300):
        s = step(s, places, params)
    rng = np.random.default_rng(count)
    knobs = params * (1 + .1 * rng.standard_normal(params.shape))
    knobs[metadata["knobs"].index("searchSeed")] = 99
    calls = [(initial, places, params), (np.asarray(s), places, params),
             (np.asarray(s) + .01 * rng.standard_normal(s.shape), np.sort(rng.uniform(-8, 8, count)), knobs)]
    expected = np.asarray([np.asarray(step(*c)) for c in calls])
    assert np.all(np.isfinite(expected))
    np.testing.assert_allclose(np.asarray(node(path, calls, "step", tmp_path)), expected, rtol=0, atol=1e-12)
    started = node(path, [examples["start"]], "start", tmp_path)[0]
    np.testing.assert_array_equal(np.asarray(started), initial)
    if count == 3:
        observe = jax.jit(functions["observe"])
        for c, got in zip(calls, node(path, calls, "observe", tmp_path)):
            for field, value in zip(exporter.OBSERVED, observe(*c)):
                np.testing.assert_allclose(got[field], np.asarray(value), rtol=0, atol=1e-12)
