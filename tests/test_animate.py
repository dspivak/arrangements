"""The recording format and the listener's scene (``arrangements/animate.py``, ``arrangements/scenes.py``).

Each test checks the construction, not a claim about any model: the scene holds what ``listener.run_script``
returned, the page inlines it so it reads back unchanged, and ``validate`` refuses a malformed scene.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import jax
import numpy as np
import pytest

from arrangements.animate import VIEWER, to_html, validate
from arrangements.listener import listener_system, run_script
from arrangements.scenes import listener_live_scene, listener_scene

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def scene():
    return listener_scene()


def test_the_scene_is_the_run(scene):
    """Every lane's placement and belief are the run's own, to the page's rounding (1e-6)."""
    for v in scene["variants"]:
        for lane, wire in zip(v["lanes"], ("oneway", "exact")):
            script = "".join({"kept": "K", "failed, account lined up": "L",
                              "failed, account did not line up": "X"}[c["label"]] for c in v["ticks"]["chips"])
            ps, ys, _ = run_script(listener_system(wire=wire), script)
            box = {b["id"]: b for b in lane["boxes"]}["listener"]
            np.testing.assert_allclose(box["x"], ps, atol=1e-6, rtol=0)
            np.testing.assert_allclose(box["readout"]["values"], ys, atol=1e-6, rtol=0)
    print("scene matches run_script on", len(scene["variants"]), "variants x 2 wires")


def test_the_page_reads_back_the_scene(scene):
    """The inlined JSON parses back to the scene, and the title is filled in."""
    page = to_html(scene)
    m = re.search(r'<script id="scene" type="application/json">(.*?)</script>', page, re.S)
    assert json.loads(m.group(1)) == json.loads(json.dumps(scene))
    assert "<title>The listener</title>" in page and "{{TITLE}}" not in page


def test_validate_refuses_a_short_series(scene):
    bad = json.loads(json.dumps(scene))
    bad["variants"][0]["lanes"][0]["boxes"][1]["x"].pop()
    with pytest.raises(ValueError, match="x needs"):
        validate(bad)


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run the page's model")
def test_the_page_model_is_the_library_run():
    """The page's own listener (viewer/models/listener.js) agrees with listener.run_script on every sample
    list and both wires, to 1e-12, given the library's expected losses (0.4, 1.4, 8) its hearing, scaled by the kernel weight, and its bell-shaped kernel. The page's own default
    derives them from one cost of a broken promise instead, (0.4, 8, 8), at David's word (2026-10-09)."""
    js = (
        "const m = require(process.argv[1]); const out = {};"
        "for (const s of m.samples) for (const w of ['oneway', 'exact']) {"
        "  const r = m.run({...m.defaults, script: s.script, ell: [0.4, 1.4, 8], hear: 'weighted', kernel: 'gaussian'}, w); out[s.script + '/' + w] = [r.ps, r.ys]; }"
        "process.stdout.write(JSON.stringify(out));"
    )
    model = str(VIEWER / "models" / "listener.js")
    out = json.loads(subprocess.run(["node", "-e", js, model], capture_output=True, text=True, check=True).stdout)
    assert len(out) == 6
    for key, (ps_js, ys_js) in out.items():
        script, wire = key.split("/")
        ps, ys, _ = run_script(listener_system(wire=wire), script)
        dp, dy = np.abs(np.array(ps_js) - ps).max(), np.abs(np.array(ys_js) - ys).max()
        print(f"{key}: max |p| {dp:.1e}, max |y| {dy:.1e}")
        assert dp < 1e-12 and dy < 1e-12


def test_the_live_page_carries_its_model():
    page = to_html(listener_live_scene())
    assert "ARR_MODELS.listener" in page and "/*MODELS*/" not in page


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run the page's model")
def test_a_drawn_round_follows_its_kind():
    """The page draws a round from a kind's numbers: 20,000 draws from the unaccountable kind come out kept,
    broken with an account that lines up, and broken with one that does not, near 60%, 8% and 32%."""
    js = (
        "const m = require(process.argv[1]); const n = {K: 0, L: 0, X: 0};"
        "for (let i = 0; i < 20000; i++) n[m.sample(m.defaults, 2)]++;"
        "process.stdout.write(JSON.stringify(n));"
    )
    model = str(VIEWER / "models" / "listener.js")
    n = json.loads(subprocess.run(["node", "-e", js, model], capture_output=True, text=True, check=True).stdout)
    freq = np.array([n["K"], n["L"], n["X"]]) / 20000
    print("drawn from the unaccountable kind:", np.round(freq, 3))
    np.testing.assert_allclose(freq, [0.6, 0.08, 0.32], atol=0.015)
