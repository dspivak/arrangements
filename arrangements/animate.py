"""Animate a run: one recording format, one player page (book repo, 2026-10-09).

David (book repo, 2026-10-09): "Can we make animations for all of the stuff in this um, repo? ... is there a
consistent way of making an animation? ... if there's an attention thing, maybe they move around in space,
otherwise they kind of stay where they are. And um, they have different states that can register on their
bodies somehow." This module is that consistent way. A run is recorded as a SCENE, a plain dict that is
JSON as it stands, and ``write_html`` inlines the scene into ``viewer/player.html``, which plays it in a
browser: play, pause, step, scrub, and switch between variants.

A scene says nothing about how the run was computed; it holds what the run did, step by step:

* ``variants``: the runs a viewer can switch between (one script each, say), each a list of ``lanes``;
* a lane: one arrangement run, drawn as a band with a ``space`` (the attention parent's placements, a line
  for now); lanes in one variant are shown together, so two arms of a comparison play side by side;
* a ``box``: a body at ``x`` (a number for a box with no placement, which stays where it is, or one number
  per step for a box on an attention parent) carrying a ``readout`` drawn on it, a ``pie`` (a belief or any
  distribution, one row per step) or a ``meter`` (a scalar between ``min`` and ``max``, one per step);
* a ``link``: a wire from one box to another drawn with one strength per step in ``[0, 1]`` (a kernel
  weight, how much the one counts on the other);
* an ``event``: what crossed a wire on a step, drawn as a flash on the box that sent it and, if ``to`` is
  given, on the box that received it, at the link's strength on that step;
* ``ticks``: every how many steps a round begins, and the round's chip (``label``, ``color``) on the
  timeline, so a viewer sees the script the environment supplied.

A LIVE scene instead names a model, ``{"live": {"model": "listener"}}``: the page runs
``viewer/models/<model>.js`` itself, so a viewer can change the model's inputs (sliders, and which outcome
each round has) and watch the run computed again. A model file defines ``ARR_MODELS[name]`` with its
defaults, samples, controls and a ``build(params)`` that returns the variant shape above; each model file is
checked against the library's own run of the same arrangement (``tests/test_animate.py``).

``validate`` checks the shapes; ``write_html`` validates, then writes one self-contained HTML file.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np

VIEWER = Path(__file__).resolve().parent.parent / "viewer"
PLAYER = VIEWER / "player.html"
PLACEHOLDER = "/*SCENE*/null"
MODELS_PLACEHOLDER = "/*MODELS*/"
STAGE_PLACEHOLDER = "/*STAGE*/"
LIVE = VIEWER / "live.html"


def series(a) -> list:
    """A numpy or jax array, or a list, as nested lists of floats rounded for the page."""
    return np.round(np.asarray(a, dtype=float), 6).tolist()


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def validate(scene: dict[str, Any]) -> None:
    """Check that every per-step series has one entry per step and every reference names a box."""
    if "live" in scene:
        _check((VIEWER / "models" / f"{scene['live']['model']}.js").exists(),
               f"no model file viewer/models/{scene['live']['model']}.js")
        return
    T = scene["steps"]
    _check(isinstance(T, int) and T > 0, "steps must be a positive integer")
    _check(len(scene["variants"]) > 0, "a scene needs at least one variant")
    for v in scene["variants"]:
        for chip in v.get("ticks", {}).get("chips", []):
            _check({"label", "color"} <= set(chip), f"variant {v['id']}: a chip needs label and color")
        for lane in v["lanes"]:
            where = f"variant {v['id']}, lane {lane['name']!r}"
            ids = {b["id"] for b in lane["boxes"]}
            for b in lane["boxes"]:
                x = b["x"]
                _check(isinstance(x, (int, float)) or len(x) == T, f"{where}, box {b['id']}: x needs {T} entries")
                r = b.get("readout")
                if r is None:
                    continue
                _check(len(r["values"]) == T, f"{where}, box {b['id']}: readout needs {T} entries")
                if r["kind"] == "pie":
                    n = len(r["labels"])
                    _check(len(r["colors"]) == n, f"{where}, box {b['id']}: one color per label")
                    _check(all(len(row) == n for row in r["values"]), f"{where}, box {b['id']}: rows of {n}")
                else:
                    _check(r["kind"] == "meter", f"{where}, box {b['id']}: readout is pie or meter")
            for link in lane.get("links", []):
                _check(link["from"] in ids and link["to"] in ids, f"{where}: a link names an unknown box")
                _check(len(link["strength"]) == T, f"{where}: link strength needs {T} entries")
            for e in lane.get("events", []):
                _check(0 <= e["step"] < T, f"{where}: event step {e['step']} out of range")
                _check(e["box"] in ids and e.get("to", e["box"]) in ids, f"{where}: an event names an unknown box")


def to_html(scene: dict[str, Any], player: Path | None = None) -> str:
    """The page for ``scene`` with the scene inlined: ``viewer/live.html`` for a live scene, which runs its model
    and takes the viewer's choices, else ``viewer/player.html``, which plays a recording. ``</`` is escaped so
    nothing inlined can close its script tag."""
    validate(scene)
    player = player or (LIVE if "live" in scene else PLAYER)
    page = player.read_text()
    _check(PLACEHOLDER in page, f"{player} has no {PLACEHOLDER} placeholder")
    data = json.dumps(scene, separators=(",", ":")).replace("</", "<\\/")
    models = ""
    if "live" in scene:
        models = (VIEWER / "models" / f"{scene['live']['model']}.js").read_text().replace("</", "<\\/")
    stage = (VIEWER / "stage.js").read_text().replace("</", "<\\/")
    return (page.replace(PLACEHOLDER, data).replace(STAGE_PLACEHOLDER, stage).replace(MODELS_PLACEHOLDER, models)
                .replace("{{TITLE}}", html.escape(scene["title"])))


def write_html(scene: dict[str, Any], path, player: Path | None = None) -> Path:
    """Write the self-contained page for ``scene`` to ``path``; returns the path."""
    path = Path(path)
    path.write_text(to_html(scene, player))
    return path
