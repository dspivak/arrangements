"""Scenes for the player (``animate.py``): each function runs an arrangement and records what it did.

``listener_live_scene``: the listener run in the page itself, inputs adjustable; ``listener_scene``: the listener of ``listener.py`` (book plan §5.9 item 63), its three scripts as
variants, the one-way and exact wires as two lanes side by side. Every number comes from
``listener.run_script`` on ``listener.listener_system``; nothing is recomputed by hand here.
"""

from __future__ import annotations

import numpy as np

from .animate import series
from .listener import DEFAULT, listener_system, run_script

# The listener's three kinds of source, and the three outcomes, in matching colors: a kept promise is what
# a reliable source mostly does, a failure whose account lined up what an accountable one does, a failure
# whose account did not line up what an unaccountable one does.
KINDS = ["reliable", "unreliable but accountable", "unaccountable"]
KIND_COLORS = ["#2f8f6b", "#d39b2a", "#c8473d"]
CHIPS = {
    "K": {"label": "kept", "color": KIND_COLORS[0]},
    "L": {"label": "failed, account lined up", "color": KIND_COLORS[1]},
    "X": {"label": "failed, account did not line up", "color": KIND_COLORS[2]},
}
SCRIPTS = [
    ("A", "One exceptional failure", "KKKKLKKKKK",
     "The source keeps nine promises out of ten; its one failure comes with an account that lines up."),
    ("B", "Habitually late, accountable", "KLKLKKLKLK",
     "The source fails four times in ten, and each time its account of why lines up with what a check found."),
    ("C", "Unaccountable", "KXKXKKXKXK",
     "The same four failures, but no account lines up. B and C differ only in that."),
]
WIRES = [
    ("oneway", "One-way wire",
     "The valuation reads the belief and nothing comes back. The belief is Bayes' rule, scaled by how much "
     "the listener counts on the source."),
    ("exact", "Exact wire",
     "The valuation's cost is pulled back onto the belief. Every step it pushes the belief toward the kind of "
     "source that is cheapest to count on, more strongly the more the listener counts on it."),
]


NEWS = {"K": "the source kept its promise",
        "L": "the source broke its promise, and its account of why lined up",
        "X": "the source broke its promise, and its account of why did not line up"}


def _captions(script, wire, ps, ys, ks, steps_per_round):
    """One caption per step, read off the run, in the step's own order: the move is set by the belief the
    listener holds going into the step (its expected loss from counting on the source against the cost of
    doing without), the news arrives in the same step, and the belief after it is the next step's."""
    out = []
    ell, c = np.asarray(DEFAULT.ell), DEFAULT.c
    pct = lambda y: (f"{100 * y[0]:.0f}% reliable, {100 * y[1]:.0f}% unreliable but accountable, "
                     f"{100 * y[2]:.0f}% unaccountable")
    last = len(ps) - 1
    for t in range(len(ps)):
        if t == last:
            out.append({"step": t, "text": f"The run ends here. The listener believes the source is {pct(ys[t])}."})
            continue
        loss, dp = float(ys[t] @ ell), ps[t + 1] - ps[t]
        way = ("so it stays about where it is" if abs(dp) < 1e-3 else
               "so it moves toward the source" if dp < 0 else "so it moves away from the source")
        move = (f"Going into this step it expects counting on the source to cost {loss:.1f}, "
                f"{'less' if loss < c else 'more'} than the {c:g} of doing without, {way}.")
        r, k = divmod(t, steps_per_round)
        if k == 0 and r < len(script):
            news = f"Round {r + 1}: {NEWS[script[r]]}. The listener hears it at strength {ks[t]:.2f}."
        elif wire == "exact":
            news = "No news, but the cost of counting on the source pushes the belief toward reliable."
        else:
            news = "No news, so the belief does not change."
        out.append({"step": t, "text": f"{move} {news} After this step it believes the source is {pct(ys[t + 1])}."})
    return out


def listener_scene(steps_per_round: int = 10) -> dict:
    tau = DEFAULT.tau
    kappa = lambda p: np.exp(-np.asarray(p) ** 2 / (2.0 * tau))
    systems = {w: listener_system(wire=w) for w, _, _ in WIRES}
    variants = []
    for vid, name, script, note in SCRIPTS:
        lanes = []
        for w, wname, wnote in WIRES:
            ps, ys, _ = run_script(systems[w], script, steps_per_round)
            T = len(ps)
            events = [{"step": r * steps_per_round, "box": "source", "to": "listener",
                       "label": CHIPS[c]["label"], "color": CHIPS[c]["color"]} for r, c in enumerate(script)]
            captions = _captions(script, w, ps, ys, kappa(ps), steps_per_round)
            lanes.append({
                "name": wname,
                "note": wnote,
                "space": {"label": "placement", "min": -0.4, "max": 3.4, "kernel": {"center": 0.0, "tau": tau}},
                "boxes": [
                    {"id": "source", "label": "source", "x": 0.0, "fixed": True},
                    {"id": "listener", "label": "listener", "x": series(ps),
                     "readout": {"kind": "pie", "title": "belief about the source", "labels": KINDS,
                                 "colors": KIND_COLORS, "values": series(ys)}},
                ],
                "links": [{"from": "listener", "to": "source", "label": "counts on the source",
                           "strength": series(kappa(ps))}],
                "events": events,
                "captions": captions,
            })
        variants.append({
            "id": vid, "name": name, "note": note,
            "ticks": {"every": steps_per_round, "label": "round", "chips": [CHIPS[c] for c in script]},
            "lanes": lanes,
        })
    return {
        "title": "The listener",
        "description": ("A listener decides how much to count on a source from what it believes about the "
                        "source; on the exact wire, what that choice costs pushes back on the belief."),
        "source": "arrangements/listener.py; book plan §5.9 item 63 (P37 (a), Claude Fable 5.1, f2836b6c)",
        "steps": T,
        "fps": 3,
        "hold": 1.5,
        "default_lane": 0,
        "legend": [{"label": k, "color": c} for k, c in zip(KINDS, KIND_COLORS)],
        "variants": variants,
        "default_variant": "C",
    }


def listener_live_scene() -> dict:
    """The same listener run in the page (``viewer/models/listener.js``), so its inputs can be changed."""
    return {
        "title": "The listener",
        "description": ("A listener decides how much to count on a source from what it believes about the "
                        "source; on the exact wire, what that choice costs pushes back on the belief."),
        "source": "arrangements/listener.py; book plan §5.9 item 63 (P37 (a), Claude Fable 5.1, f2836b6c)",
        "live": {"model": "listener"},
        "fps": 5,
        "hold": 0.8,
    }


SCENES = {"listener": listener_live_scene, "listener-recorded": listener_scene}
