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


# ---------------------------------------------------------------------------
# The joint readout (book plan P73 (b)): why the bad prime-finders break apart, and what breaking up does.
# ---------------------------------------------------------------------------

# Settings (book notes/runs/p73-b-56518228/RESULTS.md, "Settings"): the parts' states follow fast, the listener's
# push is small (alpha), and the placements move slowly, so the run is near its slaved limit and has no oscillation.
JR_TAU, JR_MOB, JR_K, JR_MS, JR_ALPHA, JR_STEPS, JR_STRIDE = 0.8, 0.05, 1.0, 0.25, 0.1, 8000, 16
JR_STARTS = [0.3, 0.3, 0.3, 0.3]   # p75's start; spread out (0.2, 0.55, 0.9, 1.25) the 57- and 91-makers flee before
# their parts pass half strength and keep reading 57 and 91 (RESULTS.md, P3)
JR_VARIANTS = [
    ("both", "Pushed away, and broken up", "outer",
     [(5, 0.6, 5), (11, 0.6, 11), (57, 0.6, 57), (91, 0.6, 91)],
     "Two prime-makers and two finders whose parts make a composite even together. The listener pushes the "
     "composite-makers away along its line, and the same push drives each one's parts apart; apart, each reads "
     "out a different composite."),
    ("inplace", "Broken up in place", 0.0,
     [(5, 0.6, 5), (11, 0.6, 11), (57, 0.6, 57), (91, 0.6, 91)],
     "The same finders, with the listener's placements held still, so no finder can move away: the "
     "composite-makers' parts come all the way apart."),
    ("loose", "Loose parts: pulled together, or let go", "outer",
     [(5, 0.6, 5), (13, 1.2, 13), (7, 1.45, 7), (11, 0.6, 11)],
     "Four prime-makers. Two start with their parts 0.6 apart and read out primes. The 13-maker starts with its "
     "parts 1.2 apart and reads out 12; the 7-maker starts 1.45 apart and reads out 6. The listener's pushback "
     "on what the 13-maker reads out pulls its parts together until it reads out 13; the 7-maker is too far "
     "gone, and is pushed away and broken up."),
]


def _jr_variant(spec, outer_mobility):
    import jax.numpy as jnp
    from . import joint_readout as jr
    kab = lambda d: float(np.exp(-d ** 2 / (2 * JR_TAU)))
    fs = []
    for N, sep, _ in spec:
        s0 = (N - 1) + kab(sep)
        parts = [jr.piece_part(N - 1, s0, k=JR_K, mobility=JR_MS, label=f"{N}{x}") for x in "ab"]
        fs.append(jr.finder(parts, tau=JR_TAU, placements=[sep / 2, -sep / 2], mobility=JR_MOB, label=f"{N}"))
    S = jr.scene(fs, tau=JR_TAU, positions=JR_STARTS[:len(spec)], mobility=outer_mobility, alpha=JR_ALPHA)
    qs = jr.trace(S, JR_STEPS)[::JR_STRIDE]
    T = qs.shape[0]
    reads = [jr.read(S, qs[t]) for t in range(T)]
    k = lambda x: np.exp(-np.asarray(x) ** 2 / (2 * JR_TAU))
    return S, reads, T, k


def _word(n):
    from .joint_readout import is_prime
    return "a prime" if is_prime(n) else "not a prime"


def disintegrate_scene() -> dict:
    variants = []
    for vid, name, outer, spec, note in JR_VARIANTS:
        S, reads, T, k = _jr_variant(spec, JR_MOB if outer == "outer" else outer)
        outer_x = np.array([[float(r["outer"][f]) for f in range(S.n)] for r in reads])
        lanes = [{
            "name": "On the listener's line",
            "note": "Each finder moves along the listener's line; the listener hears it more loudly the nearer it is.",
            "space": {"label": "the listener's line", "min": 0, "max": 4.0,
                      "kernel": {"center": 0.0, "tau": JR_TAU, "shape": "bell"}},
            "boxes": [{"id": "listener", "label": "the listener", "x": 0.0}] + [
                {"id": f"f{f}", "label": f"{N}-maker", "x": series(outer_x[:, f])}
                for f, (N, _, _) in enumerate(spec)],
            "links": [{"from": f"f{f}", "to": "listener", "label": f"how loudly the listener hears the {N}-maker",
                       "strength": series(k(outer_x[:, f]))} for f, (N, _, _) in enumerate(spec)],
        }]
        for f, (N, _, _) in enumerate(spec):
            inner = np.array([[float(x) for x in r["inner"][f]] for r in reads])
            sep = inner[:, 0] - inner[:, 1]
            hear = k(sep)
            v = np.array([float(r["readout"][f][0] / r["readout"][f][1]) for r in reads])
            whole = np.rint(v).astype(int)
            step = lambda t: t * JR_STRIDE
            caps = [{"step": 0, "text": (
                f"The parts sit {sep[0]:.2f} apart and hear each other at {hear[0]:.2f}. Each part's number is "
                f"{N - 1} plus how loudly it hears the other, so the {N}-maker reads out {v[0]:.2f}; the nearest "
                f"whole number is {whole[0]}, {_word(whole[0])}.")}]
            for t in range(1, T):
                if whole[t] != whole[t - 1]:
                    caps.append({"step": t, "text": (
                        f"Step {step(t)}. The parts are now {sep[t]:.2f} apart and hear each other at {hear[t]:.3f}, "
                        f"{'more' if hear[t] > 0.5 else 'less'} than half strength: the {N}-maker reads out "
                        f"{v[t]:.2f}, nearest whole number {whole[t]}, {_word(whole[t])}.")})
            caps.append({"step": T - 1, "text": (
                f"The run ends with the parts {sep[-1]:.2f} apart, hearing each other at {hear[-1]:.2f}; the "
                f"{N}-maker reads out {v[-1]:.2f}, nearest whole number {whole[-1]}, {_word(whole[-1])}.")})
            lanes.append({
                "name": f"Inside the {N}-maker",
                "note": f"Its two parts, and what it reads out: {N} when they hear each other well, {N - 1} apart.",
                "space": {"label": f"inside the {N}-maker", "min": -4.5, "max": 4.5},
                "boxes": [
                    {"id": "r", "label": "reads out", "x": 0.0,
                     "readout": {"kind": "meter", "title": f"what the {N}-maker reads out", "label": "value",
                                 "min": N - 1.5, "max": N + 0.5, "values": series(v)},
                     "sub": [f"{w}, {_word(w)}" for w in whole]},
                    {"id": "a", "label": f"{N}a", "x": series(inner[:, 0])},
                    {"id": "b", "label": f"{N}b", "x": series(inner[:, 1])},
                ],
                "links": [{"from": "a", "to": "b", "label": "how loudly the parts hear each other",
                           "strength": series(hear)}],
                "captions": caps,
            })
        variants.append({"id": vid, "name": name, "note": note, "lanes": lanes})
    return {
        "title": "Why the bad prime-finders break apart",
        "description": ("A listener needs primes. Each finder's two parts make its number only together: each "
                        "part's number is its piece plus how loudly it hears the other. A finder that reads out "
                        "a composite is pushed away and broken up, and broken up it reads out another composite."),
        "source": ("arrangements/joint_readout.py, run by Phi and recorded by arrangements/scenes.py "
                   "(disintegrate_scene); book plan P73 (b), notes/runs/p73-b-56518228/."),
        "steps": T,
        "stride": JR_STRIDE,
        "fps": 12,
        "default_lane": -1,
        "variants": variants,
        "default_variant": "both",
    }


SCENES = {"listener": listener_live_scene, "listener-recorded": listener_scene, "disintegrate": disintegrate_scene}
