"""Piece 4: run each sampled pair, and the same pair with each part's push from the coupler shown
separately. Writes notes/lay-pairs/pairs.json after every pair (record as you go)."""
import json, sys, time
from arrangements.lay_pairs import pair, trajectory, summarize
sample = json.load(open("notes/lay-pairs/sample.json"))
alone = json.load(open("notes/lay-pairs/alone.json"))
which = [int(a) for a in sys.argv[1:]] or range(1, len(sample) + 1)
try:
    out = json.load(open("notes/lay-pairs/pairs.json"))
except FileNotFoundError:
    out = {}
for i in which:
    a, b, c = sample[i - 1]
    t = time.time()
    r = trajectory([a, b], pair(a, b, c))
    res = {"first": a, "second": b, "coupler": c,
           a + "#1": summarize(r[:, 0]), b + "#2": summarize(r[:, 1]),
           "seconds": round(time.time() - t, 1)}
    out[str(i)] = res
    print(i, a, b, c, flush=True)
    for k in (a + "#1", b + "#2"):
        s = res[k]; al = alone[k[:-2]]
        print(f"   {k:12s} end {s['end']:.4g} (alone {al['end']:.4g})  min {s['min']:.4g} max {s['max']:.4g}  "
              f"tail_turns {s['tail_turns']} tail_range {s['tail_range']:.3g} finite {s['finite']}", flush=True)
    json.dump(out, open("notes/lay-pairs/pairs.json", "w"), indent=1)
