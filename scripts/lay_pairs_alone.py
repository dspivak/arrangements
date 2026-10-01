"""Piece 1: each part alone, against its plain description. Writes notes/lay-pairs/alone.json."""
import json, time
from arrangements.lay_pairs import PARTS, alone, trajectory, summarize
out = {}
for n in PARTS:
    t = time.time()
    r = trajectory([n], alone(n))
    out[n] = summarize(r[:, 0]); out[n]["seconds"] = round(time.time() - t, 1)
    print(n, out[n], flush=True)
    json.dump(out, open("notes/lay-pairs/alone.json", "w"), indent=1)
