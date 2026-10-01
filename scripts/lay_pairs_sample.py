"""Piece 2: the random sample. Seed 20261001; 12 draws without replacement from ordered pairs of
parts (a part may pair with a copy of itself) times couplers."""
import json, itertools
import numpy as np
from arrangements.lay_pairs import PARTS, COUPLERS
combos = [(a, b, c) for a in PARTS for b in PARTS for c in COUPLERS]
rng = np.random.default_rng(20261001)
pick = rng.choice(len(combos), size=12, replace=False)
sample = [combos[i] for i in pick]
json.dump(sample, open("notes/lay-pairs/sample.json", "w"), indent=1)
for i, s in enumerate(sample, 1): print(i, *s)
