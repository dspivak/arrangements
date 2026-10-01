"""Piece 5: the mechanism behind the fox results. A 'plain fox' dies at the same rate alone but has a
POSITIVE response, like every other part. If the predator box's negative response is what reverses
couplers, the plain fox keeps in step with the chooser (lives longer) and starves as an eater."""
import json
import jax.numpy as jnp
from arrangements import lay_pairs as L
from arrangements.coalgebra import configuration
from arrangements.lotka_volterra import log_population
def plain_fox():
    return L._part(configuration(log_population(+1.0, L.DT), init=jnp.zeros(1)),
                   lambda q: 0.5 * jnp.exp(q[0]), lambda q: jnp.exp(q), "plain_fox")
L.PARTS["plain_fox"] = plain_fox
out = {}
for a, b, c in [("plain_fox", "plain_fox", None), ("foxes", "chooser", "in_step"), ("plain_fox", "chooser", "in_step"),
                ("tank", "foxes", "eats"), ("tank", "plain_fox", "eats")]:
    arr = L.alone(a) if c is None else L.pair(a, b, c)
    r = L.trajectory([a] if c is None else [a, b], arr)
    out[f"{a},{b},{c}"] = [L.summarize(r[:, k]) for k in range(r.shape[1])]
    print(a, b, c, [round(s["end"], 6) for s in out[f"{a},{b},{c}"]], flush=True)
json.dump(out, open("notes/lay-pairs/diagnose.json", "w"), indent=1)
