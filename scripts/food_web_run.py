"""Run §7's predictions P1-P3, writing notes/lay-pairs/food_web.json after each."""
import itertools, json
import jax.numpy as jnp
from arrangements import food_web as F
from arrangements.lotka_volterra import lotka_volterra
from arrangements.commons import closed_step
out = {}
def save(): json.dump(out, open("notes/lay-pairs/food_web.json", "w"), indent=1)
# P1: two species, against the library's signed-sharp Lotka-Volterra
a, b, g, d, dt, N = 1.0, 0.5, 0.5, 0.25, 0.01, 2000  # rest at (2, 2); start (1, 1) moves
lib = lotka_volterra(a, b, g, d, dt)
step, P = closed_step(lib), lib.P
s = P.init  # library starts at u = v = 0, populations 1 and 1
lib_q = [P.position(s)]
for _ in range(N):
    s = step(s); lib_q.append(P.position(s))
lib_q = jnp.stack(lib_q)
mine = F.log_states(F.web([F.species(a, 0, 1.0, dt), F.species(-g, 0, 1.0, dt)], [(0, 1, b, d)]), N)
out["P1_max_diff"] = float(jnp.max(jnp.abs(lib_q - mine))); out["P1_range"] = [float(v) for v in (jnp.exp(mine).max(0) - jnp.exp(mine).min(0))]; print("P1", out["P1_max_diff"], out["P1_range"]); save()
# P2: enumeration of signs for exact couplers
def ok(edges, n):
    hits = 0
    for s_sp in itertools.product([1, -1], repeat=n):
        for sig in itertools.product([1, -1], repeat=len(edges)):
            if all(s_sp[i] * sg > 0 and s_sp[j] * sg < 0 for (i, j), sg in zip(edges, sig)):
                hits += 1
    return hits
out["P2_chain_assignments"] = ok([(0, 1), (1, 2)], 3)          # rabbit<-weasel<-fox
out["P2_triangle_assignments"] = ok([(0, 1), (0, 2), (1, 2)], 3)  # plus fox eats rabbit
print("P2", out["P2_chain_assignments"], out["P2_triangle_assignments"]); save()
# P3: the triangle. rabbits r=0.8 c=1; weasels die 0.2; foxes die 0.2;
# weasel eats rabbit (1, 0.6), fox eats rabbit (1, 0.2), fox eats weasel (1, 0.5); rest at (0.5, 0.2, 0.1)
r, c = [0.8, -0.2, -0.2], [1.0, 0.0, 0.0]
edges = [(0, 1, 1.0, 0.6), (0, 2, 1.0, 0.2), (1, 2, 1.0, 0.5)]
x0, N3 = [0.6, 0.3, 0.2], 20000
sp = [F.species(r[i], c[i], x0[i], dt) for i in range(3)]
us = F.log_states(F.web(sp, edges), N3)
hand = F.by_hand(r, c, edges, x0, dt, N3)
xs = jnp.exp(us)
out["P3_max_diff_vs_hand"] = float(jnp.max(jnp.abs(us - hand)))
out["P3_end"] = [float(v) for v in xs[-1]]; out["P3_min"] = [float(v) for v in xs.min(0)]
out["P3_tail_range"] = [float(v) for v in (xs[-5000:].max(0) - xs[-5000:].min(0))]
print("P3", out); save()
