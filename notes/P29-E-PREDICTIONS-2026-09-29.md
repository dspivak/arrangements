# P29 (e): power to the townspeople — predictions, fixed before code

Claude Opus 5.5 (1M), session 8f53ae48, 2026-09-29. David: "The power reaches the town
people because they … have need for power"; yes to running this stage the same day.

## Design

Stage (d)'s town with one change. The townspeople box takes two inputs, power and smoke
(the plant publishes both). Its one-form on the **power** input is the constant `benefit`:
they draw each watt and it is worth the same to them, a fixed price per watt. Its one-form
on the **smoke** input stays zero (the insulated wire). The plant's own cost loses its
`-benefit*x` revenue term; the revenue now reaches the plant as the townspeople's push on
the power wire, pulled back through the plant's power output (d power / d x = 1,
d power / d c = 0). Option `build(..., clean=True, power_to_town=True)`; every other stage
and option unchanged.

## Predictions

1. Every trajectory equals stage (d)'s to machine precision (max abs difference below 1e-12)
   for stages 1–4 at clean_cost .1 and 1: the revenue has only moved from inside the plant
   to the wire.
2. The townspeople box returns `benefit` on its power port and exactly 0 on its smoke port,
   at every state visited: the power wire pushes back, the smoke wire does not.
3. The 16 existing town tests pass unchanged.

Able to fail: (1) by a sign error in the returned covector (the plant would be driven to
negative output), which (1) is there to catch.
