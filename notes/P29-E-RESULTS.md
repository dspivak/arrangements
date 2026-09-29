# P29 (e): power to the townspeople — results

Claude Opus 5.5 (1M), session 8f53ae48, 2026-09-29. Predictions:
`notes/P29-E-PREDICTIONS-2026-09-29.md`, fixed before code. Construction:
`build(..., clean=True, power_to_town=True)` in `arrangements/town.py`. Tests:
`tests/test_town_power.py` (9 passed); the five earlier town files still pass (16).

1. **Held.** Every trajectory equals stage (d)'s, max abs difference below 1e-12, stages 1–4
   at clean_cost .1 and 1. The plant's revenue, removed from its own cost, arrives as the
   residents' push on the power wire, pulled back through d power / d x = 1.
2. **Held.** The residents' box returns `benefit` on its power port and exactly 0 on its smoke
   port at every state sampled: the power wire pushes back, the smoke wire is an insulator.
3. **Held.** The 16 earlier town tests pass.

Reasonable or weird (agent judgment, not evidence): a fixed price per watt is a flat tariff,
reasonable for a town's utility bill. Weird: the residents' push does not depend on how much
they draw, so their demand never saturates; the plant's rising cost is what limits output.
Their payment does not depend on their illness or income. The push is a price (value per
watt), the dual of a quantity; the physical current is the dual of a voltage, a different
wire, left out of the model (a Ch 5 exercise: with voltage v on the wire, a resistive load
drawing v/R, and a fixed price per watt, the payment's push on v is 2·price·current).

What this does not establish: anything new about the plant's choices; (e) only relocates
the revenue so the book's sketches can show the power wire pushing back.
