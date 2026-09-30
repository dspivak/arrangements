# A million block shuffles of the genetic code (scripts/genetic_code_million.py)

Predictions are in the script's docstring, fixed before the run. Claude Opus 5.5 (1M), 16311c9d, 2026-09-29.

### 1966 scale: standard code slip cost 160.3291, count 40
-    50,000 shuffles: slip cost <= standard 0 (0.00e+00); count <= standard 0 (0.00e+00); 1 s

### 1966 scale: standard code slip cost 160.3291, count 40
-   100,000 shuffles: slip cost <= standard 0 (0.00e+00); count <= standard 0 (0.00e+00); 2 s
-   200,000 shuffles: slip cost <= standard 0 (0.00e+00); count <= standard 0 (0.00e+00); 4 s
-   300,000 shuffles: slip cost <= standard 0 (0.00e+00); count <= standard 0 (0.00e+00); 6 s
-   400,000 shuffles: slip cost <= standard 1 (2.50e-06); count <= standard 0 (0.00e+00); 8 s
-   500,000 shuffles: slip cost <= standard 1 (2.00e-06); count <= standard 0 (0.00e+00); 11 s
-   600,000 shuffles: slip cost <= standard 1 (1.67e-06); count <= standard 0 (0.00e+00); 13 s
-   700,000 shuffles: slip cost <= standard 2 (2.86e-06); count <= standard 0 (0.00e+00); 15 s
-   800,000 shuffles: slip cost <= standard 2 (2.50e-06); count <= standard 0 (0.00e+00); 17 s
-   900,000 shuffles: slip cost <= standard 2 (2.22e-06); count <= standard 0 (0.00e+00); 19 s
- 1,000,000 shuffles: slip cost <= standard 3 (3.00e-06); count <= standard 0 (0.00e+00); 21 s

### 2008 scale: standard code slip cost 155.3368, count 48
-   100,000 shuffles: slip cost <= standard 0 (0.00e+00); count <= standard 37 (3.70e-04); 2 s
-   200,000 shuffles: slip cost <= standard 0 (0.00e+00); count <= standard 83 (4.15e-04); 4 s
-   300,000 shuffles: slip cost <= standard 0 (0.00e+00); count <= standard 128 (4.27e-04); 6 s
-   400,000 shuffles: slip cost <= standard 0 (0.00e+00); count <= standard 166 (4.15e-04); 9 s
-   500,000 shuffles: slip cost <= standard 0 (0.00e+00); count <= standard 205 (4.10e-04); 11 s
-   600,000 shuffles: slip cost <= standard 0 (0.00e+00); count <= standard 244 (4.07e-04); 13 s
-   700,000 shuffles: slip cost <= standard 0 (0.00e+00); count <= standard 280 (4.00e-04); 15 s
-   800,000 shuffles: slip cost <= standard 0 (0.00e+00); count <= standard 318 (3.98e-04); 18 s
-   900,000 shuffles: slip cost <= standard 0 (0.00e+00); count <= standard 364 (4.04e-04); 21 s
- 1,000,000 shuffles: slip cost <= standard 0 (0.00e+00); count <= standard 406 (4.06e-04); 23 s

## Verdicts (Claude Opus 5.5 (1M), 16311c9d, 2026-09-29)

The batched measures were checked against the scripts' own functions on the standard code and 20
shuffles before each run (`check`). Seed 0, 1,000,000 block shuffles per scale, about 20 s each.

- **(1) held.** Slip cost, 1966 scale: 3 of 1,000,000 shuffles do as well as the standard code
  (3.0e-6), inside the predicted 1e-6 to 1e-4: the "one in a million" order, from sourced inputs.
- **(2) refuted.** Slip cost, 2008 scale: 0 of 1,000,000, so the standard code does *better* on the
  recomputed scale, not worse as predicted.
- **(3) held on the 1966 scale, refuted on the 2008 scale.** Count (gamma = 3): 0 of 1,000,000 on
  the 1966 scale; 406 of 1,000,000 (4.1e-4) on the 2008 scale, below the predicted 1e-3 to 1e-2. The
  earlier "2 to 10 of 2,000" on that scale was across several gammas and a small sample.

**For the book** (§1.5's genetic-code item): on the cost of slips itself, at most three shuffles in a
million do as well as the code cells use, on either scale. **Reasonable?** The order matches the
published one-in-a-million result for this kind of weighted measure (Freeland and Hurst 1998, not read
here: paywalled); a biologist's standing objection (item 49) is that polar requirement is a proxy for
damage, which this does not touch. **Weird**: nothing new; the two scales disagree on which measure is
stronger, so neither number should be quoted without its scale in the plan.
