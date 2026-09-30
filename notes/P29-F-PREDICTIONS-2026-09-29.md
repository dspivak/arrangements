# P29 stage (f): the insurer's due diligence. Predictions, fixed before code

2026-09-29, Claude Opus 5.5 (1M), subagent of 16311c9d. Plan: `AGENTDOCPLAN.md` §8.2 P29 (f).
Written before any change to `arrangements/town.py`.

## The problem

In `eve_town(4)` the insurer's rate follows her estimate only. Her fit is a one-step gradient on
the latest record and stalls at 1.674 against beta = 2 once the plant cleans (its gradient falls
with smoke squared), so the insurer underprices and its fund falls for good (-0.42 at t = 30).
The book's sentence (§1.6.3): "They'll do their own due diligence from there, because it's their
money on the line."

## The design (the weighting is this run's choice, *proposed*)

`build(..., diligence=True)` (stage 3 or 4; `eve_town(4, diligence=True)` passes it through).
The insurer becomes two boxes:

- **`books`**, a record node like `money`: three running sums, integrated at `dt`,
  `S_uy = sum u_c*y_c`, `C = sum u_c^2` over the claims it has paid (the delayed records
  `(u_c, y_c) = (s^2/2, incidence)` that the court sees), and `E = sum u_e^2` over the exposure it
  has written (`u_e = s^2/2`, the same regressor as her undelayed record: an insurer knows at once
  what it insures and learns late what it costs). So `C(t) = E(t - delay)`.
- **`premium`**, the same box as before, cost `(rate - target)^2/2`, descended at
  `dt*quote_rate`, reading her estimate (or the court's fit when her report is disconnected) and
  its books.

**The weighting.** `own = S_uy / C` is the least-squares fit of the harm rate on every claim paid
so far (guarded: 0 when `C = 0`). The target is

    target = est + (C/E) * (own - est),

a blend in which the insurer's own fit carries the share `C/E` of the exposure's evidence whose
claims it has paid, and her estimate carries the rest, `(E - C)/E`, the exposure whose claims
have not yet arrived (guarded: `target = est` when `E = 0`). **Reason:** the evidence a record
carries about the harm rate in least squares is `u^2`; split all records into those whose claims
are paid and those not yet paid, and the least-squares fit on all of them is exactly
`(C*own + (E - C)*fit_on_unpaid)/E`. The insurer cannot see the unpaid records, so her estimate
stands in for what they will say. It is how an insurer prices unreported losses from an a priori
rate until its own losses develop (the actuaries' Bornhuetter-Ferguson idea, stated here as an
analogy, not a citation). The literal precision-weighted average `(E*est + C*own)/(E + C)` was
rejected: her evidence contains the insurer's, so it counts the paid records twice and would end
near 1.84, not 2.

## Predictions (each able to fail)

1. **Until the first claim the trajectory is stage 4's**, to 1e-12 in every coordinate the two
   towns share, through the step at which the first claim is paid (target = est exactly while
   C = 0). **Harm over 30 units stays within 5% of stage 4's 0.80.**
2. **Once claims arrive the rate ends within 2% of beta** (final premium in [1.96, 2.04] at
   t = 30) **and the fund stops falling**: the insurer's fund changes over the last six units
   (one claims delay, t in [24, 30]) by less than 5% of stage 4's fall over the same window.
   Able to fail if the claims fit also stalls.
3. **With her report disconnected, harm is no worse than stage 3's 6.13** (the insurer then blends
   the court's fit with its own claims fit, both delayed).
4. **The existing town tests pass** (7 files), and with `diligence=False` every stage's trajectory
   is bitwise equal to a snapshot taken before the edit.

Own predictions:

5. **The claims fit is exact from the first claim**: `own = S_uy/C` equals beta to 1e-9 at every
   step after the first claim is paid (noiseless records carry the harm rate exactly; least
   squares over all records does not stall, unlike a one-step gradient).
6. **The court is not repaired**: the court's fit (`assessed`) ends within 2% of stage 4's (about
   1.674); diligence fixes the insurer's price, not the court's coefficient.
7. **The insurer does not recoup its early loss**: its final fund is within 0.05 of stage 4's
   -0.42, because it prices expected loss going forward at beta, never above it.
8. **Harm is no higher than stage 4's** (a higher rate after t = 6 only cleans the plant more).
