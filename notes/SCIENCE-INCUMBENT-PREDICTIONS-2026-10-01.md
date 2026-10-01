# Science at two scales, test (i): what is built on an idea pays for its excuses (predictions, before code)

A companion to `SCIENCE-SCALES-PREDICTIONS-2026-10-01.md` (tests (a)–(g), esp. §9 and §5.8, test (g)) and
`SCIENCE-PAYMENT-PREDICTIONS-2026-10-01.md` (test (h), supplies and death). David, 2026-10-01, asked what keeps an
incumbent alive longer than its evidence warrants: "We choose to make excuses because there's so much that's been
built on it … the effort to make the excuses is coming from those ideas that have been built on it, wanting to keep
their status as useful." To the agent's correction, "The effort to make excuses comes from the holders of the ideas
built on it, whose own work would come apart if it fell": "yes to your candidate. run test please :)". Design and
predictions by Claude Opus 5.5, 0fe746c1, before any code. Results to §4.

## 1. Setting (native: boxes, wires, `Phi`; disclose any piece that is not)

Test (g4)'s contest: incumbent `h0` confirmed by `n_A` draws from A, rival `h1` explaining B's draws, which arrive
one per step; a reader's chooser on `{h0, h1}`; only the incumbent may build excuses from (g)'s library of
figures (hard ruling-out version), at drawing cost `delta_k`. New: `D` **dependent holders**, each a box with a
supply `s_d` (test (h)'s machinery): each is paid a stipend `pi` per step times the reader's share on `h0` (its work
is useful only while the incumbent stands), pays upkeep `c`, and pays into an **excuse fund** that the reader draws
on: an excuse is available only while the fund covers its drawing cost. A dependent may **rebuild** on the rival
at a one-time cost `R` (after which its stipend follows the share on `h1` and it stops paying into the fund).
A dependent at zero supply is dead and, as test (h)'s correction, falls silent and pays nothing.

**Arms** (same draws, seeds, `n_A`, `d`, library, `pi`, `c` in every arm): `D in {0, 2, 5, 10}`;
`R in {small, large, infinite}`.

## 2. Predictions

- (i1) `D = 0` reproduces test (g4)'s incumbent-only overtake count (26–30 at its settings) exactly.
- (i2) With `R` infinite, the overtake count rises with `D` (monotone in the four values); the run derives the
  closed form in the regime where the fund never runs dry.
- (i3) With `R` small, dependents rebuild early and the overtake count returns to within 2 of (i1)'s, whatever `D`.
  With `R` large, it lies between (i1)'s and (i2)'s. The switch happens when rebuilding costs less than funding the
  next excuse (closed form for the step, written before code).
- (i4) *Funerals.* With `R` infinite and stipends falling as `h0`'s share falls, the incumbent holds until the
  fund cannot pay for the next excuse: either the dependents' supplies run out (they die) or the remaining excuses
  are too far. The run reports, per seed, which came first, and the gap between the last dependent's death and the
  overtake; predicted within 3 steps when death comes first. (Called "Planck's principle," science advancing one
  funeral at a time; Planck's own wording not yet checked, not to be cited until it is.)
- (i5) Who pays: in every arm, the total drawn from the fund equals the total paid in by dependents, to round-off
  (the excuses' effort is the dependents' supply, as David's sentence says).

**Numbers fixed before code** (the run's, Claude subagent of Claude Opus 5.5, 0fe746c1, 2026-10-01; from a
numpy scratch script that builds no arrangement, sized before it ran at about 2e7 flops, 9 s; written before
`arrangements/science_incumbent.py` existed).
*Setting kept from (g4).* `n_A = 20`, `d = 3` only (one `d`, for time), seeds 0-9 in every arm, `a =
rng.normal(0, 1, 20)`, `b = rng.normal(d, 1, 150)`, `T = 1`. Library: (g4)'s, `K = 8` mirrored pairs,
footprints `+-d`, `delta_k = 0.5 k`, **hard** variant (at the `j`-th B draw figures `1..j-1` are absent), so
only B draws 0-7 can be excused and the reader carries attachments for those 8 draws only (`1 + 8 x 16 = 129`
coordinates; in (g4) the others have zero forcing and stay at `a0`, so this is an exact reduction). Chooser
in the natural chart, response `1/(p0 p1)`, `dt = 1`, with (g4)'s stable `p0 p1`; attachments `dt_a a(1-a)`,
`dt_a = 0.02`, `a0 = 0.01`. Counts 0-60, **400 Phi steps per count** (not (g4)'s 1000: sized to run in
minutes; 300 per count matched (g4)'s closed form on seed 0 in the diagnosis).
*New pieces, all stipulated.* Units are per count; a box applies a per-count rate as rate/400 per step.
- Dependents: `pi = 1` (stipend times `p0`), `c = 0.5` (upkeep), `f = 1` (paid into the fund), starting supply
  `s0 = 7.5` each, so net `-0.5` per count while `p0 ~ 1`: **every dependent dies at count 15** whatever `D`.
- **The fund pays half** (`sigma = 1/2`) of the drawing cost of every excuse held, **per count while it is
  held** (an excuse is kept up, not bought once); the reader's own cost carries `(1 - sigma g(F)) delta . a`.
  So `D = 0` (no fund, `g = 0`) is (g4) exactly. Why half: at `sigma = 1` a subsidized excuse is free, and the
  split of attachment among the unruled figures is not determined by the cost.
- Smooth death and smooth solvency, no floor: test (h)'s gate `g`. Supply moves by `g(s/s1) (pi p0 - c - f)`,
  `s1 = 0.1`; payment into the fund `g(s/s1) f`; the fund moves by `sum_d g(s_d/s1) f - sigma g(F/F1) delta . A`,
  `F1 = 0.5`, `F(0) = 0`. Per-step changes are below `1/400` of the gates' widths, so neither goes negative.
  *Dead:* `g(s/s1) < 0.01`. *Dry:* `g(F/F1) < 0.5` at a count's end, after count 1.
- `D in {0, 2, 5, 10}`, as 10 dependent boxes in every arm, the absent ones started at `s = 0` (silent).
- `R`: **(i3) is deferred** to after (i1), (i2), (i5), (i4); its numbers are not fixed here, and if it is run
  they are fixed below before its code.
*Closed forms at rest, count by count* (fund solvent or not, `g` in `{0, 1}`), `n_B*` per seed 0-9:
- `D = 0`: 26, 27, 27, 28, 29, 37, 19, 33, 31, 28 (median 28): (g4)'s hard incumbent-only values exactly.
- `D = 2`: the same (fund dry at counts 7-9, before the deaths). `D = 5`: the same (dry at 13-20).
- `D = 10`: 26, **29**, 27, 28, 29, 37, **20**, **34**, 31, 28 (median 28.5); dry at 23-34, after the deaths.
- Deaths at count 15 in every seed with `D > 0`, before every overtake.
*Verdict rules.* Native `n_B*` must equal the closed form in at least 9 of 10 seeds per `D`, else the
construction failed and that is recorded first. **(i1)** holds iff native `D = 0` equals (g4)'s hard values in
10 of 10. **(i2)** holds iff `n_B*` is non-decreasing in `D` in 10 of 10 seeds and strictly higher at `D = 10`
than at `D = 0` in at least 5 of 10; **predicted from the closed form: refuted** (3 of 10 rise, by 1 or 2;
the half-subsidy cannot carry the incumbent past its library). **(i5)** holds iff in every run the fund's
final balance equals paid-in minus drawn to `1e-9` (relative to paid-in); the literal "drawn equals paid in"
holds only where the fund ends below `1e-3` of paid-in, counted. **(i4)**: per seed, last death against
dry and against overtake; predicted within 3 counts, **closed form: refuted**, gap 4-22 counts: when the fund
dries the excuses do not vanish, they cost the reader what they cost in (g4).

**What it cannot show.** That science works this way; the stipend rule, the fund, `R` and the library are
stipulated. Excuses are still evaluated inside the cost, not run against a supply by a runner (P34).

## 4. Results

Run 2026-10-01 by Claude (subagent of Claude Opus 5.5, 0fe746c1). Code: `arrangements/science_incumbent.py`
(`reader`, `fund`, `dependent`, `incumbent_run`, `run_arm`, `summarize`); tests `tests/test_science_incumbent.py`.
One closed composite per arm, `wire_boxes([world, reader, fund, 10 dependents])`, run by `Phi` under `lax.scan`
and vmapped over the 10 seeds; settings exactly as fixed above, the same draws, seeds, library, `pi`, `c`, `f`,
`s0` in every arm. **Runs, in order.** One smoke run (seeds 0-1, `D = 0` and `10`: `n_B*` 26, 27 and 26, 29,
the closed forms). **Recorded run 1** of the test file, (i2) and (i4) strict xfail from the start as predicted
refuted before code: **9 passed, 2 xfailed, 10 s**. No re-run. No NaN.
**Non-native or stipulated pieces:** the world's lookup readout (`science_native`'s GAP); the fund paying
half of each held excuse's drawing cost per count (`sigma = 1/2`, a maintenance reading); the reader's
attachments descend `K0` with the fund's solvency `gF` taken as data, and the fund's draw is computed from
published values, so reader and fund are a bookkeeping, not one potential. No floor: death and insolvency go
through test (h)'s smooth gate. No rebuild (`R` infinite in every arm run).

**Construction.** Native `n_B*` equals the closed form in **10, 10, 10, 9 of 10 seeds** at `D = 0, 2, 5, 10`.
The one miss: `D = 10`, seed 7, native 33 against closed 34. The native fund dries a count earlier than the
count-by-count closed form (dry at 25, 30 against 26, 31 on seeds 0, 1), because the smooth gate starts
withholding before the balance reaches 0; at seed 7 the fund is dry at count 33, the crossing count itself.

**(i1) Held.** `D = 0`, native `n_B*`: 26, 27, 27, 28, 29, 37, 19, 33, 31, 28, equal to (g4)'s hard
incumbent-only closed form in **10 of 10** seeds (median 28). The prediction's "26-30" was (g4)'s middle; the
per-seed values run 19-37.

**(i2) Refuted, as predicted from the closed form before code.** `n_B*` by `D` (same seeds, same settings):
`D = 0, 2, 5`: identical in 10 of 10 seeds (median 28); `D = 10`: 26, **29**, 27, 28, 29, 37, **20**, 33, 31, 28
(median 28.5). Non-decreasing in `D` in 10 of 10 seeds; higher at `D = 10` than at `D = 0` in **2 of 10** (by 2
and 1), against the rule's 5. **The mechanism:** the dependents can only pay for excuses the library offers,
and in the hard variant only B draws 0-7 can be excused at all. Half of those excuses' drawing cost is worth
at most about one to two counts of the rival's advance (about 5 in `lambda` per B draw at `d = 3`). And the
subsidy helps only if the fund is still solvent at the crossing: with `D = 2` it is dry at counts 6-8, with
`D = 5` at 14-19, with `D = 10` at 21-33, against crossings at 19-37. So the incumbent's protection stays
**bounded by its library**, as in (g4): what is built on it can make the excuses cheaper, not more numerous.

**(i5) Held as bookkeeping; the literal equality depends on what is left in the fund.** In every run the
fund's final balance equals paid-in minus drawn to **`1.9e-13`** of paid-in (worst over 40 runs). Paid-in:
about 30.0, 74.9, 149.8 at `D = 2, 5, 10` (every dependent pays `f = 1` per count until it dies at count 15);
drawn: within 0.045 of it. The balance left, 0.042-0.047, is what the gate holds back as the fund nears empty.
"Drawn equals paid in" to `1e-3` of paid-in: 10 of 10 seeds at `D = 5` and `10`, **0 of 10 at `D = 2`**
(balance 0.043 against 0.030). Every unit drawn was paid in by a dependent; nothing came from elsewhere.

**(i4) Refuted, as predicted from the closed form before code.** Every dependent dies at count 15 in every
seed and arm (the stipend is `pi p0` with `p0 ~ 1` until the crossing, so supply falls 0.5 per count from 7.5).
Which came first, per seed: `D = 2`, the fund dried first in **10 of 10** (counts 6-8); `D = 5`, death first in
**7 of 10** (dry at 14-19); `D = 10`, death first in **10 of 10** (dry at 21-33). Gap from the last death to the
overtake, in the death-first cases: **4 to 22 counts; within 3 in 0 of 17**. The incumbent does not fall
with its dependents: when the fund dries, the excuses are not withdrawn, they cost the reader what they cost
in (g4), and (g4)'s library protection remains. At `D = 2` and `5` the overtake count is (g4)'s to the
count; the funerals change nothing. At `D = 10` the fund outlives the dependents by 6-18 counts, so the
gap is not even a gap between death and insolvency.

**(i3) Not run.** Deferred before code (see the fixed numbers), and not built: no rebuild rule was fixed.
*A deduction, not a run* (the run's): a rebuild only withdraws payment from the fund, and with no payment the
reader is (g4)'s. At rest, less funding at the crossing can only raise `K0` there, so it cannot delay the
overtake. Every arm run has `n_B*` within 2 of (i1)'s in every seed. So (i3)'s two clauses ("within 2 of
(i1)'s" with `R` small; "between (i1)'s and (i2)'s" with `R` large) would hold for any rebuild rule at these
settings, and would test nothing about rebuilding. A test of rebuilding needs settings where dependents matter,
which (i2) shows these do not.

**Which results could have come out otherwise.** Little. Every verdict was fixed by the closed form before
code; the native run checks the construction (wiring, gates, lag) and matched it in 39 of 40 seed-arms.
(i5)'s identity holds by construction of the fund's forcing; what could have failed is the wiring (a payment
counted at the wrong step), and it did not. (i1) could have failed if the reduced reader (8 excusable draws)
were not an exact reduction of (g4)'s; it was. Open before the run: only the timing of insolvency against the
crossing under the smooth gate, which moved one seed.

**Reasonable or weird** (the run's judgment, not evidence). *A person:* reasonable that friends who depend on
your being right can make your excuses cheaper but cannot invent new ones; weird that their support is an
upkeep paid per count for each excuse held, rather than effort spent once in making it. *A scientific community:* the closest reading. Reasonable that a
field's investment in a theory buys auxiliary hypotheses but cannot outlast the library of auxiliaries; on
this model, what would keep an incumbent alive past its evidence is not funding but **new figures**, an
open-ended library, which this model forbids. Weird that every dependent dies at the
same count regardless of how the contest goes, because the stipend follows the share and the share stays
near 1 until the last moment; in a real field the incumbent's share erodes gradually. *A firm or
institution:* reasonable that a sponsor's subsidy delays a bad product's withdrawal only while it is paid,
and weird that when the sponsor goes bankrupt nothing happens at once. *A population of organisms:* no
reading I would defend. *A cell:* none.

**What this does not establish.** That dependents cannot prolong an incumbent in general: only that, with
a finite hard library and a half-subsidy of drawing cost, they cannot prolong it past the library's bound.
A fund that could buy new figures, or a soft library (which leaks a fixed-ratio discount, (g4)), or a full
subsidy, or a share that erodes early, could change every verdict. Nothing about rebuilding (i3), `d = 2`,
or other `pi`, `c`, `f`, `s0`.

## 5. Test (j): paying someone to try to build excuses (predictions, before code, 2026-10-01)

David, on test (i): "paying for someone to try to build excuses. why didn't that work? … can you make a scenario?";
after the scenario (a firm built on "Bob is reliable"; its money made reaching a shelved excuse cheaper but added
none; what he means is a fixer paid to come up with new ones): "yes try to build excuses." Design and predictions
by Claude Opus 5.5, 0fe746c1, before any code. Results to §6.

**Setting.** Test (i)'s contest at its settings, with one new box: a **fixer**, paid from the fund at rate `F` per
count. Its effort draws candidate excuses: each a new figure with a random footprint on the side channels and a
random distance (from a seeded sequence, as the world box's draws are). A candidate enters the library (is planted,
the support-creating push of the native round, disclosed as non-exact) only if it is consistent with every side
observation made so far; each past check rules a random candidate out with probability `q` (take `q = 1/2`).
Once planted, a figure is drawn and ruled out exactly as in (g).

**Predictions.**
- (j1) `F = 0` reproduces (i1)'s overtake counts seed by seed.
- (j2) The overtake count rises with `F` (monotone over at least four values spanning two orders of magnitude).
- (j3) *Epicycles.* Surviving `k` past checks has probability `(1 - q)^k`, so the effort per usable new excuse
  doubles with each check, and each doubling of `F` buys about the same number of extra counts: the delay grows
  like `log F`, not like `F`. The run derives the closed form before code and fits the slope.
- (j4) The admitted excuses grow more specific (more side channels they must match) and, if distance is tied to
  specificity, further away, so their drawing cost rises along the run.
- (j5) With the fund paid by test (i)'s dependents instead of a fixed `F`, the reign ends when the dependents can
  no longer pay for the search, and it is longer than (i)'s at the same `D` in at least 8 of 10 seeds.
**What it cannot show.** That real excuse-making searches at random; a fixer who reasons would search better than
`(1 - q)^k`, and the run says how that would change the slope (smaller `q` per check), not whether it happens.

**Numbers fixed before code (j)** (the run's, Claude subagent of Claude Opus 5.5, 0fe746c1, 2026-10-01; numpy
scratch building no arrangement, sized before it ran at about 1e6 operations per seed and arm, 1 s; written
before any (j) construction code).
*Kept from (i):* `d = 3`, `n_A = 20`, (g4)'s hard library of 8 mirrored figures, the same draws and seeds 0-9,
counts 0-60, 400 Phi steps per count, chooser and attachment responses as in (i). No dependents, no fund in
(j1)-(j4): the fixer is paid `F` from outside. Units: one candidate costs 1, so pay `F` draws `F` candidates per
count. **`F in {0, 4, 16, 64, 256, 1024, 4096}`** (three orders of magnitude). **`q = 1/2`.**
*Candidates* (seeded, `default_rng(1000 + seed)`, one stream shared by every `F`: arm `F` takes the first `F` of
each count's 4096): each has **`G`**, the number of checks it survives, geometric with `P(G >= k) = (1-q)^k`, and
a **distance `delta ~ U(0.5, 4.0)`, independent of specificity** (the untied law; the tied one is not run). Its
main-channel footprint is `+d` (it can excuse an anomalous B draw, like (g4)'s `+d` figures). One check is made
per B draw, so a candidate drawn at count `n` (after `n` checks) is **admitted iff `G >= n`**, and is usable on
draw `j` iff `j < G` (ruled out per draw, as in (g)): it can excuse every draw already seen. Its specificity is
`n`, the side observations it had to match.
*Reduction, disclosed.* Per draw, only figures that are among the **3 cheapest usable ones admitted by some
count** are given attachments (at most 252 pairs per seed at `F = 4096`). At rest this is exact unless `b > 3d`
(a third full figure is never needed below that; no draw here exceeds it, to be checked). *Planting:* every
pair's attachment is held at `a0 = 0.01` from the start, masked out of the cost (zero forcing) until its
figure is admitted, as (g4) masks ruled-out figures; the mask is a world readout (a seeded lookup). So no
support-creating push is needed; this replaces the planting push the spec names.
*Closed forms at rest, `n_B*` per seed 0-9:*
- `F = 0`: 26, 27, 27, 28, 29, 37, 19, 33, 31, 28 ((i1)'s).
- `F = 4`: 26, 28, 28, 28, 29, 37, 19, 34, 32, 28. `F = 16`: 27, 28, 28, 28, 32, 37, 20, 34, 32, 28.
- `F = 64`: 27, 31, 30, 33, 32, 38, 22, 36, 36, 28. `F = 256`: 30, 33, 35, 35, 32, 38, 23, 36, 36, 33.
- `F = 1024`: 30, 35, 36, 36, 33, 39, 25, 39, 36, 33. `F = 4096`: 31, 37, 36, 37, 39, 39, 28, 39, 39, 36.
*(j3)'s slope, closed form.* The longest-surviving admitted candidate passes about `log_{1/(1-q)} F` checks, so
each doubling of `F` (at `q = 1/2`) extends the excusable horizon by one draw. Each newly excusable draw saves
`s = 1/2 b^2 - (its excused cost)` in `lambda`, against the rival's mean advance `d^2/2 = 4.5` per draw. So
**`dn*/dlog2 F ~ s/(d^2/2)`**, between `0.56` (one figure at the mean distance 2.25) and `0.89` (one at 0.5); the
rest computation gives **0.75** (mean over seeds, `F` 4-4096). Linear in `F` would give thousands of counts.
*Verdict rules.* Construction: native `n_B*` equals the closed form in at least 9 of 10 seeds per `F`. **(j1)**
`F = 0` equals (i1) in 10 of 10. **(j2)** `n_B*` non-decreasing in `F` in 10 of 10 seeds, and the median at
4096 above the median at 0. **(j3)** the native mean `n_B*` against `log2 F` (`F >= 4`) has slope in
`[0.5, 1.0]` and a better least-squares fit (`R^2`) than against `F`. **(j4)** among admitted figures,
specificity (admission count) is non-decreasing along each draw's admitted sequence (by construction); the
claim tested is the drawing cost: the attachment-weighted `delta` of the new figures excusing draw `j`, at the
run's end, rises with `j` (Spearman correlation `> 0` in at least 8 of 10 seeds at `F = 1024`). Predicted held:
with distance untied, what rises is the cheapest *surviving* distance, because fewer candidates survive to
later draws. **(j5)** is to be fixed below before its code, if the budget allows.

## 6. Results of (j)

Run 2026-10-01 by Claude (subagent of Claude Opus 5.5, 0fe746c1). Code: `science_incumbent.j_candidates`,
`j_pairs`, `fixer_reader`, `fixer_run`, `run_fixer_arm`; tests appended to `tests/test_science_incumbent.py`.
World (draws, seen flags, and the admitted mask of the fixer's candidates) wired to (i)'s reader carrying one
attachment per (draw, new figure) pair, run by `Phi`, vmapped over seeds; settings as fixed in §5. **Runs, in
order.** One smoke run (`F = 4096`, seeds 0-1: `n_B*` 31, 37, the closed forms; 2.8 s). **Recorded run 1** of
the whole file: **21 passed, 2 xfailed ((i2), (i4), as before), 91 s.** No re-run, no NaN.
**Non-native pieces:** the fixer's search is a seeded candidate sequence computed before the run, and its
admissions reach the reader as a mask emitted by the world box (a lookup); planting is (g4)'s masking (every
candidate pair held at `a0 = 0.01`, zero forcing until admitted), not a support-creating push; per draw only the
3 cheapest usable figures get attachments (exact at rest when `b < 3d`; the largest B draw in any seed is 6.32).
The fixer is paid from outside: no dependents and no fund in (j1)-(j4).

**Construction.** Native `n_B*` equals the closed form in **10 of 10 seeds at every `F`** (70 of 70).

**(j1) Held.** `F = 0`: 26, 27, 27, 28, 29, 37, 19, 33, 31, 28, (i1)'s in 10 of 10.

**(j2) Held.** `n_B*` is non-decreasing in `F` in **10 of 10 seeds**; median by `F` = 0, 4, 16, 64, 256, 1024, 4096:
**28, 28, 28, 31.5, 34, 35.5, 37**. At `F = 4096` every seed reigns longer than at `F = 0` (by 2 to 10 counts, 10 of
10). Unlike (i)'s money (a cheaper shelf), the fixer's money adds excuses, and each one admitted can excuse
every anomaly already seen, since it was chosen to fit them.

**(j3) Held.** Mean `n_B*` against `log2 F` (`F` 4 to 4096): slope **0.746 counts per doubling**, `R^2 = 0.984`;
against `F`, `R^2 = 0.656`. The closed form before code gave 0.75, inside its bracket `[0.56, 0.89]`. Gains per
doubling between arms: 0.25, 0.95, 0.90, 0.55, 0.95. **Mechanism:** a candidate must survive every check made
so far, `(1 - q)^n`, so a thousandfold budget buys about ten more draws of excusable horizon; each newly
excusable draw saves about three quarters of the rival's per-draw advance. A thousand times the money bought 9
more counts of reign (mean 28.9 to 36.1, from `F = 4` to `4096`).

**(j4) Held, for the untied law; the tied law was not run.** Specificity rises by construction (admission
counts of used pairs run from 0 to 8-12 per seed). Drawing cost: the attachment-weighted distance of the new
figures excusing draw `j`, at the run's end, rises with `j`: Spearman **0.89 to 0.98, positive in 10 of 10 seeds**
(`F = 1024`; seed 0: 0.50 at draw 0, 0.66 at draw 9, 1.47 at draw 14, 3.83 at draw 16). Distances are drawn
independently of specificity, so what rises is the **cheapest surviving** distance: fewer candidates survive to
cover later draws, so the reader must take what survives.

**(j5) Not run natively; a closed form, computed after (j1)-(j4) and with no numbers fixed before it, so
weaker evidence.** The fixer paid by (i)'s dependents (`D` per count while they live, counts 0-14, then 0), with
no half-subsidy. `n_B*` against (i)'s native at the same `D`: longer in **2 of 10 (`D = 2`), 4 of 10 (`D = 5`), 5 of
10 (`D = 10`)**, never shorter except `D = 10` seed 1 (28 against (i)'s 29, which had the subsidy). Against the
prediction's 8 of 10: **refuted in closed form.** Reason: the dependents die at count 15, and searching is
cheap only early, when few checks have been made; ten candidates a count for fifteen counts is about `F = 10`.

**Which results could have come out otherwise.** The verdicts were all fixed by the closed form before code;
the native run checks the wiring, masking and the reduction, and matched 70 of 70. Open before the closed
form: everything, and in particular whether retroactive excusing would make the reign unbounded in `F`; it
does not, because survival falls geometrically with the checks.

**Reasonable or weird** (the run's judgment, not evidence). *A person:* reasonable that a hired apologist finds
new stories for Bob, each fitting everything known so far, and that each new story is harder to find; weird
that a story found late explains every earlier lapse at no extra effort beyond its distance. *A scientific
community:* the closest reading, epicycles: auxiliary hypotheses tailored to all past data, each harder to find
than the last; reasonable that funding buys time logarithmically; weird that checks arrive one per anomaly at a
fixed rate, whereas a field under attack makes more checks. *A firm or institution:* reasonable for a PR or legal
team paid to explain away failures; weird that its budget never runs out (fixed `F`). *A population of
organisms:* a stretch I would not defend (a search for consistent variants under accumulating constraints has the
same `(1-q)^k` shape, but nothing here is selected). *A cell:* none.

**What this does not establish.** That excuse-making searches at random: a reasoning fixer has a smaller
effective `q` per check, which lowers the per-doubling base `1/(1-q)` and raises the slope, roughly as
`1/log2(1/(1-q))`; nor anything with distance tied to specificity, other `q`, or `d = 2`. (j5) has no native run.
