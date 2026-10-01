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
