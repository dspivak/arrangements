# Science at two scales, test (k): publishing failed replications (predictions, before code, 2026-10-01)

A companion to `SCIENCE-SCALES-PREDICTIONS-2026-10-01.md`, building on §8 (the native round) and §5.7 stage 2
(conclusion as a seed vs as evidence). David, 2026-10-01, asked whether a failed replication is published back or
the seed quietly dies: "both can happen." To the proposal to test publishing failures: "run it." Design and
predictions by Claude Opus 5.5, 0fe746c1, before any code. Results to §3.

## 1. Setting (native; reuse `science_native.py`'s stage-2 construction, its settings and seeds)

Stage 2's seed arm (m = 8 holders, ring and all-to-all, n = 20 hypotheses, disjoint slices, the truth in one slice,
eps, tau, dt and 150 draws as fixed there, 20 seeds). New: each holder has a second out-port, its **failure report**:
for each hypothesis it has received as a planted seed, how far its own data have driven that hypothesis's weight
back down (a smooth, nonnegative function, zero for a hypothesis never received; the run fixes its form before
code). Accounts still go forward only. Three arms, same everything else:
- **silent:** stage 2's seed arm (reports not wired), as recorded.
- **report lowers the seed:** a reader plants less of a hypothesis that neighbours report failed (its seed term
  for that hypothesis scaled by a smooth `1 - g(report)`); its evidence is untouched.
- **report counted as evidence:** a reader's fitness for that hypothesis is lowered by `beta * report`, as if the
  report were data against it, `beta in {0.1, 1, 10}`.

## 2. Predictions

- (k1) Lower-the-seed: 0 of 20 seeds end with any holder on a wrong hypothesis (silent: 0); the truth reaches every
  holder in at least as many seeds as silent (19 of 20 ring, 20 of 20 all-to-all); arrival time per ring hop within
  10% of silent's; the seed mass planted on hypotheses later rejected is lower than silent's in at least 15 of 20
  seeds.
- (k2) Counted as evidence: at `beta = 0.1` within 2 seeds of silent; as `beta` grows, the count of seeds in which
  the truth never reaches some holder rises, above zero at `beta = 10`. A true hypothesis can be reported dead by
  bad luck and the reports keep it dead: a negative echo chamber.
- (k3) Mechanism: in (k2)'s failures, the first report against the truth comes from a holder whose first few local
  draws favoured a neighbouring hypothesis; the run reports, per failing seed, whether that holds.
- (k4) Same split as stage 2: a report used to lower a seed is safe, one counted as evidence is not.
**What it cannot show.** That journals publish failures this way or that readers weigh them so; the report's form,
`g` and `beta` are stipulated.

**Numbers fixed before code** (the run's, Claude subagent of Claude Opus 5.5, 0fe746c1, 2026-10-01; written after a
scratch script that ran only stage 2's own code, `science_native.conclusions_native_run`, seed arm, truth 5.0, and
before any construction code for (k)).
*Settings.* Stage 2's, unchanged: `m = 8`, `n = 20` (`mu = 0, 0.5, ..., 9.5`), slices `{2j, 2j+1}`, truth 5.0
(index 10, holder 5's slice) only, ring and all-to-all with neighbours averaged (`W = A / degree`), seeds 0-19,
`default_rng(s).normal(5.0, 1, (150, 8))`, `eps = 0.05`, `tau = 0.05`, Euclidean frame (run 2's: replicator field
as the non-exact data forcing), `dt = 1/160`, each draw held 160 steps, 24,000 steps of `Phi`. (`alpha` is
unused: no arm has the evidence bonus.)
*The failure report.* Each holder's state is `(p, z, a)`, all Euclidean with response `dt`. `z_h` integrates
the data's push on `p_h` with the sign reversed, forcing `-p_h (f_h - <f, p>)` (minus the replicator field's
`h` component, own draws only, never the reports): the net weight its own data have taken off `h`, per draw
held. `z_h = 0` exactly for a hypothesis never received (its `p_h` stays exactly 0). The report is
`R_h = exp(-z0 / z_h)` for `z_h > 0`, `0` otherwise, `z0 = 0.5`: smooth (C-infinity, not analytic), nonnegative,
below 1, zero for `z_h <= 0`. Out-port: `(c, R)`, `2n` coordinates; a reader averages its neighbours' reports
as it averages their conclusions, `R̄ = W R`. `a_h` is an observation register, forcing the planted inflow
`eps c̃_h`; it feeds nothing back.
*The arms.* Silent: reports not read; `p`'s forcing is stage 2's seed arm. Lower-the-seed: `g(R̄) = R̄`, so
`c̃ = (1 - R̄) ⊙ c̄`, `S = sum c̃`, seed forcing `eps (c̃ - S p)`, the differential of
`eps (<c̃, p> - S |p|^2 / 2)` in `p`: still exact, and it sums to zero, so the simplex is kept (scaling stage
2's term `eps (c̄_h - p_h)` per component would not sum to zero; this is the disclosed change). Evidence: data
fitness `f - beta R̄` in the replicator field, seed unchanged, `beta in {0.1, 1, 10}`. In every arm the forcing
on in-ports is zero. Step size: `dt (largest per-draw gap 40.6 + beta 10 + eps + 1) = 0.32 < 1`, so weights
stay positive; `dt` unchanged.
*Measures.* "The truth reaches holder `j`": its weight on the truth exceeds 1/2 after some draw (stage 2's
arrival); "never reaches some holder": a seed in which some holder's weight on the truth never exceeds 1/2.
"Ends on a wrong hypothesis": weight `> 0.99` on one non-true hypothesis at the last step. Per-hop arrival: the
ring's least-squares slope of median delay against distance from the first finder (stage 2's). "Seed mass
planted on hypotheses later rejected": per seed, the sum over holders `j` and hypotheses `h` with final weight
`< 0.01` at `j` of `a_{j,h}` at the last step (the integral of `eps c̃_{j,h}` over all steps). Also reported:
min weight, `max |sum p - 1|`, every holder's top the truth at the last step.
*Silent arm, from the scratch (stage 2's code).* Ring: the truth reaches every holder in **19 of 20** (seed 14:
holders 0 and 1 never; top-at-end also 19), wrong 0/0, slope **10.0** draws per hop. All-to-all: **20 of 20**,
wrong 0/0. These are (k1)'s baselines; the run's silent arm must equal stage 2's weights to `1e-12` at every
recorded draw.
*Verdicts fixed now.* (k1) holds iff, per wiring: wrong count 0; reach count `>= ` silent's; ring slope within
10% of silent's (`[9.0, 11.0]`); planted-on-rejected lower than silent's in `>= 15` of 20 seeds. (k2) holds iff,
per wiring, with `c(.)` the never-reaches count: `|c(0.1) - c(silent)| <= 2`, `c(0.1) <= c(1) <= c(10)`, and
`c(10) > c(silent)`. (k3), per failing seed (never-reaches in evidence and not in silent, any `beta`): the
first report against the truth is the first (draw, holder) at which some holder's `R` on the truth exceeds
0.01; it holds for that seed iff that holder's 5 draws ending at that draw have a summed log-likelihood
favouring 4.5 or 5.5 over 5.0. **Base rate from the scratch: 0.59** of all (seed, holder) 5-draw windows do,
since the test takes the better of two neighbours; so (k3) is informative only if its fraction in failing seeds
is well above 0.59, and the run reports both. (k3) as a whole holds iff it holds in every failing seed; with no
failing seed it is untested. (k4) holds iff (k1)'s wrong count and reach count hold and
`c(10) > c(silent)` in at least one wiring. Closed form: with reports silent, `z` and `a` change nothing, so
silent = stage 2 to round-off.

## 3. Results

Run 2026-10-01 by Claude (subagent of Claude Opus 5.5, 0fe746c1). Code: `arrangements/science_failure.py`
(imports `science_native.py`, which is unchanged); tests `tests/test_science_failure.py`.

**Run 1, failed construction (recorded before any change).** The whole file, 54 s: every ring run, in every arm
(the silent arm included), was NaN in 20 of 20 seeds; every all-to-all run was finite, and its silent arm equalled
stage 2 to `0.0` (bit-identical). Cause, found with a one-seed probe (finite at hold 1 and 2, NaN within 3 draws at
hold 160): the report `exp(-z0/z)` at a tiny positive `z` has derivative `0 * inf = NaN`, and `compose_seq`'s
pullback evaluates the readout's derivative, times a zero cotangent, so the NaN enters `p` even where no one reads
the report. On the ring some `z_h` sit at tiny positive values. **Fix, disclosed:** the report is cut to 0 below
`z = z0/700`, where its value is under `1e-304`; so `R` is smooth except for a jump below `1e-304` at that point.
Nothing else changed; the file was re-run once with the same settings. Run 1's all-to-all numbers, for the record:
(k1) held all-to-all (wrong 0/0, reach 20, planted-on-rejected lower in 19 of 20); (k2) never-reaches counts 0, 0,
0, 0 (silent, beta 0.1, 1, 10), so its "rises, above zero at beta 10" failed all-to-all.

**Run 2, recorded (53 s; then re-run unchanged, 53 s, only to mark the refuted tests strict xfail, the same
numbers).** No NaN in any run; min weight exactly 0, `max |sum p - 1| <= 7.8e-16` in every arm and wiring. The
silent arm equals stage 2's seed arm **bit-identically** (`max |p - p_stage2| = 0.0`), ring and all-to-all, so the
`z` and `a` blocks change nothing when no one reads the reports.

**(k1) Held, both wirings.** Same settings, lower-the-seed against silent: ends on a wrong hypothesis 0 of 20 against
0 of 20, ring and all-to-all. The truth reaches every holder: ring **20 of 20** against silent's 19 (seed 14, lost
by silent's holders 0 and 1, is reached); all-to-all 20 against 20. Ring slope **9.90** draws per hop against 10.00
(median delays 0, 8, 21, 32, 37.5 against 0, 8, 21, 32, 38), inside `[9.0, 11.0]`. Seed mass planted on hypotheses
later rejected: lower than silent's in **19 of 20** seeds, ring and all-to-all, but by little (median 9.60 against
9.77 ring, 5.93 against 6.01 all-to-all, about 2%): a report needs the data to have removed weight, and a seed
held at tiny weight loses little, so most planted mass on rejected hypotheses is never reported.

**(k2) Refuted, both wirings.** Seeds in which the truth never reaches some holder, at the same settings,
silent / `beta = 0.1` / `1` / `10`: ring **1 / 0 / 0 / 0**, all-to-all **0 / 0 / 0 / 0**. "Within 2 of silent at
0.1" held; "rises, above zero at 10" did not: counted as evidence, reports never kept the truth from any holder,
and on the ring they rescued silent's seed 14. Ends on a wrong hypothesis: 0 of 20 at every `beta`. Mechanism of
the refutation (read off the runs, not predicted): reports against the truth occur (some holder's report on it
above 0.01 in 16 of 20 seeds at `beta = 10`, at most 0.40 ring and 0.59 all-to-all), but reports against its
neighbours 4.5 and 5.5 are larger (at most 0.73 ring and 0.88 all-to-all), because the data remove more weight from a wrong
hypothesis that was held than from a true one; so counting reports pushes holders off the wrong neighbours
faster (median arrival draw 19 against silent's 24 on the ring, 13 against 14 all-to-all).
*What did happen, post hoc, not a prediction:* at `beta = 10` the truth arrives and is then lost at some holder in
a few seeds. Every holder's top at the last step is the truth: ring 19 of 20 (silent 19, but a different seed:
seed 3, holders 4 and 6, which had reached 0.83 and 0.98 on the truth, end on 5.5 at 0.06 and 0.04), all-to-all
**18 of 20** (silent 20; seeds 3 and 19, one holder each, from 0.91 and 0.94 down to 0.49 and 0.15). In each, the
holder's own data favour the truth (its own report on it is 0), and its neighbours' averaged report on the truth
(0.11, 0.05, 0.02; one reporter each, at 0.23, 0.35, 0.15), times `beta = 10`, is 1.1, 0.5 and 0.2 nats per draw
against the truth's expected advantage of 0.125. That is the negative echo
chamber's shape, after arrival rather than before it, in 1 and 2 of 20 seeds against silent's 0 and 0.

**(k3) Untested** (no seed fails under the fixed measure). Applied post hoc to the three late-loss seeds above: the
first report against the truth comes from holder 5 at draw 5 (ring seed 3 and all-to-all seed 3; its 5 draws favour
5.5 over 5.0 by 0.72 nats) and from holder 4 at draw 11 (all-to-all seed 19; favour 4.5 by 0.99): 3 of 3, against
the base rate 0.59, so suggestive and no more. In ring seed 3, holder 5 ends at 0.992 on the truth and still
reports it failed (0.23): its early draws took weight off the truth, the seed returned it, and a holder near 1 on
a hypothesis gains no weight from its data, so `z` never comes back down. The report is a fossil of early bad luck.

**(k4) Refuted.** Lowering the seed is safe (both wirings: no wrong ending, the truth reaches as many holders as
silent or more); counting reports as evidence at `beta = 10` loses the truth from a holder that never had it in no
more seeds than silent (ring 0 against 1, all-to-all 0 against 0). Stage 2's split, seed safe and evidence not, is
not reproduced by failure reports at these settings. The only harm from counting is the post-hoc late loss above.

**Which results could have come out otherwise.** The silent arm's equality with stage 2 was the closed form fixed
before code; it checked the construction. Everything else was open: (k1)'s 20 and 9.90 and 19 of 20, (k2)'s zeros,
the late losses, and the fossil report. (k1)'s "planted on rejected lower" held in 19 of 20 but by about 2%, so it
is a weak result. (k2) was refuted by the measure as fixed; the one place the predicted mechanism appears is a
measure chosen after the run (top at the last step), which I report and do not count for (k2).

**Reasonable or weird** (the run's judgment, not evidence). *A scientific community:* reasonable that a published
failure used to deprioritise a lead costs nothing, and that failure reports mostly help, since wrong leads fail more
visibly than true ones. Weird: the report is counted again at every draw, so a standing report of 0.2 at `beta =
10` is worth two nats per experiment forever, test (f)'s no-provenance overcount in another form; and a lab still
reports a failure it has since overturned, because the report measures weight lost, which a lab near certainty
cannot regain. Real failure reports are retracted or superseded; this one is not. *A person:* hearing that someone
else's test failed and discounting a belief you are confirming at your own bench every day is plausible at low
`beta`, implausible at `beta = 10`. *A firm or institution:* a post-mortem that lowers investment in a direction
(lower-the-seed) is the sane use; one scored into every later decision (evidence) is how a firm abandons a product
that its own numbers support (the late loss); a loose analogy. *A population of organisms:* the report has no
clear counterpart; reading the seed as migration, lowering it is fewer migrants of a type that failed nearby,
which is reasonable; a fitness penalty for a type because it failed in a neighbouring deme is not biology. *A
cell:* none I would defend.

**What this does not establish.** That reports counted as evidence are safe: only that, with this report (net weight
the data removed, `exp(-0.5/z)`), these `beta`, and stage 2's settings at truth 5.0, they did not keep the truth
from any holder that never had it; the post-hoc late losses (1 and 2 of 20) say they can take it from holders that
had it. Not run: other report forms (a retractable or decaying report would remove the fossil), other `eps`, `tau`,
`z0`, the truth in no slice, or anything about how journals or readers handle failures.

## 4. Correction: a report says what the holder now finds (predictions, before code)

Claude Opus 5.5, 0fe746c1, 2026-10-01, on §3's unpredicted finding that "a holder at 0.99 on the truth still
reports it failed": that is a defect of the construction, as the dead bettors who kept publishing were in test (h)
(David: "if you're dead, you should have fallen apart"). A failure report should say what the holder's data now
say about a planted hypothesis, not a lifetime tally of weight once lost. Fix: the report is a smooth, nonnegative
function of the **current** shortfall, how far the hypothesis's present weight sits below the seed mass planted on
it, so a holder that has come to hold the hypothesis reports nothing against it. Same arms, settings and seeds.

**Predictions.** (k5) Lower-the-seed: §3's (k1) counts within 1 seed each way. (k6) Counted as evidence: the
truth, once reached, is lost in 0 of 20 seeds at every beta on both networks (§3: 1 and 2 of 20 at beta 10), since
holders who hold the truth stop reporting against it. (k7) The never-reached counts stay within 1 of §3's.
Results to §5.

**Numbers fixed before code (§4)** (the run's, Claude subagent of Claude Opus 5.5, 0fe746c1, 2026-10-01; after a
scratch script that ran only §3's code, and before any §4 construction code).
*The planted mass, current.* A new state block `b` (Euclidean, response `dt`, starting at 0) whose forcing is the
seed forcing with `b` in place of `p`: `eps (c̄ - b)` (silent, evidence), `eps (c̃ - S b)` (lower). So `b_h` is the
weight the seed alone would now have put on `h`, planted mass that decays at the seed's own rate; it lies in
`[0, 1]`, tracks the neighbours' current conclusions, and is exactly 0 for a hypothesis never received.
*The report.* Shortfall `x_h = b_h - p_h`; report `R_h = exp(-x0 / x_h)` for `x_h > 0`, else 0, `x0 = 0.5` (the
§3 `z0`, unchanged). So a holder whose present weight on `h` is at or above what the seed planted reports nothing
against `h`, whatever it once lost. The guard is smooth, not a cut: `R` is computed by `jnp.where` on `x > 0` with
a safe argument, and its derivative by a custom JVP as `exp(log x0 - x0/x - 2 log x)` for `x > 0`, 0 else, which is
the exact derivative evaluated without `0 * inf`. So `R` is exactly C-infinity (not analytic at 0) and §3's
`1e-304` cut is gone. `z` is no longer read by anyone (kept in the state only as a record); `a` unchanged. Arms,
settings, seeds, `g(R) = R`, `beta in {0.1, 1, 10}`, `dt`: §3's.
*Measures.* "Lost after reach": a holder whose weight on the truth exceeded 1/2 after some draw and whose top at the
last step is not the truth; a seed counts if any holder is. §3's counts from the scratch (§3's construction):
ring silent **1** (seed 14, holder 2: reached 0.626, ends on 4.5), lower 0, evidence 0, 0, **1** (seed 3, holders 4
and 6) at `beta` 0.1, 1, 10; all-to-all silent 0, lower 0, evidence 0, 0, **2** (seeds 3 and 19). Silent does not
read reports, so its 1 and 0 stand in §4.
*Verdicts fixed now.* (k5) holds iff, per wiring, lower-the-seed's wrong count, reach count and
planted-on-rejected-lower count are each within 1 of §3's (0; 20 and 20; 19 and 19); the slope is reported.
(k6) holds iff the evidence arm's lost-after-reach count is 0 at every `beta`, both wirings. (k7) holds iff each
never-reaches count (lower, and evidence at each `beta`) is within 1 of §3's (all 0, both wirings). Silent must
equal §3's silent bit-identically.

## 5. Results of the correction

Run 2026-10-01 by Claude (subagent of Claude Opus 5.5, 0fe746c1). Code: `science_failure.py`, `form="shortfall"`
(§3's form kept as `form="tally"`; a `b` block added to both, which §3's dynamics never read); tests appended to
`tests/test_science_failure.py`. **One recorded run** (the whole file, §3 and §4 together, 128 s): no NaN and no
construction failure, so no re-run was needed for the numbers; the file was re-run once, unchanged except that the
refuted tests were marked strict xfail. Every arm stayed on the simplex (min weight exactly 0, `max |sum p - 1| <=
7.8e-16`); §4's silent arm equals §3's silent **bit-identically**, both wirings. The custom-JVP guard needed no
cut: no NaN anywhere.

**The fossil is gone.** (Seed, holder) pairs above 0.99 on the truth at the end that still report it failed
(report above 0.01): §3, across arms, 0-1 ring and 2-4 all-to-all; §4, **0 in every arm and wiring**.

**(k5) Held on the ring, refuted all-to-all.** Lower-the-seed, (wrong, truth reaches every holder, planted on
rejected lower than silent), §3 against §4 at the same settings: ring (0, 20, 19) against (0, 20, 19), slope 10.00
(§3 9.90); all-to-all (0, 20, 19) against **(0, 20, 17)**, a difference of 2 against the 1 allowed. Medians of
planted-on-rejected 9.70 ring and 5.93 all-to-all (silent 9.77 and 6.01): lowering still saves about 1%.

**(k6) Refuted, both wirings.** Seeds in which the truth, once reached, is lost, §3 / §4, at `beta` 0.1, 1, 10:
ring 0/0, 0/0, **1/1**; all-to-all 0/0, 0/0, **2/2**. §4's losses are other seeds and a different event: ring seed
3, all-to-all **seeds 4 and 8**, and in each **every** holder, after reaching 0.73-1.0 on the truth, ends with its
top on 5.5, 4.5 and 5.5 respectively (seeds with some holder above 0.99 on a wrong hypothesis at `beta = 10`: §4
ring 1, all-to-all 2; §3 0 and 0). Silent and
lower: ring 1 each (seed 14, holder 2, as silent's), all-to-all 0.
*Mechanism, read off the runs (not predicted).* The shortfall `b - p` cannot tell "my data reject it" from "I have
not caught up yet". `b` follows the neighbours' conclusions with the seed's slow time constant (`1/eps = 20`
draws), while `p` falls fast under a run of draws favouring 5.5 (or 4.5), and the sharpened readout `p^20` falls
faster still. So a dip in the truth at a few holders opens a shortfall at all of them at once: at the peak every
holder reports the truth failed at 0.28-0.51 with its own weight on it 0.00-0.24 and `b` 0.40-0.77; times
`beta = 10` that is 3-5 nats per draw against the truth's 0.125. The reports deepen the dip that made them. The
first report above 0.01 comes at draw 24, 14 and 22, from holders 5, 4 and 0, each then at 0.27-0.38 on the truth
with `b` 0.40-0.53. This is (k2)'s negative echo chamber, community-wide, produced by the corrected report.

**(k7) Held.** Never-reaches counts, §3 / §4: lower 0/0, evidence 0/0 at each `beta`, both wirings.

**Which results could have come out otherwise.** Silent's equality was a closed form. The fossil's disappearance
was nearly built in (a holder above `b` reports nothing); that it held in every pair was open. Open and refuted:
(k6), and (k5) all-to-all by one seed beyond the margin. The collapse at `beta = 10` was not foreseen by me or by
§4's predictions.

**Reasonable or weird** (the run's judgment, not evidence). *A scientific community:* reasonable that a report no
longer outlives its author's change of mind; weird, and instructive, that "my result falls short of what you
published" is reported the moment a lab dips, before its own data have had time to decide, and that counting such
reports as evidence at high weight brings down a correct consensus in 1-2 of 20 communities: a replication-crisis
panic, every lab's shortfall feeding every other's. Real labs report a failure after a finished study, not a
running deficit. *A person:* doubt reported mid-experiment and taken by others as a result, a loose analogy.
*A firm or institution:* a dashboard metric "below plan" read as "failed" and fed back into the plan is a known
way for a firm to talk itself out of a working product; analogy. *A population of organisms:* none I would defend
for the report as evidence. *A cell:* none.

**What this does not establish.** That a current report is unsafe in general: only this one (shortfall against an
EMA of planted mass, time constant 20 draws, `exp(-0.5/x)`), at `beta = 10`, at stage 2's settings. A report that
waits for the holder's own data (say, the shortfall weighted by how many draws it has tested the hypothesis) is the
obvious next form, not run. Lowering the seed stayed safe (0 wrong, 20 and 20 reached) under both forms.
