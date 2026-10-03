- 2026-10-01 — **Test (g4) diagnosis** (Claude (subagent of Claude Opus 5.5, 0fe746c1)). The native
  (g4) run disagreed with its closed form. Shrunk to one mirrored figure, n_A = 3, B counts 0-8: the two
  agree. At 8 figures and n_A = 20 they part at count 0 by exactly K_0 = 7.575. Cause: my chooser's
  dU/dlambda by autodiff through sigmoid, whose derivative sigma(1 - sigma) underflows to 0 for
  lambda > ~37 and drops K_0's term; the closed form was right. Fixed by writing
  p0 p1 (K1 - K0 - T lambda) with a stable p0 p1. After the fix, seed 0, d = 3, B counts 0-40: native
  n_B* equals the closed form in all six unilateral arms (soft 22/23/30, hard 22/23/26). The
  one-potential arm gives 30 against 23, as predicted. Not rerun at the full settings (dense response,
  hours). Results in §5.8, "(g4) diagnosis".
- 2026-10-01 — **Science at two scales, test (g): explaining away by building excuses** (Claude (subagent
  of Claude Opus 5.5, 0fe746c1)). Numbers and the convexity call fixed in the predictions note §9 before
  code. `science_native.excuse_world` / `excuse_reader` / `excuses_native_run` (world + reader wired, run
  by `Phi`; attachments on `choice.py`'s {attached, not} chooser chart); 9 tests, file 30 passed + 3 strict
  xfail. Native equals the numpy scratch to 1e-16. Convex: (g1), (g2) held, (g3) refuted as called
  (gradual: threshold move 0.100 < earlier 0.150). Double well (an attachment is made or not): (g1)-(g3)
  held (jump 0.124 after exact zeros). Found: old excuses are dropped once theta moves; figure 4 lost a
  rate race. (g4): closed forms only (symmetric 21.5 = (e); incumbent 28 hard / 34.5 soft against (e)'s
  82.5); the native (g4) smoke run disagreed with the closed form (cause not found) and was slow (dense
  response matrix); code marked UNVERIFIED. Results in §5.8.
- 2026-10-01 — **Science at two scales, native round, stage 2: (n3)-(n5)** (Claude (subagent of Claude
  Opus 5.5, 0fe746c1)). Stage-2 numbers and an analysis fixed in the predictions note §8 before code
  (eps 0.05, alpha 0.5, tau 0.05, dt 1/160 from the largest per-draw fitness gap 73.8, each draw held 160
  steps, 150 draws, 20 seeds). `science_native.conclusion_holder` / `conclusions_native` /
  `conclusions_native_run`; 9 stage-2 tests, file 23 passed + 2 strict xfail, 40 s. Run 1 (Shahshahani,
  as designed): seed arm NaN in 20 of 20 (infinite differential on the face). Run 2 (disclosed before
  code: Euclidean response, replicator field as non-exact data forcing, exact seed): never left the
  simplex. (n5) refuted: planting needs a non-exact piece (or an infinite one). (n3) refuted on the ring
  (19 of 20; 10.0 draws per hop against [12, 48]), held all-to-all and with the truth in no slice (7.5,
  20 of 20). (n4) held only trivially (evidence 20/0: it cannot create support); plant-and-count 7/6
  ring, 13/13 all-to-all against seed 0/0. One assertion failed on the first full run; the case was
  marked strict xfail and the file re-run unchanged. Results in §5.7.
- 2026-10-01 — **Science at two scales, native round, stage 1: (n1), (n2)** (Claude (subagent of Claude
  Opus 5.5, 0fe746c1)). Stage-1 numbers fixed in the predictions note §8 before code (numpy, no
  arrangement). New `arrangements/science_native.py`: `port_wire` (R^- of a function on port
  coordinates; `graph_wire` needs one port dimension), `wire_boxes`, `draw_source` (clock world, table
  lookup), readers whose forcing is zero on their in-ports, `community_native`, `summaries_native`; 16
  tests in `tests/test_science_native.py`, 16 passed, 10 s, one run, no re-run. (n1) held: the wired
  world + 10 holders equals round two's monolithic box, bit-identical in 322 of 400 runs, worst 4.1e-15
  relative, all classifications reproduced. (n2) held: no provenance equals round two to 7.1e-12 in
  weights, 7/20 (5.0) and 9/20 (5.2) locked in both wirings; journal reader equals the pooled closed form
  to 4.5e-13. Gaps: a readout cannot depend on the current input (a one-step register bridges it, shift
  of 2-3 steps); no smooth source of a fixed sequence (clock + lookup). Results in §5.7.
- 2026-10-01 — **Science at two scales, second round, test (f)** (Claude (subagent of Claude Opus 5.5,
  0fe746c1)). m=8, n=20, K=2 disjoint slices, r=5, c=0.1, T=60(+4)/150, 20 seeds, ring and
  all-to-all, definitions fixed in the predictions note §7 before code. `science.run_summaries`,
  `run_conclusions`, `run_bounded_mind` (bookkeeping in plain code; beliefs moved by the jitted
  translation arrangement `translate_beliefs`). No construction failure, no re-run. (f1) held (2.3e-13);
  (f2) refuted on the ring (early steps less confident), its lock count held (7 of 20 wrong vs 0);
  (f3) refuted as written (all holders > 0.99 in 18 of 20; conclusion = truth 20 of 20; 7.1 steps per
  hop; never found outside every slice; bounded mind only with the truth in its K); (f4) held (11-12 of
  20 wrong without replication vs 0 with). Results in §5.6. Also moved §5.5 back inside §5 (it had
  landed after §6/§7 through end-of-file appends).
- 2026-10-01 — **Science at two scales, second round, test (e)** (Claude (subagent of Claude Opus 5.5,
  0fe746c1)). n_A=20, d in {3, 2}, 10 seeds, B counts 0-150, closed-form crossings and definitions
  fixed in the predictions note §6 before code. `science.rival_chooser` (test (b)'s chooser plus trust
  boxes, totals, one jitted traced build per arm). First run, as fixed: NaN from step-size overshoot
  (seed 7 at B count 0; fixed arm d=3 after the crossing), so (e1) and (e3) at d=3 and (e4) were not met
  (strict xfail). Re-run 1 (dt sized from the closed-form jump) failed at the cold start in "both learn";
  re-run 2 (cold dt from the starting gap) made all 120 crossings equal the closed form, and (e1)-(e4)
  held. At d=3, fixed / both / incumbent-only, the rival overtook by B count 31 / 31 / 92 in 10 of 10.
  Unpredicted: a losing hypothesis's trust stops learning (logit off by up to 4.6, no crossing moved).
  Results in §5.5.
- 2026-10-01 — **Science at two scales, test (d)** (Claude (subagent of Claude Opus 5.5, 0fe746c1)).
  Charts, eps {0, 1e-3, 1e-2, 1e-1}, 120 steps, seeds 0-4, definitions and closed-form values fixed in
  the predictions note §4 before code. `science.mutation_selection` (abundance chart, response diag(l),
  exact forcing d(1/2 N^T (M - D_l^{-1}) N); exact because M is symmetric, and cheap) and
  `science.drift_filter` (probability chart, direct non-exact forcing). (d1), (d2) held (3.5 first
  largest at steps 39/25/20, seed 0; spread 0.22/0.23/0.38; peak reached, not kept); (d3) refuted at
  the 1e-12 fixed before code (1.1e-11, from the population realization's cancellation; strict xfail).
  Results in §5.4.
- 2026-10-01 — **Science at two scales, test (c)** (Claude (subagent of Claude Opus 5.5, 0fe746c1)).
  Derivation of both account steps under the inverse-Fisher response written into the predictions
  note §3 before code (both predicted steps right); settings fixed there (m=10, n=5, ring and
  all-to-all, 20 seeds, T=100, "locked" defined). Realized in the gauge-free log-weight chart
  (`science.account_community`), not `inverse_fisher`, because of (a1). (c1), (c2), (c4) held; (c3)
  held at alpha/beta 0, 0.1, 1 and refuted at 10 (synchronous agree step unstable, 0 of 20 both
  wirings; strict xfail). Evidence arm locked wrong 0, 0, 4, 15 of 20 in both wirings; agree 0
  throughout. Results in §5.3.
- 2026-10-01 — **Science at two scales, test (b)** (Claude (subagent of Claude Opus 5.5, 0fe746c1)).
  Numbers fixed before code in `notes/SCIENCE-SCALES-PREDICTIONS-2026-10-01.md` §2 (L=10, c=1.6, T=1,
  n=200, 41-point d grid; (b4) judged untestable as worded, sharpened to (b4') before running).
  `science.holder_chooser` (Shahshahani simplex chooser, Euclidean `ell` box tensored in the learned
  arm, constant shift B for choice.py's CONDITION); 6 tests, all passed. (b1)-(b3), (b4') held to
  1e-15; tally fixed hold 15 / drop 26, learned hold 40 / drop 1. Results and the guaranteed-by-
  construction caveat in §5.2. Also added a dated scope sentence (exact in the mathematics, lost by
  the dense inverse below ~1e-5 weight) to `notes/BAYES-INTEGRATOR.md` §1 and `STATUS.md` §3.24.
- 2026-10-01 — **Science at two scales, test (a)** (Claude (subagent of Claude Opus 5.5, 0fe746c1)).
  Predictions fixed before code in `notes/SCIENCE-SCALES-PREDICTIONS-2026-10-01.md` §1 (numbers added
  there before code); new `arrangements/science.py` (mind = closed `belief_arrangement` rebuilt per
  datum; population = n holder boxes, log abundance, constant forcing, as a tensor and as one box) and
  `tests/test_science_scales.py` (3 passed, 3 strict xfail). (a2) held exactly; (a3) held for the
  population; (a1) and the mind's (a3) refuted: the inverse-Fisher step `G^{-1}(G t)` loses the exact
  translation once the least-favoured hypothesis's weight falls below ~1e-10 (cond(G) ~ 1/min q),
  max |dq| 2.2e-4 well specified, NaN from step 110 misspecified. Not repaired; results in §5.1.
  Library change suggested, not made: a closed-form or better-gauged `G^{-1}` in `choice.py`.

- 2026-09-29 — **P30 (d) word of mouth and (e) many parts** (Claude Fable 5.1, session `47a21ac3`).
  Predictions fixed before code in `notes/WORD-OF-MOUTH-PREDICTIONS-2026-09-29.md` and
  `notes/MANY-PARTS-PREDICTIONS-2026-09-29.md`; `tests/test_many_parts.py` (5 passed, 16 s) and
  `tests/test_word_of_mouth.py` (6 passed, ~8 min), run one piece at a time under
  `scripts/run_tests.sh`; the baselines `test_mutual_need.py` and `test_turnover.py` still pass.
  (e): the supply threshold is the pair's for every `n` and both wirings (symmetric mode); a strict
  ring dies from one cut, a slack one is graded downstream; all-to-all survives a loss iff
  `s > 2 sqrt(K(n-1)/(n-2))`; the collapse is a front. (d): the return is exact and slower (`1/g`),
  no floor unless the leaver was the whole channel and is frozen dead; a propped leaver whispers;
  compounding is the channel thinning; the dependent falls by its wiring and everyone is propped;
  a doodad reading the joint's account closes the whole, one reading the whole's felt deficit does
  not; a fast alarm on a stiff joint rings in discrete time (alarm slowed). STATUS §3.30, §3.31;
  results notes beside the predictions. No commit authorized yet.

- 2026-09-28 — **P29, town's mathematician** (Codex/GPT-6, session `0c750220`). Design
  fixed before code in P29-TOWN-PREDICTIONS-2026-09-28.md. Built town.py from Arrangement
  primitives; 11 tests across four individually capped stage files passed. Predictions held:
  asymmetric smoke, late evidence, historical premium lag, earlier account reducing harm.
  Final initialization corrected to avoid a phantom first observation; affected checks rerun.
  55.2813% less harm over 30 units (55.1688% at half dt), with disconnection and informed-plant
  controls. Unlimited credit; positive fund with her account is below unpaid claims. No claim
  of solvency, wiring discovery or book understanding. STATUS section 3.28; full scope,
  measurements and proposed Ch 1 / Ch 5 placement in P29-TOWN-RESULTS.md. Book recovery log:
  ../articulating-forms-that-matter/notes/P29-RUNNING-LOG.md. No commit authorized yet.

- 2026-09-28 — **P29 closed** (Codex/GPT-6, session `0c750220`). David answered “yes”
  to Chapter 1 story / Chapter 5 construction placement and checkpoint commits in both
  repositories. Approval recorded in results and STATUS; all 11 checks already passed.
  Committing the authorized repository dump with subject “Complete P29 town mathematician”.
- 2026-09-29 — **P5 (b), active-inference tests** (Claude Opus 5.5, session `9b16a421`).
  Predictions fixed first (`notes/ACTIVE-INFERENCE-TESTS.md` §§1–3); `arrangements/active_inference.py`,
  `tests/test_active_inference.py` (17 passed, 6 strict xfails = refuted predictions). (i) held;
  (ii-a)/(ii-c) refuted by a drift off the simplex when `log Z < 1`, a property of the state in `R^n` noted in §3.24 (docstring
  of `choice.py` and STATUS §3.24 updated); Addendum A written after the failure and before more code,
  held; (iii) held for rollout and information gain, choices held once the values are shifted; exact
  wiring is wishful thinking. STATUS §3.29.
- 2026-09-29 (Claude Fable 5.1, ec5ece87). `scripts/maxwell2d.py` added: the light-on-a-lattice numpy check of the book plan's §5.9 item 31, written by session c306598e on 2026-09-28 in a scratchpad the system clears, rerun here with the recorded numbers reproduced (peak radius 60 at time 60, energy drift 2.0%, Gauss change 4e-17). Not a library test; the library test with predictions is the book plan's P32 (a). Found by the springs-to-towns audit (`notes/SPRINGS-TO-TOWNS-AUDIT-2026-09-29.md` in the book repo).
- 2026-09-29 — P5 (b) addenda B and C (Claude Opus 5.5, `9b16a421`): `natural_chooser` (fix for (ii)),
  `hand`/`predictor`/`hierarchy` (David's picture, nested); predictions note §9, verdicts §10, all held;
  25 passed, 6 strict xfails. STATUS §3.29.
- 2026-09-29 — **P32 (c) swing locking** (Claude Fable 5.1, ec5ece87). `tests/test_swing_locking.py`, 4 passed in 2 s; predictions fixed first in `notes/P32-C-PREDICTIONS-2026-09-29.md`, all held: with `coalgebra.damped` the Kuramoto pair locks at arcsin(dw/2K) and advances mean(omega)/(c m) per step; undamped it never settles; overshoot iff the linearization's eigenvalues are complex. STATUS §3.32.
- 2026-09-29 — **P32 (e) Kirchhoff node** (Claude Fable 5.1, ec5ece87, run by a subagent). `tests/test_kirchhoff_node.py`, 6 passed in 21 s; predictions in `notes/P32-E-PREDICTIONS-2026-09-29.md`, all three held: a capacitor copied to three resistors steps by −dt·ΣI_j to 1e-15 with no sum written anywhere; a capacitor triangle conserves charge; two conductances rest at the conductance-weighted mean. Limit: a chargeless nodal-analysis node is an algebraic constraint the box does not express. STATUS §3.33.
- 2026-09-29 — **P32 (a) light in the library** (Claude Fable 5.1, ec5ece87, run by a subagent). `arrangements/light_lattice.py`, `tests/test_light_lattice.py`, 6 passed in 16 s; predictions in `notes/P32-A-PREDICTIONS-2026-09-29.md`: four held (forcing emerges; one step equals `scripts/maxwell2d.py`'s to 1e-12; Gauss conserved to 1e-12; lattice dispersion to 1e-9), the energy prediction refuted as written (plain energy swings 33–119% at step 0.5, bounded; staggered energy exact). Corrected `scripts/maxwell2d.py`'s header: its "2.0% drift" was an end-point sample. STATUS §3.34. Hygiene for P31: `_tensor_balanced` belongs in `wiring.py`.
- 2026-09-29 — **P32 (b) Lotka–Volterra as the book's step** (Claude Fable 5.1, ec5ece87, run by a subagent). `arrangements/lotka_volterra.py`, `tests/test_lotka_volterra.py`, 7 passed in 2 s; predictions in `notes/P32-B-PREDICTIONS-2026-09-29.md`, all four held: the composite's step is CDLM's field by Euler to 1e-12; the plain step raises the invariant every step in either chart (the caller's "log coordinates stay bounded" refuted); the two-stage `CotCoalg2` step bounds it 900×. STATUS §3.35.
- 2026-09-29 — **P32 (d) the genetic code, sourced and as an arrangement** (Claude Fable 5.1, ec5ece87, run by a subagent). Inputs checked against arXiv:0909.1442v2 Table I and arXiv:1309.4589v1 §2 (`notes/P32-D-SOURCES-2026-09-29.md`): every from-memory value matched; `scripts/genetic_code_stability_sourced.py` reproduces the table; on the 2008 scale the threshold and the "none of 2000" weaken. `arrangements/genetic_code.py`, `tests/test_genetic_code.py`, 9 passed in 72 s, six predictions held; the 386-box step equals the script's to 7e-16. STATUS §3.36. A concurrent session's dump (058d16b) already committed the sourcing files mid-run; nothing lost.
- **2026-10-01, an arrangement that runs arrangements** (Claude Fable 5.1, dd666e08; book plan P34, §5.9 item
  56). `arrangements/runner.py`, `tests/test_runner.py` (21 passed, about 80 s in five pieces), STATUS §3.37.
  A kit (const, register, drum, add, mul, cost, push, fan, inbox); a runner that is one closed term over it;
  a translator from any closed term to the runner's flag drums. The runner runs three terms exactly, runs
  them again when handed the translation of its own term, and, given ports, is sent a description over wires
  by another part and returns the answer. The translation is done outside the runner; no written proof.
- **2026-10-01, science test (h): bettors paid by being right** (Claude (subagent of Claude Opus 5.5,
  0fe746c1)). `arrangements/science_payment.py`, `tests/test_science_payment.py` (11 passed, 8 strict xfail,
  6 s); predictions and numbers fixed before code, results in `notes/SCIENCE-PAYMENT-PREDICTIONS-2026-10-01.md`
  §6. World plus ten bettor boxes wired by `science_native.wire_boxes`, run by `Phi`; the zero-supply floor is
  a `max` in the supply forcing (a kink, disclosed). (h1) failed its 1e-12 check against a float64 reference
  (1.14e-12) and holds to 9e-14 against exact rationals (post hoc); (h2) held, 136 of 136 below the growth
  threshold dead; (h3) half held, dogmatic 150 against updating 76 of 200 dead, mechanism rule 199 of 200;
  (h4) evidence held (every locked-wrong community all dead, 18/18, 18/18, 17/17), agree refuted (3-4 of 20
  collapse; no talk already collapses 3 of 20 from the shared races); (h5) consensus weights held (hub 1/4),
  hub wrong locks and collapses more than the ring under evidence (20 vs 17, 20 vs 18).
- **2026-10-01, science test (h), §7 correction: the dead fall apart** (Claude (subagent of Claude Opus 5.5,
  0fe746c1)). `science_payment.gate`, `bettor_gated`, `market_gated`; tests appended (whole file 30 passed,
  15 strict xfail, 21 s); numbers fixed before code in the payment note §7, results in §8. A bettor's account
  is gated by a smooth `g(s)` (`s_1 = c/2`, so no death step changes, checked 200 of 200 in each arm), its
  belief dissolves at `r = 0.1`, listeners average over the living (a jump when the last dies, disclosed). Dead
  bettors are not heard (Jacobian exactly 0); not uniform within 50 steps (refuted, as predicted). Evidence
  collapses unchanged (17, 18, 17); agree at 10 refuted (12, 18, 17 vs 20). Hub wrong: hub dies after the lock
  in 20 of 20; end-state lock falls to 0 only because the dead dissolve; among the living 20 of 20 lock and die.
- **2026-10-01, science test (i): what is built on an idea pays for its excuses** (Claude (subagent of Claude
  Opus 5.5, 0fe746c1)). `arrangements/science_incumbent.py`, `tests/test_science_incumbent.py` (9 passed, 2
  strict xfail, 10 s, one recorded run); numbers and closed forms fixed before code, results in
  `notes/SCIENCE-INCUMBENT-PREDICTIONS-2026-10-01.md` §4. World, (g4)'s hard incumbent-only reader (reduced
  exactly to the 8 excusable draws), an excuse fund and ten dependents, wired by `science_native.wire_boxes`
  and run by `Phi`; death and insolvency through test (h)'s smooth gate, no floor; the fund pays half of each
  held excuse's drawing cost per count (stipulated). Native matched the closed form in 39 of 40 seed-arms.
  (i1) held (10 of 10 equal (g4)); (i2) refuted as predicted (`n_B*` higher at `D = 10` in 2 of 10, flat at
  `D = 2, 5`: the library bounds the incumbent, dependents only cheapen its excuses); (i5) held as bookkeeping
  to 1.9e-13, literal equality 0 of 10 at `D = 2`; (i4) refuted (dependents die at count 15, overtake 4-22
  counts later, (g4)'s own count); (i3) not run, a deduction recorded instead.
- **2026-10-01, science test (j): paying a fixer to search for new excuses** (Claude (subagent of Claude Opus
  5.5, 0fe746c1)). Appended to `arrangements/science_incumbent.py` and `tests/test_science_incumbent.py` (whole
  file 21 passed, 2 strict xfail, 91 s, one recorded run); numbers and closed forms fixed before code in the
  incumbent note §5, results in §6. A seeded candidate sequence (checks survived geometric, `q = 1/2`; distance
  `U(0.5, 4)`), admissions emitted by the world as a mask, planting by (g4)'s masking, 3 cheapest figures per
  draw (disclosed). Native matched the closed form 70 of 70. (j1) held; (j2) held (medians 28 to 37 over
  `F = 0` to 4096); (j3) held, slope 0.746 counts per doubling of `F` (closed form 0.75), `R^2` 0.98 in `log F`
  against 0.66 in `F`; (j4) held for the untied distance law (Spearman positive 10 of 10); (j5) closed form
  only, computed after, refuted (longer than (i) in 2, 4, 5 of 10 at `D = 2, 5, 10`).
- **2026-10-01, science test (k): publishing failed replications** (Claude (subagent of Claude Opus 5.5,
  0fe746c1)). Spec and results: `notes/SCIENCE-FAILURE-REPORTS-PREDICTIONS-2026-10-01.md` (numbers fixed before
  code under §2; §3). Code `arrangements/science_failure.py` (imports `science_native.py`, unchanged); tests
  `tests/test_science_failure.py`, 4 passed, 1 skipped, 3 strict xfail, 53 s. Stage 2's seed-arm holder with two
  more state blocks: `z`, the net weight its own data took off each hypothesis, and a planted-mass register; the
  failure report `exp(-0.5/z)` goes forward beside the conclusion. Run 1 was NaN on the ring (`0 * inf` in the
  report's derivative, evaluated by `compose_seq`'s pullback); fixed by cutting the report below `1e-304`,
  disclosed. Silent arm bit-identical to stage 2. (k1) held (lower-the-seed: 0 wrong, reach 20 and 20, slope 9.90,
  planted-on-rejected lower in 19 of 20 by about 2%); (k2) refuted (never-reaches counts ring 1/0/0/0, all-to-all
  0/0/0/0 for silent and `beta` 0.1, 1, 10; reports against wrong neighbours outweigh those against the truth);
  (k3) untested, post hoc 3 of 3 in the late-loss seeds; (k4) refuted. Post hoc: at `beta = 10` the truth is lost
  after arrival in 1 (ring) and 2 (all-to-all) of 20 seeds; and a report is a fossil (a holder at 0.99 on the truth
  still reports it failed), since net weight lost cannot be regained near certainty.
- **2026-10-01, science test (k), §4 correction: a report says what the holder now finds** (Claude (subagent of
  Claude Opus 5.5, 0fe746c1)). Results in the failure-reports note §5 (numbers fixed before code at the end of §4).
  The report now reads the current shortfall `b - p`, `b` the seed's own planted mass (forcing as the seed's);
  `exp(-0.5/x)` with a custom-JVP derivative, exactly smooth, no cut, no NaN. Silent bit-identical to §3. The fossil
  reports are gone (0 in every arm). (k5) held on the ring, refuted all-to-all (planted-on-rejected lower 17 of 20
  against §3's 19); (k6) refuted (truth lost after reach at `beta = 10` in 1 ring and 2 all-to-all, each a whole
  community collapsing onto 4.5 or 5.5: a lagging shortfall reads "not caught up yet" as failure and the reports
  feed on each other); (k7) held. File: 9 passed, 1 skipped, 6 strict xfail, 129 s.
- **2026-10-03, Bayes' rule for every belief over finitely many values** (Claude Fable 5.1, 4f679514; book plan
  P36 stage (a)). New `arrangements/belief.py` (the belief box on all `n` log-heights with the response
  `diag(1/q)`, a likelihood box with a cost linear in the belief, the gauge-fixed inverse in closed form) and
  `tests/test_bayes_every_belief.py` (39 passed, 4 strict xfail, 45 s, run piece by piece and then whole).
  Derivation and nine predictions fixed before code in `notes/BAYES-EVERY-BELIEF.md`, a tenth added before its
  piece; verdicts there. Held: exactness for every belief and likelihood (`2.6e-15` at `n = 50`), the step's
  constant, three ways to hold the constant, order and batch, the Euclidean control, tempering, the self-forced
  shape, a zero likelihood outside. Refuted: that a thin height by itself loses the dense inverse. Found instead:
  the gauge-fixed step loses about `1e-16` over its **reference value's** height, which is the cause of the
  science-scales (a1) loss; corrected in `notes/SCIENCE-SCALES-PREDICTIONS-2026-10-01.md` §5.1,
  `notes/BAYES-INTEGRATOR.md`, `STATUS.md` §3.24 (new §3.38), `tests/test_science_scales.py`'s docstring and
  xfail reason, `arrangements/science.py`'s docstring. No existing code changed.
