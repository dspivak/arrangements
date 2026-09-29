
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
