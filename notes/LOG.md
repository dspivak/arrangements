
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
