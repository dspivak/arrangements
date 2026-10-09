# Same form for a learner: predictions (book P41 (b); Claude Opus 5.5, 5f5f4248, 2026-10-09)

Written before `tests/test_same_form_learner.py`. The question (David, 2026-10-09): two sources hold the same form
for a reader when the reader, holding the one, "doesn't have to move" when the other arrives, judged inside one
arrangement. Here the reader is a line-learner, `learning.parameterized_map` of F(q, x) = q0 x + q1 stepped by
`Phi` (configuration coalgebra, Euclidean sharp eta), fed one datum per step, cycling, by the environment through
its input port. The environment's schedule: dataset D1 for many passes, then dataset Dk. Nothing is substituted in
the arrangement; only what the port receives changes, and the schedule is the environment's.

Datasets. D1: 12 points, x in [-2, 2], y = 0.7 + 1.3x + noise; w1 its least-squares line.
D2: 9 points, x in [-1.5, 2.5], larger scatter, built so its least-squares line is exactly w1.
D3: a parabola, x symmetric, y = w1(x) + s (x^2 - mean x^2), whose least-squares line is exactly w1.
D4: 12 points about a different line, -0.5 + 0.4x.

1. Full pass at w1 (theorem: the summed forcing of squared error over a dataset, pulled back to q, is minus
   the gradient of its total loss, zero exactly at its least-squares line). The summed xi_Q over D2 and over
   D3 at q = w1 is 0 to 1e-9; over D4 it is not (order 1).
2. Held, then switched (eta = 0.05, then 0.01): one datum per step holds a small cycle about a centre near w1.
   After the switch to D2 or D3, the centre (mean over the last pass) moves by an amount that shrinks with eta
   (at most 0.1 at eta = 0.05, and smaller at eta = 0.01). After the switch to D4 it moves by about |w4 - w1|,
   whatever eta.
3. What the learner registers, inside its own run: its net displacement over the first pass after the switch.
   Small for D2 and D3 (shrinking with eta), of order eta * n * |w4 - w1| for D4.
4. Expected weirdness, reasonable: the parabola is the same form as the straight data for a line-learner,
   which can read only lines (as a phase reader cannot hear timbre). A form is the reader's.

## Verdicts (run 2026-10-09, `tests/test_same_form_learner.py`, 4 passed: 0.7 s, 113 s, 214 s; log `notes/same-form-learner-run.log`)

1. Held. Summed pulled-back forcing at w1: D2 (-4e-16, -2e-15), D3 (-9e-16, 0); D4 (-17.9, -15.1).
2. Held. Centre shift after the switch, eta 0.05 then 0.01: D2 0.058 then 0.0096; D3 0.035 then 0.0089; D4 1.58 both
   times (|w4 - w1| = 1.579). The small shifts are the per-datum step's own bias about the least-squares line, which
   depends on the data and their order and vanishes with eta.
3. Held. First-pass motion, eta 0.05 then 0.01: D2 0.061 then 0.0024; D3 0.16 then 0.0081; D4 0.85 then 0.22.
4. As expected. **Weird**: D3's first pass moves more than D2's (0.16 against 0.061 at eta 0.05): the parabola's
   residuals come in order (large at both ends, negative in the middle), so a pass swings the learner out and back.
   Its net pull over the pass is zero; its swing is not. So "does not have to move" holds for a pass, not for a datum.
