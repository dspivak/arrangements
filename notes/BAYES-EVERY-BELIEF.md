# Bayes' rule for every belief over finitely many values (`tests/test_bayes_every_belief.py`)

P36 stage (a) of the book's `AGENTDOCPLAN.md` §8.2 (David, 2026-10-03: Bayes' rule for a belief that is not
confined to a family with finitely many natural parameters). Claude Fable 5.1, 4f679514, 2026-10-03. The proof by
hand, in the book's letters, is the book repo's `notes/bayes-every-belief.md`; this note holds the library's
construction, the predictions, and the verdicts.

Everything in §§1–2 was written **before any construction code existed**. Verdicts are §3, written piece by piece
as each lands.

## 1. The derivation, fixed before the run

`n` values. The belief box's state is `l in R^n`, the belief's log-heights; the belief is `q = softmax(l)`, and
`l + c 1` is the same belief. Its readout is `q`; it has zero forcing. The likelihood box, downstream, has trivial
parameter and the cost `k(y, x) = <y, nll(x)>`, with `nll(x)_i = -log Pr(x | z_i)`; its forcing is the exact
`-dk`, so it pushes `xi = -nll(x) = log Pr(x | .)` back on the belief's wire, whatever the belief. The pullback
along the readout is `(diag(q) - q q^T) xi = q * (xi - <q, xi>)`. The **response** is

    sharp_l = tau * diag(1 / softmax(l)),        (`log_height_response`)

diagonal and positive definite on all of `R^n`, so one step at duration 1 is

    l' = l + xi - <q, xi> 1,     softmax(l') = q * Pr(x | .) / <q, Pr(x | .)>,

Bayes' rule exactly, for every `l` and every positive likelihood. `diag(q) - q q^T` is singular along `1`, so
`sharp` is not its inverse but a generalized inverse (`G sharp G = G`); in the gauge `l_n = 0` of
`choice.belief_arrangement` the block `G` is invertible with the closed form
`G^{-1} = diag(1/q_i) + (1/q_n) 1 1^T`, which is `sharp` followed by subtracting the last coordinate.

## 2. Predictions, each able to fail

1. **Exactness, every belief and every likelihood.** For `n = 2, 3, 7, 50`, random beliefs with every height
   positive and random positive likelihood columns, one step of `Phi(likelihood . belief)` gives
   `softmax(l')` equal to the posterior computed directly, `q p / <q, p>`, to relative error below `1e-12`. The
   hand example: belief `(1/2, 1/2)`, likelihoods `(3/4, 1/4)`, posterior `(3/4, 1/4)`, and
   `l' = (log(3)/2, -log(3)/2)`.
2. **The constant is the belief's average of the force.** On `R^n` the step is `l + xi - <q, xi> 1` to `1e-12`, so
   `<q, l' - l> = 0`; it is *not* the normalizing constant `log <q, p>`, and `sum exp(l')` exceeds `1` after a step
   from normalized log-heights by the factor `exp(log <q,p> - <q, log p>) >= 1`.
3. **Three ways to hold the constant, one belief.** The `R^n` box with the diagonal response, the gauge-fixed box
   with the closed-form inverse, and the gauge-fixed box with the dense `inv(G)` (`choice.inverse_fisher`) land on
   the same belief to `1e-12` from a belief with every height above `1e-3`. For `n = 2` the gauge state is the
   log-odds and the step adds the log-likelihood ratio.
4. **A stream of observations; order does not matter.** With `z` on a grid, bell-curve noise of variance 1 and a
   two-bump prior (in no bell-curve family), 30 real observations stepped one at a time give the directly computed
   posterior to `1e-10`, the same in reversed and shuffled order, and the same as one step with the summed cost.
5. **The response is what does the work.** With the Euclidean response `eta I` in the same arrangement: from the
   uniform belief `eta = n` is Bayes exactly (there `diag(1/q) = n I`), and from a non-uniform belief no
   `eta in {0.1, 0.5, 1, n, 2n}` gives the posterior (belief error above `1e-3`); and with `eta = n` a second
   step from the uniform belief misses.
6. **Thin beliefs.** In the mathematics the step is exact at every positive height. In float64, the diagonal
   response stays exact (step error against `xi - <q, xi>` below `1e-9`) with a height as small as `1e-300`,
   where the dense `inv(G)` has lost it by `1e-8` (the science-scales run found loss below about `1e-5`,
   `SCIENCE-SCALES-PREDICTIONS-2026-10-01.md` §5.1). Over a 400-draw stream that drives heights below `1e-100`
   the diagonal box tracks the directly accumulated log-posterior differences to `1e-8`. It fails only when a
   height underflows to exactly `0` in float64 (a log-height more than about 745 below the largest): then
   `0 / 0`, a non-finite state.
7. **A zero likelihood is outside.** `Pr(x | z_i) = 0` gives a non-finite state after one step; `1e-300` in its
   place gives the direct posterior to `1e-12`.
8. **Duration is tempering.** A step at `tau` gives `q p^tau / <q, p^tau>`; two half steps equal one whole step.
9. **The forcing may sit in the belief box.** The belief box carrying the cost as its own forcing (item 27's
   shape, no likelihood box) moves the state identically, to `1e-12`.

10. *(Added 2026-10-03 after prediction 6's dense-inverse clause was refuted and a scratch probe pointed at the
   gauge's reference value; written before this piece was run.)* **The science-scales stream.** On that run's own
   misspecified stream (200 draws from `default_rng(0)` about `2.3`, hypotheses at `mu = 0, ..., 4`, winner
   `mu = 2`), the `R^n` diagonal box tracks the directly accumulated log-posterior differences to `1e-9` at every
   draw; so does the gauge box with the winner as its reference value, with either inverse; and the gauge box with
   the reference at `mu = 4`, as `science.mind_step` has it, is off by more than `1e-6` by draw 30.

If 1–9 hold, one step of the arrangement is Bayes' rule for every belief with positive heights over finitely many
values and every positive likelihood, with no family chosen, and the gauge is a convenience.

## 3. Verdicts

Each piece run by `PYTEST_ADDOPTS=-s scripts/run_tests.sh 120 <node id>`, full output teed to
`notes/BAYES-EVERY-BELIEF.log`; written here as each landed, 2026-10-03.

**1. Held.** The hand example gives `l' = (0.54930614, -0.54930614) = (log(3)/2, -log(3)/2)` and the belief
`(0.75, 0.25)`. Over 24 random (belief, likelihood) pairs at each size, the worst relative error of `softmax(l')`
against `q p / <q, p>` is `6.5e-16` (`n = 2`), `1.7e-15` (`n = 3`), `1.3e-15` (`n = 7`), `2.6e-15` (`n = 50`).
*Not predicted, two checks added while writing the test:* the book's worked step (`N(0,1)`, reading `x = 2`) on a
161-point grid lands on the restriction of `N(1, 1/2)` to the grid, max abs error `2.8e-17`, mean `1.000000000000`,
variance `0.500000000000`, with no family told to the box; and the covector sent back to the observation's slot is
`E_q[z] - x` (`-3.201581164814` both ways), the book's `y_1 - x`.

**2. Held.** At `n = 2, 5, 20` the step equals `xi - <q, xi> 1` to `0`, `8.9e-16`, `6.7e-16`; `<q, l' - l>` is
`5.3e-17`, `3.0e-16`, `-1.5e-16`; and from normalized log-heights `sum exp(l')` is `1.819450`, `8.592771`,
`7.518164`, each equal to `exp(log <q,p> - <q, log p>)` to `1e-12`. So the step's constant is the belief's average
of the force, not the normalizing constant; the readout does the normalizing.

**3. Held.** The `R^n` box with the diagonal response, the gauge box with the closed-form inverse and the gauge
box with the dense `inv(G)` agree with the direct posterior to worst relative error `2.0e-14` (`n = 2`), `1.4e-15`
(`n = 3`), `9.4e-15` (`n = 7`) over five beliefs each with every height above `1e-3`. For two values the log-odds
go `0 -> 1.098612288668`, `-1.3 -> -2.804077396776`, `2.1 -> -0.384906649788`, each the prior log-odds plus the
log-likelihood ratio to `1e-12`.

**4. Held.** 41-point grid on `[-4, 6]`, a two-bump prior (bumps at `-1.5` and `2.5`, half the mass on each),
30 readings drawn about `2.0` with unit noise: max abs belief error against the direct posterior `1.3e-15`
(in order), `2.1e-15` (reversed), `1.1e-15` (shuffled), `3.1e-15` (one step with the summed cost). The left bump's
mass goes from `0.5001` to `1.3e-24`, and the smallest height after is `1.8e-246`, so the stream already runs
through very thin heights.

**5. Held.** Euclidean response `eta I`, three values, likelihood `(0.2, 0.5, 0.9)`: from the uniform belief
`eta = 3` gives the posterior with error `0`; from the belief `(0.6, 0.3, 0.1)` the belief errors at
`eta = 0.1, 0.5, 1, 3, 6` are `0.257, 0.219, 0.170, 0.111, 0.277`; and a second step from the uniform belief at
`eta = 3` misses by `0.044`.

**6, first three pieces. Held for the diagonal response; the dense-inverse clause REFUTED.** Three values, the
first with height `h`, likelihood `(0.9, 0.2, 0.5)`. Diagonal response, step error against `xi - <q, xi>`:
`6.7e-16, 3.3e-16, 2.2e-16, 8.7e-15, 8.7e-15` at `h = 1e-5, 1e-8, 1e-12, 1e-100, 1e-300`. The gauge box whose
reference value is thick is just as exact, dense `inv(G)` and closed form alike: `4.4e-16, 5.6e-16, 4.4e-16,
6.7e-15, 5.0e-14`. So a thin height does not by itself lose the dense inverse, and the prediction's "lost by
`1e-8`" is false (kept as four strict xfails). **What loses the gauge step is a thin reference value** (not
predicted; `test_the_gauge_needs_a_thick_reference_value`): with the reference value at height `h`, the gauge
step error is `1.2e-11, 1.7e-8, 1.4e-4, 2.4e+83, 3.7e+283` (closed form) and `8.6e-12, 2.0e-8, 1.4e-4, 14.5,
0.50` (dense) at the same five heights, about `1e-16 / h`, while the `R^n` diagonal box on the same beliefs stays
at `8.9e-16, 3.3e-16, 2.2e-16, 8.7e-15, 8.7e-15`. The cause: in the gauge the response adds
`(sum_{i<n} eta_i) / q_n` to every coordinate, and that sum, which is `-eta_n`, is computed by cancellation among
numbers of size about `1`, so its round-off, about `1e-16`, is divided by the reference's height. A scratch probe
on five values (in the log) agrees: heights down to `1e-200` with a thick reference, error `5e-15`, the library's
own `choice.belief_arrangement` included; reference at `1e-5`, `2e-11`; reference at `1e-13`, `2e-3`.

**6, the 400-draw stream. Held.** Three values at `-0.25, 0.5, 1`, 400 readings about `1`: the smallest height
reaches `2.8e-145` and the diagonal box's log-height differences are within `1.1e-13` of the directly accumulated
ones. The largest `|l|` is `330.6`, the thinnest value's own log-height; the constant does not run away
(`sum exp(l) = 9.21` after 400 draws), since the step leaves the belief's average of `l` where it was.

**10. Held.** On the science-scales run's misspecified stream (200 draws, smallest height at the end `3.3e-229`):
the `R^n` diagonal box's log-height differences stay within `1.7e-13` of the directly accumulated ones at every
draw; the gauge box with the winner `mu = 2` as its reference stays within `4.6e-13`, closed form and dense
`inv(G)` alike; the gauge box with the reference at `mu = 4` first exceeds `1e-12` at draw 10 (both inverses), is
off by `437` (closed form) and `6.65` (dense) at draw 30, and the dense one is non-finite from draw 74. **So that
run's loss of the Bayes step was the gauge's reference value going thin, not the response, not the dense inverse,
and not `cond(G)`**: its own one-step errors (`1.6e-13`, `3e-11` at its steps 5 and 10) are about `1e-16` over the
reference's height there, not `1e-16 cond(G)` (`4e-11`, `5e-5`).

**6, underflow. Held, with the threshold corrected.** Continuing the same stream, the first non-finite state is at
draw 909 and the smallest height just before it is exactly `0.0`. The threshold is not 745 below the largest
log-height but about 708: jax flushes subnormal numbers to zero, so a height below about `2.2e-308` is held as `0`
(`1e-307` is still exact, step error `8.7e-15`; `1e-310` is `0` and the step is `nan`). So the diagonal box is
exact for every height float64 can hold, and a height it cannot hold does not fade out: it poisons the state.

**7. Held.** A likelihood of `0` at one value gives the state `[nan nan nan]`; `1e-300` in its place gives the
direct posterior to relative error `2.4e-14`.

**8. Held.** At `tau = 0, 0.25, 0.5, 1, 2` the belief after is `q p^tau / <q, p^tau>` to `2.8e-16, 2.1e-16,
3.4e-16, 3.3e-16, 2.5e-16`; two half steps equal one whole step to `3.3e-16`.

**9. Held.** The belief box carrying the cost as its own forcing moves the state identically to the composite
with a likelihood box, max abs difference `0` (`n = 2, 5, 12`).

Whole file, one run: `scripts/run_tests.sh 240 tests/test_bayes_every_belief.py`, **39 passed, 4 xfailed (strict;
prediction 6's refuted clause), 45 s**, 2026-10-03.

### What the run shows

One step of the arrangement is Bayes' rule for every belief with positive heights over finitely many values and
every positive likelihood, with no family chosen: the state is the list of log-heights, the readout is the belief,
the likelihood box's cost is linear in the belief, and the response divides by the belief's height. The gauge is a
convenience, and a costly one in float64: its step loses about `1e-16` over the reference value's height, while the
box on all `n` log-heights is exact down to the smallest height float64 holds.

### Reasonable or weird, by scale (the agent's judgment, not evidence)

* **A population.** Reasonable, and the best fit. The response `1/q` says a value's log-height moves by its own
  log-likelihood however little of the belief sits there, which is per-capita growth not depending on abundance:
  the rare type's log-abundance moves as fast as the common one's. The height plays the role of a mass, and the
  thin values are the light ones.
* **A cell or a neuron.** Reasonable: an accumulator of log-likelihoods per alternative, read out through a
  normalizing nonlinearity, is the standard account of evidence accumulation. The readout normalizes; the stored
  numbers need not.
* **A person.** Weird in two ways. Nothing is ever ruled out and nothing ruled out could come back: a height of
  zero is not a state, and no step reaches it. A person does rule things out, and reconsiders. And, as for
  `BAYES-INTEGRATOR.md`'s finding (c), the force is never zero unless the likelihood is flat, so the box never
  settles.
* **Any body.** Weird: the response is unbounded as the belief thins, exactly cancelled by a responsibility that
  vanishes. In the mathematics the product is `1` at every positive height. In matter, nothing responds without
  bound; float64's version of that is the `nan` at draw 909, where a hypothesis too thin to hold kills the whole
  box instead of dropping out. A box made of matter would have a floor, and below it would not be doing Bayes'
  rule.
* **The gauge.** Weird and informative: choosing a reference value changes nothing in the mathematics and decides
  whether a long run survives in float64, as `choice.py`'s shift of every value by a constant did. A
  model-irrelevant choice of chart is again a part of what is run.

### What this does not establish

A belief over `n` values is in a family with `n - 1` natural parameters, so this run says nothing about the
question P36 exists for, a belief that is a whole curve: which curves are a state space, and whether "divide by
the height" is a response there, is stage (b). What the finite case hands it: the step never uses the response
alone, only the response after the responsibility, which is `xi |-> xi - <q, xi>` and needs only the belief's
average of the force to be finite. Nothing here covers a belief or a likelihood with a zero. Nothing is new about
learning beyond `choice.py`'s categorical step, which had the same mathematics from the uniform belief in the
gauge; what is new is every belief and every likelihood, the box without a gauge, the thin-height numbers, and
the corrected cause of the science-scales loss.
