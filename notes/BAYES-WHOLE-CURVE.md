# Bayes' posterior on grids (`tests/test_bayes_whole_curve.py`)

P36 stage (b) of the book's `AGENTDOCPLAN.md` §8.2 (David, 2026-10-03: Bayes' rule for a belief that is not
confined to a family with finitely many natural parameters). Claude Fable 5.1, 9948b53c, 2026-10-03. The proof by
hand, in the book's letters, is Part II of the book repo's `notes/bayes-every-belief.md`; this note holds the
library's construction, the predictions, and the verdicts. Stage (a) is `notes/BAYES-EVERY-BELIEF.md`.

Current proof scope (Codex, 2026-10-08, P36 (d)): the posterior identity holds under its two integrability
conditions; the differential derivation holds on the narrower weighted Banach domain, not at every
finite-variance belief. The historical derivation and predictions in §§1–2 below are retained as written
before the experiment. The book's `notes/bayes-proof-review-20261008.md` gives the independent review,
counterexample, corrected covector statements and local-domain composition theorem.

Everything in §§1–2 was written **before any construction code existed** (the closed forms of §2 are arithmetic on
a mixture of bell curves, not the construction). Verdicts are §3, written piece by piece as each lands.

## 1. The derivation, fixed before the run

`z` real. The belief box's state is the log-height curve `l`, the belief is `q = exp(l) / integral exp(l)`. The
likelihood box's cost is `k(q, x) = integral q nll(x)`, linear in the belief, so it pushes the curve
`xi = -nll(x) = log Pr(x | .)` back whatever the belief. The pullback along the readout is the curve
`q (xi - E_q xi)`; the response divides by the height; one step at duration `tau` is

    l' = l + tau (xi - E_q xi),        belief(l') = q Pr(x | .)^tau / integral q Pr(x | .)^tau,

Bayes' rule at `tau = 1`, provided `E_q |xi|` is finite (H1) and `integral q Pr(x | .)` is finite (H2).

A machine holds a grid, so the run is the stage (a) box on `n` grid values, and what is tested is how that box
stands to the curve as the grid grows. Two things are fixed here before the run:

* **Restriction is exact.** The step is "add the force at each `z`, then a constant," so the grid box started at
  `l(z_i)` lands on `l'(z_i)` up to a constant, at every `n`. Only the constant and the averages converge.
* **The box that divides cannot hold a curve in float64; the box whose configuration is the belief can.** With
  `Q` the log-heights (`belief.belief_box`, stage (a)) the response is `diag(1 / q)`; on a grid wide enough that a
  height underflows this is `0 * inf`. With `Q` the beliefs and `S` the log-heights (`def.duration_integrator`
  allows `S != Q`), the coalgebra presents the belief and its update is the response after the responsibility,
  `l |-> l + tau (xi - <q, xi>)`, with no division. To be built as `belief.belief_coalgebra` and
  `belief.updating_on_beliefs`. Mathematically the two boxes take the same step.

The three beliefs, each two bumps at `-1.5` and `2.5` with half the mass on each:

* **G**: bell curves, standard deviations `0.6` and `0.8`;
* **L**: Laplace bumps `exp(-|z - mu| / b) / 2b`, `b = 0.5` (finite variance, tails fatter than any bell curve);
* **C**: Cauchy bumps `gamma / (pi (gamma^2 + (z - mu)^2))`, `gamma = 0.5` (no variance).

The likelihood is bell-curve noise of variance 1, `Pr(x | z) = N(x; z, 1)`, and the reading is `x = 1`, unless a
piece says otherwise. Grids are `n` equally spaced values on `[-a, a]`, spacing `h = 2a / (n - 1)`.

Closed forms for G (arithmetic by hand, to be recomputed in the test): `E z = 0.5`, `E z^2 = 4.75`, so
`E_q xi = -2.375 - log(2 pi)/2 = -3.2939`; `Pr(x) = (N(1; -1.5, 1.36) + N(1; 2.5, 1.64))/2 = 0.09563`,
`log Pr(x) = -2.3473`; `KL(q || q') = log Pr(x) - E_q xi = 0.9466`; posterior weights `0.1797, 0.8203`, component
means `-0.8382, 1.9146`, variances `0.2647, 0.3902`; posterior mean `1.4199`.

## 2. Predictions, each able to fail

1. **Restriction is exact at every grid size.** Belief G, window `a = 8`, `n = 5, 9, 17, 33, 65, 129, 257, 513,
   1025`. One step of stage (a)'s box `Phi(likelihood . belief)` from `l_i = log q(z_i)`: the differences
   `l'_i - log q'(z_i)`, with `q'` the closed-form posterior density, are the same at every `i`, spread (max minus
   min) below `1e-10`, at every `n`, the five-point grid included. The belief after equals `q'(z_i) / sum_j q'(z_j)`
   to relative error `1e-10`.
2. **The grid's numbers converge to the curve's, faster than any power.** Belief G, window `a = 12`,
   `h = 2, 1, 0.5, 0.25, 0.125, 0.0625` (`n = 13, 25, 49, 97, 193, 385`). The constant the step subtracts
   `<q, xi>`, the posterior mean and variance, and `log(h sum_i exp(l'_i))` (which should tend to `KL(q || q')`)
   differ from the closed forms by more than `1e-4` at `h = 1`, less than `1e-6` at `h = 0.5`, and less than `1e-11`
   from `h = 0.25` on (trapezoid sums of bell curves: error about `exp(-2 pi^2 s^2 / h^2)`, `s = 0.51` the
   narrowest posterior bump).
3. **The response alone has no limit; the step does.** Window `a = 8`, `n = 65, 129, 257, 513, 1025`. The smallest
   and the largest entries of the response `diag(1 / q)` both grow like `1 / h`: the smallest entry times `h`
   tends to `1 / max_z q(z)` (within 1% at `n = 1025` of its value at `n = 513`), and the largest exceeds `1e25`
   at every `n`. The largest entry of the responsibility `q (xi - <q, xi>)` shrinks like `h` (halves, within 5%,
   each time `n - 1` doubles). The step `l' - l` equals `xi - <q, xi>` to `1e-10` and its largest entry is the
   same at every `n` to 1%.
4. **A thin tail kills the box that divides; the box on beliefs holds.** Belief G. On the window `a = 24`
   (`h = 0.125`) every height is above float64's floor and stage (a)'s box is exact (spread below `1e-9`). On
   `a = 40` heights underflow (where `log q < -708`, beyond `z = 32.6` and `z = -27.6`) and stage (a)'s box
   returns a non-finite state. `updating_on_beliefs` on the same `a = 40` grid gives finite log-heights with
   spread against `log q'(z_i)` below `1e-9`. And from the single bell curve `N(0, 1)` with reading `2` on that
   grid it lands on `-(z_i - 1)^2` up to a constant, to `1e-9`: the book's step, `prop.bayes`'s plane.
5. **Three tails, one noise, the window widening.** Beliefs G, L, C with bell-curve noise, spacing `h = 0.05`,
   windows `a = 10, 20, 40, 80, 160, 320`, the box on beliefs.
   * The belief converges in all three: the differences `l'_i - l'_j` over the grid points in `[-10, 10]` are the
     same at every window to `1e-9`, and the posterior mean and variance at `a >= 20` agree with those at `a = 10`
     to `1e-9` (the posterior's tails are bell-curve thin whatever the prior's).
   * The constant converges for G and L, to the closed form `-(E z^2 - 2x E z + x^2)/2 - log(2 pi)/2` (`E z^2 =
     4.75` for G and `4.25 + 2 b^2 = 4.75` for L; the same number by the choice of `b`), within `1e-6` at every
     window for G; for L within `1e-3` at `a >= 20` (Laplace bumps with their kinks on grid points, error of
     order `h^2`).
   * The constant diverges for C, linearly: `(c(2a) - c(a)) / a` tends to `-gamma / pi = -0.1592`, within 2% for
     `a = 80 -> 160` and `160 -> 320`. So in all log-heights the state after the step has no limit as the window
     grows, while the belief has one.
6. **H1 without H2: a force that flattens.** Belief C, force `xi = -log q` (a likelihood `1 / q(z)`, unbounded in
   `z`), same windows and spacing. The belief after is uniform on the window at every window (largest height over
   smallest within `1e-9` of 1), so its variance is `a^2 / 3` to 1% and it has no limit; the constant `<q, xi>`
   converges, to the entropy of C (computed by quadrature), within `0.05` at `a = 320` and closer at each
   doubling.
7. **Duration: one step leaves area `exp(KL)`, many small steps leave area 1.** Belief G, window `a = 12`,
   `h = 0.125`, from `l_i = log q(z_i)`. One step: `log(h sum exp(l'))` equals `KL(q || q') = 0.9466` to `1e-9`.
   `N` steps of duration `1/N`, `N = 1, 2, 4, ..., 1024`: the belief after is the same at every `N` (to `1e-10`
   relative); `log(h sum exp(l))` after the `N` steps is positive, falls by a factor between `1.8` and `2.2` at
   each doubling from `N = 16` on, and at `N = 1024` is within 10% of `(E_{q'} xi - E_q xi) / 2048`. Two half
   steps equal one whole step in belief.
8. **A stream on the wide window.** Belief G, `a = 40`, `h = 0.125`, 30 readings drawn about `2.0` with unit
   noise (`RandomState(7)`), the box on beliefs: the belief after equals the closed-form mixture posterior (each
   component updated conjugately, weights by each component's evidence) in mean and variance to `1e-9`, and the
   log-heights have spread against the closed form's log-density below `1e-8` on the grid points where the closed
   form is computable in logs; the same in reversed order to `1e-9`. Stage (a)'s box on that grid is non-finite
   at the first step (piece 4).

9. *(Added 2026-10-03 after the eight above had run, and before this piece was written or run; prompted by the
   book's `sec.bayes`, which now says of a reading reported only as "heavier" or "lighter" that "a box whose state
   is two numbers cannot hold" the lopsided new belief; that paragraph was cut from the book later the same day.)* **"Heavier", held on a curve.** Belief `N(0, 1)`; the
   scale's reading has bell-curve noise of variance 1 and only its sign is told, so `Pr(heavier | z) = Phi(z)`,
   the normal distribution function, and `xi = log Phi(z)`: bounded above, quadratic below, average finite.
   Window `a = 12`, `h = 0.05`, the box on beliefs. Predicted: the log-heights after one step equal
   `-z^2/2 + log Phi(z)` up to a constant (spread below `1e-10`), the belief `2 phi(z) Phi(z)` (a skew bell
   curve), with mean `1 / sqrt(pi) = 0.5641895835` and variance `1 - 1/pi = 0.6816901138` to `1e-9`; the constant
   `<q, xi> = E[log Phi(z)]` agrees with quadrature to `1e-9`; and the new log-height is not a parabola: its best
   quadratic fit over `[-3, 3]` misses by more than `0.01` somewhere, where the prior's misses by less than
   `1e-12`. A second "heavier" gives `phi Phi^2` up to scale, spread below `1e-10`.

If 1–5 and 7–8 hold, the grid box is the curve's box restricted, at every size; the curve's step is Bayes' rule
whenever the force has a finite average under the belief; and a machine holds it only in the form whose
configuration is the belief. 6 and the Cauchy arm of 5 are where it stops.

## 3. Verdicts

Each piece run by `PYTEST_ADDOPTS=-s scripts/run_tests.sh 110 <node id>`, full output teed to
`notes/BAYES-WHOLE-CURVE.log`; written here as each landed, 2026-10-03.

**1. Held.** Belief G, window 8, one step of stage (a)'s box: the spread of `l'_i - log q'(z_i)` is `2.1e-14`
(`n = 5`), `2.1e-14` (9), `1.4e-14` (17), `2.8e-14` (33), `3.6e-14` (65), `4.6e-14` (129), `4.3e-14` (257, 513,
1025), and the belief after equals the closed-form posterior restricted to the grid and rescaled to relative
error at most `3.2e-14`. So the five-point box is as exact as the 1025-point one: a grid loses the values between
its points and nothing at them.

**2. Held.** Belief G, window 12. Closed forms: `E_q xi = -3.2939385332`, `log Pr(x) = -2.3472939936`,
`KL(q || q') = 0.9466445396`, posterior mean `1.4198750887`, variance `1.4849021652`. Errors of the grid box's
(constant, mean, variance, log area): `h = 2`: `0.35, 0.37, 0.92, 0.49`; `h = 1`: `1.4e-3, 3.8e-3, 1.0e-2,
3.0e-3`; `h = 0.5`: `4.2e-12, 5.9e-10, 5.0e-9, 1.4e-10`; `h = 0.25, 0.125, 0.0625`: all at most `4.4e-16`. Faster
than any power, as trapezoid sums of bell curves are; the log area tends to the divergence from the old belief to
the new, so the step's constant is the average surprisal and not the surprisal of the observation.

**3. Held.** Window 8, `n = 65, 129, 257, 513, 1025`. `h` times the smallest entry of `diag(1 / q)` is `3.007946`
at every `n`, which is `1 / max_z q(z)`; so every entry of the response grows like `1 / h`, and its largest is
`3.7e26, 7.3e26, 1.5e27, 2.9e27, 5.9e27`. The largest entry of the responsibility is `0.1248, 0.0624, 0.0312,
0.0156, 0.0078`, halving exactly. The step equals `xi - <q, xi>` to `1.4e-14` and its largest entry is `38.125`
at every `n`. So on a curve neither the responsibility nor the response has a limit as a list of numbers: one is
a density (it scales like `h`), the other divides by a density (it scales like `1 / h`), and only their composite
is a function of `z`.

**4. Held.** Belief G, `h = 0.125`. Window 24: the smallest height is `1.7e-240` and stage (a)'s box has spread
`2.3e-13`. Window 40: 160 of the 641 heights are held as `0` (the first at `z = -27.625`, as computed), and one
step of stage (a)'s box leaves exactly those 160 log-heights non-finite. `updating_on_beliefs` on the same grid:
all finite, spread `4.6e-13` against the closed-form log-density; and from `N(0, 1)` with reading `2` it lands on
`-(z - 1)^2` up to a constant with spread `3.6e-15`, mean `1.0000000000`, variance `0.5`: the book's step, with
tails held down to `exp(-1600)`. Where both boxes are finite (window 8, `n = 257`) their steps differ by
`1.4e-14`. So a machine holds a curve's step only as the response after the responsibility.

**5. Held, every clause.** Spacing `0.05`, windows `10, 20, 40, 80, 160, 320` (`n` up to 12801), the box on
beliefs, bell-curve noise, reading `1`.
* *The belief converges in all three.* The log-height differences over `[-10, 10]` agree with the window-10 run
  to at most `1.6e-12` (G), `5.7e-13` (L), `5.1e-13` (C), and the posterior mean and variance are the same to
  twelve places at every window: G `1.419875088677, 1.484902165229`; L `1.518959888941, 1.572072978232`; C
  `1.383108798727, 1.471469811819`.
* *The constant converges for G and L.* G: within `1.8e-15` of `-3.2939385332` at every window. L: `2.1e-4` off
  at every window (the kinks' `h^2` error, about the `2e-4` estimated by hand), unchanged as the window grows.
* *The constant diverges for C.* `<q, xi> = -4.445, -6.097, -9.310, -15.691, -28.431, -53.899` at the six
  windows; `(c(2a) - c(a)) / a = -0.16521, -0.16064, -0.15953, -0.15925, -0.15918`, tending to
  `-gamma / pi = -0.15915`. So with a Cauchy belief and bell-curve noise the state in all log-heights runs off
  with the window while the belief it reads out does not move in the twelfth place.

**6. Held.** Belief C, force `xi = -log q`, same windows. (The first run of this piece stopped in the test's own
reference integral, `scipy.integrate.quad` refusing break points with infinite limits, before any step was
taken; the integral was split in three and the piece run again, nothing else changed.) The entropy of C by
quadrature is `2.31387010`. The belief after is uniform on the window at every window (largest height over
smallest within `1.8e-15` of 1), its variance over `a^2 / 3` is `1.0050, 1.0025, 1.0013, 1.0006, 1.0003, 1.0002`
(the grid's `(n + 1) / (n - 1)`), so it grows without bound and the belief has no limit. The constant is
`2.1071, 2.1915, 2.2424, 2.2728, 2.2906, 2.3009`, its gap to the entropy `0.207, 0.122, 0.072, 0.041, 0.023,
0.013`. The mirror image of piece 5's Cauchy arm: there the belief had a limit and the constant none; here the
constant has a limit and the belief none.

**7. Held.** Belief G, window 12, `h = 0.125`. One step from `l = log q`: `log(h sum exp(l')) = 0.946644539603`
and `KL(q || q') = 0.946644539603`. Two half steps give the same belief as one whole step to `2.9e-14`. `N` steps
of duration `1/N`: the belief after is the closed-form posterior to at most `2.6e-13` relative at every `N`; the
log area is `0.9466, 0.4304, 0.2041, 0.0993, 0.0490, 0.0243, 0.0121, 0.0060, 0.0030, 0.0015, 0.00075` at
`N = 1, 2, ..., 1024`, falling by `2.200, 2.108, 2.056, 2.028, 2.014, 2.007, 2.004, 2.002, 2.001, 2.000` at each
doubling; and `N` times it is `0.77237` at `N = 1024` against the predicted `(E_{q'} xi - E_q xi) / 2 = 0.77220`.
So splitting the duration moves the step's constant from the average surprisal toward the surprisal of the
observation, at rate `1 / N`, and the heights' area toward `1`.

**8. Held.** Belief G, window 40, `h = 0.125`, 30 readings about `2.0`. Closed form: weights `5.5e-7, 0.9999995`,
mean `1.955311129505`, variance `0.031683223414`. The box on beliefs: mean off by `2.7e-15`, variance by
`6.5e-15`, spread of the log-heights against the closed form's log-density `2.2e-11`, reversed order the same to
`1.5e-11`; the smallest log-height held is `-27771`.

**9. Held.** Belief `N(0, 1)`, the observation "heavier" (`Pr(heavier | z) = Phi(z)`), window 12, `h = 0.05`,
the box on beliefs. The log-heights after equal `-z^2/2 + log Phi(z)` up to a constant with spread `7.1e-15`;
mean `0.5641895835` (`1 / sqrt(pi)`), variance `0.6816901138` (`1 - 1/pi`), both to ten places; the constant
`<q, xi>` is `-1.0000000000`, as quadrature gives (exactly `-1`: under `N(0, 1)` the number `Phi(z)` is uniform on
`[0, 1]`, and the average of `log u` there is `-1`). So the average surprisal of "heavier" is `1` while its
surprisal is `log 2`, and their difference `1 - log 2 = 0.307` is the divergence from the old belief to the new.
The prior's log-height is a parabola on `[-3, 3]` to `2.7e-15`; the new one misses its best parabola by `0.378`.
A second "heavier" gives `phi Phi^2` up to scale, spread `5.7e-14`. This is the belief `sec.bayes` says "a box
whose state is two numbers cannot hold"; the curve's box holds it, and every later one.

Whole file, one run: `scripts/run_tests.sh 240 tests/test_bayes_whole_curve.py`, **24 passed, 32 s**, 2026-10-03,
before piece 9 was added; with it, 25 passed. All nine predictions held; none was refuted.

### What the run shows

The stage (a) box on a grid is the curve's box restricted, exactly and at every size: its belief after a step is
the curve's posterior at the grid's values. What a growing grid converges in is the step's constant and the
averages, and they converge as fast as the sums converge to integrals. The response alone and the responsibility
alone do not converge as lists of numbers (one scales like `1 / h`, the other like `h`); their composite does.
The box that divides fails in float64 as soon as the window holds a tail thin enough to underflow, which for a
curve is every window wide enough to hold the curve; the box whose configuration is the belief and whose update
is the composite is exact there, through a 30-reading stream with log-heights down to `-27771`.

Where it stops, as the hand proof says: with a Cauchy belief and bell-curve noise the force's average is
infinite, the constant runs off linearly in the window, and only the belief converges; with a force unbounded
above the area is infinite and the belief has no limit although the constant has one.

### Reasonable or weird, by scale (the agent's judgment, not evidence)

* **A population.** Reasonable, as in stage (a), now with a trait that varies continuously: each trait value's
  log-density moves by its own fitness less the population's average, whatever its density.
* **A cell or a neuron, any body.** Weird: no body holds a whole curve. What a body can hold is a finite list, a
  grid or a family, and piece 1 says what a grid costs it: nothing at the values it holds, everything between
  them. A body that divides by a height it holds as a number fails where float64 failed; one that adds the force
  to a stored log-height does not.
* **A person.** Weird in the Cauchy arm: a belief that gives real weight to very large values expects an infinite
  squared error from any reading, so the likelihood box's cost is infinite, yet the update itself is tame and
  lands on a belief with bell-curve tails. The infinite number is a constant, and a constant does not change a
  belief. Weird the other way in piece 6: a force can have a finite average and still flatten the belief into
  nothing.
* **The response.** The raw multiplier `1/q` is unbounded on a positive curve of finite area. Whether the
  response is bounded depends on its domain and norm: on the hand proof's chosen covectors `eta = q f`,
  with norm `||eta/q||`, division is an isometry. The finite-grid underflow measurements stand; they do not
  establish an unbounded response in every continuum model (Codex correction, 2026-10-08). A response
  with a floor was not built or run here.
* **Splitting the duration** (piece 7). Reasonable: taking the evidence in smaller pieces changes only the
  bookkeeping constant, toward the one Bayes' rule is usually written with; the belief is the same at every `N`.

### What this does not establish

The run is on grids. The separate smoothness proof for the exponential-slack Banach domain (book repo,
`notes/bayes-every-belief.md` II.4) was independently confirmed by Codex, 2026-10-08. A positive continuous
finite-variance belief can lie outside that domain and fail the proposed measure-space differentiation.
Nothing here makes the whole-curve box an integrator in Ch 3's total sense, or proves a cotangent functor
stable under arbitrary smooth pullback. Gaussian forces preserve the proved domain; general likelihoods
need evidence and domain conditions. The local-composition theorem in the review does not remove those
limits. No point-mass force, zero height, higher-dimensional likelihood or response with a floor was run.
