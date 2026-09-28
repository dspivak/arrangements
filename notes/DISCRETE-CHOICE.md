# Discrete choice as an integrator (`tests/test_discrete_choice.py`)

P25 stage (b) of the book's `AGENTDOCPLAN.md` §8.2, from David's question in the §1.6 run
(2026-09-27): does discrete choice come into the framework "through softmax /
stochasticity"? The claim under test is Chapter 2's — learning and mechanics are
arrangements of one kind — which stage (a) widened from differentiable learning to
exponential-family Bayes.

Everything in §§1–4 below was written **before any construction code existed**; the
predictions are quoted verbatim in the test module's docstring. Verdicts are §5.

**The stage's prediction as P25 (b) states it is corrected here in one clause** (Claude Opus 5
(1M), 2026-09-27, found while doing the derivation; the stage's wording is Fable's,
*proposed*). It reads: "a choice actually made is a sample, which is §5.7's noise (`rand.py`),
and the jump it makes in the state is outside the smooth setting unless the state is the
distribution." The first half stands. The second does not: `rand`'s whole point is that the
branching lives in its *positions*, so a map `S -> rand(S)` needs only a smooth weight map
`S -> Delta_n` and `n` smooth return maps `S -> S`, and for a chooser the return maps are the
constant maps onto the vertices. A **sampled** collapse is therefore inside `poly(SmSet)` with
no extension at all. What is outside the smooth setting is the **deterministic** chooser,
`argmax`, which has no smooth extension across a tie. Determinism is the obstruction, not the
jump. Prediction 4 is the test of that correction.

## 1. The derivation, fixed before the run

**Setup.** `n` options, the `i`th carrying a real value `v_i` (a utility, a fitness, a
log-likelihood — the construction does not care which). Write `Delta_n` for the open simplex
of probability vectors `p` with `p_i > 0` and `sum_i p_i = 1`, and `T > 0` for a temperature.
The **logit** or **softmax** choice distribution is

    p*_i = exp(v_i / T) / sum_j exp(v_j / T).

### 1.1 The chooser: an entropy-regularized ascent on the simplex

Take the configuration `Q = Delta_n subset R^n`, the state `S = Q`, the readout the identity,
and the **exact** forcing `omega = dU` of the potential

    U(p) = <v, p> + T * H(p),    H(p) = -sum_i p_i log p_i,

i.e. the cost `k = -U` is the expected loss less the temperature times the entropy. In the
book's sign convention (`omega = dU`, `q' = q + sharp omega`) the components of the forcing are

    xi_i = v_i - T log p_i - T.

For the **response** take the Shahshahani sharp already in the library
(`replicator.shahshahani`), scaled by the duration `dt`:

    sharp_p = dt * (diag(p) - p p^T).

One step of `coalgebra.configuration` on that reactive vector space is then

    p_i  |->  p_i + dt * p_i * ( xi_i - <p, xi> )
           =  p_i + dt * p_i * ( (v_i - T log p_i) - sum_j p_j (v_j - T log p_j) ),

the constant `-T` cancelling because it is constant. At `T = 0` this is the forward-Euler
**replicator step** for the linear fitness `v`; at `T > 0` it is that step with the entropy
term added.

**The rest point is the logit.** The increment vanishes iff `v_i - T log p_i` is independent
of `i`, i.e. iff `p_i` is proportional to `exp(v_i/T)`; with `sum p = 1` that is `p*`. The
condition does not involve `dt`, so the rest point is the same at every duration.

**The mechanism, and it is the response's.** `1^T sharp_p = dt * (1 - sum p) p^T`, which is
zero exactly on the simplex. So (i) the jump preserves `sum_i p_i = 1` exactly, not to first
order; and (ii) the constant part of the forcing — the `-T`, and equally the Lagrange
multiplier that normalization would have introduced — is annihilated. **The normalization of
the choice distribution is contributed by the response, not by the cost.** That is the one
thing to hold on to, and Prediction 2 is designed to break it: with a Euclidean response the
same cost and the same wiring settle instead on the *unnormalized* Gibbs weights.

### 1.2 The same distribution as one Bayes step, in the dual chart

Stage (a)'s construction applies to the categorical family itself. Fix the gauge by
`lambda_n = 0`, so `lambda` ranges over `R^{n-1}`, with log-partition and Legendre map

    AA(lambda) = log( 1 + sum_{i<n} exp(lambda_i) ),    mu = grad AA(lambda) = softmax(lambda, 0)_{i<n},
    G(lambda) = grad^2 AA(lambda) = diag(mu) - mu mu^T   (the (n-1) x (n-1) block).

The gauge is needed: on the full `R^n` chart `grad^2 AA` is singular along `1`, so `G^{-1}`
does not exist. This is a wrinkle stage (a)'s continuous conjugate families did not have.

With the affine cost `k(lambda) = -<t, mu(lambda)>` for `t_i = (v_i - v_n)/T`, the forcing is
the exact `-dk` with components `G(lambda) t`, the response is `sharp = G^{-1}`, and one step
at duration 1 is the translation `lambda |-> lambda + t`. From the **uniform** belief
`lambda = 0` that lands on `softmax(v/T)` exactly: choosing with the logit probabilities is
the posterior over options after evidence whose log-likelihood ratios are the utility
differences, divided by the temperature.

**The two charts carry the same response and different forcings.** The Legendre map's
Jacobian is `G`, so pushing `sharp = G^{-1}` forward to the mean chart gives `G G^{-1} G = G`
— the Shahshahani sharp of §1.1, restricted to the gauge-fixed block. So §1.1 and §1.2 are the
*same* response in the two Legendre-dual charts. They are not the same arrangement: §1.1's
forcing carries the entropy term and §1.2's does not, and a `configuration` step is affine in
whatever chart its state lives in, so the chart is part of the arrangement. The consequence to
test is asymmetric: §1.2 reaches the logit in one exact step, §1.1 only in the limit.

### 1.3 A choice actually made

The distribution is not yet a choice. A chooser that collapses is a
`(cot(Q) tri rand)`-coalgebra, `rand.CotCoalgR` over `FiniteRV`: at state `p` it presents `p`
and, on a covector, returns the finite random variable with weights `p` and outcomes the `n`
vertices `e_1, ..., e_n`. Each branch is a *constant* map `Delta_n -> Delta_n`, hence smooth,
and the branching sits in `rand`'s positions, so nothing leaves `poly(SmSet)`. `rand.seeded`
with a sampler turns it back into an ordinary `CotCoalg`; `FiniteRV.bind` iterates it exactly
with no sampler at all.

Two designs, and the difference is a claim about state, not about probability:

* **the chooser that stores its choice** — state the vertex — is *absorbing*: at `e_i` the
  weights are `e_i`, so the next step returns `e_i` with probability one. It cannot reconsider.
* **the chooser that reports its choice** — state the distribution, the sample carried forward
  on the interface — never collapses, and the next covector still moves it.

What has no place here is `argmax`: as `T -> 0` at a fixed utility gap the derivative of the
choice in the gap goes to zero, and at a tie the choice is `1/n` for every `T`, so the pointwise
limit is locally constant off the tie set and has no smooth extension across it.

## 2. The small example, run first

Two options, `v = (1, 0)`, `T = 1`. The binary logit is
`p*_1 = 1/(1 + exp(-(v_1 - v_2)/T)) = 0.7310585786300049`, `p*_2 = 0.2689414213699951`.

Reducing §1.1's jump to the one free coordinate, `p_1' = p_1 + dt * g(p_1)` with
`g(p_1) = p_1 (1 - p_1) * (Dv - T log(p_1/(1-p_1)))` and `Dv = 1`. At `p*` the bracket is zero,
so `g'(p*) = p*_1 p*_2 * (-T / (p*_1 p*_2)) = -T`, and the jump's multiplier at the fixed point
is `1 - dt * T`. So the chooser converges iff `dt * T < 2` — the threshold is `dt = 2/T`,
which is §5.2's "largest eigenvalue of the mobility-weighted Hessian below 2" in this case.

Three options, `v = (2, 0.5, -1)`, `T = 0.7`: the logit is
`(0.8841084533193249, 0.10372286648030102, 0.01216868020037423)`.

## 3. Predictions, each able to fail

Fixed before any construction code; numbers are closed forms computed from the formulas above.

1. **The rest point is the logit.** Iterating §1.1's arrangement from the uniform `p` at a
   duration below the threshold converges to the logit: `(1,0)` at `T = 1` to
   `0.7310585786300049`, and `(2, 0.5, -1)` at `T = 0.7` to
   `(0.8841084533193249, 0.10372286648030102, 0.01216868020037423)`, each to relative error
   below 1e-12. The forcing is exact throughout, so choice needs no non-exact forcing. And
   `sum_i p_i = 1` is preserved by every *jump*, to machine precision, not to first order.
2. **The response supplies the normalization, and the Euclidean response does not.** With
   `rvect.euclidean(n, eta)` in place of the Shahshahani sharp — the same cost, the same
   forcing, the same wiring, the same interface — the same arrangement settles on the
   *unnormalized* Gibbs weights `p_i = exp(v_i/T - 1)`, i.e. `(1.0, 0.36787944117144233)`
   summing to `1.3678794411714423` at `v = (1,0)`, `T = 1`, and
   `(6.405409432177272, 0.751477293075286, 0.08816268936235741)` summing to
   `7.245049414614916` at `v = (2, 0.5, -1)`, `T = 0.7`. So it is the response's degeneracy
   along `1`, not the cost, that makes the answer a distribution.
3. **Softmax is also one Bayes step, in the dual chart.** §1.2's arrangement — natural chart,
   inverse-Fisher response, affine forcing — reaches the same logit in **one** step at duration
   1 from `lambda = 0`, to relative error below 1e-12; and its response pushed forward along
   the Legendre map equals §1.1's Shahshahani sharp on the gauge-fixed block, to 1e-12. The
   mean-chart Euler jump on the *affine* cost (no entropy term) does **not** land on the logit,
   because a jump is affine in its own chart; it misses by order `|t|^2`.
4. **A made choice is a sample, and it stays inside; `argmax` does not.** The collapsing
   `CotCoalgR` of §1.3, seeded with `rand.key_sampler`, has empirical frequencies matching `p`
   within `4/sqrt(N)` at `N = 20000`; `FiniteRV.bind` gives its exact two-step law with no
   sampling, and that law equals the one-step law, so the collapsed chooser is absorbing. It
   tensors with an ordinary `CotCoalg` and wires unchanged. The deterministic limit fails
   instead: the derivative of the binary choice in the utility gap, at gap 1, falls
   `0.196612 -> 0.000453958 -> 0` at `T = 1, 0.1, 0.01`, while at a tie the choice is exactly
   `1/2` at every `T`.
5. **Choice is selection at a temperature.** At `T = 0` §1.1's arrangement agrees step for step
   with the library's own `replicator_arrangement` for the symmetric payoff
   `A = v 1^T + 1 v^T` (whose `dU` differs from `v` by a multiple of `1`, which the response
   kills), to 1e-14. It approaches the best vertex without reaching it, since `sharp_p` vanishes
   there. And the duration matters for the path, not the destination: the measured stability
   threshold is `dt = 2/T` (2.0 at `T = 1`, 4.0 at `T = 0.5`) to within 1e-6, and the rest point
   is the same at every stable duration.

**Amendment to Prediction 5's measurement, made after construction code existed** (Claude Opus 5
(1M), 2026-09-27, and marked because it breaks the discipline of §§1–3). The threshold was to be
found by bisecting a convergence test to 1e-6; at 60 bisection steps of 4000 un-jitted `Phi`
steps each that run had to be stopped. The threshold is measured instead as an eigenvalue of one
step's Jacobian at the rest point, which should be exactly `1 - dt*T`: a stronger check, to
1e-12, of the same claim. The predicted *content* — threshold `dt*T = 2`, rest point independent
of the duration — is unchanged.

## 4. What this cannot establish, whatever the run shows

* Nothing about **whose** utilities, or where `v` comes from. `v` is an input; the construction
  is indifferent to its interpretation, and a book sentence that reads `v` as a preference is
  making a claim this run does not support.
* Nothing about **heredity**. That is stage (c), and it is not copying with variation:
  Prediction 5's `T = 0` case is differential persistence under a shared supply, which the book
  already has (§5.9 item 20, §9.1 *select*).
* Nothing about a chooser whose **option set changes**, since `n` is fixed in the positions of
  `rand` and in the dimension of `Q`.
* Nothing about whether a real chooser's frequencies are logit. That is an empirical claim about
  people or animals, and no run here touches it.
* Nothing about **continuous** choice, a distribution over an uncountable option set. That is
  §5.3's open row (c‴), and the finite case is inside the admissible subcategory as before.

## 5. Verdicts

`tests/test_discrete_choice.py`, 28 passed (85s), `arrangements/choice.py` the construction.
Predictions 2, 3 and 4 held exactly as written; Prediction 1 held except for the duration
condition it borrows from Prediction 5; Prediction 5's two numerical parts held and its vertex
clause is only half testable.

### Prediction 1 — the rest point is the logit. **Held, except the duration condition.**

Two options, `v = (1,0)`, `T = 1`, `dt = 0.5`, 400 steps: the settled state is
`0.7310585786300049` against the closed form's `0.7310585786300049`, relative error 1.5e-16, and
the sum is `0.9999999999999999`. Three options, `v = (2, 0.5, -1)`, `T = 0.7`, 800 steps:
`(0.88410845, 0.10372287, 0.01216868)`, relative error 2.7e-16. The rest point is the same at
`dt = 0.1, 0.3, 0.6`, as predicted. The forcing is exact: `jax` sees a symmetric Jacobian of
`omega`, so choice needs no non-exact forcing, unlike the replicator field's direct realization.
`sum_i p_i = 1` is preserved with maximum deviation 0.0 over 500 jumps at `T = 1`.

**What did not hold: `dt*T < 2` is not the operative condition.** At `dt = 0.9`, `T = 0.7`
(`dt*T = 0.63`, well inside the stability bound) the run gives `nan`. The binding condition is
positivity of the **transient**: a jump stays in the positive orthant only while
`dt * max_i(xibar - xi_i) < 1` at the state it leaves, since the increment is
`dt * p_i * (xi_i - xibar)`; below zero the entropy's `log` is `nan`. From the uniform start the
spread is exactly 1.5, so the bound is `dt < 2/3`, measured between 0.66 (holds) and 0.67
(`nan`) — where `dt*T < 2` would have allowed everything below `dt = 2.857`. At `T = 0` the same
bound reads `dt * max_i(vbar - v_i) < 1` and is measured between `dt*spread` 0.990 and 1.020.
So there are two duration conditions and the stability one is the looser; **the prediction named
the wrong one.**

### Prediction 2 — the response supplies the normalization. **Held.**

The same cost, forcing, wiring and interface with `rvect.euclidean` settle on
`(1.0, 0.36787944117144233)`, sum `1.3678794411714421`, at `v = (1,0)`, `T = 1`, and
`(6.40540943, 0.75147729, 0.08816269)`, sum `7.245049414614834`, at `v = (2, 0.5, -1)`,
`T = 0.7` — the unnormalized Gibbs weights `exp(v_i/T - 1)` in both cases, to 1e-10, and more
than 0.5 away from the logit. Directly: `sharp_p(1) = 0` to 1e-15 on the simplex and is of order
1e-3 off it. **The normalization of the choice distribution is the response's contribution, not
the cost's** — which is also why a Lagrange multiplier never appears.

### Prediction 3 — softmax is one Bayes step in the dual chart. **Held.**

One step at duration 1 from `lambda = 0` lands on the logit with relative error 8.6e-16, and the
step is the translation `lambda |-> lambda + t` to 1e-12. The inverse-Fisher response pushed
forward along the Legendre map equals the Shahshahani sharp on the gauge-fixed block to 2.8e-17.
The mean-chart jump on the affine cost misses: the one-step error at `eps * v` is 0.104, 0.030,
0.0078 at `eps = 0.5, 0.25, 0.125`, ratios 3.5 and 3.8, i.e. second order in `|t|`; iterated it
runs to a vertex, having no interior rest point at all. So a `configuration` step is affine in
its own chart, and **which chart the state lives in is part of the arrangement.**

### Prediction 4 — a made choice is a sample, and `argmax` is what is outside. **Held.**

The collapsing chooser's step is a `FiniteRV` with weights `p` and outcomes exactly the
vertices. Seeded with `key_sampler(0)`, 20000 draws at `p = (0.6, 0.3, 0.1)` give
`(0.6066, 0.2987, 0.0947)`, maximum deviation 0.0066 against the bound 0.0283. `FiniteRV.bind`
gives the exact two-step law with no sampler: nine branches whose weights are `(0.6, 0, 0, 0,
0.3, 0, 0, 0, 0.1)`, so the law on vertices is `p` again, deviation exactly 0.0 — the collapsed
chooser is absorbing, and that is visible without sampling. It tensors with an ordinary
`CotCoalg` (`Q_dim` 5, kind `FiniteRV`), so the wiring composes it unchanged. The deterministic
limit fails instead: the choice's derivative in the utility gap, at gap 1, is 0.196612,
0.000453958, 0.0 at `T = 1, 0.1, 0.01`, while at a tie it is exactly 0.5 at every `T`. **So the
stage's own prediction was wrong in one clause and the correction is confirmed: a sampled
collapse is inside `poly(SmSet)` with no extension; determinism is the obstruction.**

### Prediction 5 — choice is selection at a temperature. **Held, one clause half testable.**

At `T = 0` the chooser agrees with the library's `replicator_dynamics` for
`A = v 1^T + 1 v^T` step for step over 50 steps, maximum deviation 5.4e-20. The multiplier at
the rest point is exactly `1 - dt*T`, to 1e-12, at `(T, dt) = (1, 0.3), (1, 1), (0.5, 0.3),
(0.5, 1), (2, 0.3)`, so the stability threshold is `dt*T = 2`, which is the plan's
mobility-weighted-Hessian bound below 2 in this case; and the rest point is the same at every
stable duration inside the positivity bound. **Half testable:** "approaches the best vertex
without reaching it" is true of the map — the vertex is an exact fixed point, and the losing
coordinates stay strictly positive for the whole run, down to 6.6e-199 — but the best coordinate
is indistinguishable from 1.0 in double precision from step 64, so "not reached" is not
observable in the state.

### Three things the run found that were not predicted

**The simplex is exactly invariant but not attracting, and the numerics is not gauge-invariant.**
The Jacobian at the rest point has a second eigenvalue, transverse to the simplex,

     1 + dt * T * (1 - log Z),    Z = sum_j exp(v_j / T),

derived from `1^T sharp_p = dt (1 - sum p) p^T` and `<p, xi> = T (log Z - 1)` at the rest point,
and matched by the run to 1e-10. It exceeds 1 whenever `Z < e`. Then rounding-level sum error
grows geometrically: at `v = (1,0)`, `T = 2`, `dt = 0.9` the deviation is 8.4e-15 at step 100,
8.3e-13 at 200, 7.6e-9 at 400, 7.0e-5 at 600, and `nan` soon after — while at `T = 1`
(`Z = 3.7183 > e`) the deviation over 600 steps is exactly 0.0. So Prediction 1's exactness holds
**per jump** and fails **over a run**. The sting: adding a constant to every `v_i` changes no
choice at all and multiplies `Z` by `exp(c/T)`, so a model-irrelevant shift decides whether the
run survives — and `v + 1` at `T = 2` does survive, with maximum deviation below 1e-13 and the
same rest point to 1e-12. `replicator.py`'s docstring says the sum's drift is "self-correcting,
since `x^T A x > 0` makes `sum = 1` attracting"; the entropy-regularized forcing is not of that
form, and here the drift is *anti*-correcting.

**Absorption is a fact about weights, not about sampling.** The exact two-step law's weights are
`e_i` block by block, so the collapsing chooser's inability to reconsider is visible in
`FiniteRV.bind` with no sampler present.

**What a chooser can reconsider is settled by what its state holds.** The reporting chooser,
state `(p, last)`, records the drawn vertex without feeding it back and keeps moving under the
same covector: `p_1` goes 0.433, 0.537, 0.636, 0.722, 0.792 while the recorded vertex changes.
Nothing probabilistic distinguishes the two choosers; the state does.

### Reasonable or weird, by scale (the agent's judgment, not evidence)

* **Reasonable at every scale: the normalization is the response's.** A chooser does not have to
  compute a partition function to be normalized; it has only to move in a way that conserves the
  total. A cell's fractions, a population's shares and a person's probabilities are all totals
  that something already conserves, and the Euclidean control shows what you get when nothing
  does — unnormalized weights, which is what a real system that just accumulates looks like.
* **Reasonable at the population scale, and the best thing here: choice is selection at a
  temperature.** `T = 0` is replicator dynamics, differential persistence under a shared supply,
  the case the book already has; raising `T` makes the population hedge and settle on shares
  rather than fix. A mixed strategy and a logit choice are the same object.
* **Weird at the scale of a person: a chooser never actually arrives.** Nothing here reaches a
  choice except by sampling, and the settling chooser only approaches its corner. That is right
  for a population fixing an allele and wrong for someone picking a sandwich — which is why the
  `rand` collapse, not the descent, is what a made choice is.
* **Weird at every scale: a chooser that stores its choice can never reconsider.** Absorption is
  exact, not approximate: at a vertex the weights are `e_i` and nothing moves it, even under an
  arbitrary forcing. No cell, body or society is like that. The reporting chooser is the honest
  one, and the lesson is about state, not about probability.
* **Weird, and an artifact rather than a finding about the world: the gauge non-invariance.** That
  a uniform shift of every utility decides whether a run survives is a property of the embedding
  of the simplex in `R^n`, not of choosing. It is a real caution for anyone running these
  arrangements and it should never become a book sentence.
* **Backwards at the scale of a body: the colder the chooser, the larger the step it tolerates.**
  The stability bound is `dt*T < 2`, so a near-deterministic chooser looks more robust — but the
  positivity bound tightens in the other direction as the rest point nears a corner, so the
  apparent robustness is the looser of the two bounds and misleading.

### What this does not establish

The five limits of §4 stand unchanged: nothing about whose utilities, nothing about heredity
(stage (c)), nothing about a changing option set, nothing empirical about real choosers, and
nothing about choice over an uncountable option set (the plan's §5.3 row (c‴)). And one the run
adds: nothing about a chooser *coupled* to what it chooses among. Every arrangement here is a
closed scalar with `v` fixed from outside; a chooser whose options push back is untested.

### Sources, with locators

No new source is cited. The Gibbs variational principle that makes softmax the argmin of
`<-v, p> - T H(p)` is `khan2023bayesian` Eq. (2), p. 2, opened and checked by stage (a)'s run;
the Shahshahani response, the replicator identification and the `1^T sharp = 0` kernel
computation are the library's own `arrangements/replicator.py`, whose docstring carries the
derivation. The logit's closed form and the replicator equation are standard and are derived here
rather than cited.

---

# Part two: choosing by basin (David, 2026-09-27)

Stage (b)'s construction fed a **fixed** value for each option in from outside. One fixed value
vector gives one rest point, so the only way it can produce a discrete outcome is an external
coin — the sampler of §1.3. David: "can you think of another way? how does nature do it? it's not
basins or something?" It is, and this half is the one that matters. Everything in §§6–8 below was
written **before any new construction code existed**; verdicts are §9.

**Two things Part one got wrong, named before anything else** (Claude Opus 5 (1M), 2026-09-27).

1. Part one's report to David called the numerical fragility "fine for the book, a trap for the
   code." That sentence is withdrawn. The fragility is in the coordinates and reaches no claim,
   which is a reason to say nothing about it, not a licence to write around it.
2. Part one concluded "**determinism** is the obstruction, not the jump" (§1.3, Prediction 4,
   `STATUS.md` §3.24, plan §5.9 item 28). Too strong, and wrong as a general claim: a basin
   chooser is deterministic and perfectly inside the framework. The correct statement is narrower
   — what has no smooth extension is `argmax` **as a function of the option values**. A
   deterministic chooser as a *dynamics* is available, and near a tie it does not jump, it slows,
   because the basin boundary is an unstable rest point. Fixed in all three places.

## 6. The derivation, fixed before the run

**One arrangement, two knobs.** Keep everything: `Q = Delta_n`, the Shahshahani response
`sharp_p = dt (diag(p) - p p^T)`, the exact forcing `omega = dU`. Replace the linear potential by
a symmetric game's quadratic one plus the entropy,

    U(p) = (1/2) p^T A p  +  T H(p),     A = A^T,

so the forcing's components are `xi_i = (A p)_i - T log p_i - T` and one step is

    p_i |-> p_i + dt p_i ( xi_i - <p, xi> ).

The two knobs are **A**, how an option's value depends on the shares already committed to each
option, and **T**, how much the chooser hedges. Three cases of one family:

* `A = v 1^T + 1 v^T` and `T > 0`: stage (b)'s chooser, one rest point, the logit. (`dU` differs
  from `v` by a multiple of `1`, which the response kills.)
* any symmetric `A` and `T = 0`: the library's `replicator_arrangement` exactly.
* a coordination `A` and small `T`: **several** stable rest points, each with a basin, and the
  dynamics chooses with no coin and no temperature.

**Where the basins come from.** For two options and `A = diag(a_1, a_2)` at `T = 0`, reduce to the
free coordinate: `g(p_1) = p_1 (1-p_1) ((a_1 + a_2) p_1 - a_2)`, so the interior rest point is

    p_1* = a_2 / (a_1 + a_2),

and `g'(p_1*) = p_1* (1 - p_1*) (a_1 + a_2) > 0`: it is **unstable**, and the two vertices are
stable. The interior point is the basin boundary. Note what this makes possible: with
`a = (1, 2)`, `p_1* = 2/3`, so every start above 2/3 commits to option 1 — the option with the
*lower* payoff. **A chooser of this kind is not an optimizer.**

**Why it slows instead of jumping.** Near an unstable fixed point the jump map's multiplier is
`mu = 1 + dt g'(p_1*) > 1`, so a displacement `delta_0` grows as `mu^k delta_0` and the decision
lands when it reaches order 1: the number of steps is `-log(delta_0) / log(mu)`, linear in
`-log` of the distance to the boundary with slope `1 / log(mu)`. A chooser near a tie takes
longer; it does not become discontinuous. (Derived here, and already consistent with Part one's
coordination-game numbers, which is a check and not a prediction: `a = (1,2)`, `dt = 0.05` give
`g'(2/3) = 2/3`, `mu = 1.033333`, slope 30.50 steps per log-unit; the run's 0.67 and 0.6667
starts are a factor 100 apart, `log 100 = 4.605`, predicted gap 140.5 steps, measured 141.)

**Temperature dissolves the basins, at a critical value.** Rest points solve
`(Ap)_1 - (Ap)_2 = T log(p_1/p_2)`. As `p_1 -> 1` the left side stays bounded and the right side
diverges, so **at any `T > 0` no vertex is a rest point**: the entropy always pushes off the
corners. For the symmetric coordination game `a_1 = a_2 = a`, writing `p = ((1+m)/2, (1-m)/2)`,

    a m = 2 T artanh(m),    i.e.   m = tanh( a m / (2T) ),

which has `m = 0` always and a nonzero pair iff `a > 2T`. So the critical temperature is

    T_c = a / 2,

a pitchfork: below it two committed states with a boundary at the midpoint, above it one hedging
state at 1/2. This is mean-field Ising with coupling `a/2`, which is a consistency check worth
stating — a two-option coordination game with an entropy term *is* the bistable switch, and that
is what makes this the mechanism nature uses rather than an analogy.

## 7. The small example, run first

Two options, `A = diag(1, 2)`, `T = 0`, `dt = 0.05`. Boundary at `p_1* = 2/3`; starts at 0.90,
0.70, 0.67, 0.6667 commit to option 1 and starts at 0.6666, 0.66, 0.40, 0.10 commit to option 2
(Part one's run, above). Symmetric case `A = diag(1, 1)`: `T_c = 0.5`.

## 8. Predictions, each able to fail

1. **The basin boundary is `a_2/(a_1+a_2)`.** At `T = 0` the flip in the outcome, located by
   bisecting the starting share, sits at `2/3` for `a = (1,2)` and at `4/5` for `a = (1,4)`, to
   1e-6. The interior rest point is unstable and the two vertices are stable, checked as
   `g'(p_1*) > 0` against the closed form `p_1*(1-p_1*)(a_1+a_2)` to 1e-12.
2. **The worse option wins from a large enough share.** With `a = (1,2)`, every start above 2/3
   commits to option 1, whose payoff is half option 2's. So no book sentence may call this
   chooser an optimizer.
3. **Deciding slows logarithmically at the boundary, at the derived rate.** Steps to decide,
   against `-log` of the distance from `p_1*`, is a straight line of slope `1/log(1 + dt g'(p_1*))`
   — 30.50 at `a = (1,2)`, `dt = 0.05`, and 15.30 at `dt = 0.1` — fitted over distances from 1e-2
   to 1e-8, to 2%. A chooser near a tie is slow, not discontinuous.
4. **The pitchfork at `T_c = a/2`.** For `A = diag(1,1)`: below `T = 0.5` there are three interior
   rest points and the outcome depends on the start; above it there is one, at `(1/2, 1/2)`, and
   every interior start reaches it. The measured critical temperature is 0.5 to 1e-4, and the
   committed share just below it grows like `sqrt(T_c - T)` (the pitchfork's exponent, slope of
   `log m` against `log(T_c - T)` equal to 1/2, to 5%).
5. **No coupling, no basins, and it is one arrangement family.** With `A = v 1^T + 1 v^T` there is
   exactly one rest point at every `T > 0`, the logit of Part one, reached from every interior
   start — so the basins are the coupling's doing and nothing else's. And the same constructor
   agrees step for step with `replicator_arrangement` at `T = 0` (to 1e-14) and with Part one's
   `logit_arrangement` for the uncoupled payoff (to 1e-14).

## 9. Verdicts, Part two

`tests/test_discrete_choice.py`, 52 passed (195s) with Part one; `arrangements/choice.py`'s
Part-two section the construction. **All five held.** One arithmetic slip of mine: Prediction 3's
second number, "15.30 at `dt = 0.1`", should be 15.4946 — the formula was right and my arithmetic
was not, and the test uses the formula.

### Prediction 1 — the basin boundary is `a_2/(a_1+a_2)`. **Held.**

Bisecting the starting share for the flip in the outcome gives `0.6666666667` for `a = (1,2)`
against the closed form `2/3`, and `0.8000000000` for `a = (1,4)` against `4/5`, each to the
bisection's 1e-6. The interior point's rate `g'(p_1*) = p_1*(1-p_1*)(a_1+a_2)` matches to 1e-12 and
is positive in both cases, so it repels; each vertex is an exact fixed point of the step. So the
chooser has **two** stable rest points and the interior one is the boundary between their basins.

### Prediction 2 — the worse option wins from a large enough share. **Held.**

Payoffs 1 and 2. Starts at 0.67, 0.70, 0.80, 0.95 all commit to the option paying **1**, in 564,
493, 444 and 395 steps; a start at 0.66 commits to the option paying 2. **So no book sentence may
call this chooser an optimizer.** What it does is lock in, and what decides is where it started.

### Prediction 3 — deciding slows logarithmically, at the derived rate. **Held.**

Steps to decide against `-log` of the distance from the boundary is a straight line. At
`a = (1,2)`, `dt = 0.05`: measured slope 30.478 against the derived `1/log(1 + dt g'(p_1*))` =
30.497, a ratio of 0.9994, over distances from 1e-2 to 1e-8 (531 to 952 steps). At `dt = 0.1`:
15.4950 against 15.4946, ratio 1.0000 (261 to 475 steps). Monotone in both. **This is the repair of
Part one's overstatement**: a deterministic chooser is available, and near a tie it does not jump,
it slows — because the tie is a hyperbolic unstable rest point whose jump multiplier is
`1 + dt g'(p_1*) > 1`, so a displacement grows geometrically and the wait is the logarithm of how
close the start was.

### Prediction 4 — the pitchfork at `T_c = a/2`. **Held.**

For `A = diag(1,1)` the measured critical temperature is `0.49999996` against `0.5` (the shortfall
is the rest-point grid's resolution, which cannot resolve roots closer together than its spacing;
it errs low by construction, never high). Three interior rest points at `T = 0.1, 0.3, 0.45`, one
at `T = 0.5, 0.7`. The outer pair matches the closed-form `m = tanh(am/2T)` to 1e-4 at every
temperature tested — at `T = 0.45`, `m = 0.5254` gives `(0.2373, 0.7627)` against the located
`(0.237285, 0.762715)`. The exponent of `m` in `T_c - T` is 0.4984 against `1/2`. Above `T_c` every
interior start (0.05, 0.3, 0.9) reaches `(1/2, 1/2)` to 1e-9; below it (`T = 0.2`) a start at 0.45
commits to option 2 and 0.55 to option 1, symmetrically at `0.007188 / 0.992812`.

Also held, and worth its own line: **at any positive temperature no vertex is a rest point.** The
entropy's logarithm diverges at a corner while the payoff stays bounded, so a chooser that hedges
at all never fully commits; what it settles on is a committed *share*. Only at `T = 0` is the
outcome a single option.

### Prediction 5 — no coupling, no basins, and it is one family. **Held.**

With `A = v 1^T + 1 v^T` there is exactly one interior rest point at `T = 0.7`, `0.80667863`,
which is Part one's logit to 1e-6; and the same constructor agrees step for step with
`logit_arrangement` to 2.8e-17 and, at `T = 0`, with the library's `replicator_dynamics` to
**exactly** 0. A non-symmetric payoff is rejected, as in `replicator.py`. **So the basins are the
coupling's doing and nothing else's**: what Part one lacked was not a construction but the
dependence of an option's value on its own share.

### What Part two adds, and what it withdraws

Adds: **discrete choice is in the mathematics twice, and the second way needs no coin.** A single
arrangement family, with two knobs — the coupling `A`, how an option's value depends on the shares
committed to it, and the temperature `T`, how much the chooser hedges — covers Part one's softmax
chooser (`A` uncoupled), the library's replicator (`T = 0`), and a committing chooser with several
stable states and a basin each (coupled `A`, `T` below `a/2`). The bistable two-option case with an
entropy term **is** mean-field Ising at coupling `a/2`, which is why this is a mechanism and not an
analogy.

Withdraws: Part one's "**determinism** is the obstruction, not the jump." A basin chooser is
deterministic and entirely inside the framework. The correct statement is narrower: what has no
smooth extension is `argmax` **as a function of the option values**. Fixed in Part one's §1.3,
`choice.py`'s module docstring, `STATUS.md` §3.24 and the plan's §5.9 item 28.

### Reasonable or weird, by scale (the agent's judgment, not evidence)

* **Reasonable at every scale, and this is the payoff: a chooser that commits is a chooser whose
  options' values depend on how many have already chosen them.** A cell committing to a type, a
  convention locking in, a supersaturated solution nucleating, a gene switch latching — all
  coupled, all bistable, none of them needing a coin. The book can say this with mathematics behind
  it.
* **Reasonable and uncomfortable: the worse option can win.** Whichever option already has enough
  share takes it. At the scale of a society that is path dependence and it is right; at the scale
  of a person it is habit; at the scale of a cell it is commitment. Nothing here optimizes.
* **Reasonable at every scale: deciding takes longer the closer the call.** That the wait is the
  logarithm of the closeness, so a near-tie is slow but never infinite, matches the experience and
  matches critical slowing in physics. It is also what makes a deterministic chooser possible at
  all: the discontinuity that `argmax` has is spent on time instead.
* **Weird at the scale of a person, reasonable at the others: above a critical temperature the
  chooser cannot choose at all, no matter the start.** For a population or a material that is a
  phase transition and exactly right. For a person "too noisy to decide" is recognizable but the
  sharp threshold is not.
* **Weird, and a limit of the two-option case: `T_c = a/2` is derived for two options only.** With
  three or more the transition structure is not established here, and a book sentence should not
  name a critical temperature without saying "two options."
* Part one's cautions stand: the simplex's numerical fragility is an artifact of its embedding and
  belongs in no book sentence, and the uncoupled chooser's softmax rest point is not a claim about
  real choosers' frequencies.

### What Part two does not establish

* **Nothing about where `A` comes from.** The coupling is supplied, as `v` was. That an option's
  value rises with its share is the assumption doing all the work, and this run does not derive it
  from anything.
* **Nothing for a non-symmetric payoff.** Only potential games are covered, the same scope limit
  `replicator.py` carries; rock-paper-scissors is outside the trick construction.
* **Nothing about more than two options at positive temperature.** The basin boundary, the slowing
  rate and the critical temperature are all derived and checked for `n = 2`.
* **Nothing about a chooser coupled to something other than itself.** The coupling here is a
  frequency dependence inside one closed scalar arrangement; a chooser wired to another box whose
  state feeds its values is still untested, and that is the honest remaining gap.
* Nothing about heredity, which is stage (c) and still David's decision.

---

# Part three: composed with other things (David, 2026-09-27)

Part two closed with "a chooser wired to another box whose state feeds its values is untested,"
which David read as "we can't tell if discrete choice can actually affect other things. that is
not useful." **He is right that the sentence was worse than the truth, and it is withdrawn.** The
truth, stated exactly:

* that these arrangements *compose* is not in question. `arr` is a category and `Phi` a functor;
  `tests/test_composition.py` checks both operad legs, including a forcing that comes from no
  potential, and `tests/test_wiring.py` the wiring. Composition in `arr` is pullback-and-sum on
  the forcing, and it is the one thing that changes when a box is wired in: the integrator — the
  response, the state, the readout — is untouched.
* so the real question for any closed test, this one included, is **whether its conclusion
  survives an added forcing term.** Part one and two tested closed scalars. That is what was
  untested, and saying it as "we can't tell if it can affect other things" was sloppy.

This part tests it. Predictions in §10 were written before any composition code existed.

## 10. Predictions, fixed before the run

The **pushed chooser** is the arrangement of §6 made open: interface `<0|1>` on the domain, so it
receives one number `b` forward, and potential `U(q) + b (q_1 - q_2)`. The **pusher** is a box
`I -> <0|1>` publishing a constant `b`. The composite is `compose_seq(pusher, chooser)`, a closed
scalar, and `Phi` runs it. For two options and `A = diag(a, a)` at `T = 0` the free coordinate obeys
`g(p_1) = p_1 p_2 (a (2 p_1 - 1) + 2 b)`.

1. **Composition is faithful.** The composite's chooser-state trajectory equals, to 1e-14, the
   closed chooser whose potential has `b (q_1 - q_2)` added by hand. So wiring does exactly
   pullback-and-sum and leaves the integrator alone.
2. **An outside push moves the basin boundary, linearly, at rate `1/a`.** The rest point is
   `p_1* = 1/2 - b/a`: with `a = 1`, `b = 0.1` the boundary is at 0.4 and with `b = -0.2` at 0.7,
   located by bisection to 1e-6. So one box **can** move where another box's choice tips.
3. **Above a critical push bistability is destroyed: `|b| >= a/2`.** The boundary must lie in
   `(0,1)`, so at `|b| >= a/2` it leaves the simplex and one option wins from every start. With
   `a = 1` that push is 0.5: at `b = 0.6` every start commits to option 1, and at `b = 0.4` the
   start still decides.
4. **And it is hysteresis: releasing the push does not undo the flip.** Push a chooser started at
   `p_1 = 0.2` past the threshold so it commits to option 1, then set `b = 0`; it stays, because a
   vertex is an exact fixed point and the response vanishes there. Checked to 0. So a chooser can
   be moved by something outside it, above a threshold, and the move is permanent — which is what
   makes commitment worth the name.
5. **The chooser affects downstream too.** With the chooser publishing its share of option 1 into a
   downstream box whose forcing is `-(y - share)^2/2`, the downstream state ends at 1 when the
   chooser commits to option 1 and at 0 when it commits to option 2, to 1e-6 — from two starts
   either side of the boundary and nothing else different. So the choice is visible outside the
   chooser, and a composite's outcome depends on which way an internal choice went.

## 11. Verdicts, Part three

`tests/test_discrete_choice.py`, 63 passed (389s, all three parts). **Predictions 2, 3, 4 and 5 held
as written. Prediction 1 was refuted as written, by something real.**

### Prediction 1 — composition is faithful. **Refuted as written: I forgot retroactivity.**

I compared the three-box composite against a hand-sum containing only the *upstream* push, and it
deviated by 3.2e-2, with the boundary landing at 0.3876 instead of the predicted 0.4. The cause is
not a bug: composition in `arr` is pullback-and-sum over **every** box, so the downstream box that
merely *reads* the chooser's share contributes `(y - q_1)` to the chooser's own forcing. **A box
that reads a choice tilts it.** That is retroactivity, which is Chapter 2's own subject, and it
arrived here unasked.

Compare the right things and composition is exact. The two-box composite, nothing reading, equals
the hand-summed push to **0.0**. The three-box composite equals a hand-sum that includes the
follower's pullback to **1.1e-16**. So the faithfulness claim holds; my statement of what to
compare it with did not.

### Prediction 2 — an outside push moves the boundary linearly. **Held.**

`p_1* = 1/2 - b/a`, located by bisection on the two-box composite: 0.39999994 at `b = 0.1`
(predicted 0.4), 0.70000001 at `b = -0.2` (predicted 0.7), 0.50000012 at `b = 0` (predicted 0.5),
each inside the bisection's 2.4e-7. **So one box can move where another box's choice tips.**

### Prediction 3 — above `|b| = a/2` the outside decides. **Held.**

At `b = 0.6 > 0.5` starts of 0.02, 0.2 and 0.49 all commit to option 1 (90, 51, 43 steps). At
`b = 0.4` the start still decides, with the boundary at `1/2 - 0.4 = 0.1`: 0.05 goes to option 2 and
0.2 to option 1. So there is a threshold of outside influence above which the chooser stops
choosing and below which it does not.

### Prediction 4 — hysteresis. **Held.**

From `p_1 = 0.2`, a push of 0.6 commits the chooser to option 1 in 59 steps. Release the push and it
stays: drift below 1e-12 over 2000 steps, the losing share still decaying (1.2e-206). Unpushed from
the same start it goes to option 2 instead. **So a chooser can be moved from outside, only above a
threshold, and the move does not undo itself.** That is what makes commitment worth the name, and it
is the one sentence of Part three most worth the book's while.

### Prediction 5 — the choice reaches downstream. **Held.**

Two runs differing only in the chooser's start: the follower ends at exactly 0.0 when the chooser
commits to option 2 and 1.0 when it commits to option 1. So which way an internal choice went
decides the composite's outcome.

### What Part three settles, and what it withdraws

Withdraws Part two's closing sentence, that a wired chooser was "untested" in a way that left it
unknown whether choice can affect other things. It can, in **both** directions at once, and the
framework supplies both without being asked: what pushes a chooser moves its boundary, and what
reads it tilts it.

Settles: **composition changes exactly one thing, the forcing**, by pullback-and-sum, leaving the
integrator — response, state, readout — alone. So the audit question for any closed-scalar result in
this library is whether its conclusion survives an added forcing term, and the term most easily
forgotten is the one a downstream *reader* contributes. That is the premise of the book plan's P26.

### Reasonable or weird, by scale (the agent's judgment, not evidence)

* **Reasonable at every scale, and the result to keep: influence has a threshold and its effect is
  permanent.** Below the threshold an outside push only shifts where the tipping point is; above it,
  the outside decides; and either way, once committed, removing the influence changes nothing. A
  convention, a habit, a differentiated cell, a market that has standardised.
* **Reasonable and important: being read changes the thing read.** A downstream box that only
  observes the chooser's share still tilts the choice. At the scale of a society that is
  measurement changing behaviour; at the scale of a circuit it is loading; in the book it is
  retroactivity, and it is not an extra assumption but what pullback-and-sum already says.
* **Weird at the scale of a person: nothing here can be persuaded back.** Once past a vertex, no
  finite push returns the chooser, because the response vanishes there. Real commitments are
  revisable, and what the model lacks is any noise at the committed state — which `rand.py` has and
  this construction did not use.
* **Weird as a limit: the threshold `a/2` is two options and a symmetric coupling.** No book
  sentence should give a number for how much influence it takes to flip a choice.

### What Part three does not establish

* Nothing about a **reader that does not tilt**. Whether an interface can read a choice without
  loading it is a real question in `arr` and is not answered here.
* Nothing about noise at a committed state, so nothing about a commitment that can be undone.
* Nothing for more than two options, a non-symmetric payoff, or a positive temperature under a
  push: all three composition results are `n = 2`, `A = diag(a,a)`, `T = 0`.
* Nothing about a **cycle**: the chooser and its pusher are a chain, not a loop, so a chooser whose
  own choice changes the push it receives is untested.

**A note for the next run:** this file now takes 389s. Select node IDs rather than the whole file
(`scripts/run_tests.sh 200 tests/test_discrete_choice.py::test_name`); the expensive tests are the
three bisections and the slowing fits.

### Addendum: a unilateral reader (David, 2026-09-27: "couldn't we just make it be unilateral?")

**Yes, and §11's headline needed narrowing.** It said "a box that reads a choice tilts it," full stop.
The true statement names the condition:

> A box whose forcing is **exact** cannot read a choice without tilting it. A box whose forcing is
> **not exact** can read it and load nothing.

The mechanism is one line of `compose_seq`. The composite's forcing pulls the downstream box's
`omega_g` back along `A : (q, l, n) |-> (q_g, m, n)` with `m = out_f(q_f, l)`, so `omega_g`'s
component on its *incoming value* `m` lands on the upstream box's configuration. That component is
the whole of the tilt. If `omega_g = dU_g` and `U_g` depends on `m` at all, then `∂U_g/∂m ≠ 0` and
the tilt is unavoidable — exactness forces reciprocity. But `arr`'s forcing need not be exact
(§5.2), so one may take the same `Q`-component, `(m - y)`, with the `m`-component set to **zero**:
a genuine one-form on `Q_g × M^+ × N^-`, not closed, which descends toward whatever arrives and
pushes nothing back.

**Checked** (Claude Opus 5 (1M), 2026-09-27, after the predictions above and reported as an
addendum, not as one of them). With the same pusher, chooser and start:

* unilateral reader against **no reader at all**: chooser trajectory deviates by **0.0** over 200
  steps;
* exact reader against no reader: 3.2e-2, as §11 reports;
* and the unilateral reader still reads — its state reaches 0.9999999999999997 while the chooser's
  share reaches 0.9999999999999998;
* the forcing is genuinely non-exact: the antisymmetric part of its Jacobian in `(y, m)` is 1.0.

**What this is, and what it is not.** It is not a trick: reading without loading is what an ideal
voltmeter or an ideal op-amp input does, and it is an *idealization* in physics — real ones load a
little. So what the non-exactness buys here is exactly that idealization, and a book sentence should
say so rather than claim observation is free. It does **not** follow that non-exact forcings are
idealizations in general: the attention paper's cue and `replicator.py`'s direct realization are
non-exact and are not.

**A shape worth noticing, not yet a claim.** The commons run found that "a fixed-rule external
enforcer is exact rather than unilateral, so what is one-way is an enforcer whose own state sets the
sanction" (plan §2, P8). Same word, same direction, and there the mechanism was state-dependence
rather than non-exactness. Whether those are one mechanism is **not established here** and is worth
P26's attention: if exactness is what forces reciprocity across the library, that is a general fact
about `arr` and not a fact about choosers.
