# Active inference: three tests (book plan §8.2 P5 (b))

David, 2026-09-29: "I don't really like that we're saying we can't do it … Can we check
anything?", then "Run P5 (b)'s three active-inference tests … Write predictions first, report each
result as held or refuted, and do not repair a failed prediction." The three tests were proposed
by Claude Opus 5.5 (1M), 16e01cc3; this run is Claude Opus 5.5, 9b16a421. The book's mapping they
test is its `notes/ACTIVE-INFERENCE.md`: at fixed precision the attention paper's cost is the
precision-weighted prediction error, and what is missing is (1) the log-determinant (normalizer)
term, (2) stochasticity, (3) expected free energy for choosing among policies.

§§1–3 were written, with their numbers, **before any test or construction code**. The numbers
come from closed forms computed with plain numpy/scipy in a scratch script, not from any
arrangement. §4 onward is the run's.

## 1. Test (i): precision learned by descent

**Construction.** A configuration coalgebra with the Euclidean response `eta` on the belief
`(mu, ell)`, `ell = log Pi` (so a precision is positive in every chart it can reach), closed, with
the exact forcing `dU`, `U = -k`, and for data `x_1 … x_N`

    k(mu, ell) = 1/2 e^ell S(mu) - 1/2 ell,    S(mu) = (1/N) sum_j (x_j - mu)^2,

the per-datum negative log-likelihood of `N(mu, 1/Pi)` less its constant. The `-1/2 ell` is the
`-1/2 log det Pi` of P5 (b)(i). The forcing's components are `Pi (xbar - mu)` on the mean — the
precision-weighted prediction error — and `1/2 (1 - Pi S(mu))` on `ell`: precision rises when the
errors are smaller than the belief expects and falls when they are larger. That is the
active-inference precision update, here a descent and nothing else.

**Data.** 1-D: `x = 0.5 * (0.3, -1.1, 0.8, 2.0, -0.4, 0.9, 1.5)`. Start `mu = 0`, `Pi = 1`.

- **(i-a) With the term, precision settles at the inverse variance of the errors.** The iteration
  converges to `mu* = xbar = 0.2857142857142857` and `Pi* = 1/S = 4.073150457190358`, the inverse
  of the **maximum-likelihood** (divide-by-`N`) variance, not the unbiased one, to relative error
  below 1e-10. The rest point does not depend on `eta`.
- **(i-b) Without the term, precision collapses to zero, and slowly.** Dropping `-1/2 ell`, the
  `ell` forcing is `-1/2 Pi S(mu) < 0` always, so `Pi` decreases monotonically at every step, and
  asymptotically `Pi_n ≈ 2/(eta S n)`: at `eta = 0.2` and `n = 20000`, `Pi_n * eta * S * n / 2` is
  within 2% of 1. The mean still converges to `xbar` (the sum of `eta Pi_n` diverges like `log n`),
  but ever more slowly. Reading (Claude's, *proposed*): a part that pays only for error weighted
  by its attention learns to attend to nothing; the normalizer is what makes attending worth it.
- **(i-b′) In two dimensions, the log det.** `Pi = L L^T`, `L` lower triangular with diagonal
  `e^{a_1}, e^{a_2}` and off-diagonal `b`, so `log det Pi = 2(a_1 + a_2)`, and
  `k = 1/2 tr(Pi Shat(mu)) - 1/2 log det Pi`. On the eight points
  `(0.2,1.0) (1.3,0.4) (-0.7,-0.2) (0.9,1.8) (0.1,-0.9) (1.6,0.7) (-0.3,0.5) (0.5,1.1)` it settles at
  `mu* = (0.45, 0.55)` and `Pi* = Shat^{-1} = [[2.259419664750335, -0.9122761407883151],
  [-0.9122761407883151, 2.0419859731634826]]`, to relative error below 1e-10 (Frobenius).
- **(i-c) Two sources of one quantity: the learned precisions are the attention weights.** A
  belief `mu` with one log-precision per source, `k = sum_s [1/2 Pi_s S_s(mu) - 1/2 ell_s]`, on
  `xA = 1 + 0.3 * (0.2, -0.9, 1.1, 0.4, -1.3, 0.6)` and `xB = 1 + (0.6, -1.1, 0.9, -0.5, 1.4, -1.2)`.
  It settles at `mu* = 1.0056988365525892`, `Pi_A = 15.618779730059675`,
  `Pi_B = 0.9951808214273963`, to relative error below 1e-9: the mean is the precision-weighted
  mean of the two sources with the precisions it learned (weight 0.940 on the tight source), not
  the plain mean 1.0108333. So the message algebra's weight `w` (a precision) is learned by the
  same descent that learns the mean, which is the note's "attention is precision optimization"
  with the precision now a state and not a set number.

## 2. Test (ii): policy selection is the chooser of §5.9 item 28

Active inference selects a policy with probability `q(pi) ∝ E(pi) exp(-gamma G(pi))`: `G` the
expected free energy, `gamma` a precision over policies, `E` a habit prior.

- **(ii-a) The discrete-choice arrangement reproduces it once `G` is given.** `choice.logit_arrangement`
  with `T = 1/gamma` and values `v = -G + T log E` settles at `q`. At `G = (1.3, 0.4, 2.1)`,
  `gamma = 2`, `E = (0.2, 0.5, 0.3)`: `q = (0.06087552357274065, 0.9206886420666024,
  0.01843583436065685)`; with `E` uniform, `(0.13790166651764385, 0.8342564671867838,
  0.02784186629557244)`. Each to relative error below 1e-12.
- **(ii-b) The chooser's cost is the free energy of the policy posterior.** With
  `k(p) = <G, p> - T <log E, p> - T H(p)`, `k(p) - k(q) = T KL(p || q)` at every interior `p`
  (checked at five seeded random points, to 1e-12), so `gamma k` is, up to a constant, active
  inference's variational free energy over policies, and the chooser descends it.
- **(ii-c) Opened, with `G` arriving from a stateless part, nothing changes.** The chooser opened
  to receive `G` on a 3-dimensional input port, composed after a box that publishes the constant
  `G` and has no state, settles at the same `q` to 1e-12: the chooser's pullback onto its input is
  nonzero, but there is no state behind it to move. This is the control for (iii-d).

## 3. Test (iii): expected free energy computed by a part that runs futures

**World.** An arrangement of the book's kind: state `x`, Euclidean response `eta_w = 0.5`, input
`u` on a 1-dimensional port, `U(x) = -1/2 kappa x^2 + u x` with `kappa = 0.4`, publishing `x`. One
step of `Phi` is `x' = A x + B u` with `A = 0.8`, `B = 0.5`. The model adds process noise
`Q = 0.005` and, per policy, observation noise `R`; the preferred outcome is `N(o*, P)` with
`o* = 1`, `P = 0.5`. Three one-step policies: **push-dark** `(u = 1, R = 0.3)`, **stay-look**
`(u = 0, R = 0.02)`, **retreat-dark** `(u = -1, R = 0.3)`.

**The simulator** runs futures by calling `Phi(world)`: the predicted mean is one `Phi` step from
the belief mean with input `u`; the predicted variance is `J Sigma J^T + Q` with `J` the Jacobian
of that step (by `jax.jacfwd`, exact since the world is linear). Then

    G(pi) = KL( N(mu_o, V + R) || N(o*, P) ) + 1/2 log(2 pi e R),    V = J Sigma J^T + Q,

risk plus ambiguity. At horizon 2 a policy is a pair of steps and the belief variance between them
is the expected posterior `(1/V + 1/R)^{-1}` (a Gaussian's posterior variance does not depend on
the value observed).

- **(iii-a) The rollout is running an arrangement.** `G` from the simulator equals the closed form
  at horizon 1 for `Sigma ∈ {2, 0.5, 0.05}` and `m ∈ {0, -0.5}`, and at horizon 2 for all nine
  sequences at `Sigma = 2` and `0.05`, `m = 0`, to 1e-12. At `m = 0`, `Sigma = 0.2`:
  `G = (1.0718873162515554, 0.7080121190054778, 3.0718873162515554)`.
- **(iii-b) Risk plus ambiguity is expected cost less information gain, and the gain is the
  note's term.** `G = -E_q[log p~(o)] - I`, with `I = 1/2 log(1 + w_obs / w_pred)`,
  `w_obs = 1/R`, `w_pred = 1/V`: `notes/ACTIVE-INFERENCE.md`'s `1/2 log(1 + w_recv/w_pi)` with the
  received weight the observation's precision and the prior weight the predicted belief's. To
  1e-12. The ambiguity `1/2 log(2 pi e R)` is `1/2 log det` of a covariance, the normalizer of
  test (i), and it does not depend on the belief at all.
- **(iii-c) Look while unsure, push once sure, and the switch is where the closed form puts it.**
  The chooser (`T = 0.2`) fed the simulator's `G` at `m = 0`: at `Sigma = 0.2` it puts
  `0.86048578343851367` on stay-look and `0.13950788291340072` on push-dark; at `Sigma = 0.02`,
  `0.71147230606178846` on push-dark and `0.28849539314548839` on stay-look (1e-10). The two are
  equal at `Sigma = 0.050968866365422215`, where the share ratio is 1 to 1e-9. Mechanism: the
  ambiguity is belief-independent, so what changes with `Sigma` is the information gain, which
  shrinks as the belief sharpens, and its advantage for the precise policy falls below the
  pragmatic advantage of pushing.
- **(iii-d) Wired exactly, the plan bends the belief toward what it prefers.** The simulator made a
  box with state `m` (Euclidean response), its own evidence cost `1/2 w_ev (m - x_obs)^2` with
  `w_ev = 4`, `x_obs = -0.5`, `Sigma = 0.5` fixed, publishing `G(m)`; composed by `compose_seq`
  with the chooser at `T = 0.5` whose forcing is **exact**, `d(-<G, p> + T H(p))`. By item 30
  (an exact reader tilts what it reads) the chooser's pullback `-sum_pi p_pi dG_pi/dm` moves the
  belief: the composite settles at `m* = -0.11620980279789729`, not at the observed `-0.5` — a
  bias of `+0.384` toward the preferred outcome — with
  `p* = (0.2703991878894675, 0.7261863226871871, 0.00341448942334541)`. With the chooser made
  **unilateral** (the forcing's component on the incoming `G` set to zero, so non-exact) the
  composite settles at `m* = -0.5` and `p* = (0.4069992837643438, 0.5914956883515895,
  0.00150502788406671)`. Both to 1e-9. Reading (Claude's, *proposed*): the unilateral wiring is
  active inference, where beliefs about the present answer to evidence and the policy answers to
  `G`; the exact wiring is one cost descended by both, and it is wishful thinking.

**Not predicted:** step counts, rates, which two-step sequence is best. **Not built:** a closed
loop acting on a world and updating its belief from what it sees; stochastic observations (the
Gaussians here are carried as parameters, never sampled).

## 4. Verdicts, tests (i) and (ii) (the run's; `tests/test_active_inference.py`)

**(i-a) Held.** At `eta = 0.2` and `0.1`, 4000 steps: `mu*` and `Pi* = 4.073150457190358` to below
1e-10; the unbiased variance's inverse (3.49) is not where it goes.
**(i-b) Held.** `Pi` fell at every one of 20000 steps; `Pi_n eta S n / 2 = 0.99776` at `n = 20000`,
`Pi_n = 0.00203`; the mean still reached `xbar` (off by 6e-15).
**(i-b′) Held.** 2-D, 20000 steps at `eta = 0.1`: `mu*` and `Pi* = Shat^{-1}` below 1e-10.
**(i-c) Held.** `mu* = 1.0056988365525892`, `Pi_A = 15.618779730059543`, `Pi_B = 0.9951808214273963`,
and `mu*` is the precision-weighted mean with those weights, below 1e-9.

**(ii-a) Refuted as written.** The chooser passes through `q` — within 1e-7 at step 100 — and then
leaves the simplex, settling at `(0.02732373, 0.16529892, 0.00551657)`, summing to 0.198: the
**unnormalized Gibbs weights** `exp(v/T - 1)`, which item 28 of the book plan said only the
Euclidean negative control reaches. Same for the habit case.
**(ii-b) Held**, at all five points, below 1e-12.
**(ii-c) Refuted as written**, the same way (it settles at the same off-simplex point).

**The mechanism** (derived after the failure, from the step itself). The Shahshahani step
changes the total by `sum p' - sum p = dt (1 - sum p) <p, xi>`, so `eps = 1 - sum p` obeys
`eps' = eps (1 - dt <p, xi>)`. Near the logit, `<p, xi> = T (log Z - 1)`, `Z = sum_i exp(v_i/T)`.
So the simplex is invariant, but it **attracts** nearby states only when
`0 < dt T (log Z - 1) < 2`, and **repels** them when `log Z < 1`: the round-off in `sum p` (about
1e-16 from the uniform start) grows by the factor `1 - dt T (log Z - 1)` each step, 1.243 here
(`log Z = -0.619`), and the state goes to the other rest point of the same step, the one where
`<p, xi> = 0`, which is `exp(v/T - 1)`. The drift is **transverse** to the simplex: a Shahshahani
step started exactly on it stays on it, so in exact arithmetic, or with the state space the simplex
itself, the chooser reaches the logit. What is refuted is the prediction about the construction as
built, state in `R^n` in floating point. Item 28's two examples had `log Z = 1.31` and `2.98`, so
they never met this. Active inference's values are `-G` with `G` an expected free energy, usually
positive, so `log Z < 1` is its **normal** case, not an edge. Shifting every value by one constant
leaves the logit unchanged and changes `log Z` by `c/T`: with the state in `R^n`, whether the
chooser keeps its normalization depends on a constant the choice itself cannot see. So item 28's
"the normalization is the response's, not the cost's" holds on the simplex and in exact
arithmetic, and needs the condition `0 < dt T (log Z - 1) < 2` for the construction in `R^n`.

## 5. Addendum A: predictions written after (ii)'s refutation, before any further code

These are new predictions about the mechanism and about a changed construction; they do not
revise (ii-a) or (ii-c), which stay refuted.

- **(A1)** For the flat case (`v = -G`, `T = 0.5`, `dt = 0.3`), `|1 - sum p_n|` grows by the factor
  `1.2428178386074` per step (measured over steps where it lies between 1e-12 and 1e-3, to 1%),
  and the final state equals `exp(v/T - 1)` to relative error below 1e-10.
- **(A2)** The **shifted chooser**, values `v + c0` with `c0 = 3` (the same logit), has factor
  `0.34282`, and settles at `q` to below 1e-12 in both the flat and the habit case, and opened
  after the stateless publisher (the shift a constant `c0 <1, p>` added to its potential, which has
  no component on the incoming `G`).
- **(A3)** Test (iii) runs as written first, with the unshifted chooser: (iii-a) and (iii-b) do not
  use the chooser and are unaffected. (iii-c) absolute shares and (iii-d) numbers will be
  refuted as written by the same drift, but (iii-c)'s share **ratio** is 1 at the switch to 1e-9,
  since the drift keeps `p` proportional to the logit. Then with the shifted chooser (`c0 = 2`),
  (iii-c) and (iii-d) hold with their numbers as written, because the shift adds no forcing on the
  incoming `G`, so the pullback onto the belief is unchanged.

## 6. Verdicts, test (iii) and Addendum A

**(A1) Held.** The factor was within 1% of 1.2428178386074 at every step in the window; the final
state equals `exp(v/T - 1)` below 1e-10. **(A2) Held**, flat, habit and opened, below 1e-12.

**(iii-a) Held.** `G` from the part that calls `Phi(world)` and its Jacobian equals the hand
closed form below 1e-12 at every point named, horizons 1 and 2.
**(iii-b) Held**, below 1e-12: risk plus ambiguity is the expected cost less
`1/2 log(1 + w_obs / w_pred)`.
**(iii-c) Refuted as written** with the unshifted chooser: at `Sigma = 0.2` it settled on
`(1.73e-3, 1.07e-2, 7.9e-8)`, summing to 0.0124, so every share is wrong; the ratio at the switch
was `0.9999999999999991`, as (A3) said. **With the shifted chooser (`c0 = 2`) it held**:
`(0.13950788, 0.86048578, 6.3e-6)` at 0.2, `(0.71147231, 0.28849539, 3.2e-5)` at 0.02, ratio 1 at
the switch to 1.2e-15.
**(iii-d) Refuted as written** unshifted: the exact composite settled at `m = -0.48277` with shares
summing to 0.036, so the bias toward the preference was 0.017 rather than 0.384 (the pullback
is `-sum p dG/dm`, and the shares had shrunk). **Shifted, it held**: exact wiring, `m* =
-0.11620980279789736` and `p*` as written; unilateral, `m* = -0.49999999999999994` and `p*` as
written.
**(A3) Held but for one clause.** It said all of (iii-d)'s numbers would fail unshifted; the
unilateral belief did not (`m = -0.5` to 6e-17), since a unilateral chooser pulls nothing back and
the belief answers only to its evidence.

## 7. Reasonable or weird, and at which scale (the agent's judgment, not evidence)

- **Precision by descent** (i-a, i-b′): reasonable at every scale — a receptor adapting its
  gain, a person learning how noisy a gauge is, a town learning how far to trust a report. The
  learned precision is the divide-by-`N` one, so a learner with few data overtrusts by `N/(N-1)`,
  which is reasonable too.
- **Without the normalizer, attention goes to zero** (i-b): reasonable and worth saying, as a
  mechanism, at the scale of a person or an institution: a part that pays only for error weighted
  by its attention does best by attending to nothing. The log-det term is what attending is
  worth. Weird only in its pace: the collapse is harmonic, `Pi ~ 2/(eta S n)`, never finished.
- **Trust by agreement** (i-c): weird, and important. A source's learned precision is
  `1/mean((x_s - mu)^2)`, its agreement with the learner's *current* mean, not a property of the
  source measured against the world. With two honest sources it works (0.94 on the tight one). In a
  society the same formula trusts whoever agrees with what one already believes. Nothing in the
  construction tells a reliable source from a conforming one; only ground truth would.
- **The chooser's normalization depends on a constant it cannot see** (§4; with the state in
  `R^n`, not on the simplex): weird at every scale as a statement about *choice*: adding the same amount to every option's value should change
  nothing, and here it decides whether the shares sum to one. It is reasonable read as a
  *population* (unnormalized abundances): with every option a cost (`log Z < 1`), the total shrinks
  and settles at `Z/e`. The construction's state space is all of `R^n`, and the simplex is only
  invariant; that is the cause. For the book's sentence (Ch 1 §1.5, a split of lunches between two
cafés resting at the softmax spread) nothing changes: a split lives on the simplex.
- **Look while unsure, push once sure** (iii-c): reasonable for a person (check, then act). But the
  setting was chosen to have a switch: with process noise `Q = 0.1`, at both `R_dark = 2` and `0.3`,
  the precise policy won at every belief width, because the observation noise `R` enters the risk
  and the ambiguity both and the ambiguity does not depend on the belief. An agent run on expected
  free energy can prefer looking forever, which is weird for a person. At horizon 2 the best sequence was push-then-look at both
  widths, not look-then-push, since the preference is scored at every step.
- **Exact wiring is wishful thinking** (iii-d): reasonable, and important, for a person (motivated
  reasoning) and an institution (a plan's feasibility estimate bent toward the plan); no cell
  reading. It is item 30's retroactivity: an exact reader of `G` tilts what `G` is computed from.
  Active inference's own separation — beliefs about the present answer to evidence, policies to
  `G` — is the **unilateral**, non-exact wiring, so active inference is not one cost descended by
  every part; it is a composite whose forcing is not exact.

## 8. What this does not establish

A closed loop: an agent that acts on a world, samples what it sees, updates its belief, and plans
again. Everything here is one planning step with Gaussians carried as parameters; stochastic
observations (`rand`, book Ch 6) were not used, and the precision learner of (i) was not wired to
the planner of (iii). The three pieces are each an arrangement and they compose; whether the loop
they would make behaves like an active-inference agent is not tested.

## 9. Addenda B and C: predictions written after David's questions of 2026-09-29, before code

David: "Is there a way to fix ii? … is there a better way to do it?"; and "my prediction is held in
state and … the difference between the prediction and the input is the potential and that
potential then drives the … outside thing to be more like what I want. Is that how you modeled
it also?" — it was not: (iii) sent `G` forward as data and simulated the world rather than
pushing it. Addendum C builds his picture.

**B. A chooser whose every state is a distribution.** State the gauge-fixed natural parameter
`lambda ∈ R^{n-1}`, `p = softmax(lambda, 0)` (item 28's dual chart), response the inverse Fisher
metric (`choice.inverse_fisher`), exact forcing `-dk` for the same policy free energy
`k = <G, p> - T <log E, p> - T H(p)`. Since `log p_i - log p_n = lambda_i`, the step is
`lambda' = (1 - dt T) lambda - dt (Delta G - T Delta log E)`, `Delta` meaning `(.)_i - (.)_n`.
- **(B1)** From `lambda = 0` it settles at `q` of (ii-a), flat and habit, to 1e-12 at `dt = 0.3`;
  at `dt = 1/T` it lands on `q` in **one** step (1e-12); `sum p = 1` to 1e-15 at every step,
  there being no direction off the simplex; adding 7 to every `G` changes the trajectory by
  less than 1e-12.
- **(B2)** Opened (receiving `G`) and composed after the planner of (iii-d), with no shift:
  exact, `m* = -0.11620980279789729` and `p*` as in (iii-d); unilateral, `m* = -0.5` and its `p*`;
  to 1e-9.

**C. David's picture, nested: a prediction discharged by the level below.** Three boxes chained
by `compose_seq`, each publishing its state upward and nothing else: the **hand** (state `h`,
Euclidean response, its own potential `-1/2 kappa h^2`, rest at 0), **level 1** (state `mu_1`, its
prediction of the hand, cost `1/2 Pi_1 (h - mu_1)^2` on the incoming `h`), **level 2** (state
`mu_2`, its prediction of level 1, cost `1/2 Pi_2 (mu_1 - mu_2)^2 + 1/2 pi_g (mu_2 - g)^2`, the
last term what it wants). No box sends the hand a command; what reaches it is only level 1's
forcing on its incoming `h`, pulled back (the one-form out the back). `Pi_1 = Pi_2 = pi_g = 4`,
`g = 1`, from `h = mu_1 = mu_2 = 0`.
- **(C1) The goal reaches the hand.** At `kappa = 1` the composite settles at
  `(h, mu_1, mu_2) = (0.5714285714285715, 0.7142857142857144, 0.8571428571428572)` (1e-10): a chain
  of springs, `h* = g k_s/(kappa + k_s)`, `1/k_s = 1/Pi_1 + 1/Pi_2 + 1/pi_g`, ordered
  `0 < h* < mu_1* < mu_2* < g`.
- **(C2) Perception or action is set by how stiff the world is.** `kappa = 100`:
  `(0.01315789473684211, 0.3421052631578948, 0.6710526315789473)`, the hand barely moves and the
  predictions give way; `kappa = 0.01`: `(0.9925558312655088, 0.9950372208436725,
  0.9975186104218362)`, the hand nearly reaches the goal (1e-10).
- **(C3) Each level must push the one below.** Level 1 made unilateral (reads `h`, pushes
  nothing): the hand stays at 0 and `(mu_1, mu_2) = (1/3, 2/3)` — prediction without discharge is
  perception only. Level 2 unilateral: `(h, mu_1, mu_2) = (0, 0, 0.5)` — the goal never leaves
  the top. (1e-10.)
- **(C4) The hand never quite arrives, and the predictions are wrong at rest.** For every
  `kappa > 0`, `h* < g` and `mu_1* > h*`: level 1 believes the hand nearer the goal than it is.
  That is the wishful tilt of (iii-d), here the mechanism that carries the goal down. Raising
  every precision a hundredfold at `kappa = 1` gives the same rest as `kappa = 0.01` (only the
  ratio counts).

Not built: noise, and a level that integrates its error (which would remove the offset).

## 10. Verdicts, addenda B and C (all held; 25 passed, 6 strict xfails)

**(B1) Held**: `q` to 1e-12 flat and habit; one step at `dt = 1/T` lands on it; `sum p = 1` to
1e-15 throughout; `G + 7` gives the same trajectory to 1e-12. **(B2) Held**, unshifted: exact
`m* = -0.1162098027978973…`, unilateral `-0.5`, shares as predicted. So test (ii)'s refutation
belonged to the chart, and the natural chart is the construction to use.
**(C1)–(C4) Held**, every rest to 1e-10: `(0.5714, 0.7143, 0.8571)` at `kappa = 1`; stiff world,
predictions give way; soft world, hand nearly at the goal; a unilateral level stops the goal at
that level; precisions ×100 equal a world ×1/100.

*Reasonable or weird* (the agent's judgment). Reasonable for a hand and a reflex arc, and for an
institution carrying a goal down through ranks: the goal travels by each level's cost on what it
receives, with no command sent down, and a level that only reads stops it. Weird for a hand: it
never arrives (`h* = 0.571` of the way at equal stiffness), since the exact hierarchy is a chain of
springs, a proportional controller with its offset; a level that integrates its error would remove
it and was not built. Weird and worth saying: at rest every level's prediction is wrong in the
wanted direction (`mu_1* > h*`), the tilt of (iii-d), which here is the mechanism and not a fault.
