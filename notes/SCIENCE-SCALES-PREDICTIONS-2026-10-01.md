# Science at two scales: predictions, written before any code (2026-10-01)

David, 2026-10-01: "do you think there could be tests that could run about scientific method as a
joint practice and as hypothesis selection being 2-scales of the same thing? like ideas hold together
like organisms do, and that if they're not fed by the outside world, they fall apart." Then, on what
feeds an idea: "no one will keep an idea in mind---to costly---unless the evidence confirms it, or
unless they can account for that failure some other way, which is work." And, on new hypotheses:
"bayes only works on things with a fixed probability space. but real hypotheses are often outside the
'box' … if robots started needing a new kind of token … the new stuff just wouldn't be in your current
model." To "shall I run tests (a)–(d)": "yes please."

Tests proposed and predictions written by Claude Opus 5.5, 0fe746c1, **before any construction or
test code**. The run fills in §5 onward; it does not edit §§1–4 except to add, in a marked
"Numbers fixed before code" paragraph per test, values computed from the closed forms below by a
scratch script that builds no arrangement. A refuted prediction is reported as refuted and never
repaired (`AGENTDOCPLAN.md` §5.9 practice). Existing results this rests on: one step of the
inverse-Fisher belief at duration 1 is Bayes' rule exactly (`notes/BAYES-INTEGRATOR.md`,
`choice.belief_arrangement`); the chooser is the replicator descent with an entropy term
(`notes/DISCRETE-CHOICE.md`); precision learned by descent with the `-1/2 log Pi` term
(`notes/ACTIVE-INFERENCE-TESTS.md` §1).

Common setting. Hypotheses `h_1 … h_n`, each a predictive distribution `p(x | h_i)`. Data from a
fixed seed. Belief in the gauge-fixed natural chart `lam in R^{n-1}`, `q = softmax(lam, 0)`.

## 1. Test (a): one mind and a population are one arrangement drawn two ways

**Mind.** `choice.belief_arrangement`'s construction, stepped once per datum at duration 1 with
evidence `t_i = log p(x|h_i) - log p(x|h_n)` (an opened version receiving `t` at a port, or a closed
one rebuilt per datum with `init` the current `lam`; say which).

**Population.** `n` holder boxes, holder `i` carrying one hypothesis, state `y_i = log N_i` (log
abundance), Euclidean response at duration 1, exact forcing `d<log p(x|h_i), y_i>` (a constant
one-form: growth by fitness `p(x|h_i)`, the discrete-time replicator in the log chart). Composed as
`n` separate boxes in one arrangement, and also as one `n`-dimensional box.

**Predictions.** (a1) At every step, `softmax(lam, 0) = softmax(y)` to round-off (< 1e-12),
with `lam_i = y_i - y_n`, for a data stream of at least 200 draws and `n >= 3`. (a2) The
composite of `n` one-dimensional holder boxes and the single `n`-dimensional box give the same
trajectory to round-off (clustering is free). (a3) Misspecified world (no `h_i` true; e.g.
Gaussian hypotheses with means on a grid, truth off the grid): both readings concentrate on the
hypothesis minimizing `KL(truth || h_i)`, the same one, with weight > 0.99 by step 200.
Mechanism: multiplying a weight by a likelihood and multiplying an abundance by a fitness are one
operation; normalization is the finite population. This is a known identity (Harper 2009,
"The replicator equation as an inference dynamic," arXiv:0911.1763, **not yet checked**; do not
cite until opened). It is not too good to be true; the test checks the library realizes it.
**What it cannot show:** that organisms *are* hypotheses. Fitness here is defined as a likelihood;
the reading of a real population is David's candidate ("an organism is a hypothesis the world
tests," his "sure"), an analogy unless something more is shown.

**Numbers fixed before code** (the run's, Claude subagent of 0fe746c1, 2026-10-01; from a scratch
script that sums log-likelihoods and normalizes, building no arrangement). Hypotheses `h_i = N(mu_i, 1)`,
`mu in {0, 1, 2, 3, 4}` (`n = 5`, gauge reference `h_n` at `mu = 4`); 200 draws from
`numpy.random.default_rng(0)`. Two streams: **misspecified**, truth `N(2.3, 1)` (no `h_i` true;
`KL(truth || h_i) = (2.3 - mu_i)^2 / 2`, least at `mu = 2`), used for (a1), (a2), (a3); and
**well specified**, truth `N(2, 1)` = `h_3`, used for (a1), (a2) as a second stream. Mind: the closed
`belief_arrangement` rebuilt per datum with `T = 1`, `v = log p(x | h_i)`, `init` the current `lam`.
Closed forms: misspecified stream `xbar = 2.3153`; the `mu = 2` weight first exceeds 0.99 at step 13;
at step 200, `1 - q_max = e^{-36.9}`. Well-specified stream: first > 0.99 at step 11; `1 - q_max =
e^{-96.9}` at 200. **A risk the prediction did not consider, stated before code:** the winner (`mu = 2`)
is not the gauge reference, so the mind's response `G^{-1}` is computed from a `G` whose winner
diagonal `mu_i(1 - mu_i)` underflows relative to 1 once `1 - q_max < 1e-16`; that happens at 2 of 200
steps (from 199) in the misspecified stream and 106 of 200 (from 94) in the well-specified one. If
(a1) fails, this is where it is expected to, and the failure would be the inverse-Fisher response's
in floating point, not the identity's.

## 2. Test (b): keeping an idea costs; confirmation pays for it, or explaining away does

**Setting.** A holder chooses between **hold** `h` ("the quantity both sources measure is 0") at
upkeep `c` per step, and **drop**, a default that predicts every datum uniformly on `[-L, L]`
(cost `log 2L` per datum, nothing to learn, so it has nothing to discount). Source A gives `n`
draws `N(0, 1)` (confirms `h`); source B gives `n` draws `N(d, 1)` (disconfirms it when `d != 0`).
Holding predicts A by `N(0, 1)` and B by `N(0, 1/Pi_B)`. The choice is `choice.py`'s chooser on
`{hold, drop}` at temperature `T`, with expected cost `a K_hold(ell) + (1-a) K_drop + c a - T H(a)`,
`ell = log Pi_B` descended jointly (its force scaled by `a`).

Per-datum mean costs, constants included (`S_A`, `S_B` the mean squares of the draws about 0):
`K_drop = 2 log 2L`; `K_hold = 1/2 S_A + 1/2 Pi_B S_B - 1/2 log Pi_B + log 2pi`.
- **Fixed trust** (`Pi_B = 1`): penalty of B is `1/2 S_B`, quadratic in `d`.
- **Learned trust** (`ell` descended): at rest `Pi_B = 1/S_B`, penalty `1/2 + 1/2 log S_B`,
  logarithmic in `d`. The `-1/2 log Pi_B` term is the cost of discounting the source: David's
  "account for that failure some other way, which is work" (reading *proposed*).

**Predictions.** (b1) *Sanity, built in, not a finding:* a hypothesis that changes no prediction
(`K_hold = K_drop` before upkeep) has hold share `1/(1+e^{c/T})` at rest, which falls to 0 as
`c/T` grows, and stays at 1/2 when `c = 0`. (b2) Hold share at rest equals
`1/(1 + exp(-(K_drop - K_hold* - c)/T))` (with `ell` at its optimum in the learned arm), to 1e-8.
(b3) The disconfirmation the holder survives: hold beats drop iff `S_B < S*` with
`S*_fixed = 2(K_drop - log 2pi - c) - S_A` and `S*_learned = exp(S*_fixed - 1)`: **exponentially
more disconfirmation survives when trust in the critic can be learned.** Choose `L, c, n` so both
thresholds fall inside `d in [0, L/2]` and say the values before code. (b4) With learned trust,
what drops `h` is a lack of confirmation, not disconfirmation: lowering A's confirmation (raising
`S_A` by moving A's mean off 0) drops `h` at a rate set by `S_A` one-for-one in both arms, while
`d` moves the decision only through `1/2 log S_B`.
**Fair-comparison note:** the two arms differ only in whether `Pi_B` is descended; the default,
`L`, `c`, `T`, `n` and seeds are the same in both. Tally decisions over a grid of `d` each way.

**Numbers fixed before code** (the run's, Claude subagent of 0fe746c1, 2026-10-01; scratch script,
no arrangement built). `L = 10`, `c = 1.6`, `T = 1`, `n = 200` draws per source; noise from
`default_rng(0)`, first 200 normals `z_A`, next 200 `z_B`; A's draws `m_A + z_A` (`m_A = 0` except in
(b4)), B's draws `d + z_B` with the SAME `z_B` at every `d` (common noise across the grid and both arms).
Grid `d in {0, 0.125, ..., 5}` (41 points, `L/2 = 5`). Largest `|draw|` 8.07 < `L`, so the default's
cost `log 2L` per datum is finite throughout. Closed forms: `K_drop = 5.991465`, `S_A = 0.924098`,
`S*_fixed = 4.183077` (at `d* = 1.858`), `S*_learned = 24.120864` (at `d* = 4.892`): both inside
`[0, 5]`. Predicted tally over the grid: fixed arm hold 15, drop 26; learned arm hold 40, drop 1. Closest
calls: `|K_drop - K_hold* - c| = 0.030` (fixed), `0.0033` (learned), both far above the 1e-8 of (b2).
Rest-share range: fixed `[2.8e-5, 0.827]`, learned `[0.495, 0.827]`.
**Construction choices.** The chooser is `choice.py`'s simplex chooser (`logit_arrangement`-style:
`replicator.shahshahani` response, entropy forcing), tensored with a Euclidean `ell` box (`eta = 1`,
`ell_0 = 0`, i.e. starting at the fixed arm's trust) in the learned arm; one closed box, one potential
`U = -[p_h (K_hold(ell) + c) + p_d K_drop] + T H(p) + B (p_h + p_d)`, so the `ell` force is scaled by `p_h`.
`dt = 0.08` (so `dt * max|K_drop - K_hold - c| < 1` and the first jump from uniform stays positive), and
`B = K_drop + 2T`: on the simplex a constant, but it puts `dt T (log Z - 1)` in `[0.08, 0.23]`, inside
the `(0, 2)` that `choice.py`'s CONDITION paragraph requires for the simplex to survive round-off
(without it `log Z < 1` and the chooser drifts to the Gibbs weights). Start `p = (1/2, 1/2)`; 3000
steps (chooser contracts by `1 - dt T = 0.92` per step near rest). Decision: hold iff share > 1/2.
(b1) run at `c in {0, 1, 3, 10}`, `T = 1`, `K_hold = K_drop`: shares `1/2, 0.2689, 0.04743, 4.54e-5`.
**(b4) as worded is too vague to hold or refute** ("drops `h` at a rate set by `S_A` one-for-one";
"lack of confirmation, not disconfirmation" is a reading, not a measurement). Sharpest version, stated
before running: **(b4')** in both arms the rested hold logit `log(a / (1-a))` moves by exactly
`-Delta S_A / 2T` when A's mean moves (the same in both arms, to 1e-8), while moving `d` moves it by
`-Delta S_B / 2T` (fixed) and `-Delta log S_B / 2T` (learned). Consequence in numbers: at `d = 0`, `h` is
dropped once `S_A > 4.048` (fixed) / `4.050` (learned), i.e. A's mean moved by `m_A* = 1.752` / `1.753`,
nearly the same in both arms; while with `m_A = 0` it takes `d* = 1.858` (fixed) and `4.892`
(learned). (b4') is implied by (b2) and checks no further machinery; its content is that comparison.

## 3. Test (c): holders who read each other and not the world

`m` holders on a ring (each hears its two neighbours; also run all-to-all), each with a belief
`lam_j` over the same `n` hypotheses, random starts from one seed per run. Each step holder `j`
may receive a private datum at weight `beta`, and its neighbours' accounts (their `q_k`, sent
forward; nothing pushed back) at weight `alpha`, in one of two costs (derive each forcing under
the inverse-Fisher response before code and write it here):
- **Agree:** `alpha sum_k W_jk KL(q_j || q_k)`; predicted step `lam_j += alpha sum_k W_jk (lam_k - lam_j)`.
- **Take as evidence:** the neighbour's log-belief treated as a datum's log-likelihood; predicted
  step `lam_j += alpha sum_k W_jk lam_k` (double counting).

**Predictions.** (c1) `beta = 0`, agree: consensus at the `W`-weighted average of the starting
`lam` (plain average for doubly stochastic `W`), at no more than the starting confidence: a
community fed only by itself agrees and learns nothing. (c2) `beta = 0`, evidence: consensus in
the direction of `W`'s leading eigenvector applied to the start, with `|lam|` growing like
`(1+alpha)^t`: sure of something no one observed. (c3) `beta > 0`, agree: every holder
concentrates on the true hypothesis (weight > 0.99) in 20 of 20 seeds. (c4) `beta > 0`, evidence:
over 20 seeds at `alpha/beta in {0, 0.1, 1, 10}`, the count locking onto a wrong hypothesis is 0 at
0 and rises with `alpha/beta`; at 10, more than half of the seeds whose starting average favoured a
wrong hypothesis lock onto it. Report counts each way at the same settings.

**Derivation, fixed before code** (the run's, Claude subagent of 0fe746c1, 2026-10-01). Holder `j`'s
belief in the gauge-fixed natural chart: `log q_{j,i} = lam_{j,i} - AA(lam_j)` (`lam_{j,n} = 0`),
`mu_j = grad AA(lam_j)`, `G_j = grad^2 AA(lam_j)`, response `G_j^{-1}`. Neighbours' `lam_k` arrive
forward and are constants for `j`'s cost (nothing pushed back).
- *Agree.* `KL(q_j || q_k) = <lam_j - lam_k, mu_j> - AA(lam_j) + AA(lam_k)`, so
  `d/dlam_j KL = mu_j + G_j (lam_j - lam_k) - mu_j = G_j (lam_j - lam_k)`. Forcing
  `-alpha sum_k W_jk G_j (lam_j - lam_k)`; step `G_j^{-1}` of it: `lam_j += alpha sum_k W_jk (lam_k - lam_j)`.
  **§3's predicted step is right.**
- *Take as evidence.* A datum with log-likelihood `log q_{k,i}` has gauge-fixed statistic
  `log q_{k,i} - log q_{k,n} = lam_{k,i}`; its cost `alpha W_jk E_{q_j}[-log q_k]` has forcing
  `alpha W_jk G_j lam_k` and step `alpha W_jk lam_k`. So `lam_j += alpha sum_k W_jk lam_k`.
  **§3's predicted step is right.** (KL differs from the cross-entropy by `-H(q_j)`, whose step is the
  `-alpha lam_j` that makes agreement a consensus instead of a sum.)
- *Private datum* at weight `beta`: step `beta t_j`, `t_{j,i} = log p(x_j|h_i) - log p(x_j|h_n)`, the
  full log-likelihood of one draw per step (totals, so beliefs sharpen with data).

**Realization, fixed before code.** Because of (a1), not through `inverse_fisher`: the beliefs live in
the gauge-free log-weight chart `y_j in R^n`, `q_j = softmax(y_j)`, Euclidean response `eta = 1`, where
each derived step is a translation. Agree: cost `alpha/2 sum_k W_jk |y_j - log q_k|^2` (not the KL; the
cost whose Euclidean step is the derived one, since `log q_k = y_k` up to the gauge `1`); evidence: the
linear cost `-alpha sum_k W_jk <log q_k, y_j>`; datum: `-beta <log p(x_j|h), y_j>`. One closed
arrangement on `Y in R^{m x n}` whose one-form is the block of each holder's own cost differential (a
game's unilateral forcing, not exact), stepped synchronously.

**Numbers fixed before code.** `n = 5`, `h_i = N(mu_i, 1)`, `mu in {0,...,4}`, truth `h_3` (`mu = 2`);
`m = 10` holders; ring `W_jk = 1/2` for `k = j +- 1`, all-to-all `W_jk = 1/9`, both doubly stochastic.
Seed `s in 0..19`: `rng = default_rng(s)`, `Y_0 = rng.normal(size=(m, n))`, then data
`rng.normal(2, 1, size=(T, m))`; the same seeds in every arm, wiring and `alpha`. The starting average
`mean_j lam_{j,0}` favours a wrong hypothesis in **17 of 20** seeds (favourites by seed: 1,3,0,0,3,3,1,4,
0,3,3,2,4,0,0,2,3,2,4,4). Eigenvalues: ring `W` in `[-1, 0.809]`, all-to-all `{1, -1/9}`.
(c1) `alpha = 0.5`, `T = 400` (ring's slowest mode `0.9045^400 = 4e-18`): all `lam_j` within 1e-8 of
`mean_j lam_{j,0}`; "no more than the starting confidence" = consensus `max_i q_i <=` the most confident
holder's starting `max_i q_{j,i}`. (c2) `alpha = 1`, `T = 100`: `lam_j(T) / 2^T` within relative 1e-3 of
`mean lam_0` (ring's second mode `0.9045^100 = 4.4e-5`), `|lam(T)| / |lam(T-1)| = 2` to 1e-6, consensus
`max q > 0.99`. (c3), (c4): `beta = 1`, `alpha in {0, 0.1, 1, 10}`, `T = 100`. **Lock onto a wrong
hypothesis** = at step 100, every holder's weight on one and the same wrong hypothesis exceeds 0.99.
"Rises with `alpha/beta`" = non-decreasing over the four values and larger at 10 than at 0; "more than
half" = at least 9 of the 17 seeds whose start favours a wrong hypothesis. **Risks stated before code:**
the synchronous agree step contracts disagreement by `1 - alpha(1 - w)`, which leaves `(-1, 1]` when
`alpha (1 - w_min) >= 2`: at `alpha = 1` the ring's alternating mode has eigenvalue exactly `-1` (it
never decays, and the data keep feeding it), and at `alpha = 10` both wirings diverge (`-19` ring,
`-10.1` all-to-all). I predict (c3) holds at `alpha = 0, 0.1` in both wirings and at `alpha = 1`
all-to-all; at `alpha = 1` ring it may fail; at `alpha = 10` it fails in both. This is the discrete-time
arrangement's stability, not the cost's; (c3) as written did not fix `alpha`, so it is tested at all four.

## 4. Test (d): nothing new without variation, at both scales

Hypotheses: Gaussians `N(mu, 1)`, `mu` on a grid `0, 0.5, …, 5`; truth `N(3.7, 1)` (KL-closest
grid point 3.5). Start: weight only on `mu in {0, 0.5, 1}`, exactly zero elsewhere (population:
abundance 0; mind: belief 0, i.e. outside its support).

**Predictions.** (d1) No variation: both readings settle on `mu = 1` forever; `3.5` never gains
weight. (d2) Mutation at rate `eps` (each step a fraction `eps` of each type's abundance moves to
its grid neighbours, half each side, then selection): the population reaches a peak at 3.5;
stationary spread increases with `eps`. (d3) The same update on the mind is Bayes' filter with
the assumption that the truth may drift by one grid step (transition matrix `M`): identical
trajectories to round-off when mutation and selection are applied in the same order. (d4) **No
prediction** on whether the mutation step is an exact forcing of the book's kind in some chart;
the run reports what it finds.
**Scope stated before the run:** (d) concerns only David's first kind of novelty, a hypothesis
inside the space at vanishing weight ("10^-{100^100} … like 10^{-4}"). His second kind, a new token,
is a new port; the library has no changeable interfaces, and nothing here tests it.

**Numbers fixed before code** (the run's, Claude subagent of 0fe746c1, 2026-10-01; a scratch numpy
script iterating the closed-form recursion, building no arrangement). **Charts:** population in the
abundance chart `N in R^11` (an exact zero is representable), mind in the probability chart `p`; with
totals, one full likelihood per datum. Grid `mu = 0, 0.5, ..., 5` (11 types), `mu = 3.5` at index 7;
truth `N(3.7, 1)`; 120 draws per seed, seeds 0-4 (seed 0 primary). Start `N = p = (1/3, 1/3, 1/3, 0, ..., 0)`.
**Mutation** `M`: each type sends `eps/2` to each grid neighbour; at the two ends the half that would
leave the grid stays (reflecting), so `M` is symmetric and column-stochastic. `eps in {0, 1e-3, 1e-2,
1e-1}` (0 is (d1)). **Step:** population `N' = D_l (M N)` with relative fitness `l_i = p(x|h_i) / max_k
p(x|h_k)`, a constant per datum, chosen so the total does not underflow in 120 steps (smallest log-total
-518 at `eps = 0`); mind `p' = D_L M p / (1^T D_L M p)`. (d3) is stated as: the normalized population
`N / sum N` equals the mind's `p` at every step, `max_t max_i |.| < 1e-12`, for every `eps`, and at
`eps = 0` both have weight exactly 0 off `{0, 0.5, 1}` at every step.
**Definitions.** (d1) holds iff at `eps = 0`, in both readings, the largest weight is on `mu = 1` at
the final step and on every step of the second half, and `mu = 3.5` has weight exactly 0 at every
step. (d2) holds iff for every `eps > 0` the population's largest weight is on 3.5 at some step, and
the spread (standard deviation of `mu` under the normalized weights, averaged over the last 40 steps)
increases strictly with `eps`. Reported per `eps`: the first step 3.5 carries the largest weight, the
largest-weight type at step 120, the fraction of the second half with the largest weight on 3.5, and the
final spread.
**Closed-form values, seed 0:** `eps = 1e-3, 1e-2, 1e-1`: 3.5 first carries the largest weight at steps
39, 25, 20; at step 120 the largest weight is on `mu = 4.0` in all three (weight on 3.5: 0.431, 0.266,
0.173); second-half fraction at 3.5: 0.50, 0.18, 0.47; spread 0.2246, 0.2282, 0.3773. `eps = 0`: largest
weight on 1.0, weight on 3.5 exactly 0. Seeds 1-4, first steps at 3.5: (34, 23, 12), (27, 22, 11),
(27, 22, 11), (41, 25, 12); at step 120 the mode is 3.5 in 11 of those 12 runs (4.0 at seed 2,
`eps = 0.1`); spread increases with `eps` at every seed. Seed 0's step-120 mode at 4.0 is the data:
the truth 3.7 is near the 3.5/4.0 midpoint 3.75, and a filter that forgets (mutation) follows the
recent sample mean. So "a peak at 3.5" is a peak the population reaches, not one it keeps.
**Realization plan, and (d4)'s question fixed before code.** Population: `configuration` with the
constant response `diag(l)` and the exact forcing `d(1/2 N^T (M - D_l^{-1}) N)`. Then
`N + D_l (M - D_l^{-1}) N = D_l M N`: mutate-then-select is one step with an exact forcing, because `M`
is symmetric. The response carries the datum, so it is rebuilt per datum, as in (a). This is cheap:
any linear step `A` with `A - I = R S`, `R` positive definite and `S` symmetric, has such a realization,
which is every `A - I` diagonalizable with real eigenvalues (STATUS §2, "Not evidence at all"). Mind:
I found no exact forcing in the probability chart. The factorization `R = D_L / Z` gives a forcing
whose Jacobian `M - D_L^{-1} p (M L)^T - Z D_L^{-1}` is not symmetric, so the mind is realized
directly: Euclidean response, forcing `F(p) - p`, not exact.

## 5. Results (the run's; one subsection per test, written as each finishes)

### 5.1 Test (a)

Run 2026-10-01 by Claude (subagent of Claude Opus 5.5, 0fe746c1). Code: `arrangements/science.py`,
`tests/test_science_scales.py` (3 passed, 3 strict xfail, 15 s). Settings as fixed above. The mind is
the **closed** `belief_arrangement` rebuilt per datum with `init` the current `lam`.

**(a1) Refuted, on both streams.** Misspecified: the mind is NaN from step 110 to 200, so
`max_t |softmax(lam,0) - softmax(y)|` is NaN. Well specified: `2.2e-4`, and
`max_t |lam - (y_i - y_n)| = 207`. The first step with a gap above `1e-12` is 13 (misspecified) and 11
(well specified). **[The cause given next was corrected 2026-10-03; see the correction after this paragraph.]**
**Cause, measured:** starting the library's mind step from the *exact* belief (the
population's `y_i - y_n`), its error against the translation `lam + t` is `2e-16` at step 0, `1.6e-13`
at step 5, `3e-11` at step 10 and `0.29` at step 20 (misspecified), tracking `cond(G)`: `5`, `4e5`,
`5e11`, `2e16`. `cond(G)` grows like `1 / min_i q_i`, the weight of the *least* favoured hypothesis
(`mu = 0`, at `1e-13` by step 13), and the step computes `G^{-1}(G t)` with a dense inverse, so the
exact cancellation `G^{-1} G = I` is lost in float64 within about ten draws; the lost digits sit in the
discarded hypotheses' coordinates, leak into the rest, compound, and in the misspecified run reach
`|lam| ~ 2e4` at step 109 and NaN at 110. **The risk stated before code named the wrong mechanism**:
it blamed the winner's diagonal underflowing (`1 - q_max < 1e-16`, from step 94 or 199); the failure
comes far earlier, from the losers' weights, and the misspecified stream fails although its winner
never gets that close to 1. The identity itself is not in question: Bayes' translation and the log
replicator step add the same `t` (the analytic step), and the closed-form numbers above match the
population exactly. **What holds instead:** the mind realizes the population to round-off only while
every hypothesis keeps weight above about `1e-5` (one-step error `<= 1.6e-13` while `cond(G) < 1e10`
on the misspecified stream; `6e-6` already at that condition on the well-specified one). A data stream
long enough to make a hypothesis negligible is exactly what this response cannot carry.

**Correction, 2026-10-03** (Claude Fable 5.1, 4f679514; `notes/BAYES-EVERY-BELIEF.md` §3, verdicts 6 and 10;
`tests/test_bayes_every_belief.py::test_prediction_10_the_science_scales_stream`). The numbers above stand; the
cause does not. The loss is the gauge's **reference value** (`mu = 4`) going thin, not the dense inverse, not
`cond(G)`, and not the response: in the gauge the response adds `(sum_{i<n} eta_i) / q_n` to every coordinate, so
the round-off of that sum, about `1e-16`, is divided by the reference's height. The one-step errors above,
`1.6e-13` and `3e-11` at steps 5 and 10, are about `1e-16 / q_n` there, not `1e-16 cond(G)` (`4e-11`, `5e-5`). On
this same misspecified stream: the gauge box with the winner `mu = 2` as its reference tracks the directly
accumulated log-posterior to `4.6e-13` over all 200 draws, dense inverse and closed form alike; the box on all
five log-heights with the response `diag(1/q)` (`belief.log_height_response`) tracks it to `1.7e-13`; and with a
thick reference the dense inverse is exact with other heights down to `1e-300`. So "a data stream long enough to
make a hypothesis negligible is exactly what this response cannot carry" is false of the response: it is true of
a gauge whose reference value is one of the negligible hypotheses. (a1) as run, with `science.mind_step` as built,
stays refuted; it has not been rerun with the other box.

**(a2) Held.** The tensor of five one-dimensional holder boxes and the single five-dimensional box
give the same trajectory exactly: `max_t |y_tensor - y_single| = 0.0` on both streams.

**(a3) Held for the population, refuted for the mind.** Population: concentrates on `mu = 2`, the
`KL(truth || h_i)` minimizer; weight first above 0.99 at step 13, and `1 - q = e^{-36.9}` at step 200
(prints as 1.0). Mind: the same hypothesis, above 0.99 from step 13 (dipping with the population at
steps 50-51), until step 109, where its belief has left the population's; NaN from step 110, so at step
200 it concentrates on nothing. Same cause as (a1).

**Reasonable or weird** (the run's judgment, not evidence). (a2) is reasonable at every scale and
guaranteed by construction (`parallel_arrangements` hands each factor its own slice), so it is a check
of the library, not a finding. (a3)'s population reading is reasonable for a population of organisms or
of cells under selection on a fixed environment: the type best matched to the environment, not a
"true" type, takes over, and its share approaches 1 geometrically. Weird at every scale: the losers are
never gone. A share of `e^{-406}` (the `mu = 0` type at step 200) is a population of fewer than one
individual, and a credence of `e^{-37}` is not a belief a person or a scientific community holds; the
construction keeps them because the log chart has no floor and no integer counts (no drift, no
extinction). The (a1) failure is that same weirdness turned into an error: the mind's response insists
on carrying every vanishing hypothesis at full relative precision, which no finite reader can do. Read
at the scale of a person or a community, that the bookkeeping of negligible alternatives is what
breaks first is plausible; read as a claim about cells or organisms, it says nothing, since the
population drawing carries the same numbers without trouble.

**What this does not establish.** That a mind and a population are one arrangement *in the library as
built*: they are one update on paper, and the population drawing realizes it, but the library's
inverse-Fisher mind does not past about ten informative draws, so (a1) cannot be cited as executable
evidence of the identity. Nor that organisms are hypotheses (fitness was defined as a likelihood here).
The repair would be in `choice.inverse_fisher` / `belief_arrangement` (a closed-form or better-gauged
`G^{-1}`), a library change this run did not make.

### 5.2 Test (b)

Run 2026-10-01 by Claude (subagent of Claude Opus 5.5, 0fe746c1). Code: `science.holder_chooser`
(`choice.py`'s Shahshahani simplex chooser, checked before building to be the `logit_arrangement`
construction and not `inverse_fisher`; tensored with a Euclidean `ell` box in the learned arm);
`tests/test_science_scales.py`, 6 tests for (b), all passed (whole file 9 passed, 3 strict xfail from
(a), 20 s). Settings as fixed above.

**(b1) Held.** Hold share at rest for `c/T = 0, 1, 3, 10`: `0.5, 0.2689414214, 0.04742587318,
4.53978687e-05`, matching `1/(1+e^{c/T})` to under 1e-8. Built in, as the prediction says.

**(b2) Held.** Over all 41 values of `d`: `max |share - closed form| = 5.6e-16` (fixed arm), `6.7e-16`
(learned arm); in the learned arm `ell` settles on `log(1/S_B)` to `8.9e-16`.

**(b3) Held.** `S*_fixed = 4.183077`, `S*_learned = 24.120864`. Tally over the same 41 values of
`d in [0, 5]`, same noise, same `L, c, T, n`: **fixed trust held `h` at 15 and dropped it at 26**
(largest `d` held 1.75); **learned trust held it at 40 and dropped it at 1** (largest `d` held 4.875).
The decision matched the rule `S_B < S*` at all 82 points, 0 mismatches.

**(b4) Not testable as worded** (said before running, §2). **(b4') Held:** moving A's mean changes
the rested hold logit by `-Delta S_A / 2T` to `1.6e-15` (fixed) and `3.8e-15` (learned); moving `d`
changes it by `-Delta S_B / 2T` (fixed, `5.1e-15`) and by `-Delta log S_B / 2T` (learned, `5.6e-16`).
In numbers: with no disconfirmation (`d = 0`), moving A's mean past `1.752` (fixed) or `1.753` (learned) drops `h`, nearly the same in both arms (closed form, §2); with
full confirmation (`m_A = 0`), the learned arm keeps `h` until B's mean is `4.89` away.

**Which of these could have come out otherwise** (STATUS §2). Only the machinery: that the simplex
chooser plus the `ell` box actually rests at the logit with `ell` at its optimum (b2), here to 1e-15
with the `B` shift. Given (b2), (b1), (b3) and (b4') are algebra on the closed forms, already in §2
before code; the run confirms the arrangement realizes them, not that they are true of anything.

**Reasonable or weird** (the run's judgment, not evidence). *A person's beliefs:* reasonable that an
idea is kept while its confirmations pay its upkeep, and that discounting the critic costs only
`1/2 log` of the disagreement; weird that the only rival is a know-nothing default. No hypothesis "the
quantity is `d`" is on offer, so a source five standard deviations off with 200 draws is absorbed as
noise rather than read as a rival: the learned arm explains away a *bias* with a *variance*.
*A scientific community:* the same move is the auxiliary-hypothesis rescue (blame the instrument),
reasonable as a description of what is cheap; weird as a norm, for the same missing-rival reason.
*Weird at every scale:* the costs are per-datum means at `T = 1`, so the hold share does not sharpen
with `n`; 200 draws or 2 give the same share for the same mean squares. *A population:* upkeep reads as a
trait's metabolic cost, and (b1) as an idle trait lost at rate `c/T`, reasonable; learned trust has no
clear population reading (what would a lineage discount?), so (b3) does not transfer. *A cell:*
learning `Pi_B` reads as sensory adaptation, turning down the gain of a sensor that keeps reporting
off-scale, reasonable; holding a set point against it is plausible homeostasis.

**What this does not establish.** That people or communities keep ideas this way; that the `-1/2 log
Pi_B` term *is* David's "account for that failure some other way, which is work" (reading still
*proposed*); or anything about a contest between `h` and a rival that predicts `d`, which this test
does not contain.

### 5.3 Test (c)

Run 2026-10-01 by Claude (subagent of Claude Opus 5.5, 0fe746c1). Code: `science.account_community`;
`tests/test_science_scales.py`, 14 tests for (c), 12 passed, 2 strict xfail (whole file 21 passed, 5
strict xfail, 39 s). **Realization used, and why:** not the inverse-Fisher chart, because (a1) showed its
dense inverse fails once a weight is small, and here weights go to `e^{-50}` and below. Both steps were
derived on paper under the inverse-Fisher response (§3, "Derivation"); §3's predicted steps were right.
The beliefs then live in the gauge-free log-weight chart with a Euclidean response, where those steps are
translations, and the run checks one step against the derived `lam` update (to `3e-16` agree, `9e-16`
evidence). In that chart the agree cost is the quadratic `alpha/2 sum W |y_j - log q_k|^2`, not the KL:
the same step, a different cost. Accounts are read through `stop_gradient` (forward only); the
one-form is each holder's own block, not exact. The settings are those fixed in §3, and they were the same
for every arm and wiring.

**(c1) Held.** `beta = 0`, agree, `alpha = 0.5`, step 400: every `lam_j` within `4.4e-15` (ring) and
`7.8e-16` (all-to-all) of `mean_j lam_{j,0}`, all 20 seeds; consensus no more confident than the most
confident start in 20 of 20 (both wirings).

**(c2) Held.** `beta = 0`, evidence, `alpha = 1`, step 100: `lam(100)/2^100` within relative `3.4e-4`
(ring) and `4.1e-15` (all-to-all) of `mean lam_0`; growth per step `2` to `1.5e-8`; every holder above
0.99 on one hypothesis in 20 of 20 seeds, with no datum read.

**(c3) Held at `alpha/beta = 0, 0.1, 1`; refuted at 10.** At `beta = 1`, `alpha/beta = 1`: ring 20 of 20
found the truth, and all-to-all 20 of 20. The risk I stated for the ring at 1 did not happen: the alternating
mode does not decay, but it stays small next to the accumulated data. At `alpha/beta = 10`: ring 0 of 20 and
all-to-all 0 of 20; all 40 runs ended "neither" (holders disagree, finite). This was predicted before code
as the synchronous step's instability (eigenvalues `-19`, `-10.1`), but it refutes (c3) as written,
which fixed no `alpha`.

**(c4) Held.** Counts locked onto a wrong hypothesis at step 100, `beta = 1`, at the same seeds and settings
in both arms:

| `alpha/beta` | 0 | 0.1 | 1 | 10 |
|---|---|---|---|---|
| evidence, ring | 0 | 0 | 4 | 15 |
| evidence, all-to-all | 0 | 0 | 4 | 15 |
| agree, ring | 0 | 0 | 0 | 0 (all "neither") |
| agree, all-to-all | 0 | 0 | 0 | 0 (all "neither") |

At `alpha/beta = 10`, 15 of the 17 seeds whose starting average favoured a wrong hypothesis locked onto it
(both wirings). At 1, ring: 4 of 20 locked wrong; all-to-all: 4 of 20; agree: 0 of 20 in both.

**Which could have come out otherwise.** (c1), (c2) and the one-step check follow from the derived
steps and `W`'s eigenvalues once the arrangement realizes them; they check the machinery. The (c3)
outcome at `alpha = 10` was predicted from the same eigenvalues. What was not fixed in advance: whether the
ring survives its undamped mode at `alpha = 1` (it did), and the (c4) counts `4` and `15`, which depend
on how the first few data weigh against the amplified start. Not predicted: the ring and all-to-all
counts are identical. The consensus direction is the plain average for both doubly stochastic `W`, so the
wiring changes only how fast holders agree, not on what.

**Reasonable or weird** (the run's judgment, not evidence). *A scientific community:* reasonable and
recognizable. Taking colleagues' conclusions as fresh evidence double-counts shared data and locks a
field onto its founders' prior (15 of 17 at strong reading), while agreeing (averaging) is harmless when
everyone also reads the world. Weird: the outcome does not depend on the network shape, unlike real
citation networks, where hubs matter; that is because every `W` here is doubly stochastic, so no holder
is louder. *A person's beliefs* (holders as one mind's modules): echo-chamber self-confirmation is a
reasonable reading of (c2); weird that a mind would hold 10 identical sub-beliefs. *A population:*
`log q_k` as evidence has no population reading I can defend (organisms do not multiply by a neighbour's
log-share); the agree arm reads loosely as migration between demes, mixing log-abundances, which is not
how migration mixes (it mixes abundances). *A cell:* a quorum-sensing collective that amplifies its
members' signals would reinforce an initial bias in this way; reasonable as an analogy only. *Weird at
every scale:* agreement at `alpha = 10` destroys consensus instead of producing it. That is an artefact
of a synchronous discrete step overshooting, not of the cost; a smaller step at the same cost would
converge.

**What this does not establish.** That communities do either of these. That the KL cost and the
quadratic `y`-chart cost agree beyond the step (they share the step, not the cost value). That anything
here survives a `W` that is not doubly stochastic, asynchronous updates, or holders whose data differ in
quality. And nothing here passed through the inverse-Fisher response the derivation used.

### 5.4 Test (d)

Run 2026-10-01 by Claude (subagent of Claude Opus 5.5, 0fe746c1). Code: `science.mutation_matrix`,
`mutation_selection`, `drift_filter`, `run_variation`; `tests/test_science_scales.py`, 7 tests for (d),
3 passed, 4 strict xfail (whole file 24 passed, 9 strict xfail, 45 s). Charts, settings and definitions
as fixed in §4: population in the abundance chart, mind in the probability chart, totals, seeds 0-4,
120 steps.

**(d1) Held.** At `eps = 0`, in all 5 seeds and both readings, the largest weight is on `mu = 1` at every
step of the second half and at step 120; the weight on every type outside `{0, 0.5, 1}`, 3.5 included, is
exactly `0.0` at every step.

**(d2) Held, as defined before code.** The population's largest weight reaches 3.5 for every `eps > 0` and
every seed, and the spread rises with `eps` at every seed. Seed 0:

| `eps` | first step 3.5 is largest | largest at step 120 | 2nd half at 3.5 | spread (last 40) |
|---|---|---|---|---|
| 1e-3 | 39 | 4.0 | 0.50 | 0.2246 |
| 1e-2 | 25 | 4.0 | 0.18 | 0.2282 |
| 1e-1 | 20 | 4.0 | 0.47 | 0.3773 |

Seeds 1-4 first steps (`eps = 1e-3, 1e-2, 1e-1`): (34, 23, 12), (27, 22, 11), (27, 22, 11), (41, 25, 12);
final spreads rise from about 0.06-0.09 to 0.17-0.22 to 0.37-0.38. At step 120 the largest weight is on 3.5
in 11 of those 12 runs, but in none of seed 0's three. The numbers match the closed-form values in §4
exactly as printed. **A reach, not a hold:** truth 3.7 sits near the 3.5/4.0 midpoint 3.75, and a filter
that forgets (mutation) follows a recent sample mean, so the peak wanders between 3.5 and 4.0. Weaker
selection memory (larger `eps`) makes it wander more (seed 4, `eps = 0.1`: 3.5 is largest for only 32%
of the second half).

**(d3) Refuted at the tolerance fixed before code** (`< 1e-12`): measured `max |N / sum N - p|` =
`4.2e-12` (`eps = 0`), `8.7e-12` (1e-3), `1.1e-11` (1e-2), `3.6e-12` (1e-1). **Cause, measured:** the mind's
arrangement matches the plain filter to `5.0e-16`; the population's arrangement differs from the plain
recursion `D_l M N` by `1.1e-11`. Its step is `N + D_l (M - D_l^{-1}) N`, which subtracts `N` and adds
it back, losing about `epsilon_machine / l_i` relative digits per step in a type with small relative
fitness `l_i`, and the losses compound over 120 steps. **What holds instead:** the two readings are one
filter; the plain recursions agree to round-off; the arrangement realizations agree to `1.1e-11`. The
gap is the price of writing the population step as an exact forcing.

**(d4) What was found.** Mutate-then-select in the abundance chart **is** an arrangement step with an
**exact** forcing, `d(1/2 N^T (M - D_l^{-1}) N)`, under the constant response `diag(l)`. Its Jacobian
is symmetric to `0.0`, and one step lands on `D_l M N` to relative `1.4e-16`. It is exact because `M` is
symmetric (reflecting ends), and the likelihood sits in the response, so the response is rebuilt per
datum. Mutation alone is `d(1/2 N^T (M - I) N) = -(eps/4) d sum (N_i - N_{i+1})^2` under the Euclidean
response, a heat-equation step on the grid. **This exactness certifies little:** every linear step
whose `A - I` is diagonalizable with real eigenvalues has such a realization (STATUS §2, "Not evidence
at all"), and it cost (d3) its tolerance. The mind's normalized filter in the probability chart: no
exact forcing found. The one factorization tried has a non-symmetric Jacobian, and the realization used,
`F(p) - p` under a Euclidean response, is not exact (relative Jacobian asymmetry `0.56`). No proof that
none exists in any chart.

**Which could have come out otherwise.** (d1) is guaranteed: with `eps = 0`, `M = I` and a zero stays
zero under multiplication. The (d2) numbers were computed from the recursion before code, so the run
checks the realization, not the prediction. The (d2) prediction itself (written before that script) could
have failed and partly did: the peak is reached and not kept. (d3)'s refutation and the (d4) findings
were open.

**Reasonable or weird** (the run's judgment, not evidence). *A population:* reasonable. Without
mutation nothing outside the founding types can appear; with it, the population climbs to the best
available type in about `log(1/eps)` steps per grid step, and the standing variation (spread) grows with
the mutation rate. That is mutation-selection balance. Weird: the reflecting ends, and that a mutant
appears as a fraction `eps/2` of a parent's abundance at once, with no lag and no integer
individuals, so even `eps = 1e-3` seeds 3.5 within 5 steps. *A cell:* the same reading at the level of
a clone's phenotypes (bet-hedging switching rates); reasonable as analogy. *A person's beliefs, a
scientific community:* reasonable that a belief which has never given weight to 3.5 cannot come to
hold it, and that assuming the world drifts ("keep an open mind") is what lets it. Weird: the drift
assumption buys novelty only *inside the grid*, and only next to what is already held. The mind moves to
3.5 one neighbouring hypothesis at a time. A community that could jump straight to a far idea is not this
filter. And the forgetting that lets it find 3.5 also keeps it from settling there (seed 0).

**What this does not establish.** Anything about David's second kind of novelty (a new token, a new
port), which §4 put out of scope; the library has no changeable interfaces. That the exact forcing
found for mutation-selection is a structure of the book's kind and not the cheap realization STATUS §2
warns about. Or that the mind's filter has no exact realization in some chart.

### 5.5 Test (e)

Run 2026-10-01 by Claude (subagent of Claude Opus 5.5, 0fe746c1). Code: `science.rival_chooser`,
`rival_closed_K`, `rival_settle_fn`; tests in `tests/test_science_scales.py`.

**First run, with the construction fixed in §6 (recorded before any change).** In 10 of 120 runs the
arrangement went non-finite, and that alone decides three verdicts. Cause: both of my step sizes were too
large. At B count 0 the seed-7, `d = 3` gap is `|K_0 - K_1| = 109`, not the population value 90 I sized
`dt = 0.01` for, so `dt * 109 > 1` and the small share went negative (fixed and incumbent-only arms).
In the fixed arm at `d = 3`, a single B draw near 5 moves the logit by `3b - 4.5 ~ 10.4`, so the warm
`dt = 0.1` overshot to a negative share at B counts 22-105 in all 10 seeds, after the crossing. Verdicts
on that run, as defined in §6:
- **(e1) at `d = 3`: not met.** Seed 7 has no crossing (NaN from B count 0). The other 9 match the
  closed form exactly: 22, 20, 20, 21, 23, 31, 15, -, 27, 21.
- **(e1) at `d = 2`: held.** 10 of 10 match the closed form (23, 20, 20, 22, 26, 37, 13, 35, 28, 21;
  median 22.5); logit slope 1.95 against 2.0.
- **(e2) held at both `d`.** Crossings match the closed form in 10 of 10 (medians 20.5 and 21.5).
  Slopes 1.175 (`d = 3`) and 0.827 (`d = 2`) against 1.151 and 0.805.
- **(e3) at `d = 3`: not met.** Seed 7 again; the other 9 match: 84, 77, 79, 87, 81, 87, 52, -, 89, 69.
- **(e3) at `d = 2`: held.** 10 of 10 match (53, 54, 54, 62, 48, 60, 25, 65, 59, 45; median 54.0
  against 49.7).
- **(e4): not met.** The rival overtook by B count 150 in 9 (fixed), 10 (both learn) and 9
  (incumbent-only) of 10 seeds; the missing seed is the NaN.

Not predicted: in the learned arms the arrangement's logit differs from the closed form by up to 4.4
(both learn) and 1.8 (incumbent-only, `d = 2`), far from the crossing, while the crossings match exactly.
The losing hypothesis's trust box is pushed by a force scaled by its share (as high as `e^{-100}` against),
so its trust stops learning. The arrangement's loser is held at a trust it never finishes fitting. The
§6 note called this second order; over many B counts it is not.

**Re-runs: only the step size changed; predictions, definitions, draws and everything else as fixed.**
The first-run verdicts above stand as recorded; those tests stay strict xfails. *Re-run 1* sized each
B count's `dt = min(0.1, 0.5 / max(1, |jump|))` from the closed-form jump in `K_0 - K_1`, with
`ceil(30/dt)` steps. It **failed differently**: in "both learn" at `d = 3`, 4 seeds (3, 4, 5, 8) went NaN
at B count 0, because the closed-form gap there (about 20, trust at rest) understates the gap at the
starting state (about 100, every trust at 1). *Re-run 2* sizes the B-count-0 step from the gap at the
starting state, with everything else as in re-run 1. Its results, all 60 runs finite:

| arm (`d = 3`) | first overtake by seed 0-9 | rival overtook by B count | in | stayed ahead | median | logit slope |
|---|---|---|---|---|---|---|
| fixed | 22, 20, 20, 21, 23, 31, 15, 30, 27, 21 | 31 | 10 of 10 | 10 | 21.5 | 4.43 |
| both learn | 18, 31, 22, 15, 20, 21, 14, 27, 25, 19 | 31 | 10 of 10 | 9 | 20.5 | 1.175 |
| only the incumbent learns | 84, 77, 79, 87, 81, 87, 52, 92, 89, 69 | 92 | 10 of 10 | 10 | 82.5 | 1.170 |

| arm (`d = 2`) | first overtake by seed 0-9 | rival overtook by B count | in | stayed ahead | median | logit slope |
|---|---|---|---|---|---|---|
| fixed | 23, 20, 20, 22, 26, 37, 13, 35, 28, 21 | 37 | 10 of 10 | 9 | 22.5 | 1.95 |
| both learn | 17, 33, 23, 14, 20, 26, 13, 31, 27, 19 | 33 | 10 of 10 | 9 | 21.5 | 0.827 |
| only the incumbent learns | 53, 54, 54, 62, 48, 60, 25, 65, 59, 45 | 65 | 10 of 10 | 9 | 54.0 | 0.821 |

Every crossing equals the closed-form sample value (120 of 120). **On re-run 2: (e1) held** (medians 21.5,
22.5 against `n_A = 20`; slopes 4.43 and 1.95 against 4.5 and 2.0). **(e2) held** (medians 20.5, 21.5;
slopes 1.175 and 0.827 against 1.151 and 0.805: the softer switch). **(e3) held** (medians 82.5 against
78.2 and 54.0 against 49.7; at `d = 3` the incumbent-only median is 3.8 times the fixed arm's). **(e4)
held:** at `d = 3` the rival overtook in 10 of 10 seeds in every arm, by B count 31, 31 and 92. That is
the `d` at which (b)'s learned-trust holder kept its idea against a know-nothing default (it held
up to `d = 4.875`). The fixed-arm logit matches the closed form to `7e-12`. The learned arms are off by up to 4.3-4.6
(both learn) and 1.8 (incumbent-only). That is the losing hypothesis's trust lag reported above, and it
does not move any crossing.

**Which could have come out otherwise.** The predictions are algebra on the closed forms, and the
scratch script had already shown, before code, that the sample values satisfy (e1)-(e4) as defined. So the
predictions themselves could only have failed through the definitions' tolerances, and they did not.
What was open was whether test (b)'s construction, scaled up to totals and a rival, realizes them.
It did not as first built: two step-size failures. It does with a rule that sizes each step from the
jump it must absorb. Also open, and not predicted: the trust of a hypothesis nobody holds stops being
learned, so away from the crossing the arrangement holds a loser at an unfitted trust. That moves the
shares but not the decisions.

**Reasonable or weird** (the run's judgment, not evidence). *A scientific community:* reasonable and
recognizable. A rival that predicts the anomaly wins after roughly as many discordant observations as
there were confirmations, whether or not trust in instruments is learned, provided both sides may learn
it. If only the incumbent may "blame the instrument", the rival needs `d^2 / log(1+d^2)` times longer
(3.8 at `d = 3`). That is the asymmetry Lakatos and Kuhn describe, and here it has a number. Weird:
every crossing is set by the founding `n_A`, so a field's inertia is exactly its stock of confirmations,
with no role for who holds them. *A person's beliefs:* reasonable that a belief yields to an alternative
that explains the surprise, where (b)'s holder, offered only "I don't know", kept its belief; this is the
run's sharpest contrast with (b). Weird: the person re-fits trust in every source from scratch at every
step. *A population:* two genotypes competing on two environments with fitness as likelihood. The trust
arms (one genotype may tune its tolerance and the other may not) read loosely as plasticity, and that
plasticity delays a sweep by the same factor. A loose analogy. *A cell:* no reading I would defend.
*Weird at every scale:* the loser stops learning. A hypothesis no one holds is not refitted, which is
recognizable sociology but is an artefact of scaling each trust box's force by its hypothesis's share.

**What this does not establish.** That communities assign trust in either of these ways; the arms are
stipulated (§6). That the symmetric arm is the fair one. That anything here holds with more than two
hypotheses, with an incumbent whose own source keeps producing data, or with the step-size rule replaced
by one not sized from the closed forms. The re-run's rule reads the closed-form jump to size its step, a
numerical choice that does not move the rest point but does use the closed form. And the first-run
failures stand as failures of the construction fixed in §6.

### 5.6 Test (f)

Run 2026-10-01 by Claude (subagent of Claude Opus 5.5, 0fe746c1). Code: `science.translate_beliefs`,
`stats_inc`, `run_summaries`, `run_conclusions`, `run_bounded_mind`; 14 tests for (f) in
`tests/test_science_scales.py`, 7 passed and 7 strict xfail (whole file 42 passed, 19 strict xfail, 57 s). One run, as fixed in §7. No construction choice failed (no NaN, no
step-size failure: the beliefs move by translation), so there is no re-run. Bookkeeping (papers, held
sets, trials) is plain code; only the beliefs are an arrangement (§7). All counts are over seeds 0-19
at the same settings in every arm.

**(f1) Held** (main variant, truth 5.0). Every holder at every step equals the one mind on exactly the
draws it holds, to `2.3e-13` (ring) and `2.3e-13` (all-to-all). After the extra steps every holder equals
the pooled mind on all 480 draws, to `1.9e-25` and `7.6e-26`. 0 of 20 seeds end on a wrong hypothesis.
*Side variant (truth 5.2):* the per-step identity is `1.4e-12` (ring) and `1.36e-12` (all-to-all), just
over the `1e-12` I fixed: incremental translation and a direct sum differ in round-off. Final agreement
`1.2e-12` and `4.2e-13`; 0 of 20 end away from the KL-closest 5.0.

**(f2) Refuted on the ring, held all-to-all.** "More confident than the pooled mind at every step" fails
on the ring in 20 of 20 seeds. Early on a ring holder holds fewer draws (3 and 12 against 8 and 16 at
steps 1-2, as §7 computed). In every seed every holder is more confident from step 9 on. All-to-all: 0 of
20 seeds violate it (truth 5.0); side variant 1 of 20 (more confident from step 4). The second half
held in both wirings: **without provenance 7 of 20 seeds end on a wrong hypothesis (all 8 holders on
it in all 7), with provenance 0 of 20**, ring and all-to-all alike. Side variant: 9 of 20 against 0.

**(f3) Refuted as written: 18 of 20, not 20 of 20; the rest held.**
- With the truth in holder 5's slice, every holder ended with the truth as its conclusion in 20 of 20
  seeds (both wirings). But all holders held it above 0.99 in only 18 of 20. In seeds 10 and 18,
  holder 4 (slice 4.0, 4.5) holds `{4.5, 5.0}` with weight 0.984 and 0.877 on 5.0: its own 150 local
  draws do not separate adjacent grid points to 0.99.
- No holder ended on a wrong hypothesis.
- Arrival on the ring: holder 5 found it first in 20 of 20 seeds (step 1 in 12, at most 11). Median
  delay by ring distance: 0, 5, 15, 25, 25.5 steps at distances 0-4, a slope of **7.1 per hop**, inside
  `[r, 2r]`. A hop costs about two trials, because a replication of 5.0 against 4.5 over 5 draws
  passes the 0.5-nat threshold only about half the time (mean 0.625, sd 1.1). Distance 4 is reached from
  both sides.
- Truth in no slice (8.5): no holder holds it in 0 of 20 seeds (both wirings); the community settles on
  7.5, the best of the union of slices.
- The bounded mind (`K = 2`, every draw) holds the truth in 20 of 20 with slice `{10, 11}`, in 0 of 20
  with `{0, 1}`, and in 0 of 20 when the truth is in no slice.

**(f4) Held.** Without replication, seeds ending on a wrong hypothesis (any holder / all holders):
**ring 11 / 7 of 20, all-to-all 12 / 12 of 20; with replication 0 / 0 in both.** All holders hold the
truth without replication in 9 (ring) and 8 (all-to-all) of 20, against 18 with replication.

**Which could have come out otherwise.** (f1) is sufficient statistics adding, guaranteed up to
round-off, and it checks the bookkeeping. (f2)'s first half was foreseeable from the counts in §7 for its
early steps, and I fixed it as written anyway. Open, and not computed before code: the lock counts (7,
9, 11, 12 against 0), the per-hop delay (7.1), and the two seeds where local data could not separate
5.0 from 4.5. Replication's 0 wrong endings could have failed: a mistaken conclusion passes a 5-draw trial
whenever its noise favours it, and none survived to the end.

**Reasonable or weird** (the run's judgment, not evidence). *A scientific community:* reasonable and,
to me, the strongest result of the series. Papers with provenance make the community exactly the pooled
mind. Papers without provenance double-count and lock 7 of 20 communities onto a wrong value, all members
together (citation cascades). Conclusions let a community hold what no member started with, and
replication keeps that honest (0 wrong against 11-12 without). A truth outside every member's starting
slice is never found: novelty still needs variation, (d) at the community's scale. Weird: the
replicating community holds the truth with less certainty than the pooled mind would (holder 4's 0.877),
because each member judges a received conclusion only on its own draws. That is realistic but is
not the "one mind" the summaries reach. Also weird: no member ever revises the threshold, and every
hop's slowness comes from testing against an adjacent hypothesis. *A person's beliefs:* modules that
pass conclusions and re-test them read as a mind checking a recalled idea against new experience; a loose
analogy. *A population:* conclusions with replication read as migrants (a received type) entering a deme
and kept only if they outgrow the resident best over `r` generations: reasonable, and (f3)'s spread at
about one hop per two trials is a travelling-wave speed. Summaries have no population reading I would
defend. *A cell:* none I would defend.

**What this does not establish.** That real science passes papers or conclusions this way; the arms are
stipulated, and "replication" here tests a received conclusion on new local draws, not by re-running a
protocol (§7). That upkeep acts as test (b)'s chooser would: it enters only as `K` and the threshold
`c r`. That the results survive other `K`, `r`, `c`, grids or a misspecified world for (f3) and (f4) (not
run). And that the bookkeeping is an arrangement: held sets that change are a changeable interface, which
the library lacks.

### 5.7 Native round

**Stage 1, (n1) and (n2).** Run 2026-10-01 by Claude (subagent of Claude Opus 5.5, 0fe746c1). Code:
`arrangements/science_native.py` (`port_wire`, `wire_boxes`, `draw_source`, `community_native`,
`summaries_native`); 16 tests in `tests/test_science_native.py`, 16 passed, 10 s. One run, as fixed in
§8's stage-1 numbers; no construction choice failed (no NaN; every step is a translation or round two's
own step), so there is no re-run. Every community here is `compose_seq(tensor(world, holders, [journal]),
port_wire)`, run by `Phi` under one `lax.scan`; no Python between steps. Every reader's forcing is the
partial differential of its own cost in its own state, zero on its in-ports, so composition pulls
nothing back onto a sender.

**(n1) Held.** Over 100 runs per arm and wiring (20 seeds at `alpha in {0, 0.1, 1, 10}`, `beta = 1`,
`T = 100`, plus (c1) or (c2)'s setting), the wired composite of a world box and 10 holder boxes equals
round two's monolithic `account_community`: **bit-identical in 100 of 100 runs** for agree (ring and
all-to-all) and evidence on the ring; evidence all-to-all bit-identical in 22 of 100, worst relative
difference `4.1e-15`, against `1e-10` fixed. Every seed's classification is reproduced in all 400 runs,
including the divergent ones (`alpha = 10`, magnitudes to `7.5e127`). No step shift: a box reads out its
state, and the monolithic box also read every account at the current state.

**(n2) Held, both arms, with one gap.** *No provenance:* with a one-step register in each holder (below),
native state `s + 2` equals round two's step `s`: worst weight difference `7.1e-12` (ring, 5.0),
`2.7e-12`, `3.8e-14`, `2.0e-13`; worst relative `Y` difference `4.4e-16`; seeds ending on a wrong
hypothesis, any / all: **7/7 of 20 at truth 5.0 and 9/9 at 5.2, ring and all-to-all**, round two's
counts. *The plain box without a register* (it publishes its stock before its own draw, so its counts at
steps 1-5 are 1, 4, 13, 40, 121 on the ring against round two's 3, 12, 39, 120, 363): the same counts,
7/7 and 9/9, as numpy fixed before code. Checked after the run, in numpy: the same seeds lock, on the
same wrong hypothesis (4.5 or 5.5), in both timings. *Journal:* a holder's out-port carries its
registered draw `(1, x)`, a journal box adds what arrives, and a holder steps onto the journal's
log-likelihood. At every step every holder equals the pooled closed form on the rows published so far:
worst weight difference `1.1e-13` (truth 5.0) and `4.5e-13` (5.2), both wirings, against `1e-12` fixed;
native state `s + 3` holds rows `0..s`, all 60 rows by step 62; 0 of 20 seeds end wrong at 5.0 (and at
5.2).

**Expressiveness, the findings.**
1. *A readout cannot depend on the current input.* A box `I -> <in | out>` reads out `out_f(q)`, its
   state alone. So round two's (f), which publishes a holder's stock after its own draw of the same step,
   and the journal arm as worded ("only its current draw's statistics"), are not expressible as stated.
   The closest native thing is a register (a pending `(count, sum)` with forcing `x_in - pend`), which
   reproduces round two's arithmetic exactly with a two-step shift; a value then crosses one box per
   step. This is the semantics, not a missing feature: what a box sends is always from its state.
2. *No smooth source of a fixed sequence.* The world is a clock (exact forcing `d(clock)`) whose readout
   is a table lookup, not a smooth map of the clock (`stop_gradient` on the index). It is sound here only
   because every reader's forcing is zero on its in-ports, so no cotangent reaches the lookup. A world
   that a reader pushed on could not be built this way. (`rand.py`'s stochastic coalgebras were not
   tried.)
3. *`graph_wire` needs one port dimension throughout.* A draw port (1 or 2 coordinates) and an account
   port (`n`) differ, so `science_native.port_wire` writes `R^-` of an arbitrary function from in-port
   coordinates to out-port coordinates, of which `graph_wire` is a case. It belongs in `wiring.py` if kept
   (not moved: that module was not to be edited).
4. *"Nothing pushed back" needed nothing new:* a one-form need not be exact, and a forcing with zero
   in-port component is one. The composite's forcing is then the block of each holder's own cost
   differential, which is what `stop_gradient` faked in round two.

**Which could have come out otherwise.** Little. (n1) is the same arithmetic routed through the wiring,
so it checks one thing: that `compose_seq` with in-port-free forcings carries nothing back and shifts
no step. It could have failed on either, and did not. (n2)'s counts were fixed in numpy before code, so
it checks the construction, not the world; the journal's agreement is sufficient statistics adding. The
one open number was the lag variant's count, computed before code at 7 and 9, and it held: the plain
box's one-step lag changes every early count by a factor of about three and no outcome.

**Reasonable or weird** (the run's judgment, not evidence). *A scientific community:* reasonable. A
paper reaches readers through a register and a journal, three steps after the draw, and a reader who
reads only the journal is exactly the pooled mind on what was published: publication delay, and no
double counting because the journal adds each draw once. Without provenance the same 7 communities lock
whether authors publish before or after their newest result, which says the lock comes from the
exponential overcount of the earliest draws, not from timing (my reading of the same-seeds check). Weird:
the journal reader drops its own unpublished draw until the journal returns it; a real scientist would
not ignore her own bench. *A person:* the register is working memory holding a percept one step before
it is reported; loose, an analogy. *A population:* none I would defend for summaries. *A cell:* a journal
box is a shared medium (a quorum-sensing pool) that every cell writes to and reads; a loose analogy, and
the medium here never decays, which a real one does.

**What this does not establish.** Anything about science beyond round two's (c) and (f) summaries: stage
1 shows that the book's grammar carries those two mechanisms with the same numbers, and where it needs a
register or a lookup to do so. Not (f)'s conclusions arms, (n3)-(n5), which are stage 2.

**Stage 2, (n3)-(n5).** Run 2026-10-01 by Claude (subagent of Claude Opus 5.5, 0fe746c1), after David
approved the seed design. Code: `science_native.conclusion_holder`, `conclusions_native`,
`conclusions_native_run` (world, 8 holders, wired, run by `Phi` for 24,000 steps, vmapped over the 20
seeds); 9 stage-2 tests in `tests/test_science_native.py`; the whole file is 23 passed and 2 strict xfail,
40 s. Settings as fixed in §8's stage-2 numbers. **Runs, in order.** A smoke run of the seed arm (2 seeds,
3 draws) showed run 1's NaN and a 20-seed timing run of run 2's seed arm (truth 5.0, ring) showed 19 of
20; neither changed any setting. The recorded run then computed run 1 (Shahshahani, as designed) and run
2 (Euclidean frame, decided and disclosed before code) together; it failed one assertion, the ring case
of (n3); I marked that case strict xfail and re-ran the file unchanged otherwise, with the same numbers.

**Run 1, as designed: the seed is NaN.** In 20 of 20 seeds, in both wirings, every weight is NaN after
the first draw. The cause was predicted before code: `d(eps sum c log p)` is infinite on the face
`p_h = 0`, and the Shahshahani sharp multiplies it by `p_h = 0`. Run 1's evidence arm is finite.

**(n5) Refuted, as analysed before code.** Checked by autodiff: run 2's seed forcing equals
`d(eps(<c̄,p> - |p|^2/2))` to `1.5e-17`; run 1's evidence forcing equals `d<f + alpha c̄, p>` exactly; the
Shahshahani sharp of 50 random covectors has normal component exactly 0 on a face. Run 2's data forcing,
the replicator field, has a Jacobian asymmetric by 3.4, so it is not a differential. **A seed that plants
support needs a non-exact piece.** Either selection is exact and the seed is infinite (run 1), or the
seed is exact and selection is non-exact (run 2). The reason: a response that keeps every fitness's
faces invariant kills the normal part of every finite covector.

**Run 2 never left the simplex.** The minimum weight was exactly 0 and `max |sum p - 1|` was `2.6e-15`
over every step of all 16 runs, with no negative weight and no NaN.

**(n3) Refuted on the ring, held all-to-all and with the truth in no slice.** Seed arm, truth 5.0: every
holder's top hypothesis is the truth in **19 of 20 seeds on the ring** (seed 14: holders 0-2 still on 4.5
at draw 150) and **20 of 20 all-to-all**. Every holder is above 0.99 on it in 18 and 19 of 20. Holder 5
finds it first in 16 (ring) and 18 of 20. Ring, median delay by distance 0-4: 0, 8, 21, 32, 38 draws, a
slope of **10.0 per hop**, against `[12, 48]` fixed (24 predicted). All-to-all: 14.5, 13, 11, 10.5, no
distance effect, as it should be. Truth 8.5 (in no slice): no holder's top is the truth in 20 of 20, its
weight exactly 0 everywhere, and every holder's top is 7.5 in 20 of 20, both wirings.

**(n4) Held, but not by its mechanism, as predicted before code.** Seeds ending on a wrong hypothesis,
any / all holders, truth 5.0, at the same settings:

| arm | ring | all-to-all |
|---|---|---|
| seed | 0/0 | 0/0 |
| evidence | 20/0 | 20/0 |
| plant and count | 7/6 | 13/13 |

Run 1's Shahshahani evidence arm gives the same 20/0. The evidence arm cannot weight anything a holder
does not hold, so 20/0 means each holder stays on its own slice's best, not that wrong conclusions
spread. In this arm every holder's top is the truth in 0 of 20. The side arm **plant and count** (seed
plus bonus, round two's "kept at once and credited") is where wrong conclusions spread: 7/6 and 13/13
against the seed arm's 0, and every holder's top is the truth in 13 and 7 of 20.

**Against round two (§5.6).** (f3): round two had 20 of 20 conclusions on the truth on the ring, 18 of 20
above 0.99, and 7.1 steps per hop; native has 19, 18 and 10.0. The difference is that round two's
replication is a 5-draw trial against a 0.5-nat threshold, while the seed is a continuous inflow judged
by selection. The truth in no slice is never found in either, and both settle on 7.5. (f4): round two's
"without replication" gave 11/7 (ring) and 12/12 (all-to-all); native plant-and-count gives 7/6 and
13/13. Native evidence alone has no round-two counterpart.

**Which could have come out otherwise.** Run 1's NaN, (n5)'s refutation, the evidence arm's 20/0 and the
truth-in-no-slice result were all derived before code; the runs confirm the derivations and the code.
These came out open: the ring's 19 of 20, the 10.0 per hop, the 18-19 above 0.99, and plant-and-count's
7 and 13. The per-hop slope beat `log(1/eps)/KL` by a factor of 2.4. My reading, not checked: the
sharpened readout `p^20` leaks a small seed before a neighbour's top switches, so seeding starts before
arrival is counted.

**Reasonable or weird** (the run's judgment, not evidence). *A scientific community:* reasonable. A
received conclusion that you hold at no weight cannot be argued into your beliefs by evidence. It must be
planted first, as a hypothesis worth testing, and then your own data decide; that is David's replication.
Planting and also counting it as evidence makes a third to two-thirds of communities lock onto a
neighbour's wrong value. Weird: the seed is a constant inflow, so a holder keeps a little weight on what
its neighbours say even after its data reject it (min final weight on the truth 0.197 on the ring). *A
population:* the seed is migration into a deme and the evidence bonus is a fitness subsidy for types that
are absent; that a subsidy cannot create an absent type is exactly right. *A person:* hearing an idea makes
it thinkable, and only then does experience weigh it; a loose analogy. *A cell:* none I would defend.

**What this does not establish.** That the book's grammar has no exact seed: only that, with selection
written as `d<f, p>` under a face-preserving response, there is none. A different chart (test (e)'s trust
boxes, or the natural chart of item 42, not run) might place the non-exact piece elsewhere. Also not
established: that the counts survive other `eps`, `alpha`, `tau` or `dt` (not run), or anything about
real science.

### 5.8 Test (g)

Run 2026-10-01 by Claude (subagent of Claude Opus 5.5, 0fe746c1). Code: `science_native.excuse_world`,
`excuse_reader`, `excuses_native_run`. A world box emits each incident (seen flags, behaviour, which side
channels came back 0) for 4000 steps. A reader box holds `theta` (Euclidean) and every incident's
attachments (`choice.py`'s `{attached, not}` chooser on each, response `dt a(1-a)`). Its forcing is the
exact `-d` of the §9 cost, zero on its in-port. They are wired and run by `Phi`, with nothing in Python
between steps. 9 tests for (g1)-(g3) in `tests/test_science_native.py`; the file is now 30 passed and 3
strict xfail, 40 s. One recorded run per version. A first (g)-only run compared against 4-decimal
scratch values; I then fixed the test to compare against the unrounded numpy scratch at the `1e-6` that
§9 fixed, and re-ran. No construction choice failed: no NaN, and the attachments stayed in `[0, 1]`
(minimum `4e-307`, maximum 1.0). The native run equals the numpy scratch to `1.1e-16` (convex) and 0
(double well).

**Convex version (`a in [0,1]`, linear drawing cost), as called before code.** `theta`: 0.975, 0.925,
0.85, 0.75, 0.625, 0.504, 0.432, 0.381, 0.343, 0.311. Moves: 0.025, 0.05, 0.075, 0.1, 0.125, 0.121,
0.072, ... The excuse at incident `i` (figure `i`) is attached at 0.875, 0.724, 0.547, 0.344, 0.115, then
none. **(g1) held:** at excused incidents 1-3, `theta` moves by exactly `delta_i/(D w)` = 0.025, 0.05,
0.075, all below the unexcused 0.2. **(g2) held:** explaining cost 0.087, 0.145, 0.165 over the excused
incidents. **(g3) refuted, as called:** at the threshold incident 4 the move is 0.100, less than the sum
0.150 of the earlier moves. The cost is convex, so the move grows with the distance of the excuse used
and saturates; there is no jump. Excuses also decay backwards: by incident 10 the excuses for incidents 1-3
are attached at only 0.211, 0.110 and 0.010. Once `theta` has moved, the old anomalies are re-explained
by the update, not by their figures.

**Double-well version (`+ kappa a(1-a)`, `kappa = 0.5`: an attachment is made or not), disclosed as the
second version.** `theta`: 1, 1, 1, 0.876, 0.778, 0.700, 0.546, 0.5, 0.462, 0.429. Attachments: 1, 1, 1,
then 0. **(g1) held:** moves of exactly 0 while excused, and explaining costs exactly `delta_i` = 0.1, 0.2,
0.3. **(g2) held.** **(g3) held:** at incident 4 `theta` jumps 0.124 against 0 before. Figure 4
(`delta = 0.4`) sits exactly at the global binary threshold `1/2 D^2 w/(w+1) = 0.4`. It loses because
`theta` responds faster than an attachment grows from 0.01: the outcome depends on the reader's rates,
not on the costs alone. There is a second avalanche at incident 7 (move 0.155): the excuse for incident 3
is dropped (final attachments 1, 1, 0 for incidents 1-3), because after `theta` has moved, keeping it
costs more than it saves.

**(g4) not run natively; its closed forms are in §9.** At `d = 3`, soft: symmetric median 21.5, the same
as (e)'s fixed arm; incumbent-only 34.5. Hard: incumbent-only 28.0. (e)'s blanket discount gave 82.5.
**The one native smoke run failed**, and this is recorded before anything else about (g4): seed 0,
`d = 3`, soft, incumbent-only, unilateral, 100 steps per count. Its `lambda` rose from 108.6 to 180 and
never crossed 0, against a closed-form crossing at count 30. I did not find the cause within budget. The
code carries an UNVERIFIED note. The run also took 55 s for 15,000 steps, because `rvect`'s response is
a dense matrix (`(1 + 2720)^2`): a cost gap for a box with many coordinates and a diagonal response. The
1000 steps per count I fixed would take about 9 minutes per seed and arm. So (g4)'s verdict rests on
closed forms only. Read that way: the symmetric arm is held (21.5 against 21.5 and 22.5). The
incumbent's advantage is **bounded by its library in the hard variant** (delay at most 7 draws in 10 of
10 seeds, within `K = 8`). In the soft variant it is **not**: a ruled-out figure keeps 8% attachment and
leaks a small fixed-ratio discount, delay 8 to 14. Both are far below the blanket discount's.

**Which could have come out otherwise.** Little in (g1)-(g3). The numpy scratch fixed every number before
code, so the native run checks the construction: wiring, chart, and nothing pushed back. The calls that
could have failed were mine before running: convex is gradual, the double well is abrupt. The two open
findings, the backwards decay of old excuses and the rate race at figure 4, were seen in the scratch
before the native code and not predicted.

**Reasonable or weird** (the run's judgment, not evidence). *A person:* reasonable, and close to David's
Bob. With "an excuse is made or not", trust holds exactly until the excuses run out, then drops at once.
Once it has dropped, the old excuses are quietly abandoned ("maybe he wasn't in a hurry that day
either"). The convex reader is the weird one: it half-believes each excuse and lets trust leak a little
every time, more as the excuse gets further-fetched. *A scientific community:* auxiliary hypotheses
protect a theory until they are ruled out one by one, then the theory falls. In closed form, a finite
library protects it for a bounded number of anomalies, unlike a blanket discount. Weird: whether figure
4 is used is decided by the reader's speed, not by evidence. *A population:* no reading I would defend.
*A cell:* none.

**What this does not establish.** That excuses are built this way: the library, distances, footprints
and ruling-out schedule are stipulated, and the figures are not run as arrangements (P34). That (g4)
holds natively: it was not run. Nor that the double well is the right form of "made or not"; a
different binary penalty could move the threshold.

**(g4) diagnosis** (the same run's, resumed 2026-10-01 at the coordinator's request; every scratch
computation was sized before it started).
1. *Smallest instance* (1 mirrored figure, `n_A = 3`, B counts 0-8, seed 0, `d = 3`, 2000 steps per
   count). Native and closed form agree at every count to 1e-4, the closed form's grid, in the "none" and
   "incumbent" arms. Here `|lambda| <= 12`.
2. *Scaled up* (8 figures, `n_A = 20`, B counts 0-8). They part **at count 0**, before any excuse
   matters: native 108.566, closed form 100.991. The gap is exactly `K_0 = 1/2 sum a^2 = 7.575`. The native
   rest was `lambda = K_1`, not `K_1 - K_0`.
3. *Cause: a floating-point loss in my chooser code, not the closed form, the sign, the gauge, the
   register, the shift constant or a share-scaled force.* `dU/dlambda` was taken by autodiff through
   `sigmoid`, whose derivative is computed as `sigma (1 - sigma)`. That is exactly 0 once `lambda > ~37`,
   so the `-p_0 K_0` term contributed nothing, while `p_1`'s derivative stayed accurate. The smallest
   instance never reached `|lambda| > 37`, which is why it agreed. **Fix:** the same differential written
   out, `dU/dlambda = p_0 p_1 (K_1 - K_0 - T lambda)`, with `p_0 p_1 = exp(-softplus(lambda) -
   softplus(-lambda))`, used both there and in the response.
4. *After the fix* (seed 0, `d = 3`, 8 figures, `n_A = 20`, B counts 0-40, 300 steps per count, both
   variants). Native `n_B*` equals the closed form in every unilateral arm: soft none / symmetric /
   incumbent 22 / 23 / 30; hard 22 / 23 / 26. The largest `|lambda - closed form|` is `7e-14` without
   excuses and 0.34 with them: at 300 steps per count the attachments have not fully converged. The
   one-potential forcing gives `n_B*` 30 against 23 in the symmetric arm, as predicted before code: the
   losing rival, at share about `e^{-100}`, builds no excuses.
5. *Not rerun at the fixed settings.* 151 counts × 1000 steps with the dense `2721^2` response come to
   about 5 minutes per seed and arm, so 120 runs would take hours. **So §9's closed forms stand,
   checked natively on one seed at counts 0-40; they are not the full native run.** The code's
   UNVERIFIED note is replaced by this diagnosis.

## 6. Second round, test (e): an idea against a rival (predictions, before code, 2026-10-01)

David, to "shall I run tests (e) and (f)": "Yes, please run those tests." Written by Claude Opus 5.5,
0fe746c1, before any code for (e); results go to §5.5. (f) waits on David's answer to how scientists
share (their conclusions or their data), which decides its design.

**Setting.** Two hypotheses, both at the same upkeep (it cancels): the incumbent `h0` ("the quantity is
0") and the rival `h1` ("the quantity is `d`"). The incumbent was established by `n_A` draws from source A,
`N(0,1)`, all seen before the contest. Then source B's draws, `N(d,1)`, arrive one per step. Costs are
**totals** over every draw seen (negative log-likelihood, constants kept), so shares sharpen with data. The
chooser is test (b)'s simplex chooser on `{h0, h1}` at `T = 1`. Each hypothesis predicts each source with
its own trust (precision), and in the arms below a trust is either fixed at 1 or learned by descent with
the `-1/2 log Pi` term, at rest `Pi = 1/S` (`S` the mean square about that hypothesis's value).

**Arms** (same draws, same `n_A`, `d`, `T`, seeds in every arm):
- **fixed:** every trust fixed at 1.
- **both learn:** every trust learned, for both hypotheses.
- **only the incumbent learns:** `h0`'s trusts learned, `h1`'s fixed at 1 (the incumbent may blame the
  instrument; the rival may not).

**Predictions** (closed forms from the sample mean squares; the run fixes `n_A, d` and the numbers before
construction code). Write `D_X^i` for the extra total cost hypothesis `i` pays on source X over fitting it.
- (e1) *fixed:* the rival overtakes (share of `h1` passes 1/2) at the B count `n_B*` where the sums of
  squares equalize; with population values, `n_B* = n_A`.
- (e2) *both learn:* the same crossing, `n_B* ~ n_A` (each hypothesis discounts the other's confirming
  source by the same `1/2 log(1+d^2)` per draw, so learning trust protects neither); the switch is
  softer, its logit growing per B draw by `1/2 log(1+d^2)` instead of `d^2/2`.
- (e3) *only the incumbent learns:* the rival needs `n_B* ~ n_A d^2 / log(1+d^2)` draws (at `d = 3`, about
  3.9 times as many as the incumbent's confirmations). Explaining away protects an idea against a rival
  only when the rival is denied the same move.
- (e4) Contrast with (b): at the same `d` where (b)'s learned-trust holder kept its idea against a
  know-nothing default, the rival here overtakes it after a finite B count in every arm.

**Domain objection, stated before the run** (a philosopher of science): theories are dropped for rivals,
not anomalies (Kuhn; Lakatos). (e2)–(e3) test whether the book's cost says the same, and under what
symmetry. **What it cannot show:** that real communities assign trust this way; the arms are stipulated.

**Numbers fixed before code** (the run's, Claude subagent of 0fe746c1, 2026-10-01; scratch numpy script
on the closed-form totals, building no arrangement). `n_A = 20`; `d in {3, 2}`; `T = 1`; seeds 0-9, each
`rng = default_rng(s)`, then `a = rng.normal(0, 1, 20)`, `b = rng.normal(d, 1, 150)`; the B count runs
`0..150`. The same draws in every arm. Upkeep is equal and is omitted (it cancels). Closed form per
hypothesis `i` with value `v_i`: for each source X with `n_X` draws and `S = mean (x - v_i)^2`, the
total cost is `n_X/2 S` (trust fixed at 1) or `n_X/2 (1 + log S)` (trust at rest `1/S`), plus
`n_X/2 log 2 pi`. The logit of `h1`'s share is `(K_0 - K_1)/T`. **Definitions.** `n_B*` is the
first B count at which `h1`'s rest share exceeds 1/2. (e1)-(e3) hold iff, at both `d`, (i) the
arrangement's `n_B*` equals the closed-form sample value in 10 of 10 seeds, and (ii) the median over
seeds is within 25% of the predicted population value: `n_A = 20` for (e1) and (e2), and `n_A d^2 /
log(1 + d^2)` = 78.2 (`d = 3`) or 49.7 (`d = 2`) for (e3). The "softer switch" of (e2) holds iff the mean
logit slope over B counts 1-150 is within 10% of `1/2 log(1+d^2)` (1.151 at `d = 3`, 0.805 at `d = 2`),
with fixed trust within 10% of `d^2/2`. (e4) holds iff at `d = 3` every arm overtakes by B count 150 in
10 of 10 seeds. **Closed-form sample values** (first overtake, seeds 0-9):
- `d = 3`, fixed: 22, 20, 20, 21, 23, 31, 15, 30, 27, 21 (median 21.5; slope 4.501).
- `d = 3`, both learn: 18, 31, 22, 15, 20, 21, 14, 27, 25, 19 (median 20.5; slope 1.163).
- `d = 3`, only the incumbent learns: 84, 77, 79, 87, 81, 87, 52, 92, 89, 69 (median 82.5; slope 1.167).
- `d = 2`, fixed: 23, 20, 20, 22, 26, 37, 13, 35, 28, 21 (median 22.5).
- `d = 2`, both learn: 17, 33, 23, 14, 20, 26, 13, 31, 27, 19 (median 21.5).
- `d = 2`, only the incumbent learns: 53, 54, 54, 62, 48, 60, 25, 65, 59, 45 (median 54.0).

The rival stays ahead after first overtaking in 10, 9, 10 seeds at `d = 3` and 9, 9, 9 at `d = 2` (fixed,
both, incumbent). So the closed forms already satisfy (e1)-(e4) as defined. The run tests whether the
arrangement realizes them.

**Construction, fixed before code.** Test (b)'s construction: the Shahshahani simplex chooser on
`{h0, h1}` tensored with Euclidean trust boxes `ell = log Pi`, one per learned (hypothesis, source)
pair. That is 4 boxes in "both learn" and 2 in "only the incumbent". Each box's force is scaled by its
hypothesis's share, and the potential is `U = -sum_i p_i K_i(ell) + T H(p) + B sum p`. Totals change
with every B draw, so the arrangement is rebuilt at each B count, warm-started from the previous rest.
The chooser and trust parameters enter as arguments of one jitted step, so the build is traced, not
recompiled. Trust-box response `eta = 1/n_X` (rate `p_i / 2` near rest), and `eta = 1` when
`n_X = 0`. Chooser `dt = 0.01` for 5000 steps at B count 0 (cold start from `p = (1/2, 1/2)`, all
`ell = 0`), then `dt = 0.1` for 300 steps per B count. `B = min_i K_i* + 5T`, a constant per B count
computed from the closed-form rest costs, which keeps `dt T (log Z - 1)` inside `(0, 2)` (choice.py's
CONDITION). The losing hypothesis's trust converges at a rate scaled by its share, so it may lag. Its
effect on the share is second order, because `K_i` is stationary in `ell` at rest.

## 7. Second round, test (f): a community against one mind holding every datum (predictions, before code)

David, 2026-10-01, asked what scientists send each other: "Scientists doing science together send each
other their conclusions, and then they um, test with local data on their own to see if they agree with
those conclusions. I think that's called um, replication of an experiment. So I suppose you could say that
the scientists send each other um, summaries of the data often, and that's kind of what a paper would be.
as well as the conclusions about what those summaries are telling them." Design and predictions by Claude
Opus 5.5, 0fe746c1, from that picture, before any code for (f); results to §5.6.

**Setting.** `m` holders on a ring (also all-to-all), each running its own local experiments: one draw per
step from the true world (Gaussian hypotheses `N(mu,1)` on a grid of `n`, truth on the grid, and a
misspecified variant off it). Each step a holder may publish a **paper** to its neighbours: a **summary**
(its sufficient statistics, the count and sum of draws) and a **conclusion** (the hypothesis it now weights
most). Benchmark: **one mind** holding every holder's draws, over the full grid.

**Arms** (same draws, seeds, `m`, `n`, steps in every arm):
- **summaries with provenance:** a paper carries only its author's own new draws' statistics; a reader adds
  each paper once.
- **summaries without provenance:** a paper carries everything its author has accumulated, read papers
  included; a reader adds it (no record of whose data it is).
- **conclusions with replication** (David's picture): each holder can hold at most `K < n` hypotheses at
  upkeep (test (b)); starting sets are disjoint slices of the grid, so no holder starts holding the truth
  unless its slice contains it. A received conclusion enters the reader's held set and is kept only if the
  reader's next `r` local draws confirm it against its current best by more than the upkeep (replication);
  otherwise it is dropped.
- **conclusions without replication:** as above, but a received conclusion is kept and weighted as if
  confirmed (test (c)'s "take as evidence").
- **one bounded mind:** a single holder with `K` hypotheses and every draw (the breadth control).

**Predictions.**
- (f1) Summaries with provenance: every holder's belief equals the one mind's on the data that has reached
  it; once every paper has reached everyone, equal to the pooled mind to round-off. Sufficient statistics
  add.
- (f2) Summaries without provenance: each holder is more confident than the pooled mind at every step
  (data counted many times); over
  20 seeds, some lock onto a wrong hypothesis when early draws mislead; the count is above the
  with-provenance arm's (which is 0 when the truth is on the grid).
- (f3) Conclusions with replication: when the truth lies in some holder's starting slice, every holder ends
  holding it (weight > 0.99) in 20 of 20 seeds, reaching each holder after roughly its ring distance from
  the first finder times `r`; when it lies in no slice, never (test (d)'s gap at the community's scale);
  the one bounded mind finds it only if its own `K` contains it.
- (f4) Conclusions without replication: wrong conclusions spread; over 20 seeds the count ending on a wrong
  hypothesis is above zero and above (f3)'s, at the same settings.

**Mechanism** (the claim being tested): pooling summaries reproduces one mind's *evidence*; passing
conclusions extends a community's *hypotheses* beyond any one holder's reach, and replication is the
selection that keeps the extension honest. **What it cannot show:** that real science runs this way; the
arms are stipulated, and "replication" here is testing a received conclusion on new local draws, not
re-running a published protocol.

**Numbers fixed before code** (the run's, Claude subagent of 0fe746c1, 2026-10-01). **Setting.** `m = 8`
holders; ring (each hears its two neighbours) and all-to-all. Grid `n = 20`, `mu = 0, 0.5, ..., 9.5`.
Truth `mu = 5.0` (index 10) in the main variant. Truth `mu = 8.5` (index 17) is the variant "in no
holder's slice". A misspecified truth `5.2` (KL-closest 5.0) is run for (f1) and (f2) only, as a side
report. Seeds 0-19: `rng = default_rng(s)`, draws `rng.normal(truth, 1, size=(T, m))`, holder `j`'s
local draw at step `t` is `[t, j]`. The one mind is fed the same draws, so the same draws reach every
arm, wiring and variant.
**Summaries arms** ((f1), (f2)): `T = 60` draw steps, then `D = 4` steps with no draws (the ring's
diameter; all-to-all needs 1). A step is (i) every holder adds its new draw, (ii) every holder publishes,
(iii) every reader adds what it received. *With provenance* a paper is `(author, step, count 1, sum x)`.
Holders pass on every paper they hold, and a reader adds each paper once (de-duplicated by author and
step). *Without provenance* a paper is the author's whole accumulated `(count, sum)` after (i), and a
reader adds it. Counts at step 1-5 on the ring are 3, 12, 39, 120, 363 against the pooled 8, 16, 24,
32, 40, and at step 60 `6.4e28`. All-to-all: 8, 72, 584, ... (`1.7e54` at step 60). So on the ring a
holder holds fewer draws than the pooled mind at steps 1 and 2. Belief over all 20 hypotheses, log-weight
`y_i = mu_i S - N mu_i^2 / 2` from `(N, S)`, dropping the term common to all `i`.
**Conclusions arms** ((f3), (f4)): `K = 2`; holder `j`'s starting slice is `{2j, 2j+1}`, covering
indices 0-15. The truth (index 10) is in holder 5's slice in the main variant and in no slice in the
other. Each holder keeps its own local `(N, S)`; a hypothesis entering its held set is evaluated on all
of its local draws (the statistics suffice). Its belief is the softmax of `y` over its held set. A holder's
**conclusion** is its held hypothesis of largest weight; each step it is sent to the neighbours. A received
conclusion not already held or on trial enters:
- *with replication:* on trial for the reader's next `r = 5` local draws. It is kept iff
  `sum over those draws of [log p(x|h) - log p(x|best)] > c r`, upkeep `c = 0.1` per step, so threshold
  0.5 nats, `best` being the reader's conclusion when the trial began. If kept, the weakest of the `K+1`
  is dropped; otherwise the newcomer is dropped and may re-enter the next time it is received.
- *without replication:* kept at once. Every receipt adds a credit `gamma = c r = 0.5` to its log-weight
  (credited with what a passing replication would have had to show), and repeated receipts add again, as in
  test (c)'s "take as evidence". The weakest of the `K+1` is dropped.

`T = 150` steps. **Upkeep enters only as the threshold `c r` and the cap `K`; test (b)'s chooser is not
used.** The held set changes by discrete events, which no arrangement here can express (no changeable
interfaces). So the bookkeeping (papers, held sets, trials) is plain code, and the beliefs alone are the
arrangement: each step, (a)'s population drawing, `Y` translated by the step's log-likelihood increment
(Euclidean response, exact constant forcing), one jitted traced build. A translation at duration 1 has
no step-size limit, which is (e)'s lesson applied from the start. **The bounded mind:** `K = 2` with every
holder's draws each step, starting slice `{0, 1}` (no truth) and `{10, 11}` (truth); it receives no
conclusions.
**Definitions.** *Holding the truth*: weight on the truth over the held set `> 0.99` at the final step.
*Ending on a wrong hypothesis*: weight `> 0.99` on one non-true hypothesis at the final step (in the
misspecified variant, "true" means the KL-closest 5.0). A seed counts if any holder ends that way, and
the all-holders count is reported beside it. *Arrival* at holder `j`: the first step its weight on the
truth exceeds 1/2. The *first finder* is the first holder to arrive.
- (f1) holds iff (i) at every step every holder's belief equals the one mind's on exactly the draws it
  holds, to `1e-12` in weights; (ii) after the `D` extra steps, every holder equals the pooled mind on all
  `8 T` draws, to `1e-12`; (iii) 0 of 20 seeds end on a wrong hypothesis (truth on the grid).
- (f2) holds iff every holder's entropy is `<=` the pooled mind's at every step `t >= 1`, and the count
  ending on a wrong hypothesis is above the with-provenance arm's.
- (f3) holds iff, truth in holder 5's slice, every holder holds the truth in 20 of 20 seeds (ring and
  all-to-all), and on the ring the least-squares slope of median arrival delay (arrival minus the first
  finder's) against ring distance from the first finder lies in `[r, 2r] = [5, 10]`. With the truth in no
  slice: 0 of 20 seeds have any holder holding it. The bounded mind holds the truth with slice
  `{10, 11}` in 20 of 20 and with `{0, 1}` in 0 of 20.
- (f4) holds iff, at the same settings, the count of seeds ending on a wrong hypothesis is above 0 and
  above (f3)'s.

I computed no outcome counts before code: apart from the draw counts above, (f) has no closed form short
of running it.

## 8. Third round: the tests natively in the book's mathematics (predictions, before code, 2026-10-01)

David, 2026-10-01, after the summary: "let's run the tests natively in our own mathematics." Audit,
design and predictions by Claude Opus 5.5, 0fe746c1, before any code for this round; results to §5.7.

**Audit of rounds one and two** (read from `science.py`):
- (a) native: holder boxes composed (`parallel_arrangements`), run by `Phi`.
- (b), (e) native: `choice.py`'s chooser tensored with trust boxes, closed, run by `Phi`. But the data enter
  as constants in the cost (mean squares), not as draws arriving on a wire from a world.
- (c) half native: one closed arrangement whose potential holds every holder; accounts are read through
  `stop_gradient`, the world's data through a clock coordinate indexing a table. Not a wiring of holder
  boxes, and no world box.
- (d) half native: one arrangement whose response is rebuilt per datum.
- (f) not native: `translate_beliefs` is an arrangement, but papers, provenance, held sets and replication
  trials are Python bookkeeping around it.

**Native means:** every part is a box with ports (a world box that emits draws; holder boxes; a journal box
where one is wanted); values go forward on wires; a reader's cost reads what arrives and pushes nothing back
(an account is a value sent forward, plan §4.1); the community is the composite (`compose_graph` or the
library's wiring), run by `Phi`. No Python decides anything between steps.

**Design for (f), changed from what was told David.** The summary said a native replicating community "would
take (b)'s keep-or-drop choice placed inside each member." I propose instead, as more native and simpler:
each holder is `choice.py`'s simplex chooser over all `n` hypotheses, starting with its weight on its own
slice and exactly zero elsewhere (support is what it holds; the replicator never creates support). Its
forcing reads two ports: its local draw (fitness = log-likelihood, selection) and its neighbours'
conclusions (a sharpened readout `softmax(y / tau)`, small `tau`). The two arms differ only in how the
conclusion enters the cost:
- **conclusion as a seed** (replication): the term `eps sum_h c_h log p_h`, whose forcing under the
  Shahshahani response is `eps (c - p)`: a little weight moved onto the received conclusion, after which
  the holder's own draws decide whether it grows (test (d)'s mutation, driven by a paper).
- **conclusion as evidence** (no replication): the term `alpha <c, p>`, a fitness bonus, counted like data.
Summaries: **journal** arm, each holder's out-port carries only its current draw's statistics, a journal box
accumulates them once (publication as an inscription), each holder reads the journal; **no provenance**,
each holder's out-port carries its accumulated statistics and neighbours add them.

**Predictions.**
- (n1) (c) rebuilt as `m` holder boxes and a world box composed by wiring gives round two's trajectories
  to round-off (< 1e-10), in both arms; the composite equals the monolithic box.
- (n2) Journal arm: a holder reading the journal equals the pooled mind on the data published so far, to
  round-off. No-provenance arm: the same trajectories as round two's bookkeeping version (same arithmetic),
  hence the same count, 7 of 20 seeds locked wrong.
- (n3) Seed arm, truth in one holder's slice: every holder's top hypothesis is the truth at the final step
  in 20 of 20 seeds (ring and all-to-all); arrival time grows with ring distance, roughly
  `log(1/eps) / (fitness gap)` per hop. Truth in no slice: 0 of 20, settling on the KL-closest hypothesis
  in the union of slices (round two: 7.5).
- (n4) Evidence arm at the same settings: some holders end on a wrong hypothesis, count above zero and above
  the seed arm's (round two: 11 and 12 of 20).
- (n5) The seed and evidence forcings are each exact (the costs above are functions); no non-exact forcing
  is needed anywhere in (f).
**Risk stated before code:** the Shahshahani step is the continuous replicator's Euler step, not the exact
multiplicative Bayes step, so with full log-likelihoods per draw it can overshoot the simplex; step sizes
must be fixed from the largest per-draw fitness gap before the run (round two's lesson). Counts may then
differ from round two's even where the mechanism is the same; that is reported, not repaired.
**What it cannot show:** more than round two about real science; it shows whether the book's own grammar
carries the mechanisms, which is David's question for this round.

**Numbers fixed before code (stage 1)** (the run's, Claude subagent of Claude Opus 5.5, 0fe746c1,
2026-10-01; a numpy script that builds no arrangement, before any native construction code).
*Construction, read off `arrangement.py` before code.* A box `I -> <in | out>` has readout
`out_f(q)`: a function of its state alone, never of what arrives this step. So a value can leave a box
at the earliest one step after it arrives, unless the box holds it in its state. The world is a box
with no input: state a clock (exact forcing `d(clock)`, so `+1` per step), readout the table row at the
clock, a lookup and not a smooth map of the clock. A reader's forcing is the partial differential of
its own cost in its own state, with component **zero on its in-ports**, so composition pulls nothing
back onto the sender. The wiring is `compose_seq(tensor(world, holders, ...), w)` with `w` the static
arrangement `R^-` of a function from in-port coordinates to out-port coordinates (`graph_wire`'s
construction, which needs one port dimension throughout; the ports here differ).
*(n1).* Test (c)'s settings unchanged (§3): `n = 5`, `m = 10`, ring and all-to-all, seeds 0-19 drawn as
in round two; the world emits the draws `x_{t,j}` and each holder computes `log p(x|h)` itself. Grid:
`beta = 1`, `alpha in {0, 0.1, 1, 10}`, `T = 100`, both arms; plus (c1) (`alpha = 0.5`, `beta = 0`,
`T = 400`) and (c2) (`alpha = 1`, `beta = 0`, `T = 100`). Timing: every readout and every forcing is
evaluated at the current state, as in the monolithic box, so step `t` of the composite should be step
`t` of round two with no shift. **Holds iff** at every step `max |Y_native - Y_mono| <= 1e-10 max(1,
max |Y_mono|)` (relative, because the trajectories reach `19^100 = 7.5e127` (agree, `alpha = 10`,
ring), `10.1^100 = 2.7e100`, `2^100 = 1.3e30`), and the round-two classification of every seed is
reproduced.
*(n2).* Test (f)'s summaries settings unchanged (§7): `m = 8`, grid of 20, truth 5.0 (and the side
variant 5.2), seeds 0-19, `T = 60`, extra steps 4 (ring) and 1 (all-to-all). Round two publishes a
holder's stock **after** its own draw of the same step, a readout of the current input, which a box
cannot do. Closest native thing: a **register** in each holder's state (a pending `(count, sum)`
that steps to whatever the world sent, forcing `x_in - pend`), published and counted one step later.
With it the arithmetic is round two's with a shift: native state index `s + 2` is round two's step `s`.
The **plain** box without a register publishes its stock before its draw (the "lag" variant): counts at
steps 1-5 for holder 0 are ring 1, 4, 13, 40, 121 and all-to-all 1, 9, 73, 585, 4681, against round
two's 3, 12, 39, 120, 363 and 8, 72, 584, 4680, 37448. Lock counts recomputed in numpy (seeds ending
with a holder above 0.99 on a wrong hypothesis, any / all): **round two's timing 7/7 of 20 (ring and
all-to-all) at truth 5.0, 9/9 at 5.2; the lag variant the same, 7/7 and 9/9.** No-provenance holds iff
native weights equal round two's at every step to `1e-10` and the counts are 7 and 9 in both wirings
(`|Y|` reaches `7.6e31` ring, `2.1e56` all-to-all, so `Y` is compared relatively, `1e-12`). Journal:
a holder's out-port carries its registered draw `(1, x)`, the journal box adds what arrives (forcing
the sum of its in-ports), a holder's belief steps to the journal's log-likelihood (cost
`1/2 |y - stats(J)|^2`, a translation onto it). So native state index `s + 3` holds the pooled mind on
draw rows `0..s`, and `T + 2` steps reach all 480 draws. **Holds iff** at every step every holder's
weights equal the pooled closed form `weights(stats_inc(8 k, sum of rows < k))` for the `k` rows
published, to `1e-12` (`|Y| <= 6.6e3`, so round-off is about `1e-12` in `Y`), and 0 of 20 seeds end on
a wrong hypothesis at truth 5.0.

**Numbers fixed before code (stage 2)** (the run's, Claude subagent of Claude Opus 5.5, 0fe746c1,
2026-10-01; numpy only, before any stage-2 construction code). David approved the seed design ("yes
that sounds great").
*Settings.* As §7's conclusions arms: `m = 8`, grid `n = 20` (`mu = 0, 0.5, ..., 9.5`), holder `j`'s
starting slice `{2j, 2j+1}` with weight 1/2 each and exact zeros elsewhere; truth 5.0 (index 10, holder
5's slice) and 8.5 (index 17, in no slice); ring and all-to-all, the neighbours' conclusions averaged
(`W = A / degree`, as test (c)); seeds 0-19, draws `default_rng(s).normal(truth, 1, (150, 8))`, the
same in every arm. `eps = 0.05`, `alpha = 0.5`, `tau = 0.05` (readout `c = p^{1/tau} / sum`, the
sharpened distribution, a function of the state, so no register is needed). Fitness of a draw:
`f_h = log p(x|h) - min_h log p(x|h) + 1` (a constant per draw, which the replicator ignores, chosen so
that `<p, f> >= 1` and the simplex is attracting: `choice.py`'s CONDITION). *Step size.* The largest
per-draw fitness gap over all 20 seeds is 40.6 (truth 5.0) and 73.8 (truth 8.5); Euler weights stay
positive iff `dt (gap + alpha + 1) < 1`; I fix `dt = 1/160` in every arm and variant
(`dt x 75.3 = 0.47`), and the world holds each draw for 160 steps, so 150 draws are 24,000 steps of
`Phi` and one draw carries the weight of one draw. *Definitions.* "Top hypothesis is the truth": the
holder's largest weight at the last step is on the truth. "Ends on a wrong hypothesis": weight `> 0.99`
on one non-true hypothesis at the last step (§7); a seed counts if any holder does, all-holders count
beside it. Arrival at holder `j`: first draw after which its weight on the truth exceeds 1/2; delay
against ring distance from the first finder, least-squares slope of the median.
*Analysis before code, which changes what the predictions can mean.* (i) **The seed is singular on the
face where it is needed.** `eps sum_h c_h log p_h` is `-inf` wherever `p_h = 0`, and its differential
`eps c_h / p_h` is infinite there; the Shahshahani sharp multiplies it by `p_h = 0`, which is NaN in
floating point. Run 1, as designed, should therefore give NaN at the first step in the seed arm.
(ii) **No finite forcing can plant one under the Shahshahani response.** The sharp's `h` component is
`dt p_h (xi_h - <p, xi>)`, zero when `p_h = 0` for every finite `xi`. More generally: a response under
which every fitness `d<f, p>` leaves the faces invariant kills the normal component of every finite
covector, since the `f` span all covectors. So keeping selection exact (data as `d<f, p>`) and planting
a seed by a finite forcing are incompatible, and (n5) as stated is predicted **refuted**: the native
seed needs a non-exact piece somewhere. (iii) **The evidence arm as specified cannot spread anything.**
`alpha <c, p>` is a finite fitness bonus, so a holder can never weight a hypothesis it does not hold.
Every holder but 5 should end on its own slice's best (index `2j+1` for `j < 5`, `2j` for `j > 5`) above
0.99, so (n4)'s count should be 20 of 20, met for a reason other than wrong conclusions spreading.
*Second run, decided now and disclosed as such.* Euclidean response on the mean chart, the replicator
field `p ⊙ (f - <f, p>)` as the data forcing (non-exact: its Jacobian `diag(f - f̄) - p f^T` is not
symmetric), the seed as the exact `d(eps (<c̄, p> - |p|^2 / 2))`, whose step is §8's `eps (c̄ - p)`.
Its dynamics are §8's design exactly; it differs from run 1 only in which piece is exact. Run in the
seed arm, the evidence arm (fitness `f + alpha c̄`), and a side arm **plant and count** (both terms,
the analogue of round two's "kept at once and credited"). **(n3) holds iff** (seed arm, truth 5.0) every
holder's top hypothesis is the truth in 20 of 20 seeds, ring and all-to-all, and the ring's slope of
median delay against distance lies in `[12, 48]` draws, around `log(1/eps) / KL(5.0 || 4.5) = 3.0 /
0.125 = 24`; with the truth in no slice, no holder's top is the truth in 20 of 20, and every holder's
top is 7.5 (index 15, KL-closest in the union of slices) in 20 of 20. **(n4) holds iff** the evidence
arm's count of seeds ending on a wrong hypothesis is above 0 and above the seed arm's, at the same
settings. Also reported: the minimum weight and `max |sum p - 1|` over every step, any negative weight
being a finding.

*Added 2026-10-01 by Claude Opus 5.5, 0fe746c1, to §5.2's reading:* David rejected reading the `-1/2 log Pi_B`
discount as his "account for that failure some other way, which is work." His picture: each anomaly is
explained by a fresh, specific excuse ("he was in a hurry that day," then "vision problems") chosen among
those not yet ruled out, and the work is finding one; a blanket discount of the source is not it, and when the
idea is about the source itself, discounting the source is the update. Test (b)'s results stand; its reading
does not.

## 9. Fourth round, test (g): explaining away by building excuses from figures (predictions, before code)

David, 2026-10-01, rejecting the reading of test (b)'s discount as his "work": "Suppose I think that Bob is
trustworthy, and then Bob does something that looks untrustworthy, and I say, okay, yes, but that's just
because um, he was in a hurry that day. And then one day Bob does something else untrustworthy, and I know he
wasn't in a hurry … vision problems … I have to do a lot of work to figure out what I haven't already ruled
out … explaining the data away um, so that it doesn't update the model is also work." On how an excuse is
made: "you construct them using the library of figures or figurines … you put this and that and this there
and uh, attach them to Bob. And then once you do, Bob acts a different way … the cost is to arrange all those
figures … such that … you would see the behavior." On a finite list of excuses: "I don't think finite list of
excuses is very good." Then: "Let's just test this one before we move on." Design by Claude Opus 5.5,
0fe746c1, before any code; results to §5.8. Native, as round three: boxes, wires, `Phi`, nothing decided in
Python between steps.

**Setting.** A reader holds Bob's figure, a trait `theta` (trustworthiness), held by a prior term
`1/2 w (theta - theta_0)^2` (Bob's past record). Each step the world shows Bob's behaviour on the main
channel, and on some steps also side channels (was he late? did he squint?). A **library** of `K` figures
(conditions: hurry, poor sight, a bad day, …), each placed at a distance `delta_k` from the playing field
and each with a footprint: attached to Bob with weight `a_k`, it shifts the predicted behaviour on the main
channel and predicts its own side channels. Excuses are **built, not listed**: any combination of attached
figures, with attachments descended coordinates in `[0, 1]` (a chart where the bounds hold, stated before
code). Per incident the reader descends one cost: main-channel prediction error + side-channel errors (this
is how an excuse is **ruled out**: "we know he wasn't in a hurry" is a side observation its figure gets
wrong) + the prior on `theta` + the **drawing cost** `sum_k delta_k a_k` (David's "the cost is to arrange
all those figures"; the book repo's `notes/ideas-locke-sensorimotor-2026-09-24.md`
§14: draw, arrange, run). **Updating** is moving `theta`.

**Predictions** (qualitative here; the run derives closed-form thresholds for its chosen numbers and writes
them under "Numbers fixed before code (g)" before construction code):
- (g1) While some unexcluded figure has a drawing cost below the cost of moving `theta` by the needed amount,
  an anomaly is absorbed by attaching it, and `theta` moves by less than a stated small bound.
- (g2) Each ruled-out figure forces the next excuse to come from further away: the per-incident explaining
  cost rises along the sequence of anomalies (monotone in the incident index, given the ruling-outs).
- (g3) Once no unexcluded figure is cheaper than the update, `theta` moves. Prediction to test, not assumed:
  the move is **abrupt**, a jump in `theta` at one incident larger than the sum of all earlier moves. If the
  cost is convex in all coordinates the move will be gradual instead; the run says which, and why.
- (g4) Against a rival (redo of test (e) with excuses in place of the blanket discount, same library for
  both sides): symmetric, the rival overtakes at about the same count as test (e)'s; only the incumbent may
  build excuses, the incumbent's advantage is **bounded by its library** (it lasts until its excuses are
  ruled out or too far), unlike (e)'s blanket discount, whose advantage was a fixed ratio.
**What it cannot show.** Running the figures is evaluated inside the cost, not run by an integrator against
a supply; charging the run as work is P34's machinery (an arrangement that runs arrangements), not built.
The library is finite in any run; what is open-ended is the combination, not the figures. Stipulated:
distances, footprints, which side channels are observed when.

**Numbers fixed before code (g)** (the run's, Claude subagent of Claude Opus 5.5, 0fe746c1, 2026-10-01;
numpy, building no arrangement, before construction code).
*Setting.* `K = 8` figures with distances `delta_k = 0.1 k` (0.1 to 0.8). Footprints: figure `k`
attached at `a` shifts the main-channel prediction by `-D a` (each alone can explain a whole anomaly)
and predicts its own side channel at value `a` (late, squinting, ...). `D = 1`, `theta_0 = 1`, `w = 4`,
side-channel precision `v = 100`. `T = 10` incidents, every one an anomaly `y_i = theta_0 - D = 0`. At
incident `i` the side channels of figures `1..i-1` are observed at 0 that day (each is ruled out for
that incident), so incident `i`'s cheapest unexcluded figure is `i`, and incidents 9 and 10 have none.
Each incident has its own attachments `a^{(i)} in [0,1]^K`, all live in the cost from the incident on.
Total cost after incident `t`:
`1/2 w (theta - theta_0)^2 + sum_{i <= t} [1/2 (y_i - theta + D sum_k a^{(i)}_k)^2 + sum_{k ruled out at i}
1/2 v (a^{(i)}_k)^2 + sum_k delta_k a^{(i)}_k]`. *Chart.* `theta` is Euclidean, `eta = 0.05`. Each
attachment is `choice.py`'s chooser on `{attached, not}` written on its free coordinate. Its response is
`dt a (1 - a)`, `dt = 0.05`, the Shahshahani sharp, whose faces are invariant, so `a` stays in `[0, 1]`.
Each attachment starts at `a = 0.01` ("thinkable, not attached"). The world holds each incident for 4000
steps of `Phi`. Stability: `eta (w + T) = 0.7 < 2` and `dt (1/4)(D^2 + v) = 1.26 < 2`.
*Convexity, and my call on (g3), before running.* The cost is a positive quadratic plus a linear drawing
cost on a box, so it is **convex**. Its minimizer moves continuously with every parameter: while
incident `i` is excused, the KKT conditions give a residual `-delta_i / D` and so a move of `theta` by
`delta_i / (D w)`, which grows with the distance of the excuse used. **I call (g3) gradual in the convex
version**: no jump, only a growing move. The principled non-convexity in David's picture is that **an
attachment is made or not** ("you put this and that and this there and uh, attach them to Bob"). The
second version adds the double well `kappa a (1 - a)`, `kappa = 0.5`, per attachment, at the same
settings. There I call (g3) **abrupt**: an excuse is either whole (free for `theta`) or absent, and a
binary excuse beats one unexcused incident (whose `theta` move is `D/(w+1) = 0.2`) iff `delta <
1/2 D^2 w/(w+1) = 0.4`. Which local well the descent reaches also depends on the race between `theta`'s
response and the attachment's.
*Scratch values* (the same steps in numpy). `theta` after incidents 1-10:
- **Convex:** 0.975, 0.925, 0.85, 0.75, 0.625, 0.504, 0.432, 0.381, 0.343, 0.311. The exact convex
  minimizer gives the same to 6e-4. Moves: 0.025, 0.05, 0.075, 0.1, 0.125, 0.121, 0.072, ...
  Excuse attachment at incident `i`: 0.875, 0.724, 0.547, 0.344, 0.115, then 0.
- **Double well:** 1, 1, 1, 0.876, 0.778, 0.700, 0.546, 0.5, 0.462, 0.429. Moves: 0, 0, 0, 0.124, 0.097,
  0.078, 0.155, ... Attachments: 1, 1, 1, then 0. Figure 4 (`delta = 0.4`) loses the race.

**Definitions and verdict rules.** *Excused:* the incident's largest attachment is `>= 1/2` at its end.
*Threshold incident:* the first that is not excused (convex: 4; double well: 4).
- **(g1)** holds iff at every excused incident `theta`'s move is `<= delta_i / (D w) + 1e-3` (0.025,
  0.05, 0.075 at incidents 1-3), every one below the unexcused move 0.2.
- **(g2)** holds iff the explaining cost of incident `i` at its end (drawing plus side-channel cost of
  its own attachments) is strictly increasing over the excused incidents. Scratch: convex 0.087, 0.145,
  0.165; double well 0.1, 0.2, 0.3.
- **(g3)** "abrupt" holds iff the move at the threshold incident exceeds the sum of all earlier moves.
  Scratch: convex 0.1 against 0.15, **refuted** (gradual, as called); double well 0.124 against 0,
  **held**.

The native run must reproduce these scratch values to `1e-6`, the same arithmetic in a different order.
**(g4)** is fixed below, before its code, if the budget allows.

**Numbers fixed before code (g4)** (the run's, same signature; numpy closed forms at rest, no
arrangement, before (g4)'s construction code). *Setting.* Test (e)'s draws, unchanged: `n_A = 20`,
`d in {3, 2}`, seeds 0-9, `a = rng.normal(0, 1, 20)`, `b = rng.normal(d, 1, 150)`, `T = 1`, every trust
fixed at 1. In place of the blanket discount there is **one library for both sides**: `K = 8` figures in
mirrored pairs, footprint `+d` or `-d` on the draw they are attached to, distances `delta_k = 0.5 k
(d^2/9)` (0.5 to 4.0 at `d = 3`, where an unexcused anomaly costs about `d^2/2 = 4.5`). A hypothesis's
anomalous draws are the other source's (A draws for the rival, B draws for the incumbent), taken in order.
At its `j`-th anomalous draw, figures `1..j-1` (both signs) are ruled out. Two variants: **soft**, an
observed side channel at precision `v = 100` as in (g); **hard**, the ruled-out figures absent from that
draw's library. Arms: **none** (test (e) fixed), **symmetric** (both may build excuses), **incumbent**
(only `h0`). `n_B*` = first B count with `K_1 < K_0` at rest.
*Closed forms* (the "none" arm reproduces §6's fixed values exactly, a check on the script):
- `d = 3`, soft: none 22, 20, 20, 21, 23, 31, 15, 30, 27, 21 (median 21.5); symmetric 23, 24, 21, 21,
  22, 29, 14, 29, 17, 21 (21.5); incumbent 30, 34, 34, 34, 36, 41, 23, 41, 38, 35 (34.5).
- `d = 3`, hard: symmetric median 21.5; incumbent 26, 27, 27, 28, 29, 37, 19, 33, 31, 28 (28.0).
- `d = 2`, soft: none median 22.5; symmetric 21.5; incumbent 35.0. Hard: symmetric 22.0; incumbent 29.0.

So, before code: the excuse library protects the incumbent by **+6.5 draws (hard) and +13 (soft)** at
`d = 3`, against (e)'s blanket discount's +61 (82.5 against 21.5). The hard variant's delay over "none"
is at most 7 in every seed, within `K = 8`. The soft variant leaks: a ruled-out figure is still attached
at `d^2/(d^2 + v) = 0.08` on every later draw, a small fixed-ratio discount like (e)'s blanket, so its
delay is not bounded by the library alone.
**(g4) holds iff**: the symmetric median is within 25% of (e)'s fixed 21.5 (d = 3) and 22.5 (d = 2), and
the incumbent-only delay over "none" is at most `K = 8` in 10 of 10 seeds (hard), and below (e)'s
blanket-discount medians (82.5, 54.0). Predicted from the closed forms: held for the hard variant, the
bound refuted for the soft one.
*Construction.* One reader box, wired to a world that emits the B draws one count at a time (1000 steps
per count, 151 counts, count 0 holding only the A draws). The chooser is in the natural chart (library
item 42): `lambda = log(p_0/p_1)`, inverse-Fisher response, `dt = 1`, so `lambda' = K_1 - K_0 - T lambda +
lambda`, the rest logit in one step. Attachments are as in (g): `dt a (1-a)` with `dt = 0.02` (stability
`0.02 x 1/4 x (2 K d^2 + v) = 1.22 < 2`), starting at `a0 = 0.01`. **Two forcings, each disclosed.**
**Unilateral** (primary): each side's attachments descend its own `K_i`, as if each hypothesis had an
advocate. This is not exact jointly. **One potential**
`U = -p_0 K_0 - p_1 K_1 + T H(p)`: exact, but each side's attachments are driven by `p_i dK_i`. I
predict it departs from the closed form in the symmetric arm, because at the start `p_1 ~ e^{-90}`, so the
losing rival builds no excuses: one builds excuses only for what one believes. The native unilateral run
must match the soft and hard closed-form `n_B*` in 10 of 10 seeds per arm; the one-potential run is
reported, with the soft variant at `d = 3` only.

*Correction, 2026-10-01, Claude Opus 5.5, 0fe746c1.* §1 calls "an organism is a hypothesis the world tests" David's
candidate; it was the agent's, to which he said "sure", offered in place of his "organisms are little experiments"
on the ground that "an experiment is chosen for what it will tell you; variation is not chosen." That ground is
withdrawn: David's NITMB talk (book repo `background/20250317-NITMB.tex`, "Bacterial machinery" and "Mate
selection") argues the search is structured ("blind search doesn't build turbines"; "stress triggers a response that
rewires future sensitivity"; mate choice: "No longer blind"), and stress-induced mutagenesis (Galhardo, Hastings and
Rosenberg, *Crit. Rev. Biochem. Mol. Biol.* 42 (2007) 399–435, read in summary) makes the rate and timing of
variation respond to stress, its direction still disputed (Roth's "selective inbreeding," *Genetics* 214 (2020)).
His "organisms are little experiments" stands open, not replaced.
