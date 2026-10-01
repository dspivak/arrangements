# Science at two scales, test (h): bettors paid by being right (predictions, before code, 2026-10-01)

A companion to `SCIENCE-SCALES-PREDICTIONS-2026-10-01.md` (tests (a)–(g)), kept in its own file so that two
runs can write at once. David, 2026-10-01, on test (c)'s "a community with no world does not fall apart": "if
the um, people in there, the bettors, are actually paid by being right, um, and if their payments go to zero,
they die or fall apart … doesn't it um, make them fall apart to be wrong, or don't the ideas fall apart? Like,
wouldn't they rather kill the idea than themselves?" Then, to "Shall I run that payment test, with a hub
network added so who hears whom can show?": "yes." Popper, "Natural Selection and the Emergence of Mind,"
*Dialectica* 32 (1978), §4: "While an uncritical animal may be eliminated together with its dogmatically held
hypotheses, we may formulate our hypotheses, and criticize them. Let our conjectures, our theories, die in our
stead!" Design and predictions by Claude Opus 5.5, 0fe746c1, before any code. Results to §6 of this file.

## 1. Setting (native: boxes, wires, `Phi`; nothing decided in Python between steps)

A world box emits one outcome per step (a race: which of `n` horses wins, from a fixed distribution with one
favourite, seeded). `m` bettor boxes, each with a belief `p_j` on the simplex and a **supply** `s_j >= 0`.
Each step a bettor stakes its whole supply in proportion to its belief at fair odds `n` (proportional,
Kelly-style betting), so after outcome `x` its supply is multiplied by `n p_j(x)`, then it pays an upkeep `c`.
A bettor whose supply reaches 0 is **dead**: it stakes nothing and its account stops (the run states how a
floor at zero is realized natively, e.g. item 53's starving part, and discloses any non-native piece).
Talk between bettors as in test (c): **agree** or **take as evidence**, on a ring, all-to-all, and a **hub**
(one bettor heard by all, the rest on a ring).

**Arms** (same outcomes, seeds, `m`, `n`, `c`, starting supplies and beliefs in every arm):
- **dogmatic:** a bettor's belief never moves (no data term, no talk).
- **updating:** a bettor's belief reads the outcomes (fitness = log-likelihood, test (a)'s update).
- **updating + talk** in each talk rule and each network.
- **hub wrong:** the hub starts sure of a wrong horse; the rest start spread.

## 2. Predictions

- (h1) *Wealth is the likelihood.* With proportional betting and no upkeep, a bettor's supply after `t` races
  is its starting supply times `n^t` times the likelihood its belief gave the outcomes; supply shares across
  silent dogmatic bettors evolve exactly as test (a)'s population, to round-off. (Known as Kelly betting's
  link to Bayes; no citation until checked.)
- (h2) *Dogmatic bettors die with their idea.* With upkeep `c > 0`, every dogmatic bettor whose belief puts
  less on the favourite than the expected-growth threshold dies (closed form: dies iff
  `E[log(n p_j(x))] < log(1 + c / s)`-type condition, the run derives the exact one before code); count each
  way over 20 seeds.
- (h3) *Updating bettors let the idea die instead.* Same settings: far fewer deaths than the dogmatic arm;
  deaths only among bettors whose early losses exhaust supply before their belief moves (a race between
  belief speed and supply). Counts each way.
- (h4) *The echo chamber falls apart together.* With talk taken as evidence at ten times the weight of
  reading outcomes (test (c)'s worst setting), in the seeds where the community locks onto a wrong horse,
  all bettors die (correlated wrongness is correlated death); in the agree rule, total collapse 0 of 20.
- (h5) *Who hears whom shows with a hub.* Agree rule, no outcomes read: the consensus is the
  centrality-weighted average of the starts, with the hub's weight its eigenvector centrality (Golub and
  Jackson 2010, Prop. 1, to be opened before citing), unlike ring and all-to-all where every weight is equal.
  Hub wrong, with outcomes and paid: more seeds lock wrong and collapse than on the ring at the same
  settings.

**Mechanism under test.** Being paid by being right makes a bettor's supply its hypothesis's likelihood, so a
population of dogmatic bettors is a population of hypotheses that dies with them (piece 1), while a bettor
that updates is a mind whose hypotheses die instead (Popper). **What it cannot show:** that real bettors,
scientists or firms are paid this way; proportional betting, fair odds and upkeep are stipulated.

**Numbers fixed before code** (Claude, subagent of Claude Opus 5.5, 0fe746c1, 2026-10-01; from scratch
scripts that sum logs and products in numpy and build no arrangement; written before
`arrangements/science_payment.py` existed).

*World and bettors.* `n = 4` horses, fair odds `4`. Hypotheses `H_i`, `i = 0..3`: "horse `i` is the
favourite", `H_i(i) = 1/2`, `H_i(x) = 1/6` otherwise; the truth is `H_0`, so `pi = (1/2, 1/6, 1/6, 1/6)`.
`m = 10` bettors. Bettor `j` holds weights `q_j = softmax(y_j)` over the four hypotheses (test (c)'s
gauge-free log-weight chart, Euclidean response 1, not the inverse-Fisher mind: §5.1 of the scales note)
and bets its predictive `p_j(x) = sum_i q_{j,i} H_i(x) = 1/6 + q_{j,x}/3`, so `p_j(x) >= 1/6` and one race
costs at most a third of a supply. Starting supply `s_0 = 1`, upkeep `c = 0.02` (and `c = 0` for (h1)).
`T = 200` races. Seed `s in 0..19`: `rng = default_rng(s)`, `Y0 = rng.normal(size=(10, 4))`, then the
outcomes `rng.choice(4, size=200, p=pi)`, one race shared by all bettors; the same in every arm.
*Update* ("updating"): `y_j += beta log H(x)`, `beta = 1`, test (a)'s step. Dogmatic: `beta = 0`, no talk.
*Talk*, test (c)'s forcings on `y_j`, neighbours' accounts `log q_k` arriving forward: agree
`alpha sum_k W_jk (log q_k - y_j)`; evidence `alpha sum_k W_jk log q_k`. *Networks* (row `j` is whom `j`
hears): ring `1/2` to `j +- 1`; all-to-all `1/9`; **hub**: bettor 0 hears the nine others at `1/9`, each
other bettor hears its two ring neighbours among bettors `1..9` and the hub, `1/3` each. Talk weights:
(h4) evidence `alpha = 10` (test (c)'s worst), agree `alpha = 10` as worded and `alpha = 0.5` (below);
(h5) agree `alpha = 0.5` with `beta = 0`; hub wrong: evidence `alpha = 1` and agree `alpha = 0.5`, `beta = 1`.
*Hub wrong:* bettor 0's start is replaced by `y_0 = (0, 10, 0, 0)` (weight `0.99986` on `H_1`), the other
nine as seeded; compared on the hub network and on the ten-bettor ring, same seeds.

*Supply, and the floor.* `s' = max(s n p_j(x) - c, 0)`, from the belief held before the race. Natively:
the supply is a state coordinate with Euclidean response 1 and forcing `max(s n p_j(x) - c, 0) - s`.
**Non-native piece, disclosed now:** the `max` is a kink in the one-form (continuous, not smooth), computed
inside the forcing by `Phi`, not decided in Python. Item 53's starving part (`test_many_parts_wirings`)
needs no floor because its drive is multiplicative in its own amplitude, so zero is a fixed point and
"dead" is a threshold (`|q| < 1e-3`); with a fixed upkeep there is no such smooth fixed point, and a
proportional upkeep (the smooth alternative, not run) would only shift the growth rate. A dead bettor's
belief still moves and its account is still published (the idea outlives its bettor); only its supply stops.

*Definitions.* **Dead**: supply exactly 0 at step 200. **Locked wrong**: at step 200 every bettor's weight on
one and the same wrong hypothesis exceeds 0.99 (test (c)'s). **Total collapse**: all ten bettors dead at 200.

*Closed forms.*
- (h1) With `c = 0`: `s_j(t) = s_0 4^t prod_{k<=t} p_j(x_k)`. For a dogmatic bettor that is its likelihood;
  the shares `s_j / sum s` are test (a)'s population with fitness `p_j(x)` (equivalently, the Bayes posterior
  over the ten bettors' beliefs as hypotheses under a uniform prior). For an updating bettor the product
  telescopes to the marginal likelihood: `s_j(t) = s_0 4^t sum_i q_{j,i}(0) prod_k H_i(x_k)`. Checked to
  relative `1e-12`, every bettor, step and seed, in both arms.
- (h2) Dogmatic, with the floor: write `G_k = 4 p_j(x_k)`, `Pi_k = G_1 ... G_k`. While positive,
  `s_t = Pi_t (s_0 - c sum_{k<=t} 1/Pi_k)`, so bettor `j` is **dead by step `t` iff
  `c sum_{k<=t} 1/Pi_k >= s_0`**, at the first such `t`. The partial sums diverge (death almost surely)
  iff the growth rate `g_j = log 4 + sum_x pi(x) log p_j(x) <= 0` (the truth's own rate is `0.1438`).
  For a belief putting `w` on the favourite and the rest evenly, `g = 0` at `w = 1/4` (and `3/4`, out of
  reach here since `p_j(0) <= 1/2`). The native run is checked against the exact condition, death step by
  death step. **Counts already computed** in numpy: 136 of the 200 bettor-seeds have `g_j <= 0`, and all 136
  are dead by step 158 (so (h2)'s claim holds by arithmetic; the run checks the construction); of the 64 with
  `g_j > 0`, 14 are dead (upkeep against early bad luck).
- (h3) **Counts already computed** in numpy, so not a prediction any longer: updating, no talk, 76 of 200
  dead, against the dogmatic 150 of 200. §2 did not quantify "far fewer"; I read 76 against 150 as about
  half, and the result will say so rather than stretch "far". Its mechanism clause is operationalized as:
  let `tau_j` be the first step bettor `j`'s weight on the truth exceeds 0.99; bettor `j` is dead at 200 iff
  its supply at `tau_j` is below what a truth-dogmatic bettor needs from there, `c sum_{k > tau_j} 1/Pi_k`
  under `p = pi`. Numpy: every bettor reaches 0.99; the rule agrees in 199 of 200; 56 of the 76 die before
  `tau_j`.
- (h4) Evidence step `y += alpha W Y + beta t`: `|y|` grows like `11^t`, `1.9e208` at step 200, still
  finite. Agree at `alpha = 10`: the synchronous step's eigenvalues `1 - alpha(1 - lambda_W)` reach `-19`
  (ring), `-10.1` (all-to-all), `-15.3` (hub), so it diverges (§5.3's "neither"); **I predict this half of
  (h4) fails at `alpha = 10`** as an artefact of the step (beliefs flip each race and bettors bet on the
  flips), and fix `alpha = 0.5` (step eigenvalues in `(0, 0.91]`) as the stable stand-in, both reported.
- (h5) For row-stochastic `W`, the agree step on the gauge-fixed `lam_j = y_j - y_{j,3}` is
  `lam <- (I - alpha(I - W)) lam`, whose limit is `lam_inf = sum_j v_j lam_j(0)` with `v` the left Perron
  eigenvector of `W` (`v W = v`, `sum v = 1`). Ring and all-to-all: `v_j = 1/10`. Hub: by balance at the
  hub, `v_0 = (m-1) r / 3` with `r` each other bettor's weight, so **`v_0 = 1/4`, `v_j = 1/12`**
  (`m = 10`). Slowest decay at `alpha = 0.5`: ring `0.905`, all-to-all `0.444`, hub `0.755`; at step 200
  all below `1e-8`. Checked: every `lam_j(200)` within `1e-8` of `lam_inf`. (Golub and Jackson 2010 not
  opened; not cited.) Hub wrong: I expect the agree rule to give 0 locked wrong on both networks (data
  accumulate while agreement only averages a bounded offset), so (h5)'s "more seeds lock wrong" can hold
  only in the evidence rule.

## 6. Results

Run 2026-10-01 by Claude (subagent of Claude Opus 5.5, 0fe746c1). Code: `arrangements/science_payment.py`
(`bettor`, `market`, `market_run`, `seeds`, `hub`), tests `tests/test_science_payment.py`. Every arm is one
closed composite, `wire_boxes([world, 10 bettors])` from `science_native`, run 200 steps by `Phi` under one
`lax.scan`, vmapped over the 20 seeds; one compiled composite serves every arm (the weights `W`, `alpha`,
`beta`, `c` are values, not structure). Settings exactly as fixed above, the same seeds, outcomes, starts,
`s_0 = 1` and `c = 0.02` in every arm. No NaN, no infinity in any run.

**Runs, in order.** One smoke run (dogmatic and updating, `c = 0.02`: 150 and 76 dead, the numpy counts).
**Recorded run 1** of the test file, every test asserting its prediction as fixed: 9 failed, 10 passed
(log kept in the run's scratch). The failures were (h1)'s tolerance, six (h4) agree cases and (h5)'s agree
hub-wrong case. I then marked those eight cases strict xfail with their reasons and added two tests after the
fact, disclosed below as post hoc: (h1) against exact rational arithmetic, and (h4)'s no-talk collapse count.
**Re-run** with nothing else changed: 11 passed, 8 strict xfail, 6 s.

**(h1) Failed its fixed check; the identity holds in an exact check made after the run.** At the fixed
tolerance, native supply against the float64 closed form `s_0 exp(t log 4 + log L)`: worst relative error
**`1.14e-12` against `1e-12` fixed** (dogmatic arm; updating `1.8e-13`). The shares of the silent dogmatic
bettors equal test (a)'s population `softmax(sum log p_j(x))` to `1e-12` at every step of every seed. Diagnosis
after the run: the gap is the reference's. Against the identity computed in exact rational arithmetic from the
same float64 predictives, the native supply is within `9.1e-14` (dogmatic) and `1.2e-14` (updating, against
the marginal likelihood `4^t sum_i q_{j,i}(0) prod_k H_i(x_k)`) over all 20 seeds; the float64 closed form,
which takes `exp` of a cumulative sum of logs near 300 in size, carries the `1e-12`. The fixed check stays
failed, as a check that set a tolerance its own reference could not meet.

**(h2) Held.** Dogmatic, upkeep `0.02`: **136 of the 136** bettor-seeds whose growth rate is `g_j <= 0` are
dead by step 200; of the 64 with `g_j > 0`, **14 are dead and 50 alive**. Every one of the 200 bettor-seeds dies,
or not, at exactly the step the closed form `c sum_{k<=t} 1/Pi_k >= s_0` names. In all: 150 of 200 dead.

**(h3) Half held.** Same settings: **dogmatic 150 of 200 dead; updating 76 of 200**. That is about half, not
"far fewer" (the counts were computed before code, and I said then I would not stretch "far"). The mechanism
clause held: in 199 of 200 bettor-seeds, a bettor is dead at 200 exactly when its supply, at the step its
weight on the truth first passes 0.99, is below what a bettor sure of the truth needs from there; 56 of the 76
were already dead at that step. Every updating bettor's belief reaches the truth; 76 of them are dead when it
does or soon after. The race is between belief speed (about 10 races to 0.99) and supply (one race costs up to
a third).

**(h4) Its evidence half held; its agree half refuted.** Evidence at `alpha = 10`, locked wrong and, of those,
seeds with all ten dead, same settings: **ring 18 locked, all ten dead in 18 of 18; all-to-all 18, 18 of 18;
hub 17, 17 of 17.** In the seeds that did not lock wrong, the community locked on the truth in 2 of 2 (ring)
and the dead were 5 and 3 (ring), 0 and 0 (all-to-all). Agree, total collapse, same settings: at `alpha = 10`
**20 of 20 on every network** (predicted before code: the synchronous step diverges, beliefs flip each race,
every bettor dies); at the stable `alpha = 0.5`, **ring 3, all-to-all 4, hub 4 of 20**, against the
prediction's 0, with 0 locked wrong. *Found after the run* (post hoc, a test added): **with no talk at all,
updating bettors collapse in 3 of 20 seeds (1, 9 and 12)**, the same three that collapse under agree (all-to-all
and hub add seed 16). The mechanism the prediction missed: every bettor faces the same races, so an early run
of outsiders kills a whole community of bettors that start spread, whatever they say to each other.
Correlated wrongness is correlated death (evidence: 18 against no talk's 3), but a shared world is already
correlated death without any wrongness held in common. Total dead bettors: no talk 76, agree 68, 68, 62, evidence
188 (ring) of 200.

**(h5) Held for the consensus and for the evidence rule; the agree hub-wrong case held for collapse and not
for locking, as stated before code.** Agree, no outcomes, `alpha = 0.5`, step 200: every bettor's `lam_j`
within **`1.3e-14` (hub), `1.1e-14` (all-to-all), `2.9e-9` (ring)** of `sum_j v_j lam_j(0)`, with `v` the left
Perron eigenvector, checked equal to the derived `v_0 = 1/4`, `v_j = 1/12` (hub) and `1/10` (ring,
all-to-all). The hub's consensus sits `0.51` (max over hypotheses and seeds, in `lam`) from the plain mean;
the ring's and all-to-all's at 0. Hub wrong, paid, same seeds and settings, ring against hub:

| rule | locked wrong, ring | hub | total collapse, ring | hub | dead bettors, ring | hub |
|---|---|---|---|---|---|---|
| evidence `alpha = 1` | 17 | 20 | 18 | 20 | 188 | 200 |
| agree `alpha = 0.5` | 0 | 0 | 3 | 9 | 86 | 121 |

**Which could have come out otherwise.** Little of (h1), (h2), (h3) and (h5)'s consensus: they follow from
the arithmetic, and their counts were in numpy before code, so they check that the wired boxes and the `max`
floor realize it (they could have failed on timing, since a box bets with the belief it holds before the race,
and did not). Open before the run: the (h4) and hub-wrong counts. Not foreseen: that the agree rule's collapses
are the no-talk collapses, caused by the shared races; and that putting one sure-wrong voice at the hub
triples agree's collapses (3 to 9) without anyone locking wrong, which is the hub's `1/4` weight dragging every
bettor's bets toward the wrong horse while data pull them back. This last is my reading of the counts; I did
not trace it bettor by bettor.

**Reasonable or weird** (my judgment, not evidence). *A person:* a dogmatist who stakes everything on a fixed
view each time and pays a fixed cost of living dies when the view is worse than chance-breakeven: reasonable
as a parable, weird as a person, since no one stakes their whole supply on every event. *A scientific
community:* reasonable and recognizable that a field that counts colleagues' conclusions as data locks wrong
and goes under together, and that one loud wrong voice at the centre makes it worse; weird that being right
is paid per event at fair odds with no lag, while real credit arrives late and goes partly to the wrong
people. *A market or firm:* the closest fit, since proportional betting at fair odds is a Kelly trader and the
upkeep a fixed cost; reasonable that correlated positions bankrupt a whole desk together, and that a shared
shock does so without any shared belief; weird that the odds are fair (no house, no price moved by the bets).
*A population of organisms:* the dogmatic arm is a population of fixed types under selection with a fixed
death rate, and (h1)'s shares are test (a)'s population exactly: reasonable. Updating bettors have no
population reading I would defend, since an organism does not revise its genotype from what it meets. *A
cell:* none I would defend beyond the population reading.

**What this does not establish.** That anyone is paid this way, or that ideas dying "in our stead" (Popper)
saves bettors in general: here updating halves the deaths, and a shared world kills whole communities that
never shared a wrong belief. Nothing beyond `n = 4`, `m = 10`, `c = 0.02`, `s_0 = 1`, `T = 200` and these
networks; the floor is a kink in the one-form, not a smooth native death; and dead bettors keep talking.

## 7. Correction: the dead fall apart (predictions, before code, 2026-10-01)

David, on "dead bettors keep publishing their beliefs": "please change the weird thing. That doesn't make any
sense … if you're dead, um, you should have fallen apart so that the very parts you have wouldn't, you'd, you
wouldn't be able to work." Design and predictions by Claude Opus 5.5, 0fe746c1, before any code for this fix.

**The fix, in the book's terms** (a form that plays is held only while something drives it, Ch 1 `sec.attuned`):
a bettor's belief is held by its supply. (i) What it publishes is gated by its supply: the account on its
out-port carries a factor `g(s)`, smooth, `g(0) = 0`, `g(s) -> 1` for supply well above the upkeep, so a listener's
forcing from a dead bettor is zero and listeners' talk weights are taken over the living only. (ii) Its belief
dissolves without supply: a term pulling its belief toward uniform with weight `(1 - g(s)) r`, so with no supply
its organization relaxes away at rate `r`. Native, as before; the run states whether (i) and (ii) are exact.

**Predictions.** (h6) After death, every listener's forcing from the dead bettor is below 1e-12, and the dead
bettor's belief is within 1e-3 of uniform after `5/r` steps. (h4') Under taking-as-evidence, whole-community
collapse counts are within 2 seeds of §6's (18, 18, 17), since a community locks before its first death. Under
agree, collapse counts within 2 of §6's. (h5') Hub wrong: once the hub dies, the living reach consensus on the
ring's equal weights; locked-wrong count on the hub falls below §6's 20 if the hub dies before the lock, and
stays if it dies after; the run reports how many hubs die before the others lock. (h1)–(h3) have no talk and are
not rerun, except that (ii) is checked not to change any dogmatic or updating death step.

**Numbers fixed before code (§7)** (Claude, subagent of Claude Opus 5.5, 0fe746c1, 2026-10-01; derived on
paper, before any §7 construction code). Every §2 setting is unchanged: same seeds, outcomes, starts, `s_0 = 1`,
`c = 0.02`, `T = 200`, networks and talk weights.

- *The gate.* `g(s) = psi(s/s_1) / (psi(s/s_1) + psi(1 - s/s_1))`, `psi(x) = e^{-1/x}` for `x > 0`, else 0:
  smooth (`C^infinity`), `g = 0` for `s <= 0`, `g = 1` for `s >= s_1`, every derivative 0 at `s = 0`.
  **`s_1 = c/2 = 0.01`.** Why that value: since `p_j(x) <= 1/2`, a race at most doubles a supply, so a bettor
  with `s < c/2` has `s' <= 2s - c < 0` and is dead at the next step whatever it believes. So `g < 1` only on
  bettors that are dead or dead next step, and **(ii) cannot change any death step, by construction**; the run
  checks it in the dogmatic and updating arms anyway (every death step equal to §6's).
- *(i) Gated account.* A bettor's out-port is `(g(s) log q, g(s))` (`K + 1` coordinates), from its state only.
  A listener averages over the living it hears: `avg_j = sum_k W_jk g_k log q_k / sum_k W_jk g_k`, and 0 when
  that denominator is 0. Agree forcing `alpha (avg_j - y_j)` (0 when no one it hears is alive), evidence
  `alpha avg_j`. **Not smooth:** the division has a jump when the last bettor a listener hears dies (from that
  bettor's `log q` to 0); disclosed, like the `max` floor, which stays.
- *(ii) Dissolution.* `xi_y = g(s_j) [beta log H(x) + talk] - (1 - g(s_j)) r (y - mean(y))`. **A choice beyond
  §7's text, disclosed:** the bettor's own reading and talk are gated by its own `g` too ("the very parts you
  have wouldn't be able to work"); without that, a dead updating bettor would settle where data balance
  dissolution, not at uniform. **Not exact:** the term is the `y`-partial of `(1 - g(s)) r/2 |y - mean y|^2`,
  without its `s` component (which would push on the supply). **`r = 0.1`**, so `5/r = 50` steps.
- *Derived consequences, stated before code.* (h6) first half: a dead bettor's out-port is exactly 0 and
  `g` is flat at 0, so the Jacobian of every listener's forcing in the dead bettor's state is exactly 0; checked
  on the composite's forcing at the final state. (h6) second half: after death the gauge-fixed `lam` contracts
  exactly by `0.9` per step, `0.9^50 = 5.2e-3`, so a bettor is within 1e-3 of uniform 50 steps after death only
  if `|lam|` at death is below about 0.8. **I predict (h6)'s second half fails for most dead bettors**: an
  updating bettor's `lam` grows about 1 per race read, and under evidence `|lam|` is near `11^t`. Counted over
  bettors dead by step 150, each arm. (h5'): "the others lock" = first step at which all nine non-hub bettors
  are above 0.99 on one wrong hypothesis; hub death step = first step its supply is 0; counted before, after,
  never, on the hub network and the ring, evidence `alpha = 1` and agree `alpha = 0.5`.

## 8. Results of the correction

Run 2026-10-01 by Claude (subagent of Claude Opus 5.5, 0fe746c1). Code: `science_payment.gate`, `bettor_gated`,
`market_gated`, `market_gated_run`, `listener_sensitivity`; §7 tests appended to `tests/test_science_payment.py`.
Settings as fixed in §7's numbers and §2's: same seeds, outcomes, starts and weights. §6's numbers stand as
recorded. **Runs, in order.** One smoke run (death steps, both arms). **Recorded run** of the whole file:
4 failed, 25 passed, 11 strict xfail (the 3 xfails of (h6)'s uniform half were marked before the run, as
predicted to fail). I then marked the 4 failures strict xfail with their reasons and added one test after the
fact (the lock among the living, below). **Re-run**, nothing else changed: **30 passed, 15 strict xfail, 21 s**.
Non-native pieces: §6's `max` floor, plus the listener's average over the living, which jumps when the last
bettor it hears dies. (ii) is not exact (the `y`-partial of a cost, without its `s` component).

**(ii) changes no death step: held.** Dogmatic and updating: every death step equals §6's in 200 of 200 each
(150 and 76 dead), as derived (`g < 1` only on a bettor that is dead or dead next step).

**(h6) Not heard: held. Dissolved within 50 steps: refuted, as predicted before code.** The Jacobian of every
listener's forcing in a dead bettor's state is exactly `0.0` over 450, 171, 180 and 450 listener-dead pairs
(evidence 10 ring, agree 0.5 ring, agree 0.5 hub, evidence 10 hub; every fourth seed). Within 1e-3 of uniform
50 steps after death, bettors dead by step 150: **no talk 1 of 76, evidence 10 ring 0 of 183, agree 0.5 ring
0 of 68**, because `|lam|` at death is far above 0.8. Given longer, they do dissolve: under hub-wrong evidence
`alpha = 1` (seed 0, deaths at steps 13-15) every dead bettor reads `0.25` on every horse at step 200.

**(h4') Evidence held; agree held at 0.5, refuted at 10 on the ring and the hub.** Total collapse, same seeds and
settings, §7 against §6: evidence `alpha = 10` ring **17 vs 18**, all-to-all **18 vs 18**, hub **17 vs 17**;
agree `alpha = 0.5` ring **3 vs 3**, all-to-all **4 vs 4**, hub **4 vs 4**; agree `alpha = 10` ring **12 vs 20**
(refuted), all-to-all **18 vs 20**, hub **17 vs 20** (refuted). In the diverging agree step, a dead bettor's
falling silent removes its flips from its neighbours' averages, and some communities then survive; my
reading, not traced. *A measurement change, found in the run:* §6's "locked wrong" reads every bettor's belief
at step 200, and under §7 the dead have dissolved, so end-state locks fall (evidence 10: ring 5, all-to-all 12,
hub 9, against 18, 18, 17). Post hoc, "ever locked wrong among the living": **ring 17, all-to-all 18, hub 17
of 20, and every one of those communities ends all dead** (17 of 17, 18 of 18, 17 of 17); §6 by the same measure
18, 18, 17.

**(h5') Refuted as fixed; the hub-death count is the finding.** Hub wrong (bettor 0 sure of `H_1`), same seeds
and settings, hub network against ring:

| rule | locked wrong at 200, ring / hub | ever locked among living, ring / hub | total collapse, ring / hub | bettor 0 dies before the others lock / after / they never lock / it survives, ring | same, hub |
|---|---|---|---|---|---|
| evidence `alpha = 1` | 0 / 0 (§6: 17 / 20) | 12 / 20 | 13 / 20 (§6: 18 / 20) | 1 / 8 / 11 / 0 | 0 / 20 / 0 / 0 |
| agree `alpha = 0.5` | 0 / 0 | not run | 3 / 9 (§6: 3 / 9) | 0 / 0 / 14 / 6 | 0 / 0 / 13 / 7 |

On the hub network **no hub died before the others locked (0 of 20); all 20 died after**, so by the prediction
the lock should have stayed at 20. It did among the living (20 of 20 ever locked, all dead), and the end-state
count fell to 0 only because the dead dissolved; refuted as worded, held in substance (my reading). On the ring
the sure-wrong bettor dies before the lock in 1 seed, and the ring's lock-and-collapse falls (12 ever locked,
13 collapsed, against §6's 17 and 18): silencing the dead does help where the wrong voice is not central.
"Once the hub dies, the living reach consensus on the ring's equal weights" was **not testable**: with no
outcomes read, every bettor dies (hub at steps 8-31, the other nine in every seed), so no seed has the hub dead
and the rest alive.

**Which could have come out otherwise.** The death steps and the zero sensitivity follow from `g` by
construction; the dissolution numbers follow from `0.9^k`. Open: every collapse count, and the hub timing. A
hub's death could have preceded the lock; it never did on the hub network, because under evidence a
community locks within a few races, before any bettor's supply runs out.

**Reasonable or weird** (my judgment, not evidence). Now reasonable at every scale I can read it at: a bankrupt
firm stops issuing forecasts, a dead organism stops signalling, a lab that loses funding stops publishing, and
its ideas decay unless someone else holds them. Still weird: the dead bettor's belief takes about `|lam|/0.1`
steps to dissolve, so a very sure belief lingers longest after death. Real records of dead ideas (papers,
genomes in a fossil) do not dissolve at all, and neither model nor run distinguishes a mind's dissolving from
a record's persisting. And the evidence-rule community still locks before anyone dies: silencing the dead
cannot stop a lock that happens while everyone is alive.

**What this does not establish.** That falling apart saves a community: in every evidence case a community that
locks wrong still dies whole. Nothing about other `r`, `s_1` or listener rules (for example, not renormalizing
over the living); and §7's consensus-after-hub-death clause is untested.
