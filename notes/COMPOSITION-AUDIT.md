# Composition audit: do the book's linked runs survive an added forcing term?

Book plan P26 (a), Ch 1's fourteen linked files (Claude Fable 5.1, 5266b8d9, 2026-10-30; David: "run p26a").
Predictions written before any run of this session; the timings were taken after, from one clean run of each
file alone (`scripts/run_tests.sh`), on this machine. Ch 3's two files and Ch 2's five are later stages.

## 0. What composition can do to a claim, in $\arr$

(Words: an arrangement's forward map is its *out-transfer* and its backward map its *in-transfer*, the book's
terms since P88, 2026-10-30; *readout* is the matter's map $r\colon S \to Q$ only.)

Composition in $\arr$ is the Kleisli composite for the writer monad $\Omega$ on forcings (DESIGN.md §2,
`wiring.compose_seq`): the composite's one-form is $A^*\omega_g + B^*\omega_f$, each factor's one-form pulled back
along the map that locates its point, then summed. The integrator (response, state, readout) of every box is
untouched (one exception, multi-round tensors, §5.9 item 6 (e); no file here uses one). So an added part (an
arrangement filling a new box) adds exactly one term to the forcing on the audited configuration $Q$, and there are two kinds:

* **(U) an upstream push.** A part whose out-transfer enters an input of the audited object. If the object's forcing
  reads that input (its cost has the input in it), the term is the change in $-\partial_q k(q, n)$ from the new
  input value. Every "received supply" cost in these files is linear in its input, so the term is a covector $b$
  on the coordinates the input reaches, constant in $q$. The dynamics changes by $\sharp b$: a hyperbolic rest
  moves by $-J^{-1}\sharp b$ (implicit function theorem), basins, thresholds and ratios move continuously, and the
  structure persists until a bifurcation (a saddle-node) removes a rest.
* **(R) a downstream reader.** A part $R$ whose input value is the object's out-transfer $m = o(q)$ and whose forcing
  has a component $\xi_m(q_R, m)$ on that input. Pullback adds $(T_q o)^\top\,\xi_m(q_R, o(q))$ on $Q$. If
  $\xi_m \equiv 0$ the reader loads nothing (unilateral, the ideal voltmeter). If $R$ is exact with cost
  $k_R(q_R, m) = \tfrac{\kappa}{2}(m - m_R)^2$, the term is $-\kappa\,(o(q) - m_R)\,(T_q o)^\top$: a spring of
  stiffness $\kappa$ pulling the read value toward $m_R$. **An exact reader with a preference is a supply to, or a
  load on, what it reads.** That is item 30's tilt (`STATUS.md` §3.26) in general form. Cases checked by hand: `test_town_account.py`'s fourth test, where an exact
  reader with cost $\tfrac12(m-2)^2$ moves the fit's forcing from $.4$ to $2.0$ at $q = .4$, and the one-way reader
  leaves it; the degenerate cases, $\kappa = 0$ and a constant out-transfer $o$ (the frozen parts' mean), which add
  nothing. Not tried: a reader with a non-convex cost (it can add rests), and multi-round tensors.

**Reciprocity is closedness** (the question P26 carries, "is exactness what forces reciprocity across the
library?"; the agent's, checked by hand, 2026-10-30). A part's one-form on $Q \times M$ is
$\omega = \xi_q\,dq + \xi_m\,dm$. It is closed iff the mixed partials agree, $\partial_m \xi_q = \partial_q \xi_m$
(Ch 2 `prop.closed`), and on these Cartesian spaces closed is exact. A one-way reader sets $\xi_m \equiv 0$, so
it is closed iff $\partial_m \xi_q = 0$: **a one-way reader whose reading moves it is never exact**, in this
library or anywhere. Reciprocity (equal cross-sensitivities, Ch 2 `sec.waterfall`) is not a consequence of
exactness; it is the same condition. Across the fourteen files: every box built by `exact(...)` pushes back on
what it reads with the cross-sensitivity its cost has; every box built by `own_cost`, `node`, `need_box`,
`fund_layer(record=...)`, `market` and `coupler` has its input covector set to zero by hand and is non-exact.
There is no third kind here.

**Records.** `memory`, `delay`, `money`, `books` (town) and the fund's balance (cascade) are boxes whose
"forcing" is a rate to accumulate and whose integrator is Euclidean at mobility $1$ or $dt$: an account. Such a
box means the same under any one-way reading. An exact reader of a record with a preference pushes on the
record's state, and the record no longer equals what it summed: **a record keeps its integrity only while what
reads it has no preference about it, or reads it one-way** (the agent's, *proposed*; it follows from (R)).

**Partner-specific rules** (the predator lesson, §5.9 item 58). The only one found: the cascade fund's
`record=(alpha, beta)` cancels the workers' pullback $g'(f)(\alpha\sum\kappa_j a_j + \beta\sum\kappa_j)$ exactly
for workers with cost $\tfrac b2 a^2 - (\alpha a + \beta)h_1$ and for no other worker. Under a worker with a
different want the wage wire is partly two-way again. No book claim rests on the cancellation; the file's own
retroactivity test says what the exact wire does (threshold 20.22 → 21.78).

**Reading the verdicts.** *Survives*: the claim holds under (U) and (R) with its numbers changed. *Narrowed*: it
holds for the small pushes and the readers it names, and the entry says which. *Fails*: an exact reader of the
coordinate the claim is about changes the claim's truth value. A one-way reader never changes anything, so
"survives being read" below always means by an exact reader.

## 1. `arrangements/town.py` (Ch 1 l.660, l.866)

**Links.** l.660: "Then she runs a simulation on her computer. [Her model runs in the companion code.]" after
"either way the plant keeps making the same power and never filters the smoke, even long-term." l.866: "Eve's
town is the first test. Drawn as an arrangement, it shows why agreeing is not enough: the plant's smoke reaches
the townspeople, but their coughing does not reach the plant. [The model runs in the companion code; its one-way
wire is the waterfall of sec.two_tanks.]"

**Object.** `eve_town(1)`: a closed composite $I \to I$, `compose_seq(tensor([plant, coughing]), wiring)`. Plant:
`own_cost` on $(x, c)$, cost $\tfrac12(1 + p(1-c)^2)x^2 + \tfrac k2 c^2$ with $p = 0$ at stage 1, publishes
$(x, (1-c)x)$; no liability port before stage 2. Townspeople: `own_cost` on coughing, inputs (power, smoke),
cost $\tfrac{r}{2}q^2 - q\beta s^2/2$, push $(\text{benefit}, 0)$ on its inputs: a constant covector on power
(exact piece, the plant's revenue arriving as retroactivity), zero on smoke (the insulator). Composite forcing
non-exact (the smoke wire). Integrators: Euclidean descent, mobility $dt$; partner-independent.

**Claim.** At stage 1, $c = 0$ exactly and $x \to 1$ for every $\beta$; the plant's forcing has no dependence
on the coughing (Jacobian block zero). Checked by `tests/test_town_clean.py` prediction 1 and
`tests/test_town_story.py`.

**Reader exact?** The townspeople read smoke one-way (zero covector) and power with a constant covector.

**(U) prediction.** Any push on the plant's configuration from a box other than the townspeople moves the rest
linearly ($x^* = 1 + b_x$, $c^* = b_c/k$) and leaves the claim's point intact: the plant stays deaf to the
coughing. A push *from* the coughing is stages 2–4. Part: the plant's quadratic cost and its declared zero input
covector.

**(R) prediction.** An exact reader of the smoke with any preference for less of it pulls back
$-\xi_s\,((1-c), -x)$ onto $(x, c)$: the plant cleans and cuts at once, with no account and no delay. **The
claim fails under an exact reader of the smoke, and that is the book's thesis**: Eve's account is such a
reader's effect brought in by a one-way reader plus a new wire. An exact reader whose cost is the harm,
$\beta s^2/2$, is the account with zero delay: at $p = \beta$ the plant's premium term $\tfrac12 p(1-c)^2x^2$ is
exactly the pullback of $-d(\text{harm})$ along $s = (1-c)x$ (checked by `jax.grad` at three points,
2026-10-30; by hand, $-\partial_x = -ps(1-c)$ and $-\partial_c = psx$ on both sides). David, 2026-10-30: the
insurance does it by a one-way wire, "but if it was perfected … it would become as though a force pushback."
Corrected the same day: a first version here said the townspeople box made exact would be that reader; its cost
$\tfrac r2 q^2 - q\beta s^2/2$ is a device that makes the coughing track $\beta s^2/2r$, and its smoke derivative
pushes the other way, so it is not the harm. Part: the reader's $\xi_s$ pulled back along the plant's out-transfer
$s = (1-c)x$.

**Verdict.** Survives (U); fails (R), by design and said in the book ("its one-way wire is the waterfall").

**Time.** 1 s wall, the docstring's run at three betas, passed.

## 2. `tests/test_town_account.py` (l.812, l.867)

**Links.** l.812 (with `test_town_clean.py`): "the plant cleans up rather than reducing its output …: the
townspeople keep nearly all their power and cough far less." l.867: "She finds a way to make a cost that would
otherwise have arrived later instead count as a cost now."

**Object.** `build(3)` and `build(4)`, the **one-dial** town (clean=False, revenue inside the plant's cost):
closed composite of `own_cost`/`node` boxes: output (the plant, reads the premium one-way), coughing, memory,
delay (300 steps), assessed (the court's fit, descends $\tfrac12(\hat\beta u - y)^2$ on delayed records,
$u = s^2/2$), premium (tracks estimate or assessed), estimate (stage 4: the same fit on undelayed records),
money. Every input covector zero: a chain of unilateral readers; composite non-exact. The one-dial plant can
only cut output, so the l.812 words "cleans up rather than reducing its output" are `test_town_clean.py`'s; this
file carries "a cost … count as a cost now" (l.809, l.867).

**Claim.** With her account the premium reaches 1 at $t = 1.56$, before the first claim at $t = 6.04$; harm
over 30 units falls by more than 10% against stage 3 (measured 55%), at two step sizes; a disconnected account
changes nothing; the fit sees no secret coefficient. Test 4 (a P26 test already): an exact downstream reader of
the fit box changes its forcing from $.4$ to $2.0$ at $q = .4$; a one-way reader leaves it.

**Reader exact?** None in the town.

**(U) prediction.** A standing push $-b$ on the estimate (a lobbyist box) biases the fit to
$\hat\beta^* = \beta - b/u^2$, $u^2 = (s^2/2)^2 \le 1/4$: the premium follows it. "Premium reaches 1 before the
first claim" survives for $b < u^2(\beta - 1) \approx .25$ and fails above. A second consumer pushing on power
raises $x^*$, $u$ and the fit's stiffness: survives. Part: the fit's quadratic cost, stiffness $u^2$.

**(R) prediction.** The fit is held only by $u^2 \le 1/4$, and less as the plant cleans (the file's own finding,
final estimate 1.674). An exact reader of the estimate with stiffness $\kappa$ and preference $\hat\beta_R$
moves it to $(u^2\beta + \kappa\hat\beta_R)/(u^2 + \kappa)$: at $\kappa = u^2$, halfway. Test 4 is this at one
point. **"Her estimate climbs to match the coughing" (l.809) holds while nothing exact reads it.** An exact plant
(its cost differentiated in the premium) pushes the premium down by $\tfrac12 s^2$: a price-taker made a
lobbyist. Part: the reader's $\xi_m$ pulled back along the identity out-transfer of the fit box.

**Verdict.** Narrowed: the account's numbers hold for one-way readers; any exact reader of the estimate with
comparable stiffness bends it, which the file already shows (test 4).

**Time.** 5 s wall, 4 passed (pytest 4.6 s).

## 3. `tests/test_town_clean.py` (l.812)

**Links.** "In her model, filtering the smoke is cheaper than cutting power, so the plant cleans up rather than
reducing its output (which was also a possible response): the townspeople keep nearly all their power and cough
far less. [… when one makes filtering cost more than cutting power, the plant output shrinks instead.]"

**Object.** `build(stage, clean=True)`: the two-dial plant, closed composite, one-way throughout, revenue inside
the plant's cost (the push version, `eve_town`, equals it: a constant covector is $d(\text{benefit}\cdot x)$,
`test_town_power.py`).

**Claim.** (1) Unpriced, $c = 0$ at every $\beta$. (2) Priced at $p = \beta = 2$, the plant cleans at a true
minimum; cleaning carries most of the harm reduction for $k < 1$, cutting for $k > 1$, the split 50/50 at
$k = 1$ (the cost is symmetric in $x$ and $1 - c$ there). (3) Her account makes the plant clean before the first
claim. (4) Power falls under 5% at small $k$ against 63.5% one-dial. (5) At $\beta = 8$, $k = .8$ two minima;
the informed plant is trapped in the dearer one, the town's path reaches the cheaper.

**(U) prediction.** A push $b_c$ on cleanliness (a regulator) adds a linear term: the minimum moves by
$H^{-1}b$ and the threshold leaves $k = 1$. The direction claim, cheaper dial moves more, survives; the
threshold's value and the shares do not. A push past the dirty minimum's basin depth (prediction 5) tips the
informed plant out of its trap: hysteresis, as item 30. Part: the plant's two-dial cost, its Hessian.

**(R) prediction.** An exact reader of smoke pushes both dials through $s = (1-c)x$ in the ratio $(1-c) : -x$,
set by the out-transfer's Jacobian, not by $k$: the split is then the reader's geometry and the plant's $k$ together,
and $k = 1$ is no longer the threshold. An exact reader of power with a nonlinear cost (diminishing returns)
breaks the $x \leftrightarrow 1 - c$ symmetry the threshold rests on. Direction survives; the numbers do not.
Part: the reader's $\xi_s$ or $\xi_x$ through the plant's out-transfer.

**Verdict.** Survives in direction (cheap filtering → clean, dear → cut); every number (threshold $k = 1$,
shares, 5%) is the one-way town's.

**Time.** 9 s wall, 5 passed (pytest 8.7 s).

## 4. `tests/test_tilted_cue.py` (l.837)

**Links.** "a brief cue moves a reader into a basin that it then descends on its own. When the reader is poised,
resting just below a low ridge beside a much deeper well, the cue pays only to cross the ridge, at which point
the reader's own potential pays for the rest of its fall. What the hearer feels is only the slope where it
stands; how far that slope leads downward is its potential gain."

**Object.** The rendezvous composite `compose_seq(tensor([S1, S2]), parent.arr())`, an **open** arrangement
$I \to B$ of the attention operad: two exact subsidiaries (cost $\tfrac12(p - z_i)^2$ on the received position)
and an exact parent (cost $\tfrac12(g - (h_2 - h_1))^2 + \phi(g) + \tfrac12 u^2 - ug + 2$, $\phi$ the tilted
double well). Run with a scheduled exterior value $u(t)$ and zero exterior covector (`scan_closed`). Composite
forcing $-dV + u\,dg$, checked (test 2). Integrators Euclidean, all mobilities $h$.

**Claim.** For the minimal brief cue, the work $W$ is between $\epsilon$ and $3.5\epsilon$, the release after the
cue $D + \epsilon$ plus a small excess, so released$/W$ grows with $D$ faster than $D/\epsilon$ (5.7, 35, 188 at
$D = 4, 16, 64$); the work up to the ridge crossing is between $\epsilon$ and $5\epsilon$ for every cue tried; the
composite's net release $V(0) - V(\text{end}) = D$ for every capturing cue. The after-cue ratio is a knife-edge
(within 1% of the minimal amplitude).

**Reader exact?** Everything exact; nothing reads the composite.

**(U) prediction.** The cue is the upstream push. A standing push $b$ on $g$ (a second source, a bias) tilts
$\phi$ to $\phi - bg$: $D$ and $\epsilon$ become $D - b$ and the new ridge height; the claim survives with those
values, and is void once $b$ removes the shallow well (saddle-node): the reader is then falling, not poised. A
push on a subsidiary's $z_i$ acts on $g$ through $K = \tfrac12(g - (z_2 - z_1))^2$ once $z$ settles: the same
tilt. Part: the parent's preference cost $\phi$.

**(R) prediction.** The composite's out-transfer is $\sigma = m_1 + m_2 = (z_1 + z_2, 2)$. An exact reader of it
constrains $z_1 + z_2$, a direction orthogonal to the one the cue acts through ($z_2 - z_1$): both $K = 0$ and
the reader's minimum are reachable at every $g$, so the ridge is still $\epsilon$ above the rest and the deep well
still $D$ below. **The energy accounting survives any exact reader of the output**; $W$ moves inside its band.
An exact reader of the commitment $g$ itself (a hand tracking it, cost $\tfrac{\kappa}{2}(h - g)^2$, slow) is a
load on $g$: the ridge of the joint landscape is higher by what it costs to drag the hand to it, the minimal cue
grows, released$/W$ falls. Part: the reader's $\xi_g$ pulled back along the identity out-transfer of $g$. Unilateral
readers change nothing.

**Verdict.** Survives readers of its output; narrowed by an exact reader of $g$ (the ridge includes the load).
The book's words hold with "the ridge" read as the joint landscape's.

**Time.** 27 s wall, 8 passed (pytest 25.7 s).

## 5. `tests/test_active_inference.py` (l.841)

**Links.** "a level whose cost is the gap between its prediction and what arrives pushes the level below toward
that prediction … so that a goal at the top level moves the hand without any separate command sent down."

**Object.** The linked claim is Addendum C, `ai.hierarchy`: `compose_seq(compose_seq(hand, predictor1),
predictor2)`, closed $I \to I$. Hand: potential $-\tfrac{\kappa}{2}h^2$, publishes $h$. Level 1: cost
$\tfrac{\Pi_1}{2}(m - \mu_1)^2$ on what arrives, publishes $\mu_1$. Level 2 (top): $\tfrac{\Pi_2}{2}(m - \mu_2)^2
+ \tfrac{\pi_g}{2}(\mu_2 - g)^2$. All exact; composite forcing $-dV$, $V$ a spring chain. Tests (i)–(iii) and
Addenda A, B are the precision learner, the chooser and the figurine planner, not linked at l.841; (iii-d) is
itself a P26 test (exact wiring bends the belief .384, unilateral leaves it on the evidence).

**Claim.** The rest is the spring chain's, $0 < h < \mu_1 < \mu_2 < g$, to $10^{-10}$ (C1, C2); a unilateral
level breaks the chain below it (C3); only the ratio of precisions counts (C4).

**Reader exact?** The claim is *about* exact readers: each level moves what it reads.

**(U) prediction.** A load on the hand (a standing force $-F$): the chain is linear, so every coordinate shifts
by $-J^{-1}F$; the hand ends further from the goal; the ordering survives while $F$ is below the chain's pull
at $h = 0$. A push on a level's prediction from the side: the same linearity. Part: the hand's potential plus
the load; the linear chain.

**(R) prediction.** An exact reader of $h$ or of any $\mu_i$ with a quadratic cost is one more spring in the
network: the rest is unique and stable (all stiffnesses positive), every number moves, "a goal at the top moves
the hand" survives weakened, C4's ratio statement survives with the reader's stiffness included. A reader with a
non-convex cost could add rests; outside the claim. Part: the reader's $\xi_m$, a spring.

**Verdict.** Survives as a mechanism; the closed-form numbers are the chain's.

**Time.** 30 s wall, 26 passed, 6 xfailed as marked (pytest 29.1 s).

## 6. `tests/test_dispersal_momentum.py` (l.852) and 7. `tests/test_scattering.py` (l.852)

**Links.** "On the collective's side, a part the others can do without is barely held onto: a push carries it
off for good, and even a random noise can carry it away."

**Object.** `collective.collective`: `compose_seq(tensor(frozen parts), parent.arr())`, an **open** arrangement
$I \to B$ run with the exterior input $((0,0),0)$ and zero exterior covector. Parts frozen (no configuration,
no cost, send $(1, 1)$). Parent: spatial attention arrangement, exact cost $\tfrac12(\nu - \sum_j\kappa_j)^2$
on the signal it sends onward, $\kappa_j = e^{-p_j^2/2\tau}$; only placements move. Integrator: phase or damped
phase (momentum), or `kicked_configuration` (descent plus independent $\pm a$ kicks). The composite is the
Hamiltonian, damped or kicked flow of one potential whose Hessian has rank one.

**Claim.** Momentum: (5) undamped, the spring's share $1/n$ of a push returns and the free share $1 - 1/n$
never does; (6) damped, a push lands the configuration somewhere new (the click); (7) with $n = 3$ at
$\nu = 1.5$ the pushed part coasts out for good, with $n = 2$ it is turned back. Scattering: (8) with the need
present, kicks lose a part in $\ge 80\%$ of $n = 3$ runs and never in $n = 2$; (9) with the cost removed,
placements are recurrent random walks returning as $t^{-1/2}$.

**Reader exact?** The parent is exact; the parts costless; nothing reads the composite.

**(U) prediction.** A push through the exterior input does nothing: the parent's cost reads only what it sends
onward, and the parts it delivers to have no cost. A standing force on a placement moves the configuration
along the $(n-1)$-dimensional family of minima for as long as the others can cover $\nu$, then meets the
family's edge: a stronger form of (7) and (8). The claims are about pushes and survive. Part: the parent's
rank-one cost.

**(R) prediction.** The out-transfer is $\sigma = (\sum_j\kappa_j, \sum_j\kappa_j)$. Any exact reader of it adds a
term in the same rank-one direction $d(\sum_j\kappa_j)$: the family of minima stays $(n-1)$-dimensional (its
level moves to the combined target), so the free directions, and (5)–(8) with them, **survive every exact reader
of the output**. A reader of the mean readout reads a constant (frozen parts send $h = 1$). Only a reader of an
individual placement, which the composite does not expose, could hold a part. (9) has no cost by premise. Part:
the parent's out-transfer, a symmetric function of the placements.

**Verdict.** Survives (U) and (R).

**Time.** `test_dispersal_momentum.py` 8 s wall, 4 passed (pytest 7.2 s); `test_scattering.py` 15 s wall, 2 passed (pytest 14.3 s).

## 8. `tests/test_dispersal.py` (l.853)

**Links.** "And a part that the collective cannot do without is pulled back by that need: without it, the
others send on less than the collective's cost asks for, and only bringing the part back makes up the
difference, so that is where the collective settles."

**Object.** The same collective with the descent integrator; open $I \to B$, zero exterior covector.

**Claim.** (1) The onward message returns to $\nu$ whenever the others can supply it; (2) a small displacement is
recovered only in the fraction $1/n$; (3) a part at $d = 3$ is pulled back for $n = 2$ and stays out for
$n = 3$; (4) total dispersal returns in a time set by distance, $\propto e^{d^2/2\tau}$, not by $n$.

**(U) prediction.** The pull on the far part in (3) is the residual times the kernel's slope,
$(\nu - \sum\kappa)\,\kappa'(d) \approx .5 \times .014 \approx .007$ at $d = 3$: **a standing outward push of
that order holds the $n = 2$ part out for good**, and (4)'s return is stopped by any push beyond the Gaussian
tail. The claim "pulled back by that need" survives as a direction; "so that is where the collective settles"
holds only against pushes smaller than the kernel's tail at the part's distance. Part: the kernel
$\kappa(p) = e^{-p^2/2\tau}$ in the parent's cost.

**(R) prediction.** As in 6–7: a reader of the output adds a term in the same direction; the coverage condition
of (3) becomes "can the rest cover the combined demand," and (1), (2), (4) survive with the combined target.
Part: the parent's out-transfer.

**Verdict.** Survives (R); narrowed under (U): the need's pull is exponentially weak at a distance.

**Time.** 14 s wall, 4 passed (pytest 13.5 s).

## 9. `tests/test_cascade.py` (l.853)

**Links.** "On the part's side, a part that doesn't get anything from the collective moves away to wherever it
can get something."

**Object.** `one_level`: `compose_seq(unit.arr, market(2))`, closed $I \to I$. `unit.arr` is
`compose_seq(collective(workers + source, zero-cost parent, two channels), fund_layer)`. Workers: exact parts,
cost $\tfrac b2 a^2 - (\alpha a + \beta)h_1$ on the supply received; their force $(\alpha a + \beta)$ on the
input is pulled back by the parent's delivery onto their placements (they climb the supply $h_1(p) = g\kappa(p)
+ s\kappa(p - d)$) and onto the fund's balance (the needy prop up the purse, 21.78 against 20.22), which
`record` cancels. Source: frozen, supply $s$, weight 0. Fund: a record, non-exact flow income $- g(f)\tilde w -
c_0 - \text{leak}\,f$, pays the wage one-way. Market: stateless, pays $r \times$ product forward, zero form:
a unilateral reader of the product. The collective is "held together by what it is paid" because its reader
is one-way.

**Claim.** (10) A threshold in shock size, all or none: below it the fund recovers, above it every worker leaves
for the source. (11) With no source the same shock disperses nothing: amplitudes fade in place. (12) The
threshold rises with $n$ by the balance, not the margin. (13), (14) nested: a permanent cut runs to every
depth; a transient shock is filtered by slow funds.

**Reader exact?** Workers exact; fund and market one-way; the composite non-exact.

**(U) prediction.** A standing subsidy into the fund raises the equilibrium balance and the threshold with it
((12): the threshold is the balance's); a standing force on a placement tilts $h_1$ and moves the folds. (10)
and (11) survive: leaving is the worker's ascent of supply, fading is its dissipation $b$. Part: the worker's
cost and the fund's flow.

**(R) prediction.** An exact buyer in place of the market, cost $\tfrac{\kappa}{2}(\text{product} - m_R)^2$,
pulls back $\xi\,a_j\,\kappa'(p_j)$ onto each placement through the fund's pass-through out-transfer and the
collective's $\sigma$: **a buyer that needs the product pulls the workers home directly, bypassing the wage**.
A part that "gets nothing from the collective" is then getting the buyer's pull, and the leaving threshold moves
up by the pull's size; with a convex buyer the pull on an absent part grows as the others leave. **(10)'s
"all leave" fails under an exact reader of the product** for pulls comparable to the wage; (11) survives (no
elsewhere). Part: the buyer's $\xi_m$ along $\sigma_{\text{product}} = \sum_j\kappa_j a_j$.

**Verdict.** Survives (U); (10) fails (R). The book's sentence is scoped by "doesn't get anything": with an
exact buyer the part gets a pull, so the sentence stands and the model's one-way market is what makes the pull
absent.

**Time.** 608 s wall, 5 passed (pytest 607.5 s, 10 min 7 s).

## 10. `tests/test_ratchet.py` (l.853)

**Links.** "In a town, where parts do not feel each other's push directly, the collective's pull has to reach a
part as what it offers: the wage that brings parts back is about seven times the wage at which they left."

**Object.** The same one-level cascade, $n = 1$ to $4$, with the fund clamped for (R2), (R3).

**Claim.** (R2) The wage at which a worker at home leaves is $g_{\text{out}} = \phi_{\min}s$ and the wage that
brings one back from the source is $g_{\text{in}} = \phi_{\max}s$, the fold points of $h_1(p) = g\kappa(p) +
s\kappa(p - d)$; their ratio is $7.39$ at $d = 2.5$, $\tau = .8$ and $8.43$ at $d = 2$, $\tau = .5$. (R3) The
two wages do not depend on $n$: parts do not call each other back. (R1) Money cannot outbid the elsewhere when
$w_{\max} \le s$. (R4), (R5) the loop closes at $w_{\max} = 4s$ and a fixed point forgets. (R6) membership is a
scar, not a ratchet. (R7) both routes are one condition.

**The ratio is a function of one number.** With $p_1 p_2 = \tau$, $p_1 + p_2 = d$,
$\phi_{\max}/\phi_{\min} = (p_1/p_2)^2\,e^{d(p_2 - p_1)/\tau}$, which depends on $d$ and $\tau$ only through
$u = d^2/\tau$: $1$ at $u = 4$ (no bistable band below it), $7.38$ at $u = 7.8$ (the run), $8.43$ at $u = 8$,
about $5{,}400$ at $u = 16$ (checked by hand, 2026-10-30, against the file's two geometries). **"About seven
times" is the run's geometry**, an elsewhere $2.8$ kernel widths away.

**(U) prediction.** A standing force on a placement tilts $h_1$: both folds move and the ratio with them;
hysteresis (a band) survives while the push is below the fold's depth. A subsidy raises the wage but not the
clamped critical wages, which are the landscape's. Part: the worker's cost through the kernel geometry.

**(R) prediction.** An exact buyer (as in 9) is an amplitude-dependent wage, $\xi a/(\alpha a + \beta)$ added to
$g$: the folds move and the ratio changes; and since the buyer's unmet demand grows as parts leave, the pull on
an absent part grows with the others' absence: **(R3) fails under an exact reader of the product; parts then do
call each other back, through the buyer's need.** The book's sentence scopes itself to "where parts do not feel
each other's push directly," which is the one-way case. Part: the buyer's $\xi_m$ along
$\sum_j\kappa_j a_j$.

**Verdict.** Survives (U) in structure; (R3) fails (R) as scoped out by the book's own clause. The number
"seven" is not a composition matter: it is one geometry's value of a ratio that runs from $1$ upward.

**Time.** 666 s wall, 6 passed (pytest 664.8 s, 11 min 5 s).

## 11. `tests/test_carving.py` (l.855)

**Links.** "And a form held long enough can create a new capacity, a place to rest that its reader did not have
before. [… the companion code carves such a place in a reader with one degree of freedom.]"

**Object.** `carving.reader([tuning], [curvature_coupling], 0, eta_q)`: an **open** arrangement
$\langle 0|0\rangle \to \langle 1|0\rangle$, one input port for the note and no output. Quick part (exact,
$q^4/4 - uq$, mobility $\eta_q$), tuning (exact, $\gamma(w - w_0)^2/2 + \beta(w - w_0)^4/4$, mobility
$\eta_w = \eta_q/25$), a stateless exact coupling (cost $-wq^2/2$), wired. One potential $k(q, w; u)$; the
note is the exterior value, run by `Phi`. Checked equal to the one-box reader and the hand loop.

**Claim.** A note held past a threshold sharp to one step (1337) leaves the reader at the carved rest
$(\sqrt{1/2}, 1/2)$, a joint fixed point that existed before the note (basin selection at the joint level, a
new landscape at the quick level); a shorter note leaves a fading trace; rain decides by its rate; spacing is
not monotone; a non-reciprocal tuning is not accountable.

**Reader exact?** All exact; the slow/quick separation is in the integrators' mobilities.

**(U) prediction.** The note is the upstream push. A standing bias on $q$ tilts the quartic: the carved pair
$\pm\sqrt w$ becomes one deep and one shallow rest; "a rest it did not have" survives for small bias and loses
the second rest past the fold. A standing push on $w$ (a second source of deepening) moves the hold threshold
(the separatrix of the slow flow, near $w = .22$); the sharpness survives. Part: the tuning's slow flow
$-q^2/2 + \gamma(w - w_0) + \beta(w - w_0)^3$.

**(R) prediction.** The reader has no output; to be read it needs a port on $q$. An exact reader of $q$ with
stiffness $\kappa$ adds $\kappa/2\,(q - q_R)^2$ to the quick landscape: its curvature at $0$ becomes
$-w + \kappa$, so **for $\kappa > 1/2$ (the carved curvature) no second well exists at $w^* = 1/2$: the capacity
cannot be read exactly by anything stiffer than itself.** A reader that holds $q$ at $q_R \ne 0$ feeds
$q_R^2/2$ to the tuning permanently and carves by itself: a held note. A weak exact reader ($\kappa < 1/2$,
$q_R$ near a carved rest) leaves the claim with moved numbers. Part: the quick part's quartic against the
reader's spring; the tuning's deepening rule.

**Verdict.** Narrowed: survives readers gentler than the capacity; an exact reader stiffer than the carved
curvature erases or forces it.

**Time.** 7 s wall, 9 passed (pytest 7.0 s).

## 12. `tests/test_many_parts.py` (l.857)

**Links.** "A collective holds together while its parts meet one another's needs … [collectives of two to eight
parts, each needing what others send. If they form a ring in which each part needs only its neighbor, it dies
from a single cut in the ring. … one in which each needs a total from all the others—at least when they're
not in dire poverty—loses only the part that was cut but otherwise survives.]"

**Object.** `collective(n, wiring, q0, s)`: closed $I \to I$; $n$ exact parts (dissipation $-d\tfrac b2 q^2$),
a supply box (zero response), a zero box, fan wiring, and per part a stateless `need_box` returning
$\alpha(s\,g(r, q) + \beta r)$ to the part and nothing to its senders or the supply: one-way; the gate
$g(r, q) = rq/(K + rq)$ non-exact (item 12). Equal to the one-coupler version (test 0b).

**Claim.** (1) Both wirings hold above the pair's threshold $s_* = 2$ for $n = 2, 3, 5, 8$. (2) At $\beta = 0$ a
cut supply ends the ring, part by part; all-to-all loses the cut part and survives iff
$s > s'_* = 2\sqrt{K(n-1)/(n-2)}$ ("dire poverty" is $s \le s'_*$). (3) The collapse is a front with a delay
set by $b$ and $s$. (4) The origin is stable; the saddle $q_-$ is the restart threshold.

**Reader exact?** The need boxes are one-way readers; "need" here is gated one-way supply.

**(U) prediction.** A standing push on any part is an ungated path to it: the part settles at the push over
$b$ instead of dying, and the ring survives a cut degraded. **This is the file's own control, $\beta = .5$**:
the cut part settles at $\alpha\beta q_{i-1}/b$ and the ring holds. "Dies from a single cut" holds for a ring in
which nothing but the neighbor reaches a part, which is the book's "needs *only* its neighbor". Part: the gate
(zero arriving, zero drive) against the part's dissipation $b$.

**(R) prediction.** An exact reader of a part's amplitude with preference $q_R > 0$ is a supply of $\kappa q_R$
at $q \to 0$, bypassing the gate: the ring survives a cut (degraded), **so "dies from a single cut" fails under
an exact reader with a nonzero preference**; with $q_R = 0$ the reader is a load, $b \to b + \kappa$ for that
part, and the thresholds $s_*$, $s'_*$ rise. Part: the reader's spring, a constant drive $\kappa q_R$ where the
gate gives none.

**Verdict.** Survives as scoped by "only"; fails under any exact reader with a preference, which is an ungated
path (item 12's condition).

**Time.** 49 s wall, 6 passed (pytest 48.1 s).

## 13. `tests/test_town_diligence.py` (l.867)

**Links.** "She finds a way to make a cost that would otherwise have arrived later instead count as a cost now.
[… with the insurer's own check in `test_town_diligence.py` …]"

**Object.** `eve_town(4, diligence=True)`: the two-dial, power-to-town town, closed, one-way throughout, with a
`books` record (sums over paid claims and written exposure) and the insurer's rate descending
$\tfrac12(\text{rate} - \text{target})^2$, target blending her estimate with its own least-squares fit by the
share of exposure whose claims have arrived.

**Claim.** Identical to stage 4 through the first paid claim; the least-squares fit over paid claims equals
$\beta$ exactly on noiseless records (where her one-step fit stalls at 1.674); the rate ends within 2% of
$\beta$; the court is not repaired; the early loss is not recouped; with her report disconnected, harm is no
worse than stage 3's.

**(U) prediction.** A standing push $-b$ on the rate (a competitor undercutting) lowers it to target $- b$: "ends
within 2% of $\beta$" fails for $b > .04$; the fit (a record) is untouched. Part: the rate's quadratic tracking
cost.

**(R) prediction.** The plant reads the rate one-way (a price-taker). An exact plant pushes it down by its
marginal liability $\tfrac12 s^2 \approx .045$ after cleaning: the rate ends near $1.955$, outside the 2%
(regulatory capture in one line). The books are read one-way by the rate; an exact reader of the books with a
preference would corrupt the record (§0). Part: the plant's $\xi_p = -\tfrac12 s^2$ along the rate's identity
out-transfer.

**Verdict.** Narrowed: the diligence mechanism survives; the 2% is the price-taker's.

**Time.** 3 s wall, 5 passed (pytest 1.8 s).

## 14. `tests/test_town_insurance.py` (l.867)

**Links.** "… `test_town_insurance.py` is the same town without her account."

**Object.** `build(3)`: the one-dial stage-3 town, closed, one-way: the insurer reads only past claims through
the assessor.

**Claim.** The premium reaches 1 after the assessor does ($7.48$ after $7.06$) and is zero through the claims
delay: a quote cannot lead evidence it has not received. The isolated quote has the declared geometric
response. Money balances.

**(U) prediction.** A push on the premium moves its level, not its timing relative to its evidence, unless the
push itself arrives earlier than the claims, in which case it is an earlier account. Part: the premium's
tracking cost.

**(R) prediction.** An exact reader of the quote with a preference moves the quote before the evidence arrives:
**the claim "a quote cannot lead evidence it has not received" fails under an exact reader with a preference,
and such a reader is exactly an earlier account** (Eve's role, by another wire). Part: the reader's $\xi_m$
along the quote's identity out-transfer.

**Verdict.** Survives (U) in its timing claim; fails (R), and the failing reader is the book's next step.

**Time.** 3 s wall, 2 passed (pytest 3.0 s).

## 15. Timings

One clean run of each file alone, `scripts/run_tests.sh LIMIT FILE`, wall-clock seconds, this machine,
2026-10-30. `arrangements/town.py` is timed by its docstring's own run (`eve_town` at $\beta = .5, 2, 8$).

| file | seconds | result |
|---|---|---|
| `arrangements/town.py` (docstring run) | 1 | passed |
| `tests/test_town_insurance.py` | 3 | 2 passed |
| `tests/test_town_diligence.py` | 3 | 5 passed |
| `tests/test_town_account.py` | 5 | 4 passed |
| `tests/test_town_clean.py` | 9 | 5 passed |
| `tests/test_carving.py` | 7 | 9 passed |
| `tests/test_dispersal_momentum.py` | 8 | 4 passed |
| `tests/test_dispersal.py` | 14 | 4 passed |
| `tests/test_scattering.py` | 15 | 2 passed |
| `tests/test_active_inference.py` | 30 | 26 passed, 6 xfailed (strict, as marked) |
| `tests/test_tilted_cue.py` | 27 | 8 passed |
| `tests/test_many_parts.py` | 49 | 6 passed |
| `tests/test_cascade.py` | 608 | 5 passed |
| `tests/test_ratchet.py` | 666 | 6 passed |
| all fourteen, in sequence | 1445 | every file clean |

Wall seconds include the interpreter's start and JAX compilation; the pytest-reported time is in each entry. The two
ten-minute files are the cascade and the ratchet, as `tests/durations.tsv` already had them (598 s, 614 s on
2026-10-11).

## 16. Findings for David, ranked

1. **The number "about seven times" (l.853)** is one geometry's value of $\phi_{\max}/\phi_{\min}$, a function
   of $d^2/\tau$ alone, from $1$ (an elsewhere within two kernel widths, no hysteresis) to thousands. A reader
   takes it as a fact about towns. Not a composition failure; found while auditing the claim's object.
2. **Every town claim is a claim about one-way readers**, and every exact reader of a town quantity is an
   account arriving by another wire: the smoke (file 1), the estimate (2), the quote (14), the rate (13). The
   book says the first (l.866, "its one-way wire is the waterfall"); the others follow from it.
   Brought 2026-10-30; David: the account is one-way, and perfected "it would become as though a force pushback";
   in the model the perfected premium is exactly the harm's covector pulled back onto the plant (file 1).
3. **A cost-bearing collective's "held together by what it is paid"** (files 9, 10) is the one-way market's. An
   exact buyer holds parts by its need and makes them call each other back (R3 fails). The book's sentences are
   scoped ("doesn't get anything", "do not feel each other's push directly").
4. **The far part's pull is the kernel's tail** (file 8): a standing push of order $.007$ at $d = 3$ holds an
   indispensable part out for good. "So that is where the collective settles" holds against nothing larger.
5. **A carved capacity cannot be read exactly by anything stiffer than itself** (file 11, $\kappa > 1/2$).
6. **An exact reader with a preference is an ungated supply** (file 12): the ring's death from one cut is a
   statement about a ring nothing else reaches, which "needs only its neighbor" says.
7. The dispersal and scattering claims (6, 7) survive every reader of their output: what is read is a symmetric
   function of the placements, so the free directions stay free.

## 17. Stage (b) for Ch 1: the composites built and run

David, 2026-10-30 (5266b8d9): "let's not leave something for later. finish it. do the work." So the (R) composites
the entries name are built in `tests/test_composition_audit.py` and run, each with its unilateral control (the same
reader with its input covector zeroed, which must leave the file's own numbers to $10^{-12}$). **Predictions fixed
here before any composite ran** (closed forms by hand, 2026-10-30, the arithmetic in this file's history). Each can
fail. A (U) push on a coordinate the ports do not expose (a placement, a dial) is not composition and is not run.

| entry | composite | prediction |
|---|---|---|
| 1, 3 | `build(1, clean=True)` plus an exact reader of the smoke, cost $\kappa s^2/2$, $\kappa = 2$ | the plant's path $(x, c)$ equals the informed stage-2 plant's at $p = 2$ pathwise to $10^{-9}$ (the identity of §4.4); rest $(x, c) = (.99541, .95196)$; the unilateral control keeps $c = 0$ |
| 2 | `build(4)` (one dial) plus an exact reader of her estimate, $\kappa = .05$, preference $0$ | the fit rests where $u^2(\hat\beta - \beta) + \kappa\hat\beta = 0$ with $u = x^2/2$, $x = 1/(1 + \hat\beta)$: self-consistently $\hat\beta^* = .72$; predicted final estimate in $(.5, .9)$ and final premium below $1$; revised before the run from "the premium never reaches 1": while the smoke is still thick $u^2 = 1/4 \gg \kappa$, so the fit first climbs toward $1.67$ and the premium crosses $1$ early, then both fall back as the plant cuts; harm reduction against stage 3 below the file's 55%, the number recorded; control equal to `build(4)` on every coordinate to $10^{-12}$ |
| 4 | the tilted-cue composite plus an exact reader of its output signal $z_1 + z_2$, $\kappa = 1$, preference $1$ (the rest's value) | the minimal $T = .2$ cue at $D = 16$, $\epsilon = .25$, re-bisected, meets `_check_minimal_brief` with $V$ including the reader's term; $V(0) - V(\text{end}) = D$ to $10^{-9}$; the control's run at the file's amplitude equals the file's to $10^{-12}$ |
| 5 | the hierarchy, the hand's wire fanned to level 1 and to an exact reader of $h$, $\kappa = 1$, preference $0$ | rest $(h, \mu_1, \mu_2) = (.4, .6, .8)$ to $10^{-8}$ (linear solve); with preference $.5$: $(.55, .7, .85)$; control $(.5714, .7143, .8571)$ |
| 6, 8 | the dispersal collective plus an exact reader of the onward signal, $\kappa = 1$, preference $\nu$ | $n = 3$, $\delta = .05$: the displaced part recovers $1/3$ within 5%, the others move in $\delta/3$, onward $\to \nu$; the far part ($d = 3$, $n = 3$) stays past $.9d$ for 20,000 steps; with preference $1.1\nu$ the onward signal settles at $(\nu + \kappa\cdot 1.1\nu)/(1 + \kappa)$ to $10^{-6}$ |
| 6 | the momentum collective (damped) plus the same reader | the push lands the configuration at $\mu\xi_0/c\,(1 - 1/n)$ and $-\mu\xi_0/c/n$ within 5%, $n = 3$: the free direction is still free |
| 9, 10 | the one-level cascade with an exact buyer in the market's place: pays $r\times$product as before and has cost $\tfrac{\kappa}{2}(\text{product} - m_R)^2$, $\kappa = 2$, $m_R$ twice the one-way equilibrium product | after the one-way run's above-threshold drain ($1.05\,S_c$), every worker's placement stays below $1$ and the fund ends above $0$ (the one-way run: all past $2$, fund $-18.4$). (R3) at equal balance and amplitudes, the force toward home on a worker at the source is larger when all four are at the source than when three are home, by more than $10^{-3}$; with the one-way market the two forces are equal to $10^{-12}$ |
| 11 | the carving reader with a third part, held at $q_R = 0$ (mobility $0$), and an exact coupling $\tfrac{\kappa}{2}(q - q_R)^2$ | **corrected before the run** (the entry said $\kappa > 1/2$): the slaved slow rest solves $(w - \kappa)/2 = \gamma(w - w_0) + \beta(w - w_0)^3$, whose two roots merge at $\kappa_c = .0196$; so $\kappa = .05$: a 4000-step note leaves $|q| < .01$ after 40,000 steps of silence; $\kappa = .01$: it leaves $(q, w) = (.6712, .4605)$ within $.01$; control $(.7051, .4972)$ |
| 12 | the ring, $n = 5$, $s = 4$, part 0's supply cut, plus an exact reader of part 0, $\kappa = .5$ | preference $1$: every part held, $q = (.3333, 1.0, 3.0, 3.6667, 3.7273)$ within $10^{-3}$ (the cut part at $\kappa q_R/(b + \kappa)$, then $q_{i+1} = (\alpha s q_i - bK)/(b q_i)$); preference $0$: every part dies, $\max q < 10^{-3}$ |
| 13 | `eve_town(4, diligence=True)` with an exact plant (its cost differentiated in the premium it reads) | **corrected before the run** (the entry said the rate ends near $1.955$): the plant's push is $-\tfrac12 s^2$ and the clean plant's $s \approx .05$, so the final premium moves by less than $.005$ and stays within 2% of $\beta$: the diligence claim survives an exact plant, which has cleaned away what it would lobby with |
| 14 | `build(3)` plus an exact reader of the premium, $\kappa = 1$, preference $2.5$ | before any evidence the premium rests at $\kappa p_R/(1 + \kappa) = 1.25 > 1$, so it reaches $1$ before $t = 2$ and before the court's fit does: "a quote cannot lead evidence it has not received" fails; the control lags as the file says |

Not run: a reader of the commitment $g$ (entry 4's narrowing), since the composite does not expose $g$; the (U)
pushes on placements and dials.

### 17.1 What the composites showed (run 2026-10-30, `tests/test_composition_audit.py`, 13 tests)

| entry | result | verdict |
|---|---|---|
| 1, 3 | the stage-1 town with the exact smoke reader follows the informed plant at $p = 2$ to $10^{-9}$ pathwise; rest $(.995406, .951962)$; the control never cleans | **held**: the perfected premium and an exact reader of the smoke are one forcing |
| 2 | final estimate $.7235$ (predicted $.72$), final premium $.72$, harm reduction against stage 3 $-15.8\%$ (the plain account's $+55.3\%$): the lobbied account does worse than no account; the premium crossed $1$ early and fell back; control identical to $10^{-12}$ | **held**, and the number says more than predicted: an exact reader of her estimate with stiffness $.05$ turns the account into a harm |
| 4 | with the output reader the minimal cue's work is $.4639$ against the file's $.464$; released $16.26$; ridge work $.4627$; $V(0) - V(\text{end}) = 16$; the control equals the file's run | **held**: the reader constrains $z_1 + z_2$, orthogonal to the cue's direction |
| 5 | rests $(.4, .6, .8)$ and $(.55, .7, .85)$ to $10^{-8}$; control $(4/7, 5/7, 6/7)$ | **held** |
| 6, 8 | recovered fraction $.3264$ ($1/3$), the others in $.01668$ ($\delta/3$), onward $= \nu$; the far part stays at $> .9d$; preference $1.1\nu$ puts the signal at the combined target to $10^{-6}$; control equals the file's | **held**: the flat directions stay flat under any reader of the total |
| 6 | the damped push lands at $(.03345, -.01657, -.01657)$ against $(.03333, -.01667, -.01667)$ | **held** |
| 9, 10 | one-way: threshold $20.22$, the drain kills the fund ($-18.39$) and sends all four to $2.5$. Exact buyer: equilibrium balance $87.6$ (placements $.036$, amplitudes $1.87$); the same drain is absorbed; **a drain of three times the balance is absorbed too**, the fund back at $87.6$ within a few hundred steps with every worker home. (R3) force toward home on the absent worker: one-way $-.1197$ in both states; buyer $-1.091$ all out against $-.723$ three home | **held, and the run's own second prediction refuted**: the buyer's covector on the product pulls back onto the amplitudes (a worker sends $(0, a, 1)$), so the buyer's need drives production with no wage, income persists and the fund refills; the collective is held together by what is needed of it. (The leak term pays into a negative balance too, a feature of the model's fund, present in the one-way run's rest at $-18.4$.) |
| 11 | $\kappa = 0$: $(.70711, .50000)$, the file's; $\kappa = .01$: $(.67123, .46055)$ against $(.6712, .4605)$; $\kappa = .05$: $q = 10^{-233}$, $w = -.473$ | **held** with the corrected threshold: the saddle-node at $\kappa_c = .0196$, not $1/2$ |
| 12 | $(.3333, 1.0, 3.0, 3.6667, 3.7273)$ exactly; preference $0$ and the control: every part dead | **held** |
| 13 | price-taker $1.99994$, exact plant $1.99882$, final smoke $.0478$ ($s^2/2 = .00114$) | **held**: the clean plant lobbies by a thousandth |
| 14 | the premium reaches $1$ at $t = .4$, the court at $8.42$; control $7.48$ after $7.06$ | **held**: an exact reader with a preference is an earlier account |

**What the stage adds to §0.** The rule held in every case, and twice it reached further than the entry had reasoned:
a reader's covector lands on *every* coordinate the out-transfer depends on, so an exact buyer of a product drives the
amplitudes that make it (entry 9), and a reader of a fit held by $u^2 = (s^2/2)^2$ can turn the account against its
purpose (entry 2). Two entries' numbers were corrected by hand before their runs (11, 13) and one prediction was
refuted by its run (9's second block); all three are recorded where they stand. No book sentence changes: each
failing claim was a claim about one-way readers, and the book's sentences carry their own scope.

## 18. Ch 3's two linked files (stage (a), predictions before any composite ran, 2026-10-30)

### 18.1 `tests/test_pendulum.py` (Ch 3 l.2525)

**Links.** `sec.arr`: "the bars vibrate along their length about a thousand times a second, while the pendulum
swings about once every two seconds, and a step has to follow the vibration: steps of a tenth of a millisecond run
well, and steps three times as long make the run blow up. [The companion code builds the $n$-fold pendulum by
nesting links, runs it, and compares it with the rigid one.]"

**Object.** `drop . link^2 . pivot`: a closed composite built by nesting; each link a bob (phase integrator, mass in
the plane) whose cost is the bar from its support to it, $\tfrac{\kappa}{8\ell^2}(|q - x|^2 - \ell^2)^2$, plus
gravity $mgq_y$; the link reads out its bob's position to the next link. All exact; one potential, the sum.

**Claim.** (1) The nested composite is the flat pendulum (one cost, symplectic Euler by hand) to $10^{-10}$. (2) As
the bars stiffen to steel's $2\times10^7$ the lower bob comes within 1 mm of the rigid double pendulum over 1 s,
stretching under 10 µm. (3) The bar sets the step: at steel $10^{-3}$ s blows up, $10^{-4}$ keeps the energy within
0.1%.

**Reader exact?** Everything; nothing reads the pendulum (`drop`).

**(U) prediction.** A standing force on a bob (wind, a potential $-F\cdot q$ by a part) tilts the rest and the
swing; (1) survives by functoriality (the nested and the flat pendulum with the same force agree), (3) survives since
the step is set by the bar's stiffness, which a soft push does not touch; (2) survives only against a rigid reference
given the same force. Part: the link's cost, and the bob's phase integrator.

**(R) prediction.** An exact reader of the lower bob's position with cost $\tfrac{\kappa_r}{2}|q - x_R|^2$ is one
more spring, from the bob to a fixed point. (1) survives: the composite with the reader equals the flat computation
with the spring's potential added, to $10^{-10}$. (3) survives for $\kappa_r \ll \kappa$: the fastest mode is still the
bar's. (2) survives against the rigid pendulum with the same spring added to its Lagrangian. The claim's words do
not change. Part: the reader's spring on the bob through the link's identity out-transfer.

**Verdict.** Survives (U) and (R); the pendulum is the one file here whose every claim is about an exact object and
reads as such.

**Time.** 5 s wall, 3 passed (pytest 3.8 s), one clean run alone, 2026-10-30.

### 18.2 `tests/test_copy_pair.py` (Ch 3 l.3104)

**Links.** `ex.copying_learner`: "With $\kappa = 1$ for 4000 steps, the copy comes to send what the original sends:
its angular frequency is 0.3 … Then with $\kappa = 0$ … it goes on sending angular frequency 0.3, while the original
sends the same free sine wave throughout. With $\kappa = 0$ from the start, nothing is copied." And: "which one is
the original is set only by which way `couple` pulls."

**Object.** `copy_pair`: two `osc` (matter the oscillator on the circle, states (angle, angular frequency), effort
field zero) and `contact_one_way(1)` on $(x, y, K)$: the forcing $K(x - y)$ on the copy's value and nothing on the
original's. Open: $\langle 0|1\rangle \to I$, the coupling $K$ its exterior value. The coupler is **non-exact by
design** (Ch 2's waterfall): the copy reads the original one-way.

**Claim.** The copy locks to the original's angular frequency (to $10^{-6}$; values to $10^{-9}$) and keeps it when
the coupling is cut; the original is untouched; uncoupled nothing is copied; without the angle rate nothing locks.

**Reader exact?** No: the whole point is a one-way reader, and the book says so ("set only by which way `couple`
pulls").

**(U) prediction.** A standing force on the copy's angle shifts its phase; it still locks in angular frequency, with
a phase lag, so "its value differs from the original's by less than $10^{-13}$" fails and "its angular frequency is
0.3" survives. Part: the oscillator's update, angle and frequency both moved by the force.

**(R) prediction.** Make the coupling exact, `contact(1)` with cost $\tfrac{K}{2}(x - y)^2$ returned to both: the
original is pushed too. **"The original is untouched" fails, and with it "which one is the original"**: the two lock
to a common angular frequency strictly between $.22$ and $.3$, and keep it when $K$ goes to 0. The book's clause
names exactly this: the asymmetry is the coupler's. Part: the exact coupler's covector on $x$, pulled back along the
original's out-transfer $\sin$ onto its matter.

**Verdict.** Survives (U) in the frequency claim; fails (R), as the book's own clause says.

**Time.** 3 s wall, 9 passed (pytest 2.2 s), one clean run alone, 2026-10-30.

## 19. Ch 2's five linked files (stage (a), predictions before any composite ran, 2026-10-30)

These are the ported physics, each a closed composite of the book's own parts wired by `boxes_and_couplers`
(one-port parts, two-port couplers). **One observation covers all five**: an exact reader of a part's read-out value
with stiffness $\kappa_r$ and preference $e_R$ is, in the book's own terms, *one more part through one more
coupler*: a reservoir at pressure $e_R$ through a pipe, a heater at $T_R$ through a joint, a wall at $x_R$
through a spring, a source at $E_R$ through a resistor, a weight decay on a parameter. So the (R) composite for each
is built with the chapter's own pieces, and what it changes is the chapter's own conservation law, since a source
is not inside the sum.

### 19.1 `tests/test_two_tanks.py` (Ch 2 l.734)

**Links.** `sec.pump`: "no cost function can give influence that goes one way only, as over the waterfall. [The
companion code runs the pipe, the waterfall, and the pump …; it checks each one's update, that the pipe evens out the
levels, and that the pipe's two sensitivities agree while the waterfall's do not.]"

**Object.** Two tanks (state the volume, out-transfer the pressure, response the cross-section) and a coupler (pipe:
exact, cost $\tfrac{\kappa}{2}(e_1 - e_2)^2$; waterfall and pump: non-exact), closed by `boxes_and_couplers`.

**Claim.** The hand steps; the pipe evens out the levels; the pipe's sensitivities agree and the waterfall's do not
(`prop.closed`); water is conserved when the two flows are opposite; composition agrees with dynamics for the
waterfall.

**(U) prediction.** A pump into one tank: the levels rest with a steady pipe flow $I_0$ and a gap $I_0/\kappa$;
"the pipe evens out the levels" fails under any standing inflow; the sensitivities (a property of the coupler's
one-form alone) are untouched. Part: the pipe's cost.

**(R) prediction.** An exact reader of tank 1's pressure, preference $e_R$, is a reservoir at $e_R$ through a pipe
of conductance $\kappa_r$. At rest the pipe's flow is zero (tank 2 still) and the reservoir's flow is zero (tank
1 still), so **both levels go to $e_R$**, whatever the two tanks held: "the pipe evens out the levels" survives, at
the reader's level; **"the total water is conserved" fails**, the reader being a source. Part: the reader's covector
on $e_1$ through the tank's out-transfer.

**Verdict.** Survives (R) in the evening-out, fails in conservation; the book claims conservation for the two tanks
alone and says the pump is one-way, so no sentence changes.

**Time.** 10 s wall, 10 passed (pytest 10.0 s), one clean run alone, 2026-10-30.

### 19.2 `tests/test_heat_equation.py` and 19.3 `tests/test_wave_equation.py` (Ch 2 l.1188)

**Links.** `sec.waves`: "the rod's temperatures settle at $\tfrac83$, while the wave's masses go past it … and never
settle. What makes the difference between them is the momentum. [The companion code builds both chains by filling
the same wiring pattern in two ways …; without momentum the masses update exactly as the rod's temperatures do and
come to rest at $\tfrac83$, and with momentum each update is the one worked here.]"

**Object.** `book_chain`: three one-port parts (segments: configuration; masses: phase, reading out $q + p/m$) joined
by two exact springs (joints), free ends, closed. Also the held chains of any length.

**Claim.** Heat: the hand steps, the sum 8 conserved, all three to $8/3$. Waves: the hand steps, momentum 0 and
positions' sum 8 conserved, the middle mass passes $8/3$ and nothing settles.

**(U) prediction.** A standing heat input to one free-ended segment raises every temperature without bound (no
rest); a standing force on a mass accelerates the chain's centre. Both chapters' conservation claims fail under a
push, as they must: the sums are conserved because the forces of a free chain sum to zero. Part: the chain's
springs, whose forces cancel in pairs.

**(R) prediction.** An exact reader of the middle segment's temperature, preference $T_R$, is a heater at $T_R$
through a joint of conductance $\kappa_r$ (`exc.heat_held`): **all three temperatures go to $T_R$**, the sum is not
conserved; the rod still settles. An exact reader of the middle mass's position is a wall at $x_R$ through a
spring: the chain **still never settles** (no friction anywhere), the positions' sum is no longer conserved, the
energy including the wall's spring stays bounded. **"What makes the difference between them is the momentum"
survives the reader**: with the same reader on both, the rod settles and the chain does not. Part: the reader's
spring on the read-out value.

**Verdict.** Survives (R) in the claim the book makes (the difference is the momentum); the free chains'
conservation laws are the free chains'.

**Time.** `test_heat_equation.py` 35 s wall, 6 passed (pytest 34.3 s); `test_wave_equation.py` 10 s wall, 7 passed (pytest 9.4 s); one clean run alone each, 2026-10-30.

### 19.4 `tests/test_kirchhoff_node.py` (Ch 2 l.1357)

**Links.** `sec.kcl`: "the dot copies the middle capacitor's voltage forward to both resistors, and the transpose of
copying adds the currents they send back. [The companion code copies one capacitor's voltage to three resistors,
without any part that adds currents, and checks that its charge changes by the sum of the three currents. It also
runs a loop of three capacitors … and checks that the total charge stays the same while the voltages even out.]"

**Object.** Capacitors (state the charge, out-transfer the voltage, response $C$), resistors (exact, the spring with
voltages), sources (constant out-transfer, zero response), wired by `boxes_and_couplers`; closed.

**Claim.** KCL at a node is the transpose of copying: the charge changes by the sum of the branch currents, produced
by the composite and written in no part; the loop conserves total charge and evens the voltages to the
capacitance-weighted mean; the rest voltage between two sources is the conductance-weighted mean.

**(U) prediction.** A current source into a node: KCL survives (one more term in the sum), the loop's total charge
grows, the voltages even out around a rising mean. Part: the dot's transpose.

**(R) prediction.** An exact reader of a capacitor's voltage, preference $E_R$, is a resistor to a source at
$E_R$, the file's own `_star`. **KCL survives** (the node's charge changes by the sum of four currents, the reader's
one of them), **"the total charge stays the same" fails** (charge flows to the source), and the loop's voltages even
out at $E_R$, not at the weighted mean. Part: the copy wire's transpose, summing the reader's current with the rest.

**Verdict.** The law the book states (KCL is the transpose of copying) survives any reader, since the reader is
one more branch; the loop's conservation is the loop's.

**Time.** 21 s wall, 7 passed (pytest 20.2 s), one clean run alone, 2026-10-30.

### 19.5 `tests/test_gradient_descent.py` (Ch 2 l.1696)

**Links.** `sec.neural_network`: "the force on each parameter is $\xi_i = -\partial_{w_i}(\text{cost}(w_3\tanh(w_1x +
w_2), \ell))$ … Passing the force backwards through the parts, one part at a time, is backpropagation … That it
gives each parameter exactly this force, in any network, is the chain rule."

**Object.** The network as drawn: a parameter box per weight (state the weight, out-transfer it, response the
learning rate), example boxes, gray boxes (static arrangements, forces back by the transpose Jacobian), the loss (a
part with cost $\tfrac12(y - \ell)^2$); closed, by `compose_seq` and `tensor_arrangements`.

**Claim.** The book's step box by box, $(2, 2, 0)$ then $(1.2, -.8, 2)$; the forces are minus the partials of the
loss (autodiff); the drawn network trains as the one-box loop does.

**(U) prediction.** The upstream push is the example: a different datum or label changes every force by the chain
rule; nothing else is upstream of a closed network. Part: the loss's cost at the new example.

**(R) prediction.** An exact reader of a parameter with preference 0 and stiffness $\kappa_r$ is **weight decay**:
the force on $w_1$ becomes $2 - \kappa_r w_1$, so the book's step gives $w_1' = 1 + .1(2 - .5) = 1.15$ at
$\kappa_r = .5$ (the others unchanged). "The force on each parameter is minus the partial derivative of the loss's
cost" **fails as worded and survives as the chain rule**: the force is minus the partial of the *composite's* cost,
loss plus the reader's. An exact reader of the answer $y$ with a preference is a second loss, the same way. Part:
the reader's spring on the weight through the parameter box's identity out-transfer.

**Verdict.** Survives as the chain rule, which is what the book says the mechanism is; the word "loss" in "the
loss's cost" is the one-loss network's.

**Time.** 37 s wall, 9 passed (pytest 37.0 s), one clean run alone, 2026-10-30.

## 20. Stage (b) for Chs 3 and 2: the composites, predictions fixed before they ran (2026-10-30)

| entry | composite | prediction |
|---|---|---|
| 18.1 | `reader . link^2 . pivot`, the reader a spring $\kappa_r = 1$ from the lower bob to $x_R$ (its start shifted by $.1$ in $x$) | nested equals the flat computation with the spring added, to $10^{-10}$ over 500 steps at $\kappa = 10^4$; at steel ($2\times10^7$, step $10^{-4}$) the energy including the spring is within 0.1% over 10,000 steps; the stiff pendulum with the spring is within 1 mm of the rigid pendulum with the same spring over 1 s |
| 18.2 | `copy_pair` with `contact(1)` in place of `contact_one_way(1)` | after 4000 steps at $K = 1$ the two angular frequencies agree to $10^{-6}$ and lie strictly inside $(.22, .3)$ (the original moved by more than $10^{-3}$); after 3000 more at $K = 0$ both keep that value; the one-way control leaves the original at $.3$ exactly |
| 19.1 | two unit cylinders from $(4, 0)$, pipe $\kappa = 1/4$, plus a source at $e_R = 3$ through a pipe $\kappa_r = .1$ to tank 1 | both pressures within $10^{-6}$ of $3$ after 400 steps; total water $6$, not $4$; the plain pair rests at $(2, 2)$ |
| 19.2 | the rod $(8, 0, 0)$, joints $1/4$, plus a heater at $T_R = 5$ through a joint $\kappa_r = .1$ to the middle segment | all three temperatures within $10^{-6}$ of $5$ after 600 steps; the sum ends near $15$ |
| 19.3 | the three masses from $(8, 0, 0)$ at rest, springs $1/4$, plus a wall at $x_R = 8/3$ through a spring $\kappa_r = .1$ to the middle mass | over 400 steps the middle mass's position ranges by more than $2$ in the last 100 steps (never settles); the positions' sum varies by more than $.5$ over the run; the energy, kinetic plus all four springs' potentials, stays within 10% of its start |
| 19.4 | the triangle of unit capacitors from $(8, 0, 0)$, conductances $1/4$, plus a resistor $R_r = 4$ from capacitor 1 to a source at $E_R = 1$ | at the first step capacitor 1's charge changes by $-\sum$ of its four branch currents to $10^{-12}$; the total charge after 400 steps is within $10^{-6}$ of $3$ (every voltage at $1$), not $8$ |
| 19.5 | the book's network, $w_1$'s wire fanned to a weight-decay reader $\kappa_r = .5$, preference $0$ | one step: $(1.15, -.8, 2.0)$ to $10^{-12}$; the control (reader one-way): $(1.2, -.8, 2.0)$ |

### 20.1 What the composites showed (run 2026-10-30; 7 tests, all passed; one prediction refuted)

| entry | result | verdict |
|---|---|---|
| 18.1 | nested with the spring equals the flat computation with the spring to $10^{-10}$; energy with the spring within 0.1% at steel, step $10^{-4}$; the stiff pendulum with the spring within $5.7\times10^{-5}$ m of the rigid one with the same spring | **held** |
| 18.2 | exact contact: both angular frequencies $.25581$ after 4000 steps (the original moved from $.30$, the copy from $.22$), kept at $K = 0$; one-way control: the original at $.30$ exactly | **held**: an exact coupling has no original, as the book's clause says |
| 19.1 | both pressures $3.000000$, water $6.000000$ (from 4); the plain pair $(2, 2)$, water 4 | **held**: the reader is a reservoir through a pipe |
| 19.2 | all three temperatures $5.000000$, sum $15$ | **held** |
| 19.3 | the middle mass ranges by $6.4$ in the last 100 steps, the positions' sum by $1.4$; **the energy prediction refuted**: kinetic plus the four springs' potentials swings by 84% of its start over the run (bounded, below twice its start) | **held** in the claim ("never settles"); the energy line was the run's own error: the book's step is coarse and symplectic Euler conserves a modified energy, not this one, and the book claims the exact invariants (momentum, the positions' sum), never this energy |
| 19.4 | capacitor 1's first change is minus the sum of its four currents to $10^{-12}$; total charge $3.0000000$ after 400 steps, every voltage $1$ | **held**: the reader is one more branch |
| 19.5 | $(1.15, -.8, 2.0)$; control $(1.2, -.8, 2.0)$ | **held**: weight decay is an exact reader of a weight |

**What the stage adds to §0.** In the ported physics an exact reader is one of the chapter's own parts through one of
its own couplers, and what it changes is the free system's conservation law, never the law the chapter states about
the wiring (the transpose of copying, the chain rule, the pipe's reciprocity, the difference between heat and waves).
Ch 3's two: the pendulum reads as what it is, one potential, and a spring on it is one more term; the copy pair's
"original" is the coupler's one-wayness, and the book says so in the sentence that links the file.

## 21. The project's question, answered for every file the book links (Chs 1–3, 21 files)

Every linked file's claim was predicted under an added push and an added exact reader (§§1–14, 18, 19), and each
reader was then built by composition and run against the prediction (§§17, 20): 20 tests, every one passed, three
predictions of this run's own corrected by hand before their runs (11, 13, 18.2's "mean") and two refuted by their
runs and recorded (9's second block; 19.3's energy). The rule of §0 held in every case. No book sentence changed for
composition's sake; one number left Ch 1 at David's word (l.853). The question P26 carried, whether exactness is
what forces reciprocity, is answered in §0: they are one condition, closedness of the one-form (Ch 2 `prop.closed`).
What this does not establish: the (U) pushes on coordinates the ports do not expose (a placement, a dial), and a
reader of the cue composite's commitment $g$, which its output does not carry; each is predicted in its entry and
not run.
