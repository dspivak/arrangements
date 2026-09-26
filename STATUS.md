# What this library establishes, and what it does not

This file is the account the code keeps of itself, and its first purpose is practical:
when the book has to decide something, this says what the library actually shows and what
it only appears to show. Every docstring that says a claim is open, inherited, or merely
illustrated points here. Read it before taking any test as evidence of anything.

The mathematics is developed in *Articulating Forms that Matter* (D. I. Spivak, in
preparation), Chapter 3; its predecessor is *Compositional dynamics in learning and
mechanics* (CDLM, in preparation), whose executable companion this replaces.

## 1. Conventions, so that nothing here is read in the wrong sign or the wrong time

**Forcing, not potential.** The driving datum of an arrangement is a one-form `omega` on
`Q x M+ x N-`, which need not be exact. When a potential exists, `omega = dU`; for a cost
`k` one takes `U = -k`, so `omega = -dk`.

**The configuration step adds.** `q' = q + sharp(omega)`. The integrator advances *along*
the forcing, whose sign already makes it descend a cost. CDLM instead calls the cost
itself `U` and subtracts; every example ported from CDLM is translated at its call site,
and the regression recordings in `tests/regress/` are stored in CDLM's sign with the
translation applied in the test.

**The integrator is part of the arrangement.** The parameter is a `cot(Q)`-coalgebra
`(Q, S, u)`, not a responsive vector space, so `Phi` takes no integrator argument. "The
same arrangement run two ways" is not expressible: heat and wave are two arrangements on
one wiring.

**Time is discrete by design.** A covector determines a displacement, one step. This is
not an Euler approximation of an underlying ODE that the library secretly means; it is
the semantics. Where a source proves a continuous-time result, the test says so and
checks the discrete analogue separately (§3.2).

## 2. What the tests are evidence of

**Strong evidence: closed forms.** Where a source states a formula, a limit, or an exact
value, the test computes it independently and compares. These are the tests to trust:
the reciprocity lemma on the three couplers, the two-tank composite, the attention
operad's substitution formulas against lens composition, and every closed form in the
attention paper's three worked examples.

**Strong evidence: two independent implementations agreeing.** The attention operad's
substitution is implemented twice — once by the paper's explicit formulas for the
composite placement, transfers, and cost, once as composition in the category — with no
shared code path. They agree to 4e-16 in the sending map, the receiving map, and every
forcing component. That is the operad-morphism proposition made executable.

**Weaker evidence: the regression recordings.** `tests/regress/*.npz` hold trajectories
produced by the CDLM companion, and the ported examples reproduce them to 1.4e-12 or
better. This proves the port is *faithful*, which is what it was for. It does not prove
either implementation is *correct*; agreement between two implementations is consistency,
not truth. Where a claim needs to be true and not merely preserved, look for a closed-form
test above, or check it by hand.

**Not evidence at all: expressibility.** Any smooth vector field `V` can be written as a
forcing by taking `omega = V^flat` with a plain sharp, making one step `x + dt V(x)`. So
"this framework can express that dynamics" is a tautology, not a result. Five examples
carry such a direct realization beside a structured one (see `reaction_network.py`), and
the difference is the point: the structured realization certifies something — a Lyapunov
function, monotone descent, dissipativity inherited under interconnection — and the
direct one certifies nothing. Admissibility as a forcing is cheap; structure is not.

**Not evidence: what the construction was built to have.** This is the expressibility warning's
near neighbour, and it is easier to miss because the run looks like a test. If the question is
"can a collective's form survive its parts turning over," and the construction is a parent over
contributors that are interchangeable by design, the answer is in the wiring diagram before
anything runs. What such a run *can* establish is the mechanism and its numbers — how the
collective degrades, how a replacement arrives, what stops it, what the cost can and cannot see —
and what it cannot establish is that the world contains collectives of that shape. The test that
separates the two: **ask which predictions could have come out otherwise.** In the turnover run,
predictions 2 and 3 could have gone either way and both went the other way; those are results.
Prediction 1, degrade and restore, was guaranteed by the construction up to the level-set
argument, and reading it as "a collective's form survives turnover" imported a premise the run
did not have (David, 2026-09-26: "you can't just go to a town and swap someone out!"). State what
a run shows about the machinery; let the claim about the world be argued separately, or not made.

## 3. Open questions the code depends on

### 3.1 The domain of `cot`

The ambient category is smooth sets, sheaves on Cartesian spaces for the differentiably
good open-cover topology. That topos does not canonically supply a fiberwise-linear
cotangent bundle on every object, so `cot` is additional structure whose domain is not
settled. **This library stays on Cartesian spaces `R^d`**, an admissible subcategory
containing every example, and records a smooth set by its dimension. Nothing here
depends on the general case, and nothing here settles it.

### 3.2 The attention paper's propositions are continuous-time

The attention operad's three dynamical results — the reader's local response, the
meeting point, and capture by a transient cue — are proved for gradient flow. The book's
dynamics is the discrete displacement map, so each needs restating with a mobility
bound, and **that restatement is not proved here**. The tests check the discrete
analogue numerically and label themselves accordingly. Six things the numerical check
settled, which any restatement has to honor:

1. **The mobility is the step.** There is no time step separate from the mobility; one
   covector, one displacement. A source's "all mobilities equal 1 over [0, 20]" becomes
   mobilities equal to `h` with the input sampled at `t_k = k h`.
2. **The bound is on the nonzero eigenvalues.** For the meeting cost the Hessian is
   positive semidefinite, and its kernel is the conserved quantity, so no mobility puts
   every eigenvalue in `(0, 2)`. The condition is that every *nonzero* eigenvalue of the
   mobility-weighted Hessian lies there, and it is an iff: straddling the critical value
   by one part in six thousand, the observed per-step growth matches `|1 - mu lambda_max|`
   to six decimals and crosses 1 exactly where the eigenvalue crosses 2.
3. **Conservation is algebraic, not first-order.** The state forces cancel identically,
   for any mobilities and any step. In float64 the drift over 500 steps is a few ulps and
   comes from the coordinate divisions, not from the dynamics.
4. **A capture hypothesis must be checked for the run actually made.** Explicit stepping
   overshoots the continuous cost; the discrete value at the end of a cue approaches the
   continuous one from above, first order in the step.
5. **Descent holds only after the cue**, during which the cost climbs. A discrete capture
   condition has to cover the driven interval, as the continuous one does.
6. **The Lipschitz constant is region-dependent, and the stiffest region may be the
   driven one**, not the sublevel set the continuous argument works over. The quartic
   preference cost has no global constant, so a restatement must name its region.

### 3.3 The integrator is `Store(S)` only

The attention paper's integrator is `(Q, C, iota)` with `C` an arbitrary polynomial
comonad, which can carry more state than the configuration. This library implements the
concrete `Store(S)` case, which is what the book's category uses. The general case is not
implemented and no example here needs it.

### 3.4 Multi-stage integrators

Leapfrog, RK4, and the K-stage machinery are present as datatypes with instances.
Whether the assignment from arrangements to K-stage coalgebras is a *functor*, and
whether it is monoidal, is left open, exactly as in CDLM. Their docstrings say so.

### 3.5 The commons model, and four things it corrected

`commons.py` and `commons_entry.py` implement a shared-resource model — one resource, `N`
extractors, and three wirings over one fixed cost — as genuine composites, so the fan-out's
transpose delivering the total to the resource is read off rather than written by hand.
Running it corrected four claims that had been made *on paper* and were wrong. They are
recorded because each was plausible and each was believed:

1. **An external enforcer applying a fixed rule is exact, not unilateral.** A coupler
   returning the same sanction to every participant while itself unmoved by them has a
   symmetric returned-covector Jacobian, so `d omega = 0`. What is genuinely non-exact is an
   enforcer whose **own state** sets the sanction while nothing comes back to it. The
   distinction is discretion, not externality.
2. **The mean readout is proximity-blind.** For a message `(kappa R, kappa)` the mean is `R`
   for every `kappa`. Proximity changes the weight, never the mean, so a quantity that
   should depend on being close cannot ride on the mean readout. Being near changes what you
   can take, not what you believe.
3. **A lifetime scaling derived at a settled state need not survive the driven regime.**
   The predicted `1/N` scaling of a supply's lifetime held only when the stock moves slowly
   against effort; at the reference starting point the stock is fast, effort is frozen, and
   the measured ratio was 17 where 2 was predicted.
4. **A convergence bound with a complex spectrum is not `2/|lambda_max|`** but
   `min_j(-2 Re lambda_j)/|lambda_j|^2`. Under the un-sanctioned wiring `N` never enters the
   effort rate at all, because the extractors are mutually invisible to one another.

**What the model does not establish**, beyond the general caveats above. The sign of a
placement coordinate's response is not fixed by the formalism — it says which boxes exist
and how they couple, not which box owns a placement — and it was chosen by hand; the other
choice reverses every entry result. Zero coupling is a fixed point with vanishing gradient,
so a participant placed far away never arrives: entry must be seeded inside a pull region
whose size nothing here fixes. The settling point is a set, not a number, selected by
initial placements. And the economics is imported wholesale.

### 3.6 Functoriality is tested, not proved

That composition agrees with dynamics is checked on instances — chains, graphs, nests,
and a non-exact coupler — not proved here. The proof belongs to the book. A passing test
suite is a warning system, not a certificate.

### 3.7 Two proposed costs on the reader, and four things they corrected

Neither construction here is in the attention paper. Both put the mean readout inside a
cost, which leaves the paper's affine-quadratic family, so the Hessian is not constant and
each run states where its mobility came from. Every docstring says which of three things
its test does: reproduce a closed form derived in the docstring, check a discrete analogue
(§3.2), or correct a claim made on paper.

**The member test** (`tests/test_member.py`; construction and numbers in
`notes/MEMBER-TEST.md`) runs the book plan's proposed definition of *member*. A part's
residual is its cost minimized over its own configuration, `r(e) = min_q k(q, e)`; it *wants*
something of its input iff `r` varies with what it receives, and it is a member at a state iff
`r(received) < r(0)`. (The book plan called that variation a *need*; since 2026-09-25 it calls
it a **want** and defines *need* separately, as what a part must receive for its own parts to
keep returning, which §3.8 below tests. The two are not genus and species: a want is read off
the part's own cost and exists only where its forcing is exact, a need needs no potential, and
§3.8's parts have needs and no wants.) What the three runs support: the residual separates a want from a
bond-like cost — a self-sufficient part receiving exactly what the needy part receives is
never a member, and the needy part becomes one once the source's signal has descended — and
a reader between a source at `0` and one at `nu` moves away from the first and toward the
second at every step. Two claims made on paper did not survive:

1. **A quadratic outer cost does not outweigh a member of a member out of membership at
   any finite weight.** With outer weight `lam` the inner residual is
   `(1/2) nu^2 (lam/(1+lam))^2`, strictly below `r(0)` for every finite `lam`; only the
   margin `r(0) - r(received)` erodes, to `0.0099` at `lam = 100`. Under the strict
   inequality loss of membership is a limit, so a definition that wants "outweighed" to end
   membership needs a threshold or a comparison of margins.
2. **Under the mean readout the need is a ray, not a point.** `argmin_e r(e)` is
   `{(nu w, w, p) : w > 0}`, and in the leaving run no configuration realizes it, so the
   reader passes the second source and keeps going, the force decaying without a settled
   state. The paper's Example 1 settles because its cost is on the weighted input.

**The curious reader** (`tests/test_curious_reader.py`) adds an information-gain term,
Lindley's expected gain of a Gaussian channel, to the reader's cost on the three-child
spatial arrangement of `tests/test_reader.py`, between a weak confirming source and a strong
disconfirming one. Eight tests in three stages. The crossover `lam*` of the midpoint force at
a held prediction is reproduced to `1e-15` relative. Two claims made on paper did not
survive, and one prediction written before its run held:

3. **A free prediction never confirms.** With nothing holding it, the prediction adjusts to
   what is heard and only precision remains, so the reader ends at the precise source for
   every `lam > 0`. Confirmation needs the prediction held by a prior term, and the dial is
   then the prior precision `w_pi`: the crossover becomes `w_pi/(1 + w_pi)` times the
   held-prediction value, `lam*/2` at `w_pi = 1`, reproduced by bisection to `1e-12`.
4. **The predicted mobility window up to `lam*` does not exist.** With the prior term the
   confirming basin closes at `lam_A = 0.807`, below `lam* = 1.19`, so the settled regime is
   set by the reduced cost's basins and `lam*` governs the transient only. Inside
   `(lam*/2, lam_A)` the prediction written first held: the ratio of the prediction's
   mobility to the position's selects the basin, a slow reviser settling in the confirming
   local minimum and a fast one in the precise global one. Equal mobilities went to the
   precise source in the two cases run; the boundary was not located.

What neither establishes: whether *member* or the information term belongs in the book is
not a question the code answers, and the definitions run here are the plan's proposals, not
the book's. The reruns the member test calls for — the want restated on the weighted input,
and a non-quadratic outer cost — are listed in the book plan and have not been run.

### 3.8 Mutual need, and what carries it

`tests/test_mutual_need.py` (8 tests; construction, closed forms and numbers in
`notes/MUTUAL-NEED-TEST.md`) builds two parts, each a driven steady state in §3.1's sense —
its configuration the amplitude of its own form, held nonzero while flux passes — each of
whose held form is supplied by the other, with one supply entering the pair from outside as a
zero-response box. Four predictions were fixed before the run: both forms held while the
supply runs; cutting the supply ends both; cutting either internal link ends both; neither
part holds its form alone. A supply wired straight to one part fails the last two. A supply
**gated** by the parts' activity, `intake = s q_A q_B/(K + q_A q_B)`, holds all four.

1. **The gated forcing is genuinely non-exact.** `(d omega)_AB = alpha(gamma - beta) - alpha s
   q_A K/(K + u)^2` with `u = q_A q_B`, reproduced to `1e-10`, equal to `-0.1141566` at the
   settled state. It does not involve the covector returned at the supply's port, whose
   response is zero, so no bookkeeping there removes it, and it varies with the state, so no
   constant `beta - gamma` does. The construction therefore sits inside the book's category,
   whose forcing need not be exact (§1), and outside the attention operad, whose forcing is
   the full differential of a nonnegative cost.
2. **One-way supply is not what costs exactness; a non-reciprocal response is.** The *linear*
   loop is exact to `1e-12`, with potential `alpha(q_A s + beta q_A q_B)`, even though its
   supply is one-way — because the supply's response is zero. This is §3.5 item 1 one case
   over, and the second time the same belief has been corrected here, so the rule is the thing
   to carry: unilaterality in the wiring does not make a forcing non-exact.
3. **Which verdicts the parameters decide.** Prediction 1 needs the supply above a threshold
   `s_*` set by the gate's half-saturation, and prediction 2 needs the internal loop gain below
   the dissipation, `alpha^2 beta gamma < b^2` — at `beta gamma = 1` the unsupplied pair holds
   its form for nothing, and above it the pair diverges. The formalism does not enforce that
   inequality. Near its boundary the run *length* decides what a run sees: at `0.9801`, two
   hundred time units report the form held at `0.500` while it is still decaying, so a
   persistence claim needs a rate, not a final reading.
4. **What the wiring decides, and it is not a parameter.** Predictions 3 and 4 hold for every
   parameter setting swept (eighteen cases, `beta gamma` from `0.04` to `1.1025`, `s` in
   `{2.2, 4, 40}`, both cuts killing both parts in all of them), provided the wiring has no
   ungated path from the supply to either part: cutting the partner's wire makes the gate read
   zero, so the part has no drive at all. Any ungated leak, however small, breaks them
   strictly. And **autocatalysis is not mutuality**: a gate reading a part's *own* activity
   leaves that part held alone at `2 + sqrt 3`. What the predictions need is that every path
   from the supply to each part pass through the other part's signal.
5. **A restart threshold, structural.** The forcing vanishes identically at the origin and the
   origin is stable, eigenvalues `-b ± alpha sqrt(beta gamma)`, so a pair started too low never
   gets going — the fixed point with vanishing gradient that §3.5 also found. The threshold is
   the lower root's stable manifold, a curve and not a number: on the ray through the saddle it
   is the saddle at every mobility, and off it moves with the step to first order.

**What this does not establish.** That arrangements tend to fall in the region where the four
predictions hold; §2's warning applies in full, since exhibiting one loop that behaves as
wanted proves nothing about the class of loops that do. What the runs deliver is the boundary
of that region. The gate's Michaelis–Menten shape is a **model** of metabolism, not a
derivation. The supply is a zero-response box, so nothing depletes and the finite-reservoir
lifetime bound is a separate matter. And the decay of an unsupplied amplitude is this
library's deterministic descent, not an entropic force: nothing here is random, and the
scattering of an unkept form is supplied by an environment the model does not draw.

### 3.9 Turnover, and two beliefs it corrected

`tests/test_turnover.py` (6 tests; construction and numbers in `notes/TURNOVER-TEST.md`) asks
whether a collective's form survives swapping a part. A spatial-attention parent holds a cost
`(1/2)(nu - h~)^2` on the weighted signal its redundant contributors send, each contributor
gating what it sends by a health configuration; one health is driven to zero and a spare sits
off to the side. Four predictions were fixed in the module docstring before the construction
existed. The collective degrades to `2 + (nu-2)/(1+lam)` with no spare and returns to `2.5`
exactly with one, the spare moving only as far as it had to — so **failure, forgetting and
replacement are all in the mathematics**, for this construction and in `2 < nu < 3`. Two of
the four predictions are refuted, and both refutations are the useful part.

1. **A cost on the weighted signal cannot see weight.** A part reporting zero at full weight
   and a part gone silent are literally the same input to it — the forces agree to the last
   bit. So a failed part is pushed out for contributing something *wrong*, never for being
   *absent*: a corrupting part is pushed at `+0.954051` where both of the others sit at
   exactly `0.0`. The predicted mechanism, weight-zero versus weight-nonzero, lives entirely in
   the mean readout, where it holds exactly and where §3.7's finding applies — that cost leaves
   the affine-quadratic family and its runs do not settle. **The aggregate that makes turnover
   settle is not the aggregate that makes a failed part get pushed out**, and one cost does not
   give both.
2. **The pull region is a budget, not a boundary.** The inward force is strictly nonzero at
   every finite distance and the deficit multiplying it does not vanish while the spare stays
   out. A spare at `d = 5` fails to arrive at 8k, 32k, 128k and 512k steps and arrives at 2M;
   the gradient reaches zero only at floating-point underflow, `d = 33.6665`, which is 37 times
   the kernel's own scale. **Reach is a rate, not a radius**, and §3.5's "a participant placed
   far out never arrives" is a statement about a budget. Every reach claim in this library now
   names its step count.

**What makes a part swappable here, and it is not being part of a collective.** The contributors
are interchangeable because the parent's cost is on an **aggregate** of what arrives, so only the
total matters and any sender is as good as another. Swappability is a property of the cost being
on what arrives rather than on who sent it — not a property of being a part. An organ sits closer
to that case than a person does, which is why a heart transplants and a best friend does not
(David, 2026-09-26). So this run models transplant, not friendship: a collective's form survives
turnover **where its form depends on a total**, and where it depends on which part is where, the
construction says nothing.

Two things the run settled that no prediction named. **The collective's own deficit props the
failing part up**: its health settles at `(nu - h~)kappa_F/lam`, so it dies fully only once the
spare has closed the deficit. And **only the signal returns**: the outgoing message is
`(h~, w~)`, and where the failed part keeps reporting, `w~` stays raised permanently, so the
collective is restored in the coordinate its cost is about and marked in the coordinate it is
blind to. A fully dead part is neither pushed out nor pulled in — contributing nothing, it
feels no force and its placement freezes where it is.

**What it does not establish.** Failure itself is stipulated: the health cost is a model of
breaking down, not a derivation. The pre-failure fixed point is not isolated — the whole level
set `sum kappa_j = nu` is fixed, which is what redundancy is here — so the settled split among
parts is undetermined. At `nu = 2` exactly the fixed point is degenerate, the force cubic and
the deficit falling as `1/t`, so a run reporting `2.0000` there would be reporting its own
budget. §2's warning applies in full.

### 3.10 The monitor: seeing, pulling, and the two put together

Three tests, each with its predictions fixed first: `tests/test_monitor_seeing.py` (5),
`tests/test_monitor_pulling.py` (6), `tests/test_monitor_composed.py` (6), with
`notes/MONITOR-SEEING.md`, `notes/MONITOR-PULLING.md`, `notes/MONITOR-COMPOSED.md`. The
question is whether one part can both see a dead part and pull a replacement in from far off.
Each builds on §3.9's collective rather than rebuilding it.

**Seeing.** The monitor is one more child of the same parent, frozen at the origin so it hears
exactly what the collective sends, carrying the curious reader's cost
`(1/2)(w - pi)^2 + (w_pi/2)(pi - nu)^2` on the total arriving weight. Settled surprise is
`s* = -(w_pi/(1+w_pi))Delta`, matching to `1e-13` in all 48 cells, so a drop registers iff
`Delta >= theta(1+w_pi)/w_pi`, and since the largest available drop is one part's share there
is a floor under a useful precision, `w_pi = theta/(kappa - theta)`. A free prediction *is*
surprised, for five steps, `s_t = -Delta·2^-t`; what it cannot do is stay surprised. The
prediction that motivated the construction is **refuted under the cost the collective
requires**: a silent failure is already visible to listening alone there (force `-0.4688` on
each live part), and is invisible only under the mean readout. Two results beyond detection:

1. **There is no passive observer.** A held expectation is a cost and a cost is a force. The
   monitor's contribution to the forcing is exactly `-s·grad(w~)`, inert to the last bit iff
   the surprise is zero, and at the silent failed state it *doubles* both the parent's prop-up
   of the failing part and its inward pull. A monitor that ends up detecting nothing has
   already moved what it was watching.
2. **The monitor repairs.** Against the reporting failure — the case §3.9 found the required
   cost cannot push out — a monitor at `w_pi = 1` drives the dead part's kernel from `0.8333`
   to `0.5000`, exactly until the weight it expects arrives. It supplies the sense the parent's
   cost lacks, and the collective uses it. So the gap §3.9 opened closes at a price: what buys
   both a settling turnover and a failed part moved out is a part holding an expectation of
   what should be arriving.

**Pulling.** Two routes. Nesting (a parent one level up with its own larger `tau`) stays exact
and trades reach against selectivity strictly: at a budget of 8000 steps reach grows as
`sqrt(tau)` and residual attention to a bad contributor grows at the same rate, and two levels
have both in one run where neither single level does. The doodad — a monitor whose own state
sets `F_p = -alpha·a·tanh(p/wid)` on the spare's placement with nothing returning — is an
ordinary non-exact arrangement and recruits in exactly `100d - 245` steps at any distance,
against two million steps for the exact route at `d = 5`. So nesting is *forced* only if the
forcing must stay exact. The unpredicted finding is the price:

3. **Making the pull exact bills the puller.** The exact completion is unique up to a function
   of the alarm alone, `U = -alpha·a·wid·log cosh(p/wid)`, so the placement must act back on the
   monitor — exactly what "nothing returns to it" forbids — and that back-action is the
   outstanding work of bringing the spare in. Billed, the alarm crosses zero at `p = 50.173`
   and the pull reverses beyond it: at 40 000 steps the billed monitor arrives from `d = 40`
   and not from `d = 60`, while the unbilled arrives from 60. **The exact pull has a boundary;
   the non-exact one has none.** This is §3.5 item 1 and §3.8 item 2 from the other side: the
   rule there was that a non-reciprocal *response* is what costs exactness, and here restoring
   exactness is exactly restoring the reciprocity. The boundary belongs to this completion,
   whose restoring cost lets the alarm go negative; one confining it to positive values would
   slow rather than reverse, and that is untested.

**Composed.** The two wired into one closed arrangement, the single new thing being that the
alarm is the surprise, `a = pi - w~`, rather than a held constant.

4. **Seeing and pulling are one quantity, not two composed capacities.** Delivery rate is
   `1/(mu·alpha·a*)` with `a* = (w_pi/(1+w_pi))Delta`, matched at `atol = 0` across
   `w_pi = 0.25, 0.5, 1, 2, 4`; at `w_pi = 0` the settled surprise is exactly zero and the
   spare never arrives at any budget. **The dial that sets how small a drop a monitor can see
   is the dial that sets how fast it can fetch.** The two thresholds differ, though: seeing has
   a floor at `w_pi > 0.0638` and delivery needs only `w_pi > 0`, so a monitor too coarse to
   report a drop still delivers against it.
5. **Delivery stops itself**, because the expression that pulls is the one that vanishes: every
   distance comes to rest with alarm `0.0` and pull `-0.0` to the last bit, with no stopping
   rule added, and the pre-failure state has composite forcing exactly `0.0`. Wire in a *held*
   alarm instead and it never rests: it holds the collective past what it asked for and, from
   nearby, drags the spare all the way in while the live pair pays.
6. **Which failure sounds the call is not the construction's to choose.** A silent failure
   sounds it at once; a reporting failure leaves `w~ = nu` exactly at the failed state, so
   nothing is called and the doodad never goes. The monitor *repairs* the failure that never
   calls and *delivers* against the failure it cannot repair. A failure that would call both
   halves at once is untested and is the obvious next question.

Where the composition earns its keep is one cell: a part gone silent with the spare beyond the
parent's own reach (`d = 20`, the parent's force on the spare about `4e-108`). There the bare
collective sits at the degraded value forever, the monitor alone sits with its surprise open,
and the loop restores the whole outgoing message exactly. With a reporting failure the loop
restores the *weight* and not the signal — the exact reverse of §3.9 — because
`w~ - h~ = kappa_F` identically.

**What the three do not establish.** The supply that a non-exact coupler needs by the physical
reading is **stipulated away** in both the pulling and the composed runs, and their notes say
so: over a trip the monitor delivers finite work and its own state ends where it began, bit
for bit. So "a collective that can replace its parts must pay for a monitor" is *not* shown —
§3.9 replaces a part with no monitor whenever the spare is already within reach. What is shown
is the conditional: a collective whose spare lies beyond its own attention cannot recover
without one, and that monitor's coupler is measurably not a descent of energy
(`max|d omega| = alpha` exactly). Whether such a collective is the normal case, this code does
not answer. Running a plan is not built: call, basin and delivery compose with nothing added
because there was nothing to plan, the spare's route being a straight descent along one
coordinate with no obstacles. The reach/selectivity trade-off is a measurement on one kernel
family, one cost and one budget, not a theorem. And the monitor was free only because
redundancy makes the resting place a manifold, so its transient moves along it: with a silent
failure the outgoing message is identical with and without it, while the live pair rests
`0.0109` apart. §2's warning applies throughout.

### 3.11 The fire and the insurance agency: saying what no part says, and being charged for a past

Two tests with their predictions fixed first: `tests/test_fire.py` (5) and
`tests/test_insurance.py` (5). The first asks whether a collective can report something none of
its parts reports; the second, whether a forecast can be held and charged when it is checked.

**The fire.** Two scouts, one reporting where a fire is and one how hard the wind blows; the
parent is charged the square of the distance from its report -- the mean readout of the message it
sends onward -- to where the fire will be.

1. **A collective that only weights and adds can only average.** With the value maps at the
   identity the report is `sum_j kappa_j w_j hbar_j / sum_j kappa_j w_j`, every coefficient
   positive: a convex combination, so it stays between the scouts' reports whatever the placements
   do, and its distance from the truth has a floor. Descent approaches that floor (`2.00024`
   against a floor of `2` after 20 000 steps) from five different starting placements and never
   crosses it. One clause of the prediction was too strong: the *ratio* of the two attentions
   reaches `1.03e4`, but the favoured scout's own `kappa` stops at `0.959`, because a cost on the
   mean readout is blind to the total weight. Both placements are still moving at the budget, so
   this is a rate.
2. **One scene is a line of value maps, not a rule.** With the maps free the report does leave the
   hull and reach the truth, geometrically at the exact rate `1 - mu|a|^2`. But the solution set is
   a line, and descent lands on the projection of wherever the parent started; the ratio of the
   settled maps is `0.4348`, not the elapsed time `4`, and a different start settles elsewhere on
   the same line. Charged on three scenes with **one pair of maps shared by every scout of a
   kind**, the quadratic is positive definite and descent settles at the unique solution: the map
   ratio *is* the elapsed time, and the parent is right about a fourth scene it was never charged
   for (`3.6e-15` against a truth of `0`). Two independent scenes already suffice.

**What the fire run does not establish.** There is no time in it: where the fire turns out to be
enters the cost as a constant target, exactly as `nu` does in §3.9, so nothing here holds a report
and is charged for it later. The sharing of one map across scenes is a constraint imposed on the
transfers, not a discovery -- in the spatial family each child has its own value map. And the
scouts are frozen and cost-free, so nothing here is about how a scout comes to report what it
reports. §2's warning applies: what the run shows is whether descent finds the rule and how many
scenes pin it down, not that the world contains collectives of this shape.

**The insurance agency.** A premium rate held as state, a fund accumulating premiums in minus
claims out, and a gain `g` with which the fund pushes the premium.

3. **A record is one-way, and the one response that repairs that is an amplifier.** On
   `(p, f)` the composite's `d omega` is `1 - g`, the only nonzero entry: no correcting response
   (`g < 0`) is exact, and the unique exact completion `g = +1` raises the premium when the fund
   is full. It runs away as `(1 + mu)^n` -- the premium at `-605` after 200 steps, the insurer
   paying people to insure them. This is §3.10 item 3 from the other side: there the exact
   completion billed the puller and cost it its reach; here it reverses the control.
4. **A pay the forecaster reads does not move the forecaster.** Give the insurer the cost
   `(c/2) f^2` on the fund arriving at its port and its premium does not move, bit for bit, over
   2000 steps: the forcing on a coordinate is the cost's derivative in *that* coordinate, and this
   cost does not mention the premium. What it charges is the fund. This is the `notes` design
   taken literally, and it is the same lesson as §3.9 item 1 -- ask what the cost is a function of
   before predicting what it will move.
5. **Charged on its record, a forecaster cycles and lags it by a quarter turn.** With a correcting
   response the loop is a rotation: `E = (p - lam)^2 + c f^2` is multiplied by exactly
   `1 + mu^2 c` each step, so it does not settle -- the cycle is the arrangement's and the slow
   growth is the integrator's, two explicit steps reading each other's old position. Started
   holding exactly the right rate with a fund in deficit, the premium rises at once, and its local
   maxima fall at steps `315`, `1571`, `2828`, exactly the steps at which the fund crosses zero.
   The debt is paid precisely when the over-charging is worst.
6. **A view of its own settles the loop at a balance it never clears.** An exact cost
   `(alpha/2)(p - pbar)^2` makes the rotation an inward spiral to `p = lam` with a permanent
   balance `alpha(pbar - lam)/c`, zero only if the view was right.

**What the insurance run does not establish.** There is no risk in it: the dynamics is
deterministic and the claim rate is a constant, so "bearing risk" is not modelled. There is no
pooling and there are no customers, so nothing makes over-charging costly; the forecaster is
charged for being wrong in either direction, not for being unprofitable. And the claim rate is
exogenous by construction -- which is the honest version of the one-wayness, since an insurer does
not cause fires.

### 3.12 Randomness: two random-variable polynomials and their samplers

`arrangements/rand.py`, `tests/test_rand.py` (7 tests). Before this module nothing in the
library was random except seeded draws of initial states, weights and data.

**What it builds.** Two polynomials, each a `tri`-monad (`bind`) and a non-commutative
`(x)`-monoid (`tensor`, independent product): `rand = sum_n Delta_n y^{underline n}`, finite
random variables (`FiniteRV`), and `rand_G = sum_n y^{R^n}`, random variables on a standard
normal seed (`GaussianRV`). The second is the reason Gaussian noise needs no separate
semantics: a polynomial of Gaussian *laws* is not closed under mixture or nonlinear
pushforward, but random variables on a Gaussian seed are, and concatenation of seeds is their
monad and their independent product. The measure sits on the directions, not the positions.
A stochastic parameter `CotCoalgR` returns a random variable over next states; it tensors
with ordinary parameters by the unit, so `wiring.py` composes stochastic arrangements
unchanged and `Phi` of one is the stochastic dynamics functor, its state update a random
variable. A `Sampler` is a `[rand, y]`-coalgebra; `seeded` pairs it with a stochastic
parameter into an ordinary `CotCoalg` on the product of states. The two kinds do not mix in
one tensor (the mixed polynomial `sum Delta_n y^{underline n x R^k}` is not built).

**What the tests are evidence of.** Strong, by closed form and by independent computations
agreeing: the finite law of four kicked steps computed by the monad's `bind` through `Phi`,
by the seeded arrangement run on all 81 replayed index sequences and weighted, and by the
closed form, agree to 1e-12, so the pairing is the monad's iteration on this instance; the
Gaussian law of six Euler--Maruyama steps by `bind` is affine in the seed and its mean and
covariance equal the closed form to 1e-14; a replayed seed gives the bound random variable's
value exactly. Statistical: a 40,000-step pseudo-random run has the discrete chain's
stationary variance `(T/a)/(1 - eta a/2)` within 4%, a third above the Gibbs variance `T/a`,
so `euler_maruyama` samples the chain, not the Gibbs law; the bias is the step's, exact for a
quadratic cost. Whether the key sampler's stream is distributed as demanded is JAX's, not
checked here beyond that run.

**The asymmetry, located.** `A (x) B` and `B (x) A` have the same law (checked exactly on
three steps). What the order decides is which part of one shared sample each box reads, so on
one shared seed a box's path depends on its position in the product; seeded box by box it
does not. How much depends on the sampler: the key sampler inverts one uniform through the
cumulative product weights, so in the test's pair the order changes A's draw only on a band
of probability 0.1 per step, and a 12-step run happened to show no difference at all. Nothing
yet uses the asymmetry; any result that changes under the swap will show it here first.

**Not established.** Langevin stationary laws, escape rates, and the continuous-time limit
are analytic theorems about these chains, not consequences of the construction. The
symmetry-restoring quotient of `rand` is not implemented.

## 4. Where the mathematics is written down

- The category of arrangements, the forcing, the coalgebra, and the dynamics functor:
  *Articulating Forms that Matter*, Chapter 3.
- Retroactivity, the reciprocity lemma, Kirchhoff's current law as the transpose of
  copying, and the three couplers: the same, Chapter 2, following
  Del Vecchio–Ninfa–Sontag and Sontag's lectures on retroactivity.
- The attention operad, its substitution, and its three worked examples: a companion
  paper on affine quadratic arrangements, in preparation.
- Everything ported from CDLM cites CDLM's own labels (`ex.`, `prop.`, `sec.`) in its
  docstring, so a reader with that draft can find the statement being implemented.
