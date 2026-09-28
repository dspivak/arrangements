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

**Running and extending tests.** Follow `README.md`, “Run it”: use
`scripts/run_tests.sh LIMIT_SECONDS TEST_FILE_OR_NODE_ID [...]` for an explicitly selected
small batch, and read its full output before choosing the next. Use `--list` to find files.
The runner streams progress, preserves logs, and reports a failed or cut-off piece as a failure.
Estimate the duration first; if progress projects near the limit, stop and split the selection
by pytest node ID. A limit is protection, not a target running time.
`tests` is a package: import another test's helpers as `from tests import test_pain as a`
(the pattern in `tests/test_pain_reserve.py`), not `import test_pain`.

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

### 3.13 The collective, and dispersal

`arrangements/collective.py` builds, once, what four earlier tests each built by hand: many
parts on one spatial-attention parent, placements moved by the parent's integrator, with the
kernel weights and the message sent onward as instruments. The choices that decided those
runs -- total or mean cost, target, kernel scale, the placements' integrator and mobility --
are required arguments. `tests/test_dispersal.py` (5 tests; predictions fixed first) asks
whether a part driven out of the collective comes back.

**What held.** (1) The onward message returns to its target whenever the other parts can
supply it. (2) The configuration does not: a small outward displacement of one of `n` parts
at a common equilibrium is recovered by that part only in the fraction `1/n` (measured
0.4927, 0.3267, 0.1951, 0.1215 at `n` = 2, 3, 5, 8), each other part moves `delta/n` inward,
and nothing further happens in another 20,000 steps. This is a theorem in the linear regime
and is built into the total cost: its minima form an `(n-1)`-dimensional family, and descent
removes only the displacement's projection on the one direction the cost sees. The run shows
the nonlinear kernel and the discrete step keep it. (3) A part displaced to distance 3 is
called back when the rest cannot cover the target (`n = 2`, target 1.5) and stays at 2.992
for 100,000 steps when they can (`n = 3`). (4) A collective displaced whole returns, in a time
set by distance and not by count (2244, 2230, 2232 steps at distance 4 for `n` = 1, 2, 4; the
lone-part quadrature gives 2215), growing like `exp(d^2 / 2 tau)`.

**With momentum.** `tests/test_dispersal_momentum.py` (4 tests; predictions fixed first) runs
the same collective under the phase and damped integrators, a part *pushed* by an initial
momentum. The cost's Hessian at a common equilibrium has rank one, so the push splits into a
spring (the total) and free directions carrying the fraction `1 - 1/n` of its energy. (5)
Undamped, each part's momentum averages to its free share (0.6744 of the push for the pushed
part at `n = 3`, against 2/3): that energy never comes back to the spring. (6) Damped, the
free share coasts `mu xi_0 / c` and stops, and the configuration lands where predicted to three
digits with the message at target. (7) Undamped, at target 1.5 and push 0.05, three parts
lose the pushed one (past distance 5 by step 190, at 109 by step 5000) and two never do (at
most 1.09 in 100,000 steps): the other part reaches the kernel's peak and turns it back.

**With noise.** `tests/test_scattering.py` (2 tests; predictions fixed first) puts
`rand.kicked_configuration` on the parent, kicks of `+-0.05` on every placement each step,
and runs many seeds in one `vmap`. (8) With the need present, 82% of three-part runs lose a
part in 20,000 steps and no two-part run lets a part past 1.32: noise does what a push does,
and redundancy decides. (9) With the parent's cost removed, each placement is a simple random
walk, and the fraction back in reach matches the exact binomial at `t` = 10^2 .. 10^5. The
prediction that it falls by `sqrt(10)` from `10^3` to `10^4` was refuted (2.15): the walk's
spread was still inside the reach. From `10^4` to `10^5` it falls by 3.31. A scattered part on
a line comes back, ever more rarely; nothing here says what happens in more dimensions, where
the walk is transient.

**What this does not establish.** The Gaussian kernel and the cost on a total are choices, and (2) follows from the second; a cost
that saw each part separately would call every part back. "Redundancy makes dispersal
irrecoverable" is what this model shows, not what collectives in general do.

### 3.15 A candidate beginning of a reader, as a composite

`tests/test_reader_origin.py` (stage A: 9 tests). The candidate and its one-box ODE are
Codex's (book repository, `notes/reader-origin-model.md`): a catalytic apparatus `q_1` whose
activity on held fuel rebuilds it and makes a boundary `q_2` that cuts its leak; a pore lets it
establish, the boundary lets it persist after. The design is `notes/reader-origin-arr.md` there.
Predictions were written in the docstring before any construction; all held.

**Reproductions.** The composite (species boxes, held supplies, one stateless transition box per
term, a routing wire) ticks as Codex's Euler step to 1e-14 and recovers his table. `d omega` equals
his formula and comes from exactly two children, boundary production and the protected leak.
Release works as a substitution of the pore child.

**(T1b) The field is a descent after all.** Derived by hand, then checked: on the open quadrant
the field is `-G grad U`, with the diagonal response `G = h diag(q_1, (k_b kappa F / l)(1+q_2)^2)`
and an explicit `U`. The structured composite (exact children, those responses) ticks with the
direct one to 1e-13, and `U` falls at every tick of the release run. The two curl sources are one
exact child, the cost `l q_1/(1+q_2)`. The price: the response ratio is forced up to a constant,
so the fuel, the match and the leakage sit in the boundary's response (its matter), not in a child.

**(T2b) Protection by mass-action binding** `A + B <-> C`, the complex not leaking. The other
terms are apportioned so that the totals obey Codex's equations with the true free amount.
- At fast binding (`k_on = 20`, `K = 1`) the candidate establishes and persists, at
  `q_1 = 1.717` against Codex's 1.788.
- Release fails below `K_* = 0.323`.
- At slow binding (`k_on = 0.1`) it establishes in the pore (1.863) and is lost on release.
- The `(A, C)` separability ratio moves from 0.823 to 3.354 as `B` goes from 1 to 4, so this
  chemistry has no species-box potential. T1b's potential belongs to the two-coordinate
  reduction.

**Stage B** (the same file, now 14 tests, one of them a strict xfail that keeps a refuted
prediction on the record; predictions fixed before the construction).

(T3) The match is a distance: the substrate drifts at 0.005 per unit time, and the match is the
kernel between placements. The fold's closed form is `d_* = 0.722` (`kappa_* = 0.770`).
- *Held.* Collapse at 0.812, inside the predicted `[d_*, d_* + 0.1]`.
- *Refuted.* The prediction that the run tracks the upper branch within 0.02 failed. The run
  lags above the branch, by 0.042 at `d_* - 0.05`, and the lag grows toward the fold because
  the relaxation rate vanishes there.
- *Forced, not a finding.* Drifting back does not restore it.

(T4) Two variants share the boundary, with matches 1 and 0.9.
- Under crowding loss on held fuel, the worse match is excluded at the closed-form rate:
  measured slope -0.3005 against -0.3.
- Under self-limiting loss on held fuel, both coexist, and the better variant ends higher with
  its partner than alone (1.883 against 1.788).
- On a fed fuel, the worse match is lost first, below supply 3.587 against the better one's
  1.666. At supply 7.17, where both persist, the better variant ends lower with its partner
  than alone (0.468 against 0.639): the competition for fuel outweighs the shared boundary.
  That last one was a guess, and it held.

Which of these a run shows depends on the saturating loss, a choice the book's chemistry does not
settle.

**What it is not.** It does not validate any chemistry.

### 3.16 A passing note and a sustained note

`tests/test_sustained_note.py` (7 tests). The book's claim (plan §4.1 step 11, David's "i love
it!", 2026-09-26): *a passing note moves a reader; a sustained note moves where it comes to
rest.* One reader component `I -> <R | R>` with the Euclidean response `eta = 0.05`, its note
supplied as the exterior input (the source left out of the picture), and two costs linear in the
note `u`: uncommitted `(1/2) q^2 - u q`, and committed `(1/4)(q^2 - 1)^2 - u q`, wells at `+-1`,
barrier `1/4`, one well gone for `|u| > 2/(3 sqrt 3)`. Noise is `euler_maruyama`; the statistics
use a vectorized copy of the step, checked pathwise against `Phi` of the seeded arrangement to
1e-12. Predictions were written in the file's docstring before anything ran.

**What could not have come out otherwise.** The uncommitted reader goes to `u/a` under a held
note and back to 0 after a passing one, at rate `1 - eta a` (closed form). The quiet committed
reader: a held 0.6 moves its rest to 1.2212, beyond the old well; a passing 0.6 must stay on 64
steps to leave it in the other well for good, and one step fewer leaves it where it was; a held
0.1 only shifts its well to -0.9456 and it never crosses.

**What could have, and how it came out.** (3) Noise `T = 0.1`, the weak held note 0.1: the
fraction of readers in the right well goes from 0 to 0.853, the two-well Gibbs value 0.855. A
note too weak to move a quiet reader sets where a noisy one rests. (4) Released after settling,
the fraction relaxes to 1/2 at 0.91 of Kramers's rate. (5) After a passing note that flips every
reader, the mark fades at a rate whose logarithm against `1/T` has slope -0.2766 against the
barrier's -0.25 (within the predicted 20%), prefactors 0.82--1.02 of Kramers's over `T` from 1/16
to 1/8: what a passing note leaves lasts as long as the reader's own commitment holds it against
wear, a time exponential in the commitment's depth over the noise. (6) Observed, not predicted,
because the first run's passing note was four steps too short: a note held 60 steps moves no
quiet reader and 60--65% of noisy ones, so "held long enough" is relative to the reader's noise
as well as to its settling time. Prediction 5's premise took two reruns to meet (60 and 100
steps too short at `T = 1/16`); both are recorded in its docstring.

**What it is not.** A one-dimensional reader. That the note is copied into the reader is the
book's model, not something this run tests; a phase reader under a held oscillating note, and a
reader with a slow coordinate that could keep the note after it goes, are not run.

### 3.17 Spells and timescales

`tests/test_spells.py` (10 tests, two of them strict xfails that keep refuted predictions on the
record). David, 2026-09-26: a spell may put a reader "into a new equilibrium that's less aligned
with your surroundings … that could change things at another time scale, or you might settle
back down … but still you can be a part of some longer-time-scale note that's playing to inchant
something else." The surrounds are a weak note 0.1 held on every committed reader, `T = 0.1`;
predictions in the docstring before any of it ran.

**(1) Held.** A spell of -0.6 against the surrounds puts 96% of settled readers in the higher
well; they return to the Gibbs value (0.849 against 0.855) at 0.96 of Kramers's rate. A reader
stays 600 steps in the higher well and 3524 in the lower, ratio 5.87 against Kramers's 6.32: a
spell into accordance lasts about six times longer than one against it.

**(3) Refuted twice, direction held.** A slow habit `w` paying `(beta/2)(q - w)^2`. Predicted:
a fast habit keeps a spell, a slow one loses it. At `T = 0.1` every habit speed kept about half
(0.51 … 0.36): the 3000-step horizon was longer than the reader-and-habit pair's own
re-randomizing time (e-fold 623 steps, checked afterwards as a prediction), so it measured
equilibrium, not the race. At `T = 0.05` the direction showed, 0.84 kept with the fastest habit
and 0.51 with the slowest, but the predicted numbers failed (slowest under 0.2). Not tuned a
third time.

**(4) Held, one number refuted.** 2000 members each flip 9.4 times on average during a
6000-step note, while their mean holds at 0.703 +- 0.011: a held note no member holds. A slower
committed reader receiving that mean crosses and is still across 6000 steps after the surrounds
go and the mean has returned to 0.011. Driven by one member instead, it switches 5 times; driven
by the mean, once, never back. The predicted threshold (1000 steps too short) was wrong: the
threshold is between 800 and 1000.

**What it is not.** The members and the slower reader are not composed in the library; the
slower reader reads the members' mean as its exterior input. One-dimensional readers throughout.

### 3.14 The cascade: a fund that pays its parts, and how deep a shock runs

`arrangements/cascade.py` (workers, sources, a fund layer that wraps a collective into a unit,
a market) and `tests/test_cascade.py` (5 tests; predictions fixed first, results recorded in
its docstring). Two channels, supply and product; a fund with an overhead and a leak.

**Retroactivity, found in building.** Under exact composition a fund that pays by a gate on
its balance is not a record: the workers' want is pulled back through the wage wire onto the
balance, `g'(f)(alpha sum kappa a + beta sum kappa)`, and props it up (the threshold rises from
20.22 to 21.78). The runs use a one-way wire that cancels that term exactly, read off the
collective's onward message; the composite is then non-exact. Which one an account is, is a
modeling choice this construction forces into the open.

**What held.** One level: a threshold, below which every worker stays and above which the fund
collapses and the workers leave for the source (all-or-none, but by symmetry, so untested);
with no source they fade in place. The threshold rises with the number of workers only through
the balance, not the margin. Nested to depth 4: a permanent cut runs to every depth; a shock
that stops the top's wage runs to every depth through funds without reserves and is absorbed
by the first level with them. **Refuted in part:** a moderate shock is attenuated level by level
even through fast funds, so depth depends on the shock's duration against the funds' response.

**What this does not establish.** Every threshold here exists because an overhead was put in;
the gate's shape, the leak, the two channels, and the geometry of the source are choices.
Workers are identical, so nothing here says how a heterogeneous collective comes apart.

### 3.18 An account restores the push, as data

`arrangements/account.py` (the cascade's one level with a third channel, a report) and
`tests/test_account.py` (7 tests; predictions fixed first, results in its docstring). Four workers
with differing `(alpha_j, beta_j)`, each sending forward `alpha_j a_j + beta_j`, its want per unit
of supply.

**A one-way wire that knows nothing of its workers.** The cascade's `record` wire subtracted the
pullback using the workers' constants. `wire="oneway"` instead applies `stop_gradient` to the
balance in the fund's sending map, so composition (`jax.vjp`) carries nothing back; the two agree
to 4e-15 on identical workers.

**What held, and why it had to.** A fund that reads the reports and adds `g'(f) sum kappa r` to
its flow reproduces the exact composite to 4e-15 through a shock. That is the chain rule (a
pullback is the sender's derivative applied to each receiver's own covector), so it shows only
that a worker can compute the report alone. Acting on the report so creates money: the exact fund
survives a drain the one-way fund does not (thresholds 21.77 and 20.22) because its workers' need
adds about 1.27 to its balance in the twenty steps after the shock; the one-way fund started from
the exact fund's equilibrium still collapses. One clause of the money prediction was ill-posed
and is marked so in the test.

**What this does not establish.** Anything about what a fund that conserves money should do with
a report. And the report's parameter cannot be learned by descent inside the composite: with a
report `theta (alpha a + beta)`, the composite's forcing on `theta` is exactly 0 under either wire,
even with a cost on unmet need, because nothing reads the report with a push.

**Stage (b), no report: a want that levels off at a need** (`needful_worker`; tests 23–25). With a
solvent fund, the workers the wage overpays back away from it, the least needy farthest, and the
underpaid stand together; nobody leaves at any cap tried. Under a drain the two neediest leave at
once. Linear workers are not strictly all-or-none: a passing shock catches the one with the largest
standing want first, since want sets speed. Two of the three predictions were refuted and the third
was ill-posed (a standing want cannot match balances); its after-the-fact version, matched by wage
cap, has the needful collective's threshold 4% higher. That result leaned on a mistake: the cost
made need a target, so excess supply cost as much as shortage.

**Rerun with need as a floor and the elsewhere felt** (`floor_worker`, tests 26–28). No one backs
away; the four stand together at every cap, since a want that always rises with pay leaves every
worker the same landscape: need sets how fast a worker moves, never where it settles. With the
elsewhere at 1.5 the two supplies overlap into one hill and the workers drift rather than leave;
released, the neediest reach the elsewhere first. No drain ruins the fund, because it is paid for
product the elsewhere's supply bought, a flaw shared with `cascade.py` (book plan P24).

### 3.19 The ratchet: shock, disperse, re-form, shock again

`tests/test_ratchet.py` (6 tests; predictions R1-R7 fixed in its docstring before any construction
code) on `cascade.py` unchanged -- this one adds no mechanism. A collapsed fund is dead and stays
dead, so re-forming comes from outside: an injection into the fund, or the source's offer falling
for a while.

**The placement law, by hand and checked.** A part's placement ascends the supply it receives,
`h_1(p) = g kappa(p) + s kappa(p - d)`, and its amplitude enters only as a positive rate, so the
stationary placements are exactly the stationary points of `h_1`, independent of the amplitude and
of the other parts. With `rho = g/s` the landscape is bistable for `phi_min < rho < phi_max`, whose
closed forms come from the roots of `p(d - p) = tau`.

**What held.** The two critical wages are `phi_min s` and `phi_max s` to five decimals, in two
geometries, and are the same for one part and for four: no part calls another back. With the wage
capped at the outside offer no injection re-forms anything, and the money drains away. When every
part returns, the re-formed collective *is* the original -- balance, placements, amplitudes and the
second break threshold all identical. A part left out is not gone: it still sends product in from
where it stands.

**Refuted.** The critical injection is 89.5% above the quasi-static landscape value, because the
balance decays while the parts walk: the wage must be high enough *for long enough* (at 2.5 times
the walking speed it drops from 38.86 to 27.88). And there is no band of injections that re-forms a
collective short a part, however unequal the walking speeds: at that wage cap a survivor of any size
pays above `phi_max s`, so the first ones home hold the door open and a part twenty times slower
still walks in. Where a survivor cannot pay it, a finite window does give the band.

**What it leaves.** Membership, and only where the door closes: the collective re-formed short one
part breaks at a third of the original shock and has no buffer left -- a scar, never a ratchet.

**Found while building.** At a wage cap of 4 the placement update at the cascade run's mobility 0.1
is unstable, a 2-cycle -0.361 <-> +0.413 that reads as two equilibria; every equilibrium reported
here is guarded by `one_step_move`, the largest coordinate change in one further step.

**What this does not establish, and it is what the question was about.** Anything about a
collective whose parts hold each other: here every part is supplied by the fund and by the
elsewhere, never by another part, so the parts depend on one another only through the paymaster.
The same cycle on a collective whose parts supply each other is the book plan's P21 stage (b).
Nor anything stochastic: this is all deterministic descent, so "the door closes" means a basin
boundary, not a probability.

### 3.20 Pain: a tracked probability of coming apart, and what it pulls in (stage a)

`arrangements/pain.py` (the cascade's one level on three channels, supply, product and pain) and
`tests/test_pain.py` (8 tests; predictions a1-a7 fixed in its docstring before any construction
code, results there). A worker splits its effort between product and learning; its skill multiplies
its product and decays without upkeep. The fund carries a second coordinate, a lagged fixed reading
`phi(f)` of its own balance, and sends it down; a worker reads it through `stop_gradient` (the pain
wire is one-way) and moves its learning share toward `logit(0.3) - Lambda h_3`. Shocks are
`rand.kicked_configuration` drains of the balance, the same seed giving the same shock times with
pain and without. A whole has come apart when its workers' mean placement passes 1.25.

**What held.** Pain shortens the window in which a second drain brings a young whole apart, 205
steps to 36. Under repeated drains set on the pain-free whole alone, pain lowers the fraction coming
apart from 68% to 38% and nearly doubles the skill still in reach at the end, since a whole that
comes apart keeps almost none (by the kernel, not by a rule). Where drains cannot bring it apart,
pain costs 0.1% of growth: a growing whole's balance soon leaves the range where they hurt. A
permanent cut the pain-free young whole cannot survive (from 24%) it survives with pain (to 40%).

**Two costs of pain.** Cuts of 38-40% leave a young whole together in lasting pain, a
fixed point with no learning; with pain that pulls in less steeply (`Lambda = 3, 5`) the same cuts
survive and learning resumes, and the survival edge (41%) is the same. And a grown whole with pain
comes apart under cuts of 52-54% that the pain-free whole survives, 18,000-31,000 steps later, as
the stopped learning decays and the income it paid for goes with it; less steep pain (`Lambda = 3`)
narrows that band to 53-54% but does not remove it.

**Refuted.** Pain buys more of the moment than predicted: the single-drain threshold rises 13%, not
under 10%, partly because `phi` reads 0.004 at rest and a whole with pain rests at a higher balance.
The band where pain brings apart a grown whole is 0.03 of cut wide, not 0.05.

**Tracking.** From the same balance, 80% of young wholes come apart within 1,000 steps and 5.5% of
grown ones; `phi` reads both 0.92. A reading of the balance alone is calibrated for the whole it
was set for and overstates a grown whole's danger seventeen-fold.

**What this does not establish.** Anything about a reserve or the combination with learning
(stage b). Any claim that pain's reading is learned: `phi` is fixed, and a whole cannot learn its
own probability of coming apart from its own life, since it comes apart once. And every number
rests on chosen rates (learning, forgetting, pain's lag and steepness).

### 3.21 A collective whose parts supply each other, and what a shock can do to it

`arrangements/mutual_supply.py` and `tests/test_mutual_ratchet.py` (12 tests; predictions B1-B8
fixed in its docstring before any construction code, results there). The book plan's P21 stage
(b): §3.13's collective and §3.14's cascade have placements and an elsewhere but no part feeds
another, and `test_mutual_need.py` (§3.8) has mutual need but no placements. One part, `supplier`,
puts them together -- it passes a share `sigma` of its activity on as supply and gates what it can
do with what arrives, `I(r) = A r/(K_s + r)` -- and the spatial parent's transfers deliver it,
since each box already receives the kernel-weighted sum of every *other* box's message. The fund is
the cascade's, and removable.

**Two things the construction taught before any prediction was tested.** Composition pulls each
part's cost back onto what its neighbours send, so a part is driven by *their* need for its supply;
that term is a tenth of the settled amplitude and the hand-derived law in the docstring is wrong by
it. And because the kernel is symmetric, the whole arrangement is exact: amplitudes and placements
ascend one potential `U = sum_j [(alpha a_j + beta) I(r_j) - (b/2) a_j^2]`, checked against the
library's forcing at a scrambled state to 1e-9. (The cascade is not a gradient system, and §3.8's
gated pair is not either; the symmetry of the kernel is what puts it back.)

**A line between passing on and making.** A fragment of `m` parts holds a form with nothing arriving
iff `2 A sigma (m - 1) > K_s` (critical share measured 0.06252 at `m = 3` against 0.06250). Below
it parts can only relay what something outside supplies; above it a pair lights itself from a seed
of 1e-12. A super-critical loop is a free lunch standing for an ambient gradient the model does not
draw -- the same move as the cascade's market paying twice the product. Everything below is above
the line.

**What mutual supply changes is not where a collective goes but whether it can be taken apart.**
The offer that moves the *body* is the cascade's, `g/phi_min`, measured 2.7161 against 2.7177 and
the same for one part and for four, because what the parts pass each other travels with them. But
one part cannot be bought out of four at any price: at offers of 10, 20, 40 and 1000 it settles at
0.1085, 0.1835, 1.1061, 1.2951, each at the peak of `U` along its placement, relaying rather than
leaving. Moving a group of four home costs a wage of 2.72; taking one part out of it costs 39.07.
One part at home calls a stray back at the wage that could call nobody in §3.19.

**So a shock moves this collective and does not break it.** Booms of 200 to 20,000 steps with the
mobilities spread eightfold leave 0 or 4 at the elsewhere and never spread the four by more than
0.024 (the control at `sigma = 0`: 2.172); when the boom ends the collective returns to its first
equilibrium to six decimals. A split is stable only once the pieces are out of each other's reach
(merging at `kappa = 2.0e-2`, holding at `kappa = 4.5e-5`), and nothing here produces one. The scar
§3.19 measured is the fund's, not the collective's: the bare threshold does not depend on size,
while the drain that kills the fund falls 32.8% when a part is missing.

**What decides is a total.** Every quantity here depends on `sigma (m - 1)` and on nothing else:
the critical share is 0.06251, 0.03127, 0.01562 at `m = 3, 5, 9`, and collectives of 3, 5 and 9
parts passing the same total hold the identical amplitude 3.464102. Many small passes and few large
ones are indistinguishable, which is §3.9's swappability one level up.

**What this does not establish.** Anything about a collective *below* the line, where the parts
relay rather than make (not run). Anything about needs of different kinds, or about needing a
particular other part: `sigma` is one number (the plan's stage (c)). And the elsewhere is an ungated
supply at a place, which is what makes dispersal possible at all here; the model has no land, no
houses and nothing sunk, so its collective is attached to its members and to supply, never to a
place.

### 3.22 Pain, stage b: a reserve in the fund, and the combination with learning

`tests/test_pain_reserve.py` (8 tests; predictions b0-b6 fixed in its docstring before any
construction code, results there), adding to `arrangements/pain.py` behind defaults that leave stage
a unchanged. Pain now pulls in the share of growth equal to its level (`worker(drive="level")`:
learning share `s0 (1 - h_3)`). The fund's reserve (`fund_layer(reserve=...)`): a share
`0.05 (1 - pi)` of income set aside, drawn into the balance at `2 pi rho`, leaking at `0.01 rho`; the
control draws only once the balance is gone.

**What held.** With the level drive, learning resumes after every permanent cut a young whole
survives (to 41%): no trap. A stock does not answer a lasting loss: grown survival edges are 39% with
no growth and with the reserve, 53% with learning; the reserve buys about 100 steps. Both kinds
together share no budget and do better against drains than either (44.5% coming apart against 53.5%).

**Refuted.** At stage a's drain rate every whole that does not learn comes apart; the reserve
answers slower drains (45% against 68% at one per 1,000 steps) and raises the largest single drain
survived from 21.5 to 30.8, but learning is ahead at every time measured (34.1 against 26.9 after
1,000 steps). The reserve drawn only by the event beats the one drawn by pain (3% against 45%): the
pain reading is never zero, so a proportional draw spends the reserve at rest (11.6 held against
37.8). The combination does not remove the grown whole's cost at a 54% cut.

**Pain zero on a good day** (`fund_layer(floor=15)`, after David: "no pain in the distracting
sense" on a good day; b6). Nothing is spent at rest (both reserves hold 38.35), and the pain-drawn
reserve beats none against slower drains (40% against 78%), but the event-drawn control still does
better (10%): a drain brings pain near full, the proportional draw pours half the jar into a balance
that would have recovered alone, and there it leaks ten times faster.

**What this does not establish.** Which mix of the two kinds is right: they share no budget here,
so there is no mix to choose, and with one the answer would be the rates'. Anything about a draw on a
threshold of pain.

### 3.23 Bayes' rule is a different integrator and nothing else

`tests/test_bayes_integrator.py` (20 tests; the derivation and five predictions fixed in writing in
`notes/BAYES-INTEGRATOR.md` §§1–3 and quoted in the module docstring before any construction code).
The book plan's P25 stage (a), from David's question: is Bayesian update "the same sort of thing,
just a different integrator?"

For a likelihood in an exponential family and a belief in the conjugate family with natural
parameter `lam` and log-partition `AA`, one observation translates the belief,
`lam -> lam + t(x)` with `t(x) = (T(x), 1)`. The cost `k_x(lam) = E_{q_lam}[-log p(x|theta)]` is
affine in the expectation parameter, so the **forcing** is the exact `-d k_x`, components
`G(lam) t(x)` with `G = grad^2 AA` the Fisher information; the **response** is
`sharp_lam = G(lam)^{-1}`, the raised inverse Fisher metric. In the library this is
`coalgebra.configuration` on `rvect.ReactiveVectorSpace(2, lam -> inv(fisher(lam)))` with
`arrangement.exact(-k_x)` — `learning.py`'s two constructors with one argument changed.

**What held**, all five predictions. Conjugate Gaussian, unknown mean, known variance; prior
`N(2.0, 1.5)`, `sig^2 = 0.7`, seven data. Posterior mean 0.74107142857142949 against the closed
form's 0.7410714285714286 and variance 0.09374999999999982 against 0.09375: relative errors 1.20e-15
and 1.92e-15, so Bayes exactly and not to first order. Batch equals sequential to 4.2e-14. Duration
is likelihood tempering (`sharp = tau G^{-1}`, the fixed-duration specialization): one step at `tau`
lands on `lam + tau t(x)`, and two half-steps equal one whole step. `Phi` of a composite moves the
belief identically, to 0.

**The response is what does the work.** The Euclidean response at `eta = 0.01 … 2.0` descends the
same cost (28.03 to between 13.7 and 18.6) and lands on the wrong belief every time, and fails the
batch-equals-sequential test. The Newton response is a different matrix
(`[[3.214, -8.679], [-8.679, 26.679]]` against the Fisher `[[1.5, -3.0], [-3.0, 7.125]]`) and its
step `(1.330, 0.838)` is nowhere near Bayes' `(3.190, 2.095)`.

**Three findings not predicted.** (a) Bayes fits the *learner's* slot as well as Newton's: the belief
box can report its expectation parameters `mu = grad AA(lam)` forward and force nothing, with a
downstream likelihood box carrying the cost and pushing `t(x)` back; the pullback along the Legendre
map is `G(lam) t(x)`, so backpropagation supplies the parameter covector, exactly as for a network.
(b) Khan and Rue's objective (`E[lbar] - H(q)`, prior inside `lbar`) needs its entropy term to make
the step a *replacement* of `lam`; in `arr` the prior is the integrator's initial state and the
increment is `configuration`'s `s + sharp(xi)`, so the forcing carries only the datum's loss and no
entropy term appears. Both give `lam^(0) + sum_i t(x_i)`; checked against their Eq. (54). (c) A Bayes
arrangement never settles: `t(x)` is never zero and `G` is positive definite, so the forcing vanishes
nowhere; the cost's infimum is on the boundary of `Lam`, infinite precision with the mean at the
datum, and the orbit on a repeated datum runs to it as `1/n` without arriving.

**Reasonable or weird** (the agent's judgment, not evidence). Reasonable for a cell: a log-odds
accumulator is the standard reading of chemotactic adaptation, and its want of a set point is what a
leak supplies — a second forcing, not another integrator. Weird for a person at long horizons: a
thousand sightings of the same thing should not multiply precision by a thousand, and the model has
nowhere to say the data are not independent, which is the *wiring's* fault and not the integrator's.
Weird and informative for a market: batch equals sequential says news order does not matter, right
for Bayes and wrong for markets, so path dependence there must come from the wiring, a non-exact
forcing, or a state carrying more than a belief.

**What this does not establish.** Nothing about non-conjugate Bayes, where the mirror-descent step
is an approximation. Nothing about a belief over a general distribution: that is an
infinite-dimensional configuration and needs a `cot` there (§3.1's admissible subcategory does not
reach it). Nothing relating this to the Bayesian lens of Braithwaite–Hedges–St Clere Smithe, which is
a lens on Markov kernels rather than on `omegalens`. Sources and locators are in
`notes/BAYES-INTEGRATOR.md` §5.

### 3.24 Discrete choice is the same integrator, and the response supplies the normalization

`tests/test_discrete_choice.py`, 28 tests, `arrangements/choice.py`; derivation, predictions fixed
before any construction code, and verdicts in `notes/DISCRETE-CHOICE.md`. Book plan P25 stage (b),
from David's question: does discrete choice come in "through softmax / stochasticity"?

For `n` options with values `v_i` at temperature `T`, the **chooser** is the configuration
coalgebra on the simplex with the Shahshahani response `sharp_p = dt (diag(p) - p p^T)`
(`replicator.py`) and the **exact** forcing `d(<v,p> + T H(p))`. Its rest point is the logit
`softmax(v/T)`, to relative error 1.5e-16 in two options and 2.7e-16 in three, and the rest point
does not depend on the duration. At `T = 0` the step *is* the forward-Euler replicator step for the
linear fitness `v`, agreeing with `replicator_dynamics` step for step to 5.4e-20: **choice and
selection are the same descent at two temperatures.** So softmax is not a fourth integrator.

**The normalization is the response's contribution, not the cost's.** `1^T sharp_p = dt (1 - sum p)
p^T` is zero exactly on the simplex, which annihilates the constant part of the forcing and the
Lagrange multiplier normalization would have introduced. The negative control shows it: the same
cost, forcing, wiring and interface with `rvect.euclidean` settle on the *unnormalized* Gibbs
weights `exp(v_i/T - 1)`, summing to 1.368 and 7.245 in the two examples, more than 0.5 from the
logit.

**The same distribution is one Bayes step in the Legendre-dual chart** (§3.23 on the categorical
family). Gauge-fixed natural parameters, inverse-Fisher response, affine forcing: one step at
duration 1 from the uniform belief lands on the logit to 8.6e-16. The gauge is needed — on all of
`R^n` the Fisher information is singular along `1` — and that is a wrinkle the continuous conjugate
families of §3.23 did not have. The two responses agree under the Legendre map to 2.8e-17, but the
arrangements differ: the mean-chart jump on the affine cost linearizes softmax, missing by order
`|t|^2` (errors 0.104, 0.030, 0.0078 at `eps = 0.5, 0.25, 0.125`), so **which chart the state lives
in is part of the arrangement.**

**A made choice is a sample, and it needs no extension of the framework.** The collapsing chooser is
a `(cot(Q) tri rand)`-coalgebra whose branches are the constant maps onto the vertices, so it is
inside `poly(SmSet)`; `key_sampler` gives frequencies within 0.0066 of `p` over 20000 draws
(bound 0.0283), `FiniteRV.bind` gives the exact two-step law with no sampler, and that law equals
the one-step law, so it is **absorbing** — a chooser that stores its choice cannot reconsider, while
one that records it without feeding it back can. What is outside the smooth setting is `argmax`: its
derivative in the utility gap falls 0.196612, 0.000453958, 0.0 at `T = 1, 0.1, 0.01` while the choice
at a tie stays 0.5, so the limit is locally constant with no smooth extension across the tie.
**The jump is not the obstruction** -- nor, it turned out, is
determinism; see the basin half below, which withdraws that.

**Two duration bounds, and the stability one is the looser.** The multiplier at the rest point is
exactly `1 - dt*T`, so stability wants `dt*T < 2`. But what binds first is positivity of the
transient, `dt * max_i(xibar - xi_i) < 1` at the state the jump leaves; below zero the entropy's
`log` gives `nan`. From the uniform start at `v = (2, 0.5, -1)`, `T = 0.7` that is `dt < 2/3`,
measured between 0.66 and 0.67, where the stability bound would have allowed `dt` up to 2.857.

**The simplex is exactly invariant but not attracting** (not predicted, and the caution to carry
away). The Jacobian's second eigenvalue, transverse to the simplex, is `1 + dt*T*(1 - log Z)` for
`Z = sum_j exp(v_j/T)`, above 1 whenever `Z < e`. Then rounding-level sum error grows
geometrically — 8.4e-15 at step 100 to 7.0e-5 at step 600, then `nan` — so exactness of
`sum p = 1` holds per jump and fails over a run. Adding a constant to every `v_i` changes no choice
and multiplies `Z`, so a model-irrelevant shift decides whether the run survives, and the shift
repairs it. `replicator.py`'s "self-correcting" drift assumes `x^T A x > 0`; the entropy-regularized
forcing is not of that form.

**What this does not establish.** Nothing about where `v` comes from, so no reading of `v` as a
preference. Nothing about heredity, which is P25 stage (c): the `T = 0` case is differential
persistence under a shared supply, which §3.13 and the book's *select* already have. Nothing about a
changing option set, `n` being fixed in `rand`'s positions and in `Q`. Nothing empirical about real
choosers. Nothing about choice over an uncountable option set (§3.1). And nothing about a chooser
*coupled* to what it chooses among: every arrangement here is a closed scalar with `v` fixed from
outside.

### 3.25 Choosing by basin: the same arrangement, coupled, commits with no coin

Same file and note as §3.24 (52 tests together); the Part-two predictions were fixed in writing
before any new construction code. David, 2026-09-27, on §3.24's report: "can you think of another
way? how does nature do it? it's not basins or something?" It is, and this is the better half.

§3.24 fed a **fixed** value for each option in from outside. One fixed value vector gives one rest
point, so the only route to a discrete outcome was the external coin of `collapsing_chooser`. Let an
option's value depend on the shares already committed -- a symmetric payoff `A` in place of the
linear `<v,p>`, so `U(p) = (1/2) p^T A p + T H(p)` -- and the same arrangement, same Shahshahani
response, same exact forcing, has **several stable rest points, each with a basin**, and chooses
with no coin and no temperature. One family, two knobs: the coupling `A` and the temperature `T`.
`A = uncoupled_payoff(v)` gives §3.24's chooser, `T = 0` gives `replicator_arrangement`, and a
coordination `A` with small `T` gives commitment.

**The basin boundary is an unstable rest point.** For two options, `A = diag(a_1, a_2)`, `T = 0`:
`p_1* = a_2/(a_1+a_2)` with `g'(p_1*) = p_1*(1-p_1*)(a_1+a_2) > 0`. Located by bisecting the
starting share, the outcome flips at `0.6666666667` for `a = (1,2)` and `0.8000000000` for
`a = (1,4)`, matching to 1e-6; the rate matches to 1e-12 and each vertex is an exact fixed point.

**This chooser is not an optimizer.** With payoffs 1 and 2, every start above 2/3 commits to the
option paying **1** (564, 493, 444, 395 steps from 0.67, 0.70, 0.80, 0.95); 0.66 commits to the
option paying 2. What decides is where it started.

**Near a tie it slows; it does not jump.** The jump multiplier at the boundary is
`1 + dt g'(p_1*) > 1`, so steps to decide is linear in `-log` of the distance from the boundary with
slope `1/log(1 + dt g'(p_1*))`: measured 30.478 against 30.497 at `dt = 0.05` and 15.4950 against
15.4946 at `dt = 0.1`, over distances 1e-2 to 1e-8. **This withdraws §3.24's "determinism is the
obstruction, not the jump."** A deterministic chooser is available and is inside the framework;
what has no smooth extension is `argmax` **as a function of the option values**, and a basin chooser
spends that discontinuity on time instead.

**Temperature dissolves the basins at `T_c = a/2`.** Rest points solve
`(Ap)_1 - (Ap)_2 = T log(p_1/p_2)`, so at any `T > 0` no vertex is a rest point -- the entropy's
logarithm diverges at a corner while the payoff stays bounded, and a hedging chooser settles on a
committed *share*, never on one option. For `A = diag(a,a)`, writing `p = ((1+m)/2,(1-m)/2)`, the
condition is `m = tanh(am/2T)`, with a nonzero root iff `a > 2T`: a pitchfork at `T_c = a/2`. This
is **mean-field Ising at coupling `a/2`** -- a two-option coordination game with an entropy term *is*
the bistable switch, which is why this is a mechanism and not an analogy. Measured `T_c = 0.49999996`
against 0.5 (grid-limited, low by construction), three interior rest points below and one above,
the outer pair matching the closed form to 1e-4, and the exponent of `m` in `T_c - T` equal to 0.4984
against 1/2. Above `T_c` every interior start hedges at `(1/2,1/2)` to 1e-9.

**No coupling, no basins.** With `A = v 1^T + 1 v^T` there is exactly one interior rest point, §3.24's
logit, and the constructor agrees with `logit_arrangement` step for step to 2.8e-17 and, at `T = 0`,
with `replicator_dynamics` to exactly 0. So what §3.24 lacked was not a construction but the
dependence of an option's value on its own share.

**Reasonable or weird.** Reasonable at every scale, and the payoff: a chooser that commits is one
whose options' values depend on how many already chose them -- a cell committing to a type, a
convention locking in, nucleation, a gene switch. Reasonable and uncomfortable: the worse option can
win, which is path dependence for a society, habit for a person, commitment for a cell. Reasonable:
deciding takes the logarithm of how close the call was. Weird for a person, right for a population or
a material: above a critical temperature no start can decide. Weird as a limit: `T_c = a/2`, the
boundary and the slowing rate are all two-option results, so no book sentence names a critical
temperature without saying "two options."

**What this does not establish.** Nothing about where `A` comes from: that an option's value rises
with its share is the assumption doing all the work. Nothing for a non-symmetric payoff (the
potential-game scope limit of `replicator.py`). Nothing for more than two options at positive
temperature. Nothing about a chooser wired to another box whose state feeds its values -- the
coupling here is a frequency dependence inside one closed scalar, and that is the honest remaining
gap. Nothing about heredity, which is P25 stage (c).

### 3.26 A choice composed: what pushes it moves it, what reads it tilts it

Same file and note as §§3.24-3.25 (63 tests together, 389s -- select node IDs, not the whole file);
the Part-three predictions were fixed in writing before any composition code. David, 2026-09-27, on
§3.25's report that a wired chooser was untested: "you're saying that we can't tell if discrete
choice can actually affect other things? that is not useful." The sentence was worse than the truth
and is withdrawn.

**What was and was not open.** That these arrangements compose is not in question: `arr` is a
category, `Phi` a functor, and §2's `test_composition.py` checks both operad legs including a forcing
from no potential. Composition is **pullback-and-sum on the forcing** and leaves the integrator --
response, state, readout -- alone. So the only open question for a closed-scalar result is whether
its conclusion survives an **added forcing term**. That is what was untested.

**A box whose forcing is EXACT cannot read a choice without tilting it.** This refuted Part three's
own first prediction, and not by a bug. Comparing a three-box composite (`pusher -> chooser -> follower`) against a hand-sum containing
only the upstream push deviates by 3.2e-2 and moves the boundary to 0.3876 from the predicted 0.4,
because the downstream box that merely *reads* the chooser's share contributes `(y - q_1)` to the
chooser's forcing. That is **retroactivity**, §4's Chapter 2 subject, arriving unasked. Compare the
right things and composition is exact: the two-box composite (nothing reading) equals the hand-summed
push to **0.0**, and the three-box composite equals a hand-sum including the follower's pullback to
**1.1e-16**.

**An outside push moves the boundary linearly, and above a threshold replaces the choice.** For
`A = diag(a,a)`, `T = 0`, a push `b` entering as `b(q_1 - q_2)` gives `p_1* = 1/2 - b/a`: located at
0.39999994, 0.70000001 and 0.50000012 for `b = 0.1, -0.2, 0` against 0.4, 0.7, 0.5. Bistability needs
`p_1*` inside `(0,1)`, so `|b| < a/2`: at `b = 0.6` every start (0.02, 0.2, 0.49) commits to option 1,
while at `b = 0.4` the start still decides, with the boundary at 0.1.

**And it is hysteresis.** From `p_1 = 0.2` a push of 0.6 commits the chooser to option 1 in 59 steps;
release the push and it stays, drift below 1e-12 over 2000 steps, where unpushed from the same start
it would have gone to option 2. A vertex is an exact fixed point and the response vanishes there, so
**no finite push returns it**. Downstream, two runs differing only in the chooser's start leave the
follower at exactly 0.0 and 1.0.

**Reasonable or weird.** Reasonable at every scale, and the result to keep: influence has a threshold
and its effect is permanent -- a convention, a habit, a differentiated cell, a standardised market.
Reasonable and important: being read changes the thing read, which is loading in a circuit,
measurement changing behaviour in a society, and retroactivity here -- not an extra assumption but
what pullback-and-sum already says. Weird for a person: nothing here can be persuaded back, because
there is no noise at the committed state; `rand.py` has it and this construction did not use it.
Weird as a limit: the threshold `a/2` is two options with a symmetric coupling, so no sentence should
give a number for how much influence flips a choice.

**A unilateral reader is a non-exact forcing** (David, 2026-09-27: "couldn't we just make it be
unilateral?"; checked after the predictions, so reported as an addendum). The tilt is entirely
`omega_g`'s component on its incoming value `m`, which the composite pulls back onto the upstream
configuration. An exact `omega_g = dU_g` that depends on `m` has `dU_g/dm != 0` and must tilt; a
non-exact forcing with the same `Q`-component and a ZERO `m`-component reads and loads nothing.
Measured: unilateral reader against no reader at all, chooser deviation **0.0** over 200 steps,
against 3.2e-2 for the exact reader, while the unilateral reader still tracks to 1e-15. Its
antisymmetric Jacobian part is 1.0, so it is genuinely not closed. This is the ideal-voltmeter
idealization and should be said as one; it does NOT make non-exact forcings idealizations in general
(the attention cue and `replicator.py`'s direct realization are neither).

**What this does not establish.** Whether exactness is what forces reciprocity throughout the
library, or only here -- the commons run's "a fixed-rule external enforcer is exact rather than
unilateral" points the same way by a different mechanism, state-dependence, and joining them is
P26's business. Nothing about noise at a
committed state, so nothing about a revisable commitment. Nothing for `n > 2`, a non-symmetric payoff,
or a positive temperature under a push. Nothing about a **cycle**: pusher and chooser form a chain,
not a loop, so a chooser whose own choice changes the push it receives is untested.

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
