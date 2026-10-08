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

**Not evidence of an event: an intervention.** A run is one arrangement running. Substituting one of
its boxes, changing a coefficient, cutting a supply or failing a part by hand between phases is the
experimenter's act, and the record never tells it as something the model did ("released," "it
leaves," "the part fails," "the spare arrives"). `test_reader_origin.py`'s T2a, the pore child
replaced by a zero drive, compares two arrangements started at the same state; it is a fact about
each one's dynamics and no event in either, and the book repository told it as the reader leaving
the pore until David cut it (2026-10-08: "there is absolutely nothing in the math about swapping out
one diagram for another … why not go in between each step of the dynamics and swap out numbers to
suit some story?"). Two arrangements may be compared, each run as itself; an input may vary in time
through a port the arrangement has, said as the environment's schedule. What the story needs to
happen is produced by the dynamics or not claimed (the wording Claude Fable 5.1's, b79de173).

**Not evidence of a cause: a quantity that rose alongside.** A diagnosis that names what grew with
the error has found a suspect, not a cause, until a control moves the one without the other. The
categorical Bayes step's loss of exactness on thin beliefs was put down on 2026-10-01 to the dense
inverse, `cond(G)` rising with the error; a prediction built on that was refuted on 2026-10-03, and
the control that would have shown it at once took one run: the same stream with the gauge's
reference moved to the winner, still through the dense inverse, and the error gone (§3.38;
`notes/SCIENCE-SCALES-PREDICTIONS-2026-10-01.md`, its 2026-10-03 correction). Write a cause with its
control beside it, or call it a suspect (suggestion-box sweep, Claude Opus 5.5 (1M), f0c52066,
from the mark of Claude Fable 5.1, 4f679514; *proposed*).

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
whether a part started far from the collective is called back (nothing drives it out; it is
started displaced).

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
activity on held fuel rebuilds it and makes a boundary `q_2` that cuts its leak. Two arrangements
are compared throughout, each run as itself: the *pore* arrangement, whose pore box publishes a
retention of 3.8, in which the candidate establishes from a trace; and the *exposed* arrangement,
the same with that box replaced by one publishing 0, started at the pore arrangement's high state,
in which the boundary keeps the high state up. "Release" below names that second arrangement and
its run; nothing leaves anything, and the replacement is the experimenter's (§2, "Not evidence of
an event"). The design is `notes/reader-origin-arr.md` there.
Predictions were written in the docstring before any construction; all held.

**Reproductions.** The composite (species boxes, held supplies, one stateless transition box per
term, a routing wire) ticks as Codex's Euler step to 1e-14 and recovers his table. `d omega` equals
his formula and comes from exactly two children, boundary production and the protected leak.
The exposed arrangement differs from the pore arrangement in one child, the pore box, and in
nothing else.

**(T1b) The field is a descent after all.** Derived by hand, then checked: on the open quadrant
the field is `-G grad U`, with the diagonal response `G = h diag(q_1, (k_b kappa F / l)(1+q_2)^2)`
and an explicit `U`. The structured composite (exact children, those responses) ticks with the
direct one to 1e-13, and `U` falls at every tick of the exposed arrangement's run. The two curl sources are one
exact child, the cost `l q_1/(1+q_2)`. The price: the response ratio is forced up to a constant,
so the fuel, the match and the leakage sit in the boundary's response (its matter), not in a child.

**(T2b) Protection by mass-action binding** `A + B <-> C`, the complex not leaking. The other
terms are apportioned so that the totals obey Codex's equations with the true free amount.
- At fast binding (`k_on = 20`, `K = 1`) the candidate establishes in the pore arrangement, and
  the exposed arrangement keeps the high state, at `q_1 = 1.717` against Codex's 1.788.
- The exposed arrangement has no high state below `K_* = 0.323`.
- At slow binding (`k_on = 0.1`) the pore arrangement establishes it (1.863) and the exposed one
  falls to zero.
- The `(A, C)` separability ratio moves from 0.823 to 3.354 as `B` goes from 1 to 4, so this
  chemistry has no species-box potential. T1b's potential belongs to the two-coordinate
  reduction.

**Stage B** (the same file, now 14 tests, one of them a strict xfail that keeps a refuted
prediction on the record; predictions fixed before the construction).

(T3) The match is a distance: the substrate is a box with a response and a constant forcing from
one more child, so it drifts at 0.005 per unit time by the dynamics, and the match is the kernel
between placements. The fold's closed form is `d_* = 0.722` (`kappa_* = 0.770`).
- *Held.* Collapse at 0.812, inside the predicted `[d_*, d_* + 0.1]`.
- *Refuted.* The prediction that the run tracks the upper branch within 0.02 failed. The run
  lags above the branch, by 0.042 at `d_* - 0.05`, and the lag grows toward the fold because
  the relaxation rate vanishes there.
- *Forced, not a finding.* A second run, the drift box's forcing reversed and started at the
  collapsed state, stays at zero: an exact zero is a fixed point with no assembly.

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
its docstring). Two channels, supply and product; a fund with an overhead and a leak. A "shock"
or "drain" is the fund's balance reduced by hand, the run from there; the top supply is an
exterior input (`supplied`), so its cut is the environment's schedule; neither is an event in a
run (§2, "Not evidence of an event").

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
run from the cap-1.0 equilibrium at cap 0.2, the neediest reach the elsewhere first. No drain ruins the fund, because it is paid for
product the elsewhere's supply bought, a flaw shared with `cascade.py` (book plan P24).

### 3.19 The ratchet: shock, disperse, re-form, shock again

`tests/test_ratchet.py` (6 tests; predictions R1-R7 fixed in its docstring before any construction
code) on `cascade.py` unchanged -- this one adds no mechanism. A collapsed fund is dead and stays
dead, so re-forming comes from the experimenter: a sum added to the fund's balance by hand, or the
source box's offer set lower for a window of steps and set back; a "shock" is the balance reduced
by hand. Every phase is one run from the previous run's end state with one such change, and none
of these words names an event in a run (§2, "Not evidence of an event").

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

**What it leaves.** Which parts are in reach, and only where the door closes: the three-part
arrangement, the one a window leaves, dies from a third of the four-part one's drain and has no
buffer left -- "a scar, never a ratchet," a comparison of two arrangements and no event in either.

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

### 3.21 A collective whose parts supply each other, and what a change of offer does to it

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

**A change of offer moves this collective and does not break it.** The elsewhere is a held box;
its offer set to 12 by hand for a window of 200 to 20,000 steps and set back (the run's "boom"),
with the mobilities spread eightfold, leaves 0 or 4 at the elsewhere and never spreads the four by
more than 0.024 (the control at `sigma = 0`: 2.172); run on at the original offer from where that
leaves it, the collective returns to its first equilibrium to six decimals. A split is stable only once the pieces are out of each other's reach
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

*Note 2026-09-29 (§3.29)*: built with its state in `R^n`, the simplex is invariant but attracts
round-off only while `0 < dt T (log Z - 1) < 2`; with every value a cost (`log Z < 1`) the same
chooser drifts off it to the unnormalized Gibbs weights. On the simplex, or in exact arithmetic,
the rest point is the logit as stated.

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
duration 1 from the uniform belief lands on the logit to 8.6e-16. *Scope (2026-10-01, cause corrected 2026-10-03):* exact in
the mathematics, but this gauge-fixed step loses it once the gauge's reference value is thin, by about
`1e-16` over its height (§3.38; not the dense inverse and not `cond(G)`, as first diagnosed). An inverse
needs the gauge, since on all of `R^n` the Fisher information is singular along `1`, a wrinkle the continuous
conjugate families of §3.23 did not have; a response that gives Bayes' rule does not need it (§3.38). The two responses agree under the Legendre map to 2.8e-17, but the
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
run on from there without the push, it stays, drift below 1e-12 over 2000 steps, where unpushed from the same start
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

### 3.27 A form played long enough carves a rest that was not there

`tests/test_carving.py`, 6 tests, 4s; predictions fixed first in
`notes/CARVING-PREDICTIONS-2026-09-28.md`. The claim is David's (2026-09-28: "i think a new form can
start to carve out a new capacity. is that not something we've imagined yet? this is how memory works,
and how river beds work"), against §3.16's sustained note, which held the reader's wells fixed. One
reader, a fast coordinate `q` and a slow one `w`, one cost both descend, `w` being the curvature of
`q`'s landscape at the origin.

**A carved memory needs the deepening to saturate.** With a quadratic deepening against a linear decay
there is no stable memory at any `gamma`: above 1/2 the only rest is the blank one, below it the carved
rest exists and is unstable, `d/dw [q^2/2 - gamma(w - w0)] = 1/2 - gamma > 0` on the slaved branch, and
`w` ran away to overflow. The quartic term is the least that holds it. This was found in construction,
before any prediction was tested, and it is the file's first result.

**Held.** The hold threshold is 1337 steps, sharp to one: 1336 leaves the reader blank, 1337 leaves it
at the carved rest `(0.7051, 0.4972)` for good. Below the threshold the tuning keeps a **fading trace**,
`w = -0.402` from `-0.5`, going back on its own decay time `1/(eta_w gamma) = 10^4` steps -- so a note
too short to carve still leaves a mark to wear off, and the threshold decides whether the mark holds
itself up. The fast landscape's curvature at `q = 0` flips sign, so the reader ends with a rest it did
not have, and a kick that does not cross zero falls back into it. The joint cost never rises, note on or
off, so dissipation stays non-negative while the tuning itself moves.

**Nothing new in the joint state space** (closed form). The carved rest solves both fixed-point equations
at `u = 0` and was a rest before any note played. Carving is basin selection at the joint level; what is
new is the *fast* landscape at the slow coordinate's new value. The book says it that way.

**The warning.** A tuning driven by `q` rather than `q^2/2` is non-reciprocal, no single cost has both
rules as its gradient, and then the joint cost rises on 2915 of 8000 steps, by up to 0.80, ending at no
rest of the exact model. A tuning that carves is accountable only while it descends the same cost its
fast coordinate does.

**Rain, at David's ask** (2026-09-28: "should have been something like a weekly rainfall … if you get to
a certain amount, then something happens"; predictions 6 and 7 fixed before that code). With the note
arriving on a cycle, what decides the outcome is the percentage of the time it rains, not the elapsed
time: 9% carves and 8% does not, at periods 100 and 200 alike. **Prediction 7 was refuted in both
halves.** I predicted bursts beat drizzle at short periods, the deepening being convex, and drizzle beats
bursts at long ones, the tuning decaying between. At a fixed 9%: drizzle carves at periods 100 and 200,
the same water every 2000 steps does not, and every 20000 steps it carves again -- because one 1800-step
burst exceeds the 1337-step held threshold and carves by itself. So spacing is not monotone, and the
long-period win is the held note, not burstiness. Reading: regular light rain keeps a channel that the
same water in occasional storms does not, unless one storm is big enough to cut it alone. A rounding bug
caught here is worth naming: with `round(duty * period)` a short period delivers more water than its
percentage says, which made drizzle look better than it was; the on-time is now exact by construction.

**Not done.** No noise, so the sharp threshold is a one-dimensional artifact and the carved rest's
lifetime under wear is unmeasured (§3.17's Kramers rate is the tool). Nothing composed: a second reader
carving on what the first outputs is untested, and so is a cycle. Nothing about *where* the supply for
the carving comes from: the cost here is nobody's.

### 3.28 The town's mathematician: an earlier account changes a priced decision

**P29, run 2026-09-28**, Codex (GPT-6), session `0c750220`. Proposed model, checked by
11 tests in `tests/test_town_{smoke,law,insurance,account}.py`; construction `arrangements/town.py`.
Predictions fixed before code; all four held in their stated scope. Results and limitations:
`notes/P29-TOWN-RESULTS.md`; reproducible numbers: `notes/P29-TOWN-MEASUREMENTS.json`.

The plant ascends its current expected profit. Her estimate of harm, learned from recent
health observations and sent to the insurer, changes its quoted premium schedule before
claims settle. Cumulative harm over 30 units falls from 8.68145 to 3.88223 (55.28%); half
step gives 55.17%. Disconnecting her report reproduces the historical-insurance town.
An informed plant acts before equally delayed payments: **delay alone does not force late
action**. Observation registers carry no future data; actual payment transfers balance.
Non-exactness excludes a potential for this forcing, not all Lyapunov functions.

Reasonable (agent judgment): earlier evidence can affect a priced decision. Artificial:
noiseless health data, known recovery, one coefficient, automatic uptake, all harm covered,
and unlimited credit. Her insurer's final cash .143891 is less than unpaid claims .672598;
no solvency or risk-bearing result. Lower harm also lowers operating surplus. No model of
understanding a book or discovering the institutional wiring, no general welfare theorem,
and no biological interpretation of courts or money. Placement approved by David, 2026-09-28:
Ch 1's town story and Ch 5's construction.

**Stage (d), smoke per watt, run 2026-09-28** (Claude Opus 5.5 subagent of session `8f53ae48`;
proposed model, checked by 5 tests in `tests/test_town_clean.py`; `build(..., clean=True)` in
`arrangements/town.py`; predictions `notes/P29-D-PREDICTIONS-2026-09-28.md` fixed before code,
results `notes/P29-D-RESULTS.md`). The plant gets a second dial, cleanliness $c$, and smoke
$s=(1-c)x$ goes to the townspeople; power is published and no box consumes it; plant cost
$(1+p(1-c)^2)x^2/2-\text{benefit}\,x+k\,c^2/2$. All four predictions held in their stated scope:
unpriced, the plant never cleans at any $\beta$; priced, it cleans, and **whether it cleans or
shrinks turns on one ratio**, the cleaning cost $k$ against production's, with the split exactly
even at $k=1$ at every price (the cost is symmetric in $x$ and $1-c$ there); with her account it
starts cleaning before the first claim is paid; at $k=.1$ power over 30 units falls 3.4% against
63.5% with one dial, and harm falls 79.5% below the one-dial account town. Not predicted: from
$k\approx 8/9$ harm is slightly *higher* than with one dial (the plant spends cleaning on keeping
its power), though net surplus is higher at every $k$; cleaning starves her fit, which freezes at
1.674 against $\beta=2$; at $\beta=8$, $k=.8$ a plant priced in full from the start is trapped small
and dirty, which the town's gradually rising price avoids. Artificial (agent judgment): no one
consumes the power, quadratic cleaning cost, cleaning cost not scaling with output, noiseless
instant smoke-to-illness. Not established: what a real plant's $k$ is.
The two-dial plant publishes power and smoke, matching the book's §1.6.1 sketches (2026-09-29,
session `8f53ae48`; every number unchanged); the townspeople's slot, formerly `'health'`, is `'coughing'` (2026-09-29: the story's townspeople cough now, the illness it warns of coming later).
**Stage (e)**, `power_to_town=True` (2026-09-29, `8f53ae48`; `tests/test_town_power.py`, 9 passed;
`notes/P29-E-RESULTS.md`): the residents take the power and push back on it at a fixed price per
watt, the plant's revenue arriving as that push; every trajectory is (d)'s to 1e-12; the power
port returns the price, the smoke port zero. No new finding; the book's sketches now match.
**The book's town** (2026-09-29, `8f53ae48`): `eve_town()` is Eve's model from §1.6.1 (the two-dial plant,
power to the townspeople), with a reader-facing module docstring; `Settings.liability` is the value on the
liability wire at stage 1; `tests/test_town_story.py` checks each sentence of §1.6.1 the code stands behind
(7 passed; 32 town tests in all).

### 3.29 Active inference in three tests: precision by descent, policy choice, expected free energy

**P5 (b), run 2026-09-29**, Claude Opus 5.5, session `9b16a421`. `tests/test_active_inference.py`,
17 passed and 6 strict xfails (the refuted predictions, kept unrepaired); construction
`arrangements/active_inference.py`; predictions fixed before code, verdicts, weirdness and limits in
`notes/ACTIVE-INFERENCE-TESTS.md`. **(i) Held**: with `-1/2 log det Pi` in the cost, descent in
`(mu, log Pi)` settles at the inverse ML variance (4.073150457190358, 1e-10), and in 2-D at the
inverse sample covariance; without it `Pi` collapses harmonically; with two sources the learned
precisions are the weights on the mean. **(ii)** The chooser's cost is the policy free energy
(held), but with values `-G` it does not settle on the policy posterior (**refuted**): the simplex
is invariant, not attracting, when `log Z < 1`; `1 - sum p` grows by `1 - dt T (log Z - 1)` per step
and the state goes to `exp(v/T - 1)`. The drift is transverse to the simplex, so it belongs to the
state in `R^n` and float round-off; on the simplex, or exactly, the logit is reached (§3.24's note). A constant shift of the values
restores it (Addendum A, predicted after the failure). **(iii) Held** for the rollout (`Phi` of a
world arrangement and its Jacobian equal the closed form, 1e-12) and for the information gain
`1/2 log(1 + w_obs/w_pred)`; the choices are refuted unshifted by the same drift and hold shifted:
the switch from looking to pushing sits where the closed form puts it, and **exact wiring bends the
planner's belief 0.384 toward its preferred outcome** (item 30's retroactivity), while unilateral,
non-exact wiring leaves it on the evidence. Not established: a closed loop acting on a world with
sampled observations.
**Addenda B and C** (same session, at David's questions; predictions §9 of the note, all held):
the chooser in the natural chart (every state a distribution) reaches the policy posterior with no
shift, so (ii)'s refutation belonged to the chart; and David's picture nested — hand, a level
predicting it, a level predicting that level and wanting a goal — moves the hand toward the goal
by pullback alone, a chain of springs whose rest is `h* = g k_s/(kappa + k_s)`, with the offset of a
proportional controller.

### 3.30 Many parts that keep one another whole: a ring and an all-to-all of parts in mutual need

`tests/test_many_parts.py` (5 tests; predictions fixed in `notes/MANY-PARTS-PREDICTIONS-2026-09-29.md`
before any construction code, results in `notes/MANY-PARTS-RESULTS.md`; the book plan's P30 (e)).
`n` parts of §3.8's kind, each with its own gated supply `s g(r_i, q_i)`, `g = rq/(K + rq)`, where
`r_i` is what the parts it needs send — the neighbour behind it (ring) or the others' mean
(all-to-all) — and `beta r_i` an ungated internal path, run at `0` and `0.5`.

**One supply condition for every size and both wirings.** The threshold is
`s_* = 2 (b - alpha beta) sqrt K / alpha`, the pair's, at `n = 2, 3, 5, 8`, and the settled
amplitude is the symmetric root `q+` to 1e-6. The mechanism: the Jacobian at `q+` is circulant, its
eigenvalues reproduced to 1e-6, and the least stable mode is the symmetric one, which does not see
`n` or the wiring. The restart threshold of §3.8 survives (origin stable, the saddle the threshold).

**A cut ends a strict ring and grades a slack one.** With no ungated path (`beta = 0`) cutting one
part's supply kills every part in turn; with `beta = 0.5` the cut part settles at
`alpha beta q_{i-1}/b` and the ring holds, degraded in the direction of need (at `n = 8`, `s = 4`:
`3.92, 5.79, 6.80, 7.32, 7.59, 7.73, 7.80, 7.83` going round, the intact `7.87`). So "one loss ends
the ring" is §3.8's wiring condition — no ungated path to a part — one level up, and one ungated
fraction turns collapse into a graded loss. All-to-all loses the cut part and the survivors hold iff
`s > s_*' = 2 sqrt(K (n-1)/(n-2))` (a closed form, reproduced): at `s = 2.5` three parts collapse
(`s_*' = 2.828`) while five and eight hold, so **a larger all-to-all collective survives a loss that
a smaller one does not**, at a supply where every intact collective held.

**The collapse is a front.** At `n = 8`, `s = 4`, `beta = 0` the deaths go round in order with a
constant delay, `3.07` h-units (spread under 2%); it doubles when `b` is halved with `s/b` fixed and
grows with `s` (`2.43, 3.07, 3.91` at `s = 3, 4, 6`); the guess `(1/b) log(q+^2/K)` tracks the
`s`-dependence and is low by a constant `0.45`. Cutting a part's outgoing wire starts the same front
one part on. The ring's coupler is non-exact (`|d omega| = 0.069` at the symmetric state); the
all-to-all's is exact at the symmetric state and non-exact off it.

**What this does not establish.** That collectives fall where the pair's threshold holds (§2). The
gate is a model. Every part here has its own gated supply, which is why a symmetric state exists; a
chain relaying one supply from one end was not run and would attenuate (§3.21, below the line). No
placements, so nothing disperses; no needs of different kinds.

### 3.31 Word of mouth: the recruiting channel dies with the leaver

`tests/test_word_of_mouth.py` (6 tests; predictions fixed in
`notes/WORD-OF-MOUTH-PREDICTIONS-2026-09-29.md` before any construction code, results in
`notes/WORD-OF-MOUTH-TEST.md`; the book plan's §5.9 item 38, P30 (d)). §3.9's turnover collective
with one change: the forcing that places the spare is multiplied by the **word**, the share of the
recruiting channel still carried by parts still claiming weight (`rho_j` a wiring datum; silent
mode). With the word intact the trajectories are §3.9's to 1e-12. The wrapped parent is inside
`arr` and outside the attention operad, `d omega` between the leaver's health and the spare's
placement measured against its closed form to 1e-12.

**Slower, not lower.** The return to `nu` is exact for every nonzero word and its time scales as
`1/g`: `40, 60, 120, 380` steps at word `1, 2/3, 1/3, 0.1`. The deficit has no floor; the word sets
the rate, §3.9's "reach is a rate" in the recruiting channel. Only a leaver dead, frozen at zero
health, who carried the whole channel leaves the spare unmoved (`h~ = 2 + kappa(d)` forever) — and
a leaver merely *failing* is propped by the collective's deficit at `2e-5` health (§3.9), which is
`2e-5` of word, and the spare creeps in over 238,000 steps. **A faint part and an absent one are
told apart only by freezing the one at zero**, §3.9's blindness to absence, now in the word.

**Compounding is the channel thinning, with a cliff at the last carrier.** Three failures in turn,
recruits carrying no word: `60, 120, never` steps (word `2/3, 1/3, 0`); at 8,000 steps
`h~ = 2.5, 2.5, 2.370`. Monotone, and the jump is a wiring event, the last share being the last of
the word — not accumulation reaching a threshold, which this library has no mathematics for.
Recruits that take over the leaver's share once in: `60, 60, 60`.

**The dependent falls by its wiring; the collective's shortfall props everyone, the leaver
included.** Part 0 fed `0.6` through the leaver settles at `0.4395`, part 1 fed from outside at
`1.0202`, by `c_j = t_j + (nu - h~) kappa_j / lam_h` to 1e-9, the propping equal for both; the
leaver at `0.0323`, propped by the deficit and by its dependent's need (composition pulls the
dependent's cost back onto it, §3.21's mechanism). The starved part is not pushed out: low, not
wrong (§3.9).

**Effort aimed by the joint's account closes the whole; aimed by the whole's felt shortfall it
leaves both short.** A doodad (§3.10 route (b)) whose alarm reads part 0's own account `1 - c_0`
and adds `gain * alarm` to its supply returns the collective to `nu` in 60 steps at every gain,
leaving the joint a residual `(1 - t_0 - pi)/(1 + gain)`: `0.30, 0.12, 0.035` at gain `1, 4, 16`,
the proportional-control offset. With its reading wire cut (an insulator) nothing moves and the
collective keeps a deficit of `0.092`. Reading the collective's felt deficit `nu - h~` instead, it
cannot tell the joint from the leaver's loss and leaves both short (`0.02`, self-consistently). Its
coupling is non-exact, `lam_h gain + lam_a gain_r`, to 1e-9: it acts and nothing returns to it.

**Found while running.** The joint-and-alarm loop is a linear map per step; at a half step each its
eigenvalue modulus is `(1/2) sqrt(1 + gain)`, and at gain 4 it rang itself to `nan` after passing
through `nu`. The continuous loop is stable at every gain; the alarm now moves a twentieth per step.

**What this does not establish.** A floor: scaling the recruit's *contribution* by the word would
build one in, and was not run for that reason (§2). "Wears out" is stipulated (§3.9), "leaves" is
not a movement, the word is one number per part, the doodad's supply is stipulated.

### 3.32 Swing locking: inertia with damping keeps two oscillators in step; inertia alone does not

`tests/test_swing_locking.py` (4 passed); predictions fixed first and verdicts in
`notes/P32-C-PREDICTIONS-2026-09-29.md` (book plan P32 (c), §5.9 item 45). The Kuramoto boxes with
`coalgebra.damped(Q, c)`: the locked difference is Kuramoto's `arcsin(dw/2K)`, an exact fixed point of
the discrete map, the common advance `mean(omega)/(c m)` per step, and the approach overshoots iff
`c + a/m < 2 sqrt(a/m)` with `a = 2K cos(delta*)`. At `c = 0` nothing settles. Closed forms of the
discrete map, so the run checks the derivation; the undamped case could have settled numerically and did
not. Weird: undamped, the natural frequencies are constant torques on unloaded rotors, and the mean angle
runs to 1e5 rad.

### 3.33 A Kirchhoff node: the current law is the transpose of copying, at three branches

`tests/test_kirchhoff_node.py` (6 passed); predictions and verdicts in
`notes/P32-E-PREDICTIONS-2026-09-29.md` (book plan P32 (e), §5.9 item 46). A capacitor box copied to
three resistor couplers to three ideal sources steps by `-dt * sum_j I_j` to 1e-15 with no sum written
anywhere; a triangle of capacitors conserves charge to 1e-15 and settles at the capacitance-weighted
mean; two conductances rest at the conductance-weighted mean of their sources. The first three checks
are the mechanism (the vjp of a gather is a scatter-add) and could fail only by a sign slip. Limit found:
a nodal-analysis node holds no charge, and its algebraic constraint is not what a state-holding box
expresses; §4's "nodal equations as the composite ODE" holds for networks with a capacitor at every node.

### 3.34 Light in the library: the lattice Maxwell model as a composed arrangement

`arrangements/light_lattice.py`, `tests/test_light_lattice.py` (6 passed); predictions fixed first and
verdicts in `notes/P32-A-PREDICTIONS-2026-09-29.md` (book plan P32 (a), §5.9 item 47). An 8×8 periodic
grid of phase edge boxes and stateless face boxes with cost (kappa/2) B^2, wired by copy: the forcing
emerges from `compose_seq` and equals -d of the total face energy; one step equals `scripts/maxwell2d.py`'s
(kappa = dt^2, momenta dt·xdot) to 1e-12; the vertex divergence of the momenta is conserved to 1e-12;
the lattice dispersion holds to 1e-9. Refuted as written: the plain energy is not within a few percent
at step 0.5 (it swings 33% and 119%, bounded, no drift, linear in the step); the staggered energy
1/2 sum xi_n^2 + 1/2 <B_n, B_{n+1}> is conserved to 1e-12. Yee's TE scheme in the temporal gauge
(Stern, Tong, Desbrun and Marsden, arXiv:0707.4470, Thm. 5.1, §4.1). "Short waves run slow" is the
lattice's numerical dispersion, not a property of light. Hygiene: `_tensor_balanced` is imported from
`hopfield.py` and belongs in `wiring.py`.

### 3.35 Lotka–Volterra as the book's step: the plain step spirals out, the two-stage step cycles

`arrangements/lotka_volterra.py`, `tests/test_lotka_volterra.py` (7 passed); predictions fixed first and
verdicts in `notes/P32-B-PREDICTIONS-2026-09-29.md` (book plan P32 (b), §5.9 item 48). Prey and
predator boxes with log state and opposite-signed responses, a stateless hunt box with potential beta·x·y,
wired as CDLM `ex.lotka_volterra`: one step of the composite is forward Euler on CDLM's field to 1e-12,
the coexistence point is fixed, composite equals monolithic. The plain configuration step raises the
Lotka–Volterra invariant at every step (4.8e-2 per cycle at dt 0.02, halving with dt), in the log chart
and the population chart alike; the two-stage `CotCoalg2` step (predator first, prey second, `pc2.Phi2`)
bounds it (max |H − H0| 3.8e-3 over 22 cycles, second order). So "rabbits and foxes rise and fall in
turn" is true of the two-stage arrangement and false of the plain one over many cycles. Runner note:
`-s` passed to `scripts/run_tests.sh` is read as a selection and starts the whole suite; use
`PYTEST_ADDOPTS="-s"`.

### 3.36 The genetic code, sourced and as an arrangement

`notes/P32-D-SOURCES-2026-09-29.md`, `scripts/genetic_code_stability_sourced.py`, `arrangements/genetic_code.py`,
`tests/test_genetic_code.py` (9 passed, 72 s); predictions in `notes/P32-D-PREDICTIONS-2026-09-29.md` (book
plan P32 (d), §5.9 item 49). The from-memory inputs of `scripts/genetic_code_stability.py` were checked
against readable reproductions with locators (polar requirements: arXiv:0909.1442v2 Table I p. 3, after
Woese 1966; slip weights: arXiv:1309.4589v1 §2 p. 8, quoting Freeland and Hurst 1998; the primaries were not
readable and are not cited as read): every value matched, so the earlier table is a reproduction. On the
2008 recomputed polar-requirement scale the ordering survives but the threshold drops to β 1.0–1.5 and the
hard-code result weakens to 2–10 of 2000: the numbers are scale-dependent and are not to be quoted bare. The
arrangement has 386 boxes (61 codon boxes on the simplex in the log chart, 263 slip-cost edges, 61
definiteness, 1 repertoire); its full-code step equals the script's to 7e-16 and the slip cost emerges from
the wiring. The biologist's objection stands: polar requirement is a poor proxy for damage, so the model is a
descent, not a claim about which amino acid a cell can least afford. One step takes 30–45 s.

### 3.37 An arrangement that runs arrangements: a runner built from a kit, handed its own description

`arrangements/runner.py`, `tests/test_runner.py` (21 passed, about 80 s in five pieces); book plan P34, §5.9
item 56; the question and the design are in the book repo's `notes/self-hosted-arr-2026-10-01.md`. The kit is
`const`, `register` (unit response), `drum` (adds what it receives to its front cell, then rotates), `add`,
`mul`, `cost`, `push` (a one-way forcing) and the wiring `fan`. A *net* is a closed term over the kit. The
runner is one net: four flag drums, a data drum, a register, the constant -1, twelve boxes, two pushes.

**Strong evidence here.** (i) Two implementations sharing no code: three nets, and ten random ones, through
`denote` and `Phi` agree with the by-hand stepper `ref_step`; the two-tank net equals its closed form
`2 +- (9/10)^k` exactly in fractions. (ii) The translation `compile_net` is exact in fractions against
`ref_step` on those three nets and on 300 random nets using every kit part. (iii) The runner, taken through
the library's own composition, decodes after one pass of its flag drums to the net's own step (two tanks to
7e-16, the other two exactly); its compiled step equals `Phi`'s on the first two steps. (iv) A larger runner, handed the
translation of the runner's own net, runs each of the three nets exactly in fractions, and after every
guest step the host's decoded state equals the runner's net stepped by hand; with the host as a library
arrangement the tower was run for the two smallest nets. Slowdown: 182 runner steps per two-tank step;
2,889,432 host steps per two-tank step when the runner runs its own description.

**The runner as a part with ports.** `port_runner_net(cells, length)` is the runner with seven inboxes (a
load gate, a run gate, four flags, a number) and one output (its data drum's front cell); `asker_net` is a
part that sends a translated description from its own drums, one piece a step, has the runner run it, and
samples the returning cells into answer registers. Asker and runner are each a term over the kit plus
`inbox` (a register that becomes what arrives at its port); `couple` wires them into one closed arrangement.
One runner (16 cells, flag drums of 208; 43 nodes), three askers: after 837 steps each asker's registers hold
its net's state after three steps, exactly in fractions by hand, and through the library's composition
(state dimension 5055) to 7e-16. A description needing more cells or steps than the runner has is refused by
the translator: nothing the runner hears can change its drums' sizes. No covector crosses between asker and
runner; only values do. The asker must know where and when the answer appears, since it did the translating.

**What this does not show.** The description the runner is handed is the translated flags and layout; the
translation, which lays out the net's backward pass as adds and multiplies and schedules it, is done outside
the runner. Only closed nets: no outer ports. That the translation is right for every closed net
that fits is proved by hand, not yet reviewed, in the book repo's `notes/self-hosted-arr-2026-10-01.md`,
"The runner's theorem" (duration one, exact arithmetic; every closed term over the kit is a net, its Lemma 0);
the random nets are evidence for it. No runner fits its own translation: each is run by a larger one. The runner has no cost and no two-way coupling: it computes a net's covectors as numbers and
undergoes none of them. The existence of a runner is not surprising (STATUS §2, "what the construction was
built to have"); what could have failed and did not is the exactness of each level against an independent
stepper, and the runner's own net fitting inside the class it runs.

### 3.38 Bayes' rule for every belief over finitely many values, with no gauge

`arrangements/belief.py`, `tests/test_bayes_every_belief.py` (39 passed, 4 strict xfail, 45 s); book plan P36
stage (a); derivation and ten predictions fixed before code, verdicts and numbers in
`notes/BAYES-EVERY-BELIEF.md`; the proof in the book's letters in the book repo's `notes/bayes-every-belief.md`.
The belief box's state is all `n` log-heights `l`, its readout the belief `softmax(l)`, its forcing zero; a
downstream likelihood box has the cost `<y, -log Pr(x | .)>`, linear in the belief; the response divides by the
belief's height, `diag(1 / softmax(l))`. One step is `l + xi - <q, xi> 1` with `xi = log Pr(x | .)`, whose
belief is the posterior.

**Strong evidence here.** A theorem for every belief with positive heights and every positive likelihood
(four lines, in the note), checked against the posterior computed directly: worst relative error `2.6e-15`
over 96 random (belief, likelihood) pairs at `n = 2, 3, 7, 50`; a 30-reading stream on a two-bump prior to
`3.1e-15` in any order and as one batch; the book's bell-curve step exactly on a grid. The Euclidean response
is Bayes only at the uniform belief with `eta = n`.

**The gauge is a convenience, and a costly one.** The step's constant is the belief's average of the force;
nothing has to normalize the stored numbers. The gauge-fixed step (`choice.belief_arrangement`, and its
closed-form inverse `belief.gauge_response`) loses about `1e-16` over the reference value's height; the box on
all `n` log-heights is exact down to `2.2e-308`, below which jax holds a height as `0` and the state is `nan`.
On the science-scales stream (§5.1 of that note) it tracks the direct posterior to `1.7e-13` over 200 draws
where the gauge referenced at a losing hypothesis is off by more than `1` at draw 30.

**What this does not show.** Anything about a belief that is a whole curve, which is §3.39 (P36 stage (b)): `n`
values is a family with `n - 1` natural parameters. Anything about a belief or a likelihood with a zero. One refuted
prediction: that a thin height by itself loses the dense inverse.

### 3.39 Bayes' rule on grids; a whole-curve identity and a narrower smooth construction

`arrangements/belief.py` (`belief_coalgebra`, `updating_on_beliefs`), `tests/test_bayes_whole_curve.py` (25
passed, 32 s); book plan P36 stage (b); derivation and eight predictions fixed before code, a ninth before its
piece, verdicts and numbers
in `notes/BAYES-WHOLE-CURVE.md`; the proof for a curve over the real line, and the smooth set it lives on, in
Part II of the book repo's `notes/bayes-every-belief.md`. The library holds grids only (v1 is `R^d`).

**Strong evidence here.** On grids of 5 to 1025 values the stage (a) box lands on the closed-form posterior of
a two-bump belief restricted to the grid, to `4.6e-14`, at every size: restriction to a grid commutes with the
step. The step's constant, the mean, the variance and the area converge to the curve's faster than any power
(all four below `5e-9` at spacing `0.5`, at round-off from `0.25`). The area left by one step from a normalized
log-height is `exp(KL(q || q'))`, to twelve places, and `N` steps of duration `1/N` bring its log down like
`0.772 / N`.

**A machine holds a curve's step only as the composite.** Every entry of `diag(1 / q)` grows like one over the
spacing and the responsibility shrinks like the spacing; on a window of half-width 40 a quarter of the heights
underflow and the box that divides returns `nan` there. `belief_coalgebra` has the belief as its configuration
and the log-heights as its state, and its update is the response after the responsibility, `l + xi - <q, xi>`;
it is exact on that window, through a 30-reading stream with log-heights down to `-27771`. It also holds what
no bell-curve family does: told only "heavier", `N(0, 1)` becomes `2 phi Phi`, mean `1 / sqrt(pi)`, to ten
places.

**Where it stops.** A Cauchy belief with bell-curve noise: the belief after the step converges as the window
widens and the step's constant diverges, slope `-gamma / pi` to four places. A force `-log q`: the constant
converges to the entropy and the belief flattens with no limit.

**Independent review, 2026-10-08 (Codex, P36 (d)).** The weighted Banach-domain smoothness proof holds;
its domain is narrower than the positive continuous finite-variance beliefs covered by the posterior identity.
A smooth density of variance one makes the proposed Gaussian-update derivative fail in the proof's weighted
measure space. The posterior identity on a finite grid is not a raw coalgebra restriction or a general
quadrature-convergence theorem. See the book's `notes/bayes-proof-review-20261008.md`, §§1–3.

**What this does not show.** A whole-curve integrator in Ch 3's total sense, or a cotangent bundle stable
under arbitrary smooth pullback. The review's §4 proves composition for local updates with their admissible
domains; it does not repair the missing finite-variance smooth realization. No response with a floor or
likelihood in more than one dimension was run here.

### 3.40 A listener who hears accounts and decides whom to count on

`arrangements/listener.py` and `tests/test_listener.py` (7 tests, 3.4 s); predictions fixed first in the book
repo's `notes/runs/p37-listener-f2836b6c/PREDICTIONS.md`, results in its `RESULTS.md` (book plan P37 (a);
Claude Fable 5.1, f2836b6c, 2026-10-07). Built from two things this file already has and nothing else: §3.38's
belief integrator (`belief_coalgebra`) and the spatial attention parent, in the curious reader's shape
`compose_seq(tensor(listener, source), spatial(2).arr())`. The source is frozen at placement 0 and sends each
round's outcome one-hot (kept; failed and the account of why lined up with a check; failed and it did not),
weight 1. The listener is a belief box after a valuation box: the belief box's cost is `<y, N h>` on the
received message, so its force is Bayes' rule tempered by the kernel weight; the valuation's cost is
`w <y, ell> + (1 - w) c`, the expected loss of counting on the source times how much it does, plus doing without
for the rest. Composition gives the placement `xi_p = -kappa'(p) (<y, ell> - c)`.

**Held.** The composite is the closed form to 4e-16. With losses (0.4, 1.4, 8) and doing without at 5, the
listener ends on the reliable source and on the unreliable-but-accountable one, and leaves the unaccountable one
to kappa = 0.005, the two unreliable sources failing on the same rounds. **Narrowed** (David, 2026-10-08): that the
honest late source is kept rests on the stipulation `ell_2 < ell_3`, "one can plan around it," which he does not
hold (padding "is just a ridiculous spiral"; lateness costs with the stakes), so what holds is that the belief tells
the two apart by whether their accounts lined up and the placement leaves whichever the losses make dearer than
doing without; "avoids the unaccountable, not the unreliable" is not established. **A listener whose choices push on its belief comes to believe whatever is cheapest to believe, and the
more it already counts on the source the harder the push.** With the belief box sending its belief
forward plainly (`wire="exact"`), the valuation's covector `-w ell` reaches the belief, which goes to "reliable"
above 0.9999 within ten steps, before any failure, and stays there through four failures whose accounts did not
line up; on the one-way wire (`stop_gradient`, §3.18's idiom) the belief is Bayes and nothing else, to 4e-16.
David's internal checker, a part that notices "I would actually prefer the thing that I'm starting to believe" and
cancels the push (2026-10-07), is §3.18's subtracting wire: `wire="subtract"` has to know the losses and the
reliance, and then agrees with the one-way wire to 8e-17 on every script (test 8, predicted after his answer and
before code; 8 tests, 3.9 s). Leaving ends the hearing: the last two outcomes arrive at kappa 0.013 and 0.007, and the final belief is 0.966
where full evidence gives 0.99998. The exact likelihood forcing pushes the listener away from a rare lined-up
failure by its surprisal, 0.1329 in placement on the step it arrives, the chain rule through the transfer; the
unilateral forcing (`science.account_community`'s idiom) does not. A repeated account counts twice: evidence
counts once per step it is on the wire.

**One clause failed as written**: prediction 3's kernel weights were read off the closed form at the wrong steps
(test 3's docstring); the claim they were attached to held.

**What this does not show.** Anything about a real account: whether it lined up is a supplied bit with supplied
likelihoods, and the losses are stipulated, `ell_2 < ell_3` being the assumption that an account which lines up
can be planned around. No promise, no composed explanation, no accounting suboperad.

### 3.41 A reader that retunes: the site placement as a slow state (superseded; file deleted 2026-10-08)

`tests/test_retuning.py`, deleted 2026-10-08 at David's "whatever you think is better" once §3.43 conformed to a source; its 11 tests and 2 strict xfails are recorded in the book repo's `notes/runs/p38-retuning-27dc74fb/RESULTS.md` and `test_retuning.log`, and the file itself in that repo's history is the commit of 2026-10-08. Then (39 s); predictions fixed first in the
book repo's `notes/runs/p38-retuning-27dc74fb/PREDICTIONS.md`, results in its `RESULTS.md` (book plan P38
stage 1; Claude Fable 5.1, 27dc74fb, 2026-10-07). §3.15's composite (stage B, T3: the match a kernel between the
apparatus's site and the substrate's placement, the substrate drifting at `v`, the site held, collapse at distance
0.812) with the site a slow state of mobility `mu` that descends the rebuild's cost `-k_r kappa F q_1`, the one
term of T1b's potential that goes through the match; its motion charged to the fuel at `-chi mu xi_s^2`; the fuel
held, or fed by T4c's law at `J = 24.246` so the reader rests at T3's state.

**Held.** With the site held the composite is T3's (collapse at 0.812). Mobile at `mu = 0.01`, the reader tracks a
drift of 0.005 indefinitely at a steady lag 0.0945, the closed form; the critical speed is `v_c = 1.767 mu`, the
maximum pull at lag 0.509 inside the fold 0.722 (measured 0.01793 against 0.01767); `v_c(tau) = v_c(1)/sqrt(tau)`,
so width works against retuning. At the same drift, fixed readers of widths 1, 2, 4, 9 collapse 4 of 4 at
`0.72 sqrt(tau)` plus the slow-passage lag; mobile ones track 4 of 4. Fed and charged, the tracking reader runs on
less fuel and is smaller exactly as the stationary balance says (`chi = 4000`: `F = 0.865`, `q_1 = 1.206`), and
the threshold falls with the charge, below the ceiling `sqrt(mu J / chi)` where tracking would cost the whole feed.
The exact control (the rebuild's covector on the substrate too) raises `v_c` by `(mu + mu_S)/mu` and moves the
substrate at `2v/3`: a source with response is moved by its reader.

**Refuted as fixed**: the fed thresholds read off collapse within 600 time units. Above the stationary threshold a
fed reader declines for hundreds of time units while its fuel pools (a reader losing its match pulls less and is
charged less), then collapses; the finite-run threshold overshoots `v_c` by 11-14%. A settled-lag criterion was
introduced after the fact and undershoots; the two bracket `v_c` to 2%, and the code says which is which.

**Pooling.** T4c's supply law (washout 1 against consumption 23 at rest) lets unspent fuel accumulate to `J`: a
fixed fed reader survives drift to 2.6, three times the held one, and the uncharged mobile fed reader tracks to
`v = 0.082`, 4.6 times the held value. This is the supply law's, not the retuning's; read every fed number against it.

**What this does not show.** Anything about allostatic load (the book plan's analogy stays an analogy); whether a
site is matter (its mobility and the exchange rate are choices; that the retune child is the rebuild child's own
forcing is not, §2's expressibility warning met by piece 0's inertness with the site held); anything about the fed
law's pooling ratio, which is T4c's.

### 3.42 A composed belief paid in arrears: responsibility is not the mistake

`arrangements/figurines.py`, `tests/test_figurines.py` (8 tests, 185 s, nearly all JAX compiling the composite) and
`tests/figurines_closed_form.py` (the forces by hand, numpy, run before the module existed); predictions fixed first
in the book repo's `notes/runs/p37-composed-belief-3f16cdb9/PREDICTIONS.md`, results in its `RESULTS.md` (book plan
P37 (b)(i); Claude Fable 5.1, 3f16cdb9, 2026-10-08). Three FIGURINE boxes `I -> B` (a driver, a vehicle, a route),
each one real state whose sending map runs a small mechanism (`jax.lax.fori_loop`) and sends its guess in minutes in
its own slot of `H = R^6`, composed on the spatial attention parent (kernel weight the invitation, value maps
frozen), then a copy box (the diagonal, with `stop_gradient` on the copies the arm makes one-way) and a bettor with
a margin state: the plan's inconvenience every step, the realized loss `(t_hat + m - t*)^2 / 2` on each round's last
step, an observation cost where the arm says; the world is the exterior input, which on an outcome step may carry an
ACCOUNT `((a f, f), 1)` of what happened by component, delivered to each figurine by the parent tempered by its
kernel weight and read by the figurine's own cost `w g^2/2 - g h`, unilaterally. Held, all eight: the composite
equals the closed form to 4e-14 on every arm and scenario; a figurine at `p = 4` gets `e^-10` of the use and of
the blame. The finding: the chain rule gives each figurine the push times its contribution and not what it was wrong
about (the driver late by 4, loss alone: the bet right by round 30, the route figurine, right, absorbing 88% of the
correction; reused with a fresh driver the figurines mispredict by 3.55 minutes; with the account each comes to its
own truth and the reuse error is 0.02). The push on a placement is the push times the figurine's guess, so the
biggest contributor is uninvited hardest (5.7x). A belief that is the bet is bent by its plan on the exact
wire: 0.95 late at a lower total cost under the squared loss, which charges a minute late as a minute early; 0.5 early
at a higher total cost under a bus loss (`cliff`, a minute late is the bus; the check of 2026-10-08, 70fcc5f2, the
arms `*-cliff`, `test_figurines.py` test 9); a margin of its own with the belief read one-way and paid by the observation keeps the belief true at
every step at the lowest total cost under either loss; on the exact wire the margin takes the bias only on the round's
average. Model, run; the loss, the plan cost, the account's contents and the parametrizations supplied;
the composition a sum, and the kernel discounts additive guesses (`kappa < 1` counts `kappa` of the guess).


### 3.43 A population tracking a moving optimum: Bürger and Lynch (1995) as an arrangement

`tests/test_moving_optimum.py` (7 tests, 13 s); predictions and the source's equations fixed first in the book
repo's `notes/runs/p38-retuning-27dc74fb/PREDICTIONS-moving-optimum.md`, results in its
`RESULTS-moving-optimum.md` (book plan P38; Claude Fable 5.1, 27dc74fb, 2026-10-08, at David's "find a case in
the world that yours matches and start using those words and those values for the test, or abandon it"). The
source's deterministic part (Evolution 49(1):151-163, pp. 152-154, eqs. 1-4, 8a, 9, 10) drawn as boxes: the
population, the mean phenotype (mobility the additive genetic variance, Lande's equation) and the optimum; viability
and birth, the carrying capacity, selection on the mean (the gradient of minus log mean fitness, nothing on the
optimum) and the environmental trend. Values from the paper's figure captions; the genetic variance 0.0703 implied by
its Fig. 1B through eq. (10), which then predicts its Fig. 3 numbers to three digits.

**Held, all seven**: one generation is the paper's recursion to 1e-12; the lag settles at `k/s` and its 95% time at
`3/s`; the critical rate is eq. (10) to 0.05%; extinction at twice it at the hand-computed generation; fixed against
evolving at the same rate, 3 of 3 extinct at the predicted generations against 3 of 3 persisting; the width of the
fitness function at fixed variance, eight thresholds within 0.2% of eq. (10), maximal at an intermediate width; the
standing load to four digits; Fig. 3's birth rates within 2% of the paper's captions. One measurement horizon
lengthened after the data (the settling time scales with `1/s`).

**What it adds.** Only that the paper's model is an arrangement, its mean phenotype a placement descending its own
cost at a mobility the genetic variance sets. §3.41's first attempt conforms to nothing in the world; its file is deleted and its results stay in the
book repo's notes.

### 3.44 The coordinator that distributes the mistake: a belief in the external box

`arrangements/coordinator.py`, `tests/test_coordinator.py` (8 tests, 119 s) and `tests/coordinator_closed_form.py` (the
forces by hand, numpy, run before the module existed); predictions fixed first in the book repo's
`notes/runs/p37-coordinator-70fcc5f2/PREDICTIONS.md`, results in its `RESULTS.md` (book plan P37 (b)(ii); Claude Fable
5.1, 70fcc5f2, 2026-10-08). §3.42's figurines (one or two buses sharing the route; a second bus's driver and vehicle at
slots 3 and 4) on the spatial parent with placements frozen, and above it the DEPOT, `<E | M> -> <E x R^{1+B+6B} | R^0>`:
state one belief per bus over which of its three figurines was wrong (`belief_coalgebra`, tensored), its backward map
computing the REPORT, each figurine's share of its bus's mistake by that belief, sent down as the parent's exterior
message (`blame="belief"`) or the whole mistake to everyone (`blame="chain"`), which the parent delivers and each
figurine's run pulls back as §3.42's chain rule exactly (reproduced to four places); the depot's cost the belief's
evidence, unilateral, so each is Bayes: a sighting one-hot from the world, the prior as the initial state, and the other
bus's mistake under the one-wrong-figurine model. A figurine reads its report as a push on its guess (`k = g h`,
unilateral) with one of two responses: `told`, `mu_g / g'(theta)^2` (the guess moves by its share), or `slope`, §3.42's
Euclidean on the parameter. Held: composite = closed form to 4e-14 on 18 arm-scenario pairs over 60 rounds. The finding:
told, the shares are the belief; by the slope they are the belief times `g'^2`, so a prior of (0.7, 0.1, 0.2) on
(driver, vehicle, route) comes out (0.40, 0.00, 0.60). A sighting every round gives the driver 0.917 of the four
minutes, not all (round 1's report precedes round 1's sighting; a look-then-report coordinator would be a `CotCoalg2`);
a second bus on the route clears the route by round 3 and cannot split driver from vehicle; a wrong prior is unmoved by
the bus's own data, and comparison clears the route of blame without restoring what it absorbed (0.065 kept, bus 2's
right figurines drifting to cover it); reuse error = 4 (1 - the driver's share). Model, run; likelihoods, priors, sigma
and the one-wrong model supplied; no magnitudes, no record of reports. Two predictions failed on their thresholds with
the mechanism intact (the sighting's 0.95; the kept share's 0.10), recorded in PREDICTIONS.

### 3.45 The EM coordinator: the chain rule of Q broadcasts, the E-step's lift distributes

`arrangements/em_coordinator.py`, `tests/test_em_coordinator.py` (9 tests, 234 s) and `tests/em_coordinator_closed_form.py`
(the forces by hand, numpy, run before the module existed); predictions fixed first in the book repo's
`notes/runs/p37-em-coordinator-1a279037/PREDICTIONS.md`, results in its `RESULTS.md` (book plan P37 (b)(iii); Claude
Fable 5.1, 1a279037, 2026-10-08). §3.44's figurines and parent, and above them the COORDINATOR, whose model sends the
figurines into the product of their stories: one latent error per figurine, `eps_j ~ N(0, V s_j)`, the SPLIT `s` a
belief over the figurines (`belief_coalgebra`, rate 0.05) and the DOUBT `V = e^Lambda` a configuration integrator; a
bus's total `t* = sum (G_j + eps_j) + N(0, sigma0^2)`, a sighting `o = G_0 + eps_0 + N(0, sigma_s^2)`; `G` the guesses as
first heard (`memory="record"`, a state with zero response) or the current guesses (`memory="none"`, §3.44's depot).
The coordinator's own state descends the marginal likelihood of everything seen (exact); its backward map sends each
figurine the GAP `g_j - G_j - eps_hat_j` to the E-step's lift `eps_hat = diag(v) A^T Sigma^{-1} y`. **The correction
the build rests on** (test 0): by Fisher's identity the gradient of EM's expected complete-data cost `Q` at the
current point is the marginal's, one number on every figurine of a bus (the shared route summed over its buses), so
the chain rule of `Q` broadcasts and the P37 block's "the chain rule of the expected complete-data cost gives each
figurine its share" was false; the M-step's minimizer, the posterior mean, distributes, and it is `Q`'s gradient in the
model's latent metric `diag(v)`, a response the coordinator owns and a figurine does not. Held: composite = closed form
on 32 arm-scenario pairs (states, split, doubt, report to 1e-12; the covector on the split to 2e-10 through a
log-determinant); memoryless at the same split the report is §3.44's share times `V / (V + sigma0^2)`, the rest the
accident's; one total is exactly flat across the split (spread 4e-12), so a wrong prior is unmoved by construction;
a sighting that is a measurement puts the split on the driver (0.904 at round 10), a second bus by a joint posterior
clears the route (under 0.05 at round 10) and cannot split driver from vehicle; with the record a cleared route is
restored to 0.000 and the second bus's right figurines stay within 0.001, on told and slope arms alike, and the slope
figurines reach the told figurines' shares (the matter sets the rate, not the destination; §3.44's override is the
memoryless coordinator's); the doubt moves toward the squared shortfall but reaches 12.80 of 16 in 60 rounds at rate
0.05 (prediction 7 failed on its threshold). Model, run; the latent structure, noises, prior split and rates supplied;
the categorical form checked by hand only (its M-step for the mixing weight is §3.44's Bayes step exactly).

### 3.46 Stock-and-flow diagrams: every one is an arrangement of its stocks, auxiliaries and flows

`arrangements/stock_flow.py`, `tests/test_stock_flow.py` (4 tests, 35 s); book Ch 4 `sec.stock_flow`, `prop.stock_flow`
(Claude Opus 5.5, d14d8d10, 2026-10-08, at David's "I want some kind of general formula for stock and flow, how to
convert them into this formalism, not just um, one-off"). `arrangement_of(diagram, dt)` builds, from any diagram with
smooth rates and acyclic auxiliaries, the closed arrangement whose internal boxes are one per stock (configuration
coalgebra, sharp `dt`, readout the level, zero forcing), one per auxiliary (a gray box) and one per flow (stateless;
forcing `-rate` on the stock it drains, `+rate` on the one it fills, `0` on its links); copies are wiring. One step is
the diagram's forward-Euler step: the epidemic (susceptible, infected, recovered, the infected fraction an auxiliary)
over 400 steps and 40 random diagrams (up to 5 stocks, 4 auxiliaries, 6 flows, shared auxiliaries, clouds) agree with
`euler` to 4e-15; the epidemic's total is conserved and its forcing is not exact. What it does not establish: the
continuous-time statement rests on the book's `prop.duration_tangent` and `prop.internalization_tangent`, which are
proof sketches; non-smooth rates (lookup tables, MIN, MAX, IF) are outside it.

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
