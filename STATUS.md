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
residual is its cost minimized over its own configuration, `r(e) = min_q k(q, e)`; it has a
need iff `r` varies with what it receives, and it is a member at a state iff
`r(received) < r(0)`. What the three runs support: the residual separates a need from a
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
the book's. The reruns the member test calls for — the need restated on the weighted input,
and a non-quadratic outer cost — are listed in the book plan and have not been run.

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
