# Bayes as an integrator (`tests/test_bayes_integrator.py`)

P25 stage (a) of `AGENTDOCPLAN.md` §8.2, from David's question in the §1.6 run (2026-09-27):
is Bayesian update "the same sort of thing, just a different integrator"? The claim under
test is Chapter 2's — learning and mechanics are arrangements of one kind — which so far
covers differentiable learning only.

Everything in §§1–3 below was written **before any construction code existed**; the
predictions are quoted verbatim in the test module's docstring. Verdicts are §5.

## 1. The derivation, fixed before the run

**Setup.** Let the likelihood be an exponential family in the unknown parameter
$\theta \in \Theta \subseteq \mathbb R^p$,

    p(x | theta) = h(x) exp( <T(x), theta> - A(theta) ),

with sufficient statistic `T` and log-partition `A`. Let the belief over `theta` range over
the conjugate family, whose sufficient statistic is `Psi(theta) = (theta, -A(theta))`:

    q_lam(theta) = exp( <lam, Psi(theta)> - AA(lam) ),   lam in Lam subset R^{p+1},

with `AA` the conjugate family's log-partition, finite on the open convex `Lam`. There `AA`
is smooth and strictly convex, so the Legendre map `grad AA : Lam -> M` onto expectation
parameters `mu = E_{q_lam}[Psi]` is a diffeomorphism with inverse `grad AA*`.

**Bayes is a translation.** `q_lam(theta) p(x | theta)` is proportional to
`exp <lam + t(x), Psi(theta)>` for

    t(x) := (T(x), 1),

so the posterior is `q_{lam + t(x)}`. One observation translates the belief's natural
parameter by the datum's sufficient statistic padded with its count.

**The cost is the expected negative log-likelihood, and it is affine in `mu`.**

    k_x(lam) := E_{q_lam}[ -log p(x | theta) ] = -<t(x), mu(lam)> - log h(x),

so the forcing, in the book's sign convention (`omega = dU = -dk`, `q' = q + sharp omega`),
is the *exact* one-form on `Lam`

    omega_x = -d k_x = d<t(x), mu>,   components  G(lam) t(x),

where `G := grad^2 AA` is the Fisher information of the belief family in natural
coordinates.

**The integrator.** Configuration `Q = Lam`, state `S = Lam`, readout the identity, and on
receiving a covector `xi` and a duration `tau`

    u_lam(xi, tau) = lam + tau * sharp_lam(xi),   sharp_lam = G(lam)^{-1} = grad^2 AA*(mu(lam)).

**The claim.** `u_lam(omega_x|_lam, 1) = lam + G^{-1} G t(x) = lam + t(x)`: one step of this
arrangement at duration 1 is Bayes' rule **exactly**, not to first order. *Scope (2026-10-01):* exact in the mathematics; the library computes the step as a dense
`inv(G)` applied to `G t`, which loses the exact translation once some hypothesis's weight falls
below about 1e-5 (`cond(G) ~ 1/min q`), shown on the categorical family over 200 draws in
`notes/SCIENCE-SCALES-PREDICTIONS-2026-10-01.md` §5.1. In Chapter 3's
`def:arr` the two objects named are therefore

* the **response** `sharp = grad^2 AA*`, the raised inverse Fisher metric, living inside the
  coalgebra's step (`def.duration_integrator`) — the slot the two-layer learner fills with
  `eta_LR * I` and the spring fills with an inverse mass;
* the **forcing one-form** `omega_x = -d k_x` on `Q = Lam`, `k_x` the expected negative
  log-likelihood of the datum — the slot the learner fills with `-d(loss)`.

The interface, the forcing lens, the wiring and the dynamics functor are untouched. In the
library this is `coalgebra.configuration` on `rvect.ReactiveVectorSpace(dim, sharp_fn)` with
`sharp_fn = lam |-> inv(hessian(AA)(lam))`, plus `arrangement.exact(-k_x)`: the same two
constructors as `learning.py`, with one argument changed.

**Two corrections to the prediction as P25 (a) first stated it** (Fable, 2026-09-27,
*proposed* there; found while fixing this derivation). (i) "`theta |-> theta + T(x)` in
natural coordinates" names the wrong `theta` and the wrong increment: what translates is the
*belief family's* natural parameter `lam`, not the likelihood's `theta`, and the increment is
`t(x) = (T(x), 1)`, the sufficient statistic padded with the observation count. (ii) "a
gradient step whose response is `grad A*`" cannot be right as stated, because a response is a
fibrewise-linear bundle map `T^*Q -> TQ`, and `grad AA*` is a map `M -> Lam` between base
spaces. The response is its derivative, `grad^2 AA* = G^{-1}`.

**Mirror descent, in the proximal form.** The same step is
`q_{t+1} = argmin_q { E_q[-log p(x|theta)] + KL(q || q_t) }` over all distributions, whose
solution is Bayes (the Gibbs variational principle). Because `-log p(x|theta)` is affine in
`Psi(theta)`, the solution stays in the conjugate family and coincides with the gradient step
above; the two readings agree here and need not in general.

## 2. The small example: conjugate Gaussian, unknown mean, known variance

Data `x_i ~ N(m, sig^2)` with `sig^2` known; prior `m ~ N(m0, sig0^2)`. Then
`Psi(m) = (m, -m^2/2)`, `t(x) = (x/sig^2, 1/sig^2)`, and

    q_lam(m) ∝ exp(lam_1 m - lam_2 m^2 / 2),   so  precision = lam_2,  mean = lam_1 / lam_2,
    AA(lam) = lam_1^2/(2 lam_2) - (1/2) log lam_2 + (1/2) log 2pi,
    G(lam) = [[ 1/lam_2,          -lam_1/lam_2^2                    ],
              [ -lam_1/lam_2^2,    lam_1^2/lam_2^3 + 1/(2 lam_2^2)  ]],   det G = 1/(2 lam_2^3) > 0.

Prior `lam^(0) = (m0/sig0^2, 1/sig0^2)`; after `n` data the closed form is

    1/sig_n^2 = 1/sig0^2 + n/sig^2,    m_n = sig_n^2 (m0/sig0^2 + n xbar / sig^2),

i.e. `lam^(n) = lam^(0) + sum_i t(x_i)`.

## 3. Predictions, each able to fail

1. **Exactness.** `n` steps of the arrangement at duration 1, one datum per step, reproduce
   the closed-form posterior mean and variance to machine precision (relative error below
   1e-12).
2. **Batch equals sequential.** One step at duration 1 with the summed forcing
   `-d sum_i k_{x_i}` gives the same `lam` as the `n` sequential steps. Gradient descent with
   any response that is not this one fails this, since only a translation is additive.
3. **Duration is tempering.** Duration `tau` gives the belief updated by `p(x|theta)^tau`
   normalized; so `tau = 0` is the identity, and two steps at `tau = 1/2` equal one step at
   `tau = 1`.
4. **The response is what is doing the work.** The Euclidean response `eta * I` in the same
   arrangement — same wiring, same forcing, same interface — does *not* give the posterior at
   any `eta`; nor does the Newton response `(grad^2 k_x)^{-1}`, which is a different matrix
   because `grad^2 k_x` involves the third derivative of `AA`.
5. **Composition survives.** Wiring the belief into a downstream box and running `Phi` on the
   composite reproduces the same posterior, so the integrator is compatible with the dynamics
   functor, not only with a hand-run step.

If 1–5 hold, Bayesian update on a conjugate exponential family is an arrangement of the same
kind as a learner and a spring, differing in its integrator and in nothing else.

## 4. What this cannot establish, whatever the run shows

The belief here stays in a fixed finite-dimensional family, so `Q = Lam` is an open convex
subset of `R^{p+1}` and sits in the admissible subcategory of `STATUS.md` §3.1, where `cot`
is available. A belief over a *general* distribution is a point of an infinite-dimensional
space, and putting it in `arr` needs row (c) of `AGENTDOCPLAN.md` §5.3 — a chosen
`cot : E -> Poly(E)` — on a domain that contains such a space. And for a non-conjugate
likelihood the mirror-descent step is an approximation to Bayes, not Bayes; the exactness in
prediction 1 comes from the log-likelihood being affine in the belief family's sufficient
statistic, which is what conjugacy means.

## 5. Verdicts

`scripts/run_tests.sh 240 tests/test_bayes_integrator.py`: **20 passed (4 s)**, 2026-09-27.

**All five predictions hold.** Prior `N(2.0, 1.5)`, `sig^2 = 0.7`, seven data
`1.3, -0.4, 2.2, 0.1, 1.9, -1.1, 0.6`:

| | arrangement | closed form | relative error |
|---|---|---|---|
| posterior mean | 0.74107142857142949 | 0.7410714285714286 | 1.20e-15 |
| posterior variance | 0.09374999999999982 | 0.09375 | 1.92e-15 |

and `lam_7 = (7.9047619, 10.66666667) = lam^(0) + sum_i t(x_i)` exactly. (2) batch equals
sequential to 4.2e-14, and the Euclidean response fails the same comparison, as predicted.
(3) one step at `tau` lands on `lam + tau t(x)` for `tau = 0, 1/4, 1/2, 1, 2`, and two
half-steps equal one whole step. (4) the Euclidean response at `eta = 0.01 … 2.0` descends the
cost (28.03 to between 13.7 and 18.6) and lands on the wrong belief every time; the Newton
matrix at `lam^(0)` is `[[3.214, -8.679], [-8.679, 26.679]]` against the Fisher
`[[1.5, -3.0], [-3.0, 7.125]]`, and its step `(1.330, 0.838)` is nowhere near Bayes'
`(3.190, 2.095)`. (5) `Phi(reporter . belief)` moves the belief identically, to 0.

**So the answer to David's question is yes**, for a conjugate exponential family: Bayes' rule
is the same arrangement with a different integrator, and the integrator differs only in its
response, which is the raised inverse Fisher metric `grad^2 AA*` instead of a learning rate or
an inverse mass. The wiring, the interface, the forcing lens, the shape of the forcing
(`-dk` for a cost) and the dynamics functor are the same objects doing the same work.

### Three things the run found that were not predicted

**(a) Bayes occupies the learner's slot, not only Newton's.** In the derivation above the
belief box carries the likelihood as its own forcing, which is where `test_newton.py` and the
spring sit. It also works in `learning.py`'s configuration, which is stronger: the belief box
reports its *expectation* parameters `mu = grad AA(lam)` forward and forces nothing; a
downstream likelihood box carries the cost `k(mu) = -<t(x), mu>` and pushes `t(x)` back; the
pullback along the Legendre map is `(grad^2 AA)^T t(x) = G(lam) t(x)`; and the inverse-Fisher
response turns that into `lam + t(x)`. So the loss is a separate box and *backpropagation*
supplies the parameter covector, exactly as for a network. Checked to 1e-12
(`test_bayes_also_has_the_learners_shape_loss_downstream`).

**(b) Where each piece of Khan and Rue's objective lands in `def:arr`.** Their rule (Eq. (3),
p. 3) descends `E_{q_lam}[lbar] - H(q_lam)` with `lbar = -log p(y, theta)`, prior included,
and the entropy term's natural gradient is `-lam` (their App. B), which turns the step into a
*replacement* of `lam` rather than an increment — their one-shot Eq. (54), p. 22. In `arr` the
prior is the integrator's initial state and the increment is the configuration coalgebra's
`s |-> s + sharp(xi)`, so the arrangement's forcing carries only the datum's loss and no
entropy term appears. Both give `lam^(0) + sum_i t(x_i)`; checked against Eq. (54) by autodiff
in `mu` (`test_matches_khan_and_rue_equation_54`).

**(c) A Bayes arrangement never settles.** Its forcing is `G(lam) t(x)` with `G` positive
definite and `t(x) = (T(x), 1)` never zero, so the forcing vanishes nowhere on `Lam`: the cost
has no critical point there. Its infimum is on the *boundary* — infinite precision, mean at the
datum — which is outside `Lam`, and the orbit on a repeated datum runs to it as `1/n`,
variance exactly `1/(1/sig0^2 + n/sig^2)`, never arriving. So unlike a spring in syrup or a
learner descending a loss with an interior minimum, this arrangement has no driven steady
state; what stops one step from collapsing the belief is the unit step size, which is the `KL`
term of the proximal form. **An earlier draft of this finding said the cost is "unbounded
below." That was wrong**: `sum_i k_{x_i}` is affine in `mu`, but `mu` ranges over the curved
realizable set, on which the infimum is finite (`-x^2/(2 sig^2)` plus a constant for one
datum) and simply not attained.

### Reasonable or weird, by scale (the agent's judgment, not evidence)

* **A cell.** Reasonable. A receptor whose internal variable is a log-odds and whose step is
  additive in the evidence is the standard reading of chemotactic adaptation, and (c) is the
  familiar complaint that a pure accumulator has no set point: real cells add a leak, which in
  this language is a second forcing on the same configuration, not a different integrator.
* **A person.** Reasonable for the update, weird for (c) at long horizons. A person who has
  seen the same thing a thousand times should not have precision a thousand times the prior's —
  the data are not independent — and the model has no place to say so. In `arr` that is not a
  limit of the integrator: it is the arrangement's, whose forcing takes each datum as a fresh
  one. Fixing it means changing the *wiring*, not the integrator.
* **A market.** Weird in an informative way. Prediction (2), batch equals sequential, says the
  order in which news arrives does not matter to this integrator. That is right for Bayes and
  wrong for markets, which is a real prediction the framework makes about the difference:
  path-dependence in a market cannot come from Bayesian updating alone, so it has to come from
  the wiring, from a non-exact forcing, or from the state carrying more than a belief.
* **Across scales.** The response is the inverse Fisher metric, so the step size is set by how
  sharply the belief already distinguishes what it might see. That is a scale-free statement and
  it is the one the book wants: the cost of moving is read off the matter that is holding the
  form, not supplied from outside.

### What this does not establish

§4 above stands, and the run narrows it once more: the belief family here is two-dimensional
and the test is one conjugate family. The exactness in prediction 1 is conjugacy — the
log-likelihood affine in the belief family's sufficient statistic — so nothing here says
anything about non-conjugate Bayes, which is where Khan and Rue's §§3–4 approximations live and
where the framework has been given no test. And the compositional half is untouched by
Braithwaite–Hedges–St Clere Smithe's lens, which is a lens on Markov kernels, not `arr`'s
`omegalens`; the relation between the two is not established here and is not obvious.

### Sources, with locators (all opened; `AGENTS.md`, "Citations")

* **Khan and Rue**, *The Bayesian Learning Rule*, JMLR **24**(281):1–46, 2023
  (`khan2023bayesian`; arXiv:2107.04562v4, open access, CC-BY; page numbers are the JMLR
  pagination):
  * **Eq. (2), p. 2** — the variational characterization of the posterior,
    `q* = argmin_q E_q[sum_i l] + KL[q || p]`, and just below it the entropy form
    `argmin_q E_q[lbar] - H(q)`. This is P25 (a)'s "Gibbs variational principle for the
    posterior"; no separate classical source for that name was opened, so the book cites this.
  * **Eq. (3), p. 3** — the Bayesian learning rule itself.
  * **Eq. (4), p. 3** — the natural gradient `F(lam)^{-1} grad_lam = grad_mu`, with
    `F(lam) = grad^2_lam A(lam)`. This equation *is* the response of the Bayes integrator.
  * **§2.2, Eqs. (20)–(23), p. 9** — the rule as mirror descent with the Bregman/KL proximal
    map, and `D_KL[q_1 || q_2] = D_{A*}(mu_1 || mu_2)`.
  * **§5.1 "Conjugate Models and Bayes' Rule", Eq. (54), p. 22**, and p. 23: "In general, for
    such conjugate exponential-family models, the BLR yields the exact posterior."
  * **§1.3.2, p. 5** — Newton's method as an instance, which is why `rvect.inverse_hessian`
    and this response are cousins and yet, as the run shows, different matrices.
* **Braithwaite, Hedges and St Clere Smithe**, *The Compositional Structure of Bayesian
  Inference*, MFCS 2023, LIPIcs **272**, 24:1–24:15 (`braithwaite2023compositional`; open
  access, Schloss Dagstuhl): **Definition 9 (Bayesian Lenses), p. 24:6**;
  **Proposition 11 (Bayesian inversion is almost functorial), p. 24:6**;
  **Theorem 20 (Unique Bayesian Inversion), p. 24:10**. The paper has no exponential families
  and no conjugate priors; it bears on the compositional half only.
