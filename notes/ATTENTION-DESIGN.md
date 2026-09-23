# arrangements stage (d): the attention operad

> **Working design note, kept as the record of why the code is shaped this way.**
> It was written before the code, to be executed against; it is not a specification of
> the current API, and its process details (token budgets, staging, "stop and report")
> are how the work was actually carried out. For what the library establishes and what
> it does not, see [`STATUS.md`](../STATUS.md).

Design note (Claude, Opus 5, for David Spivak, 2026-09-22). Companion to `DESIGN.md`,
whose §§1–3 and §7 still bind: same datatypes, same sign convention, same rules. This
note covers only what is new. The mathematics is
`../../articulating-forms-that-matter/background/affine_quadratic_operad.tex` (cited below
by its own labels), as constrained by the book: `AGENTDOCPLAN.md` §5.4 (the summary and
what may not be overstated), §5.2 (displacement semantics), §5.9 items 1–3 (what is open).

**Read:** the attention paper in full (660 lines), `DESIGN.md`, `dap/arrangement.py`,
`dap/coalgebra.py`, `dap/wiring.py`, `tests/test_two_tanks.py` for the idiom. Not
`CDLM.tex`, not this repo's other examples.

**There is no old attention code**, so there are no regression `.npz` files for this
stage. Acceptance is against the paper's closed forms and stated numbers (§5 below),
which is a stronger check where it applies and no check at all where it does not — say
which is which in every test docstring.

## 1. The interface match

The port is nearly mechanical because the paper's data are already an arrangement's data.
Signal space $H$, weight space $W=\mathbb R$, position space $P$, message space
$M=H\times W$, input space $E=M\times P$, interface $B=(E,M)$, interaction space
$D_n=M^n\times E$. An $n$-ary attention arrangement is a morphism $B^{\otimes n}\to B$:

| paper | `Arrangement` field | note |
|---|---|---|
| $B$ as the domain's output side, $M$ | `out_dim_M = n * (dim_H + 1)` | $n$ children, each sending a message |
| $B$ as the domain's input side, $E^n$ | `in_dim_M = n * (dim_H + 1 + dim_P)` | |
| $B$ as codomain: output $M$, input $E$ | `out_dim_N = dim_H + 1`, `in_dim_N = dim_H + 1 + dim_P` | |
| $D_n=M^n\times E$ | exactly `(m_out, n_in)` | the crux: the paper's interaction space *is* dap's pair |
| sending $\sigma_a(q,\mathbf m)=\sum_j T_{0j}(q)m_j$ (`eq:maps`) | `out_f(q, m_out)` | |
| receiving $\rho_a(q,\mathbf m,e)_i=(T_{i0}m_0+\sum_j T_{ij}m_j,\;p_0+\ell_i(q))$ | `in_f(q, m_out, n_in)` | |
| forcing $\omega_a=-dk_a$ on $Q\times D_n$, **the full differential** (`eq:forcing`) | `omega = exact(lambda q, m, n: -k(q, m, n))` | the paper already uses the book's convention $U=-k$, $\omega=dU$; **no sign translation**, unlike every CDLM port |
| integrator $I=(Q,C,\iota)$ | `CotCoalg`, i.e. $C=\Store(S)$ only | see §3 |

So `attention_arrangement(n, placement, transfers, cost, parameter)` builds an
`Arrangement` directly, with $Q=\mathbb R^{\dim Q}$ as always (v1 stays on Cartesian
spaces). Transfers are a function $Q\to\mathcal T_H^{(n+1)\times(n+1)}$ returning, for
each ordered pair, the pair $(\gamma,L)$ of `eq:transfer`, with zero diagonal; apply them
with the algebra of `eq:transfer-algebra`, never by building a dense $\End(M)$ matrix.

## 2. Substitution is composition in $\arr$ — the one real theorem to test

`prop:operad` says substitution makes $\Attention$ a symmetric operad and that
$(I_a,\sigma_a,\rho_a,\omega_a)$ is an operad morphism into $\End_\arr(B)$. The paper
gives substitution twice over: abstractly (composition of lenses in $\arr$) and
explicitly (`eq:composite-state`, `eq:composite-transfers`, `eq:composite-cost`). **The
test that earns this stage is that the two agree.** Implement both:

- `substitute(a, [a_1, ..., a_n])` — the paper's explicit formulas, computing the
  composite's placement, transfers, and cost with no reference to `wiring.py`.
- the composite in $\arr$, `compose_seq(tensor_arrangements([a_1, ..., a_n]), a)`.

Then test, on random configurations and interactions: equal $\sigma$, equal $\rho$, equal
forcing (all components, $q$ and $m$ and $n$ alike), and equal `Phi` trajectories over
many steps. Do this for a two-level nest with $n=2$, $n_i=2$, and for an uneven nest
($n=3$, arities $1,2,0$), which exercises the empty sum. Also test what the proof turns
on and a reader would doubt: associativity across two bracketings of a three-level nest;
the symmetric action (`prop:operad`'s $\tau$ formulas) commuting with substitution; the
unit (arity one, zero placement, zero cost, $T_{01}=T_{10}=M$) acting as an identity; and
the zero diagonal doing its job — a group's outgoing message does not return to that
group.

Second structural test, cheap and worth it: the **pure placement suboperad**
$\Placement\subseteq\Attention$ ($T_{ij}=M$ for $i\ne j$, $k=0$) is closed under
substitution, and the forget-then-resupply retraction of the paper's display is the
identity on it. Include the paper's observation as an assertion: a placement arrangement
with zero cost still receives position covectors from the components placed inside it.

## 3. What the book keeps and what it drops

The paper's integrator is $I=(Q,C,\iota)$ with $C$ an arbitrary polynomial comonad in
$\CatSharp$. **The book's $\arr$ uses the concrete $\Store(S)$ comonad only**
(`AGENTDOCPLAN.md` §5.2, §5.9 item 2). So `CotCoalg` is the whole of the integrator here,
and the code must say, once, in the module docstring: the general $C$ can encode more
state than the configuration, this port does not implement it, and the examples below do
not need it. Do not silently present $\Store(S)$ as the paper's $I$.

Two constraints $\mathbb R^d$ does not enforce, to state in docstrings and guard in
constructors rather than pretend away:

- **Admissible messages** are $\{0_M\}\cup(H\times\mathbb R_{>0})$. A transfer preserves
  admissibility ($\gamma>0$), but nothing stops a test from handing in a negative weight.
  Provide `is_admissible(m)` and assert it where the paper assumes it.
- **Mean readout** $\mean(h,w)=h/w$ is defined only for $w>0$ (`eq:mean`). Keep it a
  separate function, never folded into a cost: the paper notes that applying it inside a
  cost leaves the quadratic family.

## 4. Discrete time: the propositions must be restated, and the code does not prove them

The paper's dynamical examples are ODEs; the book's time is discrete by design, one
covector giving one displacement (`AGENTDOCPLAN.md` §5.2). Restating each proposition for
the displacement map with a mobility bound is an open book task (§5.9 item 1), **not this
stage's job**. This stage supplies the numerical check, and every test docstring says
which of the three it is doing: reproducing a closed form the paper proves, checking the
discrete analogue the book will have to prove, or merely illustrating.

The facts the tests may rely on, from §5.2:

- For a quadratic cost with Hessian $\mathsf H$ and constant mobilities
  $D=\operatorname{diag}(\mu_i)$, the step is $y\mapsto y-D\nabla K$, and it converges iff
  every **nonzero** eigenvalue of $D^{1/2}\mathsf H D^{1/2}$ lies in $(0,2)$. (Corrected
  2026-09-22, d2: "every eigenvalue" was wrong. $\mathsf H$ is positive *semi*definite with
  kernel $z_1=z_2=x_1=x_2$, the conserved $S$ of `prop:meeting`, so no mobility puts every
  eigenvalue in $(0,2)$; the zero one is the direction whose limit the initial condition
  fixes. `AGENTDOCPLAN.md` §5.2's own wording, largest eigenvalue below 2, was right.)
  Test the bound is *real*:
  one test at a mobility just inside it (converges) and one just outside (diverges). A
  bound never exhibited failing is not a bound.
- The conserved quantity $S=z_1/\alpha_1+z_2/\alpha_2+x_1/\beta_1+x_2/\beta_2$ of
  `prop:meeting` is conserved **exactly** by the discrete map, because $K$ at fixed $g$
  depends only on differences, so its force components sum to zero. Check to machine
  precision, not to a tolerance.
- For `prop:capture`, which is not quadratic (the quartic $\varphi_\lambda$), the discrete
  argument needs the descent lemma on a neighborhood of the sublevel set: each mobility
  below $2/L$ with $L$ the Lipschitz constant of the gradient there. Estimate $L$
  numerically on the region the run visits and report it; do not assert the hypothesis
  holds globally.

## 5. The three examples, and exactly what to check

**(i) A prediction error moves a reader toward a source** (`eq:reader-cost`,
`eq:reader-force`). $H=W=P=\mathbb R$, three children, identity value maps, zero
arrangement cost, exterior input $(0_M,0)$, reader message $0_M$, sources $(1,1)$ and
$(0,1)$, reader prediction $\widehat e_1=((1,1),0)$ held fixed. Check the substituted cost
is `eq:reader-cost` and that at $(p_1,p_2,p_3)=(0,d,-d)$ with $d=\sqrt{2\tau\log 2}$ the
positional forcing is exactly $\frac{d}{4\tau}(dp_1-dp_2)$, the force on the third child
is zero, and one step with positive scalar mobility decreases $|p_1-p_2|$ and increases
$\kappa_{12}$. **State in the docstring what the paper states: this is a local response
calculation, not a convergence claim, and the reader's prediction is held fixed.** Do not
write a test called `test_reader_converges`. Add the book's own observation (§5.4) as a
separate test: give the sources zero response and the reader does all the moving.

**(ii) Rendezvous** (`eq:meeting-cost`, `eq:meeting-dynamics`, `prop:meeting`). Build it
by substitution of the two subsidiaries into the parent, never by writing $K$ by hand —
`eq:meeting-cost` must *emerge*, as the composite potential does in `hopfield.py`. Check
the state-force coefficients $(1,-1,0,0)$ at $g=0$, $(z,x)=(0,1,0,1)$ — the paper's point
that expectations move before positions — then the limit: $c=1/2$ for equal mobilities,
$c=11/13$ for $\beta_2=1/10$, both from `eq:meeting-point`, and $S$ conserved exactly
along the way.

**(iii) A transient cue selects an autonomous commitment** (`eq:cue-parent-cost`,
`prop:capture`, and the numerical table at the end of the paper). Check the nonnegativity
decomposition the paper gives for the cost; that the cue covector $(g-u)\,du$ is present
in the forcing; that from the resting state $(1,0,1,0,1)$ with $\lambda=64$, unit
mobilities and $u(t)=-12\,b_0(t/0.2)$ the capture condition `eq:capture-condition` holds
at $T=0.2$ with $\mathcal V\approx0.3836<1$; and that the autonomous run afterwards goes
to $(0,\tfrac12,\tfrac12,\tfrac12,\tfrac12)$ with $\mathcal V$ monotonically decreasing.
**The paper's table is RK4 of the ODE at step $10^{-3}$; the book's dynamics is the
discrete map.** So do not assert the table's digits of the discrete run. Assert instead:
the discrete run has the same limit; $\mathcal V$ decreases at every step under the
mobility bound of §4; and the discrete trajectory approaches the table's sampled values as
the step shrinks (report the observed difference at two or three step sizes, so the reader
sees the convergence rather than a bare pass).

**(iv) Placement changes the effective wiring** — the book's reclustering point (§5.4),
which no test in the paper isolates: with $\kappa$ as in `eq:spatial-transfers`, two
children driven far apart transfer a message of weight below any given $\epsilon$, so the
effective wiring changes within a fixed syntax during a run. One test, plus a docstring
noting what stays fixed (the number of boxes and their port types).

Finally, the paper's own caveat, to carry as a test docstring and not to paper over: the
Gaussian family supplies elementary operations but is **not closed under substitution** —
a composite generally has different effective value maps per sender–receiver pair and is
not a single Gaussian kernel. Write the test that shows this (compose two spatial
arrangements, exhibit a pair whose effective transfer is not $\kappa_{ij}V_j$).

## 6. Stages, caps, and what to report

Two runs, a check-in between, as in `DESIGN.md` §6.

- **(d1) The operad**, cap 250k: `dap/attention.py` with the message/transfer algebra, the
  arrangement constructor, the explicit `substitute`, the spatial family, and prediction
  costs; `tests/test_attention_operad.py` with everything in §2 and §3. Report: what
  agrees, what does not and why, any place this note was wrong or silent, and the shape of
  the module's public API.
- **(d2) The examples**, cap 250k, after David reads (d1)'s report: `tests/test_reader.py`,
  `tests/test_rendezvous.py`, `tests/test_cue.py`, `tests/test_placement_wiring.py`, per
  §4 and §5. Report the closed forms matched, the mobility bounds observed, the
  step-size study for (iii), and every place the discrete map differs from the paper's ODE
  in a way the book will have to state.

Keep the whole suite green (`uv run --extra test pytest`) at the end of each module. Never
edit `dap-core-old`; never commit.
