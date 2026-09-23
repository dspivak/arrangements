# arrangements: the category of arrangements $\arr$, executable

> **Working design note, kept as the record of why the code is shaped this way.**
> It was written before the code, to be executed against; it is not a specification of
> the current API, and its process details (token budgets, staging, "stop and report")
> are how the work was actually carried out. For what the library establishes and what
> it does not, see [`STATUS.md`](../STATUS.md).

Design note (Claude, Opus 5, for David Spivak, 2026-09-21). This is the contract every
stage of the port is built against. The mathematics it implements is the book
*Articulating Forms that Matter*, Chapter 3 (`3-arrangements-that-matter.tex`,
`def:arr`) as elaborated in `AGENTDOCPLAN.md` §4.2, §4.8, §4.9; the source it is
ported from is `../dynamic-algebra-potentials/dap-core-old` (tag `cdlm-final`), the executable companion of CDLM.
Do not read `CDLM.tex` whole; read the old module you are porting.

## 0. What changes, in one paragraph

CDLM's arrangement carries a **potential** $U\colon Q\times M^+\times N^-\to\mathbb R$ and
chooses its **integrator** in the semantics (`Phiconf`, `Phiphase`). The book's
arrangement carries a **forcing one-form** $\omega$ on $Q\times M^+\times N^-$, which need
not be exact, and its **integrator is part of the arrangement**: the parameter is a
$\cot(Q)$-coalgebra $(Q,S,u)$, not a reactive vector space. There is **one** dynamics
functor $\Phi\colon\arr\to\pc$. Everything else (polynomials, $\pc$, the covector
triple, lens internalization, wiring) is unchanged in substance.

## 1. The mathematics, and the datatype for each piece

Smooth sets are $\smoothset=\operatorname{Shv}(\operatorname{CartSp})$. In code, v1 stays
on Cartesian spaces $\mathbb R^d$, as before (a "smooth set" is a dimension). This is
the admissible subcategory of §4.8(c); say so in the docstring, do not pretend
otherwise.

| the book | code |
|---|---|
| interface $M=\langle M^-\mid M^+\rangle$, two smooth sets | `(in_dim, out_dim)` |
| $\Omega^1$, one-forms; a map $X\to\Omega^1$ *is* a one-form on $X$; writer monad $\Omega(X)=X\times\Omega^1$ | a one-form on $\mathbb R^{a}\times\mathbb R^{b}\times\mathbb R^{c}$ is a callable `omega(x_a, x_b, x_c) -> (xi_a, xi_b, xi_c)` returning the covector's components on each factor at that point |
| $\omegalens$ morphism $M\to N$: $f^+\colon M^+\to N^+$ and $f^-\colon M^+\times N^-\to\Omega(M^-)=M^-\times\Omega^1$ | `out_f(m_out) -> n_out`, `in_f(m_out, n_in) -> m_in`, and `omega(m_out, n_in) -> (xi_mout, xi_nin)`, a one-form on $M^+\times N^-$ |
| Kleisli composition of $\omegalens$ = **pullback-and-sum** | see §2 |
| $\cotcoalg$ object $(Q,S,u)$, $u\colon S\to\cot(Q)(S)$, i.e. $s\mapsto(q(s),\ \xi\mapsto s')$ | `CotCoalg(Q_dim, init: S, position: S -> Q, step: (S, xi_Q) -> S, label)`; the response $\sharp$ lives inside `step`, it is not a separate field |
| tensor of coalgebras: base $Q\times Q'$, state $S\times S'$, product coalgebra | `CotCoalg.tensor` : state a pair, position the concatenation, step componentwise after splitting `xi_Q` |
| morphism $(f,g)$ of coalgebras with $u'\circ g=\cot(f)_{S'}\circ\cot(Q)(g)\circ u$ | not a datatype; one test that checks the equation for a reparameterization (optional, stage c) |
| $\arr=\para{\cotcoalg}{\omegalens}$; an arrangement $M\to N$ with parameter $P=(Q,S,u)$ is an $\omegalens$ morphism $Q\otimes M\to N$ | `Arrangement(P: CotCoalg, in_dim_M, out_dim_M, in_dim_N, out_dim_N, out_f(q, m_out), in_f(q, m_out, n_in), omega(q, m_out, n_in) -> (xi_q, xi_mout, xi_nin), label)` |
| $\omega=dU$ when a potential exists | helper `exact(U)`: `omega = jax.grad(U, argnums=(0,1,2))`. Every CDLM example becomes `omega = exact(lambda q,m,n: -U_cdlm(q,m,n))`; see §3 |
| dynamics functor $\Phi\colon\arr\to\pc$, one functor | `Phi(arr) -> PCMorphism`; no integrator argument; `Phiconf`/`Phiphase` are deleted |
| $\Phi$ on an interface: positions $M^+\times\Omega(M^-)$ (the output and the one-form the box publishes on its input port), directions $(M^+)^*\times M^-$ | unchanged: `phi_box_poly`, `trivial_omega` |
| displacement semantics: one covector, one step | the coalgebra step; time is discrete by design, no ODE reading |

## 2. Composition: pullback-and-sum

For $f\colon L\to M$ with one-form $\omega_f$ on $L^+\times M^-$ and $g\colon M\to N$ with
$\omega_g$ on $M^+\times N^-$, the composite $g\circ f$ has, at $(l,n)$,
$$
\omega_{g\circ f}(l,n)
= \bigl((l,n)\mapsto(f^+(l),\,n)\bigr)^*\,\omega_g
\;+\;
\bigl((l,n)\mapsto(l,\,g^-(f^+(l),n))\bigr)^*\,\omega_f .
$$
In code: evaluate $\omega_g$ at $(f^+(l),n)$ and $\omega_f$ at $(l,g^-(f^+(l),n))$, and
pull each back with `jax.vjp` of the map written above; add. With parameters, the
same with $q$ and $q'$ carried along: the composite's parameter is $P\otimes P'$ and its
one-form lives on $Q\times Q'\times L^+\times N^-$. **This replaces `compose_seq`'s "sum the
two $U$'s at the composed points": by the chain rule the two agree when both are exact,
and only the pullback form is right when they are not.** `parallel_arrangements` and
`tensor_arrangements` are the direct sum of one-forms (no pullback). The wirings
(`chain_wire`, `graph_wire`) have zero one-form and trivial parameter, as before.

The interpretation (`interpretation.py`) already pulls the *incoming* field
$\omega_M$ back along `in_f` with `vjp`; the only change is that `jax.grad(U)` at three
places becomes the corresponding component of `omega(q, out_m, in_n)`. The
published field is $\omega_N(n)=(f^-(q,m,-))^*\omega_M+\omega(q,m,n)|_{N^-}$; the
covector triple is $\xi_Q=(f^+)^*\xi_N|_Q+(f^-)^*\omega_M|_Q+\omega|_Q$, likewise for
$\xi_M$.

## 3. Sign convention (the one place a port can go silently wrong)

The book: whenever a potential exists, $\omega=dU$; for a cost $k$ one has $U=-k$, so
$\omega=-dk$; the configuration coalgebra **adds**: $q'=q+\sharp\,\omega$. Descent of $k$
is then automatic.

CDLM and `dap-core-old`: `U` *is* the cost and the integrator subtracts,
`s - Q.apply_sharp(s, xi)`.

Translation, applied mechanically to every ported module: `U_old` becomes
`omega = exact(lambda q, m, n: -U_old(q, m, n))`, and the configuration coalgebra's
step becomes `s + sharp(s, xi)`. The phase coalgebra flips the sign of its kick the same
way (the position half is unchanged). Incoming environment covectors `xi_N` and
published fields are forcings, so a test that supplied a cost gradient in the old code
supplies its negative here (a pin with cost $\tfrac{\kappa}{2}q^2$ sends $\xi_N=-\kappa q$);
only states are unchanged. The retroactivity note's one-forms are also in CDLM's sign
(`AGENTDOCPLAN.md` §4.9): import them with a minus.
**Acceptance:** every ported example reproduces the old trajectory to machine precision
(`jnp.allclose` with `rtol=1e-10` under `jax_enable_x64`) when the old test is run on
`dap-core-old` and the new one on `arrangements`, on the same seeds and inputs. Write that
comparison as a test that imports nothing from the old package: record the old
trajectories once (a small `.npz` per example, generated by a script in
`tests/regress/`, kept in the repo) and compare.

## 4. The standard coalgebras (what `rvect.py` becomes)

Keep `ReactiveVectorSpace` (position-dependent sharp) as a *constructor input*, not as
the parameter. Provide:

- `configuration(Q: ReactiveVectorSpace) -> CotCoalg`: state $S=Q$, position the
  identity, step $s+\sharp_s\xi$.
- `phase(Q: ReactiveVectorSpace) -> CotCoalg`: state $T^*Q=Q\times Q^*$ in coordinates,
  position the first factor, step the book's one-step phase update (port the old
  `phase_integrator` with the sign of §3).
- `damped(...)`, `gyro(...)`, and the multi-stage `Integrator2`/`IntegratorK`, `leapfrog`,
  `rk4`: port as extensions in stage (c), docstrings keeping their "beyond the paper"
  and "monoidality open" (§4.5 item 7) status verbatim. Anything the old code marks
  EXTENSION stays marked EXTENSION.

## 5. Tests that are new, because they could not be written before

- **Reciprocity.** A stateless two-port coupler with one-form $\omega$: check
  $d\omega$ against the antisymmetric part of the returned-covector Jacobian, on the
  pipe (exact, $d\omega=0$) and the waterfall $\omega=\varphi(e_1)(de_1-de_2)$
  ($d\omega\ne0$); `background/note-retroactivity.tex` §3 in the book repo.
- **Two tanks, three couplers** (book Ch 2 §2.1): pipe, waterfall, pump, each as an
  arrangement; the composite ODE step matches the note's two lines; total water is
  conserved for the pipe and for the waterfall (the stream lands in tank 2; what the
  waterfall loses is reciprocity, not conservation) and not for a leak.
- **Composition agrees with dynamics** for a non-exact coupler: run the composite of
  two boxes and a waterfall as one arrangement and as `Phi` of the parts composed in
  $\pc$; equal step-for-step. (The old `test_composition.py` does this for exact ones.)
- **The integrator is part of the arrangement:** the same wiring with `configuration`
  and with `phase` parameters is two *different* arrangements; `Phi` takes no
  integrator; there is no way to run "the same arrangement two ways". A test that the
  heat and wave examples are built as two arrangements on one wiring.

## 6. Stages, caps, and what to report

Each stage is one agent run with a token cap; stop and report at the cap or at the
end, whichever comes first. Do not start the next stage in the same run.

- **(b) Core**, cap 250k. `arrangement.py`, `coalgebra.py` (new; `rvect.py` reduced per
  §4), `interpretation.py`, `functors.py` (one `Phi`), `wiring.py`, `polynomial.py` and
  `pc.py` copied unchanged. Port exactly these examples and their tests, with the
  regression `.npz` files: `learning.py` (gradient descent, backprop),
  `test_heat_equation.py` and `test_wave_equation.py` (one wiring, two arrangements),
  `test_composition.py`, `test_functors.py`, `test_wiring.py`, `test_rvect.py`,
  `test_polynomial.py`. Add the §5 reciprocity and two-tanks tests. Report: what was
  ported, the test count old/new, the regression status per example, any place the
  design note was wrong or silent.
- **(c) Examples**, cap 250k, after David has read (b)'s report. The remaining modules,
  one at a time, each with its regression test: amplifier, dc_motor, lc_circuit,
  kuramoto, hopfield, replicator, moleculardyn, reaction_network, logic, control,
  gyroscope(+faithful), system_id, pinn, newton/optimizers, the multi-stage
  integrators, `build.py`/`demo.py`. Non-gradient examples the old code realized by
  tricks (reaction networks with a detailed-balance potential, Lotka–Volterra with
  oppositely signed sharps) keep those realizations *and* gain the direct non-exact
  one-form realization as a second test, with a docstring saying which is which.
- **(d) The commons**, separate design note first (book plan §6 item 43(b1)); not part
  of this port.

## 7. Rules

- Match the old code's idiom: docstrings cite the book's labels (`def:arr`,
  `sec.forcing`, `sec.integrators`, …) the way the old ones cite CDLM's; where the
  book has no label yet, cite `AGENTDOCPLAN.md §4.2` and say so.
- Names follow the book's `NOTATION.md`: the parameter's base is $Q$, its state $S$,
  the coalgebra $u$, the forcing $\omega$, the dynamics functor bare $\Phi$.
- No new abstractions beyond the table in §1. If a datatype is not in the table, ask.
- Never edit `dap-core-old`. Never commit. Run tests with `uv run pytest` from
  `arrangements` (copy `.venv` handling from the old `pyproject.toml`; the lock file is
  already here).
