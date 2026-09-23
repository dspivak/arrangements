# How the mathematics maps onto the code

This is the correspondence between the book's definitions and this library's datatypes.
It says how the code is shaped and why. For what the tests establish and what they do
not, see [`STATUS.md`](../STATUS.md); for conventions of sign and time, `STATUS.md` §1.

The ambient category is smooth sets, sheaves on Cartesian spaces. **The code stays on
Cartesian spaces `R^d`**, an admissible subcategory containing every example, and records
a smooth set by its dimension (`STATUS.md` §3.1).

## 1. Objects and their datatypes

| the mathematics | the code |
|---|---|
| interface `M = <M- \| M+>`, two smooth sets | a pair of dimensions, `in_dim` and `out_dim` |
| `Omega^1`, one-forms; a map `X -> Omega^1` *is* a one-form on `X`; writer monad `Omega(X) = X x Omega^1` | a one-form on a product is a callable returning the covector's components on each factor: `omega(x_a, x_b, x_c) -> (xi_a, xi_b, xi_c)` |
| a lens `M -> N` in `omegalens`: `f+ : M+ -> N+` and `f- : M+ x N- -> Omega(M-)` | `out_f(m_out) -> n_out`, `in_f(m_out, n_in) -> m_in`, and `omega(m_out, n_in)`, a one-form on `M+ x N-` |
| Kleisli composition in `omegalens` | pullback-and-sum; §2 |
| an object `(Q, S, u)` of `cotcoalg`, `u : S -> cot(Q)(S)`, i.e. `s |-> (q(s), xi |-> s')` | `CotCoalg(Q_dim, init, position, step)`. The response `sharp` lives *inside* `step`; it is not a field |
| tensor of coalgebras: base `Q x Q'`, state `S x S'` | `CotCoalg.tensor`: state a pair, position the concatenation, step componentwise after splitting the covector |
| `arr = Para(cotcoalg, omegalens)`; an arrangement with parameter `P` is a lens `Q (x) M -> N` | `Arrangement(P, in_dim_M, out_dim_M, in_dim_N, out_dim_N, out_f, in_f, omega)`, the maps now taking `q` first |
| `omega = dU` where a potential exists | `exact(U)`, which is `jax.grad(U, argnums=(0,1,2))` |
| the dynamics functor `Phi : arr -> pc` | `Phi(arr) -> PCMorphism`, taking no integrator argument |
| `Phi` on an interface: positions `M+ x Omega(M-)`, directions `(M+)* x M-` | `phi_box_poly`, with `trivial_omega` for a trivial input side |
| displacement semantics: one covector, one step | the coalgebra's `step` |

## 2. Composition is pullback-and-sum

For `f : L -> M` carrying `omega_f` on `L+ x M-`, and `g : M -> N` carrying `omega_g` on
`M+ x N-`, the composite carries, at `(l, n)`,

```
omega_{g.f}(l, n) = A* omega_g  +  B* omega_f,
    A : (l, n) |-> (f+(l), n)
    B : (l, n) |-> (l, g-(f+(l), n))
```

In code: evaluate each constituent at its own local point and pull it back with
`jax.vjp` of the map above, then add. With parameters, `q` and `q'` come along: the
composite's parameter is the tensor, and its one-form lives on `Q x Q' x L+ x N-`.

This is what replaces CDLM's "sum the two potentials at the composed points." By the
chain rule the two agree whenever both are exact; when they are not, only the pullback
form is correct. `parallel_arrangements` and `tensor_arrangements` are the direct sum of
one-forms, with no pullback. Wirings (`chain_wire`, `graph_wire`) carry the zero one-form
and a trivial parameter.

The interpretation pulls the *incoming* field back along `in_f`. The published field is
`omega_N(n) = (f-(q, m, -))* omega_M + omega(q, m, n)|_{N-}`, and the covector triple is
`xi_Q = (f+)* xi_N|_Q + (f-)* omega_M|_Q + omega|_Q`, likewise for `xi_M`.

## 3. Why the sign translation exists

The book takes `omega = dU`, so a cost `k` gives `U = -k` and `omega = -dk`, and the
configuration coalgebra **adds**: `q' = q + sharp(omega)`. Descent of `k` is then
automatic, carried by the forcing's own sign.

CDLM instead calls the cost itself `U` and has the integrator subtract. So every ported
example was translated: `U_old` became `exact(lambda q, m, n: -U_old(q, m, n))`, and the
step became `s + sharp(s, xi)`. The phase coalgebra flips the sign of its kick the same
way, the position half unchanged.

The translation reaches past the potentials. Environment covectors and published fields
are forcings, so where the old code supplied a cost gradient this one supplies its
negative: a pin with cost `(kappa/2) q^2` sends `xi_N = -kappa q`. States are unchanged.
Sources written in CDLM's convention — the retroactivity note among them — are imported
with a minus.

Because a sign error here is silent, every ported example carries a recorded trajectory
from the old implementation in `tests/regress/`, and reproduces it. See `STATUS.md` §2
for what that does and does not show.

## 4. The standard coalgebras

`ReactiveVectorSpace` survives as a *constructor input* — a position-dependent `sharp` —
rather than as the parameter itself.

- `configuration(Q)`: state `S = Q`, position the identity, step `s + sharp_s(xi)`.
- `phase(Q)`: state `T*Q = Q x Q*`, position the first factor, one symplectic-Euler step.
- `damped`, `gyro`, `phase_on_cotangent`, and the multi-stage `leapfrog`, `rk4`,
  `CotCoalg2`, `CotCoalgK` are extensions beyond either paper, marked as such where they
  are defined. Whether the multi-stage assignment is functorial is open (`STATUS.md` §3.4).

Since the parameter no longer exposes a `sharp`, tests that need one read it off the
coalgebra as the Jacobian of `step` at zero covector. For a configuration coalgebra that
*is* the sharp, and it turns assertions about a potential into assertions about the
forcing, which are stronger: symmetry of a Hessian becomes exactness of the composite
one-form.

## 5. The attention operad

An `n`-ary attention arrangement is `(I, l, T, k)`: an integrator, a placement map, a
family of message transfers with zero diagonal, and a nonnegative cost quadratic in the
interaction. The port is close to mechanical because the paper's data are already an
arrangement's data:

| the paper | the code |
|---|---|
| interaction space `D_n = M^n x E` | exactly the `(m_out, n_in)` pair |
| sending map `sigma_a(q, m) = sum_j T_0j(q) m_j` | `out_f(q, m_out)` |
| receiving map `rho_a(q, m, e)_i` | `in_f(q, m_out, n_in)` |
| forcing `omega_a = -dk_a`, the *full* differential | `exact(lambda q, m, n: -k(q, m, n))` |

The paper already uses the book's convention (`U = -k`, `omega = dU`), so there is **no
sign translation here**, unlike every CDLM module.

Two layers, matching the paper's own. `AttentionArrangement` is an element of
`Attention(n)`, because substitution is defined on `(I, l, T, k)` and not on the lens
they produce — a composite has to be substitutable again. Its `.arr()` is the operad
morphism into `End_arr(B)`. Substitution is implemented twice on purpose: `substitute`
follows the paper's explicit formulas with no reference to `wiring.py`, and the same
composite is built by `compose_seq(tensor_arrangements(...), a)`. Testing that these
agree is the operad-morphism proposition made executable.

Two limits the module declares rather than hides. The integrator is `Store(S)` only, not
the paper's general polynomial comonad (`STATUS.md` §3.3). And the operad laws hold up to
the coherence isomorphism, not on the nose: substitution takes `Q_b = Q_a x prod_i Q_{a_i}`,
so different bracketings order the configuration factors differently and the symmetric
action permutes them; the tests compare after the evident block permutation, which
Definition 1's "up to integrator isomorphism" licenses.

Guards that must read concrete values — message admissibility, the mean readout's
positive weight, `validate` — are eager-only and not `jit`-able, and `validate` checks
the transfer conditions at one configuration, not for all `q`.

## 6. What is worth testing, and why

Three kinds of test carry the weight, and `STATUS.md` §2 says what each is worth:

- **Closed forms.** Where a source states a formula, a limit, or an exact value, compute
  it independently and compare.
- **Two independent implementations.** The attention operad's substitution, above.
- **Recorded trajectories.** Faithfulness of the port, nothing more.

The tests that could not be written before forcing one-forms existed are the ones worth
adding to first: reciprocity (`d omega = 0` for the pipe, nonzero for the waterfall, and
`d omega` the antisymmetric part of the returned-covector Jacobian), the two tanks with
three couplers, composition agreeing with dynamics for a non-exact coupler, and heat and
wave built as two arrangements on one wiring.
