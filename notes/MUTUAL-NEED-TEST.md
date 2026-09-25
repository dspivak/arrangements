# The mutual-need test (`tests/test_mutual_need.py`)

Eight runs of the mutual-need construction of `notes/MEMBER-CENSUS-2026-09-24.md` §4.3 in the
book repository: two parts, each of whose held form is supplied by the other, with one supply
entering the pair from outside. `uv run --extra test pytest tests/test_mutual_need.py -q`:
8 passed (13 s). The four predictions are §4.3's, quoted in the module docstring and fixed
before any run here: **(1)** both forms are held while the supply runs; **(2)** cutting the
supply ends both; **(3)** cutting either internal link ends both; **(4)** neither part holds
its form alone.

## The construction

Each part is §5.6's driven steady state: an arrangement `<R^0|R^0> -> <R^0|R>` whose parameter
is the configuration coalgebra on its own amplitude `q` with mobility `h`, publishing `q` at
its port and carrying its own dissipation as the exact forcing of `(b/2) q^2` (the motor's
friction `b`). The supply is CDLM's drive idiom, a box with `sharp = 0` holding its value `s`.
The three are tensored and closed by a stateless coupler `<R^0|R^3> -> <R^0|R^0>` returning

    F_A(e) = alpha (intake + beta q_B),   F_B(e) = alpha gamma q_A,   F_S(e) = f_S(e)

on `e = (q_A, q_B, s)` — the two-tank idiom of `tests/test_two_tanks.py`. Every `out_f` is the
identity, so the composite forcing on `Q = R^3` is

    xi_A = alpha(intake + beta q_B) - b q_A,   xi_B = alpha gamma q_A - b q_B,   xi_S = f_S,

and one step is `q + h xi`. **Linear wiring:** `intake = s`. **Gated wiring:**
`intake = s g`, `g = q_A q_B / (K + q_A q_B)`. Fixed points are the zeros of `xi` and so do
not depend on `h`; what depends on `h` is which of them a run reaches.

Throughout: `alpha = b = K = 1`, `beta = gamma = 1/2`, `h = 0.01`, 20000 steps (200 units of
time), and a form is **held** at the end of a run iff its amplitude exceeds `1e-3`. `s = 1`
in the linear runs (§4.3's value) and `s = 4` in the gated ones. A **cut** wire delivers `0`
at the coupler, which is how §4.3's own hand-check read it. Test 0 checks that `Phi` of the
composite, `run_closed`, and the `fori_loop` used for the long runs agree exactly over fifty
steps, so every trajectory here is `Phi`'s.

## Test 1: the linear wiring reproduces §4.3's failure

Closed form, derived in the docstring: `q_A = alpha s/(b - alpha^2 beta gamma / b)`,
`q_B = (alpha gamma / b) q_A`.

| case | q_A | q_B | prediction |
|---|---|---|---|
| both links, supply on | 1.333333 | 0.666667 | (1) holds |
| supply cut | 1.1e-43 | 1.1e-43 | (2) holds |
| link B→A cut | 1.000000 | 0.500000 | (3) **fails**: both live |
| link A→B cut | 1.000000 | 5.1e-88 | (3), (4) **fail**: A holds alone |
| A alone (B absent) | 1.000000 | — | (4) **fails** |
| B alone (A absent) | 1.2e-87 | — | — |

This reproduces §4.3's hand-checked table to the digits it gives.

## Test 2: is the linear loop exact? Yes, and one-way supply is not what would cost it

`(d omega)_AB = alpha(gamma - beta)`, `(d omega)_AS = d f_S/d q_A - alpha`,
`(d omega)_BS = d f_S/d q_B`. With `f_S = 0` the form is non-exact, `|d omega| = 1` in the
`A`–`S` block. But the supply's response is zero, so `f_S` moves nothing and is free:
`f_S = alpha q_A` makes `d omega = 0` to `1e-12` at five random points, with potential
`V = alpha(q_A s + beta q_A q_B)` (checked against `F` to `1e-12`). At `beta != gamma` no
`f_S` repairs it: `(d omega)_AB = 0.4` at `beta = 0.5, gamma = 0.9`. **So the one-way supply
is not what costs exactness — non-reciprocal internal coupling is.** That is STATUS.md §3.5
item 1 one case over. `V` takes both signs at the sample points, so `k = -V` is not a
nonnegative cost and the linear loop is not an attention arrangement either.

## Tests 3 and 4: the gated loop, all four predictions

Closed forms derived in the docstring, with `c = alpha gamma/b = 1/2` and
`Bq = b - alpha^2 beta gamma/b = 3/4`:

    Bq c q_A^2 - alpha s c q_A + Bq K = 0,        s_* = 2 Bq sqrt(K/c)/alpha = 3/sqrt(2) = 2.121320,

two nonzero fixed points above `s_*`, coinciding there at `q_A = sqrt(K/c) = sqrt(2)`. At
`s = 4`: `q_A = 4.9274433` (stable) and `0.4058900` (a saddle), `q_B = q_A/2`.

| case | q_A | q_B | prediction |
|---|---|---|---|
| both links, supply on, from (1,1) | 4.9274433 | 2.4637217 | (1) **holds** |
| supply cut | 1.1e-43 | 1.1e-43 | (2) **holds** |
| link B→A cut | 2.5e-87 | 2.5e-85 | (3) **holds** |
| link A→B cut | 1.6e-85 | 1.2e-87 | (3) **holds** |
| A alone (B absent) | 2.5e-87 | — | (4) **holds** |
| B alone (A absent) | 1.2e-87 | — | (4) **holds** |

The settled state matches the closed form to `1e-6` with the forcing below `1e-9`. It is a
*driven* steady state, not an equilibrium: the intake `alpha s g = 3.6955825` is nonzero and
balances `b q_A - alpha beta q_B`, so flux passes while the amplitude stands still — §5.6's
"a fixed point in coordinates that are flow amplitudes".

## Test 5: the gate is a non-exact forcing, and no bookkeeping repairs it

    (d omega)_AB = alpha(gamma - beta) - alpha s q_A K/(K + u)^2,    u = q_A q_B,

reproduced to `1e-10` at five points for each of three choices of `f_S`, including one
nonlinear in all three ports. At the settled state it is **`-0.1141566`**; at the saddle
`-1.3858434`. It does not involve `f_S`, so the supply port's covector cannot repair it, and
it varies with `q_A` and `u`, so no constant `beta - gamma` can either (checked at two points
at once for three values of the gap). The same entry appears in the *composite's* forcing
Jacobian on `Q`, to `1e-10`: the parts' own dissipation is exact and diagonal and contributes
nothing to it. By the reciprocity lemma the forcing is not `dU` for any `U`. **So the gated
loop is inside `arr`, whose forcing need not be exact (STATUS.md §1), and outside the
attention operad, whose forcing is the full differential of a nonnegative cost.**

## Test 6: the restart threshold

The forcing vanishes identically at the origin (`atol = 0`), and since `dg = 0` there the
linearization is the unsupplied pair, eigenvalues `-b ± alpha sqrt(beta gamma) = -0.5, -1.5`:
**the origin is a stable fixed point with vanishing gradient, so a pair started too low never
gets going** — the same fixed point STATUS.md §3.5 records in the commons. The lower root is a
saddle (eigenvalues `0.377459`, `-1.684537`) and its stable manifold is the threshold, so the
threshold is a *curve*, not a number:

| ray from which the loop is started | threshold |
|---|---|
| `q_B = q_A/2` (through the saddle) | 0.4058900 = the saddle, `h`-independent |
| `q_B = 0` | 0.7435645 |
| `q_A = 0` | 0.6580994 |
| `q_A = q_B` | 0.2927452 |

On the ray through the saddle the crossing is the saddle itself for every mobility: the
bisection returns `0.40589000562522` at `h = 0.002, 0.01, 0.2` and `0.5` alike, against the
closed-form saddle `0.40589000562491`. Off it the threshold moves with the
step, first order, as an explicit discretization must: on `q_B = 0` it is `0.743081`
(`h = 0.002`), `0.743564` (`0.01`), `0.746122` (`0.05`), `0.758454` (`0.2`).

## Test 7: which verdicts the parameters and the wiring decided

§4.3 says of the linear failure "the reason is structural, not a matter of parameters". That
is right for the linear wiring. It is **not** right for the gated loop: three of its four
verdicts are decided by choices made here.

**(a) Prediction 2 holds only while the internal loop gain is below the dissipation,
`alpha^2 beta gamma < b^2`.** Chosen: `beta gamma = 0.25` against `b^2 = 1`.

| beta·gamma | after the supply is cut | |
|---|---|---|
| 0.25 | 1.1e-43 | dies |
| 0.64 | 1.5e-17 | dies |
| 0.9801 | 0.5000927 | still decaying at rate 0.01; the *run* says held |
| 1.00 | 3.6955825 | held with no supply at all (eigenvalue 0, the diagonal conserved) |
| 1.1025 | 8.1e+4 | grows without bound |

So prediction 2 is a parameter inequality, and near its boundary the run length decides what
a run sees rather than the loop.

**(b) Predictions 3 and 4 hold only for a fully gated supply.** An ungated fraction `leak`
leaves `A` at `alpha leak s / b` after the B→A cut — `0.0004` at `leak = 1e-4` — nonzero for
every `leak > 0`. Strictly the two predictions fail for any leak at all; under this run's
`HELD = 1e-3` they fail once `leak > 2.5e-4`. **Mutuality is knife-edge in the wiring.**

**(c) Prediction 4 is decided by which activities the gate reads.**

| gate | settled (q_A, q_B) | after the B→A cut | |
|---|---|---|---|
| `q_A q_B/(K + q_A q_B)` (chosen) | (4.92744, 2.46372) | (2.5e-87, 2.5e-85) | (3), (4) hold |
| `q_B^2/(K + q_B^2)`, the partner alone | (4.43050, 2.21525) | (2.2e-87, 2.3e-85) | (3), (4) hold |
| `q_A^2/(K + q_A^2)`, A's own activity | (5.13873, 2.56937) | (3.732051, 1.866025) | (3), (4) **fail** |

Autocatalysis is not mutuality. What the predictions need is that **every path from the supply
to `A` pass through `B`'s signal**; gating on the pair's joint activity is one way and gating
on the partner alone is another.

**(d) Prediction 1 holds only above the supply threshold**, and `s = 4` was chosen against
`s_* = 2.121320`: at `s = 2` nothing is held from any start (`8.6e-43` from `(5, 2.5)`), at
`s = 2.2` the loop settles at `1.85540`, and at exactly `s = s_*` it hovers at `1.44` after
200 units of time without settling, the two roots having merged. `s_*` scales as
`sqrt(K)`, so the gate's half-saturation decides it too.

**And the reading of the cut decides prediction 3.** Cutting the `beta` term alone and leaving
the gate reading `q_B` — the other reading of "cutting the internal link B→A" — leaves the
pair at `(3.4142136, 1.7071068) = (2 + sqrt 2, 1 + sqrt(2)/2)`, both held: prediction 3 fails
under that reading. The reading used above, one wire delivering `0`, is §4.3's own.

## What the runs support, and what they do not

They support the shape of §4.3's correction: the linear wiring's failure is reproduced
exactly, and a supply gated by the parts' activity does deliver all four predictions, with the
autocatalytic threshold §4.3 anticipated. They answer the open question §4.3 raised: the gate
*is* expressible as a forcing in this library, and it is genuinely non-exact — the
antisymmetric part is a closed form, `-0.1141566` at the settled state, and no bookkeeping at
the supply port removes it — so the construction sits inside `arr` and outside the attention
operad, as §4.3 expected. They also show that the one-way *supply* is not what costs
exactness; the gate is.

They do not establish that mutual need is generic. Three of the four verdicts turn on
quantities chosen here — `beta gamma` against `b^2`, a supply with no ungated path at all, a
gate reading the partner — and STATUS.md §2's warning applies in full: any smooth vector field
can be written as a forcing, so exhibiting a loop that behaves as wanted proves nothing about
the class of loops that do. What the runs deliver is the boundary of the region in which the
four predictions hold, stated above, not a demonstration that arrangements tend to fall in it.
The gate's Michaelis–Menten shape is a **model** of metabolism, not a derivation, and the
decay of an unsupplied amplitude is the book's deterministic descent, not an entropic force
(§4.3's "entropy and the bugs": the scattering is supplied by an environment the model does
not draw).

## Adapted from the brief

- §4.3 gives the parts as `dot q = alpha e - b q` in continuous time; here `e` is what the
  part receives at its port, the dissipation `-b q` is the part's *own* exact forcing from the
  cost `(b/2) q^2` (§5.6's friction, kept inside the part), and the drive is the coupler's
  returned covector. One covector, one step (STATUS.md §1, §3.2): the run is the discrete
  displacement map, and every fixed point quoted is a fixed point of that map exactly.
- "Cutting a link" is a wire delivering `0`, which is what §4.3's hand-check did. Under a gate
  that zeroes `q_B` in the gate as well as in the `beta` term, since one wire carries `q_B` to
  the coupler and the coupler uses it in both places. The other reading is run in test 7 and
  reverses prediction 3.
- "Neither part holds its form alone" is run as a genuine two-box composite (the part and the
  supply) whose coupler is the same `F` with the absent port's value `0`, not as a masked
  three-box run.
- `f_S`, the covector returned at the supply's port, is bookkeeping: the supply's response is
  zero, so no choice of it moves any trajectory. Test 2 uses that freedom; test 5 shows the
  gate's non-exactness survives every choice of it.
- The supply is a zero-response box, so nothing here depletes. The finite-reservoir lifetime
  bound (§6, P8) is a separate run and is not made here.
