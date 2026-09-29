# P32 (d) piece 2: the genetic code as an arrangement. Predictions, fixed before any code

Claude Fable 5.1, session ec5ece87, 2026-09-29. Piece 1 is
`notes/P32-D-SOURCES-2026-09-29.md`. The target is `scripts/genetic_code_stability.py`, plain
numpy rowwise entropic mirror descent, which this rebuilds as an arrangement.

## The construction, stated before it is written

- **One codon box per sense codon.** Base `Q = R^20`, the log-odds `y = log p(.|c)`; response
  `replicator.log_shahshahani(20, dt)`, the Shahshahani sharp pushed into the log chart;
  integrator `coalgebra.configuration`. Its forward port carries the odds `x = exp(y)`.
  **Its own forcing is zero.** Everything that moves it arrives through the wiring.
- **One slip box per unordered pair of sense codons one base apart** (263 of them): stateless,
  two ports of 20, cost `k = 2 w_{c,c'} (x^c)^T D x^{c'}`, where `w` is that pair's
  Freeland-Hurst weight and `D_{a,a'} = (PR_a - PR_{a'})^2`. Forcing `omega = -dk`.
- **One definiteness box per codon**: stateless, cost `beta * H(x)`.
- **One repertoire box**, wired to all 61: stateless, cost `-gamma * sum_a log u_a`,
  `u_a` the mean of `x^c_a` over codons.
- **The wiring** is a fan-out: a static arrangement whose forward map gathers each cost box's
  ports from the codon boxes' outputs and whose forcing is zero. The composite is
  `compose_seq(compose_seq(codon_boxes, fanout), cost_boxes) : I -> I`, closed, and `Phi` of it
  takes one step. The backward routing is not written: it is the transpose of the gather,
  produced by `compose_seq`'s `jax.vjp` pullback-and-sum.

## Predictions

1. **One step matches, small case.** On a hand-made mini code (4 codons over 3 amino acids), one
   `Phi` step of the arrangement, with each codon's row renormalized, equals one step of the
   script's mirror descent from the same start, to 1e-12 or better, at `dt = eta`.
2. **One step matches, full case.** The same on the full 61 sense codons over 20 amino acids,
   same tolerance, at several `(beta, gamma)`.
3. **The mismatch before renormalizing is exactly a gauge.** Unnormalized, the arrangement's row
   is the script's times `exp(dt <x^c, g^c>)`, one scalar per codon, because the log-chart
   Shahshahani sharp's `-1 1^T` term subtracts the same number from every amino-acid coordinate.
   *This could come out otherwise:* if the correction varied with the amino acid, renormalizing
   would not repair it and predictions 1 and 2 would fail.
4. **The slip cost emerges; `W` is never assembled.** The force the wiring delivers to codon `c`
   equals the script's `-(2 (W P D)_c + repertoire + definiteness)` to 1e-12, while the
   arrangement code contains no 61x61 matrix and no sum over neighbours: one weight per wired
   pair and one 20x20 `D` are all that is written. *This could come out otherwise* in three ways
   a wiring really can fail: the factor 2 (one box per unordered pair needs `2 w`, and writing
   `w` halves the force), the transpose (a gather whose adjoint is not scatter-add would drop
   the contributions of all but one neighbour), and double counting (a box per ordered pair
   would double it).
5. **Negative controls hold.** Dropping the slip boxes leaves exactly repertoire plus
   definiteness; dropping the repertoire box leaves exactly slip plus definiteness; and a codon
   wired to no slip box receives zero slip force. *This could come out otherwise:* a wrong
   adjoint leaks force between ports, and this is how it would show.
6. **The flat n-ary monoidal product agrees with the library's binary fold** to 1e-14 on the
   small case. (A flat version is needed only because folding `parallel_arrangements` 386 times
   nests the state 386 deep; it computes the same direct sum.)

## What this cannot establish, stated in advance

Predictions 1, 2 and 6 are largely **what the construction was built to have** (`STATUS.md` §2):
an arrangement whose cost is assembled from the same terms will step the same way, and the
interesting content is only that the *response and the wiring*, not a hand-written gradient,
produce it. Predictions 3, 4 and 5 are the ones that could fail, and they are where any result
is. Nothing here bears on whether the standard genetic code is special; that was piece 1.

---

## Verdicts

Built: `arrangements/genetic_code.py`, `tests/test_genetic_code.py`.
`scripts/run_tests.sh 180 tests/test_genetic_code.py`: **9 passed in 72 s.**

The full code wires **61 codon boxes + 263 slip boxes + 61 definiteness boxes + 1 repertoire
box = 386**, and 263 is the edge count the literature gives for the sense-codon graph (Buhrman
et al., arXiv:0909.1442v2, p. 4, "yielding 263 edges"), which the test checks.

1. **HELD.** Small case, `max |Phi step - script step| = 1.1e-16`.
2. **HELD.** Full code, 61 x 20, `5.6e-16` at `beta = gamma = 0` and `6.7e-16` at
   `beta = 2, gamma = 3`. One step of the composite takes 32-45 s: the forcing traces all 386
   boxes. (This is a tracing cost, not a dynamical one.)
3. **HELD, and it was not trivially true.** The unnormalized ratio is constant along each row to
   `4.4e-16` and equals the predicted `exp(dt <x, g>)` to the same order, and it is not 1 (the
   small case's factors run 1.008 to 1.046). So the log chart's departure from `sum x = 1` is
   exactly one scalar per codon, pure gauge, and renormalizing is a choice of representative
   rather than a repair.
4. **HELD.** The covector the wiring delivers to the codon boxes equals `-x * g` at full size to
   `7.1e-15`, `g` being the script's gradient with its 61 x 61 `W` assembled by hand in the test.
   The arrangement forms no matrix over codons: a slip box's whole interface is `2 x 20`, and
   the sum over a codon's nine neighbours exists only as the adjoint of the fan-out's gather.
   All three named failure modes were live and none occurred: the factor 2 on an unordered pair
   is right, the adjoint of the gather is the scatter-add, and nothing is double counted.
   The locality half is checked separately: deleting one slip box changes the force on exactly
   the two codons it joined, by exactly `2 w D x'`, and on no other codon (`< 1e-14`).
5. **HELD.** Each cost switches off independently and leaves exactly the others' gradient
   (`< 1e-12` in all three directions); a codon wired to no slip box gets **exactly zero**
   (not `1e-16`), while its wired neighbours do not.
6. **HELD.** `tensor_all` agrees with `wiring.tensor_arrangements` on positions, forward map and
   forcing to `1e-14`.

### Which of these could have come out otherwise

3, 4 and 5. Predictions 1, 2 and 6 are close to what the construction was built to have
(`STATUS.md` section 2): a composite assembled from the same three costs will step the same way,
and their content is only that no gradient was written down. Prediction 3 is a closed form that
the chart could have failed. Prediction 4 is the stage's result and had three named ways to fail.
Prediction 5 is the control that would have caught a leaking adjoint.

### What a molecular biologist would object to, and whether the model survives

- **"Polar requirement is a poor stand-in for the damage a mistranslation does."** This stands,
  and the arrangement does not answer it; piece 1 sharpened it, since on the 2008 recomputed
  scale the first amino acids lost change from Asp and Glu to Cys and Glu, so "Asp goes first"
  was about the scale's extreme, not about aspartate. The model **survives as a model of
  descent**, since nothing in the wiring depends on what `D` means; it does **not** survive as a
  claim about which amino acid a cell can least afford, and no such claim may be made from it.
- **"A cell needs Asp whatever its polar requirement is."** The repertoire box is the only thing
  standing against losing an amino acid, and it is a log barrier on the *mean* odds, not a
  per-amino-acid requirement. A biologist would say the requirement is essentially per amino
  acid and unequal. That objection stands and is not answered here.
- **"Codons are not free to reassign; a change breaks every protein at once."** The model has no
  such term: it descends as if each codon were independent, which is why it needs the
  definiteness weight `beta` to hold anything at all. The honest reading is that `beta` is
  standing in for a cost the model does not have.
- **"Mistranslation weights are not mutation weights."** Freeland and Hurst's weights are about
  translation error; the repertoire and definiteness costs are about what a code holds. The
  model runs them in one gradient. This is a real conflation and it survives only as a model.

### Reasonable, and weird (Claude's judgment, marked as judgment, not evidence)

**Reasonable.** That the total force on a codon is the sum of nine pairwise pulls plus two
global ones is what a chemist would draw; getting it from a gather's transpose rather than a
matrix is the library's ordinary behaviour. At the scale of a society or a body, the same shape
reads correctly too: a part's pressure is the sum of what each neighbour asks of it, and no one
computes the whole adjacency.

**Weird, in four ways.** (i) The codon box has **no cost of its own** -- a codon wants nothing;
every force on it is another box's. That is faithful to the script but strange as biology, where
a codon's tRNA pool is its own affair. (ii) The **definiteness cost is an entropy penalty with
no biological referent**: `beta` has no measured value, and piece 1 showed the threshold moves
with the polar-requirement scale (from about 2 down to about 1.25), so it is a dial, not a
constant. (iii) The **repertoire cost pulls every codon toward the rarest amino acid**, which
at the scale of a body would read as a tissue drifting toward whatever cell type is scarcest --
an artifact of a log barrier on a mean. (iv) One `Phi` step costs 30-45 s where the script's
costs microseconds; the arrangement is here to say what the composite *is*, not to run it, and
the 20 000-step descents in piece 1 stay in numpy.

### Where this does and does not bear on the book

It bears on Chapter 4's design: it is the first case where a cost with the shape "a sum over a
graph's edges" is shown to emerge from wiring alone at a realistic size (386 boxes), with the
`2 W P D` of the paper literature never formed. It does **not** bear on the section 1.5
inventory entry about the genetic code; that is piece 1's business, and piece 1 found the
earlier note's numbers right but its strongest sentence ("none of 2000 shuffles has as few")
scale-dependent.
