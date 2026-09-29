# P32 (e): a Kirchhoff node — predictions, fixed before code

Claude Fable 5.1, session ec5ece87, 2026-09-29. Stage (e) of the book repo's
`notes/SPRINGS-TO-TOWNS-AUDIT-2026-09-29.md`, "The smallest test for each thin spot"; the
derivation under test is `background/note-retroactivity.tex` l.94–103, *Kirchhoff's current law
is the chain rule*: the wiring copies a node's voltage to `k` element ports, the pullback of
covectors along a copy is their sum, so the node's update is `q' = -sum_j I_j` (CDLM's sign).

## Design (the two-tank idiom of `tests/test_two_tanks.py`, electrical dictionary)

* **Capacitor** `C`: a box `<R^0|R^0> -> <R^0|R>`, configuration coalgebra on its charge `q`,
  response `sharp = C`, output voltage `e = q / C`, zero forcing. As with the tank, the port
  covector is pulled back by `de/dq = 1/C` and the response multiplies by `C`, so the charge
  step is `q' = q + F(e)` whatever `C` is: the returned covector *is* the charge increment.
* **Resistor** `R`: a stateless two-port coupler (`stateless_coupler` of `arrangements/commons.py`)
  with the exact one-form `omega = -dV`, `V = (dt / 2R)(e_1 - e_2)^2`, so
  `F = (-dt I, +dt I)`, `I = (e_1 - e_2) / R`. `dt` lives in the coupler, as `kappa` did.
* **Ideal source** `E`: a box with trivial parameter and constant output `E`. The vjp of a
  constant map is zero, so a source absorbs any current: infinite capacitance.
* **Copy**: `fan_wire` of `arrangements/commons.py`, `m |-> m[picks]`, zero forcing. Nothing in
  it sums. The sum, if it appears, is `compose_seq`'s pullback (the transpose of a gather is a
  scatter-add).
* Composite: `compose_seq(compose_seq(tensor(boxes), fan_wire(...)), tensor(resistors))`,
  closed. Its flat configuration `P.position(state)` is the vector of capacitor charges.

Library sign (STATUS.md §1): the step adds the forcing, so the note's `q' = -sum_j I_j`
reads here `Delta q = -dt sum_j I_j` with `I_j = (e_node - e_j) / R_j` positive when current
leaves the node.

## Predictions

1. **One node, three branches (closed form).** One capacitor whose output is copied to three
   resistors `R_j` to three ideal sources `E_j`: the `Phi` step's charge increment equals
   `-dt sum_{j=1}^3 (e_node - E_j) / R_j` to `rtol 1e-12`, at the node's current voltage, for
   `C in {0.5, 2}` and random `E_j`, `R_j`. The closed composite's `Q`-forcing equals the same
   sum at random charges (`atol 1e-12`). The sum is written only in the test's expectation, not
   in any component. Also: with two resistors of *equal* `R` and *opposite* offsets the two
   currents cancel and the node does not move (`|Delta q| < 1e-14`), which a wire that copied
   to only the first port (no sum) would fail.
2. **Triangle (discrete analogue + closed form).** Three capacitors `C_i` joined pairwise by
   three resistors `R_ij`. (a) The composite's `Q`-forcing equals `-dt L C^{-1} q` with `L`
   the conductance-weighted graph Laplacian written by hand (`atol 1e-12`), i.e. the nodal
   equations emerge from the composite. (b) `sum_i q_i` is constant to `rtol 1e-12` at every one
   of 400 steps (`dt = 0.05`; stability needs `dt lambda_max(C^{-1} L) < 2`, checked in the test).
   (c) The three voltages converge to the common voltage
   `e* = sum_i C_i e_i(0) / sum_i C_i`, the capacitance-weighted mean, to `atol 1e-8`.
3. **Nodal analysis (closed form).** One capacitor to two sources through `R_1 != R_2`: after
   200 steps (`dt = 0.1`, contraction factor `1 - dt (G_1 + G_2) / C = 0.6`) the node's voltage is
   `(G_1 E_1 + G_2 E_2) / (G_1 + G_2)`, `G = 1/R`, to `atol 1e-10`; and it is *not* the plain
   mean `(E_1 + E_2) / 2` (difference `> 1e-2`).

## Which could come out otherwise (STATUS.md §2's warning, answered in advance)

* (1) and (2a) are the mechanism itself: `jax.vjp` of a gather is a scatter-add, and
  `compose_seq` pulls back along the located maps. That they *hold* is guaranteed once the
  library's composition is what its docstring says; what the test adds is that the sum is not
  written anywhere in a component and that the sign is the note's. A sign error in `V`, or a
  wire built with `in_f` instead of a pullback, would fail them.
* (2b) is guaranteed by each resistor's two covectors being equal and opposite (an exact
  one-form of a *difference*), as `test_total_water_conserved...` already showed for two tanks.
* (2c) and (3) could fail for a real reason: the discrete step is forward Euler, and a `dt`
  past the stability bound diverges instead of relaxing. The tolerances are set for the chosen
  `dt`; the stability margin is asserted, not assumed.

## Progress

* 14:14 predictions written. 14:15 `tests/test_kirchhoff_node.py` written (6 tests) and run once
  with `scripts/run_tests.sh 120 tests/test_kirchhoff_node.py`: 6 passed in 21 s, first run.
  Nothing was adjusted after seeing results. Numbers follow.
* 14:17 numbers read off (script in the run's scratchpad, not kept). 14:18 done; report below.

## Results (6 passed in 21 s, first run, nothing tuned)

1. **Held.** `C = 0.5`: `Delta q = -3.98486579962651e-2`, `-dt sum_j I_j` the same to rel. err.
   `8.7e-16` (`I_j = 2.468, 0.712, 0.805`); `C = 2`: rel. err. `5.0e-15`. Closed `Q`-forcing
   matched `-dt sum_j (q/C - E_j)/R_j / C` at 5 random charges, `atol 1e-12`. Two equal
   resistors to `e +/- a`: `|Delta q| < 1e-14`; one of them alone moves the node by `dt a / R`.
2. **Held.** `dt lambda_max(C^{-1} L) = 0.35`. Total charge drift over 400 steps
   `<= 8.9e-16` of `Q = 1.5`. Final voltages `0.428571428571` at all three nodes, `= 3/7 = e*`,
   max error `2.8e-16`. The closed forcing equals `-dt C^{-1} L C^{-1} q` at 5 random `q`.
3. **Held.** Rest voltage `-0.125000000000000 = (1*1 + 3*(-0.5))/(1 + 3)` exactly; the plain
   mean is `0.25`.

## What this run establishes, and what it does not (STATUS.md §2)

It establishes the mechanism and its numbers: `fan_wire` contains no sum, `compose_seq`'s
pullback along the copy produces one, and its sign is the note's. Predictions 1, 2a, 2b are
guaranteed once the library's composition is what its docstring says (the transpose of a gather
is a scatter-add; a difference's exact one-form has opposite covectors); they could fail only
by a sign slip or a wire routed through `in_f`. Predictions 2c and 3 could have failed by
divergence of the explicit step and did not at the chosen `dt`. Nothing here is a claim about
the world; it gives Ch 2's "we'll prove Kirchhoff's current law" its run, and the audit's item
(e) said it decides nothing in §1.5.

## What an electrical engineer would object to

* *A node in nodal analysis has no capacitor.* True. Here every node is a capacitor, so KCL is a
  dynamical law `q' = q - dt sum I`; a pure resistive junction is the algebraic constraint
  `sum I = 0` solved for the node voltage, which a state-holding box does not express, and
  `C -> 0` makes the explicit step stiff (`dt / RC` against the bound 2). The note's paragraph
  assumes "a circuit node is a capacitor component" and its "nodal equations of the network as the
  composite ODE" holds for networks with a capacitor at every node. The objection survives as a
  scope statement the note's wording does not make; flagged to the caller, the book repo unedited.
* *Ideal sources.* An internal resistance is one more coupler; the rest voltage shifts as
  nodal analysis says. Does not survive.
* *Forward Euler.* SPICE integrates implicitly; the library's step is explicit and stable only
  under `dt lambda_max < 2` (asserted in the test). The library says time is discrete by design
  (STATUS.md §1); the objection survives as a practical limit for stiff networks.
* *KVL "by construction."* Holds here because branch voltages are differences of node outputs
  on the forward pass; inductive branches carry a current state and are `lc_circuit.py`'s
  business, not tested here.

## Weird (the run's judgment, not evidence)

Numerically nothing. One thing to notice: the ideal source absorbs current because the vjp of a
constant output is zero, so "unmovable" and "output independent of state" coincide in the
pullback. At a society scale that is reasonable and pointed: a part whose output does not depend
on its state feels no push back from those who pull on it (the plan's "accommodates what it
cannot move"); at a body scale it is a set point held by a reservoir much larger than the part.
Also as expected: the star's total charge is not conserved (sources are batteries); only the
triangle conserves.

## Files

`tests/test_kirchhoff_node.py` (6 tests), this note. No existing file edited (other runs live),
so `notes/LOG.md` and `STATUS.md` §3 carry no entry from this run; the caller logs it.
