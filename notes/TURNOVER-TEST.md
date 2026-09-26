# The turnover test (`tests/test_turnover.py`)

Six runs of the Ship-of-Theseus construction of `AGENTDOCPLAN.md` §5.9 item 13 in the book
repository, from David's design: a parent over redundant contributors, its cost on an
aggregate of what they send, one contributor's health driven to zero, a spare off to the
side. `uv run --extra test pytest tests/test_turnover.py -q`: **6 passed (52 s)**. The four
predictions are quoted verbatim in the module docstring and were written there before any
construction code existed: **(1)** with no spare the collective settles at a degraded
outgoing value, and with a spare recruited it returns to the pre-failure one; **(2)** a
failed part is pushed out only if it keeps claiming attention, and one that goes silent is
forgotten for free; **(3)** a spare is recruited only from within the pull region; **(4)**
if (1)–(3) hold, Ship of Theseus is modeled for this case.

Verdicts: **(1) holds**, inside a redundancy window. **(2) refuted** for the cost this
construction is required to carry, and **holds** for the mean readout. **(3) refuted** in
the direction opposite to the one (4) worried about: the spare is *always* recruited, given
time. **(4)**: the mathematics has failure, forgetting and replacement, so Ship of Theseus
is modeled for this case — but only the *signal* of the outgoing message returns, and "the
pull region" is a run length, not a boundary.

## The construction

A contributor is a component `I -> B` (`tests/test_member.py`'s idiom) whose parameter is
the configuration coalgebra on its health `c`, carrying its own health cost
`(lam/2)(c - target)^2` as the exact forcing `-dk`, and sending a message gated by `c`:

    reporting   m = (c SIG, WGT)      health gates the signal, the weight is kept
    silent      m = (c SIG, c WGT)    health gates both, so c = 0 sends 0_M
    corrupting  m = ((2c - 1) SIG, WGT)   c = 1 signals +SIG, c = 0 signals -SIG

The first two are the brief's two modes; the third is mine, run only to locate the
mechanism in test 3. The parent is `spatial_arrangement(n, 1, 1, tau)` — one scalar `tau`
for every pair, as eq:spatial-transfers gives it — with the value maps frozen at the
identity, so the parent's only adaptation is attentional: where it holds each contributor.
What it sends onward is `sigma_a` (eq:maps), written out and checked against the library:

    m~ = (h~, w~) = (sum_j kappa_j V_j h_j,  sum_j kappa_j w_j),   kappa_j = exp(-p_j^2/2 tau)

since `l_0 = 0_P`. The parent's cost is on the **weighted signal**, `k = (1/2)(nu - h~)^2`,
which is the brief's constraint and keeps `k(q, -)` of degree two in `D_n`; the **mean**
variant `k = (1/2)(nu - h~/w~)^2` is run only where prediction (2) needs a readout that
sees weight. The composite is `compose_seq(tensor_arrangements([contributors]),
parent.arr())` with configuration `(c_1..c_n, p_1..p_n, V_1..V_n)`.

Throughout: `tau = 0.8`, `SIG = WGT = 1`, `nu = 2.5`, `lam = 50`, health mobility `0.5/lam`
(so health halves each step at any stiffness), position mobility `0.5`, 8000 steps per
settling run, the spare at `d = 2`. The three contributors start at the position that makes
the collective send exactly `nu` — `3 kappa(p) + kappa(d) = nu`, so `p = 0.5874783`,
`kappa(p) = 0.8059717` with the spare and `p = 0.5401060`, `kappa = 0.8333333` without —
so the pre-failure state is a fixed point by construction and the run has only to confirm
it. Settling is read off the **motion** (how far a further 500 steps move it), never off
`max|xi|`: a frozen health carries a nonzero forcing forever without moving. Test 0 checks
that `Phi` of the composite, `run_closed` and the `fori_loop` of `settle` agree over fifty
steps to `2.2e-16`, one unit in the last place — the mutual-need test got exact equality;
here the jitted composite reassociates one sum of kernel terms.

## Test 1: the baseline

`sigma_a` equals the hand-derived `outgoing` bit for bit. At the chosen positions the
signal sent onward is `2.5` to `1e-14`, every coordinate of the composite forcing is below
`1e-14`, and 8000 steps move the configuration by exactly `0`. The fixed point is **not
isolated**: the whole level set `sum_j kappa_j = nu` is fixed, which is what redundancy
means here, and it is why the settled split among the parts in test 2 is not determined.

## Test 2: prediction (1) — degrade, then return

The forcing on the failed part's health is `-lam c + (nu - h~) k_F SIG`: **the collective's
own deficit props the failing part up**, so `c` settles at `(nu - h~) k_F / lam`, not at
`0`, unless the deficit itself vanishes. With no spare everyone is pulled to the parent
(`kappa -> 1`) and `h~ = 2 + c` with `c = (nu - 2)/(1 + lam)`.

| case | h~ sent | w~ sent | failed part's health | kernels (C0, C1, failed, spare) | drift/500 steps |
|---|---|---|---|---|---|
| pre-failure | 2.5 | 2.5 | 1 | (0.80597, 0.80597, 0.80597, 0.08208) | 0 |
| failed, no spare | **2.0098039** | 3.000000 | 0.00980392 | (1, 1, 1, —) | 1.3e-11 |
| failed, spare at 2 | **2.5000000** | 3.374244 | 0 | (0.99971, 0.99971, 0.87424, 0.50057) | 0 |

Both closed forms hold to `1e-9`: `c = 0.5/51 = 0.0098039`, `h~ = 2.0098039`. **Prediction
(1) holds** on the signal: degraded to 80% of the target without a spare, dead-on `2.5`
with one, and the spare's kernel rose from `0.0821` to `0.5006` — it came in from `p = 2`
to `p = 1.0524`, and only as far as it had to. Once the deficit closes, the pull on the
failed part's health vanishes and `c -> 0` exactly: **the dead part dies only because the
spare replaced it.** Its kernel then freezes at `0.87424` — a dead part contributes
nothing, so it feels no force, and it is neither pushed out nor pulled in.

**But only the signal returns.** The collective sends a message `m~ = (h~, w~)`, and in the
reporting mode `w~` goes from `2.5` to `3.3742`: the dead part still claims its weight and
the spare has added its own, so the *mean* of what the collective sends falls from `1` to
`0.7409` and stays there. In the silent mode `h_j = w_j` for every contributor, so
`h~ = w~` identically and the whole message returns to `(2.5, 2.5)` to `1e-9`. **A failed
part that keeps claiming attention leaves a permanent mark on what the collective sends,
even after the signal is restored.** Which reading of "returns to its pre-failure value" is
taken therefore decides prediction (1) for the reporting mode; the signal is the reading
the parent's cost is about, and it is the one used above.

## Test 3: prediction (2) — refuted for the required cost, and why

The positional forcing at the failed state (health held at `0`, live parts and spare in
full health at the pre-failure positions), on `(C0, C1, failed, spare)`. `C0` sits at
`+p`, `C1` at `-p`, the failed part at `+p`, the spare at `+2`; a **positive** force on the
failed part moves it **away** from the parent.

| cost | failure mode | force on (C0, C1, **failed**, spare) | h~ | w~ | mean |
|---|---|---|---|---|---|
| weighted signal | reporting | (−0.477025, +0.477025, **0**, −0.165396) | 1.694028 | 2.5 | 0.677611 |
| weighted signal | silent | (−0.477025, +0.477025, **0**, −0.165396) | 1.694028 | 1.694028 | 1.000000 |
| weighted signal | corrupting | (−0.954051, +0.954051, **+0.954051**, −0.330791) | 0.888057 | 2.5 | 0.355223 |
| mean readout | reporting | (−0.024606, +0.024606, **+0.051718**, −0.008532) | 1.694028 | 2.5 | 0.677611 |
| mean readout | silent | (0, 0, **0**, 0) | 1.694028 | 1.694028 | 1.000000 |
| mean readout | corrupting | (−0.098424, +0.098424, **+0.206872**, −0.034126) | 0.888057 | 2.5 | 0.355223 |

Under the cost the construction is required to carry, `xi_F = (nu - h~) v_F dk_F/dp_F` with
`v_F` what the part signals. Reporting and silent both give `v_F = 0`, so the force is
**exactly `0.0` in both**, and the two rows of the table are identical to the last bit —
the weight never enters `h~` at all. **Prediction (2) is refuted here: a contribution that
is merely absent exerts no pull either way, whatever weight it claims.** The corrupting
mode, `v_F = -SIG`, is pushed out with force `+0.9540505`, matching
`-(nu - h~) SIG dk_F/dp_F` to `1e-12`. So **what decides the push-out under a cost on the
weighted signal is not whether the part claims weight but whether its contribution is wrong
rather than absent.**

Under the mean readout, `d(h~/w~)/dk_F = (v_F - h~/w~) w_F / w~`, and prediction (2) holds
exactly as stated. Reporting keeps `w_F = 1` while `v_F = 0`, so the failed part drags the
readout and descent pushes it out: force `+0.0517180`, matching the derived expression to
`1e-10`. Silent has `w_F = 0`, so the readout does not depend on `k_F` at all — and, the
readout being already exactly `nu`, the **whole** composite forcing is `0.0` and *nothing in
the collective moves*: 300 steps reproduce the initial configuration bit for bit. Forgotten
for free, in the strongest sense the model allows. The reporting run under the mean readout:

| step | p of the failed part | its kernel | mean sent | p of the spare |
|---|---|---|---|---|
| 0 | 0.587478 | 0.805972 | 0.677611 | 2.000000 |
| 1 | 0.613337 | 0.790481 | 0.683791 | 1.995734 |
| 10 | 0.854488 | 0.633596 | 0.740948 | 1.962968 |
| 50 | 1.547740 | 0.223759 | 0.896680 | 1.903448 |
| 100 | 1.837165 | 0.121301 | 0.941992 | 1.885805 |
| 300 | 2.173481 | 0.052208 | 0.974487 | 1.868194 |

Away at every step, the readout rising at every step and never reaching `nu`: the mean
readout has no settled state here, exactly as `notes/MEMBER-TEST.md` found for the reader.

**So prediction (2) names a mechanism that lives in the mean readout, and the mean readout
is outside the affine-quadratic family.** The brief's constraint and the prediction pull
against each other; both halves of the result are above, and neither was worked around.

## Test 4: prediction (3) — the pull region is a time budget, not a boundary

The force on a spare at `d`, read off the composite at the failed state, is
`(nu - h~) v dk_d/dp = -kappa(p) (d/tau) kappa(d)`: strictly inward for every finite `d`,
peaking near `sqrt(tau) = 0.8944` and then Gaussian. The deficit it multiplies does **not**
go to zero while the spare stays out — the live pair can only reach `kappa = 1`, leaving
`nu - 2 = 0.5`. So the spare always eventually arrives. At 8000 steps:

| spare at d | kappa(d) | inward force at the start | h~ sent | settled distance | settled kernel |
|---|---|---|---|---|---|
| 0.5 | 8.5535e-01 | −0.29307 | 2.5000000 | 0.2658 | 0.956801 |
| 1.0 | 5.3526e-01 | −0.43819 | 2.5000000 | 0.6450 | 0.771030 |
| 1.5 | 2.4506e-01 | −0.34537 | 2.5000000 | 0.9710 | 0.554734 |
| 2.0 | 8.2085e-02 | −0.16540 | 2.5000000 | 1.0524 | 0.500467 |
| 2.5 | 2.0116e-02 | −0.051963 | 2.5000000 | 1.0531 | 0.500000 |
| 3.0 | 3.6066e-03 | −0.011254 | 2.5000000 | 1.0531 | 0.500000 |
| 3.5 | 4.7308e-04 | −0.0017244 | 2.5000000 | 1.0531 | 0.500000 |
| 4.0 | 4.5400e-05 | −0.00018916 | 2.5000000 | 1.0531 | 0.500000 |
| 4.5 | 3.1876e-06 | −1.4942e-05 | 2.0000041 | 4.4601 | 3.98e-06 |
| 5.0 | 1.6374e-07 | −8.5280e-07 | 2.0000002 | 4.9962 | 1.67e-07 |

Where the spare arrives after the live pair has been pulled home, the settled state has a
closed form: `kappa_spare = nu - 2 = 0.5` and `p = sqrt(2 tau log 2) = 1.0531` (the member
test's own `d`). Where it arrives first, the run lands elsewhere on the fixed manifold
`sum_j kappa_j = nu` — `0.500467` at `d = 2`.

The threshold is a budget. One distance, `d = 5`, five run lengths:

| steps | h~ sent | spare's distance |
|---|---|---|
| 8 000 | 2.0000002 | 4.9962 |
| 32 000 | 2.0000002 | 4.9847 |
| 128 000 | 2.0000002 | 4.9390 |
| 512 000 | 2.0000014 | 4.6511 |
| 2 000 000 | **2.5000000** | 1.0531 |

At `d = 6` it is still at `6.0000` after 2 000 000 steps. The kernel and its gradient vanish
*exactly* only at floating-point underflow: bisection puts the kernel's last nonzero value
at **`d = 33.666516`**, which is `sqrt(-2 tau log(DBL_MIN))` to `1e-9` (jax flushes
subnormals, so the boundary is the smallest *normal* double), and the composite force's at
**`d = 33.650956`**, slightly nearer, because the product `(nu - h~)(d/tau) kappa(d)`
underflows before `kappa` does. **So prediction (3)'s stated mechanism is real but sits at
thirty-seven times the kernel's own scale.** "A collective recruits only from what is
already near it" is true of any fixed budget and false of the model.

## Test 5: prediction (4), and which verdicts the parameters decided

**(a) Prediction (1) lives in `2 < nu < 3`** — the target against the number of live
contributions. Below `2` the two survivors already meet it and nothing degrades, so there
is no turnover to model; above `3` the survivors plus the spare cannot meet it.

| nu | h~ sent, no spare | h~ sent, spare at 2 | |
|---|---|---|---|
| 1.5 | 1.500000000 | 1.500000000 | no degradation at all |
| 2.0 | 1.999977777 | 2.000000000 | degenerate: still moving |
| 2.2 | 2.003921569 | 2.200000000 | (1) holds |
| 2.5 | 2.009803922 | 2.500000000 | (1) holds |
| 2.9 | 2.017647059 | 2.900000000 | (1) holds |

At `nu = 2` exactly — the survivors' own capacity — the deficit is quadratic in the
positions, so the force is cubic and the approach is **algebraic**: `1.0e-4` short at 8000
steps, `2.5e-5` at 32 000, the deficit falling as `1/t` (the ratio of the two deficits is
`4.000` for a fourfold run, to `1e-3`). No run length settles it, and a run that reported
`h~ = 2.0000` there would be reporting its own budget.

**(b) `lam` decides how dead a dead part is.** With no spare the failed part's settled
health is `(nu - 2)/(1 + lam)`:

| lam | settled health | h~ sent |
|---|---|---|
| 0.5 | 0.333333 | 2.333333 |
| 5 | 0.083333 | 2.083333 |
| 50 | 0.009804 | 2.009804 |
| 500 | 0.000998 | 2.000998 |

At `lam = 0.5` the collective's own deficit holds the failing part at a third of full health
and the degradation is slight. **The failure being a failure is a parameter choice.**

## What the runs support, and what they do not

They support the shape of the construction: a parent whose cost is on an aggregate of what
its redundant contributors send does degrade and not die when one fails, does recruit a
spare, and does return its outgoing signal to the pre-failure value exactly, with the
failed part's health going to zero precisely because the replacement closed the deficit.
For that reading of "the outgoing value", **Ship of Theseus is modeled for this case**, and
prediction (4)'s worry — failure and forgetting but not replacement — does not materialize.

Four things they do not support, three of them corrections to the predictions.

First, **prediction (2) is refuted for the cost the construction is required to carry.** A
cost on the weighted signal cannot see weight, so a part that goes silent and a part that
keeps reporting a zero signal are literally the same input to it, and neither is pushed out.
The mechanism the prediction describes exists, and I ran it: it needs the mean readout,
which leaves the affine-quadratic family and, as `notes/MEMBER-TEST.md` already found, has
no settled state. The choice is real and is not resolvable inside this test: **the aggregate
that makes turnover settle is not the aggregate that makes a failed part get pushed out.**

Second, **prediction (3) fails in the opposite direction.** Recruitment has no radius in
this model, only a rate; the only genuine boundary is floating-point underflow at
`d = 33.67`. If the book wants "a collective recruits only from what is already near it",
the mathematics does not yet supply it — a real boundary would need something the model does
not have, a cost of moving, a finite lifetime, or a kernel with compact support.

Third, **only the signal returns.** In the reporting mode the outgoing *message* does not:
its weight rises from `2.5` to `3.3742` and stays there. Read as a message, or as a mean,
the collective's form is not the one it had.

Fourth, and standing behind all of it, STATUS.md §2's warning applies in full: any smooth
vector field can be written as a forcing, so a construction that behaves as wanted proves
nothing about the class of constructions that do. Every verdict above is the boundary of a
region — `2 < nu < 3`, `lam` large enough, a run long enough — and not a claim that
arrangements tend to fall in it. The health cost `(lam/2)(c - target)^2` is a
**stipulation** of what failure is, not a derivation of one, and the interesting thing the
runs found about it was unpredicted: the collective's own deficit props a failing part up,
at `(nu - h~) k_F / lam`, so a part only fully dies once it has been replaced.

## Adapted from the brief

- "Its cost on an aggregate of what they send" is read as the **weighted signal** `h~` of
  the message the parent sends onward, which is the brief's own constraint (a cost on the
  mean readout leaves the affine-quadratic family). Prediction (2) forces the mean readout,
  as the brief anticipated it might; rather than work around it, both costs are run and both
  verdicts reported.
- "One part's health is driven to zero" is a process inside the dynamics: the failing
  contributor's own cost pulls its health from `1` toward `0` at `0.5/lam` mobility. Where a
  force is read at one configuration rather than along a run (test 3, test 4), the failure is
  a *state* instead — health held at `0`, immobile — so that the force read is attentional
  alone.
- The **corrupting** mode is mine, not the brief's. It is run only in test 3, to locate what
  the push-out actually turns on once the two predicted modes came out identical.
- The spare is an ordinary contributor in full health at distance `d`; "a spare recruited"
  is its kernel rising, and "no spare" is a genuine three-ary composite with no spare box at
  all, not a masked four-ary run.
- The value maps `V_j` are frozen at the identity, so every verdict here is about the
  parent's *attention* and not about it relearning what a contributor's signal means.
- `tau` is one scalar for the whole arrangement, as eq:spatial-transfers gives it; no part
  has longer reach than its peers.
