# Seeing and pulling together (`tests/test_monitor_composed.py`)

The final stage of David's monitor test, 2026-09-25: "there can't be a monitor who is able
to see dead parts and also able to pull from far? maybe that's something else to test
separately and then we can put them together and test that too?" Seeing is
`notes/MONITOR-SEEING.md`, pulling is `notes/MONITOR-PULLING.md`; this is **putting them
together**. Success is the turnover test's criterion: the collective's outgoing message
returns to its pre-failure value. `uv run --extra test pytest
tests/test_monitor_composed.py -q`: **6 passed (164 s)**.

The four predictions are quoted verbatim in the module docstring and were written there
before any construction code existed; none was revised after a number was seen.

| | prediction | verdict |
|---|---|---|
| **C1** | the composite restores the whole message, both `h~` and `w~` | **held** in the silent mode, exactly and from any distance; **refuted** in the reporting mode, where the *weight* returns and the *signal* does not |
| **C2** | an alarm driven by the surprise ends delivery with no stopping rule added | **held**: at rest the alarm is `0.0` and the pull `-0.0`, to the last bit, from every distance |
| **C3** | the loop is one arrangement, and the only stipulation left is the supply | **held** as to structure — no schedule, threshold or switch — and **corrected**: which failure *sounds the call* is not the construction's to choose |
| **C4** | the monitor is idle where it is not needed | **held on the message** (`0.0` change) and **refuted on the configuration** (the live pair rests `0.0109` nearer the parent) |

And the largest thing in the stage, which none of the four mentions: **a free monitor
delivers nothing.** The haul runs at `1/(MU_P alpha a*)` steps per unit with
`a* = (w_pi/(1 + w_pi)) Delta`, so at `w_pi = 0` there is no pull at all and the spare
never arrives at any budget. Seeing and pulling are not two capacities that happen to be
composable here; they are one quantity, and the dial `notes/MONITOR-SEEING.md` found sets
the *detectable drop* is the same dial that sets the *delivery rate*.

## The construction

Three pieces, each imported rather than rebuilt.

**The collective and the spare** are `tests/test_turnover.py`'s: a `spatial_arrangement`
parent over redundant contributors, cost `(1/2)(nu - h~)^2` on the weighted signal it
sends onward, value maps frozen at the identity, `tau = 0.8`, `nu = 2.5`, `SIG = WGT = 1`,
placement mobility `MU_P = 0.5`. Two contributors in full health at `+-p`, one dead at
`+p`, a spare in full health at `d`, with `3 kappa(p) + kappa(d) = nu` so that the
*pre-failure* state is a fixed point by construction. Failure modes as there: `reporting`
sends `(c SIG, WGT)`, `silent` sends `(c SIG, c WGT)`.

**The monitor** is `tests/test_monitor_seeing.py`'s, one more child of the same parent,
frozen at `p = 0` — the parent's own reference position, so what it receives is exactly
the outgoing message `m~ = (h~, w~)` — carrying
`k(pi, e) = (1/2)(w(e) - pi)^2 + (w_pi/2)(pi - nu)^2` on the total arriving weight. Its
surprise is `s = w~ - pi`; `w_pi = 1` throughout except where it is swept.

**The doodad** is `notes/MONITOR-PULLING.md`'s route (b) with one change. The composite
publishes two more coordinates at its port (only `out_f` changes; the forcing is
untouched), so the ports are `e = (h~, w~, p_spare, pi)`, and a stateless coupler in the
two-tank idiom returns

    alarm   a(e) = pi - w~ = -s
    F_p(e)  = -alpha a tanh(p / wid),   F_h~ = F_w~ = F_pi = 0,   alpha = 0.02, wid = 0.25

with nothing returning to the monitor. The change is that the pulling test's alarm was a
configuration coordinate of its own, held at `a0 = 1` by a restoring cost; **here the
alarm is the surprise**. The whole loop is
`compose_seq(publishing(watched collective), coupler)`, a closed arrangement whose
configuration is the watched collective's — the coupler is stateless — so nothing was
added to hold the loop together.

Test 0: `Phi` of the composite, `run_closed` and the `lax.scan` of `traj` agree over fifty
steps to `5.6e-17`, and the scan against `run_closed` bit for bit, for the loop and for
the watched collective without the doodad. Every trajectory below is `Phi`'s.

Budgets: `8000` steps where the spare is near, `40000` where it is far; drift is over a
further `500`. Every reach number below is a budget, as `notes/TURNOVER-TEST.md`
established, and every one carries its rate.

## Test 1: the loop has no switch to be off

Four children in full health send `m~ = (nu, nu)` — the signal because that is how `p` was
chosen, the weight because in full health each claims `WGT` under the same kernels. So the
parent's cost is at its minimum, the monitor's expectation is met, its surprise is `0.0`,
the alarm `a = pi - w~` is `0.0` and the pull `-alpha a tanh(p/wid)` is `-0.0`. Every
coordinate of the composite forcing is **exactly** `0.0` and 200 steps reproduce the
initial configuration bit for bit. The loop is quiescent not because anything switched it
off but because the expression it is made of vanishes where nothing is wrong, and that
same expression is what pulls in test 3. The ports read `(2.5, 2.5, 2.0, 2.5)`.

## Test 2: (C1) — twelve settled runs, and the mode decides

| mode | `d` | arrangement | `h~` | `w~` | drift/500 |
|---|---|---|---|---|---|
| silent | 2 | bare | 2.5 | 2.5 | 0 |
| silent | 2 | + monitor | 2.5 | 2.5 | 0 |
| silent | 2 | loop | 2.5 | 2.5 | 0 |
| silent | 20 | bare | 2.0 | 2.0 | 0 |
| silent | 20 | + monitor | 2.0 | 2.0 (surprise open at `-0.25`) | 0 |
| silent | 20 | **loop** | **2.5** | **2.5** | **0** |
| reporting | 2 | bare | 2.5 | 3.305972 | 0 |
| reporting | 2 | + monitor | 2.498156 | 2.503688 | 1.8e-3 |
| reporting | 2 | loop | 2.498108 | 2.503672 | 1.8e-3 |
| reporting | 20 | bare | 2.0 | 2.833333 | 0 |
| reporting | 20 | + monitor | 2.0 | 2.5 | 0 |
| reporting | 20 | loop | 2.0 | 2.5 (spare pushed *out*, to 20.016) | 0 |

**The silent mode, and the cell the composition is for.** A silent part sends `0_M`, so
`h~ = w~` identically and the message is one number. At `d = 2` the bare collective
already restores it — the turnover run's result, and (C4)'s premise — and neither half is
needed. At `d = 20` the parent's own inward force on the spare is
`-(nu - h~)(d/tau) kappa(d)`, about `4e-108`, so the bare collective sits at `2.0` forever, and
the monitor alone sits there too with its surprise open at `-0.25`, seeing the drop and
unable to close it. **The loop closes it exactly**: `h~ = w~ = 2.5`, drift `0`, the spare
at rest at `sqrt(2 tau log 2) = 1.053108`. **(C1) holds in the silent mode, and the
composition earns its keep in exactly the cell where neither half alone does.**

**The reporting mode, where the two coordinates cannot both be met.** A reporting part
keeps claiming `WGT` at a zero signal, so `w~ - h~ = kappa_F` identically: both
coordinates equal `nu` only if `kappa_F = 0`, which no finite placement gives. At `d = 20`
the monitor pushes the dead part out until `w~ = nu` exactly (`kappa_F = 0.5`, the seeing
run's repair) and the signal stays at `2.0`. **(C1) is refuted there, and in the direction
opposite to the turnover run's**: that run restored the signal and left the weight raised
for ever; this one restores the weight and leaves the signal short.

At `d = 2` both coordinates do approach `nu`, but only as the dead part is pushed to
infinity, and the approach is **algebraic**:

| steps | `h~` | `w~` | `kappa_F` | `p_F` |
|---|---|---|---|---|
| 8 000 | 2.498042 | 2.503799 | 5.757e-3 | 2.8726 |
| 32 000 | 2.499097 | 2.501753 | 2.657e-3 | 3.0804 |
| 128 000 | 2.499578 | 2.500820 | 1.242e-3 | 3.2720 |
| 512 000 | 2.499801 | 2.500387 | 5.858e-4 | 3.4508 |

`kappa_F` and both deficits fall by `2.1669`, `2.1391`, `2.1201` for each fourfold in the
run length — about `t^{-0.55}`, with the exponent still drifting at half a million steps.
**No budget closes it.** What the monitor buys in that cell is real and is not what (C1)
claimed: it turns the turnover run's *permanent* mark on the outgoing weight (`3.305972`,
drift `0`) into a decaying one.

## Test 3: (C2) — delivery stops itself, and a held alarm never does

While the live pair is home and the spare is still out, `w~ = 2` and the monitor settles
at `pi* = (w~ + w_pi nu)/(1 + w_pi)`, so the alarm is constant along the haul at

    a* = pi* - w~ = (w_pi/(1 + w_pi)) Delta,   Delta = nu - w~ = 0.5,

which at `w_pi = 1` is `0.25`. The pull is `alpha a*` past the ramp and the placement's
mobility is `MU_P`, so the trip is **linear in the distance** at `1/(MU_P alpha a*) = 400`
steps per unit:

| spare at `d` | arrival step | route (a) at the same `tau`, from `notes/TURNOVER-TEST.md` |
|---|---|---|
| 5 | 791 | 2 000 000 |
| 10 | 2 791 | — |
| 20 | 6 791 | never, from `d = 6`, at 2 000 000 |
| 50 | 18 791 | — |

Successive slopes are `400.0` steps per unit at `atol = 0`, matching the derived rate.

**And it stops.** From every one of the four distances the spare comes to rest at the same
place, `p = 1.053108`, where `kappa = nu - 2` — the share the dead part gave up, and the
resting place the turnover run found for a spare recruited by attention alone. There the
alarm is `0.0` and the pull `-0.0`, both to the last bit. **(C2) holds: the delivery ends
exactly when the surprise closes, and no stopping rule was added.**

The **held** alarm — `notes/MONITOR-PULLING.md`'s doodad wired into this composite
unchanged, `a = 1` — never rests with zero force, at any distance run:

| `d` | `h~ = w~` at rest | spare's placement | live pair | pull still on the spare |
|---|---|---|---|---|
| 5 | 2.5 | below 1e-300 | `+-0.678450` | nonzero |
| 10 | 2.5 | below 1e-260 | `+-0.678450` | nonzero |
| 20 | **2.5200424** | 1.022816 | `0` | **-0.0199893** |

From `d = 20` it holds the collective above its target, as the pulling run found at
`2.529935`, with the residual pull balanced by the parent pushing back. From `d = 5` and
`d = 10` it drags the spare the whole way to the parent and the collective pays by holding
its two live contributors out at `+-0.678450` instead of at `0`. **So the held alarm's
resting place depends on where the spare started, and the driven alarm's does not.**

## Test 4: (C3) — the call, the basin, the delivery, and the one thing stipulated

**The call is the surprise, and it does not sound for every failure.** At the failed state
with the spare still out:

| mode | `h~` | `w~` | surprise `s` | alarm | pull on the spare at step 0 | detected at `theta = 0.05`? |
|---|---|---|---|---|---|---|
| silent | 1.6666667 | 1.6666667 | **-0.8333333** | +0.8333333 | -0.0166667 | yes |
| reporting | 1.6666667 | 2.5000000 | **0.0** | 0.0 | -0.0 | no |

A silent part takes its weight with it, so `s = -kappa(p)`, the dead part's whole share,
and the doodad is already pulling at step `0`. A reporting part keeps claiming `WGT`, so
`w~ = nu` exactly, the surprise is `0.0` to the last bit, and **the doodad never goes.**
This is the correction (C3) needs: the loop composes, but **the failure that sounds the
call is not the failure the monitor can repair.** The seeing run's repair works against a
*reporting* failure and cannot touch a silent one; the delivery works against a *silent*
failure and is never called by a reporting one. In this construction the two halves never
need each other for the same failure, and test 2's reporting rows are what that costs.

**The basin is a crossing, not a chosen radius.** From `d = 20` the doodad carries the
spare for 6671 steps while the parent's own force on it is below the pull — about `4e-108` at
the start. At step **6671**, placement **3.081955**, `kappa = 0.0026412`, the parent's
force overtakes the doodad's, and the last 120 steps are the collective's own attention:
arrival at 6791. Nothing chose that number; it is where two forces cross. That is "falling
into the basin," measured.

**The delivery rate is the prior precision**, and a free monitor has nothing to pull with:

| `w_pi` | `a* = (w_pi/(1+w_pi)) Delta` | derived rate, steps/unit | arrival from `d = 5` | from `d = 10` | measured slope |
|---|---|---|---|---|---|
| 0 | 0 | — | **none in 40 000** | **none** | — |
| 0.25 | 0.100000 | 1000 | 1 726 | 6 726 | 1000.0 |
| 0.5 | 0.166667 | 600 | 1 121 | 4 121 | 600.0 |
| 1 | 0.250000 | 400 | 791 | 2 791 | 400.0 |
| 2 | 0.333333 | 300 | 613 | 2 113 | 300.0 |
| 4 | 0.400000 | 250 | 520 | 1 770 | 250.0 |

Every measured slope matches the derived rate at `atol = 0`. **At `w_pi = 0` the spare
never arrives at any budget**, because a free prediction's settled surprise is exactly zero
(the seeing run's closed form `s* = -(w_pi/(1+w_pi)) Delta`) and the alarm is that surprise.
This was not predicted and it is the strongest result in the stage. Note the two roles the
one dial plays are not the same number: the seeing run's detection floor is
`w_pi > theta/(kappa - theta) = 0.0638`, while delivery needs only `w_pi > 0` and is merely
slower below it. **A monitor too coarse to report the drop still delivers against it.**

**What is stipulated.** The coupler returns nothing at the monitor's port, `F_pi = 0`, so

    (d omega)_{pi p} = -alpha tanh(p / wid),   (d omega)_{w~ p} = +alpha tanh(p / wid),

the only nonzero entries — `-0.0075990` and `+0.0075990` at `p = 0.1`, `-0.02` and `+0.02`
to sixteen digits once the ramp saturates — and the **composite's** own antisymmetric part
on `Q` carries `-0.02` at the monitor's expectation and the spare's placement, with
`max|d omega| = alpha` exactly. By the reciprocity lemma (§5.5) the pull is not `dU` for
any `U`: inside the book's category, whose forcing need not be exact (STATUS.md §1),
outside the attention operad. Nothing depletes: the work delivered over the trip from
`d = 20` is `0.0916588`, finite **because the job ends**, not because anything was charged
for it. **The supply is stipulated away here exactly as it was in stage B**, and (C3)'s
"the only stipulation left is the supply" holds in that sense and in no stronger one.

This bears directly on the plan's claim that *a collective that can replace its parts must
pay for a monitor, and that monitor cannot be energy-derived*. The run supports the second
half and does not test the first: the recruiting coupler is measurably non-exact, so it is
not the differential of any cost and cannot be read as a descent of energy; and its
inexhaustibility is put in by hand, not derived. What the run does *not* show is that a
collective must pay — it shows one collective that cannot recover without it (silent
failure, spare at `d = 20`) and one that recovers fine without it (silent failure, spare at
`d = 2`). Whether the first case is the normal one is not a question this code answers.

## Test 5: (C4) — nothing on the message, a hundredth on the placements

The silent failure at `d = 2`, the case that repairs itself, in four arrangements:

| | `h~ - nu` | `w~ - nu` | the live pair's placement |
|---|---|---|---|
| bare | 0.0 | 0.0 | `+-0.01932436` |
| + free monitor (`w_pi = 0`) | -4.4e-16 | -4.4e-16 | `+-0.01019096` |
| + held monitor (`w_pi = 1`) | 0.0 | 0.0 | `+-0.00845884` |
| loop | 0.0 | 0.0 | `+-0.00916135` |

**On the message (C4) holds exactly** — which is the criterion this whole stage is
measured by — with `0` change in the two held cases and one unit in the last place with a
free prediction. **On the configuration it does not**: the live pair rests `0.0109` nearer
the parent with the monitor than without, and the spare correspondingly further out. The
seeing run's warning about a five-step transient materializes; it costs nothing here for a
reason the turnover run gave, that the fixed set is the whole level manifold
`sum_j kappa_j = nu`, so the transient moves the collective *along* it. **A collective
whose fixed point were isolated would pay for the same transient**, and this construction
does not exhibit one.

## Does the loop compose as one arrangement, and what had to be added

It composes. `compose_seq(publishing(watched collective), coupler)` is one closed
arrangement built from the library's own pieces; the coupler is stateless, so the loop's
configuration is the collective's unchanged; nothing schedules, switches or thresholds
anything. Three of David's four pieces are present and each is a quantity in the forcing
rather than a stage of a procedure: the **call** is the monitor's surprise, the **basin**
is the placement at which the parent's attention overtakes the pull, the **delivery** is
the pull, and it ends because the call it is made of closes.

The fourth piece, **running a plan to go and get X**, is not built and this stage was
scoped to leave it out. That scoping is the agent's proposal, not David's word, so the
honest report on it: the three pieces here compose *without* it, and the reason they can is
that there is nothing to plan — the spare's route is a straight descent along one placement
coordinate in a space with no obstacles. What plan-execution would add is a route that is
not a descent, and nothing in this run bears on whether that can be built. **So the answer
is: these three compose without the fourth, and that is not evidence that the fourth is
unnecessary.**

The detection threshold `theta = 0.05` appears nowhere in the dynamics; it is used only to
*report* whether a surprise counts as detected, as in stage A. No number in the composite
had to be chosen to make the loop start or stop.

## What is stipulated, and what is derived

**Stipulated.** The doodad's pull law — the bounded `tanh` ramp, `alpha = 0.02`,
`wid = 0.25` — inherited from stage B, and that nothing returns to the monitor, which is
what makes it non-exact and what leaves the supply unpaid. The alarm being `pi - w~`
rather than some other function of the surprise: the sign is forced (a shortfall must pull
inward) but the linearity is not. The monitor's expectation `pi_0 = nu`, its position `0`,
that position frozen, its mobility `1/(2(1 + w_pi))`, its sending `0_M` — all stage A's.
The collective, the cost on the weighted signal, the failure modes and the health gating —
all the turnover test's, where they are stipulations too. The failure as a **state**, the
dead part's health held at `0` from step `0` and immobile, so that every force read is
attentional: this is why the bare reporting cell reads `w~ = 3.305972` here where the
turnover run, which ran the failure as a process, read `3.3742`. The budgets, `8000` and
`40000`. The detection threshold `theta = 0.05`, for reporting only.

**Derived.** That the pre-failure state is an exact fixed point of the whole loop, forcing
`0.0`. That the ports carry `(h~, w~, p_spare, pi)`. `w~ - h~ = kappa_F` in the reporting
mode, and hence that both coordinates reach `nu` only at `kappa_F = 0`. The haul alarm
`a* = (w_pi/(1+w_pi)) Delta` and the rate `1/(MU_P alpha a*)`, matched at `atol = 0` at
five precisions. The resting place `sqrt(2 tau log 2) = 1.053108` and `kappa = nu - 2`.
The call's `s = -kappa(p) = -0.8333333` in the silent mode and `s = 0.0` in the reporting
one. `(d omega)_{pi p} = -alpha tanh(p/wid)` and `(d omega)_{w~ p} = +alpha tanh(p/wid)`,
and the composite's `max|d omega| = alpha`. The algebraic rate `t^{-0.55}` of test 2's
reporting cell, which is measured and not derived, and whose exponent is still drifting.

## What the runs support, and what they do not

They support that seeing and pulling compose into one arrangement, that the composite
restores the whole outgoing message where a silent failure leaves a spare out of the
parent's reach, that the delivery ends by itself, and that the prior precision which makes
a monitor able to see is the same quantity that makes it able to pull.

Five things they do not support.

First, **(C1) does not hold in general.** In the reporting mode the composite restores the
weight and not the signal; at close range it approaches both but only at `t^{-0.55}`.
"The outgoing message returns to its pre-failure value" is true of this construction for
one failure mode and false for the other, and the turnover run's version of the same
sentence was true for the *other* mode. Which coordinate survives is decided by which
failure is run, and nothing in the mathematics prefers one.

Second, **the two halves never help each other for one failure.** The monitor repairs a
reporting failure, which never sounds the call; the doodad delivers against a silent
failure, which the monitor cannot repair. So the composite is not a collective that sees a
dead part *and* pulls from far in one event; it is a collective with two mechanisms, each
idle in the other's case. Whether a failure mode exists that calls both is not tested here,
and a construction in which one did would be the obvious next question.

Third, **(C4) holds on the message only because the fixed set is a manifold.** The monitor
moved every placement, and it was free only because redundancy makes the resting place a
level set. Nothing here shows that a monitor is cheap in general.

Fourth, **the supply is stipulated away.** The monitor's own state is not charged for the
work it delivers, exactly as in stage B; `notes/MONITOR-PULLING.md` measured what charging
it costs (a genuine boundary at `p = 50.173`, from that particular exact completion), and
the billed variant is not rerun here. So this stage supports "the recruiting coupler is not
exact" and not "a monitor must be paid for out of something."

Fifth, and standing behind all of it, STATUS.md §2's warning applies in full: any smooth
vector field can be written as a forcing, so a construction that behaves as wanted proves
nothing about the class of constructions that do. Every verdict above is the boundary of a
region — one failure mode, one distance, one `w_pi`, one budget — and not a claim that
arrangements tend to fall in it.

## Adapted from the brief

- The brief asked for "monitor plus redundancy plus spare" against the turnover criterion.
  Both failure modes are run and both verdicts are reported, rather than a mode chosen;
  they disagree, and the disagreement is the result.
- The **held**-alarm rows are stage B's doodad wired into this composite unchanged. They
  are run because (C2) is a comparison and needs the thing it is compared to.
- The alarm is the surprise itself rather than a configuration coordinate driven by it.
  A separate alarm coordinate with its own relaxation would add a lag and a mobility to
  choose; taking the surprise directly is the reading of "driven by the surprise" that adds
  no parameter, and it is what makes test 1's exact zero and test 3's exact stop possible.
- The brief's scoping of plan-execution out of this stage is followed, and the section
  above says plainly what that leaves unanswered rather than treating it as settled.
- `theta = 0.05` is carried from stage A for reporting only. No threshold enters the
  dynamics of the loop.
