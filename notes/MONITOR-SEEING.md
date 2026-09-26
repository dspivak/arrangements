# The monitor test, stage A: seeing (`tests/test_monitor_seeing.py`)

Five runs of a monitor watching the Ship-of-Theseus collective of `tests/test_turnover.py`,
from David's design, 2026-09-25: "there can't be a monitor who is able to see dead parts and
also able to pull from far? maybe that's something else to test separately and then we can
put them together and test that too?" This is **seeing** alone; pulling is stage B and
composing the two is a later stage. Nothing here is about reach.
`uv run --extra test pytest tests/test_monitor_seeing.py -q`: **5 passed (74 s)**.

The prediction is quoted verbatim in the module docstring and was written into the file
before any construction code existed:

> A silent failure is invisible to listening alone, so detection needs a **held**
> expectation of what should be arriving — the curious-reader result that nothing surprises
> a free prediction and surprise needs a view that lags what is heard. So a monitor with a
> prior on the total weight detects the drop and one without does not, and the prior
> precision `w_pi` sets how large a drop it takes. **Able to fail:** if listening alone
> suffices, the curious-reader mechanism is not needed here.

Verdicts, in the prediction's own three parts.

| the prediction's clause | weighted-signal cost (required) | mean readout |
|---|---|---|
| "a silent failure is invisible to listening alone" | **refuted**: force `−0.4688420` on each live part, the collective moves | **holds**: the collective's whole forcing is `0.0` |
| "with a prior detects, without does not" | **holds**: `−0.25` against `0` at a threshold of `0.05` | **holds**: `−0.25` against `0` |
| "`w_pi` sets how large a drop it takes" | **holds**, exactly: `Delta_min = theta (1 + w_pi)/w_pi` | same (the sweep is cost-independent) |
| **Able to fail** | **fires**: listening alone already sees it | does not fire: nothing else sees it |

And one thing the prediction does not mention, which is the largest effect in the runs:
**the monitor is not a detector, it is an intervention.** Its held expectation is a cost, so
its surprise is a force on everything its input depends on. Under the required cost it
repairs the one failure `notes/TURNOVER-TEST.md` found that cost cannot push out.

## The construction

The collective is imported from `tests/test_turnover.py`, not copied, so it is literally the
one run there: three contributors, each gating what it sends by a health `c`, under a
`spatial_arrangement` parent whose cost is either the **weighted signal** `(1/2)(nu − h~)^2`
(the affine-quadratic one the paper requires) or the **mean readout** `(1/2)(nu_bar − h~/w~)^2`.
Failure modes as there: `reporting` sends `(c SIG, WGT)`, `silent` sends `(c SIG, c WGT)`.

The **monitor** is one more child of the same parent. It sends `0_M`, so it adds nothing to
either coordinate of what the collective sends onward. It sits at `p = 0`, the parent's own
reference position, and its position is frozen; since `l_0 = 0_P` in eq:spatial-transfers,
its kernel to contributor `j` is the parent's own kernel `kappa_j`, so what it receives is
exactly the outgoing message `m~ = (h~, w~)` — checked bit for bit against `a.sigma`. It
carries one configuration coordinate `pi`, its held expectation of the **total arriving
weight**, and the curious reader's cost with the weight in place of the mean readout:

    k_mon(pi, e) = (1/2)(w(e) − pi)^2 + (w_pi/2)(pi − pi_0)^2

Both terms are quadratic in the received message, so **the monitor stays inside the
affine-quadratic family under either parent cost**; only the parent's mean readout leaves it.
Its **surprise** is `s = w~ − pi`, which is the whole force its error term puts on `pi`.

Throughout: `tau = 0.8`, `SIG = WGT = 1`, `nu = 2.5`, three contributors at `±p` with
`3 kappa(p) = nu`, so `p = 0.5401060` and `kappa(p) = 5/6 = 0.8333333`; `pi_0 = nu = 2.5`,
the pre-failure total weight; position mobility `0.5`; 4000 steps per settling run. The
monitor's mobility is `1/(2(1 + w_pi))`, half the descent-lemma step for the exact Hessian
`1 + w_pi` of its own cost, so **every run contracts by exactly `1/2` per step and the sweep
over `w_pi` is not also a sweep over the step**. Test 0 checks `Phi` of the composite,
`run_closed` and the `fori_loop` of `settle` agree over fifty steps to `1e-15`.

## Test 1: the monitor hears `m~`, and acts exactly when it is surprised

What it hears is `sigma_a(q, m)` to the last bit, and pre-failure `m~ = (2.5, 2.5)`, the
monitor at `pi = pi_0` is unsurprised, the whole composite forcing is below `1e-14` and 200
steps move nothing at all.

What it *does* was not anticipated by me and is not in the prediction. The forcing the
monitor adds to the composite is `−s grad w~`, so on contributor `j`

    health:    −s kappa_j dw_j/dc_j    (WGT in the silent mode, 0 in the reporting mode)
    position:  −s w_j dkappa_j/dp_j = s w_j (p_j/tau) kappa_j

At the silent failed state `s = −kappa(p) = −0.8333333`, and the added forces are

| | force without the monitor | force added by the monitor |
|---|---|---|
| dead part's health | `+0.6944444` (weighted signal) / `0` (mean) | `+0.6944444` |
| each live part's position | `∓0.4688420` (weighted signal) / `0` (mean) | `∓0.4688420` |

Under the weighted-signal cost the monitor **exactly doubles** both: it doubles the prop-up
that `notes/TURNOVER-TEST.md` found keeps a failing part alive, and it doubles the parent's
inward pull on the survivors. Under the mean readout, where the parent's forcing is
identically zero, these are the *only* forces in the composite. In the reporting mode
`s = 0` at this state and the monitor is inert to the last bit.

**So the monitor is passive exactly when it is unsurprised, and seeing cannot be separated
from intervening.** This is a correction to an assumption of my own construction — that a
part which sends `0_M` is an observer — not to anything David or the prediction said.

## Test 2: listening alone, which is the **Able to fail** measurement

At the silent failed state the collective sends `h~ = 2 kappa(p) = 1.6666667`, so the
parent's deficit is `nu − h~ = kappa(p)`, the dead part's whole share.

| cost | force on a live part's position | does the collective move in 300 steps? |
|---|---|---|
| weighted signal | `−0.4688420` (`= −kappa(p) SIG (p/tau) kappa(p)`, to `1e-12`) | yes, by `> 0.5` |
| mean readout | `0.0` exactly, and every other coordinate too | no, bit for bit |

**A silent failure is not invisible to a cost on the weighted signal.** The premise of the
prediction is false for the cost the construction is required to carry, and true for the
mean readout. The reason is the one the turnover run found: the weighted signal cannot see
weight, but a silent part's signal falls too, so the deficit is there to be felt.

## Test 3: `w_pi` against the size of the drop

Here every health and every position is frozen — **stipulated**, and it is what suppresses
test 1's back-action, so that the monitor only watches. The only moving coordinate is `pi`,
`w~` is a constant of the run, and the step is affine:

    pi* = (w~ + w_pi pi_0)/(1 + w_pi),   s* = w~ − pi* = −(w_pi/(1 + w_pi)) Delta,
    s_t = s* − 2^{−t}(Delta + s*),       Delta = pi_0 − w~ = (1 − c) kappa(p)

all confirmed to `1e-13` over sixty steps at every cell below. **At `w_pi = 0`, `s* = 0`
exactly: a free prediction is not surprised.** The settled surprise, with the detection
threshold `theta = 0.05` (stipulated) marked in bold:

| `w_pi` | `Delta_min` | `c=1` (Δ=0) | `0.9` (0.08333) | `0.75` (0.20833) | `0.5` (0.41667) | `0.25` (0.625) | `0` (0.83333) |
|---|---|---|---|---|---|---|---|
| 0 | ∞ | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.0625 | 0.8500 | 0 | −0.00490 | −0.01225 | −0.02451 | −0.03676 | −0.04902 |
| 0.1 | 0.5500 | 0 | −0.00758 | −0.01894 | −0.03788 | **−0.05682** | **−0.07576** |
| 0.25 | 0.2500 | 0 | −0.01667 | −0.04167 | **−0.08333** | **−0.12500** | **−0.16667** |
| 0.5 | 0.1500 | 0 | −0.02778 | **−0.06944** | **−0.13889** | **−0.20833** | **−0.27778** |
| 1 | 0.1000 | 0 | −0.04167 | **−0.10417** | **−0.20833** | **−0.31250** | **−0.41667** |
| 2 | 0.0750 | 0 | **−0.05556** | **−0.13889** | **−0.27778** | **−0.41667** | **−0.55556** |
| 4 | 0.0625 | 0 | **−0.06667** | **−0.16667** | **−0.33333** | **−0.50000** | **−0.66667** |

Detection is exactly `Delta >= theta (1 + w_pi)/w_pi`, in every one of the 48 cells. The
largest drop available is one part's whole share `kappa(p) = 0.8333333`, so there is a floor
under a useful precision at `w_pi = theta/(kappa − theta) = 0.0638298`: at `w_pi = 0.0625`
even total silence goes undetected, at `w_pi = 0.1` it does not.

**The rate, so that "one without does not detect" is not a budget.** At `w_pi = 0` the
surprise is `s_t = −Delta 2^{−t}`: the whole drop at step 0, falling by half each step, and
at full silence it is above threshold at steps `0..4` — `ceil(log2(Delta/theta)) = 5` — and
below it at every later step, reaching `< 1e-15` by step 60. A free prediction *is*
surprised, for five steps; what it cannot do is stay surprised.

## Test 4: the collective live, and the repair

Four settled states at 4000 steps, with the monitor at `w_pi = 0` and `w_pi = 1`; drift is
over a further 500 steps.

| cost | mode | `w_pi` | `w~` arriving | settled surprise | `kappa` of the failed part | drift |
|---|---|---|---|---|---|---|
| weighted signal | reporting | 0 | 2.7456255 | 0 | 0.7456255 | 0 |
| weighted signal | reporting | 1 | **2.5000000** | **0** | **0.5000000** | 0 |
| weighted signal | silent | 0 | 2.0000000 | 0 | 0.8333333 | 0 |
| weighted signal | silent | 1 | 2.0000000 | **−0.25** | 0.8333333 | 0 |
| mean readout | reporting | 0 | 1.9926702 | −1.9e−6 | 0.0094600 | 2.0e−2 |
| mean readout | reporting | 1 | 2.3862888 | **−0.0568556** | 0.3862888 | 5.5e−76 |
| mean readout | silent | 0 | 1.9230078 | 0 | 0.8333333 | 0 |
| mean readout | silent | 1 | 2.0000000 | **−0.25** | 0.8333333 | 2.2e−296 |

Three things, of which only the second was predicted.

**The free monitor's settled surprise is zero in every row** — to one unit in the last place,
`2.2e-16` — except in the one run that never settles, the mean readout with a reporting
failure, where `kappa_F` is still falling and the free monitor lags by `1.9e-6`, a
thousandth of the way to the threshold. A free prediction is surprised only by what is still
changing, and only in proportion to how fast.

**The monitor with a prior detects the silent failure under both costs**, `−0.25` against a
threshold of `0.05`, and the closed form `s = −(w_pi/(1+w_pi))(pi_0 − w~)` holds in every
settled row. Under the mean readout it is the only thing in the composite that registers
anything at all.

**But it does not only detect: it acts, and what it does is repair.** Under the required
cost with a **reporting** failure — the case the turnover run found that cost cannot push
out, because a zero signal at full weight and a silent part are literally the same input to
it — the monitor drives the dead part's kernel from `0.8333333` to `0.5000000`, exactly
until the weight it expects arrives, and its own surprise closes to zero. *The monitor
supplies the sense the parent's cost lacks, and the collective then uses it.* It cannot do
this to a silent failure: a silent part carries no weight, so moving it changes nothing, and
the surprise stays open at `−0.25` forever. Under the mean readout with a reporting failure
the prior does the opposite service — turnover found that run has no settled state, the
failed part pushed away at every step; with the prior it settles at `kappa_F = 0.3862888`,
the monitor's surprise arresting the runaway.

Note also the two free-monitor rows that are *not* the collective's own resting place:
`2.7456255` and `1.9230078`. With `w_pi = 0` the monitor exerts no settled force, but its
*transient* surprise — five steps of it — kicks the collective, and it settles somewhere
else. **Even a monitor that ends up detecting nothing has already changed what it watched.**

## What the runs support, and what they do not

They support the prediction's mechanism, in the two halves that are about the monitor: a
free prediction's settled surprise is exactly zero, a held one's is `w_pi/(1 + w_pi)` of the
drop, and the prior precision fixes the detectable drop at `theta(1 + w_pi)/w_pi` with a
floor at `theta/(kappa − theta)`. They support "a monitor can see a dead part" for a
**silent** failure — the one the mean readout cannot see at all — at every precision above
that floor.

Four things they do not support.

First, **the prediction's premise is refuted for the cost the construction is required to
carry.** A silent failure is not invisible to a cost on the weighted signal: the signal
falls with the weight, the deficit is `kappa(p)`, and the collective moves. The **Able to
fail** clause therefore fires under the required cost. The prediction's premise holds only
under the mean readout, which leaves the affine-quadratic family — so, exactly as in
`notes/TURNOVER-TEST.md`, the aggregate that makes the construction settle is not the
aggregate that makes the failure invisible. The mechanism is *needed* only where the
mathematics is worse behaved.

Second, **"detects" is not a thing this formalism can do on its own.** A held expectation is
a cost and a cost is a force; the monitor's surprise acts on every coordinate its input
depends on. Test 3's clean table exists only because the collective is frozen there by
stipulation. There is no passive observer in `arr`, and a later stage that composes seeing
with pulling will be composing two interventions, not a sensor with an actuator.

Third, **the repair is the interesting result and it was not predicted.** That a monitor
holding an expectation of the arriving weight pushes out a part that claims weight without
contributing — the Ship-of-Theseus case the required cost is blind to — is a claim about
collectives, not only about the model, and it is the thing to put to David. It is also
narrow: it works only against a failure that keeps claiming weight, and not at all against
one that goes silent.

Fourth, and standing behind all of it, STATUS.md §2's warning applies in full: any smooth
vector field can be written as a forcing, so a construction that behaves as wanted proves
nothing about the class of constructions that do.

## What is stipulated, and what is derived

**Stipulated.** The detection threshold `theta = 0.05` — every "detected" in this file is
relative to it, and the closed form `Delta_min = theta(1 + w_pi)/w_pi` is what survives if
it changes. The monitor's expectation `pi_0 = nu`, its position `p = 0` and that position
being frozen (listening from the parent's own vantage point). The monitor sending `0_M`. The
monitor's mobility `1/(2(1 + w_pi))`, chosen so the contraction is `1/2` in every run. The
collective being frozen in test 3. The failure modes and the health gating, inherited from
`tests/test_turnover.py`, where they are stipulations too.

**Derived.** `kappa(p) = nu/3` and `p = 0.5401060` from `3 kappa(p) = nu`. That the monitor's
input is `sigma_a(q, m)` (from `l_0 = 0_P` and `p_MON = 0`). The added forcing `−s grad w~`
and both of its components. `pi*`, `s*`, and `s_t = s* − 2^{−t}(Delta + s*)`. `Delta_min` and
the precision floor `theta/(kappa − theta)`. The settled `w~` and `kappa_F` of test 4, which
are run results, not derivations, and carry their drift.

## Adapted from the brief

- The brief asked for both costs and both to be reported, not a choice between them; that is
  what tests 2 and 4 do, and the two verdicts differ.
- "Whether the drop registers as a force on the monitor" is read as the **settled surprise**
  `w~ − pi`, which is exactly the force the monitor's error term holds on `pi` against its
  prior. At `w_pi = 0` that force is zero at settlement and the transient is reported with
  its rate.
- The monitor watches the collective with **no spare**, so that the drop is not closed by a
  replacement; recruitment is the turnover test's subject and reach is stage B's.
- The `corrupting` mode of `tests/test_turnover.py` is not run here: it changes the signal,
  not the weight, and the monitor watches the weight.
