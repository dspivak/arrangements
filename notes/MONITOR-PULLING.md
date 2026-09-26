# The pulling test (`tests/test_monitor_pulling.py`)

Stage B of David's monitor test, 2026-09-25, from his question: "there can't be a monitor
who is able to see dead parts and also able to pull from far? ... you're saying that you
can't make a doodad that is able to apply force to things, like walk over to something and
pull it over to another location? i also don't see why it has to be an exact form." Stage A
is seeing; this file is pulling. `uv run --extra test pytest tests/test_monitor_pulling.py
-q`: **6 passed (116 s)**.

The prediction is quoted verbatim in the module docstring and was written there before any
construction code existed:

> Route (a) extends the pull region with `tau` but flattens selectivity at the same level,
> so one level cannot have both and the nesting is forced; route (b) recruits at any
> distance and shows a nonzero antisymmetric part in the recruiting coupler. **Able to
> fail:** if (a) reaches far *and* stays selective, no monitor is needed.

Verdicts. **Route (a): held.** Reach and selectivity move in opposite directions over five
doublings of `tau`, monotonically and without a threshold having to be chosen, and the
two-level arrangement has both at once. **Route (b): held.** The doodad recruits from
`d = 100` in 9755 steps where one level at `tau = 0.8` needs two million for `d = 5` and
never arrives from `d = 6`; the recruiting coupler's antisymmetric part is
`-alpha tanh(p/wid)`, `-0.02` where the pull is saturated, and it is the only nonzero
entry.

One word of the prediction does not survive: **"forced" holds only inside the attention
operad.** The run shows that one level cannot have both and that nesting can — and it also
shows that a doodad can, without nesting. So nesting is what is forced if the forcing must
stay exact, and route (b) is the price of not nesting: a supply, which this construction
stipulates and does not supply.

## The constructions

**The collective** is the turnover test's, with `tau` a parameter: a spatial-attention
parent over contributors, its cost `(1/2)(nu - h~)^2` on the weighted signal
`h~ = sum_j kappa_j v_j` it sends onward, `kappa_j = exp(-p_j^2/2 tau)`, value maps frozen
at the identity so that the parent's only adaptation is attentional. A contributor's
configuration is the value `v` it signals, frozen: `v = 1` in full health, `v = 0` dead but
still claiming weight, `v = -1` wrong rather than absent. There is no health dynamics in
this stage, so every force below is the parent's attention or the doodad's pull. The
failed state and the spare at `d` are the turnover test's, `nu = 2.5`, `MU_P = 0.5`.

**Route (a), one level:** the same collective, `tau` swept over `0.2 ... 6.4`.

**Route (a), two levels:** the inner parent (`tau_in = 0.2`) holds a good and a bad
contributor at `+-1`; the outer parent (`tau_out = 3.2`) holds *that collective* and a
spare at `d = 6`. A collective's interface is a contributor's — both are `<E|M>` — so
`compose_seq` takes the nesting as it stands. Nothing is stipulated to make it compose.

**Route (b), the doodad:** a monitor part whose configuration is its own alarm `a`,
published at its port, carrying its own exact cost `(lam/2)(a - a0)^2`. The collective is
wrapped so that it publishes the spare's placement — only `out_f` changes, the forcing is
untouched — and a stateless coupler in the two-tank idiom returns

    F_p(e) = -alpha a tanh(p / wid),    F_a(e) = 0,    F_h~ = F_w~ = 0

on the ports `e = (a, h~, w~, p)`: a bounded inward pull of magnitude `alpha a` once
`|p| >> wid`, with nothing returning to the monitor. Publishing a placement so that
something other than the parent can put a force on it is the step that leaves the attention
operad.

Test 0: `Phi` of the composite, `run_closed` and the `lax.scan` of `traj` agree over fifty
steps to within one unit in the last place, for all three composites, so every trajectory
here is `Phi`'s. Test 1 checks the hand-written `sigma_a` against the library's bit for
bit at two `tau`, which licenses reading `h~` off a whole trajectory in numpy.

## Test 1: a reach number is a pair, never a radius

At `tau = 0.8` a spare at `d = 3` arrives at step **158**; at `tau = 0.4` the same distance
takes **6108**; at `tau = 0.2` it does not arrive within 8000 steps, nor within 200 000 —
and the force on it is inward at every step: `-(nu - h~)(d/tau) kappa(d) = -2.11487e-9`,
reproduced to `1e-12`, which at mobility `0.5` is about `1e-9` of distance a step and so
some `2e9` steps for the trip. `notes/TURNOVER-TEST.md`'s correction, restated in this
file's terms: every reach number below is a budget.

## Test 2: route (a) at one level — the sweep

All at the same budget, **8000 steps**, in the same fixed spatial units: a collective's
`tau` may change but the world's distances do not. *Reach* is the largest `d` from which
the spare is recruited within the budget, by bisection to `0.05`. *Selectivity, statically*
is the ratio of the parent's attention across a fixed gap, `kappa(1)/kappa(1.5)`, closed
form `exp(0.625/tau)`. *Selectivity, dynamically* runs a good and a bad contributor from
equal distances `+-1` with target `1`, and reports where they are after the same 8000
steps.

| `tau` | reach @ 8k | reach / `sqrt(tau)` | arrival from `d = 3` | `kappa_good` | `kappa_bad` | `1/kappa_bad` | static `kappa(1)/kappa(1.5)` | drift / 500 |
|---|---|---|---|---|---|---|---|---|
| 0.2 | 2.197 | 4.913 | — (none in 200 000) | 1 | 0.00143053 | 699.0 | 22.7599 | 0.0040 |
| 0.4 | 3.007 | 4.755 | 6108 | 1 | 0.00210778 | 474.4 | 4.77073 | 0.0060 |
| 0.8 | 4.126 | 4.613 | 158 | 1 | 0.00309412 | 323.2 | 2.18420 | 0.0088 |
| 1.6 | 5.630 | 4.451 | 137 | 1 | 0.00455220 | 219.7 | 1.47790 | 0.0130 |
| 3.2 | 7.636 | 4.269 | 164 | 1 | 0.00672388 | 148.7 | 1.21569 | 0.0193 |
| 6.4 | 10.298 | 4.071 | 219 | 1 − 1e−15 | 0.00998415 | 100.2 | 1.10258 | 0.0289 |

Reach rises monotonically and **as the kernel's own scale does**: `reach / sqrt(tau)` stays
in `[4.07, 4.91]` across five doublings. The residual attention to a bad contributor rises
with it, nearly as `sqrt(tau)` as well. So **reach and the residual grow together, and on
`0.2 <= tau <= 6.4` the two cannot be improved together.** That is a strict trade-off, not
a threshold: no number had to be chosen to say which levels count as far enough or
selective enough. The drift column is there because the selectivity figure is a budget
figure — the bad contributor is still moving out at step 8000, at every `tau`, and would
keep moving; it is a rate, like the reach.

The bisection is well posed: at `tau = 0.8` the arrivals within budget at
`d = 1, 2, 3, 4, 4.5, 5, 6` are `yes, yes, yes, yes, no, no, no` — one threshold, not a
patchwork.

## Test 3: route (a) at two levels — both at once, and nothing stipulated

One run, the same 8000-step budget.

| | outer level (`tau = 3.2`) | inner level (`tau = 0.2`) |
|---|---|---|
| what it holds | the inner collective, and a spare at `d = 6` | a good and a bad contributor at `+-1` |
| result | target `1.5` reached at step **643**; spare in from `6.0` to `2.105856`, `kappa = 0.500118` | `kappa_good = 1`, `kappa_bad = 1.1811e-4`, ratio **8467** |
| one level alone could do | `tau = 0.2` does not recruit from `6` at all (reach `2.20`) | `tau = 3.2` discriminates at `149`, not `8467` |

Both at once, each better than one level at either `tau`. The inner ratio exceeds the `699`
of a lone `tau = 0.2` level because the outer parent's own deficit drives the inner pair
apart as well: **the coarse level helps the fine one**, which was not predicted. And the
nesting itself costs nothing to build — a collective's interface is a contributor's, so
this is ordinary operadic composition, each level with its own single scalar `tau`.

## Test 4: route (b) — the doodad recruits at any distance, at a linear rate

The alarm sits at `a = 1`; the placement's mobility is `0.5`; so one step moves the spare
`MU_P alpha = 0.01` and the trip is **linear in the distance**, at
`1/(MU_P alpha a) = 100` steps per unit. Measured arrival is exactly `100 d - 245`:

| spare at `d` | arrival step | route (a) at `tau = 0.8`, same collective |
|---|---|---|
| 5 | 255 | 2 000 000 (`notes/TURNOVER-TEST.md`) |
| 10 | 755 | — |
| 20 | 1755 | — |
| 50 | 4755 | — |
| 100 | 9755 | not from `d = 6` at 2 000 000 |

The successive differences are `100.0` steps per unit of distance exactly, at `atol = 0`.

**Where it settles is not where the parent alone would have put it.** The two live
contributors come home to `p = 0` and the spare stops at `1.007969`, where the parent's
outward push balances the pull,

    (h~ - nu)(p/tau) kappa(p) = alpha a tanh(p / wid),    h~ = 2 + kappa(p),

reproduced to `1e-4`, with the composite forcing on that coordinate below `1e-5` and the
last two steps identical bit for bit. So the collective settles **above** its target, at
`h~ = 2.529935` rather than `2.5`: a doodad that keeps pulling holds the collective past
what it asked for. The dead contributor never moves at all — it sits at its initial
`0.540106` for fifteen thousand steps, since a contribution that is merely absent exerts no
pull either way (`notes/TURNOVER-TEST.md` test 3), and the doodad pulls the spare, not it.

**The supply is stipulated away, and the test says so.** Over the trip from `d = 100` the
monitor delivers `1.979639` units of work — `alpha a` times the distance travelled, to
`1e-4` — and its own state ends where it began, `1.0`, bit for bit, minimum and maximum
alike. The doodad is an inexhaustible source. That is the one price the plan named for
route (b), and this construction does not pay it.

## Test 5: the coupler is not exact, and what making it exact costs

`F = (F_a, 0, 0, F_p)` on `e = (a, h~, w~, p)` with `F_a = 0`, so

    (d omega)_{a p} = d_a F_p - d_p F_a = -alpha tanh(p / wid),

the only nonzero entry, reproduced to `1e-12` against the hand-derived expression:

| `p` | `(d omega)_{a p}` | hand-derived |
|---|---|---|
| 0.1 | −0.007598979245 | `-0.02 tanh(0.4)` |
| 0.4 | −0.018433371088 | `-0.02 tanh(1.6)` |
| 1.0 | −0.019986585995 | `-0.02 tanh(4.0)` |
| 5.0 | −0.02 (to `1e-16`) | `-0.02 tanh(20)` |

The **composite's** forcing on `Q` carries the same entry at the alarm and the spare's
placement, `max|d omega| = alpha = 0.02`: the collective's own forcing is the differential
of a cost and contributes nothing antisymmetric. By the reciprocity lemma (§5.5) the
recruiting coupler is therefore not `dU` for any `U` — inside the book's category, whose
forcing need not be exact (STATUS.md §1), outside the attention operad, whose forcing is
the full differential of a nonnegative cost. **The prediction's second clause holds, at
`-0.02`.** Compare `tests/test_mutual_need.py`'s `(d omega)_AB = -0.1142`: the same
measurement, one case over.

**The exact completion is unique up to a function of the alarm alone.** `F_p` fixes
`U = -alpha a Psi(p)`, `Psi(p) = wid log cosh(p/wid)`, `Psi' = tanh(p/wid)`, so `F_a` must
carry `-alpha Psi(p)` — the spare's placement acting back on the monitor, which is exactly
what "nothing returning to it" forbids. With it, `d omega` is zero to `1e-17` and the
composite's antisymmetric part to `1e-15`, and `grad U = F` to `1e-12`. And `alpha a
Psi(p)` is, up to `alpha a wid log 2`, the work `alpha a |p|` of bringing the spare in from
`p`: **the exact completion is the monitor being charged the outstanding work.**

What I expected that to cost was nothing but bookkeeping. **It costs the reach**, and this
is the sharpest thing the stage found and was not predicted. The alarm settles at
`a* = a0 - alpha Psi(p)/lam`, which crosses zero at `p = lam a0/alpha + wid log 2 =
50.173`; beyond that the pull reverses, the spare runs out, the reversal deepens, and the
run diverges.

| | unbilled (non-exact) | billed (exact) |
|---|---|---|
| `d = 5`, 15 000 steps | arrives at 255 | arrives at 266 |
| `d = 40`, 40 000 steps | arrives | arrives |
| `d = 60`, 40 000 steps | arrives | **never; the configuration diverges** |

So **the exact pull has a genuine boundary and the non-exact one does not.** A bisection at
the same budget puts the boundary at `d = 51.1` (run outside the test file, same
construction; the test asserts only the `40`/`60` bracket and the closed form `50.173`).
The dynamical boundary exceeds the static crossing because the alarm relaxes at mobility
`0.01` while the spare is already moving, so it gets a head start.

## What the runs support, and what they do not

They support the prediction, both clauses, with the correction to "forced" recorded at the
top: within the attention operad one level cannot buy reach and selectivity together and
nesting buys both; outside it, a part that walks over and pulls is an ordinary arrangement,
built here out of the library's own pieces with no new machinery, and it recruits from a
hundred units away in ten thousand steps.

Five things they do not support.

First, **the trade-off is a measurement on one family**, not a theorem: one Gaussian
spatial kernel, one cost, one yardstick for selectivity (`d0 = 1`, `gap = 0.5`), one
budget (8000), one range of `tau`. What is established is that on that family the two
quantities move oppositely and monotonically. A different kernel — one with a shape
parameter beside its scale — might buy both, and nothing here rules that out.

Second, **the nesting run is one nested arrangement**, not a claim that nesting always
delivers both. That the coarse level helped the fine one is a fact about this wiring, where
the outer target is above what the inner pair currently sends; wired the other way the
outer parent would fight the inner one, and that was not run.

Third, **the exact completion's boundary is a property of this completion.** Its restoring
cost on the alarm is quadratic and lets the alarm go negative. A completion confining the
alarm to positive values would not reverse, though its pull would weaken with distance; the
reach would then be polynomial in time rather than linear. That is not tested here, so
"exactness costs reach" is established for the natural completion and not in general.

Fourth, **whether the exact pull could be carried as an attention cost is not tested.**
`k = -U = alpha a Psi(p) + (lam/2)(a - a0)^2` is nonnegative for `a >= 0` and depends only
on the configuration, so neither nonnegativity nor the degree condition in the messages
excludes it — but then the monitor's alarm is a coordinate of the parent's own
configuration and nothing walks over. Making that precise is a further question.

Fifth, and standing behind all of it, STATUS.md §2's warning applies in full: any smooth
vector field can be written as a forcing, so a construction that behaves as wanted proves
nothing about the class of constructions that do. Route (b) shows that the category admits
a doodad, not that arrangements tend to contain one.

## Stipulated, and derived

**Stipulated.** The collective, inherited from the turnover test: the cost on the weighted
signal, `nu = 2.5`, the failed state, `MU_P = 0.5`. The contributors' signals frozen at
`+1 / 0 / -1`, so this stage has no health dynamics. The selectivity yardstick, `d0 = 1`
and `gap = 0.5` — where two things are and how far apart. The budget, 8000 steps. The two
levels' scales `0.2` and `3.2`, `D_FAR = 6`, `NU_OUT = 1.5`. The doodad's pull law — a
bounded `tanh` ramp, `alpha = 0.02`, `wid = 0.25`, `a0 = 1`, `lam = 1`, alarm mobility
`0.01`. That the monitor does not read the collective: its alarm is constant, and reading
is stage A's question. That nothing returns to it — the brief's own stipulation. And the
supply: absent.

**Derived.** `sigma_a`'s closed form, checked against the library bit for bit. The inward
force `-2.11487e-9` at `d = 3`, `tau = 0.2`. The whole reach/selectivity table, measured.
The static selectivity `exp(gap(2 d0 + gap)/2 tau)`. The arrival law `100 d - 245` with the
slope `1/(MU_P alpha a)` derived and the intercept measured. The settled force balance
`(h~ - nu)(p/tau) kappa(p) = alpha a tanh(p/wid)` and `h~ = 2.529935`. The antisymmetric
part `-alpha tanh(p/wid)`. The exact completion `U = -alpha a Psi(p)`, its uniqueness up to
a function of the alarm, and the crossing `lam a0/alpha + wid log 2 = 50.173`.

## Adapted from the brief

- The brief asked for reach to be posed as a rate. Reach is therefore reported twice: as
  the largest distance recruited within a fixed budget (the sweep) and as the step at which
  a fixed distance is recruited (test 1 and test 4), and no distance is called out of reach
  without a step count beside it.
- Selectivity is reported twice as well, statically as a kernel ratio across a fixed gap
  and dynamically as where a good and a bad contributor end up at the same budget. The
  brief offered either; they agree in direction, and the static one is threshold-free.
- The **corrupting** contributor, `v = -1`, is the turnover test's mode, not the brief's; it
  is used because a cost on the weighted signal cannot see a merely absent contribution at
  all, so it is the only failure a selectivity measure can be built on.
- The **billed** variant — the exact completion, and its boundary — is mine, not the
  brief's. It was added because David's question included "i also don't see why it has to
  be an exact form," and the run can say what the alternative costs rather than only that
  it exists.
