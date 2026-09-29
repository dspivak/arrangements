# P30 (d): word of mouth — predictions, fixed before code

Claude Fable 5.1, session 47a21ac3, 2026-09-29. The book plan's §5.9 item 38 (Claude, 2026-09-28,
*proposed*, from David's words: "because other members of the town rely on those people, they wear
down … there is stress, and people move away. word of mouth can make them hard to replace"; "bad
juju builds up and that can knock things out of their equilibrium, so that a lot of effort can (if
the accounting backs it up) aim at the problem joint"). Baseline: the turnover run
(`tests/test_turnover.py`, STATUS.md §3.9), whose numbers are the comparison.

## Design

The turnover construction unchanged: a spatial-attention parent with cost `(1/2)(nu - h~)^2` on
the weighted signal it sends onward, `nu = 2.5`, `tau = 0.8`; three contributors at the positions
that make the pre-failure state a fixed point; one contributor's health driven to zero by its own
cost (the failure stipulation); a spare at `d = 2`. One change, the plan's: **the forcing that
places a spare is multiplied by the word `g` of the parts still present**,

    g(m) = sum_j rho_j w_j / sum_j rho_j WGT      (j over the original incumbents),

where `w_j` is the weight in contributor `j`'s message and `rho_j >= 0` is the share of the
recruiting channel that part carries, a wiring datum. In the reporting mode `w_j = WGT` whatever
the health, so `g = 1` throughout; in the silent mode `w_j = c_j WGT`, so a failing part's word
dies with it. A spare carries `rho = 0` (a newcomer brings no word) unless the run says
`inherit`. Nothing else moves: the cost, the kernel, the failure and the spare's own message are
item 13's. Implemented as a wrapper on the parent's arrangement whose one-form scales the spare
placements' components of the forcing by `g(m_out)`; the forcing is no longer `-dk`, so the
wrapped parent is inside `arr` and outside the attention operad.

For prediction (3) the survivors' health is made mobile and fed through a wiring: part `j`'s
health cost is `(lam_h/2)(c_j - t_j)^2` with target `t_j = (1 - a_j) + a_j c_F`, a share `a_j` of
its supply arriving through the failing part `F` and the rest from outside, ungated. For (4) a
doodad (item 14 route (b)): a part whose alarm `a` is its state, driven by what it reads, and
whose returned covector adds `gain * a` to the dependent part's supply at the joint; a control
whose reading wire is cut (an insulator: it reads `0`).

## The plan's predictions (§5.9 item 38, fixed 2026-09-28), and how each is read here

1. **No exact return.** The settled outgoing signal stays below `nu`: the deficit has a positive
   floor rather than item 13's zero. *Read as:* silent mode, `rho` shared equally by the three
   incumbents (`g = 2/3` after the failure), spare at `d = 2`; held iff `h~` stays below
   `nu - 1e-3` at every budget up to 2,000,000 steps.
2. **Compounding, and smooth.** Successive failures (three, each followed by a spare) give a
   monotone increasing sequence of settled deficits with no jump. *Read as:* the deficit at a
   fixed budget of 8,000 steps after each failure, and the recovery time to `nu - 1e-3`.
3. **Mutual worsening is ordered by the wiring.** After the failure the settled health of a part
   whose supply ran through the failed one falls further than that of a part whose did not.
4. **The accounting condition.** The deficit recovers when the joint's cost reaches a part that
   can act on it, and does not when the same cost exists and reaches no actor; the doodad's
   forcing is non-exact (`d omega != 0`).

## The run's own predictions (Fable, before code; each able to fail)

- **R0 (a check, not a result).** In the reporting mode `g = 1`, and every trajectory equals the
  turnover run's to `1e-12`: the construction reduces to item 13 when the word is intact.
- **R1.** Prediction 1 is refuted for every `g > 0`: the return is exact, only slower, since
  item 13 found reach to be a rate and not a radius; the time to return scales about as `1/g`.
  It holds only at `g = 0` (the leaver carried the whole channel, `rho_F = 1`, silent mode),
  where the spare never moves and the collective settles at item 13's no-spare value
  `2 + (nu - 2)/(1 + lam) = 2.0098`.
- **R2.** So prediction 2 splits: the recovery *times* increase monotonically and the last
  (word exhausted, `g = 0`) is infinite; at a fixed budget the deficits are monotone and the
  jump at the last failure is the channel running out, a wiring event, not a threshold reached by
  accumulation. With `inherit` (a recruit takes over the leaver's share) there is no compounding:
  the recovery times are equal.
- **R3.** Prediction 3 holds by construction (the wiring says who loses supply); what could go
  otherwise and is the result: the parent's own deficit props a starved part up, as it propped
  the failing part in item 13, so the starved part's settled health sits *above* its wiring
  target by `(nu - h~) kappa_j SIG / lam_h`, and the ordering by wiring survives that propping.
  The starved part is not pushed out: its contribution is low, not wrong (item 13).
- **R4.** With the doodad wired, the collective's deficit is smaller than with it insulated,
  but does **not** close: a proportional alarm needs a nonzero deficit to keep acting, so a
  residual `~ 1/gain` remains (a proportional-control offset), the same shape as item 14's
  doodad that holds the collective past what it asked for. `d omega` of the doodad's coupler is
  nonzero: it acts and nothing returns to it.
- **R5.** The word-gated parent is non-exact: the antisymmetric part of the composite forcing's
  Jacobian between a failing part's health and the spare's placement is nonzero at the failed
  state (the word moves the spare; the spare's arrival returns nothing to the word).

## Not run, and why

A second reading, in which the recruit's *contribution* is also scaled by `g` (the newcomer can
do only what the incumbents show it), gives a positive floor `nu - 2 - g` by construction
whenever `g < nu - 2`; that is what the construction was built to have (STATUS.md §2) and it is
not run.
