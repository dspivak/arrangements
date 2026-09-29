# P30 (d): word of mouth — results (`tests/test_word_of_mouth.py`)

Claude Fable 5.1, session 47a21ac3, 2026-09-29. Predictions:
`notes/WORD-OF-MOUTH-PREDICTIONS-2026-09-29.md`, the plan's four (§5.9 item 38, fixed
2026-09-28) and the run's own R0–R5, fixed before code. Tests: 6, all passed (about 8 min, most
of it the runs that wait 200,000–400,000 steps for a return that never comes). Baseline: the
turnover run (`tests/test_turnover.py`, STATUS.md §3.9), reproduced to `1e-12` when the word
is intact (R0).

## The construction, in one paragraph

Item 13's collective: a spatial-attention parent, cost `(1/2)(nu - h~)^2` on the weighted signal
it sends onward, three incumbents, one failing, a spare at `d = 2`. One change: the forcing that
places the spare is multiplied by the **word** `g = sum_j rho_j w_j / sum_j rho_j WGT`, the share
of the recruiting channel still carried by parts still claiming weight (`rho_j` a wiring datum;
silent mode, so a leaver's weight dies with its health). For predictions 3 and 4, a supply cost
feeding part 0 through the failing part by a share `0.6`, and a **doodad** (item 14 route (b))
whose alarm reads a deficit through an insulated wire and adds `gain * alarm` to part 0's supply.

## The plan's four predictions

1. **Refuted, except where the channel is wholly gone.** The time to return to `nu`:

   | leaver's share `rho_F` | word after | steps to return |
   |---|---|---|
   | 0 (item 13) | 1 | 40 |
   | 1/3 | 2/3 | 60 |
   | 2/3 | 1/3 | 120 |
   | 0.9 | 0.1 | 380 |
   | 1, leaver dead | 0 | never: the spare's forcing is zero, it sits at `d`, `h~ = 2.082085 = 2 + kappa(d)` |
   | 1, leaver failing | a whisper | 238,000 |

   The return is exact and the time scales as `1/g` (R1 held). The deficit has no floor; what
   the word sets is the *rate*, item 13's "reach is a rate, not a radius" in the recruiting
   channel. The last row is a finding: the parent's deficit props the failing part's health at
   `(nu - h~) kappa_F / lam = 2e-5` (item 13), that whisper of health is a whisper of word, and
   the spare creeps in over 238,000 steps. A part that is faint and a part that is gone are
   told apart only by freezing the one at zero.
2. **Held at a fixed budget, and the jump is the channel running out.** Three failures in turn,
   each followed by a spare, recruits carrying no word: steps to return `60, 120, never`
   (word `2/3, 1/3, 0`); at item 13's budget of 8,000 steps `h~ = 2.5, 2.5, 2.370`. The sequence
   is monotone; the last step is a jump from full recovery to none, and it is the last carrier's
   share being the last of the word, a wiring event, not accumulation reaching a threshold
   (R2 held). Control: recruits that take over the leaver's share once in return in
   `60, 60, 60` steps; no compounding at all. David's "bad juju … knock[s] things out of
   equilibrium" is therefore not what this model has; what it has is a channel that thins.
3. **Held, by the wiring, and the propping is the finding.** No spare: after the failure, part 0
   (fed `0.6` through the leaver) settles at `0.4395`, part 1 (fed from outside) at `1.0202`,
   the leaver at `0.0323`, `h~ = 1.4919`, everyone pulled home (`kappa = 1`). The closed forms
   hold to `1e-9`: `c_j = t_j + (nu - h~) kappa_j / lam_h`, the collective's deficit propping
   every health by the same amount, so the ordering is the wiring's; and
   `c_F = [(nu - h~) kappa_F + lam_h a (c_0 - t_0)] / lam`, the leaver propped by the
   collective's deficit *and* by its dependent's need (composition pulls the dependent's cost
   back onto the leaver, item 26's mechanism). Part 0 is not pushed out: low, not wrong (item 13).
4. **Held, with a residual that names the mechanism** (R4 held). Spare at `d = 2`, word `2/3`,
   doodad at the joint:

   | doodad reads | gain | steps to return | `c_0` | alarm | collective's deficit |
   |---|---|---|---|---|---|
   | nothing (wire cut) | 4 | never (200,000) | 0.4036 | 0 | 0.0916 |
   | the joint's account `1 - c_0` | 1 | 60 | 0.7000 | 0.3000 | 0 |
   | the joint's account | 4 | 60 | 0.8800 | 0.1200 | 0 |
   | the joint's account | 16 | 60 | 0.9647 | 0.0353 | 0 |
   | the collective's felt deficit `nu - h~` | 4 | never (200,000) | 0.4793 | 0.0196 | 0.0196 |

   The joint's residual is `(1 - t_0 - pi)/(1 + gain)`, a proportional-control offset: effort
   proportional to a felt shortfall never quite closes it. The collective's deficit closes iff
   the joint's account reaches the actor: read by the collective's own felt deficit, the doodad
   aims at the joint it cannot tell apart from the leaver's loss and leaves both short
   (`5 delta = 0.1`, `delta = 0.02`, self-consistently). The doodad's coupling is non-exact: the
   antisymmetric part between alarm and joint is `lam_h gain + lam_a gain_r` wired and
   `lam_h gain` cut, to `1e-9`; it acts and nothing returns to it, so by §5.5's physical
   reading it needs its own supply, which this construction stipulates.

R5 held: the word-gated parent is non-exact, `(d omega)(c_F, p_s) = (rho_F / sum rho) xi_s -
(1 - g) d xi_s/d c_F`, `-0.0551` at `rho_F = 1/3`, `-0.165` at `1`, `0` at `0`, to `1e-12`.

**Found while running.** The joint-and-alarm loop is a linear map per step; with both moving a
half step it has eigenvalue modulus `(1/2) sqrt(1 + gain)`, and at `gain = 4` the first run
passed through `nu` and rang itself to `nan` within 3,000 steps. The continuous-time loop is
stable at every gain. The alarm now moves a twentieth per step; the fixed points are unchanged.

## Reasonable or weird (the agent's judgment, not evidence)

- **Slower, not lower.** Reasonable for a town or a firm: losing the person who knew everyone
  makes the next hire slow. Weird that any whisper suffices given time; with a finite patience
  (a budget, item 13) it reads as a floor, and that is how a town would read it.
- **The propped leaver still vouches.** Weird at every scale — a part the collective's need
  keeps at `2e-5` still recruits — and it is item 13's "cannot see absence" again: the cost sees
  a faint contribution and an absent one alike, and so does the word.
- **Compounding as a thinning channel, with a cliff at the last carrier.** Reasonable for small
  towns and firms (institutional memory leaving with the last old hand); the *knock* David
  described is not here, and the book has no mathematics for accumulation-then-threshold.
- **The dependent falls by its wiring; the collective's shortfall pulls everyone, the leaver
  included.** Reasonable for a body (a failing organ propped by the whole's demand) and for a
  team; nothing weird.
- **Effort aimed by the joint's account closes the whole; effort aimed by the whole's felt
  shortfall leaves both short.** Reasonable for management at every scale, and it is what David
  asked: "a lot of effort can (if the accounting backs it up) aim at the problem joint." The
  proportional residual is control theory's, and reasonable.
- **A fast alarm on a stiff joint rings.** Reasonable (overshoot); weird only that the book's
  integrator, being a step, sees it where the continuous loop would not.

## What this does not establish

The floor prediction 1 asked for is not in this construction and would be built in by scaling
the recruit's *contribution* by the word (not run: what it was built to have). "Wears out" is a
stipulation, as in item 13; "leaves" is not a movement, the failed part stays where it is.
The word is one number per part, not a message with content. The doodad's supply is stipulated.
Every verdict is the boundary of a region, not a tendency (STATUS.md §2).
