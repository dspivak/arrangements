# P30 (e): many parts that keep one another whole — results

Claude Fable 5.1, session 47a21ac3, 2026-09-29. Predictions:
`notes/MANY-PARTS-PREDICTIONS-2026-09-29.md`, fixed before code. Tests:
`tests/test_many_parts.py`, 5 passed (about 20 s). Construction: `n` parts of the mutual-need
kind (`tests/test_mutual_need.py`), one supply box, a stateless coupler returning
`F_i = alpha (s m_i g(r_i, q_i) + beta r_i)` with `g(r, q) = rq/(K + rq)`, `r_i` the needed
neighbour's amplitude (ring) or the others' mean (all-to-all); `alpha = b = K = 1`, `h = 0.01`.

## The plan's three predictions

1. **Held.** The supply threshold is `s_* = 2 (b - alpha beta) sqrt K / alpha` (`2` at
   `beta = 0`, `1` at `beta = 0.5`) for `n = 2, 3, 5, 8` and both wirings: nothing held at
   `s = 1.5`, every part at `q+` (`2.000000`, `3.732051`) at `s = 2.5, 4`, to `1e-6`. Mechanism
   (E1, held): the Jacobian at `q+` is circulant with eigenvalues
   `alpha s gam (1 + w^k) - b + alpha beta w^k`, reproduced to `1e-6`; the largest real part is
   the symmetric mode's, `2 alpha s gam + alpha beta - b`, which is the pair's, so the threshold
   is carried by the mode that does not see `n` or the wiring.
2. **Held in half, and the half that fails names the condition.** *Ring, `beta = 0`*: cutting
   one part's supply kills every part (`< 1e-30` at `n = 3, 5, 8`, `s = 2.5, 4`). *Ring,
   `beta = 0.5`* (E2, held): the cut part settles at `alpha beta q_{i-1} / b`, not zero, and the
   ring holds, degraded in the direction of need — at `n = 8`, `s = 4`, amplitudes `3.917,
   5.790, 6.796, 7.319, 7.589, 7.727, 7.798, 7.835` going round from the cut part, the intact
   value `7.873`. So "ends the ring" needs no ungated internal path, item 12's condition again,
   and one ungated fraction turns collapse into a graded loss. *All-to-all*: the cut part dies
   and the survivors hold iff `s > s_*' = 2 sqrt(K (n-1)/(n-2))` (E2, held): at `s = 4` they
   settle at `3.414214, 3.632993, 3.683251` (`n = 3, 5, 8`), below the intact `3.732051`; at
   `s = 2.5` the three-part collective collapses (`s_*' = 2.828`) while five and eight hold
   (`1.728714`, `1.879153`; `s_*' = 2.309, 2.160`). A larger all-to-all collective survives a
   loss that a smaller one does not, at a supply where every intact collective held.
3. **Held as a front, with the rate named.** At `n = 8`, `s = 4`, `beta = 0` the deaths go round
   in order at h-times `8.19, 11.32, 14.39, 17.45, 20.52, 23.59, 26.66, 29.72`: the delay is
   `3.07` for every part after the first (spread under 2%); halving `b` with `s/b` fixed doubles
   it (an exact rescaling of the field); it grows with `s`, `2.43, 3.07, 3.91` at `s = 3, 4, 6`.
   E3's guess `(1/b) log(q+^2/K)` gives `1.92, 2.63, 3.53`: right in its `s`-dependence
   (differences `0.64, 0.85` measured against `0.71, 0.89`), low by about `0.45` units at every
   `s`, the time a part takes to fall from the gate's closing to the `1e-3` line. So the speed is
   the settling rate times how far above the gate's scale the parts were held, plus a constant.
   E4 held: cutting a part's outgoing wire starts the same front one part on (`1, 2, …, 7, 0`).

E5 held: the origin is stable (`-b I`), and on the symmetric ray `0.9 q-` dies while `1.1 q-`
settles at `q+`, both wirings, `n = 3, 5`. The ring's coupler is non-exact (`|d omega| = 0.069`
at the symmetric state, `0.60` off it); all-to-all's is exact *at* the symmetric state and
non-exact off it (`0.017`, `0.004`), the symmetry putting it back as item 26's kernel did.

## Reasonable or weird (the agent's judgment, not evidence)

- **Threshold independent of size and wiring.** Reasonable for a body (a bigger body is not
  harder to keep running per organ) and for a firm. Weird if read as "size never matters": it
  holds because supply enters *every* part and is gated by a neighbour; a chain that relays one
  supply from one end would attenuate (item 26's "below the line"), and that was not run.
- **A strict ring collapses from one cut; one ungated fraction grades the loss by distance
  downstream.** Reasonable at every scale: just-in-time chains fail whole, chains with slack
  fail upstream-to-downstream. Weird for a town read at `beta = 0`, where every part has some
  ungated path (savings, a second supplier).
- **Bigger all-to-all collectives survive a loss smaller ones do not.** Reasonable for tissue
  (redundant cells) and for teams; the threshold `s_*'` is a closed form a reader can check.
- **The front's constant delay.** Reasonable as a cascade with a fixed per-link lag; weird only
  in its cleanness, since every part here is identical.

## What this does not establish

That collectives fall in the region where the pair's threshold holds (STATUS.md §2). The gate's
shape is a model. The pair of item 12 had one supply entering one part and asymmetric internal
couplings; these collectives give every part its own gated supply, which is why the symmetric
state exists. Needs of different kinds, or need of a *particular* other part beyond the ring's
neighbour, are not run. Nothing here disperses: there are no placements.
