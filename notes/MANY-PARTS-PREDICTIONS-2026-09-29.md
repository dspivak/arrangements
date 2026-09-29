# P30 (e): many parts that keep one another whole — predictions, fixed before code

Claude Fable 5.1, session 47a21ac3, 2026-09-29. The book plan's P30 stage (e) (David, 2026-09-29,
on §1.7's footnote "larger collectives are still being worked out": "then we should do it now. we
are at the point where we need to know"; the stage Claude Opus 5.5 (1M), 16e01cc3, *proposed*).
Baseline: the mutual-need run (`tests/test_mutual_need.py`, STATUS.md §3.8): two parts, each a
driven steady state, each one's held form supplied by the other through a gate.

## Design

`n >= 3` parts, each item 12's part: a configuration coalgebra on its amplitude `q_i` with
mobility `h`, publishing `q_i`, carrying its own dissipation `-d (b/2) q_i^2`. One supply box
holding `s` (zero response). A stateless coupler on the ports `(q_1..q_n, s)` returns

    F_i = alpha ( s m_i g(r_i, q_i) + beta r_i ),      g(r, q) = r q / (K + r q),

where `r_i` is what part `i` receives from the parts it needs — **ring**: `r_i = q_{i-1}`;
**all-to-all**: `r_i = mean_{j != i} q_j` — and `m_i in {0, 1}` is the supply mask (a cut supply
wire delivers `0`, item 12's reading of a cut). The gate is item 12's product gate, moved onto
what arrives from the needed parts: with `beta = 0` a part with nothing arriving has no drive at
all (item 12's "no ungated path to a part"); `beta > 0` is item 12's ungated internal coupling,
run as a control. `alpha = b = K = 1`, `h = 0.01`, `beta in {0, 0.5}`, `s in {2.5, 4}`,
`n in {2, 3, 5, 8}`; `n = 2` is the pair. A form is held iff its amplitude exceeds `1e-3` at
the end of a 20,000-step run, and a persistence claim carries its rate.

Closed forms, derived here and to be reproduced. At the symmetric state `q_i = q` both wirings
have `r = q`, so the fixed points solve `(b - alpha beta) q (K + q^2) = alpha s q^2`:

    q± = [alpha s ± sqrt(alpha^2 s^2 - 4 (b - alpha beta)^2 K)] / (2 (b - alpha beta)),
    s_* = 2 (b - alpha beta) sqrt(K) / alpha,

so `s_* = 2` at `beta = 0` and `1` at `beta = 0.5`, for every `n` and both wirings. Stability at
`q+`: with `gam = K q / (K + q^2)^2` the ring's Jacobian is circulant with eigenvalues
`alpha s gam (1 + e^{2 pi i k/n}) - b + alpha beta e^{2 pi i k/n}`, whose largest real part is
the symmetric mode `k = 0`, `2 alpha s gam + alpha beta - b`, and that is the pair's condition;
all-to-all's other modes are `alpha s gam - b - (alpha s gam + alpha beta)/(n - 1)`, more stable.
After one part dies in all-to-all, the survivors' symmetric state has `r = c_n q`,
`c_n = (n - 2)/(n - 1)`, and holds iff `s > s_*' = 2 (b - alpha beta c_n) sqrt(K / c_n) / alpha`:
`2 sqrt 2 = 2.828` at `n = 3`, `beta = 0`.

## The plan's predictions (P30 (e), fixed 2026-09-29), and how each is read here

1. **Both wirings hold together under the same supply condition as the pair.** *Read as:* the
   held/not-held verdict at `s` on either side of `s_*` is the same for the ring and all-to-all
   at every `n` run, and the same as the pair's; the settled amplitude equals `q+` to `1e-6`.
2. **Cutting one part's supply ends the ring (every part, in turn), while all-to-all loses only
   that part and settles lower.** *Read as:* `m_i = 0` for one part, from the settled state.
3. **The ring's collapse runs around it at a speed set by the parts' settling time.** *Read as:*
   the times at which successive parts fall below `1e-3`, their differences constant around the
   ring and scaling as `1/b`.

## The run's own predictions (Fable, before code; each able to fail)

- **E1.** Prediction 1 holds, and the mechanism is that the threshold is carried by the symmetric
  mode, which does not see `n` or the wiring; the ring's non-symmetric modes are all more stable
  than the symmetric one, so no ring mode goes unstable at any `n` run.
- **E2.** Prediction 2's ring half holds only at `beta = 0`: with `beta = 0.5` the cut part
  settles at `alpha beta q / b = q/2` rather than dying, its neighbour's gate still reads well
  above `K`, and the ring holds with one part degraded. Prediction 2's all-to-all half is decided
  by `s` against `s_*'`: held at `s = 4`, `n = 3` (survivors settle at `q = 3.414`, below the
  intact `3.732`); refuted at `s = 2.5`, `n = 3` (`2.5 < 2.828`, the survivors collapse, though the
  intact collective held there); held again at `n = 5` and `8` at `s = 2.5` (`s_*' = 2.309`,
  `2.160`). So a bigger all-to-all collective survives a loss that a smaller one does not, at the
  same supply; the ring at `beta = 0` never does.
- **E3.** Prediction 3 holds as a front: the delay between successive deaths is the same for every
  part around the ring (to a few percent), is about `(1/b) log(q+^2 / K)` (the time a part takes to
  fall from its amplitude to where its neighbour's gate closes), doubles when `b` is halved with
  `s/b` held fixed, and grows with `s`. So the speed is set by the settling rate and by how far
  above the gate's scale the parts were held, not by the settling rate alone.
- **E4.** Cutting a part's *outgoing* wire instead (it is removed from what its neighbours
  receive) has the same effect around the ring as cutting its supply, one part later: the ring's
  collapse is the same front starting one part on.
- **E5.** The restart threshold of item 12 survives: the origin is stable for every `n`
  (eigenvalues `-b` and `-b + alpha beta`), so a collective started below the saddle `q-` on the
  symmetric ray never lights up, and one started above does.
