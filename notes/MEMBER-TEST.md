# The member test (`tests/test_member.py`)

Three runs of the book plan's proposed definition of *member* (AGENTDOCPLAN.md §9.1) on
the attention operad, built exactly as `tests/test_rendezvous.py` and `tests/test_reader.py`
build theirs. `uv run --extra test pytest tests/test_member.py -q`: 3 passed.

**The definition as run.** A part has cost `k_i(q, e)`, `q` its own configuration, `e in E =
M x P` what it receives. Its residual is `r_i(e) = min_q k_i(q, e)`. It has a *need* iff `r_i`
is not constant (checked at four sample inputs). At a state it is a *member* iff
`r_i(received) < r_i(0) - 1e-9`, `0 = (0_M, 0)`; the `1e-9` is a numerical margin on the
strict inequality. The minimum over `q` is taken by gradient descent from `q = 0` (200
steps, mobility 1/2), with the gradient at the minimizer checked below `1e-8`; for a part
with no configuration `r_i = k_i`.

Throughout: `H = W = P = R`, `nu = 1`, exterior input `(0_M, 0)`, the reader `R` sends `0_M`
and has no configuration, and its cost is `(1/2)(nu - readout(m))^2` on the message `m` of
what it receives, `readout(h, w) = h / w` extended by `0_M |-> 0` (see "Adapted" below).

## Test 1: member versus part

*Construction.* Pure-placement parent (identity transfers, zero cost, trivial parameter,
placement `0`) over `[S, R, C]`. `S` sends `(h_S, 1)`, zero cost, `h_S` mobility `0.5` from
`0`. `C` sends `0_M`, cost `(1/2)(h_C - 1)^2`, `h_C` mobility `0.5` from `0`. Composite
configuration `(h_S, h_C)`. Run: `run_closed`, 100 steps (forcing below `1e-9` after 30).

*Numbers.* Settled `(h_S, h_C) = (1, 1)` to `1e-12`. `R` receives `(0, 1, 0)` at the start
and `(1, 1, 0)` settled (so does `C`; `S` receives `0_M`). Residuals over the sample inputs
`{0, ((1,1),0), ((0.5,1),0.3), ((-2,4),-1)}`: `R`: `0.5, 0, 0.125, 1.125` (need); `S` and
`C`: `0, 0, 0, 0` (no need).

| part | r(received) start | r(received) settled | r(0) | member at start | member settled |
|---|---|---|---|---|---|
| R | 0.5 | 0.0 | 0.5 | no (tie) | yes |
| C | 0.0 | 0.0 | 0.0 | no | no |
| S | 0.0 | 0.0 | 0.0 | no | no |

`C`'s own cost went from `0.5` to `0` by its own motion.

## Test 2: leaving

*Construction.* The spatial arrangement of `test_reader.py` (`tau = 0.8`, identity value
maps, zero cost) over `[R, A, B]`, positions `(0, -d, d)` with `d = sqrt(2 tau log 2) =
1.05311`, so both kernels are `1/2` at the start. `A` sends `(0, 1)`, `B` sends `(nu, 1)`,
both with zero cost and no configuration. Mobility `0.5` on `p_R`, `0` on `p_A`, `p_B` and
all three value maps. Run: 300 steps.

*Closed form (derived in the docstring, checked to `1e-12` along the run).* `R` reads
`nu kappa_B / (kappa_A + kappa_B) = nu sigmoid(2 d p_R / tau)`; its residual is
`(1/2) nu^2 (1 - sigmoid)^2`, decreasing in `p_R` with no minimum.

*Positional forcing at the initial configuration* on `(R, A, B)`: `(0.32910, -0.16455,
-0.16455) = (d/4tau, -d/8tau, -d/8tau)`, summing to zero: `R` is pushed toward `B`, `A`
away from `R`, `B` toward `R`; the frozen sources do not move. The forcing on `B`'s value
map is `0.25` (as in `test_reader.py`), frozen by the integrator.

| step | p_R | dist to A | dist to B | reads | r_R(received) | force on R |
|---|---|---|---|---|---|---|
| 0 | 0.000 | 1.053 | 1.053 | 0.500 | 0.1250 | 3.3e-1 |
| 1 | 0.165 | 1.218 | 0.889 | 0.607 | 0.0774 | 2.5e-1 |
| 10 | 0.677 | 1.730 | 0.376 | 0.856 | 0.0104 | 4.7e-2 |
| 30 | 0.937 | 1.990 | 0.116 | 0.922 | 0.0031 | 1.5e-2 |
| 50 | 1.051 | 2.104 | 0.002 | 0.941 | 0.0018 | 8.7e-3 |
| 100 | 1.200 | 2.253 | 0.146 | 0.959 | 0.0008 | 4.2e-3 |
| 300 | 1.426 | 2.479 | 0.373 | 0.977 | 0.0003 | 1.3e-3 |

Asserted: `p_R` and the distance to `A` increase at every step; the distance to `B`
decreases at every step until `R` passes `B` at step 51 and increases at every step after;
`r_R(received)` decreases at every step; `R` is a member at the start (`0.125 < 0.5`) and at
the end; the run is not settled at step 300.

## Test 3: a member of a member, outweighed

*Construction.* `b` a pure-placement parent (trivial parameter) over `[R, S]`, `S` as in
test 1 with mobility `0.01`; `a` a unary parent over `[b]` with identity transfers,
placement `0`, trivial parameter, and cost `lam (1/2) readout(m~)^2` on the message `m~` that
`b` sends. Composite `compose_seq(compose_seq(tensor([R, S]), b.arr()), a.arr())`,
configuration `(h_S)`. Run: 3000 steps per `lam` (settled to `1e-10` in every case).

`b`'s outgoing message is `m_R + m_S = (h_S, 1)`: it carries `S`'s signal as the brief
assumed, and `R` receives `(h_S, 1)` through `b` (`a` delivers `0_M` to `b`). The composite
cost is `(1/2)(nu - h_S)^2 + lam (1/2) h_S^2`, settling at `h_S = nu / (1 + lam)`.

| lam | h_S settled | r_R(received) | r_R(0) | margin r_R(0) - r_R(received) | member |
|---|---|---|---|---|---|
| 0 | 1.000000 | 0.000000 | 0.5 | 0.500000 | yes |
| 1 | 0.500000 | 0.125000 | 0.5 | 0.375000 | yes |
| 10 | 0.090909 | 0.413223 | 0.5 | 0.086777 | yes |
| 100 | 0.009901 | 0.490148 | 0.5 | 0.009852 | yes |

`r_R(received) = (1/2) nu^2 (lam/(1+lam))^2` to `1e-10`; the margin is
`(1/2) nu^2 (1+2 lam)/(1+lam)^2`.

## What the numbers support, and what they do not

They support the residual as the right quantity for *need*: `C`'s bond-like cost, which it
zeroes by moving itself, gives a constant residual, and `C` is never a member although it
moves and receives exactly what `R` receives; `R`'s food cost gives a non-constant residual,
and `R` becomes a member only once `h_S` has descended (at the start it reads `0`, a tie with
`r_R(0)`, so membership is a property of a state, as the definition says). Test 2 supports
the leaving picture at the level of direction: at every step `R` moves toward `B` and away
from `A`, its residual falling. Two things the numbers do not support. First, the brief's
"lose membership as `lam` grows": at every finite `lam` run, `r_R(received) < r_R(0)`
strictly, so `R` stays a member; only the margin shrinks, to `0.0099` at `lam = 100`,
reaching `0` in the limit alone. A definition that wants "outweighed" to end membership
needs a threshold or a comparison of margins, which the strict inequality does not give.
Second, with the mean readout the need has no *need point* in the brief's sense: `argmin_e
r_R(e)` is the whole ray `{(nu w, w, p) : w > 0}`, not a point, and in test 2 no
configuration realizes it, so `R` passes `B` at step 51 and keeps going, the force decaying
without a settled state. The paper's Example 1 has a settled reading only because its cost
is on the weighted input (signal and weight), not on the mean.

## Adapted from the brief

- `readout` is `h / w` extended by `0_M |-> 0`. The library's `mean` refuses `w = 0`, and
  `r(0)` evaluates the cost at `0_M`; the brief's `r_R(0) = (1/2) nu^2` already assumes this
  extension. It is stated in the test's docstring. With this readout inside the cost the
  reader leaves the paper's affine-quadratic family (as `attention.py` warns and
  `test_curious_reader.py` already does); in tests 1 and 3 the received weight is constant
  `1` so the cost is quadratic in `h_S` there, in test 2 it is not.
- Test 1's parent is the brief's pure placement, so nothing "draws `R` and `S` together":
  identity transfers ignore position, and no position coordinate receives any force. The
  parent has the trivial parameter. `C` sends `0_M`; its verdict does not depend on what it
  sends or receives, since its residual is constant.
- Test 2 freezes the sources' positions as well as their signals (mobility `0`), so that
  the distances read as the reader's motion alone; the sources still receive their
  covectors (reported above). The run stops at 300 steps by choice, not by settling.
- Test 3's outer parent is unary (no dummy component), which the operad allows; the
  composite is built by two `compose_seq`s rather than `substitute`, the two being tested
  equal in `test_attention_operad.py`. The per-`lam` mobility is one value, `0.01`, so that
  the `lam = 100` step (`1 - 0.01 * 101 = -0.01`) is stable; 3000 steps settle `lam = 0`.
- Membership uses a `1e-9` numerical margin on the strict inequality (so that two equal
  residuals computed by descent, as `C`'s are, compare as equal).
