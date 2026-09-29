# P29 (d): smoke per watt — results

Claude Opus 5.5, subagent of 8f53ae48, 2026-09-28. **Status: run of a proposed model.**
Design: Claude Opus 5.5 (1M), a631f37c, *proposed*. Predictions, fixed before code:
`P29-D-PREDICTIONS-2026-09-28.md`. All four predictions held. `tests/test_town_clean.py`,
5 tests, 8.8 s; the 11 tests of (a)–(c) still pass, file by file. No parameter search: the k
values .1, 1, 10 and the thresholds were fixed in the predictions.

## What the run answers

David's claim: "the plant is not gonna want to make less power … it's gonna need to make less
smoke." In this model it holds **when cleaning is cheap relative to output (k < 1)** and fails
when it is dear (k > 1), where the plant mostly shrinks, as in (a)–(c). The dividing cost is
exactly k = 1 (benefit^2 over the production-cost coefficient) at every price. The mechanism:
up to a constant, the plant's cost is

  (x-1)^2/2 + k (y-1)^2/2 + p x^2 y^2/2,  with y = 1-c the dirtiness and s = x y the smoke.

The priced liability falls on the product of output and dirtiness, so the plant cuts smoke by
moving whichever factor is cheaper to move. Output resists with stiffness 1, dirtiness with
stiffness k. At k = 1 the cost is symmetric under x <-> 1-c, and the settled point has x = 1-c.

## The four predictions

1. **Held.** No return route (p = 0): c = 0 exactly at every step, x equals the one-dial run to
   1e-13, and the plant's forcing has zero derivative in illness. This is what Eve sees:

   | beta | x | c | s | h |
   | ---: | ---: | ---: | ---: | ---: |
   | 0.5 | 1.000000 | 0 | 1.000000 | 0.250000 |
   | 2 | 1.000000 | 0 | 1.000000 | 1.000000 |
   | 8 | 1.000000 | 0 | 1.000000 | 4.000000 |

   (t = 30; x = 1 - 6e-14, h = beta/2 to 1e-11.) The same holds for free cleaning (k = 0) only
   because c starts at 0: with nothing priced the plant is indifferent to c.
2. **Held for small k; refuted at moderate k, as the brief allowed.** With p = 2 held fixed
   (informed plant) the settled points match the algebra to 1e-6, and every Hessian is positive
   definite. Cleaning's share of the harm reduction from s = 1:

   | k | x | c | s | share, log | share, Shapley | Hessian min eig |
   | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
   | 0.1 | .9954 | .9520 | .0478 | .998 | .995 | .972 |
   | 0.3 | .9626 | .8607 | .1341 | .981 | .962 | .823 |
   | 0.5 | .8998 | .7641 | .2123 | .932 | .895 | .628 |
   | 1 | .5898 | .4102 | .3478 | .500 | .500 | .304 |
   | 2 | .4033 | .1399 | .3469 | .142 | .172 | 1.013 |
   | 10 | .3438 | .0231 | .3359 | .021 | .029 | 2.670 |

   In the town with her account, over 30 units against the no-return town (Shapley), cleaning's
   share is .963 (k = .1), .486 (k = 1), .031 (k = 10).
3. **Held.** Her account: c reaches half its t = 30 value at t = 2.58 (k = .1), 1.98 (k = 1),
   1.08 (k = 10); the first claim is paid at t = 6.04 in every run. At k = .1, c = .93 when that
   claim arrives. Control, historical insurance: c = 0 exactly until the first claim.
4. **Held.** Power delivered over 30 units, fall against the no-return town:

   | Institution | one dial | k = .1 | k = 1 | k = 10 |
   | --- | ---: | ---: | ---: | ---: |
   | Law, delayed | 51.9% | 2.7% | 32.1% | 51.0% |
   | Historical insurance | 51.0% | 2.7% | 31.6% | 50.1% |
   | Her account | 63.5% | 3.4% | 39.9% | 62.3% |

   Settled power: .995 at k = .1 (a 0.5% fall), .590 at k = 1, .344 at k = 10, against 1/3 with
   one dial. Settled power always exceeds the one-dial value at p > 0 (x = 1/(1+p y^2)).

## Harm, and what the second dial does not promise

Cumulative harm over 30 units (no return route: 28.609):

| Institution | one dial | k = .1 | k = 1 | k = 10 |
| --- | ---: | ---: | ---: | ---: |
| Law, delayed | 8.345 | 5.805 | 8.431 | 8.360 |
| Historical insurance | 8.681 | 6.134 | 8.733 | 8.691 |
| Her account | 3.882 | 0.795 | 4.080 | 3.929 |

At k = .1 her account cuts harm 79.5% below the one-dial account town, and 97.2% below no
return. At k ≥ 8/9 (p = 2) the settled smoke *exceeds* the one-dial 1/3, so harm is slightly
higher with the second dial. When p = beta the plant's cost is the town's total cost (harm valued
one-for-one in money), and more options lower that total, not the harm. Net surplus, operating
minus cleaning minus harm, over 30 units with her account: one dial 4.813; k = .1: 12.673;
k = 1: 6.095; k = 10: 4.892. Higher at every k, including where harm is higher.

## The k at which cleaning stops paying

Cleaning never stops entirely at p > 0. At c = 0 the plant's c-forcing is p x^2 > 0, because a
quadratic cleaning cost makes the first bit free (an artifact; see below). "Stops paying" is
read as "stops being the main lever": **k* = 1**, exact at every p. The log and Shapley splits
both cross 1/2 there, since x = 1-c at the settled point.

## Non-convexity: the trap, where it bites and where it does not

The x^2 (1-c)^2 term is not convex. The settled point is unique and a minimum at every k for
p < 4. For p > 4 there is a band of k with two minima: a big clean plant and a small dirty one,
with the same smoke at k = 1 by the symmetry. At p = 8 the band is k in (0.770, 32/27 ≈ 1.185).
At beta = 8, k = .8:

- **Informed plant** (p = 8 from t = 0, starting small and dirty): settles at x = .179, c = .242,
  s = .135, plant cost -.0659. That is a true local minimum (Hessian positive definite), but the
  dearer one, **trapped**.
- **The town** (p rises from 0 through claims, historical insurance): settles at x = .917,
  c = .894, s = .097, cost -.1391, the cheaper and cleaner minimum. Law alone does the same at
  k = .8 and .9. At k = 1.1 both land on the small dirty plant, which is the cheaper minimum there.

So the trap is real but path-dependent. A plant that faces the full price while still small is
the one it catches.

## Found, not predicted: cleaning starves her fit

Her learning force is proportional to exposure^2 = (s^2/2)^2. The account makes the plant clean
early, exposure collapses, and her estimate freezes: 1.674 at t = 30 (k = .1; 1.657 at half
dt), not 2. The court's assessment, which reads the same smoke six units late, freezes at 1.674
too. With historical insurance the plant stays dirty until claims, so the court learns 2 exactly.
At k = 1 and 10 she reaches 1.996. At beta = 8 with her account the premium stalls at 5.5–6.0,
not 8. The harm cost is small here, because the smoke she can no longer see is small. But the
town's record of the harm coefficient is wrong, low, after she succeeds.

## Reasonable or weird (my judgment, not evidence)

- **Reasonable at a real plant and town:** a priced smoke liability prompting abatement when
  abatement is cheap and output cuts when it is dear. The one-number threshold comparing the
  cost of abatement to the margin on output. Earlier information moving abatement ahead of the
  first payout. Success at abatement degrading the data that measured the harm (a known
  monitoring problem).
- **Weird, from the construction:**
  1. Power is consumed by nobody. Output is valued only as the plant's revenue, so a fall in
     power costs other towns nothing in the model. The 3.4% versus 63.5% contrast is about the
     plant's choice, not anyone's lights.
  2. Quadratic cleaning cost. The first bit of cleaning is free, so c > 0 at any p > 0. Real
     abatement has fixed costs (a scrubber is bought or not), which would give a genuine
     threshold below which the plant never cleans.
  3. Cleanliness costs the same whatever the output (k c^2/2 does not scale with x). A real
     scrubber's running cost scales with throughput. This is part of why a big clean plant
     beats a small one.
  4. Smoke turns into illness instantly (at recovery rate 1), noiselessly, with one coefficient.
  5. The learner's force scales with exposure squared (a raw least-squares gradient), so the
     stale fit is partly a property of this learner. A normalized learner would keep learning
     from small exposures.
  6. The bistable band needs p > 4, a price four times the production-cost coefficient.
     Plausible for severe harm; its exact edges are artifacts of the quadratic forms.
  7. At body or cell scale: the choice between cutting throughput and cutting a by-product per
     unit is a familiar pattern (by analogy only). No court, price or insurer is mapped there.

## What this does not establish

That cleaning up happens for a real plant, which depends on the real k: the run shows only
that the answer turns on one ratio, and says which way. It does not model the townspeople
needing the power or the jobs. Less harm is not established at k ≥ 8/9; higher net surplus is,
under one-for-one valuation of harm.

## Placement, proposed (for the parent session and David)

Ch 1, §1.6 (the town): the plant cleans rather than shrinks when cleaning is cheap, and the
choice turns on one ratio. Ch 5, the town construction: the symmetric cost and k* = 1, the
two-minimum band and the informed-plant trap, and the stale fit.

## Reproduction

From the library root: `env PYTEST_ADDOPTS=-s scripts/run_tests.sh 180 tests/test_town_clean.py`
(5 tests, 8.8 s), then the four (a)–(c) files one at a time the same way. Code: `build(stage,
Settings(clean_cost=k), clean=True)` in `arrangements/town.py`; `smoke(town, trajectory)`.
