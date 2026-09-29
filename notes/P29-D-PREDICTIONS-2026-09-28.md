# P29 (d): smoke per watt — predictions fixed before construction code

Claude Opus 5.5, subagent of 8f53ae48, 2026-09-28. Status: predictions for a proposed model.
Design: Claude Opus 5.5 (1M), a631f37c, *proposed* (book plan §8.2 P29 (d)). David's claim under
test: "the plant is not gonna want to make less power … it's gonna need to make less smoke."
Written before any change to `arrangements/town.py`; the numbers in "Analytic expectations" come
from the algebra of the designed plant cost alone (scratch root-finding), not from the town.

## Design (as given; not redesigned)

- Plant: one box, configuration (x, c): output x and cleanliness c in [0,1). Power x leaves the
  town (no box consumes it). Smoke s = (1-c) x goes to the townspeople.
- Plant cost, descended in (x, c) only: (1 + p (1-c)^2) x^2/2 - benefit*x + k c^2/2, where p
  is the price/premium on its input wire, exactly as in (a)–(c), and k is the cost of cleaning.
- Townspeople: forcing beta*s^2/2 - recovery*h (the existing one, s in place of x).
- Court, insurer, the mathematician: unchanged, except that the observation memory records smoke,
  so exposure is s^2/2 and the learned coefficient is again beta. Operating surplus becomes
  benefit*x - x^2/2 - k c^2/2; the premium bill is premium*s^2/2. Harm = integral of beta*s^2/2.
- Implementation: a `clean=True` option of `build`, `Settings.clean_cost` (k) and
  `Settings.clean0` (initial c, 0). With `clean=False` every existing stage is unchanged.
- Initial c = 0 in every run. Other constants as in (a)–(c): benefit 1, beta 2, recovery 1,
  dt .02, delay 300 steps, horizon 30, learn rate 40, quote rate 2.

## The four predictions (from the brief, verbatim in substance)

1. At p = 0 (stage 1, no return route) the plant settles at c = 0, x = 1, whatever beta; it
   never cleans however sick people get. Recorded at beta = 0.5, 2, 8: settled x, c, s, h.
   Expected h = beta/2: 0.25, 1, 4.
2. At p > 0 it settles with c > 0, and for small k most of the harm reduction comes from c, not
   from x. The split is measured two ways (below).
3. With the mathematician's account feeding the insurer (stage 4), the plant starts cleaning
   before the first claim is paid.
4. Power delivered to other towns falls by less than in the one-dial town; for small k it barely
   falls. "Stage (c)" is read as the one-dial run of the same institution; the headline
   comparison is her account (build stage 4), with stages 2 and 3 also reported.

Able to fail, from the brief: (2) at moderate k; the non-convex liability term trapping descent
(Hessian of the plant's cost at the settled point); the k at which cleaning stops paying.

## Measures, fixed now

- **Harm split, settled.** With u = 1-c and baseline s = 1 (p = 0): log split, share from
  cleaning = ln u / ln s; and Shapley split of 1 - s^2 between the two factors, cleaning's part
  (1-u^2)(1+x^2)/2. Both are exactly 1/2 when x = u.
- **Harm split, cumulative** over the run, against the no-return town's output x0(t): cleaning's
  part of x0^2 - u^2 x^2 is (1-u^2)(x0^2+x^2)/2, output's (x0^2-x^2)(1+u^2)/2 (Shapley).
- **Power fall** = 1 - (integral of x)/(integral of x in the no-return town), same horizon, and
  settled power x(30). "Barely falls" means settled power fall under 5%.
- **Starts cleaning** (prediction 3): c reaches half its value at t = 30 before the first claim is
  paid. Control: in stage 3 (historical insurance) c stays exactly 0 until the first claim.
- **Small k** = 0.1; **moderate k** = 1; large k = 10.

## Analytic expectations (algebra of the plant cost only; my additions, fixed before code)

Write y = 1-c. With benefit 1 the plant's cost is, up to a constant,
(x-1)^2/2 + k (y-1)^2/2 + p x^2 y^2/2. The liability p s^2/2 is symmetric in output and
dirtiness; the plant reduces s = x y by whichever factor is cheaper to move: output has stiffness
1 (in general benefit^2 over the production-cost coefficient), dirtiness has stiffness k.

A. **Threshold k* = 1, for every p.** At k = 1 the cost is symmetric under x <-> 1-c, so at the
   settled point x = 1-c and both splits are exactly 50/50. Cleaning supplies most of the harm
   reduction iff k < 1; shrinking does iff k > 1. Cleaning never stops entirely at p > 0: at c = 0
   the c-forcing is p x^2 > 0, because the quadratic cleaning cost makes the first bit free.
   So "stops paying" can only mean "stops being the main lever"; that is k* = 1.
B. **p = beta = 2 settled points** (unique; Hessian positive definite):
   k = 0.1: x .9954, c .9520, s .0478, cleaning's share .998 (log) / .995 (Shapley).
   k = 1: x = 1-c = .5898, s .3478, shares .500 / .500.  k = 10: x .3438, c .0231, s .3359, shares
   .021 / .029. One-dial town at p = 2: x = s = 1/3.
C. **Settled power is always higher than one-dial** at p > 0 (x = 1/(1+p y^2) > 1/(1+p)).
D. **Settled harm is not always lower than one-dial.** At p = 2 the two-dial settled smoke exceeds
   1/3 for k at and above about 1 (s^2 ratio 1.089 at k = 1, 1.015 at k = 10). The second dial
   lowers the plant's cost, which equals total social cost only when p = beta; it does not
   promise less harm. Expected, not a prediction under test.
E. **Convexity.** Stationary points are unique and minima for p < 4 at every k. For p > 4 there
   is a band of k with two minima (a clean big plant and a dirty small one, same smoke at k = 1)
   and a saddle between. At p = 8 the band is k in (0.770, 32/27 = 1.185). So at beta = 8 the
   descent can be trapped; which minimum it reaches depends on the path of p. Measured at
   beta = 8, k = 0.8, 0.9, 1.1 (informed plant, p = 8 from the start).
F. **A new way to fail (mine): cleaning starves her fit.** Her learning force scales with
   exposure^2 = (s^2/2)^2. If the plant cleans before her estimate reaches beta, exposure
   collapses and her estimate (so the premium) may stall below beta. At small k this could leave
   the premium well short of 2. Recorded: her final estimate and the premium at t = 30. The
   court learns from six-unit-old, dirty exposures and should reach beta.

## Tests and time

One file, `tests/test_town_clean.py`: (i) existing stages untouched and p = 0 table; (ii) settled
point, Hessian and split at informed p = 2 for k = 0.1, 1, 10; (iii) her account: cleaning before
first claim, stage 3 control; (iv) power versus one-dial; (v) p = 8 band. A refuted prediction is
marked xfail with its measured numbers, not loosened. Each town run of 30 units takes about 1–4 s
jitted (from (a)–(c)); the file should run under 60 s; cap 180 s. Existing four town files re-run
one at a time, cap 180 s each.
