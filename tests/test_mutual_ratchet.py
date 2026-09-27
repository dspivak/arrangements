"""The ratchet on a collective whose parts supply each other: shock, disperse, re-form.

The book plan's P21 stage (b) (AGENTDOCPLAN.md §8.2), from David, 2026-09-26, reading the
stage (a) run: "don't the parts know that they are actually getting stuff they need from each
other? like i can only do my job if i get what i need from you. is that not in there?" -- in
the cascade, no: every part there is supplied by the fund and by the elsewhere, never by
another part (§5.9 item 22). And: "if the paymaster leaves, other parts can pay for their
needs." Stage (a)'s seven answers (R1-R7) are the thing to compare against; each prediction
below says which of them it puts at risk.

**What has to be in one construction**, and was not in any earlier one: item 12's pair has
mutual need but no placements, so nothing in it can disperse; item 18's collective has
placements but its parts do not feed each other. Here: placements, one kind of supply, each
part gated on what arrives from the others, an elsewhere at a distance, and the fund
removable.

**The pieces** (``arrangements/mutual_supply.py``, written after this docstring). On
``collective.py`` with ``H = R^2`` as in the cascade: channel 1 carries *supply*, channel 2
carries *product*. The parent is cost-free, so a placement moves only by what its own part's
cost says about what arrives (the cascade's idiom); the parent's transfers already deliver to
each box the kernel-weighted sum of *every other* box's message and of the exterior's
(``spatial_transfers``, ``T_ii = 0``), so parts supply each other with nothing added to the
library.

* ``supplier``: a part holding an amplitude ``a`` (mobility ``mu_a``), sending
  ``((sigma a, a), 1)`` -- a share ``sigma`` of its activity passed on as supply to whoever is
  near it, and its product upward -- and carrying the cost

      k(a, e) = (b/2) a^2 - (alpha a + beta) I(e_1),    I(r) = A r / (K_s + r),

  where ``e_1`` is the channel-1 total that arrives. So ``I`` is item 12's Michaelis-Menten
  gate moved onto the receiving side: a part converts what arrives into its own sustenance,
  and what it passes on is proportional to its own activity. A part with nothing arriving has
  no drive at all and only its dissipation remains, which is item 12's wiring condition
  (no ungated path to a part) in a collective: the *only* ungated supplies here are the fund's
  wage at the origin and the elsewhere's offer at distance ``d``, and both are removable.
* ``source``: the elsewhere, the cascade's frozen part offering ``s`` on channel 1 at distance
  ``d``, weight 0.
* the fund: the cascade's ``fund_layer`` and ``market``, unchanged, wrapping the collective;
  or, removed, the bare collective driven by a constant exterior supply ``g`` at the origin
  (``cascade.supplied``), which is stage (a)'s clamped wage exactly.

**Four choices, said as choices.** (i) ``I``'s Michaelis-Menten shape is item 12's model of
metabolism, not a derivation; (ii) ``sigma`` is one number, so there is one kind of supply and
a part cannot need a particular other part (the plan's stage (c) is the web of kinds);
(iii) the elsewhere is an ungated supply *at a place*, so a part alone there sustains itself --
that is what makes dispersal possible at all, and it is the cascade's geometry, not something
added here; (iv) the fund runs with ``record=None``, since the exact cancellation of the wage's
retroactive pull needs each part's arriving total and the fund sees only the collective's
onward message. So the balance is propped up by the need of those it pays (``cascade.py``'s
own docstring); the size of that term at each equilibrium reported is printed and is small
where the gate saturates.

**The landscape, derived by hand before the run, and it is stage (a)'s.** A part's placement
feels only its own cost, and ``I`` is increasing, so the stationary placements are exactly the
stationary points of the supply that arrives,

    r(p) = G kappa(p) + S kappa(p - d) + sigma sum_{j != i} a_j kappa(p - p_j),

*the gate changes the rate, never the place*. With everyone clustered at the two places, this
is stage (a)'s ``h_1`` with ``g`` replaced by

    G = g + sigma sum_{j at home} a_j        (what home offers)
    S = s + sigma sum_{j at the elsewhere} a_j   (what the elsewhere offers),

so ``phi``, ``phi_min = 0.367959``, ``phi_max = 2.717697`` and the hysteresis ratio
**7.3859** at ``d = 2.5, tau = 0.8`` are unchanged from stage (a). What has changed is that
``G`` and ``S`` are no longer dials: they are what the parts in each place are doing.

**The amplitudes, self-consistently.** ``m`` parts clustered where the ungated supply is ``g``
hold ``a = A r / (K_s + r)`` with ``r = g + sigma (m - 1) a``. With no ungated supply the
nonzero root exists iff ``sigma (m - 1) A > K_s``, and then

    a*(m) = A - K_s / (sigma (m - 1)).

At ``A = 4, K_s = 1, sigma = 0.5``: ``a*(1)`` does not exist (a lone part dies), ``a*(2) = 2``,
``a*(3) = 3``, ``a*(4) = 10/3``. Where the local supply is ``s = 1``: ``2.0000, 2.8284,
3.2361, 3.4415`` at ``m = 1, 2, 3, 4``.

**The predictions, fixed here before any construction code existed. Each must be able to
fail.** ``alpha = b = beta = 1``, ``A = 4``, ``K_s = 1``, ``sigma = 0.5``, ``tau = 0.8``,
``d = 2.5``, ``s = 1``, as above.

(B1) *Mutual need, with a critical fragment size.* With the fund removed and the elsewhere out
of reach, a fragment of ``m`` parts holds a form iff ``sigma (m - 1) A > K_s``, and then every
amplitude is ``a*(m)`` to within 1%: a lone part dies (below 1e-3), and 2, 3, 4 parts hold
2.000, 3.000, 3.333. *Puts at risk*: nothing of stage (a)'s; this is the check that the
construction has what (a) lacked. Fails if a lone part holds a form, or if ``a*`` misses by
more than 1%.

(B2) *The paymaster can leave.* Stage (a)'s R1 says money cannot outbid the elsewhere and its
R4 that the fund's collapse sends every part to the elsewhere. Here: drain the fund past the
threshold that killed the cascade, and **nobody leaves** -- all four placements stay within
0.1 of home for 2*10^5 steps and the amplitudes fall only to the no-fund value ``a*(4) =
3.333`` within 2%. And the same collective runs with the fund layer removed altogether.
*Puts at risk*: R4 (the collapse) and the whole reading that re-forming must wait on a
paymaster. Fails if any placement passes 1.0 after the fund dies.

(B3) *One leaving takes the rest: the collective's threshold depends on its size.* With no
fund, the offer at which a cluster of ``m`` at home comes apart is
``s_c(m) = sigma (m - 1) a*(m) / phi_min``, within 8%: predicted **2.718, 8.153, 13.589** at
``m = 2, 3, 4``. So there is a wide band of offers, ``8.15 < s < 13.59``, at which a four-part
collective holds and a three-part one does not: *inside that band, losing one part disperses
the rest*. Fails if any ``s_c(m)`` misses by more than 8%, or if the band is empty.

(B4) *A fragment calls the rest home, and no paymaster is needed.* With the fund removed and
the offer back at ``s = 1``, a part at the elsewhere returns iff what home offers exceeds
``phi_max s``: ``G_home(m) = sigma m a*(m)`` is 2.000 at ``m = 2`` and 4.500 at ``m = 3``
against ``phi_max s = 2.718``, so **three at home call the fourth back and two do not** -- the
one at the elsewhere stays past 10^5 steps. *Puts at risk*: R1 and R4 -- in (a) nothing but the
fund could call a part home. Fails if a two-part fragment calls it back, or a three-part one
does not.

(B5) *The wage that hires someone away from a group is the group's supply, not the geometry.*
Stage (a)'s R2/R3: two critical wages set by the geometry alone, identical for ``n = 1`` and
``n = 4``. Here, with the wage clamped as a dial, the wage that brings a part home out of a
cluster of ``m`` sitting at the elsewhere is ``g_in(m) = phi_max (s + sigma (m - 1) a_src(m))``
within 5%: predicted **2.718, 6.561, 11.512, 16.747** at ``m = 1, 2, 3, 4``. It rises with
``m``, and its ratio to the departure wage is still 7.39. Fails if ``g_in`` is independent of
``m`` to within 5%, or if any value misses the closed form by more than 5%.

(B6) *A fixed point still forgets.* If every part ends at home, the re-formed collective is the
original: placements and amplitudes agree to 1e-4 and the second critical offer equals the
first to 0.5%. *Puts at risk*: nothing; R5 is predicted to stand. Fails if they differ.

(B7) *What re-forming can now carry is a configuration, and no door has to shut.* With the
placement mobilities spread eightfold and the elsewhere's offer raised for a window of ``W``
steps and then dropped back, there is a band of ``W`` for **each** of ``k = 1, 2, 3`` parts
left at the elsewhere, and every such split state persists 10^5 steps with both clusters
holding nonzero amplitudes -- with no fund at all, nothing closed behind anyone, and both
clusters stationary because ``phi_min < G/S < phi_max``. *Puts at risk*: R6, where the split
needed the door to shut (a survivor unable to pay the return wage) and gave only one value of
``k``. Fails if every window gives ``k = 0`` or ``k = 4``, or if a split state drifts -- a
cluster dissolving, or the two merging -- within 10^5 steps.

(B8) *Still a scar, and still no ratchet.* A collective re-formed short ``k`` parts comes apart
at a lower offer than the original, monotonically in ``k``, tracking ``s_c(4 - k)`` of (B3)
within 8%; and nothing that re-formed withstands a larger shock than the original. *Puts at
risk*: nothing; R6 (ii)'s scar is predicted to stand, in a collective held together the other
way. Fails if any re-formed state withstands a larger offer than the original -- which would be
the first ratchet in these runs.

**What happened** (2026-09-26, written after the runs; the predictions above untouched). The
file takes about eight minutes. Two things were found while building, before any prediction
was tested, and both are in tests of their own.

*(i) The composite generates a term the docstring's amplitude law has not got.* Every part's
cost is exact and the parent is cost-free, so composition pulls each part's cost back onto
what its *neighbours* send: the forcing on part ``i``'s amplitude is

    xi_i = -b a_i + alpha I(r_i) + sigma sum_{j != i} (alpha a_j + beta) I'(r_j) kappa_ij,

the last term being the others' need for what ``i`` sends. It is a tenth of the amplitude at
the settled state (3.8679 against the 3.4887 the docstring's law gives), and it is why the
docstring's ``a*(m)`` and its threshold are both wrong. More: the kernel is symmetric, so the
*whole* arrangement is exact, and amplitudes and placements alike ascend one potential
``U = sum_j [(alpha a_j + beta) I(r_j) - (b/2) a_j^2]`` -- checked against the library's own
forcing at a scrambled state, agreeing to 1e-9. A part is held by the others not only by what
it receives from them but by their need for what it sends, and that is in ``U`` and not in the
arriving supply. Every correction below comes from this one fact.

*(ii) ``beta`` was set to 0.* With the cascade's ``beta = 1`` the reciprocity term is nonzero
at ``a = 0``, so two parts with *nothing* arriving drive each other up from rest, to 3.7588:
production from nothing, and not what the construction is for. At ``beta = 0`` the zero state
is a fixed point. The price is stated where it is paid: a part with nothing arriving has no
placement force either, so it cannot crawl toward supply.

(B1) **Held in its verdict, refuted in its numbers and in its law.** A lone part dies and 2,
3, 4 parts hold a form -- but at 3.46410, 3.82843, 3.91532, not the 2.000, 3.000, 3.333
predicted. The true law is ``a = I(r) + sigma (m-1) a I'(r)``, matched to 1e-4 at every size
and both supplies, and the true threshold is ``2 A sigma (m - 1) > K_s``, a factor of two from
the docstring's. The discriminating case, where the two rules disagree, is ``sigma = 0.1,
m = 3``: the docstring says dead, the corrected law says 1.89898, and the run says 1.89898
from a seed of 0.01 and from 3.9. The critical ``sigma`` at ``m = 3`` is 0.06252 against the
closed form 0.06250 (the docstring's rule would give 0.125).

**The line this threshold draws, which was not foreseen at all.** Below it a part can only
*pass on* what something outside supplies; above it a collective holds a form with nothing
arriving -- a pair lights itself from a seed of 1e-12 and settles at 3.4641. A super-critical
loop is a free lunch, and it is the same modelling move as the cascade's market paying twice
the product: it stands for an ambient gradient the model does not draw. Everything below runs
above the line, at ``sigma = 0.5``.

(B2) **Refuted in its letter, answered in what it was about.** With the wage cut from 4 to 0
the collective does leave home -- home then offers nothing -- but it walks to the elsewhere as
a body and *keeps its form*: 3.8266 before, 3.8686 after, +1.1%. The control, the same run at
``sigma = 0``, where each part is paid and supplies nobody, is left with exactly what the
elsewhere pays, ``I(s) = 2.0000`` against 3.2033, **-37.6%**. David, 2026-09-26: "if the
paymaster leaves, other parts can pay for their needs." In this model they can.

(B3) **Refuted twice, and both corrections are the finding.** There are two thresholds, not
one. The **body** threshold is stage (a)'s exactly: what the parts pass each other travels
with them, so it puts no gradient on the cluster's own position, and the collective moves at
``s = 2.7161`` against ``g/phi_min = 2.7177``, *the same for one part and for four*.
**Defection does not happen at all**: at offers of 10, 20, 40 and 1000 the one free part
settles at 0.1085, 0.1835, 1.1061 and 1.2951, each within 0.006 of the peak of ``U`` along its
placement, and it never reaches the elsewhere. A bigger offer slides it partway out and it
stops there, relaying: far enough to pick the offer up, near enough to keep passing it back.
The prediction had it leaving above 18.49.

(B4) **Refuted in its number; the correction is the stronger claim.** With nobody at home the
stray stays at the elsewhere (2.4409, the landscape's peak 2.447) -- stage (a)'s bistability,
at the wage ``g = s = 1`` that (a)'s R1 called unable to outbid the elsewhere. **One part at
home is enough** to bring it back (0.0598), where the prediction needed two: what calls it is
home's need for what it sends, not what home would give it.

(B5) **Refuted as stated, and split like (B3).** The whole group comes home above
``g = 2.7161`` for ``m = 1, 2, 4`` alike -- ``phi_max s = 2.7177``, and stage (a)'s R3 stands
for the body. One part out of two needs 12.6243 (the landscape 12.5811) and one out of four
needs 39.0706 (39.0850), against the 7.3630 and 18.4882 predicted. **Moving a group of four
costs 2.72; taking one part out of it costs 39.07, fourteen times as much.**

(B6) **Held, exactly.** A boom to ``s = 12`` takes the collective to the elsewhere; when it
ends, the same wage brings it home to placements 0.013026 and amplitudes 3.826558, the first
equilibrium to every digit printed.

(B7) **Refuted: there is no band, because there is no split.** With mobilities spread
eightfold and booms of 200 to 20000 steps, every window gives 0 or 4 at the elsewhere, and the
spread across the four during the crossing never exceeds 0.024 -- the fast are held by the
slow and the slow pulled by the fast. The control at ``sigma = 0`` strews them out: spread
2.172 at the same window. A split *is* stable, but what makes it stick is distance, not the
wage: two and two placed apart merge at ``d = 2.5`` (``kappa = 2.0e-2``) and hold their places
at ``d = 4`` (``kappa = 4.5e-5``), each pair at 3.4186. Nothing in this model puts them there.

(B8) **Refuted for the collective, held for the fund.** The offer that moves a collective does
not depend on its size (B3), so a collective short a part is exactly as hard to move: **no
scar**. With the fund back, the scar is back -- the drain that kills it is 508.95 at four
parts and 342.26 at three, **-32.8%** -- because the missing part's product is what the
balance was made of. What stage (a) weakened was the paymaster, not the collective.

*Asked after the results* (David, 2026-09-26, which case is real: "either one, but i bet it's
many order of magnitude more common to pass intra and many order of magnitude bigger loads or
more significant when coming from the outside"). In this model the two are the same thing:
every quantity that decides anything depends on ``sigma (m - 1)``, the total that arrives, and
on nothing else. Measured: the critical share is 0.06251, 0.03127, 0.01562 at ``m = 3, 5, 9``,
so ``sigma (m - 1) = 0.125`` throughout; and three, five and nine parts passing the same total
hold the identical amplitude 3.464102. Many small passes and few large ones are
indistinguishable here -- the same fact as the turnover run's swappable contributors (§5.9
item 13), since the cost is on a total.

**Which of stage (a)'s seven answers change.** R1 (money cannot outbid the elsewhere):
**changed** -- one part at home calls a stray back at the wage that could call nobody in (a).
R2 (two critical values, set by the geometry, ratio 7.39): **stands**, for the body, to 0.06%.
R3 (the same values for one part and for four): **stands for the body, fails for a part in a
group** -- 2.72, 12.62, 39.07 at one, two and four. R4 (the loop closes when the fund can
outbid, and the wage must last): **not retested**; nothing here runs the fund's recovery. R5
(a fixed point forgets): **stands**, to six decimals. R6 (membership is the only memory, and
it needs the door to shut): **changed** -- there is no membership memory because nothing is
lost; no shock here takes a part away. R7 (one condition, two routes): **stands**, the offer
route and the wage route giving 2.7161 both ways (``phi_min phi_max = 1``).

*What it says* (the agent's, *proposed*, and scoped to a collective above the line): **what
mutual supply changes is not where a collective goes but whether it can be taken apart.**
Every threshold about *moving* is stage (a)'s, unchanged and independent of size; every
threshold about *removing one part* is multiplied by what the others give it, and defection by
offer is impossible at any price. So the question P21 asks -- does re-forming remember? -- has
no purchase on this collective: it is not broken by shocks, only moved, and when it is moved
back it is what it was. The one memory such a collective could carry is a split, and a split
holds only once the pieces are out of each other's reach, which is item 18's finding one level
up: what makes dispersal stick is distance, not the shock.

*Weird, and where* (the agent's judgment, not evidence). Reasonable at the scale of a firm or
a town: you can move the whole of it more cheaply than you can hire one person out of it, and
a paymaster's collapse need not end it. Weird at every scale: the collective walks to the
elsewhere *as a body* the moment home stops paying, because a collective here is attached to
its members and to supply, never to a place -- no land, no houses, nothing sunk. Weird at the
scale of a cell: above the line, parts hold a form with nothing arriving, which is perpetual
motion unless it is read as an ambient gradient the model does not draw. Backwards at the
scale of a body: with ``beta = 0`` a part with nothing arriving cannot move at all, so a
starving part cannot go looking -- hunger drives search in the world, and not here.
"""

import jax
import jax.numpy as jnp
import numpy as np

# Construction (everything below this line was written after the docstring above).

from arrangements.cascade import EMPTY, closed_step, gate, read
from arrangements.collective import run, step_fn
from arrangements.mutual_supply import (
    amplitudes, arriving, bare, funded, intake, placements, supplied, suppliers,
)

ALPHA, B, BETA = 1.0, 1.0, 0.0    # beta = 0: see `test_why_beta_is_zero`
A_MAX, K_S = 4.0, 1.0             # the gate I(r) = A r / (K_s + r)
SIGMA = 0.5                       # the share of its activity a part passes on as supply
TAU, D_SRC, S_SRC = 0.8, 2.5, 1.0     # the cascade's geometry, unchanged
MU_A, MU_P = 0.1, 0.02            # amplitude and placement mobilities
KW = dict(alpha=ALPHA, b=B, beta=BETA, A=A_MAX, K_s=K_S, mu=MU_A)
PHI_MIN, PHI_MAX = 0.36795858803693265, 2.717697133623168


# ---------------------------------------------------------------------------
# The closed forms, all derived by hand (docstring, and the corrections below).
# ---------------------------------------------------------------------------


def kappa(x, tau=TAU):
    return np.exp(-np.asarray(x, float) ** 2 / (2 * tau))


def I(r):
    return A_MAX * r / (K_S + r)


def dI(r):
    return A_MAX * K_S / (K_S + r) ** 2


def cluster_amplitude(m, g, sigma=SIGMA):
    """The self-consistent amplitude of ``m`` parts clustered where the ungated supply is
    ``g``, *including the reciprocity term the composite generates* (the docstring's
    ``a*(m)`` omits it, and is wrong by it):

        a = I(r) + sigma (m - 1) a I'(r),    r = g + sigma (m - 1) a,

    the second term being the pullback of the *others'* cost onto this part's amplitude --
    each neighbour's need for what it sends drives it to send more. At ``g = 0`` the nonzero
    root is ``a = (x - K_s)/c`` with ``c = sigma (m - 1)`` and
    ``x = (A c + sqrt(A^2 c^2 + 4 A c K_s))/2``, and it exists iff ``2 A c > K_s``."""
    c = sigma * (m - 1)
    if c == 0.0:
        return I(g)
    f = lambda a: -a + I(g + c * a) + c * a * dI(g + c * a)
    lo, hi = 1e-12, 100.0
    assert f(hi) < 0
    if f(lo) < 0 and g == 0.0:
        return 0.0
    for _ in range(300):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


def sigma_critical(m):
    """``2 A sigma (m - 1) = K_s``: below it a fragment of ``m`` cannot hold a form with no
    ungated supply (it can only pass on what arrives); above it, it can hold one from
    nothing."""
    return K_S / (2 * A_MAX * (m - 1))


def offers(m_here, m_there, g, s=S_SRC, sigma=SIGMA):
    """``(G, S)``: what each place offers a part *at* it -- the ungated supply there plus
    what the others there pass on. ``rho = G/S`` is stage (a)'s ratio."""
    return (g + sigma * (m_here - 1) * cluster_amplitude(m_here, g, sigma),
            s + sigma * (m_there - 1) * cluster_amplitude(m_there, s, sigma))


# ---------------------------------------------------------------------------
# Running.
# ---------------------------------------------------------------------------


def setup(n, *, positions, mobilities=MU_P, sigma=SIGMA, s=S_SRC, d=D_SRC, a0=1.0, tau=TAU):
    mob = [mobilities] * n if np.isscalar(mobilities) else list(mobilities)
    return bare(suppliers(n, a0, sigma=sigma, **KW), positions=list(positions),
                mobilities=mob, tau=tau, s=s, d=d)


def settle(C, steps, g, state=None):
    e = supplied(g)
    return run(C, steps, state=state, e_ext=e)


def look(C, state, g):
    """``(placements, amplitudes, arriving supplies, largest one-step move)``."""
    e = supplied(g)
    q = np.asarray(C.arr.P.position(state))
    q2 = np.asarray(C.arr.P.position(run(C, 1, state=state, e_ext=e)))
    return (np.asarray(placements(C, q)), np.asarray(amplitudes(C, q)),
            np.asarray(arriving(C, q, e)), float(np.max(np.abs(q2 - q))))


def bisect(pred, lo, hi, iters=13):
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if pred(mid) else (mid, hi)
    return (lo + hi) / 2


def runner(C, steps, g):
    step = step_fn(C, supplied(g))
    return jax.jit(lambda s: jax.lax.fori_loop(0, steps, lambda i, x: step(x), s))


# ---------------------------------------------------------------------------
# Not predictions: the composite's own law, and two things found while building.
# ---------------------------------------------------------------------------


def test_the_law_the_composite_actually_obeys():
    """Not a prediction: what the settled state satisfies, checked term by term. The
    docstring's amplitude law is *missing a term*, and the run is how it was found. The
    composite's forcing on part ``i``'s amplitude is

        xi_i = -b a_i + alpha I(r_i) + sigma sum_{j != i} (alpha a_j + beta) I'(r_j) kappa_ij,

    the last term being the pullback of the *others'* costs onto what ``i`` sends: each
    neighbour's need for ``i``'s supply drives ``i`` to make more of it. Composition
    generates it; nothing here put it in. The placement law of the docstring is untouched by
    it: a placement is stationary where the arriving supply is."""
    C = setup(4, positions=[0.0] * 4)
    st = settle(C, 60_000, 1.0)
    p, a, r, mv = look(C, st, 1.0)
    K = np.array([[kappa(p[i] - p[j]) for j in range(4)] for i in range(4)])
    xi = np.array([-a[i] + I(r[i]) + SIGMA * sum(a[j] * dI(r[j]) * K[i, j]
                                                 for j in range(4) if j != i) for i in range(4)])
    naive = np.array([I(r[i]) for i in range(4)])
    # and the placement: stationary where r'(p) = 0
    rprime = np.array([
        -(1.0 * p[i] / TAU) * kappa(p[i])
        - (S_SRC * (p[i] - D_SRC) / TAU) * kappa(p[i] - D_SRC)
        - SIGMA * sum(a[j] * ((p[i] - p[j]) / TAU) * K[i, j] for j in range(4) if j != i)
        for i in range(4)])
    print(f"\n  placements {np.round(p, 6)}  amplitudes {np.round(a, 6)}  arriving {np.round(r, 5)}"
          f"\n  xi (with the reciprocity term) {np.max(np.abs(xi)):.2e};"
          f" without it the residual is {np.max(np.abs(-a + naive)):.4f}"
          f"\n  r'(p) {np.max(np.abs(rprime)):.2e}   one-step move {mv:.1e}")
    assert np.max(np.abs(xi)) < 1e-6 and np.max(np.abs(rprime)) < 1e-6 and mv < 1e-9
    assert np.max(np.abs(-a + naive)) > 0.3          # the missing term is not small
    # the closed form of `cluster_amplitude`, at the supply the cluster actually receives
    assert abs(a[0] - cluster_amplitude(4, 1.0 * kappa(p[0]) + S_SRC * kappa(p[0] - D_SRC))) < 2e-3


def test_why_beta_is_zero():
    """Found while building, before any prediction was tested. The docstring's ``beta = 1``
    (the cascade's worker, whose placement then climbs toward supply even when it is doing
    nothing) makes the reciprocity term ``sigma (alpha a_j + beta) I'(r_j)`` nonzero at
    ``a = 0``: two parts with *nothing* arriving drive each other up from rest. With
    ``beta = 0`` the zero state is a fixed point and a part with nothing arriving stays
    dead, which is what the construction is for. The price, stated: a dead part has no
    placement force at all, so it cannot crawl toward supply."""
    out = {}
    for beta, a0 in ((1.0, 0.0), (0.0, 0.0), (0.0, 1e-12)):
        kw = dict(KW, beta=beta)
        C = bare(suppliers(2, a0, sigma=SIGMA, **kw), positions=[0.0, 0.0],
                 mobilities=[0.0, 0.0], tau=TAU, s=S_SRC, d=50.0)   # kappa(50) underflows to 0
        out[(beta, a0)] = float(np.asarray(
            amplitudes(C, C.arr.P.position(settle(C, 60_000, 0.0))))[0])
    print(f"\n  two parts, nothing arriving at all: beta = 1 from rest -> {out[(1.0, 0.0)]:.4f},"
          f" beta = 0 from rest -> {out[(0.0, 0.0)]:.2e},"
          f" beta = 0 from a seed of 1e-12 -> {out[(0.0, 1e-12)]:.4f}")
    assert out[(1.0, 0.0)] > 1.0 and out[(0.0, 0.0)] == 0.0
    assert out[(0.0, 1e-12)] > 1.0     # and a super-critical loop lights itself from any seed


# ---------------------------------------------------------------------------
# (B1) Mutual need, and what a fragment can hold with nothing arriving.
# ---------------------------------------------------------------------------


def alone(m, sigma, a0, steps=60_000):
    """``m`` parts clustered with *nothing* arriving: no wage, and the elsewhere pushed to
    distance 50, where the kernel underflows to zero. Placements frozen."""
    C = bare(suppliers(m, a0, sigma=sigma, **KW), positions=[0.0] * m,
             mobilities=[0.0] * m, tau=TAU, s=S_SRC, d=50.0)
    return float(np.asarray(amplitudes(C, C.arr.P.position(settle(C, steps, 0.0))))[0])


def test_B1_mutual_need_and_the_critical_fragment():
    """(B1). Held in its verdict at ``sigma = 0.5`` (a lone part dies, 2, 3, 4 hold),
    refuted in its numbers and in its law: the docstring's ``a*(m) = A - K_s/(sigma(m-1))``
    and its threshold ``sigma (m-1) A > K_s`` both omit the reciprocity term, and the true
    threshold is ``2 A sigma (m - 1) > K_s`` -- a factor of two, which is exactly where the
    two disagree at ``sigma = 0.1, m = 3``."""
    said = lambda m: A_MAX - K_S / (SIGMA * (m - 1)) if SIGMA * (m - 1) * A_MAX > K_S else 0.0
    print()
    for m in (1, 2, 3, 4):
        got, want = alone(m, SIGMA, 1.0), cluster_amplitude(m, 0.0)
        print(f"  sigma 0.5, m = {m}: {got:.5f}   closed form {want:.5f}"
              f"   docstring said {said(m):.4f}")
        assert abs(got - want) < 1e-4
    assert alone(1, SIGMA, 1.0) < 1e-9                       # a lone part dies
    # the discriminating case: the docstring's rule says a three-part fragment dies here
    for m, a0 in ((2, 3.9), (3, 0.01), (3, 3.9), (4, 0.01)):
        got, want = alone(m, 0.1, a0), cluster_amplitude(m, 0.0, 0.1)
        said_lo = A_MAX - K_S / (0.1 * (m - 1)) if 0.1 * (m - 1) * A_MAX > K_S else 0.0
        print(f"  sigma 0.1, m = {m}, from {a0}: {got:.5f}   closed form {want:.5f}"
              f"   docstring said {said_lo:.4f}")
        assert abs(got - want) < 1e-3
    # and the threshold itself, found by bisection on sigma at m = 3
    crit = bisect(lambda sg: alone(3, sg, 1.0, 120_000) > 1e-3, 0.02, 0.2, 12)
    print(f"  critical sigma at m = 3: measured {crit:.5f}, closed form"
          f" {sigma_critical(3):.5f} (the docstring's rule would give {K_S/(A_MAX*2):.5f})")
    assert abs(crit - sigma_critical(3)) < 0.05 * sigma_critical(3)


# ---------------------------------------------------------------------------
# (B3) What offer takes the collective, and what offer takes one part out of it.
# ---------------------------------------------------------------------------


HOME, T_MOVE, MU_SWEEP = 0.06, 200_000, 0.03


def where_they_end(*, m_home, m_else, s, g, free, T=T_MOVE, mu_p=MU_SWEEP, sigma=SIGMA):
    """``m_home`` parts at home and ``m_else`` at the elsewhere; only the parts in ``free``
    (indices into the whole list, home first) may move. Returns their placements."""
    n = m_home + m_else
    pos = [HOME] * m_home + [D_SRC] * m_else
    mob = [mu_p if i in free else 0.0 for i in range(n)]
    C = bare(suppliers(n, [cluster_amplitude(max(m_home, 1), g)] * m_home
                       + [cluster_amplitude(max(m_else, 1), s)] * m_else, sigma=sigma, **KW),
             positions=pos, mobilities=mob, tau=TAU, s=s, d=D_SRC)
    return look(C, runner(C, T, g)(C.arr.P.init), g)[0]


def test_B3_the_offer_that_takes_the_collective():
    """(B3). *Refuted as stated, twice over, and both corrections are the finding.*

    First, the composite is a **gradient system**: every part's cost is exact and the parent
    is cost-free, and the kernel is symmetric (``kappa_ij = kappa_ji``), so the whole
    arrangement ascends one potential,

        U(q) = sum_j [ (alpha a_j + beta) I(r_j) - (b/2) a_j^2 ],

    checked against the run in ``test_the_potential_the_whole_thing_ascends``. Placements
    included: a part is pulled toward the others not only by what it receives from them but
    by *their* need for what it sends.

    Second, there are two thresholds, not the one the prediction named, and only one of them
    is a threshold at all.

    * The **body** threshold. What the parts pass to each other travels with them, so it puts
      no gradient on the cluster's own position: ``dU/dp`` for a whole cluster is
      ``[sum_j a_j I'(r_j)] (g kappa'(p) + s kappa'(p - d))``, which vanishes exactly where
      stage (a)'s ``h_1`` does. The collective moves when ``g/s < phi_min`` -- the offer that
      moves one part moves four, and stage (a)'s R2/R3 stand for the body.
    * **Defection does not happen at all.** The prediction said one part leaves at
      ``s > (g + sigma (m-1) a)/phi_min = 18.49``. It never leaves: raising the offer slides
      it *partway out* and it stops, at the maximum of ``U`` along its own placement -- far
      enough to pick the big offer up, near enough to keep passing it back, which is worth
      more to the whole than leaving. At ``s = 1000`` it sits at ``p = 1.3``.
    """
    print()
    a4 = cluster_amplitude(4, 1.0)
    for m in (1, 4):
        body = bisect(lambda s: np.mean(where_they_end(
            m_home=m, m_else=0, s=s, g=1.0, free=range(m))) > 1.5, 1.0, 12.0, 11)
        print(f"  body of {m}: moves above s = {body:.4f}   closed form g/phi_min ="
              f" {1.0 / PHI_MIN:.4f}", flush=True)
        assert abs(body - 1.0 / PHI_MIN) < 0.08 * (1.0 / PHI_MIN)
    said = (1.0 + SIGMA * 3 * a4) / PHI_MIN
    print(f"  (B3) said one part of four leaves above s = {said:.4f}, with a = {a4:.4f}")
    for s in (10.0, 20.0, 40.0, 1000.0):
        p = where_they_end(m_home=4, m_else=0, s=s, g=1.0, free=[0])[0]
        pk, _ = landscape_peak(s)
        print(f"  offer {s:7.1f}: the one free part settles at {p:.4f};"
              f" the peak of U along its placement is at {pk:.4f}", flush=True)
        assert abs(p - pk) < 0.05 and p < 1.5      # it never reaches the elsewhere


def u_peaks(others, s, g, lo=-0.5, hi=3.2, grid=281):
    """Every local maximum of ``U`` along one part's placement, with the other parts held at
    the placements ``others`` and every amplitude at its fixed point: the model's own
    equations, solved by hand, not a run. This is the landscape the free part climbs, and it
    is *not* the supply that arrives at it -- it also carries the others' need for what it
    sends them."""
    ps = np.linspace(lo, hi, grid)
    n = len(others) + 1
    us = []
    for p in ps:
        P = np.array([p] + list(others))
        Km = np.array([[kappa(P[i] - P[j]) for j in range(n)] for i in range(n)])
        np.fill_diagonal(Km, 0.0)
        a = np.full(n, 2.0)
        for _ in range(1500):
            r = g * kappa(P) + s * kappa(P - D_SRC) + SIGMA * (Km @ a)
            a = a + 0.2 * (I(r) + SIGMA * (Km @ (a * dI(r))) - a)
        r = g * kappa(P) + s * kappa(P - D_SRC) + SIGMA * (Km @ a)
        us.append(float(np.sum(a * I(r) - 0.5 * a * a)))
    us = np.array(us)
    return [float(ps[i]) for i in range(1, grid - 1) if us[i] > us[i - 1] and us[i] > us[i + 1]]


def landscape_peak(s, g=1.0, others=3, p_others=HOME):
    """The peak of ``U`` nearest the others, for (B3)."""
    pk = u_peaks([p_others] * others, s, g)
    return min(pk, key=lambda x: abs(x - p_others)), pk


# ---------------------------------------------------------------------------
# (B4) Who calls whom home, and (B5) what it costs to hire one out of a group.
# ---------------------------------------------------------------------------


def test_B4_a_fragment_calls_a_stray_home():
    """(B4). *Refuted in its number, and the correction is the stronger claim.* The
    prediction put the return at ``what home offers > phi_max s``, which needs two parts at
    home. One is enough, and the reason is the same one that makes (B3)'s defection
    impossible: what calls the stray back is not only what home would give it but **home's
    need for what it sends**, which is in ``U`` and not in the arriving supply. With nobody
    at home the elsewhere is a stable place for it (stage (a)'s bistability, at the wage
    ``g = s = 1`` that (a)'s R1 called unable to outbid the elsewhere); with one part at
    home that stable place is gone."""
    print()
    for m in (0, 1, 2):
        G = 1.0 + SIGMA * m * cluster_amplitude(max(m, 1), 1.0)
        pk = u_peaks([HOME] * m, S_SRC, 1.0)
        p = where_they_end(m_home=m, m_else=1, s=S_SRC, g=1.0, free=[m], T=150_000)
        stays = any(x > 1.5 for x in pk)
        print(f"  {m} at home, wage 1: peaks of U along the stray {np.round(pk, 3)};"
              f" it ends at {p[m]:.4f}   (what home offers, {G:.4f}, against phi_max s ="
              f" {PHI_MAX:.4f}: the prediction said it {'stays' if G < PHI_MAX else 'returns'})",
              flush=True)
        assert (p[m] > 1.5) == stays


def test_B5_the_wage_that_hires_one_out_of_a_group():
    """(B5). *Refuted as stated, and split in two like (B3).* The wage that brings the whole
    group home is ``phi_max s`` for every group size -- stage (a)'s R3 stands for the body.
    The wage that brings **one** part home out of a group of ``m`` is
    ``phi_max (s + sigma (m-1) a(m, s))``, which rises steeply: what you must outbid is not
    the elsewhere but what the others there are giving each other."""
    print()
    for m in (1, 2, 4):
        body = bisect(lambda g: np.mean(where_they_end(
            m_home=0, m_else=m, s=S_SRC, g=g, free=range(m))) < 1.5, 1.0, 12.0, 11)
        print(f"  the whole group of {m} comes home above g = {body:.4f}"
              f"   closed form phi_max s = {PHI_MAX:.4f}")
        assert abs(body - PHI_MAX) < 0.08 * PHI_MAX
    for m in (2, 4):
        one = bisect(lambda g: where_they_end(
            m_home=0, m_else=m, s=S_SRC, g=g, free=[0])[0] < 1.5, 1.0, 60.0, 11)
        said = PHI_MAX * (S_SRC + SIGMA * (m - 1) * cluster_amplitude(m, S_SRC))
        from_U = bisect(lambda g: not any(
            x > 1.5 for x in u_peaks([D_SRC] * (m - 1), S_SRC, g)), 1.0, 60.0, 9)
        print(f"  one part out of {m} comes home above g = {one:.4f}; the landscape says"
              f" {from_U:.4f}; (B5) said phi_max (s + sigma (m-1) a) = {said:.4f}", flush=True)
        assert abs(one - from_U) < 0.10 * from_U


# ---------------------------------------------------------------------------
# The potential, and what it means that there is one.
# ---------------------------------------------------------------------------


def potential(a, p, g, s=S_SRC, sigma=SIGMA):
    """``U(q) = sum_j [(alpha a_j + beta) I(r_j) - (b/2) a_j^2]``, on amplitudes and
    placements, with the elsewhere at ``D_SRC`` and the wage ``g`` at the origin."""
    kap = lambda x: jnp.exp(-x ** 2 / (2 * TAU))
    K = kap(p[:, None] - p[None, :]) - jnp.eye(p.shape[0])
    r = g * kap(p) + s * kap(p - D_SRC) + sigma * (K @ a)
    return jnp.sum((ALPHA * a + BETA) * (A_MAX * r / (K_S + r)) - 0.5 * B * a ** 2)


def test_the_potential_the_whole_thing_ascends():
    """Not a prediction. Every part's forcing is exact and the parent is cost-free, and the
    spatial kernel is symmetric, so the composite is exact too: amplitudes *and* placements
    ascend one potential ``U``. Checked away from equilibrium, at a scrambled state, against
    the library's own forcing. (Stage (a)'s cascade is not a gradient system -- the fund's
    flow is non-exact -- and ``test_mutual_need.py``'s gated pair is not either, its gate
    sitting in the coupler and breaking the symmetry. Here the symmetry of the kernel puts it
    back.)"""
    g = 1.0
    C = setup(4, positions=[0.1, -0.2, 0.35, 1.1], a0=[1.0, 2.5, 0.4, 3.0])
    q = np.asarray(C.arr.P.position(C.arr.P.init))
    xi = np.asarray(C.arr.omega(jnp.asarray(q), EMPTY, supplied(g))[0])
    a, p = jnp.asarray(q[:4]), jnp.asarray(q[4:8])
    grad_a, grad_p = jax.grad(lambda a, p: potential(a, p, g), argnums=(0, 1))(a, p)
    print(f"\n  forcing on the amplitudes {np.round(xi[:4], 6)}"
          f"\n  dU/da                     {np.round(np.asarray(grad_a), 6)}"
          f"\n  forcing on the placements {np.round(xi[4:8], 6)}"
          f"\n  dU/dp                     {np.round(np.asarray(grad_p), 6)}")
    assert np.max(np.abs(xi[:4] - np.asarray(grad_a))) < 1e-9
    assert np.max(np.abs(xi[4:8] - np.asarray(grad_p))) < 1e-9


# ---------------------------------------------------------------------------
# (B2) The paymaster leaves.
# ---------------------------------------------------------------------------


FUND = dict(f0=20.0, mu=0.05, wmax=4.0, K=1.0, c0=2.0, leak=0.1, record=None)
R_MKT = 2.0


def funded_setup(n=4, *, sigma, s=S_SRC, mu_p=0.02, pos=HOME):
    return funded(suppliers(n, 1.0, sigma=sigma, **dict(KW, beta=1.0 if sigma == 0.0 else BETA)),
                  positions=[pos] * n, mobilities=[mu_p] * n, tau=TAU, s=s, d=D_SRC,
                  fund=FUND, r_market=R_MKT)


def funded_run(arr, steps, state=None):
    step = closed_step(arr, EMPTY)
    f = jax.jit(lambda s: jax.lax.fori_loop(0, steps, lambda i, x: step(x), s))
    return f(arr.P.init if state is None else state)


def funded_read(u, arr, state):
    d = read(u, arr.P.position(state))
    return (float(d["fund"]), np.asarray(d["positions"]),
            np.array([float(c["a"]) for c in d["children"]]))


def test_B2_the_paymaster_can_leave():
    """(B2). *Refuted in its letter, answered in what it was about.* The letter said nobody
    leaves home. They do leave: with the wage gone, home offers nothing and the elsewhere
    offers ``s``, so the landscape has one peak and the collective walks to it -- but it
    walks **as a body and keeps its form**. What it loses is measured against the control,
    the same run with ``sigma = 0``, where each part is paid and supplies nobody: those parts
    are left with exactly what the elsewhere pays, ``I(s) = 2``, a third less than they held.
    The collective's parts lose almost nothing, because what they live on is each other.

    David, 2026-09-26: "if the paymaster leaves, other parts can pay for their needs." In
    this model they can, and what they pay with is the supply they were passing on anyway."""
    print()
    for sigma in (0.0, SIGMA):
        C = setup(4, positions=[HOME] * 4, sigma=sigma)
        paid = settle(C, 150_000, 4.0)
        p0, a0, r0, _ = look(C, paid, 4.0)
        gone = settle(C, 400_000, 0.0, state=paid)
        p1, a1, r1, mv = look(C, gone, 0.0)
        print(f"  sigma = {sigma}: wage 4, placements {np.round(p0, 4)} amplitudes"
              f" {np.round(a0, 4)} arriving {np.round(r0, 4)}")
        print(f"                 wage 0, placements {np.round(p1, 4)} amplitudes"
              f" {np.round(a1, 4)} arriving {np.round(r1, 4)}"
              f"  ({100 * (a1[0] / a0[0] - 1):+.1f}%, one-step move {mv:.1e})", flush=True)
        assert np.all(p1 > 2.0)                       # both walk to the elsewhere
        if sigma == 0.0:
            assert abs(a1[0] - I(S_SRC)) < 1e-4 and a1[0] < 0.8 * a0[0]
        else:
            assert a1[0] > 0.95 * a0[0] and a1[0] > 1.8 * I(S_SRC)


def test_B6_a_fixed_point_forgets():
    """(B6). Held. A boom at the elsewhere takes the collective there; when it ends, the
    same wage brings it home, and home is where it was: placements, amplitudes and the
    offer that would take it again all agree with the first equilibrium."""
    C = setup(4, positions=[HOME] * 4)
    first = settle(C, 200_000, 4.0)
    p0, a0, _, _ = look(C, first, 4.0)
    boom = setup(4, positions=[HOME] * 4, s=12.0)
    away = runner(boom, 200_000, 4.0)(first)
    p1, _, _, _ = look(boom, away, 4.0)
    back = settle(C, 300_000, 4.0, state=away)
    p2, a2, _, mv = look(C, back, 4.0)
    print(f"\n  first equilibrium: placements {np.round(p0, 6)} amplitudes {np.round(a0, 6)}"
          f"\n  during the boom (s = 12): placements {np.round(p1, 4)}"
          f"\n  after it: placements {np.round(p2, 6)} amplitudes {np.round(a2, 6)}"
          f"   one-step move {mv:.1e}")
    assert np.all(p1 > 2.0) and np.max(np.abs(p2 - p0)) < 1e-4 and np.max(np.abs(a2 - a0)) < 1e-4


def test_B7_what_a_boom_can_and_cannot_do():
    """(B7). *Refuted: there is no band, because there is no split.* With the placement
    mobilities spread eightfold and a boom above the body threshold, every window gives
    ``k = 0`` or ``k = 4``: the fast parts are held by the slow ones (and the slow pulled by
    the fast), so the collective arrives together or not at all, and the spread never
    approaches the kernel's reach. The control, ``sigma = 0``, is the same run with nothing
    holding them: there the parts strew out along the way, and a window leaves them at
    different places. So in this model a collective whose parts supply each other is not
    *broken* by a shock at all; it is *moved*. Stage (a)'s R6 -- membership is the one memory
    a break can leave -- has nothing to record here, because nothing is lost."""
    print()
    MOB = [0.08, 0.04, 0.02, 0.01]
    for sigma in (SIGMA, 0.0):
        C = setup(4, positions=[HOME] * 4, mobilities=MOB, sigma=sigma)
        boom = setup(4, positions=[HOME] * 4, mobilities=MOB, sigma=sigma, s=12.0)
        home_state = settle(C, 200_000, 4.0)
        for W in (200, 600, 1_500, 4_000, 20_000):
            away = runner(boom, W, 4.0)(home_state)
            p_boom = look(boom, away, 4.0)[0]
            end = runner(C, 300_000, 4.0)(away)
            p = look(C, end, 4.0)[0]
            print(f"  sigma = {sigma}, boom of {W:6d} steps: placements at its end"
                  f" {np.round(p_boom, 3)} (spread {np.ptp(p_boom):.3f});"
                  f" after it {np.round(p, 3)}", flush=True)
            if sigma == SIGMA:
                k = int(np.sum(p > 1.5))
                assert k in (0, 4) and np.ptp(p_boom) < 1.0
    # and a split *is* stable, but what makes it stick is distance, not the wage: two
    # clusters hold their places once they are out of each other's reach, and merge while
    # they can still feel each other -- each pulled by the other's need as well as its
    # supply. Nothing in this model puts them there; only an initial condition does.
    for d in (D_SRC, 4.0):
        C = setup(4, positions=[HOME, HOME, d, d], d=d)
        p, a, r, mv = look(C, settle(C, 300_000, 1.0), 1.0)
        print(f"  two and two, placed apart at the start, wage 1, the elsewhere at {d}"
              f" (kappa = {kappa(d):.2e}): placements {np.round(p, 4)} amplitudes"
              f" {np.round(a, 4)}  one-step move {mv:.1e}", flush=True)
        if d == D_SRC:
            assert np.ptp(p) < 0.01                     # they merge, at the elsewhere
        else:
            assert np.all(p[:2] < 0.5) and np.all(p[2:] > 3.5) and np.all(a > 3.0)


# ---------------------------------------------------------------------------
# Many small passes, or few large ones (David, 2026-09-26, asked which is the real case:
# "either one, but i bet it's many order of magnitude more common to pass intra and many
# order of magnitude bigger loads or more significant when coming from the outside").
# ---------------------------------------------------------------------------


def test_many_small_passes_and_few_large_ones_are_the_same_thing():
    """Not one of (B1)-(B8): run because David's answer named the case. Nothing in the
    construction reads how many parts a part is fed by, or how much each sends; every
    quantity that decides anything here -- whether a fragment holds a form, what it holds,
    what it costs to take one part out of it -- depends on ``sigma (m - 1)``, the *total*
    that arrives, and on nothing else. So a collective of nine parts each passing on 1/16 of
    itself and a collective of three each passing on 1/4 are the same collective to this
    model, and the line between passing on and making is at ``sigma (m - 1) = K_s / 2A``
    wherever it is crossed.

    This is the same fact as the turnover run's (§5.9 item 13): the cost is on a total, which
    is why parts are swappable there and why count trades against size here. It is what makes
    the model *able* to hold David's case -- very many very small internal passes -- and it is
    also its limit: it cannot tell that case apart from a few large ones."""
    print()
    for m in (3, 5, 9):
        crit = bisect(lambda sg: alone(m, sg, 1.0, 120_000) > 1e-3, 0.002, 0.3, 13)
        print(f"  m = {m}: critical sigma measured {crit:.5f}, closed form"
              f" {sigma_critical(m):.5f}; sigma (m-1) = {crit * (m - 1):.5f}", flush=True)
        assert abs(crit - sigma_critical(m)) < 0.03 * sigma_critical(m)
    held = [alone(m, 0.5 / (m - 1), 1.0) for m in (3, 5, 9)]
    print(f"  the same total, sigma (m-1) = 0.5, at m = 3, 5, 9: {np.round(held, 6)}")
    assert np.ptp(held) < 1e-6 and held[0] > 1.0


# ---------------------------------------------------------------------------
# (B8) The scar.
# ---------------------------------------------------------------------------


def test_B8_the_scar_of_stage_a_was_the_funds():
    """(B8). *Refuted for the collective, held for the fund, and the split is the finding.*
    Stage (a)'s R6 (ii): a collective re-formed short a part broke at 67% less. Here the
    offer that moves a collective does not depend on its size at all -- 2.716 for one part
    and for four (B3) -- so a collective short a part is exactly as hard to move: **no
    scar**. Put the fund back and the scar returns, because the missing part's product is
    what the balance was made of. What stage (a) weakened was the paymaster, not the
    collective."""
    print()
    thresholds = {}
    for n in (4, 3):
        u, arr = funded_setup(n=n, sigma=SIGMA)
        eq = funded_run(arr, 150_000)
        f_eq = funded_read(u, arr, eq)[0]
        S = bisect(lambda x: funded_read(u, arr, funded_run(
            arr, 200_000, ((eq[0][0], eq[0][1] - x), eq[1])))[0] < 0.0, 0.0, 4 * f_eq, 9)
        thresholds[n] = S
        print(f"  {n} parts: balance {f_eq:.4f}, the drain that kills the fund {S:.4f}",
              flush=True)
    print(f"  short one part, the fund falls to a drain"
          f" {100 * (thresholds[3] / thresholds[4] - 1):+.1f}% of the original's")
    assert thresholds[3] < thresholds[4]
