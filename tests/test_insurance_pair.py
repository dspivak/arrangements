"""The insurance pair: an insurer and an insured, not an insurer and its record.

David, 2026-09-26: "run the other half of the insurance test: the insurer and the insured, not
the insurer and its record. one box pays premiums and is paid claims, the other holds the
forecast." The first half (`tests/test_insurance.py`) put a forecast beside the record it is
charged on. This one puts it beside the party that pays it.

Two 0-ary boxes, each publishing its configuration and receiving the other's through one static
closing arrangement (the normal form of `notes/TESTS-SYNTHESIS-2026-09-26.md` §3.1). The
insured's configuration is the cover ``a`` it holds; the insurer's is the premium ``p`` it
quotes and, from (5), the wealth ``W`` it writes cover against. The insured's paying and being
paid is in its cost, ``a(p - LAM)``; its wealth is not carried as a coordinate, because it would
be exactly ``-W`` up to its starting value -- the same record the first half already ran, in a
second copy. Claims arrive at the world's rate ``LAM`` per unit cover,
with variance ``SIG2`` per unit cover. So the contract's money, per step, is

    the insured pays   a p      and is paid   a LAM,   cost  k1 = a (p - LAM) + (its risk)
    the insurer is paid  a p    and pays      a LAM,   cost  k2 = -a (p - LAM) + (its risk).

**Which of the two things David asked about exactness is it?** He put it exactly: "tell me
whether that pair can be exact, or whether exactness would be built in by the way the cost is
written -- i want to know which." It is both, and the fork is the content. Two ways to wire the
same story:

    *own side only* -- each box reads the other's published value, but the port component of
        its cost is dropped, so each party's forcing acts on its own coordinate alone. This is
        how a one-way forcing is written, and it is what the story says: my wish for a lower
        premium does not lower it.
    *both sides* -- each box's forcing is the full ``d(-k_b)``, so the port component is
        transposed back onto the other's configuration. Each party feels the other's cost as
        its own.

Under *own side only*, ``(d omega)_ap = d_a d_p (k1 - k2)``: exactness is the cross-partial of
the two costs' **difference**, so it is decided by the relation between them and not by how
either is written. Under *both sides* the composite is exact whatever the costs are, since a
pullback of an exact form is exact -- that is exactness built in, and it is not free.

**The predictions, fixed here before any construction code existed, each able to fail.**

    (1)   *A transfer has no total to descend.* With no risk the costs are exactly opposed,
          ``k1 + k2 = 0``. Under *both sides* the total forcing is ``-d(k1 + k2) = 0``: the
          pair is exact and **nothing moves at all**, from any starting point, bit for bit.
          Under *own side only* ``(d omega)_ap = 2 d_a d_p k1 = 2``, the largest it can be for
          this contract, and the pair is a pure rotation about ``(a, p) = (0, LAM)``: the
          insured cuts cover while the premium is above the world's rate, the insurer stops
          raising it as the cover goes, the price falls back through ``LAM``, and the cover is
          bought again. **So the insurance cycle needs neither randomness nor risk** -- it is
          the non-exactness of a transfer, and it is the same rotation the record half found,
          ``E = (p - LAM)^2 + a^2`` multiplied by exactly ``1 + mu^2`` a step.
          **Able to fail:** *both sides* could have moved; the own-side loop could have settled
          or run away instead of circling; ``d omega`` could have been something other than 2.

    (2)   *A shared cost is exact whatever it is; that is the mutual, not the insurer.* Two
          parties carrying one cost ``k1 = k2 = k`` give ``(d omega)_ap = 0`` under *own side
          only* as well, for any ``k``. So exactness is available and it costs the insurance:
          it is two members of a mutual, not an insurer and an insured.
          **Able to fail:** if a shared cost were non-exact when each party acts on its own
          coordinate only.

    (3)   *Risk is the non-zero-sum part of the transfer, and the exact pair cannot see the
          price.* Let the insured retain variance ``(1-a)^2 SIG2`` at aversion ``RHO1`` and the
          insurer take on ``a^2 SIG2`` at ``RHO2``. Now ``k1 + k2 = (RHO1/2)(1-a)^2 SIG2 +
          (RHO2/2) a^2 SIG2`` is not zero, and that residue is the whole reason the pair moves:
          it is what insurance is for. Under *both sides* the pair descends it and settles at
          the optimal coinsurance ``a* = RHO1/(RHO1 + RHO2) = 0.8``, with **the premium
          unmoved, bit for bit, wherever it started**: the exact dynamics moves the risk to
          whoever minds it less and is blind to who pays. Under *own side only*
          ``(d omega)_ap`` is still exactly 2, because the risk terms are functions of ``a``
          alone and add nothing to the cross-partial.
          **Able to fail:** the exact pair could settle somewhere other than ``RHO1/(RHO1 +
          RHO2)``; the premium could move; the risk terms could have changed ``d omega``.

    (4)   *The own-gradient pair gets a price and loses the trade.* Under *own side only* with
          risk, ``F_a = RHO1 (1-a) SIG2 - (p - LAM)`` and ``F_p = a``: the insurer raises its
          premium at a rate proportional to the cover it has sold and nothing stops it, because
          **a gradient on its own coordinate sees the present cover and not how the cover will
          respond** -- the insurer cannot see the demand curve. The rotation of (1) becomes an
          inward spiral (Jacobian ``[[1 - mu RHO1 SIG2, -mu], [mu, 1]]``, determinant
          ``0.9845``, modulus ``0.99222``) and it settles at **no cover at all**, ``a = 0``, at
          the insured's choke price ``p = LAM + RHO1 SIG2 = 0.48``. So the exact pair allocates
          the risk and prices nothing; the own-gradient pair prices it and allocates nothing.
          Neither is a market.
          **Able to fail:** it could settle at a positive cover, or not settle; the premium
          could stop below the choke price.

    (5)   *What happens to the insurance as the insurer approaches zero wealth* (David,
          2026-09-26). Ruin is not available -- a coordinate going negative is not ruin, it is
          a number going negative, and the record half already measured one running to ``-605``
          and continuing. The nearest thing that is available is a **barrier**: give the
          insurer's wealth the cost ``-BETA log W``, whose quadratic expansion in a small shock
          is ``-BETA log W + (BETA/2W^2) Var``, so its aversion is ``RHO2(W) = BETA/W^2`` and
          rises without bound as the wealth falls. Two things follow, and the second is the
          answer.

          First, the aversion has nowhere to act unless the insurer holds a forecast. The
          insurer sets the price and the insured sets the quantity, so an aversion about
          quantity never reaches the insurer's own coordinate. For its wealth to change its
          price the insurer must hold a forecast of the demand -- present state in a 0-ary box,
          exactly as the record half's premium was. Held here: ``ahat(p) = ADEM - GAM (p -
          LAM)`` with ``ADEM = 1`` and ``GAM = 1/(RHO1 SIG2)``, which is the insured's *true*
          settled response, so the insurer's expectation and the insured's choice agree and
          nothing in the run turns on the forecast being wrong about demand. (Its forecast of
          the claim rate is what goes wrong, in (6).)

          Second, with that forecast, writing ``r = BETA SIG2 / W^2`` and ``d = p - LAM``, the
          insurer's premium settles at ``d* = ADEM (1 + GAM r)/(GAM (2 + GAM r))`` and the cover
          at ``ahat* = ADEM/(2 + GAM r)``. So as ``W`` falls the premium rises **monotonically
          from the monopoly price ``LAM + ADEM/(2 GAM) = 0.30`` to the choke price
          ``LAM + ADEM/GAM = LAM + RHO1 SIG2 = 0.48``, which is the insured's own choke price of
          (4)**, and the cover written falls to zero as ``W^2``, at ``ahat* ~ ADEM W^2/(GAM BETA
          SIG2)``. Predicted at ``W = 1, 0.1, 0.05, 0.02, 0.01``: ``d* = 0.18045, 0.216, 0.27,
          0.33517, 0.35308`` and ``ahat* = 0.49875, 0.4, 0.25, 0.06897, 0.019231``. **The
          insurer never fails.** It withdraws continuously; there is no last moment, and the
          insurance runs out before the insurer does without the insurer ever arriving anywhere.
          **Able to fail:** the premium could fall as the wealth falls (a gamble for
          resurrection), or the cover could go to zero at a positive wealth rather than
          asymptotically, or either closed form could be wrong.

    (6)   *And the insurer cannot get there by itself; being wrong is what takes it there.*
          With the claim rate constant and the premium above it, the insurer's wealth is
          **monotone increasing** -- it charges more than it pays, every step -- so near-ruin is
          an initial condition and not something that happens. What makes wealth fall is a
          forecast that is wrong: the insurer prices off a held ``LAMHAT`` and pays the world's
          ``LAM``, losing ``DEL = LAM - LAMHAT`` per unit cover. That joins this half of the
          test to the other: **being wrong drives the wealth down, and low wealth drives the
          premium up.** Since the insurer's markup over its own forecast settles at ``d*(W)``,
          its wealth grows exactly when ``d*(W) > DEL``, and with ``u = DEL/(RHO1 SIG2)``:

            * ``DEL < 0.18`` (the monopoly markup): wealth grows without bound, no hard market;
            * ``0.18 < DEL < 0.36``: the wealth falls to the stable value ``W* = sqrt(GAM BETA
              SIG2 (1 - u)/(2u - 1))``, where the premium it is then forced to charge exactly
              covers the error. **A hard market: the insurer settles poorer and dearer.**
              Predicted ``W* = 0.070711`` at ``DEL = 0.24`` and ``0.035355`` at ``DEL = 0.30``;
            * ``DEL > 0.36 = RHO1 SIG2``, the insured's choke markup: no price the insured will
              pay covers the error, the wealth falls through zero, and the construction stops
              being defined (``log W`` of a negative). **That is where ruin would be if this
              mathematics had it, and what it has instead is a model that stops.**

          **Able to fail:** wealth might not be monotone under a correct forecast; there might
          be no critical error, or it might not be the choke markup; ``W*`` might be wrong or
          unstable.

**What this does not model, and says so rather than hiding it.** There is still no ruin: the
barrier of (5) replaces an event by a boundary the insurer is repelled from, which is a
different thing, and David's reading -- "bearing risk ... you can disintegrate from being
wrong" -- waits on disintegration (`AGENTDOCPLAN.md` §7 item 35). There is no pooling here and
there are no other insurers: the price in (4) and (5) is a monopolist's, and what a competitor
would do is not in the construction. The variance is a coordinate and a parameter, not a draw;
nothing here is random (the library has no randomness in its dynamics), so "risk" means the
second moment carried as present state and nothing more.

Every docstring below says which of three things its test does (STATUS.md §3.7): reproduce a
closed form derived in the docstring, check a discrete analogue, or correct a claim made on
paper.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact
from arrangements.coalgebra import configuration
from arrangements.rvect import diagonal, euclidean
from arrangements.wiring import compose_seq, tensor_arrangements, trivial_parameter

from tests.chain import flat_state
from tests.test_rendezvous import run_closed, run_phi

EMPTY = jnp.zeros(0)

LAM = 0.12       # the world's claim rate per unit cover
SIG2 = 0.09      # the variance of the loss per unit cover
RHO1 = 4.0       # the insured's aversion to variance
RHO2 = 1.0       # the insurer's, where it is a constant (parts 1-4)
MU = 0.05        # the mobility of the cover and of the premium
A0 = 0.6         # the cover the insured starts holding
P0 = 0.20        # the premium the insurer starts quoting

BETA = 0.02      # the weight of the insurer's log-wealth barrier
ADEM = 1.0       # the intercept of the demand curve the insurer holds
GAM = 1.0 / (RHO1 * SIG2)   # its slope: the insured's true settled response


# ---------------------------------------------------------------------------
# Parts 1-4: two 0-ary boxes, one coordinate each, closed by one static swap.
# ---------------------------------------------------------------------------


def party(k, q0, feels_both, mu=MU, label="party"):
    """A 0-ary box publishing its one coordinate and receiving the other party's at its input
    port, carrying the cost ``k(own, other)``.

    ``feels_both`` is the whole fork. False: the port component of ``d(-k)`` is dropped, so the
    party's forcing acts on its own coordinate alone -- my wish for a lower premium does not
    lower it. True: the full ``d(-k)``, whose port component the wiring transposes back onto the
    other party's configuration, so each feels the other's cost as its own."""
    full = exact(lambda q, m_out, n_in: -k(q[0], n_in[0]))

    def own_only(q, m_out, n_in):
        xi_q, xi_m, xi_n = full(q, m_out, n_in)
        return (xi_q, xi_m, jnp.zeros_like(xi_n))

    return Arrangement(
        P=configuration(euclidean(1, mu), init=jnp.array([float(q0)])),
        in_dim_M=0, out_dim_M=0, in_dim_N=1, out_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: EMPTY,
        omega=full if feels_both else own_only,
        label=label)


def swap():
    """The static closing arrangement: each box receives the other's published value. Stateless
    and its one-form is zero, so all the state and all the effort are in the boxes."""
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=2, out_dim_M=2, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, e: EMPTY,
        in_f=lambda q, e, n_in: jnp.array([e[1], e[0]]),
        omega=lambda q, e, n_in: (EMPTY, jnp.zeros(2), EMPTY),
        label="swap")


def pair(k1, k2, feels_both, a0=A0, p0=P0, mu=MU):
    """``compose_seq(tensor(insured, insurer), swap)``: a closed arrangement whose
    configuration is ``(a, p)``, the cover held and the premium quoted."""
    return compose_seq(
        tensor_arrangements([party(k1, a0, feels_both, mu, "insured"),
                             party(k2, p0, feels_both, mu, "insurer")]),
        swap())


# the contract's money alone: what one pays the other receives
k_insured_money = lambda a, p: a * (p - LAM)
k_insurer_money = lambda p, a: -a * (p - LAM)

# with the risk each side carries: the insured retains (1-a)^2, the insurer takes on a^2
k_insured = lambda a, p: a * (p - LAM) + 0.5 * RHO1 * SIG2 * (1.0 - a) ** 2
k_insurer = lambda p, a: -a * (p - LAM) + 0.5 * RHO2 * SIG2 * a ** 2

# a mutual: both members carry the same cost, the insured's, and it does couple a and p
k_mutual_1 = lambda a, p: k_insured(a, p)
k_mutual_2 = lambda p, a: k_insured(a, p)


# ---------------------------------------------------------------------------
# Part 5-6: the insurer's wealth, its barrier, and the demand curve it holds.
# ---------------------------------------------------------------------------


def insured_box(a0=A0, mu=MU):
    """The insured: cover ``a``, published; the premium received at its port; the cost
    ``a(p - LAM) + (RHO1/2)(1-a)^2 SIG2``, acting on its own coordinate only."""
    return party(k_insured, a0, feels_both=False, mu=mu, label="insured")


def insurer_box(p0=P0, w0=1.0, mu=MU, mu_w=0.0, lamhat=LAM, beta=BETA):
    """The insurer: configuration ``(p, W)``, both published (its balance sheet is public, and
    the wealth must be a port for the ledger to reach it).

    Its cost is written against the demand it *holds*, ``ahat(p) = ADEM - GAM (p - LAM)``, which
    is the insured's true settled response, and against its wealth-dependent aversion
    ``BETA/W^2``, the quadratic expansion of the barrier ``-BETA log W``:

        k(p, W) = -ahat(p)(p - lamhat) + (BETA/2W^2) ahat(p)^2 SIG2.

    The only thing it can be wrong about is the claim rate: it believes ``lamhat`` and pays the
    world's ``LAM``, so it prices a margin it does not have. Its view of the demand is right.

    Its forcing acts on ``p`` only. The ``W`` component is dropped deliberately: a preference
    for being richer is not a cash flow, and what moves the wealth is the ledger below.
    ``mu_w = 0`` freezes the wealth, to read the premium it quotes at one balance sheet."""

    def k(q):
        p, W = q[0], q[1]
        ahat = ADEM - GAM * (p - LAM)
        return -ahat * (p - lamhat) + 0.5 * (beta / W ** 2) * ahat ** 2 * SIG2

    def omega(q, m_out, n_in):
        g = jax.grad(k)(q)
        return (jnp.array([-g[0], 0.0]), EMPTY, jnp.zeros(1))

    return Arrangement(
        P=configuration(diagonal(jnp.array([mu, mu_w])),
                        init=jnp.array([float(p0), float(w0)])),
        in_dim_M=0, out_dim_M=0, in_dim_N=1, out_dim_N=2,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: EMPTY,
        omega=omega,
        label="insurer")


def ledger(lam=LAM):
    """The static-interface coupler on the three published ports ``e = (a, p, W)``: it hands
    the premium to the insured and the cover to the insurer, and puts the contract's net cash
    ``a(p - lam)`` on the insurer's wealth. Nothing else; the wealth is a record."""
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=2, out_dim_M=3, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, e: EMPTY,
        in_f=lambda q, e, n_in: jnp.array([e[1], e[0]]),
        omega=lambda q, e, n_in: (EMPTY,
                                  jnp.array([0.0, 0.0, e[0] * (e[1] - lam)]),
                                  EMPTY),
        label="ledger")


MU_A = 1.0 / (2.0 * RHO1 * SIG2)   # the insured's own mobility, from its own curvature


def market(a0=A0, p0=P0, w0=1.0, mu=MU, mu_w=0.0, lamhat=LAM, beta=BETA, mu_a=MU_A):
    """``compose_seq(tensor(insured, insurer), ledger)``: configuration ``(a, p, W)``.

    Each party's mobility comes from its own curvature, so that neither run below is also a
    sweep over the step: the insured's cost has curvature ``RHO1 SIG2`` in the cover and the
    insurer's has ``GAM(2 + GAM r)`` in the premium (``mu_for``), and both contract by 1/2."""
    return compose_seq(
        tensor_arrangements([insured_box(a0, mu_a),
                             insurer_box(p0, w0, mu, mu_w, lamhat, beta)]),
        ledger())


def quoted(W, adem=ADEM, gam=GAM, beta=BETA):
    """The closed forms of prediction (5): with ``r = beta SIG2 / W^2``, the markup the insurer
    settles on and the cover that goes with it."""
    r = beta * SIG2 / W ** 2
    return adem * (1.0 + gam * r) / (gam * (2.0 + gam * r)), adem / (2.0 + gam * r)


def mu_for(W, gam=GAM, beta=BETA):
    """A mobility fixed by the curvature ``gam(2 + gam r)`` of the insurer's cost in its own
    coordinate, so that a sweep over the wealth is not also a sweep over the step: every run
    below contracts by exactly 1/2 a step."""
    r = beta * SIG2 / W ** 2
    return 1.0 / (2.0 * gam * (2.0 + gam * r))


# ---------------------------------------------------------------------------
# Running, and the closed form the rotation of (1) is checked against.
# ---------------------------------------------------------------------------


def traj(b, steps):
    """The whole trajectory of a closed composite, inside a ``lax.scan`` (test 0 checks it
    against ``Phi``)."""
    P, omega = b.P, b.omega

    @jax.jit
    def step(s, _):
        xi_q, _, _ = omega(P.position(s), EMPTY, EMPTY)
        s2 = P.step(s, xi_q)
        return s2, flat_state(s2)

    return np.asarray(jax.lax.scan(step, P.init, None, length=steps)[1])


def d_omega(b, q):
    """``(d omega)_{ij} = d_i F_j - d_j F_i`` of the composite's forcing on its configuration,
    exactly as `tests/test_insurance.py` and the mutual-need and pulling tests compute it."""
    J = jax.jacobian(lambda x: b.omega(x, EMPTY, EMPTY)[0])(jnp.asarray(q, float))
    return np.asarray(J.T - J)


def rotation(n, a0=A0, p0=P0, mu=MU, lam=LAM):
    """The closed form for the transfer alone. In ``(A, D) = (a, p - lam)`` the step is
    ``A' = A - mu D``, ``D' = D + mu A``, that is ``rho R(theta)`` with
    ``rho = sqrt(1 + mu^2)`` and ``theta = arctan(mu)``."""
    rho, th = np.sqrt(1.0 + mu * mu), np.arctan(mu)
    A0_, D0 = a0, p0 - lam
    A = rho ** n * (A0_ * np.cos(n * th) - D0 * np.sin(n * th))
    D = rho ** n * (A0_ * np.sin(n * th) + D0 * np.cos(n * th))
    return np.stack([A, lam + D], axis=-1)


# ---------------------------------------------------------------------------
# Test 0: the runs are Phi's.
# ---------------------------------------------------------------------------


def test_0_the_runs_are_the_dynamics_functors():
    """LICENSES the run form used below; it establishes nothing about the predictions. For the
    transfer alone, the pair with risk, the exact pair and the market with a live wealth,
    ``Phi``, ``run_closed`` and the ``lax.scan`` of ``traj`` agree over fifty steps to within
    one unit in the last place."""
    for b in (pair(k_insured_money, k_insurer_money, False),
              pair(k_insured, k_insurer, False),
              pair(k_insured, k_insurer, True),
              market(w0=0.2, mu_w=0.02)):
        by_phi = run_phi(b, 50, EMPTY)
        by_closed = run_closed(b, 50, lambda t: EMPTY)
        assert float(np.max(np.abs(by_phi - by_closed))) < 1e-14
        np.testing.assert_array_equal(traj(b, 50)[-1], by_closed[-1])


# ---------------------------------------------------------------------------
# Test 1: a transfer has no total to descend, and the pair that does not descend it circles.
# ---------------------------------------------------------------------------


def test_1_a_transfer_has_no_total_to_descend_and_the_pair_circles_instead():
    """REPRODUCES the closed form ``rotation`` and CHECKS prediction (1)."""
    both = pair(k_insured_money, k_insurer_money, True)
    for q in ([A0, P0], [-1.3, 0.7], [0.0, LAM]):
        np.testing.assert_array_equal(np.asarray(both.omega(jnp.asarray(q), EMPTY, EMPTY)[0]),
                                      [0.0, 0.0])
        np.testing.assert_allclose(d_omega(both, q), np.zeros((2, 2)), atol=1e-16)
    rows_both = traj(both, 2000)
    np.testing.assert_array_equal(rows_both, np.tile([A0, P0], (2000, 1)))   # bit for bit

    own = pair(k_insured_money, k_insurer_money, False)
    for q in ([A0, P0], [-1.3, 0.7]):
        np.testing.assert_allclose(d_omega(own, q)[0, 1], 2.0, atol=1e-16)
        np.testing.assert_allclose(np.abs(d_omega(own, q)).max(), 2.0, atol=1e-16)

    n = np.arange(1, 4001)
    rows = traj(own, 4000)
    np.testing.assert_allclose(rows, rotation(n), atol=1e-10)
    E = rows[:, 0] ** 2 + (rows[:, 1] - LAM) ** 2
    np.testing.assert_allclose(E, (A0 ** 2 + (P0 - LAM) ** 2) * (1.0 + MU * MU) ** n, rtol=1e-9)
    assert np.all(np.diff(E) > 0.0)                          # a cycle, and it does not settle
    period = 2.0 * np.pi / np.arctan(MU)
    np.testing.assert_allclose(period, 125.8, atol=0.1)
    crossings = np.where(np.diff(np.sign(rows[:, 1] - LAM)) < 0)[0] + 2
    np.testing.assert_allclose(np.diff(crossings[:4]), period, atol=1.0)
    assert np.any(rows[:, 0] < 0.0) and np.any(rows[:, 1] < LAM)   # the soft half of the cycle


# ---------------------------------------------------------------------------
# Test 2: a shared cost is exact whatever it is.
# ---------------------------------------------------------------------------


def test_2_a_shared_cost_is_exact_whatever_it_is_and_that_is_the_mutual():
    """REPRODUCES the closed form ``(d omega)_ap = d_a d_p (k1 - k2)`` and CHECKS prediction
    (2). Three shared costs, each with a genuine cross term, each exact under *own side only*;
    and the same three with the sign of the second party's cost flipped, each non-exact by
    exactly twice the cross-partial."""
    shared = [(lambda a, p: a * (p - LAM),            1.0),
              (k_insured,                             1.0),
              (lambda a, p: jnp.sin(a * p) + a ** 3,  None)]
    for K, cross in shared:
        b_shared = pair(lambda a, p: K(a, p), lambda p, a: K(a, p), False)
        b_opposed = pair(lambda a, p: K(a, p), lambda p, a: -K(a, p), False)
        for q in ([A0, P0], [-0.4, 0.9]):
            np.testing.assert_allclose(np.abs(d_omega(b_shared, q)).max(), 0.0, atol=1e-14)
            dadp = float(jax.grad(lambda p: jax.grad(K)(jnp.asarray(q[0]), p))(jnp.asarray(q[1])))
            np.testing.assert_allclose(d_omega(b_opposed, q)[0, 1], 2.0 * dadp, atol=1e-13)
            if cross is not None:
                np.testing.assert_allclose(dadp, cross, atol=1e-14)


# ---------------------------------------------------------------------------
# Test 3: risk is the non-zero-sum part, and the exact pair cannot see the price.
# ---------------------------------------------------------------------------


def test_3_risk_is_the_non_zero_sum_part_and_the_exact_pair_cannot_see_the_price():
    """REPRODUCES the closed form ``a* = RHO1/(RHO1 + RHO2)`` and CHECKS prediction (3)."""
    own = pair(k_insured, k_insurer, False)
    for q in ([A0, P0], [-0.4, 0.9]):
        np.testing.assert_allclose(d_omega(own, q)[0, 1], 2.0, atol=1e-14)

    both = pair(k_insured, k_insurer, True)
    star = RHO1 / (RHO1 + RHO2)
    for p0 in (P0, 0.0, 5.0, -3.0):
        rows = traj(pair(k_insured, k_insurer, True, p0=p0), 20000)
        np.testing.assert_allclose(rows[-1, 0], star, atol=1e-12)
        np.testing.assert_allclose(rows[-1, 0], 0.8, atol=1e-12)
        assert np.all(rows[:, 1] == p0)                      # the premium never moves, bit for bit
    np.testing.assert_allclose(np.abs(d_omega(both, [A0, P0])).max(), 0.0, atol=1e-14)

    # the total is the whole reason it moves: with no risk it is zero and nothing happens (test 1)
    total = lambda a, p: k_insured(a, p) + k_insurer(p, a)
    assert abs(float(jax.grad(lambda p: total(0.3, p))(0.7))) < 1e-14   # no price in the total
    np.testing.assert_allclose(float(jax.grad(lambda a: total(a, 0.7))(star)), 0.0, atol=1e-7)


# ---------------------------------------------------------------------------
# Test 4: the own-gradient pair gets a price and loses the trade.
# ---------------------------------------------------------------------------


def test_4_the_own_gradient_pair_gets_a_price_and_loses_the_trade():
    """REPRODUCES the closed forms for the spiral's modulus and its fixed point, and CHECKS
    prediction (4)."""
    J = np.array([[1.0 - MU * RHO1 * SIG2, -MU], [MU, 1.0]])
    det = float(np.linalg.det(J))
    np.testing.assert_allclose(det, 0.9845, rtol=1e-12)
    ev = np.linalg.eigvals(J)
    assert np.max(np.abs(ev.imag)) > 0.0
    np.testing.assert_allclose(np.abs(ev).max(), np.sqrt(det), rtol=1e-12)
    np.testing.assert_allclose(np.abs(ev).max(), 0.99222, atol=1e-5)

    choke = LAM + RHO1 * SIG2
    np.testing.assert_allclose(choke, 0.48, rtol=1e-14)
    for a0, p0 in ((A0, P0), (0.2, 0.05), (1.0, 0.9)):
        rows = traj(pair(k_insured, k_insurer, False, a0=a0, p0=p0), 20000)
        np.testing.assert_allclose(rows[-1, 0], 0.0, atol=1e-10)      # no cover at all
        np.testing.assert_allclose(rows[-1, 1], choke, atol=1e-10)    # at the insured's choke price
        assert np.abs(rows[-1] - rows[-2]).max() < 1e-12


# ---------------------------------------------------------------------------
# Test 5: what happens to the insurance as the insurer approaches zero wealth.
# ---------------------------------------------------------------------------


def test_5_as_the_insurers_wealth_falls_its_premium_rises_to_the_choke_and_the_cover_vanishes():
    """REPRODUCES the closed forms ``d*(W)`` and ``ahat*(W)`` of ``quoted`` and CHECKS
    prediction (5). The wealth is frozen (mobility zero), so each run reads the premium the
    insurer quotes at one balance sheet; the mobility of the premium is fixed from the
    curvature at that wealth (``mu_for``), so the sweep over the wealth is not also a sweep
    over the step."""
    Ws = [1.0, 0.1, 0.05, 0.02, 0.01]
    quotes, covers = [], []
    for W in Ws:
        mu = mu_for(W)
        rows = traj(market(w0=W, mu=mu, mu_w=0.0), 4000)
        a, p, Wend = rows[-1]
        np.testing.assert_allclose(Wend, W, atol=0.0)                 # frozen, bit for bit
        d_star, ahat_star = quoted(W)
        np.testing.assert_allclose(p - LAM, d_star, rtol=1e-9)
        np.testing.assert_allclose(a, ahat_star, rtol=1e-7)           # and the insured agrees
        quotes.append(p - LAM)
        covers.append(a)

    np.testing.assert_allclose(quotes, [0.18045, 0.216, 0.27, 0.33517, 0.35308], atol=1e-5)
    np.testing.assert_allclose(covers, [0.49875, 0.4, 0.25, 0.06897, 0.019231], atol=1e-5)
    assert np.all(np.diff(quotes) > 0.0)                              # dearer as it gets poorer
    assert np.all(np.diff(covers) < 0.0)                              # and it writes less

    monopoly, choke = ADEM / (2.0 * GAM), ADEM / GAM
    np.testing.assert_allclose([monopoly, choke], [0.18, 0.36], rtol=1e-12)
    np.testing.assert_allclose(choke, RHO1 * SIG2, rtol=1e-12)        # the insured's own choke
    np.testing.assert_allclose(quoted(1e6)[0], monopoly, rtol=1e-9)
    np.testing.assert_allclose(quoted(1e-6)[0], choke, rtol=1e-9)
    # the cover vanishes as W^2, and at no positive wealth does it reach zero
    for W in (1e-3, 1e-4):
        np.testing.assert_allclose(quoted(W)[1], ADEM * W ** 2 / (GAM * BETA * SIG2), rtol=2e-3)
        assert quoted(W)[1] > 0.0


# ---------------------------------------------------------------------------
# Test 6: and the insurer cannot get there by itself; being wrong is what takes it there.
# ---------------------------------------------------------------------------


def test_6_being_wrong_drives_the_wealth_down_and_low_wealth_drives_the_premium_up():
    """REPRODUCES two closed forms and **REFUTES the second half of prediction (6)**, which is
    left above exactly as it was written.

    *What holds.* With a correct claim forecast the insurer's wealth is monotone increasing at
    every step of twenty thousand: it charges more than it pays, so near-ruin is an initial
    condition and never something that happens. And there is a single critical error, and it is
    the one predicted: ``DEL* = RHO1 SIG2 = 0.36``, the insured's choke markup.

    *What is refuted.* The prediction named two thresholds and a crossing, and both are wrong,
    because they were read off an insurer whose demand curve moved with its claim forecast. With
    the demand curve right and only the claim rate wrong, the markup it settles on at large
    wealth is ``ADEM/(2 GAM) - DEL/2 = 0.18 - DEL/2`` (measured ``0.130001``, ``0.060002``,
    ``0.030006`` at ``DEL = 0.10, 0.24, 0.30``), which is positive for every error below
    ``0.36``: **below the critical error the wealth grows without bound and there is no hard
    market at all**, where the prediction had one from ``0.18`` up.

    Above ``0.36`` the wealth does not fall through zero. It settles, at

        W* = sqrt(GAM BETA SIG2 / (u - 1)),      u = DEL / (RHO1 SIG2),

    measured ``0.113393``, ``0.0707118``, ``0.0500003`` at ``DEL = 0.50, 0.72, 1.08`` against
    ``0.113389``, ``0.0707107``, ``0.05``, and reached from above and from below. And at ``W*``
    the markup is **exactly zero to six decimals and the cover is exactly one**: the insurer
    ends up writing full cover at precisely the world's claim rate. Not from honesty -- it still
    believes claims are cheap -- but because at that wealth its risk loading exactly cancels the
    margin its wrong forecast told it to give away. **So the insurer never fails, at any
    forecast error however large.** The barrier does not hold the wealth up by pushing on it;
    it holds it up by making the insurer price its way out, and the price it arrives at is the
    fair one.

    *And where a run does cross zero, it is the step and not the arrangement.* At a wealth
    mobility of ``1e-3`` the ``DEL = 10`` run overshoots ``W* = 0.0136646`` in a single step and
    is never defined again (``log W`` of a negative); at ``1e-4`` and ``1e-5`` the same
    arrangement settles on ``W*`` to five digits and the wealth is monotone down and positive
    throughout. So even the one way this construction could be made to die is the explicit
    integrator taking too long a stride, which is the suite's standing division: the settled
    value is the arrangement's and which zero a run reaches is the integrator's.

    That is the sharpest form of the thing the record half already showed and (5) repeated: this
    mathematics has no ruin in it, and what it puts there instead is a boundary approached and
    never reached."""
    choke = RHO1 * SIG2
    np.testing.assert_allclose(choke, 0.36, rtol=1e-14)

    def stable_wealth(DEL):
        """``W*`` where the markup the insurer is driven to exactly covers its error, or ``inf``
        below the critical error."""
        u = DEL / choke
        return np.sqrt(GAM * BETA * SIG2 / (u - 1.0)) if u > 1.0 else np.inf

    def run(DEL, w0, steps=400000, mu_w=1e-3):
        """The premium's mobility is fixed from the curvature at the *smallest* wealth the run
        visits, not at its start: the curvature grows as ``1/W^2``, so a step chosen when rich
        is unstable by the time the insurer is poor."""
        low = min(w0, stable_wealth(DEL))
        return traj(market(w0=w0, mu=mu_for(low), mu_w=mu_w, lamhat=LAM - DEL), steps)

    # (a) a correct forecast: monotone up, every step, from a near-ruin start
    rows = traj(market(w0=0.05, mu=mu_for(0.05), mu_w=1e-3, lamhat=LAM), 20000)
    assert np.all(np.diff(rows[:, 2]) > 0.0)
    assert rows[-1, 2] > 0.05

    # (b) below the critical error: unbounded growth, at the markup ADEM/(2 GAM) - DEL/2
    for DEL, markup in ((0.10, 0.13), (0.24, 0.06), (0.30, 0.03)):
        rows = run(DEL, 1.0)
        assert np.all(np.diff(rows[:, 2]) > 0.0)               # no hard market anywhere
        assert rows[-1, 2] > 10.0
        np.testing.assert_allclose(rows[-1, 1] - LAM, markup, atol=1e-5)
        np.testing.assert_allclose(markup, ADEM / (2.0 * GAM) - DEL / 2.0, atol=1e-14)

    # (c) above it: a hard market that settles at W*, from above and from below, pricing at cost
    for DEL, W_star in ((0.50, 0.113389), (0.72, 0.0707107), (1.08, 0.05)):
        closed = stable_wealth(DEL)
        np.testing.assert_allclose(closed, W_star, atol=1e-6)
        for w0 in (1.0, 0.3 * closed):
            rows = run(DEL, w0)
            W, p, a = rows[:, 2], rows[-1, 1], rows[-1, 0]
            np.testing.assert_allclose(rows[-1, 2], closed, rtol=1e-4)
            assert np.all(W > 0.0)                             # and never through zero
            np.testing.assert_allclose(p - LAM, 0.0, atol=1e-5)    # exactly the world's rate
            np.testing.assert_allclose(a, 1.0, atol=1e-4)          # writing full cover

    # (d) the crossing the prediction named does not happen, at any error -- and where a run
    #     does cross, it is the explicit wealth step overshooting, not the arrangement
    for DEL in (0.40, 2.0, 10.0):
        W = run(DEL, 1.0, 400000, mu_w=1e-5)[:, 2]
        assert np.all(W > 0.0) and np.all(np.isfinite(W))
        assert np.all(W >= stable_wealth(DEL) - 1e-9)        # approached from above, never past
        assert np.all(np.diff(W[50:]) <= 0.0) and W[-1] < W[50]   # and monotonically, once
        #     the premium has come down past LAM (for the first few steps the insurer still
        #     holds the over-LAM premium it started with and is still making money on it), the
        #     difference reaching exactly zero where the run has settled on ``W*``
    coarse = run(10.0, 1.0, 400000, mu_w=1e-3)[:, 2]
    assert np.min(coarse) < 0.0                              # the same error, a hundred times
    fine = run(10.0, 1.0, 400000, mu_w=1e-4)[:, 2]           # the wealth step: one overshoot
    np.testing.assert_allclose(fine[-1], stable_wealth(10.0), rtol=1e-5)
