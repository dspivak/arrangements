"""Insurance: a forecast held as state, and charged when it is checked.

David, 2026-09-26: "insurance companies are basically paid by past forecasts being right,
right? is there a way to build that?" -- and, of the fire (`tests/test_fire.py`), "i feel like
this is back to the insurance agency". The construction proposed in
`notes/TESTS-SYNTHESIS-2026-09-26.md` §4 was: one 0-ary box holds the forecast as its state,
published as the doodad publishes its alarm (`tests/test_monitor_pulling.py`); another
accumulates what arrives; and the forecaster's pay reads both. Whether it is exact was left
open. This file builds it and settles that.

An insurer holds a premium rate ``p``: its forecast of what the world will cost it, per step.
The world sends claims at the rate ``LAM``. A fund ``f`` accumulates the difference, premiums
in and claims out, so ``f`` is the record of how right the insurer's past forecasts have been --
and it is money. The insurer is charged by that record: what moves its premium is the state of
the fund, which is the whole of its past and nothing of its present.

**The four predictions, fixed here before any construction code existed, and each able to
fail.**

    (i)   The record is one-way. With no response the fund is exactly ``f_0 + n mu (p_0 - LAM)``
          and the premium does not move at all. Written as the note wrote it -- the pay a cost
          the insurer carries on the fund it reads -- the charge lands on the fund and not on
          the premium, since the forcing on a coordinate is the cost's derivative in *that*
          coordinate: the premium stays where it started, bit for bit, and the fund is merely
          drained toward ``(p_0 - LAM)/c``.
          **Able to fail:** if the premium moves under that wiring.

    (ii)  On ``(p, f)`` the composite's ``d omega`` is ``1 - g``, where ``g`` is the coefficient
          with which the fund pushes the premium. So no correcting response (``g < 0``) is
          exact; the unique exact completion is ``g = +1``, which raises the premium when the
          fund is full. It should run away: both coordinates grow as ``(1 + mu)^n``.
          **Able to fail:** some correcting gain could have turned out exact, or the exact one
          could have corrected anyway.

    (iii) With a correcting response ``g = -c`` the loop is a rotation. ``E = (p - LAM)^2 + c
          f^2`` is multiplied by exactly ``1 + mu^2 c`` each step and the angle per step is
          ``arctan(mu sqrt c)``, so it does not settle: a forecaster charged on its record
          cycles. And the premium is furthest from the true rate exactly a quarter turn after
          it starts, which is exactly when the fund crosses zero -- it is wrong now to pay for
          having been wrong before, and it is still wrong when the debt is paid.
          **Able to fail:** it might settle; the extremum and the crossing might not coincide.

    (iv)  With a view of its own -- an exact cost ``(alpha/2)(p - pbar)^2`` on its own premium --
          the loop spirals in and settles at ``p = LAM`` with a permanent balance
          ``f* = alpha (pbar - LAM) / c``, which is zero only if the view was right.
          **Able to fail:** if it settled at ``f = 0`` whatever the view.

**What this does not model, and says so rather than hiding it.** There is no risk here: the
dynamics is deterministic and the claim rate is a constant, so "bearing risk" -- which David
reads as being able to disintegrate from being wrong (`AGENTDOCPLAN.md` §7 item 35) -- is not
in it. There is no pooling and there are no customers: nothing here makes over-charging costly,
so the insurer is charged for being *wrong*, in either direction, and not for being unprofitable.
What the construction does carry is the part David asked for: a forecast held as present state,
a record of what arrived that is also present state, and a charge that falls on the forecaster
later.

Every docstring below says which of three things its test does (STATUS.md §3.7): reproduce a
closed form derived in the docstring, check a discrete analogue, or correct a claim made on
paper.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact, zero_form
from arrangements.coalgebra import configuration
from arrangements.rvect import euclidean
from arrangements.wiring import compose_seq, tensor_arrangements, trivial_parameter

from tests.chain import flat_state
from tests.test_rendezvous import run_closed, run_phi

EMPTY = jnp.zeros(0)

LAM = 0.12       # the world's claim rate: what arrives each step, whatever anyone forecasts
P0 = 0.05        # the premium the insurer starts holding: it under-charges
F0 = 0.0         # the fund it starts with
C = 0.01         # the gain of the correcting response: how hard the fund moves the premium
MU = 0.05        # the mobility of the premium and of the fund
PBAR = 0.05      # the insurer's own view of the rate, in part 5
ALPHA = 0.1      # how stiffly it holds that view


# ---------------------------------------------------------------------------
# The two boxes and the coupler between them.
# ---------------------------------------------------------------------------


def insurer(p0=P0, mu=MU, alpha=0.0, pbar=PBAR):
    """The forecaster: a 0-ary box whose configuration is the premium rate it holds, published
    at its port as the doodad publishes its alarm (`tests/test_monitor_pulling.py`). Its own
    forcing is the exact cost ``(alpha/2)(p - pbar)^2`` of holding a view of its own, and is
    zero at ``alpha = 0``: in parts 1-4 nothing about the premium comes from the insurer."""
    return Arrangement(
        P=configuration(euclidean(1, mu), init=jnp.array([float(p0)])),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: EMPTY,
        omega=exact(lambda q, m_out, n_in: -0.5 * alpha * (q[0] - pbar) ** 2),
        label="insurer")


def fund(f0=F0, mu=MU):
    """The record: a 0-ary box whose configuration is the accumulated balance, published at its
    port, with no forcing of its own. Everything that happens to it comes through the coupler,
    which is what makes it a record of something other than itself."""
    return Arrangement(
        P=configuration(euclidean(1, mu), init=jnp.array([float(f0)])),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: EMPTY,
        omega=zero_form,
        label="fund")


def ledger_F(g, lam=LAM):
    """The returned covector of the coupler on the two published ports ``e = (p, f)``
    (`tests/test_two_tanks.py`'s idiom, as the pulling test uses it):

        F_f(e) = p - lam     premiums in at the rate the insurer holds, claims out at the
                             world's rate: the fund accumulates the forecast's error;
        F_p(e) = g f         the fund pushes the premium, at the gain ``g``.

    ``g = -c`` corrects (a fat fund lowers the premium, an empty one raises it); ``g = 0`` is no
    response at all; ``g = +1`` is the unique exact completion (test 2)."""

    def F(e):
        return jnp.array([g * e[1], e[0] - lam])

    return F


def coupler(F, ports=2, back=0):
    """A stateless coupler whose forcing on its ``ports`` is the returned covector ``F``. With
    ``back = 1`` it also delivers the fund's published value to the first box's input port,
    which is the wiring part 1's second half needs."""
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=back, out_dim_M=ports, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, e: EMPTY,
        in_f=lambda q, e, n_in: e[1:1 + back],
        omega=lambda q, e, n_in: (EMPTY, F(e), EMPTY),
        label="ledger")


def insured(g, p0=P0, f0=F0, mu=MU, alpha=0.0, pbar=PBAR, lam=LAM):
    """``compose_seq(tensor(insurer, fund), coupler)``: a closed arrangement whose configuration
    is ``(p, f)``, the premium the insurer holds and the record of what its forecasts have cost.
    """
    F = ledger_F(g, lam)
    boxes = tensor_arrangements([insurer(p0, mu, alpha, pbar), fund(f0, mu)])
    return compose_seq(boxes, coupler(F)), F


def paid_but_unmoved(c=C, p0=P0, f0=F0, mu=MU, lam=LAM):
    """The note's design taken literally: "the forecaster's pay reads both". The insurer carries
    the cost ``(c/2) f^2`` on the fund it receives at its input port, and the coupler accumulates
    ``p - lam`` on the fund and routes the fund's published value back to the insurer. The
    composite's configuration is again ``(p, f)``."""
    paid = Arrangement(
        P=configuration(euclidean(1, mu), init=jnp.array([float(p0)])),
        in_dim_M=0, out_dim_M=0, in_dim_N=1, out_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: EMPTY,
        omega=exact(lambda q, m_out, n_in: -0.5 * c * n_in[0] ** 2),
        label="insurer(paid)")
    F = lambda e: jnp.array([0.0 * e[0], e[0] - lam])
    return compose_seq(tensor_arrangements([paid, fund(f0, mu)]), coupler(F, back=1))


# ---------------------------------------------------------------------------
# Running, and the two closed forms the runs are checked against.
# ---------------------------------------------------------------------------


def traj(b, steps):
    """The whole trajectory of a closed composite, inside a ``lax.scan`` (test 0 checks it
    against ``Phi``). Returns ``steps`` rows of ``(p, f)``."""
    P, omega = b.P, b.omega

    @jax.jit
    def step(s, _):
        xi_q, _, _ = omega(P.position(s), EMPTY, EMPTY)
        s2 = P.step(s, xi_q)
        return s2, flat_state(s2)

    return np.asarray(jax.lax.scan(step, P.init, None, length=steps)[1])


def d_omega(F):
    """``(d omega)_{ij} = d_i F_j - d_j F_i``, the antisymmetric part of the returned-covector
    Jacobian (the reciprocity lemma, §5.5), exactly as `tests/test_mutual_need.py` and the
    pulling test compute it."""
    def at(e):
        J = jax.jacobian(F)(jnp.asarray(e, float))    # J[j, i] = d F_j / d e_i
        return np.asarray(J.T - J)
    return at


def rotation(n, p0=P0, f0=F0, c=C, mu=MU, lam=LAM):
    """The closed form for the correcting loop. In ``(A, B) = (p - lam, sqrt(c) f)`` the step is

        A' = A - mu sqrt(c) B,   B' = B + mu sqrt(c) A,

    that is ``rho R(theta)`` with ``rho = sqrt(1 + mu^2 c)`` and ``theta = arctan(mu sqrt c)``.
    So ``(A_n, B_n) = rho^n R(n theta) (A_0, B_0)``, and ``E = A^2 + B^2`` is multiplied by
    exactly ``1 + mu^2 c`` each step."""
    s = np.sqrt(c)
    rho, th = np.sqrt(1.0 + mu * mu * c), np.arctan(mu * s)
    A0, B0 = p0 - lam, s * f0
    A = rho ** n * (A0 * np.cos(n * th) - B0 * np.sin(n * th))
    B = rho ** n * (A0 * np.sin(n * th) + B0 * np.cos(n * th))
    return np.stack([lam + A, B / s], axis=-1)


def runaway(n, p0=P0, f0=F0, mu=MU, lam=LAM):
    """The closed form for the exact completion ``g = +1``. In ``(A, f) = (p - lam, f)`` the step
    is ``A' = A + mu f``, ``f' = f + mu A``, with eigenvectors ``(1, 1)`` and ``(1, -1)`` and
    eigenvalues ``1 + mu`` and ``1 - mu``; so with ``a = (A_0 + f_0)/2`` and ``b = (A_0 -
    f_0)/2``, ``A_n = a (1+mu)^n + b (1-mu)^n`` and ``f_n = a (1+mu)^n - b (1-mu)^n``."""
    A0 = p0 - lam
    a, b = 0.5 * (A0 + f0), 0.5 * (A0 - f0)
    up, dn = (1.0 + mu) ** n, (1.0 - mu) ** n
    return np.stack([lam + a * up + b * dn, a * up - b * dn], axis=-1)


# ---------------------------------------------------------------------------
# Test 0: the runs are Phi's.
# ---------------------------------------------------------------------------


def test_0_the_runs_are_the_dynamics_functors():
    """LICENSES the run form used below; it establishes nothing about the predictions. For the
    correcting loop, the exact completion, the loop with a view of its own and the note's own
    wiring, ``Phi``, ``run_closed`` and the ``lax.scan`` of ``traj`` agree over fifty steps to
    within one unit in the last place."""
    for b in (insured(-C)[0], insured(1.0)[0], insured(-C, alpha=ALPHA)[0], paid_but_unmoved()):
        by_phi = run_phi(b, 50, EMPTY)
        by_closed = run_closed(b, 50, lambda t: EMPTY)
        assert float(np.max(np.abs(by_phi - by_closed))) < 1e-15
        np.testing.assert_array_equal(traj(b, 50)[-1], by_closed[-1])


# ---------------------------------------------------------------------------
# Test 1: the record is one-way, and a pay the forecaster merely reads does not move it.
# ---------------------------------------------------------------------------


def test_1_the_record_accumulates_the_forecasts_error_and_does_not_move_the_forecaster():
    """REPRODUCES two closed forms and CORRECTS the design in
    `notes/TESTS-SYNTHESIS-2026-09-26.md` §4, "the forecaster's pay reads both".

    With no response (``g = 0``) the premium is bit-for-bit what it started as over 2000 steps,
    and the fund is ``f_n = f_0 + n mu (p_0 - LAM)`` to 1e-13 relative: at ``mu (p_0 - LAM) =
    -0.0035`` a step it is at ``-7`` after 2000. That is the record doing its work: what the
    insurer holds is present state, and the fund turns a standing forecast into an accumulated
    charge without anything remembering anything.

    Now give the insurer the pay the note's design gives it -- the cost ``(c/2) f^2`` on the fund
    it reads at its port. **The premium still does not move, bit for bit.** The forcing on a
    coordinate is the cost's derivative in *that* coordinate, and this cost does not mention the
    premium; what the charge does reach is the fund, which it drains toward the balance where the
    two forcings cancel, ``f* = (p_0 - LAM)/c = -7``, geometrically at the exact rate
    ``1 - mu c = 0.9995`` a step (so ``-4.4255`` at 2000 steps, and ``-7`` to 1e-12 by 60 000).
    So a pay that merely reads the record charges the record. For the record to move the
    forecaster something must return to the premium, and that is the coupler of test 2."""
    b, _ = insured(0.0)
    rows = traj(b, 2000)
    assert np.all(rows[:, 0] == P0)                              # the premium does not move
    n = np.arange(1, 2001)
    np.testing.assert_allclose(rows[:, 1], F0 + n * MU * (P0 - LAM), rtol=1e-13)
    np.testing.assert_allclose(rows[-1, 1], -7.0, rtol=1e-12)

    rows2 = traj(paid_but_unmoved(), 2000)
    assert np.all(rows2[:, 0] == P0)                             # still does not move
    star = (P0 - LAM) / C
    np.testing.assert_allclose(rows2[:, 1], star + (F0 - star) * (1.0 - MU * C) ** n, rtol=1e-12)
    np.testing.assert_allclose(rows2[-1, 1], -4.425487835, rtol=1e-9)
    rows3 = traj(paid_but_unmoved(), 60000)
    assert np.all(rows3[:, 0] == P0)
    np.testing.assert_allclose(rows3[-1, 1], star, rtol=1e-12)
    assert abs(rows3[-1, 1] - rows3[-2, 1]) < 1e-12              # and the fund has settled


# ---------------------------------------------------------------------------
# Test 2: no correcting response is exact, and the exact one runs away.
# ---------------------------------------------------------------------------


def test_2_no_correcting_response_is_exact_and_the_exact_one_runs_away():
    """REPRODUCES the closed form for ``d omega`` and for the exact completion's run, and so
    settles the question the note left open ("whether it is exact is open").

    On ``e = (p, f)`` the coupler returns ``F = (g f, p - LAM)``, so

        (d omega)_{p f} = d_p F_f - d_f F_p = 1 - g,

    the only nonzero entry, and the composite's forcing on the configuration carries the same one
    -- the insurer's own cost is the differential of a cost and contributes nothing
    antisymmetric. **So the loop is exact at ``g = 1`` and at no other gain**: ``1.01`` for the
    correcting ``g = -0.01``, ``1`` for no response at all, ``2`` for ``g = -1``. The record is
    one-way by its own construction, since it reads the premium and returns nothing, and the
    single response that repairs that is the one which *raises* the premium when the fund is
    full.

    And that response is not a correction but an amplifier. With ``g = 1`` the step on
    ``(p - LAM, f)`` has eigenvalues ``1 +- mu``, so the run leaves along the growing
    eigenvector: the insurer starts under-charging by ``0.07``, the fund goes into deficit, the
    deficit lowers the premium further, and after 200 steps the premium is ``-605.12`` and the
    fund ``-605.24`` -- it pays people to insure them. The measured trajectory is the closed form
    ``runaway`` to 1e-11, and by step 200 the step-to-step growth is ``1 + mu = 1.05`` to nine
    digits, the decaying mode having fallen to ``(0.95/1.05)^200 = 2e-9`` of the growing one."""
    for g, expected in ((-C, 1.0 + C), (0.0, 1.0), (-1.0, 2.0), (1.0, 0.0)):
        F = ledger_F(g)
        M = d_omega(F)([P0, -1.3])
        np.testing.assert_allclose(M[0, 1], expected, atol=1e-16)
        np.testing.assert_allclose(np.abs(M).max(), expected, atol=1e-16)

        b, _ = insured(g)
        q0 = jnp.asarray(flat_state(b.P.init))
        J = jax.jacobian(lambda q: b.omega(q, EMPTY, EMPTY)[0])(q0)
        comp = np.asarray(J.T - J)
        np.testing.assert_allclose(np.abs(comp).max(), expected, atol=1e-16)

    # with a view of its own the verdict is unchanged: an exact cost adds nothing antisymmetric
    b_view, _ = insured(-C, alpha=ALPHA)
    J = jax.jacobian(lambda q: b_view.omega(q, EMPTY, EMPTY)[0])(jnp.asarray([P0, -1.3]))
    np.testing.assert_allclose(np.abs(np.asarray(J.T - J)).max(), 1.0 + C, atol=1e-16)

    b_ex, _ = insured(1.0)
    rows = traj(b_ex, 200)
    np.testing.assert_allclose(rows, runaway(np.arange(1, 201)), rtol=1e-11)
    np.testing.assert_allclose(rows[-1], [-605.1203, -605.2403], atol=1e-3)
    assert np.all(np.diff(rows[:, 0]) < 0.0)                     # the premium falls, and falls
    ratio = (rows[-1, 1] - rows[-2, 1]) / (rows[-2, 1] - rows[-3, 1])
    np.testing.assert_allclose(ratio, 1.0 + MU, rtol=1e-9)


# ---------------------------------------------------------------------------
# Test 3: the correcting loop is a rotation, and the premium lags the record by a quarter turn.
# ---------------------------------------------------------------------------


def test_3_charged_on_its_record_the_forecaster_cycles_and_lags_a_quarter_turn():
    """REPRODUCES the closed form ``rotation``, and CHECKS prediction (iii), which holds.

    Start the insurer holding exactly the right rate, ``p = LAM``, with a fund of ``-1``: a past
    of under-charging, and nothing wrong with the present forecast. It is charged anyway, because
    what it is charged on is the record. The measured run is the closed form
    ``rho^n R(n theta) (A_0, B_0)`` to 1e-12 over four thousand steps, with
    ``theta = arctan(mu sqrt c) = 0.00499996`` and ``rho = 1.0000125``:

    * the premium rises at once, to ``LAM + 5e-4`` after one step, and keeps rising;
    * ``E = (p - LAM)^2 + c f^2`` is multiplied by exactly ``1 + mu^2 c = 1.000025`` each step
      (to ``2.4e-13`` relative after four thousand of them, which is their accumulated
      rounding), so the loop does not settle -- the cycle is the arrangement's and the slow growth
      is the integrator's, two explicit steps reading each other's old position;
    * the fund crosses zero at steps ``315``, ``1571``, ``2828``, and the premium's local maxima
      are at steps ``315``, ``1571``, ``2828`` -- **the same steps, a quarter turn
      (``(pi/2)/theta = 314.2``) after the start and a full period (``1256.6``) apart.** The
      debt is paid exactly when the over-charging is worst, at ``LAM + 0.100394``.

    That is what "paid by past forecasts being right" comes to in this model. The insurer is
    wrong now to pay for having been wrong before, and it is still wrong when the account is
    square, so it over-shoots into surplus and the whole thing turns over again: with these
    numbers the full period is ``1256.6`` steps."""
    b, _ = insured(-C, p0=LAM, f0=-1.0)
    rows = traj(b, 4000)
    n = np.arange(1, 4001)
    np.testing.assert_allclose(rows, rotation(n, p0=LAM, f0=-1.0), atol=1e-12)

    th, rho = np.arctan(MU * np.sqrt(C)), np.sqrt(1.0 + MU * MU * C)
    np.testing.assert_allclose([th, rho], [0.004999958, 1.0000125], rtol=1e-6)
    np.testing.assert_allclose(rows[0, 0], LAM + MU * C * 1.0, rtol=1e-14)

    E = (rows[:, 0] - LAM) ** 2 + C * rows[:, 1] ** 2
    E0 = C * 1.0
    np.testing.assert_allclose(E, E0 * (1.0 + MU * MU * C) ** n, rtol=1e-12)
    assert np.all(np.diff(E) > 0.0)                              # it does not settle

    peaks = np.where(np.diff(np.sign(np.diff(rows[:, 0]))) < 0)[0] + 2
    crossings = np.where(np.diff(np.sign(rows[:, 1])) > 0)[0] + 2
    np.testing.assert_array_equal(peaks[:3], crossings[:3])      # the same steps, to the step
    np.testing.assert_array_equal(peaks[:3], [315, 1571, 2828])
    np.testing.assert_allclose(0.5 * np.pi / th, 314.16, atol=0.01)
    np.testing.assert_allclose(2.0 * np.pi / th, 1256.6, atol=0.1)
    np.testing.assert_allclose(np.diff(peaks[:3]), 2.0 * np.pi / th, atol=1.0)
    peak = int(peaks[0])
    np.testing.assert_allclose(rows[peak - 1, 0] - LAM, 0.100394, atol=1e-6)
    np.testing.assert_allclose(rows[peak - 1, 0] - LAM, np.sqrt(E0) * rho ** peak, rtol=1e-5)

    # and it is over-charging throughout the stretch in which the debt is repaid
    assert np.all(rows[:peak, 0] > LAM)
    assert np.all(rows[:peak - 1, 1] < 0.0)


# ---------------------------------------------------------------------------
# Test 4: with a view of its own the loop settles, holding a balance it never clears.
# ---------------------------------------------------------------------------


def test_4_a_view_of_its_own_settles_the_loop_at_a_balance_it_never_clears():
    """REPRODUCES a closed form and CHECKS prediction (iv), which holds.

    Give the insurer an exact cost ``(alpha/2)(p - pbar)^2`` of its own -- a view of the rate it
    holds against what the record tells it. The step's Jacobian is
    ``[[1 - mu alpha, -mu c], [mu, 1]]``, determinant ``1 - mu alpha + mu^2 c = 0.995025``, so
    the eigenvalues are complex with modulus ``0.997509``: the rotation of test 3 becomes a
    spiral inward, and it settles.

    Where it settles is the point of the test. At rest the fund is still, so the premium is the
    world's rate exactly, ``p = LAM = 0.12`` -- the forecast comes right whatever the view. But
    the premium is still, too, so the view's pull must be balanced by the record's, and that
    fixes a permanent balance

        f* = alpha (pbar - LAM) / c = -0.7,

    measured to 1e-12: an insurer whose view was ``0.07`` below the world's rate ends holding a
    deficit of ``0.7`` for ever, and one whose view was right ends at zero. The account comes to
    zero only for a forecaster that was right to begin with; every other one buys a right
    forecast with a standing debt, and the stiffer its view the larger that debt."""
    b, _ = insured(-C, alpha=ALPHA)
    rows = traj(b, 20000)
    np.testing.assert_allclose(rows[-1, 0], LAM, atol=1e-12)
    np.testing.assert_allclose(rows[-1, 1], ALPHA * (PBAR - LAM) / C, atol=1e-12)
    np.testing.assert_allclose(rows[-1, 1], -0.7, atol=1e-12)
    assert np.abs(rows[-1] - rows[-2]).max() < 1e-14             # settled

    J = np.array([[1.0 - MU * ALPHA, -MU * C], [MU, 1.0]])
    det = float(np.linalg.det(J))
    np.testing.assert_allclose(det, 1.0 - MU * ALPHA + MU * MU * C, rtol=1e-14)
    np.testing.assert_allclose(det, 0.995025, rtol=1e-9)
    ev = np.linalg.eigvals(J)
    assert np.max(np.abs(ev.imag)) > 0.0                         # a spiral, not two real modes
    np.testing.assert_allclose(np.abs(ev).max(), np.sqrt(det), rtol=1e-12)
    np.testing.assert_allclose(np.abs(ev).max(), 0.9975094, rtol=1e-6)

    # a view that was right leaves no balance; a stiffer view leaves a larger one
    b_right, _ = insured(-C, alpha=ALPHA, pbar=LAM)
    np.testing.assert_allclose(traj(b_right, 20000)[-1], [LAM, 0.0], atol=1e-12)
    for alpha in (0.05, 0.2):
        b_a, _ = insured(-C, alpha=alpha)
        np.testing.assert_allclose(traj(b_a, 40000)[-1, 1], alpha * (PBAR - LAM) / C, atol=1e-10)
