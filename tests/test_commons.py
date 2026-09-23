"""The commons: three wirings on one cost, and the fifteen predictions of §9.

The design is ``articulating-forms-that-matter/notes/COMMONS-DESIGN.md`` (P8.1), written
in full before any of this existed; section and prediction numbers below are its. **Every
docstring says which of three things its test does** -- reproduce a closed form the note
derives, check a discrete-time analogue the note proves only for the ODE (STATUS.md §3.2,
the note's strain S10), or merely illustrate.

What these tests are worth, stated once and not softened (STATUS.md §2, the note's §10):
the settled states are algebra, so reproducing them is a *bookkeeping* check on the
implementation, not evidence for the model. The tests that carry weight are the ones that
could have come out otherwise: the exactness bookkeeping of predictions 7 and 8, the
reciprocity of the sanction against the external enforcer, and the failures recorded in
``notes/COMMONS-RESULTS.md``.

Reference parameters (§9): ``r = 1, K = 100, p = 0.01, gamma = 1``, so ``beta = 1/N`` and
``s* = K/r = 100``. The mobility is the integrator's, not the cost's.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arrangements.commons import (  # noqa
    Params, catch_coupler, catch_coupler_insulated, closed_step, commons,
    composite_port_form, d_omega, external_enforcer, governed, insulated, open_access,
    unmoved_monitor,
    resource, extractor, returned_covector, run, sanction_coupler, settle, split,
)
from arrangements.functors import Phi
from arrangements.interpretation import trivial_omega

PAR = Params()
NS = (1, 5, 20)
S_STAR = PAR.s_optimal()          # 100.0
STEPS = 60_000                    # mobility 0.01, so 600 time units
IN_POS = (jnp.zeros(0), trivial_omega(0))


def settled(arr):
    return split(settle(arr, STEPS))


# ---------------------------------------------------------------------------
# The implementation is the library's own composition.
# ---------------------------------------------------------------------------


def test_the_closed_step_is_Phi():
    """CHECKS THE IMPLEMENTATION, not the model: the fast closed step used by every run
    below is ``Phi`` of the composite, step for step. The identity of
    ``tests/test_rendezvous.py``'s ``run_closed`` -- for a closed arrangement the covector
    triple collapses to the forcing's own ``Q``-component -- is verified, not assumed."""
    a = governed(PAR, 3, s=S_STAR)
    O, step = Phi(a), closed_step(a)
    s_phi, s_fast = O.state, a.P.init
    for _ in range(25):
        _, _, s_phi = O.with_state(s_phi).run_one(IN_POS, lambda _o: (jnp.zeros(0), jnp.zeros(0)))
        s_fast = step(s_fast)
        np.testing.assert_allclose(a.P.position(s_phi), a.P.position(s_fast), rtol=1e-12)


def test_one_cost_three_wirings_no_retuning():
    """ILLUSTRATES §10's F2, the criterion the note sets itself: the three wirings share
    the same boxes, the same ``r, K, p, gamma`` and the same fan-out. Governed differs from
    open access by *one extra coupler* in the tensor and nothing else; insulated differs by
    the catch couplers' one-form and nothing else. Read off the constructor: the three
    calls differ in one keyword."""
    oa, gov, ins = open_access(PAR, 4), governed(PAR, 4, s=S_STAR), insulated(PAR, 4, Rhat=10.0)
    for a in (oa, gov, ins):
        assert (a.in_dim_M, a.out_dim_M, a.in_dim_N, a.out_dim_N) == (0, 0, 0, 0)
        assert a.P.Q_dim == 5                        # (R, x_1, ..., x_4)
    # open access and insulated are the same syntax; governed has one more coupler port.
    np.testing.assert_allclose(oa.P.position(oa.P.init), ins.P.position(ins.P.init))
    np.testing.assert_allclose(oa.P.position(oa.P.init), gov.P.position(gov.P.init))


def test_the_extractors_are_mutually_invisible():
    """REPRODUCES §3's structural claim, which is the design brief for governance: under
    open access ``d omega_{x_i}/d x_j = 0`` for ``i != j`` -- each extractor's port mentions
    no other's effort -- while the resource's port receives ``sum_i x_i``, the transpose of
    the fan-out copy. Neither fact is written anywhere in ``commons.py``; both are read off
    the composite ``compose_seq`` produces. Adding the sanction fills exactly those zeros
    with ``s``."""
    N = 4
    F_oa = composite_port_form(open_access(PAR, N))
    F_gov = composite_port_form(governed(PAR, N, s=0.7))
    q = jnp.array([37.0, 0.11, 0.23, 0.07, 0.31])
    J_oa, J_gov = jax.jacobian(F_oa)(q), jax.jacobian(F_gov)(q)
    off = np.asarray(J_oa)[1:, 1:] - np.diag(np.diag(np.asarray(J_oa)[1:, 1:]))
    np.testing.assert_allclose(off, 0.0, atol=1e-12)          # mutually invisible
    np.testing.assert_allclose(np.asarray(J_oa)[0, 1:], -1.0, atol=1e-12)   # resource feels each
    filled = np.asarray(J_gov)[1:, 1:] - np.asarray(J_oa)[1:, 1:]
    np.testing.assert_allclose(filled, 0.7, atol=1e-12)       # the sanction fills them, uniformly
    np.testing.assert_allclose(np.asarray(J_gov)[0, :], np.asarray(J_oa)[0, :], atol=1e-12)


# ---------------------------------------------------------------------------
# Predictions 1-5: the regenerating variant.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", NS)
def test_prediction_1_open_access_converges(N):
    """PREDICTION 1. REPRODUCES the closed form §5 derives, ``R* = r gamma K/(r gamma + NpK)``
    and ``x_i* = p r K/(r gamma + NpK)``, and CHECKS THE DISCRETE ANALOGUE of the note's
    Dulac/Poincare-Bendixson convergence argument, which is continuous-time (strain S10):
    four interior starts, each converging to the same point, with no cycle. Numerically
    ``R* = 50.00, 16.67, 4.76``."""
    for R0, x0 in ((90.0, 0.01), (5.0, 2.0), (99.0, 0.9), (1.0, 0.001)):
        R, x = settled(open_access(PAR, N, R0=R0, x0=x0))
        np.testing.assert_allclose(R, PAR.R_open_access(N), rtol=2e-5)
        np.testing.assert_allclose(np.asarray(x), PAR.p * R / PAR.gamma, rtol=2e-5)


def test_prediction_1_no_limit_cycle():
    """PREDICTION 1, the "no limit cycle" half. CHECKS THE DISCRETE ANALOGUE: the distance
    to the settled point decreases monotonically over the tail of the run, so the trajectory
    is not orbiting. Not a proof; the note's proof is for the ODE (S10)."""
    N = 5
    traj = np.asarray(run(open_access(PAR, N, R0=95.0, x0=0.9), 40_000, every=200))
    tgt = np.concatenate([[PAR.R_open_access(N)], np.full(N, PAR.p * PAR.R_open_access(N))])
    d = np.linalg.norm(traj - tgt, axis=1)
    live = d[20:][d[20:] > 1e-9]                  # below that it is at the float64 floor
    assert np.all(np.diff(live) < 0.0)
    assert d[-1] < 1e-12 * d[0]


def test_prediction_2_rent_dissipation():
    """PREDICTION 2. REPRODUCES §5's closed forms: ``R*`` strictly decreasing in ``N`` and
    tending to ``0``; settled harvest ``E* = rK beta/(1+beta)^2`` maximized at ``beta = 1``,
    i.e. ``N = gamma r/(Kp) = 1`` here; total rent ``Pi = N p^2 R*^2/(2 gamma)``, numerically
    ``0.125, 0.0694, 0.0227``, falling like ``gamma r^2/(2N)``. Gordon's rent dissipation."""
    Rs = [settled(open_access(PAR, N))[0] for N in NS]
    assert Rs[0] > Rs[1] > Rs[2] > 0
    rents = [PAR.rent(R, N) for R, N in zip(Rs, NS)]
    np.testing.assert_allclose(rents, [0.125, 0.069444, 0.022676], rtol=1e-3)
    assert rents[0] > rents[1] > rents[2]
    harvest = [PAR.harvest(R, PAR.X_of_R(R)) for R in Rs]
    np.testing.assert_allclose(harvest, [PAR.r * PAR.K * PAR.beta(N) / (1 + PAR.beta(N)) ** 2
                                         for N in NS], rtol=1e-3)
    assert harvest[0] == max(harvest)          # maximized at beta = 1, i.e. N = 1
    for N in (100, 200, 400):
        R, x = settled(open_access(PAR, N))
        np.testing.assert_allclose(PAR.rent_from(R, x), PAR.gamma * PAR.r ** 2 / (2 * N),
                                   rtol=0.02)       # Pi_oa ~ gamma r^2/(2N) -> 0


@pytest.mark.parametrize("N", NS)
def test_prediction_3_any_sanction_raises_the_stock(N):
    """PREDICTION 3, and §10's F1 -- the criterion that would kill the reading of the
    sanction's sign. REPRODUCES ``R_gov = K beta'/(1+beta')``, ``beta' = beta + rs/K``, and
    checks it is strictly increasing in ``s`` and reproduces open access at ``s = 0`` to
    machine precision."""
    Rs = []
    for s in (0.0, 1.0, 10.0, 100.0, 1000.0):
        # the mobility is the integrator's: the sanction raises the effort rate to
        # gamma' = gamma + Nps, so the step has to respect gamma' (prediction 9).
        par = Params(mobility=min(0.01, 0.5 / (PAR.gamma + N * PAR.p * s)))
        R, _ = split(settle(governed(par, N, s=s), int(STEPS * 0.01 / par.mobility)))
        np.testing.assert_allclose(R, PAR.R_governed(N, s), rtol=2e-5)
        Rs.append(R)
    np.testing.assert_allclose(Rs[0], settled(open_access(PAR, N))[0], rtol=1e-12)
    assert all(b > a for a, b in zip(Rs, Rs[1:]))
    assert Rs[1] > settled(open_access(PAR, N))[0]      # any s > 0 strictly raises it


def test_prediction_4_s_equals_K_over_r_hits_the_cooperative_optimum():
    """PREDICTION 4. REPRODUCES §4's ``R_gov = R_coop`` iff ``s = K/r``, numerically
    ``66.67, 54.55, 51.22`` at ``N = 1, 5, 20``, and checks the *independence* the note
    claims by varying ``p`` and ``gamma`` as well as ``N``: the same constant ``K/r = 100``
    works for every one. This is the note's own "arranged" half (strain S2) -- the constant
    is clean because the sanction was chosen quadratic in the total."""
    np.testing.assert_allclose([PAR.R_cooperative(N) for N in NS],
                               [66.667, 54.545, 51.220], rtol=1e-4)
    for N in NS:
        R, _ = settled(governed(PAR, N, s=S_STAR))
        np.testing.assert_allclose(R, PAR.R_cooperative(N), rtol=2e-5)
    for p, g in ((0.02, 1.0), (0.01, 4.0), (0.05, 0.5)):
        par = Params(p=p, gamma=g, mobility=0.002)
        for N in (2, 7):
            R, _ = settled(governed(par, N, s=par.s_optimal()))
            np.testing.assert_allclose(R, par.R_cooperative(N), rtol=1e-3)
    # and the optimum is above open access, always (§4: beta(2+beta) < (1+beta)^2)
    for N in NS:
        assert PAR.R_cooperative(N) > PAR.R_open_access(N)


def test_prediction_5_large_N_limits():
    """PREDICTION 5. REPRODUCES the two limits: governed with ``s = K/r`` tends to maximum
    sustainable yield ``R -> K/2 = 50``, ``E -> rK/4 = 25``, rent ``-> pKr/4 = 0.25``, while
    open access tends to ``(0, 0, 0)``. Checked at ``N = 200`` rather than in the limit."""
    N = 200
    par = Params(mobility=0.5 / (PAR.gamma + N * PAR.p * S_STAR))       # gamma' = 201
    R_gov, x_gov = split(settle(governed(par, N, s=S_STAR), 400_000))
    np.testing.assert_allclose(R_gov, 50.0, rtol=2e-2)
    np.testing.assert_allclose(PAR.harvest(R_gov, float(jnp.sum(x_gov))), 25.0, rtol=4e-2)
    np.testing.assert_allclose(PAR.rent_from(R_gov, x_gov), 0.25, rtol=5e-2)
    R_oa, x_oa = settled(open_access(PAR, N))
    assert R_oa < 0.6 and PAR.rent_from(R_oa, x_oa) < 1e-2


# ---------------------------------------------------------------------------
# Prediction 6: the insulated wiring.
# ---------------------------------------------------------------------------


def test_prediction_6_insulated_effort_is_deaf_and_the_threshold_is_exact():
    """PREDICTION 6. REPRODUCES §6's closed forms: ``x_i -> p Rhat/gamma``, constant and
    independent of ``R``; ``R -> K(1 - N p Rhat/(gamma r))`` when ``N p Rhat/gamma < r``, and
    ``R -> 0`` otherwise. At ``N = 5`` the threshold is ``Rhat = 20``: ``Rhat = 10`` settles
    at ``R = 50`` and ``Rhat = 50`` collapses."""
    N = 5
    for Rhat, R_pred in ((10.0, 50.0), (4.0, 80.0), (19.0, 5.0)):
        R, x = settled(insulated(PAR, N, Rhat=Rhat, R0=90.0))
        np.testing.assert_allclose(np.asarray(x), PAR.p * Rhat / PAR.gamma, rtol=1e-6)
        np.testing.assert_allclose(R, R_pred, rtol=1e-3)
    R, x = settled(insulated(PAR, N, Rhat=50.0, R0=90.0))
    assert R < 1e-6                                        # collapse, N p Rhat/gamma = 2.5 > r
    np.testing.assert_allclose(np.asarray(x), PAR.p * 50.0, rtol=1e-9)   # still deaf, still 0.5
    # deaf: the settled effort does not move with the stock's initial condition
    for R0 in (5.0, 99.0):
        _, x2 = settled(insulated(PAR, N, Rhat=10.0, R0=R0))
        np.testing.assert_allclose(np.asarray(x2), PAR.p * 10.0, rtol=1e-6)


# ---------------------------------------------------------------------------
# Predictions 7 and 8: the exactness bookkeeping. These are the load-bearing tests.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", (1, 3, 6))
def test_prediction_7_open_access_and_governed_are_exact_insulated_is_not(N):
    """PREDICTION 7, and §10's F3, the highest-stakes check in the note. REPRODUCES three
    statements about the *composite* one-form, computed from the composite and not from any
    hand-written potential:

    * ``d omega = 0`` for open access and for governed, to machine precision;
    * for insulated the ``(R, x_i)`` component is exactly ``1`` for every ``i`` and every
      ``Rhat`` -- so no choice of stale account makes it exact, which is §6's theorem;
    * ``omega_ins - omega_oa = sum_i (R - Rhat) dx_i`` identically.
    """
    rng = np.random.default_rng(11)
    f_oa = composite_port_form(open_access(PAR, N))
    f_gov = composite_port_form(governed(PAR, N, s=0.37))
    for _ in range(4):
        q = jnp.asarray(np.concatenate([[abs(rng.normal()) + 1.0], rng.standard_normal(N)]))
        np.testing.assert_allclose(d_omega(f_oa)(q), 0.0, atol=1e-11)
        np.testing.assert_allclose(d_omega(f_gov)(q), 0.0, atol=1e-11)
        for Rhat in (0.0, 10.0, 1e4):
            f_ins = composite_port_form(insulated(PAR, N, Rhat=Rhat))
            dw = np.asarray(d_omega(f_ins)(q))
            for i in range(1, N + 1):
                np.testing.assert_allclose(dw[0, i], 1.0, atol=1e-11)   # exactly 1, always
                np.testing.assert_allclose(dw[i, 0], -1.0, atol=1e-11)
            np.testing.assert_allclose(dw[1:, 1:], 0.0, atol=1e-11)
            eta = np.asarray(f_ins(q)) - np.asarray(f_oa(q))
            np.testing.assert_allclose(eta[0], 0.0, atol=1e-11)
            np.testing.assert_allclose(eta[1:], float(q[0]) - Rhat, atol=1e-10)


def test_prediction_8_the_sanction_is_reciprocal():
    """PREDICTION 8, the note's own "single most interesting check in the list", and §10's
    F4. REPRODUCES the reciprocity lemma on the sanction coupler: its returned-covector
    Jacobian is ``s`` times the all-ones matrix, symmetric, antisymmetric part zero, so
    ``d omega_sanc = 0`` and what restrains user ``i`` is the same object as what user ``i``
    exerts on every other -- with no third box. F4 is not met: the mutual sanction is
    available exactly, on its own, with no external box."""
    N, s = 4, 0.6
    F = returned_covector(sanction_coupler(s, N))
    rng = np.random.default_rng(5)
    for _ in range(4):
        e = jnp.asarray(rng.standard_normal(N))
        np.testing.assert_allclose(jax.jacobian(F)(e), s * np.ones((N, N)), atol=1e-12)
        np.testing.assert_allclose(d_omega(F)(e), 0.0, atol=1e-12)


def test_prediction_8_the_note_s_external_enforcer_is_exact_and_that_is_a_correction():
    """PREDICTION 8's second half, and it comes out **against the note**. §4 says a monitor
    that "returns ``sX`` to each user while being itself unmoved by them" has a nonzero
    antisymmetric part. Computed, it does not: that coupler's one-form is ``d((s/2)X^2)``
    with a zero component on the monitor's port, and it is exact. Being unmoved costs
    nothing, because ``X`` does not mention the monitor's coordinate, so there is no mixed
    partial to break.

    What *is* non-exact is a monitor whose own state sets the sanction: returning ``yX`` to
    every user while feeling nothing back gives ``dF_i/dy = X`` against ``dF_y/dx_i = 0``,
    the waterfall of ``tests/test_two_tanks.py``. Among the users it is still symmetric.
    So the note's reading survives, but its executable form was wrong, and the corrected
    statement is narrower: **non-exactness comes from the enforcer having a state of its own
    that the users do not move, not from the enforcer being unmoved.**"""
    N, s = 4, 0.6
    rng = np.random.default_rng(7)
    flat = returned_covector(unmoved_monitor(s, N))
    for _ in range(4):
        e = jnp.asarray(rng.standard_normal(N + 1))
        np.testing.assert_allclose(d_omega(flat)(e), 0.0, atol=1e-12)     # exact after all
    G = returned_covector(external_enforcer(N))
    for _ in range(4):
        e = jnp.asarray(rng.standard_normal(N + 1))
        dw = np.asarray(d_omega(G)(e))
        X, y = float(jnp.sum(e[:N])), float(e[N])
        np.testing.assert_allclose(dw[:N, :N], 0.0, atol=1e-12)           # users: reciprocal
        for i in range(N):
            np.testing.assert_allclose(dw[N, i], X, atol=1e-12)           # monitor: not
            np.testing.assert_allclose(dw[i, N], -X, atol=1e-12)
        assert abs(X) > 1e-3 and np.max(np.abs(dw)) > 1e-3


def test_the_catch_coupler_is_one_term_with_two_faces():
    """REPRODUCES §2's hinge, on which §6 turns: ``U_catch = -x R`` returns ``-x`` to the
    resource port and ``-R`` to the effort port, and ``d^2U/dx dR = -1`` is *one* number.
    The insulated coupler breaks exactly that identity: it keeps ``-x`` on the resource face
    and replaces ``-R`` by ``-Rhat`` on the effort face, so the two mixed partials differ by
    ``1``, whatever ``Rhat`` is."""
    e = jnp.array([37.0, 0.42])
    F = returned_covector(catch_coupler())
    np.testing.assert_allclose(F(e), [-0.42, -37.0], atol=1e-12)
    np.testing.assert_allclose(d_omega(F)(e), 0.0, atol=1e-12)
    G = returned_covector(catch_coupler_insulated(10.0))
    np.testing.assert_allclose(G(e), [-0.42, -10.0], atol=1e-12)
    J = np.asarray(jax.jacobian(G)(e))
    assert abs(J[0, 1] - J[1, 0]) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Prediction 9: discrete versus continuous.
# ---------------------------------------------------------------------------


def test_prediction_9_settled_states_are_mobility_free_and_the_bound_shrinks_with_N():
    """PREDICTION 9. REPRODUCES the first half exactly: the settled states are the zeros of
    ``omega`` and the responses are invertible at interior points, so they are *identical*
    for every mobility -- only the transient differs. CHECKS THE DISCRETE ANALOGUE for the
    second half (STATUS.md §3.2 item 2, the note's S10): each wiring has a finite ``mu_max``
    above which the settled state is repelling, computed as ``2/|lambda_max|`` of the
    mobility-one Jacobian, and it shrinks as ``N`` grows."""
    for mu in (0.02, 0.005, 0.001):
        par = Params(mobility=mu)
        R, _ = settled(open_access(par, 5))
        np.testing.assert_allclose(R, PAR.R_open_access(5), rtol=1e-3)

    def mu_max(N, s=None):
        """``max{mu : |1 + mu lambda| < 1 for every eigenvalue}``, i.e.
        ``min_j (-2 Re lambda_j)/|lambda_j|^2`` -- **not** ``2/|lambda_max|``, which is only
        the real case; the settled state here has a complex pair."""
        par = Params(mobility=1.0)
        a = governed(par, N, s=s) if s is not None else open_access(par, N)
        R = PAR.R_governed(N, s) if s is not None else PAR.R_open_access(N)
        x = PAR.p * R / (PAR.gamma + N * PAR.p * (s or 0.0))
        q = jnp.asarray(np.concatenate([[R], np.full(N, x)]))
        sharp = jnp.diag(jnp.concatenate([q[:1], jnp.full(N, -par.p)]))
        ev = np.linalg.eigvals(np.asarray(sharp @ jax.jacobian(composite_port_form(a))(q)))
        ev = ev[ev.real < -1e-12]
        return float(np.min(-2.0 * ev.real / np.abs(ev) ** 2))

    bounds = [mu_max(N) for N in NS]
    assert bounds[0] > bounds[1] > bounds[2] > 0
    np.testing.assert_allclose(bounds, [1.5, 7 / 6, 22 / 21], rtol=1e-6)
    # It shrinks, but only from 1.5 to 1.0: under open access the extractors are mutually
    # invisible (§3), so N does not enter the effort block's rate at all, and the whole
    # N-dependence is the (R, X) pair's. Under the sanction gamma' = gamma + Nps does enter,
    # and the bound collapses like 1/N.
    gov = [mu_max(N, S_STAR) for N in NS]
    assert gov[0] > gov[1] > gov[2]
    assert gov[2] < bounds[2] / 10
    np.testing.assert_allclose(gov[2], 2.0 / (PAR.gamma + 20 * PAR.p * S_STAR), rtol=0.05)
    # straddle the bound for N = 5: just below it the settled state attracts, just above it repels.
    N, b = 5, bounds[1]
    tgt = np.concatenate([[PAR.R_open_access(N)], np.full(N, PAR.p * PAR.R_open_access(N))])
    for factor, growing in ((0.9, False), (1.05, True)):
        arr = open_access(Params(mobility=factor * b), N,
                          R0=float(tgt[0]) + 1e-9, x0=float(tgt[1]) + 1e-9)
        traj = np.asarray(run(arr, 60, every=1))
        d = np.linalg.norm(traj - tgt, axis=1)
        assert (d[-1] > d[0]) == growing


# ---------------------------------------------------------------------------
# Predictions 10-15: the finite-supply variant (r = 0) and dissipation.
# ---------------------------------------------------------------------------


FIN = Params(r=0.0, mobility=0.01)

# The regime predictions 11-13 are derived in. Their derivations are *quasi-static*: they
# put ``x_i = pR/gamma`` inside ``Rdot = -RX``, which needs the effort to relax faster than
# the stock, i.e. ``X = N p R_0/gamma << gamma``. The note's own reference start
# (``R_0 = 100, p = 0.01, gamma = 1``) does **not** satisfy that -- there ``X_0 = N``, so the
# stock is the fast variable and the run is nothing like the closed form. See
# ``test_prediction_11_fails_outside_the_quasi_static_regime``. ``R_0 = 5`` with ``eps = 0.2``
# puts ``X_0 = 0.05N``, twenty-fold separated at ``N = 1``.
R_FIN, EPS = 5.0, 0.2


def exhaustion_time(arr, R0=R_FIN, eps=EPS, steps=60_000, every=2):
    """The first sampled step with ``R <= eps R_0``, in time units (§7's ``T_eps``)."""
    traj = np.asarray(run(arr, steps, every=every))
    hit = np.nonzero(traj[:, 0] <= eps * R0)[0]
    return None if len(hit) == 0 else float(hit[0] * every * FIN.mobility)


def test_prediction_10_the_supply_bound_holds_on_the_run():
    """PREDICTION 10. REPRODUCES §7's identity and bound on an actual trajectory:
    ``R_0 - R(T) = int_0^T E dt`` (here the telescoped sum of the steps) and hence
    ``T >= (R_0 - R(T))/max E``. The note says the equality case ``T = R_0/E`` is an
    idealization and is *not* checked against a trajectory of this model; it is not
    checked here either, and the run shows why: ``E`` falls by three orders of magnitude
    over the run, so the bound is loose by that factor."""
    R0, N = 100.0, 5
    traj = np.asarray(run(open_access(FIN, N, R0=R0, x0=0.05), 200_000, every=1))
    R, X = traj[:, 0], traj[:, 1:].sum(axis=1)
    E = R * X
    consumed = R0 - R[-1]
    np.testing.assert_allclose(consumed, FIN.mobility * E[:-1].sum(), rtol=2e-3)
    T = (len(traj) - 1) * FIN.mobility
    assert T >= consumed / E.max()
    assert E.max() / E[-1] > 1e3          # nowhere near constant spend


def test_prediction_11_open_access_lifetime_falls_like_one_over_N():
    """PREDICTION 11, MET **in the quasi-static regime its derivation assumes**. REPRODUCES
    §7's closed form: with ``r = 0`` and effort tracking, ``Rdot ~ -(Np/gamma)R^2``, so
    ``R(t) = R_0/(1 + (Np/gamma)R_0 t)`` and ``T_eps = (gamma/(N p R_0))(1/eps - 1)``,
    proportional to ``1/N``. Measured ``78.4, 38.3, 18.3`` at ``N = 1, 2, 4`` against
    ``80, 40, 20``, and the ratios halve."""
    Ts = [exhaustion_time(open_access(FIN, N, R0=R_FIN, x0=FIN.p * R_FIN)) for N in (1, 2, 4)]
    assert all(T is not None for T in Ts)
    for N, T in zip((1, 2, 4), Ts):
        np.testing.assert_allclose(
            T, PAR.gamma / (N * PAR.p * R_FIN) * (1 / EPS - 1), rtol=0.10)
    np.testing.assert_allclose(Ts[0] / Ts[1], 2.0, rtol=0.05)
    np.testing.assert_allclose(Ts[1] / Ts[2], 2.0, rtol=0.05)


def test_prediction_11_fails_outside_the_quasi_static_regime():
    """PREDICTION 11, **NOT met at the note's own reference start**, and this is the honest
    half. With ``R_0 = 100, p = 0.01, gamma = 1`` the initial total effort is ``X_0 = N``,
    so the *stock* is the fast variable and the effort is frozen through the collapse: the
    run is an exponential ``R_0 e^{-X_0 t}`` crossing over to the hyperbolic tail, and the
    measured lifetimes at ``N = 2, 4, 8`` are ``39.0, 2.25, 0.70``, against the closed form's
    ``49.5, 24.8, 12.4``. The ``1/N`` law is a statement about the slow-stock regime only,
    and §7 does not say so."""
    R0 = 100.0
    Ts = [exhaustion_time(open_access(FIN, N, R0=R0, x0=FIN.p * R0), R0=R0, eps=0.01,
                          steps=400_000, every=5) for N in (2, 4, 8)]
    closed = [PAR.gamma / (N * PAR.p * R0) * 99.0 for N in (2, 4, 8)]
    assert Ts[0] < closed[0] and Ts[1] < 0.2 * closed[1] and Ts[2] < 0.1 * closed[2]
    assert Ts[0] / Ts[1] > 10.0          # nothing like the predicted factor of two


def test_prediction_12_the_governed_lifetime_flattens_in_N():
    """PREDICTION 12, MET; it is §7's real content and §10's F6. REPRODUCES: for any fixed
    ``s > 0`` the total effort is ``X = NpR/(gamma + Nps) -> R/s``, so ``Rdot -> -R^2/s`` and
    ``T_eps -> (s/R_0)(1/eps - 1) = 80``, an ``N``-independent limit. Measured
    ``159, 119, 100, 90, 85`` at ``N = 1, 2, 4, 8, 16``: governed loses a factor of 1.9 over
    a sixteenfold ``N`` where open access loses the full sixteen."""
    s = 100.0
    Ns = (1, 2, 4, 8, 16)
    Ts = [exhaustion_time(governed(FIN, N, s=s, R0=R_FIN,
                                   x0=FIN.p * R_FIN / (PAR.gamma + N * PAR.p * s)),
                          steps=30_000) for N in Ns]
    assert all(T is not None for T in Ts)
    for N, T in zip(Ns, Ts):
        np.testing.assert_allclose(
            T, (PAR.gamma + N * PAR.p * s) / (N * PAR.p * R_FIN) * (1 / EPS - 1), rtol=0.02)
    limit = s / R_FIN * (1 / EPS - 1)
    assert Ts[0] > Ts[1] > Ts[2] > Ts[3] > Ts[4] > limit
    assert Ts[0] / Ts[4] < 2.0
    T_oa = [exhaustion_time(open_access(FIN, N, R0=R_FIN, x0=FIN.p * R_FIN),
                            steps=30_000) for N in (1, 16)]
    assert T_oa[0] / T_oa[1] > 14.0


def test_prediction_13_insulated_decays_exponentially_and_is_the_shortest():
    """PREDICTION 13, MET. REPRODUCES §7: with ``Rhat = R_0`` the total effort is the
    constant ``N p R_0/gamma``, so ``R(t) = R_0 e^{-Xt}`` and
    ``T_eps = (gamma/(N p R_0)) ln(1/eps)``, again ``propto 1/N`` -- and this one needs no
    quasi-static hypothesis, because the effort really is constant. At matched initial
    conditions the ordering is ``T_ins < T_oa < T_gov``: exponential, hyperbolic, and
    hyperbolic with a smaller coefficient (32.2, 78.4, 159.2 at ``N = 1``)."""
    for N in (1, 2, 4):
        T = exhaustion_time(insulated(FIN, N, Rhat=R_FIN, R0=R_FIN, x0=FIN.p * R_FIN))
        np.testing.assert_allclose(
            T, PAR.gamma / (N * PAR.p * R_FIN) * np.log(1 / EPS), rtol=0.02)
    N, s = 1, 100.0
    T_ins = exhaustion_time(insulated(FIN, N, Rhat=R_FIN, R0=R_FIN, x0=FIN.p * R_FIN))
    T_oa = exhaustion_time(open_access(FIN, N, R0=R_FIN, x0=FIN.p * R_FIN))
    T_gov = exhaustion_time(governed(FIN, N, s=s, R0=R_FIN,
                                     x0=FIN.p * R_FIN / (PAR.gamma + N * PAR.p * s)))
    assert T_ins < T_oa < T_gov


def test_prediction_14_finite_supply_dissipation_ends_at_zero():
    """PREDICTION 14. REPRODUCES §7's separation of the two regimes: with ``r = 0`` the only
    settled state is exhaustion, and there ``E = XR -> 0`` for all three wirings and every
    ``N``. The ``1/N`` statements of 11-13 are about the driven regime only."""
    R0 = R_FIN
    for N in (2, 8):
        for arr in (open_access(FIN, N, R0=R0, x0=FIN.p * R0),
                    governed(FIN, N, s=100.0, R0=R0, x0=FIN.p * R0),
                    insulated(FIN, N, Rhat=R0, R0=R0, x0=FIN.p * R0)):
            R, x = split(settle(arr, 600_000))
            assert PAR.harvest(R, float(jnp.sum(x))) < 1e-4


def test_prediction_15_at_the_regenerating_settled_state_dissipation_is_not_zero():
    """PREDICTION 15, and §7's correction to the plan. REPRODUCES two separate statements
    that must not be conflated: (a) the *regenerating* settled state is a **driven** steady
    state, ``E* = r R*(1 - R*/K) > 0``, so dissipation there is *not* zero; (b) what shows
    at it is forgone rent ``Pi_coop - Pi_oa > 0``, tending to ``pKr/4 = 0.25`` as ``N`` grows.
    Reported as residual cost, never as dissipation."""
    for N in NS:
        R, x = settled(open_access(PAR, N))
        E = PAR.harvest(R, float(jnp.sum(x)))
        assert E > 1e-3
        np.testing.assert_allclose(E, PAR.r * R * (1 - R / PAR.K), rtol=1e-3)
    gaps = [PAR.rent_cooperative(N) - PAR.rent(PAR.R_open_access(N), N) for N in NS]
    assert all(g > 0 for g in gaps)
    big = PAR.rent_cooperative(400) - PAR.rent(PAR.R_open_access(400), 400)
    np.testing.assert_allclose(big, PAR.p * PAR.K * PAR.r / 4.0, rtol=0.02)


# ---------------------------------------------------------------------------
# The two claims already checked by hand, put in the record.
# ---------------------------------------------------------------------------


def test_hand_checked_the_open_access_settled_state():
    """REPRODUCES, at eight values of ``N`` and three parameter sets, the value checked by
    hand: ``R* = r gamma K/(r gamma + N p K)``, as the settled state of a genuine composite
    rather than of a hand-written ODE."""
    for par in (PAR, Params(r=0.7, K=40.0, p=0.02, gamma=1.3, mobility=0.004),
                Params(r=2.0, K=10.0, p=0.05, gamma=0.5, mobility=0.002)):
        for N in (1, 2, 3, 5, 8, 13, 21, 34):
            R, _ = split(settle(open_access(par, N, R0=0.9 * par.K, x0=0.01), 120_000))
            np.testing.assert_allclose(
                R, par.r * par.gamma * par.K / (par.r * par.gamma + N * par.p * par.K), rtol=1e-3)


def test_hand_checked_the_sanction_bounds_total_spend_uniformly_in_N():
    """REPRODUCES §7's corrected frustration statement, the one the note says has real
    content (strain S8): at a common stock ``R``, the collective's total effort response is
    ``X_oa(R) = NpR/gamma``, **linear and unbounded in ``N``**, while
    ``X_gov(R) = NpR/(gamma + Nps) <= R/s``, **bounded uniformly in ``N``** and saturating
    at ``R/s``. Read off the settled efforts at a stock held at ``R`` by the insulated
    device of holding the resource's own state fixed -- no: read off the algebraic response,
    then confirmed on runs by the lifetimes of prediction 12."""
    R, s = 40.0, 20.0
    for N in (1, 4, 16, 64, 256):
        X_oa = N * PAR.p * R / PAR.gamma
        X_gov = N * PAR.p * R / (PAR.gamma + N * PAR.p * s)
        np.testing.assert_allclose(X_oa, N * 0.4, rtol=1e-12)     # linear in N
        assert X_gov < R / s                                      # 2.0, for every N
    np.testing.assert_allclose(256 * PAR.p * R / (PAR.gamma + 256 * PAR.p * s), R / s, rtol=0.03)
    # and on the runs: governed settled X saturates in N, open-access settled X does not
    # exceed r (the note's own correction) but its *response* at a fixed stock is unbounded.
    Xg = [float(jnp.sum(settled(governed(PAR, N, s=s))[1])) for N in (1, 5, 20)]
    assert Xg[0] < Xg[1] < Xg[2] < PAR.r
    for N in (1, 5, 20):
        R_g, x_g = settled(governed(PAR, N, s=s))
        np.testing.assert_allclose(float(jnp.sum(x_g)),
                                   N * PAR.p * R_g / (PAR.gamma + N * PAR.p * s), rtol=2e-4)
