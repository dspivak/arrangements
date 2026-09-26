"""Pulling: can one part reach far and stay selective, or does that force a monitor?

Stage B of David's monitor test (2026-09-25, his words): "there can't be a monitor who is
able to see dead parts and also able to pull from far? ... you're saying that you can't
make a doodad that is able to apply force to things, like walk over to something and pull
it over to another location? i also don't see why it has to be an exact form." Stage A is
seeing; this file is pulling.

**The prediction, fixed in advance and written here before any construction code
existed. It must be able to fail.**

    Route (a) extends the pull region with ``tau`` but flattens selectivity at the same
    level, so one level cannot have both and the nesting is forced; route (b) recruits at
    any distance and shows a nonzero antisymmetric part in the recruiting coupler.
    **Able to fail:** if (a) reaches far *and* stays selective, no monitor is needed.

The two routes, as the plan states them:

**(a) Nesting, and it stays exact.** The monitor is a parent one level up with its own
larger ``tau``; reach comes from being at a coarser level. Inside the attention operad, no
stipulation. It predicts a two-level structure -- small ``tau`` inside for selectivity,
large ``tau`` outside for reach.

**(b) Discretion, the doodad, and it is ordinary.** A monitor whose own state sets a
forcing on the spare's placement with nothing returning to it. Non-exact forcing is why
the book has one-forms at all -- a pump, a waterfall, a controller are all non-exact -- so
a part that walks over and pulls something is an ordinary inhabitant of the category of
arrangements, not a difficulty. Its single price is a supply: by the physical reading a
non-exact coupler needs one. Route (b) is inside the category of arrangements and outside
the attention operad.

Settled before this run and not rediscovered here: the spatial kernel's scale ``tau`` is a
single scalar per arrangement (``spatial_kappa``, ``arrangements/attention.py``), one
``tau`` for every pair, so a monitor cannot simply be given longer reach beside its peers
at one level. That is what forces the two routes. And ``notes/TURNOVER-TEST.md`` settled
that **reach is a rate, not a radius**: the inward force on a spare is strictly nonzero at
every finite distance, and what looks like a pull region is a run-length budget. So every
reach claim below carries a rate and a step budget.

The parent construction is ``tests/test_turnover.py`` and ``notes/TURNOVER-TEST.md``: a
spatial-attention parent over redundant contributors, its cost on the weighted signal it
sends onward, one contributor dead, a spare off to the side. What is new here is that
``tau`` varies, that the parent is nested inside a second parent, and that a monitor part
pulls the spare through a coupler instead of through attention.

Every docstring below says which of three things its test does (STATUS.md §3.7):
reproduce a closed form derived in the docstring, check a discrete analogue, or correct a
claim made on paper. The numbers, and which verdicts the parameters decide rather than
discover: ``notes/MONITOR-PULLING.md``.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact, zero_form
from arrangements.attention import (
    input_dim, make_input, message, msg_dim, signal, spatial_arrangement, spatial_config,
    spatial_kappa, spatial_split, weight,
)
from arrangements.coalgebra import configuration
from arrangements.rvect import constant, euclidean
from arrangements.wiring import compose_seq, tensor_arrangements, trivial_parameter

from tests.chain import flat_state
from tests.test_rendezvous import run_closed, run_phi

DH, DP = 1, 1                                   # H = W = P = R
MDIM, EDIM = msg_dim(DH), input_dim(DH, DP)     # dim M = 2, dim E = 3
EXT = make_input(message(0.0, 0.0), 0.0)        # the exterior input (0_M, 0)
EMPTY = jnp.zeros(0)

SIG, WGT = 1.0, 1.0   # what a contributor in full health signals, and the weight it claims
NU = 2.5              # the parent's target for the weighted signal it sends onward
MU_P = 0.5            # the mobility of a placement coordinate
BUDGET = 8000         # the step budget every reach and selectivity number below is read at
TAUS = (0.2, 0.4, 0.8, 1.6, 3.2, 6.4)      # the sweep: five doublings of the kernel scale

D0, GAP = 1.0, 0.5    # the selectivity reference distance, and the fixed gap across it

TAU_IN, TAU_OUT = 0.2, 3.2        # route (a)'s two levels
D_FAR = 6.0                       # where the outer level's spare sits
NU_IN, NU_OUT = 1.0, 1.5          # the two levels' targets

ALPHA, WID = 0.02, 0.25           # the doodad's pull: strength, and the width of its ramp
A0, MU_A, LAM_A = 1.0, 0.01, 1.0  # the monitor's alarm, its mobility, its own stiffness


# ---------------------------------------------------------------------------
# One level: contributors, a spatial parent, and what the collective sends onward.
# ---------------------------------------------------------------------------


def contributor(v, label="C"):
    """A component ``I -> B`` (the idiom of ``tests/test_member.py`` and of the turnover
    test's contributors) whose configuration is the value ``v`` it signals, frozen at
    mobility ``0``, sending ``(v SIG, WGT)``. ``v = 1`` is a contributor in full health,
    ``v = 0`` one that is dead but still claims its weight, ``v = -1`` one whose
    contribution is wrong rather than absent -- the turnover test's *corrupting* mode,
    which is the only failure a cost on the weighted signal can see. Its own forcing is
    zero: there is no health dynamics in this stage, so every forcing below is the
    parent's attention, or the doodad's pull."""
    return Arrangement(
        P=configuration(euclidean(1, 0.0), init=jnp.array([float(v)])),
        in_dim_M=0, out_dim_M=0, in_dim_N=EDIM, out_dim_N=MDIM,
        out_f=lambda q, m_out: message(q[0] * SIG, WGT),
        in_f=lambda q, m_out, n_in: EMPTY,
        omega=zero_form,
        label=label,
    )


def outgoing(n, q_parent, m_out, tau):
    """``sigma_a(q, m) = sum_j T_{0j}(q) m_j`` (eq:maps), written out, as the turnover
    test writes it, but with ``tau`` a parameter:

        m~ = (sum_j kappa_j V_j h_j,  sum_j kappa_j w_j),  kappa_j = exp(-||p_j||^2/2 tau),

    since ``l_0 = 0_P`` and ``V_0 = H``. Checked against ``a.sigma`` in test 1."""
    pos, V = spatial_split(q_parent, n, DH, DP)
    h, w = jnp.zeros(DH), jnp.asarray(0.0)
    for j in range(n):
        k = spatial_kappa(pos[j], jnp.zeros(DP), tau)
        m_j = m_out[j * MDIM:(j + 1) * MDIM]
        h = h + k * (V[j] @ signal(m_j))
        w = w + k * weight(m_j)
    return message(h, w)


def parent(n, tau, nu, positions, label="parent"):
    """A spatial-attention parent whose cost is ``(1/2)|nu - h~|^2`` on the weighted
    signal it sends onward -- the turnover test's required cost, which keeps ``k(q, -)``
    of degree two in ``D_n``. Its value maps are frozen at the identity (mobility ``0``),
    so its only adaptation is attentional: where it holds each child. ``tau`` is one
    scalar for every pair (eq:spatial-transfers); this function is the only place a
    ``tau`` is chosen, and route (a) is the question of what choosing it costs."""
    cost = lambda q, m_out, n_in: 0.5 * jnp.sum(
        (nu - signal(outgoing(n, q, m_out, tau))) ** 2)
    mobilities = jnp.array([MU_P] * n + [0.0] * (n * DH * DH))
    return spatial_arrangement(
        n, DH, DP, tau, cost=cost,
        parameter=configuration(constant(jnp.diag(mobilities)),
                                init=spatial_config(positions, [1.0] * n)),
        label=label)


def collective(values, positions, tau, nu, children=None):
    """``compose_seq(tensor(children), parent.arr())``: the composite configuration is
    ``(v_1..v_n, p_1..p_n, V_1..V_n)``, the children's blocks first. A child may be any
    arrangement with a contributor's interface ``<E|M>`` -- in test 3 one of them is
    itself a collective."""
    n = len(positions)
    kids = children or [contributor(v, f"C{i}") for i, v in enumerate(values)]
    a = parent(n, tau, nu, positions)
    return kids, a, compose_seq(tensor_arrangements(kids), a.arr())


def kappa_at(d, tau):
    return float(np.exp(-np.asarray(d, float) ** 2 / (2.0 * tau)))


def equilibrium_position(n_live, extra, tau, nu=NU):
    """The position at which ``n_live`` contributors in full health, plus a further
    contribution ``extra``, send exactly ``nu``: the turnover test's device for starting
    at the pre-failure fixed point."""
    return float(np.sqrt(-2.0 * tau * np.log((nu - extra) / n_live)))


def forcing(b, q, ext=EXT):
    """The composite forcing's configuration component at ``q``."""
    xi_q, _, _ = b.omega(jnp.asarray(q, float), EMPTY, ext)
    return np.asarray(xi_q)


def traj(b, steps, ext=EXT):
    """The whole trajectory of a closed composite, as ``run_closed`` computes it but
    inside a ``lax.scan`` (test 0 checks the two agree). Returns ``steps`` rows."""
    P, omega = b.P, b.omega

    @jax.jit
    def step(s, _):
        xi_q, _, _ = omega(P.position(s), EMPTY, ext)
        s2 = P.step(s, xi_q)
        return s2, flat_state(s2)

    return np.asarray(jax.lax.scan(step, P.init, None, length=steps)[1])


def h_sent(rows, n, values, tau, offset=0):
    """``h~`` along a trajectory, in numpy: ``sum_j kappa(p_j) v_j``, which is the signal
    of ``outgoing`` when every ``V_j`` is the identity and child ``j`` signals ``v_j``.
    Test 1 licenses this against ``a.sigma``; it is used instead of the library call so
    that a whole run's worth of positions can be read at once."""
    pos = rows[:, offset + n:offset + 2 * n]
    return (np.exp(-pos ** 2 / (2.0 * tau)) * np.asarray(values)[None, :]).sum(axis=1)


# ---------------------------------------------------------------------------
# Route (a): reach and selectivity at one level.
# ---------------------------------------------------------------------------


def failed_collective(d, tau, nu=NU):
    """The turnover test's failed state with ``tau`` a parameter: two contributors in full
    health at ``+-p``, a dead one at ``+p``, a spare in full health at ``d``, with ``p``
    chosen so that before the failure the collective sent exactly ``nu``. The deficit at
    this state is ``kappa(p) = (nu - kappa(d))/3``, the dead contributor's own share."""
    values = [1.0, 1.0, 0.0, 1.0]
    p = equilibrium_position(3, kappa_at(d, tau), tau, nu)
    kids, a, b = collective(values, [p, -p, p, d], tau, nu)
    return kids, a, b, values


def arrival(d, tau, steps=BUDGET, nu=NU, tol=1e-6):
    """The step at which the collective's outgoing signal first reaches ``nu``, with a
    spare at ``d``, or ``None`` within the budget. This is the turnover test's
    "recruited", timed: reach is a rate, so a reach claim is a pair (distance, steps)."""
    _, _, b, values = failed_collective(d, tau, nu)
    h = h_sent(traj(b, steps), 4, values, tau)
    hit = np.nonzero(h > nu - tol)[0]
    return (int(hit[0]) + 1) if len(hit) else None


def reach(tau, steps=BUDGET, lo=0.5, hi=40.0, tol=0.05):
    """The largest ``d`` from which the spare is recruited within ``steps``, by bisection
    (test 2 checks on a grid that there is one threshold in ``d``, not a patchwork)."""
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if arrival(mid, tau, steps) else (lo, mid)
    return lo


def discrimination(tau, steps=BUDGET, d0=D0):
    """The dynamic selectivity of one level at ``tau``, at the same budget as ``reach``.

    A parent over two contributors at equal distance ``d0`` on either side, one signalling
    ``+SIG`` and one ``-SIG``, with target ``nu = 1``: the collective must tell a good
    contributor from a bad one when its attention starts equal. Descent pulls the good one
    in and pushes the bad one out; the number reported is ``kappa_good / kappa_bad`` after
    ``steps``, and the drift is how far the positions move in a further 500."""
    _, _, b = collective([1.0, -1.0], [d0, -d0], tau, NU_IN)
    rows = traj(b, steps + 500)
    q, q2 = rows[steps - 1], rows[-1]
    k_g, k_b = kappa_at(q[2], tau), kappa_at(q[3], tau)
    return k_g, k_b, float(np.max(np.abs(q2 - q)))


# ---------------------------------------------------------------------------
# Route (a): the two-level arrangement.
# ---------------------------------------------------------------------------


def nested():
    """A parent whose child is a parent. The inner collective is ``discrimination``'s good
    and bad pair at ``tau_in``; the outer parent, at ``tau_out``, holds that collective and
    a spare contributor at ``D_FAR``. A collective's interface is a contributor's -- both
    are ``<E|M>`` -- so the nesting needs no new construction, which is route (a)'s claim
    that it is inside the operad and stipulates nothing.

    The composite configuration is ``(v_good, v_bad, p_good, p_bad, V_in; v_spare;
    p_inner, p_spare, V_out)``: the inner placements are at indices 2, 3, read with
    ``tau_in``, and the outer at 7, 8, read with ``tau_out``."""
    kids_in, a_in, b_in = collective([1.0, -1.0], [D0, -D0], TAU_IN, NU_IN)
    spare = contributor(1.0, "spare")
    a_out = parent(2, TAU_OUT, NU_OUT, [D0, D_FAR], label="outer")
    b_out = compose_seq(tensor_arrangements([b_in, spare]), a_out.arr())
    return (kids_in, a_in, b_in), a_out, b_out


# ---------------------------------------------------------------------------
# Route (b): the doodad.
# ---------------------------------------------------------------------------


def monitor(a0=A0, mu=MU_A, lam=LAM_A):
    """The doodad: a part whose configuration is its own alarm ``a``, published at its
    port, carrying its own exact cost ``(lam/2)(a - a0)^2``. Nothing else is stipulated
    about it -- in particular it does not read the collective, which is stage A's
    question."""
    return Arrangement(
        P=configuration(euclidean(1, mu), init=jnp.array([float(a0)])),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: EMPTY,
        omega=exact(lambda q, m_out, n_in: -0.5 * lam * (q[0] - a0) ** 2),
        label="monitor")


def publishing(b, idx):
    """The same arrangement, additionally publishing configuration coordinate ``idx`` at
    its port. Only ``out_f`` changes: the forcing is untouched, so every non-exactness
    measured below is the coupler's and not an artefact of the wrapper. Publishing a
    placement is what lets something other than the parent put a force on it, and it is
    the step that leaves the attention operad."""
    return replace(b, out_dim_N=b.out_dim_N + 1,
                   out_f=lambda q, l_out: jnp.concatenate([b.out_f(q, l_out),
                                                           q[idx:idx + 1]]))


def pull_F(alpha=ALPHA, wid=WID, billed=False):
    """The returned covector of the recruiting coupler on the ports ``e = (a, h~, w~, p)``
    (``tests/test_two_tanks.py``'s idiom, as ``tests/test_mutual_need.py`` uses it):

        F_p(e) = -alpha a tanh(p / wid),    F_a(e) = 0,    F_h~ = F_w~ = 0.

    A bounded pull, of magnitude ``alpha a`` once ``|p| >> wid``: the doodad walks over
    and pulls, at a force that does not grow with distance. Nothing returns to the
    monitor, which is the brief's stipulation, and by the reciprocity lemma (§5.5) that is
    exactly what makes the coupler non-exact.

    ``billed=True`` adds the one return that repairs it,
    ``F_a(e) = -alpha Psi(p)``, ``Psi(p) = wid log cosh(p / wid)``, ``Psi' = tanh(p/wid)``:
    then ``F = dU`` for ``U(a, p) = -alpha a Psi(p)``, so the pull is exact. ``alpha a
    Psi(p)`` is, up to ``alpha a wid log 2``, the work ``alpha a |p|`` of bringing the
    spare in from ``p`` at alarm ``a``: the exact completion is the monitor being charged
    the outstanding work. Test 5 measures what that costs."""

    def F(e):
        a, p = e[0], e[3]
        F_p = -alpha * a * jnp.tanh(p / wid)
        F_a = (-alpha * wid * jnp.log(jnp.cosh(p / wid))) if billed else 0.0 * a
        return jnp.array([F_a, 0.0 * e[1], 0.0 * e[2], F_p])

    return F


def coupler(F, ports):
    """A stateless coupler whose forcing on its ``ports`` is the returned covector ``F``
    and whose ``in_f`` closes the collective on the exterior input ``EXT``."""
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=EDIM, out_dim_M=ports, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, e: EMPTY,
        in_f=lambda q, e, n_in: EXT,
        omega=lambda q, e, n_in: (EMPTY, F(e), EMPTY),
        label="pull")


SPARE_P = 7      # the spare's placement in the four-child collective's configuration
PULLED_P = 8     # ... and in the monitor-and-collective composite's


def pulled(d, tau=0.8, **kw):
    """The doodad wired to the collective: ``compose_seq(tensor(monitor, collective
    publishing the spare's placement), coupler)``, a closed arrangement whose
    configuration is ``(a; v_1..v_4, p_1..p_4, V_1..V_4)``, so the alarm is coordinate
    ``0`` and the spare's placement is coordinate ``8``."""
    kids, a, b, values = failed_collective(d, tau)
    tensored = tensor_arrangements([monitor(), publishing(b, SPARE_P)])
    F = pull_F(**kw)
    return kids, a, compose_seq(tensored, coupler(F, 1 + MDIM + 1)), values, F


def pulled_arrival(d, steps, tau=0.8, tol=1e-6, **kw):
    """``(arrival step or None, trajectory)`` for the pulled composite."""
    _, _, b, values, _ = pulled(d, tau, **kw)
    rows = traj(b, steps, ext=EMPTY)
    h = h_sent(rows, 4, values, tau, offset=1)
    hit = np.nonzero(h > NU - tol)[0]
    return ((int(hit[0]) + 1) if len(hit) else None), rows


def d_omega(F):
    """``(d omega)_{ij} = d_i F_j - d_j F_i``, the antisymmetric part of the
    returned-covector Jacobian (the reciprocity lemma, §5.5), exactly as
    ``tests/test_mutual_need.py`` computes it."""
    def at(e):
        J = jax.jacobian(F)(jnp.asarray(e, float))   # J[j, i] = d F_j / d e_i
        return np.asarray(J.T - J)
    return at


# ---------------------------------------------------------------------------
# Test 0: the runs are Phi's.
# ---------------------------------------------------------------------------


def test_0_the_runs_are_the_dynamics_functors():
    """LICENSES the run forms used below; it establishes nothing about the prediction.
    For all three composites -- one level, two levels, and the monitor wired to the
    collective -- ``Phi``, ``run_closed`` and the ``lax.scan`` of ``traj`` agree over
    fifty steps to within one unit in the last place, so every trajectory here is
    ``Phi``'s."""
    _, _, one, _ = failed_collective(2.0, 0.8)
    _, _, two = nested()
    _, _, three, _, _ = pulled(5.0)
    for b, ext in ((one, EXT), (two, EXT), (three, EMPTY)):
        by_phi = run_phi(b, 50, ext)
        by_closed = run_closed(b, 50, lambda t: ext)
        assert float(np.max(np.abs(by_phi - by_closed))) < 1e-15
        np.testing.assert_array_equal(traj(b, 50, ext)[-1], by_closed[-1])


# ---------------------------------------------------------------------------
# Test 1: what the collective sends, and that a reach number is a rate.
# ---------------------------------------------------------------------------


def test_1_the_outgoing_signal_is_sigma_and_a_reach_number_is_a_pair():
    """REPRODUCES the closed form for ``sigma_a`` at two ``tau`` and LICENSES ``h_sent``.

    ``sigma_a(q, m) = (sum_j kappa_j V_j h_j, sum_j kappa_j w_j)`` with ``kappa_j =
    exp(-p_j^2/2 tau)``, which with ``V_j = H`` and child ``j`` signalling ``v_j`` makes
    the outgoing signal ``sum_j kappa_j v_j`` -- the numpy form every table below reads
    off a trajectory. At the failed state the collective sends ``nu - kappa(p)``, short by
    exactly the dead contributor's share, and after recruitment it sends ``nu``.

    And the point ``notes/TURNOVER-TEST.md`` established, restated in the form this file
    needs: at one distance the verdict depends on the budget, so "recruited from ``d``" is
    never said without a step count. At ``tau = 0.8`` the spare at ``d = 3`` arrives at
    step 158; at ``tau = 0.4`` the same distance takes 6108 steps; at ``tau = 0.2`` it
    does not arrive within 8000, nor within 200 000, and yet the force on it is inward at
    the first step and every step after: ``-(nu - h~)(d/tau) kappa(d) = -2.11487e-9`` at
    ``d = 3``, ``tau = 0.2``, which at mobility ``MU_P`` is about ``1e-9`` of distance a
    step and so some ``2e9`` steps for the trip. The reach numbers of test 2 are
    therefore budgets throughout, never radii."""
    for tau in (0.2, 3.2):
        kids, a, b, values = failed_collective(2.0, tau)
        q0 = jnp.asarray(flat_state(b.P.init))
        m = jnp.concatenate([k.out_f(q0[i:i + 1], EMPTY) for i, k in enumerate(kids)])
        np.testing.assert_allclose(outgoing(4, q0[4:], m, tau), a.sigma(q0[4:], m), atol=0.0)
        rows = traj(b, 2)
        np.testing.assert_allclose(
            h_sent(np.asarray(q0)[None, :], 4, values, tau)[0],
            float(signal(a.sigma(q0[4:], m))[0]), rtol=1e-14)
        # short by the dead contributor's own share, at the start
        p = equilibrium_position(3, kappa_at(2.0, tau), tau)
        np.testing.assert_allclose(h_sent(np.asarray(q0)[None, :], 4, values, tau)[0],
                                   NU - kappa_at(p, tau), rtol=1e-12)
        assert rows.shape == (2, 12)

    assert arrival(3.0, 0.8) == 158
    assert arrival(3.0, 0.4) == 6108
    assert arrival(3.0, 0.2) is None
    assert arrival(3.0, 0.2, steps=200000) is None

    # ... and yet the force on it is inward: a budget, not a boundary
    _, _, b3, _ = failed_collective(3.0, 0.2)
    xi = forcing(b3, flat_state(b3.P.init))[4:8]
    p3 = equilibrium_position(3, kappa_at(3.0, 0.2), 0.2)
    np.testing.assert_allclose(
        xi[3], -kappa_at(p3, 0.2) * (3.0 / 0.2) * kappa_at(3.0, 0.2), rtol=1e-12)
    np.testing.assert_allclose(xi[3], -2.11487e-9, rtol=1e-5)
    assert xi[3] < 0.0


# ---------------------------------------------------------------------------
# Test 2: route (a) at one level -- the sweep.
# ---------------------------------------------------------------------------


REACH = (2.197, 3.007, 4.126, 5.630, 7.636, 10.298)     # d reached within 8000 steps
KAPPA_BAD = (0.00143053, 0.00210778, 0.00309412, 0.00455220, 0.00672388, 0.00998415)


def test_2_route_a_buys_reach_with_selectivity():
    """REPRODUCES the sweep and CHECKS the prediction's first clause for route (a).

    Reach and selectivity are measured at the same budget, 8000 steps, in the same fixed
    spatial units -- a collective's ``tau`` may change but the world's distances do not.

    *Reach.* The largest ``d`` from which the spare is recruited within the budget, by
    bisection. It rises monotonically with ``tau``, from ``2.20`` at ``tau = 0.2`` to
    ``10.30`` at ``tau = 6.4``, and it rises as the kernel's own scale does:
    ``reach / sqrt(tau)`` stays in ``[4.07, 4.91]`` across five doublings.

    *Selectivity, statically.* The ratio of the parent's attention across a fixed gap,
    ``kappa(d0)/kappa(d0 + gap) = exp(gap(2 d0 + gap) / 2 tau)``, at ``d0 = 1``,
    ``gap = 0.5``: ``exp(0.625/tau)``, which falls from ``22.76`` to ``1.10``. A parent
    with a large ``tau`` cannot hold two things half a unit apart at appreciably different
    attention, whatever it does.

    *Selectivity, dynamically.* A good and a bad contributor start at equal distance on
    either side; after the same 8000 steps the good one is at ``kappa = 1`` to fourteen
    digits and the bad one at ``kappa_bad``, which *rises* with ``tau`` -- ``0.00143`` at
    ``tau = 0.2``, ``0.00998`` at ``tau = 6.4``, nearly as ``sqrt(tau)``. The ratio
    ``1/kappa_bad`` falls from ``699`` to ``100``.

    So reach and residual attention to a bad part grow together, both about as
    ``sqrt(tau)``: **on ``0.2 <= tau <= 6.4`` the two cannot be improved together, and the
    prediction's first clause holds.** The measurement is a strict trade-off, not a
    threshold: no number had to be chosen to say which levels count as far or selective.
    """
    rows = []
    for tau in TAUS:
        k_g, k_b, drift = discrimination(tau)
        rows.append((tau, reach(tau), k_g, k_b, drift))

    reaches = np.array([r[1] for r in rows])
    kb = np.array([r[3] for r in rows])
    np.testing.assert_allclose(reaches, REACH, atol=0.06)
    np.testing.assert_allclose(kb, KAPPA_BAD, rtol=1e-4)

    assert np.all(np.diff(reaches) > 0.0)               # reach rises with tau ...
    assert np.all(np.diff(kb) > 0.0)                    # ... and so does the residual
    assert np.all([r[2] > 1.0 - 1e-14 for r in rows])   # the good one saturates at kappa=1
    assert np.all([r[4] > 1e-6 for r in rows])          # still moving: this is a rate

    scaled = reaches / np.sqrt(np.array(TAUS))
    assert 4.0 < scaled.min() and scaled.max() < 5.0    # reach grows as sqrt(tau)

    static = np.exp(GAP * (2.0 * D0 + GAP) / (2.0 * np.array(TAUS)))
    np.testing.assert_allclose(static, [22.7599, 4.77073, 2.18420, 1.47790, 1.21569,
                                        1.10258], rtol=1e-5)
    np.testing.assert_allclose(
        static[2], kappa_at(D0, 0.8) / kappa_at(D0 + GAP, 0.8), rtol=1e-12)
    assert np.all(np.diff(static) < 0.0)

    # one threshold in d, not a patchwork: the bisection above is well posed
    near = [arrival(d, 0.8) is not None for d in (1.0, 2.0, 3.0, 4.0, 4.5, 5.0, 6.0)]
    assert near == [True, True, True, True, False, False, False]


# ---------------------------------------------------------------------------
# Test 3: route (a) at two levels.
# ---------------------------------------------------------------------------


def test_3_route_a_nesting_has_both_and_needs_no_stipulation():
    """REPRODUCES the two-level run, and so CHECKS the prediction's "the nesting is
    forced" against the one thing that could refute it -- that nesting might not deliver
    both either.

    The outer parent, ``tau_out = 3.2``, holds the inner collective and a spare at
    ``d = 6``; the inner parent, ``tau_in = 0.2``, holds a good and a bad contributor at
    ``+-1``. In one run of 8000 steps: the outer's outgoing signal reaches its target
    ``1.5`` at step 643, the spare having come in from ``6`` to ``2.106``
    (``kappa = 0.5001``); and inside, the bad contributor is at ``kappa = 1.181e-4``
    against the good one's ``1``, a ratio of ``8467``.

    Both at once, and each better than one level could do at either ``tau``: a single
    level at ``tau_in = 0.2`` does not recruit from ``6`` within the budget at all (its
    reach is ``2.20``), and a single level at ``tau_out = 3.2`` discriminates at ``149``,
    not ``8467``. The inner ratio exceeds the ``699`` of a lone ``tau = 0.2`` level
    because the outer's own deficit drives the inner pair apart as well, so the two levels
    do not merely coexist here -- the coarse one helps the fine one.

    Nothing was stipulated to get it: a collective's interface is a contributor's, so
    ``compose_seq`` takes the nesting as it stands, each level with its own single scalar
    ``tau``.
    """
    (kids_in, a_in, b_in), a_out, b_out = nested()
    rows = traj(b_out, BUDGET)
    out = np.asarray(jax.jit(jax.vmap(lambda q: b_out.out_f(q, EMPTY)))(jnp.asarray(rows)))

    hit = np.nonzero(out[:, 0] > NU_OUT - 1e-6)[0]
    assert len(hit) and int(hit[0]) + 1 == 643
    q = rows[-1]

    np.testing.assert_allclose(out[-1, 0], NU_OUT, atol=1e-9)
    np.testing.assert_allclose(q[8], 2.105856, atol=1e-5)            # the spare came in
    np.testing.assert_allclose(kappa_at(q[8], TAU_OUT), 0.500118, atol=1e-6)

    k_good, k_bad = kappa_at(q[2], TAU_IN), kappa_at(q[3], TAU_IN)
    np.testing.assert_allclose([k_good, k_bad], [1.0, 1.1811e-4], rtol=1e-4)
    assert k_good / k_bad > 8000.0

    # neither level alone, at the same budget
    assert arrival(D_FAR, TAU_IN) is None                   # tau_in cannot reach 6
    assert 1.0 / discrimination(TAU_OUT)[1] < 150.0         # tau_out cannot discriminate
    assert k_good / k_bad > 1.0 / discrimination(TAU_IN)[1]  # and the outer helps the inner


# ---------------------------------------------------------------------------
# Test 4: route (b) -- the doodad recruits at any distance.
# ---------------------------------------------------------------------------


DISTANCES_B = (5.0, 10.0, 20.0, 50.0, 100.0)


def test_4_route_b_recruits_at_any_distance_at_a_linear_rate():
    """REPRODUCES the closed form for the arrival law, and so CHECKS the prediction's
    second clause.

    The monitor's alarm sits at ``a = 1`` and the coupler returns ``F_p = -alpha a
    tanh(p/wid)``, a bounded inward force of magnitude ``alpha = 0.02`` once
    ``|p| >> wid``. The placement's mobility is ``MU_P``, so one step moves the spare
    ``MU_P alpha = 0.01`` and the trip is **linear in the distance**, at

        1 / (MU_P alpha a) = 100 steps per unit of distance,

    against a constant offset for the endgame: the measured arrival is exactly
    ``100 d - 245`` at ``d = 5, 10, 20, 50, 100`` -- 255, 755, 1755, 4755 and 9755 steps.
    Compare route (a) at the same ``tau = 0.8``, where ``notes/TURNOVER-TEST.md`` found
    that a spare at ``d = 5`` needs two million steps and one at ``d = 6`` never arrives
    within that: the doodad recruits from ``d = 100`` in ten thousand.

    The alarm does not move: the monitor delivers ``1.9796`` units of work over the trip
    from ``d = 100`` (``alpha a`` times the distance travelled, to ``1e-4``) and its own
    state ends where it began, bit for bit. **So the supply the physical reading requires
    is stipulated away here, and this test says so rather than hiding it**: the doodad is
    an inexhaustible source. Test 5 puts the bill back and measures what it costs.

    Where the pulled collective settles is not where the parent alone would have put it.
    The two live contributors come home to ``p = 0`` and the spare stops at ``1.007969``,
    where the parent's outward push balances the pull:

        (h~ - nu)(p/tau) kappa(p) = alpha a tanh(p/wid),   h~ = 2 + kappa(p),

    so the collective settles *above* its target, at ``h~ = 2.529935`` rather than
    ``2.5``: a doodad that keeps pulling holds the collective past what it asked for. The
    dead contributor never moves at all -- it sits at its initial ``0.540106`` for fifteen
    thousand steps, since a contribution that is merely absent exerts no pull either way
    (``notes/TURNOVER-TEST.md``, test 3), and the doodad pulls the spare, not it.
    """
    steps = 15000
    arrivals, works = [], []
    for d in DISTANCES_B:
        step, rows = pulled_arrival(d, steps)
        p, a = rows[:, PULLED_P], rows[:, 0]
        F_p = -ALPHA * a[:-1] * np.tanh(p[:-1] / WID)
        arrivals.append(step)
        works.append(float(np.sum(F_p * np.diff(p))))
        assert float(a.min()) == A0 and float(a.max()) == A0   # nothing returns to it

    assert arrivals == [int(100.0 * d - 245.0) for d in DISTANCES_B]
    np.testing.assert_allclose(np.diff(arrivals) / np.diff(DISTANCES_B),
                               1.0 / (MU_P * ALPHA * A0), atol=0.0)

    _, _, b_last, _, _ = pulled(100.0)
    rows = traj(b_last, steps, ext=EMPTY)
    np.testing.assert_allclose(works[-1], 1.979639, atol=1e-5)
    np.testing.assert_allclose(works[-1], ALPHA * A0 * (100.0 - rows[-1, PULLED_P]),
                               rtol=1e-3)

    # and where it settles, which is not where the parent alone would have put it
    values = [1.0, 1.0, 0.0, 1.0]
    np.testing.assert_allclose(rows[-1, 5:9], [0.0, 0.0, 0.540106, 1.007969], atol=1e-6)
    np.testing.assert_array_equal(rows[-1, 5:9], rows[-2, 5:9])          # settled exactly
    h_end = h_sent(rows[-1:], 4, values, 0.8, offset=1)[0]
    np.testing.assert_allclose(h_end, 2.529935, atol=1e-6)
    p_star = float(rows[-1, PULLED_P])
    np.testing.assert_allclose(
        (h_end - NU) * (p_star / 0.8) * kappa_at(p_star, 0.8),
        ALPHA * A0 * np.tanh(p_star / WID), rtol=1e-4)
    assert abs(float(forcing(b_last, rows[-1], ext=EMPTY)[PULLED_P])) < 1e-5


# ---------------------------------------------------------------------------
# Test 5: route (b) -- the coupler is not exact, and what making it exact costs.
# ---------------------------------------------------------------------------


def test_5_route_b_the_coupler_is_not_exact_and_the_bill_costs_reach():
    """REPRODUCES the closed form for ``d omega`` and CORRECTS, in passing, a guess of my
    own about what exactness would cost.

    ``F = (F_a, 0, 0, F_p)`` on ``e = (a, h~, w~, p)`` with ``F_p = -alpha a tanh(p/wid)``
    and ``F_a = 0``, so

        (d omega)_{a p} = d_a F_p - d_p F_a = -alpha tanh(p / wid),

    the only nonzero entry: ``-0.00759898`` at ``p = 0.1``, ``-0.0184334`` at ``p = 0.4``,
    ``-0.02`` to sixteen digits once ``p`` saturates the ramp. The *composite's* forcing
    on ``Q`` carries the same entry, at the alarm and the spare's placement: the
    collective's own forcing is the differential of a cost and contributes nothing
    antisymmetric. By the reciprocity lemma (§5.5) the recruiting coupler is therefore not
    ``dU`` for any ``U`` -- inside the book's category, whose forcing need not be exact
    (STATUS.md §1), outside the attention operad, whose forcing is the full differential
    of a nonnegative cost. **The prediction's second clause holds, at ``-0.02``.**

    The exact completion is unique up to a function of ``a`` alone: ``F_p`` fixes
    ``U = -alpha a Psi(p)``, ``Psi = wid log cosh(p/wid)``, so ``F_a`` must carry
    ``-alpha Psi(p)`` -- the spare's placement acting back on the monitor, which is
    precisely what "nothing returns to it" forbids. With it, ``d omega`` is zero to
    ``1e-16`` and the composite's antisymmetric part with it.

    What I expected that to cost was nothing but bookkeeping. It costs the reach. The
    alarm settles at ``a* = a0 - alpha Psi(p)/lam``, which crosses zero at
    ``p = lam a0/alpha + wid log 2 = 50.173``: beyond that the pull reverses, the spare
    runs out, the reversal deepens, and the run diverges. Measured at a budget of 40 000
    steps, the billed monitor still recruits from ``d = 40`` and does not from ``d = 60``;
    from ``d = 5`` it takes 266 steps against the unbilled 255. So **the exact pull has a
    genuine boundary and the non-exact one does not** -- which is the sharpest thing this
    stage found, and it was not predicted. The boundary is a property of *this* exact
    completion: its restoring cost on ``a`` is quadratic and lets the alarm go negative. A
    completion that confined the alarm to positive values would not reverse, though its
    pull would weaken with distance; that is not tested here.
    """
    _, _, b, _, F = pulled(5.0)
    dw = d_omega(F)
    for p in (0.1, 0.4, 1.0, 5.0):
        M = dw([A0, 1.0, 2.0, p])
        np.testing.assert_allclose(M[0, 3], -ALPHA * np.tanh(p / WID), rtol=1e-12)
        np.testing.assert_allclose(np.abs(M).max(), abs(M[0, 3]), rtol=0.0)
    np.testing.assert_allclose(dw([A0, 1.0, 2.0, 0.1])[0, 3], -0.007598979245, atol=1e-12)
    np.testing.assert_allclose(dw([A0, 1.0, 2.0, 5.0])[0, 3], -0.02, atol=1e-16)

    q0 = jnp.asarray(flat_state(b.P.init))
    J = jax.jacobian(lambda q: b.omega(q, EMPTY, EMPTY)[0])(q0)
    comp = np.asarray(J.T - J)
    np.testing.assert_allclose(comp[0, PULLED_P], -ALPHA * np.tanh(5.0 / WID), rtol=1e-12)
    np.testing.assert_allclose(np.abs(comp).max(), ALPHA, rtol=1e-12)

    # the one return that repairs it, and the potential it comes from
    _, _, b_ex, _, F_ex = pulled(5.0, billed=True)
    for p in (0.1, 1.0, 5.0):
        assert float(np.abs(d_omega(F_ex)([A0, 1.0, 2.0, p])).max()) < 1e-17
    U = lambda e: -ALPHA * e[0] * WID * jnp.log(jnp.cosh(e[3] / WID))
    for e in ([A0, 1.0, 2.0, 0.3], [0.5, 1.0, 2.0, 2.0]):
        np.testing.assert_allclose(np.asarray(jax.grad(U)(jnp.asarray(e))),
                                   np.asarray(F_ex(jnp.asarray(e))), atol=1e-12)
    q0x = jnp.asarray(flat_state(b_ex.P.init))
    Jx = jax.jacobian(lambda q: b_ex.omega(q, EMPTY, EMPTY)[0])(q0x)
    assert float(np.abs(np.asarray(Jx.T - Jx)).max()) < 1e-15

    # and what it costs: a boundary, where the unbilled pull has none
    assert pulled_arrival(5.0, 15000, billed=True)[0] == 266
    assert pulled_arrival(5.0, 15000)[0] == 255
    assert pulled_arrival(40.0, 40000, billed=True)[0] is not None
    assert pulled_arrival(60.0, 40000, billed=True)[0] is None
    assert pulled_arrival(60.0, 40000)[0] is not None          # unbilled: it arrives
    np.testing.assert_allclose(LAM_A * A0 / ALPHA + WID * np.log(2.0), 50.173, atol=1e-3)
