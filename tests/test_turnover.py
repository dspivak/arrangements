"""Ship of Theseus: does a collective's form survive swapping a part?

The construction is the book plan's (AGENTDOCPLAN.md §5.9 item 13 in the book
repository), from David's design, in his words: "could we do a test where we have some
guys handling something together with some redundancy and then one part breaks down and
that causes it to be attentionally moved out and another part swapped in?" A parent over
``n`` redundant contributors, its cost on an aggregate of what they send, so losing one
degrades and does not kill; each contributor carries a health configuration gating what
it sends; one part's health is driven to zero; a spare sits off to the side. Success is
measured on what the collective sends onward: the form is held through turnover iff that
outgoing message returns to its pre-failure value.

**The four predictions, fixed in advance and written here before any construction code
existed. Each must be able to fail.**

(1) With no spare the collective settles at a degraded outgoing value, and with a spare
recruited it returns to the pre-failure one.

(2) A failed part is pushed out only if it keeps claiming attention. A part that sends a
bad signal at nonzero weight drags the readout, so descent moves the collective away from
it -- the same mechanism as the member test's leaving run. A part that goes silent, weight
zero, is not pushed out at all, because there is nothing to move away from; it is
forgotten for free. If this holds, the interesting failure in a collective is the one that
keeps reporting.

(3) A spare is recruited only from within the pull region. Far out, the kernel and its
gradient both vanish, so the spare never arrives. So a collective recruits only from what
is already near it, which would be a substantive claim about collectives and not only
about the model.

(4) If (1)-(3) hold, the form is held through turnover and Ship of Theseus is modeled for
this case; if (3) fails in the direction of never recruiting, the mathematics has failure
and forgetting but not replacement.

The mathematics of the arrangements is ``background/affine_quadratic_operad.tex`` in the
book repository (def:arrangement, eq:maps, eq:spatial-state, eq:spatial-transfers,
eq:spatial-mean, eq:mean). The constructions follow ``tests/test_member.py`` and
``tests/test_reader.py``: contributors are plain ``Arrangement``s on the unit interface,
the parent is a spatial attention arrangement, and each composite is
``compose_seq(tensor_arrangements([...]), parent.arr())``, run with ``run_closed``. Every
docstring below says which of three things its test does (STATUS.md §3.7): reproduce a
closed form derived in the docstring, check a discrete analogue, or correct a claim made
on paper.

The numbers, and which verdicts the parameters decide rather than discover:
``notes/TURNOVER-TEST.md``.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact
from arrangements.attention import (
    input_dim, make_input, message, msg_dim, signal, spatial_arrangement, spatial_config,
    spatial_kappa, spatial_split, split_messages, weight,
)
from arrangements.coalgebra import configuration
from arrangements.functors import Phi
from arrangements.rvect import constant, euclidean
from arrangements.wiring import compose_seq, tensor_arrangements

from tests.chain import flat_state
from tests.test_rendezvous import run_closed, run_phi

DH, DP = 1, 1                                   # H = W = P = R
MDIM, EDIM = msg_dim(DH), input_dim(DH, DP)     # dim M = 2, dim E = 3
EXT = make_input(message(0.0, 0.0), 0.0)        # the exterior input (0_M, 0)
EMPTY = jnp.zeros(0)

TAU = 0.8          # the single scalar scale of the spatial kernel (eq:spatial-transfers)
SIG = 1.0          # the value a contributor in full health sends
WGT = 1.0          # the weight a contributor claims while it keeps reporting
NU = 2.5           # the parent's target for the aggregate it sends onward
NU_BAR = 1.0       # the parent's target under the mean readout (a healthy part reads SIG)
LAM = 50.0         # the stiffness of a contributor's own health cost
MU_C = 0.01        # the mobility of a health coordinate (LAM * MU_C = 1/2 per step)
MU_P = 0.5         # the mobility of a position coordinate
STEPS = 8000       # steps per settling run
SETTLED = 1e-9     # the forcing below which a run is called settled


# ---------------------------------------------------------------------------
# The contributors: what a health configuration gates.
# ---------------------------------------------------------------------------


def readout(m):
    """The mean readout ``h / w`` (eq:mean), extended by ``0_M |-> 0`` as in
    ``tests/test_member.py``. The library's ``mean`` is defined on ``M_{>0}`` only, and a
    silent contributor makes ``0_M`` reachable. The two-``where`` form keeps the gradient
    finite at ``w = 0``."""
    w = weight(m)
    safe = jnp.where(w > 0.0, w, 1.0)
    return jnp.where(w > 0.0, signal(m) / safe, jnp.zeros_like(signal(m)))


def reporting(q):
    """A contributor whose health gates its *signal* and not its weight: it sends
    ``(c SIG, WGT)``. At ``c = 0`` it still claims weight ``WGT`` while signalling ``0``."""
    return message(q[0] * SIG, WGT)


def silent(q):
    """A contributor whose health gates signal and weight together: it sends
    ``(c SIG, c WGT)``, so at ``c = 0`` it sends ``0_M`` and claims nothing."""
    return message(q[0] * SIG, q[0] * WGT)


def corrupting(q):
    """A contributor whose health gates its signal *through zero*: it sends
    ``((2c - 1) SIG, WGT)``, so full health signals ``+SIG`` and ``c = 0`` signals
    ``-SIG`` at full weight. Not one of the brief's two modes; it is run in test 3 to
    locate what the push-out mechanism actually turns on."""
    return message((2.0 * q[0] - 1.0) * SIG, WGT)


MODES = {"reporting": reporting, "silent": silent, "corrupting": corrupting}


class Spec(NamedTuple):
    """One contributor: how its health gates what it sends, its initial health, the
    health its own cost wants, its health mobility, and where the parent has it."""

    mode: str
    c0: float
    target: float
    mu: float
    pos: float
    lam: float = LAM


def healthy(pos, mode="reporting"):
    """A contributor in full health whose health does not move (mobility ``0``)."""
    return Spec(mode, 1.0, 1.0, 0.0, pos)


def failing(pos, mode="reporting", lam=LAM):
    """The contributor whose health is driven to zero: its own cost ``(lam/2) c^2`` pulls
    ``c`` down from ``1``. The mobility is ``0.5 / lam``, so the health halves each step
    whatever the stiffness, and the sweep over ``lam`` in test 5 is a sweep over how hard
    the failure is rather than over the step."""
    return Spec(mode, 1.0, 0.0, 0.5 / lam, pos, lam)


def dead(pos, mode="reporting"):
    """A contributor held at zero health (mobility ``0``): the failure as a state
    rather than as a process, used where a force is read at one configuration."""
    return Spec(mode, 0.0, 0.0, 0.0, pos)


class Part(NamedTuple):
    arr: Arrangement
    spec: Spec
    label: str


def contributor(spec, label):
    """A component ``I -> B`` (the idiom of ``tests/test_member.py``): parameter the
    configuration coalgebra on its health ``c`` with Euclidean mobility, sending
    ``MODES[spec.mode](c)``, and carrying its own health cost
    ``(lam/2)(c - target)^2`` as the exact forcing ``-dk``."""
    send = MODES[spec.mode]
    P = configuration(euclidean(1, spec.mu), init=jnp.array([float(spec.c0)]))
    arr = Arrangement(
        P=P,
        in_dim_M=0, out_dim_M=0, in_dim_N=EDIM, out_dim_N=MDIM,
        out_f=lambda q, m_out: send(q),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -0.5 * spec.lam * (q[0] - spec.target) ** 2),
        label=label,
    )
    return Part(arr, spec, label)


# ---------------------------------------------------------------------------
# The parent: what the collective sends onward, and the cost on it.
# ---------------------------------------------------------------------------


def outgoing(n, q_parent, m_out):
    """``sigma_a(q, m) = sum_{j=1}^n T_{0j}(q) m_j`` (eq:maps), written out here.

    With the conventions ``l_0 = 0_P`` and ``V_0 = H`` of eq:spatial-transfers the
    kernel to the exterior slot is ``kappa_j = exp(-||p_j||^2 / 2 tau)`` and
    ``T_{0j} m_j = (kappa_j V_j h_j, kappa_j w_j)``, so the collective sends

        m~ = (sum_j kappa_j V_j h_j,  sum_j kappa_j w_j).

    Checked against ``a.sigma`` in test 1; written only to be checked against, and so
    that the cost below can be formed before the arrangement exists.
    """
    pos, V = spatial_split(q_parent, n, DH, DP)
    ms = split_messages(m_out, n, DH)
    h, w = jnp.zeros(DH), jnp.asarray(0.0)
    for j in range(n):
        k = spatial_kappa(pos[j], jnp.zeros(DP), TAU)
        h = h + k * (V[j] @ signal(ms[j]))
        w = w + k * weight(ms[j])
    return message(h, w)


def total_cost(n, nu=NU):
    """``k(q, m, e) = (1/2)|nu - h~|^2`` on the *weighted* signal the collective sends.

    This is the aggregate the brief's constraint asks for: a cost on the weighted input,
    which keeps ``k(q, -)`` of degree two in ``D_n`` (def:arrangement) since ``h~`` is
    linear in the child messages. It reads the signal only; the weight ``w~`` does not
    enter it.
    """
    return lambda q, m_out, n_in: 0.5 * jnp.sum((nu - signal(outgoing(n, q, m_out))) ** 2)


def mean_cost(n, nu=NU_BAR):
    """``k(q, m, e) = (1/2)|nu - h~/w~|^2`` on the *mean* readout of what the collective
    sends (eq:spatial-mean with ``i = 0``). It leaves the affine-quadratic family --
    ``attention.py`` says so and ``notes/MEMBER-TEST.md`` records the consequence -- and
    is run here only where prediction (2) requires a readout that sees weight."""
    return lambda q, m_out, n_in: 0.5 * jnp.sum((nu - readout(outgoing(n, q, m_out))) ** 2)


COSTS = {"total": total_cost, "mean": mean_cost}


def collective(specs, cost_kind="total", nu=None, mu_pos=MU_P):
    """The composite: the contributors tensored, then the spatial parent.

    The parent's configuration is ``(p_1..p_n, V_1..V_n)`` (eq:spatial-state); the value
    maps start at the identity and are frozen (mobility ``0``), so the only adaptation the
    parent has is attentional -- where it holds each contributor. ``tau`` is one scalar for
    every pair, as eq:spatial-transfers gives it. The composite configuration is
    ``(c_1..c_n, p_1..p_n, V_1..V_n)``.
    """
    n = len(specs)
    nu = (NU if cost_kind == "total" else NU_BAR) if nu is None else nu
    parts = [contributor(s, f"C{i}") for i, s in enumerate(specs)]
    mobilities = jnp.array([mu_pos] * n + [0.0] * (n * DH * DH))
    a = spatial_arrangement(
        n, DH, DP, TAU, cost=COSTS[cost_kind](n, nu),
        parameter=configuration(
            constant(jnp.diag(mobilities)),
            init=spatial_config([s.pos for s in specs], [1.0] * n)),
        label=f"parent({n})")
    b = compose_seq(tensor_arrangements([p.arr for p in parts]), a.arr())
    return parts, a, b


# ---------------------------------------------------------------------------
# Reading a composite configuration.
# ---------------------------------------------------------------------------


def split_composite(q, n):
    """``q |-> (healths, positions, value maps)``: the order ``compose_seq`` gives the
    tensored parameter, the contributors' blocks first and the parent's last."""
    q = jnp.asarray(q, float)
    pos, V = spatial_split(q[n:], n, DH, DP)
    return q[:n], jnp.ravel(pos), V


def emitted(parts, q):
    """The contributors' messages at a composite configuration, in index order."""
    n = len(parts)
    return jnp.concatenate([p.arr.out_f(q[i:i + 1], EMPTY) for i, p in enumerate(parts)])


def sent_onward(parts, a, q):
    """``m~``, the message the collective sends onward at composite configuration ``q``."""
    q = jnp.asarray(q, float)
    return a.sigma(q[len(parts):], emitted(parts, q))


def kappas(q, n):
    """``kappa_j = exp(-||p_j||^2 / 2 tau)``, the parent's attention to each contributor."""
    _, pos, _ = split_composite(q, n)
    return np.array([float(spatial_kappa(pos[j:j + 1], jnp.zeros(DP), TAU)) for j in range(n)])


def forcing(b, q):
    """The composite forcing's configuration component at ``q``."""
    xi_q, _, _ = b.omega(jnp.asarray(q, float), EMPTY, EXT)
    return np.asarray(xi_q)


def drifted(b, steps=STEPS, extra=500):
    """``(q at steps, how far it moves in a further ``extra`` steps)``.

    Settling is read off the *motion*, not off the forcing: a coordinate with mobility
    ``0`` (every frozen health here) carries a nonzero forcing forever without moving, so
    ``max|xi|`` never falls. This also gives a persistence claim its rate.
    """
    q1 = settle(b, steps)
    q2 = settle(b, steps + extra)
    return q2, float(np.max(np.abs(q2 - q1)))


def settle(b, steps=STEPS):
    """The final configuration of the closed composite, as ``run_closed`` computes it but
    inside a ``fori_loop`` (test 0 checks the two agree exactly)."""
    P, omega = b.P, b.omega

    def step(_i, s):
        xi_q, _, _ = omega(P.position(s), EMPTY, EXT)
        return P.step(s, xi_q)

    return np.asarray(flat_state(jax.lax.fori_loop(0, steps, step, P.init)))


def equilibrium_position(n_live=3, extra=0.0, nu=NU):
    """The position ``p`` at which ``n_live`` contributors in full health, together with a
    further contribution ``extra`` to the weighted signal, send exactly ``nu``:
    ``n_live exp(-p^2 / 2 tau) + extra = nu``. Used to start a run *at* the pre-failure
    fixed point, so that the baseline is a fixed point by construction and the run has
    only to confirm that the forcing vanishes there."""
    kap = (nu - extra) / n_live
    return float(np.sqrt(-2.0 * TAU * np.log(kap)))


def kappa_at(d):
    return float(spatial_kappa(jnp.array([float(d)]), jnp.zeros(DP), TAU))


D_SPARE = 2.0      # where the spare sits off to the side in the headline runs
FAIL = 2           # which contributor's health is driven to zero


def pre_failure(with_spare=True, mode="reporting", cost_kind="total", d=D_SPARE):
    """Three contributors in full health, held at the position that makes the collective
    send exactly ``nu``, with a spare off to the side at ``d`` (also in full health).
    Every health is frozen, so this is the baseline the turnover has to return to."""
    extra = kappa_at(d) if with_spare else 0.0
    p = equilibrium_position(3, extra)
    specs = [healthy(p), healthy(-p), healthy(p, mode)]
    if with_spare:
        specs.append(healthy(d))
    return collective(specs, cost_kind)


def after_failure(with_spare=True, mode="reporting", cost_kind="total",
                  d=D_SPARE, lam=LAM, nu=None):
    """The same collective with contributor ``FAIL``'s health mobile and its own cost
    pulling it to ``0``: the failure as a process inside the dynamics."""
    extra = kappa_at(d) if with_spare else 0.0
    p = equilibrium_position(3, extra, nu or NU)
    specs = [healthy(p), healthy(-p), failing(p, mode, lam)]
    if with_spare:
        specs.append(healthy(d))
    return collective(specs, cost_kind, nu=nu)


# ---------------------------------------------------------------------------
# Test 0: the runs are Phi's.
# ---------------------------------------------------------------------------


def test_0_the_runs_are_the_dynamics_functors():
    """LICENSES the run forms used below; it establishes nothing about the predictions.
    ``Phi`` of the composite, ``run_closed``, and the ``fori_loop`` of ``settle`` agree
    over fifty steps to one unit in the last place, so every trajectory here is ``Phi``'s.
    (The mutual-need test got exact equality; here the jitted composite reassociates one
    sum of kernel terms, and the three runs differ by ``2.2e-16``.)"""
    _, _, b = after_failure()
    by_phi = run_phi(b, 50, EXT)
    by_closed = run_closed(b, 50, lambda t: EXT)
    assert float(np.max(np.abs(by_phi - by_closed))) < 1e-15
    assert float(np.max(np.abs(settle(b, 50) - by_closed[-1]))) < 1e-15


# ---------------------------------------------------------------------------
# Test 1: the baseline, and what the collective sends onward.
# ---------------------------------------------------------------------------


def test_1_the_baseline_is_a_fixed_point_and_sends_nu():
    """REPRODUCES the closed form derived here: the message the collective sends onward is
    ``sigma_a``, and the pre-failure state is an exact fixed point.

    ``sigma_a(q, m) = sum_j T_{0j} m_j = (sum_j kappa_j V_j h_j, sum_j kappa_j w_j)`` with
    ``kappa_j = exp(-p_j^2 / 2 tau)`` (``l_0 = 0_P``), which is what ``outgoing`` writes
    out. With ``V_j = H``, every contributor in full health sending ``(SIG, WGT)``, and the
    three positions chosen so that ``3 kappa(p) + kappa(d) = nu``, the signal sent onward
    is ``nu`` exactly, so ``dk = 0`` and *every* coordinate of the composite forcing
    vanishes: the baseline is a fixed point by construction and the run confirms it. Note
    the fixed point is not isolated -- the whole level set ``sum_j kappa_j = nu`` is fixed
    -- which is what redundancy means here.
    """
    parts, a, b = pre_failure()
    q0 = jnp.asarray(flat_state(b.P.init))
    n = len(parts)

    # the hand-derived outgoing message is the library's sigma
    m = emitted(parts, q0)
    np.testing.assert_allclose(outgoing(n, q0[n:], m), a.sigma(q0[n:], m), atol=0.0)

    np.testing.assert_allclose(float(signal(sent_onward(parts, a, q0))[0]), NU, atol=1e-14)
    assert float(np.max(np.abs(forcing(b, q0)))) < 1e-14

    q1 = settle(b, STEPS)
    np.testing.assert_array_equal(q1, np.asarray(q0))          # nothing moves at all
    np.testing.assert_allclose(kappas(q0, n)[:3].sum() + kappas(q0, n)[3], NU, atol=1e-14)


# ---------------------------------------------------------------------------
# Test 2: prediction (1) -- degrade, then return.
# ---------------------------------------------------------------------------


def test_2_prediction_1_degrades_without_a_spare_and_returns_with_one():
    """REPRODUCES the closed forms derived here for both settled states, and so tests
    prediction (1).

    Write ``c`` for the failed contributor's settled health and ``k_j`` for the parent's
    kernel to contributor ``j``. The forcing on ``c`` is ``-LAM c + (nu - h~) k_F SIG``:
    the collective's own deficit props the failing part up, so ``c`` settles at
    ``(nu - h~) k_F / LAM``, not at ``0``, unless the deficit itself vanishes.

    *No spare.* The two live contributors are pulled to ``p = 0`` (``k -> 1``) and so is
    the failed one, and ``h~ = 2 + c`` with ``c = (nu - h~)/LAM``, giving
    ``c = (nu - 2)/(1 + LAM) = 0.5/51 = 0.0098039`` and ``h~ = 2.0098039``: degraded from
    ``2.5``, and not dead.

    *With a spare at ``d``.* Three live contributions are available (two live parts and
    the spare), and ``3 > nu``, so the deficit can be closed: the run settles with
    ``h~ = nu`` exactly, and there the pull on the failed part's health vanishes, so
    ``c -> 0`` and the failed part really dies.

    *But which "outgoing value" returns.* The collective sends a message
    ``m~ = (h~, w~)``, and the run restores its *signal* only. Before the failure every
    part sends ``(SIG, WGT) = (1, 1)``, so ``h~ = w~ = nu = 2.5``. After the turnover, in
    the reporting mode, the dead part still claims its weight and the spare has added
    its own: ``h~ = 2.5`` again but ``w~ = 3.3742``, so the mean of what the collective
    sends has fallen from ``1`` to ``0.7409``. In the silent mode ``h_j = w_j`` for every
    contributor, so ``h~ = w~`` identically and the whole message returns to ``(2.5,
    2.5)``. **A failed part that keeps claiming attention leaves a permanent mark on what
    the collective sends, even after the signal is restored.**
    """
    rows = []
    for with_spare in (False, True):
        parts, a, b = after_failure(with_spare=with_spare)
        n = len(parts)
        q1, drift = drifted(b, STEPS)
        m = sent_onward(parts, a, q1)
        c = float(split_composite(q1, n)[0][FAIL])
        rows.append((float(signal(m)[0]), c, kappas(q1, n), drift, float(weight(m))))

    (h_bare, c_bare, k_bare, f_bare, _), (h_sp, c_sp, k_sp, f_sp, w_sp) = rows
    assert f_bare < SETTLED and f_sp < SETTLED   # (drift over a further 500 steps)

    # no spare: degraded, and not dead
    np.testing.assert_allclose(c_bare, (NU - 2.0) / (1.0 + LAM), atol=1e-9)
    np.testing.assert_allclose(h_bare, 2.0 + (NU - 2.0) / (1.0 + LAM), atol=1e-9)
    assert h_bare < NU - 0.4 and h_bare > 0.7 * NU
    np.testing.assert_allclose(k_bare[:3], 1.0, atol=1e-6)     # everyone pulled to the parent

    # with a spare: back to the pre-failure value, and the failed part dies
    np.testing.assert_allclose(h_sp, NU, atol=1e-9)
    assert c_sp < 1e-9
    assert k_sp[3] > 6.0 * kappa_at(D_SPARE)                    # the spare has come in ...
    assert k_sp[3] < 0.99                                       # ... only as far as it had to

    # the weight of the outgoing message does not return, in the mode that keeps reporting
    parts, a, b = pre_failure()
    m_pre = sent_onward(parts, a, jnp.asarray(flat_state(b.P.init)))
    np.testing.assert_allclose([float(signal(m_pre)[0]), float(weight(m_pre))], [NU, NU],
                               atol=1e-14)
    assert w_sp > NU + 0.8                                      # 3.3742: a permanent mark
    parts, a, b = after_failure(mode="silent")
    m_si = sent_onward(parts, a, settle(b, STEPS))
    np.testing.assert_allclose([float(signal(m_si)[0]), float(weight(m_si))], [NU, NU],
                               atol=1e-9)                       # silent: the whole message


def failed_state(d=D_SPARE, mode="reporting", cost_kind="total", with_spare=True):
    """The failure as a *state*: the same collective with contributor ``FAIL``'s health
    held at ``0`` and immobile, the live contributors and the spare in full health at the
    pre-failure positions. Only the parent's attention moves, so a force read here is the
    attentional force alone. The deficit at this state has a closed form: with
    ``3 kappa(p) + kappa(d) = nu`` the weighted signal is ``2 kappa(p) + kappa(d)``, so
    ``nu - h~ = kappa(p)`` -- exactly the failed contributor's share."""
    extra = kappa_at(d) if with_spare else 0.0
    p = equilibrium_position(3, extra)
    specs = [healthy(p), healthy(-p), dead(p, mode)]
    if with_spare:
        specs.append(healthy(d))
    return collective(specs, cost_kind)


def positional_forcing(b, n):
    """The composite forcing's position block at the initial configuration."""
    q0 = jnp.asarray(flat_state(b.P.init))
    return forcing(b, q0)[n:2 * n], q0


# ---------------------------------------------------------------------------
# Test 3: prediction (2) -- is a failed part pushed out?
# ---------------------------------------------------------------------------


def test_3_prediction_2_the_push_out_turns_on_the_readout_not_on_the_weight():
    """CORRECTS prediction (2) for the cost this construction is required to carry, and
    REPRODUCES the closed forms for the forces, derived here.

    At the failed state the failed contributor sits at ``p_F > 0`` with kernel ``k_F`` and
    the deficit is ``kappa(p)``. Write ``dk_F/dp_F = -(p_F/tau) k_F < 0``.

    *Cost on the weighted signal* ``h~ = sum_j k_j v_j`` (what the brief's constraint
    requires). The force on the failed part's position is
    ``xi = (nu - h~) v_F dk_F/dp_F``, with ``v_F`` what it signals. Reporting and silent
    both give ``v_F = 0``, so ``xi = 0`` in *both*: the weight never enters ``h~`` at all,
    and a contribution that is merely absent exerts no pull either way. The two modes are
    indistinguishable here. Only the corrupting mode, ``v_F = -SIG``, gives
    ``xi = -(nu - h~) SIG dk_F/dp_F > 0``: pushed out. **So under a cost on the weighted
    signal what decides the push-out is not whether the part claims weight but whether its
    contribution is wrong rather than absent.**

    *Cost on the mean readout* ``h~/w~`` (outside the affine-quadratic family; run here
    only because prediction (2) names a readout that sees weight). Now
    ``d(h~/w~)/dk_F = (v_F - h~/w~) w_F / w~``. Reporting keeps ``w_F = WGT`` while
    ``v_F = 0``, so lowering ``k_F`` raises the readout toward ``nu``: pushed out. Silent
    has ``w_F = 0``, so the readout does not depend on ``k_F`` at all -- and, the readout
    being already exactly ``nu``, the *whole* forcing is exactly zero and nothing in the
    collective moves. Forgotten for free, in the strongest sense the model allows.

    So prediction (2) is refuted for the required cost and holds for the mean readout.
    """
    table = {}
    for cost_kind in ("total", "mean"):
        for mode in ("reporting", "silent", "corrupting"):
            parts, a, b = failed_state(mode=mode, cost_kind=cost_kind)
            n = len(parts)
            xi_pos, q0 = positional_forcing(b, n)
            m = sent_onward(parts, a, q0)
            table[(cost_kind, mode)] = (
                np.asarray(xi_pos), float(signal(m)[0]), float(weight(m)),
                float(readout(m)[0]), float(np.max(np.abs(forcing(b, q0)))))

    p = equilibrium_position(3, kappa_at(D_SPARE))
    kF, deficit = kappa_at(p), kappa_at(p)
    dk = -(p / TAU) * kF

    # the weighted signal: both of the brief's modes give exactly zero force
    for mode in ("reporting", "silent"):
        xi = table[("total", mode)][0]
        assert xi[FAIL] == 0.0
    np.testing.assert_allclose(table[("total", "reporting")][0],
                               table[("total", "silent")][0], atol=0.0)
    np.testing.assert_allclose(table[("total", "reporting")][1],
                               table[("total", "silent")][1], atol=0.0)
    # ... and the corrupting mode is pushed out, with the derived force
    xi_c = table[("total", "corrupting")][0]
    h_c = table[("total", "corrupting")][1]
    np.testing.assert_allclose(xi_c[FAIL], -(NU - h_c) * SIG * dk, rtol=1e-12)
    assert xi_c[FAIL] > 0.0                                   # p_F > 0, so: away

    # the mean readout: reporting is pushed out, silent moves nothing at all
    xi_r = table[("mean", "reporting")][0]
    assert xi_r[FAIL] > 0.0
    np.testing.assert_allclose(
        xi_r[FAIL], (NU_BAR - table[("mean", "reporting")][3]) *
        (0.0 - table[("mean", "reporting")][3]) * WGT / table[("mean", "reporting")][2] * dk,
        rtol=1e-10)
    assert table[("mean", "silent")][4] == 0.0                # the whole forcing vanishes
    np.testing.assert_allclose(table[("mean", "silent")][3], NU_BAR, atol=1e-15)

    # and in a run under the mean readout: away at every step, or not a step at all
    parts, a, b = failed_state(mode="reporting", cost_kind="mean")
    tr = run_closed(b, 300, lambda t: EXT)
    n = len(parts)
    dist = np.abs(tr[:, n + FAIL])
    assert np.all(np.diff(dist) > 0.0)                        # away from the parent, always
    reads = np.array([float(readout(sent_onward(parts, a, jnp.asarray(q)))[0]) for q in tr])
    assert np.all(np.diff(reads) > 0.0) and reads[-1] < NU_BAR   # toward nu, never reaching it
    _, a2, b2 = failed_state(mode="silent", cost_kind="mean")
    tr2 = run_closed(b2, 300, lambda t: EXT)
    np.testing.assert_array_equal(tr2, np.repeat(tr2[:1], len(tr2), axis=0))


# ---------------------------------------------------------------------------
# Test 4: prediction (3) -- how far away a spare can be recruited from.
# ---------------------------------------------------------------------------


DISTANCES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)


def sent_drift(parts, a, b, steps=STEPS, extra=500):
    """What the collective sends after ``steps``, and how far that moves in ``extra``
    more: settling read off the observable the predictions are about."""
    h = [float(signal(sent_onward(parts, a, settle(b, n)))[0]) for n in (steps, steps + extra)]
    return h[1], abs(h[1] - h[0])


def bisect(pred, lo, hi, tol):
    """The largest ``d`` at which ``pred`` still holds, to within ``tol``."""
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if pred(mid) else (lo, mid)
    return lo


def spare_force(d):
    """The force the failed collective exerts on a spare sitting at ``d``, read off the
    composite forcing at the failed state: ``-kappa(p) (d/tau) kappa(d) SIG``."""
    parts, _, b = failed_state(d=d)
    return float(positional_forcing(b, len(parts))[0][3])


def recruited(d, steps=STEPS, cost_kind="total"):
    """Run the failed state with a spare at ``d`` and report what the collective ends up
    sending, how near the spare came, and the force it felt at the start."""
    parts, a, b = failed_state(d=d, cost_kind=cost_kind)
    n = len(parts)
    xi0, _ = positional_forcing(b, n)
    q1 = settle(b, steps)
    _, pos, _ = split_composite(q1, n)
    return (float(xi0[3]), float(signal(sent_onward(parts, a, q1))[0]),
            float(jnp.abs(pos[3])), kappas(q1, n)[3])


def test_4_prediction_3_the_pull_region_is_a_time_budget_not_a_boundary():
    """CORRECTS prediction (3): the kernel and its gradient do not vanish at any
    moderate distance, so what a distance sweep at fixed run length measures is a budget.

    At the failed state the force on the spare is ``(nu - h~) v dk_d/dp = -kappa(p)
    (d/tau) kappa(d)``, a Gaussian in ``d`` times ``d``: it is strictly negative (inward)
    for every finite ``d > 0``, and the deficit it multiplies does *not* go to zero while
    the spare stays out (the live parts can only reach ``kappa = 1``, leaving a deficit of
    ``nu - 2``). So the spare always eventually arrives, and the sweep below shows a
    threshold in ``d`` that moves with the number of steps -- at ``d = 4`` the spare is not
    recruited in 8000 steps and is recruited in 512000. The gradient vanishes *exactly*
    only where ``exp(-d^2/2 tau)`` underflows, which the bisection below locates near
    ``d = 33.6665``, where ``exp`` reaches the smallest normal double (jax flushes
    subnormals to zero); beyond that the spare never arrives at any run length, which is
    prediction (3)'s stated mechanism at a distance thirty-seven times the kernel's own
    scale ``sqrt(tau) = 0.894``.
    """
    rows = [(d,) + recruited(d) for d in DISTANCES]
    forces, sent, dists, ks = (np.array([r[i] for r in rows]) for i in (1, 2, 3, 4))

    assert np.all(forces < 0.0)                         # always inward, at every distance
    assert np.all(np.diff(np.abs(forces[1:])) < 0.0)    # vanishing as a Gaussian ...
    assert np.abs(forces[1]) == np.max(np.abs(forces))  # ... past its peak near sqrt(tau)
    near = sent > NU - 1e-9
    assert near[0] and near[3] and near[7]              # recruited out to d = 4.0
    assert not near[8] and not near[9]                  # not at 4.5 or 5.0 in 8000 steps
    assert np.all(np.diff(near.astype(int)) <= 0)       # one threshold, not a patchwork

    # where the spare arrives *after* the live pair has been pulled home (kappa = 1 each)
    # the settled kernel has a closed form; where it arrives first, the run lands
    # elsewhere on the fixed manifold ``sum_j kappa_j = nu`` (0.500467 at d = 2.0)
    np.testing.assert_allclose(ks[5], NU - 2.0, atol=1e-9)
    np.testing.assert_allclose(dists[5], np.sqrt(-2.0 * TAU * np.log(NU - 2.0)), atol=1e-8)
    assert 0.5 < ks[3] < 0.501

    # the threshold moves with the run length: one distance, five budgets
    budgets = [recruited(5.0, steps=n)[1] for n in (8000, 32000, 128000, 512000, 2000000)]
    assert all(b < NU - 0.4 for b in budgets[:4])       # degraded through 512000 steps ...
    assert budgets[-1] > NU - 1e-9                      # ... and recruited by 2000000
    assert recruited(6.0, steps=2000000)[1] < NU - 0.4  # but not from d = 6 even then

    # where the kernel, and then the force, is exactly zero: two bisections
    d_kappa = bisect(lambda x: kappa_at(x) != 0.0, 5.0, 100.0, 1e-9)
    np.testing.assert_allclose(  # the smallest *normal* double: jax flushes subnormals
        d_kappa, np.sqrt(-2.0 * TAU * np.log(np.finfo(np.float64).tiny)), rtol=1e-9)
    d_force = bisect(lambda x: spare_force(x) != 0.0, 5.0, 100.0, 1e-4)
    assert 33.5 < d_force < d_kappa < 33.67             # the product underflows first
    assert spare_force(d_kappa) == 0.0


# ---------------------------------------------------------------------------
# Test 5: prediction (4), and which verdicts the parameters decided.
# ---------------------------------------------------------------------------


def test_5_prediction_4_and_what_the_parameters_decided():
    """CORRECTS prediction (4) by showing that two of prediction (1)'s two halves are
    parameter inequalities rather than structure.

    *The redundancy margin ``nu`` against the number of live contributors.* "Losing one
    degrades and does not kill" needs ``2 < nu``: at ``nu <= 2`` the two survivors already
    meet the target by themselves, so nothing degrades and no spare is ever wanted -- there
    is no turnover to model. And "with a spare recruited it returns" needs ``nu < 3``, the
    number of live contributions available afterwards. So prediction (1) lives in
    ``2 < nu < 3``, which is the statement that the collective is asking for more than the
    survivors can give and no more than the survivors plus one spare can give.

    *The stiffness of the failure.* The failed part's settled health with no spare is
    ``(nu - 2)/(1 + lam)``, so ``lam`` decides how dead a dead part is: at ``lam = 0.5``
    the collective's own deficit holds it at a third of full health and the degradation is
    slight. The failure being a *failure* is a parameter choice, not a structural one.

    Prediction (4) is therefore answered as: (1) holds inside ``2 < nu < 3``; (2) is
    refuted for the required cost and holds for the mean readout; (3) fails in the
    direction of *always* recruiting, given time, not of never. So the mathematics has
    failure, forgetting and replacement, and Ship of Theseus is modeled for this case --
    but "the pull region" is a run length, and what is forgotten for free is a part whose
    contribution is absent, not one whose weight is zero.
    """
    nus = []
    for nu in (1.5, 2.0, 2.2, 2.5, 2.9):
        rows = []
        for with_spare in (False, True):
            parts, a, b = after_failure(with_spare=with_spare, nu=nu)
            rows.append(sent_drift(parts, a, b, 4 * STEPS, 4000))
        nus.append((nu, rows[0], rows[1]))

    for nu, (bare, bare_drift), (sp, sp_drift) in nus:
        np.testing.assert_allclose(sp, nu, atol=1e-8)         # the spare always restores it
        assert sp_drift < SETTLED
        if nu < 2.0:
            np.testing.assert_allclose(bare, nu, atol=1e-8)   # nothing to degrade
            assert bare_drift < SETTLED
        elif nu > 2.0:
            np.testing.assert_allclose(bare, 2.0 + (nu - 2.0) / (1.0 + LAM), atol=1e-8)
            assert bare < nu - 1e-3 and bare_drift < SETTLED

    # nu = 2.0 exactly, the survivors' own capacity, is the one degenerate case: the
    # deficit is quadratic in the positions there, so the force is cubic and the approach
    # is algebraic, not exponential -- 1.0e-4 short at 8000 steps and 2.5e-5 at 32000,
    # a factor four for a factor four. It is not settled, and no run length settles it.
    _, (_, drift_2), _ = nus[1]
    parts, a, b = after_failure(with_spare=False, nu=2.0)
    near, far = (float(signal(sent_onward(parts, a, settle(b, n)))[0])
                 for n in (4 * STEPS, 16 * STEPS))
    np.testing.assert_allclose((2.0 - near) / (2.0 - far), 4.0, rtol=1e-3)
    assert drift_2 > SETTLED

    lams = []
    for lam in (0.5, 5.0, 50.0, 500.0):
        parts, a, b = after_failure(with_spare=False, lam=lam)
        h, drift = sent_drift(parts, a, b, 4 * STEPS, 4000)
        c = float(split_composite(settle(b, 4 * STEPS), len(parts))[0][FAIL])
        lams.append((lam, c, h))
        assert drift < SETTLED

    for lam, c, h in lams:
        np.testing.assert_allclose(c, (NU - 2.0) / (1.0 + lam), atol=1e-9)
        np.testing.assert_allclose(h, 2.0 + (NU - 2.0) / (1.0 + lam), atol=1e-9)
    assert lams[0][2] > 2.3 and lams[-1][2] < 2.002             # slight, then total
