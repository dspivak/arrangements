"""Member versus part: a proposed definition, run on the attention operad.

The definition under test is the book plan's proposed one (AGENTDOCPLAN.md §9.1,
*member*). For a part with cost ``k_i(q, e)`` -- here a component ``I -> B`` whose cost
depends on its own configuration ``q`` and on the input ``e in E`` it receives -- its
*residual* is

    r_i(e) = min_q k_i(q, e),

a function of what it receives alone. The part has a *need* iff ``r_i`` is not constant,
and its need point is ``argmin_e r_i(e)``. At a state of a composite the part is a
*member* of the collective iff

    r_i(received at that state) < r_i(0),        0 = (0_M, 0) in E:

the residual at what it receives is lower than the residual alone at zero input. A
bond-like cost the part can zero by moving itself is not a need; a "food" cost
``(1/2)(nu - readout(received))^2`` with ``nu != 0`` is.

The mathematics of the arrangements is ``background/affine_quadratic_operad.tex`` in the
book repository (def:arrangement, eq:maps, eq:composite-transfers, eq:composite-cost,
eq:spatial-transfers, eq:mean). The constructions follow ``tests/test_rendezvous.py`` and
``tests/test_reader.py``: components are plain ``Arrangement``s on the unit interface,
parents are attention arrangements, and each composite is
``compose_seq(tensor_arrangements([...]), parent.arr())``, run with ``run_closed``.
**Every docstring below says which of the brief's three tests it is and what its run
demonstrates.** None reproduces a closed form of the paper: the definition is not in the
paper, and the closed forms checked here are derived in the docstrings.

Two departures from the paper, both stated where they bite. The reader's cost takes the
*mean readout* ``h / w`` (eq:mean) of what it receives, as the brief asks; the paper keeps
that readout out of costs because it leaves the affine-quadratic family, and in test 2 it
does. And ``readout`` below extends eq:mean by ``0_M |-> 0``, which the library's ``mean``
refuses; the brief's ``r_R(0) = (1/2) nu^2`` is that extension.

The numbers, and what they do and do not support: ``notes/MEMBER-TEST.md``.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact
from arrangements.attention import (
    attention_arrangement, identity_transfers, input_dim, make_input, message, msg_dim,
    placement_arrangement, signal, spatial_arrangement, spatial_config, spatial_kappa,
    split_input, weight, zero_cost,
)
from arrangements.coalgebra import configuration
from arrangements.rvect import constant, euclidean, trivial
from arrangements.wiring import compose_seq, tensor_arrangements

from tests.test_rendezvous import run_closed

DH, DP = 1, 1                                   # H = W = P = R
MDIM, EDIM = msg_dim(DH), input_dim(DH, DP)     # dim M = 2, dim E = 3
NU = 1.0                                        # the reader's need: it wants to read nu
EXT = make_input(message(0.0, 0.0), 0.0)        # the exterior input (0_M, 0)
ZERO_INPUT = EXT                                # "alone at zero input" is the same point of E
TOL = 1e-9                                      # numerical margin on the strict inequality


# ---------------------------------------------------------------------------
# The readout, the need, and the parts.
# ---------------------------------------------------------------------------


def readout(m):
    """The mean readout ``h / w`` (eq:mean), extended by ``0_M |-> 0``.

    The library's ``mean`` is defined on ``M_{>0}`` only. The residual at zero input,
    ``r(0)``, evaluates a cost at ``0_M``, so the readout needs a value there, and ``0`` is
    the value the brief assumes when it writes ``r_R(0) = (1/2) nu^2``. The two-``where``
    form keeps the gradient finite at ``w = 0``; no run below ever delivers ``w = 0``.
    """
    w = weight(m)
    safe = jnp.where(w > 0.0, w, 1.0)
    return jnp.where(w > 0.0, signal(m) / safe, jnp.zeros_like(signal(m)))


def need_cost(e):
    """The "food" cost ``(1/2)|nu - readout(m)|^2`` on a received input ``e = (m, p)``.

    It depends on the message alone, not on the reference position, so the need is for a
    reading, wherever it is had.
    """
    m, _ = split_input(e, DH)
    return 0.5 * jnp.sum((NU - readout(m)) ** 2)


class Part(NamedTuple):
    """A component with its cost kept in hand, so the residual can be minimized."""

    arr: Arrangement
    cost: object          # (q, e) -> R_{>=0}
    q_dim: int
    label: str


def component(send, cost, P=None, label=""):
    """A component ``I -> B`` (as in test_rendezvous and test_reader): parameter ``P``
    (trivial by default), sending ``send(q) in M``, receiving ``e in E`` through the unique
    map to the unit's input, and carrying the cost ``cost(q, e)``."""
    P = configuration(trivial()) if P is None else P
    arr = Arrangement(
        P=P,
        in_dim_M=0, out_dim_M=0, in_dim_N=EDIM, out_dim_N=MDIM,
        out_f=lambda q, m_out: send(q),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -cost(q, n_in)),
        label=label,
    )
    return Part(arr, cost, P.Q_dim, label)


def reader(label="R"):
    """The needy reader: sends ``0_M``, has no configuration, cost ``need_cost``."""
    return component(lambda q: message(0.0, 0.0), lambda q, e: need_cost(e), label=label)


def mobile_source(mu, h0=0.0, label="S"):
    """A source whose signal is its configuration: sends ``(h, 1)`` with weight ``1``,
    zero cost, ``h`` with Euclidean mobility ``mu`` starting at ``h0``."""
    return component(
        lambda q: message(q[0], 1.0), lambda q, e: jnp.asarray(0.0),
        P=configuration(euclidean(1, mu), init=jnp.array([float(h0)])), label=label,
    )


def fixed_source(h, label):
    """A frozen source: sends ``(h, 1)``, zero cost, no configuration."""
    return component(lambda q: message(h, 1.0), lambda q, e: jnp.asarray(0.0), label=label)


def self_sufficient(mu, h0=0.0, label="C"):
    """The self-sufficient child: sends ``0_M``; its cost ``(1/2)(h_C - 1)^2`` depends on
    its own configuration only, so it can zero it by moving itself."""
    return component(
        lambda q: message(0.0, 0.0), lambda q, e: 0.5 * (q[0] - 1.0) ** 2,
        P=configuration(euclidean(1, mu), init=jnp.array([float(h0)])), label=label,
    )


# ---------------------------------------------------------------------------
# Residual, need, membership.
# ---------------------------------------------------------------------------


def residual(part, e, mu=0.5, steps=200):
    """``r(e) = min_q k(q, e)``: gradient descent in ``q`` from ``0``, which is exact
    enough for the quadratic costs here; the gradient at the reported minimizer is
    checked. A part with no configuration has ``r = k``."""
    e = jnp.asarray(e, float)
    if part.q_dim == 0:
        return float(part.cost(jnp.zeros(0), e))
    g = jax.grad(part.cost, argnums=0)
    q = jnp.zeros(part.q_dim)
    for _ in range(steps):
        q = q - mu * g(q, e)
    assert float(jnp.max(jnp.abs(g(q, e)))) < 1e-8
    return float(part.cost(q, e))


SAMPLE_INPUTS = [
    ZERO_INPUT,
    make_input(message(NU, 1.0), 0.0),
    make_input(message(0.5, 1.0), 0.3),
    make_input(message(-2.0, 4.0), -1.0),
]


def has_need(part, samples=SAMPLE_INPUTS):
    """Is the residual non-constant over the sample inputs? (A check at sample points.)"""
    rs = [residual(part, e) for e in samples]
    return max(rs) - min(rs) > TOL


def membership(part, e_received):
    """``(r(received), r(0), member?)`` for a part at what it receives."""
    r_recv = residual(part, e_received)
    r_zero = residual(part, ZERO_INPUT)
    return r_recv, r_zero, bool(r_recv < r_zero - TOL)


# ---------------------------------------------------------------------------
# Composite plumbing: what each part receives at a composite configuration.
# ---------------------------------------------------------------------------


def split_configs(parts, q):
    """A composite's flat configuration cut into the parts' blocks, in order, and the
    remainder (the parents'): the order ``compose_seq`` gives the tensored parameter."""
    qs, i = [], 0
    for p in parts:
        qs.append(q[i:i + p.q_dim])
        i += p.q_dim
    return qs, q[i:]


def emitted(parts, qs):
    """The parts' messages, concatenated in index order."""
    return jnp.concatenate([p.arr.out_f(qi, jnp.zeros(0)) for p, qi in zip(parts, qs)])


def received_inputs(parent, parts, q, e_ext=EXT):
    """What each part receives at composite configuration ``q``: the parent's ``rho``
    (eq:maps) at the parts' emitted messages and the exterior input."""
    qs, q_parent = split_configs(parts, q)
    m = emitted(parts, qs)
    return [parent.received(q_parent, m, e_ext, i + 1) for i in range(len(parts))]


def settled(arr, q, e_ext=EXT, tol=1e-9):
    """Is the forcing on the configuration below ``tol`` at ``q``?"""
    xi_q, _, _ = arr.omega(jnp.asarray(q), jnp.zeros(0), e_ext)
    return float(jnp.max(jnp.abs(xi_q))) < tol if xi_q.shape[0] else True


# ---------------------------------------------------------------------------
# Test 1: member versus part.
# ---------------------------------------------------------------------------


def collective_1(mu=0.5):
    """A pure-placement parent over ``[S, R, C]``: identity transfers, zero cost, trivial
    parameter (positions play no role under identity transfers). ``S`` is the mobile
    source, ``R`` the needy reader, ``C`` the self-sufficient child; the composite's
    configuration is ``(h_S, h_C)``."""
    parts = [mobile_source(mu), reader(), self_sufficient(mu)]
    parent = placement_arrangement(3, DH, DP, lambda q: jnp.zeros((3, DP)), label="place")
    return parts, parent, compose_seq(tensor_arrangements([p.arr for p in parts]), parent.arr())


def test_1_member_versus_part():
    """TEST 1 OF 3, member versus part. ILLUSTRATES the definition separating a needy part
    from a self-sufficient one at the settled state of one collective.

    Under identity transfers ``R`` receives ``m_S + m_C + m_0 = (h_S, 1)`` (``C`` and the
    exterior send ``0_M``), so its cost pulled back to the composite is
    ``(1/2)(nu - h_S)^2``, and ``C``'s is ``(1/2)(h_C - 1)^2``; both coordinates descend
    from ``0`` with mobility ``1/2`` and the composite settles at ``(h_S, h_C) = (nu, 1)``.
    There ``r_R(received) = 0 < r_R(0) = (1/2) nu^2``, so ``R`` is a member; ``r_C`` is
    ``0`` at every input, so ``C`` has no need and is not a member, though it moved. At
    the initial state ``R`` reads ``0`` and is not yet a member: ``r_R(received) = r_R(0)``.
    Positions receive no force at all here, since the transfers ignore them; nothing
    "draws ``R`` and ``S`` together", and membership comes from ``h_S`` alone.
    """
    parts, parent, b = collective_1(mu=0.5)
    S, R, C = parts
    tr = run_closed(b, 100, lambda t: EXT)
    q0, q1 = jnp.asarray(tr[0]), jnp.asarray(tr[-1])
    assert settled(b, q1)
    np.testing.assert_allclose(q1, [NU, 1.0], atol=1e-12)

    # need: R has one, S and C do not (the sample-point check)
    assert has_need(R)
    assert not has_need(S)
    assert not has_need(C)

    # what each receives at the settled state
    e1 = received_inputs(parent, parts, q1)
    np.testing.assert_allclose(e1[1], make_input(message(NU, 1.0), 0.0), atol=1e-12)

    rR_recv, rR_zero, R_member = membership(R, e1[1])
    rC_recv, rC_zero, C_member = membership(C, e1[2])
    rS_recv, rS_zero, S_member = membership(S, e1[0])
    np.testing.assert_allclose(rR_zero, 0.5 * NU ** 2, atol=1e-15)
    assert rR_recv < 1e-12 and R_member
    assert rC_recv < 1e-12 and rC_zero < 1e-12 and not C_member
    assert rS_recv == 0.0 and rS_zero == 0.0 and not S_member

    # at the start R receives (0, 1), reads 0, and is not a member: a tie, not a margin
    e0 = received_inputs(parent, parts, q0)
    r_start, _, member_start = membership(R, e0[1])
    np.testing.assert_allclose(r_start, 0.5 * NU ** 2, atol=1e-15)
    assert not member_start

    # C zeroed its own cost by moving itself, from 0 to 1
    np.testing.assert_allclose(float(C.cost(q0[1:2], e0[2])), 0.5, atol=1e-15)
    np.testing.assert_allclose(float(C.cost(q1[1:2], e1[2])), 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# Test 2: leaving.
# ---------------------------------------------------------------------------


TAU = 0.8                                       # the Gaussian scale of eq:spatial-transfers
D = float(np.sqrt(2.0 * TAU * np.log(2.0)))     # the paper's d = sqrt(2 tau log 2): kappa = 1/2


def collective_2(mu):
    """The spatial arrangement of test_reader over ``[R, A, B]``: ``A`` at ``-d`` sending
    ``(0, 1)``, ``B`` at ``+d`` sending ``(nu, 1)``, ``R`` at ``0``; value maps identity.
    Only ``p_R`` has mobility (``mu``); the sources' positions and all value maps are
    frozen (mobility ``0``), so the reader does all the moving."""
    parts = [reader(), fixed_source(0.0, "A"), fixed_source(NU, "B")]
    mobilities = jnp.array([mu, 0.0, 0.0, 0.0, 0.0, 0.0])
    a = spatial_arrangement(
        3, DH, DP, TAU, cost=zero_cost,
        parameter=configuration(constant(jnp.diag(mobilities)),
                                init=spatial_config((0.0, -D, D), (1.0, 1.0, 1.0))),
        label="spatial(3)")
    return parts, a, compose_seq(tensor_arrangements([p.arr for p in parts]), a.arr())


def reader_reads(p_R, p_A=-D, p_B=D):
    """The closed form of what ``R`` reads at position ``p_R``: the sources' messages
    arrive as ``kappa_A (0, 1) + kappa_B (nu, 1)``, so the mean readout is
    ``nu kappa_B / (kappa_A + kappa_B)``. Written only to be checked against."""
    kA = spatial_kappa(jnp.array([p_R]), jnp.array([p_A]), TAU)
    kB = spatial_kappa(jnp.array([p_R]), jnp.array([p_B]), TAU)
    return NU * kB / (kA + kB)


def test_2_leaving():
    """TEST 2 OF 3, leaving. CHECKS THE DISCRETE ANALOGUE of the paper's Example 1 in
    structure: the reader moves toward the source whose signal reduces its cost.

    ``R`` reads ``nu kappa_B / (kappa_A + kappa_B)``; its cost as a function of ``p_R``
    is ``(1/2) nu^2 (1 - s)^2`` with ``s = sigmoid(2 d p_R / tau)``, decreasing in
    ``p_R`` without a minimum: the reading equals ``nu`` only where ``A``'s kernel
    vanishes. So the run shows ``R`` moving toward ``B`` and away from ``A`` at every
    step, its distance to ``A`` growing and its residual falling monotonically; it also
    shows ``R`` passing ``B`` and continuing, with the force decaying. The positional
    forcing at the initial configuration is ``(d/4tau, -d/8tau, -d/8tau) nu^2`` on
    ``(R, A, B)``, summing to zero; the sources receive their covectors and, frozen,
    do not move. ``R`` is a member from the start (it reads ``nu/2``) and more so after.
    """
    mu, steps = 0.5, 300
    parts, a, b = collective_2(mu)
    R = parts[0]
    q0 = a.P.init

    # the positional forcing at the initial configuration
    xi_q, _, _ = b.omega(q0, jnp.zeros(0), EXT)
    c = NU ** 2 * D / (4.0 * TAU)
    np.testing.assert_allclose(np.asarray(xi_q)[:3], [c, -c / 2.0, -c / 2.0], atol=1e-14)

    tr = run_closed(b, steps, lambda t: EXT)
    p_R = tr[:, 0]
    np.testing.assert_allclose(tr[:, 1], -D, atol=1e-15)     # A never moves
    np.testing.assert_allclose(tr[:, 2], D, atol=1e-15)      # B never moves
    dist_A, dist_B = np.abs(p_R + D), np.abs(p_R - D)

    # toward B and away from A, at every step
    assert np.all(np.diff(p_R) > 0.0)
    assert np.all(np.diff(dist_A) > 0.0)
    crossing = int(np.argmax(p_R > D)) if np.any(p_R > D) else None
    assert crossing is not None                              # it passes B ...
    assert np.all(np.diff(dist_B[:crossing]) < 0.0)          # ... having approached it until then
    assert np.all(np.diff(dist_B[crossing:]) > 0.0)          # ... and then leaves it too

    # what it reads, and its residual, along the run
    reads = np.array([float(readout(split_input(received_inputs(a, parts, jnp.asarray(q))[0], DH)[0])[0])
                      for q in tr])
    np.testing.assert_allclose(reads, [float(reader_reads(p)) for p in p_R], atol=1e-12)
    res = 0.5 * (NU - reads) ** 2
    assert np.all(np.diff(res) < 0.0)
    r0, r_zero, member0 = membership(R, received_inputs(a, parts, jnp.asarray(tr[0]))[0])
    r1, _, member1 = membership(R, received_inputs(a, parts, jnp.asarray(tr[-1]))[0])
    np.testing.assert_allclose(r0, 0.125 * NU ** 2, atol=1e-14)
    assert member0 and member1 and r1 < r0 < r_zero
    assert not settled(b, tr[-1], tol=1e-9)                  # the force has not vanished


# ---------------------------------------------------------------------------
# Test 3: a member of a member, outweighed.
# ---------------------------------------------------------------------------


def outer_cost(lam):
    """``k_a(q, m~, e) = lam (1/2) |readout(m~) - 0|^2``: the outer parent wants the
    message its one child ``b`` sends to read ``0``."""
    return lambda q, m_out, n_in: lam * 0.5 * jnp.sum(readout(m_out) ** 2)


def nest_3(lam, mu):
    """``a(b(R, S))``: ``b`` a pure-placement parent over ``[R, S]``, ``a`` a unary parent
    over ``[b]`` with identity transfers and cost ``outer_cost(lam)``; both parents have
    the trivial parameter, so the composite's configuration is ``(h_S)`` alone."""
    parts = [reader(), mobile_source(mu)]
    b = placement_arrangement(2, DH, DP, lambda q: jnp.zeros((2, DP)), label="b")
    a = attention_arrangement(
        1, DH, DP, lambda q: jnp.zeros((1, DP)), identity_transfers(1, DH),
        cost=outer_cost(lam), label="a")
    inner = compose_seq(tensor_arrangements([p.arr for p in parts]), b.arr())
    return parts, b, a, compose_seq(inner, a.arr())


def received_in_nest(a, b, parts, q, e_ext=EXT):
    """What each part of ``b`` receives inside ``a(b(...))``: ``b``'s ``rho`` at the parts'
    messages and at the input ``a``'s ``rho`` delivers to ``b`` (eq:local-interactions).
    Also returns ``b``'s outgoing message ``m~ = sigma_b``."""
    qs, rest = split_configs(parts, q)
    q_b, q_a = rest[:b.P.Q_dim], rest[b.P.Q_dim:]
    m = emitted(parts, qs)
    m_tilde = b.sigma(q_b, m)
    e_b = a.received(q_a, m_tilde, e_ext, 1)
    return [b.received(q_b, m, e_b, i + 1) for i in range(len(parts))], m_tilde


def test_3_member_of_a_member_outweighed():
    """TEST 3 OF 3, a member of a member, outweighed. CHECKS the composite against the
    closed form derived here, and ILLUSTRATES the definition under a competing need.

    ``b``'s outgoing message is ``m_R + m_S = (h_S, 1)``, so it carries ``S``'s signal as
    the brief assumes, and ``R`` receives ``(h_S, 1)`` through ``b`` (the exterior input
    reaching ``b`` through ``a`` is ``0_M``). The composite cost is
    ``(1/2)(nu - h_S)^2 + lam (1/2) h_S^2``, settling at ``h_S = nu / (1 + lam)``, where
    ``r_R(received) = (1/2) nu^2 (lam / (1 + lam))^2``. At ``lam = 0`` this is ``0``; it
    rises with ``lam`` toward ``r_R(0) = (1/2) nu^2`` and reaches it only as
    ``lam -> infinity``. So ``R`` is a member at every finite ``lam`` run here, by a margin
    ``(1/2) nu^2 (1 + 2 lam) / (1 + lam)^2`` that shrinks to ``0.0099`` at ``lam = 100``;
    membership is lost only in the limit, not at any of the four ``lam``.
    """
    mu, steps = 0.01, 3000
    table = []
    for lam in (0.0, 1.0, 10.0, 100.0):
        parts, b, a, comp = nest_3(lam, mu)
        R, S = parts
        e_parts0, m_tilde0 = received_in_nest(a, b, parts, jnp.array([0.0]))
        np.testing.assert_allclose(m_tilde0, message(0.0, 1.0), atol=1e-15)   # b sends S's message

        tr = run_closed(comp, steps, lambda t: EXT)
        q1 = jnp.asarray(tr[-1])
        assert settled(comp, q1, tol=1e-10)
        h_S = float(q1[0])
        np.testing.assert_allclose(h_S, NU / (1.0 + lam), atol=1e-10)

        e_parts, m_tilde = received_in_nest(a, b, parts, q1)
        np.testing.assert_allclose(m_tilde, message(h_S, 1.0), atol=1e-12)
        np.testing.assert_allclose(e_parts[0], make_input(message(h_S, 1.0), 0.0), atol=1e-12)
        r_recv, r_zero, member = membership(R, e_parts[0])
        np.testing.assert_allclose(r_recv, 0.5 * NU ** 2 * (lam / (1.0 + lam)) ** 2, atol=1e-10)
        np.testing.assert_allclose(r_zero, 0.5 * NU ** 2, atol=1e-15)
        table.append((lam, h_S, r_recv, r_zero - r_recv, member))

    lams, hs, rs, margins, members = map(np.array, zip(*table))
    assert rs[0] < 1e-12 and members[0]                       # a member at lam = 0
    assert np.all(np.diff(rs) > 0.0)                          # r_R rises with lam
    assert np.all(np.diff(margins) < 0.0)                     # the margin shrinks ...
    assert margins[-1] < 0.01                                 # ... to under a hundredth
    assert np.all(members)                                    # ... but never to zero here
