"""Mutual need: two parts, each one's held form supplied by the other, fed from outside.

The construction is the book plan's, not the operad's: AGENTDOCPLAN.md §5.6 (the driven
steady state) and §5.5 (retroactivity and the reciprocity lemma), with the four
predictions and the hand-checked linear failure of
``notes/MEMBER-CENSUS-2026-09-24.md`` §4.3 in the book repository. **The four predictions,
quoted from §4.3 and fixed before any run here:** for two parts, each of whose held form
is supplied by the other, with one supply entering the pair from outside, (1) both forms
are held while the supply runs; (2) cutting the supply ends both; (3) cutting either
internal link ends both; (4) neither part holds its form alone.

**The pieces.** Each part is §5.6's driven steady state: an arrangement
``<R^0|R^0> -> <R^0|R>`` whose parameter is the configuration coalgebra on its own
amplitude ``q``, mobility ``h``, publishing ``q`` at its port, and carrying its own
dissipation as the exact forcing of the cost ``(b/2) q^2`` (the motor's friction ``b``,
§5.6). The supply is CDLM's drive idiom, a box with zero response that holds its value:
the configuration coalgebra with ``sharp = 0`` and initial state ``s``, publishing ``s``.
The three boxes are tensored and closed by a *stateless* coupler
``<R^0|R^3> -> <R^0|R^0>`` whose forcing on its ports is a returned covector ``F(e)``,
exactly the two-tank idiom of ``tests/test_two_tanks.py``:

    F_A(e) = alpha (intake + beta q_B),   F_B(e) = alpha gamma q_A,   F_S(e) = f_S(e),

with ``e = (q_A, q_B, s)`` the three published values. The *linear* wiring of §4.3 takes
``intake = s``: the supply reaches ``A`` directly. The *gated* wiring takes

    intake = s g(q_A, q_B),    g(q_A, q_B) = q_A q_B / (K + q_A q_B),

the food entering only while the parts are working (§4.3: "the hands must work for the
food to come in"). Since every ``out_f`` is the identity, ``compose_seq``'s pullback is
trivial and the composite forcing on ``Q = R^3`` is

    xi_A = alpha (intake + beta q_B) - b q_A,   xi_B = alpha gamma q_A - b q_B,   xi_S = f_S,

and one step is ``q + h xi`` (STATUS.md §1: the configuration step *adds*; time is
discrete by design). Fixed points are the zeros of ``xi``, so they do not depend on ``h``;
what depends on ``h`` is which of them a run reaches. ``f_S`` is bookkeeping: the supply's
response is zero, so no choice of ``f_S`` moves anything, and test 2 uses that freedom.

**Two modeling choices, said as choices.** The gate's Michaelis-Menten shape is a model of
metabolism, not a derivation; and "cutting a link" is implemented as the wire delivering
``0`` at the coupler, which is how §4.3's own hand-check read it (its "link B->A cut" row
is ``q_B`` replaced by ``0`` in ``A``'s intake). Under the gate that zeroes ``q_B`` in the
gate as well as in the ``beta`` term, since one wire carries ``q_B`` to the coupler and the
coupler uses it in both places; test 7 runs the other reading, where only the ``beta`` term
is cut, and it reverses prediction 3.

Every docstring below says which of three things its test does (STATUS.md §3.7): reproduce
a closed form derived in the docstring, check a discrete analogue, or correct a claim made
on paper. The numbers, and which verdicts the parameters decide rather than discover:
``notes/MUTUAL-NEED-TEST.md``.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact, zero_form
from arrangements.coalgebra import configuration
from arrangements.functors import Phi
from arrangements.rvect import constant, euclidean
from arrangements.wiring import compose_seq, tensor_arrangements, trivial_parameter

from tests.chain import flat_state
from tests.test_rendezvous import run_closed, run_phi

ALPHA = 1.0        # the drive gain: how much of what a part receives becomes its amplitude
DISS = 1.0         # b, the dissipation of a part's own amplitude (§5.6's friction)
BETA = 0.5         # the internal coupling B -> A
GAMMA = 0.5        # the internal coupling A -> B
KGATE = 1.0        # the gate's half-saturation
SUPPLY = 4.0       # s, the value the supply box holds in the gated runs
LINEAR_SUPPLY = 1.0  # s in the linear runs, the value §4.3's hand-check used
MOB = 0.01         # h, the mobility of each part (its sharp is h I)
STEPS = 20000      # steps per run: 200 units of h-time
HELD = 1e-3        # a form is *held* at the end of a run iff its amplitude exceeds this
EMPTY = jnp.zeros(0)


# ---------------------------------------------------------------------------
# The three boxes and the coupler.
# ---------------------------------------------------------------------------


def part(q0, b=DISS, h=MOB, label="part"):
    """A part as §5.6's driven steady state: amplitude ``q`` with mobility ``h``, published
    at its port, carrying its own dissipation ``omega = -d (b/2) q^2``."""
    return Arrangement(
        P=configuration(euclidean(1, h), init=jnp.array([float(q0)])),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: EMPTY,
        omega=exact(lambda q, m_out, n_in: -0.5 * b * q[0] ** 2),
        label=label,
    )


def supply(s=SUPPLY):
    """The drive: a box with zero response holding its value (§5.6, CDLM ~l.2709)."""
    return Arrangement(
        P=configuration(constant(jnp.zeros((1, 1))), init=jnp.array([float(s)])),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: EMPTY,
        omega=zero_form,
        label="S",
    )


def coupler(F, ports, label="coupler"):
    """A stateless coupler whose forcing on its ``ports`` is the returned covector ``F``
    (``tests/test_two_tanks.py``)."""
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0, out_dim_M=ports, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, e: EMPTY,
        in_f=lambda q, e, n_in: EMPTY,
        omega=lambda q, e, n_in: (EMPTY, F(e), EMPTY),
        label=label,
    )


def michaelis(q_A, q_B, K=KGATE):
    """The gate ``g = u / (K + u)``, ``u = q_A q_B``: the supply enters only while both
    parts are active. ``g(0, ., .) = g(., 0, .) = 0`` and ``dg = 0`` at the origin."""
    u = q_A * q_B
    return u / (K + u)


def F_loop(beta=BETA, gamma=GAMMA, gate=michaelis, leak=0.0, alpha=ALPHA,
           f_supply=lambda e: 0.0, link_ba=True, link_ab=True, supply_on=True,
           beta_only_cut=False):
    """The returned covector ``F`` on the ports ``e = (q_A, q_B, s)``.

    ``gate=None`` is the linear wiring (``intake = s``). A cut wire delivers ``0``:
    ``link_ba=False`` zeroes ``q_B`` where it enters ``A``'s intake (gate and ``beta``
    term both), ``link_ab=False`` zeroes ``q_A`` where it enters ``B``'s, and
    ``supply_on=False`` zeroes ``s``. ``beta_only_cut`` is the other reading of the
    ``B -> A`` cut, run in test 7: the ``beta`` term alone is cut and the gate keeps
    reading ``q_B``. ``leak`` splits the intake into an ungated fraction ``leak`` and a
    gated ``1 - leak``.
    """

    def F(e):
        q_A, q_B, s = e[0], e[1], e[2]
        s = s if supply_on else 0.0 * s
        into_A = q_B if link_ba else 0.0 * q_B
        into_B = q_A if link_ab else 0.0 * q_A
        if gate is None:
            intake = s
        else:
            gate_B = q_B if (link_ba or beta_only_cut) else 0.0 * q_B
            intake = leak * s + (1.0 - leak) * s * gate(q_A, gate_B)
        beta_term = (0.0 * q_B if beta_only_cut else into_A)
        return jnp.array([alpha * (intake + beta * beta_term),
                          alpha * gamma * into_B,
                          f_supply(e)])

    return F


def loop(q_A=1.0, q_B=1.0, s=SUPPLY, h=MOB, b=DISS, **kw):
    """The closed arrangement ``compose_seq(tensor(A, B, S), coupler)``, configuration
    ``(q_A, q_B, s)`` in that order."""
    parts = [part(q_A, b=b, h=h, label="A"), part(q_B, b=b, h=h, label="B"), supply(s)]
    return compose_seq(tensor_arrangements(parts), coupler(F_loop(**kw), 3))


def alone(which, q0, s=SUPPLY, h=MOB, b=DISS, **kw):
    """One part and the supply, the other part absent: a genuine two-port composite whose
    coupler is ``F_loop`` with the absent port's value ``0``."""
    F3 = F_loop(**kw)
    i = 0 if which == "A" else 1
    if which == "A":
        place = lambda e: jnp.array([e[0], 0.0, e[1]])
    else:
        place = lambda e: jnp.array([0.0, e[0], e[1]])
    F = lambda e: jnp.array([F3(place(e))[i], 0.0])
    boxes = [part(q0, b=b, h=h, label=which), supply(s)]
    return compose_seq(tensor_arrangements(boxes), coupler(F, 2))


# ---------------------------------------------------------------------------
# Running, and the closed forms.
# ---------------------------------------------------------------------------


def settle(arr, steps=STEPS):
    """The final configuration of a closed arrangement, as ``run_closed`` computes it but
    inside a ``fori_loop`` (test 0 checks the two agree exactly)."""
    P, omega = arr.P, arr.omega

    def step(_i, s):
        xi_q, _, _ = omega(P.position(s), EMPTY, EMPTY)
        return P.step(s, xi_q)

    return np.asarray(flat_state(jax.lax.fori_loop(0, steps, step, P.init)))


def forcing(arr, q):
    """The composite forcing's configuration component at ``q``."""
    xi_q, _, _ = arr.omega(jnp.asarray(q, float), EMPTY, EMPTY)
    return np.asarray(xi_q)


def held(q):
    return abs(float(q)) > HELD


def linear_steady(s=LINEAR_SUPPLY, beta=BETA, gamma=GAMMA, alpha=ALPHA, b=DISS):
    """The linear wiring's steady state, derived in test 1's docstring:
    ``q_A = alpha s / (b - alpha^2 beta gamma / b)``, ``q_B = (alpha gamma / b) q_A``."""
    q_A = alpha * s / (b - alpha ** 2 * beta * gamma / b)
    return q_A, (alpha * gamma / b) * q_A


def gated_roots(s=SUPPLY, beta=BETA, gamma=GAMMA, alpha=ALPHA, b=DISS, K=KGATE):
    """The gated loop's nonzero fixed points and its supply threshold, derived in test 3's
    docstring: with ``c = alpha gamma / b`` and ``Bq = b - alpha^2 beta gamma / b``,

        Bq c q_A^2 - alpha s c q_A + Bq K = 0,
        s_* = 2 Bq sqrt(K / c) / alpha,

    returning ``(q_A^-, q_A^+, s_*)`` and ``nan`` for the roots below ``s_*``."""
    c = alpha * gamma / b
    Bq = b - alpha ** 2 * beta * gamma / b
    s_star = 2.0 * Bq * np.sqrt(K / c) / alpha
    disc = (alpha * s * c) ** 2 - 4.0 * Bq ** 2 * c * K
    if disc < 0.0:
        return float("nan"), float("nan"), s_star
    root = np.sqrt(disc)
    return ((alpha * s * c - root) / (2.0 * Bq * c),
            (alpha * s * c + root) / (2.0 * Bq * c), s_star)


def d_omega(F):
    """``(d omega)_{ij} = d_i F_j - d_j F_i``, the antisymmetric part of the
    returned-covector Jacobian (the reciprocity lemma, §5.5; ``tests/test_two_tanks.py``)."""
    def at(e):
        J = jax.jacobian(F)(jnp.asarray(e, float))   # J[j, i] = d F_j / d e_i
        return np.asarray(J.T - J)
    return at


# ---------------------------------------------------------------------------
# Test 0: the runs are Phi's.
# ---------------------------------------------------------------------------


def test_0_the_runs_are_the_dynamics_functors():
    """LICENSES the run forms used below; it establishes nothing about the predictions.
    ``Phi`` of the composite, ``run_closed``, and the ``fori_loop`` of ``settle`` agree
    exactly over fifty steps, so every trajectory here is ``Phi``'s."""
    b = loop()
    by_phi = run_phi(b, 50, EMPTY)
    by_closed = run_closed(b, 50, lambda t: EMPTY)
    assert float(np.max(np.abs(by_phi - by_closed))) == 0.0
    np.testing.assert_array_equal(settle(b, 50), by_closed[-1])


# ---------------------------------------------------------------------------
# Test 1: the linear wiring, and predictions 3 and 4.
# ---------------------------------------------------------------------------


def test_1_the_linear_wiring_fails_predictions_3_and_4():
    """REPRODUCES the closed form derived here and the hand-check of §4.3, and so CORRECTS
    two of the four predictions for this wiring.

    With ``intake = s`` the composite forcing vanishes where ``b q_A = alpha(s + beta q_B)``
    and ``b q_B = alpha gamma q_A``, i.e. at ``q_A = alpha s / (b - alpha^2 beta gamma / b)``
    and ``q_B = (alpha gamma / b) q_A``: with ``alpha = b = 1``, ``beta = gamma = 1/2`` and
    ``s = 1`` (§4.3's own values), ``(4/3, 2/3)``. Cutting the supply leaves the linear pair ``xi = ((-b, beta), (gamma,
    -b)) q``, whose eigenvalues ``-b +- alpha sqrt(beta gamma)`` are both negative here, so
    both amplitudes decay to zero: prediction 2 holds. Cutting ``B -> A`` leaves
    ``q_A = alpha s / b = 1`` and ``q_B = alpha gamma q_A / b = 1/2``, both held --
    prediction 3 fails. Cutting ``A -> B`` leaves ``q_A = 1`` and ``q_B = 0``, so ``A``
    holds its form with ``B`` dead -- predictions 3 and 4 both fail. The reason is
    structural: a supply reaching ``A`` directly makes ``B`` dispensable to ``A``.
    """
    q_A, q_B = linear_steady()
    np.testing.assert_allclose([q_A, q_B], [4.0 / 3.0, 2.0 / 3.0], atol=1e-14)
    lin = dict(gate=None, s=LINEAR_SUPPLY)

    on = settle(loop(**lin))
    np.testing.assert_allclose(on[:2], [q_A, q_B], atol=1e-9)
    assert held(on[0]) and held(on[1])                       # prediction 1 holds

    cut_s = settle(loop(**lin, q_A=q_A, q_B=q_B, supply_on=False))
    assert not held(cut_s[0]) and not held(cut_s[1])         # prediction 2 holds
    assert float(np.max(np.abs(cut_s[:2]))) < 1e-30

    cut_ba = settle(loop(**lin, q_A=q_A, q_B=q_B, link_ba=False))
    np.testing.assert_allclose(cut_ba[:2], [1.0, 0.5], atol=1e-9)
    assert held(cut_ba[0]) and held(cut_ba[1])               # prediction 3 FAILS

    cut_ab = settle(loop(**lin, q_A=q_A, q_B=q_B, link_ab=False))
    np.testing.assert_allclose(cut_ab[0], 1.0, atol=1e-9)
    assert held(cut_ab[0]) and not held(cut_ab[1])           # predictions 3 and 4 FAIL

    a_alone = settle(alone("A", q_A, **lin))
    np.testing.assert_allclose(a_alone[0], 1.0, atol=1e-9)
    assert held(a_alone[0])                                  # prediction 4 FAILS
    assert not held(settle(alone("B", q_B, **lin))[0])


# ---------------------------------------------------------------------------
# Test 2: is the linear loop exact?
# ---------------------------------------------------------------------------


def test_2_the_linear_coupler_is_exact_iff_the_internal_coupling_is_reciprocal():
    """REPRODUCES the closed form for ``d omega`` derived here.

    For the linear coupler ``F = (alpha(s + beta q_B), alpha gamma q_A, f_S)`` the
    antisymmetric part of ``DF`` has entries ``(d omega)_{AB} = alpha(gamma - beta)``,
    ``(d omega)_{AS} = d f_S/d q_A - alpha``, ``(d omega)_{BS} = d f_S / d q_B``. With
    ``f_S = 0`` it is nonzero -- the supply feeds ``A`` and nothing returns to it -- but
    the supply's response is zero, so ``f_S`` is free, and ``f_S = alpha q_A`` makes
    ``omega`` exact whenever ``beta = gamma``, with potential
    ``V = alpha(q_A s + beta q_A q_B)``; at ``beta != gamma`` no ``f_S`` repairs it, since
    the ``AB`` entry does not involve ``f_S``. So one-way supply is not what costs
    exactness here (STATUS.md §3.5 item 1, the same correction one case over): what would
    cost it is non-reciprocal internal coupling. That ``V`` is nevertheless not an
    attention cost: ``k = -V`` takes both signs, and the operad's costs are nonnegative.
    """
    rng = np.random.default_rng(11)
    pts = [rng.standard_normal(3) for _ in range(5)]

    bare = d_omega(F_loop(gate=None))
    for e in pts:
        np.testing.assert_allclose(bare(e), [[0.0, 0.0, -ALPHA],
                                             [0.0, 0.0, 0.0],
                                             [ALPHA, 0.0, 0.0]], atol=1e-12)

    booked = d_omega(F_loop(gate=None, f_supply=lambda e: ALPHA * e[0]))
    for e in pts:
        np.testing.assert_allclose(booked(e), np.zeros((3, 3)), atol=1e-12)

    skew = d_omega(F_loop(gate=None, beta=0.5, gamma=0.9, f_supply=lambda e: ALPHA * e[0]))
    for e in pts:
        np.testing.assert_allclose(skew(e)[0, 1], ALPHA * (0.9 - 0.5), atol=1e-12)

    V = lambda e: ALPHA * (e[0] * e[2] + BETA * e[0] * e[1])
    grad_V = jax.grad(V)
    for e in pts:                                    # the potential is the one named
        np.testing.assert_allclose(np.asarray(grad_V(jnp.asarray(e))),
                                   np.asarray(F_loop(gate=None,
                                                     f_supply=lambda x: ALPHA * x[0])(jnp.asarray(e))),
                                   atol=1e-12)
    vals = [float(V(jnp.asarray(e))) for e in pts]
    assert max(vals) > 0.0 and min(vals) < 0.0       # so -V is not a nonnegative cost


# ---------------------------------------------------------------------------
# Test 3: the gated loop, predictions 1 and 2.
# ---------------------------------------------------------------------------


def test_3_the_gated_loop_holds_both_forms_while_the_supply_runs():
    """REPRODUCES the closed forms derived here (predictions 1 and 2).

    With ``intake = s g``, ``g = q_A q_B/(K + q_A q_B)``, the forcing vanishes where
    ``q_B = c q_A``, ``c = alpha gamma / b``, and, writing ``Bq = b - alpha^2 beta gamma/b``,

        Bq c q_A^2 - alpha s c q_A + Bq K = 0,

    so besides the origin there are two fixed points when ``alpha^2 s^2 c >= 4 Bq^2 K``,
    i.e. above the supply threshold ``s_* = 2 Bq sqrt(K/c)/alpha``, which for
    ``alpha = b = K = 1``, ``beta = gamma = 1/2`` is ``3/sqrt(2) = 2.121320``, the two roots
    coinciding there at ``q_A = sqrt(K/c) = sqrt(2)``. At ``s = 4`` they are
    ``q_A = 4.927443`` and ``0.405890``, with ``q_B = q_A/2``. Prediction 1: from
    ``(1, 1)`` the loop settles at the upper root, forcing below ``1e-9``, with nonzero
    throughput -- at the fixed point the intake ``alpha s g = 3.695582`` balances the
    dissipation less what ``B`` returns, ``b q_A - alpha beta q_B``, so flux passes and
    this is a driven steady state, not an equilibrium. Prediction 2: with the supply cut the only fixed point is the origin
    (``Bq > 0`` here) and both amplitudes decay below ``1e-30``.
    """
    lower, upper, s_star = gated_roots()
    np.testing.assert_allclose(s_star, 3.0 / np.sqrt(2.0), atol=1e-14)
    np.testing.assert_allclose([lower, upper], [0.4058901, 4.9274432], atol=1e-6)

    on = loop()
    q = settle(on)
    np.testing.assert_allclose(q[:2], [upper, 0.5 * upper], atol=1e-6)
    assert held(q[0]) and held(q[1])                                   # prediction 1
    assert float(np.max(np.abs(forcing(on, q)))) < 1e-9
    intake = ALPHA * SUPPLY * float(michaelis(q[0], q[1]))
    np.testing.assert_allclose(intake, DISS * q[0] - ALPHA * BETA * q[1], rtol=1e-6)
    np.testing.assert_allclose(intake, 3.695582, atol=1e-5)            # flux passes

    cut = settle(loop(q_A=upper, q_B=0.5 * upper, supply_on=False))
    assert not held(cut[0]) and not held(cut[1])                       # prediction 2
    assert float(np.max(np.abs(cut[:2]))) < 1e-30


# ---------------------------------------------------------------------------
# Test 4: the gated loop, predictions 3 and 4.
# ---------------------------------------------------------------------------


def test_4_cutting_either_internal_link_or_either_part_ends_both():
    """REPRODUCES the closed forms derived here (predictions 3 and 4).

    Cut ``B -> A``: the coupler reads ``q_B = 0``, so ``g = 0`` and the ``beta`` term is
    ``0``, leaving ``xi_A = -b q_A``; ``A`` decays, and then ``xi_B = alpha gamma q_A - b
    q_B`` decays with it. Cut ``A -> B``: ``xi_B = -b q_B`` so ``q_B -> 0``, the gate
    closes with it, and ``A`` follows. Either way both amplitudes end below ``1e-84``: the
    decay is geometric by ``1 - h b = 0.99``, and ``0.99^20000 = 10^-87.3``. Prediction
    4: with ``B`` absent, ``A``'s intake is ``s g(q_A, 0) = 0``, and with ``A`` absent
    ``B``'s drive is ``alpha gamma . 0``; each alone decays from the upper root to zero.
    Every one of these four verdicts is a consequence of ``g`` vanishing when either
    argument does, not of the parameter values -- but test 7 shows that the same four
    verdicts are reversed by an arbitrarily small ungated leak, and by the other reading
    of the ``B -> A`` cut.
    """
    _, upper, _ = gated_roots()
    start = dict(q_A=upper, q_B=0.5 * upper)

    for cut in (dict(link_ba=False), dict(link_ab=False)):
        q = settle(loop(**start, **cut))
        assert not held(q[0]) and not held(q[1])                       # prediction 3
        np.testing.assert_allclose(q[:2], [0.0, 0.0], atol=1e-84)

    for which, q0 in (("A", upper), ("B", 0.5 * upper)):
        q = settle(alone(which, q0))
        assert not held(q[0])                                          # prediction 4
        np.testing.assert_allclose(q[0], 0.0, atol=1e-84)


# ---------------------------------------------------------------------------
# Test 5: the gate as a forcing, and which side of the operad it lands on.
# ---------------------------------------------------------------------------


def test_5_the_gate_is_a_non_exact_forcing_and_no_bookkeeping_repairs_it():
    """REPRODUCES the closed form for ``d omega`` derived here.

    ``dF_A/dq_B = alpha(beta + s dg/dq_B)`` with ``dg/dq_B = q_A K/(K + u)^2``,
    ``u = q_A q_B``, while ``dF_B/dq_A = alpha gamma``, so

        (d omega)_{AB} = alpha(gamma - beta) - alpha s q_A K / (K + u)^2,

    which at ``beta = gamma`` is ``-alpha s q_A K/(K+u)^2``, nonzero wherever the supply
    runs and ``A`` is active: ``-0.1141566`` at the settled state. It does not involve
    ``f_S``, so no choice of the supply port's covector repairs it, and it varies with
    ``q_A`` and ``u``, so no choice of ``beta - gamma`` repairs it either (checked at two
    points at once). By the reciprocity lemma (§5.5) the forcing is therefore not ``dU``
    for any ``U``: the gated loop is an arrangement whose forcing is non-exact. That is
    inside the book's category, whose forcing need not be exact (STATUS.md §1), and
    outside the attention operad, whose forcing is the full differential of a nonnegative
    cost (notes/DESIGN.md §5). The same antisymmetric part appears in the *composite's*
    forcing on ``Q``, not only in the coupler's: the parts' own dissipation is exact and
    diagonal, so it contributes nothing to it.
    """
    _, upper, _ = gated_roots()
    settled = np.array([upper, 0.5 * upper, SUPPLY])

    def closed_form_AB(e, beta=BETA, gamma=GAMMA):
        q_A, q_B, s = float(e[0]), float(e[1]), float(e[2])
        u = q_A * q_B
        return ALPHA * (gamma - beta) - ALPHA * s * q_A * KGATE / (KGATE + u) ** 2

    rng = np.random.default_rng(12)
    pts = [settled] + [np.abs(rng.standard_normal(3)) + 0.25 for _ in range(4)]
    for f_S in (lambda e: 0.0, lambda e: ALPHA * e[0], lambda e: e[0] * e[1] + e[2] ** 2):
        dw = d_omega(F_loop(f_supply=f_S))
        for e in pts:
            np.testing.assert_allclose(dw(e)[0, 1], closed_form_AB(e), rtol=1e-10)
    np.testing.assert_allclose(closed_form_AB(settled), -0.1141566, atol=1e-7)
    assert abs(closed_form_AB(settled)) > 1e-3

    # no (beta, gamma) kills it at two points at once
    e1, e2 = settled, np.array([1.0, 1.0, SUPPLY])
    for gap in (-1.0, 0.0, ALPHA * SUPPLY * upper * KGATE / (KGATE + 0.5 * upper ** 2) ** 2):
        vals = [abs(closed_form_AB(e, beta=BETA, gamma=BETA + gap)) for e in (e1, e2)]
        assert max(vals) > 1e-3

    # the composite's own forcing on Q carries the same antisymmetric part
    arr = loop()
    J = jax.jacobian(lambda q: arr.omega(q, EMPTY, EMPTY)[0])(jnp.asarray(settled))
    dw_composite = np.asarray(J.T - J)
    np.testing.assert_allclose(dw_composite[0, 1], closed_form_AB(settled), rtol=1e-10)


# ---------------------------------------------------------------------------
# Test 6: the restart threshold.
# ---------------------------------------------------------------------------


def separatrix(ray, h=MOB, steps=STEPS, lo=0.0, hi=6.0, iters=40, **kw):
    """The smallest ``x`` on a ray ``x |-> (q_A, q_B)`` from which the loop still holds
    both forms after the run, by bisection."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        q_A, q_B = ray(mid)
        q = settle(loop(q_A=q_A, q_B=q_B, h=h, **kw), steps)
        lo, hi = (lo, mid) if held(q[0]) else (mid, hi)
    return 0.5 * (lo + hi)


def test_6_zero_activity_is_a_fixed_point_and_the_loop_has_a_restart_threshold():
    """REPRODUCES the closed form derived here (the saddle) and LOCATES numerically the
    separatrix through it.

    The forcing vanishes identically at ``q = 0``: the gate's derivative
    ``dg/dq_B = q_A K/(K+u)^2`` is ``0`` there, so the linearization at the origin is the
    unsupplied pair ``((-b, alpha beta), (alpha gamma, -b))``, eigenvalues
    ``-b +- alpha sqrt(beta gamma) = -0.5, -1.5``: the origin is a *stable* fixed point
    with vanishing gradient, and a pair started too low never gets going (STATUS.md §3.5
    records the same fixed point in the commons). The lower root ``q_A = 0.405890`` is a
    saddle -- the continuous field's Jacobian there has eigenvalues ``0.377459`` and
    ``-1.684537`` -- and its stable manifold is the threshold. So the threshold is a curve,
    not a number: on the ray ``q_B = c q_A``, which passes through the saddle itself, it is
    the saddle, ``0.405890``, and independent of the mobility; on the ray ``q_B = 0`` it is
    ``0.7436`` and moves with ``h`` (``0.74308`` at ``h = 0.002``, ``0.75845`` at
    ``h = 0.2``), first order in the step, as an explicit discretization must.
    """
    arr = loop()
    np.testing.assert_allclose(forcing(arr, [0.0, 0.0, SUPPLY])[:2], [0.0, 0.0], atol=0.0)

    def field(q):
        xi = arr.omega(jnp.array([q[0], q[1], SUPPLY]), EMPTY, EMPTY)[0]
        return xi[:2]

    eig_0 = np.linalg.eigvals(np.asarray(jax.jacobian(field)(jnp.array([0.0, 0.0]))))
    np.testing.assert_allclose(np.sort(eig_0.real), [-1.5, -0.5], atol=1e-12)
    assert np.all(eig_0.real < 0.0)                     # the origin is locally stable

    lower, upper, _ = gated_roots()
    eig_lo = np.linalg.eigvals(np.asarray(jax.jacobian(field)(jnp.array([lower, 0.5 * lower]))))
    np.testing.assert_allclose(np.sort(eig_lo.real), [-1.684537, 0.377459], atol=1e-6)
    assert np.max(eig_lo.real) > 0.0                    # a saddle: this is the threshold

    on_ray = separatrix(lambda x: (x, 0.5 * x))
    np.testing.assert_allclose(on_ray, lower, atol=1e-5)
    for h in (0.002, 0.2):
        scaled = separatrix(lambda x: (x, 0.5 * x), h=h, steps=int(STEPS * MOB / h))
        np.testing.assert_allclose(scaled, lower, atol=1e-4)

    off_ray = [separatrix(lambda x: (x, 0.0), h=h, steps=int(STEPS * MOB / h))
               for h in (0.002, MOB, 0.2)]
    np.testing.assert_allclose(off_ray, [0.743079, 0.743564, 0.758454], atol=1e-5)
    assert off_ray[0] < off_ray[1] < off_ray[2]         # the threshold moves with the step


# ---------------------------------------------------------------------------
# Test 7: which verdicts the parameters decide.
# ---------------------------------------------------------------------------


def test_7_three_verdicts_are_decided_by_the_parameters_and_the_wiring():
    """CORRECTS a claim made on paper: §4.3's "the reason is structural, not a matter of
    parameters" holds of the *linear* wiring's failure, but three of the gated loop's four
    verdicts are decided by choices made here, and the ranges are these.

    (a) Prediction 2 (cutting the supply ends both) holds iff the internal loop gain is
    below the dissipation, ``alpha^2 beta gamma < b^2``. At ``beta gamma = 0.25`` (chosen)
    both die; at ``beta gamma = 1 = b^2`` the pair holds ``q_A = q_B = 3.6956`` with no
    supply at all -- the linearization's eigenvalue ``-b + alpha sqrt(beta gamma)`` is
    ``0`` and the diagonal is conserved -- and above it the amplitudes grow without bound.
    Near the boundary the run length decides what a run sees, not the loop: at
    ``beta gamma = 0.9801`` the unsupplied pair decays at rate
    ``b - alpha sqrt(beta gamma) = 0.01`` and is still at ``0.50009`` after a run's 200
    units of time, so at ``HELD = 1e-3`` that run reports both forms held.
    (b) Predictions 3 and 4 hold only for a *fully* gated supply. An ungated fraction
    ``leak`` leaves ``A`` at ``alpha leak s / b`` after the ``B -> A`` cut, nonzero for
    every ``leak > 0``: strictly, the two predictions fail for any leak at all, and under
    the run's ``HELD = 1e-3`` reading they fail once ``leak > 2.5e-4``.
    (c) Prediction 4 is decided by which activities the gate reads. Gating on the product
    ``q_A q_B`` (chosen) or on the partner alone ``q_B^2`` kills both on a cut; gating on
    ``A``'s own activity ``q_A^2`` -- autocatalysis without mutuality -- leaves ``A``
    at ``3.7321 = 2 + sqrt(3)`` after the cut, so predictions 3 and 4 fail. What the
    predictions need is that every path from the supply to ``A`` pass through ``B``'s
    signal.
    (d) Prediction 1 holds only above the supply threshold ``s_*`` of test 3, and the run's
    ``s = 4`` was chosen against ``s_* = 2.1213``: at ``s = 2.0`` nothing is held from any
    start. Also the reading of the ``B -> A`` cut decides prediction 3: cutting the
    ``beta`` term alone and leaving the gate reading ``q_B`` leaves the pair at
    ``2 + sqrt(2) = 3.414214``, both held.
    """
    _, upper, s_star = gated_roots()
    start = dict(q_A=upper, q_B=0.5 * upper)

    # (a) the internal loop gain against the dissipation
    for beta, gamma, dies in ((0.5, 0.5, True), (0.8, 0.8, True), (1.0, 1.0, False),
                              (1.05, 1.05, False)):
        q = settle(loop(**start, supply_on=False, beta=beta, gamma=gamma))
        assert (not held(q[0])) == dies, (beta, gamma, q)
    marginal = settle(loop(**start, supply_on=False, beta=1.0, gamma=1.0))
    np.testing.assert_allclose(marginal[:2], [3.695583, 3.695583], atol=1e-4)
    slow = settle(loop(**start, supply_on=False, beta=0.99, gamma=0.99))
    np.testing.assert_allclose(slow[:2], [0.5000927, 0.5000927], atol=1e-6)
    assert held(slow[0])            # the run's verdict, not the loop's: it is still decaying

    # (b) an ungated leak
    for leak in (0.0, 1e-4, 1e-3, 1e-2, 0.1):
        q = settle(loop(**start, link_ba=False, leak=leak))
        np.testing.assert_allclose(q[0], ALPHA * leak * SUPPLY / DISS, atol=1e-9)
        assert held(q[0]) == (leak > HELD * DISS / (ALPHA * SUPPLY))

    # (c) which activities the gate reads
    gates = {"product": michaelis,
             "partner": lambda q_A, q_B, K=KGATE: (q_B * q_B) / (K + q_B * q_B),
             "own": lambda q_A, q_B, K=KGATE: (q_A * q_A) / (K + q_A * q_A)}
    survives = {}
    for name, g in gates.items():
        full = settle(loop(gate=g))
        assert held(full[0]) and held(full[1])
        q = settle(loop(q_A=full[0], q_B=full[1], gate=g, link_ba=False))
        survives[name] = held(q[0])
        if name == "own":
            np.testing.assert_allclose(q[0], 2.0 + np.sqrt(3.0), atol=1e-6)
    assert survives == {"product": False, "partner": False, "own": True}

    # (d) the supply threshold, and the other reading of the cut
    assert np.isnan(gated_roots(s=2.0)[1]) and 2.0 < s_star
    assert not held(settle(loop(s=2.0, q_A=5.0, q_B=2.5))[0])
    assert held(settle(loop(s=2.2, q_A=5.0, q_B=2.5))[0])
    partial = settle(loop(**start, beta_only_cut=True))
    np.testing.assert_allclose(partial[:2], [2.0 + np.sqrt(2.0), 1.0 + np.sqrt(2.0) / 2.0],
                               atol=1e-6)
    assert held(partial[0]) and held(partial[1])
