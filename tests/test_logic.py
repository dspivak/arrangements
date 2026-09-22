"""Tests for ``logic`` -- CMOS from non-reciprocal MOSFETs (EXTENSION, not in the paper).

The claims under test:

1.  *Reciprocity is forced instantaneously.*  At a fixed parameter position the
    port covectors are ``omega`` on the input interface, so for an *exact*
    forcing the port Jacobian is ``-d^2 U`` and is symmetric -- for every such
    arrangement, transistor or not.
2.  *Non-reciprocity comes from the sharp* (the TRICK realization).  Once the
    state is adiabatically eliminated the same device has ``dI_G/de_D = 0``
    exactly (the sharp's kernel) while ``dI_D/de_G`` is the transconductance.
3.  *Both survive composition.*  Parameters tensor and forcings pull back and
    add, and the devices have no output ports, so ``omega_tot|_{Q_j} =
    omega_j|_{Q_j}``; a device whose gate-source pair is pinned does not move
    when its drain swings.
4.  The composite is a NAND, and two of them cross-coupled are a bistable latch.
5.  *Trick vs direct.*  CDLM had only a potential, so it put the drain-sensitive
    component of ``dU`` in ``ker(sharpR)``.  The book's arrangement carries a
    forcing that need not be exact, so ``mosfet_direct`` writes ``omega|_Q`` by
    hand on the plain Euclidean sharp -- a genuinely non-exact one-form.  The
    two realizations agree step for step, and it is the direct one whose
    ``d omega != 0`` exhibits the non-reciprocity without any adiabatic limit.

Sign (DESIGN.md §3): the framework's port covector is now minus the current, so
the ``load`` the old tests presented as a cost gradient is negated here; states
are unchanged.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arrangements.functors import Phi
from arrangements.interpretation import trivial_omega
from arrangements.logic import (
    KERNEL_SHARP,
    chain_state,
    flat_of,
    inverter,
    inverter_chain,
    inverter_state,
    latch_state,
    mosfet,
    mosfet_direct,
    nand_gate,
    nand_state,
    rail,
    run,
    sr_latch,
    state_of,
    stepper,
)
from arrangements.pcK import PhiK
from arrangements.rk4 import rk4
from arrangements.wiring import tensor_arrangements

from tests.chain import load_regress

_IN_POS = (jnp.zeros(0), trivial_omega(0))
HI, LO = 1.0, 0.0


def _settle_fet(fet, e, n=300):
    """Run one device's ``Phi`` to its fixed point at terminal voltages ``e``."""
    O, q = Phi(fet), jnp.zeros(2)
    for _ in range(n):
        _, fiber = O.step(q)
        _, at_pos = fiber(_IN_POS)
        _, q = at_pos((jnp.zeros(0), e))
    return q


def _fet_currents(fet, q, e):
    """The device's port covector ``omega_N(e)`` at parameter position ``q`` (minus the currents)."""
    _, fiber = Phi(fet).step(q)
    (_, omega_N), _ = fiber(_IN_POS)
    return omega_N(e)


# ---------------------------------------------------------------------------
# 1-2.  The device.
# ---------------------------------------------------------------------------


def test_sharp_is_degenerate_and_nonsymmetric():
    """CDLM def.rvect asks only for ``Q -> vect(Q^*, Q)``: no symmetry, no invertibility."""
    S = KERNEL_SHARP
    assert not np.allclose(S, S.T)
    assert np.linalg.matrix_rank(np.asarray(S)) == 1


def test_gate_state_tracks_vgs_independently_of_drain():
    """The sharp reads only the ``q1``-component, where ``vds`` does not appear."""
    fet = mosfet("n")
    qs = [_settle_fet(fet, jnp.array([0.8, vd, 0.0])) for vd in (0.2, 0.5, 0.9, 2.0)]
    for q in qs:
        np.testing.assert_allclose(q[0], 0.8, atol=1e-5)  # q0 -> vgs
        np.testing.assert_allclose(q[1], 0.0, atol=1e-12)  # q1 frozen
    for q in qs[1:]:
        np.testing.assert_allclose(q, qs[0], atol=1e-6)  # identical, not merely close


def test_device_currents_sum_to_zero():
    """``U`` sees only voltage differences, so ``dU/deG + dU/deD + dU/deS = 0``."""
    fet = mosfet("n")
    e = jnp.array([0.8, 0.5, 0.1])
    I = _fet_currents(fet, _settle_fet(fet, e), e)
    np.testing.assert_allclose(float(I.sum()), 0.0, atol=1e-9)


def test_instantaneous_port_jacobian_is_symmetric():
    """Claim 1: at fixed ``q`` the port Jacobian is ``-d^2 U``, hence reciprocal."""
    fet = mosfet("n")
    e = jnp.array([0.8, 0.5, 0.0])
    q = _settle_fet(fet, e)
    J = jax.jacobian(lambda ee: _fet_currents(fet, q, ee))(e)
    np.testing.assert_allclose(J, J.T, atol=1e-6)


def test_steady_state_port_jacobian_is_not():
    """Claim 2: adiabatic elimination breaks it, and ``dI_G/de_D`` is exactly 0."""
    fet = mosfet("n")
    e = jnp.array([0.8, 0.5, 0.0])
    J = jax.jacobian(lambda ee: _fet_currents(fet, _settle_fet(fet, ee), ee))(e)
    assert abs(float(J[1, 0])) > 1.0            # transconductance dI_D/de_G
    np.testing.assert_allclose(J[0, 1], 0.0, atol=1e-9)   # dI_G/de_D, the kernel
    assert float(np.abs(J - J.T).max()) > 1.0


def test_rail_is_frozen():
    """An ideal source is a state with ``sharpR = 0``: its coalgebra step moves nothing."""
    P = rail().P
    for xi in (jnp.array([1.0]), jnp.array([-3.5])):
        np.testing.assert_allclose(P.step(jnp.array([0.7]), xi), jnp.array([0.7]))


# ---------------------------------------------------------------------------
# 3-4.  The composites.
# ---------------------------------------------------------------------------


def test_nand_is_built_by_composition():
    """Seven boxes, no hand-written composite forcing."""
    g = nand_gate()
    assert g.P.Q_dim == 4 * 2 + 3          # four devices (R^2), three reporters (R^1)
    assert (g.out_dim_M, g.in_dim_M) == (0, 0)
    assert (g.in_dim_N, g.out_dim_N) == (2, 1)


@pytest.mark.parametrize(
    "a,b,expected", [(LO, LO, HI), (LO, HI, HI), (HI, LO, HI), (HI, HI, LO)]
)
def test_nand_truth_table(a, b, expected):
    _, outs, currents = run(nand_gate(), nand_state(0.5), lambda t: (a, b), 600)
    np.testing.assert_allclose(float(outs[-1][0]), expected, atol=1e-3)
    assert float(jnp.abs(currents[-1]).max()) < 1e-4   # no DC input current


def test_input_current_is_purely_capacitive():
    """The residual reciprocal coupling is the gate capacitance ``kappa`` and nothing else."""
    g = nand_gate()
    st, _, _ = run(g, nand_state(1.0), lambda t: (LO, HI), 400)
    _, _, currents = run(g, st, lambda t: (HI, HI), 200)
    assert float(jnp.abs(currents[0][0])) > 0.1    # a transient on the transition
    assert float(jnp.abs(currents[-1][0])) < 1e-3  # gone once the gates settle


def test_drain_swing_does_not_move_a_pinned_gate_state():
    """Claim 3: N2's gate-source is pinned to (b, ground); its drain V_mid is not.

    The loads are the book's covectors, minus CDLM's currents (DESIGN.md §3).
    """
    g, states = nand_gate(), []
    for load in (0.0, -0.5, -2.0):
        st, outs, _ = run(g, nand_state(0.5), lambda t: (HI, HI), 400, load=[load])
        states.append(st)
    assert abs(float(states[0][8]) - float(states[2][8])) > 0.3   # V_out really swung
    for st in states[1:]:
        np.testing.assert_allclose(st[6], states[0][6], atol=1e-9)  # N2 gate state
        np.testing.assert_allclose(st[0], states[0][0], atol=1e-9)  # P1 gate state
    np.testing.assert_allclose(jnp.abs(states[-1][1:8:2]).max(), 0.0, atol=1e-12)


def test_latch_holds_both_states():
    L = sr_latch()
    for q, qbar in ((0.9, 0.1), (0.1, 0.9)):
        _, outs, _ = run(L, latch_state(q, qbar), lambda t: (HI, HI), 1500)
        np.testing.assert_allclose(float(outs[-1][0]), round(q), atol=1e-3)
        np.testing.assert_allclose(float(outs[-1][1]), round(qbar), atol=1e-3)


def test_latch_set_reset_and_remember():
    L, st = sr_latch(), latch_state(0.5, 0.5)
    seen = []
    for sbar, rbar in [(LO, HI), (HI, HI), (HI, LO), (HI, HI), (LO, HI), (HI, HI)]:
        st, outs, _ = run(L, st, lambda t: (sbar, rbar), 500)
        seen.append(round(float(outs[-1][0])))
    assert seen == [1, 1, 0, 0, 1, 1]   # set, hold, reset, hold, set, hold


def test_latch_metastable_point_is_unstable():
    """The symmetric fixed point is exact; either sign of perturbation resolves it."""
    L = sr_latch()
    outs = {}
    for eps in (0.0, 1e-4, -1e-4):
        _, o, _ = run(L, latch_state(0.5 + eps, 0.5), lambda t: (HI, HI), 3000)
        outs[eps] = float(o[-1][0])
    np.testing.assert_allclose(outs[0.0], 0.5, atol=1e-3)
    np.testing.assert_allclose(outs[1e-4], 1.0, atol=1e-3)
    np.testing.assert_allclose(outs[-1e-4], 0.0, atol=1e-3)


# ---------------------------------------------------------------------------
# 5.  What the tick budget actually scales with.
# ---------------------------------------------------------------------------


def _settle_ticks(arr, st, n_in, tol=0.01, maxt=4000):
    one, xi = stepper(arr), jnp.zeros(arr.out_dim_N)
    traj = []
    for _ in range(maxt):
        out, _, st = one(st, jnp.asarray(n_in, float), xi)
        traj.append(np.asarray(out))
    fin = traj[-1]
    return next(t for t in range(len(traj)) if np.abs(traj[t] - fin).max() < tol), fin


def test_ticks_scale_with_logic_depth_not_component_count():
    """Depth costs ticks; width does not.

    A chain of ``K`` inverters settles in time affine in ``K`` (a signal really
    does have to cross ``K`` stages), while ``K`` *independent* inverters settle
    in the same number of ticks no matter how many there are.  So a wide circuit
    is free and a deep one costs its depth -- the physical delay, not a
    numerical penalty.
    """
    depths = [1, 2, 4, 6]
    ticks = [_settle_ticks(inverter_chain(K), chain_state(K), [HI])[0] for K in depths]
    assert ticks == sorted(ticks)
    per_stage = [(ticks[i] - ticks[0]) / (depths[i] - depths[0]) for i in range(1, 4)]
    assert max(per_stage) - min(per_stage) < 3.0        # affine in depth
    assert ticks[-1] < 4 * ticks[0]                     # and cheaply so

    wide = [
        _settle_ticks(
            tensor_arrangements([inverter() for _ in range(K)]),
            jnp.concatenate([inverter_state(0.5) for _ in range(K)]),
            [HI] * K,
        )[0]
        for K in (1, 4, 16)
    ]
    assert len(set(wide)) == 1                          # width is free


def test_rk4_buys_1_4x_the_step_for_4x_the_work():
    """Explicit multistage does not pay for itself on a step-limited circuit.

    Fixing the circuit's *rate ratio* and sweeping a single timestep ``h``, the
    four-stage ``rk4`` parameter (``pc^(4)``, four gradients per macro-tick; the
    integrator is part of the arrangement, def:arr) is stable where Euler is not
    -- but only by the classical linear-stability factor ``2.785/2 = 1.39``, so
    its net throughput is about ``0.35``.  If the step is the binding constraint
    the answer is a better *sharp* (preconditioning, cf.
    ``rvect.inverse_hessian``) or an implicit stage loop, not more explicit stages.
    """
    KAPPA0, C0 = 0.5, 0.02
    triv = lambda _o: (jnp.zeros(1), jnp.array([HI]))

    def euler_ok(h, T=800):
        one = stepper(inverter(c=C0 * h, kappa=KAPPA0 * h))
        s = inverter_state(0.5)
        for _ in range(T):
            _, _, s = one(s, jnp.array([HI]), jnp.zeros(1))
            if not jnp.isfinite(s).all() or float(jnp.abs(s).max()) > 20:
                return False
        return abs(float(s[4])) < 0.02

    def rk4_ok(h, T=800):
        arr = inverter(c=C0, kappa=KAPPA0, parameter=lambda Q: rk4(Q, h))
        O = PhiK(arr)
        s = inverter_state(0.5)
        for _ in range(T):
            s = flat_of(O.with_state(state_of(arr, s)).run_one(_IN_POS, triv)[-1])
            if not jnp.isfinite(s).all() or float(jnp.abs(s).max()) > 20:
                return False
        return abs(float(s[4])) < 0.02

    assert euler_ok(3.5) and not euler_ok(4.5)      # Euler's limit is near h = 4.0
    assert rk4_ok(5.0)                              # RK4 is stable past it ...
    assert not rk4_ok(6.0)                          # ... but only by ~1.39x,
    # so 4 gradients per tick buy well under 2x the step: net throughput < 1.


def test_frozen_modes_are_exactly_frozen():
    """Three unit eigenvalues of the one-tick map: two gate ``q1``'s and the rail."""
    one = stepper(inverter(c=0.02))
    upd = jax.jit(lambda s: one(s, jnp.array([HI]), jnp.zeros(1))[2])
    st = inverter_state(0.5)
    for _ in range(2000):
        st = upd(st)
    ev = np.sort(np.abs(np.linalg.eigvals(np.asarray(jax.jacobian(upd)(st)))))[::-1]
    np.testing.assert_allclose(ev[:3], 1.0, atol=1e-6)   # sharp kernel + sharp = 0
    assert ev[3] < 0.99                                  # everything else contracts


def test_explicit_euler_limit_cycles_when_the_node_step_is_too_big():
    """The integrator is the binding constraint: it degrades to a cycle, it does not diverge.

    The channel's ``tanh`` bounds the current, so an over-large node step
    ``c = dt/C`` does not blow up -- it settles onto a period-4 cycle at a wrong
    logic level.  A silent wrong answer, not a NaN.
    """
    one = stepper(nand_gate(c=0.10))
    st, n_in, xi = nand_state(0.5), jnp.array([HI, HI]), jnp.zeros(1)
    for _ in range(2000):
        out, _, st = one(st, n_in, xi)
    tail = []
    for _ in range(8):
        out, _, st = one(st, n_in, xi)
        tail.append(float(out[0]))
    assert max(tail) - min(tail) > 0.1                      # cycling, not settled
    np.testing.assert_allclose(tail[:4], tail[4:], atol=1e-4)  # with period 4
    assert min(tail) > 0.01                                 # and at the wrong level


# ---------------------------------------------------------------------------
# 6.  The direct (non-exact) realization of the same device.
# ---------------------------------------------------------------------------


def test_direct_forcing_is_not_exact():
    """``d omega != 0`` for ``mosfet_direct``: the mixed partials disagree.

    On ``Q x N^-`` the one-form's ``q0``-component does not see ``e_D`` at all,
    while its ``e_D``-component sees ``q0`` through the channel model -- exactly
    the non-reciprocity, now visible in the forcing itself with no adiabatic
    limit and no degenerate sharp.  The trick device's forcing, being ``-dU``,
    has symmetric mixed partials at every point.
    """
    q, e = jnp.array([0.8, 0.0]), jnp.array([0.8, 0.5, 0.0])

    def mixed(arr):
        d_qn = jax.jacobian(lambda ee: arr.omega(q, jnp.zeros(0), ee)[0])(e)  # d omega_Q / d e
        d_nq = jax.jacobian(lambda qq: arr.omega(qq, jnp.zeros(0), e)[2])(q)  # d omega_N / d q
        return d_qn, d_nq

    d_qn, d_nq = mixed(mosfet_direct("n"))
    np.testing.assert_allclose(d_qn[0, 1], 0.0, atol=1e-12)   # no drain in omega|_Q
    assert abs(float(d_nq[1, 0])) > 1.0                       # but drain current sees q0
    assert float(np.abs(d_qn - d_nq.T).max()) > 1.0           # so d omega != 0

    d_qn, d_nq = mixed(mosfet("n"))
    np.testing.assert_allclose(d_qn, d_nq.T, atol=1e-8)       # the trick's is exact


def test_trick_and_direct_agree_step_for_step():
    """The two realizations of the device are the same dynamics (rtol 1e-12).

    ``KERNEL_SHARP @ (-dU)|_Q`` (trick) equals ``I @ omega|_Q`` (direct) at every
    state, so device, inverter, NAND and latch all run identically.
    """
    for fet, fet_d in ((mosfet("n"), mosfet_direct("n")), (mosfet("p"), mosfet_direct("p"))):
        for e in (jnp.array([0.8, 0.5, 0.0]), jnp.array([0.2, 0.9, 0.1])):
            for n in (1, 7, 300):
                np.testing.assert_allclose(
                    np.asarray(_settle_fet(fet_d, e, n)), np.asarray(_settle_fet(fet, e, n)),
                    rtol=1e-12, atol=1e-14,
                )
            q = _settle_fet(fet, e, 50)
            np.testing.assert_allclose(
                np.asarray(_fet_currents(fet_d, q, e)), np.asarray(_fet_currents(fet, q, e)),
                rtol=1e-12, atol=1e-14,
            )

    for build in (lambda **kw: inverter(**kw), lambda **kw: nand_gate(**kw)):
        trick, direct = build(), build(device=mosfet_direct)
        st_t, st_d = jnp.zeros(trick.P.Q_dim), jnp.zeros(direct.P.Q_dim)
        st_t = nand_state(0.5) if trick.P.Q_dim == 11 else inverter_state(0.5)
        st_d = st_t
        n_in = jnp.array([HI, HI]) if trick.in_dim_N == 2 else jnp.array([HI])
        one_t, one_d = stepper(trick), stepper(direct)
        xi = jnp.zeros(trick.out_dim_N)
        for _ in range(200):
            o_t, c_t, st_t = one_t(st_t, n_in, xi)
            o_d, c_d, st_d = one_d(st_d, n_in, xi)
            np.testing.assert_allclose(np.asarray(st_d), np.asarray(st_t), rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(np.asarray(o_d), np.asarray(o_t), rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(np.asarray(c_d), np.asarray(c_t), rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# 7.  Regression against CDLM.
# ---------------------------------------------------------------------------


def test_regression_matches_cdlm():
    """The CDLM runs (``regress/logic.npz``) are reproduced to rtol 1e-10 (DESIGN.md §3).

    Covers both dynamics the old tests run: the configuration parameter
    (device, NAND, drain swing, latch, inverter chain, one plain tick) and the
    four-stage ``rk4`` parameter (``PhiK``). Loads and port covectors are
    negated on the way in and out; states are compared unchanged.
    """
    R = load_regress("logic")
    fet = mosfet("n")

    # 1. device
    q = np.stack([np.asarray(_settle_fet(fet, jnp.array([0.8, float(vd), 0.0]))) for vd in R["fet_vds"]])
    np.testing.assert_allclose(q, R["fet_q"], rtol=1e-10, atol=1e-12)
    e = jnp.asarray(R["fet_e"])
    cur = np.asarray(_fet_currents(fet, _settle_fet(fet, e), e))
    np.testing.assert_allclose(-cur, R["fet_currents"], rtol=1e-10, atol=1e-12)

    # 2. NAND truth table
    steps = int(R["nand_steps"])
    for k, (a, b) in enumerate(R["nand_inputs"]):
        st, outs, currents = run(nand_gate(), jnp.asarray(R["nand_state0"]),
                                 lambda t: (float(a), float(b)), steps)
        np.testing.assert_allclose(np.asarray(st), R["nand_final"][k], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(np.asarray(outs[-1]), R["nand_out"][k], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(-np.asarray(currents[-1]), R["nand_current"][k], rtol=1e-10, atol=1e-12)

    # 3. drain swing (loads negated)
    for k, load in enumerate(R["swing_loads"]):
        st, _, _ = run(nand_gate(), jnp.asarray(R["nand_state0"]), lambda t: (HI, HI),
                       int(R["swing_steps"]), load=[-float(load)])
        np.testing.assert_allclose(np.asarray(st), R["swing_final"][k], rtol=1e-10, atol=1e-12)

    # 4. SR latch sequence
    L, st = sr_latch(), jnp.asarray(R["latch_state0"])
    for k, (sbar, rbar) in enumerate(R["latch_seq"]):
        st, outs, _ = run(L, st, lambda t: (float(sbar), float(rbar)), int(R["latch_steps"]))
        np.testing.assert_allclose(np.asarray(outs[-1]), R["latch_outs"][k], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.asarray(st), R["latch_final"], rtol=1e-10, atol=1e-12)

    # 5. inverter chain
    st, outs, _ = run(inverter_chain(int(R["chain_K"])), jnp.asarray(R["chain_state0"]),
                      lambda t: (HI,), int(R["chain_steps"]))
    np.testing.assert_allclose(np.asarray(st), R["chain_final"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.stack([np.asarray(o) for o in outs[::50]]), R["chain_outs"],
                               rtol=1e-10, atol=1e-12)

    # 6. the four-stage rk4 parameter
    h = float(R["rk4_h"])
    arr = inverter(c=0.02, kappa=0.5, parameter=lambda Q: rk4(Q, h))
    O = PhiK(arr)
    triv = lambda _o: (jnp.zeros(1), jnp.array([HI]))
    s = jnp.asarray(R["rk4_state0"])
    traj = [np.asarray(s)]
    for _ in range(int(R["rk4_steps"])):
        s = flat_of(O.with_state(state_of(arr, s)).run_one(_IN_POS, triv)[-1])
        traj.append(np.asarray(s))
    np.testing.assert_allclose(np.stack(traj), R["rk4_traj"], rtol=1e-10, atol=1e-12)

    # 7. one plain tick, with its port covector
    out, cur, st = stepper(inverter(c=0.02))(jnp.asarray(R["rk4_state0"]), jnp.array([HI]), jnp.zeros(1))
    np.testing.assert_allclose(np.asarray(out), R["tick_out"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(-np.asarray(cur), R["tick_current"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.asarray(st), R["tick_state"], rtol=1e-10, atol=1e-12)
