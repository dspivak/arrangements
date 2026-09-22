"""The LC circuit: one lens datum, two parameters; the electrical reading acquires magnetism.

The module (dap/lc_circuit.py) stores the capacitor charge ``q`` with reactive
sharp ``xi |-> xi/L`` (inductance in the mass slot) and forcing ``omega = -dU``
for the capacitor energy ``U(q) = q^2/(2C)``. The claims under audit:

* ``phase``: the conjugate momentum is the flux linkage ``lambda = L dq/dt``,
  the kinetic term ``xi^2/(2L)`` is the inductor's magnetic energy, and the
  circuit oscillates at ``omega = 1/sqrt(LC)``.
* ``configuration``: the same lens datum relaxes RC-style with no oscillation;
  the number in the sharp then functions as a resistance ``R := L`` (mobility
  ``1/R``), not an inductance. Two arrangements, one datum (def:arr): the
  integrator is the parameter, and ``Phi`` takes no integrator.
* Coupled pair: two capacitor boxes wired through a stateless coupling box
  exhibit the two normal modes ``omega_+ = 1/sqrt(LC)`` and
  ``omega_- = sqrt(1/(LC) + 2/(L C_c))``.
* Trick vs direct (DESIGN.md §6(c)): the two-box LC rotation, realized by
  CDLM's oppositely-signed sharps with an exact forcing (``lc_conf``) and
  directly by the non-exact coupler one-form ``v di - i dv`` with both sharps
  positive (``lc_conf_direct``); the two agree step for step.

Discrete relations, derived from the code (not from the continuum):

* ``phase`` (coalgebra.py) steps ``(q, xi) |-> (q + sharp(xi), xi + xi_Q)``
  with ``xi_Q = omega|_Q = -dU`` at the presented position ``q~ = q + sharp(xi)``.
  For the LC cell: ``q_{n+1} = q_n + xi_n/L`` and ``xi_{n+1} = xi_n - q_{n+1}/C``
  -- symplectic (semi-implicit) Euler at tick ``dt = 1``. Hence EXACTLY
  ``xi_n = L (q_{n+1} - q_n)/dt``, the flux linkage (test 2), and eliminating
  ``xi``: ``L (q_{n+1} - 2 q_n + q_{n-1}) = -q_n/C``, a linear two-step
  recurrence whose rotation angle per tick is ``theta = 2 arcsin(omega/2)``,
  ``omega := 1/sqrt(LC)``. So the discrete period is ``2 pi / theta`` ticks:
  shorter than the continuum ``2 pi sqrt(LC)`` by the relative frequency bias
  ``omega^2/24 + O(omega^4)`` -- an O(dt^2) error (the map's trace ``2 - omega^2``
  coincides with velocity Verlet's, so leapfrog has the SAME period; it differs
  only in the energy band, test 3).
* ``leapfrog`` (leapfrog.py) stores integer-tick ``(q, xi)`` with half-kicks;
  eliminating the half-step momenta gives the exact centered relation
  ``xi_n = L (q_{n+1} - q_{n-1})/2`` (test 2).
* ``configuration`` steps ``q |-> q + sharp(omega|_Q) = (1 - 1/(LC)) q``: exact
  geometric decay, forward Euler for ``dq/dt = -q/(LC)`` (test 4).

There is no ``dt`` in the framework (the tick is unit time), so "small dt"
means ``omega = 1/sqrt(LC)`` small in tick units; physical ``(L, C)`` at step
``dt`` correspond to ``(L/dt, C/dt)`` here, and then ``xi`` is the physical
flux linkage.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.coalgebra import configuration, phase
from arrangements.functors import Phi
from arrangements.leapfrog import leapfrog
from arrangements.lc_circuit import (
    capacitor_cell,
    coupled_lc_pair,
    coupling_capacitor,
    current_voltage_coupler_direct,
    inductor_cell,
    lc_cell,
    lc_conf,
    lc_conf_direct,
)
from arrangements.pc2 import Phi2

from tests.chain import IN_POS, load_regress

# Unloaded output port: the environment exerts the zero covector on the cell's
# single terminal (and there are no input ports). The pair is fully closed.
_DIR_CELL = lambda _out_pos: (jnp.zeros(1), jnp.zeros(0))
_DIR_PAIR = lambda _out_pos: (jnp.zeros(0), jnp.zeros(0))
# The two-box composite has one outer output port (the terminal voltage), unloaded.
_DIR_CONF = lambda _out_pos: (jnp.zeros(1), jnp.zeros(0))


def _run(O, q0, xi0, steps, dir_from):
    """Iterate a 1-stage pc morphism from phase state ``(q0, xi0)``; return q, xi paths."""
    state = (jnp.asarray(q0, float), jnp.asarray(xi0, float))
    qs, xis = [state[0]], [state[1]]
    for _ in range(steps):
        _, _, state = O.with_state(state).run_one(IN_POS, dir_from)
        qs.append(state[0])
        xis.append(state[1])
    return np.stack([np.asarray(x) for x in qs]), np.stack([np.asarray(x) for x in xis])


def _run2(O2, q0, xi0, steps, dir_from):
    """Same, for a 2-stage (leapfrog) pc^(2) morphism."""
    state = (jnp.asarray(q0, float), jnp.asarray(xi0, float))
    qs, xis = [state[0]], [state[1]]
    for _ in range(steps):
        _, _, state = O2.with_state(state).run_one(IN_POS, dir_from)
        qs.append(state[0])
        xis.append(state[1])
    return np.stack([np.asarray(x) for x in qs]), np.stack([np.asarray(x) for x in xis])


# The composite's parameter is ``(P_1 (x) P_2) (x) P_wire``, so its state is
# ``((s_1, s_2), s_wire)`` with ``s_wire = R^0``; these convert to flat arrays.


def _pair_state(q0, xi0):
    q0, xi0 = jnp.asarray(q0, float), jnp.asarray(xi0, float)
    return (((q0[0:1], xi0[0:1]), (q0[1:2], xi0[1:2])), jnp.zeros(0))


def _pair_flat(state):
    (s1, s2), _ = state
    return jnp.concatenate([s1[0], s2[0]]), jnp.concatenate([s1[1], s2[1]])


def _run_pair(O, q0, xi0, steps, dir_from):
    state = _pair_state(q0, xi0)
    q, xi = _pair_flat(state)
    qs, xis = [q], [xi]
    for _ in range(steps):
        _, _, state = O.with_state(state).run_one(IN_POS, dir_from)
        q, xi = _pair_flat(state)
        qs.append(q)
        xis.append(xi)
    return np.stack([np.asarray(x) for x in qs]), np.stack([np.asarray(x) for x in xis])


def _conf_state(iv):
    iv = jnp.asarray(iv, float)
    return ((iv[0:1], iv[1:2]), jnp.zeros(0))


def _conf_flat(state):
    (i, v), _ = state
    return jnp.concatenate([i, v])


def _measured_period(q):
    """Average tick-count between upward zero crossings (linear interpolation).

    The trajectory is an exact discrete sinusoid (linear symplectic map), so
    interpolation error in each crossing is O(theta^2) ticks and averaging over
    several periods leaves a relative error ~1e-7 at omega = 0.05 (measured).
    """
    ts = [n + q[n] / (q[n] - q[n + 1]) for n in range(len(q) - 1) if q[n] < 0.0 <= q[n + 1]]
    assert len(ts) >= 3, "trajectory too short to measure a period"
    return (ts[-1] - ts[0]) / (len(ts) - 1)


# ---------------------------------------------------------------------------
# 1. Frequency: omega = 1/sqrt(LC), three (L, C) pairs, LC spanning 400..10^4.
# ---------------------------------------------------------------------------


def test_lc_frequency_is_one_over_sqrt_LC():
    """Measured period = 2 pi sqrt(LC) within the integrator's known O(dt^2) bias.

    Tolerance justification: the discrete rotation angle is
    ``theta = 2 arcsin(omega/2)``, so the relative period deviation is
    ``omega/theta - 1 = -omega^2/24 + O(omega^4)``. We assert (a) agreement
    with the continuum within ``omega^2/12`` (twice the known bias -- NOT a
    loose tolerance: (c) asserts the bias is actually resolved), (b) agreement
    with the exact discrete prediction ``2 pi/theta`` to 1e-6 (measurement
    error only), (c) the deviation from the continuum has the predicted sign
    and at least half the predicted size, so (a)'s margin is doing no work.
    """
    for L, C in [(20.0, 20.0), (80.0, 20.0), (100.0, 100.0)]:
        w = 1.0 / np.sqrt(L * C)                     # omega * dt, dt = 1 tick
        T_cont = 2.0 * np.pi / w                     # 2 pi sqrt(LC)
        T_disc = 2.0 * np.pi / (2.0 * np.arcsin(w / 2.0))
        steps = int(5.2 * T_cont)
        qs, _ = _run(Phi(lc_cell(L, C, phase)), [1.0], [0.0], steps, _DIR_CELL)
        T_meas = _measured_period(qs[:, 0])

        assert abs(T_meas - T_cont) / T_cont <= w * w / 12.0            # (a)
        assert abs(T_meas - T_disc) / T_disc <= 1e-6                    # (b)
        assert T_cont - T_meas >= (w * w / 48.0) * T_cont               # (c)


def test_leapfrog_has_the_same_period():
    """The leapfrog arrangement measures the SAME discrete period (equal trace
    2 - omega^2), so using ``phase`` above loses no frequency accuracy --
    leapfrog only tightens the energy band (test 3)."""
    L, C = 20.0, 20.0
    w = 1.0 / np.sqrt(L * C)
    T_disc = 2.0 * np.pi / (2.0 * np.arcsin(w / 2.0))
    qs, _ = _run2(Phi2(lc_cell(L, C, leapfrog)), [1.0], [0.0], int(5.2 * 2 * np.pi / w), _DIR_CELL)
    assert abs(_measured_period(qs[:, 0]) - T_disc) / T_disc <= 1e-6


# ---------------------------------------------------------------------------
# 2. Keystone: the stored momentum IS the flux linkage lambda = L dq/dt.
# ---------------------------------------------------------------------------


def test_momentum_is_flux_linkage():
    """``phase``: ``xi_n = L (q_{n+1} - q_n)/dt`` EXACTLY (dt = 1 tick).

    Read off ``coalgebra.phase``: ``q_{n+1} = q_n + sharp(xi_n) = q_n + xi_n/L``
    before any force acts, so the identity is exact in floating point (~1e-14),
    not an approximation: the momentum slot stores inductance times current,
    i.e. the flux linkage conjugate to charge.
    """
    L, C = 20.0, 20.0
    qs, xis = _run(Phi(lc_cell(L, C, phase)), [1.0], [0.0], 300, _DIR_CELL)
    np.testing.assert_allclose(xis[:-1, 0], L * (qs[1:, 0] - qs[:-1, 0]), atol=1e-12)


def test_leapfrog_momentum_is_centered_flux_linkage():
    """Leapfrog: the integer-tick momentum is the CENTERED difference,
    ``xi_n = L (q_{n+1} - q_{n-1})/2``, exactly.

    Derivation from leapfrog.py: ``xi_{n+1/2} = xi_n + (1/2) xi_Q(q_n)`` and
    ``xi_{n+1} = xi_{n+1/2} + (1/2) xi_Q(q_{n+1}) = xi_{n+3/2} + ... `` give
    ``xi_{n+1} = (xi_{n+1/2} + xi_{n+3/2})/2`` with ``q_{n+1} - q_n = xi_{n+1/2}/L``.
    """
    L, C = 20.0, 20.0
    qs, xis = _run2(Phi2(lc_cell(L, C, leapfrog)), [1.0], [0.0], 300, _DIR_CELL)
    np.testing.assert_allclose(
        xis[1:-1, 0], L * (qs[2:, 0] - qs[:-2, 0]) / 2.0, atol=1e-12
    )


# ---------------------------------------------------------------------------
# 3. Energy exchange: q^2/(2C) + xi^2/(2L) bounded over many periods.
# ---------------------------------------------------------------------------


def _energy(qs, xis, L, C):
    return qs[:, 0] ** 2 / (2.0 * C) + xis[:, 0] ** 2 / (2.0 * L)


def test_energy_exchange_is_bounded_phase():
    """``phase`` over 12 periods: electric + magnetic energy oscillates in a band
    of relative half-width ~ omega/2 (symplectic Euler's O(dt) energy wobble;
    it preserves a nearby quadratic invariant, so no secular drift).

    Measured: max relative deviation = 0.513 * omega (at omega = 0.05); we
    assert the band is [0.45, 0.55] * omega -- two-sided, so the bound is the
    measured O(dt) wobble, not slack -- and that the last quarter of the run
    stays in the same band (bounded, not drifting).
    """
    L, C = 20.0, 20.0
    w = 1.0 / np.sqrt(L * C)
    steps = int(12.0 * 2.0 * np.pi / w)
    qs, xis = _run(Phi(lc_cell(L, C, phase)), [1.0], [0.0], steps, _DIR_CELL)
    H = _energy(qs, xis, L, C)
    dev = np.abs(H / H[0] - 1.0)
    assert dev.max() <= 0.55 * w                       # bounded by the O(dt) wobble
    assert dev.max() >= 0.45 * w                       # ... and the bound is tight
    assert dev[3 * len(dev) // 4:].max() <= 0.55 * w   # no secular drift


def test_energy_exchange_is_bounded_leapfrog():
    """Leapfrog over 10 periods: the band tightens to O(dt^2).

    Measured: max relative deviation = 0.25 * omega^2 (vs 0.513 * omega for
    ``phase`` -- a factor ~40 at omega = 0.05). Asserted two-sided at
    [0.2, 0.3] * omega^2; this is the 'report the measured bound' number.
    """
    L, C = 20.0, 20.0
    w = 1.0 / np.sqrt(L * C)
    steps = int(10.0 * 2.0 * np.pi / w)
    qs, xis = _run2(Phi2(lc_cell(L, C, leapfrog)), [1.0], [0.0], steps, _DIR_CELL)
    H = _energy(qs, xis, L, C)
    dev = np.abs(H / H[0] - 1.0)
    assert dev.max() <= 0.30 * w * w
    assert dev.max() >= 0.20 * w * w


# ---------------------------------------------------------------------------
# 4. The configuration arrangement on the SAME lens datum: RC relaxation.
# ---------------------------------------------------------------------------


def test_configuration_is_rc_relaxation():
    """``configuration``: ``q_{n+1} = (1 - 1/(LC)) q_n`` exactly; monotone decay to 0.

    From ``coalgebra.configuration``: ``q -> q + sharp(omega|_Q) = q - q/(LC)``.
    This is forward Euler at dt = 1 for ``dq/dt = -q/(LC)``. Electrical
    reading, honestly stated: RC discharge ``dq/dt = -q/(RC)`` with ``R := L``
    -- under the first-order parameter the sharp is a mobility/conductance
    ``1/R`` (Ohm's law ``I = -V/R``, ``V = dU/dq = q/C``), NOT an inductance;
    the same stored number changes physical role (ohms vs henries) with the
    coalgebra. Asserted: (a) the exact geometric law at machine precision,
    (b) strictly monotone decay, no sign change (no oscillation; requires
    ``LC > 1``, i.e. dt inside the forward-Euler stability region),
    (c) the continuum exponential within the forward-Euler O(dt) bound
    ``max_n |(1-x)^n - e^{-nx}| ~ x/(2e)`` with ``x = 1/(LC)``.
    """
    L, C = 4.0, 25.0                                   # x = 1/(LC) = 0.01
    x = 1.0 / (L * C)
    O = Phi(lc_cell(L, C, configuration))
    state = jnp.array([1.0])
    qs = [1.0]
    for _ in range(400):
        _, _, state = O.with_state(state).run_one(IN_POS, _DIR_CELL)
        qs.append(float(state[0]))
    qs = np.asarray(qs)
    n = np.arange(len(qs))

    np.testing.assert_allclose(qs, (1.0 - x) ** n, atol=1e-12)          # (a)
    assert np.all(np.diff(qs) < 0.0) and np.all(qs > 0.0)               # (b)
    assert qs[-1] < 0.02                                                # decayed to ~0
    assert np.max(np.abs(qs - np.exp(-x * n))) <= 0.2 * x               # (c)


# ---------------------------------------------------------------------------
# 5. Coupled LC pair: two capacitor boxes + stateless coupling box.
# ---------------------------------------------------------------------------


def test_pair_construction_is_the_advertised_composite():
    """The wired pair's data emerge from genuine ``arr`` composition: parameter
    base R^2, sharp diag(1/L, 1/L) (read off the configuration step), and the
    composite forcing ``-dU`` for ``U = q1^2/(2C) + q2^2/(2C) + (q1-q2)^2/(2 C_c)``
    -- not written by hand."""
    L, C, Cc = 20.0, 20.0, 5.0
    pair = coupled_lc_pair(L, C, Cc, configuration)
    assert pair.P.Q_dim == 2
    assert (pair.out_dim_M, pair.in_dim_M, pair.out_dim_N, pair.in_dim_N) == (0, 0, 0, 0)
    assert coupling_capacitor(Cc).P.Q_dim == 0         # the coupling box is stateless

    # the sharp lives inside the coalgebra's step: s |-> s + sharp(xi).
    s0 = ((jnp.zeros(1), jnp.zeros(1)), jnp.zeros(0))
    for k, e in enumerate([jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])]):
        moved, _ = _pair_flat_conf(pair.P.step(s0, e))
        np.testing.assert_allclose(np.asarray(moved), np.eye(2)[k] / L, atol=1e-14)

    rng = np.random.default_rng(7)
    for _ in range(5):
        q = rng.standard_normal(2)
        xi_q = np.asarray(pair.omega(jnp.asarray(q), jnp.zeros(0), jnp.zeros(0))[0])
        dU = np.array([q[0] / C + (q[0] - q[1]) / Cc, q[1] / C - (q[0] - q[1]) / Cc])
        np.testing.assert_allclose(xi_q, -dU, atol=1e-12)


def _pair_flat_conf(state):
    """Flat positions of a pair of *configuration* cells (and the trivial wire state)."""
    (s1, s2), w = state
    return jnp.concatenate([s1, s2]), w


def test_coupled_pair_normal_modes():
    """Both predicted normal-mode frequencies appear:

        omega_+ = 1/sqrt(LC)                  (in phase: coupling capacitor idle),
        omega_- = sqrt(1/(LC) + 2/(L C_c))    (antiphase).

    With (L, C, C_c) = (20, 20, 5): omega_+ = 0.05, omega_- = 0.15 exactly.
    The discrete map is linear and preserves the symmetric/antisymmetric
    subspaces, so starting in an eigenmode we measure that mode's period
    (tolerances as in test 1: known bias omega^2/24, factor-2 margin, plus the
    sharp discrete check) and assert the trajectory stays in the eigenspace to
    machine precision -- both modes are exhibited, cleanly separated.
    """
    L, C, Cc = 20.0, 20.0, 5.0
    pair = coupled_lc_pair(L, C, Cc, phase)
    w_plus = 1.0 / np.sqrt(L * C)
    w_minus = np.sqrt(1.0 / (L * C) + 2.0 / (L * Cc))
    np.testing.assert_allclose([w_plus, w_minus], [0.05, 0.15], atol=1e-15)

    for q0, w, purity in [
        ([1.0, 1.0], w_plus, lambda q: q[:, 0] - q[:, 1]),
        ([1.0, -1.0], w_minus, lambda q: q[:, 0] + q[:, 1]),
    ]:
        T_cont = 2.0 * np.pi / w
        T_disc = 2.0 * np.pi / (2.0 * np.arcsin(w / 2.0))
        qs, _ = _run_pair(Phi(pair), q0, [0.0, 0.0], int(5.2 * T_cont), _DIR_PAIR)
        T_meas = _measured_period(qs[:, 0])
        assert abs(T_meas - T_cont) / T_cont <= w * w / 12.0
        assert abs(T_meas - T_disc) / T_disc <= 1e-5
        assert np.max(np.abs(purity(qs))) <= 1e-12     # stays in the eigenmode


# ---------------------------------------------------------------------------
# 6. The two-box configuration encoding (CDLM ex.further_reach, "LC circuit").
# ---------------------------------------------------------------------------


def test_lc_conf_is_the_advertised_composite():
    """The paper item's data emerge from genuine ``arr`` composition: parameter
    base R^2 = (i, v); sharp diag(-1/L, +1/C) -- OPPOSITE signs (the trick);
    composite forcing ``-d(i*v)`` (not written by hand); outer output reporting v."""
    L, C = 5.0, 20.0
    arr = lc_conf(L, C)
    assert arr.P.Q_dim == 2
    assert (arr.out_dim_N, arr.in_dim_N) == (1, 0)
    assert current_voltage_coupler_direct().P.Q_dim == 0   # the coupler is stateless

    s0 = ((jnp.zeros(1), jnp.zeros(1)), jnp.zeros(0))
    sharp = np.stack([
        np.asarray(_conf_flat(arr.P.step(s0, e)))
        for e in (jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]))
    ]).T
    np.testing.assert_allclose(sharp, np.diag([-1.0 / L, 1.0 / C]), atol=1e-14)

    rng = np.random.default_rng(11)
    for _ in range(5):
        q = rng.standard_normal(2)
        xi_q = np.asarray(arr.omega(jnp.asarray(q), jnp.zeros(0), jnp.zeros(0))[0])
        np.testing.assert_allclose(xi_q, [-q[1], -q[0]], atol=1e-12)     # -d(i v)
        v = np.asarray(arr.out_f(jnp.asarray(q), jnp.zeros(0)))
        np.testing.assert_allclose(v, q[1:2], atol=1e-14)


def test_lc_conf_one_tick_is_forward_euler():
    """One tick of ``Phi`` is EXACTLY  i -> i + v/L,  v -> v - i/C  (the
    forward-Euler step of the inductor law di/dt = v/L and the capacitor law
    dv/dt = -i/C), for random states and several (L, C)."""
    rng = np.random.default_rng(3)
    for L, C in [(5.0, 20.0), (2.0, 2.0), (30.0, 7.0)]:
        O = Phi(lc_conf(L, C))
        for _ in range(4):
            iv = rng.standard_normal(2)
            _, _, state = O.with_state(_conf_state(iv)).run_one(IN_POS, _DIR_CONF)
            expect = np.array([iv[0] + iv[1] / L, iv[1] - iv[0] / C])
            np.testing.assert_allclose(np.asarray(_conf_flat(state)), expect, atol=1e-13)


def test_lc_conf_energy_law_and_period():
    """The lossless forward-Euler signature, exactly and in aggregate:

    (a) the energy H = (L i^2 + C v^2)/2 grows by EXACTLY 1 + 1/(LC) per tick
        (in scaled coordinates the map is sqrt(1 + omega^2) times a rotation,
        omega = 1/sqrt(LC); cf. dap/dc_motor.py's lossless law);
    (b) the rotation angle per tick is arctan(omega), so the measured period
        is 2 pi / arctan(omega) ticks, within the continuum 2 pi sqrt(LC) up
        to the forward-Euler period bias omega^2/3 (arctan(w) = w - w^3/3 + ...).
    """
    L, C = 20.0, 20.0                                  # omega = 0.05
    w = 1.0 / np.sqrt(L * C)
    O = Phi(lc_conf(L, C))
    state = _conf_state([0.0, 1.0])
    ivs = [np.asarray(_conf_flat(state))]
    for _ in range(400):
        _, _, state = O.with_state(state).run_one(IN_POS, _DIR_CONF)
        ivs.append(np.asarray(_conf_flat(state)))
    ivs = np.stack(ivs)
    H = (L * ivs[:, 0] ** 2 + C * ivs[:, 1] ** 2) / 2.0

    np.testing.assert_allclose(H[1:] / H[:-1], 1.0 + w**2, atol=1e-12)      # (a)

    T = _measured_period(ivs[:, 1])
    T_disc = 2.0 * np.pi / np.arctan(w)
    T_cont = 2.0 * np.pi * np.sqrt(L * C)
    np.testing.assert_allclose(T, T_disc, rtol=1e-4)                        # (b)
    assert abs(T - T_cont) / T_cont < 0.5 * w**2                            # bias ~ w^2/3


def test_lc_conf_picks_up_a_radio_signal():
    """The circuit tunes: a sinusoidal covector fed at the terminal is an
    antenna, and the steady-state response peaks at omega = 1/sqrt(LC).

    The drive enters as the forcing ``xi_N(t) = -sin(Omega t)`` at the single
    outer port (CDLM fed the cost gradient ``+sin``; DESIGN.md §3); its pullback
    along the report ``(i, v) |-> v`` lands in the ``v``-slot, so the tick
    becomes ``v -> v - (i + sin(Omega t))/C``: an injected current. A small
    series resistance ``R`` (cost ``-(R/2) i^2`` in the negative-sharp inductor
    cell) supplies the damping that makes the peak finite and the transient die;
    it also swamps the forward-Euler energy growth
    (``R/(2L) = 0.005 > omega^2/2 = 0.00125``), so the undriven map is stable.
    Asserted: over a sweep of drive frequencies the largest steady-state
    amplitude of ``v`` occurs exactly at ``Omega = omega``, and beats the
    +-25% detunings by more than 2x (measured ~3x, Q = omega L / R = 5).
    """
    L, C, R = 20.0, 20.0, 0.2
    w = 1.0 / np.sqrt(L * C)
    O = Phi(lc_conf(L, C, R))
    freqs = [0.6 * w, 0.8 * w, w, 1.25 * w, 1.6 * w]
    amps = []
    for W in freqs:
        state = _conf_state([0.0, 0.0])
        vs = []
        for t in range(2000):
            drive = lambda _out_pos, t=t: (jnp.array([-np.sin(W * t)]), jnp.zeros(0))
            _, _, state = O.with_state(state).run_one(IN_POS, drive)
            vs.append(float(_conf_flat(state)[1]))
        amps.append(float(np.max(np.abs(np.asarray(vs[1000:])))))
    assert int(np.argmax(amps)) == freqs.index(w)
    assert amps[freqs.index(w)] > 2.0 * max(amps[1], amps[3])


# ---------------------------------------------------------------------------
# 7. Trick vs direct: the same LC rotation from a non-exact forcing.
# ---------------------------------------------------------------------------


def test_direct_coupler_one_form_is_not_exact():
    """``current_voltage_coupler_direct``'s forcing ``v di - i dv`` has
    ``d omega = -2 di ^ dv != 0``: the antisymmetric part of the returned
    covector's Jacobian is nonzero, so no potential produces it (contrast the
    trick coupler ``-d(i v)``, whose Jacobian is symmetric)."""
    cpl = current_voltage_coupler_direct()

    def F(e):
        return cpl.omega(jnp.zeros(0), e, jnp.zeros(0))[1]

    J = np.asarray(jax.jacobian(F)(jnp.array([0.3, -1.2])))
    np.testing.assert_allclose(J.T - J, np.array([[0.0, -2.0], [2.0, 0.0]]), atol=1e-14)

    trick = lc_conf(5.0, 20.0)

    def G(e):
        return trick.omega(jnp.asarray(e), jnp.zeros(0), jnp.zeros(0))[0]

    Jt = np.asarray(jax.jacobian(G)(jnp.array([0.3, -1.2])))
    np.testing.assert_allclose(Jt.T - Jt, np.zeros((2, 2)), atol=1e-14)


def test_direct_realization_agrees_with_trick():
    """``lc_conf_direct`` (positive sharps, non-exact coupler one-form) and
    ``lc_conf`` (oppositely-signed sharps, exact forcing) step identically,
    driven and undriven -- the trick moves out of the sharp and into ``omega``
    (DESIGN.md §6(c))."""
    L, C, R = 20.0, 20.0, 0.2
    trick, direct = lc_conf(L, C, R), lc_conf_direct(L, C, R)

    # both of the direct realization's sharps are positive (inductance, capacitance)
    s0 = ((jnp.zeros(1), jnp.zeros(1)), jnp.zeros(0))
    sharp = np.stack([
        np.asarray(_conf_flat(direct.P.step(s0, e)))
        for e in (jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]))
    ]).T
    np.testing.assert_allclose(sharp, np.diag([1.0 / L, 1.0 / C]), atol=1e-14)

    Ot, Od = Phi(trick), Phi(direct)
    st, sd = _conf_state([0.3, -0.7]), _conf_state([0.3, -0.7])
    w = 1.0 / np.sqrt(L * C)
    for t in range(200):
        drive = lambda _out_pos, t=t: (jnp.array([-np.sin(w * t)]), jnp.zeros(0))
        _, _, st = Ot.with_state(st).run_one(IN_POS, drive)
        _, _, sd = Od.with_state(sd).run_one(IN_POS, drive)
        np.testing.assert_allclose(
            np.asarray(_conf_flat(sd)), np.asarray(_conf_flat(st)), rtol=1e-12, atol=1e-14
        )


# ---------------------------------------------------------------------------
# 8. Regression against the CDLM trajectories.
# ---------------------------------------------------------------------------


def test_regression_matches_cdlm():
    """The CDLM trajectories (``regress/lc_circuit.npz``) to rtol 1e-10 (DESIGN.md §3).

    Every distinct dynamics the old test ran is checked: the phase cell (three
    ``(L, C)``), the leapfrog cell, the configuration cell, the coupled pair in
    both normal modes, and ``lc_conf`` undriven (one-tick states, the energy
    trajectory) and driven (the antenna, inputs negated). The two long runs --
    the slow ``(L, C)`` phase cells and the four off-resonance drive
    frequencies -- are compared on a prefix of the recorded trajectory; the
    dynamics they exercise is the same as the run checked in full beside them.
    """
    R = load_regress("lc_circuit")

    # phase cells
    for k, (L, C) in enumerate(R["phase_LC"]):
        n = min(int(R["phase_steps"][k]), 800)
        qs, xis = _run(Phi(lc_cell(L, C, phase)), [1.0], [0.0], n, _DIR_CELL)
        np.testing.assert_allclose(qs, R[f"phase_q_{k}"][:n + 1], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(xis, R[f"phase_xi_{k}"][:n + 1], rtol=1e-10, atol=1e-12)

    # leapfrog cell
    L, C, n = R["leap_params"]
    qs, xis = _run2(Phi2(lc_cell(L, C, leapfrog)), [1.0], [0.0], int(n), _DIR_CELL)
    np.testing.assert_allclose(qs, R["leap_q"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(xis, R["leap_xi"], rtol=1e-10, atol=1e-12)

    # configuration cell (RC decay)
    L, C, n = R["conf_params"]
    O = Phi(lc_cell(L, C, configuration))
    state = jnp.array([1.0])
    qs = [np.asarray(state)]
    for _ in range(int(n)):
        _, _, state = O.with_state(state).run_one(IN_POS, _DIR_CELL)
        qs.append(np.asarray(state))
    np.testing.assert_allclose(np.stack(qs), R["conf_q"], rtol=1e-10, atol=1e-12)

    # coupled pair, both modes
    Lp, Cp, Ccp = R["pair_params"]
    pair = Phi(coupled_lc_pair(Lp, Cp, Ccp, phase))
    for tag, q0 in (("plus", [1.0, 1.0]), ("minus", [1.0, -1.0])):
        n = int(R[f"pair_steps_{tag}"])
        qs, xis = _run_pair(pair, q0, [0.0, 0.0], n, _DIR_PAIR)
        np.testing.assert_allclose(qs, R[f"pair_q_{tag}"], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(xis, R[f"pair_xi_{tag}"], rtol=1e-10, atol=1e-12)

    # lc_conf one-tick states
    for (L2, C2), iv, want in zip(R["tick_LC"], R["tick_in"], R["tick_out"]):
        _, _, s = Phi(lc_conf(L2, C2)).with_state(_conf_state(iv)).run_one(IN_POS, _DIR_CONF)
        np.testing.assert_allclose(np.asarray(_conf_flat(s)), want, rtol=1e-10, atol=1e-12)

    # lc_conf energy trajectory
    L3, C3, n3 = R["energy_params"]
    O3 = Phi(lc_conf(L3, C3))
    state = _conf_state([0.0, 1.0])
    ivs = [np.asarray(_conf_flat(state))]
    for _ in range(int(n3)):
        _, _, state = O3.with_state(state).run_one(IN_POS, _DIR_CONF)
        ivs.append(np.asarray(_conf_flat(state)))
    np.testing.assert_allclose(np.stack(ivs), R["energy_iv"], rtol=1e-10, atol=1e-12)

    # the driven radio sweep (inputs negated, DESIGN.md §3)
    L4, C4, R4, n4 = R["radio_params"]
    O4 = Phi(lc_conf(L4, C4, R4))
    for k, (row, W) in enumerate(zip(R["radio_v"], R["radio_freqs"])):
        n = int(n4) if k == 2 else 300      # the resonant drive in full, the detunings on a prefix
        state = _conf_state([0.0, 0.0])
        vs = []
        for t in range(n):
            drive = lambda _o, t=t, W=W: (jnp.array([-np.sin(W * t)]), jnp.zeros(0))
            _, _, state = O4.with_state(state).run_one(IN_POS, drive)
            vs.append(float(_conf_flat(state)[1]))
        np.testing.assert_allclose(np.asarray(vs), row[:n], rtol=1e-10, atol=1e-12)
