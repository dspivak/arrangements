"""The LC circuit: the electrical reading of one arrangement datum under two parameters.

An electrical instance of the wave/heat dichotomy (cf. test_wave_equation.py /
test_heat_equation.py): ONE lens datum whose parameter base is the capacitor
charge ``q``, with

* reactive sharp  ``sharp(xi) = xi / L``     -- the inductance ``L`` in the mass slot,
* forcing         ``omega = -dU``, ``U(q) = q^2 / (2 C)`` -- the capacitor's
  electrostatic energy (a cost; STATUS.md §1),

made into two arrangements (def:arr) by two parameters (sec.integrators), read
by the one dynamics functor ``Phi`` (sec.dynamics_functor):

* ``phase(Q)`` stores ``(q, xi)`` and steps ``q -> q + xi/L``, ``xi -> xi - q~/C``
  (forces at the presented position ``q~``). Because
  ``q_{n+1} - q_n = xi_n / L`` exactly, the stored momentum is
  ``xi_n = L (q_{n+1} - q_n) / dt``: the *flux linkage* ``lambda = L I``
  conjugate to charge, whose kinetic term ``xi^2/(2L)`` is the magnetic energy
  of the inductor. The circuit is a harmonic oscillator with angular frequency
  ``omega = 1/sqrt(LC)`` -- the same arrangement shape as the mechanical
  particle, with ``(mass, stiffness) = (L, 1/C)``.

* ``configuration(Q)`` stores ``q`` alone and steps ``q -> q - (1/L)(q/C)``:
  forward Euler for the first-order relaxation ``dq/dt = -q/(L C)``. Electrically
  this is RC discharge ``dq/dt = -q/(R C)`` with ``R := L``: under the first-order
  parameter the sharp is a *mobility* (a conductance ``1/R``, Ohm's law
  ``I = -V/R`` with ``V = dU/dq = q/C``), not an inductance. The lens datum
  fixes one number; the coalgebra decides whether it plays the reactive role
  (henries) or the dissipative one (ohms) -- exactly the mass-vs-mobility
  reading of the wave/heat pair (sec.heat_waves).

Units and time step: the coalgebra tick is one unit of time -- the coalgebras
of coalgebra.py carry no ``dt`` (the phase step advances by the sharp itself).
Working in tick units, physical data ``(L_phys, C_phys)`` at step ``dt``
correspond to ``(L, C) = (L_phys/dt, C_phys/dt)`` here, and then the stored
``xi`` *is* the physical flux linkage. Resolution therefore means choosing
``omega = 1/sqrt(LC)`` small in tick units.

The coupled pair (the multi-box variant): two capacitor boxes, each exposing
its charge on an output port, composed in ``arr`` with a *stateless* coupling
box -- trivial parameter, no state -- carrying the coupling capacitor's energy
``(q_1 - q_2)^2/(2 C_c)`` read off its two input ports. The composite forcing
emerges from ``compose_seq`` (it is not written by hand), and the normal modes
are ``omega_+ = 1/sqrt(LC)`` (charges in phase; the coupling capacitor carries
no charge) and ``omega_- = sqrt(1/(LC) + 2/(L C_c))`` (charges in antiphase).

The two-box configuration encoding (CDLM ex.further_reach, "LC circuit") comes
in two realizations, a TRICK and a DIRECT one (STATUS.md §2, trick vs direct):

* ``lc_conf`` -- the TRICK: an inductor cell and a capacitor cell with
  OPPOSITELY-SIGNED sharps, wired by a stateless coupler whose forcing is exact,
  ``-d(i*v)``. CDLM's arrangement carried only a potential, and a potential
  coupling has a symmetric cross-Hessian, so producing a rotation forced the two
  cells' sharps to differ in sign (the Lotka-Volterra / DC-motor mechanism,
  cf. dc_motor.py point 2).
* ``lc_conf_direct`` -- the DIRECT realization: both sharps positive (plain
  ``configuration`` parameters, the inductance and capacitance where they
  belong) and the coupler carrying the NON-EXACT one-form ``v di - i dv`` on its
  two ports, written by hand -- ``d(v di - i dv) = -2 di ^ dv != 0``, so no
  potential produces it. The two realizations agree step for step
  (tests/test_lc_circuit.py).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from .arrangement import Arrangement, exact, zero_form
from .coalgebra import configuration, phase
from .rvect import diagonal
from .wiring import compose_seq, tensor_arrangements, trivial_parameter


def lc_cell(L: float, C: float, parameter=phase) -> Arrangement:
    """A single LC cell ``<R^0|R^0> -> <R^1_out | R^0_in>``.

    Parameter ``parameter(Q)`` (``phase`` by default, ``configuration`` for the
    RC reading, ``leapfrog`` for the two-stage one) on ``Q = R``, the capacitor
    charge ``q``, with constant reactive sharp ``xi |-> xi/L`` (the inductance
    in the mass slot) and forcing ``omega = -dU``, ``U(q) = q^2/(2C)`` (the
    capacitor energy, a cost). The output port exposes the charge; there are no
    input ports. Run closed, feed the port the zero covector ``xi_N = 0`` (an
    unloaded terminal).
    """
    return Arrangement(
        P=parameter(diagonal(jnp.array([1.0 / L]))),
        in_dim_M=0,
        out_dim_M=0,
        in_dim_N=0,
        out_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -(q[0] ** 2) / (2.0 * C)),
        label=f"LC(L={L:g},C={C:g})",
    )


def coupling_capacitor(C_c: float) -> Arrangement:
    """The stateless coupling box ``<R^2_out | R^0_in> -> <R^0|R^0>``.

    Trivial parameter ``R^0`` (no state), no ports on the outside: it closes
    the pair. Its forcing is exact, ``-d`` of the coupling capacitor's energy
    ``(m_1 - m_2)^2/(2 C_c)`` read off the two charges presented at its input
    ports; ``compose_seq`` pulls it back and adds it to the cells' own forcings
    (pullback-and-sum, sec.forcing), and the interpretation's cotangent pullback
    (CDLM eqn.bigtheta) routes it back to both cells' parameter covectors.
    """
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0,
        out_dim_M=2,
        in_dim_N=0,
        out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -((m_out[0] - m_out[1]) ** 2) / (2.0 * C_c)),
        label=f"Cc({C_c:g})",
    )


def coupled_lc_pair(L: float, C: float, C_c: float, parameter=phase) -> Arrangement:
    """Two LC cells wired through the stateless coupling box, in ``arr``:

        coupled = compose_seq( lc_cell (x) lc_cell,  coupling_capacitor ),

    a closed 0-ary arrangement with parameter base ``R^2 = (q_1, q_2)``, sharp
    ``diag(1/L, 1/L)``, and composite forcing ``-dU`` for

        U = q_1^2/(2C) + q_2^2/(2C) + (q_1 - q_2)^2/(2 C_c)

    emerging from genuine composition (no hand-specialization). The state is
    ``((s_1, s_2), s_wire)`` with ``s_wire = R^0`` the coupler's trivial state.
    """
    cells = tensor_arrangements([lc_cell(L, C, parameter), lc_cell(L, C, parameter)])
    return compose_seq(cells, coupling_capacitor(C_c))


# ---------------------------------------------------------------------------
# The two-box configuration encoding (CDLM ex.further_reach, "LC circuit"): an
# inductor cell and a capacitor cell with OPPOSITELY-SIGNED sharps, wired by a
# stateless arrangement whose potential is the product i*v and which reports v.
# This is the TRICK realization; the DIRECT one follows below.
# ---------------------------------------------------------------------------


def inductor_cell(L: float, R: float = 0.0, parameter=configuration) -> Arrangement:
    """The inductor as a configuration cell ``<R^0|R^0> -> <R^1_out | R^0_in>``.

    Parameter ``i`` (the current through the inductor), constant scalar sharp
    ``-1/L`` (NEGATIVE: a potential coupling has a symmetric cross-Hessian, so
    producing a rotation forces the two cells' sharps to differ in sign -- the
    Lotka-Volterra / DC-motor mechanism, cf. dc_motor.py point 2),
    reporting ``i`` on its single output port. An optional series resistance
    ``R`` enters as the potential ``-(R/2) i^2`` -- NEGATED, because the cell's
    sharp is negative (dc_motor.py's mechanical-cell convention: the flow
    ``-sharp(dU)`` is invariant under ``(sharp, U) -> (-sharp, -U)``); the
    self-dynamics is then the damped ``i -> i + (v - R i)/L``. In the book's
    sign the forcing is ``omega = -dU = +d((R/2) i^2)``.

    TRICK realization (see the module docstring and ``inductor_cell_direct``).
    """
    return Arrangement(
        P=parameter(diagonal(jnp.array([-1.0 / L]))),
        in_dim_M=0,
        out_dim_M=0,
        in_dim_N=0,
        out_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: (R / 2.0) * q[0] ** 2),
        label=f"ind(L={L:g},R={R:g})",
    )


def capacitor_cell(C: float, parameter=configuration) -> Arrangement:
    """The capacitor as a configuration cell ``<R^0|R^0> -> <R^1_out | R^0_in>``:
    parameter ``v`` (the voltage across it), constant scalar sharp ``+1/C``,
    zero forcing, reporting ``v``. (Same convention as logic.py's
    ``node``: state = voltage, sharp = 1/C.) Shared by the trick and the direct
    realization."""
    return Arrangement(
        P=parameter(diagonal(jnp.array([1.0 / C]))),
        in_dim_M=0,
        out_dim_M=0,
        in_dim_N=0,
        out_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=zero_form,
        label=f"cap(C={C:g})",
    )


def current_voltage_coupler() -> Arrangement:
    """The stateless arrangement ``f: <R^2_out|R^0> -> <R^1_out|R^0>`` of the
    paper's LC-circuit item: potential ``W(i, v) = i * v`` read off the two
    reports (forcing ``omega = -dW``, exact), output map ``(i, v) |-> v`` (the
    circuit's terminal voltage). TRICK realization: exact, so it needs the
    opposite-sign sharps of ``inductor_cell`` to produce a rotation."""
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0,
        out_dim_M=2,
        in_dim_N=0,
        out_dim_N=1,
        out_f=lambda q, m_out: m_out[1:2],
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -(m_out[0] * m_out[1])),
        label="iv",
    )


def lc_conf(L: float, C: float, R: float = 0.0, parameter=configuration) -> Arrangement:
    """``compose_seq( inductor_cell (x) capacitor_cell, current_voltage_coupler )``.

    Parameter base ``R^2 = (i, v)``, sharp ``diag(-1/L, +1/C)``, composite
    forcing ``-d(i*v)`` (emerging from composition, not written by hand). One
    tick of ``Phi`` is exactly the forward-Euler step of the LC equations

        i -> i + v/L,        v -> v - i/C,

    the discrete rotation-with-growth whose energy law is pinned in
    tests/test_lc_circuit.py. With ``R > 0`` the ``i``-update gains the damping
    ``-R i / L``, and a time-varying covector fed at the outer port (an
    antenna: it enters the ``v``-update through the report's pullback) drives
    a resonance peaked at ``omega = 1/sqrt(LC)`` -- the tuning of a radio.

    TRICK realization (opposite-sign sharps, exact forcing); the state is
    ``((i, v), s_wire)``. ``lc_conf_direct`` is the direct one.
    """
    cells = tensor_arrangements([inductor_cell(L, R, parameter), capacitor_cell(C, parameter)])
    return compose_seq(cells, current_voltage_coupler())


# ---------------------------------------------------------------------------
# The DIRECT realization: positive sharps, the rotation carried by a non-exact
# forcing on the coupler's two ports (STATUS.md §2, trick vs direct).
# ---------------------------------------------------------------------------


def inductor_cell_direct(L: float, R: float = 0.0, parameter=configuration) -> Arrangement:
    """The inductor with its sharp where it belongs: constant scalar sharp
    ``+1/L``, parameter ``i``, reporting ``i``. The series resistance is the
    cost ``(R/2) i^2``, forcing ``omega = -d((R/2) i^2) = -R i di`` (exact):
    self-dynamics ``i -> i - R i / L``. DIRECT realization; the rotation comes
    from ``current_voltage_coupler_direct``, not from a sign in the sharp."""
    return Arrangement(
        P=parameter(diagonal(jnp.array([1.0 / L]))),
        in_dim_M=0,
        out_dim_M=0,
        in_dim_N=0,
        out_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -(R / 2.0) * q[0] ** 2),
        label=f"ind_direct(L={L:g},R={R:g})",
    )


def current_voltage_coupler_direct() -> Arrangement:
    """The stateless coupler ``<R^2_out|R^0> -> <R^1_out|R^0>`` with the
    NON-EXACT forcing, written by hand, on its two ports ``(i, v)``:

        omega(i, v) = v di - i dv,      i.e.  (xi_i, xi_v) = (v, -i).

    ``d omega = dv ^ di - di ^ dv = -2 di ^ dv != 0``: no potential produces it
    (the "reciprocity" of an exact coupler, ``dF_i/de_j = dF_j/de_i``, fails --
    the i-v coupling of an LC loop is a non-reciprocal, gyrator-like element).
    With both cells' sharps positive the pullback along the reports gives the
    LC rotation ``i -> i + v/L, v -> v - i/C`` directly. Output map
    ``(i, v) |-> v``, as in the trick coupler. DIRECT realization."""
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0,
        out_dim_M=2,
        in_dim_N=0,
        out_dim_N=1,
        out_f=lambda q, m_out: m_out[1:2],
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=lambda q, m_out, n_in: (
            jnp.zeros(0),
            jnp.array([m_out[1], -m_out[0]]),
            jnp.zeros(0),
        ),
        label="iv_direct",
    )


def lc_conf_direct(L: float, C: float, R: float = 0.0, parameter=configuration) -> Arrangement:
    """``compose_seq( inductor_cell_direct (x) capacitor_cell, current_voltage_coupler_direct )``.

    The DIRECT realization of ``lc_conf``: parameter base ``R^2 = (i, v)`` with
    sharp ``diag(+1/L, +1/C)`` (plain ``configuration`` parameters, both
    positive), and composite forcing

        omega = (v - R i) di - i dv,

    non-exact, emerging from the coupler's hand-written ``v di - i dv`` pulled
    back along the reports plus the inductor's ``-R i di``. Its ``Phi`` agrees
    with ``lc_conf``'s step for step (same state shape ``((i, v), s_wire)``,
    same outer port), including the antenna drive: the outer covector pulls
    back along ``(i, v) |-> v`` into the ``v``-slot in both.
    """
    cells = tensor_arrangements([inductor_cell_direct(L, R, parameter), capacitor_cell(C, parameter)])
    return compose_seq(cells, current_voltage_coupler_direct())
