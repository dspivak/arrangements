"""Impedance (P-)control as a virtual spring.

EXTENSION (beyond the paper) -- an application of the existing machinery, no
new formal content. An impedance controller of stiffness ``k`` is a *stateless*
box whose entire content is the forcing ``omega_c = -dk_c`` of the cost

    k_c(q_rep, s) = (k/2) |q_rep - s|^2,

a virtual spring hung between the plant's reported position ``q_rep`` and a
setpoint ``s``. The plant is the particle pattern of CDLM sec.spring_first_pass
with zero forcing of its own (the simplest honest choice: all closed-loop forces
come from the controller); the coupling is a wiring; and the closed-loop
behavior is whatever the dynamics functor ``Phi`` (sec.dynamics_functor) makes
of the composite -- which behavior it is now depends on the *parameter*, the
coalgebra inside the arrangement (def:arr, sec.matter), not on a choice
made in the semantics:

* ``configuration`` -- proportional setpoint tracking,
  ``q <- q + sharpR_q(-k(q - s))``;
* ``phase``         -- the classical stiffness-only impedance law: undamped
  oscillation about ``s`` (semi-implicit Euler on the harmonic oscillator).

The setpoint is NOT baked into the arrangement. The composite is an *open*
morphism ``<R^0|R^0> -> <R^d|R^d>`` whose outer input port carries ``s``; the
environment supplies it per step through ``PCMorphism.run_one``'s
``in_dir_from`` closure, exactly as the training environment streams data
``(x, lambda)`` in ``learning.train`` (CDLM ex.training_data). Changing the value
passed at step ``t`` changes the setpoint at step ``t`` -- a genuine run-time
input.

Boundary accounting: the composite's outer *position* also emits the one-form
``omega_N(s) = omega_c|_{N^-} = +k(q_rep - s)`` on the setpoint port
(sec.forcing; CDLM eqn.omegaprime with the sign of STATUS.md §1) -- the reaction
of the virtual spring on whoever holds its far end. It is equal and opposite to
the covector ``xi_Q = -k(q_rep - s)`` that drives the plant, because the spring's
cost depends only on the difference ``q_rep - s`` (exactness alone gives
reciprocity, not equal and opposite). ``run_tracking`` below *discards*
it: the setpoint holder is treated as infinitely stiff, and the work it does
against the spring is untracked.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, List, Sequence, Tuple

import jax.numpy as jnp
from jax import Array

from .arrangement import Arrangement, exact, zero_form
from .coalgebra import configuration
from .interpretation import trivial_omega
from .pc import PCMorphism
from .rvect import ReactiveVectorSpace
from .wiring import compose_seq, parallel_arrangements, trivial_parameter


# ---------------------------------------------------------------------------
# The three boxes: plant, controller, wiring.
# ---------------------------------------------------------------------------


def plant_box(Q: ReactiveVectorSpace, parameter=configuration) -> Arrangement:
    """The plant ``Part_plant : <R^0|R^0> -> <R^d|R^0>``, ``d = Q.dim``.

    Parameter the coalgebra ``parameter(Q)`` on the given reactive vector space
    (the sharp carries the timestep/mass: ``euclidean(d, dt)`` for descent,
    ``diagonal([1/m]*d)`` for phase dynamics, per CDLM eqn.learning_sharp /
    CDLM sec.spring_first_pass); output map the identity readout ``q``; no input
    ports; zero forcing.
    """
    d = Q.dim
    return Arrangement(
        P=parameter(Q),
        in_dim_M=0,
        out_dim_M=0,
        in_dim_N=0,
        out_dim_N=d,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=zero_form,
        label="Part_plant",
    )


def impedance_controller(k: float, d: int = 1) -> Arrangement:
    """The stateless controller ``Part_ctrl : <R^0|R^0> -> <R^0|R^{2d}>``.

    Trivial parameter ``R^0`` (stateless: no state, no dynamics of its own),
    no output ports, and ``2d`` input ports read as ``(q_rep, s)``; the entire
    controller is the virtual-spring forcing ``omega_c = -dk_c`` of the cost

        k_c(q_rep, s) = (k/2) |q_rep - s|^2.
    """

    def cost(q: Array, m_out: Array, n_in: Array) -> Array:
        q_rep, s = n_in[:d], n_in[d:]
        return 0.5 * k * jnp.sum((q_rep - s) ** 2)

    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0,
        out_dim_M=0,
        in_dim_N=2 * d,
        out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -cost(q, m_out, n_in)),
        label=f"Part_ctrl(k={k:g})",
    )


def impedance_wire(d: int = 1) -> Arrangement:
    """The wiring ``<R^{2d}|R^d> -> <R^d|R^d>`` closing the loop.

    A static arrangement (trivial parameter, zero forcing) routing, on the model
    of ``chain_wire``:

    * outer output   <- plant's reported position (``out_f = id`` on ``m_out``);
    * controller port 1 <- plant's reported position (feedback);
    * controller port 2 <- outer input (the setpoint ``s``).

    ``M = plant_iface (x) ctrl_iface`` has out-ports ``R^d`` (plant) and
    in-ports ``R^{2d}`` (controller), so ``in_f(m_out, n_in) = (m_out, n_in)``.
    """

    def out_f(q_wire: Array, m_out: Array) -> Array:
        return m_out

    def in_f(q_wire: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.concatenate([m_out, n_in])

    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=2 * d,
        out_dim_M=d,
        in_dim_N=d,
        out_dim_N=d,
        out_f=out_f,
        in_f=in_f,
        omega=zero_form,
        label=f"impedance_wire(d={d})",
    )


def impedance_arrangement(
    Q: ReactiveVectorSpace, k: float, parameter=configuration
) -> Arrangement:
    """``wire(Part_plant, Part_ctrl) : <R^0|R^0> -> <R^d|R^d>`` by genuine composition.

        compose_seq( plant_box(Q, parameter) || impedance_controller(k, d),
                     impedance_wire(d) ).

    The composite parameter is the plant's coalgebra tensored with the trivial
    ones of the controller and the wiring (the controller and wiring contribute
    no state), on base ``Q (+) R^0 (+) R^0 = Q``, and the composite forcing
    *emerges* from ``compose_seq`` (pullback-and-sum, sec.forcing) as
    ``omega = -d (k/2)|q - s|^2`` -- it is not written by hand. The outer input
    port carries the setpoint ``s``; the outer output port reports the plant
    position (at whichever parameter position the coalgebra presents: ``q`` for
    ``configuration``, ``q~ = q + sharpR_q(xi)`` for ``phase``).
    """
    d = Q.dim
    inner = parallel_arrangements(plant_box(Q, parameter=parameter), impedance_controller(k, d))
    wired = compose_seq(inner, impedance_wire(d))
    return replace(wired, label=f"impedance(d={d}, k={k:g})")


# ---------------------------------------------------------------------------
# The composite's state shape.
# ---------------------------------------------------------------------------


def impedance_state(s_plant: Any):
    """The composite's state from the plant coalgebra's state.

    The parameter is ``(P_plant (x) P_ctrl) (x) P_wire`` with the controller's
    and the wiring's states both ``R^0``, so the state is
    ``((s_plant, z), z)``, ``z = R^0``; ``s_plant`` is ``q`` for
    ``configuration`` and ``(q, xi)`` for ``phase``.
    """
    z = jnp.zeros(0)
    return ((s_plant, z), z)


def impedance_flat(state: Any):
    """Inverse of ``impedance_state``: the plant coalgebra's own state."""
    ((s_plant, _), _) = state
    return s_plant


# ---------------------------------------------------------------------------
# Running the closed loop: the setpoint as a per-step run-time input.
# ---------------------------------------------------------------------------

_IN_POS = (jnp.zeros(0), trivial_omega(0))  # the trivial <R^0|R^0> source position


def run_tracking(
    O: PCMorphism, state: Any, setpoints: Sequence
) -> Tuple[List[Array], List[Any]]:
    """Step the coalgebra once per entry of ``setpoints``, threading the state.

    At step ``t`` the environment closes the boundary with the in-direction
    ``(xi_N, in_n) = (0, s_t)``: no external covector pushes on the reported-
    position port, and the setpoint port carries ``s_t`` -- supplied from
    outside at run time, so a mid-run change of setpoint is just a change in
    the list. Returns ``(outs, states)`` with ``outs[t]`` the reported outer
    position after reading ``s_t`` and ``states`` of length ``len(setpoints)+1``
    (including the initial state).

    The emitted one-form ``omega_N`` on the setpoint port (see module
    docstring) is discarded here: the setpoint holder's work is untracked.
    """
    outs: List[Array] = []
    states: List[Any] = [state]
    for s in setpoints:
        s_arr = jnp.atleast_1d(jnp.asarray(s, dtype=float))
        in_dir = (jnp.zeros(s_arr.shape[0]), s_arr)
        out_pos, _out_dir, state = O.with_state(state).run_one(
            _IN_POS, lambda _op, d=in_dir: d
        )
        outs.append(out_pos[0])
        states.append(state)
    return outs, states
