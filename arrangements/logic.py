"""CMOS logic from non-reciprocal MOSFETs (EXTENSION -- not part of the paper).

A MOSFET is *non-reciprocal*: the drain current depends strongly on the gate
voltage, the gate current not at all on the drain voltage.  An arrangement's
port covectors are the forcing ``omega`` restricted to its input interface
(sec.lens_internalization, CDLM eqn.covector_triple, whose ``outp f`` summand is
extended by ``0`` on the ``inpt N`` factor), so for a *stateless* box with an
*exact* forcing ``omega = -dU`` the port Jacobian is ``-d^2 U`` -- symmetric,
hence reciprocal.  No stateless potential is a transistor.

There are two ways to escape that, and this module carries both.

**The TRICK realization** (``mosfet``, CDLM's own, ported unchanged in
substance).  It gets non-reciprocity from the **sharp** instead.  With
parameter ``configuration(Q)`` on ``Q = R^2`` with the constant, degenerate,
non-symmetric reaction

    sharpR = [[0, 1],
              [0, 0]]              (CDLM def.rvect asks only for Q -> vect(Q^*, Q))

and the cost

    U(q, eG, eD, eS) = (kappa/2) (vgs - q0 - q1)^2 + W(q0, vds),

where ``vgs = eG - eS`` and ``vds = eD - eS``, the forcing is ``omega = -dU``
(sec.forcing; DESIGN.md §3) and the configuration step *adds* ``sharp(omega)``
(sec.displacement), so

    dq_0 = (sharpR omega)_0 = -dU/dq1 = kappa (vgs - q0 - q1),
    dq_1 = 0.

The whole ``vds``-dependence of ``dU`` sits in the ``q0``-component, which
``sharpR`` reads as ``0``.  So the gate state tracks ``vgs`` with *no* drain
back-action -- exactly, for **any** channel model ``W``.  Started at ``q1 = 0``
the state settles at ``q0 = vgs`` and the drain returns ``I_D = dW/dvds``.

We call that the **kernel condition**: the drain-sensitive component of ``dU``
lies in ``ker(sharpR)``.  It is *not* the paper's ``unilateral``
(CDLM def.arrangement_terminology), which is the output-side condition
``sharpR_q((partial_q outp f)^T xi) = 0`` for covectors ``xi`` at the output.  Each
device below does satisfy that too -- but vacuously, because it has no output port
(``out_dim_N = 0``, so ``partial_q outp f = 0``), which has nothing to do with being
a transistor.  The kernel condition is input-side, and it is what buys
non-reciprocity here.

**The DIRECT realization** (``mosfet_direct``; DESIGN.md §6(c)).  CDLM had to
route the non-reciprocity through the sharp because its arrangement carried only
a potential.  The book's arrangement carries a *forcing* that need not be exact
(sec.forcing), so the same device is written with the plain parameter
``configuration(Q)`` on the Euclidean ``Q = R^2`` and the hand-written one-form

    omega(q, e)|_Q     = ( kappa (vgs - q0 - q1),  0 ),
    omega(q, e)|_{N^-} = -dU|_{N^-}   (the device's constitutive port law),

which is *not* ``dU`` for any ``U``: its ``Q``-component simply does not depend
on ``vds``, while ``omega|_{N^-}`` does depend on ``q0`` -- the mixed partials
disagree, ``d omega != 0``, and that failure of exactness IS the
non-reciprocity.  The kernel condition becomes the statement that the
drain-sensitive component is *absent from* ``omega|_Q``, rather than being
annihilated by the sharp afterwards.  The two realizations step identically
(tests/test_logic.py::test_trick_and_direct_agree_step_for_step).

Two further structural facts, neither of them put in by hand:

* ``U`` depends on the terminal voltages only through differences, so
  ``dU/deG + dU/deD + dU/deS = 0``: the device's three port currents sum to
  zero.  Device KCL is shift-invariance of the potential.
* Composing devices with a *static* wiring in ``arr`` (compose_seq) adds the
  pulled-back forcings and tensors the parameters (CDLM prop.suboperads,
  DESIGN.md §2).  Since the devices have no output ports, ``omega_tot|_{Q_j} =
  omega_j|_{Q_j}``: each device's kernel condition survives composition
  unchanged, and the node states' updates are the sums of the currents drawn on
  them -- circuit KCL, emergent.

Sign (DESIGN.md §3): CDLM's ``U`` was the cost and its integrator subtracted;
here the forcing is ``-dU`` and the configuration coalgebra adds.  States are
therefore identical to CDLM's, while every covector crossing a port -- the
``load`` presented at the output ports, and the ``port_currents`` ``run``
returns -- is *minus* the current, i.e. minus CDLM's.

Contents: ``mosfet`` (the trick device), ``mosfet_direct`` (the same device as a
non-exact forcing), ``node``/``rail`` (a capacitor / an ideal source, the latter
a *frozen* state, ``sharpR = 0``), ``nand_gate`` (four devices wired by
``nand_wire``), ``sr_latch`` (two ``nand_gate``s cross-coupled), and ``run`` (a
``Phi`` rollout).  The composites take a ``device=`` keyword, so each is
available in either realization, and a ``parameter=`` keyword (a constructor
``Q -> CotCoalg``, ``configuration`` by default) where CDLM chose its integrator
at ``Phi*`` time.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, List, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from .arrangement import Arrangement, exact, zero_form
from .coalgebra import CotCoalg, configuration
from .functors import Phi
from .interpretation import trivial_omega
from .rvect import ReactiveVectorSpace, constant, euclidean, trivial
from .wiring import compose_chain, compose_seq, tensor_arrangements

# The non-reciprocal reaction: degenerate and non-symmetric.  It reads only the
# q1-component of the covector and writes only the q0-component of the state, so
# the q0-component of dU -- which carries all the drain dependence -- is in its
# kernel (the "kernel condition" of the header; NOT CDLM def.arrangement_terminology's
# output-side ``unilateral``).
KERNEL_SHARP = jnp.array([[0.0, 1.0], [0.0, 0.0]])

_IN_POS = (jnp.zeros(0), trivial_omega(0))

Parameter = Callable[[ReactiveVectorSpace], CotCoalg]


# ---------------------------------------------------------------------------
# Flat coordinates for a composite's state (CDLM's states were flat arrays).
# ---------------------------------------------------------------------------


def flat_of(state) -> Array:
    """The composite state's coordinates, its leaves in tensor order.

    A composite's parameter is a tensor of coalgebras (``CotCoalg.tensor``), so
    its state is a tree of pairs whose leaves are the boxes' states in tensor
    order -- the order in which ``position`` concatenates them.  CDLM's
    parameter was one flat reactive vector space, so its states were flat
    arrays; these two helpers convert, and ``stepper``/``run`` take and return
    the flat form, as CDLM's did.
    """
    return jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(state)])


def state_of(arr: Arrangement, flat: Array):
    """The state of ``arr``'s parameter with coordinates ``flat`` (inverse of ``flat_of``)."""
    flat = jnp.asarray(flat, float)
    leaves, treedef = jax.tree_util.tree_flatten(arr.P.init)
    pieces, i = [], 0
    for leaf in leaves:
        n = int(jnp.size(leaf))
        pieces.append(jnp.reshape(flat[i:i + n], jnp.shape(leaf)))
        i += n
    if i != flat.shape[0]:
        raise ValueError(f"state_of: {flat.shape[0]} coordinates for a state of size {i}")
    return jax.tree_util.tree_unflatten(treedef, pieces)


# ---------------------------------------------------------------------------
# Devices.
# ---------------------------------------------------------------------------


def _fet_cost(
    kind: str, kappa: float, beta: float, Vt: float, Vsat: float, sharpness: float
) -> Callable[[Array, Array, Array], Array]:
    """The device's cost ``U(q, m_out, n_in)``: the gate term plus the channel model ``W``.

    Square-law saturation with a ``tanh`` triode knee:
    ``I_D = g(vgs) tanh(vds / Vsat)`` with ``g(v) = beta * relu_s(+-v - Vt)^2``
    (``+`` for ``kind="n"``, ``-`` for ``kind="p"``), realized as
    ``W(q0, vds) = g(q0) * Vsat * logcosh(vds / Vsat)``.  Shared by the trick
    realization (which takes ``omega = -dU``) and the direct one (which takes
    only ``-dU|_{N^-}`` from it and writes ``omega|_Q`` by hand).
    """
    if kind not in ("n", "p"):
        raise ValueError("kind must be 'n' or 'p'")
    sign = 1.0 if kind == "n" else -1.0

    def g(v: Array) -> Array:
        # smooth relu, sharp enough that a sub-threshold device is really off
        return beta * (jax.nn.softplus(sharpness * (sign * v - Vt)) / sharpness) ** 2

    def U(q: Array, m_out: Array, n_in: Array) -> Array:
        eG, eD, eS = n_in[0], n_in[1], n_in[2]
        vgs, vds = eG - eS, eD - eS
        gate = 0.5 * kappa * (vgs - q[0] - q[1]) ** 2
        x = vds / Vsat
        logcosh = jnp.logaddexp(x, -x) - jnp.log(2.0)
        return gate + g(q[0]) * Vsat * logcosh

    return U


def mosfet(
    kind: str = "n",
    kappa: float = 0.5,
    beta: float = 4.0,
    Vt: float = 0.3,
    Vsat: float = 0.1,
    sharpness: float = 20.0,
    label: str = "",
    parameter: Parameter = configuration,
) -> Arrangement:
    """A non-reciprocal MOSFET ``<R^3|R^0>`` system: ports ``(eG, eD, eS)``.

    The TRICK realization (module header): parameter ``parameter(Q)`` on the
    *degenerate, non-symmetric* ``KERNEL_SHARP``, forcing the exact
    ``omega = -dU`` of ``_fet_cost``.  The gate returns ``I_G =
    kappa (vgs - q0 - q1)``, which vanishes once the gate state settles; the
    drain returns ``I_D = dW/dvds`` (the framework's covector is ``-I``, the
    book's sign).  ``mosfet_direct`` is the same device with the plain
    Euclidean sharp and a non-exact forcing.
    """
    U = _fet_cost(kind, kappa, beta, Vt, Vsat, sharpness)

    return Arrangement(
        P=parameter(constant(KERNEL_SHARP)),
        in_dim_M=0,
        out_dim_M=0,
        in_dim_N=3,
        out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -U(q, m_out, n_in)),
        label=label or f"{kind}mos",
    )


def mosfet_direct(
    kind: str = "n",
    kappa: float = 0.5,
    beta: float = 4.0,
    Vt: float = 0.3,
    Vsat: float = 0.1,
    sharpness: float = 20.0,
    label: str = "",
    parameter: Parameter = configuration,
) -> Arrangement:
    """The same MOSFET with the non-reciprocity in the FORCING, not the sharp.

    The DIRECT realization (module header; DESIGN.md §6(c)): parameter
    ``parameter(Q)`` on the plain Euclidean ``Q = R^2`` and the hand-written,
    *non-exact* one-form

        omega|_Q     = ( kappa (vgs - q0 - q1),  0 ),
        omega|_{N^-} = -dU|_{N^-},

    with ``U`` the device's cost (``_fet_cost``).  Only the port components come
    from a potential -- they are the device's constitutive law, and must be, for
    any channel model ``W``; the ``Q``-component is written down, and its
    independence of ``vds`` (against the ``q0``-dependence of ``omega|_{N^-}``)
    is precisely the failure of ``d omega = 0`` that CDLM could only obtain by
    putting the drain-sensitive component in ``ker(sharpR)``.  Step for step
    this is ``mosfet``: ``I @ omega|_Q`` here equals ``KERNEL_SHARP @ (-dU|_Q)``
    there, since ``-dU/dq1 = kappa (vgs - q0 - q1)`` and ``-dU/dq0`` is
    discarded either way.
    """
    U = _fet_cost(kind, kappa, beta, Vt, Vsat, sharpness)
    port = jax.grad(lambda q, m_out, n_in: -U(q, m_out, n_in), argnums=2)

    def omega(q: Array, m_out: Array, n_in: Array):
        eG, eS = n_in[0], n_in[2]
        vgs = eG - eS
        xi_q = jnp.stack([kappa * (vgs - q[0] - q[1]), jnp.zeros(())])
        return (xi_q, jnp.zeros(0), port(q, m_out, n_in))

    return Arrangement(
        P=parameter(euclidean(2)),
        in_dim_M=0,
        out_dim_M=0,
        in_dim_N=3,
        out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=omega,
        label=label or f"{kind}mos_direct",
    )


def node(c: float = 0.02, label: str = "node", parameter: Parameter = configuration) -> Arrangement:
    """A capacitive node ``<R^0|R^1>``: reports its voltage, integrates its current.

    State is the node voltage, ``outp(q) = q``, ``sharpR = c`` (i.e. ``dt / C``),
    zero forcing.  The only covector reaching its state is ``(d_q outp)^T xi_N =
    xi_N``, minus the total current drawn on the node (the book's sign), so
    ``Phi`` steps ``q -> q + c * xi_N = q - c * I``.
    """
    return Arrangement(
        P=parameter(constant(jnp.array([[c]]))),
        in_dim_M=0,
        out_dim_M=0,
        in_dim_N=0,
        out_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=zero_form,
        label=label,
    )


def rail(label: str = "rail", parameter: Parameter = configuration) -> Arrangement:
    """An ideal voltage source ``<R^0|R^1>``: a ``node`` with ``sharpR = 0``.

    A frozen state.  It reports its voltage and absorbs any current without
    moving, which is what makes it the circuit's power source: the composite
    cost is not bounded below, and this is where the energy enters.
    """
    return replace(
        node(0.0, parameter=parameter),
        P=parameter(constant(jnp.zeros((1, 1)))),
        label=label,
    )


# ---------------------------------------------------------------------------
# The NAND gate: four devices, three voltage-reporters, one static wiring.
# ---------------------------------------------------------------------------

# tensor order of the inner boxes
_NAND_BOXES = ["P1", "P2", "N1", "N2", "out", "mid", "vdd"]


def nand_wire() -> Arrangement:
    """The static wiring ``<R^12|R^3> -> <R^2|R^1>`` of a CMOS NAND.

    Stateless and forcing-free (a ``wire`` in the sense of CDLM
    ex.tanks_connected).  Reads the three reported node voltages
    ``(V_out, V_mid, V_dd)`` and the two outer inputs ``(a, b)``, and hands each
    device its ``(eG, eD, eS)``: ``P1 = (a, V_out, V_dd)``,
    ``P2 = (b, V_out, V_dd)``, ``N1 = (a, V_out, V_mid)``, ``N2 = (b, V_mid, 0)``.
    Reports ``V_out``.
    """

    def out_f(q_w: Array, m_out: Array) -> Array:
        return m_out[0:1]  # V_out

    def in_f(q_w: Array, m_out: Array, n_in: Array) -> Array:
        v_out, v_mid, v_dd = m_out[0], m_out[1], m_out[2]
        a, b = n_in[0], n_in[1]
        zero = jnp.array(0.0)
        return jnp.stack(
            [
                a, v_out, v_dd,   # P1
                b, v_out, v_dd,   # P2
                a, v_out, v_mid,  # N1
                b, v_mid, zero,   # N2
            ]
        )

    return Arrangement(
        P=configuration(trivial()),
        in_dim_M=12,
        out_dim_M=3,
        in_dim_N=2,
        out_dim_N=1,
        out_f=out_f,
        in_f=in_f,
        omega=zero_form,
        label="nand_wire",
    )


def nand_gate(
    c: float = 0.02, device: Callable[..., Arrangement] = mosfet,
    parameter: Parameter = configuration, **fet,
) -> Arrangement:
    """``nand_wire(P1, P2, N1, N2, node_out, node_mid, rail)``, by genuine composition.

    A ``<R^2|R^1>`` system: two input ports (the logic inputs) and one output
    port (the reported output-node voltage).  Its forcing is *not* written by
    hand -- ``compose_seq`` pulls back and sums the four device forcings at the
    port voltages the wiring delivers (DESIGN.md §2).  ``device`` selects the
    realization (``mosfet``, the trick, or ``mosfet_direct``).
    """
    boxes = [
        device("p", label="P1", parameter=parameter, **fet),
        device("p", label="P2", parameter=parameter, **fet),
        device("n", label="N1", parameter=parameter, **fet),
        device("n", label="N2", parameter=parameter, **fet),
        node(c, "node_out", parameter=parameter),
        node(c, "node_mid", parameter=parameter),
        rail(parameter=parameter),
    ]
    return replace(
        compose_seq(tensor_arrangements(boxes), nand_wire()), label="nand"
    )


def nand_state(v_out: float, v_mid: float = 0.0, vdd: float = 1.0) -> Array:
    """Initial state coordinates for ``nand_gate``: gate states zeroed, nodes/rail set.

    Layout is the tensor order ``_NAND_BOXES``: four ``R^2`` gate states, then
    ``V_out``, ``V_mid``, ``V_dd``.  Flat, as in CDLM; ``state_of`` puts it into
    the composite parameter's state shape.
    """
    return jnp.array([0.0] * 8 + [v_out, v_mid, vdd])


# ---------------------------------------------------------------------------
# The inverter, and chains of them (for depth-scaling studies).
# ---------------------------------------------------------------------------


def inverter_wire() -> Arrangement:
    """The static wiring ``<R^6|R^2> -> <R^1|R^1>`` of a CMOS inverter.

    Reads ``(V_out, V_dd)`` and the outer input ``a``; hands ``P = (a, V_out, V_dd)``
    and ``N = (a, V_out, 0)``.  Reports ``V_out``.
    """

    def in_f(q_w: Array, m_out: Array, n_in: Array) -> Array:
        v_out, v_dd, a = m_out[0], m_out[1], n_in[0]
        return jnp.stack([a, v_out, v_dd, a, v_out, jnp.array(0.0)])

    return Arrangement(
        P=configuration(trivial()),
        in_dim_M=6,
        out_dim_M=2,
        in_dim_N=1,
        out_dim_N=1,
        out_f=lambda q_w, m_out: m_out[0:1],
        in_f=in_f,
        omega=zero_form,
        label="inverter_wire",
    )


def inverter(
    c: float = 0.02, device: Callable[..., Arrangement] = mosfet,
    parameter: Parameter = configuration, **fet,
) -> Arrangement:
    """``inverter_wire(P, N, node_out, rail)``: a ``<R^1|R^1>`` system, ``Q = R^6``."""
    boxes = [
        device("p", label="P", parameter=parameter, **fet),
        device("n", label="N", parameter=parameter, **fet),
        node(c, "node_out", parameter=parameter),
        rail(parameter=parameter),
    ]
    return replace(compose_seq(tensor_arrangements(boxes), inverter_wire()), label="inv")


def inverter_state(v_out: float, vdd: float = 1.0) -> Array:
    """Initial state coordinates for ``inverter``: gate states zeroed, then ``V_out``, ``V_dd``."""
    return jnp.array([0.0] * 4 + [v_out, vdd])


def inverter_chain(
    K: int, c: float = 0.02, device: Callable[..., Arrangement] = mosfet,
    parameter: Parameter = configuration, **fet,
) -> Arrangement:
    """``chain_wire(K)(inv, ..., inv)``: ``K`` inverters in series, ``<R^1|R^1>``.

    Uses the paper's own chain wiring (sec.heat_waves; CDLM sec.spring_first_pass),
    which feeds each box from the previous one's output and the first from the outer
    input.  Depth ``K``, so it isolates *logic depth* from *component count*.
    """
    return replace(
        compose_chain([inverter(c, device=device, parameter=parameter, **fet) for _ in range(K)]),
        label=f"inv_chain({K})",
    )


def chain_state(K: int, v_out: float = 0.5, vdd: float = 1.0) -> Array:
    return jnp.concatenate([inverter_state(v_out, vdd) for _ in range(K)])


# ---------------------------------------------------------------------------
# The SR latch: two NAND gates, cross-coupled.
# ---------------------------------------------------------------------------


def latch_wire() -> Arrangement:
    """The static feedback wiring ``<R^4|R^2> -> <R^2|R^2>`` of an SR latch.

    Outer inputs ``(Sbar, Rbar)``, outer outputs ``(Q, Qbar)``.  Gate 1 is fed
    ``(Sbar, Qbar)``, gate 2 is fed ``(Rbar, Q)``.  The loop is well-posed
    because each gate's ``outp f`` reads only its own node state, so no output
    depends instantaneously on any input.
    """

    def in_f(q_w: Array, m_out: Array, n_in: Array) -> Array:
        Q, Qbar = m_out[0], m_out[1]
        Sbar, Rbar = n_in[0], n_in[1]
        return jnp.stack([Sbar, Qbar, Rbar, Q])

    return Arrangement(
        P=configuration(trivial()),
        in_dim_M=4,
        out_dim_M=2,
        in_dim_N=2,
        out_dim_N=2,
        out_f=lambda q_w, m_out: m_out,
        in_f=in_f,
        omega=zero_form,
        label="latch_wire",
    )


def sr_latch(
    c: float = 0.02, device: Callable[..., Arrangement] = mosfet,
    parameter: Parameter = configuration, **fet,
) -> Arrangement:
    """``latch_wire(nand, nand)``: a ``<R^2|R^2>`` system, by genuine composition."""
    g = nand_gate(c, device=device, parameter=parameter, **fet)
    return replace(compose_seq(tensor_arrangements([g, g]), latch_wire()), label="sr_latch")


def latch_state(q: float, qbar: float, vdd: float = 1.0) -> Array:
    """Initial state coordinates for ``sr_latch``: the two gates' states concatenated."""
    return jnp.concatenate([nand_state(q, 0.0, vdd), nand_state(qbar, 0.0, vdd)])


# ---------------------------------------------------------------------------
# Rollout.
# ---------------------------------------------------------------------------


def stepper(arr: Arrangement) -> Callable:
    """A jitted one-tick map ``(state, n_in, xi_N) -> (out_n, port_covector, state')``.

    One application of the ``Phi(arr)`` coalgebra (sec.dynamics_functor) at
    ``state``: the readout's position ``out_n``, the readout's covector field
    ``omega_N`` *evaluated at the driven inputs* -- i.e. minus the currents the
    circuit draws on its input ports (the book's sign, DESIGN.md §3), computed
    by the framework rather than by hand -- and the updated state.  States are
    the flat coordinates of ``flat_of``/``state_of``.
    """
    O = Phi(arr)

    def one(flat: Array, n_in: Array, xi_N: Array):
        state = state_of(arr, flat)
        _, fiber = O.step(state)
        (out_n, omega_N), at_pos = fiber(_IN_POS)
        _, new_state = at_pos((xi_N, n_in))
        return out_n, omega_N(n_in), flat_of(new_state)

    return jax.jit(one)


def run(
    arr: Arrangement,
    state: Array,
    inputs: Callable[[int], Sequence[float]],
    steps: int,
    load: Sequence[float] | None = None,
) -> Tuple[Array, List[Array], List[Array]]:
    """Step ``Phi(arr)`` for ``steps`` ticks, driving the input ports.

    ``inputs(t)`` supplies the outer input voltages at tick ``t``; ``load`` is the
    covector presented at the output ports (minus the current drawn by whatever
    the circuit drives -- the book's sign), default zero.  Returns
    ``(final_state, outputs, port_covectors)`` with the states flat.
    """
    one = stepper(arr)
    xi_N = jnp.zeros(arr.out_dim_N) if load is None else jnp.asarray(load, float)
    outs: List[Array] = []
    currents: List[Array] = []
    for t in range(steps):
        out_n, cur, state = one(state, jnp.asarray(inputs(t), float), xi_N)
        outs.append(out_n)
        currents.append(cur)
    return state, outs, currents
