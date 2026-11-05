"""Coils, coupled coils and a charge in a magnetic field, as arrangements (book plan P76).

Two typings of a circuit's parts, both with Ch 2 ``sec.kcl``'s capacitor charge and voltage:

* *Draft* (the book's draft subsection ``sec.coils``): a capacitor's state is its charge ``s``, it reads out its
  voltage ``s/C`` and receives a current (configuration integrator, response ``C``); a coil is a part with two
  voltage inputs, state its flux ``lam``, response 1, internal inclination ``q1 - q2`` and pushback
  ``-+(lam + q1 - q2)/L``, the current it is about to carry.
* *Node flux*: a capacitor is the mass of Ch 2 ``sec.waves`` with its capacitance as the mass: state
  ``(phi, s)``, ``phi`` the node's flux (the time integral of its voltage) and ``s`` its charge, presented through
  ``phase_on_cotangent`` so that it reads out the flux it is headed to, ``phi + tau s/C``, and its voltage ``s/C``.
  A coil is then the spring of ``sec.springs`` on the fluxes at its two ends, stateless, cost ``(phi1-phi2)^2/2L``;
  coupled coils are one stateless part whose cost is ``1/2 lam^T L^{-1} lam`` on their branch fluxes ``lam``. A
  resistor reads the two voltages and returns its current on the flux coordinates (a damper).

A charge in a given field is a mass presented the same way (position and velocity read out), and the field a
stateless part returning ``e v x B`` on the position coordinates and nothing on the velocity ones.

A third, *proposed* (book plan §5.9 item 90; the last section of this file): the *two-port capacitor*, one box with a charge
and a flux coordinate, a load on either port adding to the charge. Ch 2's tank is a quotient of it, part by part. A resistor is
then an exact cost on the voltage ports, a coil a stateless spring on the flux ports, and a motor's rotor the same box with
moment of inertia in the capacitance slot.
"""

from __future__ import annotations

import jax.numpy as jnp

from .arrangement import Arrangement, exact, zero_form
from .coalgebra import CotCoalgK, configuration, phase_on_cotangent
from .rvect import constant, diagonal
from .wiring import compose_seq, tensor_arrangements, trivial_parameter

_z = jnp.zeros(0)


# ---------------------------------------------------------------------------
# The draft's typing: capacitor (charge -> voltage), coil with a state.
# ---------------------------------------------------------------------------


def capacitor_v(C: float, s0: float = 0.0) -> Arrangement:
    """Ch 2 ``sec.kcl``'s capacitor: state the charge ``s``, readout the voltage ``s/C``, response ``C``,
    so a current ``f`` coming back steps ``s -> s + f`` (``eqn.circuit_step``)."""
    return Arrangement(
        P=configuration(diagonal(jnp.array([float(C)])), init=jnp.array([float(s0)])),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda s, m: s / C,
        in_f=lambda s, m, n: _z,
        omega=zero_form,
        label=f"capV({C:g})",
    )


def coil_draft(L: float, lam0: float = 0.0, now: bool = False) -> Arrangement:
    """The draft's coil between two voltage ports: state its flux ``lam``, response 1, internal inclination
    ``q1 - q2``, pushback ``-+(lam + q1 - q2)/L`` (the current it is about to carry), or ``-+lam/L`` with
    ``now=True`` (the current it carries now)."""

    def omega(lam, m, n):
        cur = lam[0] / L if now else (lam[0] + m[0] - m[1]) / L
        return (jnp.array([m[0] - m[1]]), jnp.array([-cur, cur]), _z)

    return Arrangement(
        P=configuration(diagonal(jnp.array([1.0])), init=jnp.array([float(lam0)])),
        in_dim_M=0, out_dim_M=2, in_dim_N=0, out_dim_N=0,
        out_f=lambda lam, m: _z,
        in_f=lambda lam, m, n: _z,
        omega=omega,
        label=f"coilD({L:g}{',now' if now else ''})",
    )


# ---------------------------------------------------------------------------
# The node-flux typing: capacitor as a mass, coil as a spring, resistor as a damper.
# ---------------------------------------------------------------------------


def capacitor_phi(C: float, phi0: float = 0.0, s0: float = 0.0) -> Arrangement:
    """A capacitor as ``sec.waves``'s mass, capacitance in the mass slot: state ``(phi, s)``, presented as
    ``(phi + tau s/C, s)``; it reads out ``(flux headed to, voltage)``. A current coming back on the flux coordinate
    is added to the charge; a covector on the voltage coordinate reaches the charge as ``/C`` and is discarded by
    the step (``phase_on_cotangent`` drives the momentum by its position covector only)."""
    return Arrangement(
        P=phase_on_cotangent(diagonal(jnp.array([1.0 / C])),
                             init=(jnp.array([float(phi0)]), jnp.array([float(s0)]))),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=2,
        out_f=lambda p, m: jnp.array([p[0], p[1] / C]),
        in_f=lambda p, m, n: _z,
        omega=zero_form,
        label=f"capPhi({C:g})",
    )


def coils_spring(Lmat, ends) -> Arrangement:
    """Coils sharing a field as one stateless part on ``2 * len(ends)`` ports of the node-flux readout.

    ``ends`` lists, for each coil, the indices (into the ports it hears, each a (flux, voltage) pair) of its two
    ends; its branch flux is ``lam_k = phi_a - phi_b``. ``Lmat`` is the inductance matrix (self inductances on the
    diagonal, mutual off it), and the cost is the magnetic energy ``1/2 lam^T Lmat^{-1} lam``. One coil is
    ``coils_spring([[L]], [(0, 1)])``, the spring of ``sec.springs`` with constant ``1/L``."""
    Linv = jnp.linalg.inv(jnp.asarray(Lmat, float))
    n_ports = 1 + max(max(e) for e in ends)

    def energy(q, m, n):
        phis = m[0::2]
        lam = jnp.stack([phis[a] - phis[b] for a, b in ends])
        return 0.5 * lam @ Linv @ lam

    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0, out_dim_M=2 * n_ports, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m: _z,
        in_f=lambda q, m, n: _z,
        omega=exact(lambda q, m, n: -energy(q, m, n)),
        label=f"coils({len(ends)})",
    )


def resistor_damper(kappa: float) -> Arrangement:
    """A resistor of conductance ``kappa`` on two node-flux ports: it reads the voltages and returns the current
    ``-+kappa (v1 - v2)`` on the flux coordinates, nothing on the voltage ones (not exact, as drag is not)."""

    def omega(q, m, n):
        cur = kappa * (m[1] - m[3])
        return (q, jnp.array([-cur, 0.0, cur, 0.0]), _z)

    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0, out_dim_M=4, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m: _z,
        in_f=lambda q, m, n: _z,
        omega=omega,
        label=f"R({kappa:g})",
    )


def ground_phi() -> Arrangement:
    """The ground as a node-flux port: flux and voltage 0 always (state ``R^0``)."""
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=2,
        out_f=lambda q, m: jnp.zeros(2),
        in_f=lambda q, m, n: _z,
        omega=zero_form,
        label="gnd",
    )


# ---------------------------------------------------------------------------
# A charge in a given magnetic field.
# ---------------------------------------------------------------------------


def charge_mass(m: float, x0, p0, dim: int = 2) -> Arrangement:
    """A mass ``m`` in ``R^dim`` presented through ``phase_on_cotangent``: it reads out the position it is headed
    to and its velocity ``p/m`` (P58 (a)'s readout)."""
    return Arrangement(
        P=phase_on_cotangent(constant(jnp.eye(dim) / m),
                             init=(jnp.asarray(x0, float), jnp.asarray(p0, float))),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=2 * dim,
        out_f=lambda pp, mm: jnp.concatenate([pp[:dim], pp[dim:] / m]),
        in_f=lambda pp, mm, n: _z,
        omega=zero_form,
        label=f"charge(m={m:g})",
    )


def magnetic_field_2d(e: float, B: float) -> Arrangement:
    """A uniform field ``B`` (out of the plane) acting on a charge ``e`` in the plane: stateless, it reads the
    charge's (position, velocity) and returns ``e v x B = e B (v_y, -v_x)`` on the position coordinates and
    nothing on the velocity ones."""

    def omega(q, mm, n):
        vx, vy = mm[2], mm[3]
        return (q, jnp.array([e * B * vy, -e * B * vx, 0.0, 0.0]), _z)

    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0, out_dim_M=4, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, mm: _z,
        in_f=lambda q, mm, n: _z,
        omega=omega,
        label=f"B(e={e:g},B={B:g})",
    )


# ---------------------------------------------------------------------------
# A transformer as windings and a core: the magnetic wire carries the core's flux forward and its
# magnetomotive force back.
# ---------------------------------------------------------------------------


def select(n_in: int, idx) -> Arrangement:
    """A stateless wiring that hears ``n_in`` values and reads out ``m[idx]`` (copying or dropping); the covectors
    coming back to a copied value add (the transpose of copying, Kirchhoff's current law)."""
    idx = jnp.asarray(idx)
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0, out_dim_M=n_in, in_dim_N=0, out_dim_N=int(idx.shape[0]),
        out_f=lambda q, m: m[idx],
        in_f=lambda q, m, n: _z,
        omega=zero_form,
        label=f"sel{tuple(int(i) for i in idx)}",
    )


def core_junction(n_in: int, ends, N, ell, R: float) -> Arrangement:
    """The core's flux as a stateless junction: it hears ``n_in`` node-flux values (pairs), passes them through, and
    reads out besides them the core flux ``Phi`` that minimizes the windings' and the core's energy,
    ``sum_k (lam_k - N_k Phi)^2 / 2 ell_k + R Phi^2 / 2`` with ``lam_k = phi_a - phi_b`` for ``ends[k] = (a, b)``:
    ``Phi = sum_k (N_k lam_k / ell_k) / (R + sum_k N_k^2 / ell_k)``. At that ``Phi`` the magnetomotive forces
    coming back to it sum to 0, so it passes nothing back (P58 N1's capacitor-less junction)."""
    N, ell = jnp.asarray(N, float), jnp.asarray(ell, float)

    def out_f(q, m):
        phis = m[0::2]
        lam = jnp.stack([phis[a] - phis[b] for a, b in ends])
        Phi = jnp.sum(N * lam / ell) / (R + jnp.sum(N ** 2 / ell))
        return jnp.concatenate([m, Phi[None]])

    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0, out_dim_M=n_in, in_dim_N=0, out_dim_N=n_in + 1,
        out_f=out_f,
        in_f=lambda q, m, n: _z,
        omega=zero_form,
        label="core_junction",
    )


def winding(N_k: float, ell_k: float) -> Arrangement:
    """A winding of ``N_k`` turns with leakage inductance ``ell_k``: it hears its two ends' fluxes and the core's
    flux ``Phi``, cost ``(phi_a - phi_b - N_k Phi)^2 / 2 ell_k``. It returns its current on the electric fluxes and
    ``N_k`` times it, the magnetomotive force, on the core's flux."""
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0, out_dim_M=3, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m: _z,
        in_f=lambda q, m, n: _z,
        omega=exact(lambda q, m, n: -((m[0] - m[1] - N_k * m[2]) ** 2) / (2.0 * ell_k)),
        label=f"wind(N={N_k:g},l={ell_k:g})",
    )


def core(R: float) -> Arrangement:
    """The core of reluctance ``R``: hears its flux ``Phi``, cost ``R Phi^2 / 2``."""
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0, out_dim_M=1, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m: _z,
        in_f=lambda q, m, n: _z,
        omega=exact(lambda q, m, n: -R * m[0] ** 2 / 2.0),
        label=f"core(R={R:g})",
    )


# ---------------------------------------------------------------------------
# The two-port capacitor and what wires to it (book plan §5.9 item 90, proposed; moved from the book's run scripts
# notes/runs/light-12b2681e/two_port_cap.py and with_everything.py). Claude Opus 5.5, session 5c9967d2, 2026-11-05.
# ---------------------------------------------------------------------------


def two_port(C: float, s0: float, tau: float, phi0: float = 0.0, label: str = "cap2") -> Arrangement:
    """The two-port capacitor (§5.9 item 90): state ``(phi, s)``. Told ``tau`` it reads out ``(phi + tau s/C, s/C)``, the flux
    it is headed to and its voltage. A load on either port adds to the charge: ``s' = s + tau (l_phi + l_v)``,
    ``phi' = phi + tau s/C``. Unital: at ``tau = 0`` nothing moves.

    With ``f(phi, v) = v``, ``g(phi, s) = s`` it maps to Ch 2's ``tank`` by a morphism of integrators. A mass with position
    ``phi`` and momentum ``s`` is the same box, so ``rotor`` and ``magnet`` below are built from it.
    Claude Opus 5.5, session 5c9967d2, 2026-11-05."""
    P = CotCoalgK(Q_dim=2, init=jnp.array([float(phi0), float(s0)]),
                  reads=(lambda p: jnp.array([p[0] + tau * p[1] / C, p[1] / C]),),
                  advances=(lambda p, xi: jnp.array([p[0] + tau * p[1] / C, p[1] + tau * (xi[0] + xi[1])]),),
                  label=label)
    return Arrangement(P=P, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=2,
                       out_f=lambda q, m: q, in_f=lambda q, m, n: _z, omega=zero_form, label=label)


cap_two_port = two_port


def tank(C: float, s0: float, tau: float, label: str = "tank") -> Arrangement:
    """Ch 2's tank at a duration, the comparison for the two-port capacitor (§5.9 item 90): it reads out ``s/C`` and steps
    ``s' = s + tau l``. Claude Opus 5.5, session 5c9967d2, 2026-11-05."""
    P = CotCoalgK(Q_dim=1, init=jnp.array([float(s0)]), reads=(lambda p: p / C,),
                  advances=(lambda p, xi: p + tau * xi,), label=label)
    return Arrangement(P=P, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
                       out_f=lambda q, m: q, in_f=lambda q, m, n: _z, omega=zero_form, label=label)


def cost_part(n_ports: int, k, label: str) -> Arrangement:
    """A stateless part hearing ``n_ports`` values with the exact cost ``k(m)``: what comes back is ``-dk``, and nothing else
    is written (§5.9 item 90). Claude Opus 5.5, session 5c9967d2, 2026-11-05."""
    return Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=n_ports, in_dim_N=0, out_dim_N=0,
                       out_f=lambda q, m: _z, in_f=lambda q, m, n: _z,
                       omega=exact(lambda q, m, n: -k(m)), label=label)


def resistor_cost(kappa: float, port_a: int, port_b: int, n_ports: int) -> Arrangement:
    """Ch 2's resistor as an exact cost on two voltage ports, ``kappa (m_a - m_b)^2 / 2`` (§5.9 item 90). Unlike
    ``resistor_damper``, what it returns lands on the voltage ports, which the two-port capacitor adds to the charge.
    Claude Opus 5.5, session 5c9967d2, 2026-11-05."""
    return cost_part(n_ports, lambda m: 0.5 * kappa * (m[port_a] - m[port_b]) ** 2, f"R({kappa:g})")


def rotor(J: float, ell0: float, tau: float, theta0: float = 0.0) -> Arrangement:
    """A motor's rotor, retyped with a positive response (§5.9 item 90, W1): the two-port box with moment of inertia ``J`` in
    the capacitance slot, state (angle, angular momentum), reading out (angle headed to, angular speed).
    Claude Opus 5.5, session 5c9967d2, 2026-11-05."""
    return two_port(J, ell0, tau, phi0=theta0, label="rotor")


def motor_winding(K: float, L: float, b: float = 0.0) -> Arrangement:
    """The winding of a motor (§5.9 item 90, W1, W2): P76's ``winding`` with the rotor's angle in place of a core's flux, on the
    ports (flux, voltage, angle, speed), one stateless cost ``(phi - K theta)^2 / 2L``. So its load on the flux is the current
    ``i`` and on the angle the torque ``K i``, both one derivative, and no gyrator. ``b`` adds the drag ``b w^2 / 2`` on the
    speed port. Claude Opus 5.5, session 5c9967d2, 2026-11-05."""
    if b == 0.0:
        return cost_part(4, lambda m: (m[0] - K * m[2]) ** 2 / (2 * L), f"winding(K={K:g},L={L:g})")
    return cost_part(4, lambda m: (m[0] - K * m[2]) ** 2 / (2 * L) + 0.5 * b * m[3] ** 2,
                     f"winding(K={K:g},L={L:g})+drag({b:g})")


def magnet(x: float, y: float, a: float, tau: float, m: float = 1.0, I: float = 1.0,
           px: float = 0.0, py: float = 0.0, pa: float = 0.0) -> list:
    """A magnet in the plane (§5.9 item 90, W5): three two-port boxes, for its position and its angle, ``[mx, my, ma]``; each
    reads out (coordinate headed to, its rate). Claude Opus 5.5, session 5c9967d2, 2026-11-05."""
    return [two_port(m, px, tau, phi0=x, label="mx"), two_port(m, py, tau, phi0=y, label="my"),
            two_port(I, pa, tau, phi0=a, label="ma")]


def dipole_cost(mu: float) -> Arrangement:
    """Two magnets' stateless exact cost (§5.9 item 90, W5), ``U = mu (cos(a1 - a2) - 3 cos(a1 - f) cos(a2 - f)) / r^3``, ``f`` the
    bearing of the line between them, on the twelve ports of two ``magnet``s (each ``x, vx, y, vy, a, wa``). This takes the far
    field as a given formula: the field is not a part here. Claude Opus 5.5, session 5c9967d2, 2026-11-05."""

    def k(m):
        dx, dy = m[6] - m[0], m[8] - m[2]
        r = jnp.sqrt(dx * dx + dy * dy)
        f = jnp.arctan2(dy, dx)
        a1, a2 = m[4], m[10]
        return mu * (jnp.cos(a1 - a2) - 3 * jnp.cos(a1 - f) * jnp.cos(a2 - f)) / r ** 3

    return cost_part(12, k, f"dipoles(mu={mu:g})")


def core_flux(g: float, tau: float, Phi0: float = 0.0) -> Arrangement:
    """A transformer core's flux, stored (book Ch 6 ``ex.transformer``; plan P76 (b), David, 2026-10-28: "if we add sluggishness then it
    becomes more realistic and it works right?"): state ``Phi``, it reads ``Phi`` out and steps by ``tau g`` times the load, ``g`` the
    resistance of the eddy turn, so that a large ``g`` is little lag. Claude Opus 5.5, session 5c9967d2, 2026-11-05."""
    P = CotCoalgK(Q_dim=1, init=jnp.array([float(Phi0)]), reads=(lambda p: p,),
                  advances=(lambda p, xi: p + tau * g * xi,), label="coreFlux")
    return Arrangement(P=P, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
                       out_f=lambda q, m: q, in_f=lambda q, m, n: _z, omega=zero_form, label=f"coreFlux(g={g:g})")


def floating_coupler(kappa: float) -> Arrangement:
    """What joins a capacitor to two junctions (book Ch 6 ``sec.circuits``): hears ``(v_a, v_b, q)`` with the cost
    ``kappa (v_a - v_b - q)^2 / 2``, ``q`` the capacitor's voltage. With ``i = kappa (v_a - v_b - q)`` it returns ``-i`` to ``a``, ``i`` to ``b`` and
    ``i`` to the capacitor. Claude Opus 5.5, session 5c9967d2, 2026-11-05."""
    return cost_part(3, lambda m: kappa * (m[0] - m[1] - m[2]) ** 2 / 2.0, f"lead(kappa={kappa:g})")


def wire(boxes, top: Arrangement) -> Arrangement:
    """``boxes`` side by side, then ``top`` over them: ``compose_seq(tensor(boxes), top)``. Claude Opus 5.5, session 5c9967d2,
    2026-11-05."""
    inner = boxes[0] if len(boxes) == 1 else tensor_arrangements(boxes)
    return compose_seq(inner, top)
