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
"""

from __future__ import annotations

import jax.numpy as jnp

from .arrangement import Arrangement, exact, zero_form
from .coalgebra import configuration, phase_on_cotangent
from .rvect import constant, diagonal
from .wiring import trivial_parameter

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
