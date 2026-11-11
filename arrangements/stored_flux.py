"""Circuits whose coils store their own flux (book Ch 6 ``sec.circuits``; plan P76 (f), (g)).

David, 2026-11-10 (session 2524b69e): "please change ch6 so that it's the ch2 version except a tangent arrangement."

The typing is Ch 2's. A capacitor is ``sec.kcl``'s: state its charge, reads out its voltage. A coil is the inertial pipe of
``ex.pipe_momentum`` with charge in place of water: it stores its flux ``phi``, receives the voltages at its two ends, has the
internal inclination ``q1 - q2`` and returns its current ``phi / L``, ``-`` to the first end and ``+`` to the second. A transformer's
winding and a motor's winding are the same part with one more port, the core's flux or the rotor's angle.

Each part is an integrand (a readout and a tangent response), stepped by a rule of ``light_tank.lift``: ``"rk4"``, ``"mid"`` or
``"euler"``. The differential equation is what the book states; Euler's step gains energy (``tests/test_stored_flux.py``).
Everything is on ``R^n``.

Claude Fable 5.2, session 2524b69e, 2026-11-10.
"""

from __future__ import annotations

import jax.numpy as jnp

from .arrangement import Arrangement, zero_form
from .light_tank import lift
from .wiring import trivial_parameter

_z = jnp.zeros(0)


def _box(P, label):
    """A part with no inputs and one output port, reading out its integrand's configuration."""
    return Arrangement(P=P, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
                       out_f=lambda q, m: q, in_f=lambda q, m, n: _z, omega=zero_form, label=label)


def _hears(P, n_ports, omega, label):
    """A part that hears ``n_ports`` values and has no output: ``omega(state, m)`` returns (inclination on its state, pushback)."""
    return Arrangement(P=P, in_dim_M=0, out_dim_M=n_ports, in_dim_N=0, out_dim_N=0,
                       out_f=lambda q, m: _z, in_f=lambda q, m, n: _z,
                       omega=lambda q, m, n: omega(q, m) + (_z,), label=label)


def tank(C: float, s0: float, rule: str, tau: float, K: int = 1) -> Arrangement:
    """Ch 2 ``sec.kcl``'s capacitor: state its charge ``s``, readout its voltage ``s/C``, rate the current that comes back."""
    return _box(lift(rule, lambda s: s / C, lambda s, xi: xi, [s0], 1, tau, K), f"tank({C:g})")


def ground_v() -> Arrangement:
    """The ground as a voltage port: no state, reads out 0."""
    return Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
                       out_f=lambda q, m: jnp.zeros(1), in_f=lambda q, m, n: _z, omega=zero_form, label="gnd")


def coil(L: float, phi0: float, rule: str, tau: float, K: int = 1) -> Arrangement:
    """A coil between two voltage ports: state its flux ``phi``, rate ``q1 - q2``, pushback ``-+ phi / L``."""
    P = lift(rule, lambda s: s, lambda s, xi: xi, [phi0], 1, tau, K)
    return _hears(P, 2, lambda phi, m: (jnp.array([m[0] - m[1]]), jnp.array([-phi[0] / L, phi[0] / L])), f"coil({L:g})")


def coils(Lmat, phi0, rule: str, tau: float, K: int = 1) -> Arrangement:
    """Coils sharing a field as one part: state the flux through each, ``phi``; coil ``k`` runs from the voltage ``m[k]`` to
    ground. Rate ``m``, pushback ``-Lmat^{-1} phi``."""
    Linv = jnp.linalg.inv(jnp.asarray(Lmat, float))
    n = Linv.shape[0]
    P = lift(rule, lambda s: s, lambda s, xi: xi, phi0, n, tau, K)
    return _hears(P, n, lambda phi, m: (m, -(Linv @ phi)), f"coils({n})")


def winding(N: float, ell: float, phi0: float, rule: str, tau: float, K: int = 1) -> Arrangement:
    """A transformer's winding, from the voltage ``m[0]`` to ground, around a core whose flux is ``m[1]``: state its flux
    ``phi``, rate ``m[0]``, current ``i = (phi - N m[1]) / ell``, pushback ``-i`` on the voltage and ``N i`` on the core's flux."""
    P = lift(rule, lambda s: s, lambda s, xi: xi, [phi0], 1, tau, K)

    def omega(phi, m):
        i = (phi[0] - N * m[1]) / ell
        return (m[:1], jnp.array([-i, N * i]))

    return _hears(P, 2, omega, f"winding(N={N:g},l={ell:g})")


def core_flux(g: float, Phi0: float, rule: str, tau: float, K: int = 1) -> Arrangement:
    """A core's flux, stored (Ch 6 ``ex.transformer``): state ``Phi``, reads it out, rate ``g`` times the load."""
    return _box(lift(rule, lambda s: s, lambda s, xi: g * xi, [Phi0], 1, tau, K), f"coreFlux(g={g:g})")


def rotor(J: float, theta0: float, p0: float, b: float, rule: str, tau: float, K: int = 1) -> Arrangement:
    """A rotor: state (angle, angular momentum), reads out its angle, rate ``(p/J, torque - b p/J)``; ``b`` is its drag."""
    P = lift(rule, lambda s: s[:1], lambda s, xi: jnp.array([s[1] / J, xi[0] - b * s[1] / J]), [theta0, p0], 1, tau, K)
    return _box(P, f"rotor(J={J:g},b={b:g})")


def motor_winding(Kc: float, L: float, phi0: float, rule: str, tau: float, K: int = 1) -> Arrangement:
    """A motor's winding, from the voltage ``m[0]`` to ground, on a rotor at the angle ``m[1]``: state its flux ``phi``, rate
    ``m[0]``, current ``i = (phi - Kc m[1]) / L``, pushback ``-i`` on the voltage and the torque ``Kc i`` on the angle."""
    P = lift(rule, lambda s: s, lambda s, xi: xi, [phi0], 1, tau, K)

    def omega(phi, m):
        i = (phi[0] - Kc * m[1]) / L
        return (m[:1], jnp.array([-i, Kc * i]))

    return _hears(P, 2, omega, f"motorWinding(K={Kc:g},L={L:g})")
