"""Changing coils on the existing field interface (book P76 (j)).

In the category of smooth-interface arrangements with K interaction rounds,
``circuit`` is a composite of a capacitor and a winding. Its exterior is exactly
``magnets.FIELDS``. ``magnets.placed`` routes their fields without knowing their
shapes, inductances, or capacitances. No pair coefficients are entered here.

The K rounds approximate implicit midpoint and the simultaneous current/flux
constraint by Picard iteration. This needs a contraction at the chosen geometry
and duration. Initial currents must match total fluxes and external fluxes;
zero currents and fluxes are always consistent. Finite K is not an exact solve.
Fixed circuits are checked against the coupled-inductor equations. Moving
circuits require separate validation. Units are those of magnets.py, per length.

GPT-6, d5e53279, 2026-11-11. Predictions and evidence in the book repository,
notes/runs/p76j-d5e53279/RESULTS.md.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .coalgebra import CotCoalgK
from .magnets import FIELDS, MAGNETIC, g, nothing, placed
from .smooth import R, SmoothArrangement, compose_seq, include, tensor, zero_form
from .stored_flux import tank
from .wiring import trivial_parameter


def winding(L, h, tau=0.05, K=16, phi0=0.0, current0=0.0, turns=1.0):
    """One winding, with fields and one voltage input; state (total flux, current).

    Configuration (trial flux, trial current, 0). The auxiliary coordinate's
    inclination carries received flux; its readout is always zero. Loads on
    current are ignored by this integrator. The received-field form still pulls
    back onto the carts' geometry through the unchanged field arrangement.
    """
    if L <= 0 or h <= 0 or tau < 0 or K < 2:
        raise ValueError("need L,h > 0, tau >= 0 and K >= 2")
    e = jnp.array([h, 0.0])
    s0 = jnp.array([phi0, current0], dtype=float)
    read = lambda s: jnp.array([s[0], s[1], 0.0])

    def improve(s, trial, xi):
        return jnp.array([s[0] + tau * xi[0] / 2,
                          (trial[0] - xi[2]) / L])

    def first(s, xi):
        return s, improve(s, s, xi)

    def advance(m, xi):
        return m[0], improve(m[0], m[1], xi)

    def last(m, xi):
        return 2 * improve(m[0], m[1], xi) - m[0]

    P = CotCoalgK(Q_dim=3, init=s0,
                   reads=(read,) + (lambda m: read(m[1]),) * (K - 1),
                   advances=(first,) + (advance,) * (K - 2) + (last,),
                   label=f"fluxMid(K={K})")

    def flux(fields):
        return turns * (fields[MAGNETIC](e) - fields[MAGNETIC](-e))

    def omega(plot, u):
        q, _m, n = plot(u)
        # Coefficients frozen at u: this is a 1-form, not the differential
        # of a product of these coefficients with the coordinate functions.
        coefficients = jnp.array([n[2][0], -q[1], flux(n), 0.5 * q[1]])

        def coordinates(v):
            vq, _vm, vn = plot(v)
            return jnp.array([vq[0], vn[2][0], vq[2], flux(vn)])

        _, pull = jax.vjp(coordinates, u)
        return pull(coefficients)[0]

    return SmoothArrangement(
        P=P, M_in=(), M_out=(), N_in=FIELDS + (R(1),), N_out=FIELDS,
        out_f=lambda q, m: (lambda r: turns * q[1] * (g(r - e) - g(r + e)), nothing),
        in_f=lambda q, m, n: (), omega=omega,
        label=f"fieldWinding(L={L:g},h={h:g},N={turns:g})")


def circuit_wire():
    """Connect a capacitor and winding; expose only the body's field ports."""
    return SmoothArrangement(
        P=trivial_parameter(), M_in=FIELDS + (R(1),), M_out=(R(1),) + FIELDS,
        N_in=FIELDS, N_out=FIELDS, out_f=lambda q, m: m[1:],
        in_f=lambda q, m, n: n + (m[0],), omega=zero_form, label="circuit")


def circuit(L=1.0, C=1.0, h=0.5, charge0=0.0, tau=0.05, K=16,
            phi0=0.0, current0=0.0, turns=1.0):
    """A capacitor wired to a winding; a body on magnets.FIELDS."""
    if C <= 0:
        raise ValueError("capacitance must be positive")
    cap = include(tank(C, charge0, "mid", tau, K))
    coil = winding(L, h, tau, K, phi0, current0, turns)
    return compose_seq(tensor([cap, coil]), circuit_wire())


def fixed_circuits(bodies, places):
    """Existing field routing, with zero fields arriving from outside."""
    composite = compose_seq(tensor(list(bodies)), placed(places))
    cap = SmoothArrangement(
        P=trivial_parameter(), M_in=FIELDS, M_out=FIELDS, N_in=(), N_out=(),
        out_f=lambda q, m: (), in_f=lambda q, m, n: (nothing, nothing),
        omega=zero_form, label="noExternalField")
    return compose_seq(composite, cap)
