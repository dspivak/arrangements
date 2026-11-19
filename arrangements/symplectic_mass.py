"""The frictionless mass regrouped: a symplectic response and a kinetic-energy cost (book Ch 3 ``ex.symplectic_form``).

Two systems ``1 -> (1, R)``, on ``R^n`` (here ``n = 1, 2``), the admissible subcategory:

* ``mass_system(m)``: the frictionless mass of ``eqn.symplectic_form_mass``, the phase coalgebra with
  sharp ``1/m``: look-ahead ``q + tau p/m``, update ``(q + tau p/m, p + tau xi)``; out-transfer the
  identity, inclination ``0`` (uninclined).
* ``regrouped_system(m)``: the ?-box of the example, an integrator on ``R^2`` with states ``(q, p)``,
  look-ahead ``(q + tau p/m, p)`` and update ``(q, p) + tau J xi`` with the response
  ``J = [[0, -1], [1, 0]]`` (a load on ``q`` goes into ``p``, a load on ``p`` into minus ``q``), the
  inverse of the matrix of ``<(q,p),(q',p')> = q p' - p q'``; with the kinetic box's cost
  ``E(p) = p^2 / 2m`` on its ``p`` output: out-transfer the first coordinate, inclination
  ``-dE = -(p/m) dp`` (not uninclined).

They are different arrangements with the same dynamics: ``Phi`` sends both to the same integrator
(``tests/test_symplectic_mass.py``). The ``m`` in the ?-box's look-ahead and the ``m`` in ``E`` must
agree; with different masses the closed step stops preserving area (also tested).
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax.numpy as jnp
from jax import Array

from .arrangement import Arrangement, Numbers
from .coalgebra import CotCoalg, phase
from .rvect import diagonal

J = jnp.array([[0.0, -1.0], [1.0, 0.0]])


def _no_inputs(q, m_out, n_in):
    return jnp.zeros(0)


def mass_system(m: float, init: Optional[Tuple[Array, Array]] = None, kappa: Optional[float] = None) -> Arrangement:
    """The frictionless mass as a system ``1 -> (1, R)``; with ``kappa`` set, closed by a spring
    of cost ``kappa x^2 / 2`` at its look-ahead instead (a system ``1 -> 1``). ``init``: the phase state
    ``(q, p)``, two arrays of length 1 (zeros if ``None``)."""
    P = phase(diagonal(jnp.array([1.0 / m])), init=init)
    if kappa is None:
        return Arrangement(
            P=P, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
            out_f=lambda q, m_out: q, in_f=_no_inputs,
            omega=lambda q, m_out, n_in: (jnp.zeros_like(q), jnp.zeros_like(m_out), jnp.zeros_like(n_in)),
            label=f"mass(m={m})",
        )
    return Arrangement(
        P=P, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0), in_f=_no_inputs,
        omega=lambda q, m_out, n_in: (-kappa * q, jnp.zeros_like(m_out), jnp.zeros_like(n_in)),
        label=f"mass(m={m}) on spring(kappa={kappa})",
    )


def symplectic_integrator(m: float, init: Optional[Numbers] = None) -> CotCoalg:
    """The ?-box: states ``(q, p)``, look-ahead ``(q + tau p/m, p)``, update ``(q, p) + tau J xi``."""
    s0 = jnp.zeros(2) if init is None else jnp.asarray(init, dtype=float)

    def look(s, tau):
        return jnp.stack([s[0] + tau * s[1] / m, s[1]])

    return CotCoalg(
        Q_dim=2,
        init=s0,
        position=lambda s: look(s, 1.0),
        step=lambda s, xi: s + J @ xi,
        label=f"symplectic(m={m})",
        timed_step=lambda s, xi, tau: s + tau * (J @ xi),
        look_ahead=look,
    )


def regrouped_system(m: float, init: Optional[Numbers] = None, kappa: Optional[float] = None,
                     m_kinetic: Optional[float] = None) -> Arrangement:
    """The ?-box with the kinetic box ``-dE`` on its ``p`` output, transferring out ``q + tau p/m``. ``init``: the
    state ``(q, p)``, a vector of length 2 (zeros if ``None``).

    With ``kappa`` set, a spring of cost ``kappa x^2 / 2`` at that output closes it (a system ``1 -> 1``).
    ``m_kinetic`` is the mass in ``E`` (default ``m``), to test what happens when the two differ.
    """
    mk = m if m_kinetic is None else m_kinetic
    P = symplectic_integrator(m, init)

    def omega_open(x, m_out, n_in):
        xi_x = jnp.stack([jnp.zeros_like(x[1]), -x[1] / mk])      # -dE at p
        return (xi_x, jnp.zeros_like(m_out), jnp.zeros_like(n_in))

    if kappa is None:
        return Arrangement(
            P=P, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
            out_f=lambda x, m_out: x[:1], in_f=_no_inputs, omega=omega_open,
            label=f"symplectic(m={m}) with kinetic(m={mk})",
        )

    def omega_closed(x, m_out, n_in):
        xi_x = jnp.stack([-kappa * x[0], -x[1] / mk])                # spring at x, -dE at p
        return (xi_x, jnp.zeros_like(m_out), jnp.zeros_like(n_in))

    return Arrangement(
        P=P, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=0,
        out_f=lambda x, m_out: jnp.zeros(0), in_f=_no_inputs, omega=omega_closed,
        label=f"symplectic(m={m}) with kinetic(m={mk}) on spring(kappa={kappa})",
    )
