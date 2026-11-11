"""A coil moved past a coil: how moving one thing near another makes current (book plan P76 (f)).

David, 2026-11-10 (session 2524b69e): "arrangments can encode physical proximity, but nothing in our model so far exported
anyting related to that so how could moving one thing near another make current? please implement this in the IMC operad."

The typing is Ch 2's: a capacitor is ``sec.kcl``'s tank (state its charge, reads out its voltage), and a coil stores its own
flux, as ``ex.pipe_momentum``'s inertial pipe stores its water's momentum. Each part is an integrand (a readout and a tangent
response), stepped by a rule of ``light_tank.lift``.

What is exported that no circuit before exported is a position: the ``cart`` carrying one coil reads out where it is, and the
two coils' shared field, ``coils_at``, receives it on a port beside the capacitor's voltage. The ground's voltage is 0 and it is
not a box, as in the book's Ch 2 ``sec.kcl``. The mutual inductance is a function of
that position, so the field's energy ``H(lam, x) = lam^T L(x)^{-1} lam / 2`` is too, and the part returns ``-dH/dx`` to the cart
as a force, and the currents ``L(x)^{-1} lam`` to the capacitors.

The profile ``M(x) = M0 / (1 + x^2)^{3/2}`` is chosen for the run, largest when the coils are closest; it is not taken from a
source. Everything is on ``R^n``.

Claude Fable 5.2, session 2524b69e, 2026-11-10.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .arrangement import Arrangement, zero_form
from .light_tank import lift
from .wiring import compose_seq, tensor_arrangements

_z = jnp.zeros(0)


def tank(C: float, s0: float, rule: str, tau: float, K: int = 1) -> Arrangement:
    """Ch 2 ``sec.kcl``'s capacitor as an integrand: state its charge ``s``, readout its voltage ``s/C``, rate the current that
    comes back; stepped by ``rule``."""
    P = lift(rule, lambda s: s / C, lambda s, xi: xi, [s0], 1, tau, K)
    return Arrangement(P=P, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
                       out_f=lambda q, m: q, in_f=lambda q, m, n: _z, omega=zero_form, label=f"tank({C:g})")


def cart(m: float, x0: float, p0: float, rule: str, tau: float, K: int = 1) -> Arrangement:
    """A frictionless mass that reads out its position: state ``(x, p)``, readout ``x``, rate ``(p/m, the force that comes
    back)``. It is what exports where the moving coil is."""
    P = lift(rule, lambda s: s[:1], lambda s, xi: jnp.array([s[1] / m, xi[0]]), [x0, p0], 1, tau, K)
    return Arrangement(P=P, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
                       out_f=lambda q, m_: q, in_f=lambda q, m_, n: _z, omega=zero_form, label=f"cart({m:g})")


def mutual(M0: float):
    """The chosen profile of the mutual inductance against the separation ``x``."""
    return lambda x: M0 / (1.0 + x * x) ** 1.5


def inductance(L1: float, L2: float, M0: float):
    """``x -> [[L1, M(x)], [M(x), L2]]``."""
    M = mutual(M0)
    return lambda x: jnp.array([[L1, M(x)], [M(x), L2]])


def field_energy(L1: float, L2: float, M0: float):
    """``(lam, x) -> lam^T L(x)^{-1} lam / 2``, the energy of the two coils' shared field."""
    Lx = inductance(L1, L2, M0)
    return lambda lam, x: 0.5 * lam @ jnp.linalg.solve(Lx(x), lam)


def coils_at(L1: float, L2: float, M0: float, lam0, rule: str, tau: float, K: int = 1) -> Arrangement:
    """Two coils and their shared field, as the book's Ch 6 ``ex.moving_coil`` draws them: state ``(lam1, lam2)``, the flux through
    each. The first coil has both ends on the ground, so its flux never changes; the second runs from a capacitor to the ground
    and rides a cart. It hears ``(q, x)``, the capacitor's voltage and the cart's position. Inclination on its state: ``(0, q)``.
    Pushback: ``-i2`` on the voltage, ``i = L(x)^{-1} lam``, and the force ``-dH/dx`` on the position. With ``M0 = 0`` it is two
    separate coils and returns no force."""
    H = field_energy(L1, L2, M0)
    dH = jax.grad(H, argnums=(0, 1))

    def omega(lam, m, n):
        i, Hx = dH(lam, m[1])
        return (jnp.array([0.0, m[0]]), jnp.array([-i[1], -Hx]), _z)

    P = lift(rule, lambda s: s, lambda s, xi: xi, lam0, 2, tau, K)
    return Arrangement(P=P, in_dim_M=0, out_dim_M=2, in_dim_N=0, out_dim_N=0,
                       out_f=lambda q, m: _z, in_f=lambda q, m, n: _z, omega=omega,
                       label=f"coils_at(L={L1:g},{L2:g};M0={M0:g})")


def magnet_past_coil(L1=1.0, L2=1.0, M0=0.5, C=1.0, m=1.0, x0=-6.0, p0=1.5, i0=(1.0, 0.0), s0=0.0,
                     rule="rk4", tau=0.05, K=1) -> Arrangement:
    """The closed system of Ch 6 ``ex.moving_coil``: a capacitor and a cart, and over them the two coils. It starts with the
    currents ``i0`` at separation ``x0``."""
    lam0 = inductance(L1, L2, M0)(x0) @ jnp.asarray(i0, float)
    boxes = tensor_arrangements([tank(C, s0, rule, tau, K), cart(m, x0, p0, rule, tau, K)])
    return compose_seq(boxes, coils_at(L1, L2, M0, lam0, rule, tau, K))
