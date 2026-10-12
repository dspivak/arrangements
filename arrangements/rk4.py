"""Classical RK4 as a four-stage coalgebra ``pc^(4)`` (rmk.multistage, ell = 4).

EXTENSION (beyond the paper). RK4 is ONE ``CotCoalgK`` (K = 4) -- a single
``Store∘S ⇒ cot^{◁4}`` rule, the ``ell = 4`` case of the multi-stage integrator the
remark proposes -- and an arrangement with it as parameter has ``pcK.PhiK``
dynamics, an ``pc^(4)`` morphism, exactly parallel to the two-stage leapfrog story
(``leapfrog.py``). It integrates the first-order flow

    ẋ = f(x),      f(x) = sharp_x(ξ_Q(x)),

i.e. the same flow the configuration coalgebra steps (sec.differential_equations; CDLM's
``-sharp(dU)`` in the book's sign, STATUS.md §1), but with the classical four-stage
Runge--Kutta update instead of explicit Euler. The per-round "force" ``ξ_Q`` is the
framework covector -- the forcing ``omega`` -- at the emitted position (computed by
the interpretation's backward pass); applying the reactive sharp gives the rate
``k = sharp(ξ_Q)``. The four rounds are the four force evaluations

    read₁ = x                 → k₁ = f(x)
    read₂ = x + ½h·k₁          → k₂ = f(read₂)
    read₃ = x + ½h·k₂          → k₃ = f(read₃)
    read₄ = x + h·k₃           → k₄ = f(read₄)
    finish: x' = x + (h/6)(k₁ + 2k₂ + 2k₃ + k₄).

For ``ẋ = -A x`` these stages collect to ``x' = [I - hA + (hA)²/2 - (hA)³/6 +
(hA)⁴/24] x``, the degree-4 Taylor polynomial of ``e^{-hA}`` -- the signature of
genuine RK4, with global error ``O(h⁴)`` (``test_pck.py`` checks the ``h⁴`` rate).

RK4 is **non-symplectic** -- fine here: it is what the multi-stage construction
gives, and what the gyroscope blog uses; symplecticity is not required. That
``arr → pc^(4)`` is a lax monoidal functor is proved by hand from the duoidal
interchange (book repo ``notes/multiround-composes-1be7dcb7.md``); the parts of a
composite take their rounds together, so RK4 parts run as RK4 on the whole.
"""

from __future__ import annotations

import jax.numpy as jnp

from .coalgebra import CotCoalgK, gyro_kick
from .rvect import ReactiveVectorSpace


def rk4(Q: ReactiveVectorSpace, h: float = 0.1, init=None) -> CotCoalgK:
    """The classical RK4 four-stage coalgebra on ``(Q, sharp_Q)``, ``Store∘S ⇒ cot^{◁4}`` (step ``h``).

    The macro-state is the position ``x``; intermediates carry ``x`` together with the
    rates ``kᵢ`` collected so far. Each ``advance`` evaluates the rate at the *same*
    position its ``read`` emitted, so a position-dependent ``sharp`` is honored.
    """
    half = 0.5 * h

    def rate(pos, xi_Q):
        # f(pos) = sharp_pos(ξ_Q): the reactive sharp of the forcing (the book's sign).
        return Q.apply_sharp(pos, xi_Q)

    def read1(x):
        return x

    def read2(s):
        x, ks = s
        return x + half * ks[0]

    def read3(s):
        x, ks = s
        return x + half * ks[1]

    def read4(s):
        x, ks = s
        return x + h * ks[2]

    def advance1(x, xi_Q):
        k1 = rate(read1(x), xi_Q)
        return (x, (k1,))

    def advance2(s, xi_Q):
        x, ks = s
        k2 = rate(read2(s), xi_Q)
        return (x, ks + (k2,))

    def advance3(s, xi_Q):
        x, ks = s
        k3 = rate(read3(s), xi_Q)
        return (x, ks + (k3,))

    def finish(s, xi_Q):
        x, ks = s
        k1, k2, k3 = ks
        k4 = rate(read4(s), xi_Q)
        return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    s0 = jnp.zeros(Q.dim) if init is None else jnp.asarray(init, dtype=float)
    return CotCoalgK(
        Q_dim=Q.dim,
        init=s0,
        reads=(read1, read2, read3, read4),
        advances=(advance1, advance2, advance3, finish),
        label=f"rk4(h={h:g})",
        durational=lambda tau: rk4(Q, h * tau, init),   # over a duration: step h tau (book P9.5)
    )


def rk4_gyro(
    Q: ReactiveVectorSpace, h: float = 0.1, drag: float = 0.0, gamma: float = 0.0,
    J=None, gyro_block=None, init=None,
) -> CotCoalgK:
    """Classical RK4 on the gyro *phase* system, an ``pc^(4)`` coalgebra (rmk.multistage).

    The faithful-machine integrator: it carries the *same* forces as ``coalgebra.gyro``
    (springs and rod gravity in the forcing ``omega``, quadratic drag and gyroscopic
    precession in the 1-form) but steps them with the blog's classical four-stage
    Runge--Kutta instead of symplectic Euler. The phase vector field on the state
    ``(q, xi)`` is, in the book's sign,

        qdot  = sharp_q(xi) =: v
        xidot = xi_Q - drag * F(v) - gamma (.) (J v),

    with ``xi_Q`` the framework covector (the forcing, ``-dU`` for a cost ``U``)
    evaluated by the interpretation at each stage's position (so it includes any
    open-port forcing), ``F`` the per-gyro quadratic drag (``quadratic_drag_kick``), and
    ``gamma`` a scalar or per-component vector. The four ``pc^(4)`` rounds are the four
    force evaluations ``k1..k4``; ``finish`` does ``(q,xi) + (h/6)(k1 + 2k2 + 2k3 + k4)``.

    Non-symplectic (RK4 is); ``drag``/``gamma`` may be traced, ``J``/``gyro_block`` are
    Python-time structural choices. State and forces mirror ``gyro``; only the
    time-stepping differs. Like ``gyro`` this is the TRICK realization (the
    velocity-dependent 1-forms live in the step); ``rk4_gyro_on_cotangent`` is the
    direct one.
    """
    half = 0.5 * h

    def deriv(q_s, xi_s, xi_Q):
        """The phase derivative ``(qdot, xidot)`` at a stage state, given ``xi_Q``."""
        v = Q.apply_sharp(q_s, xi_s)
        return (v, xi_Q + gyro_kick(v, gamma, J, drag, gyro_block))

    from dataclasses import replace
    return replace(_rk4_phase(Q, h, deriv, init, label=f"rk4_gyro(h={h:g})"),
                   durational=lambda tau: rk4_gyro(Q, h * tau, drag, gamma, J, gyro_block, init))


def rk4_gyro_on_cotangent(Q: ReactiveVectorSpace, h: float = 0.1, init=None) -> CotCoalgK:
    """RK4 on the phase system with base ``T^*Q`` (the DIRECT realization, cf.
    ``coalgebra.phase_on_cotangent``): the state ``(q, xi)`` presents ``(q, xi)`` itself
    at each stage, so the arrangement's forcing on ``Q x Q^*`` may depend on the
    momentum, and the phase derivative is ``(sharp_q(xi), xi_Q|_Q)``. With the gyro
    and drag terms written as a forcing this agrees with ``rk4_gyro`` step for step
    (tests/test_rk4_gyro.py)."""
    n = Q.dim

    def deriv(q_s, xi_s, xi_Qp):
        return (Q.apply_sharp(q_s, xi_s), xi_Qp[:n])

    from dataclasses import replace
    return replace(_rk4_phase(Q, h, deriv, init, label=f"rk4_on_T*Q(h={h:g})", present=lambda q, xi: jnp.concatenate([q, xi])),
                   durational=lambda tau: rk4_gyro_on_cotangent(Q, h * tau, init))


def _rk4_phase(Q, h, deriv, init, label, present=lambda q, xi: q):
    """The four RK4 rounds on a phase state, given the stage derivative and what a stage presents."""
    half = 0.5 * h

    def axpy(base, a, k):
        (q, xi), (kq, kxi) = base, k
        return (q + a * kq, xi + a * kxi)

    def read1(base):
        return present(*base)

    def read2(mid):
        base, ks = mid
        return present(*axpy(base, half, ks[0]))

    def read3(mid):
        base, ks = mid
        return present(*axpy(base, half, ks[1]))

    def read4(mid):
        base, ks = mid
        return present(*axpy(base, h, ks[2]))

    def advance1(base, xi_Q):
        k1 = deriv(base[0], base[1], xi_Q)
        return (base, (k1,))

    def advance2(mid, xi_Q):
        base, ks = mid
        s = axpy(base, half, ks[0])
        return (base, ks + (deriv(s[0], s[1], xi_Q),))

    def advance3(mid, xi_Q):
        base, ks = mid
        s = axpy(base, half, ks[1])
        return (base, ks + (deriv(s[0], s[1], xi_Q),))

    def finish(mid, xi_Q):
        base, ks = mid
        s = axpy(base, h, ks[2])
        k4 = deriv(s[0], s[1], xi_Q)
        k1, k2, k3 = ks
        q0, xi0 = base
        new_q = q0 + (h / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        new_xi = xi0 + (h / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        return (new_q, new_xi)

    s0 = (jnp.zeros(Q.dim), jnp.zeros(Q.dim)) if init is None else init
    Q_dim = present(jnp.zeros(Q.dim), jnp.zeros(Q.dim)).shape[0]
    return CotCoalgK(
        Q_dim=Q_dim,
        init=s0,
        reads=(read1, read2, read3, read4),
        advances=(advance1, advance2, advance3, finish),
        label=label,
    )
