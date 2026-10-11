"""Lotka--Volterra predator--prey by composition in ``arr`` (CDLM ``ex.lotka_volterra``).

TARGET (CDLM, ``background/CDLM.tex`` ``ex.lotka_volterra``). Prey population ``x``,
predator population ``y``,

    x' = x (alpha - beta y),        y' = y (delta x - gamma),

with prey birth rate ``alpha``, encounter rate ``beta``, predator death rate ``gamma``,
and ``delta/beta`` predators produced per prey consumed. In the log chart
``u = log x``, ``v = log y`` this is

    u' = alpha - beta e^v,          v' = delta e^u - gamma.

THE COMPOSITION (CDLM's diagram: two species boxes wired to a stateless hunt box):

    lotka_volterra = compose_seq( prey (x) predator, hunt )

* ``prey``:     state ``u``, sharp ``+dt e^{-u}`` (CDLM's response ``e^{-u} xi``, the
                time step riding in the sharp as in ``dc_motor.py``), CDLM cost
                ``U_prey = -alpha e^u``, forcing ``omega = -dU_prey`` (STATUS.md §1),
                reporting ``x = e^u`` on its one output port.
* ``predator``: state ``v``, sharp ``-(delta/beta) dt e^{-v}`` -- the OPPOSITE sign, the
                device of ``dc_motor.py`` point 2 -- cost ``U_pred = -(gamma beta/delta)
                e^v``, forcing ``-dU_pred``, reporting ``y = e^v``.
* ``hunt``:     stateless, two input ports receiving ``(x, y)``, mass-action cost
                ``V(x, y) = beta x y``, forcing ``-dV``.

``compose_seq`` pulls the hunt's covectors ``-beta y dx``, ``-beta x dy`` back through
the reports (factors ``e^u``, ``e^v``) and adds them to the species' own; each sharp
then divides by the population, so one ``Phi`` step of the configuration coalgebra is

    u' = u + dt (alpha - beta e^v),     v' = v + dt (delta e^u - gamma),

forward Euler on the target field in the log chart (tests/test_lotka_volterra.py,
prediction 1). The composite forcing is ``-dU_total`` for
``U_total = beta e^{u+v} - alpha e^u - (gamma beta/delta) e^v`` (CDLM's), emerging from
pullback-and-sum; the module writes only the three costs above. The prey descends
``V`` while the predator ascends it, which is what the opposite sharps do.

MONOLITHIC CONTROL. ``lotka_volterra_monolithic`` is one closed box on ``R^2`` with the
block-diagonal sharp ``diag(dt e^{-u}, -(delta/beta) dt e^{-v})`` and ``U_total`` written
by hand (prediction 4: it agrees with the composite step for step).

THE TWO-STAGE PARAMETER (prediction 3). In the log chart the system is canonically
Hamiltonian with ``q = v``, ``p = u`` and the SEPARABLE

    H(u, v) = delta e^u - gamma u + beta e^v - alpha v,      v' = dH/du,  u' = -dH/dv,

the Lotka--Volterra invariant (``delta x - gamma log x + beta y - alpha log y``). Its
Hessian ``diag(delta e^u, beta e^v)`` is positive definite, so the plain configuration
step raises ``H`` every step by ``(dt^2/2) f^T Hess(H) f + O(dt^3)`` and the orbit
spirals OUTWARD -- in this chart as in population coordinates (Hessian
``diag(gamma/x^2, alpha/y^2)``, also positive definite). The two-round coalgebras
``goes_first`` and ``goes_second`` (``CotCoalg2``, ``rmk.multistage``, the shape of
``leapfrog.py``) give the SAME composed arrangement a second parameter: the predator
advances in round 1 from the forcing at ``(u, v)``, the prey finishes in round 2 from the
forcing at ``(u, v')``. That is symplectic Euler, ``v' = v + dt dH/du(u)``,
``u' = u - dt dH/dv(v')``, and read at the staggered midpoint
``u_sync = u - (dt/2)(alpha - beta e^v)`` it is Stoermer--Verlet: ``H`` stays bounded,
``O(dt)`` raw and ``O(dt^2)`` synchronized. Both parameters run through the one
functor family (``functors.Phi`` for one stage, ``pc2.Phi2`` for two).

State shapes: the composite's parameter is the tensor of its factors', so its state is
``((u, v), s_hunt)`` with ``s_hunt`` the stateless hunt's ``R^0``; the monolithic box's
state is the flat ``(u, v)``.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
from jax import Array

from .arrangement import Arrangement, exact
from .coalgebra import CotCoalg2, configuration
from .rvect import ReactiveVectorSpace, trivial
from .wiring import compose_seq, tensor_arrangements


# ---------------------------------------------------------------------------
# Reactive vector spaces: a log-population coordinate with response ``s dt e^{-u}``.
# ---------------------------------------------------------------------------


def log_population(scale: float, dt: float) -> ReactiveVectorSpace:
    """``R^1`` with the state-dependent sharp ``xi |-> scale * dt * e^{-u} xi``: CDLM's
    response ``e^{-u} xi`` (a flux of individuals divided by the population gives the
    change of log-population), times the time step and a signed scale."""
    return ReactiveVectorSpace(
        dim=1, sharp_fn=lambda q: jnp.array([[scale * dt * jnp.exp(-q[0])]])
    )


# ---------------------------------------------------------------------------
# Two-stage coalgebras for the symplectic split (prediction 3).
# ---------------------------------------------------------------------------


def goes_first(Q: ReactiveVectorSpace, init: Array = None) -> CotCoalg2:
    """Two-stage coalgebra on ``(Q, sharp_Q)``: the full configuration step
    ``s + sharp_s(xi)`` in round 1, nothing in round 2."""
    s0 = jnp.zeros(Q.dim) if init is None else jnp.asarray(init, dtype=float)
    return CotCoalg2(
        Q_dim=Q.dim, init=s0,
        read1=lambda s: s,
        advance=lambda s, xi: s + Q.apply_sharp(s, xi),
        read2=lambda mid: mid,
        finish=lambda s, mid, xi: mid,
        label="first",
        durational=lambda tau: CotCoalg2(          # the push held for the duration (book P9.5)
            Q_dim=Q.dim, init=s0, read1=lambda s: s,
            advance=lambda s, xi: s + tau * Q.apply_sharp(s, xi), read2=lambda mid: mid,
            finish=lambda s, mid, xi: mid, label=f"first@{tau}"),
    )


def goes_second(Q: ReactiveVectorSpace, init: Array = None) -> CotCoalg2:
    """Two-stage coalgebra on ``(Q, sharp_Q)``: nothing in round 1, the full
    configuration step in round 2, from the covector read at the other boxes' advanced
    positions."""
    s0 = jnp.zeros(Q.dim) if init is None else jnp.asarray(init, dtype=float)
    return CotCoalg2(
        Q_dim=Q.dim, init=s0,
        read1=lambda s: s,
        advance=lambda s, xi: s,
        read2=lambda mid: mid,
        finish=lambda s, mid, xi: mid + Q.apply_sharp(mid, xi),
        label="second",
        durational=lambda tau: CotCoalg2(          # the push held for the duration (book P9.5)
            Q_dim=Q.dim, init=s0, read1=lambda s: s, advance=lambda s, xi: s, read2=lambda mid: mid,
            finish=lambda s, mid, xi: mid + tau * Q.apply_sharp(mid, xi), label=f"second@{tau}"),
    )


# ---------------------------------------------------------------------------
# The three boxes.
# ---------------------------------------------------------------------------


def prey(alpha: float, dt: float = 0.01, parameter: Callable = configuration) -> Arrangement:
    """The prey box ``<R^0|R^0> -> <R^1_out | R^0_in>``: state ``u = log x``, sharp
    ``+dt e^{-u}``, CDLM cost ``U_prey = -alpha e^u`` entering as the forcing ``-dU_prey``,
    reporting ``x = e^u``. Alone: ``u -> u + dt alpha``."""

    def U(q: Array, m_out: Array, n_in: Array) -> Array:
        return -alpha * jnp.exp(q[0])

    return Arrangement(
        P=parameter(log_population(+1.0, dt)),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m_out: jnp.exp(q),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -U(q, m_out, n_in)),
        label=f"prey(alpha={alpha:g})",
    )


def predator(
    beta: float, gamma: float, delta: float, dt: float = 0.01, parameter: Callable = configuration,
) -> Arrangement:
    """The predator box ``<R^0|R^0> -> <R^1_out | R^0_in>``: state ``v = log y``, sharp
    ``-(delta/beta) dt e^{-v}`` (OPPOSITE sign to the prey's; ``delta/beta`` predators per
    prey consumed), CDLM cost ``U_pred = -(gamma beta/delta) e^v`` entering as ``-dU_pred``,
    reporting ``y = e^v``. Alone: ``v -> v - dt gamma``."""

    def U(q: Array, m_out: Array, n_in: Array) -> Array:
        return -(gamma * beta / delta) * jnp.exp(q[0])

    return Arrangement(
        P=parameter(log_population(-delta / beta, dt)),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m_out: jnp.exp(q),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -U(q, m_out, n_in)),
        label=f"predator(beta={beta:g},gamma={gamma:g},delta={delta:g})",
    )


def hunt(beta: float) -> Arrangement:
    """The stateless hunt box ``<R^2_out | R^0_in> -> <R^0|R^0>``: trivial parameter, no
    outer ports, mass-action cost ``V(x, y) = beta x y`` read off its two input ports (the
    reported populations), forcing ``-dV``. Its covectors ``-beta y dx``, ``-beta x dy``
    pull back through the reports to ``-beta y e^u`` on ``u`` and ``-beta x e^v`` on ``v``."""

    def V(q: Array, m_out: Array, n_in: Array) -> Array:
        return beta * m_out[0] * m_out[1]

    return Arrangement(
        P=configuration(trivial()),
        in_dim_M=0, out_dim_M=2, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -V(q, m_out, n_in)),
        label=f"hunt(beta={beta:g})",
    )


# ---------------------------------------------------------------------------
# The composed arrangement, with either parameter.
# ---------------------------------------------------------------------------


def lotka_volterra(
    alpha: float, beta: float, gamma: float, delta: float, dt: float = 0.01,
    two_stage: bool = False,
) -> Arrangement:
    """``compose_seq( prey (x) predator, hunt )``: a closed 0-ary arrangement with
    parameter on ``R^2 = (u, v)``, sharp ``diag(dt e^{-u}, -(delta/beta) dt e^{-v})``, and
    composite forcing ``-dU_total`` emerging from pullback-and-sum (module docstring).

    ``two_stage=False``: both species carry the configuration coalgebra; run it with
    ``functors.Phi``; one step is forward Euler in the log chart.
    ``two_stage=True``: the predator carries ``goes_first`` and the prey ``goes_second``;
    run it with ``pc2.Phi2``; one step is symplectic Euler (predator from the old prey,
    prey from the new predator)."""
    if two_stage:
        species = [
            prey(alpha, dt, parameter=goes_second),
            predator(beta, gamma, delta, dt, parameter=goes_first),
        ]
    else:
        species = [prey(alpha, dt), predator(beta, gamma, delta, dt)]
    return compose_seq(tensor_arrangements(species), hunt(beta))


def lotka_volterra_monolithic(
    alpha: float, beta: float, gamma: float, delta: float, dt: float = 0.01,
) -> Arrangement:
    """The control: ONE closed box on ``R^2 = (u, v)`` with the block-diagonal
    state-dependent sharp ``diag(dt e^{-u}, -(delta/beta) dt e^{-v})`` and CDLM's total
    cost ``U_total = beta e^{u+v} - alpha e^u - (gamma beta/delta) e^v`` written by hand,
    forcing ``-dU_total``. One ``Phi`` step is the same Euler step as the composite's."""

    def sharp_fn(q: Array) -> Array:
        return jnp.diag(jnp.array([dt * jnp.exp(-q[0]), -(delta / beta) * dt * jnp.exp(-q[1])]))

    def U(q: Array, m_out: Array, n_in: Array) -> Array:
        return (
            beta * jnp.exp(q[0] + q[1])
            - alpha * jnp.exp(q[0])
            - (gamma * beta / delta) * jnp.exp(q[1])
        )

    return Arrangement(
        P=configuration(ReactiveVectorSpace(dim=2, sharp_fn=sharp_fn)),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -U(q, m_out, n_in)),
        label="lotka_volterra_mono",
    )


def invariant(alpha: float, beta: float, gamma: float, delta: float) -> Callable[[Array], Array]:
    """The Lotka--Volterra invariant in the log chart,
    ``H(u, v) = delta e^u - gamma u + beta e^v - alpha v``
    (``= delta x - gamma log x + beta y - alpha log y``), conserved by the ODE."""

    def H(z: Array) -> Array:
        return delta * jnp.exp(z[0]) - gamma * z[0] + beta * jnp.exp(z[1]) - alpha * z[1]

    return H
