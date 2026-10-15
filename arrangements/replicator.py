"""Replicator dynamics as ``Phi`` of a Shahshahani configuration arrangement.

The replicator equation for a *symmetric* payoff matrix ``A`` (a partnership
game),

    x_i' = x_i ((A x)_i - x^T A x),

is the dynamics of the closed arrangement (an ``arr``-scalar ``I -> I`` (CDLM's
terms, cf. test_newton.py; def.arr) whose parameter is the configuration
coalgebra ``configuration(Q)`` on the reactive vector space ``Q = R^n`` with the
*state-dependent* Shahshahani sharp

    sharpR_x(xi) = dt * (diag(x) xi - (x . xi) x),      i.e. the matrix
    sharpR_x    = dt * (diag(x) - x x^T),

and whose forcing is exact, ``omega = dU`` for the mean fitness
``U(x) = (1/2) x^T A x`` (CDLM carried the *cost* ``U_cdlm = -U``; STATUS.md §1).

Sign derivation against the code's conventions (no free choices left):
the configuration coalgebra steps ``x -> x + sharpR_x(xi_Q)``
(``coalgebra.configuration``, sec.differential_equations), and for a closed scalar the
covector triple's parameter component is ``xi_Q = omega|_x`` (sec.dynamics_functor;
CDLM eqn.bigtheta, the other two pullbacks vanish). With ``A`` symmetric,
``omega|_x = dU|_x = A x``, so one step is

    x -> x + dt * (diag(x) A x - (x^T A x) x)
       = x + dt * (x_i ((A x)_i - x^T A x))_i,

the forward-Euler replicator step. The forcing is the differential of the mean
fitness ``(1/2) x^T A x``, so the integrator *ascends* it -- the Shahshahani sharp
is positive semidefinite on the positive orthant
(``xi^T sharpR_x xi = dt * (E_x[xi^2] - E_x[xi]^2) >= 0`` for ``x`` a
probability vector, by Jensen), so ``U`` increases, i.e. fitness increases,
along the flow (Fisher's fundamental theorem, in this discretization only
approximately per step).

The time step ``dt`` is carried by the sharp (a positive scalar multiple of a
sharp is a sharp), exactly as ``rvect.euclidean(dim, eta)`` carries the
learning rate; the configuration coalgebra itself has no ``dt``.

Degeneracy and the simplex: ``sharpR_x`` is symmetric and degenerate. Its
kernel contains the constant covectors *exactly on the simplex*:

    sharpR_x(c * 1) = dt * c * x * (1 - sum(x)),

which vanishes iff ``sum(x) = 1`` (or ``c = 0``, or ``x = 0``). Dually,
``1^T sharpR_x = dt * (1 - sum(x)) x^T``, so the increment of ``sum(x)``
in one step is ``dt * (x^T A x) * (1 - sum(x))``: the simplex ``sum = 1``
is invariant (exactly in real arithmetic; in floats the per-step drift is
at rounding level and is self-correcting, since ``x^T A x > 0`` makes
``sum = 1`` attracting for positive-definite-on-the-orthant payoffs).
Positivity of the coordinates is preserved by the flow but only
approximately by Euler steps (a coordinate can overshoot 0 for large
``dt``); we keep ``dt`` small and start in the interior.

SCOPE LIMIT (of the TRICK realization): this construction covers partnership
games only. For a non-symmetric payoff matrix -- e.g. rock-paper-scissors -- the
replicator vector field is NOT a Shahshahani gradient of any potential (RPS
orbits cycle around the interior equilibrium and ascend nothing), so it does
not arise as ``Phi`` of this arrangement; ``dU|_x = (1/2)(A + A^T) x`` would
symmetrize the game, changing the dynamics. Hawk-Dove is covered despite its
textbook matrix being non-symmetric, because replicator dynamics is invariant
under column shifts ``A -> A + 1 c^T`` and every 2x2 game is column-equivalent
to a symmetric one (see the tests).

TRICK versus DIRECT. The Shahshahani construction above is CDLM's *trick*: with
only a potential to hand, a non-gradient vector field was reached by moving
the state dependence into the sharp (the same device as CDLM's Lotka--Volterra
pattern, ``reaction_network.py`` §3: "log states, monomial potential,
oppositely-SIGNED sharps"), which is exactly what confines it to the Kolmogorov
class ``x' = sharp_x dU`` -- symmetric ``A`` here. The book's arrangement
carries a forcing one-form that need not be exact (sec.forcing), so the same
dynamics is also realized DIRECTLY (``direct_replicator_arrangement``): the
parameter is the plain ``configuration(euclidean(n))`` and the forcing is the
replicator vector field itself read as a covector through the Euclidean flat,

    omega|_x = dt * (x_i ((A x)_i - x^T A x))_i,

for ANY payoff matrix ``A``, symmetric or not. This ``omega`` is not closed even
for symmetric ``A`` (its Jacobian is not symmetric), so it is ``dU`` for no
``U``; the scope limit above is a limit of the trick, not of ``arr``. For a
symmetric ``A`` the two arrangements' ``Phi`` agree step for step
(tests/test_replicator.py), since ``sharpR_x(A x)`` is that very vector.

LOG COORDINATES (cf. reaction_network, encoding 5): the same flow can be run
in the chart ``y = log x``, pushing the Shahshahani sharp forward along the
diffeomorphism (``sharp' = J sharp J^T``, ``J = diag(1/x)``):

    sharpR^log_y = dt * (diag(1/x) - 1 1^T),      x = e^y,

symmetric, and PSD on the simplex (Cauchy--Schwarz), with the SAME potential
``U = (1/2) x^T A x`` read through the chart. One step is then

    y_i -> y_i + dt * ((A x)_i - x^T A x),
    x_i -> x_i * exp(dt * ((A x)_i - x^T A x)),

the forward-Euler replicator step in log coordinates (the multiplicative-
weights update). Fractions are POSITIVE BY CONSTRUCTION at any ``dt`` --
where the standard tick can overshoot a coordinate below 0 -- but the sum
``sum(x) = 1``, exactly invariant for the standard tick, now drifts by
``O(dt^2)`` per step: the same positivity-for-conservation trade as the
Petri encodings 4 vs 5. No doubled state is needed here, unlike encoding 5,
because the replicator field IS a sharp-gradient (potential game).
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
from jax import Array

from .arrangement import Arrangement, exact
from .coalgebra import CotCoalg, configuration
from .functors import Phi
from .interpretation import trivial_omega
from .pc import PCMorphism
from .rvect import ReactiveVectorSpace, euclidean


def shahshahani(dim: int, dt: float = 1.0) -> ReactiveVectorSpace:
    """The Shahshahani reactive vector space ``(R^dim, dt * (diag(x) - x x^T))``.

    A state-dependent, symmetric, degenerate sharp (see the module docstring
    for its kernel); ``dt`` is the Euler step size, carried by the sharp as
    ``euclidean`` carries the learning rate.
    """

    def sharp_fn(x: Array) -> Array:
        return dt * (jnp.diag(x) - jnp.outer(x, x))

    return ReactiveVectorSpace(dim=dim, sharp_fn=sharp_fn)


def _symmetric_payoff(A: Array, who: str) -> Array:
    A = jnp.asarray(A, dtype=float)
    n = int(A.shape[0])
    if A.shape != (n, n) or not bool(jnp.allclose(A, A.T)):
        raise ValueError(f"{who}: payoff matrix must be square and symmetric")
    return A


def replicator_arrangement(
    A: Array, dt: float = 1.0, parameter: Callable[[ReactiveVectorSpace], CotCoalg] = configuration,
) -> Arrangement:
    """The TRICK realization: ``(configuration(R^n, sharpR^Shah), !, !, d((1/2) x^T A x)) : I -> I``.

    ``A`` must be symmetric (a partnership game); this is asserted, since for
    non-symmetric ``A`` the construction would silently symmetrize the game
    (module docstring, scope limit). For a non-symmetric ``A`` use
    ``direct_replicator_arrangement``. ``parameter`` is the coalgebra constructor
    applied to the Shahshahani space (default ``configuration``, the coalgebra
    CDLM's ``Phiconf`` semantics used).
    """
    A = _symmetric_payoff(A, "replicator_arrangement")
    n = int(A.shape[0])

    return Arrangement(
        P=parameter(shahshahani(n, dt)),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        # omega = -dU_cdlm, U_cdlm = -(1/2) x^T A x (STATUS.md §1)
        omega=exact(lambda q, m_out, n_in: 0.5 * jnp.dot(q, A @ q)),
        label="replicator",
    )


def replicator_dynamics(A: Array, dt: float = 1.0, parameter=configuration) -> PCMorphism:
    """``Phi`` of the (trick) replicator arrangement: the closed pc-morphism."""
    return Phi(replicator_arrangement(A, dt, parameter))


_IN_POS = (jnp.zeros(0), trivial_omega(0))
_IN_DIR = (jnp.zeros(0), jnp.zeros(0))


def replicator_step(O: PCMorphism, x: Array) -> Array:
    """One Euler step of the closed system ``Phi`` from state ``x``."""
    _, _, new_x = O.with_state(x).run_one(_IN_POS, lambda _o: _IN_DIR)
    return new_x


# ---------------------------------------------------------------------------
# The DIRECT realization: the replicator field as a non-exact forcing.
# ---------------------------------------------------------------------------


def replicator_field(A: Array, dt: float = 1.0) -> Callable[[Array], Array]:
    """The forward-Euler replicator increment ``x |-> dt * (x_i ((A x)_i - x^T A x))_i``
    for any square ``A``; the vector field the direct realization publishes as a covector."""
    A = jnp.asarray(A, dtype=float)

    def field(x: Array) -> Array:
        Ax = A @ x
        return dt * (x * (Ax - jnp.dot(x, Ax)))

    return field


def direct_replicator_arrangement(A: Array, dt: float = 1.0) -> Arrangement:
    """The DIRECT realization: ``(configuration(euclidean(n)), !, !, omega) : I -> I``
    with the replicator vector field itself as the forcing,

        omega(x, -, -) = (dt * (x_i ((A x)_i - x^T A x))_i, 0, 0),

    read as a covector through the Euclidean flat, so that ``sharp = id`` sends it
    back to the field and the configuration step ``x + sharp(omega)`` is the
    forward-Euler replicator step. ``omega`` is written by hand: it is not exact
    (not ``dU`` for any ``U``, even for symmetric ``A``), which the book's
    arrangement permits (sec.forcing, def.arr) and CDLM's did not -- hence CDLM's
    trick ``replicator_arrangement`` (Shahshahani sharp, potential ``(1/2) x^T A x``)
    and its scope limit to partnership games (module docstring). Here ``A`` may
    be any square payoff matrix, rock-paper-scissors included; for symmetric
    ``A`` this agrees with the trick step for step.
    """
    A = jnp.asarray(A, dtype=float)
    n = int(A.shape[0])
    if A.shape != (n, n):
        raise ValueError("direct_replicator_arrangement: payoff matrix must be square")
    field = replicator_field(A, dt)

    return Arrangement(
        P=configuration(euclidean(n)),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=lambda q, m_out, n_in: (field(q), jnp.zeros(0), jnp.zeros(0)),
        label="direct_replicator",
    )


def direct_replicator_dynamics(A: Array, dt: float = 1.0) -> PCMorphism:
    """``Phi`` of the direct replicator arrangement; step it with ``replicator_step``."""
    return Phi(direct_replicator_arrangement(A, dt))


# ---------------------------------------------------------------------------
# Log coordinates (the trick realization read through the chart y = log x).
# ---------------------------------------------------------------------------


def log_shahshahani(dim: int, dt: float = 1.0) -> ReactiveVectorSpace:
    """The pushforward of ``shahshahani`` along ``y = log x`` (module docstring):
    ``sharpR^log_y = dt * (diag(1/x) - 1 1^T)``, ``x = e^y``. Symmetric,
    state-dependent, PSD on the simplex."""

    def sharp_fn(y: Array) -> Array:
        x = jnp.exp(y)
        return dt * (jnp.diag(1.0 / x) - jnp.ones((dim, dim)))

    return ReactiveVectorSpace(dim=dim, sharp_fn=sharp_fn)


def log_replicator_arrangement(
    A: Array, dt: float = 1.0, parameter: Callable[[ReactiveVectorSpace], CotCoalg] = configuration,
) -> Arrangement:
    """The trick realization in log coordinates:
    ``(configuration(R^n, sharpR^log), !, !, d((1/2) e^y^T A e^y)) : I -> I``,
    the replicator arrangement read through the chart ``y = log x``."""
    A = _symmetric_payoff(A, "log_replicator_arrangement")
    n = int(A.shape[0])

    def U(q: Array, m_out: Array, n_in: Array) -> Array:
        x = jnp.exp(q)
        return 0.5 * jnp.dot(x, A @ x)  # = -U_cdlm (STATUS.md §1)

    return Arrangement(
        P=parameter(log_shahshahani(n, dt)),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(U),
        label="log_replicator",
    )


def log_replicator_dynamics(A: Array, dt: float = 1.0, parameter=configuration) -> PCMorphism:
    """``Phi`` of the log-coordinate replicator arrangement. The state is
    ``y = log x``; step it with ``replicator_step`` and read fractions off as
    ``exp(y)``, which are positive by construction."""
    return Phi(log_replicator_arrangement(A, dt, parameter))
