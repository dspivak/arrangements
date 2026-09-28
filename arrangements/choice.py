"""Discrete choice: the logit as an arrangement, and a made choice as a sample.

P25 stage (b) of the book plan, ``notes/DISCRETE-CHOICE.md``; stage (a) is
``bayes`` in ``notes/BAYES-INTEGRATOR.md``. ``n`` options, the ``i``th carrying a
real value ``v_i``, and a temperature ``T > 0``. The softmax or logit distribution
is ``p*_i = exp(v_i/T) / sum_j exp(v_j/T)``.

THE CHOOSER (``logit_arrangement``). Configuration ``Q = Delta_n subset R^n``,
state the position, and the EXACT forcing ``omega = dU`` of

    U(p) = <v, p> + T H(p),      H(p) = -sum_i p_i log p_i,

so the cost ``k = -U`` is the expected loss less the temperature times the
entropy, and the forcing's components are ``xi_i = v_i - T log p_i - T``. The
response is the Shahshahani sharp of ``replicator.py``, ``sharp_p = dt (diag(p) -
p p^T)``, so one ``coalgebra.configuration`` step is

    p_i |-> p_i + dt p_i ( (v_i - T log p_i) - sum_j p_j (v_j - T log p_j) ),

the constant ``-T`` cancelling. At ``T = 0`` this IS the forward-Euler replicator
step for the linear fitness ``v`` (``replicator_arrangement`` with the symmetric
payoff ``A = v 1^T + 1 v^T``, whose ``dU`` differs from ``v`` by a multiple of
``1``); at ``T > 0`` it is that step with the entropy term. Its rest point is
``p*``, since the increment vanishes iff ``v_i - T log p_i`` is independent of
``i``; the rest point does not involve ``dt``, only the stability does.

THE MECHANISM IS THE RESPONSE'S, not the cost's. ``1^T sharp_p = dt (1 - sum p)
p^T``, zero exactly on the simplex, so (i) a jump preserves ``sum_i p_i = 1``
exactly, not to first order, and (ii) the constant part of the forcing -- the
``-T``, and equally the Lagrange multiplier that normalizing would have
introduced -- is annihilated. THE NORMALIZATION OF THE CHOICE DISTRIBUTION IS
CONTRIBUTED BY THE RESPONSE. ``logit_arrangement_euclidean`` is the negative
control that shows it: the same cost, forcing, wiring and interface with
``rvect.euclidean`` settle instead on the UNNORMALIZED Gibbs weights
``p_i = exp(v_i/T - 1)``.

THE SAME DISTRIBUTION AS ONE BAYES STEP (``belief_arrangement``), in the dual
chart. Stage (a)'s construction applies to the categorical family itself, once the
gauge is fixed by ``lambda_n = 0`` (on all of ``R^n`` the Fisher information is
singular along ``1``, so ``G^{-1}`` does not exist -- a wrinkle the continuous
conjugate families of stage (a) did not have). With

    AA(lambda) = log(1 + sum_{i<n} exp(lambda_i)),   G = grad^2 AA = diag(mu) - mu mu^T,

the affine cost ``k(lambda) = -<t, mu(lambda)>`` for ``t_i = (v_i - v_n)/T``, the
forcing ``-dk`` with components ``G t``, and the response ``sharp = G^{-1}``, one
step at duration 1 is the translation ``lambda |-> lambda + t``; from the uniform
belief ``lambda = 0`` it lands on ``softmax(v/T)`` EXACTLY. The Legendre map's
Jacobian is ``G``, so this response pushed forward to the mean chart is
``G G^{-1} G = G``, the Shahshahani sharp on the gauge-fixed block: the two
constructions carry the SAME response in the two Legendre-dual charts. They are
not the same arrangement -- the mean chart's forcing carries the entropy term and
the natural chart's does not, and a ``configuration`` step is affine in whatever
chart its state lives in, so the chart is part of the arrangement. Hence the
asymmetry: the natural chart reaches the logit in one exact step, the mean chart
only in the limit.

A CHOICE ACTUALLY MADE (``collapsing_chooser``) is a ``(cot(Q) tri rand)``-coalgebra
(``rand.CotCoalgR`` over ``FiniteRV``): at state ``p`` it presents ``p`` and, on a
covector, returns the finite random variable with weights ``p`` and outcomes the
vertices. Each branch is a CONSTANT map, hence smooth, and the branching sits in
``rand``'s positions, so nothing leaves ``poly(SmSet)``: a SAMPLED collapse needs no
extension of the framework. ``rand.seeded`` turns it into an ordinary ``CotCoalg``
and ``FiniteRV.bind`` iterates it exactly with no sampler. It is ABSORBING -- at a
vertex ``e_i`` the weights are ``e_i`` -- so a chooser that stores its choice cannot
reconsider, where one that reports its choice and keeps the distribution as state
can. What has no place here is ``argmax`` AS A FUNCTION OF THE VALUES: off a tie it is
locally constant (the derivative of the choice in the utility gap goes to 0 as
``T -> 0``) and across a tie it jumps, so it has no smooth extension. THE JUMP IS NOT
THE OBSTRUCTION, which corrects the stage's own prediction
(``notes/DISCRETE-CHOICE.md``, head). Nor is determinism: a DETERMINISTIC chooser is
available and is the better answer -- see "Part two: choosing by basin" below, where the
tie is an unstable rest point and a chooser near one SLOWS instead of jumping. Part
one's "determinism is the obstruction" was too strong and is withdrawn there.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .arrangement import Arrangement, exact, zero_form
from .coalgebra import CotCoalg, configuration
from .functors import Phi
from .interpretation import trivial_omega
from .pc import PCMorphism
from .rand import CotCoalgR, FiniteRV
from .replicator import shahshahani
from .rvect import ReactiveVectorSpace, euclidean
from .wiring import compose_seq, trivial_parameter


# ---------------------------------------------------------------------------
# The closed-form targets.
# ---------------------------------------------------------------------------


def logit(v: Array, T: float) -> Array:
    """The softmax choice distribution ``exp(v/T) / sum exp(v/T)``."""
    return jax.nn.softmax(jnp.asarray(v, dtype=float) / T)


def gibbs_weights(v: Array, T: float) -> Array:
    """The UNNORMALIZED stationary point of the same cost under a Euclidean response,
    ``p_i = exp(v_i/T - 1)`` (module docstring)."""
    return jnp.exp(jnp.asarray(v, dtype=float) / T - 1.0)


def utility_entropy(v: Array, T: float) -> Callable[[Array], Array]:
    """The potential ``U(p) = <v, p> + T H(p)`` whose differential is the forcing.

    Finite only on the positive orthant; the constructors start in the interior.
    """
    v = jnp.asarray(v, dtype=float)

    def U(p: Array) -> Array:
        return jnp.dot(v, p) - T * jnp.sum(p * jnp.log(p))

    return U


# ---------------------------------------------------------------------------
# The chooser, and its negative control.
# ---------------------------------------------------------------------------


def _closed(P: CotCoalg, U: Callable[[Array], Array], label: str) -> Arrangement:
    """A closed ``arr``-scalar ``I -> I`` with parameter ``P`` and exact forcing ``dU``."""
    return Arrangement(
        P=P,
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: U(q)),
        label=label,
    )


def logit_arrangement(v: Array, T: float, dt: float = 1.0, init: Array = None) -> Arrangement:
    """The chooser: Shahshahani response, exact forcing ``d(<v,p> + T H(p))`` (docstring).

    ``T = 0`` drops the entropy term and gives the replicator step for the linear
    fitness ``v``. ``init`` defaults to the uniform distribution.
    """
    v = jnp.asarray(v, dtype=float)
    n = int(v.shape[0])
    p0 = jnp.full(n, 1.0 / n) if init is None else jnp.asarray(init, dtype=float)
    U = (lambda p: jnp.dot(v, p)) if T == 0.0 else utility_entropy(v, T)
    return _closed(configuration(shahshahani(n, dt), init=p0), U, f"logit(T={T:g})")


def logit_arrangement_euclidean(
    v: Array, T: float, eta: float = 1.0, init: Array = None
) -> Arrangement:
    """The negative control: the same cost, forcing, wiring and interface with the
    Euclidean response ``eta * I``. Its stationary point is ``gibbs_weights(v, T)``,
    unnormalized -- the normalization was the response's (module docstring)."""
    v = jnp.asarray(v, dtype=float)
    n = int(v.shape[0])
    p0 = jnp.full(n, 1.0 / n) if init is None else jnp.asarray(init, dtype=float)
    U = utility_entropy(v, T)
    return _closed(configuration(euclidean(n, eta), init=p0), U, f"logit-euclid(eta={eta:g})")


# ---------------------------------------------------------------------------
# The dual chart: the logit as one Bayes step on the categorical family.
# ---------------------------------------------------------------------------


def log_partition(n: int) -> Callable[[Array], Array]:
    """The gauge-fixed categorical log-partition ``AA(lambda) = log(1 + sum_{i<n} e^{lambda_i})``
    on ``R^{n-1}``; ``grad AA`` is the Legendre map onto the first ``n-1`` mean coordinates."""

    def AA(lam: Array) -> Array:
        return jax.nn.logsumexp(jnp.concatenate([lam, jnp.zeros(1)]))

    return AA


def fisher(n: int) -> Callable[[Array], Array]:
    """``G(lambda) = grad^2 AA(lambda)``, the ``(n-1) x (n-1)`` block ``diag(mu) - mu mu^T``."""
    return jax.hessian(log_partition(n))


def inverse_fisher(n: int) -> ReactiveVectorSpace:
    """The response of stage (a) for the categorical family: ``sharp = G^{-1}``."""
    G = fisher(n)
    return ReactiveVectorSpace(dim=n - 1, sharp_fn=lambda lam: jnp.linalg.inv(G(lam)))


def evidence(v: Array, T: float) -> Array:
    """The datum's sufficient statistic in the gauge, ``t_i = (v_i - v_n)/T`` (``i < n``)."""
    v = jnp.asarray(v, dtype=float)
    return (v[:-1] - v[-1]) / T


def belief_arrangement(v: Array, T: float, init: Array = None) -> Arrangement:
    """Stage (a)'s arrangement on the categorical family: natural chart, inverse-Fisher
    response, affine forcing ``-dk`` for ``k(lambda) = -<t, mu(lambda)>``. One step at
    duration 1 from ``lambda = 0`` (the uniform belief) lands on ``logit(v, T)`` exactly."""
    n = int(jnp.asarray(v).shape[0])
    t = evidence(v, T)
    AA = log_partition(n)
    lam0 = jnp.zeros(n - 1) if init is None else jnp.asarray(init, dtype=float)
    U = lambda lam: jnp.dot(t, jax.grad(AA)(lam))  # noqa: E731  -- U = -k, k affine in mu
    return _closed(configuration(inverse_fisher(n), init=lam0), U, f"belief(T={T:g})")


def belief_to_distribution(lam: Array) -> Array:
    """The Legendre map to the full distribution, ``softmax(lambda, 0)``."""
    return jax.nn.softmax(jnp.concatenate([lam, jnp.zeros(1)]))


def mean_chart_affine_arrangement(
    v: Array, T: float, dt: float = 1.0, init: Array = None
) -> Arrangement:
    """The mean chart's Euler jump on the AFFINE cost, with no entropy term: the same
    response as ``logit_arrangement`` and the same forcing as ``belief_arrangement``.
    It does NOT land on the logit, because a jump is affine in its own chart."""
    v = jnp.asarray(v, dtype=float)
    n = int(v.shape[0])
    p0 = jnp.full(n, 1.0 / n) if init is None else jnp.asarray(init, dtype=float)
    U = lambda p: jnp.dot(v / T, p)  # noqa: E731
    return _closed(configuration(shahshahani(n, dt), init=p0), U, "mean-affine")


# ---------------------------------------------------------------------------
# Running a closed chooser.
# ---------------------------------------------------------------------------


_IN_POS = (jnp.zeros(0), trivial_omega(0))
_IN_DIR = (jnp.zeros(0), jnp.zeros(0))


def choice_step(O: PCMorphism, s: Array) -> Array:
    """One step of the closed system ``Phi`` from state ``s`` (as ``replicator_step``)."""
    _, _, new_s = O.with_state(s).run_one(_IN_POS, lambda _o: _IN_DIR)
    return new_s


def settle(arr: Arrangement, steps: int, s: Array = None) -> Array:
    """Iterate ``Phi(arr)`` for ``steps`` steps from ``s`` (default the parameter's init)."""
    O = Phi(arr)
    s = arr.P.init if s is None else jnp.asarray(s, dtype=float)
    for _ in range(steps):
        s = choice_step(O, s)
    return s


def trajectory(arr: Arrangement, steps: int, s: Array = None) -> Array:
    """The states visited, ``steps + 1`` of them, starting from ``s``."""
    O = Phi(arr)
    s = arr.P.init if s is None else jnp.asarray(s, dtype=float)
    out = [s]
    for _ in range(steps):
        s = choice_step(O, s)
        out.append(s)
    return jnp.stack(out)


# ---------------------------------------------------------------------------
# A choice actually made.
# ---------------------------------------------------------------------------


def collapsing_chooser(n: int, init: Array = None) -> CotCoalgR:
    """The chooser that stores its choice: weights the current distribution, outcomes the
    ``n`` vertices. Each branch is a constant map, so this is a ``(cot(Q) tri rand)``-coalgebra
    in ``poly(SmSet)`` with no extension. It is absorbing: at ``e_i`` the weights are ``e_i``.
    """
    p0 = jnp.full(n, 1.0 / n) if init is None else jnp.asarray(init, dtype=float)
    vertices = jnp.eye(n)
    return CotCoalgR(
        Q_dim=n, init=p0, position=lambda p: p,
        step=lambda p, xi_Q: FiniteRV(p, vertices),
        kind=FiniteRV, label="collapse",
    )


def reporting_chooser(Q: ReactiveVectorSpace, init: Array = None) -> CotCoalgR:
    """The chooser that records its choice but keeps the distribution as state.

    State ``(p, e)``: the distribution and the vertex last drawn. The step moves ``p`` by the
    ordinary configuration displacement and draws with the moved weights, so the drawn vertex
    is recorded and never fed back. The next covector still moves it, so it can reconsider;
    ``collapsing_chooser``, whose weights come from the state it collapsed, cannot.
    """
    n = Q.dim
    p0 = jnp.full(n, 1.0 / n) if init is None else jnp.asarray(init, dtype=float)
    vertices = jnp.eye(n)

    def step(state, xi_Q):
        p, _last = state
        moved = p + Q.apply_sharp(p, xi_Q)
        return FiniteRV(moved, (jnp.broadcast_to(moved, (n, n)), vertices))

    return CotCoalgR(Q_dim=n, init=(p0, vertices[0]), position=lambda state: state[0],
                     step=step, kind=FiniteRV, label="report")


def choice_law(n: int, p: Array, steps: int) -> FiniteRV:
    """The exact law of ``steps`` steps of ``collapsing_chooser`` from ``p``, by
    ``FiniteRV.bind`` -- the ``tri``-monad multiplication, with no sampler at all."""
    P = collapsing_chooser(n, init=p)
    xi = jnp.zeros(n)
    rv = P.step(jnp.asarray(p, dtype=float), xi)
    for _ in range(steps - 1):
        rv = rv.bind(lambda s: P.step(s, xi))
    return rv


def soft_choice(gap: float, T: float) -> Array:
    """The binary chooser's probability of the better option at utility gap ``gap``:
    ``1/(1 + exp(-gap/T))``. Smooth in ``gap`` at every ``T > 0``; its derivative at a fixed
    positive gap goes to 0 as ``T -> 0``, and at ``gap = 0`` it is ``1/2`` for every ``T``,
    so the pointwise limit (``argmax``) has no smooth extension across the tie."""
    return jax.nn.sigmoid(jnp.asarray(gap, dtype=float) / T)


# ---------------------------------------------------------------------------
# Part two: choosing by basin (notes/DISCRETE-CHOICE.md §§6-9).
# ---------------------------------------------------------------------------
#
# Everything above fed a FIXED value for each option in from outside, so there was one rest
# point and the only way to a discrete outcome was the external coin of ``collapsing_chooser``.
# Let an option's value depend on the shares already committed -- a symmetric game's payoff ``A``
# in place of the linear ``<v, p>`` -- and the same arrangement has SEVERAL stable rest points,
# each with a basin, and the dynamics chooses with no coin and no temperature.
#
#     U(p) = (1/2) p^T A p + T H(p),    xi_i = (A p)_i - T log p_i - T,
#
# with the same Shahshahani response and the same exact forcing. One family, two knobs: ``A``, how
# an option's value depends on its own share, and ``T``, how much the chooser hedges.
#
#   * ``A = uncoupled_payoff(v)`` and ``T > 0``  -> ``logit_arrangement``, one rest point.
#   * any symmetric ``A`` and ``T = 0``          -> ``replicator_arrangement``.
#   * a coordination ``A`` and small ``T``       -> several rest points, choice by basin.
#
# For two options and ``A = diag(a_1, a_2)`` at ``T = 0`` the free coordinate obeys
# ``g(p_1) = p_1 (1-p_1) ((a_1+a_2) p_1 - a_2)``, so the interior rest point is
# ``p_1* = a_2/(a_1+a_2)`` with ``g'(p_1*) = p_1*(1-p_1*)(a_1+a_2) > 0``: UNSTABLE, hence the basin
# boundary, with the two vertices stable. With ``a = (1,2)`` that is 2/3, so every start above 2/3
# commits to option 1, whose payoff is HALF option 2's -- THIS CHOOSER IS NOT AN OPTIMIZER.
#
# Near that unstable point the jump multiplier is ``mu = 1 + dt g'(p_1*) > 1``, so a displacement
# grows as ``mu^k`` and deciding takes ``-log(delta_0)/log(mu)`` steps: a chooser near a tie SLOWS,
# it does not become discontinuous. That is what repairs Part one's overstated "determinism is the
# obstruction" -- a deterministic chooser is fine; what has no smooth extension is ``argmax`` as a
# function of the values.
#
# At any ``T > 0`` no vertex is a rest point (the entropy's ``log`` diverges at a corner while the
# payoff stays bounded). For the symmetric two-option game ``a_1 = a_2 = a``, writing
# ``p = ((1+m)/2, (1-m)/2)``, the rest points solve ``m = tanh(a m / (2T))``, so a committed pair
# exists iff ``a > 2T``: the critical temperature is ``T_c = a/2``, a pitchfork. This is mean-field
# Ising with coupling ``a/2`` -- a two-option coordination game with an entropy term IS the
# bistable switch, which is why this is the mechanism and not an analogy.


def uncoupled_payoff(v: Array) -> Array:
    """``A = v 1^T + 1 v^T``, the symmetric payoff whose ``dU`` differs from ``v`` by a multiple of
    ``1`` on the simplex -- killed by the response, so it reproduces ``logit_arrangement``. The
    negative control for coupling: one rest point at every temperature."""
    v = jnp.asarray(v, dtype=float)
    return jnp.outer(v, jnp.ones_like(v)) + jnp.outer(jnp.ones_like(v), v)


def game_potential(A: Array, T: float) -> Callable[[Array], Array]:
    """``U(p) = (1/2) p^T A p + T H(p)``; ``T = 0`` drops the entropy term."""
    A = jnp.asarray(A, dtype=float)

    def U(p: Array) -> Array:
        mean = 0.5 * jnp.dot(p, A @ p)
        return mean if T == 0.0 else mean - T * jnp.sum(p * jnp.log(p))

    return U


def game_arrangement(A: Array, T: float, dt: float = 1.0, init: Array = None) -> Arrangement:
    """The basin chooser: Shahshahani response, exact forcing ``d((1/2) p^T A p + T H(p))``.

    ``A`` must be symmetric, as for ``replicator_arrangement``: for a non-symmetric payoff the
    field is not a Shahshahani gradient of any potential and this construction would silently
    symmetrize the game (``replicator.py``, scope limit). ``init`` defaults to the barycenter,
    which for a symmetric coordination game sits exactly ON the basin boundary, so callers that
    want a decision pass a start off it.
    """
    A = jnp.asarray(A, dtype=float)
    n = int(A.shape[0])
    if A.shape != (n, n) or not bool(jnp.allclose(A, A.T)):
        raise ValueError("game_arrangement: payoff matrix must be square and symmetric")
    p0 = jnp.full(n, 1.0 / n) if init is None else jnp.asarray(init, dtype=float)
    return _closed(configuration(shahshahani(n, dt), init=p0), game_potential(A, T),
                   f"game(T={T:g})")


# ---- the closed forms of §6 ----


def basin_boundary(a: Array) -> float:
    """``p_1* = a_2/(a_1 + a_2)``, the unstable interior rest point of the two-option game
    ``A = diag(a_1, a_2)`` at ``T = 0`` -- the basin boundary."""
    a = jnp.asarray(a, dtype=float)
    return a[1] / (a[0] + a[1])


def boundary_rate(a: Array) -> float:
    """``g'(p_1*) = p_1*(1 - p_1*)(a_1 + a_2)``, positive, so the boundary is unstable."""
    a = jnp.asarray(a, dtype=float)
    p = basin_boundary(a)
    return p * (1.0 - p) * (a[0] + a[1])


def slowing_slope(a: Array, dt: float) -> float:
    """Steps to decide per unit of ``-log`` distance from the boundary: ``1/log(1 + dt g'(p_1*))``."""
    return 1.0 / float(jnp.log1p(dt * boundary_rate(a)))


def critical_temperature(a: float) -> float:
    """``T_c = a/2`` for the symmetric two-option coordination game ``A = diag(a, a)``: below it a
    committed pair of rest points, above it only the hedging one at ``(1/2, 1/2)``."""
    return a / 2.0


def committed_share(a: float, T: float) -> float:
    """The positive root ``m`` of ``m = tanh(a m / (2T))``, so the committed rest point is
    ``((1+m)/2, (1-m)/2)``; 0 above ``T_c``. Found by bisection on ``tanh(a m/(2T)) - m``."""
    if T >= critical_temperature(a):
        return 0.0
    lo, hi = 1e-14, 1.0 - 1e-14
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(jnp.tanh(a * mid / (2.0 * T))) - mid > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---- deciding, and locating a basin boundary by experiment ----


def decide(arr: Arrangement, p0: Array, max_steps: int = 40000, tol: float = 1e-10):
    """Run until one share exceeds ``1 - tol`` (or the state settles) and report
    ``(winner, steps)``; ``winner`` is ``-1`` if nothing committed within ``max_steps``."""
    O = Phi(arr)
    s = jnp.asarray(p0, dtype=float)
    for k in range(1, max_steps + 1):
        s = choice_step(O, s)
        if not bool(jnp.all(jnp.isfinite(s))):
            return -1, k
        if float(jnp.max(s)) > 1.0 - tol:
            return int(jnp.argmax(s)), k
    return -1, max_steps


def locate_boundary(A: Array, T: float, dt: float, rounds: int = 40, max_steps: int = 4000) -> float:
    """Bisect the starting share of option 1 for the outcome's flip: the experimental basin
    boundary, to be compared with ``basin_boundary``."""
    lo, hi = 1e-6, 1.0 - 1e-6  # winner at lo is option 2, at hi option 1
    for _ in range(rounds):
        mid = 0.5 * (lo + hi)
        w, _ = decide(game_arrangement(A, T, dt=dt), jnp.array([mid, 1.0 - mid]),
                      max_steps=max_steps)
        if w == 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def rest_points_2(A: Array, T: float, grid: int = 20001) -> Array:
    """The interior rest points of the two-option game: sign changes of
    ``(Ap)_1 - (Ap)_2 - T log(p_1/p_2)`` on a grid of ``p_1``, refined by bisection."""
    A = jnp.asarray(A, dtype=float)

    def h(p1):
        p = jnp.array([p1, 1.0 - p1])
        Ap = A @ p
        return float(Ap[0] - Ap[1] - T * jnp.log(p1 / (1.0 - p1)))

    xs = np.linspace(1e-9, 1.0 - 1e-9, grid)
    vals = np.array([h(x) for x in xs])
    out = []
    for i in range(grid - 1):
        if vals[i] == 0.0:
            out.append(xs[i])
        elif vals[i] * vals[i + 1] < 0.0:
            lo, hi = xs[i], xs[i + 1]
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if h(lo) * h(mid) <= 0.0:
                    hi = mid
                else:
                    lo = mid
            out.append(0.5 * (lo + hi))
    return jnp.asarray(out)


# ---------------------------------------------------------------------------
# Part three: composed with other things (notes/DISCRETE-CHOICE.md §§10-11).
# ---------------------------------------------------------------------------
#
# Composition in ``arr`` is pullback-and-sum on the FORCING and leaves the integrator -- response,
# state, readout -- untouched (``wiring.compose_seq``; ``tests/test_composition.py`` checks both
# operad legs, non-exact forcing included). So the only question a closed test leaves open is
# whether its conclusion survives an ADDED FORCING TERM. For the chooser it does not survive
# unchanged, and the way it fails is the interesting part: an outside push moves the basin boundary
# linearly, destroys bistability above ``|b| = a/2``, and -- since a vertex is an exact fixed point
# where the response vanishes -- the flip it causes is PERMANENT. That is hysteresis, and it is why
# commitment is worth the name.


def pushed_chooser(A: Array, T: float, dt: float = 1.0, init: Array = None,
                   publish: bool = True) -> Arrangement:
    """The chooser of ``game_arrangement`` made open: domain interface ``<0|1>``, so it receives
    one number ``b`` forward, and potential ``U(q) + b (q_1 - q_2)``.

    With ``publish`` its codomain is ``<0|1>`` and it sends the share of option 1 forward, so a
    downstream box can read the choice -- and, by pullback, PUSH BACK on it (retroactivity: the
    composite's forcing sums both boxes' pullbacks, so a reader of the choice also tilts it).
    Without ``publish`` the codomain is the unit and the chooser is pushed from upstream only.
    """
    A = jnp.asarray(A, dtype=float)
    n = int(A.shape[0])
    if A.shape != (n, n) or not bool(jnp.allclose(A, A.T)):
        raise ValueError("pushed_chooser: payoff matrix must be square and symmetric")
    p0 = jnp.full(n, 1.0 / n) if init is None else jnp.asarray(init, dtype=float)
    U = game_potential(A, T)
    tilt = jnp.zeros(n).at[0].set(1.0).at[-1].set(-1.0)

    return Arrangement(
        P=configuration(shahshahani(n, dt), init=p0),
        in_dim_M=0, out_dim_M=1, in_dim_N=0, out_dim_N=1 if publish else 0,
        out_f=(lambda q, m_out: q[:1]) if publish else (lambda q, m_out: jnp.zeros(0)),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: U(q) + m_out[0] * jnp.dot(tilt, q)),
        label=f"pushed(T={T:g}{'' if publish else ',mute'})",
    )


def pusher(b: float) -> Arrangement:
    """A box ``I -> <0|1>`` publishing the constant push ``b``, with the unit parameter and no
    forcing of its own: the upstream thing whose state the chooser's values depend on."""
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m_out: jnp.full(1, float(b)),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=zero_form,
        label=f"push({b:g})",
    )


def follower(eta: float = 0.3, init: float = 0.5) -> Arrangement:
    """A downstream box ``<0|1> -> I`` tracking whatever arrives: state ``y``, Euclidean response,
    potential ``-(y - m)^2 / 2``, so it descends to the incoming value."""
    return Arrangement(
        P=configuration(euclidean(1, eta), init=jnp.full(1, init)),
        in_dim_M=0, out_dim_M=1, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -0.5 * (q[0] - m_out[0]) ** 2),
        label="follower",
    )


def pushed_boundary(a: float, b: float) -> float:
    """``p_1* = 1/2 - b/a``, the moved basin boundary for ``A = diag(a,a)`` at ``T = 0`` under a
    push ``b``; bistable only while it lies in ``(0,1)``, i.e. while ``|b| < a/2``."""
    return 0.5 - b / a


def critical_push(a: float) -> float:
    """``|b| = a/2``: above it the boundary leaves the simplex and one option wins from every
    start, so the outside decides and the chooser no longer does."""
    return a / 2.0


def composite_choice_state(arr: Arrangement, steps: int, which: int = 1) -> Array:
    """Run ``Phi`` of a closed composite and return the ``which``-th factor of its paired state
    (``compose_seq`` makes the parameter ``P_f (x) P_g``, so the state is a pair)."""
    s = settle(arr, steps)
    return s[which] if isinstance(s, tuple) else s


def choice_chain(b: float, A: Array, T: float, dt: float = 1.0, init: Array = None,
                 eta: float = 0.3, y0: float = 0.5) -> Arrangement:
    """``pusher(b) -> pushed_chooser -> follower``, a closed composite ``I -> I`` in ``arr``: one box
    whose state feeds the chooser's values, and one box downstream reading its choice."""
    return compose_seq(
        compose_seq(pusher(b), pushed_chooser(A, T, dt=dt, init=init)),
        follower(eta, y0),
    )


def chain_states(s):
    """``(p, y)`` out of ``choice_chain``'s nested paired state ``((R^0, p), y)``."""
    inner, y = s
    return inner[1], y


def locate_pushed_boundary(b: float, A: Array, T: float, dt: float, rounds: int = 24,
                           max_steps: int = 3000, read_downstream: bool = False) -> float:
    """Bisect the chooser's starting share for the flip in the composite's outcome; on
    ``push_chain`` by default, on ``choice_chain`` (with a reader pushing back) if asked."""
    lo, hi = 1e-6, 1.0 - 1e-6

    def winner(p1):
        if read_downstream:
            arr = choice_chain(b, A, T, dt=dt, init=jnp.array([p1, 1.0 - p1]))
            unpack = lambda st: chain_states(st)[0]  # noqa: E731
        else:
            arr = push_chain(b, A, T, dt=dt, init=jnp.array([p1, 1.0 - p1]))
            unpack = push_states
        O = Phi(arr)
        s = arr.P.init
        for _ in range(max_steps):
            s = choice_step(O, s)
            p = unpack(s)
            if float(jnp.max(p)) > 1.0 - 1e-10:
                return int(jnp.argmax(p))
        return -1

    for _ in range(rounds):
        mid = 0.5 * (lo + hi)
        if winner(mid) == 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def push_chain(b: float, A: Array, T: float, dt: float = 1.0, init: Array = None) -> Arrangement:
    """``pusher(b) -> pushed_chooser(mute)``: a closed two-box composite, the chooser pushed from
    upstream with nothing downstream reading it. This is the case whose boundary is
    ``pushed_boundary(a, b)`` exactly; ``choice_chain`` adds a reader and the reader pushes back."""
    return compose_seq(pusher(b), pushed_chooser(A, T, dt=dt, init=init, publish=False))


def push_states(s):
    """``p`` out of ``push_chain``'s paired state ``(R^0, p)``."""
    return s[1]
