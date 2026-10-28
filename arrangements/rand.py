"""Randomness: random-variable polynomials, samplers, and stochastic integrators (STATUS.md §3.12).

Two random-variable polynomials in ``poly(SmSet)``, each a ``tri``-monad and an
``(x)``-monoid (independent product), neither commutative:

* ``rand = sum_n Delta_n y^{underline n}`` (NOTATION.md "Random-variable polynomial"):
  an element ``(n, P, f)`` of ``rand(X)`` is a finite random variable, ``P`` a
  probability on ``underline n`` and ``f : underline n -> X`` the return map.
  Here ``FiniteRV(weights=P, outcomes=f)``, the outcomes stacked along a leading axis.
* ``rand_G = sum_n y^{R^n}``: an element ``(n, f)`` is the random variable
  ``f : R^n -> X`` on the standard normal seed ``gamma_n``. Here ``GaussianRV(n, ret=f)``.
  The measure sits on the directions, not in the positions; ``gamma_{n+m} =
  gamma_n (x) gamma_m`` is what makes concatenation an independent product.

Each element presents a *position* (the demand: ``P`` for ``rand``, ``n`` for
``rand_G``) and returns an outcome on a *sample* in the direction set over it (an
index in ``underline n``, a vector in ``R^n``). ``tensor`` is the independent product
and ``bind`` the ``tri``-monad's multiplication (Kleisli extension), which is how a
closed stochastic coalgebra ``S -> rand(S)`` is iterated exactly, with no sampler.

The product is not symmetric: in ``X (x) Y`` the sample space is presented as
``underline n x underline m`` (resp. ``R^n x R^m``) in that order. Every law is
symmetric under the swap; what the order decides is which part of one shared
sample each factor reads (STATUS.md §3.12).

A ``CotCoalgR`` is a parameter ``u : S -> (cot(Q) tri rand)(S)``: it presents a
position deterministically and, on a covector, returns a random variable over
next states. It tensors with ordinary ``CotCoalg``s (lifted by the unit), so the
wiring of ``wiring.py`` composes stochastic arrangements unchanged, and ``Phi``
applied to one is the stochastic dynamics functor: its state update is a random
variable. A ``Sampler`` is a ``[rand, y]``-coalgebra: at each state it holds a
sample for every demand and advances on the demand made. ``seeded`` pairs the two,
pushing along evaluation ``[rand, y] (x) rand -> y``, into an ordinary
``CotCoalg`` on the product of states: a stochastic integrator plus a seed is an
ordinary integrator.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional, Any, Callable

import jax
import jax.numpy as jnp
from jax import Array
from jax.tree_util import tree_map

from .arrangement import Arrangement
from .coalgebra import CotCoalg
from .rvect import ReactiveVectorSpace


# ---------------------------------------------------------------------------
# The two random-variable polynomials.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FiniteRV:
    """An element ``(n, P, f)`` of ``rand(X)``: ``weights`` is ``P in Delta_n``, and
    ``outcomes`` is ``f``, a pytree of ``X``-values stacked along a leading axis ``n``."""

    weights: Array
    outcomes: Any

    @property
    def n(self) -> int:
        return int(self.weights.shape[0])

    @property
    def position(self) -> Array:
        """The demand ``(n, P)`` a sampler answers with an index in ``underline n``."""
        return self.weights

    def ret(self, i) -> Any:
        """The return map at the sample ``i in underline n``."""
        return tree_map(lambda a: a[i], self.outcomes)

    @staticmethod
    def unit(x) -> "FiniteRV":
        """The unit ``y -> rand``: the certain random variable on ``underline 1``."""
        return FiniteRV(jnp.ones(1), tree_map(lambda a: jnp.asarray(a)[None], x))

    def tensor(self, other: "FiniteRV") -> "FiniteRV":
        """Independent product on ``underline n x underline m``, presented row-major:
        sample ``k`` is ``(k // m, k % m)``."""
        n, m = self.n, other.n
        left = tree_map(lambda a: jnp.repeat(a, m, axis=0), self.outcomes)
        right = tree_map(lambda a: jnp.tile(a, (n,) + (1,) * (a.ndim - 1)), other.outcomes)
        return FiniteRV(jnp.outer(self.weights, other.weights).reshape(-1), (left, right))

    def bind(self, k: Callable[[Any], "FiniteRV"]) -> "FiniteRV":
        """The ``tri``-monad multiplication after ``rand(k)``: sample ``i``, then ``k(f(i))``.
        The inner size may depend on ``i`` (positions of ``rand`` are discrete)."""
        parts = [k(self.ret(i)) for i in range(self.n)]
        weights = jnp.concatenate([w * p.weights for w, p in zip(self.weights, parts)])
        outcomes = tree_map(lambda *xs: jnp.concatenate(xs), *[p.outcomes for p in parts])
        return FiniteRV(weights, outcomes)

    def expect(self, g: Callable[[Any], Array]) -> Array:
        """The expectation of ``g`` under the pushforward law."""
        return sum(w * g(self.ret(i)) for i, w in enumerate(self.weights))


@dataclass(frozen=True)
class GaussianRV:
    """An element ``(n, f)`` of ``rand_G(X)``: the random variable ``ret : R^n -> X``
    on the standard normal seed ``gamma_n``."""

    n: int
    ret: Callable[[Array], Any]

    @property
    def position(self) -> int:
        """The demand ``n`` a sampler answers with a vector in ``R^n``."""
        return self.n

    @staticmethod
    def unit(x) -> "GaussianRV":
        """The unit ``y -> rand_G``: the certain random variable on ``R^0``."""
        return GaussianRV(0, lambda z: x)

    def tensor(self, other: "GaussianRV") -> "GaussianRV":
        """Independent product on ``R^n x R^m = R^{n+m}``, in that order."""
        n = self.n
        return GaussianRV(n + other.n, lambda z: (self.ret(z[:n]), other.ret(z[n:])))

    def bind(self, k: Callable[[Any], "GaussianRV"]) -> "GaussianRV":
        """The ``tri``-monad multiplication after ``rand_G(k)``: the seed ``R^n x R^m``.

        A smooth map ``R^n -> N`` is constant, so the inner size ``m`` does not depend
        on the sample; it is read at the zero sample."""
        n = self.n
        m = k(self.ret(jnp.zeros(n))).n
        return GaussianRV(n + m, lambda z: k(self.ret(z[:n])).ret(z[n:]))


@dataclass(frozen=True)
class CovGaussianRV:
    """An element ``(n, Sigma, f)`` of ``rand_N = sum_n PSD_n y^{R^n}``: the random
    variable ``ret : R^n -> X`` on the centered normal law ``N(0, Sigma)``, ``Sigma``
    positive semidefinite, degenerate allowed (plan P6 (d) (1)).

    Here the measure's size sits in the position, as ``P`` does for ``rand``, so a
    variance that grows with the duration is a smooth position; on ``rand_G`` it must
    sit in the return map as ``sqrt(duration)``, which is not smooth at duration zero.
    A monoid for the tensor by block sum, ``N(0, S (+) S') = N(0, S) (x) N(0, S')``.
    No ``bind``: flattening a covariance that depends on an earlier sample needs its
    square root, which is not smooth where it degenerates."""

    n: int
    cov: Array
    ret: Callable[[Array], Any]

    @property
    def position(self):
        """The demand ``(n, Sigma)``."""
        return self.cov

    @staticmethod
    def unit(x) -> "CovGaussianRV":
        """The unit ``y -> rand_N``: the certain random variable on ``R^0``."""
        return CovGaussianRV(0, jnp.zeros((0, 0)), lambda z: x)

    def tensor(self, other: "CovGaussianRV") -> "CovGaussianRV":
        """Independent product on ``R^n x R^m = R^{n+m}``, covariance the block sum."""
        n = self.n
        cov = jax.scipy.linalg.block_diag(self.cov, other.cov) if n + other.n else jnp.zeros((0, 0))
        return CovGaussianRV(n + other.n, cov, lambda z: (self.ret(z[:n]), other.ret(z[n:])))


# ---------------------------------------------------------------------------
# Stochastic parameters: (cot(Q) tri rand)-coalgebras.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CotCoalgR:
    """A ``(cot(Q) tri rand)``-coalgebra ``(Q, S, u)`` with an initial state.

    As ``CotCoalg``, except that ``step(s, xi_Q)`` returns a random variable over
    next states, of type ``kind`` (``FiniteRV`` or ``GaussianRV``).
    """

    Q_dim: int
    init: Any
    position: Callable[[Any], Array]
    step: Callable[[Any, Array], Any]
    kind: type
    label: str = ""
    timed_step: Optional[Callable[[Any, Array, Any], Any]] = None

    def step_over(self, s, xi_Q, duration=None):
        """The random variable over next states after a step of ``duration`` (book P9.5); ``None`` is
        the unit step; a part with no duration law raises."""
        if duration is None:
            return self.step(s, xi_Q)
        if self.timed_step is None:
            raise TypeError(f"part {self.label or '?'} has no duration law (timed_step is None)")
        return self.timed_step(s, xi_Q, duration)

    def position_at(self, s, duration=None):
        """The look-ahead: deterministic, and here the same at every duration."""
        return self.position(s)

    def tensor(self, other) -> "CotCoalgR":
        """The parallel product: base ``Q x Q'``, state a pair, and the independent
        product of the two steps' random variables. An ordinary ``CotCoalg`` factor
        is lifted by the unit; the two kinds do not mix."""
        other = lift(other, self.kind) if isinstance(other, CotCoalg) else other
        if other.kind is not self.kind:
            raise ValueError(f"tensor of a {self.kind.__name__} and a {other.kind.__name__} parameter")
        n = self.Q_dim

        def position(s):
            return jnp.concatenate([self.position(s[0]), other.position(s[1])])

        def step(s, xi_Q):
            return self.step(s[0], xi_Q[:n]).tensor(other.step(s[1], xi_Q[n:]))

        timed_step = None
        if self.timed_step is not None and other.timed_step is not None:
            def timed_step(s, xi_Q, tau):            # one duration, shared
                return self.timed_step(s[0], xi_Q[:n], tau).tensor(other.timed_step(s[1], xi_Q[n:], tau))

        return CotCoalgR(
            Q_dim=n + other.Q_dim, init=(self.init, other.init),
            position=position, step=step, kind=self.kind,
            label=f"({self.label} (x) {other.label})", timed_step=timed_step,
        )

    def rtensor(self, left: CotCoalg) -> "CotCoalgR":
        """``left (x) self`` for an ordinary ``left`` (called from ``CotCoalg.tensor``)."""
        return lift(left, self.kind).tensor(self)


def lift(P: CotCoalg, kind: type) -> CotCoalgR:
    """An ordinary coalgebra as a stochastic one, along the unit ``y -> rand``."""
    return CotCoalgR(
        Q_dim=P.Q_dim, init=P.init, position=P.position,
        step=lambda s, xi_Q: kind.unit(P.step(s, xi_Q)),
        kind=kind, label=P.label,
        timed_step=(lambda s, xi_Q, tau: kind.unit(P.step_over(s, xi_Q, tau))) if P.timed_step is not None else None,
    )


def kicked_configuration(Q: ReactiveVectorSpace, weights, kicks, init: Array = None) -> CotCoalgR:
    """The configuration step followed by a finite random kick:
    ``s |-> s + sharp_s(xi_Q) + kicks[i]`` with probability ``weights[i]``."""
    weights, kicks = jnp.asarray(weights, dtype=float), jnp.asarray(kicks, dtype=float)
    s0 = jnp.zeros(Q.dim) if init is None else jnp.asarray(init, dtype=float)
    return CotCoalgR(
        Q_dim=Q.dim, init=s0, position=lambda s: s,
        step=lambda s, xi_Q: FiniteRV(weights, (s + Q.apply_sharp(s, xi_Q))[None, :] + kicks),
        kind=FiniteRV, label="kicked",
    )


def euler_maruyama(Q: ReactiveVectorSpace, temperature: float, init: Array = None) -> CotCoalgR:
    """The configuration step plus a Gaussian kick of covariance ``2 T sharp_s``:

        s |-> s + sharp_s(xi_Q) + sqrt(2T) L_s z,   L_s L_s^T = sharp_s,  z ~ gamma_d.

    With ``xi_Q = -dk`` this is the Euler--Maruyama step of overdamped Langevin
    dynamics at temperature ``T`` with the response as mobility times step. The
    chain's stationary law is *not* the Gibbs law ``exp(-k/T)``; for a quadratic
    cost the bias is exact and tested (STATUS.md §3.12).
    """
    c = jnp.sqrt(2.0 * temperature)
    s0 = jnp.zeros(Q.dim) if init is None else jnp.asarray(init, dtype=float)

    def step(s, xi_Q):
        drift = s + Q.apply_sharp(s, xi_Q)
        L = jnp.linalg.cholesky(Q.sharp_at(s))
        return GaussianRV(Q.dim, lambda z: drift + c * (L @ z))

    def timed_step(s, xi_Q, tau):
        # over a duration: the push held for it, the jostling grown by its square root, which is how
        # independent kicks add up and the only scaling with a limit as steps shrink (book P9.5)
        drift = s + tau * Q.apply_sharp(s, xi_Q)
        L = jnp.linalg.cholesky(Q.sharp_at(s))
        return GaussianRV(Q.dim, lambda z: drift + c * jnp.sqrt(tau) * (L @ z))

    return CotCoalgR(Q_dim=Q.dim, init=s0, position=lambda s: s, step=step,
                     kind=GaussianRV, label=f"EM(T={temperature:g})", timed_step=timed_step)


def euler_maruyama_cov(Q: ReactiveVectorSpace, temperature: float, init: Array = None) -> CotCoalgR:
    """Euler--Maruyama on ``rand_N``: over a duration ``tau`` the random variable
    ``(d, 2 T tau sharp_s, z |-> s + tau sharp_s(xi_Q) + z)``.

    The same law as ``euler_maruyama``'s timed step at every ``tau > 0``, but smooth
    up to duration zero: the variance ``2 T tau sharp_s`` is linear in ``tau`` and sits
    in the position, and the return map is affine (plan P6 (d) (1))."""
    c = 2.0 * temperature
    s0 = jnp.zeros(Q.dim) if init is None else jnp.asarray(init, dtype=float)

    def timed_step(s, xi_Q, tau):
        drift = s + tau * Q.apply_sharp(s, xi_Q)
        return CovGaussianRV(Q.dim, c * tau * Q.sharp_at(s), lambda z: drift + z)

    return CotCoalgR(Q_dim=Q.dim, init=s0, position=lambda s: s,
                     step=lambda s, xi_Q: timed_step(s, xi_Q, 1.0),
                     kind=CovGaussianRV, label=f"EMcov(T={temperature:g})", timed_step=timed_step)


# ---------------------------------------------------------------------------
# Samplers: [rand, y]-coalgebras, and the pairing.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sampler:
    """A ``[rand, y]``-coalgebra. ``[p, y] = prod_{i in p(1)} p[i] y``, so at state
    ``t`` it holds ``draw(t, i)``, a sample in the direction set over every demand
    ``i``, and ``step(t, i)`` advances on the demand made. Whether its samples are
    distributed as the demand says is a property of the stream, checked separately."""

    init: Any
    draw: Callable[[Any, Any], Any]
    step: Callable[[Any, Any], Any]
    label: str = ""


def key_sampler(seed: int) -> Sampler:
    """A pseudo-random sampler on a JAX key: an index drawn by ``P`` for a ``rand``
    demand, a standard normal vector for a ``rand_G`` demand ``n``."""

    def draw(t, i):
        sub = jax.random.split(t)[1]
        if isinstance(i, int):
            return jax.random.normal(sub, (i,))
        return jax.random.choice(sub, i.shape[0], p=i)

    return Sampler(init=jax.random.PRNGKey(seed), draw=draw,
                   step=lambda t, i: jax.random.split(t)[0], label=f"key({seed})")


def replay_sampler(samples) -> Sampler:
    """A deterministic sampler that answers the ``t``-th demand with ``samples[t]``."""
    samples = jnp.asarray(samples)
    return Sampler(init=jnp.asarray(0), draw=lambda t, i: samples[t],
                   step=lambda t, i: t + 1, label="replay")


def seeded(P: CotCoalgR, sampler: Sampler) -> CotCoalg:
    """Pair a stochastic parameter with a sampler: along evaluation
    ``[rand, y] (x) rand -> y`` this is an ordinary ``cot(Q)``-coalgebra on ``S x T``."""

    def step(state, xi_Q):
        s, t = state
        rv = P.step(s, xi_Q)
        i = rv.position
        return (rv.ret(sampler.draw(t, i)), sampler.step(t, i))

    def timed_step(state, xi_Q, tau):
        s, t = state
        rv = P.step_over(s, xi_Q, tau)
        i = rv.position
        return (rv.ret(sampler.draw(t, i)), sampler.step(t, i))

    return CotCoalg(
        Q_dim=P.Q_dim, init=(P.init, sampler.init),
        position=lambda state: P.position(state[0]), step=step,
        label=f"{P.label}|{sampler.label}",
        timed_step=timed_step if P.timed_step is not None else None,
    )


def seed(arr: Arrangement, sampler: Sampler) -> Arrangement:
    """A stochastic arrangement with its parameter paired with ``sampler``."""
    return dataclasses.replace(arr, P=seeded(arr.P, sampler), label=f"{arr.label}|{sampler.label}")
