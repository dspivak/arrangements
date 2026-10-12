"""Reactive vector spaces: the response ``sharp``, as a constructor input.

A *reactive vector space* is a finite-dimensional real vector space ``Q``
together with a smooth *sharp* map

    sharp_Q : Q -> vect(Q^*, Q),

i.e. a (possibly position-dependent) bundle map from covectors to vectors;
we write ``sharp_q := sharp_Q(q) : Q^* -> Q`` for the value at ``q in Q``.
The reaction is *constant* if ``sharp_q = sharp_0`` for all ``q``, and
*nondegenerate* if every ``sharp_q`` is an isomorphism.

In the book (def:arr, sec.matter) the response is not a component of an
arrangement: it lives inside the coalgebra ``u : S -> cot(Q)(S)`` that is the
arrangement's parameter (STATUS.md §1, "the integrator is part of the
parameter"). This module therefore keeps ``ReactiveVectorSpace`` only as the
*input* to the standard coalgebra constructors ``coalgebra.configuration`` and
``coalgebra.phase`` -- the learning rate or inverse mass one hands to them -- not
as a parameter in its own right. It was CDLM's ``rvect`` (def.rvect); the
constant case (``constant``/``diagonal``/``euclidean``) recovers the physics
examples, and the position-dependent case is what Newton's method needs
(``inverse_hessian``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array


SharpFn = Callable[[Array], Array]  # q |-> (dim, dim) matrix for sharp_q : Q^* -> Q


@dataclass(frozen=True)
class ReactiveVectorSpace:
    """A reactive vector space ``(Q, sharp_Q)`` (CDLM def.rvect; here a constructor input).

    ``sharp_fn(q)`` returns the ``(dim, dim)`` matrix of ``sharp_q : Q^* -> Q``.
    """

    dim: int
    sharp_fn: SharpFn

    # ---- the reaction ----

    def sharp_at(self, q: Array) -> Array:
        """The matrix ``sharp_q : Q^* -> Q`` at the basepoint ``q``."""
        return self.sharp_fn(q)

    def apply_sharp(self, q: Array, xi: Array) -> Array:
        """Apply ``sharp_q`` to a covector ``xi in Q^*``."""
        return self.sharp_fn(q) @ xi

    # ---- the nondegenerate (paired) case ----

    def flat_at(self, q: Array) -> Array:
        """The flat map ``flat_q := sharp_q^{-1} : Q -> Q^*`` (nondegenerate case)."""
        return jnp.linalg.inv(self.sharp_fn(q))

    def pairing(self, q: Array, v: Array, vp: Array) -> Array:
        """The pairing at ``q``: ``<v, vp>_q := flat_q(v) . vp``."""
        return jnp.dot(self.flat_at(q) @ v, vp)

    # ---- monoidal structure ----

    def direct_sum(self, other: "ReactiveVectorSpace") -> "ReactiveVectorSpace":
        """Direct-sum reactive vector space with block-diagonal sharp,
        ``sharp_{(q,w)} = sharp_q (+) sharp_w``."""
        n, m = self.dim, other.dim

        def sharp_fn(q: Array) -> Array:
            q1, q2 = q[:n], q[n:]
            S = jnp.zeros((n + m, n + m))
            S = S.at[:n, :n].set(self.sharp_fn(q1))
            S = S.at[n:, n:].set(other.sharp_fn(q2))
            return S

        return ReactiveVectorSpace(dim=n + m, sharp_fn=sharp_fn)


# ---------------------------------------------------------------------------
# Constructors.
# ---------------------------------------------------------------------------


def constant(matrix: Array) -> ReactiveVectorSpace:
    """A reactive vector space with *constant* sharp given by ``matrix``."""
    matrix = jnp.asarray(matrix)
    return ReactiveVectorSpace(dim=int(matrix.shape[0]), sharp_fn=lambda q: matrix)


def diagonal(diag: Array) -> ReactiveVectorSpace:
    """Constant sharp that is diagonal with the given entries (e.g. inverse masses)."""
    diag = jnp.asarray(diag)
    return constant(jnp.diag(diag))


def euclidean(dim: int, eta: float = 1.0) -> ReactiveVectorSpace:
    """The Euclidean sharp ``eta * I``.

    With ``eta = eta_LR`` this is the constant learning-rate response whose
    configuration dynamics is vanilla gradient descent (book sec.neural_network).
    """
    return constant(eta * jnp.eye(dim))


def inverse_hessian(k: Callable[[Array], Array], dim: int) -> ReactiveVectorSpace:
    """The Newton response ``sharp_q = (T_q dk)^{-1}`` for a cost ``k : R^dim -> R``.

    ``T_q(dk)`` is the Hessian of ``k`` at ``q``; the standing hypothesis is that
    it is invertible at every ``q``. This is the prototypical *position-dependent*
    sharp. With the book's sign (``omega = -dk``, ``q' = q + sharp omega``) the
    configuration step is Newton's ``q - (T_q dk)^{-1} dk``.
    """

    hess = jax.hessian(k)
    return ReactiveVectorSpace(dim=dim, sharp_fn=lambda q: jnp.linalg.inv(hess(q)))


def trivial() -> ReactiveVectorSpace:
    """The monoidal unit: the zero reactive vector space ``R^0``."""
    return ReactiveVectorSpace(dim=0, sharp_fn=lambda q: jnp.zeros((0, 0)))


SmoothMap = Callable[..., Array]
