"""Arrangements: morphisms of ``arr = Para(cotcoalg, omegalens)`` (def.arr).

An arrangement ``a : M -> N`` with parameter ``P = (Q, S, u)`` (a ``cot(Q)``-coalgebra,
coalgebra.py) is a morphism ``Q (x) M -> N`` of ``omegalens``, the lens category
over the writer monad ``Omega(X) = X x Omega^1`` of one-forms (sec.forcing,
sec.forces). Writing an interface as ``M = <M^- | M^+>`` (two smooth sets, the
input and the output side), the lens morphism consists of smooth maps

    out_f : Q x M^+          -> N^+
    in_f  : Q x M^+ x N^-    -> M^-
    omega : Q x M^+ x N^-    -> Omega^1,   a one-form on Q x M^+ x N^-,

the last being the *forcing*, the arrangement's driving datum. It need not be
exact; when it comes from a potential, ``omega = dU`` (``exact(U)``), and for a
cost ``k`` one has ``U = -k``, so ``omega = -dk`` (STATUS.md §1; the sign
translation from CDLM is in STATUS.md §1).

In v1 the smooth sets are Cartesian spaces ``R^d``, recorded by their
dimensions -- the admissible subcategory of STATUS.md §3.1, not the whole
of ``SmSet``. A one-form on ``R^a x R^b x R^c`` is represented as a callable
``omega(x_a, x_b, x_c) -> (xi_a, xi_b, xi_c)`` returning the covector's components
on each factor at that point. This is the object the dynamics functor ``Phi``
(functors.py, sec.dynamics_functor) acts on; it was ``SmoothArrangement`` (with a
reactive vector space ``Q`` and a potential ``U``) in CDLM's companion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import jax
from jax import Array

from .coalgebra import CotCoalg
from .rvect import SmoothMap


OneForm = Callable[..., Tuple[Array, ...]]  # (x_1, ..., x_k) -> (xi_1, ..., xi_k)


@dataclass(frozen=True)
class Arrangement:
    """An arrangement ``a : M -> N`` with parameter ``P`` (def.arr).

    The (possibly tensored) domain ``M = <M^- | M^+>`` has input dimension
    ``in_dim_M`` and output dimension ``out_dim_M``; the codomain ``N`` has
    ``in_dim_N`` and ``out_dim_N``. The parameter ``P = (Q, S, u)`` is a
    ``cot(Q)``-coalgebra: base ``Q = R^{P.Q_dim}``, and the integrator is part of
    the arrangement.
    """

    P: CotCoalg
    in_dim_M: int
    out_dim_M: int
    in_dim_N: int
    out_dim_N: int
    out_f: SmoothMap  # (q, m_out)        -> n_out
    in_f: SmoothMap   # (q, m_out, n_in)  -> m_in
    omega: OneForm    # (q, m_out, n_in)  -> (xi_q, xi_mout, xi_nin)
    label: str = ""


def exact(U: Callable[[Array, Array, Array], Array]) -> OneForm:
    """The exact one-form ``omega = dU`` of a potential ``U : Q x M^+ x N^- -> R``.

    Every CDLM example, whose ``U_cdlm`` is a cost, becomes
    ``omega = exact(lambda q, m, n: -U_cdlm(q, m, n))`` (STATUS.md §1).
    """
    return jax.grad(U, argnums=(0, 1, 2))


def zero_form(q: Array, m_out: Array, n_in: Array) -> Tuple[Array, Array, Array]:
    """The zero one-form on ``Q x M^+ x N^-`` (a wiring's forcing)."""
    return (
        jax.numpy.zeros_like(q),
        jax.numpy.zeros_like(m_out),
        jax.numpy.zeros_like(n_in),
    )
