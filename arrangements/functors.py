"""The cotangent functor and the dynamics functor ``Phi : arr -> pc`` (sec.dynamics_functor).

There is **one** dynamics functor (NOTATION.md "Dynamics functor"): the
integrator is the coalgebra ``u`` inside the arrangement's parameter (def:arr),
so ``Phi`` takes no integrator argument. CDLM's ``Phiconf`` and ``Phiphase`` are
``Phi`` applied to arrangements whose parameter is ``coalgebra.configuration(Q)``
or ``coalgebra.phase(Q)``; they are different arrangements, not two readings of
one. ``Phi`` composes the state-free interpretation (interpretation.py) at the
position the coalgebra presents with the coalgebra's step.

This module also exposes the cotangent functor ``cot`` on ``R^d``
(sec.poly_smoothset), used to interpret interfaces, and ``phi_laxator``, the
comparison map of the *lax* monoidal structure of ``Phi`` on objects, by which
one combines several coalgebras before post-composing a wiring (CDLM
sec.spring_second_pass; the book's Ch 4).
"""

from __future__ import annotations

from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from .arrangement import Arrangement
from .interpretation import interpretation
from .pc import PCMorphism
from .polynomial import Cot, DirichletProduct, Poly, PolyMap, nest_left, unnest_left
from .rvect import SmoothMap


# ---------------------------------------------------------------------------
# cot : SmSet -> poly(SmSet) (sec.poly_smoothset), restricted to R^d.
# ---------------------------------------------------------------------------


def cot_object(d: int) -> Cot:
    """Apply ``cot`` on objects: ``R^d |-> Cot(d)``."""
    return Cot(d)


def cot_map(f: SmoothMap, src_dim: int, tgt_dim: int, label: str = "") -> PolyMap:
    """Apply ``cot`` on morphisms: forward part ``f``, backward part ``(T_x f)^T``.

    The cotangent pullback at ``x`` is computed with ``jax.vjp``.
    """

    def position_action(x: Array) -> Array:
        return f(x)

    def direction_action(x: Array, xi_target: Array) -> Array:
        _, vjp_fn = jax.vjp(f, x)
        (xi_source,) = vjp_fn(xi_target)
        return xi_source

    return PolyMap(
        src=Cot(src_dim),
        tgt=Cot(tgt_dim),
        position_action=position_action,
        direction_action=direction_action,
        label=label or "cot(f)",
    )


def phi_box_poly(d_out: int, d_in: int) -> Poly:
    """``Phi(<R^{d_in} | R^{d_out}>)`` as a poly object (sec.dynamics_functor).

    Positions are pairs ``(out_m, omega)`` -- the output and the one-form the box
    publishes on its input port, ``M^+ x Omega(M^-)`` -- and directions
    ``(xi, in_m)`` in ``(M^+)^* x M^-``; we use a tagged
    ``DirichletProduct(Cot(d_out), Cot(d_in))`` to record the dimensions and
    thread the one-form data through the step closures.
    """
    return DirichletProduct(Cot(d_out), Cot(d_in))


# ---------------------------------------------------------------------------
# The lax monoidal comparison of Phi on objects.
# ---------------------------------------------------------------------------


def phi_laxator(boxes: Sequence[Tuple[int, int]], label: str = "lax") -> PolyMap:
    """The laxator ``Phi(b_1) (x) ... (x) Phi(b_K) -> Phi(b_1 (x) ... (x) b_K)``.

    ``boxes`` lists the interfaces as ``(d_out, d_in)`` pairs. This is the
    comparison map of the *lax* monoidal structure by which one "combines the K
    coalgebras" before post-composing a wiring (CDLM sec.spring_second_pass):
    ``Phi`` sends a box ``<R^{d_in}|R^{d_out}>`` to positions
    ``R^{d_out} x Omega(R^{d_in})`` and directions ``(R^{d_out})^* x R^{d_in}``,
    so on directions the comparison is the reassociating isomorphism, but on
    positions it is only a map one way:

        Omega(R^{a_1}) x ... x Omega(R^{a_K})  -->  Omega(R^{a_1 + ... + a_K}),
        (omega_1, ..., omega_K) |-> (z |-> (omega_1(z_1), ..., omega_K(z_K))),

    the direct sum of one-forms, which is not surjective -- a one-form on a
    product is not a tuple of one-forms on its factors. Hence *lax*, not strong.

    The source is left-nested to match ``PCMorphism.parallel``, which folds
    pairwise; the target is the flat ``phi_box_poly(sum d_out, sum d_in)`` that
    ``Phi`` assigns to the tensored interface.
    """
    K = len(boxes)
    if K < 1:
        raise ValueError("phi_laxator: need at least one box")
    outs = [d_out for d_out, _ in boxes]
    ins = [d_in for _, d_in in boxes]
    o_cuts = [sum(outs[:k]) for k in range(K + 1)]
    i_cuts = [sum(ins[:k]) for k in range(K + 1)]

    def position_action(i):
        pairs = unnest_left(i, K)                    # [(out_m_k, omega_k)]

        def omega(z: Array) -> Array:            # the direct sum of the K forms
            return jnp.concatenate(
                [w(z[i_cuts[k]:i_cuts[k + 1]]) for k, (_, w) in enumerate(pairs)]
            )

        return jnp.concatenate([m for m, _ in pairs]), omega

    def direction_action(i, d):
        xi, in_m = d                             # xi : (R^{sum d_out})^*, in_m : R^{sum d_in}
        return nest_left([
            (xi[o_cuts[k]:o_cuts[k + 1]], in_m[i_cuts[k]:i_cuts[k + 1]])
            for k in range(K)
        ])

    src = phi_box_poly(*boxes[0])
    for d_out, d_in in boxes[1:]:
        src = DirichletProduct(src, phi_box_poly(d_out, d_in))

    return PolyMap(
        src=src,
        tgt=phi_box_poly(sum(outs), sum(ins)),
        position_action=position_action,
        direction_action=direction_action,
        label=label,
    )


# ---------------------------------------------------------------------------
# Phi : arr -> pc (sec.dynamics_functor).
# ---------------------------------------------------------------------------


def Phi(arr: Arrangement) -> PCMorphism:
    """The dynamics functor ``Phi`` applied to ``arr`` (sec.dynamics_functor).

    The state set is the coalgebra's ``S``; at state ``s`` the readout and the
    returned direction are the interpretation's, evaluated at the position
    ``arr.P.position(s)`` the coalgebra presents; the state update is
    ``arr.P.step(s, xi_Q)`` on the parameter covector of the covector triple.
    Time is discrete by design: one covector, one step (sec.displacement).
    """

    P = arr.P
    interp = interpretation(arr)
    src_p = phi_box_poly(arr.out_dim_M, arr.in_dim_M)
    tgt_p = phi_box_poly(arr.out_dim_N, arr.in_dim_N)

    def step(state):
        q = P.position(state)
        position_action, direction_action = interp(q)

        def act_positions(in_pos):
            out_m, omega_M = in_pos
            return position_action(out_m, omega_M)

        def act_directions(in_pos, in_dir):
            out_m, omega_M = in_pos
            xi_N, in_n = in_dir
            _, xi_M, in_m = direction_action(out_m, omega_M, xi_N, in_n)
            return (xi_M, in_m)

        act = PolyMap(
            src=src_p,
            tgt=tgt_p,
            position_action=act_positions,
            direction_action=act_directions,
            label=f"Phi({arr.label})",
        )

        def fiber(in_pos):
            out_m, omega_M = in_pos
            out_pos = position_action(out_m, omega_M)

            def at_pos(in_dir):
                xi_N, in_n = in_dir
                xi_Q, xi_M, in_m = direction_action(out_m, omega_M, xi_N, in_n)
                out_dir = (xi_M, in_m)
                new_state = P.step(state, xi_Q)
                return out_dir, new_state

            return out_pos, at_pos

        return act, fiber

    return PCMorphism(
        src_poly=src_p,
        tgt_poly=tgt_p,
        state=P.init,
        step=step,
    )
