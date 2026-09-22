"""The polynomial interpretation of an arrangement at a parameter position (sec.lens_internalization).

This is the *state-free* half of the dynamics functor ``Phi`` (sec.dynamics_functor):
it computes everything that depends only on the parameter *position* ``q in Q``
that the coalgebra presents, and not on how the state is stored or updated. The
coalgebra ``u`` of the arrangement supplies ``q`` and consumes the parameter
covector this half produces (functors.py).

Given the parameter position ``q``, the interpretation provides:

* ``position_action(out_m, omega_M) -> (out_n, omega_N)``
  the readout, with ``out_n`` from ``out_f`` and the published one-form
  ``omega_N`` on ``N^-`` -- the pullback of the incoming ``omega_M`` along
  ``in_f`` plus the ``N^-``-component of the arrangement's forcing:

      omega_N(n) = (in_f(q, out_m, -))^* omega_M + omega(q, out_m, n)|_{N^-}.

* ``direction_action(out_m, omega_M, xi_N, in_n) -> (xi_Q, xi_M, in_m)``
  with ``in_m`` from ``in_f`` and the covector triple the sum of three pullbacks
  (CDLM eqn.bigtheta, with ``dU`` replaced by ``omega``):

      xi_Q = (out_f)^* xi_N |_Q + (in_f)^* omega_M |_Q + omega|_Q,
      xi_M = (out_f)^* xi_N |_M + (in_f)^* omega_M |_M + omega|_{M^+};

  ``xi_Q in Q^*`` is the parameter covector the coalgebra consumes, ``xi_M`` is
  returned to the source interface.

A one-form ``omega_M in Omega(M^-)`` is represented as the actual smooth map
``M^- -> (M^-)^*`` (a callable), so the interpretation is exact for *every* smooth
arrangement, not only the affine/quadratic ones -- there is no reconstruction from
value-and-Jacobian-at-zero. ``trivial_omega(d)`` is the zero form on ``R^d``, used
wherever an interface's input side is trivial.
"""

from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from .arrangement import Arrangement


def trivial_omega(d: int) -> Callable[[Array], Array]:
    """The zero one-form on ``R^d`` (``z |-> 0``)."""
    return lambda z: jnp.zeros(d)


def interpretation(
    arr: Arrangement,
) -> Callable[[Array], Tuple[Callable, Callable]]:
    """Return ``q |-> (position_action, direction_action)`` for ``arr`` at position ``q``."""

    out_f, in_f, omega = arr.out_f, arr.in_f, arr.omega

    def at_position(q: Array):
        # ---- readout: (out_m, omega_M) |-> (out_n, omega_N) ----
        def position_action(out_m: Array, omega_M):
            out_n = out_f(q, out_m)

            # omega_N(in_n) = (in_f(q, out_m, -))^* omega_M + omega(q, out_m, in_n)|_{N^-}.
            # Returned AS THE FORM (a callable), exact for any smooth omega / in_f.
            def omega_N(in_n: Array) -> Array:
                in_f_qm = lambda y: in_f(q, out_m, y)
                in_m_at = in_f_qm(in_n)
                omega_M_val = omega_M(in_m_at)
                _, vjp_in_f = jax.vjp(in_f_qm, in_n)
                (pull_back,) = vjp_in_f(omega_M_val)
                _, _, omega_n = omega(q, out_m, in_n)
                return pull_back + omega_n

            return out_n, omega_N

        # ---- backward: (out_m, omega_M, xi_N, in_n) |-> (xi_Q, xi_M, in_m) ----
        def direction_action(out_m: Array, omega_M, xi_N: Array, in_n: Array):
            in_m = in_f(q, out_m, in_n)
            omega_M_val = omega_M(in_m)  # the incoming one-form evaluated at in_m

            # The three pullbacks summed in the covector triple.
            _, vjp_out_f = jax.vjp(lambda qq, mm: out_f(qq, mm), q, out_m)
            xi_Q_a, xi_M_a = vjp_out_f(xi_N)

            _, vjp_in_f = jax.vjp(
                lambda qq, mm, nn: in_f(qq, mm, nn), q, out_m, in_n
            )
            xi_Q_b, xi_M_b, _ = vjp_in_f(omega_M_val)

            omega_q, omega_m, _ = omega(q, out_m, in_n)

            xi_Q = xi_Q_a + xi_Q_b + omega_q
            xi_M = xi_M_a + xi_M_b + omega_m
            return xi_Q, xi_M, in_m

        return position_action, direction_action

    return at_position
