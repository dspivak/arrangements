"""Gradient descent and backpropagation via ``Phi`` (book sec.two_layer; CDLM sec.dl_warmup).

A feedforward network is a smooth parameterized map ``F : Q x R^m -> R^n``,
regarded as an arrangement

    f : <R^0 / R^m> -> <R^0 / R^n>

with ``out_f = F``, trivial input map, and zero forcing (the learning signal
comes from whatever the network is wired into, not from ``omega``), whose
parameter is the configuration coalgebra on ``(Q, eta_LR * I)``. Under the
dynamics functor ``Phi`` this is the learner: a coalgebra on state ``Q`` whose
backward pass pulls an output covector ``xi`` back to a parameter covector
``xi_Q`` and an input covector ``xi_M``, and whose state update displaces along
the forcing, ``q |-> q + sharp_q(xi_Q)`` (sec.displacement). With the incoming
covector the forcing ``omega = -dk`` of the loss (a cost) and
``sharp_q = eta_LR * I`` this is vanilla gradient descent at rate ``eta_LR``, and
backpropagation is the chain rule packaged as the functoriality of ``Phi``.

The training loop (``train``) realizes the *environment*: the loss is a 2-ary
closed system comparing a prediction ``y`` to a label ``lambda``, and the
environment streams data ``(x, lambda)`` and seeds the output covector
``xi = omega_loss(-, lambda)|_y = -dk(-, lambda)|_y`` on each step.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import jax.numpy as jnp
from jax import Array

from .arrangement import Arrangement, zero_form
from .coalgebra import CotCoalg
from .functors import Phi
from .interpretation import interpretation, trivial_omega


# ---------------------------------------------------------------------------
# A parameterized map as an arrangement.
# ---------------------------------------------------------------------------


def parameterized_map(
    F: Callable[[Array, Array], Array],
    P: CotCoalg,
    in_dim: int,
    out_dim: int,
    label: str = "net",
) -> Arrangement:
    """The arrangement ``f`` of a feedforward network ``F : Q x R^m -> R^n``.

    The data ``x`` lives in the output port ``out_M = R^m`` and the prediction
    ``y = F(q, x)`` in ``out_N = R^n``; the input ports are trivial and
    ``omega = 0``. The parameter ``P`` is any ``cot(Q)``-coalgebra (a
    ``configuration`` for gradient descent).
    """

    return Arrangement(
        P=P,
        in_dim_M=0,
        out_dim_M=in_dim,
        in_dim_N=0,
        out_dim_N=out_dim,
        out_f=lambda q, x: F(q, x),
        in_f=lambda q, x, n_in: jnp.zeros(0),
        omega=zero_form,
        label=label,
    )


def forward_backward(
    arr: Arrangement, q: Array, x: Array, xi_out: Array
) -> Tuple[Array, Array, Array]:
    """One forward+backward pass of the learner at parameter position ``q`` on input ``x``.

    Returns ``(y, xi_Q, xi_M)`` where ``y = F(q, x)`` is the prediction, and
    ``(xi_Q, xi_M) = (T_{(q,x)} F)^T xi_out`` is the cotangent pullback of the
    incoming output covector. The parameter covector ``xi_Q`` is the
    backpropagated forcing that drives the displacement step.
    """

    interp = interpretation(arr)
    position_action, direction_action = interp(q)
    omega_M = trivial_omega(arr.in_dim_M)
    out_n, _ = position_action(x, omega_M)
    xi_Q, xi_M, _ = direction_action(x, omega_M, xi_out, jnp.zeros(arr.in_dim_N))
    return out_n, xi_Q, xi_M


# ---------------------------------------------------------------------------
# Loss (a 2-ary closed system) -- as a cost and the forcing it seeds.
# ---------------------------------------------------------------------------


def squared_error(y: Array, lam: Array) -> Array:
    """The regression cost ``k(y, lambda) = 1/2 ||y - lambda||^2``."""
    return 0.5 * jnp.sum((y - lam) ** 2)


def squared_error_forcing(y: Array, lam: Array) -> Array:
    """The forcing ``omega = -dk(-, lambda)|_y = lambda - y`` of the regression cost.

    CDLM seeded ``dU|_y = y - lambda`` and let the integrator subtract; here the
    sign is in the forcing and the coalgebra adds (DESIGN.md §3).
    """
    return lam - y


# ---------------------------------------------------------------------------
# The training loop: Phi(net) closed by a data-streaming environment.
# ---------------------------------------------------------------------------


def train(
    arr: Arrangement,
    init_q: Array,
    data: Sequence[Tuple[Array, Array]],
    *,
    forcing: Callable[[Array, Array], Array] = squared_error_forcing,
    loss: Callable[[Array, Array], Array] = squared_error,
    steps: int = None,
) -> Tuple[Array, List[float]]:
    """Run gradient descent by stepping the coalgebra ``Phi(arr)``.

    At each step the environment emits a datum ``(x, lambda)``, the network
    outputs ``y = F(q, x)``, the loss seeds the output covector
    ``xi = forcing(y, lambda)``, which is pulled back to ``xi_Q`` and displaces
    the parameter. Returns the final parameter and the per-step loss history.
    ``init_q`` is the initial *state* of ``arr.P`` (for a configuration
    coalgebra, the parameter position).
    """

    O = Phi(arr)
    q = jnp.asarray(init_q, dtype=float)
    history: List[float] = []
    n_steps = len(data) if steps is None else steps
    omega_M = trivial_omega(arr.in_dim_M)

    for t in range(n_steps):
        x, lam = data[t % len(data)]
        x = jnp.asarray(x, dtype=float)
        lam = jnp.asarray(lam, dtype=float)
        in_pos = (x, omega_M)
        pc = O.with_state(q)
        out_pos, _out_dir, new_q = pc.run_one(
            in_pos,
            lambda op, lam=lam: (forcing(op[0], lam), jnp.zeros(arr.in_dim_N)),
        )
        history.append(float(loss(out_pos[0], lam)))
        q = new_q

    return q, history
