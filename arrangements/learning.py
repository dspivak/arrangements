"""Gradient descent and backpropagation via ``Phi``, the whole network as one box (CDLM sec.dl_warmup).

The book's run of sec.neural_network is ``network.drawn_network``: a box per parameter, the
datum, the label and the loss each a box, run by ``Phi`` with nothing computed outside it (plan
P43 (b) job 1). This module's one-box network, with the loss's covector computed in code by
``train``, is kept as a fast check, equal to the drawn arrangement at the test's settings
(``tests/test_gradient_descent.py``); a run that cites it as a composite of boxes is wrong.

A feedforward network is a smooth parameterized map ``F : Q x R^m -> R^n``,
regarded as an arrangement

    f : <R^0 / R^m> -> <R^0 / R^n>

with ``out_f = F``, trivial input map, and zero forcing (the learning signal
comes from whatever the network is wired into, not from ``omega``), whose
parameter is the configuration coalgebra on ``(Q, eta_LR * I)``. Under the
dynamics functor ``Phi`` this is the learner: a coalgebra on state ``Q`` whose
backward pass pulls an output covector ``xi`` back to a parameter covector
``xi_Q`` and an input covector ``xi_M``, and whose state update displaces along
the forcing, ``q |-> q + sharp_q(xi_Q)`` (sec.differential_equations). With the incoming
covector the forcing ``omega = -dk`` of the loss (a cost) and
``sharp_q = eta_LR * I`` this is vanilla gradient descent at rate ``eta_LR``, and
backpropagation is the chain rule packaged as the functoriality of ``Phi``.

The training loop (``train``) realizes the *environment*: the loss is a 2-ary
closed system comparing a prediction ``y`` to a label ``lambda``, and the
environment streams data ``(x, lambda)`` and seeds the output covector
``xi = omega_loss(-, lambda)|_y = -dk(-, lambda)|_y`` on each step.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .arrangement import Arrangement, exact, zero_form
from .coalgebra import CotCoalg
from .functors import Phi, closed_step
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
    sign is in the forcing and the coalgebra adds (STATUS.md §1).
    """
    return lam - y


def loss_box(loss: Callable[[Array, Array], Array], n: int) -> Arrangement:
    """The loss as a box: the 2-ary closed arrangement ``<0|n> (x) <0|n> -> I`` with cost
    ``loss(y, lambda)`` on the answer ``y`` and the label ``lambda``; its 1-form is ``-d loss``."""
    from .wiring import trivial_parameter

    return Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=2 * n, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m: jnp.zeros(0), in_f=lambda q, m, nn: jnp.zeros(0),
        omega=exact(lambda q, m, nn: -loss(m[:n], m[n:])), label="loss",
    )


def training_arrangement(
    arr: Arrangement, init_q, data: Sequence[Tuple[Array, Array]],
    loss: Callable[[Array, Array], Array] = None,
) -> Arrangement:
    """The closed arrangement that trains ``arr`` on a list of examples (book sec.neural_network).

    The datum's box and the label's box (``network.example_box``: state the place in the list,
    stepping to the next whatever comes back), the learner ``arr`` started at the state ``init_q``
    beside plain wires carrying the label, and the loss as a box (``loss_box``), composed by
    ``compose_seq`` and ``tensor_arrangements``. Its state is
    ``(((datum, label), (learner, wires)), loss)``.
    """
    from .network import example_box, identity_box
    from .wiring import compose_seq, tensor_arrangements

    loss = squared_error if loss is None else loss
    if arr.in_dim_M or arr.in_dim_N:
        raise ValueError("training_arrangement: the learner's input ports must be trivial")
    xs = jnp.stack([jnp.asarray(x, dtype=float).reshape(arr.out_dim_M) for x, _ in data])
    lams = jnp.stack([jnp.asarray(l, dtype=float).reshape(arr.out_dim_N) for _, l in data])
    learner = replace(arr, P=replace(arr.P, init=jnp.asarray(init_q, dtype=float)))
    n = arr.out_dim_N
    fed = compose_seq(
        tensor_arrangements([example_box(xs, "datum"), example_box(lams, "label")]),
        tensor_arrangements([learner, identity_box(n)]),
    )
    return compose_seq(fed, loss_box(loss, n))


# ---------------------------------------------------------------------------
# The training loop: the closed training arrangement, run by Phi.
# ---------------------------------------------------------------------------


def train(
    arr: Arrangement,
    init_q: Array,
    data: Sequence[Tuple[Array, Array]],
    *,
    forcing: Callable[[Array, Array], Array] = squared_error_forcing,
    loss: Callable[[Array, Array], Array] = squared_error,
    steps: int = None,
    path: bool = False,
):
    """Run gradient descent as ``Phi`` of the closed training arrangement (``training_arrangement``).

    The datum and the label are boxes stepping through ``data`` (cycling), the loss is a box with
    cost ``loss``; nothing is computed outside the arrangement (plan P43 (b) job 5). ``forcing`` is
    no longer used to drive the run; it is checked, at the first example, to be ``-d loss`` in the
    answer, since a caller passing the two has said they agree. Returns the learner's final state
    and the per-step loss history, read off the state before each step (and, with ``path``, the
    learner's state after each step). The compiled step is checked against ``Phi`` on its first
    two steps (``functors.closed_step``). The run this replaced, the learner alone with the loss's
    covector computed in code each step, is ``train_by_port``, kept as the check.
    """
    A = training_arrangement(arr, init_q, data, loss)
    n_steps = len(data) if steps is None else steps
    m, n, k = arr.out_dim_M, arr.out_dim_N, arr.P.Q_dim
    x0, lam0 = (jnp.asarray(v, dtype=float) for v in data[0])
    y0 = arr.out_f(arr.P.position(jnp.asarray(init_q, dtype=float)), x0.reshape(m))
    pushed = jax.grad(lambda y: -loss(y, lam0.reshape(n)))(y0)
    if not np.allclose(np.asarray(forcing(y0, lam0.reshape(n))), np.asarray(pushed), atol=1e-9):
        raise ValueError("train: forcing is not -d loss at the first example")
    if n_steps == 0:
        return jnp.asarray(init_q, dtype=float), ([], []) if path else []
    step = closed_step(A)

    def body(s, _):
        pos = A.P.position(s)
        x, lam, q = pos[:m], pos[m:m + n], pos[m + n:m + n + k]
        s = step(s)
        return s, (loss(arr.out_f(q, x), lam), s[0][1][0])

    s, (hist, states) = jax.lax.scan(body, A.P.init, None, length=n_steps)
    history = [float(h) for h in np.asarray(hist)]
    final = s[0][1][0]
    return (final, history, states) if path else (final, history)


def train_by_port(
    arr: Arrangement,
    init_q: Array,
    data: Sequence[Tuple[Array, Array]],
    *,
    forcing: Callable[[Array, Array], Array] = squared_error_forcing,
    loss: Callable[[Array, Array], Array] = squared_error,
    steps: int = None,
) -> Tuple[Array, List[float]]:
    """The run ``train`` replaced: the learner alone, stepped by ``Phi(arr)``, the loss in code.

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
