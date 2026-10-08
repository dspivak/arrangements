"""A listener who hears accounts and decides whom to count on (book plan P37, ``AGENTDOCPLAN.md`` §8.2).

David (book repo, 2026-10-07): "changing beliefs is in the plan and repo i believe, and who you hang around
with (attention wise) is also in there ... you can avoid people who are unaccountable." This module builds
that sentence out of two mechanisms the library already has, and nothing else: the belief integrator
(``belief.belief_coalgebra``, Bayes' rule for every belief over finitely many values, P36 (a)) and the
spatial attention parent (``attention.spatial_arrangement``, the paper's eq:spatial-transfers). Predictions
fixed before this file existed: the book repo's ``notes/runs/p37-listener-f2836b6c/PREDICTIONS.md``; the
tests are ``tests/test_listener.py``.

THE SOURCE is frozen at placement 0 and sends ``(x, 1)``: weight 1 always (it is there, to be counted on
or not), signal ``x in R^3`` the one-hot outcome of a round on the step that outcome arrives and ``0`` on
every other step. The outcomes: the promise was KEPT; it FAILED and the account of why LINED UP with what a
check found; it FAILED and the account DID NOT LINE UP. The signal is the source's configuration, held by a
zero response, and set from outside between steps (``set_signal``): the world's doing, not the source's.

THE LISTENER is ``compose_seq(belief_box, valuation_box)``, a part ``I -> B`` on the parent's interface.

* ``belief_box``: configuration a belief ``y`` over the three kinds of source (``belief_coalgebra``: state
  the log-heights, duration 1), cost ``k_B(y, (h, w)) = <y, N h>`` with ``N_io = -log Pr(o | z_i)`` -- the
  book's likelihood cost on the RECEIVED message, linear in both, no readout. On the received one-hot
  ``h = kappa e_o`` the force on the belief is ``kappa log Pr(o | .)``: Bayes' rule tempered by the kernel
  weight. Its forcing is UNILATERAL (``likelihood="unilateral"``): the cost's differential in the belief
  only, zero on the input, the idiom of ``science.account_community``; ``likelihood="exact"`` is ``-dk_B``
  on the input too, the comparison arm. It sends its belief forward on a ONE-WAY wire
  (``wire="oneway"``, ``stop_gradient`` in its sending map, the idiom of ``account.fund_layer``), or
  plainly (``wire="exact"``), the comparison arm.
* ``valuation_box``: trivial parameter, cost ``k_V(y, (h, w)) = w <y, ell> + (1 - w) c``: the loss it
  expects from counting on the source, times how much it counts on it (the received weight), plus the cost
  of doing without for the rest. Composition pulls its covector on the received weight back through the
  transfer onto the listener's placement, ``xi_p = -kappa'(p) (<y, ell> - c)``; on the exact wire its
  covector on the belief, ``-w ell``, reaches the belief box's state.

THE COMPOSITE is ``compose_seq(tensor(listener, source), spatial(2).arr())``, the shape of
``tests/test_curious_reader.py``; only the listener's placement moves.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .arrangement import Arrangement
from .attention import input_dim, make_input, message, msg_dim, spatial_arrangement, spatial_config
from .belief import belief_coalgebra, belief_of
from .coalgebra import configuration
from .rvect import constant, trivial
from .wiring import compose_seq, tensor_arrangements, trivial_parameter

DH, DP = 3, 1
MDIM, EDIM = msg_dim(DH), input_dim(DH, DP)     # dim M = 4, dim E = 5
EMPTY = jnp.zeros(0)
EXT = make_input(message(jnp.zeros(DH), 0.0), jnp.zeros(DP))   # the exterior input (0_M, 0)

KEPT, LINED_UP, NOT_LINED_UP = 0, 1, 2
OUTCOME = {"K": KEPT, "L": LINED_UP, "X": NOT_LINED_UP}


class Model(NamedTuple):
    """The numbers: kernel scale, the likelihood table (rows the kinds of source, columns the outcomes),
    the loss of counting on each kind, the cost of doing without, the listener's mobility and start."""
    tau: float = 0.8
    pi: tuple = ((0.98, 0.019, 0.001), (0.6, 0.38, 0.02), (0.6, 0.08, 0.32))
    ell: tuple = (0.4, 1.4, 8.0)
    c: float = 5.0
    mu_p: float = 0.1
    p0: float = 0.5


DEFAULT = Model()


def nll_table(model: Model) -> Array:
    """``N_io = -log Pr(o | z_i)``."""
    return -jnp.log(jnp.asarray(model.pi, float))


# ---------------------------------------------------------------------------
# The listener's two boxes.
# ---------------------------------------------------------------------------


def belief_box(model: Model, n: int = 3, *, wire: str = "oneway", likelihood: str = "unilateral",
               init=None) -> Arrangement:
    """``I -> <E | R^n>``: receives the parent's input ``e = ((h, w), p_0)``, sends its belief forward."""
    assert wire in ("oneway", "exact", "subtract") and likelihood in ("unilateral", "exact")
    N = nll_table(model)
    ell = jnp.asarray(model.ell, float)
    init = jnp.zeros(n) if init is None else jnp.asarray(init, float)

    def k_B(y, e):
        return jnp.dot(y, N @ e[:DH])

    def omega(y, m_out, e):
        if likelihood == "exact":
            xi_y, xi_e = jax.grad(lambda y_, e_: -k_B(y_, e_), argnums=(0, 1))(y, e)
        else:
            xi_y = jax.grad(lambda y_: -k_B(y_, e))(y)
            xi_e = jnp.zeros_like(e)
        if wire == "subtract":
            # The checker (David, 2026-10-07: "somebody checking that force and making sure it's not there"):
            # the belief is sent forward plainly, so the valuation's ``-w ell`` arrives on it, and a part that
            # knows the listener's losses and reads how much it is counting on the source adds ``+w ell`` back.
            # The cascade's ``wire="subtract"`` (account.py), which must know the push; ``oneway`` needs nothing.
            xi_y = xi_y + e[DH] * ell
        return (xi_y, jnp.zeros(0), xi_e)

    send = (lambda y, m_out: jax.lax.stop_gradient(y)) if wire == "oneway" else (lambda y, m_out: y)
    return Arrangement(
        P=belief_coalgebra(n, init),
        in_dim_M=0, out_dim_M=0, in_dim_N=EDIM, out_dim_N=n,
        out_f=send,
        in_f=lambda y, m_out, e: jnp.zeros(0),
        omega=omega,
        label=f"belief({wire},{likelihood})",
    )


def valuation_box(model: Model, n: int = 3) -> Arrangement:
    """``<E | R^n> -> B``: passes the parent's input down to the belief box, sends ``0_M`` up, and
    carries the cost ``w <y, ell> + (1 - w) c`` on the belief it is sent and the input it receives."""
    ell = jnp.asarray(model.ell, float)
    c = float(model.c)

    def k_V(q, y, e):
        w = e[DH]
        return w * jnp.dot(y, ell) + (1.0 - w) * c

    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=EDIM, out_dim_M=n, in_dim_N=EDIM, out_dim_N=MDIM,
        out_f=lambda q, y: message(jnp.zeros(DH), 0.0),
        in_f=lambda q, y, e: e,
        omega=jax.grad(lambda q, y, e: -k_V(q, y, e), argnums=(0, 1, 2)),
        label="valuation",
    )


def listener(model: Model = DEFAULT, **kw) -> Arrangement:
    """The listener ``I -> B``: the valuation box after the belief box."""
    return compose_seq(belief_box(model, **kw), valuation_box(model))


def source() -> Arrangement:
    """The source ``I -> B``: its configuration is the signal it sends, held by the zero response (set
    from outside by ``set_signal``); weight 1 always; no cost."""
    return Arrangement(
        P=configuration(constant(jnp.zeros((DH, DH))), init=jnp.zeros(DH)),
        in_dim_M=0, out_dim_M=0, in_dim_N=EDIM, out_dim_N=MDIM,
        out_f=lambda x, m_out: message(x, 1.0),
        in_f=lambda x, m_out, e: jnp.zeros(0),
        omega=lambda x, m_out, e: (jnp.zeros(DH), jnp.zeros(0), jnp.zeros(EDIM)),
        label="source",
    )


# ---------------------------------------------------------------------------
# The composite, and its state.
# ---------------------------------------------------------------------------


def listener_system(model: Model = DEFAULT, **kw) -> Arrangement:
    """``spatial(2)(listener, source) : I -> B``; the listener at ``p0``, the source at ``0``; only the
    listener's placement responds (mobility ``mu_p``), the value maps and the source's placement frozen."""
    n_Q = 2 * DP + 2 * DH * DH
    diag = jnp.zeros(n_Q).at[0].set(model.mu_p)
    parent = spatial_arrangement(
        2, DH, DP, model.tau,
        parameter=configuration(constant(jnp.diag(diag)),
                                init=spatial_config((model.p0, 0.0), (jnp.eye(DH), jnp.eye(DH)))),
        label="spatial(2)")
    return compose_seq(tensor_arrangements([listener(model, **kw), source()]), parent.arr())


# The composite's state is ``(((l, ()), x), q_parent)``: the listener's pair (belief log-heights, the
# valuation's empty state), the source's signal, the parent's configuration.

def log_heights(state) -> Array:
    return state[0][0][0]


def placement(state) -> Array:
    return state[1][0]


def set_signal(state, x):
    """The state with the source's signal set to ``x`` (``jnp.zeros(DH)`` for no news)."""
    ((l, z), _), q = state
    return ((l, z), jnp.asarray(x, float)), q


def one_hot(o) -> Array:
    return jnp.zeros(DH).at[o].set(1.0)


def step_fn(arr: Arrangement):
    """One step of ``Phi(arr)`` on the exterior input ``(0_M, 0)`` with zero exterior covector,
    specialized to the configuration component of the forcing (the form ``collective.step_fn`` uses)."""
    P, omega = arr.P, arr.omega

    @jax.jit
    def step(s):
        xi_q, _, _ = omega(P.position(s), EMPTY, EXT)
        return P.step(s, xi_q), xi_q

    return step


def run_script(arr: Arrangement, script: str, steps_per_round: int = 10, repeats: Sequence = ()):
    """Run ``script`` (one letter per round, ``K``, ``L`` or ``X``), the outcome on each round's first
    step and no news on the rest; ``repeats`` are ``(step, letter)`` pairs sent again on those steps.
    Returns the trajectory of (placement, kernel weight, belief) at steps ``0..T`` as arrays."""
    step = step_fn(arr)
    extra = dict(repeats)
    s = arr.P.init
    ps, ys = [], []
    T = steps_per_round * len(script)
    for t in range(T):
        r, k = divmod(t, steps_per_round)
        letter = extra.get(t, script[r] if k == 0 else None)
        s = set_signal(s, one_hot(OUTCOME[letter]) if letter else jnp.zeros(DH))
        ps.append(float(placement(s)))
        ys.append(np.asarray(belief_of(log_heights(s))))
        s, _ = step(s)
    ps.append(float(placement(s)))
    ys.append(np.asarray(belief_of(log_heights(s))))
    return np.array(ps), np.array(ys), s
