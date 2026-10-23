"""Five peers who move to hear one another better (the book's Ch 4 ``ex.five_peers``, plan P5 (k)).

Five people stand on a line, ``P = R`` with divergence ``d(k, q) = (1/2)(k - q)^2``, and all want to know the
same number ``x``. Each reports ``x`` plus their own error, independent of the others', of expected value ``0``;
persons 1-4 have variance ``1`` and person 5 variance ``sigma^2 > 1``. Person ``j`` listens from their place
``k_j`` and hears everyone but themself, with weights ``P_ji`` the softmin of ``d(k_i, k_j)`` over the other
four, so what person ``j`` hears is off from ``x`` by ``sum_i P_ji eps_i``, of expected square

    k_j(q) = sum_i sigma_i^2 P_ji(q)^2.

The book claims, from runs (it does not prove it), that descent on ``k_1 + ... + k_5`` puts persons 1-4 together
and person 5 at ``sqrt(2 ln sigma^2)`` from them, and that descent on each round's *realized* squared error, the
number coming to light after each round and the errors drawn fresh, comes to the same rest without anyone using a
variance. ``tests/test_five_peers.py`` checks both; its predictions were written first, in the book repository,
``notes/runs/p5k-b5cf8328/PREDICTIONS.md``.

**It is an attention arrangement** (Ch 4 ``def.attention_arrangement``; ``attention.py``), run through ``Phi``,
not plain descent. ``n = 5``, ``H = P = R``. Its matter is the five places, ``Q = R^5``, a configuration
integrator with Euclidean response ``eta * I``; the placement is ``l_j(q) = k_j``; the transfer from person ``i`` to
person ``j`` is ``T_{gamma, gamma id}`` with ``gamma = exp(-d(k_i, k_j))``, the hearing of Ch 4 ``sec.messages``;
every transfer to or from the outside is ``0``, so no one hears the outside and the group sends nothing out. The
five internal boxes are filled by five *persons*, systems ``I -> B`` (not operations of the attention operad, for
the reason ``tests/test_rendezvous.py`` gives for its subsidiaries: an operation of arity zero sends ``0_M``). A
person's state is their current error ``eps_j``, and they read out ``msg(x + eps_j)``. A person's step ignores the
force on what they report and draws a fresh error ``eps_j ~ N(0, sigma_j^2)``: a ``GaussianRV`` step of
``rand.py`` (STATUS.md §3.12), paired with a sampler by ``seed``. So each person reports what they themself
observed, whatever the others' cost says, which is the example's closing sentence.

Two parents on this one wiring, differing only in the cost function:

* ``expected_cost``: ``sum_j k_j(q)``. It uses the variances and reads no message.
* ``realized_cost``: ``sum_j (sum_i P_ji(q) h_i - h_0)^2``, where ``h_i`` is the value person ``i`` sends and
  ``h_0`` the value of the message that comes in from outside, the number revealed after the round. It uses no
  variance. The term in parentheses is person ``j``'s heard value minus the number. It is written with ``q``'s
  weights ``P_ji(q)``, not as the value ``h / w`` of the message the delivery hands person ``j``, so that the
  cost function is a polynomial in the messages (``def.attention_arrangement`` (iv)); the two agree whenever every
  person sends weight ``1``, as each does, and a test checks that they do.

The realized cost's expected value over the errors is the expected cost, exactly, since ``h_i - x = eps_i`` and the
``P_ji`` sum to ``1`` (a test checks it in closed form). That is the book's "over time these moves add up to the
ones that lower the expected value."

Everything here is the agents' construction of the book's example (Claude Opus 5.5, b5cf8328, 2026-10-23).
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp

from .arrangement import Arrangement, zero_form
from .attention import (
    AttentionArrangement, Transfer, attention_arrangement, input_dim, message, msg_dim, zero_transfer,
)
from .coalgebra import configuration
from .rand import CotCoalgR, GaussianRV, key_sampler, seed
from .rvect import euclidean
from .wiring import compose_seq, tensor_arrangements

N = 5                     # five persons
DH, DP = 1, 1             # values and positions are numbers
MDIM, EDIM = msg_dim(DH), input_dim(DH, DP)


def variances(sigma2: float) -> jnp.ndarray:
    """``(1, 1, 1, 1, sigma^2)``: persons 1-4 fairly reliable, person 5 noisier."""
    return jnp.array([1.0, 1.0, 1.0, 1.0, float(sigma2)])


def divergence(k: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
    """``d(k, q) = (1/2)(k - q)^2`` (Ch 4 ``sec.simplices_divergences``)."""
    return 0.5 * (k - q) ** 2


def gains(q: jnp.ndarray) -> jnp.ndarray:
    """``G[j, i] = exp(-d(k_i, k_j))`` for ``i != j`` and ``0`` on the diagonal: how loudly ``j`` hears ``i``."""
    G = jnp.exp(-divergence(q[None, :], q[:, None]))
    return G * (1.0 - jnp.eye(N))


def weights(q: jnp.ndarray) -> jnp.ndarray:
    """``P[j, i]``, person ``j``'s weight on person ``i``: the softmin over the other four, ``P_jj = 0``."""
    G = gains(q)
    return G / jnp.sum(G, axis=1, keepdims=True)


def person_costs(q: jnp.ndarray, sigma2: float) -> jnp.ndarray:
    """``(k_1, ..., k_5)``, ``k_j = sum_i sigma_i^2 P_ji^2``, each person's expected squared error."""
    return jnp.sum(variances(sigma2)[None, :] * weights(q) ** 2, axis=1)


def _values(m_out: jnp.ndarray) -> jnp.ndarray:
    """The five sent values ``h_i`` from the concatenated messages ``[h_1, w_1, ..., h_5, w_5]``."""
    return jnp.reshape(m_out, (N, MDIM))[:, 0]


def expected_cost(sigma2: float):
    """The parent's cost ``sum_j k_j(q)``: reads no message, uses the variances."""
    return lambda q, m_out, n_in: jnp.sum(person_costs(q, sigma2))


def realized_cost(q, m_out, n_in):
    """The parent's cost ``sum_j (sum_i P_ji(q) h_i - h_0)^2``: each person's heard value minus the number that
    came in from outside, squared. No variance enters."""
    return jnp.sum((weights(q) @ _values(m_out) - n_in[0]) ** 2)


def transfers(q: jnp.ndarray):
    """``T[j][i]`` for ``i, j`` in ``0..5``: ``T_{gamma, gamma id}``, ``gamma = exp(-d(k_i, k_j))``, between persons;
    ``0`` to and from the outside (index ``0``) and on the diagonal."""
    G = gains(q)
    Z = zero_transfer(DH)
    rows = []
    for j in range(N + 1):
        row = []
        for i in range(N + 1):
            if i == 0 or j == 0 or i == j:
                row.append(Z)
            else:
                g = G[j - 1, i - 1]
                row.append(Transfer(gamma=g, L=g * jnp.eye(DH)))
        rows.append(tuple(row))
    return tuple(rows)


def parent(cost, places, eta: float) -> AttentionArrangement:
    """The 5-ary attention arrangement: matter the five places, placement ``l_j = k_j``, the hearing above."""
    return attention_arrangement(
        N, DH, DP,
        placement=lambda q: jnp.reshape(q, (N, DP)),
        transfers=transfers,
        cost=cost,
        parameter=configuration(euclidean(N, eta), init=jnp.asarray(places, float)),
        label="peers",
    )


def person(x: float, var: float, label: str) -> Arrangement:
    """A person ``I -> B``: state their current error ``eps``, readout ``msg(x + eps)``, and a step that ignores
    the force on their report and draws a fresh error ``eps' = sqrt(var) z``, ``z ~ N(0, 1)``."""
    sd = float(jnp.sqrt(var))
    P = CotCoalgR(
        Q_dim=1, init=jnp.zeros(1), position=lambda s: s,
        step=lambda s, xi: GaussianRV(1, lambda z: sd * z),
        kind=GaussianRV, label=f"draw({var:g})",
    )
    return Arrangement(
        P=P, in_dim_M=0, out_dim_M=0, in_dim_N=EDIM, out_dim_N=MDIM,
        out_f=lambda eps, m_out: message(x + eps[0], 1.0),
        in_f=lambda eps, m_out, n_in: jnp.zeros(0),
        omega=zero_form,
        label=label,
    )


def five_peers(sigma2: float, places: Sequence[float], *, realized: bool, x: float = 7.0, eta: float = 1.0,
               sampler_seed: int = 0) -> Arrangement:
    """The composite ``peers(person_1, ..., person_5) : I -> B``, by composition in ``arr``, seeded.

    Its state is ``(((eps_1, ..., eps_5), places), sampler key)`` in ``compose_seq``'s order; its external input
    is the message ``msg(x)`` and a position, and with the realized cost the number is read from it.
    """
    var = variances(sigma2)
    persons = [person(x, float(var[i]), f"person{i + 1}") for i in range(N)]
    cost = realized_cost if realized else expected_cost(sigma2)
    composite = compose_seq(tensor_arrangements(persons), parent(cost, places, eta).arr())
    return seed(composite, key_sampler(sampler_seed))


def exterior(x: float, p0: float = 0.0) -> jnp.ndarray:
    """The external input: the number ``x`` as the message ``msg(x)``, and a position ``p0``."""
    return jnp.concatenate([message(x, 1.0), jnp.array([p0])])


def places_of(state) -> jnp.ndarray:
    """The five places from a composite state."""
    return state[0][1]


def rest_distance(sigma2: float) -> float:
    """``sqrt(2 ln sigma^2)``, person 5's distance from the four at the book's rest."""
    return float(jnp.sqrt(2.0 * jnp.log(sigma2)))
