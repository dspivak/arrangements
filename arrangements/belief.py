"""Bayes' rule for every belief over finitely many values (book plan P36 stage (a)).

``notes/BAYES-EVERY-BELIEF.md`` holds the derivation, fixed before this file existed, the
predictions and the verdicts; the proof in the book's letters is the book repo's
``notes/bayes-every-belief.md``.

THE BELIEF BOX. ``n`` values. The state is ``l in R^n``, the belief's log-heights; the belief is
``q = softmax(l)``, and ``l + c 1`` is the same belief. The box reads the belief out and forces
nothing, as ``learning.py``'s network does: its learning signal comes from what it is wired into.

THE LIKELIHOOD BOX, downstream, has trivial parameter and the cost ``k(y, x) = <y, nll(x)>``, the
average under the belief it is shown of ``nll(x)_i = -log Pr(x | z_i)``. The cost is linear in the
belief, so the covector it sends back on the belief's wire is ``xi = log Pr(x | .)`` whatever the
belief is.

THE RESPONSE (``log_height_response``) divides by the belief's height,
``sharp_l = tau * diag(1 / softmax(l))``: diagonal and positive definite on all of ``R^n``. The
pullback of ``xi`` along the readout is ``(diag(q) - q q^T) xi = q * (xi - <q, xi>)``, so one step
at duration 1 is

    l' = l + xi - <q, xi> 1,        softmax(l') = q * Pr(x | .) / <q, Pr(x | .)>,

Bayes' rule exactly, for every ``l`` and every positive likelihood, with no family chosen: the
categorical family is every belief over the ``n`` values. ``diag(q) - q q^T`` is singular along
``1``, so the response is a generalized inverse of it, not an inverse; nothing has to fix the
constant. ``choice.belief_arrangement`` fixes it by the gauge ``l_n = 0`` and inverts the
``(n-1)``-block numerically; ``gauge_response`` here is that inverse in closed form,
``diag(1/q_i) + (1/q_n) 1 1^T``.

WHERE IT STOPS. A height of exactly zero, in the belief or the likelihood, is outside: the
log-height is ``-inf``. In float64 the diagonal response is exact until a height underflows to
``0``; the dense ``inv`` of the gauge block loses the cancellation ``G^{-1} G = I`` long before
(``cond(G) ~ 1 / min q``).

A WHOLE CURVE (P36 stage (b), ``notes/BAYES-WHOLE-CURVE.md``). On a grid over the real line wide
enough to hold a curve's tails, heights underflow and ``diag(1 / q)`` is ``inf``, although the
step itself, ``xi - <q, xi>``, is finite. ``belief_coalgebra`` is the same integrator with the
BELIEF as its configuration and the log-heights as its state (an integrator's state need not be
its configuration, ``def.duration_integrator``): it presents ``softmax(l)`` and its update is the
response after the responsibility, ``l |-> l + tau (xi - <softmax(l), xi>)``, with no division.
``updating_on_beliefs`` wires it into the same likelihood box.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from .arrangement import Arrangement, exact
from .coalgebra import CotCoalg, configuration
from .functors import Phi
from .interpretation import trivial_omega
from .rvect import ReactiveVectorSpace
from .wiring import compose_seq, trivial_parameter

Array = jnp.ndarray


# ---------------------------------------------------------------------------
# Beliefs, and Bayes' rule computed directly.
# ---------------------------------------------------------------------------


def belief_of(l: Array) -> Array:
    """The belief with log-heights ``l``: ``softmax(l)``."""
    return jax.nn.softmax(jnp.asarray(l, dtype=float))


def gauge_belief_of(lam: Array) -> Array:
    """The belief with log-odds ``lam`` against the last value: ``softmax(lam, 0)``."""
    return jax.nn.softmax(jnp.concatenate([jnp.asarray(lam, dtype=float), jnp.zeros(1)]))


def posterior(q: Array, p: Array) -> Array:
    """Bayes' rule computed directly: ``q p / <q, p>``."""
    w = jnp.asarray(q, dtype=float) * jnp.asarray(p, dtype=float)
    return w / jnp.sum(w)


# ---------------------------------------------------------------------------
# The responses.
# ---------------------------------------------------------------------------


def log_height_response(n: int, tau: float = 1.0) -> ReactiveVectorSpace:
    """``sharp_l = tau * diag(1 / softmax(l))`` on ``R^n``: divide by the belief's height."""
    return ReactiveVectorSpace(dim=n, sharp_fn=lambda l: tau * jnp.diag(1.0 / belief_of(l)))


def gauge_response(n: int, tau: float = 1.0) -> ReactiveVectorSpace:
    """The inverse of the gauge-fixed block ``diag(q) - q q^T`` (``i < n``) in closed form,
    ``diag(1/q_i) + (1/q_n) 1 1^T`` (Sherman-Morrison), on ``R^{n-1}``."""

    def sharp_fn(lam: Array) -> Array:
        q = gauge_belief_of(lam)
        return tau * (jnp.diag(1.0 / q[:-1]) + jnp.ones((n - 1, n - 1)) / q[-1])

    return ReactiveVectorSpace(dim=n - 1, sharp_fn=sharp_fn)


# ---------------------------------------------------------------------------
# The boxes.
# ---------------------------------------------------------------------------


def belief_box(
    response: ReactiveVectorSpace,
    n: int,
    obs_dim: int,
    init: Array,
    readout: Callable[[Array], Array] = belief_of,
) -> Arrangement:
    """The belief box ``<R^0|R^obs_dim> -> <R^0|R^{n+obs_dim}>``: it reports its belief (``n``
    heights) forward beside the observation passed through, and forces nothing."""
    dim = response.dim
    return Arrangement(
        P=configuration(response, init=jnp.asarray(init, dtype=float)),
        in_dim_M=0, out_dim_M=obs_dim, in_dim_N=0, out_dim_N=n + obs_dim,
        out_f=lambda l, x: jnp.concatenate([readout(l), x]),
        in_f=lambda l, x, n_in: jnp.zeros(0),
        omega=lambda l, x, n_in: (jnp.zeros(dim), jnp.zeros(obs_dim), jnp.zeros(0)),
        label="belief",
    )


def likelihood_box(nll: Callable[[Array], Array], n: int, obs_dim: int) -> Arrangement:
    """The likelihood box, closing the composite: trivial parameter and the cost
    ``k(y, x) = <y, nll(x)>``, ``nll(x)_i = -log Pr(x | z_i)``; forcing the exact ``-dk``."""
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0, out_dim_M=n + obs_dim, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, w: jnp.zeros(0),
        in_f=lambda q, w, n_in: jnp.zeros(0),
        omega=exact(lambda q, w, n_in: -jnp.dot(w[:n], nll(w[n:]))),
        label="likelihood",
    )


def updating(
    response: ReactiveVectorSpace,
    nll: Callable[[Array], Array],
    n: int,
    obs_dim: int,
    init: Array,
    readout: Callable[[Array], Array] = belief_of,
) -> Arrangement:
    """The book's *updating* picture (Ch 2's ``sec.bayes``, cut 2026-10-03; its home is now Ch 4 ``sec.bayes_curve``): the belief box wired into the
    likelihood box, the observation arriving from outside. ``<R^0|R^obs_dim> -> <R^0|R^0>``."""
    return compose_seq(
        belief_box(response, n, obs_dim, init, readout), likelihood_box(nll, n, obs_dim)
    )


def self_forced_belief(
    response: ReactiveVectorSpace,
    nll: Callable[[Array], Array],
    obs_dim: int,
    init: Array,
    readout: Callable[[Array], Array] = belief_of,
) -> Arrangement:
    """The same update with the cost carried by the belief box as its own forcing and no
    likelihood box (``tests/test_bayes_integrator.py``'s first shape)."""
    return Arrangement(
        P=configuration(response, init=jnp.asarray(init, dtype=float)),
        in_dim_M=0, out_dim_M=obs_dim, in_dim_N=0, out_dim_N=0,
        out_f=lambda l, x: jnp.zeros(0),
        in_f=lambda l, x, n_in: jnp.zeros(0),
        omega=exact(lambda l, x, n_in: -jnp.dot(readout(l), nll(x))),
        label="belief(self-forced)",
    )


# ---------------------------------------------------------------------------
# The box whose configuration is the belief (stage (b)).
# ---------------------------------------------------------------------------


def belief_coalgebra(n: int, init: Array, tau: float = 1.0) -> CotCoalg:
    """The belief integrator with ``Q`` the beliefs and ``S`` the log-heights.

    The state ``l in R^n`` presents the belief ``softmax(l)``; on a covector ``xi`` at that
    belief (a force on the belief, read up to a constant) it moves by the response after the
    responsibility,

        l |-> l + tau (xi - <softmax(l), xi>),

    which is ``log_height_response`` applied to the pullback of ``xi`` along ``softmax``, written
    without the division: ``diag(1/q) (diag(q) - q q^T) xi = xi - <q, xi> 1``. Where every height
    is a positive float the two boxes take the same step; where a height underflows only this one
    is finite.
    """
    return CotCoalg(
        Q_dim=n,
        init=jnp.asarray(init, dtype=float),
        position=belief_of,
        step=lambda l, xi: l + tau * (xi - jnp.dot(belief_of(l), xi)),
        label="belief(Q=beliefs)",
        # over a duration t (book P9.5): the Euler family of its response, still at t = 0
        timed_step=lambda l, xi, t: l + t * tau * (xi - jnp.dot(belief_of(l), xi)),
    )


def updating_on_beliefs(
    nll: Callable[[Array], Array],
    n: int,
    obs_dim: int,
    init: Array,
    tau: float = 1.0,
) -> Arrangement:
    """The *updating* picture with ``belief_coalgebra`` as the belief box's parameter: the box
    sends its configuration, the belief, forward unchanged beside the observation, and the
    likelihood box closes it. ``<R^0|R^obs_dim> -> <R^0|R^0>``."""
    box = Arrangement(
        P=belief_coalgebra(n, init, tau),
        in_dim_M=0, out_dim_M=obs_dim, in_dim_N=0, out_dim_N=n + obs_dim,
        out_f=lambda y, x: jnp.concatenate([y, x]),
        in_f=lambda y, x, n_in: jnp.zeros(0),
        omega=lambda y, x, n_in: (jnp.zeros(n), jnp.zeros(obs_dim), jnp.zeros(0)),
        label="belief(Q=beliefs)",
    )
    return compose_seq(box, likelihood_box(nll, n, obs_dim))


# ---------------------------------------------------------------------------
# Likelihoods.
# ---------------------------------------------------------------------------


def column_nll(c: Array) -> Array:
    """The observation *is* its column of costs: the environment supplies
    ``c_i = -log Pr(x | z_i)`` on the wire (``obs_dim = n``). Any likelihood whatever."""
    return c


def bell_nll(grid: Array) -> Callable[[Array], Array]:
    """Bell-curve noise of variance 1 about each grid value: ``nll(x)_i = (x - z_i)^2 / 2``, up
    to a constant (``obs_dim = 1``). The book's likelihood box, the loss averaged over the belief."""
    grid = jnp.asarray(grid, dtype=float)
    return lambda x: 0.5 * (x[0] - grid) ** 2


# ---------------------------------------------------------------------------
# Running.
# ---------------------------------------------------------------------------


def run(arr: Arrangement, observations, state=None):
    """Step ``Phi(arr)`` once per observation; the arrangement is closed on the right."""
    O = Phi(arr)
    s = O.state if state is None else state
    for x in observations:
        _, _, s = O.with_state(s).run_one(
            (jnp.atleast_1d(jnp.asarray(x, dtype=float)), trivial_omega(0)),
            lambda _o: (jnp.zeros(0), jnp.zeros(0)),
        )
    return s


def log_heights(state) -> Array:
    """The belief box's state out of a composite's state pair."""
    return state[0] if isinstance(state, tuple) else state
