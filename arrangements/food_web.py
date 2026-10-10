"""A food web whose species boxes are the same whatever they meet (``notes/LAY-PAIRS-2026-10-01.md`` §7).

D. Spivak, 2026-10-01: "could it be modeled in such a way that it wouldn't be dependent on knowing what we
were going to do ahead of time … That's what really makes something compositional." The Lotka–Volterra port
(``lotka_volterra.py``) puts who-eats-whom in the predator's NEGATIVE response, with a symmetric ``hunt``
cost; that predator box reverses every other coupler. Here every species is one ordinary box, log
population with the positive response ``dt e^{-u}`` and its own rate ``r - c x``, and the direction lives in
the coupler: ``directed_eats(beta, delta)`` reads (prey, eater) and carries the NON-EXACT forcing

    omega = -beta * y dx + delta * x dy,

so the prey loses in proportion to the eater and the eater gains in proportion to the prey. A web is

    compose_seq( tensor(species), compose_seq(copy, tensor(couplers)) ),

``copy`` sending each species' one report to every coupler that reads it (its transpose sums the
covectors, by ``jax.vjp`` inside ``compose_seq``).
"""

from __future__ import annotations

from typing import Sequence, Tuple

import jax
import jax.numpy as jnp

from .arrangement import Arrangement, exact, zero_form
from .coalgebra import configuration
from .commons import EMPTY, closed_step, stateless_coupler
from .lotka_volterra import log_population
from .wiring import compose_seq, tensor_arrangements, trivial_parameter


def species(r: float, c: float = 0.0, x0: float = 1.0, dt: float = 0.01, label: str = "sp") -> Arrangement:
    """An ordinary species: ``du/dt = r - c x`` alone (growth or death, optionally logistic), reporting ``x``."""
    return Arrangement(
        P=configuration(log_population(+1.0, dt), init=jnp.array([jnp.log(x0)])),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m: jnp.exp(q),
        in_f=lambda q, m, n: EMPTY,
        omega=exact(lambda q, m, n: r * jnp.exp(q[0]) - 0.5 * c * jnp.exp(2 * q[0])),
        label=label,
    )


def directed_eats(beta: float, delta: float) -> Arrangement:
    """(prey, eater): ``-beta y dx + delta x dy``. Not exact: ``d omega = (delta + beta) dx ^ dy``."""

    def omega(q, e, n):
        return (jnp.zeros_like(q), jnp.array([-beta * e[1], delta * e[0]]), jnp.zeros_like(n))

    return stateless_coupler(2, omega, f"eats({beta:g},{delta:g})")


def copy(n_species: int, idx: Sequence[int]) -> Arrangement:
    idx = jnp.asarray(idx, dtype=int)
    return Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=n_species, in_dim_N=0, out_dim_N=len(idx),
        out_f=lambda q, m: m[idx], in_f=lambda q, m, n: EMPTY, omega=zero_form, label="copy",
    )


def web(sp: Sequence[Arrangement], edges: Sequence[Tuple[int, int, float, float]]) -> Arrangement:
    """``edges``: (prey index, eater index, beta, delta)."""
    idx = [k for (i, j, _, _) in edges for k in (i, j)]
    couplers = tensor_arrangements([directed_eats(b, d) for (_, _, b, d) in edges])
    return compose_seq(tensor_arrangements(list(sp)), compose_seq(copy(len(sp), idx), couplers))


def log_states(arr: Arrangement, steps: int):
    step, P = closed_step(arr), arr.P

    def body(s, _):
        s2 = step(s)
        return s2, P.position(s2)

    _, qs = jax.lax.scan(body, P.init, None, length=steps)
    return jnp.concatenate([P.position(P.init)[None, :], qs])


def by_hand(r, c, edges, x0, dt, steps):
    """The equations written out: ``du_i = dt (r_i - c_i x_i - sum_eaters beta x_j + sum_prey delta x_k)``."""
    r, c = jnp.asarray(r), jnp.asarray(c)

    def f(u, _):
        x = jnp.exp(u)
        g = r - c * x
        for (i, j, b, d) in edges:
            g = g.at[i].add(-b * x[j]).at[j].add(d * x[i])
        u2 = u + dt * g
        return u2, u2

    u0 = jnp.log(jnp.asarray(x0))
    _, us = jax.lax.scan(f, u0, None, length=steps)
    return jnp.concatenate([u0[None, :], us])


# ---------------------------------------------------------------------------
# The same web in the population chart (plan P43 (b) job 7): state the population itself.
# ---------------------------------------------------------------------------


def population_species(r: float, x0: float = 1.0, dt: float = 0.01, label: str = "sp") -> Arrangement:
    """A species in the population chart: state ``x``, Euclidean response ``dt``, its own growth
    ``r x`` as a 1-form (``d(r x^2 / 2)``), reporting ``x``. Alone, ``x' = x + dt r x``."""
    from .rvect import euclidean

    return Arrangement(
        P=configuration(euclidean(1, dt), init=jnp.array([x0])),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m: q,
        in_f=lambda q, m, n: EMPTY,
        omega=exact(lambda q, m, n: 0.5 * r * q[0] ** 2),
        label=label,
    )


def population_eats(beta: float, delta: float) -> Arrangement:
    """(prey, eater) in the population chart: ``-beta x y dx + delta x y dy``, not exact."""

    def omega(q, e, n):
        return (jnp.zeros_like(q), jnp.array([-beta * e[0] * e[1], delta * e[0] * e[1]]), jnp.zeros_like(n))

    return stateless_coupler(2, omega, f"eats_pop({beta:g},{delta:g})")


def population_web(sp: Sequence[Arrangement], edges: Sequence[Tuple[int, int, float, float]]) -> Arrangement:
    """``web`` with population-chart couplers; the species must be ``population_species``."""
    idx = [k for (i, j, _, _) in edges for k in (i, j)]
    couplers = tensor_arrangements([population_eats(b, d) for (_, _, b, d) in edges])
    return compose_seq(tensor_arrangements(list(sp)), compose_seq(copy(len(sp), idx), couplers))
