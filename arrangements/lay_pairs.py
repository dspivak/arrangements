"""Lay readers' predictions for random pairs of parts, run (book plan P26, a lay-reader stage).

The question (D. Spivak, 2026-10-01): "make guesses based on an ordinary lay reader's understanding of
what different models might be, and then what would happen when you put them together, and then check
that those lay readers' results are or are not correct ... construct lay readers' beliefs for a bunch of
different ones randomly sampled, and then run them." Predictions, sample and verdicts are in
``notes/LAY-PAIRS-2026-10-01.md``.

The catalog. Each PART is a closed-input box ``I -> <0|1>``: it holds a state with its own integrator and
own cost, and publishes one number (its *report*). Each COUPLER is a stateless box ``<0|2> -> I``
reading two reports, as ``commons.stateless_coupler`` does. A pair is

    compose_seq( tensor(part_1, part_2), coupler ),

so the composite forcing is pullback-and-sum and nothing is written by hand. Alone, a part is
``compose_seq(part, cap)`` with ``cap`` discarding the report.

Parts are the library's where the library has one (``lotka_volterra.predator``, ``kuramoto.kuramoto_oscillator``,
``choice.pushed_chooser``); the rest are one-line boxes in the library's idiom. Time is in physical units:
every integrator carries the step ``DT``. The weight's integrator, ``phase_dt``, is the phase coalgebra
with the step applied to both halves (the library's ``phase`` takes a unit step and puts ``dt`` in the
costs, which a coupler shared with a configuration part cannot do).
"""

from __future__ import annotations

from typing import Callable, Dict

import jax
import jax.numpy as jnp
from jax import Array

from .arrangement import Arrangement, exact, zero_form
from .choice import pushed_chooser, pusher
from .coalgebra import CotCoalg, configuration
from .commons import EMPTY, closed_step, stateless_coupler
from .kuramoto import kuramoto_oscillator
from .lotka_volterra import log_population, predator
from .rvect import euclidean
from .wiring import compose_seq, tensor_arrangements

DT = 0.01
STEPS = 3000  # thirty time units


# ---------------------------------------------------------------------------
# Parts.
# ---------------------------------------------------------------------------


def phase_dt(dt: float, q0: float, p0: float = 0.0) -> CotCoalg:
    """Symplectic Euler with step ``dt`` on ``R^1``, unit mass: ``(q, p) |-> (q + dt p, p + dt xi)``,
    the force read at the presented position ``q + dt p``."""
    return CotCoalg(
        Q_dim=1,
        init=(jnp.array([q0]), jnp.array([p0])),
        position=lambda s: s[0] + dt * s[1],
        step=lambda s, xi: (s[0] + dt * s[1], s[1] + dt * xi),
        label="phase_dt",
    )


def _part(P: CotCoalg, cost: Callable, report: Callable, label: str) -> Arrangement:
    return Arrangement(
        P=P, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m: report(q),
        in_f=lambda q, m, n: EMPTY,
        omega=exact(lambda q, m, n: -cost(q)),
        label=label,
    )


def weight() -> Arrangement:
    """A weight on a spring: unit mass and stiffness, released from 1, reporting its position."""
    return _part(phase_dt(DT, 1.0), lambda q: 0.5 * q[0] ** 2, lambda q: q, "weight")


def tank() -> Arrangement:
    """A leaky tank under a tap: level ``h``, ``dh/dt = 1 - h`` (cost ``h^2/2 - h``), from empty."""
    return _part(configuration(euclidean(1, DT), init=jnp.zeros(1)),
                 lambda q: 0.5 * q[0] ** 2 - q[0], lambda q: q, "tank")


def rabbits() -> Arrangement:
    """Rabbits: log-population ``u`` with the prey's response ``dt e^{-u}``, logistic cost
    ``-(x - x^2/2)``, ``x = e^u``, so ``dx/dt = x(1 - x)``; from 0.1, reporting ``x``."""
    return _part(configuration(log_population(+1.0, DT), init=jnp.array([jnp.log(0.1)])),
                 lambda q: -(jnp.exp(q[0]) - 0.5 * jnp.exp(2 * q[0])),
                 lambda q: jnp.exp(q), "rabbits")


def foxes() -> Arrangement:
    """Foxes: the library's predator box (``lotka_volterra.predator``, beta = delta = 1, gamma = 0.5),
    its response NEGATIVE, ``-dt e^{-v}``; from 1, reporting ``y``. Alone ``dy/dt = -y/2``."""
    return predator(beta=1.0, gamma=0.5, delta=1.0, dt=DT)


def metronome() -> Arrangement:
    """A metronome: the library's Kuramoto oscillator, natural frequency 1, reporting its phase."""
    return kuramoto_oscillator(omega=1.0, degree=1, sharp_scale=DT)


def chooser() -> Arrangement:
    """A chooser between two options: the library's bistable chooser (``A = I``, ``T = 0``), started
    leaning 60/40 to option 1, unpushed, reporting the share of option 1."""
    return compose_seq(pusher(0.0), pushed_chooser(jnp.eye(2), 0.0, dt=DT,
                                                   init=jnp.array([0.6, 0.4])))


def needle() -> Arrangement:
    """A needle on a dial: no cost of its own, so it moves only when pushed; at 0, reporting it."""
    return _part(configuration(euclidean(1, DT), init=jnp.zeros(1)),
                 lambda q: 0.0 * q[0], lambda q: q, "needle")


PARTS: Dict[str, Callable[[], Arrangement]] = {
    "weight": weight, "tank": tank, "rabbits": rabbits, "foxes": foxes,
    "metronome": metronome, "chooser": chooser, "needle": needle,
}


# ---------------------------------------------------------------------------
# Couplers, each reading (x, y) = (first part's report, second part's report).
# ---------------------------------------------------------------------------


def band() -> Arrangement:
    """A rubber band between the two reports: cost ``(x - y)^2 / 2``."""
    return stateless_coupler(2, exact(lambda q, e, n: -0.5 * (e[0] - e[1]) ** 2), "band")


def eats() -> Arrangement:
    """The library's hunt (``lotka_volterra.hunt``, beta = 1): mass-action cost ``x y``. In Lotka-Volterra
    the second eats the first."""
    return stateless_coupler(2, exact(lambda q, e, n: -e[0] * e[1]), "eats")


def in_step() -> Arrangement:
    """The library's Kuramoto edge, K = 0.5: cost ``-K cos(x - y)``."""
    return stateless_coupler(2, exact(lambda q, e, n: 0.5 * jnp.cos(e[0] - e[1])), "in_step")


def watches() -> Arrangement:
    """The first watches the second and is drawn toward it; the second is not touched. Not exact:
    ``omega = -(x - y) dx``, the commons' insulated idiom."""

    def omega(q, e, n):
        return (jnp.zeros_like(q), jnp.array([-(e[0] - e[1]), 0.0]), jnp.zeros_like(n))

    return stateless_coupler(2, omega, "watches")


COUPLERS: Dict[str, Callable[[], Arrangement]] = {
    "band": band, "eats": eats, "in_step": in_step, "watches": watches,
}


# ---------------------------------------------------------------------------
# Building and running.
# ---------------------------------------------------------------------------


def cap() -> Arrangement:
    """Discard one report: ``<0|1> -> I``, zero forcing."""
    return stateless_coupler(1, zero_form, "cap")


def alone(name: str) -> Arrangement:
    return compose_seq(PARTS[name](), cap())


def pair(first: str, second: str, coupler: str) -> Arrangement:
    return compose_seq(tensor_arrangements([PARTS[first](), PARTS[second]()]), COUPLERS[coupler]())


def trajectory(names, arr: Arrangement, steps: int = STEPS) -> Array:
    """Run the closed composite and return each named part's report at every step."""
    parts = [PARTS[n]() for n in names]
    dims = [p.P.Q_dim for p in parts]
    step, P = closed_step(arr), arr.P

    def read(q):
        out, i = [], 0
        for p, d in zip(parts, dims):
            out.append(p.out_f(q[i:i + d], EMPTY)[0])
            i += d
        return jnp.stack(out)

    def body(s, _):
        s2 = step(s)
        return s2, read(P.position(s2))

    _, rs = jax.lax.scan(body, P.init, None, length=steps)
    return jnp.concatenate([read(P.position(P.init))[None, :], rs])


def summarize(r: Array) -> dict:
    """Plain measures of one report's run: start, end, min, max, and whether the last third swings."""
    r = jnp.asarray(r)
    tail = r[2 * len(r) // 3:]
    d = jnp.diff(tail)
    turns = int(jnp.sum(jnp.sign(d[1:]) * jnp.sign(d[:-1]) < 0))
    return {
        "start": float(r[0]), "end": float(r[-1]),
        "min": float(jnp.min(r)), "max": float(jnp.max(r)),
        "tail_turns": turns, "tail_range": float(jnp.max(tail) - jnp.min(tail)),
        "finite": bool(jnp.all(jnp.isfinite(r))),
    }
