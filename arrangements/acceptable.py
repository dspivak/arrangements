"""Acceptable durations, and the furniture room whose chairs are moved in bursts (plan P41 (e), Claude Opus 5.5,
dd3daf04, 2026-10-30; the book's ``notes/runs/p41e-dd3daf04/RESULTS.md``).

**Acceptable durations** (David's candidate, 2026-10-30, b94c8c22, "I'm not sure about this"): a duration is acceptable
for a part when one step of it is within ``eps`` of two steps of half of it, for every shorter duration too, under
assume-guarantee bounds on what arrives. ``defect`` measures the difference on chosen coordinates over a set of
states; ``acceptable_duration`` reads off the largest acceptable duration on a grid. Both compare runs of one
arrangement stepped by the dynamics functor at two durations; the distance between the results is arithmetic beside
``arr``.

**The burst room**: two people (quick), two chairs, a store of the people's discomfort ``z``, and a switch ``y``
whose two rests are *calm* and *moving the furniture* (David, 2026-10-30: "When the people don't like where the
furniture is enough, it surpasses a certain threshold and they put a bunch of force on the furniture"). Every
coupling is stateless and one-way except ``talk`` and the parts' own costs:

* ``seat`` on ``(q, w)``: force ``-a (q - w)`` on the person only; the chair is not pulled.
* ``talk``, ``routine``: as in ``two_clocks``.
* ``strain`` on ``(q1, w1, q2, w2, z)``: force ``sigma * (a/2) sum (q_i - w_i)^2`` on ``z`` only: a value read forward
  from the people and chairs, stored.
* ``leak``: cost ``lam z^2 / 2`` on ``z``.  ``switch``: cost ``y^4/4 - y^2/2`` on ``y``.
* ``bias`` on ``(z, y)``: force ``(z - theta) / kappa`` on ``y`` only. The calm rest disappears at
  ``z = theta + 0.385 kappa``, the moving rest at ``z = theta - 0.385 kappa``: the threshold is the switch's fold.
* ``push`` on ``(y, q1, w1, q2, w2)``: force ``G g(y) (q_i - w_i)`` on chair ``i`` only, ``g(y) = (1 + tanh 10y)/2``.
* ``drain`` on ``(y, z)``: force ``-rho g(y)`` on ``z`` only: moving the furniture relieves the discomfort.

State position: ``(q1, q2, w1, w2, z, y)``. The room is ``I -> <1|0>``, the routine ``u`` on its one input port.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .arrangement import Arrangement
from .coalgebra import configuration
from .functors import Phi
from .interpretation import trivial_omega
from .rvect import euclidean
from .two_clocks import (_cost_coupling, matter, person, route, routine, routine_step, set_position, talk, tidy)
from .wiring import compose_seq, tensor_arrangements, trivial_parameter


# ---------------------------------------------------------------------------
# Steps, defects, acceptable durations.
# ---------------------------------------------------------------------------


def closed_step(arr: Arrangement, duration: float):
    """One ``Phi`` step of a closed system ``I -> I`` at ``duration``, compiled, checked against ``Phi`` once."""
    if arr.in_dim_M or arr.out_dim_M or arr.in_dim_N or arr.out_dim_N:
        raise ValueError("closed_step needs I -> I")
    P, omega, z = arr.P, arr.omega, jnp.zeros(0)

    @jax.jit
    def step(s):
        xi_q, _, _ = omega(P.position_at(s, duration), z, z)
        return P.step_over(s, xi_q, duration)

    _, _, s_new = Phi(arr).with_state(P.init).run_one((z, trivial_omega(0)), lambda _o: (z, z), duration)
    for x, y in zip(jax.tree_util.tree_leaves(s_new), jax.tree_util.tree_leaves(step(P.init))):
        if not np.allclose(np.asarray(x), np.asarray(y), rtol=1e-12, atol=1e-14):
            raise AssertionError("closed_step differs from Phi")
    return step


def timed_closed_step(arr: Arrangement):
    """The step of a closed system ``I -> I`` as a function of ``(s, duration)``, compiled once; checked against
    ``Phi`` at duration ``0.1``."""
    if arr.in_dim_M or arr.out_dim_M or arr.in_dim_N or arr.out_dim_N:
        raise ValueError("timed_closed_step needs I -> I")
    P, omega, z = arr.P, arr.omega, jnp.zeros(0)

    def step(s, d):
        xi_q, _, _ = omega(P.position_at(s, d), z, z)
        return P.step_over(s, xi_q, d)

    _, _, s_new = Phi(arr).with_state(P.init).run_one((z, trivial_omega(0)), lambda _o: (z, z), 0.1)
    for x, y in zip(jax.tree_util.tree_leaves(s_new), jax.tree_util.tree_leaves(step(P.init, 0.1))):
        if not np.allclose(np.asarray(x), np.asarray(y), rtol=1e-12, atol=1e-14):
            raise AssertionError("timed_closed_step differs from Phi")
    return step


def defect_curve(arr: Arrangement, step, taus, starts, coords=None):
    """For each ``tau`` in ``taus``: ``max`` over ``starts`` of the max-norm distance, on ``coords``, between one step
    of ``tau`` and two of ``tau/2``; ``step(s, d)`` is the arrangement's step at duration ``d`` (``timed_closed_step``)."""
    coords = list(range(len(np.asarray(starts)[0]))) if coords is None else list(coords)
    cidx = jnp.asarray(coords)

    @jax.jit
    def worst(starts, d):
        def one(p):
            s = set_position_traced(arr, p)
            a = arr.P.position(step(s, d))
            b = arr.P.position(step(step(s, d / 2), d / 2))
            return jnp.max(jnp.abs(a - b)[cidx])
        return jnp.max(jax.vmap(one)(starts))

    st = jnp.asarray(np.asarray(starts, dtype=float))
    return [float(worst(st, float(t))) for t in taus]


def _positions(arr: Arrangement, step, starts):
    """``step`` applied to each start position (rows of ``starts``); returns the new positions."""
    f = jax.jit(jax.vmap(lambda p: arr.P.position(step(set_position_traced(arr, p)))))
    return np.asarray(f(jnp.asarray(np.asarray(starts, dtype=float))))


def set_position_traced(arr: Arrangement, pos):
    leaves, treedef = jax.tree_util.tree_flatten(arr.P.init)
    out, i = [], 0
    for leaf in leaves:
        k = int(jnp.size(leaf))
        out.append(pos[i:i + k].reshape(jnp.shape(leaf)))
        i += k
    return jax.tree_util.tree_unflatten(treedef, out)


def defect(arr: Arrangement, make_step: Callable[[float], Callable], tau: float, starts, coords=None) -> float:
    """``max`` over ``starts`` of the max-norm distance, on ``coords``, between one step of ``tau`` and two of
    ``tau/2``. ``make_step(d)`` returns the arrangement's step at duration ``d``."""
    one, half = make_step(tau), make_step(tau / 2)
    twice = lambda s: half(half(s))
    a, b = _positions(arr, one, starts), _positions(arr, twice, starts)
    diff = np.abs(a - b) if coords is None else np.abs(a - b)[:, list(coords)]
    return float(diff.max())


def acceptable_duration(defects: Sequence[float], taus: Sequence[float], eps: float) -> float:
    """The largest ``tau`` on the increasing grid ``taus`` such that it and every shorter grid duration has defect
    at most ``eps`` ("for every shorter duration too"); ``0`` if the shortest fails."""
    best = 0.0
    for t, d in zip(taus, defects):
        if d > eps:
            break
        best = t
    return best


# ---------------------------------------------------------------------------
# One part and two coupled parts: springs.
# ---------------------------------------------------------------------------


def spring(c: float) -> Arrangement:
    """``<0|2> -> I``: cost ``c (x - y)^2 / 2`` on the two wires it hears (two-way)."""
    return _cost_coupling(2, lambda m: 0.5 * c * (m[0] - m[1]) ** 2, f"spring({c:g})")


def one_part(k: float, x0: float = 0.0, eta: float = 1.0) -> Arrangement:
    """A part alone, closed: ``x`` with cost ``k x^2 / 2`` (``tidy(k, 0)``)."""
    return replace(compose_seq(person(x0, 1, eta), tidy(k, 0.0)), label=f"one_part({k:g})")


def two_parts(k1: float, k2: float, c: float, eta1: float = 1.0, eta2: float = 1.0) -> Arrangement:
    """Two parts, each with its own cost ``k_i x_i^2 / 2``, coupled by ``spring(c)``; closed. Position ``(x1, x2)``."""
    parts = tensor_arrangements([person(0.0, 2, eta1), person(0.0, 2, eta2)])      # x1 x1 x2 x2
    wired = compose_seq(parts, route(4, [0, 2, 1, 3]))                               # x1 | x2 | x1 x2
    return replace(compose_seq(wired, tensor_arrangements([tidy(k1, 0.0), tidy(k2, 0.0), spring(c)])),
                   label=f"two_parts({k1:g},{k2:g},{c:g})")


def part_open(k: float, c: float, eta: float = 1.0) -> Arrangement:
    """Part 1 of ``two_parts`` with its partner's box left open: ``<0|1> -> I`` on the partner's readout. Its matter
    is ``x1`` alone; a contract assumes where the partner is and how fast it moves."""
    inner = tensor_arrangements([_identity(1), person(0.0, 2, eta)])                  # v, x, x
    wired = compose_seq(inner, route(3, [1, 1, 0]))                                  # x | x v
    return replace(compose_seq(wired, tensor_arrangements([tidy(k, 0.0), spring(c)])), label="part_open")


def _identity(k: int) -> Arrangement:
    return Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=k, in_dim_N=0, out_dim_N=k,
                       out_f=lambda q, m_out: m_out, in_f=lambda q, m_out, n_in: jnp.zeros(0),
                       omega=lambda s, m, n: (jnp.zeros_like(s), jnp.zeros_like(m), jnp.zeros_like(n)),
                       label=f"id<0|{k}>")


def open_defect(arr: Arrangement, tau: float, xs, vs, speed: float) -> float:
    """Defect of an open part ``<0|1> -> I`` under the contract: its state in ``xs``, its partner's readout in ``vs``
    at the start and, after half the duration, anywhere within ``speed * tau / 2`` of where it was. The part is
    stepped by the dynamics functor with its open box filled by the stateless system reading out the held value
    (``two_clocks.open_step``, checked against ``Phi``)."""
    from .two_clocks import open_step
    one = open_step(arr, tau, check_at=[float(vs[0])])
    half = open_step(arr, tau / 2)
    worst = 0.0
    for x in xs:
        s = set_position(arr, [x])
        for v in vs:
            a = float(np.asarray(arr.P.position(one(s, jnp.array([v]))))[0])
            s1 = half(s, jnp.array([v]))
            for dv in (-speed * tau / 2, 0.0, speed * tau / 2):
                b = float(np.asarray(arr.P.position(half(s1, jnp.array([v + dv]))))[0])
                worst = max(worst, abs(a - b))
    return worst


# ---------------------------------------------------------------------------
# The burst room.
# ---------------------------------------------------------------------------


def _force_coupling(k: int, force, label: str, in_dim: int = 0) -> Arrangement:
    """``<0|k> -> <in_dim|0>`` whose effort field on the ``k`` wires it hears is ``force(m)`` (a covector, not
    necessarily the differential of a cost); it returns nothing on its input ports."""
    def omega(s, m, n):
        return jnp.zeros_like(s), force(m), jnp.zeros_like(n)
    return Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=k, in_dim_N=in_dim, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0), in_f=lambda q, m_out, n_in: jnp.zeros(in_dim), omega=omega,
        label=label,
    )


def gate(y):
    return 0.5 * (1.0 + jnp.tanh(10.0 * y))


def seat_oneway(a: float) -> Arrangement:
    """``(q, w)``: force ``-a (q - w)`` on the person; nothing on the chair."""
    return _force_coupling(2, lambda m: jnp.array([-a * (m[0] - m[1]), 0.0]), f"seat>({a:g})")


def strain(a: float, sigma: float) -> Arrangement:
    """``(q1, w1, q2, w2, z)``: force ``sigma (a/2) sum (q_i - w_i)^2`` on ``z``; nothing on the people or chairs."""
    def f(m):
        s = 0.5 * a * ((m[0] - m[1]) ** 2 + (m[2] - m[3]) ** 2)
        return jnp.array([0.0, 0.0, 0.0, 0.0, sigma * s])
    return _force_coupling(5, f, f"strain({a:g},{sigma:g})")


def switch_cost() -> Arrangement:
    return _cost_coupling(1, lambda m: m[0] ** 4 / 4 - m[0] ** 2 / 2, "switch")


def bias(theta: float, kappa: float) -> Arrangement:
    """``(z, y)``: force ``(z - theta)/kappa`` on ``y``; nothing on ``z``."""
    return _force_coupling(2, lambda m: jnp.array([0.0, (m[0] - theta) / kappa]), f"bias({theta:g},{kappa:g})")


def push(G: float) -> Arrangement:
    """``(y, q1, w1, q2, w2)``: force ``G g(y) (q_i - w_i)`` on chair ``i``; nothing on ``y`` or the people."""
    def f(m):
        g = G * gate(m[0])
        return jnp.array([0.0, 0.0, g * (m[1] - m[2]), 0.0, g * (m[3] - m[4])])
    return _force_coupling(5, f, f"push({G:g})")


def drain(rho: float) -> Arrangement:
    """``(y, z)``: force ``-rho g(y)`` on ``z``; nothing on ``y``."""
    return _force_coupling(2, lambda m: jnp.array([0.0, -rho * gate(m[0])]), f"drain({rho:g})")


BURST = dict(a=1.0, c=3.0, ell=1.0, sigma=0.5, lam=0.0125, theta=30.0, kappa=26.0, eta_y=50.0, rho=100.0, G=0.15,
             eta_w=1.0)


def burst_room(q=(-2.5, 2.5), w=(-2.5, 2.5), z0=0.0, y0=-1.43, **kw) -> Arrangement:
    """``I -> <1|0>``: the burst room (module docstring). Position ``(q1, q2, w1, w2, z, y)``. ``eta_w`` is the chairs'
    response; ``eta_w=0`` holds them where they are (plan P41 (f): the people's transit with the chairs held)."""
    p = {**BURST, **kw}
    parts = tensor_arrangements([
        person(q[0], 5), person(q[1], 5),
        matter(configuration(euclidean(1, p["eta_w"]), init=jnp.array([w[0]])), 3, "chair"),
        matter(configuration(euclidean(1, p["eta_w"]), init=jnp.array([w[1]])), 3, "chair"),
        matter(configuration(euclidean(1, 1.0), init=jnp.array([z0])), 4, "store"),
        matter(configuration(euclidean(1, p["eta_y"]), init=jnp.array([y0])), 4, "switch"),
    ])
    # wires: q1 0-4, q2 5-9, w1 10-12, w2 13-15, z 16-19, y 20-23
    idx = [0, 10, 5, 13,            # seat1 (q1, w1), seat2 (q2, w2)
           1, 6, 2, 7,              # talk (q1, q2), routine (q1, q2)
           3, 11, 8, 14, 16,        # strain (q1, w1, q2, w2, z)
           17, 20, 18, 21,          # leak (z), switch (y), bias (z, y)
           22, 4, 12, 9, 15,        # push (y, q1, w1, q2, w2)
           23, 19]                  # drain (y, z)
    wired = compose_seq(parts, route(24, idx))
    couplings = tensor_arrangements([
        seat_oneway(p["a"]), seat_oneway(p["a"]), talk(p["c"], p["ell"]), routine(),
        strain(p["a"], p["sigma"]), _cost_coupling(1, lambda m: 0.5 * p["lam"] * m[0] ** 2, "leak"),
        switch_cost(), bias(p["theta"], p["kappa"]), push(p["G"]), drain(p["rho"]),
    ])
    return replace(compose_seq(wired, couplings), label="burst_room")


def input_step(arr: Arrangement, duration: float):
    """The room's step at ``duration`` with the routine ``u`` held, checked against ``Phi`` (``routine_step``)."""
    return routine_step(arr, duration)


def timed_input_step(arr: Arrangement, u: float):
    """The step of a room ``I -> <1|0>`` with the routine held at ``u``, as a function of ``(s, duration)``;
    checked against ``Phi`` at duration ``0.004`` (``routine_step``)."""
    routine_step(arr, 0.004, check_inputs=(u, u))
    P, omega, z = arr.P, arr.omega, jnp.zeros(0)

    def step(s, d):
        xi_q, _, _ = omega(P.position_at(s, d), z, jnp.array([u]))
        return P.step_over(s, xi_q, d)
    return step


def defect_per_state(arr: Arrangement, step, taus, starts, coords):
    """Array ``[state, tau]`` of the max-norm distance on ``coords`` between one step of ``tau`` and two of
    ``tau/2``, from each start position."""
    cidx = jnp.asarray(list(coords))

    @jax.jit
    def one(p, d):
        s = set_position_traced(arr, p)
        a = arr.P.position(step(s, d))
        b = arr.P.position(step(step(s, d / 2), d / 2))
        return jnp.max(jnp.abs(a - b)[cidx])

    f = jax.jit(jax.vmap(jax.vmap(one, in_axes=(None, 0)), in_axes=(0, None)))
    return np.asarray(f(jnp.asarray(np.asarray(starts, dtype=float)), jnp.asarray(np.asarray(taus, dtype=float))))
