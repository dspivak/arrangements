"""Two clocks: slow matter as the parameter of a quick arrangement (book Ch 1 ``sec.held_form``, l.565, and
``exc.furniture``; plan P41 (d), Claude Opus 5.5, b94c8c22, 2026-10-30).

Ch 1: "a reader has slow parts as well as quick ones, and where the slower parts sit sets out the shape that
decides where the quick ones can settle ... A form that runs long enough moves the slow-moving parts ... a capacity
can hold forms, and forms---given time---can create new capacities."

**Withdrawn reading (David, 2026-10-30: "It's not a sub-operad. It's just a crazy thing you made up ... the durations
are something that the user picks").** This module first read a slow part as one "on a slower clock" and the room as a
``Para`` morphism over ``arr``; neither is the book's meaning. Slow and quick in Ch 1 are whether a form stays long
enough for a part to settle (``sec.held_form``). What the code does is unchanged and plain:

* ``slowed(P, eps)`` steps ``P`` as if told ``eps * tau``; for the configuration integrator used here that is the
  same part with its response multiplied by ``eps``: a chair that responds weakly. ``slowed(P, 0)`` holds the part
  still, used to read where the people can settle with the chairs where they are.
* ``unit_quick`` and ``fill`` build a person-and-chair unit with the chair's box left open and then fill it; the
  tests use them only to check that the room built three ways steps alike.
* The furniture room: two people (quick matter), two chairs (slow matter), and stateless couplings, each a cost:
  ``seat`` ``a (q - w)^2 / 2``, ``talk`` ``-c exp(-(q1 - q2)^2 / 2 l^2)``, ``tidy`` ``b (w - w0)^2 / 2``, and the
  ``routine`` ``u (q1 - q2)^2 / 2`` with ``u`` arriving on the room's one input port (it pushes nothing back there).

The room is built three ways and the tests check that they step alike: flat (all four parts tensored, then wired),
as the composite of two person-and-chair units (``unit_quick``, each a two-clock arrangement) filled unit by unit,
and as the composite of the units' quick arrangements filled once (``Para`` composition: the slow boxes tensor).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .arrangement import Arrangement, zero_form
from .coalgebra import CotCoalg, configuration
from .functors import Phi
from .interpretation import trivial_omega
from .rvect import euclidean
from .wiring import compose_seq, tensor_arrangements, trivial_parameter


# ---------------------------------------------------------------------------
# The slower clock, and filling slow boxes.
# ---------------------------------------------------------------------------


def slowed(P: CotCoalg, eps: float) -> CotCoalg:
    """``P`` told ``eps * tau`` when told ``tau``: its look-ahead and its update both run on the slower clock.

    The unit step (``duration=None``) is the step of duration ``eps``.
    """
    if P.timed_step is None:
        raise TypeError(f"slowed: part {P.label or '?'} has no duration law")
    look_ahead = None
    if P.look_ahead is not None:
        look_ahead = lambda s, tau: P.position_at(s, eps * tau)
    return CotCoalg(
        Q_dim=P.Q_dim,
        init=P.init,
        position=P.position,
        step=lambda s, xi: P.timed_step(s, xi, eps),
        label=f"slowed({P.label}, {eps:g})",
        timed_step=lambda s, xi, tau: P.timed_step(s, xi, eps * tau),
        look_ahead=look_ahead,
    )


def _none(q, m_out, n_in):
    return jnp.zeros(0)


def matter(P: CotCoalg, degree: int, label: str) -> Arrangement:
    """The integrator ``P`` as a system ``I -> <0|degree>``: it reads its configuration out ``degree`` times and has
    effort field 0 (Ch 3: "an integrator can be considered as a system")."""
    d = P.Q_dim
    return Arrangement(
        P=P, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=degree * d,
        out_f=lambda q, m_out: jnp.tile(q, degree), in_f=_none, omega=zero_form, label=label,
    )


def constant(v: float) -> Arrangement:
    """The stateless system ``I -> <0|1>`` reading out ``v`` with effort field 0 (as ``pivot`` reads out ``0``): a
    support that never moves, whatever force it receives."""
    return Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m_out: jnp.array([v]), in_f=_none, omega=zero_form, label=f"const({v:g})",
    )


def identity(k: int) -> Arrangement:
    """The identity on ``<0|k>``: stateless, effort field 0."""
    return Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=k, in_dim_N=0, out_dim_N=k,
        out_f=lambda q, m_out: m_out, in_f=_none, omega=zero_form, label=f"id<0|{k}>",
    )


def route(out_dim: int, idx: Sequence[int], label: str = "route") -> Arrangement:
    """The static wiring ``<0|out_dim> -> <0|len(idx)>`` presenting wire ``idx[j]`` as wire ``j`` (copies allowed;
    forces on copies add on the way back)."""
    idx = jnp.asarray(list(idx), dtype=int)
    return Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=out_dim, in_dim_N=0, out_dim_N=len(idx),
        out_f=lambda q, m_out: m_out[idx], in_f=_none, omega=zero_form, label=label,
    )


def fill(quick: Arrangement, slow: Sequence[Arrangement]) -> Arrangement:
    """``quick : <0|k> -> N`` with its open boxes filled by the slow systems ``slow`` (each ``I -> <0|k_i>``,
    ``sum k_i = k``): ``quick . (slow_1 (x) ... (x) slow_n)``."""
    return compose_seq(tensor_arrangements(list(slow)), quick)


# ---------------------------------------------------------------------------
# The couplings: stateless arrangements, each with a cost.
# ---------------------------------------------------------------------------


def _cost_coupling(k: int, cost, label: str) -> Arrangement:
    """``<0|k> -> I`` with effort field ``-d cost`` on the ``k`` wires it hears."""
    def omega(s, m, n):
        return jnp.zeros_like(s), -jax.grad(cost)(m), jnp.zeros_like(n)
    return Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=k, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0), in_f=_none, omega=omega, label=label,
    )


def seat(a: float) -> Arrangement:
    """``<0|2> -> I`` on ``(q, w)``: cost ``a (q - w)^2 / 2``; the person is pulled to the chair and the chair to the
    person."""
    return _cost_coupling(2, lambda m: 0.5 * a * (m[0] - m[1]) ** 2, f"seat({a:g})")


def talk(c: float, ell: float = 1.0) -> Arrangement:
    """``<0|2> -> I`` on ``(q1, q2)``: cost ``-c exp(-(q1 - q2)^2 / 2 ell^2)``; two people close enough talk."""
    return _cost_coupling(2, lambda m: -c * jnp.exp(-(m[0] - m[1]) ** 2 / (2 * ell ** 2)), f"talk({c:g})")


def tidy(b: float, w0: float) -> Arrangement:
    """``<0|1> -> I`` on ``w``: cost ``b (w - w0)^2 / 2``; the room is tidied back toward its plan."""
    return _cost_coupling(1, lambda m: 0.5 * b * (m[0] - w0) ** 2, f"tidy({b:g},{w0:g})")


def routine() -> Arrangement:
    """``<0|2> -> <1|0>`` on ``(q1, q2)``, hearing ``u`` on its one input: cost ``u (q1 - q2)^2 / 2``. It returns no
    force to the port the routine arrives on (the schedule is not pushed)."""
    def omega(s, m, n):
        g = jax.grad(lambda mm: 0.5 * n[0] * (mm[0] - mm[1]) ** 2)(m)
        return jnp.zeros_like(s), -g, jnp.zeros_like(n)
    return Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=2, in_dim_N=1, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0), in_f=lambda q, m_out, n_in: jnp.zeros(0), omega=omega,
        label="routine",
    )


# ---------------------------------------------------------------------------
# The room, three ways.
# ---------------------------------------------------------------------------


def person(q0: float, degree: int, eta: float = 1.0) -> Arrangement:
    return matter(configuration(euclidean(1, eta), init=jnp.array([q0])), degree, "person")


def chair_integrator(w_start: float, eta: float = 1.0) -> CotCoalg:
    return configuration(euclidean(1, eta), init=jnp.array([w_start]))


def room_flat(eps: float, q: Sequence[float], w: Sequence[float], a=1.0, c=3.0, ell=1.0, b=2.6,
              w0=(-2.5, 2.5)) -> Arrangement:
    """``I -> <1|0>``: people ``q``, chairs ``w`` (each on the clock ``eps``), all four tensored, then wired.
    State position: ``(q1, q2, w1, w2)``."""
    parts = tensor_arrangements([
        person(q[0], 3), person(q[1], 3),
        matter(slowed(chair_integrator(w[0]), eps), 2, "chair"),
        matter(slowed(chair_integrator(w[1]), eps), 2, "chair"),
    ])
    # wires: p1 p1 p1 p2 p2 p2 c1 c1 c2 c2 -> seat1(p1,c1) seat2(p2,c2) talk(p1,p2) routine(p1,p2) tidy1 tidy2
    wired = compose_seq(parts, route(10, [0, 6, 3, 8, 1, 4, 2, 5, 7, 9]))
    couplings = tensor_arrangements([seat(a), seat(a), talk(c, ell), routine(), tidy(b, w0[0]), tidy(b, w0[1])])
    return replace(compose_seq(wired, couplings), label=f"room_flat(eps={eps:g})")


def unit_quick(q0: float, w0: float, a=1.0, b=2.6) -> Arrangement:
    """One person-and-chair unit's quick arrangement ``<0|1> -> <0|2>``: the chair's box left open; the person (its
    quick matter), seated by ``seat``, the chair's box tidied by ``tidy``; the person read out twice."""
    inner = tensor_arrangements([identity(1), person(q0, 1)])          # wires: w, p
    wired = compose_seq(inner, route(2, [1, 0, 0, 1, 1]))               # p w w p p
    return replace(compose_seq(wired, tensor_arrangements([seat(a), tidy(b, w0), identity(2)])),
                   label="unit_quick")


def room_quick(c=3.0, ell=1.0) -> Arrangement:
    """The room's own wiring ``<0|4> -> <1|0>`` on ``(p1, p1, p2, p2)``: talk and the routine."""
    return replace(compose_seq(route(4, [0, 2, 1, 3]), tensor_arrangements([talk(c, ell), routine()])),
                   label="room_quick")


def room_para(eps: float, q, w, a=1.0, c=3.0, ell=1.0, b=2.6, w0=(-2.5, 2.5)) -> Arrangement:
    """``Para`` composition, then one fill: the units' quick arrangements composed into the room with both chair
    boxes open, ``<0|2> -> <1|0>``, then filled by both chairs on the clock ``eps``."""
    quick = compose_seq(tensor_arrangements([unit_quick(q[0], w0[0], a, b), unit_quick(q[1], w0[1], a, b)]),
                        room_quick(c, ell))
    chairs = [matter(slowed(chair_integrator(w[i]), eps), 1, "chair") for i in range(2)]
    return replace(fill(quick, chairs), label=f"room_para(eps={eps:g})")


def room_unitwise(eps: float, q, w, a=1.0, c=3.0, ell=1.0, b=2.6, w0=(-2.5, 2.5)) -> Arrangement:
    """Each unit filled by its own chair first, then the two filled units composed into the room."""
    units = [fill(unit_quick(q[i], w0[i], a, b), [matter(slowed(chair_integrator(w[i]), eps), 1, "chair")])
             for i in range(2)]
    return replace(compose_seq(tensor_arrangements(units), room_quick(c, ell)), label=f"room_unitwise(eps={eps:g})")


# ---------------------------------------------------------------------------
# Running: the step on the routine, compiled and checked against Phi.
# ---------------------------------------------------------------------------


def routine_step(arr: Arrangement, duration: float, check_inputs: Sequence[float] = (1.0, 0.0)):
    """One ``Phi`` step of a room ``I -> <1|0>`` held for ``duration``, the routine ``u`` on its port; compiled, and
    checked against ``Phi(arr).run_one`` at the same duration on the inputs ``check_inputs``."""
    if arr.in_dim_M or arr.out_dim_M or arr.out_dim_N or arr.in_dim_N != 1:
        raise ValueError("routine_step needs I -> <1|0>")
    P, omega, z = arr.P, arr.omega, jnp.zeros(0)

    @jax.jit
    def step(s, u):
        xi_q, _, _ = omega(P.position_at(s, duration), z, jnp.reshape(u, (1,)))
        return P.step_over(s, xi_q, duration)

    O, s_phi = Phi(arr), P.init
    for u in check_inputs:
        _, _, s_new = O.with_state(s_phi).run_one(
            (z, trivial_omega(0)), lambda _o, u=u: (z, jnp.array([float(u)])), duration)
        s_fast = step(s_phi, float(u))
        for x, y in zip(jax.tree_util.tree_leaves(s_new), jax.tree_util.tree_leaves(s_fast)):
            if not np.allclose(np.asarray(x), np.asarray(y), rtol=1e-12, atol=1e-14):
                raise AssertionError("routine_step differs from Phi")
        s_phi = s_new
    return step


def set_position(arr: Arrangement, position) -> object:
    """The state whose position (leaves in order) is ``position``."""
    leaves, treedef = jax.tree_util.tree_flatten(arr.P.init)
    pos = jnp.asarray(np.asarray(position, dtype=float))
    out, i = [], 0
    for leaf in leaves:
        k = int(jnp.size(leaf))
        out.append(pos[i:i + k].reshape(jnp.shape(leaf)))
        i += k
    assert i == pos.shape[0]
    return jax.tree_util.tree_unflatten(treedef, out)


def run(arr: Arrangement, step, inputs, start=None, record: bool = False):
    """Run on the schedule ``inputs`` (one routine value per step) from ``start`` (a position) or the initial state;
    returns the final position, or every step's."""
    inputs = jnp.asarray(np.asarray(inputs, dtype=float))
    s0 = arr.P.init if start is None else set_position(arr, start)

    def body(s, u):
        s = step(s, u)
        return s, (arr.P.position(s) if record else None)

    s, path = jax.lax.scan(body, s0, inputs)
    return np.asarray(path) if record else np.asarray(arr.P.position(s))


def run_batch(arr: Arrangement, step, inputs, starts):
    """Final positions from each start position in ``starts`` (an array, one row per start), on one schedule."""
    inputs = jnp.asarray(np.asarray(inputs, dtype=float))
    leaves, treedef = jax.tree_util.tree_flatten(arr.P.init)
    sizes = [int(jnp.size(l)) for l in leaves]
    shapes = [jnp.shape(l) for l in leaves]

    def to_state(pos):
        out, i = [], 0
        for k, sh in zip(sizes, shapes):
            out.append(pos[i:i + k].reshape(sh))
            i += k
        return jax.tree_util.tree_unflatten(treedef, out)

    @jax.jit
    def go(starts):
        def one(pos):
            s, _ = jax.lax.scan(lambda s, u: (step(s, u), None), to_state(pos), inputs)
            return arr.P.position(s)
        return jax.vmap(one)(starts)

    return np.asarray(go(jnp.asarray(np.asarray(starts, dtype=float))))


def open_step(arr: Arrangement, duration: float, check_at=None):
    """One step of an arrangement ``<0|k> -> I`` whose open boxes read out ``m`` (a held value per box):
    ``(s, m) |-> u(s, omega(q, m, ())|_Q)``, compiled; at ``check_at`` it is checked against the closed arrangement
    with each open box filled by the stateless system reading out that value (``constant``), stepped by ``Phi``."""
    if arr.in_dim_M or arr.in_dim_N or arr.out_dim_N:
        raise ValueError("open_step needs <0|k> -> I")
    P, omega, z = arr.P, arr.omega, jnp.zeros(0)

    @jax.jit
    def step(s, m):
        xi_q, _, _ = omega(P.position_at(s, duration), jnp.asarray(m), z)
        return P.step_over(s, xi_q, duration)

    if check_at is not None:
        m = jnp.asarray(np.asarray(check_at, dtype=float))
        closed = compose_seq(tensor_arrangements([constant(float(v)) for v in check_at]), arr)
        _, _, s_new = Phi(closed).with_state(closed.P.init).run_one(
            (z, trivial_omega(0)), lambda _o: (z, z), duration)
        s_new = s_new[1]      # the constants' trivial matter, then arr's
        for x, y in zip(jax.tree_util.tree_leaves(s_new), jax.tree_util.tree_leaves(step(P.init, m))):
            if not np.allclose(np.asarray(x), np.asarray(y), rtol=1e-12, atol=1e-14):
                raise AssertionError("open_step differs from Phi")
    return step


def person_open(q0: float, a=1.0, c=3.0, ell=1.0, b=2.6, w0=2.5) -> Arrangement:
    """Person 2's unit with the talk coupling, its partner and its chair left open: ``<0|2> -> I`` on
    ``(partner, chair)``. Its matter is person 2 alone; a certificate for it assumes where the partner and the chair
    are (the routine is off)."""
    unit = unit_quick(q0, w0, a, b)                                   # <0|1> (chair) -> <0|2> (p, p)
    with_partner = tensor_arrangements([identity(1), unit])            # <0|2> (partner, chair) -> <0|3> (v, p, p)
    wired = compose_seq(with_partner, route(3, [0, 1]))                # (v, p): talk hears partner then person
    return replace(compose_seq(wired, talk(c, ell)), label="person_open")
