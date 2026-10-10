"""A jointed arm pulled at its tip by springs, composed (book plan P43 (b) job 7; ``tests/test_linkage_backprop.py``).

The picture (David, 2026-09-27): a planar arm whose joint angles are the parameters; for each training example
a copy of the arm with its base turned by the input, all copies' joints geared together; a zero-length spring
from each copy's tip to its target; the joints in syrup, or in air with friction (momentum). The test ran it
as plain jax/numpy (P43 (a), cb545383). Here each piece is a box:

* the JOINTS, state the joint angles: in syrup the configuration integrator with response ``dt``; in air the
  damped phase integrator (``coalgebra.damped``, response 1, keeping ``beta`` of the momentum, the springs'
  costs scaled by the step ``lr``). The gearing is the copy of the joints' angles to every copy of the arm;
* for the two-link arm, LINK boxes per copy: the first turns the base by the input and the first joint and
  presents its end and its direction; the second adds the second joint and presents the tip. The torque on a
  joint is what ``Phi`` passes back through the links, ``J^T F``, nobody computing a gradient;
* for the hundred-link arm, one gray box per copy, the arm's shape as one smooth map from the angles to the tip;
* a TARGET per copy, a box with no state presenting the target, and a SPRING per copy between tip and target.

``two_link_arm`` and ``many_link_arm`` compose them, closed. Their position is the joint angles.
"""

from __future__ import annotations

from typing import Callable, Sequence

import jax
import jax.numpy as jnp

from .arrangement import Arrangement, zero_form
from .coalgebra import CotCoalg, configuration, damped
from .network import fan_box, map_box
from .rod import spring_box
from .rvect import euclidean
from .wiring import compose_seq, tensor_arrangements


def joints_box(q0, *, dt: float = None, beta: float = None, n_out: int = 1) -> Arrangement:
    """The geared joints ``<0|0> -> <0|n n_out>``: syrup (``dt``) or air keeping ``beta`` of the momentum.
    The angles are presented ``n_out`` times, once per copy of the arm."""
    q0 = jnp.asarray(q0, dtype=float)
    n = q0.shape[0]
    if dt is not None:
        P = configuration(euclidean(n, dt), init=q0)
    else:
        P = damped(euclidean(n, 1.0), 1.0 - beta, init=(q0, jnp.zeros(n)))
    return Arrangement(
        P=P, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=n * n_out,
        out_f=lambda q, m: jnp.tile(q, n_out), in_f=lambda q, m, nn: jnp.zeros(0), omega=zero_form,
        label="joints",
    )


def geared_joints(q0, beta: float, n_out: int = 1) -> Arrangement:
    """The geared joints in air with friction, their step held in the state: ``<0|0> -> <0|n n_out>``.

    State ``(q, p, lr)``: the angles, the momentum and the step. It presents ``q + lr p``, the angles the
    step moves to, and on the torque ``tau`` there moves to ``(q + lr p, beta p + tau, lr)``: heavy-ball
    momentum keeping ``beta`` of the velocity ``lr p``. ``beta = 0`` is syrup at step ``lr``. ``lr`` never
    moves; it is set by hand where a run starts (an experimenter's choice, said so where it is used)."""
    q0 = jnp.asarray(q0, dtype=float)
    n = q0.shape[0]
    P = CotCoalg(
        Q_dim=n, init=(q0, jnp.zeros(n), jnp.ones(1)),
        position=lambda s: s[0] + s[2] * s[1],
        step=lambda s, tau: (s[0] + s[2] * s[1], beta * s[1] + tau, s[2]),
        label=f"geared(beta={beta:g})",
    )
    return Arrangement(
        P=P, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=n * n_out,
        out_f=lambda q, m: jnp.tile(q, n_out), in_f=lambda q, m, nn: jnp.zeros(0), omega=zero_form,
        label="geared joints",
    )


def target_box(t) -> Arrangement:
    t = jnp.asarray(t, dtype=float)
    return map_box(lambda v: t, 0, 2, "target")


def first_link(x: float, L1: float) -> Arrangement:
    """``<0|1> -> <0|3>``: from the first joint's angle, the first link's end ``L1 u(x + q1)`` and its
    direction ``x + q1``."""
    def g(v):
        a = x + v[0]
        return jnp.stack([L1 * jnp.cos(a), L1 * jnp.sin(a), a])

    return map_box(g, 1, 3, "link 1")


def second_link(L2: float) -> Arrangement:
    """``<0|4> -> <0|2>``: from the first link's end and direction and the second joint's angle, the tip."""
    def g(v):
        a = v[2] + v[3]
        return jnp.stack([v[0] + L2 * jnp.cos(a), v[1] + L2 * jnp.sin(a)])

    return map_box(g, 4, 2, "link 2")


def _close(joints: Arrangement, copies: Sequence[Arrangement], targets, k: float, scale: float, n: int):
    """joints -> one arm per copy -> its tip and its target -> a spring."""
    m = len(copies)
    parts = tensor_arrangements([joints] + [target_box(t) for t in targets])
    arms = tensor_arrangements(list(copies) + [fan_box(2 * m, range(2 * m), "targets")])
    tips = compose_seq(parts, arms)                      # wires: m tips (2 each), then m targets (2 each)
    pair = [w for j in range(m) for w in (2 * j, 2 * j + 1, 2 * m + 2 * j, 2 * m + 2 * j + 1)]
    springs = tensor_arrangements([spring_box(k, scale) for _ in range(m)])
    return compose_seq(compose_seq(tips, fan_box(4 * m, pair, "pair")), springs)


def two_link_arm(q0, xs: Sequence[float], targets, *, L1: float, L2: float, k: float = 1.0,
                 dt: float = None, beta: float = None, lr: float = 1.0) -> Arrangement:
    """The two-link arm: geared joints, two link boxes per copy, a spring per copy. Closed."""
    m = len(xs)
    joints = joints_box(q0, dt=dt, beta=beta, n_out=m)

    def copy_arm(x):
        # wires in: (q1, q2); link 1 hears q1, link 2 hears link 1's (end, direction) and q2
        l1 = tensor_arrangements([first_link(x, L1), fan_box(1, [0], "q2")])
        return compose_seq(l1, second_link(L2))

    return _close(joints, [copy_arm(float(x)) for x in xs], targets, k, 1.0 if dt is not None else lr, 2)


def many_link_arm(q0, xs, targets, tip: Callable, *, k: float = 1.0, dt: float = None, beta: float = None,
                  lr: float = 1.0, held_step: bool = False) -> Arrangement:
    """The many-link arm: geared joints, the arm's shape ``tip(q, x)`` as one gray box per copy, a spring
    per copy. Closed. With ``held_step`` the joints are ``geared_joints`` (step in the state)."""
    q0 = jnp.asarray(q0, dtype=float)
    n, m = q0.shape[0], len(xs)
    joints = geared_joints(q0, beta, n_out=m) if held_step else joints_box(q0, dt=dt, beta=beta, n_out=m)
    copies = [map_box(lambda v, x=x: tip(v, x), n, 2, "arm") for x in xs]
    return _close(joints, copies, targets, k, 1.0 if dt is not None else lr, n)


def many_link_arm_power(q0, xs, targets, tip: Callable, *, k: float = 1.0, beta: float = 0.0) -> Arrangement:
    """``many_link_arm`` with ``held_step``, its ``m`` copies built as tensor powers: the arm-shape box, the
    target box and the spring box each the ``m``-fold tensor power of one, as one box acting copy by copy
    (``jax.vmap``). Equal to ``many_link_arm`` (checked in the test); built so for speed."""
    from .arrangement import exact
    from .wiring import trivial_parameter

    q0 = jnp.asarray(q0, dtype=float)
    xs = jnp.asarray(xs, dtype=float)
    T = jnp.asarray(targets, dtype=float).reshape(-1, 2)
    n, m = q0.shape[0], xs.shape[0]
    joints = geared_joints(q0, beta, n_out=m)
    arms = map_box(lambda v: jax.vmap(tip)(v.reshape(m, n), xs).reshape(-1), n * m, 2 * m, f"arm^{m}")
    targets_box = map_box(lambda v: T.reshape(-1), 0, 2 * m, f"target^{m}")
    springs = Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=4 * m, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, mm: jnp.zeros(0), in_f=lambda q, mm, nn: jnp.zeros(0),
        omega=exact(lambda q, mm, nn: -0.5 * k * jnp.sum((mm[:2 * m] - mm[2 * m:]) ** 2)),
        label=f"spring^{m}",
    )
    tips = compose_seq(tensor_arrangements([joints, targets_box]),
                       tensor_arrangements([arms, fan_box(2 * m, range(2 * m), "targets")]))
    return compose_seq(tips, springs)
