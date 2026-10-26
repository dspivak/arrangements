"""Copying as a learner: a voice that sings along, and the ear that corrects it (book P27).

The book's agreed statement (plan P27, *How a form comes to be held*): understanding is building in yourself
something that would produce what arrives; copying is the case where that something is you, producing it. Here the
something is a voice, and the arrangement is

    W = E . (V (x) id_2) : <0 | 2> -> I,

whose external input is the sound at the ear, the source's value ``s`` and its loudness ``K``, on the environment's
schedule. Its two internal boxes:

* the **voice** ``V : I -> <0 | 1>``, configuration ``q = (w1, w2, theta)`` with the response
  ``diag(eta_w, eta_w, eta_theta)``. The *fast part* ``(w1, w2)`` is the last two values it sang (where it is in the
  note, how loud); the *slow part* ``theta = 2 cos(omega)`` is the pitch it can produce. It sends what it sings next,
  ``y = theta w1 - w2``, and its own forcing steps its window, ``(w1, w2) -> (y, w1)`` when nothing pushes. That step is
  a rotation, so the forcing is not exact (a sung note descends no cost); the recurrence keeps
  ``w1^2 - theta w1 w2 + w2^2``, so a voice left alone holds its loudness.
* the **ear** ``E : <0 | 3> -> I``, no state, the cost ``(K/2)(y - s)^2``. Its pushback on the voice's wire is
  ``K (s - y)``, which reaches the voice by the pullback along ``y``: ``K (s - y)(theta, -1)`` on the fast part and
  ``K (s - y) w1`` on the slow part. Since what the voice sends depends on ``theta``, the pushback reaches the pitch.

Not a singer: the names are only names. In a person sound pushes nothing back on the larynx, and the correction runs
through the brain over 100-200 ms; here the pushback runs back along the wire within a step (book P27, *First learner: run*).

Predictions and verdicts: book repo, ``notes/runs/p27-copying-dfb252e2/``; tests in ``tests/test_copying_learner.py``.
"""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .arrangement import Arrangement, exact, zero_form
from .coalgebra import CotCoalg, configuration
from .functors import Phi
from .interpretation import trivial_omega
from .rvect import diagonal
from .wiring import compose_seq, parallel_arrangements, trivial_parameter


def voice(eta_w: float, eta_theta: float, init) -> Arrangement:
    """The voice ``V : I -> <0 | 1>``: a window of its last two values, and a pitch."""

    def sing(q):
        w1, w2, theta = q
        return theta * w1 - w2

    def omega(q, m_out, n_in):
        w1, w2, _ = q
        y = sing(q)
        xi_q = jnp.array([(y - w1) / eta_w, (w1 - w2) / eta_w, 0.0])
        return xi_q, jnp.zeros_like(m_out), jnp.zeros_like(n_in)

    return Arrangement(
        P=configuration(diagonal(jnp.array([eta_w, eta_w, eta_theta])), init=init),
        in_dim_M=0,
        out_dim_M=0,
        in_dim_N=0,
        out_dim_N=1,
        out_f=lambda q, m: jnp.reshape(sing(q), (1,)),
        in_f=lambda q, m, n: jnp.zeros(0),
        omega=omega,
        label="voice",
    )


def identity(d: int) -> Arrangement:
    """The identity wire on ``<0 | d>``."""
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0,
        out_dim_M=d,
        in_dim_N=0,
        out_dim_N=d,
        out_f=lambda q, m: m,
        in_f=lambda q, m, n: jnp.zeros(0),
        omega=zero_form,
        label=f"id{d}",
    )


def ear() -> Arrangement:
    """The ear ``E : <0 | 3> -> I`` on ``(y, s, K)``: the cost ``(K/2)(y - s)^2``."""
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0,
        out_dim_M=3,
        in_dim_N=0,
        out_dim_N=0,
        out_f=lambda q, m: jnp.zeros(0),
        in_f=lambda q, m, n: jnp.zeros(0),
        omega=exact(lambda q, m, n: -0.5 * m[2] * (m[0] - m[1]) ** 2),
        label="ear",
    )


def singing_along(eta_w: float, eta_theta: float, init) -> Arrangement:
    """``W = E . (V (x) id_2) : <0 | 2> -> I``; the external input is ``(s, K)``."""
    return compose_seq(parallel_arrangements(voice(eta_w, eta_theta, init), identity(2)), ear())


def voice_q(state) -> np.ndarray:
    """The voice's ``(w1, w2, theta)`` from ``W``'s state ``((s_V, s_id), s_E)``."""
    return np.asarray(state[0][0])


def driven_step(arr: Arrangement, check_inputs: Sequence = ()) -> Callable:
    """One step of ``Phi(arr)`` for ``arr : <0 | m> -> I``, compiled, checked against ``Phi``.

    With no backward ports, the covector triple's ``Q`` part is the forcing's own ``Q`` component at the input value
    ``m``, so the step is ``(s, m) |-> u(s, omega(q(s), m, ())|_Q)``. Each input in ``check_inputs`` is also stepped
    through ``Phi(arr).run_one`` from the current state and the two compared, as ``functors.closed_step`` does.
    """
    if arr.in_dim_M or arr.in_dim_N or arr.out_dim_N:
        raise ValueError("driven_step needs <0 | m> -> I")
    P, omega, z = arr.P, arr.omega, jnp.zeros(0)

    @jax.jit
    def step(s, m):
        xi_q, _, _ = omega(P.position(s), m, z)
        return P.step(s, xi_q)

    O, s_phi = Phi(arr), P.init
    for m in check_inputs:
        m = jnp.asarray(m, dtype=float)
        _, _, s_new = O.with_state(s_phi).run_one((m, trivial_omega(0)), lambda _o: (jnp.zeros(0), jnp.zeros(0)))
        s_fast = step(s_phi, m)
        for a, b in zip(jax.tree_util.tree_leaves(s_new), jax.tree_util.tree_leaves(s_fast)):
            if not np.allclose(np.asarray(a), np.asarray(b), atol=1e-12):
                raise AssertionError("driven_step differs from Phi")
        s_phi = s_new
    return step


def run(arr: Arrangement, step: Callable, state, schedule: Sequence[Tuple[float, float]]):
    """Run ``W`` on the environment's schedule of ``(s, K)``; return the final state and, per step, ``(q, y, s, K)``."""
    qs, ys = [], []
    for m in schedule:
        q = voice_q(state)
        qs.append(q)
        ys.append(q[2] * q[0] - q[1])
        state = step(state, jnp.asarray(m, dtype=float))
    sched = np.asarray(schedule, dtype=float)
    return state, np.array(qs), np.array(ys), sched


def tone(omega: float, amp: float = 1.0, phase: float = 0.0, t0: int = 0):
    """The source's value at step ``t``: ``amp sin(omega (t0 + t) + phase)``."""
    return lambda t: amp * np.sin(omega * (t0 + t) + phase)


def schedule(source: Callable[[int], float], steps: int, K: float = 1.0):
    """The environment's schedule: ``source`` at loudness ``K`` for ``steps`` steps (``K = 0``: silence, ``s = 0``)."""
    return [((source(t) if K else 0.0), K) for t in range(steps)]


def pitch(theta: float) -> float:
    """The pitch ``omega`` with ``theta = 2 cos(omega)``."""
    return float(np.arccos(np.clip(theta / 2.0, -1.0, 1.0)))


def loudness(q) -> float:
    """The amplitude the window holds: ``sqrt((w1^2 - theta w1 w2 + w2^2) / sin(omega)^2)``, the recurrence's invariant."""
    w1, w2, theta = q
    return float(np.sqrt(max(w1 * w1 - theta * w1 * w2 + w2 * w2, 0.0) / max(1.0 - theta * theta / 4.0, 1e-300)))


# ---------------------------------------------------------------------------
# Template copying: a row of two-well parts, each tipped by its partner in a source row (book P27, the survey's
# common case: part meets part, the copy held by its own wells).
# ---------------------------------------------------------------------------


def two_well(alpha: float, eta: float, init: float) -> Arrangement:
    """A part ``I -> <0 | 1>`` with two wells, cost ``(alpha/4)(q^2 - 1)^2``, sending ``q`` (a latch, a fixed base)."""
    return Arrangement(
        P=configuration(diagonal(jnp.array([eta])), init=jnp.array([float(init)])),
        in_dim_M=0,
        out_dim_M=0,
        in_dim_N=0,
        out_dim_N=1,
        out_f=lambda q, m: q,
        in_f=lambda q, m, n: jnp.zeros(0),
        omega=exact(lambda q, m, n: -0.25 * alpha * (q[0] ** 2 - 1.0) ** 2),
        label="part",
    )


def contact(g: float) -> Arrangement:
    """One contact ``<0 | 3> -> I`` on ``(a, b, K)``: the cost ``(g K / 2)(a - b)^2``, conducting with the write-enable ``K``.

    Its pushback reaches both parts it joins (a pass transistor between two latches; pairing at a template).
    """
    return Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=3, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m: jnp.zeros(0), in_f=lambda q, m, n_in: jnp.zeros(0),
        omega=exact(lambda q, m, n_in: -0.5 * g * m[2] * (m[0] - m[1]) ** 2),
        label="contact",
    )


def contact_one_way(g: float) -> Arrangement:
    """One one-way contact ``<0 | 3> -> I`` on ``(a, b, K)``: the forcing ``g K (a - b)`` on ``b`` and nothing on ``a``.

    Not the differential of any cost (Ch 2's waterfall, sec.waterfall): the copy is pushed toward the source and nothing is
    pulled back to the source, as light off a still life is absorbed by the eye and never returns to the fruit.
    """
    def omega(q, m, n_in):
        return jnp.zeros_like(q), jnp.array([0.0, g * m[2] * (m[0] - m[1]), 0.0]), jnp.zeros_like(n_in)

    return Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=3, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m: jnp.zeros(0), in_f=lambda q, m, n_in: jnp.zeros(0), omega=omega, label="one-way contact",
    )


def pair_wire(n: int) -> Arrangement:
    """The wiring ``<0 | 2n + 1> -> <0 | 3n>``: ``(a_1..a_n, b_1..b_n, K) |-> ((a_1, b_1, K), ..., (a_n, b_n, K))``.

    Static, zero forcing: it routes each source part and its partner to their contact and copies the write-enable to every
    contact; its backward map, the transpose, sums the copies' covectors (as ``wiring.chain_wire`` routes a chain).
    """
    idx = jnp.array([k for i in range(n) for k in (i, n + i, 2 * n)])
    return Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=2 * n + 1, in_dim_N=0, out_dim_N=3 * n,
        out_f=lambda q, m: m[idx], in_f=lambda q, m, n_in: jnp.zeros(0), omega=zero_form, label=f"pair_wire({n})",
    )


def contacts(n: int, g: float, one_way: bool = False) -> Arrangement:
    """The ``n`` contacts composed: ``(C_1 (x) ... (x) C_n) . pair_wire(n) : <0 | 2n + 1> -> I``."""
    C = [contact_one_way(g) if one_way else contact(g) for _ in range(n)]
    row = C[0]
    for c in C[1:]:
        row = parallel_arrangements(row, c)
    return compose_seq(pair_wire(n), row)


def template(a0, b0, alpha: float = 4.0, g: float = 1.0, eta: float = 0.05, one_way: bool = False) -> Arrangement:
    """``W = C . ((A_1 (x) .. (x) A_n) (x) (B_1 (x) .. (x) B_n) (x) id_1) : <0 | 1> -> I``; the external input is ``K``.

    The source parts have depth ``alpha``, the copy parts depth 1.
    """
    n = len(a0)
    A = [two_well(alpha, eta, a) for a in a0]
    B = [two_well(1.0, eta, b) for b in b0]
    parts = A + B
    row = parts[0]
    for p in parts[1:]:
        row = parallel_arrangements(row, p)
    C = contacts(n, g, one_way)
    return compose_seq(parallel_arrangements(row, identity(1)), C)


def template_states(arr: Arrangement, state, n: int):
    """The source and copy rows ``(a, b)`` read off ``W``'s state by its position."""
    q = np.asarray(arr.P.position(state))
    return q[:n], q[n:2 * n]


def run_scan(step: Callable, state, inputs):
    """``step`` folded over ``inputs`` (one external input per step) by ``jax.lax.scan``; returns the final state."""
    def body(s, m):
        return step(s, m), None
    final, _ = jax.lax.scan(body, state, jnp.asarray(inputs, dtype=float))
    return final


def template_init(arr: Arrangement, a0, b0):
    """``W``'s state with the source row at ``a0`` and the copy row at ``b0`` (the parts' leaves come first, in order)."""
    leaves, tree = jax.tree_util.tree_flatten(arr.P.init)
    vals = list(a0) + list(b0)
    new = [jnp.array([float(v)]) for v in vals] + leaves[len(vals):]
    return jax.tree_util.tree_unflatten(tree, new)


# ---------------------------------------------------------------------------
# A copy and its original: two voices of one interface, the copy coupled one way to the original (book Ch 3
# ``ex.copying_learner``, David, 2026-10-19: "I don't want the two oscillators to learn each other … the copier is not
# supposed to push back on the other one").
# ---------------------------------------------------------------------------


def oscillator(lam: float, eta: float, init) -> CotCoalg:
    """The oscillator's matter (book Ch 3 ``ex.oscillator``, ``eqn.osc_integrator``): states ``(alpha, nu)``.

    ``alpha`` is the angle, a point of the circle ``R / 2 pi Z`` (held here as a real number; only ``sin`` and ``cos``
    of it are ever read), and ``nu`` the angular frequency in radians per unit of duration. The configuration space is
    the circle: told a duration ``tau`` it reads out where its point is headed, the look-ahead ``alpha + tau nu``, as
    the mass on ice reads out ``q + tau p / m``; told a force ``xi`` there it updates to
    ``(alpha + tau nu + tau lam xi, nu + tau eta xi)``. Without any force a step of duration ``tau`` turns the angle by
    ``tau nu``, so steps of any durations adding to ``t`` land where one step of ``t`` does (David, 2026-10-20: "what you
    see when you do smaller time steps is just a more precise version"). The angle moves by ``lam`` times the force, as a
    parameter does; the angular frequency gains ``eta`` times it, as the mass's momentum gains the impulse.
    """
    def timed_step(s, xi, tau):
        alpha, nu = s[0], s[1]
        return jnp.array([alpha + tau * nu + tau * lam * xi[0], nu + tau * eta * xi[0]])

    return CotCoalg(
        Q_dim=1,
        init=jnp.asarray(init, dtype=float),
        position=lambda s: jnp.reshape(s[0] + s[1], (1,)),
        step=lambda s, xi: timed_step(s, xi, 1.0),
        label="oscillator",
        timed_step=timed_step,
        look_ahead=lambda s, tau: jnp.reshape(s[0] + tau * s[1], (1,)),
    )


def osc(lam: float, eta: float, init, amp: float = 1.0) -> Arrangement:
    """The system ``osc : I -> <0 | 1>`` (book Ch 3 ``ex.oscillator``): the oscillator's matter in the box of the static
    arrangement whose forward map sends the configuration ``q`` on the circle to ``amp sin q``; effort field 0, so a
    force ``xi`` at the output reaches the matter only by the transpose of that map's differential, ``xi amp cos q``."""
    return Arrangement(
        P=oscillator(lam, eta, init),
        in_dim_M=0,
        out_dim_M=0,
        in_dim_N=0,
        out_dim_N=1,
        out_f=lambda q, m: jnp.reshape(amp * jnp.sin(q[0]), (1,)),
        in_f=lambda q, m, n: jnp.zeros(0),
        omega=zero_form,
        label="osc",
    )


def copy_pair(eta_w: float, eta_theta: float, init_original, init_copy) -> Arrangement:
    """``W = C . (V_orig (x) V_copy (x) id_1) : <0 | 1> -> I``; the external input is the coupling ``K``.

    Two ``osc`` (book Ch 3 ``ex.oscillator``) of one kind, the same matter, started at different angular frequencies. The coupling ``C`` is
    ``contact_one_way(1)`` on ``(x, y, K)``: the forcing ``K (x - y)`` on the copy's value ``y`` and nothing on the
    original's ``x`` (Ch 2's waterfall), so the original receives no force and runs as a free oscillator; while
    ``K = 1`` the copy's angular frequency is pulled to the original's, and at ``K = 0`` each runs on its own.
    """
    V = parallel_arrangements(osc(eta_w, eta_theta, init_original), osc(eta_w, eta_theta, init_copy))
    return compose_seq(parallel_arrangements(V, identity(1)), contact_one_way(1.0))


def copy_pair_q(arr: Arrangement, state):
    """The original's and the copy's states ``(alpha, nu)``, read off ``W``'s state ``(((s_orig, s_copy), _), _)``."""
    (s_orig, s_copy), _ = state[0]
    return np.asarray(s_orig), np.asarray(s_copy)


def couple() -> Arrangement:
    """The system ``couple : I -> <3 | 0>`` (book Ch 3 ``ex.copying_learner``): stateless, inputs only, receiving the
    original's value ``x``, the signal ``kappa`` and the copy's value ``y``, in that order.

    It pushes back ``kappa (x - y)`` at ``y`` and ``0`` at ``x`` and at ``kappa``: the forcing of ``contact_one_way(1)``,
    with its three values arriving at inputs of its own.
    """
    def omega(q, m, n_in):
        x, kappa, y = n_in
        return jnp.zeros_like(q), jnp.zeros_like(m), jnp.array([0.0, 0.0, kappa * (x - y)])

    return Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=0, in_dim_N=3, out_dim_N=0,
        out_f=lambda q, m: jnp.zeros(0), in_f=lambda q, m, n_in: jnp.zeros(0), omega=omega, label="couple",
    )


def copier_wire() -> Arrangement:
    """The copier's wiring, static: ``<3 | 2> -> <1 | 2>`` from the boxes ``osc``, ``osc`` and ``couple`` to the copier.

    The original's value ``x`` and the copy's ``y`` split, each to an external output and to ``couple``, and the external
    input ``kappa`` goes to ``couple``: forward ``(x, y) |-> (x, y)``, and ``couple``'s inputs ``(x, kappa, y)``.
    """
    return Arrangement(
        P=trivial_parameter(), in_dim_M=3, out_dim_M=2, in_dim_N=1, out_dim_N=2,
        out_f=lambda q, m: m, in_f=lambda q, m, n_in: jnp.array([m[0], n_in[0], m[1]]), omega=zero_form,
        label="copier_wire",
    )


def copy_pair_out(eta_w: float, eta_theta: float, init_original, init_copy) -> Arrangement:
    """The system ``copier : I -> <1 | 2>`` of book Ch 3 ``ex.copying_learner``: ``copier_wire`` with its boxes filled by
    the original ``osc``, the copy ``osc`` and ``couple``. The signal ``kappa`` arrives at its external input, and the
    original's value ``x`` and the copy's ``y`` are sent out (David, 2026-10-19: "neither of these two X's and Y's are
    exported out, so how am I even hearing any of this stuff?"). With the force ``0`` at its outputs it steps as
    ``copy_pair`` told ``K = kappa``."""
    boxes = parallel_arrangements(
        parallel_arrangements(osc(eta_w, eta_theta, init_original), osc(eta_w, eta_theta, init_copy)), couple())
    return compose_seq(boxes, copier_wire())


def copier_q(arr: Arrangement, state):
    """The original's and the copy's states ``(alpha, nu)``, read off ``copier``'s state ``(((s_orig, s_copy), _), _)``."""
    (s_orig, s_copy), _ = state[0]
    return np.asarray(s_orig), np.asarray(s_copy)


def copier_step(arr: Arrangement, check_inputs: Sequence = ()) -> Callable:
    """One step of ``Phi(arr)`` for a system ``arr : I -> <n | m>`` told its input values and the force ``0`` at its
    outputs, compiled, checked against ``Phi``, as ``driven_step`` is for ``<0 | m> -> I``.

    With no ``M`` ports and no force at the outputs, the step is ``(s, n_in) |-> u(s, omega(q(s), (), n_in)|_Q)``. Each
    input in ``check_inputs`` is also stepped through ``Phi(arr).run_one`` from the current state and the two compared.
    """
    if arr.in_dim_M or arr.out_dim_M:
        raise ValueError("copier_step needs a system I -> <n | m>")
    P, omega, z = arr.P, arr.omega, jnp.zeros(0)

    @jax.jit
    def step(s, n_in):
        xi_q, _, _ = omega(P.position(s), z, n_in)
        return P.step(s, xi_q)

    O, s_phi = Phi(arr), P.init
    for n_in in check_inputs:
        n_in = jnp.asarray(n_in, dtype=float)
        _, _, s_new = O.with_state(s_phi).run_one((z, trivial_omega(0)), lambda _o: (jnp.zeros(arr.out_dim_N), n_in))
        s_fast = step(s_phi, n_in)
        for a, b in zip(jax.tree_util.tree_leaves(s_new), jax.tree_util.tree_leaves(s_fast)):
            if not np.allclose(np.asarray(a), np.asarray(b), atol=1e-12):
                raise AssertionError("copier_step differs from Phi")
        s_phi = s_new
    return step
