"""A reader with a quick part and slow parts, each its own box (book Ch 1 ``sec.held_form``; plan P43 (b) job 3).

Ch 1 says "a reader has slow parts as well as quick ones, and where the slower parts sit sets out the
shape that decides where the quick ones can settle." ``tests/test_carving.py`` and
``tests/test_carving_which.py`` ran this as one box whose state held a quick coordinate and slow ones
under one cost (P43 (a), cb545383). Here each is a part:

* the QUICK part (``quick_box``): state ``q``, response ``eta_q``; it hears the note ``u`` on the
  reader's port and pays ``q^4/4 - u q``; it reads ``q`` out to each coupling;
* the TUNING (``tuning_box``), a slow part: state ``w``, response ``eta_w``, its own cost
  ``gamma (w - w0)^2/2 + beta (w - w0)^4/4`` (disuse decays, deepening saturates);
* the BIAS (``bias_box``), a slow part: state ``b``, response ``eta_b``, cost ``rho b^2/2``;
* stateless COUPLINGS between the quick part and a slow one: ``curvature_coupling``, cost
  ``-w q^2/2`` (the tuning sets the curvature of the quick part's landscape at 0), and
  ``tilt_coupling``, cost ``-b q`` (the bias tilts it). ``nonreciprocal_coupling(lam)`` returns
  ``w q`` to the quick part, as the curvature coupling does, but ``lam q`` to the tuning instead of
  ``q^2/2``: a 1-form with no cost behind it.

``reader`` composes them (``compose_seq``, ``tensor_arrangements``) into ``<0|0> -> <1|0>``, the note
arriving on its one port; ``Phi`` runs it, the note the environment's schedule. The sum of the parts'
and couplings' costs is the one cost of the old one-box readers, so the composite's step is theirs;
the tests check it, at the runs' settings.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .arrangement import Arrangement, exact, zero_form
from .coalgebra import configuration
from .functors import Phi
from .interpretation import trivial_omega
from .rvect import euclidean
from .wiring import compose_seq, tensor_arrangements, trivial_parameter


def _none(q, m_out, n_in):
    return jnp.zeros(0)


def _part(eta: float, x0: float, cost, degree: int, hears_note: bool, label: str) -> Arrangement:
    """A part with one coordinate: state ``x``, response ``eta``, read out to ``degree`` couplings."""
    return Arrangement(
        P=configuration(euclidean(1, eta), init=jnp.array([x0], dtype=float)),
        in_dim_M=0, out_dim_M=0, in_dim_N=1 if hears_note else 0, out_dim_N=degree,
        out_f=lambda q, m_out: jnp.tile(q, degree),
        in_f=_none,
        omega=exact(lambda q, m_out, n_in: -cost(q[0], n_in)),
        label=label,
    )


def quick_box(eta_q: float, q0: float, degree: int) -> Arrangement:
    """The quick part ``<0|0> -> <1|degree>``: hears the note ``u``, pays ``q^4/4 - u q``."""
    return _part(eta_q, q0, lambda q, n: 0.25 * q ** 4 - n[0] * q, degree, True, "quick")


def tuning_box(eta_w: float, w_start: float, w0: float, gamma: float, beta: float) -> Arrangement:
    """The tuning ``<0|0> -> <0|1>``: pays ``gamma (w - w0)^2/2 + beta (w - w0)^4/4``."""
    return _part(eta_w, w_start,
                 lambda w, n: 0.5 * gamma * (w - w0) ** 2 + 0.25 * beta * (w - w0) ** 4, 1, False, "tuning")


def bias_box(eta_b: float, b_start: float, rho: float) -> Arrangement:
    """The bias ``<0|0> -> <0|1>``: pays ``rho b^2/2``."""
    return _part(eta_b, b_start, lambda b, n: 0.5 * rho * b ** 2, 1, False, "bias")


def _coupling(omega, label: str) -> Arrangement:
    return Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=2, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0), in_f=_none, omega=omega, label=label,
    )


def curvature_coupling() -> Arrangement:
    """``<0|2> -> I`` on ``(q, w)``, cost ``-w q^2/2``: returns ``w q`` to ``q`` and ``q^2/2`` to ``w``."""
    return _coupling(exact(lambda s, m, n: 0.5 * m[1] * m[0] ** 2), "curvature")


def tilt_coupling() -> Arrangement:
    """``<0|2> -> I`` on ``(q, b)``, cost ``-b q``: returns ``b`` to ``q`` and ``q`` to ``b``."""
    return _coupling(exact(lambda s, m, n: m[1] * m[0]), "tilt")


def nonreciprocal_coupling(lam: float) -> Arrangement:
    """``<0|2> -> I`` on ``(q, w)``: returns ``w q`` to ``q`` and ``lam q`` to ``w``; no cost has this
    1-form as its derivative unless ``q = lam`` (its two cross derivatives are ``q`` and ``lam``)."""
    def omega(s, m, n):
        q, w = m[0], m[1]
        return jnp.zeros_like(s), jnp.array([w * q, lam * q]), jnp.zeros_like(n)

    return _coupling(omega, f"nonreciprocal({lam:g})")


def _route(out_dim: int, idx: Sequence[int]) -> Arrangement:
    """The wiring ``<1|out_dim> -> <1|len(idx)>``: presents the listed wires, and carries the note
    from the reader's port to the quick part."""
    idx = jnp.asarray(list(idx), dtype=int)
    return Arrangement(
        P=trivial_parameter(), in_dim_M=1, out_dim_M=out_dim, in_dim_N=1, out_dim_N=len(idx),
        out_f=lambda q, m_out: m_out[idx], in_f=lambda q, m_out, n_in: n_in,
        omega=zero_form, label="route",
    )


def _port() -> Arrangement:
    """The reader's port ``<1|0> -> <1|0>``: the note passes inward."""
    return Arrangement(
        P=trivial_parameter(), in_dim_M=1, out_dim_M=0, in_dim_N=1, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0), in_f=lambda q, m_out, n_in: n_in,
        omega=zero_form, label="port",
    )


def reader(slow_parts: Sequence[Arrangement], couplings: Sequence[Arrangement], q0: float,
           eta_q: float) -> Arrangement:
    """The reader ``<0|0> -> <1|0>``: the quick part, the slow parts, and coupling ``i`` between the
    quick part and slow part ``i``; the note on its port. Its state's position is ``(q, *slow)``."""
    n = len(slow_parts)
    assert len(couplings) == n
    parts = tensor_arrangements([quick_box(eta_q, q0, n)] + list(slow_parts))
    # wires out of the parts: [q (n copies), slow_1, ..., slow_n]; each coupling hears (q, slow_i)
    routed = compose_seq(parts, _route(2 * n, [x for i in range(n) for x in (i, n + i)]))
    wired = compose_seq(routed, tensor_arrangements(list(couplings) + [_port()]))
    return replace(wired, label=f"reader({', '.join(p.label for p in slow_parts)})")


def note_step(arr: Arrangement, check_notes: Sequence[float] = (0.6, 0.0)):
    """One ``Phi`` step of a reader ``<0|0> -> <1|0>`` on a note, compiled, checked against ``Phi``.

    The reader presents nothing, so the covector triple's ``Q`` part is its forcing's at the note:
    the step is ``(s, u) |-> u(s, omega(q(s), (), u)|_Q)``. Each note in ``check_notes`` is also
    stepped through ``Phi(arr).run_one`` and the two compared, as ``functors.closed_step`` does.
    """
    if arr.in_dim_M or arr.out_dim_M or arr.out_dim_N or arr.in_dim_N != 1:
        raise ValueError("note_step needs <0|0> -> <1|0>")
    P, omega, z = arr.P, arr.omega, jnp.zeros(0)

    @jax.jit
    def step(s, u):
        xi_q, _, _ = omega(P.position(s), z, jnp.reshape(u, (1,)))
        return P.step(s, xi_q)

    O, s_phi = Phi(arr), P.init
    for u in check_notes:
        _, _, s_new = O.with_state(s_phi).run_one(
            (z, trivial_omega(0)), lambda _o, u=u: (z, jnp.array([float(u)])))
        s_fast = step(s_phi, float(u))
        for a, b in zip(jax.tree_util.tree_leaves(s_new), jax.tree_util.tree_leaves(s_fast)):
            if not np.allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-14):
                raise AssertionError("note_step differs from Phi")
        s_phi = s_new
    return step


def start_at(arr: Arrangement, position) -> object:
    """The reader's state whose position is ``(q, *slow)`` (its couplings and wires hold nothing)."""
    leaves, treedef = jax.tree_util.tree_flatten(arr.P.init)
    pos = jnp.asarray(np.asarray(position, dtype=float))
    out, i = [], 0
    for leaf in leaves:
        k = int(jnp.size(leaf))
        out.append(pos[i:i + k].reshape(jnp.shape(leaf)))
        i += k
    assert i == pos.shape[0]
    return jax.tree_util.tree_unflatten(treedef, out)


def play(arr: Arrangement, step, notes, record: bool = False, start=None):
    """Run the reader on the environment's schedule of notes, from its initial state or from the
    position ``start``. Returns the final position ``(q, *slow)``, or every step's when ``record``."""
    notes = jnp.asarray(np.asarray(notes, dtype=float))
    s0 = arr.P.init if start is None else start_at(arr, start)

    def body(s, u):
        s = step(s, u)
        return s, (arr.P.position(s) if record else None)

    s, path = jax.lax.scan(body, s0, notes)
    return np.asarray(path) if record else np.asarray(arr.P.position(s))
