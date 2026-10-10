"""A population's note and the slower reader it reaches, composed (book plan P43 (b) job 7; ``tests/test_spells.py`` test 4).

David, 2026-09-26: "you can be a part of some longer-time-scale note that's playing to inchant something else."
The spells run's test 4 averaged 2000 members in plain code and fed the mean to a slower reader written as a
hand loop (P43 (a), cb545383). Here each piece is a box (David, 2026-10-10: "division is a box"):

* the MEMBERS, ``N`` committed readers, each with cost ``(1/4)(q^2 - 1)^2 - u q`` on the note ``u`` it hears
  and Euler--Maruyama noise at temperature ``T`` (``rand.euler_maruyama``), side by side: ``members`` builds
  the ``N``-fold tensor power directly, one vector of ``N`` states with the same forms coordinatewise, which
  ``tensor_arrangements`` of ``N`` copies is (checked equal for small ``N`` in the test);
* the NOTE's copy, a wire carrying the one note on the population's port to every member;
* the MEAN, a ``+`` box over the ``N`` members followed by a box multiplying by ``1/N``;
* the SLOW READER ``R2``, a committed reader with response ``eta_2`` and no noise, whose cost reads the mean
  ``m`` as its note, ``(1/4)(q_2^2 - 1)^2 - m q_2``, ONE-WAY: it reads the mean and pushes nothing back
  (its 1-form has no component on what it hears), as the old run's ``R2`` was driven by the mean and did not
  act on the members.

``population(N, ...)`` is ``<0|0> -> <1|0>``, the surrounds' note arriving on its port as the environment's
schedule; ``Phi`` of it, paired with a sampler (``rand.seed``), runs the whole. Its position is
``(q_1, ..., q_N, q_2)``.
"""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp

from .arrangement import Arrangement, exact, zero_form
from .coalgebra import configuration
from .network import map_box, plus_box
from .rand import euler_maruyama
from .rvect import euclidean
from .wiring import compose_seq, tensor_arrangements, trivial_parameter


def k_com(q, u):
    """The committed reader's cost: two wells at ``q = +-1``, tilted by the note ``u``."""
    return 0.25 * (q ** 2 - 1.0) ** 2 - u * q


def member(eta: float, T: float, q0: float = -1.0) -> Arrangement:
    """One member ``<0|0> -> <1|1>``: hears the note, presents its state; noise at ``T``."""
    return Arrangement(
        P=euler_maruyama(euclidean(1, eta), T, init=jnp.array([q0])),
        in_dim_M=0, out_dim_M=0, in_dim_N=1, out_dim_N=1,
        out_f=lambda q, m_out: q, in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -k_com(q[0], n_in[0])),
        label="member",
    )


def members(N: int, eta: float, T: float, q0: float = -1.0) -> Arrangement:
    """The ``N``-fold tensor power of ``member``, ``<0|0> -> <N|N>``, built as one vector of states.

    The power of the member's parameter is Euler--Maruyama on ``R^N`` with response ``eta I`` (the
    independent product of ``N`` one-dimensional Gaussian kicks, in order); its maps and 1-form act
    coordinatewise, so it is ``tensor_arrangements([member] * N)`` up to the nesting of the state."""
    return Arrangement(
        P=euler_maruyama(euclidean(N, eta), T, init=jnp.full(N, q0)),
        in_dim_M=0, out_dim_M=0, in_dim_N=N, out_dim_N=N,
        out_f=lambda q, m_out: q, in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -jnp.sum(k_com(q, n_in))),
        label=f"members({N})",
    )


def slow_reader(eta2: float, q0: float = -1.0) -> Arrangement:
    """``R2``, ``<0|1> -> I``: state ``q_2``, response ``eta2``, cost ``k_com(q_2, m)`` on the mean ``m``
    it hears; one-way, so nothing goes back along the mean's wire."""
    grad = exact(lambda q, m, n: -k_com(q[0], m[0]))

    def one_way(q, m, n):
        xi_q, _, xi_n = grad(q, m, n)
        return xi_q, jnp.zeros_like(m), xi_n

    return Arrangement(
        P=configuration(euclidean(1, eta2), init=jnp.array([q0])),
        in_dim_M=0, out_dim_M=1, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0), in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=one_way, label="R2",
    )


def _port() -> Arrangement:
    """The population's port ``<1|0> -> <1|0>``: the note passes inward."""
    return Arrangement(
        P=trivial_parameter(), in_dim_M=1, out_dim_M=0, in_dim_N=1, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0), in_f=lambda q, m_out, n_in: n_in,
        omega=zero_form, label="port",
    )


def _copy_note(N: int) -> Arrangement:
    """``<N|N> -> <1|N>``: the members' states pass on; the one note is copied to all ``N`` members."""
    return Arrangement(
        P=trivial_parameter(), in_dim_M=N, out_dim_M=N, in_dim_N=1, out_dim_N=N,
        out_f=lambda q, m_out: m_out, in_f=lambda q, m_out, n_in: jnp.tile(n_in, N),
        omega=zero_form, label="copy note",
    )


def mean_box(N: int) -> Arrangement:
    """The members' mean: a ``+`` box over ``N`` wires, then a box multiplying by ``1/N``."""
    return compose_seq(plus_box(N), map_box(lambda v: v / N, 1, 1, f"/{N}"))


def population(N: int, eta: float, T: float, eta2: float, members_arr: Arrangement = None,
               reader: Arrangement = None) -> Arrangement:
    """``<0|0> -> <1|0>``: the members, the note copied to them, their mean, and ``R2`` reading it.

    ``members_arr`` (default ``members(N, ...)``) may be any ``<0|0> -> <N|N>``; ``reader`` (default
    ``slow_reader(eta2)``) any ``<0|1> -> I``."""
    M = members(N, eta, T) if members_arr is None else members_arr
    R = slow_reader(eta2) if reader is None else reader
    heard = compose_seq(M, _copy_note(N))
    averaged = compose_seq(heard, tensor_arrangements([mean_box(N), _port()]))
    return replace(compose_seq(averaged, tensor_arrangements([R, _port()])), label=f"population({N})")
