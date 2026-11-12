"""Ch 5 ``ex.finite_projection_lift``: the lift ``pi^! D = D x Q`` for identity steps on finite spaces.

What runs in the dynamics functor: a source ``b`` and a reader ``q`` that hears it one-way, both configuration parts
with inclination field ``0``, wired with ``compose_seq``; and the source alone. Their steps ``F`` and ``G``, at durations
1/4 and 1, are read at the four points of ``B x Q = {0, 1} x {0, 1}`` inside ``R^2`` (the library stays on ``R^n``;
the finite discrete spaces are these points). What is an ordinary calculation beside the runs: the attractors, the
image map ``pi_*``, and ``pi^! D = lim_F(pi^{-1} V)`` with ``V = D``, on subsets of those four points.

Predictions, written before the first run (Claude Opus 5.5, 00f2d232, 2026-11-12): ``F`` and ``G`` are the identity at
both durations; ``pi o F = G o pi`` on the four points (a skew product); every subset is an attractor; ``pi^! D = D x Q``
for each of the four ``D``; ``pi_* A <= D`` exactly when ``A <= pi^! D`` for all sixteen ``A``; ``pi_* pi^! = id``.
"""

from itertools import chain, combinations

import jax
import jax.numpy as jnp

from arrangements.arrangement import Arrangement, zero_form
from arrangements.coalgebra import configuration
from arrangements.functors import closed_step
from arrangements.rvect import diagonal
from arrangements.wiring import compose_seq, tensor_arrangements, trivial_parameter

Z = jnp.zeros(0)
B, Q = (0.0, 1.0), (0.0, 1.0)
POINTS = [(b, q) for b in B for q in Q]


def source(b0):
    return Arrangement(P=configuration(diagonal(jnp.array([1.0])), init=jnp.array([b0])), in_dim_M=0, out_dim_M=0,
                       in_dim_N=0, out_dim_N=1, out_f=lambda q, m: q, in_f=lambda q, m, n: Z, omega=zero_form,
                       label="source")


def reader(q0):
    """Hears the source and returns it no load: one-way."""
    return Arrangement(P=configuration(diagonal(jnp.array([1.0])), init=jnp.array([q0])), in_dim_M=0, out_dim_M=0,
                       in_dim_N=1, out_dim_N=0, out_f=lambda q, m: Z, in_f=lambda q, m, n: Z, omega=zero_form,
                       label="reader")


def whole(b0, q0):
    wire = Arrangement(P=trivial_parameter(), in_dim_M=1, out_dim_M=1, in_dim_N=0, out_dim_N=0, out_f=lambda q, m: Z,
                       in_f=lambda q, m, n: m, omega=zero_form, label="wire")
    return compose_seq(tensor_arrangements([source(b0), reader(q0)]), wire)


def drop():
    return Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=1, in_dim_N=0, out_dim_N=0, out_f=lambda q, m: Z,
                       in_f=lambda q, m, n: Z, omega=zero_form, label="drop")


def flat(s):
    return [float(v) for x in jax.tree_util.tree_leaves(s) for v in jnp.ravel(x)]


def subsets(xs):
    xs = list(xs)
    return [frozenset(c) for c in chain.from_iterable(combinations(xs, k) for k in range(len(xs) + 1))]


def test_finite_projection_lift():
    for tau in (0.25, 1.0):
        F = {}
        for b, q in POINTS:
            arr = whole(b, q)
            assert "source" in arr.label and "reader" in arr.label
            F[(b, q)] = tuple(flat(closed_step(arr, duration=tau)(arr.P.init)))      # leaves: source, reader
        G = {}
        for b in B:
            arr = compose_seq(source(b), drop())
            (G[b],) = flat(closed_step(arr, duration=tau)(arr.P.init))
        assert all(F[x] == x for x in POINTS) and all(G[b] == b for b in B)
        assert all(F[x][0] == G[x[0]] for x in POINTS)                      # pi o F = G o pi

    def limset(U):                       # on a finite discrete space every set is closed; F is the identity here
        return frozenset(F[x] for x in U)

    attr_X = [A for A in subsets(POINTS) if limset(A) == A]
    attr_B = [D for D in subsets(B) if frozenset(G[b] for b in D) == D]
    assert len(attr_X) == 16 and len(attr_B) == 4
    push = lambda A: frozenset(x[0] for x in A)
    lift = lambda D: limset([x for x in POINTS if x[0] in D])                # pi^! D = lim_F(pi^{-1} V), V = D
    for D in attr_B:
        assert lift(D) == frozenset((b, q) for b in D for q in Q)            # eqn.finite_fibre_lift
        assert push(lift(D)) == D
        for A in attr_X:
            assert (push(A) <= D) == (A <= lift(D))
