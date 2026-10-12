"""P34(k): the original runner kit from attention and three additions.

The additions are a unit-message source, scalar multiplication, and the
non-exact forcing d dr. Everything else is composed using AttentionArrangement
and explicit scalar/message adapters. The adapters only pad/project signal
coordinates and insert a fixed weight; they never average or compute a guest
step. Constants and memory readouts are configuration-dependent transfers.

The exact interpreter uses zero/identity and signed linear signal maps in the
FULL attention operad. It is not claimed to arise from spatial attraction.
``linear_attention`` also accepts a smooth positive gain, including a Bregman
gain; its attenuation and derivatives really reach values and forces. An
arbitrary attenuated transfer does not preserve the interpreter's exact flags.
See STATUS.md, "Attention and three additions", and the book's
notes/runs/p34-attention-1eda4400/DESIGN.md, §§3–5.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .arrangement import Arrangement, zero_form
from .attention import attention_arrangement, transfer, zero_transfer
from .coalgebra import CotCoalg, configuration
from .rvect import ReactiveVectorSpace
from .wiring import compose_seq, tensor_arrangements, trivial_parameter


def _static(n, k, forward, label, *, in_M=0, in_N=0, backward=None,
            omega=zero_form):
    return Arrangement(
        P=trivial_parameter(), in_dim_M=in_M, out_dim_M=n,
        in_dim_N=in_N, out_dim_N=k, out_f=forward,
        in_f=backward or (lambda q, m, e: jnp.zeros(in_M)),
        omega=omega, label=label,
    )


def _encode(n, width):
    # One tagged vector message, not a sum of unlabelled scalar messages.
    return _static(n, width + 1,
                   lambda q, m: jnp.concatenate((m, jnp.zeros(width-n), jnp.ones(1))),
                   f"encode[{n}->{width}+weight]", in_N=width+1)


def _decode(k, width):
    return _static(width+1, k, lambda q, m: m[:k],
                   f"decode[{width}->{k}]", in_M=width+1)


def _parent(width, matrix, parameter, gain, label):
    def transfers(q):
        g = gain(q)
        z = zero_transfer(width)
        return [[z, transfer(g, g * matrix(q))], [z, z]]

    return attention_arrangement(
        1, width, 0, lambda q: jnp.zeros((1, 0)), transfers,
        parameter=parameter, label=label,
    ).arr()


def linear_attention(matrix, *, parameter=None, gain=None, label="attention-linear"):
    """A scalar-vector linear map through a genuine unary attention operation.

    ``matrix`` has shape (outputs, inputs); one message retains each scalar in
    its own signal coordinate. ``gain(q)`` must be smooth and strictly positive
    throughout the parameter's configuration domain (checked only initially).
    The actual forward map is gain(q)*matrix@input; all its derivatives are
    pulled back by ordinary arrangement composition. No gain is compensated.
    """
    matrix = jnp.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("matrix must have shape (outputs, inputs)")
    k, n = matrix.shape
    width = max(k, n, 1)
    padded = jnp.pad(matrix, ((0, width-k), (0, width-n)))
    parameter = trivial_parameter() if parameter is None else parameter
    gain = (lambda q: jnp.array(1.0)) if gain is None else gain
    parent = _parent(width, lambda q: padded, parameter, gain, label)
    return compose_seq(compose_seq(_encode(n, width), parent), _decode(k, width))


def bregman_gain(sender, receiver, potential, temperature=1.0):
    """exp(-D_potential(sender, receiver)/temperature), with ordered arguments.

    The caller supplies a differentiable convex scalar potential on a convex
    domain containing both points. No metric or symmetry is assumed. Invalid
    domains and nonconvex potentials are not made valid by this constructor.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    sender = jnp.asarray(sender, dtype=float)
    receiver = jnp.asarray(receiver, dtype=float)
    divergence = (potential(sender) - potential(receiver)
                  - jnp.vdot(jax.grad(potential)(receiver), sender-receiver))
    return jnp.exp(-divergence / temperature)


def unit_source():
    """Addition 1: a state-free source of the scalar message (1,1)."""
    return _static(0, 2, lambda q, m: jnp.ones(2), "unit-source", in_N=2)


def _emit(parameter, coefficient, label):
    parent = _parent(1, lambda q: jnp.reshape(coefficient(q), (1, 1)),
                     parameter, lambda q: jnp.array(1.0), label)
    return compose_seq(compose_seq(unit_source(), parent), _decode(1, 1))


def const(value):
    return _emit(trivial_parameter(), lambda q: jnp.array(float(value)), "attention-const")


def register(initial):
    matter = configuration(ReactiveVectorSpace(1, sharp_fn=lambda q: jnp.eye(1)),
                           init=jnp.array([float(initial)]))
    return _emit(matter, lambda q: q[0], "attention-register")


def drum(initial):
    initial = jnp.asarray(initial, dtype=float)
    if initial.ndim != 1 or initial.size == 0:
        raise ValueError("a drum needs a nonempty one-dimensional cell array")
    matter = CotCoalg(Q_dim=initial.size, init=initial, position=lambda s: s,
                     step=lambda s, xi: jnp.roll(s+xi, -1),
                     timed_step=lambda s, xi, t: jnp.roll(s+xi, -1), label="drum")
    return _emit(matter, lambda q: q[0], "attention-drum")


def add():
    return linear_attention([[1.0, 1.0]], label="attention-add")


def fan(n, indices):
    if n < 0 or any(i < 0 or i >= n for i in indices):
        raise ValueError("fan indices must name an input")
    matrix = jnp.zeros((len(indices), n))
    for row, col in enumerate(indices):
        matrix = matrix.at[row, col].set(1.0)
    return linear_attention(matrix, label=f"attention-fan[{n}:{tuple(indices)}]")


def mul():
    """Addition 2: static multiplication, including its ordinary pullback."""
    return _static(2, 1, lambda q, m: m[:1]*m[1:], "multiply")


def push():
    """Addition 3: d dr; force on the donor d is zero."""
    return _static(2, 0, lambda q, m: jnp.zeros(0), "one-way-push",
                   omega=lambda q, m, e: (jnp.zeros_like(q),
                                           jnp.array([0.0, m[0]]), jnp.zeros_like(e)))


def cost():
    # The input v is routed beside the constant -1, then push(-1,v) gives -dv.
    return compose_seq(tensor_arrangements([const(-1), fan(1, [0])]), push())


def kit():
    return dict(const=const, register=register, drum=drum, add=add,
                mul=mul, cost=cost, push=push, fan=fan)


def denote(term):
    """Interpret the original eight-part syntax with the constructed kit.

    Compilation is external, as in P34(d). Unknown extensions (e.g. exp,
    reciprocal, inbox) fail instead of silently using the old kit.
    """
    constructors = kit()

    def go(t):
        head, args = t[0], t[1:]
        if head == "seq":
            return compose_seq(go(args[0]), go(args[1]))
        if head == "tensor":
            return tensor_arrangements([go(part) for part in args])
        if head not in constructors:
            raise ValueError(f"not in the original attention-runner kit: {head}")
        return constructors[head](*args)

    return go(term)
