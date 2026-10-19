"""A neural network as drawn: one box per parameter, the loss a box (book sec.neural_network; plan P43 (b) job 1).

The book pictures training as an arrangement of boxes (Ch 2, ``sec.neural_network``):

* a white box per parameter, holding it as its state and reading it out unchanged, with no internal
  inclination and the learning rate as its response (``parameter_box``);
* a white box for the datum and one for the label, each holding its place in a list of examples,
  reading out that example's value, ignoring the force that comes back, and stepping to the next
  example (``example_box``);
* gray boxes ``x``, ``+`` and ``tanh``, each a smooth map as a static arrangement (state ``R^0``,
  1-form ``0``; plan §5, "A smooth map is a static arrangement"), which passes the force at its
  output back to its inputs by the transpose of its differential (``map_box``);
* the wiring's dots, which copy a value to several boxes and add the forces that come back
  (``fan_box``, the image of a finite-set function);
* the loss, a 2-ary closed arrangement with cost ``k(y, l) = 1/2 |y - l|^2`` (``loss_box``).

``drawn_network`` composes them with ``compose_seq`` and ``tensor_arrangements`` (wiring.py) into
one closed arrangement, and ``Phi`` runs it. Nothing is computed outside it: the datum, the label
and the loss are boxes, so a step of ``Phi`` is a step of training.

It replaces, as the book's run, ``learning.parameterized_map`` of the whole network (one box) with
the loss's covector computed in code by ``learning.train`` (P43 (a), cb545383). That one-box run is
kept, and the test checks the drawn arrangement equal to it at its settings.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Sequence

import jax.numpy as jnp
from jax import Array

from .arrangement import Arrangement, exact, zero_form
from .coalgebra import CotCoalg, configuration
from .rvect import euclidean
from .wiring import compose_seq, tensor_arrangements, trivial_parameter


def _no_input(q, m_out, n_in):
    return jnp.zeros(0)


# ---------------------------------------------------------------------------
# The boxes.
# ---------------------------------------------------------------------------


def parameter_box(w0: float, eta: float, label: str = "w") -> Arrangement:
    """A parameter's white box ``<0|0> -> <0|1>``: state ``w``, read out unchanged, response ``eta``.

    Its 1-form is ``0`` (no internal inclination), so the force ``xi`` arriving at its output is
    its state's whole inclination and one step is ``w' = w + eta xi`` (book eqn.nn_step).
    """
    return Arrangement(
        P=configuration(euclidean(1, eta), init=jnp.array([w0], dtype=float)),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m: q,
        in_f=_no_input,
        omega=zero_form,
        label=label,
    )


def example_coalgebra(values: Array) -> CotCoalg:
    """The integrator of a list of examples: state its place ``s`` in the list, position ``values[s]``.

    ``u : S -> cot(Q)(S)`` sends ``s`` to ``(values[s mod n], xi |-> s + 1)``: whatever covector
    comes back, the box steps to the next example. ``Q = R^d``, ``S`` the integers.

    Over any duration it also steps to the next example (David, 2026-10-10: "can we just always show
    the next one? … They do 0.01, bam, bam, bam … we just show them 100 new data points per second"):
    the data source has no rate, and a run of short steps runs the examples fast. It has no limit as the
    duration shrinks, so no differential equation of its own; a learner fed by it does (full-batch gradient
    flow, ``tests/test_duration.py``). Durations include zero (Ch 3), and at zero it still steps to the next
    example, as the book's counter does, so it is an integrator that is not differentiable (Ch 4).
    """
    values = jnp.asarray(values, dtype=float)
    n, d = values.shape
    return CotCoalg(
        Q_dim=d,
        init=jnp.array(0),
        position=lambda s: values[s % n],
        step=lambda s, xi: s + 1,
        label="examples",
        timed_step=lambda s, xi, tau: s + 1,
    )


def example_box(values: Array, label: str = "examples") -> Arrangement:
    """The datum's or the label's white box ``<0|0> -> <0|d>``: reads out the current example.

    ``values`` has one row per example. The box ignores the force that comes back: its integrator
    (``example_coalgebra``) steps to the next row whatever covector it receives.
    """
    P = example_coalgebra(values)
    return Arrangement(
        P=P,
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=P.Q_dim,
        out_f=lambda q, m: q,
        in_f=_no_input,
        omega=zero_form,
        label=label,
    )


def map_box(g: Callable[[Array], Array], in_dim: int, out_dim: int, label: str) -> Arrangement:
    """A smooth map ``g : R^in_dim -> R^out_dim`` as a gray box, the static arrangement
    ``<0|in_dim> -> <0|out_dim>`` with forward map ``g``, state ``R^0`` and 1-form ``0``.

    ``Phi`` passes a force ``xi`` at its output back to its inputs as ``(T g)^T xi``.
    """
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0, out_dim_M=in_dim, in_dim_N=0, out_dim_N=out_dim,
        out_f=lambda q, m: g(m),
        in_f=_no_input,
        omega=zero_form,
        label=label,
    )


def times_box() -> Arrangement:
    """The gray ``x`` box: ``(a, b) |-> ab``."""
    return map_box(lambda m: m[0:1] * m[1:2], 2, 1, "x")


def plus_box(k: int = 2) -> Arrangement:
    """The gray ``+`` box with ``k`` inputs: ``(a_1, ..., a_k) |-> a_1 + ... + a_k``."""
    return map_box(lambda m: jnp.sum(m, keepdims=True), k, 1, "+")


def tanh_box() -> Arrangement:
    """The gray ``tanh`` box: ``z |-> tanh z``."""
    return map_box(jnp.tanh, 1, 1, "tanh")


def fan_box(in_dim: int, idx: Sequence[int], label: str = "fan") -> Arrangement:
    """The wiring ``<0|in_dim> -> <0|len(idx)>`` presenting the listed wires: copy, permute, drop.

    The image of a finite-set function; the forces that come back to a copied wire add.
    """
    idx = jnp.asarray(list(idx), dtype=int)
    return map_box(lambda m: m[idx], in_dim, len(idx), label)


def loss_box(dim: int = 1) -> Arrangement:
    """The loss, the 2-ary closed arrangement ``<0|dim> (x) <0|dim> -> <0|0>`` with cost
    ``k(y, l) = 1/2 |y - l|^2`` on the answer ``y`` and the label ``l``.

    Its 1-form is ``-dk``: it returns ``l - y`` to the answer and ``y - l`` to the label
    (book exc.nn_loss).
    """
    def U(q, m, n):
        y, lab = m[:dim], m[dim:]
        return -0.5 * jnp.sum((y - lab) ** 2)

    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0, out_dim_M=2 * dim, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m: jnp.zeros(0),
        in_f=_no_input,
        omega=exact(U),
        label="loss",
    )


def identity_box(dim: int) -> Arrangement:
    """Plain wires passing ``dim`` values through a layer of the drawing."""
    return fan_box(dim, range(dim), "id")


# ---------------------------------------------------------------------------
# The drawing composed.
# ---------------------------------------------------------------------------


def _layer(boxes: Sequence[Arrangement], passing: int) -> Arrangement:
    """Boxes side by side, with ``passing`` wires carried past them unchanged."""
    return tensor_arrangements(list(boxes) + ([identity_box(passing)] if passing else []))


def drawn_network(
    h: int, m: int, n: int, params0: Array, eta: float, data: Array, labels: Array,
    *, hidden: bool = True, out_bias: bool = True,
) -> Arrangement:
    """The closed arrangement that trains ``y = W2 tanh(W1 x + b1) + b2`` on a list of examples.

    Leaves, side by side: the datum's box (``m`` wires), one parameter box per entry of ``W1``
    (``h x m``, row-major), ``b1`` (``h``), ``W2`` (``n x h``), ``b2`` (``n``, if ``out_bias``),
    and the label's box (``n`` wires). Then, layer by layer: a fan copies each datum entry to the
    ``h`` boxes that read it; an ``x`` box per entry of ``W1``; a ``+`` box per hidden unit,
    adding its bias and its ``m`` products; a ``tanh`` box per hidden unit; a fan copying each
    hidden value to the ``n`` boxes that read it; an ``x`` box per entry of ``W2``; a ``+`` box per
    output (its bias and its ``h`` products); the loss. Every box is composed by ``compose_seq``
    and ``tensor_arrangements``; nothing is written as a composite by hand.

    With ``hidden=False`` the network is linear, ``y = W x + b`` (``h`` is ignored, ``W`` is
    ``n x m``): leaves datum, ``W``, ``b``, label; an ``x`` per entry, a ``+`` per output, the loss.
    ``params0`` is flat in the order of ``learning``'s one-box nets (the test's ``_two_layer``,
    ``_linear``), so the parameters sit at ``position[m : m + len(params0)]`` of the state.

    The book's network (``sec.neural_network``) is ``h = m = n = 1``, ``out_bias=False``.
    """
    params0 = jnp.asarray(params0, dtype=float)
    data = jnp.asarray(data, dtype=float).reshape(-1, m)
    labels = jnp.asarray(labels, dtype=float).reshape(-1, n)
    leaves = [example_box(data, "datum")]
    leaves += [parameter_box(w, eta, f"w{i}") for i, w in enumerate(params0)]
    leaves += [example_box(labels, "label")]
    net = tensor_arrangements(leaves)
    P = len(params0)

    def affine(net, k_in, k_out, rest, W_at, b_at):
        """Wires ``[v (k_in), W (k_out x k_in), b (k_out or none), rest]`` -> ``[W v + b, rest]``."""
        width = k_in + k_out * k_in + (k_out if b_at is not None else 0) + rest
        pairs = [i for j in range(k_out) for i_ in range(k_in)
                 for i in (W_at + j * k_in + i_, i_)]
        tail = list(range(width - rest, width))
        biases = list(range(b_at, b_at + k_out)) if b_at is not None else []
        net = compose_seq(net, fan_box(width, pairs + biases + tail, "copy"))
        net = compose_seq(net, _layer([times_box()] * (k_out * k_in), len(biases) + rest))
        # wires now [products (k_out * k_in), biases, rest]; group each unit's products with its bias
        groups = []
        for j in range(k_out):
            if b_at is not None:
                groups.append(k_out * k_in + j)
            groups += [j * k_in + i_ for i_ in range(k_in)]
        width2 = k_out * k_in + len(biases) + rest
        net = compose_seq(net, fan_box(width2, groups + list(range(width2 - rest, width2)), "group"))
        k_sum = k_in + (1 if b_at is not None else 0)
        return compose_seq(net, _layer([plus_box(k_sum)] * k_out, rest))

    if hidden:
        # [x (m), W1 (hm), b1 (h), W2 (nh), b2 (n?), label (n)]
        rest = n * h + (n if out_bias else 0) + n
        net = affine(net, m, h, rest, W_at=m, b_at=m + h * m)
        net = compose_seq(net, _layer([tanh_box()] * h, rest))
        # [a (h), W2 (nh), b2 (n?), label (n)]
        net = affine(net, h, n, n, W_at=h, b_at=(h + n * h) if out_bias else None)
    else:
        # [x (m), W (nm), b (n?), label (n)]
        net = affine(net, m, n, n, W_at=m, b_at=(m + n * m) if out_bias else None)
    # [y (n), label (n)]
    net = compose_seq(net, loss_box(n))
    assert net.P.Q_dim == m + P + n
    return replace(net, label=f"drawn_network(h={h}, m={m}, n={n})")


def parameters(arr: Arrangement, state, m: int, size: int) -> Array:
    """The parameters held in the drawn network's state (its parameter boxes' positions)."""
    return arr.P.position(state)[m:m + size]
