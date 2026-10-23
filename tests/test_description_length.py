"""Description length of Ch 2's weight-decay network, as an arrangement (book plan P71 (a)).

The book's ``exc.nn_decay`` gives each parameter the cost ``gamma/2 w^2`` of its own state, and its
footnote says both that cost and the loss are description lengths: minus the log-density of the
parameter under a Gaussian of variance ``1/gamma``, and of the label under a Gaussian centered at
the answer. P71 asks whether a *length of the arrangement*, parameters plus wiring, is a cost
compatible with nesting. Two halves, kept apart:

**The parameter half runs in ``arr``.** The network of ``sec.neural_network`` is built from the
library's boxes (``decay_parameter_box`` below is ``network.parameter_box`` with the forcing
``-d(gamma/2 w^2)``; the maps, fans, example and loss boxes are ``network``'s), composed by
``compose_seq`` and ``tensor_arrangements``, and run by ``Phi``. Checked:

* one ``Phi`` step is the exercise's (b): ``(1.1, -0.7, 1.8)`` from ``(1, -1, 2)`` at ``gamma = 1``,
  ``eta = 1/10``; with ``gamma = 0`` the book's ``(1.2, -0.8, 2)``;
* the composite's forcing on its configuration is ``-d`` of one function, the loss's cost plus
  the three parameters' lengths, so the differential of the length is the regularization force;
* nested as a two-input substitution (the first layer and the second weight filling two holes of
  an outer arrangement holding the label and the loss), the composite's forcing is ``-d`` of the
  sum of the pieces' cost functions at the points the wiring delivers (``prop.exact_lenses`` on a
  branching composite), equals the flat network's, and steps the same; an identity inserted
  between layers changes nothing (the unit law).

**The wiring half is a count, not a run.** A length ``c_box`` per internal box and ``c_wire`` per
wire is constant in the configuration, so its differential is ``0``: ``Phi`` never sees it, and no
part is pushed by it. Counted on the same nest, with the book's convention that a wire is a source
port (``def.wiring_diagram``: ``out(X) + in(Y)`` sources):

* the count is fixed by the interfaces alone, not by the pattern of the wiring;
* it is not additive under substitution: the composite's count is the pieces' sum minus the
  count of each glued interface (its hole and its ports), and the identity's count is not ``0``;
* the one way to make it additive, counting each interface once, leaves ``c_wire`` times
  (external inputs minus internal inputs), a boundary term, ``0`` on the whole training system.

Reasonable: the parameter half is ``prop.exact_lenses`` and holds at any scale a stored quantity costs something
to keep. Weird: a uniform count of boxes and wires says nothing about the wiring, and the length of the arrangement a
nest denotes is not the sum of the nest's pieces. Results and the judgment: the book's
``notes/runs/p71a-c3bdb524/RESULTS.md``.
"""

from dataclasses import dataclass
from typing import List

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact
from arrangements.coalgebra import configuration
from arrangements.functors import Phi
from arrangements.interpretation import trivial_omega
from arrangements.network import (
    example_box, fan_box, identity_box, loss_box, parameter_box, plus_box, tanh_box, times_box,
)
from arrangements.rvect import euclidean
from arrangements.wiring import compose_seq, tensor_arrangements

_IN = (jnp.zeros(0), trivial_omega(0))
_NONE = jnp.zeros(0)
GAMMA, ETA, X, LABEL = 1.0, 0.1, 1.0, 1.0
W0 = (1.0, -1.0, 2.0)


def decay_parameter_box(w0: float, eta: float, gamma: float, label: str = "w") -> Arrangement:
    """``parameter_box`` with the cost ``gamma/2 w^2`` of its own state (``exc.nn_decay``).

    The cost is the parameter's description length under a Gaussian of variance ``1/gamma``, up to
    the constant ``log(2 pi / gamma) / 2`` (nats), which has differential ``0``. The forcing is
    ``-dk = -gamma w``, the exercise's internal inclination.
    """
    def U(q, m, n):
        return -0.5 * gamma * q[0] ** 2

    return Arrangement(
        P=configuration(euclidean(1, eta), init=jnp.array([w0], dtype=float)),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m: q,
        in_f=lambda q, m, n: jnp.zeros(0),
        omega=exact(U),
        label=label,
    )


# ---------------------------------------------------------------------------
# Building, with a tally of internal boxes and wires carried beside the arrangement.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Built:
    """An arrangement with its count: ``leaves`` internal boxes (the leaves of the library's
    tree), and ``sources`` the wires inside it that start at a leaf's output port. The wires of
    ``a : M -> N`` as the book draws them are the sources ``out(M) + leaves' outputs + in(N)``."""
    arr: Arrangement
    leaves: int
    sources: int

    @property
    def wires(self) -> int:
        return self.arr.out_dim_M + self.sources + self.arr.in_dim_N

    def then(self, g: "Built") -> "Built":
        return Built(compose_seq(self.arr, g.arr), self.leaves + g.leaves, self.sources + g.sources)


def leaf(a: Arrangement) -> Built:
    return Built(a, 1, a.out_dim_N)


def wiring(a: Arrangement) -> Built:
    """A fan or an identity: the wiring's own boxes, which are not boxes of the drawing."""
    return Built(a, 0, 0)


def side_by_side(parts: List[Built]) -> Built:
    return Built(tensor_arrangements([p.arr for p in parts]),
                 sum(p.leaves for p in parts), sum(p.sources for p in parts))


def layer(boxes: List[Built], passing: int) -> Built:
    return side_by_side(boxes + ([wiring(identity_box(passing))] if passing else []))


def flat_network(gamma: float, eta: float = ETA, w0=W0, with_identity: bool = False) -> Built:
    """The training arrangement of ``sec.neural_network`` with weight decay, from leaves.

    Wires after each stage are written on the right; the configuration is ``[x, w1, w2, w3, l]``.
    """
    box = (lambda w, name: leaf(decay_parameter_box(w, eta, gamma, name))) if gamma else \
          (lambda w, name: leaf(parameter_box(w, eta, name)))
    net = side_by_side([leaf(example_box([[X]], "datum")), box(w0[0], "w1"), box(w0[1], "w2"),
                        box(w0[2], "w3"), leaf(example_box([[LABEL]], "label"))])  # [x w1 w2 w3 l]
    net = net.then(wiring(fan_box(5, [1, 0, 2, 3, 4])))                              # [w1 x w2 w3 l]
    net = net.then(layer([leaf(times_box())], 3))                                    # [w1x w2 w3 l]
    net = net.then(wiring(fan_box(4, [1, 0, 2, 3])))                                 # [w2 w1x w3 l]
    net = net.then(layer([leaf(plus_box(2))], 2))                                    # [z w3 l]
    net = net.then(layer([leaf(tanh_box())], 2))                                     # [a w3 l]
    if with_identity:
        net = net.then(wiring(identity_box(3)))
    net = net.then(wiring(fan_box(3, [1, 0, 2])))                                    # [w3 a l]
    net = net.then(layer([leaf(times_box())], 1))                                    # [y l]
    return net.then(leaf(loss_box(1)))                                               # []


def first_layer(gamma: float, eta: float = ETA, w0=W0) -> Built:
    """The system ``<> -> <0|1>`` reading out ``tanh(w1 x + w2)``: datum, ``w1``, ``w2``, ``x``, ``+``, ``tanh``."""
    net = side_by_side([leaf(example_box([[X]], "datum")),
                        leaf(decay_parameter_box(w0[0], eta, gamma, "w1")),
                        leaf(decay_parameter_box(w0[1], eta, gamma, "w2"))])        # [x w1 w2]
    net = net.then(wiring(fan_box(3, [1, 0, 2])))                                    # [w1 x w2]
    net = net.then(layer([leaf(times_box())], 1))                                    # [w1x w2]
    net = net.then(wiring(fan_box(2, [1, 0])))                                       # [w2 w1x]
    net = net.then(layer([leaf(plus_box(2))], 0))                                    # [z]
    return net.then(layer([leaf(tanh_box())], 0))                                    # [a]


def outer(eta: float = ETA) -> Built:
    """The outer arrangement ``<0|1> (x) <0|1> -> <>`` with two holes, for the first layer's value
    ``a`` and the second weight ``w3``, holding the label, the second ``x`` and the loss."""
    net = side_by_side([wiring(identity_box(2)), leaf(example_box([[LABEL]], "label"))])  # [a w3 l]
    net = net.then(wiring(fan_box(3, [1, 0, 2])))                                    # [w3 a l]
    net = net.then(layer([leaf(times_box())], 1))                                    # [y l]
    return net.then(leaf(loss_box(1)))                                               # []


def nest(gamma: float, eta: float = ETA, w0=W0):
    """The two-input substitution: the first layer and ``w3`` fill the outer's holes."""
    pieces = [first_layer(gamma, eta, w0), leaf(decay_parameter_box(w0[2], eta, gamma, "w3"))]
    return pieces, outer(eta), side_by_side(pieces).then(outer(eta))


def total_cost(gamma: float):
    """The one function the network's forcing should be ``-d`` of: the loss's cost at the answer
    plus the three parameters' lengths, on the configuration ``[x, w1, w2, w3, l]``."""
    def k(q):
        x, w1, w2, w3, lab = q
        return 0.5 * (w3 * jnp.tanh(w1 * x + w2) - lab) ** 2 + 0.5 * gamma * (w1**2 + w2**2 + w3**2)
    return k


def _step(A: Arrangement):
    _, _, s = Phi(A).run_one(_IN, lambda _o: (_NONE, _NONE))
    return s


def _forcing_on_configuration(A: Arrangement, q):
    xi_q, _, _ = A.omega(q, _NONE, _NONE)
    return xi_q


# ---------------------------------------------------------------------------
# The parameter half, in arr.
# ---------------------------------------------------------------------------


def test_step_is_the_exercise():
    """``exc.nn_decay`` (b): ``(1, -1, 2)`` -> ``(1.1, -0.7, 1.8)``; without decay the book's ``(1.2, -0.8, 2)``."""
    A = flat_network(GAMMA).arr
    s = _step(A)
    np.testing.assert_allclose(A.P.position(s)[1:4], [1.1, -0.7, 1.8], atol=1e-15)
    B = flat_network(0.0).arr
    np.testing.assert_allclose(B.P.position(_step(B))[1:4], [1.2, -0.8, 2.0], atol=1e-15)


def test_forcing_is_minus_d_of_the_total_length():
    """The composite's forcing on ``[x, w1, w2, w3, l]`` is ``-d`` of loss plus parameter lengths; on the
    parameters it is the backpropagated force plus ``-gamma w``: ``(2 - 1, 2 + 1, 0 - 2)``."""
    A = flat_network(GAMMA).arr
    q = A.P.position(A.P.init)
    xi = _forcing_on_configuration(A, q)
    np.testing.assert_allclose(xi, -jax.grad(total_cost(GAMMA))(q), atol=1e-14)
    np.testing.assert_allclose(xi[1:4], [1.0, 3.0, -2.0], atol=1e-14)
    # the same at a point where nothing is zero
    q2 = jnp.array([0.7, 1.3, -0.4, 0.9, -0.2])
    np.testing.assert_allclose(_forcing_on_configuration(A, q2), -jax.grad(total_cost(GAMMA))(q2), atol=1e-13)


def test_nest_sums_the_pieces_costs_and_runs_like_the_flat_network():
    """``prop.exact_lenses`` on a two-input substitution: the nest's forcing is ``-d`` of the pieces' cost
    functions summed at the delivered points, equals the flat network's, and the step agrees."""
    pieces, g, N = nest(GAMMA)
    A = flat_network(GAMMA).arr
    assert N.arr.P.Q_dim == A.P.Q_dim == 5

    def pieces_summed(q):
        x, w1, w2, w3, lab = q
        k_layer1 = 0.5 * GAMMA * (w1**2 + w2**2)                       # the first layer's own cost
        k_w3 = 0.5 * GAMMA * w3**2                                      # the second weight's
        a = jnp.tanh(w1 * x + w2)                                       # what the layer delivers
        k_outer = 0.5 * (w3 * a - lab) ** 2                             # the outer's, at (a, w3, l)
        return k_layer1 + k_w3 + k_outer

    for q in (N.arr.P.position(N.arr.P.init), jnp.array([0.7, 1.3, -0.4, 0.9, -0.2])):
        xi = _forcing_on_configuration(N.arr, q)
        np.testing.assert_allclose(xi, -jax.grad(pieces_summed)(q), atol=1e-13)
        np.testing.assert_allclose(xi, _forcing_on_configuration(A, q), atol=1e-13)
    np.testing.assert_allclose(N.arr.P.position(_step(N.arr))[1:4], [1.1, -0.7, 1.8], atol=1e-15)
    # the pieces' own forcings, each at its own configuration: -gamma w on the parameters it holds
    L = pieces[0].arr
    np.testing.assert_allclose(_forcing_on_configuration(L, L.P.position(L.P.init)),
                               [0.0, -GAMMA * W0[0], -GAMMA * W0[1]], atol=1e-15)   # datum, w1, w2
    print("label of the nest:", N.arr.label)


def test_identity_between_layers_changes_nothing():
    """The unit law for the cost: an identity box inserted between layers leaves forcing and step alone."""
    A, B = flat_network(GAMMA).arr, flat_network(GAMMA, with_identity=True).arr
    q = jnp.array([0.7, 1.3, -0.4, 0.9, -0.2])
    np.testing.assert_allclose(_forcing_on_configuration(A, q), _forcing_on_configuration(B, q), atol=1e-14)
    np.testing.assert_allclose(A.P.position(_step(A)), B.P.position(_step(B)), atol=1e-15)


# ---------------------------------------------------------------------------
# The wiring half: counts, beside the arrangements they describe. Nothing here is run by Phi.
# ---------------------------------------------------------------------------

C_BOX, C_WIRE = 3.0, 1.0     # any fixed counts; the identities below hold for every choice


def length_leaf_picture(b: Built) -> float:
    """Boxes are the leaves; a hole is not a box (the library's tree)."""
    return C_BOX * b.leaves + C_WIRE * b.wires


def length_hole_picture(b: Built, holes: int) -> float:
    """The book's picture: an outer arrangement's holes are its internal boxes too."""
    return C_BOX * (b.leaves + holes) + C_WIRE * b.wires


def interface_length(in_ports: int, out_ports: int) -> float:
    return C_BOX + C_WIRE * (in_ports + out_ports)


def test_wiring_count_is_fixed_by_the_interfaces_and_not_additive():
    """The flat network: 10 internal boxes, 9 wires. The nest's pieces sum to 12 boxes and 11 wires; the
    glued interfaces, two holes with one port each, are the difference. The identity on ``<0|1>`` counts
    one box and one wire, not ``0``. The additive remainder, ``c_wire (in Y - sum in X_i)``, is ``0`` here."""
    flat = flat_network(GAMMA)
    assert (flat.leaves, flat.wires) == (10, 9)
    pieces, g, N = nest(GAMMA)
    assert (pieces[0].leaves, pieces[0].wires) == (6, 6)
    assert (pieces[1].leaves, pieces[1].wires) == (1, 1)
    assert (g.leaves, g.wires) == (3, 4)          # label, x, loss; sources: two holes' outputs, label, x
    assert (N.leaves, N.wires) == (10, 9)
    # the gluing formula, in the book's picture (holes are boxes of the outer arrangement)
    L_pieces = sum(length_hole_picture(p, 0) for p in pieces) + length_hole_picture(g, holes=2)
    L_glued = 2 * interface_length(0, 1)
    assert L_pieces - L_glued == length_hole_picture(N, 0) == C_BOX * 10 + C_WIRE * 9
    assert L_pieces != length_hole_picture(N, 0)  # not additive
    # the same formula in the library's picture: boxes add, wires lose the glued ports
    assert sum(length_leaf_picture(p) for p in pieces) + length_leaf_picture(g) - 2 * C_WIRE == length_leaf_picture(N)
    # the unit: an identity on <0|1> is one hole and one wire, not 0
    idb = wiring(identity_box(1))
    assert length_hole_picture(idb, holes=1) == interface_length(0, 1) != 0
    # degenerate cases: the identity on the closed interface, and a box with no ports
    idc = wiring(identity_box(0))
    assert (length_leaf_picture(idc), length_hole_picture(idc, holes=1)) == (0, C_BOX)
    noports = leaf(Arrangement(P=configuration(euclidean(1, 1.0)), in_dim_M=0, out_dim_M=0, in_dim_N=0,
                               out_dim_N=0, out_f=lambda q, m: jnp.zeros(0), in_f=lambda q, m, n: jnp.zeros(0),
                               omega=lambda q, m, n: (jnp.zeros(1), jnp.zeros(0), jnp.zeros(0)), label="mute"))
    assert (noports.leaves, noports.wires) == (1, 0)
    # the additive remainder after each interface is counted once: c_wire (in Y - sum_i in X_i)
    def remainder(b: Built, holes_in: List[int]) -> float:
        return C_WIRE * (b.arr.in_dim_N - sum(holes_in))
    assert remainder(flat, []) == remainder(N, [0, 0]) == remainder(g, [0, 0]) == remainder(idb, [0]) == 0.0


def test_wiring_count_has_zero_differential():
    """Constant in the configuration: the arrangement's forcing does not change when the count does.
    Two networks that differ only in an identity box between layers have the same forcing everywhere
    (``test_identity_between_layers_changes_nothing``), and the same count once the identity is glued in."""
    A, B = flat_network(GAMMA), flat_network(GAMMA, with_identity=True)
    assert A.wires == B.wires           # an identity wiring adds no source, in the library's picture
    assert length_hole_picture(B, holes=0) == length_hole_picture(A, holes=0)
    # and were the identity kept as a hole of a nest, its interface's length would be added and then
    # subtracted by the gluing formula; either way Phi cannot tell the two apart
    q = jnp.array([0.7, 1.3, -0.4, 0.9, -0.2])
    np.testing.assert_allclose(_forcing_on_configuration(A.arr, q), _forcing_on_configuration(B.arr, q), atol=1e-14)
