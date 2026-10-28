"""A shared load on breakable bonds (book Ch 4, ``ex.shared_load`` and ``exc.load_tears``, cut from the book 2026-10-26, archived in its `notes/archive/ch4-shared-load-cut-2026-10-26.tex`;
plan notes ``notes/runs/ch4-bond-970ef227/PREDICTIONS.md``, written before this file).

**Every operation here runs in arr.** The holding is an attention arrangement of
``attention.py``; the holders and the load are plain ``Arrangement``s on the interface
``B`` (components ``I -> B``, as in tests/test_rendezvous.py); the composite is
``compose_seq(tensor_arrangements(parts), phi.arr())`` and its forcing is that
composition's pullback-and-sum. The closed-form cost below is only what the composite is
checked against; nothing builds the composite from it.

``phi : V, V, V -> V`` (holders 1 and 2, Alice and Bob; the load 3) has configuration
``y``, how far the load has dropped below the holders' hands; placements ``(0, 0, y)``;
each holder hears the load at ``exp(-y^2/2)`` (``T_{13}``, ``T_{23}``; ``T_{23} = 0``
when Bob is not holding), all other transfers zero; cost ``-W y``, gravity on a load
``W > 0``. Each holder's cost is ``-w``, ``w`` the weight of the message it receives: it
wants to hear the load loudly. The load sends ``msg(0)`` with no cost. Composite cost
``-W y - n exp(-y^2/2)`` for ``n`` holders, so a holder's pull is ``y exp(-y^2/2)``: about
``y`` near ``0`` (a spring of stiffness 1), largest, ``exp(-1/2)``, at ``y = 1``.

``H = W = P = R`` throughout, the integrator the configuration coalgebra (gradient
descent) at the learning rate ``1/2``.

ILLUSTRATES (STATUS.md §3.2): the runs are discrete gradient descent at a fixed rate; the
rest points are checked against the closed form's critical points.
(An earlier version, 89a01e25, modeled the load as a box wanting the lifts it hears to sum
to ``W``; nothing in it could break, and the book's example was rebuilt on bonds.)
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact
from arrangements.attention import (
    attention_arrangement, input_dim, make_input, message, msg_dim, transfer, zero_transfer,
)
from arrangements.coalgebra import configuration
from arrangements.rvect import euclidean, trivial
from arrangements.wiring import compose_seq, tensor_arrangements

from tests.test_rendezvous import run_closed

DH, DP = 1, 1
MDIM, EDIM = msg_dim(DH), input_dim(DH, DP)
EXT = make_input(message(0.0, 0.0), 0.0)
ETA = 0.5


def strength(y):
    return jnp.exp(-0.5 * y ** 2)


def holder(label):
    """A holder: no configuration, sends ``msg(0)``, cost ``-w`` on the weight ``w`` of the
    message it receives (``exact`` takes minus the cost)."""
    return Arrangement(
        P=configuration(trivial()),
        in_dim_M=0, out_dim_M=0, in_dim_N=EDIM, out_dim_N=MDIM,
        out_f=lambda q, m_out: message(0.0, 1.0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: n_in[1]),
        label=label,
    )


def the_load():
    return Arrangement(
        P=configuration(trivial()),
        in_dim_M=0, out_dim_M=0, in_dim_N=EDIM, out_dim_N=MDIM,
        out_f=lambda q, m_out: message(0.0, 1.0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: jnp.asarray(0.0)),
        label="load",
    )


def holding(n, W, y0=0.0):
    """Alice, Bob if ``n == 2``, and the load, in ``phi``."""
    Z = zero_transfer(DH)

    def T(q):
        g = strength(q[0])
        H = transfer(g, jnp.eye(DH) * g)
        return ((Z, Z, Z, Z), (Z, Z, Z, H), (Z, Z, Z, H if n == 2 else Z), (Z, Z, Z, Z))

    phi = attention_arrangement(
        3, DH, DP,
        placement=lambda q: jnp.stack([jnp.zeros(1), jnp.zeros(1), q]),
        transfers=T, cost=lambda q, m_out, n_in: -W * q[0],
        parameter=configuration(euclidean(1, ETA), init=jnp.array([y0])), label="holding")
    return compose_seq(tensor_arrangements([holder("Alice"), holder("Bob"), the_load()]), phi.arr())


def closed(y, n, W):
    return -W * y - n * strength(y)


def stretch(n, W, steps=4000):
    return float(run_closed(holding(n, W), steps, lambda t: EXT)[-1][-1])


def test_P1_composite_cost_is_the_sum():
    """The composite's forcing on ``y`` is minus the derivative of the closed form."""
    r = np.random.default_rng(7)
    for n in (1, 2):
        g = jax.grad(lambda y: closed(y, n, 0.4))
        for y in r.uniform(-2.0, 3.0, 5):
            a = holding(n, 0.4, y0=float(y))
            xi_q, _, _ = a.omega(a.P.position(a.P.init), jnp.zeros(0), EXT)
            np.testing.assert_allclose(np.ravel(xi_q), [-float(g(y))], atol=1e-12)


def test_P2_alice_alone_holds_a_light_load_and_her_bond_breaks_under_a_heavy_one():
    np.testing.assert_allclose(stretch(1, 0.05), 0.0501, atol=1e-4)
    np.testing.assert_allclose(stretch(1, 0.3), 0.3153, atol=1e-4)
    np.testing.assert_allclose(stretch(1, 0.5), 0.5978, atol=1e-4)
    assert stretch(1, 0.7) > 10.0


def test_P3_together_they_hold_what_breaks_either_alone():
    np.testing.assert_allclose(stretch(2, 1.0), 0.5978, atol=1e-4)
    np.testing.assert_allclose(stretch(2, 1.0), stretch(1, 0.5), atol=1e-9)
    np.testing.assert_allclose(stretch(2, 1.2), 0.8978, atol=1e-4)
    assert stretch(2, 1.25) > 10.0


def test_P4_a_broken_bond_does_not_come_back():
    y = stretch(1, 0.7)
    assert float(strength(y)) < 1e-20
