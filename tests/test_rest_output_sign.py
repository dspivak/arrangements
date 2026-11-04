"""Book Ch 5 ``sec.groupings_at_rest``, the paragraph after ``def.rest_relation``: why the load at an output port is
recorded with a minus (Claude Opus 5.5, ccc7e1fe, 2026-11-04).

Every system here is an arrangement run by the dynamics functor: ``open_step`` is ``Phi``'s step, checked against it.
The two beads are built with ``parallel_arrangements`` and ``compose_seq`` from a wiring lens. A state is at rest when a
step leaves it fixed, under the input value and the output load it is told. The closedness check is an ordinary
Jacobian of the rest map beside those runs.

Predictions, written before the first run: (a) a bead with input x, output y and stiffness k is at rest with zero
momentum exactly when the load it receives at its output is k(y - x), and it then pushes back the same k(y - x); (b) its
outward pair (pushback, minus received) is -dC for C = k/2 (x-y)^2 and is closed, while the received pair is not;
(c) two beads of stiffness 1 and 3, ends 0 and 1, wired output to input, rest with y = 3/4 and one load 3/4 on all
three wires, the outward loads on the inner wire cancelling; (d) their port data (0, 1, 3/4, -3/4) equal those of two
springs joined at a mass with both ends inputs, though the far end is an output in one and an input in the other.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact
from arrangements.coalgebra import phase
from arrangements.functors import open_step
from arrangements.rvect import diagonal
from arrangements.wiring import compose_seq, parallel_arrangements, trivial_parameter

Z = jnp.zeros(0)
DURATIONS = (0.25, 1.0)


def mass(m, q0=0.0):
    return phase(diagonal(jnp.array([1.0 / m])), init=(jnp.array([q0]), jnp.array([0.0])))


def bead(m, k, y0=0.0):
    """Ch 3 ``ex.bead``: a mass and a spring, one input x, one output y, effort field -d(k/2 (x-y)^2)."""
    return Arrangement(
        P=mass(m, y0), in_dim_M=0, out_dim_M=0, in_dim_N=1, out_dim_N=1,
        out_f=lambda q, m_: q, in_f=lambda q, m_, n: Z,
        omega=exact(lambda q, m_, n: -0.5 * k * jnp.sum((n - q) ** 2)),
        label=f"bead({k:g})",
    )


def beads_wire():
    """The wiring diagram: first bead's output to second's input; external input to first; second's output out."""
    return Arrangement(
        P=trivial_parameter(), in_dim_M=2, out_dim_M=2, in_dim_N=1, out_dim_N=1,
        out_f=lambda q, m_: m_[1:], in_f=lambda q, m_, n: jnp.concatenate([n, m_[:1]]),
        omega=lambda q, m_, n: (Z, jnp.zeros(2), jnp.zeros(1)),
        label="wire",
    )


def fixed(arr, n_in, xi_out, duration):
    s = arr.P.init
    s1 = open_step(arr, n_in, xi_out, check=2, duration=duration)(s)
    return all(
        bool(jnp.allclose(a, b, atol=1e-6))
        for a, b in zip(jax.tree_util.tree_leaves(s), jax.tree_util.tree_leaves(s1))
    )


def pushback(arr, q, n_in):
    return np.asarray(arr.omega(q, Z, jnp.asarray(n_in, dtype=float))[2])


def test_one_bead_receives_what_it_pushes_back():
    k, x, y = 2.0, 1.0, 4.0
    b = bead(1.0, k, y)
    for d in DURATIONS:
        assert fixed(b, [x], [k * (y - x)], d)
        assert not fixed(b, [x], [k * (y - x) + 0.5], d)
    assert np.allclose(pushback(b, jnp.array([y]), [x]), k * (y - x))


def test_outward_pair_is_minus_dC_and_closed_received_pair_is_not():
    k = 2.0

    def rest_pairs(v):
        x, y = v[:1], v[1:]
        xi_q, _, xi_n = bead(1.0, k).omega(y, Z, x)
        received = -xi_q  # condition (iv): xi_q + received = 0
        return jnp.concatenate([xi_n, -received]), jnp.concatenate([xi_n, received])

    v = jnp.array([1.0, 4.0])
    outward, received = rest_pairs(v)
    minus_dC = -jax.grad(lambda u: 0.5 * k * (u[0] - u[1]) ** 2)(v)
    assert np.allclose(outward, minus_dC)
    J_out, J_rec = jax.jacobian(lambda u: rest_pairs(u)[0])(v), jax.jacobian(lambda u: rest_pairs(u)[1])(v)
    assert np.allclose(J_out, J_out.T)
    assert not np.allclose(J_rec, J_rec.T)


def test_two_beads_carry_one_load_and_the_inner_outward_loads_cancel():
    k1, k2, x, z = 1.0, 3.0, 0.0, 1.0
    y = (k1 * x + k2 * z) / (k1 + k2)
    load = k1 * k2 / (k1 + k2) * (z - x)
    first, second = bead(1.0, k1, y), bead(1.0, k2, z)
    two = compose_seq(parallel_arrangements(first, second), beads_wire())
    for d in DURATIONS:
        assert fixed(two, [x], [load], d)              # the composite, external load on the last wire
        assert fixed(first, [x], [load], d)            # each part alone, told the load on its wires
        assert fixed(second, [y], [load], d)
    external_pushback = pushback(two, jnp.array([y, z]), [x])
    inner_pushback = pushback(second, jnp.array([z]), [y])
    assert np.allclose(external_pushback, load)
    assert np.allclose(inner_pushback, load)
    assert np.allclose(-load + inner_pushback, 0.0)     # first's outward at its output, second's at its input


def test_port_data_agree_with_two_springs_both_ends_inputs():
    k1, k2, x1, x2 = 1.0, 3.0, 0.0, 1.0
    q = (k1 * x1 + k2 * x2) / (k1 + k2)
    springs = Arrangement(
        P=mass(1.0, q), in_dim_M=0, out_dim_M=0, in_dim_N=2, out_dim_N=0,
        out_f=lambda q_, m_: Z, in_f=lambda q_, m_, n: Z,
        omega=exact(lambda q_, m_, n: -0.5 * k1 * jnp.sum((n[0] - q_) ** 2) - 0.5 * k2 * jnp.sum((n[1] - q_) ** 2)),
        label="two_springs",
    )
    for d in DURATIONS:
        assert fixed(springs, [x1, x2], None, d)
    data_springs = np.concatenate([[x1, x2], pushback(springs, jnp.array([q]), [x1, x2])])

    load = k1 * k2 / (k1 + k2) * (x2 - x1)
    two = compose_seq(parallel_arrangements(bead(1.0, k1, q), bead(1.0, k2, x2)), beads_wire())
    data_beads = np.concatenate([[x1, x2], pushback(two, jnp.array([q, x2]), [x1]), [-load]])
    assert np.allclose(data_springs, data_beads)
    assert np.allclose(data_beads, [0.0, 1.0, 0.75, -0.75])
