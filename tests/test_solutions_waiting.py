"""Book Ch 5 ``sec.solutions``: ``ex.waiting_rod_mass``, ``exc.decay_growth`` and the composition law of
``prop.solutions_functor`` (Claude Opus 5.5, c9fce825, 2026-11-03).

The library has no tangent linkages, so a solution is approached by running a linkage at shrinking
durations: each arrangement here is built in ``arr`` and run by the dynamics functor ``Phi`` (``closed_step`` is
``Phi``'s step, checked against it). The composition law is checked exactly at each duration, one step at a time,
which is the law of ``prop.solutions_functor`` before the limit: a run of the composite is a pair of runs of the parts
that agree on the shared port, value and load.

Predictions, written before the first run: (a) a motionless rod segment stays put; (b) a mass on a spring has error
against the book's formula that halves with the duration, and returns after 2 pi / nu up to that error; (c) decay
and growth are matched by ``h(s) = 1/s`` with error that shrinks with the duration; (d) the composite
closed system and the whole chain have equal states at every step; (e) each part alone, told the whole's value and
load at the shared port, reproduces its own state at every step; (f) the unit changes nothing.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact, zero_form
from arrangements.coalgebra import configuration, phase
from arrangements.functors import Phi, closed_step
from arrangements.interpretation import trivial_omega
from arrangements.rvect import diagonal
from arrangements.wiring import boxes_and_couplers, compose_seq, trivial_parameter

Z = jnp.zeros(0)
IN = (Z, trivial_omega(0))


def closed_part(parameter, response, init, cost, label):
    """A closed system: one part with effort field ``-d cost`` on its configuration."""
    return Arrangement(
        P=parameter(diagonal(jnp.array([response])), init=init),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m: Z, in_f=lambda q, m, n: Z,
        omega=exact(lambda q, m, n: -cost(q[0])),
        label=label,
    )


def run(arr, total, n):
    """The state of ``arr`` after ``n`` steps of duration ``total / n``, through the dynamics functor."""
    step = closed_step(arr, check=2, duration=total / n)
    s = arr.P.init
    for _ in range(n):
        s = step(s)
    return s


def flat(s):
    return np.concatenate([np.ravel(x) for x in jax.tree_util.tree_leaves(s)])


# ---------------------------------------------------------------------------
# ex.waiting_rod_mass
# ---------------------------------------------------------------------------


def test_motionless_rod_is_fixed_by_waiting():
    rod = closed_part(configuration, 1.0, jnp.array([0.7]), lambda q: 0.0 * q, "rod")
    assert rod.label == "rod"
    np.testing.assert_allclose(flat(run(rod, 3.0, 30)), [0.7], atol=1e-14)


def test_mass_on_a_spring_moves_by_the_books_formula_and_returns():
    m, kappa, q0, p0 = 2.0, 3.0, 0.7, 1.1
    nu = np.sqrt(kappa / m)
    mass = closed_part(phase, 1.0 / m, (jnp.array([q0]), jnp.array([p0])), lambda q: 0.5 * kappa * q**2, "mass")

    def expected(tau):
        return np.array([q0 * np.cos(nu * tau) + p0 / (m * nu) * np.sin(nu * tau),
                         p0 * np.cos(nu * tau) - m * nu * q0 * np.sin(nu * tau)])

    tau = 1.3
    errors = [np.linalg.norm(flat(run(mass, tau, n)) - expected(tau)) for n in (400, 800, 1600)]
    assert errors[2] < errors[1] < errors[0]
    np.testing.assert_allclose(errors[0] / errors[1], 2.0, atol=0.1)
    np.testing.assert_allclose(errors[1] / errors[2], 2.0, atol=0.1)
    # it moves, unlike the rod, and returns after 2 pi / nu
    assert np.linalg.norm(flat(run(mass, 2 * np.pi / nu / 4, 800)) - np.array([q0, p0])) > 0.5
    np.testing.assert_allclose(flat(run(mass, 2 * np.pi / nu, 6400)), [q0, p0], atol=5e-3)


# ---------------------------------------------------------------------------
# exc.decay_growth
# ---------------------------------------------------------------------------


def test_decay_and_growth_are_matched_by_one_over_s():
    """Two closed systems on ``R``, no rod and no heat: the cost ``-s^2/2`` is false of a rod segment (David, 2026-11-04)."""
    s0, tau = 0.8, 1.1
    errors = []
    for n in (400, 800, 1600):
        decay = closed_part(configuration, 1.0, jnp.array([s0]), lambda q: 0.5 * q**2, "decay")
        growth = closed_part(configuration, 1.0, jnp.array([1.0 / s0]), lambda q: -0.5 * q**2, "growth")
        a, b = flat(run(decay, tau, n))[0], flat(run(growth, tau, n))[0]
        errors.append(abs(1.0 / a - b))
    assert errors[2] < errors[1] < errors[0]
    # (1 - d)^(-n) against (1 + d)^n at d = tau / n: the error is e^tau tau d / s0 to first order in d
    # (a bound guessed at 2e-3 failed first, at 2.84e-3; this estimate, by hand, is what the run gives).
    np.testing.assert_allclose(errors[2], np.exp(tau) * tau * (tau / 1600) / s0, rtol=0.01)


# ---------------------------------------------------------------------------
# prop.solutions_functor, the composition law, one step at a time
# ---------------------------------------------------------------------------

C1, C2, C3, K = 1.0, 2.0, 0.5, 0.75
TH0 = (1.0, -0.4, 2.5)


def segment(c, th0):
    return Arrangement(
        P=configuration(diagonal(jnp.array([1.0 / c])), init=jnp.array([th0])),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m: q, in_f=lambda q, m, n: Z, omega=zero_form, label="segment",
    )


def joint():
    return Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=2, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m: Z, in_f=lambda q, m, n: Z,
        omega=exact(lambda q, m, n: -0.5 * K * (m[0] - m[1]) ** 2), label="joint",
    )


def left_pair():
    """Segments 1 and 2 wired to the first joint, showing the second segment's temperature: ``1 -> <0|R>``."""
    from arrangements.wiring import tensor_arrangements
    wire = Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=2, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m: m[1:], in_f=lambda q, m, n: Z,
        omega=exact(lambda q, m, n: -0.5 * K * (m[0] - m[1]) ** 2), label="joint_and_port",
    )
    return compose_seq(tensor_arrangements([segment(C1, TH0[0]), segment(C2, TH0[1])]), wire)


def right_part():
    """Segment 3 with the second joint, reading the shared value: ``<0|R> -> 1``."""
    return Arrangement(
        P=configuration(diagonal(jnp.array([1.0 / C3])), init=jnp.array([TH0[2]])),
        in_dim_M=0, out_dim_M=1, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m: Z, in_f=lambda q, m, n: Z,
        omega=exact(lambda q, m, n: -0.5 * K * (m[0] - q[0]) ** 2), label="segment_and_joint",
    )


def test_composite_is_the_whole_chain_step_by_step():
    whole = boxes_and_couplers([segment(C1, TH0[0]), segment(C2, TH0[1]), segment(C3, TH0[2])],
                               [(0, 1), (1, 2)], [joint(), joint()])
    grouped = compose_seq(left_pair(), right_part())
    assert "segment" in whole.label and "segment" in grouped.label
    for dur in (0.02, 0.1):
        a, b = whole.P.init, grouped.P.init
        sa, sb = closed_step(whole, check=2, duration=dur), closed_step(grouped, check=2, duration=dur)
        for _ in range(40):
            a, b = sa(a), sb(b)
            np.testing.assert_allclose(flat(a), flat(b), atol=1e-12)


def test_each_part_alone_reproduces_the_composite_from_the_shared_port():
    """A run of the composite is a pair of runs of the parts agreeing on the shared value and load."""
    left, right = left_pair(), right_part()
    grouped = compose_seq(left, right)
    for dur in (0.02, 0.1):
        step = closed_step(grouped, check=2, duration=dur)
        s_whole = grouped.P.init
        s_left, s_right = left.P.init, right.P.init
        O_left, O_right = Phi(left), Phi(right)
        for _ in range(40):
            th = flat(s_whole)
            # the shared port's value and the load the left part's output receives
            value = jnp.array([th[1]])
            load = jnp.array([K * (th[2] - th[1])])
            _, _, s_left = O_left.with_state(s_left).run_one(IN, lambda _o: (load, Z), dur)
            _, _, s_right = O_right.with_state(s_right).run_one((value, trivial_omega(0)), lambda _o: (Z, Z), dur)
            s_whole = step(s_whole)
            np.testing.assert_allclose(np.concatenate([flat(s_left), flat(s_right)]), flat(s_whole), atol=1e-12)


def test_the_unit_changes_nothing():
    unit = Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=1, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m: m, in_f=lambda q, m, n: Z, omega=zero_form, label="id",
    )
    plain = compose_seq(left_pair(), right_part())
    left_unit = compose_seq(compose_seq(left_pair(), unit), right_part())
    right_unit = compose_seq(left_pair(), compose_seq(unit, right_part()))
    a, b, c = plain.P.init, left_unit.P.init, right_unit.P.init
    sa, sb, sc = (closed_step(x, check=2, duration=0.05) for x in (plain, left_unit, right_unit))
    for _ in range(40):
        a, b, c = sa(a), sb(b), sc(c)
        np.testing.assert_allclose(flat(a), flat(b), atol=1e-12)
        np.testing.assert_allclose(flat(a), flat(c), atol=1e-12)
