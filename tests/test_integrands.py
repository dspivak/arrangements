"""Integrands and the differential equation of a closed system: Ch 5 ``ex.tangent_heat``, ``ex.tangent_heat_chain`` and
``ex.tangent_phase`` (with ``exc.tangent_two_masses``).

The library has no tangent arrangements. What it has is the step of an arrangement at every duration, so the
differential equation is read where the book reads it, ``eqn.closed_system_rate``: the derivative at duration ``0`` of
the closed system's step, taken by ``jax.jvp`` through ``Phi``'s own step with the duration as the variable. Every
system is an arrangement built with ``compose_seq`` and run by ``Phi``; the grouped chain's label is read.

Predictions, written before the first run (Claude Opus 5.5, 00f2d232, 2026-11-12):
(a) a rod segment of heat capacity ``C = 4``, told a load ``xi``, changes its temperature at the rate ``xi / C`` at
    every duration (1/4 and 1) and at duration 0 (``ex.tangent_heat``);
(b) three segments of capacities ``(1, 2, 4)`` joined by conductance ``1/2``: the derivative at 0 of the step is
    ``eqn.three_segment_ode`` divided through by the capacities, and grouping the first two segments with their joint
    gives the same step at durations 1/4 and 1 and the same derivative (``ex.tangent_heat_chain``);
(c) a mass ``m = 2`` with the symplectic Euler integrator and the cost ``c(q) = q^4/4 - q^2``: its step at durations
    1/4 and 1 is ``(q + tau p/m, p - tau c'(q + tau p/m))``, and the derivative at 0 is ``(p/m, -c'(q))``, Hamilton's
    equations (``ex.tangent_phase``); two masses on a spring keep their total momentum at every step and the rate of
    each momentum is its load (``exc.tangent_two_masses``).
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact, zero_form
from arrangements.coalgebra import configuration, phase
from arrangements.functors import Phi, closed_step, open_step
from arrangements.interpretation import trivial_omega
from arrangements.rvect import diagonal
from arrangements.wiring import boxes_and_couplers, compose_seq, tensor_arrangements, trivial_parameter

jax.config.update("jax_enable_x64", True)

Z = jnp.zeros(0)
IN = (Z, trivial_omega(0))
TRIV = lambda _o: (jnp.zeros(0), jnp.zeros(0))


def flat(s):
    return jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(s)])


def rate_at_zero(arr, s):
    """``d/dtau`` at ``0`` of the closed system's step from ``s`` (``eqn.closed_system_rate``), through ``Phi``."""
    O = Phi(arr)
    f = lambda tau: flat(O.with_state(s).run_one(IN, TRIV, tau)[2])
    at0, rate = jax.jvp(f, (jnp.asarray(0.0),), (jnp.asarray(1.0),))
    np.testing.assert_allclose(at0, flat(s), atol=1e-12)          # the zero-duration condition
    return np.asarray(rate)


def part(P, label):
    return Arrangement(P=P, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1, out_f=lambda q, m: q,
                       in_f=lambda q, m, n: Z, omega=zero_form, label=label)


def segment(C, th):
    return part(configuration(diagonal(jnp.array([1.0 / C])), init=jnp.array([th])), f"segment({C:g})")


def joint(kappa):
    return Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=2, in_dim_N=0, out_dim_N=0, out_f=lambda q, m: Z,
                       in_f=lambda q, m, n: Z, omega=exact(lambda q, m, n: -0.5 * kappa * (m[1] - m[0]) ** 2),
                       label="joint")


def test_rod_segment_rate_is_load_over_capacity():
    """Prediction (a)."""
    C, th, xi = 4.0, 3.0, 0.6
    seg = segment(C, th)
    for tau in (0.25, 1.0):
        s = open_step(seg, [], [xi], duration=tau)(seg.P.init)
        np.testing.assert_allclose((float(s[0]) - th) / tau, xi / C, atol=1e-12)
    f = lambda tau: Phi(seg).with_state(seg.P.init).run_one(IN, lambda _o: (jnp.array([xi]), Z), tau)[2]
    _, rate = jax.jvp(f, (jnp.asarray(0.0),), (jnp.asarray(1.0),))
    np.testing.assert_allclose(rate, [xi / C], atol=1e-12)


CS, KAPPA, TH = (1.0, 2.0, 4.0), 0.5, (3.0, -1.0, 2.0)


def three_segments():
    return boxes_and_couplers([segment(c, t) for c, t in zip(CS, TH)], [(0, 1), (1, 2)], [joint(KAPPA), joint(KAPPA)])


def grouped():
    """The picture: the first two segments and their joint grouped, the group passing out ``theta_2``."""
    inner = Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=2, in_dim_N=0, out_dim_N=1,
                        out_f=lambda q, m: m[1:], in_f=lambda q, m, n: Z,
                        omega=exact(lambda q, m, n: -0.5 * KAPPA * (m[1] - m[0]) ** 2), label="joint+dot")
    group = compose_seq(tensor_arrangements([segment(CS[0], TH[0]), segment(CS[1], TH[1])]), inner)
    return boxes_and_couplers([group, segment(CS[2], TH[2])], [(0, 1)], [joint(KAPPA)])


def test_three_segments_ode_and_grouping():
    """Prediction (b)."""
    t1, t2, t3 = TH
    ode = np.array([KAPPA * (t2 - t1) / CS[0], (KAPPA * (t1 - t2) + KAPPA * (t3 - t2)) / CS[1], KAPPA * (t2 - t3) / CS[2]])
    a, b = three_segments(), grouped()
    assert b.label.count("segment") == 3 and "joint+dot" in b.label
    ra, rb = rate_at_zero(a, a.P.init), rate_at_zero(b, b.P.init)
    np.testing.assert_allclose(ra, ode, atol=1e-12)
    np.testing.assert_allclose(rb, ode, atol=1e-12)
    for tau in (0.25, 1.0):
        sa, sb = closed_step(a, duration=tau)(a.P.init), closed_step(b, duration=tau)(b.P.init)
        np.testing.assert_allclose(flat(sa), np.array(TH) + tau * ode, atol=1e-12)
        np.testing.assert_allclose(flat(sb), flat(sa), atol=1e-12)


def test_mass_gives_hamiltons_equations():
    """Prediction (c), ``ex.tangent_phase``."""
    m, q0, p0 = 2.0, 0.8, -0.5
    dc = lambda q: q ** 3 - 2.0 * q                        # c(q) = q^4/4 - q^2
    arr = Arrangement(P=phase(diagonal(jnp.array([1.0 / m])), init=(jnp.array([q0]), jnp.array([p0]))),
                      in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=0, out_f=lambda q, m_: Z, in_f=lambda q, m_, n: Z,
                      omega=exact(lambda q, m_, n: -(q[0] ** 4 / 4 - q[0] ** 2)), label="mass")
    for tau in (0.25, 1.0):
        x = q0 + tau * p0 / m
        np.testing.assert_allclose(flat(closed_step(arr, duration=tau)(arr.P.init)), [x, p0 - tau * dc(x)], atol=1e-12)
    np.testing.assert_allclose(rate_at_zero(arr, arr.P.init), [p0 / m, -dc(q0)], atol=1e-12)


def test_two_masses_exchange_equal_and_opposite_loads():
    """Prediction (c), ``exc.tangent_two_masses``."""
    m1, m2, kappa = 1.0, 3.0, 2.0
    q, p = (0.0, 1.5), (0.4, -0.1)
    masses = [part(phase(diagonal(jnp.array([1.0 / mi])), init=(jnp.array([qi]), jnp.array([pi]))), "mass")
              for mi, qi, pi in zip((m1, m2), q, p)]
    arr = boxes_and_couplers(masses, [(0, 1)], [joint(kappa)])
    r = rate_at_zero(arr, arr.P.init)                       # leaves in order: q1, p1, q2, p2
    np.testing.assert_allclose(r, [p[0] / m1, kappa * (q[1] - q[0]), p[1] / m2, kappa * (q[0] - q[1])], atol=1e-12)
    for tau in (0.25, 1.0):
        step = closed_step(arr, duration=tau)
        s = arr.P.init
        for _ in range(20):
            s = step(s)
            f = flat(s)
            assert abs(f[1] + f[3] - (p[0] + p[1])) < 1e-12
