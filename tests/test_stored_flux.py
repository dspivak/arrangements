"""Circuits whose coils store their own flux (book Ch 6 ``sec.circuits``; plan P76 (g); the predictions are in the book repo's
``notes/runs/moving-coil-2524b69e/RESULTS.md``, written first).

Every test runs a composed arrangement through ``PhiK``; energies are read off its states by ordinary code beside it. Everything is
on ``R^n``. Claude Fable 5.2, session 2524b69e, 2026-11-10.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arrangements.interpretation import trivial_omega
from arrangements.magnetism import cost_part, resistor_cost, select
from arrangements.pcK import PhiK
from arrangements.stored_flux import coil, coils, core_flux, ground_v, motor_winding, rotor, tank, winding
from arrangements.wiring import compose_seq, tensor_arrangements

jax.config.update("jax_enable_x64", True)

_IN = (jnp.zeros(0), trivial_omega(0))
_TRIV = lambda op: (jnp.zeros(0), jnp.zeros(0))


def trace(arr, steps):
    """The states of ``steps`` steps of the closed arrangement, each flattened in the order composed."""
    O = PhiK(arr)
    step = jax.jit(lambda s: O.with_state(s).run_one(_IN, _TRIV)[1])
    flat = lambda s: np.concatenate([np.asarray(x, float).ravel() for x in jax.tree_util.tree_leaves(s)])
    s = O.state
    out = [flat(s)]
    for _ in range(steps):
        s = step(s)
        out.append(flat(s))
    return np.array(out)


def lc(rule, tau, kappa=0.0, s=(2.0, 0.0), C=1.0, L=2.0):
    caps = tensor_arrangements([tank(C, s[0], rule, tau), tank(C, s[1], rule, tau)])
    if kappa == 0.0:
        return compose_seq(caps, coil(L, 0.0, rule, tau))
    top = tensor_arrangements([coil(L, 0.0, rule, tau), resistor_cost(kappa, 0, 1, 2)])
    return compose_seq(compose_seq(caps, select(2, [0, 1, 0, 1])), top)


def test_labels():
    assert lc("rk4", 0.05).label == "(coil(2) . (tank(1) || tank(1)))"
    assert lc("rk4", 0.05, kappa=0.2).label == "((coil(2) || R(0.2)) . (sel(0, 1, 0, 1) . (tank(1) || tank(1))))"


def test_g1_two_capacitors_and_a_coil():
    tau = 0.05
    X = trace(lc("rk4", tau), 400)
    t = tau * np.arange(401)
    assert np.abs(X[:, 0] - (1 + np.cos(t))).max() < 1e-6
    assert np.abs(X[:, 1] - (1 - np.cos(t))).max() < 1e-6
    assert np.abs(X[:, 2] - 2 * np.sin(t)).max() < 3e-6   # amplitude 2: the same relative error as the charges (2.0e-6 found)
    E = X[:, 0] ** 2 / 2 + X[:, 1] ** 2 / 2 + X[:, 2] ** 2 / 4
    assert np.abs(E - 2).max() < 1e-7


def test_g2_euler_gains_energy_by_an_exact_factor():
    tau = 0.5
    arr = compose_seq(tensor_arrangements([tank(1.0, 1.0, "euler", tau), ground_v()]), coil(1.0, 0.0, "euler", tau))
    X = trace(arr, 20)
    E = X[:, 0] ** 2 / 2 + X[:, 1] ** 2 / 2
    assert np.allclose(E[1:] / E[:-1], 1 + tau ** 2, rtol=1e-12)
    X = trace(lc("euler", 1.0), 4)
    E = X[:, 0] ** 2 / 2 + X[:, 1] ** 2 / 2 + X[:, 2] ** 2 / 4
    assert np.allclose(E, [2, 3, 5, 9, 17], rtol=1e-12)


def test_g3_a_resistor_burns_what_the_circuit_loses():
    tau, kappa = 0.05, 0.2
    X = trace(lc("rk4", tau, kappa=kappa), 400)
    E = X[:, 0] ** 2 / 2 + X[:, 1] ** 2 / 2 + X[:, 2] ** 2 / 4
    power = kappa * (X[:, 0] - X[:, 1]) ** 2
    trapezoid = tau * (power[:-1] + power[1:]).sum() / 2
    burned = tau / 3 * (power[0] + power[-1] + 4 * power[1:-1:2].sum() + 2 * power[2:-1:2].sum())   # Simpson's rule
    assert abs((E[0] - E[-1]) - trapezoid) < 2e-4   # 1.3e-4 found: the trapezoid's own error
    assert abs((E[0] - E[-1]) - burned) < 1e-6
    assert np.diff(E).max() < 1e-8


def transformer(g, tau, rule="rk4"):
    parts = tensor_arrangements([tank(1.0, 1.0, rule, tau), tank(1.0, 0.0, rule, tau), core_flux(g, 0.0, rule, tau)])
    tops = tensor_arrangements([winding(1.0, 0.9, 0.0, rule, tau), winding(1.0, 0.9, 0.0, rule, tau),
                                cost_part(1, lambda m: 10.0 * m[0] ** 2 / 2, "core(R=10)")])
    return compose_seq(compose_seq(parts, select(3, [0, 2, 1, 2, 2])), tops)


def one_part_coils(tau, rule="rk4"):
    caps = tensor_arrangements([tank(1.0, 1.0, rule, tau), tank(1.0, 0.0, rule, tau)])
    return compose_seq(caps, coils([[1.0, 0.1], [0.1, 1.0]], [0.0, 0.0], rule, tau))


def test_transformer_label():
    assert transformer(10.0, 0.001).label == ("(((winding(N=1,l=0.9) || winding(N=1,l=0.9)) || core(R=10)) . "
                                              "(sel(0, 2, 1, 2, 2) . ((tank(1) || tank(1)) || coreFlux(g=10))))")


@pytest.mark.parametrize("g,gap_want,end_want", [(1.0, 0.224, 0.657), (10.0, 0.0285, 0.945), (100.0, 0.00292, 0.994)])
def test_g4_transformer_with_a_stored_core(g, gap_want, end_want):
    tau, steps = 0.001, 70000
    X = trace(transformer(g, tau), steps)      # s1, s2, Phi, phi1, phi2
    Y = trace(one_part_coils(tau), steps)      # s1, s2, phi1, phi2
    gap = max(np.abs(X[:, [0, 1, 3, 4]] - Y).max(), 0.0)
    assert gap == pytest.approx(gap_want, rel=0.02)
    i1, i2 = (X[:, 3] - X[:, 2]) / 0.9, (X[:, 4] - X[:, 2]) / 0.9
    E = X[:, 0] ** 2 / 2 + X[:, 1] ** 2 / 2 + 0.9 * i1 ** 2 / 2 + 0.9 * i2 ** 2 / 2 + 10.0 * X[:, 2] ** 2 / 2
    assert E[-1] / E[0] == pytest.approx(end_want, rel=0.02)
    assert np.diff(E).max() < 1e-9


def motor(Kc, b, C, s0, rule, tau):
    parts = tensor_arrangements([tank(C, s0, rule, tau), rotor(1.0, 0.0, 0.0, b, rule, tau)])
    return compose_seq(parts, motor_winding(Kc, 1.0, 0.0, rule, tau))


def test_g5_motor_without_drag():
    X = trace(motor(2.0, 0.0, 1.0, 1.0, "rk4", 0.02), 1000)   # s, theta, p, phi
    assert motor(2.0, 0.0, 1.0, 1.0, "rk4", 0.02).label == "(motorWinding(K=2,L=1) . (tank(1) || rotor(J=1,b=0)))"
    assert np.abs(X[:, 0] + X[:, 2] / 2.0 - 1.0).max() < 1e-12
    i = X[:, 3] - 2.0 * X[:, 1]
    E = X[:, 0] ** 2 / 2 + i ** 2 / 2 + X[:, 2] ** 2 / 2
    assert np.abs(E - 0.5).max() < 1e-7
    assert np.abs(X[:, 2]).max() > 0.1


def test_g5_motor_settles_at_the_battery_voltage_over_K():
    X = trace(motor(2.0, 1.0, 1e4, 1e4, "rk4", 0.02), 2000)
    speed, current = X[-1, 2], X[-1, 3] - 2.0 * X[-1, 1]
    assert speed == pytest.approx(0.5, rel=0.01)
    assert current == pytest.approx(0.25, rel=0.02)
