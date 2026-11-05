"""Light typed as the coil is: capacitor edges and coil faces (book plan §5.9 item 90; run notes ``notes/runs/light-12b2681e/`` in the
book, L1--L6 of its RESULTS.md). N = 6 and every setting is RESULTS's: none had to be made smaller for time.

Every test runs a composed arrangement through ``PhiK``, labels ``(lattice_wire(N=6) . (... cap ... coil4 ...))`` and ``(lattice_wire(N=6)
. (... mass ... spring4 ...))``; energy, Gauss and the comparison under ``g`` are checks read off the states. Every test says which of
three things it does (the convention of test_rendezvous.py): reproduce a CLOSED FORM, check a DISCRETE ANALOGUE, or merely ILLUSTRATE.
The tolerances bracket the numbers RESULTS.md reports.

Claude Opus 5.5, session 5c9967d2, 2026-11-05.
"""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arrangements.interpretation import trivial_omega
from arrangements.light_tank import (curl_matrix, energy_tank, flux_lattice, flux_read, flux_to_tank, grad_matrix, mode_start, tank_lattice,
                                     tank_lattice_stagger, tank_read)
from arrangements.pcK import PhiK

jax.config.update("jax_enable_x64", True)

N = 6
_IN = (jnp.zeros(0), trivial_omega(0))
_TRIV = lambda _o: (jnp.zeros(0), jnp.zeros(0))
C = curl_matrix(N)
D = grad_matrix(N)


def _random_start(seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=2 * N * N)
    X = rng.normal(size=2 * N * N)
    return A, X


A0, X0 = _random_start()
E0, B0 = flux_to_tank(N, A0, X0)


def jit_stepper(arr):
    O = PhiK(arr)
    return O.state, jax.jit(lambda s: O.with_state(s).run_one(_IN, _TRIV)[1])


def tank_of(rule, tau, K=1):
    if rule == "stagger":
        return tank_lattice_stagger(N, E0, B0, tau)
    return tank_lattice(N, E0, B0, rule, tau, K)


@lru_cache(maxsize=None)
def band_run(rule, tau, K=1, T=50.0):
    """Over time T from the random start: (largest |E/E0 - 1|, E_end/E0, Gauss drift, mean B drift, largest |E|), sampled as RESULTS's
    script sampled, about 200 times."""
    s, f = jit_stepper(tank_of(rule, tau, K))
    n = int(round(T / tau))
    E, B = tank_read(N, s)
    e0 = energy_tank(E, B)
    div0 = D.T @ E
    mb0 = B.mean()
    band, last = 0.0, 1.0
    every = max(1, n // 200)
    for i in range(n):
        s = f(s)
        if (i + 1) % every == 0 or i == n - 1:
            E, B = tank_read(N, s)
            last = energy_tank(E, B) / e0
            band = max(band, abs(last - 1))
    E, B = tank_read(N, s)
    return band, last, np.abs(D.T @ E - div0).max(), abs(B.mean() - mb0), np.abs(E).max()


def worst_gap(sF, fF, sT, fT, steps):
    """The largest difference between the tank lattice's state and g of the flux lattice's, over the steps."""
    worst = 0.0
    for _ in range(steps):
        sF, sT = fF(sF), fT(sT)
        a, x = flux_read(sF)
        E, B = tank_read(N, sT)
        gE, gB = flux_to_tank(N, a, x)
        worst = max(worst, np.abs(E - gE).max(), np.abs(B - gB).max())
    return worst


def one_step_ratio(rule, tau):
    B, _ = mode_start(N, 1, 1)
    s, f = jit_stepper(tank_lattice(N, np.zeros(2 * N * N), B, rule, tau))
    E_, B_ = tank_read(N, s)
    e0 = energy_tank(E_, B_)
    E_, B_ = tank_read(N, f(s))
    return energy_tank(E_, B_) / e0


def test_d_d_is_zero():
    """CLOSED FORM. d d = 0, on which Gauss's law rests (d_0^T d^T = 0)."""
    assert np.abs(C @ D).max() == 0


def test_labels():
    """ILLUSTRATE. The composites are what RESULTS.md says they are, read off their labels."""
    tank = tank_of("euler", 0.1).label
    flux = flux_lattice(N, A0, X0, "symplectic", 0.1).label
    assert tank.startswith("(lattice_wire(N=6) . ") and "cap" in tank and "coil4" in tank
    assert flux.startswith("(lattice_wire(N=6) . ") and "mass" in flux and "spring4" in flux


@pytest.mark.parametrize("tau", [0.5, 0.1, 0.01])
def test_L1_eigenmode_factor(tau):
    """CLOSED FORM. L1: one round on an eigenmode, omega = sqrt 2, multiplies its energy by exactly 1 + (omega tau)^2 in one step. RESULTS:
    1.500000000000, 1.020000000000, 1.000200000000."""
    _, w = mode_start(N, 1, 1)
    assert w == pytest.approx(np.sqrt(2), abs=1e-12)
    assert one_step_ratio("euler", tau) == pytest.approx(1 + (w * tau) ** 2, abs=1e-12)


@pytest.mark.parametrize("tau,end_want,rel", [(0.5, 8.6e44, 0.01), (0.1, 1.7e14, 0.05), (0.01, 11.95, 0.001)])
def test_L1_random_grows(tau, end_want, rel):
    """DISCRETE ANALOGUE. L1: from a random start over time 50, one round grows without bound at every fixed step, and falls toward 1 as tau
    shrinks. RESULTS: E_end/E0 8.6e44, 1.7e14, 11.95."""
    assert band_run("euler", tau)[1] == pytest.approx(end_want, rel=rel)


@pytest.mark.parametrize("tau,gap", [(0.5, 1.5e-13), (0.1, 1.6e-14), (0.01, 3.6e-15)])
def test_L2_stagger_equals_flux_lattice(tau, gap):
    """CLOSED FORM. L2: the two-round tank lattice equals the one-round flux lattice under g, to rounding, at every tau, over 200 steps.
    RESULTS: worst gap 1.5e-13, 1.6e-14, 3.6e-15. So the mass's look-ahead does in one round what coil and capacitor need two for."""
    sF, fF = jit_stepper(flux_lattice(N, A0, X0, "symplectic", tau))
    sT, fT = jit_stepper(tank_lattice_stagger(N, E0, B0, tau))
    assert worst_gap(sF, fF, sT, fT, 200) < 2 * gap


@pytest.mark.parametrize("tau,band_want", [(0.5, 1.21), (0.1, 0.108), (0.01, 0.0097)])
def test_L2_energy_band(tau, band_want):
    """DISCRETE ANALOGUE. L2: Yee's step over time 50: an energy band in proportion to tau, no drift. RESULTS: 1.21, 0.108, 0.0097."""
    assert band_run("stagger", tau)[0] == pytest.approx(band_want, rel=0.01)


@pytest.mark.parametrize("tau,band_want,rel", [(0.7, 4.63, 0.002), (0.71, 4e20, 0.2)])
def test_L2_limit(tau, band_want, rel):
    """CLOSED FORM, its bound. L2: over time 100, bounded at omega tau = 1.98 and blown up at 2.008, omega the largest, sqrt 8. RESULTS: 4.63,
    4e20."""
    assert band_run("stagger", tau, T=100.0)[0] == pytest.approx(band_want, rel=rel)


@pytest.mark.parametrize("tau", [0.5, 0.1])
def test_L3_eigenmode_factor(tau):
    """CLOSED FORM. L3: Ch 7 prop.runge_kutta's four rounds multiply an eigenmode's energy by 1 - x^6/72 + x^8/576, x = omega tau, in one step.
    RESULTS: 0.99837239583333, 0.99999988916667."""
    _, w = mode_start(N, 1, 1)
    x = w * tau
    assert one_step_ratio("rk4", tau) == pytest.approx(1 - x ** 6 / 72 + x ** 8 / 576, abs=1e-13)


@pytest.mark.parametrize("tau,gap", [(0.5, 9.7e-14), (0.1, 2.3e-14)])
def test_L3_intertwines(tau, gap):
    """CLOSED FORM. L3: g, being linear, intertwines Runge--Kutta on both typings, to rounding, over 200 steps. RESULTS: 9.7e-14, 2.3e-14."""
    sF, fF = jit_stepper(flux_lattice(N, A0, X0, "rk4", tau))
    sT, fT = jit_stepper(tank_lattice(N, E0, B0, "rk4", tau))
    assert worst_gap(sF, fF, sT, fT, 200) < 2 * gap


@pytest.mark.parametrize("tau,end_want", [(0.5, 0.367), (0.1, 0.99921)])
def test_L3_energy_falls(tau, end_want):
    """DISCRETE ANALOGUE. L3: over time 50 it never grows, and the prediction "falls very slowly" is false at the long step: the short waves
    die. RESULTS: E_end/E0 0.367, 0.99921."""
    assert band_run("rk4", tau)[1] == pytest.approx(end_want, abs=1e-3 if tau == 0.5 else 1e-5)


@pytest.mark.parametrize("K,gap", [(2, 6.7e-15), (8, 8.0e-15)])
def test_L4_intertwines(K, gap):
    """CLOSED FORM. L4: g intertwines the midpoint rule's K rounds, to rounding, at tau = 0.1 over 100 steps. RESULTS: 6.7e-15, 8.0e-15."""
    sF, fF = jit_stepper(flux_lattice(N, A0, X0, "mid", 0.1, K))
    sT, fT = jit_stepper(tank_lattice(N, E0, B0, "mid", 0.1, K))
    assert worst_gap(sF, fF, sT, fT, 100) < 2 * gap


@pytest.mark.parametrize("tau,K,band_want,rel", [(0.1, 4, 3.5e-3, 0.02), (0.1, 8, 8.3e-7, 0.02), (0.5, 4, 0.81, 0.01), (0.5, 8, 0.40, 0.02)])
def test_L4_energy_band(tau, K, band_want, rel):
    """DISCRETE ANALOGUE. L4: the band over time 50 at K rounds. RESULTS: at tau = 0.1, 3.5e-3, 8.3e-7 at K = 4, 8; at tau = 0.5, 0.81, 0.40. The
    rounds converge as (omega tau / 2)^K, here 0.707^K at the fastest mode."""
    assert band_run("mid", tau, K)[0] == pytest.approx(band_want, rel=rel)


@pytest.mark.parametrize("tau,band_want,slack", [(0.1, 4.4e-16, 5.0), (0.5, 1.0e-4, 1.1)])
def test_L4_thirty_rounds(tau, band_want, slack):
    """DISCRETE ANALOGUE. L4 at K = 30. RESULTS: 4.4e-16 at tau = 0.1, to rounding; 1.0e-4 at tau = 0.5, so "keeps energy to rounding" is false as
    written at the long step."""
    band = band_run("mid", tau, 30)[0]
    assert band < slack * band_want
    if tau == 0.5:
        assert band > 0.9e-4


@pytest.mark.parametrize("rule,tau,K", [("stagger", 0.5, 1), ("stagger", 0.1, 1), ("stagger", 0.01, 1), ("rk4", 0.5, 1), ("rk4", 0.1, 1), ("mid", 0.1, 4),
                                        ("mid", 0.5, 8), ("euler", 0.01, 1)])
def test_L5_gauss_and_mean_flux(rule, tau, K):
    """CLOSED FORM. L5: Gauss's law, the divergence of E at each node, and the mean of B are unchanged to rounding at every integrator, being linear
    quantities the wire kills. RESULTS: drift at most 1.8e-14 wherever the values stay small, 3.1e-14 for one round at tau = 0.01."""
    _, _, dg, dm, _ = band_run(rule, tau, K)
    limit = 4e-14 if rule == "euler" else 2e-14
    assert dg < limit
    assert dm < limit


def test_L5_blown_up_run_is_rounding_relative_to_the_field():
    """CLOSED FORM. L5: in one round's blown-up run at tau = 0.5 the absolute Gauss drift is large against values large in proportion. RESULTS: 2.5e7
    against values "of order 3e22". RESULTS does not say which measure that is; the largest |E| here is 5.8e22, the same order."""
    _, _, dg, _, largest = band_run("euler", 0.5)
    assert dg == pytest.approx(2.5e7, rel=0.05)
    assert 1e22 < largest < 1e23
    assert dg / largest < 1e-14


@pytest.mark.parametrize("tau,period_want", [(0.5, 6.287186), (0.1, 6.283203)])
def test_L6_the_wave(tau, period_want):
    """DISCRETE ANALOGUE. L6: the mode k = (2 pi / 6, 0) has omega = 1, so period 2 pi to order tau^4 under Runge--Kutta; phase speed omega / k = 0.955,
    the lattice's, which is false of light in empty space and true of this coarse grid. RESULTS: period 6.287186, 6.283203; phase speed 0.9549."""
    B, w = mode_start(N, 1, 0)
    assert w == pytest.approx(1.0, abs=1e-12)
    assert w / (2 * np.pi / N) == pytest.approx(0.9549, abs=1e-4)
    s, f = jit_stepper(tank_lattice(N, np.zeros(2 * N * N), B, "rk4", tau))
    n = int(round(2 * np.pi / w / tau)) + 5
    trace = [1.0]
    for _ in range(n):
        s = f(s)
        _, B_ = tank_read(N, s)
        trace.append(B_ @ B / (B @ B))
    trace = np.array(trace)
    times = np.arange(len(trace)) * tau
    i = np.argmax(trace[len(trace) // 2:]) + len(trace) // 2
    y0, y1, y2 = trace[i - 1], trace[i], trace[min(i + 1, len(trace) - 1)]
    shift = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    assert times[i] + shift * tau == pytest.approx(period_want, abs=1e-6)
