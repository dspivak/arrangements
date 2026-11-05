"""The two-port capacitor with circuits, a motor, a transformer and two magnets (book plan §5.9 item 90; run notes
``notes/runs/light-12b2681e/`` in the book, T1--T3 and W1--W5 of its RESULTS.md, at its settings).

Every test runs a composed arrangement through ``PhiK``; energy and the like are checks read off the states. Every test says
which of three things it does (the convention of test_rendezvous.py): reproduce a CLOSED FORM, check a DISCRETE ANALOGUE, or merely
ILLUSTRATE. The tolerances bracket the numbers RESULTS.md reports.

Claude Opus 5.5, session 5c9967d2, 2026-11-05.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arrangements.interpretation import trivial_omega
from arrangements.magnetism import (coils_spring, core_flux, cost_part, dipole_cost, floating_coupler, magnet, motor_winding,
                                    resistor_cost, rotor, select, tank, two_port, wire)
from arrangements.pcK import PhiK
from arrangements.wiring import compose_seq, tensor_arrangements

jax.config.update("jax_enable_x64", True)

_IN = (jnp.zeros(0), trivial_omega(0))
_TRIV = lambda _o: (jnp.zeros(0), jnp.zeros(0))


def runner(arr):
    O = PhiK(arr)
    return O.state, jax.jit(lambda s: O.with_state(s).run_one(_IN, _TRIV)[1])


def flat(st):
    return np.concatenate([np.ravel(np.asarray(x, float)) for x in jax.tree_util.tree_leaves(st) if np.asarray(x).size])


def rlc(C, L, kappa, tau, s=(2.0, 0.0)):
    """Two two-port capacitors, a coil on their flux ports (L not None), a resistor on their voltage ports (kappa not None)."""
    caps = tensor_arrangements([two_port(C, s[0], tau), two_port(C, s[1], tau)])
    outer, idx = [], []
    if L is not None:
        outer.append(coils_spring([[L]], [(0, 1)]))
        idx += [0, 1, 2, 3]
    if kappa is not None:
        outer.append(resistor_cost(kappa, 1, 3, 4))
        idx += [0, 1, 2, 3]
    top = outer[0] if len(outer) == 1 else tensor_arrangements(outer)
    return compose_seq(compose_seq(caps, select(4, idx)), top)


def rc_tank(C, kappa, tau, s=(2.0, 0.0)):
    return wire([tank(C, s[0], tau), tank(C, s[1], tau)], resistor_cost(kappa, 0, 1, 2))


def period_of(trace, tau):
    """Mean spacing of upward zero crossings of the mean-removed trace."""
    x = np.array(trace) - np.mean(trace)
    idx = np.where((x[:-1] < 0) & (x[1:] >= 0))[0]
    cross = idx + (-x[idx]) / (x[idx + 1] - x[idx])
    return np.mean(np.diff(cross)) * tau


def test_label_shows_the_composite():
    """ILLUSTRATE. The composite is what RESULTS.md says it is, read off its label."""
    assert rlc(1.0, 2.0, 0.2, 0.1).label == "((coils(1) || R(0.2)) . (sel(0, 1, 2, 3, 0, 1, 2, 3) . (cap2 || cap2)))"


@pytest.mark.parametrize("tau", [1.0, 0.5, 0.1])
def test_T1_charges_equal_the_tanks(tau):
    """CLOSED FORM. T1: with no coil the charges equal Ch 2's tanks and resistor step for step, since the flux coordinate is read
    by nothing; the morphism f(phi, v) = v, g(phi, s) = s seen in a run. RESULTS: worst gap 0.0 over 200 steps."""
    a, fa = runner(rlc(1.0, None, 0.25, tau))
    b, fb = runner(rc_tank(1.0, 0.25, tau))
    worst = 0.0
    for _ in range(200):
        a, b = fa(a), fb(b)
        la, lb = flat(a), flat(b)
        worst = max(worst, abs(la[1] - lb[0]), abs(la[3] - lb[1]))
    assert worst < 1e-13


@pytest.mark.parametrize("tau,band_want,end_want", [(0.5, 0.1667, 0.938), (0.1, 0.02631, 1.012), (0.01, 0.002513, 1.001)])
def test_T2_lossless_band(tau, band_want, end_want):
    """DISCRETE ANALOGUE. T2: no resistor, start (2, 0), C = 1, L = 2, time 50: energy in the band of P76 (a3)'s node flux, in one
    round, shrinking with tau, no drift. RESULTS: band 0.1667, 0.02631, 0.002513; end 0.938, 1.012, 1.001."""
    C, L = 1.0, 2.0
    s, f = runner(rlc(C, L, None, tau))
    x = flat(s)
    E = lambda x: x[1] ** 2 / (2 * C) + x[3] ** 2 / (2 * C) + (x[0] - x[2]) ** 2 / (2 * L)
    e0 = E(x)
    band = 0.0
    for _ in range(int(round(50 / tau))):
        s = f(s)
        x = flat(s)
        band = max(band, abs(E(x) / e0 - 1))
    assert band == pytest.approx(band_want, rel=2e-3)
    assert E(x) / e0 == pytest.approx(end_want, abs=1e-3)


@pytest.mark.parametrize("tau,rate_want", [(0.1, 0.2051), (0.01, 0.2013)])
def test_T3_envelope_rate(tau, rate_want):
    """CLOSED FORM, approached. T3: coil and resistor, kappa = 0.2: the difference mode's envelope falls at rate kappa as tau
    shrinks. RESULTS: 0.2051, 0.2013."""
    C, L, kappa = 1.0, 2.0, 0.2
    s, f = runner(rlc(C, L, kappa, tau))
    osc = []
    for _ in range(int(round(30 / tau))):
        s = f(s)
        x = flat(s)
        d = x[1] - x[3]
        osc.append(d ** 2 / 4 + (x[0] - x[2]) ** 2 / (2 * L))
    osc = np.array(osc)
    t = (np.arange(len(osc)) + 1) * tau
    mask = t > 5
    rate = -np.polyfit(t[mask], np.log(osc[mask]), 1)[0] / 2
    assert rate == pytest.approx(rate_want, abs=1e-4)


@pytest.mark.parametrize("tau,band_want,period_want", [(0.5, 0.2734, 4.3468), (0.1, 0.03805, 4.4392), (0.01, 0.003561, 4.4428)])
def test_W1_motor_lossless(tau, band_want, period_want):
    """DISCRETE ANALOGUE. W1: motor at positive responses, C = L = K = J = 1, one round, time 60: period toward 2 pi / sqrt 2 =
    4.4429, energy band shrinking with tau, C v + J w / K constant to rounding. RESULTS: band 0.2734, 0.03805, 0.003561; period
    4.3468, 4.4392, 4.4428; drift 3e-15."""
    C, L, K, J = 1.0, 1.0, 1.0, 1.0
    arr = wire([two_port(C, 1.0, tau), rotor(J, 0.0, tau)], motor_winding(K, L))
    assert arr.label == "(winding(K=1,L=1) . (cap2 || rotor))"
    s, f = runner(arr)
    E = lambda x: x[1] ** 2 / (2 * C) + x[3] ** 2 / (2 * J) + (x[0] - K * x[2]) ** 2 / (2 * L)
    inv = lambda x: x[1] + x[3] / K
    x = flat(s)
    e0, i0 = E(x), inv(x)
    band, dinv, trace = 0.0, 0.0, []
    for _ in range(int(round(60 / tau))):
        s = f(s)
        x = flat(s)
        band = max(band, abs(E(x) / e0 - 1))
        dinv = max(dinv, abs(inv(x) - i0))
        trace.append(x[3])
    assert band == pytest.approx(band_want, rel=2e-3)
    assert period_of(trace, tau) == pytest.approx(period_want, abs=1e-3)
    assert dinv < 1e-14


@pytest.mark.parametrize("K,speed_want,volts_want", [(1.0, 0.81826, 0.818), (2.0, 0.47551, 0.951)])
def test_W2_motor_spins_up(K, speed_want, volts_want):
    """DISCRETE ANALOGUE. W2: a capacitor of 1000 at 1 volt as the battery, drag b = 0.5 on the rotor's speed, tau = 0.01, time 400.
    RESULTS: the speed is the battery's voltage now over K, and the current b w / K: 0.81826 against 0.81816 at K = 1, 0.47551 against
    0.47551 at K = 2. The prediction "as the battery barely falls" was false: drag draws it down to 0.818 at K = 1, so the speed is
    not yet V / K. (The voltage 0.951 at K = 2 is read off this test, not stated in RESULTS.)"""
    C, L, J, b, V, tau = 1000.0, 0.5, 1.0, 0.5, 1.0, 0.01
    s, f = runner(wire([two_port(C, C * V, tau), rotor(J, 0.0, tau)], motor_winding(K, L, b=b)))
    for _ in range(int(round(400 / tau))):
        s = f(s)
    x = flat(s)
    speed, volts, cur = x[3] / J, x[1] / C, (x[0] - K * x[2]) / L
    assert speed == pytest.approx(speed_want, abs=1e-5)
    assert speed == pytest.approx(volts / K, abs=2e-4)
    # RESULTS gives no number for the current. The gap is 4e-4 at K = 1, which is J dw/dt / K as the battery falls (dV/dt = -i / C).
    assert cur == pytest.approx(b * speed / K, abs=5e-4)
    assert volts ==pytest.approx(volts_want, abs=1e-3)


def test_W3_transformer_with_a_resistor():
    """ILLUSTRATE. W3: coupled coils (L = 1, M = 0.1) on two two-port capacitors, a resistor (0.02) on the second's voltage port only,
    tau = 0.01: energy moves across with no wire, then burned in the second circuit. RESULTS: second circuit peaks at 0.375 at
    t = 29.0; total 0.380, 0.266, 0.143 at t = 30, 60, 120."""
    L, M, kappa, tau = 1.0, 0.1, 0.02, 0.01
    Linv = np.linalg.inv(np.array([[L, M], [M, L]]))

    def k(m):
        lam = jnp.array([m[0], m[2]])
        return 0.5 * lam @ jnp.asarray(Linv) @ lam + 0.5 * kappa * m[3] ** 2

    s, f = runner(wire([two_port(1.0, 1.0, tau), two_port(1.0, 0.0, tau)], cost_part(4, k, "coils(2)+R on second")))
    t2, e2, tot = [], [], []
    for i in range(int(round(120 / tau))):
        s = f(s)
        if i % 50 == 0:
            x = flat(s)
            lam = np.array([x[0], x[2]])
            tot.append(0.5 * x[1] ** 2 + 0.5 * x[3] ** 2 + 0.5 * lam @ Linv @ lam)
            e2.append(0.5 * x[3] ** 2 + 0.5 * x[2] ** 2 * Linv[1, 1])
            t2.append((i + 1) * tau)
    e2, tot = np.array(e2), np.array(tot)
    assert e2.max() == pytest.approx(0.375, abs=1e-3)
    assert t2[int(np.argmax(e2))] == pytest.approx(29.0, abs=0.5)
    got = [tot[np.searchsorted(t2, 30)], tot[np.searchsorted(t2, 60)], tot[-1]]
    np.testing.assert_allclose(got, [0.380, 0.266, 0.143], atol=1e-3)


def _series(c, tau):
    """A coil from a very large capacitor (flux 1) to a small capacitor c at the middle node, a resistor from there to ground."""
    R, L = 2.0, 1.0
    top = cost_part(4, lambda m: (m[0] - m[2]) ** 2 / (2 * L) + 0.5 * (1.0 / R) * m[3] ** 2, "coil+R to ground")
    return wire([two_port(1e6, 0.0, tau, phi0=1.0), two_port(c, 0.0, tau)], top)


@pytest.mark.parametrize("c,rate_want", [(0.1, 2.2778), (0.01, 2.0638), (0.001, 2.0060)])
def test_W4_series_needs_a_node_capacitor(c, rate_want):
    """CLOSED FORM, approached. W4: series coil and resistor (L = 1, R = 2) with a small capacitor c at the node, tau = min(R c / 2,
    0.01): the current's decay rate tends to R / L = 2 as c shrinks. RESULTS: 2.2778, 2.0638, 2.0060."""
    R, L = 2.0, 1.0
    tau = min(0.5 * R * c, 0.01)
    s, f = runner(_series(c, tau))
    ts, cur = [], []
    n = int(round(3.0 / tau))
    for i in range(n):
        s = f(s)
        if i % max(1, n // 300) == 0:
            x = flat(s)
            ts.append((i + 1) * tau)
            cur.append((x[0] - x[2]) / L)
    ts, cur = np.array(ts), np.array(cur)
    mask = (ts > 0.5) & (ts < 2.5) & (np.abs(cur) > 1e-12)
    rate = -np.polyfit(ts[mask], np.log(np.abs(cur[mask])), 1)[0]
    assert rate == pytest.approx(rate_want, abs=1e-3)


def test_W4_stiffness_blows_up():
    """ILLUSTRATE. W4's price: c = 0.01, tau = 0.1, over 2 R c = 0.04, blows up. RESULTS: largest value after 100 steps 7e66."""
    s, f = runner(_series(0.01, 0.1))
    for _ in range(100):
        s = f(s)
    largest = np.abs(flat(s)).max()
    assert 1e66 < largest < 1e68


@pytest.mark.parametrize("a1,a2,want", [(0.0, 0.0, 0.375), (0.0, np.pi, -0.375), (np.pi / 2, np.pi / 2, -0.1875),
                                        (np.pi / 2, -np.pi / 2, 0.1875)])
def test_W5_forces_at_rest(a1, a2, want):
    """CLOSED FORM. W5: at rest, r = 2, mu = 1: head to tail attracts with 6 mu / r^4, side by side parallel repels with 3 mu / r^4,
    flipping one magnet reverses each. RESULTS: +0.37500, -0.37500, -0.18750, +0.18750 (one step of the charge coordinate)."""
    mu, r, tau = 1.0, 2.0, 0.01
    arr = wire(magnet(0.0, 0.0, a1, tau) + magnet(r, 0.0, a2, tau), dipole_cost(mu))
    s, f = runner(arr)
    x0, x1 = flat(s), flat(f(s))
    assert (x1[1] - x0[1]) / tau == pytest.approx(want, abs=1e-9)


@pytest.mark.parametrize("tau,band_want", [(0.04, 1.63), (0.02, 0.668), (0.01, 0.321)])
def test_W5_free_pair(tau, band_want):
    """DISCRETE ANALOGUE. W5: a free misaligned pair up to contact (r = 1.5): energy band in proportion to tau, of a small |E0|;
    momentum drift 0, angular momentum drift 2e-16. RESULTS: band 1.63, 0.668, 0.321. Point magnets fall into each other (which is
    false of real magnets, which have size): the measure stops at r = 1.5."""
    mu = 1.0
    s, f = runner(wire(magnet(0.0, 0.0, 0.3, tau) + magnet(4.0, 0.5, -0.4, tau), dipole_cost(mu)))

    def energy(x):
        dx, dy = x[6] - x[0], x[8] - x[2]
        r_ = np.hypot(dx, dy)
        fi = np.arctan2(dy, dx)
        U = mu * (np.cos(x[4] - x[10]) - 3 * np.cos(x[4] - fi) * np.cos(x[10] - fi)) / r_ ** 3
        return U + 0.5 * (x[1] ** 2 + x[3] ** 2 + x[5] ** 2 + x[7] ** 2 + x[9] ** 2 + x[11] ** 2)

    ang = lambda x: x[0] * x[3] - x[2] * x[1] + x[6] * x[9] - x[8] * x[7] + x[5] + x[11]
    x = flat(s)
    e0, a0 = energy(x), ang(x)
    band, dmom, dang, t = 0.0, 0.0, 0.0, 0.0
    while np.hypot(x[6] - x[0], x[8] - x[2]) > 1.5 and t < 200:
        s = f(s)
        x = flat(s)
        t += tau
        band = max(band, abs(energy(x) - e0) / abs(e0))
        dmom = max(dmom, abs(x[1] + x[7]), abs(x[3] + x[9]))
        dang = max(dang, abs(ang(x) - a0))
    assert t < 200
    assert band == pytest.approx(band_want, rel=5e-3)
    assert dmom < 1e-12
    assert dang < 1e-13


def test_floating_capacitor():
    """CLOSED FORM. Book Ch 6 exc.floating_capacitor: a capacitor between two junctions, each a tank, joined by the cost on the difference of
    voltages. One step by hand, the rest state by the series capacitance, the junctions' total charge kept."""
    arr = wire([tank(1.0, 3.0, 1.0), tank(1.0, 0.0, 1.0), tank(1.0, 0.0, 1.0)], floating_coupler(0.25))
    assert "lead" in arr.label and arr.label.count("tank") == 3
    s, f = runner(arr)
    s = f(s)
    np.testing.assert_allclose(flat(s), [2.25, 0.75, 0.75], atol=1e-12)
    for _ in range(400):
        s = f(s)
        x = flat(s)
        assert abs(x[0] + x[1] - 3.0) < 1e-12
    np.testing.assert_allclose(x, [2.0, 1.0, 1.0], atol=1e-10)


def test_capacitor_over_the_mass():
    """CLOSED FORM. With its voltage port unwired, the two-port capacitor on a coil to a wall is the mass C on a spring 1/L, by the morphism of
    integrators f(phi, v) = phi, g = id. So thm.energy_ports' kept energy, s^2/2C + phi phi'/2L, is kept."""
    C, L, tau = 2.0, 3.0, 0.4
    arr = wire([two_port(C, 1.0, tau, phi0=0.5)], cost_part(2, lambda m: m[0] ** 2 / (2 * L), "coil"))
    s, f = runner(arr)
    q, p = 0.5, 1.0
    first = None
    for _ in range(100):
        x_next = q + tau * p / C
        kept = p ** 2 / (2 * C) + q * x_next / (2 * L)
        first = kept if first is None else first
        assert abs(kept - first) < 1e-12
        s = f(s)
        q, p = x_next, p - tau * x_next / L
        np.testing.assert_allclose(flat(s), [q, p], atol=1e-12)


def _transformer(g, tau):
    Nt, ell, R = (1.0, 1.0), (0.9, 0.9), 10.0
    w1 = cost_part(2, lambda m: (m[0] - Nt[0] * m[1]) ** 2 / (2 * ell[0]), "winding")
    w2 = cost_part(2, lambda m: (m[0] - Nt[1] * m[1]) ** 2 / (2 * ell[1]), "winding")
    c = cost_part(1, lambda m: R * m[0] ** 2 / 2, "core")
    parts = tensor_arrangements([two_port(1.0, 1.0, tau), two_port(1.0, 0.0, tau), core_flux(g, tau)])
    # The outputs are (phi1, v1, phi2, v2, Phi); the two voltages are dropped and the core's flux is copied.
    return compose_seq(compose_seq(parts, select(5, [0, 4, 2, 4, 4])), tensor_arrangements([w1, w2, c]))


@pytest.mark.parametrize("g,want", [(1.0, 0.22), (10.0, 0.029), (100.0, 0.0029)])
def test_transformer_with_a_stored_core(g, want):
    """DISCRETE ANALOGUE. Book Ch 6 ex.transformer: two windings, a core and the core's flux stored, on two-port capacitors. As the eddy
    resistance grows, the run approaches the coupled coils of one box; plan P76 (b4)'s gaps."""
    tau, T = 0.001, 70.0
    Nt, ell, R = np.array([1.0, 1.0]), np.array([0.9, 0.9]), 10.0
    Linv = np.linalg.inv(np.diag(ell) + np.outer(Nt, Nt) / R)
    one = wire([two_port(1.0, 1.0, tau), two_port(1.0, 0.0, tau)],
               cost_part(4, lambda m: 0.5 * jnp.array([m[0], m[2]]) @ jnp.asarray(Linv) @ jnp.array([m[0], m[2]]), "coils"))
    arr = _transformer(g, tau)
    assert arr.label.count("winding") == 2 and "coreFlux" in arr.label
    sa, fa = runner(one)
    sb, fb = runner(arr)
    gap = 0.0
    top = 0.0
    for i in range(int(round(T / tau))):
        sa, sb = fa(sa), fb(sb)
        if i % 50 == 0:
            xa, xb = flat(sa), flat(sb)
            gap = max(gap, np.abs(xa[:4] - xb[:4]).max())
            top = max(top, 0.5 * xb[3] ** 2)
    assert gap == pytest.approx(want, rel=0.1)
    assert top > 0.2  # energy reaches the second circuit, with no wire between the two
