"""A charge and the field moving each other, and a source on a 3D grid (book plan §5.9 item 90; run notes ``notes/runs/light-12b2681e/`` in the book, D0, D1 and
S1 of its RESULTS.md, at its settings; S2, S3 and the plain loop at large N are in the run scripts, too slow for this suite).

D1 and S1 run composed arrangements through ``PhiK``, the labels ``(field_and_contact . (... edge ... || charge))`` and ``(lattice3d . (clock ...))``; energy is read off the
states. D0 evaluates the contact box's own form at a point and runs nothing. Every test says which of three things it does (the convention of test_rendezvous.py):
reproduce a CLOSED FORM, check a DISCRETE ANALOGUE, or merely ILLUSTRATE. The tolerances bracket the numbers RESULTS.md reports.

Claude Opus 5.5, session 5c9967d2, 2026-11-05.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arrangements.interpretation import trivial_omega
from arrangements.light_lattice import edge_ax, edge_ay
from arrangements.light_sources import (charge_energy, charge_on_lattice, charge_read, edge_3d, face_table, faces_cost_load, field_and_contact, loop_step, source_lattice_3d,
                                        whitney)
from arrangements.pcK import PhiK

jax.config.update("jax_enable_x64", True)

_IN = (jnp.zeros(0), trivial_omega(0))
_TRIV = lambda _o: (jnp.zeros(0), jnp.zeros(0))


def stepper(arr):
    O = PhiK(arr)
    return O.state, jax.jit(lambda s: O.with_state(s).run_one(_IN, _TRIV)[1])


def test_D0_push_is_the_derivative_of_the_coupling():
    """CLOSED FORM. D0: the contact box's push, at E = 0, against the derivative of the coupling e v . A_phys(x), A_phys by Whitney's shapes, at 50 random points inside
    cells, N = 12, e = 1. RESULTS: worst gap 8.9e-16. RESULTS's script compared a copy of the box's formula; here it is the box's own form that is evaluated."""
    N, e_ch = 12, 1.0
    NE = 2 * N * N
    rng = np.random.default_rng(3)
    A = jnp.asarray(rng.normal(size=NE))
    table = face_table(N)
    box = field_and_contact(N, e_ch)

    def a_phys(x):
        i, j, fx, fy = whitney(N, x)
        ax = (1 - fy) * A[edge_ax(N, i, j)] + fy * A[edge_ax(N, i, j + 1)]
        ay = (1 - fx) * A[edge_ay(N, i, j)] + fx * A[edge_ay(N, i + 1, j)]
        return -jnp.array([ax, ay])

    worst = 0.0
    for _ in range(50):
        x = jnp.asarray(rng.uniform(0.1, N - 0.1, size=2))
        v = jnp.asarray(rng.normal(size=2))
        J = jax.jacfwd(a_phys)(x)
        lorentz = e_ch * (J.T @ v - J @ v)
        m = jnp.zeros(2 * NE + 4).at[0:2 * NE:2].set(A).at[2 * NE:2 * NE + 2].set(x).at[2 * NE + 2:].set(v)
        _, load, _ = box.omega(jnp.zeros(0), m, jnp.zeros(0))
        worst = max(worst, float(jnp.abs(lorentz - load[2 * NE:2 * NE + 2]).max()))
        i, j, _, _ = whitney(N, x)
        _, B = faces_cost_load(table, A)
        Bz = -B[(i % N) * N + (j % N)]
        assert float(jnp.abs(load[2 * NE:2 * NE + 2] - e_ch * jnp.array([v[1] * Bz, -v[0] * Bz])).max()) < 1e-15
    assert worst < 2e-15


def test_label():
    """ILLUSTRATE. The composite is what RESULTS.md says it is, read off its label."""
    label = charge_on_lattice(12, (3.3, 6.4), (0.3, 0.1), 0.1).label
    assert label.startswith("(field_and_contact . ") and "edge" in label and "charge" in label


@pytest.mark.parametrize("e_ch,v0,tau,band_want", [
    (1.0, (0.3, 0.1), 0.2, 4.8e-4), (1.0, (0.3, 0.1), 0.1, 1.7e-4), (1.0, (0.3, 0.1), 0.05, 4.5e-5), (1.0, (0.3, 0.1), 0.025, 1.1e-5),
    (0.2, (0.5, 0.2), 0.2, 3.3e-4), (0.2, (0.5, 0.2), 0.1, 1.1e-4), (0.2, (0.5, 0.2), 0.05, 2.7e-5), (0.2, (0.5, 0.2), 0.025, 6.8e-6)])
def test_D1_energy_band(e_ch, v0, tau, band_want):
    """DISCRETE ANALOGUE. D1: N = 12, field at rest, charge from (3.3, 6.4), time 20, Runge--Kutta on every part: total energy, field and kinetic, in a band falling as
    tau^2 from tau = 0.1 on. RESULTS: 4.8e-4, 1.7e-4, 4.5e-5, 1.1e-5 at e = 1; 3.3e-4, 1.1e-4, 2.7e-5, 6.8e-6 at e = 0.2, which crosses 13 cells."""
    N = 12
    s, f = stepper(charge_on_lattice(N, (3.3, 6.4), v0, tau, e_ch=e_ch))
    A, E, x, p = charge_read(N, s)
    e0 = charge_energy(N, A, E, p)
    band = 0.0
    for _ in range(int(round(20.0 / tau))):
        s = f(s)
        A, E, x, p = charge_read(N, s)
        band = max(band, abs(charge_energy(N, A, E, p) / e0 - 1))
    assert band == pytest.approx(band_want, rel=0.05)


def test_S1_loop_equals_arrangement():
    """CLOSED FORM. S1: the plain loop equals the arrangement to rounding at N = 4 over 40 steps, tau = 0.25, the clock's current sin(2 pi t / 3) on one z-edge, from a random
    start. RESULTS: worst gap 9.3e-15. This is what lets a loop stand beside the arrangement at large N."""
    N, tau, w = 4, 0.25, 2 * np.pi / 3.0
    rng = np.random.default_rng(1)
    A0 = rng.normal(size=3 * N ** 3)
    E0 = rng.normal(size=3 * N ** 3)
    src = edge_3d(N, 2, 1, 2, 3)
    arr = source_lattice_3d(N, A0, E0, src, tau, w)
    assert arr.label.startswith("(lattice3d . ") and "clock" in arr.label
    s, f = stepper(arr)
    A = [A0[i * N ** 3:(i + 1) * N ** 3].reshape(N, N, N) for i in range(3)]
    E = [E0[i * N ** 3:(i + 1) * N ** 3].reshape(N, N, N) for i in range(3)]

    def J(E, t, tau_):
        E[2][1, 2, 3] += tau_ * np.sin(w * t)

    t, worst = 0.0, 0.0
    for _ in range(40):
        s = f(s)
        A, E, t = loop_step(A, E, t, tau, J)
        lv = [np.ravel(np.asarray(x, float)) for x in jax.tree_util.tree_leaves(s) if np.asarray(x).size]
        flat = np.concatenate(lv)[1: 1 + 2 * 3 * N ** 3].reshape(-1, 2)
        worst = max(worst, np.abs(flat[:, 0] - np.concatenate([a.ravel() for a in A])).max(),
                    np.abs(flat[:, 1] - np.concatenate([e.ravel() for e in E])).max())
    assert worst < 2e-14
