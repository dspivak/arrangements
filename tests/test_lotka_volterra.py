"""P32 (b): Lotka--Volterra as the book's step (CDLM ``ex.lotka_volterra``;
``notes/P32-B-PREDICTIONS-2026-09-29.md``, predictions fixed before code).

The arrangement is ``compose_seq( prey (x) predator, hunt )`` from
``arrangements/lotka_volterra.py``: log-population states, opposite-signed sharps, a
stateless hunt box with cost ``beta x y``. Parameters here: ``alpha=1.1, beta=0.7,
gamma=0.9, delta=0.5``, all distinct and non-round.

The four predictions, quoted from the note, and what each test is evidence of
(STATUS.md §2):

1. "One ``Phi`` step is Euler in log coordinates": ``u' = u + dt(alpha - beta e^v)``,
   ``v' = v + dt(delta e^u - gamma)`` to 1e-12, and the composite forcing is
   ``-grad U_total`` to 1e-12 with ``U_total`` written only here.
   -- ``test_one_step_is_euler_in_log_coordinates``, ``test_composite_forcing_is_minus_d_total_cost``:
   CLOSED FORM.
2. "Coexistence is fixed": ``(log(gamma/delta), log(alpha/beta))`` moves by less than 1e-14.
   -- ``test_coexistence_is_a_fixed_point``: CLOSED FORM (follows from 1).
3. "The plain step spirals OUTWARD, and the chart does not save it": ``H`` rises every step;
   per-cycle growth halves when ``dt`` halves (ratio in [1.7, 2.3]); Euler in population
   coordinates grows at a comparable rate. "A two-stage step keeps ``H`` bounded": at least
   10x smaller excursion over 20 cycles; scaling exponent in [0.8, 1.2] raw, [1.7, 2.3]
   synchronized.
   -- ``test_plain_step_raises_H_every_step_and_growth_per_cycle_halves``: DISCRETE ANALOGUE
      of the ODE's conservation of ``H`` (the continuous result is exact conservation;
      the discrete step is checked on its own terms).
   -- ``test_population_chart_euler_also_spirals_outward``: ILLUSTRATION (a numpy Euler in
      ``(x, y)``, not an arrangement; it answers the audit's "or Euler in log coordinates").
   -- ``test_two_stage_step_keeps_H_bounded``: DISCRETE ANALOGUE (symplectic Euler /
      Stoermer--Verlet bounded-energy behaviour, checked numerically, not proved).
4. "Composed equals monolithic": step for step over 200 steps to 1e-10.
   -- ``test_composed_equals_monolithic``: TWO IMPLEMENTATIONS AGREEING (the composite's
      forcing from pullback-and-sum, the control's written by hand).

State shapes: the composite's state is ``((u, v), ())``; ``_nest``/``_flat`` convert to and
from the flat ``(u, v)`` (the pattern of ``tests/test_dc_motor.py``).
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import exact
from arrangements.functors import Phi
from arrangements.interpretation import trivial_omega
from arrangements.lotka_volterra import (
    invariant,
    lotka_volterra,
    lotka_volterra_monolithic,
)
from arrangements.pc2 import Phi2

_IN_POS = (jnp.zeros(0), trivial_omega(0))
_TRIV = lambda _o: (jnp.zeros(0), jnp.zeros(0))

ALPHA, BETA, GAMMA, DELTA = 1.1, 0.7, 0.9, 0.5
_P = (ALPHA, BETA, GAMMA, DELTA)

# Start at twice the coexistence prey population, coexistence predator population.
U0 = float(np.log(2.0 * GAMMA / DELTA))
V0 = float(np.log(ALPHA / BETA))
Z0 = np.array([U0, V0])


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _nest(z):
    z = jnp.asarray(z, float)
    return ((z[:1], z[1:]), jnp.zeros(0))


def _flat(s):
    (a, b), _ = s
    return jnp.concatenate([a, b])


def _step_fn(arr, functor=Phi):
    """The framework's own closed-system step (``Phi`` or ``Phi2``) as a flat state map."""
    O = functor(arr)

    def f(z):
        _, _, s2 = O.with_state(_nest(z)).run_one(_IN_POS, _TRIV)
        return _flat(s2)

    return f


def _mono_step_fn(arr):
    O = Phi(arr)

    def f(z):
        _, _, s2 = O.with_state(jnp.asarray(z, float)).run_one(_IN_POS, _TRIV)
        return s2

    return f


def _run(step_fn, z0, steps):
    step = jax.jit(step_fn)
    z = jnp.asarray(z0, float)
    out = np.empty((steps + 1, 2))
    out[0] = np.asarray(z)
    for i in range(1, steps + 1):
        z = step(z)
        out[i] = np.asarray(z)
    return out


def _euler_log(z, dt):
    """CDLM's field in the log chart, one forward-Euler step (the closed form)."""
    u, v = z[0], z[1]
    return jnp.array([u + dt * (ALPHA - BETA * jnp.exp(v)), v + dt * (DELTA * jnp.exp(u) - GAMMA)])


def _upward_crossings(traj):
    """Indices where ``u`` crosses the coexistence value upward: one per cycle."""
    u_star = np.log(GAMMA / DELTA)
    u = traj[:, 0]
    return np.nonzero((u[:-1] < u_star) & (u[1:] >= u_star))[0] + 1


def _growth_per_cycle(H, crossings, cycles):
    """Mean ``H`` increment between consecutive upward crossings over ``cycles`` cycles."""
    assert len(crossings) > cycles, f"only {len(crossings)} crossings"
    Hc = H[crossings[: cycles + 1]]
    return float(np.mean(np.diff(Hc)))


H = invariant(*_P)
_H = lambda traj: np.asarray(jax.vmap(H)(jnp.asarray(traj)))


# ---------------------------------------------------------------------------
# 1. One Phi step is Euler in the log chart; the forcing is -dU_total.
# ---------------------------------------------------------------------------


def test_one_step_is_euler_in_log_coordinates():
    """Prediction 1: from 5 random states the composite's step equals the closed form."""
    dt = 0.013
    step = _step_fn(lotka_volterra(*_P, dt=dt))
    rng = np.random.default_rng(1)
    for _ in range(5):
        z = jnp.asarray(rng.uniform(-1.0, 1.5, size=2))
        np.testing.assert_allclose(np.asarray(step(z)), np.asarray(_euler_log(z, dt)), rtol=0, atol=1e-12)


def test_composite_forcing_is_minus_d_total_cost():
    """Prediction 1, second half: ``compose_seq``'s pullback-and-sum equals ``-dU_total``,
    ``U_total = beta e^{u+v} - alpha e^u - (gamma beta/delta) e^v``, written only here."""
    arr = lotka_volterra(*_P, dt=0.013)
    U_total = lambda q: BETA * jnp.exp(q[0] + q[1]) - ALPHA * jnp.exp(q[0]) - (GAMMA * BETA / DELTA) * jnp.exp(q[1])
    want = exact(lambda q, m_out, n_in: -U_total(q))
    rng = np.random.default_rng(2)
    for _ in range(5):
        z = jnp.asarray(rng.uniform(-1.0, 1.5, size=2))
        got = arr.omega(z, jnp.zeros(0), jnp.zeros(0))[0]
        np.testing.assert_allclose(np.asarray(got), np.asarray(want(z, jnp.zeros(0), jnp.zeros(0))[0]), rtol=0, atol=1e-12)
        # And the covector is not the species' own alone: the hunt's pullback is in it.
        own = jnp.array([ALPHA * jnp.exp(z[0]), (GAMMA * BETA / DELTA) * jnp.exp(z[1])])
        assert float(jnp.max(jnp.abs(got - own))) > 1e-3


# ---------------------------------------------------------------------------
# 2. Coexistence is a fixed point.
# ---------------------------------------------------------------------------


def test_coexistence_is_a_fixed_point():
    """Prediction 2: ``(x, y) = (gamma/delta, alpha/beta)`` does not move (follows from 1)."""
    z_star = jnp.array([np.log(GAMMA / DELTA), np.log(ALPHA / BETA)])
    for dt in (0.013, 0.2):
        step = _step_fn(lotka_volterra(*_P, dt=dt))
        assert float(jnp.max(jnp.abs(step(z_star) - z_star))) < 1e-14


# ---------------------------------------------------------------------------
# 3. The plain step spirals outward; the two-stage step keeps H bounded.
# ---------------------------------------------------------------------------


def test_plain_step_raises_H_every_step_and_growth_per_cycle_halves():
    """Prediction 3, first half: ``H`` increases at EVERY step (``Hess(H)`` is positive
    definite), and the growth per cycle is ``O(dt)``: ratio at ``dt = 0.02 / 0.01`` in [1.7, 2.3]."""
    growth = {}
    for dt in (0.02, 0.01):
        steps = int(round(3.6 * 7.0 / dt))  # a little over 3 cycles (period ~ 6.9 at this amplitude)
        traj = _run(_step_fn(lotka_volterra(*_P, dt=dt)), Z0, steps)
        Hs = _H(traj)
        assert np.all(np.diff(Hs) > 0.0), f"H fell at some step, dt={dt}"
        growth[dt] = _growth_per_cycle(Hs, _upward_crossings(traj), 3)
        print(f"plain step dt={dt}: H growth per cycle = {growth[dt]:.4e} (H0 = {Hs[0]:.4f})")
    ratio = growth[0.02] / growth[0.01]
    print(f"plain step: growth ratio dt=0.02 / dt=0.01 = {ratio:.3f}")
    assert 1.7 <= ratio <= 2.3


def test_population_chart_euler_also_spirals_outward():
    """Prediction 3, the chart's irrelevance (ILLUSTRATION, numpy only): forward Euler in
    ``(x, y)`` raises ``H`` at a comparable per-cycle rate to the log-chart step."""
    dt = 0.02
    steps = int(round(3.6 * 7.0 / dt))
    xy = np.empty((steps + 1, 2))
    xy[0] = np.exp(Z0)
    for i in range(1, steps + 1):
        x, y = xy[i - 1]
        xy[i] = [x + dt * x * (ALPHA - BETA * y), y + dt * y * (DELTA * x - GAMMA)]
    traj_pop = np.log(xy)
    H_pop = _H(traj_pop)
    g_pop = _growth_per_cycle(H_pop, _upward_crossings(traj_pop), 3)
    traj_log = _run(_step_fn(lotka_volterra(*_P, dt=dt)), Z0, steps)
    g_log = _growth_per_cycle(_H(traj_log), _upward_crossings(traj_log), 3)
    print(f"dt={dt}: per-cycle H growth, population chart {g_pop:.4e}, log chart {g_log:.4e}")
    assert g_pop > 0.0 and g_log > 0.0
    assert 1.0 / 3.0 <= g_pop / g_log <= 3.0


def test_two_stage_step_keeps_H_bounded():
    """Prediction 3, second half. The SAME composed arrangement with the two-round parameter
    (predator ``goes_first``, prey ``goes_second``) run through ``Phi2``: one step is symplectic
    Euler (closed form, checked first); over 20 cycles ``max |H_n - H_0|`` is at least 10x below
    the plain step's growth at ``dt = 0.02``; the excursion scales as ``dt^p`` with ``p`` in
    [0.8, 1.2] raw and [1.7, 2.3] at the synchronized ``u - (dt/2)(alpha - beta e^v)``."""
    # (i) closed form of one two-stage step
    dt = 0.013
    step2 = _step_fn(lotka_volterra(*_P, dt=dt, two_stage=True), functor=Phi2)
    rng = np.random.default_rng(3)
    for _ in range(5):
        z = jnp.asarray(rng.uniform(-1.0, 1.5, size=2))
        v1 = z[1] + dt * (DELTA * jnp.exp(z[0]) - GAMMA)
        u1 = z[0] + dt * (ALPHA - BETA * jnp.exp(v1))
        np.testing.assert_allclose(np.asarray(step2(z)), np.asarray(jnp.array([u1, v1])), rtol=0, atol=1e-12)

    # (ii) boundedness over 20 cycles, raw and synchronized
    cycles = 20
    exc_raw, exc_sync, plain_growth = {}, {}, {}
    for dt in (0.02, 0.01):
        steps = int(round((cycles + 0.6) * 7.0 / dt))
        t0 = time.time()
        traj = _run(_step_fn(lotka_volterra(*_P, dt=dt, two_stage=True), functor=Phi2), Z0, steps)
        n_cycles = len(_upward_crossings(traj))
        Hraw = _H(traj)
        sync = traj.copy()
        sync[:, 0] = traj[:, 0] - 0.5 * dt * (ALPHA - BETA * np.exp(traj[:, 1]))
        Hsync = _H(sync)
        exc_raw[dt] = float(np.max(np.abs(Hraw - Hraw[0])))
        exc_sync[dt] = float(np.max(np.abs(Hsync - Hsync[0])))
        plain = _run(_step_fn(lotka_volterra(*_P, dt=dt)), Z0, steps)
        Hp = _H(plain)
        plain_growth[dt] = float(Hp[-1] - Hp[0])
        print(
            f"two-stage dt={dt}: {n_cycles} cycles in {steps} steps ({time.time() - t0:.1f}s); "
            f"max|H-H0| raw {exc_raw[dt]:.3e}, synchronized {exc_sync[dt]:.3e}; "
            f"plain step H_N - H_0 = {plain_growth[dt]:.3e}"
        )
        assert n_cycles >= cycles
    assert exc_raw[0.02] < 0.1 * plain_growth[0.02]
    p_raw = np.log2(exc_raw[0.02] / exc_raw[0.01])
    p_sync = np.log2(exc_sync[0.02] / exc_sync[0.01])
    print(f"two-stage scaling exponents: raw p = {p_raw:.3f}, synchronized p = {p_sync:.3f}")
    assert 0.8 <= p_raw <= 1.2
    assert 1.7 <= p_sync <= 2.3


# ---------------------------------------------------------------------------
# 4. Composed equals monolithic.
# ---------------------------------------------------------------------------


def test_composed_equals_monolithic():
    """Prediction 4: the composite (forcing from pullback-and-sum) and the one-box control
    (``U_total`` by hand, block-diagonal sharp) agree step for step over 200 steps."""
    dt = 0.02
    a = _run(_step_fn(lotka_volterra(*_P, dt=dt)), Z0, 200)
    b = _run(_mono_step_fn(lotka_volterra_monolithic(*_P, dt=dt)), Z0, 200)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-10)
