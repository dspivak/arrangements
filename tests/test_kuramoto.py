"""Kuramoto = ``Phi`` of an edge-box arrangement with configuration parameters;
the swing equation = ``Phi`` of the same boxes with phase parameters.

Audit of the claim (see ``dap/kuramoto.py``): the Kuramoto model

    theta_i' = omega_i + K * sum_{j ~ i} sin(theta_j - theta_i)

is the dynamics of the closed arrangement whose forcing is ``omega = -dk`` for

    k(theta) = -K * sum_{(i,j) in edges} cos(theta_i - theta_j)
               - sum_i omega_i * theta_i,

built compositionally (oscillator boxes + stateless edge coupling boxes +
routing wire, composed with ``compose_seq``/``tensor_arrangements``), and the
swing equation is the *same lens data* with the ``phase`` parameter and sharp
``xi/m``. Because the integrator is part of the arrangement (def:arr), these
are two arrangements, not two readings of one; ``Phi`` takes no integrator.
Conventions, as in test_heat_equation / test_wave_equation: the framework's
step is unit-time and the time step lives in the sharp -- ``sharp_scale = dt``
for ``configuration`` (forward Euler with step ``dt``), ``sharp_scale = 1/m``
for ``phase`` (semi-implicit Euler, unit step, inertia ``m``; resolution comes
from taking ``m`` large, exactly as the graph-wave test takes ``kappa/m`` small).

Checks:

0. the total forcing and the routing bijection EMERGE from the composition;
1. one-step exactness of the configuration dynamics against the hand-derived
   Euler-Kuramoto update (pins every sign);
2. two oscillators above threshold (K > |omega_0 - omega_1| / 2): phase
   locking at delta* = arcsin(dw / 2K), both frequencies -> mean(omega);
3. same below threshold: the phase difference drifts without bound;
4. the phase arrangement, omega = 0: the phase difference oscillates around 0
   without decay (amplitude preserved, energy bounded), rather than converging;
5. the CDLM trajectories (``regress/kuramoto.npz``) to rtol 1e-10.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.coalgebra import configuration, phase
from arrangements.functors import Phi
from arrangements.kuramoto import kuramoto_arrangement, kuramoto_wire
from arrangements.polynomial import nest_left, unnest_left

from tests.chain import IN_POS, load_regress

_TRIV = lambda _o: (jnp.zeros(0), jnp.zeros(0))


def _stepper(O):
    """One closed-system step of the pc-morphism ``O`` as a jitted state map.

    This calls the framework's own ``run_one`` (nothing is re-derived); jit
    only compiles that very computation so the long runs are affordable.
    """

    def f(s):
        _, _, s2 = O.with_state(s).run_one(IN_POS, _TRIV)
        return s2

    return jax.jit(f)


# A small irregular graph: path 0-1-2-3 plus chord 0-2 (degrees 2, 2, 3, 1).
_N = 4
_EDGES = [(0, 1), (1, 2), (2, 3), (0, 2)]


# The composite's parameter is ``(P_osc_1 (x) ... (x) P_edge_E) (x) P_wire``, so its
# state is ``(nest_left([s_1, ..., s_N, e_1, ..., e_E]), s_wire)`` with the edge
# boxes' and the wiring's states the trivial ``R^0``; these convert to flat angles.


def _conf_state(theta, E):
    theta = jnp.asarray(theta, float)
    N = theta.shape[0]
    parts = [theta[i:i + 1] for i in range(N)] + [jnp.zeros(0)] * E
    return (nest_left(parts), jnp.zeros(0))


def _phase_state(theta, xi, E):
    theta, xi = jnp.asarray(theta, float), jnp.asarray(xi, float)
    N = theta.shape[0]
    parts = [(theta[i:i + 1], xi[i:i + 1]) for i in range(N)] + [jnp.zeros(0)] * E
    return (nest_left(parts), jnp.zeros(0))


def _conf_flat(state, N, E):
    parts = unnest_left(state[0], N + E)
    return jnp.concatenate(parts[:N])


def _phase_flat(state, N, E):
    parts = unnest_left(state[0], N + E)[:N]
    return (jnp.concatenate([q for q, _ in parts]), jnp.concatenate([p for _, p in parts]))


def _rhs(theta, omegas, edges, K):
    """Oracle: the Kuramoto right-hand side omega_i + K sum_{j~i} sin(theta_j - theta_i)."""
    out = np.array(omegas, dtype=float)
    for (i, j) in edges:
        out[i] += K * np.sin(theta[j] - theta[i])
        out[j] += K * np.sin(theta[i] - theta[j])
    return out


def _cost(theta, omegas, edges, K):
    """The cost ``k`` whose forcing is ``omega = -dk`` (DESIGN.md §3)."""
    return -float(np.dot(omegas, theta)) - K * sum(
        np.cos(theta[i] - theta[j]) for (i, j) in edges
    )


# ---------------------------------------------------------------------------
# 0. The composite is what the claim says it is.
# ---------------------------------------------------------------------------


def test_total_forcing_emerges_from_composition():
    """``omega|_Q = -dk``, ``k = -sum_i omega_i theta_i - K sum_e cos(theta_i - theta_j)``,
    from ``compose_seq``'s pullback-and-sum -- not written by hand."""
    K = 0.7
    omegas = [0.9, -0.4, 0.2, 1.1]
    arr = kuramoto_arrangement(omegas, _EDGES, K, sharp_scale=0.05)
    assert (arr.out_dim_M, arr.in_dim_M, arr.out_dim_N, arr.in_dim_N) == (0, 0, 0, 0)
    assert arr.P.Q_dim == _N
    grad_k = jax.grad(lambda t: -jnp.dot(jnp.asarray(omegas), t)
                      - K * sum(jnp.cos(t[i] - t[j]) for (i, j) in _EDGES))
    rng = np.random.default_rng(0)
    for _ in range(5):
        theta = jnp.asarray(rng.standard_normal(_N))
        got = np.asarray(arr.omega(theta, jnp.zeros(0), jnp.zeros(0))[0])
        np.testing.assert_allclose(got, -np.asarray(grad_k(theta)), atol=1e-12)


def test_wire_routing_is_a_bijection():
    """The wire routes the 2E broadcast out-ports bijectively onto the 2E edge in-ports."""
    wire = kuramoto_wire(_N, _EDGES)
    E2 = 2 * len(_EDGES)
    assert (wire.out_dim_M, wire.in_dim_M, wire.out_dim_N, wire.in_dim_N) == (E2, E2, 0, 0)
    m_out = jnp.arange(E2, dtype=float)
    routed = np.asarray(wire.in_f(jnp.zeros(0), m_out, jnp.zeros(0)))
    assert sorted(routed.tolist()) == list(range(E2))


# ---------------------------------------------------------------------------
# 1. One-step exactness of the configuration dynamics (pins the signs).
# ---------------------------------------------------------------------------


def test_configuration_one_step_is_euler_kuramoto():
    """One step == theta_i + dt * (omega_i + K sum_{j~i} sin(theta_j - theta_i))."""
    K, dt = 0.7, 0.07
    omegas = [0.9, -0.4, 0.2, 1.1]
    E = len(_EDGES)
    O = Phi(kuramoto_arrangement(omegas, _EDGES, K, sharp_scale=dt))
    rng = np.random.default_rng(1)
    for _ in range(5):
        theta = rng.standard_normal(_N)
        _, _, new = O.with_state(_conf_state(theta, E)).run_one(IN_POS, _TRIV)
        want = theta + dt * _rhs(theta, omegas, _EDGES, K)
        np.testing.assert_allclose(np.asarray(_conf_flat(new, _N, E)), want, rtol=0, atol=1e-13)


# ---------------------------------------------------------------------------
# 2 & 3. Two oscillators: locking above threshold, drift below.
# ---------------------------------------------------------------------------

_TWO_OMEGAS = (0.8, -0.2)  # dw = 1.0, mean = 0.3; threshold K_c = dw/2 = 0.5


def _run_two(K, dt, steps, theta0=(0.1, 0.0), parameter=configuration):
    O = Phi(kuramoto_arrangement(list(_TWO_OMEGAS), [(0, 1)], K, sharp_scale=dt,
                                 parameter=parameter))
    step = _stepper(O)
    s = _conf_state(theta0, 1)
    traj = np.empty((steps + 1, 2))
    traj[0] = np.asarray(_conf_flat(s, 2, 1))
    for t in range(steps):
        s = step(s)
        traj[t + 1] = np.asarray(_conf_flat(s, 2, 1))
    return traj


def test_two_oscillators_lock_above_threshold():
    """K = 0.8 > 0.5: delta -> arcsin(dw/2K), both frequencies -> mean(omega)."""
    K, dt, steps = 0.8, 0.02, 5000
    traj = _run_two(K, dt, steps)
    delta = traj[:, 0] - traj[:, 1]
    dw = _TWO_OMEGAS[0] - _TWO_OMEGAS[1]

    # The locked difference: delta* = arcsin(dw / 2K) is a fixed point of the
    # discrete map too (the per-step increment of delta is dt*(dw - 2K sin delta)),
    # so convergence is to it exactly, not merely O(dt)-near it.
    delta_star = np.arcsin(dw / (2 * K))
    np.testing.assert_allclose(delta[-1], delta_star, rtol=0, atol=1e-10)
    # ...and it has genuinely settled: constant over the last 500 steps.
    assert np.max(np.abs(delta[-500:] - delta[-1])) < 1e-10

    # Both frequencies converge to the mean frequency (0.3): measured over the
    # last 1000 steps, each oscillator advances at mean(omega) per unit time.
    window = 1000
    freqs = (traj[-1] - traj[-1 - window]) / (window * dt)
    np.testing.assert_allclose(freqs, np.mean(_TWO_OMEGAS), rtol=0, atol=1e-8)


def test_two_oscillators_drift_below_threshold():
    """K = 0.3 < 0.5: the phase difference never settles -- it drifts without bound."""
    K, dt, steps = 0.3, 0.02, 3000
    traj = _run_two(K, dt, steps)
    delta = traj[:, 0] - traj[:, 1]

    # d(delta)/dt = dw - 2K sin(delta) >= 1.0 - 0.6 = 0.4 > 0: monotone drift.
    assert np.all(np.diff(delta) > 0)
    # Far more than a full turn -- delta(T) - delta(0) >= 0.4 * T = 24 > 4 pi.
    assert delta[-1] - delta[0] > 4 * np.pi
    # No locking: the two measured frequencies stay distinct (by ~sqrt(dw^2-4K^2)).
    window = 1000
    freqs = (traj[-1] - traj[-1 - window]) / (window * dt)
    assert freqs[0] - freqs[1] > 0.4


# ---------------------------------------------------------------------------
# 4. The phase arrangement on the same lens data: swing oscillation, no decay.
# ---------------------------------------------------------------------------


def test_phase_oscillates_without_decay():
    """omega = 0, sharp = xi/m: delta oscillates around 0, amplitude and energy preserved."""
    K, m, steps = 1.0, 400.0, 2500
    arr = kuramoto_arrangement([0.0, 0.0], [(0, 1)], K, sharp_scale=1.0 / m, parameter=phase)
    O = Phi(arr)
    step = _stepper(O)

    theta0 = jnp.array([0.5, -0.5])  # delta_0 = 1 rad, at rest
    s = _phase_state(theta0, jnp.zeros(2), 1)
    delta = np.empty(steps + 1)
    energy = np.empty(steps + 1)

    def _energy(state):
        q, xi = _phase_flat(state, 2, 1)
        return float(jnp.sum(xi**2) / (2 * m) - K * jnp.cos(q[0] - q[1]))

    delta[0], energy[0] = float(theta0[0] - theta0[1]), _energy(s)
    for t in range(steps):
        s = step(s)
        q, _ = _phase_flat(s, 2, 1)
        delta[t + 1] = float(q[0] - q[1])
        energy[t + 1] = _energy(s)

    # Pendulum in delta: m ddot(delta) = -2K sin(delta), linear frequency
    # sqrt(2K/m) ~ 0.0707/step, period ~ 95 steps at amplitude 1 -> ~26 periods.
    # (a) It oscillates: many sign changes, symmetric about 0.
    crossings = int(np.sum(np.diff(np.sign(delta)) != 0))
    assert crossings > 20
    assert abs(np.mean(delta)) < 0.05

    # (b) No decay: the peak amplitude over the last 500 steps matches the
    # first 500 (>= 5 periods each) to a few percent -- and in particular has
    # not converged toward the minimum delta = 0 (contrast the configuration
    # arrangement, which would settle there).
    a_first = np.max(np.abs(delta[:500]))
    a_last = np.max(np.abs(delta[-500:]))
    assert 0.95 < a_last / a_first < 1.05
    assert a_last > 0.9  # still swinging out to ~ the initial 1 rad

    # (c) Energy-conserving-ish: semi-implicit Euler keeps the energy in a
    # bounded band around E_0 (relative to the well depth E_0 - U_min), with
    # no systematic trend.
    U_min = -K  # the cost at delta = 0 (the omega terms vanish)
    band = np.max(np.abs(energy - energy[0])) / (energy[0] - U_min)
    assert band < 0.05


# ---------------------------------------------------------------------------
# 5. Regression against the CDLM trajectories.
# ---------------------------------------------------------------------------


def test_regression_matches_cdlm():
    """The CDLM trajectories (``regress/kuramoto.npz``) to rtol 1e-10 (DESIGN.md §3;
    the system is closed, so no covector crosses a port and only states are compared)."""
    R = load_regress("kuramoto")

    # one configuration step on the 4-oscillator graph
    K, dt = R["step_params"]
    omegas = list(R["step_omegas"])
    edges = [tuple(int(x) for x in e) for e in R["step_edges"]]
    E = len(edges)
    O = Phi(kuramoto_arrangement(omegas, edges, K, sharp_scale=dt))
    for theta, want in zip(R["step_in"], R["step_out"]):
        _, _, new = O.with_state(_conf_state(theta, E)).run_one(IN_POS, _TRIV)
        np.testing.assert_allclose(
            np.asarray(_conf_flat(new, len(omegas), E)), want, rtol=1e-10, atol=1e-12)

    # two oscillators, locking and drifting
    two = list(R["two_omegas"])
    for tag in ("lock", "drift"):
        Kc, dtc, steps = R[f"{tag}_params"]
        Oc = Phi(kuramoto_arrangement(two, [(0, 1)], Kc, sharp_scale=dtc))
        step = _stepper(Oc)
        s = _conf_state((0.1, 0.0), 1)
        traj = np.empty((int(steps) + 1, 2))
        traj[0] = np.asarray(_conf_flat(s, 2, 1))
        for t in range(int(steps)):
            s = step(s)
            traj[t + 1] = np.asarray(_conf_flat(s, 2, 1))
        np.testing.assert_allclose(traj, R[f"{tag}_traj"], rtol=1e-10, atol=1e-12)

    # the swing equation (phase parameter)
    Kp, m, steps = R["swing_params"]
    Op = Phi(kuramoto_arrangement([0.0, 0.0], [(0, 1)], Kp, sharp_scale=1.0 / m,
                                  parameter=phase))
    step = _stepper(Op)
    s = _phase_state(jnp.array([0.5, -0.5]), jnp.zeros(2), 1)
    qs = np.empty((int(steps) + 1, 2))
    ps = np.empty((int(steps) + 1, 2))
    qs[0], ps[0] = (np.asarray(x) for x in _phase_flat(s, 2, 1))
    for t in range(int(steps)):
        s = step(s)
        qs[t + 1], ps[t + 1] = (np.asarray(x) for x in _phase_flat(s, 2, 1))
    np.testing.assert_allclose(qs, R["swing_q"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(ps, R["swing_p"], rtol=1e-10, atol=1e-12)
