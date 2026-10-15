"""Continuous Hopfield network = ``Phi`` of a neuron/synapse-box arrangement.

Audit of the claim (see ``hopfield.py``): the continuous Hopfield network
(Hopfield 1984, ``R_i = C_i = 1``)

    u_i' = -u_i + sum_j W_ij g(u_j) + I_i,        g = tanh,  W symmetric,

is the dynamics (sec.dynamics_functor) of the closed arrangement whose
parameters are configuration coalgebras (sec.matter) and whose total
forcing is ``-dU`` (sec.forcing) for the cost

    U(u) = sum_i [u_i g(u_i) - log cosh(u_i) - I_i g(u_i)]
           - sum_{i<j} W_ij g(u_i) g(u_j),

built compositionally (neuron boxes with state-dependent sharp
``dt / g'(u) = dt cosh(u)^2`` + stateless synapse boxes + routing wire,
composed with ``compose_seq``/``parallel_arrangements``). Conventions as in
test_kuramoto: the framework's step is unit-time and ``dt`` lives in the
sharp.

Checks:

0. the total forcing and the routing bijection EMERGE from the composition
   (the forcing equals ``-d`` of the hand formula at random points);
1. one-step exactness of ``Phi`` against the hand Euler-Hopfield update
   ``u + dt * (-u + W g(u) + I)`` (pins every sign, including the exact
   cancellation of the chain-rule ``g'`` against the state-dependent sharp);
2. the arrangement's emergent cost in u-coordinates equals Hopfield's
   energy ``E(V) = -(1/2) sum W_ij V_i V_j + sum_i int_0^{V_i} g^{-1}(s) ds
   - sum_i I_i V_i`` at ``V = g(u)`` EXACTLY (additive constant zero), via
   ``int_0^V arctanh = V arctanh(V) + (1/2) log(1 - V^2) = u g(u) - log cosh u``;
3. energy monotonicity along the framework trajectory: strictly decreasing
   while the gradient is appreciable, never increasing, and converging;
4. associative memory: Hebbian storage ``W = (gain/N) sum_p xi_p xi_p^T``
   (zero diagonal, gain 2 -- at gain 1 the tanh network has NO retrieval
   attractors, see ``hebbian_weights``), 3 patterns in N = 40 neurons,
   17.5% corruption, sign(V) recovery measured honestly over 10 trials per
   seed for several seeds.

CDLM read the emergent *potential* off the arrangement (``arr.U``); the book's
arrangement carries the forcing, so checks 0 and 2 read the forcing
``arr.omega`` and compare ``-omega|_Q`` with ``dU`` of the hand formula
(STATUS.md §1). The arrangement is closed, so the states themselves are
unchanged by the sign translation.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.functors import Phi
from arrangements.hopfield import hebbian_weights, hopfield_arrangement, hopfield_wire
from arrangements.interpretation import trivial_omega

from tests.chain import flat_state, load_regress, state_like

_IN_POS = (jnp.zeros(0), trivial_omega(0))
_TRIV = lambda _o: (jnp.zeros(0), jnp.zeros(0))


def _stepper(O):
    """One closed-system step of the pc-morphism ``O`` as a jitted state map.

    Calls the framework's own ``run_one`` (nothing is re-derived); jit only
    compiles that very computation so the long retrieval runs are affordable.
    """

    def f(s):
        _, _, s2 = O.with_state(s).run_one(_IN_POS, _TRIV)
        return s2

    return jax.jit(f)


def _run(arr, u, steps):
    """``steps`` closed steps of ``Phi(arr)`` from the flat state ``u``; flat result."""
    O = Phi(arr)
    step = _stepper(O)
    s = state_like(O.state, jnp.asarray(u))
    traj = [flat_state(s)]
    for _ in range(steps):
        s = step(s)
        traj.append(flat_state(s))
    return np.stack([np.asarray(x) for x in traj])


def _random_symmetric(rng, N):
    """A random symmetric weight matrix with zero diagonal."""
    A = rng.standard_normal((N, N))
    W = (A + A.T) / 2.0
    np.fill_diagonal(W, 0.0)
    return W


def _hand_potential(W, I, u):
    """Oracle: the total cost in u-coordinates, assembled by hand."""
    V = np.tanh(u)
    leak = np.sum(u * V - np.log(np.cosh(u)))
    return leak - I @ V - 0.5 * V @ W @ V  # zero diagonal: -(1/2)sum_{ij} = -sum_{i<j}


def _hand_potential_j(W, I, u):
    """``_hand_potential`` in ``jnp``, so it can be differentiated (the forcing is ``-dU``)."""
    V = jnp.tanh(u)
    leak = jnp.sum(u * V - jnp.log(jnp.cosh(u)))
    return leak - jnp.asarray(I) @ V - 0.5 * V @ jnp.asarray(W) @ V


def _hopfield_energy(W, I, V):
    """Oracle: Hopfield's energy in V-coordinates (Hopfield 1984, eq. (7)).

    ``E(V) = -(1/2) sum_{ij} W_ij V_i V_j + sum_i int_0^{V_i} arctanh(s) ds
    - sum_i I_i V_i`` with the integral in closed form
    ``V arctanh(V) + (1/2) log(1 - V^2)``.
    """
    integral = np.sum(V * np.arctanh(V) + 0.5 * np.log1p(-(V**2)))
    return -0.5 * V @ W @ V + integral - I @ V


def _hopfield_energy_j(W, I, V):
    """``_hopfield_energy`` in ``jnp``, so it can be differentiated."""
    integral = jnp.sum(V * jnp.arctanh(V) + 0.5 * jnp.log1p(-(V ** 2)))
    return -0.5 * V @ jnp.asarray(W) @ V + integral - jnp.asarray(I) @ V


def _forcing_Q(arr, u):
    """The closed arrangement's forcing restricted to its parameter base, ``omega|_Q(u)``."""
    return arr.omega(jnp.asarray(u), jnp.zeros(0), jnp.zeros(0))[0]


# ---------------------------------------------------------------------------
# 0. The composite is what the claim says it is.
# ---------------------------------------------------------------------------


def test_total_forcing_emerges_from_composition():
    """omega|_Q = -dU with U = sum_i leak_i - sum_i I_i g(u_i) - sum_{i<j} W_ij g(u_i) g(u_j)."""
    N, dt = 5, 0.05
    rng = np.random.default_rng(3)
    W = _random_symmetric(rng, N)
    W[0, 3] = W[3, 0] = 0.0  # some absent synapses, so edge selection is exercised
    W[1, 4] = W[4, 1] = 0.0
    I = rng.standard_normal(N)
    arr = hopfield_arrangement(W, I, dt)
    assert (arr.out_dim_M, arr.in_dim_M, arr.out_dim_N, arr.in_dim_N) == (0, 0, 0, 0)
    assert arr.P.Q_dim == N
    grad = jax.grad(lambda u: _hand_potential_j(W, I, u))
    for _ in range(5):
        u = jnp.asarray(rng.standard_normal(N) * 1.5)
        np.testing.assert_allclose(
            np.asarray(_forcing_Q(arr, u)), -np.asarray(grad(u)), rtol=0, atol=1e-12
        )


def test_wire_routing_is_a_bijection():
    """The wire routes the 2E broadcast out-ports bijectively onto the 2E synapse in-ports."""
    edges = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)]
    wire = hopfield_wire(4, edges)
    E2 = 2 * len(edges)
    assert (wire.out_dim_M, wire.in_dim_M, wire.out_dim_N, wire.in_dim_N) == (E2, E2, 0, 0)
    m_out = jnp.arange(E2, dtype=float)
    routed = np.asarray(wire.in_f(jnp.zeros(0), m_out, jnp.zeros(0)))
    assert sorted(routed.tolist()) == list(range(E2))


def test_state_dependent_sharp_is_positive():
    """The neuron sharp ``dt * cosh(u)^2`` is positive at every state (g' > 0).

    The sharp is now inside the neurons' configuration coalgebras (def.arr,
    sec.matter), so it is read off the step: ``step(u, xi) - u = sharp_u(xi)``.
    """
    N, dt = 3, 0.1
    rng = np.random.default_rng(4)
    arr = hopfield_arrangement(_random_symmetric(rng, N), np.zeros(N), dt)
    P = arr.P
    for u in [np.zeros(N), np.array([3.0, -2.0, 0.5]), rng.standard_normal(N) * 4]:
        s = state_like(P.init, jnp.asarray(u))
        cols = [flat_state(P.step(s, jnp.eye(N)[i])) - jnp.asarray(u) for i in range(N)]
        S = np.stack([np.asarray(c) for c in cols], axis=1)
        # block-diagonal (here 1x1 blocks), entries dt / g'(u_i) = dt cosh^2(u_i)
        np.testing.assert_allclose(S, np.diag(dt * np.cosh(u) ** 2), rtol=1e-12, atol=1e-14)
        assert np.all(np.diag(S) > 0)


# ---------------------------------------------------------------------------
# 1. One-step exactness of Phi (pins the signs; the keystone).
# ---------------------------------------------------------------------------


def test_phi_one_step_is_euler_hopfield():
    """One ``Phi`` step == u + dt * (-u + W tanh(u) + I).

    The chain-rule factor ``g'(u_i)`` in the pulled-back covector must cancel
    the ``1/g'(u_i)`` in the state-dependent sharp exactly.
    """
    N, dt = 4, 0.07
    rng = np.random.default_rng(1)
    W = _random_symmetric(rng, N)
    I = rng.standard_normal(N)
    arr = hopfield_arrangement(W, I, dt)
    for _ in range(5):
        u = rng.standard_normal(N) * 1.5
        new = _run(arr, u, 1)[1]
        want = u + dt * (-u + W @ np.tanh(u) + I)
        # atol 5e-13: pure roundoff -- the cancellation dt*cosh^2 * g' = dt is
        # exact in exact arithmetic, and cosh^2(u) ~ 20 at u ~ 2 amplifies the
        # ~1e-16 relative float error (measured worst error 1.1e-13).
        np.testing.assert_allclose(new, want, rtol=0, atol=5e-13)


# ---------------------------------------------------------------------------
# 2. The emergent cost IS Hopfield's energy function.
# ---------------------------------------------------------------------------


def test_arrangement_forcing_is_minus_d_hopfield_energy():
    """omega|_Q(u) == -d/du E_Hopfield(tanh(u)) exactly.

    The precise relation verified: the arrangement's forcing is ``-dU`` for a
    cost in u-coordinates; substituting ``V = g(u)`` into Hopfield's
    V-coordinate energy gives the SAME function, because
    ``int_0^{tanh(u)} arctanh(s) ds = u tanh(u) - log cosh(u)`` is the neuron's
    leak cost on the nose (so the two costs agree with additive constant zero,
    and hence so do their differentials).
    """
    N, dt = 6, 0.05
    rng = np.random.default_rng(7)
    W = _random_symmetric(rng, N)
    I = rng.standard_normal(N)
    arr = hopfield_arrangement(W, I, dt)
    for _ in range(8):
        u = rng.standard_normal(N) * 2.0
        # the costs agree pointwise (additive constant zero) ...
        np.testing.assert_allclose(
            _hand_potential(W, I, u), _hopfield_energy(W, I, np.tanh(u)), rtol=0, atol=1e-12
        )
        # ... and the arrangement's forcing is minus the differential of that cost.
        E_grad = jax.grad(
            lambda uu: _hopfield_energy_j(W, I, jnp.tanh(uu))
        )(jnp.asarray(u))
        np.testing.assert_allclose(
            np.asarray(_forcing_Q(arr, u)), -np.asarray(E_grad), rtol=0, atol=1e-12
        )


# ---------------------------------------------------------------------------
# 3. Energy monotonicity along the framework trajectory.
# ---------------------------------------------------------------------------


def test_energy_decreases_along_trajectory():
    """E strictly decreases while the gradient is appreciable; never increases; converges."""
    N, dt, steps = 8, 0.05, 1500
    rng = np.random.default_rng(11)
    W = _random_symmetric(rng, N)
    I = 0.3 * rng.standard_normal(N)
    arr = hopfield_arrangement(W, I, dt)

    u0 = rng.standard_normal(N) * 1.5
    traj = _run(arr, u0, steps)
    E = np.array([_hand_potential(W, I, u) for u in traj])
    # residual |du/dt| = |-u + W g(u) + I|
    G = np.array([np.linalg.norm(-u + W @ np.tanh(u) + I) for u in traj])

    dE = np.diff(E)
    # (a) never increases (up to roundoff on O(1) energies);
    assert np.all(dE < 1e-12)
    # (b) strictly decreases at every step where the flow is appreciable;
    assert np.all(dE[G[:-1] > 1e-6] < 0)
    # (c) genuinely descended, and converged to a critical point.
    assert E[-1] < E[0] - 0.1
    assert G[-1] < 1e-10


# ---------------------------------------------------------------------------
# 4. Associative memory retrieval (measured honestly).
# ---------------------------------------------------------------------------

# 3 patterns in 40 neurons (load 0.075), Hebbian with gain 2 (gain 1 provably
# retrieves nothing -- the overlap equation m = tanh(m (N-1)/N) has only m = 0;
# see hebbian_weights), 7/40 = 17.5% corruption, 600 steps of dt = 0.1 (T = 60).
_N, _P, _GAIN, _DT, _STEPS, _FLIPS = 40, 3, 2.0, 0.1, 600, 7


def _retrieval_run(seed):
    """10 trials for one seed: fresh patterns, corrupted init, count exact sign recovery.

    Success = sign(tanh(u_final)) equals the stored pattern in EVERY coordinate
    (convergence to the mirror pattern -xi or a mixture state counts as failure),
    and the trajectory has actually converged (tiny fixed-point residual).
    """
    rng = np.random.default_rng(seed)
    patterns = rng.choice([-1.0, 1.0], size=(_P, _N))
    W = hebbian_weights(patterns, gain=_GAIN)
    arr = hopfield_arrangement(W, np.zeros(_N), _DT)
    O = Phi(arr)
    step = _stepper(O)

    successes = 0
    for _ in range(10):
        xi = patterns[rng.integers(_P)].copy()
        u0 = xi.copy()
        u0[rng.choice(_N, size=_FLIPS, replace=False)] *= -1  # 17.5% corruption
        s = state_like(O.state, jnp.asarray(u0))
        for _ in range(_STEPS):
            s = step(s)
        u_fin = np.asarray(flat_state(s))
        assert np.linalg.norm(-u_fin + W @ np.tanh(u_fin)) < 1e-8  # converged
        if np.all(np.sign(np.tanh(u_fin)) == xi):
            successes += 1
    return successes


def test_associative_memory_retrieval():
    """>= 8/10 exact recoveries per seed, over 3 seeds; actual counts reported."""
    counts = {seed: _retrieval_run(seed) for seed in (0, 1, 2)}
    print(f"retrieval successes per seed (out of 10): {counts}")
    for seed, c in counts.items():
        assert c >= 8, f"seed {seed}: only {c}/10 exact recoveries"


# ---------------------------------------------------------------------------
# Regression against CDLM.
# ---------------------------------------------------------------------------


def test_regression_matches_cdlm():
    """The CDLM trajectories (``regress/hopfield.npz``) are reproduced to rtol 1e-10.

    The arrangement is closed (no port covector crosses an interface), so the
    inputs need no sign translation and the states are identical (STATUS.md §1).
    """
    R = load_regress("hopfield")

    # one-step exactness scenario
    arr = hopfield_arrangement(R["step_W"], R["step_I"], float(R["step_dt"]))
    new = np.stack([_run(arr, u, 1)[1] for u in R["step_u"]])
    np.testing.assert_allclose(new, R["step_new"], rtol=1e-10, atol=1e-12)

    # 1500-step energy trajectory
    arr = hopfield_arrangement(R["traj_W"], R["traj_I"], float(R["traj_dt"]))
    traj = _run(arr, R["traj_u"][0], R["traj_u"].shape[0] - 1)
    np.testing.assert_allclose(traj, R["traj_u"], rtol=1e-10, atol=1e-12)

    # associative-memory retrieval (seed 0): 10 corrupted inits, 600 steps each
    W = hebbian_weights(R["mem_patterns"], gain=float(R["mem_gain"]))
    arr = hopfield_arrangement(W, np.zeros(W.shape[0]), float(R["mem_dt"]))
    steps = int(R["mem_steps"])
    final = np.stack([_run(arr, u0, steps)[-1] for u0 in R["mem_u0"]])
    np.testing.assert_allclose(final, R["mem_final"], rtol=1e-10, atol=1e-12)
